"""Point-in-time forecasting for the rolling replay.

At each decision timestamp ``as_of`` the forecaster produces a
:class:`ForecastVintage` for the remaining Settlement Periods using **only**
records the PIT store exposes at ``as_of``:

* **Baseline** per target SP, in order of preference:
  1. the published day-ahead wholesale forecast for the target day (kept with
     its own provenance so external vs internal forecasts stay comparable);
  2. the same-SP price observed on the most recent visible prior day (lag);
  3. the same-SP median over visible prior days;
  4. the global median of visible observations.
* **Intraday correction**: an exponentially-weighted mean of
  (observed − baseline) over today's already-published SPs, decayed with
  distance ahead — recent surprises inform the near horizon more than the far.
* **Uncertainty**: same-SP dispersion across visible prior days, widened with
  lead time; q10/q50/q90 via a normal approximation.

This is a transparent, chronologically-safe model — not a state-of-the-art
price forecaster. Provenance of every output row is ``model_forecast``; the
basis used for each row is recorded so the UI can display it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from gb_battery.replay.pit import PITDataStore, _utc
from gb_battery.replay.records import ForecastRecord, ObservationRecord, Provenance
from gb_battery.settlement import SettlementPeriod

Z10 = 1.2815515655446004  # standard-normal 90th percentile

SIGMA_FLOOR = 3.0  # GBP/MWh — never claim tighter certainty than this
SIGMA_CAP = 80.0
BIAS_HALF_LIFE_SP = 4.0  # EWMA half-life over today's observed SPs
BIAS_DECAY_AHEAD = 0.9  # per-SP decay of the intraday correction into the future
SIGMA_WIDEN_PER_SP = 0.02  # relative widening per SP of lead time (capped 2x)


@dataclass
class VintageRow:
    """One target Settlement Period of one forecast vintage."""

    settlement_date: date
    settlement_period: int
    start_utc: datetime
    point: float
    q10: float
    q50: float
    q90: float
    sigma: float
    basis: str  # day_ahead_forecast | lag_same_sp | median_same_sp | global_median
    intraday_bias: float
    provenance: Provenance = Provenance.MODEL_FORECAST
    # Fundamentals passed through from published forecasts (display only).
    demand_forecast_mw: float | None = None
    wind_forecast_mw: float | None = None
    solar_forecast_mw: float | None = None
    fundamentals_provenance: Provenance | None = None


@dataclass
class ForecastVintage:
    """All forecasts issued at one decision timestamp."""

    issued_at: datetime
    information_cutoff: datetime
    rows: list[VintageRow]
    basis_max_published_at: datetime | None  # newest input record used (audit)
    n_input_observations: int
    n_input_forecasts: int
    warnings: list[str] = field(default_factory=list)

    def row_for(self, sp: int) -> VintageRow | None:
        for r in self.rows:
            if r.settlement_period == sp:
                return r
        return None


class PITForecaster:
    """Stateless point-in-time forecaster over a :class:`PITDataStore`."""

    def __init__(
        self,
        *,
        sigma_floor: float = SIGMA_FLOOR,
        bias_half_life_sp: float = BIAS_HALF_LIFE_SP,
        bias_decay_ahead: float = BIAS_DECAY_AHEAD,
    ) -> None:
        self.sigma_floor = sigma_floor
        self.bias_half_life_sp = bias_half_life_sp
        self.bias_decay_ahead = bias_decay_ahead

    def forecast(
        self,
        store: PITDataStore,
        as_of: datetime,
        target_periods: list[SettlementPeriod],
    ) -> ForecastVintage:
        if not target_periods:
            raise ValueError("target_periods must be non-empty")
        as_of = _utc(as_of)
        day = target_periods[0].settlement_date
        warnings: list[str] = []

        # --- PIT-visible inputs (single gateway; audited below) -------------
        obs = store.observations_at(as_of, "wholesale_price")
        da_price = store.forecasts_at(as_of, "wholesale_price", day=day)
        da_demand = store.forecasts_at(as_of, "demand_mw", day=day)
        da_wind = store.forecasts_at(as_of, "wind_mw", day=day)
        da_solar = store.forecasts_at(as_of, "solar_mw", day=day)
        used_records: list[ObservationRecord | ForecastRecord] = [
            *obs, *da_price.values(), *da_demand.values(), *da_wind.values(), *da_solar.values()
        ]
        store.check_no_leakage(used_records, as_of)  # hard guarantee

        prior_obs = [o for o in obs if o.settlement_date < day]
        today_obs = [o for o in obs if o.settlement_date == day]
        if not prior_obs and not da_price:
            warnings.append("No prior price history visible at as_of; using flat fallback.")

        # Same-SP history across prior days.
        by_sp: dict[int, list[float]] = {}
        latest_by_sp: dict[int, tuple[date, float]] = {}
        for o in prior_obs:
            by_sp.setdefault(o.settlement_period, []).append(o.value)
            cur = latest_by_sp.get(o.settlement_period)
            if cur is None or o.settlement_date > cur[0]:
                latest_by_sp[o.settlement_period] = (o.settlement_date, o.value)
        global_median = float(np.median([o.value for o in prior_obs])) if prior_obs else 50.0
        global_sigma = float(np.std([o.value for o in prior_obs])) if len(prior_obs) > 2 else 15.0

        def baseline(sp: int) -> tuple[float, str]:
            if sp in da_price:
                return da_price[sp].value, "day_ahead_forecast"
            if sp in latest_by_sp:
                return latest_by_sp[sp][1], "lag_same_sp"
            if sp in by_sp:
                return float(np.median(by_sp[sp])), "median_same_sp"
            return global_median, "global_median"

        def sigma_for(sp: int) -> float:
            vals = by_sp.get(sp)
            s = float(np.std(vals)) if vals and len(vals) > 2 else global_sigma
            return float(min(max(s, self.sigma_floor), SIGMA_CAP))

        # --- Intraday bias from today's already-published SPs ----------------
        # EWMA of (observed - baseline), most recent SP weighted highest.
        bias = 0.0
        if today_obs:
            today_sorted = sorted(today_obs, key=lambda o: o.settlement_period)
            lam = 0.5 ** (1.0 / self.bias_half_life_sp)
            num = den = 0.0
            w = 1.0
            for o in reversed(today_sorted):
                b, _ = baseline(o.settlement_period)
                num += w * (o.value - b)
                den += w
                w *= lam
            bias = num / den if den > 0 else 0.0

        # --- Assemble rows ----------------------------------------------------
        rows: list[VintageRow] = []
        for k, per in enumerate(target_periods):
            b, basis = baseline(per.settlement_period)
            decayed_bias = bias * (self.bias_decay_ahead ** k)
            point = b + decayed_bias
            widen = min(1.0 + SIGMA_WIDEN_PER_SP * k, 2.0)
            sigma = sigma_for(per.settlement_period) * widen
            dem = da_demand.get(per.settlement_period)
            wnd = da_wind.get(per.settlement_period)
            sol = da_solar.get(per.settlement_period)
            fund_prov = dem.provenance if dem else (wnd.provenance if wnd else None)
            rows.append(
                VintageRow(
                    settlement_date=per.settlement_date,
                    settlement_period=per.settlement_period,
                    start_utc=per.start_utc,
                    point=round(point, 3),
                    q10=round(point - Z10 * sigma, 3),
                    q50=round(point, 3),
                    q90=round(point + Z10 * sigma, 3),
                    sigma=round(sigma, 3),
                    basis=basis,
                    intraday_bias=round(decayed_bias, 3),
                    demand_forecast_mw=dem.value if dem else None,
                    wind_forecast_mw=wnd.value if wnd else None,
                    solar_forecast_mw=sol.value if sol else None,
                    fundamentals_provenance=fund_prov,
                )
            )

        return ForecastVintage(
            issued_at=as_of,
            information_cutoff=as_of,
            rows=rows,
            basis_max_published_at=store.max_published_at(used_records),
            n_input_observations=len(obs),
            n_input_forecasts=len(da_price) + len(da_demand) + len(da_wind) + len(da_solar),
            warnings=warnings,
        )

    def vintage_records(self, vintage: ForecastVintage) -> list[ForecastRecord]:
        """Materialise a vintage as :class:`ForecastRecord` rows (for storage/export)."""
        out: list[ForecastRecord] = []
        for r in vintage.rows:
            for q, val in [(None, r.point), (0.1, r.q10), (0.5, r.q50), (0.9, r.q90)]:
                out.append(
                    ForecastRecord(
                        variable="wholesale_price",
                        settlement_date=r.settlement_date,
                        settlement_period=r.settlement_period,
                        start_utc=r.start_utc,
                        value=val,
                        issued_at=vintage.issued_at,
                        published_at=vintage.issued_at,
                        source="pit_forecaster",
                        provenance=Provenance.MODEL_FORECAST,
                        unit="GBP/MWh",
                        quantile=q,
                    )
                )
        return out
