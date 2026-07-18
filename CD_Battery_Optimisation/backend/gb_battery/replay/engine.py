"""Rolling-horizon replay engine.

For each Settlement Period of the chosen day, in chronological order:

1. ``as_of`` = the period's start (the decision gate);
2. build the information set from the PIT store (``published_at <= as_of``);
3. forecast every remaining period of the day;
4. optimise the remaining horizon (wholesale-only revenue streams);
5. **execute only the first period** of the proposed schedule;
6. settle it against the realised outturn once that is available;
7. carry state of charge, cycle budget and cumulative P&L forward.

The full proposed schedule and the forecast vintage are stored at every step so
the UI can show how the plan and the forecasts evolved through the day.

Execution assumption (documented, deliberate): committed energy is always
executable at the MID reference price. No bid/ask spread, liquidity, partial
fills or market impact are modelled.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.deterministic import optimise
from gb_battery.optimiser.inputs import OptimisationInputs, PeriodInput, RevenueStreams
from gb_battery.replay.forecaster import ForecastVintage, PITForecaster
from gb_battery.replay.pit import (
    PITDataStore,
    _utc,
    store_from_elexon,
    store_from_sample,
    store_from_synthetic,
)
from gb_battery.replay.records import Provenance
from gb_battery.settlement import SettlementPeriod, settlement_periods_for_day

WHOLESALE_ONLY = RevenueStreams(
    wholesale=True,
    upward_availability=False,
    downward_availability=False,
    bm_activation=False,
    imbalance=False,
)

EXECUTION_ASSUMPTION = (
    "Executed at the MID reference price: no bid/ask spread, liquidity, "
    "partial fills or market impact are modelled."
)


class ReplayOptions(BaseModel):
    """Configuration of one replay session."""

    source: Literal["sample", "synthetic", "elexon"] = "sample"
    strategy: Literal["rolling_forecast", "rolling_threshold"] = "rolling_forecast"
    history_days: int = Field(default=14, ge=3, le=60)
    mid_lag_minutes: int = Field(default=10, ge=0, le=120)
    seed: int = 42
    live: bool = False  # cap information & settlement at wall-clock now


class ProposedPeriod(BaseModel):
    """One period of a proposed (not executed) schedule."""

    settlement_period: int
    start_utc: datetime
    action: str
    charge_mw: float
    discharge_mw: float
    ending_soc_mwh: float
    forecast_price: float
    expected_pnl_gbp: float


class DecisionRecord(BaseModel):
    """The audited outcome of one replay step."""

    step: int
    settlement_date: date
    settlement_period: int
    start_utc: datetime
    end_utc: datetime
    duration_hours: float

    # Information set
    as_of: datetime
    information_cutoff: datetime
    basis_max_published_at: datetime | None
    n_input_observations: int
    n_input_forecasts: int

    # Forecast for the executed period
    forecast_price: float
    forecast_q10: float
    forecast_q90: float
    forecast_sigma: float
    forecast_basis: str
    forecast_provenance: str

    # Decision
    action: str
    charge_mw: float
    discharge_mw: float
    soc_before_mwh: float
    soc_after_mwh: float
    expected_immediate_pnl_gbp: float
    expected_horizon_pnl_gbp: float
    explanation: str
    binding_constraints: list[str]
    proposed_schedule: list[ProposedPeriod]

    # Settlement
    settlement_status: Literal["settled", "pending"]
    actual_price: float | None = None
    actual_price_available_at: datetime | None = None
    actual_price_provenance: str | None = None
    realised_pnl_gbp: float | None = None
    degradation_cost_gbp: float = 0.0
    forecast_error: float | None = None  # forecast - actual

    warnings: list[str] = Field(default_factory=list)


def inputs_from_vintage(
    vintage: ForecastVintage,
    periods: list[SettlementPeriod],
    *,
    streams: RevenueStreams = WHOLESALE_ONLY,
) -> OptimisationInputs:
    """Build optimiser inputs from a forecast vintage (forecast provenance only)."""
    rows: list[PeriodInput] = []
    for per in periods:
        row = vintage.row_for(per.settlement_period)
        if row is None:
            continue
        rows.append(
            PeriodInput(
                settlement_date=per.settlement_date,
                settlement_period=per.settlement_period,
                start_utc=per.start_utc,
                end_utc=per.end_utc,
                duration_hours=per.duration_hours,
                wholesale_price=row.point,
                wholesale_price_sigma=row.sigma,
                demand_forecast_mw=row.demand_forecast_mw,
                wind_forecast_mw=row.wind_forecast_mw,
                solar_forecast_mw=row.solar_forecast_mw,
            )
        )
    return OptimisationInputs(periods=rows, revenue_streams=streams)


def _threshold_first_action(
    config: BatteryConfig, inputs: OptimisationInputs, soc: float
) -> tuple[float, float]:
    """First-period action of the transparent threshold rule (30/70 percentiles)."""
    from gb_battery.backtest.strategies import threshold_rule

    sched = threshold_rule(config, inputs, soc)
    first = sched[0]
    return first.charge_mw, first.discharge_mw


class ReplayEngine:
    """Chronological replay of one settlement date for one battery."""

    def __init__(
        self,
        config: BatteryConfig,
        day: date,
        options: ReplayOptions | None = None,
        store: PITDataStore | None = None,
        forecaster: PITForecaster | None = None,
        periods: list[SettlementPeriod] | None = None,
    ) -> None:
        self.options = options or ReplayOptions()
        self.day = day
        self.config = config
        self.forecaster = forecaster or PITForecaster()
        self.store = store if store is not None else self._build_store()
        # `periods` override exists for small hand-verifiable tests.
        self.periods = periods if periods is not None else settlement_periods_for_day(self.day)
        self.soc = float(
            min(max(config.initial_soc_mwh, config.effective_min_soc), config.effective_max_soc)
        )
        self.discharged_mwh = 0.0  # cumulative, for the rolling cycle budget
        self.decisions: list[DecisionRecord] = []
        self.vintages: list[ForecastVintage] = []
        self.warnings: list[str] = list(self.store.notes)

    # ------------------------------------------------------------------ store
    def _build_store(self) -> PITDataStore:
        opt = self.options
        if opt.source == "synthetic":
            return store_from_synthetic(
                self.day,
                history_days=opt.history_days,
                mid_lag_minutes=opt.mid_lag_minutes,
                seed=opt.seed,
            )
        if opt.source == "sample":
            store, day = store_from_sample(self.day, mid_lag_minutes=opt.mid_lag_minutes)
            self.day = day
            return store
        return store_from_elexon(
            self.day,
            history_days=opt.history_days,
            mid_lag_minutes=opt.mid_lag_minutes,
        )

    # ------------------------------------------------------------------ state
    @property
    def step_index(self) -> int:
        return len(self.decisions)

    def is_complete(self, now: datetime | None = None) -> bool:
        if self.step_index >= len(self.periods):
            return True
        if self.options.live:
            per = self.periods[self.step_index]
            now = _utc(now) if now is not None else _utc(datetime.now(tz=UTC))
            # A live step only runs once its period has completed (else the
            # action would be un-settleable and the loop would fabricate the
            # future). The forward view is served by propose() instead.
            return per.end_utc > now
        return False

    # ------------------------------------------------------------------- step
    def step(self, now: datetime | None = None) -> DecisionRecord | None:
        """Execute the next Settlement Period; return its decision record."""
        if self.options.live and now is None:
            now = datetime.now(tz=UTC)  # live settlement must never see past "now"
        if self.is_complete(now):
            return None
        t = self.step_index
        per = self.periods[t]
        as_of = per.start_utc  # decision gate = period start
        remaining = self.periods[t:]

        vintage = self.forecaster.forecast(self.store, as_of, remaining)
        inputs = inputs_from_vintage(vintage, remaining)
        cfg = self._step_config(remaining)

        step_warnings = list(vintage.warnings)
        if self.options.strategy == "rolling_threshold":
            charge_mw, discharge_mw = _threshold_first_action(cfg, inputs, self.soc)
            proposed: list[ProposedPeriod] = []
            expected_horizon = 0.0
            explanation = (
                "Threshold rule: charge below the 30th, discharge above the 70th "
                "percentile of the forecast horizon prices."
            )
            binding: list[str] = []
        else:
            result = optimise(cfg, inputs, compute_marginals=False)
            if result.status != "optimal" or not result.periods:
                step_warnings.append(f"Optimiser status '{result.status}'; holding (IDLE).")
                charge_mw = discharge_mw = 0.0
                proposed = []
                expected_horizon = 0.0
                explanation = "Optimisation failed for this horizon; no action taken."
                binding = []
            else:
                first = result.periods[0]
                charge_mw, discharge_mw = first.charge_mw, first.discharge_mw
                explanation = first.explanation
                binding = first.binding_constraints
                expected_horizon = result.total_expected_pnl_gbp
                proposed = [
                    ProposedPeriod(
                        settlement_period=p.settlement_period,
                        start_utc=p.start_utc,
                        action=p.action.value if hasattr(p.action, "value") else str(p.action),
                        charge_mw=p.charge_mw,
                        discharge_mw=p.discharge_mw,
                        ending_soc_mwh=p.ending_soc_mwh,
                        forecast_price=p.wholesale_price,
                        expected_pnl_gbp=p.total_expected_pnl_gbp,
                    )
                    for p in result.periods
                ]

        # --- Execute the first action only, physically clamped ---------------
        dt = per.duration_hours
        charge_mw, discharge_mw, soc_after = self._apply_physics(charge_mw, discharge_mw, dt)
        action = "CHARGE" if charge_mw > 1e-4 else ("DISCHARGE" if discharge_mw > 1e-4 else "IDLE")

        row = vintage.rows[0]
        deg = self.config.degradation_cost_gbp_per_mwh_throughput * (charge_mw + discharge_mw) * dt
        expected_immediate = row.point * (discharge_mw - charge_mw) * dt - deg

        # --- Settle against the outturn (only once it exists / is published) --
        outturn = self.store.outturn("wholesale_price", per.settlement_date, per.settlement_period)
        settle_now = outturn is not None
        if outturn is not None and self.options.live and now is not None:
            settle_now = _utc(outturn.published_at) <= _utc(now)
        if settle_now and outturn is not None:
            realised = outturn.value * (discharge_mw - charge_mw) * dt - deg
            record_settlement = {
                "settlement_status": "settled",
                "actual_price": outturn.value,
                "actual_price_available_at": outturn.published_at,
                "actual_price_provenance": outturn.provenance.value,
                "realised_pnl_gbp": round(realised, 4),
                "forecast_error": round(row.point - outturn.value, 4),
            }
        else:
            record_settlement = {"settlement_status": "pending"}

        record = DecisionRecord(
            step=t,
            settlement_date=per.settlement_date,
            settlement_period=per.settlement_period,
            start_utc=per.start_utc,
            end_utc=per.end_utc,
            duration_hours=dt,
            as_of=as_of,
            information_cutoff=vintage.information_cutoff,
            basis_max_published_at=vintage.basis_max_published_at,
            n_input_observations=vintage.n_input_observations,
            n_input_forecasts=vintage.n_input_forecasts,
            forecast_price=row.point,
            forecast_q10=row.q10,
            forecast_q90=row.q90,
            forecast_sigma=row.sigma,
            forecast_basis=row.basis,
            forecast_provenance=row.provenance.value,
            action=action,
            charge_mw=round(charge_mw, 4),
            discharge_mw=round(discharge_mw, 4),
            soc_before_mwh=round(self.soc, 4),
            soc_after_mwh=round(soc_after, 4),
            expected_immediate_pnl_gbp=round(expected_immediate, 4),
            expected_horizon_pnl_gbp=round(expected_horizon, 2),
            explanation=explanation,
            binding_constraints=binding,
            proposed_schedule=proposed,
            degradation_cost_gbp=round(deg, 4),
            warnings=step_warnings,
            **record_settlement,
        )

        self.soc = soc_after
        self.discharged_mwh += discharge_mw * dt
        self.decisions.append(record)
        self.vintages.append(vintage)
        return record

    def run(self, max_steps: int | None = None, now: datetime | None = None) -> list[DecisionRecord]:
        """Step until the day (or the live boundary) is exhausted."""
        out: list[DecisionRecord] = []
        while max_steps is None or len(out) < max_steps:
            rec = self.step(now)
            if rec is None:
                break
            out.append(rec)
        return out

    # ------------------------------------------------------------- live view
    def propose(self, as_of: datetime) -> tuple[ForecastVintage, list[ProposedPeriod]] | None:
        """Forecast + optimise the not-yet-executed remainder of the day at ``as_of``.

        Nothing is executed or settled — this is the forward-looking paper view.
        """
        t = self.step_index
        if t >= len(self.periods):
            return None
        remaining = self.periods[t:]
        vintage = self.forecaster.forecast(self.store, _utc(as_of), remaining)
        inputs = inputs_from_vintage(vintage, remaining)
        cfg = self._step_config(remaining)
        result = optimise(cfg, inputs, compute_marginals=False)
        proposed = [
            ProposedPeriod(
                settlement_period=p.settlement_period,
                start_utc=p.start_utc,
                action=p.action.value if hasattr(p.action, "value") else str(p.action),
                charge_mw=p.charge_mw,
                discharge_mw=p.discharge_mw,
                ending_soc_mwh=p.ending_soc_mwh,
                forecast_price=p.wholesale_price,
                expected_pnl_gbp=p.total_expected_pnl_gbp,
            )
            for p in result.periods
        ]
        return vintage, proposed

    # ---------------------------------------------------------------- helpers
    def _step_config(self, remaining: list[SettlementPeriod]) -> BatteryConfig:
        """Per-step config: current SoC and the *remaining* cycle budget."""
        updates: dict = {"initial_soc_mwh": self.soc}
        if self.config.maximum_cycles_per_day is not None:
            total_budget = self.config.maximum_cycles_per_day * self.config.energy_capacity_mwh
            remaining_budget = max(total_budget - self.discharged_mwh, 0.0)
            frac_day = max(sum(p.duration_hours for p in remaining) / 24.0, 1e-9)
            # build_model caps discharge at cycles * capacity * frac_day; invert.
            cycles = remaining_budget / (self.config.energy_capacity_mwh * frac_day)
            updates["maximum_cycles_per_day"] = max(cycles, 1e-6)
        return self.config.model_copy(update=updates)

    def _apply_physics(
        self, charge_mw: float, discharge_mw: float, dt: float
    ) -> tuple[float, float, float]:
        """Clamp an action to physical feasibility and evolve SoC."""
        cfg = self.config
        charge_mw = min(max(charge_mw, 0.0), cfg.maximum_charge_mw)
        discharge_mw = min(max(discharge_mw, 0.0), cfg.maximum_discharge_mw)
        if charge_mw > 0 and discharge_mw > 0:  # never simultaneous
            if charge_mw >= discharge_mw:
                charge_mw, discharge_mw = charge_mw - discharge_mw, 0.0
            else:
                charge_mw, discharge_mw = 0.0, discharge_mw - charge_mw
        add = cfg.charge_efficiency * charge_mw * dt
        room = cfg.effective_max_soc - self.soc
        if add > room:
            charge_mw = room / max(cfg.charge_efficiency * dt, 1e-9)
            add = room
        remove = discharge_mw * dt / cfg.discharge_efficiency
        avail = self.soc - cfg.effective_min_soc
        if remove > avail:
            discharge_mw = avail * cfg.discharge_efficiency / max(dt, 1e-9)
            remove = avail
        return charge_mw, discharge_mw, self.soc + add - remove

    # -------------------------------------------------------------- summaries
    def realised_summary(self) -> dict:
        settled = [d for d in self.decisions if d.settlement_status == "settled"]
        pending = [d for d in self.decisions if d.settlement_status == "pending"]
        realised = sum(d.realised_pnl_gbp or 0.0 for d in settled)
        revenue = sum(
            (d.actual_price or 0.0) * d.discharge_mw * d.duration_hours for d in settled
        )
        cost = sum((d.actual_price or 0.0) * d.charge_mw * d.duration_hours for d in settled)
        deg = sum(d.degradation_cost_gbp for d in self.decisions)
        errors = [d.forecast_error for d in settled if d.forecast_error is not None]
        cum, peak, max_dd = 0.0, 0.0, 0.0
        for d in settled:
            cum += d.realised_pnl_gbp or 0.0
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
        return {
            "n_steps": len(self.decisions),
            "n_settled": len(settled),
            "n_pending": len(pending),
            "realised_pnl_gbp": round(realised, 2),
            "wholesale_revenue_gbp": round(revenue, 2),
            "charging_cost_gbp": round(cost, 2),
            "degradation_cost_gbp": round(deg, 2),
            "cycles": round(self.discharged_mwh / self.config.energy_capacity_mwh, 3)
            if self.config.energy_capacity_mwh
            else 0.0,
            "max_drawdown_gbp": round(max_dd, 2),
            "ending_soc_mwh": round(self.soc, 3),
            "forecast_mae": round(
                sum(abs(e) for e in errors) / len(errors), 3
            )
            if errors
            else None,
            "forecast_rmse": round((sum(e * e for e in errors) / len(errors)) ** 0.5, 3)
            if errors
            else None,
            "forecast_bias": round(sum(errors) / len(errors), 3) if errors else None,
            "execution_assumption": EXECUTION_ASSUMPTION,
        }


def perfect_foresight_day(
    config: BatteryConfig, store: PITDataStore, day: date
) -> tuple[OptimisationInputs, list[float]] | None:
    """Optimiser inputs built from the realised price path (benchmark only).

    Returns ``None`` when any period of the day lacks an outturn (e.g. live).
    """
    periods = settlement_periods_for_day(day)
    rows: list[PeriodInput] = []
    prices: list[float] = []
    for per in periods:
        obs = store.outturn("wholesale_price", day, per.settlement_period)
        if obs is None:
            return None
        prices.append(obs.value)
        rows.append(
            PeriodInput(
                settlement_date=day,
                settlement_period=per.settlement_period,
                start_utc=per.start_utc,
                end_utc=per.end_utc,
                duration_hours=per.duration_hours,
                wholesale_price=obs.value,
                wholesale_price_kind="observed",
            )
        )
    return (
        OptimisationInputs(periods=rows, revenue_streams=WHOLESALE_ONLY),
        prices,
    )


__all__ = [
    "EXECUTION_ASSUMPTION",
    "DecisionRecord",
    "ProposedPeriod",
    "ReplayEngine",
    "ReplayOptions",
    "inputs_from_vintage",
    "perfect_foresight_day",
    "Provenance",
]
