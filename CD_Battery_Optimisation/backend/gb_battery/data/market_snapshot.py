"""Assemble a resilient per-period market snapshot for a settlement date.

Each source (MID, system prices, demand forecast, wind/solar forecast, generation
mix) is fetched independently. If a source fails or the app is offline, that series
degrades gracefully to a cached value or a clearly-labelled synthetic estimate — the
whole snapshot never fails because of one bad source. Per-series provenance and the
last successful update time are recorded for the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from gb_battery.data.cache import ParquetCache
from gb_battery.data.elexon import ElexonClient
from gb_battery.data.settings import DataSettings, get_settings
from gb_battery.demo.scenarios import _fundamentals, diurnal_price_shape
from gb_battery.lineage import DataKind
from gb_battery.optimiser.inputs import OptimisationInputs, PeriodInput, RevenueStreams
from gb_battery.settlement import UTC, settlement_periods_for_day


@dataclass
class SourceStatus:
    source: str
    ok: bool
    kind: DataKind
    detail: str = ""
    retrieved_at: datetime | None = None


@dataclass
class MarketSnapshot:
    day: date
    frame: pd.DataFrame  # per-period wide frame
    statuses: list[SourceStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_optimisation_inputs(
        self,
        *,
        upward_availability_price: float = 6.0,
        downward_availability_price: float = 4.0,
        revenue_streams: RevenueStreams | None = None,
        price_sigma_frac: float = 0.15,
    ) -> OptimisationInputs:
        """Convert the snapshot into optimiser inputs."""
        periods = {p.settlement_period: p for p in settlement_periods_for_day(self.day)}
        rows: list[PeriodInput] = []
        wholesale_kind = _kind_for(self.statuses, "wholesale")
        for _, r in self.frame.sort_values("settlement_period").iterrows():
            sp = int(r["settlement_period"])
            per = periods.get(sp)
            if per is None:
                continue
            price = _num(r.get("wholesale_price"))
            sysp = _num(r.get("system_price"))
            demand = _num(r.get("demand_forecast_mw"))
            wind = _num(r.get("wind_forecast_mw"))
            solar = _num(r.get("solar_forecast_mw"))
            residual = None
            if demand is not None:
                residual = demand - (wind or 0.0) - (solar or 0.0)
            prob_short = None
            if residual is not None:
                prob_short = min(max((residual - 20000) / 15000.0, 0.02), 0.98)
            rows.append(
                PeriodInput(
                    settlement_date=self.day,
                    settlement_period=sp,
                    start_utc=per.start_utc,
                    end_utc=per.end_utc,
                    duration_hours=per.duration_hours,
                    wholesale_price=price if price is not None else 0.0,
                    wholesale_price_sigma=(abs(price) * price_sigma_frac + 3.0) if price is not None else 5.0,
                    wholesale_price_kind=wholesale_kind,
                    system_price=sysp,
                    prob_short=round(prob_short, 3) if prob_short is not None else None,
                    expected_imbalance_price=sysp,
                    system_price_kind=_kind_for(self.statuses, "system_price"),
                    upward_availability_price=upward_availability_price,
                    downward_availability_price=downward_availability_price,
                    availability_price_kind=DataKind.ASSUMPTION,
                    expected_bm_up_margin_gbp_per_mw=1.0,
                    expected_bm_down_margin_gbp_per_mw=1.0,
                    bm_kind=DataKind.ESTIMATED,
                    demand_forecast_mw=demand,
                    wind_forecast_mw=wind,
                    solar_forecast_mw=solar,
                    residual_demand_mw=round(residual, 1) if residual is not None else None,
                )
            )
        return OptimisationInputs(periods=rows, revenue_streams=revenue_streams or RevenueStreams())


def _num(x: Any) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _kind_for(statuses: list[SourceStatus], key: str) -> DataKind:
    for s in statuses:
        if s.source == key:
            return s.kind
    return DataKind.FORECAST


def _synthetic_frame(day: date) -> pd.DataFrame:
    periods = settlement_periods_for_day(day)
    n = len(periods)
    rows = []
    for p in periods:
        demand, wind, solar = _fundamentals(p.settlement_period, n)
        price = diurnal_price_shape(p.settlement_period, n)
        rows.append(
            {
                "settlement_period": p.settlement_period,
                "start_utc": p.start_utc,
                "wholesale_price": price,
                "system_price": price + (12.0 if demand - wind - solar > 25000 else -8.0),
                "demand_forecast_mw": demand,
                "wind_forecast_mw": wind,
                "solar_forecast_mw": solar,
            }
        )
    return pd.DataFrame(rows)


def build_market_snapshot(
    day: date,
    settings: DataSettings | None = None,
    client: ElexonClient | None = None,
    cache: ParquetCache | None = None,
) -> MarketSnapshot:
    """Build a per-period market snapshot with per-source resilience."""
    settings = settings or get_settings()
    statuses: list[SourceStatus] = []
    warnings: list[str] = []

    if settings.offline:
        frame = _synthetic_frame(day)
        statuses = [
            SourceStatus("wholesale", True, DataKind.SYNTHETIC, "offline demo"),
            SourceStatus("system_price", True, DataKind.SYNTHETIC, "offline demo"),
            SourceStatus("demand", True, DataKind.SYNTHETIC, "offline demo"),
            SourceStatus("wind_solar", True, DataKind.SYNTHETIC, "offline demo"),
        ]
        warnings.append("Offline mode: all series are synthetic demo data.")
        return MarketSnapshot(day, frame, statuses, warnings)

    client = client or ElexonClient(settings)
    cache = cache or ParquetCache(settings)
    frm = datetime(day.year, day.month, day.day, 0, 0, tzinfo=UTC)
    to = frm.replace(hour=23, minute=59)

    # Start from the DST-correct period grid.
    base = pd.DataFrame(
        {
            "settlement_period": [p.settlement_period for p in settlement_periods_for_day(day)],
            "start_utc": [p.start_utc for p in settlement_periods_for_day(day)],
        }
    )

    def _merge(df: pd.DataFrame, cols: list[str], source: str, kind: DataKind, retrieved_at):
        nonlocal base
        keep = ["settlement_period", *cols]
        base = base.merge(df[keep], on="settlement_period", how="left")
        statuses.append(SourceStatus(source, True, kind, "live", retrieved_at))

    # Wholesale (MID).
    try:
        mid = client.market_index_data(frm, to)
        mid = mid.rename(columns={"mid_price": "wholesale_price"})
        ts = mid["retrieved_at"].iloc[0] if len(mid) else None
        _merge(mid, ["wholesale_price"], "wholesale", DataKind.OBSERVED, ts)
        cache.put("elexon.MID", day.isoformat(), mid, ts)  # best-effort
    except Exception as exc:  # noqa: BLE001
        syn = _synthetic_frame(day)
        base = base.merge(syn[["settlement_period", "wholesale_price"]], on="settlement_period", how="left")
        statuses.append(SourceStatus("wholesale", False, DataKind.SYNTHETIC, f"fallback: {exc}"))
        warnings.append(f"Wholesale (MID) unavailable — using synthetic prices ({exc}).")

    # System (imbalance) price.
    try:
        sp = client.system_prices(day)
        sp = sp.rename(columns={"system_sell_price": "system_price"})
        ts = sp["retrieved_at"].iloc[0] if len(sp) else None
        _merge(sp, ["system_price"], "system_price", DataKind.OBSERVED, ts)
        cache.put("elexon.system_prices", day.isoformat(), sp, ts)  # best-effort
    except Exception as exc:  # noqa: BLE001
        statuses.append(SourceStatus("system_price", False, DataKind.ESTIMATED, f"fallback: {exc}"))
        warnings.append(f"System price unavailable ({exc}).")

    # Demand forecast.
    try:
        dem = client.demand_forecast(frm, to)
        dem = dem.rename(columns={"national_demand_forecast_mw": "demand_forecast_mw"})
        dem = dem.dropna(subset=["demand_forecast_mw"]).drop_duplicates("settlement_period")
        _merge(dem, ["demand_forecast_mw"], "demand", DataKind.FORECAST, dem["retrieved_at"].iloc[0] if len(dem) else None)
    except Exception as exc:  # noqa: BLE001
        statuses.append(SourceStatus("demand", False, DataKind.ESTIMATED, f"fallback: {exc}"))
        warnings.append(f"Demand forecast unavailable ({exc}).")

    # Wind & solar forecast.
    try:
        ws = client.wind_solar_forecast(frm, to).drop_duplicates("settlement_period")
        _merge(ws, ["wind_forecast_mw", "solar_forecast_mw"], "wind_solar", DataKind.FORECAST,
               ws["retrieved_at"].iloc[0] if len(ws) else None)
    except Exception as exc:  # noqa: BLE001
        statuses.append(SourceStatus("wind_solar", False, DataKind.ESTIMATED, f"fallback: {exc}"))
        warnings.append(f"Wind/solar forecast unavailable ({exc}).")

    # Ensure expected columns exist.
    for col in ["wholesale_price", "system_price", "demand_forecast_mw", "wind_forecast_mw", "solar_forecast_mw"]:
        if col not in base.columns:
            base[col] = pd.NA

    return MarketSnapshot(day, base, statuses, warnings)
