"""Shared services for the API: build optimiser inputs from a chosen data source."""

from __future__ import annotations

from datetime import date

import pandas as pd

from gb_battery.data.market_snapshot import MarketSnapshot, build_market_snapshot
from gb_battery.data.settings import DataSettings
from gb_battery.demo.sample_data import sample_day
from gb_battery.demo.scenarios import synthetic_day_inputs
from gb_battery.lineage import DataKind
from gb_battery.optimiser.inputs import OptimisationInputs, PeriodInput, RevenueStreams
from gb_battery.settlement import settlement_periods_for_day

# Data source options exposed to the UI.
SOURCES = ["synthetic", "sample", "elexon"]


def _inputs_from_sample(day: date, streams: RevenueStreams) -> OptimisationInputs:
    df = sample_day(day)
    if df.empty:
        # Fall back to the first available sample day.
        df = sample_day()
        day = df["settlement_date"].iloc[0]
    periods = settlement_periods_for_day(day)
    by_sp = {int(r["settlement_period"]): r for _, r in df.iterrows()}
    rows: list[PeriodInput] = []
    for per in periods:
        r = by_sp.get(per.settlement_period)
        if r is None:
            continue
        demand = float(r["demand_forecast_mw"])
        wind = float(r["wind_forecast_mw"])
        solar = float(r["solar_forecast_mw"])
        residual = demand - wind - solar
        prob_short = min(max((residual - 20000) / 15000.0, 0.02), 0.98)
        rows.append(
            PeriodInput(
                settlement_date=day,
                settlement_period=per.settlement_period,
                start_utc=per.start_utc,
                end_utc=per.end_utc,
                duration_hours=per.duration_hours,
                wholesale_price=float(r["wholesale_price_forecast"]),
                wholesale_price_sigma=abs(float(r["wholesale_price_forecast"])) * 0.15 + 3.0,
                wholesale_price_kind=DataKind.FORECAST,
                system_price=float(r["system_price"]),
                prob_short=round(prob_short, 3),
                expected_imbalance_price=float(r["system_price"]),
                upward_availability_price=6.0,
                downward_availability_price=4.0,
                availability_price_kind=DataKind.ASSUMPTION,
                expected_bm_up_margin_gbp_per_mw=1.0,
                expected_bm_down_margin_gbp_per_mw=1.0,
                demand_forecast_mw=demand,
                wind_forecast_mw=wind,
                solar_forecast_mw=solar,
                residual_demand_mw=round(residual, 1),
            )
        )
    return OptimisationInputs(periods=rows, revenue_streams=streams)


def build_inputs_for_day(
    day: date,
    source: str,
    *,
    offline: bool = False,
    streams: RevenueStreams | None = None,
) -> tuple[OptimisationInputs, MarketSnapshot | None]:
    """Build optimiser inputs for ``day`` from the requested source.

    Returns (inputs, snapshot_or_None). ``snapshot`` is populated for the 'elexon'
    source so the caller can surface data-source provenance.
    """
    streams = streams or RevenueStreams()
    if source == "synthetic":
        return synthetic_day_inputs(day, revenue_streams=streams), None
    if source == "sample":
        return _inputs_from_sample(day, streams), None
    if source == "elexon":
        settings = DataSettings(offline=offline)
        snap = build_market_snapshot(day, settings=settings)
        return snap.to_optimisation_inputs(revenue_streams=streams), snap
    raise ValueError(f"Unknown source '{source}' (choose from {SOURCES})")


def snapshot_to_payload(snap: MarketSnapshot) -> dict:
    """Serialise a MarketSnapshot for the API."""
    frame = snap.frame.copy()
    # Convert timestamps to ISO strings for JSON.
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = frame[col].astype(str)
    return {
        "day": snap.day.isoformat(),
        "periods": frame.where(pd.notna(frame), None).to_dict(orient="records"),
        "statuses": [
            {
                "source": s.source,
                "ok": s.ok,
                "kind": s.kind.value,
                "detail": s.detail,
                "retrieved_at": s.retrieved_at.isoformat() if s.retrieved_at else None,
            }
            for s in snap.statuses
        ],
        "warnings": snap.warnings,
    }
