"""Synthetic, clearly-labelled demo scenarios.

Everything produced here is ``DataKind.SYNTHETIC`` — it is generated, not observed,
and must never be presented as real market data. The generator produces a plausible
GB half-hourly shape: an overnight trough, a midday solar-driven dip (occasionally
negative) and an evening peak, plus correlated wind/demand/solar fundamentals.
"""

from __future__ import annotations

import math
from datetime import date, datetime

from gb_battery.lineage import DataKind
from gb_battery.optimiser.inputs import (
    OptimisationInputs,
    PeriodInput,
    RevenueStreams,
)
from gb_battery.settlement import (
    UTC,
    SettlementPeriod,
    settlement_periods_for_day,
)


def diurnal_price_shape(sp: int, n: int, *, base: float = 70.0, amp: float = 55.0,
                        midday_dip: float = 45.0, negative_afternoon: bool = False) -> float:
    """Return a GBP/MWh wholesale price for settlement period ``sp`` of ``n``."""
    frac = (sp - 1) / max(n - 1, 1)
    hour = frac * 24.0
    # Evening peak ~ 18:00, morning shoulder ~ 08:00.
    evening = amp * math.exp(-((hour - 18.0) ** 2) / (2 * 2.5**2))
    morning = 0.5 * amp * math.exp(-((hour - 8.0) ** 2) / (2 * 2.0**2))
    # Midday solar dip ~ 13:00.
    solar_dip = midday_dip * math.exp(-((hour - 13.0) ** 2) / (2 * 2.5**2))
    overnight = -18.0 * math.exp(-((hour - 3.5) ** 2) / (2 * 3.0**2))
    price = base + evening + morning - solar_dip + overnight
    if negative_afternoon:
        price -= 60.0 * math.exp(-((hour - 13.5) ** 2) / (2 * 1.8**2))
    return round(price, 2)


def _fundamentals(sp: int, n: int) -> tuple[float, float, float]:
    """Return (demand_mw, wind_mw, solar_mw) for a period."""
    frac = (sp - 1) / max(n - 1, 1)
    hour = frac * 24.0
    demand = 30000 + 9000 * math.sin((hour - 6) / 24 * 2 * math.pi) + 4000 * math.exp(
        -((hour - 18) ** 2) / (2 * 2.0**2)
    )
    wind = 8000 + 5000 * math.sin((hour - 2) / 24 * 2 * math.pi * 0.7)
    solar = max(0.0, 7000 * math.exp(-((hour - 13) ** 2) / (2 * 3.0**2)))
    return round(demand, 1), round(max(wind, 0.0), 1), round(solar, 1)


def build_period_inputs(
    periods: list[SettlementPeriod],
    prices: list[float],
    *,
    price_sigma_frac: float = 0.15,
    upward_availability_price: float = 6.0,
    downward_availability_price: float = 4.0,
    price_kind: DataKind = DataKind.SYNTHETIC,
) -> list[PeriodInput]:
    """Assemble :class:`PeriodInput` rows from settlement periods and a price path."""
    n = len(periods)
    rows: list[PeriodInput] = []
    for per, price in zip(periods, prices, strict=True):
        demand, wind, solar = _fundamentals(per.settlement_period, n)
        residual = demand - wind - solar
        # System (imbalance) price loosely tracks wholesale with a spread; short when tight.
        system_price = price + (12.0 if residual > 25000 else -8.0)
        prob_short = min(max((residual - 20000) / 15000.0, 0.02), 0.98)
        rows.append(
            PeriodInput(
                settlement_date=per.settlement_date,
                settlement_period=per.settlement_period,
                start_utc=per.start_utc,
                end_utc=per.end_utc,
                duration_hours=per.duration_hours,
                wholesale_price=price,
                wholesale_price_sigma=abs(price) * price_sigma_frac + 3.0,
                wholesale_price_kind=price_kind,
                system_price=round(system_price, 2),
                prob_short=round(prob_short, 3),
                expected_imbalance_price=round(system_price, 2),
                system_price_kind=price_kind,
                upward_availability_price=upward_availability_price,
                downward_availability_price=downward_availability_price,
                availability_price_kind=DataKind.ASSUMPTION,
                expected_bm_up_margin_gbp_per_mw=2.0 if prob_short > 0.5 else 0.5,
                expected_bm_down_margin_gbp_per_mw=0.5 if prob_short > 0.5 else 2.0,
                bm_kind=DataKind.ESTIMATED,
                demand_forecast_mw=demand,
                wind_forecast_mw=wind,
                solar_forecast_mw=solar,
                residual_demand_mw=round(residual, 1),
            )
        )
    return rows


def synthetic_day_inputs(
    day: date | None = None,
    *,
    negative_afternoon: bool = False,
    revenue_streams: RevenueStreams | None = None,
) -> OptimisationInputs:
    """Build a full synthetic settlement-day of optimiser inputs (DST-aware count)."""
    day = day or date(2025, 1, 15)
    periods = settlement_periods_for_day(day)
    n = len(periods)
    prices = [
        diurnal_price_shape(p.settlement_period, n, negative_afternoon=negative_afternoon)
        for p in periods
    ]
    rows = build_period_inputs(periods, prices)
    return OptimisationInputs(
        periods=rows,
        revenue_streams=revenue_streams or RevenueStreams(),
    )


def synthetic_horizon_inputs(
    start_utc: datetime, horizon_periods: int, *, negative_afternoon: bool = False
) -> OptimisationInputs:
    """Build inputs for an arbitrary horizon starting at ``start_utc`` (UTC)."""
    from gb_battery.settlement import settlement_periods_between

    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=UTC)
    periods = settlement_periods_between(start_utc, horizon_periods)
    prices = [
        diurnal_price_shape(p.settlement_period, 48, negative_afternoon=negative_afternoon)
        for p in periods
    ]
    rows = build_period_inputs(periods, prices)
    return OptimisationInputs(periods=rows)
