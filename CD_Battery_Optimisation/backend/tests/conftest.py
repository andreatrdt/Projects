"""Shared pytest fixtures and builders for constructing controlled optimiser inputs."""

from __future__ import annotations

from datetime import datetime

import pytest
from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.inputs import OptimisationInputs, PeriodInput, RevenueStreams
from gb_battery.settlement import UTC, settlement_periods_between


def make_inputs(
    prices: list[float],
    *,
    start: datetime | None = None,
    up_avail: list[float] | float | None = None,
    down_avail: list[float] | float | None = None,
    imb_price: list[float] | float | None = None,
    bm_up: list[float] | float | None = None,
    bm_down: list[float] | float | None = None,
    streams: RevenueStreams | None = None,
    portfolio: bool = False,
    renewable: list[float] | None = None,
    demand_obligation: list[float] | None = None,
    imbalance_penalty: float = 0.0,
) -> OptimisationInputs:
    """Build ``OptimisationInputs`` from a price path plus optional service prices."""
    start = start or datetime(2025, 1, 15, 0, 0, tzinfo=UTC)
    n = len(prices)
    periods = settlement_periods_between(start, n)

    def _at(x, i, default=0.0):
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        return float(x[i])

    rows: list[PeriodInput] = []
    for i, per in enumerate(periods):
        rows.append(
            PeriodInput(
                settlement_date=per.settlement_date,
                settlement_period=per.settlement_period,
                start_utc=per.start_utc,
                end_utc=per.end_utc,
                duration_hours=per.duration_hours,
                wholesale_price=prices[i],
                upward_availability_price=_at(up_avail, i),
                downward_availability_price=_at(down_avail, i),
                expected_imbalance_price=_at(imb_price, i, default=prices[i]),
                expected_bm_up_margin_gbp_per_mw=_at(bm_up, i),
                expected_bm_down_margin_gbp_per_mw=_at(bm_down, i),
                renewable_generation_mwh=(renewable[i] if renewable else None),
                demand_obligation_mwh=(demand_obligation[i] if demand_obligation else None),
            )
        )
    return OptimisationInputs(
        periods=rows,
        revenue_streams=streams or RevenueStreams(),
        portfolio_mode=portfolio,
        imbalance_penalty_gbp_per_mwh=imbalance_penalty,
    )


@pytest.fixture
def lossless_config() -> BatteryConfig:
    """A simple, lossless 50MW/100MWh battery for clean arithmetic in tests."""
    return BatteryConfig(
        name="test",
        energy_capacity_mwh=100.0,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=100.0,
        initial_soc_mwh=50.0,
        maximum_charge_mw=50.0,
        maximum_discharge_mw=50.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        grid_import_limit_mw=50.0,
        grid_export_limit_mw=50.0,
        degradation_cost_gbp_per_mwh_throughput=0.0,
        minimum_terminal_soc_mwh=0.0,
        preferred_terminal_soc_mwh=0.0,
        terminal_soc_value_gbp_per_mwh=0.0,
        maximum_cycles_per_day=10.0,
        upward_service_duration_h=1.0,
        downward_service_duration_h=1.0,
    )


@pytest.fixture
def wholesale_only() -> RevenueStreams:
    return RevenueStreams(
        wholesale=True,
        upward_availability=False,
        downward_availability=False,
        bm_activation=False,
        imbalance=False,
    )
