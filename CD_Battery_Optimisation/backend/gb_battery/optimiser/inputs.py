"""Inputs consumed by the optimiser.

These models deliberately decouple the optimiser from any data source: forecasts,
Elexon MID, NESO data, CSV uploads and synthetic scenarios all resolve down to a
list of :class:`PeriodInput`. Every economic driver is an explicit field so the UI
can show *why* a decision was taken.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from gb_battery.lineage import DataKind


class RevenueStreams(BaseModel):
    """Toggle which revenue streams the optimiser co-optimises."""

    wholesale: bool = True
    upward_availability: bool = True
    downward_availability: bool = True
    bm_activation: bool = True
    imbalance: bool = True


class PeriodInput(BaseModel):
    """All economic drivers for a single Settlement Period.

    Prices are GBP/MWh unless noted. Availability prices are GBP/MW/h.
    ``*_kind`` fields record whether each headline value is observed/forecast/etc.
    """

    settlement_date: date
    settlement_period: int
    start_utc: datetime
    end_utc: datetime
    duration_hours: float = 0.5

    # --- Wholesale ---
    wholesale_price: float = 0.0  # GBP/MWh, forecast or observed reference price
    wholesale_price_sigma: float = 0.0  # forecast std-dev (0 => deterministic)
    wholesale_price_kind: DataKind = DataKind.FORECAST

    # --- Imbalance / system ---
    system_price: float | None = None  # GBP/MWh single imbalance price (SSP=SBP)
    prob_short: float | None = None  # P(system is short) in [0,1]
    expected_imbalance_price: float | None = None  # GBP/MWh, used for imbalance cost
    system_price_kind: DataKind = DataKind.FORECAST

    # --- Availability (ancillary service) prices, GBP/MW/h ---
    upward_availability_price: float = 0.0
    downward_availability_price: float = 0.0
    availability_price_kind: DataKind = DataKind.ASSUMPTION

    # --- Expected BM activation value, GBP per MW of reserved capability per hour ---
    # = acceptance_probability * accepted_fraction * activation_margin_per_mwh
    # Pre-computed upstream (see service.py) so the LP stays linear.
    expected_bm_up_margin_gbp_per_mw: float = 0.0
    expected_bm_down_margin_gbp_per_mw: float = 0.0
    bm_kind: DataKind = DataKind.ESTIMATED

    # --- Fundamentals (for display / explanation, not directly in objective) ---
    demand_forecast_mw: float | None = None
    wind_forecast_mw: float | None = None
    solar_forecast_mw: float | None = None
    residual_demand_mw: float | None = None

    # --- Portfolio mode ---
    renewable_generation_mwh: float | None = None
    demand_obligation_mwh: float | None = None

    @property
    def net_residual_demand(self) -> float | None:
        if self.residual_demand_mw is not None:
            return self.residual_demand_mw
        if self.demand_forecast_mw is None:
            return None
        wind = self.wind_forecast_mw or 0.0
        solar = self.solar_forecast_mw or 0.0
        return self.demand_forecast_mw - wind - solar


class OptimisationInputs(BaseModel):
    """Complete input bundle for one optimisation run."""

    periods: list[PeriodInput] = Field(..., min_length=1)
    revenue_streams: RevenueStreams = Field(default_factory=RevenueStreams)

    # Risk / mode
    risk_aversion: float = Field(default=0.0, ge=0)  # CVaR weight (stochastic mode)
    cvar_alpha: float = Field(default=0.95, gt=0, lt=1)

    # Portfolio
    portfolio_mode: bool = False
    imbalance_penalty_gbp_per_mwh: float = Field(default=0.0, ge=0)

    @property
    def horizon(self) -> int:
        return len(self.periods)
