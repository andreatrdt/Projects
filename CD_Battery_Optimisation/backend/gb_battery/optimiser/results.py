"""Optimiser output models."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel


class ActionLabel(StrEnum):
    CHARGE = "CHARGE"
    DISCHARGE = "DISCHARGE"
    IDLE = "IDLE"
    RESERVE_UP = "RESERVE UP"
    RESERVE_DOWN = "RESERVE DOWN"


class MarginalValues(BaseModel):
    """Shadow / perturbation values exposed per period where meaningful.

    Sign convention: a positive ``stored_energy`` means one extra MWh *already in the
    battery* at the start of this period would improve the objective by that many GBP.
    """

    stored_energy_gbp_per_mwh: float | None = None  # value of +1 MWh of charge
    empty_capacity_gbp_per_mwh: float | None = None  # value of +1 MWh of headroom
    charge_power_gbp_per_mw: float | None = None
    discharge_power_gbp_per_mw: float | None = None


class PeriodResult(BaseModel):
    """Everything the Optimisation Terminal shows for one Settlement Period."""

    settlement_date: date
    settlement_period: int
    start_utc: datetime
    end_utc: datetime
    duration_hours: float

    # Inputs echoed for the table / explanation
    wholesale_price: float
    wholesale_price_sigma: float
    system_price: float | None
    prob_short: float | None
    demand_forecast_mw: float | None
    wind_forecast_mw: float | None
    solar_forecast_mw: float | None
    residual_demand_mw: float | None

    # Decisions
    action: ActionLabel
    charge_mw: float
    discharge_mw: float
    energy_charged_mwh: float
    energy_discharged_mwh: float
    beginning_soc_mwh: float
    ending_soc_mwh: float
    upward_reserved_mw: float
    downward_reserved_mw: float

    # Economics (all GBP for this period; sign convention: revenue positive)
    wholesale_pnl_gbp: float
    service_pnl_gbp: float
    bm_activation_pnl_gbp: float
    degradation_cost_gbp: float
    imbalance_cost_gbp: float
    total_expected_pnl_gbp: float

    # Portfolio (optional)
    portfolio_position_mwh: float | None = None
    residual_imbalance_mwh: float | None = None

    # Diagnostics
    binding_constraints: list[str] = []
    marginals: MarginalValues = MarginalValues()
    explanation: str = ""


class OptimisationResult(BaseModel):
    """Full optimiser output for one run."""

    status: str  # "optimal", "infeasible", ...
    solver: str
    objective_gbp: float
    periods: list[PeriodResult]

    # Summary economics
    total_wholesale_pnl_gbp: float
    total_service_pnl_gbp: float
    total_bm_activation_pnl_gbp: float
    total_degradation_cost_gbp: float
    total_imbalance_cost_gbp: float
    terminal_soc_value_gbp: float
    total_expected_pnl_gbp: float

    # Cycling
    full_cycle_equivalents: float

    # Metadata
    horizon: int
    portfolio_mode: bool
    risk_aversion: float
    warnings: list[str] = []
