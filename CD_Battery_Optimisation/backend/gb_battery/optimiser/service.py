"""Ancillary-service value decomposition.

A single "service price" is misleading. The economic value of reserving a MW of
flexible capability for one hour is::

    service_value
        =  availability_payment
         + expected_activation_margin
         - wholesale_opportunity_cost
         - energy_restoration_cost
         - degradation
         - efficiency_losses
         - expected_non_delivery_penalty

This module builds those components explicitly and, critically, computes the
**expected BM activation margin** as ``acceptance_probability * accepted_fraction *
activation_margin`` — submitting a bid/offer price alone does not earn money.

Everything here produces *estimated* values (``DataKind.ESTIMATED`` /
``ASSUMPTION``). The results are fed into :class:`~gb_battery.optimiser.inputs.PeriodInput`
so the optimiser stays linear.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ServiceValueInputs(BaseModel):
    """Assumptions/estimates needed to value one direction of a reserve product."""

    availability_price_gbp_per_mw_h: float = 0.0
    # BM activation
    acceptance_probability: float = Field(default=0.0, ge=0, le=1)
    accepted_fraction: float = Field(default=1.0, ge=0, le=1)  # of reserved MW
    activation_margin_gbp_per_mwh: float = 0.0  # (bid/offer price - marginal cost)
    activation_duration_h: float = Field(default=0.5, gt=0)
    # Costs of holding / delivering the reserve
    wholesale_opportunity_cost_gbp_per_mw_h: float = 0.0
    energy_restoration_cost_gbp_per_mw_h: float = 0.0
    degradation_cost_gbp_per_mw_h: float = 0.0
    efficiency_loss_gbp_per_mw_h: float = 0.0
    non_delivery_probability: float = Field(default=0.0, ge=0, le=1)
    non_delivery_penalty_gbp_per_mw: float = 0.0


class ServiceValueBreakdown(BaseModel):
    availability_payment: float
    expected_activation_margin: float
    wholesale_opportunity_cost: float
    energy_restoration_cost: float
    degradation: float
    efficiency_losses: float
    expected_non_delivery_penalty: float
    net_value_gbp_per_mw_h: float

    @property
    def is_worth_reserving(self) -> bool:
        return self.net_value_gbp_per_mw_h > 0


def value_service(x: ServiceValueInputs) -> ServiceValueBreakdown:
    """Decompose the per-MW-per-hour value of reserving a service."""
    expected_activation_margin = (
        x.acceptance_probability
        * x.accepted_fraction
        * x.activation_margin_gbp_per_mwh
        * x.activation_duration_h
    )
    expected_penalty = x.non_delivery_probability * x.non_delivery_penalty_gbp_per_mw
    net = (
        x.availability_price_gbp_per_mw_h
        + expected_activation_margin
        - x.wholesale_opportunity_cost_gbp_per_mw_h
        - x.energy_restoration_cost_gbp_per_mw_h
        - x.degradation_cost_gbp_per_mw_h
        - x.efficiency_loss_gbp_per_mw_h
        - expected_penalty
    )
    return ServiceValueBreakdown(
        availability_payment=x.availability_price_gbp_per_mw_h,
        expected_activation_margin=expected_activation_margin,
        wholesale_opportunity_cost=x.wholesale_opportunity_cost_gbp_per_mw_h,
        energy_restoration_cost=x.energy_restoration_cost_gbp_per_mw_h,
        degradation=x.degradation_cost_gbp_per_mw_h,
        efficiency_losses=x.efficiency_loss_gbp_per_mw_h,
        expected_non_delivery_penalty=expected_penalty,
        net_value_gbp_per_mw_h=net,
    )
