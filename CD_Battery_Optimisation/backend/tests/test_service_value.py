"""Service-value decomposition tests."""

from __future__ import annotations

from gb_battery.optimiser.service import ServiceValueInputs, value_service


def test_availability_only_value():
    b = value_service(ServiceValueInputs(availability_price_gbp_per_mw_h=8.0))
    assert b.net_value_gbp_per_mw_h == 8.0
    assert b.expected_activation_margin == 0.0


def test_expected_activation_margin_scales_with_probability():
    b = value_service(
        ServiceValueInputs(
            availability_price_gbp_per_mw_h=0.0,
            acceptance_probability=0.5,
            accepted_fraction=1.0,
            activation_margin_gbp_per_mwh=40.0,
            activation_duration_h=1.0,
        )
    )
    # 0.5 * 1.0 * 40 * 1.0 = 20
    assert abs(b.expected_activation_margin - 20.0) < 1e-9


def test_costs_reduce_net_value():
    b = value_service(
        ServiceValueInputs(
            availability_price_gbp_per_mw_h=10.0,
            degradation_cost_gbp_per_mw_h=2.0,
            efficiency_loss_gbp_per_mw_h=1.0,
            wholesale_opportunity_cost_gbp_per_mw_h=3.0,
            non_delivery_probability=0.1,
            non_delivery_penalty_gbp_per_mw=50.0,
        )
    )
    # 10 - 2 - 1 - 3 - 0.1*50 = -1
    assert abs(b.net_value_gbp_per_mw_h - (-1.0)) < 1e-9
    assert not b.is_worth_reserving


def test_submitting_price_alone_earns_nothing():
    # Zero acceptance probability -> no activation revenue regardless of margin.
    b = value_service(
        ServiceValueInputs(acceptance_probability=0.0, activation_margin_gbp_per_mwh=100.0)
    )
    assert b.expected_activation_margin == 0.0
