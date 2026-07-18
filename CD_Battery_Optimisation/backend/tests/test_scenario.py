"""Scenario generation, Scenario Lab transforms, and stochastic optimisation."""

from __future__ import annotations

from datetime import date

from gb_battery.battery.config import BatteryConfig
from gb_battery.demo.scenarios import synthetic_day_inputs
from gb_battery.scenario.generator import generate_price_scenarios
from gb_battery.scenario.lab import apply_scenario, negative_price_afternoon
from gb_battery.scenario.stochastic import optimise_stochastic


def test_scenario_generation_shape_and_weights():
    inp = synthetic_day_inputs(date(2025, 1, 15))
    sc = generate_price_scenarios(inp, n_scenarios=15, seed=3)
    assert sc.wholesale_prices.shape == (15, len(inp.periods))
    assert abs(sc.probabilities.sum() - 1.0) < 1e-9


def test_scenario_lab_transform_is_nondestructive():
    inp = synthetic_day_inputs(date(2025, 1, 15))
    base_prices = [p.wholesale_price for p in inp.periods]
    modified = negative_price_afternoon(inp)
    # Base is unchanged; modified has lower afternoon prices.
    assert [p.wholesale_price for p in inp.periods] == base_prices
    assert any(m.wholesale_price < b.wholesale_price for m, b in zip(modified.periods, inp.periods, strict=True))


def test_apply_named_scenario_round_trip():
    cfg = BatteryConfig()
    inp = synthetic_day_inputs(date(2025, 1, 15))
    cfg2, inp2 = apply_scenario("battery_nearly_full", cfg, inp)
    assert cfg2.initial_soc_mwh > cfg.initial_soc_mwh


def test_risk_aversion_reduces_expected_pnl():
    cfg = BatteryConfig()
    inp = synthetic_day_inputs(date(2025, 1, 15))
    sc = generate_price_scenarios(inp, n_scenarios=25, seed=7)
    neutral = optimise_stochastic(cfg, inp, scenarios=sc, risk_aversion=0.0)
    averse = optimise_stochastic(cfg, inp, scenarios=sc, risk_aversion=3.0)
    assert neutral.status == "optimal" and averse.status == "optimal"
    # A risk-averse objective never has higher expected P&L than the neutral one.
    assert averse.total_expected_pnl_gbp <= neutral.total_expected_pnl_gbp + 1e-6


def test_robust_mode_is_feasible():
    cfg = BatteryConfig()
    inp = synthetic_day_inputs(date(2025, 1, 15))
    res = optimise_stochastic(cfg, inp, n_scenarios=10, robust=True, seed=1)
    assert res.status == "optimal"
