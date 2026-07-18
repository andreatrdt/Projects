"""Economic-behaviour tests: arbitrage, degradation, objective decomposition."""

from __future__ import annotations

from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.deterministic import optimise

from tests.conftest import make_inputs


def test_negative_price_triggers_charging(lossless_config, wholesale_only):
    res = optimise(lossless_config, make_inputs([-50.0], streams=wholesale_only),
                   compute_marginals=False)
    p = res.periods[0]
    assert p.charge_mw > 1e-3
    # Charging at a negative price is a *revenue* (you are paid to consume).
    assert p.wholesale_pnl_gbp > 0


def test_high_price_triggers_discharging(lossless_config, wholesale_only):
    res = optimise(lossless_config, make_inputs([500.0], streams=wholesale_only),
                   compute_marginals=False)
    p = res.periods[0]
    assert p.discharge_mw > 1e-3
    assert p.wholesale_pnl_gbp > 0


def test_arbitrage_buys_low_sells_high(lossless_config, wholesale_only):
    # Cheap then expensive: charge in period 0, discharge in period 1.
    # Start empty so the behaviour is pure arbitrage, not inventory liquidation.
    cfg = lossless_config.model_copy(update={"initial_soc_mwh": 0.0})
    res = optimise(cfg, make_inputs([10.0, 100.0], streams=wholesale_only),
                   compute_marginals=False)
    assert res.periods[0].charge_mw > 1e-3
    assert res.periods[1].discharge_mw > 1e-3
    assert res.total_expected_pnl_gbp > 0


def test_degradation_prevents_uneconomic_cycling(wholesale_only):
    # Spread of only 5 GBP/MWh; degradation of 10/MWh (each direction) kills the trade.
    # Start empty (real cycle) and lossless (isolate degradation from efficiency loss).
    cfg = BatteryConfig(initial_soc_mwh=0.0, minimum_terminal_soc_mwh=0.0,
                        preferred_terminal_soc_mwh=0.0, terminal_soc_value_gbp_per_mwh=0.0,
                        charge_efficiency=1.0, discharge_efficiency=1.0,
                        degradation_cost_gbp_per_mwh_throughput=10.0)
    res = optimise(cfg, make_inputs([50.0, 55.0], streams=wholesale_only),
                   compute_marginals=False)
    throughput = sum(p.energy_charged_mwh + p.energy_discharged_mwh for p in res.periods)
    assert throughput < 1e-2  # essentially no cycling

    # With zero degradation the same spread *is* worth cycling.
    cfg2 = cfg.model_copy(update={"degradation_cost_gbp_per_mwh_throughput": 0.0})
    res2 = optimise(cfg2, make_inputs([50.0, 55.0], streams=wholesale_only),
                    compute_marginals=False)
    throughput2 = sum(p.energy_charged_mwh + p.energy_discharged_mwh for p in res2.periods)
    assert throughput2 > 1.0


def test_objective_components_sum_to_total(lossless_config):
    prices = [-30, 20, 90, 10, 80]
    res = optimise(lossless_config, make_inputs(prices, up_avail=5.0, down_avail=3.0),
                   compute_marginals=False)
    recomputed = (
        res.total_wholesale_pnl_gbp
        + res.total_service_pnl_gbp
        + res.total_bm_activation_pnl_gbp
        - res.total_degradation_cost_gbp
        - res.total_imbalance_cost_gbp
        + res.terminal_soc_value_gbp
    )
    assert abs(recomputed - res.total_expected_pnl_gbp) < 1e-2


def test_period_pnl_sums_to_total(lossless_config):
    prices = [-30, 20, 90, 10, 80]
    res = optimise(lossless_config, make_inputs(prices, up_avail=5.0, down_avail=3.0),
                   compute_marginals=False)
    period_sum = sum(p.total_expected_pnl_gbp for p in res.periods)
    # Total includes the terminal SoC value which is not attributed to any period.
    assert abs(period_sum + res.terminal_soc_value_gbp - res.total_expected_pnl_gbp) < 1e-2


def test_availability_revenue_is_positive_when_offered(lossless_config):
    res = optimise(lossless_config, make_inputs([50.0, 50.0], up_avail=20.0, down_avail=20.0),
                   compute_marginals=False)
    assert res.total_service_pnl_gbp > 0
