"""Physical-feasibility and constraint tests for the optimiser."""

from __future__ import annotations

from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.deterministic import optimise

from tests.conftest import make_inputs


def _all_socs(res):
    socs = [p.beginning_soc_mwh for p in res.periods]
    socs.append(res.periods[-1].ending_soc_mwh)
    return socs


def test_soc_stays_within_bounds(lossless_config):
    prices = [10, -30, 90, 20, 80, 15, 100, 5]
    res = optimise(lossless_config, make_inputs(prices), compute_marginals=False)
    for s in _all_socs(res):
        assert lossless_config.effective_min_soc - 1e-6 <= s <= lossless_config.effective_max_soc + 1e-6


def test_soc_evolution_charge_adds_energy(lossless_config, wholesale_only):
    # Single strongly-negative period -> the battery charges; SoC rises by charge*dt.
    res = optimise(lossless_config, make_inputs([-100.0], streams=wholesale_only),
                   compute_marginals=False)
    p = res.periods[0]
    assert p.charge_mw > 0
    expected_soc = p.beginning_soc_mwh + p.charge_mw * p.duration_hours  # eff = 1
    assert abs(p.ending_soc_mwh - expected_soc) < 1e-4


def test_efficiency_losses_reduce_stored_energy(wholesale_only):
    cfg = BatteryConfig(charge_efficiency=0.9, discharge_efficiency=0.9,
                        initial_soc_mwh=0.0, minimum_terminal_soc_mwh=0.0,
                        terminal_soc_value_gbp_per_mwh=0.0, degradation_cost_gbp_per_mwh_throughput=0.0)
    res = optimise(cfg, make_inputs([-100.0], streams=wholesale_only), compute_marginals=False)
    p = res.periods[0]
    # Stored energy added = charge_efficiency * grid energy.
    stored = p.ending_soc_mwh - p.beginning_soc_mwh
    grid = p.charge_mw * p.duration_hours
    assert abs(stored - 0.9 * grid) < 1e-4


def test_no_simultaneous_charge_and_discharge(lossless_config):
    prices = [-50, 100, -40, 90, -30, 80]
    res = optimise(lossless_config, make_inputs(prices), compute_marginals=False)
    for p in res.periods:
        assert not (p.charge_mw > 1e-4 and p.discharge_mw > 1e-4)


def test_power_limits_respected():
    cfg = BatteryConfig(maximum_charge_mw=30.0, maximum_discharge_mw=40.0,
                        initial_soc_mwh=50.0, minimum_terminal_soc_mwh=0.0,
                        terminal_soc_value_gbp_per_mwh=0.0)
    res = optimise(cfg, make_inputs([-100, 100, -100, 100]), compute_marginals=False)
    for p in res.periods:
        assert p.charge_mw <= 30.0 + 1e-4
        assert p.discharge_mw <= 40.0 + 1e-4


def test_grid_export_limit_binds():
    cfg = BatteryConfig(grid_export_limit_mw=20.0, maximum_discharge_mw=50.0,
                        initial_soc_mwh=100.0, minimum_terminal_soc_mwh=0.0,
                        terminal_soc_value_gbp_per_mwh=0.0)
    res = optimise(cfg, make_inputs([200.0]), compute_marginals=False)
    assert res.periods[0].discharge_mw <= 20.0 + 1e-4


def test_terminal_soc_floor_enforced():
    cfg = BatteryConfig(initial_soc_mwh=60.0, minimum_terminal_soc_mwh=40.0,
                        terminal_soc_value_gbp_per_mwh=0.0)
    # High prices everywhere would empty the battery absent the terminal floor.
    res = optimise(cfg, make_inputs([500, 500, 500, 500]), compute_marginals=False)
    assert res.periods[-1].ending_soc_mwh >= 40.0 - 1e-3


def test_upward_reserve_energy_limited(lossless_config):
    # A single period offering large upward availability; SoC caps reservable MW.
    cfg = lossless_config.model_copy(update={"initial_soc_mwh": 10.0,
                                             "upward_service_duration_h": 1.0})
    res = optimise(cfg, make_inputs([50.0], up_avail=[1000.0]), compute_marginals=False)
    p = res.periods[0]
    # up * h_up / discharge_eff <= soc - min  => up <= 10 (h=1, eff=1, soc=10)
    assert p.upward_reserved_mw <= 10.0 + 1e-3


def test_downward_reserve_energy_limited(lossless_config):
    cfg = lossless_config.model_copy(update={"initial_soc_mwh": 90.0,
                                             "downward_service_duration_h": 1.0})
    res = optimise(cfg, make_inputs([50.0], down_avail=[1000.0]), compute_marginals=False)
    p = res.periods[0]
    # eff_c * down * h_down <= max_soc - soc => down <= 10
    assert p.downward_reserved_mw <= 10.0 + 1e-3


def test_upward_reserve_power_limited():
    # Plenty of energy but limited discharge headroom while also discharging.
    cfg = BatteryConfig(initial_soc_mwh=100.0, maximum_discharge_mw=50.0,
                        upward_service_duration_h=0.5, minimum_terminal_soc_mwh=0.0,
                        terminal_soc_value_gbp_per_mwh=0.0)
    res = optimise(cfg, make_inputs([50.0], up_avail=[1000.0]), compute_marginals=False)
    p = res.periods[0]
    # up <= max_discharge - net_export; with no forced discharge, up <= 50.
    assert p.upward_reserved_mw <= 50.0 + 1e-3
