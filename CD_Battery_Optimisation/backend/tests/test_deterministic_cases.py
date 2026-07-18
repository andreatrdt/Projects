"""The six named deterministic optimisation acceptance cases from the spec."""

from __future__ import annotations

from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.deterministic import optimise
from gb_battery.optimiser.inputs import RevenueStreams

from tests.conftest import make_inputs

WHOLESALE_ONLY = RevenueStreams(
    wholesale=True, upward_availability=False, downward_availability=False,
    bm_activation=False, imbalance=False,
)


def _case_config(**overrides) -> BatteryConfig:
    base = {
        "energy_capacity_mwh": 100.0, "minimum_soc_mwh": 0.0, "maximum_soc_mwh": 100.0,
        "maximum_charge_mw": 50.0, "maximum_discharge_mw": 50.0,
        "charge_efficiency": 1.0, "discharge_efficiency": 1.0,
        "grid_import_limit_mw": 50.0, "grid_export_limit_mw": 50.0,
        "degradation_cost_gbp_per_mwh_throughput": 0.0,
        "minimum_terminal_soc_mwh": 0.0, "preferred_terminal_soc_mwh": 0.0,
        "terminal_soc_value_gbp_per_mwh": 0.0, "maximum_cycles_per_day": 10.0,
        "upward_service_duration_h": 1.0, "downward_service_duration_h": 1.0,
    }
    base.update(overrides)
    return BatteryConfig(**base)


def test_case1_negative_price_no_future_charges_to_max():
    """25 MWh empty, current price negative, no future opportunity -> fill up."""
    cfg = _case_config(initial_soc_mwh=75.0)  # 25 MWh of headroom
    res = optimise(cfg, make_inputs([-50.0, 0.0, 0.0], streams=WHOLESALE_ONLY),
                   compute_marginals=False)
    # It charges the full 25 MWh headroom while paid to do so.
    assert res.periods[0].ending_soc_mwh >= 99.9
    assert res.periods[0].energy_charged_mwh >= 24.9


def test_case2_mild_now_strong_future_preserves_capacity():
    """25 MWh empty, mildly negative now, strongly negative later -> save capacity."""
    cfg = _case_config(initial_soc_mwh=75.0)
    res = optimise(cfg, make_inputs([-5.0, -100.0], streams=WHOLESALE_ONLY),
                   compute_marginals=False)
    # Capacity is preserved for the strongly-negative period.
    assert res.periods[0].charge_mw < 1e-3
    assert res.periods[1].charge_mw > 1e-3


def test_case3_high_future_upward_value_keeps_stored_energy():
    """High later upward-service value -> keep enough energy to reserve up."""
    cfg = _case_config(initial_soc_mwh=40.0, upward_service_duration_h=1.0)
    # Period 0 tempts a discharge (decent price); period 1 pays richly for upward reserve.
    res = optimise(
        cfg,
        make_inputs([70.0, 60.0], up_avail=[0.0, 500.0],
                    streams=RevenueStreams(wholesale=True, upward_availability=True,
                                           downward_availability=False, bm_activation=False,
                                           imbalance=False)),
        compute_marginals=False,
    )
    # It holds energy so it can reserve upward capability in the high-value period.
    assert res.periods[1].upward_reserved_mw > 1e-3
    assert res.periods[1].beginning_soc_mwh >= res.periods[1].upward_reserved_mw - 1e-3


def test_case4_high_future_downward_value_keeps_empty_space():
    """High later downward-service value -> keep empty capacity to reserve down."""
    cfg = _case_config(initial_soc_mwh=60.0, downward_service_duration_h=1.0)
    res = optimise(
        cfg,
        make_inputs([-40.0, -20.0], down_avail=[0.0, 500.0],
                    streams=RevenueStreams(wholesale=True, upward_availability=False,
                                           downward_availability=True, bm_activation=False,
                                           imbalance=False)),
        compute_marginals=False,
    )
    assert res.periods[1].downward_reserved_mw > 1e-3
    # Empty headroom preserved for the reserve.
    headroom = cfg.maximum_soc_mwh - res.periods[1].beginning_soc_mwh
    assert headroom >= res.periods[1].downward_reserved_mw - 1e-3


def test_case5_high_degradation_avoids_marginal_cycling():
    """High degradation cost -> skip marginally profitable cycles.

    Start empty so any discharge requires charging first (a genuine cycle). The
    50->55 spread of 5 GBP/MWh cannot cover a 10 GBP/MWh degradation charge on each
    of charge and discharge.
    """
    cfg = _case_config(initial_soc_mwh=0.0, degradation_cost_gbp_per_mwh_throughput=10.0)
    res = optimise(cfg, make_inputs([50.0, 55.0], streams=WHOLESALE_ONLY),
                   compute_marginals=False)
    throughput = sum(p.energy_charged_mwh + p.energy_discharged_mwh for p in res.periods)
    assert throughput < 1e-2
    # ... and with no degradation the same spread *is* worth cycling.
    cfg2 = cfg.model_copy(update={"degradation_cost_gbp_per_mwh_throughput": 0.0})
    res2 = optimise(cfg2, make_inputs([50.0, 55.0], streams=WHOLESALE_ONLY),
                    compute_marginals=False)
    throughput2 = sum(p.energy_charged_mwh + p.energy_discharged_mwh for p in res2.periods)
    assert throughput2 > 1.0


def test_case6_terminal_soc_constraint_binds():
    """Terminal SoC requirement -> end at or above the required SoC."""
    cfg = _case_config(initial_soc_mwh=50.0, minimum_terminal_soc_mwh=40.0)
    res = optimise(cfg, make_inputs([500.0, 500.0], streams=WHOLESALE_ONLY),
                   compute_marginals=False)
    assert res.periods[-1].ending_soc_mwh >= 40.0 - 1e-3
    # And it is binding (would have gone lower without the constraint).
    assert res.periods[-1].ending_soc_mwh <= 40.0 + 1e-2
