"""Backtest engine: benchmark ordering, leakage audit, SoC feasibility."""

from __future__ import annotations

from gb_battery.backtest import compare_strategies, run_backtest
from gb_battery.battery.config import BatteryConfig
from gb_battery.demo.sample_data import generate_synthetic_history


def _cfg() -> BatteryConfig:
    return BatteryConfig(degradation_cost_gbp_per_mwh_throughput=2.0, minimum_terminal_soc_mwh=10.0)


def test_backtest_soc_feasible_every_period():
    hist = generate_synthetic_history(days=12)
    res = run_backtest(_cfg(), hist, "deterministic_optimiser")
    cfg = _cfg()
    for _, r in res.ledger.iterrows():
        assert cfg.effective_min_soc - 1e-3 <= r["ending_soc_mwh"] <= cfg.effective_max_soc + 1e-3


def test_benchmark_ordering_and_upper_bound():
    hist = generate_synthetic_history(days=20)
    cmp = compare_strategies(_cfg(), hist)
    by = {row["strategy"]: row["total_pnl_gbp"] for row in cmp["table"]}
    # No-op earns ~0; optimiser beats naive rules; perfect foresight is the upper bound.
    assert abs(by["no_operation"]) < 1e-6
    assert by["deterministic_optimiser"] >= by["threshold_rule"] - 1e-6
    assert by["perfect_foresight"] >= by["deterministic_optimiser"] - 1e-6


def test_backtest_leakage_audit_passes():
    hist = generate_synthetic_history(days=15)
    cmp = compare_strategies(_cfg(), hist, strategies=["deterministic_optimiser", "perfect_foresight"])
    audit = {a["check"]: a["ok"] for a in cmp["leakage_audit"]}
    assert audit.get("forecast_not_outturn") is True


def test_no_operation_has_zero_throughput():
    hist = generate_synthetic_history(days=8)
    res = run_backtest(_cfg(), hist, "no_operation")
    assert res.summary["full_cycle_equivalents"] == 0.0
