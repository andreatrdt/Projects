"""Run and compare all backtest benchmark strategies."""

from __future__ import annotations

import pandas as pd

from gb_battery.backtest.engine import BacktestResult, run_backtest
from gb_battery.backtest.leakage import audit_backtest_alignment
from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.inputs import RevenueStreams

BENCHMARK_ORDER = [
    "no_operation",
    "threshold_rule",
    "fixed_percentile",
    "deterministic_optimiser",
    "perfect_foresight",
]


def compare_strategies(
    config: BatteryConfig,
    history: pd.DataFrame,
    *,
    strategies: list[str] | None = None,
    streams: RevenueStreams | None = None,
    up_availability_price: float = 0.0,
    down_availability_price: float = 0.0,
) -> dict:
    """Run each strategy and return a comparison table + leakage audit.

    The perfect-foresight P&L is the upper bound; each strategy's capture ratio is
    reported against it.
    """
    strategies = strategies or BENCHMARK_ORDER
    results: dict[str, BacktestResult] = {}
    for s in strategies:
        results[s] = run_backtest(
            config, history, s, streams=streams,
            up_availability_price=up_availability_price,
            down_availability_price=down_availability_price,
        )

    upper = results.get("perfect_foresight")
    upper_pnl = upper.summary.get("total_pnl_gbp") if upper else None

    table = []
    for s in strategies:
        summ = results[s].summary
        row = {"strategy": s, **summ}
        if upper_pnl and upper_pnl != 0 and s != "perfect_foresight":
            row["capture_of_perfect_pct"] = round(100.0 * summ.get("total_pnl_gbp", 0) / upper_pnl, 1)
        results[s] = results[s]
        table.append(row)

    # Leakage audit on the deterministic optimiser's ledger (representative).
    audit = []
    if "deterministic_optimiser" in results:
        audit = [f.__dict__ for f in audit_backtest_alignment(results["deterministic_optimiser"].ledger)]

    return {
        "table": table,
        "perfect_foresight_pnl_gbp": upper_pnl,
        "leakage_audit": audit,
        "results": results,
    }
