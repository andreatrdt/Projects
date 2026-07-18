"""Chronological, leakage-audited backtesting with benchmark strategies."""

from gb_battery.backtest.engine import BacktestResult, run_backtest
from gb_battery.backtest.runner import BENCHMARK_ORDER, compare_strategies

__all__ = ["run_backtest", "BacktestResult", "compare_strategies", "BENCHMARK_ORDER"]
