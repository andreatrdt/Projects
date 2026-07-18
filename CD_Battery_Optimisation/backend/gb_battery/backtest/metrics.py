"""Backtest performance metrics."""

from __future__ import annotations

import numpy as np


def max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """Largest peak-to-trough drop in cumulative P&L (GBP, reported as a positive)."""
    c = np.asarray(cumulative_pnl, dtype=float)
    if c.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(c)
    drawdowns = running_max - c
    return float(np.max(drawdowns))


def cvar(pnl_per_step: np.ndarray, alpha: float = 0.95) -> float:
    """Conditional Value-at-Risk (expected shortfall) of step P&L at level ``alpha``.

    Returns the mean of the worst ``(1-alpha)`` tail of step P&Ls (a loss is negative,
    so a more-negative number is worse).
    """
    x = np.asarray(pnl_per_step, dtype=float)
    if x.size == 0:
        return 0.0
    q = np.quantile(x, 1 - alpha)
    tail = x[x <= q]
    return float(tail.mean()) if tail.size else float(q)


def value_at_risk(pnl_per_step: np.ndarray, alpha: float = 0.95) -> float:
    x = np.asarray(pnl_per_step, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, 1 - alpha))
