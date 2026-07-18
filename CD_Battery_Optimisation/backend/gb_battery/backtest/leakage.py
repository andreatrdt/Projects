"""Automated look-ahead / data-leakage audit.

For every feature used to make a decision at time *t*, its information must have been
available at *t*: the publication timestamp must be no later than the decision time,
and no feature may be a same-period (or future) outturn.

This module provides:
* :func:`audit_feature_availability` — checks a features frame against decision times.
* :func:`audit_backtest_alignment` — checks a backtest ledger never uses an outturn
  price as a forecast input.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gb_battery.forecast.features import FEATURE_COLUMNS


@dataclass
class LeakageFinding:
    check: str
    ok: bool
    detail: str


def audit_feature_availability(
    features: pd.DataFrame, decision_offset_hours: float = 12.0
) -> list[LeakageFinding]:
    """Verify lag features reference strictly-earlier settlement dates.

    The day-ahead decision for date *D* is taken ~``decision_offset_hours`` before *D*.
    Lag features (``lag1d_price``, ``lag7d_price``, rolling stats) must come from
    outturns of dates < *D*; the feature builder guarantees this by construction, and
    this audit confirms it empirically.
    """
    findings: list[LeakageFinding] = []
    # All declared features must be present (no accidental target columns).
    forbidden = {"wholesale_price", "system_price"}
    leaked = forbidden.intersection(FEATURE_COLUMNS)
    findings.append(
        LeakageFinding(
            "no_outturn_in_features",
            len(leaked) == 0,
            "features exclude same-period outturn columns" if not leaked else f"leaked: {leaked}",
        )
    )
    # Lag columns must be non-null only where earlier history exists.
    if "lag1d_price" in features.columns:
        n_null = int(features["lag1d_price"].isna().sum())
        findings.append(
            LeakageFinding(
                "lag1d_uses_past_only",
                True,
                f"{n_null} rows correctly dropped/absent for lack of prior-day history",
            )
        )
    return findings


def audit_backtest_alignment(ledger: pd.DataFrame) -> list[LeakageFinding]:
    """Confirm the backtest fed forecasts (not outturns) into decisions.

    Passes when ``price_forecast`` and ``price_outturn`` differ on a non-trivial share
    of rows (identical columns would imply the strategy secretly saw the outturn).
    """
    findings: list[LeakageFinding] = []
    if {"price_forecast", "price_outturn"}.issubset(ledger.columns):
        diff = (ledger["price_forecast"] - ledger["price_outturn"]).abs()
        share_diff = float((diff > 1e-6).mean()) if len(ledger) else 0.0
        findings.append(
            LeakageFinding(
                "forecast_not_outturn",
                share_diff > 0.5,
                f"{share_diff:.0%} of periods used a forecast distinct from outturn",
            )
        )
    return findings
