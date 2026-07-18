"""Feature engineering for day-ahead price forecasting — leakage-safe by design.

A forecast for settlement date *D* is made the day before (day-ahead). Therefore the
only permissible inputs are:

* calendar variables (known arbitrarily far ahead);
* day-ahead **forecasts** of demand / wind / solar (published before D);
* **lagged outturn** prices from strictly earlier settlement dates (D-1, D-7, ...).

Same-day outturn prices are never used as features. :mod:`gb_battery.backtest.leakage`
independently audits that every feature's information was available at decision time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "wholesale_price"

FEATURE_COLUMNS = [
    "settlement_period",
    "sp_sin",
    "sp_cos",
    "day_of_week",
    "is_weekend",
    "month",
    "demand_forecast_mw",
    "wind_forecast_mw",
    "solar_forecast_mw",
    "residual_forecast_mw",
    "lag1d_price",
    "lag7d_price",
    "roll7d_median_price",
    "roll7d_std_price",
]


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame (one row per settlement date & period).

    ``history`` must contain: ``settlement_date``, ``settlement_period``,
    ``wholesale_price`` (outturn), ``demand_forecast_mw``, ``wind_forecast_mw``,
    ``solar_forecast_mw``. Rows without sufficient lag history are dropped.
    """
    df = history.copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

    # Calendar.
    n_by_day = df.groupby("settlement_date")["settlement_period"].transform("max")
    df["sp_sin"] = np.sin(2 * np.pi * df["settlement_period"] / n_by_day)
    df["sp_cos"] = np.cos(2 * np.pi * df["settlement_period"] / n_by_day)
    df["day_of_week"] = df["settlement_date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["settlement_date"].dt.month

    df["residual_forecast_mw"] = (
        df["demand_forecast_mw"].fillna(0)
        - df["wind_forecast_mw"].fillna(0)
        - df["solar_forecast_mw"].fillna(0)
    )

    # Lagged OUTTURN prices, computed within each settlement period across days.
    df = df.sort_values(["settlement_period", "settlement_date"])
    grp = df.groupby("settlement_period", group_keys=False)
    df["lag1d_price"] = grp[TARGET].shift(1)
    df["lag7d_price"] = grp[TARGET].shift(7)
    # Rolling stats over the *previous* 7 same-SP days (shifted to exclude today).
    df["roll7d_median_price"] = grp[TARGET].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).median()
    )
    df["roll7d_std_price"] = grp[TARGET].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).std()
    )

    df = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)
    df = df.dropna(subset=["lag1d_price", "roll7d_median_price"])
    return df


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature frame into (X, y)."""
    x = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET].astype(float)
    return x, y
