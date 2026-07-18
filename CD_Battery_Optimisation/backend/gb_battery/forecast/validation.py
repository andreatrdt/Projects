"""Chronological (expanding-window) validation for the price forecasters.

Splits are strictly time-ordered: every test fold lies entirely *after* its training
data. There is no shuffling and no k-fold mixing of future and past. Metrics are
MAE, RMSE and pinball loss for each quantile, reported per fold and in aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gb_battery.forecast.features import TARGET, build_features
from gb_battery.forecast.metrics import mae, pinball_loss, rmse


@dataclass
class FoldResult:
    train_end: str
    test_start: str
    test_end: str
    n_test: int
    mae: float
    rmse: float
    pinball: dict[float, float]


@dataclass
class ValidationReport:
    model: str
    folds: list[FoldResult]

    @property
    def mean_mae(self) -> float:
        return float(np.mean([f.mae for f in self.folds])) if self.folds else float("nan")

    @property
    def mean_rmse(self) -> float:
        return float(np.mean([f.rmse for f in self.folds])) if self.folds else float("nan")

    def mean_pinball(self) -> dict[float, float]:
        if not self.folds:
            return {}
        qs = self.folds[0].pinball.keys()
        return {q: float(np.mean([f.pinball[q] for f in self.folds])) for q in qs}


def expanding_window_validate(
    history: pd.DataFrame,
    forecaster_factory,
    model_name: str,
    *,
    n_splits: int = 4,
    min_train_days: int = 10,
    test_days: int = 3,
) -> ValidationReport:
    """Run expanding-window chronological validation.

    ``forecaster_factory`` is a zero-arg callable returning a fresh forecaster. The
    forecaster is re-fitted on the growing training window for each fold.
    """
    df = history.copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    days = sorted(df["settlement_date"].unique())
    folds: list[FoldResult] = []

    # Place folds at the tail of the timeline.
    total_test = n_splits * test_days
    start_idx = max(min_train_days, len(days) - total_test)

    for k in range(n_splits):
        test_lo = start_idx + k * test_days
        test_hi = test_lo + test_days
        if test_hi > len(days):
            break
        train_days = days[:test_lo]
        test_window = days[test_lo:test_hi]
        if len(train_days) < min_train_days:
            continue

        train = df[df["settlement_date"].isin(train_days)]
        # Build features over train+test so lag features for the test fold exist,
        # then predict only on the test rows (no target leakage: features use only
        # lagged outturns strictly before each row's date).
        combined = df[df["settlement_date"].isin(train_days + list(test_window))]
        feats = build_features(combined)
        test_feats = feats[feats["settlement_date"].isin(test_window)]
        if test_feats.empty:
            continue

        fc = forecaster_factory()
        fc.fit(train)
        pred = fc.predict(test_feats)

        y_true = test_feats[TARGET].to_numpy()
        y_point = pred.point()
        pb = {q: pinball_loss(y_true, pred.quantile(q), q) for q in pred.quantiles}
        folds.append(
            FoldResult(
                train_end=str(train_days[-1].date()),
                test_start=str(test_window[0].date()),
                test_end=str(test_window[-1].date()),
                n_test=len(y_true),
                mae=mae(y_true, y_point),
                rmse=rmse(y_true, y_point),
                pinball=pb,
            )
        )
    return ValidationReport(model_name, folds)
