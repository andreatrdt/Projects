"""Forecasting: leakage-safe features, chronological validation, metrics."""

from __future__ import annotations

import numpy as np
from gb_battery.demo.sample_data import generate_synthetic_history
from gb_battery.forecast import (
    BaselineForecaster,
    GradientBoostingQuantileForecaster,
    build_features,
    expanding_window_validate,
)
from gb_battery.forecast.features import FEATURE_COLUMNS
from gb_battery.forecast.metrics import pinball_loss


def test_features_exclude_same_period_outturn():
    # The target/outturn columns must never appear as model features.
    assert "wholesale_price" not in FEATURE_COLUMNS
    assert "system_price" not in FEATURE_COLUMNS


def test_build_features_drops_rows_without_lag_history():
    hist = generate_synthetic_history(days=10)
    feats = build_features(hist)
    # First day(s) lack a previous-day lag and are dropped.
    assert feats["lag1d_price"].notna().all()
    assert len(feats) < len(hist)


def test_pinball_loss_is_zero_for_perfect_forecast():
    y = np.array([10.0, 20.0, 30.0])
    assert pinball_loss(y, y, 0.5) == 0.0


def test_chronological_validation_runs_and_orders_time():
    hist = generate_synthetic_history(days=30)
    rep = expanding_window_validate(
        hist, lambda: BaselineForecaster("lag1d"), "lag1d",
        n_splits=3, min_train_days=10, test_days=3,
    )
    assert len(rep.folds) >= 1
    for f in rep.folds:
        assert f.test_start > f.train_end  # test strictly after train
    assert rep.mean_mae > 0


def test_gbm_forecaster_produces_ordered_quantiles():
    hist = generate_synthetic_history(days=25)
    feats = build_features(hist)
    train_feats = hist  # fit on full history
    fc = GradientBoostingQuantileForecaster(quantiles=(0.1, 0.5, 0.9)).fit(train_feats)
    pred = fc.predict(feats.tail(48))
    q10, q50, q90 = pred.quantile(0.1), pred.quantile(0.5), pred.quantile(0.9)
    assert np.all(q10 <= q50 + 1e-6)
    assert np.all(q50 <= q90 + 1e-6)
