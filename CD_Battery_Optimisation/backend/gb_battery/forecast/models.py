"""Price forecasters: transparent baselines and a gradient-boosted quantile model.

All models share a ``fit(history) -> predict(feature_frame)`` interface and produce
both a point forecast and quantile forecasts (prediction intervals). Validation is
always chronological (see :mod:`gb_battery.forecast.validation`); these classes never
shuffle data themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gb_battery.forecast.features import FEATURE_COLUMNS, build_features, feature_matrix

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class PriceForecast:
    """Forecast output aligned to (settlement_date, settlement_period)."""

    frame: pd.DataFrame  # columns: settlement_date, settlement_period, point, q10, q50, q90...
    quantiles: tuple[float, ...]

    def point(self) -> np.ndarray:
        return self.frame["point"].to_numpy()

    def quantile(self, alpha: float) -> np.ndarray:
        return self.frame[f"q{int(round(alpha * 100)):02d}"].to_numpy()


class BaselineForecaster:
    """Transparent baseline: same-SP lag or rolling median, with a normal interval.

    method:
        * ``lag1d``  — previous day, same settlement period
        * ``lag7d``  — previous week, same settlement period
        * ``roll``   — rolling 7-day median, same settlement period
    """

    def __init__(self, method: str = "roll", quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> None:
        self.method = method
        self.quantiles = quantiles

    def fit(self, history: pd.DataFrame) -> BaselineForecaster:  # noqa: ARG002 - stateless
        return self

    def predict(self, features: pd.DataFrame) -> PriceForecast:
        col = {"lag1d": "lag1d_price", "lag7d": "lag7d_price", "roll": "roll7d_median_price"}[self.method]
        point = features[col].to_numpy(dtype=float)
        sigma = features["roll7d_std_price"].fillna(features["roll7d_std_price"].median()).to_numpy()
        out = pd.DataFrame(
            {
                "settlement_date": features["settlement_date"].to_numpy(),
                "settlement_period": features["settlement_period"].to_numpy(),
                "point": point,
            }
        )
        for q in self.quantiles:
            z = _norm_ppf(q)
            out[f"q{int(round(q * 100)):02d}"] = point + z * sigma
        return PriceForecast(out, self.quantiles)


class GradientBoostingQuantileForecaster:
    """HistGradientBoostingRegressor point + quantile models (no external deps).

    LightGBM is intentionally not required; scikit-learn's histogram gradient boosting
    provides native quantile loss and installs cleanly everywhere.
    """

    def __init__(self, quantiles: tuple[float, ...] = DEFAULT_QUANTILES, random_state: int = 0) -> None:
        self.quantiles = quantiles
        self.random_state = random_state
        self._point: Any = None
        self._q_models: dict[float, Any] = {}

    def fit(self, history: pd.DataFrame) -> GradientBoostingQuantileForecaster:
        from sklearn.ensemble import HistGradientBoostingRegressor

        feats = build_features(history)
        x, y = feature_matrix(feats)
        self._point = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=200, learning_rate=0.06,
            max_depth=None, random_state=self.random_state,
        ).fit(x, y)
        for q in self.quantiles:
            self._q_models[q] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=200, learning_rate=0.06,
                random_state=self.random_state,
            ).fit(x, y)
        return self

    def predict(self, features: pd.DataFrame) -> PriceForecast:
        if self._point is None:
            raise RuntimeError("Forecaster not fitted")
        x = features[FEATURE_COLUMNS].astype(float)
        out = pd.DataFrame(
            {
                "settlement_date": features["settlement_date"].to_numpy(),
                "settlement_period": features["settlement_period"].to_numpy(),
                "point": self._point.predict(x),
            }
        )
        preds = {q: self._q_models[q].predict(x) for q in self.quantiles}
        # Enforce non-crossing quantiles.
        stacked = np.sort(np.vstack([preds[q] for q in sorted(self.quantiles)]), axis=0)
        for i, q in enumerate(sorted(self.quantiles)):
            out[f"q{int(round(q * 100)):02d}"] = stacked[i]
        return PriceForecast(out, self.quantiles)


@dataclass
class FittedForecast:
    """Convenience bundle of a fitted forecaster and the feature frame it predicts on."""

    forecaster: Any
    features: pd.DataFrame = field(repr=False)

    def forecast(self) -> PriceForecast:
        return self.forecaster.predict(self.features)


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (Acklam approximation) — avoids a scipy dep."""
    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
