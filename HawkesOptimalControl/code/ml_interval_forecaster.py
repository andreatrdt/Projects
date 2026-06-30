from __future__ import annotations

"""Probabilistic next-price forecasting for the Hawkes optimal-control project.

This module adds a third pillar to the project:

1. Hawkes order-flow modelling.
2. HJB / optimal market-making control.
3. ML-based short-horizon predictive intervals for the next mid-price.

The model is deliberately formulated as probabilistic forecasting rather than
point prediction. The target is the future mid-price change

    y_t = S_{t+h} - S_t,

and the model estimates conditional quantiles of y_t. Adding those quantiles to
the current mid-price S_t gives predictive price bands.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_pinball_loss
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    HistGradientBoostingRegressor = None
    RandomForestRegressor = None
    MLPRegressor = None
    Pipeline = None
    StandardScaler = None
    mean_absolute_error = None
    mean_pinball_loss = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


@dataclass(frozen=True)
class IntervalForecasterConfig:
    """Configuration for short-horizon price interval forecasting.

    Parameters
    ----------
    horizon_steps:
        Number of rows/grid points ahead to forecast.
    lookback:
        Number of lagged observations used to create autoregressive and
        order-flow features.
    quantiles:
        Quantiles of the future mid-price change to estimate.
    test_fraction:
        Fraction of the final time-ordered sample used as out-of-sample test.
    model_type:
        One of ``"gbrt_quantile"``, ``"rf_quantile"`` or ``"mlp_quantile"``.
        ``gbrt_quantile`` is the default and should be the main baseline.
    conformal_alpha:
        Optional conformal calibration level. If set to 0.10, the lower/upper
        interval is widened on a calibration split to target approximately
        90% marginal coverage.
    calibration_fraction:
        Fraction of the training sample reserved for conformal calibration.
    """

    horizon_steps: int = 50
    lookback: int = 20
    quantiles: Tuple[float, ...] = (0.05, 0.50, 0.95)
    test_fraction: float = 0.25
    model_type: str = "gbrt_quantile"
    conformal_alpha: Optional[float] = None
    calibration_fraction: float = 0.20
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 30
    l2_regularization: float = 0.0
    random_state: int = 42

    def validate(self) -> None:
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        if not (0.05 <= self.test_fraction <= 0.5):
            raise ValueError("test_fraction should be in [0.05, 0.5]")
        if not (0.05 <= self.calibration_fraction <= 0.5):
            raise ValueError("calibration_fraction should be in [0.05, 0.5]")
        if len(self.quantiles) < 2:
            raise ValueError("provide at least two quantiles")
        if tuple(sorted(self.quantiles)) != tuple(self.quantiles):
            raise ValueError("quantiles must be sorted increasingly")
        if any(q <= 0.0 or q >= 1.0 for q in self.quantiles):
            raise ValueError("all quantiles must be in (0, 1)")
        if self.model_type not in {"gbrt_quantile", "rf_quantile", "mlp_quantile"}:
            raise ValueError("model_type must be one of: gbrt_quantile, rf_quantile, mlp_quantile")
        if self.conformal_alpha is not None and not (0.01 < self.conformal_alpha < 0.5):
            raise ValueError("conformal_alpha must be None or in (0.01, 0.5)")


def _require_sklearn() -> None:
    if HistGradientBoostingRegressor is None:
        raise ImportError(
            "scikit-learn is required for ml_interval_forecaster.py. "
            "Install it with `pip install scikit-learn`."
        ) from _SKLEARN_IMPORT_ERROR


def _as_length(x: Optional[np.ndarray], n: int, fill_value: float = np.nan) -> np.ndarray:
    if x is None:
        return np.full(n, fill_value, dtype=float)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size == n - 1:
        return np.r_[arr, fill_value]
    if arr.size > n:
        return arr[:n]
    return np.r_[arr, np.full(n - arr.size, fill_value, dtype=float)]


def _safe_divide(num: pd.Series, den: pd.Series, eps: float = 1e-12) -> pd.Series:
    return num / (den.abs() + eps)


def build_backtest_feature_frame(res: Dict[str, np.ndarray], cfg: IntervalForecasterConfig) -> pd.DataFrame:
    """Build a leak-safe supervised learning frame from backtest output.

    The returned frame contains only features observable at time t and two
    target columns:

    - ``target_delta`` = mid[t + horizon] - mid[t]
    - ``target_price`` = mid[t + horizon]
    """

    cfg.validate()

    required = {"time", "mid", "bid", "ask"}
    missing = required - set(res.keys())
    if missing:
        raise KeyError(f"Backtest result is missing required keys: {sorted(missing)}")

    mid = np.asarray(res["mid"], dtype=float).reshape(-1)
    n = mid.size
    if n <= cfg.horizon_steps + cfg.lookback + 10:
        raise ValueError("Not enough observations for the requested lookback/horizon.")

    df = pd.DataFrame(
        {
            "time": _as_length(res.get("time"), n),
            "mid": mid,
            "bid": _as_length(res.get("bid"), n),
            "ask": _as_length(res.get("ask"), n),
            "q": _as_length(res.get("q"), n, fill_value=0.0),
            "pnl": _as_length(res.get("pnl"), n, fill_value=0.0),
            "lam_plus": _as_length(res.get("lam_plus"), n, fill_value=0.0),
            "lam_minus": _as_length(res.get("lam_minus"), n, fill_value=0.0),
            "f_a": _as_length(res.get("f_a"), n, fill_value=0.0),
            "f_b": _as_length(res.get("f_b"), n, fill_value=0.0),
            "H_a": _as_length(res.get("H_a"), n, fill_value=0.0),
            "H_b": _as_length(res.get("H_b"), n, fill_value=0.0),
        }
    )

    # Contemporaneous state variables available at decision time t.
    df["quoted_spread"] = df["ask"] - df["bid"]
    df["half_spread"] = 0.5 * df["quoted_spread"]
    df["mid_to_bid"] = df["mid"] - df["bid"]
    df["ask_to_mid"] = df["ask"] - df["mid"]
    df["dmid"] = df["mid"].diff()
    df["abs_dmid"] = df["dmid"].abs()
    df["ret"] = _safe_divide(df["dmid"], df["mid"].shift(1))
    df["abs_ret"] = df["ret"].abs()

    lam_sum = df["lam_plus"] + df["lam_minus"]
    df["flow_imbalance"] = (df["lam_plus"] - df["lam_minus"]) / (lam_sum + 1e-12)
    df["log_lam_plus"] = np.log1p(np.maximum(df["lam_plus"], 0.0))
    df["log_lam_minus"] = np.log1p(np.maximum(df["lam_minus"], 0.0))
    df["log_lam_sum"] = np.log1p(np.maximum(lam_sum, 0.0))
    df["fill_prob_imbalance"] = df["f_b"] - df["f_a"]
    df["net_fills"] = df["H_b"] - df["H_a"]
    df["inventory_abs"] = df["q"].abs()

    # Lagged features. These are leak-safe because all shifts are positive.
    for lag in range(1, cfg.lookback + 1):
        df[f"dmid_lag_{lag}"] = df["dmid"].shift(lag)
        df[f"abs_dmid_lag_{lag}"] = df["abs_dmid"].shift(lag)
        df[f"flow_imbalance_lag_{lag}"] = df["flow_imbalance"].shift(lag)
        df[f"ret_lag_{lag}"] = df["ret"].shift(lag)

    # Rolling features use information up to and including time t only.
    for window in (5, 10, 20, 50):
        if window <= max(5, cfg.lookback * 3):
            df[f"dmid_sum_{window}"] = df["dmid"].rolling(window).sum()
            df[f"realized_vol_{window}"] = df["dmid"].rolling(window).std()
            df[f"flow_imbalance_mean_{window}"] = df["flow_imbalance"].rolling(window).mean()
            df[f"fill_imbalance_sum_{window}"] = df["net_fills"].rolling(window).sum()

    df["target_price"] = df["mid"].shift(-cfg.horizon_steps)
    df["target_delta"] = df["target_price"] - df["mid"]

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model features, excluding target and timestamp columns."""

    excluded = {"time", "target_price", "target_delta"}
    return [c for c in frame.columns if c not in excluded]


class QuantilePriceIntervalForecaster:
    """Estimate conditional quantiles of the future mid-price change.

    Implementation choices:

    - ``gbrt_quantile`` trains one gradient-boosting quantile model per quantile.
    - ``rf_quantile`` trains one random forest and extracts empirical quantiles
      across trees. This is a useful non-parametric robustness check.
    - ``mlp_quantile`` trains one neural network per quantile with squared loss.
      It is not a true quantile-loss neural network, so treat it as exploratory.
    """

    def __init__(self, cfg: IntervalForecasterConfig):
        _require_sklearn()
        cfg.validate()
        self.cfg = cfg
        self.models: Dict[float | str, object] = {}
        self.feature_names_: Optional[list[str]] = None
        self.conformal_padding_: Optional[float] = None

    def _new_gbrt_model(self, q: float) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(q),
            max_iter=self.cfg.max_iter,
            learning_rate=self.cfg.learning_rate,
            max_leaf_nodes=self.cfg.max_leaf_nodes,
            min_samples_leaf=self.cfg.min_samples_leaf,
            l2_regularization=self.cfg.l2_regularization,
            random_state=self.cfg.random_state,
        )

    def _new_rf_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=max(2, self.cfg.min_samples_leaf // 5),
            n_jobs=-1,
            random_state=self.cfg.random_state,
        )

    def _new_mlp_model(self, seed_offset: int = 0) -> Pipeline:
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=max(1e-6, self.cfg.l2_regularization),
                        learning_rate_init=1e-3,
                        max_iter=max(300, self.cfg.max_iter),
                        random_state=self.cfg.random_state + seed_offset,
                        early_stopping=True,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y_delta: pd.Series | np.ndarray) -> "QuantilePriceIntervalForecaster":
        X = pd.DataFrame(X).copy()
        y = np.asarray(y_delta, dtype=float).reshape(-1)
        if X.shape[0] != y.size:
            raise ValueError("X and y_delta have incompatible lengths")

        self.feature_names_ = list(X.columns)
        self.models = {}
        self.conformal_padding_ = None

        if self.cfg.model_type == "gbrt_quantile":
            for q in self.cfg.quantiles:
                model = self._new_gbrt_model(q)
                model.fit(X, y)
                self.models[float(q)] = model

        elif self.cfg.model_type == "rf_quantile":
            model = self._new_rf_model()
            model.fit(X, y)
            self.models["rf"] = model

        elif self.cfg.model_type == "mlp_quantile":
            # Approximate quantile architecture: train one independent MLP per
            # quantile target generated from empirical residual shifts. This is
            # intentionally marked as exploratory in the notebook.
            center_model = self._new_mlp_model(seed_offset=0)
            center_model.fit(X, y)
            residuals = y - center_model.predict(X)
            self.models["center"] = center_model
            for j, q in enumerate(self.cfg.quantiles):
                shifted_target = y + np.quantile(residuals, q)
                model = self._new_mlp_model(seed_offset=j + 1)
                model.fit(X, shifted_target)
                self.models[float(q)] = model

        return self

    def predict_delta_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.models or self.feature_names_ is None:
            raise RuntimeError("Model is not fitted yet")

        X = pd.DataFrame(X).copy()[self.feature_names_]

        if self.cfg.model_type == "gbrt_quantile":
            raw = np.column_stack([self.models[float(q)].predict(X) for q in self.cfg.quantiles])

        elif self.cfg.model_type == "rf_quantile":
            model = self.models["rf"]
            tree_preds = np.column_stack([tree.predict(X) for tree in model.estimators_])
            raw = np.quantile(tree_preds, self.cfg.quantiles, axis=1).T

        elif self.cfg.model_type == "mlp_quantile":
            raw = np.column_stack([self.models[float(q)].predict(X) for q in self.cfg.quantiles])

        else:  # pragma: no cover
            raise RuntimeError(f"Unknown model_type={self.cfg.model_type}")

        # Avoid quantile crossing by enforcing monotonicity row-by-row.
        raw = np.maximum.accumulate(raw, axis=1)

        if self.conformal_padding_ is not None and raw.shape[1] >= 2:
            raw[:, 0] -= self.conformal_padding_
            raw[:, -1] += self.conformal_padding_

        cols = [f"q{int(q * 100):02d}_delta" for q in self.cfg.quantiles]
        return pd.DataFrame(raw, columns=cols, index=X.index)

    def predict_price_quantiles(self, X: pd.DataFrame, mid_now: Sequence[float]) -> pd.DataFrame:
        delta_q = self.predict_delta_quantiles(X)
        mid_now = np.asarray(mid_now, dtype=float).reshape(-1)
        if mid_now.size != delta_q.shape[0]:
            raise ValueError("mid_now must have the same length as X")

        out = delta_q.copy()
        for col in delta_q.columns:
            out[col.replace("_delta", "_price")] = mid_now + delta_q[col].to_numpy()
        return out

    def calibrate_conformal(self, X_cal: pd.DataFrame, y_cal_delta: Sequence[float]) -> float:
        """Widen lower/upper bands using split conformal calibration.

        The conformity score is max(lower - y, y - upper, 0). The returned
        padding is added symmetrically to the lower and upper quantile forecasts.
        """

        if self.cfg.conformal_alpha is None:
            self.conformal_padding_ = None
            return 0.0

        pred = self.predict_delta_quantiles(X_cal).to_numpy(dtype=float)
        y = np.asarray(y_cal_delta, dtype=float).reshape(-1)
        lower = pred[:, 0]
        upper = pred[:, -1]
        scores = np.maximum.reduce([lower - y, y - upper, np.zeros_like(y)])
        level = min(1.0, (1.0 - self.cfg.conformal_alpha) * (len(scores) + 1) / len(scores))
        padding = float(np.quantile(scores, level, method="higher"))
        self.conformal_padding_ = padding
        return padding


def train_cal_test_split_time_ordered(
    frame: pd.DataFrame, cfg: IntervalForecasterConfig
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Split data in chronological order into train/calibration/test."""

    n = len(frame)
    n_test = max(int(round(n * cfg.test_fraction)), 1)
    train_cal = frame.iloc[: n - n_test].copy()
    test = frame.iloc[n - n_test :].copy()

    cal = None
    train = train_cal
    if cfg.conformal_alpha is not None:
        n_cal = max(int(round(len(train_cal) * cfg.calibration_fraction)), 1)
        train = train_cal.iloc[: len(train_cal) - n_cal].copy()
        cal = train_cal.iloc[len(train_cal) - n_cal :].copy()

    if len(train) < 100 or len(test) < 20:
        raise ValueError("Too few samples after time split. Lower horizon/lookback or use more data.")
    if cal is not None and len(cal) < 20:
        raise ValueError("Too few calibration samples for conformal intervals.")

    return train, cal, test


def winkler_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """Mean Winkler interval score for nominal coverage 1-alpha."""

    width = upper - lower
    below = y < lower
    above = y > upper
    score = width.copy()
    score[below] += (2.0 / alpha) * (lower[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - upper[above])
    return float(np.mean(score))


def evaluate_interval_predictions(
    y_true_delta: Sequence[float],
    pred_delta: pd.DataFrame,
    quantiles: Tuple[float, ...],
) -> Dict[str, float]:
    """Evaluate probabilistic forecasts out of sample."""

    y = np.asarray(y_true_delta, dtype=float).reshape(-1)
    pred = pred_delta.to_numpy(dtype=float)
    metrics: Dict[str, float] = {}

    for j, q in enumerate(quantiles):
        metrics[f"pinball_q{int(q * 100):02d}"] = float(mean_pinball_loss(y, pred[:, j], alpha=q))

    lower = pred[:, 0]
    upper = pred[:, -1]
    median_idx = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    median = pred[:, median_idx]
    nominal_coverage = float(quantiles[-1] - quantiles[0])
    alpha = max(1e-6, 1.0 - nominal_coverage)

    metrics["nominal_coverage"] = nominal_coverage
    metrics["empirical_coverage"] = float(np.mean((y >= lower) & (y <= upper)))
    metrics["coverage_error"] = metrics["empirical_coverage"] - nominal_coverage
    metrics["avg_interval_width"] = float(np.mean(upper - lower))
    metrics["winkler_score"] = winkler_score(y, lower, upper, alpha=alpha)
    metrics["median_mae"] = float(mean_absolute_error(y, median))
    metrics["directional_accuracy"] = float(np.mean(np.sign(y) == np.sign(median)))
    return metrics


def fit_price_interval_model_from_backtest(
    res: Dict[str, np.ndarray],
    cfg: Optional[IntervalForecasterConfig] = None,
) -> Dict[str, object]:
    """End-to-end helper used by the main notebook."""

    cfg = cfg or IntervalForecasterConfig()
    frame = build_backtest_feature_frame(res, cfg)
    train_frame, cal_frame, test_frame = train_cal_test_split_time_ordered(frame, cfg)
    cols = feature_columns(frame)

    model = QuantilePriceIntervalForecaster(cfg)
    model.fit(train_frame[cols], train_frame["target_delta"])

    conformal_padding = None
    if cal_frame is not None:
        conformal_padding = model.calibrate_conformal(cal_frame[cols], cal_frame["target_delta"])

    pred_delta = model.predict_delta_quantiles(test_frame[cols])
    pred_price = model.predict_price_quantiles(test_frame[cols], test_frame["mid"])
    metrics = evaluate_interval_predictions(test_frame["target_delta"], pred_delta, cfg.quantiles)

    return {
        "config": cfg,
        "config_dict": asdict(cfg),
        "model": model,
        "frame": frame,
        "train_frame": train_frame,
        "calibration_frame": cal_frame,
        "test_frame": test_frame,
        "features": cols,
        "pred_delta": pred_delta,
        "pred_price": pred_price,
        "metrics": metrics,
        "conformal_padding": conformal_padding,
    }


def compare_interval_models(
    res: Dict[str, np.ndarray],
    base_cfg: Optional[IntervalForecasterConfig] = None,
    model_types: Sequence[str] = ("gbrt_quantile", "rf_quantile"),
) -> pd.DataFrame:
    """Train several interval models and return an out-of-sample comparison."""

    base_cfg = base_cfg or IntervalForecasterConfig()
    rows = []
    for model_type in model_types:
        cfg = IntervalForecasterConfig(**{**asdict(base_cfg), "model_type": model_type})
        result = fit_price_interval_model_from_backtest(res, cfg)
        rows.append({"model_type": model_type, **result["metrics"]})
    return pd.DataFrame(rows).sort_values(["winkler_score", "avg_interval_width"]).reset_index(drop=True)


def plot_price_intervals(result: Dict[str, object], max_points: int = 500, title: Optional[str] = None):
    """Plot true future price, median prediction and predictive interval."""

    test_frame = result["test_frame"]
    pred_price = result["pred_price"]
    cfg = result["config"]

    n = min(max_points, len(test_frame))
    tf = test_frame.iloc[:n]
    pp = pred_price.iloc[:n]

    q_cols = [f"q{int(q * 100):02d}_price" for q in cfg.quantiles]
    lower_col = q_cols[0]
    upper_col = q_cols[-1]
    median_idx = int(np.argmin(np.abs(np.asarray(cfg.quantiles) - 0.5)))
    median_col = q_cols[median_idx]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = tf["time"].to_numpy()

    ax.plot(x, tf["mid"].to_numpy(), label="current mid", linewidth=1.0, alpha=0.6)
    ax.plot(x, tf["target_price"].to_numpy(), label=f"realized mid t+{cfg.horizon_steps}", linewidth=1.3)
    ax.plot(x, pp[median_col].to_numpy(), label="predicted median", linewidth=1.3)
    ax.fill_between(
        x,
        pp[lower_col].to_numpy(),
        pp[upper_col].to_numpy(),
        alpha=0.20,
        label=f"{int((cfg.quantiles[-1] - cfg.quantiles[0]) * 100)}% predictive interval",
    )
    ax.set_title(title or "Next-price ML predictive interval")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_interval_residuals(result: Dict[str, object]):
    """Plot realized future price changes against predicted interval errors."""

    test_frame = result["test_frame"]
    pred_delta = result["pred_delta"]
    cfg = result["config"]

    q_cols = [f"q{int(q * 100):02d}_delta" for q in cfg.quantiles]
    lower = pred_delta[q_cols[0]].to_numpy()
    upper = pred_delta[q_cols[-1]].to_numpy()
    y = test_frame["target_delta"].to_numpy()
    inside = (y >= lower) & (y <= upper)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(test_frame["time"].to_numpy(), y, label="realized future delta", linewidth=1.0)
    ax.fill_between(test_frame["time"].to_numpy(), lower, upper, alpha=0.20, label="predicted delta interval")
    ax.scatter(test_frame["time"].to_numpy()[~inside], y[~inside], s=10, label="outside interval")
    ax.set_title("Future price-change interval diagnostics")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Future mid-price change")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig, ax
