from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_pinball_loss
except ImportError as exc:  # pragma: no cover
    HistGradientBoostingRegressor = None
    mean_pinball_loss = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


@dataclass(frozen=True)
class IntervalForecasterConfig:
    """Configuration for next-price probabilistic forecasting.

    The target is the future mid-price change:
        y_t = mid_{t+horizon_steps} - mid_t

    Predicted quantiles of y_t are then shifted by the current mid to obtain
    price confidence bands.
    """

    horizon_steps: int = 50
    lookback: int = 20
    quantiles: Tuple[float, ...] = (0.05, 0.50, 0.95)
    test_fraction: float = 0.25
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
        if len(self.quantiles) < 2:
            raise ValueError("provide at least two quantiles")
        if tuple(sorted(self.quantiles)) != tuple(self.quantiles):
            raise ValueError("quantiles must be sorted increasingly")
        if any(q <= 0.0 or q >= 1.0 for q in self.quantiles):
            raise ValueError("all quantiles must be in (0, 1)")


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


def build_backtest_feature_frame(res: Dict[str, np.ndarray], cfg: IntervalForecasterConfig) -> pd.DataFrame:
    """Build a supervised learning frame from the control backtest output.

    Parameters
    ----------
    res:
        Output dictionary returned by ControlBacktester.run().
    cfg:
        Forecasting configuration.

    Returns
    -------
    pd.DataFrame
        Feature frame with target columns:
        - target_delta: future mid change
        - target_price: future mid level
    """

    cfg.validate()

    required = {"time", "mid", "bid", "ask"}
    missing = required - set(res.keys())
    if missing:
        raise KeyError(f"Backtest result is missing required keys: {sorted(missing)}")

    mid = np.asarray(res["mid"], dtype=float).reshape(-1)
    n = mid.size
    if n <= cfg.horizon_steps + cfg.lookback + 5:
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
            "f_a": _as_length(res.get("f_a"), n),
            "f_b": _as_length(res.get("f_b"), n),
            "H_a": _as_length(res.get("H_a"), n, fill_value=0.0),
            "H_b": _as_length(res.get("H_b"), n, fill_value=0.0),
        }
    )

    df["quoted_spread"] = df["ask"] - df["bid"]
    df["mid_to_bid"] = df["mid"] - df["bid"]
    df["ask_to_mid"] = df["ask"] - df["mid"]
    df["dmid"] = df["mid"].diff()
    df["abs_dmid"] = df["dmid"].abs()

    lam_sum = df["lam_plus"] + df["lam_minus"]
    df["flow_imbalance"] = (df["lam_plus"] - df["lam_minus"]) / (lam_sum + 1e-12)
    df["log_lam_plus"] = np.log1p(np.maximum(df["lam_plus"], 0.0))
    df["log_lam_minus"] = np.log1p(np.maximum(df["lam_minus"], 0.0))
    df["net_fills"] = df["H_b"] - df["H_a"]

    for lag in range(1, cfg.lookback + 1):
        df[f"dmid_lag_{lag}"] = df["dmid"].shift(lag)
        df[f"abs_dmid_lag_{lag}"] = df["abs_dmid"].shift(lag)
        df[f"flow_imbalance_lag_{lag}"] = df["flow_imbalance"].shift(lag)

    for window in (5, 10, 20, 50):
        if window <= cfg.lookback * 3:
            df[f"dmid_sum_{window}"] = df["dmid"].rolling(window).sum()
            df[f"realized_vol_{window}"] = df["dmid"].rolling(window).std()
            df[f"fill_imbalance_sum_{window}"] = df["net_fills"].rolling(window).sum()

    df["target_price"] = df["mid"].shift(-cfg.horizon_steps)
    df["target_delta"] = df["target_price"] - df["mid"]

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"time", "target_price", "target_delta"}
    return [c for c in frame.columns if c not in excluded]


class QuantilePriceIntervalForecaster:
    """Train one gradient-boosting quantile regressor per quantile."""

    def __init__(self, cfg: IntervalForecasterConfig):
        _require_sklearn()
        cfg.validate()
        self.cfg = cfg
        self.models: Dict[float, HistGradientBoostingRegressor] = {}
        self.feature_names_: Optional[list[str]] = None

    def _new_model(self, q: float) -> HistGradientBoostingRegressor:
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

    def fit(self, X: pd.DataFrame, y_delta: pd.Series | np.ndarray) -> "QuantilePriceIntervalForecaster":
        X = pd.DataFrame(X).copy()
        y = np.asarray(y_delta, dtype=float).reshape(-1)
        if X.shape[0] != y.size:
            raise ValueError("X and y_delta have incompatible lengths")
        self.feature_names_ = list(X.columns)
        self.models = {}
        for q in self.cfg.quantiles:
            model = self._new_model(q)
            model.fit(X, y)
            self.models[float(q)] = model
        return self

    def predict_delta_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.models or self.feature_names_ is None:
            raise RuntimeError("Model is not fitted yet")
        X = pd.DataFrame(X).copy()[self.feature_names_]
        raw = np.column_stack([self.models[q].predict(X) for q in self.cfg.quantiles])
        # Avoid quantile crossing by enforcing monotonicity row-by-row.
        raw = np.maximum.accumulate(raw, axis=1)
        return pd.DataFrame(raw, columns=[f"q{int(q * 100):02d}_delta" for q in self.cfg.quantiles], index=X.index)

    def predict_price_quantiles(self, X: pd.DataFrame, mid_now: Sequence[float]) -> pd.DataFrame:
        delta_q = self.predict_delta_quantiles(X)
        mid_now = np.asarray(mid_now, dtype=float).reshape(-1)
        if mid_now.size != delta_q.shape[0]:
            raise ValueError("mid_now must have the same length as X")
        out = delta_q.copy()
        for col in delta_q.columns:
            out[col.replace("_delta", "_price")] = mid_now + delta_q[col].to_numpy()
        return out


def train_test_split_time_ordered(frame: pd.DataFrame, cfg: IntervalForecasterConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_test = max(int(round(len(frame) * cfg.test_fraction)), 1)
    n_train = len(frame) - n_test
    if n_train < 100 or n_test < 20:
        raise ValueError("Too few samples after time split. Lower horizon/lookback or use more data.")
    return frame.iloc[:n_train].copy(), frame.iloc[n_train:].copy()


def evaluate_interval_predictions(
    y_true_delta: Sequence[float],
    pred_delta: pd.DataFrame,
    quantiles: Tuple[float, ...],
) -> Dict[str, float]:
    y = np.asarray(y_true_delta, dtype=float).reshape(-1)
    pred = pred_delta.to_numpy(dtype=float)
    metrics: Dict[str, float] = {}

    for j, q in enumerate(quantiles):
        metrics[f"pinball_q{int(q * 100):02d}"] = float(mean_pinball_loss(y, pred[:, j], alpha=q))

    lower = pred[:, 0]
    upper = pred[:, -1]
    median = pred[:, int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))]
    metrics["coverage"] = float(np.mean((y >= lower) & (y <= upper)))
    metrics["avg_interval_width"] = float(np.mean(upper - lower))
    metrics["median_mae"] = float(np.mean(np.abs(y - median)))
    metrics["directional_accuracy"] = float(np.mean(np.sign(y) == np.sign(median)))
    return metrics


def fit_price_interval_model_from_backtest(
    res: Dict[str, np.ndarray],
    cfg: Optional[IntervalForecasterConfig] = None,
) -> Dict[str, object]:
    """End-to-end training helper for the main notebook.

    Returns a dictionary containing the fitted model, train/test frames,
    predictions and evaluation metrics.
    """

    cfg = cfg or IntervalForecasterConfig()
    frame = build_backtest_feature_frame(res, cfg)
    train_frame, test_frame = train_test_split_time_ordered(frame, cfg)
    cols = feature_columns(frame)

    model = QuantilePriceIntervalForecaster(cfg)
    model.fit(train_frame[cols], train_frame["target_delta"])

    pred_delta = model.predict_delta_quantiles(test_frame[cols])
    pred_price = model.predict_price_quantiles(test_frame[cols], test_frame["mid"])
    metrics = evaluate_interval_predictions(test_frame["target_delta"], pred_delta, cfg.quantiles)

    return {
        "config": cfg,
        "model": model,
        "frame": frame,
        "train_frame": train_frame,
        "test_frame": test_frame,
        "features": cols,
        "pred_delta": pred_delta,
        "pred_price": pred_price,
        "metrics": metrics,
    }


def plot_price_intervals(result: Dict[str, object], max_points: int = 500, title: Optional[str] = None):
    """Plot true future price, median prediction and confidence band."""

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
        label=f"{int((cfg.quantiles[-1] - cfg.quantiles[0]) * 100)}% interval",
    )
    ax.set_title(title or "Next-price quantile confidence interval")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig, ax
