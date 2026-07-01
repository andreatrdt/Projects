from __future__ import annotations

"""Clock-time uncertainty bands for the Hawkes optimal-control project.

This module implements the final ML layer used in the notebook.

The key modelling choice is deliberately conservative:

    point forecast:      S_hat[t+h] = S[t]
    uncertainty model:   r_hat[t,h] ~= conditional high quantile of |S[t+h] - S[t]|
    price band:          [S[t] - r_hat[t,h], S[t] + r_hat[t,h]]

So the ML model is not asked to predict direction. It only predicts the size of
future uncertainty around the naive/persistence mid-price. This matched the
empirical diagnostics better than using ML as the median price forecaster.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)

try:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.metrics import mean_pinball_loss
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    ExtraTreesClassifier = None
    ExtraTreesRegressor = None
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None
    RandomForestRegressor = None
    MLPClassifier = None
    Pipeline = None
    StandardScaler = None
    mean_pinball_loss = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

try:  # Optional. The tree models work without torch.
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


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


def infer_time_step_seconds(time_values: Sequence[float]) -> float:
    """Infer the median positive time step from a monotone time grid."""
    t = np.asarray(time_values, dtype=float).reshape(-1)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Cannot infer time step from non-increasing or empty time grid.")
    return float(np.median(dt))


def steps_from_seconds(seconds: float, sample_dt_seconds: float) -> int:
    """Convert clock seconds to grid steps with a safety check."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if sample_dt_seconds <= 0:
        raise ValueError("sample_dt_seconds must be positive")
    steps = int(round(seconds / sample_dt_seconds))
    return max(1, steps)


def resample_backtest_result(res: Dict[str, np.ndarray], dt_seconds: float = 0.1) -> Dict[str, np.ndarray]:
    """Resample dense HJB/backtest output onto a regular clock-time grid.

    Prices, quotes, intensities, fill probabilities and PnL are linearly
    interpolated. Inventory and count-like diagnostics are held piecewise
    constant using the last observed value.
    """
    if "time" not in res:
        raise KeyError("res must contain a 'time' array")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")

    t = np.asarray(res["time"], dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError("Need at least two time points to resample.")

    t0, t1 = float(t[0]), float(t[-1])
    t_new = np.arange(t0, t1 + 0.5 * dt_seconds, dt_seconds)
    t_new = t_new[t_new <= t1 + 1e-12]

    out: Dict[str, np.ndarray] = {"time": t_new}
    hold_previous = {"q", "H_a", "H_b"}

    for key, values in res.items():
        if key == "time":
            continue
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.size != t.size:
            continue

        arr_float = arr.astype(float)
        if key in hold_previous:
            idx = np.searchsorted(t, t_new, side="right") - 1
            idx = np.clip(idx, 0, len(t) - 1)
            out[key] = arr_float[idx]
        else:
            out[key] = np.interp(t_new, t, arr_float)

    return out


def _default_lag_steps(lookback: int) -> Tuple[int, ...]:
    candidates = [1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, lookback]
    return tuple(sorted({int(x) for x in candidates if 1 <= int(x) <= lookback}))


def _default_rolling_windows(lookback: int) -> Tuple[int, ...]:
    candidates = [5, 10, 20, 50, 100, 200, 300, 500, lookback]
    return tuple(sorted({int(x) for x in candidates if 2 <= int(x) <= lookback}))


@dataclass(frozen=True)
class UncertaintyBandConfig:
    """Configuration for naive-centered ML uncertainty bands.

    Parameters
    ----------
    horizon_steps:
        Number of rows/grid points ahead to evaluate S[t+h].
    lookback:
        Maximum feature lookback in rows/grid points.
    radius_quantile:
        Quantile level for |S[t+h] - S[t]|. For a symmetric band centered at
        S[t], 0.90 targets an approximately 90% prediction interval.
    model_type:
        One of ``gbrt_radius``, ``extra_trees_radius``, ``rf_radius`` or
        ``net_radius``.
    conformal_alpha:
        If not None, uses split conformal calibration to widen the radius. For
        a 90% interval, use 0.10.
    """

    horizon_steps: int
    lookback: int
    radius_quantile: float = 0.90
    test_fraction: float = 0.25
    model_type: str = "gbrt_radius"
    conformal_alpha: Optional[float] = 0.10
    calibration_fraction: float = 0.20
    sample_dt_seconds: Optional[float] = None
    horizon_seconds: Optional[float] = None
    lookback_seconds: Optional[float] = None

    # Feature controls. Keep these sparse; do not create 300 raw lags.
    lag_steps: Optional[Tuple[int, ...]] = None
    rolling_windows: Optional[Tuple[int, ...]] = None

    # Tree/boosting controls.
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 30
    l2_regularization: float = 0.0
    n_estimators: int = 250
    random_state: int = 42

    # Neural net controls.
    hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32)
    net_batch_size: int = 512
    net_patience: int = 25
    net_val_fraction: float = 0.20

    def validate(self) -> None:
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        if not (0.5 < self.radius_quantile < 0.999):
            raise ValueError("radius_quantile should be in (0.5, 0.999)")
        if not (0.05 <= self.test_fraction <= 0.5):
            raise ValueError("test_fraction should be in [0.05, 0.5]")
        if not (0.05 <= self.calibration_fraction <= 0.5):
            raise ValueError("calibration_fraction should be in [0.05, 0.5]")
        allowed = {"gbrt_radius", "extra_trees_radius", "rf_radius", "net_radius"}
        if self.model_type not in allowed:
            raise ValueError(f"model_type must be one of: {sorted(allowed)}")
        if self.conformal_alpha is not None and not (0.01 < self.conformal_alpha < 0.5):
            raise ValueError("conformal_alpha must be None or in (0.01, 0.5)")
        if not (0.05 <= self.net_val_fraction <= 0.4):
            raise ValueError("net_val_fraction should be in [0.05, 0.4]")

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - float(self.conformal_alpha) if self.conformal_alpha is not None else float(self.radius_quantile)


def _base_frame_from_res(res: Dict[str, np.ndarray]) -> pd.DataFrame:
    required = {"time", "mid", "bid", "ask"}
    missing = required - set(res.keys())
    if missing:
        raise KeyError(f"Backtest result is missing required keys: {sorted(missing)}")

    mid = np.asarray(res["mid"], dtype=float).reshape(-1)
    n = mid.size
    return pd.DataFrame(
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


def build_uncertainty_feature_frame(res: Dict[str, np.ndarray], cfg: UncertaintyBandConfig) -> pd.DataFrame:
    """Build a leak-safe supervised frame for uncertainty-radius prediction.

    Targets:
    - ``target_delta`` = S[t+h] - S[t]
    - ``target_abs_delta`` = |S[t+h] - S[t]|
    - ``target_price`` = S[t+h]

    The model is trained on ``target_abs_delta``. The center remains ``mid``.
    """
    cfg.validate()
    df = _base_frame_from_res(res)
    n = len(df)
    if n <= cfg.horizon_steps + cfg.lookback + 10:
        raise ValueError("Not enough observations for the requested lookback/horizon.")

    # Contemporaneous state available at decision time t.
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

    lag_steps = cfg.lag_steps or _default_lag_steps(cfg.lookback)
    for lag in lag_steps:
        df[f"dmid_lag_{lag}"] = df["dmid"].shift(lag)
        df[f"abs_dmid_lag_{lag}"] = df["abs_dmid"].shift(lag)
        df[f"ret_lag_{lag}"] = df["ret"].shift(lag)
        df[f"flow_imbalance_lag_{lag}"] = df["flow_imbalance"].shift(lag)
        df[f"lam_sum_lag_{lag}"] = lam_sum.shift(lag)
        df[f"q_lag_{lag}"] = df["q"].shift(lag)

    windows = cfg.rolling_windows or _default_rolling_windows(cfg.lookback)
    for window in windows:
        df[f"dmid_sum_{window}"] = df["dmid"].rolling(window).sum()
        df[f"realized_vol_{window}"] = df["dmid"].rolling(window).std()
        df[f"abs_dmid_mean_{window}"] = df["abs_dmid"].rolling(window).mean()
        df[f"abs_dmid_max_{window}"] = df["abs_dmid"].rolling(window).max()
        df[f"flow_imbalance_mean_{window}"] = df["flow_imbalance"].rolling(window).mean()
        df[f"flow_imbalance_std_{window}"] = df["flow_imbalance"].rolling(window).std()
        df[f"fill_imbalance_sum_{window}"] = df["net_fills"].rolling(window).sum()
        df[f"lam_sum_mean_{window}"] = lam_sum.rolling(window).mean()
        df[f"lam_sum_std_{window}"] = lam_sum.rolling(window).std()

    df["target_price"] = df["mid"].shift(-cfg.horizon_steps)
    df["target_delta"] = df["target_price"] - df["mid"]
    df["target_abs_delta"] = df["target_delta"].abs()

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def uncertainty_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"time", "target_price", "target_delta", "target_abs_delta"}
    return [c for c in frame.columns if c not in excluded]


def train_cal_test_split_time_ordered(
    frame: pd.DataFrame, cfg: UncertaintyBandConfig
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Split a time-ordered frame into train/calibration/test blocks."""
    n = len(frame)
    n_test = max(int(round(n * cfg.test_fraction)), 1)
    train_cal = frame.iloc[: n - n_test].copy()
    test = frame.iloc[n - n_test :].copy()

    train = train_cal
    cal = None
    if cfg.conformal_alpha is not None:
        n_cal = max(int(round(len(train_cal) * cfg.calibration_fraction)), 1)
        train = train_cal.iloc[: len(train_cal) - n_cal].copy()
        cal = train_cal.iloc[len(train_cal) - n_cal :].copy()

    if len(train) < 100 or len(test) < 20:
        raise ValueError("Too few samples after time split. Lower horizon/lookback or use more data.")
    if cal is not None and len(cal) < 20:
        raise ValueError("Too few calibration samples for conformal intervals.")
    return train, cal, test


def pinball_loss_np(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


def winkler_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """Mean Winkler interval score for nominal coverage 1-alpha."""
    width = upper - lower
    below = y < lower
    above = y > upper
    score = width.copy()
    score[below] += (2.0 / alpha) * (lower[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - upper[above])
    return float(np.mean(score))


class _TorchRadiusQuantileRegressor:
    """Small PyTorch MLP trained with pinball loss on |future price move|."""

    def __init__(
        self,
        quantile: float,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32),
        max_epochs: int = 250,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        l2_regularization: float = 1e-5,
        patience: int = 25,
        val_fraction: float = 0.20,
        random_state: int = 42,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for model_type='net_radius'. Install torch or use a tree model.")
        self.quantile = float(quantile)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.l2_regularization = float(l2_regularization)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.model_: Optional[nn.Module] = None
        self.best_val_loss_: Optional[float] = None

    def _build_model(self, n_features: int) -> nn.Module:
        layers = []
        prev = n_features
        for width in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(int(width)))
            prev = int(width)
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Softplus())  # positive radius
        return nn.Sequential(*layers)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = target - pred
        q = self.quantile
        return torch.mean(torch.maximum(q * err, (q - 1.0) * err))

    def fit(self, X: pd.DataFrame, y: Sequence[float]) -> "_TorchRadiusQuantileRegressor":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y have incompatible lengths")

        n = X_arr.shape[0]
        n_val = max(20, int(round(n * self.val_fraction)))
        n_train = n - n_val
        if n_train < 50:
            n_train = n
            n_val = 0

        X_train_raw = X_arr[:n_train]
        y_train = y_arr[:n_train]
        X_scaled_train = self.scaler.fit_transform(X_train_raw).astype(np.float32)

        if n_val > 0:
            X_val = self.scaler.transform(X_arr[n_train:]).astype(np.float32)
            y_val = y_arr[n_train:]
        else:
            X_val = None
            y_val = None

        self.model_ = self._build_model(X_arr.shape[1])
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_regularization,
        )

        ds = TensorDataset(torch.from_numpy(X_scaled_train), torch.from_numpy(y_train))
        loader = DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True)

        best_state = None
        best_val = float("inf")
        bad_epochs = 0

        for _epoch in range(self.max_epochs):
            self.model_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model_(xb)
                loss = self._loss(pred, yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                if X_val is not None:
                    pred_val = self.model_(torch.from_numpy(X_val))
                    val_loss = float(self._loss(pred_val, torch.from_numpy(y_val)).item())
                else:
                    pred_train = self.model_(torch.from_numpy(X_scaled_train))
                    val_loss = float(self._loss(pred_train, torch.from_numpy(y_train)).item())

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_loss_ = best_val
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Net is not fitted yet")
        X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X_arr).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.from_numpy(X_scaled)).cpu().numpy().reshape(-1)
        return np.maximum(pred, 0.0)


class RadiusUncertaintyBandForecaster:
    """Predict symmetric uncertainty radius around the naive mid-price center."""

    def __init__(self, cfg: UncertaintyBandConfig):
        _require_sklearn()
        cfg.validate()
        self.cfg = cfg
        self.model: Optional[object] = None
        self.feature_names_: Optional[list[str]] = None
        self.conformal_padding_: Optional[float] = None

    def _new_model(self):
        if self.cfg.model_type == "gbrt_radius":
            return HistGradientBoostingRegressor(
                loss="quantile",
                quantile=self.cfg.radius_quantile,
                max_iter=self.cfg.max_iter,
                learning_rate=self.cfg.learning_rate,
                max_leaf_nodes=self.cfg.max_leaf_nodes,
                min_samples_leaf=self.cfg.min_samples_leaf,
                l2_regularization=self.cfg.l2_regularization,
                random_state=self.cfg.random_state,
            )
        if self.cfg.model_type == "extra_trees_radius":
            return ExtraTreesRegressor(
                n_estimators=self.cfg.n_estimators,
                min_samples_leaf=max(2, self.cfg.min_samples_leaf // 5),
                max_features=0.8,
                bootstrap=True,
                n_jobs=-1,
                random_state=self.cfg.random_state,
            )
        if self.cfg.model_type == "rf_radius":
            return RandomForestRegressor(
                n_estimators=self.cfg.n_estimators,
                min_samples_leaf=max(2, self.cfg.min_samples_leaf // 5),
                max_features=0.8,
                n_jobs=-1,
                random_state=self.cfg.random_state,
            )
        if self.cfg.model_type == "net_radius":
            return _TorchRadiusQuantileRegressor(
                quantile=self.cfg.radius_quantile,
                hidden_layer_sizes=self.cfg.hidden_layer_sizes,
                max_epochs=self.cfg.max_iter,
                batch_size=self.cfg.net_batch_size,
                learning_rate=max(1e-4, self.cfg.learning_rate * 0.02),
                l2_regularization=max(1e-7, self.cfg.l2_regularization),
                patience=self.cfg.net_patience,
                val_fraction=self.cfg.net_val_fraction,
                random_state=self.cfg.random_state,
            )
        raise RuntimeError(f"Unknown model_type={self.cfg.model_type}")

    def fit(self, X: pd.DataFrame, y_abs_delta: Sequence[float]) -> "RadiusUncertaintyBandForecaster":
        X = pd.DataFrame(X).copy()
        y = np.asarray(y_abs_delta, dtype=float).reshape(-1)
        if X.shape[0] != y.size:
            raise ValueError("X and y_abs_delta have incompatible lengths")
        self.feature_names_ = list(X.columns)
        self.conformal_padding_ = None
        self.model = self._new_model()
        self.model.fit(X, y)
        return self

    def predict_radius(self, X: pd.DataFrame, include_conformal: bool = True) -> pd.Series:
        if self.model is None or self.feature_names_ is None:
            raise RuntimeError("Model is not fitted yet")
        X = pd.DataFrame(X).copy()[self.feature_names_]

        if self.cfg.model_type in {"extra_trees_radius", "rf_radius"}:
            X_values = X.to_numpy(dtype=float)
            tree_preds = np.column_stack([tree.predict(X_values) for tree in self.model.estimators_])
            radius = np.quantile(tree_preds, self.cfg.radius_quantile, axis=1)
        else:
            radius = self.model.predict(X)

        radius = np.maximum(np.asarray(radius, dtype=float).reshape(-1), 0.0)
        if include_conformal and self.conformal_padding_ is not None:
            radius = radius + float(self.conformal_padding_)
        return pd.Series(radius, index=X.index, name="pred_radius")

    def predict_price_band(self, X: pd.DataFrame, mid_now: Sequence[float]) -> pd.DataFrame:
        radius = self.predict_radius(X, include_conformal=True).to_numpy(dtype=float)
        mid_now = np.asarray(mid_now, dtype=float).reshape(-1)
        if mid_now.size != radius.size:
            raise ValueError("mid_now must have the same length as X")
        return pd.DataFrame(
            {
                "center_price": mid_now,
                "lower_price": mid_now - radius,
                "upper_price": mid_now + radius,
                "pred_radius": radius,
            },
            index=X.index,
        )

    def calibrate_conformal(self, X_cal: pd.DataFrame, y_cal_abs_delta: Sequence[float]) -> float:
        """Conformalize the radius using chronological calibration data."""
        if self.cfg.conformal_alpha is None:
            self.conformal_padding_ = None
            return 0.0
        raw_radius = self.predict_radius(X_cal, include_conformal=False).to_numpy(dtype=float)
        y = np.asarray(y_cal_abs_delta, dtype=float).reshape(-1)
        scores = y - raw_radius
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.cfg.conformal_alpha)) / n)
        padding = float(np.quantile(scores, level, method="higher"))
        padding = max(0.0, padding)
        self.conformal_padding_ = padding
        return padding


def evaluate_uncertainty_bands(
    y_true_delta: Sequence[float],
    pred_radius: Sequence[float],
    radius_quantile: float,
    nominal_coverage: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate naive-centered uncertainty bands."""
    y_delta = np.asarray(y_true_delta, dtype=float).reshape(-1)
    y_abs = np.abs(y_delta)
    radius = np.maximum(np.asarray(pred_radius, dtype=float).reshape(-1), 0.0)
    nominal = float(nominal_coverage if nominal_coverage is not None else radius_quantile)
    alpha = max(1e-6, 1.0 - nominal)

    lower_delta = -radius
    upper_delta = radius
    inside = (y_delta >= lower_delta) & (y_delta <= upper_delta)

    return {
        "nominal_coverage": nominal,
        "empirical_coverage": float(np.mean(inside)),
        "coverage_error": float(np.mean(inside) - nominal),
        "avg_radius": float(np.mean(radius)),
        "avg_interval_width": float(np.mean(2.0 * radius)),
        "radius_pinball_loss": float(
            mean_pinball_loss(y_abs, radius, alpha=radius_quantile)
            if mean_pinball_loss is not None
            else pinball_loss_np(y_abs, radius, radius_quantile)
        ),
        "winkler_score": winkler_score(y_delta, lower_delta, upper_delta, alpha=alpha),
        "naive_center_mae": float(np.mean(y_abs)),
        "point_forecast": "persistence_mid",
        "point_improvement_vs_naive": 0.0,
        "avg_abs_move": float(np.mean(y_abs)),
        "max_abs_move": float(np.max(y_abs)),
    }


def fit_uncertainty_band_model_from_backtest(
    res: Dict[str, np.ndarray],
    cfg: UncertaintyBandConfig,
) -> Dict[str, object]:
    """Fit one naive-centered uncertainty model for a fixed horizon."""
    cfg.validate()
    frame = build_uncertainty_feature_frame(res, cfg)
    train_frame, cal_frame, test_frame = train_cal_test_split_time_ordered(frame, cfg)
    cols = uncertainty_feature_columns(frame)

    model = RadiusUncertaintyBandForecaster(cfg)
    model.fit(train_frame[cols], train_frame["target_abs_delta"])

    raw_radius = model.predict_radius(test_frame[cols], include_conformal=False)
    raw_metrics = evaluate_uncertainty_bands(
        test_frame["target_delta"], raw_radius, cfg.radius_quantile, nominal_coverage=cfg.radius_quantile
    )

    conformal_padding = None
    if cal_frame is not None:
        conformal_padding = model.calibrate_conformal(cal_frame[cols], cal_frame["target_abs_delta"])

    pred_radius = model.predict_radius(test_frame[cols], include_conformal=True)
    pred_price = model.predict_price_band(test_frame[cols], test_frame["mid"])
    metrics = evaluate_uncertainty_bands(
        test_frame["target_delta"],
        pred_radius,
        cfg.radius_quantile,
        nominal_coverage=cfg.nominal_coverage,
    )
    metrics["raw_empirical_coverage"] = raw_metrics["empirical_coverage"]
    metrics["raw_avg_interval_width"] = raw_metrics["avg_interval_width"]
    metrics["conformal_padding"] = float(conformal_padding or 0.0)

    return {
        "config": cfg,
        "config_dict": asdict(cfg),
        "model": model,
        "frame": frame,
        "train_frame": train_frame,
        "calibration_frame": cal_frame,
        "test_frame": test_frame,
        "features": cols,
        "pred_radius": pred_radius,
        "pred_price": pred_price,
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "conformal_padding": conformal_padding,
    }


def fit_uncertainty_bands_multi_horizon(
    res_clock: Dict[str, np.ndarray],
    sample_dt_seconds: float = 0.1,
    horizons_seconds: Sequence[float] = (1.0, 5.0, 10.0),
    lookback_seconds: float = 30.0,
    model_types: Sequence[str] = ("gbrt_radius", "extra_trees_radius", "net_radius"),
    radius_quantile: float = 0.90,
    conformal_alpha: Optional[float] = 0.10,
    test_fraction: float = 0.25,
    calibration_fraction: float = 0.20,
    max_iter: int = 200,
    n_estimators: int = 250,
    random_state: int = 42,
) -> Dict[str, object]:
    """Fit several radius models at 1/5/10s style horizons and select one per horizon."""
    candidate_rows = []
    all_results: Dict[float, Dict[str, Dict[str, object]]] = {}
    selected_results: Dict[float, Dict[str, object]] = {}

    lookback_steps = steps_from_seconds(lookback_seconds, sample_dt_seconds)

    for h in horizons_seconds:
        h = float(h)
        horizon_steps = steps_from_seconds(h, sample_dt_seconds)
        all_results[h] = {}
        for model_type in model_types:
            cfg = UncertaintyBandConfig(
                horizon_steps=horizon_steps,
                lookback=lookback_steps,
                radius_quantile=radius_quantile,
                test_fraction=test_fraction,
                model_type=model_type,
                conformal_alpha=conformal_alpha,
                calibration_fraction=calibration_fraction,
                sample_dt_seconds=sample_dt_seconds,
                horizon_seconds=h,
                lookback_seconds=lookback_seconds,
                max_iter=max_iter,
                n_estimators=n_estimators,
                random_state=random_state,
            )
            result = fit_uncertainty_band_model_from_backtest(res_clock, cfg)
            all_results[h][model_type] = result
            candidate_rows.append(
                {
                    "horizon_seconds": h,
                    "horizon_steps": horizon_steps,
                    "model_type": model_type,
                    **result["metrics"],
                }
            )

        # Select by Winkler score: rewards narrow intervals but penalizes misses.
        best_type = min(
            all_results[h].keys(),
            key=lambda mt: all_results[h][mt]["metrics"]["winkler_score"],
        )
        selected_results[h] = all_results[h][best_type]

    candidate_table = pd.DataFrame(candidate_rows).sort_values(
        ["horizon_seconds", "winkler_score", "avg_interval_width"]
    ).reset_index(drop=True)

    selected_rows = []
    for h, result in selected_results.items():
        selected_rows.append(
            {
                "horizon_seconds": h,
                "selected_model": result["config"].model_type,
                **result["metrics"],
            }
        )
    metrics_table = pd.DataFrame(selected_rows).sort_values("horizon_seconds").reset_index(drop=True)

    return {
        "results_by_horizon": selected_results,
        "all_results_by_horizon": all_results,
        "metrics_table": metrics_table,
        "candidate_table": candidate_table,
        "sample_dt_seconds": sample_dt_seconds,
        "lookback_seconds": lookback_seconds,
    }


def plot_realized_vs_uncertainty_band(
    result: Dict[str, object],
    max_points: int = 1200,
    title: Optional[str] = None,
):
    """Plot realized future mid against naive center and predicted uncertainty band."""
    test_frame = result["test_frame"]
    pred_price = result["pred_price"]
    cfg: UncertaintyBandConfig = result["config"]

    n = min(max_points, len(test_frame))
    tf = test_frame.iloc[:n]
    pp = pred_price.iloc[:n]
    h_label = f"{cfg.horizon_seconds:g}s" if cfg.horizon_seconds is not None else f"{cfg.horizon_steps} steps"
    x = tf["time"].to_numpy() + (cfg.horizon_seconds or cfg.horizon_steps)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, tf["target_price"].to_numpy(), label=f"realized mid at t+{h_label}", linewidth=1.3)
    ax.plot(x, pp["center_price"].to_numpy(), label="naive center: current mid", linewidth=1.2, linestyle="--")
    ax.fill_between(
        x,
        pp["lower_price"].to_numpy(),
        pp["upper_price"].to_numpy(),
        alpha=0.22,
        label=f"naive-centered {int(round(result['metrics']['nominal_coverage'] * 100))}% band",
    )
    ax.set_title(title or f"Realized vs uncertainty band, horizon {h_label}")
    ax.set_xlabel("Target time since market open, seconds")
    ax.set_ylabel("Mid-price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def _predict_one_origin_from_result(result: Dict[str, object], origin_time: float) -> Dict[str, float]:
    frame = result["frame"]
    features = result["features"]
    model: RadiusUncertaintyBandForecaster = result["model"]
    cfg: UncertaintyBandConfig = result["config"]

    idx = int(np.argmin(np.abs(frame["time"].to_numpy() - origin_time)))
    row = frame.iloc[[idx]]
    pred = model.predict_price_band(row[features], row["mid"])
    h = float(cfg.horizon_seconds if cfg.horizon_seconds is not None else cfg.horizon_steps)
    return {
        "origin_time": float(row["time"].iloc[0]),
        "future_time": float(row["time"].iloc[0]) + h,
        "mid_now": float(row["mid"].iloc[0]),
        "center": float(pred["center_price"].iloc[0]),
        "lower": float(pred["lower_price"].iloc[0]),
        "upper": float(pred["upper_price"].iloc[0]),
        "radius": float(pred["pred_radius"].iloc[0]),
    }


def plot_forecast_fan(
    results_by_horizon: Dict[float, Dict[str, object]],
    res_clock: Dict[str, np.ndarray],
    history_seconds: float = 60.0,
    origin_time: Optional[float] = None,
    symbol: Optional[str] = None,
):
    """Plot recent history plus future naive-centered uncertainty bands."""
    horizons = sorted(results_by_horizon.keys())
    if not horizons:
        raise ValueError("results_by_horizon is empty")

    if origin_time is None:
        # Use the largest-horizon frame to pick a safe origin with known target.
        origin_time = float(results_by_horizon[max(horizons)]["frame"]["time"].iloc[-1])

    t = np.asarray(res_clock["time"], dtype=float)
    mid = np.asarray(res_clock["mid"], dtype=float)
    mask = (t >= origin_time - history_seconds) & (t <= origin_time)

    origin_mid = float(np.interp(origin_time, t, mid))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t[mask], mid[mask], label=f"last {history_seconds:g}s mid", linewidth=1.5)
    ax.scatter([origin_time], [origin_mid], s=45, label="forecast origin")

    future_times = []
    centers = []
    lowers = []
    uppers = []
    for h in horizons:
        pred = _predict_one_origin_from_result(results_by_horizon[h], origin_time)
        future_times.append(pred["future_time"])
        centers.append(pred["center"])
        lowers.append(pred["lower"])
        uppers.append(pred["upper"])
        ax.vlines(pred["future_time"], pred["lower"], pred["upper"], linewidth=5, alpha=0.35, label=f"{h:g}s band")
        ax.scatter([pred["future_time"]], [pred["center"]], s=50)

    ax.plot([origin_time] + future_times, [origin_mid] + centers, linestyle="--", linewidth=1.2, label="naive center path")
    ax.fill_between([origin_time] + future_times, [origin_mid] + lowers, [origin_mid] + uppers, alpha=0.12, label="forecast fan visual aid")

    name = f"{symbol}: " if symbol else ""
    ax.set_title(f"{name}recent mid-price plus naive-centered uncertainty bands")
    ax.set_xlabel("Time since market open, seconds")
    ax.set_ylabel("Mid-price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_predicted_radius_vs_realized_abs_move(
    result: Dict[str, object],
    max_points: int = 5000,
    title: Optional[str] = None,
):
    """Scatter diagnostic: predicted radius vs realized absolute future move."""
    test_frame = result["test_frame"]
    pred_radius = result["pred_radius"]
    cfg: UncertaintyBandConfig = result["config"]

    n = min(max_points, len(test_frame))
    y_abs = test_frame["target_abs_delta"].iloc[:n].to_numpy(dtype=float)
    r = pred_radius.iloc[:n].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(r, y_abs, s=10, alpha=0.35)
    lim = float(max(np.nanmax(r), np.nanmax(y_abs))) if len(y_abs) else 1.0
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.0, label="realized = predicted")
    h_label = f"{cfg.horizon_seconds:g}s" if cfg.horizon_seconds is not None else f"{cfg.horizon_steps} steps"
    ax.set_title(title or f"Predicted radius vs realized absolute move, horizon {h_label}")
    ax.set_xlabel("Predicted uncertainty radius")
    ax.set_ylabel("Realized |future mid - current mid|")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Full-day original LOB model: hybrid center + uncertainty bands
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LOBHybridBandConfig:
    """Configuration for full-day LOB hybrid center/radius bands.

    The preferred center is a bounded up/down/no-move classifier:

        raw_center_delta[t,h] = move_size_h * (P(up|X_t) - P(down|X_t))
        center[t,h] = mid[t] + alpha_h * raw_center_delta[t,h]

    The shrinkage coefficient alpha_h is calibrated before the application
    window. If the directional classifier is not useful, alpha_h can still go
    to zero and the model falls back to persistence. The radius model then
    learns the absolute residual around this center.
    """

    horizon_steps: int
    lookback: int
    center_model_type: str = "extra_trees_classifier_center"
    radius_model_type: str = "extra_trees_radius"
    radius_quantile: float = 0.90
    conformal_alpha: Optional[float] = 0.10
    calibration_fraction: float = 0.20
    sample_dt_seconds: Optional[float] = None
    horizon_seconds: Optional[float] = None
    lookback_seconds: Optional[float] = None
    max_iter: int = 100
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 30
    l2_regularization: float = 0.0
    n_estimators: int = 150
    random_state: int = 42
    hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32)
    net_batch_size: int = 512
    net_patience: int = 20
    net_val_fraction: float = 0.20
    lag_steps: Optional[Tuple[int, ...]] = None
    rolling_windows: Optional[Tuple[int, ...]] = None

    # Classifier-center controls. The classifier predicts down/no-move/up.
    # The predicted center tilt is bounded by classifier_move_size:
    #     raw_delta = classifier_move_size * (P(up) - P(down)).
    center_tick_size: Optional[float] = None
    classifier_deadzone_ticks: float = 0.5
    classifier_move_size: Optional[float] = None
    alpha_move_weight: float = 5.0

    def validate(self) -> None:
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        if not (0.5 < self.radius_quantile < 0.999):
            raise ValueError("radius_quantile should be in (0.5, 0.999)")
        allowed_center = {
            "extra_trees_center",
            "gbrt_center",
            "net_center",
            "extra_trees_classifier_center",
            "gbrt_classifier_center",
            "net_classifier_center",
        }
        if self.center_model_type not in allowed_center:
            raise ValueError(f"center_model_type must be one of: {sorted(allowed_center)}")
        allowed_radius = {"gbrt_radius", "extra_trees_radius", "rf_radius", "net_radius"}
        if self.radius_model_type not in allowed_radius:
            raise ValueError(f"radius_model_type must be one of: {sorted(allowed_radius)}")
        if self.conformal_alpha is not None and not (0.01 < self.conformal_alpha < 0.5):
            raise ValueError("conformal_alpha must be None or in (0.01, 0.5)")
        if not (0.05 <= self.calibration_fraction <= 0.5):
            raise ValueError("calibration_fraction should be in [0.05, 0.5]")
        if self.classifier_deadzone_ticks < 0.0:
            raise ValueError("classifier_deadzone_ticks must be non-negative")
        if self.classifier_move_size is not None and self.classifier_move_size <= 0.0:
            raise ValueError("classifier_move_size must be positive when provided")
        if self.alpha_move_weight < 0.0:
            raise ValueError("alpha_move_weight must be non-negative")

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - float(self.conformal_alpha) if self.conformal_alpha is not None else float(self.radius_quantile)


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _numeric_series_from_column(df: pd.DataFrame, col: Optional[str], n: int, fill: float = 0.0) -> np.ndarray:
    if col is None:
        return np.full(n, fill, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def resample_lob_orderbook_clock_time(
    lob_data: pd.DataFrame,
    dt_seconds: float = 0.1,
    time_col: str = "time",
) -> pd.DataFrame:
    """Resample an original LOB/order-book state dataframe to a clock-time grid.

    This function is deliberately independent from the HJB/backtest result. It
    uses previous-tick carry-forward because the limit order book is a state
    observed after each event. It accepts common column names produced by the
    project's ``OrderBook`` wrapper and by raw LOBSTER-style dataframes.

    Required, either directly or inferable:
    - time column;
    - mid price, or bid/ask columns from which mid can be computed.
    """
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    if not isinstance(lob_data, pd.DataFrame):
        lob_data = pd.DataFrame(lob_data)
    if time_col not in lob_data.columns:
        raise KeyError(f"LOB dataframe must contain a '{time_col}' column")

    df = lob_data.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least two LOB observations to resample.")

    t = df[time_col].to_numpy(dtype=float)
    # Drop duplicate timestamps by keeping the last state at that time.
    keep = np.r_[np.diff(t) > 0, True]
    if not np.all(keep):
        df = df.loc[keep].reset_index(drop=True)
        t = df[time_col].to_numpy(dtype=float)

    t0, t1 = float(t[0]), float(t[-1])
    grid = np.arange(t0, t1 + 0.5 * dt_seconds, dt_seconds)
    grid = grid[grid <= t1 + 1e-12]
    idx = np.searchsorted(t, grid, side="right") - 1
    idx = np.clip(idx, 0, len(df) - 1)

    mid_col = _first_existing_column(df, ["mid", "mid_price", "Mid", "midprice"])
    bid_col = _first_existing_column(df, ["bid", "bid_price", "bid_price_1", "BidPrice1", "bid1"])
    ask_col = _first_existing_column(df, ["ask", "ask_price", "ask_price_1", "AskPrice1", "ask1"])
    bid_size_col = _first_existing_column(df, ["bid_size", "bid_size_1", "BidSize1", "bid_volume_1", "bid_qty_1"])
    ask_size_col = _first_existing_column(df, ["ask_size", "ask_size_1", "AskSize1", "ask_volume_1", "ask_qty_1"])

    n = len(df)
    bid_raw = _numeric_series_from_column(df, bid_col, n, np.nan)
    ask_raw = _numeric_series_from_column(df, ask_col, n, np.nan)

    if mid_col is not None:
        mid_raw = _numeric_series_from_column(df, mid_col, n, np.nan)
    elif bid_col is not None and ask_col is not None:
        mid_raw = 0.5 * (bid_raw + ask_raw)
    else:
        raise KeyError("Cannot infer mid price. Need 'mid_price'/'mid' or bid/ask columns.")

    if bid_col is None:
        bid_raw = mid_raw.copy()
    if ask_col is None:
        ask_raw = mid_raw.copy()

    bid_size_raw = _numeric_series_from_column(df, bid_size_col, n, 0.0)
    ask_size_raw = _numeric_series_from_column(df, ask_size_col, n, 0.0)

    out = pd.DataFrame(
        {
            "time": grid,
            "mid": mid_raw[idx],
            "bid": bid_raw[idx],
            "ask": ask_raw[idx],
            "bid_size_1": bid_size_raw[idx],
            "ask_size_1": ask_size_raw[idx],
        }
    )

    # Include useful precomputed state columns when present.
    optional_cols = [
        "rho_5L",
        "rho_1L",
        "imbalance",
        "depth_imbalance",
        "spread",
        "mid_price",
        "bid_price",
        "ask_price",
    ]
    for col in optional_cols:
        if col in df.columns and col not in out.columns:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            out[col] = vals[idx]

    # Also keep the first few LOBSTER level prices/sizes if present. This gives
    # the full-day model more true order-book information without exploding the
    # feature dimension.
    for prefix in ("bid_price_", "ask_price_", "bid_size_", "ask_size_"):
        for level in range(1, 6):
            col = f"{prefix}{level}"
            if col in df.columns and col not in out.columns:
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                out[col] = vals[idx]

    out["spread"] = out["ask"] - out["bid"]
    out["half_spread"] = 0.5 * out["spread"]
    denom = out["bid_size_1"] + out["ask_size_1"]
    out["top_imbalance"] = (out["bid_size_1"] - out["ask_size_1"]) / (denom.abs() + 1e-12)
    out["microprice"] = (
        out["ask"] * out["bid_size_1"] + out["bid"] * out["ask_size_1"]
    ) / (denom.abs() + 1e-12)
    out["microprice_minus_mid"] = out["microprice"] - out["mid"]
    out["dmid"] = out["mid"].diff()
    out["abs_dmid"] = out["dmid"].abs()
    out["ret"] = _safe_divide(out["dmid"], out["mid"].shift(1))
    out["abs_ret"] = out["ret"].abs()
    out["spread_change"] = out["spread"].diff()

    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def infer_backtest_application_window(
    res_clock: Dict[str, np.ndarray] | pd.DataFrame,
    lob_clock_df: pd.DataFrame,
    proposed_time_offset: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Align local backtest time with the original full-day LOB clock.

    Backtest time is often local to the episode, for example 0..600 seconds.
    Original LOBSTER time is usually seconds from midnight, for example
    34200..57600. If ``proposed_time_offset=t_start`` is supplied, the function
    maps local backtest time to original time via ``time + t_start``.
    """
    if isinstance(res_clock, pd.DataFrame):
        res_t = res_clock["time"].to_numpy(dtype=float)
    else:
        res_t = np.asarray(res_clock["time"], dtype=float).reshape(-1)
    lob_t = lob_clock_df["time"].to_numpy(dtype=float)
    if res_t.size == 0 or lob_t.size == 0:
        raise ValueError("res_clock and lob_clock_df must both contain time values")

    res_start, res_end = float(np.nanmin(res_t)), float(np.nanmax(res_t))
    lob_start, lob_end = float(np.nanmin(lob_t)), float(np.nanmax(lob_t))

    candidates = []
    if proposed_time_offset is not None:
        candidates.append(float(proposed_time_offset))
    candidates.append(0.0)
    candidates.append(lob_end - res_end)

    for offset in candidates:
        app_start = res_start + offset
        app_end = res_end + offset
        if app_start >= lob_start and app_end <= lob_end:
            return float(app_start), float(app_end), float(offset)

    # Last-resort clipping: put the application window at the end of the LOB day.
    offset = lob_end - res_end
    app_start = max(lob_start, res_start + offset)
    app_end = min(lob_end, res_end + offset)
    warnings.warn(
        "Could not perfectly align the backtest window inside the LOB clock. "
        "Using a clipped end-aligned application window.",
        RuntimeWarning,
    )
    return float(app_start), float(app_end), float(offset)


def _build_lob_supervised_frame(lob_clock_df: pd.DataFrame, cfg: LOBHybridBandConfig) -> pd.DataFrame:
    cfg.validate()
    df = pd.DataFrame(lob_clock_df).copy().sort_values("time").reset_index(drop=True)
    required = {"time", "mid", "bid", "ask"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"lob_clock_df is missing required columns: {sorted(missing)}")
    if len(df) <= cfg.lookback + cfg.horizon_steps + 10:
        raise ValueError("Not enough LOB clock observations for requested lookback/horizon.")

    # Make sure core derived columns exist even if user passed a pre-built frame.
    df["spread"] = df.get("spread", df["ask"] - df["bid"])
    df["half_spread"] = df.get("half_spread", 0.5 * df["spread"])
    if "dmid" not in df.columns:
        df["dmid"] = df["mid"].diff()
    if "abs_dmid" not in df.columns:
        df["abs_dmid"] = df["dmid"].abs()
    if "ret" not in df.columns:
        df["ret"] = _safe_divide(df["dmid"], df["mid"].shift(1))
    if "abs_ret" not in df.columns:
        df["abs_ret"] = df["ret"].abs()
    if "top_imbalance" not in df.columns:
        if {"bid_size_1", "ask_size_1"}.issubset(df.columns):
            denom = df["bid_size_1"] + df["ask_size_1"]
            df["top_imbalance"] = (df["bid_size_1"] - df["ask_size_1"]) / (denom.abs() + 1e-12)
        else:
            df["top_imbalance"] = 0.0
    if "microprice_minus_mid" not in df.columns:
        if {"microprice", "mid"}.issubset(df.columns):
            df["microprice_minus_mid"] = df["microprice"] - df["mid"]
        else:
            df["microprice_minus_mid"] = 0.0

    # Sparse lags from LOB features. Do not create all 300 raw lags.
    base_lag_cols = [
        "dmid",
        "abs_dmid",
        "ret",
        "abs_ret",
        "spread",
        "top_imbalance",
        "microprice_minus_mid",
    ]
    for extra in ("rho_5L", "rho_1L", "depth_imbalance"):
        if extra in df.columns:
            base_lag_cols.append(extra)

    lag_steps = cfg.lag_steps or _default_lag_steps(cfg.lookback)
    lagged = {}
    for lag in lag_steps:
        for col in base_lag_cols:
            if col in df.columns:
                lagged[f"{col}_lag_{lag}"] = df[col].shift(lag)
    if lagged:
        df = pd.concat([df, pd.DataFrame(lagged, index=df.index)], axis=1)

    windows = cfg.rolling_windows or _default_rolling_windows(cfg.lookback)
    rolled = {}
    for window in windows:
        rolled[f"dmid_sum_{window}"] = df["dmid"].rolling(window).sum()
        rolled[f"realized_vol_{window}"] = df["dmid"].rolling(window).std()
        rolled[f"abs_dmid_mean_{window}"] = df["abs_dmid"].rolling(window).mean()
        rolled[f"spread_mean_{window}"] = df["spread"].rolling(window).mean()
        rolled[f"imbalance_mean_{window}"] = df["top_imbalance"].rolling(window).mean()
        rolled[f"microprice_minus_mid_mean_{window}"] = df["microprice_minus_mid"].rolling(window).mean()
    if rolled:
        df = pd.concat([df, pd.DataFrame(rolled, index=df.index)], axis=1)

    df["target_price"] = df["mid"].shift(-cfg.horizon_steps)
    df["target_time"] = df["time"].shift(-cfg.horizon_steps)
    df["target_delta"] = df["target_price"] - df["mid"]
    df["target_abs_delta"] = df["target_delta"].abs()

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _lob_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"time", "target_price", "target_time", "target_delta", "target_abs_delta"}
    # Keep numeric features only.
    cols = []
    for c in frame.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[c]):
            cols.append(c)
    return cols


def _is_classifier_center_type(center_model_type: str) -> bool:
    return center_model_type in {
        "extra_trees_classifier_center",
        "gbrt_classifier_center",
        "net_classifier_center",
    }


def _infer_price_tick_from_series(values: Sequence[float], fallback: float = 0.005) -> float:
    """Infer a robust mid-price tick from observed clock-time mid values."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float(fallback)
    # Work with observed mid changes, not all pairwise unique differences.
    changes = np.abs(np.diff(np.round(arr, 8)))
    changes = changes[np.isfinite(changes) & (changes > 1e-9)]
    if changes.size == 0:
        return float(fallback)
    # Use a lower quantile instead of the minimum to avoid floating-point noise.
    tick = float(np.quantile(changes, 0.10))
    if not np.isfinite(tick) or tick <= 0:
        tick = float(np.min(changes))
    return max(float(tick), 1e-6)


def _target_delta_to_direction_class(y_delta: Sequence[float], threshold: float) -> np.ndarray:
    """Map future mid-price changes to {-1, 0, +1}."""
    y = np.asarray(y_delta, dtype=float).reshape(-1)
    out = np.zeros(y.shape[0], dtype=int)
    out[y > threshold] = 1
    out[y < -threshold] = -1
    return out


def _typical_move_size(y_delta: Sequence[float], threshold: float, fallback: float) -> float:
    y = np.abs(np.asarray(y_delta, dtype=float).reshape(-1))
    moves = y[np.isfinite(y) & (y > threshold)]
    if moves.size == 0:
        return max(float(fallback), 1e-6)
    return max(float(np.median(moves)), float(fallback), 1e-6)


class _ConstantDirectionClassifier:
    """Fallback classifier used when the training block contains one class only."""

    def __init__(self, constant_class: int = 0):
        self.constant_class = int(constant_class)
        self.classes_ = np.array([-1, 0, 1], dtype=int)

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=int).reshape(-1)
        if y_arr.size:
            vals, counts = np.unique(y_arr, return_counts=True)
            self.constant_class = int(vals[int(np.argmax(counts))])
        return self

    def predict_proba(self, X):
        n = len(X)
        proba = np.zeros((n, 3), dtype=float)
        idx = int(np.where(self.classes_ == self.constant_class)[0][0])
        proba[:, idx] = 1.0
        return proba

    def predict(self, X):
        return np.full(len(X), self.constant_class, dtype=int)


def _classifier_probabilities(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return p_down, p_no_move, p_up from any sklearn-style classifier."""
    proba = model.predict_proba(X)
    classes = np.asarray(getattr(model, "classes_", np.array([-1, 0, 1])), dtype=int)

    def _class_prob(label: int) -> np.ndarray:
        mask = classes == int(label)
        if np.any(mask):
            return np.asarray(proba[:, mask], dtype=float).sum(axis=1)
        return np.zeros(proba.shape[0], dtype=float)

    p_down = _class_prob(-1)
    p_no_move = _class_prob(0)
    p_up = _class_prob(1)
    return p_down, p_no_move, p_up


def _fit_lob_center_model(
    X_train: pd.DataFrame,
    y_train_delta: Sequence[float],
    reference_frame: pd.DataFrame,
    cfg: LOBHybridBandConfig,
) -> tuple[object, Dict[str, float | str]]:
    """Fit either a signed-delta regressor or a bounded direction classifier."""
    model = _new_lob_center_model(cfg)
    if _is_classifier_center_type(cfg.center_model_type):
        tick = float(cfg.center_tick_size or _infer_price_tick_from_series(reference_frame["mid"]))
        threshold = float(cfg.classifier_deadzone_ticks * tick)
        y_class = _target_delta_to_direction_class(y_train_delta, threshold)
        move_size = float(cfg.classifier_move_size or _typical_move_size(y_train_delta, threshold, fallback=tick))
        if np.unique(y_class).size < 2:
            model = _ConstantDirectionClassifier().fit(X_train, y_class)
        else:
            model.fit(X_train, y_class)
        meta = {
            "center_kind": "direction_classifier",
            "center_tick_size": tick,
            "move_threshold": threshold,
            "classifier_move_size": move_size,
        }
        return model, meta

    model.fit(X_train, y_train_delta)
    return model, {
        "center_kind": "signed_delta_regression",
        "center_tick_size": np.nan,
        "move_threshold": 0.0,
        "classifier_move_size": np.nan,
    }


def _predict_lob_center_delta(
    model,
    X: pd.DataFrame,
    cfg: LOBHybridBandConfig,
    center_meta: Dict[str, float | str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Predict raw center delta and optional classifier probabilities."""
    if _is_classifier_center_type(cfg.center_model_type):
        p_down, p_no_move, p_up = _classifier_probabilities(model, X)
        move_size = float(center_meta.get("classifier_move_size", 0.0))
        raw_delta = move_size * (p_up - p_down)
        probs = pd.DataFrame(
            {
                "p_down": p_down,
                "p_no_move": p_no_move,
                "p_up": p_up,
                "directional_edge": p_up - p_down,
            },
            index=X.index,
        )
        return raw_delta.astype(float), probs

    raw = np.asarray(model.predict(X), dtype=float).reshape(-1)
    return raw, pd.DataFrame(index=X.index)


class _TorchCenterRegressor:
    """Small PyTorch MLP for signed future-delta regression."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32),
        max_epochs: int = 100,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        l2_regularization: float = 1e-6,
        patience: int = 20,
        val_fraction: float = 0.20,
        random_state: int = 42,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for net_center. Install torch or remove 'net_center'.")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.l2_regularization = float(l2_regularization)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.model_: Optional[nn.Module] = None

    def _build_model(self, n_features: int) -> nn.Module:
        layers = []
        prev = n_features
        for width in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(int(width)))
            layers.append(nn.Dropout(0.05))
            prev = int(width)
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X: pd.DataFrame, y: Sequence[float]) -> "_TorchCenterRegressor":
        torch.manual_seed(self.random_state)
        X_np = self.scaler.fit_transform(pd.DataFrame(X).to_numpy(dtype=np.float32)).astype(np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n = len(y_np)
        n_val = max(1, int(round(n * self.val_fraction)))
        n_train = max(1, n - n_val)
        X_train, y_train = X_np[:n_train], y_np[:n_train]
        X_val, y_val = X_np[n_train:], y_np[n_train:]
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_ds, batch_size=min(self.batch_size, len(train_ds)), shuffle=True)
        self.model_ = self._build_model(X_np.shape[1])
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization)
        loss_fn = nn.MSELoss()
        best_state = None
        best_val = float("inf")
        bad = 0
        Xv = torch.from_numpy(X_val)
        yv = torch.from_numpy(y_val)
        for _ in range(self.max_epochs):
            self.model_.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(Xv), yv).item()) if len(X_val) else 0.0
            if val_loss < best_val - 1e-10:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet")
        X_np = self.scaler.transform(pd.DataFrame(X).to_numpy(dtype=np.float32)).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.from_numpy(X_np)).cpu().numpy().reshape(-1)
        return pred.astype(float)


def _new_lob_center_model(cfg: LOBHybridBandConfig):
    _require_sklearn()
    if cfg.center_model_type == "extra_trees_center":
        return ExtraTreesRegressor(
            n_estimators=cfg.n_estimators,
            min_samples_leaf=max(2, cfg.min_samples_leaf // 5),
            max_features=0.8,
            bootstrap=True,
            n_jobs=-1,
            random_state=cfg.random_state,
        )
    if cfg.center_model_type == "gbrt_center":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=cfg.max_iter,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes,
            min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=cfg.l2_regularization,
            random_state=cfg.random_state,
        )
    if cfg.center_model_type == "net_center":
        return _TorchCenterRegressor(
            hidden_layer_sizes=cfg.hidden_layer_sizes,
            max_epochs=cfg.max_iter,
            batch_size=cfg.net_batch_size,
            learning_rate=max(1e-4, cfg.learning_rate * 0.02),
            l2_regularization=max(1e-7, cfg.l2_regularization),
            patience=cfg.net_patience,
            val_fraction=cfg.net_val_fraction,
            random_state=cfg.random_state,
        )
    if cfg.center_model_type == "extra_trees_classifier_center":
        return ExtraTreesClassifier(
            n_estimators=cfg.n_estimators,
            min_samples_leaf=max(2, cfg.min_samples_leaf // 5),
            max_features=0.8,
            bootstrap=True,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=cfg.random_state,
        )
    if cfg.center_model_type == "gbrt_classifier_center":
        return HistGradientBoostingClassifier(
            max_iter=cfg.max_iter,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes,
            min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=cfg.l2_regularization,
            random_state=cfg.random_state,
        )
    if cfg.center_model_type == "net_classifier_center":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=cfg.hidden_layer_sizes,
                        activation="relu",
                        alpha=max(1e-7, cfg.l2_regularization),
                        learning_rate_init=max(1e-4, cfg.learning_rate * 0.02),
                        max_iter=max(100, cfg.max_iter),
                        early_stopping=True,
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )
    raise RuntimeError(f"Unknown center_model_type={cfg.center_model_type}")


def _radius_model_type_for_center(center_model_type: str) -> str:
    if center_model_type in {"net_center", "net_classifier_center"}:
        return "net_radius"
    if center_model_type in {"gbrt_center", "gbrt_classifier_center"}:
        return "gbrt_radius"
    return "extra_trees_radius"


def _calibrate_shrinkage_alpha(
    y_delta: np.ndarray,
    raw_delta_pred: np.ndarray,
    move_threshold: float = 0.0,
    move_weight: float = 0.0,
) -> float:
    """Choose alpha in [0, 1] by weighted calibration MAE.

    For classifier centers, move rows can receive extra weight so the many
    no-move observations do not automatically force alpha to zero. Metrics are
    still reported both unweighted and move-conditional.
    """
    y = np.asarray(y_delta, dtype=float).reshape(-1)
    p = np.asarray(raw_delta_pred, dtype=float).reshape(-1)
    if y.size == 0 or p.size == 0 or np.allclose(p, 0.0):
        return 0.0
    weights = np.ones_like(y, dtype=float)
    if move_weight > 0.0 and move_threshold >= 0.0:
        weights = weights + float(move_weight) * (np.abs(y) > float(move_threshold))
    grid = np.linspace(0.0, 1.0, 101)
    losses = [float(np.average(np.abs(y - a * p), weights=weights)) for a in grid]
    return float(grid[int(np.argmin(losses))])


def _evaluate_hybrid_price_bands(
    y_delta: Sequence[float],
    center_delta: Sequence[float],
    radius: Sequence[float],
    nominal_coverage: float,
    move_threshold: float = 0.0,
) -> Dict[str, float]:
    y = np.asarray(y_delta, dtype=float).reshape(-1)
    c = np.asarray(center_delta, dtype=float).reshape(-1)
    r = np.maximum(np.asarray(radius, dtype=float).reshape(-1), 0.0)
    resid = y - c
    inside = (resid >= -r) & (resid <= r)
    alpha = max(1e-6, 1.0 - float(nominal_coverage))
    naive_mae = float(np.mean(np.abs(y)))
    center_mae = float(np.mean(np.abs(resid)))
    move_mask = np.abs(y) > float(move_threshold)
    if np.any(move_mask):
        naive_move_mae = float(np.mean(np.abs(y[move_mask])))
        center_move_mae = float(np.mean(np.abs(resid[move_mask])))
        move_improvement = float(1.0 - center_move_mae / naive_move_mae) if naive_move_mae > 0 else 0.0
        direction_move = float(np.mean(np.sign(y[move_mask]) == np.sign(c[move_mask])))
    else:
        naive_move_mae = np.nan
        center_move_mae = np.nan
        move_improvement = np.nan
        direction_move = np.nan
    return {
        "nominal_coverage": float(nominal_coverage),
        "empirical_coverage": float(np.mean(inside)),
        "coverage_error": float(np.mean(inside) - nominal_coverage),
        "avg_radius": float(np.mean(r)),
        "avg_interval_width": float(np.mean(2.0 * r)),
        "winkler_score": winkler_score(resid, -r, r, alpha=alpha),
        "center_mae": center_mae,
        "naive_center_mae": naive_mae,
        "center_improvement_vs_naive": float(1.0 - center_mae / naive_mae) if naive_mae > 0 else 0.0,
        "directional_accuracy_raw_center": float(np.mean(np.sign(y) == np.sign(c))),
        "move_rate": float(np.mean(move_mask)),
        "center_mae_move_only": center_move_mae,
        "naive_center_mae_move_only": naive_move_mae,
        "center_improvement_vs_naive_move_only": move_improvement,
        "directional_accuracy_move_only": direction_move,
        "avg_abs_move": float(np.mean(np.abs(y))),
        "max_abs_move": float(np.max(np.abs(y))),
    }


def fit_lob_hybrid_price_band_model(
    lob_clock_df: pd.DataFrame,
    application_start_time: float,
    application_end_time: float,
    cfg: LOBHybridBandConfig,
) -> Dict[str, object]:
    """Fit one full-day LOB hybrid center/radius model for a fixed horizon."""
    cfg.validate()
    frame = _build_lob_supervised_frame(lob_clock_df, cfg)
    cols = _lob_feature_columns(frame)

    # Strictly leakage-safe: training/calibration targets must end before the
    # application/backtest window starts.
    pre_app = frame[frame["target_time"] < float(application_start_time)].copy()
    app = frame[(frame["time"] >= float(application_start_time)) & (frame["target_time"] <= float(application_end_time))].copy()
    if len(pre_app) < 200:
        raise ValueError("Too few pre-application rows. Move the backtest window later or use more LOB data.")
    if len(app) < 20:
        raise ValueError("Too few application rows for this horizon/window alignment.")

    n_cal = max(int(round(len(pre_app) * cfg.calibration_fraction)), 20)
    n_cal = min(n_cal, max(20, len(pre_app) // 2))
    train = pre_app.iloc[: len(pre_app) - n_cal].copy()
    cal = pre_app.iloc[len(pre_app) - n_cal :].copy()

    center_model, center_meta = _fit_lob_center_model(train[cols], train["target_delta"], frame, cfg)

    train_raw_delta, train_center_probs = _predict_lob_center_delta(center_model, train[cols], cfg, center_meta)
    cal_raw_delta, cal_center_probs = _predict_lob_center_delta(center_model, cal[cols], cfg, center_meta)
    app_raw_delta, app_center_probs = _predict_lob_center_delta(center_model, app[cols], cfg, center_meta)

    alpha_shrinkage = _calibrate_shrinkage_alpha(
        cal["target_delta"].to_numpy(),
        cal_raw_delta,
        move_threshold=float(center_meta.get("move_threshold", 0.0)),
        move_weight=float(cfg.alpha_move_weight if _is_classifier_center_type(cfg.center_model_type) else 0.0),
    )

    train_center_delta = alpha_shrinkage * train_raw_delta
    cal_center_delta = alpha_shrinkage * cal_raw_delta
    app_center_delta = alpha_shrinkage * app_raw_delta

    # Radius target is the residual around the hybrid center, not around naive.
    train_abs_resid = np.abs(train["target_delta"].to_numpy(dtype=float) - train_center_delta)
    cal_abs_resid = np.abs(cal["target_delta"].to_numpy(dtype=float) - cal_center_delta)

    radius_cfg = UncertaintyBandConfig(
        horizon_steps=cfg.horizon_steps,
        lookback=cfg.lookback,
        radius_quantile=cfg.radius_quantile,
        test_fraction=0.25,
        model_type=cfg.radius_model_type,
        conformal_alpha=cfg.conformal_alpha,
        calibration_fraction=cfg.calibration_fraction,
        sample_dt_seconds=cfg.sample_dt_seconds,
        horizon_seconds=cfg.horizon_seconds,
        lookback_seconds=cfg.lookback_seconds,
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        net_batch_size=cfg.net_batch_size,
        net_patience=cfg.net_patience,
        net_val_fraction=cfg.net_val_fraction,
    )
    radius_model = RadiusUncertaintyBandForecaster(radius_cfg)
    radius_model.fit(train[cols], train_abs_resid)
    raw_radius_app = radius_model.predict_radius(app[cols], include_conformal=False)
    conformal_padding = None
    if cfg.conformal_alpha is not None:
        conformal_padding = radius_model.calibrate_conformal(cal[cols], cal_abs_resid)
    radius_app = radius_model.predict_radius(app[cols], include_conformal=True)

    pred = pd.DataFrame(index=app.index)
    pred["time"] = app["time"].to_numpy(dtype=float)
    pred["target_time"] = app["target_time"].to_numpy(dtype=float)
    pred["mid"] = app["mid"].to_numpy(dtype=float)
    pred["target_price"] = app["target_price"].to_numpy(dtype=float)
    pred["target_delta"] = app["target_delta"].to_numpy(dtype=float)
    pred["naive_center_price"] = app["mid"].to_numpy(dtype=float)
    pred["raw_center_delta"] = app_raw_delta
    if not app_center_probs.empty:
        for prob_col in app_center_probs.columns:
            pred[prob_col] = app_center_probs[prob_col].to_numpy(dtype=float)
    pred["alpha_shrinkage"] = alpha_shrinkage
    pred["center_delta"] = app_center_delta
    pred["center_price"] = pred["mid"] + pred["center_delta"]
    pred["pred_radius"] = radius_app.to_numpy(dtype=float)
    pred["raw_pred_radius"] = raw_radius_app.to_numpy(dtype=float)
    pred["lower_price"] = pred["center_price"] - pred["pred_radius"]
    pred["upper_price"] = pred["center_price"] + pred["pred_radius"]

    metrics = _evaluate_hybrid_price_bands(
        pred["target_delta"],
        pred["center_delta"],
        pred["pred_radius"],
        nominal_coverage=cfg.nominal_coverage,
        move_threshold=float(center_meta.get("move_threshold", 0.0)),
    )
    metrics["alpha_shrinkage"] = float(alpha_shrinkage)
    metrics["conformal_padding"] = float(conformal_padding or 0.0)
    metrics["center_model_type"] = cfg.center_model_type
    metrics["radius_model_type"] = cfg.radius_model_type
    metrics.update({k: v for k, v in center_meta.items() if isinstance(v, (int, float, str, np.floating))})

    return {
        "config": cfg,
        "config_dict": asdict(cfg),
        "frame": frame,
        "train_frame": train,
        "calibration_frame": cal,
        "test_frame": app,
        "application_frame": app,
        "features": cols,
        "center_model": center_model,
        "center_meta": center_meta,
        "radius_model": radius_model,
        "pred_frame": pred.reset_index(drop=True),
        "pred_price": pred.reset_index(drop=True),
        "metrics": metrics,
        "alpha_shrinkage": alpha_shrinkage,
        "conformal_padding": conformal_padding,
    }


def fit_lob_hybrid_price_bands_multi_horizon(
    lob_clock_df: pd.DataFrame,
    application_start_time: float,
    application_end_time: float,
    sample_dt_seconds: float = 0.1,
    horizons_seconds: Sequence[float] = (1.0, 5.0, 10.0),
    lookback_seconds: float = 30.0,
    center_model_types: Sequence[str] = ("extra_trees_classifier_center", "gbrt_classifier_center"),
    radius_quantile: float = 0.90,
    conformal_alpha: Optional[float] = 0.10,
    calibration_fraction: float = 0.20,
    max_iter: int = 80,
    n_estimators: int = 150,
    random_state: int = 42,
    skip_failed_models: bool = True,
) -> Dict[str, object]:
    """Train full-day LOB hybrid center/radius models and select per horizon."""
    rows = []
    all_results: Dict[float, Dict[str, Dict[str, object]]] = {}
    selected: Dict[float, Dict[str, object]] = {}
    lookback_steps = steps_from_seconds(lookback_seconds, sample_dt_seconds)

    for h in horizons_seconds:
        h = float(h)
        horizon_steps = steps_from_seconds(h, sample_dt_seconds)
        all_results[h] = {}
        for center_model_type in center_model_types:
            radius_model_type = _radius_model_type_for_center(center_model_type)
            cfg = LOBHybridBandConfig(
                horizon_steps=horizon_steps,
                lookback=lookback_steps,
                center_model_type=center_model_type,
                radius_model_type=radius_model_type,
                radius_quantile=radius_quantile,
                conformal_alpha=conformal_alpha,
                calibration_fraction=calibration_fraction,
                sample_dt_seconds=sample_dt_seconds,
                horizon_seconds=h,
                lookback_seconds=lookback_seconds,
                max_iter=max_iter,
                n_estimators=n_estimators,
                random_state=random_state,
            )
            try:
                result = fit_lob_hybrid_price_band_model(
                    lob_clock_df=lob_clock_df,
                    application_start_time=application_start_time,
                    application_end_time=application_end_time,
                    cfg=cfg,
                )
                all_results[h][center_model_type] = result
                rows.append(
                    {
                        "horizon_seconds": h,
                        "horizon_steps": horizon_steps,
                        "center_model_type": center_model_type,
                        "radius_model_type": radius_model_type,
                        **result["metrics"],
                    }
                )
            except Exception as exc:
                if not skip_failed_models:
                    raise
                rows.append(
                    {
                        "horizon_seconds": h,
                        "horizon_steps": horizon_steps,
                        "center_model_type": center_model_type,
                        "radius_model_type": radius_model_type,
                        "failed": True,
                        "error": repr(exc),
                    }
                )

        valid = [r for r in all_results[h].values() if "metrics" in r]
        if not valid:
            raise RuntimeError(f"All LOB hybrid models failed for horizon {h:g}s")
        # Primary goal: good uncertainty interval; secondary: good center MAE.
        best = min(valid, key=lambda r: (r["metrics"]["winkler_score"], r["metrics"]["center_mae"]))
        selected[h] = best

    candidate_table = pd.DataFrame(rows)
    selected_rows = []
    for h, result in selected.items():
        selected_rows.append(
            {
                "horizon_seconds": h,
                "horizon_steps": result["config"].horizon_steps,
                "selected_center_model_type": result["config"].center_model_type,
                "selected_radius_model_type": result["config"].radius_model_type,
                **result["metrics"],
            }
        )
    metrics_table = pd.DataFrame(selected_rows).sort_values("horizon_seconds").reset_index(drop=True)
    if not candidate_table.empty and "failed" in candidate_table.columns:
        candidate_table["failed"] = candidate_table["failed"].fillna(False)
    return {
        "results_by_horizon": selected,
        "all_results_by_horizon": all_results,
        "metrics_table": metrics_table,
        "candidate_table": candidate_table,
    }


def plot_lob_hybrid_center_vs_realized(
    result: Dict[str, object],
    max_points: int = 1200,
    title: Optional[str] = None,
):
    """Plot realized future mid against naive and hybrid learned centers."""
    pred = result.get("pred_frame", result.get("pred_price"))
    cfg: LOBHybridBandConfig = result["config"]
    n = min(max_points, len(pred))
    pf = pred.iloc[:n]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = pf["target_time"].to_numpy(dtype=float)
    ax.plot(x, pf["target_price"].to_numpy(dtype=float), label="realized future mid", linewidth=1.3)
    ax.plot(x, pf["naive_center_price"].to_numpy(dtype=float), label="naive center: current mid", linewidth=1.0, alpha=0.75)
    ax.plot(x, pf["center_price"].to_numpy(dtype=float), label="hybrid learned center", linewidth=1.2)
    ax.fill_between(
        x,
        pf["lower_price"].to_numpy(dtype=float),
        pf["upper_price"].to_numpy(dtype=float),
        alpha=0.18,
        label=f"{int(round(cfg.nominal_coverage * 100))}% hybrid prediction band",
    )
    h_label = f"{cfg.horizon_seconds:g}s" if cfg.horizon_seconds is not None else f"{cfg.horizon_steps} steps"
    ax.set_title(title or f"Full-day LOB hybrid center vs realized mid, horizon {h_label}")
    ax.set_xlabel("Target time, seconds")
    ax.set_ylabel("Mid-price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def _predict_lob_hybrid_origin(result: Dict[str, object], origin_time: float) -> Dict[str, float]:
    pred = result.get("pred_frame", result.get("pred_price"))
    idx = int(np.argmin(np.abs(pred["time"].to_numpy(dtype=float) - float(origin_time))))
    row = pred.iloc[idx]
    return {
        "time": float(row["time"]),
        "target_time": float(row["target_time"]),
        "mid": float(row["mid"]),
        "center_price": float(row["center_price"]),
        "lower_price": float(row["lower_price"]),
        "upper_price": float(row["upper_price"]),
        "target_price": float(row["target_price"]),
    }


def plot_lob_hybrid_forecast_fan(
    results_by_horizon: Dict[float, Dict[str, object]],
    lob_clock_df: pd.DataFrame,
    history_seconds: float = 60.0,
    symbol: Optional[str] = None,
    origin_time: Optional[float] = None,
    title: Optional[str] = None,
):
    """Plot recent original LOB mid plus +1s/+5s/+10s hybrid forecast bands."""
    horizons = sorted(float(h) for h in results_by_horizon.keys())
    if not horizons:
        raise ValueError("results_by_horizon is empty")
    max_h = max(horizons)
    base_pred = results_by_horizon[max_h].get("pred_frame", results_by_horizon[max_h].get("pred_price"))
    if origin_time is None:
        origin_time = float(base_pred["time"].iloc[-1])
    origin_time = float(origin_time)

    lob = pd.DataFrame(lob_clock_df).copy()
    hist = lob[(lob["time"] >= origin_time - history_seconds) & (lob["time"] <= origin_time)]
    if hist.empty:
        hist = lob.iloc[-min(len(lob), 1000) :]
    mid_now = float(hist["mid"].iloc[-1])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist["time"], hist["mid"], label=f"last {history_seconds:g}s LOB mid", linewidth=1.4)
    ax.scatter([origin_time], [mid_now], s=45, label="forecast origin")

    future_times = []
    centers = []
    lows = []
    highs = []
    realized = []
    for h in horizons:
        p = _predict_lob_hybrid_origin(results_by_horizon[h], origin_time)
        future_times.append(p["target_time"])
        centers.append(p["center_price"])
        lows.append(p["lower_price"])
        highs.append(p["upper_price"])
        realized.append(p["target_price"])
        ax.vlines(p["target_time"], p["lower_price"], p["upper_price"], linewidth=5, alpha=0.35, label=f"{h:g}s band")
        ax.scatter([p["target_time"]], [p["center_price"]], s=55, label=f"{h:g}s hybrid center")
        ax.scatter([p["target_time"]], [p["target_price"]], s=45, marker="x", label=f"{h:g}s realized")

    ax.plot([origin_time] + future_times, [mid_now] + centers, linestyle="--", linewidth=1.2, label="hybrid center path")
    ax.fill_between([origin_time] + future_times, [mid_now] + lows, [mid_now] + highs, alpha=0.12, label="forecast fan visual aid")
    name = f"{symbol}: " if symbol else ""
    ax.set_title(title or f"{name}full-day LOB hybrid forecast fan")
    ax.set_xlabel("Time, seconds")
    ax.set_ylabel("Mid-price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig, ax
