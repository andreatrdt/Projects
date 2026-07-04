from __future__ import annotations

"""Rolling-window full-day LOB mid-price forecasting.

This module mirrors the rolling-window construction used in the user's
energy-price forecasting project. Every sample is a fixed historical LOB
sequence; overlapping chronological windows form the train/validation/
calibration sets, and the final backtest interval is held out for application.

The model uses a shared 1D CNN with direct 1s/5s/10s classification heads and
uncertainty-radius heads. It never shuffles observations across time.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None

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

def winkler_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """Mean Winkler interval score for nominal coverage 1-alpha."""
    width = upper - lower
    below = y < lower
    above = y > upper
    score = width.copy()
    score[below] += (2.0 / alpha) * (lower[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - upper[above])
    return float(np.mean(score))

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

# ---------------------------------------------------------------------------
# Rolling-window full-day LOB sequence forecaster
# ---------------------------------------------------------------------------

ROLLING_SEQUENCE_API_VERSION = "2026.07.04-rolling-lob-cnn-v1"


@dataclass(frozen=True)
class RollingLOBSequenceConfig:
    """Configuration for the rolling-window LOB sequence model.

    The construction mirrors the rolling-window design used in the energy
    forecasting project: every forecast origin receives a fixed historical
    window, and those overlapping windows form the chronological training set.

    The original LOB is kept on ``sample_dt_seconds``. To keep a seven-minute
    sequence computationally manageable, observations *inside* each input
    window can be sampled more coarsely using ``sequence_sample_seconds``.
    Forecast targets still use the original 0.1-second clock grid.
    """

    sample_dt_seconds: float = 0.1
    rolling_window_minutes: float = 7.0
    sequence_sample_seconds: float = 0.5
    training_origin_stride_seconds: float = 1.0
    application_origin_stride_seconds: float = 0.1
    horizons_seconds: Tuple[float, ...] = (1.0, 5.0, 10.0)

    calibration_fraction: float = 0.15
    validation_fraction: float = 0.15
    deadzone_ticks: float = 0.5
    radius_quantile: float = 0.90
    conformal_alpha: Optional[float] = 0.10
    move_weight_for_alpha: float = 5.0
    class_weight_power: float = 0.5

    hidden_channels: Tuple[int, ...] = (32, 64, 128)
    dropout: float = 0.10
    max_epochs: int = 15
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    radius_loss_weight: float = 0.5

    random_state: int = 42
    device: str = "auto"
    num_workers: int = 0

    def validate(self) -> None:
        if self.sample_dt_seconds <= 0:
            raise ValueError("sample_dt_seconds must be positive")
        if self.rolling_window_minutes <= 0:
            raise ValueError("rolling_window_minutes must be positive")
        if self.sequence_sample_seconds < self.sample_dt_seconds:
            raise ValueError("sequence_sample_seconds cannot be finer than sample_dt_seconds")
        if self.training_origin_stride_seconds < self.sample_dt_seconds:
            raise ValueError("training_origin_stride_seconds cannot be finer than sample_dt_seconds")
        if self.application_origin_stride_seconds < self.sample_dt_seconds:
            raise ValueError("application_origin_stride_seconds cannot be finer than sample_dt_seconds")
        if not self.horizons_seconds or any(float(h) <= 0 for h in self.horizons_seconds):
            raise ValueError("horizons_seconds must contain positive values")
        if not (0.05 <= self.calibration_fraction <= 0.4):
            raise ValueError("calibration_fraction must be in [0.05, 0.4]")
        if not (0.05 <= self.validation_fraction <= 0.4):
            raise ValueError("validation_fraction must be in [0.05, 0.4]")
        if self.calibration_fraction + self.validation_fraction >= 0.8:
            raise ValueError("validation + calibration fractions leave too little training data")
        if self.deadzone_ticks < 0:
            raise ValueError("deadzone_ticks must be non-negative")
        if not (0.5 < self.radius_quantile < 0.999):
            raise ValueError("radius_quantile must be in (0.5, 0.999)")
        if self.conformal_alpha is not None and not (0.01 < self.conformal_alpha < 0.5):
            raise ValueError("conformal_alpha must be None or in (0.01, 0.5)")
        if self.max_epochs < 1 or self.batch_size < 1:
            raise ValueError("max_epochs and batch_size must be positive")
        if not self.hidden_channels:
            raise ValueError("hidden_channels cannot be empty")

    @property
    def rolling_window_seconds(self) -> float:
        return 60.0 * float(self.rolling_window_minutes)

    @property
    def nominal_coverage(self) -> float:
        if self.conformal_alpha is not None:
            return 1.0 - float(self.conformal_alpha)
        return float(self.radius_quantile)


def _integer_ratio(value: float, base: float, name: str) -> int:
    ratio = float(value) / float(base)
    rounded = int(round(ratio))
    if rounded < 1 or not np.isclose(ratio, rounded, atol=1e-7, rtol=1e-7):
        raise ValueError(f"{name}={value} must be an integer multiple of sample_dt_seconds={base}")
    return rounded


def _estimate_price_tick_from_lob(lob_clock_df: pd.DataFrame) -> float:
    """Estimate a robust positive price tick from bid/ask changes and spread."""
    df = pd.DataFrame(lob_clock_df)
    candidates = []
    for col in ("bid", "ask"):
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            d = np.abs(np.diff(x))
            d = d[np.isfinite(d) & (d > 1e-10)]
            if d.size:
                candidates.append(d)
    if {"bid", "ask"}.issubset(df.columns):
        spread = (
            pd.to_numeric(df["ask"], errors="coerce")
            - pd.to_numeric(df["bid"], errors="coerce")
        ).to_numpy(dtype=float)
        spread = spread[np.isfinite(spread) & (spread > 1e-10)]
        if spread.size:
            candidates.append(spread)
    if not candidates:
        return 0.01
    values = np.concatenate(candidates)
    # The lower positive tail is more robust than the absolute minimum, which
    # may contain floating-point noise.
    q = float(np.quantile(values, 0.05))
    return max(q, 1e-6)


def prepare_rolling_lob_sequence_features(
    lob_clock_df: pd.DataFrame,
    price_tick: Optional[float] = None,
) -> Tuple[pd.DataFrame, list[str], float]:
    """Create stationary LOB features for the sequence network.

    Absolute price level is retained only for constructing targets and plots.
    Network inputs are expressed in ticks, basis points, imbalances and log
    depth so that an intraday trend cannot be learned as a spurious price-level
    shortcut.
    """
    df = pd.DataFrame(lob_clock_df).copy().sort_values("time").reset_index(drop=True)
    required = {"time", "mid", "bid", "ask"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"lob_clock_df is missing required columns: {sorted(missing)}")

    tick = float(price_tick) if price_tick is not None else _estimate_price_tick_from_lob(df)
    if not np.isfinite(tick) or tick <= 0:
        raise ValueError("price_tick must be a positive finite number")

    for col in ("time", "mid", "bid", "ask"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "spread" not in df.columns:
        df["spread"] = df["ask"] - df["bid"]
    if "dmid" not in df.columns:
        df["dmid"] = df["mid"].diff()
    if "abs_dmid" not in df.columns:
        df["abs_dmid"] = df["dmid"].abs()
    if "microprice_minus_mid" not in df.columns:
        if "microprice" in df.columns:
            df["microprice_minus_mid"] = df["microprice"] - df["mid"]
        else:
            df["microprice_minus_mid"] = 0.0
    if "top_imbalance" not in df.columns:
        if {"bid_size_1", "ask_size_1"}.issubset(df.columns):
            denom = df["bid_size_1"] + df["ask_size_1"]
            df["top_imbalance"] = (df["bid_size_1"] - df["ask_size_1"]) / (denom.abs() + 1e-12)
        else:
            df["top_imbalance"] = 0.0

    dmid_ticks = df["dmid"] / tick
    spread_ticks = df["spread"] / tick
    micro_offset_ticks = df["microprice_minus_mid"] / tick
    ret_bps = df["mid"].pct_change() * 1e4

    features = pd.DataFrame(index=df.index)
    features["dmid_ticks"] = dmid_ticks
    features["abs_dmid_ticks"] = dmid_ticks.abs()
    features["return_bps"] = ret_bps
    features["spread_ticks"] = spread_ticks
    features["spread_change_ticks"] = spread_ticks.diff()
    features["top_imbalance"] = pd.to_numeric(df["top_imbalance"], errors="coerce")
    features["microprice_offset_ticks"] = micro_offset_ticks

    bid_size = pd.to_numeric(df.get("bid_size_1", 0.0), errors="coerce")
    ask_size = pd.to_numeric(df.get("ask_size_1", 0.0), errors="coerce")
    if not isinstance(bid_size, pd.Series):
        bid_size = pd.Series(np.full(len(df), float(bid_size)), index=df.index)
    if not isinstance(ask_size, pd.Series):
        ask_size = pd.Series(np.full(len(df), float(ask_size)), index=df.index)
    features["log_bid_size_1"] = np.log1p(np.maximum(bid_size, 0.0))
    features["log_ask_size_1"] = np.log1p(np.maximum(ask_size, 0.0))
    features["depth_log_ratio"] = features["log_bid_size_1"] - features["log_ask_size_1"]

    # Existing multi-level imbalance from the OrderBook wrapper is useful when
    # available. Fall back to top-of-book imbalance otherwise.
    for source, target in (
        ("rho_5L", "rho_5L"),
        ("rho_1L", "rho_1L"),
        ("depth_imbalance", "depth_imbalance"),
    ):
        if source in df.columns:
            features[target] = pd.to_numeric(df[source], errors="coerce")

    # Small clock-time summaries complement the CNN without creating thousands
    # of manually lagged columns.
    dt = infer_time_step_seconds(df["time"].to_numpy(dtype=float))
    w1 = max(2, int(round(1.0 / dt)))
    w5 = max(w1 + 1, int(round(5.0 / dt)))
    features["vol_1s_ticks"] = dmid_ticks.rolling(w1, min_periods=2).std()
    features["vol_5s_ticks"] = dmid_ticks.rolling(w5, min_periods=2).std()
    features["imbalance_mean_1s"] = features["top_imbalance"].rolling(w1, min_periods=1).mean()
    features["imbalance_mean_5s"] = features["top_imbalance"].rolling(w5, min_periods=1).mean()
    features["signed_move_1s_ticks"] = dmid_ticks.rolling(w1, min_periods=1).sum()
    features["signed_move_5s_ticks"] = dmid_ticks.rolling(w5, min_periods=1).sum()

    feature_cols = list(features.columns)
    out = pd.concat([df[["time", "mid", "bid", "ask"]], features], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    # Forward fill state-like columns, then set the tiny initial derivative gap
    # to zero. No future values are used.
    out[feature_cols] = out[feature_cols].ffill().fillna(0.0)
    out = out.dropna(subset=["time", "mid", "bid", "ask"]).reset_index(drop=True)
    return out, feature_cols, tick


class _RollingLOBWindowDataset(torch.utils.data.Dataset if torch is not None else object):
    """Lazy rolling-window dataset; sequences are sliced only when requested."""

    def __init__(
        self,
        feature_matrix: np.ndarray,
        mid: np.ndarray,
        origin_indices: np.ndarray,
        window_intervals: int,
        sequence_step: int,
        horizon_steps: Sequence[int],
        price_tick: float,
        deadzone_ticks: float,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for rolling sequence forecasting")
        self.X = np.asarray(feature_matrix, dtype=np.float32)
        self.mid = np.asarray(mid, dtype=np.float64)
        self.origins = np.asarray(origin_indices, dtype=np.int64)
        self.window_intervals = int(window_intervals)
        self.sequence_step = int(sequence_step)
        self.horizon_steps = np.asarray(horizon_steps, dtype=np.int64)
        self.price_tick = float(price_tick)
        self.deadzone_ticks = float(deadzone_ticks)

    def __len__(self) -> int:
        return int(self.origins.size)

    def __getitem__(self, item: int):
        origin = int(self.origins[item])
        start = origin - self.window_intervals
        seq = self.X[start : origin + 1 : self.sequence_step]
        # [features, time] for Conv1d.
        seq = np.ascontiguousarray(seq.T, dtype=np.float32)
        future = self.mid[origin + self.horizon_steps]
        delta_ticks = ((future - self.mid[origin]) / self.price_tick).astype(np.float32)
        labels = np.ones(len(self.horizon_steps), dtype=np.int64)
        labels[delta_ticks > self.deadzone_ticks] = 2
        labels[delta_ticks < -self.deadzone_ticks] = 0
        return (
            torch.from_numpy(seq),
            torch.from_numpy(labels),
            torch.from_numpy(delta_ticks),
            torch.tensor(origin, dtype=torch.long),
        )


class _RollingLOBMultiHorizonCNN(nn.Module if nn is not None else object):
    """Shared temporal CNN with direct classification and radius heads."""

    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        hidden_channels: Sequence[int],
        dropout: float,
    ):
        if nn is None:
            raise ImportError("PyTorch is required for rolling sequence forecasting")
        super().__init__()
        layers = []
        in_ch = int(n_features)
        for out_ch in hidden_channels:
            out_ch = int(out_ch)
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.encoder = nn.Sequential(*layers)
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, in_ch),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.class_heads = nn.ModuleList([nn.Linear(in_ch, 3) for _ in range(int(n_horizons))])
        self.radius_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_ch, 1), nn.Softplus()) for _ in range(int(n_horizons))]
        )

    def forward(self, x: torch.Tensor):
        z = self.shared(self.encoder(x))
        logits = torch.stack([head(z) for head in self.class_heads], dim=1)
        radius = torch.cat([head(z) for head in self.radius_heads], dim=1)
        return logits, radius


def _resolve_torch_device(requested: str) -> torch.device:
    if torch is None:
        raise ImportError("PyTorch is required for rolling sequence forecasting")
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU", RuntimeWarning)
        return torch.device("cpu")
    return device


def _build_origin_splits(
    times: np.ndarray,
    window_intervals: int,
    horizon_steps: Sequence[int],
    application_start_time: float,
    application_end_time: float,
    training_stride_steps: int,
    application_stride_steps: int,
    validation_fraction: float,
    calibration_fraction: float,
) -> Dict[str, np.ndarray]:
    max_h = int(max(horizon_steps))
    n = len(times)
    earliest = int(window_intervals)
    latest = n - max_h - 1
    if latest <= earliest:
        raise ValueError("Not enough rows for the rolling window and forecast horizons")

    train_candidates = np.arange(earliest, latest + 1, int(training_stride_steps), dtype=np.int64)
    train_candidates = train_candidates[times[train_candidates + max_h] < float(application_start_time)]

    app_candidates = np.arange(earliest, latest + 1, int(application_stride_steps), dtype=np.int64)
    app_candidates = app_candidates[
        (times[app_candidates] >= float(application_start_time))
        & (times[app_candidates + max_h] <= float(application_end_time))
    ]

    if len(train_candidates) < 300:
        raise ValueError(
            f"Only {len(train_candidates)} pre-application rolling samples are available. "
            "Move the application window later, shorten the rolling window, or use a finer origin stride."
        )
    if len(app_candidates) < 20:
        raise ValueError("Too few application-window forecast origins")

    n_cal = max(50, int(round(len(train_candidates) * calibration_fraction)))
    train_val = train_candidates[:-n_cal]
    calibration = train_candidates[-n_cal:]
    n_val = max(50, int(round(len(train_val) * validation_fraction)))
    train = train_val[:-n_val]
    validation = train_val[-n_val:]
    if len(train) < 100:
        raise ValueError("Too few training windows after chronological splitting")

    return {
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "application": app_candidates,
    }


def _compute_sequence_scaler(
    feature_values: np.ndarray,
    training_origins: np.ndarray,
    window_intervals: int,
) -> Tuple[np.ndarray, np.ndarray]:
    first = max(0, int(training_origins[0]) - int(window_intervals))
    last = int(training_origins[-1]) + 1
    sample = np.asarray(feature_values[first:last], dtype=np.float64)
    mean = np.nanmean(sample, axis=0)
    std = np.nanstd(sample, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _make_sequence_loader(
    dataset: _RollingLOBWindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(42)
    return DataLoader(
        dataset,
        batch_size=min(int(batch_size), max(1, len(dataset))),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def _class_weights_by_horizon(
    dataset: _RollingLOBWindowDataset,
    power: float,
) -> list[torch.Tensor]:
    origins = dataset.origins
    future = dataset.mid[origins[:, None] + dataset.horizon_steps[None, :]]
    delta_ticks = (future - dataset.mid[origins, None]) / dataset.price_tick
    labels = np.ones_like(delta_ticks, dtype=np.int64)
    labels[delta_ticks > dataset.deadzone_ticks] = 2
    labels[delta_ticks < -dataset.deadzone_ticks] = 0
    weights = []
    for j in range(labels.shape[1]):
        counts = np.bincount(labels[:, j], minlength=3).astype(float)
        raw = (counts.sum() / np.maximum(counts, 1.0)) ** float(power)
        raw = raw / np.mean(raw)
        raw = np.clip(raw, 0.25, 6.0)
        weights.append(torch.tensor(raw, dtype=torch.float32))
    return weights


def _pinball_torch(target: torch.Tensor, pred: torch.Tensor, q: float) -> torch.Tensor:
    err = target - pred
    return torch.maximum(float(q) * err, (float(q) - 1.0) * err).mean()


def _sequence_loss(
    logits: torch.Tensor,
    radius_ticks: torch.Tensor,
    labels: torch.Tensor,
    delta_ticks: torch.Tensor,
    class_weights: Sequence[torch.Tensor],
    move_sizes_ticks: torch.Tensor,
    radius_quantile: float,
    radius_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce = torch.zeros((), device=logits.device)
    for j in range(logits.shape[1]):
        ce = ce + nn.functional.cross_entropy(
            logits[:, j, :], labels[:, j], weight=class_weights[j].to(logits.device)
        )
    ce = ce / logits.shape[1]

    probs = torch.softmax(logits, dim=-1)
    raw_center_ticks = move_sizes_ticks.to(logits.device)[None, :] * (probs[:, :, 2] - probs[:, :, 0])
    residual_abs_ticks = torch.abs(delta_ticks - raw_center_ticks.detach())
    radius_loss = _pinball_torch(residual_abs_ticks, radius_ticks, radius_quantile)
    total = ce + float(radius_loss_weight) * radius_loss
    return total, {"classification_loss": float(ce.detach().cpu()), "radius_loss": float(radius_loss.detach().cpu())}


def _predict_sequence_dataset(
    model: _RollingLOBMultiHorizonCNN,
    dataset: _RollingLOBWindowDataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> Dict[str, np.ndarray]:
    loader = _make_sequence_loader(dataset, batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    logits_all, radius_all, labels_all, delta_all, origins_all = [], [], [], [], []
    with torch.no_grad():
        for xb, labels, delta_ticks, origins in loader:
            xb = xb.to(device, non_blocking=True)
            logits, radius = model(xb)
            logits_all.append(logits.cpu().numpy())
            radius_all.append(radius.cpu().numpy())
            labels_all.append(labels.numpy())
            delta_all.append(delta_ticks.numpy())
            origins_all.append(origins.numpy())
    logits = np.concatenate(logits_all, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    return {
        "logits": logits,
        "probabilities": probs,
        "radius_ticks_raw": np.concatenate(radius_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "delta_ticks": np.concatenate(delta_all, axis=0),
        "origins": np.concatenate(origins_all, axis=0),
    }


def _select_shrinkage_alpha(
    y_ticks: np.ndarray,
    raw_center_ticks: np.ndarray,
    labels: np.ndarray,
    move_weight: float,
) -> Tuple[float, float]:
    grid = np.linspace(0.0, 1.0, 101)
    weights = 1.0 + float(move_weight) * (labels != 1)
    losses = np.array(
        [np.average(np.abs(y_ticks - a * raw_center_ticks), weights=weights) for a in grid],
        dtype=float,
    )
    best = int(np.argmin(losses))
    return float(grid[best]), float(losses[best])


def _conformal_radius_padding(
    residual_abs_ticks: np.ndarray,
    raw_radius_ticks: np.ndarray,
    alpha: Optional[float],
) -> float:
    if alpha is None:
        return 0.0
    scores = np.asarray(residual_abs_ticks, dtype=float) - np.asarray(raw_radius_ticks, dtype=float)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1.0 - float(alpha))) / n)
    return max(0.0, float(np.quantile(scores, level, method="higher")))


def fit_rolling_lob_sequence_forecaster(
    lob_clock_df: pd.DataFrame,
    application_start_time: float,
    application_end_time: float,
    config: Optional[RollingLOBSequenceConfig] = None,
    price_tick: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train and evaluate a rolling-window multi-horizon LOB CNN.

    Every sample is a seven-minute (configurable) historical LOB sequence. The
    split is chronological and purged: all training/calibration target times
    finish strictly before the application/backtest window begins.
    """
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for fit_rolling_lob_sequence_forecaster")
    cfg = config or RollingLOBSequenceConfig()
    cfg.validate()

    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_state)

    frame, feature_cols, tick = prepare_rolling_lob_sequence_features(lob_clock_df, price_tick=price_tick)
    times = frame["time"].to_numpy(dtype=float)
    mid = frame["mid"].to_numpy(dtype=float)

    base_dt = infer_time_step_seconds(times)
    if not np.isclose(base_dt, cfg.sample_dt_seconds, atol=1e-5, rtol=1e-4):
        warnings.warn(
            f"Configured sample_dt_seconds={cfg.sample_dt_seconds}, but inferred grid is {base_dt:.8g}. "
            "Using the inferred grid for index conversion.",
            RuntimeWarning,
        )
    window_intervals = _integer_ratio(cfg.rolling_window_seconds, base_dt, "rolling_window_seconds")
    sequence_step = _integer_ratio(cfg.sequence_sample_seconds, base_dt, "sequence_sample_seconds")
    train_stride = _integer_ratio(cfg.training_origin_stride_seconds, base_dt, "training_origin_stride_seconds")
    app_stride = _integer_ratio(cfg.application_origin_stride_seconds, base_dt, "application_origin_stride_seconds")
    horizon_steps = [_integer_ratio(float(h), base_dt, f"horizon {h}") for h in cfg.horizons_seconds]

    splits = _build_origin_splits(
        times=times,
        window_intervals=window_intervals,
        horizon_steps=horizon_steps,
        application_start_time=float(application_start_time),
        application_end_time=float(application_end_time),
        training_stride_steps=train_stride,
        application_stride_steps=app_stride,
        validation_fraction=cfg.validation_fraction,
        calibration_fraction=cfg.calibration_fraction,
    )

    raw_features = frame[feature_cols].to_numpy(dtype=np.float32)
    scaler_mean, scaler_std = _compute_sequence_scaler(raw_features, splits["train"], window_intervals)
    feature_values = np.clip((raw_features - scaler_mean) / scaler_std, -12.0, 12.0).astype(np.float32)

    datasets = {
        name: _RollingLOBWindowDataset(
            feature_matrix=feature_values,
            mid=mid,
            origin_indices=origins,
            window_intervals=window_intervals,
            sequence_step=sequence_step,
            horizon_steps=horizon_steps,
            price_tick=tick,
            deadzone_ticks=cfg.deadzone_ticks,
        )
        for name, origins in splits.items()
    }

    train_ds = datasets["train"]
    origins = train_ds.origins
    future = train_ds.mid[origins[:, None] + train_ds.horizon_steps[None, :]]
    train_delta_ticks = (future - train_ds.mid[origins, None]) / tick
    train_labels = np.ones_like(train_delta_ticks, dtype=np.int64)
    train_labels[train_delta_ticks > cfg.deadzone_ticks] = 2
    train_labels[train_delta_ticks < -cfg.deadzone_ticks] = 0
    move_sizes_ticks = []
    for j in range(train_delta_ticks.shape[1]):
        moved = np.abs(train_delta_ticks[:, j][train_labels[:, j] != 1])
        move_sizes_ticks.append(float(np.median(moved)) if moved.size else 1.0)
    move_sizes_ticks_arr = np.maximum(np.asarray(move_sizes_ticks, dtype=np.float32), 0.5)

    class_weights = _class_weights_by_horizon(train_ds, cfg.class_weight_power)
    device = _resolve_torch_device(cfg.device)
    model = _RollingLOBMultiHorizonCNN(
        n_features=len(feature_cols),
        n_horizons=len(horizon_steps),
        hidden_channels=cfg.hidden_channels,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    train_loader = _make_sequence_loader(
        train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = _make_sequence_loader(
        datasets["validation"], cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    move_sizes_tensor = torch.tensor(move_sizes_ticks_arr, dtype=torch.float32, device=device)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history_rows = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses = []
        for xb, labels, delta_ticks, _origins in train_loader:
            xb = xb.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            delta_ticks = delta_ticks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, radius = model(xb)
            loss, _parts = _sequence_loss(
                logits,
                radius,
                labels,
                delta_ticks,
                class_weights,
                move_sizes_tensor,
                cfg.radius_quantile,
                cfg.radius_loss_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, labels, delta_ticks, _origins in val_loader:
                xb = xb.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                delta_ticks = delta_ticks.to(device, non_blocking=True)
                logits, radius = model(xb)
                loss, _parts = _sequence_loss(
                    logits,
                    radius,
                    labels,
                    delta_ticks,
                    class_weights,
                    move_sizes_tensor,
                    cfg.radius_quantile,
                    cfg.radius_loss_weight,
                )
                val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if verbose:
            print(f"[rolling CNN] epoch {epoch:02d}/{cfg.max_epochs} train={train_loss:.5f} val={val_loss:.5f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if verbose:
                    print(f"[rolling CNN] early stopping after epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    cal_pred = _predict_sequence_dataset(
        model, datasets["calibration"], cfg.batch_size, device, cfg.num_workers
    )
    app_pred = _predict_sequence_dataset(
        model, datasets["application"], cfg.batch_size, device, cfg.num_workers
    )

    results_by_horizon: Dict[float, Dict[str, object]] = {}
    metric_rows = []
    alpha_values = []
    conformal_values = []

    # Metrics imported locally to keep the legacy part of this module stable.
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    for j, h in enumerate(cfg.horizons_seconds):
        h = float(h)
        move_size_ticks = float(move_sizes_ticks_arr[j])

        cal_probs = cal_pred["probabilities"][:, j, :]
        cal_edge = cal_probs[:, 2] - cal_probs[:, 0]
        cal_raw_center_ticks = move_size_ticks * cal_edge
        alpha_h, _ = _select_shrinkage_alpha(
            cal_pred["delta_ticks"][:, j],
            cal_raw_center_ticks,
            cal_pred["labels"][:, j],
            cfg.move_weight_for_alpha,
        )
        cal_residual_abs = np.abs(cal_pred["delta_ticks"][:, j] - alpha_h * cal_raw_center_ticks)
        padding_ticks = _conformal_radius_padding(
            cal_residual_abs,
            cal_pred["radius_ticks_raw"][:, j],
            cfg.conformal_alpha,
        )
        alpha_values.append(alpha_h)
        conformal_values.append(padding_ticks)

        probs = app_pred["probabilities"][:, j, :]
        labels = app_pred["labels"][:, j]
        target_delta_ticks = app_pred["delta_ticks"][:, j]
        edge = probs[:, 2] - probs[:, 0]
        raw_center_ticks = move_size_ticks * edge
        center_ticks = alpha_h * raw_center_ticks
        radius_ticks = np.maximum(app_pred["radius_ticks_raw"][:, j] + padding_ticks, 0.0)

        origin_idx = app_pred["origins"].astype(int)
        target_idx = origin_idx + int(horizon_steps[j])
        mid_now = mid[origin_idx]
        target_price = mid[target_idx]
        center_price = mid_now + center_ticks * tick
        radius_price = radius_ticks * tick
        lower_price = center_price - radius_price
        upper_price = center_price + radius_price

        pred_class = np.argmax(probs, axis=1)
        naive_mae = float(np.mean(np.abs(target_price - mid_now)))
        center_mae = float(np.mean(np.abs(target_price - center_price)))
        improvement = 1.0 - center_mae / naive_mae if naive_mae > 0 else 0.0
        inside = (target_price >= lower_price) & (target_price <= upper_price)
        nominal = cfg.nominal_coverage
        alpha_interval = max(1e-6, 1.0 - nominal)
        interval_score = winkler_score(
            target_price - center_price,
            -radius_price,
            radius_price,
            alpha=alpha_interval,
        )
        move_mask = labels != 1
        move_direction_accuracy = (
            float(np.mean(pred_class[move_mask] == labels[move_mask])) if np.any(move_mask) else np.nan
        )

        pred_frame = pd.DataFrame(
            {
                "origin_time": times[origin_idx],
                "target_time": times[target_idx],
                "mid_now": mid_now,
                "target_price": target_price,
                "target_delta": target_price - mid_now,
                "target_class": labels,
                "predicted_class": pred_class,
                "p_down": probs[:, 0],
                "p_no_move": probs[:, 1],
                "p_up": probs[:, 2],
                "directional_edge": edge,
                "raw_center_delta": raw_center_ticks * tick,
                "alpha_shrinkage": alpha_h,
                "center_delta": center_ticks * tick,
                "center_price": center_price,
                "naive_center_price": mid_now,
                "raw_radius": app_pred["radius_ticks_raw"][:, j] * tick,
                "pred_radius": radius_price,
                "lower_price": lower_price,
                "upper_price": upper_price,
            }
        )

        metrics = {
            "horizon_seconds": h,
            "n_application_samples": len(pred_frame),
            "alpha_shrinkage": alpha_h,
            "typical_move_ticks": move_size_ticks,
            "typical_move_price": move_size_ticks * tick,
            "naive_center_mae": naive_mae,
            "center_mae": center_mae,
            "center_improvement_vs_naive": improvement,
            "classification_accuracy": float(accuracy_score(labels, pred_class)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred_class)),
            "macro_f1": float(f1_score(labels, pred_class, average="macro", zero_division=0)),
            "move_only_direction_accuracy": move_direction_accuracy,
            "nominal_coverage": nominal,
            "empirical_coverage": float(np.mean(inside)),
            "coverage_error": float(np.mean(inside) - nominal),
            "avg_interval_width": float(np.mean(upper_price - lower_price)),
            "winkler_score": interval_score,
            "conformal_padding_ticks": padding_ticks,
            "conformal_padding_price": padding_ticks * tick,
            "target_down_share": float(np.mean(labels == 0)),
            "target_no_move_share": float(np.mean(labels == 1)),
            "target_up_share": float(np.mean(labels == 2)),
        }
        metric_rows.append(metrics)
        results_by_horizon[h] = {
            "metrics": metrics,
            "pred_frame": pred_frame,
            "horizon_steps": int(horizon_steps[j]),
            "horizon_seconds": h,
            "alpha_shrinkage": alpha_h,
            "move_size_ticks": move_size_ticks,
            "conformal_padding_ticks": padding_ticks,
        }

    split_summary = pd.DataFrame(
        [
            {
                "split": name,
                "n_samples": len(origins),
                "first_origin_time": float(times[origins[0]]),
                "last_origin_time": float(times[origins[-1]]),
            }
            for name, origins in splits.items()
        ]
    )

    sequence_length = window_intervals // sequence_step + 1
    return {
        "api_version": ROLLING_SEQUENCE_API_VERSION,
        "config": cfg,
        "model": model,
        "device": str(device),
        "frame": frame,
        "feature_columns": feature_cols,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "price_tick": tick,
        "window_intervals": window_intervals,
        "sequence_step": sequence_step,
        "sequence_length": sequence_length,
        "horizon_steps": tuple(horizon_steps),
        "splits": splits,
        "split_summary": split_summary,
        "training_history": pd.DataFrame(history_rows),
        "results_by_horizon": results_by_horizon,
        "metrics_table": pd.DataFrame(metric_rows).sort_values("horizon_seconds").reset_index(drop=True),
        "alpha_by_horizon": dict(zip(map(float, cfg.horizons_seconds), alpha_values)),
        "conformal_padding_ticks_by_horizon": dict(
            zip(map(float, cfg.horizons_seconds), conformal_values)
        ),
    }


def plot_rolling_sequence_predictions(
    result: Dict[str, object],
    max_points: int = 1200,
    title: Optional[str] = None,
):
    """Plot realized mid, persistence center, learned center and prediction band."""
    pf = pd.DataFrame(result["pred_frame"])
    if max_points is not None and len(pf) > int(max_points):
        pf = pf.tail(int(max_points))
    h = float(result["horizon_seconds"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pf["target_time"], pf["target_price"], label=f"realized mid at t+{h:g}s", linewidth=1.3)
    ax.plot(
        pf["target_time"],
        pf["naive_center_price"],
        label="persistence center",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
    )
    ax.plot(
        pf["target_time"],
        pf["center_price"],
        label="rolling-CNN learned center",
        linewidth=1.2,
    )
    ax.fill_between(
        pf["target_time"].to_numpy(),
        pf["lower_price"].to_numpy(),
        pf["upper_price"].to_numpy(),
        alpha=0.20,
        label=f"{int(round(result['metrics']['nominal_coverage'] * 100))}% prediction band",
    )
    ax.set_title(title or f"Rolling-window LOB forecast, {h:g}s horizon")
    ax.set_xlabel("Target time since market open, seconds")
    ax.set_ylabel("Mid-price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_rolling_sequence_forecast_fan(
    sequence_output: Dict[str, object],
    history_seconds: float = 60.0,
    origin_time: Optional[float] = None,
    symbol: Optional[str] = None,
):
    """Plot recent original LOB mid and 1s/5s/10s rolling-CNN bands."""
    results = sequence_output["results_by_horizon"]
    horizons = sorted(results)
    if not horizons:
        raise ValueError("No horizon results available")

    common_times = None
    for h in horizons:
        times_h = np.round(results[h]["pred_frame"]["origin_time"].to_numpy(dtype=float), 8)
        common_times = times_h if common_times is None else np.intersect1d(common_times, times_h)
    if common_times is None or len(common_times) == 0:
        raise ValueError("No common forecast origin across horizons")
    if origin_time is None:
        chosen = float(common_times[-1])
    else:
        chosen = float(common_times[np.argmin(np.abs(common_times - float(origin_time)))])

    frame = sequence_output["frame"]
    t = frame["time"].to_numpy(dtype=float)
    mid = frame["mid"].to_numpy(dtype=float)
    hist_mask = (t >= chosen - float(history_seconds)) & (t <= chosen)
    origin_mid = float(np.interp(chosen, t, mid))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t[hist_mask], mid[hist_mask], label=f"last {history_seconds:g}s original mid", linewidth=1.5)
    ax.scatter([chosen], [origin_mid], s=45, label="forecast origin")

    future_times, centers, lowers, uppers = [], [], [], []
    for h in horizons:
        pf = results[h]["pred_frame"]
        idx = int(np.argmin(np.abs(pf["origin_time"].to_numpy(dtype=float) - chosen)))
        row = pf.iloc[idx]
        future_times.append(float(row["target_time"]))
        centers.append(float(row["center_price"]))
        lowers.append(float(row["lower_price"]))
        uppers.append(float(row["upper_price"]))
        ax.vlines(
            float(row["target_time"]),
            float(row["lower_price"]),
            float(row["upper_price"]),
            linewidth=5,
            alpha=0.35,
            label=f"{h:g}s band",
        )
        ax.scatter([float(row["target_time"])], [float(row["center_price"])], s=50)

    ax.plot(
        [chosen] + future_times,
        [origin_mid] + centers,
        linestyle="--",
        linewidth=1.2,
        label="rolling-CNN center path",
    )
    ax.fill_between(
        [chosen] + future_times,
        [origin_mid] + lowers,
        [origin_mid] + uppers,
        alpha=0.12,
        label="prediction fan visual aid",
    )
    prefix = f"{symbol}: " if symbol else ""
    ax.set_title(f"{prefix}rolling-window LOB forecast fan")
    ax.set_xlabel("Time since market open, seconds")
    ax.set_ylabel("Mid-price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_rolling_sequence_training_history(sequence_output: Dict[str, object]):
    """Plot chronological training and validation loss by epoch."""
    history = pd.DataFrame(sequence_output["training_history"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["validation_loss"], label="validation")
    ax.set_title("Rolling LOB CNN training history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig, ax
