from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Any
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank, select_order
from statsmodels.stats.diagnostic import acorr_ljungbox

try:
    import optuna
except Exception:  # pragma: no cover - optuna is optional for non-tuning use
    optuna = None


DEFAULT_TARGET_COMPONENTS: Dict[str, float] = {
    "HFRXGL Index": 0.50,
    "MXWO Index": 0.25,
    "LEGATRUU Index": 0.25,
}

DEFAULT_FUTURES: Tuple[str, ...] = (
    "RX1 Comdty",
    "TY1 Comdty",
    "GC1 Comdty",
    "CO1 Comdty",
    "ES1 Comdty",
    "VG1 Comdty",
    "NQ1 Comdty",
    "TP1 Comdty",
    "DU1 Comdty",
    "TU2 Comdty",
)


@dataclass(frozen=True)
class VECMConfig:
    """Configuration for the causal rolling VECM replication backtest.

    The model operates on *return exposure weights*, not physical futures
    contract counts. Contract multipliers, FX conversion and margin are not
    represented by the source dataset and must be added before production use.
    """

    rolling_window: int = 156
    rebalance_every: int = 4
    maxlags_for_order: int = 2
    deterministic: str = "ci"
    rank_method: str = "trace"
    rank_signif: float = 0.05
    max_selected_futures: int = 5
    min_selected_futures: int = 2
    max_gross_exposure: float = 1.0
    transaction_cost_rate: float = 0.0005
    var_confidence: float = 0.95
    var_horizon: int = 1
    max_var_threshold: float = 0.05
    var_lookback: int = 52
    annual_factor: int = 52
    diagnostic_every: int = 13
    failure_mode: str = "hold"  # hold previous weights on rank=0/fit failure

    def validate(self) -> None:
        if self.rolling_window < 52:
            raise ValueError("rolling_window should be at least 52 weekly observations.")
        if self.rebalance_every < 1:
            raise ValueError("rebalance_every must be >= 1.")
        if self.maxlags_for_order < 0:
            raise ValueError("maxlags_for_order must be >= 0.")
        if self.deterministic not in {"n", "ci", "co"}:
            raise ValueError("deterministic must be one of {'n', 'ci', 'co'}.")
        if self.rank_method not in {"trace", "maxeig"}:
            raise ValueError("rank_method must be 'trace' or 'maxeig'.")
        if self.rank_signif not in {0.1, 0.05, 0.01}:
            raise ValueError("rank_signif must be 0.10, 0.05, or 0.01 for statsmodels Johansen tables.")
        if not (0.5 < self.var_confidence < 1.0):
            raise ValueError("var_confidence is a confidence level and must lie in (0.5, 1).")
        if self.var_horizon < 1:
            raise ValueError("var_horizon must be >= 1.")
        if self.max_var_threshold <= 0:
            raise ValueError("max_var_threshold must be positive.")
        if self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be positive.")
        if self.transaction_cost_rate < 0:
            raise ValueError("transaction_cost_rate cannot be negative.")
        if self.max_selected_futures < self.min_selected_futures:
            raise ValueError("max_selected_futures must be >= min_selected_futures.")
        if self.failure_mode not in {"hold", "zero"}:
            raise ValueError("failure_mode must be 'hold' or 'zero'.")


# -----------------------------------------------------------------------------
# Data loading and construction
# -----------------------------------------------------------------------------


def resolve_dataset_path(base_dir: str | Path = ".") -> Path:
    """Find the BC3 workbook using repo and legacy names."""
    base = Path(base_dir)
    candidates = [
        base / "Data" / "Dataset3_PortfolioReplicaStrategy.xlsx",
        base / "Dataset3_PortfolioReplicaStrategy.xlsx",
        base / "BC3_PortfolioReplica" / "Data" / "Dataset3_PortfolioReplicaStrategy.xlsx",
        base / "Data" / "Dataset3_PortfolioReplicaStrategyErrataCorrige.xlsx",
        base / "Dataset3_PortfolioReplicaStrategyErrataCorrige.xlsx",
        base / "BC3_PortfolioReplica" / "Data" / "Dataset3_PortfolioReplicaStrategyErrataCorrige.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the BC3 data workbook. Expected one of:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_bc3_dataset(path: str | Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load the unusual Bloomberg-export-style workbook used by BC3.

    Returns
    -------
    data : DataFrame
        Numeric level data indexed by Date.
    variable_info : dict
        Ticker -> full name mapping extracted from workbook header rows.
    """
    path = Path(path)
    full_names_df = pd.read_excel(path, header=None, skiprows=3, nrows=1)
    tickers_df = pd.read_excel(path, header=None, skiprows=5, nrows=1)
    full_names = full_names_df.iloc[0].tolist()[1:]
    tickers = tickers_df.iloc[0].tolist()[1:]
    variable_info = dict(zip(tickers, full_names))

    data_raw = pd.read_excel(path, header=None, skiprows=6)
    if data_raw.shape[1] != len(tickers) + 1:
        raise ValueError(
            f"Workbook shape mismatch: {data_raw.shape[1]} columns but "
            f"{len(tickers)} tickers plus Date were expected."
        )
    data_raw.columns = ["Date"] + tickers
    data_raw["Date"] = pd.to_datetime(data_raw["Date"], dayfirst=True, errors="coerce")
    data_raw = data_raw.dropna(subset=["Date"]).set_index("Date").sort_index()
    data_raw = data_raw[~data_raw.index.duplicated(keep="last")]
    data = data_raw.apply(pd.to_numeric, errors="coerce")
    return data, variable_info


def build_replication_inputs(
    data: pd.DataFrame,
    target_components: Dict[str, float] = DEFAULT_TARGET_COMPONENTS,
    futures: Sequence[str] = DEFAULT_FUTURES,
) -> Dict[str, pd.DataFrame | pd.Series]:
    """Construct an exact weekly-rebalanced target and causal model inputs.

    Target construction is done in SIMPLE returns:
        R_target,t = sum_i w_i R_i,t
    and only then compounded into a target wealth index.

    VECM levels use normalized log levels of the target wealth index and the
    candidate futures price series. Portfolio P&L uses simple futures returns.
    """
    missing = [c for c in list(target_components) + list(futures) if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required source columns: {missing}")

    weights = pd.Series(target_components, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Target weights must sum to 1; got {weights.sum():.8f}.")

    component_levels = data[list(target_components)].astype(float)
    futures_levels_raw = data[list(futures)].astype(float)

    component_returns = component_levels.pct_change(fill_method=None)
    target_returns = component_returns.mul(weights, axis=1).sum(axis=1, min_count=len(weights))
    target_returns.name = "Target_Index"

    futures_returns = futures_levels_raw.pct_change(fill_method=None)

    common = target_returns.dropna().index.intersection(futures_returns.dropna(how="any").index)
    target_returns = target_returns.loc[common]
    futures_returns = futures_returns.loc[common]
    futures_levels_raw = futures_levels_raw.loc[common]

    if len(common) < 10:
        raise ValueError("Too few fully aligned observations after return construction.")

    # Exact wealth index for a weekly rebalanced benchmark.
    target_wealth = (1.0 + target_returns).cumprod()
    target_wealth.name = "Target_Index"

    # Cointegration is modeled on normalized positive levels. Normalization by
    # the first common observation affects only scale, not return dynamics.
    if (futures_levels_raw <= 0).any().any() or (target_wealth <= 0).any():
        raise ValueError(
            "Non-positive levels found. Log-level VECM is not valid for this sample. "
            "Inspect the continuous futures construction before proceeding."
        )

    normalized_futures = futures_levels_raw.divide(futures_levels_raw.iloc[0])
    normalized_target = target_wealth / target_wealth.iloc[0]
    model_levels = pd.concat([normalized_target, normalized_futures], axis=1)
    log_levels = np.log(model_levels)

    combined_returns = pd.concat([target_returns, futures_returns], axis=1)

    return {
        "target_returns": target_returns,
        "futures_returns": futures_returns,
        "combined_returns": combined_returns,
        "target_wealth": target_wealth,
        "model_levels": log_levels,
        "raw_futures_levels": futures_levels_raw,
    }


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def _safe_adf(series: pd.Series) -> float:
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 20 or s.nunique() < 3:
        return np.nan
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return np.nan


def _safe_kpss(series: pd.Series) -> float:
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 20 or s.nunique() < 3:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(kpss(s, regression="c", nlags="auto")[1])
    except Exception:
        return np.nan


def integration_diagnostics(levels: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Check the I(1) premise using ADF and KPSS on levels and first differences.

    `i1_adf` is deliberately conservative and requires failure to reject a unit
    root in levels plus rejection after first differencing. KPSS is reported as
    complementary evidence rather than silently forcing a classification.
    """
    rows = []
    for col in levels.columns:
        s = levels[col].dropna()
        d = s.diff().dropna()
        adf_level = _safe_adf(s)
        adf_diff = _safe_adf(d)
        kpss_level = _safe_kpss(s)
        kpss_diff = _safe_kpss(d)
        rows.append(
            {
                "series": col,
                "adf_level_p": adf_level,
                "adf_diff_p": adf_diff,
                "kpss_level_p": kpss_level,
                "kpss_diff_p": kpss_diff,
                "i1_adf": bool(np.isfinite(adf_level) and np.isfinite(adf_diff) and adf_level > alpha and adf_diff < alpha),
                "i1_kpss_support": bool(
                    np.isfinite(kpss_level)
                    and np.isfinite(kpss_diff)
                    and kpss_level < alpha
                    and kpss_diff > alpha
                ),
            }
        )
    return pd.DataFrame(rows).set_index("series")


def data_quality_report(
    data: pd.DataFrame,
    target_components: Dict[str, float] = DEFAULT_TARGET_COMPONENTS,
    futures: Sequence[str] = DEFAULT_FUTURES,
) -> pd.DataFrame:
    """Surface stale prices, smoothing and suspicious jumps instead of hiding them."""
    cols = [c for c in list(target_components) + list(futures) if c in data.columns]
    rows = []
    for col in cols:
        level = data[col].astype(float)
        ret = level.pct_change(fill_method=None)
        finite_ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        sigma = finite_ret.std(ddof=1)
        med = finite_ret.median()
        jump_share = np.nan
        if np.isfinite(sigma) and sigma > 0:
            jump_share = float(((finite_ret - med).abs() > 8.0 * sigma).mean())
        rows.append(
            {
                "series": col,
                "missing_pct": float(level.isna().mean()),
                "zero_return_pct": float((finite_ret == 0).mean()) if len(finite_ret) else np.nan,
                "lag1_return_autocorr": float(finite_ret.autocorr(1)) if len(finite_ret) > 2 else np.nan,
                "8sigma_jump_pct": jump_share,
                "generic_continuous_future": bool(col.endswith("Comdty") and any(ch.isdigit() for ch in col)),
            }
        )
    return pd.DataFrame(rows).set_index("series")


def continuous_futures_caveat(futures: Sequence[str] = DEFAULT_FUTURES) -> str:
    generic = [c for c in futures if c.endswith("Comdty") and any(ch.isdigit() for ch in c)]
    if not generic:
        return "No Bloomberg-style generic futures tickers detected."
    return (
        "Generic continuous futures detected: "
        + ", ".join(generic)
        + ". The code cannot infer Bloomberg roll/back-adjustment settings from the workbook. "
          "Confirm the roll methodology before treating long-run levels as economically meaningful."
    )


# -----------------------------------------------------------------------------
# Core econometrics
# -----------------------------------------------------------------------------


def gaussian_var(
    returns: Iterable[float],
    confidence: float = 0.95,
    horizon: int = 1,
    include_mean: bool = False,
) -> float:
    """Positive Gaussian loss VaR.

    With include_mean=False (default), VaR = z_conf * sigma * sqrt(horizon).
    This avoids the sign/convention bug in the original project.
    """
    if not (0.5 < confidence < 1.0):
        raise ValueError("confidence must be a confidence level in (0.5, 1).")
    arr = np.asarray(list(returns), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    sigma = float(np.std(arr, ddof=1))
    z = float(stats.norm.ppf(confidence))
    var = z * sigma * math.sqrt(horizon)
    if include_mean:
        var -= float(np.mean(arr)) * horizon
    return float(max(0.0, var))


def _det_order(deterministic: str) -> int:
    # statsmodels' rank test has a coarser deterministic specification than VECM.
    # Both 'ci' and 'co' imply a constant somewhere in the system => det_order=0.
    if deterministic == "n":
        return -1
    if deterministic in {"ci", "co"}:
        return 0
    raise ValueError(f"Unsupported deterministic specification: {deterministic}")


def _select_futures_by_return_correlation(
    window_returns: pd.DataFrame,
    target_col: str,
    futures: Sequence[str],
    max_selected: int,
    min_selected: int,
) -> list[str]:
    with np.errstate(invalid="ignore", divide="ignore"):
        corrs = window_returns[list(futures)].corrwith(window_returns[target_col]).abs()
    corrs = corrs.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    selected = corrs.index[:max_selected].tolist()
    if len(selected) < min_selected:
        raise ValueError(
            f"Only {len(selected)} futures have usable rolling correlations; need {min_selected}."
        )
    return selected


def _target_oriented_coint_vector(
    beta_matrix: np.ndarray,
    window_levels: np.ndarray,
    target_idx: int = 0,
    ridge: float = 1e-10,
) -> np.ndarray:
    """Choose a target-normalized vector from a rank-r cointegration space.

    When r>1, taking beta[:, 0] is arbitrary because individual columns are not
    uniquely identified. We choose the linear combination beta*c that minimizes
    the in-window variance of the stationary relation subject to target loading=1.
    """
    B = np.asarray(beta_matrix, dtype=float)
    if B.ndim != 2 or B.shape[0] != window_levels.shape[1]:
        raise ValueError("beta_matrix has incompatible dimensions.")
    a = B[target_idx, :]
    if np.linalg.norm(a) < 1e-12:
        raise ValueError("Cointegration space has negligible target loading.")

    centered = np.asarray(window_levels, dtype=float) - np.nanmean(window_levels, axis=0)
    S = np.cov(centered, rowvar=False, ddof=1)
    Q = B.T @ S @ B + ridge * np.eye(B.shape[1])
    Qinv = np.linalg.pinv(Q)
    denom = float(a @ Qinv @ a)
    if abs(denom) < 1e-14:
        raise ValueError("Could not normalize target-oriented cointegration vector.")
    c = (Qinv @ a) / denom
    beta_star = B @ c
    target_loading = float(beta_star[target_idx])
    if abs(target_loading) < 1e-12:
        raise ValueError("Target loading is numerically zero after cointegration-space projection.")
    return beta_star / target_loading


def _cap_gross(weights: np.ndarray, cap: float) -> np.ndarray:
    w = np.asarray(weights, dtype=float).copy()
    gross = float(np.sum(np.abs(w)))
    if gross > cap and gross > 1e-14:
        w *= cap / gross
    return w


def _residual_diagnostics(residuals: np.ndarray, max_lag: int = 4) -> Dict[str, float]:
    """Lightweight residual autocorrelation / ARCH-style checks.

    The minimum p-value across equations is reported for Ljung-Box tests on
    residuals and squared residuals. This is not a full multivariate ARCH test,
    but it makes model misspecification visible instead of suppressing warnings.
    """
    resid = np.asarray(residuals, dtype=float)
    if resid.ndim != 2 or resid.shape[0] <= max_lag + 2:
        return {"resid_lb_min_p": np.nan, "sq_resid_lb_min_p": np.nan}
    p_raw, p_sq = [], []
    for j in range(resid.shape[1]):
        x = resid[:, j]
        if np.nanstd(x) <= 1e-14:
            continue
        try:
            p_raw.append(float(acorr_ljungbox(x, lags=[max_lag], return_df=True)["lb_pvalue"].iloc[-1]))
            p_sq.append(float(acorr_ljungbox(x**2, lags=[max_lag], return_df=True)["lb_pvalue"].iloc[-1]))
        except Exception:
            continue
    return {
        "resid_lb_min_p": float(np.min(p_raw)) if p_raw else np.nan,
        "sq_resid_lb_min_p": float(np.min(p_sq)) if p_sq else np.nan,
    }


def _fallback_weights(previous: np.ndarray, mode: str) -> np.ndarray:
    if mode == "hold":
        return previous.copy()
    return np.zeros_like(previous)


def run_vecm_backtest(
    model_levels: pd.DataFrame,
    combined_returns: pd.DataFrame,
    futures: Sequence[str] = DEFAULT_FUTURES,
    config: VECMConfig = VECMConfig(),
    evaluation_start: Optional[pd.Timestamp | str] = None,
    evaluation_end: Optional[pd.Timestamp | str] = None,
    target_col: str = "Target_Index",
) -> Dict[str, Any]:
    """Causal rolling VECM portfolio-replication backtest.

    Timing for realized return at date t:
        1. use levels/returns strictly before t;
        2. if scheduled, fit/select VECM and determine candidate weights;
        3. compute VaR from historical returns strictly before t and scale weights;
        4. charge transaction costs on the FINAL executed turnover;
        5. apply executed weights to futures return observed at t.

    Returns are recorded every period, even when rebalancing is less frequent.
    """
    config.validate()
    futures = list(futures)
    required_level_cols = [target_col] + futures
    required_ret_cols = [target_col] + futures
    missing_levels = [c for c in required_level_cols if c not in model_levels.columns]
    missing_returns = [c for c in required_ret_cols if c not in combined_returns.columns]
    if missing_levels or missing_returns:
        raise KeyError(f"Missing levels={missing_levels}, missing returns={missing_returns}")

    levels = model_levels[required_level_cols].copy().sort_index()
    returns = combined_returns[required_ret_cols].copy().sort_index()
    common = levels.index.intersection(returns.index)
    levels = levels.loc[common].dropna()
    returns = returns.loc[levels.index].dropna()
    levels = levels.loc[returns.index]

    n = len(levels)
    if n <= config.rolling_window:
        raise ValueError(f"Need more than {config.rolling_window} observations; got {n}.")

    eval_start = pd.Timestamp(evaluation_start) if evaluation_start is not None else levels.index[config.rolling_window]
    eval_end = pd.Timestamp(evaluation_end) if evaluation_end is not None else levels.index[-1]

    all_futures_idx = {f: i for i, f in enumerate(futures)}
    previous_weights = np.zeros(len(futures), dtype=float)
    last_rebalance_t: Optional[int] = None

    records: list[dict[str, Any]] = []
    diagnostic_counter = 0

    for t in range(config.rolling_window, n):
        date = levels.index[t]
        if date < eval_start or date > eval_end:
            continue

        scheduled = last_rebalance_t is None or (t - last_rebalance_t) >= config.rebalance_every
        executed = previous_weights.copy()
        selected: list[str] = []
        rank = np.nan
        k_ar_diff = np.nan
        fit_status = "hold"
        fit_error = ""
        beta_full = np.full(len(futures), np.nan)
        var_value = np.nan
        var_scale = 1.0
        resid_lb_min_p = np.nan
        sq_resid_lb_min_p = np.nan

        if scheduled:
            window_levels_all = levels.iloc[t - config.rolling_window : t]
            window_returns_all = returns.iloc[t - config.rolling_window : t]
            try:
                selected = _select_futures_by_return_correlation(
                    window_returns_all,
                    target_col,
                    futures,
                    max_selected=min(config.max_selected_futures, len(futures)),
                    min_selected=config.min_selected_futures,
                )
                cols = [target_col] + selected
                window_levels = window_levels_all[cols]
                nvars = len(cols)

                # AIC lag selection. select_order already returns VECM k_ar_diff;
                # unlike the original code, zero lagged differences are allowed.
                maxlags = min(config.maxlags_for_order, max(0, (config.rolling_window - 20) // max(3 * nvars, 1)))
                if maxlags > 0:
                    order_res = select_order(window_levels, maxlags=maxlags, deterministic=config.deterministic)
                    selected_aic = order_res.selected_orders.get("aic", 0)
                    k_ar_diff = int(0 if selected_aic is None else selected_aic)
                else:
                    k_ar_diff = 0

                rank_res = select_coint_rank(
                    window_levels,
                    det_order=_det_order(config.deterministic),
                    k_ar_diff=int(k_ar_diff),
                    method=config.rank_method,
                    signif=config.rank_signif,
                )
                rank = int(rank_res.rank)
                if rank <= 0 or rank >= nvars:
                    fit_status = "rank_unusable"
                    executed = _fallback_weights(previous_weights, config.failure_mode)
                else:
                    vecm_model = VECM(
                        window_levels,
                        k_ar_diff=int(k_ar_diff),
                        coint_rank=rank,
                        deterministic=config.deterministic,
                    )
                    vecm_res = vecm_model.fit()
                    beta_star = _target_oriented_coint_vector(
                        vecm_res.beta,
                        window_levels.to_numpy(),
                        target_idx=0,
                    )
                    # target + beta_f' futures is stationary => target ~ (-beta_f)' futures.
                    hedge_coeffs = -beta_star[1:]
                    candidate = np.zeros(len(futures), dtype=float)
                    for f, coeff in zip(selected, hedge_coeffs):
                        candidate[all_futures_idx[f]] = float(coeff)
                    candidate = _cap_gross(candidate, config.max_gross_exposure)

                    # Causal VaR: evaluate the CURRENT candidate on returns known before t.
                    hist_start = max(0, t - config.var_lookback)
                    hist_futures = returns.iloc[hist_start:t][futures].to_numpy(dtype=float)
                    hist_portfolio = hist_futures @ candidate
                    var_value = gaussian_var(
                        hist_portfolio,
                        confidence=config.var_confidence,
                        horizon=config.var_horizon,
                    )
                    if np.isfinite(var_value) and var_value > config.max_var_threshold:
                        var_scale = config.max_var_threshold / var_value
                        candidate = candidate * var_scale

                    executed = candidate
                    for f in selected:
                        beta_full[all_futures_idx[f]] = executed[all_futures_idx[f]]
                    fit_status = "ok"

                    if config.diagnostic_every > 0 and diagnostic_counter % config.diagnostic_every == 0:
                        diag = _residual_diagnostics(vecm_res.resid)
                        resid_lb_min_p = diag["resid_lb_min_p"]
                        sq_resid_lb_min_p = diag["sq_resid_lb_min_p"]
                    diagnostic_counter += 1

            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
                fit_status = "fit_failure"
                fit_error = f"{type(exc).__name__}: {exc}"
                executed = _fallback_weights(previous_weights, config.failure_mode)

            last_rebalance_t = t

        # Costs are charged on FINAL executed weights, after risk scaling.
        turnover = float(np.sum(np.abs(executed - previous_weights)))
        cost = float(config.transaction_cost_rate * turnover)

        # Causal realized P&L at t. Simple returns are used consistently.
        fut_ret_t = returns.iloc[t][futures].to_numpy(dtype=float)
        target_ret_t = float(returns.iloc[t][target_col])
        replica_ret_t = float(fut_ret_t @ executed - cost)

        records.append(
            {
                "date": date,
                "target_return": target_ret_t,
                "replica_return": replica_ret_t,
                "turnover": turnover,
                "transaction_cost": cost,
                "gross_exposure": float(np.sum(np.abs(executed))),
                "net_exposure": float(np.sum(executed)),
                "var": var_value,
                "var_scale": var_scale,
                "rebalanced": bool(scheduled),
                "rank": rank,
                "k_ar_diff": k_ar_diff,
                "fit_status": fit_status,
                "fit_error": fit_error,
                "selected_futures": tuple(selected),
                "resid_lb_min_p": resid_lb_min_p,
                "sq_resid_lb_min_p": sq_resid_lb_min_p,
                **{f"w::{f}": executed[all_futures_idx[f]] for f in futures},
            }
        )
        previous_weights = executed.copy()

    history = pd.DataFrame(records).set_index("date") if records else pd.DataFrame()
    if history.empty:
        raise ValueError("No evaluation observations were produced. Check evaluation dates.")

    metrics = compute_replication_metrics(
        history["replica_return"],
        history["target_return"],
        turnover=history["turnover"],
        costs=history["transaction_cost"],
        annual_factor=config.annual_factor,
    )
    rebalances = history[history["rebalanced"]]
    metrics.update(
        {
            "avg_gross_exposure": float(history["gross_exposure"].mean()),
            "avg_abs_net_exposure": float(history["net_exposure"].abs().mean()),
            "avg_var": float(history["var"].dropna().mean()) if history["var"].notna().any() else np.nan,
            "var_scaling_frequency": float((history["var_scale"] < 1.0).mean()),
            "fit_success_rate": float((rebalances["fit_status"] == "ok").mean()) if len(rebalances) else np.nan,
            "rank_zero_or_unusable_rate": float((rebalances["fit_status"] == "rank_unusable").mean()) if len(rebalances) else np.nan,
            "fit_failure_rate": float((rebalances["fit_status"] == "fit_failure").mean()) if len(rebalances) else np.nan,
        }
    )

    return {
        "history": history,
        "metrics": metrics,
        "weights": history[[f"w::{f}" for f in futures]].rename(columns=lambda c: c[3:]),
        "cumulative_replica": (1.0 + history["replica_return"]).cumprod(),
        "cumulative_target": (1.0 + history["target_return"]).cumprod(),
        "config": asdict(config),
    }


# -----------------------------------------------------------------------------
# Metrics and validation
# -----------------------------------------------------------------------------


def compute_replication_metrics(
    replica_returns: pd.Series,
    target_returns: pd.Series,
    turnover: Optional[pd.Series] = None,
    costs: Optional[pd.Series] = None,
    annual_factor: int = 52,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    aligned = pd.concat(
        [pd.Series(replica_returns, name="replica"), pd.Series(target_returns, name="target")],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        raise ValueError("Need at least two aligned returns for metrics.")
    rep = aligned["replica"].astype(float)
    tgt = aligned["target"].astype(float)
    active = rep - tgt
    years = len(aligned) / annual_factor

    def cagr(r: pd.Series) -> float:
        wealth = float((1.0 + r).prod())
        if wealth <= 0 or years <= 0:
            return np.nan
        return wealth ** (1.0 / years) - 1.0

    rep_vol = float(rep.std(ddof=1) * np.sqrt(annual_factor))
    tgt_vol = float(tgt.std(ddof=1) * np.sqrt(annual_factor))
    te = float(active.std(ddof=1) * np.sqrt(annual_factor))
    active_ann_mean = float(active.mean() * annual_factor)
    rf_period = risk_free_rate / annual_factor
    rep_sharpe = float((rep.mean() - rf_period) / rep.std(ddof=1) * np.sqrt(annual_factor)) if rep.std(ddof=1) > 0 else np.nan
    tgt_sharpe = float((tgt.mean() - rf_period) / tgt.std(ddof=1) * np.sqrt(annual_factor)) if tgt.std(ddof=1) > 0 else np.nan
    ir = float(active_ann_mean / te) if te > 0 else np.nan

    wealth = (1.0 + rep).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0

    out = {
        "n_obs": int(len(aligned)),
        "replica_cagr": cagr(rep),
        "target_cagr": cagr(tgt),
        "replica_ann_mean": float(rep.mean() * annual_factor),
        "target_ann_mean": float(tgt.mean() * annual_factor),
        "replica_vol": rep_vol,
        "target_vol": tgt_vol,
        "replica_sharpe": rep_sharpe,
        "target_sharpe": tgt_sharpe,
        "tracking_error": te,
        "information_ratio": ir,
        "correlation": float(rep.corr(tgt)),
        "max_drawdown": float(drawdown.min()),
    }
    if turnover is not None:
        t = pd.Series(turnover).reindex(aligned.index).fillna(0.0)
        out["total_turnover"] = float(t.sum())
        out["annualized_turnover"] = float(t.sum() / years) if years > 0 else np.nan
    if costs is not None:
        c = pd.Series(costs).reindex(aligned.index).fillna(0.0)
        out["total_transaction_cost"] = float(c.sum())
        out["annualized_transaction_cost"] = float(c.sum() / years) if years > 0 else np.nan
    return out


def chronological_split_dates(
    index: pd.DatetimeIndex,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> Dict[str, pd.Timestamp]:
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Need train_frac>0, val_frac>0 and train_frac+val_frac<1.")
    train_end_pos = max(1, int(len(idx) * train_frac))
    val_end_pos = max(train_end_pos + 1, int(len(idx) * (train_frac + val_frac)))
    if val_end_pos >= len(idx):
        raise ValueError("Not enough observations for requested split.")
    return {
        "train_end": pd.Timestamp(idx[train_end_pos]),
        "val_end": pd.Timestamp(idx[val_end_pos]),
        "test_end": pd.Timestamp(idx[-1]),
    }


def tune_vecm_on_validation(
    model_levels: pd.DataFrame,
    combined_returns: pd.DataFrame,
    split_dates: Dict[str, pd.Timestamp],
    futures: Sequence[str] = DEFAULT_FUTURES,
    n_trials: int = 30,
    storage: str = "sqlite:///optuna_VECM_v3.db",
    study_name: str = "VECM_tracking_error_v3",
    base_config: VECMConfig = VECMConfig(),
    turnover_penalty: float = 5e-4,
    failure_penalty: float = 0.02,
    random_seed: int = 42,
):
    """Tune only on the chronological validation interval.

    The primary objective is tracking error. Small turnover and fit-failure
    penalties discourage pathological solutions. The test interval is never
    touched by this function.
    """
    if optuna is None:
        raise ImportError("Optuna is required for tuning.")

    val_start = split_dates["train_end"]
    val_end = split_dates["val_end"] - pd.Timedelta(nanoseconds=1)

    def objective(trial):
        cfg = VECMConfig(
            **{
                **asdict(base_config),
                "rolling_window": trial.suggest_categorical("rolling_window", [104, 156, 208]),
                "rebalance_every": trial.suggest_categorical("rebalance_every", [1, 2, 4]),
                "maxlags_for_order": trial.suggest_int("maxlags_for_order", 0, 2),
                "max_selected_futures": trial.suggest_int("max_selected_futures", 3, min(6, len(futures))),
                "max_gross_exposure": trial.suggest_float("max_gross_exposure", 0.75, 1.5),
            }
        )
        try:
            result = run_vecm_backtest(
                model_levels,
                combined_returns,
                futures=futures,
                config=cfg,
                evaluation_start=val_start,
                evaluation_end=val_end,
            )
        except (ValueError, np.linalg.LinAlgError):
            raise optuna.TrialPruned()

        m = result["metrics"]
        te = m["tracking_error"]
        turnover = m.get("annualized_turnover", 0.0)
        failure = 1.0 - m.get("fit_success_rate", 0.0)
        score = te + turnover_penalty * turnover + failure_penalty * failure
        trial.set_user_attr("tracking_error", float(te))
        trial.set_user_attr("annualized_turnover", float(turnover))
        trial.set_user_attr("fit_success_rate", float(m.get("fit_success_rate", np.nan)))
        return float(score)

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )
    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(
            "VECM Optuna study has no COMPLETE trials. Run with n_trials>0 or use a populated v3 study."
        )
    return study


def config_from_best_trial(study, base_config: VECMConfig = VECMConfig()) -> VECMConfig:
    params = dict(study.best_trial.params)
    return VECMConfig(**{**asdict(base_config), **params})


def final_test_backtest(
    model_levels: pd.DataFrame,
    combined_returns: pd.DataFrame,
    split_dates: Dict[str, pd.Timestamp],
    config: VECMConfig,
    futures: Sequence[str] = DEFAULT_FUTURES,
) -> Dict[str, Any]:
    """Run one untouched chronological test after hyperparameters are fixed."""
    return run_vecm_backtest(
        model_levels,
        combined_returns,
        futures=futures,
        config=config,
        evaluation_start=split_dates["val_end"],
        evaluation_end=split_dates["test_end"],
    )
