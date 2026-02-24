"""
thesis_figures_v2.py

Thesis-ready figure factory for the Hawkes-driven market-making pipeline.

Design goals
------------
1) Produce figures that *test and support* the modelling assumptions:
   - order-flow clustering (Hawkes vs Poisson)
   - dominance of aggressive executions (2D reduction)
   - stability/subcriticality across windows (rho(K))
   - simulation realism (real vs sim stats)
   - reduced-form microstructure proxies (fill/marking) validity
2) Be compatible with your codebase (OrderBook / event_cleaning / HawkesCalibrator / PointProcessPlotSuite).
3) Save figures directly into your thesis Images/ folder (pdf + png).

Minimal dependencies: numpy, pandas, matplotlib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse your time-axis formatting if available.
try:
    from utils import apply_time_of_day_axis
except Exception:  # pragma: no cover
    apply_time_of_day_axis = None

# Optional: reuse your existing Hawkes plots if available.
try:
    from PointProcessPlotSuite import PointProcessPlotSuite
except Exception:  # pragma: no cover
    PointProcessPlotSuite = None


# =============================================================================
# Helpers
# =============================================================================
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _save(fig, out_dir: Optional[str], name: str, fmts: Tuple[str, ...] = ("pdf", "png"), dpi: int = 220) -> None:
    if out_dir is None:
        return
    _ensure_dir(out_dir)
    for ext in fmts:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=dpi, bbox_inches="tight")

def _spectral_radius(M: np.ndarray) -> float:
    if M.size == 0:
        return float("nan")
    vals = np.linalg.eigvals(M)
    return float(np.max(np.abs(vals)))

def _branching_matrix(A: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Source-specific decay beta_j (column j):
        K_ij = A_ij / beta_j.
    """
    A = np.asarray(A, float)
    beta = np.asarray(beta, float).reshape(-1)
    denom = np.maximum(beta, 1e-12)
    return A / denom[np.newaxis, :]

def _bin_counts(times: np.ndarray, T: float, bin_size: float) -> np.ndarray:
    times = np.asarray(times, float)
    times = times[np.isfinite(times)]
    times = times[(times >= 0.0) & (times < float(T))]
    nb = int(np.ceil(float(T) / float(bin_size)))
    if nb <= 0:
        return np.array([], float)
    bins = np.linspace(0.0, float(T), nb + 1)
    c, _ = np.histogram(times, bins=bins)
    return c.astype(float)

def _fano_factor(counts: np.ndarray) -> float:
    counts = np.asarray(counts, float)
    if counts.size < 2:
        return float("nan")
    m = float(np.mean(counts))
    v = float(np.var(counts, ddof=1))
    return float(v / m) if m > 0 else float("nan")

def _acf(x: np.ndarray, max_lag: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.arange(0), np.arange(0)
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom <= 0:
        return np.arange(0), np.arange(0)
    lags = np.arange(0, max_lag + 1)
    acf_vals = np.empty_like(lags, dtype=float)
    for k in lags:
        if k == 0:
            acf_vals[k] = 1.0
        else:
            acf_vals[k] = np.dot(x[:-k], x[k:]) / denom
    return lags, acf_vals

def _as_list_win_results(win_results: Any) -> List[Any]:
    """
    Your `win_results` (from lob2.fit_windows) is typically a dict:
        {win_idx: WindowResult}
    Iterating a dict gives keys (ints) -> this caused your 'int is not subscriptable' error.

    This helper returns a list of WindowResult objects regardless of dict/list input.
    """
    if win_results is None:
        return []
    if isinstance(win_results, dict):
        return list(win_results.values())
    if isinstance(win_results, (list, tuple)):
        return list(win_results)
    return [win_results]

def _infer_tstart_from_window(w: Any) -> float:
    for attr in ("t0", "t_start", "start", "tmin"):
        if hasattr(w, attr):
            return float(getattr(w, attr))
    return 0.0

def _infer_T_from_window(w: Any) -> float:
    if hasattr(w, "t1") and hasattr(w, "t0"):
        return float(getattr(w, "t1") - getattr(w, "t0"))
    for attr in ("T", "length", "dt"):
        if hasattr(w, attr):
            return float(getattr(w, attr))
    return float("nan")

def _infer_fit_from_window(w: Any) -> Any:
    if hasattr(w, "fit"):
        return getattr(w, "fit")
    return w

def _infer_mu_A_beta(fit: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for mu_name in ("mu", "mu_hat", "mu_est"):
        if hasattr(fit, mu_name):
            mu = np.asarray(getattr(fit, mu_name), float)
            break
    else:
        raise AttributeError("Could not find mu in fit result.")
    for A_name in ("A", "A_hat", "A_est", "alpha"):
        if hasattr(fit, A_name):
            A = np.asarray(getattr(fit, A_name), float)
            break
    else:
        raise AttributeError("Could not find A in fit result.")
    for b_name in ("beta", "beta_hat", "beta_est", "decay"):
        if hasattr(fit, b_name):
            beta = np.asarray(getattr(fit, b_name), float)
            break
    else:
        raise AttributeError("Could not find beta in fit result.")
    return mu, A, beta


# =============================================================================
# Public containers
# =============================================================================
@dataclass
class HawkesFitSummary:
    """Lightweight container for plotting stability across windows."""
    t_start: float
    T: float
    mu: np.ndarray
    A: np.ndarray
    beta: np.ndarray
    meta: Optional[dict] = None

    @property
    def K(self) -> np.ndarray:
        return _branching_matrix(self.A, self.beta)

    @property
    def rho(self) -> float:
        return _spectral_radius(self.K)


# =============================================================================
# Main class
# =============================================================================
class ThesisFigureFactory:
    """
    One-stop figure generator for your thesis.

    Typical inputs produced by your pipeline:
    - events_rel: dict[str -> np.ndarray] with relative episode times in [0,T]
    - win_results: dict[int -> WindowResult] from LOBHawkes.fit_windows
    - mid_times, mid_prices: arrays from your episode slice
    - sim_events: dict[str -> np.ndarray] from your simulator (e.g. events_sim_2D)

    All `plot_*` methods:
    - return (fig, ax/axes, optional_data)
    - save to `out_dir` automatically as PDF+PNG
    """

    def __init__(self, out_dir: str = "Images"):
        self.out_dir = out_dir

    # ---------------------------------------------------------------------
    # A) Hawkes vs Poisson evidence: burstiness / clustering
    # ---------------------------------------------------------------------
    def plot_fano_factor_vs_bins(
        self,
        events_rel: Dict[str, np.ndarray],
        T: float,
        components: Iterable[str],
        bin_sizes: Iterable[float] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        *,
        title: str = "Burstiness (Fano factor) across time scales",
        save_name: str = "pp_fano_vs_bins",
    ):
        comps = list(components)
        bin_sizes = [float(b) for b in bin_sizes]

        fig, ax = plt.subplots(1, 1, figsize=(8.5, 5))
        for name in comps:
            t = np.asarray(events_rel.get(name, np.array([])), float)
            ff = []
            for b in bin_sizes:
                c = _bin_counts(t, T=T, bin_size=b)
                ff.append(_fano_factor(c))
            ax.plot(bin_sizes, ff, marker="o", label=name)

        ax.axhline(1.0, linestyle="--", linewidth=1.2, label="Poisson baseline (1)")
        ax.set_xscale("log")
        ax.set_xlabel("Bin size Δ (seconds, log-scale)")
        ax.set_ylabel("Fano factor Var/Mean")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()

        _save(fig, self.out_dir, save_name)
        return fig, ax

    def plot_fano_real_vs_sim(
        self,
        real_events: np.ndarray,
        sim_events: np.ndarray,
        T: float,
        bin_sizes: Iterable[float] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        *,
        title: str = "Burstiness across time scales: real vs simulated (Fano factor)",
        save_name: str = "pp_fano_real_vs_sim",
    ):
        bin_sizes = [float(b) for b in bin_sizes]
        real_events = np.asarray(real_events, float)
        sim_events = np.asarray(sim_events, float)

        def fano_curve(t):
            out = []
            for b in bin_sizes:
                c = _bin_counts(t, T=T, bin_size=b)
                out.append(_fano_factor(c))
            return np.array(out, float)

        fr = fano_curve(real_events)
        fs = fano_curve(sim_events)

        fig, ax = plt.subplots(1, 1, figsize=(8.5, 5))
        ax.plot(bin_sizes, fr, marker="o", label="real")
        ax.plot(bin_sizes, fs, marker="o", label="sim")
        ax.axhline(1.0, linestyle="--", linewidth=1.2, label="Poisson baseline (1)")
        ax.set_xscale("log")
        ax.set_xlabel("Bin size Δ (seconds, log-scale)")
        ax.set_ylabel("Fano factor Var/Mean")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()

        _save(fig, self.out_dir, save_name)
        return fig, ax, {"bin_sizes": bin_sizes, "fano_real": fr, "fano_sim": fs}

    # ---------------------------------------------------------------------
    # B) Stability / subcriticality evidence: rho(K) over windows
    # ---------------------------------------------------------------------
    def summaries_from_win_results(self, win_results: Any) -> List[HawkesFitSummary]:
        ws = _as_list_win_results(win_results)
        out: List[HawkesFitSummary] = []
        for w in ws:
            t0 = _infer_tstart_from_window(w)
            T = _infer_T_from_window(w)
            fit = _infer_fit_from_window(w)
            mu, A, beta = _infer_mu_A_beta(fit)
            out.append(HawkesFitSummary(t_start=t0, T=T, mu=mu, A=A, beta=beta))
        return out

    def plot_hawkes_stability_over_windows(
        self,
        fits: Union[List[HawkesFitSummary], Any],
        *,
        save_name: str = "pp_hawkes_stability_over_windows",
        title: str = "Hawkes parameter stability across intraday windows",
        show_components: bool = True,
    ):
        if not isinstance(fits, list) or (len(fits) > 0 and not isinstance(fits[0], HawkesFitSummary)):
            fits = self.summaries_from_win_results(fits)

        if len(fits) == 0:
            raise ValueError("fits is empty")

        fits = sorted(fits, key=lambda f: f.t_start)
        t0 = np.array([f.t_start for f in fits], float)
        rho = np.array([f.rho for f in fits], float)

        d = int(len(fits[0].mu))
        mu = np.vstack([np.asarray(f.mu, float).reshape(1, -1) for f in fits])
        beta = np.vstack([np.asarray(f.beta, float).reshape(1, -1) for f in fits])

        nrows = 1 + (2 if show_components else 0)
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 3.2 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]

        ax0 = axes[0]
        ax0.plot(t0, rho, marker="o")
        ax0.set_ylabel(r"$\rho(K)$")
        ax0.set_title(title)
        ax0.grid(True, alpha=0.25)
        if apply_time_of_day_axis is not None:
            apply_time_of_day_axis(ax0, base_seconds=0.0, rotate=30)

        if show_components:
            ax1 = axes[1]
            for i in range(d):
                ax1.plot(t0, mu[:, i], marker="o", label=rf"$\mu_{i}$")
            ax1.set_ylabel(r"$\mu$")
            ax1.grid(True, alpha=0.25)
            ax1.legend(ncol=min(d, 4))

            ax2 = axes[2]
            for j in range(d):
                ax2.plot(t0, beta[:, j], marker="o", label=rf"$\beta_{j}$")
            ax2.set_ylabel(r"$\beta$")
            ax2.set_xlabel("Window start time (seconds from market open)")
            ax2.grid(True, alpha=0.25)
            ax2.legend(ncol=min(d, 4))
            if apply_time_of_day_axis is not None:
                apply_time_of_day_axis(ax2, base_seconds=0.0, rotate=30)
        else:
            ax0.set_xlabel("Window start time (seconds from market open)")

        plt.tight_layout()
        _save(fig, self.out_dir, save_name)
        return fig, axes

    def plot_rho_histogram(
        self,
        win_results: Any,
        *,
        bins: int = 20,
        save_name: str = "pp_rho_histogram",
        title: str = r"Distribution of $\rho(K)$ across windows",
    ):
        fits = self.summaries_from_win_results(win_results)
        rho = np.array([f.rho for f in fits], float)

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
        ax.hist(rho[np.isfinite(rho)], bins=bins, density=False)
        ax.set_xlabel(r"$\rho(K)$")
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

        _save(fig, self.out_dir, save_name)
        return fig, ax, {"rho": rho}

    # ---------------------------------------------------------------------
    # C) 2D reduction evidence: event -> price move association
    # ---------------------------------------------------------------------
    def plot_event_type_midmove_lift(
        self,
        mid_times: np.ndarray,
        mid_prices: np.ndarray,
        events_rel: Dict[str, np.ndarray],
        T: float,
        components: Iterable[str],
        *,
        horizon_s: float = 1.0,
        tick_size: float = 0.01,
        save_name: str = "pp_eventtype_midmove_lift",
        title: str = "Event-type lift: probability of one-tick mid move after an event",
    ):
        mid_times = np.asarray(mid_times, float)
        mid_prices = np.asarray(mid_prices, float)
        if mid_times.ndim != 1 or mid_prices.ndim != 1 or mid_times.size != mid_prices.size:
            raise ValueError("mid_times and mid_prices must be 1D arrays with same length")

        msk = (mid_times >= 0.0) & (mid_times <= float(T)) & np.isfinite(mid_times) & np.isfinite(mid_prices)
        mid_times = mid_times[msk]
        mid_prices = mid_prices[msk]
        if mid_times.size < 10:
            raise ValueError("Not enough mid samples in [0,T].")

        idx_fwd = np.searchsorted(mid_times, mid_times + float(horizon_s), side="left")
        valid = idx_fwd < mid_times.size
        dm = np.zeros_like(mid_prices)
        dm[valid] = mid_prices[idx_fwd[valid]] - mid_prices[valid]
        baseline = float(np.mean(np.abs(dm[valid]) >= float(tick_size)))

        comps = list(components)
        lifts, counts, probs = [], [], []

        for name in comps:
            t_ev = np.asarray(events_rel.get(name, np.array([])), float)
            t_ev = t_ev[np.isfinite(t_ev)]
            t_ev = t_ev[(t_ev >= 0.0) & (t_ev <= float(T))]
            if t_ev.size == 0:
                lifts.append(np.nan); counts.append(0); probs.append(np.nan)
                continue

            idx0 = np.searchsorted(mid_times, t_ev, side="left")
            idx0 = idx0[idx0 < mid_times.size]
            if idx0.size == 0:
                lifts.append(np.nan); counts.append(0); probs.append(np.nan)
                continue

            t0 = mid_times[idx0]
            idx1 = np.searchsorted(mid_times, t0 + float(horizon_s), side="left")
            ok = idx1 < mid_times.size
            idx0 = idx0[ok]; idx1 = idx1[ok]

            moved = np.abs(mid_prices[idx1] - mid_prices[idx0]) >= float(tick_size)
            p = float(np.mean(moved)) if moved.size else np.nan
            probs.append(p)
            lifts.append(p / baseline if baseline > 1e-12 else np.nan)
            counts.append(int(moved.size))

        fig, ax = plt.subplots(1, 1, figsize=(11, 4.8))
        x = np.arange(len(comps))
        ax.bar(x, lifts)
        ax.axhline(1.0, linestyle="--", linewidth=1.2, label="baseline (lift=1)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n(n={counts[i]})" for i, c in enumerate(comps)], rotation=0)
        ax.set_ylabel(f"Lift = P(move within {horizon_s:g}s | event) / P(move within {horizon_s:g}s)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

        plt.tight_layout()
        _save(fig, self.out_dir, save_name)
        return fig, ax, {"baseline_prob": baseline, "probs": dict(zip(comps, probs)),
                         "lifts": dict(zip(comps, lifts)), "counts": dict(zip(comps, counts))}

    def plot_event_type_midmove_lift_multi(
        self,
        mid_times: np.ndarray,
        mid_prices: np.ndarray,
        events_rel: Dict[str, np.ndarray],
        T: float,
        components: Iterable[str],
        horizons_s: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0),
        *,
        tick_size: float = 0.01,
        save_name: str = "pp_eventtype_midmove_lift_multi",
        title: str = "Event-type lift vs horizon (one-tick mid move)",
    ):
        comps = list(components)
        horizons_s = [float(h) for h in horizons_s]

        fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.2))
        out = {}
        for c in comps:
            lifts_c = []
            for h in horizons_s:
                fig_tmp, ax_tmp, info = self.plot_event_type_midmove_lift(
                    mid_times, mid_prices, events_rel, T, [c],
                    horizon_s=h, tick_size=tick_size,
                    save_name="_tmp_do_not_save", title=""
                )
                lifts_c.append(info["lifts"][c])
                plt.close(fig_tmp)
            out[c] = lifts_c
            ax.plot(horizons_s, lifts_c, marker="o", label=c)

        ax.axhline(1.0, linestyle="--", linewidth=1.2, label="baseline (1)")
        ax.set_xscale("log")
        ax.set_xlabel("Horizon (seconds, log-scale)")
        ax.set_ylabel("Lift")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=min(len(comps), 6))

        _save(fig, self.out_dir, save_name)
        return fig, ax, {"horizons": horizons_s, "lifts": out}

    def plot_event_impulse_response(
        self,
        mid_times: np.ndarray,
        mid_prices: np.ndarray,
        event_times: np.ndarray,
        *,
        T: float,
        tau_max: float = 5.0,
        n_tau: int = 60,
        tick_size: float = 0.01,
        save_name: str = "pp_event_impulse_response",
        title: str = "Event-conditioned midprice response",
        label: str = "event",
    ):
        mid_times = np.asarray(mid_times, float)
        mid_prices = np.asarray(mid_prices, float)
        event_times = np.asarray(event_times, float)

        msk = (mid_times >= 0.0) & (mid_times <= float(T)) & np.isfinite(mid_times) & np.isfinite(mid_prices)
        mid_times = mid_times[msk]
        mid_prices = mid_prices[msk]
        event_times = event_times[np.isfinite(event_times)]
        event_times = event_times[(event_times >= 0.0) & (event_times <= float(T))]
        if mid_times.size < 50 or event_times.size < 10:
            raise ValueError("Not enough mid samples or events.")

        tau_grid = np.linspace(0.0, float(tau_max), int(n_tau))

        idx0 = np.searchsorted(mid_times, event_times, side="left")
        idx0 = idx0[idx0 < mid_times.size]
        t0 = mid_times[idx0]
        m0 = mid_prices[idx0]

        means = np.zeros_like(tau_grid)
        q25 = np.zeros_like(tau_grid)
        q75 = np.zeros_like(tau_grid)

        for k, tau in enumerate(tau_grid):
            idx1 = np.searchsorted(mid_times, t0 + tau, side="left")
            ok = idx1 < mid_times.size
            if not np.any(ok):
                means[k] = np.nan; q25[k] = np.nan; q75[k] = np.nan
                continue
            dm = mid_prices[idx1[ok]] - m0[ok]
            means[k] = float(np.mean(dm))
            q25[k] = float(np.quantile(dm, 0.25))
            q75[k] = float(np.quantile(dm, 0.75))

        fig, ax = plt.subplots(1, 1, figsize=(9.5, 5))
        ax.plot(tau_grid, means / tick_size, linewidth=2, label=f"E[Δm|{label}] (in ticks)")
        ax.fill_between(tau_grid, q25 / tick_size, q75 / tick_size, alpha=0.25, label="IQR (25–75%)")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("τ (seconds)")
        ax.set_ylabel("Mid response (ticks)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()

        _save(fig, self.out_dir, save_name)
        return fig, ax, {"tau": tau_grid, "mean": means, "q25": q25, "q75": q75}

    def plot_flow_imbalance_vs_returns(
        self,
        mid_times: np.ndarray,
        mid_prices: np.ndarray,
        buy_mo_times: np.ndarray,
        sell_mo_times: np.ndarray,
        *,
        T: float,
        delta: float = 1.0,
        tick_size: float = 0.01,
        nbins: int = 9,
        save_name: str = "pp_flow_imbalance_vs_returns",
        title: str = "Flow imbalance vs short-horizon mid returns",
    ):
        mid_times = np.asarray(mid_times, float)
        mid_prices = np.asarray(mid_prices, float)
        msk = (mid_times >= 0.0) & (mid_times <= float(T)) & np.isfinite(mid_times) & np.isfinite(mid_prices)
        mid_times = mid_times[msk]
        mid_prices = mid_prices[msk]

        buy_mo_times = np.asarray(buy_mo_times, float)
        sell_mo_times = np.asarray(sell_mo_times, float)
        buy_mo_times = buy_mo_times[np.isfinite(buy_mo_times)]
        sell_mo_times = sell_mo_times[np.isfinite(sell_mo_times)]
        buy_mo_times = buy_mo_times[(buy_mo_times >= 0.0) & (buy_mo_times <= float(T))]
        sell_mo_times = sell_mo_times[(sell_mo_times >= 0.0) & (sell_mo_times <= float(T))]

        edges = np.arange(0.0, float(T) + float(delta), float(delta))
        if edges.size < 5:
            raise ValueError("T/delta too small for binning.")

        c_buy, _ = np.histogram(buy_mo_times, bins=edges)
        c_sell, _ = np.histogram(sell_mo_times, bins=edges)
        dN = c_buy - c_sell

        idx0 = np.searchsorted(mid_times, edges[:-1], side="left")
        idx1 = np.searchsorted(mid_times, edges[1:], side="left")
        ok = (idx0 < mid_times.size) & (idx1 < mid_times.size)
        ret = np.full(idx0.shape, np.nan, dtype=float)
        ret[ok] = mid_prices[idx1[ok]] - mid_prices[idx0[ok]]

        df = pd.DataFrame({"dN": dN, "ret": ret})
        df = df[np.isfinite(df["ret"])]
        if df.shape[0] < 50:
            raise ValueError("Not enough windows with mid data to compute returns.")

        qs = np.quantile(df["dN"], np.linspace(0.0, 1.0, nbins + 1))
        qs = np.unique(qs)
        if qs.size < 4:
            mn, mx = int(df["dN"].min()), int(df["dN"].max())
            qs = np.linspace(mn, mx + 1, nbins + 1)

        df["bin"] = pd.cut(df["dN"], bins=qs, include_lowest=True)
        grp = df.groupby("bin", observed=True)
        x = grp["dN"].mean().values
        y = (grp["ret"].mean().values) / tick_size
        yerr = (grp["ret"].std(ddof=1).values / np.sqrt(np.maximum(grp.size().values, 1))) / tick_size

        fig, ax = plt.subplots(1, 1, figsize=(8.8, 5))
        ax.errorbar(x, y, yerr=yerr, fmt="o-", linewidth=2)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel(r"Mean net flow $\Delta N = N^+ - N^-$ per window")
        ax.set_ylabel("Mean mid return over window (ticks)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

        _save(fig, self.out_dir, save_name)
        return fig, ax, {"bin_edges": qs, "x_mean_dN": x, "y_mean_ret_ticks": y}

    # ---------------------------------------------------------------------
    # D) Simulation realism: inter-arrivals & counts
    # ---------------------------------------------------------------------
    def plot_interarrival_real_vs_sim(
        self,
        real_times: np.ndarray,
        sim_times: np.ndarray,
        *,
        max_dt: float = 30.0,
        bins: int = 60,
        save_name: str = "pp_interarrival_real_vs_sim",
        title: str = "Inter-arrival times: real vs simulated",
    ):
        real_times = np.sort(np.asarray(real_times, float))
        sim_times = np.sort(np.asarray(sim_times, float))

        real_dt = np.diff(real_times)
        sim_dt = np.diff(sim_times)
        real_dt = real_dt[np.isfinite(real_dt)]
        sim_dt = sim_dt[np.isfinite(sim_dt)]

        fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
        ax.hist(real_dt[real_dt <= max_dt], bins=bins, density=True, alpha=0.5, label="real")
        ax.hist(sim_dt[sim_dt <= max_dt], bins=bins, density=True, alpha=0.5, label="sim")
        ax.set_xlabel("Δt (seconds)")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()

        _save(fig, self.out_dir, save_name)
        return fig, ax

    def plot_count_distribution_real_vs_sim(
        self,
        real_times: np.ndarray,
        sim_times: np.ndarray,
        *,
        T: float,
        bin_size: float = 1.0,
        max_count: Optional[int] = None,
        save_name: str = "pp_counts_real_vs_sim",
        title: str = "Count distribution in fixed bins: real vs simulated",
    ):
        real_c = _bin_counts(np.asarray(real_times, float), T=T, bin_size=bin_size).astype(int)
        sim_c = _bin_counts(np.asarray(sim_times, float), T=T, bin_size=bin_size).astype(int)

        if max_count is None:
            max_count = int(max(np.max(real_c) if real_c.size else 0, np.max(sim_c) if sim_c.size else 0))
            max_count = max(max_count, 10)

        xs = np.arange(0, max_count + 1)
        pr = np.array([(real_c == k).mean() for k in xs], float)
        ps = np.array([(sim_c == k).mean() for k in xs], float)

        fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
        ax.bar(xs - 0.15, pr, width=0.3, alpha=0.6, label="real")
        ax.bar(xs + 0.15, ps, width=0.3, alpha=0.6, label="sim")
        ax.set_xlabel(f"count per {bin_size:g}s bin")
        ax.set_ylabel("probability")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

        _save(fig, self.out_dir, save_name)
        return fig, ax, {"xs": xs, "pmf_real": pr, "pmf_sim": ps}

    # ---------------------------------------------------------------------
    # F) Residual dependence check
    # ---------------------------------------------------------------------
    def plot_residual_acf(
        self,
        residuals: np.ndarray,
        *,
        max_lag: int = 50,
        save_name: str = "pp_residual_acf",
        title: str = "Time-rescaled residual ACF (should be near 0 under correct specification)",
    ):
        r = np.asarray(residuals, float)
        r = r[np.isfinite(r)]
        if r.size < 20:
            raise ValueError("Not enough residuals.")

        lags, ac = _acf(r, max_lag=max_lag)
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        ax.stem(lags, ac, basefmt=" ")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

        _save(fig, self.out_dir, save_name)
        return fig, ax

    # ---------------------------------------------------------------------
    # G) Convenience wrapper: PointProcessPlotSuite
    # ---------------------------------------------------------------------
    def hawkes_standard_figures(
        self,
        events: Dict[str, np.ndarray],
        fit_res: Any,
        *,
        components: Optional[List[str]] = None,
        t_start: float = 0.0,
        T: float = 600.0,
        warmup: float = 0.0,
        n_grid: int = 20000,
        prefix: str = "",
    ):
        if PointProcessPlotSuite is None:
            raise ImportError("PointProcessPlotSuite could not be imported.")

        pps = PointProcessPlotSuite(events=events, fit_res=fit_res, components=components)

        if hasattr(pps, "plot_2d_counting_and_intensity"):
            pps.plot_2d_counting_and_intensity(
                t_start=t_start, T=T, warmup=warmup, n_grid=n_grid,
                out_dir=self.out_dir, save_name=f"{prefix}pp_2d_counting_intensity"
            )
        if hasattr(pps, "plot_branching_heatmap"):
            pps.plot_branching_heatmap(out_dir=self.out_dir, save_name=f"{prefix}pp_branching_heatmap")
        if hasattr(pps, "plot_kernel_response"):
            pps.plot_kernel_response(out_dir=self.out_dir, save_name=f"{prefix}pp_kernel_response")
        elif hasattr(pps, "plot_kernel_cumulative"):
            pps.plot_kernel_cumulative(out_dir=self.out_dir, save_name=f"{prefix}pp_kernel_cumulative")
        if hasattr(pps, "plot_time_rescaling_gof"):
            pps.plot_time_rescaling_gof(
                t_start=t_start, T=T, warmup=warmup,
                out_dir=self.out_dir, save_name=f"{prefix}pp_time_rescaling_gof"
            )

        return pps
