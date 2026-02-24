# diagnostics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import numpy as np

# Optional helpers from your project.
try:
    from hjb_theta_solver import (
        intensity_on_real_episode_2d, slice_event_counts, slice_mid_episode
    )
except Exception:
    intensity_on_real_episode_2d = None
    slice_event_counts = None
    slice_mid_episode = None



# -------------------------
# Time-axis helpers
# -------------------------
try:
    from config import MARKET_OPEN_SECONDS  # seconds from midnight
except Exception:
    MARKET_OPEN_SECONDS = 9.5 * 3600  # 09:30

def _sec_to_clock_str(sec, show_seconds=False):
    sec = float(sec) % 86400.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if show_seconds:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}"

def _choose_time_tick_step(span_seconds):
    span_seconds = abs(float(span_seconds))
    if span_seconds <= 15*60:
        return 60
    if span_seconds <= 60*60:
        return 5*60
    if span_seconds <= 3*60*60:
        return 15*60
    if span_seconds <= 8*60*60:
        return 30*60
    return 60*60

def apply_time_of_day_axis(ax, base_seconds=0.0, origin_seconds=MARKET_OPEN_SECONDS, show_seconds=False, rotate=30):
    # Treat axis units as seconds since market open. Labels show wall-clock time.
    x0, x1 = ax.get_xlim()
    step = _choose_time_tick_step(x1 - x0)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: _sec_to_clock_str(origin_seconds + base_seconds + x, show_seconds)))
    if rotate is not None:
        ax.tick_params(axis="x", rotation=rotate)
    return ax

class diagnostics:
    def __init__(self, symbol, data: pd.DataFrame, events: dict):
        # Better than "figures"+symbol (no separator)
        self.FIGDIR = Path("figures") / str(symbol)
        self.FIGDIR.mkdir(parents=True, exist_ok=True)

        self.lob_data = data
        self.events = events

    def _save(self, fig, name: str, dpi: int = 300):
        fig.savefig(self.FIGDIR / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.FIGDIR / f"{name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # -----------------------
    # A1: Mid + events
    # -----------------------
    def fig_mid_and_events(self, lob_data: pd.DataFrame, t_start: float, T: float, name="A1_mid_events"):
        if not np.isscalar(t_start):
            raise TypeError(
                f"t_start must be a number (float seconds). Got {type(t_start)}. "
                "This usually happens if you accidentally passed `events` as the second argument."
            )

        df = lob_data
        t = df["time"].to_numpy(float)
        m = df["mid_price"].to_numpy(float)

        mask = (t >= float(t_start)) & (t <= float(t_start) + float(T))
        tloc = t[mask] - float(t_start)
        mloc = m[mask]

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(tloc, mloc, linewidth=1.0, label="mid")

        # event stems
        for key in ("E_a", "E_b"):
            if key in self.events:
                ev = np.asarray(self.events[key], float)
                ev = ev[(ev >= float(t_start)) & (ev <= float(t_start) + float(T))] - float(t_start)
                if ev.size:
                    ax.vlines(ev, np.nanmin(mloc), np.nanmax(mloc), linewidth=0.6, alpha=0.2, label=key)

        ax.set_xlabel("Time")
        apply_time_of_day_axis(ax, base_seconds=t_start)
        ax.set_ylabel("mid price")
        ax.set_title("Episode mid-price and extracted events")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        self._save(fig, name)

    # -----------------------
    # Hawkes helpers
    # -----------------------
    def _as_beta_vec(self, beta, N: int):
        beta = np.asarray(beta, float).reshape(-1)
        if beta.size == 1:
            beta = np.full(N, float(beta[0]))
        if beta.size != N:
            raise ValueError(f"beta must have size {N}, got {beta.size}")
        return beta

    def _branching_matrix(self, A, beta):
        A = np.asarray(A, float)
        N = A.shape[0]
        beta = self._as_beta_vec(beta, N)
        return A / beta[None, :]  # divide each COLUMN j by beta_j

    def _rho_spectral_radius(self, M):
        M = np.asarray(M, float)
        eig = np.linalg.eigvals(M)
        return float(np.max(np.abs(eig)))

    # -----------------------
    # B1: Hawkes kernels (2D)
    # -----------------------

    def fig_compare_pnl_q(
        self,
        res_a,
        res_b,
        labels=("HJB", "FixedSkew"),
        name="E1b_compare_pnl_q",
        t0_seconds=None,
        plot = False,
    ):
        """Overlay inventory and relative PnL for two backtest result dicts.
        If time grids differ, interpolate res_b onto res_a's grid.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        t_a = np.asarray(res_a["time"], float)
        q_a = np.asarray(res_a["q"], float)
        pnl_a = np.asarray(res_a["pnl"], float)

        t_b = np.asarray(res_b["time"], float)
        q_b_raw = np.asarray(res_b["q"], float)
        pnl_b_raw = np.asarray(res_b["pnl"], float)

        same_grid = (t_a.shape == t_b.shape) and np.allclose(t_a, t_b, rtol=0.0, atol=1e-12)
        if same_grid:
            t = t_a
            q_b = q_b_raw
            pnl_b = pnl_b_raw
        else:
            t = t_a
            q_b = np.interp(t, t_b, q_b_raw)
            pnl_b = np.interp(t, t_b, pnl_b_raw)

        lab_a, lab_b = labels

        fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

        axs[0].plot(t, q_a, linewidth=1.7, label=f"q ({lab_a})")
        axs[0].plot(t, q_b, linewidth=1.7, linestyle="--", label=f"q ({lab_b})")
        axs[0].set_ylabel("inventory q")
        axs[0].legend(fontsize=8)

        axs[1].plot(t, pnl_a - pnl_a[0], linewidth=2.0, label=f"PnL ({lab_a})")
        axs[1].plot(t, pnl_b - pnl_b[0], linewidth=2.0, linestyle="--", label=f"PnL ({lab_b})")
        axs[1].set_ylabel("PnL (relative)")
        axs[1].set_xlabel("time (s)")
        axs[1].legend(fontsize=8)

        fig.suptitle("Backtest comparison: inventory and PnL")
        fig.tight_layout()
        if plot:
            plt.show(fig)
        self._save(fig, name)

    def fig_quote_level_on_lob(
        self,
        res,
        t_start: float,
        T: float,
        max_levels: int = 10,
        quote_tick: float = 0.01,
        name: str = "F1_quote_level_on_lob",
    ):
        """
        Plot where your policy quotes land relative to the historical LOB levels.

        Robust to whether res times are:
        - relative (0..T), or
        - absolute (same scale as lob_data['time']).

        Definition (competitive depth rank):
        Ask: level_a = min L s.t. ask_quote <= ask_price_L  (L=1 means at/inside best ask)
        Bid: level_b = min L s.t. bid_quote >= bid_price_L  (L=1 means at/inside best bid)
        If beyond max_levels => level = max_levels+1
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # --- pull time/bid/ask arrays robustly ---
        if "time" in res:
            t_raw = np.asarray(res["time"], float)
        elif "t_secs" in res:
            t_raw = np.asarray(res["t_secs"], float)
        elif "t" in res:
            t_raw = np.asarray(res["t"], float)
        else:
            raise KeyError("res must contain one of: 'time', 't_secs', or 't'.")

        bid_q = np.asarray(res["bid"], float)
        ask_q = np.asarray(res["ask"], float)

        t0 = float(t_start)
        T = float(T)

        # --- detect whether res time is relative or absolute ---
        # Heuristic: if most times lie in [-eps, T+eps], treat as relative.
        epsT = max(1e-6, 1e-3 * T)
        tmin = np.nanmin(t_raw)
        tmax = np.nanmax(t_raw)
        is_relative = (tmin >= -epsT) and (tmax <= T + epsT)

        if is_relative:
            t_loc = t_raw                      # 0..T
            t_abs = t0 + t_raw                 # align to lob_data time scale
            win_mask = (t_loc >= 0.0) & (t_loc <= T)
        else:
            t_abs = t_raw
            t_loc = t_raw - t0
            win_mask = (t_abs >= t0) & (t_abs <= t0 + T)

        # Validity mask (quotes finite + within window)
        mask = win_mask & np.isfinite(bid_q) & np.isfinite(ask_q)
        if not np.any(mask):
            raise ValueError(
                "No valid quote points after filtering. "
                f"(is_relative={is_relative}, t_raw in [{tmin:.3f},{tmax:.3f}], "
                f"window=[{t0:.3f},{(t0+T):.3f}])"
            )

        t_abs = t_abs[mask]
        t_loc = t_loc[mask]
        bid_q = bid_q[mask]
        ask_q = ask_q[mask]

        # --- map each quote time to last LOB snapshot <= t_abs ---
        df = self.lob_data
        t_lob = df["time"].to_numpy(float)

        # If lob time is relative too (rare), align it to absolute window
        lob_min = np.nanmin(t_lob)
        lob_max = np.nanmax(t_lob)
        lob_is_relative = (lob_min >= -epsT) and (lob_max <= (np.nanmax(t_abs - t0) + epsT))
        if lob_is_relative:
            # interpret lob times as 0.. and shift to match episode
            t_lob_abs = t0 + t_lob
        else:
            t_lob_abs = t_lob

        idx = np.searchsorted(t_lob_abs, t_abs, side="right") - 1
        idx = np.clip(idx, 0, len(t_lob_abs) - 1)

        max_levels = int(max_levels)
        max_levels = max(1, min(max_levels, 10))

        ask_mat = np.column_stack([df[f"ask_price_{L}"].to_numpy(float)[idx] for L in range(1, max_levels + 1)])
        bid_mat = np.column_stack([df[f"bid_price_{L}"].to_numpy(float)[idx] for L in range(1, max_levels + 1)])

        ask1 = ask_mat[:, 0]
        bid1 = bid_mat[:, 0]

        qt = float(quote_tick)
        eps = 0.5 * qt

        # inside-spread flags
        inside_spread_ask = ask_q < (ask1 - eps)
        inside_spread_bid = bid_q > (bid1 + eps)

        # --- compute level ranks ---
        level_a = np.full(len(t_abs), max_levels + 1, dtype=int)
        for L in range(1, max_levels + 1):
            cond = ask_q <= (ask_mat[:, L - 1] + eps)
            level_a = np.where((level_a == max_levels + 1) & cond, L, level_a)

        level_b = np.full(len(t_abs), max_levels + 1, dtype=int)
        for L in range(1, max_levels + 1):
            cond = bid_q >= (bid_mat[:, L - 1] - eps)
            level_b = np.where((level_b == max_levels + 1) & cond, L, level_b)

        # --- plotting ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        ax = axes[0]
        ax.plot(t_loc, level_b, lw=0.9, label="Bid level (1=touch)")
        ax.plot(t_loc, level_a, lw=0.9, label="Ask level (1=touch)")
        ax.set_ylabel("Quoted LOB level rank")
        ax.set_xlabel("Time")
        apply_time_of_day_axis(ax, base_seconds=t0)
        ax.set_ylim(0.5, max_levels + 1.5)
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.set_title("Policy quote level relative to historical LOB ladder")

        ax = axes[1]
        bins = np.arange(1, max_levels + 3) - 0.5
        ax.hist(level_b, bins=bins, alpha=0.7, label="Bid", density=True)
        ax.hist(level_a, bins=bins, alpha=0.7, label="Ask", density=True)
        xt = list(range(1, max_levels + 1)) + [max_levels + 1]
        xl = [f"L{L}" for L in range(1, max_levels + 1)] + [f">L{max_levels}"]
        ax.set_xticks(xt)
        ax.set_xticklabels(xl)
        ax.set_ylabel("Fraction of time")
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.set_title(f"Inside-spread rates: ask={inside_spread_ask.mean():.2%}, bid={inside_spread_bid.mean():.2%}")

        fig.tight_layout()
        self._save(fig, name)


    def fig_branching_heatmap(self, fit_res, name="B0_branching_heatmap", labels=("− (E_b)", "+ (E_a)")):
        """
        Plot the branching matrix K_ij = ∫_0^∞ φ_ij(t) dt.
        Under the exponential-kernel parametrization used in this project:
            φ_ij(t) = A_ij exp(-beta_j t)  =>  K_ij = A_ij / beta_j   (column-wise division).
        """
        A = np.asarray(fit_res.A, float)
        N = A.shape[0]
        beta = self._as_beta_vec(getattr(fit_res, "beta", 1.0), N)

        K = getattr(fit_res, "K", None)
        if K is None:
            K = self._branching_matrix(A, beta)

        rho = getattr(fit_res, "rho", None)
        if rho is None:
            rho = self._rho_spectral_radius(K)

        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        im = ax.imshow(K, aspect="equal")

        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("source component j")
        ax.set_ylabel("target component i")
        ax.set_title(rf"Branching matrix $K=A/\beta$ ,  $\rho(K)\approx {float(rho):.3f}$")

        for i in range(N):
            for j in range(N):
                ax.text(j, i, f"{K[i,j]:.3f}", ha="center", va="center")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("K_ij")

        fig.tight_layout()
        self._save(fig, name)

    def fig_hawkes_kernels_2d(
        self,
        fit_res,
        t_max=None,
        n=4000,
        name="B1_hawkes_kernels_2d",
        ylog=False,
        show_cum=True,
    ):
        A = np.asarray(fit_res.A, float)
        N = A.shape[0]
        beta = self._as_beta_vec(getattr(fit_res, "beta", 1.0), N)

        # Choose horizon automatically if not provided:
        # time to reach 99.5% of cumulative mass for the slowest decay
        if t_max is None:
            beta_min = float(np.min(beta))
            t995 = -np.log(1.0 - 0.995) / beta_min  # ~5.3/beta_min
            t_max = max(5 * t995, 0.010)            # >=10ms, usually 20–50ms

        t = np.linspace(0.0, float(t_max), int(n))

        # phi_ij(t) = A_ij * exp(-beta_j t)
        phi = A[:, :, None] * np.exp(-beta[None, :, None] * t[None, None, :])

        # Cumulative mass up to time t: ∫_0^t phi_ij(u) du = (A_ij/beta_j) * (1 - exp(-beta_j t))
        if show_cum:
            cum = (A[:, :, None] / beta[None, :, None]) * (1.0 - np.exp(-beta[None, :, None] * t[None, None, :]))

        fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        axs = axs.ravel()

        t_ms = 1e3 * t
        for k, (i, j) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            axs[k].plot(t_ms, phi[i, j, :], linewidth=1.8, label=r"$\phi_{%d,%d}$" % (i+1, j+1))
            if show_cum:
                ax2 = axs[k].twinx()
                ax2.plot(t_ms, cum[i, j, :], linewidth=1.2, linestyle="--", alpha=0.8, label="cum mass")
                ax2.set_ylabel("cum mass")
            axs[k].set_title(rf"$\phi_{{{i+1},{j+1}}}(t)$  ($\beta_{{{j+1}}}={beta[j]:.2g}$)")
            axs[k].set_xlabel("t (ms)")
            axs[k].set_ylabel("kernel value")
            if ylog:
                axs[k].set_yscale("log")

        K = getattr(fit_res, "K", None)
        if K is None:
            K = self._branching_matrix(A, beta)
        rho = getattr(fit_res, "rho", None)
        if rho is None:
            rho = self._rho_spectral_radius(K)

        fig.suptitle(rf"2D Hawkes kernels (zoomed)   $\rho(K)\approx {float(rho):.3f}$", y=1.02)
        fig.tight_layout()
        self._save(fig, name)


    # -----------------------
    # B3: time-rescaling residuals (QQ-style)
    # -----------------------
    def fig_hawkes_residual_qq(self, hc, fit_res, comp_names=None, name="B3_hawkes_residuals_qq"):
        if comp_names is None:
            comp_names = getattr(hc, "COMPONENTS", ["E_b", "E_a"])

        fig, ax = plt.subplots(figsize=(5.5, 5))
        for comp in comp_names:
            z = hc.time_rescaled_residuals(fit_res.mu, fit_res.A, fit_res.beta, comp)
            z = z[np.isfinite(z)]
            if z.size < 50:
                continue

            z_sorted = np.sort(z)
            u = (np.arange(1, z_sorted.size + 1) - 0.5) / z_sorted.size
            q_theory = -np.log(1.0 - u)
            ax.plot(q_theory, z_sorted, linewidth=1.0, label=comp)

        mx = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, mx], [0, mx], linewidth=1.0, linestyle="--")

        ax.set_xlabel("Exp(1) theoretical quantiles")
        ax.set_ylabel("Empirical rescaled residual quantiles")
        ax.set_title("Hawkes time-rescaling residual QQ check")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._save(fig, name)

    # -----------------------
    # C1: fill calibration curve (empirical + fitted)
    # -----------------------
    def _collect_fill_samples(self, tick_size=0.01, max_levels=3, max_exec_events=3000, seed=0):
        rng = np.random.default_rng(seed)
        df = self.lob_data

        exec_mask = df["lob_action"].isin([4, 5])
        exec_idx = df.index[exec_mask].to_numpy()
        if exec_idx.size > max_exec_events:
            exec_idx = rng.choice(exec_idx, size=max_exec_events, replace=False)

        # levels available
        L_avail = []
        for L in range(1, max_levels + 1):
            colsL = {f"ask_price_{L}", f"ask_size_{L}", f"bid_price_{L}", f"bid_size_{L}"}
            if colsL.issubset(df.columns):
                L_avail.append(L)

        deltas, pfill = [], []
        eps_px = 0.1 * float(tick_size) + 1e-12

        for i in exec_idx:
            m = float(df.at[i, "mid_price"])
            v = float(df.at[i, "lob_size"])
            px = float(df.at[i, "execution"])
            ask1 = float(df.at[i, "ask_price_1"])
            bid1 = float(df.at[i, "bid_price_1"])

            if not (np.isfinite(m) and np.isfinite(v) and v > 0 and np.isfinite(px) and np.isfinite(ask1) and np.isfinite(bid1)):
                continue

            buy = (px >= ask1 - eps_px)   # buy MO consumes asks
            sell = (px <= bid1 + eps_px)  # sell MO consumes bids
            if not (buy or sell):
                continue

            cum = 0.0
            if buy:
                for L in L_avail:
                    pL = float(df.at[i, f"ask_price_{L}"])
                    qL = float(df.at[i, f"ask_size_{L}"])
                    if not (np.isfinite(pL) and np.isfinite(qL) and qL > 0):
                        break
                    delta = pL - m
                    consume = v - cum
                    pf = 0.0 if consume <= 0 else float(np.clip(consume / qL, 0.0, 1.0))
                    if delta >= 0 and np.isfinite(delta):
                        deltas.append(delta)
                        pfill.append(pf)
                    cum += qL
            else:
                for L in L_avail:
                    pL = float(df.at[i, f"bid_price_{L}"])
                    qL = float(df.at[i, f"bid_size_{L}"])
                    if not (np.isfinite(pL) and np.isfinite(qL) and qL > 0):
                        break
                    delta = m - pL
                    consume = v - cum
                    pf = 0.0 if consume <= 0 else float(np.clip(consume / qL, 0.0, 1.0))
                    if delta >= 0 and np.isfinite(delta):
                        deltas.append(delta)
                        pfill.append(pf)
                    cum += qL

        deltas = np.asarray(deltas, float)
        pfill = np.asarray(pfill, float)
        m = np.isfinite(deltas) & np.isfinite(pfill)
        return deltas[m], pfill[m]
    
    def fig_fill_calibration(self, A_fill, k_fill, tick_size=0.01, max_levels=10, name="C1_fill_calibration", delta0_fill=None):
        """
        Plot fill-proxy calibration in the SAME variable used by the model:
            f(delta) = min(1, A exp(-k * max(delta - delta0, 0))).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        deltas, pfill = self._collect_fill_samples(tick_size=tick_size, max_levels=max_levels)
        if deltas.size == 0:
            raise ValueError("[Diagnostics] No fill samples available.")

        deltas = np.asarray(deltas, float)
        pfill  = np.asarray(pfill,  float)

        msk = np.isfinite(deltas) & np.isfinite(pfill) & (deltas >= 0)
        deltas = deltas[msk]
        pfill  = pfill[msk]

        if deltas.size == 0:
            raise ValueError("[Diagnostics] No finite fill samples after filtering.")

        # If caller didn't pass delta0_fill, estimate "touch distance" from lower tail
        if delta0_fill is None:
            delta0_fill = float(np.quantile(deltas, 0.05))  # robust proxy for spread/2

        # ---- binning (same as before, but on raw delta so axis stays "offset from mid") ----
        nbins = 40
        qs = np.linspace(0, 1, nbins + 1)
        edges = np.unique(np.quantile(deltas, qs))
        if edges.size < 3:
            edges = np.linspace(deltas.min(), deltas.max(), nbins + 1)

        bin_idx = np.digitize(deltas, edges) - 1
        bx, by = [], []
        for j in range(edges.size - 1):
            m = (bin_idx == j)
            if m.sum() < 50:
                continue
            bx.append(0.5 * (edges[j] + edges[j + 1]))
            by.append(np.mean(pfill[m]))

        bx = np.asarray(bx)
        by = np.asarray(by)

        # ---- fitted curve (THIS is the actual model) ----
        x = np.linspace(0.0, float(np.quantile(deltas, 0.995)), 300)
        yfit = np.minimum(1.0, float(A_fill) * np.exp(-float(k_fill) * np.maximum(x - float(delta0_fill), 0.0)))

        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        ax.scatter(bx, by, s=25, label="binned empirical fill")
        ax.plot(x, yfit, linewidth=2.5,
                label=r"fit: $\min(1,\;A e^{-k\max(\delta-\delta_0,0)})$")

        ax.set_title("Microstructure fill calibration")
        ax.set_xlabel(r"offset $\delta$ (price units, from mid)")
        ax.set_ylabel("fill probability proxy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self._save(fig, name)

    # -----------------------
    # C2: impact marking calibration
    # -----------------------
    def _collect_impact_samples(self, mid_tick=0.005, horizon_steps: int = 1):
        """
        Collect samples (rho, s, y) where:
        - s in {+1,-1} is MO sign (+1 buy MO, -1 sell MO)
        - rho is imbalance (either provided or computed)
        - y is 1 if mid moves in the MO direction by >= mid_tick over the next `horizon_steps` snapshots.

        IMPORTANT: uses forward move: mid[i+h]-mid[i], not backward diff.
        """
        df = self.lob_data

        is_exec = df["lob_action"].isin([4, 5])
        idx = df.index[is_exec].to_numpy()

        # +1 buy MO, -1 sell MO (keep your convention)
        s = (-df.loc[is_exec, "lob_dir"].astype(float)).to_numpy()
        ok = np.isfinite(s) & (np.abs(s) == 1.0)
        idx = idx[ok]
        s = s[ok]

        # forward move requires i+h within bounds
        h = int(max(1, horizon_steps))
        idx = idx[idx <= (len(df) - 1 - h)]
        s = s[: len(idx)]

        # imbalance proxy
        if "rho_5L" in df.columns and df["rho_5L"].notna().any():
            rho = df.loc[idx, "rho_5L"].astype(float).to_numpy()
        else:
            vb = df.loc[idx, "bid_size"].astype(float).to_numpy()
            va = df.loc[idx, "ask_size"].astype(float).to_numpy()
            rho = (vb - va) / (vb + va + 1e-12)

        rho = np.clip(rho, -1.0, 1.0)

        mid = df["mid_price"].astype(float).to_numpy()
        dmid_fwd = mid[idx + h] - mid[idx]

        tol = float(mid_tick) - 1e-12
        y = (((s > 0) & (dmid_fwd >= tol)) | ((s < 0) & (dmid_fwd <= -tol))).astype(float)

        m = np.isfinite(rho) & np.isfinite(s) & np.isfinite(y)
        return rho[m], s[m], y[m]
    
    def fig_impact_calibration(
        self,
        p_bar,
        k_imp,
        theta_imp,
        mid_tick=0.005,
        name="C2_impact_calibration",
        horizon_steps: int = 1,
        do_refit: bool = True,
    ):
        """
        Plot + calibrate directional marking probabilities vs imbalance.

        - Uses forward mid move over `horizon_steps` snapshots.
        - Adds binomial error bars and bin counts.
        - Optionally refits separate (pbar,k,theta) for buy/sell curves via weighted LS on binned rates.
        """

        rho, s, y = self._collect_impact_samples(mid_tick=mid_tick, horizon_steps=horizon_steps)

        edges = np.linspace(-1, 1, 21)
        centers = 0.5 * (edges[:-1] + edges[1:])

        def binned_stats(mask):
            rate = np.full_like(centers, np.nan, dtype=float)
            nbin = np.zeros_like(centers, dtype=int)
            se   = np.full_like(centers, np.nan, dtype=float)
            for i in range(len(centers)):
                mm = mask & (rho >= edges[i]) & (rho < edges[i + 1])
                n = int(mm.sum())
                nbin[i] = n
                if n >= 50:
                    p = float(np.mean(y[mm]))
                    rate[i] = p
                    se[i] = np.sqrt(max(p * (1.0 - p), 1e-12) / n)
            return rate, se, nbin

        buy_rate, buy_se, buy_n = binned_stats(s > 0)
        sell_rate, sell_se, sell_n = binned_stats(s < 0)

        # ----- "given" param curves (your current model) -----
        rgrid = np.linspace(-1, 1, 400)
        p_plus_given  = float(p_bar) / (1.0 + np.exp(-float(k_imp) * (rgrid - float(theta_imp))))
        p_minus_given = float(p_bar) / (1.0 + np.exp(+float(k_imp) * (rgrid - float(theta_imp))))

        # ----- optional refit: separate params for buy/sell to match the dots -----
        def _model_plus(r, pbar, k, th):
            return pbar / (1.0 + np.exp(-k * (r - th)))

        def _model_minus(r, pbar, k, th):
            return pbar / (1.0 + np.exp(+k * (r - th)))

        def _refit_curve(x, yhat, w, is_plus: bool):
            # weighted least squares on binned means
            m = np.isfinite(yhat) & np.isfinite(w) & (w > 0)
            if m.sum() < 5:
                return None  # not enough bins

            x = x[m]; yhat = yhat[m]; w = w[m].astype(float)

            # initial guesses
            p0 = float(min(0.99, max(np.nanmax(yhat) + 0.05, 0.05)))
            k0 = 2.0
            th0 = 0.0

            try:
                from scipy.optimize import minimize

                def obj(v):
                    pbar, k, th = v
                    if not (1e-4 < pbar < 0.999 and 1e-4 < k < 100 and -1.0 <= th <= 1.0):
                        return 1e9
                    pred = (_model_plus(x, pbar, k, th) if is_plus else _model_minus(x, pbar, k, th))
                    return float(np.sum(w * (pred - yhat) ** 2))

                res = minimize(
                    obj,
                    x0=np.array([p0, k0, th0]),
                    method="L-BFGS-B",
                    bounds=[(1e-4, 0.999), (1e-4, 100.0), (-1.0, 1.0)],
                )
                if not res.success:
                    return None
                return tuple(map(float, res.x))

            except Exception:
                # fallback: coarse grid search (no scipy)
                best = None
                best_val = np.inf
                for th in np.linspace(-0.5, 0.5, 41):
                    for k in np.logspace(-1, 1, 40):  # 0.1..10
                        for pbar in np.linspace(max(0.05, np.nanmax(yhat)), 0.95, 20):
                            pred = (_model_plus(x, pbar, k, th) if is_plus else _model_minus(x, pbar, k, th))
                            val = float(np.sum(w * (pred - yhat) ** 2))
                            if val < best_val:
                                best_val = val
                                best = (float(pbar), float(k), float(th))
                return best

        p_plus_fit = p_minus_fit = None
        if do_refit:
            # weights = bin counts
            p_plus_fit  = _refit_curve(centers, buy_rate,  buy_n,  is_plus=True)
            p_minus_fit = _refit_curve(centers, sell_rate, sell_n, is_plus=False)

        # build fitted curves if available
        p_plus_refit = p_minus_refit = None
        if p_plus_fit is not None:
            p_plus_refit = _model_plus(rgrid, *p_plus_fit)
        if p_minus_fit is not None:
            p_minus_refit = _model_minus(rgrid, *p_minus_fit)

        # ----- plot -----
        fig, ax = plt.subplots(figsize=(8.5, 5))

        # empirical with error bars
        ax.errorbar(centers, buy_rate,  yerr=buy_se,  fmt="o", capsize=2, label="empirical buy (up) rate")
        ax.errorbar(centers, sell_rate, yerr=sell_se, fmt="o", capsize=2, label="empirical sell (down) rate")

        # given curves (dashed)
        ax.plot(rgrid, p_plus_given,  linewidth=2.0, linestyle="--", label=r"given fit $p_+(\rho)$")
        ax.plot(rgrid, p_minus_given, linewidth=2.0, linestyle="--", label=r"given fit $p_-(\rho)$")

        # refit curves (solid), if any
        if p_plus_refit is not None:
            ax.plot(rgrid, p_plus_refit, linewidth=2.5, label=r"refit $p_+(\rho)$")
        if p_minus_refit is not None:
            ax.plot(rgrid, p_minus_refit, linewidth=2.5, label=r"refit $p_-(\rho)$")

        base_all  = float(np.mean(y)) if y.size else np.nan
        base_buy  = float(np.mean(y[s > 0])) if np.any(s > 0) else np.nan
        base_sell = float(np.mean(y[s < 0])) if np.any(s < 0) else np.nan

        ax.set_xlabel(r"imbalance $\rho$")
        ax.set_ylabel("directional move probability")
        ax.set_title(
            f"Microstructure impact marking calibration (h={horizon_steps})\n"
            f"base: all={base_all:.3f}, buy={base_buy:.3f}, sell={base_sell:.3f}"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        # optional: show total counts
        ax.text(
            0.02, 0.02,
            f"Nbuy={int(np.sum(s>0))}, Nsell={int(np.sum(s<0))}",
            transform=ax.transAxes,
            fontsize=9,
            alpha=0.8
        )

        fig.tight_layout()
        self._save(fig, name)

        # print refit params for logging
        if do_refit:
            print("[impact refit] plus (pbar,k,theta):", p_plus_fit)
            print("[impact refit] minus(pbar,k,theta):", p_minus_fit)

    # -----------------------
    # D1: policy slices
    # -----------------------
    def fig_policy_slices(self, policy, t, S, lam_plus, lam_minus, q_grid, name="D1_policy_slices"):
        q_grid = np.asarray(q_grid, float)
        del_a = np.zeros_like(q_grid)
        del_b = np.zeros_like(q_grid)

        for i, q in enumerate(q_grid):
            b, a = policy(t=t, S=S, q=float(q), lam_plus=float(lam_plus), lam_minus=float(lam_minus))
            del_a[i] = max(a - S, 0.0)
            del_b[i] = max(S - b, 0.0)

        skew = 0.5 * (del_a - del_b)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(q_grid, del_b, linewidth=2.0, label=r"$\delta_b(q)$")
        ax.plot(q_grid, del_a, linewidth=2.0, label=r"$\delta_a(q)$")
        ax.plot(q_grid, skew, linewidth=2.0, linestyle="--", label="skew(q)")

        ax.set_xlabel("inventory q")
        ax.set_ylabel("offset (price units)")
        ax.set_title("HJB policy slice (fixed t, S, λ±)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._save(fig, name)

    # -----------------------
    # E1: backtest path
    # -----------------------
    def fig_backtest_path(self, res, name="E1_backtest_path", t0_seconds=None, plot = False):
        t = np.asarray(res["time"], float)
        mid = np.asarray(res["mid"], float)
        bid = np.asarray(res["bid"], float)
        ask = np.asarray(res["ask"], float)
        q = np.asarray(res["q"], float)
        pnl = np.asarray(res["pnl"], float)

        # If caller didn't pass an anchor time, try to read it from the result dict.
        if t0_seconds is None and isinstance(res, dict) and ("t0_seconds" in res):
            t0_seconds = res["t0_seconds"]

        fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

        axs[0].plot(t, mid, linewidth=1.2, label="mid")
        axs[0].plot(t, bid, linewidth=0.9, label="bid quote")
        axs[0].plot(t, ask, linewidth=0.9, label="ask quote")
        axs[0].set_ylabel("price")
        axs[0].legend(fontsize=8)

        axs[1].plot(t, q, linewidth=1.2)
        axs[1].set_ylabel("inventory q")

        axs[2].plot(t, pnl - pnl[0], linewidth=1.5)
        axs[2].set_ylabel("PnL (relative)")
        axs[2].set_xlabel("Time")
        apply_time_of_day_axis(axs[2], base_seconds=float(t0_seconds) if t0_seconds is not None else 0.0)

        fig.suptitle("Backtest path: quotes, inventory, PnL")
        fig.tight_layout()
        if plot:
            plt.show(fig)
        self._save(fig, name)

    

    # -----------------------
    # E2: PnL decomposition (flexible key names)
    # -----------------------
    def fig_pnl_decomposition(self, res, name="E2_pnl_decomposition", t0_seconds=None, plot = False):
        t = np.asarray(res["time"], float)

        # If caller didn't pass an anchor time, try to read it from the result dict.
        if t0_seconds is None and isinstance(res, dict) and ("t0_seconds" in res):
            t0_seconds = res["t0_seconds"]

        # accept either naming convention
        if "d_trade" in res and "d_mtm" in res:
            d_trade = np.asarray(res["d_trade"], float)
            d_mtm = np.asarray(res["d_mtm"], float)
        elif "trade_pnl_step" in res and "inv_pnl_step" in res:
            d_trade = np.asarray(res["trade_pnl_step"], float)
            d_mtm = np.asarray(res["inv_pnl_step"], float)
        else:
            raise KeyError("Need either (d_trade, d_mtm) OR (trade_pnl_step, inv_pnl_step) in res.")

        trade_cum = np.cumsum(d_trade)
        mtm_cum = np.cumsum(d_mtm)
        total = trade_cum + mtm_cum

        # align time if needed
        tt = t[-len(trade_cum):]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tt, trade_cum, linewidth=2.0, label="cumulative trade PnL")
        ax.plot(tt, mtm_cum, linewidth=2.0, label="cumulative inventory MTM")
        ax.plot(tt, total, linewidth=2.5, linestyle="--", label="total")
        ax.set_xlabel("Time")
        apply_time_of_day_axis(ax, base_seconds=float(t0_seconds) if t0_seconds is not None else 0.0)
        ax.set_ylabel("PnL contribution")
        ax.set_title("PnL decomposition")
        ax.legend(fontsize=8)
        fig.tight_layout()
        if plot:
            plt.show(fig)
        self._save(fig, name)

    # -----------------------
    # B0: intraday seasonality
    # -----------------------
    def _events_in_range(self, ev, t0, t1):
        ev = np.asarray(ev, float)
        return ev[(ev >= float(t0)) & (ev < float(t1))]

    def fig_intraday_seasonality(self, day_start=0.0, day_end=23400.0, bin_sec=300, name="B0_intraday_seasonality"):
        edges = np.arange(float(day_start), float(day_end) + float(bin_sec), float(bin_sec), dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        fig, ax = plt.subplots(figsize=(9, 3))
        for key, lab in (("E_a", "buy MOs (E_a)"), ("E_b", "sell MOs (E_b)")):
            if key not in self.events:
                continue
            ev = self._events_in_range(self.events[key], day_start, day_end)
            counts, _ = np.histogram(ev, bins=edges)
            rate = counts / width
            ax.plot(centers, rate, linewidth=1.8, label=lab)

        ax.set_xlabel("Time")
        apply_time_of_day_axis(ax, base_seconds=0.0)
        ax.set_ylabel("event rate (1/s)")
        ax.set_title(f"Intraday seasonality: binned event rates (bin={int(bin_sec)}s)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._save(fig, name)

    def fig_backtest_path_classic(
        self,
        res,
        t_start: float = 0.0,
        T: Optional[float] = None,
        Q_max: Optional[float] = None,
        name: str = "E1_backtest_path",
    ):
        """
        3-panel plot:
        (1) Bid/Ask quotes + Mid
        (2) Inventory q with optional +/- Q_max bands
        (3) Cumulative PnL (relative)

        Robust to time being relative (0..T) or absolute.
        """

        import numpy as np
        import matplotlib.pyplot as plt

        # --- time handling ---
        if "time" in res:
            t_raw = np.asarray(res["time"], float)
        elif "t" in res:
            t_raw = np.asarray(res["t"], float)
        else:
            raise KeyError("res must contain 'time' or 't'.")

        # detect relative vs absolute
        t0 = float(t_start)
        if T is None:
            # if we don't know T, infer from res horizon
            T_inf = float(np.nanmax(t_raw) - np.nanmin(t_raw))
            T = T_inf if T_inf > 0 else 1.0
        T = float(T)

        epsT = max(1e-6, 1e-3 * T)
        is_relative = (np.nanmin(t_raw) >= -epsT) and (np.nanmax(t_raw) <= T + epsT)

        if is_relative:
            t_loc = t_raw
            # (optional) filter to [0,T]
            mask_t = (t_loc >= 0.0) & (t_loc <= T)
        else:
            # absolute -> convert to local time from episode start
            t_loc = t_raw - t0
            mask_t = (t_raw >= t0) & (t_raw <= t0 + T)

        # --- series ---
        mid = np.asarray(res["mid"], float)
        bid = np.asarray(res["bid"], float)
        ask = np.asarray(res["ask"], float)
        q   = np.asarray(res["q"], float)

        # optional pnl
        pnl = res.get("pnl", None)
        if pnl is not None:
            pnl = np.asarray(pnl, float)
            pnl_rel = pnl - pnl[0]
        else:
            # reconstruct if steps exist
            if ("trade_pnl_step" in res and "inv_pnl_step" in res):
                d_trade = np.asarray(res["trade_pnl_step"], float)
                d_mtm   = np.asarray(res["inv_pnl_step"], float)
            elif ("d_trade" in res and "d_mtm" in res):
                d_trade = np.asarray(res["d_trade"], float)
                d_mtm   = np.asarray(res["d_mtm"], float)
            else:
                raise KeyError("Need 'pnl' OR (trade_pnl_step, inv_pnl_step) OR (d_trade, d_mtm) in res.")
            pnl_rel = np.concatenate([[0.0], np.cumsum(d_trade + d_mtm)])

        # align lengths safely
        n = min(len(t_loc), len(mid), len(bid), len(ask), len(q), len(pnl_rel))
        t_loc = t_loc[:n]
        mid = mid[:n]; bid = bid[:n]; ask = ask[:n]; q = q[:n]; pnl_rel = pnl_rel[:n]
        mask = mask_t[:n] if mask_t is not None else np.ones(n, dtype=bool)

        t_loc = t_loc[mask]
        mid = mid[mask]; bid = bid[mask]; ask = ask[mask]; q = q[mask]; pnl_rel = pnl_rel[mask]

        # --- plotting (style matches your example) ---
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # (1) quotes + mid
        axs[0].plot(t_loc, bid, linewidth=1.0, alpha=0.6, label="Bid")
        axs[0].plot(t_loc, ask, linewidth=1.0, alpha=0.6, label="Ask")
        axs[0].plot(t_loc, mid, linewidth=2.0, linestyle="--", label="Mid")
        axs[0].legend(loc="lower left", fontsize=9)

        # (2) inventory
        axs[1].plot(t_loc, q, linewidth=2.0)
        axs[1].set_title("Inventory Evolution")
        axs[1].set_ylabel("Inventory q")
        if Q_max is not None:
            Qm = float(Q_max)
            axs[1].axhline(+Qm, linestyle="--", linewidth=1.5, alpha=0.6)
            axs[1].axhline(-Qm, linestyle="--", linewidth=1.5, alpha=0.6)

        # (3) pnl
        axs[2].plot(t_loc, pnl_rel, linewidth=2.0)
        axs[2].set_title("Cumulative Profit & Loss")
        axs[2].set_ylabel("PnL")
        axs[2].set_xlabel("Time")
        apply_time_of_day_axis(axs[2], base_seconds=float(t_start))

        fig.tight_layout()
        self._save(fig, name)

    def fig_hawkes_intensity_episode(
        self,
        fit_res,
        t_start: float,
        T: float,
        n_grid_points: int = 2000,
        name: str = "B2_hawkes_intensity_episode",
    ):
        """
        Plot λ_-(t) and λ_+(t) on the REAL episode path using intensity_on_real_episode_2d.
        """
        if intensity_on_real_episode_2d is None:
            raise RuntimeError("intensity_on_real_episode_2d is not available (import failed).")

        t0 = float(t_start)
        T = float(T)

        t_grid, lam_plus, lam_minus = intensity_on_real_episode_2d(
            self.events, fit_res, t0, T, int(n_grid_points)
        )

        # event times (episode-local) for stems
        Eb = np.asarray(self.events.get("E_b", []), float)
        Ea = np.asarray(self.events.get("E_a", []), float)
        Eb = Eb[(Eb >= t0) & (Eb <= t0 + T)] - t0
        Ea = Ea[(Ea >= t0) & (Ea <= t0 + T)] - t0

        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(t_grid, lam_minus, lw=1.6, label=r"$\lambda_-(t)$ (E_b)")
        ax.plot(t_grid, lam_plus,  lw=1.6, label=r"$\lambda_+(t)$ (E_a)")

        ymax = float(np.nanmax([np.nanmax(lam_plus), np.nanmax(lam_minus), 1.0]))
        stem_h = 0.15 * ymax
        if Eb.size: ax.vlines(Eb, 0.0, stem_h, lw=0.6, alpha=0.25)
        if Ea.size: ax.vlines(Ea, 0.0, stem_h, lw=0.6, alpha=0.25)

        ax.set_xlabel("Time")
        apply_time_of_day_axis(ax, base_seconds=t_start)
        ax.set_ylabel("intensity (1/s)")
        ax.set_title("Hawkes intensities on the real episode (with event times)")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")

        fig.tight_layout()
        self._save(fig, name)



    # -----------------------
    # user hook
    # -----------------------
    def make_all_figures(
        self,
        fit_res,
        t_start: float,
        T: float,
        policy=None,
        res=None,
        A_fill=None,
        k_fill=None,
        p_bar=None,
        k_imp=None,
        theta_imp=None,
        quote_tick=0.01,
        mid_tick=0.005,
        hc_for_residuals=None,
    ):
        # FIX 1: don't pass self.events as t_start
        self.fig_mid_and_events(self.lob_data, t_start, T)

        self.fig_hawkes_kernels_2d(fit_res)
        self.fig_branching_heatmap(fit_res)

        # FIX 2: signature is (fit_res, t_start, T)
        if intensity_on_real_episode_2d is not None:
            self.fig_hawkes_intensity_episode(fit_res, t_start, T)

        if hc_for_residuals is not None:
            self.fig_hawkes_residual_qq(hc_for_residuals, fit_res)

        # FIX 3: don't pass self.lob_data into these methods
        if (A_fill is not None) and (k_fill is not None):
            self.fig_fill_calibration(A_fill, k_fill, tick_size=quote_tick, max_levels=10)

        if (p_bar is not None) and (k_imp is not None) and (theta_imp is not None):
            self.fig_impact_calibration(p_bar, k_imp, theta_imp,
                mid_tick=mid_tick, horizon_steps=5, do_refit=False)


        if policy is not None:
            q_grid = np.arange(-30, 31, 2)
            S0 = float(np.nanmedian(self.lob_data["mid_price"].values))

            # For a quick slice, use mu as a proxy baseline (or pass explicit λ± if you prefer)
            mu = getattr(fit_res, "mu", None)
            lam_minus0 = float(mu[0]) if (mu is not None and len(mu) >= 2) else 1.0
            lam_plus0  = float(mu[1]) if (mu is not None and len(mu) >= 2) else 1.0

            self.fig_policy_slices(policy, t=0.0, S=S0, lam_plus=lam_plus0, lam_minus=lam_minus0, q_grid=q_grid)

        if res is not None:
            self.fig_backtest_path(res, t0_seconds=t_start)

            if (("d_trade" in res and "d_mtm" in res) or ("trade_pnl_step" in res and "inv_pnl_step" in res)):
                self.fig_pnl_decomposition(res, t0_seconds=t_start)

        self.fig_intraday_seasonality(day_start=1800, day_end=23400, bin_sec=300)

        self.fig_quote_level_on_lob(res, t_start=t_start, T=T, max_levels=10, quote_tick=quote_tick)

        self.fig_backtest_path_classic(res, t_start=t_start, T=T, Q_max=20)

    def backtest_report(
        self,
        res: Dict[str, Any],
        *,
        params: Optional[Any] = None,
        sol: Optional[Any] = None,
        policy: Optional[Callable[..., Tuple[float, float]]] = None,
        # fill model (optional)
        A_fill: Optional[float] = None,
        k_fill: Optional[float] = None,
        N_plus_counts: Optional[np.ndarray] = None,
        N_minus_counts: Optional[np.ndarray] = None,
        # intensity arrays for grid-hit diagnostics (optional)
        lam_plus_real: Optional[Sequence[float]] = None,
        lam_minus_real: Optional[Sequence[float]] = None,
        # policy probe settings
        q_test: Tuple[float, float] = (+10.0, -10.0),
        i_probe: int = 50,
        eps_grid: float = 1e-12,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Diagnostics report for a backtest result dict `res`.

        Expectations:
        - res contains arrays: mid/bid/ask/q/time/pnl (some may have alternate names)
        - optional arrays: X, trade_pnl_step, inv_pnl_step, H_a, H_b, lam_plus, lam_minus
        - policy signature: policy(t=..., S=..., q=..., lam_plus=..., lam_minus=...) -> (bid, ask)

        Returns:
        dict of scalar diagnostics + small policy probe dicts (safe to log / json).
        """

        # ---------------- helpers ----------------
        def _arr(key: str, dtype=float) -> Optional[np.ndarray]:
            if key in res and res[key] is not None:
                return np.asarray(res[key], dtype=dtype)
            return None

        def _coalesce_arr(keys: Sequence[str], dtype=float) -> Optional[np.ndarray]:
            for k in keys:
                a = _arr(k, dtype=dtype)
                if a is not None:
                    return a
            return None

        def _get_float_attr(obj: Any, name: str) -> Optional[float]:
            if obj is None:
                return None
            v = getattr(obj, name, None)
            try:
                return None if v is None else float(v)
            except Exception:
                return None

        def _finite_mean(x: Optional[np.ndarray]) -> Optional[float]:
            if x is None or x.size == 0:
                return None
            return float(np.nanmean(x))

        def _finite_ptp(x: Optional[np.ndarray]) -> Optional[float]:
            if x is None or x.size == 0:
                return None
            return float(np.nanmax(x) - np.nanmin(x))

        def _fmt(x: Optional[float]) -> str:
            if x is None or not np.isfinite(x):
                return "n/a"
            return f"{x:.6g}"

        # -------------- core series --------------
        time = _coalesce_arr(["time", "t"], dtype=float)
        mid  = _coalesce_arr(["mid", "mid_price", "midprice"], dtype=float)
        bid  = _coalesce_arr(["bid", "bid_quote"], dtype=float)
        ask  = _coalesce_arr(["ask", "ask_quote"], dtype=float)
        q    = _coalesce_arr(["q", "inventory"], dtype=float)
        X    = _coalesce_arr(["X", "cash"], dtype=float)
        pnl  = _coalesce_arr(["pnl", "PnL"], dtype=float)

        if mid is None or bid is None or ask is None or q is None:
            missing = []
            if mid is None: missing.append("mid/mid_price")
            if bid is None: missing.append("bid")
            if ask is None: missing.append("ask")
            if q is None:   missing.append("q")
            raise KeyError(f"backtest_report: missing required series in res: {missing}")

        n = int(len(mid))
        if len(bid) != n or len(ask) != n or len(q) != n:
            raise ValueError(
                f"backtest_report: inconsistent lengths: "
                f"len(mid)={len(mid)}, len(bid)={len(bid)}, len(ask)={len(ask)}, len(q)={len(q)}"
            )
        n1 = max(n - 1, 0)

        # -------------- micro quantities --------------
        spread = (ask[:n1] - bid[:n1]) if n1 > 0 else np.asarray([], float)
        delta_a = np.maximum(ask[:n1] - mid[:n1], 0.0) if n1 > 0 else np.asarray([], float)
        delta_b = np.maximum(mid[:n1] - bid[:n1], 0.0) if n1 > 0 else np.asarray([], float)
        skew_path = (0.5 * (ask[:n1] + bid[:n1]) - mid[:n1]) if n1 > 0 else np.asarray([], float)

        # -------------- wealth identity --------------
        wealth = None
        wealth_ptp = None
        identity_err_max = None
        if X is not None and len(X) == n:
            wealth = X + q * mid
            wealth_ptp = _finite_ptp(wealth)
            if pnl is not None and len(pnl) == n:
                identity_err_max = float(np.nanmax(np.abs(wealth - pnl)))

        # -------------- pnl decomposition --------------
        trade_step = _coalesce_arr(["trade_pnl_step", "d_trade", "trade_step"], dtype=float)
        inv_step   = _coalesce_arr(["inv_pnl_step", "d_mtm", "mtm_step"], dtype=float)

        trade_cum = float(np.nansum(trade_step)) if trade_step is not None else 0.0
        mtm_cum   = float(np.nansum(inv_step))   if inv_step   is not None else 0.0
        total_cum = trade_cum + mtm_cum

        pnl_rel_end = None
        if pnl is not None and len(pnl) == n:
            pnl_rel_end = float(pnl[-1] - pnl[0])

        # -------------- execution quality (needs fills H_a/H_b) --------------
        Ha = _coalesce_arr(["H_a", "Ha", "fills_a"], dtype=float)
        Hb = _coalesce_arr(["H_b", "Hb", "fills_b"], dtype=float)

        avg_sell_capture = avg_buy_capture = None
        fills_sells = fills_buys = None
        sell_events = buy_events = None

        if Ha is not None and Hb is not None and n1 > 0:
            Ha = np.asarray(Ha, float).reshape(-1)
            Hb = np.asarray(Hb, float).reshape(-1)
            m = min(len(Ha), len(Hb), n1)

            Ha = Ha[:m]
            Hb = Hb[:m]

            # adverse selection aware capture
            sell_cap = ask[:m] - mid[1:m+1]  # sell at ask, compare to next mid
            buy_cap  = mid[1:m+1] - bid[:m]  # buy at bid, compare to next mid

            sum_Ha = float(np.nansum(Ha))
            sum_Hb = float(np.nansum(Hb))

            avg_sell_capture = float(np.nansum(Ha * sell_cap) / max(sum_Ha, 1.0))
            avg_buy_capture  = float(np.nansum(Hb * buy_cap)  / max(sum_Hb, 1.0))

            fills_sells = int(round(sum_Ha))
            fills_buys  = int(round(sum_Hb))
            sell_events = int(np.where(Ha > 0)[0].size)
            buy_events  = int(np.where(Hb > 0)[0].size)

        # -------------- fill model implied probs + expected fills (optional) --------------
        mean_prob_a = mean_prob_b = None
        exp_fills_a = exp_fills_b = None

        if (A_fill is not None) and (k_fill is not None) and n1 > 0:
            Af = float(A_fill)
            kf = float(k_fill)

            prob_a = np.minimum(1.0, Af * np.exp(-kf * delta_a))
            prob_b = np.minimum(1.0, Af * np.exp(-kf * delta_b))

            mean_prob_a = float(np.nanmean(prob_a)) if prob_a.size else None
            mean_prob_b = float(np.nanmean(prob_b)) if prob_b.size else None

            if (N_plus_counts is not None) and (N_minus_counts is not None):
                Np = np.asarray(N_plus_counts, float).reshape(-1)[:n1]
                Nm = np.asarray(N_minus_counts, float).reshape(-1)[:n1]
                mm = min(len(Np), len(Nm), len(prob_a), len(prob_b))
                exp_fills_a = float(np.nansum(prob_a[:mm] * Np[:mm]))
                exp_fills_b = float(np.nansum(prob_b[:mm] * Nm[:mm]))

        # -------------- grid-hit fraction for intensities (optional) --------------
        if lam_plus_real is None:
            lam_plus_real = res.get("lam_plus", None)
        if lam_minus_real is None:
            lam_minus_real = res.get("lam_minus", None)

        lp_hit = lm_hit = None
        if sol is not None and lam_plus_real is not None and hasattr(sol, "lam_plus_grid"):
            lp = np.asarray(lam_plus_real, float).reshape(-1)
            top = float(np.asarray(sol.lam_plus_grid, float).reshape(-1)[-1])
            lp_hit = float(np.mean(lp >= top * (1.0 - eps_grid)))

        if sol is not None and lam_minus_real is not None and hasattr(sol, "lam_minus_grid"):
            lm = np.asarray(lam_minus_real, float).reshape(-1)
            top = float(np.asarray(sol.lam_minus_grid, float).reshape(-1)[-1])
            lm_hit = float(np.mean(lm >= top * (1.0 - eps_grid)))

        # -------------- policy probes (optional) --------------
        policy_t0: Dict[str, Any] = {}
        policy_probe: Dict[str, Any] = {}

        if policy is not None:
            mid0 = float(mid[0])
            lp0 = float(np.asarray(lam_plus_real, float)[0]) if lam_plus_real is not None else 0.0
            lm0 = float(np.asarray(lam_minus_real, float)[0]) if lam_minus_real is not None else 0.0

            for qq in q_test:
                b, a = policy(t=0.0, S=mid0, q=float(qq), lam_plus=lp0, lam_minus=lm0)
                b = float(b); a = float(a)
                policy_t0[f"q={qq:g}"] = {
                    "bid": b,
                    "ask": a,
                    "delta_b": max(mid0 - b, 0.0),
                    "delta_a": max(a - mid0, 0.0),
                    "skew": 0.5 * (a + b) - mid0,
                }

            ii = int(np.clip(i_probe, 0, n - 1))
            t_probe = float(time[ii]) if time is not None and len(time) == n else float(ii)
            S_probe = float(mid[ii])
            lp_i = float(np.asarray(lam_plus_real, float)[ii]) if lam_plus_real is not None else 0.0
            lm_i = float(np.asarray(lam_minus_real, float)[ii]) if lam_minus_real is not None else 0.0

            for qq in q_test:
                b, a = policy(t=t_probe, S=S_probe, q=float(qq), lam_plus=lp_i, lam_minus=lm_i)
                b = float(b); a = float(a)
                policy_probe[f"q={qq:g}"] = {
                    "i": ii,
                    "t": t_probe,
                    "S": S_probe,
                    "bid": b,
                    "ask": a,
                    "delta_b": max(S_probe - b, 0.0),
                    "delta_a": max(a - S_probe, 0.0),
                    "skew": 0.5 * (a + b) - S_probe,
                }

        # -------------- correlations --------------
        corr_q_skew = None
        if n1 > 2 and skew_path.size and q.size >= n1:
            qq = np.asarray(q[:n1], float)
            if np.nanstd(qq) > 0 and np.nanstd(skew_path) > 0:
                corr_q_skew = float(np.corrcoef(qq, skew_path)[0, 1])

        q_min = float(np.nanmin(q)) if q is not None and q.size else None
        q_max = float(np.nanmax(q)) if q is not None and q.size else None
        mean_skew = _finite_mean(skew_path)

        fills_total = None
        if fills_sells is not None and fills_buys is not None:
            fills_total = int(fills_sells) + int(fills_buys)

        out: Dict[str, Any] = {
            "params": {
                "gamma": _get_float_attr(params, "gamma"),
                "eta": _get_float_attr(params, "eta"),
                "eta_T": _get_float_attr(params, "eta_T"),
                "eta_run": _get_float_attr(params, "eta_run"),
            },
            "pnl": {
                "trade_cum": trade_cum,
                "mtm_cum": mtm_cum,
                "total_cum": total_cum,
                "pnl_rel_end": pnl_rel_end,
                "wealth_ptp": wealth_ptp,
                "identity_err_max": identity_err_max,
            },
            "fills": {
                "mean_prob_a": mean_prob_a,
                "mean_prob_b": mean_prob_b,
                "exp_fills_a": exp_fills_a,
                "exp_fills_b": exp_fills_b,
            },
            "micro": {
            "mean_spread": _finite_mean(spread),
            "mean_delta_a": _finite_mean(delta_a),
            "mean_delta_b": _finite_mean(delta_b),
            "mean_skew": mean_skew,
            "corr(q,skew)": corr_q_skew,
            },
            "execution": {
            "avg_sell_capture": avg_sell_capture,
            "avg_buy_capture": avg_buy_capture,
            "fills_sells": fills_sells,
            "fills_buys": fills_buys,
            "fills_total": fills_total,
            "sell_events": sell_events,
            "buy_events": buy_events,
            },
            "inventory": {
            "q_min": q_min,
            "q_max": q_max,
            },

            "grid_hits": {
                "lam_plus_hit": lp_hit,
                "lam_minus_hit": lm_hit,
            },
            "policy": {
                "t0": policy_t0,
                "probe": policy_probe,
            },
        }

        if verbose:
            lines = []
            lines.append("=== Backtest diagnostics report ===")

            p = out["params"]
            if any(v is not None for v in p.values()):
                lines.append(
                    f"params: gamma={_fmt(p['gamma'])}  eta={_fmt(p['eta'])}  "
                    f"eta_T={_fmt(p['eta_T'])}  eta_run={_fmt(p['eta_run'])}"
                )

            pn = out["pnl"]
            lines.append(
                f"PnL: trade={_fmt(pn['trade_cum'])}  mtm={_fmt(pn['mtm_cum'])}  "
                f"total={_fmt(pn['total_cum'])}  (end_rel={_fmt(pn['pnl_rel_end'])})"
            )
            lines.append(
                f"Wealth identity: ptp={_fmt(pn['wealth_ptp'])}  max|wealth-pnl|={_fmt(pn['identity_err_max'])}"
            )

            mc = out["micro"]
            lines.append(
                f"Micro: mean_spread={_fmt(mc['mean_spread'])}  "
                f"mean_delta_b={_fmt(mc['mean_delta_b'])}  mean_delta_a={_fmt(mc['mean_delta_a'])}  "
                f"corr(q,skew)={_fmt(mc['corr(q,skew)'])}"
            )

            fl = out["fills"]
            if (A_fill is not None) and (k_fill is not None):
                s = f"FillModel: mean_prob ask={_fmt(fl['mean_prob_a'])}  bid={_fmt(fl['mean_prob_b'])}"
                if fl["exp_fills_a"] is not None and fl["exp_fills_b"] is not None:
                    s += (
                        f"  |  exp_fills ask={_fmt(fl['exp_fills_a'])}  bid={_fmt(fl['exp_fills_b'])}  "
                        f"total={_fmt(fl['exp_fills_a'] + fl['exp_fills_b'])}"
                    )
                lines.append(s)

            ex = out["execution"]
            if ex["avg_sell_capture"] is not None:
                lines.append(
                    f"ExecQuality: avg_sell_cap={_fmt(ex['avg_sell_capture'])}  "
                    f"avg_buy_cap={_fmt(ex['avg_buy_capture'])}  "
                    f"fills(sell)={ex['fills_sells']}  fills(buy)={ex['fills_buys']}  "
                    f"events(sell)={ex['sell_events']}  events(buy)={ex['buy_events']}"
                )

            gh = out["grid_hits"]
            if gh["lam_plus_hit"] is not None or gh["lam_minus_hit"] is not None:
                lines.append(
                    f"Grid hits: lam_plus_hit={_fmt(gh['lam_plus_hit'])}  lam_minus_hit={_fmt(gh['lam_minus_hit'])}"
                )

            if policy is not None and (policy_t0 or policy_probe):
                lines.append("Policy sanity:")
                for k, d in policy_t0.items():
                    lines.append(
                        f"  t=0  {k}: skew={_fmt(d['skew'])}  del_b={_fmt(d['delta_b'])}  del_a={_fmt(d['delta_a'])}"
                    )
                for k, d in policy_probe.items():
                    lines.append(
                        f"  i={d['i']}  {k}: skew={_fmt(d['skew'])}  del_b={_fmt(d['delta_b'])}  del_a={_fmt(d['delta_a'])}"
                    )

            print("\n".join(lines))

        return out



def plot_hjb_policy_heatmap(sol, t_idx=0):
    """
    Visualizes the optimal 'Skew' (Bid_dist - Ask_dist).
    FIXED: Uses raw strings (r"...") to prevent LaTeX errors.
    """
    # 1. Extract Grids
    q_grid = sol.q_grid
    lam_plus_grid = sol.lam_plus_grid
    lam_minus_grid = sol.lam_minus_grid
    
    # Slice: Fix lambda_minus to its MEAN value
    i_minus_fixed = len(lam_minus_grid) // 2
    lam_minus_val = lam_minus_grid[i_minus_fixed]
    
    # 2. Calculate Skew
    skew_map = np.zeros((len(q_grid), len(lam_plus_grid)))
    for i_q, q in enumerate(q_grid):
        for i_lp, lam_p in enumerate(lam_plus_grid):
            d_a = sol.delta_a[t_idx, i_q, i_lp, i_minus_fixed]
            d_b = sol.delta_b[t_idx, i_q, i_lp, i_minus_fixed]
            skew_map[i_q, i_lp] = d_a - d_b 

    # 3. Plot
    plt.figure(figsize=(10, 6))
    extent = [
        lam_plus_grid.min(), lam_plus_grid.max(),
        q_grid.min(), q_grid.max()
    ]
    
    im = plt.imshow(
        skew_map, 
        aspect='auto', 
        origin='lower', 
        cmap='RdBu', 
        extent=extent
    )
    
    cbar = plt.colorbar(im)
    # FIX 1: Use r"" for delta symbols
    cbar.set_label(r"Skew ($\delta^a - \delta^b$)")
    
    # FIX 2: Use rf"" (raw f-string) so \approx works
    plt.xlabel(rf"Buy Intensity $\lambda^+$ (assuming $\lambda^- \approx {lam_minus_val:.1f}$)")
    
    plt.ylabel(r"Inventory $q$")
    plt.title(f"HJB Optimal Policy Heatmap (t_idx={t_idx})")
    
    plt.text(lam_plus_grid.max()*0.8, q_grid.max()*0.8, "SELL", 
            color='white', ha='center', fontweight='bold')
    plt.text(lam_plus_grid.min()*0.2, q_grid.min()*0.8, "BUY", 
            color='white', ha='center', fontweight='bold')
    
    plt.grid(False)
    plt.show()