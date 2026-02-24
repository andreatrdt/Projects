# PointProcessPlotSuite.py

import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Time-axis helpers
# -------------------------
try:
    # Optional project-level constant (seconds from midnight) for the market open.
    from config import MARKET_OPEN_SECONDS  # type: ignore
except Exception:
    # Default: US equities regular session open = 09:30.
    MARKET_OPEN_SECONDS = 9.5 * 3600

from matplotlib.ticker import FuncFormatter, MultipleLocator

def _sec_to_clock_str(sec: float, show_seconds: bool = False) -> str:
    """Format seconds from midnight into HH:MM (or HH:MM:SS)."""
    sec = float(sec) % 86400.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if show_seconds else f"{h:02d}:{m:02d}"

def _choose_tick_step(span_seconds: float) -> int:
    """Pick a human-friendly major tick step (in seconds) based on axis span."""
    span_seconds = abs(float(span_seconds))
    if span_seconds <= 15 * 60:
        return 60          # 1 min
    if span_seconds <= 60 * 60:
        return 5 * 60      # 5 min
    if span_seconds <= 3 * 60 * 60:
        return 15 * 60     # 15 min
    if span_seconds <= 8 * 60 * 60:
        return 30 * 60     # 30 min
    return 60 * 60         # 1 hour

def apply_time_of_day_axis(
    ax,
    *,
    base_seconds: float = 0.0,
    origin_seconds: float = MARKET_OPEN_SECONDS,
    show_seconds: bool = False,
    rotate: int = 0,
):
    """Format an x-axis that is in *seconds since market open* into wall-clock time.

    If your plotted x-values are seconds since the *episode* start, pass base_seconds=t_start.
    If your plotted x-values are already seconds since open, leave base_seconds=0.

    The displayed label is computed as origin_seconds + base_seconds + x.
    """
    try:
        xmin, xmax = ax.get_xlim()
        step = _choose_tick_step(xmax - xmin)
        ax.xaxis.set_major_locator(MultipleLocator(step))
    except Exception:
        # If locator fails for any reason, at least keep the formatter.
        pass

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: _sec_to_clock_str(origin_seconds + base_seconds + x, show_seconds=show_seconds))
    )
    if rotate:
        for lab in ax.get_xticklabels():
            lab.set_rotation(rotate)
            lab.set_horizontalalignment('right')

from config import COMPONENTS_2D
from HawkesCalibrator import HawkesCalibrator


class PointProcessPlotSuite:
    """
    Figures for thesis:
      - 1D data-vs-Poisson benchmark (counting + event times + intensity)
      - 2D counting processes + intensities
      - branching matrix heatmap
      - kernel response + cumulative kernel
      - rate by bins
      - time-rescaling GOF (exact, exp-kernel) + KS vs Exp(1)
    """

    def __init__(self, events: dict, fit_res, components=None):
        """
        events: dict[name -> np.ndarray of event times (seconds from open)]
        fit_res: window fit object with attributes mu, A, beta (and maybe rho)
        components: list of component names (default COMPONENTS_2D)
        """
        self.events = events
        self.fit_res = fit_res
        self.components = list(components) if components is not None else list(COMPONENTS_2D)

    # -----------------------
    # helpers
    # -----------------------
    def _slice_events(self, t_start: float, T: float, warmup: float = 0.0):
        """
        Return dict of events in [t_start-warmup, t_start+T), shifted so that t_start -> 0.
        If warmup>0, you will get negative times for pre-history.
        """
        out = {}
        t0 = t_start - float(warmup)
        t1 = t_start + float(T)
        for name in self.components:
            t = np.asarray(self.events.get(name, np.array([])), dtype=float)
            m = (t >= t0) & (t < t1)
            out[name] = t[m] - t_start
        return out

    def _merge_times_comps(self, events_rel: dict):
        """Build merged (times, comps) arrays (sorted)."""
        times_win, comps_win = [], []
        for j, name in enumerate(self.components):
            t_rel = np.asarray(events_rel.get(name, np.array([])), dtype=float)
            if t_rel.size:
                times_win.append(t_rel)
                comps_win.append(np.full_like(t_rel, j, dtype=int))
        if len(times_win) == 0:
            return np.array([], dtype=float), np.array([], dtype=int)

        times_win = np.concatenate(times_win)
        comps_win = np.concatenate(comps_win)
        order = np.argsort(times_win)
        return times_win[order], comps_win[order]

    def _poisson_sim(self, rate: float, T: float, seed: int | None = None):
        """Simulate Poisson event times in [0,T)."""
        rng = np.random.default_rng(seed)
        if rate <= 0 or T <= 0:
            return np.array([], dtype=float)
        t = 0.0
        out = []
        while True:
            t += rng.exponential(1.0 / rate)
            if t >= T:
                break
            out.append(t)
        return np.array(out, dtype=float)

    def _count_on_grid(self, t_grid: np.ndarray, t_events: np.ndarray):
        """N(t) evaluated on grid via searchsorted."""
        t_events = np.asarray(t_events, dtype=float)
        if t_events.size == 0:
            return np.zeros_like(t_grid, dtype=int)
        return np.searchsorted(t_events, t_grid, side="right")

    def _compute_intensity_grid(self, times_win, comps_win, T, n_grid):
        """
        Event-aligned intensity grid (Option B):
        - coarse backbone grid on [0,T]
        - all event times in [0,T]
        - tiny 'just after event' points to show the post-event jump
        This prevents "flat intensity" plots when beta is very large (ms decay).
        """
        import numpy as np

        empty_events = {name: np.array([]) for name in self.components}
        cal_dummy = HawkesCalibrator(empty_events, T=T)

        times_win = np.asarray(times_win, dtype=float)
        comps_win = np.asarray(comps_win, dtype=int)

        # events inside plotting window (ignore warmup times here)
        times0 = times_win[(times_win >= 0.0) & (times_win <= float(T)) & np.isfinite(times_win)]

        # choose a small epsilon based on beta so we see the jump but don’t create duplicates
        beta_max = float(np.max(np.asarray(self.fit_res.beta, float))) if hasattr(self.fit_res, "beta") else 1.0
        beta_max = max(beta_max, 1e-12)

        # epsilon: ~5% of the fastest decay time, capped for numerical sanity
        eps = min(1e-4, 0.05 / beta_max)
        eps = max(eps, 1e-9)

        # coarse backbone grid (n_grid acts as backbone resolution, not "final points")
        n_backbone = int(max(500, min(int(n_grid), 5000)))
        t_backbone = np.linspace(0.0, float(T), n_backbone)

        # add event times + immediate post-event points (and a second post point helps visibility)
        if times0.size > 0:
            t_evt = times0
            t_evt_p1 = np.clip(times0 + eps, 0.0, float(T))
            t_evt_p2 = np.clip(times0 + 5.0 * eps, 0.0, float(T))
            t_grid = np.unique(np.concatenate([t_backbone, t_evt, t_evt_p1, t_evt_p2]))
        else:
            t_grid = np.unique(t_backbone)

        t_grid, lam_grid = cal_dummy.intensity_from_path(
            mu=self.fit_res.mu,
            A=self.fit_res.A,
            beta=self.fit_res.beta,
            times=times_win,     # includes warmup/pre-history if provided
            comps=comps_win,
            T=float(T),
            n_grid=int(n_grid),  # fallback only
            t_grid=t_grid,       # <-- key change
        )

        return t_grid, lam_grid


    def _save_fig(self, fig, out_dir, name, fmts=("png", "pdf"), dpi=200):
        if out_dir is None:
            return
        os.makedirs(out_dir, exist_ok=True)
        for ext in fmts:
            path = os.path.join(out_dir, f"{name}.{ext}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")

    # -----------------------
    # exact compensator + exact time-rescaling residuals (exp kernel)
    # -----------------------
    @staticmethod
    def compensator_exact(times, comps, mu, A, beta, T, d=None):
        """
        Exact Λ(T)=∫_0^T λ(t) dt for exp-kernel Hawkes with:
            λ(t)=mu + A R(t),
            R_j(t)=∑_{k:comp=j, t_k<t} exp(-beta_j (t-t_k))
        Column-wise decay: beta_j applies to source j (column j).

        times may contain negative values (pre-history). Integral is still over [0,T].
        Pre-history affects R(0) automatically.
        """
        times = np.asarray(times, float)
        comps = np.asarray(comps, int)
        mu = np.asarray(mu, float)
        A = np.asarray(A, float)
        beta = np.asarray(beta, float)

        if d is None:
            d = len(mu)

        # We compute I_j = ∫_0^T R_j(t) dt exactly, allowing events before 0.
        I = np.zeros(d, float)

        # For each source j, build its event times (can be <0)
        for j in range(d):
            tj = times[comps == j]
            if tj.size == 0:
                continue
            tj = np.sort(tj)

            # r(t) is the running value of R_j(t) just after processing events <= current time
            r = 0.0
            last = 0.0
            bj = float(beta[j])

            # First: process pre-history events (<0) to get r at 0-
            for t in tj[tj < 0.0]:
                # decay from last to t (both negative, last starts at 0 -> handle carefully)
                dt = t - last
                # last starts at 0, so dt negative here: instead, we should build r at 0
                # easiest: update r forward in time from that event to 0 later; do it by direct accumulation:
                # Add this event's contribution to r at 0: exp(-bj*(0 - t))
                r += np.exp(-bj * (0.0 - t))
            last = 0.0  # now we start integrating from 0

            # Now handle events in [0,T)
            for t in tj[(tj >= 0.0) & (tj < T)]:
                dt = float(t - last)
                if dt > 0:
                    I[j] += r * (1.0 - np.exp(-bj * dt)) / bj
                    r *= np.exp(-bj * dt)
                r += 1.0
                last = float(t)

            # Tail [last, T]
            dt = float(T - last)
            if dt > 0:
                I[j] += r * (1.0 - np.exp(-bj * dt)) / bj

        Lambda = mu * float(T) + A @ I
        return Lambda, I

    @staticmethod
    def time_rescaling_residuals_exact(times, comps, mu, A, beta, T, components=None):
        """
        Exact time-rescaling residuals for each component i:
            r_k^i = Λ_i(t_k^i) - Λ_i(t_{k-1}^i)
        Under correct model, residuals are i.i.d. Exp(1).

        Works with negative pre-history times (affecting initial state).
        """
        times = np.asarray(times, float)
        comps = np.asarray(comps, int)
        mu = np.asarray(mu, float)
        A = np.asarray(A, float)
        beta = np.asarray(beta, float)

        d = len(mu)
        order = np.argsort(times)
        times = times[order]
        comps = comps[order]

        # Running excitation state R(t) for sources
        R = np.zeros(d, float)

        # Initialize R at time 0 using pre-history (times<0)
        for t, j in zip(times[times < 0.0], comps[times < 0.0]):
            R[j] += np.exp(-beta[j] * (0.0 - t))

        # Residual accumulators
        residuals = {i: [] for i in range(d)}
        last_Li_at_event = np.zeros(d, float)   # last Λ_i at last event of type i
        Li = np.zeros(d, float)                 # running Λ_i(t) (integral from 0)

        t_prev = 0.0

        # Iterate events in [0,T)
        mask = (times >= 0.0) & (times < float(T))
        for t, j in zip(times[mask], comps[mask]):
            t = float(t)
            dt = t - t_prev
            if dt < 0:
                continue

            # exact integral increment on (t_prev, t):
            # ∫ mu ds = mu*dt
            inc = mu * dt

            # plus ∑_k A[:,k] ∫ R_k(s) ds, with R_k decaying exponentially from its value at t_prev
            # ∫_{0}^{dt} R_k(t_prev) e^{-beta_k u} du = R_k(t_prev) * (1 - e^{-beta_k dt})/beta_k
            decay_factor = (1.0 - np.exp(-beta * dt)) / np.maximum(beta, 1e-12)
            inc += A @ (R * decay_factor)

            Li += inc

            # At event time, record residual for target component (=event’s own comp)
            # residual is increment of Λ_{j}(t) since previous event of type j
            r = Li[j] - last_Li_at_event[j]
            residuals[j].append(float(r))
            last_Li_at_event[j] = Li[j]

            # update R to just after the event
            R *= np.exp(-beta * dt)
            R[j] += 1.0

            t_prev = t

        # Convert to arrays, map to names if provided
        out = {}
        if components is None:
            for i in range(d):
                out[i] = np.asarray(residuals[i], float)
        else:
            for i, name in enumerate(components):
                out[name] = np.asarray(residuals[i], float)
        return out

    @staticmethod
    def ks_exp1(residuals):
        """
        KS test vs Exp(1). Returns (D, pvalue).
        """
        r = np.asarray(residuals, float)
        r = r[np.isfinite(r)]
        r = r[r >= 0]
        if r.size < 5:
            return None

        try:
            from scipy.stats import kstest
            # Exp(1) CDF: 1 - exp(-x)
            D, p = kstest(r, lambda x: 1.0 - np.exp(-x))
            return float(D), float(p)
        except Exception:
            return None

    # -----------------------
    # main plots
    # -----------------------
    def plot_1d_vs_poisson(
        self,
        component: str = "E_a",
        t_start: float = 0.0,
        T: float = 300.0,
        warmup: float = 0.0,
        n_grid: int = 20000,
        poisson_rate: float | None = None,
        seed: int = 42,
        source: str = "data",  # "data" or "sim"
        out_dir: str | None = None,
        save_name: str | None = None,
        intensity_clip_q: float | None = None,  # e.g. 0.995 to clip spikes for readability
        add_mu_line: bool = True,
        verbose: bool = True,
    ):
        """
        2-panel figure:
          (top) counting + event times (Hawkes component vs Poisson benchmark)
          (bottom) intensity λ_i(t) vs constant Poisson rate (and optional μ_i)

        warmup>0 includes pre-history to compute intensity/Λ more correctly at t=0.
        """
        if component not in self.components:
            raise ValueError(f"component='{component}' not in components={self.components}")
        i = self.components.index(component)
        T = float(T)

        if source == "data":
            events_rel = self._slice_events(t_start, T, warmup=warmup)
            times_win, comps_win = self._merge_times_comps(events_rel)

            # component events in [0,T) only (exclude warmup <0)
            t_comp_all = np.asarray(events_rel[component], dtype=float)
            t_comp = t_comp_all[(t_comp_all >= 0.0) & (t_comp_all < T)]

            t_grid, lam_grid = self._compute_intensity_grid(times_win, comps_win, T, n_grid)
            lam = lam_grid[i, :]

            # Poisson benchmark: match empirical rate unless overridden
            rate = (t_comp.size / T) if poisson_rate is None else float(poisson_rate)
            t_pois = self._poisson_sim(rate, T, seed=seed)

            title_prefix = f"DATA ({component}) vs Poisson benchmark"
            events_label = "Data events"

        elif source == "sim":
            # simulate Hawkes path using your calibrator method (if present)
            empty_events = {name: np.array([]) for name in self.components}
            sim_engine = HawkesCalibrator(empty_events, T=T)

            streams, (t_sim, c_sim), (t_grid, lam_grid) = sim_engine.simulate_and_intensity(
                self.fit_res.mu, self.fit_res.A, self.fit_res.beta,
                T=T, n_grid=n_grid, seed=seed
            )

            times_win = np.asarray(t_sim, float)
            comps_win = np.asarray(c_sim, int)

            lam = lam_grid[i, :]
            t_comp = np.asarray(streams[self.components[i]], dtype=float)

            rate = (t_comp.size / T) if poisson_rate is None else float(poisson_rate)
            t_pois = self._poisson_sim(rate, T, seed=seed + 1)

            title_prefix = f"SIM ({component}) Hawkes vs Poisson benchmark"
            events_label = "Hawkes-sim events"

        else:
            raise ValueError("source must be 'data' or 'sim'")

        # Optional clipping for readability
        lam_plot = lam.copy()
        if intensity_clip_q is not None and 0 < intensity_clip_q < 1 and lam_plot.size:
            qv = float(np.quantile(lam_plot[np.isfinite(lam_plot)], intensity_clip_q))
            lam_plot = np.minimum(lam_plot, qv)

        # Counting on grid
        N_h = self._count_on_grid(t_grid, t_comp)
        N_p = self._count_on_grid(t_grid, t_pois)

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        # --- Top: counting + event times
        ax0 = axes[0]
        ax0.plot(t_grid, N_h, label=f"{events_label} (counting)")
        ax0.plot(t_grid, N_p, linestyle="--", label="Poisson (counting)")

        if t_comp.size:
            ax0.vlines(t_comp, ymin=0, ymax=max(1, N_h.max()), alpha=0.25, label=f"{events_label} times")
        if t_pois.size:
            ax0.vlines(t_pois, ymin=0, ymax=max(1, N_p.max()), alpha=0.25, label="Poisson times")

        ax0.set_ylabel("N(t)")
        ax0.set_title(f"{title_prefix} — Counting processes + event times")
        ax0.legend(loc="upper left")
        ax0.grid(True, alpha=0.25)

        # --- Bottom: intensity
        ax1 = axes[1]
        ax1.plot(t_grid, lam_plot, label=rf"Hawkes $\lambda_{{{component}}}(t)$")
        ax1.axhline(rate, linestyle="--", label=f"Poisson λ = {rate:.4g}")
        if add_mu_line:
            ax1.axhline(float(self.fit_res.mu[i]), linestyle=":", label=rf"$\mu_{{{component}}}$ = {self.fit_res.mu[i]:.4g}")
        ax1.set_ylabel(r"$\lambda(t)$")
        ax1.set_xlabel("Time")
        apply_time_of_day_axis(ax1, base_seconds=t_start, rotate=30)
        ax1.set_title("Intensity: Hawkes vs Poisson")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.25)

        plt.tight_layout()

        if save_name is None:
            save_name = f"pp_1d_{component}_{source}_t{int(t_start)}_T{int(T)}"
        self._save_fig(fig, out_dir, save_name)

        # Exact compensator check (with warmup if provided)
        Lambda_vec, _ = self.compensator_exact(
            times=times_win,
            comps=comps_win,
            mu=self.fit_res.mu,
            A=self.fit_res.A,
            beta=self.fit_res.beta,
            T=T
        )
        N_i = int(t_comp.size)
        Lambda_i = float(Lambda_vec[i])

        if verbose:
            z = (N_i - Lambda_i) / np.sqrt(max(Lambda_i, 1e-12))
            print(f"[{component}] N/T = {N_i/T:.6f} | Λ/T = {Lambda_i/T:.6f} | z = {z:.3f}")

        return fig, axes

    def plot_2d_counting_and_intensity(
        self,
        t_start: float = 0.0,
        T: float = 300.0,
        warmup: float = 0.0,
        n_grid: int = 20000,
        out_dir: str | None = None,
        save_name: str | None = None,
        intensity_clip_q: float | None = None,
    ):
        """
        2-panel figure:
          (top) N1(t), N2(t) with event times
          (bottom) λ1(t), λ2(t) plus baselines μ1, μ2
        warmup>0 includes pre-history for intensity.
        """
        T = float(T)
        events_rel = self._slice_events(t_start, T, warmup=warmup)
        times_win, comps_win = self._merge_times_comps(events_rel)
        t_grid, lam_grid = self._compute_intensity_grid(times_win, comps_win, T, n_grid)

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        # --- Top: counting + event times
        ax0 = axes[0]
        for i, name in enumerate(self.components):
            t_i_all = np.asarray(events_rel[name], float)
            t_i = t_i_all[(t_i_all >= 0.0) & (t_i_all < T)]
            N_i = self._count_on_grid(t_grid, t_i)
            ax0.plot(t_grid, N_i, label=f"{name}: N(t)")
            if t_i.size:
                ax0.vlines(t_i, ymin=0, ymax=max(1, N_i.max()), alpha=0.20, label=f"{name} events")

        ax0.set_ylabel("N_i(t)")
        ax0.set_title("Hawkes 2D — Counting processes + event times")
        ax0.legend(loc="upper left", ncol=2)
        ax0.grid(True, alpha=0.25)

        # --- Bottom: intensities + baselines
        ax1 = axes[1]
        for i, name in enumerate(self.components):
            lam = lam_grid[i, :].copy()
            if intensity_clip_q is not None and 0 < intensity_clip_q < 1 and lam.size:
                qv = float(np.quantile(lam[np.isfinite(lam)], intensity_clip_q))
                lam = np.minimum(lam, qv)

            ax1.plot(t_grid, lam, label=fr"$\lambda_{{{name}}}(t)$")
            ax1.axhline(float(self.fit_res.mu[i]), linestyle="--", label=fr"$\mu_{{{name}}}$")

        ax1.set_ylabel(r"$\lambda_i(t)$")
        ax1.set_xlabel("Time")
        apply_time_of_day_axis(ax1, base_seconds=t_start, rotate=30)
        ax1.set_title("Intensities of the two processes")
        ax1.legend(loc="upper right", ncol=2)
        ax1.grid(True, alpha=0.25)

        plt.tight_layout()

        if save_name is None:
            save_name = f"pp_2d_data_t{int(t_start)}_T{int(T)}"
        self._save_fig(fig, out_dir, save_name)

        return fig, axes

    # -----------------------
    # additional thesis plots
    # -----------------------
    def plot_branching_heatmap(self, out_dir: str | None = None, save_name: str = "pp_branching_heatmap"):
        """Heatmap of branching matrix K_ij = A_ij / beta_j (column-wise)."""
        A = np.asarray(self.fit_res.A, dtype=float)
        beta = np.asarray(self.fit_res.beta, dtype=float).reshape(-1)
        K = A / beta[None, :]

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
        im = ax.imshow(K, aspect="auto")
        ax.set_xticks(range(len(self.components)))
        ax.set_yticks(range(len(self.components)))
        ax.set_xticklabels(self.components)
        ax.set_yticklabels(self.components)
        ax.set_title(r"Branching matrix $K_{ij}=A_{ij}/\beta_j$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                ax.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center", fontsize=9)

        plt.tight_layout()
        self._save_fig(fig, out_dir, save_name)
        return fig, ax

    def plot_kernel_cumulative(self, t_max=None, n=800, time_unit="ms", out_dir=None, save_name="pp_kernel_cum"):
        A = np.asarray(self.fit_res.A, dtype=float)
        beta = np.asarray(self.fit_res.beta, dtype=float).reshape(-1)
        tau = 1.0 / np.maximum(beta, 1e-12)
        if t_max is None:
            t_max = 10.0 * float(np.max(tau))

        unit_scale = {"s": 1.0, "ms": 1e3, "us": 1e6}
        sx = unit_scale[time_unit]
        t = np.linspace(0.0, t_max, n)
        tx = t * sx

        fig, axes = plt.subplots(len(self.components), len(self.components), figsize=(12, 9), sharex=True, sharey=True)

        for i in range(len(self.components)):
            for j in range(len(self.components)):
                ax = axes[i, j]
                Kij = A[i, j] / max(beta[j], 1e-12)
                y = Kij * (1.0 - np.exp(-beta[j] * t))
                ax.plot(tx, y)
                ax.axhline(Kij, ls="--", alpha=0.5)
                ax.grid(True, alpha=0.15)
                if i == len(self.components) - 1:
                    ax.set_xlabel(f"src {self.components[j]} (t in {time_unit})")
                if j == 0:
                    ax.set_ylabel(f"tgt {self.components[i]}")
                ax.set_title(r"$\int_0^t \phi_{ij}$", fontsize=10)

        fig.suptitle("Cumulative kernel response (converges to K=A/β)", y=1.02)
        plt.tight_layout()
        self._save_fig(fig, out_dir, save_name)
        return fig, axes

    def plot_kernel_response(
        self,
        t_max: float | None = None,
        n: int = 800,
        time_unit: str = "ms",
        logy: bool = False,
        normalize: bool = False,
        eps: float = 1e-9,
        out_dir: str | None = None,
        save_name: str = "pp_kernel_response",
    ):
        """
        Kernel: phi_ij(t) = A_ij * exp(-beta_j t)   (beta per source j).
        """
        A = np.asarray(self.fit_res.A, dtype=float)
        beta = np.asarray(self.fit_res.beta, dtype=float).reshape(-1)

        tau = 1.0 / np.maximum(beta, 1e-12)
        if t_max is None:
            t_max = 10.0 * float(np.max(tau))

        unit_scale = {"s": 1.0, "ms": 1e3, "us": 1e6}
        if time_unit not in unit_scale:
            raise ValueError("time_unit must be 's', 'ms' or 'us'")
        sx = unit_scale[time_unit]

        t = np.linspace(max(eps, 0.0), t_max, n)
        tx = t * sx

        fig, axes = plt.subplots(
            len(self.components), len(self.components),
            figsize=(12, 9), sharex=True, sharey=False
        )

        for i in range(len(self.components)):
            for j in range(len(self.components)):
                ax = axes[i, j]
                if normalize:
                    y = np.exp(-beta[j] * t)
                    ttl = r"$e^{-\beta_j t}$"
                else:
                    y = A[i, j] * np.exp(-beta[j] * t)
                    ttl = r"$A_{ij}e^{-\beta_j t}$"

                ax.plot(tx, y)
                ax.grid(True, alpha=0.15)

                if i == len(self.components) - 1:
                    ax.set_xlabel(f"src {self.components[j]} (t in {time_unit})")
                if j == 0:
                    ax.set_ylabel(f"tgt {self.components[i]}")
                ax.set_title(ttl, fontsize=10)

                if logy and not normalize:
                    ax.set_yscale("log")

        fig.suptitle("Kernel responses (auto-scaled time axis)", y=1.02)
        plt.tight_layout()
        self._save_fig(fig, out_dir, save_name)
        return fig, axes

    def plot_rate_by_bins(
        self,
        t_start: float = 0.0,
        T: float = 600.0,
        bin_size: float = 1.0,
        out_dir: str | None = None,
        save_name: str | None = None,
    ):
        """Empirical rate per bin."""
        T = float(T)
        events_rel = self._slice_events(t_start, T, warmup=0.0)
        bins = np.arange(0.0, T + bin_size, bin_size)

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        for name in self.components:
            t = np.asarray(events_rel[name], float)
            t = t[(t >= 0.0) & (t < T)]
            counts, _ = np.histogram(t, bins=bins)
            rate = counts / bin_size
            centers = 0.5 * (bins[:-1] + bins[1:])
            ax.plot(centers, rate, label=f"{name} rate")

        ax.set_title(f"Empirical event rate by bins (bin={bin_size}s)")
        ax.set_xlabel("Time")
        apply_time_of_day_axis(ax, base_seconds=t_start, rotate=30)
        ax.set_ylabel("events / s")
        ax.grid(True, alpha=0.25)
        ax.legend()

        plt.tight_layout()
        if save_name is None:
            save_name = f"pp_rate_bins_t{int(t_start)}_T{int(T)}_bin{bin_size}"
        self._save_fig(fig, out_dir, save_name)
        return fig, ax

    def plot_time_rescaling_gof(
        self,
        t_start: float = 0.0,
        T: float = 600.0,
        warmup: float = 0.0,
        out_dir: str | None = None,
        save_name: str | None = None,
    ):
        """
        Time-rescaling GOF: residuals should be Exp(1) if Hawkes is well specified.
        Uses exact residuals for exp-kernel (no grid approximation).
        """
        T = float(T)
        events_rel = self._slice_events(t_start, T, warmup=warmup)
        times_win, comps_win = self._merge_times_comps(events_rel)

        residuals_by_comp = self.time_rescaling_residuals_exact(
            times=times_win,
            comps=comps_win,
            mu=self.fit_res.mu,
            A=self.fit_res.A,
            beta=self.fit_res.beta,
            T=T,
            components=self.components,
        )

        ks_by_comp = {}
        for name in self.components:
            ks_by_comp[name] = self.ks_exp1(residuals_by_comp.get(name, np.array([])))

        fig, axes = plt.subplots(len(self.components), 2, figsize=(14, 4 * len(self.components)))
        if len(self.components) == 1:
            axes = np.array([axes])

        for i, name in enumerate(self.components):
            r = np.asarray(residuals_by_comp.get(name, np.array([])), dtype=float)

            ax_h = axes[i, 0]
            ax_c = axes[i, 1]

            ax_h.hist(r, bins=40, density=True, alpha=0.6, label="Residuals")
            x = np.linspace(0, max(5.0, np.quantile(r, 0.99) if r.size else 5.0), 300)
            ax_h.plot(x, np.exp(-x), label="Exp(1) pdf")
            ax_h.set_title(f"{name} — residuals histogram")
            ax_h.grid(True, alpha=0.2)
            ax_h.legend()

            if r.size:
                rs = np.sort(r)
                ecdf = np.arange(1, len(rs) + 1) / len(rs)
                ax_c.plot(rs, ecdf, label="Empirical CDF")
                ax_c.plot(x, 1 - np.exp(-x), label="Exp(1) CDF")
            ax_c.set_title(f"{name} — CDF check (time-rescaling)")
            ax_c.grid(True, alpha=0.2)
            ax_c.legend()

            ks = ks_by_comp.get(name, None)
            if ks is not None:
                ax_c.text(
                    0.05, 0.05, f"KS D={ks[0]:.3f}, p={ks[1]:.3g}",
                    transform=ax_c.transAxes, fontsize=10
                )

        fig.suptitle("Time-rescaling goodness-of-fit (residuals ~ Exp(1))", y=1.02)
        plt.tight_layout()

        if save_name is None:
            save_name = f"pp_time_rescaling_t{int(t_start)}_T{int(T)}"
        self._save_fig(fig, out_dir, save_name)
        return fig, axes

    def make_suite(
        self,
        t_start: float,
        T: float,
        out_dir: str | None = None,
        seed: int = 42,
        warmup: float = 0.0,
    ):
        """One-call generator for the whole set of thesis plots."""
        figs = {}

        figs["1d_Ea"] = self.plot_1d_vs_poisson(
            component="E_a", t_start=t_start, T=T, warmup=warmup,
            seed=seed, source="data",
            out_dir=out_dir, save_name=f"pp_1d_Ea_data_t{int(t_start)}_T{int(T)}",
            intensity_clip_q=0.995,  # keep thesis-readable; set None to see full spikes
            verbose=True,
        )[0]

        figs["1d_Eb"] = self.plot_1d_vs_poisson(
            component="E_b", t_start=t_start, T=T, warmup=warmup,
            seed=seed, source="data",
            out_dir=out_dir, save_name=f"pp_1d_Eb_data_t{int(t_start)}_T{int(T)}",
            intensity_clip_q=0.995,
            verbose=True,
        )[0]

        figs["2d"] = self.plot_2d_counting_and_intensity(
            t_start=t_start, T=T, warmup=warmup,
            out_dir=out_dir, save_name=f"pp_2d_t{int(t_start)}_T{int(T)}",
            intensity_clip_q=0.995,
        )[0]

        figs["branching"] = self.plot_branching_heatmap(out_dir=out_dir)[0]
        figs["kernel"] = self.plot_kernel_response(out_dir=out_dir, time_unit="ms", normalize=False, logy=False)[0]
        figs["kernel_cum"] = self.plot_kernel_cumulative(out_dir=out_dir, time_unit="ms")[0]
        figs["rate_bins"] = self.plot_rate_by_bins(t_start=t_start, T=T, bin_size=1.0, out_dir=out_dir)[0]
        figs["rescaling"] = self.plot_time_rescaling_gof(t_start=t_start, T=T, warmup=warmup, out_dir=out_dir)[0]




        return figs
