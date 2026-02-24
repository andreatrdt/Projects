import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from numpy.linalg import eigvals
from scipy.optimize import minimize
from scipy.stats import kstest, expon
import matplotlib.pyplot as plt
from config import COMPONENTS_2D




@dataclass
class HawkesFitResult:
    mu: np.ndarray        # (self.N,)
    A: np.ndarray         # (self.N,self.N)
    beta: np.ndarray      # (self.N,)
    success: bool
    message: str
    K: np.ndarray         # (self.N,self.N) branching matrix
    rho: float            # spectral radius(K)
    n_params: int


class HawkesCalibrator:
    """
    Hawkes ND con kernel esponenziale, nessuna struttura tied:
    λ(t) = μ + A * history  (con colonna j associata a β_j)
    """

    def __init__(self, events: Dict[str, np.ndarray], T: float, dimension = 2):
        # Assicuriamoci che tutti i self.N componenti esistano
        self.N = dimension

        # to be changed !!!!!
        #######################################
        self.COMPONENTS = COMPONENTS_2D
        ########################################

        self.events = {}
        for name in self.COMPONENTS:
            v = np.asarray(events.get(name, np.array([], dtype=float)), dtype=float)
            self.events[name] = np.sort(v)
        self.T = float(T)

        # Merged timeline
        self._merged_times, self._merged_comp = self._merge_events()

    # ---------- utilities ----------

    def _merge_events(self) -> Tuple[np.ndarray, np.ndarray]:
        times = []
        comps = []
        for j, name in enumerate(self.COMPONENTS):
            tj = self.events[name]
            times.append(tj)
            comps.append(np.full_like(tj, j, dtype=int))
        if times:
            t = np.concatenate(times)
            c = np.concatenate(comps)
            order = np.argsort(t)
            return t[order], c[order]
        return np.array([], dtype=float), np.array([], dtype=int)

    @staticmethod
    def _softplus(x):
        x = np.asarray(x, dtype=float)
        return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

    @staticmethod
    def _inv_softplus(y):
        y = np.asarray(y, dtype=float)
        y = np.maximum(y, 1e-12)
        return y + np.log1p(-np.exp(-y))

    def _build_int_kernel(self, A: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        Branching matrix K_ij = A_ij / β_j (β per sorgente j).
        """
        A = np.asarray(A, dtype=float)
        beta = np.asarray(beta, dtype=float)
        betac = np.maximum(beta, 1e-12)  # (self.N,)
        return A / betac[np.newaxis, :]

    def _rho_of_K(self, K: np.ndarray) -> float:
        if not np.all(np.isfinite(K)):
            return np.inf
        try:
            vals = eigvals(K)
        except Exception:
            return np.inf
        r = np.max(np.abs(vals))
        return float(r) if np.isfinite(r) else np.inf

    def _stabilize_A(self, A_raw: np.ndarray, beta: np.ndarray, rho_target: float):
        """
        Riscalo A_raw per imporre rho(K) <= rho_target.
        """
        K_raw = self._build_int_kernel(A_raw, beta)
        rho_raw = self._rho_of_K(K_raw)

        if (not np.isfinite(rho_raw)) or (rho_raw <= 0.0):
            return A_raw, rho_raw

        if rho_raw <= rho_target:
            return A_raw, rho_raw

        scale = rho_target / rho_raw
        A_stab = A_raw * scale
        return A_stab, rho_raw

    def _pack(self, mu, A, beta):
        return np.concatenate([mu, A.reshape(-1), beta])

    def _unpack(self, theta):
        theta = np.asarray(theta, dtype=float)
        off = 0
        mu = self._softplus(theta[off:off + self.N]); off += self.N
        A_flat = self._softplus(theta[off:off + self.N * self.N]); off += self.N * self.N
        beta = self._softplus(theta[off:off + self.N]); off += self.N
        A = A_flat.reshape(self.N, self.N)
        return mu, A, beta

    # ---------- log-likelihood ----------
    def log_likelihood(self, theta, rho_cap_fit: float = 0.999) -> float:
        mu, A_raw, beta = self._unpack(theta)
        if (not np.all(np.isfinite(mu)) or
            not np.all(np.isfinite(A_raw)) or
            not np.all(np.isfinite(beta))):
            return -1e50

        # --- Option C: DO NOT stabilize in fit ---
        K_raw = self._build_int_kernel(A_raw, beta)
        rho_raw = self._rho_of_K(K_raw)
        if (not np.isfinite(rho_raw)):
            return -1e50

        # Hard barrier to prevent optimizer from going supercritical
        # (keeps things sane + makes theory rates meaningful)
        if rho_raw >= rho_cap_fit:
            return -1e50

        A = A_raw  # <-- use raw A in the LL

        betac = np.maximum(beta, 1e-12)

        # ----- integral term -----
        S = np.zeros(self.N)
        for j in range(self.N):
            tj = self.events[self.COMPONENTS[j]]
            if tj.size:
                bj = betac[j]
                S[j] = (-np.expm1(-bj * (self.T - tj))).sum()

        integ = mu * self.T + (A @ (S / betac)).reshape(-1)
        if not np.all(np.isfinite(integ)):
            return -1e50

        # ----- event term (Ogata recursion) -----
        logterm = 0.0
        decays = np.zeros(self.N)
        last = 0.0

        times = self._merged_times
        comps = self._merged_comp
        if times.size == 0:
            return -integ.sum()

        for t_k, j_k in zip(times, comps):
            dt = t_k - last
            if dt < 0 or not np.isfinite(dt):
                return -1e50
            if dt > 0:
                decays *= np.exp(-betac * dt)

            lam = mu + A @ decays
            if (not np.all(np.isfinite(lam))) or np.any(lam <= 0):
                return -1e50

            logterm += np.log(lam[j_k])
            decays[j_k] += 1.0
            last = t_k

        LL = logterm - integ.sum()
        return LL if np.isfinite(LL) else -1e50


    def neg_loglik(self, theta, rho_cap_fit: float = 0.999) -> float:
        return -self.log_likelihood(theta, rho_cap_fit=rho_cap_fit)

    # ---------- fit ----------

    def fit(self, x0=None, rho_cap_fit: float = 0.999) -> HawkesFitResult:


        if x0 is None:
            # init: mu ≈ emp_rate, A small, β “rapidi”
            counts = np.array(
                [self.events[name].size for name in self.COMPONENTS],
                dtype=float,
            )
            emp_rate = counts / max(self.T, 1e-9)
            mu0 = np.maximum(emp_rate, 1e-6)
            A0 = np.full((self.N, self.N), 0.05, dtype=float)
            beta0 = np.full(self.N, 1000.0, dtype=float)
            x0 = self._pack(self._inv_softplus(mu0),
                            self._inv_softplus(A0),
                            self._inv_softplus(beta0))

        best_f = np.inf
        stall = 0
        TOL = 1e-4
        PATIENCE = 500
        last_xk = x0.copy()

        def _cb(xk):
            nonlocal best_f, stall, last_xk
            last_xk = xk.copy()
            f = float(self.neg_loglik(xk, rho_cap_fit=rho_cap_fit))

            mu_, A_raw, beta_ = self._unpack(xk)
            rho = self._rho_of_K(self._build_int_kernel(A_raw, beta_))
            print(f"iter: LL={-f:.3f}  f={f:.3f}  rho(K)={rho:.4f}")



            if f < best_f - TOL:
                best_f = f
                stall = 0
            else:
                stall += 1
                if stall >= PATIENCE:
                    raise StopIteration("early-stop")

        try:
            
            obj = lambda th: self.neg_loglik(th, rho_cap_fit=rho_cap_fit)
            res = minimize(obj, x0, method="L-BFGS-B", callback=_cb)

            x_opt = res.x
            success = bool(res.success)
            message = str(res.message)
        except StopIteration as e:
            x_opt = last_xk
            success = True
            message = str(e)

        mu, A_raw, beta = self._unpack(x_opt)

        K = self._build_int_kernel(A_raw, beta)
        rho = self._rho_of_K(K)

        return HawkesFitResult(
            mu=mu,
            A=A_raw,          # <-- RAW fit
            beta=beta,
            success=success,
            message=message,
            K=K,
            rho=rho,
            n_params = self.N + self.N*self.N + self.N
        )


    # ---------- residuals & diagnostics ----------

    def time_rescaled_residuals(
        self,
        mu: np.ndarray,
        A: np.ndarray,
        beta: np.ndarray,
        comp: str,
    ) -> np.ndarray:
        i = self.COMPONENTS.index(comp)
        times = self._merged_times
        comps = self._merged_comp
        betac = np.maximum(beta, 1e-12)

        if times.size == 0 or self.events[comp].size == 0:
            return np.array([])

        R = np.zeros(self.N, dtype=float)
        last = 0.0
        z = []
        acc = 0.0

        for t_k, j_k in zip(times, comps):
            dt = t_k - last
            if dt < 0:
                raise ValueError("Event times must be non-decreasing.")

            # ∫ μ_i ds
            acc += mu[i] * dt
            if dt > 0:
                decay_term = (1.0 - np.exp(-betac * dt)) / betac
                acc += (A[i, :] * R * decay_term).sum()

            if dt > 0:
                R *= np.exp(-betac * dt)

            R[j_k] += 1.0

            if j_k == i:
                z.append(acc)
                acc = 0.0

            last = t_k

        return np.asarray(z, dtype=float)

    def ks_exp_test(self, mu, A, beta):
        out = {}
        for comp in self.COMPONENTS:
            z = self.time_rescaled_residuals(mu, A, beta, comp)
            if z.size >= self.N:
                stat, p = kstest(z, expon(scale=1.0).cdf)
                out[comp] = (float(stat), float(p))
            else:
                out[comp] = (float("nan"), float("nan"))
        return out

    def residual_means(self, mu, A, beta):
        means = []
        for comp in self.COMPONENTS:
            z = self.time_rescaled_residuals(mu, A, beta, comp)
            means.append(float(np.nan if z.size == 0 else z.mean()))
        return np.array(means, dtype=float)

    def print_rate_diagnostics(self, mu, A, beta):
        K = self._build_int_kernel(A, beta)
        I = np.eye(self.N)
        try:
            Lambda_theory = np.linalg.solve(I - K, mu)
        except np.linalg.LinAlgError:
            Lambda_theory = np.full(self.N, np.nan)
        counts = np.array(
            [self.events[name].size for name in self.COMPONENTS],
            dtype=float,
        )
        emp_rate = counts / self.T
        print("comp   emp_rate    theory_rate    count")
        for i, name in enumerate(self.COMPONENTS):
            print(f"{name:>3}  {emp_rate[i]:10.4f}  {Lambda_theory[i]:12.4f}   {int(counts[i])}")
            
    def simulate_path(self, mu, A, beta, T=None, seed=None, rho_sim_target: float = 0.9, return_A_used: bool = False):
        A_used, rho_raw = self._stabilize_A(A, beta, rho_target=rho_sim_target)
        A = A_used

        if T is None:
            T = self.T

        rng = np.random.default_rng(seed)
        mu = np.asarray(mu, dtype=float)
        A = np.asarray(A, dtype=float)
        beta = np.asarray(beta, dtype=float)
        betac = np.maximum(beta, 1e-12)

        t = 0.0
        R = np.zeros(self.N, dtype=float)  # accumulatore per sorgente
        lam = mu.copy()               # intensità post-jump a t=0
        lam_star = lam.sum()

        times = []
        comps = []

        while True:
            Lambda_star = lam_star
            if Lambda_star <= 0:
                break

            # propose jump time
            w = -np.log(rng.random()) / Lambda_star
            s = t + w
            if s > T:
                break

            # decay degli accumulatori fino a s
            dt = s - t
            if dt < 0 or not np.isfinite(dt):
                break
            if dt > 0:
                R *= np.exp(-betac * dt)

            # intensità vera a s (pre-jump)
            lam_true = mu + A @ R
            Lambda_true = lam_true.sum()
            if Lambda_true <= 0 or (not np.all(np.isfinite(lam_true))):
                # processo muore
                t = s
                lam = lam_true
                lam_star = max(Lambda_true, 0.0)
                continue

            # accept / reject
            if rng.random() <= (Lambda_true / Lambda_star):
                # scegli quale componente salta
                probs = lam_true / Lambda_true
                j = rng.choice(self.N, p=probs)

                times.append(s)
                comps.append(j)

                # aggiorna accumulatore sorgente j
                R[j] += 1.0
                # post-jump intensities
                lam = lam_true + A[:, j]
                lam_star = lam.sum()
                t = s
            else:
                # rifiutato: stringi il bound
                lam = lam_true
                lam_star = lam.sum()
                t = s

        times = np.asarray(times, dtype=float)
        comps = np.asarray(comps, dtype=int)

        streams = {
            name: times[comps == i].copy()
            for i, name in enumerate(self.COMPONENTS)
        }
        if return_A_used:
            return streams, (times, comps), A_used
        return streams, (times, comps)

    def simulate_from_fit(self, fit, T=None, seed=None, rho_sim_target: float = 0.9):
        return self.simulate_path(fit.mu, fit.A, fit.beta, T=T, seed=seed, rho_sim_target=rho_sim_target)

    
    def plot_data_vs_simulation(self,
                                mu: np.ndarray,
                                A: np.ndarray,
                                beta: np.ndarray,
                                T: float = None,
                                seed: int = None,
                                title: str = None):

        if T is None:
            T = self.T

        # Simula
        streams_sim, (t_sim, c_sim) = self.simulate_path(
            mu, A, beta, T=T, seed=seed
        )

        fig, axes = plt.subplots(
            len(self.COMPONENTS),
            1,
            sharex=True,
            figsize=(10, 6),
            constrained_layout=True,
        )

        if title is None:
            title = "Empirical vs simulated events (Hawkes ND)"
        fig.suptitle(title)

        for i, name in enumerate(self.COMPONENTS):
            ax = axes[i]
            t_data = self.events[name]
            t_s = streams_sim[name]

            # dati reali
            if t_data.size > 0:
                ax.vlines(t_data, 0.0, 1.0,
                          linewidth=0.6,
                          alpha=0.7,
                          label="data")

            # simulato
            if t_s.size > 0:
                ax.vlines(t_s, 0.0, 1.0,
                          linewidth=0.6,
                          alpha=0.7,
                          label="sim",
                          colors="C1")

            ax.set_ylabel(name)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)

            # legende solo sulla prima riga
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("time")

        axes[0].set_xlim(0.0, T)

        return fig, axes, (streams_sim, (t_sim, c_sim))

    def intensity_from_path(self,
                            mu: np.ndarray,
                            A: np.ndarray,
                            beta: np.ndarray,
                            times: np.ndarray,
                            comps: np.ndarray,
                            T: float = None,
                            n_grid: int = 1000,
                            t_grid: np.ndarray | None = None):
        """
        Given a Hawkes trajectory (times, comps) (simulated or real),
        compute λ_i(t) on a time grid.

        If t_grid is provided, it is used directly (can be non-uniform, event-aligned).
        Otherwise a regular grid of n_grid points on [0,T] is used.

        Supports negative times in `times` (warmup / pre-history): they affect R(0),
        while the grid is on [0,T].

        Parameters
        ----------
        mu, A, beta : model parameters (d,), (d,d), (d,)
            beta_j applies to source/column j.
        times : (N,) event times (can include <0 warmup)
        comps : (N,) event component indices in {0,...,d-1}
        T : horizon (if None uses max(times) or self.T)
        n_grid : number of points if using regular grid
        t_grid : optional custom grid (sorted/unique will be enforced)

        Returns
        -------
        t_grid : (M,)
        lam_grid : (d, M) with λ_i(t_grid[k])
        """
        import numpy as np

        mu = np.asarray(mu, dtype=float)
        A = np.asarray(A, dtype=float)
        beta = np.asarray(beta, dtype=float)
        betac = np.maximum(beta, 1e-12)

        times = np.asarray(times, dtype=float)
        comps = np.asarray(comps, dtype=int)

        if T is None:
            if times.size > 0 and np.isfinite(times).any():
                T = float(np.nanmax(times))
            else:
                T = float(self.T)

        # --- build / validate grid ---
        if t_grid is None:
            t_grid = np.linspace(0.0, float(T), int(n_grid))
        else:
            t_grid = np.asarray(t_grid, dtype=float)
            # keep only finite points within [0,T]
            t_grid = t_grid[np.isfinite(t_grid)]
            t_grid = t_grid[(t_grid >= 0.0) & (t_grid <= float(T))]
            if t_grid.size == 0:
                t_grid = np.linspace(0.0, float(T), int(n_grid))

            # enforce sorted unique (important for dt >= 0)
            t_grid = np.unique(t_grid)

        lam_grid = np.zeros((self.N, t_grid.size), dtype=float)

        # --- sort events (allow negative pre-history) ---
        order = np.argsort(times)
        times = times[order]
        comps = comps[order]

        valid = np.isfinite(times) & (comps >= 0) & (comps < self.N)
        times = times[valid]
        comps = comps[valid]
        n_events = times.size

        R = np.zeros(self.N, dtype=float)  # kernel state per source
        # start from first negative event time if exists, else 0
        t_curr = float(times[0]) if (n_events > 0 and times[0] < 0.0) else 0.0
        ev_idx = 0

        # --- main sweep: process events up to each grid time ---
        for k, g in enumerate(t_grid):
            g = float(g)

            # incorporate all events with time <= g
            while ev_idx < n_events and float(times[ev_idx]) <= g:
                t_e = float(times[ev_idx])
                dt = t_e - t_curr
                if dt < -1e-12:
                    raise ValueError("Event times not non-decreasing after sort.")
                if dt > 0.0:
                    R *= np.exp(-betac * dt)
                    t_curr = t_e
                j = int(comps[ev_idx])
                R[j] += 1.0
                ev_idx += 1

            # decay from last processed time to g
            dt = g - t_curr
            if dt > 0.0:
                R *= np.exp(-betac * dt)
                t_curr = g

            lam_grid[:, k] = mu + A @ R

        return t_grid, lam_grid


    def simulate_and_intensity(self,
                               mu: np.ndarray,
                               A: np.ndarray,
                               beta: np.ndarray,
                               T: float = None,
                               seed: int = None,
                               n_grid: int = 1000):

        streams_sim, (times_sim, comps_sim), A_used = self.simulate_path(
            mu=mu, A=A, beta=beta, T=T, seed=seed, rho_sim_target=0.9, return_A_used=True
        )
        t_grid, lam_grid = self.intensity_from_path(
            mu=mu, A=A_used, beta=beta, times=times_sim, comps=comps_sim, T=T, n_grid=n_grid
        )
        return streams_sim, (times_sim, comps_sim), (t_grid, lam_grid)
    
    def plot_simulated_intensities(self,
                                   mu: np.ndarray,
                                   A: np.ndarray,
                                   beta: np.ndarray,
                                   T: float = None,
                                   seed: int = None,
                                   n_grid: int = 1000,
                                   title: str = None):

        streams_sim, (times_sim, comps_sim), (t_grid, lam_grid) = \
            self.simulate_and_intensity(
                mu=mu, A=A, beta=beta, T=T, seed=seed, n_grid=n_grid
            )

        fig, axes = plt.subplots(
            len(self.COMPONENTS),
            1,
            sharex=True,
            figsize=(10, 6),
            constrained_layout=True,
        )

        if title is None:
            title = "Simulated intensities (Hawkes ND)"
        fig.suptitle(title)

        for i, name in enumerate(self.COMPONENTS):
            ax = axes[i]
            lam_i = lam_grid[i, :]
            ax.plot(t_grid, lam_i, linewidth=1.0)
            t_s = streams_sim[name]
            if t_s.size > 0:
                ax.vlines(t_s, 0.0, lam_i.max() * 0.1,
                          linewidth=0.5, alpha=0.3)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("time")

        return fig, axes, (streams_sim, (times_sim, comps_sim)), (t_grid, lam_grid)


    
# =============================================================================
# Wrapper per LOB: aggregazione 8D→2D, finestre, seasonality opzionale
# =============================================================================

@dataclass
class WindowResult:
    t0: float
    t1: float
    fit: HawkesFitResult
    emp_rate: np.ndarray
    theory_rate: np.ndarray
    meanZ: np.ndarray
    ks: Dict[str, Tuple[float, float]]


class LOBHawkes:


    def __init__(self, events: Dict[str, np.ndarray], T: float, dimension = 2):
        self.events = {k: np.asarray(v, dtype=float) for k, v in events.items()}
        self.T = float(T)
        self.N = dimension
        self.COMPONENTS = COMPONENTS_2D
        self.events = self._build_events(self.events)
        # placeholders per seasonality
        self._season_bins = None
        self._season_s = None       # s_k
        self._season_cum = None     # cum integrale
        self._season_dt = None      # bin width

    # ---------- 8D -> ND aggregation ----------

    def _build_events(self, ev8: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:


        E_b = np.concatenate([
            np.asarray(ev8.get("E_b", []), dtype=float),
            np.asarray(ev8.get("HE_b", []), dtype=float),

        ])
        E_a = np.concatenate([
            np.asarray(ev8.get("E_a", []), dtype=float),
            np.asarray(ev8.get("HE_a", []), dtype=float),

        ])

        return {
            "E_b": np.sort(E_b),
            "E_a": np.sort(E_a),
        }

    # ---------- seasonality estimation & time-change ----------

    def estimate_seasonality(self, n_bins: int = 48, smooth: bool = True):
        """
        Stima s(t) piecewise-costante sui n_bins, sulla base del totale degli
        eventi (somma ND). Normalizza s_k a media 1.
        """
        T = self.T
        dt = T / n_bins
        counts_per_bin = np.zeros(n_bins, dtype=float)

        all_times = np.concatenate(list(self.events.values()))
        all_times = all_times[(all_times >= 0.0) & (all_times < T)]

        if all_times.size == 0:
            raise ValueError("No events to estimate seasonality.")

        idx = np.minimum((all_times / dt).astype(int), n_bins - 1)
        for k in idx:
            counts_per_bin[k] += 1.0

        rates = counts_per_bin / dt
        mean_rate = rates.mean() if rates.mean() > 0 else 1.0
        s = rates / mean_rate

        if smooth and n_bins >= 3:
            # moving average semplice
            s_pad = np.r_[s[-1], s, s[0]]
            s_sm = 0.25 * s_pad[:-2] + 0.5 * s_pad[1:-1] + 0.25 * s_pad[2:]
            s = s_sm

        # rinormalizza a media 1
        s /= max(s.mean(), 1e-12)

        # precomputo l'integrale cumulativo S(t) = ∫ s(u) du
        cum = np.zeros(n_bins + 1, dtype=float)
        for k in range(1, n_bins + 1):
            cum[k] = cum[k - 1] + s[k - 1] * dt

        self._season_bins = n_bins
        self._season_dt = dt
        self._season_s = s
        self._season_cum = cum

    def _S(self, t: np.ndarray) -> np.ndarray:
        """
        Time-change: S(t) = ∫_0^t s(u) du (piecewise costante).
        t può essere array.
        """
        if self._season_s is None:
            raise RuntimeError("Seasonality not estimated. Call estimate_seasonality().")

        t = np.asarray(t, dtype=float)
        n_bins = self._season_bins
        dt = self._season_dt
        s = self._season_s
        cum = self._season_cum

        # indice di bin
        idx = np.clip((t / dt).astype(int), 0, n_bins - 1)
        base = cum[idx]
        t0 = idx * dt
        return base + s[idx] * (t - t0)

    def build_deseasoned_events(self,
                                events: Dict[str, np.ndarray],
                                t0: float = 0.0,
                                t1: Optional[float] = None) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Applica il time-change S(t) su [t0, t1) e riporta i tempi
        a partire da 0: τ = S(t) - S(t0).
        Restituisce (events_rescaled, T_rescaled).
        """
        if t1 is None:
            t1 = self.T

        S0 = float(self._S(np.array([t0]))[0])
        S1 = float(self._S(np.array([t1]))[0])
        T_rescaled = S1 - S0

        out = {}
        for name, times in events.items():
            mask = (times >= t0) & (times < t1)
            t_sel = times[mask]
            if t_sel.size == 0:
                out[name] = np.array([], dtype=float)
            else:
                tau = self._S(t_sel) - S0
                out[name] = np.sort(tau)

        return out, T_rescaled

    # ---------- window calibration ----------

    def fit_windows(self,
                    windows: List[Tuple[float, float]],
                    deseasonalize: bool = False,
                    n_bins_season: int = 48,
                    rho_cap_fit: float = 0.999)-> Dict[int, WindowResult]:

        if deseasonalize:
            self.estimate_seasonality(n_bins=n_bins_season, smooth=True)

        results = {}

        for k, (t0, t1) in enumerate(windows):
            if deseasonalize:
                ev2w, Tw = self.build_deseasoned_events(self.events, t0=t0, t1=t1)
            else:
                # semplice ritaglio + shift a 0
                ev2w = {}
                for name, times in self.events.items():
                    mask = (times >= t0) & (times < t1)
                    ev2w[name] = np.sort(times[mask] - t0)
                Tw = t1 - t0

            print(f"\n=== Window {k}: [{t0:.2f}, {t1:.2f})  (T={Tw:.2f}) ===")
            cal = HawkesCalibrator(ev2w, Tw)
            fit = cal.fit(rho_cap_fit=rho_cap_fit)
            print(f"success={fit.success}, rho(K)={fit.rho:.2f}")


            # diagnostics
            K = cal._build_int_kernel(fit.A, fit.beta)
            I = np.eye(self.N)
            try:
                Lambda_theory = np.linalg.solve(I - K, fit.mu)
            except np.linalg.LinAlgError:
                Lambda_theory = np.full(self.N, np.nan)
            counts = np.array(
                [ev2w[name].size for name in self.COMPONENTS],
                dtype=float,
            )
            emp_rate = counts / max(Tw, 1e-9)

            meanZ = cal.residual_means(fit.mu, fit.A, fit.beta)
            ks = cal.ks_exp_test(fit.mu, fit.A, fit.beta)

            results[k] = WindowResult(
                t0=t0,
                t1=t1,
                fit=fit,
                emp_rate=emp_rate,
                theory_rate=Lambda_theory,
                meanZ=meanZ,
                ks=ks,
            )

        return results
    # ---------- simulation ----------

# ---------- simulation ----------

    def simulate_piecewise_ogata(self, window_results, seed: int = None, rho_sim_target: float = 0.9):


        # normalizza in lista ordinata per t0
        if isinstance(window_results, dict):
            win_list = list(window_results.values())
        else:
            win_list = list(window_results)

        if not win_list:
            return (
                {"E_b": np.array([]),
                 "E_a": np.array([]),},
                np.array([]),
                np.array([], dtype=int),
            )

        win_list.sort(key=lambda w: w.t0)

        rng = np.random.default_rng(seed)

        all_times: List[float] = []
        all_comps: List[int] = []

        # per memoria: lista dei tempi simulati per ciascun tipo j=0..3
        times_per_comp = [[] for _ in range(self.N)]

        # tempo globale iniziale
        t = float(win_list[0].t0)

        for w in win_list:
            t0 = float(w.t0)
            t1 = float(w.t1)
            mu = np.asarray(w.fit.mu, dtype=float)
            A = np.asarray(w.fit.A, dtype=float)
            beta = np.asarray(w.fit.beta, dtype=float)
            # stabilize ONLY for simulation
            K_raw = A / np.maximum(beta, 1e-12)[np.newaxis, :]
            rho_raw = float(np.max(np.abs(eigvals(K_raw))))
            if np.isfinite(rho_raw) and rho_raw > rho_sim_target:
                A = A * (rho_sim_target / rho_raw)

            betac = np.maximum(beta, 1e-12)

            # assicuriamoci di non andare indietro nel tempo
            if t < t0:
                t = t0
            if t >= t1:
                # questa finestra è già "saltata", vai alla prossima
                continue

            # ricostruisci lo stato dei kernel R_j(t) al tempo corrente t
            R = np.zeros(self.N, dtype=float)
            for j in range(self.N):
                ts_j = np.asarray(times_per_comp[j], dtype=float)
                if ts_j.size > 0:
                    R[j] = np.exp(-betac[j] * (t - ts_j)).sum()

            # intensità iniziale in questa finestra
            lam = mu + A @ R
            lam_star = lam.sum()

            # Ogata dentro [t0, t1) con parametri fissi (mu,A,beta)
            while True:
                Lambda_star = lam_star
                if Lambda_star <= 0 or not np.isfinite(Lambda_star):
                    break

                # proponi tempo del prossimo evento
                w_wait = -np.log(rng.random()) / Lambda_star
                s = t + w_wait
                if s >= t1:
                    t = t1   # NOT s
                    break


                dt = s - t
                if dt < 0 or not np.isfinite(dt):
                    break

                # decadimento kernel fino a s
                if dt > 0:
                    R *= np.exp(-betac * dt)

                # intensità vera a s
                lam_true = mu + A @ R
                Lambda_true = lam_true.sum()
                if Lambda_true <= 0 or (not np.all(np.isfinite(lam_true))):
                    t = s
                    lam = lam_true
                    lam_star = max(Lambda_true, 0.0)
                    continue

                # accept / reject
                if rng.random() <= (Lambda_true / Lambda_star):
                    # scegli tipo di evento
                    probs = lam_true / Lambda_true
                    j = rng.choice(self.N, p=probs)

                    all_times.append(s)
                    all_comps.append(int(j))
                    times_per_comp[j].append(s)

                    # aggiorna stato dopo il salto
                    R[j] += 1.0
                    lam = lam_true + A[:, j]
                    lam_star = lam.sum()
                    t = s
                else:
                    # rifiutato: stringi il bound
                    lam = lam_true
                    lam_star = lam.sum()
                    t = s

        times_sim = np.asarray(all_times, dtype=float)
        comps_sim = np.asarray(all_comps, dtype=int)

        streams_sim = {
            name: times_sim[comps_sim == i].copy()
            for i, name in enumerate(self.COMPONENTS)
        }
        return streams_sim, times_sim, comps_sim

