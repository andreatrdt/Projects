from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from HawkesCalibrator import HawkesCalibrator

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pathlib import Path
import hashlib
from datetime import time
import os
import pickle


@dataclass(frozen=True)
class HJBSolverParams:
    # horizon
    T: float = 600.0
    n_time: int = 300  # set via dt_hjb: n_time = int(T/dt_hjb)

    # inventory grid
    Q: int = 10
    q0: int = 1

    # preferences / price
    gamma: float = 0.8
    sigma: float = 0.01
    eta: float = 0.0

    # Inventory penalty (terminal + running).
    # Backward-compatible behavior:
    #   - eta_T: terminal penalty coefficient (if None, uses eta)
    #   - eta_run: running penalty coefficient inside the PDE (if None, uses eta)
    # If you want the OLD behavior (terminal-only), set eta_run=0.0 explicitly.
    eta_T: Optional[float] = None
    eta_run: Optional[float] = None
    kappa_jump: float = 0.01  # mid-price jump size ("mid tick")

    # quoting tick (price grid for bid/ask). If None, falls back to delta_step or kappa_jump.
    quote_tick: Optional[float] = None

    # execution capture f(δ) = min(1, A_fill * exp(-k_fill δ))
    A_fill_a: float = 1.0
    A_fill_b: float = 1.0
    k_fill_a: float = 1.0
    k_fill_b: float = 1.0

    # marking probabilities p±(rho)
    p_bar: float = 0.2
    k_imp: float = 10.0
    theta_imp: float = 0.0

    # intensity grids (geometric, as in main.tex)
    M_plus: int = 25
    M_minus: int = 25
    grid_gamma: float = 2.0
    lmax_mult: float = 8.0

    # quote grid-search
    delta_max: float = 0.20

    delta_min: Optional[float] = None    # default: 1 tick
    delta_step: Optional[float] = None   # default: 1 tick
    delta0_fill_a: float = 0.0
    delta0_fill_b: float = 0.0

    store_every: int = 1


@dataclass
class HJBSolution:
    t_grid: np.ndarray
    q_grid: np.ndarray
    lam_plus_grid: np.ndarray
    lam_minus_grid: np.ndarray
    delta_a: np.ndarray
    delta_b: np.ndarray

    tick: float              # mid-jump tick (kappa_jump)
    quote_tick: float        # <-- ADD THIS
    store_every: int



def _geom_grid(mu: float, M: int, g: float, lmax_mult: float) -> np.ndarray:
    mu = float(mu)
    if M < 2:
        return np.array([mu], dtype=float)
    l_min = mu
    l_max = max(l_min * lmax_mult, l_min + 1e-12)
    k = np.arange(M, dtype=float)
    frac = (np.exp(g * (k / (M - 1.0))) - 1.0) / (np.exp(g) - 1.0)
    grid = l_min + (l_max - l_min) * frac
    grid[0] = l_min
    grid[-1] = l_max
    return grid


def _upwind_drift(grid: np.ndarray, beta: float, mu: float) -> sp.csr_matrix:
    grid = np.asarray(grid, dtype=float)
    M = grid.size
    diag = np.zeros(M)
    sub = np.zeros(M - 1)
    for i in range(1, M):
        denom = grid[i] - grid[i - 1]
        rate = float(beta) * max(grid[i] - float(mu), 0.0) / max(denom, 1e-18)
        sub[i - 1] = rate
        diag[i] = -rate
    diag[0] = 0.0
    return sp.diags([diag, sub], offsets=[0, -1], format="csr")


def _interp2d(V: np.ndarray, gx: np.ndarray, gy: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = np.clip(x, gx[0], gx[-1])
    y = np.clip(y, gy[0], gy[-1])

    ix = np.searchsorted(gx, x, side="right") - 1
    iy = np.searchsorted(gy, y, side="right") - 1
    ix = np.clip(ix, 0, gx.size - 2)
    iy = np.clip(iy, 0, gy.size - 2)

    x0 = gx[ix]; x1 = gx[ix + 1]
    y0 = gy[iy]; y1 = gy[iy + 1]
    wx = (x - x0) / (x1 - x0 + 1e-18)
    wy = (y - y0) / (y1 - y0 + 1e-18)

    v00 = V[ix, iy]
    v10 = V[ix + 1, iy]
    v01 = V[ix, iy + 1]
    v11 = V[ix + 1, iy + 1]
    return ((1-wx)*(1-wy)*v00 + wx*(1-wy)*v10 + (1-wx)*wy*v01 + wx*wy*v11)


class HJBThetaIMEXSolver:
    """
    Memory-safe IMEX solver aligned with your main.tex structure but upgraded:
      - no giant theta[t,q,lp,lm] allocation (rolling theta)
      - tick-consistent delta grid
      - vectorized delta argmin (fast)
      - stores only delta_a/delta_b (policy), optionally downsampled in time
    """

    def __init__(
        self,
        mu_plus: float, mu_minus: float,
        A_pm: np.ndarray,          # rows/cols ordered (+,-)
        beta_plus: float, beta_minus: float,
        params: Optional[HJBSolverParams] = None
    ):
        self.mu_plus = float(mu_plus)
        self.mu_minus = float(mu_minus)
        self.A = np.asarray(A_pm, dtype=float).reshape(2, 2)
        self.beta_plus = float(beta_plus)
        self.beta_minus = float(beta_minus)
        self.p = params if params is not None else HJBSolverParams()

    def _build_delta_grid(self) -> np.ndarray:
        p = self.p
        tick = float(p.kappa_jump)
        delta_min = tick if p.delta_min is None else float(p.delta_min)
        delta_step = tick if p.delta_step is None else float(p.delta_step)

        delta_min = max(delta_min, 0.0)
        delta_step = max(delta_step, 1e-12)

        # ensure at least 1 point and include delta_max
        grid = np.arange(delta_min, p.delta_max + 0.5 * delta_step, delta_step, dtype=float)
        if grid.size == 0:
            grid = np.array([p.delta_max], dtype=float)
        if grid[-1] < p.delta_max - 1e-12:
            grid = np.append(grid, p.delta_max)
        return grid

    def solve(self) -> HJBSolution:
        p = self.p
        dt = p.T / p.n_time

        q_grid = np.arange(-p.Q, p.Q + 1, dtype=int)
        nQ = q_grid.size

        lam_plus_grid = _geom_grid(self.mu_plus, p.M_plus, p.grid_gamma, p.lmax_mult)
        lam_minus_grid = _geom_grid(self.mu_minus, p.M_minus, p.grid_gamma, p.lmax_mult)

        Dp = _upwind_drift(lam_plus_grid, self.beta_plus, self.mu_plus)
        Dm = _upwind_drift(lam_minus_grid, self.beta_minus, self.mu_minus)
        I_p = sp.eye(p.M_plus, format="csr")
        I_m = sp.eye(p.M_minus, format="csr")
        L_hawkes = sp.kron(Dp, I_m, format="csr") + sp.kron(I_p, Dm, format="csr")

        nL = p.M_plus * p.M_minus

        # rolling theta: only keep (t_{n+1}) and (t_n), for all q
        theta_next = np.zeros((nQ, p.M_plus, p.M_minus), dtype=np.float64)
        theta_curr = np.zeros_like(theta_next)

        # terminal condition
        # NOTE: eta_T controls the terminal inventory penalty, eta_run controls the running penalty.
        # By default (eta_T=None, eta_run=None) we use `eta` for BOTH, to ensure the policy
        # actively mean-reverts inventory throughout the horizon (not only at maturity).
        # If you want terminal-only (old behavior), pass eta_run=0.0.
        eta_T = float(p.eta if p.eta_T is None else p.eta_T)
        eta_run = float(p.eta if p.eta_run is None else p.eta_run)
        theta_T = np.exp(0.5 * p.gamma * eta_T * (q_grid.astype(float) ** 2))
        for iq in range(nQ):
            theta_next[iq, :, :] = theta_T[iq]

        # factorize implicit matrices per q
        LU: Dict[int, spla.SuperLU] = {}
        for iq, q in enumerate(q_grid):
            pot = (0.5 * (p.sigma ** 2) * (p.gamma ** 2) * (float(q) ** 2)
                   + 0.5 * p.gamma * eta_run * (float(q) ** 2))
            M = (sp.eye(nL, format="csr") - dt * L_hawkes) - (dt * pot) * sp.eye(nL, format="csr")
            LU[iq] = spla.splu(M.tocsc())

        # grids for p±(rho)
        LP, LM = np.meshgrid(lam_plus_grid, lam_minus_grid, indexing="ij")
        denom = LP + LM
        rho = np.zeros_like(denom)
        m = denom > 1e-12
        rho[m] = (LP[m] - LM[m]) / denom[m]

        # Marking probabilities based on order flow imbalance
        # Parameterization (scaled logistic):
        #   z = k_imp * (rho - theta_imp)
        #   p_plus  = p_bar / (1 + exp(-z))   -> increases with rho
        #   p_minus = p_bar / (1 + exp(+z))   -> decreases with rho
        # Here p_bar is the *maximum* marking probability (as |z| -> +inf).
        p_bar = float(np.clip(p.p_bar, 1e-6, 1.0))
        z = p.k_imp * (rho - p.theta_imp)
        p_plus = p_bar / (1.0 + np.exp(-z))
        p_minus = p_bar / (1.0 + np.exp(+z))


        # Hawkes jumps in intensity after events (+ or -)
        dLP_plus,  dLM_plus  = self.A[0, 0], self.A[1, 0]
        dLP_minus, dLM_minus = self.A[0, 1], self.A[1, 1]

        # delta grid (tick-based)
        dgrid = self._build_delta_grid()
        nD = dgrid.size

        # precompute fill weights for each side on the delta grid
        d_eff_a = np.maximum(dgrid - float(getattr(p, "delta0_fill_a", 0.0)), 0.0)
        d_eff_b = np.maximum(dgrid - float(getattr(p, "delta0_fill_b", 0.0)), 0.0)

        f_a = np.minimum(1.0, p.A_fill_a * np.exp(-p.k_fill_a * d_eff_a))
        f_b = np.minimum(1.0, p.A_fill_b * np.exp(-p.k_fill_b * d_eff_b))

        w_a = f_a * np.exp(-p.gamma * p.q0 * dgrid)  # shape (nD,)
        w_b = f_b * np.exp(-p.gamma * p.q0 * dgrid)

        # store deltas only every store_every steps
        # quoting tick: used to round bid/ask quotes (distinct from kappa_jump which is the mid-jump size)
        quote_tick = (float(p.quote_tick) if getattr(p, "quote_tick", None) is not None
                      else (float(p.delta_step) if p.delta_step is not None else float(p.kappa_jump)))
        store_every = max(int(p.store_every), 1)
        store_times = np.arange(0, p.n_time, store_every, dtype=int)
        t_full = np.linspace(0.0, p.T, p.n_time + 1)
        t_grid = t_full[store_times]  # policy time grid

        n_store = store_times.size
        delta_a = np.zeros((n_store, nQ, p.M_plus, p.M_minus), dtype=np.float32)
        delta_b = np.zeros((n_store, nQ, p.M_plus, p.M_minus), dtype=np.float32)

        def interp(slice2d, x, y):
            return _interp2d(slice2d, lam_plus_grid, lam_minus_grid, x, y)

        for n in range(p.n_time - 1, -1, -1):
            # post-event intensities (same for all q)
            LPp = np.clip(LP + dLP_plus,  lam_plus_grid[0],  lam_plus_grid[-1])
            LMp = np.clip(LM + dLM_plus,  lam_minus_grid[0], lam_minus_grid[-1])
            LPm = np.clip(LP + dLP_minus, lam_plus_grid[0],  lam_plus_grid[-1])
            LMm = np.clip(LM + dLM_minus, lam_minus_grid[0], lam_minus_grid[-1])

            # for each q
            for iq, q in enumerate(q_grid):
                Th_curr = theta_next[iq, :, :]  # explicit control uses t_{n+1}

                # miss terms (no inventory change)
                Th_plus0  = interp(theta_next[iq, :, :], LPp, LMp)
                Th_minus0 = interp(theta_next[iq, :, :], LPm, LMm)

                # hit terms (inventory changes)
                iq_sell = iq - p.q0  # ask hit -> we sell
                iq_buy  = iq + p.q0  # bid hit -> we buy

                Th_hit_a = interp(theta_next[iq_sell, :, :], LPp, LMp) if (0 <= iq_sell < nQ) else None
                Th_hit_b = interp(theta_next[iq_buy,  :, :], LPm, LMm) if (0 <= iq_buy  < nQ) else None

                # price impact factors (jump part acts on q*S term under exponential transform)
                C_a_hit = (1.0 - p_plus) + p_plus * np.exp(-p.gamma * p.kappa_jump * float(q - p.q0))
                C_a_0   = (1.0 - p_plus) + p_plus * np.exp(-p.gamma * p.kappa_jump * float(q))

                C_b_hit = (1.0 - p_minus) + p_minus * np.exp(+p.gamma * p.kappa_jump * float(q + p.q0))
                C_b_0   = (1.0 - p_minus) + p_minus * np.exp(+p.gamma * p.kappa_jump * float(q))

                # --- control: delta_a (MIN in Theta space) ---
                if Th_hit_a is not None:
                    A = (C_a_hit * Th_hit_a)[None, :, :]     # (1,Mp,Mm)
                    B = (C_a_0   * Th_plus0)[None, :, :]
                    payoff_a = w_a[:, None, None] * A + (1.0 - f_a)[:, None, None] * B  # (nD,Mp,Mm)
                    idx_a = np.argmin(payoff_a, axis=0)
                    best_da = dgrid[idx_a]
                    best_a  = payoff_a[idx_a, np.arange(p.M_plus)[:, None], np.arange(p.M_minus)[None, :]]
                else:
                    # cannot sell: "do nothing" candidate
                    best_da = np.full_like(Th_curr, p.delta_max, dtype=float)
                    best_a  = C_a_0 * Th_plus0

                # --- control: delta_b (MIN in Theta space) ---
                if Th_hit_b is not None:
                    A = (C_b_hit * Th_hit_b)[None, :, :]
                    B = (C_b_0   * Th_minus0)[None, :, :]
                    payoff_b = w_b[:, None, None] * A + (1.0 - f_b)[:, None, None] * B
                    idx_b = np.argmin(payoff_b, axis=0)
                    best_db = dgrid[idx_b]
                    best_b  = payoff_b[idx_b, np.arange(p.M_plus)[:, None], np.arange(p.M_minus)[None, :]]
                else:
                    best_db = np.full_like(Th_curr, p.delta_max, dtype=float)
                    best_b  = C_b_0 * Th_minus0

                # store policy if this time is on the storage grid
                if (n % store_every) == 0:
                    k = n // store_every
                    delta_a[k, iq, :, :] = best_da.astype(np.float32)
                    delta_b[k, iq, :, :] = best_db.astype(np.float32)

                # source term
                source = (LP * (best_a - Th_curr)) + (LM * (best_b - Th_curr))

                rhs = Th_curr.reshape(-1) + dt * source.reshape(-1)
                theta_curr[iq, :, :] = LU[iq].solve(rhs).reshape(p.M_plus, p.M_minus)

            # roll time
            theta_next, theta_curr = theta_curr, theta_next
            theta_curr.fill(0.0)

        return HJBSolution(
            t_grid=t_grid,
            q_grid=q_grid,
            lam_plus_grid=lam_plus_grid,
            lam_minus_grid=lam_minus_grid,
            delta_a=delta_a,
            delta_b=delta_b,
            tick=float(p.kappa_jump),
            quote_tick=float(quote_tick),
            store_every=store_every,
        )


class HJBQuotePolicyFromSolution:
    def __init__(self, sol: HJBSolution):
        self.sol = sol

    def __call__(self, t: float, S: float, q: float, lam_plus: float, lam_minus: float) -> Tuple[float, float]:
        sol = self.sol

        # time index on policy grid
        it = int(np.clip(np.searchsorted(sol.t_grid, t, side="right") - 1, 0, sol.t_grid.size - 1))

        # inventory index
        q_int = int(np.clip(np.round(q), sol.q_grid[0], sol.q_grid[-1]))
        iq = int(q_int - sol.q_grid[0])

        # clip intensities
        lp = float(np.clip(lam_plus, sol.lam_plus_grid[0], sol.lam_plus_grid[-1]))
        lm = float(np.clip(lam_minus, sol.lam_minus_grid[0], sol.lam_minus_grid[-1]))

        da = float(_interp2d(sol.delta_a[it, iq, :, :], sol.lam_plus_grid, sol.lam_minus_grid, lp, lm))
        db = float(_interp2d(sol.delta_b[it, iq, :, :], sol.lam_plus_grid, sol.lam_minus_grid, lp, lm))

        # raw quotes
        bid = float(S) - max(db, 0.0)
        ask = float(S) + max(da, 0.0)

        # snap quotes to the quoting tick (NOT the mid-jump tick)
        tick = float(getattr(sol, "quote_tick", sol.tick))
        bid = np.floor(bid / tick) * tick
        ask = np.ceil(ask / tick) * tick
        if ask <= bid:
            ask = bid + tick

        return float(bid), float(ask)



def estimate_tick_size_from_series(x: np.ndarray) -> float:
    """
    Stima tick size come moda dei delta positivi non-null.
    Usa mid o (meglio) best bid/ask se disponibili.
    """
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    dx = dx[np.isfinite(dx)]
    dx = dx[dx > 0]
    if len(dx) == 0:
        raise ValueError("Impossibile stimare tick: nessun delta positivo trovato.")
    # robust: prendi i più piccoli e fai moda su rounding a 1e-6
    dx_small = np.sort(dx)[:max(1000, int(0.01 * len(dx)))]
    dx_round = np.round(dx_small, 6)
    vals, counts = np.unique(dx_round, return_counts=True)
    tick = float(vals[np.argmax(counts)])
    return tick


def estimate_quote_tick_from_bid_ask(bid: np.ndarray, ask: np.ndarray) -> float:
    """Estimate the *quoting* tick from best bid/ask series.

    In LOB data, the mid-price tick can be half of the quote tick (because mid=(bid+ask)/2).
    For quoting and snapping bid/ask prices, the relevant tick is the quote tick.

    Strategy: take non-zero absolute differences of bid and ask separately, pool them,
    and return the modal small increment (rounded to 1e-6).
    """
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    if bid.shape != ask.shape:
        raise ValueError("bid and ask must have the same shape")

    db = np.abs(np.diff(bid))
    da = np.abs(np.diff(ask))
    d = np.concatenate([db, da])
    d = d[np.isfinite(d)]
    d = d[d > 0]
    if d.size == 0:
        raise ValueError("Cannot estimate quote tick: no non-zero bid/ask changes found")

    # focus on the smallest changes (robust to big jumps)
    d_small = np.sort(d)[:max(2000, int(0.01 * d.size))]
    d_round = np.round(d_small, 6)
    vals, counts = np.unique(d_round, return_counts=True)
    tick = float(vals[np.argmax(counts)])
    return tick

def estimate_sigma_arithmetic(mid: np.ndarray, t: np.ndarray, dt_vol: float = 1.0) -> float:
    """
    Stima sigma per modello ARITMETICO: dS = sigma dW
    -> sigma in [price / sqrt(second)].
    Interpola mid su griglia uniforme dt_vol (tipicamente 0.5s o 1s).
    """
    mid = np.asarray(mid, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = np.isfinite(mid) & np.isfinite(t)
    mid, t = mid[mask], t[mask]
    order = np.argsort(t)
    mid, t = mid[order], t[order]

    t0, t1 = float(t[0]), float(t[-1])
    if t1 - t0 < 10 * dt_vol:
        raise ValueError("Finestra troppo corta per stimare sigma in modo stabile.")

    grid = np.arange(t0, t1, dt_vol)
    mid_g = np.interp(grid, t, mid)
    dS = np.diff(mid_g)
    # sigma^2 ≈ E[dS^2]/dt
    sigma = float(np.sqrt(np.mean(dS**2) / dt_vol))
    return sigma

def cache_load_or_compute(filepath: str | Path, compute_fn):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        with open(filepath, "rb") as f:
            return pickle.load(f)
    out = compute_fn()
    with open(filepath, "wb") as f:
        pickle.dump(out, f)
    return out

def hjb_signature(params: "HJBSolverParams") -> str:
    """
    Firma breve per evitare di ricaricare soluzioni HJB sbagliate
    quando cambi gamma/eta/sigma/tick/griglie.
    """
    # metti qui SOLO ciò che cambia davvero la soluzione
    payload = dict(
        T=params.T,
        n_time=params.n_time,
        Q=params.Q,
        q0=params.q0,
        gamma=params.gamma,
        eta=params.eta,
        sigma=params.sigma,
        tick=params.kappa_jump,
        quote_tick=getattr(params, "quote_tick", None),
        M_plus=params.M_plus,
        M_minus=params.M_minus,
        delta_max=params.delta_max,
        delta_min=params.delta_min,
        delta_step=params.delta_step,
        A_fill_a=params.A_fill_a,
        A_fill_b=params.A_fill_b,
        k_fill_a=params.k_fill_a,
        k_fill_b=params.k_fill_b,
        p_bar=params.p_bar,
        k_imp=params.k_imp,
        theta_imp=params.theta_imp,
        store_every=getattr(params, "store_every", None),

    )
    raw = repr(payload).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:10]


class HJBPolicyAdapter:
    """Collega la policy ottima del solver HJB al backtester."""
    def __init__(self, solver_policy_fn, fit_res):
        self.policy_fn = solver_policy_fn
        self.mu = fit_res.mu

    def __call__(self, t, S, q, lam_plus, lam_minus):
        # Se il backtester passa None (inizio), usa le medie storiche
        lp = lam_plus if lam_plus is not None else self.mu[1]
        lm = lam_minus if lam_minus is not None else self.mu[0]
        
        # Ottieni spread ottimi dal solver
        # Nota: policy_fn deve accettare (t, q, lam_b, lam_a)
        db, da = self.policy_fn(t, q, lm, lp)
        
        # Converte in prezzi limite
        bid = S - db
        ask = S + da
        return bid, ask

@dataclass
class HawkesBacktestModel:
    """
    Modello di esecuzione per il backtest.
    Usa la probabilità di fill calibrata: Lambda = lambda_mkt * A * exp(-k * delta)
    """
    A_fill: float
    k_fill: float
    delta0_fill: float

    def intensities(self, t, delta_a, delta_b, lam_plus=None, lam_minus=None):
        # Intensità base del mercato (Hawkes)
        lp = lam_plus if lam_plus is not None else 0.0
        lm = lam_minus if lam_minus is not None else 0.0
        
        # Probabilità di esecuzione (Microstructure model)
        da = max(float(delta_a) - float(self.delta0_fill), 0.0)
        db = max(float(delta_b) - float(self.delta0_fill), 0.0)
        prob_a = min(1.0, self.A_fill * np.exp(-self.k_fill * da))
        prob_b = min(1.0, self.A_fill * np.exp(-self.k_fill * db))


        
        # Intensità effettiva di esecuzione per noi
        lam_exec_a = lp * prob_a
        lam_exec_b = lm * prob_b
        
        return lam_exec_a, lam_exec_b 


def slice_mid_episode(data, t_start, T, t_grid):
    t_data = data["time"].values
    mid_data = data["mid_price"].values
    # shift to local time
    mid_ep = np.interp(t_start + t_grid, t_data, mid_data)
    return mid_ep

def slice_event_counts(events, t_start, T, t_grid, key_plus="E_a", key_minus="E_b"):
    buy = events[key_plus]
    sell = events[key_minus]

    mask_b = (buy  >= t_start) & (buy  < t_start + T)
    mask_s = (sell >= t_start) & (sell < t_start + T)

    buy_local  = buy[mask_b]  - t_start
    sell_local = sell[mask_s] - t_start

    Np = np.histogram(buy_local,  bins=t_grid)[0]
    Nm = np.histogram(sell_local, bins=t_grid)[0]
    return Np, Nm

def intensity_on_real_episode_2d(events, fit_res, t_start, T, n_grid_points):
    # Localize events to the episode [t_start, t_start+T)
    E_b = events["E_b"]
    E_a = events["E_a"]

    E_b_loc = E_b[(E_b >= t_start) & (E_b < t_start + T)] - t_start
    E_a_loc = E_a[(E_a >= t_start) & (E_a < t_start + T)] - t_start

    # Build a tiny "episode events dict" (2D)
    ep_events = {"E_b": E_b_loc, "E_a": E_a_loc}

    hc = HawkesCalibrator(ep_events, T, dimension=2)

    t_grid, lam_grid = hc.intensity_from_path(
        mu=fit_res.mu,
        A=fit_res.A,
        beta=fit_res.beta,
        times=hc._merged_times,
        comps=hc._merged_comp,
        T=T,
        n_grid=n_grid_points,
    )
    # COMPONENTS_2D = ["E_b","E_a"]
    lam_minus = lam_grid[0, :]  # E_b
    lam_plus  = lam_grid[1, :]  # E_a
    return t_grid, lam_plus, lam_minus
