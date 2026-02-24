# control_frame.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, Dict
import numpy as np


# -----------------------------
# Interfaces
# -----------------------------
class QuotePolicy(Protocol):
    """
    Policy interface.
    Given state, return (bid, ask) quotes.
    You can ignore lambda+/- if your policy doesn't use them (e.g. Avellaneda-Stoikov).
    """
    def __call__(
        self,
        t: float,
        S: float,
        q: float,
        lam_plus: Optional[float] = None,
        lam_minus: Optional[float] = None,
    ) -> Tuple[float, float]:
        ...


class ArrivalModel(Protocol):
    """
    Arrival model interface.
    Given offsets delta_a, delta_b, return intensities (Lambda^a, Lambda^b).
    """
    def intensities(
        self,
        t: float,
        delta_a: float,
        delta_b: float,
        lam_plus: Optional[float] = None,
        lam_minus: Optional[float] = None,
    ) -> Tuple[float, float]:
        ...


# -----------------------------
# Arrival models
# -----------------------------
@dataclass
class PoissonExpArrival:
    """
    Classic Avellaneda–Stoikov: Lambda = A * exp(-kappa * delta)
    Matches your current simulator logic. :contentReference[oaicite:1]{index=1}
    """
    A: float
    kappa: float

    def intensities(self, t, delta_a, delta_b, lam_plus=None, lam_minus=None):
        lam_a = float(self.A) * np.exp(-float(self.kappa) * max(float(delta_a), 0.0))
        lam_b = float(self.A) * np.exp(-float(self.kappa) * max(float(delta_b), 0.0))
        return lam_a, lam_b


@dataclass
class HawkesThinningArrival:
    """
    Thesis-consistent thinning:
        Lambda^a_t = f_a(delta_a) * lambda^+_t
        Lambda^b_t = f_b(delta_b) * lambda^-_t
    with f(delta)=exp(-kappa*delta) by default, mirroring the AS shape. :contentReference[oaicite:2]{index=2}
    """
    kappa: float
    eps: float = 1e-12  # avoid negative/zero intensities

    def intensities(self, t, delta_a, delta_b, lam_plus=None, lam_minus=None):
        lp = float(lam_plus) if lam_plus is not None else 0.0
        lm = float(lam_minus) if lam_minus is not None else 0.0
        fa = np.exp(-float(self.kappa) * max(float(delta_a), 0.0))
        fb = np.exp(-float(self.kappa) * max(float(delta_b), 0.0))
        lam_a = max(self.eps, fa * lp)
        lam_b = max(self.eps, fb * lm)
        return lam_a, lam_b

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class ControlBacktester:
    """
    Thesis-consistent backtest frame.

    - Aggressive order flow: N^+ ~ Poisson(lambda^+ dt), N^- ~ Poisson(lambda^- dt)
    - Our executions are a thinning of N^+, N^- via arrival.intensities()
    - Midprice: diffusion + marked jumps M^+, M^- (Bernoulli marking of N^+, N^-)
    """
    policy: "QuotePolicy"
    arrival: "ArrivalModel"
    Q_max: float = np.inf

    # price model
    tick_size: float = 0.01      # jump size κ
    sigma: float = 0.1           # diffusion vol

    # marking probabilities p±(rho)
    p_bar: float = 0.2
    k_imp: float = 10.0
    theta: float = 0.0

    def run(
        self,
        mid: np.ndarray,
        mid_replay: Optional[np.ndarray],
        t_secs: np.ndarray,
        q0: float = 0.0,
        x0: float = 0.0,
        trade_size: float = 1.0,
        lam_plus_path: Optional[np.ndarray] = None,
        lam_minus_path: Optional[np.ndarray] = None,
        N_plus_counts: Optional[np.ndarray] = None,
        N_minus_counts: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:

        mid = np.asarray(mid, dtype=float)
        t_secs = np.asarray(t_secs, dtype=float)

        if mid.ndim != 1 or t_secs.ndim != 1 or mid.size != t_secs.size:
            raise ValueError("mid and t_secs must be 1D arrays of same length.")
        if mid.size < 2:
            raise ValueError("Need at least 2 time points.")

        if mid_replay is not None:
            mid_replay = np.asarray(mid_replay, dtype=float)
            if mid_replay.shape != mid.shape:
                raise ValueError("mid_replay must have same shape as mid.")

        if lam_plus_path is not None:
            lam_plus_path = np.asarray(lam_plus_path, dtype=float)
            if lam_plus_path.shape != mid.shape:
                raise ValueError("lam_plus_path must have same shape as mid.")
        if lam_minus_path is not None:
            lam_minus_path = np.asarray(lam_minus_path, dtype=float)
            if lam_minus_path.shape != mid.shape:
                raise ValueError("lam_minus_path must have same shape as mid.")

        if trade_size <= 0:
            raise ValueError("trade_size must be positive.")

        rng = np.random.default_rng(seed)

        n = mid.size
        dt = np.diff(t_secs)

        bid = np.full(n, np.nan, dtype=float)
        ask = np.full(n, np.nan, dtype=float)
        q = np.zeros(n, dtype=float)
        X = np.zeros(n, dtype=float)
        pnl = np.zeros(n, dtype=float)

        # diagnostics
        lam_plus_arr = np.zeros(n, dtype=float)
        lam_minus_arr = np.zeros(n, dtype=float)
        f_a_arr = np.zeros(n - 1, dtype=float)
        f_b_arr = np.zeros(n - 1, dtype=float)

        H_a_arr = np.zeros(n - 1, dtype=int)   # ask hit -> we SELL
        H_b_arr = np.zeros(n - 1, dtype=int)   # bid hit -> we BUY

        # identity decomposition of wealth increments:
        # dW = trade_pnl_step + inv_pnl_step
        dW = np.zeros(n - 1, dtype=float)
        trade_pnl_step = np.zeros(n - 1, dtype=float)
        inv_pnl_step = np.zeros(n - 1, dtype=float)

        # diagnostic: executions vs next mid (adverse-selection-aware)
        adverse_trade_step = np.zeros(n - 1, dtype=float)

        q[0] = float(q0)
        X[0] = float(x0)
        pnl[0] = X[0] + q[0] * mid[0]

        eps = 1e-12

        for i in range(n - 1):
            t = float(t_secs[i])
            S = float(mid[i])
            dti = float(dt[i])

            # Intensities
            lp = float(lam_plus_path[i]) if lam_plus_path is not None else 0.0
            lm = float(lam_minus_path[i]) if lam_minus_path is not None else 0.0
            lam_plus_arr[i] = lp
            lam_minus_arr[i] = lm

            # quotes from policy
            b_i, a_i = self.policy(t=t, S=S, q=float(q[i]), lam_plus=lp, lam_minus=lm)
            if not np.isfinite(b_i) or not np.isfinite(a_i):
                b_i, a_i = S, S
            if a_i < b_i:
                a_i, b_i = b_i, a_i

            bid[i] = float(b_i)
            ask[i] = float(a_i)

            # If time doesn't advance, carry state
            if dti <= 0:
                q[i + 1] = q[i]
                X[i + 1] = X[i]
                pnl[i + 1] = X[i + 1] + q[i + 1] * mid[i + 1]
                continue

            delta_a = max(a_i - S, 0.0)
            delta_b = max(S - b_i, 0.0)

            # our execution intensities (ask, bid)
            lam_a, lam_b = self.arrival.intensities(
                t=t, delta_a=delta_a, delta_b=delta_b, lam_plus=lp, lam_minus=lm
            )
            lam_a = float(max(lam_a, 0.0))
            lam_b = float(max(lam_b, 0.0))

            # -------------------------
            # Aggressive market orders
            # -------------------------
            if N_plus_counts is None:
                N_plus = int(rng.poisson(max(lp * dti, 0.0)))
            else:
                N_plus = int(N_plus_counts[i])

            if N_minus_counts is None:
                N_minus = int(rng.poisson(max(lm * dti, 0.0)))
            else:
                N_minus = int(N_minus_counts[i])

            # -------------------------
            # Our fills = thinning of N^+, N^-
            # -------------------------
            f_a = float(np.clip(lam_a / (lp + eps), 0.0, 1.0)) if lp > 0 else 0.0
            f_b = float(np.clip(lam_b / (lm + eps), 0.0, 1.0)) if lm > 0 else 0.0
            f_a_arr[i] = f_a
            f_b_arr[i] = f_b

            H_a = int(rng.binomial(N_plus, f_a)) if N_plus > 0 else 0   # we SELL at ask
            H_b = int(rng.binomial(N_minus, f_b)) if N_minus > 0 else 0 # we BUY at bid

            # -------------------------
            # Marking for price moves
            # -------------------------
            total = lp + lm
            rho = (lp - lm) / total if total > eps else 0.0

            p_bar = float(np.clip(self.p_bar, 1e-6, 1.0))
            z = float(self.k_imp) * (rho - float(self.theta))
            p_plus  = p_bar / (1.0 + np.exp(-z))
            p_minus = p_bar / (1.0 + np.exp(+z))


            M_plus = int(rng.binomial(N_plus, p_plus)) if N_plus > 0 else 0
            M_minus = int(rng.binomial(N_minus, p_minus)) if N_minus > 0 else 0

            # -------------------------
            # price update
            # -------------------------
            if mid_replay is None:
                dw = rng.normal(0.0, np.sqrt(dti))
                mid[i + 1] = mid[i] + self.sigma * dw + self.tick_size * float(M_plus - M_minus)
            else:
                mid[i + 1] = float(mid_replay[i + 1])

            # -------------------------
            # inventory cap
            # -------------------------
            if np.isfinite(self.Q_max):
                tsz = float(trade_size)
                max_buy = max((float(self.Q_max) - float(q[i])) / tsz, 0.0)
                max_sell = max((float(q[i]) + float(self.Q_max)) / tsz, 0.0)
                H_b = int(min(H_b, np.floor(max_buy)))
                H_a = int(min(H_a, np.floor(max_sell)))

            # store fills AFTER cap
            H_a_arr[i] = H_a
            H_b_arr[i] = H_b

            # -------------------------
            # wealth decomposition (IDENTITY)
            # -------------------------
            # spread capture vs contemporaneous mid S
            trade_pnl_step[i] = float(trade_size) * (H_a * (a_i - S) + H_b * (S - b_i))

            # post-trade inventory
            q_post = q[i] + float(trade_size) * (H_b - H_a)

            # inventory MtM over [i, i+1]
            dS = mid[i + 1] - S
            inv_pnl_step[i] = q_post * dS

            # total wealth increment
            dW[i] = trade_pnl_step[i] + inv_pnl_step[i]

            # diagnostic: executions vs next mid (adverse-selection-aware)
            adverse_trade_step[i] = float(trade_size) * (
                H_a * (a_i - mid[i + 1]) + H_b * (mid[i + 1] - b_i)
            )

            # -------------------------
            # update inventory + cash (self-financing)
            # -------------------------
            q[i + 1] = q_post
            X[i + 1] = X[i] + float(trade_size) * (H_a * a_i - H_b * b_i)
            pnl[i + 1] = X[i + 1] + q[i + 1] * mid[i + 1]

        # last quotes (optional)
        tL = float(t_secs[-1])
        SL = float(mid[-1])
        lpL = float(lam_plus_path[-1]) if lam_plus_path is not None else 0.0
        lmL = float(lam_minus_path[-1]) if lam_minus_path is not None else 0.0
        bL, aL = self.policy(t=tL, S=SL, q=float(q[-1]), lam_plus=lpL, lam_minus=lmL)
        bid[-1], ask[-1] = float(bL), float(aL)

        return {
            "time": t_secs,
            "mid": mid,
            "bid": bid,
            "ask": ask,
            "q": q,
            "X": X,
            "pnl": pnl,
            "lam_plus": lam_plus_arr,
            "lam_minus": lam_minus_arr,
            "f_a": f_a_arr,
            "f_b": f_b_arr,
            "H_a": H_a_arr,
            "H_b": H_b_arr,
            "dW": dW,
            "trade_pnl_step": trade_pnl_step,
            "inv_pnl_step": inv_pnl_step,
            "adverse_trade_step": adverse_trade_step,
            "d_trade": trade_pnl_step,
            "d_mtm": inv_pnl_step,
        }



def simulate_mid_paths(mid0, t_grid, lam_plus, lam_minus, sigma, tick_size,
                       p_bar, k_imp, theta, n_paths=100, seed=0, debug=False):
    rng = np.random.default_rng(seed)
    t_grid = np.asarray(t_grid, float)
    lam_plus = np.asarray(lam_plus, float)
    lam_minus = np.asarray(lam_minus, float)

    dt = np.diff(t_grid)
    n = len(t_grid)

    S = np.empty((n_paths, n), dtype=float)
    S[:, 0] = float(mid0)

    # debug stats
    stats = []

    for p in range(n_paths):
        Np_tot = Nm_tot = Mp_tot = Mm_tot = 0

        for i, dti in enumerate(dt):
            lp, lm = float(lam_plus[i]), float(lam_minus[i])

            Np = rng.poisson(max(lp * dti, 0.0))
            Nm = rng.poisson(max(lm * dti, 0.0))
            Np_tot += int(Np)
            Nm_tot += int(Nm)

            rho = (lp - lm) / (lp + lm + 1e-12)

            p_bar_c = float(np.clip(p_bar, 1e-6, 1.0))
            z = float(k_imp) * (rho - float(theta))
            p_plus  = p_bar_c / (1.0 + np.exp(-z))
            p_minus = p_bar_c / (1.0 + np.exp(+z))


            Mp = rng.binomial(Np, p_plus) if Np > 0 else 0
            Mm = rng.binomial(Nm, p_minus) if Nm > 0 else 0
            Mp_tot += int(Mp)
            Mm_tot += int(Mm)

            dw = rng.normal(0.0, np.sqrt(dti))
            S[p, i+1] = S[p, i] + sigma * dw + tick_size * (Mp - Mm)

        if debug:
            dS = np.diff(S[p])
            nz = np.abs(dS[np.abs(dS) > 0])
            min_step = float(np.min(nz)) if nz.size else 0.0
            stats.append((Np_tot, Nm_tot, Mp_tot, Mm_tot, min_step))

    if debug and stats:
        Np_tot, Nm_tot, Mp_tot, Mm_tot, min_step = np.mean(np.array(stats), axis=0)
        print(f"[SIM DEBUG] avg Np={Np_tot:.1f}, Nm={Nm_tot:.1f}  |  avg Mp={Mp_tot:.1f}, Mm={Mm_tot:.1f}")
        print(f"[SIM DEBUG] avg net marked = {Mp_tot - Mm_tot:.2f}")
        print(f"[SIM DEBUG] min nonzero step (avg over paths) ≈ {min_step:.6g}")

    return S



from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

@dataclass
class FixedSkewFromMidPolicy:
    """
    Benchmark policy: quotes are a fixed asymmetric offset from mid.

    We parameterize via:
        delta0  = base half-spread (>=0)
        skew    = asymmetry (can be +/-)

    Quotes:
        bid = S - max(delta0 - skew, 0)
        ask = S + max(delta0 + skew, 0)

    Interpretation:
        skew > 0  -> "buy-tilted": bid closer, ask farther
        skew < 0  -> "sell-tilted": bid farther, ask closer
    """
    delta0: float
    skew: float = 0.0
    quote_tick: Optional[float] = None
    min_spread: Optional[float] = None  # default: quote_tick if provided

    def __call__(
        self,
        t: float,
        S: float,
        q: float,
        lam_plus: Optional[float] = None,
        lam_minus: Optional[float] = None,
    ) -> Tuple[float, float]:

        S = float(S)
        d0 = float(self.delta0)
        s  = float(self.skew)

        delta_b = max(d0 - s, 0.0)
        delta_a = max(d0 + s, 0.0)

        bid = S - delta_b
        ask = S + delta_a

        # Optional tick-rounding (recommended for thesis cleanliness)
        if self.quote_tick is not None and self.quote_tick > 0:
            qt = float(self.quote_tick)
            bid = np.floor(bid / qt) * qt
            ask = np.ceil(ask / qt) * qt

            # enforce non-negative (and preferably >= 1 tick) spread
            ms = qt if self.min_spread is None else float(self.min_spread)
            if ask < bid + ms:
                ask = bid + ms


        

        return float(bid), float(ask)
