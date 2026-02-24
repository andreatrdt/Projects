# microstructure_calibration.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize


# =============================================================================
# Fill probability calibration
# =============================================================================
def calibrate_fill_probability(
    lob_data: pd.DataFrame,
    tick_size: float = 0.01,
    max_levels: int = 10,
    min_events: int = 2000,
    delta_max_fit: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Calibrate fill-prob proxy curve (SAME MODEL):
        p_fill(delta) ≈ min(1, A * exp(-k * max(delta - delta0, 0)))

    Returns:
        A_hat         in (0,1]
        k_hat_price   (per price unit)
        delta0_fill   (typical touch distance ~ spread/2 for mid=(bid+ask)/2)

    Protocol fixes (no model change):
    - Bin (delta, pfill_proxy) and fit to bin means (more stable)
    - Weight bins by sqrt(count) so sparse tail bins don't dominate
    - Robust loss (soft_l1) to reduce outlier influence
    - Fit only up to delta_max_fit (recommended = max quoting offset);
      if delta_max_fit is None, we auto-set it to the 90th percentile of delta samples
    """
    df = lob_data

    ask1_col = "ask_price_1" if "ask_price_1" in df.columns else ("ask_price" if "ask_price" in df.columns else None)
    bid1_col = "bid_price_1" if "bid_price_1" in df.columns else ("bid_price" if "bid_price" in df.columns else None)

    if ask1_col is None or bid1_col is None:
        raise KeyError("[Microstructure] Missing best bid/ask columns.")

    needed = {"mid_price", "lob_action", "execution", "lob_size", ask1_col, bid1_col}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"[Microstructure] Missing columns for fill calibration: {missing}")

    # levels available
    L_avail = []
    for L in range(1, max_levels + 1):
        colsL = {f"ask_price_{L}", f"ask_size_{L}", f"bid_price_{L}", f"bid_size_{L}"}
        if colsL.issubset(df.columns):
            L_avail.append(L)

    if not L_avail:
        if {"ask_size", "bid_size"}.issubset(df.columns):
            L_avail = [1]
        else:
            raise KeyError("[Microstructure] No usable LOB levels found.")

    # select executions
    exec_mask = df["lob_action"].isin([4, 5])
    dfe = df.loc[exec_mask, ["mid_price", "execution", "lob_size", ask1_col, bid1_col]].copy()
    if dfe.shape[0] < min_events:
        print("[Microstructure] Too few execution events. Fallback.")
        return 1.0, 10.0, 0.5 * float(tick_size)

    mid = dfe["mid_price"].to_numpy(float)
    px  = dfe["execution"].to_numpy(float)
    V   = dfe["lob_size"].to_numpy(float)
    ask1 = dfe[ask1_col].to_numpy(float)
    bid1 = dfe[bid1_col].to_numpy(float)

    eps_px = 0.1 * float(tick_size) + 1e-12
    is_buy  = np.isfinite(px) & np.isfinite(ask1) & (px >= ask1 - eps_px)
    is_sell = np.isfinite(px) & np.isfinite(bid1) & (px <= bid1 + eps_px)

    keep = np.isfinite(mid) & np.isfinite(V) & (V > 0) & (is_buy | is_sell)
    idx = dfe.index.to_numpy()[keep]
    if idx.size < min_events:
        print("[Microstructure] Too few clean execution events after filtering. Fallback.")
        return 1.0, 10.0, 0.5 * float(tick_size)

    deltas = []
    pfill  = []
    delta0_samples = []

    for i in idx:
        m = float(df.at[i, "mid_price"])
        v = float(df.at[i, "lob_size"])

        buy = bool(
            np.isfinite(df.at[i, "execution"]) and np.isfinite(df.at[i, ask1_col])
            and df.at[i, "execution"] >= df.at[i, ask1_col] - eps_px
        )

        cum = 0.0
        if buy:
            # buy MO consumes ASK side
            for L in L_avail:
                p_col = f"ask_price_{L}" if f"ask_price_{L}" in df.columns else ask1_col
                q_col = f"ask_size_{L}"  if f"ask_size_{L}"  in df.columns else "ask_size"
                if p_col not in df.columns or q_col not in df.columns:
                    break

                pL = float(df.at[i, p_col])
                qL = float(df.at[i, q_col])
                if not (np.isfinite(pL) and np.isfinite(qL) and qL > 0):
                    break

                delta = pL - m
                consume = v - cum
                pf = 0.0 if consume <= 0 else float(np.clip(consume / qL, 0.0, 1.0))

                if delta >= 0 and np.isfinite(delta):
                    deltas.append(delta)
                    pfill.append(pf)
                    if L == 1:
                        delta0_samples.append(delta)

                cum += qL
        else:
            # sell MO consumes BID side
            for L in L_avail:
                p_col = f"bid_price_{L}" if f"bid_price_{L}" in df.columns else bid1_col
                q_col = f"bid_size_{L}"  if f"bid_size_{L}"  in df.columns else "bid_size"
                if p_col not in df.columns or q_col not in df.columns:
                    break

                pL = float(df.at[i, p_col])
                qL = float(df.at[i, q_col])
                if not (np.isfinite(pL) and np.isfinite(qL) and qL > 0):
                    break

                delta = m - pL
                consume = v - cum
                pf = 0.0 if consume <= 0 else float(np.clip(consume / qL, 0.0, 1.0))

                if delta >= 0 and np.isfinite(delta):
                    deltas.append(delta)
                    pfill.append(pf)
                    if L == 1:
                        delta0_samples.append(delta)

                cum += qL

    deltas = np.asarray(deltas, dtype=float)
    pfill  = np.asarray(pfill, dtype=float)

    msk = np.isfinite(deltas) & np.isfinite(pfill) & (deltas >= 0.0)
    deltas = deltas[msk]
    pfill  = pfill[msk]

    if deltas.size < 5000:
        print("[Microstructure] Too few (delta, pfill) samples. Fallback.")
        return 1.0, 10.0, 0.5 * float(tick_size)

    # delta0_fill: robust touch distance estimate (median of L1 deltas)
    if len(delta0_samples) > 100:
        delta0_fill = float(np.nanmedian(np.asarray(delta0_samples, dtype=float)))
    else:
        delta0_fill = 0.5 * float(tick_size)

    # ------------------------------
    # FITTING REFACTOR (same model)
    # ------------------------------
    # Fit cutoff: if not provided, auto-cut at 90th percentile to avoid tail domination
    if delta_max_fit is None:
        delta_max_fit_used = float(np.quantile(deltas, 0.90))
    else:
        delta_max_fit_used = float(delta_max_fit)

    # Keep only samples in [0, delta_max_fit_used] for calibration objective
    mfit = (deltas >= 0.0) & (deltas <= delta_max_fit_used)
    d_fit = deltas[mfit]
    p_fit = pfill[mfit]

    if d_fit.size < 2000:
        print("[Microstructure] Too few samples after fit cutoff. Fallback.")
        return 1.0, 10.0, delta0_fill

    # Bin by delta (not d_eff) to stabilize the objective
    nbins = 30
    min_bin = 50

    edges = np.linspace(0.0, delta_max_fit_used, nbins + 1)
    bin_id = np.digitize(d_fit, edges) - 1
    ok = (bin_id >= 0) & (bin_id < nbins)
    bin_id = bin_id[ok]
    p_fit = p_fit[ok]

    n_bin = np.bincount(bin_id, minlength=nbins).astype(float)
    s_bin = np.bincount(bin_id, weights=p_fit, minlength=nbins).astype(float)
    keep_bins = n_bin >= float(min_bin)

    if np.sum(keep_bins) < 6:
        print("[Microstructure] Too few populated bins. Fallback.")
        return 1.0, 10.0, delta0_fill

    centers = 0.5 * (edges[:-1] + edges[1:])
    d_cent = centers[keep_bins]
    p_hat  = (s_bin / np.maximum(n_bin, 1.0))[keep_bins]
    w      = np.sqrt(n_bin[keep_bins])  # key: tail bins do not dominate

    # Model evaluated at bin centers
    def pred(A, k):
        deff = np.maximum(d_cent - delta0_fill, 0.0)
        return np.minimum(1.0, A * np.exp(-k * deff))

    # Robust bounded least squares
    from scipy.optimize import least_squares

    # init (avoid log-regression bias)
    A0 = float(np.clip(np.percentile(p_hat, 90), 1e-4, 0.999))
    deff_cent = np.maximum(d_cent - delta0_fill, 0.0)
    span = float(np.quantile(deff_cent, 0.80) - np.quantile(deff_cent, 0.20))
    k0 = float(5.0 / (span + 1e-8))
    x0 = np.array([A0, k0], dtype=float)

    lo = np.array([1e-6, 0.0], dtype=float)
    hi = np.array([0.999999, 5000.0], dtype=float)

    def resid(x):
        A, k = x
        return w * (pred(A, k) - p_hat)

    opt = least_squares(
        resid,
        x0=x0,
        bounds=(lo, hi),
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=5000,
    )

    if not opt.success:
        print(f"[Microstructure] Fill fit failed ({opt.message}). Using init.")
        A_hat, k_hat_price = float(A0), float(k0)
    else:
        A_hat, k_hat_price = map(float, opt.x)

    k_hat_tick = k_hat_price * float(tick_size)

    print(
        f"[Microstructure] Fill Params (fit): "
        f"A={A_hat:.4f}, k_tick={k_hat_tick:.4f} (per tick), k_price={k_hat_price:.4f} (per price unit), "
        f"delta0={delta0_fill:.6f}, fit_cutoff={delta_max_fit_used:.6f}, bins_kept={int(np.sum(keep_bins))}"
    )

    return A_hat, k_hat_price, delta0_fill

# =============================================================================
# Event-level price impact calibration
# =============================================================================

def calibrate_price_impact(
    events_df,                      # kept for your call signature compatibility (unused)
    ticker_data: pd.DataFrame,
    tick_size: float = 0.005,
    horizon_events: int = 1,
) -> Tuple[float, float, float]:
    """
    Event-level calibration for marking probability (thesis form).

    Uses only executions (lob_action in {4,5}).
    Aggressor side:
        s = +1 for buy market orders (hit ask)  -> lob_dir = -1 in LOBSTER convention
        s = -1 for sell market orders (hit bid) -> lob_dir = +1

    Label y uses forward horizon in *event rows*:
        dmid_h = mid[t+h] - mid[t]
        y = 1 if (s=+1 and dmid_h >= +tick) or (s=-1 and dmid_h <= -tick)

    Fits:
        P(y=1 | rho, s) = p_bar / (1 + exp(-s*k*(rho-theta)))
    """

    df = ticker_data.copy()

    needed = {"lob_action", "lob_dir", "mid_price"}
    if not needed.issubset(df.columns):
        raise KeyError(f"[Microstructure] Missing columns for event-level impact: {needed - set(df.columns)}")

    # positional exec indices (robust even if df.index is not 0..n-1)
    is_exec = df["lob_action"].isin([4, 5]).to_numpy()
    exec_pos_all = np.flatnonzero(is_exec)

    if exec_pos_all.size < 500:
        print("[Microstructure] Too few execution events for robust impact calibration. Fallback.")
        return 0.05, 0.0, 0.0

    lob_dir = df["lob_dir"].to_numpy(dtype=float)
    s_all = -lob_dir[exec_pos_all]  # +1 buy MO, -1 sell MO

    ok = np.isfinite(s_all) & (np.abs(s_all) == 1.0)
    exec_pos = exec_pos_all[ok]
    s = s_all[ok]

    if exec_pos.size < 500:
        print("[Microstructure] Too few clean execution events after filtering. Fallback.")
        return 0.05, 0.0, 0.0

    # rho feature
    if "rho_5L" in df.columns and df["rho_5L"].notna().any():
        rho = df["rho_5L"].to_numpy(dtype=float)[exec_pos]
        src = "rho_5L"
    else:
        # fallback: L1 imbalance if sizes exist
        # accept either bid_size/ask_size or bid_size_1/ask_size_1
        bs = "bid_size_1" if "bid_size_1" in df.columns else ("bid_size" if "bid_size" in df.columns else None)
        aS = "ask_size_1" if "ask_size_1" in df.columns else ("ask_size" if "ask_size" in df.columns else None)
        if bs is None or aS is None:
            print("[Microstructure] No rho_5L and no bid/ask sizes. Disabling imbalance effect.")
            return 0.05, 0.0, 0.0
        vb = df[bs].to_numpy(dtype=float)[exec_pos]
        va = df[aS].to_numpy(dtype=float)[exec_pos]
        rho = (vb - va) / (vb + va + 1e-12)
        src = "L1 volume imbalance"

    rho = np.clip(rho, -1.0, 1.0)

    mid = df["mid_price"].to_numpy(dtype=float)

    # tick_size guard (if you accidentally pass 0.0 you make "directional moves" too easy)
    tick_size = float(tick_size)
    if not np.isfinite(tick_size) or tick_size <= 0:
        diffs = np.abs(np.diff(mid))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        tick_size = float(np.percentile(diffs, 5)) if diffs.size else 1e-6

    h = int(max(1, horizon_events))

    # dmid1 at event (backward diff)
    dmid1 = np.empty_like(mid)
    dmid1[:] = np.nan
    dmid1[1:] = mid[1:] - mid[:-1]
    dmid1_exec = dmid1[exec_pos]

    # dmid_h forward horizon
    dmid_h = np.empty_like(mid)
    dmid_h[:] = np.nan
    if h < mid.size:
        dmid_h[:-h] = mid[h:] - mid[:-h]
    dmid_h_exec = dmid_h[exec_pos]

    # label
    tol = tick_size - 1e-12
    y = ((s > 0) & (dmid_h_exec >= tol)) | ((s < 0) & (dmid_h_exec <= -tol))
    y = y.astype(float)

    # filter finite
    m = np.isfinite(rho) & np.isfinite(s) & np.isfinite(y) & np.isfinite(dmid_h_exec)
    rho = rho[m]
    s = s[m]
    y = y[m]
    dmid1_exec = dmid1_exec[m]
    dmid_h_exec = dmid_h_exec[m]

    if y.size < 500:
        print("[Microstructure] Too few clean samples after filtering. Fallback.")
        return 0.05, 0.0, 0.0

    # diagnostics
    p_emp = float(np.mean(y))
    p_buy = float(np.mean(y[s > 0])) if np.any(s > 0) else float("nan")
    p_sell = float(np.mean(y[s < 0])) if np.any(s < 0) else float("nan")

    print(f"[Microstructure] Event-level impact: samples={y.size}, rho_source={src}, horizon_events={h}")
    print(f"  Sanity mean dmid1 at exec: buy(s=+1) = {np.nanmean(dmid1_exec[s>0]) if np.any(s>0) else np.nan}  "
          f"sell(s=-1) = {np.nanmean(dmid1_exec[s<0]) if np.any(s<0) else np.nan}")
    print(f"  Sanity mean dmid_h:        buy(s=+1) = {np.nanmean(dmid_h_exec[s>0]) if np.any(s>0) else np.nan}  "
          f"sell(s=-1) = {np.nanmean(dmid_h_exec[s<0]) if np.any(s<0) else np.nan}")
    print(f"[Microstructure] Directional move rate: overall={p_emp:.3f}, buy(up)={p_buy:.3f}, sell(down)={p_sell:.3f}")

    # If too few directional moves, rho can't help: fall back to constant marking (k=0)
    if float(np.sum(y)) < 50:
        p_bar_const = float(np.clip(2.0 * p_emp, 1e-6, 0.9))
        print("[Microstructure] Too few directional moves -> using constant marking (k=0, theta=0).")
        return p_bar_const, 0.0, 0.0

    eps = 1e-9

    def nll(params):
        p_bar, k, theta = params
        if not (eps < p_bar < 1.0 - eps):
            return 1e18
        if k <= 0:
            return 1e18
        if not (-1.0 <= theta <= 1.0):
            return 1e18

        z = s * k * (rho - theta)
        p = p_bar / (1.0 + np.exp(-z))
        p = np.clip(p, eps, 1.0 - eps)

        return -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    p0 = float(np.clip(2.0 * p_emp, 0.001, 0.5))
    x0 = np.array([p0, 5.0, 0.0], dtype=float)

    bounds = [(0.001, 0.9), (0.001, 100.0), (-1.0, 1.0)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)

    if not res.success:
        print(f"[Microstructure] MLE failed ({res.message}). Using fallback constants.")
        return p0, 0.0, 0.0

    p_bar_hat, k_hat, theta_hat = map(float, res.x)

    stuck = []
    if abs(k_hat - bounds[1][0]) < 1e-6 or abs(k_hat - bounds[1][1]) < 1e-6:
        stuck.append("k")
    if abs(theta_hat - bounds[2][0]) < 1e-6 or abs(theta_hat - bounds[2][1]) < 1e-6:
        stuck.append("theta")
    if abs(p_bar_hat - bounds[0][0]) < 1e-6 or abs(p_bar_hat - bounds[0][1]) < 1e-6:
        stuck.append("p_bar")
    if stuck:
        print(f"[Microstructure] Warning: parameters on bounds -> {stuck} (rho may have weak predictive power).")

    print(f"[Microstructure] Impact Params (event-level MLE): p_bar={p_bar_hat:.3f}, k={k_hat:.6f}, theta={theta_hat:.3f}")
    return p_bar_hat, k_hat, theta_hat
