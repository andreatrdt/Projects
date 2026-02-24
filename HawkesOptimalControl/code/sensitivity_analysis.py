#!/usr/bin/env python
# coding: utf-8

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import time

# --- Import your modules (same as main.py) ---
from OrderBook import OrderBook
from HawkesCalibrator import LOBHawkes, HawkesCalibrator
from control_frame import ControlBacktester
from microstructure_calibration import calibrate_fill_probability, calibrate_price_impact
from hjb_theta_solver import (
    HJBThetaIMEXSolver, HJBSolverParams, HJBQuotePolicyFromSolution,
    hjb_signature, estimate_sigma_arithmetic, estimate_quote_tick_from_bid_ask,
    estimate_tick_size_from_series, intensity_on_real_episode_2d,
    slice_event_counts, slice_mid_episode, HawkesBacktestModel, cache_load_or_compute
)
from event_cleaning import prepare_hawkes_events
from config import COMPONENTS_2D, MARKET_OPEN_SECONDS

# -----------------------------------------------------------------------------
# 1. SETUP & DATA LOADING (Reusing logic from main.py)
# -----------------------------------------------------------------------------
def t_rel(h: int, m: int, s: int = 0) -> float:
    return (h * 3600 + m * 60 + s) - MARKET_OPEN_SECONDS

# CONFIGURATION
# -----------------
TICKER = "AAPL"  # Or extract from filename if you prefer
MSG_FILE = "LOBSTER_sample/AAPL_2012-06-21_34200000_57600000_message_10.csv"
LOB_FILE = "LOBSTER_sample/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv"
T_EPISODE = 600.0   # Length of backtest episode (seconds)
T_START = t_rel(14, 25, 0) # Start time (seconds from open)
CACHE_DIR = "cache"

# HJB STATIC PARAMS
DT_HJB = 0.01
STORE_DT = 0.1

print(f"--- 1. Loading Data for {TICKER} ---")
ob = OrderBook(message_file=MSG_FILE, orderbook_file=LOB_FILE, ticker=TICKER)
df = ob.LimitOrderBook
# Filter for market hours
df = df[df["time"] >= MARKET_OPEN_SECONDS].copy().reset_index(drop=True)
df["time"] -= MARKET_OPEN_SECONDS
ob.LimitOrderBook = df
data = ob.LimitOrderBook

# Build Events
events, T_day = ob.to_hawkes_events_10d_robust(debug=False)
events, marks, meta = prepare_hawkes_events(events, COMPONENTS_2D, eps="auto")

# -----------------------------------------------------------------------------
# 2. LOAD/COMPUTE CALIBRATIONS (Hawkes, Sigma, Micro)
# -----------------------------------------------------------------------------
print("--- 2. Loading Calibrations ---")

# A. Hawkes (Load Cached)
hawkes_cache_path = os.path.join(CACHE_DIR, f"hawkes_2d_fit_results_{TICKER}.pkl")
if os.path.exists(hawkes_cache_path):
    print(f"   Loading Hawkes fit from {hawkes_cache_path}...")
    with open(hawkes_cache_path, "rb") as f:
        win_results = joblib.load(f)
else:
    raise FileNotFoundError(f"Please run main.py first to generate {hawkes_cache_path}")

# Pick fit for our time window
def pick_fit_for_time(win_results, t_start):
    for k, wr in win_results.items():
        if wr.t0 <= t_start < wr.t1:
            return wr.fit
    raise ValueError(f"t_start={t_start} not covered by fitted windows.")

fit_res = pick_fit_for_time(win_results, T_START)

# B. Sigma & Ticks
quote_tick = estimate_quote_tick_from_bid_ask(data["bid_price"].values, data["ask_price"].values)
mid_tick = estimate_tick_size_from_series(data["mid_price"].values)
sigma_hat = estimate_sigma_arithmetic(data["mid_price"].values, data["time"].values)
print(f"   Sigma: {sigma_hat:.4f}, Quote Tick: {quote_tick:.4f}, Mid Tick: {mid_tick:.4f}")

# C. Microstructure (Load Cached)
# Reusing the cache key logic from main.py approximately
micro_cache_name = f"micro_params_{TICKER}_qt{quote_tick:.6g}_mt{mid_tick:.6g}_L3_dmax0p05_h5.pkl"
micro_path = Path(CACHE_DIR) / micro_cache_name

if micro_path.exists():
    print(f"   Loading Microstructure from {micro_path}...")
    micro_params = joblib.load(micro_path)
else:
    # Fallback: simple calibration if cache missing (or run main.py)
    print("   [WARN] Microstructure cache not found. Running quick calibration...")
    A_fill, k_fill, d0_fill = calibrate_fill_probability(data, tick_size=quote_tick)
    p_bar, k_imp, theta_imp = calibrate_price_impact(None, data, tick_size=mid_tick)
    micro_params = {
        "A_fill": A_fill, "k_fill": k_fill, "delta0_fill": d0_fill,
        "p_bar": p_bar, "k_imp": k_imp, "theta_imp": theta_imp
    }

# Unpack
A_fill, k_fill, delta0_fill = micro_params["A_fill"], micro_params["k_fill"], micro_params["delta0_fill"]
p_bar, k_imp, theta_imp = micro_params["p_bar"], micro_params["k_imp"], micro_params["theta_imp"]

# -----------------------------------------------------------------------------
# 3. PREPARE REAL MARKET EPISODE (Common for all backtests)
# -----------------------------------------------------------------------------
print(f"--- 3. Extracting Real Episode [{T_START:.0f}, {T_START+T_EPISODE:.0f}) ---")
# This ensures we backtest every Gamma against the EXACT same market history
n_grid = int(T_EPISODE / 0.002) + 1  # 2ms step for backtest

t_grid_real, lam_plus_real, lam_minus_real = intensity_on_real_episode_2d(
    events=events, fit_res=fit_res, t_start=T_START, T=T_EPISODE, n_grid_points=n_grid
)
N_plus_counts, N_minus_counts = slice_event_counts(events, T_START, T_EPISODE, t_grid_real)
mid_real_grid = slice_mid_episode(data, T_START, T_EPISODE, t_grid_real)



# -----------------------------------------------------------------------------
# 4. SENSITIVITY LOOP (The Core Task)
# -----------------------------------------------------------------------------
print("\n--- 4. Running Sensitivity Analysis (Efficient Frontier) ---")

# Define Gamma Range (Logarithmic scale usually captures the behavior best)
gamma_values = [0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0]
results = []

# Prepare Helper for HJB matrix construction
idx_minus = COMPONENTS_2D.index("E_b")
idx_plus  = COMPONENTS_2D.index("E_a")
mu_plus, mu_minus = fit_res.mu[idx_plus], fit_res.mu[idx_minus]
beta_plus, beta_minus = fit_res.beta[idx_plus], fit_res.beta[idx_minus]
A_fit = fit_res.A
A_pm = np.array([
    [A_fit[idx_plus,  idx_plus],  A_fit[idx_plus,  idx_minus]],
    [A_fit[idx_minus, idx_plus],  A_fit[idx_minus, idx_minus]],
])

def calculate_max_drawdown(pnl_series):
    cumulative = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak)
    return drawdown.min()

for g in tqdm(gamma_values, desc="Testing Gammas"):
    # A. Configure Solver for this Gamma
    # Note: We keep eta (inventory penalty) low or zero to isolate Gamma's effect on risk
    params = HJBSolverParams(
        T=T_EPISODE,
        n_time=int(T_EPISODE / DT_HJB),
        Q=20,
        q0=1,
        gamma=float(g),       # <--- VARYING PARAMETER
        eta=0.01,             # Small running penalty
        eta_T=0.02,
        eta_run=1e-4,
        sigma=sigma_hat,
        kappa_jump=mid_tick,
        quote_tick=quote_tick,
        delta_min=quote_tick,
        delta_step=quote_tick,
        delta_max=0.20,
        delta0_fill_a=delta0_fill, delta0_fill_b=delta0_fill,
        A_fill_a=A_fill, A_fill_b=A_fill,
        k_fill_a=k_fill, k_fill_b=k_fill,
        p_bar=p_bar, k_imp=k_imp, theta_imp=theta_imp,
        M_plus=45, M_minus=45,
        store_every=int(round(STORE_DT / DT_HJB)),
    )
    
    # B. Solve HJB (with Caching)
    # The signature includes 'gamma', so this will cache separately for each gamma
    solver = HJBThetaIMEXSolver(mu_plus, mu_minus, A_pm, beta_plus, beta_minus, params)
    
    # Define a local load function to handle caching inside the loop
    cache_file = Path(CACHE_DIR) / f"hjb_sol_{TICKER}_{hjb_signature(params)}.joblib"
    if cache_file.exists():
        sol = joblib.load(cache_file)
    else:
        sol = solver.solve()
        joblib.dump(sol, cache_file, compress=3)
        
    policy = HJBQuotePolicyFromSolution(sol)

    # C. Run Backtest (Real Data)
    backtester = ControlBacktester(
        policy=policy,
        arrival=HawkesBacktestModel(A_fill=A_fill, k_fill=k_fill, delta0_fill=delta0_fill),
        Q_max=params.Q,
        tick_size=mid_tick,
        sigma=params.sigma,
        p_bar=params.p_bar,
        k_imp=params.k_imp,
        theta=params.theta_imp,
    )

    res = backtester.run(
        mid=np.full_like(t_grid_real, mid_real_grid[0]), # Placeholder for mid container
        mid_replay=mid_real_grid,    # <--- REPLAY REAL PRICE
        t_secs=t_grid_real,
        lam_plus_path=lam_plus_real, # <--- REPLAY REAL INTENSITY
        lam_minus_path=lam_minus_real,
        N_plus_counts=N_plus_counts, # <--- REPLAY REAL MARKET ORDERS
        N_minus_counts=N_minus_counts,
        seed=42
    )

    # D. Collect Metrics
    final_pnl = res["total_pnl"].iloc[-1]
        
    # Calcolo volatilità PnL (step-by-step)
    pnl_changes = res["total_pnl"].diff().fillna(0)
    pnl_vol = pnl_changes.std()
    
    # Calcolo Inventario Medio (assoluto) e Volatilità Inventario
    inv_mean_abs = res["q"].abs().mean()
    inv_std = res["q"].std()
    
    # *** NUOVO: Max Drawdown ***
    # Se 'pnl_step' esiste usiamo quello, altrimenti diff di total_pnl
    if 'pnl_step' in res.columns:
        mdd = calculate_max_drawdown(res['pnl_step'])
    else:
        mdd = calculate_max_drawdown(pnl_changes)

    results.append({
        "gamma": g,
        "Total_PnL": final_pnl,
        "PnL_Vol": pnl_vol,
        "Inventory_Mean_Abs": inv_mean_abs,
        "Inventory_Std": inv_std,
        "Max_Drawdown": mdd,   # <--- La metrica critica per la tesi
        "Sharpe": (final_pnl / pnl_vol) if pnl_vol > 1e-6 else 0.0
    })

# -----------------------------------------------------------------------------
# 5. PLOTTING THE EFFICIENT FRONTIER
# -----------------------------------------------------------------------------
df_res = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
# Scatter plot colorato dalla Gamma
sc = plt.scatter(df_res["Max_Drawdown"], df_res["Total_PnL"], 
                 c=df_res["gamma"], cmap='viridis', s=100, edgecolors='k')

plt.colorbar(sc, label=r"Avversione al Rischio $\gamma$")
plt.xlabel("Maximum Drawdown ($)")
plt.ylabel("Total Profit ($)")
plt.title(f"Risk-Reward Tradeoff: Drawdown vs PnL ({TICKER})")
plt.grid(True, linestyle='--', alpha=0.5)

# Annotazioni
for i, row in df_res.iterrows():
    plt.annotate(f"$\\gamma={row['gamma']:.1f}$", 
                 (row["Max_Drawdown"], row["Total_PnL"]),
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.savefig("sensitivity_drawdown.png")
plt.show()

# Plot 1: Efficient Frontier (Risk vs Reward)
plt.figure(figsize=(10, 6))
# X-axis: PnL Volatility (Risk), Y-axis: Total PnL (Reward)
plt.plot(df_res["PnL_Vol"], df_res["Total_PnL"], 'o-', markersize=8, linewidth=2, color='b', label='HJB Frontier')

for i, row in df_res.iterrows():
    # Annotate points
    plt.annotate(f"$\gamma={row['gamma']}$", 
                 (row["PnL_Vol"], row["Total_PnL"]), 
                 textcoords="offset points", xytext=(5,5), ha='left')

plt.title(f"Efficient Frontier: HJB Market Making ({TICKER})")
plt.xlabel("Risk (Std Dev of PnL Changes)")
plt.ylabel("Reward (Total PnL)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("efficient_frontier.png")
plt.show()

# Plot 2: Gamma vs Inventory Risk (The Mechanism)
plt.figure(figsize=(10, 6))
plt.plot(df_res["gamma"], df_res["Inventory_Std"], 's-', color='darkorange', linewidth=2)
plt.xscale('log') # Log scale for gamma usually looks better
plt.title("Impact of Risk Aversion ($\gamma$) on Inventory Holding")
plt.xlabel("Risk Aversion $\gamma$ (Log Scale)")
plt.ylabel("Inventory Std Deviation ($q$)")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("gamma_mechanism.png")
plt.show()

print("\n[DONE] Plots saved to efficient_frontier.png and gamma_mechanism.png")