import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
from scipy.stats import expon
import math
from datetime import datetime, time

from config import COMPONENTS, COMPONENTS_4D, COMPONENTS_2D
from HawkesCalibrator import HawkesCalibrator, LOBHawkes

class HawkesPlotter:
    def __init__(self, comp_labels=None):
        self.comp_labels = comp_labels

    # =========================================================================
    #  PART 1: MARKET DATA VISUALIZATION (Human Readable Time 'ts')
    # =========================================================================

    def plot_price_action(self, data, title="Bid-Ask & Executions"):
        """
        Plotta l'andamento intero della giornata: Bid, Ask ed Esecuzioni.
        Sostituisce: ob.plot_bid_ask_exec()
        """
        df = data.copy()
        # Assicura datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ts']):
            df['ts'] = pd.to_datetime(df['ts'])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot Bid e Ask (Linee sottili)
        ax.plot(df['ts'], df['bid_price'], label='Best Bid', color='green', linewidth=0.8, alpha=0.8)
        ax.plot(df['ts'], df['ask_price'], label='Best Ask', color='red', linewidth=0.8, alpha=0.8)
        
        # Plot Esecuzioni (Pallini)
        # Filtriamo dove c'è stata una esecuzione reale (execution price > 0 o presente)
        if 'execution' in df.columns:
            execs = df[df['execution'].notna() & (df['execution'] > 0)]
            # Distinguiamo Buy/Sell se possibile, altrimenti tutti neri
            # Se 'lob_action' o 'operation' identifica buy/sell market order
            ax.scatter(execs['ts'], execs['execution'], color='black', s=10, label='Executions', zorder=5)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_price_zoom(self, data, start_time, end_time, title="Microstructure Zoom"):
        """
        Zoom su un intervallo specifico (es. 12:25:00 - 12:25:10).
        Sostituisce: ob.cut_and_plot(...)
        
        start_time, end_time: possono essere oggetti datetime.time (es. time(10,0))
                              o stringhe "HH:MM:SS".
        """
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ts']):
            df['ts'] = pd.to_datetime(df['ts'])
        
        # Gestione input time (se passo time(10,30), devo combinarlo con la data del df)
        base_date = df['ts'].iloc[0].date()
        
        if isinstance(start_time, time):
            start_dt = datetime.combine(base_date, start_time)
        elif isinstance(start_time, str):
            t = datetime.strptime(start_time, "%H:%M:%S").time()
            start_dt = datetime.combine(base_date, t)
        else:
            start_dt = start_time

        if isinstance(end_time, time):
            end_dt = datetime.combine(base_date, end_time)
        elif isinstance(end_time, str):
            t = datetime.strptime(end_time, "%H:%M:%S").time()
            end_dt = datetime.combine(base_date, t)
        else:
            end_dt = end_time

        # Filtro
        mask = (df['ts'] >= start_dt) & (df['ts'] <= end_dt)
        sub = df.loc[mask]
        
        if sub.empty:
            print(f"Warning: No data found between {start_time} and {end_time}")
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Step plot è meglio per la microstruttura (il prezzo rimane costante fino al cambio)
        ax.step(sub['ts'], sub['ask_price'], where='post', label='Ask', color='red', linewidth=1.5)
        ax.step(sub['ts'], sub['bid_price'], where='post', label='Bid', color='green', linewidth=1.5)
        
        # Esecuzioni
        if 'execution' in sub.columns:
            ex = sub[sub['execution'].notna() & (sub['execution'] > 0)]
            ax.scatter(ex['ts'], ex['execution'], color='blue', s=40, marker='x', label='Trade', zorder=5)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.title(f"{title} ({start_time} - {end_time})")
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_candles(self, data, resample_rule='1T', title="Order Book Candles"):
        """
        Plotta le candele OHLC.
        Sostituisce: ob.plot_execution_candles()
        """
        df_plot = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_plot['ts']):
            df_plot['ts'] = pd.to_datetime(df_plot['ts'])
        
        if df_plot.index.name != 'ts':
            df_plot.set_index('ts', inplace=True)
            
        # Usa Mid-Price o Execution
        col_to_use = 'mid_price' if 'mid_price' in df_plot.columns else 'ask_price'
        ohlc = df_plot[col_to_use].resample(resample_rule).ohlc()
        ohlc.dropna(inplace=True)

        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Width dinamica
        freq_map = {'1T': 1, '5T': 5, '15T': 15, '1H': 60, '1S': 1/60}
        minutes = freq_map.get(resample_rule, 1)
        width = (minutes / (24 * 60)) * 0.8 
        
        up = ohlc[ohlc.close >= ohlc.open]
        down = ohlc[ohlc.close < ohlc.open]
        
        ax.bar(up.index, up.close - up.open, width, bottom=up.open, color='green', alpha=0.7, edgecolor='green')
        ax.vlines(up.index, up.low, up.high, color='green', linewidth=1)
        
        ax.bar(down.index, down.close - down.open, width, bottom=down.open, color='red', alpha=0.7, edgecolor='red')
        ax.vlines(down.index, down.low, down.high, color='red', linewidth=1)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        
        plt.xticks(rotation=45)
        plt.title(f"{title} ({resample_rule})")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

    # =========================================================================
    #  PART 2: HAWKES MODEL VISUALIZATION (Mathematical Time 'time')
    # =========================================================================
    def hawkes_rate(self, t, d, events, mu, A, beta):
        """Compute Hawkes intensity for component d at time t.

        Convention used across this codebase:
            phi_{d,j}(u) = A[d,j] * exp(-beta[j] * u),  u>0

        Hence A is the jump size and the branching matrix is K = A / beta.
        """
        events = np.asarray(events)
        times = events[:, 0]
        comps = events[:, 1].astype(int)
        mask = times < t
        times = times[mask]
        comps = comps[mask]

        lam = float(mu[d])
        for s, j in zip(times, comps):
            bj = beta[j]
            lam += A[d, j] * np.exp(-bj * (t - s))
        return lam


    def plot_diagnostics(self, cal, fit, max_qq=None, comp_names=None):
        mu, A, beta, K = fit.mu, fit.A, fit.beta, fit.K
        mu = np.asarray(mu, dtype=float)
        K = np.asarray(K, dtype=float)
        dim = mu.size

        if comp_names is None:
            if self.comp_labels is not None:
                comp_names = list(self.comp_labels)[:dim]
            else:
                if len(COMPONENTS) >= dim:
                    comp_names = COMPONENTS[:dim]
                else:
                    comp_names = list(cal.events.keys())[:dim]

        # 1. Rates
        counts = np.array([cal.events[k].size for k in comp_names], dtype=float)
        emp_rate = counts / cal.T
        I = np.eye(dim)
        try:
            theory_rate = np.linalg.solve(I - K, mu)
        except np.linalg.LinAlgError:
            theory_rate = np.full(dim, np.nan)

        x = np.arange(dim)
        width = 0.35
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(x - width/2, emp_rate, width, label='Empirical')
        ax1.bar(x + width/2, theory_rate, width, label='Theoretical')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comp_names)
        ax1.set_title("Event Rates")
        ax1.legend()
        plt.tight_layout()

        # 2. QQ Plots
        rows = []
        Zs = {}
        for comp in comp_names:
            z = cal.time_rescaled_residuals(mu, A, beta, comp)
            Zs[comp] = z
            if z.size:
                rows.append({"comp": comp, "mean(Z)": z.mean(), "n": z.size})
        
        n_cols = min(dim, 4)
        n_rows = math.ceil(dim / n_cols)
        fig2, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        axes = np.atleast_1d(axes).ravel()

        for i, comp in enumerate(comp_names):
            ax = axes[i]
            z = Zs[comp]
            if z.size >= 4:
                z_sorted = np.sort(z)
                n = z_sorted.size
                p = (np.arange(1, n+1) - 0.5)/n
                theo = expon.ppf(p, scale=1.0)
                ax.scatter(theo, z_sorted, s=5, alpha=0.6)
                mx = max(theo.max(), z_sorted.max())
                if max_qq: mx = min(mx, max_qq)
                ax.plot([0, mx], [0, mx], 'r-', lw=1)
                if max_qq:
                    ax.set_xlim(0, max_qq)
                    ax.set_ylim(0, max_qq)
            ax.set_title(f"QQ {comp}")
        plt.tight_layout()
        plt.show()

        
    def plot_frame_events_and_intensity_2D(
        self,
        lob4: LOBHawkes,
        win_results,
        t0_abs: float,
        t1_abs: float,
        events_sim_2D=None,
        seed: int = 42,
        n_grid: int = 2000,
        k_window: int = None,
        *,
        max_events: int = 1500,          # per componente e per sorgente (Real/Sim)
        intensity_max_pts: int = 5000,   # decima curva per velocità/leggibilità
        event_height_real: float = 0.18, # altezza tacche (frazione di max intensità)
        event_height_sim: float = 0.10,
        alpha_real: float = 0.45,
        alpha_sim: float = 0.30,
        show_clock: bool = True,
        rotate_xticks: int = 45,
        show_seconds: bool = False,
    ):
        """
        Plot chiaro: intensità come linea + eventi come rug in basso (real vs sim).
        t0_abs, t1_abs sono *secondi da open* (assoluti).
        """

        # ----------------------------
        # 1) trova finestra che contiene [t0_abs, t1_abs]
        # ----------------------------
        items = list(win_results.items()) if isinstance(win_results, dict) else list(enumerate(win_results))

        if k_window is None:
            for k, w in items:
                if (t0_abs >= w.t0) and (t1_abs <= w.t1):
                    k_window = k
                    break
            if k_window is None:
                raise ValueError("Frame not inside any window.")
        w = win_results[k_window]

        T_win = w.t1 - w.t0
        t0_rel = t0_abs - w.t0
        t1_rel = t1_abs - w.t0

        # ----------------------------
        # 2) eventi reali nella finestra (relativi per calcolo intensità)
        # ----------------------------
        times_win = []
        comps_win = []
        for j, name in enumerate(COMPONENTS_2D):
            t_comp = lob4.events[name]
            mask = (t_comp >= w.t0) & (t_comp < w.t1)
            t_rel = t_comp[mask] - w.t0
            times_win.append(t_rel)
            comps_win.append(np.full_like(t_rel, j, dtype=int))

        times_win = np.concatenate(times_win) if len(times_win) else np.array([])
        comps_win = np.concatenate(comps_win) if len(comps_win) else np.array([], dtype=int)
        if times_win.size:
            order = np.argsort(times_win)
            times_win = times_win[order]
            comps_win = comps_win[order]

        # ----------------------------
        # 3) intensità fitted (da path reale)
        # ----------------------------
        empty_events = {name: np.array([]) for name in COMPONENTS_2D}
        cal_dummy = HawkesCalibrator(empty_events, T=T_win)

        t_grid, lam_grid = cal_dummy.intensity_from_path(
            mu=w.fit.mu, A=w.fit.A, beta=w.fit.beta,
            times=times_win, comps=comps_win,
            T=T_win, n_grid=n_grid
        )

        mask_grid = (t_grid >= t0_rel) & (t_grid <= t1_rel)
        t_plot_abs = (t_grid[mask_grid] + w.t0)  # assoluto (sec da open)

        # decimazione curva intensità
        if t_plot_abs.size > intensity_max_pts:
            step = max(1, int(np.ceil(t_plot_abs.size / intensity_max_pts)))
            t_plot = t_plot_abs[::step]
            lam_plot = lam_grid[:, mask_grid][:, ::step]
        else:
            t_plot = t_plot_abs
            lam_plot = lam_grid[:, mask_grid]

        # ----------------------------
        # 4) sim (se non fornita)
        # ----------------------------
        if events_sim_2D is None:
            events_sim_2D, _, _ = lob4.simulate_piecewise_ogata(win_results, seed=seed)

        # ----------------------------
        # 5) plot
        # ----------------------------
        fig, axes = plt.subplots(len(COMPONENTS_2D), 1, sharex=True, figsize=(14, 6))
        if len(COMPONENTS_2D) == 1:
            axes = [axes]

        fig.suptitle(f"Intensity & Events [{t0_abs:.0f}, {t1_abs:.0f}]")

        rng = np.random.default_rng(seed)

        for i, name in enumerate(COMPONENTS_2D):
            ax = axes[i]

            lam = lam_plot[i, :]
            ax.plot(t_plot, lam, lw=1.5, alpha=0.9, label="Intensity (fit)")

            lam_max = float(np.nanmax(lam)) if lam.size else 1.0
            y_real = max(1e-9, event_height_real * lam_max)
            y_sim  = max(1e-9, event_height_sim  * lam_max)

            # --- REAL events ---
            t_real = lob4.events[name]
            t_real = t_real[(t_real >= t0_abs) & (t_real <= t1_abs)]
            if (max_events is not None) and (t_real.size > max_events):
                idx = rng.choice(t_real.size, size=max_events, replace=False)
                t_real = np.sort(t_real[idx])
            if t_real.size:
                ax.vlines(t_real, 0.0, y_real, alpha=alpha_real, lw=0.7, label="Real")

            # --- SIM events ---
            t_sim = events_sim_2D.get(name, np.array([]))
            t_sim = t_sim[(t_sim >= t0_abs) & (t_sim <= t1_abs)]
            if (max_events is not None) and (t_sim.size > max_events):
                idx = rng.choice(t_sim.size, size=max_events, replace=False)
                t_sim = np.sort(t_sim[idx])
            if t_sim.size:
                ax.vlines(t_sim, 0.0, y_sim, alpha=alpha_sim, lw=0.7, linestyle=":", label="Sim")

            ax.set_ylabel(name)
            ax.grid(True, alpha=0.15, linestyle="--")
            if i == 0:
                ax.legend(loc="upper right", frameon=True)

        axes[-1].set_xlabel("Time" if show_clock else "Time (sec from open)")
        if show_clock:
            apply_time_of_day_axis(
                axes[-1],
                base_seconds=0.0,       # x già in sec-from-open
                show_seconds=show_seconds,
                rotate=rotate_xticks,
            )

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        return fig, axes
