"""Frozen synthetic public-data sample.

A deterministic, seeded multi-day half-hourly dataset so reviewers can run the
optimiser, forecasters and backtest **entirely offline**. Everything here is
``DataKind.SYNTHETIC`` — it resembles GB market structure (diurnal price shape,
weekday/weekend demand, variable wind, midday solar) but is generated, not observed.

Both *forecast* and *outturn* columns are produced, with distinct publication
timestamps, so the backtest's leakage audit and forecast-error metrics have
something realistic to work with.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from gb_battery.demo.scenarios import _fundamentals, diurnal_price_shape
from gb_battery.settlement import settlement_periods_for_day

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data_samples"
SAMPLE_FILE = SAMPLE_DIR / "synthetic_history.parquet"

DEFAULT_START = date(2025, 1, 6)  # a Monday
DEFAULT_DAYS = 35


def generate_synthetic_history(
    start: date = DEFAULT_START, days: int = DEFAULT_DAYS, seed: int = 42
) -> pd.DataFrame:
    """Generate a seeded synthetic half-hourly history spanning ``days`` days."""
    rng = np.random.default_rng(seed)
    frames: list[dict] = []
    for d in range(days):
        day = start + timedelta(days=d)
        periods = settlement_periods_for_day(day)
        n = len(periods)
        is_weekend = day.weekday() >= 5
        # Daily regime multipliers (persistent wind/demand states).
        wind_state = float(np.clip(rng.normal(1.0, 0.35), 0.2, 1.9))
        demand_scale = 0.9 if is_weekend else 1.0
        price_level = float(rng.normal(0.0, 12.0))  # daily price shift
        for p in periods:
            demand0, wind0, solar0 = _fundamentals(p.settlement_period, n)
            demand = demand0 * demand_scale
            wind = wind0 * wind_state
            solar = solar0
            residual = demand - wind - solar
            # Price: diurnal shape + daily level + residual-demand tilt + noise.
            base_price = diurnal_price_shape(p.settlement_period, n)
            resid_tilt = (residual - 20000) / 1000.0  # +1 GBP per 1 GW residual over 20 GW
            outturn_price = base_price + price_level + resid_tilt + float(rng.normal(0, 6))
            # Day-ahead forecast = outturn minus the noise, plus forecast error.
            fc_price = base_price + price_level + resid_tilt + float(rng.normal(0, 8))
            # System (imbalance) price tracks wholesale with a state-dependent spread.
            short = residual > 24000
            system_price = outturn_price + (float(rng.normal(15, 8)) if short else float(rng.normal(-6, 6)))
            frames.append(
                {
                    "settlement_date": day,
                    "settlement_period": p.settlement_period,
                    "start_utc": p.start_utc,
                    # Forecasts (available day-ahead)
                    "wholesale_price_forecast": round(fc_price, 2),
                    "demand_forecast_mw": round(demand + rng.normal(0, 300), 1),
                    "wind_forecast_mw": round(max(wind + rng.normal(0, 400), 0.0), 1),
                    "solar_forecast_mw": round(max(solar + rng.normal(0, 150), 0.0), 1),
                    # Outturn (observed after the fact)
                    "wholesale_price": round(outturn_price, 2),
                    "system_price": round(system_price, 2),
                    "demand_outturn_mw": round(demand, 1),
                    "wind_outturn_mw": round(max(wind, 0.0), 1),
                    "solar_outturn_mw": round(max(solar, 0.0), 1),
                    "residual_demand_mw": round(residual, 1),
                    "prob_short_label": int(short),
                    # Lineage timestamps
                    "forecast_published_at": datetime.combine(day - timedelta(days=1), datetime.min.time()),
                    "event_at": p.start_utc,
                    "source": "synthetic",
                }
            )
    return pd.DataFrame(frames)


def freeze_sample(path: Path = SAMPLE_FILE) -> Path:
    """Write the frozen synthetic history to Parquet (committed to the repo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_history()
    df.to_parquet(path, index=False)
    return path


def load_sample() -> pd.DataFrame:
    """Load the frozen sample, generating it on the fly if absent."""
    if SAMPLE_FILE.exists():
        return pd.read_parquet(SAMPLE_FILE)
    return generate_synthetic_history()


def sample_day(day: date | None = None) -> pd.DataFrame:
    """Return one day of the frozen sample (defaults to the first day)."""
    df = load_sample()
    if day is None:
        day = df["settlement_date"].min()
    return df[df["settlement_date"] == day].sort_values("settlement_period").reset_index(drop=True)
