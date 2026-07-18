"""Benchmark strategies evaluated on the same PIT store as the main replay.

All rolling benchmarks run through the *identical* engine loop (same
information sets, same settlement rule), so differences come from the policy —
not from information advantages. Perfect foresight is the only strategy allowed
to see the realised path, and it is labelled as an unattainable upper bound.
"""

from __future__ import annotations

from datetime import date

from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.deterministic import optimise
from gb_battery.replay.engine import (
    ReplayEngine,
    ReplayOptions,
    perfect_foresight_day,
)
from gb_battery.replay.pit import PITDataStore

PERFECT_FORESIGHT_LABEL = "Perfect foresight benchmark — not a tradable strategy."


def run_perfect_foresight(
    config: BatteryConfig, store: PITDataStore, day: date
) -> dict | None:
    """One-shot optimisation on the realised price path (upper bound)."""
    built = perfect_foresight_day(config, store, day)
    if built is None:
        return None
    inputs, prices = built
    result = optimise(config, inputs, compute_marginals=False)
    if result.status != "optimal":
        return None
    revenue = sum(
        p.wholesale_price * p.discharge_mw * p.duration_hours for p in result.periods
    )
    cost = sum(p.wholesale_price * p.charge_mw * p.duration_hours for p in result.periods)
    return {
        "strategy": "perfect_foresight",
        "label": PERFECT_FORESIGHT_LABEL,
        "realised_pnl_gbp": round(result.total_wholesale_pnl_gbp - result.total_degradation_cost_gbp, 2),
        "wholesale_revenue_gbp": round(revenue, 2),
        "charging_cost_gbp": round(cost, 2),
        "degradation_cost_gbp": round(result.total_degradation_cost_gbp, 2),
        "cycles": result.full_cycle_equivalents,
        "schedule": [
            {
                "settlement_period": p.settlement_period,
                "action": str(p.action.value if hasattr(p.action, "value") else p.action),
                "charge_mw": p.charge_mw,
                "discharge_mw": p.discharge_mw,
                "ending_soc_mwh": p.ending_soc_mwh,
                "price": p.wholesale_price,
            }
            for p in result.periods
        ],
        "day_prices": prices,
    }


def _peak_trough_capture(engine: ReplayEngine) -> dict:
    """Did the strategy discharge at the day's realised peak / charge at the trough?"""
    settled = [d for d in engine.decisions if d.settlement_status == "settled"]
    if not settled:
        return {"peak_captured": None, "trough_captured": None}
    peak = max(settled, key=lambda d: d.actual_price or float("-inf"))
    trough = min(settled, key=lambda d: d.actual_price or float("inf"))
    return {
        "peak_sp": peak.settlement_period,
        "peak_price": peak.actual_price,
        "peak_captured": peak.discharge_mw > 1e-4,
        "trough_sp": trough.settlement_period,
        "trough_price": trough.actual_price,
        "trough_captured": trough.charge_mw > 1e-4,
    }


def compare_replay_strategies(engine: ReplayEngine) -> dict:
    """Compare the finished main replay against benchmarks on the same store.

    Returns a table of per-strategy realised metrics plus the perfect-foresight
    capture ratio and forecast-accuracy diagnostics for the main strategy.
    """
    main = engine.realised_summary()
    main_row = {
        "strategy": engine.options.strategy,
        "label": "Rolling point-in-time strategy (main result)",
        **main,
    }

    rows = [main_row]

    # No-operation: identically zero P&L (kept explicit for the comparison table).
    rows.append(
        {
            "strategy": "no_operation",
            "label": "Battery never trades",
            "realised_pnl_gbp": 0.0,
            "wholesale_revenue_gbp": 0.0,
            "charging_cost_gbp": 0.0,
            "degradation_cost_gbp": 0.0,
            "cycles": 0.0,
            "max_drawdown_gbp": 0.0,
        }
    )

    # Rolling threshold rule on the same PIT information sets.
    alt = "rolling_threshold" if engine.options.strategy != "rolling_threshold" else "rolling_forecast"
    alt_engine = ReplayEngine(
        engine.config,
        engine.day,
        options=ReplayOptions(**{**engine.options.model_dump(), "strategy": alt}),
        store=engine.store,
        forecaster=engine.forecaster,
    )
    alt_engine.run(max_steps=len(engine.decisions))
    rows.append(
        {
            "strategy": alt,
            "label": "Alternative rolling policy on identical information",
            **alt_engine.realised_summary(),
        }
    )

    pf = run_perfect_foresight(engine.config, engine.store, engine.day)
    pf_pnl = pf["realised_pnl_gbp"] if pf else None
    if pf:
        rows.append({k: v for k, v in pf.items() if k not in {"schedule", "day_prices"}})

    for r in rows:
        if pf_pnl and r["strategy"] != "perfect_foresight" and pf_pnl != 0:
            r["capture_of_perfect_pct"] = round(100.0 * r.get("realised_pnl_gbp", 0.0) / pf_pnl, 1)

    return {
        "table": rows,
        "perfect_foresight_pnl_gbp": pf_pnl,
        "perfect_foresight_label": PERFECT_FORESIGHT_LABEL,
        "perfect_foresight_schedule": pf["schedule"] if pf else None,
        "peak_trough": _peak_trough_capture(engine),
        "note": (
            "All rolling strategies observed identical point-in-time information. "
            "Perfect foresight saw the realised path and is not attainable."
        ),
    }
