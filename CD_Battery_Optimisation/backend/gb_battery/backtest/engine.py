"""Chronological rolling-horizon backtest engine.

For each settlement date the engine:
1. builds optimiser inputs from **day-ahead forecasts** (fundamentals + price forecast);
2. asks the strategy for a schedule (SoC carried from the previous day);
3. **settles** the committed schedule against the **realised outturn** price;
4. rolls SoC forward to the next day.

Only the perfect-foresight benchmark is allowed to see outturns when deciding — it is
the theoretical upper bound, explicitly labelled as unachievable live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from gb_battery.backtest.metrics import cvar, max_drawdown
from gb_battery.backtest.strategies import STRATEGIES, Schedule
from gb_battery.battery.config import BatteryConfig
from gb_battery.forecast.metrics import mae, rmse
from gb_battery.optimiser.inputs import OptimisationInputs, PeriodInput, RevenueStreams
from gb_battery.settlement import settlement_periods_for_day


@dataclass
class BacktestResult:
    strategy: str
    ledger: pd.DataFrame = field(repr=False)
    summary: dict

    def to_summary_dict(self) -> dict:
        return {"strategy": self.strategy, **self.summary}


def _build_inputs(
    day: date,
    rows: pd.DataFrame,
    price_col: str,
    streams: RevenueStreams,
    up_avail: float,
    down_avail: float,
) -> OptimisationInputs:
    periods = {p.settlement_period: p for p in settlement_periods_for_day(day)}
    out: list[PeriodInput] = []
    for _, r in rows.sort_values("settlement_period").iterrows():
        sp = int(r["settlement_period"])
        per = periods.get(sp)
        if per is None:
            continue
        demand = _num(r.get("demand_forecast_mw"))
        wind = _num(r.get("wind_forecast_mw"))
        solar = _num(r.get("solar_forecast_mw"))
        residual = None if demand is None else demand - (wind or 0) - (solar or 0)
        out.append(
            PeriodInput(
                settlement_date=day,
                settlement_period=sp,
                start_utc=per.start_utc,
                end_utc=per.end_utc,
                duration_hours=per.duration_hours,
                wholesale_price=float(r[price_col]),
                upward_availability_price=up_avail,
                downward_availability_price=down_avail,
                demand_forecast_mw=demand,
                wind_forecast_mw=wind,
                solar_forecast_mw=solar,
                residual_demand_mw=residual,
            )
        )
    return OptimisationInputs(periods=out, revenue_streams=streams)


def _settle(
    config: BatteryConfig,
    schedule: Schedule,
    outturn_inputs: OptimisationInputs,
    forecast_inputs: OptimisationInputs,
    soc0: float,
    streams: RevenueStreams,
) -> tuple[list[dict], float]:
    """Settle a committed schedule against outturn prices; return (rows, ending_soc)."""
    rows: list[dict] = []
    soc = soc0
    out_by_sp = {p.settlement_period: p for p in outturn_inputs.periods}
    fc_by_sp = {p.settlement_period: p for p in forecast_inputs.periods}
    for dec in schedule:
        op = out_by_sp[dec.settlement_period]
        fp = fc_by_sp[dec.settlement_period]
        dt = op.duration_hours
        # Physically evolve SoC and clamp to feasibility (guards rule strategies).
        add = config.charge_efficiency * dec.charge_mw * dt
        remove = dec.discharge_mw * dt / config.discharge_efficiency
        new_soc = soc + add - remove
        new_soc = min(max(new_soc, config.effective_min_soc), config.effective_max_soc)
        # Recompute effective energies from the clamped SoC change.
        delta = new_soc - soc
        eff_charge = max(delta, 0.0) / config.charge_efficiency / dt if delta > 0 else 0.0
        eff_discharge = max(-delta, 0.0) * config.discharge_efficiency / dt if delta < 0 else 0.0
        charge_mw = eff_charge if delta > 0 else 0.0
        discharge_mw = eff_discharge if delta < 0 else 0.0

        e_charge = charge_mw * dt
        e_discharge = discharge_mw * dt
        wholesale_pnl = op.wholesale_price * (discharge_mw - charge_mw) * dt
        service_pnl = 0.0
        if streams.upward_availability:
            service_pnl += op.upward_availability_price * dec.up_mw * dt
        if streams.downward_availability:
            service_pnl += op.downward_availability_price * dec.down_mw * dt
        degradation = config.degradation_cost_gbp_per_mwh_throughput * (charge_mw + discharge_mw) * dt
        total_pnl = wholesale_pnl + service_pnl - degradation

        at_limit = (
            charge_mw > config.maximum_charge_mw - 1e-3
            or discharge_mw > config.maximum_discharge_mw - 1e-3
            or new_soc <= config.effective_min_soc + 1e-3
            or new_soc >= config.effective_max_soc - 1e-3
        )
        rows.append(
            {
                "settlement_date": op.settlement_date,
                "settlement_period": op.settlement_period,
                "start_utc": op.start_utc,
                "price_forecast": fp.wholesale_price,
                "price_outturn": op.wholesale_price,
                "charge_mw": round(charge_mw, 4),
                "discharge_mw": round(discharge_mw, 4),
                "beginning_soc_mwh": round(soc, 4),
                "ending_soc_mwh": round(new_soc, 4),
                "up_mw": dec.up_mw,
                "down_mw": dec.down_mw,
                "energy_charged_mwh": round(e_charge, 4),
                "energy_discharged_mwh": round(e_discharge, 4),
                "wholesale_pnl_gbp": round(wholesale_pnl, 4),
                "service_pnl_gbp": round(service_pnl, 4),
                "degradation_cost_gbp": round(degradation, 4),
                "total_pnl_gbp": round(total_pnl, 4),
                "at_limit": at_limit,
            }
        )
        soc = new_soc
    return rows, soc


def run_backtest(
    config: BatteryConfig,
    history: pd.DataFrame,
    strategy: str,
    *,
    forecast_price_col: str = "wholesale_price_forecast",
    outturn_price_col: str = "wholesale_price",
    streams: RevenueStreams | None = None,
    up_availability_price: float = 0.0,
    down_availability_price: float = 0.0,
    initial_soc: float | None = None,
) -> BacktestResult:
    """Run a chronological backtest of ``strategy`` over ``history``."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}' (have {list(STRATEGIES)})")
    strat_fn = STRATEGIES[strategy]
    streams = streams or RevenueStreams(
        wholesale=True, upward_availability=bool(up_availability_price),
        downward_availability=bool(down_availability_price), bm_activation=False, imbalance=False,
    )

    df = history.copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
    days = sorted(df["settlement_date"].unique())

    soc = initial_soc if initial_soc is not None else config.initial_soc_mwh
    ledger_rows: list[dict] = []

    for day in days:
        rows = df[df["settlement_date"] == day]
        fc_inputs = _build_inputs(day, rows, forecast_price_col, streams, up_availability_price, down_availability_price)
        out_inputs = _build_inputs(day, rows, outturn_price_col, streams, up_availability_price, down_availability_price)
        # Perfect foresight decides on outturn; everyone else on forecast.
        decision_inputs = out_inputs if strategy == "perfect_foresight" else fc_inputs
        cfg_day = config.model_copy(update={"initial_soc_mwh": _clamp(config, soc)})
        schedule = strat_fn(cfg_day, decision_inputs, _clamp(config, soc))
        settled, soc = _settle(config, schedule, out_inputs, fc_inputs, soc, streams)
        ledger_rows.extend(settled)

    ledger = pd.DataFrame(ledger_rows)
    summary = _summarise(config, ledger)
    return BacktestResult(strategy=strategy, ledger=ledger, summary=summary)


def _summarise(config: BatteryConfig, ledger: pd.DataFrame) -> dict:
    if ledger.empty:
        return {}
    pnl = ledger["total_pnl_gbp"].to_numpy()
    cum = np.cumsum(pnl)
    total_discharge = ledger["energy_discharged_mwh"].sum()
    n = len(ledger)
    return {
        "total_pnl_gbp": round(float(pnl.sum()), 2),
        "wholesale_pnl_gbp": round(float(ledger["wholesale_pnl_gbp"].sum()), 2),
        "service_pnl_gbp": round(float(ledger["service_pnl_gbp"].sum()), 2),
        "degradation_cost_gbp": round(float(ledger["degradation_cost_gbp"].sum()), 2),
        "full_cycle_equivalents": round(float(total_discharge / config.energy_capacity_mwh), 3),
        "avg_soc_mwh": round(float(ledger["ending_soc_mwh"].mean()), 2),
        "pct_time_at_limits": round(float(ledger["at_limit"].mean() * 100), 1),
        "max_drawdown_gbp": round(max_drawdown(cum), 2),
        "cvar95_period_pnl_gbp": round(cvar(pnl, 0.95), 2),
        "price_forecast_mae": round(mae(ledger["price_outturn"], ledger["price_forecast"]), 2),
        "price_forecast_rmse": round(rmse(ledger["price_outturn"], ledger["price_forecast"]), 2),
        "n_periods": int(n),
        "n_days": int(ledger["settlement_date"].nunique()),
    }


def _num(x: Any) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _clamp(config: BatteryConfig, soc: float) -> float:
    return float(min(max(soc, config.effective_min_soc), config.effective_max_soc))
