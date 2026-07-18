"""Backtest strategies: rule-based benchmarks and optimiser-driven policies.

Each strategy maps (battery config, forecast inputs, starting SoC) to a per-period
schedule of charge/discharge/reserve MW. Optimiser strategies reuse the production
:func:`gb_battery.optimiser.deterministic.optimise`; rule strategies are simple,
transparent baselines a desk would recognise.

Crucially, strategies only ever see *forecast* inputs — the realised outturn used to
settle P&L is applied later by the engine, so there is no look-ahead leakage (except
the deliberately-labelled perfect-foresight upper bound).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.deterministic import optimise
from gb_battery.optimiser.inputs import OptimisationInputs


@dataclass
class PeriodDecision:
    settlement_period: int
    charge_mw: float
    discharge_mw: float
    up_mw: float = 0.0
    down_mw: float = 0.0


Schedule = list[PeriodDecision]


def _empty(inputs: OptimisationInputs) -> Schedule:
    return [PeriodDecision(p.settlement_period, 0.0, 0.0) for p in inputs.periods]


def no_operation(config: BatteryConfig, inputs: OptimisationInputs, soc: float) -> Schedule:  # noqa: ARG001
    """Do nothing — a battery that never trades."""
    return _empty(inputs)


def threshold_rule(
    config: BatteryConfig, inputs: OptimisationInputs, soc: float,
    low_pct: float = 30.0, high_pct: float = 70.0,
) -> Schedule:
    """Charge below the ``low_pct`` price percentile, discharge above ``high_pct``.

    A transparent, SoC-aware rule using this horizon's forecast price distribution.
    """
    prices = np.array([p.wholesale_price for p in inputs.periods])
    lo, hi = np.percentile(prices, [low_pct, high_pct])
    sched: Schedule = []
    cur = soc
    for p in inputs.periods:
        dt = p.duration_hours
        charge = discharge = 0.0
        if p.wholesale_price <= lo:
            room = config.effective_max_soc - cur
            charge = min(config.maximum_charge_mw, room / max(config.charge_efficiency * dt, 1e-9))
            charge = max(charge, 0.0)
            cur += config.charge_efficiency * charge * dt
        elif p.wholesale_price >= hi:
            avail = cur - config.effective_min_soc
            discharge = min(config.maximum_discharge_mw, avail * config.discharge_efficiency / max(dt, 1e-9))
            discharge = max(discharge, 0.0)
            cur -= discharge * dt / config.discharge_efficiency
        sched.append(PeriodDecision(p.settlement_period, round(charge, 4), round(discharge, 4)))
    return sched


def fixed_percentile(config: BatteryConfig, inputs: OptimisationInputs, soc: float) -> Schedule:
    """Fixed 25/75 percentile variant of the threshold rule."""
    return threshold_rule(config, inputs, soc, low_pct=25.0, high_pct=75.0)


def deterministic_optimiser(config: BatteryConfig, inputs: OptimisationInputs, soc: float) -> Schedule:
    """MPC policy: optimise this horizon on the point forecast."""
    cfg = config.model_copy(update={"initial_soc_mwh": _clamp_soc(config, soc)})
    res = optimise(cfg, inputs, compute_marginals=False)
    return [
        PeriodDecision(p.settlement_period, p.charge_mw, p.discharge_mw, p.upward_reserved_mw, p.downward_reserved_mw)
        for p in res.periods
    ]


def perfect_foresight(config: BatteryConfig, inputs: OptimisationInputs, soc: float) -> Schedule:
    """Upper bound: optimise directly on the realised outturn inputs.

    Not achievable live — the engine passes outturn (not forecast) inputs here.
    """
    return deterministic_optimiser(config, inputs, soc)


def _clamp_soc(config: BatteryConfig, soc: float) -> float:
    return float(min(max(soc, config.effective_min_soc), config.effective_max_soc))


STRATEGIES = {
    "no_operation": no_operation,
    "threshold_rule": threshold_rule,
    "fixed_percentile": fixed_percentile,
    "deterministic_optimiser": deterministic_optimiser,
    "perfect_foresight": perfect_foresight,
}
