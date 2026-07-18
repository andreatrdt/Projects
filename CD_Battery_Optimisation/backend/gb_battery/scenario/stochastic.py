"""Scenario-weighted stochastic optimisation with an optional CVaR risk penalty.

A single physical schedule (charge/discharge/SoC/reserve) is committed *here-and-now*
— you cannot dispatch a different battery per scenario — while profit is evaluated
across price scenarios. Risk aversion penalises downside via CVaR using the
Rockafellar–Uryasev linear formulation:

    CVaR_alpha(loss) = eta + 1/(1-alpha) * E[(loss - eta)+]

with ``loss_s = -profit_s``. The objective maximises::

    E[profit] - risk_aversion * CVaR_alpha(loss)

A ``robust`` mode instead optimises the worst-case scenario (max-min).
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo

from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.deterministic import build_result_from_model
from gb_battery.optimiser.inputs import OptimisationInputs
from gb_battery.optimiser.model import build_model
from gb_battery.optimiser.results import OptimisationResult
from gb_battery.optimiser.solver import solve
from gb_battery.scenario.generator import ScenarioSet, generate_price_scenarios


def _deterministic_part(m: pyo.ConcreteModel, config: BatteryConfig, inputs: OptimisationInputs):
    """Scenario-independent profit terms (services, BM, degradation, terminal value)."""
    streams = inputs.revenue_streams
    periods = inputs.periods
    expr = 0.0
    for t in m.T:
        dt = periods[t].duration_hours
        if streams.upward_availability:
            expr += periods[t].upward_availability_price * m.up[t] * dt
        if streams.downward_availability:
            expr += periods[t].downward_availability_price * m.down[t] * dt
        if streams.bm_activation:
            expr += periods[t].expected_bm_up_margin_gbp_per_mw * m.up[t] * dt
            expr += periods[t].expected_bm_down_margin_gbp_per_mw * m.down[t] * dt
        expr -= config.degradation_cost_gbp_per_mwh_throughput * (m.charge[t] + m.discharge[t]) * dt
    expr += m.terminal_value_coeff * m.soc[len(periods)]
    return expr


def _wholesale_profit(m: pyo.ConcreteModel, inputs: OptimisationInputs, prices: np.ndarray):
    """Scenario wholesale profit expression for a price path."""
    periods = inputs.periods
    if not inputs.revenue_streams.wholesale:
        return 0.0
    return sum(
        prices[t] * (m.discharge[t] - m.charge[t]) * periods[t].duration_hours for t in m.T
    )


def optimise_stochastic(
    config: BatteryConfig,
    inputs: OptimisationInputs,
    *,
    scenarios: ScenarioSet | None = None,
    n_scenarios: int = 20,
    risk_aversion: float | None = None,
    cvar_alpha: float | None = None,
    robust: bool = False,
    seed: int = 0,
) -> OptimisationResult:
    """Solve the scenario-weighted (optionally CVaR / robust) dispatch."""
    risk_aversion = inputs.risk_aversion if risk_aversion is None else risk_aversion
    cvar_alpha = inputs.cvar_alpha if cvar_alpha is None else cvar_alpha
    scenarios = scenarios or generate_price_scenarios(inputs, n_scenarios=n_scenarios, seed=seed)

    m = build_model(config, inputs)  # physical constraints + a deterministic objective
    m.objective.deactivate()  # replace with the stochastic objective

    S = scenarios.n_scenarios
    probs = scenarios.probabilities
    det = _deterministic_part(m, config, inputs)

    # Per-scenario profit expressions.
    profit = {s: det + _wholesale_profit(m, inputs, scenarios.wholesale_prices[s]) for s in range(S)}
    expected_profit = sum(probs[s] * profit[s] for s in range(S))

    if robust:
        m.worst = pyo.Var(domain=pyo.Reals)
        m.robust_con = pyo.Constraint(
            range(S), rule=lambda mm, s: mm.worst <= profit[s]
        )
        m.stoch_obj = pyo.Objective(expr=m.worst, sense=pyo.maximize)
    elif risk_aversion and risk_aversion > 0:
        # Rockafellar–Uryasev CVaR of the loss (= -profit).
        m.eta = pyo.Var(domain=pyo.Reals)
        m.z = pyo.Var(range(S), domain=pyo.NonNegativeReals)
        m.cvar_con = pyo.Constraint(
            range(S), rule=lambda mm, s: mm.z[s] >= (-profit[s]) - mm.eta
        )
        cvar = m.eta + (1.0 / (1.0 - cvar_alpha)) * sum(probs[s] * m.z[s] for s in range(S))
        m.stoch_obj = pyo.Objective(expr=expected_profit - risk_aversion * cvar, sense=pyo.maximize)
    else:
        m.stoch_obj = pyo.Objective(expr=expected_profit, sense=pyo.maximize)

    outcome = solve(m)
    if outcome.status not in {"optimal", "time_limit"}:
        return OptimisationResult(
            status=outcome.status, solver=outcome.solver, objective_gbp=0.0, periods=[],
            total_wholesale_pnl_gbp=0.0, total_service_pnl_gbp=0.0,
            total_bm_activation_pnl_gbp=0.0, total_degradation_cost_gbp=0.0,
            total_imbalance_cost_gbp=0.0, terminal_soc_value_gbp=0.0,
            total_expected_pnl_gbp=0.0, full_cycle_equivalents=0.0, horizon=len(inputs.periods),
            portfolio_mode=inputs.portfolio_mode, risk_aversion=risk_aversion,
            warnings=[f"Stochastic solve returned '{outcome.status}'."],
        )

    # Expected profit realised by the chosen schedule (for reporting).
    exp_profit_val = float(pyo.value(expected_profit))
    warnings = [
        f"Stochastic mode: {S} scenarios, "
        + ("robust (worst-case)" if robust else f"CVaR α={cvar_alpha}, risk_aversion={risk_aversion}"),
    ]
    # Report the expected-price per-period economics via the shared builder.
    inputs_expected = inputs.model_copy(update={"risk_aversion": risk_aversion})
    result = build_result_from_model(
        config, inputs_expected, m, outcome.solver, exp_profit_val, warnings=warnings
    )
    return result
