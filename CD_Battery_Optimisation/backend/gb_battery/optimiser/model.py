"""Pyomo model builder for the deterministic battery co-optimisation.

The model is a Mixed-Integer Linear Program (binaries only enforce that charging and
discharging cannot happen simultaneously). All economics are linear so it solves
quickly with the open-source HiGHS solver.

Sign & unit conventions
------------------------
* ``charge_mw`` / ``discharge_mw`` are **grid-side (AC) power** in MW.
* Energy drawn from the grid while charging = ``charge_mw * dt`` (MWh);
  stored-energy added = ``charge_efficiency * charge_mw * dt``.
* Energy delivered to the grid while discharging = ``discharge_mw * dt`` (MWh);
  stored-energy removed = ``discharge_mw * dt / discharge_efficiency``.
* ``net_export_mw = discharge_mw - charge_mw`` (positive = exporting to grid).
* Objective is a **maximisation** of expected profit in GBP.

The service (reserve) constraints are deliberately *conservative simplifications* of
real GB product rules — see :mod:`gb_battery.optimiser.service` and the methodology
docs. Real Dynamic Containment / Balancing Reserve rules require more detail.
"""

from __future__ import annotations

from statistics import median

import pyomo.environ as pyo

from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.inputs import OptimisationInputs


def _terminal_value(config: BatteryConfig, inputs: OptimisationInputs) -> float:
    """GBP/MWh applied to the ending SoC.

    If not configured, value carried-over energy at the discharge-efficiency-adjusted
    median horizon wholesale price. This is an *assumption* (documented in the UI): it
    stops the optimiser from dumping all stored energy in the final period purely
    because there is no modelled future.
    """
    if config.terminal_soc_value_gbp_per_mwh is not None:
        return config.terminal_soc_value_gbp_per_mwh
    prices = [p.wholesale_price for p in inputs.periods]
    return config.discharge_efficiency * float(median(prices)) if prices else 0.0


def build_model(
    config: BatteryConfig,
    inputs: OptimisationInputs,
    scenario: dict[str, list[float]] | None = None,
) -> pyo.ConcreteModel:
    """Construct the concrete Pyomo model.

    ``scenario`` optionally overrides per-period price paths (used by the stochastic
    and scenario-lab layers). Keys may include ``wholesale_price``,
    ``expected_imbalance_price``. Values are lists aligned to ``inputs.periods``.
    """
    periods = inputs.periods
    n = len(periods)
    streams = inputs.revenue_streams
    m = pyo.ConcreteModel(name="gb_battery_coopt")

    # --- Sets ---
    m.T = pyo.RangeSet(0, n - 1)  # period index
    m.Tsoc = pyo.RangeSet(0, n)  # SoC nodes (start of each period + terminal)

    # --- Parameters (plain floats captured via closures for speed) ---
    def price(t: int) -> float:
        if scenario and "wholesale_price" in scenario:
            return scenario["wholesale_price"][t]
        return periods[t].wholesale_price

    def imb_price(t: int) -> float:
        if scenario and "expected_imbalance_price" in scenario:
            return scenario["expected_imbalance_price"][t]
        p = periods[t]
        return p.expected_imbalance_price if p.expected_imbalance_price is not None else 0.0

    dt = [p.duration_hours for p in periods]

    # --- Decision variables ---
    m.charge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, config.maximum_charge_mw))
    m.discharge = pyo.Var(
        m.T, domain=pyo.NonNegativeReals, bounds=(0, config.maximum_discharge_mw)
    )
    m.soc = pyo.Var(
        m.Tsoc, domain=pyo.NonNegativeReals,
        bounds=(config.effective_min_soc, config.effective_max_soc),
    )
    m.cbin = pyo.Var(m.T, domain=pyo.Binary)
    m.dbin = pyo.Var(m.T, domain=pyo.Binary)
    m.up = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.down = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # Disable service variables when the stream is off.
    if not streams.upward_availability and not streams.bm_activation:
        for t in m.T:
            m.up[t].fix(0.0)
    if not streams.downward_availability and not streams.bm_activation:
        for t in m.T:
            m.down[t].fix(0.0)

    # Portfolio-mode extras.
    m.portfolio = inputs.portfolio_mode
    if inputs.portfolio_mode:
        m.wsell = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.wbuy = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.resid_pos = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # long imbalance (MWh)
        m.resid_neg = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # short imbalance (MWh)

    # --- Constraints ---
    m.soc_init = pyo.Constraint(expr=m.soc[0] == config.initial_soc_mwh)

    def _soc_balance(m, t):
        return m.soc[t + 1] == (
            m.soc[t]
            + config.charge_efficiency * m.charge[t] * dt[t]
            - m.discharge[t] * dt[t] / config.discharge_efficiency
        )

    m.soc_balance = pyo.Constraint(m.T, rule=_soc_balance)

    m.no_simultaneous = pyo.Constraint(m.T, rule=lambda m, t: m.cbin[t] + m.dbin[t] <= 1)
    m.charge_cap = pyo.Constraint(
        m.T, rule=lambda m, t: m.charge[t] <= config.maximum_charge_mw * m.cbin[t]
    )
    m.discharge_cap = pyo.Constraint(
        m.T, rule=lambda m, t: m.discharge[t] <= config.maximum_discharge_mw * m.dbin[t]
    )

    # Grid connection limits on net export/import.
    m.grid_export = pyo.Constraint(
        m.T, rule=lambda m, t: m.discharge[t] - m.charge[t] <= config.grid_export_limit_mw
    )
    m.grid_import = pyo.Constraint(
        m.T, rule=lambda m, t: m.charge[t] - m.discharge[t] <= config.grid_import_limit_mw
    )

    # Simplified power-headroom for reserve.
    m.up_headroom = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.up[t] <= config.maximum_discharge_mw - (m.discharge[t] - m.charge[t]),
    )
    m.down_headroom = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.down[t] <= (m.discharge[t] - m.charge[t]) + config.maximum_charge_mw,
    )

    # Simplified energy-duration for reserve (conservative).
    h_up = config.upward_service_duration_h
    h_down = config.downward_service_duration_h
    m.up_energy = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.soc[t] - config.effective_min_soc
        >= m.up[t] * h_up / config.discharge_efficiency,
    )
    m.down_energy = pyo.Constraint(
        m.T,
        rule=lambda m, t: config.effective_max_soc - m.soc[t]
        >= config.charge_efficiency * m.down[t] * h_down,
    )

    # Optional ramp limits on net export between consecutive periods.
    if config.ramp_up_mw_per_period is not None or config.ramp_down_mw_per_period is not None:
        def _ramp(m, t):
            if t == 0:
                return pyo.Constraint.Skip
            delta = (m.discharge[t] - m.charge[t]) - (m.discharge[t - 1] - m.charge[t - 1])
            up_lim = config.ramp_up_mw_per_period
            down_lim = config.ramp_down_mw_per_period
            if up_lim is not None and down_lim is not None:
                return (-down_lim, delta, up_lim)
            if up_lim is not None:
                return delta <= up_lim
            return delta >= -down_lim

        m.ramp = pyo.Constraint(m.T, rule=_ramp)

    # Terminal SoC floor.
    m.terminal_soc = pyo.Constraint(expr=m.soc[n] >= config.minimum_terminal_soc_mwh)

    # Cycling limit (equivalent full cycles per day, measured by discharge throughput).
    if config.maximum_cycles_per_day is not None:
        total_hours = sum(dt)
        days = max(total_hours / 24.0, 1e-9)
        m.cycle_limit = pyo.Constraint(
            expr=sum(m.discharge[t] * dt[t] for t in m.T)
            <= config.maximum_cycles_per_day * config.energy_capacity_mwh * days
        )

    # Portfolio balance -> residual imbalance.
    if inputs.portfolio_mode:
        def _balance(m, t):
            renewable = periods[t].renewable_generation_mwh or 0.0
            demand_ob = periods[t].demand_obligation_mwh or 0.0
            net_export_mwh = (m.discharge[t] - m.charge[t]) * dt[t]
            return m.resid_pos[t] - m.resid_neg[t] == (
                renewable + net_export_mwh + m.wbuy[t] - demand_ob - m.wsell[t]
            )

        m.portfolio_balance = pyo.Constraint(m.T, rule=_balance)

    # --- Objective ---
    terminal_val = _terminal_value(config, inputs)
    m.terminal_value_coeff = terminal_val

    def _objective(m):
        expr = 0.0
        for t in m.T:
            if inputs.portfolio_mode:
                if streams.wholesale:
                    expr += price(t) * (m.wsell[t] - m.wbuy[t])
                # Imbalance settlement + risk penalty on residual.
                if streams.imbalance:
                    expr += imb_price(t) * (m.resid_pos[t] - m.resid_neg[t])
                    expr -= inputs.imbalance_penalty_gbp_per_mwh * (
                        m.resid_pos[t] + m.resid_neg[t]
                    )
            else:
                if streams.wholesale:
                    expr += price(t) * (m.discharge[t] - m.charge[t]) * dt[t]

            # Availability payments.
            if streams.upward_availability:
                expr += periods[t].upward_availability_price * m.up[t] * dt[t]
            if streams.downward_availability:
                expr += periods[t].downward_availability_price * m.down[t] * dt[t]
            # Expected BM activation margin (per MW of reserved capability per hour).
            if streams.bm_activation:
                expr += periods[t].expected_bm_up_margin_gbp_per_mw * m.up[t] * dt[t]
                expr += periods[t].expected_bm_down_margin_gbp_per_mw * m.down[t] * dt[t]

            # Degradation on throughput (both directions, per user spec).
            expr -= config.degradation_cost_gbp_per_mwh_throughput * (
                m.charge[t] + m.discharge[t]
            ) * dt[t]

        # Terminal battery value.
        expr += terminal_val * m.soc[n]
        return expr

    m.objective = pyo.Objective(rule=_objective, sense=pyo.maximize)
    return m
