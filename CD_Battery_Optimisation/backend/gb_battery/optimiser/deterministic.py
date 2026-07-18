"""High-level deterministic co-optimisation entry point.

``optimise()`` builds the Pyomo model, solves it with HiGHS, then translates the raw
solution into a rich :class:`OptimisationResult`: per-period economics, binding
constraints, marginal (shadow / perturbation) values and trader explanations.
"""

from __future__ import annotations

import pyomo.environ as pyo

from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.explain import explain_period
from gb_battery.optimiser.inputs import OptimisationInputs
from gb_battery.optimiser.model import build_model
from gb_battery.optimiser.results import (
    ActionLabel,
    MarginalValues,
    OptimisationResult,
    PeriodResult,
)
from gb_battery.optimiser.solver import guarded_solve, solve

TOL = 1e-4
SNAP = 1e-3  # dispatch magnitudes below this (MW) are numerical noise -> 0


def _v(model: pyo.ConcreteModel, comp, t) -> float:
    return float(pyo.value(comp[t]))


def _snap(x: float) -> float:
    """Zero out tiny numerical residuals so degenerate LP solutions read cleanly."""
    return 0.0 if abs(x) < SNAP else x


def _classify_action(charge: float, discharge: float, up: float, down: float) -> ActionLabel:
    if charge > TOL:
        return ActionLabel.CHARGE
    if discharge > TOL:
        return ActionLabel.DISCHARGE
    if up > TOL:
        return ActionLabel.RESERVE_UP
    if down > TOL:
        return ActionLabel.RESERVE_DOWN
    return ActionLabel.IDLE


def _binding_constraints(
    config: BatteryConfig,
    charge: float,
    discharge: float,
    soc_begin: float,
    soc_end: float,
    up: float,
    down: float,
    is_terminal: bool,
    terminal_soc: float,
) -> list[str]:
    b: list[str] = []
    if charge > config.maximum_charge_mw - TOL:
        b.append("charge_power_max")
    if discharge > config.maximum_discharge_mw - TOL:
        b.append("discharge_power_max")
    if soc_end > config.effective_max_soc - TOL:
        b.append("battery_full")
    if soc_end < config.effective_min_soc + TOL:
        b.append("battery_empty")
    if discharge - charge > config.grid_export_limit_mw - TOL:
        b.append("grid_export_limit")
    if charge - discharge > config.grid_import_limit_mw - TOL:
        b.append("grid_import_limit")
    # Reserve energy limits (tight to within tolerance).
    up_room = soc_begin - config.effective_min_soc - up * config.upward_service_duration_h / config.discharge_efficiency
    if up > TOL and abs(up_room) < 1e-3:
        b.append("up_energy")
    down_room = config.effective_max_soc - soc_begin - config.charge_efficiency * down * config.downward_service_duration_h
    if down > TOL and abs(down_room) < 1e-3:
        b.append("down_energy")
    up_power_room = config.maximum_discharge_mw - (discharge - charge) - up
    if up > TOL and abs(up_power_room) < 1e-3:
        b.append("up_power")
    if is_terminal and abs(soc_end - terminal_soc) < 1e-3 and terminal_soc > TOL:
        b.append("terminal_soc")
    return b


def _perturbation_marginals(
    config: BatteryConfig, inputs: OptimisationInputs, base_obj: float, eps: float = 0.5
) -> dict[str, float]:
    """Horizon-level marginal values via small perturbations of the config.

    Robust to the presence of binaries (no duals needed). Each value is the change in
    optimal objective per unit change of the resource.
    """
    out: dict[str, float] = {}

    def resolve(cfg: BatteryConfig) -> float | None:
        m = build_model(cfg, inputs)
        oc = solve(m)
        return oc.objective if oc.status in {"optimal", "time_limit"} else None

    # +eps MWh of stored energy available at start.
    if config.initial_soc_mwh + eps <= config.effective_max_soc:
        cfg = config.model_copy(update={"initial_soc_mwh": config.initial_soc_mwh + eps})
        obj = resolve(cfg)
        if obj is not None:
            out["stored_energy_gbp_per_mwh"] = (obj - base_obj) / eps
    # +eps MWh of usable capacity (raise max SoC and energy capacity together).
    cfg = config.model_copy(
        update={
            "maximum_soc_mwh": config.maximum_soc_mwh + eps,
            "energy_capacity_mwh": config.energy_capacity_mwh + eps,
        }
    )
    obj = resolve(cfg)
    if obj is not None:
        out["empty_capacity_gbp_per_mwh"] = (obj - base_obj) / eps
    # +eps MW of charge power.
    cfg = config.model_copy(update={"maximum_charge_mw": config.maximum_charge_mw + eps})
    obj = resolve(cfg)
    if obj is not None:
        out["charge_power_gbp_per_mw"] = (obj - base_obj) / eps
    # +eps MW of discharge power.
    cfg = config.model_copy(update={"maximum_discharge_mw": config.maximum_discharge_mw + eps})
    obj = resolve(cfg)
    if obj is not None:
        out["discharge_power_gbp_per_mw"] = (obj - base_obj) / eps
    return out


def _perperiod_stored_energy_duals(
    config: BatteryConfig, inputs: OptimisationInputs, model: pyo.ConcreteModel
) -> list[float] | None:
    """Per-period marginal value of stored energy via SoC-balance duals.

    Fix the binaries from the MILP solution, re-solve the resulting LP and read the
    dual of each SoC-balance constraint. Returns ``None`` if duals are unavailable in
    this Pyomo/HiGHS build (the caller falls back to horizon-level marginals).
    """
    try:
        from pyomo.contrib.appsi.solvers.highs import Highs

        # Work on a clone; fix binaries to the incumbent and *relax integrality*
        # (a fixed binary is still integer to HiGHS, which then returns no duals).
        lp = model.clone()
        for t in lp.T:
            cval = round(float(pyo.value(lp.cbin[t])))
            dval = round(float(pyo.value(lp.dbin[t])))
            lp.cbin[t].domain = pyo.Reals
            lp.dbin[t].domain = pyo.Reals
            lp.cbin[t].fix(cval)
            lp.dbin[t].fix(dval)
        opt = Highs()
        opt.config.stream_solver = False
        opt.config.load_solution = False
        res = guarded_solve(opt, lp)
        if str(res.termination_condition).split(".")[-1] != "optimal":
            return None
        duals = res.solution_loader.get_duals()
        # soc_balance[t]: soc[t+1] = soc[t] + ... ; its dual is the marginal value
        # (GBP/MWh) of energy at node t. Report magnitude as the water-value.
        vals: list[float] = []
        for t in lp.T:
            con = lp.soc_balance[t]
            vals.append(abs(float(duals.get(con, 0.0))))
        return vals
    except Exception:
        return None


def optimise(
    config: BatteryConfig,
    inputs: OptimisationInputs,
    compute_marginals: bool = True,
) -> OptimisationResult:
    """Run the deterministic co-optimisation and return a full result."""
    n = len(inputs.periods)

    model = build_model(config, inputs)
    outcome = solve(model)

    if outcome.status not in {"optimal", "time_limit"}:
        return OptimisationResult(
            status=outcome.status, solver=outcome.solver, objective_gbp=0.0, periods=[],
            total_wholesale_pnl_gbp=0.0, total_service_pnl_gbp=0.0,
            total_bm_activation_pnl_gbp=0.0, total_degradation_cost_gbp=0.0,
            total_imbalance_cost_gbp=0.0, terminal_soc_value_gbp=0.0,
            total_expected_pnl_gbp=0.0, full_cycle_equivalents=0.0, horizon=n,
            portfolio_mode=inputs.portfolio_mode, risk_aversion=inputs.risk_aversion,
            warnings=[f"Solver returned status '{outcome.status}'."],
        )

    base_obj = outcome.objective or 0.0

    # Optional marginal values.
    per_period_stored: list[float] | None = None
    horizon_marg: dict[str, float] = {}
    if compute_marginals:
        per_period_stored = _perperiod_stored_energy_duals(config, inputs, model)
        horizon_marg = _perturbation_marginals(config, inputs, base_obj)

    return build_result_from_model(
        config, inputs, model, outcome.solver, base_obj,
        per_period_stored=per_period_stored, horizon_marg=horizon_marg,
    )


def build_result_from_model(
    config: BatteryConfig,
    inputs: OptimisationInputs,
    model: pyo.ConcreteModel,
    solver_name: str,
    base_obj: float,
    *,
    per_period_stored: list[float] | None = None,
    horizon_marg: dict[str, float] | None = None,
    warnings: list[str] | None = None,
) -> OptimisationResult:
    """Translate a solved Pyomo model into a rich :class:`OptimisationResult`.

    Shared by the deterministic and stochastic optimisers.
    """
    periods = inputs.periods
    n = len(periods)
    streams = inputs.revenue_streams
    horizon_marg = horizon_marg or {}

    results: list[PeriodResult] = []
    tot_ws = tot_srv = tot_bm = tot_deg = tot_imb = 0.0
    total_discharge_mwh = 0.0

    terminal_val = model.terminal_value_coeff

    for t in range(n):
        p = periods[t]
        dt = p.duration_hours
        charge = _snap(_v(model, model.charge, t))
        discharge = _snap(_v(model, model.discharge, t))
        up = _snap(_v(model, model.up, t))
        down = _snap(_v(model, model.down, t))
        soc_begin = max(_v(model, model.soc, t), 0.0)
        soc_end = max(_v(model, model.soc, t + 1), 0.0)

        e_charge = charge * dt
        e_discharge = discharge * dt
        total_discharge_mwh += e_discharge

        if inputs.portfolio_mode:
            wsell = _v(model, model.wsell, t)
            wbuy = _v(model, model.wbuy, t)
            resid_pos = _v(model, model.resid_pos, t)
            resid_neg = _v(model, model.resid_neg, t)
            ws_pnl = p.wholesale_price * (wsell - wbuy) if streams.wholesale else 0.0
            imb_price = p.expected_imbalance_price or 0.0
            imb_settle = imb_price * (resid_pos - resid_neg) if streams.imbalance else 0.0
            imb_penalty = inputs.imbalance_penalty_gbp_per_mwh * (resid_pos + resid_neg)
            imb_cost = imb_penalty - imb_settle  # positive = net cost
            residual_imbalance = resid_pos - resid_neg
            portfolio_position = (p.renewable_generation_mwh or 0.0) + (
                discharge - charge
            ) * dt - (p.demand_obligation_mwh or 0.0)
        else:
            ws_pnl = p.wholesale_price * (discharge - charge) * dt if streams.wholesale else 0.0
            imb_cost = 0.0
            residual_imbalance = None
            portfolio_position = None

        srv_pnl = 0.0
        if streams.upward_availability:
            srv_pnl += p.upward_availability_price * up * dt
        if streams.downward_availability:
            srv_pnl += p.downward_availability_price * down * dt
        bm_pnl = 0.0
        if streams.bm_activation:
            bm_pnl += p.expected_bm_up_margin_gbp_per_mw * up * dt
            bm_pnl += p.expected_bm_down_margin_gbp_per_mw * down * dt
        deg_cost = config.degradation_cost_gbp_per_mwh_throughput * (charge + discharge) * dt

        total_pnl = ws_pnl + srv_pnl + bm_pnl - deg_cost - imb_cost
        tot_ws += ws_pnl
        tot_srv += srv_pnl
        tot_bm += bm_pnl
        tot_deg += deg_cost
        tot_imb += imb_cost

        action = _classify_action(charge, discharge, up, down)
        binding = _binding_constraints(
            config, charge, discharge, soc_begin, soc_end, up, down,
            is_terminal=(t == n - 1), terminal_soc=config.minimum_terminal_soc_mwh,
        )

        marg = MarginalValues()
        if per_period_stored is not None:
            marg.stored_energy_gbp_per_mwh = round(per_period_stored[t], 4)
        if t == 0 and horizon_marg:
            marg.empty_capacity_gbp_per_mwh = round(
                horizon_marg.get("empty_capacity_gbp_per_mwh", 0.0), 4
            )
            marg.charge_power_gbp_per_mw = round(
                horizon_marg.get("charge_power_gbp_per_mw", 0.0), 4
            )
            marg.discharge_power_gbp_per_mw = round(
                horizon_marg.get("discharge_power_gbp_per_mw", 0.0), 4
            )
            if marg.stored_energy_gbp_per_mwh is None:
                marg.stored_energy_gbp_per_mwh = round(
                    horizon_marg.get("stored_energy_gbp_per_mwh", 0.0), 4
                )

        pr = PeriodResult(
            settlement_date=p.settlement_date,
            settlement_period=p.settlement_period,
            start_utc=p.start_utc,
            end_utc=p.end_utc,
            duration_hours=dt,
            wholesale_price=p.wholesale_price,
            wholesale_price_sigma=p.wholesale_price_sigma,
            system_price=p.system_price,
            prob_short=p.prob_short,
            demand_forecast_mw=p.demand_forecast_mw,
            wind_forecast_mw=p.wind_forecast_mw,
            solar_forecast_mw=p.solar_forecast_mw,
            residual_demand_mw=p.net_residual_demand,
            action=action,
            charge_mw=round(charge, 4),
            discharge_mw=round(discharge, 4),
            energy_charged_mwh=round(e_charge, 4),
            energy_discharged_mwh=round(e_discharge, 4),
            beginning_soc_mwh=round(soc_begin, 4),
            ending_soc_mwh=round(soc_end, 4),
            upward_reserved_mw=round(up, 4),
            downward_reserved_mw=round(down, 4),
            wholesale_pnl_gbp=round(ws_pnl, 2),
            service_pnl_gbp=round(srv_pnl, 2),
            bm_activation_pnl_gbp=round(bm_pnl, 2),
            degradation_cost_gbp=round(deg_cost, 2),
            imbalance_cost_gbp=round(imb_cost, 2),
            total_expected_pnl_gbp=round(total_pnl, 2),
            portfolio_position_mwh=round(portfolio_position, 4) if portfolio_position is not None else None,
            residual_imbalance_mwh=round(residual_imbalance, 4) if residual_imbalance is not None else None,
            binding_constraints=binding,
            marginals=marg,
        )
        pr.explanation = explain_period(
            pr, config.maximum_charge_mw, config.maximum_discharge_mw,
            config.round_trip_efficiency,
        )
        results.append(pr)

    soc_terminal = _v(model, model.soc, n)
    terminal_soc_value = terminal_val * soc_terminal
    full_cycles = total_discharge_mwh / config.energy_capacity_mwh if config.energy_capacity_mwh else 0.0
    total_pnl_all = tot_ws + tot_srv + tot_bm - tot_deg - tot_imb + terminal_soc_value

    return OptimisationResult(
        status="optimal",
        solver=solver_name,
        objective_gbp=round(base_obj, 2),
        periods=results,
        total_wholesale_pnl_gbp=round(tot_ws, 2),
        total_service_pnl_gbp=round(tot_srv, 2),
        total_bm_activation_pnl_gbp=round(tot_bm, 2),
        total_degradation_cost_gbp=round(tot_deg, 2),
        total_imbalance_cost_gbp=round(tot_imb, 2),
        terminal_soc_value_gbp=round(terminal_soc_value, 2),
        total_expected_pnl_gbp=round(total_pnl_all, 2),
        full_cycle_equivalents=round(full_cycles, 3),
        horizon=n,
        portfolio_mode=inputs.portfolio_mode,
        risk_aversion=inputs.risk_aversion,
        warnings=warnings or [],
    )
