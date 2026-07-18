"""Optimisation & scenario endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from gb_battery.api.schemas import OptimiseRequest, ScenarioRequest
from gb_battery.api.services import build_inputs_for_day, snapshot_to_payload
from gb_battery.optimiser.deterministic import optimise
from gb_battery.scenario.lab import ALL_SCENARIOS, apply_scenario
from gb_battery.scenario.stochastic import optimise_stochastic

router = APIRouter(prefix="/api", tags=["optimise"])


def _run(config, inputs, mode, req: OptimiseRequest | None = None, **kw):
    if mode == "deterministic":
        return optimise(config, inputs, compute_marginals=kw.get("compute_marginals", True))
    if mode == "stochastic":
        return optimise_stochastic(
            config, inputs, n_scenarios=kw.get("n_scenarios", 20),
            risk_aversion=kw.get("risk_aversion", 0.0), cvar_alpha=kw.get("cvar_alpha", 0.95),
        )
    if mode == "robust":
        return optimise_stochastic(config, inputs, n_scenarios=kw.get("n_scenarios", 20), robust=True)
    raise HTTPException(400, f"Unknown mode '{mode}'")


@router.post("/optimise")
def optimise_endpoint(req: OptimiseRequest) -> dict:
    try:
        inputs, snap = build_inputs_for_day(
            req.day, req.source, offline=req.offline, streams=req.streams
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Failed to build inputs: {exc}") from exc

    result = _run(
        req.config, inputs, req.mode,
        n_scenarios=req.n_scenarios, risk_aversion=req.risk_aversion,
        cvar_alpha=req.cvar_alpha, compute_marginals=req.compute_marginals,
    )
    return {
        "result": result.model_dump(mode="json"),
        "source": req.source,
        "snapshot": snapshot_to_payload(snap) if snap else None,
    }


@router.get("/scenario/list")
def scenario_list() -> dict:
    return {"scenarios": ALL_SCENARIOS}


@router.post("/scenario")
def scenario_endpoint(req: ScenarioRequest) -> dict:
    if req.scenario_name not in ALL_SCENARIOS:
        raise HTTPException(400, f"Unknown scenario '{req.scenario_name}'")
    try:
        inputs, _ = build_inputs_for_day(req.day, req.source, offline=req.offline, streams=req.streams)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Failed to build inputs: {exc}") from exc

    base = optimise(req.config, inputs, compute_marginals=False)
    scen_cfg, scen_inputs = apply_scenario(req.scenario_name, req.config, inputs)
    scenario = optimise(scen_cfg, scen_inputs, compute_marginals=False)
    return {
        "scenario_name": req.scenario_name,
        "base": base.model_dump(mode="json"),
        "scenario": scenario.model_dump(mode="json"),
    }
