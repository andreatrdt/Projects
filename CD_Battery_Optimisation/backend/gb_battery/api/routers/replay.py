"""Replay & live paper-trading endpoints.

Modes served here:

* **Historical Replay** — chronological simulation of a completed day; every
  decision uses only information published at or before its gate.
* **Live Paper Trading** — today: completed periods are replayed and settled,
  the remainder is a forward proposal at ``as_of = now``. Future actuals are
  never present in responses.
* **Perfect Foresight** — exposed *only* inside ``/metrics`` as a labelled
  benchmark, never mixed into the rolling strategy's results.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, date, datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from gb_battery.battery.config import BatteryConfig
from gb_battery.replay.benchmarks import compare_replay_strategies
from gb_battery.replay.engine import (
    EXECUTION_ASSUMPTION,
    ReplayEngine,
    ReplayOptions,
)
from gb_battery.replay.pit import _utc
from gb_battery.replay.session import REGISTRY, ReplaySession
from gb_battery.settlement import LONDON

router = APIRouter(prefix="/api", tags=["replay"])


class ReplayStartRequest(BaseModel):
    config: BatteryConfig = Field(default_factory=BatteryConfig)
    day: date = date(2025, 1, 20)
    source: str = "sample"  # sample | synthetic | elexon
    strategy: str = "rolling_forecast"  # rolling_forecast | rolling_threshold
    history_days: int = Field(default=14, ge=3, le=60)
    mid_lag_minutes: int = Field(default=10, ge=0, le=120)
    seed: int = 42
    auto_run: bool = False


class ReplayStepRequest(BaseModel):
    replay_id: str
    n_steps: int = Field(default=1, ge=1, le=50)


class ReplayRunRequest(BaseModel):
    replay_id: str


class LiveOptimiseRequest(BaseModel):
    config: BatteryConfig = Field(default_factory=BatteryConfig)
    source: str = "elexon"
    strategy: str = "rolling_forecast"
    history_days: int = Field(default=14, ge=3, le=60)
    mid_lag_minutes: int = Field(default=10, ge=0, le=120)
    seed: int = 42


def _get_session(replay_id: str) -> ReplaySession:
    session = REGISTRY.get(replay_id)
    if session is None:
        raise HTTPException(404, f"Unknown replay '{replay_id}' (sessions are in-memory).")
    return session


def _status_payload(session: ReplaySession, now: datetime | None = None) -> dict:
    eng = session.engine
    n = len(eng.periods)
    complete = eng.is_complete(now)
    next_per = eng.periods[eng.step_index] if eng.step_index < n else None
    return {
        "replay_id": session.replay_id,
        "mode": session.mode,
        "day": eng.day.isoformat(),
        "options": eng.options.model_dump(),
        "n_periods": n,
        "step_index": eng.step_index,
        "complete": complete,
        "soc_mwh": round(eng.soc, 4),
        "next_settlement_period": next_per.settlement_period if next_per else None,
        "next_period_start_utc": next_per.start_utc.isoformat() if next_per else None,
        "summary": eng.realised_summary(),
        "warnings": eng.warnings,
        "execution_assumption": EXECUTION_ASSUMPTION,
    }


@router.post("/replay/start")
def replay_start(req: ReplayStartRequest) -> dict:
    if req.day >= date.today() and req.source == "elexon":
        raise HTTPException(
            400,
            "Historical replay needs a completed day; use /api/live/optimise for today.",
        )
    options = ReplayOptions(
        source=req.source,
        strategy=req.strategy,
        history_days=req.history_days,
        mid_lag_minutes=req.mid_lag_minutes,
        seed=req.seed,
        live=False,
    )
    try:
        engine = ReplayEngine(req.config, req.day, options)
    except Exception as exc:  # noqa: BLE001 — data-source failures become clear API errors
        raise HTTPException(
            502,
            f"Could not build the point-in-time store from '{req.source}': {exc}. "
            "The 'sample' and 'synthetic' sources work offline.",
        ) from exc
    session = REGISTRY.create(engine, mode="historical")
    if req.auto_run:
        engine.run()
    payload = _status_payload(session)
    if req.auto_run:
        payload["decisions"] = [d.model_dump(mode="json") for d in engine.decisions]
    return payload


@router.post("/replay/step")
def replay_step(req: ReplayStepRequest) -> dict:
    session = _get_session(req.replay_id)
    new = session.engine.run(max_steps=req.n_steps)
    payload = _status_payload(session)
    payload["new_decisions"] = [d.model_dump(mode="json") for d in new]
    return payload


@router.post("/replay/run")
def replay_run(req: ReplayRunRequest) -> dict:
    session = _get_session(req.replay_id)
    session.engine.run()
    payload = _status_payload(session)
    payload["decisions"] = [d.model_dump(mode="json") for d in session.engine.decisions]
    return payload


@router.get("/replay/{replay_id}")
def replay_status(replay_id: str) -> dict:
    return _status_payload(_get_session(replay_id))


@router.get("/replay/{replay_id}/decisions")
def replay_decisions(replay_id: str, format: str = Query(default="json")):
    session = _get_session(replay_id)
    decisions = session.engine.decisions
    if format == "csv":
        buf = io.StringIO()
        cols = [
            "step", "settlement_date", "settlement_period", "as_of", "forecast_price",
            "forecast_q10", "forecast_q90", "forecast_basis", "action", "charge_mw",
            "discharge_mw", "soc_before_mwh", "soc_after_mwh",
            "expected_immediate_pnl_gbp", "actual_price", "realised_pnl_gbp",
            "forecast_error", "settlement_status",
        ]
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in decisions:
            w.writerow({k: v for k, v in d.model_dump(mode="json").items() if k in cols})
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")
    return {"decisions": [d.model_dump(mode="json") for d in decisions]}


@router.get("/replay/{replay_id}/forecasts")
def replay_forecasts(
    replay_id: str,
    step: int | None = Query(default=None, description="Vintage index; default latest"),
    format: str = Query(default="json"),
):
    session = _get_session(replay_id)
    vintages = session.engine.vintages
    if not vintages:
        return {"step": None, "vintage": None, "n_vintages": 0}
    idx = len(vintages) - 1 if step is None else step
    if not 0 <= idx < len(vintages):
        raise HTTPException(404, f"No forecast vintage at step {idx} (have {len(vintages)}).")
    v = vintages[idx]
    rows = [
        {
            "settlement_date": r.settlement_date.isoformat(),
            "settlement_period": r.settlement_period,
            "start_utc": r.start_utc.isoformat(),
            "point": r.point,
            "q10": r.q10,
            "q50": r.q50,
            "q90": r.q90,
            "sigma": r.sigma,
            "basis": r.basis,
            "intraday_bias": r.intraday_bias,
            "provenance": r.provenance.value,
            "demand_forecast_mw": r.demand_forecast_mw,
            "wind_forecast_mw": r.wind_forecast_mw,
            "solar_forecast_mw": r.solar_forecast_mw,
            "fundamentals_provenance": (
                r.fundamentals_provenance.value if r.fundamentals_provenance else None
            ),
        }
        for r in v.rows
    ]
    if format == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")
    return {
        "step": idx,
        "n_vintages": len(vintages),
        "issued_at": v.issued_at.isoformat(),
        "information_cutoff": v.information_cutoff.isoformat(),
        "basis_max_published_at": (
            v.basis_max_published_at.isoformat() if v.basis_max_published_at else None
        ),
        "n_input_observations": v.n_input_observations,
        "n_input_forecasts": v.n_input_forecasts,
        "rows": rows,
        "warnings": v.warnings,
    }


@router.get("/replay/{replay_id}/metrics")
def replay_metrics(replay_id: str) -> dict:
    session = _get_session(replay_id)
    eng = session.engine
    comparison = compare_replay_strategies(eng)
    # Leakage audit: prove every decision's inputs predate its gate.
    audit = []
    for d in eng.decisions:
        basis_ok = d.basis_max_published_at is None or _utc(d.basis_max_published_at) <= _utc(d.as_of)
        settle_ok = (
            d.actual_price_available_at is None
            or _utc(d.actual_price_available_at) > _utc(d.as_of)
            or d.charge_mw + d.discharge_mw < 1e-9  # idle decisions can't have used it anyway
        )
        audit.append(
            {
                "step": d.step,
                "settlement_period": d.settlement_period,
                "as_of": d.as_of.isoformat(),
                "basis_max_published_at": (
                    d.basis_max_published_at.isoformat() if d.basis_max_published_at else None
                ),
                "inputs_predate_decision": basis_ok,
                "outturn_published_after_decision": settle_ok,
            }
        )
    comparison["leakage_audit"] = audit
    comparison["leakage_ok"] = all(
        a["inputs_predate_decision"] and a["outturn_published_after_decision"] for a in audit
    )
    return comparison


@router.get("/replay/{replay_id}/inputs")
def replay_inputs(
    replay_id: str,
    step: int | None = Query(default=None),
    format: str = Query(default="json"),
    max_observations: int = Query(default=800, ge=1, le=5000),
):
    """The exact PIT-visible rows behind one decision (forecasts + observations)."""
    session = _get_session(replay_id)
    eng = session.engine
    if not eng.decisions:
        raise HTTPException(404, "No decisions yet — step the replay first.")
    idx = len(eng.decisions) - 1 if step is None else step
    if not 0 <= idx < len(eng.decisions):
        raise HTTPException(404, f"No decision at step {idx}.")
    d = eng.decisions[idx]
    as_of = d.as_of
    obs = eng.store.observations_at(as_of, "wholesale_price")[-max_observations:]
    obs_rows = [
        {
            "variable": o.variable,
            "settlement_date": o.settlement_date.isoformat(),
            "settlement_period": o.settlement_period,
            "value": o.value,
            "published_at": o.published_at.isoformat(),
            "source": o.source,
            "provenance": o.provenance.value,
            "publication_reconstructed": o.publication_reconstructed,
        }
        for o in obs
    ]
    v = eng.vintages[idx]
    fc_rows = [
        {
            "variable": "wholesale_price",
            "settlement_date": r.settlement_date.isoformat(),
            "settlement_period": r.settlement_period,
            "point": r.point,
            "q10": r.q10,
            "q90": r.q90,
            "basis": r.basis,
            "provenance": r.provenance.value,
            "issued_at": v.issued_at.isoformat(),
        }
        for r in v.rows
    ]
    if format == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(
            buf,
            fieldnames=[
                "record_type", "variable", "settlement_date", "settlement_period",
                "value", "published_at", "source", "provenance",
            ],
            extrasaction="ignore",
        )
        w.writeheader()
        for o in obs_rows:
            w.writerow({"record_type": "observation", **o})
        for f in fc_rows:
            w.writerow(
                {
                    "record_type": "forecast",
                    "variable": f["variable"],
                    "settlement_date": f["settlement_date"],
                    "settlement_period": f["settlement_period"],
                    "value": f["point"],
                    "published_at": f["issued_at"],
                    "source": "pit_forecaster",
                    "provenance": f["provenance"],
                }
            )
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")
    return {
        "step": idx,
        "as_of": as_of.isoformat(),
        "settlement_period": d.settlement_period,
        "observations_visible": obs_rows,
        "n_observations_visible": len(obs_rows),
        "forecasts_used": fc_rows,
        "store_notes": eng.store.notes,
    }


@router.post("/live/optimise")
def live_optimise(req: LiveOptimiseRequest) -> dict:
    """Live paper trading for today: settle the past, propose the future.

    Completed periods are replayed and settled with actual outturns; periods at
    or after ``now`` carry forecasts only — their actual values do not exist
    here and are reported as null.
    """
    now = datetime.now(tz=UTC)
    today = now.astimezone(LONDON).date()  # GB settlement date is the local calendar day
    options = ReplayOptions(
        source=req.source,
        strategy=req.strategy,
        history_days=req.history_days,
        mid_lag_minutes=req.mid_lag_minutes,
        seed=req.seed,
        live=True,
    )
    try:
        engine = ReplayEngine(req.config, today, options)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            502,
            f"Could not build the live point-in-time store from '{req.source}': {exc}. "
            "Try source='synthetic' for an offline demonstration.",
        ) from exc
    session = REGISTRY.create(engine, mode="live")
    engine.run(now=now)  # replays only periods completed & published before now

    proposal = engine.propose(as_of=now)
    vintage_payload = None
    proposed_payload = None
    if proposal is not None:
        vintage, proposed = proposal
        proposed_payload = [p.model_dump(mode="json") for p in proposed]
        vintage_payload = {
            "issued_at": vintage.issued_at.isoformat(),
            "information_cutoff": vintage.information_cutoff.isoformat(),
            "rows": [
                {
                    "settlement_period": r.settlement_period,
                    "start_utc": r.start_utc.isoformat(),
                    "point": r.point,
                    "q10": r.q10,
                    "q90": r.q90,
                    "basis": r.basis,
                    "provenance": r.provenance.value,
                }
                for r in vintage.rows
            ],
        }

    payload = _status_payload(session, now=now)
    payload["now_utc"] = now.isoformat()
    payload["decisions"] = [d.model_dump(mode="json") for d in engine.decisions]
    payload["forward_proposal"] = proposed_payload
    payload["forward_vintage"] = vintage_payload
    payload["disclaimer"] = (
        "Paper trading only — simulated decisions, no orders are submitted. "
        + EXECUTION_ASSUMPTION
    )
    return payload
