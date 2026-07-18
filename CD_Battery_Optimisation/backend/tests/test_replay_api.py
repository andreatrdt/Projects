"""Replay API: session lifecycle, PIT guarantees at the HTTP boundary."""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient
from gb_battery.api.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def finished_replay() -> dict:
    r = client.post(
        "/api/replay/start",
        json={"day": "2025-01-20", "source": "sample", "auto_run": True},
    )
    assert r.status_code == 200
    return r.json()


def test_start_and_run_completes_all_periods(finished_replay: dict) -> None:
    assert finished_replay["complete"] is True
    assert finished_replay["n_periods"] == 48
    assert len(finished_replay["decisions"]) == 48
    assert finished_replay["summary"]["n_settled"] == 48


def test_step_endpoint_advances_incrementally() -> None:
    r = client.post("/api/replay/start", json={"day": "2025-01-20", "source": "sample"})
    rid = r.json()["replay_id"]
    assert r.json()["step_index"] == 0
    r = client.post("/api/replay/step", json={"replay_id": rid, "n_steps": 3})
    assert r.status_code == 200
    assert r.json()["step_index"] == 3
    assert len(r.json()["new_decisions"]) == 3
    r = client.get(f"/api/replay/{rid}")
    assert r.json()["step_index"] == 3


def test_metrics_include_leakage_audit_and_pf_separation(finished_replay: dict) -> None:
    rid = finished_replay["replay_id"]
    r = client.get(f"/api/replay/{rid}/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["leakage_ok"] is True
    assert len(body["leakage_audit"]) == 48
    strategies = {row["strategy"] for row in body["table"]}
    assert {"rolling_forecast", "no_operation", "perfect_foresight"} <= strategies
    pf = next(row for row in body["table"] if row["strategy"] == "perfect_foresight")
    assert "not a tradable strategy" in pf["label"]
    main = next(row for row in body["table"] if row["strategy"] == "rolling_forecast")
    # The rolling strategy must not exceed the perfect-foresight upper bound.
    assert main["realised_pnl_gbp"] <= pf["realised_pnl_gbp"] + 1e-6


def test_forecast_vintages_are_retrievable_per_step(finished_replay: dict) -> None:
    rid = finished_replay["replay_id"]
    r0 = client.get(f"/api/replay/{rid}/forecasts", params={"step": 0})
    r5 = client.get(f"/api/replay/{rid}/forecasts", params={"step": 5})
    assert r0.status_code == r5.status_code == 200
    assert len(r0.json()["rows"]) == 48
    assert len(r5.json()["rows"]) == 43
    assert r0.json()["information_cutoff"] <= r5.json()["information_cutoff"]


def test_inputs_endpoint_exposes_provenance(finished_replay: dict) -> None:
    rid = finished_replay["replay_id"]
    r = client.get(f"/api/replay/{rid}/inputs", params={"step": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["n_observations_visible"] > 0
    provs = {o["provenance"] for o in body["observations_visible"]}
    assert provs <= {"observed", "synthetic"}
    assert all(o["published_at"] <= body["as_of"] for o in body["observations_visible"])
    assert all(f["issued_at"] <= body["as_of"] for f in body["forecasts_used"])


def test_csv_downloads(finished_replay: dict) -> None:
    rid = finished_replay["replay_id"]
    for path in [f"/api/replay/{rid}/decisions", f"/api/replay/{rid}/forecasts"]:
        r = client.get(path, params={"format": "csv"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        assert len(r.text.splitlines()) > 1


def test_unknown_replay_id_is_404() -> None:
    assert client.get("/api/replay/nope").status_code == 404


def test_historical_replay_rejects_today_on_elexon() -> None:
    r = client.post(
        "/api/replay/start",
        json={"day": date.today().isoformat(), "source": "elexon"},
    )
    assert r.status_code == 400
    assert "live" in r.json()["detail"].lower()


def test_live_paper_trading_never_reveals_future_actuals() -> None:
    r = client.post("/api/live/optimise", json={"source": "synthetic"})
    assert r.status_code == 200
    body = r.json()
    now = body["now_utc"]
    # Executed decisions cover only periods that had completed by `now`.
    for d in body["decisions"]:
        assert d["end_utc"] <= now
        if d["settlement_status"] == "pending":
            assert d["actual_price"] is None
            assert d["realised_pnl_gbp"] is None
    # The forward proposal is forecast-only: no actual field exists on it.
    if body["forward_proposal"] is not None:
        executed_sps = {d["settlement_period"] for d in body["decisions"]}
        for p in body["forward_proposal"]:
            assert p["settlement_period"] not in executed_sps
            assert "actual_price" not in p
    assert "paper" in body["disclaimer"].lower()
    # Expected (future) and realised (settled past) P&L are reported separately.
    assert "realised_pnl_gbp" in body["summary"]
