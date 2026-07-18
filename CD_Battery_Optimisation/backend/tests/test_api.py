"""API smoke tests via FastAPI TestClient."""

from __future__ import annotations

import warnings

from fastapi.testclient import TestClient
from gb_battery.api.main import app

warnings.filterwarnings("ignore")
client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_default_config():
    r = client.get("/api/config/default")
    assert r.status_code == 200
    assert r.json()["energy_capacity_mwh"] == 100.0


def test_optimise_synthetic_returns_schedule():
    r = client.post("/api/optimise", json={"day": "2025-01-15", "source": "synthetic", "compute_marginals": False})
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["status"] == "optimal"
    assert len(result["periods"]) == 48
    # Every period reports a feasible SoC.
    for p in result["periods"]:
        assert 0.0 - 1e-6 <= p["ending_soc_mwh"] <= 100.0 + 1e-6


def test_optimise_sample_source():
    r = client.post("/api/optimise", json={"day": "2025-01-06", "source": "sample", "compute_marginals": False})
    assert r.status_code == 200
    assert r.json()["result"]["status"] == "optimal"


def test_scenario_list_and_apply():
    names = client.get("/api/scenario/list").json()["scenarios"]
    assert "negative_price_afternoon" in names
    r = client.post("/api/scenario", json={
        "day": "2025-01-15", "source": "synthetic", "scenario_name": "negative_price_afternoon"})
    assert r.status_code == 200
    body = r.json()
    assert body["base"]["status"] == "optimal"
    assert body["scenario"]["status"] == "optimal"


def test_backtest_endpoint():
    r = client.post("/api/backtest", json={"days": 8})
    assert r.status_code == 200
    body = r.json()
    strategies = {row["strategy"] for row in body["table"]}
    assert "perfect_foresight" in strategies
    assert body["leakage_audit"][0]["ok"] is True


def test_forecast_validate_endpoint():
    r = client.post("/api/forecast/validate", json={"days": 24, "models": ["lag1d", "roll"], "n_splits": 2})
    assert r.status_code == 200
    assert len(r.json()["reports"]) == 2


def test_upload_price_csv_validation():
    csv = b"settlement_period,price\n1,10\n2,20\n"
    r = client.post("/api/data/upload-prices", files={"file": ("p.csv", csv, "text/csv")})
    assert r.status_code == 200
    assert r.json()["n_rows"] == 2

    bad = b"period,value\n1,2\n"
    r2 = client.post("/api/data/upload-prices", files={"file": ("p.csv", bad, "text/csv")})
    assert r2.status_code == 422
