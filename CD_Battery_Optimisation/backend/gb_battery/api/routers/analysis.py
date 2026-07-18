"""Backtest & forecast-validation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from gb_battery.api.schemas import BacktestRequest, ForecastValidateRequest
from gb_battery.backtest import compare_strategies
from gb_battery.demo.sample_data import generate_synthetic_history
from gb_battery.forecast import (
    BaselineForecaster,
    GradientBoostingQuantileForecaster,
    expanding_window_validate,
)

router = APIRouter(prefix="/api", tags=["analysis"])

_FORECASTERS = {
    "lag1d": lambda: BaselineForecaster("lag1d"),
    "lag7d": lambda: BaselineForecaster("lag7d"),
    "roll": lambda: BaselineForecaster("roll"),
    "gbm": lambda: GradientBoostingQuantileForecaster(),
}


@router.post("/backtest")
def backtest_endpoint(req: BacktestRequest) -> dict:
    hist = generate_synthetic_history(days=req.days)
    cmp = compare_strategies(
        req.config, hist, strategies=req.strategies,
        up_availability_price=req.up_availability_price,
        down_availability_price=req.down_availability_price,
    )
    # Provide the deterministic optimiser's equity curve for charting.
    det = cmp["results"].get("deterministic_optimiser")
    equity = []
    if det is not None and not det.ledger.empty:
        cum = det.ledger["total_pnl_gbp"].cumsum()
        equity = [
            {"index": i, "date": str(r["settlement_date"]), "sp": int(r["settlement_period"]),
             "cumulative_pnl": round(float(c), 2)}
            for i, ((_, r), c) in enumerate(zip(det.ledger.iterrows(), cum, strict=True))
        ]
    return {
        "table": cmp["table"],
        "perfect_foresight_pnl_gbp": cmp["perfect_foresight_pnl_gbp"],
        "leakage_audit": cmp["leakage_audit"],
        "equity_curve": equity,
    }


@router.post("/forecast/validate")
def forecast_validate_endpoint(req: ForecastValidateRequest) -> dict:
    hist = generate_synthetic_history(days=req.days)
    reports = []
    for name in req.models:
        if name not in _FORECASTERS:
            raise HTTPException(400, f"Unknown model '{name}'")
        rep = expanding_window_validate(
            hist, _FORECASTERS[name], name,
            n_splits=req.n_splits, test_days=req.test_days,
        )
        reports.append(
            {
                "model": name,
                "mean_mae": round(rep.mean_mae, 3),
                "mean_rmse": round(rep.mean_rmse, 3),
                "mean_pinball": {str(q): round(v, 3) for q, v in rep.mean_pinball().items()},
                "n_folds": len(rep.folds),
                "folds": [
                    {"train_end": f.train_end, "test_start": f.test_start,
                     "test_end": f.test_end, "mae": round(f.mae, 3), "rmse": round(f.rmse, 3)}
                    for f in rep.folds
                ],
            }
        )
    return {"reports": reports}
