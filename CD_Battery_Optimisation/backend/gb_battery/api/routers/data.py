"""Data explorer & CSV upload endpoints."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from gb_battery.data.providers import CsvUploadProvider
from gb_battery.demo.sample_data import load_sample

router = APIRouter(prefix="/api", tags=["data"])


@router.get("/data/sample")
def sample_preview(limit: int = Query(default=96, ge=1, le=2000)) -> dict:
    """Preview the frozen synthetic sample (source & timestamps included)."""
    df = load_sample().head(limit).copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return {
        "columns": list(df.columns),
        "rows": df.where(pd.notna(df), None).to_dict(orient="records"),
        "total_rows": int(len(load_sample())),
        "note": "Frozen synthetic sample (DataKind.SYNTHETIC) — not observed market data.",
    }


@router.post("/data/upload-prices")
async def upload_prices(file: UploadFile = File(...)) -> dict:
    """Validate an uploaded wholesale-price CSV and echo a preview."""
    content = await file.read()
    try:
        provider = CsvUploadProvider(content)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    from datetime import date

    df = provider.get_prices(date(2025, 1, 15))
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return {
        "n_rows": int(len(df)),
        "preview": df.head(10).where(pd.notna(df), None).to_dict(orient="records"),
        "kind": "assumption",
    }


@router.get("/data/price-template")
def price_template() -> dict:
    """A CSV template for the price-forecast upload."""
    template = "settlement_period,price\n" + "\n".join(f"{sp},0.0" for sp in range(1, 49))
    return {"filename": "price_forecast_template.csv", "content": template}
