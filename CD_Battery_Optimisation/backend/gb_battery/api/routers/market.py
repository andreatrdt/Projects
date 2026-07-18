"""Market overview & configuration endpoints."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query

from gb_battery.api.services import snapshot_to_payload
from gb_battery.battery.config import BatteryConfig
from gb_battery.data.cache import ParquetCache
from gb_battery.data.market_snapshot import build_market_snapshot
from gb_battery.data.settings import DataSettings, get_settings

router = APIRouter(prefix="/api", tags=["market"])


@router.get("/config/default")
def default_config() -> dict:
    return BatteryConfig().model_dump(mode="json")


@router.get("/market/snapshot")
def market_snapshot(
    day: date = Query(default=date(2025, 1, 14)),
    offline: bool = Query(default=False),
) -> dict:
    settings = DataSettings(offline=offline)
    try:
        snap = build_market_snapshot(day, settings=settings)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Snapshot failed: {exc}") from exc
    return snapshot_to_payload(snap)


@router.get("/market/status")
def data_status() -> dict:
    """Last successful update per cached source (for the data-freshness panel)."""
    settings = get_settings()
    cache = ParquetCache(settings)
    entries = cache.all_entries()
    return {
        "offline": settings.offline,
        "stale_after_hours": settings.stale_after_hours,
        "sources": [
            {
                "source": e.source,
                "key": e.key,
                "rows": e.rows,
                "retrieved_at": e.retrieved_at.isoformat() if e.retrieved_at else None,
                "stale": e.is_stale(settings.stale_after_hours),
            }
            for e in entries
        ],
    }
