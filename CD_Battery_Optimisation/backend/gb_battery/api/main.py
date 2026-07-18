"""FastAPI application for the GB Battery Co-Optimisation Terminal.

This is a **research & decision-support** API. It does not place market orders, does
not control any physical asset, and makes no proprietary-data claims.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gb_battery import __version__
from gb_battery.api.routers import analysis, data, market, optimise, replay

app = FastAPI(
    title="GB Battery Co-Optimisation Terminal",
    version=__version__,
    description=(
        "Research/decision-support API for GB battery charge/discharge co-optimisation "
        "across wholesale, balancing-service availability and imbalance value. "
        "Not a live trading or asset-control system."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market.router)
app.include_router(optimise.router)
app.include_router(analysis.router)
app.include_router(data.router)
app.include_router(replay.router)


@app.get("/api/health", tags=["health"])
def health() -> dict:
    return {"status": "ok", "version": __version__, "service": "gb-battery-coopt"}


@app.get("/", tags=["health"])
def root() -> dict:
    return {
        "service": "GB Battery Co-Optimisation Terminal",
        "docs": "/docs",
        "disclaimer": "Research & decision-support only. No live trading, no asset control.",
    }
