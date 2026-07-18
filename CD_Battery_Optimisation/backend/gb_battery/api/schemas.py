"""API request/response models."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.inputs import RevenueStreams


class OptimiseRequest(BaseModel):
    config: BatteryConfig = Field(default_factory=BatteryConfig)
    day: date = date(2025, 1, 15)
    source: str = "synthetic"  # synthetic | sample | elexon
    offline: bool = False
    streams: RevenueStreams = Field(default_factory=RevenueStreams)
    mode: str = "deterministic"  # deterministic | stochastic | robust
    risk_aversion: float = 0.0
    cvar_alpha: float = 0.95
    n_scenarios: int = 20
    compute_marginals: bool = True


class ScenarioRequest(BaseModel):
    config: BatteryConfig = Field(default_factory=BatteryConfig)
    day: date = date(2025, 1, 15)
    source: str = "synthetic"
    offline: bool = False
    scenario_name: str
    streams: RevenueStreams = Field(default_factory=RevenueStreams)


class BacktestRequest(BaseModel):
    config: BatteryConfig = Field(default_factory=BatteryConfig)
    days: int = Field(default=21, ge=2, le=120)
    strategies: list[str] | None = None
    up_availability_price: float = 0.0
    down_availability_price: float = 0.0


class ForecastValidateRequest(BaseModel):
    days: int = Field(default=30, ge=15, le=120)
    models: list[str] = Field(default_factory=lambda: ["lag1d", "roll", "gbm"])
    n_splits: int = 4
    test_days: int = 3
