"""Scenario Lab: named, reproducible transformations of a base optimiser input.

Each transform returns a *new* ``OptimisationInputs`` so the base case is preserved
for side-by-side comparison. Transforms are deterministic and clearly documented so a
trader can reason about what changed.
"""

from __future__ import annotations

from collections.abc import Callable

from gb_battery.battery.config import BatteryConfig
from gb_battery.optimiser.inputs import OptimisationInputs


def _copy(inputs: OptimisationInputs) -> OptimisationInputs:
    return inputs.model_copy(deep=True)


def _hour(p) -> float:
    return (p.settlement_period - 1) / 2.0


def negative_price_afternoon(inputs: OptimisationInputs, depth: float = 70.0) -> OptimisationInputs:
    """Drive early-afternoon (12:00–15:00) prices sharply negative (solar glut)."""
    out = _copy(inputs)
    for p in out.periods:
        h = _hour(p)
        if 12.0 <= h <= 15.0:
            p.wholesale_price -= depth
    return out


def evening_price_spike(inputs: OptimisationInputs, height: float = 120.0) -> OptimisationInputs:
    """Add a sharp evening peak (17:00–20:00)."""
    out = _copy(inputs)
    for p in out.periods:
        h = _hour(p)
        if 17.0 <= h <= 20.0:
            p.wholesale_price += height
    return out


def falling_wind_tightening_system(inputs: OptimisationInputs) -> OptimisationInputs:
    """Wind collapses through the day; system tightens (higher prob short, imbalance)."""
    out = _copy(inputs)
    n = len(out.periods)
    for i, p in enumerate(out.periods):
        frac = i / max(n - 1, 1)
        if p.wind_forecast_mw is not None:
            p.wind_forecast_mw *= 1.0 - 0.6 * frac
        p.wholesale_price += 40.0 * frac
        p.prob_short = min(0.98, (p.prob_short or 0.5) + 0.4 * frac)
        if p.expected_imbalance_price is not None:
            p.expected_imbalance_price += 60.0 * frac
    return out


def high_upward_service_value(inputs: OptimisationInputs, price: float = 30.0) -> OptimisationInputs:
    out = _copy(inputs)
    for p in out.periods:
        p.upward_availability_price = price
    return out


def high_downward_service_value(inputs: OptimisationInputs, price: float = 30.0) -> OptimisationInputs:
    out = _copy(inputs)
    for p in out.periods:
        p.downward_availability_price = price
    return out


def forecast_reversal(inputs: OptimisationInputs) -> OptimisationInputs:
    """Flip the price shape (what looked cheap is now expensive) — a forecast bust."""
    out = _copy(inputs)
    prices = [p.wholesale_price for p in out.periods]
    hi, lo = max(prices), min(prices)
    for p in out.periods:
        p.wholesale_price = hi + lo - p.wholesale_price
    return out


def extreme_imbalance_event(inputs: OptimisationInputs, price: float = 500.0) -> OptimisationInputs:
    """A short window of extreme system-short imbalance prices."""
    out = _copy(inputs)
    n = len(out.periods)
    mid = n // 2
    for i, p in enumerate(out.periods):
        if mid <= i <= mid + 3:
            p.expected_imbalance_price = price
            p.prob_short = 0.99
    return out


def service_activation(inputs: OptimisationInputs) -> OptimisationInputs:
    """Raise expected BM activation margins (higher chance of being called)."""
    out = _copy(inputs)
    for p in out.periods:
        p.expected_bm_up_margin_gbp_per_mw = 8.0
        p.expected_bm_down_margin_gbp_per_mw = 8.0
    return out


# Battery-state scenarios modify the config rather than the inputs.
def battery_nearly_full(config: BatteryConfig) -> BatteryConfig:
    return config.model_copy(update={"initial_soc_mwh": min(config.effective_max_soc * 0.95, config.effective_max_soc)})


def battery_nearly_empty(config: BatteryConfig) -> BatteryConfig:
    return config.model_copy(update={"initial_soc_mwh": max(config.effective_min_soc + config.effective_max_soc * 0.05, config.effective_min_soc)})


INPUT_SCENARIOS: dict[str, Callable[[OptimisationInputs], OptimisationInputs]] = {
    "negative_price_afternoon": negative_price_afternoon,
    "evening_price_spike": evening_price_spike,
    "falling_wind_tightening_system": falling_wind_tightening_system,
    "high_upward_service_value": high_upward_service_value,
    "high_downward_service_value": high_downward_service_value,
    "forecast_reversal": forecast_reversal,
    "extreme_imbalance_event": extreme_imbalance_event,
    "service_activation": service_activation,
}

CONFIG_SCENARIOS: dict[str, Callable[[BatteryConfig], BatteryConfig]] = {
    "battery_nearly_full": battery_nearly_full,
    "battery_nearly_empty": battery_nearly_empty,
}


def apply_scenario(
    name: str, config: BatteryConfig, inputs: OptimisationInputs
) -> tuple[BatteryConfig, OptimisationInputs]:
    """Apply a named scenario, returning the (possibly) modified config and inputs."""
    if name in INPUT_SCENARIOS:
        return config, INPUT_SCENARIOS[name](inputs)
    if name in CONFIG_SCENARIOS:
        return CONFIG_SCENARIOS[name](config), inputs
    raise ValueError(f"Unknown scenario '{name}'")


ALL_SCENARIOS = list(INPUT_SCENARIOS) + list(CONFIG_SCENARIOS)
