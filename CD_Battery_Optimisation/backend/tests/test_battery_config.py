"""BatteryConfig validation tests."""

from __future__ import annotations

import pytest
from gb_battery.battery.config import BatteryConfig
from pydantic import ValidationError


def test_defaults_are_consistent():
    cfg = BatteryConfig()
    assert cfg.round_trip_efficiency == pytest.approx(0.95 * 0.95)
    assert cfg.usable_capacity_mwh == 100.0


def test_initial_soc_outside_band_rejected():
    with pytest.raises(ValidationError):
        BatteryConfig(initial_soc_mwh=150.0)


def test_max_soc_cannot_exceed_capacity():
    with pytest.raises(ValidationError):
        BatteryConfig(energy_capacity_mwh=100.0, maximum_soc_mwh=120.0)


def test_operating_band_percentages_tighten_bounds():
    cfg = BatteryConfig(
        energy_capacity_mwh=100.0, maximum_soc_mwh=100.0, minimum_soc_mwh=0.0,
        minimum_operating_soc_pct=0.1, maximum_operating_soc_pct=0.9,
        initial_soc_mwh=50.0,
    )
    assert cfg.effective_min_soc == 10.0
    assert cfg.effective_max_soc == 90.0
    assert cfg.usable_capacity_mwh == 80.0


def test_efficiency_must_be_in_unit_interval():
    with pytest.raises(ValidationError):
        BatteryConfig(charge_efficiency=1.5)
