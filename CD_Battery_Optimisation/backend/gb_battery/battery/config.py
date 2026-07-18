"""Battery physical & economic configuration.

All quantities are in SI-ish market units:

* power  -> MW
* energy -> MWh
* prices -> GBP/MWh (energy) or GBP/MW/h (availability)

Efficiencies are one-way (round-trip = charge_efficiency * discharge_efficiency).
State of charge (SoC) is tracked as *usable stored energy* in MWh.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class BatteryConfig(BaseModel):
    """Configurable battery asset definition.

    The defaults describe a generic 50 MW / 100 MWh (2-hour) grid-scale battery and
    are clearly assumptions, not observed values for any real asset.
    """

    model_config = {"extra": "forbid"}

    name: str = Field(default="Demo 50MW / 100MWh battery")

    # --- Energy ratings ---
    energy_capacity_mwh: float = Field(default=100.0, gt=0)
    minimum_soc_mwh: float = Field(default=0.0, ge=0)
    maximum_soc_mwh: float = Field(default=100.0, gt=0)
    initial_soc_mwh: float = Field(default=50.0, ge=0)

    # --- Power ratings ---
    maximum_charge_mw: float = Field(default=50.0, gt=0)
    maximum_discharge_mw: float = Field(default=50.0, gt=0)

    # --- Efficiencies (one-way, 0<eff<=1) ---
    charge_efficiency: float = Field(default=0.95, gt=0, le=1)
    discharge_efficiency: float = Field(default=0.95, gt=0, le=1)

    # --- Grid connection limits ---
    grid_import_limit_mw: float = Field(default=50.0, gt=0)
    grid_export_limit_mw: float = Field(default=50.0, gt=0)

    # --- Economics ---
    degradation_cost_gbp_per_mwh_throughput: float = Field(default=3.0, ge=0)

    # --- Terminal SoC targets ---
    minimum_terminal_soc_mwh: float = Field(default=20.0, ge=0)
    preferred_terminal_soc_mwh: float = Field(default=50.0, ge=0)
    # Value applied to the *ending* SoC in the objective (GBP/MWh). Represents the
    # opportunity value of energy carried past the horizon; if None a sensible
    # default (median of horizon wholesale price) is used by the optimiser.
    terminal_soc_value_gbp_per_mwh: float | None = Field(default=None)

    # --- Cycling ---
    maximum_cycles_per_day: float | None = Field(default=2.0, gt=0)

    # --- Optional operating band (fraction of capacity) ---
    minimum_operating_soc_pct: float | None = Field(default=None, ge=0, le=1)
    maximum_operating_soc_pct: float | None = Field(default=None, ge=0, le=1)

    # --- Optional ramp limits (MW per Settlement Period) ---
    ramp_up_mw_per_period: float | None = Field(default=None, gt=0)
    ramp_down_mw_per_period: float | None = Field(default=None, gt=0)

    # --- Optional service-duration requirements (hours of sustained delivery) ---
    upward_service_duration_h: float = Field(default=1.0, gt=0)
    downward_service_duration_h: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> BatteryConfig:
        if self.maximum_soc_mwh > self.energy_capacity_mwh + 1e-9:
            raise ValueError("maximum_soc_mwh cannot exceed energy_capacity_mwh")
        if self.minimum_soc_mwh > self.maximum_soc_mwh:
            raise ValueError("minimum_soc_mwh cannot exceed maximum_soc_mwh")
        if not (self.effective_min_soc <= self.initial_soc_mwh <= self.effective_max_soc):
            raise ValueError(
                f"initial_soc_mwh {self.initial_soc_mwh} outside operating band "
                f"[{self.effective_min_soc}, {self.effective_max_soc}]"
            )
        if self.minimum_terminal_soc_mwh > self.effective_max_soc:
            raise ValueError("minimum_terminal_soc_mwh exceeds usable capacity")
        return self

    # ------------------------------------------------------------------ helpers
    @property
    def effective_min_soc(self) -> float:
        """Lower SoC bound after applying the optional operating-band percentage."""
        floor = self.minimum_soc_mwh
        if self.minimum_operating_soc_pct is not None:
            floor = max(floor, self.minimum_operating_soc_pct * self.energy_capacity_mwh)
        return floor

    @property
    def effective_max_soc(self) -> float:
        """Upper SoC bound after applying the optional operating-band percentage."""
        cap = self.maximum_soc_mwh
        if self.maximum_operating_soc_pct is not None:
            cap = min(cap, self.maximum_operating_soc_pct * self.energy_capacity_mwh)
        return cap

    @property
    def round_trip_efficiency(self) -> float:
        return self.charge_efficiency * self.discharge_efficiency

    @property
    def usable_capacity_mwh(self) -> float:
        return self.effective_max_soc - self.effective_min_soc
