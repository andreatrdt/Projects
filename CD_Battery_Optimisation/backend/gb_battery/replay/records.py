"""Point-in-time record types.

Two record shapes cover everything the replay engine may look at:

* :class:`ObservationRecord` — a realised value (outturn). It *exists* from the
  moment the underlying event completes, but it is only *usable* by a decision
  once its ``published_at`` has passed.
* :class:`ForecastRecord` — a forward-looking value for a target Settlement
  Period, stamped with when it was issued and when it became available.

``Provenance`` is deliberately finer-grained than the UI-level ``DataKind``:
it distinguishes a genuinely published third-party forecast from our own model
forecast and from a *reconstructed* availability assumption (e.g. "MID for a
period is available 10 minutes after the period ends" — plausible, documented,
but not a recorded publication timestamp).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class Provenance(StrEnum):
    OBSERVED = "observed"  # realised market/outturn value from a public source
    PUBLISHED_FORECAST = "published_forecast"  # third-party forecast with real publish time
    MODEL_FORECAST = "model_forecast"  # produced by our own point-in-time model
    RECONSTRUCTED = "reconstructed"  # availability/publication time assumed, not recorded
    SYNTHETIC = "synthetic"  # generated demo data, not from any market
    ASSUMED = "assumed"  # user- or default-supplied assumption
    PERFECT_FORESIGHT = "perfect_foresight"  # benchmark-only: uses the realised path


@dataclass(frozen=True)
class ObservationRecord:
    """A realised value for one Settlement Period of one variable."""

    variable: str  # e.g. "wholesale_price", "system_price", "demand_mw"
    settlement_date: date
    settlement_period: int
    start_utc: datetime
    end_utc: datetime
    value: float
    published_at: datetime  # when this value became publicly available
    source: str  # e.g. "elexon.MID", "synthetic"
    provenance: Provenance = Provenance.OBSERVED
    unit: str = ""
    publication_reconstructed: bool = False  # True if published_at is an assumption


@dataclass(frozen=True)
class ForecastRecord:
    """A forecast value for one target Settlement Period of one variable."""

    variable: str
    settlement_date: date
    settlement_period: int
    start_utc: datetime
    value: float
    issued_at: datetime  # when the forecast was generated
    published_at: datetime  # when it became available to a decision-maker
    source: str
    provenance: Provenance = Provenance.PUBLISHED_FORECAST
    unit: str = ""
    quantile: float | None = None  # None = point forecast, else e.g. 0.1/0.5/0.9

    @property
    def horizon_hours(self) -> float:
        """Lead time from issue to delivery start (negative = issued after start)."""
        return (self.start_utc - self.issued_at).total_seconds() / 3600.0
