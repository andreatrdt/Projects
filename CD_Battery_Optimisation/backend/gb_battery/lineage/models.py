"""Provenance / lineage models attached to every processed record."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DataKind(StrEnum):
    """How a value should be interpreted — drives colour/labelling in the UI.

    Never render an ``ESTIMATED`` or ``SYNTHETIC`` value with the same styling as an
    ``OBSERVED`` one.
    """

    OBSERVED = "observed"  # actual metered/settled market value
    FORECAST = "forecast"  # a genuine forward-looking forecast (ours or third-party)
    ESTIMATED = "estimated"  # derived/modelled proxy where no observation exists
    ASSUMPTION = "assumption"  # a user-supplied or default assumption
    SYNTHETIC = "synthetic"  # generated demo/scenario data, not from any market


class QualityStatus(StrEnum):
    OK = "ok"
    ESTIMATED = "estimated"
    MISSING = "missing"
    STALE = "stale"
    REVISED = "revised"


class Lineage(BaseModel):
    """Provenance metadata carried alongside a processed record."""

    model_config = {"frozen": True}

    source: str  # e.g. "elexon.MID", "neso.ckan.<resource>", "user_upload", "synthetic"
    kind: DataKind = DataKind.OBSERVED
    quality: QualityStatus = QualityStatus.OK

    # Timestamps kept *separately* so backtests can respect information availability.
    retrieved_at: datetime | None = None  # when we fetched it
    published_at: datetime | None = None  # when the source published it
    event_at: datetime | None = None  # the instant the value refers to

    # Market coordinates (optional — not every record is half-hourly).
    settlement_date: date | None = None
    settlement_period: int | None = None
    delivery_start: datetime | None = None
    delivery_end: datetime | None = None

    unit: str | None = None
    version: str | None = None  # revision / dataset version indicator
    note: str | None = None

    def available_at(self, decision_time: datetime) -> bool:
        """True if this value was already published at ``decision_time``.

        Used by the leakage audit: a feature must not be used at time *t* unless its
        ``published_at`` (falling back to ``event_at``) was no later than *t*.
        """
        stamp = self.published_at or self.event_at
        if stamp is None:
            return True  # cannot prove unavailability; caller should flag separately
        return stamp <= decision_time


class Tagged[T](BaseModel):
    """A value with its lineage attached."""

    value: T
    lineage: Lineage = Field(...)
