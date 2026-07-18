"""GB Settlement Period calendar with correct daylight-saving handling.

A GB *Settlement Date* runs from local (Europe/London) midnight to the next local
midnight. It is divided into *Settlement Periods* (SPs) of 30 real minutes each.

Because of British Summer Time transitions the number of SPs per day is **not**
constant:

* normal day .......... 48 SPs
* spring-forward day .. 46 SPs (clocks 01:00 -> 02:00, one hour skipped)
* autumn-back day ..... 50 SPs (clocks 02:00 -> 01:00, one hour repeated)

Each SP is always exactly 30 minutes of *real* (UTC) time, so the optimiser's
per-period duration ``delta_t`` is always 0.5 h — but code must never assume that
every calendar day contains 48 of them.

Internally everything is stored in UTC; ``Europe/London`` is used only to define
day and period boundaries for market interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
UTC = ZoneInfo("UTC")

SP_MINUTES = 30
SP_DURATION = timedelta(minutes=SP_MINUTES)
SP_DURATION_HOURS = SP_MINUTES / 60.0


@dataclass(frozen=True)
class SettlementPeriod:
    """A single half-hour trading interval.

    Attributes
    ----------
    settlement_date:
        The GB settlement date this period belongs to (local calendar day).
    settlement_period:
        1-indexed period number within the settlement date (1..46/48/50).
    start_utc / end_utc:
        UTC instants bounding the period. ``end_utc - start_utc`` is always 30 min.
    """

    settlement_date: date
    settlement_period: int
    start_utc: datetime
    end_utc: datetime

    @property
    def duration_hours(self) -> float:
        """Real duration in hours, derived from timestamps (always 0.5)."""
        return (self.end_utc - self.start_utc).total_seconds() / 3600.0

    @property
    def start_local(self) -> datetime:
        return self.start_utc.astimezone(LONDON)

    @property
    def end_local(self) -> datetime:
        return self.end_utc.astimezone(LONDON)

    @property
    def label(self) -> str:
        return f"{self.settlement_date.isoformat()} SP{self.settlement_period:02d}"


def _london_midnight_utc(day: date) -> datetime:
    """UTC instant of local midnight starting ``day`` in Europe/London."""
    local = datetime(day.year, day.month, day.day, 0, 0, tzinfo=LONDON)
    return local.astimezone(UTC)


def periods_in_day(day: date) -> int:
    """Number of Settlement Periods in a GB settlement date (46, 48 or 50)."""
    start = _london_midnight_utc(day)
    end = _london_midnight_utc(day + timedelta(days=1))
    return int(round((end - start).total_seconds() / (SP_MINUTES * 60)))


def settlement_periods_for_day(day: date) -> list[SettlementPeriod]:
    """Return every Settlement Period of a settlement date, DST-aware."""
    start = _london_midnight_utc(day)
    n = periods_in_day(day)
    out: list[SettlementPeriod] = []
    for i in range(n):
        s = start + i * SP_DURATION
        out.append(
            SettlementPeriod(
                settlement_date=day,
                settlement_period=i + 1,
                start_utc=s,
                end_utc=s + SP_DURATION,
            )
        )
    return out


def settlement_periods_between(
    start_utc: datetime, horizon_periods: int
) -> list[SettlementPeriod]:
    """Return ``horizon_periods`` consecutive SPs starting at ``start_utc``.

    ``start_utc`` should already be aligned to a half-hour boundary; if not it is
    floored to the enclosing SP. Settlement date and period numbers are derived
    from the Europe/London calendar so they stay correct across DST changes.
    """
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=UTC)
    start_utc = start_utc.astimezone(UTC)
    # Floor to half-hour boundary.
    floored = start_utc.replace(second=0, microsecond=0)
    minute = 0 if floored.minute < 30 else 30
    floored = floored.replace(minute=minute)

    out: list[SettlementPeriod] = []
    cursor = floored
    for _ in range(horizon_periods):
        out.append(_period_for_instant(cursor))
        cursor = cursor + SP_DURATION
    return out


def _period_for_instant(instant_utc: datetime) -> SettlementPeriod:
    """Map a UTC instant to its Settlement Date and Period."""
    local = instant_utc.astimezone(LONDON)
    day = local.date()
    day_start = _london_midnight_utc(day)
    # Guard against the ambiguous/backward hour: if before local midnight of `day`
    # (can happen on the long autumn day) recompute against previous day.
    if instant_utc < day_start:
        day = day - timedelta(days=1)
        day_start = _london_midnight_utc(day)
    index = int(round((instant_utc - day_start).total_seconds() / (SP_MINUTES * 60)))
    return SettlementPeriod(
        settlement_date=day,
        settlement_period=index + 1,
        start_utc=instant_utc,
        end_utc=instant_utc + SP_DURATION,
    )
