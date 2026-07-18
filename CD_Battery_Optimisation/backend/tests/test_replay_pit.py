"""Point-in-time store: availability filtering, vintages and leakage rejection."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from gb_battery.replay.pit import PITDataStore, PITViolation, store_from_synthetic
from gb_battery.replay.records import ForecastRecord, ObservationRecord, Provenance
from gb_battery.settlement import periods_in_day, settlement_periods_for_day

DAY = date(2025, 6, 2)


def _obs(sp: int, value: float, published_at: datetime) -> ObservationRecord:
    periods = settlement_periods_for_day(DAY)
    per = periods[sp - 1]
    return ObservationRecord(
        variable="wholesale_price",
        settlement_date=DAY,
        settlement_period=sp,
        start_utc=per.start_utc,
        end_utc=per.end_utc,
        value=value,
        published_at=published_at,
        source="test",
        provenance=Provenance.OBSERVED,
    )


def _fc(sp: int, value: float, published_at: datetime, quantile=None) -> ForecastRecord:
    per = settlement_periods_for_day(DAY)[sp - 1]
    return ForecastRecord(
        variable="wholesale_price",
        settlement_date=DAY,
        settlement_period=sp,
        start_utc=per.start_utc,
        value=value,
        issued_at=published_at,
        published_at=published_at,
        source="test",
        provenance=Provenance.PUBLISHED_FORECAST,
        quantile=quantile,
    )


def test_observations_filtered_by_published_at() -> None:
    store = PITDataStore()
    t0 = datetime(2025, 6, 2, 10, 0, tzinfo=UTC)
    store.add_observations([_obs(1, 50.0, t0), _obs(2, 60.0, t0 + timedelta(hours=1))])
    visible = store.observations_at(t0 + timedelta(minutes=1), "wholesale_price")
    assert [o.settlement_period for o in visible] == [1]
    visible = store.observations_at(t0 + timedelta(hours=2), "wholesale_price")
    assert [o.settlement_period for o in visible] == [1, 2]


def test_forecasts_at_returns_latest_visible_vintage() -> None:
    store = PITDataStore()
    early = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
    late = datetime(2025, 6, 1, 18, 0, tzinfo=UTC)
    store.add_forecasts([_fc(5, 40.0, early), _fc(5, 45.0, late)])
    # Before the second vintage was published, the first must be served.
    got = store.forecasts_at(datetime(2025, 6, 1, 12, 0, tzinfo=UTC), "wholesale_price", day=DAY)
    assert got[5].value == 40.0
    got = store.forecasts_at(datetime(2025, 6, 1, 19, 0, tzinfo=UTC), "wholesale_price", day=DAY)
    assert got[5].value == 45.0
    # Nothing is visible before any publication.
    got = store.forecasts_at(datetime(2025, 6, 1, 8, 0, tzinfo=UTC), "wholesale_price", day=DAY)
    assert got == {}


def test_check_no_leakage_raises_for_future_record() -> None:
    t0 = datetime(2025, 6, 2, 10, 0, tzinfo=UTC)
    future = _obs(3, 70.0, t0 + timedelta(hours=3))
    with pytest.raises(PITViolation):
        PITDataStore.check_no_leakage([future], t0)
    # And passes when the record predates the decision.
    PITDataStore.check_no_leakage([future], t0 + timedelta(hours=4))


def test_outturn_is_settlement_only_accessor() -> None:
    store = PITDataStore()
    t0 = datetime(2025, 6, 2, 10, 0, tzinfo=UTC)
    store.add_observations([_obs(7, 88.0, t0)])
    rec = store.outturn("wholesale_price", DAY, 7)
    assert rec is not None and rec.value == 88.0
    assert store.outturn("wholesale_price", DAY, 8) is None


def test_synthetic_store_reconstructs_publication_lag() -> None:
    lag = 10
    store = store_from_synthetic(DAY, history_days=3, mid_lag_minutes=lag)
    periods = settlement_periods_for_day(DAY)
    sp1 = periods[0]
    # At SP2's start, SP1's outturn (published end+10min) must NOT be visible...
    visible = store.observations_at(sp1.end_utc, "wholesale_price", day=DAY)
    assert all(o.settlement_period != 1 for o in visible)
    # ...but it is visible after the lag has elapsed.
    visible = store.observations_at(sp1.end_utc + timedelta(minutes=lag), "wholesale_price", day=DAY)
    assert any(o.settlement_period == 1 for o in visible)
    # Reconstructed availability is flagged as such.
    assert all(o.publication_reconstructed for o in store.observations)


def test_synthetic_store_covers_dst_days() -> None:
    for day, n in [(date(2025, 3, 30), 46), (date(2025, 10, 26), 50)]:
        assert periods_in_day(day) == n
        store = store_from_synthetic(day, history_days=2)
        sps = {
            o.settlement_period
            for o in store.observations
            if o.settlement_date == day and o.variable == "wholesale_price"
        }
        assert sps == set(range(1, n + 1))
