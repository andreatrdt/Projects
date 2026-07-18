"""Settlement Period calendar, especially DST-transition day lengths."""

from __future__ import annotations

from datetime import date, datetime

from gb_battery.settlement import (
    SP_DURATION_HOURS,
    UTC,
    periods_in_day,
    settlement_periods_between,
    settlement_periods_for_day,
)


def test_normal_day_has_48_periods():
    assert periods_in_day(date(2025, 1, 15)) == 48
    assert len(settlement_periods_for_day(date(2025, 1, 15))) == 48


def test_spring_forward_day_has_46_periods():
    # UK clocks go forward on 30 March 2025 -> a short day.
    assert periods_in_day(date(2025, 3, 30)) == 46


def test_autumn_back_day_has_50_periods():
    # UK clocks go back on 26 October 2025 -> a long day.
    assert periods_in_day(date(2025, 10, 26)) == 50


def test_every_period_is_30_real_minutes_even_across_dst():
    for day in (date(2025, 3, 30), date(2025, 10, 26)):
        for sp in settlement_periods_for_day(day):
            assert abs(sp.duration_hours - SP_DURATION_HOURS) < 1e-9


def test_period_numbering_is_sequential_and_dst_aware():
    periods = settlement_periods_for_day(date(2025, 10, 26))
    assert [p.settlement_period for p in periods] == list(range(1, 51))
    # Consecutive UTC starts differ by exactly 30 minutes.
    for a, b in zip(periods, periods[1:], strict=False):
        assert (b.start_utc - a.start_utc).total_seconds() == 1800


def test_mwh_conversion_uses_period_duration():
    # 50 MW for one 30-minute settlement period == 25 MWh.
    periods = settlement_periods_between(datetime(2025, 1, 15, tzinfo=UTC), 1)
    energy_mwh = 50.0 * periods[0].duration_hours
    assert energy_mwh == 25.0


def test_settlement_periods_between_crosses_midnight():
    periods = settlement_periods_between(datetime(2025, 1, 15, 23, 0, tzinfo=UTC), 4)
    # SP47, SP48 of the 15th then SP1, SP2 of the 16th.
    assert periods[0].settlement_period == 47
    assert periods[2].settlement_date == date(2025, 1, 16)
    assert periods[2].settlement_period == 1
