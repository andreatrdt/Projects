"""Rolling replay engine: chronology, physics, settlement and reproducibility."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from gb_battery.battery.config import BatteryConfig
from gb_battery.replay.engine import ReplayEngine, ReplayOptions
from gb_battery.replay.pit import PITDataStore
from gb_battery.replay.records import ForecastRecord, ObservationRecord, Provenance
from gb_battery.settlement import settlement_periods_for_day

DAY = date(2025, 6, 2)
LAG_MIN = 10


def _tiny_battery() -> BatteryConfig:
    """1 MWh / 2 MW lossless battery with no frictions — hand-computable."""
    return BatteryConfig(
        name="test 2MW/1MWh",
        energy_capacity_mwh=1.0,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=1.0,
        initial_soc_mwh=0.0,
        maximum_charge_mw=2.0,
        maximum_discharge_mw=2.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        grid_import_limit_mw=2.0,
        grid_export_limit_mw=2.0,
        degradation_cost_gbp_per_mwh_throughput=0.0,
        minimum_terminal_soc_mwh=0.0,
        preferred_terminal_soc_mwh=0.0,
        terminal_soc_value_gbp_per_mwh=0.0,
        maximum_cycles_per_day=None,
    )


def _manual_store(
    forecasts: list[float], actuals: list[float]
) -> tuple[PITDataStore, list]:
    """A store for the first ``len(forecasts)`` SPs of DAY.

    Day-ahead forecasts published the prior day; actuals published SP end + lag.
    """
    periods = settlement_periods_for_day(DAY)[: len(forecasts)]
    store = PITDataStore()
    day_ahead = periods[0].start_utc - timedelta(hours=12)
    store.add_forecasts(
        [
            ForecastRecord(
                variable="wholesale_price",
                settlement_date=DAY,
                settlement_period=p.settlement_period,
                start_utc=p.start_utc,
                value=f,
                issued_at=day_ahead,
                published_at=day_ahead,
                source="test.day_ahead",
                provenance=Provenance.PUBLISHED_FORECAST,
            )
            for p, f in zip(periods, forecasts, strict=True)
        ]
    )
    store.add_observations(
        [
            ObservationRecord(
                variable="wholesale_price",
                settlement_date=DAY,
                settlement_period=p.settlement_period,
                start_utc=p.start_utc,
                end_utc=p.end_utc,
                value=a,
                published_at=p.end_utc + timedelta(minutes=LAG_MIN),
                source="test.MID",
                provenance=Provenance.OBSERVED,
            )
            for p, a in zip(periods, actuals, strict=True)
        ]
    )
    return store, periods


def _manual_engine(forecasts, actuals, config=None) -> ReplayEngine:
    store, periods = _manual_store(forecasts, actuals)
    return ReplayEngine(
        config or _tiny_battery(),
        DAY,
        ReplayOptions(source="synthetic", mid_lag_minutes=LAG_MIN),
        store=store,
        periods=periods,
    )


class TestManualThreePeriodCase:
    """Forecasts [10, 100, 20], actuals [12, 90, 25] — verifiable by hand.

    SP1: cheap → charge 2 MW (fills the 1 MWh battery), realised cost 12·1 = £12.
    SP2: dear → discharge 2 MW (empties it), realised revenue 90·1 = £90.
    SP3: nothing left worth doing → IDLE, realised £0. Total = £78.
    """

    @pytest.fixture(scope="class")
    def decisions(self):
        eng = _manual_engine([10.0, 100.0, 20.0], [12.0, 90.0, 25.0])
        eng.run()
        return eng.decisions, eng.realised_summary()

    def test_expected_action_sequence(self, decisions) -> None:
        recs, _ = decisions
        assert [d.action for d in recs] == ["CHARGE", "DISCHARGE", "IDLE"]
        assert recs[0].charge_mw == pytest.approx(2.0)
        assert recs[1].discharge_mw == pytest.approx(2.0)

    def test_soc_trajectory(self, decisions) -> None:
        recs, _ = decisions
        assert [d.soc_before_mwh for d in recs] == pytest.approx([0.0, 1.0, 0.0])
        assert [d.soc_after_mwh for d in recs] == pytest.approx([1.0, 0.0, 0.0])

    def test_realised_pnl_uses_actual_prices(self, decisions) -> None:
        recs, summary = decisions
        assert recs[0].realised_pnl_gbp == pytest.approx(-12.0)  # bought 1 MWh at 12
        assert recs[1].realised_pnl_gbp == pytest.approx(90.0)  # sold 1 MWh at 90
        assert summary["realised_pnl_gbp"] == pytest.approx(78.0)

    def test_expected_pnl_uses_forecast_prices(self, decisions) -> None:
        recs, _ = decisions
        assert recs[0].expected_immediate_pnl_gbp == pytest.approx(-10.0)
        assert recs[1].expected_immediate_pnl_gbp == pytest.approx(100.0)

    def test_forecast_errors_settled_even_when_idle(self, decisions) -> None:
        recs, _ = decisions
        # SP1: forecast 10 vs actual 12 → −2.
        # SP2: SP1's outturn publishes at SP1 end + 10 min, i.e. AFTER the SP2
        #      gate, so the forecast stays 100 → error +10 (the lag matters).
        # SP3: SP1's outturn IS visible by now; the intraday bias correction
        #      (+2 surprise) lifts the forecast to 22 → error 22 − 25 = −3.
        assert [d.forecast_error for d in recs] == pytest.approx([-2.0, 10.0, -3.0])
        assert recs[2].forecast_price == pytest.approx(22.0)


def test_no_input_published_after_decision_gate() -> None:
    eng = _manual_engine([10.0, 100.0, 20.0], [12.0, 90.0, 25.0])
    eng.run()
    for d in eng.decisions:
        assert d.basis_max_published_at is not None
        assert d.basis_max_published_at <= d.as_of
        # The settled outturn only became available after the decision was made.
        assert d.actual_price_available_at > d.as_of


def test_only_first_action_executed_and_replanned() -> None:
    eng = _manual_engine([10.0, 100.0, 20.0, 5.0], [12.0, 90.0, 25.0, 4.0])
    eng.run()
    # One vintage and one executed decision per period; horizons shrink by one.
    assert len(eng.vintages) == 4
    assert [len(v.rows) for v in eng.vintages] == [4, 3, 2, 1]
    for d in eng.decisions:
        assert d.proposed_schedule[0].settlement_period == d.settlement_period
        # Later proposed periods were NOT executed at this step.
        executed_sps = {x.settlement_period for x in [d]}
        assert executed_sps == {d.settlement_period}


def test_soc_carries_forward_between_steps() -> None:
    eng = _manual_engine([10.0, 100.0, 20.0], [12.0, 90.0, 25.0])
    eng.run()
    for prev, nxt in zip(eng.decisions, eng.decisions[1:], strict=False):
        assert nxt.soc_before_mwh == pytest.approx(prev.soc_after_mwh)


def test_replay_is_reproducible() -> None:
    opts = ReplayOptions(source="synthetic", seed=7, history_days=5)
    a = ReplayEngine(BatteryConfig(), date(2025, 2, 10), opts)
    b = ReplayEngine(BatteryConfig(), date(2025, 2, 10), opts)
    a.run()
    b.run()
    assert [d.model_dump() for d in a.decisions] == [d.model_dump() for d in b.decisions]


@pytest.mark.parametrize(
    ("day", "n"),
    [(date(2025, 3, 30), 46), (date(2025, 6, 2), 48), (date(2025, 10, 26), 50)],
)
def test_dst_days_have_correct_step_counts(day: date, n: int) -> None:
    eng = ReplayEngine(
        BatteryConfig(), day, ReplayOptions(source="synthetic", history_days=3)
    )
    eng.run()
    assert len(eng.decisions) == n
    assert {d.settlement_period for d in eng.decisions} == set(range(1, n + 1))


def test_degradation_charged_on_throughput() -> None:
    cfg = _tiny_battery().model_copy(update={"degradation_cost_gbp_per_mwh_throughput": 4.0})
    eng = _manual_engine([10.0, 100.0], [12.0, 90.0], config=cfg)
    eng.run()
    # 1 MWh charged + 1 MWh discharged → £8 degradation in total.
    assert sum(d.degradation_cost_gbp for d in eng.decisions) == pytest.approx(8.0)
    assert eng.realised_summary()["realised_pnl_gbp"] == pytest.approx(78.0 - 8.0)


def test_cycle_budget_is_consumed_not_reset() -> None:
    cfg = BatteryConfig(maximum_cycles_per_day=1.0, degradation_cost_gbp_per_mwh_throughput=0.0)
    eng = ReplayEngine(cfg, date(2025, 2, 10), ReplayOptions(source="synthetic", history_days=5))
    eng.run()
    assert eng.discharged_mwh <= cfg.maximum_cycles_per_day * cfg.energy_capacity_mwh + 1e-6


def test_live_mode_does_not_execute_unfinished_periods() -> None:
    store, periods = _manual_store([10.0, 100.0, 20.0], [12.0, 90.0, 25.0])
    eng = ReplayEngine(
        _tiny_battery(),
        DAY,
        ReplayOptions(source="synthetic", live=True, mid_lag_minutes=LAG_MIN),
        store=store,
        periods=periods,
    )
    # "Now" is just after SP2 completes but before its outturn is published.
    now = periods[1].end_utc + timedelta(minutes=1)
    eng.run(now=now)
    assert len(eng.decisions) == 2  # SP3 has not completed → not executed
    assert eng.decisions[0].settlement_status == "settled"
    assert eng.decisions[1].settlement_status == "pending"  # outturn not yet published
    assert eng.decisions[1].actual_price is None
    assert eng.decisions[1].realised_pnl_gbp is None
    # The forward proposal covers the remaining period without any actuals.
    proposal = eng.propose(as_of=now)
    assert proposal is not None
    vintage, proposed = proposal
    assert [p.settlement_period for p in proposed] == [3]
