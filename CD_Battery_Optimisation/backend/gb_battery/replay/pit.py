"""Point-in-time (PIT) data store — the only data gateway during a replay.

Every read is parameterised by ``as_of`` and returns only records whose
``published_at`` is at or before that instant. The one deliberate exception is
:meth:`PITDataStore.outturn`, used to *settle* a decision after its period has
completed; the engine records the outturn's availability time alongside the
realised result so the audit trail stays provable.

Availability assumptions (documented, flagged ``publication_reconstructed``):

* **MID wholesale price** — Elexon publishes MID shortly after each half-hour.
  The API does not return a per-record publish time, so we reconstruct
  availability as ``period end + mid_lag_minutes`` (default 10). Conservative:
  at the decision gate for period *t* (its start), the newest usable price is
  therefore *t-2*.
* **Synthetic outturns** — same rule, so offline replays exercise the exact
  code path used with real data.
* **Day-ahead demand / wind / solar forecasts** — Elexon returns a real
  ``publishTime``; it is used verbatim. Synthetic forecasts are stamped
  day-ahead midnight UTC by the sample generator.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from gb_battery.replay.records import ForecastRecord, ObservationRecord, Provenance
from gb_battery.settlement import UTC

MID_LAG_MINUTES_DEFAULT = 10


class PITViolation(RuntimeError):
    """A record published after the decision timestamp reached a decision."""


def _utc(dt: datetime | pd.Timestamp) -> datetime:
    """Coerce to a tz-aware UTC datetime (naive input is assumed UTC)."""
    ts = pd.Timestamp(dt)
    ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
    return ts.to_pydatetime()


@dataclass
class PITDataStore:
    """In-memory PIT store for one replay session."""

    observations: list[ObservationRecord] = field(default_factory=list)
    forecasts: list[ForecastRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ writes
    def add_observations(self, records: Iterable[ObservationRecord]) -> None:
        self.observations.extend(records)

    def add_forecasts(self, records: Iterable[ForecastRecord]) -> None:
        self.forecasts.extend(records)

    # ------------------------------------------------------------------- reads
    def observations_at(
        self,
        as_of: datetime,
        variable: str,
        *,
        day: date | None = None,
    ) -> list[ObservationRecord]:
        """Observations of ``variable`` published at or before ``as_of``."""
        as_of = _utc(as_of)
        out = [
            r
            for r in self.observations
            if r.variable == variable
            and _utc(r.published_at) <= as_of
            and (day is None or r.settlement_date == day)
        ]
        out.sort(key=lambda r: (r.settlement_date, r.settlement_period))
        return out

    def observation_frame_at(self, as_of: datetime, variable: str) -> pd.DataFrame:
        """Tidy frame of PIT-visible observations (for the forecaster)."""
        rows = self.observations_at(as_of, variable)
        return pd.DataFrame(
            {
                "settlement_date": [r.settlement_date for r in rows],
                "settlement_period": [r.settlement_period for r in rows],
                "start_utc": [r.start_utc for r in rows],
                "value": [r.value for r in rows],
                "published_at": [r.published_at for r in rows],
            }
        )

    def forecasts_at(
        self,
        as_of: datetime,
        variable: str,
        *,
        day: date | None = None,
        quantile: float | None = None,
    ) -> dict[int, ForecastRecord]:
        """Latest forecast per target SP with ``published_at <= as_of``.

        "Latest" = greatest ``published_at`` (ties broken by ``issued_at``).
        """
        as_of = _utc(as_of)
        best: dict[int, ForecastRecord] = {}
        for r in self.forecasts:
            if r.variable != variable or r.quantile != quantile:
                continue
            if day is not None and r.settlement_date != day:
                continue
            if _utc(r.published_at) > as_of:
                continue
            cur = best.get(r.settlement_period)
            if cur is None or (
                (_utc(r.published_at), _utc(r.issued_at))
                > (_utc(cur.published_at), _utc(cur.issued_at))
            ):
                best[r.settlement_period] = r
        return best

    # -------------------------------------------------------- settlement reads
    def outturn(self, variable: str, day: date, sp: int) -> ObservationRecord | None:
        """The realised value for (day, sp) regardless of ``as_of``.

        Settlement-only accessor: callers must record the outturn's
        ``published_at`` and must never feed the value back into a decision for
        a period at or before it.
        """
        for r in self.observations:
            if r.variable == variable and r.settlement_date == day and r.settlement_period == sp:
                return r
        return None

    # ------------------------------------------------------------------ audits
    @staticmethod
    def check_no_leakage(
        records: Sequence[ObservationRecord | ForecastRecord], as_of: datetime
    ) -> None:
        """Raise :class:`PITViolation` if any record postdates ``as_of``."""
        as_of = _utc(as_of)
        for r in records:
            if _utc(r.published_at) > as_of:
                raise PITViolation(
                    f"{type(r).__name__} for {r.settlement_date} SP{r.settlement_period} "
                    f"({r.variable}) published {r.published_at} > as_of {as_of}"
                )

    @staticmethod
    def max_published_at(
        records: Sequence[ObservationRecord | ForecastRecord],
    ) -> datetime | None:
        stamps = [_utc(r.published_at) for r in records]
        return max(stamps) if stamps else None


# ---------------------------------------------------------------------- builders


def store_from_history_frame(
    history: pd.DataFrame,
    *,
    source: str,
    provenance: Provenance,
    mid_lag_minutes: int = MID_LAG_MINUTES_DEFAULT,
    forecast_provenance: Provenance | None = None,
) -> PITDataStore:
    """Build a PIT store from a tidy per-period history frame.

    Expected columns: ``settlement_date``, ``settlement_period``, ``start_utc``,
    ``wholesale_price`` (outturn), optionally ``wholesale_price_forecast``,
    ``demand_forecast_mw``, ``wind_forecast_mw``, ``solar_forecast_mw`` and
    ``forecast_published_at``.

    Outturn availability is reconstructed as ``end + mid_lag_minutes`` — the
    same rule applied to live MID data — so synthetic replays exercise the
    identical PIT code path.
    """
    store = PITDataStore()
    lag = timedelta(minutes=mid_lag_minutes)
    fc_prov = forecast_provenance or (
        Provenance.SYNTHETIC if provenance == Provenance.SYNTHETIC else Provenance.PUBLISHED_FORECAST
    )

    obs: list[ObservationRecord] = []
    fcs: list[ForecastRecord] = []
    for _, r in history.iterrows():
        day = pd.Timestamp(r["settlement_date"]).date()
        sp = int(r["settlement_period"])
        start = _utc(r["start_utc"])
        end = start + timedelta(minutes=30)
        price = r.get("wholesale_price")
        if price is not None and not pd.isna(price):
            obs.append(
                ObservationRecord(
                    variable="wholesale_price",
                    settlement_date=day,
                    settlement_period=sp,
                    start_utc=start,
                    end_utc=end,
                    value=float(price),
                    published_at=end + lag,
                    source=f"{source}.MID",
                    provenance=provenance,
                    unit="GBP/MWh",
                    publication_reconstructed=True,
                )
            )
        fc_published = r.get("forecast_published_at")
        if fc_published is None or pd.isna(fc_published):
            # Day-ahead convention: available at 00:00 UTC on the prior day.
            fc_published = datetime(day.year, day.month, day.day, tzinfo=UTC) - timedelta(days=1)
        fc_published = _utc(fc_published)
        for var, col, unit in [
            ("wholesale_price", "wholesale_price_forecast", "GBP/MWh"),
            ("demand_mw", "demand_forecast_mw", "MW"),
            ("wind_mw", "wind_forecast_mw", "MW"),
            ("solar_mw", "solar_forecast_mw", "MW"),
        ]:
            val = r.get(col)
            if val is None or pd.isna(val):
                continue
            fcs.append(
                ForecastRecord(
                    variable=var,
                    settlement_date=day,
                    settlement_period=sp,
                    start_utc=start,
                    value=float(val),
                    issued_at=fc_published,
                    published_at=fc_published,
                    source=f"{source}.day_ahead",
                    provenance=fc_prov,
                    unit=unit,
                )
            )
    store.add_observations(obs)
    store.add_forecasts(fcs)
    store.notes.append(
        f"Outturn availability reconstructed as period end + {mid_lag_minutes} min."
    )
    return store


def store_from_synthetic(
    day: date,
    *,
    history_days: int = 14,
    mid_lag_minutes: int = MID_LAG_MINUTES_DEFAULT,
    seed: int = 42,
) -> PITDataStore:
    """Synthetic PIT store: ``history_days`` of context ending at ``day``.

    Fully offline and deterministic for a given (day, history_days, seed).
    """
    from gb_battery.demo.sample_data import generate_synthetic_history

    start = day - timedelta(days=history_days)
    hist = generate_synthetic_history(start=start, days=history_days + 1, seed=seed)
    return store_from_history_frame(
        hist,
        source="synthetic",
        provenance=Provenance.SYNTHETIC,
        mid_lag_minutes=mid_lag_minutes,
    )


def store_from_sample(
    day: date | None = None,
    *,
    mid_lag_minutes: int = MID_LAG_MINUTES_DEFAULT,
) -> tuple[PITDataStore, date]:
    """PIT store from the frozen bundled sample; returns (store, replay day).

    If ``day`` is missing from the sample the last sample day is used, so the
    demo works with no arguments.
    """
    from gb_battery.demo.sample_data import load_sample

    hist = load_sample()
    days = sorted(pd.to_datetime(hist["settlement_date"]).dt.date.unique())
    if day is None or day not in days:
        day = days[-1]
    store = store_from_history_frame(
        hist,
        source="sample",
        provenance=Provenance.SYNTHETIC,
        mid_lag_minutes=mid_lag_minutes,
    )
    return store, day


def store_from_elexon(
    day: date,
    *,
    history_days: int = 14,
    mid_lag_minutes: int = MID_LAG_MINUTES_DEFAULT,
    settings=None,
    client=None,
) -> PITDataStore:
    """PIT store from live Elexon data for ``day`` plus lag history.

    * MID outturns (target day + prior ``history_days``): availability
      reconstructed as period end + lag (MID has no per-record publish time).
    * Day-ahead demand and wind/solar forecasts: real Elexon ``publishTime``
      used verbatim (``published_forecast`` provenance).

    Raises the underlying error if MID (the essential series) cannot be
    fetched; fundamentals failures degrade to notes.
    """
    from gb_battery.data.elexon import ElexonClient
    from gb_battery.data.settings import DataSettings

    settings = settings or DataSettings()
    client = client or ElexonClient(settings)
    store = PITDataStore()
    lag = timedelta(minutes=mid_lag_minutes)

    frm = datetime(day.year, day.month, day.day, tzinfo=UTC) - timedelta(days=history_days)
    to = datetime(day.year, day.month, day.day, 23, 59, tzinfo=UTC)

    # The MID endpoint rejects ranges over 7 days inclusive — fetch in chunks.
    chunks: list[pd.DataFrame] = []
    cursor = frm
    while cursor <= to:
        chunk_end = min(cursor + timedelta(days=6, hours=23, minutes=59), to)
        chunks.append(client.market_index_data(cursor, chunk_end))
        cursor = chunk_end + timedelta(minutes=1)
    mid = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    mid = mid.dropna(subset=["mid_price"]).drop_duplicates(
        subset=["settlement_date", "settlement_period"], keep="last"
    )
    obs = []
    for _, r in mid.iterrows():
        start = _utc(r["start_utc"])
        end = start + timedelta(minutes=30)
        obs.append(
            ObservationRecord(
                variable="wholesale_price",
                settlement_date=pd.Timestamp(r["settlement_date"]).date(),
                settlement_period=int(r["settlement_period"]),
                start_utc=start,
                end_utc=end,
                value=float(r["mid_price"]),
                published_at=end + lag,
                source="elexon.MID",
                provenance=Provenance.OBSERVED,
                unit="GBP/MWh",
                publication_reconstructed=True,
            )
        )
    store.add_observations(obs)
    store.notes.append(
        f"Elexon MID availability reconstructed as period end + {mid_lag_minutes} min "
        "(the API exposes no per-record publish time)."
    )

    day_frm = datetime(day.year, day.month, day.day, tzinfo=UTC)
    day_to = day_frm.replace(hour=23, minute=59)
    fcs: list[ForecastRecord] = []
    try:
        dem = client.demand_forecast(day_frm, day_to)
        dem = dem.dropna(subset=["national_demand_forecast_mw"]).drop_duplicates(
            "settlement_period", keep="last"
        )
        for _, r in dem.iterrows():
            pub = r.get("published_at")
            reconstructed = pub is None or pd.isna(pub)
            pub = _utc(pub) if not reconstructed else day_frm - timedelta(hours=15)
            fcs.append(
                ForecastRecord(
                    variable="demand_mw",
                    settlement_date=pd.Timestamp(r["settlement_date"]).date(),
                    settlement_period=int(r["settlement_period"]),
                    start_utc=_utc(r["start_utc"]),
                    value=float(r["national_demand_forecast_mw"]),
                    issued_at=pub,
                    published_at=pub,
                    source="elexon.demand_forecast.day_ahead",
                    provenance=(
                        Provenance.RECONSTRUCTED if reconstructed else Provenance.PUBLISHED_FORECAST
                    ),
                    unit="MW",
                )
            )
    except Exception as exc:  # noqa: BLE001 — fundamentals are optional
        store.notes.append(f"Demand forecast unavailable: {exc}")
    try:
        ws = client.wind_solar_forecast(day_frm, day_to).drop_duplicates(
            "settlement_period", keep="last"
        )
        for _, r in ws.iterrows():
            pub = r.get("published_at")
            reconstructed = pub is None or pd.isna(pub)
            pub = _utc(pub) if not reconstructed else day_frm - timedelta(hours=15)
            for var, col in [("wind_mw", "wind_forecast_mw"), ("solar_mw", "solar_forecast_mw")]:
                val = r.get(col)
                if val is None or pd.isna(val):
                    continue
                fcs.append(
                    ForecastRecord(
                        variable=var,
                        settlement_date=pd.Timestamp(r["settlement_date"]).date(),
                        settlement_period=int(r["settlement_period"]),
                        start_utc=_utc(r["start_utc"]),
                        value=float(val),
                        issued_at=pub,
                        published_at=pub,
                        source="elexon.wind_solar_forecast.day_ahead",
                        provenance=(
                            Provenance.RECONSTRUCTED
                            if reconstructed
                            else Provenance.PUBLISHED_FORECAST
                        ),
                        unit="MW",
                    )
                )
    except Exception as exc:  # noqa: BLE001
        store.notes.append(f"Wind/solar forecast unavailable: {exc}")
    store.add_forecasts(fcs)
    return store
