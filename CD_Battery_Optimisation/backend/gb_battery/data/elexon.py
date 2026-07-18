"""Elexon Insights (BMRS) data adapter.

Endpoint paths and field names below were confirmed against the live API
(https://developer.data.elexon.co.uk/ , https://bmrs.elexon.co.uk/api-documentation)
rather than assumed. Each method returns a tidy pandas DataFrame carrying lineage
columns (``source``, ``retrieved_at``, ``published_at``, ``event_at``).

No API key is required; the client is polite (rate-limited, retrying, identified by
User-Agent) and every raw payload can be cached for auditability.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from gb_battery.data.http import DataSourceError, ResilientClient
from gb_battery.data.settings import DataSettings, get_settings
from gb_battery.settlement import UTC

# Canonical Elexon Insights v1 endpoints.
EP_MID = "/balancing/pricing/market-index"
EP_SYSTEM_PRICES_DATE = "/balancing/settlement/system-prices/{date}"
EP_DEMAND_OUTTURN = "/demand/outturn"
EP_DEMAND_FORECAST = "/forecast/demand/day-ahead"
EP_WIND_SOLAR = "/forecast/generation/wind-and-solar/day-ahead"
EP_FUELINST = "/datasets/FUELINST"
EP_BOD = "/datasets/BOD"
EP_BOALF = "/datasets/BOALF"

SOURCE = "elexon"


def _parse_dt(x: str | None) -> datetime | None:
    if not x:
        return None
    return pd.to_datetime(x, utc=True).to_pydatetime()


class ElexonClient:
    """Typed-ish adapter over the Elexon Insights REST API."""

    def __init__(self, settings: DataSettings | None = None, client: ResilientClient | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client or ResilientClient(self.settings)

    # -- low-level ----------------------------------------------------------
    def _get(self, path: str, params: dict | None = None) -> tuple[list[dict], datetime]:
        url = self.settings.elexon_base_url + path
        payload, retrieved_at = self._client.get_json(url, params=params)
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(data, list):
            raise DataSourceError(f"Unexpected Elexon payload shape for {path}")
        return data, retrieved_at

    @staticmethod
    def _lineage(df: pd.DataFrame, retrieved_at: datetime, kind: str = "observed") -> pd.DataFrame:
        df = df.copy()
        df["source"] = f"{SOURCE}.{kind}"
        df["retrieved_at"] = retrieved_at
        return df

    # -- datasets -----------------------------------------------------------
    def market_index_data(self, frm: datetime, to: datetime, provider: str = "APXMIDP") -> pd.DataFrame:
        """Market Index Data (MID): short-term wholesale reference price & volume.

        MID is a *reference* price, not a full EPEX order book (see methodology docs).
        """
        data, ts = self._get(
            EP_MID,
            {"from": _iso(frm), "to": _iso(to), "dataProviders": provider},
        )
        rows = [
            {
                "settlement_date": pd.to_datetime(r["settlementDate"]).date(),
                "settlement_period": int(r["settlementPeriod"]),
                "start_utc": _parse_dt(r["startTime"]),
                "mid_price": _f(r.get("price")),
                "mid_volume": _f(r.get("volume")),
                "data_provider": r.get("dataProvider"),
                "published_at": None,
                "event_at": _parse_dt(r["startTime"]),
            }
            for r in data
        ]
        return self._lineage(pd.DataFrame(rows), ts, "MID")

    def system_prices(self, day: date) -> pd.DataFrame:
        """Imbalance system prices for a settlement date (single-price regime)."""
        data, ts = self._get(EP_SYSTEM_PRICES_DATE.format(date=day.isoformat()))
        rows = [
            {
                "settlement_date": pd.to_datetime(r["settlementDate"]).date(),
                "settlement_period": int(r["settlementPeriod"]),
                "start_utc": _parse_dt(r.get("startTime")),
                "system_sell_price": _f(r.get("systemSellPrice")),
                "system_buy_price": _f(r.get("systemBuyPrice")),
                "net_imbalance_volume": _f(r.get("netImbalanceVolume")),
                "published_at": _parse_dt(r.get("createdDateTime")),
                "event_at": _parse_dt(r.get("startTime")),
            }
            for r in data
        ]
        return self._lineage(pd.DataFrame(rows), ts, "system_prices")

    def demand_outturn(self, frm: date, to: date) -> pd.DataFrame:
        """Initial national + transmission demand outturn (INDO / ITSDO), MW."""
        data, ts = self._get(
            EP_DEMAND_OUTTURN,
            {"settlementDateFrom": frm.isoformat(), "settlementDateTo": to.isoformat()},
        )
        rows = [
            {
                "settlement_date": pd.to_datetime(r["settlementDate"]).date(),
                "settlement_period": int(r["settlementPeriod"]),
                "start_utc": _parse_dt(r.get("startTime")),
                "national_demand_mw": _f(r.get("initialDemandOutturn")),
                "transmission_demand_mw": _f(r.get("initialTransmissionSystemDemandOutturn")),
                "published_at": _parse_dt(r.get("publishTime")),
                "event_at": _parse_dt(r.get("startTime")),
            }
            for r in data
        ]
        return self._lineage(pd.DataFrame(rows), ts, "demand_outturn")

    def demand_forecast(self, frm: datetime, to: datetime) -> pd.DataFrame:
        """Day-ahead national demand forecast, MW."""
        data, ts = self._get(EP_DEMAND_FORECAST, {"from": _iso(frm), "to": _iso(to)})
        rows = [
            {
                "settlement_date": pd.to_datetime(r["settlementDate"]).date(),
                "settlement_period": int(r["settlementPeriod"]),
                "start_utc": _parse_dt(r.get("startTime")),
                "national_demand_forecast_mw": _f(r.get("nationalDemand")),
                "transmission_demand_forecast_mw": _f(r.get("transmissionSystemDemand")),
                "boundary": r.get("boundary"),
                "published_at": _parse_dt(r.get("publishTime")),
                "event_at": _parse_dt(r.get("startTime")),
            }
            for r in data
        ]
        df = pd.DataFrame(rows)
        # Keep the national boundary row per period if a boundary column is present.
        if "boundary" in df.columns and df["boundary"].notna().any():
            df = df.sort_values("settlement_period")
        return self._lineage(df, ts, "demand_forecast")

    def wind_solar_forecast(self, frm: datetime, to: datetime) -> pd.DataFrame:
        """Day-ahead wind & solar forecast (MW), pivoted to wind/solar columns."""
        data, ts = self._get(
            EP_WIND_SOLAR, {"from": _iso(frm), "to": _iso(to), "processType": "day ahead"}
        )
        recs: dict[tuple, dict] = {}
        for r in data:
            key = (pd.to_datetime(r["settlementDate"]).date(), int(r["settlementPeriod"]))
            rec = recs.setdefault(
                key,
                {
                    "settlement_date": key[0],
                    "settlement_period": key[1],
                    "start_utc": _parse_dt(r.get("startTime")),
                    "wind_forecast_mw": 0.0,
                    "solar_forecast_mw": 0.0,
                    "published_at": _parse_dt(r.get("publishTime")),
                    "event_at": _parse_dt(r.get("startTime")),
                },
            )
            psr = (r.get("psrType") or "").lower()
            q = _f(r.get("quantity")) or 0.0
            if "solar" in psr:
                rec["solar_forecast_mw"] += q
            elif "wind" in psr:
                rec["wind_forecast_mw"] += q
        return self._lineage(pd.DataFrame(list(recs.values())), ts, "wind_solar_forecast")

    def generation_by_fuel(self, frm: datetime, to: datetime) -> pd.DataFrame:
        """Half-hourly generation mix (MW by fuel type) aggregated from FUELINST."""
        data, ts = self._get(
            EP_FUELINST, {"publishDateTimeFrom": _iso(frm), "publishDateTimeTo": _iso(to)}
        )
        rows = [
            {
                "settlement_date": pd.to_datetime(r["settlementDate"]).date(),
                "settlement_period": int(r["settlementPeriod"]),
                "fuel_type": r.get("fuelType"),
                "generation_mw": _f(r.get("generation")),
                "published_at": _parse_dt(r.get("publishTime")),
                "event_at": _parse_dt(r.get("startTime")),
            }
            for r in data
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            return self._lineage(df, ts, "generation_by_fuel")
        agg = (
            df.groupby(["settlement_date", "settlement_period", "fuel_type"], as_index=False)
            .agg(generation_mw=("generation_mw", "mean"))
        )
        return self._lineage(agg, ts, "generation_by_fuel")

    def bid_offer_data(self, frm: datetime, to: datetime, bm_unit: str | None = None) -> pd.DataFrame:
        """Bid-Offer Data (BOD): submitted bid/offer price-volume pairs per BM unit."""
        params = {"from": _iso(frm), "to": _iso(to)}
        if bm_unit:
            params["bmUnit"] = bm_unit
        data, ts = self._get(EP_BOD, params)
        rows = [
            {
                "settlement_date": pd.to_datetime(r["settlementDate"]).date(),
                "settlement_period": int(r["settlementPeriod"]),
                "bm_unit": r.get("bmUnit"),
                "pair_id": r.get("pairId"),
                "offer_price": _f(r.get("offer")),
                "bid_price": _f(r.get("bid")),
                "level_from": _f(r.get("levelFrom")),
                "level_to": _f(r.get("levelTo")),
                "time_from": _parse_dt(r.get("timeFrom")),
                "time_to": _parse_dt(r.get("timeTo")),
                "event_at": _parse_dt(r.get("timeFrom")),
                "published_at": None,
            }
            for r in data
        ]
        return self._lineage(pd.DataFrame(rows), ts, "BOD")

    def bid_offer_acceptances(self, frm: datetime, to: datetime, bm_unit: str | None = None) -> pd.DataFrame:
        """Bid-Offer Acceptance Level data (BOALF): accepted actions with timestamps."""
        params = {"from": _iso(frm), "to": _iso(to)}
        if bm_unit:
            params["bmUnit"] = bm_unit
        data, ts = self._get(EP_BOALF, params)
        rows = [
            {
                "settlement_date": pd.to_datetime(r["settlementDate"]).date(),
                "settlement_period_from": int(r["settlementPeriodFrom"]),
                "settlement_period_to": int(r["settlementPeriodTo"]),
                "bm_unit": r.get("bmUnit"),
                "acceptance_number": r.get("acceptanceNumber"),
                "acceptance_time": _parse_dt(r.get("acceptanceTime")),
                "level_from": _f(r.get("levelFrom")),
                "level_to": _f(r.get("levelTo")),
                "time_from": _parse_dt(r.get("timeFrom")),
                "time_to": _parse_dt(r.get("timeTo")),
                "so_flag": bool(r.get("soFlag")),
                "stor_flag": bool(r.get("storFlag")),
                "event_at": _parse_dt(r.get("acceptanceTime")),
                "published_at": _parse_dt(r.get("acceptanceTime")),
            }
            for r in data
        ]
        return self._lineage(pd.DataFrame(rows), ts, "BOALF")


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def _f(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
