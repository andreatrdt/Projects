"""Wholesale market-data providers behind one replaceable interface.

The application works fully using ``ElexonMIDProvider`` (real public data),
``CsvUploadProvider`` (user data) or ``SyntheticOrderBookProvider`` (demo). The
``EpexLicensedProviderStub`` documents what a licensed EPEX feed would require and
refuses to fabricate it.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from datetime import date, datetime

import pandas as pd

from gb_battery.data.elexon import ElexonClient
from gb_battery.data.settings import DataSettings, get_settings
from gb_battery.demo.scenarios import diurnal_price_shape
from gb_battery.lineage import DataKind
from gb_battery.settlement import UTC, settlement_periods_for_day

# Canonical columns every provider returns.
PRICE_COLUMNS = [
    "settlement_date",
    "settlement_period",
    "start_utc",
    "price",
    "kind",
    "source",
]


class NotConfigured(RuntimeError):
    """Raised by a provider that requires configuration/licensing not present."""


class WholesaleMarketDataProvider(ABC):
    """Interface for a source of half-hourly wholesale prices (GBP/MWh)."""

    name: str = "abstract"
    kind: DataKind = DataKind.OBSERVED

    @abstractmethod
    def get_prices(self, day: date) -> pd.DataFrame:
        """Return a canonical price frame (see :data:`PRICE_COLUMNS`) for a day."""
        raise NotImplementedError


class ElexonMIDProvider(WholesaleMarketDataProvider):
    """Wholesale reference price from Elexon Market Index Data (real public data)."""

    name = "elexon_mid"
    kind = DataKind.OBSERVED

    def __init__(self, client: ElexonClient | None = None, provider: str = "APXMIDP") -> None:
        self._client = client or ElexonClient()
        self._provider = provider

    def get_prices(self, day: date) -> pd.DataFrame:
        frm = datetime(day.year, day.month, day.day, 0, 0, tzinfo=UTC)
        to = frm.replace(hour=23, minute=59)
        mid = self._client.market_index_data(frm, to, provider=self._provider)
        out = mid.rename(columns={"mid_price": "price"})[
            ["settlement_date", "settlement_period", "start_utc", "price"]
        ].copy()
        out["kind"] = DataKind.OBSERVED.value
        out["source"] = "elexon.MID"
        return out.sort_values("settlement_period").reset_index(drop=True)


class CsvUploadProvider(WholesaleMarketDataProvider):
    """Wholesale prices from a user-uploaded CSV (a *forecast/assumption*)."""

    name = "csv_upload"
    kind = DataKind.ASSUMPTION

    REQUIRED = {"settlement_period", "price"}

    def __init__(self, csv_bytes: bytes, day: date | None = None) -> None:
        self._df = pd.read_csv(io.BytesIO(csv_bytes))
        missing = self.REQUIRED - set(self._df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        self._day = day

    def get_prices(self, day: date) -> pd.DataFrame:
        day = self._day or day
        periods = {p.settlement_period: p for p in settlement_periods_for_day(day)}
        rows = []
        for _, r in self._df.iterrows():
            sp = int(r["settlement_period"])
            per = periods.get(sp)
            rows.append(
                {
                    "settlement_date": day,
                    "settlement_period": sp,
                    "start_utc": per.start_utc if per else None,
                    "price": float(r["price"]),
                    "kind": DataKind.ASSUMPTION.value,
                    "source": "user_upload",
                }
            )
        return pd.DataFrame(rows, columns=PRICE_COLUMNS)


class SyntheticOrderBookProvider(WholesaleMarketDataProvider):
    """Deterministic synthetic prices for offline demo (clearly SYNTHETIC)."""

    name = "synthetic"
    kind = DataKind.SYNTHETIC

    def __init__(self, negative_afternoon: bool = False) -> None:
        self._negative_afternoon = negative_afternoon

    def get_prices(self, day: date) -> pd.DataFrame:
        periods = settlement_periods_for_day(day)
        n = len(periods)
        rows = [
            {
                "settlement_date": day,
                "settlement_period": p.settlement_period,
                "start_utc": p.start_utc,
                "price": diurnal_price_shape(
                    p.settlement_period, n, negative_afternoon=self._negative_afternoon
                ),
                "kind": DataKind.SYNTHETIC.value,
                "source": "synthetic",
            }
            for p in periods
        ]
        return pd.DataFrame(rows, columns=PRICE_COLUMNS)


class EpexLicensedProviderStub(WholesaleMarketDataProvider):
    """Placeholder for a licensed EPEX SPOT order-book feed.

    This project ships **no** EPEX data and makes **no** assumptions about private
    endpoints or credentials. A real implementation would require:

    * a commercial EPEX SPOT data licence and API credentials;
    * agreement to EPEX redistribution restrictions (order-book data may not be
      published or committed to source control);
    * mapping of EPEX continuous/auction products to GB Settlement Periods.

    Until configured it raises :class:`NotConfigured`.
    """

    name = "epex_licensed_stub"
    kind = DataKind.OBSERVED

    def get_prices(self, day: date) -> pd.DataFrame:  # noqa: ARG002
        raise NotConfigured(
            "EPEX licensed order-book data is not configured. This public build does "
            "not include EPEX data or endpoints. Provide a licensed feed adapter, or "
            "use ElexonMIDProvider / CsvUploadProvider / SyntheticOrderBookProvider."
        )


def get_provider(name: str, settings: DataSettings | None = None, **kwargs) -> WholesaleMarketDataProvider:
    """Factory: resolve a provider by name."""
    settings = settings or get_settings()
    if name == "elexon_mid":
        return ElexonMIDProvider(**kwargs)
    if name == "synthetic":
        return SyntheticOrderBookProvider(**kwargs)
    if name == "csv_upload":
        return CsvUploadProvider(**kwargs)
    if name == "epex_licensed_stub":
        return EpexLicensedProviderStub()
    raise ValueError(f"Unknown provider '{name}'")
