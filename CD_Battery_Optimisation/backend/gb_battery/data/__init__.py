"""Data adapters, lineage and caching.

Public data sources (all replaceable behind interfaces):
* Elexon Insights (BMRS) — MID, system prices, demand, wind/solar, BOD/BOALF.
* NESO data portal (CKAN) — demand/forecasts, balancing/constraint costs, services.
* User CSV uploads and clearly-labelled synthetic demo data.
"""

from gb_battery.data.elexon import ElexonClient
from gb_battery.data.market_snapshot import MarketSnapshot, build_market_snapshot
from gb_battery.data.neso import NesoClient
from gb_battery.data.providers import (
    CsvUploadProvider,
    ElexonMIDProvider,
    EpexLicensedProviderStub,
    NotConfigured,
    SyntheticOrderBookProvider,
    WholesaleMarketDataProvider,
    get_provider,
)

__all__ = [
    "ElexonClient",
    "NesoClient",
    "MarketSnapshot",
    "build_market_snapshot",
    "WholesaleMarketDataProvider",
    "ElexonMIDProvider",
    "CsvUploadProvider",
    "SyntheticOrderBookProvider",
    "EpexLicensedProviderStub",
    "NotConfigured",
    "get_provider",
]
