"""Data-adapter tests: providers, offline snapshot, cache, Elexon parsing (mocked)."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest
import respx
from gb_battery.data.cache import ParquetCache
from gb_battery.data.elexon import ElexonClient
from gb_battery.data.market_snapshot import build_market_snapshot
from gb_battery.data.providers import (
    CsvUploadProvider,
    EpexLicensedProviderStub,
    NotConfigured,
    SyntheticOrderBookProvider,
)
from gb_battery.data.settings import DataSettings
from gb_battery.settlement import UTC
from httpx import Response


def test_synthetic_provider_labels_data_synthetic():
    df = SyntheticOrderBookProvider().get_prices(date(2025, 1, 15))
    assert len(df) == 48
    assert (df["kind"] == "synthetic").all()


def test_epex_stub_refuses_without_licence():
    with pytest.raises(NotConfigured):
        EpexLicensedProviderStub().get_prices(date(2025, 1, 15))


def test_csv_upload_provider_reads_prices():
    csv = b"settlement_period,price\n1,10.5\n2,-3.0\n48,99.9\n"
    df = CsvUploadProvider(csv).get_prices(date(2025, 1, 15))
    assert list(df["price"]) == [10.5, -3.0, 99.9]
    assert (df["kind"] == "assumption").all()


def test_csv_upload_rejects_missing_columns():
    with pytest.raises(ValueError):
        CsvUploadProvider(b"period,value\n1,2\n")


def test_offline_snapshot_is_all_synthetic_and_dst_correct():
    settings = DataSettings(offline=True)
    # Autumn long day -> 50 periods.
    snap = build_market_snapshot(date(2025, 10, 26), settings=settings)
    assert len(snap.frame) == 50
    assert all(s.kind.value == "synthetic" for s in snap.statuses)
    inputs = snap.to_optimisation_inputs()
    assert inputs.horizon == 50


def test_cache_put_get_and_last_update(tmp_path):
    settings = DataSettings(cache_dir=tmp_path)
    cache = ParquetCache(settings)
    df = pd.DataFrame({"settlement_period": [1, 2], "price": [10.0, 20.0]})
    ts = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
    cache.put("elexon.MID", "2025-01-15", df, ts)
    back = cache.get("elexon.MID", "2025-01-15")
    assert back is not None and len(back) == 2
    assert cache.last_update("elexon.MID") == ts
    entry = cache.entry("elexon.MID", "2025-01-15")
    assert entry is not None and entry.rows == 2


@respx.mock
def test_elexon_mid_parsing_from_mocked_api():
    base = "https://data.elexon.co.uk/bmrs/api/v1"
    payload = {
        "metadata": {},
        "data": [
            {
                "startTime": "2025-01-14T00:00:00Z",
                "dataProvider": "APXMIDP",
                "settlementDate": "2025-01-14",
                "settlementPeriod": 1,
                "price": 80.6,
                "volume": 1200.0,
            }
        ],
    }
    respx.get(f"{base}/balancing/pricing/market-index").mock(return_value=Response(200, json=payload))
    client = ElexonClient(DataSettings())
    df = client.market_index_data(
        datetime(2025, 1, 14, tzinfo=UTC), datetime(2025, 1, 14, 1, tzinfo=UTC)
    )
    assert df["mid_price"].iloc[0] == 80.6
    assert df["settlement_period"].iloc[0] == 1
    assert df["source"].iloc[0] == "elexon.MID"
    assert df["event_at"].iloc[0] is not None
