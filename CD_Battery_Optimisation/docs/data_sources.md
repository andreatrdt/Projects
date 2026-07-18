# Data sources & adapters

All adapters are separate and replaceable. Endpoint paths and field names below were
**confirmed against the live APIs**, not assumed. Every processed record carries lineage:
`source`, `retrieved_at`, `published_at`, `event_at`, plus settlement date/period where
relevant.

## 1. Elexon Insights (BMRS)

Base: `https://data.elexon.co.uk/bmrs/api/v1`
Docs: <https://bmrs.elexon.co.uk/api-documentation>, <https://developer.data.elexon.co.uk/>

No API key required. The client (`gb_battery.data.elexon.ElexonClient`) is polite:
rate-limited, retrying with exponential backoff (`tenacity`), timeouts, a clear
`User-Agent`, and raw-payload caching for auditability.

| Method | Endpoint | Key fields (as returned) |
|--------|----------|--------------------------|
| `market_index_data` | `/balancing/pricing/market-index` | `settlementDate, settlementPeriod, startTime, price, volume, dataProvider` |
| `system_prices` | `/balancing/settlement/system-prices/{date}` | `systemSellPrice, systemBuyPrice, netImbalanceVolume, createdDateTime` |
| `demand_outturn` | `/demand/outturn` | `initialDemandOutturn (INDO), initialTransmissionSystemDemandOutturn (ITSDO)` |
| `demand_forecast` | `/forecast/demand/day-ahead` | `nationalDemand, transmissionSystemDemand, boundary, publishTime` |
| `wind_solar_forecast` | `/forecast/generation/wind-and-solar/day-ahead` | `psrType (Solar / Wind Onshore / Wind Offshore), quantity` |
| `generation_by_fuel` | `/datasets/FUELINST` | `fuelType, generation` (aggregated to SP mean) |
| `bid_offer_data` | `/datasets/BOD` | `bmUnit, pairId, offer, bid, levelFrom, levelTo, timeFrom, timeTo` |
| `bid_offer_acceptances` | `/datasets/BOALF` | `bmUnit, acceptanceNumber, acceptanceTime, levelFrom, levelTo, soFlag, storFlag` |

MID is a **short-term wholesale reference price**, not a full EPEX order book.

## 2. NESO data portal (CKAN)

Base: `https://api.neso.energy/api/3/action`
Portal: <https://www.neso.energy/data-portal>

`gb_battery.data.neso.NesoClient` uses CKAN `package_search` / `datastore_search` to
**discover** resources rather than hardcoding a single resource id, which drifts over time.
Relevant datasets: demand data update, national demand forecasts & outturn, wind/solar
forecasts & outturn, interconnector flows, daily balancing costs, constraint volumes/costs,
system warnings, EAC auction results, Dynamic Containment / Moderation / Regulation,
Balancing/Quick/Slow Reserve, and service requirements where published.

## 3. EPEX adapter — licensed data is NOT included

`WholesaleMarketDataProvider` has four implementations:

- `ElexonMIDProvider` — real public wholesale reference price
- `CsvUploadProvider` — user-supplied prices (labelled *assumption*)
- `SyntheticOrderBookProvider` — deterministic demo prices (labelled *synthetic*)
- `EpexLicensedProviderStub` — **raises `NotConfigured`**; contains no fake credentials and
  no private-endpoint assumptions. It documents what a licensed EPEX feed would require.

The application works fully using Elexon MID, forecasts, and uploaded/synthetic data.

## 4. User-supplied CSV

Upload forward price forecasts, intraday bid/ask, order-book depth, battery telemetry,
renewable portfolio forecasts, contracted positions and custom service prices/acceptance
probabilities. Every upload is validated (`/api/data/upload-prices`); a template is at
`/api/data/price-template`.

## Data quality & lineage

`DataKind ∈ {observed, forecast, estimated, assumption, synthetic}` drives UI colour so an
estimate is never shown as an observed market value. Timezones: **Europe/London** for market
interpretation, **UTC** for storage; DST transitions produce 46/48/50 SP days and are handled
explicitly. Backtests never use a value whose `published_at` is later than the decision time.

## Licences & attribution

- **Elexon Insights / BMRS** — open data; attribute Elexon and comply with Elexon terms.
- **NESO** — dataset-specific open licences (NESO Open Licence / CC-BY); attribute NESO.
- **EPEX SPOT** — licensed; order-book data must not be published or committed. None is included.
