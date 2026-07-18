# Limitations

Honest boundaries of this research platform. If a claim is not on the "modelled" side
of this page, the project does not make it.

## Market data

- **Market Index Data (MID)** is a short-term wholesale *reference* price, not a full
  EPEX order book. Executing at MID is an assumption, not a guarantee.
- MID exposes no per-record publish time; replay availability is **reconstructed** as
  period end + 10 minutes and flagged as such (`publication_reconstructed`).
- No EPEX licensed data is included or redistributed; the EPEX adapter is a documented
  stub that raises `NotConfigured`.

## Execution

Not modelled: bid/ask spreads, order-book depth, partial fills, market impact,
transaction fees, credit/collateral, intraday gate closures. Committed energy is
assumed executable at the reference price — labelled on every relevant page.

## Reserve & Balancing Mechanism

- Reserve headroom/energy constraints are conservative simplifications of real
  Dynamic Containment / Dynamic Moderation / Dynamic Regulation / Balancing Reserve
  product rules (no EFA-block procurement, no performance monitoring).
- Default availability prices and BM activation margins are **assumptions**.
- BM acceptance modelling is exploratory; pairId↔BOALF attribution from public
  schemas is approximate.
- Consequently, reserve/BM revenue is **experimental / assumption-based** and is
  excluded from the credible rolling-strategy P&L in Replay & Live.

## Forecasting

The point-in-time forecaster is a transparent baseline (lags + intraday bias +
empirical dispersion), not a state-of-the-art price model. Quantiles use a normal
approximation. Historical *published* wholesale-price forecast vintages are not
freely available, so replay forecasts are model-generated and labelled
`model_forecast`.

## Backtesting & results

- Backtested and replayed P&L is illustrative and **not achievable live**.
- The bundled sample is synthetic (seeded generator), clearly labelled, and never
  presented as observed market data.
- Perfect foresight is an upper bound, labelled "not a tradable strategy".

## Operational

- Replay sessions are in-memory (restart clears them).
- This system submits no orders and controls no physical asset.
