# Architecture

## Data flow

```
┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌────────────────┐
│ External     │   │ Raw data layer  │   │ Validation &     │   │ Feature store  │
│ APIs         │──▶│ typed adapters  │──▶│ normalisation    │──▶│ leakage-safe   │
│ Elexon, NESO │   │ raw payloads    │   │ Pandera schemas  │   │ features       │
│ + CSV upload │   │ lineage + cache │   │ Europe/London tz │   │                │
│ + synthetic  │   │ (Parquet/DuckDB)│   │ DST-aware SPs    │   │                │
└──────────────┘   └─────────────────┘   └──────────────────┘   └───────┬────────┘
                                                                         │
        ┌────────────────────────────────────────────────────────────────┘
        ▼
┌────────────────┐   ┌───────────────────────────┐   ┌──────────────┐   ┌──────────────┐
│ Forecasts &    │──▶│ Optimiser (Pyomo/HiGHS)   │──▶│ FastAPI      │──▶│ Web dashboard│
│ scenarios      │   │ deterministic / stochastic│   │ REST API     │   │ Next.js + TS │
│ baselines+GBM  │   │ / CVaR + rolling horizon  │   │              │   │ Recharts     │
└────────────────┘   └─────────────┬─────────────┘   └──────────────┘   │ TanStack     │
                                   ▼                                     └──────────────┘
                     ┌───────────────────────────┐
                     │ Backtester                │
                     │ benchmarks + perfect      │
                     │ foresight + leakage audit │
                     └───────────────────────────┘
```

## Layers

| Layer | Package | Responsibility |
|-------|---------|----------------|
| Settlement calendar | `gb_battery.settlement` | DST-aware SP grid (46/48/50 SPs), UTC storage, London interpretation |
| Domain | `gb_battery.battery` | `BatteryConfig` physical/economic parameters + validation |
| Data adapters | `gb_battery.data` | Elexon/NESO clients, `WholesaleMarketDataProvider` implementations, Pandera schemas, Parquet+DuckDB cache, resilient `MarketSnapshot` |
| Lineage | `gb_battery.lineage` | `DataKind` (observed/forecast/estimated/assumption/synthetic), timestamps |
| Forecasting | `gb_battery.forecast` | Leakage-safe features, baselines, GBM quantile models, chronological CV |
| Optimisation | `gb_battery.optimiser` | Pyomo model, HiGHS solver wrapper, result extraction, marginal values, explanations |
| Scenarios | `gb_battery.scenario` | Scenario generation, Scenario Lab transforms, stochastic + CVaR + robust |
| Backtesting | `gb_battery.backtest` | Daily engine, benchmark strategies, metrics, leakage audit |
| Replay | `gb_battery.replay` | Point-in-time store (`published_at ≤ as_of` gateway), PIT forecaster with vintages, rolling execute-one-period engine, labelled perfect-foresight benchmark, in-memory sessions — see [replay_methodology.md](replay_methodology.md) |
| BM research | `gb_battery.bm` | Exploratory BOD↔BOALF acceptance modelling |
| API | `gb_battery.api` | FastAPI routers (market, optimise, analysis, data, replay, live) |
| Frontend | `frontend/` | Next.js pages, typed API client, shared state, charts, tables |

## Key design decisions

- **Everything reduces to `OptimisationInputs`** (a list of `PeriodInput`). Elexon, NESO,
  CSV, synthetic and scenario data all resolve to the same structure, so the optimiser is
  fully decoupled from data sources.
- **Provenance is first-class.** Each value carries a `DataKind`; the UI never renders an
  estimate like an observed value.
- **Resilience by default.** Each data source is fetched independently; one failing source
  degrades that series to cache/synthetic without breaking the app. Offline mode serves the
  frozen sample.
- **Solver isolation.** HiGHS (via Pyomo APPSI) captures OS file descriptors while solving,
  which deadlocks inside a server worker thread. `optimiser.solver.guarded_solve` redirects
  fds to `os.devnull` under a process-wide lock, making solves server-safe. Gurobi is used
  only if a licence is detected — never mandatory.
- **Marginal values.** Per-period water values come from SoC-balance duals (solve the LP with
  binaries fixed and integrality relaxed); horizon-level resource values come from small
  perturbation resolves. Both degrade gracefully if duals are unavailable.

## Storage

- **PostgreSQL** (optional, `backend[db]`) for application/metadata in a deployed setup.
- **DuckDB + Parquet** for analytical time-series and the cache metadata index.
- The frozen public-data sample ships in `gb_battery/data_samples/` so reviewers can run the
  optimiser and backtest with no database and no network.
