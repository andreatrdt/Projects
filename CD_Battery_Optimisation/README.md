# GB Battery Co-Optimisation Terminal

A research & decision-support platform showing how a battery trader in **Great Britain**
could combine public market data, forecasts, balancing data and constrained optimisation
to decide how much a battery should **charge, discharge or keep available** during each
half-hourly **Settlement Period**.

> **Disclaimer.** This is a research and decision-support project (built for a CV /
> portfolio). It is **not** a live trading or asset-control system. It submits **no**
> market orders, controls **no** physical asset, and makes **no** claim of proprietary
> data access or affiliation with any employer. Public data only (Elexon & NESO open
> licences). No EPEX order-book data is included or redistributed. Backtested returns
> are illustrative and are **not** achievable in live trading.

---

## What it demonstrates

- GB power-market structure (wholesale, Balancing Mechanism, imbalance settlement)
- Battery **state-of-charge optimisation** with physical & economic constraints
- A **point-in-time replay engine**: for any decision at time *t*, it can be proven
  that every input was published at or before *t* (machine-checked leakage audit)
- **Rolling-horizon operation**: forecast → optimise → execute one Settlement Period →
  settle against the published outturn → carry SoC forward
- **Historical Replay, Live Paper Trading and a Perfect-Foresight benchmark** as three
  strictly separated modes (hindsight is labelled, never mixed into strategy results)
- Real public-API ingestion with **data lineage** (source / retrieval / publication / event timestamps)
- **Rigorous, chronological** time-series forecasting (no leakage, quantile bands, vintages)
- **Constrained optimisation** in Pyomo solved with the open-source HiGHS solver
- **Scenario & stochastic** optimisation with a CVaR risk penalty
- A **trader-focused** web interface that explains *why* each action was chosen

### Replay & Live Trading (the centrepiece)

The **Replay & Live** page answers: *at every historical or live decision timestamp,
what information was available, what did the model forecast, what action did it
choose, and how did that decision perform once the actual outcome became known?*

- **Historical Replay** steps through a completed day chronologically; each decision
  gate sees only records with `published_at ≤ as_of` (MID availability is
  reconstructed as period end + 10 min and flagged as such).
- **Live Paper Trading** applies the same loop to today: settled paper P&L for
  completed periods, forecast-only for the future — future actuals are null by
  construction. No orders are submitted anywhere.
- **Perfect Foresight** optimises on the realised path and is labelled
  *"not a tradable strategy"* — it exists only as an upper bound.

The credible headline P&L is **wholesale-only** (realised wholesale P&L −
degradation); reserve/BM revenue elsewhere in the app is labelled
*experimental / assumption-based*. Execution is assumed at the MID reference price —
no spread, liquidity, partial fills or market impact are modelled. See
[docs/replay_methodology.md](docs/replay_methodology.md).

## Architecture

```
 External APIs            Raw data layer         Validation /            Feature store
 (Elexon, NESO)   ─────▶  (typed adapters,  ───▶ normalisation    ───▶  (leakage-safe
   + CSV upload           raw payloads,           (Pandera schemas,       features)
   + synthetic            lineage, cache)         Europe/London tz)          │
                                                                             ▼
   Web dashboard   ◀────  FastAPI      ◀────  Optimiser (Pyomo/HiGHS)  ◀── Forecasts &
   (Next.js, TS,          (REST API)          deterministic / stochastic    scenarios
    Recharts,                                 / CVaR + rolling horizon
    TanStack Table)                                    ▲
                                                       ├── Replay engine (PIT store:
                                                       │   published_at ≤ as_of,
                                                       │   forecast vintages, rolling
                                                       │   execute-one-period loop,
                                                       │   leakage audit, live paper)
                                                       └── Backtester (benchmarks,
                                                           perfect-foresight bound,
                                                           leakage audit)
```

See [docs/architecture.md](docs/architecture.md) for detail.

## Repository layout

```
backend/            Python 3.12 package `gb_battery` + tests
  gb_battery/
    settlement.py       GB Settlement Period calendar (DST-aware: 46/48/50 SPs)
    battery/            BatteryConfig (physical & economic parameters)
    optimiser/          Pyomo model, solver, deterministic co-optimisation, explanations
    data/               Elexon + NESO adapters, providers, lineage, cache, market snapshot
    forecast/           Leakage-safe features, baselines, quantile models, chronological CV
    scenario/           Scenario generation, Scenario Lab, stochastic + CVaR optimiser
    backtest/           Daily backtest engine, benchmarks, leakage audit, metrics
    replay/             Point-in-time store, PIT forecaster, rolling replay engine,
                        benchmarks (perfect foresight labelled), session registry
    bm/                 BM acceptance research module (exploratory)
    api/                FastAPI app (routers: market, optimise, analysis, data, replay)
    demo/               Synthetic scenarios + frozen public-data sample
    data_samples/       Frozen synthetic Parquet (offline demo)
  tests/                pytest suite (98 tests)
frontend/           Next.js + TypeScript + Tailwind + Recharts + TanStack Table
docs/               Architecture, data sources, methodology, model, backtesting, limitations
```

## Quick start

### Option A — Docker (one command)

```bash
docker compose up --build
# Frontend: http://localhost:3000   API: http://localhost:8000/docs
```

### Option B — local (Python + Node)

```bash
# 1. Backend
make install            # creates .venv and installs backend[dev]
make ingest-demo        # builds the offline demo snapshot (no network needed)
make test               # run the 98-test suite
make run                # FastAPI on http://localhost:8000  (add GBB_OFFLINE=1 for offline)

# 2. Frontend (separate terminal)
make install-frontend
make run-frontend       # Next.js on http://localhost:3000
```

On Windows without `make`, run the underlying commands in [docs/deployment.md](docs/deployment.md).

### Try it without the network

Everything runs **offline** using the frozen synthetic sample and demo scenarios:

```bash
GBB_OFFLINE=1 make run                          # backend serves synthetic/demo data
python -m gb_battery.cli backtest --days 21     # rolling backtest, prints a summary
```

## Data sources

| Source | Datasets used | Licence |
|--------|---------------|---------|
| **Elexon Insights (BMRS)** | Market Index Data (MID), system/imbalance prices, demand outturn & forecast, wind/solar forecast, generation by fuel, BOD, BOALF | Elexon open data — attribute Elexon |
| **NESO data portal (CKAN)** | Demand/forecasts, balancing & constraint costs, EAC/DC/DM/DR & reserve services | Dataset-specific open licences — attribute NESO |
| **EPEX SPOT** | *(none included)* — licensed order-book data; only a documented stub adapter | Not redistributed |
| **User CSV** | price forecasts, order-book depth, telemetry, contracted positions, service prices | your own |

Endpoint paths & field names were confirmed against the live APIs (see
[docs/data_sources.md](docs/data_sources.md)), not assumed.

## Mathematical formulation (summary)

For each Settlement Period *t* of duration Δt:

**State of charge**
```
soc[t+1] = soc[t] + η_c · charge_mw[t] · Δt − discharge_mw[t] · Δt / η_d
soc_min ≤ soc[t] ≤ soc_max
```

**No simultaneous charge/discharge** (binaries): `cbin[t] + dbin[t] ≤ 1`, with
`0 ≤ charge ≤ P_c·cbin`, `0 ≤ discharge ≤ P_d·dbin`.

**Conservative reserve headroom** (power and energy-duration), grid import/export limits,
ramp and daily-cycle limits, and a terminal-SoC floor + value.

**Objective (maximise)** wholesale + service availability + expected BM activation +
terminal value, minus charging cost, degradation, efficiency losses and imbalance
exposure — avoiding double-counting the same energy/capacity. See
[docs/optimisation_model.md](docs/optimisation_model.md).

## Example result

A single-day **rolling point-in-time replay** (50 MW / 100 MWh demo battery, bundled
synthetic sample, 2025-01-15): at each of the 48 gates the model forecast the rest of
the day from information published before the gate, optimised, and committed one
period. All 48 decisions passed the leakage audit.

| Strategy (same information sets) | Realised P&L | Capture of perfect foresight |
|----------------------------------|--------------|------------------------------|
| No-operation | £0 | 0% |
| Rolling threshold rule | ~£6.9k | ~62% |
| **Rolling forecast optimiser** | **~£10.2k** | **~91%** |
| Perfect foresight (labelled upper bound) | ~£11.2k | 100% |

1-step-ahead forecast MAE ≈ £7.4/MWh, bias ≈ −£0.3/MWh.
*(Synthetic sample; illustrative only — not achievable live.)*

## Testing

```bash
make test        # pytest: settlement/DST, constraints, the 6 deterministic cases,
                 # economics, forecasting, backtest ordering, scenario/stochastic, API,
                 # and the replay suite (PIT filtering, leakage rejection, SoC carry,
                 # first-action-only execution, 46/48/50-SP days, reproducibility,
                 # live-mode null future actuals, a hand-verifiable 3-period case)
make lint        # ruff
make typecheck   # mypy (backend)
cd frontend && npm run build   # Next.js production build + type-check
```

The suite includes the six named deterministic acceptance cases (negative-price charging,
capacity preservation, service value preservation, degradation-aware cycling, terminal SoC).

## Data limitations

- Market Index Data is a **short-term wholesale reference**, not a full live EPEX order book.
- MID has no per-record publish time; replay availability is **reconstructed** as
  period end + 10 minutes and flagged `publication_reconstructed` in every record.
- Reserve constraints are **conservative simplifications** of real DC/DM/DR & Balancing Reserve rules.
- BM acceptance modelling is **exploratory**; the production optimiser uses user/historical
  service-value assumptions and labels every estimate.
- Synthetic/estimated inputs are for demonstration and **clearly distinguished** from observed data in the UI.

See [docs/limitations.md](docs/limitations.md).

## Licence

Code: MIT (see [LICENSE](LICENSE)). Third-party **data** is governed by Elexon and NESO open
licences; EPEX order-book data is licensed and **not** included. Attribute Elexon and NESO
when redistributing data.
