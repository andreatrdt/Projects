# Replay & Live Trading methodology

The Replay subsystem answers the question a credible trading-research platform must be
able to prove, not just assert:

> At every historical or live decision timestamp, what information was available, what
> did the model forecast, what action did it choose, and how did that decision perform
> once the actual outcome became known?

## Three strictly separated modes

| Mode | Decides on | Settles on | Purpose |
|------|-----------|------------|---------|
| **Historical Replay** | Point-in-time information only | Published outturns | The credible strategy result |
| **Live Paper Trading** | Point-in-time information up to *now* | Outturns as they publish | Same loop, applied to today |
| **Perfect Foresight** | The realised price path | The realised price path | Labelled upper bound only |

Perfect foresight is computed and displayed **only** as a benchmark
("Perfect foresight benchmark — not a tradable strategy") and is never mixed into
replay or paper-trading P&L.

## The rolling decision loop

For each Settlement Period *t* of the chosen day, in chronological order:

1. **Gate**: the decision timestamp `as_of` is the period's start instant.
2. **Information set**: the `PITDataStore` returns only records with
   `published_at <= as_of`. This filter is the *single* gateway to data during a
   replay; a record that postdates the gate raises `PITViolation`.
3. **Forecast**: the `PITForecaster` issues a vintage for every remaining period
   (point + q10/q50/q90), recording the newest input publication time it used.
4. **Optimise**: the existing Pyomo/HiGHS deterministic optimiser runs on the
   remaining horizon with **wholesale-only** revenue streams, the current SoC, the
   *remaining* daily cycle budget, and the end-of-day terminal-SoC constraints.
5. **Execute one period**: only the first action of the proposed schedule is
   committed (physically clamped). The full proposed schedule is stored so the UI
   can show how the plan evolved.
6. **Settle**: once the period's outturn is published, realised P&L
   (`actual_price × net_export × Δt − degradation`) and the forecast error are
   recorded. In live mode an unpublished outturn leaves the decision `pending`.
7. **Carry state**: SoC, cumulative discharge (cycle budget) and cumulative P&L
   persist to the next gate. The battery is never reset intra-day.

The engine is deterministic: identical options reproduce an identical decision log.

## Point-in-time data rules

Every record carries `published_at` (when a decision-maker could have seen it) in
addition to its event time. Provenance is one of:

`observed` · `published_forecast` · `model_forecast` · `reconstructed` ·
`synthetic` · `assumed` · `perfect_foresight`

Availability assumptions, stated openly:

- **Elexon MID** exposes no per-record publish time, so availability is
  *reconstructed* as `period end + 10 minutes` (configurable). Records are flagged
  `publication_reconstructed`. Consequence: at the gate for period *t*, the newest
  usable price is period *t−2*.
- **Day-ahead demand / wind / solar forecasts** use the real Elexon `publishTime`
  verbatim where returned.
- **Synthetic data** follows exactly the same availability rule as live MID so the
  offline demo exercises the identical code path.

When real point-in-time historical forecast vintages are unavailable, the model
generates its own (provenance `model_forecast`); the actual outturn is **never**
silently substituted for a forecast.

## The point-in-time forecaster

Transparent by design (not state-of-the-art, and honest about it):

- **Baseline** per target SP, in order of preference: published day-ahead forecast →
  same-SP price on the most recent visible prior day → same-SP median over visible
  prior days → global median.
- **Intraday correction**: an exponentially-weighted mean (half-life 4 SPs) of
  (observed − baseline) over today's already-published SPs, decayed by 0.9 per SP of
  lead time.
- **Uncertainty**: same-SP dispersion across visible prior days (floor £3/MWh),
  widened ~2%/SP of lead time; q10/q90 via a normal approximation.

Every vintage records `issued_at`, `information_cutoff`, the newest input
publication time, and input counts — retrievable per step via
`GET /api/replay/{id}/forecasts?step=k`.

## Leakage audit

`GET /api/replay/{id}/metrics` includes a per-decision audit proving:

- `basis_max_published_at <= as_of` (no input postdated the gate), and
- the settled outturn's availability time postdates the gate (the decision could not
  have seen its own answer).

The test suite additionally covers: rejection of future-published records, SoC
carry-forward, first-action-only execution, per-step reforecasting, DST days with
46/48/50 periods, live-mode null future actuals, and a hand-verifiable 3-period case.

## Execution assumptions (what is NOT modelled)

Committed energy is assumed executable at the MID reference price. The following are
**not** modelled, and results must be read accordingly:

- exchange order-book execution, bid/ask spreads, market depth;
- partial fills or market impact;
- real reserve procurement, commitment or activation;
- guaranteed BM acceptance;
- live production trading of any kind.

Reserve and BM revenue streams elsewhere in the app are labelled
**experimental / assumption-based** and are excluded from the rolling strategy's
headline P&L, which is:

```
realised wholesale P&L − degradation
```

Expected future P&L (forecast-based) is always reported separately from realised P&L.
