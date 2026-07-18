# Backtesting & forecasting integrity

## Chronological, leakage-free by construction

Forecasts for Settlement Date *D* are made the day before. The only permissible model inputs
are therefore:

- calendar variables (known arbitrarily far ahead);
- day-ahead **forecasts** of demand / wind / solar (published before *D*);
- **lagged outturn** prices from strictly earlier dates (D−1, D−7, rolling stats).

Same-period outturn prices are **never** features (`gb_battery.forecast.features`). The
optimiser sees only forecast inputs; realised outturn is applied afterwards to settle P&L.

## Rolling backtest

For each settlement date the engine (`gb_battery.backtest.engine`):

1. builds `OptimisationInputs` from day-ahead forecasts;
2. asks the strategy for a schedule (SoC carried from the previous day);
3. **settles** the committed schedule against realised outturn prices;
4. rolls SoC forward.

Only the **perfect-foresight** benchmark is allowed to decide on outturns — it is the
theoretical upper bound and is explicitly labelled as unachievable live.

## Benchmarks

| Strategy | Description |
|----------|-------------|
| `no_operation` | Battery never trades (£0). |
| `threshold_rule` | Charge below the 30th price percentile, discharge above the 70th. |
| `fixed_percentile` | Fixed 25/75 percentile variant. |
| `deterministic_optimiser` | MPC on the point forecast (the production policy). |
| `perfect_foresight` | Optimise on realised outturns — upper bound. |

## Metrics

Total & decomposed P&L (wholesale / service / degradation), full-cycle equivalents,
degradation-adjusted return, average SoC, % of periods at physical limits, price-forecast
MAE/RMSE, maximum drawdown, and CVaR₉₅ of per-period P&L. Forecast validation additionally
reports **pinball loss** per quantile under expanding-window CV.

## Leakage audit

`gb_battery.backtest.leakage` provides an automated audit that (a) confirms no outturn column
is used as a feature, and (b) confirms the backtest fed forecasts (not outturns) into decisions
— it passes only when the forecast differs from outturn on a non-trivial share of periods.

## Honesty

Backtested returns are illustrative and are **not** achievable in live trading. The synthetic
sample exists so the pipeline is reproducible offline; on real data, results depend on data
quality, execution, and product rules not fully modelled here.
