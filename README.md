# Projects

A curated collection of financial engineering, banking analytics, quantitative finance and machine-learning projects.

## Highlight: Hawkes Optimal Control

The `HawkesOptimalControl` project has been expanded into a three-pillar architecture:

1. Hawkes modelling of aggressive order-flow.
2. HJB-based optimal market-making control.
3. ML-based short-horizon predictive intervals for the next mid-price.

The new ML interval layer lives in `HawkesOptimalControl/code/ml_interval_forecaster.py` and is demonstrated in `HawkesOptimalControl/code/main.ipynb`. It estimates conditional quantiles of future mid-price changes using Hawkes intensities, LOB state variables and HJB backtest diagnostics.

## Repository structure

```text
Projects/
├── HawkesOptimalControl/
│   ├── README.md
│   └── code/
│       ├── main.py
│       ├── main.ipynb
│       ├── ml_interval_forecaster.py
│       ├── HawkesCalibrator.py
│       ├── hjb_theta_solver.py
│       └── ...
└── ...
```
