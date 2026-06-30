# Projects

A collection of financial engineering, banking analytics, quantitative finance and machine-learning projects.

## Hawkes Optimal Control

The `HawkesOptimalControl` project now has a three-pillar architecture:

1. Hawkes modelling of aggressive order-flow.
2. HJB-based optimal market-making control.
3. ML-based short-horizon predictive intervals for the next mid-price.

The ML interval layer lives in `HawkesOptimalControl/code/ml_interval_forecaster.py` and is demonstrated in `HawkesOptimalControl/code/main.ipynb`.

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
