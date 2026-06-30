# Projects

A curated collection of financial engineering, banking analytics, quantitative finance and machine-learning projects. The repository includes code related to credit-risk modelling, term-structure construction, Hawkes-driven optimal control, ML-based price interval forecasting and applied financial analytics.

## Overview

This repository is organized as a practical toolbox of projects developed across financial engineering, data science and quantitative finance. The main areas covered are:

- **Credit risk and banking analytics**: probability of default, scorecards, model validation and portfolio-level risk diagnostics.
- **Fixed income and term-structure modelling**: yield curves, discount factors, bootstrapping and bond analytics.
- **Market microstructure and optimal control**: Hawkes order-flow calibration, HJB-based market-making policies and backtesting.
- **ML price interval forecasting**: short-horizon predictive bands for future mid-price changes, using Hawkes intensities, LOB state variables and control-policy diagnostics.
- **General machine-learning workflows**: feature engineering, model training, evaluation and visualization.

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

## Hawkes Optimal Control

The `HawkesOptimalControl` project combines three pillars:

1. Hawkes modelling of aggressive order-flow.
2. HJB-based optimal market-making control.
3. ML-based short-horizon predictive intervals for the next mid-price.

The ML interval layer estimates conditional quantiles of future mid-price changes and produces predictive bands around the next price level. It is designed as a probabilistic risk layer on top of the Hawkes/HJB framework rather than a standalone point-prediction model.

See `HawkesOptimalControl/README.md` and `HawkesOptimalControl/code/main.ipynb` for the full architecture and workflow.
