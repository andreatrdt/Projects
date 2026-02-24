# Financial Engineering & FinTech Projects

A compact “toolbox” repository collecting **assignments**, **case studies**, and **projects** developed during Financial Engineering (2023–2024), plus additional **banking/fintech analytics** work and code related to **Hawkes-driven optimal control**.

> Educational/academic purposes only. Not financial advice.

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Assignments](#assignments)
  - [Assignment 1: Option Pricing and Error Analysis](#assignment-1-option-pricing-and-error-analysis)
  - [Assignment 2: Yield Curves and Sensitivities](#assignment-2-yield-curves-and-sensitivities)
  - [Assignment 3: Asset Swaps, CDS, and Time-Series Analysis](#assignment-3-asset-swaps-cds-and-time-series-analysis)
  - [Assignment 4: Risk Management (VaR & ES)](#assignment-4-risk-management-var--es)
  - [Assignment 5: Structured Products & Vol Surface Calibration](#assignment-5-structured-products--vol-surface-calibration)
  - [Assignment 6: Interest Rate Risk & Hedging](#assignment-6-interest-rate-risk--hedging)
  - [Assignment 7: Bermudan Swaptions & Certificates](#assignment-7-bermudan-swaptions--certificates)
- [Energy Price and Load Forecasting](#energy-price-and-load-forecasting)
  - [EPLF 1: Regularization Techniques](#eplf-1-regularization-techniques)
  - [EPLF 2: DNN Hyperparameter Tuning](#eplf-2-dnn-hyperparameter-tuning)
  - [EPLF 3: Distributional Models & Quantile Regression](#eplf-3-distributional-models--quantile-regression)
- [Risk Management](#risk-management)
  - [RM 1: Hazard Rate & Z-Spread](#rm-1-hazard-rate--z-spread)
  - [RM 2: Present Value & Credit VaR](#rm-2-present-value--credit-var)
- [FinTech / Banking Analytics](#fintech--banking-analytics)
  - [BC1: Segmenting Clients](#bc1-segmenting-clients)
  - [BC2: Estimating Needs](#bc2-estimating-needs)
  - [BC3: Portfolio Replica](#bc3-portfolio-replica)
  - [BC4: Early Warning System](#bc4-early-warning-system)
  - [Customer Portfolio Recommendation](#customer-portfolio-recommendation)
- [Hawkes Optimal Control](#hawkes-optimal-control)
- [Other](#other)
  - [Contest / Coding](#contest--coding)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository includes:
- **Derivatives pricing** (analytical + numerical + Monte Carlo)
- **Curve building** and sensitivities (DV01/BPV/duration)
- **Credit** (asset swap, CDS bootstrapping, First-to-Default)
- **Risk management** (VaR/ES, PCA, simulations)
- **Energy forecasting** (regularization, deep learning, probabilistic forecasting)
- **FinTech / banking analytics** (client segmentation, needs estimation, portfolio replication, early warning)
- **Hawkes-driven market-making control** (code/prototypes for an optimal control pipeline)

---

## Repository Structure

Main folders (as in the repository root):
- `A1_...` → Financial Engineering assignments
- `Assignment1_RM`, `Assignment2_RM` → Risk Management assignments
- `Electricity_Price_Load_Forecasting_...` → energy forecasting projects
- `BC1_...` → banking/fintech case studies
- `HawkesOptimalControl/code` → Hawkes / optimal control code
- `contest_2` → contest / coding material

---

## Assignments

### Assignment 1: Option Pricing and Error Analysis
- European call pricing with Black, CRR binomial tree, Monte Carlo
- Error rescaling and parameter selection
- Exotic: Knock-In call (multiple approaches)
- Sensitivities (Vega) numerical/analytical
- Bermudan option behavior vs European
- Variance reduction (antithetic variables)

[More Details & Code](./A1_EU_OptionPricing)

### Assignment 2: Yield Curves and Sensitivities
- Bootstrap discount factors and zero rates
- DV01 / BPV / duration for swaps and bonds
- Theoretical exercises (bond pricing, Garman–Kohlhagen)

[More Details & Code](./A2_Bootstrap)

### Assignment 3: Asset Swaps, CDS, and Time-Series Analysis
- Asset swap spread
- CDS curve bootstrapping (with interpolation/splines where needed)
- First-to-Default pricing via Monte Carlo
- Time-series analysis (log-returns, regressions/fits)

[More Details & Code](./A3_Credit_AS_CDS_FtD)

### Assignment 4: Risk Management (VaR & ES)
- VaR/ES (variance–covariance, historical simulation, bootstrap)
- PCA for dimensionality reduction
- Monte Carlo VaR (delta-normal and full simulation)
- Examples on path-dependent products / cliquet (where applicable)

[More Details & Code](./A4_RiskManagment)

### Assignment 5: Structured Products & Vol Surface Calibration
- Certificate pricing via Monte Carlo
- Digital options and method comparison
- Lewis formula / FFT pricing
- Volatility surface calibration (e.g., mean-variance mixture)

[More Details & Code](./A5_StructuredProducts)

### Assignment 6: Interest Rate Risk & Hedging
- Extended bootstrapping on longer maturities
- Caplet pricing (Bachelier) and spot vol calibration
- Upfront payment on IR structured products
- Delta/Vega and hedging strategies (swap/cap)

[More Details & Code](./A6_StructuredProducts)

### Assignment 7: Bermudan Swaptions & Certificates
- Bermudan swaption (Hull–White + tree methods)
- Certificate pricing (NIG + FFT / Monte Carlo)
- Black model adjustments and digital risk

[More Details & Code](./A7_StructuredProducts)

---

## Energy Price and Load Forecasting

### EPLF 1: Regularization Techniques
- Lasso / Ridge / Elastic Net
- Feature selection
- Seasonality analysis on time series

[More Details & Code](./Electricity_Price_Load_Forecasting_1)

### EPLF 2: DNN Hyperparameter Tuning
- Hyperparameter optimization (e.g., Optuna or random/grid search)
- Overfitting vs generalization analysis
- Architecture/loss comparisons

[More Details & Code](./Electricity_Price_Load_Forecasting_2)

### EPLF 3: Distributional Models & Quantile Regression
- Quantile regression (pinball loss)
- Distributional models (e.g., Normal / Johnson SU, where used)
- Probabilistic metrics (Pinball, Winkler, etc.)

[More Details & Code](./Electricity_Price_Load_Forecasting_3)

---

## Risk Management

### RM 1: Hazard Rate & Z-Spread
- Hazard rate curve bootstrapping (IG/HY)
- Z-spread via parallel shift of the risk-free curve
- (If included) market-implied transition matrix

[More Details & Code](./Assignment1_RM)

### RM 2: Present Value & Credit VaR
- PV under default scenarios
- Credit VaR under different correlations
- Concentration vs diversification

[More Details & Code](./Assignment2_RM)

---

## FinTech / Banking Analytics

This section groups projects oriented to **client analytics**, **needs estimation**, **portfolio replication**, and **risk monitoring** (details depend on the notebooks/scripts inside each folder).

### BC1: Segmenting Clients
- Client segmentation (typically clustering + feature engineering + cluster interpretation)
- Expected outputs: customer profiles/clusters + actionable insights

[More Details & Code](./BC1_SegmentingClients)

### BC2: Estimating Needs
- Estimating needs/propensity/next-best-action (typically classification/regression)
- Expected outputs: customer scoring + driver analysis (feature importance)

[More Details & Code](./BC2_EstimatingNeeds)

### BC3: Portfolio Replica
- Replication approaches (constraints, tracking error, regression/optimization, etc.)
- Expected outputs: replicating portfolio + tracking metrics

[More Details & Code](./BC3_PortfolioReplica)

### BC4: Early Warning System
- Early warning: risk/deterioration signals (anomaly detection, classification, etc.)
- Expected outputs: alerts/ranking + backtest/validation

[More Details & Code](./BC4_EarlyWarningSystem)

### Customer Portfolio Recommendation
- Portfolio/product recommendation logic (rule-based or model-based depending on the files)
- Expected outputs: ranked recommendations + explainability criteria

[More Details & Code](./CustomerPortfolioRecomendation)

---

## Hawkes Optimal Control

Code related to the “Hawkes / market making / optimal control” track (calibration pipeline + numerical components and/or backtests, depending on what is included).

[More Details & Code](./HawkesOptimalControl/code)

---

## Other

### Contest / Coding
Miscellaneous coding/contest material.

[More Details & Code](./contest_2)

---

## Getting Started

### Requirements
Most projects are notebook-based. Typically you need:
- Python 3.9+ (ideally 3.10/3.11)
- Jupyter / VS Code
- Common libs: `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, and optionally `statsmodels`, `pytorch`/`tensorflow` for DL

### Setup (recommended)
```bash
# clone
git clone <this-repo-url>
cd <repo-folder>

# create env (example with venv)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install base
pip install -U pip
pip install numpy pandas scipy matplotlib scikit-learn jupyter
