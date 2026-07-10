# Financial Markets Research Portfolio

**Algorithmic trading · Market microstructure · Energy forecasting · Portfolio modelling · Derivatives and risk**

[![Python](https://img.shields.io/badge/Python-Research%20%26%20Trading-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Probabilistic%20ML-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-Algorithms-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-Numerical%20Finance-EA7600)](https://www.mathworks.com/products/matlab.html)

I am **Andrea Tarditi**, an MSc graduate in Mathematical Engineering and Quantitative Finance from Politecnico di Milano. I build research systems that connect financial models with practical decisions, from market-data processing and signal construction to backtesting, risk diagnostics and deployment.

Previously, I supported the **UniCredit Government Bonds Trading Desk**, developing Python tools for position-risk analysis, yield-curve monitoring and intraday decision support.

[LinkedIn](https://www.linkedin.com/in/andrea-tarditi/) · [Email](mailto:andreatrdt@gmail.com) · [GitHub profile](https://github.com/andreatrdt)

---

## Repository map

This repository contains **18 top-level project folders**. Some are complete standalone projects; others are successive versions, coursework implementations or archived development trees. Every folder is listed below rather than presenting iterative copies as unrelated achievements.

| Area | Projects |
|---|---|
| **Algorithmic trading and microstructure** | [Hawkes Optimal Control](./HawkesOptimalControl/) |
| **Electricity-market forecasting** | [Forecasting v1](./Electricity_Price_Load_Forecasting_1/) · [Forecasting v2](./Electricity_Price_Load_Forecasting_2/) · [Forecasting v3](./Electricity_Price_Load_Forecasting_3/) |
| **Portfolio research and machine learning** | [Big Levy Investments](./Big_Levy_Investments/) · [Customer Portfolio Recommendation](./CustomerPortfolioRecomendation/) · [Portfolio Replica](./BC3_PortfolioReplica/) · [Early Warning System](./BC4_EarlyWarningSystem/) |
| **Fixed income, credit and risk** | [Curve Bootstrapping](./A2_Bootstrap/) · [Credit Derivatives](./A3_Credit_AS_CDS_FtD/) · [Market Risk](./A4_RiskManagment/) · [Risk Management 1](./Assignment1_RM/) · [Risk Management 2](./Assignment2_RM/) · [Credit-Risk Contest](./contest_2/) |
| **Derivatives and structured products** | [European Option Pricing](./A1_EU_OptionPricing/) · [Structured Products 5](./A5_StructuredProducts/) · [Structured Products 6](./A6_StructuredProducts/) · [Structured Products 7](./A7_StructuredProducts/) |

---

# Complete project catalogue

## Algorithmic trading and market microstructure

### [Hawkes-Driven Market Making and Price Forecasting](./HawkesOptimalControl/)

An event-driven research framework for automated market making on high-frequency limit-order-book data.

- Reconstructs LOBSTER order books and cleans execution, submission and cancellation events.
- Calibrates multivariate Hawkes processes to model clustered aggressive buy and sell flow.
- Solves a finite-difference HJB problem to generate inventory-aware bid and ask quotes.
- Replays historical order-book events to evaluate fills, quote placement, inventory and P&L.
- Produces short-horizon predictive intervals for future mid-price changes using Hawkes and control-state features.
- Evaluates spread capture, adverse selection, drawdowns, inventory exposure, empirical coverage, pinball loss and directional accuracy.

`Python` `pandas` `NumPy` `SciPy` `scikit-learn` `Hawkes processes` `HJB control` `LOB replay`

---

## Electricity-market forecasting

### [Electricity Price and Load Forecasting v1](./Electricity_Price_Load_Forecasting_1/)

The first implementation of the electricity forecasting workflow, focused on data preparation, autoregressive baselines, neural models and rolling recalibration for multi-horizon forecasts.

`Python` `ARX` `Neural networks` `Rolling recalibration` `Time series`

### [Electricity Price and Load Forecasting v2](./Electricity_Price_Load_Forecasting_2/)

An expanded forecasting version adding a broader model pipeline, intraday experiments and tools for prediction-quantile construction and probabilistic evaluation.

`Python` `TensorFlow` `Quantile analysis` `Multi-horizon forecasting` `Time series`

### [Electricity Price and Load Forecasting v3](./Electricity_Price_Load_Forecasting_3/)

The final probabilistic forecasting framework, comparing quantile-regression, Gaussian and Johnson SU neural networks over a 24-hour horizon.

| Model | Average pinball loss | Average Winkler score |
|---|---:|---:|
| **Johnson SU-DNN** | **0.277** | **17.23** |
| Normal-DNN | 0.285 | 17.69 |
| QR-DNN | 0.332 | 20.41 |

The Johnson SU specification achieved the strongest saved probabilistic results, improving average pinball loss by approximately **16.5%** relative to the QR-DNN benchmark.

`Python` `TensorFlow` `TensorFlow Probability` `Keras` `Probabilistic forecasting` `Pinball loss` `Winkler score`

---

## Portfolio research and financial machine learning

### [Big Levy Investments](./Big_Levy_Investments/)

An end-to-end portfolio research platform combining investor classification, index replication, anomaly detection and interactive portfolio recommendations.

- Maps investor characteristics to risk profiles and suitable portfolio families.
- Compares Elastic Net, Kalman Filter and Ensemble Kalman Filter replication models.
- Assesses tracking error, risk, transaction costs and portfolio behaviour across market conditions.
- Combines passive replicas with anomaly signals to produce active risk-management overlays.
- Connects the research pipeline to a Flask application.

Developed as a five-person FinTech project, covering the workflow from data preparation and model tuning to portfolio construction and deployment.

`Python` `scikit-learn` `Elastic Net` `Kalman filtering` `Optuna` `Flask` `Portfolio analytics`

### [Customer Portfolio Recommendation](./CustomerPortfolioRecomendation/)

A development and archival tree of the portfolio-recommendation project, containing intermediate business cases, calibration utilities, pricing modules, model experiments and the completed application pipeline. It overlaps substantially with **Big Levy Investments** and is retained to show the development process rather than presented as a separate final product.

`Python` `Machine learning` `Portfolio recommendation` `Model calibration` `Web application`

### [Portfolio Replica with Cointegration and VECM](./BC3_PortfolioReplica/)

A portfolio-replication study using econometric relationships between a target portfolio and candidate underlyings, with emphasis on cointegration and Vector Error Correction Models.

`Python` `VECM` `Cointegration` `Portfolio replication` `Econometrics`

### [Financial Early Warning System](./BC4_EarlyWarningSystem/)

A market-regime and systemic-risk project that frames risk-off periods as anomalies and tests data-driven methods for identifying abnormal financial conditions as early as possible.

The folder includes anomaly-detection research and experimental neural approaches such as LSTM and MLP specifications.

`Python` `Anomaly detection` `Novelty detection` `LSTM` `Financial regimes` `Systemic risk`

---

## Fixed income, credit and risk

### [European Interest-Rate Curve Bootstrapping](./A2_Bootstrap/)

Builds discount factors and zero rates from deposits, futures and swaps, then applies the resulting curve to pricing and sensitivity analysis for fixed-income instruments.

`MATLAB` `Curve bootstrapping` `Discount factors` `Zero rates` `DV01` `Fixed income`

### [Credit, Asset Swaps, CDS and First-to-Default](./A3_Credit_AS_CDS_FtD/)

A credit-derivatives implementation covering risky-bond valuation, CDS-curve bootstrapping, asset-swap analysis and basket-credit products including first-to-default structures.

`MATLAB` `Python` `Credit curves` `CDS` `Asset swaps` `First-to-default`

### [Market Risk and Monte Carlo Analysis](./A4_RiskManagment/)

A Python and MATLAB risk-management project using historical market data and simulation-based methods to study portfolio losses, dependence and risk measures.

`Python` `MATLAB` `Monte Carlo` `Market risk` `Portfolio losses` `Risk measures`

### [Risk Management Assignment 1](./Assignment1_RM/)

A credit-risk toolkit covering rating transitions, migration probabilities, hazard curves and risky-bond valuation under alternative credit assumptions.

`MATLAB` `Credit migration` `Transition matrices` `Hazard rates` `Risky bonds`

### [Risk Management Assignment 2](./Assignment2_RM/)

A second credit-risk implementation focused on portfolio credit loss, regulatory-style risk components and credit VaR calculations.

`MATLAB` `Credit VaR` `IRB framework` `Default risk` `Portfolio risk`

### [Credit-Risk Contest Implementation](./contest_2/)

A compact contest-oriented implementation of credit-risk functions, including hazard-curve construction, risky-bond valuation, transition-matrix tools and IRB calculations.

`MATLAB` `Credit risk` `Hazard curves` `IRB` `Risky bonds`

---

## Derivatives and structured products

### [European Option Pricing](./A1_EU_OptionPricing/)

Compares analytical, tree-based and simulation approaches for European and barrier-style option pricing, including convergence analysis, variance reduction and Greek estimation.

`MATLAB` `Black-Scholes` `Cox-Ross-Rubinstein` `Monte Carlo` `Antithetic variates` `Greeks`

### [Structured Products 5: Equity Certificate and Digital Options](./A5_StructuredProducts/)

Prices an equity-linked certificate and digital-option components using Black-style models, volatility-smile interpolation and large-scale Monte Carlo simulation.

`MATLAB` `Structured products` `Digital options` `Volatility smile` `Monte Carlo` `Black model`

### [Structured Products 6: Interest-Rate Certificate](./A6_StructuredProducts/)

Builds an interest-rate structured-product valuation workflow from curve bootstrapping and cap-volatility calibration through upfront pricing, bucketed DV01, vega and hedging analysis.

`MATLAB` `Interest-rate derivatives` `Cap volatilities` `Bachelier model` `DV01` `Vega` `Hedging`

### [Structured Products 7: Lévy Models and Advanced Pricing](./A7_StructuredProducts/)

Prices structured payoffs under Normal Inverse Gaussian and Variance Gamma dynamics, comparing closed-integral, Monte Carlo, FFT and Black-with-skew approaches. The wider folder also contains tree and interest-rate pricing utilities.

`MATLAB` `Normal Inverse Gaussian` `Variance Gamma` `FFT` `Monte Carlo` `Volatility skew`

---

## Research principles

- **Chronology before convenience:** time-series models use ordered validation rather than random splits when leakage would distort results.
- **Baselines before complexity:** advanced models are compared with simpler statistical, analytical or rule-based alternatives.
- **Diagnostics before headline performance:** returns and forecast scores are assessed together with risk, calibration and operational behaviour.
- **Models as systems:** the strongest projects cover data ingestion, modelling, evaluation, visualisation and deployment rather than isolated notebooks.
- **Reproducibility:** reusable modules, configuration files and documented workflows are preferred over one-off analysis.

## Technical stack

**Programming:** Python, C++, MATLAB  
**Data and modelling:** pandas, NumPy, SciPy, scikit-learn, TensorFlow, Keras, TensorFlow Probability, statsmodels  
**Finance:** market microstructure, backtesting, stochastic control, time-series modelling, Monte Carlo, fixed income, portfolio analytics, derivatives pricing  
**Development:** Git, Jupyter, Flask, Streamlit

---

## Authorship and repository structure

Several projects were developed in academic teams. The relevant project files identify collaborators where applicable. Iterative folders and archived development trees are explicitly labelled above to avoid presenting duplicate versions as independent work.

## Contact

For trading, quantitative research or analytical opportunities:

**Andrea Tarditi** · [andreatrdt@gmail.com](mailto:andreatrdt@gmail.com) · [LinkedIn](https://www.linkedin.com/in/andrea-tarditi/)