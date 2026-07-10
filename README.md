# Quantitative Trading & Financial Engineering

**Market microstructure · Automated trading · Probabilistic forecasting · Systematic portfolio research**

[![Python](https://img.shields.io/badge/Python-Research%20%26%20Trading-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Probabilistic%20ML-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-Algorithms-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-Numerical%20Finance-EA7600)](https://www.mathworks.com/products/matlab.html)

I am **Andrea Tarditi**, an MSc graduate in Mathematical Engineering and Quantitative Finance from Politecnico di Milano. I build research systems that connect financial models with practical trading decisions, from market-data processing and signal construction to backtesting, risk diagnostics and deployment.

Previously, I supported the **UniCredit Government Bonds Trading Desk**, developing Python tools for position-risk analysis, yield-curve monitoring and intraday decision support.

[LinkedIn](https://www.linkedin.com/in/andrea-tarditi/) · [Email](mailto:andreatrdt@gmail.com) · [GitHub profile](https://github.com/andreatrdt)

---

## Featured projects

### 1. [Hawkes-Driven Market Making and Price Forecasting](./HawkesOptimalControl/)

An event-driven research framework for automated market making on high-frequency limit-order-book data.

**What it does**

- Reconstructs LOBSTER order books and cleans execution, submission and cancellation events.
- Calibrates multivariate Hawkes processes to model clustered aggressive buy and sell flow.
- Solves a finite-difference HJB problem to generate inventory-aware bid and ask quotes.
- Replays historical order-book events to evaluate fills, quote placement, inventory and P&L.
- Produces short-horizon predictive intervals for future mid-price changes using Hawkes and control-state features.

**Research focus**

The project tests how order-flow intensity, inventory risk and short-horizon price uncertainty can be combined in an automated quoting policy. Evaluation includes strategy-level diagnostics such as spread capture, adverse selection, inventory exposure and relative P&L, alongside forecast metrics such as empirical coverage, pinball loss and Winkler score.

`Python` `pandas` `NumPy` `SciPy` `scikit-learn` `Hawkes processes` `HJB control` `LOB replay`

---

### 2. [Electricity Price and Load Forecasting](./Electricity_Price_Load_Forecasting_3/)

A 24-hour-ahead forecasting pipeline comparing point forecasts with probabilistic neural-network models for electricity markets.

**What it does**

- Builds multi-horizon forecasts from historical, calendar and forward-looking features.
- Compares autoregressive and neural-network specifications.
- Implements quantile-regression, Gaussian and Johnson SU distributional DNNs.
- Evaluates forecast sharpness and calibration using pinball loss and Winkler interval scores.

**Selected result**

| Model | Average pinball loss | Average Winkler score |
|---|---:|---:|
| **Johnson SU-DNN** | **0.277** | **17.23** |
| Normal-DNN | 0.285 | 17.69 |
| QR-DNN | 0.332 | 20.41 |

The Johnson SU specification achieved the strongest overall probabilistic performance, improving average pinball loss by approximately **16.5%** relative to the QR-DNN benchmark.

`Python` `TensorFlow` `TensorFlow Probability` `Keras` `Probabilistic forecasting` `Time series`

---

### 3. [Big Levy Investments](./Big_Levy_Investments/)

A machine-learning portfolio research platform combining investor classification, index replication and anomaly-aware portfolio allocation.

**What it does**

- Maps investor characteristics to risk profiles and portfolio recommendations.
- Compares passive replication methods including Elastic Net, Kalman Filter and Ensemble Kalman Filter models.
- Uses tracking-error, risk and transaction-cost diagnostics to assess replica behaviour.
- Combines passive portfolios with anomaly signals to create active risk-management overlays.
- Connects the research pipeline to a Flask application for interactive portfolio recommendations.

This was developed as a five-person FinTech project, covering the complete workflow from data preparation and model tuning to portfolio construction and web deployment.

`Python` `scikit-learn` `Elastic Net` `Kalman filtering` `Optuna` `Flask` `Portfolio analytics`

---

## Additional quantitative work

| Area | Selected implementations |
|---|---|
| **Fixed income** | [Curve bootstrapping and sensitivity analysis](./A2_Bootstrap/) |
| **Credit and market risk** | [Credit derivatives](./A3_Credit_AS_CDS_FtD/) · [Monte Carlo risk analysis](./A4_RiskManagment/) |
| **Derivatives pricing** | [European option pricing](./A1_EU_OptionPricing/) · [Multivariate Lévy and structured-product models](./A7_StructuredProducts/) |
| **Portfolio modelling** | [Portfolio replication research](./BC3_PortfolioReplica/) |

---

## Research principles

- **Chronology before convenience:** time-series models use ordered validation rather than random splits where leakage would distort results.
- **Baselines before complexity:** advanced models are compared with simpler statistical or rule-based alternatives.
- **Trading diagnostics before headline returns:** P&L is assessed together with fills, inventory, drawdowns, adverse selection and risk exposure.
- **Models as systems:** projects cover data ingestion, modelling, evaluation, visualisation and deployment rather than isolated notebooks.
- **Reproducibility:** reusable modules, configuration files and documented workflows are preferred over one-off analysis.

## Technical stack

**Programming:** Python, C++, MATLAB  
**Data and modelling:** pandas, NumPy, SciPy, scikit-learn, TensorFlow, Keras, TensorFlow Probability, statsmodels  
**Finance:** market microstructure, backtesting, stochastic control, time-series modelling, Monte Carlo, portfolio analytics, derivatives pricing  
**Development:** Git, Jupyter, Flask, Streamlit

---

### Authorship note

Several projects in this repository were developed in academic teams. The relevant project documentation identifies collaborators and distinguishes shared work from individual extensions. All results shown above are drawn from the corresponding code and saved experiment outputs.

### Contact

For trading, quantitative research or analytical opportunities:

**Andrea Tarditi** · [andreatrdt@gmail.com](mailto:andreatrdt@gmail.com) · [LinkedIn](https://www.linkedin.com/in/andrea-tarditi/)
