# Financial Engineering Projects

---

## Table of Contents
- [Overview](#overview)
- [Assignments](#assignments)
  - [Assignment 1: Option Pricing and Error Analysis](#assignment-1-option-pricing-and-error-analysis)
  - [Assignment 2: Yield Curves and Sensitivities](#assignment-2-yield-curves-and-sensitivities)
  - [Assignment 3: Asset Swaps, CDS, and Python Time-Series Analysis](#assignment-3-asset-swaps-cds-and-python-time-series-analysis)
  - [Assignment 4: Value at Risk (VaR) and Expected Shortfall (ES)](#assignment-4-value-at-risk-var-and-expected-shortfall-es)
  - [Assignment 5: Advanced Derivative Pricing and Volatility Surface Calibration](#assignment-5-advanced-derivative-pricing-and-volatility-surface-calibration)
  - [Assignment 6: Interest Rate Risk and Hedging](#assignment-6-interest-rate-risk-and-hedging)
  - [Assignment 7: Bermudan Swaptions and Certificate Pricing](#assignment-7-bermudan-swaptions-and-certificate-pricing)
- [Energy Price and Load Forecasting](#energy-price-and-load-forecasting)
  - [EPLF Assignment 1: Regularization Techniques in Energy Price Forecasting](#eplf-assignment-1-regularization-techniques-in-energy-price-forecasting)
  - [EPLF Assignment 2: Deep Neural Networks (DNN) Hyperparameter Tuning](#eplf-assignment-2-deep-neural-networks-dnn-hyperparameter-tuning)
  - [EPLF Assignment 3: Distributional Neural Networks and Quantile Regression](#eplf-assignment-3-distributional-neural-networks-and-quantile-regression)
- [Risk Management](#risk-management)
  - [RM Assignment 1: Hazard Rate and Z-Spread Calculation](#rm-assignment-1-hazard-rate-and-z-spread-calculation)
  - [RM Assignment 2: Present Value and Credit VaR](#rm-assignment-2-present-value-and-credit-var)
- [Multivariate Pricing for Financial Derivatives](#multivariate-pricing-for-financial-derivatives)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Additional Features](#additional-features)

---

## Overview

This repository contains a series of assignments for the Financial Engineering course **(2023-2024)** developed by **Group 16**. The projects cover a wide range of topics related to financial derivatives, pricing models, numerical methods, and risk management. Each assignment delves into state-of-the-art techniques—from analytical methods to advanced Monte Carlo simulations and deep learning for time-series analysis.

Explore interactive notebooks and detailed documentation for each project. Use the table of contents to navigate to sections that interest you!

---

## Assignments

### Assignment 1: Option Pricing and Error Analysis
- **European Call Option Pricing:** Computed using Black’s formula, CRR binomial tree, and Monte Carlo (MC) methods.
- **Error Rescaling:** Evaluated pricing error and determined optimal parameters.
- **Exotic Options:** Implemented pricing for a Knock-In Call option using multiple approaches.
- **Vega Sensitivity:** Analyzed Vega via numerical and analytical methods.
- **Bermudan Option Pricing:** Explored pricing behavior compared to vanilla European options.
- **Monte Carlo Enhancements:** Reduced variance using antithetic variables.

[🔗 More Details & Code](./A1_EU_OptionPricing)

### Assignment 2: Yield Curves and Sensitivities
- **Bootstrap Yield Curve Construction:** Built discount factor curves and zero rates.
- **Sensitivity Analysis:** Computed DV01, BPV, and duration for interest rate swaps.
- **Theoretical Exercises:** Derived bond pricing models and applied the Garman-Kohlhagen formula.

[🔗 More Details & Code](./A2_Bootstrap)

### Assignment 3: Asset Swaps, CDS, and Python Time-Series Analysis
- **Asset Swap Spread:** Calculated spreads using market data.
- **CDS Bootstrapping:** Constructed CDS curves via bootstrapping and spline interpolation.
- **First-to-Default Pricing:** Priced a First-to-Default (FtD) contract with Monte Carlo simulations.
- **Python Time-Series Analysis:** Implemented log-return plotting and regression analysis.

[🔗 More Details & Code](./A3_Credit_AS_CDS_FtD)

### Assignment 4: Value at Risk (VaR) and Expected Shortfall (ES)
- **Variance-Covariance Method:** Computed VaR and ES at a 99% confidence level.
- **Historical Simulation & Bootstrap:** Evaluated VaR accuracy.
- **Principal Component Analysis (PCA):** Applied PCA to reduce dimensionality.
- **Monte Carlo VaR:** Implemented Delta-Normal and full simulation approaches.
- **Cliquet Option Pricing:** Explored pricing under counterparty risk.

[🔗 More Details & Code](./A4_RiskManagment)

### Assignment 5: Advanced Derivative Pricing and Volatility Surface Calibration
- **Certificate Pricing:** Priced certificates using Monte Carlo simulations.
- **Digital Option Pricing:** Compared methods including Black-Scholes and implied volatility.
- **Lewis Formula and FFT:** Utilized FFT for efficient option pricing.
- **Volatility Surface Calibration:** Calibrated a surface using a mean-variance mixture model.

[🔗 More Details & Code](./A5_StructuredProducts)

### Assignment 6: Interest Rate Risk and Hedging
- **Bootstrap Market Discounts:** Extended bootstrapping for long-term interest rates.
- **Caplet Pricing:** Calibrated spot volatilities using the Bachelier formula.
- **Upfront Payment Calculation:** Priced structured interest rate products.
- **Delta and Vega Sensitivities:** Developed hedging strategies with swaps and caps.

[🔗 More Details & Code](./A6_StructuredProducts)

### Assignment 7: Bermudan Swaptions and Certificate Pricing
- **Bermudan Swaption Pricing:** Applied the Hull-White model with tree-based methods.
- **Certificate Pricing via NIG Model:** Employed FFT and Monte Carlo methods.
- **Black Model Adjustments:** Evaluated the impact of digital risk adjustments.

[🔗 More Details & Code](./A7_StructuredProducts)

---

## Energy Price and Load Forecasting

### EPLF Assignment 1: Regularization Techniques in Energy Price Forecasting
- **Lasso, Ridge, and Elastic Net Regression:** Improved forecasting accuracy for electricity prices.
- **Feature Selection:** Identified key predictors for time-series models.
- **Seasonality Analysis:** Explored seasonal trends impacting energy prices.

[🔗 More Details & Code](./Electricity_Price_Load_Forecasting_1)

### EPLF Assignment 2: Deep Neural Networks (DNN) Hyperparameter Tuning
- **Hyperparameter Optimization:** Used Optuna for random search optimization.
- **Loss Function Minimization:** Analyzed different configurations and their effects.
- **DNN Performance:** Examined overfitting and generalization in deep models.

[🔗 More Details & Code](./Electricity_Price_Load_Forecasting_2)

### EPLF Assignment 3: Distributional Neural Networks and Quantile Regression
- **Quantile Regression Neural Networks:** Captured data distribution boundaries with pinball loss.
- **Distributional Neural Networks (DNN):** Modeled probabilistic forecasting using Normal and Johnson's SU distributions.
- **Model Comparison:** Evaluated performance using Pinball and Winkler scores.

[🔗 More Details & Code](./Electricity_Price_Load_Forecasting_3)

---

## Risk Management

### RM Assignment 1: Hazard Rate and Z-Spread Calculation
- **Hazard Rate Curve Bootstrapping:** Constructed hazard rate curves for investment grade (IG) and high yield (HY) bonds.
- **Z-Spread Calculation:** Aligned defaultable and risk-free bond prices using parallel shifts.
- **Market-Implied Transition Matrix:** Developed transition matrices for rating migrations.

[🔗 More Details & Code](./Assignment1_RM)

### RM Assignment 2: Present Value and Credit VaR
- **Present Value Calculation:** Evaluated present value with Monte Carlo simulations, accounting for default scenarios.
- **Credit VaR Simulation:** Assessed risk under different correlation conditions.
- **Concentration Risk Analysis:** Demonstrated the benefits of diversification.

[🔗 More Details & Code](./Assignment2_RM)

---

## Multivariate Pricing for Financial Derivatives

This final project focuses on:
- **Multivariate Lévy Model:** Calibrating models with NIG marginals for S&P 500 and EURO STOXX 50.
- **Martingality and Drift Compensation:** Ensuring forward prices are martingales.
- **Calibration Techniques:** Joint calibration of marginal and dependence parameters.
- **Synthetic Forwards:** Using put/call parity for forward price computation.
- **Model Comparison:** Evaluating performance versus the classic Black model.
- **Exotic Derivative Pricing:** Pricing a derivative with conditional payoffs.

[🔗 More Details & Code](./FinalProject)

---
