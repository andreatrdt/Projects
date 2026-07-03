# Probabilistic Power Price Forecasting

This folder supports the CV project:

> Built probabilistic deep-learning models for 24-hour electricity price forecasting, comparing QR-DNN, Normal-DNN and Johnson SU-DNN approaches; evaluated predictive distributions with Pinball loss and Winkler score, producing upside/downside price bands for scenario analysis and risk-aware trading decisions.

## What the project forecasts

The target is the next 24 hourly values of `TARG__EM_price`.

Each sample is built as:

```text
X = previous 7 days of hourly price history
    + future-known load, solar and wind forecasts for the next 24 hours
    + calendar features

Y = next 24 hourly electricity prices
```

The main dataset contains:

```text
TARG__EM_price
FUTU__EM_load_f
FUTU__EM_solar_f
FUTU__EM_wind_f
CONST__wd_sin, CONST__wd_cos
CONST__mnth_sin, CONST__mnth_cos
CONST__yd_sin, CONST__yd_cos
```

## Models compared

The CV-aligned comparison uses three DNN specifications:

| Model | `exper_setup` | `PF_method` | Output interpretation |
|---|---:|---:|---|
| QR-DNN | `QR-DNN` | `qr` | Direct quantile forecasts |
| Normal-DNN | `N-DNN` | `Normal` | Normal loc and scale for each forecast hour |
| Johnson SU-DNN | `JSU-DNN` | `JSU` | Johnson SU loc, scale, tailweight and skewness for each forecast hour |

QR-DNN is trained with Pinball loss. Normal-DNN and Johnson SU-DNN are trained with negative log-likelihood.

## Recalibration setup

The project uses rolling daily recalibration.

For each test day:

```text
1. Build historical rolling windows.
2. Use older windows for training.
3. Use the most recent validation windows for early stopping/model selection.
4. Use the final window as the current test day.
5. Train a fresh model.
6. Predict the next 24-hour distribution.
```

So the model is not trained once and tested once. It is retrained for each daily test block.

## Evaluation metrics

### Pinball loss

Pinball loss evaluates individual predicted quantiles. It is asymmetric: for a high quantile such as q_0.95, underpredicting is penalized more than overpredicting.

### Winkler score

Winkler score evaluates central prediction intervals such as `[q_0.05, q_0.95]`. It rewards narrow intervals but adds a penalty when the realized price falls outside the interval.

## CV-aligned runner

Run the comparison from this directory with:

```bash
python run_cv_project_pipeline.py
```

This reuses saved recalibration results when available.

To force retraining/recalibration:

```bash
python run_cv_project_pipeline.py --run-recalibration
```

The script writes outputs to:

```text
experiments/tasks/EM_price/cv_project_summary/
```

Main outputs:

```text
model_comparison_summary.csv
QR_DNN_pinball_by_hour_quantile.csv
Normal_DNN_pinball_by_hour_quantile.csv
Johnson_SU_DNN_pinball_by_hour_quantile.csv
QR_DNN_winkler_by_hour_interval.csv
Normal_DNN_winkler_by_hour_interval.csv
Johnson_SU_DNN_winkler_by_hour_interval.csv
QR_DNN_price_bands.csv
Normal_DNN_price_bands.csv
Johnson_SU_DNN_price_bands.csv
best_model_price_bands.csv
```

## Interview version

A concise explanation:

> I built a probabilistic day-ahead electricity price forecasting pipeline. Instead of predicting only a point estimate, I forecasted the conditional distribution of the next 24 hourly prices. I compared QR-DNN, Normal-DNN and Johnson SU-DNN. QR-DNN directly outputs quantiles and is trained with Pinball loss, while Normal-DNN and Johnson SU-DNN output distribution parameters and are trained with negative log-likelihood. I evaluated the forecasts using Pinball loss and Winkler score, then converted the predictive distributions into upside and downside price bands for scenario analysis and risk-aware trading decisions.
