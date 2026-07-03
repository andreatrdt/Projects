"""
CV-aligned probabilistic power price forecasting pipeline.

This script makes the project match the CV description in one reproducible entry point:

    Probabilistic Power Price Forecasting | Python | 2025
    Built probabilistic deep-learning models for 24-hour electricity price forecasting,
    comparing QR-DNN, Normal-DNN and Johnson SU-DNN approaches; evaluated Pinball loss
    and Winkler score; produced upside/downside price bands for scenario analysis.

It can either reuse saved recalibration results or rerun the recalibration loop.

Typical usage from Electricity_Price_Load_Forecasting_3/:

    python run_cv_project_pipeline.py

To force model retraining/recalibration:

    python run_cv_project_pipeline.py --run-recalibration

Outputs are written to:

    experiments/tasks/EM_price/cv_project_summary/
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tools.PrTSF_Recalib_tools import PrTsfRecalibEngine, load_data_model_configs


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_NAME = "EM_price"
DEFAULT_RUN_ID = "recalib_opt_random_1_2"

# These three experiments are the ones named in the CV bullet.
EXPERIMENTS = {
    "QR-DNN": {
        "exper_setup": "QR-DNN",
        "run_id": DEFAULT_RUN_ID,
        "apply_arcsinh_transf": False,
        "description": "Direct quantile-regression DNN baseline",
    },
    "Normal-DNN": {
        "exper_setup": "N-DNN",
        "run_id": DEFAULT_RUN_ID,
        "apply_arcsinh_transf": False,
        "description": "Distributional DNN with Normal output layer",
    },
    "Johnson SU-DNN": {
        "exper_setup": "JSU-DNN",
        "run_id": DEFAULT_RUN_ID,
        "apply_arcsinh_transf": False,
        "description": "Distributional DNN with Johnson SU output layer",
    },
}


def get_pred_horizon(configs: Dict) -> int:
    """
    Return the prediction horizon.

    In this codebase, `pred_horiz` originally lives in `data_config`. The recalibration
    engine also copies it into `model_config`, but that only happens after the engine is
    instantiated. When we load saved predictions directly, `model_config['pred_horiz']`
    may therefore be missing. This helper handles both cases.
    """
    if "pred_horiz" in configs["model_config"]:
        return int(configs["model_config"]["pred_horiz"])
    return int(configs["data_config"].pred_horiz)


def build_target_quantiles(target_alpha: List[float]) -> List[float]:
    """Build central-interval quantiles from alpha levels, including the median."""
    target_quantiles = [0.5]
    for alpha in target_alpha:
        target_quantiles.append(alpha / 2)
        target_quantiles.append(1 - alpha / 2)
    return sorted(target_quantiles)


def compute_pinball_scores(
    y_true: np.ndarray,
    pred_quantiles: np.ndarray,
    quantiles_levels: List[float],
) -> np.ndarray:
    """
    Compute Pinball scores by forecast hour and quantile.

    Shapes:
        y_true:          n_days x pred_horiz
        pred_quantiles:  n_days x pred_horiz x n_quantiles

    Return:
        pred_horiz x n_quantiles
    """
    score = []
    for i, q in enumerate(quantiles_levels):
        error = np.subtract(y_true, pred_quantiles[:, :, i])
        loss_q = np.maximum(q * error, (q - 1) * error)
        score.append(np.expand_dims(loss_q, -1))
    return np.mean(np.concatenate(score, axis=-1), axis=0)


def compute_winkler_scores(
    y_true: np.ndarray,
    pred_quantiles: np.ndarray,
    quantiles_levels: List[float],
) -> Tuple[np.ndarray, List[float]]:
    """
    Compute Winkler interval scores by forecast hour and central interval.

    For the central interval [q_tau, q_(1-tau)], the miscoverage level is alpha = 2*tau.
    The standard Winkler/interval score is:

        (U - L) + 2/alpha * (L - y)_+ + 2/alpha * (y - U)_+

    Return:
        scores:          pred_horiz x n_intervals
        lower_quantiles: list of lower quantiles defining each interval
    """
    score = []
    lower_quantiles = quantiles_levels[: len(quantiles_levels) // 2]

    for i, tau in enumerate(lower_quantiles):
        lower = pred_quantiles[:, :, i]
        upper = pred_quantiles[:, :, -i - 1]
        alpha = 2 * tau

        width = np.subtract(upper, lower)
        below_lower = np.maximum(np.subtract(lower, y_true), 0)
        above_upper = np.maximum(np.subtract(y_true, upper), 0)

        loss_q = width + 2 / alpha * (below_lower + above_upper)
        score.append(np.expand_dims(loss_q, -1))

    return np.mean(np.concatenate(score, axis=-1), axis=0), lower_quantiles


def get_target_column(predictions: pd.DataFrame, task_name: str) -> str:
    """Find the realized target column in a recalibration result DataFrame."""
    candidates = [task_name, f"TARG__{task_name}"]
    for candidate in candidates:
        if candidate in predictions.columns:
            return candidate
    raise KeyError(
        f"Could not find target column. Expected one of {candidates}, got {list(predictions.columns)}"
    )


def get_quantile_columns(predictions: pd.DataFrame, quantiles: List[float]) -> List:
    """Return DataFrame columns corresponding to the expected quantile levels."""
    cols = []
    for q in quantiles:
        if q in predictions.columns:
            cols.append(q)
        elif str(q) in predictions.columns:
            cols.append(str(q))
        else:
            raise KeyError(f"Missing quantile column for q={q}")
    return cols


def result_pickle_path(root: Path, task_name: str, exper_setup: str, run_id: str, optuna_m: str) -> Path:
    return (
        root
        / "experiments"
        / "tasks"
        / task_name
        / exper_setup
        / run_id
        / "results"
        / f"recalib_test_results-tuned-{optuna_m}.p"
    )


def load_or_run_experiment(
    model_name: str,
    exp_conf: Dict,
    task_name: str,
    hyper_mode: str,
    run_recalibration: bool,
) -> Tuple[pd.DataFrame, Dict]:
    """Load saved recalibration predictions, or run the recalibration loop if requested/needed."""
    configs = load_data_model_configs(
        task_name=task_name,
        exper_setup=exp_conf["exper_setup"],
        run_id=exp_conf["run_id"],
    )
    optuna_m = configs["model_config"]["optuna_m"]
    result_path = result_pickle_path(
        PROJECT_DIR,
        task_name,
        exp_conf["exper_setup"],
        exp_conf["run_id"],
        optuna_m,
    )

    if result_path.exists() and not run_recalibration:
        print(f"[{model_name}] loading saved predictions: {result_path}")
        with open(result_path, "rb") as file:
            return pickle.load(file), configs

    print(f"[{model_name}] running recalibration. This can be slow.")
    dataset_path = PROJECT_DIR / "data" / "datasets" / configs["data_config"].dataset_name
    ds = pd.read_csv(dataset_path)

    if exp_conf.get("apply_arcsinh_transf", False):
        ds[f"TARG__{task_name}"] = np.arcsinh(ds[f"TARG__{task_name}"])

    engine = PrTsfRecalibEngine(
        dataset=ds,
        data_configs=configs["data_config"],
        model_configs=configs["model_config"],
    )
    model_hyperparams = engine.get_model_hyperparams(
        method=hyper_mode,
        optuna_m=configs["model_config"]["optuna_m"],
    )
    predictions = engine.run_recalibration(
        model_hyperparams=model_hyperparams,
        plot_history=False,
        plot_weights=False,
    )

    if exp_conf.get("apply_arcsinh_transf", False):
        predictions = np.sinh(predictions)

    return predictions, configs


def score_experiment(
    model_name: str,
    predictions: pd.DataFrame,
    configs: Dict,
    task_name: str,
    output_dir: Path,
) -> Dict:
    """Compute Pinball/Winkler scores and write per-model artefacts."""
    pred_steps = get_pred_horizon(configs)
    quantiles = build_target_quantiles(configs["model_config"]["target_alpha"])
    target_col = get_target_column(predictions, task_name)
    quantile_cols = get_quantile_columns(predictions, quantiles)

    y_true = predictions[target_col].to_numpy().reshape(-1, pred_steps)
    pred_quantiles = predictions[quantile_cols].to_numpy().reshape(-1, pred_steps, len(quantiles))

    pinball = compute_pinball_scores(y_true, pred_quantiles, quantiles)
    winkler, lower_quantiles = compute_winkler_scores(y_true, pred_quantiles, quantiles)

    pinball_df = pd.DataFrame(
        pinball,
        columns=[f"q_{q:g}" for q in quantiles],
        index=[f"Hour {i + 1}" for i in range(pred_steps)],
    )
    winkler_df = pd.DataFrame(
        winkler,
        columns=[f"PI_{q:g}_{1 - q:g}" for q in lower_quantiles],
        index=[f"Hour {i + 1}" for i in range(pred_steps)],
    )

    safe_model_name = model_name.replace(" ", "_").replace("-", "_")
    pinball_df.to_csv(output_dir / f"{safe_model_name}_pinball_by_hour_quantile.csv")
    winkler_df.to_csv(output_dir / f"{safe_model_name}_winkler_by_hour_interval.csv")

    price_bands = build_price_bands(predictions, task_name, quantiles)
    price_bands.to_csv(output_dir / f"{safe_model_name}_price_bands.csv")

    return {
        "model": model_name,
        "PF_method": configs["model_config"]["PF_method"],
        "model_class": configs["model_config"]["model_class"],
        "description": EXPERIMENTS[model_name]["description"],
        "pred_horizon_hours": pred_steps,
        "test_rows": len(predictions),
        "test_days": len(predictions) / pred_steps,
        "avg_pinball_loss": float(np.mean(pinball)),
        "avg_winkler_score": float(np.mean(winkler)),
        "median_pinball_loss": float(np.median(pinball)),
        "median_winkler_score": float(np.median(winkler)),
        "price_bands_file": f"{safe_model_name}_price_bands.csv",
    }


def build_price_bands(predictions: pd.DataFrame, task_name: str, quantiles: List[float]) -> pd.DataFrame:
    """Create upside/downside bands used for scenario analysis."""
    target_col = get_target_column(predictions, task_name)

    def q_col(q: float):
        if q in predictions.columns:
            return q
        if str(q) in predictions.columns:
            return str(q)
        raise KeyError(f"Missing quantile column for q={q}")

    lower_q = 0.05 if 0.05 in quantiles else quantiles[0]
    upper_q = 0.95 if 0.95 in quantiles else quantiles[-1]
    median_q = 0.5

    band_df = pd.DataFrame(index=predictions.index)
    band_df["realized_price"] = predictions[target_col]
    band_df["median_price"] = predictions[q_col(median_q)]
    band_df[f"downside_q_{lower_q:g}"] = predictions[q_col(lower_q)]
    band_df[f"upside_q_{upper_q:g}"] = predictions[q_col(upper_q)]
    band_df["downside_risk_vs_median"] = band_df["median_price"] - band_df[f"downside_q_{lower_q:g}"]
    band_df["upside_potential_vs_median"] = band_df[f"upside_q_{upper_q:g}"] - band_df["median_price"]
    band_df["band_width"] = band_df[f"upside_q_{upper_q:g}"] - band_df[f"downside_q_{lower_q:g}"]
    return band_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CV-aligned probabilistic forecasting comparison.")
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--hyper-mode", default="load_tuned", choices=["load_tuned", "optuna_tuner"])
    parser.add_argument(
        "--run-recalibration",
        action="store_true",
        help="Force rerunning the recalibration loop instead of using saved predictions.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_DIR)
    output_dir = PROJECT_DIR / "experiments" / "tasks" / args.task_name / "cv_project_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    predictions_by_model = {}
    configs_by_model = {}

    for model_name, exp_conf in EXPERIMENTS.items():
        predictions, configs = load_or_run_experiment(
            model_name=model_name,
            exp_conf=exp_conf,
            task_name=args.task_name,
            hyper_mode=args.hyper_mode,
            run_recalibration=args.run_recalibration,
        )
        predictions_by_model[model_name] = predictions
        configs_by_model[model_name] = configs
        summary_rows.append(
            score_experiment(
                model_name=model_name,
                predictions=predictions,
                configs=configs,
                task_name=args.task_name,
                output_dir=output_dir,
            )
        )

    summary = pd.DataFrame(summary_rows).sort_values("avg_pinball_loss")
    summary.to_csv(output_dir / "model_comparison_summary.csv", index=False)

    best_model = summary.iloc[0]["model"]
    best_configs = configs_by_model[best_model]
    best_bands = build_price_bands(
        predictions_by_model[best_model],
        task_name=args.task_name,
        quantiles=build_target_quantiles(best_configs["model_config"]["target_alpha"]),
    )
    best_bands.to_csv(output_dir / "best_model_price_bands.csv")

    print("\n=== CV project summary ===")
    print(summary.to_string(index=False))
    print(f"\nBest model by average Pinball loss: {best_model}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
