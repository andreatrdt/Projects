from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.exceptions import TrialPruned


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _as_1d_array(y) -> np.ndarray:
    arr = np.asarray(y)
    return arr.reshape(-1)


def _best_f1_threshold(y_true, scores, default: float = 0.5) -> Tuple[float, float]:
    """Return (best_f1, threshold), using only the supplied calibration sample."""
    y_true = _as_1d_array(y_true).astype(int)
    scores = _as_1d_array(scores).astype(float)

    mask = np.isfinite(scores)
    y_true = y_true[mask]
    scores = scores[mask]

    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return 0.0, float(default)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.0, float(default)

    # precision/recall have one extra terminal element compared with thresholds.
    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    f1_vals = np.divide(
        2.0 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )

    idx = int(np.nanargmax(f1_vals))
    return float(f1_vals[idx]), float(thresholds[idx])


def _seed_torch(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# EDA
# -----------------------------------------------------------------------------

class AnomalyDetectionEDA:
    """Exploratory utilities for the market anomaly dataset."""

    def __init__(self, data_df=None, metadata_df=None, X_df=None, y=None):
        self.data_df = data_df
        self.metadata_df = metadata_df
        self.X_df = X_df
        self.y = y

    def generate_metadata(self):
        enhanced_metadata = []
        ticker_col = "ticker" if "ticker" in self.metadata_df.columns else self.metadata_df.columns[0]
        desc_col = "description" if "description" in self.metadata_df.columns else self.metadata_df.columns[1]

        for ticker in self.X_df.columns:
            meta_row = self.metadata_df[self.metadata_df[ticker_col] == ticker]
            description = meta_row[desc_col].values[0] if not meta_row.empty else ticker
            series = self.X_df[ticker]
            enhanced_metadata.append(
                {
                    "Ticker": ticker,
                    "Description": description,
                    "Mean": series.mean(),
                    "Std.Dev": series.std(),
                    "Min": series.min(),
                    "Max": series.max(),
                    "Missing values": series.isna().sum(),
                    "Missing (%)": f"{series.isna().mean() * 100:.2f}%",
                }
            )

        meta_df = pd.DataFrame(enhanced_metadata)
        print("\nEnhanced Metadata:")
        try:
            from IPython.display import display
            display(meta_df)
        except Exception:
            print(meta_df)
        return meta_df

    def plot_anomalies(self, index_col="MXUS"):
        if self.y is None or index_col not in self.X_df.columns:
            print("Either labels or index_col are missing.")
            return

        y_ser = pd.Series(_as_1d_array(self.y), index=self.X_df.index)
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(self.X_df.index, self.X_df[index_col], linewidth=2.5, label=index_col)

        first = True
        for date in self.X_df.index[y_ser.eq(1)]:
            ax.axvspan(
                date,
                date + pd.Timedelta(days=7),
                alpha=0.3,
                label="Risk-on/Risk-off" if first else None,
            )
            first = False

        ax.set_xlabel("Timeline")
        ax.set_ylabel(index_col)
        ax.set_title(f"{index_col} and risk-on/risk-off periods")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        plt.show()

    def filter_negative_anomalies(self, cols_to_use):
        """
        Keep a labelled anomaly only when at least one selected asset has a
        negative contemporaneous log return.

        IMPORTANT: the method updates BOTH data_df['Y'] and self.y and returns the
        updated label Series, so downstream models use the same labels shown in EDA.
        """
        missing = [c for c in cols_to_use if c not in self.X_df.columns]
        if missing:
            raise KeyError(f"Columns not found in X_df: {missing}")

        selected = self.X_df[cols_to_use].astype(float)
        if (selected <= 0).any().any():
            raise ValueError(
                "filter_negative_anomalies uses log returns, but at least one selected "
                "series contains non-positive levels. Use an appropriate return/difference "
                "transformation for those columns instead."
            )

        returns = np.log(selected / selected.shift(1))
        has_negative_return = (returns < 0).any(axis=1)

        current_y = pd.Series(_as_1d_array(self.y), index=self.X_df.index, name="Y").astype(int)
        original_anomalies = current_y.eq(1)
        updated_y = current_y.copy()
        updated_y.loc[original_anomalies & ~has_negative_return] = 0

        self.y = updated_y
        if self.data_df is not None and "Y" in self.data_df.columns:
            self.data_df.loc[updated_y.index, "Y"] = updated_y

        print(f"Number of anomalies with negative returns: {int(updated_y.sum())}")
        print(f"Original number of anomalies: {int(original_anomalies.sum())}")
        print(f"Removed non-negative anomalies: {int(original_anomalies.sum() - updated_y.sum())}")

        if self.data_df is not None and "Y" in self.data_df.columns:
            assert int(self.data_df.loc[updated_y.index, "Y"].sum()) == int(updated_y.sum())

        return updated_y

    def visualize_assets(self, asset_columns, bonds):
        returns_dict = {}
        for col in asset_columns:
            if col in bonds:
                returns_dict[col] = np.diff(self.X_df[col].astype(float))
            else:
                vals = self.X_df[col].astype(float)
                if (vals <= 0).any():
                    raise ValueError(f"Cannot take log returns of non-positive series {col}.")
                returns_dict[col] = np.diff(np.log(vals))

        fig, axes = plt.subplots(4, len(asset_columns), figsize=(18, 16), squeeze=False)
        fig.suptitle("Asset Analysis: Examples of Bond and Equity Returns", fontsize=16)
        plt.subplots_adjust(hspace=0.4)

        for i, col in enumerate(asset_columns):
            axes[0, i].plot(self.X_df.index, self.X_df[col])
            axes[0, i].set_title(f"{col}: time plot")
            axes[1, i].plot(self.X_df.index[1:], returns_dict[col])
            axes[1, i].set_title(f'{col}: {"First Differences" if col in bonds else "Log Returns"}')
            axes[2, i].hist(returns_dict[col], bins=50, alpha=0.7)
            axes[2, i].set_title(f"{col}: returns distribution")
            stats.probplot(returns_dict[col], dist="norm", plot=axes[3, i])
            axes[3, i].set_title(f"{col}: QQplot vs Gaussian")

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    def plot_interest_rate_distribution(self, column="GTDEM2Y"):
        if column not in self.X_df.columns:
            print(f"Column '{column}' not found in the dataset.")
            return
        plt.figure(figsize=(12, 6))
        plt.hist(self.X_df[column].dropna(), bins=50, alpha=0.7)
        plt.title(f"{column} yield: Checking for negative yields")
        plt.xlabel(f"{column} yield")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def adf_stationarity_test(self):
        def adf_test(series):
            result = adfuller(series.dropna())
            return {
                "ADF Statistic": result[0],
                "p-value": result[1],
                "Used Lag": result[2],
                "Observations Used": result[3],
                "Critical Value (1%)": result[4]["1%"],
                "Critical Value (5%)": result[4]["5%"],
                "Critical Value (10%)": result[4]["10%"],
            }

        adf_results = {col: adf_test(self.X_df[col]) for col in self.X_df.columns}
        adf_df = pd.DataFrame(adf_results).T
        adf_df["Stationary (<0.05 p-value)"] = adf_df["p-value"] < 0.05
        adf_df = adf_df.sort_values(by="p-value")
        return adf_df


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

class AnomalyDataPreparer:
    """
    Causal time-series preparation.

    Defaults are intentionally chronological. Scaling parameters are fitted on
    training data only. Date cutoffs are preferred to fractions because
    differencing changes the number of rows by one.
    """

    def __init__(
        self,
        shuffle_data: bool = False,
        scale_data: bool = True,
        do_split: bool = True,
        chronological_split: bool = True,
        make_stationary: bool = True,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        random_state: int = 42,
        train_end: Optional[pd.Timestamp] = None,
        val_end: Optional[pd.Timestamp] = None,
    ):
        self.shuffle_data = shuffle_data
        self.scale_data = scale_data
        self.do_split = do_split
        self.chronological_split = chronological_split
        self.make_stationary = make_stationary
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.random_state = random_state
        self.train_end = pd.Timestamp(train_end) if train_end is not None else None
        self.val_end = pd.Timestamp(val_end) if val_end is not None else None
        self.scaler = StandardScaler()

        if not (0 < train_frac < 1):
            raise ValueError("train_frac must be in (0, 1).")
        if not (0 <= val_frac < 1):
            raise ValueError("val_frac must be in [0, 1).")
        if self.train_end is None and train_frac + val_frac >= 1:
            raise ValueError("train_frac + val_frac must be < 1.")
        if (self.train_end is None) != (self.val_end is None):
            raise ValueError("train_end and val_end must be supplied together.")
        if self.train_end is not None and self.train_end >= self.val_end:
            raise ValueError("train_end must be strictly earlier than val_end.")
        if self.shuffle_data and self.chronological_split:
            raise ValueError("Cannot request both shuffle_data=True and chronological_split=True.")

    def make_data_stationary(self, X_df, y=None):
        X_df = pd.DataFrame(X_df).copy()
        X_df.index = pd.to_datetime(X_df.index)
        X_df = X_df.sort_index()

        indices_currencies = {
            "XAUBGNL", "BDIY", "CRY", "Cl1", "DXY", "EMUSTRUU", "GBP", "JPY",
            "LF94TRUU", "LF98TRUU", "LG30TRUU", "LMBITR", "LP01TREU", "LUACTRUU",
            "LUMSTRUU", "MXBR", "MXCN", "MXEU", "MXIN", "MXJP", "MXRU", "MXUS", "VIX",
        }
        interest_rates = {
            "EONIA", "GTDEM10Y", "GTDEM2Y", "GTDEM30Y", "GTGBP20Y", "GTGBP2Y",
            "GTGBP30Y", "GTITL10YR", "GTITL2YR", "GTITL30YR", "GTJPY10YR",
            "GTJPY2YR", "GTJPY30YR", "US0001M", "USGG3M", "USGG2YR", "GT10", "USGG30YR",
        }

        out = pd.DataFrame(index=X_df.index)
        for col in X_df.columns:
            s = X_df[col].astype(float)
            if col in indices_currencies:
                if (s <= 0).any():
                    # Log differences are invalid for non-positive levels.
                    out[col] = s.diff()
                else:
                    out[col] = np.log(s).diff()
            elif col in interest_rates:
                out[col] = s.diff()
            elif col == "ECSURPUS":
                # This series was treated as already stationary in the original project.
                out[col] = s
            else:
                # Preserve unclassified features rather than silently dropping them.
                out[col] = s

        # Only the first row is structurally missing because of one-step transformations.
        out = out.iloc[1:].copy()
        if out.isna().any().any():
            bad = out.columns[out.isna().any()].tolist()
            raise ValueError(f"NaNs remain after stationarization in columns: {bad}")

        y_out = None
        if y is not None:
            if isinstance(y, pd.Series):
                y_ser = y.copy()
                y_ser.index = pd.to_datetime(y_ser.index)
                y_ser = y_ser.sort_index().loc[out.index]
            else:
                arr = _as_1d_array(y)
                if len(arr) != len(X_df):
                    raise ValueError("y length must match X_df length before stationarization.")
                y_ser = pd.Series(arr, index=X_df.index).loc[out.index]
            y_out = y_ser.astype(int)

        return out, y_out

    def prepare(self, X_df, y):
        X_df = pd.DataFrame(X_df).copy()
        X_df.index = pd.to_datetime(X_df.index)
        X_df = X_df.sort_index()

        if isinstance(y, pd.Series):
            y_ser = y.copy()
            y_ser.index = pd.to_datetime(y_ser.index)
            y_ser = y_ser.reindex(X_df.index)
        else:
            y_ser = pd.Series(_as_1d_array(y), index=X_df.index)

        if y_ser.isna().any():
            raise ValueError("Labels are not aligned with X_df.")

        if self.make_stationary:
            X_df, y_ser = self.make_data_stationary(X_df, y_ser)

        if self.shuffle_data:
            rng = np.random.default_rng(self.random_state)
            order = rng.permutation(len(X_df))
            X_df = X_df.iloc[order]
            y_ser = y_ser.iloc[order]
        elif self.chronological_split:
            X_df = X_df.sort_index()
            y_ser = y_ser.loc[X_df.index]

        if not self.do_split:
            if self.scale_data:
                vals = self.scaler.fit_transform(X_df)
                X_df = pd.DataFrame(vals, index=X_df.index, columns=X_df.columns)
            return X_df, y_ser

        if self.train_end is not None:
            X_train = X_df.loc[X_df.index < self.train_end]
            X_val = X_df.loc[(X_df.index >= self.train_end) & (X_df.index < self.val_end)]
            X_test = X_df.loc[X_df.index >= self.val_end]
        else:
            n = len(X_df)
            train_size = int(self.train_frac * n)
            val_size = int(self.val_frac * n)
            X_train = X_df.iloc[:train_size]
            X_val = X_df.iloc[train_size: train_size + val_size]
            X_test = X_df.iloc[train_size + val_size:]

        y_train = y_ser.loc[X_train.index]
        y_val = y_ser.loc[X_val.index]
        y_test = y_ser.loc[X_test.index]

        if min(len(X_train), len(X_val), len(X_test)) == 0:
            raise ValueError(
                f"Empty split produced: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}."
            )

        if self.scale_data:
            X_train_vals = self.scaler.fit_transform(X_train)
            X_val_vals = self.scaler.transform(X_val)
            X_test_vals = self.scaler.transform(X_test)
            X_train = pd.DataFrame(X_train_vals, index=X_train.index, columns=X_df.columns)
            X_val = pd.DataFrame(X_val_vals, index=X_val.index, columns=X_df.columns)
            X_test = pd.DataFrame(X_test_vals, index=X_test.index, columns=X_df.columns)

        print(f"Training set size: {len(X_train)} ({int((y_train == 0).sum())} normal, {int((y_train == 1).sum())} anomalies)")
        print(f"Validation set size: {len(X_val)} ({int((y_val == 0).sum())} normal, {int((y_val == 1).sum())} anomalies)")
        print(f"Test set size: {len(X_test)} ({int((y_test == 0).sum())} normal, {int((y_test == 1).sum())} anomalies)")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def raw_training_subset(self, X_df, y):
        """Return the raw chronological training subset for leakage-free metric calibration."""
        X_df = pd.DataFrame(X_df).copy()
        X_df.index = pd.to_datetime(X_df.index)
        X_df = X_df.sort_index()
        y_ser = y.copy() if isinstance(y, pd.Series) else pd.Series(_as_1d_array(y), index=X_df.index)
        y_ser.index = pd.to_datetime(y_ser.index)
        y_ser = y_ser.reindex(X_df.index)

        if self.train_end is not None:
            idx = X_df.index < self.train_end
            return X_df.loc[idx], y_ser.loc[idx]

        n_train = int(self.train_frac * len(X_df))
        return X_df.iloc[:n_train], y_ser.iloc[:n_train]


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

class DetectionMethodsEvaluation:
    """Leakage-free evaluation and business-oriented score."""

    def __init__(
        self,
        preparer: AnomalyDataPreparer,
        metadata_df: pd.DataFrame,
        X_df: pd.DataFrame,
        y: pd.Series,
        calibration_X: Optional[pd.DataFrame] = None,
        calibration_y: Optional[pd.Series] = None,
        action_cost: float = 0.2,
    ):
        self.preparer = preparer
        self.metadata_df = metadata_df
        self.X_df = X_df
        self.y = y
        self.action_cost = float(action_cost)

        if calibration_X is None or calibration_y is None:
            calibration_X, calibration_y = preparer.raw_training_subset(X_df, y)

        self.calibration_X = calibration_X
        self.calibration_y = calibration_y
        self.mean_normal_weighted = None
        self.mean_anomaly_weighted = None
        self._compute_weighted_means()

    def _compute_weighted_means(self):
        stationary_df, y_stationary = self.preparer.make_data_stationary(
            self.calibration_X, self.calibration_y
        )

        # Put heterogeneous market variables onto a comparable z-score scale.
        scaler = StandardScaler()
        standardized = pd.DataFrame(
            scaler.fit_transform(stationary_df),
            index=stationary_df.index,
            columns=stationary_df.columns,
        )

        variable_col = "Variable name" if "Variable name" in self.metadata_df.columns else self.metadata_df.columns[0]
        type_col = "Type" if "Type" in self.metadata_df.columns else self.metadata_df.columns[1]
        class_map = self.metadata_df.set_index(variable_col)[type_col].to_dict()

        bucket_map = defaultdict(list)
        for ticker, cls in class_map.items():
            if ticker in standardized.columns:
                bucket_map[cls].append(ticker)

        if not bucket_map:
            raise ValueError("No metadata asset classes matched the anomaly feature columns.")

        ratio_df = pd.DataFrame(index=standardized.index)
        for cls, tickers in bucket_map.items():
            col_name = f"{str(cls).lower().replace(' ', '_')}_ratio"
            ratio_df[col_name] = standardized[tickers].mean(axis=1)

        y_stationary = pd.Series(_as_1d_array(y_stationary), index=ratio_df.index, name="Y").astype(int)

        if not {0, 1}.issubset(set(y_stationary.unique())):
            raise ValueError(
                "Financial-score calibration requires both normal and anomaly labels in the training calibration sample."
            )

        corrs = ratio_df.corrwith(y_stationary).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        abs_corr = corrs.abs()
        if abs_corr.sum() > 0:
            w = abs_corr / abs_corr.sum()
        else:
            w = pd.Series(1.0 / len(abs_corr), index=abs_corr.index)

        grouped = ratio_df.assign(Y=y_stationary).groupby("Y").mean()
        mean_normal = grouped.loc[0]
        mean_anomaly = grouped.loc[1]

        common = w.index.intersection(mean_normal.index).intersection(mean_anomaly.index)
        w = w.loc[common]
        self.mean_normal_weighted = float((mean_normal.loc[common] * w).sum())
        self.mean_anomaly_weighted = float((mean_anomaly.loc[common] * w).sum())

    def financial_score(self, y_true, y_pred):
        y_true = _as_1d_array(y_true).astype(int)
        y_pred = _as_1d_array(y_pred).astype(int)
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

        cost_anomaly = 100.0 * abs(self.mean_anomaly_weighted)
        cost_normal = 10.0 * abs(self.mean_normal_weighted)
        action = self.action_cost

        fn = ((y_true == 1) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()

        total_cost = fn * cost_anomaly + fp * cost_normal + y_pred.sum() * action

        # Normalize relative to perfect classification so a perfect classifier scores 1.
        perfect_cost = (y_true == 1).sum() * action
        worst_cost = (
            (y_true == 1).sum() * max(cost_anomaly, action)
            + (y_true == 0).sum() * (cost_normal + action)
        )

        denom = worst_cost - perfect_cost
        if denom <= 0:
            return 1.0
        score = 1.0 - (total_cost - perfect_cost) / denom
        return float(np.clip(score, 0.0, 1.0))

    def evaluate_model(self, y_true, y_pred, y_score, model_name):
        y_true = _as_1d_array(y_true).astype(int)
        y_pred = _as_1d_array(y_pred).astype(int)
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"{model_name}: y_true has length {len(y_true)} but y_pred has length {len(y_pred)}."
            )

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fin_score = self.financial_score(y_true, y_pred)

        print(f"\n{model_name} Performance:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Financial Score: {fin_score:.4f}")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.show()

        if y_score is not None:
            y_score = _as_1d_array(y_score).astype(float)
            if len(y_score) != len(y_true):
                raise ValueError(
                    f"{model_name}: y_score has length {len(y_score)} but y_true has length {len(y_true)}."
                )
            if np.unique(y_true).size == 2:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
                plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve - {model_name}")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.show()

                precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                plt.figure(figsize=(8, 6))
                plt.plot(recall_vals, precision_vals, lw=2, label=f"AP = {ap:.2f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"Precision-Recall Curve - {model_name}")
                plt.legend(loc="upper right")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.show()

        return float(precision), float(recall), float(f1), float(fin_score)


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------

class MLPAnomalyDetector:
    def __init__(self, input_dim, random_state: int = 42):
        self.input_dim = input_dim
        self.random_state = random_state
        self.best_params = None
        self.best_threshold = None
        self.model = None

    def objective(self, trial, X_train, y_train, X_val, y_val):
        # The original 5-7 layer / 512-1024 unit search was far too large for a
        # weekly anomaly dataset. This range is still flexible but materially safer.
        n_layers = trial.suggest_int("n_layers", 1, 4)
        first_layer_size = trial.suggest_int("n_units_l0", 32, 256, log=True)
        hidden_layer_sizes = [first_layer_size]
        prev_size = first_layer_size
        for i in range(1, n_layers):
            low = max(8, prev_size // 4)
            high = max(low, prev_size)
            next_size = trial.suggest_int(f"n_units_l{i}", low, high, log=True)
            hidden_layer_sizes.append(next_size)
            prev_size = next_size

        lr = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)

        model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            learning_rate_init=lr,
            alpha=alpha,
            activation="relu",
            solver="adam",
            max_iter=1500,
            early_stopping=False,
            random_state=self.random_state,
        )
        model.fit(X_train, _as_1d_array(y_train))
        probs = model.predict_proba(X_val)[:, 1]
        best_f1, best_thresh = _best_f1_threshold(y_val, probs)
        trial.set_user_attr("best_threshold", best_thresh)
        return best_f1

    def tune_hyperparameters(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=100,
        storage="sqlite:///optuna_MLP.db",
    ):
        study = optuna.create_study(
            study_name="MLP_Study_v2",
            storage=storage,
            load_if_exists=True,
            direction="maximize",
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if n_trials > 0:
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
                n_trials=n_trials,
                show_progress_bar=True,
            )

        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise RuntimeError("MLP Optuna study has no COMPLETE trials.")

        self.best_params = study.best_params
        self.best_threshold = float(study.best_trial.user_attrs["best_threshold"])
        print("Best validation F1:", study.best_value)
        print("Best hyperparameters:", self.best_params)
        print("Best validation threshold:", self.best_threshold)
        return study

    def train_final_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Refit on TRAIN ONLY for a clean test evaluation.

        The threshold was calibrated on validation data using a model fitted on the
        training set. Re-fitting on train+validation would change the probability
        calibration while keeping the old threshold, which is inconsistent.
        """
        if self.best_params is None or self.best_threshold is None:
            raise RuntimeError("Tune hyperparameters before training the final MLP.")

        hidden_layer_sizes = tuple(
            self.best_params[f"n_units_l{i}"] for i in range(self.best_params["n_layers"])
        )
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=self.best_params["learning_rate_init"],
            alpha=self.best_params["alpha"],
            activation="relu",
            solver="adam",
            max_iter=1500,
            early_stopping=False,
            random_state=self.random_state,
        )
        self.model.fit(X_train, _as_1d_array(y_train))
        return self

    def predict_scores(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate_on_test(self, X_test, y_test):
        y_prob = self.predict_scores(X_test)
        y_pred = (y_prob > self.best_threshold).astype(int)
        print("Test set performance:")
        print("Precision:", precision_score(y_test, y_pred, zero_division=0))
        print("Recall:", recall_score(y_test, y_pred, zero_division=0))
        print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
        return y_pred, y_prob


# -----------------------------------------------------------------------------
# LSTM Autoencoder
# -----------------------------------------------------------------------------

class LSTMAEAnomalyDetector:
    """
    One-class LSTM autoencoder.

    Training uses only windows containing no labelled anomaly. Validation labels
    are used to calibrate/tune the anomaly threshold, so this is best described as
    one-class / semi-supervised anomaly detection rather than fully unsupervised.
    """

    def __init__(self, input_dim, device=None, random_state: int = 42):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.best_threshold = None
        self.best_epoch = None

    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_features, hidden_dim, num_layers):
            super().__init__()
            self.encoder = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, n_features)

        def forward(self, x):
            _, (h_n, _) = self.encoder(x)
            h = h_n[-1]
            dec_input = h.unsqueeze(1).repeat(1, x.size(1), 1)
            dec_out, _ = self.decoder(dec_input)
            return self.fc(dec_out)

    @staticmethod
    def create_sequences(X, y=None, seq_length=30):
        X = np.asarray(X, dtype=np.float32)
        if len(X) < seq_length:
            raise ValueError(f"Need at least seq_length={seq_length} observations, got {len(X)}.")
        y_arr = None if y is None else _as_1d_array(y).astype(int)
        if y_arr is not None and len(y_arr) != len(X):
            raise ValueError("X and y must have the same length.")

        sequences, endpoint_labels, normal_windows = [], [], []
        for i in range(len(X) - seq_length + 1):
            j = i + seq_length
            sequences.append(X[i:j])
            if y_arr is None:
                endpoint_labels.append(0)
                normal_windows.append(True)
            else:
                window_y = y_arr[i:j]
                endpoint_labels.append(int(window_y[-1]))
                normal_windows.append(bool(np.all(window_y == 0)))

        return (
            np.stack(sequences),
            np.asarray(endpoint_labels, dtype=int),
            np.asarray(normal_windows, dtype=bool),
        )

    @staticmethod
    def create_contextual_sequences(X_context, X_target, seq_length):
        """One causal sequence per target row, using only past context and the row itself."""
        X_context = np.asarray(X_context, dtype=np.float32)
        X_target = np.asarray(X_target, dtype=np.float32)
        need = seq_length - 1
        if len(X_context) < need:
            raise ValueError(
                f"Need at least {need} context rows for seq_length={seq_length}; got {len(X_context)}."
            )
        prefix = X_context[-need:] if need > 0 else X_context[:0]
        combined = np.vstack([prefix, X_target])
        windows = [combined[i:i + seq_length] for i in range(len(X_target))]
        return np.stack(windows)

    @torch.no_grad()
    def _endpoint_scores(self, model, sequences, batch_size=256):
        model.eval()
        loader = DataLoader(TensorDataset(torch.as_tensor(sequences, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
        scores = []
        for (seqs,) in loader:
            seqs = seqs.to(self.device)
            out = model(seqs)
            # Score ONLY the last timestamp. The sequence ends at t, so this is causal.
            err = ((out[:, -1, :] - seqs[:, -1, :]) ** 2).mean(dim=1)
            scores.extend(err.cpu().numpy())
        return np.asarray(scores, dtype=float)

    def objective(self, trial, X_train, y_train, X_val, y_val):
        _seed_torch(self.random_state)

        seq_len = trial.suggest_int("seq_len", 20, 50, step=5)
        hidden_dim = trial.suggest_int("hidden_dim", 32, 192, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        epochs, patience = 30, 6

        X_train_seq, _, normal_windows = self.create_sequences(X_train, y_train, seq_len)
        X_train_normal = X_train_seq[normal_windows]
        if len(X_train_normal) == 0:
            raise TrialPruned("No fully-normal training sequences for this seq_len.")

        X_val_seq = self.create_contextual_sequences(X_train, X_val, seq_len)
        y_val_arr = _as_1d_array(y_val).astype(int)

        generator = torch.Generator().manual_seed(self.random_state)
        train_loader = DataLoader(
            TensorDataset(torch.as_tensor(X_train_normal, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        model = self.LSTMAutoencoder(X_train_seq.shape[2], hidden_dim, num_layers).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_f1 = -1.0
        best_threshold = 0.0
        best_epoch = 0
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            for (seqs,) in train_loader:
                seqs = seqs.to(self.device)
                optimizer.zero_grad()
                out = model(seqs)
                loss = criterion(out, seqs)
                loss.backward()
                optimizer.step()

            val_scores = self._endpoint_scores(model, X_val_seq, batch_size=256)
            f1, threshold = _best_f1_threshold(y_val_arr, val_scores, default=float(np.quantile(val_scores, 0.95)))

            # Study direction is maximize, so pruning is now based on a quantity where higher is better.
            trial.report(f1, epoch)
            if trial.should_prune():
                raise TrialPruned()

            if f1 > best_f1 + 1e-12:
                best_f1 = f1
                best_threshold = threshold
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        trial.set_user_attr("best_threshold", float(best_threshold))
        trial.set_user_attr("best_epoch", int(best_epoch))
        return float(max(best_f1, 0.0))

    def tune_hyperparameters(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=20,
        storage="sqlite:///optuna_LSTM_AE.db",
    ):
        study = optuna.create_study(
            study_name="LSTM_AE_study_v2",
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if n_trials > 0:
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
                n_trials=n_trials,
                show_progress_bar=True,
            )

        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise RuntimeError("LSTM AE Optuna study has no COMPLETE trials.")

        self.best_params = study.best_params
        self.best_threshold = float(study.best_trial.user_attrs["best_threshold"])
        self.best_epoch = int(study.best_trial.user_attrs["best_epoch"])
        print("Best validation F1:", study.best_value)
        print("Best params:", self.best_params)
        print("Best threshold:", self.best_threshold)
        print("Best epoch:", self.best_epoch + 1)
        return study

    def train_final_model(self, X_train, y_train):
        if self.best_params is None or self.best_epoch is None:
            raise RuntimeError("Tune hyperparameters before final training.")

        _seed_torch(self.random_state)
        seq_len = self.best_params["seq_len"]
        X_seq, _, normal_windows = self.create_sequences(X_train, y_train, seq_len)
        X_normal = X_seq[normal_windows]
        if len(X_normal) == 0:
            raise ValueError("No fully-normal training sequences available.")

        hidden_dim = self.best_params["hidden_dim"]
        num_layers = self.best_params["num_layers"]
        lr = self.best_params["lr"]
        batch_size = self.best_params["batch_size"]

        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(torch.as_tensor(X_normal, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        self.model = self.LSTMAutoencoder(self.input_dim, hidden_dim, num_layers).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        epochs = self.best_epoch + 1
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for (seqs,) in loader:
                seqs = seqs.to(self.device)
                optimizer.zero_grad()
                out = self.model(seqs)
                loss = criterion(out, seqs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(seqs)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(loader.dataset):.6f}")
        return self

    def predict_scores(self, X_target, X_context):
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        seq_len = self.best_params["seq_len"]
        seqs = self.create_contextual_sequences(X_context, X_target, seq_len)
        return self._endpoint_scores(self.model, seqs)

    def evaluate_on_test(self, X_test, best_threshold=None, X_context=None):
        if X_context is None:
            raise ValueError("X_context is required so the first test observations have causal history.")
        threshold = self.best_threshold if best_threshold is None else float(best_threshold)
        test_err = self.predict_scores(X_test, X_context)
        y_pred = (test_err > threshold).astype(int)
        return y_pred, test_err


# -----------------------------------------------------------------------------
# LSTM classifier
# -----------------------------------------------------------------------------

class LSTMClassifierAnomalyDetector:
    def __init__(self, input_dim, device=None, random_state: int = 42):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.best_threshold = None
        self.best_epoch = None

    class LSTMClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            h_last = h_n[-1]
            return self.fc(h_last).squeeze(1)  # logits

    @staticmethod
    def create_sequences(X, y=None, seq_length=30):
        X = np.asarray(X, dtype=np.float32)
        if len(X) < seq_length:
            raise ValueError(f"Need at least seq_length={seq_length} observations, got {len(X)}.")
        y_arr = None if y is None else _as_1d_array(y).astype(int)
        sequences, labels = [], []
        for i in range(len(X) - seq_length + 1):
            j = i + seq_length
            sequences.append(X[i:j])
            labels.append(0 if y_arr is None else int(y_arr[j - 1]))
        return np.stack(sequences), np.asarray(labels, dtype=int)

    @staticmethod
    def create_contextual_sequences(X_context, X_target, seq_length):
        return LSTMAEAnomalyDetector.create_contextual_sequences(X_context, X_target, seq_length)

    @torch.no_grad()
    def _predict_proba_model(self, model, sequences, batch_size=256):
        model.eval()
        loader = DataLoader(TensorDataset(torch.as_tensor(sequences, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
        probs = []
        for (xb,) in loader:
            logits = model(xb.to(self.device))
            probs.extend(torch.sigmoid(logits).cpu().numpy())
        return np.asarray(probs, dtype=float)

    def objective(self, trial, X_train, y_train, X_val, y_val):
        _seed_torch(self.random_state)

        seq_len = trial.suggest_int("seq_len", 20, 50, step=5)
        hidden_dim = trial.suggest_int("hidden_dim", 32, 192, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        epochs, patience = 30, 6

        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, seq_len)
        X_val_seq = self.create_contextual_sequences(X_train, X_val, seq_len)
        y_val_arr = _as_1d_array(y_val).astype(int)

        generator = torch.Generator().manual_seed(self.random_state)
        train_loader = DataLoader(
            TensorDataset(
                torch.as_tensor(X_train_seq, dtype=torch.float32),
                torch.as_tensor(y_train_seq, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        model = self.LSTMClassifier(self.input_dim, hidden_dim, num_layers).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        n_pos = max(int((y_train_seq == 1).sum()), 1)
        n_neg = max(int((y_train_seq == 0).sum()), 1)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_f1 = -1.0
        best_threshold = 0.5
        best_epoch = 0
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            val_probs = self._predict_proba_model(model, X_val_seq)
            f1, threshold = _best_f1_threshold(y_val_arr, val_probs)
            trial.report(f1, epoch)
            if trial.should_prune():
                raise TrialPruned()

            if f1 > best_f1 + 1e-12:
                best_f1 = f1
                best_threshold = threshold
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        trial.set_user_attr("best_threshold", float(best_threshold))
        trial.set_user_attr("best_epoch", int(best_epoch))
        return float(max(best_f1, 0.0))

    def tune_hyperparameters(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=20,
        storage="sqlite:///optuna_LSTM_classifier.db",
    ):
        study = optuna.create_study(
            direction="maximize",
            study_name="LSTM_Classifier_Study_v2",
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if n_trials > 0:
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
                n_trials=n_trials,
                show_progress_bar=True,
            )

        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise RuntimeError("LSTM classifier Optuna study has no COMPLETE trials.")

        self.best_params = study.best_params
        self.best_threshold = float(study.best_trial.user_attrs["best_threshold"])
        self.best_epoch = int(study.best_trial.user_attrs["best_epoch"])
        print("Best validation F1:", study.best_value)
        print("Best parameters:", self.best_params)
        print("Best validation threshold:", self.best_threshold)
        print("Best epoch:", self.best_epoch + 1)
        return study

    def train_final_model(self, X_train, y_train):
        """Train on the training sample only so the validation-calibrated threshold stays valid."""
        if self.best_params is None or self.best_epoch is None:
            raise RuntimeError("Tune hyperparameters before final training.")

        _seed_torch(self.random_state)
        seq_len = self.best_params["seq_len"]
        X_seq, y_seq = self.create_sequences(X_train, y_train, seq_len)

        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(
                torch.as_tensor(X_seq, dtype=torch.float32),
                torch.as_tensor(y_seq, dtype=torch.float32),
            ),
            batch_size=self.best_params["batch_size"],
            shuffle=True,
            generator=generator,
        )

        self.model = self.LSTMClassifier(
            self.input_dim,
            self.best_params["hidden_dim"],
            self.best_params["num_layers"],
        ).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.best_params["lr"])

        n_pos = max(int((y_seq == 1).sum()), 1)
        n_neg = max(int((y_seq == 0).sum()), 1)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        epochs = self.best_epoch + 1
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(loader.dataset):.6f}")
        return self

    def predict(self, X_target, X_context=None):
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        seq_len = self.best_params["seq_len"]
        if X_context is None:
            X_seq, _ = self.create_sequences(X_target, None, seq_len)
        else:
            X_seq = self.create_contextual_sequences(X_context, X_target, seq_len)
        return self._predict_proba_model(self.model, X_seq)
