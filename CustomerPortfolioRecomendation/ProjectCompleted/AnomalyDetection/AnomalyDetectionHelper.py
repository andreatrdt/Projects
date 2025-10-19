import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
import umap.umap_ as umap
from IPython.display import display
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.exceptions import TrialPruned
from sklearn.neural_network import MLPClassifier



class AnomalyDetectionEDA:
    """
    A class for performing exploratory data analysis (EDA) on anomaly detection datasets.
    This class provides methods to generate metadata, plot anomalies, filter negative anomalies,
    visualize assets, and perform stationarity tests.
    """

    def __init__(self, data_df=None, metadata_df=None, X_df=None, y=None):
        """
        Initializes the AnomalyDetectionEDA class with data and metadata.
        :param data_df:     DataFrame containing the main dataset with anomalies.
        :param metadata_df: DataFrame containing metadata for the dataset.
        :param X_df:        DataFrame containing the features (independent variables).
        :param y:           Series containing the target variable (dependent variable), typically indicating anomalies.
        """
        self.data_df = data_df
        self.metadata_df = metadata_df
        self.X_df = X_df
        self.y = y      

    def generate_metadata(self):
        """
        Generates enhanced metadata for the features in the dataset.
        This method computes basic statistics for each feature and combines it with metadata.
        :return:           DataFrame containing enhanced metadata for each feature.
        """
        # Initialize an empty list to store metadata for each ticker
        enhanced_metadata = []
        # Define the columns for ticker and description based on the metadata DataFrame
        ticker_col = 'ticker' if 'ticker' in self.metadata_df.columns else self.metadata_df.columns[0]
        desc_col = 'description' if 'description' in self.metadata_df.columns else self.metadata_df.columns[1]

        # Iterate through each ticker in the X_df DataFrame
        for ticker in self.X_df.columns:
            # Get the corresponding metadata row for the ticker
            meta_row = self.metadata_df[self.metadata_df[ticker_col] == ticker]
            # If metadata row is empty, use the ticker as description
            description = meta_row[desc_col].values[0] if not meta_row.empty else ticker
            # Get the time series data for the ticker
            series = self.X_df[ticker]

            # Append the computed statistics and metadata to the list
            enhanced_metadata.append({
                'Ticker': ticker,
                'Description': description,
                'Mean': series.mean(),
                'Std.Dev': series.std(),
                'Min': series.min(),
                'Max': series.max(),
                'Missing values': series.isna().sum(),
                'Missing (%)': f"{series.isna().mean()*100:.2f}%"
            })

        # Create a DataFrame from the enhanced metadata list
        meta_df = pd.DataFrame(enhanced_metadata)
        # Display the metadata DataFrame
        print("\nEnhanced Metadata:")
        display(meta_df)

        return meta_df
    

    def plot_anomalies(self, index_col='MXUS'):
        """
        Plots the anomalies in the dataset based on the specified index column.
        :param index_col:   The column name in the X_df DataFrame to plot. Default is 'MXUS'.
        """
        # Check if the 'Y' column exists and if the index_col is in the X_df DataFrame
        if self.y is not None and index_col in self.X_df.columns:
            # Create a figure and axis for the plot
            fig, ax = plt.subplots(figsize=(14, 8))
            # Plot the index column data
            ax.plot(self.X_df.index, self.X_df[index_col], color='darkred', linewidth=2.5, label=index_col)
            y_min, y_max = ax.get_ylim()

            # Iterate through the anomalies and highlight them on the plot
            for i, (date, is_anomaly) in enumerate(zip(self.X_df.index, self.y)):
                if is_anomaly == 1:
                    ax.axvspan(date, date + pd.Timedelta(days=7), alpha=0.3, color='navy', label='Risk-on/Risk-off' if i == 0 else "")

            ax.set_xlabel('Timeline')
            ax.set_ylabel(index_col)
            ax.set_title(index_col + ' and risk-on/risk-off periods')
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.tight_layout()
            plt.show()
        else:
            print("Either 'Y' column or index_col is missing.")


    def filter_negative_anomalies(self, cols_to_use):
        """
        Filters the anomalies in the dataset to identify those with negative returns.
        :param cols_to_use: List of columns to use for calculating returns.
        This method calculates the log returns for the specified columns and identifies anomalies
        that have negative returns. It prints the number of anomalies with negative returns and
        the total number of anomalies. It also prints the dates of positive anomalies.
        """
        # Calculate log returns for the specified columns
        returns = np.log(self.X_df[cols_to_use]/self.X_df[cols_to_use].shift(1))
        # Find anomalies with negative returns
        negative_returns_mask = (returns < 0).any(axis=1)
        # Filter the data for anomalies with negative returns
        negative_anomalies = self.data_df.loc[(self.data_df['Y'] == 1) & negative_returns_mask]

        # Print the number of anomalies with negative returns and total anomalies
        print(f"Number of anomalies with negative returns: {len(negative_anomalies)}")
        print(f"Total number of anomalies: {np.sum(self.data_df['Y'] == 1)}")
        
        # Set the 'Y' column to 0 for anomalies with positive returns
        self.data_df.loc[(returns > 0).all(axis=1), 'Y'] = 0
        # self.y.loc[(returns > 0).all(axis=1) > 0] = 0

        # Verify that in y there is the same number of anomalies as in the data_df
        assert np.sum(self.data_df['Y'] == 1) == np.sum(self.y == 1), "Mismatch in number of anomalies in data_df and y"
        


    def visualize_assets(self, asset_columns, bonds):
        """
        Visualizes the time series data and returns of specified asset columns.
        :param asset_columns: List of asset columns to visualize.
        :param bonds:         List of bond columns to differentiate between bond and equity returns.
        This method creates a series of subplots showing the time series data, first differences (for bonds) or log returns (for equities),
        distribution of returns, and QQ plots for the specified asset columns.
        """

        # Initialize a dictionary to store returns for each asset column
        returns_dict = {}
        # Calculate returns for each asset column
        for col in asset_columns:
            if col in bonds:
                returns_dict[col] = np.diff(self.X_df[col])
            else:
                returns_dict[col] = np.diff(np.log(self.X_df[col]))

        # Create subplots for each asset column
        fig, axes = plt.subplots(4, len(asset_columns), figsize=(18, 16))
        fig.suptitle('Asset Analysis: Examples of Bond and Equity Returns', fontsize=16)
        plt.subplots_adjust(hspace=0.4)

        # Plot time series, returns, distribution, and QQ plots for each asset column
        for i, col in enumerate(asset_columns):
            axes[0, i].plot(self.X_df.index, self.X_df[col])
            axes[0, i].set_title(f'{col}: time plot')

            axes[1, i].plot(self.X_df.index[1:], returns_dict[col])
            axes[1, i].set_title(f'{col}: {"First Differences" if col in bonds else "Log Returns"}')

            axes[2, i].hist(returns_dict[col], bins=50, alpha=0.7)
            axes[2, i].set_title(f'{col}: returns distribution')

            stats.probplot(returns_dict[col], dist="norm", plot=axes[3, i])
            axes[3, i].set_title(f'{col}: QQplot vs Gaussian')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()


    def plot_interest_rate_distribution(self, column='GTDEM2Y'):
        """
        Plot the histogram of a specified interest rate series to check for negative values.
        :param column: The name of the column representing the interest rate (default is 'GTDEM2Y').
        """
        
        # Check if the column exists in the X_df DataFrame
        if column in self.X_df.columns:
            plt.figure(figsize=(12, 6))
            plt.hist(self.X_df[column].dropna(), bins=50, alpha=0.7)
            plt.title(f'{column} yield: Checking for negative yields!')
            plt.xlabel(f'{column} yield')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
        else:
            print(f"Column '{column}' not found in the dataset.")



    def adf_stationarity_test(self):
        """
        Performs the Augmented Dickey-Fuller (ADF) stationarity test on each column of the X_df DataFrame.
        This method computes the ADF statistic, p-value, and critical values for each column,
        and displays the results in a styled DataFrame.
        """

        def adf_test(series):
            """
            Performs the Augmented Dickey-Fuller test on a time series.
            """

            result = adfuller(series.dropna())
            return {
                'ADF Statistic': result[0],
                'p-value': result[1],
                'Used Lag': result[2],
                'Observations Used': result[3],
                'Critical Value (1%)': result[4]['1%'],
                'Critical Value (5%)': result[4]['5%'],
                'Critical Value (10%)': result[4]['10%']
            }

        # Perform ADF test for each column in the X_df DataFrame
        adf_results = {col: adf_test(self.X_df[col]) for col in self.X_df.columns}
        # Convert the results to a DataFrame and format it
        adf_df = pd.DataFrame(adf_results).T
        # Add a column to indicate if the series is stationary based on the p-value
        adf_df['Stationary (<0.05 p-value)'] = adf_df['p-value'] < 0.05
        # Sort the DataFrame by p-value
        adf_df = adf_df.sort_values(by='p-value')

        def binary_highlight(val):
            """
            Highlights the background color based on the p-value.
            """

            if val < 0.05:
                return 'background-color: #006837; color: white'
            else:
                return 'background-color: #a50026; color: white'
            
        # Style the DataFrame for better visualization
        styled_adf = (
            adf_df.style
            .map(binary_highlight, subset=['p-value'])
            .format({'ADF Statistic': '{:.3f}', 'p-value': '{:.4f}'})
            .set_caption("ADF Stationarity Test Results")
        )
        display(styled_adf)





class AnomalyDataPreparer:
    """
    A class for preparing anomaly detection datasets.
    This class provides methods to shuffle, scale, split, and make data stationary.
    It can also handle chronological splits and ensure the data is ready for training, validation, and testing.
    """

    def __init__(
        self,
        shuffle_data=True,
        scale_data=True,
        do_split=True,
        chronological_split=False,
        make_stationary=True,
        train_frac=0.8,
        val_frac=0.1,
        random_state=42,
    ):
        """
        Initializes the AnomalyDataPreparer class with parameters for data preparation.
        :param shuffle_data:          Whether to shuffle the data before splitting.
        :param scale_data:            Whether to scale the data using StandardScaler.
        :param do_split:              Whether to split the data into training, validation, and test sets.
        :param chronological_split:   Whether to perform a chronological split of the data.
        :param make_stationary:       Whether to make the data stationary by differencing.
        :param train_frac:            Fraction of data to use for training (default is 0.8).
        :param val_frac:              Fraction of data to use for validation (default is 0.1).
        :param random_state:          Random state for reproducibility.
        """
        # Initialize parameters
        self.shuffle_data = shuffle_data
        self.scale_data = scale_data
        self.do_split = do_split
        self.chronological_split = chronological_split
        self.make_stationary = make_stationary
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.random_state = random_state
        self.scaler = StandardScaler()

    def make_data_stationary(self, X_df, y=None):
        """
        Makes the data stationary by differencing and returns the stationary DataFrame.
        :param X_df: DataFrame containing the features (independent variables).
        :param y:    Series containing the target variable (dependent variable), typically indicating anomalies.
        :return:     A DataFrame with stationary features and a Series with stationary target variable.
        """

        # Currencies and indices columns
        indices_currencies = [col for col in X_df.columns if col in [
            'XAUBGNL', 'BDIY', 'CRY', 'Cl1', 'DXY', 'EMUSTRUU', 'GBP', 'JPY', 'LF94TRUU',
            'LF98TRUU', 'LG30TRUU', 'LMBITR', 'LP01TREU', 'LUACTRUU', 'LUMSTRUU',
            'MXBR', 'MXCN', 'MXEU', 'MXIN', 'MXJP', 'MXRU', 'MXUS', 'VIX'
        ]]

        # Interest rates columns
        interest_rates = [col for col in X_df.columns if col in [
            'EONIA', 'GTDEM10Y', 'GTDEM2Y', 'GTDEM30Y', 'GTGBP20Y', 'GTGBP2Y', 'GTGBP30Y',
            'GTITL10YR', 'GTITL2YR', 'GTITL30YR', 'GTJPY10YR', 'GTJPY2YR', 'GTJPY30YR',
            'US0001M', 'USGG3M', 'USGG2YR', 'GT10', 'USGG30YR'
        ]]
        # Create a new DataFrame for stationary data
        stationary_df = pd.DataFrame(index=X_df.index[1:])

        # Apply differencing for currencies and indices
        for col in indices_currencies:
            if col in X_df.columns:
                stationary_df[col] = np.diff(np.log(X_df[col]))
        for col in interest_rates:
            if col in X_df.columns:
                stationary_df[col] = np.diff(X_df[col])
        if 'ECSURPUS' in X_df.columns:
            stationary_df['ECSURPUS'] = X_df['ECSURPUS'].values[1:]

        y_stationary = y[1:] if y is not None else None
        return stationary_df, y_stationary

    def prepare(self, X_df, y):
        # Ensure index is datetime
        X_df.index = pd.to_datetime(X_df.index)

        # === STATIONARIZE DATA IF REQUESTED ===
        if self.make_stationary:
            X_df, y = self.make_data_stationary(X_df, y)

        # === SHUFFLE DATA IF REQUESTED ===
        if self.shuffle_data:
            X_df, y = shuffle(X_df, y, random_state=self.random_state)

        # === SPLIT INTO TRAINING, VALIDATION AND TEST SET ===
        # If not splitting, return the full dataset
        if not self.do_split:
            if self.scale_data:
                X_scaled = self.scaler.fit_transform(X_df)
                X_df = pd.DataFrame(X_scaled, index=X_df.index, columns=X_df.columns)
            return X_df, pd.Series(y, index=X_df.index)

        # Check if chronological splitting is needed
        if self.chronological_split:
            # Sort by index (date) for chronological split
            X_df = X_df.sort_index()
            if y is not None:
                y = pd.Series(y, index=X_df.index).sort_index()
        
        # Number of data        
        n = len(X_df)
        
        train_size = int(self.train_frac * n)
        val_size = int(self.val_frac * n)

        X_train = X_df.iloc[:train_size]
        y_train = y.loc[X_train.index]
        
        X_val = X_df.iloc[train_size:train_size + val_size]
        y_val = y.loc[X_val.index]
        
        X_test = X_df.iloc[train_size + val_size:]
        y_test = y.loc[X_test.index]

        # === SCALING ===
        if self.scale_data:
            X_train_vals = self.scaler.fit_transform(X_train)
            X_val_vals = self.scaler.transform(X_val)
            X_test_vals = self.scaler.transform(X_test)

            X_train = pd.DataFrame(X_train_vals, index=X_train.index, columns=X_df.columns)
            X_val = pd.DataFrame(X_val_vals, index=X_val.index, columns=X_df.columns)
            X_test = pd.DataFrame(X_test_vals, index=X_test.index, columns=X_df.columns)

        # === SUMMARY ===
        print(f"Training set size: {X_train.shape[0]} ({sum(y_train==0)} normal, {sum(y_train==1)} anomalies)")
        print(f"Validation set size: {X_val.shape[0]} ({sum(y_val==0)} normal, {sum(y_val==1)} anomalies)")
        print(f"Test set size: {X_test.shape[0]} ({sum(y_test==0)} normal, {sum(y_test==1)} anomalies)")

        return X_train, y_train, X_val, y_val, X_test, y_test
    



class DetectionMethodsEvaluation:
    """
    A class for evaluating anomaly detection methods.
    This class provides methods to evaluate model performance using precision, recall, F1 score,
    and a Financial Business Oriented Score, computed using the prepared dataset.
    """

    def __init__(self, preparer, metadata_df, X_df, y):
        """
        Initializes the DetectionMethodsEvaluation class with data preparation and metadata.
        :param preparer:    An instance of AnomalyDataPreparer for data preparation.
        :param metadata_df: DataFrame containing metadata for the dataset.
        :param X_df:        DataFrame containing the features (independent variables).
        :param y:           Series containing the target variable (dependent variable), typically indicating anomalies.
        """

        self.preparer = preparer
        self.metadata_df = metadata_df
        self.X_df = X_df
        self.y = y
        self.mean_normal_weighted = None
        self.mean_anomaly_weighted = None
        self._compute_weighted_means()

    def _compute_weighted_means(self):
        from collections import defaultdict

        # Make the data stationary
        stationary_df, y_stationary = self.preparer.make_data_stationary(self.X_df, self.y)

        # Map tickers to asset classes
        class_map = self.metadata_df.set_index('Variable name')['Type'].to_dict()

        # Group tickers by asset class
        bucket_map = defaultdict(list)
        for ticker, cls in class_map.items():
            if ticker in stationary_df.columns:
                bucket_map[cls].append(ticker)

        # Compute average z-score per asset class
        ratio_df = pd.DataFrame(index=stationary_df.index)
        for cls, tickers in bucket_map.items():
            col_name = f"{cls.lower().replace(' ','_')}_ratio"
            ratio_df[col_name] = stationary_df[tickers].mean(axis=1)

        # Ensure y_stationary is a Series aligned with ratio_df
        y_stationary = pd.Series(
            y_stationary.flatten() if hasattr(y_stationary, 'flatten') else y_stationary,
            index=ratio_df.index,
            name='Y'
        )

        # Compute correlation weights
        corrs = ratio_df.corrwith(y_stationary)
        weights_df = corrs.abs().div(corrs.abs().sum()).to_frame(name='weight')

        # Create compact_df with Y labels
        compact_df = ratio_df.copy()
        compact_df['Y'] = y_stationary
        grouped = compact_df.groupby('Y').mean()

        # Mean z-scores per class
        mean_normal = grouped.loc[0]
        mean_anomaly = grouped.loc[1]

        # Align and compute weighted means
        common = weights_df.index.intersection(mean_normal.index)
        w = weights_df.loc[common, 'weight']
        self.mean_normal_weighted = (mean_normal.loc[common] * w).sum()
        self.mean_anomaly_weighted = (mean_anomaly.loc[common] * w).sum()


    def financial_score(self, y_true, y_pred):
        """
        Compute the financial score based on the true and predicted labels.
        """

        # Ensure y_true and y_pred are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Compute costs based on the weighted means
        cost_anomaly = 100 * abs(self.mean_anomaly_weighted)
        cost_normal = 10 * abs(self.mean_normal_weighted)
        const_trans = y_pred.sum() * 0.2

        # Calculate mismatches
        anomaly_mismatch = (y_true == 1) & (y_pred == 0)
        normal_mismatch = (y_true == 0) & (y_pred == 1)

        # Calculate total cost
        total_cost = (
            anomaly_mismatch.sum() * cost_anomaly +
            normal_mismatch.sum() * cost_normal +
            const_trans
        )

        # Calculate the worst-case scenario cost
        total_worst = (
            (y_true == 1).sum() * cost_anomaly +
            (y_true == 0).sum() * cost_normal +
            0.1 * len(y_true)
        )

        return 1.0 - (total_cost / total_worst) if total_worst != 0 else 1.0


    def evaluate_model(self, y_true, y_pred, y_score, model_name):
        """
        Evaluate model performance with precision, recall, F1 score, Financial Score,
        and generate Confusion Matrix, ROC, Precision–Recall, and Score Histogram plots.
        """

        # Compute precision, recall, F1 score, and financial score
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fin_score = self.financial_score(y_true, y_pred)

        # Print performance metrics
        print(f"\n{model_name} Performance:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Financial Score: {fin_score:.4f}")

        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=['Normal', 'Anomaly'],
                     yticklabels=['Normal', 'Anomaly'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix — {model_name}')
        plt.tight_layout()
        plt.show()

        if y_score is not None:
            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve — {model_name}')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.show()

            # Plot Precision–Recall curve
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)

            plt.figure(figsize=(8, 6))
            plt.plot(recall_vals, precision_vals, lw=2, label=f'AP = {ap:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision–Recall Curve — {model_name}')
            plt.legend(loc='upper right')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        return precision, recall, f1, fin_score




class MLPAnomalyDetector:
    """
    MLP Classifier for anomaly detection.
    This class provides methods for hyperparameter tuning with Optuna,
    final model training, and evaluation on the test set.
    """

    def __init__(self, input_dim):
        """
        Initialize the MLP anomaly detector.
        :param input_dim: Number of features in the input data.
        """

        self.input_dim = input_dim
        self.best_params = None
        self.model = None

    def objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Objective function for Optuna tuning, ensuring decreasing layer sizes.
        """
        # Hyperparameters
        n_layers = trial.suggest_int("n_layers", 5, 7)

        # First layer size
        first_layer_size = trial.suggest_int("n_units_l0", 512, 1024, log=True)
        hidden_layer_sizes = [first_layer_size]
        prev_size = first_layer_size
        for i in range(1, n_layers):
            next_size = trial.suggest_int(
                f"n_units_l{i}",
                int(prev_size / 2),
                int(prev_size)
            )
            hidden_layer_sizes.append(next_size)
            prev_size = next_size

        # Other hyperparameters
        lr = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        activation = trial.suggest_categorical("activation", ["relu"])
        threshold = trial.suggest_float("threshold", 0.5, 0.9)

        # Model
        model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            learning_rate_init=lr,
            alpha=alpha,
            activation=activation,
            solver="adam",
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        )

        # Train and evaluate
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs > threshold).astype(int)

        return f1_score(y_val, preds)

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=100, storage="sqlite:///optuna_MLP.db"):
        """
        Tune hyperparameters using Optuna.
        """
        def wrapped_objective(trial):
            return self.objective(trial, X_train, y_train, X_val, y_val)

        study = optuna.create_study(
            study_name="MLP_Study",
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
        )
        # Set Optuna logging level to WARNING to reduce verbosity
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        print("Best F1:", study.best_value)
        print("Best hyperparameters:", self.best_params)

        return study

    def train_final_model(self, X_train, y_train, X_val, y_val):
        """
        Train the final MLP model on the combined train+val set using the best hyperparameters.
        """
        hidden_layer_sizes = tuple(self.best_params[f"n_units_l{i}"] for i in range(self.best_params["n_layers"]))

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=self.best_params["learning_rate_init"],
            alpha=self.best_params["alpha"],
            activation=self.best_params["activation"],
            solver="adam",
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        )

        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.hstack([y_train, y_val])

        self.model.fit(X_train, y_train)

    def evaluate_on_test(self, X_test, y_test):
        """
        Evaluate the trained MLP model on the test set.
        :return: y_pred, y_prob
        """
        # Predict probabilities and apply the threshold
        y_prob = self.model.predict_proba(X_test)[:, 1]
        threshold = self.best_params["threshold"]
        y_pred = (y_prob > threshold).astype(int)

        # Calculate performance metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Print performance metrics
        print("Test set performance:")
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        return y_pred, y_prob





class LSTMAEAnomalyDetector:
    """
    LSTM Autoencoder for anomaly detection in time series data.
    This class provides methods for hyperparameter tuning using Optuna,
    training the final model, and reconstructing time series data.
    It uses an LSTM-based autoencoder architecture to learn the normal patterns in the data
    and detect anomalies based on reconstruction errors.
    """

    def __init__(self, input_dim, device=None):
        """
        Initialize the LSTM Autoencoder anomaly detector.
        :param input_dim: Number of features in the input data.
        :param device: Device to run the model on (CPU or GPU).

        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.best_params = None

    class LSTMAutoencoder(nn.Module):
        """
        LSTM Autoencoder for time series reconstruction.
        """
        def __init__(self, n_features, hidden_dim, num_layers):
            """
            Initialize the LSTM Autoencoder.
            :param n_features: Number of features in the input data.
            :param hidden_dim: Number of hidden units in the LSTM layers.
            :param num_layers: Number of LSTM layers in the encoder and decoder.
            """

            super().__init__()
            self.encoder = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, n_features)

        def forward(self, x):
            """
            Forward pass through the LSTM Autoencoder.
            """

            _, (h_n, _) = self.encoder(x)
            h = h_n[-1]
            dec_input = h.unsqueeze(1).repeat(1, x.size(1), 1)
            dec_out, _ = self.decoder(dec_input)
            return self.fc(dec_out)

        @torch.no_grad()
        def reconstruct_full(self, X_full, seq_len, device):
            """
            Reconstruct the full time series data using the trained model.
            :param X_full:  Full time series data to reconstruct.
            :param seq_len: Length of the sequences to use for reconstruction.
            :param device:  Device to run the model on (CPU or GPU).
            :return:        Reconstructed full time series data and error per time step.
            """

            self.eval()
            T, F = X_full.shape
            windows, idxs = [], []
            for start in range(T - seq_len + 1):
                windows.append(X_full[start:start + seq_len])
                idxs.append(np.arange(start, start + seq_len))
            windows = np.stack(windows)
            recon_acc = np.zeros((T, F), dtype=float)
            hit_count = np.zeros(T, dtype=float)
            for w, inds in zip(windows, idxs):
                w_t = torch.from_numpy(w).unsqueeze(0).float().to(device)
                w_rec = self(w_t).cpu().numpy().squeeze(0)
                recon_acc[inds] += w_rec
                hit_count[inds] += 1
            recon_full = recon_acc / hit_count[:, None]
            err_per_t = ((X_full - recon_full) ** 2).mean(axis=1)
            return recon_full, err_per_t

    @staticmethod
    def create_sequences(X, y=None, seq_length=30):
        """
        Create sequences of data for LSTM input.
        """

        sequences, seq_labels = [], []
        for i in range(len(X) - seq_length + 1):
            seq = X[i: i + seq_length]
            label = y[i + seq_length - 1] if y is not None else 0
            sequences.append(seq)
            seq_labels.append(label)
        return np.stack(sequences), np.array(seq_labels)

    

    def objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Objective function for Optuna hyperparameter tuning.
        Includes threshold search for maximizing F1-score on validation set.
        """
        # Hyperparameters
        seq_len = trial.suggest_int('seq_len', 20, 50, step=5)
        hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=16)
        num_layers = trial.suggest_int('num_layers', 1, 5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        epochs, patience = 20, 5

        # Create Sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, seq_len)
        
        # Keep only normal sequences for training
        X_train_seq = X_train_seq[y_train_seq == 0]  # Only normal sequences

        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq)), batch_size=batch_size, shuffle=True)
        cv_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_seq), torch.LongTensor(y_val_seq)), batch_size=batch_size, shuffle=False)

        # Initialize model, optimizer, and loss function
        model = self.LSTMAutoencoder(X_train_seq.shape[2], hidden_dim, num_layers).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        no_improve_epochs = 0

        # Training loop with early stopping
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                seqs = batch[0].to(self.device)
                optimizer.zero_grad()
                out = model(seqs)
                loss = criterion(out, seqs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * seqs.size(0)

            avg_loss = running_loss / len(train_loader.dataset)
            trial.report(avg_loss, epoch)
            if trial.should_prune():
                raise TrialPruned()

            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    break

        # Validation
        model.eval()
        errors, labels = [], []
        with torch.no_grad():
            for seqs, labs in cv_loader:
                seqs = seqs.to(self.device)
                out = model(seqs)
                batch_err = ((out - seqs) ** 2).mean(dim=(1, 2)).cpu().numpy()
                errors.extend(batch_err)
                labels.extend(labs.numpy())

        errors, labels = np.array(errors), np.array(labels)
        if np.sum(labels == 0) == 0:
            return 0.0

        # Threshold tuning: search over 80th-99th percentiles
        best_f1, best_thresh = 0.0, None
        for p in np.arange(80, 100, 1):
            thresh = np.percentile(errors[labels == 0], p)
            preds = (errors > thresh).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

        # Log best threshold
        trial.set_user_attr("best_threshold", best_thresh)
        trial.report(best_f1, epoch)
        if trial.should_prune():
            raise TrialPruned()
        return best_f1


    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=20, storage='sqlite:///optuna_LSTM_BOND_AE_USA_F1.db'):
        """
        Tune hyperparameters using Optuna.
        :param X_train:      Training features.
        :param y_train:      Training labels.
        :param X_val:        Validation features.
        :param y_val:        Validation labels.
        :param n_trials:     Number of trials for hyperparameter tuning.
        :return:             Optuna study object with best parameters.
        """

        def wrapped(trial):
            return self.objective(trial, X_train, y_train, X_val, y_val)

        study = optuna.create_study(
            study_name='LSTM_AE_study',
            storage=storage,
            load_if_exists=True,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
        # Set Optuna logging level to WARNING to reduce verbosity
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(wrapped, n_trials=n_trials, show_progress_bar=True)
        self.best_params = study.best_params
        self.best_threshold = study.best_trial.user_attrs["best_threshold"]
        
        print("Best params:", self.best_params)
        return study

    def train_final_model(self, X_train_seq, input_dim):
        """
        Train the final LSTM Autoencoder model using the best hyperparameters.
        :param X_train_seq:  Training sequences created from the training data.
        :param input_dim:    Number of features in the input data.
        """

        # Extract best hyperparameters
        seq_len = self.best_params['seq_len']
        hidden_dim = self.best_params['hidden_dim']
        num_layers = self.best_params['num_layers']
        lr = self.best_params['lr']
        batch_size = self.best_params['batch_size']
        epochs = 30

        # Load training sequences
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq)), batch_size=batch_size, shuffle=True)
        self.model = self.LSTMAutoencoder(input_dim, hidden_dim, num_layers).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in train_loader:
                seqs = batch[0].to(self.device)
                optimizer.zero_grad()
                out = self.model(seqs)
                loss = criterion(out, seqs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * seqs.size(0)
            avg_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")


    def evaluate_on_test(self, X_test, best_threshold):
        """
        Evaluate the trained model on the test set and return predictions and reconstruction errors.
        :param X_test:          Test features.
        :param best_threshold:  Best threshold for anomaly detection.
        :return:                Predictions and reconstruction errors on the test set.
        """

        seq_len = self.best_params['seq_len']
        _, test_err = self.model.reconstruct_full(X_test, seq_len, self.device)

        y_pred = (test_err > best_threshold).astype(int)
        return y_pred, test_err




class LSTMClassifierAnomalyDetector:
    """
    LSTM Classifier for anomaly detection in time series data.
    This class provides methods for hyperparameter tuning using Optuna,
    training the final model, and evaluating performance.
    It uses an LSTM-based architecture to classify sequences as normal or anomalous.
    """

    def __init__(self, input_dim, device=None):
        """
        Initialize the LSTM Classifier anomaly detector.
        :param input_dim: Number of features in the input data.
        :param device:    Device to run the model on (CPU or GPU).
        """
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.best_params = None

    class LSTMClassifier(nn.Module):
        """
        LSTM Classifier for time series anomaly detection.
        This class defines the architecture of the LSTM-based classifier.
        """
        def __init__(self, input_dim, hidden_dim, num_layers):
            """
            Initialize the LSTM Classifier.
            :param input_dim:  Number of features in the input data.
            :param hidden_dim: Number of hidden units in the LSTM layers.
            :param num_layers: Number of LSTM layers in the model.
            """

            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            """
            Forward pass through the LSTM Classifier.
            :param x: Input tensor of shape (batch_size, seq_length, input_dim).
            :return: Output tensor of shape (batch_size, 1) with probabilities.
            """

            out, (h_n, _) = self.lstm(x)
            h_last = h_n[-1]  # Last layer's hidden state
            return self.sigmoid(self.fc(h_last)).squeeze(1)

    @staticmethod
    def create_sequences(X, y=None, seq_length=30):
        """
        Create sequences of data for LSTM input.
        :param X:          Input features.
        :param y:          Optional target labels.
        :param seq_length: Length of the sequences to create.
        :return:           Tuple of sequences and corresponding labels.
        """

        sequences, seq_labels = [], []
        for i in range(len(X) - seq_length + 1):
            seq = X[i: i + seq_length]
            label = y[i + seq_length - 1] if y is not None else 0
            sequences.append(seq)
            seq_labels.append(label)
        return np.stack(sequences), np.array(seq_labels)

    def objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Objective function for Optuna hyperparameter tuning.
        :param trial:      Optuna trial object for hyperparameter tuning.
        :param X_train:    Training features.
        :param y_train:    Training labels.
        :param X_val:      Validation features.
        :param y_val:      Validation labels.
        :return:           Best F1 score achieved during validation.
        """

        seq_len = trial.suggest_int('seq_len', 20, 50, step=5)
        hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=16)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64])
        epochs, patience = 20, 5

        # Prepare sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, seq_length=seq_len)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, seq_length=seq_len)
        

        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq)),
                                batch_size=batch_size, shuffle=False)

        model = self.LSTMClassifier(X_train_seq.shape[2], hidden_dim, num_layers).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        best_f1 = 0.0
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                yb = yb.view(-1)  # Flatten target
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    preds = model(xb).cpu().numpy()
                    all_preds.extend((preds > 0.5).astype(int))
                    all_labels.extend(yb.numpy())

            f1 = f1_score(all_labels, all_preds, zero_division=0)
            trial.report(f1, epoch)
            if trial.should_prune():
                raise TrialPruned()

            if f1 > best_f1:
                best_f1 = f1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return best_f1

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=20, storage='sqlite:///optuna_LSTM_BOND_classifier_USA.db'):
        """
        Tune hyperparameters using Optuna.
        :param X_train:      Training features.
        :param y_train:      Training labels.
        :param X_val:        Validation features.
        :param y_val:        Validation labels.
        :param n_trials:     Number of Optuna trials.
        :param storage:      Database URL for storing the study.
        """

        def wrapped(trial):
            return self.objective(trial, X_train, y_train, X_val, y_val)

        study = optuna.create_study(
            direction='maximize',
            study_name='LSTM_Classifier_Study',
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner()
        )
        # Set Optuna logging level to WARNING to reduce verbosity
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(wrapped, n_trials=n_trials, show_progress_bar=True)
        self.best_params = study.best_params
        print("Best parameters:", self.best_params)
        return study

    def train_final_model(self, X, y):
        """
        Train the final LSTM Classifier model using the best hyperparameters.
        :param X: Input features for training.
        :param y: Input labels for training.
        """ 

        seq_len = self.best_params['seq_len']
        hidden_dim = self.best_params['hidden_dim']
        num_layers = self.best_params['num_layers']
        lr = self.best_params['lr']
        batch_size = self.best_params['batch_size']
        epochs = 30

        X_seq, y_seq = self.create_sequences(X, y, seq_len)
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)),
                            batch_size=batch_size, shuffle=True)

        self.model = self.LSTMClassifier(X.shape[1], hidden_dim, num_layers).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
            print(f"Epoch {epoch + 1} - Loss: {total_loss / len(loader.dataset):.6f}")

    def predict(self, X):
        seq_len = self.best_params['seq_len']
        X_seq, _ = self.create_sequences(X, None, seq_len)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        return preds

