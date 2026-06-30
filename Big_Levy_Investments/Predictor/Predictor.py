import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import seaborn as sns
import joblib
import json
from scipy import stats
import os
import math
from tabulate import tabulate
from utils_bc2 import (
    test_normality, plot_histogram_and_qq, plot_histogram, plot_boxplot,
    plot_qq, plot_violin, plot_joint_distribution, plot_multiple_violin_plots,
    plot_permutation_importance, plot_partial_dependence
)
from config_bc2 import cost_map_single
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import optuna

def best_tau_f1(y_true, probs):
    """
    Find the threshold that maximizes the F1 score.
    """

    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    return thr[np.nanargmax(f1)]


def best_tau_penalty(y_true, probs, cost):
    """
    Find the threshold that minimizes the expected cost.
    """

    taus = np.linspace(0, 1, 1001)
    exp  = []
    for τ in taus:
        y_pred = (probs >= τ).astype(int)
        e = np.mean([cost[(int(a), int(p))] for a, p in zip(y_true, y_pred)])
        exp.append(e)
    return taus[np.argmin(exp)]


def create_variable_summary(df, metadata_df):
    """
    Create a summary DataFrame with statistics for each variable in the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        metadata_df (pd.DataFrame): DataFrame containing metadata for variable descriptions.
    Returns:
        pd.DataFrame: A summary DataFrame with statistics for each variable.
    """

    stats_dict = {
        'Variable': [],
        'Description': [],
        'Mean': [],
        'Std': [],
        'Missing': [],
        'Min': [],
        'Max': []
    }
    meta_dict = dict(zip(metadata_df['Metadata'], metadata_df['Unnamed: 1']))

    for col in df.columns:
        stats_dict['Variable'].append(col)
        stats_dict['Description'].append(meta_dict.get(col, 'N/A'))

        if pd.api.types.is_numeric_dtype(df[col]):
            stats_dict['Mean'].append(f"{df[col].mean():.2f}")
            stats_dict['Std'].append(f"{df[col].std():.2f}")
            stats_dict['Min'].append(f"{df[col].min():.2f}")
            stats_dict['Max'].append(f"{df[col].max():.2f}")
        else:
            stats_dict['Mean'].append('N/A')
            stats_dict['Std'].append('N/A')
            stats_dict['Min'].append('N/A')
            stats_dict['Max'].append('N/A')

        stats_dict['Missing'].append(df[col].isna().sum())

    return pd.DataFrame(stats_dict)


class Predictor:
    """
    A class for predicting investment needs based on client data.
    This class handles data preprocessing, exploratory data analysis (EDA),
    model training, evaluation, and prediction for both single and joint outputs.
    Attributes:
        model: The machine learning model used for predictions.
        needs_df: DataFrame containing client needs data.
        products_df: DataFrame containing product data.
        metadata_df: DataFrame containing metadata for variable descriptions.
        transformed_df: DataFrame after applying transformations and normalizations.
        scaler: MinMaxScaler instance for feature scaling.
        cost_map_single: Cost map for single-output predictions.
    """

    def __init__(self, model, needs_df=None, products_df=None, metadata_df=None):
        """
        Initializes the Predictor with a model and optional datasets.
        Args:
            model: The machine learning model to be used for predictions.
            needs_df (pd.DataFrame): DataFrame containing client needs data.
            products_df (pd.DataFrame): DataFrame containing product data.
            metadata_df (pd.DataFrame): DataFrame containing metadata for variable descriptions.
        """

        self.model = model
        # Load the dataset
        needs_df = needs_df
        self.products_df = products_df
        self.metadata_df = metadata_df

        # Drop ID and create summaries
        needs_df = needs_df.drop('ID', axis=1)
        print("NEEDS VARIABLES SUMMARY:")
        needs_summary = create_variable_summary(needs_df, self.metadata_df)
        display(needs_summary.style.set_properties(**{'text-align': 'left'}).hide(axis='index'))

        print("\nPRODUCTS VARIABLES SUMMARY:")
        products_summary = create_variable_summary(self.products_df, self.metadata_df)
        display(products_summary.style.set_properties(**{'text-align': 'left'}).hide(axis='index'))

        # Data transformations
        self.transformed_df = needs_df.copy()
        log_wealth = self.apply_log_transform(self.transformed_df, 'Wealth')
        self.transformed_df['Log_Wealth'] = log_wealth
        exponent = 0.3
        power_income = self.apply_power_transform(self.transformed_df, 'Income', exponent)
        self.transformed_df[f'Power_Income_{exponent}'] = power_income

        shapiro_stat_income, shapiro_p_income, ks_stat_income, ks_p_income = \
            test_normality(self.transformed_df, f'Power_Income_{exponent}')
        print(f"\nShapiro-Wilk test for Power_Income_{exponent}: Statistic={shapiro_stat_income}, p-value={shapiro_p_income}")

        self.scaler = MinMaxScaler()
        self.transformed_df = self.transformed_df.drop(columns=['Wealth', 'Income'])
        self.vars_to_normalize = ['Age','FamilyMembers','RiskPropensity','Log_Wealth',f'Power_Income_{exponent}']
        self.transformed_df[self.vars_to_normalize] = self.scaler.fit_transform(self.transformed_df[self.vars_to_normalize])
        self.cost_map_single = cost_map_single

    # EDA method (with subplots for categorical and numeric comparisons)
    def eda(self):
        """
        Performs exploratory data analysis (EDA) on the transformed DataFrame.
        This includes:
        - Histograms for categorical targets
        - Correlation matrix of transformed and normalized variables
        - Histograms and QQ plots for transformed numeric features
        - Boxplots and Violin plots for numeric features by categorical target
        - Joint distribution of Power_Income and Log_Wealth
        - Multiple violin plots for numeric columns by both categories
        - QQ plots for all numeric columns
        """

        exponent = 0.3
        numeric_cols = ['Age', 'FamilyMembers', 'RiskPropensity', 'Log_Wealth', f'Power_Income_{exponent}']

        # 1. Histograms for categorical targets side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(self.transformed_df['IncomeInvestment'], bins=2, edgecolor='black')
        axes[0].set_title('Need of Income Investment - Class balance analysis')
        axes[0].set_xlabel('Income Investment Need (categorical Y=1, N=0)')
        axes[0].set_ylabel('Count')

        axes[1].hist(self.transformed_df['AccumulationInvestment'], bins=2, edgecolor='black')
        axes[1].set_title('Need of Accumulation Investment - Class balance analysis')
        axes[1].set_xlabel('Accumulation Investment Need (categorical Y=1, N=0)')
        axes[1].set_ylabel('Count')

        plt.tight_layout()
        plt.show()

        # 2. Correlation matrix of transformed and normalized variables
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.transformed_df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation matrix of transformed and normalized variables')
        plt.show()

        # 3. Histograms and QQ plots for transformed numeric features
        plot_histogram_and_qq(self.transformed_df, 'Log_Wealth', 'Log Wealth')
        plot_histogram_and_qq(self.transformed_df, f'Power_Income_{exponent}', f'Power Income (exp={exponent})')

        # 4. Boxplots and Violin plots for each numeric feature by categorical target
        for col in ['Power_Income_' + str(exponent), 'Log_Wealth']:
            # Boxplots across both targets in subplots
            fig, axes = plt.subplots(2, 1, figsize=(8, 10))
            sns.boxplot(x='IncomeInvestment', y=col, data=self.transformed_df, ax=axes[0])
            axes[0].set_title(f'Boxplot of {col} by Income Investment')
            sns.boxplot(x='AccumulationInvestment', y=col, data=self.transformed_df, ax=axes[1])
            axes[1].set_title(f'Boxplot of {col} by Accumulation Investment')
            plt.tight_layout()
            plt.show()

            # Violin plots across both targets in subplots
            fig, axes = plt.subplots(2, 1, figsize=(8, 10))
            sns.violinplot(x='IncomeInvestment', y=col, data=self.transformed_df, ax=axes[0])
            axes[0].set_title(f'Violin plot of {col} by Income Investment')
            sns.violinplot(x='AccumulationInvestment', y=col, data=self.transformed_df, ax=axes[1])
            axes[1].set_title(f'Violin plot of {col} by Accumulation Investment')
            plt.tight_layout()
            plt.show()

        # 5. Joint distribution of Power_Income and Log_Wealth
        plot_joint_distribution(self.transformed_df, 'Power_Income_0.3', 'Log_Wealth',
                                title='Joint Distribution of Power Income and Log Wealth')

        # 6. Multiple violin plots for numeric columns by both categories (unchanged)
        plot_multiple_violin_plots(self.transformed_df, numeric_cols, ['IncomeInvestment', 'AccumulationInvestment'], ncols=2)

        # 7. QQ plots for all numeric columns
        for col in numeric_cols:
            plot_qq(self.transformed_df, col, title=f'Q-Q Plot for {col}')
        plt.show()

    # Custom penalty (unchanged)
    def custom_penalty_single(self, y_true, y_pred, cost_map_single):
        """
        Calculate the custom penalty based on the cost map for single-output predictions.
        Args:
            y_true (list): True labels.
            y_pred (list): Predicted labels.
            cost_map_single (dict): Cost map for single-output predictions.
        Returns:
            float: Average penalty per instance.
        """

        total_penalty = 0.0
        n = len(y_true)
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        for i in range(n):
            actual = y_true_arr[i]
            predicted = y_pred_arr[i]
            total_penalty += cost_map_single[(actual, predicted)]
        return (total_penalty / n)/10

    # Train-test split helper
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.
        Args:
            X (pd.DataFrame): Features DataFrame.
            y (pd.Series): Target variable Series.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test

    # Display results table
    def display_results_table(self, results_dict, model_name, feature_type):
        """
        Displays a formatted table of cross-validation and test results.
        Args:
            results_dict (dict): Dictionary containing cross-validation and test metrics.
            model_name (str): Name of the model used.
            feature_type (str): Type of features used in the model.
        """

        cv_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'Custom Penalty'],
            'CV Mean': [
                results_dict['cv_metrics']['accuracy']['mean'],
                results_dict['cv_metrics']['precision']['mean'],
                results_dict['cv_metrics']['recall']['mean'],
                results_dict['cv_metrics']['f1']['mean'],
                results_dict['cv_metrics']['custom_penalty']['mean']
            ],
            'CV Std': [
                results_dict['cv_metrics']['accuracy']['std'],
                results_dict['cv_metrics']['precision']['std'],
                results_dict['cv_metrics']['recall']['std'],
                results_dict['cv_metrics']['f1']['std'],
                results_dict['cv_metrics']['custom_penalty']['std']
            ],
            'Test Set': [
                results_dict['test_metrics']['accuracy'],
                results_dict['test_metrics']['precision'],
                results_dict['test_metrics']['recall'],
                results_dict['test_metrics']['f1'],
                results_dict['test_metrics']['custom_penalty']
            ]
        }
        df = pd.DataFrame(cv_data).round(3)
        print(f"\n{model_name} - {feature_type}")
        print("=" * 60)
        print(tabulate(df, headers='keys', tablefmt='pretty'))

    # Power transform helper
    def apply_power_transform(self, data, feature, exponent):
        """
        Applies a power transformation to the specified feature in the DataFrame.
        Args:
            data (pd.DataFrame): DataFrame containing the feature to transform.
            feature (str): Name of the feature to apply the power transformation to.
            exponent (float): Exponent for the power transformation.
        Returns:
            pd.Series: Transformed feature as a Series.
        """

        return np.power(data[feature], exponent)

    # Log transform helper
    def apply_log_transform(self, data, feature):
        """
        Applies a log transformation to the specified feature in the DataFrame.
        Args:
            data (pd.DataFrame): DataFrame containing the feature to transform.
            feature (str): Name of the feature to apply the log transformation to.
        Returns:
            pd.Series: Transformed feature as a Series.
        """

        return np.log1p(data[feature])

    # Predict new client (single-output)
    def pred_new_client(self, new_client):
        """
        Predicts the investment need for a new client based on the trained model.
        """

        exp = 0.3
        new_client = new_client.copy()
        new_client["Log_Wealth"] = self.apply_log_transform(new_client, "Wealth")
        new_client[f"Power_Income_{exp}"] = self.apply_power_transform(new_client, "Income", exp)
        for col, lo, hi in zip(self.vars_to_normalize, self.scaler.data_min_, self.scaler.data_max_):
            new_client[col] = new_client[col].clip(lower=lo, upper=hi)
        new_client[self.vars_to_normalize] = self.scaler.transform(new_client[self.vars_to_normalize])
        new_client = new_client.drop(columns=["Wealth", "Income"])
        X_new = new_client[self.model.feature_names_in_]
        prob = self.model.predict_proba(X_new)[:, 1][0]
        label = int(prob >= getattr(self, "tau", 0.5))
        print(f"{self.name}  →  p={prob:.3f}  τ={getattr(self,'tau',0.5):.3f}  label={label}")
        return label, prob

    # Optuna objective
    def _objective(self, trial, target):
        """
        Objective function for Optuna optimization.
        Args:
            trial (optuna.Trial): The Optuna trial object.
            target (str): The target variable to optimize for ('AccumulationInvestment' or 'IncomeInvestment').
        Returns:
            float: The F1 score for the model with the given hyperparameters.
        """

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
            "max_depth": trial.suggest_int("max_depth", 5, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.5, 0.8]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
            "random_state": 42, "n_jobs": -1
        }
        model = RandomForestClassifier(**params)
        X = self.transformed_df.drop(columns=['IncomeInvestment', 'AccumulationInvestment'])
        y = self.transformed_df[target]
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        score = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(f1_score, average="binary"), n_jobs=-1).mean()
        return score

    # Optuna study
    def study_optuna(self, target="AccumulationInvestment", n_trials=60, timeout=None, show_plot=True):
        """
        Runs an Optuna study to optimize hyperparameters for the RandomForestClassifier.
        Args:
            target (str): The target variable to optimize for ('AccumulationInvestment' or 'IncomeInvestment').
            n_trials (int): Number of trials for the optimization.
            timeout (int): Maximum time in seconds for the optimization.
            show_plot (bool): Whether to show the optimization history plot.
        Returns:
            dict: The best hyperparameters found by Optuna.
        """

        # Current working directory
        current_dir = os.getcwd()
        # Define the directory for Optuna studies and create it if it doesn't exist
        optuna_studies_dir = os.path.join(current_dir, "OptunaStudies")
        os.makedirs(optuna_studies_dir, exist_ok=True)
        # Build the full path for the db file
        db_file_path = os.path.join(optuna_studies_dir, f"sqlite:///optuna_{self.name}.db")
        # Build the SQLite URI
        optuna_db_uri = f"sqlite:///{db_file_path}"

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{self.name}_{target}",
            storage=optuna_db_uri,
            load_if_exists=True
        )
        study.optimize(lambda tr: self._objective(tr, target), n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        print("Best F1 = %.4f" % study.best_value)
        print("Best params:", study.best_params)
        self.best_params = study.best_params
        if show_plot:
            try:
                optuna.visualization.plot_optimization_history(study)
                plt.show()
            except Exception:
                pass
        return study.best_params

    # Joint training & evaluation
    def train_evaluate_joint(self, k_folds: int = 5, tau_selector=None, test_size: float = 0.20, random_state: int = 42):
        """
        Trains and evaluates a joint model for predicting both 'AccumulationInvestment' and 'IncomeInvestment'.
        Args:
            k_folds (int): Number of folds for cross-validation.
            tau_selector (callable): Function to select the optimal threshold τ.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        Returns:
            cv_report (pd.DataFrame): Cross-validation report with mean and std for each metric.
            test_report (pd.DataFrame): Test set report with metrics for each target.
            tau_dict (dict): Dictionary containing optimal thresholds for each target.
        """

        data   = self.transformed_df.copy()
        X_full = data.drop(columns=['AccumulationInvestment', 'IncomeInvestment'])
        y_full = data[['AccumulationInvestment', 'IncomeInvestment']]
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size,
                                                            random_state=random_state, shuffle=True)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        base = self.model.__class__(**self.model.get_params())
        joint_model = MultiOutputClassifier(base)
        oof_probs   = {lbl: np.zeros(len(X_train)) for lbl in y_train.columns}
        fold_stats  = {lbl: {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'custom_penalty']} for lbl in y_train.columns}

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            joint_model.fit(X_tr, y_tr)
            prob_list = joint_model.predict_proba(X_val)
            for i, lbl in enumerate(y_train.columns):
                probs = prob_list[i][:, 1]
                preds = (probs >= 0.5).astype(int)
                true  = y_val.iloc[:, i]
                oof_probs[lbl][val_idx] = probs
                fold_stats[lbl]['accuracy'].append(accuracy_score(true, preds))
                fold_stats[lbl]['precision'].append(precision_score(true, preds, zero_division=0))
                fold_stats[lbl]['recall'].append(recall_score(true, preds, zero_division=0))
                fold_stats[lbl]['f1'].append(f1_score(true, preds, zero_division=0))
                fold_stats[lbl]['custom_penalty'].append(self.custom_penalty_single(true, preds, self.cost_map_single))

        if tau_selector is None:
            tau_selector = best_tau_f1
        tau_dict = {lbl: tau_selector(y_train[lbl], oof_probs[lbl]) for lbl in y_train.columns}
        for lbl, τ in tau_dict.items():
            print(f"Optimal τ for {lbl}: {τ:.3f}")

        joint_model.fit(X_train, y_train)
        self.joint_model = joint_model
        self.tau_dict    = tau_dict

        # Determine the directory where the script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Create the path to the 'ModelForTheWebsite' folder
        output_folder = os.path.join(current_dir, 'ModelForTheWebsite')
        # Ensure the folder exists
        os.makedirs(output_folder, exist_ok=True)
        # Save the joint model
        joint_model_path = os.path.join(output_folder, 'joint_model.joblib')
        joblib.dump(self.joint_model, joint_model_path)

        # Save the tau_dict JSON file
        tau_dict_path = os.path.join(output_folder, 'tau_dict.json')
        with open(tau_dict_path, 'w') as f:
            json.dump(self.tau_dict, f)

        # joblib.dump(self.joint_model, 'joint_model.joblib')
        # with open('tau_dict.json', 'w') as f:
        #     json.dump(self.tau_dict, f)

        prob_test  = joint_model.predict_proba(X_test)
        test_stats = {}
        for i, lbl in enumerate(y_train.columns):
            probs = prob_test[i][:, 1]
            preds = (probs >= tau_dict[lbl]).astype(int)
            true  = y_test.iloc[:, i]
            test_stats[lbl] = dict(
                accuracy       = accuracy_score(true, preds),
                precision      = precision_score(true, preds, zero_division=0),
                recall         = recall_score(true, preds, zero_division=0),
                f1             = f1_score(true, preds, zero_division=0),
                custom_penalty = self.custom_penalty_single(true, preds, self.cost_map_single)
            )

        cv_report = pd.DataFrame({lbl: {m: f"{np.mean(vals):.3f}±{np.std(vals):.3f}" for m, vals in metr.items()} for lbl, metr in fold_stats.items()}).T
        test_report = pd.DataFrame(test_stats).T.round(3)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.model = joint_model

        # Display results
        for label in y_train.columns:
            results_dict = {
                'cv_metrics': {metric: {'mean': np.mean(fold_stats[label][metric]), 'std': np.std(fold_stats[label][metric])} for metric in fold_stats[label]},
                'test_metrics': test_stats[label]
            }
            self.display_results_table(results_dict, model_name="Random Forest", feature_type=label)

        # 2. Plot confusion matrix for 'AccumulationInvestment'
        self.plot_confusion_matrix_for_target('AccumulationInvestment')

        # 3. Plot confusion matrix for 'IncomeInvestment'
        self.plot_confusion_matrix_for_target('IncomeInvestment')
        return cv_report, test_report, tau_dict
    
    def explainability(self):
        """
        Generates SHAP and LIME explanations for both targets in the joint model.
        This method will plot SHAP summary bar and dot plots, and LIME explanations for the first instance in the test set.
        It will iterate over both targets in the joint model and generate explanations for each.
        """

        for target in self.y_train.columns:
            print(f"\nExplaining target: {target}")
            self.plot_shap_for_target(target_label=target, class_idx=1)
            self.plot_lime_for_target(target_label=target, instance_idx=0, num_features=10)



    # Predict both targets for a new client (joint)
    def pred_new_client_joint(self, new_client):
        """
        Predicts both 'AccumulationInvestment' and 'IncomeInvestment' for a new client.
        Args:
            new_client (pd.DataFrame): DataFrame containing the new client's data.
        """

        exp = 0.3
        new = new_client.copy()
        new["Log_Wealth"]       = self.apply_log_transform(new, "Wealth")
        new[f"Power_Income_{exp}"] = self.apply_power_transform(new, "Income", exp)
        for col, lo, hi in zip(self.vars_to_normalize, self.scaler.data_min_, self.scaler.data_max_):
            new[col] = new[col].clip(lo, hi)
        new[self.vars_to_normalize] = self.scaler.transform(new[self.vars_to_normalize])
        new = new.drop(columns=["Wealth", "Income"])
        X_new = new[self.joint_model.estimators_[0].feature_names_in_]
        preds = self.joint_model.predict(X_new)
        probas = [est.predict_proba(X_new)[:, 1] for est in self.joint_model.estimators_]
        acc_pred, inc_pred = preds[0][0], preds[0][1]
        acc_prob, inc_prob = probas[0][0], probas[1][0]
        print(f"AccumulationInvestment → p={acc_prob:.3f} → label={acc_pred}")
        print(f"IncomeInvestment       → p={inc_prob:.3f} → label={inc_pred}")
        return (acc_pred, acc_prob), (inc_pred, inc_prob)

    # New: SHAP plotting as a method
    def plot_shap_for_target(self, target_label= 'AccumulationInvestment', class_idx=1):
        """
        Plots SHAP summary bar and dot plots for the specified target.
        target_label: must be one of ['AccumulationInvestment','IncomeInvestment']
        class_idx: which class index to explain (for binary, use 1)
        """
        
        if not hasattr(self, 'joint_model'):
            raise RuntimeError("Model is not trained yet. Call train_evaluate_joint() first.")
        # Determine which estimator index corresponds to target
        if target_label not in self.y_train.columns:
            raise ValueError(f"Unknown target_label: {target_label}")
        idx = list(self.y_train.columns).index(target_label)
        estimator = self.joint_model.estimators_[idx]
        X = self.X_test
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
        shap_vals = shap_values[class_idx] if isinstance(shap_values, list) else shap_values

        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
        plt.title(f"{target_label} - SHAP Importance (bar)")
        plt.tight_layout()
        plt.show()

        # Dot plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X, show=False)
        plt.title(f"{target_label} - SHAP Feature Impacts")
        plt.tight_layout()
        plt.show()

    # New: LIME plotting as a method
    def plot_lime_for_target(self, target_label='AccumulationInvestment', instance_idx=0, num_features=10):
        """
        Generates a LIME explanation plot for a single test instance for the specified target.
        target_label: must be one of ['AccumulationInvestment','IncomeInvestment']
        instance_idx: index of the row in X_test to explain
        num_features: how many top features to display
        """

        if not hasattr(self, 'joint_model'):
            raise RuntimeError("Model is not trained yet. Call train_evaluate_joint() first.")
        if target_label not in self.y_train.columns:
            raise ValueError(f"Unknown target_label: {target_label}")
        idx = list(self.y_train.columns).index(target_label)
        estimator = self.joint_model.estimators_[idx]
        X_train = self.X_train
        X_test  = self.X_test
        instance = X_test.iloc[instance_idx]
        feature_names = list(X_train.columns)
        class_names = ['Class 0', 'Class 1']

        explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )
        exp = explainer.explain_instance(instance.values, estimator.predict_proba, num_features=num_features)
        fig = exp.as_pyplot_figure()
        plt.title(f"LIME Explanation for {target_label}, instance {instance_idx}")
        plt.tight_layout()
        plt.show()
        return exp
    
    def plot_confusion_matrix_for_target(self, target_label='AccumulationInvestment'):
        """
        Plots the confusion matrix on the test set for the specified target.
        Must call train_evaluate_joint() before using this.
        """
        
        # Find the index of the specified target
        idx = list(self.y_test.columns).index(target_label)
        estimator = self.joint_model.estimators_[idx]
        
        # True labels and predicted labels based on τ
        y_true = self.y_test[target_label]
        probs = estimator.predict_proba(self.X_test)[:, 1]
        thresh = self.tau_dict[target_label]
        y_pred = (probs >= thresh).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        
        plt.figure(figsize=(6, 6))
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix for {target_label}")
        plt.tight_layout()
        plt.show()
