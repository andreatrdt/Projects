from scipy.stats import shapiro, kstest, boxcox
from sklearn.preprocessing import PowerTransformer

import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import math
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, make_scorer
from tabulate import tabulate
from sklearn.model_selection import cross_val_predict
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib
import json
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from lime.lime_tabular import LimeTabularExplainer

# Function to perform Shapiro-Wilk test and Kolmogorov-Smirnov test
def test_normality(data, feature):
    shapiro_stat, shapiro_p = shapiro(data[feature])
    ks_stat, ks_p = kstest(data[feature], 'norm', args=(data[feature].mean(), data[feature].std()))

    # Check if p-value is less than 0.05
    print(f"Shapiro-Wilk test for {feature}: Statistic={shapiro_stat}, p-value={shapiro_p}")
    if shapiro_p < 0.05:
        print(f"{feature} does not follow a normal distribution (Shapiro-Wilk test).")
    else:
        print(f"{feature} follows a normal distribution (Shapiro-Wilk test).")

    print(f"KS test for {feature}: Statistic={ks_stat}, p-value={ks_p}")
    # Check if p-value is less than 0.05
    if ks_p < 0.05:
        print(f"{feature} does not follow a normal distribution (KS test).")
    else:
        print(f"{feature} follows a normal distribution (KS test).")
    # Return the test statistics and p-values
    return shapiro_stat, shapiro_p, ks_stat, ks_p

# Function to plot histogram and Q-Q plot
def plot_histogram_and_qq(data, feature, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data[feature], kde=True)
    plt.title(f'Histogram of {title}')
    plt.xlabel(title)

    plt.subplot(1, 2, 2)
    stats.probplot(data[feature], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {title}')
    plt.show()

# Create a function for histogram plotting
def plot_histogram(data, title, xlabel, ylabel="Frequency"):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True)
    plt.title(title, pad=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Box Plot: To identify outliers in a numeric feature (optionally grouped by a categorical variable)
def plot_boxplot(data, feature, group=None, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(10, 6))
    if group:
        sns.boxplot(x=group, y=feature, data=data)
        plt.xlabel(xlabel if xlabel else group)
    else:
        sns.boxplot(y=data[feature])
    plt.title(title if title else f'Boxplot of {feature}')
    plt.ylabel(ylabel if ylabel else feature)
    plt.show()

# Q-Q Plot: To check if a numeric feature follows a normal distribution
def plot_qq(data, feature, title=None):
    plt.figure(figsize=(8, 6))
    stats.probplot(data[feature], dist="norm", plot=plt)
    plt.title(title if title else f'Q-Q Plot for {feature}')
    plt.show()

# Violin Plot: To compare the distribution of a numeric feature across categories
def plot_violin(data, numeric_feature, categorical_feature, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=categorical_feature, y=numeric_feature, data=data)
    plt.title(title if title else f'Violin Plot of {numeric_feature} by {categorical_feature}')
    plt.xlabel(xlabel if xlabel else categorical_feature)
    plt.ylabel(ylabel if ylabel else numeric_feature)
    plt.show()


# Joint Plot: To visualize the joint and marginal distributions of two features
def plot_joint_distribution(data, x_feature, y_feature, title=None):
    jp = sns.jointplot(data=data, x=x_feature, y=y_feature, kind='scatter', height=8, marginal_kws=dict(bins=20, fill=True))
    jp.figure.suptitle(title if title else f'Joint Distribution of {x_feature} and {y_feature}', y=1.02)
    plt.show()

# Function to plot multiple violin plots for several numeric features against one or more categorical features
def plot_multiple_violin_plots(data, numeric_features, categorical_features, ncols=2):
    for cat_feature in categorical_features:
        n = len(numeric_features)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
        axes = axes.flatten()
        for i, num_feature in enumerate(numeric_features):
            sns.violinplot(x=cat_feature, y=num_feature, data=data, ax=axes[i])
            axes[i].set_title(f'{num_feature} by {cat_feature}')
            axes[i].set_xlabel(cat_feature)
            axes[i].set_ylabel(num_feature)
        # Remove any extra axes if they exist
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.suptitle(f'Violin Plots for {cat_feature}', y=1.02)
        plt.tight_layout()
        plt.show()



def plot_feature_importance_ax(model, feature_names, title, ax):
    """
    Plots feature importance on the provided axis.
    
    Parameters:
        model: Trained model with a 'feature_importances_' attribute.
        feature_names (iterable): List of feature names.
        title (str): Title for the plot.
        ax (matplotlib.axes.Axes): Axis to plot on.
    """
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=importances, x='importance', y='feature', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature')


def plot_shap_values(model, X, title, class_idx=1):
    """
    Plots SHAP values for a RandomForestClassifier (binary or multiclass).
    class_idx: index of the class to explain (default: 1 for binary classification).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle binary or multiclass classification
    if isinstance(shap_values, list):
        shap_vals = shap_values[class_idx]  # Take class 1 by default
    else:
        shap_vals = shap_values

    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # Dot plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, show=False)
    plt.title(f"{title} - Feature Impacts")
    plt.tight_layout()
    plt.show()



def plot_shap_values_force(model, X, title):
    """
    Generates SHAP visualizations for the given model and dataset X:
    - A bar plot of aggregated absolute SHAP values (global feature importance).
    - A summary (dot) plot showing the distribution of SHAP values per feature.
    - A force plot for local interpretability on an individual prediction.
    """
    # Create SHAP explainer and compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
       
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
    plt.title(f"{title} - Individual Force Plot")
    plt.show()
    
    return explainer, shap_values

def plot_permutation_importance(model, X_test, y_test, scoring='accuracy'):
    """
    Computes and plots permutation importance for the given model.
    """
    result = permutation_importance(model, X_test, y_test, scoring=scoring, n_repeats=10, random_state=42)

    sorted_idx = result.importances_mean.argsort()[::-1]
    plt.figure(figsize=(10, 8))

    feature_names = X_test.columns
    
    plt.bar(range(len(sorted_idx)), result.importances_mean[sorted_idx])
    plt.xticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx], rotation=90)
    plt.title('Permutation Importance')
    plt.tight_layout()
    plt.show()

def plot_partial_dependence(model, X, features, target=None):
    """
    Generates a Partial Dependence Plot (PDP) for the specified features using the given model and dataset.
    
    Parameters:
    - model: Trained model.
    - X: DataFrame or array-like feature data.
    - features: List of feature indices or names to plot.
    - target: (Optional) For classifiers, specify the target class for which to compute the PDP.
    """
    print(f"Partial dependence for {features}")

    # Validate that the provided features are in the data
    if isinstance(X, pd.DataFrame):
        available_features = list(X.columns)
        # If features are provided as indices, convert them to names
        if isinstance(features[0], int):
            features = [available_features[i] for i in features if i < len(available_features)]
    else:
        if isinstance(features[0], int):
            if max(features) >= X.shape[1]:
                raise ValueError(f"Feature index out of range. X has {X.shape[1]} features.")
    
    fig, ax = plt.subplots(1,4, figsize=(20, 5))
    PartialDependenceDisplay.from_estimator(model, X, features, target=target, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_lime_explanation(model, X_train, instance, feature_names, class_names, num_features=10):
    """
    Generates a local explanation for a single instance using LIME.
    
    Parameters:
    - model: Trained model with a predict_proba method (for classification).
    - X_train: Training data as a NumPy array or DataFrame.
    - instance: A single instance (row) from the dataset (as 1D array or Series).
    - feature_names: List of feature names.
    - class_names: List of class names (for classification).
    - num_features: Number of features to include in the explanation.
    """
    # Convert to numpy array if needed.
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(instance, 'values'):
        instance = instance.values

    explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names, mode='classification')
    exp = explainer.explain_instance(instance, model.predict_proba, num_features=num_features)
    
    # Display the explanation as a matplotlib figure
    fig = exp.as_pyplot_figure()
    plt.title("LIME Explanation")
    plt.tight_layout()
    plt.show()
    
    return exp