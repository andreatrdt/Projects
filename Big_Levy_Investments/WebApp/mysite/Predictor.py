from config_bc2 import cost_map_single

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

#%%
class Predictor:
    def __init__(
            self,
            model
        ):

        self.model = model
        # needs_df = pd.read_excel('Final_Project/BC_2/Dataset2_Needs.xls', sheet_name='Needs')
        needs_df = pd.read_excel('Dataset2_Needs.xls', sheet_name='Needs')
        self.products_df = pd.read_excel('Dataset2_Needs.xls', sheet_name='Products')
        self.metadata_df = pd.read_excel('Dataset2_Needs.xls', sheet_name='Metadata')       


        needs_df = needs_df.drop('ID', axis=1)

                # Create summary tables
        print("NEEDS VARIABLES SUMMARY:")
        needs_summary = create_variable_summary( needs_df, self.metadata_df)
        display(needs_summary.style
                .set_properties(**{'text-align': 'left'})
                .hide(axis='index'))

        print("\nPRODUCTS VARIABLES SUMMARY:")
        products_summary = create_variable_summary( self.products_df, self.metadata_df)
        display(products_summary.style
                .set_properties(**{'text-align': 'left'})
                .hide(axis='index'))
        
        # Create df with all transformations and normalizations
        self.transformed_df = needs_df.copy()

        
        log_wealth = self.apply_log_transform( self.transformed_df, 'Wealth')

        self.transformed_df['Log_Wealth'] = log_wealth

        # Power transformation with different exponents
        print("\n\033[1mPower transformation with different exponents\033[0m")
        exponent = 0.3
        power_income = self.apply_power_transform( self.transformed_df, 'Income', exponent)
            
        # Wrap the transformed arrays in DataFrames with correct column names
        self.transformed_df[f'Power_Income_{exponent}'] = power_income
            
        shapiro_stat_income, shapiro_p_income, ks_stat_income, ks_p_income = test_normality(self.transformed_df, f'Power_Income_{exponent}')
        print(f"\nShapiro-Wilk test for Power_Income_{exponent}: Statistic={shapiro_stat_income}, p-value={shapiro_p_income}")

        self.scaler = MinMaxScaler()
        self.transformed_df = self.transformed_df.drop(columns=['Wealth', 'Income'])

        # and update the list you later scale
        self.vars_to_normalize = [
            'Age','FamilyMembers','RiskPropensity',
            'Log_Wealth',               # keep the transformed one
            f'Power_Income_{exponent}'
        ]
        # then fit the scaler
        self.transformed_df[self.vars_to_normalize] = (
            self.scaler.fit_transform(self.transformed_df[self.vars_to_normalize])
        )
        self.cost_map_single = cost_map_single

    # #####################################################################
    # def train_evaluate_AccumulationInvestment(self):
    #     data = self.transformed_df.copy()
    #     self.name = 'AccumulationInvestment'
    #     X = data.drop(columns=['IncomeInvestment', self.name])
    #     y = data[self.name]

    #     # split_data returns: X_train, X_test, y_train, y_test
    #     X_train, X_test, y_train, y_test = self.split_data(X, y)

    #     _, self.y_pred_Acc, self.y_probs_Acc, self.results_train_Acc = \
    #         self.train_evaluate_model(X_train, y_train, X_test, y_test, 5)

    #     self.display_results_table(self.results_train_Acc,
    #                                "Random Forest",
    #                                "Accumulation Investment")

    # def train_evaluate_IncomeInvestment(self):
    #     data = self.transformed_df.copy()
    #     # remove that trailing comma here!
    #     self.name = 'IncomeInvestment'
    #     X = data.drop(columns=[self.name, 'AccumulationInvestment'])
    #     y = data[self.name]

    #     X_train, X_test, y_train, y_test = self.split_data(X, y)

    #     _, self.y_pred_Inc, self.y_probs_Inc, self.results_train_Inc = \
    #         self.train_evaluate_model(X_train, y_train, X_test, y_test, 5)

    #     self.display_results_table(self.results_train_Inc,
    #                                "Random Forest",
    #                                "Income Investment")
    # #####################################################################


    def custom_penalty_single(self, y_true, y_pred, cost_map_single):
        total_penalty = 0.0
        n = len(y_true)

        # convert to arrays to avoid index issues
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        for i in range(n):
            actual = y_true_arr[i]     # 0 or 1
            predicted = y_pred_arr[i]  # 0 or 1
            total_penalty += cost_map_single[(actual, predicted)]

        return (total_penalty / n)/10
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def train_evaluate_model(self, X_train, y_train, X_test, y_test, k_folds=5):

        # -------- Optuna hyperparameter tuning --------
        best_params = self.study_optuna(
            target=self.name,
            n_trials=1,
            timeout=None,
            show_plot=True)

        # -------- set best params to model --------
        self.model.set_params(**best_params)


        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'custom_penalty': []
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold =y_train.iloc[train_idx], y_train.iloc[val_idx]

            self.model.fit(X_train_fold, y_train_fold)
            y_val_pred = self.model.predict(X_val_fold)

            cv_metrics['accuracy'].append(accuracy_score(y_val_fold, y_val_pred))
            cv_metrics['precision'].append(precision_score(y_val_fold, y_val_pred))
            cv_metrics['recall'].append(recall_score(y_val_fold, y_val_pred))
            cv_metrics['f1'].append(f1_score(y_val_fold, y_val_pred))
            cv_metrics['custom_penalty'].append(self.custom_penalty_single(y_val_fold, y_val_pred, cost_map_single))

        self.model.fit(X_train, y_train)
        # -------- store CV-probabilities for threshold tuning --------
        if hasattr(self, "name"):                     # set by calling method
            cv_probs = cross_val_predict(
                self.model, X_train, y_train,
                cv=5, method="predict_proba", n_jobs=-1
            )[:, 1]

            if self.name == "AccumulationInvestment":
                self.tau = best_tau_f1(y_train, cv_probs)       # or use best_tau_penalty(...)
            else:
                self.tau = best_tau_f1(y_train, cv_probs)
            print(f"Optimal τ ({self.name}) = {self.tau:.3f}")

        y_test_pred = self.model.predict(X_test)

        if hasattr(self.model, "predict_proba"):
            # If the model has a predict_proba method, use it to get probabilities
            y_test_pred_proba = self.model.predict_proba(X_test)[:,1]
        else:
            # If not, use the decision function or similar method
            y_test_pred_proba = 0

        return y_val_pred, y_test_pred, y_test_pred_proba, {
            'cv_metrics': {
                metric: {
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                } for metric, scores in cv_metrics.items()
            },
            'test_metrics': {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred),
                'f1': f1_score(y_test, y_test_pred),
                'custom_penalty': self.custom_penalty_single(y_test, y_test_pred, cost_map_single)
            }
        }
    
    def display_results_table(self, results_dict, model_name, feature_type):
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

        df = pd.DataFrame(cv_data)
        df = df.round(3)

        print(f"\n{model_name} - {feature_type}")
        print("=" * 60)
        print(tabulate(df, headers='keys', tablefmt='pretty'))

    # Function to apply Power transformation with different exponents
    def apply_power_transform(self, data, feature, exponent):
        transformed_data = np.power(data[feature], exponent)
        return transformed_data

    # Function to apply log transformation
    def apply_log_transform(self, data, feature):
        # np.log1p handles zero values.
        transformed_data = np.log1p(data[feature])
        return transformed_data
    def pred_new_client(self, new_client):
        # ---------- make feature vector ----------
        exp = 0.3
        new_client = new_client.copy()
        new_client["Log_Wealth"]        = self.apply_log_transform(new_client, "Wealth")
        new_client["Power_Income_0.3"]  = self.apply_power_transform(new_client, "Income", exp)

        # clip to training range (optional but safe)
        for col, lo, hi in zip(self.vars_to_normalize,
                               self.scaler.data_min_,
                               self.scaler.data_max_):
            new_client[col] = new_client[col].clip(lower=lo, upper=hi)

        new_client[self.vars_to_normalize] = self.scaler.transform(
            new_client[self.vars_to_normalize])

        new_client = new_client.drop(columns=["Wealth", "Income"])
        X_new = new_client[self.model.feature_names_in_]

        # ---------- predict ----------
        prob = self.model.predict_proba(X_new)[:, 1][0]
        label = int(prob >= getattr(self, "tau", 0.5))   # default 0.5 if tau missing
        print(f"{self.name}  →  p={prob:.3f}  τ={getattr(self,'tau',0.5):.3f}  label={label}")
        return label, prob
    
    def _objective(self, trial, target):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
            "max_depth":    trial.suggest_int("max_depth", 5, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 15),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None, 0.5, 0.8]),
            "bootstrap":    trial.suggest_categorical("bootstrap",[True, False]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]),
            "random_state": 42, "n_jobs": -1
        }
        model = RandomForestClassifier(**params)

        # -------- build X,y once per call --------
        X = self.transformed_df.drop(columns=['IncomeInvestment','AccumulationInvestment'])
        y = self.transformed_df[target]

        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        score = cross_val_score(model, X, y,
                                cv=cv,
                                scoring=make_scorer(f1_score, average="binary"),
                                n_jobs=-1).mean()
        return score   # Optuna will maximise


    def study_optuna(self,
                    target="AccumulationInvestment",
                    n_trials=60,
                    timeout=None,
                    show_plot=True):


        # study = optuna.create_study(direction="maximize",
        #                             study_name=f"{self.name}_{target}")
        
        # simple, works fine
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{self.name}_{target}",
            storage=f"sqlite:///optuna_{self.name}.db",
            load_if_exists=True
        )
        
        study.optimize(lambda tr: self._objective(tr, target),
                    n_trials=n_trials,
                    timeout=timeout,
                    show_progress_bar=True)

        print("Best F1 = %.4f" % study.best_value)
        print("Best params:", study.best_params)
        self.best_params = study.best_params  # store for inspection

        if show_plot:
            try:
                optuna.visualization.plot_optimization_history(study)
                plt.show()
            except Exception:
                pass

        return study.best_params
        
    def train_evaluate_joint(
            self,
            k_folds: int = 5,
            tau_selector=None,          # default → best_tau_f1
            test_size: float = 0.20,
            random_state: int = 42,
        ):
        """
        Jointly predict AccumulationInvestment & IncomeInvestment with

        • K-fold cross-validation on the training portion  
        • per-label τ tuning (default: F1-optimal)  
        • a final fit on all training data and evaluation on a hold-out test set

        Returns
        -------
        cv_report   : pd.DataFrame  (mean ± std over folds)
        test_report : pd.DataFrame  (metrics on the test set)
        tau_dict    : dict          {label: optimal τ}
        """

        # ------------- 1. Hold-out split -----------------
        data   = self.transformed_df.copy()
        X_full = data.drop(columns=['AccumulationInvestment', 'IncomeInvestment'])
        y_full = data[['AccumulationInvestment', 'IncomeInvestment']]

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full,
            test_size   = test_size,
            random_state= random_state,
            shuffle     = True,          # no 2-D stratify
        )

        # ------------- 2. Cross-validation ---------------
        kf          = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        base        = self.model.__class__(**self.model.get_params())   # fresh clone
        joint_model = MultiOutputClassifier(base)

        # Containers
        oof_probs   = {lbl: np.zeros(len(X_train))                for lbl in y_train.columns}
        fold_stats  = {lbl: {m: [] for m in
                    ['accuracy', 'precision', 'recall', 'f1', 'custom_penalty']}
                    for lbl in y_train.columns}

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            joint_model.fit(X_tr, y_tr)
            prob_list = joint_model.predict_proba(X_val)   # list of (n_val × 2) arrays

            for i, lbl in enumerate(y_train.columns):
                probs = prob_list[i][:, 1]
                preds = (probs >= 0.5).astype(int)         # 0.5 for CV metric collection
                true  = y_val.iloc[:, i]

                # store OOF probs (for τ)
                oof_probs[lbl][val_idx] = probs

                # CV metrics
                fold_stats[lbl]['accuracy'      ].append(accuracy_score(true, preds))
                fold_stats[lbl]['precision'     ].append(precision_score(true, preds, zero_division=0))
                fold_stats[lbl]['recall'        ].append(recall_score(true, preds, zero_division=0))
                fold_stats[lbl]['f1'            ].append(f1_score(true, preds, zero_division=0))
                fold_stats[lbl]['custom_penalty'].append(
                    self.custom_penalty_single(true, preds, self.cost_map_single)
                )

        # ------------- 3. τ tuning -----------------------
        if tau_selector is None:
            tau_selector = best_tau_f1

        tau_dict = {
            lbl: tau_selector(y_train[lbl], oof_probs[lbl])
            for lbl in y_train.columns
        }
        for lbl, τ in tau_dict.items():
            print(f"Optimal τ for {lbl}: {τ:.3f}")

        # ------------- 4. Final fit & test ---------------
        joint_model.fit(X_train, y_train)
        self.joint_model = joint_model
        self.tau_dict    = tau_dict

        # Save the model and tau_dict
        joblib.dump(self.joint_model, 'joint_model.joblib')
        with open('tau_dict.json', 'w') as f:
            json.dump(self.tau_dict, f)
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

        # ------------- 5. Summaries ----------------------
        cv_report = pd.DataFrame({
            lbl: {m: f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
                for m, vals in metr.items()}
            for lbl, metr in fold_stats.items()
        }).T

        test_report = pd.DataFrame(test_stats).T.round(3)

        return cv_report, test_report, tau_dict
    

    def pred_new_client_joint(self, new_client):
        """
        Predict both targets for a new client, without Optuna thresholds.
        """
        exp = 0.3
        new = new_client.copy()
        new["Log_Wealth"]       = self.apply_log_transform(new, "Wealth")
        new["Power_Income_0.3"] = self.apply_power_transform(new, "Income", exp)

        for col, lo, hi in zip(self.vars_to_normalize,
                            self.scaler.data_min_,
                            self.scaler.data_max_):
            new[col] = new[col].clip(lo, hi)
        new[self.vars_to_normalize] = self.scaler.transform(new[self.vars_to_normalize])
        new = new.drop(columns=["Wealth", "Income"])

        X_new = new[self.joint_model.estimators_[0].feature_names_in_]

        # predict
        preds = self.joint_model.predict(X_new)           # shape (1, 2)
        probas = [est.predict_proba(X_new)[:, 1] 
                for est in self.joint_model.estimators_]

        # unpack the first (only) row
        acc_pred, inc_pred = preds[0]
        acc_prob, inc_prob = probas[0][0], probas[1][0]

        print(f"AccumulationInvestment → p={acc_prob:.3f} → label={acc_pred}")
        print(f"IncomeInvestment       → p={inc_prob:.3f} → label={inc_pred}")
        return (acc_pred, acc_prob), (inc_pred, inc_prob)





    
def best_tau_f1(y_true, probs):
    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    return thr[np.nanargmax(f1)]

# choose τ that minimises expected penalty
def best_tau_penalty(y_true, probs, cost):
    taus = np.linspace(0, 1, 1001)
    exp  = []
    for τ in taus:
        y_pred = (probs >= τ).astype(int)
        e = np.mean([cost[(int(a), int(p))] for a, p in zip(y_true, y_pred)])
        exp.append(e)
    return taus[np.argmin(exp)]
                


def create_variable_summary(df, metadata_df):
    # Create empty lists to store the chosen statistics
    stats_dict = {
        'Variable': [],
        'Description': [],
        'Mean': [],
        'Std': [],
        'Missing': [],
        'Min': [],
        'Max': []
    }

    # Create a metadata dictionary for easy lookup
    meta_dict = dict(zip(metadata_df['Metadata'], metadata_df['Unnamed: 1']))

    for col in df.columns:
        stats_dict['Variable'].append(col)
        stats_dict['Description'].append(meta_dict.get(col, 'N/A'))

        # Calculate some statistics for each column
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



from scipy.stats import shapiro, kstest, boxcox
from sklearn.preprocessing import PowerTransformer

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




