"""PortfolioReplicator: class to replicate a target index using a basket of underlyings

Author: <Nicolò Toia>
Date: 2025‑05‑25

Overview
--------
The class provides a high‑level, batteries‑included workflow:
    >>> rep = PortfolioReplicator(df_target, df_underlyings)
    >>> rep.eda()
    >>> rep.run_eqw()
    >>> rep.run_elasticnet(use_optuna=True, n_trials=100)
    >>> rep.display_results()

Public API
~~~~~~~~~~
- eda()                     : basic exploratory data analysis
- run_eqw()                 : equal‑weight replication
- run_elasticnet()          : Elastic‑Net (with optional Optuna tuning)
- run_kf()                  : Kalman Filter (PYKalman)
- run_ekf()                 : Extended Kalman Filter (filterpy)
- display_results()         : tables and charts summarising the chosen run

Private helpers (underscored) implement plumbing and hyper‑parameter search.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, List, Any

import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

# Optional dependencies
try:
    from pykalman import KalmanFilter  # type: ignore
except ImportError:
    KalmanFilter = None  # runtime check later

try:
    import filterpy.kalman as fk  # type: ignore
except ImportError:
    fk = None

try:
    import optuna  # type: ignore
except ImportError:
    optuna = None

from IPython.display import display
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from IPython.display import display
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from sklearn.linear_model import LinearRegression
from typing import Sequence
import optuna
from sklearn.linear_model import Ridge
from pykalman import KalmanFilter  # ensure pykalman is installed
from optuna.exceptions import TrialPruned
from typing import Tuple


class PortfolioReplicator:
    """Replicate a target index using a set of underlyings.

    Parameters
    ----------
    df_target : pd.Series | pd.DataFrame
        Target series (single column) of index returns.
    df_underlyings : pd.DataFrame
        DataFrame whose columns are candidate replicating assets (returns).
    frequency : str, default "daily"
        Display label only; does not affect calculations.
    name : str, optional
        Human‑friendly name of the replica experiment.
    """

    def __init__(
        self,
        df_target: pd.DataFrame,
        index_components: Dict[str, float],
        df_underlyings: pd.DataFrame,
        var_parameters: Dict[str, float] = {},
        transaction_cost: float = 0.0,  # default no transaction costs
        annual_factor: int = 52,  # default to weekly data
        *,
        frequency: str = "weekly",
        name: Optional[str] = None,
    ) -> None:
        self.frequency = frequency
        self.name = name or "Replica"
        
        self._y = pd.DataFrame(df_target.astype(float))
        self._X = pd.DataFrame(df_underlyings.astype(float))

        # # sync indices & drop missing
        # common_idx = self._y.index.intersection(self._X.index)
        # self._y = self._y.loc[common_idx].dropna()
        # self._X = self._X.loc[common_idx].dropna(how="any")

        # # keep final common index
        # self._index = self._y.index.intersection(self._X.index)
        # self._y = self._y.loc[self._index]
        # self._X = self._X.loc[self._index]

        if self._X.isnull().values.any():
            raise ValueError("NaNs remain in replicating underlyings after alignment.")

        if self._y.isnull().values.any():
            raise ValueError("NaNs remain in target series after alignment.")
        ##########################################################################################
        component_returns = self._y.pct_change().dropna() 
        component_returns_log = np.log(1 + component_returns)

        # Create the target index using weighted returns
        weighted_returns = pd.DataFrame(index=component_returns.index)
        weighted_returns_log = pd.DataFrame(index=component_returns_log.index)
        for component, weight in index_components.items():
            weighted_returns[component] = component_returns[component] * weight
            weighted_returns_log[component] = component_returns_log[component] * weight

        self.target_returns = pd.DataFrame(weighted_returns.sum(axis=1))
        self.target_returns_log = pd.DataFrame(weighted_returns_log.sum(axis=1))
        ###########################################################################################

        self.underlyings_returns = self._X.pct_change().dropna()
        self.underlyings_returns_log = np.log(1 + self.underlyings_returns)

        # result containers --------------------------------------------------------------
        self.weights_: Optional[pd.Series] = None
        self.replica_return_: Optional[pd.Series] = None
        self.tracking_error_: Optional[float] = None
        self._run_name: Optional[str] = None  # last run name (eqw, elasticnet, ...)

        # cache scaler to reuse when tuning ENet -----------------------------------------
        self._scaler = StandardScaler()
        self._X_scaled: Optional[np.ndarray] = None

        # Set the var parameters
        self.var_confidence = var_parameters.get('var_confidence')
        self.var_horizon = var_parameters.get('var_horizon')   
        self.max_var_threshold = var_parameters.get('max_var_threshold') 

        self.transaction_cost_rate = transaction_cost  # transaction cost per trade
        self.annual_factor = annual_factor  # annualization factor for returns


    ###################################################################################################
    ###################################################################################################
    # -----------------------------------------------------------------------------
    # 1. EDA
    # -----------------------------------------------------------------------------
    ###################################################################################################
    ###################################################################################################
    
    
    def plot_cumulative_returns(self) -> None:
        """Plot cumulative returns of target and underlyings."""

        cum_target = (1 + self.target_returns).cumprod()
        cum_underlyings = (1 + self.underlyings_returns).cumprod()

        plt.figure(figsize=(12, 6))
        plt.plot(cum_target, label="Target", color="black")
        plt.title("Target Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        for col in cum_underlyings.columns:
            plt.plot(cum_underlyings[col], label=col)
        plt.title("Underlyings Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.show()

    def fin_stats(self) -> pd.DataFrame:
        ''' the function must return:
            stats = pd.DataFrame({
    'Annualized Return': returns.mean() * annual_factor,
    'Annualized Volatility': returns.std() * np.sqrt(annual_factor),
    'Sharpe Ratio': (returns.mean() * annual_factor) / (returns.std() * np.sqrt(annual_factor)),
    'Max Drawdown': returns.apply(lambda x: (1 + x).cumprod().div((1 + x).cumprod().cummax()) - 1).min(),
    'Skewness': returns.skew(),
    'Kurtosis': returns.kurtosis()
        })
        '''
        annual_factor = self.annual_factor
        target_stats = pd.DataFrame({
            'Annualized Return': self.target_returns.mean() * annual_factor,
            'Annualized Volatility': self.target_returns.std() * np.sqrt(annual_factor),
            'Sharpe Ratio': (self.target_returns.mean() * annual_factor) / (self.target_returns.std() * np.sqrt(annual_factor)),
            'Max Drawdown': self.target_returns.apply(lambda x: (1 + x).cumprod().div((1 + x).cumprod().cummax()) - 1).min(),
            'Skewness': self.target_returns.skew(),
            'Kurtosis': self.target_returns.kurtosis()
            })
        
        underlying_stats = pd.DataFrame({
                 'Annualized Return': self.underlyings_returns.mean() * annual_factor,
                'Annualized Volatility': self.underlyings_returns.std() * np.sqrt(annual_factor),
                'Sharpe Ratio': (self.underlyings_returns.mean() * annual_factor) / (self.underlyings_returns.std() * np.sqrt(annual_factor)),
                'Max Drawdown': self.underlyings_returns.apply(lambda x: (1 + x).cumprod().div((1 + x).cumprod().cummax()) - 1).min(),
                'Skewness': self.underlyings_returns.skew(),
                'Kurtosis': self.underlyings_returns.kurtosis()
            })
        
        # Format as percentage with 2 decimal places
        def format_pct(x):
            return f"{x*100:.2f}%"
        
        for col in ['Annualized Return', 'Annualized Volatility', 'Max Drawdown']:
            target_stats[col] = target_stats[col].apply(format_pct)

        for col in ['Annualized Return', 'Annualized Volatility', 'Max Drawdown']:
            underlying_stats[col] = underlying_stats[col].apply(format_pct)

        return target_stats, underlying_stats
    

    def corr_map(self) -> None:
        """Plot correlation heatmap of target and underlyings."""
        corr = pd.concat([self.target_returns, self.underlyings_returns], axis=1).corr()
        plt.figure(figsize=(17, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
        plt.title("Correlation Heatmap")
        plt.show()

    def return_distribution(self) -> None:
        """Plot distribution of target and underlyings returns. in a for loop in a subplot"""
        """Plot distribution of target and underlyings returns."""

        n = len(self.underlyings_returns.columns) + len(self.target_returns.columns)
        r = np.ceil(n / 4).astype(int) 

        plt.figure(figsize=(r*5, r * 5))
        for i, col in enumerate(self.underlyings_returns.columns):
            plt.subplot(r, 4, i + 2)
            sns.histplot(self.underlyings_returns[col], kde=True, stat="density", bins=30)
            plt.title(f"{col}")
            plt.xlabel("Returns")
            plt.ylabel("Density")

        plt.subplot(r, 4, 1)
        tgt_col = self.target_returns.columns[0]
        sns.histplot(
            self.target_returns[tgt_col],
            kde=True,
            stat="density",
            bins=30,
            legend=False
        )
        plt.title(tgt_col)
        plt.xlabel("Returns")
        plt.ylabel("Density")

        plt.tight_layout()
        plt.show()


    def plot_log_returns(self) -> None:
        """Plot log returns of target and underlyings."""
        n = len(self.underlyings_returns.columns) + len(self.target_returns.columns)
        r = np.ceil(n / 3).astype(int) 

        plt.figure(figsize=(r*5, r * 5))
        for i, col in enumerate(self.underlyings_returns.columns):
            plt.subplot(r, 3, i + 2)
            plt.plot(self.underlyings_returns_log[col], label=col)
            plt.title(f"{col}")
            plt.xlabel("Returns")
            plt.ylabel("Density")

        plt.subplot(r, 3, 1)
        tgt_col = self.target_returns.columns[0]
        plt.plot(self.target_returns_log[tgt_col], label=tgt_col)
        plt.title(tgt_col)
        plt.xlabel("Returns")
        plt.ylabel("Density")

        plt.tight_layout()
        plt.show()

    def plot_rolling_correlation(self, window: int = 30) -> None:
        """Plot rolling correlation between target and underlyings."""
        n = len(self.underlyings_returns.columns) + len(self.target_returns.columns)
        r = np.ceil(n / 3).astype(int) 

        plt.figure(figsize=(r*5, r * 5))
        for i, col in enumerate(self.underlyings_returns.columns):
            plt.subplot(r, 3, i + 1)
            rolling_corr = self.target_returns.rolling(window).corr(self.underlyings_returns[col])
            plt.plot(rolling_corr, label=f"Corr with {col}")
            plt.title(f"Rolling Correlation with {col}")
            plt.xlabel("Date")
            plt.ylabel("Correlation")

        plt.tight_layout()
        plt.show()

    def bar_corr_target_underlyings(self) -> None:
        """
        Plot a horizontal bar chart of the correlation coefficients
        between the target series and each underlying series.
        """
        # 1) extract target as a Series
        if isinstance(self._y, pd.DataFrame):
            # assume single‐column DataFrame
            target = self._y.iloc[:, 0]
        else:
            target = self._y

        # 2) compute correlations
        corrs = self._X.corrwith(target)  # Series indexed by underlying names
        df_corr = pd.DataFrame({
            "Correlation": corrs,
        })
        # sort by absolute first
        df_corr["AbsCorr"] = df_corr["Correlation"].abs()
        df_corr = df_corr.sort_values("AbsCorr", ascending=False).drop("AbsCorr", axis=1)
        # for plotting, sort by signed correlation
        plot_df = df_corr.sort_values("Correlation", ascending=False)

        # 3) plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x="Correlation",
            y=plot_df.index,
            data=plot_df.reset_index(),
            palette="coolwarm"
        )
        plt.title(f"{self.name}: Corr(Target, Underlyings)", fontsize=16)
        plt.xlabel("Correlation coefficient", fontsize=14)
        plt.ylabel("Underlying", fontsize=14)
        plt.axvline(0, color="black", linestyle="--", alpha=0.3)
        plt.grid(True, axis="x", alpha=0.3)

        # 4) annotate bars with values
        for patch, corr in zip(ax.patches, plot_df["Correlation"]):
            width = patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            # place text slightly inside bar
            x = width - 0.02 * np.sign(width)
            ha = "right" if width > 0 else "left"
            ax.text(
                x, y,
                f"{corr:.2f}",
                va="center",
                ha=ha,
                fontsize=12,
                fontweight="bold",
                color="black"
            )

        plt.tight_layout()
        plt.show()

    def normality_check(self,
        top_n: int = 6,
        alpha: float = 0.05,
        figsize: tuple[int,int] = (16, 12),
    ) -> pd.DataFrame:
        """
        For the target series and the top_n underlyings (by abs corr),
        1) Draw QQ‐plots vs a normal dist.
        2) Run Shapiro–Wilk tests and tabulate statistic, p-value, and reject decision.
        Returns
        -------
        pd.DataFrame with columns ['Series','W-stat','p-value','Normal?'].
        """

        # 1) pull target as a Series
        if isinstance(self._y, pd.DataFrame):
            tgt = self._y.iloc[:, 0]
            tgt_name = self._y.columns[0]
        else:
            tgt = self._y
            tgt_name = getattr(self._y, "name", "Target")

        # 2) compute correlations to pick top underlyings
        corrs = self._X.corrwith(tgt).abs().sort_values(ascending=False)
        top_contracts = corrs.iloc[:top_n].index.tolist()

        # 3) assemble the list of (name, series) to test
        to_test = [(tgt_name, tgt)] + [
            (c, self._X[c].dropna()) for c in top_contracts
        ]

        # 4) prepare results storage
        results = []

        # 5) set up QQ‐plot grid
        nplots = len(to_test)
        ncols = 2
        nrows = int(np.ceil(nplots / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()

        for ax, (name, series) in zip(axes, to_test):
            # a) QQ‐plot
            stats.probplot(series, dist="norm", plot=ax)
            ax.set_title(f"QQ: {name}", fontsize=14)
            ax.grid(True, alpha=0.3)

            # b) Shapiro–Wilk
            W, p = stats.shapiro(series)
            reject = "No" if p >= alpha else "Yes"
            # c) annotate on plot
            txt = f"W={W:.3f}\np={p:.3g}\nReject? {reject}"
            ax.text(0.05, 0.85, txt, transform=ax.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round", fc="w", alpha=0.7))

            results.append({
                "Series": name,
                "W-stat": W,
                "p-value": p,
                "Normal?": "✔️" if reject=="No" else "❌"
            })

        # 6) drop any empty axes
        for ax in axes[nplots:]:
            fig.delaxes(ax)

        fig.tight_layout()
        plt.show()

        # 7) summary table
        df_results = (
            pd.DataFrame(results)
              .set_index("Series")
              .sort_values("p-value")
        )
        print("\nShapiro–Wilk Normality Test Results (α=" + str(alpha) + "):")
        display(df_results)

        return

    def stationarity_check(
        self,
        alpha: float = 0.05,
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Perform Augmented Dickey–Fuller tests on each underlying (and target if requested).

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for rejecting the null of a unit root.
        include_target : bool, default True
            Also test the target series.

        Returns
        -------
        pd.DataFrame
            Indexed by series name, with columns:
            ['Test Statistic','p-value','Used Lag','Nobs',
             'Crit(1%)','Crit(5%)','Crit(10%)','Stationary']
        """
        # 1) Build dict of series to test
        series_dict: Dict[str, pd.Series] = {
            col: self._X[col].dropna() for col in self._X.columns
        }
        if include_target:
            if isinstance(self._y, pd.DataFrame):
                tgt_name = self._y.columns[0]
                tgt_series = self._y.iloc[:, 0].dropna()
            else:
                tgt_name = self._y.name or "Target"
                tgt_series = self._y.dropna()
            series_dict[tgt_name] = tgt_series

        results = []
        # 2) Loop & ADF
        for name, series in series_dict.items():
            stat, pval, usedlag, nobs, crits, icbest = adfuller(series, autolag="AIC")
            stationary = pval < alpha
            results.append({
                "Series": name,
                "Test Statistic": stat,
                "p-value": pval,
                "Used Lag": usedlag,
                "Nobs": nobs,
                "Crit(1%)": crits["1%"],
                "Crit(5%)": crits["5%"],
                "Crit(10%)": crits["10%"],
                "Stationary": "✔️" if stationary else "❌"
            })

        # 3) Assemble DataFrame
        df = (
            pd.DataFrame(results)
              .set_index("Series")
              .sort_values("p-value")
        )

        # 4) Display
        print(f"\nAugmented Dickey–Fuller Test (α = {alpha}):")
        display(df.style.format({
            "Test Statistic": "{:.4f}",
            "p-value": "{:.4g}",
            "Crit(1%)": "{:.4f}",
            "Crit(5%)": "{:.4f}",
            "Crit(10%)": "{:.4f}"
        }))

        return


    ###################################################################################################
    ###################################################################################################
    # -----------------------------------------------------------------------------
    # 2. Replication Engines
    # -----------------------------------------------------------------------------
    ###################################################################################################
    ###################################################################################################
    
    # Function to plot the results
    def plot_metrics(self, best_config_normalized, method_title): 
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot(best_config_normalized['cumulative_target'], label='Target index', color='blue')
        plt.plot(best_config_normalized['cumulative_replica'], label='Replica portfolio', color='red')
        plt.title('Cumulative returns: target vs replica - ' + method_title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot drawdowns
        plt.figure(figsize=(12, 6))
        target_drawdown = 1 - best_config_normalized['cumulative_target'] / best_config_normalized['cumulative_target'].cummax()
        replica_drawdown = 1 - best_config_normalized['cumulative_replica'] / best_config_normalized['cumulative_replica'].cummax()
        plt.plot(target_drawdown, label='Target index', color='blue')
        plt.plot(replica_drawdown, label='Replica portfolio', color='red')
        plt.title('Drawdowns: target vs replica - ' + method_title)
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot gross exposure over time
        if 'gross_exposures' in best_config_normalized and best_config_normalized['gross_exposures'] and not all(np.isnan(best_config_normalized['gross_exposures'])):
            plt.figure(figsize=(12, 6))
            gross_exposure_series = pd.Series(best_config_normalized['gross_exposures'], index=best_config_normalized['replica_returns'].index)
            plt.plot(gross_exposure_series, color='purple')
            plt.title('Gross exposure over time - ' + method_title)
            plt.xlabel('Date')
            plt.ylabel('Gross exposure')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Plot VaR over time
        if 'var_values' in best_config_normalized and best_config_normalized['var_values'] and not all(np.isnan(best_config_normalized['var_values'])):
            plt.figure(figsize=(12, 6))
            var_series = pd.Series(best_config_normalized['var_values'], index=best_config_normalized['replica_returns'].index)
            plt.plot(var_series, color='orange')
            plt.axhline(y=self.max_var_threshold, color='r', linestyle='--', label=f'VaR threshold ({self.max_var_threshold*100}%)')
            plt.title('Value at Risk (VaR) over time - ' + method_title)
            plt.xlabel('Date')
            plt.ylabel('VaR (1%, 1M)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Plot scaling factors over time
        if 'scaling_factors' in best_config_normalized and best_config_normalized['scaling_factors'] and not all(np.isnan(best_config_normalized['scaling_factors'])):
            plt.figure(figsize=(12, 6))
            scaling_series = pd.Series(best_config_normalized['scaling_factors'], index=best_config_normalized['replica_returns'].index)
            plt.plot(scaling_series, color='green')
            plt.title('Risk scaling factors over time - ' + method_title)
            plt.xlabel('Date')
            plt.ylabel('Scaling factor')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # plot cumulative costs over time
        if 'transaction_costs' in best_config_normalized and len(best_config_normalized['transaction_costs']) > 0:
            plt.figure(figsize=(12, 6))
            transaction_costs_series = pd.Series(best_config_normalized['transaction_costs'], index=best_config_normalized['replica_returns'].index)
            cumulative_costs = transaction_costs_series.cumsum()
            plt.plot(cumulative_costs, color='brown')
            plt.title('Cumulative transaction costs over time - ' + method_title)
            plt.xlabel('Date')
            plt.ylabel('Cumulative cost')
            plt.grid(True)
            plt.tight_layout()
            plt.show()


        # Plot weights over time
        if 'weights_history' in best_config_normalized and best_config_normalized['weights_history']: # and not all(np.isnan(best_config_normalized['weights_history'])):
            weights_history = best_config_normalized['weights_history']
            weights_df = pd.DataFrame(weights_history, index=best_config_normalized['replica_returns'].index)
            # Get the column names from the original futures data
            weights_df.columns = self._X.columns
            plt.figure(figsize=(16, 6))
            for col in weights_df:
                plt.plot(weights_df[col], label=col)
            plt.title('Portfolio weights over time')
            plt.xlabel('Date')
            plt.ylabel('Weight')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    # Function to calculate VaR
    def calculate_var(self, returns: Sequence[float]) -> float:
        """
        Calculate Value at Risk (VaR) using a simple Gaussian model.

        Parameters
        ----------
        returns : array-like of floats
            Historical returns to use in the VaR calculation.

        Returns
        -------
        float
            VaR as a positive loss number.
        """
        # standard deviation of the historical returns
        sigma = np.std(returns)

        # Gaussian quantile for the configured confidence
        z = stats.norm.ppf(self.var_confidence)

        # scale to the time horizon (weeks → sqrt scaling)
        var = -z * sigma * np.sqrt(self.var_horizon)
        return var
    
    
   
    def run_optuna_normalized(
        self,
        n_trials:   int,
        storage:    str,
        study_name: str,
        obj,
        show_progress_bar: bool = True
    ) -> optuna.Study:
        """
        Run an Optuna study (minimizing) with optional progress bar.

        Parameters
        ----------
        n_trials : int
            Number of trials to run.
        storage : str
            File name (without .db) for the SQLite backend.
        study_name : str
            A unique name for this study.
        obj : Callable[[optuna.Trial], float]
            The objective function (e.g. self._optuna_objective_EN).
        show_progress_bar : bool, default True
            If True, display Optuna’s built‐in tqdm progress bar.

        Returns
        -------
        optuna.Study
            The completed study.
        """
        storage_url = f"sqlite:///{storage}.db"
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True
        )
        study.optimize(
            obj,
            n_trials=n_trials,
            show_progress_bar=show_progress_bar
        )
        self._best_params_norm = study.best_params
        return study


    #############################################################################################################
    #############################################################################################################
    # -----------------------------------------------------------------------------
    # 2.1 Equal Weight
    # -----------------------------------------------------------------------------
    def run_equal_weight_portfolio(self, window: int = 0) -> Dict[str, Any]:

        annual_factor = self.annual_factor
        # Common dates
        common_dates = self.target_returns.index.intersection(self.underlyings_returns.index)
        common_dates = common_dates[window:]

        # Set the variables for the equal-weighted portfolio
        futures_contracts = self.underlyings_returns.columns.tolist()
        equal_weighted_returns = (self.underlyings_returns.mean(axis=1)).squeeze()
        aligned_target = (self.target_returns).squeeze()

        equal_weighted_returns = equal_weighted_returns[common_dates]
        aligned_target = aligned_target[common_dates]

        # Cumulative returns
        cumulative_portfolio = (1 + equal_weighted_returns).cumprod()
        cumulative_target = (1 + aligned_target).cumprod()

        # Annualized return and volatility
        ann_return_portfolio = equal_weighted_returns.mean() * annual_factor
        ann_return_target = aligned_target.mean() * annual_factor

        ann_vol_portfolio = equal_weighted_returns.std() * np.sqrt(annual_factor)
        ann_vol_target = aligned_target.std() * np.sqrt(annual_factor)

        # Sharpe ratios
        sharpe_portfolio = ann_return_portfolio / ann_vol_portfolio if ann_vol_portfolio > 0 else 0
        sharpe_target = ann_return_target / ann_vol_target if ann_vol_target > 0 else 0

        # Tracking error and information ratio
        tracking_error = (equal_weighted_returns - aligned_target).std() * np.sqrt(annual_factor)
        information_ratio = (ann_return_portfolio - ann_return_target) / tracking_error if tracking_error > 0 else 0

        # Correlation
        correlation = equal_weighted_returns.corr(aligned_target)

        # Max drawdown
        drawdown = 1 - cumulative_portfolio / cumulative_portfolio.cummax()
        target_drawdown = 1 - cumulative_target / cumulative_target.cummax()
        max_drawdown = drawdown.max()
        target_max_drawdown = target_drawdown.max()

        # No scaling or exposure tracking for equal-weighted portfolio
        avg_gross_exposure = 1.0
        avg_var = np.nan

        # Weights history
        n_assets = len(futures_contracts)
        equal_weights = np.array([1.0 / n_assets] * n_assets)
        weights_history_equal = [equal_weights.copy()] * len(equal_weighted_returns)

        # Transaction costs only at the first time step
        # (assuming no rebalancing costs for equal-weighted portfolio)
        # In a real scenario, you would calculate transaction costs based on the changes in weights
        # so the value of cost is transiction_cost*n, where n is the number of asssets i am buying/selling
        costs = self.transaction_cost_rate * (len(futures_contracts))

        transaction_costs_series = pd.Series(0.0, index=equal_weighted_returns.index)
        if len(equal_weighted_returns) > 0:
            transaction_costs_series.iloc[0] = costs
        else:
            transaction_costs_series.iloc[0] = 0.0

        return {
            'model': 'EqualWeighted',
            'replica_return': ann_return_portfolio,
            'target_return': ann_return_target,
            'replica_vol': ann_vol_portfolio,
            'target_vol': ann_vol_target,
            'replica_sharpe': sharpe_portfolio,
            'target_sharpe': sharpe_target,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'correlation': correlation,
            'max_drawdown': max_drawdown,
            'target_max_drawdown': target_max_drawdown,
            # 'gross_exposures' : np.nan,  # No gross exposure for equal-weighted portfolio
            'avg_gross_exposure': avg_gross_exposure,
            'avg_var': avg_var,
            'replica_returns': equal_weighted_returns,
            'aligned_target': aligned_target,
            'cumulative_replica': cumulative_portfolio,
            'cumulative_target': cumulative_target,
            'weights_history': weights_history_equal,
            'transaction_costs': transaction_costs_series
            #'scaling_factors': np.nan      # No scaling for equal-weighted portfolio
        }

    #############################################################################################################
    #############################################################################################################
    # -----------------------------------------------------------------------------
    # 2.2 OLS
    # -----------------------------------------------------------------------------

    # def run_linear_regression_normalized(
    #     self,
    #     rolling_window: int,
    #     rebalancing_window: int,
    # ) -> Dict[str, Any]:
    #     """
    #     Rolling, rebalanced linear‐regression on *arithmetic* returns,
    #     with MinMax normalization on X only, VaR‐scaling and transaction costs.
    #     """
    #     annual_factor = self.annual_factor
    #     # 1) Align and pull arithmetic returns
    #     common = self.underlyings_returns.index.intersection(self.target_returns.index)
    #     X_df = self.underlyings_returns.loc[common]      # arithmetic returns
    #     y_ser = self.target_returns.loc[common].squeeze()
    #     X = X_df.values
    #     y = y_ser.values
    #     dates = X_df.index



    #     # 2) Params from self
    #     tc_rate = getattr(self, "transaction_cost", 0.0)
    #     max_var  = getattr(self, "max_var_threshold", np.inf)

    #     # 3) Containers
    #     weights_history   = []
    #     replica_returns   = []
    #     used_dates        = []
    #     gross_exposures   = []
    #     var_values        = []
    #     scaling_factors   = []
    #     transaction_costs = []

    #     # 4) Main loop
    #     n_obs = len(X)
    #     for t in range(n_obs - rolling_window):
    #         # 4a) Rebalance step
    #         if t % rebalancing_window == 0:
    #             X_train = X[t : t + rolling_window]
    #             y_train = y[t : t + rolling_window]

    #             # scale X only
    #             scaler = MinMaxScaler().fit(X_train)
    #             Xn = scaler.transform(X_train)

    #             # fit on raw y
    #             lr = LinearRegression(fit_intercept=False).fit(Xn, y_train)

    #             # undo X‐scaling
    #             w_norm = lr.coef_
    #             w_orig = w_norm / scaler.scale_

    #             # VaR down‐scaling
    #             sf = 1.0
    #             if len(replica_returns) >= 12:
    #                 recent = replica_returns[-min(len(replica_returns), 52):]
    #                 v = self.calculate_var(recent)
    #                 if v > max_var:
    #                     sf = max_var / v
    #                     w_orig *= sf
    #                     v = self.calculate_var([r * sf for r in recent])
    #                 var_values.append(v)
    #             else:
    #                 var_values.append(np.nan)
    #             scaling_factors.append(sf)

    #             # exposures & costs
    #             gross_exposures.append(np.abs(w_orig).sum())
    #             if weights_history:
    #                 turnover = np.abs(w_orig - weights_history[-1]).sum()
    #                 cost = tc_rate * turnover
    #             else:
    #                 cost = 0.0
    #             transaction_costs.append(cost)

    #             weights_history.append(w_orig.copy())

    #         # 4b) One‐step‐ahead return
    #         ret = X[t + rolling_window] @ w_orig
    #         replica_returns.append(ret - cost)
    #         used_dates.append(dates[t + rolling_window])

    #     # 5) Build series & cum‐returns
    #     rep_ser = pd.Series(replica_returns, index=used_dates)
    #     tgt_ser = y_ser.loc[used_dates]
    #     cum_rep = (1 + rep_ser).cumprod()
    #     cum_tgt = (1 + tgt_ser).cumprod()

    #     # 6) Annualized stats
    #     ann = annual_factor
    #     mu_rep = rep_ser.mean() * ann
    #     mu_tgt = tgt_ser.mean() * ann
    #     vol_rep = rep_ser.std() * np.sqrt(ann)
    #     vol_tgt = tgt_ser.std() * np.sqrt(ann)
    #     sharpe_rep = mu_rep / vol_rep if vol_rep>0 else 0
    #     sharpe_tgt = mu_tgt / vol_tgt if vol_tgt>0 else 0
    #     te = (rep_ser - tgt_ser).std() * np.sqrt(ann)
    #     ir = (mu_rep - mu_tgt) / te if te>0 else 0
    #     corr = rep_ser.corr(tgt_ser)
    #     dd_rep = (1 - cum_rep / cum_rep.cummax()).max()
    #     dd_tgt = (1 - cum_tgt / cum_tgt.cummax()).max()

    #     return {
    #         "model": "LinearRegressionNorm",
    #         "rolling_window": rolling_window,
    #         "rebalancing_window": rebalancing_window,
    #         "replica_return": mu_rep,
    #         "target_return": mu_tgt,
    #         "replica_vol": vol_rep,
    #         "target_vol": vol_tgt,
    #         "replica_sharpe": sharpe_rep,
    #         "target_sharpe": sharpe_tgt,
    #         "tracking_error": te,
    #         "information_ratio": ir,
    #         "correlation": corr,
    #         "max_drawdown": dd_rep,
    #         "target_max_drawdown": dd_tgt,
    #         "avg_gross_exposure": np.mean(gross_exposures),
    #         "avg_var": np.nanmean(var_values),
    #         "replica_returns": rep_ser,
    #         "aligned_target": tgt_ser,
    #         "cumulative_replica": cum_rep,
    #         "cumulative_target": cum_tgt,
    #         "gross_exposures": gross_exposures,
    #         "var_values": var_values,
    #         "scaling_factors": scaling_factors,
    #         "weights_history": weights_history,
    #         "transaction_costs": transaction_costs,
    #     }

    # #############################################################################################################
    # #############################################################################################################
    # -----------------------------------------------------------------------------
    # 2.2 Elastic Net
    # -----------------------------------------------------------------------------

    def _optuna_objective_EN(self, trial: optuna.Trial) -> float:
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        rolling_window = trial.suggest_categorical("rolling_window", [52,104,156,208])
        alpha = trial.suggest_loguniform("alpha", 1e-4, 1e-1)
        rebalancing_window = trial.suggest_categorical("rebalancing_window", [4])

        result = self.run_elasticnet_normalized(
            l1_ratio=l1_ratio,
            rolling_window=rolling_window,
            alpha=alpha,
            rebalancing_window=rebalancing_window
        )
        # we want to maximize IR, Optuna minimizes by default → return -IR
        return -(result["information_ratio"])
    
    def run_elasticnet_normalized(
        self,
        l1_ratio: float,
        rolling_window: int,
        alpha: float,
        rebalancing_window: int,
    ) -> Dict[str, Any]:
        """
        ElasticNet replication with rolling-window normalization, VaR scaling, and
        transaction cost adjustment.  On non-rebalance days it reuses the previous weights.
        """

        X_df        = self.underlyings_returns
        y_series    = self.target_returns.iloc[:, 0]
        dates       = y_series.index
        X_values    = X_df.values
        y_values    = y_series.values
        annual_fac  = self.annual_factor

        # result containers
        weights_history   = []
        replica_returns   = []
        target_dates      = []
        gross_exposures   = []
        var_values        = []
        scaling_factors   = []
        transaction_costs = []

        # start with zeros (or you could seed equal‐weight)
        w_orig = np.zeros(X_values.shape[1])
        last_cost = 0.0

        for t in range(rolling_window, len(X_values)):
            # on rebalance days, refit
            if (t - rolling_window) % rebalancing_window == 0:
                start = t - rolling_window
                end   = t

                X_train = X_values[start:end]
                y_train = y_values[start:end]

                # normalize X and y
                scaler_X = MinMaxScaler().fit(X_train)
                Xn       = scaler_X.transform(X_train)
                scaler_y = MinMaxScaler().fit(y_train.reshape(-1, 1))
                yn       = scaler_y.transform(y_train.reshape(-1, 1)).flatten()

                # fit ElasticNet
                model = ElasticNet(
                    alpha=alpha, l1_ratio=l1_ratio,
                    fit_intercept=False, max_iter=10000, tol=1e-6
                )
                model.fit(Xn, yn)

                # back to original scale
                w_norm = model.coef_
                w_orig = w_norm * (scaler_X.scale_ / scaler_y.scale_)

                # VaR scaling
                scale = 1.0
                if len(replica_returns) >= int(rolling_window/4):
                    # last up to rolling_window returns
                    hist = [
                        np.dot(X_values[start + j], weights_history[-1])
                        for j in range(min(len(replica_returns), rolling_window))
                    ]
                    v = self.calculate_var(hist)
                    if self.max_var_threshold and v > self.max_var_threshold:
                        scale = self.max_var_threshold / v
                        w_orig *= scale
                        v = self.calculate_var([r * scale for r in hist])
                    var_values.append(v)
                else:
                    var_values.append(np.nan)
                scaling_factors.append(scale)

                # gross exposure
                gross_exposures.append(np.abs(w_orig).sum())

                # turnover cost
                if weights_history:
                    turnover = np.abs(w_orig - weights_history[-1]).sum()
                    cost = self.transaction_cost_rate * turnover
                else:
                    cost = 0.0
                transaction_costs.append(cost)
                last_cost = cost

                # store new weights
                weights_history.append(w_orig.copy())
            else:
                # non-rebalance day: carry forward last metrics
                var_values.append(var_values[-1] if var_values else np.nan)
                scaling_factors.append(scaling_factors[-1] if scaling_factors else 1.0)
                gross_exposures.append(np.abs(w_orig).sum())
                transaction_costs.append(0.0)               # no cost if no turnover
                weights_history.append(w_orig.copy())       # same weights

            # compute return at t using current w_orig
            ret = np.dot(X_values[t], w_orig) - last_cost
            replica_returns.append(ret)
            target_dates.append(dates[t])

        # assemble pandas and metrics
        replica_series  = pd.Series(replica_returns, index=target_dates)
        aligned_target  = y_series.loc[replica_series.index]
        cum_tgt         = (1 + aligned_target).cumprod()
        cum_rep         = (1 + replica_series).cumprod()

        # performance numbers
        ann = annual_fac
        mu_r = replica_series.mean() * ann
        mu_t = aligned_target.mean() * ann
        vol_r = replica_series.std() * np.sqrt(ann)
        vol_t = aligned_target.std() * np.sqrt(ann)
        sharpe_r = mu_r / vol_r if vol_r else 0
        sharpe_t = mu_t / vol_t if vol_t else 0
        te = (replica_series - aligned_target).std() * np.sqrt(ann)
        ir = ((replica_series.mean()*ann - aligned_target.mean()*ann) / te
            if te else 0)
        corr = replica_series.corr(aligned_target)
        dd_r = (1 - cum_rep / cum_rep.cummax()).max()
        dd_t = (1 - cum_tgt / cum_tgt.cummax()).max()

        self._run_name = "ElasticNetNorm"
        return {
            'l1_ratio': l1_ratio,
            'rolling_window': rolling_window,
            'alpha': alpha,
            'rebalancing_window': rebalancing_window,
            'replica_return': mu_r,
            'target_return': mu_t,
            'replica_vol': vol_r,
            'target_vol': vol_t,
            'replica_sharpe': sharpe_r,
            'target_sharpe': sharpe_t,
            'tracking_error': te,
            'information_ratio': ir,
            'correlation': corr,
            'max_drawdown': float(dd_r),
            'target_max_drawdown': float(dd_t),
            'avg_gross_exposure': float(np.mean(gross_exposures)),
            'avg_var': float(np.nanmean(var_values)),
            'replica_returns': replica_series,
            'aligned_target': aligned_target,
            'cumulative_replica': cum_rep,
            'cumulative_target': cum_tgt,
            'gross_exposures': gross_exposures,
            'var_values': var_values,
            'scaling_factors': scaling_factors,
            'weights_history': weights_history,
            'transaction_costs': transaction_costs
        }

    #################################################################################################################################
    #################################################################################################################################
    # -----------------------------------------------------------------------------
    # 2.3 Kalman Filter
    # -----------------------------------------------------------------------------

    def run_kalman_filter_model(
        self,
        rolling_window: int,
        rebalancing_window: int,
    ) -> Dict[str, Any]:
        """
        Time‐varying weights via KalmanFilter, with initial Ridge estimate.
        On non‐rebalance days, holds the last weight vector constant.
        """
        # 1) pull series & arrays
        X_df = self.underlyings_returns
        y_ser = self.target_returns.iloc[:, 0]
        dates = y_ser.index.to_list()
        X = X_df.values
        y = y_ser.values

        tc = self.transaction_cost_rate
        max_var = self.max_var_threshold

        # containers
        weights_history   = []
        replica_returns   = []
        target_dates      = []
        gross_exposures   = []
        transaction_costs = []
        var_values        = []
        scaling_factors   = []

        # 2) initial weights via Ridge
        X0, y0 = X[:rolling_window], y[:rolling_window]
        init_ridge = Ridge(alpha=1.0, fit_intercept=False).fit(X0, y0)
        current_state = init_ridge.coef_.copy()
        current_cov   = np.eye(len(current_state)) * 0.1

        # 3) set up KalmanFilter
        A = np.eye(len(current_state))
        Q = np.eye(len(current_state)) * 0.01
        resid = y0 - X0 @ current_state
        R = np.var(resid)
        kf = KalmanFilter(
            transition_matrices=A,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=current_state,
            initial_state_covariance=current_cov
        )

        last_cost = 0.0

        # 4) rolling & rebalance loop
        for t in range(rolling_window, len(X)):
            # rebalance check
            if (t - rolling_window) % rebalancing_window == 0:
                # update observation matrix & perform filter update
                kf.observation_matrices = X[t].reshape(1, -1)
                obs = np.array([y[t]])
                current_state, current_cov = kf.filter_update(
                    filtered_state_mean=current_state,
                    filtered_state_covariance=current_cov,
                    observation=obs
                )
                w = current_state.copy()

                # VaR scaling
                scale = 1.0
                if len(replica_returns) >= int(rolling_window/4):
                    hist = [
                        np.dot(X[rolling_window + j], weights_history[-1])
                        for j in range(min(len(replica_returns), rolling_window))
                    ]
                    v = self.calculate_var(hist)
                    if v > max_var:
                        scale = max_var / v
                        w *= scale
                        v = self.calculate_var([r * scale for r in hist])
                    var_values.append(v)
                else:
                    var_values.append(np.nan)
                scaling_factors.append(scale)

                # transaction cost
                if weights_history:
                    turn = np.abs(w - weights_history[-1]).sum()
                    cost = tc * turn
                else:
                    cost = 0.0
                transaction_costs.append(cost)
                last_cost = cost

                weights_history.append(w.copy())
            else:
                # hold weights, no cost, carry forward metrics
                w = weights_history[-1].copy()
                var_values.append(var_values[-1])
                scaling_factors.append(scaling_factors[-1])
                transaction_costs.append(0.0)
                weights_history.append(w.copy())

            # gross exposure
            gross_exposures.append(np.abs(w).sum())

            # next‐period return
            ret = float(X[t] @ w - last_cost)
            replica_returns.append(ret)
            target_dates.append(dates[t])

        # 5) assemble pandas & metrics
        rep_ser = pd.Series(replica_returns, index=target_dates)
        tgt_ser = y_ser.loc[rep_ser.index]
        cum_rep = (1 + rep_ser).cumprod()
        cum_tgt = (1 + tgt_ser).cumprod()

        ann = self.annual_factor
        mu_r = rep_ser.mean() * ann
        mu_t = tgt_ser.mean() * ann
        vol_r = rep_ser.std() * np.sqrt(ann)
        vol_t = tgt_ser.std() * np.sqrt(ann)
        sharpe_r = mu_r / vol_r if vol_r else 0
        sharpe_t = mu_t / vol_t if vol_t else 0
        te = (rep_ser - tgt_ser).std() * np.sqrt(ann)
        ir = ((mu_r - mu_t) / te) if te else 0
        corr = rep_ser.corr(tgt_ser)
        dd_r = (1 - cum_rep / cum_rep.cummax()).max()
        dd_t = (1 - cum_tgt / cum_tgt.cummax()).max()

        self._run_name = "KalmanFilter"
        return {
            'replica_return': mu_r,
            'target_return': mu_t,
            'replica_vol': vol_r,
            'target_vol': vol_t,
            'replica_sharpe': sharpe_r,
            'target_sharpe': sharpe_t,
            'tracking_error': te,
            'information_ratio': ir,
            'correlation': corr,
            'max_drawdown': float(dd_r),
            'target_max_drawdown': float(dd_t),
            'gross_exposures': gross_exposures,
            'replica_returns': rep_ser,
            'aligned_target': tgt_ser,
            'cumulative_replica': cum_rep,
            'cumulative_target': cum_tgt,
            'weights_history': weights_history,
            'transaction_costs': transaction_costs,
            'var_values': var_values,
            'scaling_factors': scaling_factors,
            'rolling_window': rolling_window,
            'rebalancing_window': rebalancing_window,
            'transaction_cost_rate': tc,
            'avg_gross_exposure': float(np.mean(gross_exposures)),
            'avg_var': float(np.nanmean(var_values))
        }

    #################################################################################################################################
    #################################################################################################################################
    # -----------------------------------------------------------------------------
    # 2.4 Extended Kalman Filter (placeholder)
    # -----------------------------------------------------------------------------
    def _optuna_objective_KFE(self, trial: optuna.Trial) -> float:
        """
        Optuna objective for the Ensemble Kalman Filter engine:
        maximizes information ratio by minimizing its negative.
        """
        # 1) suggest hyper-parameters
        ensemble_size        = trial.suggest_int("ensemble_size", 10, 50)
        process_noise_scale  = trial.suggest_float("process_noise_scale", 1e-3, 1e-1, log=True)
        rolling_window       = trial.suggest_categorical("rolling_window", [52, 104, 156, 208])
        rebalancing_window   = trial.suggest_categorical("rebalancing_window", [1, 2, 4])

        # 2) run your ensemble KF model
        result = self.run_ensemble_kalman_filter_model(
            rolling_window       = rolling_window,
            rebalancing_window   = rebalancing_window,
            ensemble_size        = ensemble_size,
            process_noise_scale  = process_noise_scale
        )

        # 3) get IR and report for pruning
        ir = result["information_ratio"]
        trial.report(-ir, step=0)
        if trial.should_prune():
            raise TrialPruned()

        # 4) Optuna minimizes, so return negative IR to maximize it
        return -ir

    def run_ensemble_kalman_filter_model(
        self,
        rolling_window: int,
        rebalancing_window: int,
        ensemble_size: int = 20,
        process_noise_scale: float = 0.01
    ) -> Dict[str, Any]:
        """
        Ensemble Kalman filter replication:
         - initial Ridge on first rolling_window
         - ensemble of weight vectors evolving by random‐walk process noise
         - observation update each rebalance
         - VaR‐scaling, transaction costs
         - holds weights constant on non‐rebalance days
        """
        # 1) pull arrays & dates
        X = self.underlyings_returns.values
        y_ser = self.target_returns.iloc[:, 0]
        y = y_ser.values
        dates = y_ser.index.to_list()

        tc_rate   = self.transaction_cost_rate
        max_var   = self.max_var_threshold
        hist_min  = int(rolling_window / 4)

        # 2) containers
        weights_history   = []
        replica_returns   = []
        target_dates      = []
        gross_exposures   = []
        net_exposure      = []
        transaction_costs = []
        var_values        = []
        scaling_factors   = []

        n_assets = X.shape[1]

        # 3) initial Ridge fit
        X0, y0 = X[:rolling_window], y[:rolling_window]
        ridge = Ridge(alpha=1.0, fit_intercept=False).fit(X0, y0)
        current_state = ridge.coef_.copy()
        current_cov   = np.eye(n_assets) * 0.1

        # 4) set up KalmanFilter skeleton
        A = np.eye(n_assets)
        Q = np.eye(n_assets) * process_noise_scale
        resid = y0 - X0 @ current_state
        R = np.var(resid) + 1e-6
        kf = KalmanFilter(
            transition_matrices=A,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=current_state,
            initial_state_covariance=current_cov
        )

        last_cost = 0.0

        # 5) rolling & rebalance loop
        for t in range(rolling_window, len(X)):
            # rebalance day?
            if (t - rolling_window) % rebalancing_window == 0:
                obs_matrix = X[t].reshape(1, -1)
                kf.observation_matrices = obs_matrix
                obs = np.array([y[t]])
                current_state, current_cov = kf.filter_update(
                    filtered_state_mean=current_state,
                    filtered_state_covariance=current_cov,
                    observation=obs
                )
                w = current_state.copy()

                # VaR scaling
                scale = 1.0
                if len(replica_returns) >= hist_min:
                    hist = [
                        np.dot(X[rolling_window + j], weights_history[-1])
                        for j in range(min(len(replica_returns), rolling_window))
                    ]
                    v = self.calculate_var(hist)
                    if v > max_var:
                        scale = max_var / v
                        w *= scale
                        v = self.calculate_var([r * scale for r in hist])
                    var_values.append(v)
                else:
                    var_values.append(np.nan)
                scaling_factors.append(scale)

                # transaction cost
                if weights_history:
                    turn = np.abs(w - weights_history[-1]).sum()
                    cost = tc_rate * turn
                else:
                    cost = 0.0
                transaction_costs.append(cost)
                last_cost = cost

                weights_history.append(w.copy())
            else:
                # hold previous weights, no cost, carry metrics forward
                w = weights_history[-1].copy()
                var_values.append(var_values[-1])
                scaling_factors.append(scaling_factors[-1])
                transaction_costs.append(0.0)
                weights_history.append(w.copy())

            # gross exposure
            gross_exposures.append(np.abs(w).sum())
            net_exposure.append((w).sum())

            # one-step-ahead return
            ret = float(X[t] @ w - last_cost)
            replica_returns.append(ret)
            target_dates.append(dates[t])

        # 6) assemble results
        rep_ser = pd.Series(replica_returns, index=target_dates)
        tgt_ser = y_ser.loc[rep_ser.index]
        cum_rep = (1 + rep_ser).cumprod()
        cum_tgt = (1 + tgt_ser).cumprod()

        ann = self.annual_factor
        mu_r = rep_ser.mean() * ann
        mu_t = tgt_ser.mean() * ann
        vol_r = rep_ser.std() * np.sqrt(ann)
        vol_t = tgt_ser.std() * np.sqrt(ann)
        sharpe_r = mu_r / vol_r if vol_r else 0
        sharpe_t = mu_t / vol_t if vol_t else 0
        te = (rep_ser - tgt_ser).std() * np.sqrt(ann)
        ir = (mu_r - mu_t) / te if te else 0
        corr = rep_ser.corr(tgt_ser)
        dd_r = (1 - cum_rep / cum_rep.cummax()).max()
        dd_t = (1 - cum_tgt / cum_tgt.cummax()).max()

        self._run_name = "EnsembleKF"
        return {
            'replica_return': mu_r,
            'target_return': mu_t,
            'replica_vol': vol_r,
            'target_vol': vol_t,
            'replica_sharpe': sharpe_r,
            'target_sharpe': sharpe_t,
            'tracking_error': te,
            'information_ratio': ir,
            'correlation': corr,
            'max_drawdown': float(dd_r),
            'target_max_drawdown': float(dd_t),
            'gross_exposures': gross_exposures,
            'replica_returns': rep_ser,
            'aligned_target': tgt_ser,
            'cumulative_replica': cum_rep,
            'cumulative_target': cum_tgt,
            'weights_history': weights_history,
            'transaction_costs': transaction_costs,
            'var_values': var_values,
            'scaling_factors': scaling_factors,
            'rolling_window': rolling_window,
            'rebalancing_window': rebalancing_window,
            'ensemble_size': ensemble_size,
            'process_noise_scale': process_noise_scale,
            'avg_gross_exposure': float(np.mean(gross_exposures)),
            'avg_var': float(np.nanmean(var_values)),
            'net_exposure': net_exposure
        }






    ##############################################################################################################################
    ##############################################################################################################################
    # -----------------------------------------------------------------------------
    # 3. Results & plotting
    # -----------------------------------------------------------------------------
    
    # Replica metrics for each model
    def extract_replica_metrics(slef, result, label) -> Dict[str, Any]:
        """
            Compute “global” performance stats on the target series
            starting from `window` onward.

            Parameters
            ----------
            window : int
                Number of initial observations to skip.

        """
        return {
            'Model': label,
            'Annualized return': f"{result['replica_return']*100:.2f}%",
            'Annualized volatility': f"{result['replica_vol']*100:.2f}%",
            'Sharpe ratio': f"{result['replica_sharpe']:.2f}",
            'Max Drawdown': f"{result['max_drawdown']*100:.2f}%",
            'Tracking Error': f"{result['tracking_error']*100:.2f}%",
            'Information ratio': f"{result['information_ratio']:.2f}",
            'Correlation': f"{result['correlation']:.4f}",
            'Average gross exposure': f"{result['avg_gross_exposure']:.4f}",
            'Average VaR (1%, 1M)': f"{result['avg_var']*100:.2f}%" if not np.isnan(result['avg_var']) else "N/A",
            'Transaction cost (bp)': f"{np.sum(result['transaction_costs'])* 10000:.2f}",
        }

    # Target metrics based on globally computed values
    def extract_target_metrics(self, window) -> Dict[str, Any]:

        target_ret = self.target_returns.iloc[window:, 0].dropna()
        annual_factor = self.annual_factor

        global_target_return = target_ret.mean() * annual_factor
        global_target_vol = target_ret.std() * np.sqrt(annual_factor)
        global_target_sharpe = global_target_return / global_target_vol
        cumulative_target = (1 + target_ret).cumprod()
        global_max_drawdown = (1 - cumulative_target / cumulative_target.cummax()).max()
        global_max_drawdown_value = cumulative_target.max() - cumulative_target.min()

        return {
            'Model': 'Target Portfolio',
            'Annualized return': f"{float(global_target_return)*100:.2f}%",
            'Annualized volatility': f"{float(global_target_vol)*100:.2f}%",
            'Sharpe ratio': f"{float(global_target_sharpe):.2f}",
            'Max Drawdown': f"{float(global_max_drawdown_value)*100:.2f}%",
            'Tracking Error': "N/A",
            'Information ratio': "N/A",
            'Correlation': "N/A",
            'Average gross exposure': "N/A",
            'Average VaR (1%, 1M)': "N/A"
        }
    

    def plot_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        figsize: Tuple[int, int] = (16, 8)
    ) -> None:
        """
        Plot normalized cumulative returns for multiple replication strategies
        alongside the target index.

        Parameters
        ----------
        results : dict
            Mapping from method name (str) to that method’s result dict, which must
            include keys 'replica_returns' (pd.Series) and 'aligned_target' (pd.Series).
        figsize : tuple, default (16,8)
            Figure size.
        """
        # 1) collect all replica return series
        replicas = {name: cfg['replica_returns'] for name, cfg in results.items()}
        # 2) find common dates
        common_idx = None
        for ser in replicas.values():
            common_idx = ser.index if common_idx is None else common_idx.intersection(ser.index)
        # 3) reindex & compute normalized cum-returns
        cum_reps = {}
        for name, ser in replicas.items():
            ser_aligned = ser.loc[common_idx]
            cum = (1 + ser_aligned).cumprod()
            cum_reps[name] = cum / cum.iloc[0]

        # 4) get target (they all share the same aligned_target)
        target = next(iter(results.values()))['aligned_target'].loc[common_idx]
        cum_tgt = (1 + target).cumprod()
        cum_tgt = cum_tgt / cum_tgt.iloc[0]

        # 5) plot
        plt.figure(figsize=figsize)
        for name, cum in cum_reps.items():
            plt.plot(cum.index, cum.values, label=name)
        plt.plot(cum_tgt.index, cum_tgt.values,
                 color='black', linestyle='--', label='Target index')
        plt.title('Cumulative returns: target vs replica methods (normalized)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative return (normalized)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

############################################################################################################
##############################################################################################################
##############################################################################################################
# -# -----------------------------------------------------------------------------
# DISCRETE REPLICATION METHODS
# -----------------------------------------------------------------------------
   
    def compute_discrete_transaction_costs(
        self,
        result: Dict[str, Any],
        portfolio_value: float,
        prices_df: pd.DataFrame,
        transaction_cost_rate: float,
        rebalancing_freq: int = 52
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Discrete allocation via signed‐Hamilton:
        p*_i = w_i * AUM / price_i
        f_i  = floor(p*_i)   (floor toward -inf)
        rem_i= p*_i - f_i
        cash = AUM - sum_long(f_i*price) + sum_short(|f_i|*price)
        then in descending rem_i order:
            if cash>=price_i:
                f_i +=1
                cash -= price_i
        """
        wh  = result["weights_history"]
        dates = result["replica_returns"].index
        tickers = prices_df.columns.tolist()

        positions_list = []
        prev_pos = np.zeros(len(tickers), dtype=int)
        initial_positions = wh[0]
        initial_cost = np.abs(initial_positions).sum() * transaction_cost_rate
        total_cost = initial_cost
 

        for date, w in zip(dates, wh):
            prices = prices_df.loc[pd.to_datetime(date)].values
            # 1) exact contracts
            exact = (w * portfolio_value) / prices
            # 2) floor toward -inf
            floored = np.floor(exact).astype(int)
            # 3) compute cash after floored longs/shorts
            longs  = np.maximum(floored, 0)
            shorts = np.maximum(-floored, 0)
            cash = portfolio_value - np.dot(longs, prices) + np.dot(shorts, prices)
            # 4) remainders
            rem = exact - floored
            # 5) allocate extras in descending remainder order
            order = np.argsort(-rem)
            extra = np.zeros_like(floored)
            for i in order:
                if cash >= prices[i]:
                    extra[i] = 1
                    cash   -= prices[i]
                else:
                    break
            # 6) combine
            pos = floored + extra
            # note: floored already carries the correct sign for negatives

            # 7) transaction cost on notional turnover
            trades = np.abs(pos - prev_pos)
            # long trades consume cash, short trades free cash but cost to trade is always paid
            cost_t = np.dot(trades, prices) * transaction_cost_rate
            total_cost += cost_t

            positions_list.append(pos)
            prev_pos = pos.copy()

        # build DataFrame
        positions_history = pd.DataFrame(positions_list, index=dates, columns=tickers)
        years = len(positions_list) / rebalancing_freq
        annualized_cost = total_cost / years if years>0 else np.nan
        return total_cost, annualized_cost, positions_history



    def compute_cumulative_equity_curve(
            self,
            positions_history: pd.DataFrame,
            prices_df: pd.DataFrame,
            portfolio_value: float
        ) -> pd.Series:
            """
            Given discrete futures positions and actual prices, compute the cumulative
            equity curve by marking‐to‐market between rebalances.

            Parameters
            ----------
            positions_history : pd.DataFrame
                Index = rebalance dates; columns = tickers; values = integer contract counts.
            prices_df : pd.DataFrame
                Full price history with the same tickers as columns and dates covering
                at least all rebalance dates.
            portfolio_value : float
                Starting capital.

            Returns
            -------
            pd.Series
                Cumulative equity indexed by the rebalance dates.
            """
            # Ensure datetime‐index alignment
            rebal_dates = pd.to_datetime(positions_history.index)
            equity = portfolio_value
            curve = [equity]

            # for each interval [t, t+1), mark‐to‐market
            for today, tomorrow in zip(rebal_dates[:-1], rebal_dates[1:]):
                # load positions and prices
                pos = positions_history.loc[today]
                p0  = prices_df.loc[today,    pos.index]
                p1  = prices_df.loc[tomorrow, pos.index]

                # P&L = Σ contracts * (price_next – price_today)
                profit = np.dot(pos.values, (p1.values - p0.values))
                equity += profit
                curve.append(equity)

            return pd.Series(curve, index=rebal_dates)










































