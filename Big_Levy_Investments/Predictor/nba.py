import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PortfolioRecommender:
    """
    A class that, given:
      - a trained Predictor instance (pred), with attributes:
          • pred.transformed_df
          • pred.X_test
      - a portfolios DataFrame with columns ['name_ptf', 'Risk', 'Income', 'Accumulation'],
    can generate personalized recommendations for either “Income” or “Accumulation” products
    based on each client’s risk propensity.
    """

    def __init__(self, predictor):
        """
        predictor:    An instance of your Predictor class (already trained & with X_test populated).
        portfolios_df: A DataFrame with columns:
                       • 'name_ptf' (product identifier)
                       • 'Risk' (numeric risk score)
                       • 'Income' (0/1 flag)
                       • 'Accumulation' (0/1 flag)
        """

        ptfs_dict = {
            'ETF_bond':      1/7,
            'ETF_equity':    3/7,
            'ETF_life':      2/7,
            'active_bond':   2/7,
            'active_equity': 4/7,
            'active_life':   3/7,
        }

        rows = []
        for name_ptf, risk in ptfs_dict.items():
            income_flag = 0
            accum_flag  = 0

            if name_ptf in ['ETF_bond', 'ETF_equity']:
                income_flag = 1
                accum_flag  = 1
            elif name_ptf == 'ETF_life':
                income_flag = 0
                accum_flag  = 1
            elif name_ptf in ['active_bond', 'active_equity']:
                income_flag = 1
                accum_flag  = 0
            elif name_ptf == 'active_life':
                income_flag = 0
                accum_flag  = 1

            rows.append({
                'name_ptf':     name_ptf,
                'Risk':         risk,
                'Income':       income_flag,
                'Accumulation': accum_flag
            })

        portfolios_df = pd.DataFrame(rows, columns=['name_ptf', 'Risk', 'Income', 'Accumulation'])

        self.pred = predictor
        self.portfolios = portfolios_df.copy().reset_index(drop=True)

        # Validate that the required columns exist in portfolios_df
        required_cols = {'name_ptf', 'Risk', 'Income', 'Accumulation'}
        missing = required_cols - set(self.portfolios.columns)
        if missing:
            raise ValueError(f"Portfolios DataFrame is missing required columns: {missing}")

    def recommend_for_target(self, y_pred, target_flag):
        """
        Generate recommendations for clients where y_pred == 1, using portfolios
        filtered by target_flag ('Income' or 'Accumulation').

        Parameters:
        -----------
        y_pred : array-like, shape (n_test_samples,)
            Binary predictions (0/1) for the desired target (e.g., Income or Accumulation).
        target_flag : str
            Either 'Income' or 'Accumulation'. We will recommend only those products
            where portfolios_df[target_flag] == 1.

        Returns:
        --------
        nba_df : pd.DataFrame with columns
            ['ClientID', 'RecommendedProductID', 'ClientRiskPropensity', 'ProductRiskLevel'].

        Additionally, prints:
          • Top 10 rows of the recommendation sheet
          • Summary statistics (how many received valid recommendations)
        And displays:
          • A scatter plot of client risk vs. recommended product risk
          • A bar plot of frequency distribution of recommended products
          • A histogram of risk distribution among recommended clients
        """
        if target_flag not in ('Income', 'Accumulation'):
            raise ValueError("target_flag must be either 'Income' or 'Accumulation'")

        # 1) Find the test‐set indices where y_pred == 1
        client_indices = np.where(y_pred == 1)[0]

        # 2) Map those indices back to “original” client IDs in transformed_df
        target_client_ids = (
            self.pred.transformed_df
                .iloc[self.pred.X_test.index[client_indices]]
                .index
                .values
        )

        # 3) For those same test‐rows, get each client’s RiskPropensity from X_test
        target_client_risk = (
            self.pred.X_test.iloc[client_indices]['RiskPropensity'].values
        )

        # 4) Filter the portfolios DataFrame to only the products flagged by target_flag
        filtered_products = self.portfolios[self.portfolios[target_flag] == 1].reset_index(drop=True)
        product_names = filtered_products['name_ptf'].values
        product_risks = filtered_products['Risk'].values

        # 5) Identify the minimum risk among these filtered products
        min_risk = product_risks.min()

        # 6) For each target client, pick the highest‐possible product risk < client’s risk appetite
        nba_id_product = []
        recommended_risk_level = []

        for client_r in target_client_risk:
            # If client’s risk > min_risk, there might be a match
            if client_r > min_risk:
                mask = (product_risks < client_r)
                if mask.any():
                    # Find the maximum risk among all products < client’s risk appetite
                    chosen_risk = product_risks[mask].max()

                    # Extract the single row in filtered_products with Risk == chosen_risk
                    chosen_row = filtered_products[filtered_products['Risk'] == chosen_risk].iloc[0]
                    chosen_name = chosen_row['name_ptf']

                    nba_id_product.append(chosen_name)
                    recommended_risk_level.append(chosen_risk)
                else:
                    # No product has risk < client_r
                    nba_id_product.append(0)
                    recommended_risk_level.append(0)
            else:
                # Client cannot tolerate even the lowest-risk product
                nba_id_product.append(0)
                recommended_risk_level.append(0)

        # 7) Build the recommendation DataFrame
        nba_df = pd.DataFrame({
            'ClientID'             : target_client_ids,
            'RecommendedProductID' : nba_id_product,
            'ClientRiskPropensity' : target_client_risk,
            'ProductRiskLevel'     : recommended_risk_level
        })

        # 8) Display Top 10 recommendations
        print(f"\nTop 10 personalized recommendations ({target_flag}):")
        print(nba_df.head(10))

        # 9) Compute summary statistics
        total_clients = len(nba_df)
        # Since we used “0” as the sentinel for “no recommendation,” we check != 0
        clients_with_recs = (nba_df['RecommendedProductID'] != 0).sum()
        perc_with_recs = (clients_with_recs / total_clients) * 100

        print(f"\nRecommendation statistics ({target_flag}):")
        print(f"  Total customers analyzed:                 {total_clients}")
        print(
            f"  Customers with valid recommendations:     {clients_with_recs} "
            f"({perc_with_recs:.2f}%)"
        )
        print(f"  Customers without suitable recommendations: {total_clients - clients_with_recs}")

        # 10) Scatter plot: client risk vs. product risk
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            nba_df['ClientRiskPropensity'],
            nba_df['ProductRiskLevel'],
            c=np.arange(total_clients),
            cmap='viridis',
            alpha=0.7,
            s=80
        )
        plt.title(f"Suitability: Client Risk vs. Product Risk ({target_flag})")
        plt.xlabel("Client Risk Propensity")
        plt.ylabel("Recommended Product Risk Level")
        plt.colorbar(scatter, label="Client (index in recommendation list)")

        # Add a reference diagonal line (perfect 1:1 match)
        max_val = max(nba_df['ClientRiskPropensity'].max(), nba_df['ProductRiskLevel'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label="Ideal Risk Match")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 11) Bar plot: frequency distribution of recommended products (excluding “0”)
        if clients_with_recs > 0:
            plt.figure(figsize=(10, 5))

            rec_counts = (
                nba_df.loc[nba_df['RecommendedProductID'] != 0, 'RecommendedProductID']
                    .value_counts()
                    .sort_index()
            )

            plt.bar(
                rec_counts.index.astype(str),
                rec_counts.values,
                color='skyblue'
            )
            plt.title(f"Frequency Distribution of Recommended {target_flag} Products")
            plt.xlabel("Product ID (name_ptf)")
            plt.ylabel("Number of Recommendations")
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()

            # 12) Print the top 3 most frequently recommended products (if at least 1 exists)
            if len(rec_counts) > 0:
                top3 = rec_counts.nlargest(3).index
                print(f"\nMost recommended {target_flag} products (Top 3):")
                for pid in top3:
                    row = filtered_products[filtered_products['name_ptf'] == pid].iloc[0]
                    print(
                        f"  • Product '{pid}': Risk = {row['Risk']}  → "
                        f"recommended to {rec_counts.loc[pid]} clients"
                    )
        else:
            print(f"\nNo {target_flag} products were recommended to these clients.")

        # 13) NEW: Histogram of risk distribution among clients with a recommended product
        # Filter only clients with a valid recommendation:
        recommended_clients = nba_df[nba_df['RecommendedProductID'] != 0]
        if not recommended_clients.empty:
            plt.figure(figsize=(10, 6))
            plt.hist(
                recommended_clients['ClientRiskPropensity'],
                bins=10,
                alpha=0.7
            )
            plt.title(f"Risk Distribution Among Clients with a Recommended {target_flag} Product")
            plt.xlabel("Risk Propensity")
            plt.ylabel("Frequency")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print(f"\nNo risk‐distribution histogram since no {target_flag} recommendations were made.")

        # Return the recommendation DataFrame
        return nba_df