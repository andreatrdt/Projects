import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
import umap.umap_ as umap

class DimensionalityVisualizer:
    """
    A class for visualizing model classification results (normal vs anomaly)
    using various dimensionality reduction techniques including:

    - PCA (Principal Component Analysis)
    - UMAP (Uniform Manifold Approximation and Projection)
    - t-SNE (t-distributed Stochastic Neighbor Embedding)
    - Spectral Embedding

    Each method provides both 2D and 3D visualization using Matplotlib and Plotly.
    Classification outcomes are color-coded as:
    - Gray: True Negatives
    - Black: True Positives
    - Red: False Positives
    - Blue: False Negatives
    """

    def __init__(self):
        self.colors = {'True Negative': 'gray', 'True Positive': 'black',
                       'False Positive': 'red', 'False Negative': 'blue'}
        self.sizes = {'True Negative': 30, 'True Positive': 40,
                      'False Positive': 80, 'False Negative': 80}
        self.alphas = {'True Negative': 0.3, 'True Positive': 0.5,
                       'False Positive': 0.8, 'False Negative': 0.8}

    def _build_df(self, X_proj, y_true, y_pred, dim_names):
        df = pd.DataFrame(X_proj, columns=dim_names)
        df['Actual'] = y_true
        df['Predicted'] = y_pred
        df['Category'] = 'Unknown'
        df.loc[(y_true == 0) & (y_pred == 0), 'Category'] = 'True Negative'
        df.loc[(y_true == 0) & (y_pred == 1), 'Category'] = 'False Positive'
        df.loc[(y_true == 1) & (y_pred == 0), 'Category'] = 'False Negative'
        df.loc[(y_true == 1) & (y_pred == 1), 'Category'] = 'True Positive'
        return df

    def visualize_pca(self, X, y_true, y_pred, model_name, num_components=3):
        """
        Perform PCA projection in 2D or 3D and visualize classification results
        using both Matplotlib and Plotly.
        """
        pca = PCA(n_components=num_components)
        X_pca = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_
        total_variance = explained_variance.sum()

        if num_components == 2:
            df = self._build_df(X_pca, y_true, y_pred, ['PC1', 'PC2'])
            plt.figure(figsize=(12, 8))
            for category, group in df.groupby('Category'):
                plt.scatter(group['PC1'], group['PC2'],
                            color=self.colors[category],
                            alpha=self.alphas[category],
                            s=self.sizes[category],
                            label=f"{category} ({len(group)})")
            plt.title(f'PCA 2D - {model_name}\nExplained Variance: {total_variance:.2%}')
            plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
            plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        elif num_components == 3:
            df = self._build_df(X_pca, y_true, y_pred, ['PC1', 'PC2', 'PC3'])
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            for cat in df['Category'].unique():
                subset = df[df['Category'] == cat]
                ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'],
                           c=self.colors.get(cat, 'orange'), label=f"{cat} ({len(subset)})", alpha=0.6)
            ax.set_title(f'3D PCA - {model_name}\nExplained Variance: {total_variance:.2%}')
            ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
            ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
            ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
            ax.legend()
            plt.tight_layout()
            plt.show()

            fig_plotly = px.scatter_3d(df, x='PC1', y='PC2', z='PC3',
                                        color='Category', color_discrete_map=self.colors,
                                        opacity=0.6,
                                        title=f'Plotly 3D PCA - {model_name}')
            fig_plotly.update_layout(
                scene=dict(
                    xaxis_title=f'PC1 ({explained_variance[0]:.2%})',
                    yaxis_title=f'PC2 ({explained_variance[1]:.2%})',
                    zaxis_title=f'PC3 ({explained_variance[2]:.2%})'))
            fig_plotly.update_traces(marker=dict(size=2))
            fig_plotly.show()

    def visualize_umap(self, X, y_true, y_pred, model_name="Model", num_components=2,
                       n_neighbors=15, min_dist=0.1, metric='euclidean'):
        """
        Apply UMAP and visualize the projection using Matplotlib and Plotly.
        UMAP captures both local and global structure, and supports 2D/3D views.
        """
        reducer = umap.UMAP(n_components=num_components, n_neighbors=n_neighbors,
                            min_dist=min_dist, metric=metric, random_state=42)
        X_umap = reducer.fit_transform(X)
        col_names = [f"UMAP{i+1}" for i in range(num_components)]
        df = self._build_df(X_umap, y_true, y_pred, col_names)

        if num_components == 2:
            plt.figure(figsize=(14, 10))
            for category, group in df.groupby('Category'):
                plt.scatter(group[col_names[0]], group[col_names[1]],
                            c=self.colors[category], s=self.sizes[category],
                            alpha=self.alphas[category], label=f"{category} ({len(group)})")
            plt.title(f'UMAP 2D - {model_name}')
            plt.xlabel(col_names[0])
            plt.ylabel(col_names[1])
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        elif num_components == 3:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            for cat in df['Category'].unique():
                subset = df[df['Category'] == cat]
                ax.scatter(subset[col_names[0]], subset[col_names[1]], subset[col_names[2]],
                           c=self.colors.get(cat, 'orange'), label=f"{cat} ({len(subset)})", alpha=0.6)
            ax.set_title(f'UMAP 3D - {model_name}')
            ax.set_xlabel(col_names[0])
            ax.set_ylabel(col_names[1])
            ax.set_zlabel(col_names[2])
            ax.legend()
            plt.tight_layout()
            plt.show()

            fig_plotly = px.scatter_3d(df, x=col_names[0], y=col_names[1], z=col_names[2],
                                        color='Category', color_discrete_map=self.colors,
                                        opacity=0.6, title=f'Plotly UMAP 3D - {model_name}')
            fig_plotly.update_traces(marker=dict(size=2))
            fig_plotly.show()

    def visualize_tsne(self, X, y_true, y_pred, model_name="Model", num_components=2,
                       perplexity=30, learning_rate=200, n_iter=1000):
        """
        Apply t-SNE and visualize the projection using Matplotlib and Plotly.
        t-SNE excels at preserving local structure and cluster formation.
        """
        tsne = TSNE(n_components=num_components, perplexity=perplexity,
                    learning_rate=learning_rate, n_iter=n_iter, random_state=42)
        X_tsne = tsne.fit_transform(X)
        col_names = [f"TSNE{i+1}" for i in range(num_components)]
        df = self._build_df(X_tsne, y_true, y_pred, col_names)

        if num_components == 2:
            plt.figure(figsize=(14, 10))
            for category, group in df.groupby('Category'):
                plt.scatter(group[col_names[0]], group[col_names[1]],
                            c=self.colors[category], s=self.sizes[category],
                            alpha=self.alphas[category], label=f"{category} ({len(group)})")
            plt.title(f't-SNE 2D - {model_name}')
            plt.xlabel(col_names[0])
            plt.ylabel(col_names[1])
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        elif num_components == 3:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            for cat in df['Category'].unique():
                subset = df[df['Category'] == cat]
                ax.scatter(subset[col_names[0]], subset[col_names[1]], subset[col_names[2]],
                           c=self.colors.get(cat, 'orange'), label=f"{cat} ({len(subset)})", alpha=0.6)
            ax.set_title(f't-SNE 3D - {model_name}')
            ax.set_xlabel(col_names[0])
            ax.set_ylabel(col_names[1])
            ax.set_zlabel(col_names[2])
            ax.legend()
            plt.tight_layout()
            plt.show()

            fig_plotly = px.scatter_3d(df, x=col_names[0], y=col_names[1], z=col_names[2],
                                        color='Category', color_discrete_map=self.colors,
                                        opacity=0.6, title=f'Plotly t-SNE 3D - {model_name}')
            fig_plotly.update_traces(marker=dict(size=2))
            fig_plotly.show()

    def visualize_spectral(self, X, y_true, y_pred, model_name="Model", num_components=2, n_neighbors=10):
        """
        Apply Spectral Embedding and visualize the projection using Matplotlib and Plotly.
        Spectral methods are graph-based and preserve manifold structure.
        """
        embedding = SpectralEmbedding(n_components=num_components, n_neighbors=n_neighbors, random_state=42)
        X_spectral = embedding.fit_transform(X)
        col_names = [f"SE{i+1}" for i in range(num_components)]
        df = self._build_df(X_spectral, y_true, y_pred, col_names)

        if num_components == 2:
            plt.figure(figsize=(14, 10))
            for category, group in df.groupby('Category'):
                plt.scatter(group[col_names[0]], group[col_names[1]],
                            c=self.colors[category], s=self.sizes[category],
                            alpha=self.alphas[category], label=f"{category} ({len(group)})")
            plt.title(f'Spectral 2D - {model_name}')
            plt.xlabel(col_names[0])
            plt.ylabel(col_names[1])
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        elif num_components == 3:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            for cat in df['Category'].unique():
                subset = df[df['Category'] == cat]
                ax.scatter(subset[col_names[0]], subset[col_names[1]], subset[col_names[2]],
                           c=self.colors.get(cat, 'orange'), label=f"{cat} ({len(subset)})", alpha=0.6)
            ax.set_title(f'Spectral 3D - {model_name}')
            ax.set_xlabel(col_names[0])
            ax.set_ylabel(col_names[1])
            ax.set_zlabel(col_names[2])
            ax.legend()
            plt.tight_layout()
            plt.show()

            fig_plotly = px.scatter_3d(df, x=col_names[0], y=col_names[1], z=col_names[2],
                                        color='Category', color_discrete_map=self.colors,
                                        opacity=0.6, title=f'Plotly Spectral 3D - {model_name}')
            fig_plotly.update_traces(marker=dict(size=2))
            fig_plotly.show()


    def compare_dim_reductions(self, X, y_true, y_pred, model_name="Model"):
        """
        Visualize PCA, UMAP, t-SNE, and Spectral Embedding 2D projections side by side.
        The legend is displayed outside the 2x2 grid of plots.
        """
        def create_category_df(X_proj, dim_names):
            df = pd.DataFrame(X_proj, columns=dim_names)
            df['Actual'] = y_true
            df['Predicted'] = y_pred
            df['Category'] = 'Unknown'
            df.loc[(y_true == 0) & (y_pred == 0), 'Category'] = 'True Negative'
            df.loc[(y_true == 0) & (y_pred == 1), 'Category'] = 'False Positive'
            df.loc[(y_true == 1) & (y_pred == 0), 'Category'] = 'False Negative'
            df.loc[(y_true == 1) & (y_pred == 1), 'Category'] = 'True Positive'
            return df

        X_pca = PCA(n_components=2, random_state=42).fit_transform(X)
        X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
        X_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42).fit_transform(X)
        X_spectral = SpectralEmbedding(n_components=2, n_neighbors=10, random_state=42).fit_transform(X)

        df_pca = create_category_df(X_pca, ['Dim1', 'Dim2'])
        df_umap = create_category_df(X_umap, ['Dim1', 'Dim2'])
        df_tsne = create_category_df(X_tsne, ['Dim1', 'Dim2'])
        df_spectral = create_category_df(X_spectral, ['Dim1', 'Dim2'])

        colors = self.colors
        sizes = self.sizes
        alphas = self.alphas

        fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
        titles = ['PCA', 'UMAP', 't-SNE', 'Spectral Embedding']
        dfs = [df_pca, df_umap, df_tsne, df_spectral]
        axes = axes.flatten()

        for ax, df, title in zip(axes, dfs, titles):
            for category, group in df.groupby('Category'):
                ax.scatter(group['Dim1'], group['Dim2'],
                            c=colors[category], s=sizes[category],
                            alpha=alphas[category], label=category)
            ax.set_title(f'{title} - {model_name}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.grid(True, alpha=0.3)

        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)

        plt.suptitle(f'Comparison of PCA, UMAP, t-SNE, and Spectral - {model_name}', fontsize=18)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        plt.show()