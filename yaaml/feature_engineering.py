"""
YAAML Feature Engineering Module
Native feature engineering using sklearn methods
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection


class FeatureEngineering:
    """Native feature engineering with multiple transformation strategies"""

    def __init__(
        self,
        decomposition_methods: list[str] = ["pca"],
        n_components: int | float = 0.95,
        clustering_features: bool = True,
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        scaling_method: str = "standard",
    ):
        """
        Initialize feature engineering pipeline

        Parameters:
        -----------
        decomposition_methods : list
            Methods to use: ['pca', 'ica', 'tsvd', 'grp', 'srp']
        n_components : int or float
            Number of components or variance ratio to retain
        clustering_features : bool
            Whether to add clustering-based features
        polynomial_features : bool
            Whether to create polynomial features
        polynomial_degree : int
            Degree for polynomial features
        scaling_method : str
            Scaling method: 'standard', 'minmax', 'none'
        """
        self.decomposition_methods = decomposition_methods
        self.n_components = n_components
        self.clustering_features = clustering_features
        self.polynomial_features = polynomial_features
        self.polynomial_degree = polynomial_degree
        self.scaling_method = scaling_method

        # Fitted components
        self.scaler = None
        self.decomposers = {}
        self.clusterers = {}
        self.poly_features = None
        self.feature_names = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineering":
        """
        Fit feature engineering transformations

        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series, optional
            Target variable (for supervised feature selection)
        """
        self.feature_names = list(X.columns)

        # Fit scaler
        if self.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            self.scaler = MinMaxScaler()

        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X), columns=X.columns, index=X.index
            )
        else:
            X_scaled = X.copy()

        # Fit decomposition methods
        for method in self.decomposition_methods:
            if method.lower() == "pca":
                decomposer = PCA(n_components=self.n_components, random_state=42)
            elif method.lower() == "ica":
                n_comp = (
                    min(X.shape[1], 10)
                    if isinstance(self.n_components, float)
                    else self.n_components
                )
                decomposer = FastICA(
                    n_components=n_comp, random_state=42, max_iter=1000
                )
            elif method.lower() == "tsvd":
                n_comp = (
                    min(X.shape[1] - 1, 50)
                    if isinstance(self.n_components, float)
                    else self.n_components
                )
                decomposer = TruncatedSVD(n_components=n_comp, random_state=42)
            elif method.lower() == "grp":
                n_comp = (
                    min(X.shape[1], 10)
                    if isinstance(self.n_components, float)
                    else self.n_components
                )
                decomposer = GaussianRandomProjection(
                    n_components=n_comp, random_state=42
                )
            elif method.lower() == "srp":
                n_comp = (
                    min(X.shape[1], 10)
                    if isinstance(self.n_components, float)
                    else self.n_components
                )
                decomposer = SparseRandomProjection(
                    n_components=n_comp, random_state=42
                )
            else:
                continue

            decomposer.fit(X_scaled)
            self.decomposers[method] = decomposer

        # Fit clustering
        if self.clustering_features:
            # Multiple cluster sizes
            cluster_sizes = [3, 5, 8, 10]
            for n_clusters in cluster_sizes:
                if n_clusters < X.shape[0]:  # Ensure we have enough samples
                    clusterer = KMeans(
                        n_clusters=n_clusters, random_state=42, n_init=10
                    )
                    clusterer.fit(X_scaled)
                    self.clusterers[f"kmeans_{n_clusters}"] = clusterer

        # Fit polynomial features
        if (
            self.polynomial_features and X.shape[1] <= 10
        ):  # Limit for computational efficiency
            self.poly_features = PolynomialFeatures(
                degree=self.polynomial_degree, include_bias=False, interaction_only=True
            )
            self.poly_features.fit(X_scaled)

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted transformations

        Parameters:
        -----------
        X : pd.DataFrame
            Features to transform

        Returns:
        --------
        pd.DataFrame
            Transformed features
        """
        if not self.fitted:
            raise ValueError("FeatureEngineering must be fitted before transform")

        # Scale features
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X), columns=X.columns, index=X.index
            )
        else:
            X_scaled = X.copy()

        # Start with original features
        X_transformed = X_scaled.copy()

        # Add decomposition features
        for method, decomposer in self.decomposers.items():
            decomp_features = decomposer.transform(X_scaled)

            # Create feature names
            for i in range(decomp_features.shape[1]):
                feature_name = f"{method.upper()}_{i}"
                X_transformed[feature_name] = decomp_features[:, i]

        # Add clustering features
        for cluster_name, clusterer in self.clusterers.items():
            # Cluster assignments
            cluster_labels = clusterer.predict(X_scaled)
            X_transformed[f"{cluster_name}_cluster"] = cluster_labels

            # Distance to cluster centers
            cluster_distances = clusterer.transform(X_scaled)
            for i in range(cluster_distances.shape[1]):
                X_transformed[f"{cluster_name}_dist_{i}"] = cluster_distances[:, i]

        # Add polynomial features
        if self.poly_features is not None:
            poly_result = self.poly_features.transform(X_scaled)
            # Convert sparse matrix to dense if needed
            if sparse.issparse(poly_result):
                poly_array: np.ndarray = poly_result.toarray()  # type: ignore
            else:
                poly_array = np.asarray(poly_result)

            poly_names = self.poly_features.get_feature_names_out(X_scaled.columns)

            # Add only interaction terms (exclude original features)
            for i, name in enumerate(poly_names):
                if " " in name:  # Interaction term
                    X_transformed[f"poly_{name}"] = poly_array[:, i]

        return X_transformed

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step

        Parameters:
        -----------
        X : pd.DataFrame
            Features to fit and transform
        y : pd.Series, optional
            Target variable

        Returns:
        --------
        pd.DataFrame
            Transformed features
        """
        return self.fit(X, y).transform(X)


class BinningTransformer:
    """Create binned features from continuous variables"""

    def __init__(self, n_bins: int = 5, strategy: str = "quantile"):
        """
        Initialize binning transformer

        Parameters:
        -----------
        n_bins : int
            Number of bins
        strategy : str
            Binning strategy: 'uniform', 'quantile'
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges = {}
        self.fitted = False

    def fit(self, X: pd.DataFrame) -> "BinningTransformer":
        """
        Fit binning on numeric columns

        Parameters:
        -----------
        X : pd.DataFrame
            Training data
        """
        numeric_columns = X.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if self.strategy == "quantile":
                # Use quantile-based binning
                _, edges = pd.qcut(
                    X[col], q=self.n_bins, retbins=True, duplicates="drop"
                )
            else:
                # Use uniform binning
                _, edges = pd.cut(X[col], bins=self.n_bins, retbins=True)

            self.bin_edges[col] = edges

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by adding binned features

        Parameters:
        -----------
        X : pd.DataFrame
            Data to transform

        Returns:
        --------
        pd.DataFrame
            Data with binned features
        """
        if not self.fitted:
            raise ValueError("BinningTransformer must be fitted before transform")

        X_binned = X.copy()

        for col, edges in self.bin_edges.items():
            if col in X.columns:
                binned_values = pd.cut(
                    X[col], bins=edges, include_lowest=True, labels=False
                )
                X_binned[f"{col}_binned"] = binned_values

        return X_binned

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)


def create_advanced_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
    target: pd.Series | None = None,
    methods: list[str] = ["pca", "clustering"],
    n_components: int | float = 0.95,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for advanced feature engineering

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    valid_df : pd.DataFrame, optional
        Validation data
    target : pd.Series, optional
        Target variable
    methods : list
        Feature engineering methods to apply
    n_components : int or float
        Number of components for decomposition

    Returns:
    --------
    pd.DataFrame or tuple
        Engineered features
    """
    # Setup feature engineering
    engineer = FeatureEngineering(
        decomposition_methods=methods,
        n_components=n_components,
        clustering_features="clustering" in methods,
        polynomial_features="polynomial" in methods,
    )

    # Transform training data
    train_transformed = engineer.fit_transform(train_df, target)

    if valid_df is not None:
        valid_transformed = engineer.transform(valid_df)
        return train_transformed, valid_transformed

    return train_transformed


# For backward compatibility
class feature_engineering_class(FeatureEngineering):
    """Legacy class name for backward compatibility"""

    pass
