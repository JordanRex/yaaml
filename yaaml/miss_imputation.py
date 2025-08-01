"""
YAAML Missing Value Imputation Module
Native imputation methods using sklearn
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


class DataFrameImputer:
    """Native DataFrame imputer using sklearn imputers"""

    def __init__(self, strategy: str = "mean"):
        """
        Initialize the imputer

        Parameters:
        -----------
        strategy : str
            Imputation strategy: 'mean', 'median', 'most_frequent', 'constant',
            'knn', 'iterative'
        """
        self.strategy = strategy
        self.num_imputer: SimpleImputer | None = None
        self.cat_imputer: SimpleImputer | None = None
        self.advanced_imputer: KNNImputer | IterativeImputer | None = None
        self.numeric_columns: list[str] | None = None
        self.categorical_columns: list[str] | None = None
        self.fitted = False

    def fit(self, X: pd.DataFrame) -> "DataFrameImputer":
        """
        Fit the imputer on training data

        Parameters:
        -----------
        X : pd.DataFrame
            Training data
        """
        # Identify numeric and categorical columns
        self.numeric_columns = list(X.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(
            X.select_dtypes(include=["object", "category"]).columns
        )

        if self.strategy in ["mean", "median"]:
            # For numeric columns
            if self.numeric_columns:
                self.num_imputer = SimpleImputer(strategy=self.strategy)
                self.num_imputer.fit(X[self.numeric_columns])

            # For categorical columns - use most frequent
            if self.categorical_columns:
                self.cat_imputer = SimpleImputer(strategy="most_frequent")
                self.cat_imputer.fit(X[self.categorical_columns])

        elif self.strategy == "most_frequent":
            # Use most frequent for all columns
            if self.numeric_columns:
                self.num_imputer = SimpleImputer(strategy="most_frequent")
                self.num_imputer.fit(X[self.numeric_columns])

            if self.categorical_columns:
                self.cat_imputer = SimpleImputer(strategy="most_frequent")
                self.cat_imputer.fit(X[self.categorical_columns])

        elif self.strategy == "knn":
            # KNN imputation for numeric data only
            if self.numeric_columns:
                self.advanced_imputer = KNNImputer(n_neighbors=5)
                self.advanced_imputer.fit(X[self.numeric_columns])

            # Fallback to most frequent for categorical
            if self.categorical_columns:
                self.cat_imputer = SimpleImputer(strategy="most_frequent")
                self.cat_imputer.fit(X[self.categorical_columns])

        elif self.strategy == "iterative":
            # Iterative imputation for numeric data only
            if self.numeric_columns:
                self.advanced_imputer = IterativeImputer(random_state=42, max_iter=10)
                self.advanced_imputer.fit(X[self.numeric_columns])

            # Fallback to most frequent for categorical
            if self.categorical_columns:
                self.cat_imputer = SimpleImputer(strategy="most_frequent")
                self.cat_imputer.fit(X[self.categorical_columns])

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted imputer

        Parameters:
        -----------
        X : pd.DataFrame
            Data to transform

        Returns:
        --------
        pd.DataFrame
            Imputed data
        """
        if not self.fitted:
            raise ValueError("Imputer must be fitted before transform")

        X_imputed = X.copy()

        # Handle numeric columns
        if self.numeric_columns:
            if (
                self.strategy in ["knn", "iterative"]
                and self.advanced_imputer is not None
            ):
                X_imputed[self.numeric_columns] = self.advanced_imputer.transform(
                    X[self.numeric_columns]
                )
            elif self.num_imputer is not None:
                X_imputed[self.numeric_columns] = self.num_imputer.transform(
                    X[self.numeric_columns]
                )

        # Handle categorical columns
        if self.categorical_columns and self.cat_imputer is not None:
            X_imputed[self.categorical_columns] = self.cat_imputer.transform(
                X[self.categorical_columns]
            )

        return X_imputed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step

        Parameters:
        -----------
        X : pd.DataFrame
            Data to fit and transform

        Returns:
        --------
        pd.DataFrame
            Imputed data
        """
        return self.fit(X).transform(X)


def impute_missing_values(
    train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None, strategy: str = "mean"
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for missing value imputation

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    valid_df : pd.DataFrame, optional
        Validation data
    strategy : str
        Imputation strategy

    Returns:
    --------
    pd.DataFrame or tuple
        Imputed training data, or tuple of (train, valid) if valid_df provided
    """
    imputer = DataFrameImputer(strategy=strategy)
    train_imputed = imputer.fit_transform(train_df)

    if valid_df is not None:
        valid_imputed = imputer.transform(valid_df)
        return train_imputed, valid_imputed

    return train_imputed
