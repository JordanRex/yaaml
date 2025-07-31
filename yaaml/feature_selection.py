"""
YAAML Feature Selection Module
Native feature selection using sklearn methods
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFECV,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)


class FeatureSelector:
    """Native feature selector with multiple selection strategies"""

    def __init__(
        self,
        methods: list[str] = ["variance", "univariate"],
        variance_threshold: float = 0.1,
        k_best: int = 10,
        percentile: float = 50,
        task_type: str = "classification",
    ):
        """
        Initialize feature selector

        Parameters:
        -----------
        methods : list
            Selection methods: ['variance', 'univariate', 'rfe', 'mutual_info']
        variance_threshold : float
            Threshold for variance-based selection
        k_best : int
            Number of best features to select
        percentile : float
            Percentile of features to select
        task_type : str
            'classification' or 'regression'
        """
        self.methods = methods
        self.variance_threshold = variance_threshold
        self.k_best = k_best
        self.percentile = percentile
        self.task_type = task_type

        # Fitted selectors
        self.variance_selector = None
        self.univariate_selector = None
        self.rfe_selector = None
        self.mutual_info_selector = None
        self.selected_features = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureSelector":
        """
        Fit feature selectors

        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series, optional
            Target variable (required for supervised methods)
        """
        self.original_features = list(X.columns)
        X_current = X.copy()
        selected_mask = np.ones(X.shape[1], dtype=bool)

        # Variance-based selection
        if "variance" in self.methods:
            self.variance_selector = VarianceThreshold(
                threshold=self.variance_threshold
            )
            variance_mask = self.variance_selector.fit(X_current).get_support()
            X_current = X_current.loc[:, variance_mask]
            selected_mask = selected_mask & variance_mask

        # Univariate statistical tests
        if "univariate" in self.methods and y is not None:
            if self.task_type == "classification":
                score_func = f_classif
            else:
                score_func = f_regression

            self.univariate_selector = SelectKBest(
                score_func=score_func, k=min(self.k_best, X_current.shape[1])
            )
            univariate_mask = self.univariate_selector.fit(X_current, y).get_support()

            # Update mask for remaining features
            temp_mask = np.zeros(len(selected_mask), dtype=bool)
            temp_mask[selected_mask] = univariate_mask
            selected_mask = temp_mask
            X_current = X.loc[:, selected_mask]

        # Recursive feature elimination
        if "rfe" in self.methods and y is not None:
            if self.task_type == "classification":
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)

            self.rfe_selector = RFECV(
                estimator=estimator,
                step=1,
                cv=3,
                scoring="accuracy" if self.task_type == "classification" else "r2",
                n_jobs=-1,
            )
            rfe_mask = self.rfe_selector.fit(X_current, y).get_support()

            # Update mask for remaining features
            temp_mask = np.zeros(len(selected_mask), dtype=bool)
            temp_mask[selected_mask] = rfe_mask
            selected_mask = temp_mask
            X_current = X.loc[:, selected_mask]

        # Mutual information
        if "mutual_info" in self.methods and y is not None:
            if self.task_type == "classification":
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression

            self.mutual_info_selector = SelectPercentile(
                score_func=score_func, percentile=int(min(self.percentile, 100))
            )
            mi_mask = self.mutual_info_selector.fit(X_current, y).get_support()

            # Update mask for remaining features
            temp_mask = np.zeros(len(selected_mask), dtype=bool)
            temp_mask[selected_mask] = mi_mask
            selected_mask = temp_mask

        # Store selected features
        self.selected_features = list(X.columns[selected_mask])
        self.fitted = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted selectors

        Parameters:
        -----------
        X : pd.DataFrame
            Features to transform

        Returns:
        --------
        pd.DataFrame
            Selected features
        """
        if not self.fitted:
            raise ValueError("FeatureSelector must be fitted before transform")

        if self.selected_features is None:
            return X.copy()

        # Return only selected features
        available_features = [f for f in self.selected_features if f in X.columns]
        return X[available_features].copy()

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
            Selected features
        """
        return self.fit(X, y).transform(X)

    def get_feature_importance(self) -> pd.DataFrame | None:
        """
        Get feature importance scores from selectors

        Returns:
        --------
        pd.DataFrame or None
            Feature importance scores
        """
        if not self.fitted:
            return None

        importance_data = []

        # Univariate scores
        if self.univariate_selector is not None and hasattr(
            self.univariate_selector, "scores_"
        ):
            scores = self.univariate_selector.scores_
            feature_names = self.univariate_selector.feature_names_in_
            for i, (name, score) in enumerate(
                zip(np.asarray(feature_names), np.asarray(scores))
            ):
                importance_data.append(
                    {"feature": name, "method": "univariate", "score": float(score)}
                )

        # RFE scores (feature ranking)
        if self.rfe_selector is not None:
            rankings = self.rfe_selector.ranking_
            feature_names = self.rfe_selector.feature_names_in_
            for name, rank in zip(feature_names, rankings):
                importance_data.append(
                    {
                        "feature": name,
                        "method": "rfe_ranking",
                        "score": 1.0
                        / rank,  # Convert ranking to score (higher is better)
                    }
                )

        if importance_data:
            return pd.DataFrame(importance_data).sort_values("score", ascending=False)

        return None


def select_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
    target: pd.Series | None = None,
    methods: list[str] = ["variance", "univariate"],
    k_best: int = 10,
    task_type: str = "classification",
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for feature selection

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training features
    valid_df : pd.DataFrame, optional
        Validation features
    target : pd.Series, optional
        Target variable
    methods : list
        Selection methods to use
    k_best : int
        Number of features to select
    task_type : str
        'classification' or 'regression'

    Returns:
    --------
    pd.DataFrame or tuple
        Selected features
    """
    selector = FeatureSelector(methods=methods, k_best=k_best, task_type=task_type)

    train_selected = selector.fit_transform(train_df, target)

    if valid_df is not None:
        valid_selected = selector.transform(valid_df)
        return train_selected, valid_selected

    return train_selected


# Legacy class for backward compatibility
class feat_selection(FeatureSelector):
    """Legacy class name for backward compatibility"""

    def __init__(self, train, valid, y_train, t=0.2):
        """Legacy constructor"""
        super().__init__(
            methods=["variance", "rfe"],
            variance_threshold=t,
            task_type="classification",
        )

        # Fit and transform immediately (legacy behavior)
        X_train_selected = self.fit_transform(train, y_train)
        X_valid_selected = self.transform(valid)

        # Store results for legacy access
        self.X_train = X_train_selected
        self.X_valid = X_valid_selected
        self.final_features = self.selected_features
