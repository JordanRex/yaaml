"""
Native Algorithm Implementations for YAAML
==========================================

This module provides sklearn-native implementations of various ML algorithms
with hyperparameter optimization, maintaining the structured approach of the
original code but using only sklearn and standard Python libraries.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV

warnings.filterwarnings("ignore")


class NativeAlgorithmBase:
    """Base class for all native algorithm implementations"""

    def __init__(self, task_type: str = "classification", random_state: int = 42):
        self.task_type = task_type
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
        self.best_model = None

    def _get_scoring_metric(self):
        """Get appropriate scoring metric based on task type"""
        if self.task_type == "classification":
            return "roc_auc"
        else:
            return "r2"

    def _validate_and_cast_params(self, params: dict, int_params: list[str]) -> dict:
        """Ensure specified parameters are integers"""
        validated_params = params.copy()
        for param in int_params:
            if param in validated_params:
                validated_params[param] = int(validated_params[param])
        return validated_params

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5,
        max_evals: int = 20,
    ) -> dict:
        """
        Perform hyperparameter optimization using sklearn's native methods

        Returns:
            Dict with best parameters and score
        """
        raise NotImplementedError("Subclasses must implement optimize method")


class NativeRandomForest(NativeAlgorithmBase):
    """Native Random Forest implementation with hyperparameter optimization"""

    def __init__(self, task_type: str = "classification", random_state: int = 42):
        super().__init__(task_type, random_state)
        self.int_params = [
            "max_depth",
            "n_estimators",
            "min_samples_split",
            "min_samples_lea",
        ]

    def _get_param_space(self, n_features: int) -> dict:
        """Define parameter space for Random Forest"""
        return {
            "n_estimators": [100, 200, 300, 500, 800],
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_lea": [1, 2, 4],
            "max_features": ["sqrt", "log2", min(20, max(1, n_features // 3))],
            "criterion": (
                ["gini", "entropy"]
                if self.task_type == "classification"
                else ["squared_error", "absolute_error"]
            ),
        }

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5,
        max_evals: int = 20,
    ) -> dict:
        """Optimize Random Forest hyperparameters"""

        # Get appropriate model class
        if self.task_type == "classification":
            model_class = RandomForestClassifier
        else:
            model_class = RandomForestRegressor

        # Define parameter space
        param_space = self._get_param_space(X_train.shape[1])

        # Create base model
        base_model = model_class(random_state=self.random_state)

        # Use RandomizedSearchCV for efficient search
        search = RandomizedSearchCV(
            base_model,
            param_space,
            n_iter=max_evals,
            cv=cv_folds,
            scoring=self._get_scoring_metric(),
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Fit and get best parameters
        search.fit(X_train, y_train)

        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.best_model = search.best_estimator_

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "cv_scores": search.cv_results_,
        }


class NativeGradientBoosting(NativeAlgorithmBase):
    """Native Gradient Boosting (sklearn's GBM as XGBoost alternative)"""

    def __init__(self, task_type: str = "classification", random_state: int = 42):
        super().__init__(task_type, random_state)
        self.int_params = [
            "max_depth",
            "n_estimators",
            "min_samples_split",
            "min_samples_lea",
        ]

    def _get_param_space(self) -> dict:
        """Define parameter space for Gradient Boosting"""
        return {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6],
            "min_samples_split": [2, 5, 10],
            "min_samples_lea": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0],
            "max_features": ["sqrt", "log2", None],
        }

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5,
        max_evals: int = 20,
    ) -> dict:
        """Optimize Gradient Boosting hyperparameters"""

        # Get appropriate model class
        if self.task_type == "classification":
            model_class = GradientBoostingClassifier
        else:
            model_class = GradientBoostingRegressor

        # Define parameter space
        param_space = self._get_param_space()

        # Create base model
        base_model = model_class(random_state=self.random_state)

        # Use RandomizedSearchCV
        search = RandomizedSearchCV(
            base_model,
            param_space,
            n_iter=max_evals,
            cv=cv_folds,
            scoring=self._get_scoring_metric(),
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Fit and get best parameters
        search.fit(X_train, y_train)

        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.best_model = search.best_estimator_

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "cv_scores": search.cv_results_,
        }


class NativeLinearModel(NativeAlgorithmBase):
    """Native Linear Model (Logistic Regression / Linear Regression)"""

    def __init__(self, task_type: str = "classification", random_state: int = 42):
        super().__init__(task_type, random_state)
        self.int_params = ["max_iter"]

    def _get_param_space(self) -> dict:
        """Define parameter space for Linear Models"""
        if self.task_type == "classification":
            return {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2", "elasticnet"],
                "solver": ["liblinear", "saga"],
                "max_iter": [1000, 2000, 3000],
            }
        else:
            return {
                "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
                "fit_intercept": [True, False],
                "max_iter": [1000, 2000, 3000],
            }

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5,
        max_evals: int = 20,
    ) -> dict:
        """Optimize Linear Model hyperparameters"""

        # Get appropriate model class
        if self.task_type == "classification":
            model_class = LogisticRegression
        else:
            from sklearn.linear_model import Ridge

            model_class = Ridge

        # Define parameter space
        param_space = self._get_param_space()

        # Create base model
        base_model = model_class(random_state=self.random_state)

        # Use RandomizedSearchCV
        search = RandomizedSearchCV(
            base_model,
            param_space,
            n_iter=min(max_evals, len(list(ParameterGrid(param_space)))),
            cv=cv_folds,
            scoring=self._get_scoring_metric(),
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Fit and get best parameters
        search.fit(X_train, y_train)

        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.best_model = search.best_estimator_

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "cv_scores": search.cv_results_,
        }


class AlgorithmFactory:
    """Factory class to create algorithm instances"""

    ALGORITHMS = {
        "random_forest": NativeRandomForest,
        "gradient_boosting": NativeGradientBoosting,  # Native XGBoost alternative
        "linear_model": NativeLinearModel,
        # Add more as needed
    }

    @classmethod
    def create_algorithm(
        cls,
        algorithm_name: str,
        task_type: str = "classification",
        random_state: int = 42,
    ) -> NativeAlgorithmBase:
        """Create an algorithm instance"""
        if algorithm_name not in cls.ALGORITHMS:
            raise ValueError(
                f"Algorithm {algorithm_name} not supported. "
                f"Available: {list(cls.ALGORITHMS.keys())}"
            )

        return cls.ALGORITHMS[algorithm_name](
            task_type=task_type, random_state=random_state
        )

    @classmethod
    def get_available_algorithms(cls) -> list[str]:
        """Get list of available algorithms"""
        return list(cls.ALGORITHMS.keys())


# Example usage and integration with main AutoML class
class NativeAlgorithmSelector:
    """Selects and optimizes the best algorithm from available options"""

    def __init__(self, task_type: str = "classification", random_state: int = 42):
        self.task_type = task_type
        self.random_state = random_state
        self.results: dict[str, Any] = {}

    def find_best_algorithm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        algorithms: list[str] | None = None,
        cv_folds: int = 5,
        max_evals_per_algo: int = 20,
    ) -> dict:
        """
        Find the best algorithm and hyperparameters

        Returns:
            Dict with best algorithm, parameters, and performance
        """
        if algorithms is None:
            algorithms = ["random_forest", "gradient_boosting", "linear_model"]

        best_score = -np.inf
        best_algorithm = None
        best_params = None
        best_model = None

        for algo_name in algorithms:
            print(f"Optimizing {algo_name}...")

            # Create algorithm instance
            algo = AlgorithmFactory.create_algorithm(
                algo_name, self.task_type, self.random_state
            )

            # Optimize hyperparameters
            result = algo.optimize(X_train, y_train, cv_folds, max_evals_per_algo)

            # Store results
            self.results[algo_name] = result

            # Check if this is the best so far
            if result["best_score"] > best_score:
                best_score = result["best_score"]
                best_algorithm = algo_name
                best_params = result["best_params"]
                best_model = algo.best_model

        return {
            "best_algorithm": best_algorithm,
            "best_params": best_params,
            "best_score": best_score,
            "best_model": best_model,
            "all_results": self.results,
        }
