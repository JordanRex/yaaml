"""
Realistic AutoML Integration Tests
=================================

This module contains comprehensive integration tests that simulate real-world
AutoML scenarios with challenging datasets, proper train/test splits, and
realistic performance expectations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from yaaml import YAAMLAutoML


class TestRealisticAutoML:
    """Test AutoML with realistic, challenging datasets"""

    @pytest.fixture
    def realistic_classification_data(self):
        """Create a realistic classification dataset with mixed data types"""
        np.random.seed(42)

        # Create base numeric features with noise
        X, y = make_classification(
            n_samples=400,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            flip_y=0.08,  # 8% label noise
            class_sep=0.75,  # Moderate class separation
            random_state=42,
        )

        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f"numeric_{i}" for i in range(X.shape[1])])

        # Add categorical features
        df["category"] = np.random.choice(["A", "B", "C", "D"], size=len(df))
        df["status"] = np.random.choice(["active", "inactive"], size=len(df))
        df["region"] = np.random.choice(
            ["north", "south", "east", "west"], size=len(df)
        )

        # Add missing values
        missing_indices = np.random.choice(len(df), size=40, replace=False)
        df.loc[list(missing_indices), "numeric_0"] = np.nan

        cat_missing = np.random.choice(len(df), size=20, replace=False)
        df.loc[list(cat_missing), "category"] = np.nan

        target = pd.Series(y, name="target")

        return df, target

    @pytest.fixture
    def challenging_classification_data(self):
        """Create a very challenging classification dataset"""
        np.random.seed(123)

        # Very challenging dataset
        X, y = make_classification(
            n_samples=300,
            n_features=8,
            n_informative=4,
            n_redundant=2,
            n_clusters_per_class=2,
            flip_y=0.15,  # 15% label noise
            class_sep=0.6,  # Lower class separation
            random_state=123,
        )

        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        # High cardinality categoricals
        df["department"] = np.random.choice(
            [
                "Engineering",
                "Sales",
                "Marketing",
                "HR",
                "Finance",
                "Operations",
                "Legal",
                "Research",
            ],
            size=len(df),
        )

        df["location"] = np.random.choice(
            ["NYC", "SF", "LA", "Chicago", "Boston", "Austin", "Seattle"],
            size=len(df),
        )

        # Substantial missing values
        missing_numeric = np.random.choice(
            len(df), size=int(0.15 * len(df)), replace=False
        )
        df.loc[list(missing_numeric), "feature_0"] = np.nan

        missing_cat = np.random.choice(len(df), size=int(0.12 * len(df)), replace=False)
        df.loc[list(missing_cat), "department"] = np.nan

        target = pd.Series(y, name="target")

        return df, target

    @pytest.fixture
    def realistic_regression_data(self):
        """Create a realistic regression dataset"""
        np.random.seed(42)

        # Regression with noise
        X, y = make_regression(
            n_samples=300,
            n_features=6,
            n_informative=4,
            noise=0.2,
            random_state=42,
        )[
            :2
        ]  # Only take X and y, not coef

        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        # Add categoricals
        df["type"] = np.random.choice(["TypeA", "TypeB", "TypeC"], size=len(df))
        df["grade"] = np.random.choice(["High", "Medium", "Low"], size=len(df))

        # Add missing values
        missing_idx = np.random.choice(len(df), size=30, replace=False)
        df.loc[list(missing_idx), "feature_1"] = np.nan

        target = pd.Series(y, name="target")

        return df, target

    def test_realistic_classification_pipeline(self, realistic_classification_data):
        """Test full AutoML pipeline on realistic classification data"""
        X, y = realistic_classification_data

        # Proper train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Initialize AutoML
        automl = YAAMLAutoML(
            mode="classification",
            max_evals=3,
            cv_folds=3,
            verbosity=0,
            feature_engineering=True,
            feature_selection=True,
        )

        # Test fitting
        automl.fit(X_train, y_train)

        # Verify model was trained
        assert automl.model is not None
        assert hasattr(automl, "model")

        # Test predictions
        predictions = automl.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)

        # Test evaluation
        metrics = automl.evaluate(X_test, y_test)

        # Verify realistic performance
        accuracy = metrics["test_accuracy"]
        assert 0.65 <= accuracy <= 0.95, f"Unrealistic accuracy: {accuracy:.3f}"

        # Verify all metrics are present
        expected_metrics = [
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
        ]
        for metric in expected_metrics:
            assert metric in metrics
            assert 0.0 <= metrics[metric] <= 1.0

    def test_challenging_classification_dataset(self, challenging_classification_data):
        """Test AutoML on very challenging dataset"""
        X, y = challenging_classification_data

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=123, stratify=y
        )

        automl = YAAMLAutoML(
            mode="classification",
            max_evals=4,
            cv_folds=3,
            verbosity=0,
            imputation_strategy="iterative",
            encoding_method="onehot",
            feature_engineering=True,
            feature_selection=True,
        )

        automl.fit(X_train, y_train)
        predictions = automl.predict(X_test)
        metrics = automl.evaluate(X_test, y_test)

        # For challenging dataset, lower expectations
        accuracy = metrics["test_accuracy"]
        assert (
            0.55 <= accuracy <= 0.90
        ), f"Unexpected accuracy on challenging data: {accuracy:.3f}"

        # But should still produce valid predictions
        assert len(predictions) == len(X_test)
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)

    def test_realistic_regression_pipeline(self, realistic_regression_data):
        """Test full AutoML pipeline on realistic regression data"""
        X, y = realistic_regression_data

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        automl = YAAMLAutoML(mode="regression", max_evals=3, cv_folds=3, verbosity=0)

        automl.fit(X_train, y_train)
        predictions = automl.predict(X_test)
        metrics = automl.evaluate(X_test, y_test)

        # Verify regression-specific behavior
        assert automl.model is not None
        assert len(predictions) == len(X_test)
        assert all(isinstance(pred, (float, np.floating)) for pred in predictions)

        # Verify regression metrics
        expected_metrics = ["test_r2", "test_mse", "test_mae", "test_rmse"]
        for metric in expected_metrics:
            assert metric in metrics

        # R² should be reasonable
        r2 = metrics["test_r2"]
        assert -0.5 <= r2 <= 1.0, f"Unreasonable R² score: {r2:.3f}"

    def test_small_dataset_handling(self):
        """Test AutoML behavior with small datasets"""
        # Very small dataset
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "num1": np.random.randn(50),
                "num2": np.random.randn(50),
                "cat": np.random.choice(["A", "B"], size=50),
            }
        )
        y = pd.Series(np.random.choice([0, 1], size=50))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        automl = YAAMLAutoML(
            mode="classification",
            max_evals=2,
            cv_folds=2,  # Reduced for small dataset
            verbosity=0,
        )

        # Should handle small dataset gracefully
        automl.fit(X_train, y_train)
        predictions = automl.predict(X_test)
        metrics = automl.evaluate(X_test, y_test)

        assert len(predictions) == len(X_test)
        assert "test_accuracy" in metrics

    def test_all_preprocessing_modules_integration(self, realistic_classification_data):
        """Test that all preprocessing modules work together"""
        X, y = realistic_classification_data

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Enable all preprocessing options
        automl = YAAMLAutoML(
            mode="classification",
            max_evals=2,
            cv_folds=2,
            verbosity=0,
            # Test all preprocessing components
            imputation_strategy="mean",
            encoding_method="onehot",
            feature_engineering=True,
            feature_selection=True,
            sampling_strategy="auto",
        )

        automl.fit(X_train, y_train)

        # Verify preprocessing worked
        assert hasattr(automl, "model")

        # Feature engineering may change dimensionality
        predictions = automl.predict(X_test)
        assert len(predictions) == len(X_test)

        metrics = automl.evaluate(X_test, y_test)
        assert "test_accuracy" in metrics

        # Get preprocessing info to verify all modules ran
        info = automl.get_preprocessing_info()
        assert "imputation_strategy" in info
        assert "encoding_method" in info
        assert "feature_engineering_enabled" in info
        assert "feature_selection_enabled" in info
        assert info["preprocessing_fitted"] is True


class TestPerformanceBenchmarks:
    """Performance and timing tests"""

    def test_automl_training_time(self):
        """Test that AutoML completes training in reasonable time"""
        import time

        # Create medium-sized dataset
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        automl = YAAMLAutoML(
            mode="classification", max_evals=2, cv_folds=2, verbosity=0
        )

        start_time = time.time()
        automl.fit(X_df, y_series)
        training_time = time.time() - start_time

        # Should complete within reasonable time (adjust as needed)
        assert training_time < 30.0, f"Training took too long: {training_time:.1f}s"

        # Verify it still works
        predictions = automl.predict(X_df.head(10))
        assert len(predictions) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
