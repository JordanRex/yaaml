"""
Test Feature Selection Module
============================

Comprehensive tests for feature selection functionality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from yaaml.feature_selection import FeatureSelector, select_features


class TestFeatureSelector:
    """Test FeatureSelector class"""

    def test_init_default(self) -> None:
        """Test FeatureSelector initialization with defaults"""
        selector = FeatureSelector()
        assert selector.methods == ["variance", "univariate"]
        assert selector.variance_threshold == 0.1
        assert selector.k_best == 10
        assert selector.percentile == 50
        assert selector.task_type == "classification"

    def test_init_custom_params(self) -> None:
        """Test FeatureSelector initialization with custom parameters"""
        selector = FeatureSelector(
            methods=["variance", "mutual_info"],
            variance_threshold=0.05,
            k_best=5,
            percentile=25,
            task_type="regression",
        )
        assert selector.methods == ["variance", "mutual_info"]
        assert selector.variance_threshold == 0.05
        assert selector.k_best == 5
        assert selector.percentile == 25
        assert selector.task_type == "regression"

    def test_variance_selection(self) -> None:
        """Test variance-based feature selection"""
        # Create data with some low-variance features
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "high_var": np.random.randn(100),
                "low_var": np.full(100, 0.001),  # Very low variance
                "good_var": np.random.randn(100) * 2,
                "zero_var": np.zeros(100),  # Zero variance
            }
        )

        selector = FeatureSelector(methods=["variance"], variance_threshold=0.01)
        X_selected = selector.fit_transform(X)

        # Should remove low variance and zero variance features
        assert X_selected.shape[1] < X.shape[1]
        assert "zero_var" not in X_selected.columns
        assert "low_var" not in X_selected.columns

    def test_univariate_selection_classification(self) -> None:
        """Test univariate feature selection for classification"""
        X, y = make_classification(
            n_samples=200, n_features=15, n_informative=5, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        selector = FeatureSelector(
            methods=["univariate"], k_best=5, task_type="classification"
        )
        X_selected = selector.fit_transform(X_df, y_series)

        assert X_selected.shape[1] == 5
        assert X_selected.shape[0] == X_df.shape[0]
        assert hasattr(selector, "selected_features")

    def test_univariate_selection_regression(self) -> None:
        """Test univariate feature selection for regression"""
        X, y = make_regression(
            n_samples=200, n_features=15, n_informative=5, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        selector = FeatureSelector(
            methods=["univariate"], k_best=5, task_type="regression"
        )
        X_selected = selector.fit_transform(X_df, y_series)

        assert X_selected.shape[1] == 5
        assert X_selected.shape[0] == X_df.shape[0]

    def test_mutual_info_selection(self) -> None:
        """Test mutual information feature selection"""
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        selector = FeatureSelector(
            methods=["mutual_info"], percentile=50, task_type="classification"
        )
        X_selected = selector.fit_transform(X_df, y_series)

        # Should select approximately 50% of features
        expected_features = max(1, int(X_df.shape[1] * 0.5))
        assert X_selected.shape[1] <= expected_features + 1  # Allow small variance

    def test_rfe_selection(self) -> None:
        """Test recursive feature elimination"""
        X, y = make_classification(
            n_samples=100, n_features=8, n_informative=4, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        selector = FeatureSelector(methods=["rfe"], task_type="classification")
        X_selected = selector.fit_transform(X_df, y_series)

        # RFE should select optimal number of features
        assert X_selected.shape[1] <= X_df.shape[1]
        assert X_selected.shape[0] == X_df.shape[0]

    def test_combined_methods(self) -> None:
        """Test combining multiple selection methods"""
        X, y = make_classification(
            n_samples=150, n_features=20, n_informative=8, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        selector = FeatureSelector(
            methods=["variance", "univariate", "mutual_info"],
            variance_threshold=0.01,
            k_best=10,
            percentile=70,
            task_type="classification",
        )
        X_selected = selector.fit_transform(X_df, y_series)

        # Combined methods should significantly reduce features
        assert X_selected.shape[1] < X_df.shape[1]
        assert X_selected.shape[0] == X_df.shape[0]

    def test_fit_transform_consistency(self) -> None:
        """Test that fit + transform gives same result as fit_transform"""
        X, y = make_classification(
            n_samples=100, n_features=8, n_informative=4, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        selector1 = FeatureSelector(methods=["univariate"], k_best=4)
        selector2 = FeatureSelector(methods=["univariate"], k_best=4)

        # Method 1: fit_transform
        X_selected1 = selector1.fit_transform(X_df, y_series)

        # Method 2: fit then transform
        selector2.fit(X_df, y_series)
        X_selected2 = selector2.transform(X_df)

        pd.testing.assert_frame_equal(X_selected1, X_selected2)

    def test_transform_before_fit_error(self) -> None:
        """Test error when transform is called before fit"""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        selector = FeatureSelector()
        with pytest.raises(ValueError, match="must be fitted before transform"):
            selector.transform(X_df)

    def test_unsupervised_methods_without_target(self) -> None:
        """Test that variance method works without target"""
        X, _ = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        selector = FeatureSelector(methods=["variance"])
        X_selected = selector.fit_transform(X_df)  # No target provided

        assert X_selected.shape[0] == X_df.shape[0]
        assert X_selected.shape[1] <= X_df.shape[1]


class TestSelectFeaturesFunction:
    """Test select_features convenience function"""

    def test_select_features_basic(self) -> None:
        """Test basic functionality of select_features function"""
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        X_selected = select_features(
            X_df, target=y_series, methods=["univariate"], k_best=5
        )

        assert isinstance(X_selected, pd.DataFrame)
        assert X_selected.shape[1] == 5
        assert X_selected.shape[0] == X_df.shape[0]

    def test_select_features_with_validation(self) -> None:
        """Test select_features with validation set"""
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        # Split into train/validation
        train_X = X_df.iloc[:150]
        valid_X = X_df.iloc[150:]
        train_y = y_series.iloc[:150]

        result = select_features(
            train_X, valid_X, target=train_y, methods=["univariate"], k_best=5
        )

        assert isinstance(result, tuple)
        train_selected, valid_selected = result
        assert train_selected.shape[1] == 5
        assert valid_selected.shape[1] == 5
        assert train_selected.columns.tolist() == valid_selected.columns.tolist()

    def test_select_features_regression(self) -> None:
        """Test select_features with regression task"""
        X, y = make_regression(
            n_samples=100, n_features=8, n_informative=4, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        X_selected = select_features(
            X_df,
            target=y_series,
            methods=["univariate"],
            k_best=4,
            task_type="regression",
        )

        assert isinstance(X_selected, pd.DataFrame)
        assert X_selected.shape[1] == 4
        assert X_selected.shape[0] == X_df.shape[0]


class TestFeatureSelectionEdgeCases:
    """Test edge cases and error conditions"""

    def test_more_features_than_available(self) -> None:
        """Test when requesting more features than available"""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        selector = FeatureSelector(methods=["univariate"], k_best=10)
        X_selected = selector.fit_transform(X_df, y_series)

        # Should return all available features (k_best is automatically limited)
        assert X_selected.shape[1] == X_df.shape[1]

    def test_empty_dataframe(self) -> None:
        """Test behavior with empty DataFrame"""
        X_df = pd.DataFrame()
        y_series = pd.Series([])

        selector = FeatureSelector(methods=["variance"])
        with pytest.raises((ValueError, IndexError)):
            selector.fit_transform(X_df, y_series)

    def test_single_feature(self) -> None:
        """Test with single feature DataFrame"""
        X, y = make_classification(
            n_samples=50,
            n_features=4,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        selector = FeatureSelector(methods=["univariate"], k_best=1)
        X_selected = selector.fit_transform(X_df, y_series)

        assert X_selected.shape[1] == 1
        assert X_selected.shape[0] == X_df.shape[0]
        # Verify selected feature is one of the original features
        assert X_selected.columns[0] in X_df.columns

    def test_supervised_methods_without_target(self) -> None:
        """Test that supervised methods fail gracefully without target"""
        X, _ = make_classification(n_samples=50, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        selector = FeatureSelector(methods=["univariate"])
        # Should handle missing target gracefully for supervised methods
        X_selected = selector.fit_transform(X_df)  # No target

        # When no target, supervised methods should be skipped
        assert X_selected.shape == X_df.shape
