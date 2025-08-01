"""
Test Sampling Module
===================

Comprehensive tests for sampling functionality.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from yaaml.sampling import NativeSampler, StratifiedSampler, apply_sampling


class TestNativeSampler:
    """Test NativeSampler class"""

    def test_init_default(self) -> None:
        """Test NativeSampler initialization with defaults"""
        sampler = NativeSampler()
        assert sampler.strategy == "auto"
        assert sampler.random_state == 42

    def test_init_custom(self) -> None:
        """Test NativeSampler initialization with custom parameters"""
        sampler = NativeSampler(strategy="smote", random_state=123)
        assert sampler.strategy == "smote"
        assert sampler.random_state == 123

    def test_smote_sampling(self) -> None:
        """Test SMOTE oversampling"""
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=4,
            n_classes=3,
            n_redundant=0,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        # Check initial imbalance
        initial_counts = y_series.value_counts()
        assert initial_counts.min() < initial_counts.max()

        sampler = NativeSampler(strategy="smote")
        X_resampled, y_resampled = sampler.fit_resample(X_df, y_series)

        # Check that minority class was oversampled
        final_counts = y_resampled.value_counts()
        assert len(X_resampled) >= len(X_df)
        assert final_counts.min() > initial_counts.min()

    def test_random_undersampling(self) -> None:
        """Test random undersampling"""
        X, y = make_classification(
            n_samples=200, n_features=5, weights=[0.8, 0.2], random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        sampler = NativeSampler(strategy="random_undersample")
        X_resampled, y_resampled = sampler.fit_resample(X_df, y_series)

        # Should reduce dataset size
        assert len(X_resampled) <= len(X_df)
        # Should balance classes
        final_counts = y_resampled.value_counts()
        assert abs(final_counts.iloc[0] - final_counts.iloc[1]) <= 1

    def test_auto_strategy_imbalanced(self) -> None:
        """Test auto strategy on imbalanced data"""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            weights=[0.95, 0.05],  # Very imbalanced
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        sampler = NativeSampler(strategy="auto")
        X_resampled, y_resampled = sampler.fit_resample(X_df, y_series)

        # Auto should detect imbalance and apply sampling
        initial_counts = y_series.value_counts()
        final_counts = y_resampled.value_counts()

        # Should improve balance
        initial_ratio = initial_counts.min() / initial_counts.max()
        final_ratio = final_counts.min() / final_counts.max()
        assert final_ratio >= initial_ratio

    def test_auto_strategy_balanced(self) -> None:
        """Test auto strategy on balanced data"""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            weights=[0.5, 0.5],
            random_state=42,  # Balanced
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        sampler = NativeSampler(strategy="auto")
        X_resampled, y_resampled = sampler.fit_resample(X_df, y_series)

        # Auto should detect balance and return original data
        assert len(X_resampled) == len(X_df)
        pd.testing.assert_frame_equal(X_resampled, X_df)
        pd.testing.assert_series_equal(y_resampled, y_series)


class TestStratifiedSampler:
    """Test StratifiedSampler class"""

    def test_init_default(self) -> None:
        """Test StratifiedSampler initialization"""
        sampler = StratifiedSampler()
        assert sampler.test_size == 0.2
        assert sampler.random_state == 42

    def test_split_data(self) -> None:
        """Test stratified train-test split"""
        X, y = make_classification(
            n_samples=200,
            n_features=5,
            n_classes=3,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        sampler = StratifiedSampler(test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = sampler.split(X_df, y_series)

        # Check sizes
        assert len(X_train) + len(X_test) == len(X_df)
        assert abs(len(X_test) / len(X_df) - 0.3) < 0.05  # Approximately 30%

        # Check stratification - class proportions should be similar
        original_props = y_series.value_counts(normalize=True).sort_index()
        train_props = y_train.value_counts(normalize=True).sort_index()
        test_props = y_test.value_counts(normalize=True).sort_index()

        # Proportions should be similar (within 10%)
        for cls in original_props.index:
            assert abs(train_props[cls] - original_props[cls]) < 0.1
            assert abs(test_props[cls] - original_props[cls]) < 0.1

    def test_custom_test_size(self) -> None:
        """Test custom test size"""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        test_sizes = [0.1, 0.25, 0.4]
        for test_size in test_sizes:
            sampler = StratifiedSampler(test_size=test_size)
            X_train, X_test, y_train, y_test = sampler.split(X_df, y_series)

            actual_test_ratio = len(X_test) / len(X_df)
            assert abs(actual_test_ratio - test_size) < 0.05


class TestApplySamplingFunction:
    """Test apply_sampling convenience function"""

    def test_apply_sampling_smote(self) -> None:
        """Test apply_sampling with SMOTE"""
        X, y = make_classification(
            n_samples=100, n_features=4, weights=[0.8, 0.2], random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        X_resampled, y_resampled = apply_sampling(X_df, y_series, strategy="smote")

        # Should increase dataset size (oversampling)
        assert len(X_resampled) >= len(X_df)

        # Should improve class balance
        original_counts = y_series.value_counts()
        resampled_counts = y_resampled.value_counts()

        original_ratio = original_counts.min() / original_counts.max()
        resampled_ratio = resampled_counts.min() / resampled_counts.max()
        assert resampled_ratio > original_ratio

    def test_apply_sampling_undersample(self) -> None:
        """Test apply_sampling with undersampling"""
        X, y = make_classification(
            n_samples=200, n_features=4, weights=[0.7, 0.3], random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        X_resampled, y_resampled = apply_sampling(
            X_df, y_series, strategy="random_undersample"
        )

        # Should decrease dataset size
        assert len(X_resampled) <= len(X_df)

    def test_apply_sampling_auto(self) -> None:
        """Test apply_sampling with auto strategy"""
        X, y = make_classification(
            n_samples=150,
            n_features=4,
            weights=[0.9, 0.1],  # Imbalanced
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        X_resampled, y_resampled = apply_sampling(X_df, y_series, strategy="auto")

        # Auto should detect imbalance and apply appropriate sampling
        original_counts = y_series.value_counts()
        resampled_counts = y_resampled.value_counts()

        original_ratio = original_counts.min() / original_counts.max()
        resampled_ratio = resampled_counts.min() / resampled_counts.max()
        assert resampled_ratio >= original_ratio

    def test_apply_sampling_none(self) -> None:
        """Test apply_sampling with None strategy"""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        X_resampled, y_resampled = apply_sampling(X_df, y_series, strategy="none")

        # Should return original data unchanged
        pd.testing.assert_frame_equal(X_resampled, X_df)
        pd.testing.assert_series_equal(y_resampled, y_series)


class TestSamplingEdgeCases:
    """Test edge cases and error conditions"""

    def test_binary_classification_edge_case(self) -> None:
        """Test sampling with minimal data points"""
        # Very small dataset
        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6], "f2": [1, 1, 2, 2, 3, 3]})
        y = pd.Series([0, 0, 0, 1, 1, 1])  # Perfectly balanced but tiny

        sampler = NativeSampler(strategy="auto")
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Should handle small dataset gracefully
        assert len(X_resampled) >= len(X)

    def test_single_class_data(self) -> None:
        """Test behavior with single class"""
        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [1, 1, 2, 2, 3]})
        y = pd.Series([0, 0, 0, 0, 0])  # All same class

        sampler = NativeSampler(strategy="smote")
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Should return original data when only one class
        assert len(X_resampled) == len(X)
        assert len(y_resampled.unique()) == 1

    def test_invalid_strategy(self) -> None:
        """Test error handling for invalid strategy"""
        sampler = NativeSampler(strategy="invalid_strategy")

        X, y = make_classification(
            n_samples=50, n_features=10, n_informative=5, n_redundant=0, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        with pytest.raises((ValueError, KeyError)):
            sampler.fit_resample(X_df, y_series)

    def test_multiclass_sampling(self) -> None:
        """Test sampling with multiple classes"""
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=6,
            n_classes=4,
            n_redundant=0,
            weights=[0.6, 0.2, 0.15, 0.05],  # Imbalanced multiclass
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        sampler = NativeSampler(strategy="smote")
        X_resampled, y_resampled = sampler.fit_resample(X_df, y_series)

        # Should increase representation of minority classes
        original_counts = y_series.value_counts().sort_index()
        resampled_counts = y_resampled.value_counts().sort_index()

        # Minority class should have more samples
        assert resampled_counts.iloc[-1] > original_counts.iloc[-1]

    def test_stratified_sampler_reproducibility(self) -> None:
        """Test that StratifiedSampler is reproducible with same random_state"""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        sampler1 = StratifiedSampler(random_state=42)
        sampler2 = StratifiedSampler(random_state=42)

        X_train1, X_test1, y_train1, y_test1 = sampler1.split(X_df, y_series)
        X_train2, X_test2, y_train2, y_test2 = sampler2.split(X_df, y_series)

        # Should produce identical splits
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)
