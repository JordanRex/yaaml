"""
Comprehensive Module Tests for Python 3.12+ Features
===================================================

Tests all individual modules with focus on Python 3.12+ features,
modern type annotations, and integration capabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from yaaml.encoding import NativeEncoder, TargetEncoder
from yaaml.feature_engineering import BinningTransformer, FeatureEngineering
from yaaml.feature_selection import FeatureSelector, select_features
from yaaml.helper_funcs import check_data_quality, detect_data_types, evaluate_model
from yaaml.miss_imputation import DataFrameImputer
from yaaml.native_algorithms import AlgorithmFactory, NativeAlgorithmSelector
from yaaml.sampling import NativeSampler, StratifiedSampler, apply_sampling


class TestModernTypingIntegration:
    """Test that all modules work with modern Python 3.12+ typing"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "numeric_1": np.random.randn(100),
                "numeric_2": np.random.randn(100),
                "category_1": np.random.choice(["A", "B", "C"], size=100),
                "category_2": np.random.choice(["X", "Y"], size=100),
            }
        )
        target = pd.Series(np.random.choice([0, 1], size=100))
        return df, target

    def test_sampling_module_typing(self, sample_data):
        """Test sampling module with modern type annotations"""
        X, y = sample_data
        X_numeric = X.select_dtypes(include=[np.number])

        # Test NativeSampler with union return types
        sampler = NativeSampler(strategy="oversample")
        X_resampled, y_resampled = sampler.fit_resample(X_numeric, y)

        # Verify tuple return type works
        assert isinstance(X_resampled, pd.DataFrame)
        assert isinstance(y_resampled, pd.Series)

        # Test StratifiedSampler
        splitter = StratifiedSampler(test_size=0.2)
        X_train, X_test, y_train, y_test = splitter.split(X_numeric, y)

        # Verify tuple[DataFrame, DataFrame, Series, Series] works
        assert all(
            isinstance(item, (pd.DataFrame, pd.Series))
            for item in [X_train, X_test, y_train, y_test]
        )

        # Test function with union return type
        result = apply_sampling(X_numeric, y, strategy="undersample")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_encoding_module_typing(self, sample_data):
        """Test encoding module with optional parameters"""
        X, y = sample_data

        # Test NativeEncoder with modern constructor
        encoder = NativeEncoder(encoding_method="onehot")
        encoded = encoder.fit_transform(X)

        assert isinstance(encoded, pd.DataFrame)
        assert encoded.shape[0] == X.shape[0]

        # Test TargetEncoder
        target_encoder = TargetEncoder()
        target_encoded = target_encoder.fit_transform(X, y)

        assert isinstance(target_encoded, pd.DataFrame)

    def test_imputation_module_typing(self, sample_data):
        """Test imputation with modern typing"""
        X, y = sample_data

        # Add missing values
        X_missing = X.copy()
        X_missing.loc[list(range(0, 10)), "numeric_1"] = np.nan
        X_missing.loc[list(range(5, 15)), "category_1"] = np.nan

        # Test DataFrameImputer
        imputer = DataFrameImputer(strategy="mean")
        imputed = imputer.fit_transform(X_missing)

        assert isinstance(imputed, pd.DataFrame)
        assert imputed.shape == X_missing.shape
        assert imputed.isnull().sum().sum() < X_missing.isnull().sum().sum()

    def test_feature_engineering_typing(self, sample_data):
        """Test feature engineering with modern annotations"""
        X, y = sample_data
        X_numeric = X.select_dtypes(include=[np.number])

        # Test FeatureEngineering with list[str] parameters
        fe = FeatureEngineering(decomposition_methods=["pca"], clustering_features=True)

        engineered = fe.fit_transform(X_numeric)
        assert isinstance(engineered, pd.DataFrame)
        assert engineered.shape[1] >= X_numeric.shape[1]

        # Test BinningTransformer
        binner = BinningTransformer(n_bins=3)
        binned = binner.fit_transform(X_numeric)

        assert isinstance(binned, pd.DataFrame)

    def test_feature_selection_typing(self, sample_data):
        """Test feature selection with modern types"""
        X, y = sample_data

        # Encode categorical data first for feature selection
        from yaaml.encoding import NativeEncoder

        encoder = NativeEncoder()
        X_encoded = encoder.fit_transform(X)

        # Test FeatureSelector with list[str] methods
        selector = FeatureSelector(methods=["variance", "univariate"])
        selector.fit(X_encoded, y)
        selected = selector.transform(X_encoded)

        assert isinstance(selected, pd.DataFrame)
        assert selected.shape[1] <= X_encoded.shape[1]

        # Test convenience function with union return type
        result = select_features(X_encoded, target=y, methods=["variance"])
        assert isinstance(result, pd.DataFrame)

    def test_algorithms_module_typing(self, sample_data):
        """Test algorithm selection with modern typing"""
        X, y = sample_data
        X_numeric = X.select_dtypes(include=[np.number])

        # Test AlgorithmFactory
        factory = AlgorithmFactory()
        available_algos = factory.get_available_algorithms()

        assert isinstance(available_algos, list)
        assert all(isinstance(algo, str) for algo in available_algos)

        # Test NativeAlgorithmSelector
        selector = NativeAlgorithmSelector()
        result = selector.find_best_algorithm(
            X_numeric, y, algorithms=["random_forest"], max_evals_per_algo=1
        )

        assert isinstance(result, dict)
        assert "best_algorithm" in result

    def test_helper_functions_typing(self, sample_data):
        """Test helper functions with modern annotations"""
        X, y = sample_data

        # Test detect_data_types with dict[str, list[str]] return
        data_types = detect_data_types(X)
        assert isinstance(data_types, dict)
        assert all(
            isinstance(k, str) and isinstance(v, list) for k, v in data_types.items()
        )

        # Test check_data_quality
        quality_info = check_data_quality(X)
        assert isinstance(quality_info, dict)

        # Train a simple model for evaluation testing
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_numeric = X.select_dtypes(include=[np.number])
        model.fit(X_numeric, y)

        # Test evaluate_model with dict[str, float] return
        metrics = evaluate_model(
            model, X_numeric, y, task_type="classification", cv_folds=2
        )
        assert isinstance(metrics, dict)
        assert all(
            isinstance(k, str) and isinstance(v, (int, float))
            for k, v in metrics.items()
        )


class TestWalrusOperatorIntegration:
    """Test walrus operator usage in realistic scenarios"""

    def test_walrus_in_data_processing(self):
        """Test walrus operator in data processing pipelines"""
        # Create test data
        data = pd.DataFrame({"values": [1, 2, 3, 4, 5, None, 7, 8, 9, 10]})

        # Use walrus operator for efficient processing
        processed_data = []
        for idx, row in data.iterrows():
            if (value := row["values"]) is not None and value > 5:
                processed_data.append(value)

        assert processed_data == [7, 8, 9, 10]

    def test_walrus_in_model_selection(self):
        """Test walrus operator in model evaluation loops"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        # Create test data
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_df = pd.DataFrame(X)

        models = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            LogisticRegression(random_state=42, max_iter=100),
        ]

        best_score = 0
        best_model = None

        for model in models:
            model.fit(X_df, y)
            if (score := accuracy_score(y, model.predict(X_df))) > best_score:
                best_score = score
                best_model = model

        assert best_model is not None
        assert 0.5 <= best_score <= 1.0


class TestUnionTypeUsage:
    """Test union type usage throughout the codebase"""

    def test_optional_dataframe_handling(self):
        """Test functions that accept DataFrame | None"""

        def process_optional_df(df: pd.DataFrame | None = None) -> int:
            if df is not None:
                return len(df)
            return 0

        # Test with DataFrame
        test_df = pd.DataFrame({"a": [1, 2, 3]})
        assert process_optional_df(test_df) == 3

        # Test with None
        assert process_optional_df(None) == 0
        assert process_optional_df() == 0

    def test_multiple_type_unions(self):
        """Test complex union types"""

        def flexible_input(data: int | float | str | pd.Series) -> str:
            if isinstance(data, (int, float)):
                return f"number: {data}"
            elif isinstance(data, str):
                return f"string: {data}"
            elif isinstance(data, pd.Series):
                return f"series: length {len(data)}"
            return "unknown"

        assert flexible_input(42) == "number: 42"
        assert flexible_input(3.14) == "number: 3.14"
        assert flexible_input("test") == "string: test"
        assert flexible_input(pd.Series([1, 2, 3])) == "series: length 3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
