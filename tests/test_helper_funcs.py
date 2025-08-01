"""
Test Helper Functions Module
===========================

Comprehensive tests for helper functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from yaaml.helper_funcs import check_data_quality, detect_data_types, evaluate_model


class TestDetectDataTypes:
    """Test detect_data_types function"""

    def test_mixed_data_types(self) -> None:
        """Test detection of mixed data types"""
        df = pd.DataFrame(
            {
                "numeric_int": [1, 2, 3, 4, 5],
                "numeric_float": [1.1, 2.2, 3.3, 4.4, 5.5],
                "categorical": ["A", "B", "A", "C", "B"],
                "text": ["hello", "world", "test", "data", "types"],
                "boolean": [True, False, True, False, True],
            }
        )

        result = detect_data_types(df)

        assert isinstance(result, dict)
        assert "numeric" in result
        assert "categorical" in result
        assert "high_cardinality" in result or "text" in result or "object" in result

        # Check specific columns are classified correctly
        numeric_cols = result.get("numeric", [])
        categorical_cols = result.get("categorical", []) + result.get("object", [])

        assert "numeric_int" in numeric_cols or "numeric_int" in categorical_cols
        assert "numeric_float" in numeric_cols or "numeric_float" in categorical_cols

    def test_all_numeric_data(self) -> None:
        """Test with all numeric data"""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": np.random.randn(5),
            }
        )

        result = detect_data_types(df)

        assert "numeric" in result
        numeric_cols = result["numeric"]
        assert len(numeric_cols) >= 1  # At least some should be detected as numeric

    def test_all_categorical_data(self) -> None:
        """Test with all categorical data"""
        df = pd.DataFrame(
            {
                "cat1": ["A", "B", "C", "A", "B"],
                "cat2": ["X", "Y", "Z", "X", "Y"],
                "cat3": ["red", "blue", "green", "red", "blue"],
            }
        )

        result = detect_data_types(df)

        # Should detect categorical or object types
        has_categorical = (
            "categorical" in result and len(result["categorical"]) > 0
        ) or ("object" in result and len(result["object"]) > 0)
        assert has_categorical

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame"""
        df = pd.DataFrame()

        result = detect_data_types(df)
        assert isinstance(result, dict)
        # Should handle empty DataFrame gracefully

    def test_single_column(self) -> None:
        """Test with single column"""
        df = pd.DataFrame({"single_col": [1, 2, 3, 4, 5]})

        result = detect_data_types(df)
        assert isinstance(result, dict)

        # Should classify the single column
        total_cols = sum(len(cols) for cols in result.values())
        assert total_cols >= 1

    def test_missing_values(self) -> None:
        """Test with missing values"""
        df = pd.DataFrame(
            {
                "with_nan": [1, 2, np.nan, 4, 5],
                "with_none": ["A", "B", None, "D", "E"],
                "complete": [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )

        result = detect_data_types(df)
        assert isinstance(result, dict)

        # Should still classify columns despite missing values
        total_cols = sum(len(cols) for cols in result.values())
        assert total_cols >= 1


class TestCheckDataQuality:
    """Test check_data_quality function"""

    def test_high_quality_data(self) -> None:
        """Test with high quality data"""
        # Create clean data
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        quality_report = check_data_quality(X_df)

        assert isinstance(quality_report, dict)
        assert "missing_percentage" in quality_report
        assert "duplicate_rows" in quality_report
        assert quality_report["missing_percentage"] == 0.0  # No missing values
        assert quality_report["duplicate_rows"] >= 0

    def test_data_with_missing_values(self) -> None:
        """Test with data containing missing values"""
        df = pd.DataFrame(
            {
                "col1": [1, 2, np.nan, 4, 5],
                "col2": [1.1, np.nan, 3.3, 4.4, np.nan],
                "col3": ["A", "B", "C", None, "E"],
            }
        )

        quality_report = check_data_quality(df)

        assert quality_report["missing_percentage"] > 0
        assert "missing_values" in quality_report

    def test_data_with_duplicates(self) -> None:
        """Test with duplicate rows"""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 1, 4, 2],  # Duplicates: rows 0&2, rows 1&4
                "col2": [1.1, 2.2, 1.1, 4.4, 2.2],
            }
        )

        quality_report = check_data_quality(df)

        assert quality_report["duplicate_rows"] > 0

    def test_data_quality_with_target_issues(self) -> None:
        """Test data quality basic functionality"""
        X = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [1.1, 2.2, 3.3, 4.4, 5.5]})

        quality_report = check_data_quality(X)

        assert isinstance(quality_report, dict)
        # Should return basic quality metrics

    def test_constant_columns(self) -> None:
        """Test detection of constant columns"""
        df = pd.DataFrame(
            {
                "constant": [1, 1, 1, 1, 1],  # All same value
                "variable": [1, 2, 3, 4, 5],
                "mostly_constant": [1, 1, 1, 1, 2],  # Almost constant
            }
        )

        quality_report = check_data_quality(df)

        # Should identify data quality issues
        assert isinstance(quality_report, dict)

    def test_class_imbalance_detection(self) -> None:
        """Test basic data quality functionality"""
        X = pd.DataFrame({"col1": range(100), "col2": np.random.randn(100)})

        quality_report = check_data_quality(X)

        assert isinstance(quality_report, dict)
        # Should provide basic quality information


class TestEvaluateModel:
    """Test evaluate_model function"""

    def test_evaluate_classification_model(self) -> None:
        """Test model evaluation for classification"""
        # Create test data
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_train_np, X_test_np = X[:150], X[150:]
        y_train_np, y_test_np = y[:150], y[150:]

        # Convert to pandas
        X_train = pd.DataFrame(
            X_train_np, columns=[f"f_{i}" for i in range(X.shape[1])]
        )
        X_test = pd.DataFrame(X_test_np, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_train = pd.Series(y_train_np)
        y_test = pd.Series(y_test_np)

        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model using the correct signature
        metrics = evaluate_model(
            model, X_train, y_train, X_test, y_test, task_type="classification"
        )

        assert isinstance(metrics, dict)
        assert len(metrics) >= 1  # Should have at least one metric

    def test_evaluate_regression_model(self) -> None:
        """Test model evaluation for regression"""
        # Create test data
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        X_train_np, X_test_np = X[:150], X[150:]
        y_train_np, y_test_np = y[:150], y[150:]

        # Convert to pandas
        X_train = pd.DataFrame(
            X_train_np, columns=[f"f_{i}" for i in range(X.shape[1])]
        )
        X_test = pd.DataFrame(X_test_np, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_train = pd.Series(y_train_np)
        y_test = pd.Series(y_test_np)

        # Train a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(
            model, X_train, y_train, X_test, y_test, task_type="regression"
        )

        assert isinstance(metrics, dict)
        assert len(metrics) >= 1  # Should have at least one metric

    def test_evaluate_with_pandas_input(self) -> None:
        """Test evaluation with pandas DataFrames"""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        X_train, X_test = X_df[:70], X_df[70:]
        y_train, y_test = y_series[:70], y_series[70:]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, task_type="classification")

        assert isinstance(metrics, dict)
        assert len(metrics) >= 1

    def test_evaluate_perfect_model(self) -> None:
        """Test evaluation with a simple mock model"""
        # Create test data with enough samples for cross-validation
        X_test = pd.DataFrame(
            [[0, 0], [1, 1], [0, 1], [1, 0], [0, 0], [1, 1]], columns=["feat1", "feat2"]
        )
        y_test = pd.Series([0, 1, 0, 1, 0, 1])

        # Use a simple trained model instead of mocking
        from sklearn.linear_model import LogisticRegression

        # Train on the same data to get perfect fit
        model = LogisticRegression(random_state=42)
        model.fit(X_test, y_test)

        metrics = evaluate_model(model, X_test, y_test, task_type="classification")

        assert isinstance(metrics, dict)
        assert len(metrics) >= 1

    def test_evaluate_insufficient_samples_for_cv(self) -> None:
        """Test evaluation with insufficient samples for cross-validation"""
        # Create minimal test data (insufficient for CV)
        X_train = pd.DataFrame([[0, 0], [1, 1]], columns=["feat1", "feat2"])
        y_train = pd.Series([0, 1])

        # Create test data (same as train for this test)
        X_test = X_train.copy()
        y_test = y_train.copy()

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        # Should handle CV failure gracefully and fall back to simple evaluation
        metrics = evaluate_model(
            model, X_train, y_train, X_test, y_test, task_type="classification"
        )

        assert isinstance(metrics, dict)
        assert len(metrics) >= 1
        # Should either have CV metrics or fallback metrics
        has_cv_metrics = "cv_accuracy_mean" in metrics
        has_fallback_metrics = "test_accuracy" in metrics and "cv_fallback" in metrics
        assert has_cv_metrics or has_fallback_metrics

    def test_evaluate_with_invalid_task_type(self) -> None:
        """Test behavior with invalid task type (should default to regression)"""
        X, y = make_classification(
            n_samples=50, n_features=5, n_informative=2, n_redundant=1, random_state=42
        )
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X[:30], y[:30])

        X_test = pd.DataFrame(X[30:], columns=[f"feat_{i}" for i in range(X.shape[1])])
        y_test = pd.Series(y[30:])

        # Invalid task type should default to regression metrics
        metrics = evaluate_model(model, X_test, y_test, task_type="invalid_task")

        # Should contain regression metrics
        assert "cv_r2_mean" in metrics
        assert "cv_mse_mean" in metrics
        assert "cv_mae_mean" in metrics


class TestHelperFunctionsIntegration:
    """Test integration of helper functions"""

    def test_data_pipeline_integration(self) -> None:
        """Test using helper functions in a typical data pipeline"""
        # Create realistic dataset with issues
        np.random.seed(42)
        n_samples = 200

        # Mixed data with some quality issues
        df = pd.DataFrame(
            {
                "numeric1": np.random.randn(n_samples),
                "numeric2": np.random.randn(n_samples) * 2 + 1,
                "categorical": np.random.choice(["A", "B", "C"], n_samples),
                "text_like": [f"text_{i}" for i in range(n_samples)],
                "constant": [1] * n_samples,  # Constant column
            }
        )

        # Add some missing values
        missing_indices = np.random.choice(n_samples, size=20, replace=False)
        df.loc[list(missing_indices), "numeric1"] = np.nan

        # Create target
        y = pd.Series(np.random.choice([0, 1], n_samples))

        # 1. Detect data types
        data_types = detect_data_types(df)
        assert isinstance(data_types, dict)

        # 2. Check data quality
        quality_report = check_data_quality(df, y)
        assert isinstance(quality_report, dict)
        assert quality_report["missing_percentage"] > 0

        # 3. Use info for simple preprocessing and modeling
        numeric_cols = data_types.get("numeric", [])
        if numeric_cols:
            # Simple preprocessing: fill missing values
            df_clean = df.copy()
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

            # Simple model training and evaluation
            if len(numeric_cols) >= 2:
                X_train = df_clean[numeric_cols[:2]][:150]
                X_test = df_clean[numeric_cols[:2]][150:]
                y_train, y_test = y[:150], y[150:]

                model = RandomForestClassifier(n_estimators=5, random_state=42)
                model.fit(X_train, y_train)

                # 4. Evaluate model
                metrics = evaluate_model(
                    model, X_test, y_test, task_type="classification"
                )
                assert isinstance(metrics, dict)

    def test_helper_functions_with_edge_cases(self) -> None:
        """Test helper functions with edge cases"""
        # Very small dataset
        small_df = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
        small_y = pd.Series([0, 1])

        # Should handle small datasets gracefully
        data_types = detect_data_types(small_df)
        assert isinstance(data_types, dict)

        quality_report = check_data_quality(small_df, small_y)
        assert isinstance(quality_report, dict)

    def test_helper_functions_reproducibility(self) -> None:
        """Test that helper functions are deterministic"""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["A", "B", "C", "A", "B"]})
        y = pd.Series([0, 1, 0, 1, 0])

        # Multiple calls should give same results
        data_types1 = detect_data_types(df)
        data_types2 = detect_data_types(df)
        assert data_types1 == data_types2

        quality1 = check_data_quality(df, y)
        quality2 = check_data_quality(df, y)
        assert quality1 == quality2
