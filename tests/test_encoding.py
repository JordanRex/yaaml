"""
Unit tests for encoding module
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from yaaml.encoding import NativeEncoder, TargetEncoder, encode_categorical_features


@pytest.fixture
def sample_data():
    """Create sample data with categorical features."""
    # Create numeric data
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])

    # Add categorical features
    X_df["cat_ordered"] = np.random.choice(["low", "medium", "high"], size=100)
    X_df["cat_nominal"] = np.random.choice(["A", "B", "C", "D"], size=100)
    X_df["cat_binary"] = np.random.choice(["yes", "no"], size=100)

    y_series = pd.Series(y, name="target")

    return train_test_split(X_df, y_series, test_size=0.3, random_state=42)


class TestNativeEncoder:
    """Test cases for NativeEncoder class."""

    def test_ordinal_encoding(self, sample_data):
        """Test ordinal encoding."""
        X_train, X_test, y_train, y_test = sample_data

        encoder = NativeEncoder(encoding_method="ordinal")
        encoder.fit(X_train)
        X_encoded = encoder.transform(X_test)

        # Check that categorical columns are encoded as numeric
        categorical_cols = X_train.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            assert pd.api.types.is_numeric_dtype(
                X_encoded[col]
            ), f"Column {col} should be numeric encoded"

        # Check that numeric columns are unchanged
        numeric_cols = X_train.select_dtypes(exclude=["object"]).columns
        for col in numeric_cols:
            assert (
                X_encoded[col].dtype == X_train[col].dtype
            ), f"Numeric column {col} should be unchanged"

    def test_onehot_encoding(self, sample_data):
        """Test one-hot encoding."""
        X_train, X_test, y_train, y_test = sample_data

        encoder = NativeEncoder(encoding_method="onehot")
        encoder.fit(X_train)
        X_encoded = encoder.transform(X_test)

        # Check that we have more columns after one-hot encoding
        assert (
            X_encoded.shape[1] > X_train.shape[1]
        ), "One-hot encoding should increase number of columns"

        # Check that all values are 0 or 1 for new columns
        categorical_cols = X_train.select_dtypes(include=["object"]).columns
        original_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()
        new_cols = [col for col in X_encoded.columns if col not in original_cols]

        for col in new_cols:
            unique_vals = X_encoded[col].unique()
            assert all(
                val in [0, 1] for val in unique_vals
            ), f"One-hot column {col} should only contain 0 and 1"

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        X_train, X_test, y_train, y_test = sample_data

        encoder = NativeEncoder(encoding_method="ordinal")
        X_encoded1 = encoder.fit_transform(X_train)

        encoder2 = NativeEncoder(encoding_method="ordinal")
        encoder2.fit(X_train)
        X_encoded2 = encoder2.transform(X_train)

        pd.testing.assert_frame_equal(X_encoded1, X_encoded2)


class TestTargetEncoder:
    """Test cases for TargetEncoder class."""

    def test_target_encoding(self, sample_data):
        """Test target encoding."""
        X_train, X_test, y_train, y_test = sample_data

        encoder = TargetEncoder()
        encoder.fit(X_train, y_train)
        X_encoded = encoder.transform(X_test)

        # Check that categorical columns are encoded as floats
        categorical_cols = X_train.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            assert X_encoded[col].dtype in [
                "float64",
                "float32",
            ], f"Target encoded column {col} should be float"

        # Check that numeric columns are unchanged
        numeric_cols = X_train.select_dtypes(exclude=["object"]).columns
        for col in numeric_cols:
            assert (
                X_encoded[col].dtype == X_train[col].dtype
            ), f"Numeric column {col} should be unchanged"

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        X_train, X_test, y_train, y_test = sample_data

        encoder = TargetEncoder()
        X_encoded1 = encoder.fit_transform(X_train, y_train)

        encoder2 = TargetEncoder()
        encoder2.fit(X_train, y_train)
        X_encoded2 = encoder2.transform(X_train)

        pd.testing.assert_frame_equal(X_encoded1, X_encoded2)


class TestEncodeCategoricalFeatures:
    """Test cases for encode_categorical_features function."""

    def test_encode_categorical_features_ordinal(self, sample_data):
        """Test the main encoding function with ordinal encoding."""
        X_train, X_test, y_train, y_test = sample_data

        result = encode_categorical_features(
            X_train, X_test, target=y_train, method="ordinal"
        )

        X_train_enc, X_test_enc = result

        # Check shapes are preserved
        assert X_train_enc.shape[0] == X_train.shape[0]
        assert X_test_enc.shape[0] == X_test.shape[0]

    def test_encode_categorical_features_target(self, sample_data):
        """Test the main encoding function with target encoding."""
        X_train, X_test, y_train, y_test = sample_data

        result = encode_categorical_features(
            X_train, X_test, target=y_train, method="target"
        )

        X_train_enc, X_test_enc = result

        # Check shapes are preserved
        assert X_train_enc.shape[0] == X_train.shape[0]
        assert X_test_enc.shape[0] == X_test.shape[0]

    def test_invalid_encoding_method(self, sample_data):
        """Test error handling for invalid encoding method."""
        X_train, X_test, y_train, y_test = sample_data

        with pytest.raises(ValueError, match="Unsupported encoding method"):
            encode_categorical_features(
                X_train, X_test, target=y_train, method="invalid_method"
            )
