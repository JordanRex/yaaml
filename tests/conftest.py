"""
Test configuration and fixtures for YAAML tests
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


@pytest.fixture
def sample_classification_data():
    """Generate sample classification dataset"""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_regression_data():
    """Generate sample regression dataset"""
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=7,
        noise=0.1,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_categorical_data():
    """Generate sample dataset with categorical features"""
    np.random.seed(42)

    n_samples = 500

    # Create mixed data
    data = {
        "numeric_1": np.random.normal(0, 1, n_samples),
        "numeric_2": np.random.uniform(-1, 1, n_samples),
        "category_1": np.random.choice(["A", "B", "C"], n_samples),
        "category_2": np.random.choice(["X", "Y", "Z", "W"], n_samples),
        "target": np.random.randint(0, 2, n_samples),
    }

    df = pd.DataFrame(data)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_missing_data():
    """Generate sample dataset with missing values"""
    np.random.seed(42)

    n_samples = 500

    data = {
        "col1": np.random.normal(0, 1, n_samples),
        "col2": np.random.uniform(-1, 1, n_samples),
        "col3": np.random.choice(["A", "B", "C"], n_samples),
        "target": np.random.randint(0, 2, n_samples),
    }

    df = pd.DataFrame(data)

    # Introduce missing values
    missing_indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[missing_indices[:25], "col1"] = np.nan
    df.loc[missing_indices[25:], "col3"] = np.nan

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
