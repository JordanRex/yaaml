"""
Unit tests for missing value imputation module
"""

import numpy as np
import pandas as pd
import pytest

from yaaml.miss_imputation import DataFrameImputer


class TestDataFrameImputer:
    """Test cases for DataFrameImputer"""

    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values"""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "numeric_1": [1.0, 2.0, np.nan, 4.0, 5.0],
                "numeric_2": [10.0, np.nan, 30.0, 40.0, 50.0],
                "categorical_1": ["A", "B", np.nan, "C", "A"],
                "categorical_2": ["X", "Y", "Z", np.nan, "X"],
            }
        )
        return data

    def test_init_default(self):
        """Test initialization with default parameters"""
        imputer = DataFrameImputer()
        assert imputer.strategy == "mean"
        assert imputer.num_imputer is None
        assert imputer.cat_imputer is None

    def test_init_custom_strategy(self):
        """Test initialization with custom strategy"""
        imputer = DataFrameImputer(strategy="median")
        assert imputer.strategy == "median"

    def test_fit(self, sample_data_with_missing):
        """Test fitting the imputer"""
        imputer = DataFrameImputer()
        imputer.fit(sample_data_with_missing)

        assert imputer.num_imputer is not None
        assert imputer.cat_imputer is not None
        assert len(imputer.numeric_columns) == 2
        assert len(imputer.categorical_columns) == 2

    def test_transform(self, sample_data_with_missing):
        """Test transforming data"""
        imputer = DataFrameImputer()
        imputer.fit(sample_data_with_missing)

        result = imputer.transform(sample_data_with_missing)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_data_with_missing.shape

        # Check that no NaN values remain
        assert not np.isnan(result).any()

    def test_fit_transform(self, sample_data_with_missing):
        """Test fit_transform method"""
        imputer = DataFrameImputer()
        result = imputer.fit_transform(sample_data_with_missing)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_data_with_missing.shape
        assert not np.isnan(result).any()

    def test_fancy_impute_knn(self, sample_data_with_missing):
        """Test KNN imputation"""
        result = DataFrameImputer.fancy_impute(
            sample_data_with_missing, which_method="KNN"
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_data_with_missing.shape

    def test_fancy_impute_iterative(self, sample_data_with_missing):
        """Test Iterative imputation"""
        result = DataFrameImputer.fancy_impute(
            sample_data_with_missing, which_method="IterativeImputer"
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_data_with_missing.shape

    def test_fancy_impute_unknown_method(self, sample_data_with_missing):
        """Test that unknown method defaults to KNN"""
        result = DataFrameImputer.fancy_impute(
            sample_data_with_missing, which_method="UnknownMethod"
        )

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_no_missing_data(self):
        """Test behavior with no missing data"""
        data = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": ["A", "B", "C", "D", "E"]}
        )

        imputer = DataFrameImputer()
        result = imputer.fit_transform(data)

        assert result is not None
        assert result.shape == data.shape

    def test_only_numeric_data(self):
        """Test with only numeric data"""
        data = pd.DataFrame(
            {"col1": [1.0, 2.0, np.nan, 4.0], "col2": [10.0, np.nan, 30.0, 40.0]}
        )

        imputer = DataFrameImputer()
        result = imputer.fit_transform(data)

        assert result is not None
        assert not np.isnan(result).any()

    def test_only_categorical_data(self):
        """Test with only categorical data"""
        data = pd.DataFrame(
            {"col1": ["A", "B", np.nan, "C"], "col2": ["X", np.nan, "Z", "W"]}
        )

        imputer = DataFrameImputer()
        result = imputer.fit_transform(data)

        assert result is not None
        assert result.shape == data.shape

    def test_different_strategies(self, sample_data_with_missing):
        """Test different imputation strategies"""
        strategies = ["mean", "median", "most_frequent"]

        for strategy in strategies:
            imputer = DataFrameImputer(strategy=strategy)
            result = imputer.fit_transform(sample_data_with_missing)

            assert result is not None
            assert not np.isnan(result).any()
