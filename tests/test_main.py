"""
Unit tests for YAAML AutoML main functionality
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from yaaml import YAAMLAutoML


class TestYAAMLAutoML:
    """Test cases for YAAMLAutoML class"""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters"""
        automl = YAAMLAutoML()

        assert automl.random_seed == 42
        assert automl.max_evals == 10
        assert automl.cv_folds == 3
        assert automl.mode == "classification"
        assert automl.verbosity == 1

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters"""
        automl = YAAMLAutoML(
            random_seed=123,
            max_evals=20,
            cv_folds=5,
            mode="regression",
            verbosity=0,
        )

        assert automl.random_seed == 123
        assert automl.max_evals == 20
        assert automl.cv_folds == 5
        assert automl.mode == "regression"
        assert automl.verbosity == 0

    def test_fit_classification(self, sample_classification_data: Any) -> None:
        """Test fitting on classification data"""
        X_train, X_test, y_train, y_test = sample_classification_data

        automl = YAAMLAutoML(mode="classification", verbosity=0)
        automl.fit(X_train, y_train)

        assert automl.model is not None
        assert automl.X_train is not None
        assert automl.y_train is not None
        # After feature engineering, we should have more features
        assert automl.feature_names is not None
        # Original features should still be present
        assert all(col in automl.feature_names for col in X_train.columns)
        # Remove duplicate line
        assert len(automl.feature_names) >= len(X_train.columns)

    def test_fit_regression(self, sample_regression_data: Any) -> None:
        """Test fitting on regression data"""
        X_train, X_test, y_train, y_test = sample_regression_data

        automl = YAAMLAutoML(mode="regression", verbosity=0)
        automl.fit(X_train, y_train)

        assert automl.model is not None
        assert automl.X_train is not None
        assert automl.y_train is not None
        assert automl.feature_names is not None
        # After feature engineering, should have more features
        assert len(automl.feature_names) >= len(X_train.columns)
        # Original features should still be present
        assert all(col in automl.feature_names for col in X_train.columns)

    def test_predict_classification(self, sample_classification_data: Any) -> None:
        """Test prediction on classification data"""
        X_train, X_test, y_train, y_test = sample_classification_data

        automl = YAAMLAutoML(mode="classification", verbosity=0)
        automl.fit(X_train, y_train)

        predictions = automl.predict(X_test)

        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_regression(self, sample_regression_data: Any) -> None:
        """Test prediction on regression data"""
        X_train, X_test, y_train, y_test = sample_regression_data

        automl = YAAMLAutoML(mode="regression", verbosity=0)
        automl.fit(X_train, y_train)

        predictions = automl.predict(X_test)

        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert all(isinstance(pred, int | float | np.number) for pred in predictions)

    def test_predict_proba_classification(
        self, sample_classification_data: Any
    ) -> None:
        """Test probability prediction on classification data"""
        X_train, X_test, y_train, y_test = sample_classification_data

        automl = YAAMLAutoML(mode="classification", verbosity=0)
        automl.fit(X_train, y_train)

        probabilities = automl.predict_proba(X_test)

        assert probabilities is not None
        prob_shape = getattr(probabilities, "shape", None)
        assert prob_shape is not None
        assert prob_shape[0] == len(X_test)
        assert prob_shape[1] == 2  # Binary classification
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_predict_proba_regression_error(self, sample_regression_data: Any) -> None:
        """Test that predict_proba raises error for regression"""
        X_train, X_test, y_train, y_test = sample_regression_data

        automl = YAAMLAutoML(mode="regression", verbosity=0)
        automl.fit(X_train, y_train)

        with pytest.raises(
            ValueError,
            match="predict_proba is not supported for regression tasks",
        ):
            automl.predict_proba(X_test)

    def test_score_classification(self, sample_classification_data: Any) -> None:
        """Test scoring on classification data"""
        X_train, X_test, y_train, y_test = sample_classification_data

        automl = YAAMLAutoML(mode="classification", verbosity=0)
        automl.fit(X_train, y_train)

        score = automl.score(X_test, y_test)

        assert isinstance(score, int | float | np.number)
        assert 0.0 <= score <= 1.0  # Accuracy should be between 0 and 1

    def test_score_regression(self, sample_regression_data: Any) -> None:
        """Test scoring on regression data"""
        X_train, X_test, y_train, y_test = sample_regression_data

        automl = YAAMLAutoML(mode="regression", verbosity=0)
        automl.fit(X_train, y_train)

        score = automl.score(X_test, y_test)

        assert isinstance(score, int | float | np.number)
        # R2 score can be negative, so just check it's a reasonable range
        assert -2.0 <= score <= 1.0

    def test_feature_importance(self, sample_classification_data: Any) -> None:
        """Test feature importance extraction"""
        X_train, X_test, y_train, y_test = sample_classification_data

        automl = YAAMLAutoML(mode="classification", verbosity=0)
        automl.fit(X_train, y_train)

        importance = automl.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        # Feature engineering creates additional features
        assert len(importance) >= len(X_train.columns)
        assert all(importance["importance"] >= 0)  # Non-negative

    def test_predict_without_fit_error(self, sample_classification_data: Any) -> None:
        """Test that predict raises error when model is not fitted"""
        X_train, X_test, y_train, y_test = sample_classification_data

        automl = YAAMLAutoML(mode="classification", verbosity=0)

        with pytest.raises(ValueError, match="Model has not been trained"):
            automl.predict(X_test)

    def test_score_without_fit_error(self, sample_classification_data: Any) -> None:
        """Test that score raises error when model is not fitted"""
        X_train, X_test, y_train, y_test = sample_classification_data

        automl = YAAMLAutoML(mode="classification", verbosity=0)

        with pytest.raises(ValueError, match="Model has not been trained"):
            automl.score(X_test, y_test)

    def test_feature_importance_without_fit_error(
        self, sample_classification_data: Any
    ) -> None:
        """Test that get_feature_importance raises error when model is
        not fitted"""
        X_train, X_test, y_train, y_test = sample_classification_data

        automl = YAAMLAutoML(mode="classification", verbosity=0)

        with pytest.raises(ValueError, match="Model has not been trained"):
            automl.get_feature_importance()

    def test_categorical_data_handling(self, sample_categorical_data: Any) -> None:
        """Test handling of categorical data"""
        X_train, X_test, y_train, y_test = sample_categorical_data

        automl = YAAMLAutoML(mode="classification", verbosity=0)
        automl.fit(X_train, y_train)

        predictions = automl.predict(X_test)

        assert predictions is not None
        assert len(predictions) == len(X_test)

    def test_missing_data_handling(self, sample_missing_data: Any) -> None:
        """Test handling of missing data"""
        X_train, X_test, y_train, y_test = sample_missing_data

        automl = YAAMLAutoML(mode="classification", verbosity=0)
        automl.fit(X_train, y_train)

        predictions = automl.predict(X_test)

        assert predictions is not None
        assert len(predictions) == len(X_test)
