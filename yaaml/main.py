"""##############################################################################
Author - Varun Rajan
Package - yaaml 0.1.0 - Native Implementation
Description - Pure Python/sklearn AutoML without external dependencies
##############################################################################"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import encoding as encoders
from . import feature_engineering as feateng
from . import feature_selection as featsel
from . import miss_imputation as missimp
from . import sampling as sampler

# Native implementations - sklearn-only AutoML
from .native_algorithms import NativeAlgorithmSelector

# ########################################################################
# NATIVE AUTOML PIPELINE - Pure Python/sklearn implementation
# ########################################################################


class YAAMLAutoML:
    """
    Native AutoML implementation without external dependencies like H2O or hyperopt.
    Uses sklearn and custom implementations for all components.
    """

    def __init__(
        self,
        random_seed: int = 42,
        max_evals: int = 10,
        cv_folds: int = 3,
        mode: str = "classification",  # 'classification' or 'regression'
        verbosity: int = 1,
        # Preprocessing options
        imputation_strategy: str = "mean",
        encoding_method: str = "ordinal",
        feature_engineering: bool = True,
        feature_selection: bool = True,
        sampling_strategy: str = "auto",
    ):
        """Initialize the AutoML pipeline with configuration parameters."""
        self.random_seed = random_seed
        self.max_evals = max_evals
        self.cv_folds = cv_folds
        self.mode = mode
        self.verbosity = verbosity

        # Preprocessing configuration
        self.imputation_strategy = imputation_strategy
        self.encoding_method = encoding_method
        self.feature_engineering = feature_engineering
        self.feature_selection = feature_selection
        self.sampling_strategy = sampling_strategy

        # Pipeline components - will be initialized when fit is called
        self.imputer = None
        self.encoder = None
        self.feature_engineer = None
        self.feature_selector = None
        self.sampler = None
        self.model = None
        self.best_params = None
        self.best_score = None

        # Training data references
        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.preprocessing_fitted = False

        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)

    def fit(self, X, y, X_valid=None, y_valid=None):
        """
        Fit the AutoML pipeline to training data.

        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series or np.array
            Training target
        X_valid : pd.DataFrame, optional
            Validation features
        y_valid : pd.Series or np.array, optional
            Validation target
        """
        if self.verbosity > 0:
            print("Starting YAAML AutoML pipeline...")
            print(f"Training data shape: {X.shape}")
            print(f"Mode: {self.mode}")

        # Store training data
        self.X_train = X.copy()
        self.y_train = y.copy() if hasattr(y, "copy") else np.array(y)
        self.feature_names = (
            list(X.columns)
            if hasattr(X, "columns")
            else [f"feature_{i}" for i in range(X.shape[1])]
        )

        # Build and fit preprocessing pipeline
        X_processed = self._fit_preprocessing_pipeline(X, y)

        # Apply sampling if needed
        if self.sampling_strategy != "none":
            X_processed, y_processed = self._apply_sampling(X_processed, y)
        else:
            y_processed = y

        # Use native algorithm selector for hyperparameter optimization
        if self.verbosity > 0:
            print("Performing algorithm selection and hyperparameter optimization...")

        algo_selector = NativeAlgorithmSelector(
            task_type=self.mode, random_state=self.random_seed
        )

        # Find best algorithm and parameters
        y_series = (
            pd.Series(y_processed)
            if not isinstance(y_processed, pd.Series)
            else y_processed
        )
        result = algo_selector.find_best_algorithm(
            X_processed,
            y_series,
            algorithms=["random_forest", "gradient_boosting", "linear_model"],
            cv_folds=self.cv_folds,
            max_evals_per_algo=max(
                5, self.max_evals // 3
            ),  # Distribute evaluations across algorithms
        )

        # Store results
        self.best_params = result["best_params"]
        self.best_score = result["best_score"]
        self.model = result["best_model"]

        if self.verbosity > 0:
            print(f"Best algorithm: {result['best_algorithm']}")
            print(f"Best CV score: {result['best_score']:.4f}")
            print("Model training completed!")

        return self

    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        X_processed = self._preprocess_features(X)
        return self.model.predict(X_processed)

    def predict_proba(self, X):
        """Get prediction probabilities for classification tasks."""
        if self.mode == "regression":
            raise ValueError("predict_proba is not supported for regression tasks")

        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        X_processed = self._preprocess_features(X)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_processed)
        else:
            raise ValueError(
                "The trained model does not support probability predictions"
            )

    def _fit_preprocessing_pipeline(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Build and fit the preprocessing pipeline."""
        if self.verbosity > 0:
            print("Building preprocessing pipeline...")

        X_processed = X.copy()

        # 1. Missing value imputation
        if self.verbosity > 0:
            print("  - Applying missing value imputation...")
        self.imputer = missimp.DataFrameImputer(strategy=self.imputation_strategy)
        X_processed = self.imputer.fit_transform(X_processed)

        # 2. Categorical encoding
        if self.verbosity > 0:
            print("  - Encoding categorical features...")
        self.encoder = encoders.NativeEncoder(encoding_method=self.encoding_method)
        X_processed = self.encoder.fit_transform(X_processed)

        # 3. Feature engineering
        if self.feature_engineering:
            if self.verbosity > 0:
                print("  - Creating engineered features...")
            self.feature_engineer = feateng.FeatureEngineering(
                decomposition_methods=["pca"],
                n_components=0.95,
                clustering_features=True,
                polynomial_features=False,  # Disable for performance
            )
            X_processed = self.feature_engineer.fit_transform(X_processed, y)

        # 4. Feature selection
        if self.feature_selection:
            if self.verbosity > 0:
                print("  - Selecting best features...")
            self.feature_selector = featsel.FeatureSelector(
                methods=["variance", "univariate"],
                variance_threshold=0.01,
                k_best=min(50, X_processed.shape[1]),  # Limit to reasonable number
                task_type=self.mode,
            )
            X_processed = self.feature_selector.fit_transform(X_processed, y)

        # Update feature names
        self.feature_names = list(X_processed.columns)
        self.preprocessing_fitted = True

        if self.verbosity > 0:
            print(f"  - Preprocessing completed. Final shape: {X_processed.shape}")

        return X_processed

    def _apply_sampling(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Apply sampling strategy for imbalanced datasets."""
        if self.mode == "regression":
            return X, y  # No sampling for regression

        if self.verbosity > 0:
            print("  - Analyzing class distribution and applying sampling...")

        # Analyze class distribution using walrus operator
        if self.verbosity > 0:
            class_dist = sampler.analyze_class_distribution(y)
            print(f"    Original distribution: {class_dist['class_counts']}")
            print(f"    Imbalance ratio: {class_dist['imbalance_ratio']:.2f}")
        else:
            class_dist = sampler.analyze_class_distribution(y)

        # Apply sampling if needed
        if class_dist["imbalance_ratio"] > 2.0:
            self.sampler = sampler.NativeSampler(
                strategy=self.sampling_strategy, random_state=self.random_seed
            )
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)

            if self.verbosity > 0 and (
                new_dist := sampler.analyze_class_distribution(y_resampled)
            ):
                print(f"    Resampled distribution: {new_dist['class_counts']}")

            return X_resampled, y_resampled
        else:
            if self.verbosity > 0:
                print("    Dataset is balanced, no sampling applied.")
            return X, y

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing to new data."""
        if not self.preprocessing_fitted:
            raise ValueError(
                "Preprocessing pipeline has not been fitted. Call fit() first."
            )

        X_processed = X.copy()

        # Apply transformations in the same order as training
        if self.imputer is not None:
            X_processed = self.imputer.transform(X_processed)

        if self.encoder is not None:
            X_processed = self.encoder.transform(X_processed)

        if self.feature_engineer is not None:
            X_processed = self.feature_engineer.transform(X_processed)

        if self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)

        return X_processed

    def score(self, X, y):
        """Get model score on given data"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        predictions = self.predict(X)

        if self.mode == "classification":
            from sklearn.metrics import accuracy_score

            return accuracy_score(y, predictions)
        else:
            from sklearn.metrics import r2_score

            return r2_score(y, predictions)

    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        if hasattr(self.model, "feature_importances_"):
            importance_df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            return importance_df
        else:
            print("Model does not support feature importance")
            return None

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """
        Comprehensive evaluation of the trained model

        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test targets

        Returns:
        --------
        dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Preprocess test data using the fitted pipeline
        X_test_processed = self._preprocess_features(X_test)

        # For test-only evaluation, we'll compute direct metrics without CV
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
        )

        # Make predictions
        y_pred = self.model.predict(X_test_processed)

        results = {}

        if self.mode == "classification":
            results.update(
                {
                    "test_accuracy": accuracy_score(y_test, y_pred),
                    "test_precision": precision_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "test_recall": recall_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "test_f1": f1_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                }
            )

            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2 and hasattr(self.model, "predict_proba"):
                try:
                    from sklearn.metrics import roc_auc_score

                    y_pred_proba = self.model.predict_proba(X_test_processed)[:, 1]
                    results["test_roc_auc"] = roc_auc_score(y_test, y_pred_proba)
                except Exception:
                    pass

        else:  # regression
            results.update(
                {
                    "test_r2": r2_score(y_test, y_pred),
                    "test_mse": mean_squared_error(y_test, y_pred),
                    "test_mae": mean_absolute_error(y_test, y_pred),
                    "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                }
            )

        return results

    def get_preprocessing_info(self) -> dict[str, Any]:
        """Get information about the preprocessing pipeline"""
        info = {
            "preprocessing_fitted": self.preprocessing_fitted,
            "imputation_strategy": self.imputation_strategy,
            "encoding_method": self.encoding_method,
            "feature_engineering_enabled": self.feature_engineering,
            "feature_selection_enabled": self.feature_selection,
            "sampling_strategy": self.sampling_strategy,
            "original_features": (
                len(self.X_train.columns) if self.X_train is not None else 0
            ),
            "final_features": len(self.feature_names) if self.feature_names else 0,
        }

        # Add sampling info if available
        if hasattr(self, "sampler") and self.sampler is not None:
            info["sampling_info"] = self.sampler.get_sampling_info()

        return info


if __name__ == "__main__":
    # Example usage
    print("YAAML AutoML - Native Implementation")

    # Create sample dataset
    from sklearn.datasets import make_classification

    print("\nCreating sample dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    print(f"Dataset shape: {X_df.shape}")
    print(f"Class distribution: {y_series.value_counts().to_dict()}")

    # Initialize AutoML
    automl = YAAMLAutoML(
        max_evals=10,
        cv_folds=3,
        verbosity=1,
        feature_engineering=True,
        feature_selection=True,
    )

    # Split data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Train AutoML pipeline
    print("\n" + "=" * 50)
    print("TRAINING AUTOML PIPELINE")
    print("=" * 50)

    automl.fit(X_train, y_train)

    # Make predictions
    print("\n" + "=" * 50)
    print("MAKING PREDICTIONS")
    print("=" * 50)

    y_pred = automl.predict(X_test)
    y_pred_proba = automl.predict_proba(X_test)

    # Evaluate
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    score = automl.score(X_test, y_test)
    print(f"Test Accuracy: {score:.4f}")

    # Get preprocessing info
    preprocessing_info = automl.get_preprocessing_info()
    print("\nPreprocessing Summary:")
    print(f"  Original features: {preprocessing_info['original_features']}")
    print(f"  Final features: {preprocessing_info['final_features']}")
    print(f"  Feature engineering: {preprocessing_info['feature_engineering_enabled']}")
    print(f"  Feature selection: {preprocessing_info['feature_selection_enabled']}")

    # Get feature importance
    importance = automl.get_feature_importance()
    if importance is not None:
        print("\nTop 5 Important Features:")
        print(importance.head())

    print("\n✅ YAAML AutoML example completed successfully!")
