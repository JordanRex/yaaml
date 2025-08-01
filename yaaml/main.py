from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import encoding as encoders
from . import feature_engineering as feateng
from . import feature_selection as featsel
from . import miss_imputation as missimp
from . import sampling as sampler
from .native_algorithms import NativeAlgorithmSelector

# ########################################################################
# NATIVE AUTOML PIPELINE - Pure Python/sklearn implementation
# ########################################################################


class YAAMLAutoML:
    """
    YAAML AutoML Pipeline - Automated Machine Learning

    Parameters
    ----------
    random_seed : int, default=42
        Random seed for reproducibility.
    max_evals : int, default=10
        Maximum hyperparameter evaluations per algorithm.
    cv_folds : int, default=3
        Cross-validation folds for model evaluation.
    mode : str, default="classification"
        Task type: "classification" or "regression".
    verbosity : int, default=1
        Logging level: 0=silent, 1=standard, 2=detailed.
    imputation_strategy : str, default="mean"
        Missing value strategy: "mean", "median", "most_frequent", "knn", "iterative".
    encoding_method : str, default="ordinal"
        Categorical encoding: "ordinal", "onehot", "target".
    feature_engineering : bool, default=True
        Whether to apply feature engineering.
    feature_selection : bool, default=True
        Whether to apply feature selection.
    sampling_strategy : str, default="auto"
        Imbalanced data handling: "auto", "smote", "random_oversample",
        "random_undersample", "none".

    Attributes
    ----------
    model : sklearn estimator
        Best trained model after fitting.
    best_params : dict
        Best hyperparameters found.
    best_score : float
        Best cross-validation score.
    feature_names : list[str]
        Final feature names after preprocessing.
    preprocessing_fitted : bool
        Whether preprocessing pipeline is fitted.
    """

    def __init__(
        self,
        random_seed: int = 42,
        max_evals: int = 10,
        cv_folds: int = 3,
        mode: str = "classification",
        verbosity: int = 1,
        # Preprocessing options
        imputation_strategy: str = "mean",
        encoding_method: str = "ordinal",
        feature_engineering: bool = True,
        feature_selection: bool = True,
        sampling_strategy: str = "auto",
    ) -> None:
        """
        Initialize the YAAML AutoML pipeline.

        Parameters
        ----------
        random_seed : int, default=42
            Random seed for reproducibility.
        max_evals : int, default=10
            Maximum hyperparameter evaluations per algorithm.
        cv_folds : int, default=3
            Cross-validation folds for model evaluation.
        mode : str, default="classification"

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        # Validate input parameters
        if mode not in ["classification", "regression"]:
            raise ValueError(
                f"mode must be 'classification' or 'regression', got '{mode}'"
            )
        if cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")
        if max_evals < 1:
            raise ValueError(f"max_evals must be >= 1, got {max_evals}")
        if verbosity not in [0, 1, 2]:
            raise ValueError(f"verbosity must be 0, 1, or 2, got {verbosity}")

        # Core configuration
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

        # Pipeline components - initialized as None, fitted during fit()
        self.imputer: missimp.DataFrameImputer | None = None
        self.encoder: encoders.NativeEncoder | encoders.TargetEncoder | None = None
        self.feature_engineer: feateng.FeatureEngineering | None = None
        self.feature_selector: featsel.FeatureSelector | None = None
        self.sampler: sampler.NativeSampler | None = None

        # Model and optimization results
        self.model: Any | None = None
        self.best_params: dict[str, Any] | None = None
        self.best_score: float | None = None

        # Training data and pipeline state
        self.X_train: pd.DataFrame | None = None
        self.y_train: pd.Series | np.ndarray | None = None
        self.feature_names: list[str] | None = None
        self.preprocessing_fitted = False

        # Set random seeds for reproducibility across all components
        np.random.seed(self.random_seed)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> "YAAMLAutoML":
        """
        Fit the complete AutoML pipeline to training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : pd.Series or np.ndarray
            Training target vector.
        X_valid : pd.DataFrame, optional
            Validation features (reserved for future use).
        y_valid : pd.Series or np.ndarray, optional
            Validation targets (reserved for future use).

        Returns
        -------
        self : YAAMLAutoML
            Returns fitted estimator for method chaining.

        Raises
        ------
        ValueError
            If X is not a pandas DataFrame.
            If X and y have different number of samples.
            If y contains only one unique value.
        """
        # ====================================================================
        # Input Validation and Logging
        # ====================================================================

        # Validate input data types and shapes
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame, got {type(X)}")

        if len(X) != len(y):
            raise ValueError(
                "X and y must have same number of samples. "
                f"Got X: {len(X)}, y: {len(y)}"
            )

        if len(np.unique(y)) <= 1:
            raise ValueError("y must contain more than one unique value")

        # Validate classification targets
        if self.mode == "classification":
            if not np.isfinite(np.array(y, dtype=float)).all():
                raise ValueError("Classification targets must be finite values")

        if self.verbosity > 0:
            print("Starting YAAML AutoML pipeline...")
            print(f"Training data shape: {X.shape}")
            print(f"Mode: {self.mode}")
            print(
                f"Target distribution: {pd.Series(y).value_counts().head().to_dict()}"
            )

        # ====================================================================
        # Data Storage and Preprocessing Setup
        # ====================================================================

        # Store deep copies of training data for pipeline fitting
        self.X_train = X.copy()
        self.y_train = y.copy() if hasattr(y, "copy") else np.array(y)

        # Generate or extract feature names for consistent column tracking
        self.feature_names = (
            list(X.columns)
            if hasattr(X, "columns")
            else [f"feature_{i}" for i in range(X.shape[1])]
        )

        # Ensure target is a pandas Series for consistent processing throughout pipeline
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()

        # ====================================================================
        # Preprocessing Pipeline Execution
        # ====================================================================

        # Build and fit the complete preprocessing pipeline:
        # 1. Missing value imputation 2. Categorical encoding
        # 3. Feature engineering 4. Feature selection
        X_processed = self._fit_preprocessing_pipeline(X, y_series)

        # ====================================================================
        # Data Sampling for Imbalanced Datasets
        # ====================================================================

        # Apply sampling strategies if configured (classification only)
        if self.sampling_strategy != "none":
            X_processed, y_processed = self._apply_sampling(X_processed, y_series)
        else:
            y_processed = y_series

        # ====================================================================
        # Algorithm Selection and Hyperparameter Optimization
        # ====================================================================

        # Initialize native algorithm selector with task-specific configuration
        if self.verbosity > 0:
            print("Performing algorithm selection and hyperparameter optimization...")

        algo_selector = NativeAlgorithmSelector(
            task_type=self.mode, random_state=self.random_seed
        )

        # Test multiple algorithms with distributed hyperparameter evaluations
        # Algorithms tested: RandomForest, GradientBoosting, Linear Models
        result = algo_selector.find_best_algorithm(
            X_processed,
            y_processed,
            algorithms=["random_forest", "gradient_boosting", "linear_model"],
            cv_folds=self.cv_folds,
            max_evals_per_algo=max(
                5, self.max_evals // 3
            ),  # Distribute total evaluations across algorithms
        )

        # ====================================================================
        # Store Optimization Results and Complete Training
        # ====================================================================

        # Store the best model and optimization results
        self.best_params = result["best_params"]
        self.best_score = result["best_score"]
        self.model = result["best_model"]

        # Log training completion and results
        if self.verbosity > 0:
            print(f"Best algorithm: {result['best_algorithm']}")
            print(f"Best CV score: {result['best_score']:.4f}")
            print("Model training completed!")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data using the trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix for prediction.

        Returns
        -------
        predictions : np.ndarray
            Model predictions.

        Raises
        ------
        ValueError
            If the model has not been trained.
        """
        # Validate that model has been trained
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Validate input format
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame, got {type(X)}")

        # Apply the same preprocessing pipeline as used during training
        X_processed = self._preprocess_features(X)

        # Generate predictions using the best trained model
        return self.model.predict(X_processed)  # type: ignore[no-any-return]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities for classification tasks.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix for probability prediction.

        Returns
        -------
        probabilities : np.ndarray
            Class membership probabilities.

        Raises
        ------
        ValueError
            If mode is "regression" or model not trained.
        """
        # Validate task type - probabilities only for classification
        if self.mode == "regression":
            raise ValueError("predict_proba is not supported for regression tasks")

        # Validate that model has been trained
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Validate input format
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame, got {type(X)}")

        # Apply preprocessing pipeline
        X_processed = self._preprocess_features(X)

        # Check if model supports probability prediction
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_processed)  # type: ignore[no-any-return]
        else:
            raise ValueError(
                f"The trained model ({type(self.model).__name__}) does not "
                "support probability predictions"
            )

    def _fit_preprocessing_pipeline(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """
        Build and fit the preprocessing pipeline.

        Applies missing value imputation, categorical encoding, feature engineering,
        and feature selection in the correct order.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : pd.Series
            Training target vector.

        Returns
        -------
        X_processed : pd.DataFrame
            Preprocessed feature matrix.
        """
        if self.verbosity > 0:
            print("Building preprocessing pipeline...")

        # Start with a copy to avoid modifying original data
        X_processed = X.copy()

        # ================================================================
        # Step 1: Missing Value Imputation
        # ================================================================
        if self.verbosity > 0:
            print("  - Applying missing value imputation...")
        self.imputer = missimp.DataFrameImputer(strategy=self.imputation_strategy)
        X_processed = self.imputer.fit_transform(X_processed)

        # ================================================================
        # Step 2: Categorical Variable Encoding
        # ================================================================
        if self.verbosity > 0:
            print("  - Encoding categorical features...")

        if self.encoding_method == "target":
            # Use target encoding which requires target variable
            self.encoder = encoders.TargetEncoder()
            X_processed = self.encoder.fit_transform(X_processed, y)
        else:
            # Use native encoder for other methods
            self.encoder = encoders.NativeEncoder(encoding_method=self.encoding_method)
            X_processed = self.encoder.fit_transform(X_processed)

        # ================================================================
        # Step 3: Feature Engineering (Optional)
        # ================================================================
        if self.feature_engineering:
            if self.verbosity > 0:
                print("  - Creating engineered features...")
            self.feature_engineer = feateng.FeatureEngineering(
                decomposition_methods=["pca"],  # Principal Component Analysis
                n_components=0.95,  # Retain 95% variance
                clustering_features=True,  # Add K-means distance features
                polynomial_features=False,  # Disabled for performance
            )
            X_processed = self.feature_engineer.fit_transform(X_processed, y)

        # ================================================================
        # Step 4: Feature Selection (Optional)
        # ================================================================
        # Remove irrelevant features and select the most predictive ones
        if self.feature_selection:
            if self.verbosity > 0:
                print("  - Selecting best features...")
            self.feature_selector = featsel.FeatureSelector(
                methods=["variance", "univariate"],  # Filter methods
                variance_threshold=0.01,  # Remove near-constant features
                k_best=min(50, X_processed.shape[1]),  # Limit to manageable number
                task_type=self.mode,  # Classification vs regression
            )
            X_processed = self.feature_selector.fit_transform(X_processed, y)

        # ================================================================
        # Pipeline Completion and State Updates
        # ================================================================
        # Update feature names to reflect all transformations applied
        self.feature_names = list(X_processed.columns)
        self.preprocessing_fitted = True

        if self.verbosity > 0:
            print(f"  - Preprocessing completed. Final shape: {X_processed.shape}")
            print(f"  - Feature count: {X.shape[1]} → {X_processed.shape[1]}")

        return X_processed

    def _apply_sampling(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply sampling strategies for imbalanced datasets.

        Parameters
        ----------
        X : pd.DataFrame
            Preprocessed feature matrix.
        y : pd.Series
            Target vector.

        Returns
        -------
        X_resampled : pd.DataFrame
            Feature matrix after sampling.
        y_resampled : pd.Series
            Target vector after sampling.
        """
        # Skip sampling for regression tasks
        if self.mode == "regression":
            return X, y

        if self.verbosity > 0:
            print("  - Analyzing class distribution and applying sampling...")

        # Analyze class distribution to determine if sampling is needed
        class_dist = sampler.analyze_class_distribution(y)

        if self.verbosity > 0:
            print(f"    Original distribution: {class_dist['class_counts']}")
            print(f"    Imbalance ratio: {class_dist['imbalance_ratio']:.2f}")

        # Apply sampling only if dataset is significantly imbalanced
        if class_dist["imbalance_ratio"] > 2.0:
            # Initialize and apply the configured sampling strategy
            self.sampler = sampler.NativeSampler(
                strategy=self.sampling_strategy, random_state=self.random_seed
            )
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)

            # Log the new distribution if verbose
            if self.verbosity > 0:
                new_dist = sampler.analyze_class_distribution(y_resampled)
                print(f"    Resampled distribution: {new_dist['class_counts']}")

            return X_resampled, y_resampled
        else:
            if self.verbosity > 0:
                print("    Dataset is balanced, no sampling applied.")
            return X, y

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted preprocessing pipeline to new data.

        Transforms new data using the same preprocessing steps that were
        fitted during training. All transformations are applied in the
        exact same order to ensure consistency.

        Parameters
        ----------
        X : pd.DataFrame
            New feature data to preprocess. Must have same column structure
            as training data.

        Returns
        -------
        X_processed : pd.DataFrame
            Preprocessed feature matrix ready for model prediction.

        Raises
        ------
        ValueError
            If preprocessing pipeline has not been fitted (fit() not called).

        """
        if not self.preprocessing_fitted:
            raise ValueError(
                "Preprocessing pipeline has not been fitted. Call fit() first."
            )

        # Apply all fitted transformations in training order
        X_processed = X.copy()

        # Step 1: Apply fitted missing value imputation
        if self.imputer is not None:
            X_processed = self.imputer.transform(X_processed)

        # Step 2: Apply fitted categorical encoding
        if self.encoder is not None:
            X_processed = self.encoder.transform(X_processed)

        # Step 3: Apply fitted feature engineering
        if self.feature_engineer is not None:
            X_processed = self.feature_engineer.transform(X_processed)

        # Step 4: Apply fitted feature selection
        if self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)

        return X_processed

    def score(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> float:
        """
        Evaluate model performance on given data.

        Computes appropriate evaluation metric based on the task type:
        accuracy for classification, R² score for regression.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for evaluation.
        y : pd.Series or np.ndarray
            True target values.

        Returns
        -------
        score : float
            Model performance score (accuracy for classification, R² for regression).

        Raises
        ------
        ValueError
            If model has not been trained.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Generate predictions using the trained model
        predictions = self.predict(X)

        # Return appropriate metric based on task type
        if self.mode == "classification":
            from sklearn.metrics import accuracy_score

            return accuracy_score(y, predictions)  # type: ignore[no-any-return]
        else:
            from sklearn.metrics import r2_score

            return r2_score(y, predictions)  # type: ignore[no-any-return]

    def get_feature_importance(self) -> pd.DataFrame | None:
        """
        Get feature importance rankings from the trained model.

        Returns
        -------
        importance_df : pd.DataFrame or None
            DataFrame with feature importance scores, or None if not supported.

        Raises
        ------
        ValueError
            If model has not been trained.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Check if model supports feature importance
        if hasattr(self.model, "feature_importances_"):
            importance_df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            return importance_df
        else:
            if self.verbosity > 0:
                print(
                    f"Model {type(self.model).__name__} does not support "
                    "feature importance"
                )
            return None

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """

        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature matrix for evaluation.
        y_test : pd.Series
            True target values for test set.

        Returns
        -------
        results : dict[str, float]
            Dictionary containing evaluation metrics:

            **Classification metrics:**
            - test_accuracy: Accuracy score (0.0 to 1.0)
            - test_precision: Macro-averaged precision (0.0 to 1.0)
            - test_recall: Macro-averaged recall (0.0 to 1.0)
            - test_f1: Macro-averaged F1 score (0.0 to 1.0)
            - test_roc_auc: ROC AUC (binary classification only)

            **Regression metrics:**
            - test_r2: R² coefficient of determination (-∞ to 1.0)
            - test_mse: Mean squared error (0.0 to +∞)
            - test_mae: Mean absolute error (0.0 to +∞)
            - test_rmse: Root mean squared error (0.0 to +∞)

        Raises
        ------
        ValueError
            If model has not been trained (fit() not called).

        --------

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
        """
        Get information about the preprocessing pipeline.

        Returns
        -------
        info : dict[str, Any]
            Dictionary containing preprocessing pipeline information.
        """
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
