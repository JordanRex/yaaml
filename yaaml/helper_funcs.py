"""
YAAML Helper Functions
Utility functions for data processing and model evaluation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    task_type: str = "classification",
    cv_folds: int = 5,
) -> dict[str, float]:
    """
    Comprehensive model evaluation with cross-validation and test metrics

    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training targets
    X_test : pd.DataFrame, optional
        Test features
    y_test : pd.Series, optional
        Test targets
    task_type : str
        'classification' or 'regression'
    cv_folds : int
        Number of cross-validation folds

    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Validate input data
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data cannot be empty")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length")

    # Adjust cv_folds if necessary
    n_samples = len(X_train)
    cv_folds = min(cv_folds, n_samples // 2) if n_samples < 10 else cv_folds

    if cv_folds < 2:
        cv_folds = 2

    results = {}

    # Cross-validation scores
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # CV scores
        cv_accuracy = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="accuracy"
        )
        cv_precision = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="precision_macro"
        )
        cv_recall = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="recall_macro"
        )
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")

        results.update(
            {
                "cv_accuracy_mean": cv_accuracy.mean(),
                "cv_accuracy_std": cv_accuracy.std(),
                "cv_precision_mean": cv_precision.mean(),
                "cv_precision_std": cv_precision.std(),
                "cv_recall_mean": cv_recall.mean(),
                "cv_recall_std": cv_recall.std(),
                "cv_f1_mean": cv_f1.mean(),
                "cv_f1_std": cv_f1.std(),
            }
        )

        # Try ROC AUC for binary classification
        if len(np.unique(y_train)) == 2:
            try:
                cv_roc_auc = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring="roc_auc"
                )
                results.update(
                    {
                        "cv_roc_auc_mean": cv_roc_auc.mean(),
                        "cv_roc_auc_std": cv_roc_auc.std(),
                    }
                )
            except Exception:
                pass

    else:  # regression
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        cv_neg_mse = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
        )
        cv_neg_mae = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
        )

        results.update(
            {
                "cv_r2_mean": cv_r2.mean(),
                "cv_r2_std": cv_r2.std(),
                "cv_mse_mean": -cv_neg_mse.mean(),
                "cv_mse_std": cv_neg_mse.std(),
                "cv_mae_mean": -cv_neg_mae.mean(),
                "cv_mae_std": cv_neg_mae.std(),
            }
        )

    # Test set evaluation
    if (
        X_test is not None
        and y_test is not None
        and len(X_test) > 0
        and len(y_test) > 0
    ):
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must have the same length")

        y_pred = model.predict(X_test)

        if task_type == "classification":
            results.update(
                {
                    "test_accuracy": accuracy_score(y_test, y_pred),
                    "test_precision": precision_score(y_test, y_pred, average="macro"),
                    "test_recall": recall_score(y_test, y_pred, average="macro"),
                    "test_f1": f1_score(y_test, y_pred, average="macro"),
                }
            )

            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
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


def detect_data_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Automatically detect column data types

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    dict
        Dictionary with lists of column names by type
    """
    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
    categorical_columns = list(df.select_dtypes(include=["object", "category"]).columns)
    datetime_columns = list(df.select_dtypes(include=["datetime64"]).columns)

    # Try to detect high cardinality categorical columns that might be better as text
    high_cardinality_cats = []
    for col in categorical_columns:
        if df[col].nunique() > 0.8 * len(df):
            high_cardinality_cats.append(col)

    # Remove high cardinality from categorical
    categorical_columns = [
        col for col in categorical_columns if col not in high_cardinality_cats
    ]

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "datetime": datetime_columns,
        "high_cardinality": high_cardinality_cats,
    }


def check_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """
    Check data quality and provide summary statistics

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    dict
        Data quality report
    """
    report = {
        "shape": df.shape,
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "missing_values": {},
        "duplicate_rows": df.duplicated().sum(),
        "column_types": detect_data_types(df),
    }

    # Missing value analysis
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    for col in df.columns:
        if missing_counts[col] > 0:
            report["missing_values"][col] = {
                "count": missing_counts[col],
                "percentage": missing_pct[col],
            }

    # Column-specific statistics
    report["numeric_stats"] = {}
    for col in report["column_types"]["numeric"]:
        stats = df[col].describe()
        report["numeric_stats"][col] = {
            "mean": stats["mean"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
            "zeros": (df[col] == 0).sum(),
            "outliers_iqr": detect_outliers_iqr(df[col]),
        }

    report["categorical_stats"] = {}
    for col in report["column_types"]["categorical"]:
        report["categorical_stats"][col] = {
            "unique_count": df[col].nunique(),
            "top_value": df[col].mode().iloc[0] if not df[col].mode().empty else None,
            "top_value_freq": (
                df[col].value_counts().iloc[0] if not df[col].empty else 0
            ),
        }

    return report


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> int:
    """
    Detect outliers using IQR method

    Parameters:
    -----------
    series : pd.Series
        Numeric series
    factor : float
        IQR multiplier for outlier detection

    Returns:
    --------
    int
        Number of outliers detected
    """
    if series.dtype not in [np.number]:
        return 0

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    return outliers


def memory_optimization(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize dataframe memory usage by downcasting numeric types

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print optimization results

    Returns:
    --------
    pd.DataFrame
        Memory-optimized dataframe
    """
    initial_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)

    df_optimized = df.copy()

    # Optimize numeric columns
    for col in df_optimized.select_dtypes(include=[np.number]).columns:
        col_type = df_optimized[col].dtype

        if str(col_type)[:3] == "int":
            # Integer optimization
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_optimized[col] = df_optimized[col].astype(np.int32)

        elif str(col_type)[:5] == "float":
            # Float optimization
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()

            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df_optimized[col] = df_optimized[col].astype(np.float32)

    # Optimize object columns to category if beneficial
    for col in df_optimized.select_dtypes(include=["object"]).columns:
        if (
            df_optimized[col].nunique() / len(df_optimized) < 0.5
        ):  # Less than 50% unique values
            df_optimized[col] = df_optimized[col].astype("category")

    final_memory = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
    memory_reduction = (initial_memory - final_memory) / initial_memory * 100

    if verbose:
        print("Memory optimization completed:")
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        print(f"Final memory usage: {final_memory:.2f} MB")
        print(f"Memory reduction: {memory_reduction:.2f}%")

    return df_optimized


def split_features_target(
    df: pd.DataFrame, target_column: str, drop_columns: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of target column
    drop_columns : list, optional
        Columns to drop from features

    Returns:
    --------
    tuple
        (features, target)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    # Extract target
    y = df[target_column].copy()

    # Extract features
    feature_columns = [col for col in df.columns if col != target_column]

    if drop_columns:
        feature_columns = [col for col in feature_columns if col not in drop_columns]

    X = df[feature_columns].copy()

    return X, y


def print_model_summary(model: Any, feature_names: list[str] | None = None) -> None:
    """
    Print a summary of the trained model

    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    feature_names : list, optional
        Names of features
    """
    print(f"\n{'='*50}")
    print(f"MODEL SUMMARY: {type(model).__name__}")
    print(f"{'='*50}")

    # Model parameters
    print("\nModel Parameters:")
    for param, value in model.get_params().items():
        print(f"  {param}: {value}")

    # Feature importance (if available)
    if hasattr(model, "feature_importances_") and feature_names:
        print("\nTop 10 Feature Importances:")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        for i in range(min(10, len(indices))):
            idx = indices[i]
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Model-specific information
    if hasattr(model, "n_estimators"):
        print(f"\nNumber of estimators: {model.n_estimators}")

    if hasattr(model, "max_depth"):
        print(f"Max depth: {model.max_depth}")

    if hasattr(model, "n_features_in_"):
        print(f"Number of features used: {model.n_features_in_}")

    print(f"{'='*50}\n")


# Backward compatibility functions
def helpers_function() -> None:
    """Placeholder for backward compatibility"""
    pass
