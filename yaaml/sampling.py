"""
YAAML Sampling Module
Native implementation of sampling techniques for imbalanced datasets
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils import resample


class NativeSampler:
    """Native implementation of sampling techniques without external dependencies"""

    def __init__(self, strategy: str = "auto", random_state: int = 42):
        """
        Initialize sampler

        Parameters:
        -----------
        strategy : str
            Sampling strategy: 'auto', 'undersample', 'oversample', 'balanced'
        random_state : int
            Random state for reproducibility
        """
        self.strategy = strategy
        self.random_state = random_state
        self.class_counts: pd.Series | None = None
        self.sampling_info: dict[str, float | bool | dict[str, Any]] = {}

    def fit_resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fit and resample the dataset

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable

        Returns:
        --------
        tuple
            Resampled (X, y)
        """
        # Analyze class distribution
        self.class_counts = y.value_counts().sort_index()

        if len(self.class_counts) < 2:
            warnings.warn("Only one class found. No sampling applied.")
            return X.copy(), y.copy()

        # Determine if dataset is imbalanced
        min_class_count = int(self.class_counts.min())
        max_class_count = int(self.class_counts.max())
        imbalance_ratio = max_class_count / min_class_count

        self.sampling_info = {
            "original_distribution": self.class_counts.to_dict(),
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": imbalance_ratio > 2.0,
        }

        # Apply sampling strategy
        if self.strategy == "auto":
            if imbalance_ratio > 5:
                return self._balanced_sampling(X, y)
            elif imbalance_ratio > 2:
                return self._moderate_oversample(X, y)
            else:
                return X.copy(), y.copy()

        elif self.strategy == "oversample":
            return self._oversample_minority(X, y)

        elif self.strategy == "undersample":
            return self._undersample_majority(X, y)

        elif self.strategy == "balanced":
            return self._balanced_sampling(X, y)

        else:
            warnings.warn(f"Unknown strategy '{self.strategy}'. No sampling applied.")
            return X.copy(), y.copy()

    def _oversample_minority(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Oversample minority classes to match majority class"""
        if self.class_counts is None:
            raise ValueError("fit_resample must be called first")

        class_counts = self.class_counts  # Type narrowing for mypy
        max_count = int(class_counts.max())

        X_resampled: list[pd.DataFrame] = []
        y_resampled: list[pd.Series] = []

        for class_label in class_counts.index:
            # Get samples for this class
            class_mask = y == class_label
            X_class = X[class_mask]
            y_class = y[class_mask]

            current_count = len(X_class)

            if current_count < max_count:
                # Oversample this class
                X_upsampled, y_upsampled = resample(
                    X_class,
                    y_class,
                    replace=True,
                    n_samples=max_count,
                    random_state=self.random_state,
                )
                # Convert back to pandas if needed
                if isinstance(X_upsampled, np.ndarray):
                    X_upsampled = pd.DataFrame(X_upsampled, columns=X_class.columns)
                if isinstance(y_upsampled, np.ndarray):
                    y_upsampled = pd.Series(y_upsampled)

                X_resampled.append(X_upsampled)
                y_resampled.append(y_upsampled)
            else:
                X_resampled.append(X_class)
                y_resampled.append(y_class)

        X_final = pd.concat(X_resampled, ignore_index=True)
        y_final = pd.concat(y_resampled, ignore_index=True)

        return X_final, y_final

    def _undersample_majority(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Undersample majority classes to match minority class"""
        if self.class_counts is None:
            raise ValueError("fit_resample must be called first")

        class_counts = self.class_counts  # Type narrowing for mypy
        min_count = int(class_counts.min())

        X_resampled: list[pd.DataFrame] = []
        y_resampled: list[pd.Series] = []

        for class_label in class_counts.index:
            # Get samples for this class
            class_mask = y == class_label
            X_class = X[class_mask]
            y_class = y[class_mask]

            current_count = len(X_class)

            if current_count > min_count:
                # Undersample this class
                X_downsampled, y_downsampled = resample(
                    X_class,
                    y_class,
                    replace=False,
                    n_samples=min_count,
                    random_state=self.random_state,
                )
                # Convert back to pandas if needed
                if isinstance(X_downsampled, np.ndarray):
                    X_downsampled = pd.DataFrame(X_downsampled, columns=X_class.columns)
                if isinstance(y_downsampled, np.ndarray):
                    y_downsampled = pd.Series(y_downsampled)

                X_resampled.append(X_downsampled)
                y_resampled.append(y_downsampled)
            else:
                X_resampled.append(X_class)
                y_resampled.append(y_class)

        X_final = pd.concat(X_resampled, ignore_index=True)
        y_final = pd.concat(y_resampled, ignore_index=True)

        return X_final, y_final

    def _balanced_sampling(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Balanced sampling using median class size as target"""
        if self.class_counts is None:
            raise ValueError("fit_resample must be called first")

        class_counts = self.class_counts  # Type narrowing for mypy
        target_count = int(class_counts.median())

        X_resampled: list[pd.DataFrame] = []
        y_resampled: list[pd.Series] = []

        for class_label in class_counts.index:
            # Get samples for this class
            class_mask = y == class_label
            X_class = X[class_mask]
            y_class = y[class_mask]

            current_count = len(X_class)

            if current_count != target_count:
                # Resample to target count
                replace_flag = current_count < target_count
                X_resampled_class, y_resampled_class = resample(
                    X_class,
                    y_class,
                    replace=replace_flag,
                    n_samples=target_count,
                    random_state=self.random_state,
                )
                # Convert back to pandas if needed
                if isinstance(X_resampled_class, np.ndarray):
                    X_resampled_class = pd.DataFrame(
                        X_resampled_class, columns=X_class.columns
                    )
                if isinstance(y_resampled_class, np.ndarray):
                    y_resampled_class = pd.Series(y_resampled_class)

                X_resampled.append(X_resampled_class)
                y_resampled.append(y_resampled_class)
            else:
                X_resampled.append(X_class)
                y_resampled.append(y_class)

        X_final = pd.concat(X_resampled, ignore_index=True)
        y_final = pd.concat(y_resampled, ignore_index=True)

        return X_final, y_final

    def _moderate_oversample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Moderate oversampling - increase minority classes but not to full
        majority size"""
        if self.class_counts is None:
            raise ValueError("fit_resample must be called first")

        class_counts = self.class_counts  # Type narrowing for mypy
        max_count = int(class_counts.max())
        min_count = int(class_counts.min())

        # Target: reduce imbalance ratio to 2:1
        target_min_count = max(min_count, max_count // 2)

        X_resampled: list[pd.DataFrame] = []
        y_resampled: list[pd.Series] = []

        for class_label in class_counts.index:
            # Get samples for this class
            class_mask = y == class_label
            X_class = X[class_mask]
            y_class = y[class_mask]

            current_count = len(X_class)

            if current_count < target_min_count:
                # Oversample this class
                X_upsampled, y_upsampled = resample(
                    X_class,
                    y_class,
                    replace=True,
                    n_samples=target_min_count,
                    random_state=self.random_state,
                )
                # Convert back to pandas if needed
                if isinstance(X_upsampled, np.ndarray):
                    X_upsampled = pd.DataFrame(X_upsampled, columns=X_class.columns)
                if isinstance(y_upsampled, np.ndarray):
                    y_upsampled = pd.Series(y_upsampled)

                X_resampled.append(X_upsampled)
                y_resampled.append(y_upsampled)
            else:
                X_resampled.append(X_class)
                y_resampled.append(y_class)

        X_final = pd.concat(X_resampled, ignore_index=True)
        y_final = pd.concat(y_resampled, ignore_index=True)

        return X_final, y_final

    def get_sampling_info(self) -> dict[str, Any]:
        """Return information about the sampling performed"""
        return self.sampling_info.copy()


class StratifiedSampler:
    """Stratified sampling for maintaining class proportions"""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize stratified sampler

        Parameters:
        -----------
        test_size : float
            Proportion of dataset to include in the test split
        random_state : int
            Random state for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Perform stratified split

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable

        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        np.random.seed(self.random_state)

        # Get unique classes and their counts
        classes = y.unique()

        train_indices = []
        test_indices = []

        for class_label in classes:
            # Get indices for this class
            class_indices = y[y == class_label].index.tolist()

            # Calculate split sizes
            n_class = len(class_indices)
            n_test = max(1, int(n_class * self.test_size))

            # Randomly sample test indices
            test_class_indices = np.random.choice(
                class_indices, size=n_test, replace=False
            ).tolist()

            train_class_indices = [
                idx for idx in class_indices if idx not in test_class_indices
            ]

            train_indices.extend(train_class_indices)
            test_indices.extend(test_class_indices)

        # Create splits
        X_train = X.loc[train_indices].reset_index(drop=True)
        X_test = X.loc[test_indices].reset_index(drop=True)
        y_train = y.loc[train_indices].reset_index(drop=True)
        y_test = y.loc[test_indices].reset_index(drop=True)

        return X_train, X_test, y_train, y_test


def apply_sampling(
    X: pd.DataFrame, y: pd.Series, strategy: str = "auto", random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function for applying sampling

    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    strategy : str
        Sampling strategy
    random_state : int
        Random state

    Returns:
    --------
    tuple
        Resampled (X, y)
    """
    sampler = NativeSampler(strategy=strategy, random_state=random_state)
    return sampler.fit_resample(X, y)


def analyze_class_distribution(y: pd.Series | np.ndarray) -> dict[str, Any]:
    """
    Analyze class distribution in target variable

    Parameters:
    -----------
    y : pd.Series or np.ndarray
        Target variable

    Returns:
    --------
    dict
        Class distribution analysis
    """
    # Convert to pandas Series if numpy array
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    counts = y.value_counts().sort_index()
    proportions = y.value_counts(normalize=True).sort_index()

    min_count = int(counts.min())
    max_count = int(counts.max())
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    return {
        "class_counts": counts.to_dict(),
        "class_proportions": proportions.to_dict(),
        "total_samples": len(y),
        "n_classes": len(counts),
        "imbalance_ratio": imbalance_ratio,
        "is_balanced": imbalance_ratio <= 2.0,
        "minority_class": counts.idxmin(),
        "majority_class": counts.idxmax(),
    }


# For backward compatibility
class sampler(NativeSampler):
    """Legacy class name for backward compatibility"""

    pass
