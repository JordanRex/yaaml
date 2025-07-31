"""
YAAML Encoding Module
Native categorical encoding implementations using sklearn
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


class NativeEncoder:
    """Native categorical encoder with multiple encoding strategies"""

    def __init__(
        self,
        encoding_method: str = "ordinal",
        handle_unknown: str = "use_encoded_value",
    ):
        """
        Initialize the encoder

        Parameters:
        -----------
        encoding_method : str
            Encoding method: 'ordinal', 'onehot', 'label', 'hash'
        handle_unknown : str
            How to handle unknown categories: 'use_encoded_value', 'ignore'
        """
        self.encoding_method = encoding_method
        self.handle_unknown = handle_unknown
        self.encoders = {}
        self.categorical_columns = None
        self.fitted = False

    def fit(self, X: pd.DataFrame) -> "NativeEncoder":
        """
        Fit the encoder on training data

        Parameters:
        -----------
        X : pd.DataFrame
            Training data
        """
        # Identify categorical columns
        self.categorical_columns = list(
            X.select_dtypes(include=["object", "category"]).columns
        )

        if not self.categorical_columns:
            self.fitted = True
            return self

        for col in self.categorical_columns:
            encoder = None  # Initialize encoder

            if self.encoding_method == "ordinal":
                encoder = OrdinalEncoder(
                    handle_unknown=(
                        "use_encoded_value"
                        if self.handle_unknown == "use_encoded_value"
                        else "error"
                    ),
                    unknown_value=(
                        -1 if self.handle_unknown == "use_encoded_value" else None
                    ),
                )
                encoder.fit(X[[col]])

            elif self.encoding_method == "onehot":
                encoder = OneHotEncoder(
                    handle_unknown=(
                        "ignore"
                        if self.handle_unknown == "ignore"
                        else "infrequent_if_exist"
                    ),
                    sparse_output=False,
                )
                encoder.fit(X[[col]])

            elif self.encoding_method == "label":
                encoder = LabelEncoder()
                # Handle unknown values for LabelEncoder manually
                unique_vals = X[col].unique()
                if self.handle_unknown == "use_encoded_value":
                    # Add a placeholder for unknown values
                    extended_vals = list(unique_vals) + ["__UNKNOWN__"]
                    encoder.fit(extended_vals)
                else:
                    encoder.fit(unique_vals)

            elif self.encoding_method == "hash":
                # Feature hashing doesn't need fitting, but we store the column info
                encoder = FeatureHasher(n_features=8, input_type="string")

            else:
                raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

            self.encoders[col] = encoder

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted encoder

        Parameters:
        -----------
        X : pd.DataFrame
            Data to transform

        Returns:
        --------
        pd.DataFrame
            Encoded data
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        if not self.categorical_columns:
            return X.copy()

        X_encoded = X.copy()
        columns_to_drop = []
        new_columns = []

        for col in self.categorical_columns:
            if col not in X.columns:
                continue

            encoder = self.encoders[col]

            if self.encoding_method == "ordinal":
                X_encoded[col] = encoder.transform(X[[col]]).flatten()

            elif self.encoding_method == "onehot":
                # Get feature names for one-hot encoded columns
                try:
                    feature_names = encoder.get_feature_names_out([col])
                except Exception:
                    # Fallback for older sklearn versions
                    n_categories = len(encoder.categories_[0])
                    feature_names = [f"{col}_{i}" for i in range(n_categories)]

                encoded_features = encoder.transform(X[[col]])

                # Add new columns
                for i, feature_name in enumerate(feature_names):
                    X_encoded[feature_name] = encoded_features[:, i]
                    new_columns.append(feature_name)

                columns_to_drop.append(col)

            elif self.encoding_method == "label":
                # Handle unknown values for LabelEncoder
                col_values = X[col].copy()
                if self.handle_unknown == "use_encoded_value":
                    # Replace unknown values with placeholder
                    known_values = set(
                        encoder.classes_[:-1]
                    )  # Exclude the __UNKNOWN__ placeholder
                    col_values = col_values.map(
                        lambda x: x if x in known_values else "__UNKNOWN__"
                    )

                X_encoded[col] = encoder.transform(col_values)

            elif self.encoding_method == "hash":
                # Feature hashing
                hashed_features = encoder.transform(X[col].astype(str)).toarray()

                # Add new columns
                for i in range(hashed_features.shape[1]):
                    feature_name = f"{col}_hash_{i}"
                    X_encoded[feature_name] = hashed_features[:, i]
                    new_columns.append(feature_name)

                columns_to_drop.append(col)

        # Drop original categorical columns if they were replaced
        if columns_to_drop:
            X_encoded = X_encoded.drop(columns=columns_to_drop)

        return X_encoded

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step

        Parameters:
        -----------
        X : pd.DataFrame
            Data to fit and transform

        Returns:
        --------
        pd.DataFrame
            Encoded data
        """
        return self.fit(X).transform(X)


class TargetEncoder:
    """Target-based encoding for categorical variables"""

    def __init__(self, smoothing: float = 1.0, min_samples_leaf: int = 1):
        """
        Initialize target encoder

        Parameters:
        -----------
        smoothing : float
            Smoothing factor for regularization
        min_samples_leaf : int
            Minimum samples required to compute encoding
        """
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encodings = {}
        self.global_mean = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        """
        Fit the target encoder

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        """
        self.global_mean = y.mean()
        categorical_columns = list(
            X.select_dtypes(include=["object", "category"]).columns
        )

        for col in categorical_columns:
            # Calculate target mean for each category
            target_col = y.name if hasattr(y, "name") and y.name else "target"
            grouped_data = pd.concat([X[col], y], axis=1)
            grouped_data.columns = [col, target_col]

            # Group by category and calculate statistics
            grouped = grouped_data.groupby(col)
            category_stats = grouped.agg({target_col: ["mean", "count"]}).reset_index()
            category_stats.columns = [col, "target_mean", "count"]

            # Apply smoothing
            smoothed_means = (
                category_stats["target_mean"] * category_stats["count"]
                + self.global_mean * self.smoothing
            ) / (category_stats["count"] + self.smoothing)

            # Create mapping dictionary
            encoding_map = dict(zip(category_stats[col], smoothed_means))
            self.encodings[col] = encoding_map

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform using fitted target encoder

        Parameters:
        -----------
        X : pd.DataFrame
            Data to transform

        Returns:
        --------
        pd.DataFrame
            Target encoded data
        """
        if not self.fitted:
            raise ValueError("TargetEncoder must be fitted before transform")

        X_encoded = X.copy()

        for col, encoding_map in self.encodings.items():
            if col in X.columns:
                # Map values, use global mean for unknown categories
                X_encoded[col] = X[col].map(encoding_map).fillna(self.global_mean)

        return X_encoded

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform in one step

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable

        Returns:
        --------
        pd.DataFrame
            Target encoded data
        """
        return self.fit(X, y).transform(X)


def encode_categorical_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
    target: pd.Series | None = None,
    method: str = "ordinal",
) -> pd.DataFrame | tuple:
    """
    Convenience function for categorical encoding

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    valid_df : pd.DataFrame, optional
        Validation data
    target : pd.Series, optional
        Target variable (required for target encoding)
    method : str
        Encoding method

    Returns:
    --------
    pd.DataFrame or tuple
        Encoded training data, or tuple of (train, valid) if valid_df provided
    """
    if method == "target":
        if target is None:
            raise ValueError("Target variable required for target encoding")
        encoder = TargetEncoder()
        train_encoded = encoder.fit_transform(train_df, target)
    else:
        encoder = NativeEncoder(encoding_method=method)
        train_encoded = encoder.fit_transform(train_df)

    if valid_df is not None:
        valid_encoded = encoder.transform(valid_df)
        return train_encoded, valid_encoded

    return train_encoded
