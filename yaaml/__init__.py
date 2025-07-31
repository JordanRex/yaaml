"""YAAML: Yet Another AutoML Library

A lightweight, native AutoML library built on scikit-learn.
"""

__version__ = "0.1.0"
__author__ = "Varun Rajan"
__email__ = "varun@example.com"
__description__ = "Yet Another AutoML - Native Python AutoML library"

from .encoding import NativeEncoder, TargetEncoder, encode_categorical_features
from .feature_engineering import (
    BinningTransformer,
    FeatureEngineering,
    create_advanced_features,
)
from .feature_selection import FeatureSelector, select_features
from .helper_funcs import (
    check_data_quality,
    detect_data_types,
    evaluate_model,
    memory_optimization,
    print_model_summary,
    split_features_target,
)

# Import main classes
from .main import YAAMLAutoML
from .miss_imputation import DataFrameImputer, impute_missing_values
from .native_algorithms import AlgorithmFactory, NativeAlgorithmSelector
from .sampling import (
    NativeSampler,
    StratifiedSampler,
    analyze_class_distribution,
    apply_sampling,
)

__all__ = [
    # Main AutoML class
    "YAAMLAutoML",
    # Algorithm selection
    "AlgorithmFactory",
    "NativeAlgorithmSelector",
    # Data preprocessing
    "DataFrameImputer",
    "impute_missing_values",
    "NativeEncoder",
    "TargetEncoder",
    "encode_categorical_features",
    # Feature engineering
    "FeatureEngineering",
    "BinningTransformer",
    "create_advanced_features",
    "FeatureSelector",
    "select_features",
    # Sampling
    "NativeSampler",
    "StratifiedSampler",
    "apply_sampling",
    "analyze_class_distribution",
    # Helper functions
    "evaluate_model",
    "detect_data_types",
    "check_data_quality",
    "memory_optimization",
    "split_features_target",
    "print_model_summary",
    # Package metadata
    "__version__",
]
