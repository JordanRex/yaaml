# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-07-31

### Added

- **YAAMLAutoML**: Complete automated ML pipeline for classification and regression
- **Zero Dependencies**: Pure scikit-learn implementation, no external ML frameworks
- **Intelligent Preprocessing**: Automated missing values, encoding, feature engineering
- **Smart Model Selection**: Hyperparameter optimization with cross-validation
- **Production Ready**: Comprehensive error handling, validation, and reproducibility

### Python 3.12+ Modernization

- **Modern Typing**: Union types (`int | str`), built-in generics (`list[str]`)
- **Walrus Operator**: Performance optimizations with `:=` syntax
- **Future Annotations**: Eliminated legacy `typing` imports

### Testing Infrastructure

- **Realistic Testing**: 60-95% accuracy targets with challenging datasets
- **Comprehensive Coverage**: 26 tests across Python features, AutoML, and integration
- **Enhanced CI/CD**: Performance validation and legacy typing detection

### Removed

- H2O AutoML, Hyperopt, and other external ML dependencies
- Legacy typing and outdated Python syntax
