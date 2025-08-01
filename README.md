# YAAML (Yet Another AutoML)

![yaaml logo](./yaaml-logo.png)

[![Test PyPI version](https://img.shields.io/badge/Test%20PyPI-v0.1.0-blue)](https://test.pypi.org/project/yaaml/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/JordanRex/yaaml/actions/workflows/ci-cd.yml/badge.svg?branch=master)](https://github.com/JordanRex/yaaml/actions)

**Lightweight, production-ready AutoML built entirely on scikit-learn with Python 3.12+ modernization.**

## Quick Start

### Installation

```bash
# From Test PyPI (current releases)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ yaaml

# From PyPI (when available)
pip install yaaml

# From GitHub (latest development)
pip install git+https://github.com/JordanRex/yaaml.git

# From GitHub releases
pip install https://github.com/JordanRex/yaaml/archive/refs/tags/v0.1.0.tar.gz
```

### Basic Usage

```python
import pandas as pd
from yaaml import YAAMLAutoML
from sklearn.model_selection import train_test_split

# Load your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# AutoML in 3 lines
automl = YAAMLAutoML(mode='classification', max_evals=20, feature_engineering=True)
automl.fit(X_train, y_train)
accuracy = automl.score(X_test, y_test)
```

## Key Features

- **🚀 Zero Dependencies**: Built entirely on scikit-learn
- **🎯 Multi-Task**: Classification and regression
- **🔄 Complete Pipeline**: End-to-end ML with intelligent preprocessing
- **🧠 Smart Automation**: Feature engineering, selection, hyperparameter optimization
- **⚡ Lightweight**: Fast training, minimal resource footprint
- **🔍 Transparent**: Full visibility into decisions and transformations

## Advanced Usage

```python
# Handle mixed data types automatically
automl = YAAMLAutoML(
    mode='classification',
    imputation_strategy='iterative',  # Advanced missing values
    encoding_method='target',         # Target encoding
    sampling_strategy='balanced',     # Handle imbalanced data
    feature_selection=True,          # Intelligent feature selection
    max_evals=50,
    cv_folds=10
)

# Works with messy real-world data
mixed_data = pd.DataFrame({
    'numeric': [1.5, 2.3, np.nan, 4.1],
    'categorical': ['A', 'B', 'A', 'C'],
    'text': ['good', 'excellent', 'poor', 'good']
})

automl.fit(mixed_data, target)
predictions = automl.predict(new_data)
```

## Project Status

**Current**: v0.1.0 - Production ready with comprehensive Python 3.12+ modernization

- ✅ **Complete**: Classification, regression, feature engineering, hyperparameter optimization
- ✅ **Tested**: Realistic test suite achieving 70% accuracy on challenging datasets
- ✅ **Modern**: Full Python 3.12+ typing with walrus operator and union types
- ✅ **Validated**: Enhanced CI/CD pipeline with performance benchmarking

See [docs/status.md](docs/status.md) for detailed roadmap and [docs/testing.md](docs/testing.md) for testing infrastructure.

## Documentation

- **[Testing Infrastructure](docs/testing.md)**: Comprehensive test suite details
- **[Project Status](docs/status.md)**: Current status and roadmap
- **[Contributing](CONTRIBUTING.md)**: Development guidelines
- **[Examples](examples/)**: Usage examples and tutorials

## License

MIT License - see [LICENSE](LICENSE) for details.
