# Development Guide

## Quick Setup

```bash
git clone https://github.com/JordanRex/yaaml.git
cd yaaml
python scripts/setup_dev.py
```

## Project Structure

```ascii
yaaml/
├── yaaml/              # Main package
├── tests/              # Test suite
├── docs/               # Documentation
└── scripts/            # Development tools
```

## Daily Development

```bash
# Run quick tests
python scripts/run_tests.py --fast

# Before committing
python scripts/run_tests.py --coverage
python scripts/check_project.py
```

## Dependencies

### Production

```toml
dependencies = [
    "numpy>=2.0.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.5.0"
]
```

### Development

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0", "pytest-cov>=4.1.0",
    "mypy>=1.17.0", "black>=23.0.0", "isort>=5.12.0", "flake8>=6.0.0",
    "pandas-stubs>=2.3.0", "pre-commit>=3.3.0",
    "build>=1.2.0", "twine>=6.0.0",
]
```

## Environment Management

```bash
# Production (11 packages)
uv sync --no-dev

# Development (62 packages)
uv sync --extra dev

# With documentation
uv sync --extra dev --extra docs

# Install all extras (dev + docs)
uv sync --all-extras
```

## Testing

```bash
# All tests
python scripts/run_tests.py

# With coverage
python scripts/run_tests.py --coverage

# Specific tests
python scripts/run_tests.py tests/test_main.py

# Fast tests only
python scripts/run_tests.py --fast
```

## Code Quality

```bash
# Format code
uv run black yaaml tests

# Sort imports
uv run isort yaaml tests

# Lint
uv run flake8 yaaml

# Type check
uv run mypy yaaml/

# All pre-commit checks
pre-commit run --all-files
```

## Building

```bash
# Build package
uv build

# Check package
uv run twine check dist/*

# Test PyPI upload
uv run twine upload --repository testpypi dist/*

# Production PyPI upload
uv run twine upload dist/*
```

## MyPy Configuration

```toml
[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
```

## VS Code Settings

```json
{
    "mypy-type-checker.importStrategy": "fromEnvironment",
    "python.analysis.typeCheckingMode": "off",
    "python.linting.enabled": false
}
```
