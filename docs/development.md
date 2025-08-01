# Development Guide

## Quick Setup

```bash
git clone https://github.com/JordanRex/yaaml.git
cd yaaml
uv run python scripts/setup_dev.py
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
uv run python scripts/run_tests.py --fast

# Before committing
uv run python scripts/run_tests.py --coverage
uv run python scripts/check_project.py
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
    "mypy>=1.17.0", "ruff>=0.8.0",
    "pandas-stubs>=2.3.0", "pre-commit>=3.3.0",
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
uv run python scripts/run_tests.py

# With coverage
uv run python scripts/run_tests.py --coverage

# Specific tests
uv run python scripts/run_tests.py tests/test_main.py

# Fast tests only
uv run python scripts/run_tests.py --fast
```

## Code Quality

```bash
# Format code
uv run ruff format yaaml tests

# Lint and auto-fix
uv run ruff check yaaml --fix

# Type check
uv run mypy yaaml/

# All pre-commit checks
pre-commit run --all-files
```

## Building & Publishing

```bash
# Build package
uv build

# Publish to Test PyPI
uv publish --publish-url https://test.pypi.org/legacy/

# Publish to PyPI (production)
uv publish
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
