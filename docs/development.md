# Development Guide

## Quick Setup

### Prerequisites

- Python 3.12+
- Git

### One-Command Setup

```bash
git clone https://github.com/JordanRex/yaaml.git
cd yaaml
python scripts/setup_dev.py
```

This sets up the complete development environment: virtual env, dependencies, pre-commit hooks, and package installation.

## Project Structure

```ascii
yaaml/
├── yaaml/                    # Main package
│   ├── main.py              # Core YAAMLAutoML class
│   ├── encoding.py          # Categorical encoding
│   ├── miss_imputation.py   # Missing value handling
│   ├── feature_engineering.py # Feature creation
│   ├── feature_selection.py # Feature selection
│   ├── sampling.py          # Data sampling
│   └── helper_funcs.py      # Utilities
├── tests/                   # Test suite
├── docs/                    # Documentation
└── scripts/                 # Development tools
```

## Daily Development

```bash
# Activate environment
source venv/bin/activate

# Run tests during development
python scripts/run_tests.py --fast

# Before committing
python scripts/run_tests.py --coverage
python scripts/check_project.py
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

- **Formatting**: Black (auto-formatted via pre-commit)
- **Imports**: isort (auto-sorted via pre-commit)
- **Linting**: flake8
- **Type Hints**: Required for all functions
- **Tests**: Required for new features

## Package Building

### Local Build

```bash
# Install build dependencies
pip install build twine

# Build wheel and source distribution
python -m build

# Check package
python -m twine check dist/*
```

### Build Outputs

- **Wheel**: `dist/yaaml-X.Y.Z-py3-none-any.whl`
- **Source**: `dist/yaaml-X.Y.Z.tar.gz`

### Manual Distribution

```bash
# Test PyPI upload (current setup)
python -m twine upload --repository testpypi dist/*

# Production PyPI upload (future)
python -m twine upload dist/*
```

**Test PyPI Project**: [https://test.pypi.org/project/yaaml/](https://test.pypi.org/project/yaaml/)

### GitHub Secrets Required

**For Test PyPI uploads**:

- **Secret Name**: `TEST_PYPI_API_TOKEN`
- **Get Token**: [https://test.pypi.org/manage/account/token/](https://test.pypi.org/manage/account/token/)
- **Scope**: yaaml project
- **Add to GitHub**: Repository Settings > Secrets and variables > Actions

**For Production PyPI** (future):

- **Secret Name**: `PYPI_API_TOKEN`
- **Get Token**: [https://pypi.org/manage/account/token/](https://pypi.org/manage/account/token/)

## Type Checking & Linting

### Pure MyPy Approach

YAAML uses a **Pure MyPy approach** for consistent type checking across all environments:

- **Terminal**: `mypy` command with pyproject.toml config
- **VS Code**: MyPy Type Checker extension
- **CI/CD**: pre-commit with mypy hook
- **Configuration**: Single source of truth in `pyproject.toml`

### Configuration Files

#### pyproject.toml (Main Configuration)

```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
ignore_missing_imports = true

# Exclude build/cache directories
exclude = [
    "^build/.*", "^dist/.*", "^venv/.*", "^\\.venv/.*",
    "^coverage/.*", "^\\.mypy_cache/.*", "^\\.pytest_cache/.*"
]

[[tool.mypy.overrides]]
module = ["tests.*"]
disallow_untyped_defs = false
disallow_incomplete_defs = false
```

#### .vscode/settings.json (VS Code Integration)

```jsonc
{
    // MyPy Type Checker Extension (PRIMARY)
    "mypy-type-checker.importStrategy": "fromEnvironment",
    "mypy-type-checker.reportingScope": "workspace", 
    "mypy-type-checker.preferDaemon": false,
    "mypy-type-checker.args": [], // Uses pyproject.toml
    
    // Pylance (INTELLISENSE ONLY)
    "python.analysis.typeCheckingMode": "off",
    
    // Disable legacy linting
    "python.linting.enabled": false,
    "python.linting.mypyEnabled": false
}
```

#### .pre-commit-config.yaml (CI/CD Integration)

```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.3.0
  hooks:
    - id: mypy
      additional_dependencies: [types-requests, pandas-stubs, types-setuptools]
      # No args - uses pyproject.toml configuration
```

### Key Principles

1. **MyPy vs Pylance**: Use MyPy for type checking, Pylance only for intellisense
2. **Single Configuration**: pyproject.toml is the single source of truth
3. **Proper Exclusions**: Exclude build/, venv/, coverage/ directories
4. **VS Code Integration**: MyPy extension + disabled Pylance type checking

**For detailed testing commands and procedures, see [Testing Documentation](testing.md).**

## Release Process

```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md
# 3. Validate
python scripts/check_project.py
python scripts/run_tests.py --coverage

# 4. Automated release (recommended)
./scripts/release.sh

# OR manual build
python -m build && python -m twine upload dist/*
```

## Contributing

1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Run validation: `python scripts/check_project.py`
5. Submit pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
