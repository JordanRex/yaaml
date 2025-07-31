# Testing Infrastructure

## Test Suite Overview

YAAML uses a comprehensive testing framework with realistic AutoML scenarios and Python 3.12+ feature validation.

### Test Categories

1. **Python 3.12+ Features** (`tests/test_python312_features.py`)
   - Walrus operator, union types, built-in generics
   - Modern typing throughout codebase
   - **Status**: 9/9 tests passing ✅

2. **Realistic AutoML** (`tests/test_realistic_automl.py`)
   - Mixed data types, missing values, realistic accuracy (60-95%)
   - Comprehensive preprocessing pipeline integration
   - **Status**: 6/6 tests passing ✅

3. **Module Integration** (`tests/test_module_integration.py`)
   - Cross-module compatibility with modern Python features
   - **Status**: 11/11 tests passing ✅

## Running Tests

### Unit and Integration Tests

```bash
# All tests with coverage
python scripts/run_tests.py --coverage

# Quick tests only (no slow tests)
python scripts/run_tests.py --fast

# Specific test file
python scripts/run_tests.py tests/test_main.py

# Using pytest directly
pytest tests/ --cov=yaaml --cov-report=term-missing -v

# Individual test suites
pytest tests/test_realistic_automl.py -v
pytest tests/test_python312_features.py -v
pytest tests/test_module_integration.py -v
```

**Coverage Reports**: Generated in `coverage/htmlcov/index.html` with automatic cleanup of stray files.

## Type Checking Tests

### Single File Type Checking

```bash
# Direct mypy on specific file (recommended)
mypy yaaml/main.py --follow-imports=silent --show-error-codes --no-error-summary

# Pre-commit mypy on specific file
pre-commit run mypy --files yaaml/main.py

# Filter pre-commit output to specific file
pre-commit run mypy --files yaaml/main.py 2>&1 | grep "main.py:"
```

### Multiple Files Type Checking

```bash
# All files in yaaml package
mypy yaaml/ --show-error-codes

# Entire project (respects exclusions)
mypy . --show-error-codes

# Specific files
mypy yaaml/main.py yaaml/encoding.py --follow-imports=silent --show-error-codes
```

### Consistency Verification

```bash
# Verify terminal vs pre-commit consistency (should match)
echo "Direct mypy errors:"
mypy yaaml/main.py --follow-imports=silent --show-error-codes --no-error-summary | wc -l

echo "Pre-commit mypy errors:"
pre-commit run mypy --files yaaml/main.py 2>&1 | grep "main.py:" | wc -l
```

### Build Directory Exclusion Test

```bash
# Test that build directories are properly excluded
mkdir -p build/test && echo "bad syntax" > build/test/bad.py
mypy . --show-error-codes  # Should ignore build/ folder
rm -rf build/  # Clean up
```

## Pre-commit Testing

### Running Pre-commit Hooks

```bash
# All hooks on all files
pre-commit run --all-files

# Specific hook on all files
pre-commit run mypy --all-files
pre-commit run black --all-files
pre-commit run flake8 --all-files

# Specific hook on specific files
pre-commit run mypy --files yaaml/main.py yaaml/encoding.py
pre-commit run black --files yaaml/main.py

# Install hooks (run automatically on commit)
pre-commit install
```

### Pre-commit Hook Types

1. **MyPy Type Checking**: Static type analysis
2. **Black**: Code formatting
3. **isort**: Import sorting
4. **flake8**: Linting and style checks
5. **Basic Checks**: Trailing whitespace, YAML syntax, etc.

## Testing Commands Reference

### Quick Testing Workflow

```bash
# Daily development testing
python scripts/run_tests.py --fast          # Quick tests only
mypy yaaml/main.py --follow-imports=silent  # Check current file

# Before committing
pre-commit run --all-files                  # All quality checks
python scripts/run_tests.py --coverage      # Full test suite

# Consistency verification
mypy yaaml/main.py --follow-imports=silent --no-error-summary | wc -l
pre-commit run mypy --files yaaml/main.py 2>&1 | grep "main.py:" | wc -l
```

### Troubleshooting Tests

#### Type Checking Issues

```bash
# Clear mypy cache
rm -rf .mypy_cache/

# Verify configuration
mypy --config-file pyproject.toml --help

# Check if exclusions work
mypy . --show-error-codes | grep -E "(build|venv|coverage)"  # Should be empty
```

#### Pre-commit Issues

```bash
# Update pre-commit hooks
pre-commit autoupdate

# Clean and reinstall
pre-commit clean
pre-commit install

# Debug specific hook
pre-commit run mypy --files yaaml/main.py --verbose
```

#### VS Code Integration Issues

```bash
# Verify extension is installed
code --list-extensions | grep mypy

# Check settings are applied
cat .vscode/settings.json | grep mypy

# Restart VS Code
# Cmd+Shift+P -> "Developer: Reload Window"
```

## Performance Validation

- **Target**: 60-95% accuracy on realistic datasets
- **Achieved**: 70% accuracy with challenging mixed-type data
- **Status**: ✅ Validated realistic AutoML behavior

## CI/CD Pipeline

Enhanced GitHub Actions pipeline with:

- Python 3.12+ feature validation
- Legacy typing detection
- Realistic AutoML integration testing
- Performance benchmarking
