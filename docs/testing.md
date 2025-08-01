# Testing Infrastructure

## Running Tests

```bash
# All tests with coverage
python scripts/run_tests.py --coverage

# Quick tests only
python scripts/run_tests.py --fast

# Specific test file
python scripts/run_tests.py tests/test_main.py

# Using pytest directly
## Type Checking

```bash
# Check specific file
mypy yaaml/main.py --follow-imports=silent --show-error-codes

# Check all files
mypy yaaml/ --show-error-codes

# Pre-commit mypy
pre-commit run mypy --files yaaml/main.py
```

## MyPy Configuration

```toml
[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
```

## Pre-commit

```bash
# All hooks on all files
pre-commit run --all-files

# Specific hook
pre-commit run mypy --all-files
pre-commit run ruff-format --all-files

# Install hooks
pre-commit install


## Troubleshooting

```bash
# Clear mypy cache
rm -rf .mypy_cache/

# Update pre-commit hooks
pre-commit autoupdate

# Clean and reinstall
pre-commit clean
pre-commit install

# Verify VS Code extension
code --list-extensions | grep mypy
```
