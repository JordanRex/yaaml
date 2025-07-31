# Development Scripts

This directory contains development and maintenance scripts for the YAAML project.

## Scripts Overview

### 🏗️ **setup_dev.py**

**Purpose**: One-time development environment setup

```bash
python scripts/setup_dev.py
```

**What it does**:

- Creates virtual environment if needed
- Installs development dependencies
- Sets up pre-commit hooks
- Installs package in development mode
- Validates the setup

**Use when**: Setting up the project for the first time or after a fresh clone.

---

### 🧪 **run_tests.py**

**Purpose**: Comprehensive test runner with coverage and cleanup

```bash
# Basic test run
python scripts/run_tests.py

# With coverage report
python scripts/run_tests.py --coverage --html-cov

# Run specific tests
python scripts/run_tests.py tests/test_main.py

# Fast tests only (skip integration)
python scripts/run_tests.py --fast

# Clean up coverage files after
python scripts/run_tests.py --coverage --clean
```

**Features**:

- Automatic Python environment detection
- Coverage reporting with organized file management
- Parallel test execution support
- Automatic cleanup of coverage artifacts
- Selective test running (by module, pattern, or file)

---

### 🔍 **check_project.py**

**Purpose**: Project health and structure validation

```bash
python scripts/check_project.py
```

**What it checks**:

- Package structure and imports
- Test coverage and functionality
- Code quality and linting
- Documentation completeness
- Build system configuration
- Git repository status

**Use when**: Before committing changes or preparing releases.

---

### 🚀 **release.sh**

**Purpose**: Automated package build and release workflow

```bash
# Build and test package
./scripts/release.sh

# Build for production deployment
./scripts/release.sh --production
```

**What it does**:

- Validates project structure
- Runs comprehensive tests
- Builds source and wheel distributions
- Performs security checks
- Publishes to PyPI (test/production)
- Creates git tags

**Use when**: Preparing package releases.

## Quick Development Workflow

### Initial Setup

```bash
# 1. Clone and setup
git clone <repository>
cd yaaml
python scripts/setup_dev.py

# 2. Verify setup
python scripts/check_project.py
```

### Daily Development

```bash
# 1. Run tests during development
python scripts/run_tests.py --fast

# 2. Full validation before commit
python scripts/run_tests.py --coverage
python scripts/check_project.py

# 3. Pre-release checks
./scripts/release.sh --dry-run
```

### Release Process

```bash
# 1. Final validation
python scripts/check_project.py
python scripts/run_tests.py --coverage --clean

# 2. Update version in pyproject.toml
# 3. Update CHANGELOG.md

# 4. Build and release
./scripts/release.sh
```

## Script Dependencies

All scripts are designed to:

- **Auto-detect Python environment** (handles python3, python, py commands)
- **Work with virtual environments** (detects venv, creates if needed)
- **Provide clear output** with emojis and status messages
- **Handle errors gracefully** with helpful error messages
- **Support CI/CD environments** (GitHub Actions compatible)

## Environment Variables

- `YAAML_DEV_MODE`: Set to `true` for additional development features
- `CI`: Automatically detected in CI environments
- `PYTHON_VERSION`: Override Python version detection

## Troubleshooting

**Virtual Environment Issues**:

```bash
# Recreate virtual environment
rm -rf venv
python scripts/setup_dev.py
```

**Permission Issues** (Linux/macOS):

```bash
# Make scripts executable
chmod +x scripts/*.sh
```

**Coverage Files Not Cleaning**:

```bash
# Manual cleanup
python scripts/run_tests.py --clean
rm -rf coverage/ .coverage* coverage.xml htmlcov/
```

---

For more detailed development information, see [`docs/development.md`](../docs/development.md).
