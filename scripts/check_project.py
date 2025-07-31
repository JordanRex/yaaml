#!/usr/bin/env python3
"""
YAAML Project Status Checker
Validates the entire project structure and setup.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_mark(condition: bool, message: str) -> bool:
    """Print a check mark or X based on condition."""
    symbol = "✅" if condition else "❌"
    print(f"{symbol} {message}")
    return condition


def warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {message}")


def info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ️  {message}")


def check_python_version() -> bool:
    """Check Python version."""
    major, minor = sys.version_info[:2]
    version_ok = major == 3 and minor >= 8
    check_mark(version_ok, f"Python version: {major}.{minor} (required: 3.8+)")
    return version_ok


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    exists = Path(filepath).exists()
    check_mark(exists, f"{description}: {filepath}")
    return exists


def check_directory_structure() -> bool:
    """Check the basic directory structure."""
    print("\n📁 Directory Structure:")
    required_dirs = [
        ("yaaml/", "Main package directory"),
        ("tests/", "Test directory"),
        (".github/workflows/", "GitHub Actions directory"),
        ("docs/", "Documentation directory"),
    ]

    all_good = True
    for dir_path, description in required_dirs:
        exists = check_file_exists(dir_path, description)
        all_good = all_good and exists

    return all_good


def check_core_files() -> bool:
    """Check core project files."""
    print("\n📄 Core Files:")
    required_files = [
        ("pyproject.toml", "Project configuration"),
        ("uv.lock", "Dependency lock file"),
        ("README.md", "Project documentation"),
        ("LICENSE", "License file"),
        (".gitignore", "Git ignore file"),
        (".pre-commit-config.yaml", "Pre-commit configuration"),
    ]

    all_good = True
    for file_path, description in required_files:
        exists = check_file_exists(file_path, description)
        all_good = all_good and exists

    return all_good


def check_python_files() -> bool:
    """Check main Python files."""
    print("\n🐍 Python Files:")
    required_files = [
        ("yaaml/__init__.py", "Package init"),
        ("yaaml/main.py", "Main AutoML class"),
        ("yaaml/encoding.py", "Encoding module"),
        ("yaaml/miss_imputation.py", "Missing value imputation"),
        ("yaaml/feature_engineering.py", "Feature engineering"),
        ("tests/conftest.py", "Test configuration"),
        ("tests/test_main.py", "Main module tests"),
        ("scripts/setup_dev.py", "Development setup script"),
        ("scripts/run_tests.py", "Test runner script"),
    ]

    all_good = True
    for file_path, description in required_files:
        exists = check_file_exists(file_path, description)
        all_good = all_good and exists

    return all_good


def check_imports() -> bool:
    """Check if main modules can be imported."""
    print("\n📦 Import Checks:")
    modules_to_check = [
        ("yaaml", "Main package"),
        ("yaaml.main", "Main module"),
        ("yaaml.encoding", "Encoding module"),
        ("yaaml.miss_imputation", "Imputation module"),
    ]

    all_good = True
    for module_name, description in modules_to_check:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                check_mark(True, f"{description} import: {module_name}")
            else:
                check_mark(False, f"{description} import: {module_name} (not found)")
                all_good = False
        except Exception as e:
            check_mark(False, f"{description} import: {module_name} (error: {e})")
            all_good = False

    return all_good


def check_dependencies() -> bool:
    """Check if required tools are available."""
    print("\n🛠️  Dependencies:")
    tools = [
        ("git", "Git version control"),
        ("python3", "Python 3"),
        ("pip", "Python package installer"),
    ]

    optional_tools = [
        ("black", "Code formatter"),
        ("flake8", "Linter"),
        ("pytest", "Test runner"),
        ("pre-commit", "Pre-commit hooks"),
    ]

    all_good = True
    for tool, description in tools:
        available = shutil.which(tool) is not None
        check_mark(available, f"{description}: {tool}")
        all_good = all_good and available

    print("\n🔧 Optional Tools:")
    for tool, description in optional_tools:
        available = shutil.which(tool) is not None
        if available:
            check_mark(True, f"{description}: {tool}")
        else:
            warning(f"{description} not found: {tool} (install with pip)")

    return all_good


def check_git_status() -> bool:
    """Check git repository status."""
    print("\n📋 Git Status:")
    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"], capture_output=True, text=True
        )
        if result.returncode == 0:
            check_mark(True, "Git repository initialized")

            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True
            )
            uncommitted = len(result.stdout.strip()) > 0
            if uncommitted:
                warning("Uncommitted changes detected")
                info("Run 'git status' to see changes")
            else:
                check_mark(True, "No uncommitted changes")

            return True
        else:
            check_mark(False, "Git repository not initialized")
            return False
    except Exception as e:
        check_mark(False, f"Git check failed: {e}")
        return False


def run_quick_tests() -> bool:
    """Run a quick test to ensure basic functionality."""
    print("\n🧪 Quick Functionality Test:")
    try:
        # Try to import and create the main class
        from yaaml.main import YAAMLAutoML

        automl = YAAMLAutoML(mode="classification")
        check_mark(True, "YAAMLAutoML class can be instantiated")

        # Try basic encoding functions
        from yaaml.encoding import NativeEncoder, encode_categorical_features

        check_mark(True, "Encoding functions available")

        # Try imputation class
        from yaaml.miss_imputation import DataFrameImputer

        imputer = DataFrameImputer()
        check_mark(True, "DataFrameImputer class available")

        return True
    except Exception as e:
        check_mark(False, f"Functionality test failed: {e}")
        return False


def main() -> None:
    """Run all checks."""
    print("🔍 YAAML Project Status Check")
    print("=" * 40)

    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", check_directory_structure),
        ("Core Files", check_core_files),
        ("Python Files", check_python_files),
        ("Dependencies", check_dependencies),
        ("Git Status", check_git_status),
        ("Import Checks", check_imports),
        ("Quick Tests", run_quick_tests),
    ]

    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results.append((check_name, False))

    # Summary
    print("\n" + "=" * 40)
    print("📊 Summary:")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        symbol = "✅" if result else "❌"
        print(f"{symbol} {check_name}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All checks passed! Your YAAML project is ready for development.")
    else:
        print("⚠️  Some checks failed. Please review and fix the issues above.")

    # Recommendations
    print("\n💡 Next Steps:")
    if passed < total:
        print("1. Fix the failing checks above")
        print("2. Run this script again to verify fixes")
    print("3. Run the development setup: python scripts/setup_dev.py")
    print("4. Run tests: python scripts/run_tests.py")
    print("5. Start developing!")


if __name__ == "__main__":
    main()
