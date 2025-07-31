#!/usr/bin/env python3
"""
Test runner script for YAAML.
Provides convenient test running with different options.
Handles different Python commands automatically.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_python_command():
    """Find the appropriate Python command to use."""
    # If we're already running with Python, use the same executable
    current_python = sys.executable
    if current_python and os.path.isfile(current_python):
        return current_python

    # Otherwise, search for available Python commands
    python_commands = ["python3", "python", "py"]

    for cmd in python_commands:
        if shutil.which(cmd):
            try:
                result = subprocess.run(
                    [cmd, "--version"], capture_output=True, text=True
                )
                if result.returncode == 0 and "Python 3." in result.stdout:
                    return cmd
            except Exception:
                continue

    return "python3"  # fallback


def run_command(command, description="Running command"):
    """Run a shell command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def cleanup_coverage_files():
    """Clean up any coverage files that may have been created in the root directory."""
    root_dir = Path(".")

    # Remove coverage files from root
    coverage_patterns = [".coverage*", "coverage.xml"]
    for pattern in coverage_patterns:
        for file in root_dir.glob(pattern):
            try:
                file.unlink()
                print(f"🧹 Cleaned up: {file}")
            except Exception as e:
                print(f"⚠️  Could not remove {file}: {e}")

    # Remove htmlcov directory from root if it exists
    htmlcov_root = root_dir / "htmlcov"
    if htmlcov_root.exists():
        try:
            shutil.rmtree(htmlcov_root)
            print(f"🧹 Cleaned up: {htmlcov_root}")
        except Exception as e:
            print(f"⚠️  Could not remove {htmlcov_root}: {e}")


def ensure_coverage_directory():
    """Ensure coverage directory exists."""
    coverage_dir = Path("coverage")
    coverage_dir.mkdir(exist_ok=True)
    return coverage_dir


def main():
    parser = argparse.ArgumentParser(description="YAAML Test Runner")
    parser.add_argument(
        "--coverage", action="store_true", help="Run tests with coverage"
    )
    parser.add_argument(
        "--html-cov", action="store_true", help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose test output"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only fast tests (skip slow integration tests)",
    )
    parser.add_argument("--module", "-m", help="Run tests for specific module only")
    parser.add_argument("--pattern", "-k", help="Run tests matching pattern")
    parser.add_argument(
        "--failed", action="store_true", help="Re-run only failed tests from last run"
    )
    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        help="Run tests in parallel (requires pytest-xdist)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean up coverage files after tests"
    )
    parser.add_argument(
        "test_paths", nargs="*", help="Specific test files or directories to run"
    )

    args = parser.parse_args()

    # Ensure coverage directory exists
    coverage_dir = ensure_coverage_directory()

    # Find the appropriate Python command
    python_cmd = find_python_command()
    print(f"🐍 Using Python: {python_cmd}")

    # Build pytest command
    cmd_parts = [python_cmd, "-m", "pytest"]

    if args.verbose:
        cmd_parts.append("-v")

    if args.coverage:
        cmd_parts.extend(["--cov=yaaml", "--cov-report=term-missing"])

    if args.html_cov:
        cmd_parts.extend(["--cov=yaaml", "--cov-report=html:coverage/htmlcov"])

    if args.fast:
        cmd_parts.extend(["-m", "not slow"])

    if args.module:
        cmd_parts.append(f"tests/test_{args.module}.py")
    elif args.test_paths:
        cmd_parts.extend(args.test_paths)
    else:
        cmd_parts.append("tests/")

    if args.pattern:
        cmd_parts.extend(["-k", args.pattern])

    if args.failed:
        cmd_parts.append("--l")

    if args.parallel:
        cmd_parts.extend(["-n", str(args.parallel)])

    # Run the tests
    cmd = " ".join(cmd_parts)
    print(f"🧪 Running tests with command: {cmd}")

    success = run_command(cmd, "Running tests")

    # Clean up any stray coverage files that might have been created in root
    print("\n🧹 Cleaning up coverage files...")
    cleanup_coverage_files()

    if args.html_cov and success:
        print(
            f"\n📊 HTML coverage report generated in {coverage_dir}/htmlcov/index.html"
        )

    if args.clean:
        print("\n🧹 Cleaning up coverage directory...")
        if coverage_dir.exists():
            shutil.rmtree(coverage_dir)
            print(f"✅ Removed {coverage_dir}")

    if not success:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")


if __name__ == "__main__":
    main()
