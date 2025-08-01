#!/usr/bin/env python3
"""
Modern Test Runner using uv
===========================

Comprehensive test runner for YAAML using uv package manager.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_uv() -> bool:
    """Check if uv is available."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🧪 {description}")
    print(f"🔧 Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}\n")
        return False


def main() -> None:
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run YAAML tests using uv")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument(
        "--html-cov", action="store_true", help="Generate HTML coverage report"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--module", help="Run tests for specific module")
    parser.add_argument("--pattern", help="Run tests matching pattern")
    parser.add_argument("--failed", action="store_true", help="Rerun only failed tests")
    parser.add_argument("--parallel", type=int, help="Number of parallel workers")
    parser.add_argument(
        "--clean", action="store_true", help="Clean coverage files before running"
    )
    parser.add_argument(
        "test_paths", nargs="*", help="Specific test files or directories"
    )

    args = parser.parse_args()

    # Check if uv is available
    if not check_uv():
        print("❌ uv is not installed or not in PATH")
        print("💡 Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    if not (project_root / "pyproject.toml").exists():
        print("❌ pyproject.toml not found. Please run from project root.")
        sys.exit(1)

    print("🐍 YAAML Test Runner (uv edition)")
    print("=" * 40)

    # Clean coverage files if requested
    if args.clean:
        print("🧹 Cleaning up coverage files...")
        subprocess.run(
            ["rm", "-rf", "coverage/", ".coverage*", "htmlcov/"], capture_output=True
        )

    # Build test command
    cmd = ["uv", "run", "pytest"]

    # Add test paths or default to tests/
    if args.test_paths:
        cmd.extend(args.test_paths)
    elif args.module:
        cmd.append(f"tests/test_{args.module}.py")
    else:
        cmd.append("tests/")

    # Add coverage options
    if args.coverage or args.html_cov:
        cmd.extend(["--cov=yaaml", "--cov-report=term-missing"])
        if args.html_cov:
            cmd.append("--cov-report=html:coverage/htmlcov")
        cmd.append("--cov-report=xml:coverage/coverage.xml")

    # Add other options
    if args.verbose:
        cmd.append("-v")

    if args.fast:
        cmd.extend(["-m", "not slow"])

    if args.pattern:
        cmd.extend(["-k", args.pattern])

    if args.failed:
        cmd.append("--lf")

    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])

    # Always add these for better output
    cmd.extend(["--tb=short"])

    # Run the tests
    success = run_command(cmd, "Running tests")

    # Clean up any stray coverage files in root (they should be in coverage/)
    if args.coverage or args.html_cov:
        print("🧹 Cleaning up coverage files...")
        subprocess.run(["rm", "-f", ".coverage*"], capture_output=True)
        subprocess.run(["rm", "-rf", "htmlcov"], capture_output=True)

    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
