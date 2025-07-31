#!/usr/bin/env python3
"""
Modern YAAML Development Environment Setup Script
Uses uv for fast, reliable dependency management.
Python 3.12+ required.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def check_uv():
    """Check if uv is installed."""
    if not shutil.which("uv"):
        print("❌ uv is not installed!")
        print("\n📦 Install uv first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("   # or")  
        print("   pip install uv")
        sys.exit(1)
    
    result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
    print(f"✅ Found uv: {result.stdout.strip()}")


def check_project():
    """Verify we're in the right directory."""
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    print("✅ Found pyproject.toml")


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def main():
    """Set up the development environment."""
    print("🚀 Setting up YAAML Development Environment")
    print("=" * 50)
    
    # Step 1: Check prerequisites
    check_uv()
    check_project()
    
    # Step 2: Install everything with uv
    if not run_command(["uv", "sync", "--all-extras"], "Installing all dependencies"):
        sys.exit(1)
    
    # Step 3: Setup pre-commit hooks
    if not run_command(["uv", "run", "pre-commit", "install"], "Installing pre-commit hooks"):
        sys.exit(1)
    
    # Step 4: Quick verification
    print("\n🧪 Quick verification...")
    if not run_command(["uv", "run", "python", "-c", "import yaaml; print('✅ Package import works')"], "Testing package import"):
        sys.exit(1)
    
    # Success!
    print("\n" + "=" * 50)
    print("🎉 Development environment ready!")
    print("\n📋 Common commands:")
    print("   uv run pytest          # Run tests")
    print("   uv run mypy yaaml       # Type checking")
    print("   uv run black yaaml      # Format code")
    print("   uv run pre-commit run --all-files  # All checks")


if __name__ == "__main__":
    main()