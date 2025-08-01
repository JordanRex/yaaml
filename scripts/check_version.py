#!/usr/bin/env python3
"""
🔢 YAAML Version Helper

This script helps you understand and manage versioning in your project.
Run this to see your current version and get suggestions for the next release.
"""

import shlex
import subprocess
import sys


def run_command(cmd: str) -> str:
    """Run a command and return its output."""
    try:
        # Handle specific commands safely without shell=True
        if cmd == "git rev-parse --git-dir":
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                check=True,
            )
        elif cmd == "git tag --list 'v*' --sort=-version:refname":
            result = subprocess.run(
                ["git", "tag", "--list", "v*", "--sort=-version:refname"],
                capture_output=True,
                text=True,
                check=True,
            )
        elif cmd == "python -m setuptools_scm":
            result = subprocess.run(
                ["python", "-m", "setuptools_scm"],
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            # For other commands, use shlex.split safely
            result = subprocess.run(
                shlex.split(cmd), capture_output=True, text=True, check=True
            )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"


def get_current_version() -> str:
    """Get the current version from setuptools-scm."""
    try:
        # Try to get version from setuptools-scm
        result = run_command("python -m setuptools_scm")
        return result
    except Exception:
        return "No version available (need git tags)"


def get_git_tags() -> list[str]:
    """Get all git tags."""
    tags = run_command("git tag --list 'v*' --sort=-version:refname")
    return tags.split("\n") if tags and not tags.startswith("Error") else []


def suggest_next_version(current_tags: list[str]) -> str | list[str]:
    """Suggest next version based on existing tags."""
    if not current_tags:
        return "v0.1.0 (first release)"

    latest = current_tags[0] if current_tags else "v0.0.0"

    # Parse version
    try:
        version_part = latest[1:]  # Remove 'v' prefix
        parts = version_part.split(".")
        major, minor, patch = (
            int(parts[0]),
            int(parts[1]),
            int(parts[2]) if len(parts) > 2 else 0,
        )

        suggestions = []
        suggestions.append(f"v{major}.{minor}.{patch + 1} (patch - bug fixes)")
        suggestions.append(f"v{major}.{minor + 1}.0 (minor - new features)")
        suggestions.append(f"v{major + 1}.0.0 (major - breaking changes)")

        return suggestions
    except Exception:
        return ["v0.1.0 (suggested first release)"]


def main() -> None:
    """Main function to display version information."""
    print("🔢 YAAML Version Helper")
    print("=" * 50)
    print()

    # Check if we're in a git repo
    git_check = run_command("git rev-parse --git-dir")
    if git_check.startswith("Error"):
        print("❌ Not in a git repository!")
        sys.exit(1)

    # Get current version
    print("📦 Current Package Version:")
    current_version = get_current_version()
    print(f"   {current_version}")
    print()

    # Get git tags
    print("🏷️  Existing Git Tags:")
    tags = get_git_tags()
    if tags:
        for tag in tags[:5]:  # Show last 5 tags
            print(f"   {tag}")
        if len(tags) > 5:
            print(f"   ... and {len(tags) - 5} more")
    else:
        print("   No version tags found")
    print()

    # Suggest next versions
    print("💡 Suggested Next Versions:")
    suggestions = suggest_next_version(tags)
    if isinstance(suggestions, list):
        for suggestion in suggestions:
            print(f"   {suggestion}")
    else:
        print(f"   {suggestions}")
    print()

    # Explain the new workflow
    print("🚀 New Workflow (Dynamic Versioning):")
    print("   1. The version is now determined by git tags automatically")
    print("   2. When you input a version in GitHub Actions, it creates a git tag")
    print("   3. The package is built using that tag's version")
    print("   4. No more version mismatches! 🎉")
    print()

    print("📝 To release a new version:")
    print("   • Test release: Use v0.x.x format (e.g., v0.1.0, v0.2.0)")
    print("   • Production release: Use v1.x.x+ format (e.g., v1.0.0, v1.1.0)")
    print("   • The GitHub workflow will handle the rest!")


if __name__ == "__main__":
    main()
