#!/usr/bin/env python3
"""Quick fixes for common linting issues."""

import re
from pathlib import Path


def fix_comparison_to_true(content: str) -> str:
    """Fix comparison to True issues."""
    # Fix to use 'is True' or just the condition
    content = re.sub(r'(\w+)\s*==\s*True\b', r'\1', content)
    return content


def fix_bare_except(content: str) -> str:
    """Fix bare except clauses."""
    content = re.sub(r'except Exception:', 'except Exception:', content)
    return content


def fix_f_string_placeholders(content: str) -> str:
    """Fix f-strings without placeholders."""
    # Simple case: "string" -> "string"
    content = re.sub(r'f"([^{}"]*)"', r'"\1"', content)
    content = re.sub(r"f'([^{}']*)'", r"'\1'", content)
    return content


def process_file(file_path: Path) -> None:
    """Process a single Python file."""
    if not file_path.suffix == '.py':
        return
        
    try:
        content = file_path.read_text()
        original_content = content
        
        content = fix_comparison_to_true(content)
        content = fix_bare_except(content)
        content = fix_f_string_placeholders(content)
        
        if content != original_content:
            file_path.write_text(content)
            print(f"Fixed: {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    """Main function."""
    root = Path(".")
    
    # Process all Python files
    for py_file in root.rglob("*.py"):
        if "/.git/" not in str(py_file) and "/__pycache__/" not in str(py_file):
            process_file(py_file)


if __name__ == "__main__":
    main()
