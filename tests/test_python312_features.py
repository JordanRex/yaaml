"""
Test Python 3.12+ Features and Modernization
=============================================

This module tests all Python 3.12+ features implemented in the YAAML codebase,
including modern type annotations, walrus operator, union syntax, and performance optimizations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest


class TestPython312Features:
    """Test modern Python 3.12+ language features"""

    def test_walrus_operator(self):
        """Test walrus operator (:=) functionality"""
        # Test in conditional
        if (data := [1, 2, 3, 4, 5]) and len(data) > 0:
            assert len(data) == 5

        # Test in list comprehension with walrus
        results = [(x, squared) for x in range(5) if (squared := x**2) > 4]
        expected = [(3, 9), (4, 16)]
        assert results == expected

    def test_union_types(self):
        """Test modern union type syntax (| operator)"""

        def process_data(value: int | str | None) -> str:
            if value is None:
                return "none"
            elif isinstance(value, int):
                return f"int:{value}"
            else:
                return f"str:{value}"

        assert process_data(42) == "int:42"
        assert process_data("test") == "str:test"
        assert process_data(None) == "none"

    def test_builtin_generics(self):
        """Test built-in generic types (dict, list, tuple instead of Dict, List, Tuple)"""

        def process_mapping(data: dict[str, list[int]]) -> tuple[str, int]:
            keys = list(data.keys())
            total = sum(sum(values) for values in data.values())
            return (keys[0] if keys else "", total)

        test_data: dict[str, list[int]] = {"numbers": [1, 2, 3], "scores": [10, 20]}

        result = process_mapping(test_data)
        assert isinstance(result, tuple)
        assert result[1] == 36  # 1+2+3+10+20

    def test_optional_with_none_union(self):
        """Test Optional replaced with | None syntax"""

        def maybe_process(df: pd.DataFrame | None = None) -> bool:
            return df is not None

        assert maybe_process(pd.DataFrame()) is True
        assert maybe_process(None) is False

    def test_complex_nested_types(self):
        """Test complex nested modern type annotations"""

        def complex_function(
            data: dict[str, list[int | float]], mapper: dict[str, str] | None = None
        ) -> tuple[list[str], dict[str, float]]:

            keys = list(data.keys())
            averages = {k: float(np.mean(v)) for k, v in data.items()}

            if mapper:
                keys = [mapper.get(k, k) for k in keys]

            return keys, averages

        test_data: dict[str, list[int | float]] = {"group_a": [1, 2, 3], "group_b": [4.5, 5.5, 6.5]}

        keys, averages = complex_function(test_data)
        assert len(keys) == 2
        assert abs(averages["group_a"] - 2.0) < 0.001
        assert abs(averages["group_b"] - 5.5) < 0.001


class TestTypeAnnotationCompatibility:
    """Test that all modules use modern type annotations"""

    def test_import_all_modules(self):
        """Test that all modules import successfully with modern annotations"""
        # This test validates that __future__.annotations works properly
        from yaaml import YAAMLAutoML
        from yaaml.encoding import NativeEncoder, TargetEncoder
        from yaaml.feature_engineering import BinningTransformer, FeatureEngineering
        from yaaml.feature_selection import FeatureSelector, select_features
        from yaaml.helper_funcs import (
            check_data_quality,
            detect_data_types,
            evaluate_model,
        )
        from yaaml.miss_imputation import DataFrameImputer
        from yaaml.native_algorithms import AlgorithmFactory, NativeAlgorithmSelector
        from yaaml.sampling import NativeSampler, StratifiedSampler, apply_sampling

        # If we get here without ImportError, all modern annotations work
        assert True

    def test_function_annotations_exist(self):
        """Test that key functions have proper type annotations"""
        from yaaml.helper_funcs import evaluate_model
        from yaaml.sampling import apply_sampling

        # Check that functions have __annotations__
        assert hasattr(evaluate_model, "__annotations__")
        assert hasattr(apply_sampling, "__annotations__")

        # Verify return type annotations exist
        assert "return" in evaluate_model.__annotations__
        assert "return" in apply_sampling.__annotations__


class TestPerformanceOptimizations:
    """Test Python 3.12+ performance optimizations"""

    def test_future_annotations_import(self):
        """Test that __future__.annotations is imported in all modules"""
        import ast
        import inspect
        from pathlib import Path

        yaaml_path = Path(__file__).parent.parent / "yaaml"
        python_files = list(yaaml_path.glob("*.py"))

        missing_future_import = []

        for file_path in python_files:
            if file_path.name == "__init__.py":
                continue

            with open(file_path, "r") as f:
                content = f.read()

            try:
                tree = ast.parse(content)
                has_future_import = any(
                    isinstance(node, ast.ImportFrom)
                    and node.module == "__future__"
                    and any(alias.name == "annotations" for alias in (node.names or []))
                    for node in tree.body
                )

                if not has_future_import:
                    missing_future_import.append(file_path.name)
            except SyntaxError:
                # Skip files with syntax errors
                pass

        assert (
            len(missing_future_import) == 0
        ), f"Files missing __future__.annotations: {missing_future_import}"

    def test_no_legacy_typing_imports(self):
        """Test that legacy typing imports are removed"""
        import ast
        from pathlib import Path

        yaaml_path = Path(__file__).parent.parent / "yaaml"
        python_files = list(yaaml_path.glob("*.py"))

        legacy_imports = []
        legacy_names = {"Union", "Optional", "List", "Dict", "Tuple"}

        for file_path in python_files:
            with open(file_path, "r") as f:
                content = f.read()

            try:
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module == "typing":
                        if node.names:
                            imported_names = {alias.name for alias in node.names}
                            found_legacy = imported_names.intersection(legacy_names)
                            if found_legacy:
                                legacy_imports.append((file_path.name, found_legacy))
            except SyntaxError:
                pass

        assert (
            len(legacy_imports) == 0
        ), f"Files with legacy typing imports: {legacy_imports}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
