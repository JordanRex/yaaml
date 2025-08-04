"""Version detection for YAAML package."""


def get_version() -> str:
    """Get package version dynamically."""
    try:
        from importlib.metadata import version

        return str(version("yaaml"))
    except ImportError:
        try:
            import importlib_metadata

            return str(importlib_metadata.version("yaaml"))
        except ImportError:
            pass
    except Exception:
        pass

    # Development mode: use setuptools-scm
    try:
        from setuptools_scm import get_version as scm_get_version

        return str(scm_get_version(root="..", relative_to=__file__))
    except (ImportError, LookupError):
        return "unknown"
