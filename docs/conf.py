"""Sphinx configuration for the xeos documentation."""

from importlib.metadata import version as _pkg_version

# -- Project information -----------------------------------------------------
project = "xeos"
author = "Henri F. Drake"
copyright = "2026, Henri F. Drake"

# Single-source the version from the installed package.
try:
    release = _pkg_version("xeos")
except Exception:  # pragma: no cover - fallback when not installed
    release = "0.0.0"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Don't fail the build if optional backends can't import their heavy deps.
autodoc_mock_imports = ["numba"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"xeos {release}"
