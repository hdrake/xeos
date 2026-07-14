"""Sphinx configuration for the xeos documentation."""

import os
import sys
from importlib.metadata import version as _pkg_version

# Make the local build-time extension in docs/_ext importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_ext"))

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
    # myst_nb supersedes myst_parser: it pulls in the MyST Markdown parser and
    # adds executed-notebook support. Listing myst_parser as well would raise a
    # "already registered" conflict, so we do not.
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    # Local extension: writes docs/_generated/selector_table.md from xeos so the
    # selector reference table can never drift from the code.
    "gen_selector_table",
]

# -- myst-nb (notebook execution) --------------------------------------------
# Execute notebooks at build time and fail loudly on any error, so a broken
# example cannot slip through the -W (warnings-as-errors) CI build.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300  # seconds; the notebook downloads a dataset via pooch

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

# myst-nb parses both plain MyST Markdown (.md) and text-based notebooks; a
# file is executed as a notebook only when it carries a jupytext/kernelspec
# front-matter (docs/examples/*.md), otherwise it renders as ordinary Markdown.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
# _generated/ holds build-time files pulled in via {include}; they must not be
# treated as standalone documents (they carry no title and belong to no toctree).
exclude_patterns = ["_build", "_generated", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"xeos {release}"
