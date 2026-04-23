"""Sphinx configuration for chaotic-pfc."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from importlib.metadata import version as _pkg_version
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Path setup — make the package importable without installing it first.
# ─────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ─────────────────────────────────────────────────────────────────────────
# Project information
# ─────────────────────────────────────────────────────────────────────────
project = "chaotic-pfc"
author = "FESA TCC team"
copyright = f"{datetime.now().year}, {author} and contributors"
language = "en"  # keep the doc itself English regardless of the builder's locale

try:
    release = _pkg_version("chaotic-pfc")
except Exception:
    release = "0.3.0"
version = ".".join(release.split(".")[:2])


# ─────────────────────────────────────────────────────────────────────────
# Sphinx extensions
# ─────────────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


# ─────────────────────────────────────────────────────────────────────────
# autodoc / autosummary
# ─────────────────────────────────────────────────────────────────────────
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"


# ─────────────────────────────────────────────────────────────────────────
# napoleon — parse NumPy-style docstrings
# ─────────────────────────────────────────────────────────────────────────
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True


# ─────────────────────────────────────────────────────────────────────────
# intersphinx — clickable cross-references to external libraries
# ─────────────────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}


# ─────────────────────────────────────────────────────────────────────────
# HTML output — furo theme
# ─────────────────────────────────────────────────────────────────────────
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"{project} {release}"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/fesa-academic-projects/chaotic-pfc/",
    "source_branch": "main",
    "source_directory": "docs/",
}


# ─────────────────────────────────────────────────────────────────────────
# Miscellaneous
# ─────────────────────────────────────────────────────────────────────────
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = False  # start loose; tighten once the doc is fleshed out

suppress_warnings = ["autosectionlabel.*"]

autodoc_mock_imports: list[str] = []


# ─────────────────────────────────────────────────────────────────────────
# Dataclass duplicate-warning filter
# ─────────────────────────────────────────────────────────────────────────
# autosummary :recursive: generates per-attribute pages for every dataclass
# field, AND the class page itself lists the same fields via autodoc — so
# Sphinx emits 'duplicate object description' once per field. The content
# is correct; only the warning is noise. Sphinx 9 no longer accepts this
# category in ``suppress_warnings``, so we filter at the logging layer.
#
# The warning message is localised by the builder's ``LANG`` / ``LC_MESSAGES``
# setting (not by the doc's ``language`` option), so the substring differs
# between machines. Match every locale we actually run on: English on CI
# and ReadTheDocs, Portuguese (pt_BR) on local dev.

_DUPLICATE_SUBSTRINGS = (
    "duplicate object description",  # English
    "descrição duplicada de objeto",  # pt_BR
)


class _DuplicateFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(sub in msg for sub in _DUPLICATE_SUBSTRINGS)


logging.getLogger("sphinx").addFilter(_DuplicateFilter())
