"""Sphinx configuration for chaotic-pfc: bilingual (EN + pt_BR)."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Path setup: make the package importable without installing it first
# ─────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ─────────────────────────────────────────────────────────────────────────
# Project information
# ─────────────────────────────────────────────────────────────────────────
project = "chaotic-pfc"
author = "Roger Freitas Pereira"
copyright = f"{datetime.now().year}, {author} and contributors"
# Read the Docs injects READTHEDOCS_LANGUAGE per project (e.g. "en", "pt-br").
# Locally, fall back to SPHINX_LANGUAGE (set by Makefile/CLI) or English.
_rtd_lang = os.environ.get("READTHEDOCS_LANGUAGE")
if _rtd_lang:
    # RTD uses kebab-case ("pt-br"); Sphinx expects "pt_BR".
    language = _rtd_lang.replace("-", "_") if _rtd_lang != "en" else "en"
    if language == "pt_br":
        language = "pt_BR"
else:
    language = os.environ.get("SPHINX_LANGUAGE", "en")
version = "0.7.0"
release = "0.7.0"

# ─────────────────────────────────────────────────────────────────────────
# i18n / l10n support
# ─────────────────────────────────────────────────────────────────────────
locale_dirs = ["locale"]
gettext_compact = False
gettext_uuid = False
gettext_location = True

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
    "sphinx.ext.ifconfig",
    "sphinx.ext.todo",
    "sphinx_copybutton",
]

source_suffix = ".rst"

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
# napoleon: parse NumPy-style docstrings
# ─────────────────────────────────────────────────────────────────────────
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# ─────────────────────────────────────────────────────────────────────────
# intersphinx: cross-references to external libraries
# ─────────────────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# ─────────────────────────────────────────────────────────────────────────
# HTML output: furo theme
# ─────────────────────────────────────────────────────────────────────────
html_theme = "furo"
html_title = f"{project} v{release}"
html_baseurl = "https://chaotic-pfc.readthedocs.io/"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/fesa-academic-projects/chaotic-pfc/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/fesa-academic-projects/chaotic-pfc/",
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
}

# ─────────────────────────────────────────────────────────────────────────
# LaTeX / PDF output
# ─────────────────────────────────────────────────────────────────────────
latex_engine = "xelatex"
latex_show_pagerefs = False
latex_show_urls = "footnote"
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "preamble": r"""
    \usepackage{amsmath,amssymb}
    """,
}

latex_documents = [
    (
        "index",
        "chaotic-pfc.tex",
        f"{project} Documentation",
        author,
        "manual",
        False,
    ),
]

# ─────────────────────────────────────────────────────────────────────────
# EPUB output
# ─────────────────────────────────────────────────────────────────────────
epub_title = f"{project} Documentation"
epub_author = author
epub_show_urls = "footnote"
epub_language = language
epub_uid = "chaotic-pfc-docs"

# ─────────────────────────────────────────────────────────────────────────
# Miscellaneous
# ─────────────────────────────────────────────────────────────────────────
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "locale/**/*.mo",
    "locale/**/LC_MESSAGES/*.pot",
]

nitpicky = False

suppress_warnings = [
    "autosectionlabel.*",
    "epub.unknown_project_files",
]

autodoc_mock_imports: list[str] = []

numfig = True

# todo extension
todo_include_todos = False

# copybutton extension
copybutton_prompt_text = r">>> |\.\.\. |\$ |chaotic-pfc |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# ─────────────────────────────────────────────────────────────────────────
# Dataclass duplicate-warning filter
# ─────────────────────────────────────────────────────────────────────────
_DUPLICATE_SUBSTRINGS = (
    "duplicate object description",
    "descrição duplicada de objeto",
)


class _DuplicateFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(sub in msg for sub in _DUPLICATE_SUBSTRINGS)


logging.getLogger("sphinx").addFilter(_DuplicateFilter())
