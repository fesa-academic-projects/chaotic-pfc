"""Manuscript tooling: recompute the article numbers and audit the sources.

The implementation lives in :mod:`chaotic_pfc.paper.numbers` and
:mod:`chaotic_pfc.paper.audit`; both are exposed through the CLI as
``chaotic-pfc paper numbers`` and ``chaotic-pfc paper audit``.
"""

from __future__ import annotations

__all__ = ["audit", "numbers"]
