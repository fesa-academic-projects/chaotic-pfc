"""``chaotic-pfc paper`` — manuscript numbers and consistency audit.

Two actions are provided:

* ``numbers`` — recompute every number of the manuscript from
  ``data/sweeps/`` and emit JSON (``chaotic_pfc.paper.numbers``).
* ``audit`` — deterministically cross-check the manuscript sources against
  the bibliography, figures and identity tables (``chaotic_pfc.paper.audit``).
"""

from __future__ import annotations

import argparse
from typing import Any

from chaotic_pfc.paper import audit, figures, numbers


def add_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    """Register the top-level ``paper`` group and its actions."""
    parser = subparsers.add_parser(
        "paper",
        help="Manuscript tooling (recompute numbers, audit the sources).",
        description="Manuscript tooling for the JCIS article.",
    )
    actions = parser.add_subparsers(
        dest="paper_action",
        title="actions",
        metavar="<action>",
        required=True,
    )
    numbers.add_parser(actions)
    audit.add_parser(actions)
    figures.add_parser(actions)
