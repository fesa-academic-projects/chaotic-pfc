"""``chaotic-pfc sweeps`` — maintenance for the versioned sweep checkpoints.

Three actions are provided:

* ``regenerate`` — recompute the versioned ``data/sweeps`` checkpoints with
  the current kernel and rewrite the ``SHA256SUMS`` manifest
  (``chaotic_pfc.sweeps.regenerate``).
* ``compare`` — compare two sweep directories file by file
  (``chaotic_pfc.sweeps.compare``).
* ``stats`` — per-configuration statistics read from the ``.npz`` checkpoints
  (``chaotic_pfc.sweeps.report``).
"""

from __future__ import annotations

import argparse

from chaotic_pfc.sweeps import compare, regenerate, report


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the top-level ``sweeps`` group and its actions."""
    parser = subparsers.add_parser(
        "sweeps",
        help="Sweep maintenance (regenerate, compare, report).",
        description="Maintenance tooling for the versioned sweep checkpoints.",
    )
    actions = parser.add_subparsers(
        dest="sweeps_action",
        title="actions",
        metavar="<action>",
        required=True,
    )
    regenerate.add_parser(actions)
    compare.add_parser(actions)
    report.add_parser(actions)
