"""
cli
===
Unified command-line interface for the chaotic_pfc package.

The parser layout is ``chaotic-pfc run <experiment> [options]``, where
``<experiment>`` is one of the pipeline stages (``attractors``,
``sensitivity``, ``comm-ideal``, ``comm-fir``, ``comm-order-n``,
``lyapunov``, ``sweep compute``, ``sweep plot``, ``all``).

Each experiment lives in its own submodule and exposes two functions:

* ``add_parser(subparsers)`` — register the experiment's own parser and
  flags onto the given ``argparse._SubParsersAction``.
* ``run(args)`` — execute the experiment using the parsed namespace.

The sub-commands under ``sweep`` (``compute`` and ``plot``) use the
same pattern recursively, producing a two-level parser tree
(``run → sweep → {compute, plot}``).

Adding a new experiment is a matter of dropping a new submodule here
and appending its name to :data:`EXPERIMENTS` below.
"""

from __future__ import annotations

import argparse
import sys

from .._version import __version__
from . import (
    analysis,
    attractors,
    comm_fir,
    comm_ideal,
    comm_order_n,
    dcsk,
    lyapunov,
    run_all,
    sensitivity,
    sweep,
)

# Ordered iterable — the help text renders subcommands in this order.
EXPERIMENTS = (
    attractors,
    sensitivity,
    comm_ideal,
    comm_fir,
    comm_order_n,
    lyapunov,
    sweep,
    dcsk,
    analysis,
    run_all,
)


def build_parser() -> argparse.ArgumentParser:
    """Assemble the top-level parser with every experiment registered.

    Returns
    -------
    argparse.ArgumentParser
        A parser whose namespace carries a ``_run`` attribute pointing
        at the callable to invoke for the selected subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="chaotic-pfc",
        description="Chaotic communication system based on the Hénon map.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Top-level groups — only "run" for now, but keeping the tree open
    # for future additions (e.g. "chaotic-pfc info", "chaotic-pfc ls").
    top_subparsers = parser.add_subparsers(
        dest="group",
        title="groups",
        metavar="<group>",
    )

    # ── `chaotic-pfc run <experiment>` ────────────────────────────────────
    run_parser = top_subparsers.add_parser(
        "run",
        help="Run an experiment or the full pipeline.",
        description="Run a single experiment or the full pipeline.",
    )
    run_subparsers = run_parser.add_subparsers(
        dest="experiment",
        title="experiments",
        metavar="<experiment>",
        required=True,
    )
    for module in EXPERIMENTS:
        module.add_parser(run_subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` (or :data:`sys.argv`) and dispatch to the right experiment.

    Parameters
    ----------
    argv
        Optional list of arguments. Defaults to :data:`sys.argv[1:]`.
        Passing an explicit ``argv`` is what makes :mod:`tests.test_cli`
        possible without shelling out to a subprocess.

    Returns
    -------
    int
        Exit code. ``0`` on success, ``2`` when the user requested help
        without picking a subcommand (argparse convention).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "_run", None):
        parser.print_help()
        return 2

    return args._run(args) or 0


if __name__ == "__main__":
    sys.exit(main())
