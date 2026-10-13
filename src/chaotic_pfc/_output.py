"""Terminal-output helpers shared by the CLI and the paper/sweeps tooling.

They live outside both ``chaotic_pfc.cli`` and ``chaotic_pfc.paper``/``sweeps``
so the tooling can use them without importing the CLI (which imports the
tooling) and creating a cycle.
"""

from __future__ import annotations

import argparse
import os
import sys


def color_enabled(args: argparse.Namespace | None) -> bool:
    """Return whether terminal colour should be used.

    An explicit ``--color always``/``--color never`` wins; ``auto`` (default)
    enables colour only on a TTY and honours a non-empty ``NO_COLOR`` (the
    ``NO_COLOR`` convention, which ``auto`` respects).
    """
    mode = getattr(args, "color", "auto")
    if mode == "always":
        return True
    if mode == "never":
        return False
    return not os.environ.get("NO_COLOR") and sys.stderr.isatty()


def progress_enabled(args: argparse.Namespace | None) -> bool:
    """Return whether a progress counter should be written to stderr.

    Progress is shown only when stderr is a terminal and neither ``--quiet``
    nor ``--no-progress`` was passed.
    """
    if getattr(args, "quiet", False) or getattr(args, "no_progress", False):
        return False
    return sys.stderr.isatty()


def progress(index: int, total: int, label: str, args: argparse.Namespace | None) -> None:
    """Write ``index/total label`` to stderr when progress is enabled."""
    if not progress_enabled(args):
        return
    print(f"{index}/{total} {label}", file=sys.stderr, flush=True)


def info(message: str, args: argparse.Namespace | None = None) -> None:
    """Print an informational message unless ``--quiet`` was passed."""
    if getattr(args, "quiet", False):
        return
    print(message)


def debug_enabled(args: argparse.Namespace | None) -> bool:
    """Return whether the full traceback should be shown on error."""
    return bool(getattr(args, "debug", False)) or os.environ.get("CHAOTIC_PFC_DEBUG") == "1"


__all__ = [
    "color_enabled",
    "debug_enabled",
    "info",
    "progress",
    "progress_enabled",
]
