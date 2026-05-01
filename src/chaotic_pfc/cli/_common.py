"""Shared helpers for every CLI experiment module."""

from __future__ import annotations

import argparse
import os
import sys


def pick_backend(no_display: bool) -> bool:
    """Switch Matplotlib to a headless backend if needed.

    Parameters
    ----------
    no_display
        ``True`` when the user passed ``--no-display``, or when the
        caller otherwise knows no interactive display is available.

    Returns
    -------
    bool
        ``True`` if the backend was forced to ``Agg``. Callers use this
        to decide whether to call ``plt.show()`` at the end.

    Notes
    -----
    The check is a disjunction: either the user opted into headless
    mode explicitly, or we detected Linux without an X display. The
    selection must happen *before* the first ``pyplot`` import,
    otherwise Matplotlib locks to the interactive backend and won't let
    us switch.
    """
    headless = no_display or (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))
    if headless:
        import matplotlib

        matplotlib.use("Agg")
    return headless


def add_save_display_flags(parser: argparse.ArgumentParser) -> None:
    """Register ``--save`` and ``--no-display`` flags on *parser*."""
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-display", dest="no_display", action="store_true")
