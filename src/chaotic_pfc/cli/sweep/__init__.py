"""Sweep Lyapunov exponents across (order, cutoff) grid.

Nested under ``chaotic-pfc run sweep ...`` with four sub-subcommands:

* ``compute`` — run the actual numerical sweep for one or more
  ``(window, filter)`` combinations and save ``.npz`` checkpoints.
  Originally ``scripts/07_henon_sweep_compute.py``.
* ``plot`` — turn previously saved ``.npz`` checkpoints into the four
  standard classification figures. Originally
  ``scripts/08_henon_sweep_plot.py``.
* ``beta-sweep`` — run Kaiser β-sweeps across multiple β values.
* ``plot-3d`` — render an interactive 3-D surface stack of Kaiser β-sweeps.

The two steps are kept separate so plotting iterations (label sizes,
colour maps, format changes) do not require rerunning the multi-hour
sweep.
"""

from __future__ import annotations

import argparse

from ._beta import _add_beta_sweep_parser
from ._beta import _beta_values as _beta_values
from ._beta import run_beta_sweep as run_beta_sweep
from ._compute import _add_compute_parser
from ._compute import run_compute as run_compute
from ._plot import _add_plot_parser
from ._plot import run_plot as run_plot
from ._plot_3d import _add_plot_3d_parser
from ._plot_3d import run_plot_3d as run_plot_3d

# ════════════════════════════════════════════════════════════════════════════
# Parser registration — two levels: sweep → {compute, plot, beta-sweep, plot-3d}
# ════════════════════════════════════════════════════════════════════════════


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run sweep`` group with its own sub-subcommands."""
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="(Run or plot) a 2-D (filter order, cutoff) Lyapunov sweep.",
        description="Run a 2-D (filter order, cutoff) Lyapunov sweep or plot a saved one.",
    )
    sweep_subparsers = sweep_parser.add_subparsers(
        dest="sweep_action",
        title="actions",
        metavar="<action>",
        required=True,
    )
    _add_compute_parser(sweep_subparsers)
    _add_plot_parser(sweep_subparsers)
    _add_beta_sweep_parser(sweep_subparsers)
    _add_plot_3d_parser(sweep_subparsers)
