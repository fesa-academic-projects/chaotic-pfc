"""Run every experiment in sequence.

Nested under ``chaotic-pfc run all``. Previously the top-level
``run_all.py`` script, which shelled out to each numbered script via
``subprocess``. The in-process version here is faster (no fork
overhead, the Numba cache is shared across experiments) and makes the
control flow much easier to follow.

Usage examples
--------------
Display every figure interactively::

    chaotic-pfc run all

Save all figures to disk (keeps display for non-sweep experiments)::

    chaotic-pfc run all --save

Headless mode (implies ``--save``)::

    chaotic-pfc run all --no-display

Skip the long Lyapunov sweep (~5 h); assumes the ``.npz`` checkpoints
are already present, so plotting still works::

    chaotic-pfc run all --no-display --skip-sweep

Run the sweep in quick mode (tiny grid, seconds) — useful for CI::

    chaotic-pfc run all --no-display --quick-sweep
"""

from __future__ import annotations

import argparse

from . import attractors, comm_fir, comm_ideal, comm_order_n, lyapunov, sensitivity
from . import sweep as sweep_mod

# Experiments run before the sweep, in order. Each module exposes
# a run(args) function whose signature is documented in its own module.
COMM_EXPERIMENTS = (
    ("01", attractors.run),
    ("02", sensitivity.run),
    ("03", comm_ideal.run),
    ("04", comm_fir.run),
    ("05", comm_order_n.run),
    ("06", lyapunov.run),
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run all`` subcommand."""
    p = subparsers.add_parser(
        "all",
        help="Run every experiment in sequence (full pipeline).",
        description="Run every experiment in sequence (full pipeline).",
    )
    p.add_argument("--save", action="store_true", help="Save figures produced by each experiment.")
    p.add_argument(
        "--no-display",
        dest="no_display",
        action="store_true",
        help="Run headless (implies --save).",
    )
    p.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the sweep compute step; plot from existing .npz only.",
    )
    p.add_argument(
        "--quick-sweep",
        action="store_true",
        help="Run the sweep compute step in quick mode (seconds instead of hours).",
    )
    p.set_defaults(_run=run)


def _banner(title: str) -> None:
    """Print a section banner that matches the legacy run_all.py output."""
    print(f"\n{'=' * 60}\nRunning: {title}\n{'=' * 60}")


def _common_args(no_display: bool, save: bool) -> dict[str, bool]:
    """Build the ``--save`` / ``--no-display`` defaults shared by every step."""
    if no_display:
        return {"no_display": True, "save": True}
    return {"no_display": False, "save": bool(save)}


def run(args: argparse.Namespace) -> int:
    """Execute ``run all``."""
    if args.skip_sweep and args.quick_sweep:
        print("run all: --skip-sweep and --quick-sweep are mutually exclusive")
        return 2

    shared = _common_args(args.no_display, args.save)

    # ── 1) Communication + Lyapunov (01–06) ───────────────────────────────
    for tag, experiment_run in COMM_EXPERIMENTS:
        _banner(tag)
        # Each experiment gets a namespace with every flag it might look up.
        # Extra fields (e.g. "taps") are harmless because argparse resolved
        # them to defaults earlier when the individual subcommand was built.
        step_args = argparse.Namespace(
            **shared,
            # Defaults matching individual-subcommand behaviour.
            # Callers that want finer control should invoke each subcommand
            # directly rather than "run all".
            steps=50_000,
            epsilon=1e-4,
            N=None,
            mu=None,
            period=None,
            cutoff=None,
            taps=None,
            # Lyapunov defaults — must match those declared in cli/lyapunov.py
            Nitera=2000,
            Ndiscard=1000,
            pole_radius=0.975,
            w0=0.0,
            n_ci=20,
            perturbation=0.1,
            data_dir="data/lyapunov",
        )
        # Fill defaults from DEFAULT_CONFIG when experiment-specific flags
        # are left as None (so each run() sees the same values as it would
        # in a direct invocation).
        _fill_config_defaults(step_args)
        experiment_run(step_args)

    # ── 2) Sweep compute (07) ─────────────────────────────────────────────
    if args.skip_sweep:
        _banner("07  (skipped)")
    else:
        _banner("07")
        compute_args = argparse.Namespace(
            **shared,
            window="hamming",
            filter_type="lowpass",
            all=False,
            quick=bool(args.quick_sweep),
            data_dir="data/sweeps",
        )
        sweep_mod.run_compute(compute_args)

    # ── 3) Sweep plot (08) ────────────────────────────────────────────────
    _banner("08")
    plot_args = argparse.Namespace(
        **shared,
        window="hamming",
        filter_type="lowpass",
        all=False,
        data_dir="data/sweeps",
        figures_dir="figures/sweeps",
        fmt=["png", "svg"],
    )
    sweep_mod.run_plot(plot_args)

    print("\nAll experiments completed successfully.")
    return 0


def _fill_config_defaults(ns: argparse.Namespace) -> None:
    """Replace ``None`` placeholders with values from ``DEFAULT_CONFIG``."""
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg

    if ns.N is None:
        ns.N = cfg.comm.N
    if ns.mu is None:
        ns.mu = cfg.comm.mu
    if ns.period is None:
        ns.period = cfg.comm.message_period
    if ns.cutoff is None:
        ns.cutoff = cfg.channel.cutoff
    if ns.taps is None:
        ns.taps = cfg.channel.num_taps
