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

Run the sweep with adaptive Lyapunov early-stop (≈3-4× speedup, mean
\\|Δλ\\| < 0.001 vs. the fixed-Nmap reference)::

    chaotic-pfc run all --no-display --adaptive

Tighten the adaptive tolerance for closer-to-reference accuracy::

    chaotic-pfc run all --no-display --adaptive --tol 1e-4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from chaotic_pfc.analysis.sweep import (
    FILTER_TYPES,
    WINDOWS,
    quick_sweep_params,
    run_sweep,
    save_sweep,
)
from chaotic_pfc.cli.sweep import _beta_values
from chaotic_pfc.comms.dcsk import DCSK_DEFAULT_WC
from chaotic_pfc.config import DEFAULT_CONFIG as cfg

from . import attractors, comm_fir, comm_ideal, comm_order_n, dcsk, lyapunov, sensitivity
from . import sweep as sweep_mod
from ._common import add_lang_flag

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
    # Adaptive Lyapunov early-stop. Defaults to off to preserve
    # bit-exactness with previous releases. See cli/sweep.py for the
    # full justification of the default Nmap_min / tol values.
    p.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "Enable adaptive Lyapunov early-stop in the sweep step. "
            "Typical speedup: 3-4× on the full sweep. Mean \\|Δλ\\| vs the "
            "fixed-Nmap reference is < 0.001. Mutually exclusive with "
            "--quick-sweep."
        ),
    )
    p.add_argument(
        "--Nmap-min",
        type=int,
        default=500,
        dest="Nmap_min",
        help=(
            "Minimum Lyapunov iterations before --adaptive may fire "
            "(default: 500). Ignored without --adaptive."
        ),
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help=("Adaptive convergence tolerance (default: 1e-3). Ignored without --adaptive."),
    )
    add_lang_flag(p)
    p.set_defaults(_run=run)


def _banner(title: str) -> None:
    """Print a section banner that matches the legacy run_all.py output."""
    print(f"\n{'=' * 60}\nRunning: {title}\n{'=' * 60}")


def _common_args(no_display: bool, save: bool) -> dict[str, bool]:
    """Build the ``--save`` / ``--no-display`` defaults shared by every step."""
    if no_display:
        return {"no_display": True, "save": True}
    return {"no_display": False, "save": bool(save)}


def _run_all_sweeps(
    shared: dict,
    quick: bool,
    adaptive: bool,
    Nmap_min: int,
    tol: float,
) -> None:
    """Run every (window, filter) combination.

    Non-Kaiser windows produce one ``.npz`` per filter type (12 total).
    Kaiser expands into a β-sweep (β ∈ [2, 14] step 0.5) per filter type
    (50 total).

    When ``adaptive`` is True, every sweep enables the Lyapunov
    early-stop with the supplied ``Nmap_min`` and ``tol`` parameters.
    """
    if quick:
        orders_lp, orders_hp, cutoffs, params = quick_sweep_params()
    else:
        orders_lp = None  # → run_sweep default (np.arange(2, 42))
        orders_hp = None  # → run_sweep default (np.arange(3, 43, 2))
        cutoffs = None
        params = dict(Nitera=500, Nmap=3000, n_initial=25, bandwidth=0.2)

    # Adaptive parameters propagated only when the user opted in.
    adaptive_kwargs: dict = {"adaptive": adaptive}
    if adaptive:
        adaptive_kwargs["Nmap_min"] = Nmap_min
        adaptive_kwargs["tol"] = tol
        print(f"[07] adaptive enabled: Nmap_min={Nmap_min}, tol={tol}")

    warmup_needed = True
    total_start = time.perf_counter()

    # Build the full sweep list: (window, filter_type, beta_or_None)
    betas = _beta_values(2.0, 14.0, 0.5)
    kaiser_dir = Path("data/sweeps/kaiser")
    sweeps: list[tuple[str, str, float | None]] = []
    for window in WINDOWS:
        if window == "kaiser":
            for ft in FILTER_TYPES:
                for b in betas:
                    sweeps.append((window, ft, b))
        else:
            for ft in FILTER_TYPES:
                sweeps.append((window, ft, None))

    total = len(sweeps)
    for idx, (window, filter_type, beta) in enumerate(sweeps, start=1):
        label = (
            f"Kaiser β={beta:.1f} / {filter_type}"
            if beta is not None
            else f"{window} / {filter_type}"
        )
        print(f"\n[07] {idx}/{total}  {label}")
        t0 = time.perf_counter()
        result = run_sweep(
            window=window,
            filter_type=filter_type,
            orders=orders_hp if filter_type in ("highpass", "bandstop") else orders_lp,
            cutoffs=cutoffs,
            warmup=warmup_needed,
            kaiser_beta=beta if beta is not None else 5.0,
            **params,  # type: ignore[arg-type]
            **adaptive_kwargs,
        )
        elapsed = time.perf_counter() - t0
        warmup_needed = False

        if beta is not None:
            out_path = kaiser_dir / filter_type / f"beta_{beta:.2f}" / "variables_lyapunov.npz"
        else:
            out_path = Path("data/sweeps") / result.display_name / "variables_lyapunov.npz"

        valid = int(np.count_nonzero(~np.isnan(result.h)))
        n_total = result.h.size
        print(f"     Elapsed: {elapsed:6.1f} s ({elapsed / 60:.1f} min)")
        print(f"     Valid points: {valid}/{n_total} ({100 * valid / n_total:.1f} %)")
        print(f"     Throughput:   {n_total / elapsed:.1f} pts/s")
        if adaptive and result.n_iters_used is not None:
            ni = result.n_iters_used[~np.isnan(result.n_iters_used)]
            if ni.size:
                print(
                    f"     Iters used:   mean={ni.mean():.0f} / "
                    f"max={int(ni.max())} (budget {result.metadata['Nmap']})"
                )
        save_sweep(result, out_path)
        print(f"     Saved -> {out_path}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n[07] done — {total} sweep(s) in {total_elapsed:.0f} s ({total_elapsed / 60:.0f} min)")


def run(args: argparse.Namespace) -> int:
    """Execute ``run all``."""
    if args.skip_sweep and args.quick_sweep:
        print("run all: --skip-sweep and --quick-sweep are mutually exclusive")
        return 2

    # --adaptive applies inside the Lyapunov loop, so it makes no sense
    # without an actual sweep run; reject early with a clear message
    # rather than silently ignoring the flag.
    adaptive = bool(getattr(args, "adaptive", False))
    if adaptive and args.skip_sweep:
        print("run all: --adaptive has no effect with --skip-sweep")
        return 2
    if adaptive and args.quick_sweep:
        print(
            "run all: --adaptive and --quick-sweep are redundant "
            "(quick mode already uses Nmap=200; adaptive early-stop has "
            "no meaningful budget left to save). Pass at most one."
        )
        return 2

    shared = _common_args(args.no_display, args.save)

    # ── 1) Communication + Lyapunov (01–06) ───────────────────────────────
    for tag, experiment_run in COMM_EXPERIMENTS:
        _banner(tag)
        # Each experiment gets a namespace with every flag it might look up.
        # Extra fields (e.g. "taps") are harmless because argparse resolved
        # them to defaults earlier when the individual subcommand was built.
        step_args = _build_step_args(shared)
        step_args.lang = getattr(args, "lang", "pt")
        # Sensitivity keeps fewer steps to avoid unreadable point overlay
        if tag == "02":
            step_args.steps = 50
        experiment_run(step_args)

    # ── 1b) DCSK comparison (06b) ───────────────────────────────────────
    _banner("06b")
    dcsk.run(
        argparse.Namespace(
            no_display=shared["no_display"],
            save=shared["save"],
            N=600,  # demo-friendly; the CLI default matches
            beta=64,
            n_taps=5,
            wc=DCSK_DEFAULT_WC,
            mu=cfg.comm.mu,
            snr_min=-6,
            snr_max=28,
            snr_step=2,
            lang=getattr(args, "lang", "pt"),
        )
    )

    # ── 2) Sweep compute (07) ─────────────────────────────────────────────
    if args.skip_sweep:
        _banner("07  (skipped)")
    else:
        _banner("07")
        _run_all_sweeps(
            shared=shared,
            quick=bool(args.quick_sweep),
            adaptive=adaptive,
            Nmap_min=args.Nmap_min,
            tol=args.tol,
        )

    # ── 3) Sweep plot (08) ────────────────────────────────────────────────
    _banner("08")
    sweep_mod.run_plot(
        argparse.Namespace(
            no_display=shared["no_display"],
            all=True,
            data_dir="data/sweeps",
            figures_dir="figures/sweeps",
            fmt=["png", "svg"],
            lang=getattr(args, "lang", "pt"),
        )
    )

    # ── 4) Plot 3-D (09) ──────────────────────────────────────────────────
    if args.skip_sweep or args.quick_sweep:
        _banner("09  (skipped — use normal run for 3-D plot)")
    else:
        _banner("09")
        sweep_mod.run_plot_3d(
            argparse.Namespace(
                all=True,
                data_dir="data/sweeps/kaiser",
                figures_dir="figures/sweeps",
            )
        )

    print("\nAll experiments completed successfully.")
    return 0


def _build_step_args(shared: dict) -> argparse.Namespace:
    """Build an argparse.Namespace for each experiment step."""
    ns = cfg.to_namespace()
    for k, v in shared.items():
        setattr(ns, k, v)
    return ns
