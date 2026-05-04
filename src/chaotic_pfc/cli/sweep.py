"""Sweep Lyapunov exponents across (order, cutoff) grid.

Nested under ``chaotic-pfc run sweep ...`` with two sub-subcommands:

* ``compute`` — run the actual numerical sweep for one or more
  ``(window, filter)`` combinations and save ``.npz`` checkpoints.
  Originally ``scripts/07_henon_sweep_compute.py``.
* ``plot`` — turn previously saved ``.npz`` checkpoints into the four
  standard classification figures. Originally
  ``scripts/08_henon_sweep_plot.py``.

The two steps are kept separate so plotting iterations (label sizes,
colour maps, format changes) do not require rerunning the multi-hour
sweep.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ._common import pick_backend

# ════════════════════════════════════════════════════════════════════════════
# Parser registration — two levels: sweep → {compute, plot}
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


def _add_compute_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``run sweep compute``."""
    from chaotic_pfc.analysis.sweep import FILTER_TYPES, WINDOWS

    p = subparsers.add_parser(
        "compute",
        help="Run the numerical sweep and save .npz checkpoints.",
        description="Compute λ_max across the (order, cutoff) grid for one or more configs.",
    )
    p.add_argument(
        "--window",
        choices=WINDOWS,
        default="hamming",
        help="FIR window (default: hamming)",
    )
    p.add_argument(
        "--filter",
        choices=FILTER_TYPES,
        default="lowpass",
        dest="filter_type",
        help="Filter pass-zero configuration (default: lowpass)",
    )
    p.add_argument("--all", action="store_true", help="Run every (window, filter) combination")
    p.add_argument(
        "--bandwidth",
        type=float,
        default=0.2,
        help="Band width for bandpass/bandstop filters, as fraction of Nyquist (default: 0.2)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Run a tiny sweep for smoke-testing (~seconds)",
    )
    p.add_argument(
        "--kaiser-beta",
        type=float,
        default=5.0,
        dest="kaiser_beta",
        help="β parameter of the Kaiser window (default: 5.0). Ignored unless --window=kaiser.",
    )
    # Adaptive Lyapunov early-stop. See chaotic_pfc.sweep.run_sweep for the
    # full criterion. Defaults match the calibrated values from
    # docs/adaptive-calibration: Nmap_min=500, tol=1e-3 give a ~3.6× speedup
    # with mean |Δλ| < 0.001 vs. the fixed-Nmap reference.
    p.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "Enable adaptive Lyapunov early-stop. Each grid point exits "
            "the iteration loop once the running λ_max estimate has "
            "stabilised within --tol. Typical speedup: 3-4× on a full "
            "(40 × 100) sweep; max |Δλ| vs non-adaptive reference < 0.01."
        ),
    )
    p.add_argument(
        "--Nmap-min",
        type=int,
        default=500,
        dest="Nmap_min",
        help=(
            "Minimum Lyapunov iterations before --adaptive may fire "
            "(default: 500). Smaller values give larger speedups but "
            "noisier λ near |λ| ≈ 0. Ignored without --adaptive."
        ),
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help=(
            "Adaptive convergence tolerance (default: 1e-3). The loop "
            "exits when |λ_t − λ_{t-1}| < tol for two consecutive "
            "checkpoints. Ignored without --adaptive."
        ),
    )
    p.add_argument(
        "--data-dir",
        default="data/sweeps",
        help="Root directory for .npz output (default: data/sweeps)",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="(accepted for CLI consistency; output is always saved)",
    )
    p.add_argument(
        "--no-display",
        dest="no_display",
        action="store_true",
        help="(accepted for CLI consistency; this command has no UI)",
    )
    p.set_defaults(_run=run_compute)


def _add_plot_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``run sweep plot``."""
    from chaotic_pfc.analysis.sweep import FILTER_TYPES, WINDOWS

    p = subparsers.add_parser(
        "plot",
        help="Plot the four standard classification figures from a saved sweep.",
        description="Generate classification figures from previously saved sweep .npz files.",
    )
    p.add_argument(
        "--window",
        choices=WINDOWS,
        default="hamming",
        help="FIR window (default: hamming)",
    )
    p.add_argument(
        "--filter",
        choices=FILTER_TYPES,
        default="lowpass",
        dest="filter_type",
        help="Filter pass-zero configuration (default: lowpass)",
    )
    p.add_argument("--all", action="store_true", help="Plot every .npz found under --data-dir")
    p.add_argument(
        "--data-dir",
        default="data/sweeps",
        help="Root directory with .npz files (default: data/sweeps)",
    )
    p.add_argument(
        "--figures-dir",
        default="figures/sweeps",
        help="Root directory for output figures (default: figures/sweeps)",
    )
    p.add_argument(
        "--fmt",
        nargs="+",
        default=["png", "svg"],
        choices=("png", "svg", "pdf"),
        help="Output format(s). Multiple values allowed, e.g. '--fmt png svg'.",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="(accepted for CLI consistency; figures are always saved)",
    )
    p.add_argument(
        "--no-display",
        dest="no_display",
        action="store_true",
        help="(accepted for CLI consistency; matplotlib runs headless)",
    )
    p.set_defaults(_run=run_plot)


# ════════════════════════════════════════════════════════════════════════════
# run sweep compute
# ════════════════════════════════════════════════════════════════════════════


def _build_combinations(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Flatten ``--all`` or ``--window``/``--filter`` into a list of pairs."""
    from chaotic_pfc.analysis.sweep import FILTER_TYPES, WINDOWS

    if args.all:
        return [(w, f) for w in WINDOWS for f in FILTER_TYPES]
    return [(args.window, args.filter_type)]


def run_compute(args: argparse.Namespace) -> int:
    """Execute ``run sweep compute``."""
    import numpy as np

    from chaotic_pfc.analysis.sweep import quick_sweep_params, run_sweep, save_sweep

    # ── Argument validation ──────────────────────────────────────────────
    # --adaptive applies inside the Lyapunov loop. --quick already runs a
    # tiny sweep (Nmap=200) where the early-stop has nothing to gain, and
    # combining the two would be confusing about which limit took effect.
    # Reject explicitly rather than silently ignore one of them.
    adaptive = bool(getattr(args, "adaptive", False))
    if adaptive and args.quick:
        print(
            "run sweep compute: --adaptive and --quick are redundant "
            "(quick mode already uses Nmap=200; adaptive early-stop has "
            "no meaningful budget left to save). Pass at most one.",
            file=sys.stderr,
        )
        return 2

    combos = _build_combinations(args)
    data_dir = Path(args.data_dir)

    params: dict[str, float | int]
    if args.quick:
        orders_lp, orders_hp, cutoffs, params = quick_sweep_params()
    else:
        orders_lp = None  # → run_sweep default: np.arange(2, 42)
        orders_hp = None  # → run_sweep default: np.arange(3, 43, 2)
        cutoffs = None  # → module default: 100 points
        params = dict(Nitera=500, Nmap=3000, n_initial=25)
    params.setdefault("bandwidth", getattr(args, "bandwidth", 0.2))

    # Adaptive parameters propagated only when the user opted in. Passing
    # adaptive=False here makes Nmap_min/tol irrelevant in run_sweep
    # (it normalises kernel_Nmap_min == Nmap internally).
    adaptive_kwargs: dict = {"adaptive": adaptive}
    if adaptive:
        adaptive_kwargs["Nmap_min"] = args.Nmap_min
        adaptive_kwargs["tol"] = args.tol
        print(f"     adaptive: Nmap_min={args.Nmap_min}, tol={args.tol}")

    total_start = time.perf_counter()
    warmup_needed = True

    for idx, (window, filter_type) in enumerate(combos, start=1):
        print(f"\n[07] {idx}/{len(combos)}  {window} / {filter_type}")
        t0 = time.perf_counter()
        result = run_sweep(
            window=window,
            filter_type=filter_type,
            orders=orders_hp if filter_type in ("highpass", "bandstop") else orders_lp,
            cutoffs=cutoffs,
            warmup=warmup_needed,
            kaiser_beta=args.kaiser_beta,
            **params,  # type: ignore[arg-type]
            **adaptive_kwargs,
        )
        elapsed = time.perf_counter() - t0
        warmup_needed = False  # only needed on the first call

        valid = int(np.count_nonzero(~np.isnan(result.h)))
        total = result.h.size
        print(f"     Elapsed: {elapsed:6.1f} s ({elapsed / 60:.1f} min)")
        print(f"     Valid points: {valid}/{total} ({100 * valid / total:.1f} %)")
        print(f"     Throughput:   {total / elapsed:.1f} pts/s")
        # In adaptive mode, report how much of the iteration budget the
        # early-stop saved on average — a useful sanity check that the
        # tolerance is not too tight (mean ~ Nmap means no real speedup).
        if adaptive and result.n_iters_used is not None:
            ni = result.n_iters_used[~np.isnan(result.n_iters_used)]
            if ni.size:
                print(
                    f"     Iters used:   mean={ni.mean():.0f} / "
                    f"max={int(ni.max())} (budget {result.metadata['Nmap']})"
                )

        out_path = data_dir / result.display_name / "variables_lyapunov.npz"
        save_sweep(result, out_path)
        print(f"     Saved -> {out_path}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n[07] done in {total_elapsed:.1f} s ({total_elapsed / 60:.1f} min)")
    return 0


# ════════════════════════════════════════════════════════════════════════════
# run sweep plot
# ════════════════════════════════════════════════════════════════════════════


def _discover_sweeps(data_dir: Path) -> list[Path]:
    """Find every ``variables_lyapunov.npz`` under ``data_dir``, sorted."""
    return sorted(data_dir.rglob("variables_lyapunov.npz"))


def _target_dir(figures_dir: Path, npz_path: Path, data_dir: Path) -> Path:
    """Mirror the data-dir subpath into figures-dir.

    ``data/sweeps/Hamming (lowpass)/variables_lyapunov.npz`` →
    ``figures/sweeps/Hamming (lowpass)/``.
    """
    try:
        rel = npz_path.parent.relative_to(data_dir)
    except ValueError:
        rel = Path(npz_path.parent.name)
    return figures_dir / rel


def run_plot(args: argparse.Namespace) -> int:
    """Execute ``run sweep plot``."""
    pick_backend(args.no_display)

    from chaotic_pfc.analysis.sweep import WINDOW_DISPLAY_NAMES, load_sweep
    from chaotic_pfc.analysis.sweep_plotting import plot_all

    data_dir = Path(args.data_dir)
    figures_dir = Path(args.figures_dir)

    if args.all:
        npz_paths = _discover_sweeps(data_dir)
        if not npz_paths:
            print(
                f"[08] No .npz files found under {data_dir}/. "
                "Run 'chaotic-pfc run sweep compute' first."
            )
            return 0
    else:
        pretty = WINDOW_DISPLAY_NAMES.get(args.window, args.window.capitalize())
        subdir = f"{pretty} ({args.filter_type})"
        candidate = data_dir / subdir / "variables_lyapunov.npz"
        if not candidate.exists():
            print(f"[08] Not found: {candidate}")
            print(
                f"     Run: chaotic-pfc run sweep compute "
                f"--window {args.window} --filter {args.filter_type}"
            )
            return 1
        npz_paths = [candidate]

    fmts_str = ", ".join(f".{f}" for f in args.fmt)
    print(f"[08] Plotting {len(npz_paths)} sweep(s) (formats: {fmts_str})")

    for npz_path in npz_paths:
        result = load_sweep(npz_path)
        out_dir = _target_dir(figures_dir, npz_path, data_dir)
        print(f"\n     {result.display_name}")
        print(f"     ← {npz_path}")
        print(f"     → {out_dir}")
        for fmt in args.fmt:
            paths = plot_all(result, out_dir, fmt=fmt)
            for p in paths:
                print(f"       {p.name}")

    print("\n[08] done")
    return 0


# ════════════════════════════════════════════════════════════════════════════
# run sweep beta-sweep
# ════════════════════════════════════════════════════════════════════════════


def _add_beta_sweep_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``run sweep beta-sweep``."""
    from chaotic_pfc.analysis.sweep import FILTER_TYPES

    p = subparsers.add_parser(
        "beta-sweep",
        help="Run one Kaiser sweep per β value and emit one .npz per β.",
        description=(
            "For each β in [beta_min, beta_max] step beta_step, run a full "
            "(order, cutoff) sweep with window=kaiser and save the result. "
            "Defaults: β in [2, 14] step 0.5 (25 values)."
        ),
    )
    p.add_argument(
        "--filter",
        choices=FILTER_TYPES,
        default="lowpass",
        dest="filter_type",
        help="Filter pass-zero configuration (default: lowpass)",
    )
    p.add_argument(
        "--beta-min",
        type=float,
        default=2.0,
        dest="beta_min",
        help="Inclusive lower bound of β range (default: 2.0)",
    )
    p.add_argument(
        "--beta-max",
        type=float,
        default=14.0,
        dest="beta_max",
        help="Inclusive upper bound of β range (default: 14.0)",
    )
    p.add_argument(
        "--beta-step",
        type=float,
        default=0.5,
        dest="beta_step",
        help="Step between consecutive β values (default: 0.5 → 25 values)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Run tiny sweeps for smoke-testing (~seconds per β)",
    )
    p.add_argument(
        "--data-dir",
        default="data/sweeps/beta",
        help="Root directory for .npz output (default: data/sweeps/beta)",
    )
    p.add_argument(
        "--no-display",
        dest="no_display",
        action="store_true",
        help="(accepted for CLI consistency; this command has no UI)",
    )
    p.set_defaults(_run=run_beta_sweep)


def _add_plot_3d_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``run sweep plot-3d``."""
    p = subparsers.add_parser(
        "plot-3d",
        help="Render an interactive 3-D surface stack of Kaiser β-sweeps.",
        description=(
            "Aggregate all per-β .npz files under --data-dir into a single "
            "3-D volume λ_max(N_z, ω_c, β) and save an interactive HTML figure."
        ),
    )
    p.add_argument(
        "--data-dir",
        default="data/sweeps/kaiser",
        help="Root directory with per-β .npz files (default: data/sweeps/kaiser)",
    )
    p.add_argument(
        "--figures-dir",
        default="figures/sweeps",
        help="Root directory for the output HTML (default: figures/sweeps)",
    )
    p.add_argument(
        "--filter-type",
        default="lowpass",
        dest="filter_type",
        help="Filter type subdirectory under --data-dir (default: lowpass)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Generate 3-D plots for every filter type found under --data-dir",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="(accepted for CLI consistency; figure is always saved)",
    )
    p.add_argument(
        "--no-display",
        dest="no_display",
        action="store_true",
        help="(accepted for CLI consistency; plotly runs headless)",
    )
    p.set_defaults(_run=run_plot_3d)


def run_plot_3d(args: argparse.Namespace) -> int:
    """Execute ``run sweep plot-3d``."""
    from chaotic_pfc.analysis.sweep_plotting_3d import (
        aggregate_beta_sweeps,
        plot_3d_beta_volume,
    )

    base_dir = Path(args.data_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        filter_types = [d.name for d in sorted(base_dir.iterdir()) if d.is_dir()]
        if not filter_types:
            print(f"[plot-3d] No filter type directories found under {base_dir}")
            return 1
    else:
        filter_types = [args.filter_type]

    for ft in filter_types:
        data_dir = base_dir / ft
        if not data_dir.is_dir():
            print(f"[plot-3d] Skipping {ft}: directory not found")
            continue

        out_html = figures_dir / f"beta_sweep_3d_{ft}.html"
        print(f"[plot-3d] Aggregating {ft} sweeps from {data_dir}")
        h_volume, betas, orders, cutoffs = aggregate_beta_sweeps(data_dir)
        print(f"[plot-3d] Loaded {len(betas)} β values: {betas[0]:.1f}..{betas[-1]:.1f}")
        print(f"[plot-3d] Volume shape: {h_volume.shape}")

        plot_3d_beta_volume(h_volume, betas, orders, cutoffs, save_path=out_html)
        print(f"[plot-3d] Saved → {out_html}")

    print("[plot-3d] done")
    return 0


def _beta_values(beta_min: float, beta_max: float, beta_step: float) -> list[float]:
    """Build the inclusive β grid, validating the range."""
    import numpy as np

    if beta_step <= 0:
        raise ValueError(f"--beta-step must be > 0, got {beta_step!r}")
    if beta_max < beta_min:
        raise ValueError(f"--beta-max ({beta_max}) must be >= --beta-min ({beta_min})")
    if beta_min < 0:
        raise ValueError(f"--beta-min must be >= 0, got {beta_min!r}")
    n = round((beta_max - beta_min) / beta_step) + 1
    grid = np.linspace(beta_min, beta_max, n)
    return [float(round(b, 6)) for b in grid]


def run_beta_sweep(args: argparse.Namespace) -> int:
    """Execute ``run sweep beta-sweep``."""
    import numpy as np

    from chaotic_pfc.analysis.sweep import quick_sweep_params, run_sweep, save_sweep

    betas = _beta_values(args.beta_min, args.beta_max, args.beta_step)
    data_dir = Path(args.data_dir)

    if args.quick:
        orders_lp, orders_hp, cutoffs, params = quick_sweep_params()
    else:
        orders_lp = None
        orders_hp = None
        cutoffs = None
        params = dict(Nitera=500, Nmap=3000, n_initial=25)
    params.setdefault("bandwidth", getattr(args, "bandwidth", 0.2))
    orders = orders_hp if args.filter_type in ("highpass", "bandstop") else orders_lp

    total_start = time.perf_counter()
    warmup_needed = True

    print(f"[beta-sweep] {len(betas)} β value(s): {betas}")
    for idx, beta in enumerate(betas, start=1):
        print(f"\n[beta-sweep] {idx}/{len(betas)}  β = {beta}")
        t0 = time.perf_counter()
        result = run_sweep(
            window="kaiser",
            filter_type=args.filter_type,
            orders=orders,
            cutoffs=cutoffs,
            warmup=warmup_needed,
            kaiser_beta=beta,
            **params,  # type: ignore[arg-type]
        )
        elapsed = time.perf_counter() - t0
        warmup_needed = False

        valid = int(np.count_nonzero(~np.isnan(result.h)))
        total = result.h.size
        print(f"     Elapsed: {elapsed:6.1f} s")
        print(f"     Valid points: {valid}/{total} ({100 * valid / total:.1f} %)")

        # One sub-directory per β so plotting / loading can iterate easily.
        out_path = data_dir / args.filter_type / f"beta_{beta:.2f}" / "variables_lyapunov.npz"
        save_sweep(result, out_path)
        print(f"     Saved -> {out_path}")

    total_elapsed = time.perf_counter() - total_start
    print(
        f"\n[beta-sweep] done — {len(betas)} sweep(s) in "
        f"{total_elapsed:.1f} s ({total_elapsed / 60:.1f} min)"
    )
    return 0


# Allow `python -m chaotic_pfc.cli.sweep <args>` for local dev convenience.
if __name__ == "__main__":
    from . import main as _main

    sys.exit(_main(["run", "sweep", *sys.argv[1:]]))
