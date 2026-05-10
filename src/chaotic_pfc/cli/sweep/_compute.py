from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


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
    p.set_defaults(_run=run_compute)


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
