from __future__ import annotations

import argparse
import time
from pathlib import Path

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
    p.set_defaults(_run=run_beta_sweep)


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
