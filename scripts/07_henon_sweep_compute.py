#!/usr/bin/env python3
"""
07_henon_sweep_compute.py
=========================
Sweep Lyapunov exponents of the filtered Hénon map across the
(filter order, cutoff) grid for one or more (window, filter-type)
configurations.

Results are written to ``data/sweeps/<Window> (<filter>)/variables_lyapunov.npz``
and can be plotted afterwards by ``08_henon_sweep_plot.py`` without
repeating the (hours-long) numerical work.

Examples
--------
Single configuration::

    python scripts/07_henon_sweep_compute.py --window hamming --filter lowpass

Every (window, filter) combination::

    python scripts/07_henon_sweep_compute.py --all

Quick smoke test (tiny grid, ~seconds)::

    python scripts/07_henon_sweep_compute.py --window hamming --quick
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    from chaotic_pfc.sweep import FILTER_TYPES, WINDOWS
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--window", choices=WINDOWS, default="hamming",
                   help="FIR window (default: hamming)")
    p.add_argument("--filter", choices=FILTER_TYPES, default="lowpass",
                   dest="filter_type",
                   help="Filter pass-zero configuration (default: lowpass)")
    p.add_argument("--all", action="store_true",
                   help="Run every (window, filter) combination")
    p.add_argument("--quick", action="store_true",
                   help="Run a tiny sweep for smoke-testing (~seconds)")
    p.add_argument("--data-dir", default="data/sweeps",
                   help="Root directory for .npz output (default: data/sweeps)")
    # Compatibility flags accepted by every other script in the pipeline:
    p.add_argument("--save", action="store_true",
                   help="(accepted for CLI consistency; output is always saved)")
    p.add_argument("--no-display", dest="no_display", action="store_true",
                   help="(accepted for CLI consistency; this script has no UI)")
    return p.parse_args()


def _build_combinations(args: argparse.Namespace) -> list[tuple[str, str]]:
    from chaotic_pfc.sweep import FILTER_TYPES, WINDOWS
    if args.all:
        return [(w, f) for w in WINDOWS for f in FILTER_TYPES]
    return [(args.window, args.filter_type)]


def main() -> None:
    args = parse_args()

    # Heavy imports delayed until after --help has been handled.
    import numpy as np

    from chaotic_pfc.sweep import run_sweep, save_sweep

    combos   = _build_combinations(args)
    data_dir = Path(args.data_dir)

    if args.quick:
        orders  = np.arange(2, 8)
        cutoffs = np.linspace(0.1, 0.9, 10)
        params  = dict(Nitera=50, Nmap=200, n_initial=3)
    else:
        orders  = None  # → module default: 2..41
        cutoffs = None  # → module default: 100 points
        params  = dict(Nitera=500, Nmap=3000, n_initial=25)

    total_start   = time.perf_counter()
    warmup_needed = True

    for idx, (window, filter_type) in enumerate(combos, start=1):
        print(f"\n[07] {idx}/{len(combos)}  {window} / {filter_type}")
        t0 = time.perf_counter()
        result = run_sweep(
            window=window, filter_type=filter_type,
            orders=orders, cutoffs=cutoffs,
            warmup=warmup_needed,
            **params,
        )
        elapsed = time.perf_counter() - t0
        warmup_needed = False  # only needed on the first call

        valid = int(np.count_nonzero(~np.isnan(result.h)))
        total = result.h.size
        print(f"     Elapsed: {elapsed:6.1f} s "
              f"({elapsed / 60:.1f} min)")
        print(f"     Valid points: {valid}/{total} "
              f"({100 * valid / total:.1f} %)")
        print(f"     Throughput:   {total / elapsed:.1f} pts/s")

        out_path = data_dir / result.display_name / "variables_lyapunov.npz"
        save_sweep(result, out_path)
        print(f"     Saved -> {out_path}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n[07] done in {total_elapsed:.1f} s "
          f"({total_elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
