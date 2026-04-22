#!/usr/bin/env python3
"""
run_all.py — Run every experiment in sequence.

Usage
-----
Display every figure interactively (local development)::

    python run_all.py

Save all figures to disk (keeps display for non-sweep scripts)::

    python run_all.py --save

Headless / CI mode (implies --save, closes figures immediately)::

    python run_all.py --no-display

Skip the long Lyapunov sweep (~5 h); assumes the ``data/sweeps/*.npz``
checkpoints are already present on disk so that script 08 can still
produce figures::

    python run_all.py --no-display --skip-sweep

Run the sweep in quick mode (tiny grid, ~seconds) — useful for CI to
exercise script 07 end-to-end without burning hours of compute::

    python run_all.py --no-display --quick-sweep
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

COMM_SCRIPTS = [
    "scripts/01_henon_attractors.py",
    "scripts/02_sensitivity.py",
    "scripts/03_comm_ideal_channel.py",
    "scripts/04_comm_fir_channel.py",
    "scripts/05_comm_order_n.py",
    "scripts/06_lyapunov.py",
]
SWEEP_COMPUTE_SCRIPT = "scripts/07_henon_sweep_compute.py"
SWEEP_PLOT_SCRIPT = "scripts/08_henon_sweep_plot.py"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--save", action="store_true", help="Save figures produced by each script")
    p.add_argument(
        "--no-display", dest="no_display", action="store_true", help="Run headless (implies --save)"
    )
    p.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip script 07 (long sweep); run 08 from existing .npz",
    )
    p.add_argument(
        "--quick-sweep",
        action="store_true",
        help="Run script 07 with a tiny grid (~seconds) instead of the full ~5 h sweep",
    )
    return p.parse_args()


def _run(script: str, extra: list[str]) -> None:
    root = Path(__file__).resolve().parent
    path = root / script
    print(f"\n{'=' * 60}\nRunning: {script}\n{'=' * 60}")
    result = subprocess.run([sys.executable, str(path), *extra])
    if result.returncode != 0:
        print(f"[ERROR] {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    args = _parse_args()

    if args.skip_sweep and args.quick_sweep:
        sys.exit("run_all.py: --skip-sweep and --quick-sweep are mutually exclusive")

    common_extra: list[str] = []
    if args.no_display:
        common_extra += ["--no-display", "--save"]
    elif args.save:
        common_extra += ["--save"]

    # 1) Communication pipeline (01–06) ────────────────────────────────────
    for script in COMM_SCRIPTS:
        _run(script, common_extra)

    # 2) Sweep compute (07) ────────────────────────────────────────────────
    if args.skip_sweep:
        print(f"\n{'=' * 60}\nSkipping: {SWEEP_COMPUTE_SCRIPT}\n{'=' * 60}")
    else:
        sweep_extra = list(common_extra)
        if args.quick_sweep:
            sweep_extra.append("--quick")
        _run(SWEEP_COMPUTE_SCRIPT, sweep_extra)

    # 3) Sweep plot (08) ────────────────────────────────────────────────────
    _run(SWEEP_PLOT_SCRIPT, common_extra)

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
