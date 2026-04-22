"""
__main__.py
===========
Dispatcher for ``python -m chaotic_pfc``.

Lets the package be invoked directly, either to run a single experiment
by name or every experiment at once. This is a thin wrapper that
``subprocess``-runs the numbered scripts — the scripts remain the
canonical entry points for local development, while this module
provides a short, installable alias (``chaotic-pfc``, declared in
``pyproject.toml`` under ``[project.scripts]``).

Usage
-----
Run every experiment::

    python -m chaotic_pfc --save --no-display

Run a single experiment by its short name::

    python -m chaotic_pfc lyapunov --save
    python -m chaotic_pfc sweep_compute
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = {
    "attractors": "scripts/01_henon_attractors.py",
    "sensitivity": "scripts/02_sensitivity.py",
    "ideal": "scripts/03_comm_ideal_channel.py",
    "fir": "scripts/04_comm_fir_channel.py",
    "order_n": "scripts/05_comm_order_n.py",
    "lyapunov": "scripts/06_lyapunov.py",
    "sweep_compute": "scripts/07_henon_sweep_compute.py",
    "sweep_plot": "scripts/08_henon_sweep_plot.py",
}


def main() -> None:
    """Parse CLI args and run the selected experiment(s) as subprocesses.

    Exits with the first non-zero return code it encounters, which lets
    CI detect failures in any of the numbered scripts.
    """
    parser = argparse.ArgumentParser(prog="chaotic_pfc")
    parser.add_argument("experiment", nargs="?", choices=[*list(SCRIPTS), "all"], default="all")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-display", dest="no_display", action="store_true")
    args = parser.parse_args()

    targets = list(SCRIPTS.values()) if args.experiment == "all" else [SCRIPTS[args.experiment]]
    extra = []
    if args.save:
        extra.append("--save")
    if args.no_display:
        extra.append("--no-display")

    root = Path(__file__).resolve().parent.parent.parent
    for script in targets:
        result = subprocess.run([sys.executable, str(root / script), *extra])
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
