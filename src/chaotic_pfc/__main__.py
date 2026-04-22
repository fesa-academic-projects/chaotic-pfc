"""
Allow: python -m chaotic_pfc [experiment] [--save]
"""
import sys
import subprocess
import argparse
from pathlib import Path

SCRIPTS = {
    "attractors":    "scripts/01_henon_attractors.py",
    "sensitivity":   "scripts/02_sensitivity.py",
    "ideal":         "scripts/03_comm_ideal_channel.py",
    "fir":           "scripts/04_comm_fir_channel.py",
    "order_n":       "scripts/05_comm_order_n.py",
    "lyapunov":      "scripts/06_lyapunov.py",
    "sweep_compute": "scripts/07_henon_sweep_compute.py",
    "sweep_plot":    "scripts/08_henon_sweep_plot.py",
}

def main() -> None:
    parser = argparse.ArgumentParser(prog="chaotic_pfc")
    parser.add_argument("experiment", nargs="?",
                        choices=list(SCRIPTS) + ["all"], default="all")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-display", dest="no_display", action="store_true")
    args = parser.parse_args()

    targets = list(SCRIPTS.values()) if args.experiment == "all" \
              else [SCRIPTS[args.experiment]]
    extra = []
    if args.save:
        extra.append("--save")
    if args.no_display:
        extra.append("--no-display")

    root = Path(__file__).resolve().parent.parent.parent
    for script in targets:
        result = subprocess.run([sys.executable, str(root / script)] + extra)
        if result.returncode != 0:
            sys.exit(result.returncode)

if __name__ == "__main__":
    main()
