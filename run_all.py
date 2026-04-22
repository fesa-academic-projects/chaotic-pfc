#!/usr/bin/env python3
"""
run_all.py — Run every experiment in sequence.

Usage:
    python run_all.py                # display all figures
    python run_all.py --save         # save all figures (.svg)
    python run_all.py --no-display   # headless / CI mode (implies --save)
"""
import argparse, subprocess, sys
from pathlib import Path

SCRIPTS = [
    "scripts/01_henon_attractors.py",
    "scripts/02_sensitivity.py",
    "scripts/03_comm_ideal_channel.py",
    "scripts/04_comm_fir_channel.py",
    "scripts/05_comm_order_n.py",
    "scripts/06_lyapunov.py",
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-display", dest="no_display", action="store_true")
    args = p.parse_args()

    extra = []
    if args.no_display:
        extra += ["--no-display", "--save"]
    elif args.save:
        extra += ["--save"]

    root = Path(__file__).parent
    for script in SCRIPTS:
        path = root / script
        print(f"\n{'='*60}\nRunning: {script}\n{'='*60}")
        result = subprocess.run([sys.executable, str(path)] + extra)
        if result.returncode != 0:
            print(f"[ERROR] {script} exited with code {result.returncode}")
            sys.exit(result.returncode)
    print("\nAll experiments completed successfully.")

if __name__ == "__main__": main()
