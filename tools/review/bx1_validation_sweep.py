#!/usr/bin/env python3
"""Single-vector validation value for Section IV-A.

Runs the production single-vector estimator on the unfiltered Henon map
(Ns=1, c0=1) with the same 25 initial conditions and seed used in the
manuscript, and writes the unrounded lambda_1 value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaotic_pfc.analysis.sweep import run_sweep


def run(out_dir: Path) -> dict[str, float | int]:
    """Run the validation and write ``bx1_validation_sweep.json``."""
    result = run_sweep(
        window="hamming",
        filter_type="lowpass",
        num_taps_grid=[1],
        cutoff_grid=[0.5],
        Nitera=500,
        Nmap=3000,
        n_initial=25,
        adaptive=True,
        Nmap_min=500,
        tol=1e-3,
        seed=42,
    )
    payload: dict[str, float | int] = {
        "lambda1": float(result.h[0, 0]),
        "n_fallback": int(result.metadata.get("n_fallback", -1)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bx1_validation_sweep.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    args = parser.parse_args()
    payload = run(Path(args.out_dir))
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
