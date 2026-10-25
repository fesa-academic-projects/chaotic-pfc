#!/usr/bin/env python3
"""Long-horizon test with 400 chaotic points per filter type.

The points are sampled from all configurations of each filter type through
``transient._sample_chaotic``.  The script writes the per-point CSV and the
Wilson intervals used in Section IV-F.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import transient
from long_horizon import _build, _lambda_at, _post_burn_in

from chaotic_pfc.analysis.sweep import load_sweep

N_PER_TYPE = 400
SEED = 42
HORIZON = 300_000


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the Wilson score interval for ``k`` successes in ``n`` trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def run(
    out_dir: Path,
    n_per_type: int = N_PER_TYPE,
    horizon: int = HORIZON,
    seed: int = SEED,
) -> None:
    """Run the long-horizon test and write its CSV and summary."""
    if not Path("data/sweeps").is_dir():
        raise SystemExit("run from the repository root (data/sweeps not found)")
    out_dir.mkdir(parents=True, exist_ok=True)

    transient.N_PER_TYPE = n_per_type
    transient.SEED = seed
    points = transient._sample_chaotic()
    print("sample", len(points), flush=True)

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for index, (rel, i, j, filter_type) in enumerate(points, start=1):
        sweep = load_sweep(Path("data/sweeps") / rel)
        orders, cutoffs, bank, gains, perturbations = _build(sweep)
        xx, c, ns = _post_burn_in(sweep, bank, gains, perturbations, i, j, 0)
        value = _lambda_at(sweep, xx, c, ns, horizon)
        rows.append(
            {
                "filter_type": filter_type,
                "path": rel,
                "i": i,
                "j": j,
                "Nz": int(orders[i]) - 1,
                "omega_c_pi": float(cutoffs[j]),
                "lambda_3e5": float(value),
                "class": transient._classify(value),
            }
        )
        if index % 50 == 0:
            print(index, filter_type, "elapsed", round(time.perf_counter() - start, 1), flush=True)

    elapsed = time.perf_counter() - start
    csv_path = out_dir / "f3_long_horizon_400.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, Any]] = []
    for filter_type in ("lowpass", "highpass", "bandpass", "bandstop"):
        subset = [row for row in rows if row["filter_type"] == filter_type]
        n = len(subset)
        keep = sum(row["class"] == "lambda_gt_theta" for row in subset)
        below = sum(row["class"] == "0_lt_lambda_le_theta" for row in subset)
        non = sum(row["class"] in ("lambda_le_0", "NaN") for row in subset)
        non_low, non_high = _wilson(non, n)
        keep_low, keep_high = _wilson(keep, n)
        summary.append(
            {
                "filter_type": filter_type,
                "n": n,
                "keep": keep,
                "keep_pct": 100.0 * keep / n,
                "below": below,
                "below_pct": 100.0 * below / n,
                "nonchaotic_or_divergent": non,
                "non_pct": 100.0 * non / n,
                "non_ci_low": 100.0 * non_low,
                "non_ci_high": 100.0 * non_high,
                "keep_ci_low": 100.0 * keep_low,
                "keep_ci_high": 100.0 * keep_high,
            }
        )
        print(filter_type, summary[-1], flush=True)

    unique_paths = len({row["path"] for row in rows})
    payload = {
        "elapsed_s": elapsed,
        "seed": seed,
        "horizon": horizon,
        "n_per_type": n_per_type,
        "rows": len(rows),
        "unique_paths": unique_paths,
        "summary": summary,
    }
    summary_path = out_dir / "f3_long_horizon_400_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("rows", len(rows), "unique_paths", unique_paths, flush=True)
    print("elapsed_s", elapsed, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    parser.add_argument("--n-per-type", type=int, default=N_PER_TYPE)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(Path(args.out_dir), args.n_per_type, args.horizon, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
