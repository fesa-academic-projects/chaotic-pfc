#!/usr/bin/env python3
"""Wave 3, step 2: best Kaiser per type and ten-seed repetitions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from chaotic_pfc._output import progress
from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD, load_sweep, run_sweep

SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 42)
TYPES = ("lowpass", "highpass", "bandpass", "bandstop")


def _records() -> list[tuple[float, int, float, Any, str]]:
    root = Path("data/sweeps")
    out = []
    for path in sorted(root.rglob("variables_lyapunov.npz")):
        sr = load_sweep(path)
        h = np.asarray(sr.h, dtype=np.float64)
        nc = int(np.count_nonzero(h > LAMBDA_CHAOS_THRESHOLD))
        out.append(
            (
                nc / h.size,
                nc,
                -float(sr.metadata.get("kaiser_beta", 5.0)),
                sr,
                path.relative_to(root).as_posix(),
            )
        )
    return out


def _best(
    records: list[tuple[float, int, float, Any, str]],
    filter_type: str,
    only_kaiser: bool,
    max_beta: float | None = None,
) -> tuple[Any, str, int]:
    subset = []
    for rho, nc, neg_beta, sr, rel in records:
        if sr.filter_type != filter_type:
            continue
        if only_kaiser and sr.window != "kaiser":
            continue
        if max_beta is not None and float(sr.metadata.get("kaiser_beta", 999.0)) > max_beta:
            continue
        subset.append((rho, nc, neg_beta, sr, rel))
    rho, nc, _, sr, rel = max(subset, key=lambda r: (r[0], r[1], r[2]))
    return sr, rel, nc


def run(out_dir: Path) -> None:
    records = _records()
    best_rows = []
    chosen: list[tuple[str, Any, str, int]] = []
    for ft in TYPES:
        sr, rel, nc = _best(records, ft, True)
        old_sr, old_rel, old_nc = _best(records, ft, True, max_beta=14.0)
        changed = rel != old_rel
        best_rows.append(
            {
                "filter_type": ft,
                "best_kaiser_path": rel,
                "best_kaiser_beta": float(sr.metadata["kaiser_beta"]),
                "best_kaiser_Nc": nc,
                "gate1_best_path": old_rel,
                "gate1_best_beta": float(old_sr.metadata["kaiser_beta"]),
                "gate1_best_Nc": old_nc,
                "changed": changed,
            }
        )
        _second_rho, second_nc, _, second_sr, second_rel = max(
            (r for r in records if r[3].filter_type == ft and r[4] != rel),
            key=lambda r: (r[0], r[1], r[2]),
        )
        chosen.append((ft, sr, rel, nc))
        chosen.append((ft, second_sr, second_rel, second_nc))
    with (out_dir / "wave3_best_kaiser.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)
    for row in best_rows:
        print("best", row)

    rows = []
    total = len(chosen) * len(SEEDS)
    done = 0
    first = True
    for ft, sr, rel, _nc in chosen:
        for seed in SEEDS:
            result = run_sweep(
                window=sr.window,
                filter_type=sr.filter_type,
                num_taps_grid=sr.num_taps_grid,
                cutoff_grid=sr.cutoff_grid,
                Nitera=int(sr.metadata["Nitera"]),
                Nmap=int(sr.metadata["Nmap"]),
                n_initial=int(sr.metadata["n_initial"]),
                alpha=float(sr.metadata["alpha"]),
                beta=float(sr.metadata["beta"]),
                seed=seed,
                warmup=first,
                kaiser_beta=float(sr.metadata.get("kaiser_beta", 5.0)),
                bandwidth=float(sr.metadata.get("bandwidth", 0.05)),
                adaptive=bool(sr.metadata["adaptive"]),
                Nmap_min=int(sr.metadata.get("Nmap_min") or 500),
                tol=float(sr.metadata.get("tol") or 1e-3),
                fallback_mode=1,
            )
            first = False
            h = np.asarray(result.h, dtype=np.float64)
            rows.append(
                {
                    "filter_type": ft,
                    "path": rel,
                    "window": sr.window,
                    "kaiser_beta": ""
                    if sr.window != "kaiser"
                    else f"{float(sr.metadata['kaiser_beta']):g}",
                    "seed": seed,
                    "Nc": int(np.count_nonzero(h > LAMBDA_CHAOS_THRESHOLD)),
                    "N_grid": int(h.size),
                }
            )
            done += 1
            progress(done, total, f"{ft}/{rel}/seed={seed}", None)
    with (out_dir / "wave3_seeds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for ft, rel in sorted({(r["filter_type"], r["path"]) for r in rows}):
        values = [r["Nc"] for r in rows if r["filter_type"] == ft and r["path"] == rel]
        summary.append(
            {
                "filter_type": ft,
                "path": rel,
                "n_seeds": len(values),
                "mean_Nc": float(np.mean(values)),
                "std_Nc": float(np.std(values, ddof=1)),
                "min_Nc": int(np.min(values)),
                "max_Nc": int(np.max(values)),
                "values": " ".join(str(v) for v in values),
            }
        )
    with (out_dir / "wave3_seeds_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print("seed", row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    args = parser.parse_args()
    run(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
