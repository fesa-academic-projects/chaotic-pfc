#!/usr/bin/env python3
"""Wave 2c, item C: transient chaos and the lowpass Nz=3 extension."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from long_horizon import HORIZONS, _build, _lambda_at, _post_burn_in, _rel_for
from repro import max_difference, reproduce_cell

from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD, load_sweep
from chaotic_pfc.paper.numbers import candidates, load_all

SEED = 42
N_PER_TYPE = 100
N_LOWPASS = 30


def _classify(value: float) -> str:
    if not np.isfinite(value):
        return "NaN"
    if value > LAMBDA_CHAOS_THRESHOLD:
        return "lambda_gt_theta"
    if value > 0.0:
        return "0_lt_lambda_le_theta"
    return "lambda_le_0"


def _sample_chaotic() -> list[tuple[str, int, int, str]]:
    root = Path("data/sweeps")
    rng = np.random.default_rng(SEED)
    out: list[tuple[str, int, int, str]] = []
    for ft in ("lowpass", "highpass", "bandpass", "bandstop"):
        candidates_ft: list[tuple[str, int, int, str]] = []
        for path in sorted(root.rglob("variables_lyapunov.npz")):
            sr = load_sweep(path)
            if sr.filter_type != ft:
                continue
            h = np.asarray(sr.h, dtype=np.float64)
            for i, j in np.argwhere(np.isfinite(h) & (h > LAMBDA_CHAOS_THRESHOLD)):
                candidates_ft.append((path.relative_to(root).as_posix(), int(i), int(j), ft))
        take = min(N_PER_TYPE, len(candidates_ft))
        idx = rng.choice(len(candidates_ft), size=take, replace=False)
        out.extend(candidates_ft[int(k)] for k in idx)
    return out


def _sample_lowpass_nz3() -> list[tuple[str, int, int, str]]:
    old_root = Path("data/sweeps")
    new_root = Path("data/sweeps")
    old = {
        path.relative_to(old_root).as_posix(): load_sweep(path)
        for path in old_root.rglob("variables_lyapunov.npz")
    }
    new = {
        path.relative_to(new_root).as_posix(): load_sweep(path)
        for path in new_root.rglob("variables_lyapunov.npz")
    }
    lowpass_rels = {_rel_for(row["sr"], old) for row in candidates(load_all(new_root), "lowpass")}
    rng = np.random.default_rng(SEED)
    by_rel: dict[str, list[tuple[str, int, int, str]]] = {}
    for rel, sr in new.items():
        if rel not in lowpass_rels:
            continue
        h_old = np.asarray(old[rel].h, dtype=np.float64)
        nz = np.asarray(sr.num_taps_grid, dtype=np.int64) - 1
        wc = np.asarray(sr.cutoff_grid, dtype=np.float64)
        mask = (nz == 3)[:, None] & (h_old > 0.0) & (h_old <= 5e-3) & (wc >= 0.05)
        points = [(rel, int(i), int(j), "lowpass_Nz3") for i, j in np.argwhere(mask)]
        if points:
            by_rel[rel] = points
    chosen = [points[int(rng.integers(len(points)))] for points in by_rel.values()]
    remaining = [p for points in by_rel.values() for p in points if p not in chosen]
    seen = {(p[0], p[2]) for p in chosen}
    remaining = [p for p in remaining if (p[0], p[2]) not in seen]
    need = N_LOWPASS - len(chosen)
    idx = rng.choice(len(remaining), size=need, replace=False)
    chosen.extend(remaining[int(k)] for k in idx)
    return chosen


def run(out_dir: Path) -> None:
    old_root = Path("data/sweeps")
    new_root = Path("data/sweeps")
    rows: list[dict[str, Any]] = []
    max_repro = 0.0

    for sample_name, points in (
        ("chaotic", _sample_chaotic()),
        ("lowpass_nz3", _sample_lowpass_nz3()),
    ):
        for rel, i, j, _sample_filter in points:
            old = load_sweep(old_root / rel)
            new = load_sweep(new_root / rel)
            orders, cutoffs, bank, gains, perturbations = _build(new)
            reproduced = reproduce_cell(new, bank, gains, perturbations, i, j)
            diff, nan_mismatch = max_difference(
                np.asarray([[new.h[i, j]]]), np.asarray([[reproduced]])
            )
            max_repro = max(max_repro, diff)
            if diff != 0.0 or nan_mismatch:
                raise SystemExit(f"reproduction failed for {rel}/{i}/{j}: {diff}")
            xx, c, Ns = _post_burn_in(new, bank, gains, perturbations, i, j)
            values = [_lambda_at(new, xx, c, Ns, horizon) for horizon in HORIZONS]
            if all(v > 0.0 for v in values):
                slope = float(np.polyfit(np.log(HORIZONS), np.log(values), 1)[0])
            else:
                slope = float("nan")
            rows.append(
                {
                    "sample": sample_name,
                    "filter_type": new.filter_type,
                    "path": rel,
                    "i": i,
                    "j": j,
                    "Nz": int(orders[i]) - 1,
                    "omega_c_pi": float(cutoffs[j]),
                    "old_h": float(old.h[i, j]),
                    "new_h": float(new.h[i, j]),
                    "lambda_3e3": values[0],
                    "lambda_3e4": values[1],
                    "lambda_3e5": values[2],
                    "slope": slope,
                    "class_3e5": _classify(values[2]),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "wave2c_transient.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "wave2c_lowpass_nz3.csv").open("w", newline="", encoding="utf-8") as handle:
        subset = [r for r in rows if r["sample"] == "lowpass_nz3"]
        writer = csv.DictWriter(handle, fieldnames=list(subset[0]))
        writer.writeheader()
        writer.writerows(subset)

    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        if row["sample"] != "chaotic":
            continue
        nz_band = "Nz_le_7" if int(row["Nz"]) <= 7 else "Nz_ge_8"
        counts[(row["filter_type"], nz_band, row["class_3e5"])] += 1
    count_rows: list[dict[str, Any]] = []
    for ft in ("lowpass", "highpass", "bandpass", "bandstop"):
        for nz_band in ("Nz_le_7", "Nz_ge_8", "all"):
            for cls in (
                "lambda_gt_theta",
                "0_lt_lambda_le_theta",
                "lambda_le_0",
                "NaN",
            ):
                if nz_band == "all":
                    value = sum(counts[(ft, b, cls)] for b in ("Nz_le_7", "Nz_ge_8"))
                else:
                    value = counts[(ft, nz_band, cls)]
                count_rows.append(
                    {
                        "filter_type": ft,
                        "Nz_band": nz_band,
                        "class": cls,
                        "count": value,
                    }
                )
    with (out_dir / "wave2c_transient_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(count_rows[0]))
        writer.writeheader()
        writer.writerows(count_rows)

    print(f"max |h_repro - h_stored| over sample: {max_repro}")
    print(f"wrote {out_dir / 'wave2c_transient.csv'}")
    print(f"wrote {out_dir / 'wave2c_lowpass_nz3.csv'}")
    print(f"wrote {out_dir / 'wave2c_transient_counts.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    args = parser.parse_args()
    run(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
