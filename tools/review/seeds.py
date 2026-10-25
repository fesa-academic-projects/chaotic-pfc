#!/usr/bin/env python3
"""Ten-seed repetitions of the reference configurations.

For each filter type the two reference configurations are the best Kaiser
window and the best fixed window by ``Nc``.  The bandstop type also repeats the
Hamming window, because it is the only fixed window that remains close to the
Kaiser curve in that type.  Seeds are 1 to 9 and 42; the summary reports
``mean +/- std`` (``ddof=1``), min, max and the seed-42 value.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from chaotic_pfc._output import progress
from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD, load_sweep, run_sweep
from chaotic_pfc.paper.numbers import FIXED, _seed_cache_identity

SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 42)
TYPES = ("lowpass", "highpass", "bandpass", "bandstop")
FIXED_NAMES = {window: name for window, name, _ in FIXED}


def _records() -> list[tuple[float, int, str, Any]]:
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
                path.relative_to(root).as_posix(),
                sr,
            )
        )
    return out


def _best(
    records: list[tuple[float, int, str, Any]],
    filter_type: str,
    only_kaiser: bool,
    max_beta: float | None = None,
) -> tuple[Any, str, int]:
    subset = []
    for rho, nc, rel, sr in records:
        if sr.filter_type != filter_type:
            continue
        if only_kaiser and sr.window != "kaiser":
            continue
        if not only_kaiser and sr.window == "kaiser":
            continue
        if max_beta is not None and float(sr.metadata.get("kaiser_beta", 999.0)) > max_beta:
            continue
        subset.append((rho, nc, rel, sr))
    rho, nc, rel, sr = max(subset, key=lambda r: (r[0], r[1], r[2]))
    return sr, rel, nc


def _fixed_record(
    records: list[tuple[float, int, str, Any]], filter_type: str, window: str
) -> tuple[Any, str, int]:
    matches = [
        (rho, nc, rel, sr)
        for rho, nc, rel, sr in records
        if sr.filter_type == filter_type and sr.window == window
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {filter_type}/{window} record, found {len(matches)}")
    _rho, nc, rel, sr = matches[0]
    return sr, rel, nc


def _label(sr: Any) -> str:
    if sr.window == "kaiser":
        return f"Kaiser (beta_K={float(sr.metadata['kaiser_beta']):g})"
    return FIXED_NAMES.get(sr.window, sr.window)


def run(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _records()
    best_rows = []
    chosen: list[dict[str, Any]] = []
    for ft in TYPES:
        sr, rel, nc = _best(records, ft, True)
        old_sr, old_rel, old_nc = _best(records, ft, True, max_beta=14.0)
        fixed_sr, fixed_rel, fixed_nc = _best(records, ft, False)
        best_rows.append(
            {
                "filter_type": ft,
                "best_kaiser_path": rel,
                "best_kaiser_beta": float(sr.metadata["kaiser_beta"]),
                "best_kaiser_Nc": nc,
                "low_beta_best_path": old_rel,
                "low_beta_best_beta": float(old_sr.metadata["kaiser_beta"]),
                "low_beta_best_Nc": old_nc,
                "changed": rel != old_rel,
                "best_fixed_path": fixed_rel,
                "best_fixed_window": fixed_sr.window,
                "best_fixed_Nc": fixed_nc,
            }
        )
        chosen.append(
            {
                "filter_type": ft,
                "role": "best_kaiser",
                "label": _label(sr),
                "path": rel,
                "sr": sr,
                "Nc": nc,
            }
        )
        chosen.append(
            {
                "filter_type": ft,
                "role": "best_fixed",
                "label": _label(fixed_sr),
                "path": fixed_rel,
                "sr": fixed_sr,
                "Nc": fixed_nc,
            }
        )
        if ft == "bandstop":
            ham_sr, ham_rel, ham_nc = _fixed_record(records, ft, "hamming")
            if ham_rel != fixed_rel:
                chosen.append(
                    {
                        "filter_type": ft,
                        "role": "bandstop_hamming",
                        "label": _label(ham_sr),
                        "path": ham_rel,
                        "sr": ham_sr,
                        "Nc": ham_nc,
                    }
                )
    with (out_dir / "best_kaiser.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)
    for row in best_rows:
        print("best", row)

    rows = []
    total = len(chosen) * len(SEEDS)
    done = 0
    first = True
    cache = Path(".cache/paper/seeds")
    cache.mkdir(parents=True, exist_ok=True)
    for item in chosen:
        sr = item["sr"]
        cache_stem, digest, source = _seed_cache_identity(Path("data/sweeps"), sr)
        for seed in SEEDS:
            cache_file = cache / f"{cache_stem}.seed{seed}.json"
            if cache_file.exists():
                payload = json.loads(cache_file.read_text())
                nc = int(payload["nc"])
                n_grid = int(np.asarray(sr.h).size)
            else:
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
                nc = int(np.count_nonzero(h > LAMBDA_CHAOS_THRESHOLD))
                n_grid = int(h.size)
                payload = {"nc": nc, "seed": seed, "sha256": digest}
                if source is not None:
                    payload["source"] = str(source)
                cache_file.write_text(json.dumps(payload))
            rows.append(
                {
                    "filter_type": item["filter_type"],
                    "role": item["role"],
                    "label": item["label"],
                    "path": item["path"],
                    "window": sr.window,
                    "kaiser_beta": ""
                    if sr.window != "kaiser"
                    else f"{float(sr.metadata['kaiser_beta']):g}",
                    "seed": seed,
                    "Nc": nc,
                    "N_grid": n_grid,
                }
            )
            done += 1
            progress(done, total, f"{item['filter_type']}/{item['path']}/seed={seed}", None)
    with (out_dir / "seeds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for item in chosen:
        values = [
            r["Nc"]
            for r in rows
            if r["filter_type"] == item["filter_type"] and r["path"] == item["path"]
        ]
        seed42 = next(
            r["Nc"]
            for r in rows
            if r["filter_type"] == item["filter_type"]
            and r["path"] == item["path"]
            and r["seed"] == 42
        )
        summary.append(
            {
                "filter_type": item["filter_type"],
                "role": item["role"],
                "label": item["label"],
                "path": item["path"],
                "window": item["sr"].window,
                "kaiser_beta": ""
                if item["sr"].window != "kaiser"
                else f"{float(item['sr'].metadata['kaiser_beta']):g}",
                "n_seeds": len(values),
                "mean_Nc": float(np.mean(values)),
                "std_Nc": float(np.std(values, ddof=1)),
                "min_Nc": int(np.min(values)),
                "max_Nc": int(np.max(values)),
                "seed42_Nc": seed42,
                "values": " ".join(str(v) for v in values),
            }
        )
    with (out_dir / "seeds_summary.csv").open("w", newline="", encoding="utf-8") as handle:
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
