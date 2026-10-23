#!/usr/bin/env python3
"""Wave 2, item 4: locality of the classification under redrawn initial states."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from repro import max_difference, reproduce_cell

from chaotic_pfc._formulas import fixed_point_positive
from chaotic_pfc.analysis.sweep import (
    LAMBDA_CHAOS_THRESHOLD,
    load_sweep,
    precompute_fir_bank,
)
from chaotic_pfc.analysis.sweep._kernel import (
    _burn_in_n12,
    _burn_in_nN,
    _lyap_online_core_v,
    _lyap_online_n12,
    _lyap_online_nN,
)
from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations

SEED = 42
N_POINTS = 200
N_IC = 25
LOW = -1.5
HIGH = 1.5


def _simulate(
    sr: Any, bank: np.ndarray, gains: np.ndarray, i: int, j: int, rng: np.random.Generator
) -> tuple[float, float, int]:
    orders = np.asarray(sr.num_taps_grid, dtype=np.int64)
    Ns = int(orders[i])
    c = np.asarray(bank[i, j, :Ns], dtype=np.float64)
    gain = float(gains[i, j])
    p = fixed_point_positive(gain, 1.4, 0.3)
    dim = 2 if Ns <= 2 else max(Ns, 4)
    nitera = int(sr.metadata["Nitera"])
    nmap = int(sr.metadata["Nmap"])
    nmap_min = int(sr.metadata.get("Nmap_min") or 500)
    tol = float(sr.metadata.get("tol") or 1e-3)

    states = [rng.uniform(LOW, HIGH, size=dim) for _ in range(N_IC)]

    single: list[float] = []
    for state in states:
        xx = np.empty((2, dim))
        xx[0] = state
        if Ns <= 2:
            _burn_in_n12(xx, nitera, 1.4, 0.3, c)
        else:
            _burn_in_nN(xx, Ns, nitera, 1.4, 0.3, c)
        if not np.isfinite(xx[0, 0]):
            single.append(np.nan)
            continue
        vv = np.empty((2, dim))
        lyap_sum = np.empty(1)
        if Ns <= 2:
            lam, _ = _lyap_online_core_v(xx, vv, lyap_sum, nmap, nmap_min, tol, 2, 0, c, 1.4, 0.3)
        else:
            lam, _ = _lyap_online_core_v(xx, vv, lyap_sum, nmap, nmap_min, tol, Ns, 1, c, 1.4, 0.3)
        single.append(float(lam))
    finite = [v for v in single if np.isfinite(v)]
    mean = float(np.mean(finite)) if finite else float("nan")
    if not np.isfinite(mean):
        return float("nan"), float("nan"), 0

    if abs(mean - LAMBDA_CHAOS_THRESHOLD) < 5e-3:
        full: list[float] = []
        for state in states:
            xx = np.empty((2, dim))
            xx[0] = state
            if Ns <= 2:
                _burn_in_n12(xx, nitera, 1.4, 0.3, c)
            else:
                _burn_in_nN(xx, Ns, nitera, 1.4, 0.3, c)
            if not np.isfinite(xx[0, 0]):
                full.append(np.nan)
                continue
            if Ns <= 2:
                ww = np.empty((2, 2, 2))
                lf = np.empty(2)
                lam, _ = _lyap_online_n12(xx, ww, lf, nmap, nmap_min, tol, 2, c, 1.4, 0.3, 0)
            else:
                ww = np.empty((2, dim, dim))
                lf = np.empty(dim)
                lam, _ = _lyap_online_nN(xx, ww, lf, nmap, nmap_min, tol, Ns, c, 1.4, 0.3, 0)
            full.append(float(lam))
        finite_full = [v for v in full if np.isfinite(v)]
        if finite_full:
            mean = float(np.mean(finite_full))
    return mean, float(p), len(finite)


def run(out_dir: Path) -> None:
    root = Path("data/sweeps")
    rng = np.random.default_rng(SEED)
    candidates: list[tuple[str, Any, np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for path in sorted(root.rglob("variables_lyapunov.npz")):
        sr = load_sweep(path)
        orders = np.asarray(sr.num_taps_grid, dtype=np.int64)
        cutoffs = np.asarray(sr.cutoff_grid, dtype=np.float64)
        bank, gains = precompute_fir_bank(
            orders,
            cutoffs,
            sr.filter_type,
            sr.window,
            kaiser_beta=float(sr.metadata.get("kaiser_beta", 5.0)),
            bandwidth=float(sr.metadata.get("bandwidth", 0.05)),
        )
        h = np.asarray(sr.h, dtype=np.float64)
        perturbations = _precompute_perturbations(
            orders,
            len(cutoffs),
            int(sr.metadata["n_initial"]),
            sr.metadata.get("seed"),
        )
        mask = (np.abs(gains) < 0.1) & np.isfinite(h) & (h <= LAMBDA_CHAOS_THRESHOLD)
        for i, j in np.argwhere(mask):
            candidates.append(
                (
                    path.relative_to(root).as_posix(),
                    sr,
                    bank,
                    gains,
                    perturbations,
                    int(i),
                    int(j),
                )
            )
    idx = rng.choice(len(candidates), size=min(N_POINTS, len(candidates)), replace=False)
    rows: list[dict[str, Any]] = []
    max_repro = 0.0
    for k in idx:
        rel, sr, bank, gains, perturbations, i, j = candidates[int(k)]
        reproduced = reproduce_cell(sr, bank, gains, perturbations, i, j)
        repro_diff, repro_nan = max_difference(
            np.asarray([[sr.h[i, j]]]), np.asarray([[reproduced]])
        )
        max_repro = max(max_repro, repro_diff)
        if repro_diff != 0.0 or repro_nan:
            raise SystemExit(f"reproduction failed for {rel}/{i}/{j}: {repro_diff}")
        mean, p, n_finite = _simulate(sr, bank, gains, i, j, rng)
        if not np.isfinite(mean):
            new_class = "divergent"
        elif mean > LAMBDA_CHAOS_THRESHOLD:
            new_class = "chaotic"
        else:
            new_class = "non-chaotic"
        rows.append(
            {
                "path": rel,
                "filter_type": sr.filter_type,
                "window": sr.window,
                "i": i,
                "j": j,
                "Nz": int(sr.num_taps_grid[i]) - 1,
                "omega_c_pi": float(sr.cutoff_grid[j]),
                "gain": float(gains[i, j]),
                "h_old": float(sr.h[i, j]),
                "p": p,
                "mean_new": mean,
                "n_finite": n_finite,
                "class_old": "non-chaotic",
                "class_new": new_class,
                "changed": new_class != "non-chaotic",
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "wave2_locality.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    changes = Counter[str](row["class_new"] for row in rows if row["changed"])
    print(f"wrote {path} ({len(rows)} points)")
    print(f"max |h_repro - h_stored| over sample: {max_repro}")
    print(f"changed: {sum(changes.values())}, directions: {dict(changes)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    args = parser.parse_args()
    run(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
