#!/usr/bin/env python3
"""Long-horizon decay of the single-vector estimator.

Sixty grid points are sampled from the canonical and earlier checkpoints:
30 with old h in (0, 5e-3] and 30 controls with new h > 2e-2. The estimator
runs without early stopping at N = 3e3, 3e4 and 3e5, starting from the first
initial condition of the original ensemble.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from repro import max_difference, reproduce_cell

from chaotic_pfc._formulas import fixed_point_positive
from chaotic_pfc.analysis.sweep import load_sweep, precompute_fir_bank
from chaotic_pfc.analysis.sweep._kernel import _burn_in_n12, _burn_in_nN, _lyap_online_core_v
from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations
from chaotic_pfc.paper.numbers import candidates, load_all

HORIZONS = (3_000, 30_000, 300_000)
N_BAND = 30
N_CONTROL = 30
SEED = 42


def _build(sr: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    perturbations = _precompute_perturbations(
        orders,
        len(cutoffs),
        int(sr.metadata["n_initial"]),
        sr.metadata.get("seed"),
    )
    return orders, cutoffs, bank, gains, perturbations  # type: ignore[return-value]


def _post_burn_in(
    sr: Any,
    bank: np.ndarray,
    gains: np.ndarray,
    perturbations: np.ndarray,
    i: int,
    j: int,
    ic: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    orders = np.asarray(sr.num_taps_grid, dtype=np.int64)
    Ns = int(orders[i])
    c = np.asarray(bank[i, j, :Ns], dtype=np.float64)
    gain = float(gains[i, j])
    p1 = fixed_point_positive(gain, 1.4, 0.3)
    noise = perturbations[i, j, ic, :Ns]
    dim = 2 if Ns <= 2 else max(Ns, 4)
    xx = np.empty((2, dim))
    if Ns <= 2:
        xx[0, 0] = 0.1 * noise[0] + p1
        xx[0, 1] = 0.0 if Ns == 1 else 0.1 * noise[1] + p1
        _burn_in_n12(xx, int(sr.metadata["Nitera"]), 1.4, 0.3, c)
    else:
        xx[0, 0] = 0.1 * noise[0] + p1
        xx[0, 1] = 0.1 * noise[1] + p1
        xx[0, 2] = 0.1 * noise[2] + gain * p1
        for k in range(3, Ns):
            xx[0, k] = 0.1 * noise[k] + p1
        _burn_in_nN(xx, Ns, int(sr.metadata["Nitera"]), 1.4, 0.3, c)
    if not np.isfinite(xx[0, 0]):
        raise RuntimeError("selected cell diverged during burn-in")
    return xx, c, Ns


def _lambda_at(sr: Any, xx: np.ndarray, c: np.ndarray, Ns: int, horizon: int) -> float:
    dim = xx.shape[1]
    state = xx.copy()
    vv = np.empty((2, dim))
    lyap_sum = np.empty(1)
    if Ns <= 2:
        lam, _ = _lyap_online_core_v(state, vv, lyap_sum, horizon, horizon, 0.0, 2, 0, c, 1.4, 0.3)
    else:
        lam, _ = _lyap_online_core_v(state, vv, lyap_sum, horizon, horizon, 0.0, Ns, 1, c, 1.4, 0.3)
    return float(lam)


def _points() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    rng = np.random.default_rng(SEED)

    all_band: list[tuple[str, int, int, str]] = []
    lowpass_nz3: list[tuple[str, int, int, str]] = []
    lowpass_configs = set()
    for row in candidates(load_all(new_root), "lowpass"):
        sr_candidate = row["sr"]
        candidate_beta = (
            None if sr_candidate.window != "kaiser" else float(sr_candidate.metadata["kaiser_beta"])
        )
        lowpass_configs.add((sr_candidate.window, candidate_beta))
    for rel, sr in new.items():
        h_old = np.asarray(old[rel].h, dtype=np.float64)
        nz = np.asarray(sr.num_taps_grid, dtype=np.int64) - 1
        mask = (h_old > 0.0) & (h_old <= 5e-3)
        for i, j in np.argwhere(mask):
            all_band.append((rel, int(i), int(j), "random_band"))
        beta = None if sr.window != "kaiser" else float(sr.metadata["kaiser_beta"])
        if sr.filter_type == "lowpass" and (sr.window, beta) in lowpass_configs:
            nz3 = mask & (nz == 3)[:, None]
            for i, j in np.argwhere(nz3):
                lowpass_nz3.append((rel, int(i), int(j), "lowpass_Nz3"))
    if not all_band:
        raise RuntimeError("no points in (0, 5e-3]")

    # One deterministic lowpass Nz=3 point per configuration, then fill the
    # remaining slots at random from the full band sample.
    chosen: list[tuple[str, int, int, str]] = []
    seen: set[str] = set()
    for rel, i, j, kind in lowpass_nz3:
        if rel not in seen:
            chosen.append((rel, i, j, kind))
            seen.add(rel)
    remaining = [p for p in all_band if p not in chosen]
    need = N_BAND - len(chosen)
    if need > 0:
        idx = rng.choice(len(remaining), size=need, replace=False)
        chosen.extend(remaining[int(k)] for k in idx)

    controls: list[tuple[str, int, int, str]] = []
    for rel, sr in new.items():
        h_new = np.asarray(sr.h, dtype=np.float64)
        for i, j in np.argwhere((~np.isnan(h_new)) & (h_new > 2e-2)):
            controls.append((rel, int(i), int(j), "control"))
    idx = rng.choice(len(controls), size=N_CONTROL, replace=False)
    controls = [controls[int(k)] for k in idx]
    return chosen, controls


def _rel_for(sr: Any, old: dict[str, Any]) -> str:
    for rel, candidate in old.items():
        if candidate.window != sr.window or candidate.filter_type != sr.filter_type:
            continue
        if sr.window == "kaiser" and float(candidate.metadata.get("kaiser_beta", -1.0)) != float(
            sr.metadata.get("kaiser_beta", -2.0)
        ):
            continue
        return rel
    raise KeyError(f"no canonical checkpoint for {sr.filter_type}/{sr.window}")


def run(out_dir: Path) -> None:
    old_root = Path("data/sweeps")
    new_root = Path("data/sweeps")
    band, controls = _points()
    rows: list[dict[str, Any]] = []
    max_repro = 0.0
    for kind, points in (("band", band), ("control", controls)):
        for rel, i, j, label in points:
            old = load_sweep(old_root / rel)
            new = load_sweep(new_root / rel)
            orders, cutoffs, bank, gains, perturbations = _build(new)
            reproduced = reproduce_cell(new, bank, gains, perturbations, i, j)
            repro_diff, repro_nan = max_difference(
                np.asarray([[new.h[i, j]]]), np.asarray([[reproduced]])
            )
            max_repro = max(max_repro, repro_diff)
            if repro_diff != 0.0 or repro_nan:
                raise SystemExit(f"reproduction failed for {rel}/{i}/{j}: {repro_diff}")
            xx, c, Ns = _post_burn_in(new, bank, gains, perturbations, i, j)
            lams = [_lambda_at(new, xx, c, Ns, h) for h in HORIZONS]
            if all(v > 0.0 for v in lams):
                slope = float(np.polyfit(np.log(HORIZONS), np.log(lams), 1)[0])
            else:
                slope = float("nan")
            rows.append(
                {
                    "case": kind,
                    "label": label,
                    "path": rel,
                    "i": i,
                    "j": j,
                    "Nz": int(orders[i]) - 1,
                    "omega_c_pi": float(cutoffs[j]),
                    "old_h": float(old.h[i, j]),
                    "new_h": float(new.h[i, j]),
                    "v_mean": float(new.v_mean[i, j]),
                    "lambda_3e3": lams[0],
                    "lambda_3e4": lams[1],
                    "lambda_3e5": lams[2],
                    "loglog_slope": slope,
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "long_horizon.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
    print(f"max |h_repro - h_stored| over sample: {max_repro}")
    print(
        "band candidates:",
        len([r for r in rows if r["case"] == "band"]),
        "controls:",
        len([r for r in rows if r["case"] == "control"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    args = parser.parse_args()
    run(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
