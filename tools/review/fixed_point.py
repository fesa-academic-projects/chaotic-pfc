#!/usr/bin/env python3
"""Wave 2, item 2: fixed-point convergence of bounded non-chaotic points."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from repro import max_difference, reproduce_grid

from chaotic_pfc._compat import njit
from chaotic_pfc._formulas import fixed_point_positive
from chaotic_pfc.analysis.sweep import (
    LAMBDA_CHAOS_THRESHOLD,
    load_sweep,
    precompute_fir_bank,
)
from chaotic_pfc.analysis.sweep._kernel import (
    _burn_in_n12,
    _burn_in_nN,
    _henon_n12_inplace,
    _henon_nN_inplace,
)
from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations

BINS = ((0.0, 1e-6), (1e-6, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, float("inf")))


@njit(cache=True)
def _distance_n12(
    xx: Any, c: Any, p: float, alpha: float, beta: float, nitera: int, nmap: int
) -> float:
    _burn_in_n12(xx, nitera, alpha, beta, c)
    if not np.isfinite(xx[0, 0]):
        return np.nan
    tick = 0
    for _ in range(nmap):
        _henon_n12_inplace(xx[tick], xx[1 - tick], alpha, beta, c)
        tick = 1 - tick
    if tick == 1:
        xx[0, 0] = xx[1, 0]
        xx[0, 1] = xx[1, 1]
    return max(abs(xx[0, 0] - p), abs(xx[0, 1] - p))


@njit(cache=True)
def _distance_nN(
    xx: Any,
    c: Any,
    p: float,
    gain: float,
    alpha: float,
    beta: float,
    nitera: int,
    nmap: int,
    Ns: int,
) -> float:
    _burn_in_nN(xx, Ns, nitera, alpha, beta, c)
    if not np.isfinite(xx[0, 0]):
        return np.nan
    tick = 0
    for _ in range(nmap):
        _henon_nN_inplace(xx[tick], xx[1 - tick], alpha, beta, c)
        tick = 1 - tick
    if tick == 1:
        for k in range(Ns):
            xx[0, k] = xx[1, k]
    dist = 0.0
    for k in range(Ns):
        target = gain * p if k == 2 else p
        delta = abs(xx[0, k] - target)
        if delta > dist:
            dist = delta
    return dist


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
        orders, len(cutoffs), int(sr.metadata["n_initial"]), sr.metadata.get("seed")
    )
    return orders, cutoffs, bank, gains, perturbations


def _scan_cells(sr: Any, data: tuple[Any, ...]) -> list[dict[str, Any]]:
    orders, cutoffs, bank, gains, perturbations = data
    h = np.asarray(sr.h, dtype=np.float64)
    nitera = int(sr.metadata["Nitera"])
    nmap = int(sr.metadata["Nmap"])
    rows: list[dict[str, Any]] = []
    beta_label = "" if sr.window != "kaiser" else f"{float(sr.metadata['kaiser_beta']):g}"
    for i, Ns_raw in enumerate(orders):
        Ns = int(Ns_raw)
        dim = 2 if Ns <= 2 else max(Ns, 4)
        xx = np.empty((2, dim))
        for j in np.flatnonzero(np.isfinite(h[i]) & (h[i] <= LAMBDA_CHAOS_THRESHOLD)):
            c = np.asarray(bank[i, j, :Ns], dtype=np.float64)
            gain = float(gains[i, j])
            p = fixed_point_positive(gain, 1.4, 0.3)
            noise = perturbations[i, j, 0, :Ns]
            xx.fill(0.0)
            if Ns <= 2:
                xx[0, 0] = 0.1 * noise[0] + p
                xx[0, 1] = 0.0 if Ns == 1 else 0.1 * noise[1] + p
                dist = _distance_n12(xx, c, p, 1.4, 0.3, nitera, nmap)
            else:
                xx[0, 0] = 0.1 * noise[0] + p
                xx[0, 1] = 0.1 * noise[1] + p
                xx[0, 2] = 0.1 * noise[2] + gain * p
                for k in range(3, Ns):
                    xx[0, k] = 0.1 * noise[k] + p
                dist = _distance_nN(xx, c, p, gain, 1.4, 0.3, nitera, nmap, Ns)
            rows.append(
                {
                    "path": "",
                    "filter_type": sr.filter_type,
                    "window": sr.window,
                    "kaiser_beta": beta_label,
                    "i": int(i),
                    "j": int(j),
                    "Nz": Ns - 1,
                    "omega_c_pi": float(cutoffs[j]),
                    "gain": gain,
                    "h": float(h[i, j]),
                    "distance": float(dist),
                    "converged": bool(np.isfinite(dist) and dist < 1e-8),
                }
            )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ft in ("lowpass", "highpass", "bandpass", "bandstop"):
        subset = [r for r in rows if r["filter_type"] == ft]
        low = [r for r in subset if abs(float(r["gain"])) < 0.1]
        for label, group in (("all", subset), ("absG_lt_0.1", low)):
            n = len(group)
            conv = sum(bool(r["converged"]) for r in group)
            finite = sum(np.isfinite(float(r["distance"])) for r in group)
            out.append(
                {
                    "filter_type": ft,
                    "subset": label,
                    "n": n,
                    "finite": finite,
                    "converged": conv,
                    "share_all_pct": 100.0 * conv / n if n else float("nan"),
                    "share_finite_pct": 100.0 * conv / finite if finite else float("nan"),
                }
            )
    return out


def _bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ft in ("lowpass", "highpass", "bandpass", "bandstop"):
        subset = [r for r in rows if r["filter_type"] == ft]
        for lo, hi in BINS:
            group = [r for r in subset if lo <= abs(float(r["gain"])) < hi]
            n = len(group)
            finite = sum(np.isfinite(float(r["distance"])) for r in group)
            conv = sum(bool(r["converged"]) for r in group)
            out.append(
                {
                    "filter_type": ft,
                    "gain_low": lo,
                    "gain_high": hi,
                    "n": n,
                    "finite": finite,
                    "converged": conv,
                    "share_all_pct": 100.0 * conv / n if n else float("nan"),
                    "share_finite_pct": 100.0 * conv / finite if finite else float("nan"),
                }
            )
    return out


def _nonconverged(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ft in ("lowpass", "highpass", "bandpass", "bandstop"):
        group = [r for r in rows if r["filter_type"] == ft and not bool(r["converged"])]
        dist = np.asarray(
            [float(r["distance"]) for r in group if np.isfinite(float(r["distance"]))],
            dtype=np.float64,
        )
        out.append(
            {
                "filter_type": ft,
                "n_nonconverged": len(group),
                "n_finite_distance": int(dist.size),
                "min": float(np.min(dist)) if dist.size else float("nan"),
                "p10": float(np.percentile(dist, 10)) if dist.size else float("nan"),
                "median": float(np.median(dist)) if dist.size else float("nan"),
                "p90": float(np.percentile(dist, 90)) if dist.size else float("nan"),
                "max": float(np.max(dist)) if dist.size else float("nan"),
            }
        )
    return out


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(out_dir: Path) -> None:
    root = Path("data/sweeps")
    all_rows: list[dict[str, Any]] = []
    max_repro = 0.0
    for path in sorted(root.rglob("variables_lyapunov.npz")):
        sr = load_sweep(path)
        rel = path.relative_to(root).as_posix()
        data = _build(sr)
        reproduced = reproduce_grid(sr, data[2], data[3], data[4])
        diff, nan_mismatch = max_difference(np.asarray(sr.h), reproduced)
        max_repro = max(max_repro, diff)
        if diff != 0.0 or nan_mismatch:
            raise SystemExit(
                f"reproduction failed for {rel}: max|dh|={diff}, NaN mismatches={nan_mismatch}"
            )
        scanned = _scan_cells(sr, data)
        for row in scanned:
            row["path"] = rel
            all_rows.append(row)
        print(f"scanned {rel}: {len(scanned)} non-chaotic cells", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "wave2_fixed_point_cells.csv", all_rows)
    _write(out_dir / "wave2_fixed_point.csv", _aggregate(all_rows))
    _write(out_dir / "wave2_fixed_point_bins.csv", _bins(all_rows))
    _write(out_dir / "wave2_fixed_point_nonconverged.csv", _nonconverged(all_rows))
    print(f"max |h_repro - h_stored| over 124 grids: {max_repro}")
    print(f"wrote {out_dir / 'wave2_fixed_point.csv'}")
    print(f"wrote {out_dir / 'wave2_fixed_point_bins.csv'}")
    print(f"wrote {out_dir / 'wave2_fixed_point_nonconverged.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    args = parser.parse_args()
    run(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
