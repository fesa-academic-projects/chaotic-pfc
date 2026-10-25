#!/usr/bin/env python3
"""Remaining raw numbers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import freqz

from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD, load_sweep, precompute_fir_bank
from chaotic_pfc.analysis.sweep_plotting import chaotic_density_norm_table
from chaotic_pfc.paper.numbers import FIXED, fixed_of, spearman_exact

ROOT = Path("data/sweeps")
TYPES = ("lowpass", "highpass", "bandpass", "bandstop")


def _records() -> list[Any]:
    return [load_sweep(p) for p in sorted(ROOT.rglob("variables_lyapunov.npz"))]


def _counts(sr: Any) -> int:
    h = np.asarray(sr.h, dtype=np.float64)
    return int(np.count_nonzero(h > LAMBDA_CHAOS_THRESHOLD))


def _nfallback(recs: list[Any]) -> dict[str, Any]:
    total = sum(int(r.metadata.get("n_fallback", 0)) for r in recs)
    cells = sum(np.asarray(r.h).size for r in recs)
    return {"n_fallback": total, "cells": cells, "pct": 100.0 * total / cells}


def _horizons(recs: list[Any]) -> dict[str, Any]:
    medians = []
    floor = float("inf")
    at_cap = 0
    for r in recs:
        h = np.asarray(r.h, dtype=np.float64)
        it = np.asarray(r.n_iters_used, dtype=np.float64)
        finite = np.isfinite(h)
        vals = it[finite]
        if vals.size:
            medians.append(float(np.median(vals)))
            floor = min(floor, float(np.min(vals)))
        at_cap += int(np.count_nonzero(finite & (it >= 3000.0)))
    return {
        "median_min": float(np.min(medians)),
        "median_p10": float(np.percentile(medians, 10)),
        "median_median": float(np.median(medians)),
        "median_p90": float(np.percentile(medians, 90)),
        "median_max": float(np.max(medians)),
        "floor": floor,
        "at_cap": at_cap,
    }


def _fig7_low_bands() -> list[dict[str, Any]]:
    nz, cutoff, percent, _ = chaotic_density_norm_table(ROOT)
    bands = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.0))
    out = []
    for lo, hi in bands:
        cmask = (cutoff > lo) & (cutoff <= hi) if lo > 0 else (cutoff <= hi)
        low = percent[(nz <= 7)[:, None] & cmask[None, :]]
        out.append(
            {
                "omega_c_low": lo,
                "omega_c_high": hi,
                "mean_Nz_le_7": float(np.nanmean(low)) if low.size else float("nan"),
            }
        )
    return out


def _psd_ratio(out_dir: Path) -> list[dict[str, Any]]:
    with open(out_dir / "psd_fraction.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for ft in ("lowpass", "highpass", "bandpass", "bandstop"):
        for band_name, band_mask in (
            ("all", lambda nz: True),
            ("Nz_le_7", lambda nz: nz <= 7),
            ("Nz_ge_8", lambda nz: nz >= 8),
        ):
            vals = []
            for r in rows:
                if r["filter_type"] != ft or not band_mask(int(r["Nz"])):
                    continue
                f = float(r["fraction"])
                b = float(r["fraction_baseline"])
                if np.isfinite(f) and np.isfinite(b) and b > 0:
                    vals.append(f / b)
            arr = np.asarray(vals, dtype=np.float64)
            out.append(
                {
                    "filter_type": ft,
                    "Nz_band": band_name,
                    "n": int(arr.size),
                    "median": float(np.median(arr)) if arr.size else float("nan"),
                    "p10": float(np.percentile(arr, 10)) if arr.size else float("nan"),
                    "p90": float(np.percentile(arr, 90)) if arr.size else float("nan"),
                }
            )
    return out


def _highpass_edge(w: np.ndarray, mag: np.ndarray) -> float:
    peak = float(np.max(mag))
    above = mag >= peak / np.sqrt(2.0)
    centre = len(mag) - 1
    left = centre
    while left > 0 and above[left - 1]:
        left -= 1
    if left == 0:
        return float(w[0])
    x0, x1 = np.log10(mag[left - 1]), np.log10(mag[left])
    y0, y1 = w[left - 1], w[left]
    if x0 == x1:
        return float(y0)
    frac = (np.log10(peak / np.sqrt(2.0)) - x0) / (x1 - x0)
    return float(y0 + frac * (y1 - y0))


def _three_db_width(w: np.ndarray, mag: np.ndarray) -> float:
    peak = float(np.max(mag))
    above = mag >= peak / np.sqrt(2.0)
    centre = int(np.argmax(mag))
    left = centre
    while left > 0 and above[left - 1]:
        left -= 1
    right = centre
    while right < len(mag) - 1 and above[right + 1]:
        right += 1
    return float(w[right] - w[left])


def _response(
    num_taps: int, cutoff: float, filter_type: str, window: str
) -> tuple[np.ndarray, np.ndarray]:
    bank, _ = precompute_fir_bank(
        np.array([num_taps], dtype=np.int64),
        np.array([cutoff], dtype=np.float64),
        filter_type,
        window,
        kaiser_beta=13.5,
        bandwidth=0.05,
    )
    c = bank[0, 0, :num_taps]
    w, h = freqz(c, worN=8192, fs=2.0)
    return w, np.abs(h)


def _spearman_widths(recs: list[Any]) -> list[dict[str, Any]]:
    out = []
    for ft in ("highpass", "bandpass"):
        ratios = []
        for nz in (2, 4, 6, 10, 20, 40):
            for wc in (0.1, 0.2, 0.4):
                w, mag = _response(nz + 1, wc, ft, "hamming")
                if ft == "highpass":
                    edge = _highpass_edge(w, mag)
                    ratios.append((np.pi - edge) / (np.pi - wc))
                else:
                    ratios.append(_three_db_width(w, mag) / 0.05)
        fixed = fixed_of(recs, ft)
        nc = np.array([_counts(fixed[window]) for window, _, _ in FIXED], dtype=np.float64)
        per_window = []
        for window, _, _ in FIXED:
            vals = []
            for nz in (2, 4, 6, 10, 20, 40):
                for wc in (0.1, 0.2, 0.4):
                    w, mag = _response(nz + 1, wc, ft, window)
                    if ft == "highpass":
                        edge = _highpass_edge(w, mag)
                        vals.append((np.pi - edge) / (np.pi - wc))
                    else:
                        vals.append(_three_db_width(w, mag) / 0.05)
            per_window.append(float(np.median(vals)))
        rho, p = spearman_exact(np.asarray(per_window), nc)
        out.append(
            {
                "filter": ft,
                "median_ratio_per_window": per_window,
                "rho": rho,
                "p_exact": p,
            }
        )
    return out


def _psd_meta(out_dir: Path) -> list[dict[str, Any]]:
    with open(out_dir / "psd_meta.csv", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(out_dir: Path) -> None:
    recs = _records()
    payload = {
        "n_fallback": _nfallback(recs),
        "horizons": _horizons(recs),
        "fig7_low_bands": _fig7_low_bands(),
        "psd_ratio": _psd_ratio(out_dir),
        "spearman_widths": _spearman_widths(recs),
        "psd_meta": _psd_meta(out_dir),
    }
    (out_dir / "revision_numbers.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    for key, value in payload.items():
        print(key, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    args = parser.parse_args()
    run(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
