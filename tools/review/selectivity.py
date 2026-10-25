#!/usr/bin/env python3
"""Realized selectivity of the FIR designs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import freqz

from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD, load_sweep, precompute_fir_bank
from chaotic_pfc.analysis.sweep._types import WINDOW_SIDELOBE_DB
from chaotic_pfc.paper.numbers import spearman_exact

FIXED = ("boxcar", "bartlett", "hann", "hamming", "blackman", "blackmanharris")
KAISER_BETA = 13.5


def _response(
    num_taps: int, cutoff: float, filter_type: str, window: str
) -> tuple[np.ndarray, np.ndarray]:
    bank, _ = precompute_fir_bank(
        np.array([num_taps], dtype=np.int64),
        np.array([cutoff], dtype=np.float64),
        filter_type,
        window,
        kaiser_beta=KAISER_BETA,
        bandwidth=0.05,
    )
    c = np.asarray(bank[0, 0, :num_taps], dtype=np.float64)
    w, h = freqz(c, worN=8192, fs=2.0)
    return w, np.abs(h)


def _minus6db(w: np.ndarray, mag: np.ndarray) -> float:
    ratio = mag / mag[0]
    target = 10.0 ** (-6.0 / 20.0)
    below = np.flatnonzero(ratio <= target)
    if not below.size:
        return float("nan")
    k = int(below[0])
    if k == 0:
        return float(w[0])
    x0, x1 = np.log10(ratio[k - 1]), np.log10(ratio[k])
    y0, y1 = w[k - 1], w[k]
    if x0 == x1:
        return float(y0)
    frac = (np.log10(target) - x0) / (x1 - x0)
    return float(y0 + frac * (y1 - y0))


def _inband(w: np.ndarray, mag2: np.ndarray, cutoff: float) -> float:
    mask = w <= cutoff
    total = float(np.trapezoid(mag2, w))
    band = float(np.trapezoid(mag2[mask], w[mask]))
    return band / total if total > 0 else float("nan")


def _attenuation(w: np.ndarray, mag: np.ndarray, frequency: float) -> float:
    k = int(np.argmin(np.abs(w - frequency)))
    value = mag[k] / mag[0] if mag[0] > 0 else float("nan")
    return float(-20.0 * np.log10(value)) if value > 0 else float("inf")


def _three_db_width(w: np.ndarray, mag: np.ndarray) -> float:
    peak = float(np.max(mag))
    if peak <= 0:
        return float("nan")
    above = mag >= peak / np.sqrt(2.0)
    center = int(np.argmax(mag))
    left = center
    while left > 0 and above[left - 1]:
        left -= 1
    right = center
    while right < len(mag) - 1 and above[right + 1]:
        right += 1
    return float(w[right] - w[left])


def run(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    medians: dict[str, float] = {}
    for window in (*FIXED, "kaiser"):
        ratios: list[float] = []
        for nz in (2, 4, 6, 10, 20, 40):
            for wc in (0.1, 0.2, 0.4):
                w, mag = _response(nz + 1, wc, "lowpass", window)
                realized = _minus6db(w, mag)
                ratio = realized / wc
                ratios.append(ratio)
                rows.append(
                    {
                        "table": "lowpass",
                        "window": window,
                        "Nz": nz,
                        "nominal_wc_pi": wc,
                        "realized_minus6db_pi": realized,
                        "ratio_realized_over_nominal": ratio,
                        "attenuation_0.6pi_db": _attenuation(w, mag, 0.6),
                        "inband_energy_ratio": _inband(w, mag**2, wc),
                    }
                )
        medians[window] = float(np.nanmedian(ratios))

    # Nc under the new threshold for the six fixed lowpass windows.
    root = Path("data/sweeps")
    nc: dict[str, int] = {}
    for path in root.rglob("variables_lyapunov.npz"):
        sr = load_sweep(path)
        if sr.filter_type == "lowpass" and (
            sr.window in FIXED
            or (
                sr.window == "kaiser"
                and abs(float(sr.metadata.get("kaiser_beta", 0.0)) - KAISER_BETA) < 1e-9
            )
        ):
            h = np.asarray(sr.h, dtype=np.float64)
            nc[sr.window] = int(np.count_nonzero(h > LAMBDA_CHAOS_THRESHOLD))
    x_ratio = np.array([medians[w] for w in FIXED])
    x_sidelobe = np.array([WINDOW_SIDELOBE_DB[w] for w in FIXED])
    y_nc = np.array([nc[w] for w in FIXED], dtype=np.float64)
    rho_ratio, p_ratio = spearman_exact(x_ratio, y_nc)
    rho_db, p_db = spearman_exact(x_sidelobe, y_nc)

    with (out_dir / "selectivity_lowpass.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "selectivity_spearman.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window",
                "median_ratio",
                "sidelobe_db",
                "Nc",
                "rho_ratio",
                "p_ratio",
                "rho_sidelobe",
                "p_sidelobe",
            ]
        )
        for window in (*FIXED, "kaiser"):
            writer.writerow(
                [
                    window,
                    medians[window],
                    WINDOW_SIDELOBE_DB.get(window, ""),
                    nc[window],
                    rho_ratio if window in FIXED else "",
                    p_ratio if window in FIXED else "",
                    rho_db if window in FIXED else "",
                    p_db if window in FIXED else "",
                ]
            )

    band_rows: list[dict[str, Any]] = []
    for filter_type in ("bandpass", "bandstop"):
        for window in (*FIXED, "kaiser"):
            for nz in (10, 20, 40):
                w, mag = _response(nz + 1, 0.5, filter_type, window)
                if filter_type == "bandstop":
                    metric_name = "attenuation_center_db"
                    metric = _attenuation(w, mag, 0.5)
                else:
                    metric_name = "realized_3db_width_pi"
                    metric = _three_db_width(w, mag)
                band_rows.append(
                    {
                        "filter_type": filter_type,
                        "window": window,
                        "Nz": nz,
                        "metric": metric_name,
                        "value": metric,
                    }
                )
    with (out_dir / "selectivity_band.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(band_rows[0]))
        writer.writeheader()
        writer.writerows(band_rows)

    print("lowpass medians:", {w: round(v, 6) for w, v in medians.items()})
    print(f"Spearman ratio vs Nc: rho={rho_ratio:.6f} p={p_ratio:.6f}")
    print(f"Spearman sidelobe vs Nc: rho={rho_db:.6f} p={p_db:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    args = parser.parse_args()
    run(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
