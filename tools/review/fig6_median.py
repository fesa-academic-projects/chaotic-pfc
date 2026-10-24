#!/usr/bin/env python3
"""Figure 6 with points selected near the median chaotic lambda_1.

For each filter type, the best configuration is fixed (best Kaiser for
lowpass, highpass and bandpass; rectangular for bandstop).  Among its chaotic
points with omega_c/pi in [0.125, 0.875] whose 25 initial conditions remain
finite for 3e5 iterations, the selected point is the one whose stored lambda_1
is closest to the median lambda_1 of the whole chaotic region of that
configuration.

For highpass and bandstop, the script also measures the fraction of the Welch
PSD power in the five largest peak bins, at the selected point and over 30
random chaotic points of the same configuration.  Peak bins must be at least
five bins apart so that one main lobe is not counted as several lines.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import phase1
import psd
from scipy.signal import find_peaks, freqz

from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD
from chaotic_pfc.paper.numbers import best_kaiser, fixed_of, load_all
from chaotic_pfc.plotting.figures import setup_rc

TYPES = ("lowpass", "highpass", "bandpass", "bandstop")
BAND = (0.125, 0.875)
PEAK_DISTANCE = 5
RANDOM_POINTS = 30
RANDOM_SEED = 42
SVG_SALT = "chaotic-pfc-psd"


def _best_record(recs: list[Any], filter_type: str) -> Any:
    if filter_type == "bandstop":
        return fixed_of(recs, filter_type)["boxcar"]
    return best_kaiser(recs, filter_type)


def _signal(
    sr: Any,
    bank: np.ndarray,
    gains: np.ndarray,
    perturbations: np.ndarray,
    i: int,
    j: int,
) -> np.ndarray | None:
    for ic in range(int(sr.metadata["n_initial"])):
        prepared = psd._state(sr, bank, gains, perturbations, i, j, ic)
        if prepared is None:
            continue
        xx, c, ns = prepared
        signal = np.empty(psd.N_BIG)
        psd._generate_signal(xx.copy(), c, ns, psd.N_BIG, signal)
        if np.all(np.isfinite(signal)):
            return signal
    return None


def _top_line_fraction(signal: np.ndarray) -> tuple[float, list[float], list[float]]:
    f, power = psd._psd(signal)
    peaks, _ = find_peaks(power, distance=PEAK_DISTANCE)
    peaks = peaks[peaks > 0]
    if peaks.size == 0:
        return 0.0, [], []
    if peaks.size > 5:
        peaks = peaks[np.argsort(power[peaks])[-5:]]
    total = float(np.sum(power))
    fraction = float(np.sum(power[peaks]) / total) if total > 0 else float("nan")
    order = np.argsort(f[peaks])
    return (
        fraction,
        [float(v) for v in f[peaks][order]],
        [float(v / total) for v in power[peaks][order]],
    )


def _select_point(sr: Any, horizon: int) -> dict[str, Any]:
    h = np.asarray(sr.h, dtype=np.float64)
    orders = np.asarray(sr.num_taps_grid, dtype=np.int64)
    cutoffs = np.asarray(sr.cutoff_grid, dtype=np.float64)
    chaotic = np.isfinite(h) & (h > LAMBDA_CHAOS_THRESHOLD)
    median_chaotic = float(np.median(h[chaotic]))
    eligible = np.argwhere(chaotic & (cutoffs[None, :] >= BAND[0]) & (cutoffs[None, :] <= BAND[1]))
    ordered = sorted(eligible, key=lambda p: abs(float(h[p[0], p[1]]) - median_chaotic))
    for i, j in ordered:
        i, j = int(i), int(j)
        ensemble = phase1._ensemble(sr, i, j, horizon)
        if int(ensemble["n_finite"]) != int(sr.metadata["n_initial"]):
            continue
        return {
            "i": i,
            "j": j,
            "Nz": int(orders[i]) - 1,
            "omega_c_pi": float(cutoffs[j]),
            "median_chaotic": median_chaotic,
            "n_eligible": int(eligible.shape[0]),
            "lambda1_canonical": float(h[i, j]),
            "ensemble": ensemble,
        }
    raise RuntimeError(f"no point with 25 finite initial conditions in {sr.filter_type} for {BAND}")


def _random_line_fractions(
    sr: Any,
    bank: np.ndarray,
    gains: np.ndarray,
    perturbations: np.ndarray,
    count: int = RANDOM_POINTS,
    seed: int = RANDOM_SEED,
) -> tuple[list[float], int]:
    h = np.asarray(sr.h, dtype=np.float64)
    chaotic = np.argwhere(np.isfinite(h) & (h > LAMBDA_CHAOS_THRESHOLD))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(chaotic))
    values: list[float] = []
    attempted = 0
    for k in order:
        attempted += 1
        i, j = (int(v) for v in chaotic[k])
        signal = _signal(sr, bank, gains, perturbations, i, j)
        if signal is None:
            continue
        fraction, _f, _p = _top_line_fraction(signal)
        if np.isfinite(fraction):
            values.append(fraction)
        if len(values) == count:
            break
    return values, attempted


def run(out_dir: Path, horizon: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = load_all(Path("data/sweeps"))
    points: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    f_base, psd_base = psd._baseline_psd()
    setup_rc()
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.4))
    for ax, filter_type in zip(axes.ravel(), TYPES, strict=True):
        sr = _best_record(recs, filter_type)
        selected = _select_point(sr, horizon)
        i, j = int(selected["i"]), int(selected["j"])
        orders, cutoffs, bank, gains, perturbations = psd._build(sr)
        signal = _signal(sr, bank, gains, perturbations, i, j)
        if signal is None:
            raise RuntimeError(f"no finite signal for selected {filter_type} point")
        wc = float(cutoffs[j])
        bw = float(sr.metadata.get("bandwidth", 0.05))
        f, power = psd._psd(signal)
        frac = psd._fraction(f, power, filter_type, wc, bw)
        base_frac = psd._fraction(f_base, psd_base, filter_type, wc, bw)
        row = {
            "filter_type": filter_type,
            "path": f"{sr.window}/{filter_type}",
            "window": sr.window,
            "kaiser_beta": sr.metadata.get("kaiser_beta"),
            "i": i,
            "j": j,
            "Nz": int(selected["Nz"]),
            "omega_c_pi": wc,
            "gain": float(gains[i, j]),
            "lambda1_canonical": float(selected["lambda1_canonical"]),
            "lambda1_mean": float(selected["ensemble"]["mean"]),
            "lambda1_min": float(selected["ensemble"]["min"]),
            "lambda1_max": float(selected["ensemble"]["max"]),
            "lambda1_std": float(selected["ensemble"]["std"]),
            "n_finite": int(selected["ensemble"]["n_finite"]),
            "median_chaotic": float(selected["median_chaotic"]),
            "n_eligible": int(selected["n_eligible"]),
            "fraction": frac,
            "fraction_baseline": base_frac,
            "f_over_f0": frac / base_frac if base_frac > 0 else float("nan"),
        }
        points.append(row)

        if filter_type in ("highpass", "bandstop"):
            frac5, freqs, parts = _top_line_fraction(signal)
            values, attempted = _random_line_fractions(sr, bank, gains, perturbations)
            lines.append(
                {
                    "filter_type": filter_type,
                    "selected_fraction_top5": frac5,
                    "selected_peak_frequencies": " ".join(f"{v:.6f}" for v in freqs),
                    "selected_peak_fractions": " ".join(f"{v:.6f}" for v in parts),
                    "median_random_fraction_top5": float(np.median(values)),
                    "p10_random_fraction_top5": float(np.percentile(values, 10)),
                    "p90_random_fraction_top5": float(np.percentile(values, 90)),
                    "n_random": len(values),
                    "n_attempted": attempted,
                    "peak_distance_bins": PEAK_DISTANCE,
                    "random_seed": RANDOM_SEED,
                }
            )

        w, response = freqz(
            np.asarray(bank[i, j, : int(orders[i])], dtype=np.float64),
            worN=2048,
            fs=2.0,
        )
        db = 10.0 * np.log10(power / np.max(power))
        i_max = int(np.argmax(power))
        ax.plot(f, db, color="black", lw=0.8, label="PSD $x_3$")
        ax.plot(
            f[i_max],
            0.0,
            marker="v",
            color="black",
            markersize=4.0,
            clip_on=False,
            zorder=10,
        )
        ax.plot(
            w,
            20.0 * np.log10(np.maximum(np.abs(response), 1e-16)),
            color="tab:red",
            lw=0.8,
            label="$|H|^2$",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(-120, 10)
        ax.set_title(
            {
                "lowpass": "(a) Lowpass",
                "highpass": "(b) Highpass",
                "bandpass": "(c) Bandpass",
                "bandstop": "(d) Bandstop",
            }[filter_type],
            fontsize=8,
        )
        ax.set_xlabel(r"$\omega/\pi$", fontsize=8)
        ax.set_ylabel("dB", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=6, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "fig6_median.png", dpi=300, bbox_inches="tight")
    plt.rcParams["svg.hashsalt"] = SVG_SALT
    fig.savefig(
        out_dir / "fig6_median.svg",
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": None},
    )
    svg = out_dir / "fig6_median.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)

    with (out_dir / "fig6_median_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    with (out_dir / "fig6_median_lines.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lines[0]))
        writer.writeheader()
        writer.writerows(lines)
    (out_dir / "fig6_median_summary.json").write_text(
        json.dumps({"points": points, "lines": lines}, indent=2) + "\n",
        encoding="utf-8",
    )
    for row in points:
        print(
            f"{row['filter_type']}: Nz={row['Nz']} wc/pi={row['omega_c_pi']:.6f} "
            f"lambda_mean={row['lambda1_mean']:.6f} "
            f"min={row['lambda1_min']:.6f} max={row['lambda1_max']:.6f} "
            f"f/f0={row['f_over_f0']:.3f}"
        )
    for row in lines:
        print(
            f"{row['filter_type']}: top5 selected={row['selected_fraction_top5']:.4f} "
            f"random median={row['median_random_fraction_top5']:.4f} "
            f"n={row['n_random']}"
        )
    print(f"wrote {out_dir / 'fig6_median.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    parser.add_argument("--horizon", type=int, default=300_000)
    args = parser.parse_args()
    run(Path(args.out_dir), args.horizon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
