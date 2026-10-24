#!/usr/bin/env python3
"""Wave 2c, item D: PSD of x3 and passband power fractions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from repro import max_difference, reproduce_cell
from scipy.signal import freqz, welch

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
from chaotic_pfc.plotting.figures import setup_rc

N_BIG = 2**18
N_SMALL = 2**16
NPERSIG = 4096
SEED = 42
SVG_SALT = "chaotic-pfc-psd"


@njit(cache=True)
def _generate_signal(xx: Any, c: Any, Ns: int, N: int, out: Any) -> None:
    tick = 0
    for n in range(N):
        if Ns <= 2:
            value = 0.0
            for k in range(Ns):
                value += c[k] * xx[tick, k]
            out[n] = value
            _henon_n12_inplace(xx[tick], xx[1 - tick], 1.4, 0.3, c)
        else:
            out[n] = xx[tick, 2]
            _henon_nN_inplace(xx[tick], xx[1 - tick], 1.4, 0.3, c)
        tick = 1 - tick


def _best_per_type() -> dict[str, tuple[Any, str]]:
    root = Path("data/sweeps")
    best: dict[str, tuple[float, int, float, Any, str]] = {}
    for path in sorted(root.rglob("variables_lyapunov.npz")):
        sr = load_sweep(path)
        h = np.asarray(sr.h, dtype=np.float64)
        nc = int(np.count_nonzero(h > LAMBDA_CHAOS_THRESHOLD))
        rho = nc / h.size
        beta = -float(sr.metadata.get("kaiser_beta", 5.0))
        value = (rho, nc, beta, sr, path.relative_to(root).as_posix())
        if sr.filter_type not in best or value[:3] > best[sr.filter_type][:3]:
            best[sr.filter_type] = value
    return {k: (v[3], v[4]) for k, v in best.items()}


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


def _state(
    sr: Any,
    bank: np.ndarray,
    gains: np.ndarray,
    perturbations: np.ndarray,
    i: int,
    j: int,
    ic: int = 0,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    orders = np.asarray(sr.num_taps_grid, dtype=np.int64)
    Ns = int(orders[i])
    c = np.asarray(bank[i, j, :Ns], dtype=np.float64)
    gain = float(gains[i, j])
    p = fixed_point_positive(gain, 1.4, 0.3)
    noise = perturbations[i, j, ic, :Ns]
    dim = 2 if Ns <= 2 else max(Ns, 4)
    xx = np.empty((2, dim))
    xx.fill(0.0)
    if Ns <= 2:
        xx[0, 0] = 0.1 * noise[0] + p
        xx[0, 1] = 0.0 if Ns == 1 else 0.1 * noise[1] + p
        _burn_in_n12(xx, int(sr.metadata["Nitera"]), 1.4, 0.3, c)
    else:
        xx[0, 0] = 0.1 * noise[0] + p
        xx[0, 1] = 0.1 * noise[1] + p
        xx[0, 2] = 0.1 * noise[2] + gain * p
        for k in range(3, Ns):
            xx[0, k] = 0.1 * noise[k] + p
        _burn_in_nN(xx, Ns, int(sr.metadata["Nitera"]), 1.4, 0.3, c)
    if not np.isfinite(xx[0, 0]):
        return None
    return xx, c, Ns


def _psd(x3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f, psd = welch(x3, fs=2.0, window="hann", nperseg=NPERSIG)
    return f, psd


def _fraction(f: np.ndarray, psd: np.ndarray, filter_type: str, wc: float, bw: float) -> float:
    if filter_type == "lowpass":
        mask = f <= wc
    elif filter_type == "highpass":
        mask = f >= wc
    elif filter_type == "bandpass":
        mask = np.abs(f - wc) <= bw / 2.0
    else:
        mask = np.abs(f - wc) > bw / 2.0
    total = float(np.sum(psd))
    band = float(np.sum(psd[mask]))
    value = band / total if total > 0 else float("nan")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"PSD fraction outside [0,1]: {value}")
    return value


def _baseline_psd() -> tuple[np.ndarray, np.ndarray]:
    p = fixed_point_positive(1.0, 1.4, 0.3)
    noise = _precompute_perturbations(np.array([1]), 1, 1, 42)[0, 0, 0, 0]
    xx = np.empty((2, 2))
    xx[0, 0] = 0.1 * noise + p
    xx[0, 1] = 0.0
    c = np.array([1.0])
    _burn_in_n12(xx, 500, 1.4, 0.3, c)
    signal = np.empty(N_SMALL)
    _generate_signal(xx.copy(), c, 1, N_SMALL, signal)
    return _psd(signal)


def run(out_dir: Path, edge_margin: float = 0.0, prefix: str = "") -> None:
    best = _best_per_type()
    figure_path = out_dir / (f"{prefix}_fig_psd.svg" if prefix else "fig_psd_review.svg")
    meta_path = out_dir / (f"{prefix}_psd_meta.csv" if prefix else "wave2_psd_meta.csv")
    fraction_path = out_dir / (
        f"{prefix}_psd_fraction.csv" if prefix else "wave2c_psd_fraction.csv"
    )
    summary_path = out_dir / (
        f"{prefix}_psd_fraction_summary.csv" if prefix else "wave2c_psd_fraction_summary.csv"
    )
    setup_rc()
    plt.rcParams["svg.hashsalt"] = SVG_SALT
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.4))
    meta: list[dict[str, Any]] = []
    fraction_rows: list[dict[str, Any]] = []
    max_repro = 0.0
    f_base, psd_base = _baseline_psd()
    order = ("lowpass", "highpass", "bandpass", "bandstop")
    for ax, filter_type in zip(axes.ravel(), order, strict=True):
        sr, rel = best[filter_type]
        orders, cutoffs, bank, gains, perturbations = _build(sr)
        h = np.asarray(sr.h, dtype=np.float64)
        mask = (~np.isnan(h)) & (h > LAMBDA_CHAOS_THRESHOLD)
        if edge_margin > 0.0 and filter_type in ("bandpass", "bandstop"):
            mask &= (cutoffs[None, :] >= edge_margin) & (cutoffs[None, :] <= 1.0 - edge_margin)
        chaotic = np.argwhere(mask)
        if not chaotic.size:
            raise RuntimeError(f"no chaotic point for {filter_type}")
        sorted_points = sorted(
            chaotic.tolist(),
            key=lambda p: (int(orders[p[0]]) - 1, h[p[0], p[1]]),
            reverse=True,
        )
        x3 = None
        xx = c = None
        Ns = 0
        i = j = 0
        for i, j in sorted_points:
            for ic in range(int(sr.metadata["n_initial"])):
                prepared = _state(sr, bank, gains, perturbations, i, j, ic)
                if prepared is None:
                    continue
                xx, c, Ns = prepared
                trial = np.empty(N_BIG)
                _generate_signal(xx.copy(), c, Ns, N_BIG, trial)
                if np.all(np.isfinite(trial)):
                    x3 = trial
                    break
            if x3 is not None:
                break
        if x3 is None:
            raise RuntimeError(f"no chaotic point with a finite signal for {filter_type}")
        wc = float(cutoffs[j])
        bw = float(sr.metadata.get("bandwidth", 0.05))
        x3 = None
        xx = c = None
        Ns = 0
        for ic in range(int(sr.metadata["n_initial"])):
            prepared = _state(sr, bank, gains, perturbations, i, j, ic)
            if prepared is None:
                continue
            xx, c, Ns = prepared
            trial = np.empty(N_BIG)
            _generate_signal(xx.copy(), c, Ns, N_BIG, trial)
            if np.all(np.isfinite(trial)):
                x3 = trial
                break
        if x3 is None:
            raise RuntimeError(f"selected point has no finite signal for {filter_type}")
        f, psd = _psd(x3)
        psd_db = 10.0 * np.log10(psd / np.max(psd))
        i_max = int(np.argmax(psd))
        w, response = freqz(c, worN=2048, fs=2.0)
        ax.plot(f, psd_db, color="black", lw=0.8, label="PSD $x_3$")
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
        meta.append(
            {
                "filter_type": filter_type,
                "path": rel,
                "i": int(i),
                "j": int(j),
                "Nz": int(orders[i]) - 1,
                "omega_c_pi": wc,
                "gain": float(gains[i, j]),
                "lambda1": float(h[i, j]),
                "samples": N_BIG,
                "nperseg": NPERSIG,
            }
        )

        for i2, j2 in chaotic:
            reproduced = reproduce_cell(sr, bank, gains, perturbations, int(i2), int(j2))
            diff, nan_mismatch = max_difference(
                np.asarray([[h[i2, j2]]]), np.asarray([[reproduced]])
            )
            max_repro = max(max_repro, diff)
            if diff != 0.0 or nan_mismatch:
                raise SystemExit(
                    f"reproduction failed for {rel}/{i2}/{j2}: {diff}, NaN={nan_mismatch}"
                )
            prepared2 = None
            for ic in range(int(sr.metadata["n_initial"])):
                prepared2 = _state(sr, bank, gains, perturbations, int(i2), int(j2), ic)
                if prepared2 is not None:
                    break
            if prepared2 is None:
                fraction = float("nan")
            else:
                xx2, c2, Ns2 = prepared2
                signal = np.empty(N_SMALL)
                _generate_signal(xx2.copy(), c2, Ns2, N_SMALL, signal)
                if not np.all(np.isfinite(signal)):
                    fraction = float("nan")
                else:
                    f2, psd2 = _psd(signal)
                    fraction = _fraction(f2, psd2, sr.filter_type, float(cutoffs[j2]), bw)
            fraction_rows.append(
                {
                    "filter_type": filter_type,
                    "path": rel,
                    "i": int(i2),
                    "j": int(j2),
                    "Nz": int(orders[i2]) - 1,
                    "omega_c_pi": float(cutoffs[j2]),
                    "gain": float(gains[i2, j2]),
                    "lambda1": float(h[i2, j2]),
                    "fraction": fraction,
                    "fraction_baseline": _fraction(
                        f_base, psd_base, sr.filter_type, float(cutoffs[j2]), bw
                    ),
                }
            )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    with meta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(meta[0]))
        writer.writeheader()
        writer.writerows(meta)
    with fraction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fraction_rows[0]))
        writer.writeheader()
        writer.writerows(fraction_rows)

    summary: list[dict[str, Any]] = []
    for filter_type in order:
        subset = [r for r in fraction_rows if r["filter_type"] == filter_type]
        for band_name, band_mask in (
            ("all", lambda nz: True),
            ("Nz_le_7", lambda nz: nz <= 7),
            ("Nz_ge_8", lambda nz: nz >= 8),
        ):
            values = np.asarray(
                [
                    float(r["fraction"])
                    for r in subset
                    if band_mask(int(r["Nz"])) and np.isfinite(float(r["fraction"]))
                ],
                dtype=np.float64,
            )
            base_values = np.asarray(
                [
                    float(r["fraction_baseline"])
                    for r in subset
                    if band_mask(int(r["Nz"])) and np.isfinite(float(r["fraction_baseline"]))
                ],
                dtype=np.float64,
            )
            summary.append(
                {
                    "filter_type": filter_type,
                    "Nz_band": band_name,
                    "n": int(values.size),
                    "median": float(np.median(values)) if values.size else float("nan"),
                    "p10": float(np.percentile(values, 10)) if values.size else float("nan"),
                    "p90": float(np.percentile(values, 90)) if values.size else float("nan"),
                    "n_baseline": int(base_values.size),
                    "median_baseline": float(np.median(base_values))
                    if base_values.size
                    else float("nan"),
                    "p10_baseline": float(np.percentile(base_values, 10))
                    if base_values.size
                    else float("nan"),
                    "p90_baseline": float(np.percentile(base_values, 90))
                    if base_values.size
                    else float("nan"),
                }
            )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"max |h_repro - h_stored| over chaotic cells: {max_repro}")
    print(f"wrote {figure_path}")
    for row in meta:
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.0,
        help="exclude bandpass/bandstop cutoffs outside [margin, 1-margin]",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="prefix for the figure and CSV names (default: manuscript names)",
    )
    args = parser.parse_args()
    run(Path(args.out_dir), args.edge_margin, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
