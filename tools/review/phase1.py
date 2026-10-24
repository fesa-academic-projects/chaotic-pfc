#!/usr/bin/env python3
"""Gate D3, items 1A to 1E: targeted numerical checks.

The script does not modify the canonical sweep tree.  It selects the Fig. 6
points from the current best configuration of each filter type, repeats them
with the full 25-initial-condition ensemble at a long horizon, and reports the
lowpass order counts, the bandstop attenuation, the validation differences and
the Kaiser low-beta transition.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import psd
from long_horizon import _build, _lambda_at, _post_burn_in
from scipy.signal import freqz, windows

from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD, precompute_fir_bank
from chaotic_pfc.paper.numbers import FIXED, counts, is_kaiser, load_all

TYPES = ("lowpass", "highpass", "bandpass", "bandstop")
REFERENCE = 0.41922
VALIDATION = (("full_spectrum", 0.4185), ("single_vector", 0.4194))


def _best_per_type(recs: list[Any]) -> dict[str, Any]:
    best = {}
    for ft in TYPES:
        subset = [r for r in recs if r.filter_type == ft]
        best[ft] = max(
            subset,
            key=lambda r: (counts(r)["nc"], -float(r.metadata.get("kaiser_beta", 999.0))),
        )
    return best


def _has_finite_signal(
    sr: Any,
    bank: np.ndarray,
    gains: np.ndarray,
    perturbations: np.ndarray,
    i: int,
    j: int,
) -> bool:
    for ic in range(int(sr.metadata["n_initial"])):
        prepared = psd._state(sr, bank, gains, perturbations, i, j, ic)
        if prepared is None:
            continue
        xx, c, Ns = prepared
        trial = np.empty(psd.N_BIG)
        psd._generate_signal(xx.copy(), c, Ns, psd.N_BIG, trial)
        if np.all(np.isfinite(trial)):
            return True
    return False


def _point(sr: Any, edge_margin: float | None) -> tuple[int, int]:
    h = np.asarray(sr.h, dtype=np.float64)
    orders = np.asarray(sr.num_taps_grid, dtype=np.int64)
    cutoff = np.asarray(sr.cutoff_grid, dtype=np.float64)
    chaotic = np.isfinite(h) & (h > LAMBDA_CHAOS_THRESHOLD)
    if edge_margin is not None and sr.filter_type in ("bandpass", "bandstop"):
        chaotic &= (cutoff[None, :] >= edge_margin) & (cutoff[None, :] <= 1.0 - edge_margin)
    points = np.argwhere(chaotic)
    if not points.size:
        raise RuntimeError(f"no eligible chaotic point for {sr.filter_type}")
    ordered = sorted(
        (tuple(p) for p in points),
        key=lambda p: (int(orders[p[0]]) - 1, float(h[p[0], p[1]])),
        reverse=True,
    )
    _orders, _cutoffs, bank, gains, perturbations = psd._build(sr)
    for i, j in ordered:
        if _has_finite_signal(sr, bank, gains, perturbations, int(i), int(j)):
            return int(i), int(j)
    raise RuntimeError(f"no eligible chaotic point with a finite signal for {sr.filter_type}")


def _ensemble(sr: Any, i: int, j: int, horizon: int) -> dict[str, Any]:
    orders, _cutoffs, bank, gains, perturbations = _build(sr)
    n_initial = int(sr.metadata["n_initial"])
    values = []
    burned_out = 0
    for ic in range(n_initial):
        try:
            xx, c, Ns = _post_burn_in(sr, bank, gains, perturbations, i, j, ic)
        except RuntimeError:
            burned_out += 1
            continue
        value = _lambda_at(sr, xx, c, Ns, horizon)
        if np.isfinite(value):
            values.append(float(value))
    values_arr = np.asarray(values, dtype=np.float64)
    finite = values_arr[np.isfinite(values_arr)]
    return {
        "Nz": int(orders[i]) - 1,
        "omega_c_pi": float(np.asarray(sr.cutoff_grid)[j]),
        "lambda1_canonical": float(np.asarray(sr.h)[i, j]),
        "n_ics": n_initial,
        "n_burnout": burned_out,
        "n_finite": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
        "min": float(np.min(finite)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
        "n_above_theta": int(np.count_nonzero(finite > LAMBDA_CHAOS_THRESHOLD)),
        "fraction_above_theta": float(
            np.mean(finite > LAMBDA_CHAOS_THRESHOLD) if finite.size else float("nan")
        ),
    }


def _response(
    num_taps: int, cutoff: float, filter_type: str, window: str, beta: float
) -> tuple[Any, Any]:
    bank, _ = precompute_fir_bank(
        np.array([num_taps], dtype=np.int64),
        np.array([cutoff], dtype=np.float64),
        filter_type,
        window,
        kaiser_beta=beta,
        bandwidth=0.05,
    )
    c = np.asarray(bank[0, 0, :num_taps], dtype=np.float64)
    w, h = freqz(c, worN=8192, fs=2.0)
    return w, np.abs(h)


def _attenuation(w: np.ndarray, mag: np.ndarray, frequency: float) -> float:
    k = int(np.argmin(np.abs(w - frequency)))
    value = mag[k] / mag[0] if mag[0] > 0 else float("nan")
    return float(-20.0 * np.log10(value)) if value > 0 else float("inf")


def _peak_sidelobe(beta: float, taps: int = 101) -> float:
    w = windows.kaiser(taps, beta, sym=True)
    spectrum = np.fft.rfft(w, 2**20)
    db = 20.0 * np.log10(np.maximum(np.abs(spectrum) / np.max(np.abs(spectrum)), 1e-300))
    slope = np.diff(db)
    minima = np.where((slope[:-1] < 0.0) & (slope[1:] > 0.0))[0] + 1
    first_null = next((int(k) for k in minima if db[k] < -10.0), None)
    if first_null is None:
        return float("nan")
    return float(np.max(db[first_null:]))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def run(out_dir: Path, horizon: int, edge_margin: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = load_all(Path("data/sweeps"))
    best = _best_per_type(recs)

    candidate_rows = []
    for ft in TYPES:
        sr = best[ft]
        for variant, margin in (("current", None), ("proposed", edge_margin)):
            i, j = _point(sr, margin)
            row = {
                "filter_type": ft,
                "variant": variant,
                "path": f"{sr.window}/{ft}",
                "i": i,
                "j": j,
                "edge_margin": "" if margin is None else margin,
                "window": sr.window,
                "kaiser_beta": ""
                if not is_kaiser(sr)
                else f"{float(sr.metadata['kaiser_beta']):g}",
            }
            ensemble = _ensemble(sr, i, j, horizon)
            row.update(ensemble)
            candidate_rows.append(row)
            print(ft, variant, row)
    _write(out_dir / "phase1_fig6_candidates.csv", candidate_rows)

    lowpass = best["lowpass"]
    orders = np.asarray(lowpass.num_taps_grid, dtype=np.int64) - 1
    h = np.asarray(lowpass.h, dtype=np.float64)
    lowpass_rows = []
    for nz in (1, 2, 3, 5):
        mask = orders == nz
        nc = int(np.count_nonzero(h[mask] > LAMBDA_CHAOS_THRESHOLD))
        n = int(h[mask].size)
        lowpass_rows.append(
            {
                "Nz": nz,
                "Nc": nc,
                "N_total": n,
                "rho_abs": 100.0 * nc / n if n else float("nan"),
                "window": lowpass.window,
                "kaiser_beta": ""
                if not is_kaiser(lowpass)
                else f"{float(lowpass.metadata['kaiser_beta']):g}",
            }
        )
    _write(out_dir / "phase1_lowpass_Nz_counts.csv", lowpass_rows)

    attenuation_rows = []
    for window, _name, _db in FIXED:
        for nz in (10, 20, 40):
            w, mag = _response(nz + 1, 0.5, "bandstop", window, 5.0)
            attenuation_rows.append(
                {
                    "window": window,
                    "kaiser_beta": "",
                    "Nz": nz,
                    "attenuation_center_db": _attenuation(w, mag, 0.5),
                }
            )
    for beta in (0.0, 0.5, 1.0, 1.5, 2.0):
        for nz in (10, 20, 40):
            w, mag = _response(nz + 1, 0.5, "bandstop", "kaiser", beta)
            attenuation_rows.append(
                {
                    "window": "kaiser",
                    "kaiser_beta": f"{beta:g}",
                    "Nz": nz,
                    "attenuation_center_db": _attenuation(w, mag, 0.5),
                }
            )
    _write(out_dir / "phase1_bandstop_attenuation.csv", attenuation_rows)

    validation_rows = []
    for name, value in VALIDATION:
        abs_diff = abs(value - REFERENCE)
        validation_rows.append(
            {
                "estimate": name,
                "value": value,
                "reference": REFERENCE,
                "abs_difference": abs_diff,
                "relative_pct": 100.0 * abs_diff / REFERENCE,
            }
        )
    validation_rows.append(
        {
            "estimate": "single_vector_vs_full_spectrum",
            "value": VALIDATION[1][1] - VALIDATION[0][1],
            "reference": VALIDATION[0][1],
            "abs_difference": abs(VALIDATION[1][1] - VALIDATION[0][1]),
            "relative_pct": 100.0 * abs(VALIDATION[1][1] - VALIDATION[0][1]) / VALIDATION[0][1],
        }
    )
    _write(out_dir / "phase1_validation.csv", validation_rows)

    sidelobe_rows = [
        {"kaiser_beta": f"{beta:g}", "taps": 101, "peak_sidelobe_db": _peak_sidelobe(beta)}
        for beta in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0)
    ]
    _write(out_dir / "phase1_kaiser_sidelobe.csv", sidelobe_rows)

    transition_rows = []
    for ft in TYPES:
        rectangular = next(r for r in recs if r.filter_type == ft and r.window == "boxcar")
        rc = counts(rectangular)
        for row in sorted(
            (r for r in recs if r.filter_type == ft and is_kaiser(r)),
            key=lambda r: float(r.metadata["kaiser_beta"]),
        ):
            beta = float(row.metadata["kaiser_beta"])
            if beta not in (0.5, 1.0, 1.5, 2.0):
                continue
            k = counts(row)
            transition_rows.append(
                {
                    "filter_type": ft,
                    "kaiser_beta": beta,
                    "Nc": k["nc"],
                    "rho_abs": 100.0 * k["nc"] / k["n"],
                    "rho_cond": 100.0 * k["nc"] / k["nb"],
                    "rectangular_Nc": rc["nc"],
                    "rectangular_rho_abs": 100.0 * rc["nc"] / rc["n"],
                    "rectangular_rho_cond": 100.0 * rc["nc"] / rc["nb"],
                }
            )
    _write(out_dir / "phase1_beta_transition.csv", transition_rows)

    summary = {
        "horizon": horizon,
        "edge_margin": edge_margin,
        "best": {
            ft: {
                "window": sr.window,
                "kaiser_beta": None if not is_kaiser(sr) else float(sr.metadata["kaiser_beta"]),
                "Nc": counts(sr)["nc"],
            }
            for ft, sr in best.items()
        },
        "fig6": candidate_rows,
        "lowpass_Nz": lowpass_rows,
        "validation": validation_rows,
        "sidelobe": sidelobe_rows,
        "beta_transition": transition_rows,
    }
    (out_dir / "phase1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir / 'phase1_summary.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    parser.add_argument("--horizon", type=int, default=300_000)
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.125,
        help="exclude bandpass/bandstop cutoffs outside [margin, 1-margin]",
    )
    args = parser.parse_args()
    run(Path(args.out_dir), args.horizon, args.edge_margin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
