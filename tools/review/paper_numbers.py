#!/usr/bin/env python3
"""Wave 3, step 3 and 4: table, figures data and raw numbers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from chaotic_pfc.analysis.sweep import (
    LAMBDA_CHAOS_THRESHOLD,
    load_sweep,
    precompute_fir_bank,
)
from chaotic_pfc.analysis.sweep._types import WINDOW_SIDELOBE_DB
from chaotic_pfc.analysis.sweep_plotting import chaotic_density_norm_table
from chaotic_pfc.paper.numbers import FIXED, fixed_of, spearman_exact

TYPES = ("lowpass", "highpass", "bandpass", "bandstop")
DISPLAY = {
    "lowpass": "Lowpass",
    "highpass": "Highpass",
    "bandpass": "Bandpass",
    "bandstop": "Bandstop",
}
ROOT = Path("data/sweeps")


def _records() -> list[Any]:
    return [load_sweep(p) for p in sorted(ROOT.rglob("variables_lyapunov.npz"))]


def _counts(sr: Any) -> tuple[int, int, int]:
    h = np.asarray(sr.h, dtype=np.float64)
    nc = int(np.count_nonzero(h > LAMBDA_CHAOS_THRESHOLD))
    nnc = int(np.count_nonzero(np.isfinite(h) & (h <= LAMBDA_CHAOS_THRESHOLD)))
    ndiv = int(np.count_nonzero(np.isnan(h)))
    return nc, nnc, ndiv


def _best(recs: list[Any], ft: str) -> Any:
    subset = [r for r in recs if r.filter_type == ft]
    return max(subset, key=lambda r: (_counts(r)[0], -float(r.metadata.get("kaiser_beta", 999.0))))


def _best_kaiser(recs: list[Any], ft: str) -> Any:
    subset = [r for r in recs if r.filter_type == ft and r.window == "kaiser"]
    return max(subset, key=lambda r: (_counts(r)[0], -float(r.metadata["kaiser_beta"])))


def _mean_std(sr: Any) -> tuple[float, float]:
    h = np.asarray(sr.h, dtype=np.float64)
    vals = h[h > LAMBDA_CHAOS_THRESHOLD]
    if vals.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
    return float(np.mean(vals)), std


def _table28(recs: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for ft in TYPES:
        fixed = fixed_of(recs, ft)
        picks = [(fixed[w], name) for w, name, _ in FIXED]
        kb = _best_kaiser(recs, ft)
        picks.append((kb, f"Kaiser ($\\beta_K={float(kb.metadata['kaiser_beta']):.1f}$)"))
        for sr, name in picks:
            nc, nnc, _ = _counts(sr)
            mean, std = _mean_std(sr)
            rows.append(
                {
                    "filter": DISPLAY[ft],
                    "window": name.replace("--", "-"),
                    "Nc": nc,
                    "rho_abs": 100.0 * nc / np.asarray(sr.h).size,
                    "rho_cond": 100.0 * nc / (nc + nnc) if nc + nnc else float("nan"),
                    "mean_lambda1": mean,
                    "std_lambda1": std,
                }
            )
    rows.sort(key=lambda r: (-r["rho_abs"], r["filter"], r["window"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def _pooled(recs: list[Any]) -> list[dict[str, Any]]:
    out = []
    for ft in TYPES:
        subset = [r for r in recs if r.filter_type == ft]
        nc = sum(_counts(r)[0] for r in subset)
        nnc = sum(_counts(r)[1] for r in subset)
        ndiv = sum(_counts(r)[2] for r in subset)
        total = nc + nnc + ndiv
        out.append(
            {
                "filter": DISPLAY[ft],
                "Nc": nc,
                "N_non_chaotic": nnc,
                "N_divergent": ndiv,
                "N_total": total,
                "pct_chaotic": 100.0 * nc / total,
                "pct_non_chaotic": 100.0 * nnc / total,
                "pct_divergent": 100.0 * ndiv / total,
            }
        )
    return out


def _fig7() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nz, cutoff, percent, _ = chaotic_density_norm_table(ROOT)
    maxima = []
    for parity, mask in (("odd", nz % 2 == 1), ("even", nz % 2 == 0)):
        sub = np.where(mask[:, None], percent, -np.inf)
        k, j = np.unravel_index(np.argmax(sub), sub.shape)
        maxima.append(
            {
                "parity": parity,
                "Nz": int(nz[k]),
                "omega_c_pi": float(cutoff[j]),
                "percent": float(percent[k, j]),
            }
        )
    bands = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.0))
    band_rows = []
    for lo, hi in bands:
        cmask = (cutoff > lo) & (cutoff <= hi) if lo > 0 else (cutoff <= hi)
        high = percent[(nz >= 8)[:, None] & cmask[None, :]]
        low = percent[(nz <= 7)[:, None] & cmask[None, :]]
        band_rows.append(
            {
                "omega_c_low": lo,
                "omega_c_high": hi,
                "mean_Nz_ge_8": float(np.nanmean(high)) if high.size else float("nan"),
                "min_Nz_le_7": float(np.nanmin(low)) if low.size else float("nan"),
                "max_Nz_le_7": float(np.nanmax(low)) if low.size else float("nan"),
            }
        )
    return maxima, band_rows


def _fig8(recs: list[Any]) -> list[dict[str, Any]]:
    out = []
    for ft in TYPES:
        subset = [r for r in recs if r.filter_type == ft and r.window == "kaiser"]
        for sr in sorted(subset, key=lambda r: float(r.metadata["kaiser_beta"])):
            nc, _, _ = _counts(sr)
            out.append(
                {
                    "filter": DISPLAY[ft],
                    "kaiser_beta": float(sr.metadata["kaiser_beta"]),
                    "Nc": nc,
                    "rho_abs": 100.0 * nc / np.asarray(sr.h).size,
                }
            )
    return out


def _dc_gain(recs: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for ft in TYPES:
        low_total = low_chaotic = 0
        hp_chaotic_high_gain = 0
        hp_chaotic_total = 0
        hp_total = 0
        for sr in (r for r in recs if r.filter_type == ft):
            orders = np.asarray(sr.num_taps_grid, dtype=np.int64)
            cutoffs = np.asarray(sr.cutoff_grid, dtype=np.float64)
            _, gains = precompute_fir_bank(
                orders,
                cutoffs,
                sr.filter_type,
                sr.window,
                kaiser_beta=float(sr.metadata.get("kaiser_beta", 5.0)),
                bandwidth=float(sr.metadata.get("bandwidth", 0.05)),
            )
            h = np.asarray(sr.h, dtype=np.float64)
            low = np.abs(gains) < 0.1
            low_total += int(low.sum())
            low_chaotic += int(np.count_nonzero(low & (h > LAMBDA_CHAOS_THRESHOLD)))
            if ft == "highpass":
                hp_total += int(h.size)
                hp_chaotic_total += int(np.count_nonzero(h > LAMBDA_CHAOS_THRESHOLD))
                hp_chaotic_high_gain += int(
                    np.count_nonzero((h > LAMBDA_CHAOS_THRESHOLD) & (np.abs(gains) >= 0.1))
                )
        rows.append(
            {
                "filter": DISPLAY[ft],
                "absG_lt_0.1_total": low_total,
                "absG_lt_0.1_chaotic": low_chaotic,
                "absG_lt_0.1_chaotic_pct": 100.0 * low_chaotic / low_total
                if low_total
                else float("nan"),
                "highpass_chaotic_absG_ge_0.1": hp_chaotic_high_gain if ft == "highpass" else "",
                "highpass_chaotic_total": hp_chaotic_total if ft == "highpass" else "",
                "highpass_total": hp_total if ft == "highpass" else "",
            }
        )
    return rows


def _boundaries(recs: list[Any]) -> list[dict[str, Any]]:
    rows = []
    best_lp = _best(recs, "lowpass")
    for nz_target in (3, 4, 5, 6, 7, 10, 20, 40):
        h = np.asarray(best_lp.h, dtype=np.float64)
        nz = np.asarray(best_lp.num_taps_grid, dtype=np.int64) - 1
        mask = nz == nz_target
        if not mask.any():
            rows.append({"filter": "Lowpass", "Nz": nz_target, "max_omega_c_pi": None})
            continue
        cols = np.flatnonzero(np.any(h[mask] > LAMBDA_CHAOS_THRESHOLD, axis=0))
        rows.append(
            {
                "filter": "Lowpass",
                "Nz": nz_target,
                "max_omega_c_pi": float(np.asarray(best_lp.cutoff_grid)[cols.max()])
                if cols.size
                else None,
            }
        )
    best_hp = _best(recs, "highpass")
    h = np.asarray(best_hp.h, dtype=np.float64)
    nz = np.asarray(best_hp.num_taps_grid, dtype=np.int64) - 1
    cutoff = np.asarray(best_hp.cutoff_grid)
    for label, mask in (("Nz_ge_6", nz >= 6), ("Nz_eq_4", nz == 4)):
        sub = h[mask]
        cols = np.flatnonzero(np.any(sub > LAMBDA_CHAOS_THRESHOLD, axis=0))
        rows.append(
            {
                "filter": "Highpass",
                "Nz": label,
                "min_omega_c_pi": float(cutoff[cols.min()]) if cols.size else None,
                "max_omega_c_pi": float(cutoff[cols.max()]) if cols.size else None,
            }
        )
    i, j = np.unravel_index(np.nanargmax(h), h.shape)
    rows.append(
        {
            "filter": "Highpass",
            "Nz": int(nz[i]),
            "max_lambda1": float(h[i, j]),
            "omega_c_pi": float(cutoff[j]),
        }
    )
    best_bp = _best(recs, "bandpass")
    hbp = np.asarray(best_bp.h, dtype=np.float64)
    i, j = np.unravel_index(np.nanargmax(hbp), hbp.shape)
    rows.append(
        {
            "filter": "Bandpass",
            "Nz": int(np.asarray(best_bp.num_taps_grid)[i]) - 1,
            "max_lambda1": float(hbp[i, j]),
            "omega_c_pi": float(np.asarray(best_bp.cutoff_grid)[j]),
        }
    )
    return rows


def _thresholds(recs: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for ft in TYPES:
        sr = _best(recs, ft)
        h = np.asarray(sr.h, dtype=np.float64)
        rows.extend(
            {
                "filter": DISPLAY[ft],
                "threshold": theta,
                "Nc": int(np.count_nonzero(h > theta)),
            }
            for theta in (1e-3, 5e-3, 1e-2)
        )
    return rows


def _latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{rlrrrrrr}",
        r"\toprule",
        r"Rank & Filter & Window & $N_c$ & $\rho_c^{\mathrm{abs}}$ (\%) & "
        r"$\rho_c^{\mathrm{cond}}$ (\%) & $\overline{\lambda}_1$ & $\sigma(\lambda_1)$ \\",
        r"\midrule",
    ]
    lines.extend(
        f"{row['rank']} & {row['filter']} & {row['window']} & {row['Nc']} & "
        f"{row['rho_abs']:.3f} & {row['rho_cond']:.3f} & "
        f"{row['mean_lambda1']:.4f} & {row['std_lambda1']:.4f} \\\\"
        for row in rows
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


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


def run(out_dir: Path) -> None:
    recs = _records()
    n_fixed = sum(1 for r in recs if r.window != "kaiser")
    betas = {round(float(r.metadata["kaiser_beta"]), 10) for r in recs if r.window == "kaiser"}
    expected = n_fixed + len(TYPES) * len(betas)
    if len(recs) != expected:
        raise RuntimeError(f"expected {expected} configurations, found {len(recs)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _table28(recs)
    _write(out_dir / "wave3_table28.csv", table)
    (out_dir / "wave3_table28.tex").write_text(_latex(table), encoding="utf-8")
    _write(out_dir / "wave3_pooled.csv", _pooled(recs))
    maxima, bands = _fig7()
    _write(out_dir / "wave3_fig7.csv", maxima)
    _write(out_dir / "wave3_fig7_bands.csv", bands)
    _write(out_dir / "wave3_fig8.csv", _fig8(recs))
    _write(out_dir / "wave3_dc_gain.csv", _dc_gain(recs))
    _write(out_dir / "wave3_boundaries.csv", _boundaries(recs))
    _write(out_dir / "wave3_thresholds.csv", _thresholds(recs))

    # Concordance on the mask-on data.
    conc = []
    for ft in TYPES:
        fx = fixed_of(recs, ft)
        arrs = [np.asarray(fx[w].h, dtype=np.float64) for w, _, _ in FIXED]
        cls = np.stack(
            [np.where(np.isnan(a), -1, np.where(a > LAMBDA_CHAOS_THRESHOLD, 1, 0)) for a in arrs]
        )
        agree = np.all(cls == cls[0], axis=0)
        orders = np.asarray(fx["boxcar"].num_taps_grid, dtype=np.int64)
        cutoffs = np.asarray(fx["boxcar"].cutoff_grid, dtype=np.float64)
        valid = np.ones_like(agree, dtype=bool)
        for w, _, _ in FIXED:
            bank, _ = precompute_fir_bank(orders, cutoffs, ft, w)
            valid &= np.isfinite(bank).all(axis=2)
        nz = orders - 1
        for label, cond in (("Nz_le_10", nz <= 10), ("Nz_gt_20", nz > 20)):
            mask = cond[:, None] & valid
            conc.append(
                {
                    "filter": DISPLAY[ft],
                    "subset": label,
                    "agree_pct": 100.0 * agree[mask].sum() / mask.sum()
                    if mask.sum()
                    else float("nan"),
                    "n": int(mask.sum()),
                }
            )
    _write(out_dir / "wave3_concordance.csv", conc)

    spearman = []
    ratio = {}
    ratio_path = out_dir / "wave2_selectivity_spearman.csv"
    if ratio_path.exists():
        for row in csv.DictReader(ratio_path.open()):
            ratio[row["window"]] = float(row["median_ratio"])
    for ft in TYPES:
        fixed = fixed_of(recs, ft)
        nc = np.array([_counts(fixed[w])[0] for w, _, _ in FIXED], dtype=np.float64)
        db = np.array([WINDOW_SIDELOBE_DB[w] for w, _, _ in FIXED])
        rho_db, p_db = spearman_exact(db, nc)
        spearman.append(
            {
                "filter": DISPLAY[ft],
                "comparison": "sidelobe_db",
                "rho": rho_db,
                "p_exact": p_db,
                "n": 6,
            }
        )
        if ft == "lowpass":
            ratios = np.array([ratio[w] for w, _, _ in FIXED], dtype=np.float64)
            rho_ratio, p_ratio = spearman_exact(ratios, nc)
            spearman.append(
                {
                    "filter": DISPLAY[ft],
                    "comparison": "median_realized_minus6db_ratio",
                    "rho": rho_ratio,
                    "p_exact": p_ratio,
                    "n": 6,
                }
            )
    _write(out_dir / "wave3_spearman.csv", spearman)

    summary = {
        "configurations": len(recs),
        "cells": int(sum(np.asarray(r.h).size for r in recs)),
        "best_table_rows": table,
        "fig7_maxima": maxima,
        "fig7_bands": bands,
        "spearman": spearman,
    }
    (out_dir / "wave3_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote wave3_table28.csv/tex")
    print("wrote pooled, fig7, fig8, dc_gain, boundaries, thresholds, concordance, spearman")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/review")
    args = parser.parse_args()
    run(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
