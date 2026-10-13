"""Compute every number of the manuscript that derives from the sweeps.

Each quantity is printed with a fixed label, so the output of two sweep
directories can be compared line by line. Fractions are computed from
integer counts and rounded with ``decimal.ROUND_HALF_UP``.

Usage::

    chaotic-pfc paper numbers --sweeps DIR --json OUT.json
    chaotic-pfc paper numbers --compare OLD.json NEW.json

The slow quantities (seed runs, the Ns=1 validation) are computed by
``--extras`` and stored in the same JSON, so that the comparison can read
them when they are present.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

from chaotic_pfc.analysis.stats import lmax_statistics
from chaotic_pfc.analysis.sweep import load_sweep, precompute_fir_bank

TYPES = ("lowpass", "highpass", "bandpass", "bandstop")
LOW_GAIN = 0.1
# Width of the pass/stop band used by the versioned sweeps (bandpass/bandstop);
# it is not stored in the metadata.
BANDWIDTH = 0.05
FIXED = (
    ("boxcar", "Rectangular", 13),
    ("bartlett", "Bartlett", 27),
    ("hann", "Hann", 32),
    ("hamming", "Hamming", 43),
    ("blackman", "Blackman", 58),
    ("blackmanharris", "Blackman--Harris", 92),
)
NITERA = 500
NMAP = 3000
N_BOOT = 1000
BOOT_SEED = 42
CACHE = Path(".cache/paper")
SEED_LIST = (1, 2, 3, 4, 5, 6, 7, 8, 9, 42)


# ── formatting ──────────────────────────────────────────────────────────────


def fmt(x, places: int) -> str:
    """Round a float with ROUND_HALF_UP to *places* decimals."""
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def pct(n: int, d: int, places: int) -> str:
    """Percentage computed from integers, rounded half up."""
    if d == 0:
        return "n/a"
    q = Decimal(1).scaleb(-places)
    return str((Decimal(100 * n) / Decimal(d)).quantize(q, rounding=ROUND_HALF_UP))


# ── loading and counts ──────────────────────────────────────────────────────


def load_all(root: Path):
    """Load every sweep checkpoint under ``root``."""
    paths = sorted(root.rglob("variables_lyapunov.npz"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints under {root}")
    return [load_sweep(p) for p in paths]


def counts(sr) -> dict:
    """Return the chaotic/periodic/divergent counts of a sweep."""
    h = np.asarray(sr.h, dtype=np.float64)
    nc = int(np.count_nonzero(h > 0))
    nper = int(np.count_nonzero((h <= 0) & ~np.isnan(h)))
    ndiv = int(np.count_nonzero(np.isnan(h)))
    return {"nc": nc, "nper": nper, "nb": nc + nper, "ndiv": ndiv, "n": h.size}


def chaotic_values(sr) -> np.ndarray:
    """Return the λ_max values of the chaotic grid points."""
    h = np.asarray(sr.h, dtype=np.float64)
    return h[h > 0]


def is_kaiser(sr) -> bool:
    """Return whether the sweep used the Kaiser window."""
    return sr.window == "kaiser"


def fixed_of(recs, ft: str) -> dict:
    """Return the fixed-window sweeps for a filter type."""
    out = {}
    for w, _name, _ in FIXED:
        match = [r for r in recs if r.filter_type == ft and r.window == w]
        if len(match) != 1:
            raise ValueError(f"expected one {ft}/{w} sweep, found {len(match)}")
        out[w] = match[0]
    return out


def best_kaiser(recs, ft: str):
    """Return the best Kaiser-β sweep for a filter type."""
    cands = [r for r in recs if r.filter_type == ft and is_kaiser(r)]
    if len(cands) != 25:
        raise ValueError(f"expected 25 Kaiser sweeps for {ft}, found {len(cands)}")
    return max(cands, key=lambda r: (counts(r)["nc"], -float(r.metadata["kaiser_beta"])))


def kaiser_label(sr) -> str:
    """Format the Kaiser-β label used in the tables."""
    return f"Kaiser ($\\beta_K={float(sr.metadata['kaiser_beta']):.1f}$)"


def candidates(recs, ft: str) -> list:
    """Six fixed windows plus the best Kaiser beta: the 7 rows per type."""
    fx = fixed_of(recs, ft)
    rows = []
    for w, name, db in FIXED:
        rows.append({"name": name, "sr": fx[w], "db": db})
    kb = best_kaiser(recs, ft)
    rows.append({"name": kaiser_label(kb), "sr": kb, "db": None})
    for row in rows:
        row["nc"] = counts(row["sr"])["nc"]
    return rows


def table2_row(row) -> dict:
    """Table II/III entry, with the statistics of ``stats.lmax_statistics``."""
    sr = row["sr"]
    c = counts(sr)
    st = lmax_statistics(sr, region="chaotic", n_bootstrap=N_BOOT, seed=BOOT_SEED)
    return {
        "name": row["name"],
        "nc": c["nc"],
        "rho_fin": pct(c["nc"], c["nb"], 1),
        "rho_abs": pct(c["nc"], c["n"], 1),
        "mean": fmt(st["mean"], 4),
        "std": fmt(st["std"], 4),
        "ci": f"[{fmt(st['ci_95_low'], 4)}, {fmt(st['ci_95_high'], 4)}]",
    }


# ── sections ────────────────────────────────────────────────────────────────


def section_recalc(recs, out: dict) -> None:
    """Fill the ``RECALC.*`` labels (fallback recomputation)."""
    total = sum(counts(r)["n"] for r in recs)
    if all("n_fallback" in r.metadata for r in recs):
        fb = sum(int(r.metadata["n_fallback"]) for r in recs)
        out["RECALC.n_fallback"] = str(fb)
        out["RECALC.pct_fallback"] = pct(fb, total, 2)
    else:
        out["RECALC.n_fallback"] = "n/a (metadata without n_fallback)"
        out["RECALC.pct_fallback"] = "n/a"
    medians, cap_points, cap_cfgs = [], 0, 0
    for r in recs:
        it = np.asarray(r.n_iters_used, dtype=np.float64)
        fin = it[~np.isnan(np.asarray(r.h, dtype=np.float64))]
        if fin.size:
            medians.append(float(np.median(fin)))
        hit = int(np.count_nonzero(fin >= NMAP))
        cap_points += hit
        cap_cfgs += int(hit > 0)
    out["RECALC.median_iters_min"] = fmt(min(medians), 0)
    out["RECALC.median_iters_max"] = fmt(max(medians), 0)
    out["RECALC.median_iters_floor"] = fmt(min(medians), 0)
    out["RECALC.cap_points"] = str(cap_points)
    out["RECALC.cap_pct"] = pct(cap_points, total, 2)
    out["RECALC.cap_configs"] = str(cap_cfgs)


def section_table2_table3(recs, out: dict) -> dict:
    """Fill the ``T2.*``/``T3.*`` labels and return the best rows."""
    best = {}
    all_rows: list = []
    for ft in TYPES:
        rows = sorted(candidates(recs, ft), key=lambda r: (-r["nc"], r["name"]))
        best[ft] = rows
        for k, row in enumerate(rows[:2], 1):
            t = table2_row(row)
            base = f"T2.{ft}.{k}"
            out[f"{base}.name"] = t["name"]
            out[f"{base}.Nc"] = str(t["nc"])
            out[f"{base}.rho_fin"] = t["rho_fin"]
            out[f"{base}.rho_abs"] = t["rho_abs"]
            out[f"{base}.mean"] = t["mean"]
            out[f"{base}.std"] = t["std"]
            out[f"{base}.ci"] = t["ci"]
        all_rows.extend((ft, row) for row in rows)
    ordered = sorted(all_rows, key=lambda p: (-p[1]["nc"], p[0], p[1]["name"]))
    for k, (ft, row) in enumerate(ordered, 1):
        t = table2_row(row)
        out[f"T3.rank{k:02d}"] = (
            f"{ft} | {t['name']} | Nc={t['nc']} | rho_fin={t['rho_fin']} | "
            f"mean={t['mean']} | ci={t['ci']}"
        )
    return best


def spearman_exact(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return the Spearman correlation and its exact p-value."""

    def rank_corr(a, b):
        ra = _rankdata(a)
        rb = _rankdata(b)
        return float(np.corrcoef(ra, rb)[0, 1])

    r_obs = rank_corr(x, y)
    vals = y.copy()
    perms = list(itertools.permutations(range(len(vals))))
    hits = 0
    for p in perms:
        r = rank_corr(x, vals[list(p)])
        if abs(r) >= abs(r_obs) - 1e-12:
            hits += 1
    return r_obs, hits / len(perms)


def _rankdata(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(a, method="average")


def section_spearman(recs, out: dict) -> None:
    """Fill the ``SPEARMAN.*`` labels."""
    db = np.array([d for _, _, d in FIXED], dtype=np.float64)
    for ft in TYPES:
        fx = fixed_of(recs, ft)
        nc = np.array([counts(fx[w])["nc"] for w, _, _ in FIXED], dtype=np.float64)
        r, p = spearman_exact(db, nc)
        out[f"SP.{ft}"] = f"r_s={fmt(r, 3)}, p={fmt(p, 4)}"


def section_totals(recs, out: dict) -> None:
    """Fill the ``TOTALS.*`` labels."""
    tot_c = tot_p = tot_d = 0
    for ft in TYPES:
        c = c_p = c_d = n_all = 0
        for r in (x for x in recs if x.filter_type == ft):
            k = counts(r)
            c += k["nc"]
            c_p += k["nper"]
            c_d += k["ndiv"]
            n_all += k["n"]
        nb = c + c_p
        out[f"TOT.{ft}.chaotic"] = str(c)
        out[f"TOT.{ft}.periodic"] = str(c_p)
        out[f"TOT.{ft}.divergent"] = str(c_d)
        out[f"TOT.{ft}.pct_divergent"] = pct(c_d, n_all, 2)
        out[f"TOT.{ft}.rho_fin_agg"] = pct(c, nb, 2)
        tot_c += c
        tot_p += c_p
        tot_d += c_d
    out["TOT.chaotic"] = str(tot_c)
    out["TOT.periodic"] = str(tot_p)
    out["TOT.divergent"] = str(tot_d)


def section_maps(best: dict, out: dict) -> None:
    """Fill the ``MAPS.*`` labels."""
    lp = best["lowpass"][0]["sr"]
    h = np.asarray(lp.h, dtype=np.float64)
    wc = np.asarray(lp.cutoff_grid, dtype=np.float64)  # already omega_c / pi
    nz = np.asarray(lp.num_taps_grid, dtype=np.int64) - 1
    entries, none = [], []
    for i, z in enumerate(nz):
        mask = h[i] > 0
        if mask.any():
            entries.append(f"{z}:{fmt(wc[np.flatnonzero(mask).max()], 2)}")
        else:
            none.append(str(z))
    out["MAP.lowpass.max_wc_by_Nz"] = ", ".join(entries)
    out["MAP.lowpass.Nz_without_chaos"] = ", ".join(none) if none else "none"

    hp = best["highpass"][0]["sr"]
    h = np.asarray(hp.h, dtype=np.float64)
    wc = np.asarray(hp.cutoff_grid, dtype=np.float64)
    nz = np.asarray(hp.num_taps_grid, dtype=np.int64) - 1
    ge6 = []
    for i, z in enumerate(nz):
        if z >= 6:
            mask = h[i] > 0
            ge6.append(
                f"{z}:{fmt(wc[np.flatnonzero(mask).max()], 2)}" if mask.any() else f"{z}:none"
            )
    out["MAP.highpass.max_wc_Nz_ge6"] = ", ".join(ge6)
    i4 = int(np.flatnonzero(nz == 4)[0])
    cols4 = np.flatnonzero(h[i4] > 0)
    out["MAP.highpass.Nz4_band"] = (
        f"[{fmt(wc[cols4.min()], 2)}, {fmt(wc[cols4.max()], 2)}]" if cols4.size else "none"
    )
    hm = np.nanmax(h)
    ii, jj = np.unravel_index(np.nanargmax(h), h.shape)
    out["MAP.highpass.lambda_max"] = f"{fmt(hm, 4)} at Nz={nz[ii]}, wc/pi={fmt(wc[jj], 5)}"

    bp = best["bandpass"][0]["sr"]
    out["MAP.bandpass.lambda_max"] = fmt(np.nanmax(np.asarray(bp.h, dtype=np.float64)), 4)

    bs = best["bandstop"][0]["sr"]
    h = np.asarray(bs.h, dtype=np.float64)
    nz = np.asarray(bs.num_taps_grid, dtype=np.int64) - 1
    no_div = [str(z) for i, z in enumerate(nz) if not np.isnan(h[i]).any()]
    all_div = [str(z) for i, z in enumerate(nz) if np.isnan(h[i]).all()]
    out["MAP.bandstop.Nz_without_divergent"] = ", ".join(no_div) if no_div else "none"
    out["MAP.bandstop.Nz_all_divergent"] = ", ".join(all_div) if all_div else "none"


def section_density(root: Path, out: dict) -> None:
    """Fill the ``DENSITY.*`` labels from the density table."""
    from chaotic_pfc.analysis.sweep_plotting import chaotic_density_norm_table

    nz, wc, percent, n_cont = chaotic_density_norm_table(root)
    k = np.rint(percent * n_cont[:, None] / 100.0)
    total = np.broadcast_to(n_cont[:, None], percent.shape)

    def pooled(rows: np.ndarray, cols: np.ndarray) -> str:
        sub = np.ix_(rows, cols)
        return pct(int(np.nansum(k[sub])), int(np.nansum(total[sub])), 1)

    bands = {
        "<=0.1": wc <= 0.1,
        "(0.1,0.2]": (wc > 0.1) & (wc <= 0.2),
        "(0.2,0.4]": (wc > 0.2) & (wc <= 0.4),
        ">0.4": wc > 0.4,
    }
    ge8 = nz >= 8
    le7 = nz <= 7
    for label, cols in bands.items():
        out[f"DEN.Nz_ge8.wc_{label}"] = pooled(ge8, cols)
    le7_vals = [float(pooled(le7, cols)) for cols in bands.values()]
    out["DEN.Nz_le7_band_range"] = f"[{fmt(min(le7_vals), 1)}, {fmt(max(le7_vals), 1)}]"

    def point(kk, jj):
        n = int(n_cont[kk])
        count = round(float(percent[kk, jj]) * n / 100.0)
        return f"Nz={nz[kk]}, wc/pi={fmt(wc[jj], 4)}, {pct(count, n, 1)}%, k={count} of n={n}"

    kk, jj = np.unravel_index(np.nanargmax(percent), percent.shape)
    out["DEN.max_global"] = point(kk, jj)
    even = np.where((nz % 2 == 0)[:, None], percent, -np.inf)
    kk, jj = np.unravel_index(np.argmax(even), even.shape)
    out["DEN.max_even_Nz"] = point(kk, jj)


def section_concordance(recs, out: dict) -> None:
    """Fill the ``CONCORDANCE.*`` labels."""
    for ft in TYPES:
        fx = fixed_of(recs, ft)
        arrs = [np.asarray(fx[w].h, dtype=np.float64) for w, _, _ in FIXED]
        cls = np.stack([np.where(np.isnan(a), -1, np.where(a > 0, 1, 0)) for a in arrs])
        agree = np.all(cls == cls[0], axis=0)
        nz = np.asarray(fx["boxcar"].num_taps_grid, dtype=np.int64) - 1
        le10 = nz <= 10
        gt20 = nz > 20
        out[f"CON.{ft}.pooled_Nz_le10"] = pct(int(agree[le10].sum()), int(agree[le10].size), 1)
        out[f"CON.{ft}.pooled_Nz_gt20"] = pct(int(agree[gt20].sum()), int(agree[gt20].size), 1)
        if ft == "lowpass":
            ncut = agree.shape[1]
            out["CON.lowpass.max_disagreement_per_Nz"] = pct(
                ncut - int(agree.sum(axis=1).min()), ncut, 1
            )


def section_kaiser(recs, out: dict) -> None:
    """Fill the ``KAISER.*`` labels."""
    for ft in TYPES:
        cands = sorted(
            (
                (float(r.metadata["kaiser_beta"]), r)
                for r in recs
                if r.filter_type == ft and is_kaiser(r)
            ),
            key=lambda p: p[0],
        )
        vals = {b: pct(counts(r)["nc"], counts(r)["n"], 1) for b, r in cands}
        out[f"KAI.{ft}.all"] = ", ".join(f"{b:g}:{vals[b]}" for b in sorted(vals))
        for b in (2.0, 10.0, 11.5, 13.0, 13.5, 14.0):
            out[f"KAI.{ft}.beta{b:g}"] = vals[b]

    def span(ft, lo, hi):
        sub = [
            float(pct(counts(r)["nc"], counts(r)["n"], 1))
            for b, r in _kaiser_pairs(recs, ft)
            if lo <= b <= hi
        ]
        return f"[{fmt(min(sub), 1)}, {fmt(max(sub), 1)}]"

    out["KAI.lowpass.plateau_10_14"] = span("lowpass", 10.0, 14.0)
    out["KAI.bandpass.plateau_10_14"] = span("bandpass", 10.0, 14.0)
    out["KAI.highpass.plateau_12_14"] = span("highpass", 12.0, 14.0)
    out["KAI.bandstop.all"] = span("bandstop", 2.0, 14.0)
    lp = dict(_kaiser_pairs(recs, "lowpass"))
    out["KAI.lowpass.Nc_13.0"] = str(counts(lp[13.0])["nc"])
    out["KAI.lowpass.Nc_13.5"] = str(counts(lp[13.5])["nc"])


def _kaiser_pairs(recs, ft):
    return [
        (float(r.metadata["kaiser_beta"]), r) for r in recs if r.filter_type == ft and is_kaiser(r)
    ]


def section_robustness(recs, best: dict, out: dict) -> None:
    """Fill the ``ROBUST.*`` labels."""

    def frac(sr, fn):
        h = np.asarray(sr.h, dtype=np.float64)
        s = np.asarray(sr.h_std, dtype=np.float64)
        ch = h > 0
        if not ch.any():
            return 0, 0
        return int(np.count_nonzero(fn(h, s)[ch])), int(ch.sum())

    def pct_pair(recs_):
        num_se = den = n1 = n5 = 0
        for r in recs_:
            a, b = frac(r, lambda h, s: h > 2 * s / math.sqrt(25))
            num_se += a
            den += b
            a1, _ = frac(r, lambda h, s: h > 1e-3)
            a5, _ = frac(r, lambda h, s: h > 5e-3)
            n1 += a1
            n5 += a5
        return num_se, den, n1, n5

    num_se, den, n1, n5 = pct_pair(recs)
    out["ROB.all.pct_mean_above_2se"] = pct(num_se, den, 1)
    out["ROB.all.kept_1e-3"] = pct(n1, den, 1)
    out["ROB.all.kept_5e-3"] = pct(n5, den, 1)

    t2 = []
    t2_1, t2_5 = [], []
    for ft in TYPES:
        for row in best[ft][:2]:
            a, b = pct_pair([row["sr"]])[:2]
            t2.append(float(pct(a, b, 1)))
            a1, _ = frac(row["sr"], lambda h, s: h > 1e-3)
            a5, _ = frac(row["sr"], lambda h, s: h > 5e-3)
            _, bb = frac(row["sr"], lambda h, s: h > 0)
            t2_1.append(float(pct(a1, bb, 1)))
            t2_5.append(float(pct(a5, bb, 1)))
    out["ROB.tableII.pct_mean_above_2se"] = f"[{fmt(min(t2), 1)}, {fmt(max(t2), 1)}]"
    out["ROB.tableII.kept_1e-3"] = f"[{fmt(min(t2_1), 1)}, {fmt(max(t2_1), 1)}]"
    out["ROB.tableII.kept_5e-3"] = f"[{fmt(min(t2_5), 1)}, {fmt(max(t2_5), 1)}]"


def section_cost(recs, out: dict) -> None:
    """Fill the ``COST.*`` labels."""
    total = 0.0
    for r in recs:
        it = np.asarray(r.n_iters_used, dtype=np.float64)
        total += float(np.nansum(it))
    out["COST.sum_n_iters_used"] = f"{total:.4g}"


# ── DC-gain analysis ────────────────────────────────────────────────────────


def _abs_gain(sr) -> np.ndarray:
    """|G| per grid point, G = sum of the FIR coefficients (DC gain)."""
    beta = float(sr.metadata.get("kaiser_beta", 5.0))
    _, gains = precompute_fir_bank(
        np.asarray(sr.num_taps_grid),
        np.asarray(sr.cutoff_grid),
        sr.filter_type,
        sr.window,
        kaiser_beta=beta,
        bandwidth=BANDWIDTH,
    )
    return np.abs(gains)


def _gain_band_table(recs, filter_type: str, out: dict, key: str) -> None:
    """Count |G| >= 0.1 points per N_z x (omega_c/pi band)."""
    nz_vals = sorted(
        {
            int(o) - 1
            for r in recs
            if r.filter_type == filter_type
            for o in np.asarray(r.num_taps_grid)
        }
    )
    n_bands = 10
    counts = {nz: [0] * n_bands for nz in nz_vals}
    for r in recs:
        if r.filter_type != filter_type:
            continue
        gain = _abs_gain(r)
        num_taps_grid = np.asarray(r.num_taps_grid)
        cut = np.asarray(r.cutoff_grid)
        for i, order in enumerate(num_taps_grid):
            row = counts[int(order) - 1]
            for j, wc in enumerate(cut):
                if gain[i, j] >= LOW_GAIN:
                    row[min(int(float(wc) * n_bands), n_bands - 1)] += 1
    header = "  Nz | " + " ".join(f"{k / 10:.1f}-{(k + 1) / 10:.1f}" for k in range(n_bands))
    lines = [header]
    for nz in nz_vals:
        if sum(counts[nz]):
            lines.append(f"{nz:>4} | " + " ".join(f"{c:>7d}" for c in counts[nz]))
    out[key] = "\n".join(lines)


def section_gain(recs, out: dict) -> None:
    """DC-gain (G = sum of coefficients) composition analysis."""
    n_low = n_per_low = n_cha_low = 0
    hp_cha_total = hp_cha_low = 0
    lp_fin_chaotic = lp_fin_periodic = lp_fin_divergent = 0
    for r in recs:
        gain = _abs_gain(r)
        h = np.asarray(r.h, dtype=np.float64)
        low = gain < LOW_GAIN
        nan = np.isnan(h)
        chaotic = h > 0
        periodic = (~nan) & (h <= 0)

        n_low += int(low.sum())
        n_per_low += int((low & periodic).sum())
        n_cha_low += int((low & chaotic).sum())

        if r.filter_type == "highpass":
            hp_cha_total += int(chaotic.sum())
            hp_cha_low += int((low & chaotic).sum())

        if r.filter_type == "lowpass":
            finite = np.isfinite(gain)
            lp_fin_chaotic += int((finite & chaotic).sum())
            lp_fin_periodic += int((finite & periodic).sum())
            lp_fin_divergent += int((finite & nan).sum())

    out["GAIN.n_absG_lt_0.1"] = str(n_low)
    out["GAIN.periodic_pct_absG_lt_0.1"] = pct(n_per_low, n_low, 1)
    out["GAIN.chaotic_count_absG_lt_0.1"] = str(n_cha_low)
    out["GAIN.highpass.chaotic_absG_lt_0.1"] = str(hp_cha_low)
    out["GAIN.highpass.chaotic_total"] = str(hp_cha_total)

    lp_fin_total = lp_fin_chaotic + lp_fin_periodic + lp_fin_divergent
    out["GAIN.lowpass.chaotic_pct"] = pct(lp_fin_chaotic, lp_fin_total, 1)
    out["GAIN.lowpass.periodic_pct"] = pct(lp_fin_periodic, lp_fin_total, 1)
    out["GAIN.lowpass.divergent_pct"] = pct(lp_fin_divergent, lp_fin_total, 1)

    _gain_band_table(recs, "highpass", out, "GAIN.highpass.Nz_x_wc")
    _gain_band_table(recs, "bandpass", out, "GAIN.bandpass.Nz_x_wc")


# ── slow quantities (extras) ────────────────────────────────────────────────


def seed_runs(root: Path, best: dict) -> dict:
    """Fill the ``SEED.*`` labels by rerunning with several seeds."""
    from chaotic_pfc.analysis.sweep import run_sweep

    cache = CACHE / "seeds"
    cache.mkdir(parents=True, exist_ok=True)
    configs = {}
    for ft in TYPES:
        kb = [r for r in best[ft] if is_kaiser(r["sr"])]
        fx = [r for r in best[ft] if not is_kaiser(r["sr"])]
        table_rows = [r for r in best[ft][:2]]
        chosen = {id(r): r for r in table_rows}
        if kb:
            chosen[id(kb[0])] = kb[0]
        if fx:
            chosen[id(fx[0])] = fx[0]
        configs[ft] = list(chosen.values())

    results: dict = {}
    for ft, rows in configs.items():
        for row in rows:
            sr = row["sr"]
            key = f"{ft}.{sr.window}.{float(sr.metadata.get('kaiser_beta', 0)):g}"
            ncs = {}
            for seed in SEED_LIST:
                f = cache / f"{key}.seed{seed}.json"
                if f.exists():
                    ncs[seed] = json.loads(f.read_text())["nc"]
                    continue
                m = sr.metadata
                res = run_sweep(
                    window=sr.window,
                    filter_type=sr.filter_type,
                    kaiser_beta=float(m["kaiser_beta"]),
                    num_taps_grid=sr.num_taps_grid,
                    cutoff_grid=sr.cutoff_grid,
                    bandwidth=0.05,
                    Nitera=int(m["Nitera"]),
                    Nmap=int(m["Nmap"]),
                    n_initial=int(m["n_initial"]),
                    alpha=float(m["alpha"]),
                    beta=float(m["beta"]),
                    seed=seed,
                    adaptive=bool(m["adaptive"]),
                    Nmap_min=int(m["Nmap_min"]),
                    tol=float(m["tol"]),
                    fallback_mode=1,
                    warmup=True,
                )
                nc = int(np.count_nonzero(np.asarray(res.h) > 0))
                ncs[seed] = nc
                f.write_text(json.dumps({"nc": nc, "seed": seed}))
            results[key] = {"ft": ft, "label": row["name"], "ncs": ncs, "kaiser": is_kaiser(sr)}
    return results


def validation_ns1() -> dict:
    """Ns=1, c0=1 map with the full-spectrum estimator, early stop off.

    Mirrors the fallback path of the sweep kernel for Ns <= 2: the full
    tangent matrix (2 x 2) is propagated and reorthonormalised, and the
    returned value is the largest exponent. The sum of both exponents is
    compared with ln(beta), which is the log of |det J| per iteration.
    """
    from chaotic_pfc.analysis.sweep._kernel import _henon_n12_inplace, _lyap_online_n12
    from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations

    alpha, beta = 1.4, 0.3
    c = np.array([1.0])
    gain = 1.0
    disc = (1.0 - beta) ** 2 + 4.0 * alpha * gain**2
    p1 = 2.0 * alpha / ((1.0 - beta) + disc**0.5)
    pert = _precompute_perturbations(np.array([1]), 1, 25, 42)
    lams, residuals = [], []
    for ci in range(25):
        noise = pert[0, 0, ci, :1]
        xx = np.empty((2, 2))
        xx[0, 0] = 0.1 * noise[0] + p1
        xx[0, 1] = 0.0
        tick = 0
        for _ in range(NITERA):
            _henon_n12_inplace(xx[tick], xx[1 - tick], alpha, beta, c)
            tick = 1 - tick
        if tick == 1:
            xx[0, 0], xx[0, 1] = xx[1, 0], xx[1, 1]
        ww = np.empty((2, 2, 2))
        lyap_sum = np.empty(2)
        lam, n_used = _lyap_online_n12(
            xx.copy(), ww, lyap_sum, NMAP, NMAP, 0.0, 1, c.copy(), alpha, beta, 0
        )
        lams.append(float(lam))
        lam_sum = float((lyap_sum[0] + lyap_sum[1]) / n_used)
        residuals.append(abs(lam_sum - math.log(beta)))
    mean = float(np.mean(lams))
    return {
        "lambda1_mean": fmt(mean, 4),
        "rel_err_pct": pct_float(abs(mean - 0.4192), 0.4192, 2),
        "max_sum_residual": f"{max(residuals):.2e}",
    }


def pct_float(num: float, den: float, places: int) -> str:
    """Format ``num/den`` as a percentage string."""
    q = Decimal(1).scaleb(-places)
    return str(
        (Decimal(repr(100.0 * num)) / Decimal(repr(den))).quantize(q, rounding=ROUND_HALF_UP)
    )


def machine() -> str:
    """Return the CPU model string."""
    model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{model}, {os.cpu_count()} logical CPUs"


# ── driver ──────────────────────────────────────────────────────────────────


def compute(root: Path, extras: bool) -> dict:
    """Compute every manuscript quantity and return the label mapping."""
    recs = load_all(root)
    out: dict = {}
    section_recalc(recs, out)
    best = section_table2_table3(recs, out)
    section_spearman(recs, out)
    section_totals(recs, out)
    section_maps(best, out)
    section_density(root, out)
    section_concordance(recs, out)
    section_kaiser(recs, out)
    section_robustness(recs, best, out)
    section_cost(recs, out)
    section_gain(recs, out)
    if extras:
        seeds = seed_runs(root, best)
        for key, info in seeds.items():
            ncs = list(info["ncs"].values())
            out[f"SEED.{key}"] = (
                f"{info['label']}: Nc mean={fmt(np.mean(ncs), 2)} sd={fmt(np.std(ncs, ddof=1), 2)} "
                f"(seeds {SEED_LIST}: {ncs})"
            )
        for ft in TYPES:
            kaiser = [v for v in seeds.values() if v["ft"] == ft and v["kaiser"]]
            fixed = [v for v in seeds.values() if v["ft"] == ft and not v["kaiser"]]
            if kaiser and fixed:
                k = max(kaiser, key=lambda v: np.mean(list(v["ncs"].values())))
                f = max(fixed, key=lambda v: np.mean(list(v["ncs"].values())))
                wins = sum(k["ncs"][s] > f["ncs"][s] for s in SEED_LIST)
                out[f"SEED.kaiser_gt_fixed.{ft}"] = f"{wins}/{len(SEED_LIST)} seeds"
        out["VAL.ns1"] = validation_ns1()
        out["COST.machine"] = machine()
    wall = CACHE / "regen_seconds.txt"
    if wall.exists():
        out["COST.regen_wall_seconds"] = wall.read_text().strip()
    return out


def compare(old: dict, new: dict) -> None:
    """Print a line-by-line diff of two ``compute`` outputs."""
    keys = list(dict.fromkeys(list(new.keys()) + list(old.keys())))
    print(f"{'rótulo':<36} | antigo | novo | mudou?")
    for k in keys:
        a = old.get(k, "n/a")
        b = new.get(k, "n/a")
        if isinstance(a, dict):
            a = json.dumps(a, sort_keys=True)
        if isinstance(b, dict):
            b = json.dumps(b, sort_keys=True)
        changed = "sim" if a != b else "não"
        print(f"{k} | {a} | {b} | {changed}")


# ── threshold sensitivity (report only; does not feed the JSON) ─────────────

SENS_THRESHOLDS = (0.0, 1e-3, 5e-3)


def counts_at(sr, thr: float) -> int:
    """Return the number of points above ``thr``."""
    h = np.asarray(sr.h, dtype=np.float64)
    return int(np.count_nonzero(h > thr))


def candidates_at(recs, ft: str, thr: float) -> list:
    """Return the fixed-window rows above ``thr``."""
    fx = fixed_of(recs, ft)
    rows = [
        {"name": name, "sr": fx[w], "db": db, "nc": counts_at(fx[w], thr)} for w, name, db in FIXED
    ]
    cands = [r for r in recs if r.filter_type == ft and is_kaiser(r)]
    kb = max(cands, key=lambda r: (counts_at(r, thr), -float(r.metadata["kaiser_beta"])))
    rows.append({"name": kaiser_label(kb), "sr": kb, "db": None, "nc": counts_at(kb, thr)})
    return rows


def sensitivity_report(recs) -> str:
    """Return the threshold sensitivity report as text."""
    per: dict = {}
    for thr in SENS_THRESHOLDS:
        allrows: list = []
        t2 = {}
        sp = {}
        for ft in TYPES:
            rows = candidates_at(recs, ft, thr)
            t2[ft] = sorted(rows, key=lambda r: (-r["nc"], r["name"]))[:2]
            allrows.extend((ft, r) for r in rows)
            fixed = [r for r in rows if r["db"] is not None]
            db = np.array([r["db"] for r in fixed], dtype=np.float64)
            nc = np.array([r["nc"] for r in fixed], dtype=np.float64)
            rs, p = spearman_exact(db, nc)
            sp[ft] = (rs, p)
        order = sorted(allrows, key=lambda q: (-q[1]["nc"], q[0], q[1]["name"]))
        per[thr] = {"order": order, "t2": t2, "sp": sp}

    def tag(ft, r):
        return f"{ft} {r['name']}"

    base = per[0.0]
    lines = ["(a) Table III order (28 rows), Nc = count(h > thr)", ""]
    header = f"{'rank':>4} | {'thr=0':<52} {'Nc':>5} | {'thr=1e-3':<52} {'Nc':>5} | {'thr=5e-3':<52} {'Nc':>5}"
    lines.append(header)
    changed = {thr: 0 for thr in SENS_THRESHOLDS}
    for k in range(28):
        cells = []
        for thr in SENS_THRESHOLDS:
            ft, r = per[thr]["order"][k]
            cells.append((tag(ft, r), r["nc"]))
            if (ft, r["name"]) != (base["order"][k][0], base["order"][k][1]["name"]):
                changed[thr] += 1
        mark = "" if all(c[0] == cells[0][0] for c in cells) else " *"
        lines.append(
            f"{k + 1:>4} | {cells[0][0]:<52} {cells[0][1]:>5} | {cells[1][0]:<52} {cells[1][1]:>5} | {cells[2][0]:<52} {cells[2][1]:>5}{mark}"
        )
    lines.append("")
    lines.append(
        "rows whose position differs from thr=0: "
        + ", ".join(f"thr={thr:g}: {changed[thr]}" for thr in SENS_THRESHOLDS[1:])
    )
    lines += ["", "(b) Table II: the two lines of each type (by Nc, ties by name)", ""]
    for ft in TYPES:
        for thr in SENS_THRESHOLDS:
            top = per[thr]["t2"][ft]
            desc = " ; ".join(f"{r['name']} Nc={r['nc']}" for r in top)
            lines.append(f"{ft:<9} thr={thr:<6g} {desc}")
        lines.append("")
    lines += ["(c) Spearman r_s and exact p per type (6 fixed windows)", ""]
    for ft in TYPES:
        sp_cells = [
            f"thr={thr:g}: r_s={per[thr]['sp'][ft][0]:+.3f} p={per[thr]['sp'][ft][1]:.4f}"
            for thr in SENS_THRESHOLDS
        ]
        lines.append(f"{ft:<9} " + " | ".join(sp_cells))
    return "\n".join(lines)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``chaotic-pfc paper numbers``."""
    parser = subparsers.add_parser(
        "numbers",
        help="Recompute every manuscript number from the versioned sweeps.",
        description="Recompute every manuscript number from the versioned sweeps.",
    )
    parser.add_argument("--sweeps", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--extras", action="store_true", help="also run seeds and the Ns=1 validation"
    )
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("OLD", "NEW"))
    parser.add_argument(
        "--sensitivity", action="store_true", help="print the threshold sensitivity report"
    )
    parser.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute ``chaotic-pfc paper numbers``."""
    if args.compare:
        old = json.loads(args.compare[0].read_text())
        new = json.loads(args.compare[1].read_text())
        compare(old, new)
        return 0
    if args.sensitivity:
        if not args.sweeps:
            raise SystemExit("--sweeps is required with --sensitivity")
        print(sensitivity_report(load_all(args.sweeps)))
        return 0
    if not args.sweeps or not args.json:
        raise SystemExit("--sweeps and --json are required unless --compare is given")
    out = compute(args.sweeps, args.extras)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in out.items():
        print(f"{k} = {v}")
    return 0
