#!/usr/bin/env python3
"""Per-configuration statistics read from the versioned ``.npz`` checkpoints.

Reads only the checkpoints (no recomputation). The classification is NOT
stored in the files, so it is reconstructed from ``h``: ``NaN`` is a diverged
orbit, ``h > 0`` is chaotic and ``h <= 0`` is periodic. That reconstruction is
a caveat, not a measurement, and the ``NaN`` count per configuration is
printed so the assumption is visible.

Divergence rule, read from the kernel (``_kernel.py``): a cell becomes ``NaN``
when at least one of the 25 initial conditions diverges during the ``Nitera``
burn-in (an OR over the ensemble, :603-608 / :626-631). An IC that diverges
later, inside the ``Nmap`` Lyapunov window, is dropped and the cell value is
the mean over the remaining finite ICs (:290-293, :330-335).

Usage::

    python tools/report_sweep_stats.py --sweep-dir data/sweeps \
        --tex articles/jcis/article.tex --csv /tmp/sweep_stats.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from chaotic_pfc.paper import audit as audit_paper

CSV_FIELDS = [
    "filter_type",
    "window",
    "kaiser_beta",
    "N_grid",
    "N_chaotic",
    "N_periodic",
    "N_divergent",
    "N_bounded",
    "rho_fin_pct",
    "rho_abs_pct",
    "n_fallback",
    "n_fallback_frac",
    "niters_min",
    "niters_q1",
    "niters_median",
    "niters_q3",
    "niters_max",
    "niters_at_cap_frac",
]


def _metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    arr = data["metadata"]
    if arr.dtype == object and getattr(arr, "ndim", 0) == 2:
        return {str(row[0]): row[1] for row in arr}
    return {}


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def load_configs(sweep_dir: Path) -> list[dict[str, Any]]:
    """Load the per-configuration statistics from the checkpoints."""
    configs: list[dict[str, Any]] = []
    for path in sorted(sweep_dir.rglob("variables_lyapunov.npz")):
        data = np.load(path, allow_pickle=True)
        md = _metadata(data)
        h = np.asarray(data["h"], dtype=float)
        finite = np.isfinite(h)
        chaotic = finite & (h > 0.0)
        periodic = finite & (h <= 0.0)
        divergent = ~finite
        n_grid = int(h.size)
        n_chaotic = int(np.count_nonzero(chaotic))
        n_periodic = int(np.count_nonzero(periodic))
        n_divergent = int(np.count_nonzero(divergent))
        n_bounded = n_grid - n_divergent
        n_fallback = int(md.get("n_fallback", 0))
        niters = np.asarray(data["n_iters_used"], dtype=float)[finite]
        cap = float(md.get("Nmap", 0) or 0)
        at_cap = (
            float(np.count_nonzero(niters >= cap) / niters.size) if niters.size and cap else 0.0
        )
        configs.append(
            {
                "path": str(path),
                "filter_type": str(data["filter_type"]),
                "window": str(data["window"]),
                "kaiser_beta": md.get("kaiser_beta"),
                "N_grid": n_grid,
                "N_chaotic": n_chaotic,
                "N_periodic": n_periodic,
                "N_divergent": n_divergent,
                "N_bounded": n_bounded,
                "rho_fin_pct": 100.0 * n_chaotic / n_bounded if n_bounded else float("nan"),
                "rho_abs_pct": 100.0 * n_chaotic / n_grid if n_grid else float("nan"),
                "n_fallback": n_fallback,
                "n_fallback_frac": n_fallback / n_grid if n_grid else float("nan"),
                "niters_min": _percentile(niters, 0),
                "niters_q1": _percentile(niters, 25),
                "niters_median": _percentile(niters, 50),
                "niters_q3": _percentile(niters, 75),
                "niters_max": _percentile(niters, 100),
                "niters_at_cap_frac": at_cap,
            }
        )
    return configs


def _sort_key(config: dict[str, Any]) -> Any:
    beta = config["kaiser_beta"]
    return (config["filter_type"], config["window"], -1.0 if beta is None else float(beta))


def print_table(configs: list[dict[str, Any]]) -> None:
    """Print the per-configuration statistics table."""
    header = (
        f"{'filter':<9} {'window':<15} {'beta':>5} {'N_grid':>6} {'N_chaotic':>9} "
        f"{'N_periodic':>10} {'N_divergent':>11} {'rho_fin%':>8} {'rho_abs%':>8} "
        f"{'n_fb':>5} {'fb_frac':>8}"
    )
    print(header)
    print("-" * len(header))
    for cfg in configs:
        beta = "-" if cfg["kaiser_beta"] is None else f"{float(cfg['kaiser_beta']):.1f}"
        print(
            f"{cfg['filter_type']:<9} {cfg['window']:<15} {beta:>5} {cfg['N_grid']:>6} "
            f"{cfg['N_chaotic']:>9} {cfg['N_periodic']:>10} {cfg['N_divergent']:>11} "
            f"{cfg['rho_fin_pct']:>8.2f} {cfg['rho_abs_pct']:>8.2f} "
            f"{cfg['n_fallback']:>5} {cfg['n_fallback_frac']:>8.4f}"
        )


def write_csv(configs: list[dict[str, Any]], path: Path) -> None:
    """Write the statistics table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for cfg in configs:
            writer.writerow({key: cfg.get(key) for key in CSV_FIELDS})


def _article_rows(tex_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    clean = audit_paper._strip_comments(text)
    best_tab = audit_paper._extract_tabular(clean, "tab:best")
    rank_tab = audit_paper._extract_tabular(clean, "tab:ranking")
    best = audit_paper._parse_best_rows(best_tab)[0] if best_tab else []
    ranking = audit_paper._parse_ranking_rows(rank_tab)[0] if rank_tab else []
    return best, ranking


def _norm_window(name: str) -> str:
    key = name.lower().replace("-", "").replace(" ", "")
    return "boxcar" if key == "rectangular" else key


def _match_config(row: dict[str, Any], configs: list[dict[str, Any]]) -> dict[str, Any] | None:
    filt = row["filter"]
    window = row["window"]
    if window.startswith("Kaiser"):
        digits = "".join(ch if ch.isdigit() or ch == "." else " " for ch in window).split()
        beta = float(digits[0]) if digits else None
        for cfg in configs:
            if (
                cfg["filter_type"] == filt
                and cfg["window"] == "kaiser"
                and beta is not None
                and cfg["kaiser_beta"] is not None
                and abs(float(cfg["kaiser_beta"]) - beta) < 1e-6
            ):
                return cfg
        return None
    for cfg in configs:
        if cfg["filter_type"] == filt and _norm_window(cfg["window"]) == _norm_window(window):
            return cfg
    return None


def print_by_filter(configs: list[dict[str, Any]]) -> None:
    """Print the per-filter-type aggregation."""
    print("\naggregate divergent fraction by filter type (G ~ 1 vs G ~ 0 groups):")
    header = (
        f"{'filter':<9} {'N_grid':>7} {'N_chaotic':>9} {'N_periodic':>10} "
        f"{'N_divergent':>11} {'div_frac%':>9} {'rho_fin%':>8} {'rho_abs%':>8}"
    )
    print(header)
    print("-" * len(header))
    for filter_type in ("lowpass", "highpass", "bandpass", "bandstop"):
        subset = [cfg for cfg in configs if cfg["filter_type"] == filter_type]
        n_grid = sum(cfg["N_grid"] for cfg in subset)
        n_chaotic = sum(cfg["N_chaotic"] for cfg in subset)
        n_periodic = sum(cfg["N_periodic"] for cfg in subset)
        n_divergent = sum(cfg["N_divergent"] for cfg in subset)
        n_bounded = n_grid - n_divergent
        div_frac = 100.0 * n_divergent / n_grid if n_grid else float("nan")
        rho_fin = 100.0 * n_chaotic / n_bounded if n_bounded else float("nan")
        rho_abs = 100.0 * n_chaotic / n_grid if n_grid else float("nan")
        print(
            f"{filter_type:<9} {n_grid:>7} {n_chaotic:>9} {n_periodic:>10} "
            f"{n_divergent:>11} {div_frac:>9.2f} {rho_fin:>8.2f} {rho_abs:>8.2f}"
        )


def print_iters_distribution(configs: list[dict[str, Any]]) -> None:
    """Print the distribution of iterations used."""
    print("\nn_iters_used distribution per configuration (sorted by fraction at the Nmap cap):")
    header = (
        f"{'filter':<9} {'window':<15} {'beta':>5} {'min':>6} {'q1':>6} {'med':>6} "
        f"{'q3':>6} {'max':>6} {'at_cap%':>8}"
    )
    print(header)
    print("-" * len(header))
    for cfg in sorted(configs, key=lambda item: -item["niters_at_cap_frac"]):
        beta = "-" if cfg["kaiser_beta"] is None else f"{float(cfg['kaiser_beta']):.1f}"
        print(
            f"{cfg['filter_type']:<9} {cfg['window']:<15} {beta:>5} {cfg['niters_min']:>6.0f} "
            f"{cfg['niters_q1']:>6.0f} {cfg['niters_median']:>6.0f} {cfg['niters_q3']:>6.0f} "
            f"{cfg['niters_max']:>6.0f} {100.0 * cfg['niters_at_cap_frac']:>8.2f}"
        )


ATTENUATION_DB = {
    "boxcar": 21,
    "rectangular": 21,
    "bartlett": 25,
    "hamming": 41,
    "hann": 44,
    "blackman": 74,
    "blackmanharris": 92,
}


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def _exact_spearman_p(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Exact two-sided Spearman p by enumerating every pairing (n! permutations)."""
    from itertools import permutations

    observed = abs(_spearman_rho(x, y))
    total = 0
    extreme = 0
    for order in permutations(range(len(y))):
        total += 1
        if abs(_spearman_rho(x, np.asarray(y)[list(order)])) >= observed - 1e-12:
            extreme += 1
    return extreme / total, total


def report_attenuation_correlation(configs: list[dict[str, Any]]) -> None:
    """Print the attenuation/exponent correlation."""
    print(
        "\nSpearman: Table I stopband attenuation versus measured coverage, "
        "fixed windows only (Kaiser excluded)."
    )
    print(
        "  8 tests were run (4 filter types x 2 metrics); p-values are exact two-sided "
        "permutation over all pairings (minimum attainable p = 2/720 = 0.0028)."
    )
    tests = 0
    for filter_type in ("lowpass", "highpass", "bandpass", "bandstop"):
        subset = [
            cfg
            for cfg in configs
            if cfg["filter_type"] == filter_type
            and cfg["window"] != "kaiser"
            and _norm_window(cfg["window"]) in ATTENUATION_DB
        ]
        if not subset:
            continue
        attenuation = np.array(
            [ATTENUATION_DB[_norm_window(c["window"])] for c in subset], dtype=float
        )
        n_chaotic = np.array([c["N_chaotic"] for c in subset], dtype=float)
        rho_fin = np.array([c["rho_fin_pct"] for c in subset], dtype=float)
        order = np.argsort(attenuation)
        pairs = ", ".join(f"({int(attenuation[k])},{int(n_chaotic[k])})" for k in order)
        print(f"\n  {filter_type}: {len(subset)} windows")
        print(f"    raw (attenuation dB, N_chaotic): {pairs}")
        for label, values in (("N_chaotic", n_chaotic), ("rho_fin", rho_fin)):
            rho = _spearman_rho(attenuation, values)
            p_value, permutations = _exact_spearman_p(attenuation, values)
            tests += 1
            print(
                f"    {label:<10} rho={rho:+.4f} p_exact={p_value:.4f} "
                f"n={len(subset)} permutations={permutations}"
            )
    print(f"  total tests: {tests} (note the multiplicity in the caption)")


def cross_check(tex_path: Path, configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-check the tables against the checkpoints."""
    best, ranking = _article_rows(tex_path)
    issues: list[str] = []
    matched = {"tab:best": 0, "tab:ranking": 0}
    for label, rows in (("tab:best", best), ("tab:ranking", ranking)):
        for row in rows:
            cfg = _match_config(row, configs)
            if cfg is None:
                issues.append(
                    f"ERROR {label}: no checkpoint matched {row['filter']}/{row['window']}"
                )
                continue
            matched[label] += 1
            if cfg["N_chaotic"] != int(row["nc"]):
                issues.append(
                    f"ERROR {label} {row['filter']}/{row['window']}: N_chaotic "
                    f"article={int(row['nc'])} checkpoint={cfg['N_chaotic']}"
                )
            if abs(cfg["rho_fin_pct"] - row["rho_fin"]) > 0.05:
                issues.append(
                    f"ERROR {label} {row['filter']}/{row['window']}: rho_fin "
                    f"article={row['rho_fin']:.2f} checkpoint={cfg['rho_fin_pct']:.2f}"
                )
    return {
        "matched_best": matched["tab:best"],
        "total_best": len(best),
        "matched_rank": matched["tab:ranking"],
        "total_rank": len(ranking),
        "issues": issues,
    }


def add_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    """Register ``chaotic-pfc sweeps stats``."""
    parser = subparsers.add_parser(
        "stats",
        help="Per-configuration statistics read from the .npz checkpoints.",
        description="Per-configuration statistics read from the .npz checkpoints.",
    )
    parser.add_argument("--sweep-dir", default="data/sweeps")
    parser.add_argument("--tex", default="articles/jcis/article.tex")
    parser.add_argument(
        "--csv", required=True, help="CSV output path (keep it out of version control)."
    )
    parser.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute ``chaotic-pfc sweeps stats``."""
    configs = load_configs(Path(args.sweep_dir))
    configs.sort(key=_sort_key)
    print_table(configs)
    write_csv(configs, Path(args.csv))
    print(f"\n{len(configs)} configurations; CSV written to {args.csv}")

    total_nan = sum(cfg["N_divergent"] for cfg in configs)
    total_fb = sum(cfg["n_fallback"] for cfg in configs)
    total_grid = sum(cfg["N_grid"] for cfg in configs)
    print(f"total NaN cells: {total_nan} / {total_grid} ({100.0 * total_nan / total_grid:.2f}%)")
    print(
        f"total n_fallback cells: {total_fb} / {total_grid} ({100.0 * total_fb / total_grid:.4f}%)"
    )

    print_by_filter(configs)

    print_iters_distribution(configs)
    report_attenuation_correlation(configs)

    coverage = cross_check(Path(args.tex), configs)
    print("\ncross-check against Tables II and III (article vs checkpoint):")
    print(
        f"  matched {coverage['matched_best']}/{coverage['total_best']} rows from tab:best, "
        f"{coverage['matched_rank']}/{coverage['total_rank']} rows from tab:ranking"
    )
    if coverage["issues"]:
        for issue in coverage["issues"]:
            print(f"  {issue}")
        print("  cross-check FAILED")
        return 1
    print("  no divergence on N_chaotic and rho_fin for the matched rows.")
    return 0
