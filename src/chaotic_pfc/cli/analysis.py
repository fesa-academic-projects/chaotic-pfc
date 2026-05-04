"""Statistical analysis of sweep data."""

from __future__ import annotations

import argparse
from pathlib import Path

from chaotic_pfc.analysis.sweep import FILTER_TYPES


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run analysis`` subcommand."""
    p = subparsers.add_parser(
        "analysis",
        help="Statistical analysis of Lyapunov sweep results.",
        description="Summarise and rank sweep data: best filters, optimal parameters, beta curves.",
    )
    p.add_argument(
        "--data-dir", default="data/sweeps", help="Root sweep directory (default: data/sweeps)"
    )
    p.add_argument(
        "--report", action="store_true", help="Generate comprehensive statistical report"
    )
    p.add_argument(
        "--json",
        default="analysis_summary.json",
        help="Output JSON path (default: analysis_summary.json)",
    )
    p.set_defaults(_run=run)


def _hr(title: str = "", width: int = 68) -> None:
    if title:
        print(f"\n{'─' * width}")
        print(f"  {title}")
        print("─" * width)
    else:
        print("─" * width)


def run(args: argparse.Namespace) -> int:
    """Execute the ``analysis`` experiment."""
    from chaotic_pfc.analysis.stats import (
        best_chaos_preserving,
        bootstrap_confidence,
        chaos_margin,
        compare_filter_types,
        correlation_matrix,
        export_summary_json,
        lmax_distribution,
        optimal_parameters,
        transition_boundary,
    )

    data_dir = Path(args.data_dir)

    print("=" * 68)
    print("  STATISTICAL ANALYSIS — Lyapunov Sweep (Henon FIR)")
    print("=" * 68)

    # ── 1. Summary table ───────────────────────────────────────────────
    path = export_summary_json(data_dir, args.json)
    print(f"  Full table  →  {path}")

    # ── 2. Filter type comparison ──────────────────────────────────────
    _hr("FILTER TYPE COMPARISON")
    print(
        f"  {'Filter':<12} {'%Chaotic':>8} {'%Periodic':>10} {'%Divergent':>10}  {'λ_max mean':>10}  {'λ_max max':>10}"
    )
    print("  " + "-" * 62)
    from chaotic_pfc.analysis.stats import summary_table

    agg: dict[str, list[dict]] = {}
    for row in summary_table(data_dir):
        agg.setdefault(row["filter_type"], []).append(row)

    for ft in FILTER_TYPES:
        entries = agg.get(ft, [])
        if not entries:
            continue
        pct_c = round(float(sum(e["pct_chaotic"] for e in entries) / len(entries)), 1)
        pct_p = round(float(sum(e["pct_periodic"] for e in entries) / len(entries)), 1)
        pct_d = round(float(sum(e["pct_divergent"] for e in entries) / len(entries)), 1)
        mn_l = round(float(sum(e["mean_lmax"] for e in entries) / len(entries)), 4)
        mx_l = round(float(max(e["max_lmax"] for e in entries)), 4)
        print(
            f"  {ft:<12} {pct_c:>7.1f}% {pct_p:>9.1f}% {pct_d:>9.1f}%  {mn_l:>10.4f}  {mx_l:>10.4f}"
        )

    # ── 3. Distribution ────────────────────────────────────────────────
    _hr("λ_max DISTRIBUTION BY FILTER")
    dist = lmax_distribution(data_dir)
    for ft in FILTER_TYPES:
        d = dist.get(ft, {})
        if not d:
            continue
        print(
            f"  {ft:<12}  n={d['n']:>7,}  μ={d['mean']:>8.4f}  σ={d['std']:>7.4f}  "
            f"skewness={d['skewness']:>7.4f}"
        )

    # ── 4. Transition boundaries ───────────────────────────────────────
    import numpy as np

    _hr("TRANSITION BOUNDARY (1st chaotic cutoff per order)")
    for ft in FILTER_TYPES:
        orders, cutoffs = transition_boundary(data_dir, filter_type=ft)
        if len(orders) == 0:
            print(f"  {ft:<12}  no data")
            continue
        n_chaotic = np.sum(~np.isnan(cutoffs))
        print(f"  {ft:<12}  {n_chaotic}/{len(orders)} orders with chaotic transition")
        if n_chaotic > 0:
            valid = cutoffs[~np.isnan(cutoffs)]
            print(
                f"             cutoff transition: {valid[0]:.4f}–{valid[-1]:.4f}  (median: {np.median(valid):.4f})"
            )

    # ── 5. Spectral robustness ─────────────────────────────────────────
    _hr("SPECTRAL ROBUSTNESS (chaotic region width)")
    for ft in FILTER_TYPES:
        orders, widths = chaos_margin(data_dir, filter_type=ft)
        if len(orders) == 0:
            print(f"  {ft:<12}  no data")
            continue
        nonzero = widths[widths > 0]
        pct = 100 * len(nonzero) / len(orders)
        print(
            f"  {ft:<12}  {len(nonzero)}/{len(orders)} orders ({pct:.0f}%) with chaos  "
            f"mean width={np.mean(widths):.4f}  max={np.max(widths):.4f}"
        )

    # ── 6. Correlation ─────────────────────────────────────────────────
    _hr("SPEARMAN CORRELATION")
    corr = correlation_matrix(data_dir)
    print(f"  n = {corr['n']:,} finite points")
    print(f"  ρ(order,  λ_max) = {corr['order_vs_lmax']:+.4f}")
    print(f"  ρ(cutoff, λ_max) = {corr['cutoff_vs_lmax']:+.4f}")

    # ── 7. Bootstrap CI ────────────────────────────────────────────────
    _hr("BOOTSTRAP 95% CI (λ_max mean)")
    ci = bootstrap_confidence(data_dir)
    print(f"  {'Filter':<12} {'Mean':>8} {'CI 2.5%':>9} {'CI 97.5%':>9} {'n':>7}")
    print("  " + "-" * 48)
    for ft in FILTER_TYPES:
        d = ci.get(ft, {})
        if not d:
            continue
        print(f"  {ft:<12} {d['mean']:>8.4f} {d['ci_low']:>9.4f} {d['ci_high']:>9.4f} {d['n']:>7,}")

    # ── 8. Best and optimal ────────────────────────────────────────────
    _hr("TOP 5 — HIGHEST % CHAOTIC")
    for rank, row in enumerate(best_chaos_preserving(data_dir, top_n=5), start=1):
        print(
            f"  {rank}. {row['window']:<15} / {row['filter_type']:<10}  {row['pct_chaotic']:>5.1f}%  λ_mean={row['mean_lmax']:.4f}"
        )

    _hr("OPTIMAL PARAMETERS (highest finite λ_max)")
    for rank, row in enumerate(optimal_parameters(data_dir, top_n=5), start=1):
        print(
            f"  {rank}. {row['window']:<15} / {row['filter_type']:<10}  order={row['order']:>3}  ωc={row['cutoff']:.4f}  λ_max={row['lmax']:.6f}"
        )

    # ── 9. Beta sweep ──────────────────────────────────────────────────
    kaiser_dir = data_dir / "kaiser"
    if kaiser_dir.is_dir():
        _hr("KAISER β-SWEEP — evolution with β")
        from chaotic_pfc.analysis.stats import beta_summary

        bs = beta_summary(kaiser_dir)
        for ft in sorted(bs):
            betas = sorted(bs[ft])
            if len(betas) < 2:
                continue
            first = bs[ft][betas[0]]["pct_chaotic"]
            last = bs[ft][betas[-1]]["pct_chaotic"]
            arrow = "↑" if last > first else "↓"
            print(
                f"  {ft:<12}  β={betas[0]:.1f} → {betas[-1]:.1f}  "
                f"{first:>5.1f}% → {last:>5.1f}%  ({arrow} {abs(last - first):.1f}pp)"
            )

    # ── 10. Interpretation ─────────────────────────────────────────────
    _hr("INTERPRETATION")
    cmp = compare_filter_types(data_dir)
    best_ft = max((ft for ft in cmp if cmp[ft]), key=lambda ft: cmp[ft]["mean_pct_chaotic"])
    worst_ft = min((ft for ft in cmp if cmp[ft]), key=lambda ft: cmp[ft]["mean_pct_chaotic"])

    print(f"  • {best_ft} best preserves chaos ({cmp[best_ft]['mean_pct_chaotic']}% of grid)")
    print(f"  • {worst_ft} most suppresses ({cmp[worst_ft]['mean_pct_chaotic']}% chaotic)")
    print(
        f"  • order × λ_max correlation: {corr['order_vs_lmax']:+.4f} (higher orders {'increase' if corr['order_vs_lmax'] > 0 else 'decrease'} λ_max)"
    )
    print(
        f"  • cutoff × λ_max correlation: {corr['cutoff_vs_lmax']:+.4f} (higher cutoffs {'increase' if corr['cutoff_vs_lmax'] > 0 else 'decrease'} λ_max)"
    )
    print("  • Positive skew in distributions → long tails of high λ_max (strong chaos in islands)")

    print(f"\n{'=' * 68}")
    print("  Analysis complete.")
    return 0
