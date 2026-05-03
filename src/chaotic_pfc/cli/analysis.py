"""Statistical analysis of sweep data."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run analysis`` subcommand."""
    p = subparsers.add_parser(
        "analysis",
        help="Statistical analysis of Lyapunov sweep results.",
        description="Summarise and rank sweep data: best filters, optimal parameters, β curves.",
    )
    p.add_argument("--data-dir", default="data/sweeps", help="Root sweep directory (default: data/sweeps)")
    p.add_argument("--report", action="store_true", help="Generate comprehensive statistical report")
    p.add_argument("--json", default="analysis_summary.json", help="Output JSON path (default: analysis_summary.json)")
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
    from chaotic_pfc.analysis import (
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
    print("  ANÁLISE ESTATÍSTICA — Varredura Lyapunov (Hénon FIR)")
    print("=" * 68)

    # ── 1. Summary table ───────────────────────────────────────────────
    path = export_summary_json(data_dir, args.json)
    print(f"  Tabela completa  →  {path}")

    # ── 2. Filter type comparison ──────────────────────────────────────
    _hr("COMPARAÇÃO POR TIPO DE FILTRO")
    print(f"  {'Filtro':<12} {'%Caótico':>8} {'%Periódico':>10} {'%Divergente':>10}  {'λ_max méd.':>10}  {'λ_max máx.':>10}")
    print("  " + "-" * 62)
    from chaotic_pfc.analysis import summary_table

    agg: dict[str, list[dict]] = {}
    for row in summary_table(data_dir):
        agg.setdefault(row["filter_type"], []).append(row)

    for ft in ("lowpass", "highpass", "bandpass", "bandstop"):
        entries = agg.get(ft, [])
        if not entries:
            continue
        pct_c = round(float(sum(e["pct_chaotic"] for e in entries) / len(entries)), 1)
        pct_p = round(float(sum(e["pct_periodic"] for e in entries) / len(entries)), 1)
        pct_d = round(float(sum(e["pct_divergent"] for e in entries) / len(entries)), 1)
        mn_l = round(float(sum(e["mean_lmax"] for e in entries) / len(entries)), 4)
        mx_l = round(float(max(e["max_lmax"] for e in entries)), 4)
        print(f"  {ft:<12} {pct_c:>7.1f}% {pct_p:>9.1f}% {pct_d:>9.1f}%  {mn_l:>10.4f}  {mx_l:>10.4f}")

    # ── 3. Distribution ────────────────────────────────────────────────
    _hr("DISTRIBUIÇÃO DE λ_max POR FILTRO")
    dist = lmax_distribution(data_dir)
    for ft in ("lowpass", "highpass", "bandpass", "bandstop"):
        d = dist.get(ft, {})
        if not d:
            continue
        print(
            f"  {ft:<12}  n={d['n']:>7,}  μ={d['mean']:>8.4f}  σ={d['std']:>7.4f}  "
            f"assimetria={d['skewness']:>7.4f}"
        )

    # ── 4. Transition boundaries ───────────────────────────────────────
    import numpy as np

    _hr("FRONTEIRA DE TRANSIÇÃO (1º cutoff caótico por ordem)")
    for ft in ("lowpass", "highpass", "bandpass", "bandstop"):
        orders, cutoffs = transition_boundary(data_dir, filter_type=ft)
        if len(orders) == 0:
            print(f"  {ft:<12}  sem dados")
            continue
        n_chaotic = np.sum(~np.isnan(cutoffs))
        print(f"  {ft:<12}  {n_chaotic}/{len(orders)} ordens com transição caótica")
        if n_chaotic > 0:
            valid = cutoffs[~np.isnan(cutoffs)]
            print(f"             cutoff transição: {valid[0]:.4f}–{valid[-1]:.4f}  (mediana: {np.median(valid):.4f})")

    # ── 5. Spectral robustness ─────────────────────────────────────────
    _hr("ROBUSTEZ ESPECTRAL (largura da região caótica)")
    for ft in ("lowpass", "highpass", "bandpass", "bandstop"):
        orders, widths = chaos_margin(data_dir, filter_type=ft)
        if len(orders) == 0:
            print(f"  {ft:<12}  sem dados")
            continue
        nonzero = widths[widths > 0]
        pct = 100 * len(nonzero) / len(orders)
        print(
            f"  {ft:<12}  {len(nonzero)}/{len(orders)} ordens ({pct:.0f}%) com caos  "
            f"largura média={np.mean(widths):.4f}  máx={np.max(widths):.4f}"
        )

    # ── 6. Correlation ─────────────────────────────────────────────────
    _hr("CORRELAÇÃO DE SPEARMAN")
    corr = correlation_matrix(data_dir)
    print(f"  n = {corr['n']:,} pontos finitos")
    print(f"  ρ(ordem,  λ_max) = {corr['order_vs_lmax']:+.4f}")
    print(f"  ρ(cutoff, λ_max) = {corr['cutoff_vs_lmax']:+.4f}")

    # ── 7. Bootstrap CI ────────────────────────────────────────────────
    _hr("BOOTSTRAP 95% CI (λ_max médio)")
    ci = bootstrap_confidence(data_dir)
    print(f"  {'Filtro':<12} {'Média':>8} {'IC 2.5%':>9} {'IC 97.5%':>9} {'n':>7}")
    print("  " + "-" * 48)
    for ft in ("lowpass", "highpass", "bandpass", "bandstop"):
        d = ci.get(ft, {})
        if not d:
            continue
        print(f"  {ft:<12} {d['mean']:>8.4f} {d['ci_low']:>9.4f} {d['ci_high']:>9.4f} {d['n']:>7,}")

    # ── 8. Best and optimal ────────────────────────────────────────────
    _hr("TOP 5 — MAIOR % CAÓTICO")
    for rank, row in enumerate(best_chaos_preserving(data_dir, top_n=5), start=1):
        print(f"  {rank}. {row['window']:<15} / {row['filter_type']:<10}  {row['pct_chaotic']:>5.1f}%  λ_med={row['mean_lmax']:.4f}")

    _hr("PARÂMETROS ÓTIMOS (maior λ_max finito)")
    for rank, row in enumerate(optimal_parameters(data_dir, top_n=5), start=1):
        print(f"  {rank}. {row['window']:<15} / {row['filter_type']:<10}  ordem={row['order']:>3}  ωc={row['cutoff']:.4f}  λ_max={row['lmax']:.6f}")

    # ── 9. Beta sweep ──────────────────────────────────────────────────
    kaiser_dir = data_dir / "kaiser"
    if kaiser_dir.is_dir():
        _hr("KAISER β-SWEEP — evolução com β")
        from chaotic_pfc.analysis import beta_summary

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
                f"{first:>5.1f}% → {last:>5.1f}%  ({arrow} {abs(last-first):.1f}pp)"
            )

    # ── 10. Interpretação ──────────────────────────────────────────────
    _hr("INTERPRETAÇÃO")
    cmp = compare_filter_types(data_dir)
    best_ft = max((ft for ft in cmp if cmp[ft]), key=lambda ft: cmp[ft]["mean_pct_chaotic"])
    worst_ft = min((ft for ft in cmp if cmp[ft]), key=lambda ft: cmp[ft]["mean_pct_chaotic"])

    print(f"  • {best_ft} é o filtro que mais preserva caos ({cmp[best_ft]['mean_pct_chaotic']}% da grade)")
    print(f"  • {worst_ft} é o que mais suprime ({cmp[worst_ft]['mean_pct_chaotic']}% caótico)")
    print(f"  • Correlação ordem × λ_max: {corr['order_vs_lmax']:+.4f} (ordens maiores {'aumentam' if corr['order_vs_lmax'] > 0 else 'reduzem'} λ_max)")
    print(f"  • Correlação cutoff × λ_max: {corr['cutoff_vs_lmax']:+.4f} (cutoffs maiores {'aumentam' if corr['cutoff_vs_lmax'] > 0 else 'reduzem'} λ_max)")
    print("  • Assimetria positiva nas distribuições → caudas longas de λ_max alto (caos forte em ilhas)")

    print(f"\n{'=' * 68}")
    print("  Análise concluída.")
    return 0
