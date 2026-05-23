"""Internationalisation layer for figure titles and labels.

Usage::

    from chaotic_pfc._i18n import t

    title = t("attractor.standard", lang="pt")
    # → "Atrator de Henon Padrao (a=1.4, b=0.3)"

Languages are two-letter codes (``"pt"``, ``"en"``). The default is
``"pt"`` because the accompanying academic article is written in
Portuguese. To switch at runtime, pass ``lang`` explicitly or set
the ``CHAOTIC_PFC_LANG`` environment variable.
"""

from __future__ import annotations

import os


def _default_lang() -> str:
    return os.environ.get("CHAOTIC_PFC_LANG", "pt")


_STRINGS: dict[str, dict[str, str]] = {
    # ── Attractors ─────────────────────────────────────────────────
    "attractor.standard": {
        "pt": r"Atrator de Hénon Padrão ($a=1.4,\; b=0.3$)",
        "en": r"Standard Hénon Attractor ($a=1.4,\; b=0.3$)",
    },
    "attractor.generalised": {
        "pt": r"Atrator de Hénon Generalizado ($\alpha=1.4,\; \beta=0.3$)",
        "en": r"Generalised Hénon Attractor ($\alpha=1.4,\; \beta=0.3$)",
    },
    "attractor.filtered": {
        "pt": r"Atrator de Hénon Filtrado ($c_0=1,\; c_1=0$)",
        "en": r"Filtered Hénon Attractor ($c_0=1,\; c_1=0$)",
    },
    # ── Sensitivity ────────────────────────────────────────────────
    "sensitivity.title": {
        "pt": r"Sensibilidade às Condições Iniciais — Mapa de Hénon",
        "en": r"Sensitivity to Initial Conditions — Hénon Map",
    },
    # ── Communication grid ─────────────────────────────────────────
    "comm.ideal": {
        "pt": "Comunicação Caótica — Canal Ideal",
        "en": "Chaotic Communication — Ideal Channel",
    },
    "comm.fir": {
        "pt": "Comunicação Caótica — Canal FIR",
        "en": "Chaotic Communication — FIR Channel",
    },
    "comm.order_n": {
        "pt": "Hénon de Ordem $N$ — Canal FIR",
        "en": "Order-$N$ Hénon — FIR Channel",
    },
    "comm.time_domain": {
        "pt": r"Domínio do Tempo",
        "en": r"Time Domain",
    },
    "comm.psd": {
        "pt": r"PSD Normalizada (Welch)",
        "en": r"Normalised PSD (Welch)",
    },
    "sweep.legend.periodic": {
        "pt": "Órbitas periódicas",
        "en": "Periodic orbits",
    },
    "sweep.legend.chaotic": {
        "pt": "Órbitas caóticas",
        "en": "Chaotic orbits",
    },
    "sweep.legend.unbounded": {
        "pt": "Órbitas não-limitadas",
        "en": "Unbounded orbits",
    },
    "sweep.chaotic_map.title": {
        "pt": "Mapa de Regiões Caóticas — União entre Todas as Configurações",
        "en": "Chaotic Regions Map — Union across All Configurations",
    },
    "sweep.chaotic_density.title": {
        "pt": "Densidade de Caos — Concordância entre Configurações",
        "en": "Chaos Density — Agreement across Configurations",
    },
    "sweep.chaotic_density.cbar": {
        "pt": "Número de configurações caóticas",
        "en": "Number of chaotic configurations",
    },
    # ── Analysis tables ────────────────────────────────────────────────
    "analysis.tables.top_k.caption": {
        "pt": "Top-3 janelas por tipo de filtro (área caótica).",
        "en": "Top-3 windows per filter type (chaotic area).",
    },
    "analysis.tables.top_k_extended.caption": {
        "pt": "Top-3 janelas por tipo de filtro com estatísticas de $\\lambda_{\\max}$.",
        "en": "Top-3 windows per filter type with $\\lambda_{\\max}$ statistics.",
    },
    "analysis.tables.full_ranking.caption": {
        "pt": "Ranking completo de todas as combinações filtro $\\times$ janela.",
        "en": "Full ranking of all filter $\\times$ window combinations.",
    },
    "analysis.tables.sweet_spots.caption": {
        "pt": "Ponto de $\\lambda_{\\max}$ máximo por tipo de filtro.",
        "en": "Maximum $\\lambda_{\\max}$ point per filter type.",
    },
    "analysis.tables.col.filter": {
        "pt": "Filtro",
        "en": "Filter",
    },
    "analysis.tables.col.window": {
        "pt": "Janela",
        "en": "Window",
    },
    "analysis.tables.col.rank": {
        "pt": "Rank",
        "en": "Rank",
    },
    "analysis.tables.col.n_chaotic": {
        "pt": "Pontos caóticos",
        "en": "Chaotic points",
    },
    "analysis.tables.col.pct_chaotic": {
        "pt": "$\\%$ caótico",
        "en": "$\\%$ chaotic",
    },
    "analysis.tables.col.pct_chaotic_finite": {
        "pt": "$\\%$ caótico (finitos)",
        "en": "$\\%$ chaotic (finite)",
    },
    "analysis.tables.col.lmax_mean": {
        "pt": "$\\overline{\\lambda}_{\\max}$",
        "en": "$\\overline{\\lambda}_{\\max}$",
    },
    "analysis.tables.col.lmax_max": {
        "pt": "$\\lambda_{\\max}$ máx",
        "en": "$\\lambda_{\\max}$ max",
    },
    "analysis.tables.col.lmax_std": {
        "pt": "$\\sigma(\\lambda_{\\max})$",
        "en": "$\\sigma(\\lambda_{\\max})$",
    },
    "analysis.tables.col.lmax_ci95": {
        "pt": "IC 95\\%",
        "en": "95\\% CI",
    },
    "analysis.tables.col.n_z": {
        "pt": "$N_z$",
        "en": "$N_z$",
    },
    "analysis.tables.col.omega_c": {
        "pt": "$\\omega_c/\\pi$",
        "en": "$\\omega_c/\\pi$",
    },
    "analysis.filter.lowpass": {
        "pt": "Passa-baixa",
        "en": "Lowpass",
    },
    "analysis.filter.highpass": {
        "pt": "Passa-alta",
        "en": "Highpass",
    },
    "analysis.filter.bandpass": {
        "pt": "Passa-faixa",
        "en": "Bandpass",
    },
    "analysis.filter.bandstop": {
        "pt": "Rejeita-faixa",
        "en": "Bandstop",
    },
    "analysis.window.kaiser": {
        "pt": "Kaiser",
        "en": "Kaiser",
    },
}


def t(key: str, *, lang: str | None = None) -> str:
    """Return the translated string for *key* in the given language.

    Parameters
    ----------
    key
        Dot-separated path (e.g. ``"attractor.standard"``).
    lang
        ``"pt"`` or ``"en"``. Defaults to ``CHAOTIC_PFC_LANG`` env var
        or ``"pt"``.

    Returns
    -------
    str
        The translated string, or *key* itself if not found.
    """
    lang = lang or _default_lang()
    bucket = _STRINGS.get(key, {})
    return bucket.get(lang, key)
