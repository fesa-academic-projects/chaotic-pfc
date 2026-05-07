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
    # ── DCSK ───────────────────────────────────────────────────────
    "dcsk.pecora_carroll": {
        "pt": "Pecora-Carroll (sincronização)",
        "en": "Pecora-Carroll (synchronisation)",
    },
    "dcsk.classic": {
        "pt": "DCSK clássico",
        "en": "Classical DCSK",
    },
    "dcsk.efficient": {
        "pt": "EF-DCSK (eficiente)",
        "en": "EF-DCSK (efficient)",
    },
    "dcsk.ber_1pct": {
        "pt": "BER = 1%",
        "en": "BER = 1%",
    },
    "dcsk.ber_50pct": {
        "pt": r"BER = 50% (colapso)",
        "en": r"BER = 50% (collapse)",
    },
    "dcsk.comparison_title": {
        "pt": "Comparação de Esquemas de Comunicação Caótica",
        "en": "Comparison of Chaotic Communication Schemes",
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
