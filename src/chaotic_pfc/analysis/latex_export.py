"""LaTeX table export for chaotic-PFC analysis results.

Every exporter accepts a ``lang`` parameter (``"pt"`` or ``"en"``)
that drives column headers, captions, and filter names via
:func:`~chaotic_pfc._i18n.t`.

All tables use ``booktabs`` rules and are wrapped in a ``table``
float (``longtable`` for the full ranking). Output encoding is
UTF-8.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from chaotic_pfc._i18n import t

if TYPE_CHECKING:
    from .stats import ConfigRank, KaiserBetaOptimal, SweetSpot


def _fmt(v: float | int, decimals: int = 3) -> str:
    """Format a number with *decimals* places, strip trailing zeros."""
    if isinstance(v, int) or v == int(v):
        return str(int(v))
    s = f"{v:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _ci_str(low: float | None, high: float | None) -> str:
    """CI as ``[low, high]`` or ``---`` if None."""
    if low is None or high is None:
        return "---"
    return f"[{_fmt(low, 4)}, {_fmt(high, 4)}]"


def _window_label(entry: ConfigRank | SweetSpot, lang: str) -> str:
    """Format a window name, handling Kaiser with beta."""
    win = entry["window"]
    if win.startswith("kaiser_"):
        beta_str = win[len("kaiser_") :]
        base = t("analysis.window.kaiser", lang=lang)
        return f"{base}($\\beta={beta_str}$)"
    from .sweep import WINDOW_DISPLAY_NAMES

    return WINDOW_DISPLAY_NAMES.get(win, win.capitalize())


def _filter_label(ft: str, lang: str) -> str:
    """Filter type display name via i18n."""
    return t(f"analysis.filter.{ft}", lang=lang)


def _render_tex(
    columns: list[str],
    rows: list[list[str]],
    caption: str,
    label: str | None,
    *,
    use_longtable: bool = False,
) -> str:
    """Build a complete LaTeX table string.

    Parameters
    ----------
    columns
        Header row cell contents.
    rows
        Data rows (each a list of strings).
    caption
        Caption text.
    label
        Optional ``\\label{...}`` value.
    use_longtable
        If True, uses ``longtable`` instead of ``tabular`` + ``table`` float.
    """
    n_cols = len(columns)
    col_spec = "l" + "r" * (n_cols - 1)

    lines: list[str] = []
    if not use_longtable:
        lines.append("\\begin{table}[htbp]")
        lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    if not use_longtable:
        lines.append("  \\footnotesize")
        lines.append("  \\setlength{\\tabcolsep}{4pt}")
        lines.append("  \\resizebox{\\textwidth}{!}{%")
    lines.append("  \\begin{longtable}" if use_longtable else f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")
    header = " & ".join(f"{c}" for c in columns) + " \\\\"
    lines.append(f"    {header}")
    lines.append("    \\midrule")
    for row in rows:
        row_str = " & ".join(str(cell) for cell in row) + " \\\\"
        lines.append(f"    {row_str}")
    lines.append("    \\bottomrule")
    if use_longtable:
        lines.append("  \\end{longtable}")
    else:
        lines.append("  \\end{tabular}")
        lines.append("  }")  # close \resizebox
        lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def export_top_k_table(
    top_k_data: dict[str, list[ConfigRank]],
    output_path: str | Path,
    caption_key: str | None = None,
    label: str | None = None,
    lang: str | None = None,
) -> Path:
    """Export Categoria A — Top-3 per filter (chaotic area only).

    Parameters
    ----------
    top_k_data
        ``{filter_type: [ConfigRank, ...]}`` from :func:`~.stats.top_k_per_filter`.
    output_path
        Destination ``.tex`` file.
    caption_key
        i18n key for the caption. Defaults to ``analysis.tables.top_k.caption``.
    label
        LaTeX label (without ``\\label`` wrapper).
    lang
        ``"pt"`` or ``"en"``. Defaults to the active locale.

    Returns
    -------
    Path
        The path written.
    """
    if caption_key is None:
        caption_key = "analysis.tables.top_k.caption"
    caption = t(caption_key, lang=lang)

    columns = [
        t("analysis.tables.col.filter", lang=lang),
        t("analysis.tables.col.rank", lang=lang),
        t("analysis.tables.col.window", lang=lang),
        t("analysis.tables.col.n_chaotic", lang=lang),
        t("analysis.tables.col.pct_chaotic", lang=lang),
    ]

    rows: list[list[str]] = []
    for ft in ["lowpass", "highpass", "bandpass", "bandstop"]:
        entries = top_k_data.get(ft, [])
        for entry in entries:
            rows.append(
                [
                    _filter_label(ft, lang or "pt"),
                    str(entry["rank"]),
                    _window_label(entry, lang or "pt"),
                    str(entry["n_chaotic"]),
                    f"{entry['pct_chaotic']:.1f}\\%",
                ]
            )

    tex = _render_tex(columns, rows, caption, label)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex, encoding="utf-8")
    return output


def export_extended_top_k_table(
    top_k_data: dict[str, list[ConfigRank]],
    output_path: str | Path,
    caption_key: str | None = None,
    label: str | None = None,
    lang: str | None = None,
) -> Path:
    """Export Categoria B — Top-3 per filter with extended λ_max statistics.

    Parameters
    ----------
    top_k_data
        ``{filter_type: [ConfigRank, ...]}`` from :func:`~.stats.top_k_per_filter`.
    output_path
        Destination ``.tex`` file.
    caption_key
        i18n key for the caption.
        Defaults to ``analysis.tables.top_k_extended.caption``.
    label
        LaTeX label.
    lang
        ``"pt"`` or ``"en"``.

    Returns
    -------
    Path
    """
    if caption_key is None:
        caption_key = "analysis.tables.top_k_extended.caption"
    caption = t(caption_key, lang=lang)

    columns = [
        t("analysis.tables.col.filter", lang=lang),
        t("analysis.tables.col.rank", lang=lang),
        t("analysis.tables.col.window", lang=lang),
        t("analysis.tables.col.n_chaotic", lang=lang),
        t("analysis.tables.col.pct_chaotic_finite", lang=lang),
        t("analysis.tables.col.lmax_mean", lang=lang),
        t("analysis.tables.col.lmax_max", lang=lang),
        t("analysis.tables.col.lmax_std", lang=lang),
        t("analysis.tables.col.lmax_ci95", lang=lang),
    ]

    rows: list[list[str]] = []
    for ft in ["lowpass", "highpass", "bandpass", "bandstop"]:
        entries = top_k_data.get(ft, [])
        for entry in entries:
            rows.append(
                [
                    _filter_label(ft, lang or "pt"),
                    str(entry["rank"]),
                    _window_label(entry, lang or "pt"),
                    str(entry["n_chaotic"]),
                    f"{entry['pct_chaotic_finite']:.1f}\\%",
                    _fmt(entry["lmax_mean"], 4),
                    _fmt(entry["lmax_max"], 4),
                    _fmt(entry["lmax_std"], 4),
                    _ci_str(entry["lmax_ci_95_low"], entry["lmax_ci_95_high"]),
                ]
            )

    tex = _render_tex(columns, rows, caption, label)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex, encoding="utf-8")
    return output


def export_full_ranking_table(
    rank_data: list[ConfigRank],
    output_path: str | Path,
    caption_key: str | None = None,
    label: str | None = None,
    lang: str | None = None,
) -> Path:
    """Export Categoria C.1 — Full ranking (longtable).

    Parameters
    ----------
    rank_data
        Sorted list from :func:`~.stats.rank_configurations`.
    output_path
        Destination ``.tex`` file.
    caption_key
        i18n key for the caption.
        Defaults to ``analysis.tables.full_ranking.caption``.
    label
        LaTeX label.
    lang
        ``"pt"`` or ``"en"``.

    Returns
    -------
    Path
    """
    if caption_key is None:
        caption_key = "analysis.tables.full_ranking.caption"
    caption = t(caption_key, lang=lang)

    columns = [
        t("analysis.tables.col.rank", lang=lang),
        t("analysis.tables.col.filter", lang=lang),
        t("analysis.tables.col.window", lang=lang),
        t("analysis.tables.col.n_chaotic", lang=lang),
        t("analysis.tables.col.pct_chaotic_finite", lang=lang),
        t("analysis.tables.col.lmax_mean", lang=lang),
        t("analysis.tables.col.lmax_max", lang=lang),
        t("analysis.tables.col.lmax_std", lang=lang),
        t("analysis.tables.col.lmax_ci95", lang=lang),
    ]

    rows: list[list[str]] = []
    for entry in rank_data:
        rows.append(
            [
                str(entry["rank"]),
                _filter_label(entry["filter_type"], lang or "pt"),
                _window_label(entry, lang or "pt"),
                str(entry["n_chaotic"]),
                f"{entry['pct_chaotic_finite']:.1f}\\%",
                _fmt(entry["lmax_mean"], 4),
                _fmt(entry["lmax_max"], 4),
                _fmt(entry["lmax_std"], 4),
                _ci_str(entry["lmax_ci_95_low"], entry["lmax_ci_95_high"]),
            ]
        )

    tex = _render_tex(columns, rows, caption, label, use_longtable=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex, encoding="utf-8")
    return output


def export_sweet_spots_table(
    sweet_spot_data: dict[str, SweetSpot],
    output_path: str | Path,
    caption_key: str | None = None,
    label: str | None = None,
    lang: str | None = None,
) -> Path:
    """Export Categoria C.2 — Sweet-spot points per filter type.

    Parameters
    ----------
    sweet_spot_data
        ``{filter_type: SweetSpot}`` from :func:`~.stats.sweet_spot_per_filter`.
    output_path
        Destination ``.tex`` file.
    caption_key
        i18n key for the caption.
        Defaults to ``analysis.tables.sweet_spots.caption``.
    label
        LaTeX label.
    lang
        ``"pt"`` or ``"en"``.

    Returns
    -------
    Path
    """
    if caption_key is None:
        caption_key = "analysis.tables.sweet_spots.caption"
    caption = t(caption_key, lang=lang)

    columns = [
        t("analysis.tables.col.filter", lang=lang),
        t("analysis.tables.col.window", lang=lang),
        t("analysis.tables.col.n_z", lang=lang),
        t("analysis.tables.col.omega_c", lang=lang),
        t("analysis.tables.col.lmax_max", lang=lang),
        t("analysis.tables.col.lmax_ci95", lang=lang),
    ]

    rows: list[list[str]] = []
    for ft in ["lowpass", "highpass", "bandpass", "bandstop"]:
        spot = sweet_spot_data.get(ft)
        if spot is None:
            continue
        rows.append(
            [
                _filter_label(ft, lang or "pt"),
                _window_label(spot, lang or "pt"),
                str(spot["n_z"]),
                _fmt(spot["omega_c"], 4),
                _fmt(spot["lmax"], 6),
                _ci_str(spot["lmax_ci_95_low"], spot["lmax_ci_95_high"]),
            ]
        )

    tex = _render_tex(columns, rows, caption, label)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex, encoding="utf-8")
    return output


def export_kaiser_beta_optimal_table(
    beta_data: dict[str, KaiserBetaOptimal],
    output_path: str | Path,
    caption_key: str | None = None,
    label: str | None = None,
    lang: str | None = None,
) -> Path:
    """Export optimal Kaiser β per filter type.

    Parameters
    ----------
    beta_data
        ``{filter_type: KaiserBetaOptimal}`` from :func:`~.stats.kaiser_beta_optimal`.
    output_path
        Destination ``.tex`` file.
    caption_key
        i18n key. Defaults to ``analysis.tables.kaiser_beta_optimal.caption``.
    label
        LaTeX label.
    lang
        ``"pt"`` or ``"en"``.

    Returns
    -------
    Path
    """
    if caption_key is None:
        caption_key = "analysis.tables.kaiser_beta_optimal.caption"
    caption = t(caption_key, lang=lang)

    columns = [
        t("analysis.tables.col.filter", lang=lang),
        t("analysis.tables.col.beta", lang=lang),
        t("analysis.tables.col.n_chaotic", lang=lang),
        t("analysis.tables.col.pct_chaotic_finite", lang=lang),
        t("analysis.tables.col.lmax_mean", lang=lang),
        t("analysis.tables.col.lmax_max", lang=lang),
    ]

    rows: list[list[str]] = []
    for ft in ["lowpass", "highpass", "bandpass", "bandstop"]:
        entry = beta_data.get(ft)
        if entry is None:
            continue
        rows.append(
            [
                _filter_label(ft, lang or "pt"),
                _fmt(entry["beta"], 2),
                str(entry["n_chaotic"]),
                f"{entry['pct_chaotic_finite']:.1f}\\%",
                _fmt(entry["lmax_mean"], 4),
                _fmt(entry["lmax_max"], 4),
            ]
        )

    tex = _render_tex(columns, rows, caption, label)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex, encoding="utf-8")
    return output


def export_consolidated_top_k_table(
    consolidated_top_k: dict[str, list[ConfigRank]],
    output_path: str | Path,
    caption_key: str | None = None,
    label: str | None = None,
    lang: str | None = None,
) -> Path:
    """Export consolidated top-k (Categoria A, Kaiser collapsed to best β).

    Parameters
    ----------
    consolidated_top_k
        ``{filter_type: [ConfigRank, ...]}`` from :func:`~.stats.top_k_per_filter`
        run on consolidated sweeps.
    output_path
        Destination ``.tex`` file.
    caption_key
        i18n key. Defaults to ``analysis.tables.consolidated_top_k.caption``.
    label
        LaTeX label.
    lang
        ``"pt"`` or ``"en"``.

    Returns
    -------
    Path
    """
    if caption_key is None:
        caption_key = "analysis.tables.consolidated_top_k.caption"
    caption = t(caption_key, lang=lang)

    columns = [
        t("analysis.tables.col.filter", lang=lang),
        t("analysis.tables.col.rank", lang=lang),
        t("analysis.tables.col.window", lang=lang),
        t("analysis.tables.col.n_chaotic", lang=lang),
        t("analysis.tables.col.pct_chaotic", lang=lang),
    ]

    rows: list[list[str]] = []
    for ft in ["lowpass", "highpass", "bandpass", "bandstop"]:
        entries = consolidated_top_k.get(ft, [])
        for entry in entries:
            rows.append(
                [
                    _filter_label(ft, lang or "pt"),
                    str(entry["rank"]),
                    _window_label(entry, lang or "pt"),
                    str(entry["n_chaotic"]),
                    f"{entry['pct_chaotic']:.1f}\\%",
                ]
            )

    tex = _render_tex(columns, rows, caption, label)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex, encoding="utf-8")
    return output


def export_consolidated_extended_table(
    consolidated_top_k: dict[str, list[ConfigRank]],
    output_path: str | Path,
    caption_key: str | None = None,
    label: str | None = None,
    lang: str | None = None,
) -> Path:
    """Export consolidated extended top-k (Categoria B).

    Parameters
    ----------
    consolidated_top_k
        From :func:`~.stats.top_k_per_filter` on consolidated sweeps.
    output_path
        Destination ``.tex`` file.
    caption_key
        i18n key. Defaults to ``analysis.tables.consolidated_extended.caption``.
    label
        LaTeX label.
    lang
        ``"pt"`` or ``"en"``.

    Returns
    -------
    Path
    """
    if caption_key is None:
        caption_key = "analysis.tables.consolidated_extended.caption"
    caption = t(caption_key, lang=lang)

    columns = [
        t("analysis.tables.col.filter", lang=lang),
        t("analysis.tables.col.rank", lang=lang),
        t("analysis.tables.col.window", lang=lang),
        t("analysis.tables.col.n_chaotic", lang=lang),
        t("analysis.tables.col.pct_chaotic_finite", lang=lang),
        t("analysis.tables.col.lmax_mean", lang=lang),
        t("analysis.tables.col.lmax_max", lang=lang),
        t("analysis.tables.col.lmax_std", lang=lang),
        t("analysis.tables.col.lmax_ci95", lang=lang),
    ]

    rows: list[list[str]] = []
    for ft in ["lowpass", "highpass", "bandpass", "bandstop"]:
        entries = consolidated_top_k.get(ft, [])
        for entry in entries:
            rows.append(
                [
                    _filter_label(ft, lang or "pt"),
                    str(entry["rank"]),
                    _window_label(entry, lang or "pt"),
                    str(entry["n_chaotic"]),
                    f"{entry['pct_chaotic_finite']:.1f}\\%",
                    _fmt(entry["lmax_mean"], 4),
                    _fmt(entry["lmax_max"], 4),
                    _fmt(entry["lmax_std"], 4),
                    _ci_str(entry["lmax_ci_95_low"], entry["lmax_ci_95_high"]),
                ]
            )

    tex = _render_tex(columns, rows, caption, label)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex, encoding="utf-8")
    return output


def export_consolidated_full_ranking(
    consolidated_rank: list[ConfigRank],
    output_path: str | Path,
    caption_key: str | None = None,
    label: str | None = None,
    lang: str | None = None,
) -> Path:
    """Export consolidated full ranking (Categoria C.1, longtable).

    Parameters
    ----------
    consolidated_rank
        From :func:`~.stats.rank_configurations` on consolidated sweeps.
    output_path
        Destination ``.tex`` file.
    caption_key
        i18n key. Defaults to ``analysis.tables.consolidated_full_ranking.caption``.
    label
        LaTeX label.
    lang
        ``"pt"`` or ``"en"``.

    Returns
    -------
    Path
    """
    if caption_key is None:
        caption_key = "analysis.tables.consolidated_full_ranking.caption"
    caption = t(caption_key, lang=lang)

    columns = [
        t("analysis.tables.col.rank", lang=lang),
        t("analysis.tables.col.filter", lang=lang),
        t("analysis.tables.col.window", lang=lang),
        t("analysis.tables.col.n_chaotic", lang=lang),
        t("analysis.tables.col.pct_chaotic_finite", lang=lang),
        t("analysis.tables.col.lmax_mean", lang=lang),
        t("analysis.tables.col.lmax_max", lang=lang),
        t("analysis.tables.col.lmax_std", lang=lang),
        t("analysis.tables.col.lmax_ci95", lang=lang),
    ]

    rows: list[list[str]] = []
    for entry in consolidated_rank:
        rows.append(
            [
                str(entry["rank"]),
                _filter_label(entry["filter_type"], lang or "pt"),
                _window_label(entry, lang or "pt"),
                str(entry["n_chaotic"]),
                f"{entry['pct_chaotic_finite']:.1f}\\%",
                _fmt(entry["lmax_mean"], 4),
                _fmt(entry["lmax_max"], 4),
                _fmt(entry["lmax_std"], 4),
                _ci_str(entry["lmax_ci_95_low"], entry["lmax_ci_95_high"]),
            ]
        )

    tex = _render_tex(columns, rows, caption, label, use_longtable=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex, encoding="utf-8")
    return output
