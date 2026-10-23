#!/usr/bin/env python3
"""Deterministic consistency audit for the JCIS manuscript.

The audit cross-checks the manuscript sources against the artefacts that back
them. It never edits the paper and never guesses: every finding is the result
of an explicit assertion in the code below.

Usage::

    chaotic-pfc paper audit \
        --tex articles/jcis/article.tex \
        --bib articles/jcis/references.bib \
        --figures figures/ \
        [--log articles/jcis/JCIS.log] [--blg articles/jcis/JCIS.blg] [--json out.json]

Only the Python standard library is used, plus the project's own
``chaotic_pfc.config`` for check C7 and the shared rounding helper ``pct`` from
:mod:`chaotic_pfc.paper.numbers` (check C6). The process exits with status 1
when at least one ERROR is reported and with status 0 otherwise; WARN never
fails the build.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
import warnings
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from importlib import resources
from pathlib import Path
from typing import Any

from chaotic_pfc.paper.numbers import pct

# ``pct`` (the shared rounding rule) is imported from
# :mod:`chaotic_pfc.paper.numbers` above, so the article, the audit and the
# numbers tool agree on the half-way cases (e.g. 434/4000 = 10.85% -> 10.9%).

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Portuguese terms that must not appear in figures submitted to JCIS, which
# requires English artwork. Comparison is accent-insensitive, see
# _normalise_text().
PORTUGUESE_TERMS = (
    "Órbita",
    "Densidade",
    "Número",
    "caótica",
    "periódica",
    "limitada",
    "Configurações",
    "Caos",
    "Concordância",
)

# Entries listed here are exempt from the "every reference needs a DOI" rule
# (check C2). The list ships as package data (``paper/doi_whitelist.txt``) and
# is read with :mod:`importlib.resources` (see :func:`_load_whitelist`).

CHECK_ORDER = ("IN", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")

CHECK_TITLES = {
    "IN": "input files",
    "C1": "citation keys",
    "C2": "DOI in references",
    "C3": "document structure",
    "C4": "compilation log",
    "C5": "figure language",
    "C6": "table identities",
    "C7": "article constants versus config",
    "C8": "text numbers versus paper numbers",
}

# LaTeX log lines treated as fatal.
_LOG_ERROR_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Citation [`'\"](?P<key>[^'\"]+)['\"][^\n]*?undefined"), "undefined citation"),
    (re.compile(r"Reference [`'\"](?P<key>[^'\"]+)['\"][^\n]*?undefined"), "undefined reference"),
    (re.compile(r"Undefined control sequence"), "undefined control sequence"),
    (re.compile(r"I couldn't open database file"), "missing bibliography database"),
    (re.compile(r"Missing \$ inserted"), "missing math shift"),
    (re.compile(r"Missing [{}] inserted"), "missing brace"),
    (re.compile(r"I was expecting a `,' or a `}'"), "malformed bib entry"),
)

# LaTeX log lines treated as warnings.
_LOG_WARN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Missing character"), "missing character"),
    (re.compile(r"^Missing\b"), "missing material"),
    (re.compile(r"Repeated entry"), "repeated bib entry"),
)

_LOG_OVERFULL_RE = re.compile(r"^(?P<kind>Overfull|Underfull) \\[hv]box \((?P<amount>[0-9.]+)pt")

_BIB_ENTRY_RE = re.compile(r"@(?P<type>[A-Za-z]+)\s*(?P<open>[{(])")
_BIB_FIELD_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_SKIPPED_BIB_TYPES = frozenset({"comment", "preamble", "string"})

_CITE_RE = re.compile(r"\\(?P<cmd>[Cc]ite[A-Za-z]*)\*?\s*(?:\[[^\]]*\]\s*)*\{(?P<keys>[^}]*)\}")
_LABEL_RE = re.compile(r"\\label\s*\{([^}]*)\}")
_REF_RE = re.compile(r"\\(?:eqref|ref|subref)\s*\{([^}]*)\}")
_GRAPHICS_RE = re.compile(r"\\includegraphics\s*\*?(?:\s*\[[^\]]*\])*\s*\{([^}]*)\}")
_INCLUDESVG_RE = re.compile(r"\\includesvg\s*\*?(?:\s*\[[^\]]*\])*\s*\{([^}]*)\}")
_INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]*)\}")
_BIBLIOGRAPHY_RE = re.compile(r"\\bibliography\s*\{([^}]*)\}")

_GRAPHICS_SUFFIXES = (".png", ".pdf", ".jpg", ".jpeg", ".svg")


@dataclass
class Finding:
    """A single audit result."""

    level: str
    check: str
    message: str
    line: int | None = None
    file: str | None = None


@dataclass
class CheckResult:
    """The findings of one check, a coverage line, and non-finding details."""

    findings: list[Finding]
    coverage: str
    details: list[str] = field(default_factory=list)


@dataclass
class BibEntry:
    """One parsed bibliography entry."""

    type: str
    key: str
    fields: dict[str, str]
    line: int


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _strip_comments(text: str) -> str:
    r"""Blank out LaTeX/BibTeX percent comments while preserving offsets.

    An escaped percent (``\\%``) is kept. Newlines are preserved so that
    character offsets still map to line numbers.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char == "\\" and i + 1 < n:
            out.append(char)
            out.append(text[i + 1])
            i += 2
            continue
        if char == "%":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _read_braced(text: str, start: int) -> tuple[str, int]:
    """Return the content of the ``{...}`` group at *start* and its end index."""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return text[start + 1 :], n


def _find_braced_commands(text: str, command: str) -> list[tuple[str, int]]:
    r"""Find every ``\\command{...}`` and return (content, position) pairs.

    An optional ``[...]`` argument between the command and the mandatory
    braces is skipped, so ``\\title[short]{Long}`` yields ``Long``.
    """
    results: list[tuple[str, int]] = []
    pattern = re.compile(
        r"\\" + re.escape(command) + r"(?![A-Za-z])\s*(?:\*)?\s*(?:\[[^\]]*\]\s*)*"
    )
    for match in pattern.finditer(text):
        j = match.end()
        while j < len(text) and text[j] in " \t\r\n":
            j += 1
        if j < len(text) and text[j] == "{":
            content, _ = _read_braced(text, j)
            results.append((content, match.start()))
    return results


def _normalise_text(text: str) -> str:
    """Lowercase *text* and strip XML entities and combining accents."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.lower()


def _extract_path_commands(text: str) -> list[tuple[str, str, int]]:
    r"""Return (kind, raw_path, line) for every path-loading command.

    Kinds are ``includegraphics``, ``includesvg``, ``input`` (for
    ``\\input``/``\\include``) and ``bibliography``.
    """
    clean = _strip_comments(text)
    commands: list[tuple[str, str, int]] = []
    for kind, pattern in (
        ("includegraphics", _GRAPHICS_RE),
        ("includesvg", _INCLUDESVG_RE),
        ("input", _INPUT_RE),
        ("bibliography", _BIBLIOGRAPHY_RE),
    ):
        for match in pattern.finditer(clean):
            commands.append((kind, match.group(1).strip(), _line_of(clean, match.start())))
    return commands


def _resolve_path(tex_dir: Path, raw: str, suffixes: tuple[str, ...]) -> Path | None:
    """Resolve *raw* relative to *tex_dir*, trying *suffixes* when missing."""
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = tex_dir / candidate
    if candidate.suffix:
        return candidate if candidate.is_file() else None
    for suffix in suffixes:
        trial = Path(str(candidate) + suffix)
        if trial.is_file():
            return trial
    return None


# ---------------------------------------------------------------------------
# Bibliography parsing
# ---------------------------------------------------------------------------


def _read_bib_value(text: str, i: int) -> tuple[str, int]:
    n = len(text)
    if i < n and text[i] == "{":
        depth = 0
        j = i
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1 : j].strip(), j + 1
            j += 1
        return text[i + 1 :].strip(), n
    if i < n and text[i] == '"':
        j = i + 1
        while j < n and text[j] != '"':
            j += 1
        return text[i + 1 : j].strip(), min(j + 1, n)
    j = i
    while j < n and text[j] != ",":
        j += 1
    return text[i:j].strip(), j


def _iter_bib_fields(text: str) -> Iterator[tuple[str, str]]:
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        match = _BIB_FIELD_NAME_RE.match(text, i)
        if match is None:
            i += 1
            continue
        name = match.group(0)
        i = match.end()
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n or text[i] != "=":
            continue
        i += 1
        while i < n and text[i] in " \t\r\n":
            i += 1
        value, i = _read_bib_value(text, i)
        yield name, value


def _parse_bib_entries(text: str) -> list[BibEntry]:
    entries: list[BibEntry] = []
    i = 0
    n = len(text)
    while True:
        match = _BIB_ENTRY_RE.search(text, i)
        if match is None:
            break
        etype = match.group("type").lower()
        opener = match.group("open")
        closer = "}" if opener == "{" else ")"
        depth = 1
        j = match.end()
        while j < n and depth > 0:
            if text[j] == opener:
                depth += 1
            elif text[j] == closer:
                depth -= 1
            j += 1
        body = text[match.end() : j - 1] if depth == 0 else text[match.end() :]
        i = j
        if etype in _SKIPPED_BIB_TYPES:
            continue
        key_match = re.match(r"\s*([^,\s]+)\s*,?", body)
        if key_match is None:
            continue
        fields = {name.lower(): value for name, value in _iter_bib_fields(body[key_match.end() :])}
        entries.append(BibEntry(etype, key_match.group(1), fields, _line_of(text, match.start())))
    return entries


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_citation_keys(tex_text: str, bib_text: str) -> CheckResult:
    """C1: cited keys must exist, defined keys must be cited, keys must be unique."""
    findings: list[Finding] = []
    clean_tex = _strip_comments(tex_text)
    entries = _parse_bib_entries(_strip_comments(bib_text))
    defined: set[str] = set()
    seen: set[str] = set()

    for entry in entries:
        if entry.key in seen:
            findings.append(
                Finding("WARN", "C1", f"duplicate bibliography key '{entry.key}'", entry.line)
            )
        seen.add(entry.key)
        defined.add(entry.key)

    cited: dict[str, int] = {}
    cite_sites = 0
    for match in _CITE_RE.finditer(clean_tex):
        cite_sites += 1
        line = _line_of(clean_tex, match.start())
        for raw_key in match.group("keys").split(","):
            key = raw_key.strip()
            if key:
                cited.setdefault(key, line)

    for key, line in cited.items():
        if key not in defined:
            findings.append(
                Finding(
                    "ERROR", "C1", f"\\cite key '{key}' is not defined in the bibliography", line
                )
            )
    for entry in entries:
        if entry.key not in cited:
            findings.append(
                Finding("WARN", "C1", f"bibliography key '{entry.key}' is never cited", entry.line)
            )
    coverage = f"parsed {len(entries)} bib entries, found {cite_sites} cite sites"
    return CheckResult(findings, coverage)


def check_dois(bib_text: str, whitelist: frozenset[str]) -> CheckResult:
    """C2: every bibliography entry needs a non-empty DOI unless whitelisted."""
    findings: list[Finding] = []
    entries = _parse_bib_entries(_strip_comments(bib_text))
    for entry in entries:
        if entry.fields.get("doi", "").strip() or entry.key in whitelist:
            continue
        findings.append(
            Finding("ERROR", "C2", f"entry '{entry.key}' has no 'doi' field", entry.line)
        )
    coverage = f"checked {len(entries)} entries, {len(whitelist)} whitelisted"
    return CheckResult(findings, coverage)


def check_structure(tex_text: str, tex_dir: Path) -> CheckResult:
    """C3: title count, undefined references, unused labels, resolvable paths."""
    findings: list[Finding] = []
    clean = _strip_comments(tex_text)

    maketitle = clean.find("\\maketitle")
    title_region = clean[:maketitle] if maketitle != -1 else clean
    titles = [
        content for content, _ in _find_braced_commands(title_region, "title") if content.strip()
    ]
    if len(titles) != 1:
        findings.append(
            Finding("ERROR", "C3", f"expected exactly one non-empty \\title, found {len(titles)}")
        )

    labels = {m.group(1).strip() for m in _LABEL_RE.finditer(clean) if m.group(1).strip()}
    refs: dict[str, int] = {}
    for match in _REF_RE.finditer(clean):
        key = match.group(1).strip()
        if key:
            refs.setdefault(key, _line_of(clean, match.start()))

    for key, line in refs.items():
        if key not in labels:
            findings.append(Finding("ERROR", "C3", f"\\ref to undefined label '{key}'", line))
    for label in sorted(labels):
        if label not in refs:
            findings.append(Finding("WARN", "C3", f"label '{label}' is never referenced"))

    # Path commands must resolve relative to the .tex directory.
    targets: list[tuple[str, str, int]] = []
    for kind, raw, line in _extract_path_commands(tex_text):
        if kind == "bibliography":
            names = [name.strip() for name in raw.split(",") if name.strip()]
        else:
            names = [raw] if raw else []
        for name in names:
            targets.append((kind, name, line))

    resolved = 0
    for kind, name, line in targets:
        suffixes: tuple[str, ...]
        if kind in ("includegraphics", "includesvg"):
            suffixes = _GRAPHICS_SUFFIXES
        elif kind == "bibliography":
            suffixes = (".bib",)
        else:
            suffixes = (".tex",)
        if _resolve_path(tex_dir, name, suffixes) is not None:
            resolved += 1
        else:
            findings.append(
                Finding("ERROR", "C3", f"\\{kind} path '{name}' does not resolve", line)
            )

    coverage = (
        f"{len(titles)} titles, {len(labels)} labels, {len(refs)} refs, "
        f"{resolved}/{len(targets)} paths resolved"
    )
    return CheckResult(findings, coverage)


def check_log(log_texts: list[tuple[str, str]]) -> CheckResult:
    """C4: scan LaTeX and BibTeX logs for undefined names and box warnings."""
    findings: list[Finding] = []
    box_warnings: list[tuple[float, str, str, str]] = []
    lines_seen = 0
    for name, text in log_texts:
        for lineno, line in enumerate(text.splitlines(), start=1):
            lines_seen += 1
            handled = False
            for pattern, description in _LOG_ERROR_PATTERNS:
                match = pattern.search(line)
                if match:
                    key = match.groupdict().get("key")
                    suffix = f": '{key}'" if key else ""
                    findings.append(
                        Finding("ERROR", "C4", f"{description}{suffix} in {name}:{lineno}")
                    )
                    handled = True
                    break
            if handled:
                continue
            overfull = _LOG_OVERFULL_RE.search(line)
            if overfull:
                box_warnings.append(
                    (
                        float(overfull.group("amount")),
                        overfull.group("kind"),
                        f"{name}:{lineno}",
                        line.strip(),
                    )
                )
                continue
            for pattern, description in _LOG_WARN_PATTERNS:
                if pattern.search(line):
                    findings.append(Finding("WARN", "C4", f"{description} in {name}:{lineno}"))
                    break
    if box_warnings:
        counts: dict[str, int] = {}
        for _, kind, _, _ in box_warnings:
            counts[kind] = counts.get(kind, 0) + 1
        summary = ", ".join(f"{counts[kind]} {kind.lower()}" for kind in sorted(counts))
        findings.append(Finding("WARN", "C4", f"box warnings: {summary}"))
        for amount, kind, where, raw in sorted(box_warnings, key=lambda item: -item[0])[:3]:
            findings.append(
                Finding("WARN", "C4", f"worst {kind.lower()} ({amount}pt) at {where}: {raw[:120]}")
            )
    coverage = f"scanned {lines_seen} log lines from {len(log_texts)} file(s)"
    return CheckResult(findings, coverage)


def _scan_terms(path: Path) -> list[str]:
    """Return the Portuguese terms present in *path* (accent-insensitive)."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    haystack = _normalise_text(content)
    return [term for term in PORTUGUESE_TERMS if _normalise_text(term) in haystack]


def check_figure_language(
    tex_text: str,
    tex_dir: Path,
    figures_dir: Path,
    *,
    all_figures: bool,
) -> CheckResult:
    r"""C5: Portuguese terms in the figures the manuscript inserts.

    Both ``\\includegraphics`` and ``\\includesvg`` inclusions are checked.
    Raster inclusions are verified through a sibling vector source
    when one exists, otherwise reported as not automatically verifiable.
    ``all_figures`` additionally scans every SVG under ``figures_dir``.
    """
    findings: list[Finding] = []
    verified = 0
    unverifiable = 0

    for kind, raw, line in _extract_path_commands(tex_text):
        if kind not in ("includegraphics", "includesvg"):
            continue
        target = _resolve_path(tex_dir, raw, _GRAPHICS_SUFFIXES)
        if target is None:
            continue  # already reported by C3
        if target.suffix.lower() == ".svg":
            source: Path | None = target
        else:
            sibling = target.with_suffix(".svg")
            source = sibling if sibling.is_file() else None
        if source is None:
            unverifiable += 1
            findings.append(
                Finding(
                    "WARN",
                    "C5",
                    f"language not automatically verifiable for '{raw}' (raster, no vector source)",
                    line,
                )
            )
            continue
        verified += 1
        matched = _scan_terms(source)
        if matched:
            findings.append(
                Finding(
                    "ERROR",
                    "C5",
                    f"Portuguese terms found: {', '.join(matched)}",
                    file=str(source),
                )
            )

    coverage = f"verified {verified} vector sources, {unverifiable} raster without source"
    if all_figures:
        if not figures_dir.is_dir():
            findings.append(Finding("WARN", "C5", f"figures directory not found: {figures_dir}"))
            coverage += f", {figures_dir} not found"
        else:
            scanned = 0
            flagged = 0
            for svg in sorted(figures_dir.rglob("*.svg")):
                scanned += 1
                matched = _scan_terms(svg)
                if matched:
                    flagged += 1
                    findings.append(
                        Finding(
                            "ERROR",
                            "C5",
                            f"Portuguese terms found: {', '.join(matched)}",
                            file=str(svg),
                        )
                    )
            coverage += f", scanned {scanned} SVGs under {figures_dir} ({flagged} flagged)"
    return CheckResult(findings, coverage)


# ---------------------------------------------------------------------------
# C6: identities inside the result tables
# ---------------------------------------------------------------------------

# Grid size per filter type, from subsection sub:grid (4000 for lowpass and
# bandpass, 2000 for highpass and bandstop).
_GRID_SIZE = {"lowpass": 4000, "bandpass": 4000, "highpass": 2000, "bandstop": 2000}
_TABLE_RULES_RE = re.compile(
    r"\\(?:top|mid|bottom)rule\b|\\hline\b|\\cmidrule(?:\[[^\]]*\])?\{[^}]*\}"
)


def _extract_tabular(clean_tex: str, label: str) -> str | None:
    """Return the ``tabular`` body of the float carrying *label*."""
    anchor = clean_tex.find("\\label{" + label + "}")
    if anchor == -1:
        return None
    start = clean_tex.find("\\begin{tabular}", anchor)
    if start == -1:
        return None
    end = clean_tex.find("\\end{tabular}", start)
    if end == -1:
        return None
    return clean_tex[start + len("\\begin{tabular}") : end]


def _clean_cell(cell: str) -> str:
    text = cell.strip()
    text = text.replace("\\,", " ").replace("\\;", " ")
    text = text.replace("\\%", "%").replace("~", " ")
    text = text.replace("$", "").replace("{", "").replace("}", "")
    text = text.replace("--", "-").replace("\\", "")
    return " ".join(text.split())


def _first_float(cell: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", cell)
    return float(match.group(0)) if match else None


def _parse_ci(cell: str) -> tuple[float, float] | None:
    numbers = re.findall(r"-?\d+\.\d+", cell)
    if len(numbers) < 2:
        return None
    return float(numbers[0]), float(numbers[1])


def _data_chunks(tabular: str) -> list[str]:
    cleaned = _TABLE_RULES_RE.sub("", tabular)
    return [chunk.strip() for chunk in cleaned.split(r"\\") if chunk.strip()]


def _parse_best_rows(tabular: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    current_filter = ""
    for chunk in _data_chunks(tabular):
        cells = [_clean_cell(cell) for cell in chunk.split("&")]
        if len(cells) != 8:
            continue
        nc = _first_float(cells[2])
        if nc is None:
            continue  # header row
        current_filter = cells[0] or current_filter
        ci = _parse_ci(cells[7])
        rho_fin = _first_float(cells[3].replace("%", ""))
        rho_abs = _first_float(cells[4].replace("%", ""))
        lam = _first_float(cells[5])
        sigma = _first_float(cells[6])
        if None in (rho_fin, rho_abs, lam, sigma, ci) or not current_filter:
            warnings.append(f"unparsed tab:best row: {' '.join(chunk.split())[:90]}")
            continue
        rows.append(
            {
                "filter": current_filter.lower(),
                "window": cells[1],
                "nc": int(nc),
                "rho_fin": rho_fin,
                "rho_abs": rho_abs,
                "lambda": lam,
                "sigma": sigma,
                "ci": ci,
            }
        )
    return rows, warnings


def _parse_ranking_rows(tabular: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for chunk in _data_chunks(tabular):
        cells = [_clean_cell(cell) for cell in chunk.split("&")]
        if len(cells) != 7:
            continue
        rank = _first_float(cells[0])
        if rank is None:
            continue  # header row
        nc = _first_float(cells[3])
        lambda_1 = _first_float(cells[5])
        ci = _parse_ci(cells[6])
        rho_fin = _first_float(cells[4].replace("%", ""))
        if nc is None or lambda_1 is None or ci is None or rho_fin is None:
            warnings.append(f"unparsed tab:ranking row: {' '.join(chunk.split())[:90]}")
            continue
        rows.append(
            {
                "filter": cells[1].lower(),
                "window": cells[2],
                "nc": int(nc),
                "rho_fin": rho_fin,
                "lambda": lambda_1,
                "ci": ci,
            }
        )
    return rows, warnings


def _compatible_nb(nc: int, grid: int, rho_fin: float) -> list[int]:
    return [nb for nb in range(nc, grid + 1) if abs(float(pct(nc, nb, 1)) - rho_fin) < 1e-9]


def _same(value_a: object, value_b: object) -> bool:
    if isinstance(value_a, tuple) and isinstance(value_b, tuple):
        return len(value_a) == len(value_b) and all(
            abs(x - y) < 1e-9 for x, y in zip(value_a, value_b, strict=True)
        )
    return abs(float(value_a) - float(value_b)) < 1e-9  # type: ignore[arg-type]  # numeric strings accepted


def check_tables(tex_text: str) -> CheckResult:
    """C6: identities linking N_c, rho_fin, rho_abs, lambda_1, sigma and CI."""
    clean = _strip_comments(tex_text)
    best_tab = _extract_tabular(clean, "tab:best")
    rank_tab = _extract_tabular(clean, "tab:ranking")
    if best_tab is None or rank_tab is None:
        missing = [
            label
            for label, tab in (("tab:best", best_tab), ("tab:ranking", rank_tab))
            if tab is None
        ]
        return CheckResult(
            [Finding("WARN", "C6", f"table(s) not found: {', '.join(missing)}")], "not found"
        )

    best, best_warnings = _parse_best_rows(best_tab)
    ranking, rank_warnings = _parse_ranking_rows(rank_tab)
    findings: list[Finding] = [
        Finding("WARN", "C6", warning) for warning in (*best_warnings, *rank_warnings)
    ]
    details: list[str] = []

    # R1: rho_abs = N_c / N_grid, rounded to 0.1%.
    for row in best:
        grid = _GRID_SIZE.get(row["filter"])
        if grid is None:
            findings.append(Finding("WARN", "C6", f"R1 unknown filter type '{row['filter']}'"))
            continue
        expected = float(pct(row["nc"], grid, 1))
        if abs(expected - row["rho_abs"]) > 1e-9:
            findings.append(
                Finding(
                    "ERROR",
                    "C6",
                    f"R1 {row['filter']}/{row['window']}: rho_abs printed {row['rho_abs']}%, "
                    f"computed {expected}%",
                )
            )

    # R2: rho_fin = N_c / N_b for an integer N_b in [N_c, N_grid].
    for source, rows in (("tab:best", best), ("tab:ranking", ranking)):
        for row in rows:
            grid = _GRID_SIZE.get(row["filter"])
            if grid is None:
                continue
            compatible = _compatible_nb(row["nc"], grid, row["rho_fin"])
            if not compatible:
                findings.append(
                    Finding(
                        "ERROR",
                        "C6",
                        f"R2 {source} {row['filter']}/{row['window']}: no N_b in "
                        f"[{row['nc']},{grid}] reproduces rho_fin={row['rho_fin']}%",
                    )
                )
                continue
            low, high = min(compatible), max(compatible)
            frac_low = 100 * (grid - high) / grid
            frac_high = 100 * (grid - low) / grid
            details.append(
                f"R2 {source} {row['filter']}/{row['window']}: N_c={row['nc']} "
                f"rho_fin={row['rho_fin']}% N_b in [{low},{high}] "
                f"implied divergent {frac_low:.2f}%-{frac_high:.2f}%"
            )

    # R3: CI half-width versus the normal approximation 1.96 sigma / sqrt(N_c).
    for row in best:
        half = (row["ci"][1] - row["ci"][0]) / 2.0
        normal = 1.96 * row["sigma"] / (row["nc"] ** 0.5)
        ratio = half / normal if normal else float("inf")
        details.append(f"R3 {row['filter']}/{row['window']}: CI ratio {ratio:.3f}")
        if not 0.5 <= ratio <= 2.0:
            findings.append(
                Finding(
                    "WARN",
                    "C6",
                    f"R3 {row['filter']}/{row['window']}: CI ratio {ratio:.2f} outside [0.5, 2.0]",
                )
            )

    # R4: combos present in both tables must agree on N_c, lambda_1 and CI.
    best_index = {(row["filter"], row["window"].replace(" ", "")): row for row in best}
    rank_index = {(row["filter"], row["window"].replace(" ", "")): row for row in ranking}
    for key in sorted(set(best_index) & set(rank_index)):
        for field_name in ("nc", "lambda", "ci"):
            if not _same(best_index[key][field_name], rank_index[key][field_name]):
                findings.append(
                    Finding(
                        "ERROR",
                        "C6",
                        f"R4 {key[0]}/{key[1]}: {field_name} diverges between tab:best "
                        f"({best_index[key][field_name]}) and tab:ranking ({rank_index[key][field_name]})",
                    )
                )

    coverage = (
        f"parsed {len(best)} rows from tab:best, {len(ranking)} rows from tab:ranking, "
        f"{len(details)} identity rows"
    )
    return CheckResult(findings, coverage, details)


# ---------------------------------------------------------------------------
# C7: constants asserted in the article versus the package configuration
# ---------------------------------------------------------------------------

# (subsection, quantity, article value, config path, searched names).
# config path None means: no constant with an obvious name, emit WARN and say
# which names were searched. Values come from sub:grid, sub:ensemble, sub:stats.
CONFIG_EXPECTATIONS: tuple[tuple[str, str, float, tuple[str, ...] | None, tuple[str, ...]], ...] = (
    ("sub:ensemble", "burn-in iterations", 500.0, ("sweep", "Nitera"), ()),
    ("sub:ensemble", "accumulation iterations", 3000.0, ("sweep", "Nmap"), ()),
    ("sub:ensemble", "initial conditions per point", 25.0, ("sweep", "n_initial"), ()),
    ("sub:grid", "normalised cutoff_grid", 100.0, ("sweep", "n_cutoffs"), ()),
    (
        "sub:grid",
        "maximum filter order N_z",
        40.0,
        ("sweep", "n_z_max"),
        (),
    ),
    ("sub:grid", "bandwidth Delta omega", 0.05, ("sweep", "bandwidth"), ()),
    ("sub:ensemble", "seed", 42.0, ("seed",), ()),
    # The article no longer asserts a divergence threshold: the sweep kernel
    # flags a cell as divergent on non-finite state detection (NaN/Inf), so
    # there is no X_max constant to locate and the check is omitted here.
    ("sub:ensemble", "initial-condition cube edge", 0.1, ("lyapunov", "perturbation"), ()),
    ("sub:stats", "bootstrap resamples", 1000.0, ("sweep", "n_bootstrap"), ()),
)


def check_config() -> CheckResult:
    """C7: compare the constants asserted in the article with the config."""
    findings: list[Finding] = []
    details: list[str] = []
    try:
        import chaotic_pfc.config as config_module
    except Exception as exc:
        return CheckResult(
            [Finding("WARN", "C7", f"could not import chaotic_pfc.config: {exc}")], "import failed"
        )
    config = config_module.DEFAULT_CONFIG
    checked = 0
    for subsection, quantity, expected, path, searched in CONFIG_EXPECTATIONS:
        if path is None:
            findings.append(
                Finding(
                    "WARN",
                    "C7",
                    f"{quantity} ({subsection}): no obvious constant, searched {', '.join(searched)}",
                )
            )
            continue
        value: Any = config
        for attr in path:
            value = getattr(value, attr, None)
            if value is None:
                break
        if value is None:
            findings.append(
                Finding("WARN", "C7", f"{quantity} ({subsection}): {'.'.join(path)} not found")
            )
            continue
        checked += 1
        if abs(float(value) - expected) > 1e-12:
            findings.append(
                Finding(
                    "ERROR",
                    "C7",
                    f"{quantity} ({subsection}): article {expected} vs config {value}",
                )
            )
        else:
            details.append(f"{quantity} ({subsection}): {value} matches the article")
    coverage = f"located {checked}/{len(CONFIG_EXPECTATIONS)} constants"
    return CheckResult(findings, coverage, details)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# C8: numbers cited in the text versus the paper-numbers labels
# ---------------------------------------------------------------------------

# Sections whose prose cites manuscript results.
_TEXT_NUMBER_SECTIONS = ("Results and Discussion", "Final Remarks")

# Tokens that are deliberately not tied to a ``paper numbers`` label, with the
# reason. Everything else that does not match a label is an ERROR, so the text
# cannot drift from ``paper numbers`` silently.
_TEXT_NUMBER_IGNORES: dict[str, str] = {
    "498": "regeneration wall time (machine-specific; see data/sweeps/ENVIRONMENT.json)",
    "1.4": "Hénon parameter alpha (map constant, not a result)",
    "1.5": "figure layout length (\\vspace fraction)",
    "0.37": "figure layout fraction (\\textwidth)",
    "0.48": "figure layout fraction (\\textwidth)",
    "0.0125": "Bonferroni threshold 0.05/4 computed inline",
    "0.0167": "Bonferroni-adjusted Spearman p-value derived in the text",
    "0.94": "alternative Spearman coefficient discussed for the Kaiser windows",
    "957": "lower end of the seed N_c range, inside the SEED label list",
    "3000": "Nmap, the maximum number of iterations (harness constant)",
    "720": "number of permutations of the six windows (6!)",
    "8.3": "ratio between configurations, derived in the text",
    "4636": "bounded bandstop cells (TOT.bandstop.chaotic + periodic)",
    "62000": "highpass grid points (sum of TOT.highpass.*)",
    "172000": "lowpass grid points (sum of TOT.lowpass.*)",
    "516000": "total grid points across the 172 sweeps (derived)",
    "8550": "CPU model in the machine description",
}

_TEXT_NUMBER_IGNORE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(19|20)\d\d$"), "reference publication year"),
)


def _paper_number_values(numbers: dict[str, Any]) -> list[float]:
    """Every number that appears inside a ``paper numbers`` value."""
    values: list[float] = []
    for value in numbers.values():
        text = str(value).replace("{,}", "").replace(",", "")
        for match in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
            try:
                values.append(float(match))
            except ValueError:  # pragma: no cover - regex guarantees a float
                continue
    return values


def _round_half_up(value: float, places: int) -> float:
    q = Decimal(1).scaleb(-places)
    return float(Decimal(repr(float(value))).quantize(q, rounding=ROUND_HALF_UP))


def _token_matches(token: str, values: list[float]) -> bool:
    """Return whether the token is the half-up rounding of some label."""
    places = len(token.split(".")[1]) if "." in token else 0
    target = float(token)
    return any(_round_half_up(value, places) == target for value in values)


def check_text_numbers(tex_text: str, numbers: dict[str, Any]) -> CheckResult:
    r"""Check that every cited text number maps to a ``paper numbers`` label.

    C8: the prose of the Results and the Final Remarks is scanned for numeric
    citations; a token is accepted when it equals (as a number, ignoring
    trailing zeros) one of the values inside the ``paper numbers`` mapping. A
    short, documented list of tokens that are not results (map/harness
    constants, figure-layout lengths, publication years, values derived inline)
    is exempted with a reason; anything else is an ERROR.
    """
    clean = _strip_comments(tex_text)
    start = -1
    for title in _TEXT_NUMBER_SECTIONS:
        pos = clean.find(f"\\section{{{title}}}")
        if pos != -1 and (start == -1 or pos < start):
            start = pos
    if start == -1:
        return CheckResult(
            [Finding("WARN", "C8", "Results/Final Remarks sections not found")],
            "sections not found",
        )
    end = clean.find("\\bibliographystyle", start)
    body = clean[start:] if end == -1 else clean[start:end]
    body = body.replace("{,}", "").replace("\\,", "").replace("~", " ")

    known = _paper_number_values(numbers)
    findings: list[Finding] = []
    matched: set[str] = set()
    ignored: dict[str, str] = {}
    for raw in re.findall(r"\d+(?:\.\d+)?", body):
        token = raw
        if _token_matches(token, known):
            matched.add(token)
            continue
        reason = _TEXT_NUMBER_IGNORES.get(token)
        if reason is None:
            for pattern, why in _TEXT_NUMBER_IGNORE_PATTERNS:
                if pattern.match(token):
                    reason = why
                    break
        if reason is not None:
            ignored[token] = reason
            continue
        findings.append(
            Finding("ERROR", "C8", f"cited number {raw} matches no 'paper numbers' label")
        )
    coverage = (
        f"{len(matched)} distinct cited numbers match a label, "
        f"{len(ignored)} intentionally ignored, {len(findings)} unmatched"
    )
    details = [f"ignored {token}: {ignored[token]}" for token in sorted(ignored)]
    details.append("covered: " + " ".join(sorted(matched)))
    return CheckResult(findings, coverage, details)


def _safe(check: str, func: Callable[[], CheckResult]) -> CheckResult:
    """Run *func*, converting any exception into a WARN so nothing aborts."""
    try:
        return func()
    except Exception as exc:
        return CheckResult(
            [Finding("WARN", check, f"check aborted: {type(exc).__name__}: {exc}")], "aborted"
        )


def _load_whitelist() -> frozenset[str]:
    """Read the DOI whitelist shipped as package data."""
    try:
        text = (
            resources.files("chaotic_pfc.paper")
            .joinpath("doi_whitelist.txt")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError):  # pragma: no cover - packaging
        return frozenset()
    keys = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            keys.add(stripped)
    return frozenset(keys)


def _read_text(path: str, findings: list[Finding]) -> str | None:
    file = Path(path)
    if not file.is_file():
        findings.append(Finding("WARN", "IN", f"file not found: {path}"))
        return None
    try:
        return file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        findings.append(Finding("WARN", "IN", f"could not read {path}: {exc}"))
        return None


def run_checks(args: argparse.Namespace) -> dict[str, CheckResult]:
    """Run every manuscript check and return the results by code."""
    results: dict[str, CheckResult] = {}
    input_findings: list[Finding] = []
    tex = _read_text(args.tex, input_findings)
    bib = _read_text(args.bib, input_findings)
    if tex is None or bib is None:
        missing = [str(p) for p in (args.tex, args.bib) if not Path(p).is_file()]
        raise FileNotFoundError(f"missing input file(s): {', '.join(missing)}")
    results["IN"] = CheckResult(
        input_findings, f"tex={'ok' if tex else 'missing'}, bib={'ok' if bib else 'missing'}"
    )

    tex_dir = Path(args.tex).resolve().parent
    whitelist = _load_whitelist()

    results["C1"] = _safe("C1", lambda: check_citation_keys(tex, bib))
    results["C2"] = _safe("C2", lambda: check_dois(bib, whitelist))
    results["C3"] = _safe("C3", lambda: check_structure(tex, tex_dir))

    log_texts: list[tuple[str, str]] = []
    for path in (args.log, args.blg):
        if not path:
            continue
        text = _read_text(path, input_findings)
        if text is not None:
            log_texts.append((path, text))
    if log_texts:
        results["C4"] = _safe("C4", lambda: check_log(log_texts))
    else:
        results["C4"] = CheckResult([], "no log or blg provided")

    results["C5"] = _safe(
        "C5",
        lambda: check_figure_language(
            tex, tex_dir, Path(args.figures), all_figures=args.all_figures
        ),
    )
    results["C6"] = _safe("C6", lambda: check_tables(tex))

    results["C7"] = _safe("C7", check_config)

    numbers_map: dict[str, Any] | None = None
    if args.numbers_json:
        numbers_map = json.loads(Path(args.numbers_json).read_text())
    elif args.sweeps:
        from chaotic_pfc.paper import numbers as numbers_module

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            numbers_map = numbers_module.compute(Path(args.sweeps), True, args)
    if numbers_map is None:
        results["C8"] = CheckResult(
            [Finding("WARN", "C8", "no --sweeps or --numbers-json given; check skipped")],
            "skipped (no paper numbers)",
        )
    else:
        results["C8"] = _safe("C8", lambda: check_text_numbers(tex, numbers_map))
    return results


def render_text(results: dict[str, CheckResult]) -> str:
    """Render the findings as grouped, human-readable text."""
    lines: list[str] = []
    all_findings: list[Finding] = []
    for check in CHECK_ORDER:
        result = results.get(check, CheckResult([], ""))
        all_findings += result.findings
        lines.append(f"[{check}] {CHECK_TITLES[check]}")
        lines.append(f"  coverage: {result.coverage}")
        if not result.findings:
            lines.append("  (no findings)")
        for finding in result.findings:
            location = f" line {finding.line}" if finding.line is not None else ""
            target = f" {finding.file}" if finding.file else ""
            lines.append(f"  {finding.level}:{location}{target} {finding.message}")
        for detail in result.details:
            lines.append(f"    {detail}")
        lines.append("")
    errors = sum(1 for finding in all_findings if finding.level == "ERROR")
    warns = sum(1 for finding in all_findings if finding.level == "WARN")
    lines.append(f"Summary: {errors} ERROR, {warns} WARN")
    return "\n".join(lines)


def add_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    """Register ``chaotic-pfc paper audit``."""
    parser = subparsers.add_parser(
        "audit",
        help="Deterministically cross-check the manuscript sources.",
        description="Deterministic consistency audit for the JCIS manuscript.",
    )
    parser.add_argument("--tex", required=True, help="Path to the .tex source.")
    parser.add_argument("--bib", required=True, help="Path to the .bib source.")
    parser.add_argument(
        "--figures",
        default="figures/",
        help="Root directory for --all-figures (default: figures/).",
    )
    parser.add_argument(
        "--all-figures",
        action="store_true",
        dest="all_figures",
        help="Also scan every SVG under --figures, not only the inserted figures.",
    )
    parser.add_argument("--log", help="Path to the LaTeX .log file (optional).")
    parser.add_argument("--blg", help="Path to the BibTeX .blg file (optional).")
    parser.add_argument(
        "--sweeps",
        type=Path,
        help="Sweep root; computes the paper numbers in-process for check C8.",
    )
    parser.add_argument(
        "--numbers-json",
        type=Path,
        dest="numbers_json",
        help="Precomputed 'paper numbers' JSON for check C8 (avoids recomputation).",
    )
    parser.add_argument("--json", dest="json_out", help="Write the findings as JSON to this path.")
    parser.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute ``chaotic-pfc paper audit``."""
    results = run_checks(args)
    print(render_text(results))
    if args.json_out:
        findings = [finding for check in CHECK_ORDER for finding in results[check].findings]
        payload = json.dumps(
            [asdict(finding) for finding in findings], indent=2, ensure_ascii=False
        )
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")
    all_findings = [finding for check in CHECK_ORDER for finding in results[check].findings]
    return 1 if any(finding.level == "ERROR" for finding in all_findings) else 0
