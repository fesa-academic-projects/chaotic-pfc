#!/usr/bin/env python3
"""Check every number in the PFC PDF against a computed or measured source.

The checker extracts the PDF text with ``pdftotext``, normalizes decimal
commas and thousands separators, removes structural figure and section
numbers, and compares the remaining tokens with the values produced by
``chaotic-pfc paper numbers`` and with ``articles/pfc/new_numbers.json``.

Usage::

    python articles/pfc/tools/check_pfc_numbers.py \
        --pdf articles/pfc/artigo.pdf \
        --numbers build/pfc/pfc_numbers.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _pdf_text(pdf: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise SystemExit("pdftotext not found")
    result = subprocess.run(  # noqa: S603
        [pdftotext, "-layout", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _normalize(text: str) -> str:
    cut = re.split(r"\n\s*REFER[ÊE]NCIAS\s*\n", text, maxsplit=1, flags=re.IGNORECASE)
    text = cut[0]
    kept: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if "Curso Superior de Bacharelado" in line or "V-05" in line:
            continue
        if line.strip().isdigit():
            continue
        if "16 a 16 de Novembro de 2026" in line:
            continue
        kept.append(line)
    text = "\n".join(kept)
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(
        r"(?:Figura|Tabela|Quadro|Seção|Subseção|Equação|Algoritmo)\s*~?\s*\d+(?:\.\d+)*", " ", text
    )
    text = re.sub(r"\(\s*\d+\s*\)", " ", text)
    text = re.sub(r"10\s*\^\s*[-]?\s*\d+", " ", text)
    text = re.sub(r"10\s*[-]\s*\d+", " ", text)
    text = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return text


def _source_values(numbers: dict[str, Any], new_numbers: dict[str, Any]) -> set[Decimal]:
    values: set[Decimal] = set()
    for value in numbers.values():
        for token in NUMBER.findall(str(value)):
            try:
                values.add(Decimal(token))
            except InvalidOperation:
                continue
    for source in new_numbers["sources"].values():
        for value in source["values"]:
            values.add(Decimal(str(value)))
    return values


def main(argv: list[str] | None = None) -> int:
    """Run the PDF number check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("articles/pfc/artigo.pdf"))
    parser.add_argument(
        "--numbers",
        type=Path,
        default=Path("build/pfc/pfc_numbers.json"),
    )
    parser.add_argument(
        "--new",
        type=Path,
        default=Path("articles/pfc/new_numbers.json"),
    )
    args = parser.parse_args(argv)
    numbers = json.loads(args.numbers.read_text(encoding="utf-8"))
    new_numbers = json.loads(args.new.read_text(encoding="utf-8"))
    known = _source_values(numbers, new_numbers)
    years = {Decimal(str(year)) for year in new_numbers["citation_years"]}
    structural = {Decimal(str(value)) for value in new_numbers["structural"]}
    text = _normalize(_pdf_text(args.pdf))
    findings: list[tuple[str, str]] = []
    matched: set[str] = set()
    ignored: dict[str, str] = {}
    for match in NUMBER.finditer(text):
        token = match.group(0)
        value = Decimal(token)
        if value in known:
            matched.add(token)
        elif value in years:
            ignored[token] = "citation year"
        elif value in structural:
            ignored[token] = "structural"
        else:
            findings.append((token, "no source"))
    print(f"matched {len(matched)} distinct tokens")
    print(f"ignored {len(ignored)} structural or citation tokens")
    for token, reason in sorted(ignored.items(), key=lambda item: (item[1], item[0])):
        print(f"  ignored {token}: {reason}")
    print(f"unmatched {len(findings)} tokens")
    if findings:
        for token, reason in findings:
            print(f"ERROR: {token}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
