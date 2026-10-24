#!/usr/bin/env python3
"""Show the annotated tables that ``latexdiff`` comments out.

The ``PICTUREENV`` setting keeps a changed ``tabular`` from being split cell
by cell, but LaTeXdiff still comments the body of a float that disappears
between the two versions.  This script reproduces those old tables in red at
the end of the diff, before the bibliography, without changing the numbering of
the current tables.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TABLE_RE = re.compile(
    r"\\begin\{(?P<env>table\*?)\}(?:\[[^\]]*\])?"
    r"(?P<body>.*?)\\end\{(?P=env)\}",
    re.S,
)
CAPTION_RE = re.compile(r"\\caption\{")
LABEL_RE = re.compile(r"\\label\{[^{}]*\}")
MARKER = "% ===== DELETED TABLES FROM THE ANNOTATED VERSION ====="


def extract_braced(text: str, opening: int) -> tuple[str, int]:
    """Return the content of the brace group starting at ``opening``."""
    if text[opening] != "{":
        raise ValueError("expected an opening brace")
    depth = 0
    for position in range(opening, len(text)):
        character = text[position]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : position], position + 1
    raise ValueError("unbalanced braces in table caption")


ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")


def table_blocks(old_tex: str) -> list[str]:
    """Extract old tables and render them as red, unnumbered floats."""
    blocks: list[str] = []
    for index, match in enumerate(TABLE_RE.finditer(old_tex), start=1):
        environment = match.group("env")
        body = match.group("body")
        caption_match = CAPTION_RE.search(body)
        caption = f"Table {index}"
        if caption_match:
            caption, end = extract_braced(body, caption_match.end() - 1)
            body = body[: caption_match.start()] + body[end:]
        caption = LABEL_RE.sub("", caption).strip()
        body = LABEL_RE.sub("", body).strip()
        numeral = ROMAN[index - 1] if index <= len(ROMAN) else str(index)
        caption_line = (
            "\\caption*{\\textcolor{red}{\\textbf{Deleted Table "
            + numeral
            + " (annotated version):} "
            + caption
            + "}}\n"
        )
        blocks.append(
            f"\\begin{{{environment}}}[!b]\n"
            "\\centering\n"
            + caption_line
            + "\\begingroup\\color{red}\n"
            + body
            + "\n\\endgroup\n"
            + f"\\end{{{environment}}}\n"
        )
    return blocks


def main() -> None:
    """Append the deleted-table appendix to a latexdiff output file."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: paper_diff_tables.py OLD.tex DIFF.tex")

    old_path, diff_path = map(Path, sys.argv[1:])
    old_tex = old_path.read_text()
    diff_tex = diff_path.read_text()
    if MARKER in diff_tex:
        return

    blocks = table_blocks(old_tex)
    if not blocks:
        return

    appendix = (
        f"\n{MARKER}\n"
        "\\clearpage\n"
        "\\section*{Deleted tables from the annotated version}\n"
        "\\noindent\\emph{These tables exist only in the annotated version; "
        "they are reproduced in red for comparison.}\n\n" + "\n".join(blocks) + "\\clearpage\n"
    )
    bibliography = "\\bibliographystyle{IEEEtran}"
    if bibliography in diff_tex:
        diff_tex = diff_tex.replace(bibliography, appendix + bibliography, 1)
    else:
        diff_tex = diff_tex.replace("\\end{document}", appendix + "\\end{document}", 1)
    diff_path.write_text(diff_tex)
    print(f"paper-diff: restored {len(blocks)} deleted tables in the diff")


if __name__ == "__main__":
    main()
