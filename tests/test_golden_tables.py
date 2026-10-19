"""Golden-file checks for the generated LaTeX tables and their compilation.

``data/analysis_output/tables`` is the versioned reference output of
``chaotic-pfc run analysis export-tables``. Regenerating it from the versioned
``.npz`` checkpoints must reproduce every file byte for byte; a difference is
reported as a unified diff so the cause is visible.

The compilation test wraps each table in a minimal document that loads the
packages the tables rely on (``booktabs``, ``longtable``, ``amsmath``,
``graphicx``) and runs ``pdflatex``. It is skipped when no TeX distribution is
installed; the CI job *Manuscript tables (LaTeX)* always runs it inside the
pinned TeX Live image.
"""

from __future__ import annotations

import difflib
import shutil
import subprocess
from pathlib import Path

import pytest

from chaotic_pfc.cli import main

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "data" / "analysis_output" / "tables"
SWEEPS = REPO / "data" / "sweeps"

MINIMAL_DOC = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{graphicx}
\begin{document}
\input{table.tex}
\end{document}
"""


def _regenerate(target: Path) -> None:
    code = main(
        [
            "run",
            "analysis",
            "export-tables",
            "--sweep-dir",
            str(SWEEPS),
            "--output-dir",
            str(target),
            "--lang",
            "all",
        ]
    )
    assert code == 0


def test_regenerated_tables_match_versioned_golden_files(tmp_path: Path) -> None:
    _regenerate(tmp_path)
    regenerated = sorted(tmp_path.rglob("*.tex"))
    versioned = sorted(TABLES.rglob("*.tex"))
    assert {p.relative_to(tmp_path) for p in regenerated} == {
        p.relative_to(TABLES) for p in versioned
    }, "the set of exported tables changed (update data/analysis_output/tables)"

    mismatches: list[str] = []
    for path in versioned:
        rel = path.relative_to(TABLES)
        expected = path.read_text(encoding="utf-8")
        actual = (tmp_path / rel).read_text(encoding="utf-8")
        if expected != actual:
            diff = "".join(
                difflib.unified_diff(
                    expected.splitlines(keepends=True),
                    actual.splitlines(keepends=True),
                    fromfile=f"versioned/{rel.as_posix()}",
                    tofile=f"regenerated/{rel.as_posix()}",
                )
            )
            mismatches.append(diff)
    assert not mismatches, "regenerated tables differ from the versioned ones:\n" + "\n".join(
        mismatches
    )


@pytest.mark.parametrize(
    "tex_path",
    sorted(TABLES.rglob("*.tex")),
    ids=lambda p: p.relative_to(TABLES).as_posix(),
)
def test_each_versioned_table_compiles(tex_path: Path, tmp_path: Path) -> None:
    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex is not installed")
    (tmp_path / "table.tex").write_text(tex_path.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "doc.tex").write_text(MINIMAL_DOC, encoding="utf-8")
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"{tex_path.relative_to(TABLES)} does not compile:\n{proc.stdout[-4000:]}"
    )
    assert (tmp_path / "doc.pdf").is_file()
