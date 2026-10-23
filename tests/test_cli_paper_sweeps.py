"""CLI tests for the ``chaotic-pfc paper`` and ``chaotic-pfc sweeps`` groups.

Each action is exercised end to end through :func:`chaotic_pfc.cli.main`: the
help screens, a success case (with reduced or example data in ``tmp_path``) and
an error case.
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import pytest

from chaotic_pfc.cli import main

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "data" / "sweeps" / "Hamming (lowpass)" / "variables_lyapunov.npz"

MINIMAL_TEX = r"""
\documentclass[journal]{IEEEtran}
\title{A Toy Manuscript}
\begin{document}
\maketitle
See \cite{alpha} and Equation~\eqref{eq:x}.
\begin{equation}\label{eq:x} x = 1 \end{equation}
\end{document}
"""

MINIMAL_BIB = """
@article{alpha,
  author = {A. Author},
  title = {A title},
  journal = {A Journal},
  year = {2020},
  doi = {10.1000/example},
}
"""

HELP_COMMANDS = (
    ["paper", "numbers", "--help"],
    ["paper", "audit", "--help"],
    ["paper", "figures", "--help"],
    ["sweeps", "regenerate", "--help"],
    ["sweeps", "compare", "--help"],
    ["sweeps", "stats", "--help"],
)


@pytest.mark.parametrize("argv", HELP_COMMANDS)
def test_help_exits_zero(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 0


def _write_manuscript(tmp_path: Path, tex: str = MINIMAL_TEX) -> tuple[Path, Path]:
    tex_path = tmp_path / "article.tex"
    bib_path = tmp_path / "references.bib"
    tex_path.write_text(tex, encoding="utf-8")
    bib_path.write_text(MINIMAL_BIB, encoding="utf-8")
    return tex_path, bib_path


# ── paper numbers ───────────────────────────────────────────────────────────


def test_paper_numbers_compare_success(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    old.write_text('{"A": 1}', encoding="utf-8")
    new.write_text('{"A": 1, "B": 2}', encoding="utf-8")
    assert main(["paper", "numbers", "--compare", str(old), str(new)]) == 0


def test_paper_numbers_missing_file_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "paper",
            "numbers",
            "--compare",
            str(tmp_path / "missing.json"),
            str(tmp_path / "also-missing.json"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "missing.json" in err
    assert "\n" not in err.rstrip("\n")


# ── paper audit ─────────────────────────────────────────────────────────────


def test_paper_audit_success(tmp_path: Path) -> None:
    tex_path, bib_path = _write_manuscript(tmp_path)
    assert main(["paper", "audit", "--tex", str(tex_path), "--bib", str(bib_path)]) == 0


def test_paper_audit_undefined_reference_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = MINIMAL_TEX.replace(r"\eqref{eq:x}", r"\ref{does-not-exist}")
    tex_path, bib_path = _write_manuscript(tmp_path, bad)
    rc = main(["paper", "audit", "--tex", str(tex_path), "--bib", str(bib_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "ERROR" in out and "Summary:" in out


def test_paper_audit_missing_input_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "paper",
            "audit",
            "--tex",
            str(tmp_path / "missing.tex"),
            "--bib",
            str(tmp_path / "missing.bib"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error: missing input file")
    assert "missing.tex" in err


# ── paper figures ───────────────────────────────────────────────────────────


def test_paper_figures_success(tmp_path: Path) -> None:
    assert main(["paper", "figures", "--only", "classification", "--out-dir", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("*.svg"))) == 4


def test_paper_figures_invalid_choice_fails() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["paper", "figures", "--only", "nope"])
    assert excinfo.value.code == 2


# ── sweeps regenerate ───────────────────────────────────────────────────────


def test_sweeps_regenerate_manifest_success(tmp_path: Path) -> None:
    (tmp_path / "lowpass").mkdir()
    (tmp_path / "lowpass" / "variables_lyapunov.npz").write_bytes(b"checkpoint")
    assert main(["sweeps", "regenerate", "--src", str(tmp_path), "--manifest-only"]) == 0
    manifest = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    assert "lowpass/variables_lyapunov.npz" in manifest


def test_sweeps_regenerate_missing_src_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["sweeps", "regenerate", "--src", str(tmp_path / "missing"), "--manifest-only"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "missing" in err


# ── sweeps compare ──────────────────────────────────────────────────────────


def test_sweeps_compare_success(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    for root in (old, new):
        (root / "Hamming (lowpass)").mkdir(parents=True)
        shutil.copy(CHECKPOINT, root / "Hamming (lowpass)" / "variables_lyapunov.npz")
    assert main(["sweeps", "compare", "--old", str(old), "--new", str(new)]) == 0


def test_sweeps_compare_bad_checkpoint_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    for root in (old, new):
        (root / "Hamming (lowpass)").mkdir(parents=True)
    shutil.copy(CHECKPOINT, old / "Hamming (lowpass)" / "variables_lyapunov.npz")
    (new / "Hamming (lowpass)" / "variables_lyapunov.npz").write_bytes(b"not a checkpoint")
    rc = main(["sweeps", "compare", "--old", str(old), "--new", str(new)])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "variables_lyapunov.npz" in err


# ── sweeps stats ────────────────────────────────────────────────────────────


def test_sweeps_stats_success(tmp_path: Path) -> None:
    """The stats command does not require tables in the manuscript."""
    tex = tmp_path / "empty.tex"
    tex.write_text("\\documentclass{article}\n\\begin{document}\n\\end{document}\n")
    argv = [
        "sweeps",
        "stats",
        "--sweep-dir",
        str(REPO / "data" / "sweeps"),
        "--tex",
        str(tex),
        "--csv",
        str(tmp_path / "stats.csv"),
    ]
    assert main(argv) == 0
    assert (tmp_path / "stats.csv").is_file()


def test_sweeps_stats_detects_stale_article_tables(tmp_path: Path) -> None:
    """Canonical tables still use the old threshold until Wave 3."""
    argv = [
        "sweeps",
        "stats",
        "--sweep-dir",
        str(REPO / "data" / "sweeps"),
        "--tex",
        str(REPO / "articles" / "jcis" / "article.tex"),
        "--csv",
        str(tmp_path / "stats.csv"),
    ]
    assert main(argv) == 1
    assert (tmp_path / "stats.csv").is_file()


def test_sweeps_stats_missing_tex_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "sweeps",
            "stats",
            "--sweep-dir",
            str(REPO / "data" / "sweeps"),
            "--tex",
            str(tmp_path / "missing.tex"),
            "--csv",
            str(tmp_path / "stats.csv"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "missing.tex" in err


def test_spearman_matches_between_report_and_numbers() -> None:
    """sweeps/report.py and paper/numbers.py share WINDOW_SIDELOBE_DB and agree on Spearman rho."""
    import numpy as np

    from chaotic_pfc.analysis.sweep._types import WINDOW_SIDELOBE_DB
    from chaotic_pfc.paper import numbers as paper_numbers
    from chaotic_pfc.sweeps import report as sweep_report

    assert WINDOW_SIDELOBE_DB["boxcar"] == WINDOW_SIDELOBE_DB["rectangular"] == 13
    assert WINDOW_SIDELOBE_DB["hann"] < WINDOW_SIDELOBE_DB["hamming"]

    sweeps_dir = REPO / "data" / "sweeps"
    recs = paper_numbers.load_all(sweeps_dir)
    out: dict[str, object] = {}
    paper_numbers.section_spearman(recs, out)

    configs = sweep_report.load_configs(sweeps_dir)
    for ft in paper_numbers.TYPES:
        subset = [
            cfg
            for cfg in configs
            if cfg["filter_type"] == ft
            and cfg["window"] != "kaiser"
            and sweep_report._norm_window(cfg["window"]) in WINDOW_SIDELOBE_DB
        ]
        att = np.array(
            [WINDOW_SIDELOBE_DB[sweep_report._norm_window(c["window"])] for c in subset],
            dtype=float,
        )
        nc = np.array([c["N_chaotic"] for c in subset], dtype=float)
        rho = sweep_report._spearman_rho(att, nc)
        p_val, _ = sweep_report._exact_spearman_p(att, nc)
        expected = f"r_s={paper_numbers.fmt(rho, 3)}, p={paper_numbers.fmt(p_val, 4)}"
        assert out[f"SP.{ft}"] == expected
