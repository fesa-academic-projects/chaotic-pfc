"""Coverage for the manuscript tooling.

Exercise the bulk of ``chaotic_pfc.paper.numbers`` (``compute`` and every
section), the ``paper.figures`` generators and the ``run paper-figures`` CLI on
a synthetic tree that mirrors the canonical layout of the versioned sweeps (six
fixed windows plus 25 Kaiser betas per filter type). The grids are reduced so
the tests stay in the fast suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import chaotic_pfc.paper.figures as paper_figures
from chaotic_pfc.cli import main
from chaotic_pfc.paper.numbers import compute, load_all, sensitivity_report
from tests._paper_synthetic import build_synthetic_sweeps

# The synthetic tree renders figures and runs the full ``compute`` pass, so the
# module is kept out of the fast suite (``make test-fast``); the CI coverage run
# executes the whole suite and therefore still measures these modules.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def sweeps_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_synthetic_sweeps(tmp_path_factory.mktemp("synthetic-sweeps"))


# ── paper numbers ───────────────────────────────────────────────────────────


def test_compute_covers_every_section(sweeps_tree: Path) -> None:
    out = compute(sweeps_tree, extras=False)
    # One label group per section: recalculation, tables, Spearman, totals,
    # maps, density, concordance, Kaiser, robustness, cost and DC gain.
    for key in (
        "RECALC.n_fallback",
        "T2.lowpass.1.Nc",
        "SP.lowpass",
        "TOT.chaotic",
        "MAP.lowpass.max_wc_by_Nz",
        "DEN.max_global",
        "CON.lowpass.pooled_Nz_le10",
        "KAI.lowpass.all",
        "ROB.all.pct_mean_above_2se",
        "COST.sum_n_iters_used",
        "GAIN.n_absG_lt_0.1",
        "GAIN.highpass.Nz_x_wc",
    ):
        assert key in out, f"missing label {key}"
    # 193 core labels for the canonical 124-configuration layout, plus
    # ``COST.regen_wall_seconds`` when the local regeneration cache
    # (``.cache/paper/regen_seconds.txt``) is present — a clean clone has none.
    assert len(set(out) - {"COST.regen_wall_seconds"}) == 193
    assert len(out) in (193, 194)
    assert len(load_all(sweeps_tree)) == 124


def test_compute_cli_writes_json(sweeps_tree: Path, tmp_path: Path) -> None:
    target = tmp_path / "numbers.json"
    assert main(["paper", "numbers", "--sweeps", str(sweeps_tree), "--json", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["TOT.chaotic"] == compute(sweeps_tree, extras=False)["TOT.chaotic"]


def test_paper_numbers_missing_sweeps_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "paper",
            "numbers",
            "--sweeps",
            str(tmp_path / "empty"),
            "--json",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 1
    assert capsys.readouterr().err.startswith("error:")


def test_sensitivity_report(sweeps_tree: Path) -> None:
    report = sensitivity_report(load_all(sweeps_tree))
    assert "Table III order (28 rows)" in report
    assert report.count("\n") > 30


# ── paper figures ───────────────────────────────────────────────────────────


def test_fig_kaiser_beta(
    sweeps_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paper_figures, "DATA_DIR", sweeps_tree)
    paper_figures.fig_kaiser_beta(tmp_path)
    assert (tmp_path / "fig_kaiser_beta.svg").is_file()


def test_fig_chaotic_density_norm(
    sweeps_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paper_figures, "DATA_DIR", sweeps_tree)
    paper_figures.fig_chaotic_density_norm(tmp_path)
    assert (tmp_path / "fig_chaotic_density_norm.svg").is_file()


# ── run paper-figures (camera-ready panels) ─────────────────────────────────


def test_run_paper_figures_cli(sweeps_tree: Path, tmp_path: Path) -> None:
    rc = main(
        [
            "run",
            "paper-figures",
            "--data-dir",
            str(sweeps_tree),
            "--output-dir",
            str(tmp_path),
            "--only",
            "all",
            "--formats",
            "svg",
        ]
    )
    assert rc == 0
    assert (tmp_path / "simac" / "simac_fig1a_hamming_highpass.svg").is_file()
    assert (tmp_path / "simac" / "simac_fig1_legenda.svg").is_file()
    assert (tmp_path / "jcis" / "kaiser_b13p5_lowpass_classification_interleaved.svg").is_file()
    assert (tmp_path / "jcis" / "fig_chaotic_density.svg").is_file()
