"""tests/test_latex_export.py — Tests for the LaTeX exporter module."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from chaotic_pfc.analysis.latex_export import (
    export_extended_top_k_table,
    export_full_ranking_table,
    export_sweet_spots_table,
    export_top_k_table,
)


def _count_data_rows(tex: str) -> int:
    """Count data rows (lines containing '&' minus the header row)."""
    return sum(1 for line in tex.splitlines() if " & " in line) - 1


def _make_config_rank(
    rank: int,
    filter_type: str,
    window: str,
    n_chaotic: int = 100,
    pct_chaotic: float = 25.0,
    pct_chaotic_finite: float = 80.0,
    lmax_mean: float = 0.15,
    lmax_max: float = 0.5,
    lmax_std: float = 0.05,
    ci_low: float = 0.14,
    ci_high: float = 0.16,
    beta: float | None = None,
) -> dict:
    return {
        "rank": rank,
        "filter_type": filter_type,
        "window": window,
        "n_chaotic": n_chaotic,
        "pct_chaotic": pct_chaotic,
        "pct_chaotic_finite": pct_chaotic_finite,
        "lmax_mean": lmax_mean,
        "lmax_max": lmax_max,
        "lmax_std": lmax_std,
        "lmax_ci_95_low": ci_low,
        "lmax_ci_95_high": ci_high,
        "beta": beta,
    }


def _make_sweet_spot(
    filter_type: str,
    window: str,
    n_z: int = 10,
    omega_c: float = 0.3,
    lmax: float = 0.5,
    ci_low: float | None = 0.45,
    ci_high: float | None = 0.55,
) -> dict:
    return {
        "filter_type": filter_type,
        "window": window,
        "n_z": n_z,
        "omega_c": omega_c,
        "lmax": lmax,
        "lmax_ci_95_low": ci_low,
        "lmax_ci_95_high": ci_high,
    }


class TestLatexExport(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _read_tex(self, path):
        return Path(path).read_text(encoding="utf-8")

    # ── Top-k table (Categoria A) ──────────────────────────────────────

    def test_top_k_table_pt(self):
        top_k = {
            "lowpass": [
                _make_config_rank(1, "lowpass", "hamming", n_chaotic=200, pct_chaotic=50.0),
                _make_config_rank(2, "lowpass", "hann", n_chaotic=150, pct_chaotic=37.5),
            ],
        }
        path = self.root / "top_k.tex"
        export_top_k_table(top_k, path, lang="pt")
        tex = self._read_tex(path)
        self.assertIn(r"\begin{tabular}", tex)
        self.assertIn(r"\end{tabular}", tex)
        self.assertIn("Passa-baixa", tex)
        self.assertIn("Pontos caóticos", tex)
        self.assertIn("Hamming", tex)
        self.assertIn("200", tex)
        num_datarows = _count_data_rows(tex)
        self.assertEqual(num_datarows, 2)

    def test_top_k_table_en(self):
        top_k = {
            "highpass": [
                _make_config_rank(1, "highpass", "boxcar", n_chaotic=80),
            ],
        }
        path = self.root / "top_k_en.tex"
        export_top_k_table(top_k, path, lang="en")
        tex = self._read_tex(path)
        self.assertIn("Highpass", tex)
        self.assertIn("Chaotic points", tex)
        self.assertIn("Rectangular", tex)

    def test_top_k_table_caption_and_label(self):
        top_k = {
            "bandpass": [
                _make_config_rank(1, "bandpass", "hamming", n_chaotic=50),
            ],
        }
        path = self.root / "top_k_cl.tex"
        export_top_k_table(
            top_k,
            path,
            caption_key="analysis.tables.top_k.caption",
            label="tab:top_k",
            lang="en",
        )
        tex = self._read_tex(path)
        self.assertIn(r"\caption{Top-3 windows per filter type (chaotic area).}", tex)
        self.assertIn(r"\label{tab:top_k}", tex)

    # ── Extended top-k table (Categoria B) ─────────────────────────────

    def test_extended_top_k_table(self):
        top_k = {
            "lowpass": [
                _make_config_rank(1, "lowpass", "hamming", n_chaotic=200, pct_chaotic_finite=90.0),
            ],
            "bandstop": [
                _make_config_rank(
                    1, "bandstop", "kaiser_5.00", n_chaotic=30, pct_chaotic_finite=50.0, beta=5.0
                ),
            ],
        }
        path = self.root / "extended.tex"
        export_extended_top_k_table(top_k, path, lang="pt")
        tex = self._read_tex(path)
        self.assertIn(r"\begin{tabular}", tex)
        self.assertIn("Passa-baixa", tex)
        self.assertIn("Rejeita-faixa", tex)
        self.assertIn(r"$\overline{\lambda}_{\max}$", tex)
        self.assertIn(r"$\sigma(\lambda_{\max})$", tex)
        self.assertIn("IC 95", tex)
        # Kaiser label
        self.assertIn(r"Kaiser($\beta=5.00$)", tex)
        # CI
        self.assertIn("[0.14, 0.16]", tex)
        num_datarows = _count_data_rows(tex)
        self.assertEqual(num_datarows, 2)

    # ── Full ranking (Categoria C.1) ───────────────────────────────────

    def test_full_ranking_longtable(self):
        rank_data = [
            _make_config_rank(i, "lowpass", f"w{i}", n_chaotic=300 - i * 10) for i in range(1, 6)
        ]
        path = self.root / "full_ranking.tex"
        export_full_ranking_table(rank_data, path, lang="en")
        tex = self._read_tex(path)
        self.assertIn(r"\begin{longtable}", tex)
        self.assertIn(r"\end{longtable}", tex)
        self.assertNotIn(r"\begin{table}", tex)
        self.assertIn("Full ranking", tex)
        num_datarows = _count_data_rows(tex)
        self.assertEqual(num_datarows, 5)

    # ── Sweet spots (Categoria C.2) ────────────────────────────────────

    def test_sweet_spots_table(self):
        ss = {
            "lowpass": _make_sweet_spot("lowpass", "hamming", n_z=12, omega_c=0.3142, lmax=0.523),
            "highpass": _make_sweet_spot(
                "highpass", "kaiser_3.00", n_z=8, omega_c=0.5, lmax=0.412, ci_low=None, ci_high=None
            ),
        }
        path = self.root / "sweet_spots.tex"
        export_sweet_spots_table(ss, path, lang="pt")
        tex = self._read_tex(path)
        self.assertIn("Passa-baixa", tex)
        self.assertIn("Passa-alta", tex)
        self.assertIn("12", tex)
        self.assertIn("0.3142", tex)
        self.assertIn("0.523", tex)
        self.assertIn(r"Kaiser($\beta=3.00$)", tex)
        self.assertIn("---", tex)  # CI None → ---
        num_datarows = _count_data_rows(tex)
        self.assertEqual(num_datarows, 2)

    def test_sweet_spots_encoding(self):
        """PT output with accents must be valid UTF-8."""
        ss = {"bandpass": _make_sweet_spot("bandpass", "hann", n_z=5)}
        path = self.root / "encoding.tex"
        export_sweet_spots_table(ss, path, lang="pt")
        tex = self._read_tex(path)
        self.assertIn("Passa-faixa", tex)
        self.assertIn("Hann", tex)


if __name__ == "__main__":
    unittest.main()
