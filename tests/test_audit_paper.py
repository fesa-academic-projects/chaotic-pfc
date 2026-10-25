"""tests/test_audit_paper.py: unit tests for the manuscript audit.

The audit lives in :mod:`chaotic_pfc.paper.audit`. Every check is exercised
with a small synthetic fixture: one case that passes and one that fails.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import pytest

from chaotic_pfc.paper import audit


def _findings(result) -> list:
    return result.findings


def _levels(result) -> list[str]:
    return [finding.level for finding in result.findings]


class TestCitationKeys(unittest.TestCase):
    GOOD_TEX = r"""
\documentclass[journal]{IEEEtran}
\title{A Toy Manuscript}
\begin{document}
See \cite{alpha, beta} and also \cite{gamma}.
\end{document}
"""
    GOOD_BIB = r"""
@article{alpha,
  title = {Alpha},
  doi = {10.1000/alpha}
}
@article{beta,
  title = {Beta},
  doi = {10.1000/beta}
}
@book{gamma,
  title = {Gamma},
  doi = {10.1000/gamma}
}
"""

    def test_all_keys_defined(self):
        result = audit.check_citation_keys(self.GOOD_TEX, self.GOOD_BIB)
        self.assertEqual(_findings(result), [])
        self.assertIn("parsed 3 bib entries", result.coverage)

    def test_undefined_key_is_error(self):
        tex = self.GOOD_TEX.replace("\\cite{gamma}", "\\cite{delta}")
        result = audit.check_citation_keys(tex, self.GOOD_BIB)
        errors = [f for f in result.findings if f.level == "ERROR"]
        self.assertEqual(len(errors), 1)
        self.assertIn("delta", errors[0].message)
        self.assertIsNotNone(errors[0].line)

    def test_unused_key_is_warn(self):
        bib = self.GOOD_BIB + "\n@misc{epsilon,\n  title = {Epsilon},\n  doi = {10.1000/e}\n}\n"
        result = audit.check_citation_keys(self.GOOD_TEX, bib)
        self.assertEqual(_levels(result), ["WARN"])
        self.assertIn("epsilon", result.findings[0].message)

    def test_comment_is_ignored(self):
        tex = self.GOOD_TEX.replace("\\end{document}", "% \\cite{ghost}\n\\end{document}")
        result = audit.check_citation_keys(tex, self.GOOD_BIB)
        self.assertEqual(_findings(result), [])

    def test_duplicate_key_is_warn(self):
        bib = self.GOOD_BIB + "\n@article{alpha,\n  title = {Alpha again},\n  doi = {10.1/a}\n}\n"
        result = audit.check_citation_keys(self.GOOD_TEX, bib)
        self.assertEqual(_levels(result), ["WARN"])
        self.assertIn("duplicate", result.findings[0].message)


class TestDois(unittest.TestCase):
    def test_all_dois_present(self):
        result = audit.check_dois(TestCitationKeys.GOOD_BIB, frozenset())
        self.assertEqual(_findings(result), [])
        self.assertIn("checked 3 entries", result.coverage)

    def test_missing_doi_is_error(self):
        result = audit.check_dois("@article{alpha,\n  title = {Alpha}\n}\n", frozenset())
        self.assertEqual(_levels(result), ["ERROR"])
        self.assertIn("alpha", result.findings[0].message)

    def test_whitelist_exempts(self):
        result = audit.check_dois("@article{alpha,\n  title = {Alpha}\n}\n", frozenset({"alpha"}))
        self.assertEqual(_findings(result), [])


class TestStructure(unittest.TestCase):
    GOOD_TEX = r"""
\documentclass[journal]{IEEEtran}
\title{A Toy Manuscript}
\begin{document}
\section{Introduction}\label{sec:intro}
See Section~\ref{sec:intro} and \eqref{eq:map}.
\begin{equation}\label{eq:map}
  x_{n+1} = a - x_n^2
\end{equation}
\end{document}
"""

    def test_well_formed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = audit.check_structure(self.GOOD_TEX, Path(tmp))
        self.assertEqual(_findings(result), [])

    def test_optional_short_title_is_accepted(self):
        tex = self.GOOD_TEX.replace("\\title{A Toy Manuscript}", "\\title[Short]{A Toy Manuscript}")
        with tempfile.TemporaryDirectory() as tmp:
            result = audit.check_structure(tex, Path(tmp))
        self.assertEqual(_findings(result), [])

    def test_missing_title_is_error(self):
        tex = self.GOOD_TEX.replace("\\title{A Toy Manuscript}", "")
        with tempfile.TemporaryDirectory() as tmp:
            result = audit.check_structure(tex, Path(tmp))
        self.assertEqual(_levels(result), ["ERROR"])
        self.assertIn("title", result.findings[0].message)

    def test_undefined_ref_is_error(self):
        tex = self.GOOD_TEX.replace("\\ref{sec:intro}", "\\ref{sec:missing}")
        with tempfile.TemporaryDirectory() as tmp:
            result = audit.check_structure(tex, Path(tmp))
        errors = [f for f in result.findings if f.level == "ERROR"]
        self.assertEqual(len(errors), 1)
        self.assertIn("sec:missing", errors[0].message)

    def test_unused_label_is_warn(self):
        tex = self.GOOD_TEX.replace("\\label{eq:map}", "\\label{eq:unused}")
        with tempfile.TemporaryDirectory() as tmp:
            result = audit.check_structure(tex, Path(tmp))
        warns = [f for f in result.findings if f.level == "WARN"]
        self.assertEqual(len(warns), 1)
        self.assertIn("eq:unused", warns[0].message)

    def test_unresolved_path_is_error(self):
        tex = self.GOOD_TEX.replace(
            "\\end{document}", "\\includegraphics{Figures/missing.png}\n\\end{document}"
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = audit.check_structure(tex, Path(tmp))
        errors = [f for f in result.findings if f.level == "ERROR"]
        self.assertEqual(len(errors), 1)
        self.assertIn("missing.png", errors[0].message)

    def test_resolved_paths_pass_and_extensionless_resolve(self):
        tex = self.GOOD_TEX.replace(
            "\\end{document}",
            "\\includegraphics{Figures/plot}\n\\input{sec/intro}\n\\bibliography{refs}\n\\end{document}",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Figures").mkdir()
            (root / "Figures" / "plot.png").write_bytes(b"png")
            (root / "sec").mkdir()
            (root / "sec" / "intro.tex").write_text("intro")
            (root / "refs.bib").write_text("@misc{x, doi={1}}\n")
            result = audit.check_structure(tex, root)
        self.assertEqual(_findings(result), [])
        self.assertIn("3/3 paths resolved", result.coverage)


class TestLog(unittest.TestCase):
    def test_clean_log(self):
        result = audit.check_log([("paper.log", "This is pdfTeX, Version 3.14\nOutput written\n")])
        self.assertEqual(_findings(result), [])
        self.assertIn("scanned 2 log lines", result.coverage)

    def test_undefined_citation_is_error(self):
        text = "LaTeX Warning: Citation `ghost' on page 1 undefined on input line 42.\n"
        result = audit.check_log([("paper.log", text)])
        self.assertEqual(_levels(result), ["ERROR"])
        self.assertIn("ghost", result.findings[0].message)

    def test_overfull_is_warn_with_count(self):
        text = (
            "Overfull \\hbox (30.0pt too wide) in paragraph at lines 10--12\n"
            "Overfull \\hbox (12.5pt too wide) in paragraph at lines 20--21\n"
        )
        result = audit.check_log([("paper.log", text)])
        self.assertEqual(_levels(result), ["WARN", "WARN", "WARN"])
        self.assertIn("2 overfull", result.findings[0].message)
        self.assertIn("30.0pt", result.findings[1].message)


class TestFigureLanguage(unittest.TestCase):
    def _tex(self, include: str) -> str:
        return "\\documentclass{journal}\n\\begin{document}\n" + include + "\n\\end{document}\n"

    def test_inserted_svg_is_scanned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fig.svg").write_text("<svg><text>Chaos density</text></svg>")
            result = audit.check_figure_language(
                self._tex("\\includegraphics{fig.svg}"), root, root, all_figures=False
            )
        self.assertEqual(_findings(result), [])
        self.assertIn("verified 1 vector sources", result.coverage)

    def test_inserted_svg_with_portuguese_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fig.svg").write_text("<!-- Densidade de Caos -->")
            result = audit.check_figure_language(
                self._tex("\\includegraphics{fig.svg}"), root, root, all_figures=False
            )
        self.assertEqual(_levels(result), ["ERROR"])
        self.assertIn("Densidade", result.findings[0].message)

    def test_raster_without_vector_source_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fig.png").write_bytes(b"png")
            result = audit.check_figure_language(
                self._tex("\\includegraphics{fig.png}"), root, root, all_figures=False
            )
        self.assertEqual(_levels(result), ["WARN"])
        self.assertIn("not automatically verifiable", result.findings[0].message)

    def test_raster_with_sibling_vector_is_scanned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fig.png").write_bytes(b"png")
            (root / "fig.svg").write_text("<!-- Densidade de Caos -->")
            result = audit.check_figure_language(
                self._tex("\\includegraphics{fig.png}"), root, root, all_figures=False
            )
        self.assertEqual(_levels(result), ["ERROR"])
        self.assertIn("Densidade", result.findings[0].message)

    def test_all_figures_scans_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            figures = root / "figures"
            figures.mkdir()
            (figures / "a.svg").write_text("<svg><text>Chaos density</text></svg>")
            (figures / "b.svg").write_text("<!-- Caos -->")
            result = audit.check_figure_language("", root, figures, all_figures=True)
        self.assertEqual(_levels(result), ["ERROR"])
        self.assertIn("scanned 2 SVGs", result.coverage)
        self.assertIn("1 flagged", result.coverage)

    def test_entity_encoded_accent_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fig.svg").write_text("<text>ca&#243;tica</text>")
            result = audit.check_figure_language(
                self._tex("\\includegraphics{fig.svg}"), root, root, all_figures=False
            )
        self.assertIn("caótica", result.findings[0].message)


class TestConfig(unittest.TestCase):
    @pytest.mark.slow
    def test_real_config_locates_known_constants(self):
        result = audit.check_config()
        self.assertIn("located", result.coverage)
        # These seven exist in config.py and match the article.
        self.assertFalse(any(f.level == "ERROR" for f in result.findings))


class TestTableTwo(unittest.TestCase):
    """C6 checks the current eight-column Table II."""

    _TABLE = r"""
\begin{table*}[!t]
\caption{Ranking\label{tab:ranking}}
\centering
\begin{tabular}{rlrrrrrr}
\toprule
Rank & Filter & Window & $N_c$ & $\rho_c^{\text{abs}}$ (\%) &
$\rho_c^{\text{cond}}$ (\%) & $\bar\lambda_1$ & $\sigma_{\lambda_1}$ \\
\midrule
1 & Lowpass & Kaiser & 654 & 16.4 & 55.3 & 0.0423 & 0.0186 \\
2 & Bandpass & Kaiser & 310 & 7.8 & 9.5 & 0.0435 & 0.0272 \\
3 & Highpass & Kaiser & 102 & 5.1 & 5.7 & 0.0406 & 0.0284 \\
\bottomrule
\end{tabular}
\end{table*}
"""

    def test_good_table_passes(self):
        result = audit.check_tables(self._TABLE)
        self.assertEqual(_levels(result), [])
        self.assertIn("parsed 3 rows", result.coverage)

    def test_rho_abs_error_is_reported(self):
        tex = self._TABLE.replace("654 & 16.4", "654 & 16.5")
        result = audit.check_tables(tex)
        self.assertEqual(_levels(result), ["ERROR"])
        self.assertIn("R1", result.findings[0].message)

    def test_two_rows_out_of_order_are_reported(self):
        tex = self._TABLE.replace(
            "2 & Bandpass & Kaiser & 310 & 7.8 & 9.5 & 0.0435 & 0.0272 \\\\\n"
            "3 & Highpass & Kaiser & 102 & 5.1 & 5.7 & 0.0406 & 0.0284",
            "2 & Highpass & Kaiser & 102 & 5.1 & 5.7 & 0.0406 & 0.0284 \\\\\n"
            "3 & Bandpass & Kaiser & 310 & 7.8 & 9.5 & 0.0435 & 0.0272",
        )
        result = audit.check_tables(tex)
        self.assertGreaterEqual(_levels(result).count("ERROR"), 1)
        self.assertTrue(any("rank order" in finding.message for finding in result.findings))

    def test_missing_ranking_table_is_warn(self):
        tex = self._TABLE.replace("\\label{tab:ranking}", "\\label{tab:other}")
        result = audit.check_tables(tex)
        self.assertEqual(_levels(result), ["WARN"])
        self.assertIn("tab:ranking", result.findings[0].message)


class TestRender(unittest.TestCase):
    def test_summary_and_coverage(self):
        results = {
            "IN": audit.CheckResult([], "tex=ok, bib=ok"),
            "C1": audit.CheckResult(
                [
                    audit.Finding("ERROR", "C1", "boom", line=3),
                    audit.Finding("WARN", "C1", "soft", line=4),
                ],
                "parsed 1 bib entries",
            ),
        }
        rendered = audit.render_text(results)
        self.assertIn("coverage: parsed 1 bib entries", rendered)
        self.assertIn("Summary: 1 ERROR, 1 WARN", rendered)
        self.assertIn("line 3", rendered)


class TestTextNumbers(unittest.TestCase):
    """C8: numbers cited in the Results/Final Remarks versus paper numbers."""

    _TEX = (
        "\\section{Results and Discussion}\n"
        "The best lowpass configuration has $N_c=980$ and "
        "$\\rho_c^{\\text{fin}}=77.6\\%$.\n"
        "\\section{Final Remarks}\n"
        "The whole run took $498$\\,s.\n"
    )

    _NUMBERS: ClassVar[dict[str, str]] = {
        "T2.lowpass.1.Nc": "980",
        "T2.lowpass.1.rho_cond": "77.6",
        "COST.regen_wall_seconds": "497.775142663",
    }

    def test_numbers_matching_labels_pass(self):
        result = audit.check_text_numbers(self._TEX, self._NUMBERS)
        self.assertEqual(_levels(result), [])
        self.assertIn("3 distinct", result.coverage)

    def test_changed_number_is_error(self):
        tex = self._TEX.replace("$N_c=980$", "$N_c=987$")
        result = audit.check_text_numbers(tex, self._NUMBERS)
        self.assertEqual(_levels(result), ["ERROR"])

    def test_citation_key_year_is_not_a_text_number(self):
        tex = "\\section{Results and Discussion}\nSee \\cite{Sprott2003}."
        result = audit.check_text_numbers(tex, {})
        self.assertEqual(_levels(result), [])

    def test_thousands_separator_is_one_token(self):
        tex = (
            "\\section{Results and Discussion}\n"
            "None of the 182,892 grid points and 2,790 chaotic points.\n"
        )
        numbers = {"GAIN.n_absG_lt_0.1": "182892", "TOT.highpass.chaotic": "2790"}
        result = audit.check_text_numbers(tex, numbers)
        self.assertEqual(_levels(result), [])

    def test_reference_exception_is_scoped_to_validation(self):
        numbers = {"VAL.ns1.single_vector": "0.4194"}
        in_validation = (
            "\\section{Results and Discussion}\n"
            "\\subsection{Validation of the Estimator}\n"
            "The literature value is 0.4192.\n"
        )
        in_maps = (
            "\\section{Results and Discussion}\n"
            "\\subsection{Classification Maps}\n"
            "The literature value is 0.4192.\n"
        )
        self.assertEqual(_levels(audit.check_text_numbers(in_validation, numbers)), [])
        self.assertEqual(_levels(audit.check_text_numbers(in_maps, numbers)), ["ERROR"])

    def test_selection_band_exception_is_scoped_to_spectrum(self):
        in_spectrum = (
            "\\section{Results and Discussion}\n"
            "\\subsection{Spectrum of the Generated Signals}\n"
            "The band is 0.125 to 0.875.\n"
        )
        in_maps = (
            "\\section{Results and Discussion}\n"
            "\\subsection{Classification Maps}\n"
            "The band is 0.125 to 0.875.\n"
        )
        self.assertEqual(_levels(audit.check_text_numbers(in_spectrum, {})), [])
        self.assertEqual(_levels(audit.check_text_numbers(in_maps, {})), ["ERROR", "ERROR"])


if __name__ == "__main__":
    unittest.main()
