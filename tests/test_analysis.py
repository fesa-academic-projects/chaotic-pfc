"""tests/test_analysis.py — Tests for the analysis module."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from chaotic_pfc.analysis.stats import (
    best_chaos_preserving,
    beta_curve,
    beta_summary,
    bootstrap_confidence,
    chaos_margin,
    compare_filter_types,
    correlation_matrix,
    export_summary_json,
    lmax_distribution,
    optimal_parameters,
    summary_table,
    transition_boundary,
)
from chaotic_pfc.analysis.sweep import FILTER_TYPES, SweepResult, save_sweep


class TestAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        # Write a few dummy sweeps
        rng = np.random.default_rng(0)
        for w, ft in [("hamming", "lowpass"), ("hamming", "highpass"), ("hann", "lowpass")]:
            h = rng.uniform(-0.5, 0.5, size=(3, 4))
            h[0, 0] = np.nan
            result = SweepResult(
                h=h,
                h_std=np.abs(h) * 0.1,
                orders=np.arange(2, 5),
                cutoffs=np.linspace(0.1, 0.9, 4),
                window=w,
                filter_type=ft,
                metadata={"Nitera": 10, "Nmap": 50},
            )
            out = cls.root / f"{w} ({ft})" / "variables_lyapunov.npz"
            save_sweep(result, out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_summary_table(self):
        rows = summary_table(self.root)
        self.assertGreaterEqual(len(rows), 3)
        for row in rows:
            self.assertIn("pct_chaotic", row)
            self.assertIn("pct_periodic", row)
            self.assertIn("pct_divergent", row)
            self.assertAlmostEqual(
                row["pct_chaotic"] + row["pct_periodic"] + row["pct_divergent"], 100.0, delta=0.15
            )

    def test_best_chaos_preserving(self):
        top = best_chaos_preserving(self.root, top_n=2)
        self.assertEqual(len(top), 2)
        self.assertGreaterEqual(top[0]["pct_chaotic"], top[1]["pct_chaotic"])

    def test_compare_filter_types(self):
        cmp = compare_filter_types(self.root)
        self.assertIn("lowpass", cmp)
        self.assertIn("highpass", cmp)

    def test_optimal_parameters(self):
        params = optimal_parameters(self.root, window="hamming", top_n=3)
        self.assertGreater(len(params), 0)
        for p in params:
            self.assertEqual(p["window"], "hamming")

    def test_export_summary_json(self):
        out = self.root / "out.json"
        path = export_summary_json(self.root, out)
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)

    def test_beta_summary_empty(self):
        bs = beta_summary(self.root)
        self.assertEqual(bs, {})

    def test_beta_curve_empty(self):
        betas, _pct = beta_curve(self.root, "lowpass")
        self.assertEqual(len(betas), 0)

    def test_lmax_distribution(self):
        dist = lmax_distribution(self.root)
        self.assertIsInstance(dist, dict)
        for ft in FILTER_TYPES:
            self.assertIn(ft, dist)

    def test_transition_boundary(self):
        orders, cutoffs = transition_boundary(self.root, filter_type="lowpass")
        self.assertGreaterEqual(len(orders), 0)
        self.assertEqual(len(orders), len(cutoffs))

    def test_transition_boundary_no_data(self):
        orders, _cutoffs = transition_boundary(self.root, filter_type="bandstop")
        self.assertEqual(len(orders), 0)

    def test_chaos_margin(self):
        orders, widths = chaos_margin(self.root, filter_type="lowpass")
        self.assertGreaterEqual(len(orders), 0)
        self.assertEqual(len(orders), len(widths))

    def test_chaos_margin_no_data(self):
        orders, _widths = chaos_margin(self.root, filter_type="bandstop")
        self.assertEqual(len(orders), 0)

    def test_correlation_matrix(self):
        corr = correlation_matrix(self.root)
        self.assertIn("n", corr)
        self.assertIn("order_vs_lmax", corr)
        self.assertIn("cutoff_vs_lmax", corr)
        self.assertGreater(corr["n"], 0)

    def test_bootstrap_confidence(self):
        ci = bootstrap_confidence(self.root, n_bootstrap=100)
        self.assertIsInstance(ci, dict)
        for ft in FILTER_TYPES:
            self.assertIn(ft, ci)

    def test_bootstrap_confidence_empty(self):
        with TemporaryDirectory() as td:
            ci = bootstrap_confidence(Path(td), n_bootstrap=10)
            self.assertIsInstance(ci, dict)
            # All filter types present with empty entries
            for ft in FILTER_TYPES:
                self.assertIn(ft, ci)
                self.assertEqual(ci[ft], {})


if __name__ == "__main__":
    unittest.main()
