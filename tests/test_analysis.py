"""tests/test_analysis.py — Tests for the analysis module."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from chaotic_pfc.analysis.stats import (
    AreaSummary,
    LmaxDistribution,
    LmaxStats,
    area_summary,
    best_chaos_preserving,
    beta_curve,
    beta_summary,
    bootstrap_confidence,
    chaos_margin,
    compare_filter_types,
    correlation_matrix,
    export_summary_json,
    lmax_distribution,
    lmax_statistics,
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

    def test_area_summary(self):
        """Validate every field of AreaSummary with known data."""
        h = np.array([[0.1, -0.2, np.nan], [np.nan, 0.0, 0.5]], dtype=float)
        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=np.arange(2, 4),
            cutoffs=np.linspace(0.1, 0.9, 3),
            window="hamming",
            filter_type="lowpass",
        )
        s = area_summary(result)
        self.assertEqual(s["n_total"], 6)
        self.assertEqual(s["n_chaotic"], 2)  # 0.1, 0.5
        self.assertEqual(s["n_periodic"], 2)  # -0.2, 0.0
        self.assertEqual(s["n_divergent"], 2)  # two NaN
        self.assertAlmostEqual(s["pct_chaotic"], 100 * 2 / 6, delta=0.15)
        self.assertAlmostEqual(s["pct_chaotic_finite"], 100 * 2 / 4, delta=0.15)

    def test_area_summary_all_divergent(self):
        h = np.full((2, 3), np.nan)
        result = SweepResult(
            h=h,
            h_std=np.zeros_like(h),
            orders=np.arange(2, 4),
            cutoffs=np.linspace(0.1, 0.9, 3),
            window="hamming",
            filter_type="lowpass",
        )
        s = area_summary(result)
        self.assertEqual(s["n_chaotic"], 0)
        self.assertEqual(s["n_periodic"], 0)
        self.assertEqual(s["n_divergent"], 6)
        self.assertAlmostEqual(s["pct_chaotic"], 0.0)
        self.assertAlmostEqual(s["pct_chaotic_finite"], 0.0)

    def test_lmax_statistics_chaotic(self):
        """Verify mean/std match known distribution and CI contains mean."""
        rng = np.random.default_rng(42)
        # 50 chaotic (positive), 30 periodic (negative), 20 NaN
        chaotic_vals = rng.uniform(0.01, 0.5, size=50)
        periodic_vals = rng.uniform(-0.5, -0.01, size=30)
        h = np.concatenate([chaotic_vals, periodic_vals, np.full(20, np.nan)])
        rng.shuffle(h)
        h = h.reshape(10, 10)

        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=np.arange(2, 12),
            cutoffs=np.linspace(0.1, 0.9, 10),
            window="hamming",
            filter_type="lowpass",
        )
        stats = lmax_statistics(result, region="chaotic", n_bootstrap=200, seed=42)
        self.assertEqual(stats["n_used"], 50)
        self.assertAlmostEqual(stats["mean"], 0.255, delta=0.1)
        self.assertAlmostEqual(stats["std"], 0.14, delta=0.1)
        self.assertGreater(stats["max"], stats["mean"])
        self.assertLess(stats["min"], stats["mean"])
        # CI should contain mean
        self.assertLessEqual(stats["ci_95_low"], stats["mean"])
        self.assertGreaterEqual(stats["ci_95_high"], stats["mean"])

    def test_lmax_statistics_all_finite(self):
        h = np.array([[0.1, -0.2], [np.nan, 0.5]], dtype=float)
        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=np.arange(2, 4),
            cutoffs=np.linspace(0.1, 0.9, 2),
            window="hamming",
            filter_type="lowpass",
        )
        stats = lmax_statistics(result, region="all_finite", n_bootstrap=50, seed=42)
        self.assertEqual(stats["n_used"], 3)

    def test_lmax_statistics_periodic(self):
        h = np.array([[0.1, -0.2, -0.3], [np.nan, -0.5, -0.1]], dtype=float)
        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=np.arange(2, 4),
            cutoffs=np.linspace(0.1, 0.9, 3),
            window="hamming",
            filter_type="lowpass",
        )
        stats = lmax_statistics(result, region="periodic", n_bootstrap=50, seed=42)
        self.assertEqual(stats["n_used"], 4)
        self.assertAlmostEqual(stats["max"], -0.1, delta=0.01)
        self.assertAlmostEqual(stats["min"], -0.5, delta=0.01)

    def test_lmax_statistics_few_points(self):
        """When fewer than 3 points, CI returns NaN."""
        h = np.array([[0.1, np.nan], [np.nan, np.nan]], dtype=float)
        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=np.arange(2, 4),
            cutoffs=np.linspace(0.1, 0.9, 2),
            window="hamming",
            filter_type="lowpass",
        )
        stats = lmax_statistics(result, region="chaotic", seed=42)
        self.assertEqual(stats["n_used"], 1)
        self.assertTrue(np.isnan(stats["ci_95_low"]))

    def test_lmax_statistics_invalid_region(self):
        h = np.array([[0.1]], dtype=float)
        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=np.array([2]),
            cutoffs=np.array([0.5]),
            window="hamming",
            filter_type="lowpass",
        )
        with self.assertRaises(ValueError):
            lmax_statistics(result, region="invalid")

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
        expected_keys = {"hist", "edges", "mean", "std", "skewness", "n"}
        for ft in FILTER_TYPES:
            self.assertIn(ft, dist)
            if dist[ft]:
                self.assertEqual(set(dist[ft].keys()), expected_keys)
                self.assertIsInstance(dist[ft]["hist"], list)
                self.assertIsInstance(dist[ft]["edges"], list)
                self.assertIsInstance(dist[ft]["n"], int)

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
