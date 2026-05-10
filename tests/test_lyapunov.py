"""tests/test_lyapunov.py — Unit tests for Lyapunov exponent computation."""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from chaotic_pfc.dynamics.lyapunov import (
    fixed_point_stability,
    lyapunov_henon2d,
    lyapunov_henon2d_ensemble,
    lyapunov_max,
    lyapunov_max_ensemble,
)


class TestLyapunov(unittest.TestCase):
    def test_lyapunov_max_returns_expected_keys(self):
        result = lyapunov_max(Nitera=200, Ndiscard=100)
        self.assertIsNotNone(result.lyapunov_max)
        self.assertIsNotNone(result.all_exponents)
        self.assertEqual(len(result.all_exponents), 4)

    def test_positive_exponent(self):
        result = lyapunov_max(pole_radius=0.0, Nitera=500, Ndiscard=200)
        self.assertGreater(result.lyapunov_max, 0)

    def test_fixed_point_stability(self):
        info = fixed_point_stability()
        self.assertIn("eigenvalues", info)
        self.assertEqual(len(info["eigenvalues"]), 4)

    def test_henon2d_chaotic(self):
        res = lyapunov_henon2d(Nitera=300, Ndiscard=100)
        self.assertGreater(res.lyapunov_max, 0)

    def test_henon2d_fixed_points(self):
        res = lyapunov_henon2d(Nitera=100, Ndiscard=50)
        self.assertIsNotNone(res.fixed_point_p)
        self.assertIsNotNone(res.fixed_point_n)
        self.assertAlmostEqual(res.fixed_point_p[0], 0.8838962679253065, places=4)
        self.assertAlmostEqual(res.fixed_point_n[0], -1.5838962679253065, places=4)


class TestLyapunovEnsemble(unittest.TestCase):
    def test_henon2d_shapes(self):
        r = lyapunov_henon2d_ensemble(Nitera=50, Ndiscard=30, n_initial=4, seed=0)
        self.assertGreaterEqual(len(r.lmax_per_ci), 4)

    def test_max_ensemble_shapes(self):
        r = lyapunov_max_ensemble(Nitera=50, Ndiscard=30, n_initial=3, seed=0)
        self.assertGreaterEqual(len(r.lmax_per_ci), 3)
        self.assertGreaterEqual(r.n_chaotic + r.n_stable, 1)

    def test_henon2d_ic_bounds(self):
        r = lyapunov_henon2d_ensemble(Nitera=50, Ndiscard=30, n_initial=5, seed=1, perturbation=0.1)
        lmax_arr = np.array(r.lmax_per_ci)
        self.assertEqual(lmax_arr.shape, (5,))
        self.assertTrue(np.isfinite(r.mean_lmax))

    def test_ensemble_aggregates_match(self):
        r = lyapunov_henon2d_ensemble(Nitera=100, Ndiscard=50, n_initial=6, seed=2)
        self.assertTrue(np.isfinite(r.mean_lmax))

    def test_determinism_with_seed(self):
        r1 = lyapunov_max_ensemble(Nitera=100, Ndiscard=50, n_initial=3, seed=42)
        r2 = lyapunov_max_ensemble(Nitera=100, Ndiscard=50, n_initial=3, seed=42)
        np.testing.assert_array_equal(r1.lmax_per_ci, r2.lmax_per_ci)
        self.assertAlmostEqual(r1.mean_lmax, r2.mean_lmax, places=10)

    def test_chaotic_average(self):
        r = lyapunov_max_ensemble(Nitera=500, Ndiscard=200, n_initial=4, seed=3, pole_radius=0.0)
        self.assertGreater(r.mean_lmax, 0)

    def test_csv_roundtrip(self):
        r = lyapunov_henon2d_ensemble(Nitera=100, Ndiscard=50, n_initial=4, seed=0)
        with TemporaryDirectory() as td:
            path = Path(td) / "out.csv"
            r.to_csv(path)
            self.assertTrue(path.exists())
            with path.open() as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(header[0], "ci")
                self.assertIn("lmax", header)
                rows = [next(reader) for _ in range(4)]
                self.assertEqual([row[0] for row in rows], ["1", "2", "3", "4"])

    def test_csv_roundtrip_4d(self):
        r = lyapunov_max_ensemble(Nitera=100, Ndiscard=50, n_initial=3, seed=0)
        with TemporaryDirectory() as td:
            path = Path(td) / "out4d.csv"
            r.to_csv(path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
