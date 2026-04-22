"""tests/test_maps.py — Unit tests for all modules."""
import unittest
import numpy as np

from chaotic_pfc.maps import henon_standard, henon_generalised, henon_filtered
from chaotic_pfc.signals import binary_message
from chaotic_pfc.transmitter import transmit
from chaotic_pfc.channel import ideal_channel, fir_channel
from chaotic_pfc.receiver import receive
from chaotic_pfc.spectral import psd_normalised
from chaotic_pfc.lyapunov import lyapunov_max, fixed_point_stability, lyapunov_henon2d


class TestHenonStandard(unittest.TestCase):
    def test_output_shape(self):
        X, Y = henon_standard(100)
        self.assertEqual(X.shape, (101,))

    def test_initial_condition(self):
        X, Y = henon_standard(10, x0=1.0, y0=2.0)
        self.assertAlmostEqual(X[0], 1.0)
        self.assertAlmostEqual(Y[0], 2.0)

    def test_first_iteration(self):
        X, Y = henon_standard(1, x0=0.0, y0=0.0)
        self.assertAlmostEqual(X[1], 1.0)
        self.assertAlmostEqual(Y[1], 0.0)

    def test_bounded(self):
        X, Y = henon_standard(5000)
        self.assertTrue(np.all(np.abs(X) < 5.0))


class TestHenonGeneralised(unittest.TestCase):
    def test_shape(self):
        X, Y = henon_generalised(50)
        self.assertEqual(X.shape, (51,))


class TestHenonFiltered(unittest.TestCase):
    def test_c0_1_c1_0_matches_generalised(self):
        X_f, Y_f = henon_filtered(500, c0=1.0, c1=0.0)
        X_g, Y_g = henon_generalised(500)
        np.testing.assert_allclose(X_f, X_g, atol=1e-12)
        np.testing.assert_allclose(Y_f, Y_g, atol=1e-12)


class TestBinaryMessage(unittest.TestCase):
    def test_length(self):
        m = binary_message(1000, period=20)
        self.assertEqual(len(m), 1000)

    def test_values(self):
        m = binary_message(200, period=20)
        self.assertTrue(set(np.unique(m)).issubset({-1.0, 1.0}))

    def test_periodicity(self):
        m = binary_message(100, period=10)
        np.testing.assert_array_equal(m[:10], m[10:20])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            binary_message(100, period=7)


class TestPipeline(unittest.TestCase):
    def test_ideal_roundtrip(self):
        N = 50_000; mu = 0.01
        m = binary_message(N)
        s = transmit(m, mu=mu)
        r = ideal_channel(s)
        rng = np.random.default_rng(0)
        m_hat = receive(r, mu=mu, y0=rng.random(), z0=rng.random())
        mse = np.mean((m[500:] - m_hat[500:]) ** 2)
        self.assertLess(mse, 0.01)

    def test_fir_channel_shape(self):
        m = binary_message(500)
        s = transmit(m)
        r, h = fir_channel(s, cutoff=0.99, num_taps=21)
        self.assertEqual(r.shape, (500,))
        self.assertEqual(h.shape, (21,))


class TestSpectral(unittest.TestCase):
    def test_psd_shape(self):
        x = np.random.randn(10_000)
        omega, psd = psd_normalised(x)
        self.assertEqual(omega.shape, psd.shape)

    def test_peak_normalised(self):
        x = np.random.randn(10_000)
        _, psd = psd_normalised(x)
        self.assertAlmostEqual(float(psd.max()), 1.0, places=9)


class TestLyapunov(unittest.TestCase):
    def test_returns_dict(self):
        result = lyapunov_max(Nitera=200, Ndiscard=100)
        self.assertIn("lyapunov_max", result)
        self.assertIn("all_exponents", result)
        self.assertEqual(len(result["all_exponents"]), 4)

    def test_positive_exponent(self):
        """With pole_radius=0 (no filter), standard Hénon is chaotic."""
        result = lyapunov_max(pole_radius=0.0, Nitera=500, Ndiscard=200)
        self.assertGreater(result["lyapunov_max"], 0,
                           "Standard Hénon (no filter) should have positive λ_max")

    def test_fixed_point_stability(self):
        info = fixed_point_stability()
        self.assertIn("eigenvalues", info)
        self.assertEqual(len(info["eigenvalues"]), 4)

    def test_henon2d_chaotic(self):
        """Pure 2-D Hénon (a=1.4, b=0.3) must be chaotic."""
        res = lyapunov_henon2d(Nitera=500, Ndiscard=200)
        self.assertGreater(res["lyapunov_max"], 0.3,
                           "2-D Hénon λ_max should be ~0.42")
        self.assertEqual(len(res["all_exponents"]), 2)

    def test_henon2d_fixed_points(self):
        """Check both fixed points exist and have correct eigenvalues."""
        res = lyapunov_henon2d(Nitera=100, Ndiscard=50)
        for key in ["fixed_point_p", "fixed_point_n",
                     "eigenvalues_p", "eigenvalues_n",
                     "stable_p", "stable_n"]:
            self.assertIn(key, res)
        # Positive fixed point should be unstable for a=1.4
        self.assertFalse(res["stable_p"])


class TestLyapunovEnsemble(unittest.TestCase):
    """Protocol: sample N_ci ICs ±perturbation around the fixed point."""

    def test_2d_shapes(self):
        from chaotic_pfc.lyapunov import (
            EnsembleResult, lyapunov_henon2d_ensemble,
        )
        r = lyapunov_henon2d_ensemble(
            Nitera=200, Ndiscard=100, n_initial=5, seed=0,
        )
        self.assertIsInstance(r, EnsembleResult)
        self.assertEqual(r.initial_conditions.shape, (5, 2))
        self.assertEqual(r.exponents_per_ci.shape,  (5, 2))
        self.assertEqual(r.lmax_per_ci.shape,       (5,))
        self.assertEqual(r.mean_exponents.shape,    (2,))

    def test_4d_shapes(self):
        from chaotic_pfc.lyapunov import lyapunov_max_ensemble
        r = lyapunov_max_ensemble(
            Nitera=200, Ndiscard=100, n_initial=5, seed=0,
        )
        self.assertEqual(r.initial_conditions.shape, (5, 4))
        self.assertEqual(r.exponents_per_ci.shape,  (5, 4))
        self.assertEqual(r.lmax_per_ci.shape,       (5,))
        self.assertEqual(r.mean_exponents.shape,    (4,))

    def test_ics_within_perturbation_box(self):
        """Every sampled IC must lie within [fp*(1-p), fp*(1+p)]."""
        from chaotic_pfc.lyapunov import lyapunov_henon2d_ensemble
        p = 0.1
        r = lyapunov_henon2d_ensemble(
            Nitera=100, Ndiscard=50, n_initial=10, perturbation=p, seed=1,
        )
        fp = r.fixed_point
        low  = np.minimum(fp * (1 - p), fp * (1 + p))
        high = np.maximum(fp * (1 - p), fp * (1 + p))
        self.assertTrue(np.all(r.initial_conditions >= low))
        self.assertTrue(np.all(r.initial_conditions <= high))

    def test_aggregates_match_arrays(self):
        """mean_lmax/max_lmax should match np.mean/np.max of lmax_per_ci."""
        from chaotic_pfc.lyapunov import lyapunov_henon2d_ensemble
        r = lyapunov_henon2d_ensemble(
            Nitera=100, Ndiscard=50, n_initial=8, seed=7,
        )
        np.testing.assert_allclose(r.mean_lmax, r.lmax_per_ci.mean())
        np.testing.assert_allclose(r.max_lmax,  r.lmax_per_ci.max())
        np.testing.assert_allclose(
            r.mean_exponents, r.exponents_per_ci.mean(axis=0),
        )
        self.assertEqual(r.n_chaotic + r.n_stable, len(r.lmax_per_ci))

    def test_determinism_with_seed(self):
        """Same seed → byte-identical arrays (no Numba prange here)."""
        from chaotic_pfc.lyapunov import lyapunov_henon2d_ensemble
        kw = dict(Nitera=100, Ndiscard=50, n_initial=5, seed=123)
        r1 = lyapunov_henon2d_ensemble(**kw)
        r2 = lyapunov_henon2d_ensemble(**kw)
        np.testing.assert_array_equal(r1.initial_conditions,
                                      r2.initial_conditions)
        np.testing.assert_array_equal(r1.exponents_per_ci,
                                      r2.exponents_per_ci)
        np.testing.assert_array_equal(r1.lmax_per_ci, r2.lmax_per_ci)

    def test_2d_henon_is_chaotic_on_average(self):
        """Standard 2-D Hénon (α=1.4, β=0.3) should yield mean λ_max > 0."""
        from chaotic_pfc.lyapunov import lyapunov_henon2d_ensemble
        r = lyapunov_henon2d_ensemble(
            Nitera=1000, Ndiscard=500, n_initial=10, seed=42,
        )
        self.assertGreater(r.mean_lmax, 0.3,
                           "Ensemble mean λ_max should be near 0.42")

    def test_csv_roundtrip(self):
        """to_csv writes a readable file with expected columns."""
        import csv
        from tempfile import TemporaryDirectory

        from chaotic_pfc.lyapunov import lyapunov_henon2d_ensemble

        r = lyapunov_henon2d_ensemble(
            Nitera=100, Ndiscard=50, n_initial=4, seed=0,
        )
        with TemporaryDirectory() as td:
            from pathlib import Path
            path = Path(td) / "out.csv"
            returned = r.to_csv(path)
            self.assertEqual(returned, path)
            self.assertTrue(path.exists())

            with path.open() as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(header[0], "ci")
                self.assertIn("lmax",   header)
                self.assertIn("status", header)
                # Read 4 data rows
                rows = [next(reader) for _ in range(4)]
                self.assertEqual([row[0] for row in rows], ["1", "2", "3", "4"])


if __name__ == "__main__":
    unittest.main()
