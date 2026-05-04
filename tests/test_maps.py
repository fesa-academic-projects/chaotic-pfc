"""tests/test_maps.py — Unit tests for Henon map variants and the end-to-end pipeline."""

import unittest

import numpy as np

from chaotic_pfc.comms.channel import ideal_channel
from chaotic_pfc.comms.receiver import receive
from chaotic_pfc.comms.transmitter import transmit
from chaotic_pfc.dynamics.maps import (
    henon_filtered,
    henon_generalised,
    henon_order_n,
    henon_standard,
)
from chaotic_pfc.dynamics.signals import binary_message


class TestHenonStandard(unittest.TestCase):
    def test_output_shape(self):
        X, _Y = henon_standard(100)
        self.assertEqual(X.shape, (101,))

    def test_initial_condition(self):
        X, Y = henon_standard(200, x0=0.5, y0=0.2)
        self.assertEqual(X[0], 0.5)
        self.assertEqual(Y[0], 0.2)

    def test_first_iteration(self):
        X, _Y = henon_standard(10, a=1.2, b=0.2, x0=0.0, y0=0.0)
        # x[1] = 1 - a*x[0]^2 + b*y[0] = 1 - 0 + 0 = 1
        self.assertAlmostEqual(X[1], 1.0, places=10)

    def test_boundedness(self):
        X, _Y = henon_standard(2000)
        self.assertTrue(np.all(np.abs(X) < 3))


class TestHenonGeneralised(unittest.TestCase):
    def test_output_shape(self):
        X, _Y = henon_generalised(100)
        self.assertEqual(X.shape, (101,))

    def test_deterministic(self):
        X1, Y1 = henon_generalised(500)
        X2, Y2 = henon_generalised(500)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)

    def test_different_params_diverge(self):
        Xa, _ = henon_generalised(100, alpha=1.4, beta=0.3)
        Xb, _ = henon_generalised(100, alpha=1.6, beta=0.3)
        self.assertFalse(np.allclose(Xa, Xb))


class TestHenonFiltered(unittest.TestCase):
    def test_c0_1_c1_0_matches_generalised(self):
        X_f, Y_f = henon_filtered(500, c0=1.0, c1=0.0)
        X_g, Y_g = henon_generalised(500)
        np.testing.assert_allclose(X_f, X_g, atol=1e-12)
        np.testing.assert_allclose(Y_f, Y_g, atol=1e-12)


class TestHenonOrderN(unittest.TestCase):
    def _simple_coeffs(self, Nc: int = 5) -> np.ndarray:
        c = np.zeros(Nc)
        c[0] = 0.6
        c[1] = 0.3
        c[2] = 0.1
        return c

    def test_output_shapes(self):
        c = self._simple_coeffs(5)
        state, output = henon_order_n(steps=100, fir_coeffs=c)
        self.assertEqual(state.shape, (5, 101))
        self.assertEqual(output.shape, (100,))

    def test_explicit_x0_is_preserved(self):
        c = self._simple_coeffs(4)
        x0 = np.array([0.1, 0.2, 0.3, 0.4])
        state, _ = henon_order_n(steps=50, fir_coeffs=c, x0=x0)
        np.testing.assert_array_equal(state[:, 0], x0)

    def test_seed_is_deterministic(self):
        c = self._simple_coeffs(5)
        s1, _ = henon_order_n(steps=50, fir_coeffs=c, seed=42)
        s2, _ = henon_order_n(steps=50, fir_coeffs=c, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_driving_signal_exists(self):
        c = self._simple_coeffs(5)
        _, output = henon_order_n(steps=100, fir_coeffs=c)
        self.assertEqual(output.shape, (100,))
        self.assertTrue(np.all(np.isfinite(output)))


class TestPipeline(unittest.TestCase):
    def test_ideal_roundtrip(self):
        N = 50_000
        mu = 0.01
        m = binary_message(N)
        s = transmit(m, mu=mu)
        r = ideal_channel(s)
        rng = np.random.default_rng(0)
        m_hat = receive(r, mu=mu, y0=rng.random(), z0=rng.random())
        mse = np.mean((m[500:] - m_hat[500:]) ** 2)
        self.assertLess(mse, 0.01)


if __name__ == "__main__":
    unittest.main()
