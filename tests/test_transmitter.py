"""tests/test_transmitter.py — Unit tests for the chaotic transmitter."""

import unittest

import numpy as np

from chaotic_pfc.comms.transmitter import transmit, transmit_order_n
from chaotic_pfc.dynamics.signals import binary_message
from tests._test_helpers import make_fir_coeffs


class TestTransmitStandard(unittest.TestCase):
    def test_output_shape(self):
        for N in (50, 500, 5000):
            m = np.zeros(N)
            s = transmit(m)
            self.assertEqual(s.shape, (N,))

    def test_first_sample_formula(self):
        """s[0] = x0 + mu * m[0] exactly, by construction."""
        for mu in (0.001, 0.01, 0.1):
            for m0 in (-1.0, 1.0):
                m = np.array([m0, 0.0, 0.0])
                s = transmit(m, mu=mu, x0=0.3, y0=0.7)
                self.assertAlmostEqual(float(s[0]), 0.3 + mu * m0, places=12)

    def test_mu_zero_is_pure_henon(self):
        """With mu=0 the output must be the autonomous Hénon trajectory,
        independent of the message content."""
        x0, y0 = 0.1, 0.2
        N = 200
        m1 = np.ones(N)
        m2 = -np.ones(N)
        s1 = transmit(m1, mu=0.0, x0=x0, y0=y0)
        s2 = transmit(m2, mu=0.0, x0=x0, y0=y0)
        np.testing.assert_array_equal(s1, s2)

    def test_message_affects_output(self):
        """With mu != 0, changing the message changes the output."""
        m_zeros = np.zeros(100)
        m_binary = binary_message(100, period=20)
        s_zeros = transmit(m_zeros, mu=0.01)
        s_binary = transmit(m_binary, mu=0.01)
        self.assertFalse(np.allclose(s_zeros, s_binary))

    def test_deterministic(self):
        """Same inputs must give same outputs; the transmitter is pure."""
        m = binary_message(1000, period=20)
        s1 = transmit(m, mu=0.01)
        s2 = transmit(m, mu=0.01)
        np.testing.assert_array_equal(s1, s2)

    def test_recurrence_holds_for_second_sample(self):
        """Check s[1] = x1[1] + mu * m[1] where x1[1] = a - s[0]^2 + b*y0.

        Tests the recurrence formula explicitly for a known seed state.
        """
        mu, a, b = 0.01, 1.4, 0.3
        x0, y0 = 0.0, 0.0
        m = np.array([0.5, -0.5, 0.0])
        s = transmit(m, mu=mu, a=a, b=b, x0=x0, y0=y0)

        # Manual one-step simulation
        s0_expected = x0 + mu * m[0]
        x1_after = a - s0_expected**2 + b * y0
        s1_expected = x1_after + mu * m[1]
        self.assertAlmostEqual(float(s[0]), s0_expected, places=12)
        self.assertAlmostEqual(float(s[1]), s1_expected, places=12)


class TestTransmitOrderN(unittest.TestCase):
    def test_output_shapes(self):
        m = np.zeros(200)
        c = make_fir_coeffs(7)
        s, state = transmit_order_n(m, c)
        self.assertEqual(s.shape, (200,))
        self.assertEqual(state.shape, (7, 201))

    def test_seed_is_deterministic(self):
        """Same seed with default x0 must give byte-identical output."""
        m = binary_message(300, period=20)
        c = make_fir_coeffs(5)
        s1, st1 = transmit_order_n(m, c, seed=42)
        s2, st2 = transmit_order_n(m, c, seed=42)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(st1, st2)

    def test_different_seeds_diverge(self):
        """Different random x0 → different trajectories."""
        m = np.zeros(100)
        c = make_fir_coeffs(5)
        s1, _ = transmit_order_n(m, c, seed=0)
        s2, _ = transmit_order_n(m, c, seed=1)
        self.assertFalse(np.allclose(s1, s2))

    def test_explicit_x0_overrides_seed(self):
        """Passing x0 must make seed irrelevant."""
        m = np.zeros(50)
        c = make_fir_coeffs(4)
        x0 = np.array([0.1, 0.2, 0.3, 0.4])
        s1, _ = transmit_order_n(m, c, x0=x0, seed=0)
        s2, _ = transmit_order_n(m, c, x0=x0, seed=99)
        np.testing.assert_array_equal(s1, s2)

    def test_initial_state_matches_x0(self):
        """The first column of the state array must equal x0."""
        c = make_fir_coeffs(6)
        x0 = np.linspace(0.0, 0.5, 6)
        _, state = transmit_order_n(np.zeros(10), c, x0=x0)
        np.testing.assert_array_equal(state[:, 0], x0)

    def test_order_n_seed_none_does_not_raise(self):
        """seed=None should work without error (leaves RNG untouched)."""
        c = make_fir_coeffs(4)
        s, _ = transmit_order_n(np.ones(50), c, seed=None)
        self.assertEqual(len(s), 50)


if __name__ == "__main__":
    unittest.main()
