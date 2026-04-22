"""tests/test_receiver.py — Unit tests for the chaos-synchronisation receiver."""

import unittest

import numpy as np
from scipy.signal import firwin

from chaotic_pfc.channel import ideal_channel
from chaotic_pfc.receiver import receive, receive_order_n
from chaotic_pfc.signals import binary_message
from chaotic_pfc.transmitter import transmit, transmit_order_n


class TestReceiveStandard(unittest.TestCase):
    def test_output_shape(self):
        for N in (50, 500, 5000):
            r = np.zeros(N)
            m_hat = receive(r)
            self.assertEqual(m_hat.shape, (N,))

    def test_first_sample_formula(self):
        """m_hat[0] = (r[0] - y1[0]) / mu, by construction."""
        r = np.array([0.5, 0.0, 0.0])
        y0 = 0.2
        mu = 0.01
        m_hat = receive(r, mu=mu, y0=y0, z0=0.0)
        self.assertAlmostEqual(float(m_hat[0]), (0.5 - y0) / mu, places=12)

    def test_deterministic(self):
        """The receiver is a pure function: same input → same output."""
        r = np.random.default_rng(0).standard_normal(500)
        m_hat_1 = receive(r, mu=0.01, y0=0.1, z0=0.2)
        m_hat_2 = receive(r, mu=0.01, y0=0.1, z0=0.2)
        np.testing.assert_array_equal(m_hat_1, m_hat_2)

    def test_recovers_message_with_matched_state(self):
        """If transmitter and receiver start from the same (x0, y0) and
        the channel is ideal, the recovered message should match the
        original for every sample (up to machine precision at n=0)."""
        mu = 0.01
        m = binary_message(1000, period=20)
        x0, y0 = 0.1, 0.2
        s = transmit(m, mu=mu, x0=x0, y0=y0)
        r = ideal_channel(s)
        m_hat = receive(r, mu=mu, y0=x0, z0=y0)
        # With synchronised initial conditions, recovery is exact at n=0.
        self.assertAlmostEqual(float(m_hat[0]), float(m[0]), places=10)

    def test_mismatched_state_still_converges(self):
        """Chaos synchronisation: mismatched ICs should produce wrong
        decisions only in a short transient, then track."""
        mu = 0.01
        N = 10_000
        m = binary_message(N, period=20)
        s = transmit(m, mu=mu, x0=0.0, y0=0.0)
        r = ideal_channel(s)
        rng = np.random.default_rng(0)
        m_hat = receive(r, mu=mu, y0=rng.random(), z0=rng.random())
        # After a transient, MSE over the message alphabet must be small.
        transient = 500
        mse = float(np.mean((m[transient:] - m_hat[transient:]) ** 2))
        self.assertLess(mse, 1e-3)


class TestReceiveOrderN(unittest.TestCase):
    def _coeffs(self, Nc: int = 5) -> np.ndarray:
        return firwin(numtaps=Nc, cutoff=0.5, window="hamming")

    def test_output_shapes(self):
        r = np.zeros(200)
        c = self._coeffs(6)
        m_hat, state = receive_order_n(r, c)
        self.assertEqual(m_hat.shape, (200,))
        self.assertEqual(state.shape, (6, 201))

    def test_seed_is_deterministic(self):
        """Same seed for random y0 must give byte-identical output."""
        r = np.random.default_rng(7).standard_normal(300)
        c = self._coeffs(5)
        mh1, st1 = receive_order_n(r, c, seed=42)
        mh2, st2 = receive_order_n(r, c, seed=42)
        np.testing.assert_array_equal(mh1, mh2)
        np.testing.assert_array_equal(st1, st2)

    def test_explicit_y0_overrides_seed(self):
        r = np.zeros(100)
        c = self._coeffs(4)
        y0 = np.array([0.1, 0.2, 0.3, 0.4])
        mh1, _ = receive_order_n(r, c, y0=y0, seed=0)
        mh2, _ = receive_order_n(r, c, y0=y0, seed=999)
        np.testing.assert_array_equal(mh1, mh2)

    def test_initial_state_matches_y0(self):
        r = np.zeros(10)
        c = self._coeffs(5)
        y0 = np.linspace(-0.2, 0.2, 5)
        _, state = receive_order_n(r, c, y0=y0)
        np.testing.assert_array_equal(state[:, 0], y0)

    def test_roundtrip_order_n(self):
        """End-to-end check: transmit_order_n → ideal → receive_order_n
        with synchronised y0 should recover the message after a small
        transient."""
        mu = 0.01
        N = 5000
        c = self._coeffs(5)
        m = binary_message(N, period=20)

        x0 = np.array([0.1, 0.2, 0.15, 0.05, 0.3])
        s, _ = transmit_order_n(m, c, mu=mu, x0=x0)
        r = ideal_channel(s)
        m_hat, _ = receive_order_n(r, c, mu=mu, y0=x0)

        transient = 500
        mse = float(np.mean((m[transient:] - m_hat[transient:]) ** 2))
        self.assertLess(mse, 1e-3)


if __name__ == "__main__":
    unittest.main()
