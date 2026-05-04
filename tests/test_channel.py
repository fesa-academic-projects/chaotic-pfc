"""tests/test_channel.py — Unit tests for the channel models."""

import unittest

import numpy as np
from scipy.signal import firwin, lfilter

from chaotic_pfc.comms.channel import fir_channel, ideal_channel


class TestIdealChannel(unittest.TestCase):
    def test_returns_equal_values(self):
        s = np.array([1.0, -0.5, 0.25, 3.14])
        r = ideal_channel(s)
        np.testing.assert_array_equal(r, s)

    def test_returns_copy_not_view(self):
        """Mutating the output must NOT mutate the input.

        This is a subtle but important property: an ideal channel is
        meant to be a noiseless pass-through, not an alias of the
        transmitted signal. Downstream code that modifies the received
        signal (e.g., for equalisation simulation) should never reach
        back into the transmitter.
        """
        s = np.array([1.0, 2.0, 3.0])
        r = ideal_channel(s)
        r[0] = 999.0
        self.assertEqual(s[0], 1.0)

    def test_preserves_length(self):
        for N in (0, 1, 10, 1000):
            s = np.random.randn(N)
            self.assertEqual(len(ideal_channel(s)), N)


class TestFirChannel(unittest.TestCase):
    def test_output_shape(self):
        s = np.random.randn(500)
        r, h = fir_channel(s, cutoff=0.5, num_taps=21)
        self.assertEqual(r.shape, (500,))
        self.assertEqual(h.shape, (21,))

    def test_num_taps_respected(self):
        """Caller-provided num_taps must be honoured exactly."""
        for n in (11, 21, 51, 101):
            _, h = fir_channel(np.zeros(10), num_taps=n)
            self.assertEqual(len(h), n)

    def test_lowpass_dc_gain_is_one(self):
        """A properly normalised low-pass FIR has DC gain ≈ 1."""
        _, h = fir_channel(np.zeros(10), cutoff=0.5, num_taps=51)
        self.assertAlmostEqual(float(h.sum()), 1.0, places=6)

    def test_impulse_response_matches_coefficients(self):
        """Feeding a discrete impulse should echo back the filter taps."""
        num_taps = 31
        impulse = np.zeros(2 * num_taps)
        impulse[0] = 1.0
        r, h = fir_channel(impulse, cutoff=0.5, num_taps=num_taps)
        # First num_taps samples of the output must equal the filter taps.
        np.testing.assert_allclose(r[:num_taps], h, atol=1e-12)
        # Everything after the filter length should be zero (FIR is finite).
        np.testing.assert_allclose(r[num_taps:], 0.0, atol=1e-12)

    def test_matches_scipy_direct_call(self):
        """The function must be a thin wrapper around scipy — no extras."""
        s = np.random.default_rng(0).standard_normal(200)
        h_expected = firwin(numtaps=33, cutoff=0.4, window="hamming", pass_zero=True, fs=2.0)
        r_expected = lfilter(h_expected, [1.0], s)
        r, h = fir_channel(s, cutoff=0.4, num_taps=33, window="hamming")
        np.testing.assert_allclose(h, h_expected)
        np.testing.assert_allclose(r, r_expected)

    def test_alternative_window(self):
        """A different window must produce a different filter."""
        s = np.zeros(5)
        _, h_hamming = fir_channel(s, num_taps=21, window="hamming")
        _, h_blackman = fir_channel(s, num_taps=21, window="blackman")
        self.assertFalse(np.allclose(h_hamming, h_blackman))


if __name__ == "__main__":
    unittest.main()
