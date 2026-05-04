"""tests/test_signals.py — Unit tests for message signal generators."""

import unittest

import numpy as np

from chaotic_pfc.dynamics.signals import binary_message, sinusoidal_message


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


class TestSinusoidalMessage(unittest.TestCase):
    def test_length(self):
        for N in (100, 500, 10_000):
            m = sinusoidal_message(N)
            self.assertEqual(len(m), N)

    def test_amplitude_bounds(self):
        m = sinusoidal_message(1000, normalised_freq=0.05)
        self.assertTrue(np.all(np.abs(m) <= 1.0 + 1e-12))

    def test_frequency_matches_requested(self):
        N = 4096
        f = 0.125
        m = sinusoidal_message(N, normalised_freq=f)
        spectrum = np.abs(np.fft.rfft(m))
        peak_idx = int(np.argmax(spectrum))
        expected_bin = round(f * N)
        self.assertEqual(peak_idx, expected_bin)

    def test_dc_frequency(self):
        m = sinusoidal_message(100, normalised_freq=0.0)
        np.testing.assert_array_equal(m, np.zeros(100))

    def test_nyquist_frequency(self):
        m = sinusoidal_message(6, normalised_freq=0.5)
        expected = np.sin(np.pi * np.arange(6))
        np.testing.assert_allclose(m, expected, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
