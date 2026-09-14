"""tests/test_signals.py — Unit tests for message signal generators."""

import unittest

import numpy as np

from chaotic_pfc.dynamics.signals import binary_message, sinusoidal_message, text_message


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


class TestTextMessage(unittest.TestCase):
    def test_length(self):
        m = text_message("hello", 1000, bit_period=20)
        self.assertEqual(len(m), 1000)

    def test_values(self):
        m = text_message("chaos", 200, bit_period=5)
        self.assertTrue(set(np.unique(m)).issubset({-1.0, 1.0}))

    def test_ascii_encoding(self):
        m = text_message("A", 8, bit_period=1)
        np.testing.assert_array_equal(m, [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0])

    def test_bit_hold(self):
        m = text_message("A", 16, bit_period=2)
        self.assertEqual(m[0], m[1])
        self.assertEqual(m[2], m[3])

    def test_irregular_not_square_wave(self):
        m = text_message("hello", 400, bit_period=10)
        sq = binary_message(400, period=10)
        self.assertFalse(np.array_equal(m, sq))

    def test_invalid_bit_period(self):
        with self.assertRaises(ValueError):
            text_message("hi", 100, bit_period=0)

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            text_message("", 100, bit_period=20)

    def test_non_ascii_raises(self):
        with self.assertRaises(UnicodeEncodeError):
            text_message("café", 100, bit_period=20)


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

    def test_dc_frequency_raises(self):
        with self.assertRaises(ValueError):
            sinusoidal_message(100, normalised_freq=0.0)

    def test_nyquist_frequency_raises(self):
        with self.assertRaises(ValueError):
            sinusoidal_message(6, normalised_freq=0.5)


if __name__ == "__main__":
    unittest.main()
