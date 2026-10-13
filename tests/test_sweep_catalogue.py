"""Filter catalogue and FIR filter design."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from chaotic_pfc.analysis.sweep import (
    FILTER_TYPES,
    WINDOW_DISPLAY_NAMES,
    WINDOWS,
    SweepResult,
    _build_task_order,
    _infer_config_from_path,
    load_sweep,
    precompute_fir_bank,
    quick_sweep_params,
    run_sweep,
    save_sweep,
)

# ═══════════════════════════════════════════════════════════════════════════
# Catalogue sanity checks


class TestCatalogue(unittest.TestCase):
    def test_windows_are_unique(self):
        self.assertEqual(len(WINDOWS), len(set(WINDOWS)))

    def test_every_window_has_display_name(self):
        for w in WINDOWS:
            self.assertIn(w, WINDOW_DISPLAY_NAMES)

    def test_filter_types_values(self):
        self.assertEqual(set(FILTER_TYPES), {"lowpass", "highpass", "bandpass", "bandstop"})


# ═══════════════════════════════════════════════════════════════════════════
# FIR bank
# ═══════════════════════════════════════════════════════════════════════════


class TestFirBank(unittest.TestCase):
    def test_shape(self):
        num_taps_grid = np.array([2, 3, 5])
        cutoff_grid = np.linspace(0.1, 0.9, 4)
        bank, gains = precompute_fir_bank(num_taps_grid, cutoff_grid, "lowpass", "hamming")
        self.assertEqual(bank.shape, (3, 4, 5))  # max_taps = max(num_taps_grid)
        self.assertEqual(gains.shape, (3, 4))

    def test_lowpass_dc_gain_approx_one(self):
        """Low-pass FIR filters should have DC gain close to 1."""
        num_taps_grid = np.array([11, 21, 41])
        cutoff_grid = np.array([0.3, 0.5])
        _, gains = precompute_fir_bank(num_taps_grid, cutoff_grid, "lowpass", "hamming")
        np.testing.assert_allclose(gains, 1.0, atol=1e-6)

    def test_highpass_requires_odd_orders(self):
        """Highpass needs an odd number of taps; an even num_taps_grid must be rejected.

        Allowing Nss|1 padding (the previous behaviour) silently truncated
        the filter at read time in the sweep kernel, which suppressed
        divergence-based early-exits and caused a ~10× slowdown.
        """
        cutoff_grid = np.array([0.3])
        with self.assertRaises(ValueError):
            precompute_fir_bank(np.array([4, 6, 8]), cutoff_grid, "highpass", "hamming")
        # Mixed odd/even — even values must still be reported.
        with self.assertRaises(ValueError):
            precompute_fir_bank(np.array([3, 4, 5]), cutoff_grid, "highpass", "hamming")

    def test_highpass_odd_orders_ok(self):
        """Highpass with odd num_taps_grid builds a filter of size == order."""
        num_taps_grid = np.array([3, 5, 7])
        cutoff_grid = np.array([0.3])
        bank, _ = precompute_fir_bank(num_taps_grid, cutoff_grid, "highpass", "hamming")
        self.assertEqual(bank.shape, (3, 1, 7))
        # Last tap of each odd-order filter must be non-zero (filter not truncated).
        for i, Nss in enumerate(num_taps_grid):
            self.assertNotEqual(bank[i, 0, int(Nss) - 1], 0.0)

    def test_kaiser_window_accepted(self):
        num_taps_grid = np.array([5])
        cutoff_grid = np.array([0.4])
        bank, gains = precompute_fir_bank(num_taps_grid, cutoff_grid, "lowpass", "kaiser")
        self.assertTrue(np.all(np.isfinite(bank)))
        self.assertTrue(np.all(np.isfinite(gains)))

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            precompute_fir_bank([3], np.array([0.5]), "lowpass", "not_a_window")

    def test_invalid_filter_raises(self):
        with self.assertRaises(ValueError):
            precompute_fir_bank([3], np.array([0.5]), "notch", "hamming")

    def test_bandpass_produces_valid_coeffs(self):
        num_taps_grid = np.array([11, 21])
        cutoff_grid = np.array([0.3, 0.5])
        bank, gains = precompute_fir_bank(num_taps_grid, cutoff_grid, "bandpass", "hamming", bandwidth=0.2)
        self.assertEqual(bank.shape, (2, 2, 21))
        self.assertEqual(gains.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(bank)))
        self.assertTrue(np.all(np.isfinite(gains)))

    def test_bandstop_produces_valid_coeffs(self):
        num_taps_grid = np.array([11, 21])
        cutoff_grid = np.array([0.3, 0.5])
        bank, _gains = precompute_fir_bank(num_taps_grid, cutoff_grid, "bandstop", "hamming", bandwidth=0.2)
        self.assertEqual(bank.shape, (2, 2, 21))
        self.assertTrue(np.all(np.isfinite(bank)))

    def test_bandpass_even_orders_accepted(self):
        """Bandpass doesn't require odd num_taps_grid (unlike highpass/bandstop)."""
        num_taps_grid = np.array([4, 6, 8])
        cutoff_grid = np.array([0.4])
        bank, _ = precompute_fir_bank(num_taps_grid, cutoff_grid, "bandpass", "hamming", bandwidth=0.2)
        self.assertEqual(bank.shape, (3, 1, 8))

    def test_bandstop_rejects_even_orders(self):
        """Bandstop requires odd num_taps_grid (same constraint as highpass)."""
        with self.assertRaises(ValueError):
            precompute_fir_bank(
                np.array([4, 6]), np.array([0.4]), "bandstop", "hamming", bandwidth=0.2
            )

    def test_bandpass_low_gain(self):
        """Bandpass filters should have near-zero DC gain."""
        num_taps_grid = np.array([21, 31])
        cutoff_grid = np.array([0.5])
        _, gains = precompute_fir_bank(num_taps_grid, cutoff_grid, "bandpass", "hamming", bandwidth=0.2)
        self.assertTrue(np.all(np.abs(gains) < 0.1))

    def test_bandwidth_clamps_to_boundaries(self):
        """Band edges should never go outside (0, 1)."""
        num_taps_grid = np.array([11])
        # centre near 0 → low edge clamped to 1e-5
        cutoff_grid = np.array([0.05])
        bank, _ = precompute_fir_bank(num_taps_grid, cutoff_grid, "bandpass", "hamming", bandwidth=0.3)
        self.assertTrue(np.all(np.isfinite(bank)))
        # centre near 1 → high edge clamped to 0.99999
        cutoff_grid = np.array([0.95])
        bank, _ = precompute_fir_bank(num_taps_grid, cutoff_grid, "bandpass", "hamming", bandwidth=0.3)
        self.assertTrue(np.all(np.isfinite(bank)))


# ═══════════════════════════════════════════════════════════════════════════
# run_sweep — tiny grid smoke tests
# ═══════════════════════════════════════════════════════════════════════════
