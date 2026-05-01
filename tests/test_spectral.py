"""tests/test_spectral.py — full coverage of spectral.py plus β-sweep ecosystem.

Covers:
- psd_normalised: all windows, DC removal, normalisation, edge cases
- config: kaiser_beta exposure
- sweep: β plumbed through FIR bank, sweep metadata
- CLI: β grid construction
- 3D aggregation: Plotly volume rendering (skipped if Plotly missing)
"""

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import numpy as np

from chaotic_pfc.cli.sweep import _beta_values
from chaotic_pfc.config import DEFAULT_CONFIG
from chaotic_pfc.spectral import _WINDOWS, psd_normalised
from chaotic_pfc.sweep import precompute_fir_bank, run_sweep, save_sweep


# ════════════════════════════════════════════════════════════════════════════
# psd_normalised — all windows, normalisation, edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestPsdNormalised(unittest.TestCase):
    def test_default_equals_hamming(self):
        x = np.linspace(0, 1, 1024)
        _, psd_a = psd_normalised(x)
        _, psd_b = psd_normalised(x, window="hamming")
        np.testing.assert_array_equal(psd_a, psd_b)

    def test_peak_normalisation_is_one(self):
        rng = np.random.default_rng(0)
        x = np.sin(2 * np.pi * 0.125 * np.arange(2048)) + 0.01 * rng.standard_normal(2048)
        _, psd = psd_normalised(x)
        self.assertAlmostEqual(float(psd.max()), 1.0, places=6)

    def test_zero_signal_produces_zero_psd(self):
        _, psd = psd_normalised(np.zeros(2048))
        np.testing.assert_array_equal(psd, np.zeros_like(psd))

    def test_remove_dc_subtracts_mean(self):
        x = np.ones(1024) * 5.0 + np.random.default_rng(1).standard_normal(1024) * 0.01
        _, psd_with_dc = psd_normalised(x, remove_dc=False)
        _, psd_no_dc = psd_normalised(x, remove_dc=True)
        self.assertGreater(psd_with_dc[0], psd_no_dc[0])

    def test_output_shape_matches_nfft(self):
        _, psd = psd_normalised(np.zeros(1024), nfft=2048)
        self.assertEqual(psd.shape[0], 2048 // 2 + 1)

    def test_frequency_axis_range(self):
        omega, _ = psd_normalised(np.zeros(1024))
        self.assertAlmostEqual(omega[0], 0.0, places=6)
        self.assertAlmostEqual(omega[-1], 1.0, places=6)

    def test_all_windows_produce_valid_output(self):
        x = np.linspace(0, 1, 1024)
        for w in _WINDOWS:
            omega, psd = psd_normalised(x, window=w)
            self.assertEqual(len(omega), len(psd))
            self.assertGreaterEqual(float(psd.max()), 0.0)

    def test_kaiser_different_betas_give_different_spectra(self):
        rng = np.random.default_rng(0)
        x = np.sin(2 * np.pi * 0.125 * np.arange(2048)) + 0.01 * rng.standard_normal(2048)
        _, psd_low = psd_normalised(x, window="kaiser", kaiser_beta=2.0)
        _, psd_high = psd_normalised(x, window="kaiser", kaiser_beta=12.0)
        self.assertFalse(np.allclose(psd_low, psd_high))

    def test_kaiser_negative_beta_raises(self):
        with self.assertRaises(ValueError):
            psd_normalised(np.zeros(128), window="kaiser", kaiser_beta=-1.0)

    def test_unknown_window_raises(self):
        with self.assertRaises(ValueError):
            psd_normalised(np.zeros(128), window="tukey")

    def test_pure_tone_peak_at_correct_frequency(self):
        # ω/π = 0.25 → f = 0.125 cycles/sample (because ω = 2πf)
        n = np.arange(2048)
        x = np.sin(2 * np.pi * 0.125 * n)
        omega, psd = psd_normalised(x)
        self.assertAlmostEqual(omega[psd.argmax()], 0.25, places=2)


# ════════════════════════════════════════════════════════════════════════════
# config.py — kaiser_beta exposed
# ════════════════════════════════════════════════════════════════════════════


class TestConfig(unittest.TestCase):
    def test_default_config_exposes_kaiser_beta(self):
        self.assertEqual(DEFAULT_CONFIG.spectral.window, "hamming")
        self.assertEqual(DEFAULT_CONFIG.spectral.kaiser_beta, 5.0)


# ════════════════════════════════════════════════════════════════════════════
# sweep.py — kaiser_beta plumbed through
# ════════════════════════════════════════════════════════════════════════════


class TestSweepKaiserBeta(unittest.TestCase):
    def test_precompute_with_two_betas_differs(self):
        orders = np.array([5, 7])
        cutoffs = np.array([0.3, 0.5])
        bank_a, _ = precompute_fir_bank(orders, cutoffs, "lowpass", "kaiser", kaiser_beta=2.0)
        bank_b, _ = precompute_fir_bank(orders, cutoffs, "lowpass", "kaiser", kaiser_beta=10.0)
        self.assertFalse(np.allclose(bank_a, bank_b))

    def test_precompute_kaiser_negative_beta_raises(self):
        with self.assertRaises(ValueError):
            precompute_fir_bank(
                np.array([5]), np.array([0.3]), "lowpass", "kaiser", kaiser_beta=-0.1
            )

    def test_run_sweep_records_beta_in_metadata(self):
        result = run_sweep(
            window="kaiser",
            filter_type="lowpass",
            orders=np.arange(2, 5),
            cutoffs=np.linspace(0.2, 0.8, 4),
            Nitera=20,
            Nmap=80,
            n_initial=2,
            seed=0,
            warmup=False,
            kaiser_beta=3.5,
        )
        self.assertEqual(result.metadata["kaiser_beta"], 3.5)


# ════════════════════════════════════════════════════════════════════════════
# CLI helper — β grid construction
# ════════════════════════════════════════════════════════════════════════════


class TestBetaValues(unittest.TestCase):
    def test_default_range_gives_25_values(self):
        betas = _beta_values(2.0, 14.0, 0.5)
        self.assertEqual(len(betas), 25)
        self.assertEqual(betas[0], 2.0)
        self.assertEqual(betas[-1], 14.0)

    def test_step_must_be_positive(self):
        with self.assertRaises(ValueError):
            _beta_values(0.0, 1.0, 0.0)

    def test_negative_min_raises(self):
        with self.assertRaises(ValueError):
            _beta_values(-0.1, 1.0, 0.05)

    def test_max_below_min_raises(self):
        with self.assertRaises(ValueError):
            _beta_values(1.0, 0.5, 0.05)

    def test_custom_range(self):
        betas = _beta_values(0.0, 1.0, 0.05)
        self.assertEqual(len(betas), 21)


# ════════════════════════════════════════════════════════════════════════════
# 3D aggregation — only runs if Plotly is installed
# ════════════════════════════════════════════════════════════════════════════


_HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


@unittest.skipUnless(_HAS_PLOTLY, "Plotly not installed (optional [viz3d] extra)")
class TestSweepPlotting3D(unittest.TestCase):
    def _write_three_beta_sweeps(self, root: Path) -> None:
        for beta in (0.0, 0.5, 1.0):
            result = run_sweep(
                window="kaiser",
                filter_type="lowpass",
                orders=np.arange(2, 5),
                cutoffs=np.linspace(0.2, 0.8, 4),
                Nitera=20,
                Nmap=80,
                n_initial=2,
                seed=0,
                warmup=False,
                kaiser_beta=beta,
            )
            save_sweep(result, root / f"beta_{beta:.2f}" / "variables_lyapunov.npz")

    def test_aggregate_and_plot(self):
        from chaotic_pfc.sweep_plotting_3d import (
            aggregate_beta_sweeps,
            plot_3d_beta_volume,
        )

        with TemporaryDirectory() as td:
            root = Path(td)
            self._write_three_beta_sweeps(root)

            volume, betas, orders, cutoffs = aggregate_beta_sweeps(root)
            self.assertEqual(volume.shape, (3, 3, 4))
            np.testing.assert_array_equal(betas, [0.0, 0.5, 1.0])

            html = root / "out.html"
            fig = plot_3d_beta_volume(volume, betas, orders, cutoffs, save_path=html)
            self.assertEqual(len(fig.data), 3)
            self.assertTrue(html.exists())
            self.assertGreater(html.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
