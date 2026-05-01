"""tests/test_sweep_plotting.py — Tests for the sweep plotting module."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")  # must come before pyplot is imported anywhere

import numpy as np

from chaotic_pfc.sweep import SweepResult
from chaotic_pfc.sweep_plotting import (
    DIFFICULTY_FIGURE_FILENAME,
    FIGURE_FILENAMES,
    _unpack,
    classify,
    plot_all,
    plot_classification_interleaved,
    plot_difficulty_map,
    plot_heatmap_continuous,
)


def _dummy_result(ncoef: int = 4, ncut: int = 6) -> SweepResult:
    """Build a small SweepResult with mixed chaotic / periodic / NaN cells."""
    rng = np.random.default_rng(0)
    h = rng.uniform(-0.5, 0.5, size=(ncoef, ncut))
    # Force some NaNs to exercise the divergent branch
    h[0, 0] = np.nan
    h[-1, -1] = np.nan
    return SweepResult(
        h=h,
        h_std=np.abs(h) * 0.1,
        orders=np.arange(2, 2 + ncoef),
        cutoffs=np.linspace(0.1, 0.9, ncut),
        window="hamming",
        filter_type="lowpass",
        metadata={"Nitera": 10, "Nmap": 50, "n_initial": 2},
    )


def _adaptive_result(ncoef: int = 4, ncut: int = 6) -> SweepResult:
    """Like ``_dummy_result`` but with non-trivial ``n_iters_used``.

    Used by tests that exercise the adaptive-only difficulty map and
    by tests that verify ``plot_all`` emits an extra figure when the
    sweep was run with ``adaptive=True``.
    """
    rng = np.random.default_rng(0)
    h = rng.uniform(-0.3, 0.3, size=(ncoef, ncut))
    h[0, 0] = np.nan  # one diverged point
    n_iters = rng.uniform(700, 3000, size=(ncoef, ncut))
    n_iters[0, 0] = np.nan  # NaN in the same cell as h
    return SweepResult(
        h=h,
        h_std=np.abs(h) * 0.1,
        orders=np.arange(2, 2 + ncoef),
        cutoffs=np.linspace(0.1, 0.9, ncut),
        window="hamming",
        filter_type="lowpass",
        n_iters_used=n_iters,
        metadata={
            "Nitera": 500,
            "Nmap": 3000,
            "n_initial": 25,
            "adaptive": True,
            "Nmap_min": 700,
            "tol": 1e-3,
        },
    )


class TestClassify(unittest.TestCase):
    def test_nan_maps_to_2(self):
        arr = np.array([[np.nan, 0.1], [-0.2, np.nan]])
        out = classify(arr)
        self.assertEqual(out[0, 0], 2)
        self.assertEqual(out[1, 1], 2)

    def test_negative_maps_to_minus_one(self):
        out = classify(np.array([[-0.01, -1.0]]))
        self.assertTrue(np.all(out == -1))

    def test_positive_maps_to_zero(self):
        out = classify(np.array([[0.01, 1.0]]))
        self.assertTrue(np.all(out == 0))

    def test_exact_zero_maps_to_minus_one(self):
        out = classify(np.array([[0.0, -0.0]]))
        self.assertTrue(np.all(out == -1))

    def test_all_periodic(self):
        out = classify(np.full((3, 2), -0.5))
        self.assertTrue(np.all(out == -1))
        self.assertEqual(out.shape, (3, 2))

    def test_all_chaotic(self):
        out = classify(np.full((3, 2), 0.3))
        self.assertTrue(np.all(out == 0))
        self.assertEqual(out.shape, (3, 2))

    def test_all_unbounded(self):
        out = classify(np.full((2, 3), np.nan))
        self.assertTrue(np.all(out == 2))
        self.assertEqual(out.shape, (2, 3))


class TestIndividualPlotters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = _dummy_result()

    def test_heatmap_saves_file(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "heat.png"
            fig = plot_heatmap_continuous(self.result, save_path=path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            fig.clear()

    def test_classification_interleaved_saves_file(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "class_inter.png"
            fig = plot_classification_interleaved(self.result, save_path=path)
            self.assertTrue(path.exists())
            fig.clear()


class TestPlotAll(unittest.TestCase):
    def test_plot_all_non_adaptive_creates_two_files(self):
        """Non-adaptive sweep: only the two always-present figures."""
        result = _dummy_result()
        with TemporaryDirectory() as td:
            paths = plot_all(result, Path(td), fmt="png")
            self.assertEqual(len(paths), 2)
            self.assertEqual(len(paths), len(FIGURE_FILENAMES))
            for p in paths:
                self.assertTrue(p.exists())
                self.assertGreater(p.stat().st_size, 0)
            # Difficulty map must NOT be present for non-adaptive sweeps.
            diff_path = Path(td) / f"{DIFFICULTY_FIGURE_FILENAME}.png"
            self.assertFalse(diff_path.exists())

    def test_plot_all_adaptive_creates_three_files(self):
        """Adaptive sweep: classification figs + difficulty map."""
        result = _adaptive_result()
        with TemporaryDirectory() as td:
            paths = plot_all(result, Path(td), fmt="png")
            self.assertEqual(len(paths), 3)
            for p in paths:
                self.assertTrue(p.exists())
                self.assertGreater(p.stat().st_size, 0)
            # Difficulty map must be the last one and exist on disk.
            diff_path = Path(td) / f"{DIFFICULTY_FIGURE_FILENAME}.png"
            self.assertTrue(diff_path.exists())
            self.assertEqual(paths[-1], diff_path)

    def test_plot_all_respects_fmt(self):
        result = _dummy_result()
        with TemporaryDirectory() as td:
            paths = plot_all(result, Path(td), fmt="svg")
            for p in paths:
                self.assertEqual(p.suffix, ".svg")

    def test_plot_all_respects_fmt_for_difficulty(self):
        """SVG fmt must apply to the optional difficulty map too."""
        result = _adaptive_result()
        with TemporaryDirectory() as td:
            paths = plot_all(result, Path(td), fmt="svg")
            for p in paths:
                self.assertEqual(p.suffix, ".svg")

    def test_plot_all_creates_missing_dir(self):
        result = _dummy_result()
        with TemporaryDirectory() as td:
            out_dir = Path(td) / "nested" / "deep"
            paths = plot_all(result, out_dir, fmt="png")
            self.assertTrue(out_dir.is_dir())
            self.assertEqual(len(paths), 2)

    def test_plot_all_close_figures_false(self):
        result = _dummy_result()
        with TemporaryDirectory() as td:
            paths = plot_all(result, Path(td), fmt="png", close_figures=False)
            self.assertEqual(len(paths), 2)
            for p in paths:
                self.assertTrue(p.exists())

    def test_plot_all_close_figures_false_adaptive(self):
        result = _adaptive_result()
        with TemporaryDirectory() as td:
            paths = plot_all(result, Path(td), fmt="png", close_figures=False)
            self.assertEqual(len(paths), 3)
            for p in paths:
                self.assertTrue(p.exists())


# ═══════════════════════════════════════════════════════════════════════════
# Difficulty map (adaptive-only figure)
# ═══════════════════════════════════════════════════════════════════════════


class TestDifficultyMap(unittest.TestCase):
    def test_saves_file(self):
        result = _adaptive_result()
        with TemporaryDirectory() as td:
            path = Path(td) / "diff.png"
            fig = plot_difficulty_map(result, save_path=path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            fig.clear()

    def test_rejects_non_adaptive_result(self):
        """Plotting a non-adaptive result is misleading (single-colour map);
        the function must raise rather than silently produce a useless figure."""
        result = _dummy_result()  # no n_iters_used, adaptive flag absent
        with self.assertRaises(ValueError):
            plot_difficulty_map(result)

    def test_rejects_when_adaptive_flag_false(self):
        """A SweepResult with n_iters_used set but adaptive=False (e.g. from
        the in-kernel non-adaptive path) must still be rejected."""
        rng = np.random.default_rng(0)
        result = SweepResult(
            h=rng.uniform(-0.3, 0.3, size=(3, 4)),
            h_std=np.zeros((3, 4)),
            orders=np.arange(2, 5),
            cutoffs=np.linspace(0.1, 0.9, 4),
            window="hamming",
            filter_type="lowpass",
            n_iters_used=np.full((3, 4), 3000.0),
            metadata={"adaptive": False},
        )
        with self.assertRaises(ValueError):
            plot_difficulty_map(result)

    def test_returns_figure(self):
        result = _adaptive_result()
        fig = plot_difficulty_map(result)
        self.assertGreaterEqual(len(fig.axes), 2)
        fig.clear()

    def test_legacy_metadata_no_Nmap_bounds(self):
        """Difficulty map with legacy result missing Nmap_min/Nmap falls back
        to data range so it still produces a readable colour scale."""
        rng = np.random.default_rng(0)
        result = SweepResult(
            h=rng.uniform(-0.3, 0.3, size=(3, 4)),
            h_std=np.zeros((3, 4)),
            orders=np.arange(2, 5),
            cutoffs=np.linspace(0.1, 0.9, 4),
            window="hamming",
            filter_type="lowpass",
            n_iters_used=np.full((3, 4), 2000.0, dtype=np.float64),
            metadata={"adaptive": True},
        )
        with TemporaryDirectory() as td:
            path = Path(td) / "diff_legacy.png"
            fig = plot_difficulty_map(result, save_path=path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            fig.clear()


class TestUnpack(unittest.TestCase):
    def test_none_result_and_none_arrays_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _unpack(None, None, None, None)
        self.assertIn("SweepResult", str(ctx.exception))

    def test_result_takes_priority(self):
        h_in = np.array([[0.1, 0.2], [0.3, 0.4]])
        r = SweepResult(
            h=h_in,
            h_std=np.zeros((2, 2)),
            orders=np.array([2, 3]),
            cutoffs=np.array([0.1, 0.9]),
            window="hamming",
            filter_type="lowpass",
            metadata={},
        )
        h_out, Nz, cutoffs = _unpack(r, None, None, None)
        np.testing.assert_array_equal(h_out, h_in)
        np.testing.assert_array_equal(Nz, np.array([1, 2]))
        np.testing.assert_array_equal(cutoffs, np.array([0.1, 0.9]))

    def test_rejects_non_adaptive_result(self):
        """Plotting a non-adaptive result is misleading (single-colour map);
        the function must raise rather than silently produce a useless figure."""
        result = _dummy_result()  # no n_iters_used, adaptive flag absent
        with self.assertRaises(ValueError):
            plot_difficulty_map(result)

    def test_rejects_when_adaptive_flag_false(self):
        """A SweepResult with n_iters_used set but adaptive=False (e.g. from
        the in-kernel non-adaptive path) must still be rejected."""
        rng = np.random.default_rng(0)
        result = SweepResult(
            h=rng.uniform(-0.3, 0.3, size=(3, 4)),
            h_std=np.zeros((3, 4)),
            orders=np.arange(2, 5),
            cutoffs=np.linspace(0.1, 0.9, 4),
            window="hamming",
            filter_type="lowpass",
            n_iters_used=np.full((3, 4), 3000.0),
            metadata={"adaptive": False},
        )
        with self.assertRaises(ValueError):
            plot_difficulty_map(result)

    def test_returns_figure(self):
        result = _adaptive_result()
        fig = plot_difficulty_map(result)
        # Sanity: the figure has exactly one axes (heatmap) plus the
        # colorbar axes — i.e. >= 2 in total.
        self.assertGreaterEqual(len(fig.axes), 2)
        fig.clear()


if __name__ == "__main__":
    unittest.main()
