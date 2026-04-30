"""tests/test_sweep_plotting.py — Tests for the sweep plotting module."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")  # must come before pyplot is imported anywhere

import numpy as np

from chaotic_pfc.sweep import SweepResult
from chaotic_pfc.sweep_plotting import (
    FIGURE_FILENAMES,
    classify,
    plot_all,
    plot_classification_interleaved,
    plot_classification_separated,
    plot_classification_simple,
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
        # λ_max == 0 is the boundary case and is classified as periodic.
        out = classify(np.array([[0.0, -0.0]]))
        self.assertTrue(np.all(out == -1))


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

    def test_classification_simple_saves_file(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "class.png"
            fig = plot_classification_simple(self.result, save_path=path)
            self.assertTrue(path.exists())
            fig.clear()

    def test_classification_separated_saves_file(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "class_sep.png"
            fig = plot_classification_separated(self.result, save_path=path)
            self.assertTrue(path.exists())
            fig.clear()

    def test_classification_interleaved_saves_file(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "class_inter.png"
            fig = plot_classification_interleaved(self.result, save_path=path)
            self.assertTrue(path.exists())
            fig.clear()


class TestPlotAll(unittest.TestCase):
    def test_plot_all_creates_four_files(self):
        result = _dummy_result()
        with TemporaryDirectory() as td:
            paths = plot_all(result, Path(td), fmt="png")
            self.assertEqual(len(paths), 4)
            self.assertEqual(len(paths), len(FIGURE_FILENAMES))
            for p in paths:
                self.assertTrue(p.exists())
                self.assertGreater(p.stat().st_size, 0)

    def test_plot_all_respects_fmt(self):
        result = _dummy_result()
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
            self.assertEqual(len(paths), 4)


if __name__ == "__main__":
    unittest.main()
