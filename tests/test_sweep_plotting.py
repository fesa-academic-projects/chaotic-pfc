"""tests/test_sweep_plotting.py — Tests for the sweep plotting module."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")  # must come before pyplot is imported anywhere

import numpy as np
from matplotlib.collections import PolyCollection, QuadMesh
from matplotlib.colors import to_rgba
from matplotlib.image import AxesImage

from chaotic_pfc.analysis.sweep import SweepResult
from chaotic_pfc.analysis.sweep._io import save_sweep
from chaotic_pfc.analysis.sweep_plotting import (
    _YTICKS,
    COLOR_CHAOTIC,
    COLOR_PERIODIC,
    COLOR_UNBOUNDED,
    DIFFICULTY_FIGURE_FILENAME,
    FIGURE_FILENAMES,
    _interleaved_expand,
    _unpack,
    classify,
    plot_all,
    plot_chaotic_density,
    plot_chaotic_map,
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


# ═══════════════════════════════════════════════════════════════════════════
# Chaotic-region union & density maps
# ═══════════════════════════════════════════════════════════════════════════


class TestChaoticRegions(unittest.TestCase):
    """Tests for :func:`plot_chaotic_map` and :func:`plot_chaotic_density`."""

    @staticmethod
    def _make_sweep_dir(base: Path) -> Path:
        """Create a minimal sweep directory with 2 windows × 2 filters."""
        rng = np.random.default_rng(42)
        for w_key, w_display in [("hamming", "Hamming"), ("hann", "Hann")]:
            for ft in ("lowpass", "highpass"):
                h = rng.uniform(-0.5, 0.5, size=(4, 6))
                h[0, 0] = np.nan  # one divergent
                sr = SweepResult(
                    h=h,
                    h_std=np.abs(h) * 0.1,
                    orders=np.arange(2, 6, dtype=np.float64),
                    cutoffs=np.linspace(0.1, 0.9, 6, dtype=np.float64),
                    window=w_key,
                    filter_type=ft,
                    metadata={"Nitera": 10, "Nmap": 50},
                )
                out_dir = base / f"{w_display} ({ft})"
                out_dir.mkdir(parents=True)
                save_sweep(sr, out_dir / "variables_lyapunov.npz")
        return base

    def test_chaotic_map_saves_svg(self):
        with TemporaryDirectory() as td:
            base = Path(td)
            sweep_dir = self._make_sweep_dir(base / "sweeps")
            stem = base / "fig_chaotic_map"
            fig = plot_chaotic_map(sweep_dir, save_path=stem)
            self.assertTrue(stem.with_suffix(".svg").exists())
            self.assertGreater(stem.with_suffix(".svg").stat().st_size, 0)
            fig.clear()

    def test_chaotic_density_saves_svg(self):
        with TemporaryDirectory() as td:
            base = Path(td)
            sweep_dir = self._make_sweep_dir(base / "sweeps")
            stem = base / "fig_chaotic_density"
            fig = plot_chaotic_density(sweep_dir, save_path=stem)
            self.assertTrue(stem.with_suffix(".svg").exists())
            self.assertGreater(stem.with_suffix(".svg").stat().st_size, 0)
            fig.clear()

    def test_chaotic_map_raises_on_empty_dir(self):
        with TemporaryDirectory() as td:
            empty = Path(td) / "empty"
            empty.mkdir()
            with self.assertRaises(FileNotFoundError):
                plot_chaotic_map(empty)

    def test_chaotic_density_raises_on_empty_dir(self):
        with TemporaryDirectory() as td:
            empty = Path(td) / "empty"
            empty.mkdir()
            with self.assertRaises(FileNotFoundError):
                plot_chaotic_density(empty)

    def test_chaotic_map_returns_figure(self):
        with TemporaryDirectory() as td:
            base = Path(td)
            sweep_dir = self._make_sweep_dir(base / "sweeps")
            fig = plot_chaotic_map(sweep_dir)
            self.assertGreaterEqual(len(fig.axes), 1)  # ax (no cbar for binary)
            fig.clear()

    def test_chaotic_density_returns_figure(self):
        with TemporaryDirectory() as td:
            base = Path(td)
            sweep_dir = self._make_sweep_dir(base / "sweeps")
            fig = plot_chaotic_density(sweep_dir)
            self.assertGreaterEqual(len(fig.axes), 2)  # ax + cbar
            fig.clear()

    def test_interleaved_expand_shape(self):
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        expanded = _interleaved_expand(data, data_slots=2, gap_slots=1)
        self.assertEqual(expanded.shape, (9, 4))
        np.testing.assert_array_equal(expanded[0, :], data[0, :])
        np.testing.assert_array_equal(expanded[1, :], data[0, :])
        self.assertTrue(np.all(np.isnan(expanded[2, :])))


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2 — plot_classification_interleaved (PolyCollection)
# ═══════════════════════════════════════════════════════════════════════════


class TestFig2PolyCollection(unittest.TestCase):
    """Structural tests for the PolyCollection-based fig 2."""

    @staticmethod
    def _synthetic_h(ncoef: int = 3, ncut: int = 5) -> SweepResult:
        rng = np.random.default_rng(0)
        h = rng.uniform(-0.5, 0.5, size=(ncoef, ncut))
        h[0, 0] = np.nan
        h[-1, -1] = np.nan
        return SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=np.arange(2, 2 + ncoef),
            cutoffs=np.linspace(0.1, 0.9, ncut),
            window="hamming",
            filter_type="lowpass",
            metadata={},
        )

    def test_fig2_contains_no_mesh(self):
        """Axes must contain no QuadMesh / AxesImage, exactly one PolyCollection."""
        result = self._synthetic_h()
        fig = plot_classification_interleaved(result=result)
        ax = fig.axes[0]

        for coll in ax.collections:
            self.assertNotIsInstance(coll, QuadMesh)
        self.assertEqual(len(ax.images), 0)

        poly_colls = [c for c in ax.collections if isinstance(c, PolyCollection)]
        self.assertEqual(len(poly_colls), 1)
        fig.clear()

    def test_fig2_rect_count_matches_rle(self):
        """Polygon count must equal the sum of RLE runs per column."""
        orders = np.array([2, 3, 5])
        cutoffs = np.array([0.2, 0.4, 0.6, 0.8, 0.95])
        h = np.array(
            [
                [-0.5, -0.3, 0.1, 0.2, np.nan],
                [0.3, 0.5, 0.2, -0.1, -0.2],
                [-0.1, -0.2, 0.5, 0.3, 0.1],
            ]
        )
        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=orders,
            cutoffs=cutoffs,
            window="hamming",
            filter_type="lowpass",
            metadata={},
        )

        h_color = classify(h)
        expected_runs = 0
        for i in range(h.shape[0]):
            col = h_color[i, :]
            diffs = np.diff(col)
            change_idx = np.flatnonzero(diffs != 0)
            expected_runs += len(change_idx) + 1

        fig = plot_classification_interleaved(result=result)
        ax = fig.axes[0]
        poly_coll = next(c for c in ax.collections if isinstance(c, PolyCollection))
        n_polys = len(poly_coll.get_paths())
        self.assertEqual(n_polys, expected_runs)
        fig.clear()

    def test_fig2_facecolors_match_classification(self):
        """Each polygon's facecolor must match the class color of its run."""
        orders = np.array([2, 3, 5])
        cutoffs = np.array([0.2, 0.4, 0.6, 0.8, 0.95])
        h = np.array(
            [
                [-0.5, -0.3, 0.1, 0.2, np.nan],
                [0.3, 0.5, 0.2, -0.1, -0.2],
                [-0.1, -0.2, 0.5, 0.3, 0.1],
            ]
        )
        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=orders,
            cutoffs=cutoffs,
            window="hamming",
            filter_type="lowpass",
            metadata={},
        )

        h_color = classify(h)

        # Build expected colors per polygon using the same RLE as the code
        class_colors = {-1.0: COLOR_PERIODIC, 0.0: COLOR_CHAOTIC, 2.0: COLOR_UNBOUNDED}
        expected_fcs: list[tuple[float, float, float, float]] = []
        for i in range(h.shape[0]):
            col = h_color[i, :]
            diffs = np.diff(col)
            change_idx = np.flatnonzero(diffs != 0)
            starts = np.concatenate([[0], change_idx + 1])
            run_values = col[starts]
            for code in run_values:
                expected_fcs.append(to_rgba(class_colors[code]))

        fig = plot_classification_interleaved(result=result)
        ax = fig.axes[0]
        poly_coll = next(c for c in ax.collections if isinstance(c, PolyCollection))
        actual_fcs = poly_coll.get_facecolors()

        self.assertEqual(len(actual_fcs), len(expected_fcs))
        for actual, expected in zip(actual_fcs, expected_fcs, strict=True):
            np.testing.assert_array_almost_equal(actual, expected, decimal=6)
        fig.clear()

    def test_fig2_geometry_matches_slot_layout(self):
        """X vertices must match the slot layout; y vertices must match edges."""
        orders = np.array([2, 3, 5])
        cutoffs = np.array([0.2, 0.4, 0.6, 0.8, 0.95])
        h = np.array(
            [
                [-0.5, -0.3, 0.1, 0.2, np.nan],
                [0.3, 0.5, 0.2, -0.1, -0.2],
                [-0.1, -0.2, 0.5, 0.3, 0.1],
            ]
        )
        data_slots, gap_slots = 3, 1
        slot_total = data_slots + gap_slots
        Ncoef = h.shape[0]
        Ncut = h.shape[1]

        # Expected y_edges
        cut = np.asarray(cutoffs, dtype=np.float64)
        mid = 0.5 * (cut[:-1] + cut[1:])
        y_edges = np.concatenate(
            [
                [cut[0] - (mid[0] - cut[0])],
                mid,
                [cut[-1] + (cut[-1] - mid[-1])],
            ]
        )

        h_color = classify(h)
        # Build expected vertices per polygon
        expected_verts: list[list[tuple[float, float]]] = []
        for i in range(Ncoef):
            x0 = i * slot_total - 0.5
            x1 = x0 + data_slots
            col = h_color[i, :]
            diffs = np.diff(col)
            change_idx = np.flatnonzero(diffs != 0)
            starts = np.concatenate([[0], change_idx + 1])
            ends = np.concatenate([change_idx, [Ncut - 1]])
            for s, e in zip(starts, ends, strict=True):
                expected_verts.append(
                    [
                        (x0, y_edges[s]),
                        (x1, y_edges[s]),
                        (x1, y_edges[e + 1]),
                        (x0, y_edges[e + 1]),
                    ]
                )

        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=orders,
            cutoffs=cutoffs,
            window="hamming",
            filter_type="lowpass",
            metadata={},
        )
        fig = plot_classification_interleaved(result=result)
        ax = fig.axes[0]
        poly_coll = next(c for c in ax.collections if isinstance(c, PolyCollection))
        paths = poly_coll.get_paths()

        self.assertEqual(len(paths), len(expected_verts))
        for path, exp_verts in zip(paths, expected_verts, strict=True):
            actual = path.vertices.tolist()
            # Path may have 5 vertices if closed (last = first); check first 4
            if len(actual) == 5:
                actual = actual[:4]
            for a, e in zip(actual, exp_verts, strict=True):
                self.assertAlmostEqual(a[0], e[0], places=10)
                self.assertAlmostEqual(a[1], e[1], places=10)
        fig.clear()

    def test_fig2_axes_contract_unchanged(self):
        """Axes limits, tick positions and labels must match expected values."""
        orders = np.array([2, 3, 5])
        cutoffs = np.array([0.2, 0.4, 0.6, 0.8, 0.95])
        h = np.array(
            [
                [-0.5, -0.3, 0.1, 0.2, np.nan],
                [0.3, 0.5, 0.2, -0.1, -0.2],
                [-0.1, -0.2, 0.5, 0.3, 0.1],
            ]
        )
        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=orders,
            cutoffs=cutoffs,
            window="hamming",
            filter_type="lowpass",
            metadata={},
        )
        fig = plot_classification_interleaved(result=result)
        ax = fig.axes[0]

        # xlim, ylim
        self.assertAlmostEqual(ax.get_xlim()[0], -0.5, places=10)
        self.assertAlmostEqual(ax.get_xlim()[1], 11.5, places=10)
        self.assertAlmostEqual(ax.get_ylim()[0], 0.0, places=10)
        self.assertAlmostEqual(ax.get_ylim()[1], 1.0, places=10)

        # yticks: _YTICKS = arange(0, 1.01, 0.1)
        np.testing.assert_array_almost_equal(ax.get_yticks(), _YTICKS, decimal=10)

        # xticks: Nz = [1, 2, 4], tick_vals = [1, 5]
        # 5 not in Nz, so only label for 1 at center = 0*4 + 1 = 1
        xticks = ax.get_xticks()
        self.assertEqual(len(xticks), 1)
        self.assertAlmostEqual(xticks[0], 1.0, places=10)
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(xticklabels, ["1"])

        fig.clear()

    def test_fig2_single_cutoff_grid(self):
        """Single cutoff must not crash and must produce valid y_edges."""
        orders = np.array([2, 3])
        cutoffs = np.array([0.5])
        h = np.array(
            [
                [-0.1],
                [0.2],
            ]
        )
        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=orders,
            cutoffs=cutoffs,
            window="hamming",
            filter_type="lowpass",
            metadata={},
        )
        fig = plot_classification_interleaved(result=result)
        ax = fig.axes[0]

        poly_colls = [c for c in ax.collections if isinstance(c, PolyCollection)]
        self.assertEqual(len(poly_colls), 1)

        paths = poly_colls[0].get_paths()
        self.assertGreater(len(paths), 0)

        # y_edges should be [0.0, 1.0] for cutoff=0.5
        for path in paths:
            verts = path.vertices
            if verts.shape[0] >= 4:
                ys = verts[:4, 1]
                self.assertGreaterEqual(ys.min(), 0.0)
                self.assertLessEqual(ys.max(), 1.0)
        fig.clear()


if __name__ == "__main__":
    unittest.main()
