"""tests/test_cli_smoke.py — End-to-end smoke tests for CLI subcommands.

Complements :mod:`tests.test_cli` (parser-level checks) by actually
executing each subcommand in-process via :func:`chaotic_pfc.cli.main`
and asserting that it exits cleanly and produces its declared artefact
(figure files for visual experiments, CSV tables for Lyapunov).

Each test runs in an isolated temporary CWD so that default output
paths like ``figures/`` (from :class:`PlotConfig`) land in the tmp
directory and do not pollute the working tree.
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

import pytest

from chaotic_pfc.cli import main

# matplotlib's Agg backend is chatty in headless tests; none of the
# warnings are actionable here.
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


class _IsolatedCwdMixin:
    """Chdir into a fresh TemporaryDirectory for each test.

    Stored on ``self.workdir`` as a :class:`~pathlib.Path`. The original
    CWD is restored and the directory cleaned up automatically via
    :meth:`addCleanup`.
    """

    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self._tmp.name)
        os.chdir(self.workdir)
        self.addCleanup(self._restore)

    def _restore(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def assertHasFigure(self, root=None):
        """Fail unless at least one PNG or SVG exists under *root*."""
        root = Path(root) if root is not None else self.workdir
        found = list(root.rglob("*.png")) + list(root.rglob("*.svg"))
        self.assertTrue(found, f"no figure (*.png or *.svg) found under {root}")


# ───────────────────────────────────────────────────────────────────────
# Experiment subcommands
# ───────────────────────────────────────────────────────────────────────


class TestAttractorsSmoke(_IsolatedCwdMixin, unittest.TestCase):
    def test_save_produces_figure(self):
        code = main(["run", "attractors", "--save", "--no-display", "--steps", "500"])
        self.assertEqual(code, 0)
        self.assertHasFigure()


class TestSensitivitySmoke(_IsolatedCwdMixin, unittest.TestCase):
    def test_save_runs(self):
        code = main(["run", "sensitivity", "--save", "--no-display", "--steps", "200"])
        self.assertEqual(code, 0)
        self.assertHasFigure()

    def test_custom_epsilon(self):
        code = main(
            [
                "run",
                "sensitivity",
                "--save",
                "--no-display",
                "--steps",
                "200",
                "--epsilon",
                "1e-8",
            ]
        )
        self.assertEqual(code, 0)


class TestCommIdealSmoke(_IsolatedCwdMixin, unittest.TestCase):
    def test_save_runs(self):
        code = main(["run", "comm-ideal", "--save", "--no-display", "--N", "2000"])
        self.assertEqual(code, 0)
        self.assertHasFigure()

    def test_custom_mu(self):
        code = main(
            [
                "run",
                "comm-ideal",
                "--save",
                "--no-display",
                "--N",
                "2000",
                "--mu",
                "0.02",
            ]
        )
        self.assertEqual(code, 0)


class TestCommFirSmoke(_IsolatedCwdMixin, unittest.TestCase):
    def test_default_flags(self):
        code = main(["run", "comm-fir", "--save", "--no-display", "--N", "2000"])
        self.assertEqual(code, 0)
        self.assertHasFigure()

    def test_custom_cutoff_and_taps(self):
        code = main(
            [
                "run",
                "comm-fir",
                "--save",
                "--no-display",
                "--N",
                "2000",
                "--cutoff",
                "0.3",
                "--taps",
                "31",
            ]
        )
        self.assertEqual(code, 0)


class TestCommOrderNSmoke(_IsolatedCwdMixin, unittest.TestCase):
    def test_save_runs(self):
        code = main(["run", "comm-order-n", "--save", "--no-display", "--N", "2000"])
        self.assertEqual(code, 0)
        self.assertHasFigure()


class TestLyapunovSmoke(_IsolatedCwdMixin, unittest.TestCase):
    """Lyapunov saves CSV under --data-dir; --no-display is a no-op here."""

    def test_save_csv_with_small_ensemble(self):
        data_dir = self.workdir / "lyap_out"
        code = main(
            [
                "run",
                "lyapunov",
                "--save",
                "--no-display",
                "--n-ci",
                "3",
                "--Nitera",
                "500",
                "--Ndiscard",
                "100",
                "--data-dir",
                str(data_dir),
            ]
        )
        self.assertEqual(code, 0)
        csv_files = list(data_dir.rglob("*.csv"))
        self.assertTrue(csv_files, f"no CSV produced under {data_dir}")


# ───────────────────────────────────────────────────────────────────────
# run all — end-to-end pipeline smoke test
# ───────────────────────────────────────────────────────────────────────


class TestRunAllSmoke(_IsolatedCwdMixin, unittest.TestCase):
    """Exercise :mod:`cli.run_all` top-to-bottom in quick mode.

    This test is slow (tens of seconds) because it chains every
    subcommand in sequence, but it is the only path that actually
    covers ``cli/run_all.py``.
    """

    @pytest.mark.slow
    def test_run_all_quick(self):
        code = main(["run", "all", "--no-display", "--quick-sweep"])
        self.assertEqual(code, 0)


# ───────────────────────────────────────────────────────────────────────
# Adaptive flag — argument validation
# ───────────────────────────────────────────────────────────────────────
#
# The full happy-path of ``--adaptive`` (an actual sweep with early-stop)
# is exercised by the unit tests in test_sweep.py — wiring is the only
# CLI-specific surface here, so we limit smoke coverage to the rejection
# paths (mutually-exclusive flag combinations).


class TestAdaptiveCliRejection(_IsolatedCwdMixin, unittest.TestCase):
    """Verify that incompatible flag combinations are rejected with exit 2."""

    def test_run_sweep_compute_quick_with_adaptive_rejected(self):
        """--quick already shrinks Nmap; --adaptive on top is redundant."""
        code = main(["run", "sweep", "compute", "--quick", "--adaptive", "--no-display"])
        self.assertEqual(code, 2)

    def test_run_all_skip_sweep_with_adaptive_rejected(self):
        """--adaptive only affects the sweep step; --skip-sweep nullifies it."""
        code = main(["run", "all", "--no-display", "--skip-sweep", "--adaptive"])
        self.assertEqual(code, 2)

    def test_run_all_quick_with_adaptive_rejected(self):
        code = main(["run", "all", "--no-display", "--quick-sweep", "--adaptive"])
        self.assertEqual(code, 2)


# ───────────────────────────────────────────────────────────────────────
# Adaptive flag — happy-path on the run_compute wiring
# ───────────────────────────────────────────────────────────────────────


class TestSweepComputeAdaptiveWiring(_IsolatedCwdMixin, unittest.TestCase):
    """Verify --adaptive propagates from CLI args to run_sweep().

    Avoids invoking the full sweep (40×100 grid would take several
    minutes). Instead, calls run_compute() directly with a Namespace
    that mimics what argparse would build, after monkey-patching
    run_sweep to (1) record the kwargs it received and (2) return a
    minimal result. Exercises the actual wiring code without the
    numerical cost.
    """

    def test_adaptive_args_forwarded_to_run_sweep(self):
        """``run_compute`` must forward ``adaptive``/``Nmap_min``/``tol``
        to :func:`chaotic_pfc.sweep.run_sweep`. We monkey-patch the
        symbol on the source module (``chaotic_pfc.sweep``); the CLI
        does ``from chaotic_pfc.analysis.sweep import run_sweep`` *inside*
        ``run_compute`` so the patch must precede that local import."""
        import argparse
        from unittest.mock import patch

        import numpy as np

        import chaotic_pfc.analysis.sweep as sweep_mod
        from chaotic_pfc.cli import sweep as cli_sweep

        captured: dict = {}

        def fake_run_sweep(**kwargs):
            captured.update(kwargs)
            # ``orders`` / ``cutoffs`` may be None (run_sweep default
            # path) or an ndarray — avoid ``a or default`` because that
            # triggers an ambiguous-truth-value error on arrays.
            orders = kwargs.get("orders")
            if orders is None:
                orders = np.array([2, 3])
            cutoffs = kwargs.get("cutoffs")
            if cutoffs is None:
                cutoffs = np.array([0.5])
            n = len(orders)
            m = len(cutoffs)
            h = np.full((n, m), 0.1)
            return sweep_mod.SweepResult(
                h=h,
                h_std=np.zeros_like(h),
                orders=np.asarray(orders),
                cutoffs=np.asarray(cutoffs),
                window=kwargs.get("window", "hamming"),
                filter_type=kwargs.get("filter_type", "lowpass"),
                n_iters_used=np.full((n, m), 700.0),
                metadata={"Nmap": kwargs.get("Nmap", 3000)},
            )

        ns = argparse.Namespace(
            window="hamming",
            filter_type="lowpass",
            all=False,
            quick=True,  # cheapest path; explicit (orders/cutoffs) used
            kaiser_beta=5.0,
            data_dir=str(self.workdir / "data"),
            adaptive=False,
            Nmap_min=500,
            tol=1e-3,
            no_display=True,
            save=False,
        )
        with (
            patch.object(sweep_mod, "run_sweep", fake_run_sweep),
            patch.object(sweep_mod, "save_sweep", lambda *a, **k: None),
        ):
            code = cli_sweep.run_compute(ns)
        self.assertEqual(code, 0)
        self.assertFalse(captured["adaptive"])
        self.assertNotIn("Nmap_min", captured)  # not forwarded when adaptive=False

        # Now exercise the adaptive=True path with a non-quick grid.
        captured.clear()
        ns.quick = False
        ns.adaptive = True
        ns.Nmap_min = 300
        ns.tol = 5e-4
        with (
            patch.object(sweep_mod, "run_sweep", fake_run_sweep),
            patch.object(sweep_mod, "save_sweep", lambda *a, **k: None),
        ):
            code = cli_sweep.run_compute(ns)
        self.assertEqual(code, 0)
        self.assertTrue(captured["adaptive"])
        self.assertEqual(captured["Nmap_min"], 300)
        self.assertEqual(captured["tol"], 5e-4)


class TestDCSKSmoke(unittest.TestCase, _IsolatedCwdMixin):
    def setUp(self):
        _IsolatedCwdMixin.setUp(self)

    def test_run_dcsk_no_display(self):
        code = main(["run", "dcsk", "--no-display"])
        self.assertEqual(code, 0)


class TestSweepPlotSmoke(unittest.TestCase, _IsolatedCwdMixin):
    def setUp(self):
        _IsolatedCwdMixin.setUp(self)

    @pytest.mark.slow
    def test_run_sweep_plot_hamming_lowpass(self):
        # Sweep plot reads from versioned .npz files in the project's
        # data/ directory, not from the isolated temp CWD.
        proj_root = Path(__file__).resolve().parents[1]
        code = main(
            [
                "run",
                "sweep",
                "plot",
                "--window",
                "hamming",
                "--filter",
                "lowpass",
                "--no-display",
                "--data-dir",
                str(proj_root / "data" / "sweeps"),
                "--figures-dir",
                str(self.workdir / "figures"),
            ]
        )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
