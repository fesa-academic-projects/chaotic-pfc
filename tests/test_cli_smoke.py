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

import matplotlib

matplotlib.use("Agg")  # must precede any pyplot import, including transitive

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

    @unittest.skipIf(
        os.environ.get("CHAOTIC_PFC_SKIP_SLOW") == "1",
        "slow test disabled via CHAOTIC_PFC_SKIP_SLOW=1",
    )
    def test_run_all_quick(self):
        code = main(["run", "all", "--no-display", "--quick-sweep"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
