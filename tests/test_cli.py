"""tests/test_cli.py — Smoke tests for the unified CLI parser."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from chaotic_pfc.cli import build_parser, main


class TestParserStructure(unittest.TestCase):
    """Ensure the parser tree matches what README and help text promise."""

    def test_build_parser_returns_parser(self):
        parser = build_parser()
        self.assertIsNotNone(parser)

    def test_no_args_prints_help_and_returns_2(self):
        """Invocation with no subcommand should not crash; it should show help."""
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            code = main([])
        self.assertEqual(code, 2)
        self.assertIn("chaotic-pfc", buf_out.getvalue())

    def test_version_flag(self):
        """--version prints version and exits 0 (argparse convention)."""
        from chaotic_pfc._version import __version__

        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            main(["--version"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn(__version__, buf.getvalue())


class TestExperimentsRegistered(unittest.TestCase):
    """Every promised subcommand must parse without errors."""

    EXPERIMENTS = (
        "attractors",
        "sensitivity",
        "comm-ideal",
        "comm-fir",
        "comm-order-n",
        "lyapunov",
        "dcsk",
        "analysis",
        "all",
    )

    def test_each_experiment_help_parses(self):
        """`chaotic-pfc run <exp> --help` must succeed for every subcommand."""
        for exp in self.EXPERIMENTS:
            with self.subTest(experiment=exp):
                buf = io.StringIO()
                with redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
                    main(["run", exp, "--help"])
                self.assertEqual(cm.exception.code, 0)
                self.assertIn(exp, buf.getvalue())

    def test_sweep_compute_help_parses(self):
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            main(["run", "sweep", "compute", "--help"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("--quick", buf.getvalue())

    def test_sweep_plot_help_parses(self):
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            main(["run", "sweep", "plot", "--help"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("--fmt", buf.getvalue())


class TestArgumentValidation(unittest.TestCase):
    """argparse should reject unknown or bad values."""

    def test_unknown_experiment_rejected(self):
        """`chaotic-pfc run nonsense` must exit with error."""
        buf = io.StringIO()
        with redirect_stderr(buf), self.assertRaises(SystemExit) as cm:
            main(["run", "nonsense"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("invalid choice", buf.getvalue())

    def test_sweep_needs_action(self):
        """`chaotic-pfc run sweep` without compute/plot must fail."""
        buf = io.StringIO()
        with redirect_stderr(buf), self.assertRaises(SystemExit) as cm:
            main(["run", "sweep"])
        self.assertEqual(cm.exception.code, 2)

    def test_mutually_exclusive_sweep_flags(self):
        """--skip-sweep and --quick-sweep can't be combined."""
        import argparse

        parser = build_parser()
        ns = parser.parse_args(["run", "all", "--skip-sweep", "--quick-sweep"])
        # Parser accepts both; conflict check happens inside run(). Simulate it.
        code = ns._run(argparse.Namespace(**vars(ns)))
        self.assertEqual(code, 2)


class TestSweepComputeSmokeTest(unittest.TestCase):
    """Minimal end-to-end: actually invoke sweep compute --quick.

    This is slow (~30 s fresh, <1 s with the Numba cache warm) but is
    the cleanest evidence that the full CLI → sweep.run_compute path
    works without shelling out to subprocess.
    """

    def test_sweep_compute_quick_creates_npz(self):
        import shutil
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            code = main(
                [
                    "run",
                    "sweep",
                    "compute",
                    "--quick",
                    "--data-dir",
                    tmp,
                ]
            )
            self.assertEqual(code, 0)
            # The output path follows the "<Window> (<filter>)" convention.
            npz_files = list(Path(tmp).rglob("variables_lyapunov.npz"))
            self.assertEqual(len(npz_files), 1)
            self.assertGreater(npz_files[0].stat().st_size, 0)
            # Explicit cleanup belt-and-suspenders; tempfile handles it too.
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
