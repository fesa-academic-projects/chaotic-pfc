"""tests/test_plotting.py — Smoke tests for plotting functions."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from chaotic_pfc._i18n import t
from chaotic_pfc.dynamics.spectral import psd_normalised
from chaotic_pfc.plotting.figures import (
    PlotGridOptions,
    latex_available,
    plot_attractor,
    plot_comm_grid,
    plot_sensitivity,
    setup_rc,
)


class TestPlottingSmoke(unittest.TestCase):
    def test_plot_attractor_returns_figure(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal(100)
        Y = rng.standard_normal(100)
        fig = plot_attractor(X, Y)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_sensitivity_returns_figure(self):
        n = np.arange(50)
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal(50)
        X2 = rng.standard_normal(50)
        fig = plot_sensitivity(n, X1, X2)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_comm_grid_returns_figure(self):
        N = 5000
        n = np.arange(N)
        rng = np.random.default_rng(42)
        m = rng.standard_normal(N)
        s = rng.standard_normal(N)
        r = rng.standard_normal(N)
        m_hat = rng.standard_normal(N)
        omega, psd_m = psd_normalised(m)
        _, psd_s = psd_normalised(s)
        _, psd_r = psd_normalised(r)
        _, psd_mhat = psd_normalised(m_hat)
        fig = plot_comm_grid(n, m, s, r, m_hat, omega, psd_m, psd_s, psd_r, psd_mhat)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_comm_grid_i18n_lang(self):
        N = 5000
        n = np.arange(N)
        rng = np.random.default_rng(42)
        m = rng.standard_normal(N)
        s = rng.standard_normal(N)
        r = rng.standard_normal(N)
        m_hat = rng.standard_normal(N)
        omega, psd_m = psd_normalised(m)
        _, psd_s = psd_normalised(s)
        _, psd_r = psd_normalised(r)
        _, psd_mhat = psd_normalised(m_hat)
        for lang in ("pt", "en"):
            fig = plot_comm_grid(
                n,
                m,
                s,
                r,
                m_hat,
                omega,
                psd_m,
                psd_s,
                psd_r,
                psd_mhat,
                lang=lang,
                suptitle=t("comm.ideal", lang=lang),
            )
            self.assertIsNotNone(fig)
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plot_grid_options_dataclass(self):
        opts = PlotGridOptions()
        self.assertEqual(opts.time_window, slice(0, 300))
        self.assertEqual(opts.suptitle, "")
        self.assertIsNone(opts.save_path)

    def test_comm_grid_explicit_default_not_overridden_by_opts(self):
        """Passing y_lim_msg explicitly with the same value as the
        default must NOT be treated as an omission — the explicit
        value must take precedence over opts."""
        N = 2048
        n = np.arange(N)
        rng = np.random.default_rng(42)
        z = rng.standard_normal(N)
        omega, psd = psd_normalised(z)
        opts = PlotGridOptions(y_lim_msg=(-3.0, 3.0))
        fig = plot_comm_grid(
            n,
            z,
            z,
            z,
            z,
            omega,
            psd,
            psd,
            psd,
            psd,
            opts=opts,
            y_lim_msg=(-1.5, 1.5),
        )
        # Explicit arg (-1.5, 1.5) must win over opts (-3, 3)
        for ax in fig.axes:
            if ax.get_ylabel() and "m[" in ax.get_ylabel():
                self.assertEqual(ax.get_ylim(), (-1.5, 1.5))
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestSetupRc(unittest.TestCase):
    def test_setup_rc_idempotent(self):
        setup_rc()
        setup_rc()

    def test_setup_rc_sets_font_family(self):
        setup_rc()
        family = plt.rcParams["font.family"]
        if isinstance(family, list):
            family = family[0]
        self.assertIn(family, {"STIXGeneral", "serif"})

    def test_latex_available_returns_bool(self):
        result = latex_available()
        self.assertIsInstance(result, bool)
