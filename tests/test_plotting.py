"""tests/test_plotting.py — Smoke tests for plotting functions."""

import unittest

import numpy as np

from chaotic_pfc._i18n import t
from chaotic_pfc.dynamics.spectral import psd_normalised
from chaotic_pfc.plotting.figures import (
    PlotGridOptions,
    plot_attractor,
    plot_comm_grid,
    plot_sensitivity,
)


class TestPlottingSmoke(unittest.TestCase):
    def test_plot_attractor_returns_figure(self):
        X = np.random.randn(100)
        Y = np.random.randn(100)
        fig = plot_attractor(X, Y)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_sensitivity_returns_figure(self):
        n = np.arange(50)
        X1 = np.random.randn(50)
        X2 = np.random.randn(50)
        fig = plot_sensitivity(n, X1, X2)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_comm_grid_returns_figure(self):
        N = 5000
        n = np.arange(N)
        m = np.random.randn(N)
        s = np.random.randn(N)
        r = np.random.randn(N)
        m_hat = np.random.randn(N)
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
        m = np.random.randn(N)
        s = np.random.randn(N)
        r = np.random.randn(N)
        m_hat = np.random.randn(N)
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
