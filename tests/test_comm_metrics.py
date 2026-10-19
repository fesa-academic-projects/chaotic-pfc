"""tests/test_comm_metrics.py — delay-aligned communication error metrics.

The band-limited channel is a linear-phase FIR filter, so it delays the
received signal by its group delay ``(N_taps - 1) / 2``. These tests pin
the behaviour once the delay is compensated, which is what distinguishes
a genuinely failed recovery from a misaligned one.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.signal import firwin

from chaotic_pfc.comms.channel import fir_channel, ideal_channel
from chaotic_pfc.comms.metrics import comm_metrics, estimate_delay
from chaotic_pfc.comms.receiver import receive, receive_order_n
from chaotic_pfc.comms.transmitter import transmit, transmit_order_n
from chaotic_pfc.config import DEFAULT_CONFIG as cfg
from chaotic_pfc.dynamics.signals import text_message

N = 100_000
TAPS = cfg.channel.num_taps
GDELAY = (TAPS - 1) // 2


class TestCommDelayAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        alpha, beta = cfg.comm.henon.alpha, cfg.comm.henon.beta
        mu, per, tr = cfg.comm.mu, cfg.comm.message_period, cfg.comm.transient
        cls.bits, cls.tr = per, tr
        m = text_message("hello world", N, bit_period=per)
        cls.m = m

        rng = np.random.default_rng(cfg.seed)
        s = transmit(m, mu=mu, alpha=alpha, beta=beta, x0=0.0, y0=0.0)
        cls.mhat_ideal = receive(
            ideal_channel(s), mu=mu, alpha=alpha, beta=beta, y0=rng.random(), z0=rng.random()
        )

        rng = np.random.default_rng(cfg.seed)
        s = transmit(m, mu=mu, alpha=alpha, beta=beta, x0=0.0, y0=0.0)
        r, _ = fir_channel(s, cutoff=cfg.channel.cutoff, num_taps=TAPS)
        cls.mhat_fir = receive(r, mu=mu, alpha=alpha, beta=beta, y0=rng.random(), z0=rng.random())

        rng = np.random.default_rng(cfg.seed)
        c = firwin(
            numtaps=cfg.internal_fir.num_taps,
            cutoff=cfg.internal_fir.cutoff,
            window="hamming",
            pass_zero=True,
            fs=2.0,
        )
        nc = len(c)
        x0 = 0.5 * rng.random(nc)
        y0 = 0.5 * rng.random(nc)
        s, _ = transmit_order_n(m, c, mu=mu, alpha=alpha, beta=beta, x0=x0)
        r, _ = fir_channel(s, cutoff=cfg.channel.cutoff, num_taps=TAPS)
        cls.mhat_order = receive_order_n(r, c, mu=mu, alpha=alpha, beta=beta, y0=y0)[0]

    def _metrics(self, mhat: np.ndarray, delay: int):
        return comm_metrics(self.m, mhat, transient=self.tr, bit_period=self.bits, delay=delay)

    def test_ideal_channel_perfect_recovery(self) -> None:
        met = self._metrics(self.mhat_ideal, 0)
        self.assertEqual(met["ber"], 0.0)
        self.assertLess(met["mse"], 1e-20)  # machine precision
        self.assertGreater(met["correlation"], 0.999999)

    def test_bandlimited_with_feedback_recovers(self) -> None:
        # measured at N=100000: BER 0, MSE ~2e-5, corr 1.0000. The MSE
        # bound leaves room for the residual synchronisation noise.
        met = self._metrics(self.mhat_order, GDELAY)
        self.assertLess(met["ber"], 0.01)
        self.assertLess(met["mse"], 1e-3)
        self.assertGreater(met["correlation"], 0.99)

    def test_bandlimited_without_feedback_fails(self) -> None:
        # measured: BER ~0.43 (near the 0.5 coin-flip); the sign of m_hat
        # carries almost no information about the message.
        met = self._metrics(self.mhat_fir, GDELAY)
        self.assertGreater(met["ber"], 0.3)
        self.assertLess(met["ber"], 0.6)
        self.assertLess(abs(met["correlation"]), 0.1)

    def test_delay_estimate_matches_group_delay(self) -> None:
        self.assertEqual(estimate_delay(self.m, self.mhat_ideal, transient=self.tr)["lag"], 0)
        self.assertEqual(estimate_delay(self.m, self.mhat_order, transient=self.tr)["lag"], GDELAY)

    def test_estimate_delay_constant_signal_no_warning(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            est = estimate_delay(np.ones(100), np.ones(100), transient=10, max_lag=20)
        self.assertEqual(est["lag"], 0)
        self.assertEqual(est["corr_at_lag"], 0.0)
        self.assertEqual(est["corr_at_zero"], 0.0)

    def test_comm_metrics_rejects_negative_delay_and_short_signal(self) -> None:
        with self.assertRaisesRegex(ValueError, "delay must be >= 0"):
            comm_metrics(np.ones(100), np.ones(100), transient=10, bit_period=20, delay=-1)
        with self.assertRaisesRegex(ValueError, "fewer samples"):
            comm_metrics(np.ones(25), np.ones(25), transient=10, bit_period=20, delay=0)


if __name__ == "__main__":
    unittest.main()
