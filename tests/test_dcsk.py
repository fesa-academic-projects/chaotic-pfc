"""tests/test_dcsk.py — Unit tests for the DCSK module."""

import unittest

import numpy as np

from chaotic_pfc.comms.channel import fir_channel
from chaotic_pfc.comms.dcsk import (
    awgn,
    ber,
    channel_impulsive,
    channel_interferers,
    channel_multipath,
    channel_urban,
    dcsk_receive,
    dcsk_transmit,
    efdcsk_receive,
    efdcsk_transmit,
    henon_fir_sequence,
)


class TestHenonFIR(unittest.TestCase):
    def test_output_shape(self):
        seq = henon_fir_sequence(100, n_taps=5, wc=0.9)
        self.assertEqual(seq.shape, (100,))

    def test_output_is_finite(self):
        seq = henon_fir_sequence(500, n_taps=5, wc=0.9)
        self.assertTrue(np.all(np.isfinite(seq)))

    def test_different_params_different_output(self):
        seq_a = henon_fir_sequence(100, n_taps=5, wc=0.9)
        seq_b = henon_fir_sequence(100, n_taps=9, wc=0.5)
        self.assertFalse(np.allclose(seq_a, seq_b))

    def test_deterministic(self):
        s1 = henon_fir_sequence(50, n_taps=5, wc=0.9)
        s2 = henon_fir_sequence(50, n_taps=5, wc=0.9)
        np.testing.assert_array_equal(s1, s2)


class TestDCSKTransmitReceive(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_roundtrip_noiseless(self):
        bits = self.rng.integers(0, 2, 100)
        sig = dcsk_transmit(bits, beta=64)
        rx_bits = dcsk_receive(sig, beta=64)
        self.assertEqual(float(ber(bits, rx_bits)), 0.0)

    def test_output_length(self):
        bits = self.rng.integers(0, 2, 50)
        sig = dcsk_transmit(bits, beta=32)
        self.assertEqual(len(sig), 50 * 2 * 32)

    def test_roundtrip_awgn(self):
        bits = self.rng.integers(0, 2, 200)
        sig = dcsk_transmit(bits, beta=64)
        rx = awgn(sig, 20.0, self.rng)
        rx_bits = dcsk_receive(rx, beta=64)
        self.assertLess(ber(bits, rx_bits), 0.01)

    def test_roundtrip_awgn_low_snr(self):
        bits = self.rng.integers(0, 2, 200)
        sig = dcsk_transmit(bits, beta=64)
        rx = awgn(sig, -6.0, self.rng)
        rx_bits = dcsk_receive(rx, beta=64)
        self.assertGreater(ber(bits, rx_bits), 0.01)
        self.assertLess(ber(bits, rx_bits), 0.5)

    def test_inverted_bit_detection(self):
        # Bit 0: data = +ref,  Bit 1: data = -ref
        bits = np.array([0, 1, 0, 1], dtype=np.int64)
        sig = dcsk_transmit(bits, beta=32)
        # Check that first half and second half of first symbol are equal (bit 0)
        np.testing.assert_array_equal(sig[:32], sig[32:64])
        # Check that second symbol has data = -ref (bit 1)
        np.testing.assert_array_equal(sig[64:96], -sig[96:128])


class TestAWGN(unittest.TestCase):
    def test_noise_adds_variance(self):
        rng = np.random.default_rng(0)
        sig = np.ones(10000)
        noisy = awgn(sig, 0.0, rng)
        self.assertGreater(float(np.std(noisy)), 0.9)

    def test_high_snr_near_identity(self):
        rng = np.random.default_rng(0)
        sig = np.ones(10000)
        noisy = awgn(sig, 80.0, rng)
        np.testing.assert_allclose(noisy, sig, atol=1e-3)


class TestChannels(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        bits = self.rng.integers(0, 2, 100)
        self.sig = dcsk_transmit(bits, beta=64)

    def test_impulsive_shape(self):
        rx = channel_impulsive(self.sig, 20.0, rng=self.rng)
        self.assertEqual(rx.shape, self.sig.shape)
        self.assertTrue(np.all(np.isfinite(rx)))

    def test_multipath_shape(self):
        rx = channel_multipath(self.sig, 20.0, rng=self.rng)
        self.assertEqual(rx.shape, self.sig.shape)
        self.assertTrue(np.all(np.isfinite(rx)))

    def test_interferers_shape(self):
        rx = channel_interferers(self.sig, 20.0, rng=self.rng)
        self.assertEqual(rx.shape, self.sig.shape)
        self.assertTrue(np.all(np.isfinite(rx)))

    def test_urban_shape(self):
        rx = channel_urban(self.sig, 20.0, rng=self.rng)
        self.assertEqual(rx.shape, self.sig.shape)
        self.assertTrue(np.all(np.isfinite(rx)))


class TestBER(unittest.TestCase):
    def test_perfect(self):
        tx = np.array([0, 1, 0, 1])
        self.assertEqual(ber(tx, tx), 0.0)

    def test_half_wrong(self):
        tx = np.array([0, 1, 0, 1])
        rx = np.array([1, 0, 0, 1])
        self.assertEqual(ber(tx, rx), 0.5)

    def test_all_wrong(self):
        tx = np.array([0, 1, 0, 1])
        rx = np.array([1, 0, 1, 0])
        self.assertEqual(ber(tx, rx), 1.0)


class TestEFDCSK(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_roundtrip_noiseless(self):
        bits = self.rng.integers(0, 2, 100)
        sig = efdcsk_transmit(bits, beta=64)
        rx_bits = efdcsk_receive(sig, beta=64)
        self.assertEqual(float(ber(bits, rx_bits)), 0.0)

    def test_output_length(self):
        bits = self.rng.integers(0, 2, 50)
        sig = efdcsk_transmit(bits, beta=32)
        self.assertEqual(len(sig), 50 * 32)

    def test_half_length_vs_classical(self):
        """EF-DCSK uses β samples per bit, classical DCSK uses 2β."""
        bits = self.rng.integers(0, 2, 20)
        sig_cl = dcsk_transmit(bits, beta=32)
        sig_ef = efdcsk_transmit(bits, beta=32)
        self.assertEqual(len(sig_ef), len(sig_cl) // 2)

    def test_roundtrip_awgn(self):
        bits = self.rng.integers(0, 2, 200)
        sig = efdcsk_transmit(bits, beta=64)
        rx = awgn(sig, 18.0, self.rng)
        rx_bits = efdcsk_receive(rx, beta=64)
        self.assertLess(ber(bits, rx_bits), 0.01)

    def test_efdcsk_vs_dcsk_noiseless(self):
        """Both schemes must decode perfectly in the noiseless case."""
        bits = self.rng.integers(0, 2, 80)
        cls = dcsk_receive(dcsk_transmit(bits, beta=64), beta=64)
        eff = efdcsk_receive(efdcsk_transmit(bits, beta=64), beta=64)
        self.assertEqual(float(ber(bits, cls)), 0.0)
        self.assertEqual(float(ber(bits, eff)), 0.0)


class TestHenonFIRDivergence(unittest.TestCase):
    def test_default_params_are_finite(self):
        """Standard Henon FIR sequence should produce finite output."""
        h = henon_fir_sequence(100, wc=0.5, n_taps=4)
        self.assertTrue(np.all(np.isfinite(h)))

    def test_non_default_params_run(self):
        """Non-default Henon params (a=1.2, b=0.2) should run without error."""
        h = henon_fir_sequence(100, wc=0.5, n_taps=4, a=1.2, b=0.2)
        self.assertEqual(h.shape, (100,))
        self.assertTrue(np.all(np.isfinite(h)))

    def test_divergent_params_raise_value_error(self):
        """henon_fir_sequence must raise ValueError when trajectory diverges."""
        with self.assertRaises(ValueError) as ctx:
            henon_fir_sequence(10_000, n_taps=3, wc=0.99, a=2.5)
        self.assertIn("diverged", str(ctx.exception))


class TestChannelKaiser(unittest.TestCase):
    def test_fir_channel_kaiser(self):
        s = np.sin(2 * np.pi * 0.05 * np.arange(100))
        r, _ = fir_channel(s, cutoff=0.3, num_taps=16, window=("kaiser", 5.0))
        self.assertEqual(r.shape, s.shape)
        self.assertTrue(np.all(np.isfinite(r)))


class TestChannelCustom(unittest.TestCase):
    def test_impulsive_custom_params(self):
        s = np.ones(100)
        r = channel_impulsive(s, snr_db=20.0, prob_impulso=0.05, amp_fator=3.0)
        self.assertEqual(r.shape, s.shape)

    def test_multipath_custom_params(self):
        s = np.ones(200)
        r = channel_multipath(s, snr_db=20.0, delays=[1, 3, 7], gains=[0.8, 0.4, 0.2])
        self.assertEqual(r.shape, s.shape)
        self.assertFalse(np.allclose(s, r))

    def test_interferers_custom_params(self):
        s = np.sin(2 * np.pi * 0.05 * np.arange(500))
        rng = np.random.default_rng(42)
        r = channel_interferers(s, snr_db=10.0, sir_dcsk_db=5.0, rng=rng)
        self.assertEqual(r.shape, s.shape)
        self.assertTrue(np.all(np.isfinite(r)))

    def test_urban_returns_finite(self):
        s = np.ones(1000)
        rng = np.random.default_rng(42)
        r = channel_urban(s, snr_db=20.0, rng=rng)
        self.assertEqual(r.shape, s.shape)
        self.assertTrue(np.all(np.isfinite(r)))

    def test_channel_interferers_reproducible(self):
        sig = dcsk_transmit(np.array([0, 1, 0, 1]), beta=64)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        rx1 = channel_interferers(sig, snr_db=20.0, rng=rng1)
        rx2 = channel_interferers(sig, snr_db=20.0, rng=rng2)
        np.testing.assert_array_equal(rx1, rx2)


if __name__ == "__main__":
    unittest.main()
