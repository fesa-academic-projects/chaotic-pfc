"""tests/test_protocols.py — Unit tests for communication protocol classes."""

import unittest

import numpy as np

from chaotic_pfc.comms.channel import ideal_channel
from chaotic_pfc.comms.protocols import Channel, Receiver, Transmitter
from chaotic_pfc.comms.receiver import receive
from chaotic_pfc.comms.transmitter import transmit
from chaotic_pfc.dynamics.signals import binary_message


class TestProtocols(unittest.TestCase):
    def test_transmitter_protocol_shape(self):
        """transmit satisfies Transmitter: (NDArray) -> NDArray."""
        tx: Transmitter = transmit
        m = binary_message(50, period=10)
        s = tx(m)
        self.assertEqual(s.shape, (50,))

    def test_channel_protocol_roundtrip(self):
        """ideal_channel satisfies Channel: (NDArray) -> NDArray."""
        ch: Channel = ideal_channel
        s = np.linspace(-1, 1, 100)
        r = ch(s)
        np.testing.assert_array_equal(r, s)

    def test_receiver_protocol_recovers_message(self):
        """receive satisfies Receiver: driven by ideal channel, recovers
        the original message after a short transient."""
        mu = 0.01
        m = binary_message(1000, period=20)
        s = transmit(m, mu=mu)
        r = ideal_channel(s)
        rx: Receiver = receive
        m_hat = rx(r, mu=mu, y0=0.0, z0=0.0)
        self.assertEqual(m_hat.shape, (1000,))
        transient = 500
        mse = float(np.mean((m[transient:] - m_hat[transient:]) ** 2))
        self.assertLess(mse, 1e-3)


if __name__ == "__main__":
    unittest.main()
