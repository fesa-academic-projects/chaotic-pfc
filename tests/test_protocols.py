"""tests/test_protocols.py — Unit tests for communication protocol classes."""

import unittest

import numpy as np

from chaotic_pfc.comms.protocols import Channel, Receiver, Transmitter
from chaotic_pfc.comms.transmitter import transmit


class TestProtocols(unittest.TestCase):
    def test_transmit_matches_transmitter_protocol(self):
        tx: Transmitter = transmit
        self.assertTrue(callable(tx))

    def test_transmitter_callable_with_kwargs(self):
        m = np.array([1.0, -1.0, 1.0, -1.0])
        tx: Transmitter = transmit
        s = tx(m, mu=0.01, a=1.4, b=0.3)
        self.assertEqual(len(s), len(m))

    def test_channel_protocol_is_callable(self):
        from chaotic_pfc.comms.channel import ideal_channel

        ch: Channel = ideal_channel
        self.assertTrue(callable(ch))

    def test_receiver_protocol_is_callable(self):
        from chaotic_pfc.comms.receiver import receive

        rx: Receiver = receive
        self.assertTrue(callable(rx))
