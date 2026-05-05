"""Structural contracts for the communication pipeline.

Each Protocol defines the interface that a concrete module
(:mod:`chaotic_pfc.comms.transmitter`, :mod:`~.channel`,
:mod:`~.receiver`) is expected to satisfy. They serve as
self-documenting contracts and enable mypy to verify that
implementations are structurally compatible.

Usage::

    from chaotic_pfc.comms.protocols import Transmitter, Channel, Receiver
    from chaotic_pfc.comms.transmitter import transmit

    tx: Transmitter = transmit  # mypy verifies structural compatibility
"""

from __future__ import annotations

from typing import Any, Protocol

from numpy.typing import NDArray


class Transmitter(Protocol):
    """Any callable that encodes a message into a chaotic carrier signal."""

    def __call__(self, message: NDArray, /, **kwargs: Any) -> NDArray: ...


class Channel(Protocol):
    """Any callable that propagates a signal through a transmission medium."""

    def __call__(self, sig: NDArray, /, **kwargs: Any) -> NDArray: ...


class Receiver(Protocol):
    """Any callable that recovers a message from a received chaotic signal."""

    def __call__(self, rx: NDArray, /, **kwargs: Any) -> NDArray: ...
