"""Structural contracts for the communication pipeline.

Each Protocol defines the interface that a concrete module
(:mod:`chaotic_pfc.comms.transmitter`, :mod:`~.channel`,
:mod:`~.receiver`) is expected to satisfy. They serve as
self-documenting contracts that codify the pipeline's data flow:
transmitter → channel → receiver.

The ``**kwargs: Any`` signature is intentionally loose — each
concrete function has its own keyword parameters (``mu``, ``a``,
``b``, ``cutoff``, etc.) and the Protocols do not attempt to
enumerate them all. This means mypy does not verify argument-level
compatibility, but the Protocols still serve as readable contracts
for developers.

Usage::

    from chaotic_pfc.comms.protocols import Transmitter
    from chaotic_pfc.comms.transmitter import transmit

    tx: Transmitter = transmit  # documents intent; mypy checks return type
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
