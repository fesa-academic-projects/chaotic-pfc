"""Exception and warning hierarchy for chaotic-pfc."""

from __future__ import annotations


class ChaoticPFCError(Exception):
    """Base exception for all chaotic-pfc domain errors."""


class InvalidParameterError(ChaoticPFCError, ValueError):
    """Raised when a public API parameter violates its domain contract."""


class NonAsciiParameterError(InvalidParameterError, UnicodeEncodeError):
    """Raised when an ASCII-only text parameter contains non-ASCII characters."""


class UntrustedDataError(ChaoticPFCError, ValueError):
    """Raised when a checkpoint requires pickle deserialisation and is not trusted."""


class DivergenceError(ChaoticPFCError, ValueError):
    """Raised when a trajectory or sweep diverges and ``on_divergence='raise'``."""


class DivergenceWarning(UserWarning):
    """Emitted when a trajectory or sweep diverges and ``on_divergence='warn'``."""


__all__ = [
    "ChaoticPFCError",
    "DivergenceError",
    "DivergenceWarning",
    "InvalidParameterError",
    "NonAsciiParameterError",
    "UntrustedDataError",
]
