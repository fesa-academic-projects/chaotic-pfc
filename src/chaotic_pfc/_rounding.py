"""Exact half-up rounding shared by the manuscript numbers and the audit."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction


def round_fraction(value: Fraction | int, places: int) -> str:
    """Round a rational value to *places* decimals with ties away from zero.

    The calculation is performed on the integer numerator and denominator, so
    an exact tie such as ``626 / 4000 = 15.65`` rounds to ``15.7`` without
    passing through a binary float.
    """
    if places < 0:
        raise ValueError("places must be non-negative")
    fraction = value if isinstance(value, Fraction) else Fraction(value)
    sign = -1 if fraction < 0 else 1
    scale = 10**places
    numerator = abs(fraction.numerator) * scale
    denominator = fraction.denominator
    quotient, remainder = divmod(numerator, denominator)
    if 2 * remainder >= denominator:
        quotient += 1
    digits = str(quotient)
    if places:
        digits = digits.rjust(places + 1, "0")
        digits = f"{digits[:-places]}.{digits[-places:]}"
    return ("-" if sign < 0 else "") + digits


def pct(numerator: int, denominator: int, places: int = 1) -> str:
    """Percentage from integer counts, with exact half-up rounding."""
    if denominator == 0:
        return "n/a"
    return round_fraction(Fraction(100 * int(numerator), int(denominator)), places)


def round_decimal(value: Decimal | str | int | float, places: int) -> str:
    """Round a decimal value with half-up, without binary-float ties."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    text = repr(value) if isinstance(value, float) else str(value)
    if text.lower() in {"nan", "inf", "-inf", "infinity", "-infinity"}:
        return text
    return round_fraction(Fraction(text), places)
