"""Rounding of the numbers printed by the manuscript figure generators."""

from __future__ import annotations

from chaotic_pfc.paper.figures import _fixed, _sci


def test_fixed_rounds_half_up() -> None:
    """A tie in the last digit rounds away from zero, as in ``paper.numbers``."""
    assert _fixed(0.07875, 4) == "0.0788"
    assert _fixed(15.65, 1) == "15.7"


def test_scientific_rounds_mantissa_half_up() -> None:
    """Scientific output keeps the half-up rule on the mantissa."""
    assert _sci(1.25e-5, 1) == "1.3e-5"
    assert _sci(831.1687, 4) == "8.3117e+2"
