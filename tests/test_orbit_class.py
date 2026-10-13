"""The ``OrbitClass`` enum and ``classify``."""

from __future__ import annotations

import numpy as np

from chaotic_pfc import OrbitClass, classify


def test_orbit_class_values_are_stable() -> None:
    # The integers are the ones the discrete colour map expects; changing them
    # would change the classification figures.
    assert (OrbitClass.PERIODIC, OrbitClass.CHAOTIC, OrbitClass.DIVERGENT) == (-1, 0, 2)


def test_classify_returns_orbit_class_values() -> None:
    got = classify(np.array([np.nan, -1.0, 0.0, 1.0]))
    assert got.tolist() == [
        float(OrbitClass.DIVERGENT),
        float(OrbitClass.PERIODIC),
        float(OrbitClass.PERIODIC),
        float(OrbitClass.CHAOTIC),
    ]
