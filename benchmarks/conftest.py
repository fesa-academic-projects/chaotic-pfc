"""Pytest and pytest-benchmark configuration for the benchmarks suite."""

from __future__ import annotations

import numpy as np


def pytest_configure(config: object) -> None:
    """Fix seeds so every benchmark run is reproducible."""
    np.random.seed(42)
