"""
signals.py
==========
Message signal generators.
"""

import numpy as np
from numpy.typing import NDArray


def binary_message(N: int, period: int = 20) -> NDArray:
    if period <= 0 or period % 2 != 0:
        raise ValueError(f"period must be a positive even integer, got {period}")
    half = period // 2
    block = np.concatenate([np.ones(half), -np.ones(half)])
    num_blocks = int(np.ceil(N / period))
    return np.tile(block, num_blocks)[:N]


def sinusoidal_message(N: int, normalised_freq: float = 0.1) -> NDArray:
    n = np.arange(N)
    return np.sin(2.0 * np.pi * normalised_freq * n)
