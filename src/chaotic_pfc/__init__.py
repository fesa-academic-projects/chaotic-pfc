"""
chaotic_pfc
===========
Chaotic communication system based on the Hénon map.

Modules
-------
- maps            : Hénon map variants (standard, generalised, filtered, order-N)
- transmitter     : chaos-based modulator
- channel         : ideal and FIR channel models
- receiver        : chaos-synchronisation demodulator
- spectral        : PSD estimation utilities (Welch)
- signals         : message signal generators
- lyapunov        : Lyapunov exponent computation for single configurations
- sweep           : (order × cutoff) Lyapunov parameter sweep (Numba-JIT)
- plotting        : publication-quality SVG figures with LaTeX labels
- sweep_plotting  : classification-map figures for sweep results
- config          : centralised configuration
"""

from .channel import fir_channel, ideal_channel
from .lyapunov import (
    EnsembleResult,
    fixed_point_stability,
    lyapunov_henon2d,
    lyapunov_henon2d_ensemble,
    lyapunov_max,
    lyapunov_max_ensemble,
)
from .maps import henon_filtered, henon_generalised, henon_order_n, henon_standard
from .receiver import receive, receive_order_n
from .signals import binary_message
from .spectral import psd_normalised
from .sweep import (
    FILTER_TYPES,
    WINDOW_DISPLAY_NAMES,
    WINDOWS,
    SweepResult,
    load_sweep,
    precompute_fir_bank,
    quick_sweep_params,
    run_sweep,
    save_sweep,
)
from .transmitter import transmit, transmit_order_n

__all__ = [
    "FILTER_TYPES",
    "WINDOWS",
    "WINDOW_DISPLAY_NAMES",
    "EnsembleResult",
    "SweepResult",
    "binary_message",
    "fir_channel",
    "fixed_point_stability",
    "henon_filtered",
    "henon_generalised",
    "henon_order_n",
    "henon_standard",
    "ideal_channel",
    "load_sweep",
    "lyapunov_henon2d",
    "lyapunov_henon2d_ensemble",
    "lyapunov_max",
    "lyapunov_max_ensemble",
    "precompute_fir_bank",
    "psd_normalised",
    "quick_sweep_params",
    "receive",
    "receive_order_n",
    "run_sweep",
    "save_sweep",
    "transmit",
    "transmit_order_n",
]
