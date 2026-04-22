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

from .maps        import henon_standard, henon_generalised, henon_filtered, henon_order_n
from .signals     import binary_message
from .transmitter import transmit, transmit_order_n
from .channel     import ideal_channel, fir_channel
from .receiver    import receive, receive_order_n
from .spectral    import psd_normalised
from .lyapunov    import lyapunov_max, fixed_point_stability, lyapunov_henon2d
from .sweep       import (
    SweepResult,
    run_sweep, save_sweep, load_sweep, precompute_fir_bank,
    WINDOWS, FILTER_TYPES, WINDOW_DISPLAY_NAMES,
)

__all__ = [
    # Maps
    "henon_standard", "henon_generalised", "henon_filtered", "henon_order_n",
    # Communication
    "binary_message",
    "transmit", "transmit_order_n",
    "ideal_channel", "fir_channel",
    "receive", "receive_order_n",
    "psd_normalised",
    # Lyapunov (single config)
    "lyapunov_max", "fixed_point_stability", "lyapunov_henon2d",
    # Sweep
    "SweepResult",
    "run_sweep", "save_sweep", "load_sweep", "precompute_fir_bank",
    "WINDOWS", "FILTER_TYPES", "WINDOW_DISPLAY_NAMES",
]
