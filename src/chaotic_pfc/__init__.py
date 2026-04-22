"""
chaotic_pfc
===========
Chaotic communication system based on the Hénon map.

Modules
-------
- maps        : Hénon map variants (standard, generalised, filtered, order-N)
- transmitter : chaos-based modulator
- channel     : ideal and FIR channel models
- receiver    : chaos-synchronisation demodulator
- spectral    : PSD estimation utilities (Welch)
- signals     : message signal generators
- lyapunov    : Lyapunov exponent computation
- plotting    : publication-quality SVG figures with LaTeX labels
- config      : centralised configuration
"""

from .maps import henon_standard, henon_generalised, henon_filtered, henon_order_n
from .signals import binary_message
from .transmitter import transmit, transmit_order_n
from .channel import ideal_channel, fir_channel
from .receiver import receive, receive_order_n
from .spectral import psd_normalised
from .lyapunov import lyapunov_max, fixed_point_stability, lyapunov_henon2d

__all__ = [
    "henon_standard", "henon_generalised", "henon_filtered", "henon_order_n",
    "binary_message",
    "transmit", "transmit_order_n",
    "ideal_channel", "fir_channel",
    "receive", "receive_order_n",
    "psd_normalised",
    "lyapunov_max", "fixed_point_stability", "lyapunov_henon2d",
]
