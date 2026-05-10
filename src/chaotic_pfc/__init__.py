"""
chaotic_pfc
===========
Chaotic communication system based on the Henon map.
"""

from chaotic_pfc._version import __version__
from chaotic_pfc.analysis.stats import (
    best_chaos_preserving,
    beta_curve,
    beta_summary,
    bootstrap_confidence,
    chaos_margin,
    compare_filter_types,
    correlation_matrix,
    export_summary_json,
    lmax_distribution,
    optimal_parameters,
    summary_table,
    transition_boundary,
)
from chaotic_pfc.analysis.sweep import (
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
from chaotic_pfc.analysis.sweep_plotting import (
    classify,
    plot_all,
    plot_classification_interleaved,
    plot_difficulty_map,
    plot_heatmap_continuous,
)
from chaotic_pfc.comms.channel import fir_channel, ideal_channel
from chaotic_pfc.comms.dcsk import (
    awgn,
    ber,
    channel_impulsive,
    channel_interferers,
    channel_multipath,
    channel_urban,
    dcsk_receive,
    dcsk_transmit,
    efdcsk_receive,
    efdcsk_transmit,
    henon_fir_sequence,
)
from chaotic_pfc.comms.protocols import Channel, Receiver, Transmitter
from chaotic_pfc.comms.receiver import receive, receive_order_n
from chaotic_pfc.comms.transmitter import transmit, transmit_order_n
from chaotic_pfc.dynamics.lyapunov import (
    EnsembleResult,
    LyapunovResult,
    fixed_point_stability,
    lyapunov_henon2d,
    lyapunov_henon2d_ensemble,
    lyapunov_max,
    lyapunov_max_ensemble,
)
from chaotic_pfc.dynamics.maps import (
    henon_filtered,
    henon_generalised,
    henon_order_n,
    henon_standard,
)
from chaotic_pfc.dynamics.signals import binary_message, sinusoidal_message
from chaotic_pfc.dynamics.spectral import psd_normalised

__all__ = [
    "FILTER_TYPES",
    "WINDOWS",
    "WINDOW_DISPLAY_NAMES",
    "Channel",
    "EnsembleResult",
    "LyapunovResult",
    "Receiver",
    "SweepResult",
    "Transmitter",
    "__version__",
    "awgn",
    "ber",
    "best_chaos_preserving",
    "beta_curve",
    "beta_summary",
    "binary_message",
    "bootstrap_confidence",
    "channel_impulsive",
    "channel_interferers",
    "channel_multipath",
    "channel_urban",
    "chaos_margin",
    "classify",
    "compare_filter_types",
    "correlation_matrix",
    "dcsk_receive",
    "dcsk_transmit",
    "efdcsk_receive",
    "efdcsk_transmit",
    "export_summary_json",
    "fir_channel",
    "fixed_point_stability",
    "henon_filtered",
    "henon_fir_sequence",
    "henon_generalised",
    "henon_order_n",
    "henon_standard",
    "ideal_channel",
    "lmax_distribution",
    "load_sweep",
    "lyapunov_henon2d",
    "lyapunov_henon2d_ensemble",
    "lyapunov_max",
    "lyapunov_max_ensemble",
    "optimal_parameters",
    "plot_all",
    "plot_classification_interleaved",
    "plot_difficulty_map",
    "plot_heatmap_continuous",
    "precompute_fir_bank",
    "psd_normalised",
    "quick_sweep_params",
    "receive",
    "receive_order_n",
    "run_sweep",
    "save_sweep",
    "sinusoidal_message",
    "summary_table",
    "transition_boundary",
    "transmit",
    "transmit_order_n",
]
