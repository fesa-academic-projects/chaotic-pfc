"""
config.py
=========
Centralised configuration for all experiments.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class HenonConfig:
    a: float = 1.4
    b: float = 0.3


@dataclass
class CommConfig:
    N: int = 1_000_000
    mu: float = 0.01
    message_period: int = 20
    transient: int = 200
    henon: HenonConfig = field(default_factory=HenonConfig)


@dataclass
class ChannelConfig:
    cutoff: float = 0.99
    num_taps: int = 201


@dataclass
class InternalFIRConfig:
    cutoff: float = 0.5
    num_taps: int = 9

    def fir_coeffs(self) -> np.ndarray:
        from scipy.signal import firwin
        return firwin(
            numtaps=self.num_taps, cutoff=self.cutoff,
            window="hamming", pass_zero=True, fs=2.0,
        )


@dataclass
class SpectralConfig:
    nfft: int = 4096
    window_length: int = 1024
    fs: float = 1.0


@dataclass
class LyapunovConfig:
    """Parameters for Lyapunov exponent computation."""
    Nitera: int = 2000
    Ndiscard: int = 1000
    perturbation: float = 0.1
    # Pole-filter params
    Gz: float = 1.0
    pole_radius: float = 0.975
    w0: float = 0.0  # normalised angular frequency (×π)


@dataclass
class PlotConfig:
    time_window_start: int = 0
    time_window_end: int = 300
    dpi: int = 150
    figures_dir: str = "figures"
    fmt: str = "svg"  # output format: "svg" or "png"


@dataclass
class ExperimentConfig:
    comm: CommConfig = field(default_factory=CommConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    internal_fir: InternalFIRConfig = field(default_factory=InternalFIRConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    lyapunov: LyapunovConfig = field(default_factory=LyapunovConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    seed: int = 42


DEFAULT_CONFIG = ExperimentConfig()
