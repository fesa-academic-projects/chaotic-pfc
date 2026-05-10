"""
config.py
=========
Centralised configuration for all experiments.

The module exposes a single :data:`DEFAULT_CONFIG` object of type
:class:`ExperimentConfig` that aggregates the per-subsystem settings
(``comm``, ``channel``, ``lyapunov``, ``sweep``, …). Every experiment
script imports it to obtain its default parameters, and CLI flags
selectively override individual fields without having to thread a full
config through the call chain.

All configs are plain dataclasses, so they can be cheaply copied,
mutated in tests, or serialised with :func:`dataclasses.asdict`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HenonConfig:
    """Parameters of the base Hénon map.

    The default ``(a, b) = (1.4, 0.3)`` is the canonical chaotic
    regime. Deviating from these requires care — the communication
    pipeline assumes a strange attractor.
    """

    a: float = 1.4
    b: float = 0.3


@dataclass
class CommConfig:
    """Top-level parameters of the communication pipeline.

    Attributes
    ----------
    N
        Default number of samples in a transmitted sequence.
    mu
        Modulation depth used by the transmitter and receiver. Must
        match on both ends.
    message_period
        Period of the default binary message, in samples.
    transient
        Number of samples discarded at the start of each run before
        computing performance metrics (MSE, BER). Lets the local
        oscillator lock into synchronisation.
    henon
        Nested Hénon parameters (α, β).
    """

    N: int = 1_000_000
    mu: float = 0.01
    message_period: int = 20
    transient: int = 200
    henon: HenonConfig = field(default_factory=HenonConfig)


@dataclass
class ChannelConfig:
    """External FIR-channel parameters (:func:`~chaotic_pfc.channel.fir_channel`)."""

    cutoff: float = 0.99
    num_taps: int = 201


@dataclass
class InternalFIRConfig:
    """FIR filter used inside the N-th order Hénon oscillator.

    Differs from :class:`ChannelConfig` in that this filter sits
    *inside* the chaos generator (it shapes the feedback loop), not on
    the transmission path. Used primarily by the order-N
    transmitter/receiver pair.
    """

    cutoff: float = 0.5
    num_taps: int = 9

    def fir_coeffs(self) -> np.ndarray:
        """Build the FIR coefficients for the configured cutoff and length.

        Uses a Hamming window with ``pass_zero=True`` and the usual
        ``fs=2.0`` normalisation so that ``cutoff`` is interpreted
        directly as ``ω_c / π``.

        Returns
        -------
        ndarray, shape (num_taps,)
            Filter coefficients.
        """
        from scipy.signal import firwin

        return firwin(
            numtaps=self.num_taps,
            cutoff=self.cutoff,
            window="hamming",
            pass_zero=True,
            fs=2.0,
        )


@dataclass
class SpectralConfig:
    """Defaults for :func:`~chaotic_pfc.spectral.psd_normalised`.

    Attributes
    ----------
    nfft, window_length, fs
        Standard Welch parameters.
    window
        FIR window applied to each Welch segment. Currently
        ``"hamming"`` (the historical default) and ``"kaiser"`` are
        supported.
    kaiser_beta
        Shape parameter of the Kaiser window. Ignored unless
        ``window == "kaiser"``. Larger values give higher stop-band
        attenuation at the cost of a wider main lobe; β ≈ 5 is a
        common ~50 dB compromise.
    """

    nfft: int = 4096
    window_length: int = 1024
    fs: float = 1.0
    window: str = "hamming"
    kaiser_beta: float = 5.0


@dataclass
class LyapunovConfig:
    """Parameters for Lyapunov exponent computation.

    Attributes
    ----------
    Nitera, Ndiscard
        Iterations used for the estimate and the transient to discard
        before starting to accumulate.
    perturbation
        Half-width of the IC sampling box around the fixed point,
        as a fraction of the coordinate (±10% by default).
    Gz
        Filter gain term used in the 4-D pole-filtered system.
    pole_radius
        Pole radius ``r ∈ (0, 1)``. Larger values make the filter
        sharper and the dynamics more dissipative.
    w0
        Normalised angular frequency of the pole pair (``× π``).
    """

    Nitera: int = 2000
    Ndiscard: int = 1000
    perturbation: float = 0.1
    n_ci: int = 20
    data_dir: str = "data/lyapunov"
    # Pole-filter params
    Gz: float = 1.0
    pole_radius: float = 0.975
    w0: float = 0.0


@dataclass
class PlotConfig:
    """Global plotting defaults used by the experiment scripts."""

    time_window_start: int = 0
    time_window_end: int = 300
    dpi: int = 150
    figures_dir: str = "figures"
    fmt: str = "svg"  # output format: "svg" or "png"


@dataclass
class SweepConfig:
    """Parameters for the 2-D (order, cutoff) Lyapunov sweep.

    Used by :mod:`chaotic_pfc.sweep`. The full grid is
    ``len(orders) × n_cutoffs`` points; at the defaults this is 4 000.

    Attributes
    ----------
    Nitera
        Burn-in iterations applied before computing the estimator.
    Nmap
        Iterations accumulated into the Lyapunov estimator.
    n_initial
        Number of random initial conditions averaged per grid point.
    order_lo, order_hi
        Filter-order range, ``order_hi`` exclusive. Defaults to
        ``range(2, 42)`` → 40 orders.
    n_cutoffs
        Number of cutoff frequencies sampled linearly in ``(0, 1)``.
    default_window, default_filter_type
        Default selections when no CLI override is given.
    data_dir, figures_dir
        Output locations. The ``.npz`` checkpoints live under
        ``data/sweeps`` and are versioned; the figures under
        ``figures/sweeps`` are derived from those checkpoints.
    fig_fmts
        Output formats produced by the plot script. Tuple because both
        PNG (quick preview, GitHub render) and SVG (paper-grade) are
        useful to have in parallel.
    """

    Nitera: int = 500
    Nmap: int = 3000
    n_initial: int = 25
    order_lo: int = 2
    order_hi: int = 42
    n_cutoffs: int = 100
    default_window: str = "hamming"
    default_filter_type: str = "lowpass"
    bandwidth: float = 0.2  # bandpass/bandstop width (×π)
    data_dir: str = "data/sweeps"
    figures_dir: str = "figures/sweeps"
    fig_fmts: tuple[str, ...] = ("png", "svg")


@dataclass
class ExperimentConfig:
    """Aggregate configuration used by every experiment script.

    Composing the individual subsystem configs into a single object
    keeps the CLI scripts simple: they import :data:`DEFAULT_CONFIG`,
    pick out the branches they need (e.g. ``cfg.comm``,
    ``cfg.lyapunov``) and only expose flags for the handful of fields
    that actually vary across experiments.
    """

    comm: CommConfig = field(default_factory=CommConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    internal_fir: InternalFIRConfig = field(default_factory=InternalFIRConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    lyapunov: LyapunovConfig = field(default_factory=LyapunovConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    seed: int = 42
    lang: str = "pt"

    def to_namespace(self) -> argparse.Namespace:
        """Build an ``argparse.Namespace`` with defaults for every CLI subcommand.

        Useful for ``run all``, which forwards a single namespace to all
        sub-experiments.
        """
        return argparse.Namespace(
            N=self.comm.N,
            mu=self.comm.mu,
            period=self.comm.message_period,
            cutoff=self.channel.cutoff,
            taps=self.channel.num_taps,
            steps=50,
            epsilon=1e-4,
            Nitera=self.lyapunov.Nitera,
            Ndiscard=self.lyapunov.Ndiscard,
            pole_radius=self.lyapunov.pole_radius,
            w0=self.lyapunov.w0,
            n_ci=self.lyapunov.n_ci,
            perturbation=self.lyapunov.perturbation,
            data_dir=self.lyapunov.data_dir,
            lang=self.lang,
        )


DEFAULT_CONFIG = ExperimentConfig()
"""The project-wide singleton. Import this, don't instantiate a new one."""
