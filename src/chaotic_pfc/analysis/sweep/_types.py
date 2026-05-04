"""Sweep data types, constants, and catalogue of supported configurations."""

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray

# ───────────────────────────────────────────────────────────────────────────
# Catalogue of supported FIR windows and filter types
# ───────────────────────────────────────────────────────────────────────────

WINDOWS: tuple[str, ...] = (
    "hamming",
    "hann",
    "blackman",
    "kaiser",
    "blackmanharris",
    "boxcar",
    "bartlett",
)

FILTER_TYPES: tuple[str, ...] = ("lowpass", "highpass", "bandpass", "bandstop")

WINDOW_DISPLAY_NAMES: dict[str, str] = {
    "hamming": "Hamming",
    "hann": "Hann",
    "blackman": "Blackman",
    "kaiser": "Kaiser",
    "blackmanharris": "Blackman-Harris",
    "boxcar": "Rectangular",
    "bartlett": "Bartlett",
}

# Kaiser window needs a beta parameter (attenuation trade-off).
# beta = 5.0 gives ~50 dB stop-band attenuation, a common default.
_KAISER_BETA: float = 5.0

# Adaptive Lyapunov: how often to evaluate the convergence criterion.
# Each "checkpoint" computes lambda_max from the running lyap_sum, compares it
# to the previous checkpoint, and stops if the difference is below ``tol``
# for ``_ADAPTIVE_STREAK`` consecutive checkpoints. 100 is a sweet spot:
# small enough that early-stop fires close to the true convergence point,
# large enough that the eval overhead is negligible (~Ns ops per check vs
# Ns^3 in the inner MGS loop).
_ADAPTIVE_CHECKPOINT_EVERY: int = 100
_ADAPTIVE_STREAK: int = 2


@dataclass
class SweepResult:
    """Result of a single (window, filter) sweep.

    Attributes
    ----------
    h : ndarray, shape (Ncoef, Ncut)
        Mean of lambda_max across ``n_initial`` random ICs, per grid point.
        NaN where the trajectory diverged for all ICs.
    h_std : ndarray, shape (Ncoef, Ncut)
        Standard deviation of lambda_max samples at each grid point.
    orders : ndarray, shape (Ncoef,)
        Filter orders N_s actually swept.
    cutoffs : ndarray, shape (Ncut,)
        Normalised cutoff frequencies omega_c in (0, 1) swept.
    window : str
        FIR window used (lower-case, e.g. ``"hamming"``).
    filter_type : {"lowpass", "highpass", "bandpass", "bandstop"}
        Filter pass-zero configuration.
    n_iters_used : ndarray, shape (Ncoef, Ncut), optional
        Average number of Lyapunov iterations actually used per grid
        point (across non-divergent ICs). In non-adaptive sweeps every
        finite cell equals ``Nmap``; in adaptive sweeps it ranges from
        ``Nmap_min`` to ``Nmap``, providing a "difficulty map" of the
        parameter space. ``None`` when loaded from a legacy ``.npz``
        without this field.
    metadata : dict
        Free-form metadata (simulation parameters, timing, etc.).
    """

    h: NDArray
    h_std: NDArray
    orders: NDArray
    cutoffs: NDArray
    window: str
    filter_type: str
    n_iters_used: NDArray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Human-readable name used for output directories."""
        pretty = WINDOW_DISPLAY_NAMES.get(self.window, self.window.capitalize())
        return f"{pretty} ({self.filter_type})"
