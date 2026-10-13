"""Shared helpers for every CLI experiment module."""

from __future__ import annotations

import argparse
import functools
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from chaotic_pfc.config import SpectralConfig


def _debug_enabled(args: argparse.Namespace | None) -> bool:
    """Return whether the full traceback should be shown on error."""
    return bool(getattr(args, "debug", False)) or os.environ.get("CHAOTIC_PFC_DEBUG") == "1"


def color_enabled(args: argparse.Namespace | None) -> bool:
    """Return whether terminal colour should be used.

    ``--color never`` and the ``NO_COLOR`` environment variable disable it;
    ``--color always`` forces it; ``auto`` (default) enables it only on a TTY.
    """
    mode = getattr(args, "color", "auto")
    if mode == "never" or os.environ.get("NO_COLOR"):
        return False
    if mode == "always":
        return True
    return sys.stderr.isatty()


def info(message: str, args: argparse.Namespace | None = None) -> None:
    """Print an informational message unless ``--quiet`` was passed."""
    if getattr(args, "quiet", False):
        return
    print(message)


def guard(run: Callable[[argparse.Namespace], int]) -> Callable[[argparse.Namespace], int]:
    """Report expected failures as a one-line message and exit code 1.

    Usage errors (unknown flags, bad choices) stay with :mod:`argparse`, which
    exits with code 2. Any other expected failure -- a missing input, an
    unreadable or corrupt checkpoint, an out-of-range value -- is printed to
    ``stderr`` as a single line and turned into exit code 1, with no traceback.
    With ``--debug`` (or ``CHAOTIC_PFC_DEBUG=1``) the exception is re-raised so
    the full traceback is shown.
    """

    @functools.wraps(run)
    def wrapper(args: argparse.Namespace) -> int:
        try:
            return run(args)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # expected I/O and parsing errors
            if _debug_enabled(args):
                raise
            message = " ".join(str(exc).split()) or type(exc).__name__
            prefix = "error:"
            if color_enabled(args):
                prefix = "\033[31merror:\033[0m"
            print(f"{prefix} {message}", file=sys.stderr)
            return 1

    return wrapper


def pick_backend(no_display: bool) -> bool:
    """Switch Matplotlib to a headless backend if needed.

    Parameters
    ----------
    no_display
        ``True`` when the user passed ``--no-display``, or when the
        caller otherwise knows no interactive display is available.

    Returns
    -------
    bool
        ``True`` if the backend was forced to ``Agg``. Callers use this
        to decide whether to call ``plt.show()`` at the end.

    Notes
    -----
    The check is a disjunction: either the user opted into headless
    mode explicitly, or we detected Linux without an X display. The
    selection must happen *before* the first ``pyplot`` import,
    otherwise Matplotlib locks to the interactive backend and won't let
    us switch.
    """
    headless = no_display or (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))
    if headless:
        import matplotlib

        matplotlib.use("Agg")
    from chaotic_pfc.plotting.figures import setup_rc

    setup_rc()
    return headless


def add_save_display_flags(parser: argparse.ArgumentParser) -> None:
    """Register ``--save`` and ``--no-display`` flags on *parser*."""
    parser.add_argument(
        "--save", action="store_true", help="Save the figures produced by the experiment."
    )
    parser.add_argument(
        "--no-display",
        dest="no_display",
        action="store_true",
        help="Run headless (implies --save).",
    )


def add_lang_flag(parser: argparse.ArgumentParser) -> None:
    """Register ``--lang`` flag for figure language (pt or en).

    The default honours the ``CHAOTIC_PFC_LANG`` environment variable and
    falls back to ``"pt"``.
    """
    parser.add_argument(
        "--lang",
        choices=["pt", "en"],
        default=os.environ.get("CHAOTIC_PFC_LANG", "pt"),
        help="Language for figure titles and labels (default: pt or $CHAOTIC_PFC_LANG)",
    )


def compute_psds(
    m: NDArray[Any],
    s: NDArray[Any],
    r: NDArray[Any],
    m_hat: NDArray[Any],
    sp_cfg: SpectralConfig,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute normalised PSDs for message, carrier, received, and recovered signals.

    Returns (omega, psd_m, psd_s, psd_r, psd_mhat).
    """
    from chaotic_pfc.dynamics.spectral import psd_normalised

    w = sp_cfg.window
    kb = sp_cfg.kaiser_beta
    omega, psd_m = psd_normalised(m, sp_cfg.nfft, sp_cfg.window_length, window=w, kaiser_beta=kb)
    _, psd_s = psd_normalised(s, sp_cfg.nfft, sp_cfg.window_length, window=w, kaiser_beta=kb)
    _, psd_r = psd_normalised(r, sp_cfg.nfft, sp_cfg.window_length, window=w, kaiser_beta=kb)
    _, psd_mhat = psd_normalised(m_hat, sp_cfg.nfft, sp_cfg.window_length, window=w, kaiser_beta=kb)
    return omega, psd_m, psd_s, psd_r, psd_mhat


def report_comm_metrics(
    m: Any,
    m_hat: Any,
    *,
    transient: int,
    bit_period: int,
    delay: int,
    args: Any = None,
) -> tuple[Any, Any]:
    """Align, measure and print the communication error metrics.

    *delay* is the applied group delay ``(N_taps - 1) / 2`` (0 for an
    ideal channel). The cross-correlation estimate is used as a check: a
    disagreement is reported, never applied silently.
    """
    from chaotic_pfc.comms.metrics import comm_metrics, estimate_delay

    est = estimate_delay(m, m_hat, transient=transient)
    if est["lag"] != delay:
        # Warn rather than fail on purpose. The check exists to catch a
        # channel that is not linear-phase (so the theoretical group delay
        # is wrong). When there is no recovery the correlation is near zero
        # and the estimated lag fits noise, so aborting the pipeline would
        # be worse than reporting the disagreement and using the known
        # theoretical delay.
        info(f"    WARNING: cross-correlation lag {est['lag']} != applied delay {delay}", args)
    met = comm_metrics(m, m_hat, transient=transient, bit_period=bit_period, delay=delay)
    info(
        f"    delay={delay}  (corr lag={est['lag']}  corr={est['corr_at_lag']:+.4f}  "
        f"corr@0={est['corr_at_zero']:+.4f})",
        args,
    )
    info(
        f"    MSE={met['mse']:.4e}  sample_err={met['per_sample_error']:.4f}  "
        f"BER={met['ber']:.4e}  corr={met['correlation']:+.4f}",
        args,
    )
    return est, met


def save_or_show(fig: Any, headless: bool, save_path: str | None, args: Any) -> None:
    """Save figure if *save_path* is given, otherwise display it."""
    import matplotlib.pyplot as plt

    if headless:
        plt.close(fig)
        if args.save:
            info(f"    Saved -> {save_path}", args)
    else:
        plt.show()
