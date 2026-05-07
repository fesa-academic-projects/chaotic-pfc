"""Chaotic communication through FIR low-pass channel."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import add_lang_flag, add_save_display_flags, compute_psds, pick_backend, save_or_show


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run comm-fir`` subcommand."""
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg

    p = subparsers.add_parser(
        "comm-fir",
        help="Transmitter/receiver over a band-limited FIR channel.",
        description="Chaotic communication through an FIR low-pass channel.",
    )
    p.add_argument("--N", type=int, default=cfg.comm.N)
    p.add_argument("--mu", type=float, default=cfg.comm.mu)
    p.add_argument("--period", type=int, default=cfg.comm.message_period)
    p.add_argument("--cutoff", type=float, default=cfg.channel.cutoff)
    p.add_argument("--taps", type=int, default=cfg.channel.num_taps)
    add_save_display_flags(p)
    add_lang_flag(p)
    p.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute the ``comm-fir`` experiment."""
    headless = pick_backend(args.no_display)

    import numpy as np

    from chaotic_pfc._i18n import t
    from chaotic_pfc.comms.channel import fir_channel
    from chaotic_pfc.comms.receiver import receive
    from chaotic_pfc.comms.transmitter import transmit
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg
    from chaotic_pfc.dynamics.signals import binary_message
    from chaotic_pfc.plotting.figures import plot_comm_grid

    a, b = cfg.comm.henon.a, cfg.comm.henon.b
    tr = cfg.comm.transient
    fmt = cfg.plot.fmt
    fdir = Path(cfg.plot.figures_dir)
    if args.save:
        fdir.mkdir(parents=True, exist_ok=True)

    m = binary_message(args.N, period=args.period)
    s = transmit(m, mu=args.mu, a=a, b=b, x0=0.0, y0=0.0)
    r, h = fir_channel(s, cutoff=args.cutoff, num_taps=args.taps)

    rng = np.random.default_rng(cfg.seed)
    m_hat = receive(r, mu=args.mu, a=a, b=b, y0=rng.random(), z0=rng.random())

    mse = np.mean((m[tr:] - m_hat[tr:]) ** 2)
    print(f"[04] FIR channel  |  N={args.N:,}  mu={args.mu}  wc/pi={args.cutoff}  taps={args.taps}")
    print(f"    MSE (n > {tr}): {mse:.4e}")

    omega, psd_m, psd_s, psd_r, psd_mhat = compute_psds(m, s, r, m_hat, cfg.spectral)

    win = slice(cfg.plot.time_window_start, cfg.plot.time_window_end)
    save_path = str(fdir / f"comm_fir_channel.{fmt}") if args.save else None

    mhat_max = np.max(np.abs(m_hat[tr:]))
    y_lim_mhat = (
        (-min(mhat_max * 1.1, 300), min(mhat_max * 1.1, 300)) if mhat_max > 5 else (-1.5, 1.5)
    )

    fig = plot_comm_grid(
        np.arange(args.N),
        m,
        s,
        r,
        m_hat,
        omega,
        psd_m,
        psd_s,
        psd_r,
        psd_mhat,
        time_window=win,
        suptitle=(
            t("comm.fir", lang=args.lang)
            + r"  ($\omega_c/\pi="
            + f"{args.cutoff}"
            + r",\; N_{{taps}}="
            + f"{args.taps}"
            + r"$)"
        ),
        y_lim_mhat=y_lim_mhat,
        h_channel=h,
        save_path=save_path,
    )

    save_or_show(fig, headless, save_path, args)
    return 0
