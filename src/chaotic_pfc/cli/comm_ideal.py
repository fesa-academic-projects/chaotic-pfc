"""Full communication pipeline over an ideal (noiseless) channel."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import add_save_display_flags, compute_psds, pick_backend, save_or_show


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run comm-ideal`` subcommand."""
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg

    p = subparsers.add_parser(
        "comm-ideal",
        help="Transmitter/receiver over an ideal (noiseless) channel.",
        description="Full communication pipeline over an ideal (noiseless) channel.",
    )
    p.add_argument("--N", type=int, default=cfg.comm.N)
    p.add_argument("--mu", type=float, default=cfg.comm.mu)
    p.add_argument("--period", type=int, default=cfg.comm.message_period)
    add_save_display_flags(p)
    p.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute the ``comm-ideal`` experiment."""
    headless = pick_backend(args.no_display)

    import numpy as np

    from chaotic_pfc.comms.channel import ideal_channel
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
    r = ideal_channel(s)

    rng = np.random.default_rng(cfg.seed)
    m_hat = receive(r, mu=args.mu, a=a, b=b, y0=rng.random(), z0=rng.random())

    mse = np.mean((m[tr:] - m_hat[tr:]) ** 2)
    print(f"[03] Ideal channel  |  N={args.N:,}  mu={args.mu}  T={args.period}")
    print(f"    MSE (n > {tr}): {mse:.4e}")

    omega, psd_m, psd_s, psd_r, psd_mhat = compute_psds(m, s, r, m_hat, cfg.spectral)

    win = slice(cfg.plot.time_window_start, cfg.plot.time_window_end)
    save_path = str(fdir / f"comm_ideal.{fmt}") if args.save else None

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
        suptitle=(r"Comunicacao Caotica — Canal Ideal  ($\mu=" + str(args.mu) + r"$)"),
        save_path=save_path,
    )

    save_or_show(fig, headless, save_path, args)
    return 0
