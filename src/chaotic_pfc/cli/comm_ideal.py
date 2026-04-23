"""Full communication pipeline over an ideal (noiseless) channel.

Originally ``scripts/03_comm_ideal_channel.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import pick_backend


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run comm-ideal`` subcommand."""
    from chaotic_pfc.config import DEFAULT_CONFIG as d

    p = subparsers.add_parser(
        "comm-ideal",
        help="Transmitter/receiver over an ideal (noiseless) channel.",
        description="Full communication pipeline over an ideal (noiseless) channel.",
    )
    p.add_argument("--N", type=int, default=d.comm.N)
    p.add_argument("--mu", type=float, default=d.comm.mu)
    p.add_argument("--period", type=int, default=d.comm.message_period)
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-display", dest="no_display", action="store_true")
    p.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute the ``comm-ideal`` experiment."""
    headless = pick_backend(args.no_display)

    import matplotlib.pyplot as plt
    import numpy as np

    from chaotic_pfc.channel import ideal_channel
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg
    from chaotic_pfc.plotting import plot_comm_grid
    from chaotic_pfc.receiver import receive
    from chaotic_pfc.signals import binary_message
    from chaotic_pfc.spectral import psd_normalised
    from chaotic_pfc.transmitter import transmit

    a, b = cfg.comm.henon.a, cfg.comm.henon.b
    tr = cfg.comm.transient
    sp_cfg = cfg.spectral
    fmt = cfg.plot.fmt
    fdir = Path(cfg.plot.figures_dir)
    if args.save:
        fdir.mkdir(parents=True, exist_ok=True)

    print(f"[03] Ideal channel  |  N={args.N:,}  μ={args.mu}  T={args.period}")

    m = binary_message(args.N, period=args.period)
    s = transmit(m, mu=args.mu, a=a, b=b, x0=0.0, y0=0.0)
    r = ideal_channel(s)

    rng = np.random.default_rng(cfg.seed)
    m_hat = receive(r, mu=args.mu, a=a, b=b, y0=rng.random(), z0=rng.random())

    mse = np.mean((m[tr:] - m_hat[tr:]) ** 2)
    print(f"    MSE (n > {tr}): {mse:.4e}")

    n = np.arange(args.N)
    omega, psd_m = psd_normalised(m, sp_cfg.nfft, sp_cfg.window_length)
    _, psd_s = psd_normalised(s, sp_cfg.nfft, sp_cfg.window_length)
    _, psd_r = psd_normalised(r, sp_cfg.nfft, sp_cfg.window_length)
    _, psd_mhat = psd_normalised(m_hat, sp_cfg.nfft, sp_cfg.window_length)

    win = slice(cfg.plot.time_window_start, cfg.plot.time_window_end)
    save_path = str(fdir / f"comm_ideal.{fmt}") if args.save else None

    fig = plot_comm_grid(
        n,
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
        suptitle=(r"Comunicação Caótica — Canal Ideal  ($\mu=" + str(args.mu) + r"$)"),
        save_path=save_path,
    )

    if headless:
        plt.close(fig)
        if args.save:
            print(f"    Saved -> {save_path}")
    else:
        plt.show()
    return 0
