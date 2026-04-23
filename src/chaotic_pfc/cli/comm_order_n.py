"""Communication using N-th order Hénon with internal FIR filter.

Originally ``scripts/05_comm_order_n.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import pick_backend


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run comm-order-n`` subcommand."""
    from chaotic_pfc.config import DEFAULT_CONFIG as d

    p = subparsers.add_parser(
        "comm-order-n",
        help="Transmitter/receiver using an N-th order Hénon with internal FIR.",
        description="Communication using N-th order Hénon with internal FIR filter.",
    )
    p.add_argument("--N", type=int, default=d.comm.N)
    p.add_argument("--mu", type=float, default=d.comm.mu)
    p.add_argument("--period", type=int, default=d.comm.message_period)
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-display", dest="no_display", action="store_true")
    p.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute the ``comm-order-n`` experiment."""
    headless = pick_backend(args.no_display)

    import matplotlib.pyplot as plt
    import numpy as np

    from chaotic_pfc.channel import fir_channel
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg
    from chaotic_pfc.plotting import plot_comm_grid
    from chaotic_pfc.receiver import receive_order_n
    from chaotic_pfc.signals import binary_message
    from chaotic_pfc.spectral import psd_normalised
    from chaotic_pfc.transmitter import transmit_order_n

    a, b = cfg.comm.henon.a, cfg.comm.henon.b
    tr = cfg.comm.transient
    sp_cfg = cfg.spectral
    fmt = cfg.plot.fmt

    c = cfg.internal_fir.fir_coeffs()
    Nc = len(c)

    fdir = Path(cfg.plot.figures_dir)
    if args.save:
        fdir.mkdir(parents=True, exist_ok=True)

    print(f"[05] N-th order Hénon  |  N={args.N:,}  μ={args.mu}  Nc={Nc}")

    rng = np.random.default_rng(cfg.seed)
    x0 = 0.5 * rng.random(Nc)
    y0 = 0.5 * rng.random(Nc)

    m = binary_message(args.N, period=args.period)
    s, _ = transmit_order_n(m, c, mu=args.mu, a=a, b=b, x0=x0)
    r, h = fir_channel(s, cutoff=cfg.channel.cutoff, num_taps=cfg.channel.num_taps)
    m_hat, _ = receive_order_n(r, c, mu=args.mu, a=a, b=b, y0=y0)

    mse = np.mean((m[tr:] - m_hat[tr:]) ** 2)
    print(f"    MSE (n > {tr}): {mse:.4e}")

    n = np.arange(args.N)
    omega, psd_m = psd_normalised(m, sp_cfg.nfft, sp_cfg.window_length)
    _, psd_s = psd_normalised(s, sp_cfg.nfft, sp_cfg.window_length)
    _, psd_r = psd_normalised(r, sp_cfg.nfft, sp_cfg.window_length)
    _, psd_mhat = psd_normalised(m_hat, sp_cfg.nfft, sp_cfg.window_length)

    win = slice(cfg.plot.time_window_start, min(cfg.plot.time_window_end, 1000))
    save_path = str(fdir / f"comm_order_n.{fmt}") if args.save else None

    mhat_ss = m_hat[tr:]
    mhat_max = np.percentile(np.abs(mhat_ss), 99)
    if mhat_max > 5:
        y_lim_mhat = (-min(mhat_max * 1.1, 300), min(mhat_max * 1.1, 300))
    else:
        y_lim_mhat = (-1.5, 1.5)

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
        suptitle=(
            r"Hénon de Ordem $N$ — Canal FIR"
            r"  ($N_c=" + str(Nc) + r",\; \mu=" + str(args.mu) + r"$)"
        ),
        y_lim_mhat=y_lim_mhat,
        h_channel=h,
        save_path=save_path,
    )

    if headless:
        plt.close(fig)
        if args.save:
            print(f"    Saved -> {save_path}")
    else:
        plt.show()
    return 0
