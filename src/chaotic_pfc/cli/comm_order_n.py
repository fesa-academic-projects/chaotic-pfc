"""Communication using N-th order Henon with internal FIR filter."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import add_lang_flag, add_save_display_flags, compute_psds, pick_backend, save_or_show


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run comm-order-n`` subcommand."""
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg

    p = subparsers.add_parser(
        "comm-order-n",
        help="Transmitter/receiver using an N-th order Henon with internal FIR.",
        description="Communication using N-th order Henon with internal FIR filter.",
    )
    p.add_argument("--N", type=int, default=cfg.comm.N)
    p.add_argument("--mu", type=float, default=cfg.comm.mu)
    p.add_argument("--period", type=int, default=cfg.comm.message_period)
    add_save_display_flags(p)
    add_lang_flag(p)
    p.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute the ``comm-order-n`` experiment."""
    headless = pick_backend(args.no_display)

    import numpy as np

    from chaotic_pfc._i18n import t
    from chaotic_pfc.comms.channel import fir_channel
    from chaotic_pfc.comms.receiver import receive_order_n
    from chaotic_pfc.comms.transmitter import transmit_order_n
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg
    from chaotic_pfc.dynamics.signals import binary_message
    from chaotic_pfc.plotting.figures import plot_comm_grid

    a, b = cfg.comm.henon.a, cfg.comm.henon.b
    fmt = cfg.plot.fmt

    c = cfg.internal_fir.fir_coeffs()
    Nc = len(c)

    fdir = Path(cfg.plot.figures_dir)
    if args.save:
        fdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    x0 = 0.5 * rng.random(Nc)
    y0 = 0.5 * rng.random(Nc)

    m = binary_message(args.N, period=args.period)
    s, _ = transmit_order_n(m, c, mu=args.mu, a=a, b=b, x0=x0)
    r, h = fir_channel(s, cutoff=cfg.channel.cutoff, num_taps=cfg.channel.num_taps)
    m_hat, _ = receive_order_n(r, c, mu=args.mu, a=a, b=b, y0=y0)

    mse = np.mean((m[cfg.comm.transient :] - m_hat[cfg.comm.transient :]) ** 2)
    print(f"[05] N-th order Henon  |  N={args.N:,}  mu={args.mu}  Nc={Nc}")
    print(f"    MSE (n > {cfg.comm.transient}): {mse:.4e}")

    omega, psd_m, psd_s, psd_r, psd_mhat = compute_psds(m, s, r, m_hat, cfg.spectral)

    win = slice(cfg.plot.time_window_start, min(cfg.plot.time_window_end, 1000))
    save_path = str(fdir / f"comm_order_n.{fmt}") if args.save else None

    mhat_ss = m_hat[cfg.comm.transient :]
    mhat_max = np.percentile(np.abs(mhat_ss), 99)
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
            t("comm.order_n", lang=args.lang)
            + r"  ($N_c="
            + str(Nc)
            + r",\; \mu="
            + str(args.mu)
            + r"$)"
        ),
        y_lim_mhat=y_lim_mhat,
        h_channel=h,
        save_path=save_path,
    )

    save_or_show(fig, headless, save_path, args)
    return 0
