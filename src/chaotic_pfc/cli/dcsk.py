"""DCSK communication experiment — BER vs SNR across chaotic modulation schemes."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import add_lang_flag, add_save_display_flags, pick_backend


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run dcsk`` subcommand."""
    from chaotic_pfc.comms.dcsk import DCSK_DEFAULT_WC

    p = subparsers.add_parser(
        "dcsk",
        help="DCSK / EF-DCSK / Pecora-Carroll — BER vs SNR comparison.",
        description="Compare chaotic communication schemes: classical DCSK, EF-DCSK, and Pecora-Carroll.",
    )
    p.add_argument("--N-bits", type=int, default=600, dest="N")
    p.add_argument(
        "--beta", type=int, default=64, help="Spreading factor for DCSK/EF-DCSK (default: 64)"
    )
    p.add_argument("--n-taps", type=int, default=5, dest="n_taps")
    p.add_argument("--wc", type=float, default=DCSK_DEFAULT_WC)
    p.add_argument(
        "--mu", type=float, default=0.01, help="Modulation index for Pecora-Carroll (default: 0.01)"
    )
    p.add_argument("--snr-min", type=float, default=-6)
    p.add_argument("--snr-max", type=float, default=28)
    p.add_argument("--snr-step", type=float, default=2)
    add_save_display_flags(p)
    add_lang_flag(p)
    p.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute the ``dcsk`` experiment."""
    headless = pick_backend(args.no_display)

    import matplotlib.pyplot as plt
    import numpy as np

    from chaotic_pfc._i18n import t
    from chaotic_pfc.comms.channel import ideal_channel
    from chaotic_pfc.comms.dcsk import (
        awgn,
        ber,
        dcsk_receive,
        dcsk_transmit,
        efdcsk_receive,
        efdcsk_transmit,
    )
    from chaotic_pfc.comms.receiver import receive
    from chaotic_pfc.comms.transmitter import transmit
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg
    from chaotic_pfc.dynamics.signals import binary_message

    rng = np.random.default_rng(42)
    snr_range = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)

    a, b_henon = cfg.comm.henon.a, cfg.comm.henon.b
    transient = cfg.comm.transient

    print(f"DCSK  |  β={args.beta}  taps={args.n_taps}  ωc={args.wc}  μ={args.mu}")
    print(f"SNR range: {snr_range[0]:+.0f}..{snr_range[-1]:+.0f} dB  step={args.snr_step}")

    # ── AWGN-only channel ─────────────────────────────────────────────────
    def awgn_chan(sig, snr):
        return awgn(sig, snr, rng)

    # NOTE: Pecora-Carroll uses a periodic binary_message(period=20) while
    # DCSK and EF-DCSK use random bits. The two are not directly comparable
    # because PC operates on continuous-time synchronisation, not on bit
    # decisions. The PC curve here serves as an indicative reference for
    # the noise floor of synchronisation, not a fair BER benchmark against
    # the discrete-modulation schemes.

    # ── Pecora-Carroll (original method) ──────────────────────────────────
    bits_pc = binary_message(args.N, period=20)
    bits_int_pc = np.where(bits_pc > 0, 0, 1).astype(np.int64)

    bers_pc, snrs_pc = [], []
    for snr in snr_range:
        s = transmit(bits_pc, mu=args.mu, a=a, b=b_henon, x0=0.0, y0=0.0)
        r = awgn(ideal_channel(s), snr, rng)
        m_hat = receive(r, mu=args.mu, a=a, b=b_henon, y0=rng.random(), z0=rng.random())
        rx_int = np.where(m_hat[transient:] > 0, 0, 1).astype(np.int64)
        ber_val = ber(bits_int_pc[transient:], rx_int)
        snrs_pc.append(snr)
        bers_pc.append(ber_val)
        if ber_val >= 0.50:
            break

    # ── DCSK (classical) ─────────────────────────────────────────────────
    bits_dcsk = rng.integers(0, 2, args.N)

    bers_dcsk, snrs_dcsk = [], []
    for snr in snr_range:
        sig = dcsk_transmit(bits_dcsk, beta=args.beta, n_taps=args.n_taps, wc=args.wc)
        rx = awgn_chan(sig, snr)
        ber_val = ber(bits_dcsk, dcsk_receive(rx, args.beta))
        snrs_dcsk.append(snr)
        bers_dcsk.append(ber_val)
        if ber_val >= 0.50:
            break

    # ── EF-DCSK (efficient) ──────────────────────────────────────────────
    bers_ef, snrs_ef = [], []
    for snr in snr_range:
        sig = efdcsk_transmit(bits_dcsk, beta=args.beta, n_taps=args.n_taps, wc=args.wc)
        rx = awgn_chan(sig, snr)
        ber_val = ber(bits_dcsk, efdcsk_receive(rx, args.beta))
        snrs_ef.append(snr)
        bers_ef.append(ber_val)
        if ber_val >= 0.50:
            break

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6.5))

    def _safe(bers):
        return np.where(np.array(bers) == 0, 1e-4, bers)

    ax.semilogy(
        snrs_pc,
        _safe(bers_pc),
        "s--",
        color="gray",
        lw=1.6,
        label=t("dcsk.pecora_carroll", lang=args.lang),
        ms=6,
    )
    ax.semilogy(
        snrs_dcsk,
        _safe(bers_dcsk),
        "o-",
        color="steelblue",
        lw=1.8,
        label=t("dcsk.classic", lang=args.lang),
        ms=6,
    )
    ax.semilogy(
        snrs_ef,
        _safe(bers_ef),
        "D-",
        color="darkorange",
        lw=1.8,
        label=t("dcsk.efficient", lang=args.lang),
        ms=6,
    )

    ax.axhline(0.01, color="gray", ls=":", lw=1, label=t("dcsk.ber_1pct", lang=args.lang))
    ax.axhline(0.50, color="red", ls="--", lw=1.2, label=t("dcsk.ber_50pct", lang=args.lang))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(
        t("dcsk.comparison_title", lang=args.lang) + "\n"
        f"Hénon FIR AWGN  (β={args.beta}, {args.n_taps} taps, ωc={args.wc})"
    )
    ax.set_ylim(5e-4, 0.7)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()

    # Asymmetry note: PC uses a periodic message; DCSK/EF-DCSK use random
    # bits. See module-level comment above the Pecora-Carroll section.
    note = (
        "Nota: PC usa mensagem periodica; DCSK/EF-DCSK usam bits aleatorios."
        if args.lang == "pt"
        else "Note: PC uses a periodic message; DCSK/EF-DCSK use random bits."
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=8, color="gray", style="italic")

    if args.save:
        fdir = Path("figures")
        fdir.mkdir(parents=True, exist_ok=True)
        path = fdir / "dcsk_comparison.svg"
        fig.savefig(path)
        print(f"    Saved -> {path}")

    if headless:
        plt.close(fig)
    else:
        plt.show()
    return 0
