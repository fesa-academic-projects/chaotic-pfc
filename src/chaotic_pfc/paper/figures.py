"""Generate the JCIS figures that the repository did not produce before.

Writes ``fig_henon_signals``, ``fig_comm_bandlimited``, ``fig_kaiser_beta``
and ``fig_chaotic_density_norm`` as SVG files in ``--out-dir`` (default
``articles/jcis/Figures``). Every number is computed from the ``chaotic_pfc`` package
and the versioned sweeps in ``data/sweeps``; the numbers quoted in the paper
are printed so that they can be checked against the text.

Output is deterministic: the SVG hash salt is fixed and the ``Date`` metadata
is omitted, so two runs produce identical files.
"""

from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin

from chaotic_pfc.analysis.sweep import load_sweep, run_sweep
from chaotic_pfc.analysis.sweep_plotting import (
    chaotic_density_norm_table,
    plot_chaotic_density_norm,
)
from chaotic_pfc.comms.channel import fir_channel, ideal_channel
from chaotic_pfc.comms.metrics import comm_metrics
from chaotic_pfc.comms.receiver import receive, receive_order_n
from chaotic_pfc.comms.transmitter import transmit, transmit_order_n
from chaotic_pfc.config import DEFAULT_CONFIG as CFG
from chaotic_pfc.dynamics.maps import henon_generalised
from chaotic_pfc.dynamics.signals import text_message
from chaotic_pfc.dynamics.spectral import psd_normalised
from chaotic_pfc.plotting.figures import setup_rc

JCIS_W = 3.50
JCIS_LABEL = 9.0
JCIS_TICK = 8.0
JCIS_LEGEND = 7.5
SVG_SALT = "chaotic-pfc-jcis"
COMM_WINDOW = (2000, 2400)
FIR_LOOP_CUTOFF = 0.52  # omega_c/pi of the FIR in the loop of Fig. 3(b), lambda_1 > 0.05
DATA_DIR = Path("data/sweeps")
FILTER_TYPES = ("lowpass", "highpass", "bandpass", "bandstop")


class _PanelOptions(TypedDict, total=False):
    figsize: tuple[float, float]
    label_fontsize: float
    tick_fontsize: float
    legend_fontsize: float
    ytick_step: float
    xtick_step: int
    legend: bool
    lang: str
    panel_label_loc: str


# Camera-ready JCIS classification panels (one IEEEtran column). This is the
# single definition of which classification figures the JCIS deliverables need;
# ``chaotic-pfc run paper-figures`` imports it from here rather than repeating
# the list and the plotting call.
JCIS_FIG: _PanelOptions = dict(
    figsize=(3.50, 2.30),
    label_fontsize=9.0,
    tick_fontsize=8.0,
    legend_fontsize=7.5,
    ytick_step=0.2,
    xtick_step=10,
    legend=True,
    lang="en",
)

JCIS_FIGURES: tuple[tuple[str, str], ...] = (
    ("kaiser/lowpass/beta_13.50", "kaiser_b13p5_lowpass_classification_interleaved"),
    ("kaiser/highpass/beta_11.50", "kaiser_b11p5_highpass_classification_interleaved"),
    ("kaiser/bandpass/beta_13.50", "kaiser_b13p5_bandpass_classification_interleaved"),
    ("Rectangular (bandstop)", "rectangular_bandstop_classification_interleaved"),
)


def pct(count: int, total: int, places: int) -> str:
    """Percentage from integers, quantized once with ROUND_HALF_UP."""
    q = Decimal(1).scaleb(-places)
    return str((Decimal(100 * count) / Decimal(total)).quantize(q, rounding=ROUND_HALF_UP))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="svg",
        facecolor="white",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": None},
    )
    plt.close(fig)


def _first_index(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def fig_henon_signals(out_dir: Path) -> None:
    """Write the Hénon signal panels."""
    n_steps = 99
    rows = {}
    for label, alpha in (("a", 1.4), ("b", 0.9)):
        x1_a, _ = henon_generalised(n_steps, x0=0.0, y0=0.0, alpha=alpha, beta=0.3)
        x1_b, _ = henon_generalised(n_steps, x0=0.0, y0=0.001, alpha=alpha, beta=0.3)
        rows[label] = (alpha, x1_a, x1_b)

    alpha_a, xa1, xa2 = rows["a"]
    delta_a = np.abs(xa1 - xa2)
    n_01 = _first_index(delta_a > 0.1)
    n_1 = _first_index(delta_a >= 1.0)
    print(f"[fig_henon_signals] (a) alpha={alpha_a}: first n with |dx1| > 0.1: {n_01}")
    print(f"[fig_henon_signals] (a) alpha={alpha_a}: first n with |dx1| >= 1: {n_1}")

    alpha_b, xb1, xb2 = rows["b"]
    print(
        f"[fig_henon_signals] (b) alpha={alpha_b}: period-2 values x1(98)={xb1[98]:.4f}, "
        f"x1(99)={xb1[99]:.4f}; x1(96)={xb1[96]:.4f} (same orbit: {np.isclose(xb1[96], xb1[98])})"
    )
    print(
        f"[fig_henon_signals] (b) alpha={alpha_b}: max|dx1| over n in [50, 99] = "
        f"{np.max(np.abs(xb1[50:] - xb2[50:])):.3e}"
    )

    fig, axes = plt.subplots(2, 1, figsize=(JCIS_W, 4.6), sharex=True)
    n = np.arange(n_steps + 1)
    for ax, (label, alpha, x1, x2) in zip(
        axes, (("a", alpha_a, xa1, xa2), ("b", alpha_b, xb1, xb2)), strict=True
    ):
        ax.plot(n, x1, color="tab:blue", lw=0.9, label=r"$x_1(0)=x_2(0)=0$")
        ax.plot(n, x2, color="tab:red", lw=0.9, ls="--", label=r"$x_1(0)=0,\ x_2(0)=0.001$")
        ax.set_title(
            f"({label})  " + rf"$\alpha={alpha}$, $\beta=0.3$",
            loc="left",
            fontsize=JCIS_LABEL,
            pad=2,
        )
        ax.set_ylabel(r"$x_1(n)$", fontsize=JCIS_LABEL)
        ax.tick_params(labelsize=JCIS_TICK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=JCIS_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
    )
    axes[1].set_xlabel("$n$", fontsize=JCIS_LABEL)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out_dir / "fig_henon_signals.svg")


def fig_comm_bandlimited(out_dir: Path) -> None:
    """Write the band-limited FIR channel figure."""
    comm = CFG.comm
    alpha, beta, mu = comm.henon.alpha, comm.henon.beta, comm.mu
    tr = comm.transient
    delay = (CFG.channel.num_taps - 1) // 2
    m = text_message(comm.message_text, comm.N, bit_period=comm.message_period)

    rng = np.random.default_rng(CFG.seed)
    s_a = transmit(m, mu=mu, alpha=alpha, beta=beta, x0=0.0, y0=0.0)
    r_a, _ = fir_channel(s_a, cutoff=CFG.channel.cutoff, num_taps=CFG.channel.num_taps)
    m_hat_a = receive(r_a, mu=mu, alpha=alpha, beta=beta, y0=rng.random(), z0=rng.random())

    c = firwin(
        numtaps=CFG.internal_fir.num_taps,
        cutoff=FIR_LOOP_CUTOFF,
        window="hamming",
        pass_zero=True,
        fs=2.0,
    )

    # Largest Lyapunov exponent of the loop filter itself, quoted in Fig. 5(b).
    # A single-cutoff sweep keeps the perturbation draw fixed and independent
    # of the other cutoffs (the per-cutoff slices of the ensemble are drawn in
    # cutoff order, so the first -- here only -- slice never depends on them).
    loop = run_sweep(
        window="hamming",
        filter_type="lowpass",
        orders=[len(c)],
        cutoffs=np.array([FIR_LOOP_CUTOFF]),
        Nitera=CFG.sweep.Nitera,
        Nmap=CFG.sweep.Nmap,
        n_initial=CFG.sweep.n_initial,
        seed=CFG.seed,
        warmup=False,
        adaptive=True,
        Nmap_min=500,
        tol=1e-3,
        fallback_mode=1,
    )
    lam_loop = float(loop.h[0, 0])
    print(
        f"[fig_comm_bandlimited] loop-filter lambda_1 = {lam_loop:.3f} "
        f"({lam_loop:.6f}) for Nz={len(c) - 1}, cutoff={FIR_LOOP_CUTOFF}pi, "
        f"window=hamming, n_initial={CFG.sweep.n_initial}, seed={CFG.seed}"
    )

    rng = np.random.default_rng(CFG.seed)
    x0 = 0.5 * rng.random(len(c))
    y0 = 0.5 * rng.random(len(c))
    s_b, _ = transmit_order_n(m, c, mu=mu, alpha=alpha, beta=beta, x0=x0)
    r_b, _ = fir_channel(s_b, cutoff=CFG.channel.cutoff, num_taps=CFG.channel.num_taps)
    m_hat_b, _ = receive_order_n(r_b, c, mu=mu, alpha=alpha, beta=beta, y0=y0)

    for label, m_hat in (("a", m_hat_a), ("b", m_hat_b)):
        met = comm_metrics(m, m_hat, transient=tr, bit_period=comm.message_period, delay=delay)
        print(
            f"[fig_comm_bandlimited] ({label}) BER={met['ber']:.4f} "
            f"MSE={met['mse']:.4e} (delay={delay}, transient={tr})"
        )
    print(
        f"[fig_comm_bandlimited] internal FIR: numtaps={len(c)} (Nz={len(c) - 1}), "
        f"cutoff={FIR_LOOP_CUTOFF}pi, window=hamming"
    )

    start, stop = COMM_WINDOW
    n = np.arange(start, stop)
    delayed = m[n - delay]

    bers = {
        label: comm_metrics(m, m_hat, transient=tr, bit_period=comm.message_period, delay=delay)[
            "ber"
        ]
        for label, m_hat in (("a", m_hat_a), ("b", m_hat_b))
    }
    descriptions = {
        "a": "without a filter in the loop",
        "b": rf"FIR in the loop ($N_z={len(c) - 1}$)",
    }
    fig, axes = plt.subplots(2, 1, figsize=(JCIS_W, 4.6), sharex=True)
    for ax, (label, m_hat) in zip(axes, (("a", m_hat_a), ("b", m_hat_b)), strict=True):
        ax.plot(n, np.clip(m_hat[n], -3.0, 3.0), color="tab:blue", lw=0.8, label=r"$\hat m[n]$")
        ax.plot(n, delayed, color="tab:red", lw=0.8, ls="--", label=r"$m[n-100]$")
        ax.set_ylim(-3.2, 3.2)
        ax.set_yticks([-3, 0, 3])
        ax.set_title(
            f"({label}) {descriptions[label]}, BER={bers[label]:.4f}",
            loc="left",
            fontsize=JCIS_LABEL,
            pad=2,
        )
        ax.set_ylabel("amplitude", fontsize=JCIS_LABEL)
        ax.tick_params(labelsize=JCIS_TICK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=JCIS_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
    )
    axes[1].set_xlabel("$n$", fontsize=JCIS_LABEL)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out_dir / "fig_comm_bandlimited.svg")


def fig_kaiser_beta(out_dir: Path) -> None:
    """Write the Kaiser-β Lyapunov curve."""
    fig, ax = plt.subplots(figsize=(JCIS_W, 2.6))
    markers = {"lowpass": "o", "highpass": "s", "bandpass": "^", "bandstop": "D"}
    for ft in FILTER_TYPES:
        rows = []
        for path in sorted((DATA_DIR / "kaiser" / ft).glob("beta_*/variables_lyapunov.npz")):
            sr = load_sweep(path)
            beta = float(sr.metadata["kaiser_beta"])
            h = np.asarray(sr.h)
            rows.append((beta, pct(int(np.count_nonzero(h > 0.0)), int(h.size), 1)))
        rows.sort()
        betas = np.array([r[0] for r in rows])
        rho = np.array([float(r[1]) for r in rows])
        ax.plot(betas, rho, marker=markers[ft], ms=3.0, lw=0.9, label=ft)
        print(f"[fig_kaiser_beta] {ft}: " + " ".join(f"{b:g}:{r}" for b, r in rows))
        best = int(np.argmax(rho))
        print(f"[fig_kaiser_beta] {ft}: max rho_abs={rows[best][1]}% at beta_K={betas[best]:g}")
    ax.set_xlabel(r"$\beta_K$", fontsize=JCIS_LABEL)
    ax.set_ylabel(r"$\rho_c^{\mathrm{abs}}$ (\%)", fontsize=JCIS_LABEL)
    ax.tick_params(labelsize=JCIS_TICK)
    ax.legend(fontsize=JCIS_LEGEND, frameon=False)
    fig.tight_layout()
    _save(fig, out_dir / "fig_kaiser_beta.svg")


def fig_chaotic_density_norm(out_dir: Path) -> None:
    """Write the normalised chaotic-density map."""
    Nz, cutoffs, percent, n_containing = chaotic_density_norm_table(DATA_DIR)
    for parity, mask in (("odd", Nz % 2 == 1), ("even", Nz % 2 == 0)):
        sub = np.where(mask[:, None], percent, -np.inf)
        k, j = np.unravel_index(np.argmax(sub), sub.shape)
        n = int(n_containing[k])
        count = round(float(percent[k, j]) * n / 100.0)
        print(
            f"[fig_chaotic_density_norm] max over Nz {parity}: Nz={Nz[k]}, "
            f"omega_c/pi={cutoffs[j]:.4f}, {pct(count, n, 1)}%, k={count} of n={n}"
        )
    fig = plot_chaotic_density_norm(
        DATA_DIR,
        lang="en",
        figsize=(JCIS_W, 2.30),
        label_fontsize=9.0,
        tick_fontsize=8.0,
        cbar_fontsize=8.0,
        show_title=False,
        marker_size=6.0,
        data_slots=1,
        gap_slots=0,
        ytick_step=0.2,
        xtick_step=10,
    )
    _save(fig, out_dir / "fig_chaotic_density_norm.svg")


def fig_comm_ideal(out_dir: Path) -> None:
    """Write the ideal-channel communication figure."""
    comm, spec = CFG.comm, CFG.spectral
    alpha, beta, mu = comm.henon.alpha, comm.henon.beta, comm.mu
    tr = comm.transient
    m = text_message(comm.message_text, comm.N, bit_period=comm.message_period)
    s = transmit(m, mu=mu, alpha=alpha, beta=beta, x0=0.0, y0=0.0)
    r = ideal_channel(s)
    rng = np.random.default_rng(CFG.seed)
    m_hat = receive(r, mu=mu, alpha=alpha, beta=beta, y0=rng.random(), z0=rng.random())

    met = comm_metrics(m, m_hat, transient=tr, bit_period=comm.message_period, delay=0)
    print(
        f"[fig_comm_ideal] MSE(n>{tr})={np.mean((m[tr:] - m_hat[tr:]) ** 2):.4e} BER={met['ber']:.4f}"
    )

    def psd(x):
        return psd_normalised(
            x, spec.nfft, spec.window_length, window=spec.window, kaiser_beta=spec.kaiser_beta
        )

    omega, psd_m = psd(m)
    psd_s = psd(s)[1]
    psd_r = psd(r)[1]
    psd_mhat = psd(m_hat)[1]

    win = slice(CFG.plot.time_window_start, CFG.plot.time_window_end)
    n = np.arange(len(m))[win]
    rows = [
        ("(a) $m[n]$", m, (-1.5, 1.5), "tab:blue"),
        ("(b) $s[n]$", s, (-2.5, 2.5), "tab:blue"),
        ("(c) $r[n]$", r, (-2.5, 2.5), "tab:blue"),
        ("(d) $\\hat{m}[n]$", m_hat, (-1.5, 1.5), "tab:blue"),
    ]
    spectra = [
        ("(e) $\\mathcal{M}(\\omega)$", psd_m),
        ("(f) $S(\\omega)$", psd_s),
        ("(g) $R(\\omega)$", psd_r),
        ("(h) $\\hat{\\mathcal{M}}(\\omega)$", psd_mhat),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(6.84, 5.88), sharex="col")
    for row, (label, sig, ylim, color) in enumerate(rows):
        ax = axes[row, 0]
        ax.plot(n, sig[win], color=color, lw=0.9)
        ax.set_ylim(*ylim)
        ax.set_ylabel(label, fontsize=JCIS_LABEL)
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=JCIS_TICK)
    for row, (label, spec_y) in enumerate(spectra):
        ax = axes[row, 1]
        ax.plot(omega, spec_y, color="tab:orange", lw=0.9)
        ax.set_ylabel(label, fontsize=JCIS_LABEL)
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=JCIS_TICK)
    axes[3, 0].set_xlabel("$n$", fontsize=JCIS_LABEL)
    axes[3, 1].set_xlabel(r"$\omega/\pi$", fontsize=JCIS_LABEL)
    axes[3, 1].set_xlim(0.0, 1.0)
    fig.tight_layout()
    _save(fig, out_dir / "fig_comm_ideal.svg")


def fig_classification(out_dir: Path) -> None:
    """Write the four JCIS classification maps."""
    from chaotic_pfc.analysis.sweep_plotting import plot_classification_interleaved

    for k, (name, stem) in enumerate(JCIS_FIGURES):
        sr = load_sweep(DATA_DIR / name / "variables_lyapunov.npz")
        opts: _PanelOptions = {**JCIS_FIG, "legend": k == 0}
        fig = plot_classification_interleaved(sr, **opts)
        _save(fig, out_dir / f"{stem}.svg")


GENERATORS = {
    "henon": fig_henon_signals,
    "comm": fig_comm_bandlimited,
    "ideal": fig_comm_ideal,
    "classification": fig_classification,
    "kaiser": fig_kaiser_beta,
    "density": fig_chaotic_density_norm,
}


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``chaotic-pfc paper figures``."""
    parser = subparsers.add_parser(
        "figures",
        help="Generate the JCIS figures that other commands do not produce.",
        description="Generate the JCIS figures that other commands do not produce.",
    )
    parser.add_argument("--out-dir", default="articles/jcis/Figures", type=Path)
    parser.add_argument("--only", choices=["all", *GENERATORS], default="all")
    parser.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute ``chaotic-pfc paper figures``."""
    setup_rc()
    plt.rcParams["svg.hashsalt"] = SVG_SALT
    names = list(GENERATORS) if args.only == "all" else [args.only]
    for name in names:
        GENERATORS[name](args.out_dir)
    print(f"figures written to {args.out_dir}")
    return 0
