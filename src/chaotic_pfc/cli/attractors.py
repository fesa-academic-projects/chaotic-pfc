"""Phase-space attractors for three Hénon variants."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import add_lang_flag, add_save_display_flags, pick_backend


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run attractors`` subcommand."""
    p = subparsers.add_parser(
        "attractors",
        help="Phase-space attractors for three Hénon variants.",
        description="Phase-space attractors for three Hénon variants.",
    )
    p.add_argument("--steps", type=int, default=50_000)
    add_save_display_flags(p)
    add_lang_flag(p)
    p.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute the ``attractors`` experiment."""
    headless = pick_backend(args.no_display)

    import matplotlib.pyplot as plt

    from chaotic_pfc._i18n import t
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg
    from chaotic_pfc.dynamics.maps import henon_filtered, henon_generalised, henon_standard
    from chaotic_pfc.plotting.figures import plot_attractor

    lang = args.lang
    a, b = cfg.comm.henon.a, cfg.comm.henon.b
    steps, fmt = args.steps, cfg.plot.fmt
    fdir = Path(cfg.plot.figures_dir)
    if args.save:
        fdir.mkdir(parents=True, exist_ok=True)

    def _build_path(name: str) -> str | None:
        return str(fdir / name) if args.save else None

    print(f"[01] Hénon attractors  |  steps={steps:,}")

    X, Y = henon_standard(steps, a=a, b=b)
    f1 = plot_attractor(
        X,
        Y,
        title=t("attractor.standard", lang=lang),
        xlabel=r"$x[n]$",
        ylabel=r"$y[n]$",
        save_path=_build_path(f"attractor_standard.{fmt}"),
    )

    X, Y = henon_generalised(steps, alpha=a, beta=b)
    f2 = plot_attractor(
        X,
        Y,
        title=t("attractor.generalised", lang=lang),
        xlabel=r"$x_1[n]$",
        ylabel=r"$x_2[n]$",
        save_path=_build_path(f"attractor_generalised.{fmt}"),
    )

    X, Y = henon_filtered(steps, alpha=a, beta=b, c0=1.0, c1=0.0)
    f3 = plot_attractor(
        X,
        Y,
        title=t("attractor.filtered", lang=lang),
        xlabel=r"$x_1[n]$",
        ylabel=r"$x_2[n]$",
        save_path=_build_path(f"attractor_filtered.{fmt}"),
    )

    if headless:
        for f in [f1, f2, f3]:
            plt.close(f)
        if args.save:
            print(f"    Saved -> {fdir}/attractor_*.{fmt}")
    else:
        plt.show()
    return 0
