"""Regenerate the versioned Lyapunov sweeps with the current kernel.

Every checkpoint under ``--src`` is recomputed with the parameters stored in
its own metadata (window, filter type, Kaiser beta, orders, cutoffs and the
simulation settings), and written to the same relative path under ``--dst``.
The bandwidth of the band filters is the one used for the paper, 0.05.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from chaotic_pfc.analysis.sweep import load_sweep, run_sweep, save_sweep

PAPER_BANDWIDTH = 0.05


def regenerate(src: Path, dst: Path) -> None:
    paths = sorted(src.rglob("variables_lyapunov.npz"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints under {src}")
    for k, path in enumerate(paths, 1):
        old = load_sweep(path)
        m = old.metadata
        t0 = time.perf_counter()
        result = run_sweep(
            window=old.window,
            filter_type=old.filter_type,
            kaiser_beta=float(m["kaiser_beta"]),
            orders=old.orders,
            cutoffs=old.cutoffs,
            bandwidth=PAPER_BANDWIDTH,
            Nitera=int(m["Nitera"]),
            Nmap=int(m["Nmap"]),
            n_initial=int(m["n_initial"]),
            alpha=float(m["alpha"]),
            beta=float(m["beta"]),
            seed=int(m["seed"]),
            adaptive=bool(m["adaptive"]),
            Nmap_min=int(m["Nmap_min"]),
            tol=float(m["tol"]),
            fallback_mode=1,
            warmup=True,
        )
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_sweep(result, out)
        print(
            f"[{k:3d}/{len(paths)}] {path.relative_to(src)} "
            f"n_fallback={result.metadata['n_fallback']} "
            f"({time.perf_counter() - t0:.1f} s)",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=Path("data/sweeps"))
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()
    regenerate(args.src, args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
