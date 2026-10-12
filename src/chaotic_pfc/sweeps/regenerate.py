"""Regenerate the versioned Lyapunov sweeps with the current kernel.

Every checkpoint under ``--src`` is recomputed with the parameters stored in
its own metadata (window, filter type, Kaiser beta, orders, cutoffs and the
simulation settings), and written to the same relative path under ``--dst``.
The bandwidth of the band filters is the one used for the paper, 0.05.

After regenerating, a ``SHA256SUMS`` manifest pinning every checkpoint is
written under ``--dst``; ``--manifest-only`` rewrites the manifest for an
existing tree under ``--src`` without recomputing anything.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from chaotic_pfc.analysis.sweep import load_sweep, run_sweep, save_sweep

PAPER_BANDWIDTH = 0.05
_MANIFEST_NAME = "SHA256SUMS"


def write_manifest(root: Path) -> Path:
    """Write ``SHA256SUMS`` for every checkpoint under ``root``.

    Paths are relative to ``root``, so ``cd <root> && sha256sum -c SHA256SUMS``
    verifies the tree. Returns the manifest path.
    """
    files = sorted(root.rglob("variables_lyapunov.npz"))
    if not files:
        raise FileNotFoundError(f"no checkpoints under {root}")
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n"
        for path in files
    ]
    manifest = root / _MANIFEST_NAME
    manifest.write_text("".join(lines), encoding="utf-8")
    return manifest


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
    print(f"wrote {write_manifest(dst)}", flush=True)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``chaotic-pfc sweeps regenerate``."""
    parser = subparsers.add_parser(
        "regenerate",
        help="Recompute the versioned sweep checkpoints.",
        description="Recompute the versioned sweep checkpoints with the current kernel.",
    )
    parser.add_argument("--src", type=Path, default=Path("data/sweeps"))
    parser.add_argument("--dst", type=Path)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="only rewrite the SHA256SUMS manifest under --src",
    )
    parser.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute ``chaotic-pfc sweeps regenerate``."""
    if args.manifest_only:
        print(f"wrote {write_manifest(args.src)}")
        return 0
    if args.dst is None:
        raise SystemExit("--dst is required unless --manifest-only is given")
    regenerate(args.src, args.dst)
    return 0
