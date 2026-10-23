"""Compare two sets of Lyapunov sweep checkpoints, file by file.

For each checkpoint present in both directories this prints the counts of
chaotic (Nc), non-chaotic (Nnc), bounded (Nb = Nc + Nnc) and divergent points,
the number of grid points that changed class and in which direction, and the
mean and maximum |delta lambda| over points that are chaotic in both sets.
A summary over all files closes the report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD, load_sweep


def classes(h: np.ndarray) -> np.ndarray:
    """Return the discrete class label for each cell."""
    out = np.empty(h.shape, dtype="<U12")
    out[np.isnan(h)] = "divergent"
    out[~np.isnan(h) & (h > LAMBDA_CHAOS_THRESHOLD)] = "chaotic"
    out[~np.isnan(h) & (h <= LAMBDA_CHAOS_THRESHOLD)] = "non-chaotic"
    return out


def compare(old_dir: Path, new_dir: Path) -> None:
    """Print a file-by-file comparison of two sweep directories."""
    rel_paths = sorted(p.relative_to(old_dir) for p in old_dir.rglob("variables_lyapunov.npz"))
    totals: Counter[str] = Counter[str]()
    directions: Counter[str] = Counter[str]()
    max_abs = 0.0
    print(
        f"{'checkpoint':<60} {'Nc o->n':>12} {'Nnc o->n':>12} {'Ndiv o->n':>12} "
        f"{'changed':>8} {'mean|dl|':>10} {'max|dl|':>10} {'nfb o->n':>12}"
    )
    for rel in rel_paths:
        new_path = new_dir / rel
        if not new_path.exists():
            print(f"{rel!s:<60} MISSING in {new_dir}")
            continue
        old = load_sweep(old_dir / rel)
        new = load_sweep(new_path)
        co, cn = classes(np.asarray(old.h)), classes(np.asarray(new.h))
        n = co.size
        nc_o, nc_n = int((co == "chaotic").sum()), int((cn == "chaotic").sum())
        np_o, np_n = int((co == "non-chaotic").sum()), int((cn == "non-chaotic").sum())
        nd_o, nd_n = int((co == "divergent").sum()), int((cn == "divergent").sum())
        changed = co != cn
        for a, b in zip(co[changed], cn[changed], strict=True):
            directions[f"{a}->{b}"] += 1
        both = (co == "chaotic") & (cn == "chaotic")
        dl = np.abs(np.asarray(new.h)[both] - np.asarray(old.h)[both])
        mean_dl = float(dl.mean()) if dl.size else float("nan")
        max_dl = float(dl.max()) if dl.size else float("nan")
        if dl.size:
            max_abs = max(max_abs, max_dl)
        nfb_o = int(old.metadata.get("n_fallback", -1))
        nfb_n = int(new.metadata.get("n_fallback", -1))
        print(
            f"{rel!s:<60} {nc_o:>5}->{nc_n:<6} {np_o:>5}->{np_n:<6} {nd_o:>5}->{nd_n:<6} "
            f"{int(changed.sum()):>8} {mean_dl:>10.3e} {max_dl:>10.3e} {nfb_o:>5}->{nfb_n:<6}"
        )
        totals["files"] += 1
        totals["grid_points"] += n
        totals["changed"] += int(changed.sum())
        totals["Nc_old"] += nc_o
        totals["Nc_new"] += nc_n
    print()
    print(f"files compared: {totals['files']}  grid points: {totals['grid_points']}")
    print(f"points that changed class: {totals['changed']}")
    print(f"total chaotic points: old={totals['Nc_old']} new={totals['Nc_new']}")
    print(f"class changes by direction: {dict(directions)}")
    print(f"max |delta lambda| over chaotic-in-both points: {max_abs:.6e}")


def add_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    """Register ``chaotic-pfc sweeps compare``."""
    parser = subparsers.add_parser(
        "compare",
        help="Compare two sweep directories file by file.",
        description="Compare two sets of Lyapunov sweep checkpoints, file by file.",
    )
    parser.add_argument(
        "--old", type=Path, required=True, help="Directory of the reference checkpoints."
    )
    parser.add_argument(
        "--new", type=Path, required=True, help="Directory of the checkpoints to compare."
    )
    parser.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute ``chaotic-pfc sweeps compare``."""
    compare(args.old, args.new)
    return 0
