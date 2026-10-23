"""Regenerate the versioned Lyapunov sweeps with the current kernel.

Every checkpoint under ``--src`` is recomputed with the parameters stored in
its own metadata (window, filter type, Kaiser beta, num_taps_grid, cutoff_grid and the
simulation settings), and written to the same relative path under ``--dst``.
The bandwidth of the band filters is the one used for the paper, 0.05.

After regenerating, an ``ENVIRONMENT.json`` recording the package stack,
platform and git state is written under ``--dst``, together with a
``SHA256SUMS`` manifest pinning every checkpoint. ``--manifest-only`` rewrites
the manifest for an existing tree under ``--src`` without recomputing
anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import Any

from chaotic_pfc._output import progress
from chaotic_pfc._version import __version__
from chaotic_pfc.analysis.sweep import load_sweep, run_sweep, save_sweep

PAPER_BANDWIDTH = 0.05
_MANIFEST_NAME = "SHA256SUMS"
_ENVIRONMENT_NAME = "ENVIRONMENT.json"
_PACKAGE_NAMES = ("numpy", "scipy", "numba", "llvmlite", "matplotlib")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_metadata() -> tuple[str | None, bool | None]:
    """Return ``(HEAD, dirty)`` for the checkout containing this module.

    Installed wheels do not have a checkout; in that case both values are
    ``None`` rather than a fabricated commit or a false cleanliness claim.
    """
    git = shutil.which("git")
    if git is None:
        return None, None
    try:
        commit = subprocess.check_output(  # noqa: S603
            [git, "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(  # noqa: S603
            [
                git,
                "-C",
                str(_REPO_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                ".",
                ":(exclude)data/sweeps",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit or None, bool(status.strip())


def _package_versions() -> dict[str, str]:
    """Return the installed versions of the packages that anchor the sweeps."""
    return {name: importlib.metadata.version(name) for name in _PACKAGE_NAMES}


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


def write_environment(
    root: Path,
    *,
    checkpoints: int | None = None,
    elapsed_seconds: float | None = None,
) -> Path:
    """Write ``ENVIRONMENT.json`` for the checkpoints under ``root``.

    The record ties the tree to the Python and scientific stack that wrote it,
    the package version from ``_version.py``, the platform and the git commit
    and dirty state. Returns the environment file path.
    """
    if checkpoints is None:
        checkpoints = len(sorted(root.rglob("variables_lyapunov.npz")))
    if checkpoints <= 0:
        raise FileNotFoundError(f"no checkpoints under {root}")
    commit, dirty = _git_metadata()
    environment: dict[str, object] = {
        "description": (
            f"Environment that regenerated the {checkpoints} versioned sweep "
            "checkpoints in data/sweeps/ and reproduces data/sweeps/SHA256SUMS "
            "bit for bit."
        ),
        "python": platform.python_version(),
        **_package_versions(),
        "platform": platform.platform(),
        "package_version": __version__,
        "regeneration_date": date.today().isoformat(),
        "checkpoints": checkpoints,
        "git_commit": commit,
        "git_dirty": dirty,
        "verification": (
            "CHAOTIC_PFC_RUN_BASELINE=1 pytest tests/test_sweep_regressions.py "
            "reproduces the checkpoints bit for bit in this environment, and "
            "requirements-lock.txt is aligned to these versions."
        ),
    }
    if elapsed_seconds is not None:
        environment["regeneration_wall_seconds"] = round(elapsed_seconds, 3)
    path = root / _ENVIRONMENT_NAME
    path.write_text(json.dumps(environment, indent=4) + "\n", encoding="utf-8")
    return path


def regenerate(src: Path, dst: Path, args: Any = None) -> None:
    """Recompute every checkpoint under ``src`` into ``dst``."""
    paths = sorted(src.rglob("variables_lyapunov.npz"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints under {src}")
    started = time.monotonic()
    for k, path in enumerate(paths, 1):
        old = load_sweep(path)
        m = old.metadata
        result = run_sweep(
            window=old.window,
            filter_type=old.filter_type,
            kaiser_beta=float(m["kaiser_beta"]),
            num_taps_grid=old.num_taps_grid,
            cutoff_grid=old.cutoff_grid,
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
        progress(k, len(paths), str(path.relative_to(src)), args)
    elapsed = time.monotonic() - started
    print(f"wrote {write_manifest(dst)}", flush=True)
    print(
        f"wrote {write_environment(dst, checkpoints=len(paths), elapsed_seconds=elapsed)}",
        flush=True,
    )


def add_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    """Register ``chaotic-pfc sweeps regenerate``."""
    parser = subparsers.add_parser(
        "regenerate",
        help="Recompute the versioned sweep checkpoints.",
        description="Recompute the versioned sweep checkpoints with the current kernel.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/sweeps"),
        help="Directory of checkpoints to read (default: data/sweeps).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        help="Directory to write the regenerated checkpoints to (required unless --manifest-only).",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only rewrite the SHA256SUMS manifest under --src (no recomputation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be regenerated without writing anything.",
    )
    parser.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute ``chaotic-pfc sweeps regenerate``."""
    if args.dry_run:
        paths = sorted(args.src.rglob("variables_lyapunov.npz"))
        if not paths:
            raise FileNotFoundError(f"no checkpoints under {args.src}")
        print(f"would regenerate {len(paths)} checkpoint(s) from {args.src}")
        if args.dst is not None:
            print(f"  -> {args.dst}")
        return 0
    if args.manifest_only:
        print(f"wrote {write_manifest(args.src)}")
        return 0
    if args.dst is None:
        raise SystemExit("--dst is required unless --manifest-only is given")
    regenerate(args.src, args.dst, args=args)
    return 0
