#!/usr/bin/env python3
"""Measure the complete 184-sweep computation.

The script reproduces the canonical sweep list from ``chaotic-pfc run all``
without writing checkpoints. It runs one small warm-up sweep to compile the
Numba kernels, then times the 184 sweeps with the production settings.

Usage::

    python tools/measure_sweeps.py --out benchmarks/baseline/sweeps_184.json

The JSON records the machine, package versions, thread count, warm-up time,
per-sweep timings and the total time.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return platform.processor() or "unknown"
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_gib() -> float | None:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 1024.0**2, 1)
    return None


def _governor() -> str:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _git(args: list[str]) -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        result = subprocess.run(  # noqa: S603
            [git, *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _metadata() -> dict[str, Any]:
    import matplotlib
    import numba
    import numpy
    import scipy

    import chaotic_pfc
    from chaotic_pfc._compat import HAS_NUMBA, get_num_threads

    return {
        "machine": {
            "platform": platform.platform(),
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "ram_gib": _ram_gib(),
            "governor": _governor(),
        },
        "versions": {
            "python": sys.version.split()[0],
            "chaotic_pfc": chaotic_pfc.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "numba": numba.__version__,
        },
        "numba": {
            "available": HAS_NUMBA,
            "threads": get_num_threads(),
            "disable_jit": os.environ.get("NUMBA_DISABLE_JIT", "0"),
        },
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"])),
        },
    }


def _sweep_list() -> list[tuple[str, str, float | None]]:
    from chaotic_pfc.analysis.sweep import FILTER_TYPES, WINDOWS
    from chaotic_pfc.cli.sweep import beta_values

    betas = beta_values(0.5, 20.0, 0.5)
    sweeps: list[tuple[str, str, float | None]] = []
    for window in WINDOWS:
        if window == "kaiser":
            sweeps.extend((window, ft, b) for ft in FILTER_TYPES for b in betas)
        else:
            sweeps.extend((window, ft, None) for ft in FILTER_TYPES)
    return sweeps


def _warmup() -> float:
    from chaotic_pfc.analysis.sweep import run_sweep

    start = time.perf_counter()
    run_sweep(
        window="hamming",
        filter_type="lowpass",
        num_taps_grid=[3],
        cutoff_grid=None,
        Nitera=50,
        Nmap=200,
        n_initial=1,
        seed=42,
        warmup=True,
    )
    return time.perf_counter() - start


def _run(args: argparse.Namespace) -> int:
    import numpy as np

    from chaotic_pfc.analysis.sweep import run_sweep

    sweeps = _sweep_list()
    if len(sweeps) != 184:
        raise SystemExit(f"expected 184 sweeps, got {len(sweeps)}")

    warmup_seconds = _warmup()
    rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for index, (window, filter_type, beta) in enumerate(sweeps, start=1):
        started = time.perf_counter()
        result = run_sweep(
            window=window,
            filter_type=filter_type,
            num_taps_grid=None,
            cutoff_grid=None,
            Nitera=500,
            Nmap=3000,
            n_initial=25,
            seed=42,
            warmup=False,
            kaiser_beta=beta if beta is not None else 5.0,
            bandwidth=0.05,
            adaptive=True,
            Nmap_min=500,
            tol=1e-3,
        )
        elapsed = time.perf_counter() - started
        valid = int((~np.isnan(result.h)).sum())
        total = int(result.h.size)
        rows.append(
            {
                "index": index,
                "window": window,
                "filter_type": filter_type,
                "kaiser_beta": beta,
                "elapsed_s": round(elapsed, 6),
                "valid_points": valid,
                "total_points": total,
                "points_per_s": round(total / elapsed, 3) if elapsed else None,
            }
        )
        print(
            f"[{index:3d}/{len(sweeps)}] {window:14s} {filter_type:8s} "
            f"{'beta=' + format(beta, '.1f') if beta is not None else '':12s} "
            f"{elapsed:7.2f} s  {total / elapsed:8.1f} pts/s",
            flush=True,
        )
        del result
        if index % 16 == 0:
            gc.collect()

    elapsed_total = time.perf_counter() - total_start
    payload: dict[str, Any] = {
        **_metadata(),
        "protocol": {
            "sweeps": len(sweeps),
            "warmup": "one small run_sweep with warmup=True, excluded from the timings",
            "Nitera": 500,
            "Nmap": 3000,
            "n_initial": 25,
            "bandwidth": 0.05,
            "adaptive": True,
            "Nmap_min": 500,
            "tol": 1e-3,
            "repetitions": 1,
        },
        "warmup_seconds": round(warmup_seconds, 6),
        "sweeps_seconds": round(elapsed_total, 6),
        "total_seconds": round(warmup_seconds + elapsed_total, 6),
        "total_minutes": round((warmup_seconds + elapsed_total) / 60.0, 3),
        "sweeps": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"184 sweeps: {elapsed_total:.1f} s ({elapsed_total / 60.0:.2f} min)")
    print(f"with warm-up: {warmup_seconds + elapsed_total:.1f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="benchmarks/baseline/sweeps_184.json",
        help="Output JSON path.",
    )
    return _run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
