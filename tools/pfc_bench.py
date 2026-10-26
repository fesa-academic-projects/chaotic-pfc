#!/usr/bin/env python3
"""Benchmark the numerical core with and without Numba.

The script measures the common workloads in two processes:

* normal Numba JIT mode;
* pure Python with ``NUMBA_DISABLE_JIT=1``.

The canonical 4,000-point lowpass Hamming sweep is measured only in Numba
one-thread and eight-thread modes. The common loads use five repetitions and
the 4,000-point load uses three. The JSON records the machine, package
versions, thread count, per-run times, medians, interquartile ranges, the JIT
factor and the parallel factor.

Usage::

    python tools/pfc_bench.py --out benchmarks/baseline/pfc_bench.json

No checkpoint or figure is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
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

    disable_jit = os.environ.get("NUMBA_DISABLE_JIT", "0")
    numba_threads = get_num_threads()
    effective_threads = 1 if disable_jit == "1" else numba_threads

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
            "threads": effective_threads,
            "numba_threads": numba_threads,
            "disable_jit": disable_jit,
            "env_num_threads": os.environ.get("NUMBA_NUM_THREADS", "default"),
        },
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"])),
        },
    }


def _bench_henon_standard() -> int:
    from chaotic_pfc.dynamics.maps import henon_standard

    x, _ = henon_standard(steps=10_000, x0=0.0, y0=0.0, alpha=1.4, beta=0.3)
    return int(x.size)


def _bench_lyapunov_single() -> float:
    from chaotic_pfc.dynamics.lyapunov import lyapunov_henon2d

    result = lyapunov_henon2d(alpha=1.4, beta=0.3, Nitera=2000, Ndiscard=500, seed=42)
    return float(result.lyapunov_max)


def _bench_lyapunov_ensemble() -> int:
    from chaotic_pfc.dynamics.lyapunov import lyapunov_henon2d_ensemble

    result = lyapunov_henon2d_ensemble(
        alpha=1.4,
        beta=0.3,
        Nitera=500,
        Ndiscard=200,
        n_initial=25,
        seed=42,
        perturbation=0.1,
    )
    return int(result.n_chaotic)


def _bench_fir_bank() -> int:
    import numpy as np

    from chaotic_pfc.analysis.sweep import precompute_fir_bank

    bank, _gains = precompute_fir_bank(
        np.arange(2, 42),
        np.linspace(0.01, 0.99, 100),
        "lowpass",
        "hamming",
    )
    return int(bank.shape[0])


def _bench_mini_sweep() -> int:
    import numpy as np

    from chaotic_pfc.analysis.sweep import run_sweep

    result = run_sweep(
        window="hamming",
        filter_type="lowpass",
        num_taps_grid=np.arange(2, 8),
        cutoff_grid=np.linspace(0.1, 0.9, 5),
        Nitera=50,
        Nmap=200,
        n_initial=3,
        seed=42,
        warmup=False,
    )
    return int(result.h.size)


def _bench_sweep_lowpass_hamming_4000() -> dict[str, Any]:
    """Run the canonical 4,000-point lowpass Hamming sweep."""
    import numpy as np

    from chaotic_pfc.analysis.sweep import run_sweep
    from chaotic_pfc.analysis.sweep_plotting._common import classify

    result = run_sweep(
        window="hamming",
        filter_type="lowpass",
        num_taps_grid=None,
        cutoff_grid=None,
        Nitera=500,
        Nmap=3000,
        n_initial=25,
        seed=42,
        warmup=False,
        kaiser_beta=5.0,
        bandwidth=0.05,
        adaptive=True,
        Nmap_min=500,
        tol=1e-3,
    )
    classes = classify(np.asarray(result.h))
    counts = {
        "periodic": int(np.count_nonzero(classes == -1)),
        "chaotic": int(np.count_nonzero(classes == 0)),
        "divergent": int(np.count_nonzero(classes == 2)),
    }
    return {
        "shape": list(classes.shape),
        "class_sha256": hashlib.sha256(classes.astype(np.int8).tobytes()).hexdigest(),
        "class_counts": counts,
    }


BENCHMARKS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("henon_standard_10000", _bench_henon_standard),
    ("lyapunov_henon2d_2000", _bench_lyapunov_single),
    ("lyapunov_henon2d_ensemble_25", _bench_lyapunov_ensemble),
    ("precompute_fir_bank_40x100", _bench_fir_bank),
    ("mini_sweep_30_points", _bench_mini_sweep),
)

PARALLEL_BENCHMARK: tuple[str, Callable[[], Any]] = (
    "sweep_lowpass_hamming_4000",
    _bench_sweep_lowpass_hamming_4000,
)


def _statistics(times: list[float]) -> dict[str, Any]:
    import numpy as np

    q25, q75 = np.percentile(np.asarray(times), [25.0, 75.0])
    return {
        "runs_s": [round(value, 9) for value in times],
        "mean_s": round(statistics.fmean(times), 9),
        "median_s": round(statistics.median(times), 9),
        "q25_s": round(float(q25), 9),
        "q75_s": round(float(q75), 9),
        "iqr_s": round(float(q75 - q25), 9),
        "min_s": round(min(times), 9),
        "max_s": round(max(times), 9),
    }


def _run(
    mode: str,
    repeat: int,
    parallel_repeat: int,
    only_parallel: bool = False,
) -> dict[str, Any]:
    if mode == "pure":
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        os.environ.pop("NUMBA_NUM_THREADS", None)
    elif mode == "numba1":
        os.environ.pop("NUMBA_DISABLE_JIT", None)
        os.environ["NUMBA_NUM_THREADS"] = "1"
    else:
        os.environ.pop("NUMBA_DISABLE_JIT", None)

    # The first benchmark import happens below, after the environment above is
    # set. In pure mode this guarantees that Numba sees NUMBA_DISABLE_JIT=1 at
    # its first import in this process.
    if only_parallel:
        if mode == "pure":
            raise SystemExit("--only-parallel requires Numba mode")
        items = [PARALLEL_BENCHMARK]
    else:
        items = list(BENCHMARKS)
        if mode != "pure":
            items.append(PARALLEL_BENCHMARK)

    results: dict[str, Any] = {}
    for name, func in items:
        repetitions = parallel_repeat if name == PARALLEL_BENCHMARK[0] else repeat
        func()  # warm-up, excluded from the timings
        times: list[float] = []
        verification: Any = None
        for _ in range(repetitions):
            start = time.perf_counter()
            value = func()
            times.append(time.perf_counter() - start)
            if isinstance(value, dict):
                verification = value
        results[name] = _statistics(times)
        results[name]["repetitions"] = repetitions
        if verification is not None:
            results[name]["verification"] = verification
        print(
            f"[{mode:5s}] {name:32s} median={results[name]['median_s']:.6f} s "
            f"IQR={results[name]['iqr_s']:.6f} s  reps={repetitions}",
            flush=True,
        )

    return {
        **_metadata(),
        "mode": mode,
        "protocol": {
            "repeat": repeat,
            "parallel_repeat": parallel_repeat,
            "parallel_load": PARALLEL_BENCHMARK[0],
            "only_parallel": only_parallel,
            "warmup": "one call per workload, excluded from the timings",
            "pure_python_env": "NUMBA_DISABLE_JIT=1",
            "one_thread_env": "NUMBA_NUM_THREADS=1",
        },
        "benchmarks": results,
        "total_median_s": round(sum(item["median_s"] for item in results.values()), 9),
    }


def _combine(
    numba: dict[str, Any],
    numba1: dict[str, Any],
    pure: dict[str, Any],
    repeat: int,
    parallel_repeat: int,
) -> dict[str, Any]:
    names = set(numba["benchmarks"]) | set(numba1["benchmarks"]) | set(pure["benchmarks"])
    speedup: dict[str, Any] = {}
    for name in sorted(names):
        n_row = numba["benchmarks"].get(name)
        one_row = numba1["benchmarks"].get(name)
        p_row = pure["benchmarks"].get(name)
        n = n_row["median_s"] if n_row else None
        one = one_row["median_s"] if one_row else None
        p = p_row["median_s"] if p_row else None
        speedup[name] = {
            "numba_median_s": n,
            "numba1_median_s": one,
            "pure_median_s": p,
            "jit_over_pure": round(p / one, 3) if p is not None and one else None,
            "parallel_over_numba1": round(one / n, 3) if one is not None and n else None,
            "pure_over_numba": round(p / n, 3) if p is not None and n else None,
        }

    parallel_name = PARALLEL_BENCHMARK[0]
    n_row = numba["benchmarks"].get(parallel_name, {})
    one_row = numba1["benchmarks"].get(parallel_name, {})
    n_ver = n_row.get("verification", {})
    one_ver = one_row.get("verification", {})
    same_classes = n_ver.get("class_sha256") is not None and n_ver.get(
        "class_sha256"
    ) == one_ver.get("class_sha256")
    parallel = {
        "load": parallel_name,
        "repetitions": parallel_repeat,
        "numba1_median_s": one_row.get("median_s"),
        "numba_median_s": n_row.get("median_s"),
        "parallel_over_numba1": (
            round(one_row["median_s"] / n_row["median_s"], 3)
            if one_row.get("median_s") and n_row.get("median_s")
            else None
        ),
        "class_sha256_numba1": one_ver.get("class_sha256"),
        "class_sha256_numba": n_ver.get("class_sha256"),
        "classes_identical": same_classes,
        "class_counts_numba1": one_ver.get("class_counts"),
        "class_counts_numba": n_ver.get("class_counts"),
    }

    return {
        "machine": numba["machine"],
        "versions": numba["versions"],
        "numba": numba["numba"],
        "git": numba["git"],
        "threads": {
            "numba": numba["numba"]["threads"],
            "numba1": numba1["numba"]["threads"],
            "pure": pure["numba"]["threads"],
        },
        "protocol": {
            "repeat": repeat,
            "parallel_repeat": parallel_repeat,
            "warmup": "one call per workload in each process, excluded from timings",
            "pure_python_env": "NUMBA_DISABLE_JIT=1",
            "one_thread_env": "NUMBA_NUM_THREADS=1",
        },
        "modes": {"numba": numba, "numba1": numba1, "pure": pure},
        "speedup": speedup,
        "parallel": parallel,
    }


def _child(
    mode: str,
    repeat: int,
    parallel_repeat: int,
    only_parallel: bool,
    out: Path,
) -> int:
    payload = _run(mode, repeat, parallel_repeat, only_parallel)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def _both(repeat: int, parallel_repeat: int, out: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="pfc-bench-") as tmp:
        tmpdir = Path(tmp)
        numba_out = tmpdir / "numba.json"
        numba1_out = tmpdir / "numba1.json"
        pure_out = tmpdir / "pure.json"
        base_env = os.environ.copy()

        env = base_env.copy()
        env.pop("NUMBA_DISABLE_JIT", None)
        env.pop("NUMBA_NUM_THREADS", None)
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "numba",
                "--repeat",
                str(repeat),
                "--parallel-repeat",
                str(parallel_repeat),
                "--out",
                str(numba_out),
            ],
            check=True,
            env=env,
        )

        env = base_env.copy()
        env.pop("NUMBA_DISABLE_JIT", None)
        env["NUMBA_NUM_THREADS"] = "1"
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "numba1",
                "--repeat",
                str(repeat),
                "--parallel-repeat",
                str(parallel_repeat),
                "--out",
                str(numba1_out),
            ],
            check=True,
            env=env,
        )

        env = base_env.copy()
        env["NUMBA_DISABLE_JIT"] = "1"
        env.pop("NUMBA_NUM_THREADS", None)
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "pure",
                "--repeat",
                str(repeat),
                "--parallel-repeat",
                str(parallel_repeat),
                "--out",
                str(pure_out),
            ],
            check=True,
            env=env,
        )
        numba = json.loads(numba_out.read_text(encoding="utf-8"))
        numba1 = json.loads(numba1_out.read_text(encoding="utf-8"))
        pure = json.loads(pure_out.read_text(encoding="utf-8"))
    payload = _combine(numba, numba1, pure, repeat, parallel_repeat)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("both", "numba", "numba1", "pure"),
        default="both",
        help="Run both processes, or only one of them.",
    )
    parser.add_argument("--repeat", type=int, default=5, help="Timed repetitions per workload.")
    parser.add_argument(
        "--parallel-repeat",
        type=int,
        default=3,
        help="Timed repetitions for the 4,000-point parallelism load.",
    )
    parser.add_argument(
        "--only-parallel",
        action="store_true",
        help="Run only the 4,000-point load in one Numba mode.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/baseline/pfc_bench.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.parallel_repeat < 1:
        parser.error("--parallel-repeat must be >= 1")
    if args.only_parallel and args.mode == "both":
        parser.error("--only-parallel requires --mode numba or --mode numba1")
    if args.mode == "both":
        return _both(args.repeat, args.parallel_repeat, args.out)
    return _child(
        args.mode,
        args.repeat,
        args.parallel_repeat,
        args.only_parallel,
        args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
