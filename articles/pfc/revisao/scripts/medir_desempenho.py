#!/usr/bin/env python3
"""Medições de desempenho da Seção 4.6 do PFC (rodada de 01/11).

Quatro otimizações, cada uma medida com cinco repetições depois de um
aquecimento do JIT, com a mediana e os extremos guardados:

1. compilação do núcleo: carga de 30 pontos, interpretada contra compilada em
   uma thread (valores do benchmark versionado, mesma máquina e protocolo);
2. paralelismo: varredura de 4.000 pontos sem parada adaptativa, 1 thread
   contra 8 threads (e também 2 e 4 threads, para o texto);
3. parada adaptativa: a mesma varredura de 4.000 pontos com 8 threads, sem e
   com a parada;
4. vetor único contra espectro completo: sistema filtrado com Ns = 41, 3.000
   iterações, sem parada, por ponto.

A varredura de 4.000 pontos com 8 threads e sem parada é uma medida só: ela
serve de referência nas linhas do paralelismo e da parada adaptativa.

Uso::

    python articles/pfc/revisao/scripts/medir_desempenho.py

A saída vai para ``saida/desempenho.json``.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numba
import numpy as np
from scipy.signal import firwin

from chaotic_pfc._formulas import fixed_point_positive
from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD, run_sweep
from chaotic_pfc.analysis.sweep._kernel import (
    _burn_in_nN,
    _lyap_online_core,
    _lyap_online_core_v,
)

OUT = Path(__file__).resolve().parent / "saida"
BENCH = Path(__file__).resolve().parents[4] / "benchmarks/baseline/pfc_bench_i7_8550u.json"
ALPHA, BETA = 1.4, 0.3
REPEATS = 5


def sweep(*, adaptive: bool = False):
    """Varredura de 4.000 pontos, passa-baixas Hamming, semente fixa."""
    return run_sweep(
        window="hamming",
        filter_type="lowpass",
        adaptive=adaptive,
        Nitera=500,
        Nmap=3000,
        n_initial=25,
        seed=42,
        fallback_mode=0,
    )


def timed(threads: int, adaptive: bool, repeats: int = REPEATS) -> dict[str, object]:
    """Roda a varredura ``repeats`` vezes e devolve mediana, mínimo e máximo."""
    numba.set_num_threads(threads)
    sweep(adaptive=adaptive)  # aquecimento do JIT neste caminho de threads
    times = []
    classes = None
    same = True
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = sweep(adaptive=adaptive)
        times.append(time.perf_counter() - t0)
        h = np.asarray(result.h, dtype=np.float64)
        c = (h > LAMBDA_CHAOS_THRESHOLD) & np.isfinite(h)
        if classes is None:
            classes = c
        elif not np.array_equal(c, classes):
            same = False
    return {
        "threads": threads,
        "adaptive": adaptive,
        "repetitions": repeats,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "classes_iguais": same,
        "runs_s": times,
    }


def compilation_row() -> dict[str, float]:
    """Tempo interpretado e compilado na carga de 30 pontos (benchmark)."""
    data = json.loads(BENCH.read_text())
    pure = data["modes"]["pure"]["benchmarks"]["mini_sweep_30_points"]["median_s"]
    jit = data["modes"]["numba1"]["benchmarks"]["mini_sweep_30_points"]["median_s"]
    return {
        "carga_pontos": 30,
        "interpretado_s": pure,
        "compilado_s": jit,
        "ganho": pure / jit,
        "repetições": data["protocol"]["repeat"],
    }


def vector_vs_spectrum() -> dict[str, float]:
    """Tempo por ponto dos dois estimadores no sistema filtrado com Ns = 41."""
    ns = 41
    coeffs = firwin(ns, 0.05, window="hamming", pass_zero=True, fs=2.0)
    gain = float(coeffs.sum())
    p1 = float(fixed_point_positive(gain, ALPHA, BETA))
    rng = np.random.default_rng(7)

    def state() -> np.ndarray:
        xx = np.zeros((2, ns))
        xx[0, 0] = p1
        xx[0, 1] = p1
        xx[0, 2] = gain * p1
        xx[0, 3:] = p1
        _burn_in_nN(xx, ns, 500, ALPHA, BETA, coeffs)
        xx[0, :] += 1e-4 * rng.standard_normal(ns)
        return xx

    points = [state() for _ in range(8)]
    vv = np.empty((2, ns))
    lyap = np.empty(1)
    ww = np.empty((2, ns, ns))
    lyap_full = np.empty(ns)

    vec, spec = [], []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for point in points:
            _lyap_online_core_v(
                point.copy(), vv, lyap, 3000, 10000, 1e-3, ns, 1, coeffs, ALPHA, BETA
            )
        vec.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for point in points:
            _lyap_online_core(
                point.copy(), ww, lyap_full, 3000, 10000, 1e-3, ns, 1, coeffs, ALPHA, BETA, 0
            )
        spec.append(time.perf_counter() - t0)

    per_vec = statistics.median(vec) / len(points)
    per_spec = statistics.median(spec) / len(points)
    return {
        "pontos": len(points),
        "Ns": ns,
        "Nitera": 3000,
        "vetor_ms": 1e3 * per_vec,
        "espectro_ms": 1e3 * per_spec,
        "ganho": per_spec / per_vec,
        "repetições": REPEATS,
        "vetor_min_ms": 1e3 * min(vec) / len(points),
        "vetor_max_ms": 1e3 * max(vec) / len(points),
        "espectro_min_ms": 1e3 * min(spec) / len(points),
        "espectro_max_ms": 1e3 * max(spec) / len(points),
    }


def warm_up() -> None:
    """Compila o kernel antes de medir, para não contar a compilação."""
    run_sweep(
        window="hamming",
        filter_type="lowpass",
        num_taps_grid=[3, 4],
        cutoff_grid=np.array([0.3, 0.5]),
        Nitera=10,
        Nmap=50,
        n_initial=2,
        seed=42,
        fallback_mode=0,
    )


def main() -> int:
    """Roda as medições e grava o JSON."""
    OUT.mkdir(parents=True, exist_ok=True)
    warm_up()

    threads = {n: timed(n, adaptive=False) for n in (1, 2, 4, 8)}
    adaptive_on = timed(8, adaptive=True)
    data = {
        "compilacao": compilation_row(),
        "threads": threads,
        "adaptativa": {
            "sem": threads[8],
            "com": adaptive_on,
            "ganho": threads[8]["median_s"] / adaptive_on["median_s"],
        },
        "vetor_vs_espectro": vector_vs_spectrum(),
    }
    (OUT / "desempenho.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    c = data["compilacao"]
    print(
        f"compilação: {c['interpretado_s']:.3f} s -> {c['compilado_s']:.4f} s (x {c['ganho']:.1f})"
    )
    for n in (1, 2, 4, 8):
        r = threads[n]
        print(
            f"threads {n}: mediana {r['median_s']:.2f} s "
            f"[{r['min_s']:.2f}, {r['max_s']:.2f}], classes iguais {r['classes_iguais']}"
        )
    a = data["adaptativa"]
    print(
        f"adaptativa: {a['sem']['median_s']:.2f} s -> {a['com']['median_s']:.2f} s "
        f"(x {a['ganho']:.1f})"
    )
    v = data["vetor_vs_espectro"]
    print(
        f"vetor x espectro: {v['espectro_ms']:.2f} ms -> {v['vetor_ms']:.2f} ms "
        f"(x {v['ganho']:.1f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
