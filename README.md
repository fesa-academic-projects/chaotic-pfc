# chaotic-pfc

[![CI](https://github.com/fesa-academic-projects/chaotic-pfc/actions/workflows/ci.yml/badge.svg)](https://github.com/fesa-academic-projects/chaotic-pfc/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fesa-academic-projects/chaotic-pfc/branch/main/graph/badge.svg)](https://codecov.io/gh/fesa-academic-projects/chaotic-pfc)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy: checked](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

Chaotic digital communication system based on the Hénon map.
Undergraduate final project (*Trabalho de Conclusão de Curso*).

## Overview

This repository contains a Python implementation of a chaos-based digital
communication system. The transmitter modulates a binary message onto the
state of a Hénon map; the receiver recovers the message via chaos
synchronisation. The pipeline also includes a full numerical study of the
Lyapunov exponents of the Hénon map and of higher-order variants with
internal FIR filters, including a 2-D parameter sweep over filter order and
cutoff frequency.

The project is organised as an installable Python package (`chaotic_pfc`)
with numbered scripts that reproduce each experiment. The heavy Lyapunov
sweep is JIT-compiled with Numba and parallelised over the `(order, cutoff)`
grid.

## Installation

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For a byte-exact reproduction of the environment tested in CI (useful
when bisecting a regression), install from the lock file instead:

```bash
pip install -r requirements-lock.txt
```

## Usage

Run the full pipeline, saving every figure headlessly:

```bash
python run_all.py --no-display
```

Run a single experiment:

```bash
python scripts/01_henon_attractors.py --save
python scripts/06_lyapunov.py --save
```

Run the Lyapunov sweep. The full sweep is expensive (typically tens of
minutes; up to several hours depending on hardware and thread count); the
quick mode runs a reduced grid in seconds to smoke-test the pipeline:

```bash
# Full sweep for one (window, filter) combination
python scripts/07_henon_sweep_compute.py --window hamming --filter lowpass

# Quick smoke test (~seconds)
python scripts/07_henon_sweep_compute.py --window hamming --filter lowpass --quick

# Generate the 4 classification figures (PNG + SVG) from the saved .npz
python scripts/08_henon_sweep_plot.py --window hamming --filter lowpass
```

The sweep checkpoints in `data/sweeps/*.npz` are versioned in the repository,
so plots can be regenerated at any time without rerunning the compute step.

## Project structure

```
chaotic-pfc/
├── pyproject.toml                 Package metadata and dependencies
├── run_all.py                     Batch runner for all experiment scripts
├── src/chaotic_pfc/               Installable library
│   ├── maps.py                    Hénon map variants (2-D, generalised, filtered, N-th order)
│   ├── signals.py                 Binary message generator
│   ├── transmitter.py             Chaos-based modulator
│   ├── channel.py                 Ideal and FIR channel models
│   ├── receiver.py                Chaos-synchronisation demodulator
│   ├── spectral.py                Welch PSD estimation
│   ├── lyapunov.py                Lyapunov exponents (single IC and ensemble)
│   ├── sweep.py                   Parallel (order × cutoff) Lyapunov sweep
│   ├── plotting.py                Publication-quality figures for experiments 01–05
│   ├── sweep_plotting.py          Classification maps for the Lyapunov sweep
│   └── config.py                  Centralised configuration
├── scripts/                       Thin CLI wrappers around the library
├── tests/                         Unit tests (52 tests)
├── data/
│   ├── lyapunov/                  CSV tables from the ensemble protocol
│   └── sweeps/                    Versioned .npz checkpoints from long sweeps
└── figures/                       Final figures (SVG for the paper, PNG for preview)
```

## Experiments

| Script | Description |
|--------|-------------|
| `01_henon_attractors.py`    | Phase-space portraits of the three Hénon variants. |
| `02_sensitivity.py`         | Sensitivity to initial conditions (SDIC). |
| `03_comm_ideal_channel.py`  | Transmitter and receiver over an ideal channel. |
| `04_comm_fir_channel.py`    | End-to-end system with FIR channel. |
| `05_comm_order_n.py`        | Order-N filtered map communication. |
| `06_lyapunov.py`            | Lyapunov spectra: single IC and N-IC ensemble for 2-D and 4-D systems. |
| `07_henon_sweep_compute.py` | Parallel Lyapunov sweep over `(filter order × cutoff)`. |
| `08_henon_sweep_plot.py`    | Classification maps from the saved sweep data. |

## Tests

```bash
pytest
```

The first run compiles the Numba kernels and takes about two minutes;
subsequent runs use the cache and finish in under 30 seconds.

## License

Distributed under the terms of the BSD 3-Clause License. See
[LICENSE](LICENSE) for the full text.

## Authors

Developed by students of **Engenheiro Salvador Arena College (FESA)**.
See [AUTHORS](AUTHORS) for the complete contributors list.
