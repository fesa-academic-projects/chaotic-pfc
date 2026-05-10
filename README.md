# chaotic-pfc

<p align="right">
  <strong><a href="./README_pt-BR.md">🇧🇷 Português (Brasil)</a></strong> |
  <strong>🇺🇸 English</strong>
</p>

[![CI](https://github.com/fesa-academic-projects/chaotic-pfc/actions/workflows/ci.yml/badge.svg)](https://github.com/fesa-academic-projects/chaotic-pfc/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/chaotic-pfc/badge/?version=latest)](https://chaotic-pfc.readthedocs.io/)
[![codecov](https://codecov.io/gh/fesa-academic-projects/chaotic-pfc/branch/main/graph/badge.svg)](https://codecov.io/gh/fesa-academic-projects/chaotic-pfc)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy: checked](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![CodeQL](https://github.com/fesa-academic-projects/chaotic-pfc/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/fesa-academic-projects/chaotic-pfc/actions/workflows/github-code-scanning/codeql)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

Chaotic digital communication system based on the Hénon map.
Undergraduate final project (*Trabalho de Conclusão de Curso*).

<p align="center">
  <img src="figures/sweeps/Hamming (lowpass)/fig2_classification_interleaved.svg" width="600" alt="Lyapunov classification map — Hamming lowpass">
</p>
<p align="center">
  <em>Lyapunov exponent classification across filter orders and cutoffs — periodic (blue), chaotic (red), unbounded (grey).</em>
</p>

## Overview

This repository contains a Python implementation of a chaos-based digital
communication system. The transmitter modulates a binary message onto the
state of a Hénon map; the receiver recovers the message via chaos
synchronisation. The pipeline also includes a full numerical study of the
Lyapunov exponents of the Hénon map and of higher-order variants with
internal FIR filters, including a 2-D parameter sweep over filter order and
cutoff frequency — classifying regions of the parameter space as chaotic,
periodic, or divergent (figure above).

The project is organised as an installable Python package (`chaotic_pfc`)
with numbered scripts that reproduce each experiment. The heavy Lyapunov
sweep is JIT-compiled with Numba and parallelised over the `(order, cutoff)`
grid.

## Installation

Requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,fast]"   # dev tools + Numba JIT acceleration
```

To install without Numba (slower but no binary dependencies):

```bash
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
chaotic-pfc run all --no-display
```

Run a single experiment:

```bash
chaotic-pfc run attractors --save
chaotic-pfc run lyapunov --save --n-ci 20
```

Run the Lyapunov sweep. The full sweep is expensive (typically tens of
minutes; up to several hours depending on hardware and thread count); the
quick mode runs a reduced grid in seconds to smoke-test the pipeline:

```bash
# Full sweep for one (window, filter) combination
chaotic-pfc run sweep compute --window hamming --filter lowpass

# Quick smoke test (~seconds)
chaotic-pfc run sweep compute --window hamming --filter lowpass --quick

# Generate the 4 classification figures (PNG + SVG) from the saved .npz
chaotic-pfc run sweep plot --window hamming --filter lowpass
```

The sweep checkpoints in `data/sweeps/*.npz` are versioned in the repository,
so plots can be regenerated at any time without rerunning the compute step.

Browse all available commands with `chaotic-pfc run --help` and each
subcommand's flags with `chaotic-pfc run <name> --help`.

## Project structure

```
chaotic-pfc/
├── pyproject.toml                 Package metadata and dependencies
├── src/chaotic_pfc/               Installable library
│   ├── dynamics/                  Henon maps, Lyapunov exponents, signals, spectral analysis
│   ├── comms/                     Transmitter, channel models, receiver, DCSK schemes
│   ├── analysis/                  Parameter sweeps, statistical post-processing, sweep plotting
│   │   └── sweep/                 Numba-JIT kernel, FIR precomputation, I/O
│   ├── plotting/                  Publication-quality figures (attractors, sensitivity, comms)
│   ├── cli/                       Unified CLI subcommand modules
│   └── config.py                  Centralised configuration
├── tests/                         Unit tests
├── data/
│   ├── lyapunov/                  CSV tables from the ensemble protocol
│   └── sweeps/                    Versioned .npz checkpoints from long sweeps
├── figures/                       Final figures (SVG for the paper, PNG for preview)
└── scripts/
    └── benchmark.py               Performance benchmarks for core operations
```

### Public API

The top-level `chaotic_pfc` namespace reexports ~65 symbols that form the
stable public API. They are importable from `chaotic_pfc` directly:

```python
from chaotic_pfc import run_sweep, henon_standard, fir_channel, LyapunovResult
```

Internal implementation details (private modules with underscore prefix, e.g.
`chaotic_pfc.analysis.sweep._kernel`) may change without notice and should
only be imported in tests or advanced research scripts.

Optional extras follow a separate import path. For example, 3-D visualisation
requires the `viz3d` extra (`pip install -e '.[viz3d]'`) and is imported
directly from its module:

```python
from chaotic_pfc.analysis.sweep_plotting_3d import plot_3d_beta_volume
```

## Experiments

| Subcommand | Description |
|------------|-------------|
| `chaotic-pfc run attractors`         | Phase-space portraits of the three Hénon variants. |
| `chaotic-pfc run sensitivity`        | Sensitivity to initial conditions (SDIC). |
| `chaotic-pfc run comm-ideal`         | Transmitter and receiver over an ideal channel. |
| `chaotic-pfc run comm-fir`           | End-to-end system with FIR channel. |
| `chaotic-pfc run comm-order-n`       | Order-N filtered map communication. |
| `chaotic-pfc run lyapunov`           | Lyapunov spectra: single IC and N-IC ensemble for 2-D and 4-D systems. |
| `chaotic-pfc run sweep compute`      | Parallel Lyapunov sweep over `(filter order × cutoff)`. |
| `chaotic-pfc run sweep plot`         | Classification maps from saved sweep data. |
| `chaotic-pfc run sweep beta-sweep`   | Kaiser β-sweep: aggregate per-β Lyapunov results. |
| `chaotic-pfc run sweep plot-3d`      | Interactive 3-D volume of Kaiser β-sweeps via Plotly. |
| `chaotic-pfc run dcsk`               | DCSK / EF-DCSK / Pecora-Carroll BER vs SNR comparison. |
| `chaotic-pfc run analysis`           | Statistical analysis of sweep results (summary, rankings, bootstrap CIs). |
| `chaotic-pfc run all`                | Full pipeline, in order. |

## Tests

Run the full suite:

```bash
pytest
```

Exclude slow tests (sweep compute) during development:

```bash
pytest -m "not slow"
```

Or use the Makefile:

```bash
make test          # full suite
make test-fast     # exclude slow tests
make check-all     # lint + format + typecheck + fast tests
```

## Makefile targets

| Target | Action |
|--------|--------|
| `make test` | Run full test suite |
| `make test-fast` | Run tests excluding `@pytest.mark.slow` |
| `make lint` | Ruff linter |
| `make format` | Ruff auto-format |
| `make format-check` | Check formatting without changing files |
| `make typecheck` | mypy static type checker |
| `make check-all` | lint + format-check + typecheck + test-fast |
| `make docs` | Build Sphinx HTML documentation (English) |
| `make docs-pt` | Build Sphinx HTML documentation (Portuguese) |
| `make docs-pdf` | Build Sphinx PDF documentation |
| `make docs-epub` | Build Sphinx EPUB documentation |
| `make benchmark` | Performance benchmarks |
| `make pre-commit` | Run all pre-commit hooks |
| `make clean` | Remove cache and build artifacts |
| `make help` | Show all targets |

## Documentation

Full documentation is hosted at
[chaotic-pfc.readthedocs.io](https://chaotic-pfc.readthedocs.io/)
in English and Portuguese (Brasil), with downloadable PDF, EPUB, and
HTMLZip formats available.

Build the HTML documentation locally:

```bash
pip install -e ".[docs]"
cd docs
make html
```

Then open `docs/_build/html/index.html` in a browser. Each module
auto-generates an API reference page from its NumPy-style docstrings.

## Development

Enable the pre-commit hooks so Ruff runs automatically on every
`git commit`:

```bash
pre-commit install
```

Run the full suite of hooks against every file (useful after
`pre-commit autoupdate` or before pushing):

```bash
pre-commit run --all-files
```

## License

Distributed under the terms of the BSD 3-Clause License. See
[LICENSE](LICENSE) for the full text.

## Authors

Developed by students of **Engenheiro Salvador Arena College (FESA)**.
See [AUTHORS](AUTHORS) for the complete contributors list.
