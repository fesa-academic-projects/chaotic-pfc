# Changelog

All notable changes to chaotic-pfc are documented in this file.

## [Unreleased]

## [0.6.0] — 2026-05-10

### Added
- `docs/background.rst` expanded with full theoretical foundations from the PFC article:
  physical-layer security, FIR filtering, Lyapunov exponents, Pecora-Carroll synchronisation,
  DCSK/EF-DCSK/CSK modulation, BER/SNR metrics, parameter dependence, and 14 references.
- `docs/internals.rst` — Numba kernel architecture, MGS, adaptive early-stop, FIR bank
  precomputation, fixed-point stability, signal generators, PSD estimation, channel models.
- `docs/development.rst` — environment setup, tooling, testing, CI/CD, documentation build
  workflow (including i18n gettext/update-po), and code style conventions.
- `docs/contributing.rst` — philosophy, contribution workflow, commit format, docstring
  conventions, public API rules, and pull request checklist.
- `docs/architecture.rst` expanded with full package tree, communication pipeline diagram,
  parameter sweep pipeline, 7 design decisions, and subpackage responsibility table.
- `docs/usage.rst` expanded with index, all CLI subcommands with examples, adaptive sweep
  options, and language support section.
- Bilingual translations (pt_BR) for all hand-written documentation pages — index,
  background, architecture, usage, internals, development, and contributing.
- `chaotic_pfc._i18n` module — bilingual figure labels (pt / en) with dictionary-based
  lookup, controllable via `CHAOTIC_PFC_LANG` env var or `--lang` CLI flag.
- `--lang pt|en` CLI flag for attractors, sensitivity, comm-*, dcsk, and run-all.
- `docs/_redirect.html` — RTD language landing page.

### Changed
- `docs/conf.py` refactored: full i18n/l10n, clean Furo sidebar, intersphinx, copybutton,
  xelatex for Unicode PDF support.
- `docs/Makefile` rewritten with `html`, `html-pt`, `html-all`, `pdf`, `pdf-pt`, `epub`,
  `epub-pt`, `gettext`, and `update-po` targets.
- Root `Makefile` updated with `docs-all`, `docs-pdf`, `docs-pdf-pt`, `docs-epub`,
  `docs-epub-pt` targets.
- `.readthedocs.yaml` simplified to default Sphinx builder with `htmlzip` format.
- `.gitignore`: added `_build/`, `_readthedocs/`, and `*.mo` patterns.
- `pyproject.toml` docs extra: removed unused `sphinxcontrib-bibtex`.
- `README.md` and `README_pt-BR.md` synchronised.
- API reference `.po` files removed — auto-generated docstrings stay in English.

### Fixed
- `run_all` not passing `lang` attribute to experiment subcommands (`AttributeError`).
- RST substitution warnings in docstrings (`|x|` → `\\|x\\|` in sweep_plotting, dcsk,
  run_all; title underline in spectral).
- LaTeX PDF build failures: switched to xelatex for Unicode characters in docstrings,
  replaced Unicode box-drawing in architecture diagram with plain ASCII.
- Placeholder `FIRST AUTHOR <EMAIL@ADDRESS>` in `.po` headers.
- Broken `spectral.py` entry in `api/index.po`.

## [0.5.0] — 2026-05-07

### Added
- Numba made optional via `_compat.py` fallback layer (`[fast]` extra, `pip install chaotic-pfc[fast]`).
- `Transmitter`, `Channel`, `Receiver` Protocols in `comms/protocols.py`.
- `PlotGridOptions` dataclass for `plot_comm_grid`.
- `.codecov.yml` with 5% threshold (warn -2%, fail -5%).
- `.github/dependabot.yml` with grouped weekly updates for pip and GitHub Actions.
- `.github/ISSUE_TEMPLATE/` (bug report and feature request).
- `RELEASING.md` with step-by-step release process.
- TestPyPI CD: `release` job in `ci.yml` using OIDC trusted publishing (triggered on `git tag v*`).
- Dual-language README: `README.md` (EN) + `README_pt-BR.md` with language switcher.
- Hero figure in README (Lyapunov classification map).
- `py.typed` marker (PEP 561).
- `chaotic_pfc.__version__` attribute.
- `CHANGELOG.md` following Keep a Changelog.
- `CONTRIBUTING.md` with development workflow and project conventions.
- `Makefile` with 13 targets.
- `strict_markers = true` in pytest config.
- Python 3.14 in CI test matrix.

### Changed
- Minimum Python bumped from 3.10 to 3.11.
- CI: sequential gate (`lint ∥ typecheck` → `test` → `pipeline` → `docs`). PRs gate at test only.
- CI: `--cov-fail-under=55` quality gate.
- `henon_order_n`: `fir_coeffs` is now keyword-only.
- DCSK transmit functions share `_chaos_sequence` helper.
- Adaptive Lyapunov early-stop extracted into `_adaptive_checkpoint`.
- CLI `comm_*` modules share `compute_psds` and `save_or_show`.
- `plotly` import is now lazy (`_get_go()`) — package imports without plotly installed.
- `aggregate_beta_sweeps` / `plot_3d_beta_volume` removed from top-level `__init__`; import directly from `chaotic_pfc.analysis.sweep_plotting_3d`.
- `pyproject.toml`: classifiers and Documentation URL added.
- `analysis_summary.json` default path moved to `data/`.

### Fixed
- `PlotGridOptions.time_window` uses `default_factory` for Python 3.11 compatibility.
- Duplicate `if:` conditions in CI pipeline and docs jobs.
- CodeQL double-import warning in `test_cli_smoke.py`.

## [0.4.0] — 2026-05-04

### Package structure
- Sources reorganised into 4 subpackages: `dynamics/`, `comms/`, `analysis/`, `plotting/`.
- `analysis.py` renamed to `stats.py`; `plotting.py` renamed to `figures.py`.
- `sweep.py` (1139 lines) split into `_types`, `_kernel`, `_orchestration`, `_io` submodules.
- `test_maps.py` split into `test_maps.py`, `test_signals.py`, `test_lyapunov.py`.

### Added
- Comprehensive statistical sweep analysis suite: filter-type comparison, lambda_max distributions, transition boundaries, spectral robustness, Spearman correlation, bootstrap confidence intervals, parameter ranking, beta-evolution curves, and interpretation.
- DCSK and EF-DCSK chaotic communication modules with 4 channel models (AWGN, impulsive, multipath, urban interferers).
- Bandpass and bandstop filter types with bandwidth parameter; `FILTER_TYPES` expanded to 4.
- `chaotic-pfc run analysis` CLI subcommand.
- `chaotic-pfc run dcsk` CLI subcommand for BER-vs-SNR comparison.
- `chaotic-pfc run sweep beta-sweep` for Kaiser beta sweeps.
- `chaotic-pfc run sweep plot-3d` for Plotly 3-D visualisation.
- `chaotic-pfc run sweep compute --bandwidth` flag.
- `chaotic-pfc run all --adaptive` with `--Nmap-min` and `--tol`.
- `py.typed` marker (PEP 561).
- `chaotic_pfc.__version__`.
- `CHANGELOG.md` following Keep a Changelog.
- `CONTRIBUTING.md`.
- `Makefile` with 13 targets.
- `strict_markers = true` in pytest config.
- `scripts/benchmark.py` — performance benchmarks for Henon maps, FIR bank, and Lyapunov exponents.
- `PlotGridOptions` dataclass as a typed alternative to `plot_comm_grid` keyword arguments.
- `ExperimentConfig.to_namespace()` — generates `argparse.Namespace` from config defaults, eliminates `_fill_config_defaults`.
- `TypedDict` definitions in `stats.py`: `SummaryRow`, `FilterTypeAggregate`, `OptimalParams`, `LmaxDistribution`, `CorrelationMatrix`, `BootstrapConfidence`.
- `tests/_test_helpers.py` with shared `make_fir_coeffs` and `assert_seed_determinism`.
- 18 new tests: kernel functions (4), DCSK channel custom parameters (4), `henon_fir_sequence` edge cases (2), `fir_channel` kaiser window (1), `transmit_order_n`/`receive_order_n` `seed=None` (2), `lyapunov_max_ensemble` CSV 4-D (1), `stats.py` distribution/boundary/correlation/bootstrap (9, partially replacing implicit coverage), `sweep.py` helpers/edge cases (9).

### Changed
- CLI output and `print()` statements translated to English (figure titles kept in Portuguese for the academic article).
- `henon_order_n` — `fir_coeffs` is now keyword-only (`*` marker in signature).
- DCSK `dcsk_transmit`/`efdcsk_transmit` share a `_chaos_sequence` helper, reducing ~20 duplicated lines.
- Adaptive Lyapunov early-stop block extracted into `_adaptive_checkpoint` (shared by n12 and nN kernels), removing 30 duplicated lines per kernel.
- CLI `comm_ideal`, `comm_fir`, `comm_order_n` share `compute_psds` and `save_or_show` via `_common`.
- `dcsk.py` CLI uses `add_save_display_flags` for consistency.
- `_save()` in `plotting/figures.py` now creates parent directories (like `sweep_plotting` already did).
- `_coeffs()` removed from `test_transmitter` and `test_receiver`; replaced by shared `make_fir_coeffs`.
- `plotly` import in `sweep_plotting_3d.py` is now lazy (`_get_go()`) — importing the package no longer crashes without plotly installed.
- `aggregate_beta_sweeps` and `plot_3d_beta_volume` removed from the top-level `__init__.py` and `analysis/__init__.py`; import directly from `chaotic_pfc.analysis.sweep_plotting_3d`.
- `coverage.run.omit` updated from dead `*/plotting.py` to `*/plotting/*`.
- CI test job now uses a Python version matrix `["3.10", "3.12"]`.
- Pre-commit config expanded with `check-ast`, `check-json`, `check-case-conflict`, `debug-statements`, `mixed-line-ending`, `detect-private-key`, and `mypy`.
- README updated with new package structure and a "Public API" section.
- `analysis_summary.json` output path moved from project root to `data/` and added to `.gitignore`.
- `run_all.py` step 08 now plots all window×filter combinations (`--all`) instead of only hamming/lowpass.

### Fixed
- Duplicate test method names in `test_sweep_plotting.py` (3 methods copy-pasted into the wrong class).
- `FILTER_TYPES` hardcoded 5 times in `cli/analysis.py` — now uses the constant from `analysis.sweep`.
- Wrong expected value in `TestHenonStandard.test_first_iteration` (1.2 → 1.0).
- Wrong fixed-point assertion in `TestLyapunov.test_henon2d_fixed_points`.
- Flaky `TestLyapunovEnsemble.test_chaotic_average` (too few iterations; now uses `pole_radius=0.0`).
- Outdated `:mod:` references in `sweep_plotting.py`, `config.py`, `channel.py`, `receiver.py`.
- Outdated `Originally scripts/...` comments in 7 CLI modules.
- Outdated module paths in `docs/api/index.rst`.

## [0.3.0] — 2026-05-03

### Added
- Kaiser beta-sweep in the Lyapunov exponent pipeline with interactive 3-D Plotly plots.
- Adaptive early-stop for Lyapunov kernels: convergence check every 100 iterations, exits early when the running lambda_max estimate stabilises within tolerance.
- Lyapunov ensemble protocol: `N_ci` ICs sampled uniformly in +/-perturbation around the fixed point, with per-IC CSV export.
- Sweep plotting: `lambda_max == 0` classified as periodic (not NaN), difficulty map for adaptive sweeps.
- Smoke tests for all CLI subcommands, with `CHAOTIC_PFC_SKIP_SLOW=1` env var for local development.
- Sphinx documentation with Furo theme, auto-generated API reference from NumPy docstrings.
- `.readthedocs.yaml` for Read the Docs hosting.
- `requirements-lock.txt` for byte-exact CI reproducibility.
- Codecov coverage upload in CI with matching badge.
- BSD 3-Clause LICENSE and AUTHORS file.

### Changed
- CLI unified: standalone scripts (01–08 + run_all.py) replaced by `chaotic-pfc run <subcommand>` with argparse.
- Codebase reformatted and linted with Ruff; type-checked with mypy.
- Pre-commit hooks added for trailing whitespace, end-of-file, YAML/TOML validation, merge-conflict checks, and Ruff lint+format.
- NumPy-style docstrings adopted across the entire library (Parameters/Returns/Notes on every public function).
- Sweep kernel refactored: in-place buffers, merged n1–n4 kernels into two regimes (n12, nN), prange load balancing via round-robin task ordering.
- Sweep kernel made deterministic under `np.random.seed` (perturbations pre-generated on the Python side).
- CI split into parallel jobs: lint, typecheck, test, pipeline smoke test, docs build.
- Sweep plots now generate both PNG and SVG by default.
- `henon_processar.py` extracted into `sweep.py` + `sweep_plotting.py` modules.
- Coverage rose from 56% to 62%; 8 modules at 100%.

### Fixed
- Build artifacts (`.egg-info/`, `__pycache__/`) removed from version control.
- Leftover `run_all.py` in project root removed (superseded by CLI).

## [0.2.0] — 2026-04-22

- Initial release: Henon map variants, FIR channel models, Pecora-Carroll synchronisation.
- Lyapunov exponent computation (single IC) and parameter sweep over `(order, cutoff)` grid.
- Basic CLI scripts (01–08) for each experiment step.
