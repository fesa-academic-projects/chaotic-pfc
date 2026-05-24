# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-05-23

### Added
- Property-based testing with Hypothesis: 11 invariants for Hénon maps, Lyapunov exponents,
  signals, area summaries, and Kaiser consolidation (`tests/test_properties.py`).
- Custom Hypothesis strategies in `tests/_hypothesis_strategies.py`.
- Hypothesis profiles (`dev` / `ci`) in `tests/conftest.py`.
- Tutorial notebook (`examples/tour.ipynb`, ~1 min) covering the full pipeline.
- `nbsphinx` for rendering notebooks in Sphinx docs.
- CI job (`notebook`) that executes `examples/tour.ipynb` on every push.
- Scientific validation tests against Wolf et al. (1985) and Sprott (2003) for Hénon Lyapunov
  exponents (`tests/test_scientific_validation.py`).
- Validation documentation (`docs/validation.rst`) with reference tables, DOI, and ISBN.
- PT-BR translation for validation page (`docs/locale/pt_BR/LC_MESSAGES/validation.po`).
- `area_summary()`, `lmax_statistics()`, `rank_configurations()`, `top_k_per_filter()`,
  `sweet_spot_per_filter()`, `consolidate_kaiser()`, `kaiser_beta_optimal()` in `stats.py`.
- `AreaSummary`, `LmaxStats`, `ConfigRank`, `SweetSpot`, `KaiserBetaOptimal` TypedDicts.
- LaTeX table export module (`latex_export.py`): 8 exporters covering top-k, extended top-k,
  full ranking, sweet spots, beta-optimal, and consolidated Kaiser variants.
- Bilingual i18n keys for analysis tables (23 keys, PT + EN).
- CLI subcommand `chaotic-pfc run analysis export-tables` with `--lang pt|en|all`.
- `docs/tutorial.rst` and `docs/testing.rst` documentation pages.

### Changed
- `CHANGELOG.md` fully conforms to Keep a Changelog 1.1.0 with reference links.
- `docs/conf.py`: `nbsphinx` extension and `nbsphinx_execute = "never"` added.
- `pyproject.toml`: `hypothesis` in `[dev]` extra, `nbsphinx` + `jupyter` in `[docs]` extra.
- CI test job now uses `--hypothesis-profile=ci`.
- `analysis_output/` output files moved to `data/analysis_output/tables/{pt,en}/`.
- `--lang` flag aligned with other subcommands (`pt`/`en` instead of `pt_BR`).
- LaTeX tabular tables now use `\resizebox{\textwidth}{!}`, `\footnotesize`, and
  `\setlength{\tabcolsep}{4pt}` for A4 fit.
- README (PT and EN) updated with tutorial mention and scientific validation mention.

### Removed
- DCSK and EF-DCSK from project scope. The library now exclusively implements
  Pecora-Carroll synchronisation with FIR-filtered Hénon map.
- References to joblib/loky parallelism in documentation — the codebase never
  used joblib; parallelism is via Numba ``prange`` only.

### Fixed
- `henon_fir_sequence` now uses the generalised Hénon form (a - xf² + b·y,
  y = x) for consistency with the rest of the codebase.
- `receive_order_n` validates ``len(fir_coeffs) >= 3``, matching the check
  in ``henon_order_n``.
- ``docs/background.rst``: Hénon equations harmonised to generalised form.
- ``docs/validation.rst``: notation updated from standard (a, b) to
  generalised (α, β), with equivalence noted.
- ``docs/source/algorithms/sweep.rst``: replaced joblib architecture
  description with the actual prange-based implementation.
- ``pyproject.toml`` description and keywords tightened to reflect
  Pecora-Carroll scope.
- README (both languages): added BibTeX Citing section.
- `load_all_sweeps()` Kaiser detection: switched from `kaiser_beta is not None` (always true —
  metadata defaults to 5.0) to `result.window == "kaiser"`.
- `top_k_per_filter()` now re-ranks entries per filter group (1, 2, 3) instead of preserving
  global rank numbers.
- Sphinx docstring warnings in `lyapunov.py` (`lyapunov_max_ensemble`,
  `lyapunov_henon2d_ensemble`) resolved.

## [0.7.0] - 2026-05-10

### Fixed
- FIR circular buffer off-by-one in `henon_fir_sequence`: `h[0]` was multiplying the
  oldest buffer value instead of the newest, scrambling coefficient order for all DCSK.
- Triple AWGN in `channel_urban`: each sub-call applied independent AWGN at the requested
  SNR, producing 3x the intended noise power. SNR is now compensated per sub-call.
- Wrong expected Lyapunov exponent sum: diagnostic printed `2*ln|beta|` instead of the
  correct `ln|beta| + 2*ln|r|` for the 4-D pole-filtered system.
- `summary_table` chaotic count silently undercounted when `h` contained NaN; now uses
  the same `~isnan` mask as periodic and divergent counts (symmetric with `beta_summary`).
- Four broken Sphinx cross-references: `chaotic_pfc.channel.*` routed to
  `chaotic_pfc.comms.channel.*`; `chaotic_pfc.spectral.*` to `chaotic_pfc.dynamics.spectral.*`.
- Missing `[Kolumban96]` and `[Kaddoum13]` bibliography entries in `background.rst`.
- `to_namespace()` missing `Nc` and `internal_cutoff` fields, breaking `run all` at
  step [05] with `AttributeError`.

### Added
- `DCSK_DEFAULT_WC` exposed at package top level and in `__all__`.
- i18n support for sweep classification legends via `_build_legend_handles(lang)` and
  new `_i18n` keys (`sweep.legend.periodic`, `.chaotic`, `.unbounded`).
- `FixedPointInfo` TypedDict for `fixed_point_stability` return type.
- Type hints (`NDArray`, `SpectralConfig`) on `compute_psds` in `cli/_common.py`.
- Regression test `test_to_namespace_has_all_required_fields` guarding against future
  `to_namespace` field omissions.
- Documentation of Pecora-Carroll vs DCSK BER comparison asymmetry in `cli/dcsk.py`.

### Changed
- `maps.py` module header updated from "Three 2-D maps" to "Four Henon map variants and
  one chaotic-sequence generator".
- `test_cli.EXPERIMENTS` extended with `dcsk` and `analysis` subcommands.
- Phantom `--save` and `--no-display` flags removed from `sweep/_beta.py`,
  `sweep/_plot.py`, and `sweep/_plot_3d.py` (they were no-ops).
- `transmit_order_n` and `receive_order_n` docstrings clarified: both explicitly state
  they do NOT implement the `Transmitter`/`Receiver` protocols.
- `to_namespace()` expanded with `seed` field; intentional omissions of `PlotConfig`,
  `SpectralConfig`, and `SweepConfig` documented.
- `run_all.py` references `DCSK_DEFAULT_WC` and `cfg.comm.mu` instead of hardcoded
  magic numbers; duplicate local `DEFAULT_CONFIG` import removed.
- `plotting/` restored to coverage measurement (was excluded despite `test_plotting.py`).
- `DEFAULT_CONFIG` singleton documented as read-only with `dataclasses.replace()` guidance.
- CI `--cov-fail-under` raised from 65 to 75.
- CI pipeline smoke test upgraded from Python 3.12 to 3.14.
- `RELEASING.md` updated to include `docs/conf.py` in version bump checklist.
- Makefile `.PHONY` completed with all targets (`format-check`, `docs-pt`, etc.).
- Test class inheritance standardized to `(_IsolatedCwdMixin, unittest.TestCase)` across
  all smoke test classes.
- `test_transmit_diverges_with_large_mu` moved from `TestTransmitOrderN` to
  `TestTransmitStandard`.
- Sphinx `.po` files regenerated via `sphinx-intl update`; 92 fuzzy entries resolved.
- Stale translations fixed: symbol count (62), `default_rng(seed)`, channel module paths.
- README `__all__` count corrected from `~65` to `62`.
- `docs/contributing.rst` symbol count corrected from `61` to `62`.

### Removed
- All em-dashes (travessões) from documentation: replaced with colons, commas, or
  semicolons. Applies to all `.rst`, `.md`, and source docstrings.

## [0.6.2] - 2026-05-10

### Fixed
- TestPyPI publish rejected: filename previously used and deleted. Patch bump only.

*Note: version 0.6.1 was a failed publish to TestPyPI (rejected filename) and was never released.*

## [0.6.0] - 2026-05-10

### Added
- `docs/background.rst` expanded with full theoretical foundations from the PFC article:
  physical-layer security, FIR filtering, Lyapunov exponents, Pecora-Carroll synchronisation,
  DCSK/EF-DCSK/CSK modulation, BER/SNR metrics, parameter dependence, and 14 references.
- `docs/internals.rst`: Numba kernel architecture, MGS, adaptive early-stop, FIR bank
  precomputation, fixed-point stability, signal generators, PSD estimation, channel models.
- `docs/development.rst`: environment setup, tooling, testing, CI/CD, documentation build
  workflow (including i18n gettext/update-po), and code style conventions.
- `docs/contributing.rst`: philosophy, contribution workflow, commit format, docstring
  conventions, public API rules, and pull request checklist.
- `docs/architecture.rst` expanded with full package tree, communication pipeline diagram,
  parameter sweep pipeline, 7 design decisions, and subpackage responsibility table.
- `docs/usage.rst` expanded with index, all CLI subcommands with examples, adaptive sweep
  options, and language support section.
- Bilingual translations (pt_BR) for all hand-written documentation pages: index,
  background, architecture, usage, internals, development, and contributing.
- `chaotic_pfc._i18n` module: bilingual figure labels (pt / en) with dictionary-based
  lookup, controllable via `CHAOTIC_PFC_LANG` env var or `--lang` CLI flag.
- `--lang pt|en` CLI flag for attractors, sensitivity, comm-*, dcsk, and run-all.
- `docs/_redirect.html`: RTD language landing page.

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
- API reference `.po` files removed: auto-generated docstrings stay in English.

### Fixed
- `run_all` not passing `lang` attribute to experiment subcommands (`AttributeError`).
- RST substitution warnings in docstrings (`|x|` → `\\|x\\|` in sweep_plotting, dcsk,
  run_all; title underline in spectral).
- LaTeX PDF build failures: switched to xelatex for Unicode characters in docstrings,
  replaced Unicode box-drawing in architecture diagram with plain ASCII.
- Placeholder `FIRST AUTHOR <EMAIL@ADDRESS>` in `.po` headers.
- Broken `spectral.py` entry in `api/index.po`.

## [0.5.0] - 2026-05-07

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
- `plotly` import is now lazy (`_get_go()`): package imports without plotly installed.
- `aggregate_beta_sweeps` / `plot_3d_beta_volume` removed from top-level `__init__`;
  import directly from `chaotic_pfc.analysis.sweep_plotting_3d`.
- `pyproject.toml`: classifiers and Documentation URL added.
- `analysis_summary.json` default path moved to `data/`.

### Fixed
- `PlotGridOptions.time_window` uses `default_factory` for Python 3.11 compatibility.
- Duplicate `if:` conditions in CI pipeline and docs jobs.
- CodeQL double-import warning in `test_cli_smoke.py`.

## [0.4.0] - 2026-05-04

### Added
- Comprehensive statistical sweep analysis suite: filter-type comparison, lambda_max
  distributions, transition boundaries, spectral robustness, Spearman correlation,
  bootstrap confidence intervals, parameter ranking, beta-evolution curves, and interpretation.
- DCSK and EF-DCSK chaotic communication modules with 4 channel models (AWGN, impulsive,
  multipath, urban interferers).
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
- `scripts/benchmark.py`: performance benchmarks for Henon maps, FIR bank, and Lyapunov exponents.
- `PlotGridOptions` dataclass as a typed alternative to `plot_comm_grid` keyword arguments.
- `ExperimentConfig.to_namespace()`: generates `argparse.Namespace` from config defaults,
  eliminates `_fill_config_defaults`.
- TypedDict definitions in `stats.py`: `SummaryRow`, `FilterTypeAggregate`, `OptimalParams`,
  `LmaxDistribution`, `CorrelationMatrix`, `BootstrapConfidence`.
- `tests/_test_helpers.py` with shared `make_fir_coeffs` and `assert_seed_determinism`.
- 18 new tests.

### Changed
- Sources reorganised into 4 subpackages: `dynamics/`, `comms/`, `analysis/`, `plotting/`.
- `analysis.py` renamed to `stats.py`; `plotting.py` renamed to `figures.py`.
- `sweep.py` (1139 lines) split into `_types`, `_kernel`, `_orchestration`, `_io` submodules.
- `test_maps.py` split into `test_maps.py`, `test_signals.py`, `test_lyapunov.py`.
- CLI output and `print()` statements translated to English (figure titles kept in
  Portuguese for the academic article).
- `henon_order_n`: `fir_coeffs` is now keyword-only (`*` marker in signature).
- DCSK `dcsk_transmit`/`efdcsk_transmit` share a `_chaos_sequence` helper.
- Adaptive Lyapunov early-stop block extracted into `_adaptive_checkpoint`.
- CLI `comm_ideal`, `comm_fir`, `comm_order_n` share `compute_psds` and `save_or_show`.
- `dcsk.py` CLI uses `add_save_display_flags` for consistency.
- `_save()` in `plotting/figures.py` now creates parent directories.
- `_coeffs()` removed from `test_transmitter` and `test_receiver`; replaced by shared `make_fir_coeffs`.
- `plotly` import in `sweep_plotting_3d.py` is now lazy (`_get_go()`).
- `aggregate_beta_sweeps` and `plot_3d_beta_volume` removed from top-level `__init__.py`.
- `coverage.run.omit` updated from dead `*/plotting.py` to `*/plotting/*`.
- CI test job now uses a Python version matrix `["3.10", "3.12"]`.
- Pre-commit config expanded.
- README updated with new package structure and a "Public API" section.
- `analysis_summary.json` output path moved to `data/` and added to `.gitignore`.
- `run_all.py` step 08 now plots all window×filter combinations.

### Fixed
- Duplicate test method names in `test_sweep_plotting.py`.
- `FILTER_TYPES` hardcoded 5 times in `cli/analysis.py`: now uses the constant from `analysis.sweep`.
- Wrong expected value in `TestHenonStandard.test_first_iteration`.
- Wrong fixed-point assertion in `TestLyapunov.test_henon2d_fixed_points`.
- Flaky `TestLyapunovEnsemble.test_chaotic_average`.
- Outdated `:mod:` references in `sweep_plotting.py`, `config.py`, `channel.py`, `receiver.py`.
- Outdated `Originally scripts/...` comments in 7 CLI modules.
- Outdated module paths in `docs/api/index.rst`.
- Build artifacts (`.egg-info/`, `__pycache__/`) removed from version control.
- Leftover `run_all.py` in project root removed.

## [0.3.0] - 2026-05-03

### Added
- Kaiser beta-sweep in the Lyapunov exponent pipeline with interactive 3-D Plotly plots.
- Adaptive early-stop for Lyapunov kernels: convergence check every 100 iterations.
- Lyapunov ensemble protocol: `N_ci` ICs sampled uniformly in +/-perturbation around the
  fixed point, with per-IC CSV export.
- Sweep plotting: `lambda_max == 0` classified as periodic, difficulty map for adaptive sweeps.
- Smoke tests for all CLI subcommands.
- Sphinx documentation with Furo theme, auto-generated API reference from NumPy docstrings.
- `.readthedocs.yaml` for Read the Docs hosting.
- `requirements-lock.txt` for byte-exact CI reproducibility.
- Codecov coverage upload in CI with matching badge.
- BSD 3-Clause LICENSE and AUTHORS file.

### Changed
- CLI unified: standalone scripts (01–08 + run_all.py) replaced by `chaotic-pfc run <subcommand>`.
- Codebase reformatted and linted with Ruff; type-checked with mypy.
- Pre-commit hooks added.
- NumPy-style docstrings adopted across the entire library.
- Sweep kernel refactored: in-place buffers, merged n1–n4 into two regimes (n12, nN),
  prange load balancing via round-robin task ordering.
- Sweep kernel made deterministic under `np.random.seed`.
- CI split into parallel jobs: lint, typecheck, test, pipeline smoke test, docs build.
- Sweep plots now generate both PNG and SVG by default.
- `henon_processar.py` extracted into `sweep.py` + `sweep_plotting.py` modules.
- Coverage rose from 56% to 62%; 8 modules at 100%.

### Fixed
- Build artifacts (`.egg-info/`, `__pycache__/`) removed from version control.
- Leftover `run_all.py` in project root removed.

## [0.2.0] - 2026-04-22

### Added
- Initial release: Henon map variants, FIR channel models, Pecora-Carroll synchronisation.
- Lyapunov exponent computation (single IC) and parameter sweep over `(order, cutoff)` grid.
- Basic CLI scripts (01–08) for each experiment step.

[Unreleased]: https://github.com/fesa-academic-projects/chaotic-pfc/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.8.0
[0.7.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.7.0
[0.6.2]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.6.2
[0.6.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.6.0
[0.5.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.5.0
[0.4.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.4.0
[0.3.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.3.0
[0.2.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.2.0
