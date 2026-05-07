.. _development:

Development
===========

This page covers the development environment setup, tooling, and
workflow for contributing to ``chaotic-pfc``.

Environment setup
-----------------

Requirements:

* **Python 3.11+** (3.11, 3.12, 3.13 tested in CI).
* **Git** for version control.
* Optional: **Numba** 0.59+ for JIT acceleration (via ``[fast]`` extra).

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/fesa-academic-projects/chaotic-pfc.git
    cd chaotic-pfc

    # Create a virtual environment
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate   # Windows

    # Install in editable mode with development tools
    pip install -e ".[dev]"              # pytest, ruff, mypy, pre-commit
    pip install -e ".[dev,fast]"         # + Numba JIT acceleration
    pip install -e ".[dev,fast,viz3d]"   # + 3-D Plotly visualisation

    # Install pre-commit hooks (runs linting before each commit)
    pre-commit install

Project structure
-----------------

.. code-block:: text

    chaotic_pfc/
    ├── src/chaotic_pfc/          # package source code
    ├── tests/                    # test suite (mirrors src/ structure)
    ├── docs/                     # Sphinx documentation (English + pt_BR)
    ├── scripts/                  # utility scripts (benchmark.py)
    ├── data/                     # Lyapunov output CSVs, sweep .npz checkpoints
    ├── figures/                  # generated SVG/PNG figures
    ├── .github/                  # CI workflows + issue templates
    ├── pyproject.toml            # package metadata + tool configuration
    ├── Makefile                  # convenience targets (test, lint, docs, …)
    └── requirements-lock.txt     # byte-exact dependency lock for CI

Tooling
-------

.. list-table:: Development tools
   :header-rows: 1
   :widths: 20 40 40

   * - Tool
     - Purpose
     - Command
   * - **pytest**
     - Test runner
     - ``make test`` or ``pytest``
   * - **ruff**
     - Linting + formatting
     - ``make lint`` / ``make format``
   * - **mypy**
     - Static type checking
     - ``make typecheck``
   * - **pre-commit**
     - Git hook automation
     - ``pre-commit run --all-files``
   * - **coverage**
     - Test coverage reports
     - ``pytest --cov=chaotic_pfc``

.. code-block:: bash

    # Run the full quality suite
    make check-all          # lint + format-check + typecheck + test-fast

    # Individual commands
    make test               # full test suite
    make test-fast          # exclude slow tests (marked @pytest.mark.slow)
    make lint               # ruff check
    make format             # ruff format
    make format-check       # ruff format --check (CI mode)
    make typecheck          # mypy src/chaotic_pfc
    make benchmark          # run performance benchmarks

Testing
-------

Tests live under ``tests/`` and mirror the source structure. Every
source module has a corresponding test file.

.. code-block:: bash

    # Full suite
    make test

    # Fast subset (excludes parameter sweeps)
    make test-fast

    # With coverage
    pytest --cov=chaotic_pfc --cov-report=term-missing

    # Run specific test file
    pytest tests/test_maps.py -v

    # Run only smoke tests
    pytest -m "not slow" -v

**Test categories:**

* **Unit tests** — verify individual functions with known inputs/outputs.
* **Integration tests** — test the communication pipeline end-to-end.
* **Smoke tests** — run the full CLI with ``--quick-sweep``.
* **Slow tests** — marked with ``@pytest.mark.slow``, run full-resolution
  sweeps. Skipped by ``make test-fast``.

Numba-compiled kernels are tested directly (not mocked) to ensure
bit-exact equivalence between JIT and pure-Python fallback paths.

CI/CD
-----

GitHub Actions runs on every push and pull request:

1. **Lint** — ``ruff check`` on the entire codebase.
2. **Format check** — ``ruff format --check``.
3. **Type check** — ``mypy`` on ``src/chaotic_pfc``.
4. **Tests** — ``pytest`` (full suite) with ``--cov``.
5. **Pipeline smoke test** — ``chaotic-pfc run all --no-display --quick-sweep``.
6. **Codecov upload** — coverage report to Codecov.

On ``git tag v*`` push, an additional **release** job builds and
publishes the wheel and sdist to TestPyPI via OIDC trusted publishing.

Documentation
-------------

The documentation is built with Sphinx and hosted on Read the Docs.
Output goes to ``docs/_build/html/`` with each language in its own
subdirectory.

.. code-block:: bash

    # Build English HTML          → _build/html/en/
    make docs

    # Build Portuguese HTML       → _build/html/pt_BR/
    make docs-pt

    # Build both languages at once
    make docs-all

    # Build English PDF
    make docs-pdf

    # Build Portuguese PDF
    make docs-pdf-pt

    # Build English EPUB
    make docs-epub

    # Build Portuguese EPUB
    make docs-epub-pt

    # Open in browser
    firefox docs/_build/html/en/index.html
    firefox docs/_build/html/pt_BR/index.html

To update translations after changing English sources:

.. code-block:: bash

    cd docs
    make gettext                          # extract .pot templates
    make update-po                        # merge into pt_BR .po files
    # edit docs/locale/pt_BR/LC_MESSAGES/*.po as needed

Documentation sources are in ``docs/`` as reStructuredText (``.rst``).
Translations (Brazilian Portuguese) are in
``docs/locale/pt_BR/LC_MESSAGES/`` as Gettext ``.po`` files.
API reference pages are **not** translated — they stay in English
because they are auto-generated from Python docstrings.

Code style
----------

The project follows these conventions:

.. list-table::
   :header-rows: 1

   * - Aspect
     - Convention
   * - Line length
     - 100 characters (ruff formatter)
   * - Quotes
     - Double quotes (ruff formatter default)
   * - Docstrings
     - NumPy-style, consumed by Sphinx with napoleon
   * - Typing
     - Full type annotations (enforced by mypy)
   * - Commit messages
     - Conventional Commits (``feat``, ``fix``, ``docs``, ``test``, ``chore``, ``ci``, ``refactor``)
   * - Language
     - Code and docstrings in English; figure labels bilingual (pt / en)

.. code-block:: bash

    # Run all formatters
    make format

    # Verify formatting (CI mode)
    make format-check

    # Run all quality checks
    make check-all
