.PHONY: help install test test-fast lint format typecheck check-all docs clean pre-commit

help:  ## Show this help
	@grep -E '^[a-zA-Z_.-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev dependencies
	pip install -e ".[dev]"

test:  ## Run the full test suite
	pytest

test-fast:  ## Run tests excluding slow ones
	pytest -m "not slow"

lint:  ## Run Ruff linter
	ruff check .

format:  ## Run Ruff auto-format
	ruff format .

format-check:  ## Check formatting without changing files
	ruff format --check .

typecheck:  ## Run mypy static type checker
	mypy src/chaotic_pfc

check-all: lint format-check typecheck test-fast  ## Run all quality checks (lint + format + typecheck + fast tests)

SPHINXBUILD = .venv/bin/sphinx-build

docs:  ## Build Sphinx HTML documentation (English)
	cd docs && ../$(SPHINXBUILD) -b html . _build/html -D language=en -W --keep-going

docs-pt:  ## Build Sphinx HTML documentation (Portuguese)
	cd docs && ../$(SPHINXBUILD) -b html . _build/html/pt_BR -D language=pt_BR -W --keep-going

docs-all: docs docs-pt  ## Build Sphinx HTML documentation (both languages)

docs-pdf:  ## Build Sphinx PDF documentation (English)
	cd docs && ../$(SPHINXBUILD) -b latex . _build/latex -D language=en

docs-pdf-pt:  ## Build Sphinx PDF documentation (Portuguese)
	cd docs && ../$(SPHINXBUILD) -b latex . _build/latex -D language=pt_BR

docs-epub:  ## Build Sphinx EPUB documentation (English)
	cd docs && ../$(SPHINXBUILD) -b epub . _build/epub -D language=en

docs-epub-pt:  ## Build Sphinx EPUB documentation (Portuguese)
	cd docs && ../$(SPHINXBUILD) -b epub . _build/epub -D language=pt_BR

benchmark:  ## Run performance benchmarks
	python scripts/benchmark.py

clean:  ## Remove build and cache artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist *.egg-info
	$(MAKE) -C docs clean 2>/dev/null || true

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files
