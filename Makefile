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

docs:  ## Build Sphinx HTML documentation
	$(MAKE) -C docs html

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
