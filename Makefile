.PHONY: help install test test-fast lint format format-check typecheck check-all check-paper jcis-figures paper-pdf paper-diff docs docs-strict docs-pt docs-all docs-pdf docs-pdf-pt docs-epub docs-epub-pt benchmark clean pre-commit check-venv

# The test and tooling targets run inside the project virtualenv, where the
# package and its dev dependencies (pytest, ruff, mypy, numba, …) live.
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python

check-venv:  ## Fail with a clear message when the project virtualenv is missing
	@test -x "$(PYTHON)" || { \
		echo "error: $(PYTHON) not found."; \
		echo "create the project virtualenv with:"; \
		echo "    python -m venv $(VENV) && $(VENV)/bin/pip install -e '.[dev,fast]'"; \
		exit 1; \
	}

help:  ## Show this help
	@grep -E '^[a-zA-Z_.-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev dependencies
	pip install -e ".[dev]"

test: check-venv  ## Run the full test suite
	$(PYTHON) -m pytest

test-fast: check-venv  ## Run tests excluding slow ones
	$(PYTHON) -m pytest -m "not slow"

lint: check-venv  ## Run Ruff linter
	$(PYTHON) -m ruff check .

format: check-venv  ## Run Ruff auto-format
	$(PYTHON) -m ruff format .

format-check: check-venv  ## Check formatting without changing files
	$(PYTHON) -m ruff format --check .

typecheck: check-venv  ## Run mypy static type checker
	$(PYTHON) -m mypy src/chaotic_pfc

check-all: lint format-check typecheck test-fast  ## Run all quality checks (lint + format + typecheck + fast tests)

PAPER_DIR ?= jcis
PAPER_TEX ?= $(PAPER_DIR)/article.tex
PAPER_BIB ?= $(PAPER_DIR)/references.bib
PAPER_STEM ?= article

check-paper: check-venv  ## Audit the JCIS manuscript sources against the repository
	$(PYTHON) tools/audit_paper.py --tex $(PAPER_TEX) --bib $(PAPER_BIB) --figures figures/

jcis-figures: check-venv  ## Regenerate the JCIS figures not produced by other commands
	$(PYTHON) tools/jcis_figures.py --out-dir $(PAPER_DIR)/Figures

paper-pdf: export SOURCE_DATE_EPOCH = $(shell git log -1 --format=%ct -- $(PAPER_TEX) $(PAPER_BIB) $(PAPER_DIR)/Figures)
paper-pdf: export FORCE_SOURCE_DATE = 1
paper-pdf:  ## Build the JCIS manuscript PDF with a clean SVG conversion cache
	rm -rf $(PAPER_DIR)/svg-inkscape
	cd $(PAPER_DIR) && pdflatex -shell-escape -interaction=nonstopmode $(PAPER_STEM).tex
	cd $(PAPER_DIR) && bibtex $(PAPER_STEM)
	cd $(PAPER_DIR) && pdflatex -shell-escape -interaction=nonstopmode $(PAPER_STEM).tex
	cd $(PAPER_DIR) && pdflatex -shell-escape -interaction=nonstopmode $(PAPER_STEM).tex

PAPER_DIFF_DIR ?= .cache/diff
PAPER_DIFF_OUT ?= $(PAPER_DIR)/diff
PAPER_DIFF_OPTS ?=

paper-diff:  ## Build a latexdiff PDF against BASE (make paper-diff BASE=<commit> NAME=<label>)
	@test -n "$(BASE)" || { echo "BASE=<commit> is required" >&2; exit 1; }
	@test -n "$(NAME)" || { echo "NAME=<label> is required" >&2; exit 1; }
	@git rev-parse --verify -q "$(BASE)^{commit}" >/dev/null || { echo "unknown commit: $(BASE)" >&2; exit 1; }
	rm -rf $(PAPER_DIFF_DIR) && mkdir -p $(PAPER_DIFF_DIR)/old $(PAPER_DIFF_DIR)/Figures $(PAPER_DIFF_OUT)
	git show $(BASE):$(PAPER_TEX) > $(PAPER_DIFF_DIR)/old/article.tex
	git show $(BASE):$(PAPER_BIB) > $(PAPER_DIFF_DIR)/old/references.bib
	cp $(PAPER_DIR)/Figures/* $(PAPER_DIFF_DIR)/Figures/
	cp $(PAPER_BIB) $(PAPER_DIFF_DIR)/references.bib
	for p in $$(grep -o 'Figures/[A-Za-z0-9_.-]*' $(PAPER_DIFF_DIR)/old/article.tex | sort -u); do \
	  f=$${p#Figures/}; \
	  [ -e $(PAPER_DIFF_DIR)/$$p ] && continue; \
	  case "$$f" in *.*) cands="$$f";; *) cands="$$f.svg $$f.pdf $$f.png";; esac; \
	  for c in $$cands; do \
	    if git cat-file -e $(BASE):$(PAPER_DIR)/Figures/$$c 2>/dev/null; then \
	      git show $(BASE):$(PAPER_DIR)/Figures/$$c > $(PAPER_DIFF_DIR)/Figures/$$c; break; \
	    fi; \
	  done; \
	done
	# No --flatten: it inlines the stale article.bbl and latexdiff then marks its
	# IEEEtran \csname lines, which breaks the bibliography. The article has no
	# \input or \include, so nothing is lost; bibtex builds the bibliography.
	latexdiff $(PAPER_DIFF_OPTS) $(PAPER_DIFF_DIR)/old/article.tex $(PAPER_TEX) > $(PAPER_DIFF_DIR)/diff.tex
	# Labels that exist only in BASE (deleted text) get their BASE numbers from its own .aux
	cd $(PAPER_DIFF_DIR)/old && ln -sfn ../Figures Figures && pdflatex -shell-escape -interaction=nonstopmode article.tex > pass.log 2>&1
	grep -o '\\label{[^}]*}' $(PAPER_TEX) | sed 's/.*{\(.*\)}/\1/' > $(PAPER_DIFF_DIR)/new_labels.txt
	grep '^\\newlabel' $(PAPER_DIFF_DIR)/old/article.aux | awk -F'[{}]' 'NR==FNR {seen[$$0]; next} !($$2 in seen)' $(PAPER_DIFF_DIR)/new_labels.txt - > $(PAPER_DIFF_DIR)/old_labels.tex
	sed -i '0,/^\\begin{document}/s//\\makeatletter\\input{old_labels.tex}\\makeatother\n\\begin{document}/' $(PAPER_DIFF_DIR)/diff.tex
	cd $(PAPER_DIFF_DIR) && pdflatex -shell-escape -interaction=nonstopmode diff.tex
	cd $(PAPER_DIFF_DIR) && bibtex diff
	cd $(PAPER_DIFF_DIR) && pdflatex -shell-escape -interaction=nonstopmode diff.tex
	cd $(PAPER_DIFF_DIR) && pdflatex -shell-escape -interaction=nonstopmode diff.tex
	cp $(PAPER_DIFF_DIR)/diff.pdf $(PAPER_DIFF_OUT)/JCIS_Diff_$(NAME).pdf
	@echo "wrote $(PAPER_DIFF_OUT)/JCIS_Diff_$(NAME).pdf"

SPHINXBUILD = $(VENV)/bin/sphinx-build

docs:  ## Build Sphinx HTML documentation (English)
	cd docs && ../$(SPHINXBUILD) -b html . _build/html -D language=en -W --keep-going

docs-pt:  ## Build Sphinx HTML documentation (Portuguese)
	cd docs && ../$(SPHINXBUILD) -b html . _build/html/pt_BR -D language=pt_BR -W --keep-going

docs-all: docs docs-pt  ## Build Sphinx HTML documentation (both languages)

docs-strict: check-venv  ## Build the docs with -W -n and run the doctests
	cd docs && ../$(SPHINXBUILD) -W -n --keep-going -b html . _build/strict_html -D language=en
	cd docs && ../$(SPHINXBUILD) -b doctest . _build/doctest -D language=en

docs-pdf:  ## Build Sphinx PDF documentation (English)
	cd docs && ../$(SPHINXBUILD) -b latex . _build/latex -D language=en

docs-pdf-pt:  ## Build Sphinx PDF documentation (Portuguese)
	cd docs && ../$(SPHINXBUILD) -b latex . _build/latex -D language=pt_BR

docs-epub:  ## Build Sphinx EPUB documentation (English)
	cd docs && ../$(SPHINXBUILD) -b epub . _build/epub -D language=en

docs-epub-pt:  ## Build Sphinx EPUB documentation (Portuguese)
	cd docs && ../$(SPHINXBUILD) -b epub . _build/epub -D language=pt_BR

benchmark: check-venv  ## Run performance benchmarks
	$(PYTHON) benchmarks/standalone.py

clean:  ## Remove build and cache artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist *.egg-info
	$(MAKE) -C docs clean 2>/dev/null || true

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files
