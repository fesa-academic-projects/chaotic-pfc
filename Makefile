.PHONY: help setup install test test-fast lint format format-check typecheck check-all check-paper regen-sweeps jcis-figures paper-pdf paper-pdf-build paper-pdf-local paper-diff docs docs-strict docs-i18n-check docs-pt docs-all docs-pdf docs-pdf-pt docs-epub docs-epub-pt benchmark clean pre-commit check-venv notebook

# The test and tooling targets run inside the project virtualenv, where the
# package and its dev dependencies (pytest, ruff, mypy, numba, …) live.
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python

check-venv:  ## Fail with a clear message when the project virtualenv is missing
	@command -v "$(PYTHON)" >/dev/null 2>&1 || { \
		echo "error: $(PYTHON) not found."; \
		echo "create the project virtualenv with:"; \
		echo "    python -m venv $(VENV) && $(VENV)/bin/pip install -e '.[dev,fast]'"; \
		exit 1; \
	}

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv, install the dev/docs extras and the pre-commit hooks
	python -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev,fast,docs]"
	$(VENV)/bin/pre-commit install
	@echo ""
	@echo "optional external tools (missing entries are warnings, not failures):"
	@for entry in "gettext:msgfmt" "pandoc:pandoc" "LibreOffice:soffice"; do \
		name=$${entry%%:*}; cmd=$${entry##*:}; \
		if command -v "$$cmd" >/dev/null 2>&1; then \
			echo "  [ok]   $$name -> $$(command -v $$cmd)"; \
		else \
			echo "  [warn] $$name not found ($$cmd); some optional targets will not run"; \
		fi; \
	done
	@if command -v docker >/dev/null 2>&1 || command -v podman >/dev/null 2>&1; then \
		echo "  [ok]   container runtime -> $$(command -v docker || command -v podman) (make paper-pdf)"; \
	else \
		echo "  [warn] neither docker nor podman found; make paper-pdf needs one (use make paper-pdf-local for a preview)"; \
	fi

install:  ## Install the package with dev dependencies and Numba
	pip install -e ".[dev,fast]"

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
	$(PYTHON) -m mypy --strict src/chaotic_pfc

notebook: check-venv  ## Execute the tutorial notebook (catches API drift in examples/)
	$(PYTHON) -m jupyter nbconvert --to notebook --execute examples/tour.ipynb --output /tmp/tour_executed.ipynb

check-all: lint format-check typecheck test-fast notebook  ## Run all quality checks (lint + format + typecheck + fast tests + notebook)

PAPER_DIR ?= articles/jcis
PAPER_FIGURES_OUT ?= $(PAPER_DIR)/Figures
# Pre-articles/ path, kept for the history- and BASE-dependent targets.
PAPER_DIR_OLD ?= jcis
PAPER_TEX ?= $(PAPER_DIR)/article.tex
PAPER_BIB ?= $(PAPER_DIR)/references.bib
PAPER_TEX_OLD ?= $(PAPER_DIR_OLD)/article.tex
PAPER_BIB_OLD ?= $(PAPER_DIR_OLD)/references.bib
PAPER_STEM ?= article

# Reproducibility stamp: the author date of the last commit that *changed the
# content* of the manuscript sources, ignoring pure renames (the jcis/ ->
# articles/jcis/ move is an add/delete under a pathspec). Both the old and the
# new paths are queried so the stamp is stable across the move.
PAPER_SOURCES = $(PAPER_TEX) $(PAPER_BIB) $(PAPER_DIR)/Figures \
	$(PAPER_TEX_OLD) $(PAPER_BIB_OLD) $(PAPER_DIR_OLD)/Figures
# A shallow checkout can see the HEAD commit without seeing the last commit
# that touched the manuscript. Fall back to the author timestamp of HEAD with
# a warning; outside git (source tarball), accept an explicit SOURCE_DATE_EPOCH.
# An empty value must never reach Inkscape because Fontconfig rejects it.
PAPER_GIT ?= $(shell command -v git 2>/dev/null)
PAPER_IN_GIT := $(if $(PAPER_GIT),$(shell $(PAPER_GIT) -C $(CURDIR) rev-parse --is-inside-work-tree 2>/dev/null))
PAPER_EPOCH_FROM_SOURCES := $(if $(PAPER_IN_GIT),$(shell $(PAPER_GIT) -C $(CURDIR) log -1 --format=%at --diff-filter=CMT -- $(PAPER_SOURCES) 2>/dev/null))
PAPER_EPOCH_FROM_HEAD := $(if $(PAPER_IN_GIT),$(shell $(PAPER_GIT) -C $(CURDIR) log -1 --format=%at 2>/dev/null))
PAPER_EPOCH = $(if $(PAPER_EPOCH_FROM_SOURCES),$(PAPER_EPOCH_FROM_SOURCES),$(if $(PAPER_EPOCH_FROM_HEAD),$(warning PAPER_EPOCH: no commit in PAPER_SOURCES; falling back to the HEAD author timestamp $(PAPER_EPOCH_FROM_HEAD))$(PAPER_EPOCH_FROM_HEAD),$(if $(SOURCE_DATE_EPOCH),$(warning PAPER_EPOCH: no git history; using SOURCE_DATE_EPOCH from the environment)$(SOURCE_DATE_EPOCH),$(error PAPER_EPOCH: no git repository and SOURCE_DATE_EPOCH is not set; run inside a git checkout or set SOURCE_DATE_EPOCH to a Unix timestamp))))

check-paper: check-venv  ## Audit the JCIS manuscript sources against the repository
	$(PYTHON) -m chaotic_pfc paper audit --tex $(PAPER_TEX) --bib $(PAPER_BIB) --figures $(PAPER_DIR)/Figures --all-figures --sweeps data/sweeps

SWEEP_SCRATCH ?= /tmp/chaotic-pfc-sweeps
regen-sweeps: check-venv  ## Recompute data/sweeps in a scratch tree and compare it
	rm -rf $(SWEEP_SCRATCH)
	$(PYTHON) -m chaotic_pfc sweeps regenerate --src data/sweeps --dst $(SWEEP_SCRATCH)
	$(PYTHON) -m chaotic_pfc sweeps compare --old data/sweeps --new $(SWEEP_SCRATCH)

jcis-figures: check-venv  ## Regenerate the JCIS figures not produced by other commands
	$(PYTHON) -m chaotic_pfc paper figures --out-dir $(PAPER_FIGURES_OUT)

# Canonical manuscript build: the PDF is compiled inside the pinned TeX Live
# image, so the versioned artefact is reproducible and matches the CI job. Other
# TeX distributions are not byte-identical (see README and RELEASING).
PAPER_IMAGE ?= texlive/texlive@sha256:acd65bd2c60671fe1537e35440b5d6374ccdcbe3e7fa8a8d7a285fc045c78ed4
CONTAINER ?= $(shell command -v docker 2>/dev/null || command -v podman 2>/dev/null)

paper-pdf:  ## Build the canonical JCIS PDF inside the pinned TeX Live image
	@test -n "$(CONTAINER)" || { \
		echo "error: neither docker nor podman was found."; \
		echo "make paper-pdf builds the canonical PDF inside the pinned image:"; \
		echo "    $(PAPER_IMAGE)"; \
		echo "install docker or podman, or run 'make paper-pdf-local' for a quick"; \
		echo "preview with the system TeX Live (written to $(PAPER_DIR)/build/)."; \
		exit 1; \
	}
	$(CONTAINER) run --rm -e HOST_UID=$$(id -u) -e HOST_GID=$$(id -g) \
		-v "$(CURDIR):/work" -w /work $(PAPER_IMAGE) bash -euc \
		'apt-get update -qq && apt-get install -y -qq --no-install-recommends inkscape make >/dev/null && \
		 git config --global --add safe.directory /work && \
		 make paper-pdf-build && \
		 chown -R "$$HOST_UID:$$HOST_GID" /work/$(PAPER_DIR)'

paper-pdf-build: export SOURCE_DATE_EPOCH = $(PAPER_EPOCH)
paper-pdf-build: export FORCE_SOURCE_DATE = 1
paper-pdf-build:  ## (internal) compile the PDF with the TeX Live of the current image
	rm -rf $(PAPER_DIR)/svg-inkscape
	cd $(PAPER_DIR) && pdflatex -shell-escape -interaction=nonstopmode $(PAPER_STEM).tex
	cd $(PAPER_DIR) && bibtex $(PAPER_STEM)
	cd $(PAPER_DIR) && pdflatex -shell-escape -interaction=nonstopmode $(PAPER_STEM).tex
	cd $(PAPER_DIR) && pdflatex -shell-escape -interaction=nonstopmode $(PAPER_STEM).tex

paper-pdf-local: export SOURCE_DATE_EPOCH = $(PAPER_EPOCH)
paper-pdf-local: export FORCE_SOURCE_DATE = 1
paper-pdf-local:  ## Preview the JCIS PDF with the system TeX Live, into $(PAPER_DIR)/build/ (not versioned)
	rm -rf $(PAPER_DIR)/build
	mkdir -p $(PAPER_DIR)/build
	cp $(PAPER_TEX) $(PAPER_BIB) $(PAPER_DIR)/build/
	cp -r $(PAPER_DIR)/Figures $(PAPER_DIR)/build/
	cd $(PAPER_DIR)/build && pdflatex -shell-escape -interaction=nonstopmode $(PAPER_STEM).tex
	cd $(PAPER_DIR)/build && bibtex $(PAPER_STEM)
	cd $(PAPER_DIR)/build && pdflatex -shell-escape -interaction=nonstopmode $(PAPER_STEM).tex
	cd $(PAPER_DIR)/build && pdflatex -shell-escape -interaction=nonstopmode $(PAPER_STEM).tex
	@echo "preview PDF (not versioned): $(PAPER_DIR)/build/$(PAPER_STEM).pdf"

PAPER_DIFF_DIR ?= .cache/diff
PAPER_DIFF_OUT ?= $(PAPER_DIR)/diff
# Treat every changed tabular as one indivisible block.  Keep tikzpicture in
# PICTUREENV: assigning it replaces latexdiff's default, and without
# tikzpicture the old TikZ source of Fig. 2 is printed as deleted text.
# latexdiff still comments the body of a float that disappears between the
# versions; tools/paper_diff_tables.py restores those old tables below the
# diff so that removals remain visible.
PAPER_DIFF_OPTS ?= --no-label \
	--config="PICTUREENV=(?:picture|tikzpicture|DIFnomarkup|tabular)[\w\d*@]*"

paper-diff: export SOURCE_DATE_EPOCH = $(PAPER_EPOCH)
paper-diff: export FORCE_SOURCE_DATE = 1
paper-diff:  ## Build a latexdiff PDF against BASE (make paper-diff BASE=<commit> NAME=<label>)
	@test -n "$(BASE)" || { echo "BASE=<commit> is required" >&2; exit 1; }
	@test -n "$(NAME)" || { echo "NAME=<label> is required" >&2; exit 1; }
	@git rev-parse --verify -q "$(BASE)^{commit}" >/dev/null || { echo "unknown commit: $(BASE)" >&2; exit 1; }
	rm -rf $(PAPER_DIFF_DIR) && mkdir -p $(PAPER_DIFF_DIR)/old $(PAPER_DIFF_DIR)/Figures $(PAPER_DIFF_OUT)
	@base_tex=$(PAPER_TEX); git cat-file -e $(BASE):$(PAPER_TEX) 2>/dev/null || base_tex=$(PAPER_TEX_OLD); \
	base_bib=$(PAPER_BIB); git cat-file -e $(BASE):$(PAPER_BIB) 2>/dev/null || base_bib=$(PAPER_BIB_OLD); \
	echo "paper-diff: BASE=$(BASE) tex=$$base_tex bib=$$base_bib"; \
	git show $(BASE):$$base_tex > $(PAPER_DIFF_DIR)/old/article.tex; \
	git show $(BASE):$$base_bib > $(PAPER_DIFF_DIR)/old/references.bib
	cp $(PAPER_DIR)/Figures/* $(PAPER_DIFF_DIR)/Figures/
	# Deleted citations keep their old keys, so compile against the union of
	# the current bibliography and entries that exist only in BASE.
	python3 tools/paper_diff_bib.py $(PAPER_BIB) \
		$(PAPER_DIFF_DIR)/old/references.bib $(PAPER_DIFF_DIR)/references.bib
	for p in $$(grep -o 'Figures/[A-Za-z0-9_.-]*' $(PAPER_DIFF_DIR)/old/article.tex | sort -u); do \
	  f=$${p#Figures/}; \
	  [ -e $(PAPER_DIFF_DIR)/$$p ] && continue; \
	  case "$$f" in *.*) cands="$$f";; *) cands="$$f.svg $$f.pdf $$f.png";; esac; \
	  found=; \
	  for c in $$cands; do \
	    for bdir in $(PAPER_DIR)/Figures $(PAPER_DIR_OLD)/Figures; do \
	      if git cat-file -e $(BASE):$$bdir/$$c 2>/dev/null; then \
	        git show $(BASE):$$bdir/$$c > $(PAPER_DIFF_DIR)/Figures/$$c; found=1; break; \
	      fi; \
	    done; \
	    [ -n "$$found" ] && break; \
	  done; \
	done
	# No --flatten: it inlines the stale article.bbl and latexdiff then marks its
	# IEEEtran \csname lines, which breaks the bibliography. The article has no
	# \input or \include, so nothing is lost; bibtex builds the bibliography.
	latexdiff $(PAPER_DIFF_OPTS) $(PAPER_DIFF_DIR)/old/article.tex $(PAPER_TEX) > $(PAPER_DIFF_DIR)/diff.tex
	# latexdiff comments the body of deleted floats; restore the old tables in
	# a red appendix so the removals remain visible in the PDF.
	python3 tools/paper_diff_tables.py $(PAPER_DIFF_DIR)/old/article.tex \
		$(PAPER_DIFF_DIR)/diff.tex
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

docs-strict: check-venv docs-i18n-check  ## Build the docs with -W -n and run the doctests
	cd docs && ../$(SPHINXBUILD) -W -n --keep-going -b html . _build/strict_html -D language=en
	cd docs && ../$(SPHINXBUILD) -b doctest . _build/doctest -D language=en

docs-i18n-check: check-venv  ## Fail when a pt-BR narrative catalogue is stale or untranslated
	SPHINX_BUILD="$(abspath $(SPHINXBUILD))" \
	SPHINX_INTL="$(abspath $(VENV))/bin/sphinx-intl" \
	PYTHON="$(abspath $(PYTHON))" \
	tools/check_pt_br_translations.sh

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
