# Sphinx Makefile for chaotic-pfc documentation
#
# Available targets:
#   html          Build English HTML        → _build/html/en/
#   html-pt       Build Portuguese HTML     → _build/html/pt_BR/
#   html-all      Build both languages
#   clean         Remove build artifacts and generated stubs
#   gettext       Extract translation templates (.pot)
#   update-po     Update .po files from .pot templates
#   pdf           Build English PDF
#   pdf-pt        Build Portuguese PDF
#   epub          Build English EPUB
#   epub-pt       Build Portuguese EPUB

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
HERE          = $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: help clean html html-pt html-all gettext update-po pdf pdf-pt epub epub-pt

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" api/generated

# ─────────────────────────────────────────────────────────────────────
# HTML builds: each language in its own subdirectory
# ─────────────────────────────────────────────────────────────────────

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html/en" \
		-D language=en $(SPHINXOPTS) $(O)

html-pt:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html/pt_BR" \
		-D language=pt_BR $(SPHINXOPTS) $(O)

html-all: html html-pt

# ─────────────────────────────────────────────────────────────────────
# i18n / l10n
# ─────────────────────────────────────────────────────────────────────

gettext:
	@$(SPHINXBUILD) -M gettext "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

update-po:
	../.venv/bin/sphinx-intl update -p _build/gettext -d locale -l pt_BR

# ─────────────────────────────────────────────────────────────────────
# PDF builds
# ─────────────────────────────────────────────────────────────────────

pdf:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" \
		-D language=en $(SPHINXOPTS) $(O)

pdf-pt:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" \
		-D language=pt_BR $(SPHINXOPTS) $(O)

# ─────────────────────────────────────────────────────────────────────
# EPUB builds
# ─────────────────────────────────────────────────────────────────────

epub:
	@$(SPHINXBUILD) -M epub "$(SOURCEDIR)" "$(BUILDDIR)" \
		-D language=en $(SPHINXOPTS) $(O)

epub-pt:
	@$(SPHINXBUILD) -M epub "$(SOURCEDIR)" "$(BUILDDIR)" \
		-D language=pt_BR $(SPHINXOPTS) $(O)

# ─────────────────────────────────────────────────────────────────────
# Catch-all: route unknown targets to sphinx-build
# ─────────────────────────────────────────────────────────────────────
%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
