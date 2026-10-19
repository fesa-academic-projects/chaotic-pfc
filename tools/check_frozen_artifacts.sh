#!/usr/bin/env bash
#
# Refuse staged changes under the frozen artefact trees.
#
# The versioned checkpoints and the submission records are canonical: they must
# be regenerated with the CLI (and reviewed deliberately), never edited by hand
# as part of an unrelated change. Set CHAOTIC_PFC_ALLOW_FROZEN=1 for a commit
# that intentionally updates them.
set -euo pipefail

if [ "${CHAOTIC_PFC_ALLOW_FROZEN:-0}" = "1" ]; then
    exit 0
fi

frozen_re='^(data/sweeps/|articles/simac/|articles/jcis/Figures/)'
offenders="$(git diff --cached --name-only --diff-filter=ACMRD | grep -E "$frozen_re" || true)"

if [ -n "$offenders" ]; then
    echo "error: staged changes under frozen artefact trees:" >&2
    echo "$offenders" | sed 's/^/  /' >&2
    echo "" >&2
    echo "These files are canonical. Regenerate them with the CLI, or set" >&2
    echo "CHAOTIC_PFC_ALLOW_FROZEN=1 when the update is intentional." >&2
    exit 1
fi
