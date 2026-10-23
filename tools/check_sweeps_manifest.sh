#!/usr/bin/env bash
#
# Verify the SHA-256 manifests of the frozen artefacts.
#
# data/sweeps/SHA256SUMS pins the 184 versioned sweep checkpoints;
# articles/simac/SHA256SUMS pins the frozen SIMAC PDFs and DOCX files.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
failed=0

for manifest in data/sweeps/SHA256SUMS articles/simac/SHA256SUMS; do
    path="$repo_root/$manifest"
    if [ ! -f "$path" ]; then
        echo "error: missing manifest $manifest" >&2
        failed=1
        continue
    fi
    dir="$(dirname "$path")"
    if ! (cd "$dir" && sha256sum --quiet -c "$(basename "$path")"); then
        echo "error: $manifest does not match the files on disk" >&2
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "Regenerate the manifest with the CLI, or restore the modified files." >&2
    exit 1
fi

echo "artefact manifests OK"
