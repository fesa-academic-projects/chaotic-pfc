#!/usr/bin/env bash
#
# Guard the Brazilian-Portuguese narrative catalogues.
#
# The English sources under ``docs/`` are the single source of truth.  This
# script extracts the gettext templates, refreshes a throwaway copy of the
# committed ``pt_BR`` catalogue and fails when a narrative page -- everything
# under ``locale/pt_BR/LC_MESSAGES`` except the auto-generated API reference
# ``api/generated`` -- is out of date or not fully translated:
#
#   * outdated msgid -- the refreshed catalogue no longer matches the committed
#     one (a source message was changed, added or removed);
#   * fuzzy entry    -- ``msgfmt --statistics`` reports a fuzzy translation;
#   * untranslated   -- ``msgfmt --statistics`` reports an untranslated message.
#
# The working tree is never touched: everything happens under a temporary
# directory that is removed on exit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
MESSAGES_SUBDIR="pt_BR/LC_MESSAGES"
GENERATED_SUBDIR="api/generated"

SPHINX_BUILD="${SPHINX_BUILD:-sphinx-build}"
SPHINX_INTL="${SPHINX_INTL:-sphinx-intl}"
MSGCAT="${MSGCAT:-msgcat}"
MSGFMT="${MSGFMT:-msgfmt}"

for tool in "$SPHINX_BUILD" "$SPHINX_INTL" "$MSGCAT" "$MSGFMT"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: '$tool' is required but was not found on PATH" >&2
        exit 2
    fi
done

# Keep the gettext tools' output -- and their sort order -- stable.
export LC_ALL=C

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# 1. Extract the message templates from the current English sources.
"$SPHINX_BUILD" -q -b gettext "$DOCS_DIR" "$work/gettext"

# 2. Refresh a scratch copy of the committed catalogue.
mkdir -p "$work/locale"
cp -R "$DOCS_DIR/locale/." "$work/locale/"
"$SPHINX_INTL" update -p "$work/gettext" -d "$work/locale" -l pt_BR >/dev/null

# Canonical message list: sorted, unwrapped, without source locations or
# comments.  Comparing two of these ignores pure formatting churn (line
# wrapping, ``#:`` locations, header timestamps) and keeps only the messages.
canonical() {
    "$MSGCAT" --no-location --no-wrap --sort-output "$1" | grep '^msgid '
}

checked=0
failed=0
while IFS= read -r refreshed; do
    rel="${refreshed#"$work/locale/$MESSAGES_SUBDIR"/}"
    case "$rel" in
        "$GENERATED_SUBDIR"/*) continue ;;
    esac
    checked=$((checked + 1))
    committed="$DOCS_DIR/locale/$MESSAGES_SUBDIR/$rel"

    if [ ! -f "$committed" ]; then
        echo "missing:      $rel (no committed translation)" >&2
        failed=1
        continue
    fi

    canonical "$committed" > "$work/committed.msgids"
    canonical "$refreshed" > "$work/refreshed.msgids"
    if ! diff -q "$work/committed.msgids" "$work/refreshed.msgids" >/dev/null; then
        echo "outdated:     $rel" >&2
        failed=1
    fi

    stats="$("$MSGFMT" --statistics -o /dev/null "$refreshed" 2>&1 || true)"
    fuzzy="$(sed -n 's/.*[^0-9]\([0-9][0-9]*\) fuzzy.*/\1/p' <<<"$stats")"
    untranslated="$(sed -n 's/.*[^0-9]\([0-9][0-9]*\) untranslated.*/\1/p' <<<"$stats")"
    if [ "${fuzzy:-0}" -ne 0 ] || [ "${untranslated:-0}" -ne 0 ]; then
        echo "unfinished:   $rel ($stats)" >&2
        failed=1
    fi
done < <(find "$work/locale/$MESSAGES_SUBDIR" -type f -name '*.po' | sort)

if [ "$failed" -ne 0 ]; then
    cat >&2 <<'EOF'

The pt_BR narrative catalogues are stale or not fully translated.
Regenerate and translate them with:

    cd docs
    sphinx-build -b gettext . _build/gettext
    sphinx-intl update -p _build/gettext -d locale -l pt_BR

Then review the newly fuzzy/untranslated entries and commit the result.
EOF
    exit 1
fi

echo "pt_BR narrative catalogues are up to date and fully translated ($checked files)."
