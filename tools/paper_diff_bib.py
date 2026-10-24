#!/usr/bin/env python3
"""Merge BibTeX entries used only by the annotated manuscript.

``latexdiff`` keeps deleted citations in the output.  Those keys may no longer
exist in the current bibliography, so ``paper-diff`` must compile against the
union of the current database and the entries that exist only in ``BASE``.
The current database always wins; old entries are appended only when their key
is missing from it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ENTRY_START = re.compile(r"(?m)^(?=@)")
ENTRY_KEY = re.compile(r"@[A-Za-z]+\{([^,]+),")


def split_entries(text: str) -> tuple[str, dict[str, str]]:
    """Return the preamble and the BibTeX entries keyed by citation key."""
    parts = ENTRY_START.split(text)
    preamble = parts[0]
    entries: dict[str, str] = {}
    for part in parts[1:]:
        match = ENTRY_KEY.match(part)
        if match:
            entries[match.group(1).strip()] = part.rstrip()
    return preamble, entries


def main() -> None:
    """Write the current bibliography plus old entries with missing keys."""
    if len(sys.argv) != 4:
        raise SystemExit("usage: paper_diff_bib.py CURRENT.bib OLD.bib OUTPUT.bib")

    current_path, old_path, output_path = map(Path, sys.argv[1:])
    current_text = current_path.read_text()
    old_text = old_path.read_text()

    _, current_entries = split_entries(current_text)
    _, old_entries = split_entries(old_text)
    missing = [key for key in old_entries if key not in current_entries]

    merged = current_text.rstrip()
    for key in missing:
        merged += f"\n\n{old_entries[key]}"
    merged += "\n"
    output_path.write_text(merged)

    if missing:
        print("paper-diff: added bibliography entries from BASE: " + ", ".join(missing))


if __name__ == "__main__":
    main()
