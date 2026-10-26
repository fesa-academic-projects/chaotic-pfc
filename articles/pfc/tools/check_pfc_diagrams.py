#!/usr/bin/env python3
"""Check that every path and symbol named in the PFC diagrams exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run the diagram path check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("articles/pfc/diagram_paths.json"),
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=Path("articles/pfc/artigo.tex"),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest = json.loads((root / args.manifest).read_text(encoding="utf-8"))
    tex = (root / args.tex).read_text(encoding="utf-8")
    failures: list[str] = []
    checked = 0
    for diagram in manifest["diagrams"]:
        label = diagram["label"]
        if label not in tex:
            failures.append(f"{label}: label not found in {args.tex}")
        for block in diagram["blocks"]:
            checked += 1
            path = root / block["path"]
            if not path.exists():
                failures.append(f"{label}/{block['name']}: missing path {block['path']}")
                continue
            symbol = block.get("symbol")
            if symbol and path.is_file():
                text = path.read_text(encoding="utf-8")
                if symbol not in text:
                    failures.append(
                        f"{label}/{block['name']}: symbol {symbol!r} not found in {block['path']}"
                    )
    print(f"checked {checked} diagram blocks in {len(manifest['diagrams'])} diagrams")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("all diagram paths and symbols exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
