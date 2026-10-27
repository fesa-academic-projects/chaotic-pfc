#!/usr/bin/env python3
"""Measure the largest vertical blank area on each page of the PFC PDF.

The article class uses non-floating blocks, so a block that does not fit on
the current page leaves a gap before the next page. This script renders every
page, ignores the header and footer bands, and reports the longest run of
all-white rows as a fraction of the page height.

Usage::

    python articles/pfc/tools/measure_pfc_blank.py articles/pfc/artigo.pdf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HEADER = 0.12
FOOTER = 0.91
SIDE = 0.04
WHITE = 245


def render(pdf: Path, dpi: int, out: Path) -> list[Path]:
    """Render every page to PNG and return the files in order."""
    binary = shutil.which("pdftoppm")
    if binary is None:
        raise SystemExit("pdftoppm not found")
    subprocess.run(  # noqa: S603
        [binary, "-png", "-r", str(dpi), str(pdf), str(out / "page")],
        check=True,
        capture_output=True,
    )
    return sorted(out.glob("page-*.png"))


def longest_blank(image: Path) -> tuple[float, float, float]:
    """Return the longest blank run as percentages of the page height."""
    pixels = np.asarray(Image.open(image).convert("L"))
    height, width = pixels.shape
    top, bottom = int(HEADER * height), int(FOOTER * height)
    left, right = int(SIDE * width), int((1.0 - SIDE) * width)
    rows = (pixels[top:bottom, left:right] < WHITE).any(axis=1)
    best = 0
    best_start = 0
    start: int | None = None
    for index, has_ink in enumerate(rows):
        if not has_ink:
            if start is None:
                start = index
        elif start is not None:
            if index - start > best:
                best, best_start = index - start, start
            start = None
    if start is not None and len(rows) - start > best:
        best, best_start = len(rows) - start, start
    page_start = (top + best_start) / height * 100.0
    return best / height * 100.0, page_start, page_start + best / height * 100.0


def main() -> int:
    """Run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, nargs="?", default=Path("articles/pfc/artigo.pdf"))
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pfc-blank-") as tmp:
        pages = render(args.pdf, args.dpi, Path(tmp))
        print(f"{'page':>4} {'blank%':>7} {'from%':>7} {'to%':>7}")
        for number, page in enumerate(pages, 1):
            blank, start, end = longest_blank(page)
            print(f"{number:>4} {blank:>7.1f} {start:>7.1f} {end:>7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
