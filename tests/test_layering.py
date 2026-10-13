"""Architecture checks for the package layers.

The manuscript tooling and the analysis package must not depend on the plotting
package (the figures are a consumer of the analysis, not the other way round).
The last edge (``analysis.sweep_plotting`` importing a private
``_save`` from ``plotting.figures``) was removed; this test keeps it removed.
"""

from __future__ import annotations

from pathlib import Path

import chaotic_pfc

PKG = Path(chaotic_pfc.__file__).parent


def _source_files(package: str) -> list[Path]:
    return sorted((PKG / package).rglob("*.py"))


def test_analysis_does_not_import_plotting() -> None:
    offenders = []
    for path in _source_files("analysis"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            # ``sweep_plotting`` is analysis' own subpackage; only the plotting
            # package is forbidden.
            if "chaotic_pfc.plotting" in stripped or "..plotting" in stripped:
                offenders.append(path.relative_to(PKG).as_posix())
    assert not offenders, f"analysis package imports plotting: {offenders}"


def test_paper_does_not_import_plotting_private_names() -> None:
    """``chaotic_pfc.paper`` may use plotting, but only its public API."""
    offenders = []
    for path in _source_files("paper"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "chaotic_pfc.plotting" in line and "import" in line and " _save" in line:
                offenders.append(path.relative_to(PKG).as_posix())
    assert not offenders, f"paper imports a private plotting name: {offenders}"
