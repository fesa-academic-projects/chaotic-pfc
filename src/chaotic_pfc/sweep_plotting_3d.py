"""
sweep_plotting_3d.py
====================
Plotly-based 3-D visualisation of a stack of Kaiser β-sweeps.

Aggregates a directory of per-β ``.npz`` checkpoints (produced by
``chaotic-pfc run sweep beta-sweep``) into a single 3-D volume
``λ_max(N_z, ω_c, β)`` and renders it as an interactive surface stack.

Plotly is an *optional* dependency declared under the ``viz3d`` extra:
install with ``pip install -e .[viz3d]``. Importing this module without
Plotly installed will raise an actionable :class:`ImportError`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .sweep import load_sweep

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover — exercised by the no-plotly test
    raise ImportError(
        "Plotly is required for 3-D sweep plotting. "
        "Install the optional dependency: pip install -e '.[viz3d]'"
    ) from exc


def aggregate_beta_sweeps(
    data_dir: str | Path,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Load every per-β sweep under ``data_dir`` into a single volume.

    Parameters
    ----------
    data_dir
        Root directory containing one sub-directory per β, each with a
        ``variables_lyapunov.npz`` file (matches the layout produced by
        ``run sweep beta-sweep``).

    Returns
    -------
    h_volume : ndarray, shape (Nbeta, Ncoef, Ncut)
        λ_max indexed by (β, order, cutoff).
    betas : ndarray, shape (Nbeta,)
        β values, sorted ascending.
    orders : ndarray
    cutoffs : ndarray
    """
    data_dir = Path(data_dir)
    npz_paths = sorted(data_dir.rglob("variables_lyapunov.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No variables_lyapunov.npz files under {data_dir}")

    by_beta: dict[float, NDArray] = {}
    orders_ref: NDArray | None = None
    cutoffs_ref: NDArray | None = None

    for path in npz_paths:
        result = load_sweep(path)
        beta = float(result.metadata.get("kaiser_beta", float("nan")))
        if np.isnan(beta):
            # Skip non-Kaiser sweeps that happen to live under the dir.
            continue
        if orders_ref is None:
            assert result.orders is not None
            orders_ref = result.orders
            cutoffs_ref = result.cutoffs
        else:
            assert cutoffs_ref is not None
            if not np.array_equal(orders_ref, result.orders) or not np.array_equal(
                cutoffs_ref, result.cutoffs
            ):
                raise ValueError(
                    f"Inconsistent grid in {path}: every β-sweep must share the same (orders, cutoffs)."
                )
        by_beta[beta] = result.h

    if not by_beta:
        raise ValueError(f"No Kaiser sweeps with kaiser_beta in metadata under {data_dir}")

    betas = np.array(sorted(by_beta), dtype=np.float64)
    h_volume = np.stack([by_beta[b] for b in betas], axis=0)
    return h_volume, betas, orders_ref, cutoffs_ref  # type: ignore[return-value]


def plot_3d_beta_volume(
    h_volume: NDArray,
    betas: NDArray,
    orders: NDArray,
    cutoffs: NDArray,
    save_path: str | Path | None = None,
) -> go.Figure:
    """Render a stack of λ_max surfaces, one per β.

    Each β contributes a 2-D heat-coloured surface placed at altitude
    ``z = β`` over the (N_z, ω_c) plane. The user can rotate, zoom and
    pick the slice they want in the browser.

    Parameters
    ----------
    h_volume
        Output of :func:`aggregate_beta_sweeps`, shape ``(Nbeta, Ncoef, Ncut)``.
    betas, orders, cutoffs
        Coordinate arrays.
    save_path
        Optional ``.html`` path. If given, the figure is also written
        to disk via ``write_html``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    Nz = np.asarray(orders) - 1
    fig = go.Figure()

    # Use the same λ scale across all surfaces so colours are comparable.
    finite = h_volume[np.isfinite(h_volume)]
    cmin = float(finite.min()) if finite.size else -1.0
    cmax = float(finite.max()) if finite.size else 1.0

    for k, beta in enumerate(betas):
        # surface needs Z(y, x); we orient so x = N_z, y = ω_c.
        z_plane = np.full((len(cutoffs), len(Nz)), float(beta))
        surfacecolor = h_volume[k].T  # shape (Ncut, Ncoef)
        fig.add_trace(
            go.Surface(
                x=Nz,
                y=cutoffs,
                z=z_plane,
                surfacecolor=surfacecolor,
                cmin=cmin,
                cmax=cmax,
                colorscale="RdBu_r",
                showscale=(k == 0),
                colorbar=dict(title="λ_max") if k == 0 else None,
                name=f"β={beta:.2f}",
                opacity=0.85,
            )
        )

    fig.update_layout(
        title="Kaiser β-sweep — λ_max(N_z, ω_c, β)",
        scene=dict(
            xaxis_title="N_z",
            yaxis_title="ω_c / π",
            zaxis_title="β",
        ),
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
    return fig
