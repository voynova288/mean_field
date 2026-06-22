from __future__ import annotations

"""Thin TMBG topology adapter.

This module deliberately contains no FHS/link/plaquette implementation. It only
binds TMBG labels/defaults and delegates to :mod:`analysis.topology`.
"""

from collections.abc import Iterable

from analysis.topology import (
    TopologyResult,
    compute_system_topology_from_eigenvectors,
    compute_system_topology_from_grid_result,
    normalize_state_indices,
)

from .bands import GridBandsResult, compute_bands_on_grid
from .lattice import TMBGLattice
from .params import TMBGParameters


def compute_topology_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
    orientation_sign: float = 1.0,
) -> TopologyResult:
    """Compute TMBG topology from an already-built eigenvector mesh."""

    return compute_system_topology_from_eigenvectors(
        eigenvectors,
        band_indices,
        system="tmbg",
        valley=valley,
        k_grid_frac=k_grid_frac,
        orientation_sign=orientation_sign,
    )


def compute_topology_from_grid_result(
    grid_result: GridBandsResult,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    orientation_sign: float = 1.0,
) -> TopologyResult:
    """Compute TMBG topology from a grid result with eigenvectors."""

    return compute_system_topology_from_grid_result(
        grid_result,
        band_indices,
        system="tmbg",
        valley=valley,
        orientation_sign=orientation_sign,
    )


def compute_topology_on_grid(
    mesh_size: int,
    lattice: TMBGLattice,
    params: TMBGParameters,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    n_bands: int | None = None,
    orientation_sign: float = 1.0,
) -> TopologyResult:
    """Compute a single explicit TMBG grid topology run.

    This is a small software adapter, not a paper-level validation workflow. For
    production topology grids, use Slurm and save full provenance separately.
    """

    requested = normalize_state_indices(band_indices)
    resolved_n_bands = max(requested) + 1 if n_bands is None else int(n_bands)
    if resolved_n_bands <= max(requested):
        raise ValueError(
            f"n_bands={resolved_n_bands} does not include requested band index {max(requested)}"
        )
    grid = compute_bands_on_grid(
        int(mesh_size),
        lattice,
        params,
        valley=int(valley),
        n_bands=resolved_n_bands,
        return_eigenvectors=True,
        endpoint=bool(endpoint),
        frac_shift=(float(frac_shift[0]), float(frac_shift[1])),
    )
    return compute_topology_from_grid_result(
        grid,
        requested,
        valley=int(valley),
        orientation_sign=float(orientation_sign),
    )


__all__ = [
    "TopologyResult",
    "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result",
    "compute_topology_on_grid",
]
