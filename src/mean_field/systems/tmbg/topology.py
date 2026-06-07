from __future__ import annotations

from typing import Iterable

from analysis.topology import (
    TopologyResult,
    compute_system_topology_from_eigenvectors,
    compute_system_topology_from_grid_result,
    compute_system_topology_on_grid,
    normalize_state_indices,
)

from .bands import GridBandsResult, compute_bands_on_grid
from .lattice import TMBGLattice
from .params import TMBGParameters


def _normalize_band_indices(band_indices: int | Iterable[int]) -> tuple[int, ...]:
    return normalize_state_indices(band_indices)


def compute_topology_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
) -> TopologyResult:
    return compute_system_topology_from_eigenvectors(
        eigenvectors,
        band_indices,
        system="tmbg",
        valley=valley,
        k_grid_frac=k_grid_frac,
    )


def compute_topology_from_grid_result(
    grid_result: GridBandsResult,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
) -> TopologyResult:
    return compute_system_topology_from_grid_result(grid_result, band_indices, system="tmbg", valley=valley)


def _topology_from_grid_result(
    grid_result: GridBandsResult,
    normalized_bands: tuple[int, ...],
    *,
    valley: int,
) -> TopologyResult:
    return compute_topology_from_grid_result(grid_result, normalized_bands, valley=valley)


def compute_topology_on_grid(
    mesh_size: int,
    lattice: TMBGLattice,
    params: TMBGParameters,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    n_bands: int | None = None,
) -> TopologyResult:
    def grid_builder(trial_mesh: int, frac_shift: tuple[float, float], resolved_n_bands: int) -> GridBandsResult:
        return compute_bands_on_grid(
            trial_mesh,
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
            return_eigenvectors=True,
            endpoint=endpoint,
            frac_shift=frac_shift,
        )

    return compute_system_topology_on_grid(
        mesh_size,
        band_indices,
        system="tmbg",
        grid_builder=grid_builder,
        topology_builder=lambda grid_result, bands: _topology_from_grid_result(grid_result, bands, valley=valley),
        valley=valley,
        n_bands=n_bands,
    )


__all__ = [
    "TopologyResult",
    "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result",
    "compute_topology_on_grid",
]
