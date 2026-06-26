from __future__ import annotations

from typing import Iterable

from analysis.topology import FHSState, fhs_state_from_grid_result as _state_from_grid, fhs_state_from_wavefunctions

from .bands import GridBandsResult, compute_bands_on_grid
from .lattice import ATMGLattice
from .params import ATMGParameters


def fhs_state_from_eigenvectors(eigenvectors, band_indices: int | Iterable[int], *, valley: int = 1, k_grid_frac=None) -> FHSState:
    return fhs_state_from_wavefunctions(eigenvectors, band_indices, k_grid_frac=k_grid_frac, system="atmg", valley=valley, reported_indices=band_indices)


def fhs_state_from_grid_result(grid_result: GridBandsResult, band_indices: int | Iterable[int], *, valley: int = 1) -> FHSState:
    return _state_from_grid(grid_result, band_indices, system="atmg", valley=valley)


def fhs_state_on_grid(
    mesh_size: int,
    lattice: ATMGLattice,
    params: ATMGParameters,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    n_bands: int | None = None,
) -> FHSState:
    requested = tuple([int(band_indices)] if isinstance(band_indices, int) else [int(x) for x in band_indices])
    resolved_n_bands = max(requested) + 1 if n_bands is None else int(n_bands)
    grid = compute_bands_on_grid(
        int(mesh_size),
        lattice,
        params,
        valley=valley,
        n_bands=resolved_n_bands,
        return_eigenvectors=True,
        endpoint=endpoint,
        frac_shift=(0.0, 0.0),
    )
    return fhs_state_from_grid_result(grid, requested, valley=valley)


__all__ = ["FHSState", "fhs_state_from_eigenvectors", "fhs_state_from_grid_result", "fhs_state_on_grid"]
