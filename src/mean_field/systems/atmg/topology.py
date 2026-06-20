from __future__ import annotations

from typing import Iterable

from analysis.topology import TopologyResult, make_topology_adapter
from .bands import GridBandsResult, compute_bands_on_grid
from .lattice import ATMGLattice
from .params import ATMGParameters


def compute_topology_from_eigenvectors(eigenvectors, band_indices: int | Iterable[int], *, valley: int = 1, k_grid_frac=None) -> TopologyResult:
    return make_topology_adapter(system="atmg", valley=valley)["from_eigenvectors"](eigenvectors, band_indices, k_grid_frac=k_grid_frac)


def compute_topology_from_grid_result(grid_result: GridBandsResult, band_indices: int | Iterable[int], *, valley: int = 1) -> TopologyResult:
    return make_topology_adapter(system="atmg", valley=valley)["from_grid_result"](grid_result, band_indices)


def compute_topology_on_grid(mesh_size: int, lattice: ATMGLattice, params: ATMGParameters, band_indices: int | Iterable[int], *, valley: int = 1, endpoint: bool = False, n_bands: int | None = None) -> TopologyResult:
    def grid_builder(trial_mesh: int, frac_shift: tuple[float, float], resolved_n_bands: int) -> GridBandsResult:
        return compute_bands_on_grid(trial_mesh, lattice, params, valley=valley, n_bands=resolved_n_bands, return_eigenvectors=True, endpoint=endpoint, frac_shift=frac_shift)
    return make_topology_adapter(system="atmg", valley=valley, grid_builder=grid_builder)["on_grid"](mesh_size, band_indices, n_bands=n_bands)


__all__ = ["TopologyResult", "compute_topology_from_eigenvectors", "compute_topology_from_grid_result", "compute_topology_on_grid"]
