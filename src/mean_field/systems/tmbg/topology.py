from __future__ import annotations
from collections.abc import Iterable, Sequence
from analysis.topology import SewingTransform, TopologyResult, compute_system_topology_from_eigenvectors, compute_system_topology_from_grid_result, normalize_state_indices
from .bands import compute_bands_on_grid


def compute_topology_from_eigenvectors(eigenvectors, band_indices: int | Iterable[int], *, valley: int = 1, k_grid_frac=None, sewing_transforms: Sequence[SewingTransform | None] | None = None, orientation_sign: float = 1.0) -> TopologyResult:
    return compute_system_topology_from_eigenvectors(eigenvectors, band_indices, system="tmbg", valley=valley, k_grid_frac=k_grid_frac, sewing_transforms=sewing_transforms, index_metadata={"boundary_sewing": sewing_transforms is not None}, orientation_sign=orientation_sign)


def compute_topology_from_grid_result(grid_result, band_indices: int | Iterable[int], *, valley: int = 1, sewing_transforms: Sequence[SewingTransform | None] | None = None, orientation_sign: float = 1.0) -> TopologyResult:
    return compute_system_topology_from_grid_result(grid_result, band_indices, system="tmbg", valley=valley, sewing_transforms=sewing_transforms, index_metadata={"boundary_sewing": sewing_transforms is not None}, orientation_sign=orientation_sign)


def compute_topology_on_grid(
    mesh_size: int, lattice, params, band_indices: int | Iterable[int], *, valley: int = 1, endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0), n_bands: int | None = None, sewing_transforms: Sequence[SewingTransform | None] | None = None,
    orientation_sign: float = 1.0,
) -> TopologyResult:
    requested = normalize_state_indices(band_indices)
    if n_bands is not None and int(n_bands) <= max(requested):
        raise ValueError(f"n_bands={int(n_bands)} does not include requested band index {max(requested)}")
    if endpoint:
        raise ValueError("Topology FHS meshes must use endpoint=False")
    grid = compute_bands_on_grid(
        int(mesh_size), lattice, params, valley=int(valley), n_bands=None if n_bands is None else int(n_bands), return_eigenvectors=True,
        endpoint=False, frac_shift=(float(frac_shift[0]), float(frac_shift[1])),
    )
    return compute_topology_from_grid_result(grid, requested, valley=int(valley), sewing_transforms=sewing_transforms, orientation_sign=float(orientation_sign))


__all__ = ["SewingTransform", "TopologyResult", "compute_topology_from_eigenvectors", "compute_topology_from_grid_result", "compute_topology_on_grid"]
