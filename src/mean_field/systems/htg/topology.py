from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from analysis.topology import SewingTransform, TopologyResult, compute_system_topology_from_eigenvectors, compute_system_topology_from_grid_result, normalize_state_indices
from .bands import compute_bands_on_grid
from ..tmbg.topology import _reciprocal_translation
def boundary_sewing_transforms(lattice) -> tuple[SewingTransform, SewingTransform]:
    return (_reciprocal_translation(lattice, block_size=6, dn1=1, dn2=0), _reciprocal_translation(lattice, block_size=6, dn1=0, dn2=1))
def _metadata(sewing_transforms: Sequence[SewingTransform | None] | None, metadata: Mapping[str, object] | None) -> dict[str, object]:
    payload = {"boundary_sewing": sewing_transforms is not None}
    payload.update(dict(metadata or {}))
    return payload
def _window(band_indices: int | Iterable[int]) -> tuple[int, ...]:
    requested = normalize_state_indices(band_indices)
    return tuple(range(min(requested), max(requested) + 1))
def compute_topology_from_eigenvectors(eigenvectors, band_indices: int | Iterable[int], *, valley: int = 1, k_grid_frac=None, sewing_transforms: Sequence[SewingTransform | None] | None = None, metadata: Mapping[str, object] | None = None, orientation_sign: float = 1.0) -> TopologyResult:
    return compute_system_topology_from_eigenvectors(eigenvectors, band_indices, system="htg", valley=valley, k_grid_frac=k_grid_frac, sewing_transforms=sewing_transforms, index_metadata=_metadata(sewing_transforms, metadata), orientation_sign=orientation_sign)
def compute_topology_from_grid_result(grid_result, band_indices: int | Iterable[int], *, valley: int = 1, sewing_transforms: Sequence[SewingTransform | None] | None = None, metadata: Mapping[str, object] | None = None, orientation_sign: float = 1.0) -> TopologyResult:
    return compute_system_topology_from_grid_result(grid_result, band_indices, system="htg", valley=valley, sewing_transforms=sewing_transforms, index_metadata=_metadata(sewing_transforms, metadata), orientation_sign=orientation_sign)
def compute_topology_on_grid(mesh_size: int, lattice, params, band_indices: int | Iterable[int], *, valley: int = 1, d_top: complex | None = None, d_bot: complex | None = None, endpoint: bool = False, frac_shift: tuple[float, float] = (0.0, 0.0), sewing_transforms: Sequence[SewingTransform | None] | None = None, boundary_sewing: bool = True, orientation_sign: float = 1.0) -> TopologyResult:
    requested = normalize_state_indices(band_indices)
    if endpoint: raise ValueError("Topology FHS meshes must use endpoint=False")
    grid = compute_bands_on_grid(int(mesh_size), lattice, params, valley=int(valley), d_top=d_top, d_bot=d_bot, band_indices=_window(requested), return_eigenvectors=True, endpoint=False, frac_shift=(float(frac_shift[0]), float(frac_shift[1])))
    if sewing_transforms is None and bool(boundary_sewing): sewing_transforms = boundary_sewing_transforms(lattice)
    return compute_topology_from_grid_result(grid, requested, valley=int(valley), sewing_transforms=sewing_transforms, orientation_sign=float(orientation_sign))
__all__ = ["SewingTransform", "TopologyResult", "boundary_sewing_transforms", "compute_topology_from_eigenvectors", "compute_topology_from_grid_result", "compute_topology_on_grid"]
