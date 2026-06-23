from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from analysis.topology import SewingTransform, TopologyResult, compute_system_topology_from_eigenvectors, compute_system_topology_from_grid_result, normalize_state_indices

from .bands import compute_bands_on_grid


def _orientation_sign(*, orientation_sign: float | None, paper_orientation: bool) -> float:
    return float(orientation_sign) if orientation_sign is not None else (-1.0 if bool(paper_orientation) else 1.0)


def _reciprocal_translation(lattice, *, layer_count: int, dn1: int, dn2: int, valley: int = 1) -> SewingTransform:
    valley_sign = int(valley)
    if valley_sign not in {-1, 1}:
        raise ValueError(f"Expected valley ±1, got {valley!r}")
    index_by_g = {tuple(int(value) for value in pair): idx for idx, pair in enumerate(lattice.g_indices)}
    block, source_dn1, source_dn2 = 2 * int(layer_count), valley_sign * int(dn1), valley_sign * int(dn2)

    def apply(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.complex128)
        if array.shape[0] != block * int(lattice.n_g):
            raise ValueError(f"Expected first axis {block * int(lattice.n_g)}, got {array.shape[0]}")
        out = np.zeros_like(array)
        for target_index, (n1, n2) in enumerate(lattice.g_indices):
            source_index = index_by_g.get((int(n1) + source_dn1, int(n2) + source_dn2))
            if source_index is not None:
                out[block * target_index : block * (target_index + 1)] = array[block * source_index : block * (source_index + 1)]
        return out

    return apply


def rlg_hbn_boundary_sewing_transforms(lattice, params, *, valley: int = 1) -> tuple[SewingTransform, SewingTransform]:
    return (
        _reciprocal_translation(lattice, layer_count=params.layer_count, dn1=1, dn2=0, valley=valley),
        _reciprocal_translation(lattice, layer_count=params.layer_count, dn1=0, dn2=1, valley=valley),
    )


def compute_topology_from_eigenvectors(
    eigenvectors, band_indices: int | Iterable[int], *, valley: int = 1, k_grid_frac=None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None, orientation_sign: float | None = None, paper_orientation: bool = False,
) -> TopologyResult:
    sign = _orientation_sign(orientation_sign=orientation_sign, paper_orientation=paper_orientation)
    return compute_system_topology_from_eigenvectors(
        eigenvectors, band_indices, system="RLG_hBN", valley=valley, k_grid_frac=k_grid_frac,
        sewing_transforms=sewing_transforms, index_metadata={"orientation_sign": float(sign)}, orientation_sign=sign,
    )


def compute_topology_from_grid_result(
    grid_result, band_indices: int | Iterable[int], *, valley: int = 1, lattice=None, params=None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None, use_boundary_sewing: bool = True,
    orientation_sign: float | None = None, paper_orientation: bool = False,
) -> TopologyResult:
    if sewing_transforms is None and use_boundary_sewing and lattice is not None and params is not None:
        sewing_transforms = rlg_hbn_boundary_sewing_transforms(lattice, params, valley=valley)
    sign = _orientation_sign(orientation_sign=orientation_sign, paper_orientation=paper_orientation)
    return compute_system_topology_from_grid_result(
        grid_result, band_indices, system="RLG_hBN", valley=valley, sewing_transforms=sewing_transforms,
        index_metadata={"boundary_sewing": sewing_transforms is not None, "orientation_sign": float(sign)}, orientation_sign=sign,
    )


def compute_topology_on_grid(
    mesh_size: int, lattice, params, band_indices: int | Iterable[int], *, valley: int = 1, endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0), n_bands: int | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None, use_boundary_sewing: bool = True,
    orientation_sign: float | None = None, paper_orientation: bool = False,
) -> TopologyResult:
    requested = normalize_state_indices(band_indices)
    resolved_n_bands = max(requested) + 1 if n_bands is None else int(n_bands)
    if resolved_n_bands <= max(requested):
        raise ValueError(f"n_bands={resolved_n_bands} does not include requested band index {max(requested)}")
    grid = compute_bands_on_grid(
        int(mesh_size), lattice, params, valley=int(valley), n_bands=resolved_n_bands, return_eigenvectors=True,
        endpoint=bool(endpoint), frac_shift=(float(frac_shift[0]), float(frac_shift[1])),
    )
    if sewing_transforms is None and bool(use_boundary_sewing):
        sewing_transforms = rlg_hbn_boundary_sewing_transforms(lattice, params, valley=valley)
    return compute_topology_from_grid_result(
        grid, requested, valley=int(valley), sewing_transforms=sewing_transforms, use_boundary_sewing=False,
        orientation_sign=orientation_sign, paper_orientation=paper_orientation,
    )


__all__ = [
    "SewingTransform", "TopologyResult", "compute_topology_from_eigenvectors", "compute_topology_from_grid_result",
    "compute_topology_on_grid", "rlg_hbn_boundary_sewing_transforms",
]
