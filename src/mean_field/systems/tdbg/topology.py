from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from analysis.topology import SewingTransform, TopologyResult, compute_system_topology_from_eigenvectors, compute_system_topology_from_grid_result, normalize_state_indices

from .bands import compute_bands_on_grid
from .projected_hf_geometry import translation_srcmap


def _q_site_sewing_transform(lattice, gvec: complex) -> SewingTransform:
    src = translation_srcmap(lattice, complex(gvec))
    n_q = int(lattice.n_q)
    valid = src >= 0

    def transform(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.complex128)
        if arr.shape[0] != 4 * n_q:
            raise ValueError(f"Expected first axis {4 * n_q} for TDBG sewing, got {arr.shape}")
        reshaped = arr.reshape((n_q, 4) + arr.shape[1:], order="C")
        out = np.zeros_like(reshaped)
        out[valid] = reshaped[src[valid]]
        return out.reshape(arr.shape, order="C")

    return transform


def boundary_sewing_transforms(lattice) -> tuple[SewingTransform, SewingTransform]:
    return (_q_site_sewing_transform(lattice, lattice.g_m1), _q_site_sewing_transform(lattice, lattice.g_m2))


def compute_topology_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    metadata: dict[str, object] | None = None,
    orientation_sign: float = 1.0,
) -> TopologyResult:
    return compute_system_topology_from_eigenvectors(
        eigenvectors, band_indices, system="tdbg", valley=valley, k_grid_frac=k_grid_frac,
        sewing_transforms=sewing_transforms, index_metadata=metadata, orientation_sign=orientation_sign,
    )


def compute_topology_from_grid_result(
    grid_result,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    metadata: dict[str, object] | None = None,
    orientation_sign: float = 1.0,
) -> TopologyResult:
    return compute_system_topology_from_grid_result(
        grid_result, band_indices, system="tdbg", valley=valley, sewing_transforms=sewing_transforms,
        index_metadata=metadata, orientation_sign=orientation_sign,
    )


def compute_topology_on_grid(
    mesh_size: int,
    lattice,
    params,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    n_bands: int | None = None,
    boundary_sewing: bool = True,
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
    return compute_topology_from_grid_result(
        grid, requested, valley=int(valley), sewing_transforms=boundary_sewing_transforms(lattice) if bool(boundary_sewing) else None,
        metadata={"boundary_sewing": bool(boundary_sewing)}, orientation_sign=float(orientation_sign),
    )


__all__ = [
    "TopologyResult", "boundary_sewing_transforms", "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result", "compute_topology_on_grid", "translation_srcmap",
]
