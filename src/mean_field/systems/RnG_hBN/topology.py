from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import numpy as np

from analysis.topology import SewingTransform, TopologyResult, make_topology_adapter

from .bands import GridBandsResult, compute_bands_on_grid
from .lattice import RLGhBNLattice
from .params import RLGhBNParams


def _reciprocal_translation(
    lattice: RLGhBNLattice,
    *,
    layer_count: int,
    dn1: int,
    dn2: int,
    valley: int = 1,
) -> SewingTransform:
    valley_sign = int(valley)
    if valley_sign not in {-1, 1}:
        raise ValueError(f"Expected valley ±1, got {valley!r}")
    index_by_g = {tuple(int(value) for value in pair): idx for idx, pair in enumerate(lattice.g_indices)}
    block = 2 * int(layer_count)
    source_dn1 = valley_sign * int(dn1)
    source_dn2 = valley_sign * int(dn2)

    def apply(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.complex128)
        if array.shape[0] != block * lattice.n_g:
            raise ValueError(f"Expected first axis {block * lattice.n_g}, got {array.shape[0]}")
        out = np.zeros_like(array)
        for target_index, (n1, n2) in enumerate(lattice.g_indices):
            source_index = index_by_g.get((int(n1) + source_dn1, int(n2) + source_dn2))
            if source_index is None:
                continue
            out[block * target_index : block * (target_index + 1), ...] = array[
                block * source_index : block * (source_index + 1), ...
            ]
        return out

    return apply


def rlg_hbn_boundary_sewing_transforms(
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    *,
    valley: int = 1,
) -> tuple[SewingTransform, SewingTransform]:
    return (
        _reciprocal_translation(lattice, layer_count=params.layer_count, dn1=1, dn2=0, valley=valley),
        _reciprocal_translation(lattice, layer_count=params.layer_count, dn1=0, dn2=1, valley=valley),
    )


def _orientation_sign(*, orientation_sign: float | None, paper_orientation: bool) -> float:
    if orientation_sign is not None:
        return float(orientation_sign)
    return -1.0 if bool(paper_orientation) else 1.0


def compute_topology_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    orientation_sign: float | None = None,
    paper_orientation: bool = False,
) -> TopologyResult:
    resolved_orientation = _orientation_sign(orientation_sign=orientation_sign, paper_orientation=paper_orientation)
    return make_topology_adapter(
        system="RLG_hBN",
        valley=valley,
        sewing_transforms=sewing_transforms,
        orientation_sign=resolved_orientation,
        index_metadata={"orientation_sign": float(resolved_orientation)},
    )["from_eigenvectors"](eigenvectors, band_indices, k_grid_frac=k_grid_frac)


def compute_topology_from_grid_result(
    grid_result: GridBandsResult,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    lattice: RLGhBNLattice | None = None,
    params: RLGhBNParams | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    use_boundary_sewing: bool = True,
    orientation_sign: float | None = None,
    paper_orientation: bool = False,
) -> TopologyResult:
    if sewing_transforms is None and use_boundary_sewing and lattice is not None and params is not None:
        sewing_transforms = rlg_hbn_boundary_sewing_transforms(lattice, params, valley=valley)
    resolved_orientation = _orientation_sign(orientation_sign=orientation_sign, paper_orientation=paper_orientation)
    return make_topology_adapter(
        system="RLG_hBN",
        valley=valley,
        sewing_transforms=sewing_transforms,
        orientation_sign=resolved_orientation,
        index_metadata={
            "boundary_sewing": sewing_transforms is not None,
            "orientation_sign": float(resolved_orientation),
        },
    )["from_grid_result"](grid_result, band_indices)


def compute_topology_on_grid(
    mesh_size: int,
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    n_bands: int | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    use_boundary_sewing: bool = True,
    orientation_sign: float | None = None,
    paper_orientation: bool = False,
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

    sewing_builder = None
    if sewing_transforms is None and use_boundary_sewing:
        sewing_builder = lambda: rlg_hbn_boundary_sewing_transforms(lattice, params, valley=valley)
    resolved_orientation = _orientation_sign(orientation_sign=orientation_sign, paper_orientation=paper_orientation)
    return make_topology_adapter(
        system="RLG_hBN",
        valley=valley,
        grid_builder=grid_builder,
        sewing_transforms=sewing_transforms,
        sewing_transforms_builder=sewing_builder,
        orientation_sign=resolved_orientation,
        index_metadata={
            "boundary_sewing": sewing_transforms is not None or sewing_builder is not None,
            "orientation_sign": float(resolved_orientation),
        },
    )["on_grid"](mesh_size, band_indices, n_bands=n_bands)


__all__ = [
    "SewingTransform",
    "TopologyResult",
    "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result",
    "compute_topology_on_grid",
    "rlg_hbn_boundary_sewing_transforms",
]
