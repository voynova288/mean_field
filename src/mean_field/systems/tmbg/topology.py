from __future__ import annotations

from typing import Iterable

import numpy as np

from analysis.topology import (
    BlockSewingSpec,
    FHSState,
    fhs_state_from_grid_result as _state_from_grid,
    fhs_state_from_wavefunctions,
)

from .bands import GridBandsResult, compute_bands_on_grid
from .lattice import TMBGLattice
from .params import TMBGParameters


def tmbg_basis_sewing(
    lattice: TMBGLattice,
    *,
    atol: float = 1.0e-8,
) -> BlockSewingSpec:
    """Describe target-side reciprocal relabeling in the finite TMBG G basis."""

    return BlockSewingSpec(
        block_coordinates=np.asarray(lattice.g_indices, dtype=float),
        local_block_size=6,
        translations=((1.0, 0.0), (0.0, 1.0)),
        atol=float(atol),
    )


def _resolve_basis_sewing(
    *,
    lattice: TMBGLattice | None,
    basis_sewing: BlockSewingSpec | None,
    use_boundary_sewing: bool,
) -> BlockSewingSpec | None:
    if basis_sewing is not None:
        return basis_sewing
    if not use_boundary_sewing:
        return None
    if lattice is None:
        raise ValueError("lattice is required when use_boundary_sewing=True")
    return tmbg_basis_sewing(lattice)


def fhs_state_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    lattice: TMBGLattice | None = None,
    valley: int = 1,
    k_grid_frac=None,
    basis_sewing: BlockSewingSpec | None = None,
    use_boundary_sewing: bool = True,
    metadata: dict[str, object] | None = None,
) -> FHSState:
    sewing = _resolve_basis_sewing(
        lattice=lattice,
        basis_sewing=basis_sewing,
        use_boundary_sewing=use_boundary_sewing,
    )
    merged = {
        "boundary_sewing": sewing is not None,
        "basis_sewing_policy": "finite_g_zero_fill" if sewing is not None else "none",
    }
    if metadata:
        merged.update(dict(metadata))
    return fhs_state_from_wavefunctions(
        eigenvectors,
        band_indices,
        k_grid_frac=k_grid_frac,
        basis_sewing=sewing,
        system="tmbg",
        valley=valley,
        metadata=merged,
        reported_indices=band_indices,
    )


def fhs_state_from_grid_result(
    grid_result: GridBandsResult,
    band_indices: int | Iterable[int],
    *,
    lattice: TMBGLattice | None = None,
    valley: int = 1,
    basis_sewing: BlockSewingSpec | None = None,
    use_boundary_sewing: bool = True,
    metadata: dict[str, object] | None = None,
) -> FHSState:
    sewing = _resolve_basis_sewing(
        lattice=lattice,
        basis_sewing=basis_sewing,
        use_boundary_sewing=use_boundary_sewing,
    )
    merged = {
        "boundary_sewing": sewing is not None,
        "basis_sewing_policy": "finite_g_zero_fill" if sewing is not None else "none",
    }
    if metadata:
        merged.update(dict(metadata))
    return _state_from_grid(
        grid_result,
        band_indices,
        basis_sewing=sewing,
        system="tmbg",
        valley=valley,
        metadata=merged,
    )


def fhs_state_on_grid(
    mesh_size: int,
    lattice: TMBGLattice,
    params: TMBGParameters,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    n_bands: int | None = None,
    basis_sewing: BlockSewingSpec | None = None,
    use_boundary_sewing: bool = True,
) -> FHSState:
    if endpoint:
        raise ValueError("endpoint-inclusive meshes are not valid FHS tori")
    requested = tuple(
        [int(band_indices)]
        if isinstance(band_indices, int)
        else [int(x) for x in band_indices]
    )
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
    return fhs_state_from_grid_result(
        grid,
        requested,
        lattice=lattice,
        valley=valley,
        basis_sewing=basis_sewing,
        use_boundary_sewing=use_boundary_sewing,
        metadata={
            "mesh_size": int(mesh_size),
            "n_shells": int(lattice.n_shells),
        },
    )


__all__ = [
    "BlockSewingSpec",
    "FHSState",
    "fhs_state_from_eigenvectors",
    "fhs_state_from_grid_result",
    "fhs_state_on_grid",
    "tmbg_basis_sewing",
]
