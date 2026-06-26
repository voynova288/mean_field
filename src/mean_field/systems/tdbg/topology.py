from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from analysis.topology import (
    BlockSewingSpec,
    FHSState,
    fhs_state_from_grid_result as _state_from_grid,
    fhs_state_from_wavefunctions,
)

from .bands import GridBandsResult, compute_bands_on_grid
from .lattice import TDBGLattice
from .params import TDBGParameters


def tdbg_basis_sewing(lattice: TDBGLattice, *, atol: float = 1.0e-8) -> BlockSewingSpec:
    q_sites = np.asarray(lattice.q_sites, dtype=float)
    if q_sites.ndim != 2 or q_sites.shape[1] < 3:
        raise ValueError(f"Expected lattice.q_sites shape (n, >=3), got {q_sites.shape}")
    return BlockSewingSpec(
        block_coordinates=q_sites[:, :2],
        local_block_size=4,
        translations=((float(lattice.g_m1.real), float(lattice.g_m1.imag)), (float(lattice.g_m2.real), float(lattice.g_m2.imag))),
        block_labels=q_sites[:, 2].astype(int),
        atol=float(atol),
    )

def fhs_state_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
    basis_sewing: BlockSewingSpec | None = None,
    metadata: dict[str, object] | None = None,
) -> FHSState:
    merged = {} if metadata is None else dict(metadata)
    merged["boundary_sewing"] = basis_sewing is not None
    return fhs_state_from_wavefunctions(
        eigenvectors,
        band_indices,
        k_grid_frac=k_grid_frac,
        basis_sewing=basis_sewing,
        system="tdbg",
        valley=valley,
        metadata=merged,
        reported_indices=band_indices,
    )


def fhs_state_from_grid_result(
    grid_result: GridBandsResult,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    basis_sewing: BlockSewingSpec | None = None,
    metadata: dict[str, object] | None = None,
) -> FHSState:
    merged = {} if metadata is None else dict(metadata)
    merged["boundary_sewing"] = basis_sewing is not None
    return _state_from_grid(grid_result, band_indices, basis_sewing=basis_sewing, system="tdbg", valley=valley, metadata=merged)


def fhs_state_on_grid(
    mesh_size: int,
    lattice: TDBGLattice,
    params: TDBGParameters,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    n_bands: int | None = None,
    boundary_sewing: bool = True,
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
    basis_sewing = tdbg_basis_sewing(lattice) if boundary_sewing else None
    return fhs_state_from_grid_result(
        grid,
        requested,
        valley=valley,
        basis_sewing=basis_sewing,
        metadata={"boundary_sewing": bool(boundary_sewing)},
    )


__all__ = [
    "FHSState",
    "BlockSewingSpec",
    "tdbg_basis_sewing",
    "fhs_state_from_eigenvectors",
    "fhs_state_from_grid_result",
    "fhs_state_on_grid",
]
