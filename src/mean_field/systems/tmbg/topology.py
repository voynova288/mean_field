from __future__ import annotations

from typing import Iterable

from analysis.topology import BlockSewingSpec, FHSState, fhs_state_from_grid_result as _state_from_grid, fhs_state_from_wavefunctions

from .bands import GridBandsResult, compute_bands_on_grid
from .lattice import TMBGLattice
from .params import TMBGParameters


def tmbg_basis_sewing(lattice: TMBGLattice, *, atol: float = 1.0e-8) -> BlockSewingSpec:
    return BlockSewingSpec(
        block_coordinates=lattice.g_indices.astype(float),
        local_block_size=6,
        translations=((1.0, 0.0), (0.0, 1.0)),
        atol=float(atol),
    )


def fhs_state_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
    basis_sewing: BlockSewingSpec | None = None,
) -> FHSState:
    return fhs_state_from_wavefunctions(
        eigenvectors,
        band_indices,
        k_grid_frac=k_grid_frac,
        basis_sewing=basis_sewing,
        system="tmbg",
        valley=valley,
        reported_indices=band_indices,
        metadata={"boundary_sewing": basis_sewing is not None},
    )


def fhs_state_from_grid_result(
    grid_result: GridBandsResult,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    basis_sewing: BlockSewingSpec | None = None,
) -> FHSState:
    return _state_from_grid(
        grid_result,
        band_indices,
        basis_sewing=basis_sewing,
        system="tmbg",
        valley=valley,
        metadata={"boundary_sewing": basis_sewing is not None},
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
    frac_shift: tuple[float, float] = (0.0, 0.0),
    use_boundary_sewing: bool = True,
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
        frac_shift=frac_shift,
    )
    basis_sewing = tmbg_basis_sewing(lattice) if use_boundary_sewing else None
    return fhs_state_from_grid_result(grid, requested, valley=valley, basis_sewing=basis_sewing)


__all__ = [
    "BlockSewingSpec",
    "FHSState",
    "fhs_state_from_eigenvectors",
    "fhs_state_from_grid_result",
    "fhs_state_on_grid",
    "tmbg_basis_sewing",
]
