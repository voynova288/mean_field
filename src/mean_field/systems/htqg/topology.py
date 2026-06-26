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
from .domains import HTQGDomain
from .lattice import HTQGLattice
from .params import HTQGParams


def htqg_basis_sewing(lattice: HTQGLattice, *, atol: float = 1.0e-8) -> BlockSewingSpec:
    return BlockSewingSpec(
        block_coordinates=np.asarray(lattice.g_indices, dtype=float),
        local_block_size=8,
        translations=((1.0, 0.0), (0.0, 1.0)),
        atol=float(atol),
    )

def fhs_state_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    basis_sewing: BlockSewingSpec | None = None,
    lattice: HTQGLattice | None = None,
    domain: str | HTQGDomain = "alpha_beta_gamma",
    valley: int = 1,
    k_grid_frac=None,
    use_boundary_sewing: bool = True,
    metadata: dict[str, object] | None = None,
) -> FHSState:
    if basis_sewing is None and use_boundary_sewing:
        if lattice is None:
            raise ValueError("lattice is required when use_boundary_sewing=True")
        basis_sewing = htqg_basis_sewing(lattice)
    merged = {"domain": str(domain), "boundary_sewing": basis_sewing is not None}
    if metadata:
        merged.update(dict(metadata))
    return fhs_state_from_wavefunctions(
        eigenvectors,
        band_indices,
        k_grid_frac=k_grid_frac,
        basis_sewing=basis_sewing,
        system="htqg",
        valley=valley,
        metadata=merged,
        reported_indices=band_indices,
    )


def fhs_state_from_grid_result(
    grid_result: GridBandsResult,
    band_indices: int | Iterable[int],
    *,
    lattice: HTQGLattice | None = None,
    domain: str | HTQGDomain = "alpha_beta_gamma",
    valley: int = 1,
    basis_sewing: BlockSewingSpec | None = None,
    use_boundary_sewing: bool = True,
    metadata: dict[str, object] | None = None,
) -> FHSState:
    if basis_sewing is None and use_boundary_sewing:
        if lattice is None:
            raise ValueError("lattice is required when use_boundary_sewing=True")
        basis_sewing = htqg_basis_sewing(lattice)
    merged = {"domain": str(domain), "boundary_sewing": basis_sewing is not None}
    if metadata:
        merged.update(dict(metadata))
    return _state_from_grid(grid_result, band_indices, basis_sewing=basis_sewing, system="htqg", valley=valley, metadata=merged)


def fhs_state_on_grid(
    mesh_size: int,
    lattice: HTQGLattice,
    params: HTQGParams,
    band_indices: int | Iterable[int],
    *,
    domain: str | HTQGDomain = "alpha_beta_gamma",
    valley: int = 1,
    endpoint: bool = False,
    central_band_count: int | None = None,
    frac_shift: tuple[float, float] | None = None,
    use_boundary_sewing: bool = True,
) -> FHSState:
    mesh = int(mesh_size)
    shift = (0.5 / float(mesh), 0.5 / float(mesh)) if frac_shift is None else tuple(float(x) for x in frac_shift)
    grid = compute_bands_on_grid(
        mesh,
        lattice,
        params,
        domain=domain,
        valley=valley,
        band_indices=None,
        central_band_count=central_band_count,
        return_eigenvectors=True,
        endpoint=endpoint,
        frac_shift=shift,
    )
    return fhs_state_from_grid_result(
        grid,
        band_indices,
        lattice=lattice,
        domain=domain,
        valley=valley,
        use_boundary_sewing=use_boundary_sewing,
        metadata={"mesh_size": mesh, "n_shells": int(lattice.n_shells), "theta_deg": float(lattice.theta_deg)},
    )


__all__ = [
    "FHSState",
    "BlockSewingSpec",
    "fhs_state_from_eigenvectors",
    "fhs_state_from_grid_result",
    "fhs_state_on_grid",
    "htqg_basis_sewing",
]
