from __future__ import annotations

import numpy as np
from scipy.linalg import eigvalsh

from ...core.bands import GridBandsResult, PathBandsResult, compute_grid_bands, compute_path_bands
from ...core.lattice import KPath
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian, hamiltonian_dimension
from .lattice import RLGhBNLattice, build_moire_k_grid
from .params import RLGhBNParams



def _make_diagonalizer(lattice: RLGhBNLattice, params: RLGhBNParams, *, valley: int, basis_dim: int):
    def _diagonalize(kval: complex, resolved_n_bands: int, want_eigenvectors: bool):
        if want_eigenvectors:
            return diagonalize_hamiltonian(kval, lattice, params, valley=valley, n_bands=resolved_n_bands)
        hamiltonian = build_hamiltonian(kval, lattice, params, valley=valley)
        if resolved_n_bands >= basis_dim:
            return np.asarray(eigvalsh(hamiltonian), dtype=float), None
        return np.asarray(eigvalsh(hamiltonian, subset_by_index=[0, resolved_n_bands - 1]), dtype=float), None

    return _diagonalize


def compute_bands_along_path(
    path: KPath,
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
) -> PathBandsResult:
    basis_dim = hamiltonian_dimension(lattice, params)
    return compute_path_bands(
        path,
        matrix_dim=basis_dim,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        diagonalize=_make_diagonalizer(lattice, params, valley=valley, basis_dim=basis_dim),
    )


def compute_bands_on_grid(
    mesh_size: int,
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> GridBandsResult:
    basis_dim = hamiltonian_dimension(lattice, params)
    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=endpoint, frac_shift=frac_shift)
    return compute_grid_bands(
        k_grid_frac=k_grid_frac,
        kvec=kvec,
        matrix_dim=basis_dim,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        diagonalize=_make_diagonalizer(lattice, params, valley=valley, basis_dim=basis_dim),
    )


__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "compute_bands_along_path",
    "compute_bands_on_grid",
]
