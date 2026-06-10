from __future__ import annotations

import numpy as np

from ...core.bands import (
    GridBandsResult,
    PathBandsResult,
    solve_bands_along_path,
    solve_bands_on_grid,
)
from ...core.lattice import KPath
from .hamiltonian import diagonalize_hamiltonian
from .lattice import TDBGLattice, build_moire_k_grid
from .params import TDBGParameters


def compute_bands_along_path(
    path: KPath,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
) -> PathBandsResult:
    def diagonalize(
        kval: complex,
        resolved_n_bands: int,
        want_eigenvectors: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        evals, evecs = diagonalize_hamiltonian(
            kval,
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
        )
        return evals, evecs if want_eigenvectors else None

    return solve_bands_along_path(
        path,
        basis_dim=lattice.matrix_dim,
        diagonalize=diagonalize,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        metadata={"system": "TDBG", "valley": int(valley)},
    )


def compute_bands_on_grid(
    mesh_size: int,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> GridBandsResult:
    k_grid_frac, kvec = build_moire_k_grid(
        lattice,
        mesh_size,
        endpoint=endpoint,
        frac_shift=frac_shift,
    )

    def diagonalize(
        kval: complex,
        resolved_n_bands: int,
        want_eigenvectors: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        evals, evecs = diagonalize_hamiltonian(
            kval,
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
        )
        return evals, evecs if want_eigenvectors else None

    return solve_bands_on_grid(
        k_grid_frac,
        kvec,
        basis_dim=lattice.matrix_dim,
        diagonalize=diagonalize,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        metadata={"system": "TDBG", "valley": int(valley), "mesh_size": int(mesh_size)},
    )


__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "compute_bands_along_path",
    "compute_bands_on_grid",
]
