from __future__ import annotations

import numpy as np

from ...core.bands import (
    GridBandsResult,
    PathBandsResult,
    solve_bands_along_path,
    solve_bands_on_grid,
)
from .core_lattice import KPath
from .hamiltonian import build_coupling_table, diagonalize_hamiltonian
from .lattice import TMBGLattice, build_moire_k_grid
from .params import TMBGParameters


def compute_bands_along_path(
    path: KPath,
    lattice: TMBGLattice,
    params: TMBGParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
) -> PathBandsResult:
    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)

    def diagonalize(
        kval: complex,
        resolved_n_bands: int,
        want_eigenvectors: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return diagonalize_hamiltonian(
            complex(kval),
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
            return_eigenvectors=want_eigenvectors,
            coupling_table=coupling_table,
        )

    return solve_bands_along_path(
        path,
        basis_dim=lattice.matrix_dim,
        diagonalize=diagonalize,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        metadata={"system": "TMBG", "valley": int(valley)},
    )


def compute_bands_on_grid(
    mesh_size: int,
    lattice: TMBGLattice,
    params: TMBGParameters,
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
    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)

    def diagonalize(
        kval: complex,
        resolved_n_bands: int,
        want_eigenvectors: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return diagonalize_hamiltonian(
            complex(kval),
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
            return_eigenvectors=want_eigenvectors,
            coupling_table=coupling_table,
        )

    return solve_bands_on_grid(
        k_grid_frac,
        kvec,
        basis_dim=lattice.matrix_dim,
        diagonalize=diagonalize,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        metadata={"system": "TMBG", "valley": int(valley), "mesh_size": int(mesh_size)},
    )


__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "compute_bands_along_path",
    "compute_bands_on_grid",
]
