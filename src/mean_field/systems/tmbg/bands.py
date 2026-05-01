from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core_lattice import KPath
from .hamiltonian import diagonalize_hamiltonian
from .lattice import TMBGLattice, build_moire_k_grid
from .params import TMBGParameters


@dataclass(frozen=True)
class PathBandsResult:
    path: KPath
    energies: np.ndarray
    eigenvectors: np.ndarray | None = None


@dataclass(frozen=True)
class GridBandsResult:
    k_grid_frac: np.ndarray
    kvec: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray | None = None


def compute_bands_along_path(
    path: KPath,
    lattice: TMBGLattice,
    params: TMBGParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
) -> PathBandsResult:
    if n_bands is None:
        n_bands = lattice.matrix_dim

    energies = np.zeros((path.kvec.size, n_bands), dtype=float)
    eigenvectors = None if not return_eigenvectors else np.zeros((path.kvec.size, lattice.matrix_dim, n_bands), dtype=np.complex128)

    for ik, kval in enumerate(path.kvec):
        evals, evecs = diagonalize_hamiltonian(
            complex(kval),
            lattice,
            params,
            valley=valley,
            n_bands=n_bands,
            return_eigenvectors=return_eigenvectors,
        )
        energies[ik, :] = evals
        if return_eigenvectors and eigenvectors is not None:
            eigenvectors[ik, :, :] = evecs

    return PathBandsResult(path=path, energies=energies, eigenvectors=eigenvectors)


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
    if n_bands is None:
        n_bands = lattice.matrix_dim

    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=endpoint, frac_shift=frac_shift)
    energies = np.zeros((mesh_size, mesh_size, n_bands), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros((mesh_size, mesh_size, lattice.matrix_dim, n_bands), dtype=np.complex128)

    for i in range(mesh_size):
        for j in range(mesh_size):
            evals, evecs = diagonalize_hamiltonian(
                complex(kvec[i, j]),
                lattice,
                params,
                valley=valley,
                n_bands=n_bands,
                return_eigenvectors=return_eigenvectors,
            )
            energies[i, j, :] = evals
            if return_eigenvectors and eigenvectors is not None:
                eigenvectors[i, j, :, :] = evecs

    return GridBandsResult(
        k_grid_frac=k_grid_frac,
        kvec=np.asarray(kvec, dtype=np.complex128),
        energies=energies,
        eigenvectors=eigenvectors,
    )
