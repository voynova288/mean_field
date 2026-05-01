from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.lattice import KPath
from .hamiltonian import diagonalize_hamiltonian
from .lattice import TDBGLattice, build_moire_k_grid
from .params import TDBGParameters


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
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
) -> PathBandsResult:
    resolved_n_bands = lattice.matrix_dim if n_bands is None else int(n_bands)
    energies = np.zeros((path.kvec.size, resolved_n_bands), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros((path.kvec.size, lattice.matrix_dim, resolved_n_bands), dtype=np.complex128)

    for ik, kval in enumerate(path.kvec):
        evals, evecs = diagonalize_hamiltonian(kval, lattice, params, valley=valley, n_bands=resolved_n_bands)
        energies[ik, :] = evals
        if return_eigenvectors and eigenvectors is not None:
            eigenvectors[ik, :, :] = evecs

    return PathBandsResult(path=path, energies=energies, eigenvectors=eigenvectors)


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
    resolved_n_bands = lattice.matrix_dim if n_bands is None else int(n_bands)
    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=endpoint, frac_shift=frac_shift)
    energies = np.zeros((mesh_size, mesh_size, resolved_n_bands), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros((mesh_size, mesh_size, lattice.matrix_dim, resolved_n_bands), dtype=np.complex128)

    for ix in range(mesh_size):
        for iy in range(mesh_size):
            evals, evecs = diagonalize_hamiltonian(
                kvec[ix, iy],
                lattice,
                params,
                valley=valley,
                n_bands=resolved_n_bands,
            )
            energies[ix, iy, :] = evals
            if return_eigenvectors and eigenvectors is not None:
                eigenvectors[ix, iy, :, :] = evecs

    return GridBandsResult(
        k_grid_frac=k_grid_frac,
        kvec=np.asarray(kvec, dtype=np.complex128),
        energies=energies,
        eigenvectors=eigenvectors,
    )
