from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.lattice import KPath
from .hamiltonian import diagonalize_hamiltonian
from .lattice import AhnTMDLattice, build_rectangular_moire_k_grid
from .params import AhnTMDParameters


@dataclass(frozen=True)
class AhnTMDPathBands:
    kpath: KPath
    energies_eV: np.ndarray
    eigenvectors: np.ndarray | None = None


@dataclass(frozen=True)
class AhnTMDGridBands:
    k_grid_frac: np.ndarray
    kvec: np.ndarray
    energies_eV: np.ndarray
    eigenvectors: np.ndarray | None = None


def compute_bands_along_path(
    path: KPath,
    lattice: AhnTMDLattice,
    params: AhnTMDParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
) -> AhnTMDPathBands:
    energies: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    for kval in path.kvec:
        vals, vecs = diagonalize_hamiltonian(complex(kval), lattice, params, valley=valley, n_bands=n_bands)
        energies.append(vals)
        if return_eigenvectors:
            vectors.append(vecs)
    eig = np.asarray(energies, dtype=float)
    vec_array = np.stack(vectors, axis=0) if return_eigenvectors else None
    return AhnTMDPathBands(kpath=path, energies_eV=eig, eigenvectors=vec_array)


def compute_bands_on_rectangular_grid(
    nx: int,
    ny: int,
    lattice: AhnTMDLattice,
    params: AhnTMDParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> AhnTMDGridBands:
    frac, k_grid = build_rectangular_moire_k_grid(lattice, int(nx), int(ny), frac_shift=frac_shift)
    flat_k = np.asarray(k_grid.reshape(-1), dtype=np.complex128)
    energies: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    for kval in flat_k:
        vals, vecs = diagonalize_hamiltonian(complex(kval), lattice, params, valley=valley, n_bands=n_bands)
        energies.append(vals)
        if return_eigenvectors:
            vectors.append(vecs)
    eig = np.asarray(energies, dtype=float)
    vec_array = np.stack(vectors, axis=0) if return_eigenvectors else None
    return AhnTMDGridBands(k_grid_frac=frac.reshape(-1, 2), kvec=flat_k, energies_eV=eig, eigenvectors=vec_array)


__all__ = ["AhnTMDGridBands", "AhnTMDPathBands", "compute_bands_along_path", "compute_bands_on_rectangular_grid"]
