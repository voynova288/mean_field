from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigvalsh

from ...core.lattice import KPath
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian, flat_band_indices, hamiltonian_dimension
from .lattice import RLGhBNLattice, build_moire_k_grid
from .params import RLGhBNParams


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


def neutrality_energy_mev(path_result: PathBandsResult, lattice: RLGhBNLattice, params: RLGhBNParams) -> float:
    """Energy zero used for paper-style RLG/hBN band plots.

    The continuum Hamiltonian includes an arbitrary fitted onsite offset from
    the RLG remote parameters.  Fig. 2 of Kwan et al. plots the single-particle
    spectrum relative to charge neutrality, so the useful reference is the
    midpoint between the path maximum of the central valence band and the path
    minimum of the central conduction band.
    """

    flat_valence, flat_conduction = flat_band_indices(lattice, params)
    energies = np.asarray(path_result.energies, dtype=float)
    if energies.ndim != 2:
        raise ValueError(f"Expected path energies with shape (n_k, n_bands), got {energies.shape}")
    if flat_conduction >= energies.shape[1]:
        raise ValueError(
            f"Path result only contains {energies.shape[1]} bands, but neutrality reference "
            f"requires central conduction index {flat_conduction}."
        )
    valence_top = float(np.max(energies[:, flat_valence]))
    conduction_bottom = float(np.min(energies[:, flat_conduction]))
    return 0.5 * (valence_top + conduction_bottom)


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
    resolved_n_bands = basis_dim if n_bands is None else int(n_bands)
    energies = np.zeros((path.kvec.size, resolved_n_bands), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros((path.kvec.size, basis_dim, resolved_n_bands), dtype=np.complex128)

    for ik, kval in enumerate(path.kvec):
        if return_eigenvectors:
            evals, evecs = diagonalize_hamiltonian(kval, lattice, params, valley=valley, n_bands=resolved_n_bands)
            if eigenvectors is not None:
                eigenvectors[ik, :, :] = evecs
        else:
            hamiltonian = build_hamiltonian(kval, lattice, params, valley=valley)
            if resolved_n_bands >= basis_dim:
                evals = eigvalsh(hamiltonian)
            else:
                evals = eigvalsh(hamiltonian, subset_by_index=[0, resolved_n_bands - 1])
        energies[ik, :] = np.asarray(evals, dtype=float)

    return PathBandsResult(path=path, energies=energies, eigenvectors=eigenvectors)


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
    resolved_n_bands = basis_dim if n_bands is None else int(n_bands)
    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=endpoint, frac_shift=frac_shift)
    energies = np.zeros((mesh_size, mesh_size, resolved_n_bands), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros((mesh_size, mesh_size, basis_dim, resolved_n_bands), dtype=np.complex128)

    for ix in range(mesh_size):
        for iy in range(mesh_size):
            if return_eigenvectors:
                evals, evecs = diagonalize_hamiltonian(
                    kvec[ix, iy],
                    lattice,
                    params,
                    valley=valley,
                    n_bands=resolved_n_bands,
                )
                if eigenvectors is not None:
                    eigenvectors[ix, iy, :, :] = evecs
            else:
                hamiltonian = build_hamiltonian(kvec[ix, iy], lattice, params, valley=valley)
                if resolved_n_bands >= basis_dim:
                    evals = eigvalsh(hamiltonian)
                else:
                    evals = eigvalsh(hamiltonian, subset_by_index=[0, resolved_n_bands - 1])
            energies[ix, iy, :] = np.asarray(evals, dtype=float)

    return GridBandsResult(
        k_grid_frac=k_grid_frac,
        kvec=np.asarray(kvec, dtype=np.complex128),
        energies=energies,
        eigenvectors=eigenvectors,
    )


__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "compute_bands_along_path",
    "compute_bands_on_grid",
    "neutrality_energy_mev",
]
