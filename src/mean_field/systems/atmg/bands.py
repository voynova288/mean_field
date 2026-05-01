from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.lattice import KPath
from .bilayer_map import build_atmg_via_tbg_sum
from .hamiltonian import diagonalize_hamiltonian
from .lattice import ATMGLattice, build_moire_k_grid
from .params import ATMGParameters


@dataclass(frozen=True)
class PathBandsResult:
    path: KPath
    energies: np.ndarray
    eigenvectors: np.ndarray | None = None
    mapped_energies: np.ndarray | None = None
    subspace_labels: tuple[str, ...] = ()
    subspace_energies: tuple[np.ndarray, ...] | None = None


@dataclass(frozen=True)
class GridBandsResult:
    k_grid_frac: np.ndarray
    kvec: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray | None = None
    mapped_energies: np.ndarray | None = None


def compute_bands_along_path(
    path: KPath,
    lattice: ATMGLattice,
    params: ATMGParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
    include_mapped: bool = False,
) -> PathBandsResult:
    basis_dim = 2 * params.n_layers * lattice.n_g
    resolved_n_bands = basis_dim if n_bands is None else int(n_bands)
    energies = np.zeros((path.kvec.size, resolved_n_bands), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros((path.kvec.size, basis_dim, resolved_n_bands), dtype=np.complex128)

    mapped_energies = None
    subspace_labels: tuple[str, ...] = tuple()
    subspace_energies: tuple[np.ndarray, ...] | None = None
    mapped_blocks: list[np.ndarray] | None = None
    if include_mapped:
        mapped_energies = np.zeros((path.kvec.size, resolved_n_bands), dtype=float)

    for ik, kval in enumerate(path.kvec):
        evals, evecs = diagonalize_hamiltonian(
            complex(kval),
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
        )
        energies[ik, :] = evals
        if return_eigenvectors and eigenvectors is not None:
            eigenvectors[ik, :, :] = evecs

        if include_mapped:
            mapped_result = build_atmg_via_tbg_sum(complex(kval), lattice, params, valley=valley)
            mapped_energies[ik, :] = mapped_result.combined_energies[:resolved_n_bands]
            if mapped_blocks is None:
                subspace_labels = mapped_result.labels
                mapped_blocks = [np.zeros((path.kvec.size, bands.size), dtype=float) for bands in mapped_result.subspace_energies]
            assert mapped_blocks is not None
            for block_index, block_energies in enumerate(mapped_result.subspace_energies):
                mapped_blocks[block_index][ik, :] = block_energies

    if mapped_blocks is not None:
        subspace_energies = tuple(mapped_blocks)

    return PathBandsResult(
        path=path,
        energies=energies,
        eigenvectors=eigenvectors,
        mapped_energies=mapped_energies,
        subspace_labels=subspace_labels,
        subspace_energies=subspace_energies,
    )


def compute_bands_on_grid(
    mesh_size: int,
    lattice: ATMGLattice,
    params: ATMGParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    include_mapped: bool = False,
) -> GridBandsResult:
    basis_dim = 2 * params.n_layers * lattice.n_g
    resolved_n_bands = basis_dim if n_bands is None else int(n_bands)
    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=endpoint, frac_shift=frac_shift)
    energies = np.zeros((mesh_size, mesh_size, resolved_n_bands), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros((mesh_size, mesh_size, basis_dim, resolved_n_bands), dtype=np.complex128)
    mapped_energies = None
    if include_mapped:
        mapped_energies = np.zeros((mesh_size, mesh_size, resolved_n_bands), dtype=float)

    for i in range(mesh_size):
        for j in range(mesh_size):
            evals, evecs = diagonalize_hamiltonian(
                complex(kvec[i, j]),
                lattice,
                params,
                valley=valley,
                n_bands=resolved_n_bands,
            )
            energies[i, j, :] = evals
            if return_eigenvectors and eigenvectors is not None:
                eigenvectors[i, j, :, :] = evecs
            if include_mapped and mapped_energies is not None:
                mapped_result = build_atmg_via_tbg_sum(complex(kvec[i, j]), lattice, params, valley=valley)
                mapped_energies[i, j, :] = mapped_result.combined_energies[:resolved_n_bands]

    return GridBandsResult(
        k_grid_frac=k_grid_frac,
        kvec=np.asarray(kvec, dtype=np.complex128),
        energies=energies,
        eigenvectors=eigenvectors,
        mapped_energies=mapped_energies,
    )


__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "compute_bands_along_path",
    "compute_bands_on_grid",
]
