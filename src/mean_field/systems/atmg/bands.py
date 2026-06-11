from __future__ import annotations

import numpy as np

from ...core.bands import GridBandsResult, PathBandsResult, compute_grid_bands, compute_path_bands
from ...core.lattice import KPath
from .bilayer_map import build_atmg_via_tbg_sum
from .hamiltonian import diagonalize_hamiltonian
from .lattice import ATMGLattice, build_moire_k_grid
from .params import ATMGParameters
from .tbg import build_coupling_table


def _make_diagonalizer(
    lattice: ATMGLattice,
    params: ATMGParameters,
    *,
    valley: int,
    coupling_table,
):
    def _diagonalize(kval: complex, resolved_band_count: int, want_eigenvectors: bool):
        del want_eigenvectors
        return diagonalize_hamiltonian(
            complex(kval),
            lattice,
            params,
            valley=valley,
            n_bands=resolved_band_count,
            coupling_table=coupling_table,
        )

    return _diagonalize


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
    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)

    if not include_mapped:
        return compute_path_bands(
            path,
            matrix_dim=basis_dim,
            n_bands=n_bands,
            return_eigenvectors=return_eigenvectors,
            diagonalize=_make_diagonalizer(lattice, params, valley=valley, coupling_table=coupling_table),
        )

    resolved_n_bands = basis_dim if n_bands is None else int(n_bands)
    energies = np.zeros((path.kvec.size, resolved_n_bands), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros((path.kvec.size, basis_dim, resolved_n_bands), dtype=np.complex128)
    mapped_energies = np.zeros((path.kvec.size, resolved_n_bands), dtype=float)
    subspace_labels: tuple[str, ...] = tuple()
    subspace_energies: tuple[np.ndarray, ...] | None = None
    mapped_blocks: list[np.ndarray] | None = None

    for ik, kval in enumerate(path.kvec):
        evals, evecs = diagonalize_hamiltonian(
            complex(kval),
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
            coupling_table=coupling_table,
        )
        energies[ik, :] = evals
        if return_eigenvectors and eigenvectors is not None:
            eigenvectors[ik, :, :] = evecs

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
    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=endpoint, frac_shift=frac_shift)
    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)

    if not include_mapped:
        return compute_grid_bands(
            k_grid_frac=k_grid_frac,
            kvec=kvec,
            matrix_dim=basis_dim,
            n_bands=n_bands,
            return_eigenvectors=return_eigenvectors,
            diagonalize=_make_diagonalizer(lattice, params, valley=valley, coupling_table=coupling_table),
        )

    resolved_n_bands = basis_dim if n_bands is None else int(n_bands)
    energies = np.zeros((mesh_size, mesh_size, resolved_n_bands), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros((mesh_size, mesh_size, basis_dim, resolved_n_bands), dtype=np.complex128)
    mapped_energies = np.zeros((mesh_size, mesh_size, resolved_n_bands), dtype=float)

    for i in range(mesh_size):
        for j in range(mesh_size):
            evals, evecs = diagonalize_hamiltonian(
                complex(kvec[i, j]),
                lattice,
                params,
                valley=valley,
                n_bands=resolved_n_bands,
                coupling_table=coupling_table,
            )
            energies[i, j, :] = evals
            if return_eigenvectors and eigenvectors is not None:
                eigenvectors[i, j, :, :] = evecs
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
