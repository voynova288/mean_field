from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.bands import solve_bands_along_path, solve_bands_on_grid
from ...core.lattice import KPath
from .bilayer_map import build_atmg_via_tbg_sum
from .hamiltonian import diagonalize_hamiltonian
from .lattice import ATMGLattice, build_moire_k_grid
from .params import ATMGParameters
from .tbg import build_coupling_table


@dataclass(frozen=True)
class PathBandsResult:
    path: KPath
    energies: np.ndarray
    eigenvectors: np.ndarray | None = None
    mapped_energies: np.ndarray | None = None
    subspace_labels: tuple[str, ...] = ()
    subspace_energies: tuple[np.ndarray, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GridBandsResult:
    k_grid_frac: np.ndarray
    kvec: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray | None = None
    mapped_energies: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _basis_dim(lattice: ATMGLattice, params: ATMGParameters) -> int:
    return int(2 * int(params.n_layers) * int(lattice.n_g))


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
    basis_dim = _basis_dim(lattice, params)
    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)

    def diagonalize(
        kval: complex,
        resolved_n_bands: int,
        want_eigenvectors: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        evals, evecs = diagonalize_hamiltonian(
            complex(kval),
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
            coupling_table=coupling_table,
        )
        return evals, evecs if want_eigenvectors else None

    core_result = solve_bands_along_path(
        path,
        basis_dim=basis_dim,
        diagonalize=diagonalize,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        metadata={"system": "ATMG", "valley": int(valley)},
    )

    mapped_energies = None
    subspace_labels: tuple[str, ...] = ()
    subspace_energies: tuple[np.ndarray, ...] | None = None
    if include_mapped:
        resolved_n_bands = int(core_result.energies.shape[1])
        mapped_energies = np.zeros((path.kvec.size, resolved_n_bands), dtype=float)
        mapped_blocks: list[np.ndarray] | None = None
        for ik, kval in enumerate(path.kvec):
            mapped_result = build_atmg_via_tbg_sum(complex(kval), lattice, params, valley=valley)
            mapped_energies[ik, :] = mapped_result.combined_energies[:resolved_n_bands]
            if mapped_blocks is None:
                subspace_labels = mapped_result.labels
                mapped_blocks = [
                    np.zeros((path.kvec.size, bands.size), dtype=float)
                    for bands in mapped_result.subspace_energies
                ]
            assert mapped_blocks is not None
            for block_index, block_energies in enumerate(mapped_result.subspace_energies):
                mapped_blocks[block_index][ik, :] = block_energies
        if mapped_blocks is not None:
            subspace_energies = tuple(mapped_blocks)

    return PathBandsResult(
        path=core_result.path,
        energies=core_result.energies,
        eigenvectors=core_result.eigenvectors,
        mapped_energies=mapped_energies,
        subspace_labels=subspace_labels,
        subspace_energies=subspace_energies,
        metadata=core_result.metadata,
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
    basis_dim = _basis_dim(lattice, params)
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
        evals, evecs = diagonalize_hamiltonian(
            complex(kval),
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
            coupling_table=coupling_table,
        )
        return evals, evecs if want_eigenvectors else None

    core_result = solve_bands_on_grid(
        k_grid_frac,
        kvec,
        basis_dim=basis_dim,
        diagonalize=diagonalize,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        metadata={"system": "ATMG", "valley": int(valley), "mesh_size": int(mesh_size)},
    )

    mapped_energies = None
    if include_mapped:
        resolved_n_bands = int(core_result.energies.shape[-1])
        mapped_energies = np.zeros((mesh_size, mesh_size, resolved_n_bands), dtype=float)
        for i in range(mesh_size):
            for j in range(mesh_size):
                mapped_result = build_atmg_via_tbg_sum(complex(kvec[i, j]), lattice, params, valley=valley)
                mapped_energies[i, j, :] = mapped_result.combined_energies[:resolved_n_bands]

    return GridBandsResult(
        k_grid_frac=core_result.k_grid_frac,
        kvec=core_result.kvec,
        energies=core_result.energies,
        eigenvectors=core_result.eigenvectors,
        mapped_energies=mapped_energies,
        metadata=core_result.metadata,
    )


__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "compute_bands_along_path",
    "compute_bands_on_grid",
]
