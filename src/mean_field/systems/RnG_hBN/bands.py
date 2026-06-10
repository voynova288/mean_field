from __future__ import annotations

import numpy as np
from scipy.linalg import eigvalsh

from ...core.bands import (
    GridBandsResult,
    PathBandsResult,
    solve_bands_along_path,
    solve_bands_on_grid,
)
from ...core.lattice import KPath
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian, flat_band_indices, hamiltonian_dimension
from .lattice import RLGhBNLattice, build_moire_k_grid
from .params import RLGhBNParams


def neutrality_energy_mev(path_result: PathBandsResult, lattice: RLGhBNLattice, params: RLGhBNParams) -> float:
    """Energy zero used for paper-style RLG/hBN band plots.

    The continuum Hamiltonian includes an arbitrary fitted onsite offset from the
    RLG remote parameters. Fig. 2 of Kwan et al. plots the single-particle
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


def _diagonalize_for_bands(
    kval: complex,
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    *,
    valley: int,
    resolved_n_bands: int,
    basis_dim: int,
    want_eigenvectors: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    if want_eigenvectors:
        evals, evecs = diagonalize_hamiltonian(
            kval,
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
        )
        return evals, evecs

    # Preserve the historical no-eigenvector path, which used eigvalsh directly
    # and avoided allocating eigenvectors.
    hamiltonian = build_hamiltonian(kval, lattice, params, valley=valley)
    if resolved_n_bands >= basis_dim:
        evals = eigvalsh(hamiltonian)
    else:
        evals = eigvalsh(hamiltonian, subset_by_index=[0, resolved_n_bands - 1])
    return np.asarray(evals, dtype=float), None


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

    def diagonalize(
        kval: complex,
        resolved_n_bands: int,
        want_eigenvectors: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return _diagonalize_for_bands(
            kval,
            lattice,
            params,
            valley=valley,
            resolved_n_bands=resolved_n_bands,
            basis_dim=basis_dim,
            want_eigenvectors=want_eigenvectors,
        )

    return solve_bands_along_path(
        path,
        basis_dim=basis_dim,
        diagonalize=diagonalize,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        metadata={"system": "RLG_hBN", "valley": int(valley)},
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
        return _diagonalize_for_bands(
            kval,
            lattice,
            params,
            valley=valley,
            resolved_n_bands=resolved_n_bands,
            basis_dim=basis_dim,
            want_eigenvectors=want_eigenvectors,
        )

    return solve_bands_on_grid(
        k_grid_frac,
        kvec,
        basis_dim=basis_dim,
        diagonalize=diagonalize,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        metadata={"system": "RLG_hBN", "valley": int(valley), "mesh_size": int(mesh_size)},
    )


__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "compute_bands_along_path",
    "compute_bands_on_grid",
    "neutrality_energy_mev",
]
