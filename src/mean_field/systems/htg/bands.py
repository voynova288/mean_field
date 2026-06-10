from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.bands import solve_bands_along_path, solve_bands_on_grid
from .hamiltonian import build_coupling_table, centered_band_indices, diagonalize_hamiltonian
from .lattice import HTGLattice, KPath, build_moire_k_grid
from .params import HTGParams


@dataclass(frozen=True)
class PathBandsResult:
    path: KPath
    energies: np.ndarray
    band_indices: tuple[int, ...]
    eigenvectors: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GridBandsResult:
    k_grid_frac: np.ndarray
    kvec: np.ndarray
    energies: np.ndarray
    band_indices: tuple[int, ...]
    eigenvectors: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _resolve_band_indices(
    lattice: HTGLattice,
    *,
    band_indices: tuple[int, ...] | None,
    central_band_count: int | None,
) -> tuple[int, ...]:
    if band_indices is not None and central_band_count is not None:
        raise ValueError("Pass either band_indices or central_band_count, not both.")
    if band_indices is not None:
        return tuple(int(index) for index in band_indices)
    if central_band_count is not None:
        return centered_band_indices(lattice.matrix_dim, int(central_band_count))
    return tuple(range(lattice.matrix_dim))


def compute_bands_along_path(
    path: KPath,
    lattice: HTGLattice,
    params: HTGParams,
    *,
    valley: int = 1,
    d_top: complex | None = None,
    d_bot: complex | None = None,
    band_indices: tuple[int, ...] | None = None,
    central_band_count: int | None = None,
    return_eigenvectors: bool = False,
) -> PathBandsResult:
    resolved_indices = _resolve_band_indices(
        lattice,
        band_indices=band_indices,
        central_band_count=central_band_count,
    )
    top_coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=1)
    bottom_coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=-1)

    def diagonalize(
        kval: complex,
        _resolved_n_bands: int,
        want_eigenvectors: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return diagonalize_hamiltonian(
            complex(kval),
            lattice,
            params,
            valley=valley,
            d_top=d_top,
            d_bot=d_bot,
            top_coupling_table=top_coupling_table,
            bottom_coupling_table=bottom_coupling_table,
            band_indices=resolved_indices,
            return_eigenvectors=want_eigenvectors,
        )

    core_result = solve_bands_along_path(
        path,
        basis_dim=lattice.matrix_dim,
        diagonalize=diagonalize,
        n_bands=len(resolved_indices),
        return_eigenvectors=return_eigenvectors,
        band_indices=resolved_indices,
        metadata={"system": "HTG", "valley": int(valley)},
    )
    return PathBandsResult(
        path=core_result.path,
        energies=core_result.energies,
        band_indices=resolved_indices,
        eigenvectors=core_result.eigenvectors,
        metadata=core_result.metadata,
    )


def compute_bands_on_grid(
    mesh_size: int,
    lattice: HTGLattice,
    params: HTGParams,
    *,
    valley: int = 1,
    d_top: complex | None = None,
    d_bot: complex | None = None,
    band_indices: tuple[int, ...] | None = None,
    central_band_count: int | None = None,
    return_eigenvectors: bool = False,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> GridBandsResult:
    resolved_indices = _resolve_band_indices(
        lattice,
        band_indices=band_indices,
        central_band_count=central_band_count,
    )
    k_grid_frac, kvec = build_moire_k_grid(
        lattice,
        mesh_size,
        endpoint=endpoint,
        frac_shift=frac_shift,
    )
    top_coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=1)
    bottom_coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=-1)

    def diagonalize(
        kval: complex,
        _resolved_n_bands: int,
        want_eigenvectors: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return diagonalize_hamiltonian(
            complex(kval),
            lattice,
            params,
            valley=valley,
            d_top=d_top,
            d_bot=d_bot,
            top_coupling_table=top_coupling_table,
            bottom_coupling_table=bottom_coupling_table,
            band_indices=resolved_indices,
            return_eigenvectors=want_eigenvectors,
        )

    core_result = solve_bands_on_grid(
        k_grid_frac,
        kvec,
        basis_dim=lattice.matrix_dim,
        diagonalize=diagonalize,
        n_bands=len(resolved_indices),
        return_eigenvectors=return_eigenvectors,
        band_indices=resolved_indices,
        metadata={"system": "HTG", "valley": int(valley), "mesh_size": int(mesh_size)},
    )
    return GridBandsResult(
        k_grid_frac=core_result.k_grid_frac,
        kvec=core_result.kvec,
        energies=core_result.energies,
        band_indices=resolved_indices,
        eigenvectors=core_result.eigenvectors,
        metadata=core_result.metadata,
    )


def estimate_central_band_metrics(result: PathBandsResult, matrix_dim: int) -> dict[str, float | None]:
    band_indices = tuple(int(index) for index in result.band_indices)
    positions = {band_index: pos for pos, band_index in enumerate(band_indices)}
    valence = matrix_dim // 2 - 1
    conduction = matrix_dim // 2
    lower_remote = valence - 1
    upper_remote = conduction + 1
    if valence not in positions or conduction not in positions:
        return {"central_bandwidth_ev": None, "remote_gap_ev": None}

    central = result.energies[:, [positions[valence], positions[conduction]]]
    span = float(np.max(central) - np.min(central))
    bandwidth = 0.5 * span
    remote_gap: float | None = None
    if lower_remote in positions and upper_remote in positions:
        lower_gap = central[:, 0] - result.energies[:, positions[lower_remote]]
        upper_gap = result.energies[:, positions[upper_remote]] - central[:, 1]
        remote_gap = float(min(np.min(lower_gap), np.min(upper_gap)))
    return {
        "central_bandwidth_ev": bandwidth,
        "central_manifold_span_ev": span,
        "remote_gap_ev": remote_gap,
    }
