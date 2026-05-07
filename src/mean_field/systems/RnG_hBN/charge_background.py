from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bands import GridBandsResult
from .hamiltonian import valence_band_count
from .lattice import RLGhBNLattice
from .params import RLGhBNParams


@dataclass(frozen=True)
class ChargeBackgroundResult:
    real_space_frac: np.ndarray
    real_space_positions_nm: np.ndarray
    density: np.ndarray
    delta_density: np.ndarray
    n_valence_bands: int


def _basis_offset(g_index: int, layer: int, sublattice: int, params: RLGhBNParams) -> int:
    return int((int(g_index) * params.layer_count + int(layer)) * 2 + int(sublattice))


def compute_valence_charge_background(
    grid_result: GridBandsResult,
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    *,
    real_space_mesh_size: int = 48,
    n_valence_bands: int | None = None,
) -> ChargeBackgroundResult:
    if grid_result.eigenvectors is None:
        raise ValueError("Grid eigenvectors are required. Recompute bands_on_grid with return_eigenvectors=True.")
    if real_space_mesh_size <= 0:
        raise ValueError(f"Expected a positive real_space_mesh_size, got {real_space_mesh_size}")

    resolved_n_valence = valence_band_count(lattice, params) if n_valence_bands is None else int(n_valence_bands)
    if resolved_n_valence <= 0 or resolved_n_valence > grid_result.eigenvectors.shape[-1]:
        raise ValueError(
            f"n_valence_bands={resolved_n_valence} is outside the available eigenvector count "
            f"{grid_result.eigenvectors.shape[-1]}"
        )

    density_fourier: dict[tuple[int, int], complex] = {}
    k_mesh_x, k_mesh_y = grid_result.eigenvectors.shape[:2]
    norm_k = float(k_mesh_x * k_mesh_y)
    for ix in range(k_mesh_x):
        for iy in range(k_mesh_y):
            occupied = grid_result.eigenvectors[ix, iy, :, :resolved_n_valence]
            density_matrix = occupied @ occupied.conjugate().T
            for row_g_index, row_coords in enumerate(lattice.g_indices):
                for col_g_index, col_coords in enumerate(lattice.g_indices):
                    coefficient = 0.0 + 0.0j
                    for layer in range(params.layer_count):
                        for sublattice in range(2):
                            row = _basis_offset(row_g_index, layer, sublattice, params)
                            col = _basis_offset(col_g_index, layer, sublattice, params)
                            coefficient += density_matrix[row, col]
                    delta = (int(row_coords[0] - col_coords[0]), int(row_coords[1] - col_coords[1]))
                    density_fourier[delta] = density_fourier.get(delta, 0.0 + 0.0j) + coefficient / norm_k

    frac = np.arange(real_space_mesh_size, dtype=float) / float(real_space_mesh_size)
    frac_i, frac_j = np.meshgrid(frac, frac, indexing="ij")
    real_positions = frac_i[..., None] * np.asarray([lattice.real_space_a1.real, lattice.real_space_a1.imag], dtype=float)
    real_positions += frac_j[..., None] * np.asarray([lattice.real_space_a2.real, lattice.real_space_a2.imag], dtype=float)
    density = np.zeros((real_space_mesh_size, real_space_mesh_size), dtype=np.complex128)
    for (dn1, dn2), coefficient in density_fourier.items():
        phase = np.exp(2.0j * np.pi * (dn1 * frac_i + dn2 * frac_j))
        density += coefficient * phase

    density_real = np.asarray(density.real, dtype=float)
    delta_density = density_real - float(np.mean(density_real))
    return ChargeBackgroundResult(
        real_space_frac=np.stack([frac_i, frac_j], axis=-1),
        real_space_positions_nm=real_positions,
        density=density_real,
        delta_density=delta_density,
        n_valence_bands=int(resolved_n_valence),
    )


__all__ = ["ChargeBackgroundResult", "compute_valence_charge_background"]
