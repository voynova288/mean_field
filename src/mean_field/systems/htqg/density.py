from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import HTQGLattice, dot_2d


@dataclass(frozen=True)
class ChargeDensityResult:
    real_grid_frac: np.ndarray
    real_space_points_nm: np.ndarray
    density: np.ndarray
    density_min: float
    density_max: float
    density_mean: float
    density_std: float
    uniformity_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "density_min": float(self.density_min),
            "density_max": float(self.density_max),
            "density_mean": float(self.density_mean),
            "density_std": float(self.density_std),
            "uniformity_ratio": float(self.uniformity_ratio),
        }


def build_real_space_grid(lattice: HTQGLattice, mesh_size: int) -> tuple[np.ndarray, np.ndarray]:
    if mesh_size <= 0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")
    frac_1d = np.arange(mesh_size, dtype=float) / float(mesh_size)
    f1, f2 = np.meshgrid(frac_1d, frac_1d, indexing="ij")
    frac = np.stack([f1, f2], axis=-1)
    points = f1 * lattice.a_m1 + f2 * lattice.a_m2
    return frac, np.asarray(points, dtype=np.complex128)


def _as_state_matrix(eigenvectors: np.ndarray, lattice: HTQGLattice) -> np.ndarray:
    array = np.asarray(eigenvectors, dtype=np.complex128)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2 or array.shape[0] != lattice.matrix_dim:
        raise ValueError(f"Expected eigenvectors shape ({lattice.matrix_dim}, n_states), got {array.shape}")
    return array


def charge_density_from_eigenvectors(
    eigenvectors: np.ndarray,
    lattice: HTQGLattice,
    *,
    mesh_size: int = 60,
    normalize_mean: bool = True,
) -> ChargeDensityResult:
    """Evaluate cell-periodic plane-wave charge density in one moiré cell.

    The input can be a single vector ``(basis_dim,)`` or multiple orthonormal
    columns ``(basis_dim, n_states)``.  The returned density is summed over
    layers, sublattices, and selected states.
    """

    vectors = _as_state_matrix(eigenvectors, lattice)
    frac, points = build_real_space_grid(lattice, mesh_size)
    n_states = vectors.shape[1]
    coeffs = vectors.reshape((lattice.n_g, 8, n_states), order="C")
    density = np.zeros((mesh_size, mesh_size), dtype=float)

    # For each orbital and state, psi_orb(r)=sum_G c_G exp(i G.r).  The density
    # is the sum of |psi_orb|^2 over orbitals and selected states.
    for orbital in range(8):
        c = coeffs[:, orbital, :]  # (N_G, n_states)
        psi = np.zeros((mesh_size, mesh_size, n_states), dtype=np.complex128)
        for ig, gvec in enumerate(lattice.g_vectors):
            phase = np.exp(1.0j * (gvec.real * points.real + gvec.imag * points.imag))
            psi += phase[:, :, np.newaxis] * c[ig, :][np.newaxis, np.newaxis, :]
        density += np.sum(np.abs(psi) ** 2, axis=-1)

    if normalize_mean:
        mean = float(np.mean(density))
        if mean > 0.0:
            density = density / mean

    density_min = float(np.min(density))
    density_max = float(np.max(density))
    density_mean = float(np.mean(density))
    density_std = float(np.std(density))
    uniformity_ratio = float(density_max / density_min) if density_min > 0.0 else float("inf")
    return ChargeDensityResult(
        real_grid_frac=frac,
        real_space_points_nm=points,
        density=np.asarray(density, dtype=float),
        density_min=density_min,
        density_max=density_max,
        density_mean=density_mean,
        density_std=density_std,
        uniformity_ratio=uniformity_ratio,
    )


__all__ = ["ChargeDensityResult", "build_real_space_grid", "charge_density_from_eigenvectors"]
