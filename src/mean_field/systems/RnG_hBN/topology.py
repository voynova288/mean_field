from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .bands import GridBandsResult, compute_bands_on_grid
from .lattice import RLGhBNLattice
from .params import RLGhBNParams


def _normalize_band_indices(band_indices: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(band_indices, (int, np.integer)):
        normalized = (int(band_indices),)
    else:
        normalized = tuple(int(index) for index in band_indices)
    if not normalized:
        raise ValueError("Expected at least one target band index.")
    if min(normalized) < 0:
        raise ValueError(f"Band indices must be non-negative, got {normalized}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Band indices must be unique, got {normalized}")
    return normalized


def _unit_link(overlap: complex, *, atol: float = 1.0e-14) -> complex:
    magnitude = abs(overlap)
    if magnitude <= atol:
        raise ValueError(
            "Encountered a near-zero overlap link while building the Fukui-Hatsugai plaquette field. "
            "The target band or subspace is likely not isolated on this grid."
        )
    return overlap / magnitude


def _subspace_link(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    *,
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
) -> complex:
    overlap_matrix = left_vectors.conjugate().T @ right_vectors
    if overlap_matrix.shape == (1, 1):
        return _unit_link(complex(overlap_matrix[0, 0]), atol=atol)

    singular_values = np.linalg.svd(overlap_matrix, compute_uv=False)
    if np.min(singular_values) <= atol:
        overlap_matrix = overlap_matrix + regularization * np.eye(overlap_matrix.shape[0], dtype=np.complex128)
    u_mat, _, vh_mat = np.linalg.svd(overlap_matrix, full_matrices=False)
    phase_matrix = u_mat @ vh_mat
    return _unit_link(complex(np.linalg.det(phase_matrix)), atol=atol)


@dataclass(frozen=True)
class TopologyResult:
    band_indices: tuple[int, ...]
    valley: int
    k_grid_frac: np.ndarray
    berry_curvature: np.ndarray
    chern_number: float
    rounded_chern_number: int

    @property
    def integer_residual(self) -> float:
        return float(abs(self.chern_number - self.rounded_chern_number))

    @property
    def is_nearly_integer(self) -> bool:
        return self.integer_residual < 1.0e-6


def compute_topology_from_eigenvectors(
    eigenvectors: np.ndarray,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac: np.ndarray | None = None,
) -> TopologyResult:
    eigenvectors = np.asarray(eigenvectors, dtype=np.complex128)
    if eigenvectors.ndim != 4:
        raise ValueError(
            "Expected eigenvectors with shape (mesh_x, mesh_y, basis_dim, n_bands), "
            f"got shape {eigenvectors.shape}"
        )

    normalized_bands = _normalize_band_indices(band_indices)
    if max(normalized_bands) >= eigenvectors.shape[-1]:
        raise ValueError(
            f"Band index {max(normalized_bands)} exceeds the available eigenvector count {eigenvectors.shape[-1]}"
        )

    mesh_x, mesh_y = eigenvectors.shape[:2]
    selected = np.take(eigenvectors, normalized_bands, axis=-1)
    ux = np.zeros((mesh_x, mesh_y), dtype=np.complex128)
    uy = np.zeros((mesh_x, mesh_y), dtype=np.complex128)
    for ix in range(mesh_x):
        ix_next = (ix + 1) % mesh_x
        for iy in range(mesh_y):
            iy_next = (iy + 1) % mesh_y
            ux[ix, iy] = _subspace_link(selected[ix, iy], selected[ix_next, iy])
            uy[ix, iy] = _subspace_link(selected[ix, iy], selected[ix, iy_next])

    berry_curvature = np.zeros((mesh_x, mesh_y), dtype=float)
    for ix in range(mesh_x):
        ix_next = (ix + 1) % mesh_x
        for iy in range(mesh_y):
            iy_next = (iy + 1) % mesh_y
            plaquette = ux[ix, iy] * uy[ix_next, iy] / (ux[ix, iy_next] * uy[ix, iy])
            berry_curvature[ix, iy] = float(np.angle(plaquette))

    chern_number = float(np.sum(berry_curvature) / (2.0 * np.pi))
    resolved_grid = np.zeros((mesh_x, mesh_y, 2), dtype=float) if k_grid_frac is None else np.asarray(k_grid_frac, dtype=float)
    return TopologyResult(
        band_indices=normalized_bands,
        valley=int(valley),
        k_grid_frac=resolved_grid,
        berry_curvature=berry_curvature,
        chern_number=chern_number,
        rounded_chern_number=int(np.rint(chern_number)),
    )


def compute_topology_from_grid_result(
    grid_result: GridBandsResult,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
) -> TopologyResult:
    if grid_result.eigenvectors is None:
        raise ValueError("Grid eigenvectors are required for topology. Recompute with return_eigenvectors=True.")
    return compute_topology_from_eigenvectors(
        grid_result.eigenvectors,
        band_indices,
        valley=valley,
        k_grid_frac=grid_result.k_grid_frac,
    )


def compute_topology_on_grid(
    mesh_size: int,
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    n_bands: int | None = None,
) -> TopologyResult:
    normalized_bands = _normalize_band_indices(band_indices)
    resolved_n_bands = max(normalized_bands) + 1 if n_bands is None else int(n_bands)
    attempts = (
        (int(mesh_size), (0.0, 0.0)),
        (int(mesh_size), (0.5 / float(mesh_size), 0.5 / float(mesh_size))),
        (int(2 * mesh_size), (0.0, 0.0)),
    )
    last_error: ValueError | None = None
    for trial_mesh, frac_shift in attempts:
        grid_result = compute_bands_on_grid(
            trial_mesh,
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
            return_eigenvectors=True,
            endpoint=endpoint,
            frac_shift=frac_shift,
        )
        try:
            return compute_topology_from_grid_result(grid_result, normalized_bands, valley=valley)
        except ValueError as exc:
            last_error = exc
            continue
    assert last_error is not None
    raise last_error


__all__ = [
    "TopologyResult",
    "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result",
    "compute_topology_on_grid",
]
