from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from analysis.topology import WavefunctionIndex, compute_lattice_topology, normalize_state_indices

from .bands import GridBandsResult, compute_bands_on_grid
from .lattice import RLGhBNLattice
from .params import RLGhBNParams


def _normalize_band_indices(band_indices: int | Iterable[int]) -> tuple[int, ...]:
    return normalize_state_indices(band_indices)


@dataclass(frozen=True)
class TopologyResult:
    band_indices: tuple[int, ...]
    valley: int
    k_grid_frac: np.ndarray
    berry_curvature: np.ndarray
    chern_number: float
    rounded_chern_number: int
    berry_connection: np.ndarray | None = None
    min_link_magnitude: float | None = None
    index_metadata: dict[str, object] | None = None

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
    resolved_grid = np.zeros((mesh_x, mesh_y, 2), dtype=float) if k_grid_frac is None else np.asarray(k_grid_frac, dtype=float)
    geometry = compute_lattice_topology(
        eigenvectors,
        normalized_bands,
        index=WavefunctionIndex(
            indices=normalized_bands,
            role="band",
            system="RLG_hBN",
            valley=int(valley),
        ),
        k_grid_frac=resolved_grid,
    )
    return TopologyResult(
        band_indices=normalized_bands,
        valley=int(valley),
        k_grid_frac=geometry.k_grid_frac,
        berry_curvature=geometry.berry_curvature,
        chern_number=geometry.chern_number,
        rounded_chern_number=geometry.rounded_chern_number,
        berry_connection=geometry.berry_connection,
        min_link_magnitude=geometry.min_link_magnitude,
        index_metadata=geometry.wavefunction_index.to_dict(),
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
