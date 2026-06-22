from __future__ import annotations

"""Small system-facing adapters for the minimal topology core.

Physical-system modules remain responsible for building wavefunction grids and
for choosing any boundary sewing transforms. These helpers only attach system
metadata and convert the common :class:`LatticeTopologyResult` into a compact
historical-style result shape.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import (
    LatticeTopologyResult,
    LinkMethod,
    SewingTransform,
    WavefunctionIndex,
    compute_lattice_topology,
    default_k_grid_frac,
    normalize_state_indices,
)


@dataclass(frozen=True)
class TopologyResult:
    """Compact topology result for system-facing adapters."""

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

    def to_dict(self) -> dict[str, object]:
        return {
            "band_indices": [int(index) for index in self.band_indices],
            "valley": int(self.valley),
            "chern_number": float(self.chern_number),
            "rounded_chern_number": int(self.rounded_chern_number),
            "integer_residual": float(self.integer_residual),
            "min_link_magnitude": None if self.min_link_magnitude is None else float(self.min_link_magnitude),
            "index_metadata": {} if self.index_metadata is None else dict(self.index_metadata),
        }


def topology_result_from_lattice_result(
    geometry: LatticeTopologyResult,
    *,
    band_indices: int | Iterable[int] | None = None,
    valley: int | None = None,
) -> TopologyResult:
    """Convert the common topology result into a compact system-facing shape."""

    resolved_bands = geometry.wavefunction_index.indices if band_indices is None else normalize_state_indices(band_indices)
    resolved_valley = geometry.wavefunction_index.valley if valley is None else int(valley)
    return TopologyResult(
        band_indices=tuple(int(index) for index in resolved_bands),
        valley=0 if resolved_valley is None else int(resolved_valley),
        k_grid_frac=geometry.k_grid_frac,
        berry_curvature=geometry.berry_curvature,
        chern_number=float(geometry.chern_number),
        rounded_chern_number=int(geometry.rounded_chern_number),
        berry_connection=geometry.berry_connection,
        min_link_magnitude=float(geometry.min_link_magnitude),
        index_metadata=geometry.wavefunction_index.to_dict(),
    )


def compute_system_topology_from_eigenvectors(
    eigenvectors: np.ndarray,
    band_indices: int | Iterable[int],
    *,
    system: str,
    valley: int = 1,
    k_grid_frac: np.ndarray | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    index_metadata: Mapping[str, object] | None = None,
    role: str = "band",
    labels: Iterable[str] = (),
    link_method: LinkMethod = "polar",
    orientation_sign: float = 1.0,
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
    metadata: Mapping[str, object] | None = None,
) -> TopologyResult:
    """Compute topology from an already-built eigenvector mesh.

    ``eigenvectors`` must have shape ``(mesh_x, mesh_y, basis_dim, n_bands)``.
    This function never builds a Hamiltonian and never infers boundary sewing;
    callers must pass any system-specific sewing transforms explicitly.
    """

    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    if vectors.ndim != 4:
        raise ValueError(
            "Expected eigenvectors with shape (mesh_x, mesh_y, basis_dim, n_bands), "
            f"got shape {vectors.shape}"
        )
    normalized_bands = normalize_state_indices(band_indices)
    if max(normalized_bands) >= vectors.shape[-1]:
        raise ValueError(
            f"Band index {max(normalized_bands)} exceeds the available eigenvector count {vectors.shape[-1]}"
        )
    mesh_x, mesh_y = vectors.shape[:2]
    resolved_grid = default_k_grid_frac(mesh_x, mesh_y) if k_grid_frac is None else np.asarray(k_grid_frac, dtype=float)
    geometry = compute_lattice_topology(
        vectors,
        normalized_bands,
        index=WavefunctionIndex(
            indices=normalized_bands,
            role=str(role),
            labels=tuple(str(label) for label in labels),
            system=str(system),
            valley=int(valley),
            metadata=dict(index_metadata or {}),
        ),
        k_grid_frac=resolved_grid,
        sewing_transforms=sewing_transforms,
        link_method=link_method,
        orientation_sign=float(orientation_sign),
        atol=float(atol),
        regularization=float(regularization),
        metadata={} if metadata is None else dict(metadata),
    )
    return topology_result_from_lattice_result(geometry, band_indices=normalized_bands, valley=int(valley))


def compute_system_topology_from_grid_result(
    grid_result: Any,
    band_indices: int | Iterable[int],
    *,
    system: str,
    valley: int = 1,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    index_metadata: Mapping[str, object] | None = None,
    role: str = "band",
    labels: Iterable[str] = (),
    link_method: LinkMethod = "polar",
    orientation_sign: float = 1.0,
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
    metadata: Mapping[str, object] | None = None,
) -> TopologyResult:
    """Compute topology from an object exposing ``eigenvectors`` and optional ``k_grid_frac``."""

    eigenvectors = getattr(grid_result, "eigenvectors", None)
    if eigenvectors is None:
        raise ValueError("Grid eigenvectors are required for topology. Recompute with return_eigenvectors=True.")
    return compute_system_topology_from_eigenvectors(
        eigenvectors,
        band_indices,
        system=system,
        valley=valley,
        k_grid_frac=getattr(grid_result, "k_grid_frac", None),
        sewing_transforms=sewing_transforms,
        index_metadata=index_metadata,
        role=role,
        labels=labels,
        link_method=link_method,
        orientation_sign=orientation_sign,
        atol=atol,
        regularization=regularization,
        metadata=metadata,
    )


__all__ = [
    "TopologyResult",
    "compute_system_topology_from_eigenvectors",
    "compute_system_topology_from_grid_result",
    "topology_result_from_lattice_result",
]
