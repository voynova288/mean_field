from __future__ import annotations

"""Common adapters for physical-system topology wrappers.

System modules remain responsible for building wavefunction grids and optional
boundary sewing transforms.  This module only packages the repeated historical
``TopologyResult`` API and delegates the Berry-link/Chern calculation to
``analysis.topology.core``.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .core import (
    LatticeTopologyResult,
    LinkMethod,
    SewingTransform,
    default_k_grid_frac,
    WavefunctionIndex,
    compute_lattice_topology,
    normalize_state_indices,
)

@dataclass(frozen=True)
class TopologyResult:
    """Historical system-wrapper topology result.

    The common numerical result is :class:`LatticeTopologyResult`.  Physical
    systems expose this narrower dataclass for backward compatibility with the
    older ``mean_field.systems.<system>.topology`` APIs.
    """

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

def topology_result_from_lattice_result(
    geometry: LatticeTopologyResult,
    *,
    band_indices: int | Iterable[int] | None = None,
    valley: int | None = None,
) -> TopologyResult:
    """Convert a unified topology result into the historical wrapper shape."""

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
    """Compute a system-wrapper ``TopologyResult`` from an eigenvector mesh.

    Parameters mirror the duplicated system wrappers: ``eigenvectors`` must have
    shape ``(mesh_x, mesh_y, basis_dim, n_bands)``, while ``system``/``valley``
    and optional ``index_metadata`` preserve the physical labels.  Boundary
    sewing transforms are forwarded unchanged to the unified topology core.
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
    """Compute system topology from a grid object with eigenvectors/k-grid data."""

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


def compute_system_topology_on_grid(
    mesh_size: int,
    band_indices: int | Iterable[int],
    *,
    system: str,
    grid_builder: Callable[[int, tuple[float, float], int], Any],
    topology_builder: Callable[[Any, tuple[int, ...]], TopologyResult] | None = None,
    valley: int = 1,
    n_bands: int | None = None,
    attempts: Sequence[tuple[int, tuple[float, float]]] | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    sewing_transforms_builder: Callable[[], Sequence[SewingTransform | None] | None] | None = None,
    index_metadata: Mapping[str, object] | None = None,
    role: str = "band",
    labels: Iterable[str] = (),
    link_method: LinkMethod = "polar",
    orientation_sign: float = 1.0,
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
    metadata: Mapping[str, object] | None = None,
) -> TopologyResult:
    """Compute system topology through a common grid/retry adapter.

    ``grid_builder`` is the system-specific compute callback.  It receives
    ``(trial_mesh, frac_shift, resolved_n_bands)`` and must return a grid object
    with ``eigenvectors`` and optional ``k_grid_frac`` attributes.  The common
    adapter then applies the same retry policy used by the historical system
    wrappers and delegates the Berry-link/Chern calculation to
    :func:`compute_system_topology_from_grid_result`, unless an optional
    ``topology_builder`` compatibility hook is supplied.
    """

    normalized_bands = normalize_state_indices(band_indices)
    resolved_n_bands = max(normalized_bands) + 1 if n_bands is None else int(n_bands)
    if resolved_n_bands <= max(normalized_bands):
        raise ValueError(
            f"n_bands={resolved_n_bands} does not include the requested target band index {max(normalized_bands)}"
        )

    base_mesh = int(mesh_size)
    resolved_attempts = (
        (
            (base_mesh, (0.0, 0.0)),
            (base_mesh, (0.5 / float(base_mesh), 0.5 / float(base_mesh))),
            (int(2 * base_mesh), (0.0, 0.0)),
        )
        if attempts is None
        else tuple((int(mesh), (float(shift[0]), float(shift[1]))) for mesh, shift in attempts)
    )

    if sewing_transforms is not None and sewing_transforms_builder is not None:
        raise ValueError("Pass either sewing_transforms or sewing_transforms_builder, not both.")

    last_error: ValueError | None = None
    for trial_mesh, frac_shift in resolved_attempts:
        grid_result = grid_builder(int(trial_mesh), (float(frac_shift[0]), float(frac_shift[1])), int(resolved_n_bands))
        try:
            if topology_builder is not None:
                return topology_builder(grid_result, normalized_bands)
            resolved_sewing = sewing_transforms_builder() if sewing_transforms_builder is not None else sewing_transforms
            return compute_system_topology_from_grid_result(
                grid_result,
                normalized_bands,
                system=system,
                valley=valley,
                sewing_transforms=resolved_sewing,
                index_metadata=index_metadata,
                role=role,
                labels=labels,
                link_method=link_method,
                orientation_sign=orientation_sign,
                atol=atol,
                regularization=regularization,
                metadata=metadata,
            )
        except ValueError as exc:
            last_error = exc
            continue
    assert last_error is not None
    raise last_error


def make_topology_adapter(
    *,
    system: str,
    grid_builder: Callable[[int, tuple[float, float], int], Any] | None = None,
    valley: int = 1,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    sewing_transforms_builder: Callable[[], Sequence[SewingTransform | None] | None] | None = None,
    index_metadata: Mapping[str, object] | None = None,
    role: str = "band",
    link_method: LinkMethod = "polar",
    orientation_sign: float = 1.0,
) -> dict[str, Callable[..., TopologyResult]]:
    """Build thin, system-labeled topology adapter callables.

    The factory only fixes repeated metadata/default arguments. Systems must
    still provide physical wavefunctions, grid builders, and any boundary
    sewing transforms; no sewing convention is inferred here.
    """

    base_kwargs = {
        "system": str(system),
        "valley": int(valley),
        "sewing_transforms": sewing_transforms,
        "index_metadata": None if index_metadata is None else dict(index_metadata),
        "role": str(role),
        "link_method": link_method,
        "orientation_sign": float(orientation_sign),
    }
    adapters: dict[str, Callable[..., TopologyResult]] = {
        "from_eigenvectors": partial(compute_system_topology_from_eigenvectors, **base_kwargs),
        "from_grid_result": partial(compute_system_topology_from_grid_result, **base_kwargs),
    }
    if grid_builder is not None:
        adapters["on_grid"] = partial(
            compute_system_topology_on_grid,
            system=str(system),
            grid_builder=grid_builder,
            valley=int(valley),
            sewing_transforms=sewing_transforms,
            sewing_transforms_builder=sewing_transforms_builder,
            index_metadata=None if index_metadata is None else dict(index_metadata),
            role=str(role),
            link_method=link_method,
            orientation_sign=float(orientation_sign),
        )
    return adapters


__all__ = [
    "TopologyResult",
    "compute_system_topology_from_eigenvectors",
    "compute_system_topology_from_grid_result",
    "compute_system_topology_on_grid",
    "make_topology_adapter",
    "topology_result_from_lattice_result",
]
