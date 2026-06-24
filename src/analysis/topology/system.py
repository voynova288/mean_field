from __future__ import annotations

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

def _bundle_metadata_candidates(bundle_or_metadata: Any) -> tuple[dict[str, object], ...]:
    if isinstance(bundle_or_metadata, Mapping):
        return (dict(bundle_or_metadata),)
    candidates: list[dict[str, object]] = []
    for attr in ("basis_metadata", "metadata"):
        value = getattr(bundle_or_metadata, attr, None)
        if isinstance(value, Mapping):
            candidates.append(dict(value))
    return tuple(candidates)

def _bundle_metadata(bundle: Any) -> dict[str, object]:
    merged: dict[str, object] = {}
    for metadata in _bundle_metadata_candidates(bundle):
        merged.update(metadata)
    return merged

def _explicit_false(value: object) -> bool:
    return isinstance(value, (bool, np.bool_)) and not bool(value)

def assert_topology_eligible(bundle_or_metadata: Any, *, context: str = "topology-from-bundle") -> None:
    """Reject reconstructed bundles explicitly marked topology-ineligible.

    Low-level array topology APIs intentionally do not call this guard; it is
    for system-facing helpers that accept bundle objects carrying provenance and
    sewing/topology eligibility metadata.
    """

    for metadata in _bundle_metadata_candidates(bundle_or_metadata):
        if not _explicit_false(metadata.get("topology_eligible")):
            continue
        reason = (
            metadata.get("topology_ineligible_reason")
            or metadata.get("topology_policy")
            or metadata.get("sewing_policy")
            or "metadata marks this reconstructed bundle as topology-ineligible"
        )
        evidence = metadata.get("evidence_paths")
        evidence_text = ""
        if evidence is not None:
            if isinstance(evidence, (str, bytes)):
                evidence_items = [str(evidence)]
            elif isinstance(evidence, Iterable):
                evidence_items = [str(item) for item in evidence]
            else:
                evidence_items = [str(evidence)]
            evidence_text = f" Evidence paths: {evidence_items}."
        raise ValueError(
            f"{context} refused because bundle metadata has topology_eligible=False. "
            f"Reason: {reason}.{evidence_text}"
        )

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
    geometry: LatticeTopologyResult, *, band_indices: int | Iterable[int] | None = None, valley: int | None = None
) -> TopologyResult:
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
    result_band_indices: int | Iterable[int] | None = None,
    role: str = "band",
    labels: Iterable[str] = (),
    link_method: LinkMethod = "polar",
    orientation_sign: float = 1.0,
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
    metadata: Mapping[str, object] | None = None,
) -> TopologyResult:
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    if vectors.ndim != 4:
        raise ValueError(f"Expected eigenvectors with shape (mesh_x, mesh_y, basis_dim, n_bands), got shape {vectors.shape}")
    normalized_bands = normalize_state_indices(band_indices)
    if max(normalized_bands) >= vectors.shape[-1]:
        raise ValueError(f"Band index {max(normalized_bands)} exceeds the available eigenvector count {vectors.shape[-1]}")
    mesh_x, mesh_y = vectors.shape[:2]
    grid = default_k_grid_frac(mesh_x, mesh_y) if k_grid_frac is None else np.asarray(k_grid_frac, dtype=float)
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
        k_grid_frac=grid,
        sewing_transforms=sewing_transforms,
        link_method=link_method,
        orientation_sign=float(orientation_sign),
        atol=float(atol),
        regularization=float(regularization),
        metadata={} if metadata is None else dict(metadata),
    )
    return topology_result_from_lattice_result(
        geometry,
        band_indices=normalized_bands if result_band_indices is None else result_band_indices,
        valley=int(valley),
    )

def _resolve_grid_band_columns(grid_result: Any, band_indices: int | Iterable[int]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    requested = normalize_state_indices(band_indices)
    eigenvectors = getattr(grid_result, "eigenvectors", None)
    if eigenvectors is None:
        raise ValueError("Grid eigenvectors are required for topology. Recompute with return_eigenvectors=True.")
    n_columns = int(np.asarray(eigenvectors).shape[-1])
    available = tuple(int(index) for index in getattr(grid_result, "band_indices", ()) or range(n_columns))
    missing = tuple(index for index in requested if index not in available)
    if missing:
        raise ValueError(f"Requested band_indices {missing} are not available in grid_result.band_indices={available}")
    return requested, tuple(available.index(index) for index in requested), available

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
    requested, columns, available = _resolve_grid_band_columns(grid_result, band_indices)
    payload = dict(index_metadata or {})
    payload.update(
        {
            "band_indices_semantics": "grid_result_band_indices",
            "absolute_band_indices": [int(index) for index in requested],
            "column_indices": [int(index) for index in columns],
            "grid_result_band_indices": [int(index) for index in available],
        }
    )
    return compute_system_topology_from_eigenvectors(
        getattr(grid_result, "eigenvectors"),
        columns,
        system=system,
        valley=valley,
        k_grid_frac=getattr(grid_result, "k_grid_frac", None),
        sewing_transforms=sewing_transforms,
        index_metadata=payload,
        result_band_indices=requested,
        role=role,
        labels=labels,
        link_method=link_method,
        orientation_sign=orientation_sign,
        atol=atol,
        regularization=regularization,
        metadata=metadata,
    )

def _bundle_wavefunctions(bundle: Any) -> np.ndarray:
    if hasattr(bundle, "psi_micro"):
        return np.asarray(getattr(bundle, "psi_micro"), dtype=np.complex128)
    if hasattr(bundle, "wavefunctions"):
        return np.asarray(getattr(bundle, "wavefunctions"), dtype=np.complex128)
    raise ValueError("Expected a topology bundle with a psi_micro or wavefunctions array.")

def _bundle_sewing_transforms(bundle: Any) -> Sequence[SewingTransform | None] | None:
    value = getattr(bundle, "sewing_transforms", None)
    if value is None:
        return None
    if isinstance(value, Sequence) and len(value) == 0:
        return None
    return value

def compute_system_topology_from_bundle(
    bundle: Any,
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
    """Compute topology from a metadata-carrying wavefunction bundle.

    This system-facing helper first enforces ``topology_eligible`` metadata so
    reconstructed bundles that explicitly lack validated torus/sewing support
    fail before entering the common FHS array kernels.
    """

    assert_topology_eligible(bundle, context="compute_system_topology_from_bundle")
    vectors = _bundle_wavefunctions(bundle)
    if vectors.ndim != 4:
        axis_order = _bundle_metadata(bundle).get("psi_micro_axis_order") or _bundle_metadata(bundle).get("wavefunction_axis_order")
        raise ValueError(
            "compute_system_topology_from_bundle requires a 4D torus wavefunction grid "
            "(mesh_x, mesh_y, basis_dim, n_states). "
            f"Got shape {vectors.shape} with axis_order={axis_order!r}; flat reconstructed bundles must use a system topology adapter or supply a validated grid reshape/sewing path."
        )
    payload = _bundle_metadata(bundle)
    payload.update(dict(index_metadata or {}))
    if k_grid_frac is None:
        k_grid_frac = getattr(bundle, "k_grid_frac", None)
    if sewing_transforms is None:
        sewing_transforms = _bundle_sewing_transforms(bundle)
    return compute_system_topology_from_eigenvectors(
        vectors,
        band_indices,
        system=system,
        valley=valley,
        k_grid_frac=k_grid_frac,
        sewing_transforms=sewing_transforms,
        index_metadata=payload,
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
    "assert_topology_eligible",
    "compute_system_topology_from_bundle",
    "compute_system_topology_from_eigenvectors",
    "compute_system_topology_from_grid_result",
    "topology_result_from_lattice_result",
]
