from __future__ import annotations

"""Minimal FHS/Wilson-link Berry curvature and Chern-number utilities.

This module is intentionally small.  The only topology algorithm kept here is:

1. build normalized Fukui-Hatsugai-Suzuki/Wilson link variables on a 2D torus;
2. compute plaquette Berry flux from those links;
3. integrate that Berry flux to a Chern number.

No projector-QGT, Fubini-Study metric, paper-target registry, wavefunction layout
adapter, or system wrapper factory belongs in this directory.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np

LinkMethod = Literal["polar", "determinant"]
SewingTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BlockSewingSpec:
    """Generic target-side boundary sewing for block-major bases.

    ``block_coordinates`` labels the reciprocal/momentum block coordinate for
    each block in the first basis axis.  ``local_block_size`` is the number of
    internal orbitals in each block.  Optional ``block_labels`` (for example a
    sector/flavor label embedded in the basis) must match under translations.
    This covers G-shell plane-wave bases and q-site bases without system-local
    FHS/sewing algorithms.
    """

    block_coordinates: np.ndarray
    local_block_size: int
    translations: tuple[tuple[float, ...], tuple[float, ...]]
    block_labels: np.ndarray | None = None
    atol: float = 1.0e-8


def _as_2d_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {arr.shape}")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    return arr


def _block_translation_transform(spec: BlockSewingSpec, translation: tuple[float, ...]) -> SewingTransform:
    coords = _as_2d_float_array(spec.block_coordinates, name="block_coordinates")
    shift = np.asarray(translation, dtype=float).reshape(-1)
    if shift.shape != (coords.shape[1],):
        raise ValueError(f"translation shape {shift.shape} does not match coordinate dimension {coords.shape[1]}")
    labels = None if spec.block_labels is None else np.asarray(spec.block_labels)
    if labels is not None and labels.shape[0] != coords.shape[0]:
        raise ValueError("block_labels must have one entry per block")
    local_block_size = int(spec.local_block_size)
    if local_block_size <= 0:
        raise ValueError("local_block_size must be positive")
    source_indices = np.full(coords.shape[0], -1, dtype=int)
    for target_idx, target_coord in enumerate(coords):
        source_coord = target_coord + shift
        matches = np.linalg.norm(coords - source_coord[None, :], axis=1) <= float(spec.atol)
        if labels is not None:
            matches &= labels == labels[target_idx]
        found = np.flatnonzero(matches)
        if found.size:
            source_indices[target_idx] = int(found[0])
    valid = source_indices >= 0

    def apply(vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.complex128)
        expected = local_block_size * coords.shape[0]
        if arr.shape[0] != expected:
            raise ValueError(f"Expected first axis {expected}, got {arr.shape[0]}")
        reshaped = arr.reshape((coords.shape[0], local_block_size) + arr.shape[1:], order="C")
        out = np.zeros_like(reshaped)
        out[valid] = reshaped[source_indices[valid]]
        return out.reshape(arr.shape, order="C")

    return apply


def sewing_transforms_from_block_spec(spec: BlockSewingSpec) -> tuple[SewingTransform, SewingTransform]:
    """Build the two generic target-side seam transforms for an FHS torus."""

    if len(spec.translations) != 2:
        raise ValueError("BlockSewingSpec.translations must contain exactly two translations")
    return (_block_translation_transform(spec, spec.translations[0]), _block_translation_transform(spec, spec.translations[1]))


@dataclass(frozen=True)
class FHSState:
    """Canonical eigenstate input for state -> FHS Berry flux -> Chern.

    The selected state axes are band and optional flavor.  Existing band-only
    wavefunction grids use ``state_indices``; future canonical grids may attach
    explicit flavor metadata in ``metadata`` without changing the FHS algorithm.
    Boundary sewing is either given explicitly or generated generically from
    ``basis_sewing``.
    """

    wavefunctions: np.ndarray
    state_indices: tuple[int, ...] | None = None
    k_grid_frac: np.ndarray | None = None
    sewing_transforms: Sequence[SewingTransform | None] | None = None
    basis_sewing: BlockSewingSpec | None = None
    link_method: LinkMethod = "polar"
    orientation_sign: float = 1.0
    atol: float = 1.0e-14
    regularization: float = 1.0e-12
    system: str | None = None
    valley: int | None = None
    labels: tuple[str, ...] = ()
    reported_indices: tuple[int, ...] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LinkVariables:
    """Normalized FHS/Wilson link variables for one band or one subspace."""

    link_1: np.ndarray
    link_2: np.ndarray
    min_link_magnitude: float

    @property
    def berry_connection(self) -> np.ndarray:
        """Discrete link phases with shape ``(2, mesh_1, mesh_2)``."""

        return np.stack((np.angle(self.link_1), np.angle(self.link_2)), axis=0)


@dataclass(frozen=True)
class LatticeTopologyResult:
    """FHS Berry flux and Chern number on a 2D momentum mesh."""

    state_indices: tuple[int, ...]
    k_grid_frac: np.ndarray
    berry_connection: np.ndarray
    berry_curvature: np.ndarray
    chern_number: float
    rounded_chern_number: int
    min_link_magnitude: float
    link_1: np.ndarray
    link_2: np.ndarray
    system: str | None = None
    valley: int | None = None
    labels: tuple[str, ...] = ()
    reported_indices: tuple[int, ...] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def band_indices(self) -> tuple[int, ...]:
        """Compatibility alias for system wrappers that use band labels."""

        return self.state_indices if self.reported_indices is None else self.reported_indices

    @property
    def index_metadata(self) -> dict[str, object]:
        """Compatibility metadata payload for existing system wrappers."""

        payload: dict[str, object] = {
            "indices": [int(index) for index in self.state_indices],
            "reported_indices": [int(index) for index in self.band_indices],
            "labels": [str(label) for label in self.labels],
        }
        if self.system is not None:
            payload["system"] = str(self.system)
        if self.valley is not None:
            payload["valley"] = int(self.valley)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @property
    def integer_residual(self) -> float:
        return float(abs(self.chern_number - self.rounded_chern_number))

    @property
    def is_nearly_integer(self) -> bool:
        return self.integer_residual < 1.0e-6

    def to_dict(self) -> dict[str, object]:
        return {
            "state_indices": [int(index) for index in self.state_indices],
            "band_indices": [int(index) for index in self.band_indices],
            "system": None if self.system is None else str(self.system),
            "valley": None if self.valley is None else int(self.valley),
            "labels": [str(label) for label in self.labels],
            "chern_number": float(self.chern_number),
            "rounded_chern_number": int(self.rounded_chern_number),
            "integer_residual": float(self.integer_residual),
            "min_link_magnitude": float(self.min_link_magnitude),
            "metadata": dict(self.metadata),
        }


def normalize_state_indices(indices: int | Iterable[int]) -> tuple[int, ...]:
    """Normalize selected state-column indices."""

    if isinstance(indices, (int, np.integer)):
        normalized = (int(indices),)
    else:
        normalized = tuple(int(index) for index in indices)
    if not normalized:
        raise ValueError("Expected at least one target state index.")
    if min(normalized) < 0:
        raise ValueError(f"State indices must be non-negative, got {normalized}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"State indices must be unique, got {normalized}")
    return normalized


def default_k_grid_frac(mesh_1: int, mesh_2: int) -> np.ndarray:
    """Return a regular fractional torus grid with shape ``(mesh_1, mesh_2, 2)``."""

    f1 = np.arange(int(mesh_1), dtype=float) / float(mesh_1)
    f2 = np.arange(int(mesh_2), dtype=float) / float(mesh_2)
    return np.stack(np.meshgrid(f1, f2, indexing="ij"), axis=-1)


def matrix_sewing_transform(matrix: np.ndarray) -> SewingTransform:
    """Return a target-side boundary sewing map ``v -> matrix @ v``."""

    sewing_matrix = np.asarray(matrix, dtype=np.complex128)
    if sewing_matrix.ndim != 2 or sewing_matrix.shape[0] != sewing_matrix.shape[1]:
        raise ValueError(f"Expected square sewing matrix, got shape {sewing_matrix.shape}")

    def apply(vectors: np.ndarray) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.complex128)
        if arr.shape[0] != sewing_matrix.shape[1]:
            raise ValueError(f"Expected first axis {sewing_matrix.shape[1]}, got {arr.shape[0]}")
        return np.tensordot(sewing_matrix, arr, axes=(1, 0))

    return apply


def select_wavefunction_subspace(
    wavefunctions: np.ndarray,
    state_indices: int | Iterable[int] | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Select state columns from a wavefunction mesh.

    Input must be either ``(mesh_1, mesh_2, basis_dim, n_state)`` or a single
    state bundle ``(mesh_1, mesh_2, basis_dim)``.
    """

    arr = np.asarray(wavefunctions, dtype=np.complex128)
    if arr.ndim == 3:
        if state_indices is not None:
            normalized = normalize_state_indices(state_indices)
            if normalized != (0,):
                raise ValueError("Rank-3 wavefunctions contain only state index 0")
        return arr[..., np.newaxis], (0,)
    if arr.ndim != 4:
        raise ValueError(
            "Expected wavefunctions with shape (mesh_1, mesh_2, basis_dim, n_state) "
            f"or (mesh_1, mesh_2, basis_dim), got {arr.shape}"
        )
    normalized = tuple(range(arr.shape[-1])) if state_indices is None else normalize_state_indices(state_indices)
    if max(normalized) >= arr.shape[-1]:
        raise ValueError(f"State index {max(normalized)} exceeds state axis {arr.shape[-1]}")
    return arr[..., normalized], normalized


def _unit_complex(value: complex, *, atol: float) -> tuple[complex, float]:
    magnitude = float(abs(value))
    if magnitude <= float(atol):
        return 1.0 + 0.0j, magnitude
    return complex(value / magnitude), magnitude


def _link_from_overlap(
    overlap_matrix: np.ndarray,
    *,
    method: LinkMethod,
    atol: float,
    regularization: float,
) -> tuple[complex, float]:
    overlap = np.asarray(overlap_matrix, dtype=np.complex128)
    if overlap.shape == (1, 1):
        return _unit_complex(complex(overlap[0, 0]), atol=atol)
    if method == "determinant":
        det = complex(np.linalg.det(overlap))
        return _unit_complex(det, atol=atol)
    if method != "polar":
        raise ValueError(f"Unknown link_method {method!r}")
    u, singular_values, vh = np.linalg.svd(overlap, full_matrices=False)
    unitary = u @ vh
    det = complex(np.linalg.det(unitary))
    unit, _ = _unit_complex(det, atol=atol)
    return unit, float(np.min(singular_values))


def _compute_link(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    *,
    method: LinkMethod,
    atol: float,
    regularization: float,
) -> tuple[complex, float]:
    left = np.asarray(left_vectors, dtype=np.complex128)
    right = np.asarray(right_vectors, dtype=np.complex128)
    if left.shape != right.shape:
        raise ValueError(f"Link endpoints must have matching shapes, got {left.shape} and {right.shape}")
    if left.ndim == 1:
        left = left[:, np.newaxis]
        right = right[:, np.newaxis]
    if left.ndim != 2:
        raise ValueError(f"Expected endpoint vectors with shape (basis, n_state), got {left.shape}")
    overlap = left.conj().T @ right
    if regularization > 0.0 and overlap.shape[0] > 1:
        overlap = overlap + float(regularization) * np.eye(overlap.shape[0], dtype=np.complex128)
    return _link_from_overlap(overlap, method=method, atol=atol, regularization=regularization)


def _normalize_sewing_transforms(
    sewing_transforms: Sequence[SewingTransform | None] | None,
) -> tuple[SewingTransform | None, SewingTransform | None]:
    if sewing_transforms is None:
        return None, None
    if len(sewing_transforms) != 2:
        raise ValueError("Expected two sewing transforms, one for each mesh direction.")
    return sewing_transforms[0], sewing_transforms[1]


def compute_link_variables(
    selected_vectors: np.ndarray,
    *,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    link_method: LinkMethod = "polar",
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
) -> LinkVariables:
    """Compute normalized FHS/Wilson links on a 2D torus mesh.

    Boundary sewing transforms are target-side maps applied only to wrapped
    forward links, e.g. ``last -> 0`` along each mesh axis.
    """

    selected, _ = select_wavefunction_subspace(selected_vectors, None)
    mesh_1, mesh_2 = selected.shape[:2]
    sew_1, sew_2 = _normalize_sewing_transforms(sewing_transforms)
    link_1 = np.zeros((mesh_1, mesh_2), dtype=np.complex128)
    link_2 = np.zeros((mesh_1, mesh_2), dtype=np.complex128)
    min_link = float("inf")
    for i in range(mesh_1):
        ip = (i + 1) % mesh_1
        for j in range(mesh_2):
            jp = (j + 1) % mesh_2
            left = selected[i, j]
            target_1 = selected[ip, j]
            if i == mesh_1 - 1 and sew_1 is not None:
                target_1 = sew_1(target_1)
            value_1, magnitude_1 = _compute_link(
                left,
                target_1,
                method=link_method,
                atol=atol,
                regularization=regularization,
            )
            link_1[i, j] = value_1
            min_link = min(min_link, float(magnitude_1))
            target_2 = selected[i, jp]
            if j == mesh_2 - 1 and sew_2 is not None:
                target_2 = sew_2(target_2)
            value_2, magnitude_2 = _compute_link(
                left,
                target_2,
                method=link_method,
                atol=atol,
                regularization=regularization,
            )
            link_2[i, j] = value_2
            min_link = min(min_link, float(magnitude_2))
    return LinkVariables(link_1=link_1, link_2=link_2, min_link_magnitude=float(min_link))


def berry_curvature_from_links(link_1: np.ndarray, link_2: np.ndarray) -> np.ndarray:
    """Return FHS Berry plaquette flux from normalized link variables.

    The returned array is the dimensionless plaquette phase in ``(-pi, pi]``.
    Chern number is obtained only by summing this flux and dividing by ``2π``.
    """

    u1 = np.asarray(link_1, dtype=np.complex128)
    u2 = np.asarray(link_2, dtype=np.complex128)
    if u1.shape != u2.shape or u1.ndim != 2:
        raise ValueError(f"Expected two rank-2 link arrays with the same shape, got {u1.shape} and {u2.shape}")
    mesh_1, mesh_2 = u1.shape
    curvature = np.zeros((mesh_1, mesh_2), dtype=float)
    for i in range(mesh_1):
        ip = (i + 1) % mesh_1
        for j in range(mesh_2):
            jp = (j + 1) % mesh_2
            plaquette = u1[i, j] * u2[ip, j] * u1[i, jp].conjugate() * u2[i, j].conjugate()
            curvature[i, j] = float(np.angle(plaquette))
    return curvature


def chern_number_from_berry_curvature(berry_curvature: np.ndarray) -> float:
    """Integrate FHS Berry plaquette flux to a Chern number."""

    return float(np.sum(np.asarray(berry_curvature, dtype=float)) / (2.0 * np.pi))


def fhs_state_from_wavefunctions(
    wavefunctions: np.ndarray,
    state_indices: int | Iterable[int] | None = None,
    *,
    k_grid_frac: np.ndarray | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    basis_sewing: BlockSewingSpec | None = None,
    link_method: LinkMethod = "polar",
    orientation_sign: float = 1.0,
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
    system: str | None = None,
    valley: int | None = None,
    labels: Iterable[str] = (),
    metadata: Mapping[str, object] | None = None,
    reported_indices: int | Iterable[int] | None = None,
) -> FHSState:
    """Package wavefunctions and sewing metadata as the sole FHS input state."""

    return FHSState(
        wavefunctions=np.asarray(wavefunctions, dtype=np.complex128),
        state_indices=None if state_indices is None else normalize_state_indices(state_indices),
        k_grid_frac=None if k_grid_frac is None else np.asarray(k_grid_frac, dtype=float),
        sewing_transforms=sewing_transforms,
        basis_sewing=basis_sewing,
        link_method=link_method,
        orientation_sign=float(orientation_sign),
        atol=float(atol),
        regularization=float(regularization),
        system=None if system is None else str(system),
        valley=None if valley is None else int(valley),
        labels=tuple(str(label) for label in labels),
        reported_indices=None if reported_indices is None else normalize_state_indices(reported_indices),
        metadata={} if metadata is None else dict(metadata),
    )


def fhs_state_from_grid_result(
    grid_result: object,
    state_indices: int | Iterable[int],
    *,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    basis_sewing: BlockSewingSpec | None = None,
    link_method: LinkMethod = "polar",
    orientation_sign: float = 1.0,
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
    system: str | None = None,
    valley: int | None = None,
    labels: Iterable[str] = (),
    metadata: Mapping[str, object] | None = None,
) -> FHSState:
    """Build an FHS state from a grid object with eigenvectors and band labels.

    If ``grid_result.band_indices`` is present, ``state_indices`` are interpreted
    as physical band labels and mapped to eigenvector columns here in the common
    layer.  System modules should not duplicate this mapping.
    """

    eigenvectors = getattr(grid_result, "eigenvectors", None)
    if eigenvectors is None:
        raise ValueError("Grid eigenvectors are required for FHS topology. Recompute with return_eigenvectors=True.")
    requested = normalize_state_indices(state_indices)
    available = tuple(int(index) for index in getattr(grid_result, "band_indices", ()) or ())
    if available:
        positions = {band: pos for pos, band in enumerate(available)}
        missing = [band for band in requested if band not in positions]
        if missing:
            raise ValueError(f"Requested band labels {missing} are not available in grid_result.band_indices={available}")
        columns = tuple(int(positions[band]) for band in requested)
        reported = requested
        grid_metadata = {
            "absolute_band_indices": list(requested),
            "column_indices": list(columns),
            "grid_result_band_indices": list(available),
        }
    else:
        columns = requested
        reported = requested
        grid_metadata = {"column_indices": list(columns)}
    merged = dict(grid_metadata)
    if metadata:
        merged.update(dict(metadata))
    return fhs_state_from_wavefunctions(
        eigenvectors,
        columns,
        k_grid_frac=getattr(grid_result, "k_grid_frac", None),
        sewing_transforms=sewing_transforms,
        basis_sewing=basis_sewing,
        link_method=link_method,
        orientation_sign=orientation_sign,
        atol=atol,
        regularization=regularization,
        system=system,
        valley=valley,
        labels=labels,
        reported_indices=reported,
        metadata=merged,
    )


def compute_lattice_topology(
    wavefunctions: FHSState | np.ndarray,
    state_indices: int | Iterable[int] | None = None,
    *,
    k_grid_frac: np.ndarray | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    link_method: LinkMethod = "polar",
    orientation_sign: float = 1.0,
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
    system: str | None = None,
    valley: int | None = None,
    labels: Iterable[str] = (),
    metadata: Mapping[str, object] | None = None,
    reported_indices: int | Iterable[int] | None = None,
) -> LatticeTopologyResult:
    """Compute FHS Berry plaquette flux and Chern number for selected states."""

    if isinstance(wavefunctions, FHSState):
        state = wavefunctions
        wavefunctions = state.wavefunctions
        state_indices = state.state_indices
        k_grid_frac = state.k_grid_frac
        sewing_transforms = state.sewing_transforms
        if sewing_transforms is None and state.basis_sewing is not None:
            sewing_transforms = sewing_transforms_from_block_spec(state.basis_sewing)
        link_method = state.link_method
        orientation_sign = state.orientation_sign
        atol = state.atol
        regularization = state.regularization
        system = state.system
        valley = state.valley
        labels = state.labels
        metadata = state.metadata
        reported_indices = state.reported_indices

    selected, normalized = select_wavefunction_subspace(wavefunctions, state_indices)
    reported = None if reported_indices is None else normalize_state_indices(reported_indices)
    mesh_1, mesh_2 = selected.shape[:2]
    if float(orientation_sign) not in {-1.0, 1.0}:
        raise ValueError("orientation_sign must be +1 or -1")
    links = compute_link_variables(
        selected,
        sewing_transforms=sewing_transforms,
        link_method=link_method,
        atol=atol,
        regularization=regularization,
    )
    link_1 = np.asarray(links.link_1, dtype=np.complex128)
    link_2 = np.asarray(links.link_2, dtype=np.complex128)
    if float(orientation_sign) == -1.0:
        link_1 = link_1.conjugate()
        link_2 = link_2.conjugate()
    curvature = berry_curvature_from_links(link_1, link_2)
    chern = chern_number_from_berry_curvature(curvature)
    grid = default_k_grid_frac(mesh_1, mesh_2) if k_grid_frac is None else np.asarray(k_grid_frac, dtype=float)
    return LatticeTopologyResult(
        state_indices=normalized,
        k_grid_frac=grid,
        berry_connection=np.stack((np.angle(link_1), np.angle(link_2)), axis=0),
        berry_curvature=curvature,
        chern_number=float(chern),
        rounded_chern_number=int(np.rint(chern)),
        min_link_magnitude=float(links.min_link_magnitude),
        link_1=link_1,
        link_2=link_2,
        system=None if system is None else str(system),
        valley=None if valley is None else int(valley),
        labels=tuple(str(label) for label in labels),
        reported_indices=reported,
        metadata={} if metadata is None else dict(metadata),
    )


TopologyResult = LatticeTopologyResult

__all__ = [
    "BlockSewingSpec",
    "FHSState",
    "LatticeTopologyResult",
    "LinkMethod",
    "LinkVariables",
    "SewingTransform",
    "TopologyResult",
    "berry_curvature_from_links",
    "chern_number_from_berry_curvature",
    "compute_lattice_topology",
    "compute_link_variables",
    "fhs_state_from_grid_result",
    "fhs_state_from_wavefunctions",
    "default_k_grid_frac",
    "matrix_sewing_transform",
    "normalize_state_indices",
    "select_wavefunction_subspace",
    "sewing_transforms_from_block_spec",
]
