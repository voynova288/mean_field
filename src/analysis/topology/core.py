from __future__ import annotations

"""System-independent lattice Berry-geometry utilities.

The routines in this module know only about wavefunctions on a two-dimensional
momentum mesh.  System-specific code is responsible for producing those
wavefunctions, choosing/labeling the state columns, and supplying any boundary
sewing map needed to compare states across the Brillouin-zone torus.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np

LinkMethod = Literal["polar", "determinant"]
SewingTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class WavefunctionIndex:
    """Metadata that identifies which wavefunction columns were used.

    The numerical FHS calculation is system independent once a selected
    wavefunction array is available.  This record preserves the physical meaning
    of the selected columns: ordinary Hamiltonian band indices, a Chern-basis
    label, a flavor index, or any other system-specific state label.
    """

    indices: tuple[int, ...]
    role: str = "band"
    labels: tuple[str, ...] = ()
    system: str | None = None
    valley: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "indices": [int(index) for index in self.indices],
            "role": str(self.role),
            "labels": [str(label) for label in self.labels],
        }
        if self.system is not None:
            payload["system"] = str(self.system)
        if self.valley is not None:
            payload["valley"] = int(self.valley)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class LinkVariables:
    """U(1) link variables for a selected line bundle or subspace."""

    link_1: np.ndarray
    link_2: np.ndarray
    min_link_magnitude: float

    @property
    def berry_connection(self) -> np.ndarray:
        """Discrete Berry connection phases with shape ``(2, mesh_1, mesh_2)``."""

        return np.stack((np.angle(self.link_1), np.angle(self.link_2)), axis=0)


@dataclass(frozen=True)
class LatticeTopologyResult:
    """Berry connection, plaquette flux, and Chern number on a 2D mesh."""

    wavefunction_index: WavefunctionIndex
    k_grid_frac: np.ndarray
    berry_connection: np.ndarray
    berry_curvature: np.ndarray
    chern_number: float
    rounded_chern_number: int
    min_link_magnitude: float
    link_1: np.ndarray
    link_2: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def integer_residual(self) -> float:
        return float(abs(self.chern_number - self.rounded_chern_number))

    @property
    def is_nearly_integer(self) -> bool:
        return self.integer_residual < 1.0e-6

    def to_dict(self) -> dict[str, object]:
        return {
            "wavefunction_index": self.wavefunction_index.to_dict(),
            "chern_number": float(self.chern_number),
            "rounded_chern_number": int(self.rounded_chern_number),
            "integer_residual": float(self.integer_residual),
            "min_link_magnitude": float(self.min_link_magnitude),
            "metadata": dict(self.metadata),
        }


def normalize_state_indices(indices: int | Iterable[int]) -> tuple[int, ...]:
    """Normalize one or more state-column indices and reject ambiguous input."""

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
    """Return the canonical fractional grid ``(i/mesh_1, j/mesh_2)``."""

    grid = np.zeros((int(mesh_1), int(mesh_2), 2), dtype=float)
    grid[:, :, 0] = np.arange(int(mesh_1), dtype=float)[:, None] / float(mesh_1)
    grid[:, :, 1] = np.arange(int(mesh_2), dtype=float)[None, :] / float(mesh_2)
    return grid


def matrix_sewing_transform(matrix: np.ndarray) -> SewingTransform:
    """Build a boundary sewing transform from a basis-space matrix."""

    sewing_matrix = np.asarray(matrix, dtype=np.complex128)
    if sewing_matrix.ndim != 2 or sewing_matrix.shape[0] != sewing_matrix.shape[1]:
        raise ValueError(f"Expected a square sewing matrix, got shape {sewing_matrix.shape}")

    def apply(vectors: np.ndarray) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.complex128)
        if array.shape[0] != sewing_matrix.shape[1]:
            raise ValueError(
                f"Sewing matrix dimension {sewing_matrix.shape} is incompatible with vector shape {array.shape}"
            )
        return sewing_matrix @ array

    return apply


def select_wavefunction_subspace(
    wavefunctions: np.ndarray,
    state_indices: int | Iterable[int] | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Select a line bundle/subspace from a wavefunction grid.

    Accepted shapes are ``(mesh_1, mesh_2, basis_dim)`` for an already-selected
    line bundle and ``(mesh_1, mesh_2, basis_dim, n_states)`` for one or more
    available state columns.
    """

    vectors = np.asarray(wavefunctions, dtype=np.complex128)
    if vectors.ndim == 3:
        if state_indices is not None:
            normalized = normalize_state_indices(state_indices)
            if normalized != (0,):
                raise ValueError(
                    "A rank-3 wavefunction grid is already a line bundle; only state index 0 can be selected."
                )
        return vectors[:, :, :, np.newaxis], (0,)
    if vectors.ndim != 4:
        raise ValueError(
            "Expected wavefunctions with shape (mesh_1, mesh_2, basis_dim) or "
            f"(mesh_1, mesh_2, basis_dim, n_states), got shape {vectors.shape}"
        )

    if state_indices is None:
        normalized = tuple(range(int(vectors.shape[-1])))
    else:
        normalized = normalize_state_indices(state_indices)
    if max(normalized) >= vectors.shape[-1]:
        raise ValueError(
            f"State index {max(normalized)} exceeds the available state count {vectors.shape[-1]}"
        )
    return np.take(vectors, normalized, axis=-1), normalized


def _unit_complex(value: complex, *, atol: float) -> tuple[complex, float]:
    magnitude = abs(complex(value))
    if magnitude <= atol:
        raise ValueError(
            "Encountered a near-zero overlap link while building the Berry-geometry plaquette field. "
            "The selected line bundle or subspace is likely not isolated on this grid."
        )
    return complex(value) / magnitude, float(magnitude)


def _link_from_overlap(
    overlap_matrix: np.ndarray,
    *,
    method: LinkMethod,
    atol: float,
    regularization: float,
) -> tuple[complex, float]:
    if overlap_matrix.shape == (1, 1):
        return _unit_complex(complex(overlap_matrix[0, 0]), atol=atol)

    if method == "determinant":
        determinant = complex(np.linalg.det(overlap_matrix))
        return _unit_complex(determinant, atol=atol)

    if method != "polar":
        raise ValueError("link_method must be 'polar' or 'determinant'")

    singular_values = np.linalg.svd(overlap_matrix, compute_uv=False)
    min_singular = float(np.min(singular_values)) if singular_values.size else 0.0
    working = np.asarray(overlap_matrix, dtype=np.complex128)
    if min_singular <= atol:
        if regularization <= 0.0:
            raise ValueError(
                "Encountered a near-singular subspace overlap while building the Berry-geometry plaquette field. "
                "Try a finer mesh, a shifted mesh, or enable a small regularization."
            )
        working = working + float(regularization) * np.eye(working.shape[0], dtype=np.complex128)
    u_mat, _, vh_mat = np.linalg.svd(working, full_matrices=False)
    phase_matrix = u_mat @ vh_mat
    phase_det = complex(np.linalg.det(phase_matrix))
    link, _ = _unit_complex(phase_det, atol=atol)
    return link, min_singular


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
    if left.ndim == 1:
        left = left[:, np.newaxis]
    if right.ndim == 1:
        right = right[:, np.newaxis]
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"Expected selected vectors to be rank 1 or 2, got {left.shape} and {right.shape}")
    if left.shape != right.shape:
        raise ValueError(f"Cannot link subspaces with different shapes: {left.shape} and {right.shape}")
    overlap = left.conjugate().T @ right
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
    """Compute normalized U(1) links for a selected subspace on a torus mesh.

    ``selected_vectors`` must have shape ``(mesh_1, mesh_2, basis_dim, n_sel)``
    or ``(mesh_1, mesh_2, basis_dim)``.  Boundary sewing transforms are applied
    only when the forward link wraps from the last mesh point back to zero.
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
    """Return the FHS Berry flux through each plaquette.

    The output is a dimensionless phase in ``(-pi, pi]`` per plaquette; summing
    it and dividing by ``2*pi`` gives the Chern number in the same orientation as
    the two mesh directions.
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
    """Integrate dimensionless plaquette Berry flux to a Chern number."""

    return float(np.sum(np.asarray(berry_curvature, dtype=float)) / (2.0 * np.pi))


def compute_lattice_topology(
    wavefunctions: np.ndarray,
    state_indices: int | Iterable[int] | None = None,
    *,
    index: WavefunctionIndex | None = None,
    k_grid_frac: np.ndarray | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    link_method: LinkMethod = "polar",
    orientation_sign: float = 1.0,
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
    metadata: Mapping[str, object] | None = None,
) -> LatticeTopologyResult:
    """Compute discrete Berry connection, plaquette flux, and Chern number.

    All system-specific meaning is carried by ``index`` and optional boundary
    sewing transforms.  Without sewing transforms the function assumes the input
    wavefunction gauge is already periodic on the mesh torus.
    """

    selected, normalized = select_wavefunction_subspace(wavefunctions, state_indices)
    mesh_1, mesh_2 = selected.shape[:2]
    resolved_index = index if index is not None else WavefunctionIndex(indices=normalized)

    links = compute_link_variables(
        selected,
        sewing_transforms=sewing_transforms,
        link_method=link_method,
        atol=atol,
        regularization=regularization,
    )
    curvature = berry_curvature_from_links(links.link_1, links.link_2)
    if orientation_sign != 1.0:
        curvature = float(orientation_sign) * curvature
    chern = chern_number_from_berry_curvature(curvature)
    grid = default_k_grid_frac(mesh_1, mesh_2) if k_grid_frac is None else np.asarray(k_grid_frac, dtype=float)

    return LatticeTopologyResult(
        wavefunction_index=resolved_index,
        k_grid_frac=grid,
        berry_connection=links.berry_connection,
        berry_curvature=curvature,
        chern_number=float(chern),
        rounded_chern_number=int(np.rint(chern)),
        min_link_magnitude=float(links.min_link_magnitude),
        link_1=links.link_1,
        link_2=links.link_2,
        metadata={} if metadata is None else dict(metadata),
    )
