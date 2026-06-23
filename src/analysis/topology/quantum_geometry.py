from __future__ import annotations
"""Gauge-invariant quantum-geometry utilities for two-dimensional meshes.

This module extends the FHS Chern framework with projector/subspace quantum
geometry.  It intentionally depends only on wavefunctions on a 2D mesh plus
optional boundary sewing.  System-specific code is still responsible for
building the wavefunctions and labelling which columns are band/flavor states.
"""
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Sequence
import numpy as np
from .core import (
    LinkMethod,
    SewingTransform,
    WavefunctionIndex,
    compute_lattice_topology,
    default_k_grid_frac,
    normalize_state_indices,
    select_wavefunction_subspace,
)
FiniteDifferenceMethod = Literal["forward", "central"]
CoordinateSystem = Literal["fractional", "cartesian"]
@dataclass(frozen=True)
class NormalizedQuantumGeometryMaps:
    """Paper-style normalized Berry/FS maps for a 2D quantum-geometry result.

    The normalized convention is ``A_BZ * quantity / (2*pi)``.  For Berry
    curvature an optional ``berry_sign`` records the relation between the
    framework convention and a paper/reference convention.  The integrated
    numbers are uniform-sample BZ averages of the normalized maps, so a uniform
    ``C=1`` Berry curvature averages to one.
    """
    quantum_metric: np.ndarray
    fubini_study_trace: np.ndarray
    berry_curvature: np.ndarray
    trace_condition_violation: np.ndarray
    integrated_berry_curvature: float
    integrated_fubini_study_trace: float
    average_trace_condition_violation: float
    bz_area: float
    berry_sign: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict)
    def to_dict(self) -> dict[str, object]:
        """Return a compact JSON-serializable summary."""
        return {
            "bz_area": float(self.bz_area),
            "berry_sign": float(self.berry_sign),
            "integrated_berry_curvature": float(self.integrated_berry_curvature),
            "integrated_fubini_study_trace": float(self.integrated_fubini_study_trace),
            "average_trace_condition_violation": float(self.average_trace_condition_violation),
            "berry_curvature_shape": list(np.asarray(self.berry_curvature).shape),
            "fubini_study_trace_shape": list(np.asarray(self.fubini_study_trace).shape),
            "quantum_metric_shape": list(np.asarray(self.quantum_metric).shape),
            "metadata": dict(self.metadata),
        }
@dataclass(frozen=True)
class QuantumGeometryResult:
    """Quantum geometric tensor, metric, Berry curvature, and Chern data.

    Component arrays use the convention ``(component_a, component_b, mesh_1,
    mesh_2)`` for rank-two tensors.  The quantum geometric tensor is

    ``Q_ab = g_ab + i Omega_ab / 2``

    with Berry-curvature convention

    ``Omega_12 = -i Tr[P [d_1 P, d_2 P]]``.

    The default coordinate system is fractional moire/BZ coordinates.  If
    ``coordinate_system='cartesian'`` was requested, metric and Berry-curvature
    components have been transformed with the supplied reciprocal-vector matrix.
    The FHS curvature, when present, remains a plaquette flux.
    """
    wavefunction_index: WavefunctionIndex
    k_grid_frac: np.ndarray
    quantum_geometric_tensor: np.ndarray
    quantum_metric: np.ndarray
    berry_curvature_density: np.ndarray
    coordinate_system: str
    derivative_coordinates: str
    coordinate_steps: tuple[float, float]
    projector_chern_number: float
    fhs_chern_number: float | None = None
    fhs_berry_curvature: np.ndarray | None = None
    fhs_berry_connection: np.ndarray | None = None
    min_link_magnitude: float | None = None
    reciprocal_vectors: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    @property
    def trace_metric(self) -> np.ndarray:
        """Return ``g_11 + g_22`` on the mesh."""
        return np.asarray(self.quantum_metric[0, 0] + self.quantum_metric[1, 1], dtype=float)
    @property
    def fubini_study_metric(self) -> np.ndarray:
        """Alias for the real quantum metric ``g_ab``."""
        return self.quantum_metric
    @property
    def fubini_study_trace(self) -> np.ndarray:
        """Return the Fubini-Study trace ``tr g_FS`` on the mesh."""
        return self.trace_metric
    @property
    def determinant_metric(self) -> np.ndarray:
        """Return ``det(g)`` on the mesh."""
        metric = np.asarray(self.quantum_metric, dtype=float)
        return metric[0, 0] * metric[1, 1] - metric[0, 1] * metric[1, 0]
    @property
    def trace_condition_residual(self) -> np.ndarray:
        """Return ``tr(g) - |Omega_12|``.

        This is the common ideal-isotropic trace-condition diagnostic.  It is
        physically meaningful only in an orthonormal momentum-coordinate basis
        such as Cartesian k coordinates.
        """
        return self.trace_metric - np.abs(self.berry_curvature_density)
    @property
    def determinant_condition_residual(self) -> np.ndarray:
        """Return ``det(g) - Omega_12^2 / 4``.

        This coordinate-sensitive diagnostic should be interpreted in the same
        coordinate convention as ``quantum_metric`` and ``berry_curvature_density``.
        """
        return self.determinant_metric - 0.25 * np.asarray(self.berry_curvature_density, dtype=float) ** 2
    @property
    def momentum_area_element(self) -> float:
        """Return the area represented by one mesh cell in the current coordinates.

        For ordinary torus meshes, finite-difference spacings and sampling-cell
        spacings are the same.  For isolated local patches, ``coordinate_steps``
        are derivative offsets, not integration weights; use
        :func:`normalize_quantum_geometry_maps` and explicit sample averaging
        for such paper-map workflows.
        """
        h1, h2 = self.coordinate_steps
        if self.coordinate_system == "cartesian":
            if self.derivative_coordinates == "fractional" and self.reciprocal_vectors is not None:
                return float(brillouin_zone_area(self.reciprocal_vectors) * h1 * h2)
            return float(h1 * h2)
        return float(h1 * h2)
    @property
    def normalized_berry_curvature(self) -> np.ndarray | None:
        """Return paper-style curvature with uniform ``C=1`` equal to one.

        This needs Cartesian reciprocal vectors.  For a scalar Berry curvature
        density ``Omega(k)``, the returned field is ``A_BZ Omega(k)/(2*pi)``.
        """
        if self.reciprocal_vectors is None:
            return None
        area_bz = brillouin_zone_area(self.reciprocal_vectors)
        return normalized_chern_density(self.berry_curvature_density, area_bz)
    @property
    def normalized_fubini_study_trace(self) -> np.ndarray | None:
        """Return ``A_BZ tr(g_FS)/(2*pi)`` for paper-style map comparison."""
        if self.reciprocal_vectors is None:
            return None
        area_bz = brillouin_zone_area(self.reciprocal_vectors)
        return normalized_chern_density(self.fubini_study_trace, area_bz)
    @property
    def average_trace_condition_violation(self) -> float | None:
        """Return the average of normalized ``tr(g_FS)-|Omega|`` over the mesh."""
        normalized_trace = self.normalized_fubini_study_trace
        normalized_curvature = self.normalized_berry_curvature
        if normalized_trace is None or normalized_curvature is None:
            return None
        return float(np.mean(normalized_trace - np.abs(normalized_curvature)))
    @property
    def integrated_fubini_study_metric(self) -> float | None:
        """Return ``G = int_BZ tr(g_FS) d^2k / (2*pi)`` when meaningful.

        This is the integrated Fubini-Study metric used as a Wannier-spread
        lower-bound diagnostic in the R5G/hBN papers.  It is only reported for
        Cartesian metric components with reciprocal vectors attached.
        """
        if self.coordinate_system != "cartesian" or self.reciprocal_vectors is None:
            return None
        return integrated_fubini_study_metric(self.fubini_study_trace, area_element=self.momentum_area_element)
    def normalized_maps(
        self,
        *,
        bz_area: float | None = None,
        berry_sign: float = 1.0,
        metadata: Mapping[str, object] | None = None,
    ) -> NormalizedQuantumGeometryMaps:
        """Return ``A_BZ/(2*pi)`` normalized Berry/FS maps.

        Use ``berry_sign`` to convert from the framework convention to a paper
        convention, e.g. ``berry_sign=-1`` for the Zhang2025 tMoTe2 Fig. 3
        checkpoint.
        """
        return normalize_quantum_geometry_maps(
            self,
            bz_area=bz_area,
            berry_sign=berry_sign,
            metadata=metadata,
        )
    def to_dict(self) -> dict[str, object]:
        """Return a compact JSON-serializable summary."""
        payload: dict[str, object] = {
            "wavefunction_index": self.wavefunction_index.to_dict(),
            "coordinate_system": str(self.coordinate_system),
            "derivative_coordinates": str(self.derivative_coordinates),
            "coordinate_steps": [float(self.coordinate_steps[0]), float(self.coordinate_steps[1])],
            "projector_chern_number": float(self.projector_chern_number),
            "metadata": dict(self.metadata),
            "quantum_metric_shape": list(self.quantum_metric.shape),
            "berry_curvature_density_shape": list(self.berry_curvature_density.shape),
            "trace_metric_min": float(np.min(self.trace_metric)),
            "trace_metric_max": float(np.max(self.trace_metric)),
            "determinant_metric_min": float(np.min(self.determinant_metric)),
            "determinant_metric_max": float(np.max(self.determinant_metric)),
            "trace_condition_residual_min": float(np.min(self.trace_condition_residual)),
            "determinant_condition_residual_min": float(np.min(self.determinant_condition_residual)),
        }
        if self.integrated_fubini_study_metric is not None:
            payload["integrated_fubini_study_metric"] = float(self.integrated_fubini_study_metric)
        if self.average_trace_condition_violation is not None:
            payload["average_trace_condition_violation"] = float(self.average_trace_condition_violation)
        if self.fhs_chern_number is not None:
            payload["fhs_chern_number"] = float(self.fhs_chern_number)
        if self.min_link_magnitude is not None:
            payload["min_link_magnitude"] = float(self.min_link_magnitude)
        if self.reciprocal_vectors is not None:
            payload["reciprocal_vectors"] = np.asarray(self.reciprocal_vectors, dtype=float).tolist()
        return payload
def _resolve_coordinate_steps(
    mesh_1: int,
    mesh_2: int,
    k_grid_frac: np.ndarray | None,
    coordinate_steps: Sequence[float] | None,
) -> tuple[float, float]:
    if coordinate_steps is not None:
        if len(coordinate_steps) != 2:
            raise ValueError("coordinate_steps must contain exactly two positive spacings")
        h1, h2 = float(coordinate_steps[0]), float(coordinate_steps[1])
        if h1 <= 0.0 or h2 <= 0.0:
            raise ValueError(f"coordinate_steps must be positive, got {(h1, h2)}")
        return h1, h2
    if k_grid_frac is None:
        return 1.0 / float(mesh_1), 1.0 / float(mesh_2)
    grid = np.asarray(k_grid_frac, dtype=float)
    if grid.shape != (mesh_1, mesh_2, 2):
        raise ValueError(
            f"k_grid_frac must have shape {(mesh_1, mesh_2, 2)}, got {grid.shape}"
        )
    if mesh_1 > 1:
        diffs_1 = np.diff(grid[:, :, 0], axis=0)
        h1 = float(np.mean(np.abs(diffs_1))) if diffs_1.size else 1.0 / float(mesh_1)
    else:
        h1 = 1.0
    if mesh_2 > 1:
        diffs_2 = np.diff(grid[:, :, 1], axis=1)
        h2 = float(np.mean(np.abs(diffs_2))) if diffs_2.size else 1.0 / float(mesh_2)
    else:
        h2 = 1.0
    if h1 <= 0.0 or h2 <= 0.0:
        return 1.0 / float(mesh_1), 1.0 / float(mesh_2)
    return h1, h2
def _orthonormalize_one(frame: np.ndarray, *, atol: float) -> np.ndarray:
    matrix = np.asarray(frame, dtype=np.complex128)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    if matrix.ndim != 2:
        raise ValueError(f"Expected a frame with shape (basis_dim, n_state), got {matrix.shape}")
    basis_dim, n_state = matrix.shape
    if n_state <= 0:
        raise ValueError("Expected at least one selected state")
    if basis_dim < n_state:
        raise ValueError(f"basis_dim={basis_dim} is smaller than selected subspace dimension {n_state}")
    if n_state == 1:
        norm = float(np.linalg.norm(matrix[:, 0]))
        if norm <= atol:
            raise ValueError("Encountered a near-zero selected wavefunction while orthonormalizing frames")
        return matrix / norm
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or float(np.min(singular_values)) <= atol:
        raise ValueError(
            "Selected wavefunction columns are rank deficient; the band/subspace is not well defined on this mesh"
        )
    q_matrix, _ = np.linalg.qr(matrix, mode="reduced")
    return np.asarray(q_matrix[:, :n_state], dtype=np.complex128)
def orthonormalize_wavefunction_frames(selected_vectors: np.ndarray, *, atol: float = 1.0e-12) -> np.ndarray:
    """Return orthonormal frames for ``(mesh_1, mesh_2, basis, n_sel)`` data."""
    selected, _ = select_wavefunction_subspace(selected_vectors, None)
    mesh_1, mesh_2, basis_dim, n_state = selected.shape
    frames = np.empty((mesh_1, mesh_2, basis_dim, n_state), dtype=np.complex128)
    if n_state == 1:
        norms = np.linalg.norm(selected[:, :, :, 0], axis=2)
        min_norm = float(np.min(norms))
        if min_norm <= atol:
            raise ValueError("Encountered a near-zero selected wavefunction while normalizing a line bundle")
        frames[:, :, :, 0] = selected[:, :, :, 0] / norms[:, :, np.newaxis]
        return frames
    for i in range(mesh_1):
        for j in range(mesh_2):
            frames[i, j] = _orthonormalize_one(selected[i, j], atol=atol)
    return frames
def _normalize_sewing_transforms(
    sewing_transforms: Sequence[SewingTransform | None] | None,
) -> tuple[SewingTransform | None, SewingTransform | None]:
    if sewing_transforms is None:
        return None, None
    if len(sewing_transforms) != 2:
        raise ValueError("Expected two sewing transforms, one for each mesh direction")
    return sewing_transforms[0], sewing_transforms[1]
def _forward_frames(
    frames: np.ndarray,
    *,
    sewing_transforms: Sequence[SewingTransform | None] | None,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    sew_1, sew_2 = _normalize_sewing_transforms(sewing_transforms)
    mesh_1, mesh_2 = frames.shape[:2]
    forward_1 = np.roll(frames, shift=-1, axis=0)
    forward_2 = np.roll(frames, shift=-1, axis=1)
    if sew_1 is not None:
        for j in range(mesh_2):
            forward_1[mesh_1 - 1, j] = _orthonormalize_one(sew_1(frames[0, j]), atol=atol)
    if sew_2 is not None:
        for i in range(mesh_1):
            forward_2[i, mesh_2 - 1] = _orthonormalize_one(sew_2(frames[i, 0]), atol=atol)
    return forward_1, forward_2
def _backward_frames(
    frames: np.ndarray,
    *,
    backward_sewing_transforms: Sequence[SewingTransform | None] | None,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    sew_1, sew_2 = _normalize_sewing_transforms(backward_sewing_transforms)
    mesh_1, mesh_2 = frames.shape[:2]
    backward_1 = np.roll(frames, shift=1, axis=0)
    backward_2 = np.roll(frames, shift=1, axis=1)
    if sew_1 is not None:
        for j in range(mesh_2):
            backward_1[0, j] = _orthonormalize_one(sew_1(frames[mesh_1 - 1, j]), atol=atol)
    if sew_2 is not None:
        for i in range(mesh_1):
            backward_2[i, 0] = _orthonormalize_one(sew_2(frames[i, mesh_2 - 1]), atol=atol)
    return backward_1, backward_2
def _frobenius_overlap_squared(left: np.ndarray, right: np.ndarray) -> float:
    overlap = left.conjugate().T @ right
    return float(np.sum(np.abs(overlap) ** 2))
def _triple_projector_trace(q0: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> complex:
    return complex(np.trace((q0.conjugate().T @ q1) @ (q1.conjugate().T @ q2) @ (q2.conjugate().T @ q0)))
def projector_qgt_forward_difference(
    selected_vectors: np.ndarray,
    *,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    coordinate_steps: Sequence[float] | None = None,
    k_grid_frac: np.ndarray | None = None,
    atol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute a gauge-invariant projector QGT by forward finite differences.

    The implementation never differentiates raw eigenvector phases.  It uses
    only projectors, represented through small overlap matrices between local
    orthonormal frames.  This keeps the calculation invariant under arbitrary
    U(1) phases or U(N) frame rotations of the selected subspace.

    Returns ``(qgt, metric, omega_12, projector_chern)`` in the input coordinate
    convention.  ``qgt`` and ``metric`` have shape ``(2, 2, mesh_1, mesh_2)``;
    ``omega_12`` has shape ``(mesh_1, mesh_2)``.
    """
    selected, _ = select_wavefunction_subspace(selected_vectors, None)
    mesh_1, mesh_2 = selected.shape[:2]
    h1, h2 = _resolve_coordinate_steps(mesh_1, mesh_2, k_grid_frac, coordinate_steps)
    frames = orthonormalize_wavefunction_frames(selected, atol=atol)
    forward_1, forward_2 = _forward_frames(frames, sewing_transforms=sewing_transforms, atol=atol)
    qgt = np.zeros((2, 2, mesh_1, mesh_2), dtype=np.complex128)
    metric = np.zeros((2, 2, mesh_1, mesh_2), dtype=float)
    omega_12 = np.zeros((mesh_1, mesh_2), dtype=float)
    rank = int(frames.shape[-1])
    for i in range(mesh_1):
        for j in range(mesh_2):
            q0 = frames[i, j]
            q1 = forward_1[i, j]
            q2 = forward_2[i, j]
            tr_p0p1 = _frobenius_overlap_squared(q0, q1)
            tr_p0p2 = _frobenius_overlap_squared(q0, q2)
            tr_p1p2 = _frobenius_overlap_squared(q1, q2)
            g11 = (float(rank) - tr_p0p1) / (h1 * h1)
            g22 = (float(rank) - tr_p0p2) / (h2 * h2)
            g12 = 0.5 * (tr_p1p2 - tr_p0p1 - tr_p0p2 + float(rank)) / (h1 * h2)
            tr_012 = _triple_projector_trace(q0, q1, q2)
            tr_021 = _triple_projector_trace(q0, q2, q1)
            omega = (-1j * (tr_012 - tr_021) / (h1 * h2)).real
            metric[0, 0, i, j] = g11
            metric[1, 1, i, j] = g22
            metric[0, 1, i, j] = g12
            metric[1, 0, i, j] = g12
            omega_12[i, j] = float(omega)
            qgt[0, 0, i, j] = complex(g11)
            qgt[1, 1, i, j] = complex(g22)
            qgt[0, 1, i, j] = complex(g12, 0.5 * float(omega))
            qgt[1, 0, i, j] = complex(g12, -0.5 * float(omega))
    projector_chern = float(np.sum(omega_12) * h1 * h2 / (2.0 * np.pi))
    return qgt, metric, omega_12, projector_chern
def projector_qgt_central_difference(
    selected_vectors: np.ndarray,
    *,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    backward_sewing_transforms: Sequence[SewingTransform | None] | None = None,
    coordinate_steps: Sequence[float] | None = None,
    k_grid_frac: np.ndarray | None = None,
    atol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute projector QGT with symmetric finite differences.

    This estimator is better suited for plotting local Berry-curvature and
    Fubini-Study metric maps than the one-sided forward stencil.  It still uses
    only projectors/overlaps, so arbitrary U(1) phases or U(N) frame rotations
    do not affect the result.  When nontrivial boundary sewing is supplied,
    ``backward_sewing_transforms`` must supply the inverse transition functions
    needed to represent the ``k-h`` neighbor in the same local chart.
    """
    if sewing_transforms is not None and backward_sewing_transforms is None:
        raise ValueError(
            "central finite differences with boundary sewing require backward_sewing_transforms "
            "for the inverse boundary transition functions"
        )
    selected, _ = select_wavefunction_subspace(selected_vectors, None)
    mesh_1, mesh_2 = selected.shape[:2]
    h1, h2 = _resolve_coordinate_steps(mesh_1, mesh_2, k_grid_frac, coordinate_steps)
    frames = orthonormalize_wavefunction_frames(selected, atol=atol)
    forward_1, forward_2 = _forward_frames(frames, sewing_transforms=sewing_transforms, atol=atol)
    backward_1, backward_2 = _backward_frames(
        frames,
        backward_sewing_transforms=backward_sewing_transforms,
        atol=atol,
    )
    qgt = np.zeros((2, 2, mesh_1, mesh_2), dtype=np.complex128)
    metric = np.zeros((2, 2, mesh_1, mesh_2), dtype=float)
    omega_12 = np.zeros((mesh_1, mesh_2), dtype=float)
    rank = int(frames.shape[-1])
    for i in range(mesh_1):
        for j in range(mesh_2):
            q0 = frames[i, j]
            p1 = forward_1[i, j]
            p2 = forward_2[i, j]
            m1 = backward_1[i, j]
            m2 = backward_2[i, j]
            g11 = (float(rank) - _frobenius_overlap_squared(p1, m1)) / (4.0 * h1 * h1)
            g22 = (float(rank) - _frobenius_overlap_squared(p2, m2)) / (4.0 * h2 * h2)
            cross = (
                _frobenius_overlap_squared(p1, p2)
                - _frobenius_overlap_squared(p1, m2)
                - _frobenius_overlap_squared(m1, p2)
                + _frobenius_overlap_squared(m1, m2)
            )
            g12 = 0.5 * cross / (4.0 * h1 * h2)
            commutator_trace = (
                _triple_projector_trace(q0, p1, p2)
                - _triple_projector_trace(q0, p1, m2)
                - _triple_projector_trace(q0, m1, p2)
                + _triple_projector_trace(q0, m1, m2)
                - _triple_projector_trace(q0, p2, p1)
                + _triple_projector_trace(q0, p2, m1)
                + _triple_projector_trace(q0, m2, p1)
                - _triple_projector_trace(q0, m2, m1)
            ) / (4.0 * h1 * h2)
            omega = (-1j * commutator_trace).real
            metric[0, 0, i, j] = g11
            metric[1, 1, i, j] = g22
            metric[0, 1, i, j] = g12
            metric[1, 0, i, j] = g12
            omega_12[i, j] = float(omega)
            qgt[0, 0, i, j] = complex(g11)
            qgt[1, 1, i, j] = complex(g22)
            qgt[0, 1, i, j] = complex(g12, 0.5 * float(omega))
            qgt[1, 0, i, j] = complex(g12, -0.5 * float(omega))
    projector_chern = float(np.sum(omega_12) * h1 * h2 / (2.0 * np.pi))
    return qgt, metric, omega_12, projector_chern
def transform_quantum_geometric_tensor(qgt: np.ndarray, derivative_transform: np.ndarray) -> np.ndarray:
    """Transform QGT components to a new coordinate basis.

    ``derivative_transform[i, a]`` must satisfy
    ``d/d(new_a) = sum_i derivative_transform[i, a] d/d(old_i)``.  For
    fractional coordinates ``f`` and Cartesian momenta ``k = B f``, pass
    ``np.linalg.inv(B)`` where the columns of ``B`` are reciprocal vectors.
    """
    tensor = np.asarray(qgt, dtype=np.complex128)
    transform = np.asarray(derivative_transform, dtype=float)
    if tensor.ndim < 2 or tensor.shape[:2] != (2, 2):
        raise ValueError(f"Expected qgt with leading shape (2, 2), got {tensor.shape}")
    if transform.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 derivative transform, got {transform.shape}")
    return np.einsum("ia,jb,ij...->ab...", transform, transform, tensor)


def qgt_to_metric_and_berry(qgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(metric, omega_12)`` from ``Q_ab = g_ab + i Omega_ab/2``."""

    tensor = np.asarray(qgt, dtype=np.complex128)
    if tensor.ndim < 2 or tensor.shape[:2] != (2, 2):
        raise ValueError(f"Expected qgt with leading shape (2, 2), got {tensor.shape}")
    metric = np.asarray(tensor.real, dtype=float)
    omega_12 = np.asarray(2.0 * tensor[0, 1].imag, dtype=float)
    return metric, omega_12


def reciprocal_vectors_to_derivative_transform(reciprocal_vectors: np.ndarray) -> np.ndarray:
    """Return the derivative transform from fractional to Cartesian k axes.

    ``reciprocal_vectors`` is a ``(2, 2)`` real matrix whose columns are the two
    reciprocal vectors in Cartesian coordinates.
    """

    basis = np.asarray(reciprocal_vectors, dtype=float)
    if basis.shape != (2, 2):
        raise ValueError(f"reciprocal_vectors must have shape (2, 2), got {basis.shape}")
    det = float(np.linalg.det(basis))
    if abs(det) <= 1.0e-15:
        raise ValueError("reciprocal_vectors are singular")
    return np.linalg.inv(basis)


def fubini_study_trace(quantum_metric: np.ndarray) -> np.ndarray:
    """Return ``tr g_FS`` from a metric array with leading shape ``(2, 2)``."""

    metric = np.asarray(quantum_metric, dtype=float)
    if metric.ndim < 2 or metric.shape[:2] != (2, 2):
        raise ValueError(f"Expected quantum_metric with leading shape (2, 2), got {metric.shape}")
    return np.asarray(metric[0, 0] + metric[1, 1], dtype=float)


def brillouin_zone_area(reciprocal_vectors: np.ndarray) -> float:
    """Return ``|det[b1,b2]|`` for a 2D reciprocal-vector matrix.

    ``reciprocal_vectors`` follows the framework convention: columns are the two
    reciprocal basis vectors in the coordinate units of the metric/Berry
    density, e.g. Angstrom^{-1} or bohr^{-1}.
    """

    basis = np.asarray(reciprocal_vectors, dtype=float)
    if basis.shape != (2, 2):
        raise ValueError(f"reciprocal_vectors must have shape (2, 2), got {basis.shape}")
    area = float(abs(np.linalg.det(basis)))
    if area <= 0.0:
        raise ValueError("reciprocal_vectors are singular")
    return area

def normalized_chern_density(density: np.ndarray, bz_area: float) -> np.ndarray:
    """Normalize a density so a uniform ``C=1`` Berry curvature equals one.

    For Berry curvature this is ``A_BZ Omega(k)/(2*pi)``.  The same scaling is
    useful for plotting the Fubini-Study trace beside the normalized curvature,
    as in R5G/hBN Fig. 6.
    """

    area = float(bz_area)
    if area <= 0.0:
        raise ValueError(f"bz_area must be positive, got {bz_area}")
    return np.asarray(density, dtype=float) * area / (2.0 * np.pi)


def infer_berry_sign_from_chern(chern_estimate: float | Sequence[float] | np.ndarray, expected_chern: float | Sequence[float] | np.ndarray) -> float:
    """Infer whether a reference convention uses ``+Omega`` or ``-Omega``.

    Returns ``+1.0`` if ``chern_estimate`` is closer to ``expected_chern`` and
    ``-1.0`` if ``-chern_estimate`` is closer.  This is only a convention helper;
    it should not be used to hide a failed topology check.
    """

    estimate = np.asarray(chern_estimate, dtype=float)
    expected = np.asarray(expected_chern, dtype=float)
    if estimate.shape != expected.shape:
        try:
            estimate, expected = np.broadcast_arrays(estimate, expected)
        except ValueError as exc:
            raise ValueError(f"chern_estimate and expected_chern cannot be broadcast: {estimate.shape}, {expected.shape}") from exc
    plus_error = float(np.linalg.norm(estimate - expected))
    minus_error = float(np.linalg.norm(-estimate - expected))
    return 1.0 if plus_error <= minus_error else -1.0

def normalize_quantum_geometry_maps(
    result: QuantumGeometryResult,
    *,
    bz_area: float | None = None,
    berry_sign: float = 1.0,
    metadata: Mapping[str, object] | None = None,
) -> NormalizedQuantumGeometryMaps:
    """Return normalized Berry/FS maps from a quantum-geometry result.

    This is the common API for paper-style map comparisons such as
    ``A_mBZ*Omega/(2*pi)`` and ``A_mBZ*Tr(g)/(2*pi)``.  It assumes the map
    samples are representative of a uniform BZ/mBZ sampling when reporting the
    integrated/averaged values.
    """

    area = float(bz_area) if bz_area is not None else None
    if area is None:
        if result.reciprocal_vectors is None:
            raise ValueError("bz_area is required when result.reciprocal_vectors is not attached")
        area = brillouin_zone_area(result.reciprocal_vectors)
    if area <= 0.0:
        raise ValueError(f"bz_area must be positive, got {area}")
    sign = float(berry_sign)
    metric = normalized_chern_density(result.quantum_metric, area)
    trace = fubini_study_trace(metric)
    berry = sign * normalized_chern_density(result.berry_curvature_density, area)
    trace_violation = trace - np.abs(berry)
    payload = dict(result.metadata)
    if metadata:
        payload.update(dict(metadata))
    return NormalizedQuantumGeometryMaps(
        quantum_metric=np.asarray(metric, dtype=float),
        fubini_study_trace=np.asarray(trace, dtype=float),
        berry_curvature=np.asarray(berry, dtype=float),
        trace_condition_violation=np.asarray(trace_violation, dtype=float),
        integrated_berry_curvature=float(np.mean(berry)),
        integrated_fubini_study_trace=float(np.mean(trace)),
        average_trace_condition_violation=float(np.mean(trace_violation)),
        bz_area=float(area),
        berry_sign=float(sign),
        metadata=payload,
    )

def integrated_fubini_study_metric(trace_metric: np.ndarray, *, area_element: float) -> float:
    """Return ``G = int_BZ tr(g_FS) d^2k / (2*pi)`` from a sampled trace map."""

    cell_area = float(area_element)
    if cell_area <= 0.0:
        raise ValueError(f"area_element must be positive, got {area_element}")
    return float(np.sum(np.asarray(trace_metric, dtype=float)) * cell_area / (2.0 * np.pi))


def trace_condition_violation(
    trace_metric: np.ndarray,
    berry_curvature_density: np.ndarray,
    *,
    bz_area: float | None = None,
) -> np.ndarray:
    """Return ``tr(g_FS)-|Omega|`` or its normalized paper-style variant."""

    trace = np.asarray(trace_metric, dtype=float)
    curvature = np.asarray(berry_curvature_density, dtype=float)
    if trace.shape != curvature.shape:
        raise ValueError(f"trace and curvature shapes must match, got {trace.shape} and {curvature.shape}")
    if bz_area is None:
        return trace - np.abs(curvature)
    normalized_trace = normalized_chern_density(trace, bz_area)
    normalized_curvature = normalized_chern_density(curvature, bz_area)
    return normalized_trace - np.abs(normalized_curvature)


def compute_quantum_geometry(
    wavefunctions: np.ndarray,
    state_indices: int | Iterable[int] | None = None,
    *,
    index: WavefunctionIndex | None = None,
    k_grid_frac: np.ndarray | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    backward_sewing_transforms: Sequence[SewingTransform | None] | None = None,
    coordinate_steps: Sequence[float] | None = None,
    coordinate_system: CoordinateSystem = "fractional",
    derivative_coordinates: CoordinateSystem = "fractional",
    reciprocal_vectors: np.ndarray | None = None,
    include_fhs: bool = True,
    link_method: LinkMethod = "polar",
    finite_difference: FiniteDifferenceMethod = "forward",
    atol: float = 1.0e-12,
    metadata: Mapping[str, object] | None = None,
) -> QuantumGeometryResult:
    """Compute gauge-invariant quantum geometry for a 2D wavefunction mesh.

    Parameters are intentionally parallel to :func:`compute_lattice_topology`.
    The selected columns may represent a single band or a multi-band/flavor
    subspace.  For nontrivial boundary gauges, pass the same sewing transforms
    used for Chern calculations.  Use ``finite_difference='central'`` for
    paper-style local Fubini-Study/Berry maps; if boundary sewing is nontrivial,
    also pass the inverse maps via ``backward_sewing_transforms``.
    """

    selected, normalized = select_wavefunction_subspace(wavefunctions, state_indices)
    mesh_1, mesh_2 = selected.shape[:2]
    resolved_grid = default_k_grid_frac(mesh_1, mesh_2) if k_grid_frac is None else np.asarray(k_grid_frac, dtype=float)
    resolved_steps = _resolve_coordinate_steps(mesh_1, mesh_2, resolved_grid, coordinate_steps)
    resolved_index = index if index is not None else WavefunctionIndex(indices=normalized)

    if finite_difference == "forward":
        qgt_frac, metric_frac, omega_frac, projector_chern = projector_qgt_forward_difference(
            selected,
            sewing_transforms=sewing_transforms,
            coordinate_steps=resolved_steps,
            k_grid_frac=resolved_grid,
            atol=atol,
        )
    elif finite_difference == "central":
        qgt_frac, metric_frac, omega_frac, projector_chern = projector_qgt_central_difference(
            selected,
            sewing_transforms=sewing_transforms,
            backward_sewing_transforms=backward_sewing_transforms,
            coordinate_steps=resolved_steps,
            k_grid_frac=resolved_grid,
            atol=atol,
        )
    else:
        raise ValueError("finite_difference must be 'forward' or 'central'")

    system = str(coordinate_system)
    derivative_system = str(derivative_coordinates)
    if system not in {"fractional", "cartesian"}:
        raise ValueError("coordinate_system must be 'fractional' or 'cartesian'")
    if derivative_system not in {"fractional", "cartesian"}:
        raise ValueError("derivative_coordinates must be 'fractional' or 'cartesian'")

    qgt = qgt_frac
    metric = metric_frac
    omega = omega_frac
    reciprocal_payload: np.ndarray | None = None
    if derivative_system == "cartesian":
        if system != "cartesian":
            raise ValueError("coordinate_system must be 'cartesian' when derivative_coordinates='cartesian'")
        if reciprocal_vectors is not None:
            reciprocal_payload = np.asarray(reciprocal_vectors, dtype=float)
    elif system == "cartesian":
        if reciprocal_vectors is None:
            raise ValueError("reciprocal_vectors are required when transforming fractional derivatives to cartesian")
        reciprocal_payload = np.asarray(reciprocal_vectors, dtype=float)
        qgt = transform_quantum_geometric_tensor(
            qgt_frac,
            reciprocal_vectors_to_derivative_transform(reciprocal_payload),
        )
        metric, omega = qgt_to_metric_and_berry(qgt)
    elif reciprocal_vectors is not None:
        reciprocal_payload = np.asarray(reciprocal_vectors, dtype=float)

    fhs_chern: float | None = None
    fhs_curvature: np.ndarray | None = None
    fhs_connection: np.ndarray | None = None
    min_link: float | None = None
    if include_fhs:
        fhs = compute_lattice_topology(
            wavefunctions,
            normalized,
            index=resolved_index,
            k_grid_frac=resolved_grid,
            sewing_transforms=sewing_transforms,
            link_method=link_method,
            atol=max(float(atol), 1.0e-14),
            metadata=metadata,
        )
        fhs_chern = float(fhs.chern_number)
        fhs_curvature = fhs.berry_curvature
        fhs_connection = fhs.berry_connection
        min_link = float(fhs.min_link_magnitude)

    return QuantumGeometryResult(
        wavefunction_index=resolved_index,
        k_grid_frac=resolved_grid,
        quantum_geometric_tensor=np.asarray(qgt, dtype=np.complex128),
        quantum_metric=np.asarray(metric, dtype=float),
        berry_curvature_density=np.asarray(omega, dtype=float),
        coordinate_system=system,
        derivative_coordinates=derivative_system,
        coordinate_steps=(float(resolved_steps[0]), float(resolved_steps[1])),
        projector_chern_number=float(projector_chern),
        fhs_chern_number=fhs_chern,
        fhs_berry_curvature=fhs_curvature,
        fhs_berry_connection=fhs_connection,
        min_link_magnitude=min_link,
        reciprocal_vectors=reciprocal_payload,
        metadata={} if metadata is None else dict(metadata),
    )


__all__ = [
    "CoordinateSystem",
    "NormalizedQuantumGeometryMaps",
    "QuantumGeometryResult",
    "brillouin_zone_area",
    "compute_quantum_geometry",
    "fubini_study_trace",
    "infer_berry_sign_from_chern",
    "integrated_fubini_study_metric",
    "normalize_quantum_geometry_maps",
    "normalized_chern_density",
    "orthonormalize_wavefunction_frames",
    "projector_qgt_central_difference",
    "projector_qgt_forward_difference",
    "qgt_to_metric_and_berry",
    "reciprocal_vectors_to_derivative_transform",
    "trace_condition_violation",
    "transform_quantum_geometric_tensor",
]
