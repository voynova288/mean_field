"""Matrix-free local stability diagnostics for fixed-point maps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

Array = np.ndarray
RealMap = Callable[[Array], Array]
VectorProjector = Callable[[Array], Array]


@dataclass(frozen=True)
class FixedPointMapSpectrum:
    eigenvalues: Array
    eigenvectors: Array
    spectral_radius: float
    converged: bool
    root_map_residual_max: float
    matvec_evaluations: int


def leading_fixed_point_map_eigenvalues(
    map_builder: RealMap,
    root_vector: Array,
    *,
    count: int = 4,
    relative_step: float = 1.0e-6,
    tolerance: float = 1.0e-7,
    max_iterations: int = 300,
    projector: VectorProjector | None = None,
) -> FixedPointMapSpectrum:
    """Estimate leading eigenvalues of the Jacobian of ``x -> M(x)``.

    A magnitude above one diagnoses instability of simple fixed-point
    iteration in the requested projected tangent sector.  It is not, by
    itself, a thermodynamic Hessian classification.
    """

    root = np.asarray(root_vector, dtype=np.float64)
    if root.ndim != 1 or not np.all(np.isfinite(root)):
        raise ValueError("root_vector must be one-dimensional and finite")
    if root.size < 3:
        raise ValueError("root_vector must contain at least three coordinates")
    if not 1 <= int(count) < root.size - 1:
        raise ValueError("count must satisfy 1 <= count < dimension-1")
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("eigensolver controls are invalid")

    def apply_projector(vector: Array) -> Array:
        candidate = np.asarray(vector, dtype=np.float64)
        if projector is not None:
            candidate = np.asarray(projector(candidate), dtype=np.float64)
        if candidate.shape != root.shape or not np.all(np.isfinite(candidate)):
            raise ValueError("tangent projector must return a finite matching vector")
        return candidate

    base = np.asarray(map_builder(root), dtype=np.float64)
    if base.shape != root.shape or not np.all(np.isfinite(base)):
        raise ValueError("fixed-point map must return a finite matching vector")
    root_residual = float(np.max(np.abs(base - root), initial=0.0))
    root_scale = max(1.0, float(np.linalg.norm(root)))
    evaluations = 0

    def matvec(direction: Array) -> Array:
        nonlocal evaluations
        tangent = apply_projector(direction)
        norm = float(np.linalg.norm(tangent))
        if norm == 0.0:
            return np.zeros_like(root)
        step = float(relative_step) * root_scale / norm
        mapped = np.asarray(map_builder(root + step * tangent), dtype=np.float64)
        if mapped.shape != root.shape or not np.all(np.isfinite(mapped)):
            raise ValueError("fixed-point map returned an invalid finite-difference value")
        evaluations += 1
        derivative = (mapped - base) / step
        return apply_projector(derivative)

    operator = LinearOperator((root.size, root.size), matvec=matvec, dtype=np.float64)
    converged = True
    try:
        values, vectors = eigs(
            operator,
            k=int(count),
            which="LM",
            tol=float(tolerance),
            maxiter=int(max_iterations),
            return_eigenvectors=True,
        )
    except ArpackNoConvergence as error:
        converged = False
        values = np.asarray(error.eigenvalues, dtype=np.complex128)
        if error.eigenvectors is None:
            vectors = np.zeros((root.size, values.size), dtype=np.complex128)
        else:
            vectors = np.asarray(error.eigenvectors, dtype=np.complex128)
    values = np.asarray(values, dtype=np.complex128)
    vectors = np.asarray(vectors, dtype=np.complex128)
    order = np.argsort(np.abs(values))[::-1]
    values = values[order]
    vectors = vectors[:, order]
    spectral_radius = float(np.max(np.abs(values), initial=0.0))
    return FixedPointMapSpectrum(
        eigenvalues=values,
        eigenvectors=vectors,
        spectral_radius=spectral_radius,
        converged=converged,
        root_map_residual_max=root_residual,
        matvec_evaluations=evaluations,
    )


__all__ = ["FixedPointMapSpectrum", "leading_fixed_point_map_eigenvalues"]
