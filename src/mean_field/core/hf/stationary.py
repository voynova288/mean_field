"""Matrix-free stationary residual solvers independent of energy descent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import NoConvergence, anderson, newton_krylov

Array = np.ndarray
RealResidual = Callable[[Array], Array]
RootLineSearch = Literal["armijo", "wolfe"] | None


def pack_hermitian_matrix_field(field: Array, *, tolerance: float = 1.0e-11) -> Array:
    """Pack a finite Hermitian ``(d,d,nk)`` field into independent real coordinates."""

    matrices = np.asarray(field, dtype=np.complex128)
    if matrices.ndim != 3 or matrices.shape[0] != matrices.shape[1]:
        raise ValueError("field must have shape (dimension,dimension,nk)")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("field must be finite")
    error = float(np.max(np.abs(matrices - np.swapaxes(matrices.conj(), 0, 1)), initial=0.0))
    if error > float(tolerance):
        raise ValueError(f"field is not Hermitian: {error:.3e}")
    dimension, _, nk = matrices.shape
    diagonal = np.diagonal(matrices, axis1=0, axis2=1).real.T.reshape(dimension * nk)
    upper = np.triu_indices(dimension, k=1)
    off_diagonal = matrices[upper[0], upper[1], :]
    return np.concatenate([diagonal, off_diagonal.real.ravel(), off_diagonal.imag.ravel()])


def unpack_hermitian_matrix_field(vector: Array, *, dimension: int, nk: int) -> Array:
    """Inverse of :func:`pack_hermitian_matrix_field`."""

    coordinates = np.asarray(vector, dtype=np.float64)
    dimension = int(dimension)
    nk = int(nk)
    if coordinates.ndim != 1 or dimension < 1 or nk < 1:
        raise ValueError("vector must be one-dimensional and dimensions positive")
    pair_count = dimension * (dimension - 1) // 2
    expected = nk * (dimension + 2 * pair_count)
    if coordinates.size != expected:
        raise ValueError(f"vector length {coordinates.size} does not match {expected}")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("vector must be finite")
    diagonal_end = dimension * nk
    real_end = diagonal_end + pair_count * nk
    diagonal = coordinates[:diagonal_end].reshape(dimension, nk)
    real = coordinates[diagonal_end:real_end].reshape(pair_count, nk)
    imag = coordinates[real_end:].reshape(pair_count, nk)
    field = np.zeros((dimension, dimension, nk), dtype=np.complex128)
    indices = np.arange(dimension)
    field[indices, indices, :] = diagonal
    upper = np.triu_indices(dimension, k=1)
    values = real + 1j * imag
    field[upper[0], upper[1], :] = values
    field[upper[1], upper[0], :] = values.conj()
    return field


@dataclass(frozen=True)
class StationarySolveConfig:
    """Configuration for Anderson initialization plus Newton--Krylov refinement."""

    residual_rms_tolerance: float = 1.0e-9
    residual_max_tolerance: float = 1.0e-8
    anderson_max_iterations: int = 120
    anderson_memory: int = 8
    anderson_regularization: float = 1.0e-2
    anderson_line_search: RootLineSearch = None
    krylov_max_iterations: int = 120
    krylov_inner_max_iterations: int = 30

    def __post_init__(self) -> None:
        positive = (
            self.residual_rms_tolerance,
            self.residual_max_tolerance,
            self.anderson_memory,
            self.anderson_regularization,
            self.krylov_max_iterations,
            self.krylov_inner_max_iterations,
        )
        if any(float(value) <= 0.0 or not np.isfinite(float(value)) for value in positive):
            raise ValueError("stationary solver tolerances and iteration controls must be positive")
        if self.anderson_max_iterations < 0:
            raise ValueError("anderson_max_iterations must be nonnegative")
        if self.anderson_line_search not in {None, "armijo", "wolfe"}:
            raise ValueError("unsupported Anderson line search")


@dataclass(frozen=True)
class StationarySolveResult:
    vector: Array
    residual: Array
    residual_rms: float
    residual_max: float
    converged: bool
    exit_reason: str
    evaluations: int
    anderson_converged: bool
    krylov_attempted: bool


def _validated_residual(residual_builder: RealResidual, vector: Array) -> Array:
    residual = np.asarray(residual_builder(vector), dtype=np.float64)
    if residual.shape != vector.shape:
        raise ValueError(f"residual shape {residual.shape} does not match vector {vector.shape}")
    if not np.all(np.isfinite(residual)):
        raise ValueError("stationary residual must be finite")
    return residual


def solve_stationary_residual(
    residual_builder: RealResidual,
    initial_vector: Array,
    *,
    config: StationarySolveConfig = StationarySolveConfig(),
) -> StationarySolveResult:
    """Solve ``R(x)=0`` without requiring the physical energy to decrease.

    Solver return codes are never accepted on their own.  The returned root is
    promoted only if its independently recomputed RMS and max residuals pass
    the configured tolerances.
    """

    initial = np.asarray(initial_vector, dtype=np.float64)
    if initial.ndim != 1 or not np.all(np.isfinite(initial)):
        raise ValueError("initial_vector must be one-dimensional and finite")
    evaluations = 0
    best_vector = initial.copy()
    best_residual = _validated_residual(residual_builder, best_vector)
    evaluations += 1
    best_rms = float(np.sqrt(np.mean(best_residual**2)))
    best_max = float(np.max(np.abs(best_residual), initial=0.0))
    if (
        best_rms <= config.residual_rms_tolerance
        and best_max <= config.residual_max_tolerance
    ):
        return StationarySolveResult(
            vector=best_vector,
            residual=best_residual,
            residual_rms=best_rms,
            residual_max=best_max,
            converged=True,
            exit_reason="initial_residual_tolerance",
            evaluations=evaluations,
            anderson_converged=False,
            krylov_attempted=False,
        )

    def tracked(vector: Array) -> Array:
        nonlocal evaluations, best_vector, best_residual, best_rms
        candidate = np.asarray(vector, dtype=np.float64)
        residual = _validated_residual(residual_builder, candidate)
        evaluations += 1
        rms = float(np.sqrt(np.mean(residual**2)))
        if rms < best_rms:
            best_rms = rms
            best_vector = candidate.copy()
            best_residual = residual.copy()
        return residual

    anderson_converged = False
    if config.anderson_max_iterations == 0:
        candidate = initial.copy()
    else:
        anderson_converged = True
        try:
            candidate = np.asarray(
                anderson(
                    tracked,
                    initial,
                    M=int(config.anderson_memory),
                    maxiter=int(config.anderson_max_iterations),
                    f_tol=float(config.residual_max_tolerance),
                    w0=float(config.anderson_regularization),
                    line_search=config.anderson_line_search,
                ),
                dtype=np.float64,
            )
        except NoConvergence as error:
            anderson_converged = False
            candidate = np.asarray(error.args[0], dtype=np.float64)
            tracked(candidate)

    candidate_residual = tracked(candidate)
    candidate_rms = float(np.sqrt(np.mean(candidate_residual**2)))
    candidate_max = float(np.max(np.abs(candidate_residual), initial=0.0))
    candidate_passes = (
        candidate_rms <= config.residual_rms_tolerance
        and candidate_max <= config.residual_max_tolerance
    )
    krylov_attempted = not candidate_passes
    if krylov_attempted:
        start = best_vector.copy()
        try:
            candidate = np.asarray(
                newton_krylov(
                    tracked,
                    start,
                    f_tol=float(config.residual_max_tolerance),
                    maxiter=int(config.krylov_max_iterations),
                    inner_maxiter=int(config.krylov_inner_max_iterations),
                    line_search=config.anderson_line_search,
                ),
                dtype=np.float64,
            )
        except NoConvergence as error:
            candidate = np.asarray(error.args[0], dtype=np.float64)
            tracked(candidate)
        tracked(candidate)

    final_vector = best_vector.copy()
    final_residual = _validated_residual(residual_builder, final_vector)
    evaluations += 1
    residual_rms = float(np.sqrt(np.mean(final_residual**2)))
    residual_max = float(np.max(np.abs(final_residual), initial=0.0))
    converged = (
        residual_rms <= config.residual_rms_tolerance
        and residual_max <= config.residual_max_tolerance
    )
    if converged:
        exit_reason = "residual_tolerance"
    elif krylov_attempted:
        exit_reason = "krylov_exhausted"
    else:
        exit_reason = "anderson_exhausted"
    return StationarySolveResult(
        vector=final_vector,
        residual=final_residual,
        residual_rms=residual_rms,
        residual_max=residual_max,
        converged=converged,
        exit_reason=exit_reason,
        evaluations=evaluations,
        anderson_converged=anderson_converged,
        krylov_attempted=krylov_attempted,
    )


def finite_difference_jacobian_vector(
    residual_builder: RealResidual,
    vector: Array,
    direction: Array,
    *,
    relative_step: float = 1.0e-6,
) -> Array:
    """Return a centered finite-difference Jacobian-vector product."""

    point = np.asarray(vector, dtype=np.float64)
    tangent = np.asarray(direction, dtype=np.float64)
    if point.ndim != 1 or tangent.shape != point.shape:
        raise ValueError("vector and direction must be matching one-dimensional arrays")
    norm = float(np.linalg.norm(tangent))
    if norm == 0.0:
        return np.zeros_like(point)
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    scale = max(1.0, float(np.linalg.norm(point)))
    step = float(relative_step) * scale / norm
    plus = _validated_residual(residual_builder, point + step * tangent)
    minus = _validated_residual(residual_builder, point - step * tangent)
    return (plus - minus) / (2.0 * step)


__all__ = [
    "StationarySolveConfig",
    "StationarySolveResult",
    "finite_difference_jacobian_vector",
    "pack_hermitian_matrix_field",
    "solve_stationary_residual",
    "unpack_hermitian_matrix_field",
]
