"""Safeguarded Pulay DIIS for Hermitian fixed-trace fixed-point maps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from .occupations import calculate_norm_convergence

PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class FixedPointEvaluation(Generic[PayloadT]):
    output: np.ndarray
    payload: PayloadT


@dataclass(frozen=True)
class PulayDIISOptions:
    max_history: int = 8
    regularization: float = 1.0e-12
    max_regularization: float = 1.0e-4
    condition_limit: float = 1.0e14
    coefficient_l1_limit: float = 10.0
    trust_radius: float = 2.0
    restart_growth_factor: float = 10.0

    def __post_init__(self) -> None:
        if int(self.max_history) < 2:
            raise ValueError("max_history must be at least two")
        for name in (
            "regularization",
            "max_regularization",
            "condition_limit",
            "coefficient_l1_limit",
            "trust_radius",
            "restart_growth_factor",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.regularization > self.max_regularization:
            raise ValueError("regularization cannot exceed max_regularization")
        if self.coefficient_l1_limit < 1.0:
            raise ValueError("coefficient_l1_limit must be at least one")


@dataclass(frozen=True)
class PulayDIISStep:
    iteration: int
    residual_norm: float
    step_kind: str
    history_size: int
    regularization: float
    coefficient_l1: float
    trust_scale: float


@dataclass(frozen=True)
class PulayOperatorEvaluation(Generic[PayloadT]):
    hamiltonian: np.ndarray
    mapped_density: np.ndarray
    error: np.ndarray
    payload: PayloadT


@dataclass(frozen=True)
class PulayOperatorDIISStep:
    iteration: int
    residual_norm: float
    error_norm: float
    step_kind: str
    history_size: int
    regularization: float
    coefficient_l1: float


@dataclass(frozen=True)
class PulayOperatorDIISRun(Generic[PayloadT]):
    density: np.ndarray
    final_evaluation: PulayOperatorEvaluation[PayloadT]
    final_residual: np.ndarray
    final_residual_norm: float
    final_error_norm: float
    iter_residual: np.ndarray
    iter_error: np.ndarray
    converged: bool
    exit_reason: str
    restart_count: int
    steps: tuple[PulayOperatorDIISStep, ...]

    @property
    def iterations(self) -> int:
        return int(self.iter_residual.size)


@dataclass(frozen=True)
class PulayDIISRun(Generic[PayloadT]):
    density: np.ndarray
    final_evaluation: FixedPointEvaluation[PayloadT]
    final_residual: np.ndarray
    final_residual_norm: float
    iter_residual: np.ndarray
    converged: bool
    exit_reason: str
    restart_count: int
    steps: tuple[PulayDIISStep, ...]

    @property
    def iterations(self) -> int:
        return int(self.iter_residual.size)


def _validate_same_shape_finite(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    result = np.asarray(candidate, dtype=np.complex128)
    if result.shape != reference.shape:
        raise ValueError(f"{name} shape differs from the fixed-point iterate")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _pulay_coefficients(
    residuals: list[np.ndarray],
    options: PulayDIISOptions,
) -> tuple[np.ndarray, float] | None:
    count = len(residuals)
    gram = np.empty((count, count), dtype=np.float64)
    for i, left in enumerate(residuals):
        for j, right in enumerate(residuals):
            gram[i, j] = float(np.vdot(left, right).real)
    scale = max(float(np.max(np.diag(gram), initial=0.0)), 1.0e-300)
    regularization = float(options.regularization)
    while regularization <= float(options.max_regularization) * (1.0 + 1.0e-12):
        augmented = np.empty((count + 1, count + 1), dtype=np.float64)
        augmented[:count, :count] = gram / scale
        augmented[:count, :count] += regularization * np.eye(count)
        augmented[:count, count] = 1.0
        augmented[count, :count] = 1.0
        augmented[count, count] = 0.0
        rhs = np.zeros(count + 1, dtype=np.float64)
        rhs[count] = 1.0
        try:
            condition = float(np.linalg.cond(augmented))
            if not np.isfinite(condition) or condition > options.condition_limit:
                raise np.linalg.LinAlgError("ill-conditioned DIIS system")
            solution = np.linalg.solve(augmented, rhs)
        except np.linalg.LinAlgError:
            regularization *= 10.0
            continue
        coefficients = np.asarray(solution[:count], dtype=np.float64)
        if not np.all(np.isfinite(coefficients)):
            regularization *= 10.0
            continue
        coefficients[-1] += 1.0 - float(np.sum(coefficients))
        if float(np.sum(np.abs(coefficients))) > options.coefficient_l1_limit:
            regularization *= 10.0
            continue
        return coefficients, regularization
    return None


def run_pulay_diis_fixed_point(
    initial_density: np.ndarray,
    evaluate: Callable[[np.ndarray], FixedPointEvaluation[PayloadT]],
    *,
    precision: float = 1.0e-8,
    max_iter: int = 300,
    options: PulayDIISOptions | None = None,
    convergence_metric: Callable[[np.ndarray, np.ndarray], float] = calculate_norm_convergence,
    validate_output: Callable[[np.ndarray], None] | None = None,
    validate_iterate: Callable[[np.ndarray], None] | None = None,
    step_callback: Callable[[PulayDIISStep, FixedPointEvaluation[PayloadT]], None]
    | None = None,
) -> PulayDIISRun[PayloadT]:
    """Solve ``evaluate(D).output == D`` with output-based Pulay DIIS.

    Affine combinations use real coefficients summing to one. Consequently,
    Hermiticity and one global fixed-trace constraint are preserved when every
    map output satisfies those invariants. Positivity is not implied for
    intermediate DIIS iterates and must not be assumed by the interaction map.
    """

    density = np.asarray(initial_density, dtype=np.complex128).copy()
    if density.size == 0 or not np.all(np.isfinite(density)):
        raise ValueError("initial_density must be finite and nonempty")
    tolerance = float(precision)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("precision must be finite and positive")
    maximum_iterations = int(max_iter)
    if maximum_iterations <= 0:
        raise ValueError("max_iter must be positive")
    policy = PulayDIISOptions() if options is None else options
    if validate_iterate is not None:
        validate_iterate(density)

    outputs: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    iter_residual: list[float] = []
    steps: list[PulayDIISStep] = []
    restart_count = 0
    exit_reason = "max_iter"
    previous_norm: float | None = None

    for iteration in range(1, maximum_iterations + 1):
        evaluation = evaluate(density)
        output = _validate_same_shape_finite(
            evaluation.output, density, name="fixed-point output"
        )
        if validate_output is not None:
            validate_output(output)
        residual = output - density
        residual_norm = float(convergence_metric(output, density))
        if not np.isfinite(residual_norm) or residual_norm < 0.0:
            raise ValueError("convergence_metric returned an invalid residual")
        iter_residual.append(residual_norm)
        if residual_norm <= tolerance:
            step = PulayDIISStep(
                iteration=iteration,
                residual_norm=residual_norm,
                step_kind="converged",
                history_size=len(residuals),
                regularization=0.0,
                coefficient_l1=1.0,
                trust_scale=1.0,
            )
            steps.append(step)
            if step_callback is not None:
                step_callback(step, evaluation)
            exit_reason = "converged"
            break

        grew = (
            previous_norm is not None
            and residual_norm > policy.restart_growth_factor * previous_norm
        )
        if grew:
            outputs.clear()
            residuals.clear()
            restart_count += 1
        outputs.append(np.array(output, copy=True))
        residuals.append(np.array(residual, copy=True))
        if len(outputs) > policy.max_history:
            outputs.pop(0)
            residuals.pop(0)

        regularization = 0.0
        coefficient_l1 = 1.0
        trust_scale = 1.0
        if len(outputs) == 1:
            next_density = np.array(output, copy=True)
            step_kind = "restart_picard" if grew else "picard"
        else:
            solved = _pulay_coefficients(residuals, policy)
            if solved is None:
                outputs[:] = outputs[-1:]
                residuals[:] = residuals[-1:]
                restart_count += 1
                next_density = np.array(output, copy=True)
                step_kind = "restart_picard"
            else:
                coefficients, regularization = solved
                coefficient_l1 = float(np.sum(np.abs(coefficients)))
                candidate = np.zeros_like(density)
                for coefficient, mapped in zip(coefficients, outputs):
                    candidate += coefficient * mapped
                correction = candidate - output
                correction_norm = float(np.linalg.norm(correction))
                trust_bound = policy.trust_radius * max(
                    float(np.linalg.norm(residual)), 1.0e-300
                )
                if correction_norm > trust_bound:
                    trust_scale = trust_bound / correction_norm
                    candidate = output + trust_scale * correction
                    step_kind = "diis_trusted"
                else:
                    step_kind = "diis"
                next_density = candidate
        next_density = _validate_same_shape_finite(
            next_density, density, name="DIIS iterate"
        )
        if validate_iterate is not None:
            validate_iterate(next_density)
        step = PulayDIISStep(
            iteration=iteration,
            residual_norm=residual_norm,
            step_kind=step_kind,
            history_size=len(residuals),
            regularization=float(regularization),
            coefficient_l1=coefficient_l1,
            trust_scale=trust_scale,
        )
        steps.append(step)
        if step_callback is not None:
            step_callback(step, evaluation)
        density = np.array(next_density, copy=True)
        previous_norm = residual_norm

    final_evaluation = evaluate(density)
    final_output = _validate_same_shape_finite(
        final_evaluation.output, density, name="final fixed-point output"
    )
    if validate_output is not None:
        validate_output(final_output)
    final_residual = final_output - density
    final_norm = float(convergence_metric(final_output, density))
    if not np.isfinite(final_norm) or final_norm < 0.0:
        raise ValueError("final replay residual is invalid")
    converged = final_norm <= tolerance
    if exit_reason == "converged" and not converged:
        exit_reason = "final_replay_failed"
    elif exit_reason == "max_iter" and converged:
        exit_reason = "converged_final_replay"
    return PulayDIISRun(
        density=np.array(density, copy=True),
        final_evaluation=final_evaluation,
        final_residual=np.asarray(final_residual, dtype=np.complex128),
        final_residual_norm=final_norm,
        iter_residual=np.asarray(iter_residual, dtype=np.float64),
        converged=converged,
        exit_reason=exit_reason,
        restart_count=restart_count,
        steps=tuple(steps),
    )


def run_pulay_operator_diis(
    initial_density: np.ndarray,
    evaluate_density: Callable[[np.ndarray], PulayOperatorEvaluation[PayloadT]],
    map_hamiltonian: Callable[[np.ndarray], np.ndarray],
    *,
    precision: float = 1.0e-8,
    max_iter: int = 300,
    options: PulayDIISOptions | None = None,
    convergence_metric: Callable[[np.ndarray, np.ndarray], float] = calculate_norm_convergence,
    validate_density: Callable[[np.ndarray], None] | None = None,
    validate_hamiltonian: Callable[[np.ndarray], None] | None = None,
    step_callback: Callable[
        [PulayOperatorDIISStep, PulayOperatorEvaluation[PayloadT]], None
    ]
    | None = None,
) -> PulayOperatorDIISRun[PayloadT]:
    """Run operator Pulay DIIS while keeping every density a map output.

    The caller returns the physical Hamiltonian and a Pulay error, normally
    ``[H, P]``, at each density. Pulay coefficients extrapolate Hamiltonians,
    and ``map_hamiltonian`` performs the ordinary fixed-filling Aufbau map.
    Convergence remains the original raw density residual.
    """

    density = np.asarray(initial_density, dtype=np.complex128).copy()
    if density.size == 0 or not np.all(np.isfinite(density)):
        raise ValueError("initial_density must be finite and nonempty")
    tolerance = float(precision)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("precision must be finite and positive")
    maximum_iterations = int(max_iter)
    if maximum_iterations <= 0:
        raise ValueError("max_iter must be positive")
    policy = PulayDIISOptions() if options is None else options
    if validate_density is not None:
        validate_density(density)

    hamiltonians: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    iter_residual: list[float] = []
    iter_error: list[float] = []
    steps: list[PulayOperatorDIISStep] = []
    restart_count = 0
    exit_reason = "max_iter"
    previous_error_norm: float | None = None

    for iteration in range(1, maximum_iterations + 1):
        evaluation = evaluate_density(density)
        hamiltonian = _validate_same_shape_finite(
            evaluation.hamiltonian, density, name="physical Hamiltonian"
        )
        mapped = _validate_same_shape_finite(
            evaluation.mapped_density, density, name="fixed-point output"
        )
        error = np.asarray(evaluation.error, dtype=np.complex128)
        if error.shape != density.shape:
            raise ValueError("Pulay operator error must match the density shape")
        if error.size == 0 or not np.all(np.isfinite(error)):
            raise ValueError("Pulay operator error must be finite and nonempty")
        if validate_hamiltonian is not None:
            validate_hamiltonian(hamiltonian)
        if validate_density is not None:
            validate_density(mapped)
        residual_norm = float(convergence_metric(mapped, density))
        error_norm = float(np.linalg.norm(error) / np.sqrt(error.size))
        if not np.isfinite(residual_norm) or residual_norm < 0.0:
            raise ValueError("convergence_metric returned an invalid residual")
        if not np.isfinite(error_norm) or error_norm < 0.0:
            raise ValueError("Pulay operator error norm is invalid")
        iter_residual.append(residual_norm)
        iter_error.append(error_norm)
        if residual_norm <= tolerance:
            step = PulayOperatorDIISStep(
                iteration=iteration,
                residual_norm=residual_norm,
                error_norm=error_norm,
                step_kind="converged",
                history_size=len(errors),
                regularization=0.0,
                coefficient_l1=1.0,
            )
            steps.append(step)
            if step_callback is not None:
                step_callback(step, evaluation)
            exit_reason = "converged"
            break

        grew = (
            previous_error_norm is not None
            and error_norm > policy.restart_growth_factor * previous_error_norm
        )
        if grew:
            hamiltonians.clear()
            errors.clear()
            restart_count += 1
        hamiltonians.append(np.array(hamiltonian, copy=True))
        errors.append(np.array(error, copy=True))
        if len(hamiltonians) > policy.max_history:
            hamiltonians.pop(0)
            errors.pop(0)

        regularization = 0.0
        coefficient_l1 = 1.0
        if len(hamiltonians) == 1:
            next_density = np.array(mapped, copy=True)
            step_kind = "restart_picard" if grew else "picard"
        else:
            solved = _pulay_coefficients(errors, policy)
            if solved is None:
                hamiltonians[:] = hamiltonians[-1:]
                errors[:] = errors[-1:]
                restart_count += 1
                next_density = np.array(mapped, copy=True)
                step_kind = "restart_picard"
            else:
                coefficients, regularization = solved
                coefficient_l1 = float(np.sum(np.abs(coefficients)))
                extrapolated = np.zeros_like(hamiltonian)
                for coefficient, stored_hamiltonian in zip(
                    coefficients, hamiltonians
                ):
                    extrapolated += coefficient * stored_hamiltonian
                extrapolated = _validate_same_shape_finite(
                    extrapolated, density, name="DIIS Hamiltonian"
                )
                if validate_hamiltonian is not None:
                    validate_hamiltonian(extrapolated)
                next_density = _validate_same_shape_finite(
                    map_hamiltonian(extrapolated),
                    density,
                    name="DIIS mapped density",
                )
                step_kind = "operator_diis"
        if validate_density is not None:
            validate_density(next_density)
        step = PulayOperatorDIISStep(
            iteration=iteration,
            residual_norm=residual_norm,
            error_norm=error_norm,
            step_kind=step_kind,
            history_size=len(errors),
            regularization=float(regularization),
            coefficient_l1=coefficient_l1,
        )
        steps.append(step)
        if step_callback is not None:
            step_callback(step, evaluation)
        density = np.asarray(next_density, dtype=np.complex128).copy()
        previous_error_norm = error_norm

    final_evaluation = evaluate_density(density)
    final_hamiltonian = _validate_same_shape_finite(
        final_evaluation.hamiltonian, density, name="final physical Hamiltonian"
    )
    final_mapped = _validate_same_shape_finite(
        final_evaluation.mapped_density, density, name="final fixed-point output"
    )
    final_error = np.asarray(final_evaluation.error, dtype=np.complex128)
    if final_error.shape != density.shape:
        raise ValueError("final Pulay operator error must match the density shape")
    if final_error.size == 0 or not np.all(np.isfinite(final_error)):
        raise ValueError("final Pulay operator error is invalid")
    if validate_hamiltonian is not None:
        validate_hamiltonian(final_hamiltonian)
    if validate_density is not None:
        validate_density(final_mapped)
    final_residual = final_mapped - density
    final_norm = float(convergence_metric(final_mapped, density))
    final_error_norm = float(np.linalg.norm(final_error) / np.sqrt(final_error.size))
    if not np.isfinite(final_norm) or final_norm < 0.0:
        raise ValueError("final replay residual is invalid")
    if not np.isfinite(final_error_norm) or final_error_norm < 0.0:
        raise ValueError("final replay operator error is invalid")
    converged = final_norm <= tolerance
    if exit_reason == "converged" and not converged:
        exit_reason = "final_replay_failed"
    elif exit_reason == "max_iter" and converged:
        exit_reason = "converged_final_replay"
    return PulayOperatorDIISRun(
        density=np.asarray(density, dtype=np.complex128).copy(),
        final_evaluation=final_evaluation,
        final_residual=np.asarray(final_residual, dtype=np.complex128),
        final_residual_norm=final_norm,
        final_error_norm=final_error_norm,
        iter_residual=np.asarray(iter_residual, dtype=np.float64),
        iter_error=np.asarray(iter_error, dtype=np.float64),
        converged=converged,
        exit_reason=exit_reason,
        restart_count=restart_count,
        steps=tuple(steps),
    )


__all__ = [
    "FixedPointEvaluation",
    "PulayDIISOptions",
    "PulayDIISRun",
    "PulayDIISStep",
    "PulayOperatorDIISRun",
    "PulayOperatorDIISStep",
    "PulayOperatorEvaluation",
    "run_pulay_diis_fixed_point",
    "run_pulay_operator_diis",
]
