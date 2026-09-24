"""Exact-unitary fixed-rank orbital trust-region minimization.

This module is system-agnostic. A physical-system adapter must supply one
scalar-functional callback whose energy, Hamiltonian, and linear Hamiltonian
response use identical conventions. Algorithmic stationarity returned here
is never a substitute for a system-specific fixed-filling replay.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

from .zero_temperature_stability import build_zero_temperature_orbital_hessian


@dataclass(frozen=True)
class FixedRankHFEvaluation:
    energy: float
    hamiltonian: np.ndarray
    hamiltonian_response: Callable[[np.ndarray], np.ndarray]
    payload: object | None = None


@dataclass(frozen=True)
class OrbitalTrustRegionOptions:
    max_iterations: int = 100
    gradient_tolerance_ev: float = 1.0e-9
    stationarity_max_tolerance_ev: float = 1.0e-9
    initial_radius: float = 0.2
    maximum_radius: float = 1.0
    minimum_radius: float = 1.0e-8
    acceptance_ratio: float = 0.1
    shrink_ratio: float = 0.25
    expansion_ratio: float = 0.75
    shrink_factor: float = 0.25
    expansion_factor: float = 2.0
    cg_max_iterations: int = 100
    cg_relative_tolerance_maximum: float = 0.5
    curvature_tolerance_ev: float = 1.0e-14
    model_reduction_tolerance_ev: float = 1.0e-16
    self_adjoint_probe_count: int = 4
    self_adjoint_seed: int = 271828
    self_adjoint_absolute_tolerance_ev: float = 1.0e-9
    self_adjoint_relative_tolerance: float = 1.0e-8
    check_stationary_curvature: bool = True
    stationary_curvature_tolerance_ev: float = 1.0e-9
    stationary_curvature_residual_tolerance_ev: float = 1.0e-9
    stationary_curvature_max_iterations: int = 300


@dataclass(frozen=True)
class SteihaugStep:
    step: np.ndarray
    hessian_step: np.ndarray
    predicted_reduction: float
    iterations: int
    exit_reason: str
    reached_boundary: bool
    negative_curvature: bool


@dataclass(frozen=True)
class OrbitalTrustRegionIteration:
    iteration: int
    energy_before: float
    energy_trial: float
    energy_after: float
    gradient_norm_ev: float
    gradient_max_abs_ev: float
    trust_radius_before: float
    trust_radius_after: float
    step_norm: float
    predicted_reduction_ev: float
    actual_reduction_ev: float
    acceptance_ratio: float
    accepted: bool
    trial_valid: bool
    cg_iterations: int
    cg_exit_reason: str
    self_adjoint_absolute_defect_ev: float
    self_adjoint_relative_defect: float


@dataclass(frozen=True)
class OrbitalTrustRegionTermination:
    iteration: int
    reason: str
    energy: float
    gradient_norm_ev: float
    gradient_max_abs_ev: float
    trust_radius: float
    self_adjoint_absolute_defect_ev: float | None
    self_adjoint_relative_defect: float | None
    lowest_curvature_ev: float | None
    lowest_curvature_residual_ev: float | None


@dataclass(frozen=True)
class OrbitalTrustRegionRun:
    physical_projector: np.ndarray
    tangent_basis: np.ndarray
    evaluation: FixedRankHFEvaluation
    history: tuple[OrbitalTrustRegionIteration, ...]
    termination: OrbitalTrustRegionTermination
    exit_reason: str
    stationary_candidate: bool
    physical_convergence_established: bool = False
    system_replay_status: str = "not_run"


@dataclass(frozen=True)
class _LowestCurvature:
    eigenvalue_ev: float
    eigenvector: np.ndarray
    residual_ev: float
    matvec_count: int


class _NoPositiveModelReduction(ValueError):
    """The finite-precision quadratic model does not predict descent."""


def _validate_options(options: OrbitalTrustRegionOptions) -> None:
    integer_values = (
        options.max_iterations,
        options.cg_max_iterations,
        options.self_adjoint_probe_count,
        options.self_adjoint_seed,
        options.stationary_curvature_max_iterations,
    )
    finite_nonnegative = (
        options.gradient_tolerance_ev,
        options.stationarity_max_tolerance_ev,
        options.curvature_tolerance_ev,
        options.model_reduction_tolerance_ev,
        options.self_adjoint_absolute_tolerance_ev,
        options.self_adjoint_relative_tolerance,
        options.stationary_curvature_tolerance_ev,
    )
    finite_positive = (
        options.initial_radius,
        options.maximum_radius,
        options.minimum_radius,
        options.cg_relative_tolerance_maximum,
        options.stationary_curvature_residual_tolerance_ev,
    )
    if (
        any(isinstance(value, bool) or not isinstance(value, Integral) for value in integer_values)
        or options.max_iterations < 0
        or options.cg_max_iterations <= 0
        or options.self_adjoint_probe_count < 2
        or options.stationary_curvature_max_iterations <= 0
        or any(not np.isfinite(value) or value < 0.0 for value in finite_nonnegative)
        or any(not np.isfinite(value) or value <= 0.0 for value in finite_positive)
        or not options.minimum_radius <= options.initial_radius <= options.maximum_radius
        or not 0.0 < float(options.acceptance_ratio) < 1.0
        or not 0.0 < float(options.shrink_ratio) < float(options.expansion_ratio) < 1.0
        or not 0.0 < float(options.shrink_factor) < 1.0
        or not np.isfinite(options.expansion_factor)
        or float(options.expansion_factor) <= 1.0
        or float(options.cg_relative_tolerance_maximum) > 1.0
    ):
        raise ValueError("orbital trust-region options are inconsistent")


def _boundary_distance(step: np.ndarray, direction: np.ndarray, radius: float) -> float:
    dd = float(np.dot(direction, direction))
    if dd <= 0.0:
        raise ValueError("Steihaug boundary direction has zero norm")
    sd = float(np.dot(step, direction))
    remaining = float(radius * radius - np.dot(step, step))
    radicand = sd * sd + dd * remaining
    scale = max(sd * sd, abs(dd * remaining), radius * radius * dd, 1.0)
    if radicand < -64.0 * np.finfo(np.float64).eps * scale:
        raise ValueError("Steihaug boundary radicand is negative")
    return float((-sd + np.sqrt(max(radicand, 0.0))) / dd)


def solve_steihaug_trust_region(
    gradient: np.ndarray,
    hessian_action: Callable[[np.ndarray], np.ndarray],
    *,
    radius: float,
    maximum_iterations: int,
    residual_tolerance: float,
    curvature_tolerance_ev: float,
    model_reduction_tolerance_ev: float,
) -> SteihaugStep:
    """Approximately minimize a quadratic model inside a Euclidean ball."""

    g = np.asarray(gradient, dtype=np.float64)
    if g.ndim != 1 or not np.all(np.isfinite(g)):
        raise ValueError("Steihaug gradient must be a finite vector")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Steihaug radius must be finite and positive")
    step = np.zeros_like(g)
    residual = g.copy()
    direction = -residual
    residual_squared = float(np.dot(residual, residual))
    if residual_squared == 0.0:
        raise ValueError(
            "zero-gradient Steihaug solve requires an external curvature direction"
        )
    exit_reason = "maximum_iterations"
    reached_boundary = False
    negative_curvature = False
    iteration_count = 0
    for iteration in range(1, int(maximum_iterations) + 1):
        iteration_count = iteration
        h_direction = np.asarray(hessian_action(direction), dtype=np.float64)
        if h_direction.shape != g.shape or not np.all(np.isfinite(h_direction)):
            raise ValueError("Steihaug Hessian action returned invalid values")
        curvature = float(np.dot(direction, h_direction))
        curvature_floor = float(curvature_tolerance_ev) * float(
            np.dot(direction, direction)
        )
        if curvature <= curvature_floor:
            tau = _boundary_distance(step, direction, radius)
            step = step + tau * direction
            negative_curvature = curvature < -abs(float(curvature_tolerance_ev)) * float(
                np.dot(direction, direction)
            )
            exit_reason = (
                "negative_curvature_boundary"
                if negative_curvature
                else "unresolved_curvature_boundary"
            )
            reached_boundary = True
            break
        alpha = residual_squared / curvature
        candidate = step + alpha * direction
        if np.linalg.norm(candidate) >= radius:
            tau = _boundary_distance(step, direction, radius)
            step = step + tau * direction
            exit_reason = "trust_region_boundary"
            reached_boundary = True
            break
        step = candidate
        new_residual = residual + alpha * h_direction
        new_residual_squared = float(np.dot(new_residual, new_residual))
        if np.sqrt(new_residual_squared) <= float(residual_tolerance):
            residual = new_residual
            exit_reason = "interior_residual"
            break
        beta = new_residual_squared / residual_squared
        direction = -new_residual + beta * direction
        residual = new_residual
        residual_squared = new_residual_squared

    h_step = np.asarray(hessian_action(step), dtype=np.float64)
    predicted = float(-np.dot(g, step) - 0.5 * np.dot(step, h_step))
    if not np.isfinite(predicted) or predicted <= float(model_reduction_tolerance_ev):
        raise _NoPositiveModelReduction(
            "Steihaug quadratic model has no positive reduction"
        )
    return SteihaugStep(
        step=step,
        hessian_step=h_step,
        predicted_reduction=predicted,
        iterations=iteration_count,
        exit_reason=exit_reason,
        reached_boundary=reached_boundary,
        negative_curvature=negative_curvature,
    )


def _lowest_curvature(
    hessian_action: Callable[[np.ndarray], np.ndarray],
    *,
    size: int,
    seed: int,
    residual_tolerance_ev: float,
    maximum_iterations: int,
) -> _LowestCurvature:
    count = 0

    def apply(vector: np.ndarray) -> np.ndarray:
        nonlocal count
        result = np.asarray(hessian_action(vector), dtype=np.float64)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError("stationary Hessian action returned invalid values")
        count += 1
        return result

    if size <= 48:
        matrix = np.column_stack(
            [apply(np.eye(size, dtype=np.float64)[:, index]) for index in range(size)]
        )
        asymmetry = float(np.max(np.abs(matrix - matrix.T), initial=0.0))
        if asymmetry > float(residual_tolerance_ev):
            raise ValueError("dense stationary Hessian is not self-adjoint")
        values, vectors = np.linalg.eigh(matrix)
        eigenvalue = float(values[0])
        eigenvector = np.asarray(vectors[:, 0], dtype=np.float64)
    else:
        rng = np.random.default_rng(int(seed))
        initial = rng.normal(size=size)
        initial /= np.linalg.norm(initial)
        operator = LinearOperator((size, size), matvec=apply, dtype=np.float64)
        try:
            values, vectors = eigsh(
                operator,
                k=1,
                which="SA",
                v0=initial,
                tol=max(float(residual_tolerance_ev), np.finfo(np.float64).eps),
                maxiter=int(maximum_iterations),
            )
        except ArpackNoConvergence as error:
            raise ValueError("stationary curvature eigensolver did not converge") from error
        eigenvalue = float(values[0])
        eigenvector = np.asarray(vectors[:, 0], dtype=np.float64)
    eigenvector /= np.linalg.norm(eigenvector)
    residual = float(np.linalg.norm(apply(eigenvector) - eigenvalue * eigenvector))
    if residual > float(residual_tolerance_ev):
        raise ValueError("stationary curvature Ritz residual exceeds tolerance")
    return _LowestCurvature(
        eigenvalue_ev=eigenvalue,
        eigenvector=eigenvector,
        residual_ev=residual,
        matvec_count=count,
    )


def _negative_curvature_boundary_step(
    gradient: np.ndarray,
    hessian_action: Callable[[np.ndarray], np.ndarray],
    curvature: _LowestCurvature,
    *,
    radius: float,
    model_reduction_tolerance_ev: float,
) -> SteihaugStep:
    candidates = (radius * curvature.eigenvector, -radius * curvature.eigenvector)
    selected_step = None
    selected_h_step = None
    selected_reduction = -np.inf
    for candidate in candidates:
        h_candidate = np.asarray(hessian_action(candidate), dtype=np.float64)
        reduction = float(
            -np.dot(gradient, candidate) - 0.5 * np.dot(candidate, h_candidate)
        )
        if reduction > selected_reduction:
            selected_step = candidate
            selected_h_step = h_candidate
            selected_reduction = reduction
    if (
        selected_step is None
        or selected_h_step is None
        or not np.isfinite(selected_reduction)
        or selected_reduction <= float(model_reduction_tolerance_ev)
    ):
        raise _NoPositiveModelReduction(
            "negative-curvature model has no positive reduction"
        )
    return SteihaugStep(
        step=selected_step,
        hessian_step=selected_h_step,
        predicted_reduction=selected_reduction,
        iterations=curvature.matvec_count,
        exit_reason="stationary_negative_curvature_boundary",
        reached_boundary=True,
        negative_curvature=True,
    )


def run_orbital_trust_region(
    initial_projector: np.ndarray,
    evaluate_projector: Callable[[np.ndarray], FixedRankHFEvaluation],
    *,
    occupied_per_k: int,
    k_weights: np.ndarray | None = None,
    initial_basis: np.ndarray | None = None,
    options: OrbitalTrustRegionOptions = OrbitalTrustRegionOptions(),
    iteration_callback: Callable[[OrbitalTrustRegionIteration], None] | None = None,
    trial_validator: Callable[[np.ndarray, FixedRankHFEvaluation], bool] | None = None,
) -> OrbitalTrustRegionRun:
    """Minimize an exact scalar functional on a product Grassmann manifold."""

    _validate_options(options)
    projector = np.asarray(initial_projector, dtype=np.complex128).copy()
    evaluation = evaluate_projector(projector)
    if not np.isfinite(evaluation.energy):
        raise ValueError("initial scalar energy is not finite")
    radius = float(options.initial_radius)
    basis = None if initial_basis is None else np.asarray(initial_basis).copy()
    history: list[OrbitalTrustRegionIteration] = []
    initial_gradient_norm: float | None = None
    stationary_candidate = False
    rng = np.random.default_rng(int(options.self_adjoint_seed))
    termination: OrbitalTrustRegionTermination | None = None

    for outer_iteration in range(int(options.max_iterations) + 1):
        hessian = build_zero_temperature_orbital_hessian(
            projector,
            np.asarray(evaluation.hamiltonian, dtype=np.complex128),
            evaluation.hamiltonian_response,
            occupied_per_k=int(occupied_per_k),
            k_weights=k_weights,
            tangent_basis=basis,
        )
        basis = np.asarray(hessian.frame.basis, dtype=np.complex128)
        gradient = hessian.frame.pack_weighted_complex(
            hessian.stationarity_residual, factor=2.0
        )
        gradient_norm = float(np.linalg.norm(gradient))
        gradient_max = float(np.max(np.abs(hessian.stationarity_residual), initial=0.0))
        if initial_gradient_norm is None:
            initial_gradient_norm = max(gradient_norm, np.finfo(np.float64).eps)

        probes = rng.normal(size=(int(options.self_adjoint_probe_count), hessian.shape[0]))
        probes /= np.linalg.norm(probes, axis=1)[:, None]
        symmetry = hessian.verify_self_adjointness(probes)
        if (
            symmetry.maximum_absolute_defect
            > float(options.self_adjoint_absolute_tolerance_ev)
            and symmetry.maximum_relative_defect
            > float(options.self_adjoint_relative_tolerance)
        ):
            termination = OrbitalTrustRegionTermination(
                iteration=outer_iteration,
                reason="invalid_hessian",
                energy=float(evaluation.energy),
                gradient_norm_ev=gradient_norm,
                gradient_max_abs_ev=gradient_max,
                trust_radius=radius,
                self_adjoint_absolute_defect_ev=symmetry.maximum_absolute_defect,
                self_adjoint_relative_defect=symmetry.maximum_relative_defect,
                lowest_curvature_ev=None,
                lowest_curvature_residual_ev=None,
            )
            break
        if radius < float(options.minimum_radius):
            termination = OrbitalTrustRegionTermination(
                iteration=outer_iteration,
                reason="radius_exhausted",
                energy=float(evaluation.energy),
                gradient_norm_ev=gradient_norm,
                gradient_max_abs_ev=gradient_max,
                trust_radius=radius,
                self_adjoint_absolute_defect_ev=symmetry.maximum_absolute_defect,
                self_adjoint_relative_defect=symmetry.maximum_relative_defect,
                lowest_curvature_ev=None,
                lowest_curvature_residual_ev=None,
            )
            break

        trust_step: SteihaugStep | None = None
        lowest: _LowestCurvature | None = None
        first_order_small = (
            gradient_norm <= float(options.gradient_tolerance_ev)
            and gradient_max <= float(options.stationarity_max_tolerance_ev)
        )
        if first_order_small:
            if not bool(options.check_stationary_curvature):
                stationary_candidate = True
                termination = OrbitalTrustRegionTermination(
                    iteration=outer_iteration,
                    reason="first_order_stationary_candidate_curvature_unchecked",
                    energy=float(evaluation.energy),
                    gradient_norm_ev=gradient_norm,
                    gradient_max_abs_ev=gradient_max,
                    trust_radius=radius,
                    self_adjoint_absolute_defect_ev=symmetry.maximum_absolute_defect,
                    self_adjoint_relative_defect=symmetry.maximum_relative_defect,
                    lowest_curvature_ev=None,
                    lowest_curvature_residual_ev=None,
                )
                break
            try:
                lowest = _lowest_curvature(
                    hessian.energy_hessian_action,
                    size=hessian.shape[0],
                    seed=int(options.self_adjoint_seed) + outer_iteration,
                    residual_tolerance_ev=float(
                        options.stationary_curvature_residual_tolerance_ev
                    ),
                    maximum_iterations=int(options.stationary_curvature_max_iterations),
                )
            except ValueError:
                termination = OrbitalTrustRegionTermination(
                    iteration=outer_iteration,
                    reason="stationary_curvature_solver_failed",
                    energy=float(evaluation.energy),
                    gradient_norm_ev=gradient_norm,
                    gradient_max_abs_ev=gradient_max,
                    trust_radius=radius,
                    self_adjoint_absolute_defect_ev=symmetry.maximum_absolute_defect,
                    self_adjoint_relative_defect=symmetry.maximum_relative_defect,
                    lowest_curvature_ev=None,
                    lowest_curvature_residual_ev=None,
                )
                break
            curvature_threshold = -float(options.stationary_curvature_tolerance_ev)
            curvature_lower_edge = lowest.eigenvalue_ev - lowest.residual_ev
            if curvature_lower_edge >= curvature_threshold:
                stationary_candidate = True
                termination = OrbitalTrustRegionTermination(
                    iteration=outer_iteration,
                    reason="second_order_stationary_candidate",
                    energy=float(evaluation.energy),
                    gradient_norm_ev=gradient_norm,
                    gradient_max_abs_ev=gradient_max,
                    trust_radius=radius,
                    self_adjoint_absolute_defect_ev=symmetry.maximum_absolute_defect,
                    self_adjoint_relative_defect=symmetry.maximum_relative_defect,
                    lowest_curvature_ev=lowest.eigenvalue_ev,
                    lowest_curvature_residual_ev=lowest.residual_ev,
                )
                break
            if lowest.eigenvalue_ev >= curvature_threshold:
                termination = OrbitalTrustRegionTermination(
                    iteration=outer_iteration,
                    reason="stationary_curvature_inconclusive",
                    energy=float(evaluation.energy),
                    gradient_norm_ev=gradient_norm,
                    gradient_max_abs_ev=gradient_max,
                    trust_radius=radius,
                    self_adjoint_absolute_defect_ev=symmetry.maximum_absolute_defect,
                    self_adjoint_relative_defect=symmetry.maximum_relative_defect,
                    lowest_curvature_ev=lowest.eigenvalue_ev,
                    lowest_curvature_residual_ev=lowest.residual_ev,
                )
                break
            if outer_iteration >= int(options.max_iterations):
                termination = OrbitalTrustRegionTermination(
                    iteration=outer_iteration,
                    reason="maximum_iterations",
                    energy=float(evaluation.energy),
                    gradient_norm_ev=gradient_norm,
                    gradient_max_abs_ev=gradient_max,
                    trust_radius=radius,
                    self_adjoint_absolute_defect_ev=symmetry.maximum_absolute_defect,
                    self_adjoint_relative_defect=symmetry.maximum_relative_defect,
                    lowest_curvature_ev=lowest.eigenvalue_ev,
                    lowest_curvature_residual_ev=lowest.residual_ev,
                )
                break
            try:
                trust_step = _negative_curvature_boundary_step(
                    gradient,
                    hessian.energy_hessian_action,
                    lowest,
                    radius=radius,
                    model_reduction_tolerance_ev=float(
                        options.model_reduction_tolerance_ev
                    ),
                )
            except _NoPositiveModelReduction:
                termination = OrbitalTrustRegionTermination(
                    iteration=outer_iteration,
                    reason="no_positive_model_reduction",
                    energy=float(evaluation.energy),
                    gradient_norm_ev=gradient_norm,
                    gradient_max_abs_ev=gradient_max,
                    trust_radius=radius,
                    self_adjoint_absolute_defect_ev=symmetry.maximum_absolute_defect,
                    self_adjoint_relative_defect=symmetry.maximum_relative_defect,
                    lowest_curvature_ev=lowest.eigenvalue_ev,
                    lowest_curvature_residual_ev=lowest.residual_ev,
                )
                break
        elif outer_iteration >= int(options.max_iterations):
            termination = OrbitalTrustRegionTermination(
                iteration=outer_iteration,
                reason="maximum_iterations",
                energy=float(evaluation.energy),
                gradient_norm_ev=gradient_norm,
                gradient_max_abs_ev=gradient_max,
                trust_radius=radius,
                self_adjoint_absolute_defect_ev=symmetry.maximum_absolute_defect,
                self_adjoint_relative_defect=symmetry.maximum_relative_defect,
                lowest_curvature_ev=None,
                lowest_curvature_residual_ev=None,
            )
            break
        else:
            forcing = min(
                float(options.cg_relative_tolerance_maximum),
                np.sqrt(gradient_norm / initial_gradient_norm),
            )
            try:
                trust_step = solve_steihaug_trust_region(
                    gradient,
                    hessian.energy_hessian_action,
                    radius=radius,
                    maximum_iterations=int(options.cg_max_iterations),
                    residual_tolerance=forcing * gradient_norm,
                    curvature_tolerance_ev=float(options.curvature_tolerance_ev),
                    model_reduction_tolerance_ev=float(
                        options.model_reduction_tolerance_ev
                    ),
                )
            except _NoPositiveModelReduction:
                termination = OrbitalTrustRegionTermination(
                    iteration=outer_iteration,
                    reason="no_positive_model_reduction",
                    energy=float(evaluation.energy),
                    gradient_norm_ev=gradient_norm,
                    gradient_max_abs_ev=gradient_max,
                    trust_radius=radius,
                    self_adjoint_absolute_defect_ev=symmetry.maximum_absolute_defect,
                    self_adjoint_relative_defect=symmetry.maximum_relative_defect,
                    lowest_curvature_ev=None,
                    lowest_curvature_residual_ev=None,
                )
                break

        if trust_step is None:
            raise RuntimeError("trust-region step construction fell through")
        energy_before = float(evaluation.energy)
        trial_basis = hessian.frame.unitary_basis(trust_step.step)
        trial_projector = np.empty_like(projector)
        for k_index in range(hessian.frame.nk):
            occupied = trial_basis[:, : hessian.frame.nocc, k_index]
            trial_projector[:, :, k_index] = occupied @ occupied.conj().T
        trial_evaluation = evaluate_projector(trial_projector)
        trial_energy = float(trial_evaluation.energy)
        actual_reduction = float(energy_before - trial_energy)
        ratio = actual_reduction / trust_step.predicted_reduction
        trial_valid = bool(
            trial_validator is None or trial_validator(trial_projector, trial_evaluation)
        )
        accepted = bool(
            trial_valid
            and np.isfinite(trial_energy)
            and np.isfinite(ratio)
            and ratio >= float(options.acceptance_ratio)
            and actual_reduction > 0.0
        )
        radius_before = radius
        if not trial_valid:
            radius = float(options.shrink_factor) * radius
        elif not np.isfinite(ratio) or ratio < float(options.shrink_ratio):
            radius = float(options.shrink_factor) * radius
        elif ratio > float(options.expansion_ratio) and trust_step.reached_boundary:
            radius = min(float(options.expansion_factor) * radius, float(options.maximum_radius))
        if accepted:
            projector = trial_projector
            basis = trial_basis
            evaluation = trial_evaluation
        receipt = OrbitalTrustRegionIteration(
            iteration=outer_iteration + 1,
            energy_before=energy_before,
            energy_trial=trial_energy,
            energy_after=float(evaluation.energy),
            gradient_norm_ev=gradient_norm,
            gradient_max_abs_ev=gradient_max,
            trust_radius_before=radius_before,
            trust_radius_after=radius,
            step_norm=float(np.linalg.norm(trust_step.step)),
            predicted_reduction_ev=trust_step.predicted_reduction,
            actual_reduction_ev=actual_reduction,
            acceptance_ratio=float(ratio),
            accepted=accepted,
            trial_valid=trial_valid,
            cg_iterations=trust_step.iterations,
            cg_exit_reason=trust_step.exit_reason,
            self_adjoint_absolute_defect_ev=symmetry.maximum_absolute_defect,
            self_adjoint_relative_defect=symmetry.maximum_relative_defect,
        )
        history.append(receipt)
        if iteration_callback is not None:
            iteration_callback(receipt)

    if termination is None:
        raise RuntimeError("orbital trust-region run ended without a terminal receipt")
    return OrbitalTrustRegionRun(
        physical_projector=projector,
        tangent_basis=np.asarray(basis, dtype=np.complex128),
        evaluation=evaluation,
        history=tuple(history),
        termination=termination,
        exit_reason=termination.reason,
        stationary_candidate=stationary_candidate,
    )


__all__ = [
    "FixedRankHFEvaluation",
    "OrbitalTrustRegionIteration",
    "OrbitalTrustRegionOptions",
    "OrbitalTrustRegionRun",
    "OrbitalTrustRegionTermination",
    "SteihaugStep",
    "run_orbital_trust_region",
    "solve_steihaug_trust_region",
]
