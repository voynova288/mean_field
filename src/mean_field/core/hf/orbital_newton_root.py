"""Exact-Jacobian Newton--MINRES solver for fixed-rank HF stationarity.

The solver is deliberately a root solver, not an energy minimizer.  At every
accepted outer step it recenters an exact-unitary occupied--virtual chart,
solves the signed equation ``H s = -g`` with :func:`scipy.sparse.linalg.minres`,
and globalizes only with the occupied--virtual residual norm.  Energy is
recorded for diagnostics but never participates in trial acceptance.

A physical-system adapter supplies :class:`FixedRankHFEvaluation`; its
``hamiltonian_response`` must be the exact linear response of the same HF
Hamiltonian represented by ``hamiltonian``.  The default self-adjointness
probe tolerances are runtime sentinels only: every production adapter must
calibrate the absolute tolerance, relative tolerance, and nonzero scale floor
for its own operator and numerical scale.  In particular, synthetic PtSe2
tests exercise generic algebra but do not authorize a PtSe2 provider for
production.  This module has no system imports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.sparse.linalg import LinearOperator, minres

from .orbital_trust_region import FixedRankHFEvaluation
from .zero_temperature_stability import (
    OrbitalTangentFrame,
    ZeroTemperatureOrbitalHessian,
    build_zero_temperature_orbital_hessian,
)


@dataclass(frozen=True)
class OrbitalNewtonRootOptions:
    """Hard limits and runtime-sentinel tolerances for the root solver.

    Production adapters must calibrate the self-adjoint absolute/relative
    tolerances and nonzero scale floor instead of treating these defaults as
    provider qualification.
    """

    max_outer_iterations: int = 40
    max_hessian_actions: int = 2_000
    max_projector_evaluations: int = 200
    residual_max_tolerance_ev: float = 1.0e-9
    residual_rms_tolerance_ev: float = 1.0e-9
    step_radius: float = 0.5
    maximum_cumulative_distance: float = 5.0
    minres_relative_tolerance: float = 1.0e-10
    minres_max_iterations: int = 300
    self_adjoint_probe_count: int = 4
    self_adjoint_seed: int = 271828
    self_adjoint_absolute_tolerance_ev: float = 1.0e-9
    self_adjoint_relative_tolerance: float = 1.0e-8
    self_adjoint_scale_floor_ev: float = 1.0e-12
    sufficient_decrease: float = 1.0e-4
    backtrack_factor: float = 0.5
    maximum_backtracks: int = 20
    minimum_step_scale: float = 1.0e-10
    projector_tolerance: float = 1.0e-8


@dataclass(frozen=True)
class OrbitalNewtonRootTrial:
    """One deterministic residual-merit line-search trial."""

    iteration: int
    trial: int
    step_scale: float
    step_norm: float
    cumulative_distance_if_accepted: float
    energy_before: float
    energy_trial: float | None
    validator_accepted: bool | None
    residual_merit_before: float
    residual_merit_trial: float | None
    residual_rms_trial_ev: float | None
    residual_max_trial_ev: float | None
    sufficient_decrease: bool
    accepted: bool
    failure_reason: str | None


@dataclass(frozen=True)
class OrbitalNewtonRootIteration:
    """Summary of one Newton system and its line search."""

    iteration: int
    residual_merit_before: float
    residual_rms_before_ev: float
    residual_max_before_ev: float
    raw_step_norm: float
    radius_limited_step_norm: float
    accepted_step_norm: float
    accepted_step_scale: float | None
    minres_info: int
    minres_iterations: int
    hessian_actions: int
    projector_evaluations: int
    backtracks: int
    accepted: bool
    exit_reason: str
    self_adjoint_absolute_defect_ev: float | None = None
    self_adjoint_relative_defect: float | None = None
    self_adjoint_maximum_hybrid_ratio: float | None = None


@dataclass(frozen=True)
class OrbitalNewtonRootRun:
    """Terminal state and immutable receipts for a Newton--MINRES run."""

    physical_projector: np.ndarray
    tangent_basis: np.ndarray
    evaluation: FixedRankHFEvaluation
    residual: np.ndarray
    residual_rms_ev: float
    residual_max_abs_ev: float
    residual_merit: float
    history: tuple[OrbitalNewtonRootIteration, ...]
    trial_history: tuple[OrbitalNewtonRootTrial, ...]
    outer_iterations: int
    hessian_actions: int
    projector_evaluations: int
    cumulative_distance: float
    converged: bool
    exit_reason: str
    initial_validator_accepted: bool | None
    final_validator_accepted: bool | None
    self_adjoint_absolute_defect_ev: float | None
    self_adjoint_relative_defect: float | None
    self_adjoint_maximum_hybrid_ratio: float | None


BudgetedJacobianAction = Callable[[np.ndarray], np.ndarray]
InitialCorrection = Callable[
    [OrbitalTangentFrame, np.ndarray, BudgetedJacobianAction], np.ndarray
]
TrialValidator = Callable[[np.ndarray, FixedRankHFEvaluation], bool]

@dataclass(frozen=True)
class _SelfAdjointnessProbeReport:
    pair_count: int
    maximum_absolute_defect: float
    maximum_relative_defect: float
    maximum_hybrid_ratio: float


class _HessianActionBudgetExhausted(RuntimeError):
    pass


def _validate_options(options: OrbitalNewtonRootOptions) -> None:
    integer_values = (
        options.max_outer_iterations,
        options.max_hessian_actions,
        options.max_projector_evaluations,
        options.minres_max_iterations,
        options.self_adjoint_probe_count,
        options.self_adjoint_seed,
        options.maximum_backtracks,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in integer_values
    ):
        raise ValueError("orbital Newton-root integer options must be integers")
    if (
        options.max_outer_iterations < 0
        or options.max_hessian_actions < 0
        or options.max_projector_evaluations < 1
        or options.minres_max_iterations <= 0
        or options.self_adjoint_probe_count < 2
        or options.self_adjoint_seed < 0
        or options.maximum_backtracks < 0
    ):
        raise ValueError("orbital Newton-root budgets are inconsistent")
    nonnegative = (
        options.residual_max_tolerance_ev,
        options.residual_rms_tolerance_ev,
        options.self_adjoint_absolute_tolerance_ev,
        options.self_adjoint_relative_tolerance,
        options.sufficient_decrease,
        options.projector_tolerance,
    )
    positive = (
        options.step_radius,
        options.maximum_cumulative_distance,
        options.minres_relative_tolerance,
        options.minimum_step_scale,
        options.self_adjoint_scale_floor_ev,
    )
    if (
        any(not np.isfinite(value) or value < 0.0 for value in nonnegative)
        or any(not np.isfinite(value) or value <= 0.0 for value in positive)
        or options.residual_max_tolerance_ev == 0.0
        or options.residual_rms_tolerance_ev == 0.0
        or (
            options.self_adjoint_absolute_tolerance_ev == 0.0
            and options.self_adjoint_relative_tolerance == 0.0
        )
        or not 0.0 < float(options.backtrack_factor) < 1.0
        or not 0.0 <= float(options.sufficient_decrease) < 1.0
        or float(options.minimum_step_scale) > 1.0
    ):
        raise ValueError("orbital Newton-root tolerances are inconsistent")


def _validate_evaluation(evaluation: FixedRankHFEvaluation) -> None:
    if not isinstance(evaluation, FixedRankHFEvaluation):
        raise ValueError("evaluate_projector must return FixedRankHFEvaluation")
    if not np.isfinite(evaluation.energy):
        raise ValueError("HF evaluation energy must be finite")


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    snapshot = np.array(value, dtype=dtype, copy=True)
    snapshot.setflags(write=False)
    return snapshot


def _snapshot_evaluation(
    evaluation: FixedRankHFEvaluation,
) -> FixedRankHFEvaluation:
    """Snapshot numerical evaluator state while leaving ``payload`` opaque.

    The Hamiltonian array is copied and made read-only.  The response callable
    and arbitrary payload cannot be generically cloned without changing their
    semantics, so providers remain responsible for their immutability.
    """

    _validate_evaluation(evaluation)
    return FixedRankHFEvaluation(
        energy=float(evaluation.energy),
        hamiltonian=_readonly_copy(
            evaluation.hamiltonian, dtype=np.dtype(np.complex128)
        ),
        hamiltonian_response=evaluation.hamiltonian_response,
        payload=evaluation.payload,
    )


def _validator_projector(projector: np.ndarray) -> np.ndarray:
    return _readonly_copy(projector, dtype=np.dtype(np.complex128))


def _probe_self_adjointness(
    action: BudgetedJacobianAction,
    *,
    size: int,
    probe_count: int,
    seed: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    scale_floor: float,
) -> _SelfAdjointnessProbeReport:
    rng = np.random.default_rng(seed)
    probes = rng.normal(size=(probe_count, size))
    norms = np.linalg.norm(probes, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("self-adjointness probe has zero norm")
    probes /= norms[:, None]
    actions = np.asarray([action(vector) for vector in probes], dtype=np.float64)
    maximum_absolute = 0.0
    maximum_relative = 0.0
    maximum_hybrid_ratio = 0.0
    pair_count = 0
    for left_index in range(probe_count):
        for right_index in range(left_index + 1, probe_count):
            lhs = float(np.dot(probes[left_index], actions[right_index]))
            rhs = float(np.dot(actions[left_index], probes[right_index]))
            defect = abs(lhs - rhs)
            scale = max(abs(lhs), abs(rhs), float(scale_floor))
            hybrid_bound = float(absolute_tolerance) + float(
                relative_tolerance
            ) * scale
            if hybrid_bound > 0.0:
                hybrid_ratio = defect / hybrid_bound
            else:
                hybrid_ratio = 0.0 if defect == 0.0 else float("inf")
            maximum_absolute = max(maximum_absolute, defect)
            maximum_relative = max(maximum_relative, defect / scale)
            maximum_hybrid_ratio = max(
                maximum_hybrid_ratio, hybrid_ratio
            )
            pair_count += 1
    return _SelfAdjointnessProbeReport(
        pair_count=pair_count,
        maximum_absolute_defect=maximum_absolute,
        maximum_relative_defect=maximum_relative,
        maximum_hybrid_ratio=maximum_hybrid_ratio,
    )


def _residual_metrics(
    hessian: ZeroTemperatureOrbitalHessian,
) -> tuple[np.ndarray, float, float, float]:
    residual = np.asarray(hessian.stationarity_residual, dtype=np.complex128)
    rms = float(np.linalg.norm(residual) / np.sqrt(residual.size))
    maximum = float(np.max(np.abs(residual), initial=0.0))
    packed = hessian.frame.pack_weighted_complex(residual)
    merit = float(np.linalg.norm(packed))
    return residual, rms, maximum, merit


def _converged(
    rms: float,
    maximum: float,
    options: OrbitalNewtonRootOptions,
) -> bool:
    return bool(
        maximum <= float(options.residual_max_tolerance_ev)
        and rms <= float(options.residual_rms_tolerance_ev)
    )



def run_orbital_newton_root(
    initial_projector: np.ndarray,
    evaluate_projector: Callable[[np.ndarray], FixedRankHFEvaluation],
    *,
    occupied_per_k: int,
    k_weights: np.ndarray | None = None,
    initial_basis: np.ndarray | None = None,
    options: OrbitalNewtonRootOptions = OrbitalNewtonRootOptions(),
    trial_validator: TrialValidator,
    initial_correction: InitialCorrection | None = None,
) -> OrbitalNewtonRootRun:
    """Polish a fixed-rank occupied projector to an orbital-stationarity root.

    ``initial_correction`` may provide a coarse-subspace initial guess ``x0``.
    It receives the recentered tangent frame and the same budgeted Jacobian
    action used by MINRES, so setup actions cannot evade the hard action
    budget.  No preconditioner API is exposed until an adapter can qualify the
    required SPD property.  The coarse correction does not change the signed
    equation ``H s = -g`` or the residual-only acceptance rule.
    ``trial_validator`` is a mandatory
    physical point gate and is called on the initial point, every evaluable
    line-search trial, and the final numerically converged point.

    Returned projector, tangent basis, residual, and evaluation Hamiltonian
    arrays are independent read-only snapshots.  ``evaluation.payload`` and
    ``evaluation.hamiltonian_response`` remain provider-owned opaque objects.
    If initial validation rejects or raises, the returned fail-closed receipt
    has empty tangent/residual arrays and NaN residual metrics because no
    Hessian or response action was authorized.
    """

    _validate_options(options)
    if trial_validator is None or not callable(trial_validator):
        raise ValueError("trial_validator must be an explicit callable")
    projector = np.asarray(initial_projector, dtype=np.complex128).copy()
    basis = None if initial_basis is None else np.asarray(initial_basis).copy()
    evaluation = _snapshot_evaluation(evaluate_projector(projector.copy()))
    projector_evaluations = 1
    hessian_actions = 0
    outer_iterations = 0
    cumulative_distance = 0.0
    history: list[OrbitalNewtonRootIteration] = []
    trial_history: list[OrbitalNewtonRootTrial] = []
    exit_reason: str | None = None
    initial_validator_accepted: bool | None = None
    final_validator_accepted: bool | None = None
    try:
        initial_validator_accepted = bool(
            trial_validator(_validator_projector(projector), evaluation)
        )
    except Exception:  # validator is an external fail-closed gate
        exit_reason = "initial_validator_failed"
    if initial_validator_accepted is False:
        exit_reason = "initial_validator_rejected"
    if exit_reason is not None:
        if basis is None:
            terminal_basis = np.empty((0, 0, 0), dtype=np.complex128)
        else:
            terminal_basis = np.asarray(basis, dtype=np.complex128)
        return OrbitalNewtonRootRun(
            physical_projector=_readonly_copy(
                projector, dtype=np.dtype(np.complex128)
            ),
            tangent_basis=_readonly_copy(
                terminal_basis, dtype=np.dtype(np.complex128)
            ),
            evaluation=_snapshot_evaluation(evaluation),
            residual=_readonly_copy(
                np.empty((0, 0, 0), dtype=np.complex128),
                dtype=np.dtype(np.complex128),
            ),
            residual_rms_ev=float("nan"),
            residual_max_abs_ev=float("nan"),
            residual_merit=float("nan"),
            history=(),
            trial_history=(),
            outer_iterations=0,
            hessian_actions=0,
            projector_evaluations=projector_evaluations,
            cumulative_distance=0.0,
            converged=False,
            exit_reason=exit_reason,
            initial_validator_accepted=initial_validator_accepted,
            final_validator_accepted=None,
            self_adjoint_absolute_defect_ev=None,
            self_adjoint_relative_defect=None,
            self_adjoint_maximum_hybrid_ratio=None,
        )

    hessian = build_zero_temperature_orbital_hessian(
        projector,
        np.asarray(evaluation.hamiltonian, dtype=np.complex128),
        evaluation.hamiltonian_response,
        occupied_per_k=int(occupied_per_k),
        k_weights=k_weights,
        projector_tolerance=float(options.projector_tolerance),
        tangent_basis=basis,
    )
    basis = np.asarray(hessian.frame.basis, dtype=np.complex128)
    residual, residual_rms, residual_max, residual_merit = _residual_metrics(hessian)

    while exit_reason is None and not _converged(
        residual_rms, residual_max, options
    ):
        if outer_iterations >= int(options.max_outer_iterations):
            exit_reason = "maximum_outer_iterations"
            break
        if projector_evaluations >= int(options.max_projector_evaluations):
            exit_reason = "projector_evaluation_budget_exhausted"
            break
        if hessian_actions >= int(options.max_hessian_actions):
            exit_reason = "hessian_action_budget_exhausted"
            break
        remaining_distance = (
            float(options.maximum_cumulative_distance) - cumulative_distance
        )
        if remaining_distance <= 0.0:
            exit_reason = "cumulative_distance_exhausted"
            break

        outer_iterations += 1
        residual_merit_before = residual_merit
        residual_rms_before = residual_rms
        residual_max_before = residual_max
        action_start = hessian_actions
        gradient = hessian.frame.pack_weighted_complex(
            hessian.stationarity_residual, factor=2.0
        )
        size = hessian.shape[0]
        minres_iterations = 0

        def counted_action(vector: np.ndarray) -> np.ndarray:
            nonlocal hessian_actions
            if hessian_actions >= int(options.max_hessian_actions):
                raise _HessianActionBudgetExhausted
            hessian_actions += 1
            output = np.asarray(
                hessian.energy_hessian_action(vector), dtype=np.float64
            )
            if output.shape != (size,) or not np.all(np.isfinite(output)):
                raise ValueError("orbital Jacobian action returned invalid values")
            return output

        operator = LinearOperator(
            (size, size), matvec=counted_action, dtype=np.dtype(np.float64)
        )
        def count_minres_iteration(_value: np.ndarray) -> None:
            nonlocal minres_iterations
            minres_iterations += 1

        minres_started = False
        self_adjoint_absolute_defect_ev: float | None = None
        self_adjoint_relative_defect: float | None = None
        self_adjoint_maximum_hybrid_ratio: float | None = None
        try:
            symmetry = _probe_self_adjointness(
                counted_action,
                size=size,
                probe_count=int(options.self_adjoint_probe_count),
                seed=int(options.self_adjoint_seed) + outer_iterations - 1,
                absolute_tolerance=float(
                    options.self_adjoint_absolute_tolerance_ev
                ),
                relative_tolerance=float(
                    options.self_adjoint_relative_tolerance
                ),
                scale_floor=float(options.self_adjoint_scale_floor_ev),
            )
            self_adjoint_absolute_defect_ev = (
                symmetry.maximum_absolute_defect
            )
            self_adjoint_relative_defect = symmetry.maximum_relative_defect
            self_adjoint_maximum_hybrid_ratio = (
                symmetry.maximum_hybrid_ratio
            )
            if self_adjoint_maximum_hybrid_ratio > 1.0:
                exit_reason = "jacobian_not_self_adjoint"
                history.append(
                    OrbitalNewtonRootIteration(
                        iteration=outer_iterations,
                        residual_merit_before=residual_merit,
                        residual_rms_before_ev=residual_rms,
                        residual_max_before_ev=residual_max,
                        raw_step_norm=np.nan,
                        radius_limited_step_norm=np.nan,
                        accepted_step_norm=0.0,
                        accepted_step_scale=None,
                        minres_info=-1,
                        minres_iterations=0,
                        hessian_actions=hessian_actions - action_start,
                        projector_evaluations=projector_evaluations,
                        backtracks=0,
                        accepted=False,
                        exit_reason=exit_reason,
                        self_adjoint_absolute_defect_ev=(
                            self_adjoint_absolute_defect_ev
                        ),
                        self_adjoint_relative_defect=(
                            self_adjoint_relative_defect
                        ),
                        self_adjoint_maximum_hybrid_ratio=(
                            self_adjoint_maximum_hybrid_ratio
                        ),
                    )
                )
                break

            x0 = None
            if initial_correction is not None:
                candidate = np.asarray(
                    initial_correction(
                        hessian.frame, -gradient.copy(), counted_action
                    ),
                    dtype=np.float64,
                )
                if candidate.shape != (size,) or not np.all(np.isfinite(candidate)):
                    raise ValueError("initial_correction returned an invalid vector")
                x0 = candidate
            if hessian_actions >= int(options.max_hessian_actions):
                raise _HessianActionBudgetExhausted
            minres_started = True
            step, minres_info = minres(
                operator,
                -gradient,
                x0=x0,
                rtol=float(options.minres_relative_tolerance),
                maxiter=min(
                    int(options.minres_max_iterations),
                    int(options.max_hessian_actions) - hessian_actions,
                ),
                callback=count_minres_iteration,
                check=False,
                show=False,
            )
        except _HessianActionBudgetExhausted:
            exit_reason = "hessian_action_budget_exhausted"
            history.append(
                OrbitalNewtonRootIteration(
                    iteration=outer_iterations,
                    residual_merit_before=residual_merit,
                    residual_rms_before_ev=residual_rms,
                    residual_max_before_ev=residual_max,
                    raw_step_norm=np.nan,
                    radius_limited_step_norm=np.nan,
                    accepted_step_norm=0.0,
                    accepted_step_scale=None,
                    minres_info=-1,
                    minres_iterations=minres_iterations,
                    hessian_actions=hessian_actions - action_start,
                    projector_evaluations=projector_evaluations,
                    backtracks=0,
                    accepted=False,
                    exit_reason=exit_reason,
                    self_adjoint_absolute_defect_ev=(
                        self_adjoint_absolute_defect_ev
                    ),
                    self_adjoint_relative_defect=(
                        self_adjoint_relative_defect
                    ),
                    self_adjoint_maximum_hybrid_ratio=(
                        self_adjoint_maximum_hybrid_ratio
                    ),
                )
            )
            break
        except Exception:
            exit_reason = (
                "minres_failed" if minres_started else "coarse_hook_failed"
            )
            history.append(
                OrbitalNewtonRootIteration(
                    iteration=outer_iterations,
                    residual_merit_before=residual_merit,
                    residual_rms_before_ev=residual_rms,
                    residual_max_before_ev=residual_max,
                    raw_step_norm=np.nan,
                    radius_limited_step_norm=np.nan,
                    accepted_step_norm=0.0,
                    accepted_step_scale=None,
                    minres_info=-1,
                    minres_iterations=minres_iterations,
                    hessian_actions=hessian_actions - action_start,
                    projector_evaluations=projector_evaluations,
                    backtracks=0,
                    accepted=False,
                    exit_reason=exit_reason,
                    self_adjoint_absolute_defect_ev=(
                        self_adjoint_absolute_defect_ev
                    ),
                    self_adjoint_relative_defect=(
                        self_adjoint_relative_defect
                    ),
                    self_adjoint_maximum_hybrid_ratio=(
                        self_adjoint_maximum_hybrid_ratio
                    ),
                )
            )
            break

        step = np.asarray(step, dtype=np.float64)
        if step.shape != (size,) or not np.all(np.isfinite(step)):
            minres_info = -1
        if int(minres_info) != 0:
            if (
                int(minres_info) > 0
                and hessian_actions >= int(options.max_hessian_actions)
            ):
                exit_reason = "hessian_action_budget_exhausted"
            else:
                exit_reason = (
                    "minres_iteration_limit"
                    if int(minres_info) > 0
                    else "minres_failed"
                )
            history.append(
                OrbitalNewtonRootIteration(
                    iteration=outer_iterations,
                    residual_merit_before=residual_merit,
                    residual_rms_before_ev=residual_rms,
                    residual_max_before_ev=residual_max,
                    raw_step_norm=float(np.linalg.norm(step)),
                    radius_limited_step_norm=0.0,
                    accepted_step_norm=0.0,
                    accepted_step_scale=None,
                    minres_info=int(minres_info),
                    minres_iterations=minres_iterations,
                    hessian_actions=hessian_actions - action_start,
                    projector_evaluations=projector_evaluations,
                    backtracks=0,
                    accepted=False,
                    exit_reason=exit_reason,
                    self_adjoint_absolute_defect_ev=(
                        self_adjoint_absolute_defect_ev
                    ),
                    self_adjoint_relative_defect=(
                        self_adjoint_relative_defect
                    ),
                    self_adjoint_maximum_hybrid_ratio=(
                        self_adjoint_maximum_hybrid_ratio
                    ),
                )
            )
            break

        raw_step_norm = float(np.linalg.norm(step))
        if not np.isfinite(raw_step_norm) or raw_step_norm == 0.0:
            exit_reason = "zero_newton_step"
            history.append(
                OrbitalNewtonRootIteration(
                    iteration=outer_iterations,
                    residual_merit_before=residual_merit,
                    residual_rms_before_ev=residual_rms,
                    residual_max_before_ev=residual_max,
                    raw_step_norm=raw_step_norm,
                    radius_limited_step_norm=0.0,
                    accepted_step_norm=0.0,
                    accepted_step_scale=None,
                    minres_info=int(minres_info),
                    minres_iterations=minres_iterations,
                    hessian_actions=hessian_actions - action_start,
                    projector_evaluations=projector_evaluations,
                    backtracks=0,
                    accepted=False,
                    exit_reason=exit_reason,
                    self_adjoint_absolute_defect_ev=(
                        self_adjoint_absolute_defect_ev
                    ),
                    self_adjoint_relative_defect=(
                        self_adjoint_relative_defect
                    ),
                    self_adjoint_maximum_hybrid_ratio=(
                        self_adjoint_maximum_hybrid_ratio
                    ),
                )
            )
            break

        allowed_norm = min(float(options.step_radius), remaining_distance)
        radius_scale = min(1.0, allowed_norm / raw_step_norm)
        bounded_step = radius_scale * step
        bounded_step_norm = float(np.linalg.norm(bounded_step))
        line_scale = 1.0
        accepted = False
        accepted_scale: float | None = None
        accepted_norm = 0.0
        backtracks = 0
        iteration_reason = "line_search_failed"

        for trial_index in range(int(options.maximum_backtracks) + 1):
            total_scale = radius_scale * line_scale
            trial_step = line_scale * bounded_step
            trial_norm = float(np.linalg.norm(trial_step))
            if line_scale < float(options.minimum_step_scale):
                break
            if projector_evaluations >= int(options.max_projector_evaluations):
                exit_reason = "projector_evaluation_budget_exhausted"
                iteration_reason = exit_reason
                break

            trial_basis = hessian.frame.unitary_basis(trial_step)
            trial_projector = np.empty_like(projector)
            for k_index in range(hessian.frame.nk):
                occupied = trial_basis[:, : hessian.frame.nocc, k_index]
                trial_projector[:, :, k_index] = occupied @ occupied.conj().T
            projector_evaluations += 1
            try:
                trial_evaluation = _snapshot_evaluation(
                    evaluate_projector(trial_projector.copy())
                )
            except Exception:  # evaluator is an external fail-closed boundary
                exit_reason = "invalid_trial_evaluation"
                iteration_reason = exit_reason
                trial_history.append(
                    OrbitalNewtonRootTrial(
                        iteration=outer_iterations,
                        trial=trial_index + 1,
                        step_scale=total_scale,
                        step_norm=trial_norm,
                        cumulative_distance_if_accepted=(
                            cumulative_distance + trial_norm
                        ),
                        energy_before=float(evaluation.energy),
                        energy_trial=None,
                        validator_accepted=None,
                        residual_merit_before=residual_merit,
                        residual_merit_trial=None,
                        residual_rms_trial_ev=None,
                        residual_max_trial_ev=None,
                        sufficient_decrease=False,
                        accepted=False,
                        failure_reason=exit_reason,
                    )
                )
                break

            try:
                validator_accepted = bool(
                    trial_validator(
                        _validator_projector(trial_projector), trial_evaluation
                    )
                )
            except Exception:  # validator is an external fail-closed gate
                exit_reason = "trial_validator_failed"
                iteration_reason = exit_reason
                trial_history.append(
                    OrbitalNewtonRootTrial(
                        iteration=outer_iterations,
                        trial=trial_index + 1,
                        step_scale=total_scale,
                        step_norm=trial_norm,
                        cumulative_distance_if_accepted=(
                            cumulative_distance + trial_norm
                        ),
                        energy_before=float(evaluation.energy),
                        energy_trial=float(trial_evaluation.energy),
                        validator_accepted=None,
                        residual_merit_before=residual_merit,
                        residual_merit_trial=None,
                        residual_rms_trial_ev=None,
                        residual_max_trial_ev=None,
                        sufficient_decrease=False,
                        accepted=False,
                        failure_reason=exit_reason,
                    )
                )
                break

            trial_hessian: ZeroTemperatureOrbitalHessian | None = None
            trial_rms: float | None = None
            trial_max: float | None = None
            trial_merit: float | None = None
            decrease = False
            if validator_accepted:
                try:
                    trial_hessian = build_zero_temperature_orbital_hessian(
                        trial_projector,
                        np.asarray(
                            trial_evaluation.hamiltonian, dtype=np.complex128
                        ),
                        trial_evaluation.hamiltonian_response,
                        occupied_per_k=int(occupied_per_k),
                        k_weights=k_weights,
                        projector_tolerance=float(options.projector_tolerance),
                        tangent_basis=trial_basis,
                    )
                except Exception:
                    exit_reason = "invalid_trial_hessian"
                    iteration_reason = exit_reason
                    trial_history.append(
                        OrbitalNewtonRootTrial(
                            iteration=outer_iterations,
                            trial=trial_index + 1,
                            step_scale=total_scale,
                            step_norm=trial_norm,
                            cumulative_distance_if_accepted=(
                                cumulative_distance + trial_norm
                            ),
                            energy_before=float(evaluation.energy),
                            energy_trial=float(trial_evaluation.energy),
                            validator_accepted=True,
                            residual_merit_before=residual_merit,
                            residual_merit_trial=None,
                            residual_rms_trial_ev=None,
                            residual_max_trial_ev=None,
                            sufficient_decrease=False,
                            accepted=False,
                            failure_reason=exit_reason,
                        )
                    )
                    break
                _, trial_rms, trial_max, trial_merit = _residual_metrics(
                    trial_hessian
                )
                target = residual_merit * (
                    1.0 - float(options.sufficient_decrease) * total_scale
                )
                decrease = bool(
                    np.isfinite(trial_merit) and trial_merit <= target
                )

            trial_accepted = bool(validator_accepted and decrease)
            trial_history.append(
                OrbitalNewtonRootTrial(
                    iteration=outer_iterations,
                    trial=trial_index + 1,
                    step_scale=total_scale,
                    step_norm=trial_norm,
                    cumulative_distance_if_accepted=(
                        cumulative_distance + trial_norm
                    ),
                    energy_before=float(evaluation.energy),
                    energy_trial=float(trial_evaluation.energy),
                    validator_accepted=validator_accepted,
                    residual_merit_before=residual_merit,
                    residual_merit_trial=trial_merit,
                    residual_rms_trial_ev=trial_rms,
                    residual_max_trial_ev=trial_max,
                    sufficient_decrease=decrease,
                    accepted=trial_accepted,
                    failure_reason=None,
                )
            )
            if trial_accepted:
                if trial_hessian is None:
                    raise RuntimeError("accepted trial has no orbital Hessian")
                projector = trial_projector
                basis = np.asarray(trial_hessian.frame.basis, dtype=np.complex128)
                evaluation = trial_evaluation
                hessian = trial_hessian
                residual, residual_rms, residual_max, residual_merit = (
                    _residual_metrics(hessian)
                )
                cumulative_distance += trial_norm
                accepted = True
                accepted_scale = total_scale
                accepted_norm = trial_norm
                iteration_reason = "accepted"
                break

            if trial_index < int(options.maximum_backtracks):
                backtracks += 1
                line_scale *= float(options.backtrack_factor)

        history.append(
            OrbitalNewtonRootIteration(
                iteration=outer_iterations,
                residual_merit_before=residual_merit_before,
                residual_rms_before_ev=residual_rms_before,
                residual_max_before_ev=residual_max_before,
                raw_step_norm=raw_step_norm,
                radius_limited_step_norm=bounded_step_norm,
                accepted_step_norm=accepted_norm,
                accepted_step_scale=accepted_scale,
                minres_info=int(minres_info),
                minres_iterations=minres_iterations,
                hessian_actions=hessian_actions - action_start,
                projector_evaluations=projector_evaluations,
                backtracks=backtracks,
                accepted=accepted,
                exit_reason=iteration_reason,
                self_adjoint_absolute_defect_ev=(
                    self_adjoint_absolute_defect_ev
                ),
                self_adjoint_relative_defect=self_adjoint_relative_defect,
                self_adjoint_maximum_hybrid_ratio=(
                    self_adjoint_maximum_hybrid_ratio
                ),
            )
        )
        if not accepted:
            if exit_reason is None:
                exit_reason = iteration_reason
            break

    numerically_converged = _converged(residual_rms, residual_max, options)
    if (
        exit_reason is None
        and numerically_converged
        and initial_validator_accepted is True
    ):
        try:
            final_validator_accepted = bool(
                trial_validator(_validator_projector(projector), evaluation)
            )
        except Exception:  # validator is an external fail-closed gate
            exit_reason = "final_validator_failed"
        if final_validator_accepted is False:
            exit_reason = "final_validator_rejected"
        elif final_validator_accepted is True:
            exit_reason = "converged"

    converged = bool(
        numerically_converged
        and initial_validator_accepted is True
        and final_validator_accepted is True
        and exit_reason == "converged"
    )
    if exit_reason is None:
        exit_reason = "failed_without_exit_reason"
    absolute_defects = [
        item.self_adjoint_absolute_defect_ev
        for item in history
        if item.self_adjoint_absolute_defect_ev is not None
    ]
    relative_defects = [
        item.self_adjoint_relative_defect
        for item in history
        if item.self_adjoint_relative_defect is not None
    ]
    hybrid_ratios = [
        item.self_adjoint_maximum_hybrid_ratio
        for item in history
        if item.self_adjoint_maximum_hybrid_ratio is not None
    ]
    return OrbitalNewtonRootRun(
        physical_projector=_readonly_copy(
            projector, dtype=np.dtype(np.complex128)
        ),
        tangent_basis=_readonly_copy(basis, dtype=np.dtype(np.complex128)),
        evaluation=_snapshot_evaluation(evaluation),
        residual=_readonly_copy(residual, dtype=np.dtype(np.complex128)),
        residual_rms_ev=residual_rms,
        residual_max_abs_ev=residual_max,
        residual_merit=residual_merit,
        history=tuple(history),
        trial_history=tuple(trial_history),
        outer_iterations=outer_iterations,
        hessian_actions=hessian_actions,
        projector_evaluations=projector_evaluations,
        cumulative_distance=cumulative_distance,
        converged=converged,
        exit_reason=exit_reason,
        initial_validator_accepted=initial_validator_accepted,
        final_validator_accepted=final_validator_accepted,
        self_adjoint_absolute_defect_ev=(
            max(absolute_defects) if absolute_defects else None
        ),
        self_adjoint_relative_defect=(
            max(relative_defects) if relative_defects else None
        ),
        self_adjoint_maximum_hybrid_ratio=(
            max(hybrid_ratios) if hybrid_ratios else None
        ),
    )


__all__ = [
    "BudgetedJacobianAction",
    "InitialCorrection",
    "OrbitalNewtonRootIteration",
    "OrbitalNewtonRootOptions",
    "OrbitalNewtonRootRun",
    "OrbitalNewtonRootTrial",
    "TrialValidator",
    "run_orbital_newton_root",
]
