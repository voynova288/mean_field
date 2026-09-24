from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .engine import (
    DensityUpdateResult,
    FinalAcceptanceResult,
    HartreeFockRun,
    HartreeFockStateProtocol,
    HartreeFockStepResult,
    InteractionEnergyResult,
    run_hartree_fock_iterations,
)


class HartreeFockInitializerProtocol(Protocol):
    def __call__(self, state: HartreeFockStateProtocol, *, init_mode: str, seed: int) -> None: ...


class HartreeFockInteractionProtocol(Protocol):
    def __call__(self, density: np.ndarray) -> np.ndarray: ...


class ProjectedDensityBuilderProtocol(Protocol):
    def __call__(self, hamiltonian: np.ndarray) -> DensityUpdateResult: ...


@dataclass(frozen=True)
class HartreeFockKernel:
    interaction_builder: HartreeFockInteractionProtocol
    density_builder: ProjectedDensityBuilderProtocol
    energy_functional: Callable[[np.ndarray, np.ndarray, np.ndarray], float]
    interaction_energy_builder: Callable[
        [np.ndarray, np.ndarray], InteractionEnergyResult
    ] | None = None
    oda_parameterizer: Callable[[HartreeFockStateProtocol, np.ndarray], float] | None = None
    oda_delta_interaction_builder: Callable[[np.ndarray], np.ndarray] | None = None
    hamiltonian_postprocessor: Callable[[np.ndarray], None] | None = None
    density_postprocessor: Callable[[np.ndarray], None] | None = None
    step_callback: Callable[[HartreeFockStateProtocol, HartreeFockStepResult], None] | None = None
    iteration_convergence_gate: Callable[
        [HartreeFockStateProtocol, HartreeFockStepResult, bool], bool
    ] | None = None
    final_state_callback: Callable[[HartreeFockStateProtocol, DensityUpdateResult], None] | None = None
    final_acceptance: Callable[
        [HartreeFockStateProtocol, DensityUpdateResult, float], FinalAcceptanceResult
    ] | None = None
    convergence_metric: Callable[[np.ndarray, np.ndarray], float] | None = None
    convergence_rule: Literal["raw", "mixed"] = "raw"


@dataclass(frozen=True)
class HartreeFockProblem:
    initializer: HartreeFockInitializerProtocol
    kernel: HartreeFockKernel


def run_hartree_fock_problem(
    state: HartreeFockStateProtocol,
    problem: HartreeFockProblem,
    *,
    init_mode: str,
    seed: int,
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
    max_oda_lambda: float | None = None,
) -> HartreeFockRun:
    execution_problem = problem
    validate_before_execution = getattr(
        type(problem), "validate_before_execution", None
    )
    if validate_before_execution is not None:
        if not callable(validate_before_execution):
            raise TypeError("problem validate_before_execution hook must be callable")
        validated = validate_before_execution(problem, state)
        if validated is not None:
            if not isinstance(validated, HartreeFockProblem):
                raise TypeError(
                    "problem validate_before_execution hook must return "
                    "HartreeFockProblem or None"
                )
            execution_problem = validated
    initializer = execution_problem.initializer
    kernel = execution_problem.kernel
    initializer(state, init_mode=init_mode, seed=seed)
    return run_hartree_fock_iterations(
        state,
        init_mode=init_mode,
        seed=seed,
        interaction_builder=kernel.interaction_builder,
        density_builder=kernel.density_builder,
        energy_functional=kernel.energy_functional,
        interaction_energy_builder=kernel.interaction_energy_builder,
        oda_parameterizer=kernel.oda_parameterizer,
        oda_delta_interaction_builder=kernel.oda_delta_interaction_builder,
        hamiltonian_postprocessor=kernel.hamiltonian_postprocessor,
        density_postprocessor=kernel.density_postprocessor,
        step_callback=kernel.step_callback,
        iteration_convergence_gate=kernel.iteration_convergence_gate,
        final_state_callback=kernel.final_state_callback,
        final_acceptance=kernel.final_acceptance,
        convergence_metric=kernel.convergence_metric,
        convergence_rule=kernel.convergence_rule,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
        max_oda_lambda=max_oda_lambda,
    )
