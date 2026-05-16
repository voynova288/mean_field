from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .engine import (
    DensityUpdateResult,
    HartreeFockRun,
    HartreeFockStateProtocol,
    HartreeFockStepResult,
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
    oda_parameterizer: Callable[[HartreeFockStateProtocol, np.ndarray], float] | None = None
    oda_delta_interaction_builder: Callable[[np.ndarray], np.ndarray] | None = None
    hamiltonian_postprocessor: Callable[[np.ndarray], None] | None = None
    density_postprocessor: Callable[[np.ndarray], None] | None = None
    step_callback: Callable[[HartreeFockStateProtocol, HartreeFockStepResult], None] | None = None
    final_state_callback: Callable[[HartreeFockStateProtocol, DensityUpdateResult], None] | None = None
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
) -> HartreeFockRun:
    problem.initializer(state, init_mode=init_mode, seed=seed)
    return run_hartree_fock_iterations(
        state,
        init_mode=init_mode,
        seed=seed,
        interaction_builder=problem.kernel.interaction_builder,
        density_builder=problem.kernel.density_builder,
        energy_functional=problem.kernel.energy_functional,
        oda_parameterizer=problem.kernel.oda_parameterizer,
        oda_delta_interaction_builder=problem.kernel.oda_delta_interaction_builder,
        hamiltonian_postprocessor=problem.kernel.hamiltonian_postprocessor,
        density_postprocessor=problem.kernel.density_postprocessor,
        step_callback=problem.kernel.step_callback,
        final_state_callback=problem.kernel.final_state_callback,
        convergence_rule=problem.kernel.convergence_rule,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
    )
