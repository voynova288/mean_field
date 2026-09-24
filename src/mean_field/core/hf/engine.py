from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from .occupations import calculate_norm_convergence


class HartreeFockStateProtocol(Protocol):
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    mu: float
    precision: float
    diagnostics: dict[str, float]

    @property
    def nk(self) -> int: ...


@dataclass(frozen=True)
class DensityUpdateResult:
    density: np.ndarray
    energies: np.ndarray
    mu: float
    observables: dict[str, np.ndarray | float] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalAcceptanceResult:
    """Fail-closed final-state acceptance supplied by a system adapter."""

    accepted: bool
    reason: str = "accepted"
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class InteractionEnergyResult:
    """One-density interaction Hamiltonian and its matching scalar energy."""

    interaction_h: np.ndarray
    energy: float


@dataclass(frozen=True)
class HartreeFockStepResult:
    iteration: int
    previous_density: np.ndarray
    interaction_h: np.ndarray
    total_hamiltonian: np.ndarray
    density_update: DensityUpdateResult
    mixed_density: np.ndarray
    oda_lambda: float
    norm_raw: float
    norm_mixed: float
    norm_selected: float
    energy: float
    delta_interaction_h: np.ndarray | None = None
    interaction_h_from_cache: bool = False

    @property
    def density_new(self) -> np.ndarray:
        return self.density_update.density

    @property
    def energies(self) -> np.ndarray:
        return self.density_update.energies

    @property
    def mu(self) -> float:
        return float(self.density_update.mu)


@dataclass(frozen=True)
class FixedMixingRun:
    density: np.ndarray
    interaction_h: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    iter_err: np.ndarray
    converged: bool
    exit_reason: str

    @property
    def iterations(self) -> int:
        return int(self.iter_err.size)


@dataclass(frozen=True)
class HartreeFockRun:
    state: HartreeFockStateProtocol
    iter_energy: np.ndarray
    iter_err: np.ndarray
    iter_oda: np.ndarray
    init_mode: str
    seed: int
    converged: bool
    exit_reason: str

    @property
    def iterations(self) -> int:
        return int(self.iter_err.size)


def compute_oda_parameter(
    state: HartreeFockStateProtocol,
    delta_density: np.ndarray,
    *,
    interaction_builder: Callable[[np.ndarray], np.ndarray] | None = None,
    delta_h: np.ndarray | None = None,
    interaction_h: np.ndarray | None = None,
) -> float:
    if delta_h is None:
        if interaction_builder is None:
            raise ValueError("Either interaction_builder or delta_h must be provided for ODA.")
        delta_h = interaction_builder(delta_density)
    if interaction_h is None:
        interaction_h = state.hamiltonian - state.h0
    # Match Julia B0's stored-projector convention:
    # `tr(transpose(delta_P) * delta_H) == sum_ab delta_P[a,b] * delta_H[a,b]`.
    a = np.einsum("abk,abk->", delta_density, delta_h, optimize=True)
    b = np.einsum("abk,abk->", delta_density, state.h0, optimize=True)
    b += np.einsum("abk,abk->", delta_density, interaction_h, optimize=True) / 2.0
    b += np.einsum("abk,abk->", state.density, delta_h, optimize=True) / 2.0
    a = float(a.real / state.nk)
    b = float(b.real / state.nk)

    if abs(a) < 1e-15:
        return 1.0 if b < 0.0 else 0.0

    lambda0 = -b / a
    if a > 0.0:
        if lambda0 <= 0.0:
            return 0.0
        if lambda0 < 1.0:
            return float(lambda0)
        return 1.0
    if lambda0 <= 0.5:
        return 1.0
    return 0.0


def run_fixed_mixing_scf(
    h0: np.ndarray,
    initial_density: np.ndarray,
    *,
    interaction_builder: Callable[[np.ndarray], np.ndarray],
    density_builder: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    max_iter: int = 300,
    mixing: float = 0.5,
    precision: float = 1.0e-8,
) -> FixedMixingRun:
    h0_arr = np.asarray(h0, dtype=np.complex128)
    density = np.asarray(initial_density, dtype=np.complex128).copy()
    if density.shape != h0_arr.shape:
        raise ValueError(f"initial_density shape {density.shape} does not match h0 shape {h0_arr.shape}")
    mix = float(mixing)
    iter_err: list[float] = []
    exit_reason = "max_iter"
    for _iteration in range(1, int(max_iter) + 1):
        interaction_h = np.asarray(interaction_builder(density), dtype=np.complex128)
        hamiltonian = h0_arr + interaction_h
        raw_density, _energies = density_builder(hamiltonian)
        raw_density = np.asarray(raw_density, dtype=np.complex128)
        mixed_density = mix * raw_density + (1.0 - mix) * density
        norm = calculate_norm_convergence(raw_density, density)
        iter_err.append(norm)
        density = mixed_density
        if norm <= float(precision):
            exit_reason = "converged"
            break
    final_interaction_h = np.asarray(interaction_builder(density), dtype=np.complex128)
    final_hamiltonian = h0_arr + final_interaction_h
    _final_density, final_energies = density_builder(final_hamiltonian)
    return FixedMixingRun(
        density=density,
        interaction_h=final_interaction_h,
        hamiltonian=final_hamiltonian,
        energies=np.asarray(final_energies, dtype=float),
        iter_err=np.asarray(iter_err, dtype=float),
        converged=exit_reason == "converged",
        exit_reason=exit_reason,
    )


def run_hartree_fock_iterations(
    state: HartreeFockStateProtocol,
    *,
    init_mode: str,
    seed: int,
    interaction_builder: Callable[[np.ndarray], np.ndarray],
    density_builder: Callable[[np.ndarray], DensityUpdateResult],
    energy_functional: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    interaction_energy_builder: Callable[
        [np.ndarray, np.ndarray], InteractionEnergyResult
    ]
    | None = None,
    oda_parameterizer: Callable[[HartreeFockStateProtocol, np.ndarray], float] | None = None,
    oda_delta_interaction_builder: Callable[[np.ndarray], np.ndarray] | None = None,
    hamiltonian_postprocessor: Callable[[np.ndarray], None] | None = None,
    density_postprocessor: Callable[[np.ndarray], None] | None = None,
    step_callback: Callable[[HartreeFockStateProtocol, HartreeFockStepResult], None] | None = None,
    iteration_convergence_gate: Callable[
        [HartreeFockStateProtocol, HartreeFockStepResult, bool], bool
    ]
    | None = None,
    final_state_callback: Callable[[HartreeFockStateProtocol, DensityUpdateResult], None] | None = None,
    final_acceptance: Callable[
        [HartreeFockStateProtocol, DensityUpdateResult, float], FinalAcceptanceResult
    ]
    | None = None,
    convergence_metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    convergence_rule: Literal["raw", "mixed"] = "raw",
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
    max_oda_lambda: float | None = None,
) -> HartreeFockRun:
    if convergence_rule not in {"raw", "mixed"}:
        raise ValueError(f"Unsupported convergence_rule={convergence_rule!r}")
    solver_precision = float(state.precision)
    if not np.isfinite(solver_precision) or solver_precision < 0.0:
        raise ValueError(
            f"state.precision must be finite and nonnegative, got {state.precision!r}"
        )
    engine_diagnostic_keys = {
        "final_raw_norm",
        "final_accepted",
        "hf_energy",
        "oda_parameter",
        "iterations",
    }

    def invoke_protected_callback(callback: Callable[..., object], *args: object) -> object:
        protected = {
            key: state.diagnostics[key]
            for key in engine_diagnostic_keys
            if key in state.diagnostics
        }
        try:
            return callback(*args)
        finally:
            state.precision = solver_precision
            for key in engine_diagnostic_keys - protected.keys():
                state.diagnostics.pop(key, None)
            state.diagnostics.update(protected)

    if max_oda_lambda is not None and not (0.0 < float(max_oda_lambda) <= 1.0):
        raise ValueError(f"max_oda_lambda must be in (0, 1], got {max_oda_lambda!r}")

    iter_energy: list[float] = []
    iter_err: list[float] = []
    iter_oda: list[float] = []
    exit_reason = "max_iter"
    cached_interaction_h: np.ndarray | None = None

    for iteration in range(1, max_iter + 1):
        previous_density = state.density.copy()
        state.hamiltonian[:, :, :] = state.h0
        interaction_h_from_cache = cached_interaction_h is not None
        interaction_energy: InteractionEnergyResult | None = None
        if cached_interaction_h is None:
            if interaction_energy_builder is None:
                interaction_h = interaction_builder(previous_density)
            else:
                interaction_energy = interaction_energy_builder(
                    previous_density, state.h0
                )
                if not isinstance(interaction_energy, InteractionEnergyResult):
                    raise TypeError(
                        "interaction_energy_builder must return InteractionEnergyResult"
                    )
                interaction_h = np.asarray(
                    interaction_energy.interaction_h, dtype=np.complex128
                )
        else:
            interaction_h = cached_interaction_h
        cached_interaction_h = None
        state.hamiltonian[:, :, :] += interaction_h
        oda_base_interaction_h = interaction_h
        if hamiltonian_postprocessor is not None:
            hamiltonian_postprocessor(state.hamiltonian)
            oda_base_interaction_h = state.hamiltonian - state.h0

        energy = (
            float(interaction_energy.energy)
            if interaction_energy is not None
            else float(energy_functional(interaction_h, state.h0, previous_density))
        )
        density_update = density_builder(state.hamiltonian)
        delta_density = density_update.density - previous_density
        delta_interaction_h: np.ndarray | None = None
        if oda_delta_interaction_builder is not None:
            delta_interaction_h = oda_delta_interaction_builder(delta_density)
            if hamiltonian_postprocessor is not None:
                delta_interaction_h = np.asarray(delta_interaction_h, dtype=np.complex128).copy()
                hamiltonian_postprocessor(delta_interaction_h)
            oda_lambda = compute_oda_parameter(
                state,
                delta_density,
                delta_h=delta_interaction_h,
                interaction_h=oda_base_interaction_h,
            )
        else:
            oda_lambda = 1.0 if oda_parameterizer is None else float(oda_parameterizer(state, delta_density))
        if max_oda_lambda is not None:
            oda_lambda = min(float(oda_lambda), float(max_oda_lambda))
        mixed_density = oda_lambda * density_update.density + (1.0 - oda_lambda) * previous_density

        metric = calculate_norm_convergence if convergence_metric is None else convergence_metric
        norm_raw = float(metric(density_update.density, previous_density))
        norm_mixed = float(metric(mixed_density, previous_density))
        if not np.isfinite(norm_raw) or not np.isfinite(norm_mixed) or norm_raw < 0.0 or norm_mixed < 0.0:
            raise ValueError(
                "convergence_metric must return finite nonnegative values; "
                f"got raw={norm_raw!r}, mixed={norm_mixed!r}"
            )
        norm_selected = norm_raw if convergence_rule == "raw" else norm_mixed

        state.density[:, :, :] = mixed_density
        if density_postprocessor is not None:
            density_postprocessor(state.density)
        elif (
            delta_interaction_h is not None
            and hamiltonian_postprocessor is None
            and interaction_energy_builder is None
        ):
            cached_interaction_h = interaction_h + oda_lambda * delta_interaction_h
        state.energies[:, :] = density_update.energies
        state.mu = float(density_update.mu)
        state.diagnostics["hf_energy"] = energy
        state.diagnostics["oda_parameter"] = oda_lambda
        state.diagnostics["iterations"] = float(iteration)

        step_result = HartreeFockStepResult(
            iteration=iteration,
            previous_density=previous_density,
            interaction_h=interaction_h,
            total_hamiltonian=state.hamiltonian.copy(),
            density_update=density_update,
            mixed_density=mixed_density,
            oda_lambda=oda_lambda,
            norm_raw=norm_raw,
            norm_mixed=norm_mixed,
            norm_selected=norm_selected,
            energy=energy,
            delta_interaction_h=delta_interaction_h,
            interaction_h_from_cache=interaction_h_from_cache,
        )
        if step_callback is not None:
            invoke_protected_callback(step_callback, state, step_result)

        iter_energy.append(energy)
        iter_err.append(norm_selected)
        iter_oda.append(oda_lambda)

        raw_converged = norm_raw <= solver_precision
        mixed_converged = norm_mixed <= solver_precision
        density_converged = (
            raw_converged if convergence_rule == "raw" else mixed_converged
        )
        gate_passed = True
        if iteration_convergence_gate is not None:
            gate_passed = invoke_protected_callback(
                iteration_convergence_gate,
                state,
                step_result,
                density_converged,
            )
            if type(gate_passed) is not bool:
                raise TypeError(
                    "iteration_convergence_gate must return an exact Boolean"
                )
        converged = density_converged and gate_passed
        if converged:
            exit_reason = "converged"
            break
        if not raw_converged and oda_lambda < oda_stall_threshold:
            exit_reason = "oda_stall"
            break

    final_interaction_energy: InteractionEnergyResult | None = None
    if cached_interaction_h is None:
        if interaction_energy_builder is None:
            final_interaction_h = interaction_builder(state.density)
        else:
            final_interaction_energy = interaction_energy_builder(
                state.density, state.h0
            )
            if not isinstance(final_interaction_energy, InteractionEnergyResult):
                raise TypeError(
                    "interaction_energy_builder must return InteractionEnergyResult"
                )
            final_interaction_h = np.asarray(
                final_interaction_energy.interaction_h, dtype=np.complex128
            )
    else:
        final_interaction_h = cached_interaction_h
    state.hamiltonian[:, :, :] = state.h0
    state.hamiltonian[:, :, :] += final_interaction_h
    if hamiltonian_postprocessor is not None:
        hamiltonian_postprocessor(state.hamiltonian)
    final_density_update = density_builder(state.hamiltonian)
    state.energies[:, :] = final_density_update.energies
    state.mu = float(final_density_update.mu)
    final_energy = (
        float(final_interaction_energy.energy)
        if final_interaction_energy is not None
        else float(energy_functional(final_interaction_h, state.h0, state.density))
    )
    final_raw_density = np.asarray(final_density_update.density, dtype=np.complex128).copy()
    if density_postprocessor is not None:
        density_postprocessor(final_raw_density)
    state.diagnostics["hf_energy"] = final_energy
    final_metric = calculate_norm_convergence if convergence_metric is None else convergence_metric
    final_norm = float(final_metric(final_raw_density, state.density))
    if not np.isfinite(final_norm) or final_norm < 0.0:
        raise ValueError(f"convergence_metric returned invalid final norm {final_norm!r}")
    state.diagnostics["final_raw_norm"] = final_norm
    if final_state_callback is not None:
        invoke_protected_callback(final_state_callback, state, final_density_update)

    iteration_declared_converged = exit_reason == "converged"
    final_accepted = final_norm <= solver_precision
    final_rejection_reason = "final_raw_residual"
    if final_acceptance is not None:
        acceptance = invoke_protected_callback(
            final_acceptance, state, final_density_update, final_norm
        )
        if not isinstance(acceptance, FinalAcceptanceResult):
            raise TypeError("final_acceptance must return FinalAcceptanceResult")
        if type(acceptance.accepted) is not bool:
            raise TypeError("FinalAcceptanceResult.accepted must be an exact Boolean")
        if type(acceptance.reason) is not str or not acceptance.reason:
            raise ValueError("final_acceptance reason must be a nonempty exact string")
        if not acceptance.accepted and acceptance.reason == "converged":
            raise ValueError("a rejected final state cannot use reason='converged'")
        for key, value in acceptance.diagnostics.items():
            if not isinstance(key, str) or not key:
                raise ValueError("final_acceptance diagnostic keys must be nonempty strings")
            if key in engine_diagnostic_keys:
                raise ValueError(
                    f"final_acceptance diagnostic {key!r} collides with an engine-owned key"
                )
            scalar = float(value)
            if not np.isfinite(scalar):
                raise ValueError(
                    f"final_acceptance diagnostic {key!r} must be finite, got {value!r}"
                )
            state.diagnostics[key] = scalar
        if not acceptance.accepted:
            if final_accepted:
                final_rejection_reason = acceptance.reason
            final_accepted = False
    state.diagnostics["final_accepted"] = float(final_accepted)
    final_converged = iteration_declared_converged and final_accepted
    if iteration_declared_converged and not final_accepted:
        exit_reason = final_rejection_reason

    return HartreeFockRun(
        state=state,
        iter_energy=np.asarray(iter_energy, dtype=float),
        iter_err=np.asarray(iter_err, dtype=float),
        iter_oda=np.asarray(iter_oda, dtype=float),
        init_mode=init_mode,
        seed=int(seed),
        converged=final_converged,
        exit_reason=exit_reason,
    )
