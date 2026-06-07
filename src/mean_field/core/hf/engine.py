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
    oda_parameterizer: Callable[[HartreeFockStateProtocol, np.ndarray], float] | None = None,
    oda_delta_interaction_builder: Callable[[np.ndarray], np.ndarray] | None = None,
    hamiltonian_postprocessor: Callable[[np.ndarray], None] | None = None,
    density_postprocessor: Callable[[np.ndarray], None] | None = None,
    step_callback: Callable[[HartreeFockStateProtocol, HartreeFockStepResult], None] | None = None,
    final_state_callback: Callable[[HartreeFockStateProtocol, DensityUpdateResult], None] | None = None,
    convergence_rule: Literal["raw", "mixed"] = "raw",
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
) -> HartreeFockRun:
    if convergence_rule not in {"raw", "mixed"}:
        raise ValueError(f"Unsupported convergence_rule={convergence_rule!r}")

    iter_energy: list[float] = []
    iter_err: list[float] = []
    iter_oda: list[float] = []
    exit_reason = "max_iter"
    cached_interaction_h: np.ndarray | None = None

    for iteration in range(1, max_iter + 1):
        previous_density = state.density.copy()
        state.hamiltonian[:, :, :] = state.h0
        interaction_h_from_cache = cached_interaction_h is not None
        if cached_interaction_h is None:
            interaction_h = interaction_builder(previous_density)
        else:
            interaction_h = cached_interaction_h
        cached_interaction_h = None
        state.hamiltonian[:, :, :] += interaction_h
        oda_base_interaction_h = interaction_h
        if hamiltonian_postprocessor is not None:
            hamiltonian_postprocessor(state.hamiltonian)
            oda_base_interaction_h = state.hamiltonian - state.h0

        energy = float(energy_functional(interaction_h, state.h0, previous_density))
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
        mixed_density = oda_lambda * density_update.density + (1.0 - oda_lambda) * previous_density

        norm_raw = float(calculate_norm_convergence(density_update.density, previous_density))
        norm_mixed = float(calculate_norm_convergence(mixed_density, previous_density))
        norm_selected = norm_raw if convergence_rule == "raw" else norm_mixed

        state.density[:, :, :] = mixed_density
        if density_postprocessor is not None:
            density_postprocessor(state.density)
        elif delta_interaction_h is not None and hamiltonian_postprocessor is None:
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
            step_callback(state, step_result)

        iter_energy.append(energy)
        iter_err.append(norm_selected)
        iter_oda.append(oda_lambda)

        raw_converged = norm_raw <= state.precision
        mixed_converged = norm_mixed <= state.precision
        converged = raw_converged if convergence_rule == "raw" else mixed_converged
        if converged:
            exit_reason = "converged"
            break
        if oda_lambda < oda_stall_threshold:
            exit_reason = "oda_stall"
            break

    if cached_interaction_h is None:
        final_interaction_h = interaction_builder(state.density)
    else:
        final_interaction_h = cached_interaction_h
    state.hamiltonian[:, :, :] = state.h0
    state.hamiltonian[:, :, :] += final_interaction_h
    if hamiltonian_postprocessor is not None:
        hamiltonian_postprocessor(state.hamiltonian)
    final_density_update = density_builder(state.hamiltonian)
    state.energies[:, :] = final_density_update.energies
    state.mu = float(final_density_update.mu)
    final_energy = float(energy_functional(final_interaction_h, state.h0, state.density))
    final_raw_density = np.asarray(final_density_update.density, dtype=np.complex128).copy()
    if density_postprocessor is not None:
        density_postprocessor(final_raw_density)
    state.diagnostics["hf_energy"] = final_energy
    state.diagnostics["final_raw_norm"] = float(calculate_norm_convergence(final_raw_density, state.density))
    if final_state_callback is not None:
        final_state_callback(state, final_density_update)

    return HartreeFockRun(
        state=state,
        iter_energy=np.asarray(iter_energy, dtype=float),
        iter_err=np.asarray(iter_err, dtype=float),
        iter_oda=np.asarray(iter_oda, dtype=float),
        init_mode=init_mode,
        seed=int(seed),
        converged=exit_reason == "converged",
        exit_reason=exit_reason,
    )
