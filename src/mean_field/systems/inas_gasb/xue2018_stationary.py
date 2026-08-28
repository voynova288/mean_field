"""Stationary-root solvers for the Xue--MacDonald Q=0 TRS branch."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from mean_field.core.hf.excitonic import FixedOccupation, fermi_density_from_hamiltonian
from mean_field.core.hf.stationary import (
    StationarySolveConfig,
    StationarySolveResult,
    pack_hermitian_matrix_field,
    solve_stationary_residual,
    unpack_hermitian_matrix_field,
)

from .xue2018 import xue2018_standard_parameters
from .xue2018_hf import (
    Xue2018HFState,
    xue2018_global_neutral_projector,
    xue2018_interaction_action,
    xue2018_reference_relative_energy_density,
)
from .xue2018_symmetry import project_xue2018_full_trs, xue2018_trs_residual
from .zeng2022 import build_zeng2022_folded_h0

Array = np.ndarray


@dataclass(frozen=True)
class Xue2018StationaryConfig:
    """Physical and numerical contract for one stationary-root solve."""

    thermal_energy_ry: float = 0.0
    root: StationarySolveConfig = StationarySolveConfig()
    trs_tolerance: float = 1.0e-10
    full_residual_rms_tolerance: float = 1.0e-9
    full_residual_max_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.thermal_energy_ry) or self.thermal_energy_ry < 0.0:
            raise ValueError("thermal_energy_ry must be finite and nonnegative")
        for value in (
            self.trs_tolerance,
            self.full_residual_rms_tolerance,
            self.full_residual_max_tolerance,
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("stationary acceptance tolerances must be finite and positive")


@dataclass(frozen=True)
class Xue2018DensityMap:
    density_delta_raw: Array
    density_delta_trs: Array
    absolute_density: Array
    hamiltonian: Array
    interaction_h: Array
    energies: Array
    chemical_potential_ry: float
    number_residual: float
    trs_leakage_max: float
    interaction_hermiticity_error: float


@dataclass(frozen=True)
class Xue2018StationaryResult:
    config: Xue2018StationaryConfig
    density_delta: Array
    absolute_density: Array
    interaction_h: Array
    hamiltonian: Array
    energies: Array
    chemical_potential_ry: float
    reference_relative_energy_ry_ab2: float
    projected_solver: StationarySolveResult
    projected_residual_rms: float
    projected_residual_max: float
    full_residual_rms: float
    full_residual_max: float
    trs_error: float
    trs_map_leakage_max: float
    number_residual: float
    converged: bool
    exit_reason: str


def _weighted_field_rms(field: Array, weights: Array) -> float:
    matrices = np.asarray(field, dtype=np.complex128)
    k_weights = np.asarray(weights, dtype=np.float64)
    numerator = float(
        np.einsum("abk,abk,k->", matrices.conj(), matrices, k_weights, optimize=True).real
    )
    denominator = float(np.sum(k_weights)) * matrices.shape[0] * matrices.shape[1]
    return float(np.sqrt(max(0.0, numerator / denominator)))


def project_xue2018_neutral_density_delta(
    state: Xue2018HFState,
    density_delta: Array,
) -> Array:
    """Project a trial ``D`` onto the fixed-total-particle-number hyperplane."""

    density = np.asarray(density_delta, dtype=np.complex128)
    if density.shape != state.reference_density.shape:
        raise ValueError("density_delta shape does not match the Xue state")
    weights = np.asarray(state.mesh.weights_ab2, dtype=np.float64)
    total = np.einsum("aak,k->", density, weights, optimize=True)
    correction = total / (density.shape[0] * np.sum(weights))
    return density - correction * np.eye(density.shape[0], dtype=np.complex128)[:, :, None]


def xue2018_stationary_density_map(
    state: Xue2018HFState,
    density_delta: Array,
    *,
    thermal_energy_ry: float,
) -> Xue2018DensityMap:
    """Apply the unrestricted Xue density map and its complete-TRS projection."""

    density = np.asarray(density_delta, dtype=np.complex128)
    if density.shape != state.reference_density.shape:
        raise ValueError("density_delta shape does not match the Xue state")
    interaction_raw = xue2018_interaction_action(state, density)
    interaction_error = float(
        np.max(
            np.abs(interaction_raw - np.swapaxes(interaction_raw.conj(), 0, 1)),
            initial=0.0,
        )
    )
    interaction_scale = max(1.0, float(np.max(np.abs(interaction_raw), initial=0.0)))
    if interaction_error > 1.0e-10 * interaction_scale:
        raise ValueError(
            "interaction action has non-roundoff Hermiticity error: "
            f"{interaction_error:.3e} at scale {interaction_scale:.3e}"
        )
    interaction_h = 0.5 * (
        interaction_raw + np.swapaxes(interaction_raw.conj(), 0, 1)
    )
    hamiltonian = state.h0 + interaction_h
    if thermal_energy_ry > 0.0:
        update = fermi_density_from_hamiltonian(
            hamiltonian,
            state.mesh.weights_ab2,
            thermal_energy=float(thermal_energy_ry),
            ensemble=FixedOccupation(occupation_per_k=2.0),
        )
        absolute_density = np.asarray(update.density, dtype=np.complex128)
        raw_delta = absolute_density - state.reference_density
        number_residual = float(update.observables["number_residual"])
    elif thermal_energy_ry == 0.0:
        update = xue2018_global_neutral_projector(
            hamiltonian,
            reference_density=state.reference_density,
        )
        raw_delta = np.asarray(update.density, dtype=np.complex128)
        absolute_density = raw_delta + state.reference_density
        number_residual = float(
            np.einsum(
                "aak,k->",
                absolute_density,
                state.mesh.weights_ab2,
                optimize=True,
            ).real
            - 2.0 * np.sum(state.mesh.weights_ab2)
        )
    else:
        raise ValueError("thermal_energy_ry must be nonnegative")
    trs_delta = project_xue2018_full_trs(raw_delta, state.mesh)
    leakage = float(np.max(np.abs(raw_delta - trs_delta), initial=0.0))
    return Xue2018DensityMap(
        density_delta_raw=raw_delta,
        density_delta_trs=trs_delta,
        absolute_density=absolute_density,
        hamiltonian=hamiltonian,
        interaction_h=interaction_h,
        energies=np.asarray(update.energies, dtype=np.float64),
        chemical_potential_ry=float(update.mu),
        number_residual=number_residual,
        trs_leakage_max=leakage,
        interaction_hermiticity_error=interaction_error,
    )


def xue2018_state_at_parameters(
    template: Xue2018HFState,
    *,
    eg_ry: float,
    hybridization_ab_ry: float,
) -> Xue2018HFState:
    """Reuse one regulator/kernel while replacing only source Hamiltonian parameters."""

    params = xue2018_standard_parameters(
        eg_ry=float(eg_ry),
        hybridization_ab_ry=float(hybridization_ab_ry),
    )
    h0 = build_zeng2022_folded_h0(template.mesh.points_ab_inv, template.basis, params)
    return replace(
        template,
        params=params,
        h0=h0,
        hamiltonian=h0.copy(),
        density=np.zeros_like(template.density),
        energies=np.zeros_like(template.energies),
        diagnostics={},
    )


def xue2018_stationary_residual_vector(
    state: Xue2018HFState,
    vector: Array,
    *,
    thermal_energy_ry: float,
) -> Array:
    """Return the packed complete-TRS stationary residual for one Xue state."""

    dimension = state.h0.shape[0]
    density = unpack_hermitian_matrix_field(vector, dimension=dimension, nk=state.nk)
    constrained_density = project_xue2018_neutral_density_delta(
        state,
        project_xue2018_full_trs(density, state.mesh),
    )
    mapped = xue2018_stationary_density_map(
        state,
        constrained_density,
        thermal_energy_ry=float(thermal_energy_ry),
    )
    return pack_hermitian_matrix_field(density - mapped.density_delta_trs)


def solve_xue2018_stationary_root(
    state: Xue2018HFState,
    initial_density_delta: Array,
    *,
    config: Xue2018StationaryConfig = Xue2018StationaryConfig(),
) -> Xue2018StationaryResult:
    """Solve the complete-TRS stationary equation independently of ODA energy descent."""

    initial = project_xue2018_neutral_density_delta(
        state,
        project_xue2018_full_trs(initial_density_delta, state.mesh),
    )
    dimension, _, nk = initial.shape
    initial_vector = pack_hermitian_matrix_field(initial)

    def residual_builder(vector: Array) -> Array:
        return xue2018_stationary_residual_vector(
            state,
            vector,
            thermal_energy_ry=config.thermal_energy_ry,
        )

    root = solve_stationary_residual(residual_builder, initial_vector, config=config.root)
    density = unpack_hermitian_matrix_field(root.vector, dimension=dimension, nk=nk)
    constrained_density = project_xue2018_neutral_density_delta(
        state,
        project_xue2018_full_trs(density, state.mesh),
    )
    mapped = xue2018_stationary_density_map(
        state,
        constrained_density,
        thermal_energy_ry=config.thermal_energy_ry,
    )
    projected_field = density - mapped.density_delta_trs
    full_field = density - mapped.density_delta_raw
    projected_rms = _weighted_field_rms(projected_field, state.mesh.weights_ab2)
    full_rms = _weighted_field_rms(full_field, state.mesh.weights_ab2)
    projected_max = float(np.max(np.abs(projected_field), initial=0.0))
    full_max = float(np.max(np.abs(full_field), initial=0.0))
    trs_error = xue2018_trs_residual(density, state.mesh)
    converged = (
        root.converged
        and full_rms <= config.full_residual_rms_tolerance
        and full_max <= config.full_residual_max_tolerance
        and trs_error <= config.trs_tolerance
        and abs(mapped.number_residual) <= config.full_residual_max_tolerance
    )
    if converged:
        exit_reason = "full_stationary_tolerance"
    elif root.converged and (
        full_rms > config.full_residual_rms_tolerance
        or full_max > config.full_residual_max_tolerance
    ):
        exit_reason = "projected_root_has_full_space_residual"
    elif root.converged and trs_error > config.trs_tolerance:
        exit_reason = "projected_root_has_trs_residual"
    else:
        exit_reason = root.exit_reason
    energy = xue2018_reference_relative_energy_density(
        mapped.interaction_h,
        state.h0,
        density,
        mesh=state.mesh,
    )
    return Xue2018StationaryResult(
        config=config,
        density_delta=density,
        absolute_density=density + state.reference_density,
        interaction_h=mapped.interaction_h,
        hamiltonian=mapped.hamiltonian,
        energies=mapped.energies,
        chemical_potential_ry=mapped.chemical_potential_ry,
        reference_relative_energy_ry_ab2=energy,
        projected_solver=root,
        projected_residual_rms=projected_rms,
        projected_residual_max=projected_max,
        full_residual_rms=full_rms,
        full_residual_max=full_max,
        trs_error=trs_error,
        trs_map_leakage_max=mapped.trs_leakage_max,
        number_residual=mapped.number_residual,
        converged=converged,
        exit_reason=exit_reason,
    )


def run_xue2018_temperature_homotopy(
    state: Xue2018HFState,
    initial_density_delta: Array,
    temperatures_ry: Iterable[float],
    *,
    root_config: StationarySolveConfig = StationarySolveConfig(),
) -> tuple[Xue2018StationaryResult, ...]:
    """Solve a declared finite-temperature sequence, feeding each accepted root forward."""

    density = np.asarray(initial_density_delta, dtype=np.complex128)
    results: list[Xue2018StationaryResult] = []
    for thermal_energy in temperatures_ry:
        result = solve_xue2018_stationary_root(
            state,
            density,
            config=Xue2018StationaryConfig(
                thermal_energy_ry=float(thermal_energy),
                root=root_config,
            ),
        )
        results.append(result)
        if not result.converged:
            break
        density = result.density_delta
    return tuple(results)


__all__ = [
    "Xue2018DensityMap",
    "Xue2018StationaryConfig",
    "Xue2018StationaryResult",
    "project_xue2018_neutral_density_delta",
    "run_xue2018_temperature_homotopy",
    "solve_xue2018_stationary_root",
    "xue2018_state_at_parameters",
    "xue2018_stationary_residual_vector",
    "xue2018_stationary_density_map",
]
