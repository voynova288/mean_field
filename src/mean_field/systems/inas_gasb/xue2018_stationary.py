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

_PAULI_0 = np.eye(2, dtype=np.complex128)
_PAULI_X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_PAULI_Y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_PAULI_Z = np.diag([1.0, -1.0]).astype(np.complex128)
_XUE2018_FOUR_TERM_OPERATORS = np.asarray(
    [
        np.kron(_PAULI_0, _PAULI_Z),
        np.kron(_PAULI_Z, _PAULI_X),
        np.kron(_PAULI_0, _PAULI_Y),
        np.kron(_PAULI_Y, _PAULI_Y),
    ]
)
_XUE2018_EXCHANGE_FORM_OPERATORS = np.asarray(
    [
        np.kron(_PAULI_0, _PAULI_Z),
        np.kron(_PAULI_Y, _PAULI_Y),
    ]
)


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
class Xue2018SelfEnergyReleaseMap:
    trial_self_energy_trs: Array
    interaction_self_energy_raw: Array
    interaction_self_energy_trs: Array
    interaction_self_energy_four_term: Array
    released_self_energy: Array
    density_delta: Array
    absolute_density: Array
    hamiltonian: Array
    energies: Array
    chemical_potential_ry: float
    number_residual: float
    full_trs_leakage_max: float


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


def _hermitize_xue2018_interaction(interaction_raw: Array) -> tuple[Array, float]:
    raw = np.asarray(interaction_raw, dtype=np.complex128)
    error = float(
        np.max(
            np.abs(raw - np.swapaxes(raw.conj(), 0, 1)),
            initial=0.0,
        )
    )
    scale = max(1.0, float(np.max(np.abs(raw), initial=0.0)))
    if error > 1.0e-10 * scale:
        raise ValueError(
            "interaction action has non-roundoff Hermiticity error: "
            f"{error:.3e} at scale {scale:.3e}"
        )
    return 0.5 * (raw + np.swapaxes(raw.conj(), 0, 1)), error


def _project_xue2018_pauli_support(self_energy: Array, operators: Array) -> Array:
    field = np.asarray(self_energy, dtype=np.complex128)
    if field.ndim != 3 or field.shape[:2] != (4, 4):
        raise ValueError("self_energy must have shape (4, 4, nk)")
    hermiticity_error = float(
        np.max(np.abs(field - np.swapaxes(field.conj(), 0, 1)), initial=0.0)
    )
    scale = max(1.0, float(np.max(np.abs(field), initial=0.0)))
    if hermiticity_error > 1.0e-10 * scale:
        raise ValueError("self_energy must be Hermitian")
    coefficients = np.asarray(
        [
            np.einsum("ab,bak->k", operator, field, optimize=True).real / 4.0
            for operator in operators
        ]
    )
    return np.einsum("iab,ik->abk", operators, coefficients, optimize=True)


def project_xue2018_four_term_self_energy(self_energy: Array) -> Array:
    """Project onto a diagnostic four-channel self-energy support.

    The channels mirror the four terms displayed in the paper's schematic
    TRS-nematic Hamiltonian, but the pointwise field projection is stronger
    than the source's Gamma-point condition and does not enforce the displayed
    radial/angular coefficient forms. It is therefore a constrained diagnostic,
    not the literal paper equation.
    """

    return _project_xue2018_pauli_support(self_energy, _XUE2018_FOUR_TERM_OPERATORS)


def project_xue2018_exchange_form_self_energy(self_energy: Array) -> Array:
    """Project interaction self-energy onto ``s0 tau_z`` and ``s_y tau_y``.

    This leaves the paper's displayed ``A k_x s_z tau_x - A k_y s0 tau_y``
    terms entirely in the bare Hamiltonian while retaining an interaction-
    renormalized band splitting and antisymmetric opposite-spin exchange.
    It is a source-shaped constrained diagnostic, not an unrestricted Eq. (5)
    solution and not a claim that the paper imposed this projection numerically.
    """

    return _project_xue2018_pauli_support(
        self_energy,
        _XUE2018_EXCHANGE_FORM_OPERATORS,
    )


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
    interaction_h, interaction_error = _hermitize_xue2018_interaction(interaction_raw)
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


def xue2018_self_energy_release_map(
    state: Xue2018HFState,
    self_energy: Array,
    *,
    constraint_weight: float,
    thermal_energy_ry: float,
) -> Xue2018SelfEnergyReleaseMap:
    """Release the diagnostic four-channel self-energy constraint continuously.

    With ``constraint_weight=1`` this is the declared diagnostic self-energy
    map ``P4 Sigma_int[D(H0+Sigma)]``. With weight zero it is the complete-TRS
    self-energy map. Intermediate values define a diagnostic homotopy; they
    are not an additional physical interaction parameter.
    """

    weight = float(constraint_weight)
    temperature = float(thermal_energy_ry)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("constraint_weight must lie in [0, 1]")
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("thermal_energy_ry must be finite and nonnegative")
    trial = np.asarray(self_energy, dtype=np.complex128)
    if trial.shape != state.h0.shape:
        raise ValueError("self_energy shape does not match the Xue state")
    trial_trs = project_xue2018_full_trs(trial, state.mesh)
    hamiltonian = state.h0 + trial_trs
    if temperature > 0.0:
        update = fermi_density_from_hamiltonian(
            hamiltonian,
            state.mesh.weights_ab2,
            thermal_energy=temperature,
            ensemble=FixedOccupation(occupation_per_k=2.0),
        )
        absolute_density = np.asarray(update.density, dtype=np.complex128)
        density_delta = absolute_density - state.reference_density
        number_residual = float(update.observables["number_residual"])
    else:
        update = xue2018_global_neutral_projector(
            hamiltonian,
            reference_density=state.reference_density,
        )
        density_delta = np.asarray(update.density, dtype=np.complex128)
        absolute_density = density_delta + state.reference_density
        number_residual = float(
            np.einsum(
                "aak,k->",
                absolute_density,
                state.mesh.weights_ab2,
                optimize=True,
            ).real
            - 2.0 * np.sum(state.mesh.weights_ab2)
        )
    interaction_raw = xue2018_interaction_action(state, density_delta)
    interaction_h, _error = _hermitize_xue2018_interaction(interaction_raw)
    interaction_trs = project_xue2018_full_trs(interaction_h, state.mesh)
    interaction_four = project_xue2018_four_term_self_energy(interaction_trs)
    released = (1.0 - weight) * interaction_trs + weight * interaction_four
    return Xue2018SelfEnergyReleaseMap(
        trial_self_energy_trs=trial_trs,
        interaction_self_energy_raw=interaction_h,
        interaction_self_energy_trs=interaction_trs,
        interaction_self_energy_four_term=interaction_four,
        released_self_energy=released,
        density_delta=density_delta,
        absolute_density=absolute_density,
        hamiltonian=hamiltonian,
        energies=np.asarray(update.energies, dtype=np.float64),
        chemical_potential_ry=float(update.mu),
        number_residual=number_residual,
        full_trs_leakage_max=float(
            np.max(np.abs(interaction_h - interaction_trs), initial=0.0)
        ),
    )


def xue2018_self_energy_release_residual_vector(
    state: Xue2018HFState,
    vector: Array,
    *,
    constraint_weight: float,
    thermal_energy_ry: float,
) -> Array:
    """Return the packed residual of the four-channel-to-full release homotopy."""

    self_energy = unpack_hermitian_matrix_field(
        vector,
        dimension=state.h0.shape[0],
        nk=state.nk,
    )
    mapped = xue2018_self_energy_release_map(
        state,
        self_energy,
        constraint_weight=constraint_weight,
        thermal_energy_ry=thermal_energy_ry,
    )
    return pack_hermitian_matrix_field(self_energy - mapped.released_self_energy)


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


def xue2018_unrestricted_density_map_vector(
    state: Xue2018HFState,
    vector: Array,
    *,
    thermal_energy_ry: float,
) -> Array:
    """Return the unrestricted fixed-number density map in packed coordinates."""

    density = unpack_hermitian_matrix_field(vector, dimension=state.h0.shape[0], nk=state.nk)
    density = project_xue2018_neutral_density_delta(state, density)
    mapped = xue2018_stationary_density_map(
        state,
        density,
        thermal_energy_ry=float(thermal_energy_ry),
    )
    return pack_hermitian_matrix_field(mapped.density_delta_raw)


def xue2018_trs_tangent_projector_vector(state: Xue2018HFState, vector: Array) -> Array:
    """Project a packed tangent onto the fixed-number complete-TRS sector."""

    field = unpack_hermitian_matrix_field(vector, dimension=state.h0.shape[0], nk=state.nk)
    projected = project_xue2018_neutral_density_delta(
        state,
        project_xue2018_full_trs(field, state.mesh),
    )
    return pack_hermitian_matrix_field(projected)


def xue2018_trsb_tangent_projector_vector(state: Xue2018HFState, vector: Array) -> Array:
    """Project a packed tangent onto the TR-odd complement at fixed number."""

    field = unpack_hermitian_matrix_field(vector, dimension=state.h0.shape[0], nk=state.nk)
    trs = project_xue2018_full_trs(field, state.mesh)
    projected = project_xue2018_neutral_density_delta(state, field - trs)
    return pack_hermitian_matrix_field(projected)


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
    "Xue2018SelfEnergyReleaseMap",
    "Xue2018StationaryConfig",
    "Xue2018StationaryResult",
    "project_xue2018_exchange_form_self_energy",
    "project_xue2018_four_term_self_energy",
    "project_xue2018_neutral_density_delta",
    "run_xue2018_temperature_homotopy",
    "solve_xue2018_stationary_root",
    "xue2018_self_energy_release_map",
    "xue2018_self_energy_release_residual_vector",
    "xue2018_state_at_parameters",
    "xue2018_trs_tangent_projector_vector",
    "xue2018_trsb_tangent_projector_vector",
    "xue2018_unrestricted_density_map_vector",
    "xue2018_stationary_residual_vector",
    "xue2018_stationary_density_map",
]
