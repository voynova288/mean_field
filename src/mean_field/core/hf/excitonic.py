"""System-independent reference-subtracted matrix Hartree--Fock API.

The API uses ket-oriented one-particle density matrices

``P[a, b, k] = <c^dagger[k, b] c[k, a]>``

and iterates ``D = P - P_ref``.  A linear self-energy functional must obey
``Sigma[a D1 + b D2] = a Sigma[D1] + b Sigma[D2]`` and be self-adjoint in
the weighted trace pairing.  Under that contract the interaction energy is
``1/2 sum_k w_k Tr(Sigma[D]_k D_k)``.

Physical systems remain responsible for constructing Hamiltonians, declaring
the electron/hole subspaces, binding electrostatic or screening conventions,
and attesting the source of any fixed chemical potential.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import operator
from typing import Literal, Protocol

import numpy as np
from scipy.optimize import brentq

from .engine import (
    DensityUpdateResult,
    HartreeFockRun,
    HartreeFockStateProtocol,
    HartreeFockStepResult,
    run_hartree_fock_iterations,
)

Array = np.ndarray
SearchMode = Literal["normal_reference", "seeded_ei"]
AbsoluteDensityBuilder = Callable[[Array], DensityUpdateResult]
InteractionComponentsBuilder = Callable[[Array], Mapping[str, Array]]
DensitySymmetrizer = Callable[[Array], Array]


class SelfEnergyBuilderProtocol(Protocol):
    """Callable linear self-energy action ``D -> Sigma[D]``."""

    def __call__(self, density_delta: Array) -> Array: ...


@dataclass(frozen=True)
class FixedChemicalPotential:
    """Grand-canonical ensemble in the Hamiltonian's energy units."""

    chemical_potential: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.chemical_potential):
            raise ValueError("chemical_potential must be finite")


@dataclass(frozen=True)
class FixedOccupation:
    """Weighted mean occupation per k point."""

    occupation_per_k: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.occupation_per_k):
            raise ValueError("occupation_per_k must be finite")


FermiEnsemble = FixedChemicalPotential | FixedOccupation


@dataclass(frozen=True)
class ThermodynamicDensityBuilder:
    """Typed absolute-density map with an attested thermal/ensemble contract."""

    builder: AbsoluteDensityBuilder
    thermal_energy: float
    constraint_label: str
    fixed_chemical_potential: float | None = None

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise TypeError("density builder must be callable")
        if not np.isfinite(self.thermal_energy) or self.thermal_energy <= 0.0:
            raise ValueError("density-builder thermal_energy must be finite and positive")
        if not self.constraint_label:
            raise ValueError("density-builder constraint_label must be nonempty")
        if self.fixed_chemical_potential is not None and not np.isfinite(
            self.fixed_chemical_potential
        ):
            raise ValueError("fixed_chemical_potential must be finite when supplied")

    def __call__(self, hamiltonian: Array) -> DensityUpdateResult:
        update = self.builder(hamiltonian)
        if self.fixed_chemical_potential is not None and not np.isclose(
            float(update.mu),
            float(self.fixed_chemical_potential),
            rtol=0.0,
            atol=1.0e-13,
        ):
            raise ValueError(
                "fixed-mu density builder changed its immutable chemical potential"
            )
        return update


@dataclass(frozen=True)
class ElectronHoleSubspaces:
    """Explicit electron and valence-electron indices in one ordinary-electron basis.

    Hole number is evaluated as the absence of ordinary electrons in
    ``hole_indices``.  The two sets need not have equal rank, but they must be
    nonempty, unique, nonnegative, and disjoint.
    """

    electron_indices: tuple[int, ...]
    hole_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        def exact_index(index: object) -> int:
            if isinstance(index, (bool, np.bool_)):
                raise TypeError("electron/hole indices must be integers, not booleans")
            try:
                return int(operator.index(index))
            except TypeError as error:
                raise TypeError("electron/hole indices must be exact integers") from error

        electron = tuple(exact_index(index) for index in self.electron_indices)
        hole = tuple(exact_index(index) for index in self.hole_indices)
        if not electron or not hole:
            raise ValueError("electron and hole subspaces must both be nonempty")
        if len(set(electron)) != len(electron) or len(set(hole)) != len(hole):
            raise ValueError("electron and hole indices must be unique")
        if min(electron + hole) < 0:
            raise ValueError("electron and hole indices must be nonnegative")
        if set(electron).intersection(hole):
            raise ValueError("electron and hole subspaces must be disjoint")
        object.__setattr__(self, "electron_indices", electron)
        object.__setattr__(self, "hole_indices", hole)

    def validate_dimension(self, dimension: int) -> None:
        if max(self.electron_indices + self.hole_indices) >= int(dimension):
            raise ValueError("electron/hole index lies outside the matrix dimension")

    def coherence_block(self, field: Array) -> Array:
        """Return the explicit electron-to-hole block with shape ``(ne, nh, nk)``."""

        matrices = _as_matrix_field(field, name="field")
        self.validate_dimension(matrices.shape[0])
        return matrices[np.asarray(self.electron_indices)[:, None], np.asarray(self.hole_indices)[None, :], :]

    def coherence_singular_values(self, field: Array) -> Array:
        """Return singular values of the E--H block at every k point."""

        block = self.coherence_block(field)
        return np.asarray(
            [np.linalg.svd(block[:, :, ik], compute_uv=False) for ik in range(block.shape[2])]
        )

    def carrier_densities(self, density: Array, k_weights: Array) -> tuple[float, float]:
        """Return electron and valence-hole densities in the declared subspaces."""

        projector = _validate_density_matrix(density, name="density")
        weights = _validate_weights(k_weights, projector.shape[2])
        self.validate_dimension(projector.shape[0])
        electron_diagonal = np.diagonal(
            projector[np.ix_(self.electron_indices, self.electron_indices, range(projector.shape[2]))],
            axis1=0,
            axis2=1,
        )
        hole_diagonal = np.diagonal(
            projector[np.ix_(self.hole_indices, self.hole_indices, range(projector.shape[2]))],
            axis1=0,
            axis2=1,
        )
        electron_density = float(np.einsum("k,ka->", weights, electron_diagonal.real))
        hole_density = float(
            len(self.hole_indices) * np.sum(weights)
            - np.einsum("k,ka->", weights, hole_diagonal.real)
        )
        return electron_density, hole_density


@dataclass(frozen=True)
class LinearSelfEnergyResiduals:
    zero_error: float
    additivity_error: float
    homogeneity_error: float
    self_adjoint_error: float
    map_scale: float
    pairing_scale: float

    @property
    def maximum_error(self) -> float:
        return max(
            self.zero_error,
            self.additivity_error,
            self.homogeneity_error,
            self.self_adjoint_error,
        )

    @property
    def maximum_relative_error(self) -> float:
        return max(
            self.zero_error / self.map_scale,
            self.additivity_error / self.map_scale,
            self.homogeneity_error / self.map_scale,
            self.self_adjoint_error / self.pairing_scale,
        )


_LINEAR_SELF_ENERGY_CERTIFICATE_TOKEN = object()


@dataclass(frozen=True, init=False)
class LinearSelfEnergyCertificate:
    """Executable two-probe certificate bound to one exact callable object."""

    label: str
    residuals: LinearSelfEnergyResiduals
    absolute_tolerance: float
    relative_tolerance: float
    operator_fingerprint: str | None
    _certified_builder: SelfEnergyBuilderProtocol = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        label: str,
        residuals: LinearSelfEnergyResiduals,
        absolute_tolerance: float,
        relative_tolerance: float,
        operator_fingerprint: str | None,
        certified_builder: SelfEnergyBuilderProtocol,
        _token: object,
    ) -> None:
        if _token is not _LINEAR_SELF_ENERGY_CERTIFICATE_TOKEN:
            raise TypeError(
                "LinearSelfEnergyCertificate can only be created by "
                "certify_linear_self_energy()"
            )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "absolute_tolerance", float(absolute_tolerance))
        object.__setattr__(self, "relative_tolerance", float(relative_tolerance))
        object.__setattr__(self, "operator_fingerprint", operator_fingerprint)
        object.__setattr__(self, "_certified_builder", certified_builder)
        self._validate()

    def _validate(self) -> None:
        if not self.label:
            raise ValueError("linear self-energy certificate label must be nonempty")
        if self.absolute_tolerance < 0.0 or self.relative_tolerance < 0.0:
            raise ValueError("certificate tolerances must be nonnegative")
        if not np.isfinite(self.absolute_tolerance) or not np.isfinite(
            self.relative_tolerance
        ):
            raise ValueError("certificate tolerances must be finite")
        if self.operator_fingerprint is not None and not self.operator_fingerprint:
            raise ValueError("operator_fingerprint must be nonempty when supplied")
        if not callable(self._certified_builder):
            raise TypeError("certificate builder must be callable")

    @property
    def accepted(self) -> bool:
        map_limit = self.absolute_tolerance + self.relative_tolerance * self.residuals.map_scale
        pairing_limit = (
            self.absolute_tolerance
            + self.relative_tolerance * self.residuals.pairing_scale
        )
        return bool(
            self.residuals.zero_error <= map_limit
            and self.residuals.additivity_error <= map_limit
            and self.residuals.homogeneity_error <= map_limit
            and self.residuals.self_adjoint_error <= pairing_limit
        )

    def is_bound_to(self, builder: SelfEnergyBuilderProtocol) -> bool:
        return self._certified_builder is builder


@dataclass(frozen=True)
class LinearSelfEnergyFunctional:
    """Certified linear, weighted-self-adjoint matrix self-energy functional."""

    self_energy_builder: SelfEnergyBuilderProtocol
    certificate: LinearSelfEnergyCertificate
    component_builder: InteractionComponentsBuilder | None = None
    label: str = "interaction"

    def __post_init__(self) -> None:
        if not callable(self.self_energy_builder):
            raise TypeError("self_energy_builder must be callable")
        if self.component_builder is not None and not callable(self.component_builder):
            raise TypeError("component_builder must be callable")
        if not isinstance(self.certificate, LinearSelfEnergyCertificate):
            raise TypeError("linear self-energy requires an executable certificate")
        if not self.certificate.is_bound_to(self.self_energy_builder):
            raise ValueError("linear self-energy certificate belongs to another builder")
        if not self.certificate.accepted:
            raise ValueError(
                "linear self-energy certificate failed: "
                f"maximum_relative_error={self.certificate.residuals.maximum_relative_error:.3e}"
            )
        if not self.label:
            raise ValueError("self-energy label must be nonempty")

    @classmethod
    def from_probes(
        cls,
        self_energy_builder: SelfEnergyBuilderProtocol,
        first_probe: Array,
        second_probe: Array,
        k_weights: Array,
        *,
        validation_label: str,
        component_builder: InteractionComponentsBuilder | None = None,
        label: str = "interaction",
        absolute_tolerance: float = 1.0e-12,
        relative_tolerance: float = 1.0e-10,
        operator_fingerprint: str | None = None,
    ) -> "LinearSelfEnergyFunctional":
        certificate = certify_linear_self_energy(
            self_energy_builder,
            first_probe,
            second_probe,
            k_weights,
            validation_label=validation_label,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            operator_fingerprint=operator_fingerprint,
        )
        return cls(
            self_energy_builder=self_energy_builder,
            certificate=certificate,
            component_builder=component_builder,
            label=label,
        )

    @property
    def validation_label(self) -> str:
        return self.certificate.label

    def self_energy(self, density_delta: Array) -> Array:
        density = _as_matrix_field(density_delta, name="density_delta")
        return _apply_self_energy(self.self_energy_builder, density)

    def components(self, density_delta: Array) -> dict[str, Array]:
        density = _as_matrix_field(density_delta, name="density_delta")
        sigma = self.self_energy(density)
        if self.component_builder is None:
            return {self.label: sigma}
        raw = self.component_builder(density)
        if not raw:
            raise ValueError("component_builder must return at least one component")
        components: dict[str, Array] = {}
        for name, values in raw.items():
            if not isinstance(name, str) or not name or name in components:
                raise ValueError("self-energy component names must be unique nonempty strings")
            component = np.asarray(values, dtype=np.complex128)
            if component.shape != density.shape or not np.all(np.isfinite(component)):
                raise ValueError(f"self-energy component {name!r} has invalid shape or values")
            _validate_hermitian(component, name=f"self-energy component {name!r}")
            components[name] = component
        component_sum = np.sum(np.stack(tuple(components.values()), axis=0), axis=0)
        closure_error = float(np.max(np.abs(component_sum - sigma)))
        component_scale = max(
            float(np.max(np.abs(sigma))),
            *(float(np.max(np.abs(component))) for component in components.values()),
        )
        if closure_error > 1.0e-12 + 1.0e-10 * component_scale:
            raise ValueError(
                "self-energy components do not close to the total action: "
                f"error={closure_error:.3e}"
            )
        return components


@dataclass(frozen=True)
class ReferenceSubtractedHFConfig:
    """Numerical contract for a reference-subtracted matrix-HF fixed point."""

    thermal_energy: float
    mixing: float = 0.2
    precision: float = 1.0e-8
    max_iter: int = 500
    search_mode: SearchMode = "normal_reference"
    grand_canonical_mu: float | None = None
    convergence_scale: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.thermal_energy) or self.thermal_energy <= 0.0:
            raise ValueError("thermal_energy must be finite and positive")
        if not 1.0e-3 <= self.mixing <= 1.0:
            raise ValueError("mixing must lie in [1e-3, 1]")
        if not np.isfinite(self.precision) or self.precision <= 0.0:
            raise ValueError("precision must be finite and positive")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be positive")
        if self.search_mode not in {"normal_reference", "seeded_ei"}:
            raise ValueError("unsupported search_mode")
        if self.grand_canonical_mu is not None and not np.isfinite(self.grand_canonical_mu):
            raise ValueError("grand_canonical_mu must be finite when supplied")
        if self.convergence_scale is not None and (
            not np.isfinite(self.convergence_scale) or self.convergence_scale <= 0.0
        ):
            raise ValueError("convergence_scale must be finite and positive when supplied")


@dataclass
class ReferenceSubtractedHFState:
    h0: Array
    density: Array
    hamiltonian: Array
    energies: Array
    mu: float
    precision: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


@dataclass(frozen=True)
class ReferenceSubtractedEnergy:
    one_body: float
    interaction_components: dict[str, float]
    internal_energy: float
    entropy_difference: float
    free_energy: float
    particle_number_change: float
    grand_potential: float | None


@dataclass(frozen=True)
class ReferenceSubtractedHFResult:
    config: ReferenceSubtractedHFConfig
    reference_density: Array
    noninteracting_density: Array
    density_delta: Array
    total_density: Array
    interaction_components: dict[str, Array]
    interaction_hamiltonian: Array
    hamiltonian: Array
    energies: Array
    chemical_potential: float
    energy: ReferenceSubtractedEnergy
    density_coherence_singular_values: Array | None
    self_energy_coherence_singular_values: Array | None
    run: HartreeFockRun
    converged: bool
    exit_reason: str


def fermi_function(energy_minus_mu: Array, thermal_energy: float) -> Array:
    """Numerically stable Fermi function in arbitrary consistent energy units."""

    if not np.isfinite(thermal_energy) or thermal_energy <= 0.0:
        raise ValueError("thermal_energy must be finite and positive")
    x = np.asarray(energy_minus_mu, dtype=float) / float(thermal_energy)
    result = np.empty_like(x)
    positive = x >= 0.0
    exp_minus = np.exp(-np.clip(x[positive], 0.0, 745.0))
    result[positive] = exp_minus / (1.0 + exp_minus)
    exp_plus = np.exp(np.clip(x[~positive], -745.0, 0.0))
    result[~positive] = 1.0 / (1.0 + exp_plus)
    return result


def fermi_density_from_hamiltonian(
    hamiltonian: Array,
    k_weights: Array,
    *,
    thermal_energy: float,
    ensemble: FermiEnsemble,
) -> DensityUpdateResult:
    """Diagonalize ``H(k)`` and construct the absolute ket-oriented density ``P``."""

    hamiltonian_field = _as_matrix_field(hamiltonian, name="hamiltonian")
    dimension, _, nk = hamiltonian_field.shape
    weights = _validate_weights(k_weights, nk)
    _validate_hermitian(hamiltonian_field, name="hamiltonian")
    if not np.isfinite(thermal_energy) or thermal_energy <= 0.0:
        raise ValueError("thermal_energy must be finite and positive")
    energies = np.empty((dimension, nk), dtype=float)
    vectors = np.empty((dimension, dimension, nk), dtype=np.complex128)
    for ik in range(nk):
        energies[:, ik], vectors[:, :, ik] = np.linalg.eigh(hamiltonian_field[:, :, ik])

    observables: dict[str, float]
    if isinstance(ensemble, FixedChemicalPotential):
        chemical_potential = float(ensemble.chemical_potential)
        occupations = fermi_function(energies - chemical_potential, thermal_energy)
        active_number = float(np.einsum("k,nk->", weights, occupations, optimize=True))
        observables = {
            "achieved_occupation_per_k": active_number / float(np.sum(weights)),
            "active_number": active_number,
            "fixed_chemical_potential": chemical_potential,
        }
    elif isinstance(ensemble, FixedOccupation):
        target_per_k = float(ensemble.occupation_per_k)
        if not 0.0 <= target_per_k <= float(dimension):
            raise ValueError("target occupation lies outside the active-space dimension")
        target = target_per_k * float(np.sum(weights))
        margin = 80.0 * float(thermal_energy)
        lower = float(np.min(energies) - margin)
        upper = float(np.max(energies) + margin)

        def number_residual(mu: float) -> float:
            occupations_at_mu = fermi_function(energies - float(mu), thermal_energy)
            return float(np.einsum("k,nk->", weights, occupations_at_mu) - target)

        if target <= 0.0:
            chemical_potential = lower
        elif target >= float(dimension) * float(np.sum(weights)):
            chemical_potential = upper
        else:
            chemical_potential = float(
                brentq(number_residual, lower, upper, xtol=1.0e-13, rtol=1.0e-14)
            )
        occupations = fermi_function(energies - chemical_potential, thermal_energy)
        achieved = float(np.einsum("k,nk->", weights, occupations, optimize=True))
        observables = {
            "target_occupation_per_k": target_per_k,
            "achieved_occupation_per_k": achieved / float(np.sum(weights)),
            "number_residual": number_residual(chemical_potential),
        }
    else:
        raise TypeError("ensemble must be FixedChemicalPotential or FixedOccupation")

    density = np.empty_like(hamiltonian_field)
    for ik in range(nk):
        density[:, :, ik] = (
            vectors[:, :, ik] * occupations[:, ik][None, :]
        ) @ vectors[:, :, ik].conj().T
    observables["hamiltonian_hermiticity_error"] = _hermiticity_error(hamiltonian_field)
    return DensityUpdateResult(
        density=density,
        energies=energies,
        mu=chemical_potential,
        observables=observables,
    )


def make_fermi_density_builder(
    k_weights: Array,
    *,
    thermal_energy: float,
    ensemble: FermiEnsemble,
) -> ThermodynamicDensityBuilder:
    """Bind a typed absolute-density builder for the selected ensemble."""

    weights = np.asarray(k_weights, dtype=float).copy()
    if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("k_weights must be a finite positive one-dimensional array")

    def build(hamiltonian: Array) -> DensityUpdateResult:
        return fermi_density_from_hamiltonian(
            hamiltonian,
            weights,
            thermal_energy=thermal_energy,
            ensemble=ensemble,
        )

    fixed_mu = (
        float(ensemble.chemical_potential)
        if isinstance(ensemble, FixedChemicalPotential)
        else None
    )
    label = "fixed_chemical_potential" if fixed_mu is not None else "fixed_occupation"
    return ThermodynamicDensityBuilder(
        builder=build,
        thermal_energy=float(thermal_energy),
        constraint_label=label,
        fixed_chemical_potential=fixed_mu,
    )


def fermionic_entropy(density: Array, k_weights: Array) -> float:
    """Return dimensionless weighted entropy ``S/k_B``.

    The weights retain their caller-defined measure.  Multiplying this value by
    ``thermal_energy = k_B T`` therefore gives the entropy contribution in the
    Hamiltonian's energy-density units.
    """

    projector = _validate_density_matrix(density, name="density")
    weights = _validate_weights(k_weights, projector.shape[2])
    entropy = 0.0
    for ik in range(projector.shape[2]):
        values = np.linalg.eigvalsh(projector[:, :, ik])
        clipped = np.clip(values, 1.0e-15, 1.0 - 1.0e-15)
        entropy -= weights[ik] * float(
            np.sum(
                clipped * np.log(clipped)
                + (1.0 - clipped) * np.log(1.0 - clipped)
            )
        )
    return entropy


def weighted_trace_product(left: Array, right: Array, k_weights: Array) -> complex:
    """Return ``sum_k w_k Tr(left_k right_k)`` in ket orientation."""

    left_field = _as_matrix_field(left, name="left")
    right_field = _as_matrix_field(right, name="right")
    if left_field.shape != right_field.shape:
        raise ValueError("left and right fields must have matching shapes")
    weights = _validate_weights(k_weights, left_field.shape[2])
    return complex(
        np.einsum("k,abk,bak->", weights, left_field, right_field, optimize=True)
    )


def relative_internal_energy(
    h0: Array,
    density_delta: Array,
    k_weights: Array,
    interaction_components: Mapping[str, Array],
) -> tuple[float, dict[str, float], float]:
    """Return one-body, component, and total reference-relative internal energy."""

    bare = _as_matrix_field(h0, name="h0")
    density = _as_matrix_field(density_delta, name="density_delta")
    if bare.shape != density.shape:
        raise ValueError("h0 and density_delta must have matching shapes")
    one_body_value = weighted_trace_product(bare, density, k_weights)
    one_body = _real_scalar(one_body_value, name="one-body energy")
    component_energies: dict[str, float] = {}
    for name, component in interaction_components.items():
        sigma = _as_matrix_field(component, name=f"interaction component {name!r}")
        if sigma.shape != density.shape:
            raise ValueError(f"interaction component {name!r} has the wrong shape")
        component_energies[name] = 0.5 * _real_scalar(
            weighted_trace_product(sigma, density, k_weights),
            name=f"interaction component {name!r} energy",
        )
    total = one_body + sum(component_energies.values())
    return one_body, component_energies, total


def linear_self_energy_residuals(
    self_energy_builder: SelfEnergyBuilderProtocol,
    first: Array,
    second: Array,
    k_weights: Array,
    *,
    scalar: float = 0.37,
) -> LinearSelfEnergyResiduals:
    """Measure linearity and weighted self-adjointness on two Hermitian probes."""

    if not callable(self_energy_builder):
        raise TypeError("self_energy_builder must be callable")
    a = _as_matrix_field(first, name="first probe")
    b = _as_matrix_field(second, name="second probe")
    if a.shape != b.shape:
        raise ValueError("self-energy probes must have matching shapes")
    _validate_hermitian(a, name="first probe")
    _validate_hermitian(b, name="second probe")
    weights = _validate_weights(k_weights, a.shape[2])
    sigma_zero = _apply_self_energy(self_energy_builder, np.zeros_like(a))
    sigma_a = _apply_self_energy(self_energy_builder, a)
    sigma_b = _apply_self_energy(self_energy_builder, b)
    sigma_sum = _apply_self_energy(self_energy_builder, a + b)
    sigma_scaled = _apply_self_energy(self_energy_builder, float(scalar) * a)
    left_pairing = weighted_trace_product(a, sigma_b, weights)
    right_pairing = weighted_trace_product(sigma_a, b, weights)
    map_scale = max(
        1.0e-30,
        *(float(np.max(np.abs(values))) for values in (
            sigma_zero,
            sigma_a,
            sigma_b,
            sigma_sum,
            sigma_scaled,
        )),
    )
    pairing_scale = max(1.0e-30, abs(left_pairing), abs(right_pairing))
    return LinearSelfEnergyResiduals(
        zero_error=float(np.max(np.abs(sigma_zero))),
        additivity_error=float(np.max(np.abs(sigma_sum - sigma_a - sigma_b))),
        homogeneity_error=float(np.max(np.abs(sigma_scaled - float(scalar) * sigma_a))),
        self_adjoint_error=float(abs(left_pairing - right_pairing.conjugate())),
        map_scale=map_scale,
        pairing_scale=float(pairing_scale),
    )


def certify_linear_self_energy(
    self_energy_builder: SelfEnergyBuilderProtocol,
    first_probe: Array,
    second_probe: Array,
    k_weights: Array,
    *,
    validation_label: str,
    absolute_tolerance: float = 1.0e-12,
    relative_tolerance: float = 1.0e-10,
    operator_fingerprint: str | None = None,
) -> LinearSelfEnergyCertificate:
    """Execute and return a two-probe linearity/self-adjointness certificate."""

    residuals = linear_self_energy_residuals(
        self_energy_builder,
        first_probe,
        second_probe,
        k_weights,
    )
    return LinearSelfEnergyCertificate(
        label=validation_label,
        residuals=residuals,
        absolute_tolerance=float(absolute_tolerance),
        relative_tolerance=float(relative_tolerance),
        operator_fingerprint=operator_fingerprint,
        certified_builder=self_energy_builder,
        _token=_LINEAR_SELF_ENERGY_CERTIFICATE_TOKEN,
    )


def run_reference_subtracted_hf(
    h0: Array,
    k_weights: Array,
    reference_density: Array,
    *,
    absolute_density_builder: ThermodynamicDensityBuilder,
    interaction: LinearSelfEnergyFunctional,
    config: ReferenceSubtractedHFConfig,
    normal_density_update: DensityUpdateResult | None = None,
    electron_hole_subspaces: ElectronHoleSubspaces | None = None,
    seed_hamiltonian: Array | None = None,
    initial_density_delta: Array | None = None,
    initial_density_is_postprocessed: bool = False,
    density_symmetrizer: DensitySymmetrizer | None = None,
    step_callback: Callable[[HartreeFockStateProtocol, HartreeFockStepResult], None] | None = None,
    final_state_callback: Callable[[HartreeFockStateProtocol, DensityUpdateResult], None] | None = None,
) -> ReferenceSubtractedHFResult:
    """Solve ``H = H0 + Sigma[P-P_ref]`` using the generic HF engine.

    ``absolute_density_builder`` must return the ordinary-electron density
    ``P(H)``.  This function alone converts it to the iterated variable
    ``D=P-P_ref``.  A supplied ``grand_canonical_mu`` is used only in the
    thermodynamic Legendre transform and is never rerooted by this function.
    """

    bare = _as_matrix_field(h0, name="h0")
    _validate_hermitian(bare, name="h0")
    weights = _validate_weights(k_weights, bare.shape[2])
    reference = _validate_density_matrix(reference_density, name="reference_density")
    if reference.shape != bare.shape:
        raise ValueError("reference_density must match h0")
    if not isinstance(absolute_density_builder, ThermodynamicDensityBuilder):
        raise TypeError(
            "absolute_density_builder must be a ThermodynamicDensityBuilder "
            "with an explicit temperature/ensemble attestation"
        )
    if not np.isclose(
        absolute_density_builder.thermal_energy,
        config.thermal_energy,
        rtol=0.0,
        atol=1.0e-14 * max(1.0, abs(config.thermal_energy)),
    ):
        raise ValueError("density-builder thermal energy disagrees with config")
    if config.grand_canonical_mu is not None:
        if absolute_density_builder.fixed_chemical_potential is None:
            raise ValueError(
                "grand_canonical_mu requires an immutable fixed-mu density builder"
            )
        if not np.isclose(
            absolute_density_builder.fixed_chemical_potential,
            config.grand_canonical_mu,
            rtol=0.0,
            atol=1.0e-13,
        ):
            raise ValueError(
                "grand_canonical_mu disagrees with the fixed-mu density builder"
            )
    if initial_density_is_postprocessed and initial_density_delta is None:
        raise ValueError("initial_density_is_postprocessed requires initial_density_delta")
    if initial_density_is_postprocessed and density_symmetrizer is None:
        raise ValueError("initial_density_is_postprocessed requires density_symmetrizer")
    if config.search_mode == "normal_reference" and (
        seed_hamiltonian is not None or initial_density_delta is not None
    ):
        raise ValueError("normal_reference mode does not accept an excitonic seed")
    if config.search_mode == "seeded_ei" and (
        (seed_hamiltonian is None) == (initial_density_delta is None)
    ):
        raise ValueError(
            "seeded_ei mode requires exactly one of seed_hamiltonian or initial_density_delta"
        )
    if electron_hole_subspaces is not None:
        electron_hole_subspaces.validate_dimension(bare.shape[0])

    normal = (
        absolute_density_builder(bare)
        if normal_density_update is None
        else normal_density_update
    )
    noninteracting = _validate_density_matrix(normal.density, name="normal density")
    if noninteracting.shape != bare.shape:
        raise ValueError("normal density must match h0")
    if np.asarray(normal.energies).shape != (bare.shape[0], bare.shape[2]):
        raise ValueError("normal energies have the wrong shape")
    if absolute_density_builder.fixed_chemical_potential is not None and not np.isclose(
        float(normal.mu),
        absolute_density_builder.fixed_chemical_potential,
        rtol=0.0,
        atol=1.0e-13,
    ):
        raise ValueError("normal density update changed the immutable chemical potential")

    if initial_density_delta is not None:
        initial_delta = _as_matrix_field(
            initial_density_delta, name="initial_density_delta"
        ).copy()
        if initial_delta.shape != bare.shape:
            raise ValueError("initial_density_delta must match h0")
    elif seed_hamiltonian is None:
        initial_delta = noninteracting - reference
    else:
        seed = _as_matrix_field(seed_hamiltonian, name="seed_hamiltonian")
        if seed.shape != bare.shape:
            raise ValueError("seed_hamiltonian must match h0")
        seeded = absolute_density_builder(bare + seed)
        initial_delta = np.asarray(seeded.density, dtype=np.complex128) - reference

    if density_symmetrizer is not None:
        postprocessed = np.asarray(density_symmetrizer(initial_delta), dtype=np.complex128)
        if postprocessed.shape != bare.shape:
            raise ValueError("density_symmetrizer changed the density shape")
        if initial_density_is_postprocessed:
            error = float(np.max(np.abs(postprocessed - initial_delta)))
            if error > 1.0e-9:
                raise ValueError(
                    "declared postprocessed initial density violates the supplied "
                    f"symmetry projector: error={error:.3e}"
                )
        else:
            initial_delta = postprocessed
    _validate_hermitian(initial_delta, name="initial density delta")

    state = ReferenceSubtractedHFState(
        h0=bare.copy(),
        density=initial_delta.copy(),
        hamiltonian=bare.copy(),
        energies=np.asarray(normal.energies, dtype=float).copy(),
        mu=float(normal.mu),
        precision=float(config.precision),
    )

    def density_builder(total_hamiltonian: Array) -> DensityUpdateResult:
        absolute = absolute_density_builder(total_hamiltonian)
        density = np.asarray(absolute.density, dtype=np.complex128) - reference
        if density_symmetrizer is not None:
            density = np.asarray(density_symmetrizer(density), dtype=np.complex128)
        _validate_hermitian(density, name="updated density delta")
        return DensityUpdateResult(
            density=density,
            energies=np.asarray(absolute.energies, dtype=float),
            mu=float(absolute.mu),
            observables=absolute.observables,
        )

    def check_hermitian(values: Array) -> None:
        _validate_hermitian(values, name="SCF matrix")

    def postprocess_density(values: Array) -> None:
        if density_symmetrizer is not None:
            values[:, :, :] = np.asarray(density_symmetrizer(values), dtype=np.complex128)
        _validate_hermitian(values, name="SCF density delta")

    reference_norm = (
        float(config.convergence_scale)
        if config.convergence_scale is not None
        else max(
            _weighted_frobenius_norm(reference, weights),
            _weighted_frobenius_norm(noninteracting, weights),
            1.0e-15,
        )
    )

    def convergence_metric(updated: Array, previous: Array) -> float:
        return _weighted_frobenius_norm(
            np.asarray(updated) - np.asarray(previous), weights
        ) / reference_norm

    def energy_functional(sigma: Array, bare_h: Array, density: Array) -> float:
        one_body, _, total = relative_internal_energy(
            bare_h, density, weights, {interaction.label: sigma}
        )
        del one_body
        return total

    run = run_hartree_fock_iterations(
        state,
        init_mode=config.search_mode,
        seed=0,
        interaction_builder=interaction.self_energy,
        density_builder=density_builder,
        energy_functional=energy_functional,
        oda_parameterizer=lambda _state, _delta: float(config.mixing),
        hamiltonian_postprocessor=check_hermitian,
        density_postprocessor=postprocess_density,
        step_callback=step_callback,
        final_state_callback=final_state_callback,
        convergence_metric=convergence_metric,
        convergence_rule="raw",
        require_final_raw_convergence=True,
        max_iter=int(config.max_iter),
        oda_stall_threshold=0.0,
    )

    density_delta = np.asarray(state.density, dtype=np.complex128).copy()
    total_density = reference + density_delta
    _validate_density_matrix(total_density, name="final density")
    interaction_hamiltonian = interaction.self_energy(density_delta)
    components = interaction.components(density_delta)
    hamiltonian = bare + interaction_hamiltonian
    closure_error = float(np.max(np.abs(hamiltonian - state.hamiltonian)))
    if closure_error > 1.0e-9 * max(1.0, float(np.max(np.abs(hamiltonian)))):
        raise ValueError(f"final Hamiltonian closure failed: error={closure_error:.3e}")

    one_body, component_energies, internal_energy = relative_internal_energy(
        bare, density_delta, weights, components
    )
    entropy_difference = (
        fermionic_entropy(total_density, weights)
        - fermionic_entropy(reference, weights)
    )
    free_energy = internal_energy - config.thermal_energy * entropy_difference
    number_change = _real_scalar(
        weighted_trace_product(np.repeat(np.eye(bare.shape[0])[:, :, None], bare.shape[2], axis=2), density_delta, weights),
        name="particle-number change",
    )
    grand_potential = (
        None
        if config.grand_canonical_mu is None
        else free_energy - float(config.grand_canonical_mu) * number_change
    )
    converged = bool(run.converged)
    exit_reason = run.exit_reason
    density_singular = (
        None
        if electron_hole_subspaces is None
        else electron_hole_subspaces.coherence_singular_values(density_delta)
    )
    self_energy_singular = (
        None
        if electron_hole_subspaces is None
        else electron_hole_subspaces.coherence_singular_values(interaction_hamiltonian)
    )
    state.diagnostics.update(
        {
            "one_body_internal_energy": one_body,
            "total_internal_energy": internal_energy,
            "entropy_difference": entropy_difference,
            "free_energy_difference": free_energy,
            "particle_number_change": number_change,
            "reference_occupation_per_k": float(
                np.einsum("k,aak->", weights, reference).real / np.sum(weights)
            ),
            "final_occupation_per_k": float(
                np.einsum("k,aak->", weights, total_density).real / np.sum(weights)
            ),
        }
    )
    for name, value in component_energies.items():
        state.diagnostics[f"{name}_internal_energy"] = value
    if grand_potential is not None:
        state.diagnostics["grand_potential_difference"] = grand_potential
        state.diagnostics["grand_canonical_mu"] = float(config.grand_canonical_mu)

    return ReferenceSubtractedHFResult(
        config=config,
        reference_density=reference.copy(),
        noninteracting_density=noninteracting.copy(),
        density_delta=density_delta,
        total_density=total_density,
        interaction_components={name: values.copy() for name, values in components.items()},
        interaction_hamiltonian=interaction_hamiltonian,
        hamiltonian=hamiltonian,
        energies=np.asarray(state.energies, dtype=float).copy(),
        chemical_potential=float(state.mu),
        energy=ReferenceSubtractedEnergy(
            one_body=one_body,
            interaction_components=component_energies,
            internal_energy=internal_energy,
            entropy_difference=entropy_difference,
            free_energy=free_energy,
            particle_number_change=number_change,
            grand_potential=grand_potential,
        ),
        density_coherence_singular_values=density_singular,
        self_energy_coherence_singular_values=self_energy_singular,
        run=run,
        converged=converged,
        exit_reason=exit_reason,
    )


def _apply_self_energy(
    self_energy_builder: SelfEnergyBuilderProtocol,
    density_delta: Array,
) -> Array:
    density = _as_matrix_field(density_delta, name="density_delta")
    sigma = np.asarray(self_energy_builder(density), dtype=np.complex128)
    if sigma.shape != density.shape or not np.all(np.isfinite(sigma)):
        raise ValueError("self-energy must be finite and match density_delta shape")
    _validate_hermitian(sigma, name="self-energy")
    return sigma


def _as_matrix_field(values: Array, *, name: str) -> Array:
    field = np.asarray(values, dtype=np.complex128)
    if field.ndim != 3 or field.shape[0] != field.shape[1]:
        raise ValueError(f"{name} must have shape (n, n, nk)")
    if not np.all(np.isfinite(field)):
        raise ValueError(f"{name} must be finite")
    return field


def _validate_weights(k_weights: Array, nk: int) -> Array:
    weights = np.asarray(k_weights, dtype=float)
    if weights.shape != (int(nk),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("k_weights must be finite, positive, and have shape (nk,)")
    return weights


def _hermiticity_error(values: Array) -> float:
    return float(np.max(np.abs(values - np.swapaxes(values.conj(), 0, 1))))


def _validate_hermitian(values: Array, *, name: str) -> None:
    error = _hermiticity_error(values)
    if error > 1.0e-9:
        raise ValueError(f"{name} is not Hermitian: error={error:.3e}")


def _validate_density_matrix(values: Array, *, name: str) -> Array:
    density = _as_matrix_field(values, name=name)
    _validate_hermitian(density, name=name)
    eigenvalues = np.concatenate(
        [np.linalg.eigvalsh(density[:, :, ik]) for ik in range(density.shape[2])]
    )
    if float(np.min(eigenvalues)) < -1.0e-8 or float(np.max(eigenvalues)) > 1.0 + 1.0e-8:
        raise ValueError(f"{name} eigenvalues lie outside [0, 1]")
    return density


def _weighted_frobenius_norm(values: Array, k_weights: Array) -> float:
    field = _as_matrix_field(values, name="matrix field")
    weights = _validate_weights(k_weights, field.shape[2])
    norm_squared = np.einsum(
        "k,abk,abk->", weights, field.conj(), field, optimize=True
    )
    return float(np.sqrt(max(_real_scalar(norm_squared, name="weighted norm squared"), 0.0)))


def _real_scalar(value: complex, *, name: str) -> float:
    scalar = complex(value)
    if abs(scalar.imag) > 1.0e-8 * max(1.0, abs(scalar.real)):
        raise ValueError(f"{name} is not real: {scalar}")
    return float(scalar.real)


__all__ = [
    "AbsoluteDensityBuilder",
    "ElectronHoleSubspaces",
    "FermiEnsemble",
    "FixedChemicalPotential",
    "FixedOccupation",
    "LinearSelfEnergyCertificate",
    "LinearSelfEnergyFunctional",
    "LinearSelfEnergyResiduals",
    "ReferenceSubtractedEnergy",
    "ReferenceSubtractedHFConfig",
    "ReferenceSubtractedHFResult",
    "ReferenceSubtractedHFState",
    "SelfEnergyBuilderProtocol",
    "ThermodynamicDensityBuilder",
    "certify_linear_self_energy",
    "fermi_density_from_hamiltonian",
    "fermi_function",
    "fermionic_entropy",
    "linear_self_energy_residuals",
    "make_fermi_density_builder",
    "relative_internal_energy",
    "run_reference_subtracted_hf",
    "weighted_trace_product",
]
