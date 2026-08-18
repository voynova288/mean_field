"""InAs/GaSb adapter for the generic reference-subtracted matrix-HF API.

The system-independent solver iterates the ket-oriented density difference
``D(k)=P(k)-P_ref(k)`` in :mod:`mean_field.core.hf.excitonic`.  This module
retains only Kane/E1/H1 physics: source-bound interactions, carrier constraints,
the E1-empty/H1-filled ordinary-electron vacuum, microscopic charge closure,
units, fingerprints, and compatibility result names.

For ``kane_poisson_fixed_mu``, the chemical potential must come from the same
energy gauge and frozen Kane--Poisson Hamiltonian as the bundle.  This solver
does not rerun Poisson or close a fixed-gate electrostatic ensemble.  When the
active particle number changes, the grand-potential difference—not the legacy
canonical-free-energy compatibility field—is the matching equilibrium
potential.  Active E1/H1 index counting also does not replace the microscopic
Kane orbital-projector carrier diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mean_field.core.hf.engine import (
    DensityUpdateResult,
    HartreeFockRun,
    HartreeFockStateProtocol,
    HartreeFockStepResult,
)
from mean_field.core.hf.excitonic import (
    ElectronHoleSubspaces,
    FixedChemicalPotential,
    FixedOccupation,
    LinearSelfEnergyFunctional,
    ReferenceSubtractedHFConfig,
    ReferenceSubtractedHFState,
    ThermodynamicDensityBuilder,
    fermi_density_from_hamiltonian,
    fermi_function,
    fermionic_entropy,
    relative_internal_energy,
    run_reference_subtracted_hf,
)

from .angular_fock import PolarHarmonicProjectedFockOperator
from .axial_fock import (
    AxialAveragedProjectedFockOperator,
    AxialProjectedFockOperator,
)
from .carriers import KaneCarrierProjectors, charge_neutral_fermi_density
from .hartree import ProjectedHartreeFockOperator, reference_subtracted_charge_density
from .projected_model import (
    Kane4Bundle,
    ProjectedFockOperator,
    excitonic_fock_singular_values,
)

KB_MEV_PER_K = 0.08617333262
ComplexArray = np.ndarray
FloatArray = np.ndarray
ReferencePolicy = Literal["normal_ordered_exchange_only", "normal_ordered_hartree_fock"]
SearchMode = Literal["normal_reference", "seeded_ei"]
ConstraintPolicy = Literal[
    "active_half_filling",
    "microscopic_charge_neutrality",
    "kane_poisson_fixed_mu",
]
NormalOrderingReferencePolicy = Literal[
    "noninteracting_fermi_state",
    "electron_hole_vacuum",
]


@dataclass(frozen=True)
class MatrixEIConfig:
    temperature_K: float = 0.1
    target_occupation_per_k: float = 2.0
    mixing: float = 0.2
    precision: float = 1.0e-8
    max_iter: int = 500
    reference_policy: ReferencePolicy = "normal_ordered_exchange_only"
    search_mode: SearchMode = "normal_reference"
    constraint_policy: ConstraintPolicy = "active_half_filling"
    fixed_mu_mev: float | None = None
    normal_ordering_reference_policy: NormalOrderingReferencePolicy = (
        "noninteracting_fermi_state"
    )

    def __post_init__(self) -> None:
        if self.temperature_K <= 0.0:
            raise ValueError("the weighted solver requires temperature_K > 0")
        if abs(self.target_occupation_per_k - 2.0) > 1.0e-12:
            raise ValueError("the Kane E1/H1 adapter requires target_occupation_per_k=2")
        if not 1.0e-3 <= self.mixing <= 1.0:
            raise ValueError("mixing must lie in [1e-3, 1]")
        if self.precision <= 0.0 or self.max_iter < 1:
            raise ValueError("precision and max_iter must be positive")
        if self.reference_policy not in {
            "normal_ordered_exchange_only",
            "normal_ordered_hartree_fock",
        }:
            raise ValueError("unsupported reference policy")
        if self.search_mode not in {"normal_reference", "seeded_ei"}:
            raise ValueError("unsupported matrix-EI search mode")
        if self.constraint_policy not in {
            "active_half_filling",
            "microscopic_charge_neutrality",
            "kane_poisson_fixed_mu",
        }:
            raise ValueError("unsupported matrix-EI constraint policy")
        if self.constraint_policy == "kane_poisson_fixed_mu":
            if self.fixed_mu_mev is None or not np.isfinite(self.fixed_mu_mev):
                raise ValueError("kane_poisson_fixed_mu requires a finite fixed_mu_mev")
        elif self.fixed_mu_mev is not None:
            raise ValueError(
                "fixed_mu_mev is only valid with constraint_policy="
                "'kane_poisson_fixed_mu'"
            )
        if self.normal_ordering_reference_policy not in {
            "noninteracting_fermi_state",
            "electron_hole_vacuum",
        }:
            raise ValueError("unsupported normal-ordering reference policy")


MatrixEIState = ReferenceSubtractedHFState


@dataclass(frozen=True)
class MatrixEIResult:
    classification: str
    config: MatrixEIConfig
    bundle_fingerprint: str
    interaction_fingerprint: str
    reference_mu_mev: float
    reference_density: ComplexArray
    noninteracting_density: ComplexArray
    density_delta: ComplexArray
    total_density: ComplexArray
    sigma_hartree_mev: ComplexArray
    sigma_fock_mev: ComplexArray
    interaction_h_mev: ComplexArray
    hamiltonian_mev: ComplexArray
    energies_mev: FloatArray
    mu_mev: float
    one_body_internal_energy_density_mev_nm2: float
    hartree_internal_energy_density_mev_nm2: float
    fock_internal_energy_density_mev_nm2: float
    total_internal_energy_density_mev_nm2: float
    entropy_difference_mev_per_K_nm2: float
    canonical_free_energy_difference_mev_nm2: float
    grand_potential_difference_mev_nm2: float | None
    excitonic_singular_values_mev: FloatArray | None
    run: HartreeFockRun


def _fermi(energy_minus_mu_mev: FloatArray, temperature_K: float) -> FloatArray:
    return fermi_function(
        energy_minus_mu_mev,
        KB_MEV_PER_K * float(temperature_K),
    )


def weighted_fermi_density(
    hamiltonian_mev: ComplexArray,
    k_weights_nm2: FloatArray,
    *,
    temperature_K: float,
    target_occupation_per_k: float,
) -> DensityUpdateResult:
    """Compatibility wrapper for generic fixed-occupation Fermi density."""

    update = fermi_density_from_hamiltonian(
        hamiltonian_mev,
        k_weights_nm2,
        thermal_energy=KB_MEV_PER_K * float(temperature_K),
        ensemble=FixedOccupation(float(target_occupation_per_k)),
    )
    return DensityUpdateResult(
        density=update.density,
        energies=update.energies,
        mu=update.mu,
        observables={
            "target_occupation_per_k": float(target_occupation_per_k),
            "achieved_occupation_per_k": float(
                update.observables["achieved_occupation_per_k"]
            ),
            "number_residual_nm2": float(update.observables["number_residual"]),
            "hamiltonian_hermiticity_error_mev": float(
                update.observables["hamiltonian_hermiticity_error"]
            ),
        },
    )


def fixed_mu_fermi_density(
    hamiltonian_mev: ComplexArray,
    k_weights_nm2: FloatArray,
    *,
    temperature_K: float,
    mu_mev: float,
) -> DensityUpdateResult:
    """Evaluate occupations at one immutable ordinary-electron chemical potential."""

    update = fermi_density_from_hamiltonian(
        hamiltonian_mev,
        k_weights_nm2,
        thermal_energy=KB_MEV_PER_K * float(temperature_K),
        ensemble=FixedChemicalPotential(float(mu_mev)),
    )
    return DensityUpdateResult(
        density=update.density,
        energies=update.energies,
        mu=update.mu,
        observables={
            "achieved_occupation_per_k": float(
                update.observables["achieved_occupation_per_k"]
            ),
            "active_number_density_nm2": float(update.observables["active_number"]),
            "fixed_mu_mev": float(mu_mev),
            "hamiltonian_hermiticity_error_mev": float(
                update.observables["hamiltonian_hermiticity_error"]
            ),
        },
    )


def fermionic_entropy_density_mev_per_K_nm2(
    density: ComplexArray,
    k_weights_nm2: FloatArray,
) -> float:
    """Return ``k_B`` times the generic dimensionless weighted entropy."""

    return KB_MEV_PER_K * fermionic_entropy(density, k_weights_nm2)


def relative_internal_energy_components_mev_nm2(
    sigma_fock_mev: ComplexArray,
    h0_mev: ComplexArray,
    density_delta: ComplexArray,
    k_weights_nm2: FloatArray,
) -> tuple[float, float, float]:
    """Return one-body, exchange, and total reference-relative internal energy."""

    one_body, components, total = relative_internal_energy(
        h0_mev,
        density_delta,
        k_weights_nm2,
        {"exchange": sigma_fock_mev},
    )
    return one_body, components["exchange"], total


def relative_hf_energy_density_mev_nm2(
    interaction_h_mev: ComplexArray,
    h0_mev: ComplexArray,
    density_delta: ComplexArray,
    k_weights_nm2: FloatArray,
) -> float:
    """Compatibility helper returning the reference-relative internal energy."""

    return relative_internal_energy_components_mev_nm2(
        interaction_h_mev,
        h0_mev,
        density_delta,
        k_weights_nm2,
    )[2]


def _validate_reference_density(
    reference: ComplexArray,
    h0_shape: tuple[int, ...],
    weights: FloatArray,
    target: float | None,
) -> None:
    p0 = np.asarray(reference, dtype=np.complex128)
    if p0.shape != h0_shape:
        raise ValueError("reference density shape does not match h0")
    hermiticity_error = float(np.max(np.abs(p0 - np.swapaxes(p0.conj(), 0, 1))))
    if hermiticity_error > 1.0e-9:
        raise ValueError("reference density is not Hermitian")
    eigenvalues = np.concatenate(
        [np.linalg.eigvalsh(p0[:, :, ik]) for ik in range(p0.shape[2])]
    )
    if float(np.min(eigenvalues)) < -1.0e-8 or float(np.max(eigenvalues)) > 1.0 + 1.0e-8:
        raise ValueError("reference density eigenvalues lie outside [0, 1]")
    if target is not None:
        achieved = float(
            np.einsum("k,aak->", weights, p0, optimize=True).real / np.sum(weights)
        )
        if abs(achieved - target) > 1.0e-9:
            raise ValueError("reference density has the wrong active-space filling")


def _interaction_certification_probes(bundle: Kane4Bundle) -> tuple[ComplexArray, ComplexArray]:
    """Return deterministic Hermitian density probes in the exact bundle basis."""

    first = np.empty_like(bundle.h0_mev, dtype=np.complex128)
    second = np.zeros_like(bundle.h0_mev, dtype=np.complex128)
    electron = bundle.basis.electron_indices
    hole = bundle.basis.hole_indices
    for ik in range(bundle.nk):
        radial_factor = float(ik + 1) / float(bundle.nk)
        first[:, :, ik] = 0.2 * radial_factor * bundle.basis.tau_z
        for pair_index, (electron_index, hole_index) in enumerate(zip(electron, hole)):
            value = radial_factor * (0.07 + 0.02j * (pair_index + 1))
            second[electron_index, hole_index, ik] = value
            second[hole_index, electron_index, ik] = value.conjugate()
    return first, second


def solve_reference_subtracted_matrix_ei(
    bundle: Kane4Bundle,
    interaction_operator: (
        ProjectedFockOperator
        | AxialProjectedFockOperator
        | AxialAveragedProjectedFockOperator
        | PolarHarmonicProjectedFockOperator
        | ProjectedHartreeFockOperator
    ),
    *,
    config: MatrixEIConfig | None = None,
    seed_hamiltonian_mev: ComplexArray | None = None,
    initial_density_delta: ComplexArray | None = None,
    initial_density_is_postprocessed: bool = False,
    density_symmetrizer: Callable[[ComplexArray], ComplexArray] | None = None,
    symmetry_constraint_label: str | None = None,
    step_callback: (
        Callable[[HartreeFockStateProtocol, HartreeFockStepResult], None] | None
    ) = None,
    final_state_callback: (
        Callable[[HartreeFockStateProtocol, DensityUpdateResult], None] | None
    ) = None,
) -> MatrixEIResult:
    """Run the source-bound Kane adapter through the generic matrix-HF core."""

    cfg = MatrixEIConfig() if config is None else config
    bundle.validate()
    bundle_fingerprint = bundle.fingerprint()
    if isinstance(interaction_operator, ProjectedHartreeFockOperator):
        required_policy = "normal_ordered_hartree_fock"
        model_label = "normal-ordered Hartree-Fock"

        def component_builder(density: ComplexArray) -> dict[str, ComplexArray]:
            sigma_hartree, sigma_fock = interaction_operator.components(density)
            return {"hartree": sigma_hartree, "fock": sigma_fock}

    elif isinstance(interaction_operator, AxialAveragedProjectedFockOperator):
        required_policy = "normal_ordered_exchange_only"
        model_label = "normal-ordered co-rotating-m0 exchange diagnostic"
        component_builder = lambda density: {"fock": interaction_operator(density)}
    elif isinstance(interaction_operator, PolarHarmonicProjectedFockOperator):
        required_policy = "normal_ordered_exchange_only"
        model_label = "normal-ordered selected-harmonic exchange diagnostic"
        component_builder = lambda density: {"fock": interaction_operator(density)}
    elif isinstance(interaction_operator, (ProjectedFockOperator, AxialProjectedFockOperator)):
        required_policy = "normal_ordered_exchange_only"
        model_label = "normal-ordered exchange-only"
        component_builder = lambda density: {"fock": interaction_operator(density)}
    else:
        raise TypeError("matrix-EI solver requires a certified projected interaction operator")
    if cfg.reference_policy != required_policy:
        raise ValueError(
            f"reference_policy={cfg.reference_policy!r} does not match interaction "
            f"{required_policy!r}"
        )
    interaction_operator.validate_against_bundle(bundle)
    interaction_fingerprint = str(interaction_operator.fingerprint())
    if not interaction_fingerprint:
        raise ValueError("interaction operator fingerprint is required")

    h0 = np.asarray(bundle.h0_mev, dtype=np.complex128)
    weights = np.asarray(bundle.weights_nm2, dtype=float)
    carrier_projectors = (
        KaneCarrierProjectors.from_bundle(bundle)
        if cfg.constraint_policy == "microscopic_charge_neutrality"
        else None
    )

    def constrained_density(total_hamiltonian: ComplexArray) -> DensityUpdateResult:
        if cfg.constraint_policy == "kane_poisson_fixed_mu":
            assert cfg.fixed_mu_mev is not None
            return fixed_mu_fermi_density(
                total_hamiltonian,
                weights,
                temperature_K=cfg.temperature_K,
                mu_mev=cfg.fixed_mu_mev,
            )
        if carrier_projectors is None:
            return weighted_fermi_density(
                total_hamiltonian,
                weights,
                temperature_K=cfg.temperature_K,
                target_occupation_per_k=cfg.target_occupation_per_k,
            )
        neutral = charge_neutral_fermi_density(
            total_hamiltonian,
            weights,
            carrier_projectors,
            temperature_K=cfg.temperature_K,
        )
        return DensityUpdateResult(
            density=neutral.density,
            energies=neutral.energies_mev,
            mu=neutral.mu_mev,
            observables={
                "electron_density_nm2": neutral.electron_density_nm2,
                "hole_density_nm2": neutral.hole_density_nm2,
                "charge_imbalance_nm2": neutral.charge_imbalance_nm2,
            },
        )

    thermodynamic_density_builder = ThermodynamicDensityBuilder(
        builder=constrained_density,
        thermal_energy=KB_MEV_PER_K * cfg.temperature_K,
        constraint_label=cfg.constraint_policy,
        fixed_chemical_potential=(
            cfg.fixed_mu_mev
            if cfg.constraint_policy == "kane_poisson_fixed_mu"
            else None
        ),
    )
    normal = thermodynamic_density_builder(h0)
    noninteracting_density = np.asarray(normal.density, dtype=np.complex128)
    if cfg.normal_ordering_reference_policy == "noninteracting_fermi_state":
        reference_density = noninteracting_density.copy()
    else:
        reference_density = np.repeat(
            bundle.basis.h1_electron_projector[:, :, None], bundle.nk, axis=2
        ).astype(np.complex128, copy=False)
    reference_target = (
        cfg.target_occupation_per_k
        if (
            cfg.constraint_policy == "active_half_filling"
            or cfg.normal_ordering_reference_policy == "electron_hole_vacuum"
        )
        else None
    )
    _validate_reference_density(reference_density, h0.shape, weights, reference_target)
    if carrier_projectors is not None:
        electron0, hole0 = carrier_projectors.densities_nm2(
            noninteracting_density, weights
        )
        if abs(electron0 - hole0) > 1.0e-9:
            raise ValueError("microscopic noninteracting density is not charge neutral")

    first_probe, second_probe = _interaction_certification_probes(bundle)
    interaction = LinearSelfEnergyFunctional.from_probes(
        interaction_operator,
        first_probe,
        second_probe,
        weights,
        validation_label="runtime Kane weighted linear/self-adjoint probes",
        component_builder=component_builder,
        label="interaction",
        absolute_tolerance=1.0e-11,
        relative_tolerance=1.0e-9,
        operator_fingerprint=interaction_fingerprint,
    )
    core_result = run_reference_subtracted_hf(
        h0,
        weights,
        reference_density,
        absolute_density_builder=thermodynamic_density_builder,
        interaction=interaction,
        config=ReferenceSubtractedHFConfig(
            thermal_energy=KB_MEV_PER_K * cfg.temperature_K,
            mixing=cfg.mixing,
            precision=cfg.precision,
            max_iter=cfg.max_iter,
            search_mode=cfg.search_mode,
            grand_canonical_mu=(
                cfg.fixed_mu_mev
                if cfg.constraint_policy == "kane_poisson_fixed_mu"
                else None
            ),
            convergence_scale=float(
                np.sqrt(
                    np.einsum(
                        "k,abk,abk->",
                        weights,
                        reference_density.conj(),
                        reference_density,
                        optimize=True,
                    ).real
                )
            ),
        ),
        normal_density_update=normal,
        electron_hole_subspaces=ElectronHoleSubspaces(
            bundle.basis.electron_indices,
            bundle.basis.hole_indices,
        ),
        seed_hamiltonian=seed_hamiltonian_mev,
        initial_density_delta=initial_density_delta,
        initial_density_is_postprocessed=initial_density_is_postprocessed,
        density_symmetrizer=density_symmetrizer,
        step_callback=step_callback,
        final_state_callback=final_state_callback,
    )

    sigma_hartree = core_result.interaction_components.get(
        "hartree", np.zeros_like(h0)
    )
    sigma_fock = core_result.interaction_components["fock"]
    charge_delta = reference_subtracted_charge_density(bundle, core_result.density_delta)
    active_number_change = core_result.energy.particle_number_change
    if abs(active_number_change - charge_delta.active_number_change_nm2) > 1.0e-9:
        raise ValueError("active-number change disagrees with integrated charge vertices")

    one_body = core_result.energy.one_body
    hartree = core_result.energy.interaction_components.get("hartree", 0.0)
    exchange = core_result.energy.interaction_components["fock"]
    internal = core_result.energy.internal_energy
    entropy_difference_per_k = core_result.energy.entropy_difference
    entropy_difference_mev_per_K = KB_MEV_PER_K * entropy_difference_per_k
    free_energy = core_result.energy.free_energy
    grand_potential = core_result.energy.grand_potential
    state = core_result.run.state
    state.diagnostics.update(
        {
            "one_body_internal_energy_density_mev_nm2": one_body,
            "hartree_internal_energy_density_mev_nm2": hartree,
            "fock_internal_energy_density_mev_nm2": exchange,
            "total_internal_energy_density_mev_nm2": internal,
            "entropy_difference_mev_per_K_nm2": entropy_difference_mev_per_K,
            "canonical_free_energy_difference_mev_nm2": free_energy,
            "active_number_change_nm2": active_number_change,
            "integrated_charge_profile_nm2": charge_delta.integrated_profile_nm2,
            "charge_sum_rule_error_nm2": charge_delta.sum_rule_error_nm2,
        }
    )
    if grand_potential is not None:
        state.diagnostics["grand_potential_difference_mev_nm2"] = grand_potential
        state.diagnostics["fixed_mu_mev"] = float(cfg.fixed_mu_mev)

    singular_values = (
        excitonic_fock_singular_values(sigma_fock, bundle.basis)
        if core_result.converged
        else None
    )
    constraint_suffix = (
        "" if symmetry_constraint_label is None else f" [{symmetry_constraint_label}]"
    )
    if cfg.constraint_policy == "kane_poisson_fixed_mu":
        constraint_suffix += (
            f" [fixed Kane-Poisson ordinary-electron mu={cfg.fixed_mu_mev:.12g} meV]"
        )
    if cfg.normal_ordering_reference_policy == "electron_hole_vacuum":
        constraint_suffix += " [E1-empty/H1-filled normal ordering]"
    classification = (
        f"converged {model_label} matrix-EI scaffold{constraint_suffix}"
        if core_result.converged
        else f"nonconverged matrix-EI scaffold{constraint_suffix}: {core_result.exit_reason}"
    )
    return MatrixEIResult(
        classification=classification,
        config=cfg,
        bundle_fingerprint=bundle_fingerprint,
        interaction_fingerprint=interaction_fingerprint,
        reference_mu_mev=float(normal.mu),
        reference_density=core_result.reference_density,
        noninteracting_density=core_result.noninteracting_density,
        density_delta=core_result.density_delta,
        total_density=core_result.total_density,
        sigma_hartree_mev=np.asarray(sigma_hartree, dtype=np.complex128),
        sigma_fock_mev=np.asarray(sigma_fock, dtype=np.complex128),
        interaction_h_mev=core_result.interaction_hamiltonian,
        hamiltonian_mev=core_result.hamiltonian,
        energies_mev=core_result.energies,
        mu_mev=core_result.chemical_potential,
        one_body_internal_energy_density_mev_nm2=one_body,
        hartree_internal_energy_density_mev_nm2=hartree,
        fock_internal_energy_density_mev_nm2=exchange,
        total_internal_energy_density_mev_nm2=internal,
        entropy_difference_mev_per_K_nm2=entropy_difference_mev_per_K,
        canonical_free_energy_difference_mev_nm2=free_energy,
        grand_potential_difference_mev_nm2=grand_potential,
        excitonic_singular_values_mev=singular_values,
        run=core_result.run,
    )


__all__ = [
    "KB_MEV_PER_K",
    "MatrixEIConfig",
    "MatrixEIResult",
    "MatrixEIState",
    "fermionic_entropy_density_mev_per_K_nm2",
    "fixed_mu_fermi_density",
    "relative_hf_energy_density_mev_nm2",
    "relative_internal_energy_components_mev_nm2",
    "solve_reference_subtracted_matrix_ei",
    "weighted_fermi_density",
]
