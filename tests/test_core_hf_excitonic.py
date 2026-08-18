from __future__ import annotations

import numpy as np
import pytest

from mean_field.api import (
    ElectronHoleSubspaces,
    FixedChemicalPotential,
    FixedOccupation,
    LinearSelfEnergyFunctional,
    ReferenceSubtractedHFConfig,
    fermi_density_from_hamiltonian,
    make_fermi_density_builder,
    run_reference_subtracted_hf,
)
from mean_field.core.hf.engine import DensityUpdateResult
from mean_field.core.hf.excitonic import (
    ReferenceSubtractedHFState,
    ThermodynamicDensityBuilder,
    certify_linear_self_energy,
    fermionic_entropy,
    linear_self_energy_residuals,
    relative_internal_energy,
    weighted_trace_product,
)


def _field(*matrices: np.ndarray) -> np.ndarray:
    return np.stack(matrices, axis=2).astype(np.complex128)


def _certified_functional(
    builder,
    shape: tuple[int, int, int],
    weights: np.ndarray,
    *,
    component_builder=None,
    label: str = "interaction",
    validation_label: str = "analytic test probes",
) -> LinearSelfEnergyFunctional:
    dimension, _, nk = shape
    first = np.zeros(shape, dtype=complex)
    second = np.zeros(shape, dtype=complex)
    for ik in range(nk):
        first[:, :, ik] = np.diag(
            np.linspace(-0.2, 0.2, dimension) * (ik + 1)
        )
        if dimension > 1:
            second[0, -1, ik] = 0.07 + 0.03j * (ik + 1)
            second[-1, 0, ik] = second[0, -1, ik].conjugate()
        else:
            second[0, 0, ik] = 0.11 * (ik + 1)
    return LinearSelfEnergyFunctional.from_probes(
        builder,
        first,
        second,
        weights,
        validation_label=validation_label,
        component_builder=component_builder,
        label=label,
    )


def test_fixed_mu_density_is_shift_covariant_and_never_reroots() -> None:
    hamiltonian = _field(
        np.array([[-0.4, 0.08j], [-0.08j, 0.3]]),
        np.array([[-0.1, 0.05], [0.05, 0.6]]),
    )
    weights = np.array([0.17, 0.83])
    thermal = 0.09
    mu = 0.13
    base = fermi_density_from_hamiltonian(
        hamiltonian,
        weights,
        thermal_energy=thermal,
        ensemble=FixedChemicalPotential(mu),
    )
    shift = 7.4
    shifted = fermi_density_from_hamiltonian(
        hamiltonian + shift * np.eye(2)[:, :, None],
        weights,
        thermal_energy=thermal,
        ensemble=FixedChemicalPotential(mu + shift),
    )
    assert base.mu == pytest.approx(mu, abs=0.0)
    assert shifted.mu == pytest.approx(mu + shift, abs=0.0)
    np.testing.assert_allclose(shifted.density, base.density, atol=2.0e-14)
    np.testing.assert_allclose(shifted.energies, base.energies + shift, atol=2.0e-14)


def test_fixed_occupation_root_uses_nonuniform_weights_and_shifts_with_energy_zero() -> None:
    hamiltonian = _field(
        np.diag([-0.8, 0.2, 0.7]),
        np.diag([-0.3, 0.1, 1.2]),
        np.diag([-0.6, 0.4, 0.9]),
    )
    weights = np.array([0.1, 0.25, 0.65])
    ensemble = FixedOccupation(1.35)
    base = fermi_density_from_hamiltonian(
        hamiltonian, weights, thermal_energy=0.07, ensemble=ensemble
    )
    achieved = np.einsum("k,aak->", weights, base.density).real / np.sum(weights)
    assert achieved == pytest.approx(1.35, abs=2.0e-13)
    shift = -3.2
    shifted = fermi_density_from_hamiltonian(
        hamiltonian + shift * np.eye(3)[:, :, None],
        weights,
        thermal_energy=0.07,
        ensemble=ensemble,
    )
    assert shifted.mu == pytest.approx(base.mu + shift, abs=2.0e-13)
    np.testing.assert_allclose(shifted.density, base.density, atol=2.0e-13)


def test_explicit_unequal_electron_hole_subspaces_define_carriers_and_coherence() -> None:
    subspaces = ElectronHoleSubspaces(electron_indices=(0,), hole_indices=(1, 2))
    density = _field(np.diag([0.3, 0.8, 0.6]))
    weights = np.array([0.4])
    electron, hole = subspaces.carrier_densities(density, weights)
    assert electron == pytest.approx(0.12)
    assert hole == pytest.approx(0.4 * ((1.0 - 0.8) + (1.0 - 0.6)))

    field = np.zeros((3, 3, 1), dtype=complex)
    field[0, 1:, 0] = np.array([3.0, 4.0j])
    singular = subspaces.coherence_singular_values(field)
    assert singular.shape == (1, 1)
    assert singular[0, 0] == pytest.approx(5.0)


def test_weighted_trace_product_is_ket_oriented_for_complex_coherence() -> None:
    left = _field(np.array([[0.2, 1.0 + 2.0j], [1.0 - 2.0j, -0.4]]))
    right = _field(np.array([[0.7, -0.3 + 0.8j], [-0.3 - 0.8j, 0.1]]))
    weights = np.array([0.37])
    expected = 0.37 * np.trace(left[:, :, 0] @ right[:, :, 0])
    actual = weighted_trace_product(left, right, weights)
    wrong_stored_orientation = 0.37 * np.einsum("ab,ab->", left[:, :, 0], right[:, :, 0])
    assert actual == pytest.approx(expected)
    assert abs(actual - wrong_stored_orientation) > 0.1


def test_linear_self_energy_contract_and_component_energy() -> None:
    weights = np.array([0.2, 0.8])
    first = _field(
        np.array([[0.1, 0.02j], [-0.02j, -0.1]]),
        np.array([[0.2, 0.04], [0.04, -0.2]]),
    )
    second = _field(
        np.array([[-0.3, 0.07], [0.07, 0.3]]),
        np.array([[0.05, -0.03j], [0.03j, -0.05]]),
    )
    coupling = -1.7
    builder = lambda density: coupling * density
    functional = _certified_functional(
        builder,
        first.shape,
        weights,
        component_builder=lambda density: {"exchange": coupling * density},
        label="exchange",
        validation_label="analytic scalar linear self-energy oracle",
    )
    residuals = linear_self_energy_residuals(builder, first, second, weights)
    assert residuals.maximum_error < 2.0e-16
    one_body, components, total = relative_internal_energy(
        np.zeros_like(first), first, weights, functional.components(first)
    )
    expected = 0.5 * coupling * weighted_trace_product(first, first, weights).real
    assert one_body == 0.0
    assert components["exchange"] == pytest.approx(expected)
    assert total == pytest.approx(expected)


def test_exact_two_level_excitonic_fixed_point_and_thermodynamics() -> None:
    thermal = 0.2
    order = 0.25
    coupling = 8.0 * thermal * np.arctanh(0.5)
    h0 = np.zeros((2, 2, 1), dtype=complex)
    reference = 0.5 * np.eye(2)[:, :, None]
    initial = np.zeros_like(h0)
    initial[0, 1, 0] = order
    initial[1, 0, 0] = order
    weights = np.array([1.0])
    ensemble = FixedChemicalPotential(0.0)
    density_builder = make_fermi_density_builder(
        weights, thermal_energy=thermal, ensemble=ensemble
    )
    functional = _certified_functional(
        lambda density: -coupling * density,
        h0.shape,
        weights,
        component_builder=lambda density: {"exchange": -coupling * density},
        label="exchange",
        validation_label="exact two-level excitonic oracle",
    )
    result = run_reference_subtracted_hf(
        h0,
        weights,
        reference,
        absolute_density_builder=density_builder,
        interaction=functional,
        config=ReferenceSubtractedHFConfig(
            thermal_energy=thermal,
            mixing=1.0,
            precision=1.0e-12,
            max_iter=4,
            search_mode="seeded_ei",
            grand_canonical_mu=0.0,
        ),
        initial_density_delta=initial,
        electron_hole_subspaces=ElectronHoleSubspaces((0,), (1,)),
    )
    assert result.converged
    assert result.exit_reason == "converged"
    np.testing.assert_allclose(result.density_delta, initial, atol=2.0e-13)
    assert result.self_energy_coherence_singular_values[0, 0] == pytest.approx(
        coupling * order, abs=2.0e-13
    )
    assert result.energy.interaction_components["exchange"] == pytest.approx(
        -coupling * order**2, abs=2.0e-13
    )
    assert result.energy.grand_potential == pytest.approx(result.energy.free_energy)


def test_interaction_off_keeps_absolute_density_not_reference_density() -> None:
    h0 = np.diag([-0.2, 0.35]).astype(complex)[:, :, None]
    weights = np.array([0.6])
    reference = np.diag([0.0, 1.0]).astype(complex)[:, :, None]
    density_builder = make_fermi_density_builder(
        weights,
        thermal_energy=0.08,
        ensemble=FixedChemicalPotential(0.0),
    )
    zero = _certified_functional(
        lambda density: np.zeros_like(density),
        h0.shape,
        weights,
        validation_label="analytic zero interaction",
    )
    result = run_reference_subtracted_hf(
        h0,
        weights,
        reference,
        absolute_density_builder=density_builder,
        interaction=zero,
        config=ReferenceSubtractedHFConfig(
            thermal_energy=0.08,
            mixing=1.0,
            precision=1.0e-13,
            max_iter=2,
            grand_canonical_mu=0.0,
        ),
    )
    normal = density_builder(h0).density
    np.testing.assert_allclose(result.total_density, normal, atol=1.0e-14)
    np.testing.assert_allclose(result.density_delta, normal - reference, atol=1.0e-14)
    assert np.max(np.abs(result.density_delta)) > 0.1
    assert fermionic_entropy(result.total_density, weights) > 0.0


def test_thermodynamic_builder_binds_temperature_and_grand_canonical_mu() -> None:
    h0 = np.diag([-0.2, 0.3]).astype(complex)[:, :, None]
    reference = 0.5 * np.eye(2)[:, :, None]
    weights = np.ones(1)
    fixed_number = make_fermi_density_builder(
        weights,
        thermal_energy=0.1,
        ensemble=FixedOccupation(1.0),
    )
    zero = _certified_functional(
        lambda density: np.zeros_like(density),
        h0.shape,
        weights,
        validation_label="analytic zero interaction",
    )
    with pytest.raises(ValueError, match="thermal energy disagrees"):
        run_reference_subtracted_hf(
            h0,
            weights,
            reference,
            absolute_density_builder=fixed_number,
            interaction=zero,
            config=ReferenceSubtractedHFConfig(thermal_energy=0.2),
        )
    with pytest.raises(ValueError, match="immutable fixed-mu"):
        run_reference_subtracted_hf(
            h0,
            weights,
            reference,
            absolute_density_builder=fixed_number,
            interaction=zero,
            config=ReferenceSubtractedHFConfig(
                thermal_energy=0.1,
                grand_canonical_mu=0.0,
            ),
        )


def test_final_raw_residual_is_the_single_convergence_authority() -> None:
    h0 = np.zeros((1, 1, 1), dtype=complex)
    reference = np.full((1, 1, 1), 0.5, dtype=complex)
    weights = np.ones(1)

    def affine_density(hamiltonian: np.ndarray) -> DensityUpdateResult:
        density = reference + 3.0 * np.asarray(hamiltonian) + 0.075
        return DensityUpdateResult(
            density=density,
            energies=np.zeros((1, 1)),
            mu=0.0,
        )

    builder = ThermodynamicDensityBuilder(
        affine_density,
        thermal_energy=0.1,
        constraint_label="analytic noncontractive final-residual oracle",
    )
    result = run_reference_subtracted_hf(
        h0,
        weights,
        reference,
        absolute_density_builder=builder,
        interaction=_certified_functional(
            lambda density: density,
            h0.shape,
            weights,
            validation_label="analytic identity self-energy",
        ),
        config=ReferenceSubtractedHFConfig(
            thermal_energy=0.1,
            mixing=0.5,
            precision=0.2,
            max_iter=2,
            search_mode="seeded_ei",
            convergence_scale=0.5,
        ),
        initial_density_delta=np.zeros_like(h0),
    )
    assert not result.converged
    assert not result.run.converged
    assert result.exit_reason == result.run.exit_reason == "final_raw_residual"
    assert result.run.iterations == 1
    assert result.run.iter_err[0] < result.config.precision
    assert result.run.state.diagnostics["final_raw_norm"] > result.config.precision


def test_zero_reference_uses_a_finite_default_convergence_scale() -> None:
    h0 = np.diag([-0.2, 0.3]).astype(complex)[:, :, None]
    reference = np.zeros((2, 2, 1), dtype=complex)
    weights = np.ones(1)
    builder = make_fermi_density_builder(
        weights,
        thermal_energy=0.1,
        ensemble=FixedChemicalPotential(0.0),
    )
    result = run_reference_subtracted_hf(
        h0,
        weights,
        reference,
        absolute_density_builder=builder,
        interaction=_certified_functional(
            lambda density: np.zeros_like(density),
            h0.shape,
            weights,
            validation_label="analytic zero interaction",
        ),
        config=ReferenceSubtractedHFConfig(
            thermal_energy=0.1,
            mixing=1.0,
            precision=1.0e-13,
            max_iter=2,
            grand_canonical_mu=0.0,
        ),
    )
    assert result.run.converged
    assert result.run.state.diagnostics["final_raw_norm"] < 1.0e-13


def test_invalid_indices_and_component_nonclosure_fail_closed() -> None:
    with pytest.raises(TypeError, match="exact integers"):
        ElectronHoleSubspaces((0.5,), (1,))
    with pytest.raises(TypeError, match="not booleans"):
        ElectronHoleSubspaces((True,), (1,))
    density = np.zeros((2, 2, 1), dtype=complex)
    weights = np.ones(1)
    with pytest.raises(ValueError, match="certificate label"):
        _certified_functional(
            lambda values: values,
            density.shape,
            weights,
            validation_label="",
        )

    first_builder = lambda values: values
    second_builder = lambda values: values.copy()
    probe = np.eye(2, dtype=complex)[:, :, None]
    certificate = certify_linear_self_energy(
        first_builder,
        probe,
        0.3 * probe,
        weights,
        validation_label="builder identity binding oracle",
    )
    with pytest.raises(ValueError, match="belongs to another builder"):
        LinearSelfEnergyFunctional(second_builder, certificate=certificate)

    identity = np.eye(2, dtype=complex)[:, :, None]
    bad_components = _certified_functional(
        lambda values: np.zeros_like(values),
        density.shape,
        weights,
        validation_label="malformed component oracle",
        component_builder=lambda _values: {
            "positive": identity,
            "negative": -identity + 1.0e-8 * identity,
        },
    )
    with pytest.raises(ValueError, match="do not close"):
        bad_components.components(density)


def test_nonlinear_and_non_self_adjoint_actions_are_detected_by_oracle() -> None:
    first = _field(np.array([[0.2, 0.1j], [-0.1j, -0.2]]))
    second = _field(np.array([[-0.1, 0.04], [0.04, 0.1]]))
    weights = np.ones(1)
    nonlinear = lambda density: np.einsum("abk,bck->ack", density, density)
    nonlinear_residuals = linear_self_energy_residuals(
        nonlinear, first, second, weights
    )
    assert nonlinear_residuals.additivity_error > 1.0e-3
    with pytest.raises(ValueError, match="certificate failed"):
        LinearSelfEnergyFunctional.from_probes(
            nonlinear,
            first,
            second,
            weights,
            validation_label="negative nonlinear diagnostic",
        )

    generator = np.diag([1.0, -1.0]).astype(complex)
    anti_self_adjoint = lambda density: 1j * (
        np.einsum("ab,bck->ack", generator, density)
        - np.einsum("abk,bc->ack", density, generator)
    )
    residuals = linear_self_energy_residuals(
        anti_self_adjoint, first, second, weights
    )
    assert residuals.self_adjoint_error > 1.0e-3
    with pytest.raises(ValueError, match="certificate failed"):
        LinearSelfEnergyFunctional.from_probes(
            anti_self_adjoint,
            first,
            second,
            weights,
            validation_label="negative anti-self-adjoint diagnostic",
        )
