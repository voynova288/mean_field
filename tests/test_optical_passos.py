from __future__ import annotations

from dataclasses import replace
from math import factorial

import numpy as np
import pytest

from analysis.optical.benchmarks.passos_2018 import (
    Passos2018GrapheneParameters,
    RiceMeleParameters,
    graphene_direct_lattice_vectors,
    graphene_reciprocal_vectors,
    iter_passos2018_graphene_kpoints,
    iter_rice_mele_kpoints,
    passos2018_graphene_grid,
    passos2018_graphene_refined_grid,
    passos2018_graphene_hamiltonian_and_derivatives,
    passos2018_sigma0_si,
    rice_mele_grid,
    rice_mele_hamiltonian_and_derivatives,
)
from analysis.optical import (
    IndependentInputAdiabaticSwitching,
    PassosFiniteBandVelocityGauge,
    canonical_response_kind,
    harmonic_response_from_kpoint_data,
    optical_response_from_kpoint_data,
    shg_susceptibility_from_response,
)
from analysis.optical.linear import E2_OVER_HBAR_S
from analysis.optical.passos import (
    PassosHarmonicConductivityResult,
    PassosKPointData,
    PassosModelCertificate,
    PassosOccupationSpec,
    PassosPrimitiveBZGrid,
    passos_harmonic_kernel,
    passos_kpoint_data_from_fixed_basis_derivatives,
    passos_shg_conductivity_from_kpoint_data,
    passos_shg_susceptibility,
    passos_si_prefactor,
    passos_thg_conductivity_from_kpoint_data,
)


def _hermitian(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
    return matrix + matrix.conjugate().T


def _one_axis_tower(maximum_order: int) -> tuple[np.ndarray, ...]:
    return tuple(
        _hermitian(order).reshape((1,) * order + (2, 2))
        for order in range(1, maximum_order + 1)
    )


def _commutator(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left @ right - right @ left


def test_passos_shg_matches_literal_eq31_eq34() -> None:
    energies = np.array([-0.8, 0.9])
    occupations = np.array([1.0, 0.0])
    derivatives = _one_axis_tower(3)
    photon = np.array([0.37])
    gamma = 0.021
    z = photon[0] + 1.0j * gamma
    delta = energies[:, None] - energies[None, :]
    rho0 = np.diag(occupations.astype(complex))
    g1 = derivatives[0][0]
    g2 = derivatives[1][0, 0]
    g3 = derivatives[2][0, 0, 0]
    rho1 = _commutator(g1, rho0) / (z - delta)
    rho2 = (
        _commutator(g1, rho1) + 0.5 * _commutator(g2, rho0)
    ) / (2.0 * z - delta)
    expected = (
        np.trace(g1 @ rho2)
        + np.trace(g2 @ rho1)
        + 0.5 * np.trace(g3 @ rho0)
    ) / z**2
    actual = passos_harmonic_kernel(
        photon,
        energies,
        occupations,
        derivatives,
        order=2,
        adiabatic_gamma_ev=gamma,
    )[0, 0, 0, 0]
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)


def test_passos_thg_matches_literal_eq31_eq32() -> None:
    energies = np.array([-0.8, 0.9])
    occupations = np.array([1.0, 0.0])
    derivatives = _one_axis_tower(4)
    photon = np.array([0.31])
    gamma = 0.017
    z = photon[0] + 1.0j * gamma
    delta = energies[:, None] - energies[None, :]
    rho0 = np.diag(occupations.astype(complex))
    g1 = derivatives[0][0]
    g2 = derivatives[1][0, 0]
    g3 = derivatives[2][0, 0, 0]
    g4 = derivatives[3][0, 0, 0, 0]
    rho1 = _commutator(g1, rho0) / (z - delta)
    rho2 = (
        _commutator(g1, rho1) + _commutator(g2, rho0) / factorial(2)
    ) / (2.0 * z - delta)
    rho3 = (
        _commutator(g1, rho2)
        + _commutator(g2, rho1) / factorial(2)
        + _commutator(g3, rho0) / factorial(3)
    ) / (3.0 * z - delta)
    expected = (
        np.trace(g1 @ rho3)
        + np.trace(g2 @ rho2)
        + np.trace(g3 @ rho1) / factorial(2)
        + np.trace(g4 @ rho0) / factorial(3)
    ) / z**3
    actual = passos_harmonic_kernel(
        photon,
        energies,
        occupations,
        derivatives,
        order=3,
        adiabatic_gamma_ev=gamma,
    )[0, 0, 0, 0, 0]
    np.testing.assert_allclose(actual, expected, rtol=3.0e-13, atol=3.0e-13)


def test_passos_fixed_basis_liouvillian_oracle() -> None:
    hamiltonian = np.array([[0.4, 0.2 - 0.3j], [0.2 + 0.3j, -0.1]])
    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    basis_derivatives = _one_axis_tower(3)
    occupations = np.array([1.0, 0.0])
    grid = PassosPrimitiveBZGrid(
        reciprocal_vectors=np.array([[2.0 * np.pi]]),
        mesh_shape=(1,),
        coordinate_length_unit_m=1.0,
        grid_label="liouvillian_oracle",
    )
    certificate = PassosModelCertificate(
        model_id="liouvillian_oracle",
        source_id="nontrivial_fixed_basis",
        basis_dimension=2,
        maximum_derivative_order=3,
        response_axis_labels=("x",),
        derivative_provenance="fixed_basis_complete",
    )
    occupation_spec = PassosOccupationSpec.stationary_custom(
        state_id="occupied_lower_eigenstate"
    )
    payload = passos_kpoint_data_from_fixed_basis_derivatives(
        hamiltonian,
        energies,
        eigenvectors,
        basis_derivatives,
        grid=grid,
        grid_index=0,
        model_certificate=certificate,
        occupation_spec=occupation_spec,
        fixed_basis_connection_is_zero=True,
        occupations=occupations,
    )
    rho0 = eigenvectors @ np.diag(occupations) @ eigenvectors.conjugate().T
    photon = 0.43
    gamma = 0.019
    z = photon + 1.0j * gamma

    def solve_liouvillian(order: int, rhs: np.ndarray) -> np.ndarray:
        superoperator = np.empty((4, 4), dtype=complex)
        for column in range(4):
            basis_matrix = np.zeros((2, 2), dtype=complex)
            basis_matrix.flat[column] = 1.0
            mapped = (
                order * z * basis_matrix
                - hamiltonian @ basis_matrix
                + basis_matrix @ hamiltonian
            )
            superoperator[:, column] = mapped.ravel()
        return np.linalg.solve(superoperator, rhs.ravel()).reshape(2, 2)

    g1 = basis_derivatives[0][0]
    g2 = basis_derivatives[1][0, 0]
    g3 = basis_derivatives[2][0, 0, 0]
    rho1 = solve_liouvillian(1, _commutator(g1, rho0))
    rho2 = solve_liouvillian(
        2,
        _commutator(g1, rho1) + 0.5 * _commutator(g2, rho0),
    )
    fixed_basis_value = (
        np.trace(g1 @ rho2)
        + np.trace(g2 @ rho1)
        + 0.5 * np.trace(g3 @ rho0)
    ) / z**2
    band_basis_value = passos_harmonic_kernel(
        np.array([photon]),
        energies,
        occupations,
        payload.covariant_derivatives_ev,
        order=2,
        adiabatic_gamma_ev=gamma,
    )[0, 0, 0, 0]
    np.testing.assert_allclose(
        band_basis_value,
        fixed_basis_value,
        rtol=5.0e-13,
        atol=5.0e-13,
    )


def test_passos_mixed_axis_ordering_and_intrinsic_permutation() -> None:
    rng = np.random.default_rng(19)
    derivatives = []
    for order in range(1, 4):
        value = np.empty((2,) * order + (2, 2), dtype=complex)
        symmetric_components: dict[tuple[int, ...], np.ndarray] = {}
        for axes in np.ndindex((2,) * order):
            key = tuple(sorted(axes))
            if key not in symmetric_components:
                matrix = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
                symmetric_components[key] = matrix + matrix.conjugate().T
            value[axes] = symmetric_components[key]
        derivatives.append(value)
    arguments = (
        np.array([0.36]),
        np.array([-0.7, 0.8]),
        np.array([1.0, 0.0]),
        tuple(derivatives),
    )
    ordered = passos_harmonic_kernel(
        *arguments,
        order=2,
        adiabatic_gamma_ev=0.02,
        intrinsic_permutation=False,
    )
    symmetric = passos_harmonic_kernel(
        *arguments,
        order=2,
        adiabatic_gamma_ev=0.02,
        intrinsic_permutation=True,
    )
    assert not np.allclose(ordered[:, 0, 0, 1], ordered[:, 0, 1, 0])
    expected_average = 0.5 * (
        ordered[:, 0, 0, 1] + ordered[:, 0, 1, 0]
    )
    np.testing.assert_allclose(symmetric[:, 0, 0, 1], expected_average)
    np.testing.assert_array_equal(symmetric[:, 0, 0, 1], symmetric[:, 0, 1, 0])


def test_passos_thg_mixed_axis_intrinsic_permutations() -> None:
    rng = np.random.default_rng(23)
    derivatives = []
    for order in range(1, 5):
        value = np.empty((3,) * order + (2, 2), dtype=complex)
        symmetric_components: dict[tuple[int, ...], np.ndarray] = {}
        for axes in np.ndindex((3,) * order):
            key = tuple(sorted(axes))
            if key not in symmetric_components:
                matrix = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
                symmetric_components[key] = matrix + matrix.conjugate().T
            value[axes] = symmetric_components[key]
        derivatives.append(value)
    arguments = (
        np.array([0.34]),
        np.array([-0.7, 0.8]),
        np.array([1.0, 0.0]),
        tuple(derivatives),
    )
    ordered = passos_harmonic_kernel(
        *arguments,
        order=3,
        adiabatic_gamma_ev=0.02,
        intrinsic_permutation=False,
    )
    symmetric = passos_harmonic_kernel(
        *arguments,
        order=3,
        adiabatic_gamma_ev=0.02,
        intrinsic_permutation=True,
    )
    permutations = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    ordered_values = np.stack(
        [ordered[(slice(None), 0, *axes)] for axes in permutations],
        axis=0,
    )
    expected = np.mean(ordered_values, axis=0)
    assert not np.allclose(ordered_values[0], ordered_values[1])
    for axes in permutations:
        np.testing.assert_allclose(symmetric[(slice(None), 0, *axes)], expected)


def test_passos_kernel_is_invariant_under_band_phases() -> None:
    derivatives = _one_axis_tower(3)
    gauge = np.diag(np.exp(1.0j * np.array([0.37, -1.12])))
    transformed = tuple(
        np.einsum(
            "ia,...ij,jb->...ab",
            gauge.conjugate(),
            derivative,
            gauge,
            optimize=True,
        )
        for derivative in derivatives
    )
    arguments = (
        np.array([0.29, 0.41]),
        np.array([-0.8, 0.9]),
        np.array([1.0, 0.0]),
    )
    reference = passos_harmonic_kernel(
        *arguments,
        derivatives,
        order=2,
        adiabatic_gamma_ev=0.013,
    )
    rotated = passos_harmonic_kernel(
        *arguments,
        transformed,
        order=2,
        adiabatic_gamma_ev=0.013,
    )
    np.testing.assert_allclose(rotated, reference, rtol=2.0e-13, atol=2.0e-13)


def test_passos_si_prefactors() -> None:
    length = 1.0e-9
    assert passos_si_prefactor(1, integration_dimension=2, length_unit_m=length) == pytest.approx(
        1.0j * E2_OVER_HBAR_S
    )
    assert passos_si_prefactor(2, integration_dimension=2, length_unit_m=length) == pytest.approx(
        E2_OVER_HBAR_S * length
    )
    assert passos_si_prefactor(3, integration_dimension=2, length_unit_m=length) == pytest.approx(
        -1.0j * E2_OVER_HBAR_S * length**2
    )
    with pytest.raises(TypeError):
        passos_si_prefactor(True, integration_dimension=2, length_unit_m=length)


def test_passos_degenerate_occupations_must_be_covariant() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        passos_harmonic_kernel(
            np.array([0.2]),
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            _one_axis_tower(3),
            order=2,
            adiabatic_gamma_ev=0.01,
        )
    with pytest.raises(ValueError, match="degeneracy_tolerance"):
        passos_harmonic_kernel(
            np.array([0.2]),
            np.array([-1.0, 1.0]),
            np.array([1.0, 0.0]),
            _one_axis_tower(3),
            order=2,
            adiabatic_gamma_ev=0.01,
            degeneracy_tolerance_ev=np.nan,
        )


def _contract_point(
    grid: PassosPrimitiveBZGrid,
    grid_index: int,
    *,
    scale: float,
) -> PassosKPointData:
    hamiltonian = np.diag([-1.0, 1.0]).astype(complex)
    energies = np.array([-1.0, 1.0])
    eigenvectors = np.eye(2, dtype=complex)
    derivatives = tuple(
        (scale * _hermitian(order)).reshape((1,) * order + (2, 2))
        for order in range(1, 4)
    )
    certificate = PassosModelCertificate(
        model_id="contract_two_band",
        source_id="test_contract_v1",
        basis_dimension=2,
        maximum_derivative_order=3,
        response_axis_labels=("x",),
        derivative_provenance="fixed_basis_complete",
    )
    occupation = PassosOccupationSpec.fermi_dirac(
        chemical_potential_ev=0.0,
        temperature_k=0.0,
    )
    return passos_kpoint_data_from_fixed_basis_derivatives(
        hamiltonian,
        energies,
        eigenvectors,
        derivatives,
        grid=grid,
        grid_index=grid_index,
        model_certificate=certificate,
        occupation_spec=occupation,
        fixed_basis_connection_is_zero=True,
    )


def test_passos_certificate_encodes_basis_connection_contract() -> None:
    fixed = PassosModelCertificate(
        model_id="fixed",
        source_id="fixed_source",
        basis_dimension=2,
        maximum_derivative_order=3,
        response_axis_labels=("x",),
        derivative_provenance="fixed_basis_complete",
    )
    covariant = PassosModelCertificate(
        model_id="moving",
        source_id="moving_source",
        basis_dimension=2,
        maximum_derivative_order=3,
        response_axis_labels=("x",),
        derivative_provenance="externally_certified_covariant",
    )
    assert fixed.basis_connection_contract == "zero_connection_k_independent_orthonormal"
    assert covariant.basis_connection_contract == "externally_supplied_covariant_tower"

    grid = PassosPrimitiveBZGrid(
        reciprocal_vectors=np.array([[2.0 * np.pi]]),
        mesh_shape=(1,),
        coordinate_length_unit_m=1.0e-9,
        grid_label="basis_connection_contract",
    )
    occupation = PassosOccupationSpec.fermi_dirac(
        chemical_potential_ev=0.0,
        temperature_k=0.0,
    )
    with pytest.raises(ValueError, match="explicit zero basis connection"):
        passos_kpoint_data_from_fixed_basis_derivatives(
            np.diag([-1.0, 1.0]).astype(complex),
            np.array([-1.0, 1.0]),
            np.eye(2, dtype=complex),
            _one_axis_tower(3),
            grid=grid,
            grid_index=0,
            model_certificate=fixed,
            occupation_spec=occupation,
            fixed_basis_connection_is_zero=False,  # type: ignore[arg-type]
        )


def test_passos_generic_shg_dispatch_matches_direct_route() -> None:
    parameters = RiceMeleParameters()
    photon = np.array([0.4, 0.6])
    direct_grid = rice_mele_grid(
        12,
        lattice_constant_m=parameters.lattice_constant_m,
    )
    generic_grid = rice_mele_grid(
        12,
        lattice_constant_m=parameters.lattice_constant_m,
    )
    direct = passos_shg_conductivity_from_kpoint_data(
        photon,
        iter_rice_mele_kpoints(direct_grid, parameters=parameters),
        adiabatic_gamma_ev=0.02,
    )
    generic = optical_response_from_kpoint_data(
        "second-harmonic",
        photon,
        iter_rice_mele_kpoints(generic_grid, parameters=parameters),
        adiabatic_gamma_ev=0.02,
    )
    assert isinstance(generic, PassosHarmonicConductivityResult)
    assert generic.order == 2
    np.testing.assert_allclose(generic.conductivity, direct.conductivity)
    assert canonical_response_kind("second-harmonic") == "second_harmonic_generation"
    assert canonical_response_kind("THG") == "third_harmonic_generation"

    graphene_parameters = Passos2018GrapheneParameters(spin_copies=1)
    graphene_grid = passos2018_graphene_grid(
        (2, 2),
        lattice_constant_m=graphene_parameters.lattice_constant_m,
    )
    thg = optical_response_from_kpoint_data(
        "third-harmonic",
        np.array([0.08]),
        iter_passos2018_graphene_kpoints(
            graphene_grid,
            parameters=graphene_parameters,
        ),
        adiabatic_gamma_ev=0.011,
    )
    assert isinstance(thg, PassosHarmonicConductivityResult)
    assert thg.order == 3
    assert np.all(np.isfinite(thg.conductivity))


def test_typed_harmonic_facade_matches_legacy_generic_route() -> None:
    parameters = RiceMeleParameters()
    photon = np.array([0.4, 0.6])
    formulation = PassosFiniteBandVelocityGauge(
        switching=IndependentInputAdiabaticSwitching(gamma_ev=0.02)
    )
    typed_grid = rice_mele_grid(12, lattice_constant_m=parameters.lattice_constant_m)
    generic_grid = rice_mele_grid(12, lattice_constant_m=parameters.lattice_constant_m)
    legacy_grid = rice_mele_grid(12, lattice_constant_m=parameters.lattice_constant_m)
    typed = harmonic_response_from_kpoint_data(
        "shg",
        photon,
        iter_rice_mele_kpoints(typed_grid, parameters=parameters),
        formulation=formulation,
    )
    generic = optical_response_from_kpoint_data(
        "shg",
        photon,
        iter_rice_mele_kpoints(generic_grid, parameters=parameters),
        formulation=formulation,
    )
    legacy = optical_response_from_kpoint_data(
        "shg",
        photon,
        iter_rice_mele_kpoints(legacy_grid, parameters=parameters),
        adiabatic_gamma_ev=0.02,
    )
    np.testing.assert_allclose(typed.conductivity, generic.conductivity)
    np.testing.assert_allclose(typed.conductivity, legacy.conductivity)
    assert typed.gauge_kind == "passos_finite_band_velocity_gauge"
    assert typed.causal_prescription_kind == "independent_input_adiabatic"
    assert typed.basis_connection_contract == "zero_connection_k_independent_orthonormal"


def test_passos_generic_dispatch_prevalidates_before_kpoint_consumption() -> None:
    photon = np.array([0.4])

    def exploding_kpoints():
        raise AssertionError("Passos k-points consumed before API validation")
        yield  # pragma: no cover

    with pytest.raises(TypeError, match="adiabatic_gamma_ev"):
        optical_response_from_kpoint_data("shg", photon, exploding_kpoints())
    formulation = PassosFiniteBandVelocityGauge(
        switching=IndependentInputAdiabaticSwitching(gamma_ev=0.02)
    )
    with pytest.raises(TypeError, match="real scalar, not bool"):
        IndependentInputAdiabaticSwitching(gamma_ev=True)
    with pytest.raises(TypeError, match="real scalar, not bool"):
        optical_response_from_kpoint_data(
            "shg",
            photon,
            exploding_kpoints(),
            adiabatic_gamma_ev=True,
        )
    with pytest.raises(ValueError, match="not both"):
        optical_response_from_kpoint_data(
            "shg",
            photon,
            exploding_kpoints(),
            formulation=formulation,
            adiabatic_gamma_ev=0.02,
        )
    with pytest.raises(ValueError, match="only for SHG/THG"):
        optical_response_from_kpoint_data(
            "linear",
            photon,
            exploding_kpoints(),
            formulation=formulation,
            si_prefactor=1.0,
        )
    with pytest.raises(TypeError, match="PassosFiniteBandVelocityGauge"):
        harmonic_response_from_kpoint_data(
            "shg",
            photon,
            exploding_kpoints(),
            formulation=object(),  # type: ignore[arg-type]
        )
    for invalid_photon in (
        np.array([], dtype=float),
        np.array([0.0]),
        np.array([-0.1]),
    ):
        with pytest.raises(ValueError, match="positive"):
            optical_response_from_kpoint_data(
                "shg",
                invalid_photon,
                exploding_kpoints(),
                adiabatic_gamma_ev=0.02,
            )
    with pytest.raises(ValueError, match="SI prefactor"):
        optical_response_from_kpoint_data(
            "shg",
            photon,
            exploding_kpoints(),
            adiabatic_gamma_ev=0.02,
            si_prefactor=1.0,
        )
    with pytest.raises(ValueError, match="selected_bands"):
        optical_response_from_kpoint_data(
            "shg",
            photon,
            exploding_kpoints(),
            adiabatic_gamma_ev=0.02,
            selected_bands=[0, 1],
        )
    with pytest.raises(ValueError, match="not eta_ev"):
        optical_response_from_kpoint_data(
            "shg",
            photon,
            exploding_kpoints(),
            adiabatic_gamma_ev=0.02,
            eta_ev=0.001,
        )
    with pytest.raises(ValueError, match="full-BZ"):
        optical_response_from_kpoint_data(
            "shg",
            photon,
            exploding_kpoints(),
            adiabatic_gamma_ev=0.02,
            include_bz_factor=True,
        )
    with pytest.raises(ValueError, match="denominator_cutoff"):
        optical_response_from_kpoint_data(
            "shg",
            photon,
            exploding_kpoints(),
            adiabatic_gamma_ev=0.02,
            denominator_cutoff_ev=1.0e-12,
        )
    with pytest.raises(ValueError, match="geometry"):
        optical_response_from_kpoint_data(
            "shg",
            photon,
            exploding_kpoints(),
            adiabatic_gamma_ev=0.02,
            geometry=object(),
        )
    with pytest.raises(TypeError, match="PassosKPointData"):
        optical_response_from_kpoint_data(
            "shg",
            photon,
            [object()],
            adiabatic_gamma_ev=0.02,
        )


def test_passos_grid_and_full_bz_contract() -> None:
    grid = PassosPrimitiveBZGrid(
        reciprocal_vectors=np.array([[2.0 * np.pi]]),
        mesh_shape=(2,),
        coordinate_length_unit_m=1.0e-9,
        grid_label="contract_grid",
    )
    np.testing.assert_allclose(grid.cartesian_point(0), [0.5 * np.pi])
    np.testing.assert_allclose(grid.cartesian_point(1), [1.5 * np.pi])
    assert grid.point_weight == pytest.approx(np.pi)
    point0 = _contract_point(grid, 0, scale=1.0)
    point1 = _contract_point(grid, 1, scale=1.1)
    result = passos_shg_conductivity_from_kpoint_data(
        np.array([0.3]),
        (point for point in (point0, point1)),
        adiabatic_gamma_ev=0.02,
    )
    assert result.kpoint_count == 2
    assert result.conductivity.flags.writeable is False
    np.testing.assert_allclose(
        np.sum(result.eq32_sector_conductivity, axis=0),
        result.conductivity,
    )
    assert np.all(
        result.absolute_contribution_sum
        + 1.0e-12 * np.maximum(result.absolute_contribution_sum, 1.0e-300)
        >= np.abs(result.conductivity)
    )
    with pytest.raises(ValueError, match="duplicate"):
        passos_shg_conductivity_from_kpoint_data(
            np.array([0.3]),
            [point0, point0],
            adiabatic_gamma_ev=0.02,
        )
    with pytest.raises(ValueError, match="increasing grid_index"):
        passos_shg_conductivity_from_kpoint_data(
            np.array([0.3]),
            [point1, point0],
            adiabatic_gamma_ev=0.02,
        )
    with pytest.raises(ValueError, match="missing"):
        passos_shg_conductivity_from_kpoint_data(
            np.array([0.3]),
            [point0],
            adiabatic_gamma_ev=0.02,
        )


def test_passos_shg_sheet_and_effective_bulk_susceptibility_units() -> None:
    grid = PassosPrimitiveBZGrid(
        reciprocal_vectors=2.0 * np.pi * np.eye(2),
        mesh_shape=(1, 1),
        coordinate_length_unit_m=1.0e-9,
        grid_label="shg_susceptibility_2d",
    )
    point = _contract_point(grid, 0, scale=1.0)
    response = passos_shg_conductivity_from_kpoint_data(
        np.array([0.3]),
        [point],
        adiabatic_gamma_ev=0.02,
    )
    thickness = 0.7e-9
    susceptibility = passos_shg_susceptibility(
        response,
        effective_thickness_m=thickness,
    )
    facade_susceptibility = shg_susceptibility_from_response(
        response,
        effective_thickness_m=thickness,
    )
    np.testing.assert_allclose(
        facade_susceptibility.sheet_susceptibility_m2_per_v,
        susceptibility.sheet_susceptibility_m2_per_v,
    )
    np.testing.assert_allclose(
        facade_susceptibility.effective_bulk_susceptibility_m_per_v,
        susceptibility.effective_bulk_susceptibility_m_per_v,
    )
    angular_frequency = response.photon_energies_ev[0] / 6.582119569e-16
    expected_sheet = (
        1.0j
        * response.conductivity
        / (2.0 * 8.8541878128e-12 * angular_frequency)
    )
    np.testing.assert_allclose(
        susceptibility.sheet_susceptibility_m2_per_v,
        expected_sheet,
    )
    np.testing.assert_allclose(
        susceptibility.effective_bulk_susceptibility_m_per_v,
        expected_sheet / thickness,
    )
    assert susceptibility.sheet_susceptibility_m2_per_v.flags.writeable is False


def test_passos_refined_bz_accumulator_matches_manual_weighted_oracle() -> None:
    grid = PassosPrimitiveBZGrid(
        reciprocal_vectors=np.array([[2.0 * np.pi]]),
        mesh_shape=(2,),
        coordinate_length_unit_m=1.0e-9,
        grid_label="manual_refined_weight_oracle",
        refined_cell_indices=(0,),
        refinement_factor=2,
    )
    points = [
        _contract_point(grid, index, scale=1.0 + 0.1 * index)
        for index in range(grid.number_of_points)
    ]
    photon = np.array([0.3])
    gamma = 0.02
    manual = np.zeros((1, 1, 1, 1), dtype=complex)
    for point in points:
        manual += (
            point.weight
            / (2.0 * np.pi) ** grid.integration_dimension
            * passos_harmonic_kernel(
                photon,
                point.energies_ev,
                point.occupations,
                point.covariant_derivatives_ev,
                order=2,
                adiabatic_gamma_ev=gamma,
            )
        )
    manual *= passos_si_prefactor(
        2,
        integration_dimension=grid.integration_dimension,
        length_unit_m=grid.coordinate_length_unit_m,
    )
    accumulated = passos_shg_conductivity_from_kpoint_data(
        photon,
        points,
        adiabatic_gamma_ev=gamma,
    )
    np.testing.assert_allclose(
        accumulated.conductivity,
        manual,
        rtol=2.0e-13,
        atol=1.0e-30,
    )


def test_passos_refined_grid_replaces_cells_and_preserves_measure() -> None:
    grid = passos2018_graphene_refined_grid(
        24,
        patch_half_width_cells=1,
        refinement_factor=2,
        lattice_constant_m=0.246e-9,
    )
    assert len(grid.refined_cell_indices) == 8
    assert grid.number_of_points == 24**2 - 8 + 8 * 2**2
    weights = np.array(
        [grid.point_weight_for_index(index) for index in range(grid.number_of_points)]
    )
    assert np.sum(weights) == pytest.approx(grid.full_bz_measure)
    assert np.count_nonzero(weights < grid.base_cell_weight) == 8 * 2**2
    with pytest.raises(ValueError, match="index-dependent"):
        _ = grid.point_weight


def test_passos_primitive_grid_rejects_boolean_mesh() -> None:
    with pytest.raises(TypeError):
        PassosPrimitiveBZGrid(
            reciprocal_vectors=np.array([[2.0 * np.pi]]),
            mesh_shape=(True,),
            coordinate_length_unit_m=1.0,
            grid_label="bad",
        )
    with pytest.raises(TypeError):
        rice_mele_grid(True, lattice_constant_m=1.0)
    with pytest.raises(TypeError):
        Passos2018GrapheneParameters(spin_copies=True)


def test_passos2018_graphene_geometry_and_derivative_identities() -> None:
    direct = graphene_direct_lattice_vectors()
    reciprocal = graphene_reciprocal_vectors()
    np.testing.assert_allclose(
        direct @ reciprocal.T,
        2.0 * np.pi * np.eye(2),
        atol=1.0e-14,
    )
    assert abs(np.linalg.det(reciprocal)) == pytest.approx(
        8.0 * np.pi**2 / np.sqrt(3.0)
    )
    parameters = Passos2018GrapheneParameters(spin_copies=1)
    gamma_h, _ = passos2018_graphene_hamiltonian_and_derivatives(
        np.zeros(2),
        parameters=parameters,
        maximum_order=4,
    )
    assert gamma_h[0, 1] == pytest.approx(3.0 * parameters.hopping_ev)
    k_point = np.array([1.0 / 3.0, 2.0 / 3.0]) @ reciprocal
    k_h, _ = passos2018_graphene_hamiltonian_and_derivatives(
        k_point,
        parameters=parameters,
        maximum_order=4,
    )
    assert abs(k_h[0, 1]) < 1.0e-13
    evaluation_point = np.array([0.41, 0.23])
    hamiltonian, derivatives = passos2018_graphene_hamiltonian_and_derivatives(
        evaluation_point,
        parameters=parameters,
        maximum_order=4,
    )
    assert hamiltonian.shape == (2, 2)
    np.testing.assert_allclose(derivatives[2][0, 0, 0], -0.25 * derivatives[0][0])
    np.testing.assert_allclose(
        derivatives[3][0, 0, 0, 0],
        -0.25 * derivatives[1][0, 0],
    )
    step = 1.0e-5
    h_plus, _ = passos2018_graphene_hamiltonian_and_derivatives(
        evaluation_point + np.array([step, 0.0]),
        parameters=parameters,
        maximum_order=1,
    )
    h_minus, _ = passos2018_graphene_hamiltonian_and_derivatives(
        evaluation_point - np.array([step, 0.0]),
        parameters=parameters,
        maximum_order=1,
    )
    np.testing.assert_allclose(
        (h_plus - h_minus) / (2.0 * step),
        derivatives[0][0],
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    paper_parameters = Passos2018GrapheneParameters(spin_copies=2)
    expected_sigma0 = (
        3.0
        * E2_OVER_HBAR_S
        * paper_parameters.lattice_constant_m**2
        * paper_parameters.hopping_ev**2
        / (16.0 * np.pi * paper_parameters.chemical_potential_ev**4)
    )
    assert passos2018_sigma0_si(paper_parameters) == pytest.approx(expected_sigma0)
    with pytest.raises(ValueError, match="spin_copies=2"):
        passos2018_sigma0_si(parameters)


def test_passos_benchmark_iterators_reject_incompatible_grids() -> None:
    graphene_parameters = Passos2018GrapheneParameters()
    wrong_graphene_grid = rice_mele_grid(
        4,
        lattice_constant_m=graphene_parameters.lattice_constant_m,
    )
    with pytest.raises(ValueError, match="two-dimensional"):
        next(
            iter_passos2018_graphene_kpoints(
                wrong_graphene_grid,
                parameters=graphene_parameters,
            )
        )
    rice_parameters = RiceMeleParameters()
    wrong_rice_grid = passos2018_graphene_grid(
        (2, 2),
        lattice_constant_m=rice_parameters.lattice_constant_m,
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        next(iter_rice_mele_kpoints(wrong_rice_grid, parameters=rice_parameters))


def test_passos_graphene_spin_copies_double_thg() -> None:
    photon = np.array([0.08])
    one_spin = Passos2018GrapheneParameters(spin_copies=1)
    two_spin = replace(one_spin, spin_copies=2)
    grid_one = passos2018_graphene_grid(
        (4, 4),
        lattice_constant_m=one_spin.lattice_constant_m,
    )
    grid_two = passos2018_graphene_grid(
        (4, 4),
        lattice_constant_m=two_spin.lattice_constant_m,
    )
    sigma_one = passos_thg_conductivity_from_kpoint_data(
        photon,
        iter_passos2018_graphene_kpoints(grid_one, parameters=one_spin),
        adiabatic_gamma_ev=0.011,
    ).conductivity
    sigma_two = passos_thg_conductivity_from_kpoint_data(
        photon,
        iter_passos2018_graphene_kpoints(grid_two, parameters=two_spin),
        adiabatic_gamma_ev=0.011,
    ).conductivity
    assert abs(sigma_one[0, 0, 0, 0, 0]) > 1.0e-30
    np.testing.assert_allclose(sigma_two, 2.0 * sigma_one, rtol=2.0e-11, atol=1.0e-30)


def test_passos_graphene_is_covariant_under_spin_subspace_rotations() -> None:
    parameters = Passos2018GrapheneParameters(spin_copies=2)
    grid = passos2018_graphene_grid(
        (1, 1),
        lattice_constant_m=parameters.lattice_constant_m,
    )
    point = next(iter_passos2018_graphene_kpoints(grid, parameters=parameters))
    rng = np.random.default_rng(27)
    gauge = np.zeros((4, 4), dtype=complex)
    for start in (0, 2):
        random_matrix = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
        unitary, triangular = np.linalg.qr(random_matrix)
        phases = np.diag(triangular).copy()
        phases = phases / np.where(np.abs(phases) > 0.0, np.abs(phases), 1.0)
        gauge[start : start + 2, start : start + 2] = (
            unitary * phases.conjugate()[None, :]
        )
    transformed = tuple(
        np.einsum(
            "ia,...ij,jb->...ab",
            gauge.conjugate(),
            derivative,
            gauge,
            optimize=True,
        )
        for derivative in point.covariant_derivatives_ev
    )
    reference = passos_harmonic_kernel(
        np.array([0.08]),
        point.energies_ev,
        point.occupations,
        point.covariant_derivatives_ev,
        order=3,
        adiabatic_gamma_ev=0.011,
    )
    rotated = passos_harmonic_kernel(
        np.array([0.08]),
        point.energies_ev,
        point.occupations,
        transformed,
        order=3,
        adiabatic_gamma_ev=0.011,
    )
    np.testing.assert_allclose(rotated, reference, rtol=5.0e-12, atol=1.0e-12)


def test_rice_mele_shg_is_nonzero_and_inversion_forbidden() -> None:
    broken = RiceMeleParameters(staggered_potential_ev=0.4)
    inversion_symmetric = replace(broken, staggered_potential_ev=0.0)
    grid_broken = rice_mele_grid(
        80,
        lattice_constant_m=broken.lattice_constant_m,
    )
    grid_symmetric = rice_mele_grid(
        80,
        lattice_constant_m=inversion_symmetric.lattice_constant_m,
    )
    photon = np.array([0.55])
    broken_result = passos_shg_conductivity_from_kpoint_data(
        photon,
        iter_rice_mele_kpoints(grid_broken, parameters=broken),
        adiabatic_gamma_ev=0.02,
    )
    symmetric_result = passos_shg_conductivity_from_kpoint_data(
        photon,
        iter_rice_mele_kpoints(grid_symmetric, parameters=inversion_symmetric),
        adiabatic_gamma_ev=0.02,
    )
    k_value = 0.73
    h_k, derivatives_k = rice_mele_hamiltonian_and_derivatives(
        k_value,
        parameters=inversion_symmetric,
    )
    h_minus_k, derivatives_minus_k = rice_mele_hamiltonian_and_derivatives(
        -k_value,
        parameters=inversion_symmetric,
    )
    inversion = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    np.testing.assert_allclose(inversion @ h_k @ inversion, h_minus_k)
    for order, (derivative_k, derivative_minus_k) in enumerate(
        zip(derivatives_k, derivatives_minus_k),
        start=1,
    ):
        np.testing.assert_allclose(
            inversion @ derivative_k.reshape(2, 2) @ inversion,
            (-1.0) ** order * derivative_minus_k.reshape(2, 2),
        )
    broken_value = broken_result.conductivity[0, 0, 0, 0]
    symmetric_value = symmetric_result.conductivity[0, 0, 0, 0]
    broken_absolute = broken_result.absolute_contribution_sum[0, 0, 0, 0]
    symmetric_absolute = symmetric_result.absolute_contribution_sum[0, 0, 0, 0]
    assert broken_absolute > 0.0
    assert symmetric_absolute > 0.0
    assert abs(broken_value) > 1.0e-8 * broken_absolute
    assert abs(symmetric_value) < 1.0e-10 * abs(broken_value)
    assert abs(symmetric_value) / symmetric_absolute < 1.0e-12
