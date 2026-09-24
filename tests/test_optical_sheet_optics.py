from __future__ import annotations

import numpy as np
import pytest

from analysis.optical import (
    KerrFaradayResponse,
    LinearConductivityResult,
    NormalIncidenceSameSideGeometry,
    Okada2016MixedPulseGeometry,
    OpticalKPointData,
    VACUUM_IMPEDANCE_OHM,
    complex_polarization_from_jones,
    faraday_kerr_from_sheet_conductivity_on_substrate,
    kerr_faraday_from_kpoint_data,
    kerr_faraday_from_kpoint_data_on_substrate,
    kerr_faraday_from_linear_conductivity,
    linear_conductivity_from_kpoint_data,
    optical_response_from_kpoint_data,
    sheet_scattering_at_normal_incidence,
    shift_current_velocity_gauge_from_kpoint_data,
    shg_conductivity_from_kpoint_data,
    unwrap_polarization_angle,
)


def _two_band_kpoint() -> OpticalKPointData:
    energies = np.asarray([-0.3, 0.5])
    occupations = np.asarray([1.0, 0.0])
    velocity = np.zeros((2, 2, 2), dtype=np.complex128)
    velocity[0, 0, 1] = 0.7 + 0.1j
    velocity[0, 1, 0] = np.conjugate(velocity[0, 0, 1])
    velocity[1, 0, 1] = -0.2 + 0.4j
    velocity[1, 1, 0] = np.conjugate(velocity[1, 0, 1])
    return OpticalKPointData(
        energies_ev=energies,
        velocity_h=velocity,
        occupations=occupations,
        weight=0.5,
    )


def test_bare_two_medium_interface_recovers_fresnel_amplitudes():
    n1 = 1.7
    n2 = 2.3
    scattering = sheet_scattering_at_normal_incidence(
        np.zeros((2, 2), dtype=np.complex128),
        incident_refractive_index=n1,
        transmitted_refractive_index=n2,
    )
    expected_t = 2.0 * n1 / (n1 + n2)
    expected_r = (n1 - n2) / (n1 + n2)
    np.testing.assert_allclose(scattering.transmission_matrix, expected_t * np.eye(2))
    np.testing.assert_allclose(scattering.reflection_matrix, expected_r * np.eye(2))


def test_scalar_sheet_recovers_analytic_interface_formula():
    n1 = 1.4
    n2 = 2.1
    sigma_scalar = 2.7e-4 + 1.1e-4j
    sigma = sigma_scalar * np.eye(2, dtype=np.complex128)
    scattering = sheet_scattering_at_normal_incidence(
        sigma,
        incident_refractive_index=n1,
        transmitted_refractive_index=n2,
    )
    y1 = n1 / VACUUM_IMPEDANCE_OHM
    y2 = n2 / VACUUM_IMPEDANCE_OHM
    expected_t = 2.0 * y1 / (y1 + y2 + sigma_scalar)
    expected_r = (y1 - y2 - sigma_scalar) / (y1 + y2 + sigma_scalar)
    np.testing.assert_allclose(scattering.transmission_matrix, expected_t * np.eye(2))
    np.testing.assert_allclose(scattering.reflection_matrix, expected_r * np.eye(2))


def test_sheet_scattering_satisfies_matrix_boundary_equations():
    sigma = np.asarray(
        [[2.0e-4 + 0.7e-4j, 1.3e-4 - 0.2e-4j],
         [-0.9e-4 + 0.4e-4j, 1.6e-4 - 0.3e-4j]],
        dtype=np.complex128,
    )
    n1 = 1.25 + 0.03j
    n2 = 2.4 + 0.08j
    scattering = sheet_scattering_at_normal_incidence(
        sigma,
        incident_refractive_index=n1,
        transmitted_refractive_index=n2,
    )
    y1 = n1 / VACUUM_IMPEDANCE_OHM
    y2 = n2 / VACUUM_IMPEDANCE_OHM
    boundary = (y1 + y2) * np.eye(2) + sigma
    np.testing.assert_allclose(
        boundary @ scattering.transmission_matrix,
        2.0 * y1 * np.eye(2),
        rtol=2.0e-14,
        atol=2.0e-18,
    )
    np.testing.assert_allclose(
        scattering.reflection_matrix,
        scattering.transmission_matrix - np.eye(2),
        rtol=0.0,
        atol=0.0,
    )


def test_hall_sheet_matches_independent_circular_eigenvalue_formula():
    n1 = 1.0
    n2 = 3.2
    longitudinal = 1.1e-4 - 0.2e-4j
    hall = 3.8e-5
    sigma = np.asarray(
        [[longitudinal, hall], [-hall, longitudinal]],
        dtype=np.complex128,
    )
    scattering = sheet_scattering_at_normal_incidence(
        sigma,
        incident_refractive_index=n1,
        transmitted_refractive_index=n2,
    )
    circular_plus = np.asarray([1.0, 1.0j]) / np.sqrt(2.0)
    expected = (
        2.0 * n1 / VACUUM_IMPEDANCE_OHM
        / ((n1 + n2) / VACUUM_IMPEDANCE_OHM + longitudinal + 1.0j * hall)
    )
    np.testing.assert_allclose(
        scattering.transmitted_field(circular_plus),
        expected * circular_plus,
        rtol=2.0e-14,
        atol=2.0e-18,
    )


def test_lossless_hall_sheet_conserves_normal_incidence_power():
    n1 = 1.3
    n2 = 2.6
    hall = 0.73 / VACUUM_IMPEDANCE_OHM
    sigma = np.asarray([[0.0, hall], [-hall, 0.0]], dtype=np.complex128)
    scattering = sheet_scattering_at_normal_incidence(
        sigma,
        incident_refractive_index=n1,
        transmitted_refractive_index=n2,
    )
    reflected = scattering.reflected_x_incident
    transmitted = scattering.transmitted_x_incident
    reflected_power = np.sum(np.abs(reflected) ** 2)
    transmitted_power = (n2 / n1) * np.sum(np.abs(transmitted) ** 2)
    np.testing.assert_allclose(reflected_power + transmitted_power, 1.0, atol=2.0e-15)


def test_sheet_scattering_broadcasts_media_and_conductivity_batches():
    sigma = np.zeros((2, 1, 2, 2), dtype=np.complex128)
    sigma[0, 0] = np.asarray([[1.0e-4, 2.0e-5], [-2.0e-5, 0.8e-4]])
    sigma[1, 0] = np.asarray([[2.0e-4, -3.0e-5], [3.0e-5, 1.5e-4]])
    n1 = np.asarray([[1.1], [1.2]])
    n2 = np.asarray([[1.5, 2.0, 3.0]])
    scattering = sheet_scattering_at_normal_incidence(
        sigma,
        incident_refractive_index=n1,
        transmitted_refractive_index=n2,
    )
    assert scattering.transmission_matrix.shape == (2, 3, 2, 2)
    assert scattering.reflection_matrix.shape == (2, 3, 2, 2)
    for i in range(2):
        for j in range(3):
            scalar = sheet_scattering_at_normal_incidence(
                sigma[i, 0],
                incident_refractive_index=n1[i, 0],
                transmitted_refractive_index=n2[0, j],
            )
            np.testing.assert_allclose(
                scattering.transmission_matrix[i, j],
                scalar.transmission_matrix,
            )
            np.testing.assert_allclose(
                scattering.reflection_matrix[i, j],
                scalar.reflection_matrix,
            )


def test_scattering_applies_to_arbitrary_broadcast_jones_fields():
    sigma = np.zeros((3, 1, 2, 2), dtype=np.complex128)
    sigma[:, 0, 0, 1] = np.asarray([1.0e-5, 2.0e-5, 3.0e-5])
    sigma[:, 0, 1, 0] = -sigma[:, 0, 0, 1]
    scattering = sheet_scattering_at_normal_incidence(sigma)
    incident = np.asarray([1.0, 0.4j])
    np.testing.assert_allclose(
        scattering.transmitted_field(incident),
        np.einsum("...ab,b->...a", scattering.transmission_matrix, incident),
    )
    per_batch_incident = np.zeros((3, 1, 2), dtype=np.complex128)
    per_batch_incident[..., 0] = 1.0
    per_batch_incident[:, 0, 1] = np.asarray([0.1j, 0.2j, 0.3j])
    per_batch_transmitted = scattering.transmitted_field(per_batch_incident)
    for index in range(3):
        np.testing.assert_allclose(
            per_batch_transmitted[index, 0],
            scattering.transmission_matrix[index, 0] @ per_batch_incident[index, 0],
        )

    batched_incident = np.zeros((1, 4, 2), dtype=np.complex128)
    batched_incident[..., 0] = 1.0
    batched_incident[..., 1] = np.linspace(0.0, 0.3, 4) * 1.0j
    transmitted = scattering.transmitted_field(batched_incident)
    assert transmitted.shape == (3, 4, 2)
    np.testing.assert_allclose(
        transmitted,
        np.einsum(
            "...ab,...b->...a",
            np.broadcast_to(scattering.transmission_matrix, (3, 4, 2, 2)),
            np.broadcast_to(batched_incident, (3, 4, 2)),
        ),
    )
    with pytest.raises(ValueError, match="final shape"):
        scattering.reflected_field(np.ones(3))


def test_complex_polarization_is_scale_safe_and_exposes_defined_masks():
    theta = np.asarray([-0.8, 0.31])
    eta = np.asarray([-0.45, 0.27])
    ex = np.cos(theta) - 1.0j * eta * np.sin(theta)
    ey = np.sin(theta) + 1.0j * eta * np.cos(theta)
    scales = np.asarray([1.0e300 * np.exp(0.4j), 1.0e-300 * np.exp(-0.8j)])
    polarization = complex_polarization_from_jones(ex * scales, ey * scales)
    np.testing.assert_allclose(polarization.rotation_angle_rad, theta, atol=3.0e-15)
    np.testing.assert_allclose(polarization.ellipticity, eta, atol=3.0e-15)
    assert np.all(polarization.polarization_defined)
    assert np.all(polarization.orientation_defined)

    circular = complex_polarization_from_jones(
        np.asarray([1.0, 1.0]),
        np.asarray([1.0j, -1.0j]),
    )
    np.testing.assert_allclose(circular.ellipticity, np.asarray([1.0, -1.0]))
    assert np.all(circular.polarization_defined)
    assert not np.any(circular.orientation_defined)
    assert np.all(np.isnan(circular.rotation_angle_rad))

    undefined = complex_polarization_from_jones(0.0, 0.0, zero_policy="nan")
    assert not bool(undefined.polarization_defined)
    assert not bool(undefined.orientation_defined)
    assert np.isnan(undefined.rotation_angle_rad)
    assert np.isnan(undefined.ellipticity)


def test_unwrap_polarization_angle_recovers_modulo_pi_continuity():
    truth = np.linspace(-1.3, 5.2, 180)
    principal = (truth + np.pi / 2.0) % np.pi - np.pi / 2.0
    recovered = unwrap_polarization_angle(principal)
    recovered += truth[0] - recovered[0]
    np.testing.assert_allclose(recovered, truth, atol=2.0e-14)

    stacked = np.stack((principal, principal + 0.2), axis=0)
    recovered_axis = unwrap_polarization_angle(stacked, axis=1)
    np.testing.assert_allclose(np.diff(recovered_axis, axis=1), np.diff(np.stack((truth, truth + 0.2))), atol=2.0e-14)


def test_unwrap_polarization_angle_does_not_cross_invalid_gaps():
    truth = np.linspace(-1.2, 3.8, 100)
    principal = (truth + np.pi / 2.0) % np.pi - np.pi / 2.0
    valid = np.ones(100, dtype=bool)
    valid[40:52] = False
    unwrapped = unwrap_polarization_angle(principal, valid=valid)
    assert np.all(np.isnan(unwrapped[40:52]))
    assert np.all(np.abs(np.diff(unwrapped[:40])) < 0.1)
    assert np.all(np.abs(np.diff(unwrapped[52:])) < 0.1)


def test_high_level_response_passes_two_medium_geometry_and_polarization():
    omega = np.asarray([0.01, 0.02, 0.03])
    sigma = np.zeros((3, 2, 2), dtype=np.complex128)
    sigma[:, 0, 0] = 1.0e-4
    sigma[:, 1, 1] = 1.0e-4
    sigma[:, 0, 1] = np.asarray([1.0e-5, 2.0e-5, 3.0e-5])
    sigma[:, 1, 0] = -sigma[:, 0, 1]
    linear = LinearConductivityResult(
        photon_energies_ev=omega,
        conductivity=sigma,
        skipped_small_denominators=0,
    )
    response = kerr_faraday_from_linear_conductivity(
        linear,
        incident_refractive_index=1.3,
        transmitted_refractive_index=2.4,
    )
    direct = sheet_scattering_at_normal_incidence(
        sigma,
        incident_refractive_index=1.3,
        transmitted_refractive_index=2.4,
    )
    assert response.geometry == "normal_incidence_same_side"
    np.testing.assert_allclose(
        response.angles.transmission_scattering.transmission_matrix,
        direct.transmission_matrix,
    )
    np.testing.assert_allclose(
        response.faraday_ellipticity,
        response.faraday_polarization.ellipticity,
    )
    np.testing.assert_allclose(
        np.exp(2.0j * response.faraday_angle_rad),
        np.exp(2.0j * response.faraday_polarization.rotation_angle_rad),
    )
    assert response.tensor_convention == "physical_j_equals_sigma_e"
    with pytest.raises(ValueError, match="conductivity shape"):
        kerr_faraday_from_linear_conductivity(
            LinearConductivityResult(
                photon_energies_ev=omega,
                conductivity=np.zeros((omega.size, 3, 3), dtype=np.complex128),
                skipped_small_denominators=0,
            )
        )


def test_linear_result_is_copied_read_only_and_high_level_media_are_pointwise():
    omega_source = np.asarray([0.01, 0.02, 0.03])
    sigma_source = np.zeros((3, 2, 2), dtype=np.complex128)
    sigma_source[:, 0, 0] = 1.0e-4
    sigma_source[:, 1, 1] = 1.0e-4
    linear = LinearConductivityResult(
        photon_energies_ev=omega_source,
        conductivity=sigma_source,
        skipped_small_denominators=0,
    )
    omega_source[:] = -1.0
    sigma_source[:] = 99.0
    np.testing.assert_allclose(linear.photon_energies_ev, [0.01, 0.02, 0.03])
    np.testing.assert_allclose(linear.conductivity[:, 0, 0], 1.0e-4)
    with pytest.raises(ValueError, match="read-only"):
        linear.conductivity[0, 0, 0] = 0.0

    n1 = np.asarray([1.0, 1.1, 1.2])
    n2 = np.asarray([2.0, 2.1, 2.2])
    response = kerr_faraday_from_linear_conductivity(
        linear,
        incident_refractive_index=n1,
        transmitted_refractive_index=n2,
    )
    assert response.faraday_angle_rad.shape == omega_source.shape
    np.testing.assert_allclose(
        response.angles.transmission_scattering.incident_refractive_index,
        n1,
    )
    with pytest.raises(ValueError, match="scalar or have the pointwise"):
        kerr_faraday_from_linear_conductivity(
            linear,
            transmitted_refractive_index=np.ones((1, 3)),
        )


def test_okada_mixed_geometry_records_both_sample_propagation_directions():
    sigma = np.asarray([[1.0e-4, 2.0e-5], [-2.0e-5, 1.0e-4]])
    amplitudes = faraday_kerr_from_sheet_conductivity_on_substrate(
        sigma,
        substrate_refractive_index=3.47,
    )
    assert (
        amplitudes.transmission_scattering.incident_propagation_direction_in_sample
        == 1
    )
    assert (
        amplitudes.reflection_scattering.incident_propagation_direction_in_sample
        == -1
    )


def test_generic_kpoint_dispatch_matches_direct_linear_and_kerr_routes():
    omega = np.asarray([0.2, 0.4])
    point = _two_band_kpoint()
    prefactor = 1.0j

    direct_linear = linear_conductivity_from_kpoint_data(
        omega,
        [point, point],
        prefactor=prefactor,
    )
    generic_linear = optical_response_from_kpoint_data(
        "linear",
        omega,
        [point, point],
        si_prefactor=prefactor,
    )
    assert isinstance(generic_linear, LinearConductivityResult)
    np.testing.assert_allclose(
        generic_linear.conductivity,
        direct_linear.conductivity,
    )

    direct_vacuum = kerr_faraday_from_kpoint_data(
        omega,
        [point, point],
        prefactor=prefactor,
    )
    generic_vacuum = optical_response_from_kpoint_data(
        "kerr",
        omega,
        [point, point],
        si_prefactor=prefactor,
    )
    assert isinstance(generic_vacuum, KerrFaradayResponse)
    np.testing.assert_allclose(
        generic_vacuum.conductivity,
        direct_vacuum.conductivity,
    )
    np.testing.assert_allclose(
        generic_vacuum.kerr_angle_rad,
        direct_vacuum.kerr_angle_rad,
        equal_nan=True,
    )
    for alias in ("faraday", "kerr_faraday", "kerr-faraday"):
        alias_response = optical_response_from_kpoint_data(
            alias,
            omega,
            [point, point],
            si_prefactor=prefactor,
        )
        np.testing.assert_allclose(
            alias_response.kerr_angle_rad,
            generic_vacuum.kerr_angle_rad,
            equal_nan=True,
        )

    same_side = NormalIncidenceSameSideGeometry(
        incident_refractive_index=np.asarray([1.1, 1.2]),
        transmitted_refractive_index=np.asarray([2.1, 2.2]),
    )
    generic_same_side = optical_response_from_kpoint_data(
        "faraday",
        omega,
        [point, point],
        si_prefactor=prefactor,
        geometry=same_side,
    )
    direct_same_side = kerr_faraday_from_kpoint_data(
        omega,
        [point, point],
        prefactor=prefactor,
        incident_refractive_index=same_side.incident_refractive_index,
        transmitted_refractive_index=same_side.transmitted_refractive_index,
    )
    np.testing.assert_allclose(
        generic_same_side.faraday_angle_rad,
        direct_same_side.faraday_angle_rad,
        equal_nan=True,
    )

    okada = Okada2016MixedPulseGeometry(substrate_refractive_index=3.47)
    generic_okada = optical_response_from_kpoint_data(
        "kerr-faraday",
        omega,
        [point, point],
        si_prefactor=prefactor,
        geometry=okada,
    )
    direct_okada = kerr_faraday_from_kpoint_data_on_substrate(
        omega,
        [point, point],
        substrate_refractive_index=3.47,
        prefactor=prefactor,
    )
    np.testing.assert_allclose(
        generic_okada.kerr_angle_rad,
        direct_okada.kerr_angle_rad,
        equal_nan=True,
    )
    assert generic_okada.geometry == "okada_2016_mixed_pulse_substrate"
    assert (
        generic_okada.angles.transmission_scattering
        .incident_propagation_direction_in_sample
        == 1
    )
    assert (
        generic_okada.angles.reflection_scattering
        .incident_propagation_direction_in_sample
        == -1
    )


def test_generic_kpoint_dispatch_validates_before_consuming_kpoints():
    omega = np.asarray([0.2])

    def exploding_kpoints():
        raise AssertionError("k-points were consumed before API validation")
        yield  # pragma: no cover

    with pytest.raises(TypeError):
        optical_response_from_kpoint_data("kerr", omega, exploding_kpoints())
    with pytest.raises(ValueError, match="finite complex scalar"):
        optical_response_from_kpoint_data(
            "kerr",
            omega,
            exploding_kpoints(),
            si_prefactor=np.nan,
        )
    with pytest.raises(ValueError, match="transition-table generic API"):
        optical_response_from_kpoint_data(
            "shift_current",
            omega,
            exploding_kpoints(),
            si_prefactor=1.0j,
        )
    with pytest.raises(ValueError, match="valid only for a Kerr"):
        optical_response_from_kpoint_data(
            "linear",
            omega,
            exploding_kpoints(),
            si_prefactor=1.0j,
            geometry=NormalIncidenceSameSideGeometry(),
        )
    with pytest.raises(TypeError, match="geometry must be"):
        optical_response_from_kpoint_data(
            "kerr",
            omega,
            exploding_kpoints(),
            si_prefactor=1.0j,
            geometry=object(),
        )
    with pytest.raises(ValueError, match="pointwise frequency shape"):
        optical_response_from_kpoint_data(
            "kerr",
            omega,
            exploding_kpoints(),
            si_prefactor=1.0j,
            geometry=NormalIncidenceSameSideGeometry(
                transmitted_refractive_index=np.ones(2),
            ),
        )
    with pytest.raises(ValueError, match="pointwise frequency shape"):
        optical_response_from_kpoint_data(
            "kerr",
            omega,
            exploding_kpoints(),
            si_prefactor=1.0j,
            geometry=Okada2016MixedPulseGeometry(
                substrate_refractive_index=np.ones(2),
            ),
        )
    with pytest.raises(ValueError, match="finite one-dimensional"):
        optical_response_from_kpoint_data(
            "linear",
            np.ones((1, 1)),
            exploding_kpoints(),
            si_prefactor=1.0j,
        )


def test_selected_band_generator_is_reused_for_every_kpoint():
    omega = np.asarray([0.2, 0.4])
    energies = np.asarray([-0.3, 0.5])
    occupations = np.asarray([1.0, 0.0])
    velocity = np.zeros((2, 2, 2), dtype=np.complex128)
    velocity[0, 0, 1] = 0.7 + 0.1j
    velocity[0, 1, 0] = np.conjugate(velocity[0, 0, 1])
    velocity[1, 0, 1] = -0.2 + 0.4j
    velocity[1, 1, 0] = np.conjugate(velocity[1, 0, 1])
    point = OpticalKPointData(
        energies_ev=energies,
        velocity_h=velocity,
        occupations=occupations,
        weight=0.5,
    )
    generator_result = linear_conductivity_from_kpoint_data(
        omega,
        [point, point],
        selected_bands=(index for index in (0, 1)),
    )
    tuple_result = linear_conductivity_from_kpoint_data(
        omega,
        [point, point],
        selected_bands=(0, 1),
    )
    np.testing.assert_allclose(generator_result.conductivity, tuple_result.conductivity)

    shift_generator = shift_current_velocity_gauge_from_kpoint_data(
        omega,
        [point, point],
        selected_bands=(index for index in (0, 1)),
    )
    shift_tuple = shift_current_velocity_gauge_from_kpoint_data(
        omega,
        [point, point],
        selected_bands=(0, 1),
    )
    np.testing.assert_allclose(shift_generator.conductivity, shift_tuple.conductivity)

    shg_generator = shg_conductivity_from_kpoint_data(
        omega,
        [point, point],
        selected_bands=(index for index in (0, 1)),
    )
    shg_tuple = shg_conductivity_from_kpoint_data(
        omega,
        [point, point],
        selected_bands=(0, 1),
    )
    np.testing.assert_allclose(shg_generator.conductivity, shg_tuple.conductivity)

    kerr_generator = kerr_faraday_from_kpoint_data(
        omega,
        [point, point],
        prefactor=1.0j,
        selected_bands=(index for index in (0, 1)),
    )
    kerr_tuple = kerr_faraday_from_kpoint_data(
        omega,
        [point, point],
        prefactor=1.0j,
        selected_bands=(0, 1),
    )
    np.testing.assert_allclose(kerr_generator.conductivity, kerr_tuple.conductivity)
    np.testing.assert_allclose(
        kerr_generator.faraday_angle_rad,
        kerr_tuple.faraday_angle_rad,
        equal_nan=True,
    )

    with pytest.raises(ValueError, match="duplicates"):
        linear_conductivity_from_kpoint_data(
            omega,
            [point],
            selected_bands=(0, 0),
        )
    with pytest.raises(TypeError, match="integer index"):
        linear_conductivity_from_kpoint_data(
            omega,
            [point],
            selected_bands=(0, 1.0),
        )
    with pytest.raises(TypeError, match="not bool"):
        linear_conductivity_from_kpoint_data(
            omega,
            [point],
            selected_bands=(0, True),
        )
