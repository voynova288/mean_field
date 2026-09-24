from __future__ import annotations

import numpy as np
import pytest

from analysis.optical import (
    E2_OVER_H_S,
    E2_OVER_HBAR_S,
    VACUUM_IMPEDANCE_OHM,
    VACUUM_PERMITTIVITY_F_PER_M,
    Okada2016MixedPulseGeometry,
    OpticalKPointData,
    complex_polarization_from_jones,
    faraday_kerr_from_sheet_conductivity,
    faraday_kerr_from_sheet_conductivity_liu_dai_2020_printed,
    faraday_kerr_from_sheet_conductivity_on_substrate,
    faraday_kerr_small_conductivity,
    kerr_faraday_from_kpoint_data,
    kerr_faraday_from_kpoint_data_on_substrate,
    linear_conductivity_from_kpoint_data,
    linear_conductivity_tensor_from_gauge_data,
    optical_response_from_kpoint_data,
    LIU_DAI_2020_EQ4_PREFAC_A_M_PER_V2,
    LIU_DAI_2020_EQ4_PREFAC_UA_NM_PER_V2,
    LIU_DAI_2020_EQ4_SHG_PREFAC_S,
    VELOCITY_GAUGE_2D_PREFAC_A_M_PER_V2,
    VELOCITY_GAUGE_2D_PREFAC_UA_NM_PER_V2,
    shift_current_conductivity_velocity_gauge,
    shg_conductivity_velocity_gauge,
    shg_susceptibility_from_conductivity,
    sheet_conductivity_from_normalized_transmission_matrix,
    sheet_conductivity_from_scalar_normalized_transmission,
)
from analysis.optical.benchmarks.okada_2016 import (
    OKADA_2016_INP_REFRACTIVE_INDEX,
    okada2016_qah_benchmark,
    okada2016_scaling_from_dc_conductivity,
    okada2016_scaling_function,
    okada2016_scaling_uncertainty,
)
from analysis.optical.benchmarks.qwz import (
    qwz_lower_band_chern_dvector,
    qwz_optical_kpoint_data,
)
from analysis.shift_current import HBAR_EV_S


def _two_level_velocity():
    energies = np.asarray([-0.5, 0.5], dtype=float)
    occupations = np.asarray([1.0, 0.0], dtype=float)
    velocity = np.zeros((2, 2, 2), dtype=np.complex128)
    velocity[0, 0, 1] = 1.0
    velocity[0, 1, 0] = 1.0
    velocity[1, 0, 1] = -1.0j
    velocity[1, 1, 0] = 1.0j
    return energies, occupations, velocity


def _bruteforce_linear_conductivity(omega, energies, velocity, occupations, *, k_weight, eta_ev, prefactor, include_bz_factor):
    ndim = velocity.shape[0]
    factor = complex(prefactor) * float(k_weight)
    if include_bz_factor:
        factor /= (2.0 * np.pi) ** ndim
    out = np.zeros((len(omega), ndim, ndim), dtype=np.complex128)
    for iw, photon_energy in enumerate(omega):
        for n in range(len(energies)):
            for m in range(len(energies)):
                if n == m:
                    continue
                static = energies[n] - energies[m]
                if abs(static) <= 1.0e-12:
                    continue
                occupation_diff = occupations[n] - occupations[m]
                if occupation_diff == 0.0:
                    continue
                resonance = occupation_diff / static / (static - photon_energy - 1.0j * eta_ev)
                for a in range(ndim):
                    for b in range(ndim):
                        out[iw, a, b] += velocity[a, m, n] * velocity[b, n, m] * resonance
    return factor * out


def _matrix_sheet_solution(sheet_conductivity):
    sigma = np.asarray(sheet_conductivity, dtype=np.complex128)
    flat = sigma.reshape((-1, 2, 2))
    y2 = 2.0 / VACUUM_IMPEDANCE_OHM
    reflected = []
    transmitted = []
    for s in flat:
        matrix = np.asarray([[y2 + s[0, 0], s[0, 1]], [s[1, 0], y2 + s[1, 1]]], dtype=np.complex128)
        t = np.linalg.solve(matrix, np.asarray([y2, 0.0], dtype=np.complex128))
        r = t - np.asarray([1.0, 0.0], dtype=np.complex128)
        transmitted.append(t)
        reflected.append(r)
    shape = sigma.shape[:-2]
    return np.asarray(reflected).reshape(shape + (2,)), np.asarray(transmitted).reshape(shape + (2,))


def test_kubo_prefactor_constant_is_two_pi_times_hall_conductance_unit():
    np.testing.assert_allclose(E2_OVER_HBAR_S, 2.0 * np.pi * E2_OVER_H_S, rtol=1.0e-15, atol=0.0)


def test_linear_conductivity_vanishes_when_occupations_equal():
    energies, _occupations, velocity = _two_level_velocity()
    omega = np.asarray([0.8, 1.0, 1.2])
    equal_occ = np.asarray([0.5, 0.5])
    result = linear_conductivity_tensor_from_gauge_data(
        omega,
        energies,
        velocity,
        equal_occ,
        eta_ev=0.02,
        include_bz_factor=False,
    )
    np.testing.assert_allclose(result.conductivity, 0.0, rtol=0.0, atol=0.0)
    assert result.skipped_small_denominators == 0


def test_linear_conductivity_returns_rank2_tensor_over_frequency():
    energies, occupations, velocity = _two_level_velocity()
    omega = np.asarray([0.8, 1.0, 1.2])
    result = linear_conductivity_tensor_from_gauge_data(
        omega,
        energies,
        velocity,
        occupations,
        eta_ev=0.02,
        include_bz_factor=False,
    )
    assert result.conductivity.shape == (omega.size, 2, 2)
    assert np.all(np.isfinite(result.conductivity))
    assert np.max(np.abs(result.conductivity)) > 0.0


def test_linear_conductivity_matches_independent_bruteforce_loop():
    rng = np.random.default_rng(20260704)
    energies = np.asarray([-1.3, -0.25, 0.42, 1.1], dtype=float)
    occupations = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=float)
    velocity = np.empty((2, 4, 4), dtype=np.complex128)
    for axis in range(2):
        raw = rng.normal(size=(4, 4)) + 1.0j * rng.normal(size=(4, 4))
        velocity[axis] = 0.5 * (raw + raw.conjugate().T)
    omega = np.asarray([0.17, 0.53, 1.2], dtype=float)
    result = linear_conductivity_tensor_from_gauge_data(
        omega,
        energies,
        velocity,
        occupations,
        k_weight=1.7,
        eta_ev=0.031,
        prefactor=1.0j * E2_OVER_HBAR_S,
        include_bz_factor=True,
    )
    expected = _bruteforce_linear_conductivity(
        omega,
        energies,
        velocity,
        occupations,
        k_weight=1.7,
        eta_ev=0.031,
        prefactor=1.0j * E2_OVER_HBAR_S,
        include_bz_factor=True,
    )
    np.testing.assert_allclose(result.conductivity, expected, rtol=3.0e-15, atol=3.0e-20)


def test_kerr_faraday_zero_hall_has_zero_rotations():
    sigma = np.zeros((3, 2, 2), dtype=np.complex128)
    sigma[:, 0, 0] = 1.0e-5 + 2.0e-6j
    sigma[:, 1, 1] = 0.8e-5 + 1.0e-6j
    angles = faraday_kerr_from_sheet_conductivity(sigma)
    np.testing.assert_allclose(angles.faraday_angle_rad, 0.0, atol=0.0)
    np.testing.assert_allclose(angles.kerr_angle_rad, 0.0, atol=0.0)


def test_kerr_faraday_matches_independent_sheet_boundary_solve():
    rng = np.random.default_rng(20260704)
    sigma = 1.3e-4 * (rng.normal(size=(97, 2, 2)) + 1.0j * rng.normal(size=(97, 2, 2)))
    angles = faraday_kerr_from_sheet_conductivity(sigma)
    reflected, transmitted = _matrix_sheet_solution(sigma)
    np.testing.assert_allclose(angles.reflected_x, reflected[:, 0], rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(angles.reflected_y, reflected[:, 1], rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(angles.transmitted_x, transmitted[:, 0], rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(angles.transmitted_y, transmitted[:, 1], rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(
        angles.faraday_angle_rad,
        np.real(np.arctan(transmitted[:, 1] / transmitted[:, 0])),
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        angles.kerr_angle_rad,
        np.real(np.arctan(reflected[:, 1] / reflected[:, 0])),
        rtol=3.0e-14,
        atol=3.0e-14,
    )


def _matrix_sheet_on_substrate_solution(sheet_conductivity, substrate_refractive_index):
    sigma = np.asarray(sheet_conductivity, dtype=np.complex128)
    flat = sigma.reshape((-1, 2, 2))
    ns = complex(substrate_refractive_index)
    y0 = 1.0 / VACUUM_IMPEDANCE_OHM
    reflected = []
    transmitted = []
    for s in flat:
        matrix = np.asarray(
            [
                [(1.0 + ns) * y0 + s[0, 0], s[0, 1]],
                [s[1, 0], (1.0 + ns) * y0 + s[1, 1]],
            ],
            dtype=np.complex128,
        )
        t_vacuum = np.linalg.solve(matrix, np.asarray([2.0 * y0, 0.0], dtype=np.complex128))
        total_substrate = np.linalg.solve(matrix, np.asarray([2.0 * ns * y0, 0.0], dtype=np.complex128))
        transmitted.append(t_vacuum)
        reflected.append(total_substrate - np.asarray([1.0, 0.0], dtype=np.complex128))
    shape = sigma.shape[:-2]
    return np.asarray(reflected).reshape(shape + (2,)), np.asarray(transmitted).reshape(shape + (2,))


def test_substrate_sheet_solution_matches_independent_boundary_solve():
    rng = np.random.default_rng(20260712)
    sigma = 8.0e-5 * (rng.normal(size=(41, 2, 2)) + 1.0j * rng.normal(size=(41, 2, 2)))
    result = faraday_kerr_from_sheet_conductivity_on_substrate(
        sigma,
        substrate_refractive_index=3.47,
    )
    reflected, transmitted = _matrix_sheet_on_substrate_solution(sigma, 3.47)
    np.testing.assert_allclose(result.reflected_x, reflected[:, 0], rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(result.reflected_y, reflected[:, 1], rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(result.transmitted_x, transmitted[:, 0], rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(result.transmitted_y, transmitted[:, 1], rtol=3.0e-14, atol=3.0e-14)


def test_substrate_sheet_solution_reduces_to_vacuum_geometry_for_ns_one():
    rng = np.random.default_rng(20260713)
    sigma = 3.0e-5 * (rng.normal(size=(13, 2, 2)) + 1.0j * rng.normal(size=(13, 2, 2)))
    vacuum = faraday_kerr_from_sheet_conductivity(sigma)
    substrate = faraday_kerr_from_sheet_conductivity_on_substrate(sigma, substrate_refractive_index=1.0)
    for name in (
        "reflected_x",
        "reflected_y",
        "transmitted_x",
        "transmitted_y",
        "faraday_angle_rad",
        "kerr_angle_rad",
    ):
        np.testing.assert_allclose(getattr(substrate, name), getattr(vacuum, name), rtol=5.0e-13, atol=5.0e-13)


def test_complex_polarization_recovers_okada_rotation_and_ellipticity():
    theta = np.asarray([0.013, -0.021])
    eta = np.asarray([0.17, -0.08])
    ratio = (np.sin(theta) + 1.0j * eta * np.cos(theta)) / (
        np.cos(theta) - 1.0j * eta * np.sin(theta)
    )
    result = complex_polarization_from_jones(np.ones_like(ratio), ratio)
    np.testing.assert_allclose(result.rotation_angle_rad, theta, rtol=0.0, atol=2.0e-16)
    np.testing.assert_allclose(result.ellipticity, eta, rtol=0.0, atol=2.0e-16)
    np.testing.assert_allclose(result.jones_ratio, ratio, rtol=0.0, atol=0.0)


def test_complex_polarization_is_global_phase_invariant_and_handles_y_linear_and_circular():
    ex = np.asarray([1.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=np.complex128)
    ey = np.asarray([0.2 + 0.1j, 1.0, 1.0j / np.sqrt(2.0)], dtype=np.complex128)
    reference = complex_polarization_from_jones(ex, ey)
    phased = complex_polarization_from_jones(ex * np.exp(0.73j), ey * np.exp(0.73j))
    np.testing.assert_allclose(phased.rotation_angle_rad[:2], reference.rotation_angle_rad[:2], atol=2.0e-16)
    np.testing.assert_allclose(phased.ellipticity, reference.ellipticity, atol=2.0e-16)
    np.testing.assert_allclose(reference.rotation_angle_rad[1], np.pi / 2.0, atol=0.0)
    np.testing.assert_allclose(reference.ellipticity[1], 0.0, atol=0.0)
    np.testing.assert_allclose(reference.ellipticity[2], 1.0, atol=2.0e-16)
    assert np.isnan(reference.rotation_angle_rad[2])
    with pytest.raises(ValueError, match="zero Jones vector"):
        complex_polarization_from_jones(0.0, 0.0)


def test_normalized_complex_transmission_inversion_round_trips_scalar_and_matrix():
    ns = OKADA_2016_INP_REFRACTIVE_INDEX
    y_sum = (1.0 + ns) / VACUUM_IMPEDANCE_OHM
    scalar_sigma = 0.37e-4 + 0.12e-4j
    scalar_transmission = y_sum / (y_sum + scalar_sigma)
    np.testing.assert_allclose(
        sheet_conductivity_from_scalar_normalized_transmission(
            scalar_transmission,
            transmitted_refractive_index=ns,
        ),
        scalar_sigma,
        rtol=1.0e-14,
        atol=0.0,
    )

    sigma = np.asarray(
        [[0.31e-4 + 0.08e-4j, 0.12e-4 - 0.03e-4j],
         [-0.11e-4 + 0.02e-4j, 0.28e-4 + 0.07e-4j]],
        dtype=np.complex128,
    )
    normalized = y_sum * np.linalg.inv(y_sum * np.eye(2) + sigma)
    recovered = sheet_conductivity_from_normalized_transmission_matrix(
        normalized,
        transmitted_refractive_index=ns,
    )
    np.testing.assert_allclose(recovered, sigma, rtol=1.0e-13, atol=1.0e-20)

    batched_sigma = np.stack((sigma, 0.7 * sigma))
    n1 = 1.4
    n2 = 2.2 + 0.08j
    batched_y_sum = (n1 + n2) / VACUUM_IMPEDANCE_OHM
    batched_transmission = batched_y_sum * np.linalg.inv(
        batched_y_sum * np.eye(2) + batched_sigma
    )
    np.testing.assert_allclose(
        sheet_conductivity_from_normalized_transmission_matrix(
            batched_transmission,
            incident_refractive_index=n1,
            transmitted_refractive_index=n2,
        ),
        batched_sigma,
        rtol=2.0e-13,
        atol=2.0e-20,
    )


def test_substrate_zero_sheet_has_correct_incidence_side_fresnel_amplitudes():
    ns = OKADA_2016_INP_REFRACTIVE_INDEX
    result = faraday_kerr_from_sheet_conductivity_on_substrate(
        np.zeros((2, 2), dtype=np.complex128),
        substrate_refractive_index=ns,
    )
    np.testing.assert_allclose(result.transmitted_x, 2.0 / (1.0 + ns), rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(result.reflected_x, (ns - 1.0) / (ns + 1.0), rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(result.transmitted_y, 0.0, atol=0.0)
    np.testing.assert_allclose(result.reflected_y, 0.0, atol=0.0)


def test_okada2016_cartesian_solver_matches_circular_formula_for_both_hall_signs():
    ns = OKADA_2016_INP_REFRACTIVE_INDEX
    longitudinal = 0.23 * E2_OVER_H_S
    for hall_sign in (-1.0, 1.0):
        hall = hall_sign * E2_OVER_H_S
        benchmark = okada2016_qah_benchmark(
            longitudinal_e2_over_h=0.23,
            hall_e2_over_h=hall_sign,
        )
        # Physical Cartesian mapping: sigma_xy(repo) = sigma_xy(Okada).
        y_plus = VACUUM_IMPEDANCE_OHM * (longitudinal + 1.0j * hall)
        y_minus = VACUUM_IMPEDANCE_OHM * (longitudinal - 1.0j * hall)
        t_plus = 2.0 / (1.0 + ns + y_plus)
        t_minus = 2.0 / (1.0 + ns + y_minus)
        r_plus = (-1.0 + ns - y_plus) / (1.0 + ns + y_plus)
        r_minus = (-1.0 + ns - y_minus) / (1.0 + ns + y_minus)
        expected_faraday = 0.5 * np.angle(t_minus / t_plus)
        expected_kerr = 0.5 * np.angle(r_minus / r_plus)
        np.testing.assert_allclose(benchmark.angles.faraday_angle_rad, expected_faraday, rtol=2.0e-14, atol=0.0)
        np.testing.assert_allclose(benchmark.angles.kerr_angle_rad, expected_kerr, rtol=2.0e-14, atol=0.0)


def test_okada2016_ideal_quantized_angles_match_analytic_values():
    benchmark = okada2016_qah_benchmark()
    np.testing.assert_allclose(benchmark.angles.faraday_angle_rad, 3.265023104438041e-3, rtol=2.0e-14, atol=0.0)
    np.testing.assert_allclose(benchmark.angles.kerr_angle_rad, 9.173741845351132e-3, rtol=2.0e-14, atol=0.0)


def test_okada2016_universal_scaling_recovers_fine_structure_constant():
    benchmark = okada2016_qah_benchmark()
    direct = okada2016_scaling_function(
        benchmark.angles.faraday_angle_rad,
        benchmark.angles.kerr_angle_rad,
    )
    np.testing.assert_allclose(direct, benchmark.fine_structure_constant, rtol=2.0e-14, atol=0.0)
    np.testing.assert_allclose(
        benchmark.fine_structure_constant,
        0.5 * VACUUM_IMPEDANCE_OHM * E2_OVER_H_S,
        rtol=0.0,
        atol=0.0,
    )


def test_okada2016_scaling_matches_exact_dc_conductivity_identity_with_sigma_xx():
    for ns in (1.2, OKADA_2016_INP_REFRACTIVE_INDEX, 5.1):
        for longitudinal_e2h, hall_e2h in ((0.0, 0.4), (0.15, 0.7), (0.42, -0.6)):
            benchmark = okada2016_qah_benchmark(
                substrate_refractive_index=ns,
                longitudinal_e2_over_h=longitudinal_e2h,
                hall_e2_over_h=hall_e2h,
            )
            expected = okada2016_scaling_from_dc_conductivity(
                longitudinal_e2h * E2_OVER_H_S,
                hall_e2h * E2_OVER_H_S,
            )
            np.testing.assert_allclose(benchmark.scaling_function, expected, rtol=3.0e-14, atol=0.0)


def test_okada2016_scaling_uncertainty_jacobian_matches_finite_difference():
    theta_f = 2.6e-3
    theta_k = 6.9e-3
    std_f = 0.15e-3
    std_k = 0.4e-3
    propagated = okada2016_scaling_uncertainty(theta_f, theta_k, std_f, std_k)
    step = 1.0e-8
    derivative_f = (
        okada2016_scaling_function(theta_f + step, theta_k)
        - okada2016_scaling_function(theta_f - step, theta_k)
    ) / (2.0 * step)
    derivative_k = (
        okada2016_scaling_function(theta_f, theta_k + step)
        - okada2016_scaling_function(theta_f, theta_k - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(propagated.derivative_faraday, derivative_f, rtol=2.0e-9, atol=0.0)
    np.testing.assert_allclose(propagated.derivative_kerr, derivative_k, rtol=2.0e-9, atol=0.0)
    expected_std = np.hypot(derivative_f * std_f, derivative_k * std_k)
    np.testing.assert_allclose(propagated.standard_deviation, expected_std, rtol=2.0e-9, atol=0.0)

    covariance = 0.4 * std_f * std_k
    correlated = okada2016_scaling_uncertainty(
        theta_f,
        theta_k,
        std_f,
        std_k,
        covariance_rad2=covariance,
    )
    expected_variance = expected_std**2 + 2.0 * derivative_f * derivative_k * covariance
    np.testing.assert_allclose(correlated.standard_deviation**2, expected_variance, rtol=3.0e-9)
    with pytest.raises(ValueError, match="non-negative"):
        okada2016_scaling_uncertainty(theta_f, theta_k, -std_f, std_k)
    with pytest.raises(ValueError, match="positive-semidefinite"):
        okada2016_scaling_uncertainty(
            theta_f,
            theta_k,
            std_f,
            std_k,
            covariance_rad2=1.01 * std_f * std_k,
        )


def test_okada2016_scaling_is_stable_for_microradian_angles():
    theta_f = 2.6e-8
    theta_k = 6.9e-8
    direct = okada2016_scaling_function(theta_f, theta_k)
    small_angle_limit = theta_f * (theta_k - theta_f) / (theta_k - 2.0 * theta_f)
    np.testing.assert_allclose(direct, small_angle_limit, rtol=2.0e-14, atol=0.0)


def test_okada2016_scaling_is_substrate_independent_for_dissipationless_hall_sheet():
    hall_e2_over_h = 0.37
    expected = hall_e2_over_h * 0.5 * VACUUM_IMPEDANCE_OHM * E2_OVER_H_S
    for ns in (1.4, OKADA_2016_INP_REFRACTIVE_INDEX, 6.0):
        benchmark = okada2016_qah_benchmark(
            substrate_refractive_index=ns,
            hall_e2_over_h=hall_e2_over_h,
        )
        np.testing.assert_allclose(benchmark.scaling_function, expected, rtol=2.0e-13, atol=0.0)


def test_qwz_kubo_to_okada_chain_closes_chern_hall_and_rotation_signs():
    mesh = 16
    chern = qwz_lower_band_chern_dvector(mesh, mass=1.0)
    response = kerr_faraday_from_kpoint_data_on_substrate(
        np.asarray([0.0]),
        qwz_optical_kpoint_data(mesh, mass=1.0),
        substrate_refractive_index=OKADA_2016_INP_REFRACTIVE_INDEX,
        prefactor=1.0j * E2_OVER_HBAR_S,
        eta_ev=1.0e-3,
    )
    generic_response = optical_response_from_kpoint_data(
        "kerr",
        np.asarray([0.0]),
        qwz_optical_kpoint_data(mesh, mass=1.0),
        si_prefactor=1.0j * E2_OVER_HBAR_S,
        eta_ev=1.0e-3,
        geometry=Okada2016MixedPulseGeometry(
            substrate_refractive_index=OKADA_2016_INP_REFRACTIVE_INDEX,
        ),
    )
    np.testing.assert_allclose(
        generic_response.conductivity,
        response.conductivity,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        generic_response.faraday_angle_rad,
        response.faraday_angle_rad,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        generic_response.kerr_angle_rad,
        response.kerr_angle_rad,
        rtol=0.0,
        atol=0.0,
    )
    sigma = response.conductivity[0] / E2_OVER_H_S
    np.testing.assert_allclose(sigma[0, 1].real, chern, rtol=3.0e-6, atol=0.0)
    np.testing.assert_allclose(sigma[1, 0].real, -chern, rtol=3.0e-6, atol=0.0)
    assert response.faraday_angle_rad[0] > 0.0
    assert response.kerr_angle_rad[0] > 0.0
    ideal = okada2016_qah_benchmark()
    np.testing.assert_allclose(response.faraday_angle_rad[0], ideal.angles.faraday_angle_rad, rtol=1.0e-4, atol=0.0)
    np.testing.assert_allclose(response.kerr_angle_rad[0], ideal.angles.kerr_angle_rad, rtol=1.0e-4, atol=0.0)


def test_small_conductivity_faraday_quantized_hall_limit_matches_alpha():
    sigma = np.zeros((2, 2), dtype=np.complex128)
    sigma[0, 1] = E2_OVER_H_S
    sigma[1, 0] = -E2_OVER_H_S
    angles = faraday_kerr_small_conductivity(sigma)
    alpha_from_constants = E2_OVER_H_S * VACUUM_IMPEDANCE_OHM / 2.0
    np.testing.assert_allclose(
        angles.faraday_angle_rad,
        np.arctan(alpha_from_constants),
        rtol=1.0e-14,
        atol=0.0,
    )


def test_liu_dai_printed_compatibility_helper_only_flips_transverse_fields():
    sigma = np.asarray(
        [[1.2e-5 + 0.3e-5j, 0.4e-5 - 0.2e-5j], [-0.4e-5 + 0.2e-5j, 0.9e-5]],
        dtype=np.complex128,
    )
    physical = faraday_kerr_from_sheet_conductivity(sigma)
    printed = faraday_kerr_from_sheet_conductivity_liu_dai_2020_printed(sigma)
    np.testing.assert_allclose(printed.reflected_x, physical.reflected_x)
    np.testing.assert_allclose(printed.transmitted_x, physical.transmitted_x)
    np.testing.assert_allclose(printed.reflected_y, -physical.reflected_y)
    np.testing.assert_allclose(printed.transmitted_y, -physical.transmitted_y)
    np.testing.assert_allclose(printed.faraday_angle_rad, -physical.faraday_angle_rad)
    np.testing.assert_allclose(printed.kerr_angle_rad, -physical.kerr_angle_rad)


def test_exact_and_small_faraday_agree_for_weak_sheet_conductivity():
    sigma = np.asarray([[1.0e-7 + 2.0e-8j, 0.0], [2.0e-8 - 1.0e-8j, 1.1e-7]], dtype=np.complex128)
    exact = faraday_kerr_from_sheet_conductivity(sigma)
    small = faraday_kerr_small_conductivity(sigma)
    np.testing.assert_allclose(exact.faraday_angle_rad, small.faraday_angle_rad, rtol=2.0e-4, atol=2.0e-10)


def test_kerr_faraday_kpoint_front_door_matches_manual_chain():
    energies, occupations, velocity = _two_level_velocity()
    omega = np.asarray([0.8, 1.0, 1.2])
    payload = OpticalKPointData(energies_ev=energies, velocity_h=velocity, occupations=occupations, weight=1.7)
    response = kerr_faraday_from_kpoint_data(
        omega,
        [payload],
        eta_ev=0.02,
        prefactor=1.0j * E2_OVER_H_S,
        include_bz_factor=False,
    )
    manual = linear_conductivity_from_kpoint_data(
        omega,
        [payload],
        eta_ev=0.02,
        prefactor=1.0j * E2_OVER_H_S,
        include_bz_factor=False,
    )
    manual_angles = faraday_kerr_from_sheet_conductivity(manual.conductivity)
    np.testing.assert_allclose(response.conductivity, manual.conductivity)
    np.testing.assert_allclose(response.faraday_angle_rad, manual_angles.faraday_angle_rad)
    np.testing.assert_allclose(response.kerr_angle_rad, manual_angles.kerr_angle_rad)
    assert response.skipped_small_denominators == manual.skipped_small_denominators


def test_kerr_faraday_substrate_front_door_matches_manual_chain():
    energies, occupations, velocity = _two_level_velocity()
    omega = np.asarray([0.8, 1.0, 1.2])
    payload = OpticalKPointData(energies_ev=energies, velocity_h=velocity, occupations=occupations, weight=1.7)
    response = kerr_faraday_from_kpoint_data_on_substrate(
        omega,
        [payload],
        substrate_refractive_index=3.47,
        eta_ev=0.02,
        prefactor=1.0j * E2_OVER_H_S,
        include_bz_factor=False,
    )
    manual = linear_conductivity_from_kpoint_data(
        omega,
        [payload],
        eta_ev=0.02,
        prefactor=1.0j * E2_OVER_H_S,
        include_bz_factor=False,
    )
    manual_angles = faraday_kerr_from_sheet_conductivity_on_substrate(
        manual.conductivity,
        substrate_refractive_index=3.47,
    )
    np.testing.assert_allclose(response.conductivity, manual.conductivity)
    np.testing.assert_allclose(response.faraday_angle_rad, manual_angles.faraday_angle_rad)
    np.testing.assert_allclose(response.kerr_angle_rad, manual_angles.kerr_angle_rad)


def test_velocity_gauge_2d_prefactors_have_expected_unit_relation():
    np.testing.assert_allclose(VELOCITY_GAUGE_2D_PREFAC_A_M_PER_V2, E2_OVER_HBAR_S * 1.0e-9)
    np.testing.assert_allclose(VELOCITY_GAUGE_2D_PREFAC_UA_NM_PER_V2, E2_OVER_HBAR_S * 1.0e6)
    np.testing.assert_allclose(VELOCITY_GAUGE_2D_PREFAC_UA_NM_PER_V2, VELOCITY_GAUGE_2D_PREFAC_A_M_PER_V2 * 1.0e15)
    np.testing.assert_allclose(LIU_DAI_2020_EQ4_PREFAC_A_M_PER_V2, -VELOCITY_GAUGE_2D_PREFAC_A_M_PER_V2)
    np.testing.assert_allclose(LIU_DAI_2020_EQ4_PREFAC_UA_NM_PER_V2, -VELOCITY_GAUGE_2D_PREFAC_UA_NM_PER_V2)
    np.testing.assert_allclose(LIU_DAI_2020_EQ4_SHG_PREFAC_S, -E2_OVER_HBAR_S)


def _three_level_velocity():
    energies = np.asarray([-0.8, -0.1, 0.55], dtype=float)
    occupations = np.asarray([1.0, 0.0, 0.0], dtype=float)
    velocity = np.asarray(
        [
            [[0.1, 0.4 + 0.2j, -0.3j], [0.4 - 0.2j, -0.2, 0.25], [0.3j, 0.25, 0.05]],
            [[-0.05, -0.1j, 0.35], [0.1j, 0.15, -0.2 + 0.1j], [0.35, -0.2 - 0.1j, -0.1]],
        ],
        dtype=np.complex128,
    )
    return energies, occupations, velocity


def _bruteforce_shift_velocity_gauge(omega, energies, velocity, occupations, eta):
    out = np.zeros((len(omega), 2, 2, 2), dtype=float)
    for l in range(3):
        for n in range(3):
            fln = occupations[l] - occupations[n]
            if fln == 0.0:
                continue
            en_l = energies[n] - energies[l]
            for m in range(3):
                en_m = energies[n] - energies[m]
                if abs(en_m) <= 1.0e-12 or abs(en_l) <= 1.0e-12:
                    continue
                for sign in (-1.0, 1.0):
                    resonance = fln / ((en_m - 1.0j * eta) * (en_l + sign * omega - 1.0j * eta)) / (omega * omega)
                    for c in range(2):
                        for a in range(2):
                            for b in range(2):
                                out[:, c, a, b] += np.real(resonance * velocity[a, n, l] * velocity[b, l, m] * velocity[c, m, n])
    return out


def test_shift_current_velocity_gauge_matches_bruteforce_three_level_sum():
    energies, occupations, velocity = _three_level_velocity()
    omega = np.asarray([0.35, 0.7], dtype=float)
    result = shift_current_conductivity_velocity_gauge(
        omega,
        energies,
        velocity,
        occupations,
        eta_ev=0.04,
        include_bz_factor=False,
    )
    expected = _bruteforce_shift_velocity_gauge(omega, energies, velocity, occupations, 0.04)
    np.testing.assert_allclose(result.conductivity, expected, rtol=2.0e-14, atol=2.0e-14)


def _bruteforce_shift_velocity_gauge_eta_regularized(omega, energies, velocity, occupations, eta):
    out = np.zeros((len(omega), 2, 2, 2), dtype=float)
    for l in range(3):
        for n in range(3):
            fln = occupations[l] - occupations[n]
            if fln == 0.0:
                continue
            en_l = energies[n] - energies[l]
            for m in range(3):
                en_m = energies[n] - energies[m]
                for sign in (-1.0, 1.0):
                    resonance = fln / ((en_m - 1.0j * eta) * (en_l + sign * omega - 1.0j * eta)) / (omega * omega)
                    for c in range(2):
                        for a in range(2):
                            for b in range(2):
                                out[:, c, a, b] += np.real(resonance * velocity[a, n, l] * velocity[b, l, m] * velocity[c, m, n])
    return out


def test_shift_current_velocity_gauge_can_eta_regularize_diagonal_denominators():
    energies, occupations, velocity = _three_level_velocity()
    omega = np.asarray([0.35, 0.7], dtype=float)
    result = shift_current_conductivity_velocity_gauge(
        omega,
        energies,
        velocity,
        occupations,
        eta_ev=0.04,
        include_bz_factor=False,
        skip_degenerate_energy_denominators=False,
    )
    expected = _bruteforce_shift_velocity_gauge_eta_regularized(omega, energies, velocity, occupations, 0.04)
    assert result.skipped_small_denominators == 0
    np.testing.assert_allclose(result.conductivity, expected, rtol=2.0e-14, atol=2.0e-14)


def _bruteforce_shg_velocity_gauge(omega, energies, velocity, occupations, eta):
    out = np.zeros((len(omega), 2, 2, 2), dtype=np.complex128)
    for l in range(3):
        for n in range(3):
            fln = occupations[l] - occupations[n]
            if fln == 0.0:
                continue
            en_l = energies[n] - energies[l]
            for m in range(3):
                en_m = energies[n] - energies[m]
                if abs(en_m) <= 1.0e-12 or abs(en_l) <= 1.0e-12:
                    continue
                for sign in (-1.0, 1.0):
                    resonance = fln / ((en_m - 2.0 * sign * omega - 1.0j * eta) * (en_l + sign * omega - 1.0j * eta)) / (omega * omega)
                    for c in range(2):
                        for a in range(2):
                            for b in range(2):
                                out[:, c, a, b] += resonance * velocity[a, n, l] * velocity[b, l, m] * velocity[c, m, n]
    return out


def test_shg_velocity_gauge_matches_bruteforce_three_level_sum():
    energies, occupations, velocity = _three_level_velocity()
    omega = np.asarray([0.35, 0.7], dtype=float)
    result = shg_conductivity_velocity_gauge(
        omega,
        energies,
        velocity,
        occupations,
        eta_ev=0.04,
        include_bz_factor=False,
    )
    expected = _bruteforce_shg_velocity_gauge(omega, energies, velocity, occupations, 0.04)
    np.testing.assert_allclose(result.conductivity, expected, rtol=2.0e-14, atol=2.0e-14)


def _bruteforce_shg_velocity_gauge_eta_regularized(omega, energies, velocity, occupations, eta):
    out = np.zeros((len(omega), 2, 2, 2), dtype=np.complex128)
    for l in range(3):
        for n in range(3):
            fln = occupations[l] - occupations[n]
            if fln == 0.0:
                continue
            en_l = energies[n] - energies[l]
            for m in range(3):
                en_m = energies[n] - energies[m]
                for sign in (-1.0, 1.0):
                    resonance = fln / ((en_m - 2.0 * sign * omega - 1.0j * eta) * (en_l + sign * omega - 1.0j * eta)) / (omega * omega)
                    for c in range(2):
                        for a in range(2):
                            for b in range(2):
                                out[:, c, a, b] += resonance * velocity[a, n, l] * velocity[b, l, m] * velocity[c, m, n]
    return out


def test_shg_velocity_gauge_can_eta_regularize_diagonal_denominators():
    energies, occupations, velocity = _three_level_velocity()
    omega = np.asarray([0.35, 0.7], dtype=float)
    result = shg_conductivity_velocity_gauge(
        omega,
        energies,
        velocity,
        occupations,
        eta_ev=0.04,
        include_bz_factor=False,
        skip_degenerate_energy_denominators=False,
    )
    expected = _bruteforce_shg_velocity_gauge_eta_regularized(omega, energies, velocity, occupations, 0.04)
    assert result.skipped_small_denominators == 0
    np.testing.assert_allclose(result.conductivity, expected, rtol=2.0e-14, atol=2.0e-14)


def test_shg_conductivity_vanishes_when_occupations_equal():
    energies, _occupations, velocity = _two_level_velocity()
    omega = np.asarray([0.4, 0.6])
    result = shg_conductivity_velocity_gauge(
        omega,
        energies,
        velocity,
        np.asarray([0.5, 0.5]),
        eta_ev=0.03,
        include_bz_factor=False,
    )
    np.testing.assert_allclose(result.conductivity, 0.0, rtol=0.0, atol=0.0)


def test_shg_susceptibility_conversion_matches_liu_dai_relation():
    omega_ev = np.asarray([0.1, 0.2], dtype=float)
    target_chi_m_per_v = np.asarray([2.0e-12, -3.0e-12], dtype=np.complex128)
    omega_rad_s = omega_ev / HBAR_EV_S
    sigma = -1.0j * 2.0 * VACUUM_PERMITTIVITY_F_PER_M * omega_rad_s * target_chi_m_per_v
    chi = shg_susceptibility_from_conductivity(sigma, omega_ev)
    np.testing.assert_allclose(chi, target_chi_m_per_v, rtol=1.0e-14, atol=0.0)
    chi_pm = shg_susceptibility_from_conductivity(sigma, omega_ev, output="pm_per_v")
    np.testing.assert_allclose(chi_pm, target_chi_m_per_v * 1.0e12, rtol=1.0e-14, atol=0.0)
