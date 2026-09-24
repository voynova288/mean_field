from __future__ import annotations

import numpy as np

from analysis.response_derivative_gauge import hamiltonian_gauge_data
from analysis.optical import (
    OpticalKPointData,
    linear_conductivity_from_kpoint_data,
    linear_conductivity_tensor_from_gauge_data,
    optical_kpoint_data_from_eigensystem,
    shift_current_conductivity_velocity_gauge,
    shift_current_velocity_gauge_from_kpoint_data,
    shg_conductivity_from_kpoint_data,
    shg_conductivity_velocity_gauge,
)


def _two_level_payload(weight: float = 1.0) -> OpticalKPointData:
    energies = np.asarray([-0.5, 0.5], dtype=float)
    occupations = np.asarray([1.0, 0.0], dtype=float)
    velocity = np.zeros((2, 2, 2), dtype=np.complex128)
    velocity[0, 0, 1] = 1.0
    velocity[0, 1, 0] = 1.0
    velocity[1, 0, 1] = -1.0j
    velocity[1, 1, 0] = 1.0j
    return OpticalKPointData(energies_ev=energies, velocity_h=velocity, occupations=occupations, weight=weight)


def test_optical_kpoint_data_from_eigensystem_rotates_dhdk():
    energies = np.asarray([-0.5, 0.5])
    evecs = np.eye(2, dtype=np.complex128)
    dhdk = np.zeros((2, 2, 2), dtype=np.complex128)
    dhdk[0, 0, 1] = 2.0
    dhdk[0, 1, 0] = 2.0
    payload = optical_kpoint_data_from_eigensystem(energies, evecs, dhdk, weight=0.25)
    np.testing.assert_allclose(payload.velocity_h, dhdk)
    np.testing.assert_allclose(payload.occupations, [1.0, 0.0])
    assert payload.weight == 0.25


def test_optical_kpoint_data_uses_shared_hamiltonian_gauge_layer():
    energies = np.asarray([-0.7, 0.2])
    theta = 0.37
    evecs = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.complex128,
    )
    dhdk = np.asarray(
        [
            [[0.3, 1.2], [1.2, -0.1]],
            [[0.0, -0.4j], [0.4j, 0.2]],
        ],
        dtype=np.complex128,
    )
    payload = optical_kpoint_data_from_eigensystem(energies, evecs, dhdk, mu_ev=-0.1)
    gauge = hamiltonian_gauge_data(energies, evecs, dhdk)
    np.testing.assert_allclose(payload.velocity_h, gauge.velocity_h)
    np.testing.assert_allclose(payload.energies_ev, gauge.energies)
    np.testing.assert_allclose(payload.occupations, [1.0, 0.0])


def test_linear_kmesh_accumulation_matches_single_combined_weight():
    omega = np.asarray([0.8, 1.0, 1.2])
    split = linear_conductivity_from_kpoint_data(
        omega,
        [_two_level_payload(0.25), _two_level_payload(0.75)],
        eta_ev=0.02,
        include_bz_factor=False,
    )
    single_payload = _two_level_payload(1.0)
    direct = linear_conductivity_tensor_from_gauge_data(
        omega,
        single_payload.energies_ev,
        single_payload.velocity_h,
        single_payload.occupations,
        eta_ev=0.02,
        include_bz_factor=False,
    )
    np.testing.assert_allclose(split.conductivity, direct.conductivity, rtol=0.0, atol=1.0e-14)
    assert split.skipped_small_denominators == direct.skipped_small_denominators * 2


def test_shift_current_velocity_gauge_kmesh_accumulation_matches_single_combined_weight():
    omega = np.asarray([0.4, 0.6])
    split = shift_current_velocity_gauge_from_kpoint_data(
        omega,
        [_two_level_payload(0.2), _two_level_payload(0.8)],
        eta_ev=0.03,
        include_bz_factor=False,
        omega_power_regularizer_ev=0.01,
    )
    single_payload = _two_level_payload(1.0)
    direct = shift_current_conductivity_velocity_gauge(
        omega,
        single_payload.energies_ev,
        single_payload.velocity_h,
        single_payload.occupations,
        eta_ev=0.03,
        include_bz_factor=False,
        omega_power_regularizer_ev=0.01,
    )
    np.testing.assert_allclose(split.conductivity, direct.conductivity, rtol=0.0, atol=1.0e-14)
    assert split.skipped_small_denominators == direct.skipped_small_denominators * 2


def test_shg_kmesh_accumulation_matches_single_combined_weight():
    omega = np.asarray([0.4, 0.6])
    split = shg_conductivity_from_kpoint_data(
        omega,
        [_two_level_payload(0.2), _two_level_payload(0.8)],
        eta_ev=0.03,
        include_bz_factor=False,
        omega_power_regularizer_ev=0.01,
    )
    single_payload = _two_level_payload(1.0)
    direct = shg_conductivity_velocity_gauge(
        omega,
        single_payload.energies_ev,
        single_payload.velocity_h,
        single_payload.occupations,
        eta_ev=0.03,
        include_bz_factor=False,
        omega_power_regularizer_ev=0.01,
    )
    np.testing.assert_allclose(split.conductivity, direct.conductivity, rtol=0.0, atol=1.0e-14)
    assert split.skipped_small_denominators == direct.skipped_small_denominators * 2
