from __future__ import annotations

import numpy as np

from analysis import response_derivative_gauge as old_gauge
from analysis.optical_response import (
    JOYA_EQ7_GEOMETRIC_CONVENTION,
    ShiftCurrentComponent,
    berry_connection_generalized_derivative,
    degenerate_band_groups,
    fermi_occupation,
    link_shift_vector,
    parse_component,
    positive_transition_pairs,
    random_block_unitary,
    wannierberri_shift_current_group_trace,
)
from analysis.optical_response import shift_current as new_shift
from analysis.optical_response.toy_models import GappedSLGParams as NewGappedSLGParams, hamiltonian as new_slg_hamiltonian
from analysis.shift_current import core as old_shift
from analysis.shift_current.toy_models import GappedSLGParams as OldGappedSLGParams, hamiltonian as old_slg_hamiltonian
from mean_field.systems.tdbg import TDBGModel, TDBGParameters
from mean_field.systems.tdbg.shift_current import model_shift_current_point_data


def test_optical_response_reexports_gauge_and_shift_current_api() -> None:
    energies = np.asarray([-1.0, 2.0], dtype=float)
    velocity = np.zeros((2, 2, 2), dtype=np.complex128)
    velocity[0, 0, 1] = 0.5
    velocity[0, 1, 0] = 0.5
    velocity[1, 0, 1] = -0.25j
    velocity[1, 1, 0] = 0.25j

    new_der = berry_connection_generalized_derivative(velocity, energies)
    old_der = old_gauge.berry_connection_generalized_derivative(velocity, energies)
    np.testing.assert_allclose(new_der.values, old_der.values)

    np.testing.assert_allclose(fermi_occupation(energies, mu_ev=0.0), old_shift.fermi_occupation(energies, mu_ev=0.0))
    assert parse_component("x;yy") == ShiftCurrentComponent(0, 1, 1)
    assert positive_transition_pairs(energies, np.asarray([1.0, 0.0])) == old_shift.positive_transition_pairs(
        energies, np.asarray([1.0, 0.0])
    )
    assert new_shift.ShiftCurrentTensors is old_shift.ShiftCurrentTensors


def test_optical_response_package_exports_split_module_symbols() -> None:
    energies = np.asarray([0.0, 1.0, 1.0 + 1.0e-6], dtype=float)
    assert degenerate_band_groups(energies, threshold=1.0e-4) == [(0, 1), (1, 3)]
    gauge = random_block_unitary([(0, 1), (1, 3)], 3, rng=123)
    np.testing.assert_allclose(gauge.conjugate().T @ gauge, np.eye(3), atol=1.0e-12)
    assert JOYA_EQ7_GEOMETRIC_CONVENTION.normalized_lorentzian is False
    imn = np.ones((3, 3, 2, 2, 2), dtype=float)
    np.testing.assert_allclose(wannierberri_shift_current_group_trace(imn, [0], [1, 2]), np.full((2, 2, 2), 2.0))
    assert callable(link_shift_vector)


def test_optical_response_toy_model_owns_old_import_path() -> None:
    params = NewGappedSLGParams(mass_ev=0.012)
    assert OldGappedSLGParams is NewGappedSLGParams
    np.testing.assert_allclose(new_slg_hamiltonian((0.1, -0.2), params), old_slg_hamiltonian((0.1, -0.2), params))


def test_tdbg_shift_current_adapter_uses_new_optical_response_path() -> None:
    model = TDBGModel.from_config(1.38, cut=0.1, params=TDBGParameters.minimal())
    point = model_shift_current_point_data(model, 0.0 + 0.0j, fd_step_nm_inv=1.0e-5)
    assert point.energies_ev.ndim == 1
    assert point.eigenvectors.shape[0] == point.energies_ev.size
    assert point.gauge_data.velocity_h.shape[0] == 2
    assert point.gauge_data.berry_connection.shape[0] == 2
