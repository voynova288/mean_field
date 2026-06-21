from __future__ import annotations

import numpy as np

from analysis import response_derivative_gauge as old_gauge
from analysis.optical_response import (
    ShiftCurrentComponent,
    berry_connection_generalized_derivative,
    fermi_occupation,
    parse_component,
    positive_transition_pairs,
)
from analysis.optical_response import shift_current as new_shift
from analysis.shift_current import core as old_shift
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


def test_tdbg_shift_current_adapter_uses_new_optical_response_path() -> None:
    model = TDBGModel.from_config(1.38, cut=0.1, params=TDBGParameters.minimal())
    point = model_shift_current_point_data(model, 0.0 + 0.0j, fd_step_nm_inv=1.0e-5)
    assert point.energies_ev.ndim == 1
    assert point.eigenvectors.shape[0] == point.energies_ev.size
    assert point.gauge_data.velocity_h.shape[0] == 2
    assert point.gauge_data.berry_connection.shape[0] == 2
