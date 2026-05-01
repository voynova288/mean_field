from __future__ import annotations

import numpy as np

from mean_field.core.hf import ScreenedCoulombParams, screened_coulomb, screened_coulomb_matrix


def test_screened_coulomb_has_finite_double_gate_zero_limit() -> None:
    params = ScreenedCoulombParams(epsilon_r=8.0, d_sc_nm=25.0)
    zero_value = screened_coulomb(0.0, params)
    small_value = screened_coulomb(1.0e-8, params, zero_cutoff_nm_inv=1.0e-12)

    assert np.isclose(zero_value, 2.0 * np.pi * 1.439964547 * 25.0 / 8.0)
    assert np.isclose(small_value, zero_value, rtol=1.0e-6)


def test_screened_coulomb_asymptotes_to_inverse_q() -> None:
    params = ScreenedCoulombParams(epsilon_r=8.0, d_sc_nm=25.0)
    q_values = np.asarray([4.0, 8.0], dtype=float)
    values = screened_coulomb_matrix(q_values, params)

    assert np.allclose(values * q_values, values[0] * q_values[0], rtol=1.0e-12)
