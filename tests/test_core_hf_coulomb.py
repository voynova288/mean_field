from __future__ import annotations

import numpy as np

from mean_field.core.hf.coulomb import (
    ScreenedCoulombParams,
    screened_coulomb,
    screened_coulomb_matrix,
)


def test_screened_coulomb_has_finite_double_gate_zero_limit() -> None:
    params = ScreenedCoulombParams(epsilon_r=8.0, d_sc_nm=25.0, finite_zero_limit=True)
    zero_value = screened_coulomb(0.0, params)
    small_value = screened_coulomb(1.0e-8, params, zero_cutoff_nm_inv=1.0e-12)

    assert np.isclose(zero_value, 2.0 * np.pi * 1.439964547 * 25.0 / 8.0)
    assert np.isclose(small_value, zero_value, rtol=1.0e-6)


def test_screened_coulomb_can_drop_exact_zero_point() -> None:
    params = ScreenedCoulombParams(epsilon_r=8.0, d_sc_nm=25.0, finite_zero_limit=False)
    zero_value = screened_coulomb(0.0, params)
    small_value = screened_coulomb(1.0e-8, params, zero_cutoff_nm_inv=1.0e-12)

    assert zero_value == 0.0
    assert small_value > 0.0


def test_screened_coulomb_asymptotes_to_inverse_q() -> None:
    params = ScreenedCoulombParams(epsilon_r=8.0, d_sc_nm=25.0)
    q_values = np.asarray([4.0, 8.0], dtype=float)
    values = screened_coulomb_matrix(q_values, params)

    assert np.allclose(values * q_values, values[0] * q_values[0], rtol=1.0e-12)


def test_screened_coulomb_matches_xi_over_two_paper_convention() -> None:
    xi_nm = 50.0
    epsilon_r = 5.0
    q = np.asarray([0.05, 0.2, 1.0], dtype=float)
    params = ScreenedCoulombParams.from_gate_separation_xi_nm(epsilon_r=epsilon_r, xi_nm=xi_nm)
    values = screened_coulomb_matrix(q, params)
    expected = 2.0 * np.pi * 1.439964547 / epsilon_r * np.tanh(q * xi_nm / 2.0) / q

    assert np.allclose(values, expected, rtol=1.0e-14, atol=0.0)
