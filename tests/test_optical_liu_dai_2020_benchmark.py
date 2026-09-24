from __future__ import annotations

import numpy as np

from analysis.optical.benchmarks.liu_dai_2020 import (
    c3_inplane_second_order_tensor,
    decompose_c3_inplane_second_order_tensor,
    qah_faraday_kerr_benchmark,
    qah_faraday_kerr_small_benchmark,
    qah_sheet_conductivity_tensor,
)
from analysis.optical.linear import E2_OVER_H_S
from analysis.optical.magneto import VACUUM_IMPEDANCE_OHM


def test_qah_sheet_conductivity_tensor_matches_physical_chern_hall_convention():
    sigma = qah_sheet_conductivity_tensor(2)
    assert sigma.shape == (2, 2)
    np.testing.assert_allclose(sigma[0, 1], 2.0 * E2_OVER_H_S)
    np.testing.assert_allclose(sigma[1, 0], -2.0 * E2_OVER_H_S)
    np.testing.assert_allclose(sigma[0, 0], 0.0)
    np.testing.assert_allclose(sigma[1, 1], 0.0)


def test_qah_small_faraday_matches_chern_alpha_relation():
    chern = 3
    small = qah_faraday_kerr_small_benchmark(chern, longitudinal_siemens=1.0e-6)
    alpha = E2_OVER_H_S * VACUUM_IMPEDANCE_OHM / 2.0
    np.testing.assert_allclose(small.faraday_angle_rad, np.arctan(chern * alpha), rtol=1.0e-14, atol=0.0)


def test_qah_exact_kerr_matches_pure_hall_sheet_limit():
    exact = qah_faraday_kerr_benchmark(1)
    alpha = E2_OVER_H_S * VACUUM_IMPEDANCE_OHM / 2.0
    # For a pure Hall sheet, Er_y/Er_x = -1/alpha, so the finite-alpha result
    # is close to, but not exactly, -pi/2.
    np.testing.assert_allclose(exact.kerr_angle_rad, -np.arctan(1.0 / alpha), rtol=0.0, atol=1.0e-9)


def test_c3_inplane_tensor_matches_liu_dai_eq3_signs():
    tensor = c3_inplane_second_order_tensor(2.0 + 1.0j, -0.5j)
    x = 0
    y = 1
    np.testing.assert_allclose(tensor[x, x, x], 2.0 + 1.0j)
    np.testing.assert_allclose(tensor[x, y, y], -(2.0 + 1.0j))
    np.testing.assert_allclose(tensor[y, x, y], -(2.0 + 1.0j))
    np.testing.assert_allclose(tensor[y, y, x], -(2.0 + 1.0j))
    np.testing.assert_allclose(tensor[y, x, x], -0.5j)
    np.testing.assert_allclose(tensor[x, x, y], -0.5j)
    np.testing.assert_allclose(tensor[x, y, x], -0.5j)
    np.testing.assert_allclose(tensor[y, y, y], 0.5j)


def test_c3_projection_recovers_independent_components_and_residual():
    tensor = c3_inplane_second_order_tensor(1.25, -0.75 + 0.2j)
    dec = decompose_c3_inplane_second_order_tensor(tensor)
    np.testing.assert_allclose(dec.sigma_xxx, 1.25)
    np.testing.assert_allclose(dec.sigma_yxx, -0.75 + 0.2j)
    np.testing.assert_allclose(dec.residual_norm, 0.0, atol=1.0e-15)

    perturbed = tensor.copy()
    perturbed[0, 0, 0] += 0.1
    dec_perturbed = decompose_c3_inplane_second_order_tensor(perturbed)
    assert dec_perturbed.residual_norm > 0.0
