from __future__ import annotations

import numpy as np

from ei_mf.grids import radial_grid
from ei_mf.observables import compute_radial_jdos
from ei_mf.pair_breaking import pair_breaking_weights


def test_nambu_current_matrix_element_matches_coherence_formula() -> None:
    xi, Delta, eta = 2.3, 1.7, -0.4
    v_e, v_h = 3.2, 1.1
    h = 0.5 * np.array([[eta + xi, Delta], [Delta, eta - xi]])
    _eigenvalues, eigenvectors = np.linalg.eigh(h)
    current = np.diag([-v_e, v_h])
    matrix_element = np.vdot(eigenvectors[:, 1], current @ eigenvectors[:, 0])
    E = np.hypot(xi, Delta)
    expected = 0.25 * (v_e + v_h) ** 2 * Delta**2 / E**2
    assert np.isclose(abs(matrix_element) ** 2, expected)


def test_parabolic_pair_velocity_and_weights() -> None:
    k, _weights, _edges = radial_grid(0.12, 180)
    alpha = 1450.0
    eps_P = 1.2 + alpha * k**2
    Delta = np.full_like(k, 2.0)
    xi = eps_P - 8.0
    eta = np.zeros_like(k)
    result = pair_breaking_weights(k, eps_P, Delta, xi, eta, 0.1)
    expected_velocity = 2.0 * alpha * k
    assert np.allclose(
        result.radial_band_velocity_mev_nm[2:-2],
        expected_velocity[2:-2],
        rtol=3e-4,
        atol=2e-4,
    )
    assert np.all(result.coherence_weight >= 0.0)
    assert np.all(result.current_coherence_weight >= 0.0)
    assert result.current_coherence_weight[0] < result.current_coherence_weight[20]


def test_weighted_radial_spectrum_is_finite_and_nonnegative() -> None:
    k, radial_weights, _edges = radial_grid(0.12, 80)
    eps_P = 1200.0 * (k**2 - 0.05**2)
    Delta = 2.0 * np.exp(-0.5 * ((k - 0.05) / 0.03) ** 2)
    xi = eps_P.copy()
    eta = np.zeros_like(k)
    E = np.sqrt(xi**2 + Delta**2)
    optical = pair_breaking_weights(k, eps_P, Delta, xi, eta, 0.1)
    omega, spectrum = compute_radial_jdos(
        k,
        E,
        radial_weights,
        0.1,
        omega_min_mev=0.0,
        omega_max_mev=10.0,
        nomega=800,
        spectral_weight=optical.conductivity_shape_weight,
    )
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0.0)
    assert np.trapezoid(spectrum, omega) > 0.0
