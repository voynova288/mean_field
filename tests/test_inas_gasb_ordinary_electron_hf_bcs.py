from __future__ import annotations

import numpy as np
import pytest

from ei_mf.gap_solver_generalized import (
    PAPER_LITERAL_LEGACY,
    paper_literal_eta_factor,
    paper_literal_xi_bracket,
    update_generalized_bcs_channels,
)
from mean_field.systems.inas_gasb.ordinary_electron_hf_bcs import (
    DERIVED_ORDINARY_ELECTRON_HF_BCS,
    derived_eta_fock_factor_from_electron_occupations,
    derived_xi_bracket,
    du_normalized_gap_from_fock,
    matrix_fermi_density,
    ordinary_electron_order_diagnostics,
    ordinary_electron_reference_density,
    two_band_hf_bcs_hamiltonian,
    two_band_ordinary_electron_state,
    validate_cartesian_2d_weights,
    validate_explicit_2d_momentum_mesh,
    validate_hartree_selection,
)
from mean_field.systems.inas_gasb.conventions import (
    E1H1BasisSpec,
)
from mean_field.systems.inas_gasb.matrix_ei import (
    MatrixEIConfig,
    relative_internal_energy_components_mev_nm2,
)


def test_t1_complex_delta_diagonalization_oracle() -> None:
    rng = np.random.default_rng(20260826)
    xi = rng.normal(size=64)
    eta = rng.normal(size=64)
    delta = rng.normal(size=64) + 1j * rng.normal(size=64)
    h = two_band_hf_bcs_hamiltonian(xi, eta, delta)
    pair_energy = np.sqrt(xi**2 + np.abs(delta) ** 2)
    expected = np.stack(((eta - pair_energy) / 2.0, (eta + pair_energy) / 2.0))
    actual = np.stack([np.linalg.eigvalsh(h[:, :, ik]) for ik in range(xi.size)]).T
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_t2_analytic_density_matches_matrix_fermi_function() -> None:
    rng = np.random.default_rng(31)
    xi = rng.uniform(-4.0, 4.0, size=23)
    eta = rng.uniform(-2.0, 2.0, size=23)
    delta = rng.normal(size=23) + 1j * rng.normal(size=23)
    state = two_band_ordinary_electron_state(
        xi, eta, delta, temperature_K=1.4
    )
    density, values = matrix_fermi_density(
        state.hamiltonian, electron_mu_mev=0.0, temperature_K=1.4
    )
    np.testing.assert_allclose(density, state.density, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(values[0], state.eigenvalue_minus, atol=1.0e-12)
    np.testing.assert_allclose(values[1], state.eigenvalue_plus, atol=1.0e-12)
    np.testing.assert_allclose(state.density[0, 1], state.coherence, atol=1.0e-13)
    np.testing.assert_allclose(
        state.density_delta[0, 0] - state.density_delta[1, 1],
        state.electron_occupation + state.hole_occupation,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        state.density_delta[0, 0] + state.density_delta[1, 1],
        state.electron_occupation - state.hole_occupation,
        atol=1.0e-13,
    )


def test_t3_reference_zero_and_bare_hybridization_is_not_order() -> None:
    nk = 2
    basis = E1H1BasisSpec()
    p_ref = ordinary_electron_reference_density(nk, basis)
    d = np.zeros_like(p_ref)
    sigma_h = np.zeros_like(p_ref)
    sigma_f = np.zeros_like(p_ref)
    h0 = np.zeros_like(p_ref)
    h0[0, 2] = 0.37
    h0[2, 0] = 0.37
    hmf = h0.copy()
    diagnostics = ordinary_electron_order_diagnostics(
        density=p_ref,
        density_delta=d,
        sigma_hartree_mev=sigma_h,
        sigma_fock_mev=sigma_f,
        h0_mev=h0,
        hmf_mev=hmf,
        k_weights_nm2=np.array([0.1, 0.2]),
        basis=basis,
    )
    np.testing.assert_array_equal(diagnostics.density_delta, 0.0)
    np.testing.assert_array_equal(diagnostics.sigma_hartree_mev, 0.0)
    np.testing.assert_array_equal(diagnostics.sigma_fock_mev, 0.0)
    np.testing.assert_array_equal(diagnostics.chi_eh, 0.0)
    np.testing.assert_array_equal(diagnostics.delta_du_eh_mev, 0.0)
    assert np.max(np.abs(diagnostics.hmf_eh_mev)) == pytest.approx(0.37)
    one_body, interaction, total = relative_internal_energy_components_mev_nm2(
        sigma_f, h0, d, np.array([0.1, 0.2])
    )
    assert one_body == interaction == total == 0.0


def test_t4_single_hole_has_negative_dbb_and_no_second_sign_flip() -> None:
    hole_occupation = 0.23
    d = np.zeros((2, 2, 1), dtype=np.complex128)
    d[1, 1, 0] = -hole_occupation
    repulsive_exchange_kernel = 5.0
    sigma_f = -repulsive_exchange_kernel * d
    assert d[1, 1, 0].real == pytest.approx(-hole_occupation)
    assert sigma_f[1, 1, 0].real == pytest.approx(
        repulsive_exchange_kernel * hole_occupation
    )
    hole_excitation_shift = -sigma_f[1, 1, 0].real
    assert hole_excitation_shift == pytest.approx(
        -repulsive_exchange_kernel * hole_occupation
    )


def test_t5_delta_zero_recovers_the_two_bare_electron_levels() -> None:
    xi = np.array([-2.0, 0.5, 3.0])
    eta = np.array([0.4, -1.0, 2.0])
    h = two_band_hf_bcs_hamiltonian(xi, eta, np.zeros(3, dtype=np.complex128))
    for ik in range(3):
        expected = np.sort([(eta[ik] + xi[ik]) / 2.0, (eta[ik] - xi[ik]) / 2.0])
        np.testing.assert_allclose(np.linalg.eigvalsh(h[:, :, ik]), expected, atol=1e-13)


def test_t6_relative_band_phase_covariance() -> None:
    xi = np.array([1.3])
    eta = np.array([-0.4])
    delta = np.array([0.7 - 0.2j])
    theta_a, theta_b = 0.37, -0.81
    phase = np.exp(1j * (theta_a - theta_b))
    gauge = np.diag([np.exp(1j * theta_a), np.exp(1j * theta_b)])
    state = two_band_ordinary_electron_state(xi, eta, delta, temperature_K=1.4)
    transformed = two_band_ordinary_electron_state(
        xi, eta, phase * delta, temperature_K=1.4
    )
    expected_h = gauge @ state.hamiltonian[:, :, 0] @ gauge.conj().T
    expected_p = gauge @ state.density[:, :, 0] @ gauge.conj().T
    np.testing.assert_allclose(transformed.hamiltonian[:, :, 0], expected_h, atol=1e-13)
    np.testing.assert_allclose(transformed.density[:, :, 0], expected_p, atol=1e-13)
    np.testing.assert_allclose(transformed.coherence, phase * state.coherence, atol=1e-13)
    np.testing.assert_allclose(transformed.pair_energy, state.pair_energy, atol=1e-13)
    np.testing.assert_allclose(np.abs(transformed.coherence), np.abs(state.coherence), atol=1e-13)
    np.testing.assert_allclose(
        du_normalized_gap_from_fock(-0.5 * phase * delta), phase * delta, atol=1e-13
    )

    # Repository frame gauge: Phi -> Phi G and local matrices M -> G^dagger M G.
    frame_phase = phase.conjugate()
    frame_transformed = two_band_ordinary_electron_state(
        xi, eta, frame_phase * delta, temperature_K=1.4
    )
    np.testing.assert_allclose(
        frame_transformed.hamiltonian[:, :, 0],
        gauge.conj().T @ state.hamiltonian[:, :, 0] @ gauge,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        frame_transformed.density[:, :, 0],
        gauge.conj().T @ state.density[:, :, 0] @ gauge,
        atol=1e-13,
    )


def test_t7_paper_xi_finite_temperature_difference_is_gminus_plus_gplus() -> None:
    xi = np.array([0.7, -1.2, 2.3])
    pair_energy = np.array([1.0, 1.5, 2.8])
    g_minus = np.array([0.0, 0.12, 0.31])
    g_plus = np.array([0.0, 0.08, 0.17])
    literal = paper_literal_xi_bracket(xi, pair_energy, g_minus, g_plus)
    derived = derived_xi_bracket(xi, pair_energy, g_minus, g_plus)
    np.testing.assert_allclose(derived - literal, g_minus + g_plus, atol=1e-14)
    np.testing.assert_allclose(literal[0], derived[0], atol=1e-14)


def test_t8_paper_eta_factor_fails_the_full_valence_reference() -> None:
    f_minus = np.array([1.0])
    f_plus = np.array([0.0])
    g_minus = 1.0 - f_minus
    g_plus = f_plus
    assert derived_eta_fock_factor_from_electron_occupations(f_minus, f_plus)[0] == 0.0
    assert paper_literal_eta_factor(g_minus, g_plus)[0] == 1.0


def test_t9_explicit_2d_weights_and_mode_isolation() -> None:
    dk = 0.2
    axis = np.array([-0.1, 0.1])
    kx, ky = np.meshgrid(axis, axis, indexing="ij")
    k = np.column_stack((kx.ravel(), ky.ravel()))
    sampled_area = (2.0 * dk) ** 2
    weights = np.full(4, dk * dk / (2.0 * np.pi) ** 2)
    validate_cartesian_2d_weights(
        k, weights, sampled_k_area_nm_minus2=sampled_area
    )
    assert validate_explicit_2d_momentum_mesh(k, weights) == pytest.approx(
        sampled_area
    )
    with pytest.raises(ValueError, match="genuinely two-dimensional"):
        validate_explicit_2d_momentum_mesh(
            np.column_stack((axis, np.zeros_like(axis))),
            np.full(2, dk * dk / (2.0 * np.pi) ** 2),
        )
    with pytest.raises(ValueError, match="sampled area"):
        validate_cartesian_2d_weights(
            k, weights * 0.5, sampled_k_area_nm_minus2=sampled_area
        )
    with pytest.raises(ValueError, match="projected Hartree"):
        validate_hartree_selection(
            projected_hartree_enabled=True, scalar_hartree_enabled=True
        )
    assert MatrixEIConfig().physics_mode == DERIVED_ORDINARY_ELECTRON_HF_BCS
    assert MatrixEIConfig().momentum_policy == "explicit_cartesian_2d"
    assert MatrixEIConfig().normal_ordering_reference_policy == "electron_hole_vacuum"

    ones = np.ones(1)
    with pytest.raises(ValueError, match="paper_literal_legacy"):
        update_generalized_bcs_channels(
            ones,
            ones,
            ones,
            np.zeros(1),
            ones,
            np.zeros(1),
            np.ones((1, 1)),
            np.ones((1, 1)),
            0.0,
            0.0,
            1.4,
            physics_mode=DERIVED_ORDINARY_ELECTRON_HF_BCS,  # type: ignore[arg-type]
        )
    literal = update_generalized_bcs_channels(
        ones,
        ones,
        ones,
        np.zeros(1),
        ones,
        np.zeros(1),
        np.ones((1, 1)),
        np.ones((1, 1)),
        0.0,
        0.0,
        1.4,
        physics_mode=PAPER_LITERAL_LEGACY,
    )
    assert all(np.all(np.isfinite(value)) for value in literal)
