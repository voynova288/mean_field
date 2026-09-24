from __future__ import annotations

import numpy as np
from ei_mf.bands import RadialBandData, epsilon_P_M_from_band_data
from ei_mf.gap_solver_generalized import (
    bare_xi_eta_channels,
    carrier_densities_cm2,
    carrier_occupations,
    chemical_potential_channels,
    pair_density_cm2,
    solve_generalized_cnp_root,
    solve_generalized_fixed_cnp,
    solve_generalized_fixed_density,
    solve_generalized_normal_cnp_root,
    solve_generalized_fixed_mu,
    solve_generalized_fixed_mu_root,
    update_generalized_bcs,
    update_generalized_bcs_channels,
)


def test_chemical_potential_channel_conventions_and_nambu_oracle() -> None:
    mu = 0.7
    assert chemical_potential_channels(mu, "literal_supplement") == (mu, mu)
    assert chemical_potential_channels(mu, "eq1_common_carrier") == (2.0 * mu, 0.0)
    assert chemical_potential_channels(mu, "physical_electron_hole") == (0.0, 2.0 * mu)

    eps_a, eps_b = 1.8, 0.9
    mu_a, mu_b = 0.4, -0.2
    A, B = eps_a - mu_a, eps_b - mu_b
    mu_sum, mu_diff = mu_a + mu_b, mu_a - mu_b
    xi, eta = bare_xi_eta_channels(
        np.array([eps_a + eps_b]), np.array([eps_a - eps_b]), mu_sum, mu_diff
    )
    Delta = 0.6
    nambu = np.array([[A, -Delta / 2.0], [-Delta / 2.0, -B]])
    E = np.sqrt(xi[0] ** 2 + Delta**2)
    expected = np.sort(np.array([(eta[0] - E) / 2.0, (eta[0] + E) / 2.0]))
    assert np.allclose(np.linalg.eigvalsh(nambu), expected, atol=1e-14)


def test_literal_update_is_exact_channel_wrapper() -> None:
    w = np.array([0.2, 0.3])
    Delta = np.array([0.5, 0.7])
    xi = np.array([1.0, -0.8])
    eta = np.array([0.1, -0.2])
    eps_P = np.array([1.2, -0.5])
    eps_M = np.array([0.3, 0.4])
    K_ab = np.eye(2)
    K_aa = 0.5 * np.eye(2)
    literal = update_generalized_bcs(
        w, Delta, xi, eta, eps_P, eps_M, K_ab, K_aa, 0.05, 2.0
    )
    explicit = update_generalized_bcs_channels(
        w, Delta, xi, eta, eps_P, eps_M, K_ab, K_aa, 0.05, 0.05, 2.0
    )
    for a, b in zip(literal, explicit):
        assert np.array_equal(a, b)


def test_finite_temperature_occupation_identities() -> None:
    Delta = np.array([0.7, 1.1])
    xi = np.array([0.4, -0.6])
    eta = np.array([0.1, -0.2])
    n_e, n_h = carrier_occupations(Delta, xi, eta, 3.0)
    E = np.sqrt(Delta**2 + xi**2)
    from ei_mf.fermi import fermi_mev
    f_minus = fermi_mev((E - eta) / 2.0, 3.0)
    f_plus = fermi_mev((E + eta) / 2.0, 3.0)
    F = 1.0 - f_minus - f_plus
    assert np.allclose(n_e + n_h, 1.0 - xi / E * F)
    assert np.allclose(n_e - n_h, f_plus - f_minus)



def test_band_csv_uses_hole_energy_convention(tmp_path) -> None:
    path = tmp_path / "bands.csv"
    path.write_text(
        "k_nm_inv,epsilon_a_meV,epsilon_b_meV\n"
        "0.00,1.0,2.0\n"
        "0.05,2.0,3.5\n"
        "0.10,4.0,6.0\n"
        "0.15,7.0,9.0\n",
        encoding="utf-8",
    )
    bands = RadialBandData.from_csv(path)
    eps_P, eps_M = epsilon_P_M_from_band_data(bands, np.array([0.025, 0.075]))
    assert np.allclose(eps_P, [4.25, 7.75])
    assert np.allclose(eps_M, [-1.25, -1.75])


def test_generalized_update_matches_supplementary_factor_convention() -> None:
    w = np.array([0.2, 0.3])
    Delta = np.array([0.5, 0.7])
    xi = np.array([1.0, -0.8])
    eta = np.array([0.1, -0.2])
    eps_P = np.array([1.2, -0.5])
    eps_M = np.array([0.3, 0.4])
    K_ab = np.eye(2)
    K_aa = 0.5 * np.eye(2)
    out = update_generalized_bcs(w, Delta, xi, eta, eps_P, eps_M, K_ab, K_aa, 0.05, 0.0)
    E = np.sqrt(Delta**2 + xi**2)
    # At T=0 with small |eta|<E, f((E±eta)/2)=0, so F_pair=1.
    assert np.allclose(out[0], K_ab @ (w * Delta / E))
    assert np.allclose(out[1], eps_P - 0.05 - K_aa @ (w * (1 - xi / E)))
    assert np.allclose(out[2], eps_M - 0.05 - K_ab @ w)


def test_generalized_update_thermodynamic_xi_convention() -> None:
    w = np.array([0.2])
    Delta = np.array([0.7])
    xi = np.array([0.4])
    eta = np.array([0.1])
    eps_P = np.array([1.2])
    eps_M = np.array([0.3])
    K_ab = np.array([[1.0]])
    K_aa = np.array([[0.5]])
    out = update_generalized_bcs(
        w, Delta, xi, eta, eps_P, eps_M, K_ab, K_aa, 0.05, 3.0,
        xi_thermal_convention="thermodynamic",
    )
    E = np.sqrt(Delta**2 + xi**2)
    from ei_mf.thermodynamics import pair_thermal_factor
    F = pair_thermal_factor(E, eta, 3.0)
    expected_xi = eps_P - 0.05 - K_aa @ (w * (1.0 - xi / E * F))
    assert np.allclose(out[1], expected_xi)


def test_generalized_fixed_mu_solver_toy_converges() -> None:
    k = np.linspace(0.01, 0.04, 4)
    w = np.ones(4) * 0.05
    eps_P = np.linspace(-0.1, 0.1, 4)
    eps_M = np.zeros(4)
    K_ab = np.eye(4) * 3.0
    K_aa = np.eye(4) * 0.1
    result = solve_generalized_fixed_mu(
        k,
        w,
        eps_P,
        eps_M,
        K_ab,
        K_aa,
        temperature_K=0.1,
        mix=0.25,
        max_iter=2000,
        tol_meV=1e-9,
    )
    assert result.converged
    assert np.allclose(result.E_pair_mev, np.sqrt(result.Delta_order_mev**2 + result.xi_mev**2))
    assert result.mu_mode == "fixed_mu"
    assert result.density_final_cm2 is not None


def test_generalized_fixed_mu_root_matches_converged_fixed_point() -> None:
    k = np.linspace(0.01, 0.04, 4)
    w = np.ones(4) * 0.05
    eps_P = np.linspace(-0.1, 0.1, 4)
    eps_M = np.zeros(4)
    K_ab = np.eye(4) * 3.0
    K_aa = np.eye(4) * 0.1
    seed = solve_generalized_fixed_mu(
        k, w, eps_P, eps_M, K_ab, K_aa,
        temperature_K=0.1, mix=0.25, max_iter=2000, tol_meV=1e-9,
    )
    result = solve_generalized_fixed_mu_root(
        k, w, eps_P, eps_M, K_ab, K_aa,
        temperature_K=0.1,
        initial_delta_mev=seed.Delta_order_mev,
        initial_xi_mev=seed.xi_mev,
        initial_eta_mev=seed.eta_mev,
        tol_meV=1e-9,
    )
    assert result.converged
    assert result.mu_mode == "fixed_mu_root"
    assert np.allclose(result.Delta_order_mev, seed.Delta_order_mev, atol=1e-7)
    assert np.allclose(result.xi_mev, seed.xi_mev, atol=1e-7)
    assert np.allclose(result.eta_mev, seed.eta_mev, atol=1e-7)


def test_pair_density_zero_temperature_bcs_limit() -> None:
    edges = np.linspace(0.0, 0.1, 101)
    k = 0.5 * (edges[:-1] + edges[1:])
    w = (edges[1:] ** 2 - edges[:-1] ** 2) / (4.0 * np.pi)
    kF = 0.05
    xi = k * k - kF * kF
    Delta = np.zeros_like(k)
    eta = np.zeros_like(k)
    density = pair_density_cm2(w, Delta, xi, eta, 0.0)
    n_e, n_h = carrier_densities_cm2(w, Delta, xi, eta, 0.0)
    expected = 2.0 * kF * kF / (4.0 * np.pi) * 1e14
    assert np.isclose(density, expected, rtol=2e-2)
    assert np.isclose(n_e, expected, rtol=2e-2)
    assert np.isclose(n_h, expected, rtol=2e-2)


def test_generalized_fixed_density_matches_fixed_mu_density() -> None:
    k = np.linspace(0.01, 0.04, 4)
    w = np.ones(4) * 0.02
    eps_P = np.linspace(-0.2, 0.2, 4)
    eps_M = np.linspace(0.05, -0.05, 4)
    K_ab = np.eye(4) * 2.5
    K_aa = np.eye(4) * 0.05
    reference = solve_generalized_fixed_mu(
        k,
        w,
        eps_P,
        eps_M,
        K_ab,
        K_aa,
        mu_mev=0.03,
        temperature_K=0.1,
        mix=0.25,
        max_iter=2000,
        tol_meV=1e-9,
    )
    result = solve_generalized_fixed_density(
        k,
        w,
        eps_P,
        eps_M,
        K_ab,
        K_aa,
        density_target_cm2=reference.density_final_cm2,
        temperature_K=0.1,
        mix=0.25,
        max_iter=2000,
        tol_meV=1e-8,
        density_tol_cm2=5e8,
        max_mu_iter=20,
        mu_min_meV=-1.0,
        mu_max_meV=1.0,
    )
    assert result.mu_mode == "fixed_density"
    assert result.converged
    assert abs(result.density_residual_cm2) < 5e8


def test_cnp_krylov_root_solves_all_equations_together() -> None:
    k = np.linspace(0.01, 0.06, 8)
    w = np.ones(8) * 0.004
    eps_P = np.linspace(-1.5, 1.5, 8)
    eps_M = np.zeros(8)
    K_ab = np.zeros((8, 8))
    K_aa = np.zeros((8, 8))
    reference = solve_generalized_fixed_mu(
        k, w, eps_P, eps_M + 1.0, K_ab, K_aa,
        mu_mev=1.0, temperature_K=1.0, mix=0.5, max_iter=1000, tol_meV=1e-10,
    )
    result = solve_generalized_cnp_root(
        k, w, eps_P, eps_M, K_ab, K_aa,
        density_target_cm2=reference.density_final_cm2,
        temperature_K=1.0, initial_mu_mev=0.8, initial_eps_M_offset_mev=0.8,
        initial_delta_mev=np.full_like(k, 0.1), tol_meV=1e-8,
        density_tol_cm2=5e8, max_iter=200,
    )
    assert result.converged
    assert abs(result.electron_density_cm2 - reference.density_final_cm2) < 5e8
    assert abs(result.hole_density_cm2 - reference.density_final_cm2) < 5e8
    assert np.isclose(result.eps_M_offset_mev, result.mu_mev, atol=5e-2)
    assert result.chemical_potential_convention == "explicit_channels"
    assert result.eps_M_frame == "input"
    assert np.isclose(result.mu_sum_mev, result.mu_mev)
    assert np.isclose(result.mu_diff_mev, result.mu_mev - result.eps_M_offset_mev)
    rhs = update_generalized_bcs_channels(
        result.weights_nm2, result.Delta_order_mev, result.xi_mev, result.eta_mev,
        result.eps_P_mev, result.eps_M_mev, K_ab, K_aa,
        result.mu_sum_mev, result.mu_diff_mev, 1.0,
    )
    assert max(np.max(np.abs(a - b)) for a, b in zip(
        (result.Delta_order_mev, result.xi_mev, result.eta_mev), rhs
    )) < 1e-8
    normal = solve_generalized_normal_cnp_root(
        k, w, eps_P, eps_M, K_ab, K_aa,
        density_target_cm2=reference.density_final_cm2,
        temperature_K=1.0, initial_mu_mev=0.8, initial_eps_M_offset_mev=0.8,
        tol_meV=1e-8, density_tol_cm2=5e8, max_iter=200,
    )
    assert normal.converged
    assert normal.eps_M_frame == "input"
    assert np.all(normal.Delta_order_mev == 0.0)
    assert np.isclose(normal.mu_sum_mev, normal.mu_mev)
    assert np.isclose(normal.mu_diff_mev, normal.mu_mev - normal.eps_M_offset_mev)
    normal_rhs = update_generalized_bcs_channels(
        normal.weights_nm2, normal.Delta_order_mev, normal.xi_mev, normal.eta_mev,
        normal.eps_P_mev, normal.eps_M_mev, K_ab, K_aa,
        normal.mu_sum_mev, normal.mu_diff_mev, 1.0,
    )
    assert max(np.max(np.abs(a - b)) for a, b in zip(
        (normal.Delta_order_mev, normal.xi_mev, normal.eta_mev), normal_rhs
    )) < 1e-8


def test_fixed_cnp_solves_density_and_charge_alignment() -> None:
    k = np.linspace(0.01, 0.06, 8)
    w = np.ones(8) * 0.004
    eps_P = np.linspace(-1.5, 1.5, 8)
    eps_M = np.zeros(8)
    K_ab = np.zeros((8, 8))
    K_aa = np.zeros((8, 8))
    reference = solve_generalized_fixed_mu(
        k, w, eps_P, eps_M + 1.0, K_ab, K_aa,
        mu_mev=1.0, temperature_K=1.0, mix=0.5, max_iter=1000, tol_meV=1e-10,
    )
    assert abs(reference.charge_imbalance_cm2) < 1e-3
    result = solve_generalized_fixed_cnp(
        k, w, eps_P, eps_M, K_ab, K_aa,
        density_target_cm2=reference.density_final_cm2,
        temperature_K=1.0, mix=0.5, max_iter=1000, tol_meV=1e-8,
        density_tol_cm2=5e8, charge_balance_tol_cm2=5e8,
        max_mu_iter=28, max_alignment_iter=28,
        eps_M_offset_min_mev=-2.0, eps_M_offset_max_mev=2.0,
    )
    assert result.converged
    assert abs(result.density_residual_cm2) < 5e8
    assert abs(result.charge_imbalance_cm2) < 5e8
    assert np.isclose(result.eps_M_offset_mev, result.mu_mev, atol=5e-2)
    assert result.eps_M_frame == "aligned"
    legacy_rhs = update_generalized_bcs_channels(
        result.weights_nm2, result.Delta_order_mev, result.xi_mev, result.eta_mev,
        result.eps_P_mev, result.eps_M_mev, K_ab, K_aa,
        result.mu_sum_mev, result.mu_diff_mev, 1.0,
    )
    assert max(np.max(np.abs(a - b)) for a, b in zip(
        (result.Delta_order_mev, result.xi_mev, result.eta_mev), legacy_rhs
    )) < 1e-8


def test_fixed_density_expands_mu_bracket_in_correct_direction() -> None:
    k = np.linspace(0.01, 0.05, 6)
    w = np.ones(6) * 0.01
    eps_P = np.linspace(-1.0, 1.0, 6)
    eps_M = np.zeros(6)
    K_ab = np.eye(6) * 0.5
    K_aa = np.eye(6) * 0.02
    reference = solve_generalized_fixed_mu(
        k, w, eps_P, eps_M, K_ab, K_aa,
        mu_mev=1.0, temperature_K=0.5, mix=0.3, max_iter=4000, tol_meV=1e-9,
    )
    result = solve_generalized_fixed_density(
        k, w, eps_P, eps_M, K_ab, K_aa,
        density_target_cm2=reference.density_final_cm2,
        temperature_K=0.5, mix=0.3, max_iter=4000, tol_meV=1e-8,
        density_tol_cm2=5e8, max_mu_iter=24,
        mu_min_meV=-0.1, mu_max_meV=0.1,
    )
    assert result.converged
    assert result.mu_mev > 0.1
    assert abs(result.density_residual_cm2) < 5e8
