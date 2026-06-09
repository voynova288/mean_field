from __future__ import annotations

import numpy as np

from mean_field.systems.atmg import (
    ATMGParameters,
    analytic_singular_values,
    build_W_matrix,
    build_coupling_table,
    build_hamiltonian,
    build_tbg_hamiltonian,
    build_atmg_lattice,
    diagonalize_hamiltonian,
    svd_decompose,
)


def test_atmg_uniform_svd_matches_analytic_formula() -> None:
    alpha = 0.37
    w_matrix = build_W_matrix(5, alpha)
    svd_result = svd_decompose(w_matrix)
    analytic = analytic_singular_values(5, alpha)

    assert np.allclose(svd_result.singular_values, analytic, atol=1.0e-12)
    assert np.allclose(svd_result.left_unitary.conjugate().T @ svd_result.left_unitary, np.eye(3), atol=1.0e-12)
    assert np.allclose(svd_result.right_unitary.conjugate().T @ svd_result.right_unitary, np.eye(2), atol=1.0e-12)
    assert np.allclose(svd_result.reconstruction, w_matrix, atol=1.0e-12)


def test_n2_atmg_hamiltonian_reduces_exactly_to_tbg_builder() -> None:
    lattice = build_atmg_lattice(1.10, n_shells=1)
    params = ATMGParameters.realistic(2, 1.10, kappa=0.8)
    k_tilde = lattice.k_m / 5.0 + lattice.m_m / 9.0

    atmg_h = build_hamiltonian(k_tilde, lattice, params, valley=1)
    tbg_h = build_tbg_hamiltonian(
        k_tilde,
        lattice,
        lambda_coupling=params.alpha,
        kappa=params.kappa,
        vf=params.vf,
        valley=1,
    )

    assert atmg_h.shape == (2 * params.n_layers * lattice.n_g, 2 * params.n_layers * lattice.n_g)
    assert np.allclose(atmg_h, atmg_h.conjugate().T, atol=1.0e-10)
    assert np.allclose(atmg_h, tbg_h, atol=1.0e-12)


def test_precomputed_coupling_table_matches_default_hamiltonian_and_spectrum() -> None:
    lattice = build_atmg_lattice(1.53, n_shells=1)
    params = ATMGParameters.chiral(3, 1.53)
    valley = -1
    k_tilde = lattice.k_m / 6.0 - lattice.m_m / 8.0
    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)

    default_h = build_hamiltonian(k_tilde, lattice, params, valley=valley)
    cached_h = build_hamiltonian(k_tilde, lattice, params, valley=valley, coupling_table=coupling_table)
    default_evals, _ = diagonalize_hamiltonian(k_tilde, lattice, params, valley=valley, n_bands=8)
    cached_evals, _ = diagonalize_hamiltonian(
        k_tilde,
        lattice,
        params,
        valley=valley,
        n_bands=8,
        coupling_table=coupling_table,
    )

    assert np.allclose(cached_h, default_h, atol=1.0e-12)
    assert np.allclose(cached_evals, default_evals, atol=1.0e-12)


def test_n3_atmg_hamiltonian_is_hermitian_and_time_reversal_symmetric() -> None:
    lattice = build_atmg_lattice(1.53, n_shells=1)
    params = ATMGParameters.chiral(3, 1.53)
    k_tilde = lattice.k_m / 4.0 + lattice.m_m / 7.0

    h_k = build_hamiltonian(k_tilde, lattice, params, valley=1)
    h_kprime = build_hamiltonian(-k_tilde, lattice, params, valley=-1)

    assert np.allclose(h_k, h_k.conjugate().T, atol=1.0e-10)
    assert np.allclose(np.linalg.eigvalsh(h_k), np.linalg.eigvalsh(h_kprime), atol=1.0e-10)
