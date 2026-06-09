from __future__ import annotations

import numpy as np

from mean_field.systems.htg import HTGModel, HTGParams, build_hamiltonian, build_htg_lattice, moire_coupling_matrix
from mean_field.systems.htg.hamiltonian import build_coupling_table, diagonalize_hamiltonian


def test_htg_hamiltonian_is_hermitian_for_small_cutoff() -> None:
    lattice = build_htg_lattice(1.5, n_shells=1)
    params = HTGParams.default()
    hmat = build_hamiltonian(lattice.gamma_m, lattice, params, valley=1)
    assert hmat.shape == (lattice.matrix_dim, lattice.matrix_dim)
    assert np.max(np.abs(hmat - hmat.conjugate().T)) < 1.0e-12


def test_htg_precomputed_coupling_tables_match_default_hamiltonian_and_eigenvalues() -> None:
    lattice = build_htg_lattice(1.5, n_shells=1)
    params = HTGParams.default()
    valley = -1
    k_tilde = lattice.gamma_m + 0.17 * lattice.b_m1 - 0.11 * lattice.b_m2
    d_top = -0.25 * lattice.delta
    d_bot = 0.50 * lattice.delta
    top_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=1)
    bottom_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=-1)

    default_hmat = build_hamiltonian(k_tilde, lattice, params, valley=valley, d_top=d_top, d_bot=d_bot)
    cached_hmat = build_hamiltonian(
        k_tilde,
        lattice,
        params,
        valley=valley,
        d_top=d_top,
        d_bot=d_bot,
        top_coupling_table=top_table,
        bottom_coupling_table=bottom_table,
    )
    default_evals, _ = diagonalize_hamiltonian(
        k_tilde,
        lattice,
        params,
        valley=valley,
        d_top=d_top,
        d_bot=d_bot,
        return_eigenvectors=False,
    )
    cached_evals, _ = diagonalize_hamiltonian(
        k_tilde,
        lattice,
        params,
        valley=valley,
        d_top=d_top,
        d_bot=d_bot,
        top_coupling_table=top_table,
        bottom_coupling_table=bottom_table,
        return_eigenvectors=False,
    )

    assert np.max(np.abs(default_hmat - cached_hmat)) < 1.0e-14
    assert np.max(np.abs(default_evals - cached_evals)) < 1.0e-12


def test_htg_moire_coupling_phase_anchors() -> None:
    params = HTGParams.default()
    t0 = moire_coupling_matrix(0, params, valley=1)
    t1 = moire_coupling_matrix(1, params, valley=1)
    t2 = moire_coupling_matrix(2, params, valley=1)
    assert np.max(np.abs(t0.imag)) < 1.0e-12
    assert np.max(np.abs(t1 - t2.conjugate())) < 1.0e-12


def test_htg_chern_basis_on_grid_uses_common_topology_core() -> None:
    model = HTGModel.from_config(1.5, n_shells=1, params=HTGParams.default())

    result = model.chern_basis_on_grid(3, valley=1)

    assert result.band_indices[0] < result.band_indices[1]
    assert result.rounded_total_chern == 0
    assert abs(result.total_chern - result.rounded_total_chern) < 1.0e-8
    assert result.sigma_z_eigenvalue_min < 0.0 < result.sigma_z_eigenvalue_max
