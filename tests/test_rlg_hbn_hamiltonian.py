from __future__ import annotations

import numpy as np

from mean_field.systems.RnG_hBN import (
    RLGhBNParams,
    build_coupling_table,
    build_hamiltonian,
    build_rlg_hbn_lattice,
    diagonalize_hamiltonian,
    flat_band_indices,
    layer_slice,
    moire_coupling_matrix,
    moire_potential,
    valence_band_count,
)


def test_rlg_hbn_moire_matrices_are_hermitian_by_delta_pairing() -> None:
    params = RLGhBNParams.from_table(layer_count=5, xi=1)

    plus = moire_potential((1, 0), (0, 0), params)
    minus = moire_potential((0, 0), (1, 0), params)

    assert moire_coupling_matrix(1, params).shape == (2, 2)
    assert np.allclose(plus, minus.conjugate().T, atol=1.0e-12)
    assert np.allclose(moire_potential((0, 0), (0, 0), params), params.moire_v0_mev * np.eye(2), atol=1.0e-12)
    assert np.count_nonzero(np.abs(moire_potential((2, 0), (0, 0), params)) > 0.0) == 0


def test_rlg_hbn_coupling_table_matches_dense_moire_pair_scan() -> None:
    params = RLGhBNParams.from_table(layer_count=4, xi=1, displacement_field_mev=12.0)
    no_moire = RLGhBNParams.without_moire(layer_count=4, xi=1, displacement_field_mev=12.0)
    lattice = build_rlg_hbn_lattice(theta_deg=0.77, shell_count=2, layer_count=params.layer_count)
    k_tilde = lattice.k_m / 6.0 + lattice.m_m / 8.0

    dense_moire = np.zeros((2 * params.layer_count * lattice.n_g, 2 * params.layer_count * lattice.n_g), dtype=np.complex128)
    for row_g_index, row_coords in enumerate(lattice.g_indices):
        row_slice = layer_slice(row_g_index, 0, params)
        for col_g_index, col_coords in enumerate(lattice.g_indices):
            col_slice = layer_slice(col_g_index, 0, params)
            dense_moire[row_slice, col_slice] += moire_potential(row_coords, col_coords, params)

    sparse_moire = build_hamiltonian(k_tilde, lattice, params) - build_hamiltonian(k_tilde, lattice, no_moire)

    assert len(build_coupling_table(lattice)) < lattice.n_g * lattice.n_g
    assert np.max(np.abs(sparse_moire - dense_moire)) < 1.0e-12


def test_rlg_hbn_hamiltonian_is_hermitian_and_moire_only_touches_bottom_layer() -> None:
    params = RLGhBNParams.from_table(layer_count=3, xi=1, displacement_field_mev=24.0)
    no_moire = RLGhBNParams.without_moire(layer_count=3, xi=1, displacement_field_mev=24.0)
    lattice = build_rlg_hbn_lattice(theta_deg=0.77, shell_count=2, layer_count=params.layer_count)
    k_tilde = lattice.k_m / 5.0 + lattice.m_m / 9.0

    hamiltonian = build_hamiltonian(k_tilde, lattice, params)
    bare = build_hamiltonian(k_tilde, lattice, no_moire)
    moire = hamiltonian - bare

    assert hamiltonian.shape == (2 * params.layer_count * lattice.n_g, 2 * params.layer_count * lattice.n_g)
    assert np.max(np.abs(hamiltonian - hamiltonian.conjugate().T)) < 1.0e-12

    mask = np.ones(moire.shape, dtype=bool)
    for row_g_index in range(lattice.n_g):
        row_slice = layer_slice(row_g_index, 0, params)
        for col_g_index in range(lattice.n_g):
            col_slice = layer_slice(col_g_index, 0, params)
            mask[row_slice, col_slice] = False
    assert np.max(np.abs(moire[mask])) < 1.0e-12
    assert np.max(np.abs(moire[~mask])) > 0.0


def test_rlg_hbn_time_reversal_and_flat_band_count_conventions() -> None:
    params = RLGhBNParams.from_table(layer_count=3, xi=0, displacement_field_mev=20.0)
    lattice = build_rlg_hbn_lattice(theta_deg=0.77, shell_count=2, layer_count=params.layer_count)
    k_tilde = lattice.k_m / 7.0 + lattice.m_m / 11.0

    evals_k, _ = diagonalize_hamiltonian(k_tilde, lattice, params, valley=1)
    evals_kprime, _ = diagonalize_hamiltonian(-k_tilde, lattice, params, valley=-1)

    assert np.max(np.abs(evals_k - evals_kprime)) < 1.0e-10
    assert valence_band_count(lattice, params) == params.layer_count * lattice.n_g
    assert flat_band_indices(lattice, params) == (params.layer_count * lattice.n_g - 1, params.layer_count * lattice.n_g)
