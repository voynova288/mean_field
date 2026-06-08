from __future__ import annotations

import numpy as np

from mean_field.systems.tmbg import (
    TMBGParameters,
    blg_interlayer,
    build_coupling_table,
    build_diagonal_block,
    build_hamiltonian,
    diagonalize_hamiltonian,
    build_tmbg_lattice,
)


def test_tmbg_diagonal_block_matches_minimal_k_zero_limit() -> None:
    lattice = build_tmbg_lattice(1.21, n_shells=0)
    params = TMBGParameters.minimal()
    block = build_diagonal_block(0.0 + 0.0j, 0.0 + 0.0j, lattice, params, valley=1)

    expected = np.sort(
        np.asarray(
            [
                -params.t1,
                -params.vf * abs(lattice.q0),
                0.0,
                0.0,
                params.vf * abs(lattice.q0),
                params.t1,
            ],
            dtype=float,
        )
    )
    assert np.allclose(block, block.conjugate().T, atol=1.0e-12)
    assert np.allclose(np.linalg.eigvalsh(block), expected, atol=1.0e-12)


def test_tmbg_gate_potential_changes_layer_traces_as_expected() -> None:
    lattice = build_tmbg_lattice(1.21, n_shells=0)
    block_zero = build_diagonal_block(0.0 + 0.0j, 0.0 + 0.0j, lattice, TMBGParameters.full(), valley=1)
    block_shifted = build_diagonal_block(
        0.0 + 0.0j,
        0.0 + 0.0j,
        lattice,
        TMBGParameters.full(interlayer_potential=0.06, staggered_potential=0.01),
        valley=1,
    )

    bottom_zero = np.trace(block_zero[0:2, 0:2]).real
    top_zero = np.trace(block_zero[4:6, 4:6]).real
    bottom_shifted = np.trace(block_shifted[0:2, 0:2]).real
    top_shifted = np.trace(block_shifted[4:6, 4:6]).real

    assert np.isclose(bottom_shifted - bottom_zero, -0.12, atol=1.0e-12)
    assert np.isclose(top_shifted - top_zero, 0.12, atol=1.0e-12)


def test_tmbg_delta_shifts_code_gauge_blg_orbitals() -> None:
    lattice = build_tmbg_lattice(1.21, n_shells=0)
    params = TMBGParameters.full(interlayer_potential=0.0, staggered_potential=0.0)
    block = build_diagonal_block(0.0 + 0.0j, 0.0 + 0.0j, lattice, params, valley=1)

    diagonal = np.diag(block).real

    assert np.isclose(diagonal[0], 0.0, atol=1.0e-12)
    assert np.isclose(diagonal[1], params.delta, atol=1.0e-12)
    assert np.isclose(diagonal[2], params.delta, atol=1.0e-12)
    assert np.isclose(diagonal[3], 0.0, atol=1.0e-12)
    assert np.allclose(diagonal[4:6], 0.0, atol=1.0e-12)


def test_tmbg_coupling_table_matches_q_shift_rule() -> None:
    lattice = build_tmbg_lattice(1.21, n_shells=2)
    table = build_coupling_table(lattice.g_vectors, lattice.q_vectors)

    for entry in table:
        shift = lattice.g_vectors[entry.top_index] - lattice.g_vectors[entry.middle_index]
        expected = lattice.q_vectors[entry.channel] - lattice.q0
        assert abs(shift - expected) < 1.0e-12


def test_tmbg_blg_interlayer_matches_standard_mccann_koshino_structure() -> None:
    params = TMBGParameters.full()
    kvec = 0.17 + 0.23j
    phi = -0.31
    coupling = blg_interlayer(kvec, phi, params, valley=1)

    q = kvec * np.exp(-1j * phi)
    expected = np.asarray(
        [
            [-params.v4 * q.conjugate(), -params.v3 * q],
            [params.t1, -params.v4 * q.conjugate()],
        ],
        dtype=np.complex128,
    )
    assert np.allclose(coupling, expected, atol=1.0e-12)


def test_tmbg_full_hamiltonian_is_hermitian_and_time_reversal_symmetric() -> None:
    lattice = build_tmbg_lattice(1.21, n_shells=1)
    params = TMBGParameters.full(interlayer_potential=0.02, staggered_potential=0.0)
    k_tilde = lattice.k_m / 5.0 + lattice.m_m / 7.0

    hamiltonian = build_hamiltonian(k_tilde, lattice, params, valley=1)
    assert hamiltonian.shape == (lattice.matrix_dim, lattice.matrix_dim)
    assert np.allclose(hamiltonian, hamiltonian.conjugate().T, atol=1.0e-10)

    evals_k, _ = diagonalize_hamiltonian(k_tilde, lattice, params, valley=1, n_bands=12)
    evals_kprime, _ = diagonalize_hamiltonian(-k_tilde, lattice, params, valley=-1, n_bands=12)
    assert np.allclose(evals_k, evals_kprime, atol=1.0e-10)
