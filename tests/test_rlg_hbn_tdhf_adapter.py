from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.core.hf import ParticleHolePair
from mean_field.devtools.run_rlg_hbn_tdhf_q0 import _shortcut_decision
from mean_field.systems.RnG_hBN import (
    RLGhBNHartreeFockRun,
    RLGhBNHartreeFockState,
    RLGhBNInteractionParams,
    RLGhBNModel,
    RLGhBNTDHFInteraction,
    RLGhBNTDHFOrbitals,
    build_rlg_hbn_hf_interaction_hamiltonian,
    build_rlg_hbn_hf_problem,
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_projected_basis,
    build_rlg_hbn_tdhf_interaction,
    build_rlg_hbn_tdhf_orbitals,
    build_rlg_hbn_tdhf_q0_matrices,
    build_rlg_hbn_tdhf_q0_matrices_from_pairs,
    build_rlg_hbn_tdhf_q0_pairs,
    rlg_hbn_flavor_occupation_counts_for_init_mode,
)


def _tiny_flavor_polarized_run() -> RLGhBNHartreeFockRun:
    model = RLGhBNModel.from_config(
        layer_count=3,
        xi=1,
        theta_deg=0.77,
        displacement_field_mev=24.0,
        shell_count=1,
    )
    interaction = RLGhBNInteractionParams(
        active_valence_bands=0,
        active_conduction_bands=1,
        k_mesh_size=1,
        interaction_cutoff_q1=1.0,
        use_screened_basis=False,
    )
    basis_data = build_rlg_hbn_projected_basis(model, interaction, mesh_size=1)
    overlap_blocks = build_rlg_hbn_layer_overlap_blocks(basis_data, shifts=((0, 0),))
    counts = rlg_hbn_flavor_occupation_counts_for_init_mode(
        "flavor",
        nu=1.0,
        active_valence_bands=basis_data.interaction.active_valence_bands,
        n_spin=basis_data.basis.n_spin,
        n_eta=basis_data.basis.n_flavor,
        n_band=basis_data.basis.n_band,
    )
    state = RLGhBNHartreeFockState.from_projected_basis(
        basis_data,
        nu=1.0,
        occupation_counts=counts,
    )
    problem = build_rlg_hbn_hf_problem(state, overlap_blocks)
    problem.initializer(state, init_mode="flavor", seed=1)
    update = problem.kernel.density_builder(state.h0)
    state.density[:, :, :] = update.density
    state.hamiltonian[:, :, :] = state.h0
    state.energies[:, :] = update.energies
    state.mu = update.mu
    return RLGhBNHartreeFockRun(
        state=state,
        iter_energy=(),
        iter_err=(),
        iter_oda=(),
        init_mode="flavor",
        seed=1,
        converged=False,
        exit_reason="tdhf-adapter-smoke",
        overlap_blocks=overlap_blocks,
        basis_data=basis_data,
    )


def test_rlg_hbn_tdhf_orbitals_and_q0_pairs_keep_fixed_momentum_sector() -> None:
    run = _tiny_flavor_polarized_run()
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    pairs = build_rlg_hbn_tdhf_q0_pairs(orbitals)

    assert orbitals.global_energies.shape == (run.state.nt * run.state.nk,)
    assert int(np.count_nonzero(orbitals.occupied_mask)) == 1
    assert len(pairs) == 3
    for pair in pairs:
        assert pair.particle_momentum == pair.hole_momentum == 0
        assert orbitals.decode_global_index(pair.particle)[1] == orbitals.decode_global_index(pair.hole)[1]


def test_rlg_hbn_tdhf_interaction_callable_and_dense_q0_smoke() -> None:
    run = _tiny_flavor_polarized_run()
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    pairs = build_rlg_hbn_tdhf_q0_pairs(orbitals)
    interaction = build_rlg_hbn_tdhf_interaction(run, orbitals)

    value = interaction(pairs[0].particle, pairs[0].hole, pairs[0].hole, pairs[0].particle)
    assert np.isfinite(value.real)
    assert np.isfinite(value.imag)

    matrices = build_rlg_hbn_tdhf_q0_matrices(run, max_pairs=8)
    assert matrices.A.shape == (3, 3)
    assert matrices.B.shape == (3, 3)
    assert matrices.L.shape == (6, 6)
    assert matrices.structure.ok


def test_rlg_hbn_tdhf_vectorized_q0_assembly_matches_generic_callable_path() -> None:
    run = _tiny_flavor_polarized_run()
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    pairs = build_rlg_hbn_tdhf_q0_pairs(orbitals)
    vectorized = build_rlg_hbn_tdhf_q0_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        assembly="vectorized",
    )
    generic = build_rlg_hbn_tdhf_q0_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        assembly="generic",
    )
    np.testing.assert_allclose(vectorized.A, generic.A, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vectorized.B, generic.B, rtol=1e-12, atol=1e-12)


def test_rlg_hbn_tdhf_a_block_matches_hf_interaction_linear_response() -> None:
    """Validate the TDHF A kernel against the HF interaction Hamiltonian.

    For a q=0 density perturbation rho_{h',p'} in the HF eigenbasis, the
    linearized HF interaction Hamiltonian projected back to the HF basis must
    reproduce the non-diagonal interaction part of A[p h, p' h'].  This catches
    prefactor/sign/conjugation mistakes without rerunning SCF.
    """

    run = _tiny_flavor_polarized_run()
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    pairs = build_rlg_hbn_tdhf_q0_pairs(orbitals)
    matrices = build_rlg_hbn_tdhf_q0_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        assembly="vectorized",
    )
    one_body = np.diag(
        [
            orbitals.global_energies[pair.particle] - orbitals.global_energies[pair.hole]
            for pair in pairs
        ]
    )
    a_interaction = matrices.A - one_body

    for row, row_pair in enumerate(pairs):
        p_local, p_k = orbitals.decode_global_index(row_pair.particle)
        h_local, h_k = orbitals.decode_global_index(row_pair.hole)
        assert p_k == h_k == 0
        u_k = orbitals.eigenvectors[:, :, p_k]
        for col, col_pair in enumerate(pairs):
            p_prime_local, p_prime_k = orbitals.decode_global_index(col_pair.particle)
            h_prime_local, h_prime_k = orbitals.decode_global_index(col_pair.hole)
            assert p_prime_k == h_prime_k == 0
            density_hf = np.zeros((orbitals.nt, orbitals.nt), dtype=np.complex128)
            density_hf[h_prime_local, p_prime_local] = 1.0
            density_basis = np.zeros_like(run.state.density)
            density_basis[:, :, p_k] = u_k.conj() @ density_hf @ u_k.T
            response_basis = build_rlg_hbn_hf_interaction_hamiltonian(
                density_basis,
                run.overlap_blocks,
                v0=run.basis_data.v0,
            )
            response_hf = u_k.conj().T @ response_basis[:, :, p_k] @ u_k
            np.testing.assert_allclose(
                a_interaction[row, col],
                response_hf[p_local, h_local],
                rtol=1e-12,
                atol=1e-12,
            )


def _synthetic_tdhf_interaction() -> RLGhBNTDHFInteraction:
    nt = 2
    nk = 2
    n_layer = 2
    eigenvectors = np.zeros((nt, nt, nk), dtype=np.complex128)
    eigenvectors[:, :, 0] = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    eigenvectors[:, :, 1] = np.asarray([[1.0, 1.0j], [1.0j, 1.0]], dtype=np.complex128) / np.sqrt(2.0)
    orbitals = RLGhBNTDHFOrbitals(
        energies=np.zeros((nt, nk), dtype=float),
        eigenvectors=eigenvectors,
        occupied_mask=np.zeros((nt, nk), dtype=bool),
        mu=0.0,
        n_spin=1,
        n_eta=1,
        n_band=2,
    )
    layer_overlaps: dict[tuple[int, int], np.ndarray] = {}
    fock_kernels: dict[tuple[int, int], np.ndarray] = {}
    for shift_index, shift in enumerate(((0, 0), (1, 0))):
        overlap = np.zeros((n_layer, nt, nk, nt, nk), dtype=np.complex128)
        for layer in range(n_layer):
            for kt in range(nk):
                for ks in range(nk):
                    overlap[layer, :, kt, :, ks] = np.asarray(
                        [
                            [1.0 + 0.2 * layer + 0.1 * shift_index, 0.3j + 0.05 * kt],
                            [0.2 - 0.1j * ks, 0.7 + 0.4 * layer + 0.2 * shift_index],
                        ],
                        dtype=np.complex128,
                    )
        kernel = np.zeros((nk, nk, n_layer, n_layer), dtype=float)
        for kt in range(nk):
            for ks in range(nk):
                for layer_t in range(n_layer):
                    for layer_s in range(n_layer):
                        kernel[kt, ks, layer_t, layer_s] = (
                            1.0
                            + 10.0 * shift_index
                            + 0.7 * kt
                            + 0.2 * ks
                            + 0.03 * layer_t
                            + 0.05 * layer_s
                        )
        layer_overlaps[shift] = overlap
        fock_kernels[shift] = kernel
    return RLGhBNTDHFInteraction(
        basis_data=SimpleNamespace(
            nt=nt,
            nk=nk,
            v0=3.0,
            k_grid_frac=np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=float),
        ),
        overlap_blocks=SimpleNamespace(
            shifts=((0, 0), (1, 0)),
            layer_overlaps=layer_overlaps,
            fock_layer_coulomb=fock_kernels,
        ),
        orbitals=orbitals,
        beta=2.0,
    )


def _manual_tdhf_interaction_value(
    interaction: RLGhBNTDHFInteraction,
    a: int,
    b: int,
    c: int,
    d: int,
) -> complex:
    a_local, a_k = interaction.orbitals.decode_global_index(a)
    b_local, b_k = interaction.orbitals.decode_global_index(b)
    c_local, c_k = interaction.orbitals.decode_global_index(c)
    d_local, d_k = interaction.orbitals.decode_global_index(d)
    total = 0.0 + 0.0j
    for shift in interaction.overlap_blocks.shifts:
        layer_overlap = interaction.overlap_blocks.layer_overlaps[shift]
        fock_kernel = interaction.overlap_blocks.fock_layer_coulomb[shift]
        for layer_t in range(layer_overlap.shape[0]):
            left = np.vdot(
                interaction.orbitals.eigenvectors[:, a_local, a_k],
                layer_overlap[layer_t, :, a_k, :, c_k] @ interaction.orbitals.eigenvectors[:, c_local, c_k],
            )
            for layer_s in range(layer_overlap.shape[0]):
                right = np.vdot(
                    interaction.orbitals.eigenvectors[:, d_local, d_k],
                    layer_overlap[layer_s, :, d_k, :, b_k] @ interaction.orbitals.eigenvectors[:, b_local, b_k],
                )
                total += interaction.scale * fock_kernel[a_k, c_k, layer_t, layer_s] * left * np.conj(right)
    return complex(total)


def test_rlg_hbn_tdhf_vectorized_assembly_matches_generic_for_multik_synthetic_blocks() -> None:
    interaction = _synthetic_tdhf_interaction()
    run = SimpleNamespace(basis_data=interaction.basis_data, overlap_blocks=interaction.overlap_blocks)
    pairs = tuple(
        ParticleHolePair(
            particle=interaction.orbitals.global_index(1, ik),
            hole=interaction.orbitals.global_index(0, ik),
            particle_momentum=ik,
            hole_momentum=ik,
        )
        for ik in range(interaction.orbitals.nk)
    )
    vectorized = build_rlg_hbn_tdhf_q0_matrices_from_pairs(
        run,
        interaction.orbitals,
        pairs,
        assembly="vectorized",
    )
    generic = build_rlg_hbn_tdhf_q0_matrices_from_pairs(
        run,
        interaction.orbitals,
        pairs,
        assembly="generic",
    )
    np.testing.assert_allclose(vectorized.A, generic.A, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vectorized.B, generic.B, rtol=1e-12, atol=1e-12)


def test_rlg_hbn_tdhf_direct_contraction_uses_hf_form_factors_and_umklapp_kernels() -> None:
    interaction = _synthetic_tdhf_interaction()
    a = interaction.orbitals.global_index(0, 0)
    c = interaction.orbitals.global_index(1, 1)
    b = interaction.orbitals.global_index(0, 1)
    d = interaction.orbitals.global_index(1, 0)

    actual = interaction(a, b, c, d)
    expected = _manual_tdhf_interaction_value(interaction, a, b, c, d)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    no_umklapp_interaction = RLGhBNTDHFInteraction(
        basis_data=interaction.basis_data,
        overlap_blocks=SimpleNamespace(
            shifts=((0, 0),),
            layer_overlaps={(0, 0): interaction.overlap_blocks.layer_overlaps[(0, 0)]},
            fock_layer_coulomb={(0, 0): interaction.overlap_blocks.fock_layer_coulomb[(0, 0)]},
        ),
        orbitals=interaction.orbitals,
        beta=interaction.beta,
    )
    assert not np.isclose(actual, no_umklapp_interaction(a, b, c, d))


def test_rlg_hbn_tdhf_interaction_enforces_momentum_and_q0_fock_conventions(monkeypatch) -> None:
    interaction = _synthetic_tdhf_interaction()
    a = interaction.orbitals.global_index(0, 0)
    b = interaction.orbitals.global_index(0, 0)
    c = interaction.orbitals.global_index(1, 1)
    d = interaction.orbitals.global_index(1, 0)
    assert interaction(a, b, c, d) == 0.0 + 0.0j

    monkeypatch.setenv("MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK", "1")
    with pytest.raises(ValueError, match="ZERO_LITERAL_Q0_FOCK"):
        _synthetic_tdhf_interaction()
    fake_run = SimpleNamespace(basis_data=interaction.basis_data, overlap_blocks=interaction.overlap_blocks)
    pairs = (
        ParticleHolePair(
            particle=interaction.orbitals.global_index(1, 0),
            hole=interaction.orbitals.global_index(0, 0),
            particle_momentum=0,
            hole_momentum=0,
        ),
    )
    with pytest.raises(ValueError, match="ZERO_LITERAL_Q0_FOCK"):
        build_rlg_hbn_tdhf_q0_matrices_from_pairs(
            fake_run,
            interaction.orbitals,
            pairs,
            assembly="vectorized",
        )


def test_rlg_hbn_tdhf_runner_does_not_apply_single_flavor_shortcut_to_all_channel() -> None:
    state = SimpleNamespace(
        active_valence_bands=0,
        occupation_counts=(1, 0, 0, 0),
        n_spin=2,
        n_eta=2,
    )
    allowed, reason = _shortcut_decision(state, "auto", "all")
    assert not allowed
    assert "all-channel" in reason
