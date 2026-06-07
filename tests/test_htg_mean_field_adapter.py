from __future__ import annotations

import numpy as np

from mean_field.core.hf import build_projected_interaction_hamiltonian
from mean_field.systems.htg.hamiltonian import build_hamiltonian, centered_band_indices
from mean_field.systems.htg.mean_field_adapter import (
    _central_chern_basis_at_k,
    _hybrid_projected_basis_at_k,
    _layer_potential_operator,
    centered_projection_band_indices,
)
from mean_field.systems.htg import (
    HTGDensityBuilder,
    HTGHartreeFockState,
    HTGInitializer,
    HTGModel,
    HTGParams,
    InteractionParams,
    build_htg_interaction_components,
    build_htg_overlap_blocks,
    build_htg_projected_basis,
    classify_htg_strong_coupling_state,
    compute_background_density,
    evaluate_htg_hf_path,
    evaluate_htg_interaction_path,
    htg_flavor_occupation_counts_for_init_mode,
    htg_filling_from_density,
    htg_occupied_bands_per_k,
    htg_seed_occupation_summary,
    projector_idempotency_residual,
    run_htg_hf,
    validate_hf_state,
    write_htg_fig8a_potential_plot,
    write_htg_fig7_spin_resolved_plot,
    write_htg_hf_path_band_plot,
)
from mean_field.systems.htg.plot import _hf_path_plot_energy_values, _spin_sector_path_bands
from mean_field.systems.htg.topology import sublattice_sigma_z


def _small_model() -> HTGModel:
    return HTGModel.from_config(1.5, n_shells=0, params=HTGParams.default())


def _full_hybrid_projected_basis_reference(
    k_tilde: complex,
    model: HTGModel,
    interaction: InteractionParams,
    projected_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lattice = model.lattice
    central_pair_raw = centered_band_indices(lattice.matrix_dim, 2)
    central_pair = (int(central_pair_raw[0]), int(central_pair_raw[1]))
    sigma_z_operator = sublattice_sigma_z(lattice)
    layer_potential = _layer_potential_operator(lattice, interaction.U_ev)
    hmat = build_hamiltonian(k_tilde, lattice, model.params, valley=1)
    if interaction.U_ev != 0.0:
        hmat = hmat + layer_potential
    evals_all, evecs_all = np.linalg.eigh(hmat)

    central_evals = np.asarray(evals_all[np.asarray(central_pair, dtype=int)], dtype=float)
    central_evecs = np.asarray(evecs_all[:, np.asarray(central_pair, dtype=int)], dtype=np.complex128)
    projected_sigma = central_evecs.conjugate().T @ sigma_z_operator @ central_evecs
    sigma_eigs, sigma_rot = np.linalg.eigh(projected_sigma)
    central_order = np.asarray([int(np.argmax(sigma_eigs)), int(np.argmin(sigma_eigs))], dtype=int)
    central_rot = np.asarray(sigma_rot[:, central_order], dtype=np.complex128)
    central_wavefunctions = central_evecs @ central_rot
    central_h = central_rot.conjugate().T @ np.diag(central_evals) @ central_rot

    lower_indices = tuple(int(index) for index in projected_indices if int(index) < int(central_pair[0]))
    upper_indices = tuple(int(index) for index in projected_indices if int(index) > int(central_pair[1]))
    ordered_vectors: list[np.ndarray] = []
    for index in lower_indices:
        ordered_vectors.append(np.asarray(evecs_all[:, index], dtype=np.complex128))
    for col in range(2):
        ordered_vectors.append(np.asarray(central_wavefunctions[:, col], dtype=np.complex128))
    for index in upper_indices:
        ordered_vectors.append(np.asarray(evecs_all[:, index], dtype=np.complex128))

    wavefunctions = np.column_stack(ordered_vectors).astype(np.complex128, copy=False)
    h_projected = np.zeros((len(projected_indices), len(projected_indices)), dtype=np.complex128)
    for out_pos, index in enumerate(lower_indices):
        h_projected[out_pos, out_pos] = float(evals_all[index])
    central_start = len(lower_indices)
    h_projected[central_start : central_start + 2, central_start : central_start + 2] = central_h
    for offset, index in enumerate(upper_indices):
        h_projected[central_start + 2 + offset, central_start + 2 + offset] = float(evals_all[index])

    sigma_projected = wavefunctions.conjugate().T @ sigma_z_operator @ wavefunctions
    sigma_diagonal = np.real(np.diag(sigma_projected))
    return wavefunctions, h_projected, sigma_projected, sigma_diagonal


def test_htg_interaction_params_defaults_match_paper_scale() -> None:
    params = InteractionParams()
    assert params.epsilon_r == 8.0
    assert params.d_sc_nm == 25.0
    assert params.U_ev == 0.0
    assert params.subtraction == "average"
    assert params.n_k == 12
    assert not params.finite_zero_limit


def test_htg_projected_basis_embeds_chern_basis_on_rectangular_g_grid() -> None:
    basis_data = build_htg_projected_basis(
        _small_model(),
        InteractionParams(n_k=2, g_shells=0),
    )

    assert basis_data.basis.n_spin == 2
    assert basis_data.basis.n_flavor == 2
    assert basis_data.basis.n_band == 2
    assert basis_data.basis.nt == 8
    assert basis_data.nk == 4
    assert basis_data.h0.shape == (8, 8, 4)
    assert basis_data.sigma_z.shape == (8, 8, 4)
    assert basis_data.moire_cell_area_nm2 > 0.0


def test_htg_projected_basis_can_include_nearest_remote_bands() -> None:
    basis_data = build_htg_projected_basis(
        _small_model(),
        InteractionParams(n_k=2, g_shells=0),
        projected_band_count=4,
    )

    assert basis_data.basis.n_band == 4
    assert basis_data.basis.nt == 16
    assert basis_data.h0.shape == (16, 16, 4)
    assert len(basis_data.projected_band_indices) == 4
    assert set(basis_data.central_band_indices).issubset(set(basis_data.projected_band_indices))


def test_htg_subset_central_chern_basis_matches_full_diagonalization_reference() -> None:
    model = _small_model()
    interaction = InteractionParams(n_k=2, g_shells=0, U_ev=0.015)
    lattice = model.lattice
    central_pair_raw = centered_band_indices(lattice.matrix_dim, 2)
    central_pair = (int(central_pair_raw[0]), int(central_pair_raw[1]))
    sigma_z_operator = sublattice_sigma_z(lattice)
    layer_potential = _layer_potential_operator(lattice, interaction.U_ev)
    k_tilde = 0.017 + 0.029j

    subset_wavefunctions, subset_h, subset_sigma, subset_labels = _central_chern_basis_at_k(
        k_tilde,
        lattice,
        model.params,
        interaction,
        valley=1,
        central_pair=central_pair,
        sigma_z_operator=sigma_z_operator,
        layer_potential=layer_potential,
    )
    full_wavefunctions, full_h, full_sigma, full_labels = _full_hybrid_projected_basis_reference(
        k_tilde,
        model,
        interaction,
        central_pair,
    )

    assert np.allclose(np.linalg.eigvalsh(subset_h), np.linalg.eigvalsh(full_h), atol=1.0e-11)
    assert np.allclose(np.linalg.eigvalsh(subset_sigma), np.linalg.eigvalsh(full_sigma), atol=1.0e-11)
    assert np.allclose(np.sort(subset_labels), np.sort(full_labels), atol=1.0e-11)
    assert np.allclose(
        subset_wavefunctions @ subset_wavefunctions.conjugate().T,
        full_wavefunctions @ full_wavefunctions.conjugate().T,
        atol=1.0e-10,
    )


def test_htg_subset_hybrid_projected_basis_matches_full_diagonalization_reference() -> None:
    model = _small_model()
    interaction = InteractionParams(n_k=2, g_shells=0, U_ev=0.015)
    lattice = model.lattice
    central_pair_raw = centered_band_indices(lattice.matrix_dim, 2)
    central_pair = (int(central_pair_raw[0]), int(central_pair_raw[1]))
    projected_indices = centered_projection_band_indices(lattice.matrix_dim, 4)
    sigma_z_operator = sublattice_sigma_z(lattice)
    layer_potential = _layer_potential_operator(lattice, interaction.U_ev)
    k_tilde = -0.011 + 0.021j

    subset_wavefunctions, subset_h, subset_sigma, subset_labels = _hybrid_projected_basis_at_k(
        k_tilde,
        lattice,
        model.params,
        interaction,
        valley=1,
        projected_indices=projected_indices,
        central_pair=central_pair,
        sigma_z_operator=sigma_z_operator,
        layer_potential=layer_potential,
    )
    full_wavefunctions, full_h, full_sigma, full_labels = _full_hybrid_projected_basis_reference(
        k_tilde,
        model,
        interaction,
        projected_indices,
    )

    assert np.allclose(np.linalg.eigvalsh(subset_h), np.linalg.eigvalsh(full_h), atol=1.0e-11)
    assert np.allclose(np.linalg.eigvalsh(subset_sigma), np.linalg.eigvalsh(full_sigma), atol=1.0e-11)
    assert np.allclose(subset_labels, full_labels, atol=1.0e-11)
    assert np.allclose(
        subset_wavefunctions @ subset_wavefunctions.conjugate().T,
        full_wavefunctions @ full_wavefunctions.conjugate().T,
        atol=1.0e-10,
    )


def test_htg_remote_band_filling_convention_keeps_nu_on_central_pair() -> None:
    counts = htg_flavor_occupation_counts_for_init_mode("sublattice", nu=3.0, n_band=4)
    assert counts == (3, 3, 3, 2)
    assert htg_occupied_bands_per_k(3.0, 16) == 11

    density = np.zeros((16, 16, 2), dtype=np.complex128)
    idx = np.arange(16, dtype=int).reshape((2, 2, 4), order="F")
    for ik in range(density.shape[2]):
        for ispin in range(2):
            for ieta in range(2):
                for iband in (1, 2):
                    state_index = int(idx[ispin, ieta, iband])
                    density[state_index, state_index, ik] = -0.5
    for ispin in range(2):
        for ieta in range(2):
            state_index = int(idx[ispin, ieta, 2])
            density[state_index, state_index, :] = 0.5
    for ispin, ieta in ((0, 0), (0, 1), (1, 0)):
        state_index = int(idx[ispin, ieta, 1])
        density[state_index, state_index, :] = 0.5

    classification = classify_htg_strong_coupling_state(density, n_band=4)
    assert np.isclose(htg_filling_from_density(density), 3.0)
    assert classification.family == "FB"
    assert classification.class_label == "[D3 B]"


def test_htg_seed_occupation_summary_documents_a_b_seed_order() -> None:
    d2a2 = htg_seed_occupation_summary("d2a2", nu=2.0)
    d2b2 = htg_seed_occupation_summary("d2b2", nu=2.0)

    assert d2a2.normalized_init_mode == "fb"
    assert d2b2.normalized_init_mode == "sublattice"
    assert d2a2.occupied_bands_per_k == 6
    assert d2a2.occupation_counts == d2b2.occupation_counts == (2, 2, 1, 1)
    assert d2a2.occupation_count_matrix == ((2, 2), (1, 1))
    assert d2a2.initial_state_labels[:4] == (
        "K_up:central_A",
        "Kprime_up:central_A",
        "K_down:central_A",
        "Kprime_down:central_A",
    )
    assert d2b2.initial_state_labels[:4] == (
        "K_up:central_B",
        "Kprime_up:central_B",
        "K_down:central_B",
        "Kprime_down:central_B",
    )
    assert d2a2.to_dict()["constrained_flavor_counts"] is True


def test_htg_seed_occupation_summary_is_remote_band_aware_without_diagonalization() -> None:
    summary = htg_seed_occupation_summary("d3b", nu=3.0, n_band=4)
    bm_summary = htg_seed_occupation_summary("bm", nu=3.0, n_band=4)

    assert summary.reference_band_occupations == (1.0, 0.5, 0.5, 0.0)
    assert summary.central_projected_band_indices == (1, 2)
    assert summary.occupied_bands_per_k == 11
    assert summary.occupation_counts == (3, 3, 3, 2)
    assert summary.initial_state_labels[:4] == (
        "K_up:lower_remote_1",
        "Kprime_up:lower_remote_1",
        "K_down:lower_remote_1",
        "Kprime_down:lower_remote_1",
    )
    assert summary.initial_state_labels[4:8] == (
        "K_up:central_B",
        "Kprime_up:central_B",
        "K_down:central_B",
        "Kprime_down:central_B",
    )
    assert bm_summary.occupation_counts is None
    assert bm_summary.initial_state_labels is None
    assert bm_summary.constrained_flavor_counts is False


def test_htg_four_band_bm_density_update_uses_remote_aware_filling() -> None:
    hamiltonian = np.zeros((16, 16, 2), dtype=np.complex128)
    for ik in range(hamiltonian.shape[2]):
        hamiltonian[:, :, ik] = np.diag(np.arange(16, dtype=float))

    density_update = HTGDensityBuilder(3.0, n_band=4)(hamiltonian)
    occupation_mask = np.asarray(density_update.observables["occupation_mask"], dtype=bool)

    assert np.all(np.sum(occupation_mask, axis=0) == 11)
    assert np.isclose(density_update.mu, 10.5)
    assert np.isclose(htg_filling_from_density(density_update.density), 3.0)
    assert projector_idempotency_residual(density_update.density) < 1.0e-12


def test_htg_four_band_bm_initializer_preserves_remote_aware_filling() -> None:
    basis_data = build_htg_projected_basis(
        _small_model(),
        InteractionParams(n_k=2, g_shells=0),
        projected_band_count=4,
    )
    state = HTGHartreeFockState.from_projected_basis(basis_data, nu=3.0, precision=1.0e-8)
    HTGInitializer()(state, init_mode="bm", seed=1)

    assert np.isclose(htg_filling_from_density(state.density), 3.0)
    assert projector_idempotency_residual(state.density) < 1.0e-10


def test_htg_overlap_zero_shift_has_identity_diagonal_and_background_density_one() -> None:
    basis_data = build_htg_projected_basis(
        _small_model(),
        InteractionParams(n_k=2, g_shells=0),
    )
    overlap_blocks = build_htg_overlap_blocks(basis_data)
    diagonal = overlap_blocks.diagonal_overlaps[(0, 0)]

    for ik in range(basis_data.nk):
        assert np.allclose(diagonal[:, :, ik], np.eye(8), atol=1.0e-10)
    assert np.isclose(compute_background_density(diagonal), 1.0 + 0.0j, atol=1.0e-10)


def test_htg_initializer_and_density_builder_preserve_filling_and_projector_constraints() -> None:
    basis_data = build_htg_projected_basis(
        _small_model(),
        InteractionParams(n_k=2, g_shells=0),
    )
    state = HTGHartreeFockState.from_projected_basis(basis_data, nu=0.0, precision=1.0e-8)
    HTGInitializer()(state, init_mode="fb", seed=1)

    assert np.isclose(htg_filling_from_density(state.density), 0.0)
    assert projector_idempotency_residual(state.density) < 1.0e-12

    density_update = HTGDensityBuilder(0.0, sigma_z=state.sigma_z)(state.h0)
    assert np.isclose(htg_filling_from_density(density_update.density), 0.0)
    assert projector_idempotency_residual(density_update.density) < 1.0e-10


def test_htg_overlap_blocks_feed_generic_projected_interaction_builder() -> None:
    basis_data = build_htg_projected_basis(
        _small_model(),
        InteractionParams(n_k=2, g_shells=0),
    )
    overlap_blocks = build_htg_overlap_blocks(basis_data)
    state = HTGHartreeFockState.from_projected_basis(basis_data, nu=0.0, precision=1.0e-8)
    HTGInitializer()(state, init_mode="fb", seed=1)

    interaction_h = build_projected_interaction_hamiltonian(
        state.density,
        overlap_blocks,
        v0=state.v0,
        use_numba=False,
    )

    assert interaction_h.shape == state.h0.shape
    for ik in range(state.nk):
        assert np.max(np.abs(interaction_h[:, :, ik] - interaction_h[:, :, ik].conjugate().T)) < 1.0e-8


def test_htg_hf_run_exports_path_bands_and_interaction_components(tmp_path) -> None:
    run = run_htg_hf(
        _small_model(),
        InteractionParams(n_k=2, g_shells=0),
        nu=0.0,
        init_mode="fb",
        seed=1,
        max_iter=1,
        use_numba=False,
    )
    components = build_htg_interaction_components(
        run.state.density,
        run.overlap_blocks,
        v0=run.state.v0,
        use_numba=False,
    )
    path_result = evaluate_htg_hf_path(
        run,
        points_per_segment=2,
        use_numba=False,
    )
    potential_path_result = evaluate_htg_interaction_path(
        run,
        points_per_segment=2,
        use_numba=False,
    )
    plot_paths = write_htg_hf_path_band_plot(tmp_path, path_result, stem="hf_path_test")
    fig7_plot_paths = write_htg_fig7_spin_resolved_plot(tmp_path, path_result, stem="fig7_test")
    potential_plot_paths = write_htg_fig8a_potential_plot(tmp_path, potential_path_result, stem="fig8a_test")
    plot_energies = _hf_path_plot_energy_values(path_result)

    assert components.hartree.shape == run.state.h0.shape
    assert components.fock.shape == run.state.h0.shape
    assert path_result.energies.shape == (path_result.path.kvec.size, run.state.nt)
    assert path_result.sigma_z_expectation.shape == path_result.energies.shape
    assert path_result.path.labels[-1] == "Gamma"
    assert potential_path_result.hartree_diagonal_ev.shape == (run.state.n_spin, run.state.n_eta, run.state.n_band, potential_path_result.path.kvec.size)
    assert potential_path_result.fock_diagonal_ev.shape == potential_path_result.hartree_diagonal_ev.shape
    assert np.allclose(plot_energies, (path_result.energies - path_result.mu) * 1000.0)
    for path in tuple(plot_paths.values()) + tuple(fig7_plot_paths.values()) + tuple(potential_plot_paths.values()):
        assert path.exists()
        assert path.stat().st_size > 0


def test_htg_fig7_spin_sector_plot_tracks_crossing_bands_by_eigenvector_overlap() -> None:
    nk = 4
    nt = 8
    hamiltonian = np.zeros((nt, nt, nk), dtype=np.complex128)
    sigma_z = np.zeros_like(hamiltonian)
    idx = np.arange(nt, dtype=int).reshape((2, 2, 2), order="F")
    spin_up_indices = np.asarray(idx[0, :, :].reshape(-1, order="C"), dtype=int)
    t_values = np.asarray([-1.0, -0.3, 0.2, 1.0], dtype=float)

    for ik, value in enumerate(t_values):
        block = np.diag([value, -value, 2.0, 3.0]).astype(np.complex128)
        hamiltonian[:, :, ik][np.ix_(spin_up_indices, spin_up_indices)] = block
        sigma_z[:, :, ik][np.ix_(spin_up_indices, spin_up_indices)] = np.diag([1.0, -1.0, 0.5, -0.5])

    energies, sigma_values = _spin_sector_path_bands(hamiltonian, sigma_z, spin_index=0)

    assert np.allclose(energies[:, 0], t_values)
    assert np.allclose(energies[:, 1], -t_values)
    assert np.allclose(sigma_values[:, 0], 1.0)
    assert np.allclose(sigma_values[:, 1], -1.0)


def test_htg_full_filling_validation_treats_internal_gap_as_not_applicable() -> None:
    basis_data = build_htg_projected_basis(
        _small_model(),
        InteractionParams(n_k=2, g_shells=0),
    )
    state = HTGHartreeFockState.from_projected_basis(basis_data, nu=4.0, precision=1.0e-8)
    HTGInitializer()(state, init_mode="fb", seed=1)
    checks = {check.name: check for check in validate_hf_state(state)}

    assert checks["hf_gap_ev"].passed
    assert checks["hf_gap_ev"].value == "not_applicable_full_or_empty_projected_space"


def test_htg_fig7_initialization_aliases_select_expected_nu2_classes() -> None:
    basis_data = build_htg_projected_basis(
        _small_model(),
        InteractionParams(n_k=2, g_shells=0),
    )

    state_d2a2 = HTGHartreeFockState.from_projected_basis(basis_data, nu=2.0, precision=1.0e-8)
    HTGInitializer()(state_d2a2, init_mode="d2a2", seed=1)
    assert classify_htg_strong_coupling_state(state_d2a2.density).class_label == "[D2 A2]"

    state_d2b2 = HTGHartreeFockState.from_projected_basis(basis_data, nu=2.0, precision=1.0e-8)
    HTGInitializer()(state_d2b2, init_mode="d2b2", seed=1)
    assert classify_htg_strong_coupling_state(state_d2b2.density).class_label == "[D2 B2]"

    state_d3 = HTGHartreeFockState.from_projected_basis(basis_data, nu=2.0, precision=1.0e-8)
    HTGInitializer()(state_d3, init_mode="d3", seed=1)
    assert classify_htg_strong_coupling_state(state_d3.density).class_label == "[D3]"


def test_htg_fig9_initialization_aliases_select_expected_nu3_classes() -> None:
    basis_data = build_htg_projected_basis(
        _small_model(),
        InteractionParams(n_k=2, g_shells=0),
    )

    state_d3a = HTGHartreeFockState.from_projected_basis(basis_data, nu=3.0, precision=1.0e-8)
    HTGInitializer()(state_d3a, init_mode="d3a", seed=1)
    assert classify_htg_strong_coupling_state(state_d3a.density).class_label == "[D3 A]"

    state_d3b = HTGHartreeFockState.from_projected_basis(basis_data, nu=3.0, precision=1.0e-8)
    HTGInitializer()(state_d3b, init_mode="d3b", seed=1)
    assert classify_htg_strong_coupling_state(state_d3b.density).class_label == "[D3 B]"

    assert htg_flavor_occupation_counts_for_init_mode("d3a", nu=3.0) == htg_flavor_occupation_counts_for_init_mode(
        "fb", nu=3.0
    )
    assert htg_flavor_occupation_counts_for_init_mode("d3b", nu=3.0) == htg_flavor_occupation_counts_for_init_mode(
        "sublattice", nu=3.0
    )


def test_htg_constrained_density_builder_preserves_d3_flavor_sector() -> None:
    basis_data = build_htg_projected_basis(
        _small_model(),
        InteractionParams(n_k=2, g_shells=0),
    )
    hamiltonian = np.zeros_like(basis_data.h0)
    idx = np.arange(8, dtype=int).reshape((2, 2, 2), order="F")
    # Make the nominally empty fourth flavor artificially lowest in energy.
    # A global filling update would occupy it; the Fig. 7 D3 sector must not.
    flavor_offsets = [0.3, 0.2, 0.1, -10.0]
    for ik in range(hamiltonian.shape[2]):
        for flavor_index, (ispin, ieta) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            block_indices = idx[ispin, ieta, :]
            hamiltonian[block_indices[0], block_indices[0], ik] = flavor_offsets[flavor_index]
            hamiltonian[block_indices[1], block_indices[1], ik] = flavor_offsets[flavor_index] + 0.05

    counts = htg_flavor_occupation_counts_for_init_mode("d3", nu=2.0)
    assert counts == (2, 2, 2, 0)
    density_update = HTGDensityBuilder(2.0, occupation_counts=counts)(hamiltonian)

    classification = classify_htg_strong_coupling_state(density_update.density)
    assert classification.family == "FI"
    assert classification.class_label == "[D3]"
    assert np.isclose(htg_filling_from_density(density_update.density), 2.0)
    assert projector_idempotency_residual(density_update.density) < 1.0e-12
