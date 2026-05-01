from __future__ import annotations

import numpy as np

from mean_field.core.hf import build_projected_interaction_hamiltonian
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
    projector_idempotency_residual,
    run_htg_hf,
    validate_hf_state,
    write_htg_fig8a_potential_plot,
    write_htg_fig7_spin_resolved_plot,
    write_htg_hf_path_band_plot,
)
from mean_field.systems.htg.plot import _hf_path_plot_energy_values


def _small_model() -> HTGModel:
    return HTGModel.from_config(1.5, n_shells=0, params=HTGParams.default())


def test_htg_interaction_params_defaults_match_paper_scale() -> None:
    params = InteractionParams()
    assert params.epsilon_r == 8.0
    assert params.d_sc_nm == 25.0
    assert params.U_ev == 0.0
    assert params.subtraction == "average"
    assert params.n_k == 12


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
