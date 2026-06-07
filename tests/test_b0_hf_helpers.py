from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field import (
    RestrictedHartreeFockState,
    build_interaction_hamiltonian,
    build_overlap_block_set,
    block_mask,
    build_full_density_from_hamiltonian,
    build_flavor_band_data,
    build_h0_from_bm,
    build_restricted_density_from_hamiltonian,
    calculate_norm_convergence,
    canonical_fig6_flavor_sequence,
    compute_hf_energy,
    coulomb_unit,
    empty_overlap_block_set,
    find_chemical_potential,
    flavor_block_indices,
    flavor_sector_metadata,
    HFOverlapBlockSet,
    initialize_restricted_density,
    initialize_restricted_state,
    normalize_restricted_init_mode,
    oda_parametrization_restricted,
    occupied_sigma_mean,
    offdiag_flavor_norm,
    project_to_flavor_diagonal,
    restricted_filling,
    restricted_gap_estimate,
    restricted_occupied_bands_per_k,
    restricted_occupied_state_count,
    reciprocal_shift_labels,
    run_full_hartree_fock,
    run_restricted_hartree_fock,
    run_restricted_hf_from_bm_solution,
    screened_coulomb,
    update_restricted_density,
    build_sigma_z_from_uk,
)
from mean_field.systems.tbg.zero_field.hf import (
    _occupied_state_linear_indices,
    contract_fock_term_from_overlap,
    initialize_full_density,
)
from mean_field.systems.tbg.zero_field.model import BMSolution


def _fake_bm_solution() -> BMSolution:
    params = TBGParameters.from_degrees(1.2)
    spectrum = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=float,
    )
    uk = np.zeros((4, 2, 2, 2), dtype=np.complex128)
    return BMSolution(
        params=params,
        lattice_kvec=np.asarray([0.0 + 0.0j, 0.1 + 0.0j], dtype=np.complex128),
        lg=1,
        nlocal=4,
        n_eta=2,
        n_spin=2,
        nb=2,
        hamiltonian=np.zeros((4, 4, 2, 2), dtype=np.complex128),
        sigma_z=np.zeros((8, 8, 2), dtype=np.complex128),
        uk=uk,
        spectrum=spectrum,
        gvec=np.asarray([0.0 + 0.0j], dtype=np.complex128),
    )


def _restricted_reference_hamiltonian() -> tuple[np.ndarray, np.ndarray]:
    hamiltonian = np.zeros((8, 8, 2), dtype=np.complex128)
    sigma_z = np.zeros((8, 8, 2), dtype=np.complex128)
    diag_k0 = np.asarray([0.2, 2.0, 0.4, 2.2, 10.0, 11.0, 12.0, 13.0], dtype=float)
    diag_k1 = np.asarray([0.3, 2.1, 0.5, 2.3, 10.1, 11.1, 12.1, 13.1], dtype=float)
    sigma_diag = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=float)
    np.fill_diagonal(hamiltonian[:, :, 0], diag_k0)
    np.fill_diagonal(hamiltonian[:, :, 1], diag_k1)
    np.fill_diagonal(sigma_z[:, :, 0], sigma_diag)
    np.fill_diagonal(sigma_z[:, :, 1], sigma_diag)
    return hamiltonian, sigma_z


def test_flavor_sector_metadata_matches_julia_order() -> None:
    labels, sectors = flavor_sector_metadata()
    assert labels == ("K_up", "Kprime_up", "K_down", "Kprime_down")
    assert sectors == ((0, 4), (2, 6), (1, 5), (3, 7))


def test_build_h0_from_bm_places_flattened_energies_on_diagonal() -> None:
    solution = _fake_bm_solution()
    h0 = build_h0_from_bm(solution)
    expected = solution.flattened_energies()
    for ik in range(solution.nk):
        assert np.allclose(np.diag(h0[:, :, ik]), expected[:, ik])
        assert np.count_nonzero(h0[:, :, ik] - np.diag(np.diag(h0[:, :, ik]))) == 0


def test_bm_solution_with_reference_uk_recomputes_sigma_z() -> None:
    solution = _fake_bm_solution()
    uk = np.zeros_like(solution.uk)
    uk[0, 0, :, :] = 1.0
    uk[1, 1, :, :] = 1.0

    updated = solution.with_reference_uk(uk)
    sigma_expected = build_sigma_z_from_uk(uk, lg=solution.lg, n_spin=solution.n_spin)

    assert np.allclose(updated.uk, uk)
    assert np.allclose(updated.sigma_z, sigma_expected)
    assert np.allclose(np.diag(updated.sigma_z[:, :, 0]), [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0])


def test_build_flavor_band_data_assigns_dominant_flavor_labels() -> None:
    hamiltonian = np.zeros((8, 8, 2), dtype=np.complex128)
    diag_k0 = np.asarray([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1], dtype=float)
    diag_k1 = np.asarray([0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2], dtype=float)
    np.fill_diagonal(hamiltonian[:, :, 0], diag_k0)
    np.fill_diagonal(hamiltonian[:, :, 1], diag_k1)

    band_data = build_flavor_band_data(hamiltonian)

    assert band_data.band_labels == (
        "K_up_b1",
        "K_down_b2",
        "Kprime_up_b3",
        "Kprime_down_b4",
        "K_up_b5",
        "K_down_b6",
        "Kprime_up_b7",
        "Kprime_down_b8",
    )
    assert np.allclose(band_data.energies[:, 0], diag_k0)
    assert np.allclose(band_data.energies[:, 1], diag_k1)


def test_restricted_block_helpers_follow_julia_conventions() -> None:
    assert flavor_block_indices() == ((0, 4), (2, 6), (1, 5), (3, 7))
    assert canonical_fig6_flavor_sequence("educated") == ((1, 0), (0, 0), (1, 1), (0, 1))
    assert canonical_fig6_flavor_sequence("spindown") == ((1, 0), (1, 1), (0, 0), (0, 1))
    assert canonical_fig6_flavor_sequence("sp") == canonical_fig6_flavor_sequence("spindown")
    assert canonical_fig6_flavor_sequence("chern") == canonical_fig6_flavor_sequence("vp")
    assert normalize_restricted_init_mode("sp") == "spindown"
    assert normalize_restricted_init_mode("chern") == "vp"
    assert reciprocal_shift_labels(3) == (-1, 0, 1)

    mask = block_mask()
    assert mask[0, 4]
    assert not mask[0, 1]

    matrix = np.ones((8, 8, 1), dtype=np.complex128)
    projected = project_to_flavor_diagonal(matrix)
    assert projected[0, 4, 0] == 1.0
    assert projected[0, 1, 0] == 0.0


def test_hf_scalar_helpers_match_reference_formulae() -> None:
    params = TBGParameters.from_degrees(1.2)
    lm = float(np.sqrt(abs(params.a1) * abs(params.a2)))

    assert coulomb_unit(params) > 0.0
    assert screened_coulomb(0.0 + 0.0j, lm) == 0.0
    assert screened_coulomb(0.2 + 0.1j, lm) > 0.0

    updated_density = np.asarray([[[1.0 + 0.0j]], [[0.0 + 0.0j]]]).reshape(1, 2, 1)
    previous_density = np.zeros((1, 2, 1), dtype=np.complex128)
    assert np.isclose(calculate_norm_convergence(updated_density, previous_density), 1.0)


def test_occupied_state_selection_uses_stable_fortran_order() -> None:
    energies = np.zeros((2, 2), dtype=float)
    order = _occupied_state_linear_indices(energies, 2)
    assert order.tolist() == [0, 1]

    hamiltonian = np.zeros((2, 2, 2), dtype=np.complex128)
    sigma_z = np.zeros_like(hamiltonian)
    density, energies_out, sigma_out, mu = build_full_density_from_hamiltonian(
        hamiltonian,
        sigma_z,
        nu=0.0,
    )

    assert np.allclose(energies_out, 0.0)
    assert np.allclose(sigma_out, 0.0)
    assert np.isclose(mu, 0.0)
    assert np.allclose(np.diag(density[:, :, 0]), [0.5, 0.5])
    assert np.allclose(np.diag(density[:, :, 1]), [-0.5, -0.5])


def test_full_density_builder_matches_julia_upper_triangle_hermitian_convention() -> None:
    hamiltonian = np.zeros((2, 2, 1), dtype=np.complex128)
    hamiltonian[:, :, 0] = np.asarray(
        [
            [0.0 + 0.0j, 1.0 + 2.0j],
            [3.0 + 4.0j, 5.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    sigma_z = np.zeros_like(hamiltonian)

    density, energies_out, sigma_out, mu = build_full_density_from_hamiltonian(
        hamiltonian,
        sigma_z,
        nu=0.0,
    )

    expected_energies, expected_vecs = np.linalg.eigh(hamiltonian[:, :, 0], UPLO="U")
    expected_density = expected_vecs[:, :1].conj() @ expected_vecs[:, :1].T - 0.5 * np.eye(2, dtype=np.complex128)

    assert np.allclose(energies_out[:, 0], expected_energies)
    assert np.allclose(density[:, :, 0], expected_density)
    assert np.allclose(sigma_out[:, 0], 0.0)
    assert np.isclose(mu, float((expected_energies[0] + expected_energies[1]) / 2.0))


def test_restricted_density_builder_matches_julia_restricted_projector_convention() -> None:
    hamiltonian = np.zeros((2, 2, 1), dtype=np.complex128)
    hamiltonian[:, :, 0] = np.asarray(
        [
            [0.0 + 0.0j, 1.0 + 2.0j],
            [1.0 - 2.0j, 5.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    sigma_z = np.zeros_like(hamiltonian)

    density, energies_out, sigma_out, mu = build_restricted_density_from_hamiltonian(
        hamiltonian,
        sigma_z,
        nu=0.0,
        n_spin=1,
        n_eta=1,
        n_band=2,
    )

    expected_energies, expected_vecs = np.linalg.eigh(hamiltonian[:, :, 0])
    occupied_vecs = expected_vecs[:, :1]
    expected_density = occupied_vecs @ occupied_vecs.conj().T - 0.5 * np.eye(2, dtype=np.complex128)

    assert np.allclose(energies_out[:, 0], expected_energies)
    assert np.allclose(density[:, :, 0], expected_density)
    assert np.allclose(sigma_out[:, 0], 0.0)
    assert np.isclose(mu, float((expected_energies[0] + expected_energies[1]) / 2.0))


def test_restricted_initializers_match_canonical_and_bm_expectations() -> None:
    h0 = np.zeros((8, 8, 2), dtype=np.complex128)
    diag_k0 = np.asarray([0.1, 1.1, 0.2, 1.2, 10.0, 11.0, 12.0, 13.0], dtype=float)
    diag_k1 = np.asarray([0.3, 1.3, 0.4, 1.4, 10.1, 11.1, 12.1, 13.1], dtype=float)
    np.fill_diagonal(h0[:, :, 0], diag_k0)
    np.fill_diagonal(h0[:, :, 1], diag_k1)

    educated = initialize_restricted_density(h0, nu=-2.0, init_mode="educated")
    assert np.allclose(np.diag(educated[:, :, 0]), [0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    assert np.allclose(np.diag(educated[:, :, 1]), [0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    assert np.isclose(restricted_filling(educated), -2.0)

    bm = initialize_restricted_density(h0, nu=-2.0, init_mode="bm")
    assert np.allclose(np.diag(bm[:, :, 0]), [0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    assert np.allclose(np.diag(bm[:, :, 1]), [0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    assert np.isclose(offdiag_flavor_norm(bm), 0.0)

    sp_alias = initialize_restricted_density(h0, nu=-2.0, init_mode="sp")
    spindown = initialize_restricted_density(h0, nu=-2.0, init_mode="spindown")
    assert np.allclose(sp_alias, spindown)

    chern_alias = initialize_restricted_density(h0, nu=-3.0, init_mode="chern")
    vp = initialize_restricted_density(h0, nu=-3.0, init_mode="vp")
    assert np.allclose(chern_alias, vp)


def test_full_sp_initializer_matches_julia_style_rotation_signature() -> None:
    h0 = np.zeros((8, 8, 2), dtype=np.complex128)

    density = initialize_full_density(h0, nu=-2.0, init_mode="sp", seed=1)

    assert np.isclose(offdiag_flavor_norm(density), 0.03770497333527351)
    assert np.isclose(np.linalg.norm(density), 1.9536508380745765)

    offdiag = density.copy()
    for ik in range(density.shape[2]):
        offdiag[:, :, ik] -= np.diag(np.diag(density[:, :, ik]))
    assert np.isclose(np.linalg.norm(offdiag), 0.06753037891380292)

    assert np.isclose(density[0, 0, 0].real, 0.4655990159255736)
    assert np.isclose(density[4, 4, 0].real, 0.4888484845543026)
    assert np.isclose(density[0, 4, 0].real, 0.013612363384187226)
    assert np.isclose(density[0, 4, 0].imag, -0.003177947472296631)


def test_restricted_state_constructor_and_update_follow_block_diagonal_density() -> None:
    state = RestrictedHartreeFockState.from_bm_solution(_fake_bm_solution(), nu=-2.0)
    filling = initialize_restricted_state(state, init_mode="educated")

    assert np.isclose(filling, -2.0)
    assert np.isclose(state.diagnostics["offdiag_flavor_norm"], 0.0)
    assert state.v0 > 0.0

    hamiltonian, sigma_z = _restricted_reference_hamiltonian()
    state.hamiltonian[:, :, :] = hamiltonian
    state.sigma_z[:, :, :] = sigma_z
    norm_convergence, mixing = update_restricted_density(state)

    assert mixing == 1.0
    assert norm_convergence > 0.0
    assert np.allclose(state.energies[:, 0], np.diag(hamiltonian[:, :, 0]).real)
    assert np.allclose(state.energies[:, 1], np.diag(hamiltonian[:, :, 1]).real)
    assert np.allclose(np.diag(state.density[:, :, 0]), [0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    assert np.allclose(np.diag(state.density[:, :, 1]), [0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    assert np.isclose(state.mu, 1.25)
    assert np.isclose(state.diagnostics["occupied_sigma_mean"], 20.0)
    assert np.isclose(state.diagnostics["offdiag_flavor_norm"], 0.0)
    assert np.isclose(state.diagnostics["restricted_gap"], 1.5)
    assert np.isclose(state.diagnostics["filling"], -2.0)


def test_build_restricted_density_from_hamiltonian_matches_scalar_diagnostics() -> None:
    hamiltonian, sigma_z = _restricted_reference_hamiltonian()
    density, energies, sigma_ztauz, mu = build_restricted_density_from_hamiltonian(hamiltonian, sigma_z, nu=-2.0)

    assert np.allclose(np.diag(density[:, :, 0]), [0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    assert np.allclose(np.diag(density[:, :, 1]), [0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    assert np.isclose(mu, find_chemical_potential(energies, 0.25))
    assert np.isclose(occupied_sigma_mean(energies, sigma_ztauz, -2.0), 20.0)


def test_hf_energy_and_restricted_diagnostics_helpers() -> None:
    interaction_h = np.zeros((2, 2, 2), dtype=np.complex128)
    h0 = np.zeros((2, 2, 2), dtype=np.complex128)
    density = np.zeros((2, 2, 2), dtype=np.complex128)

    interaction_h[:, :, 0] = np.diag([1.0, 2.0])
    interaction_h[:, :, 1] = np.diag([3.0, 4.0])
    h0[:, :, 0] = np.diag([0.5, 1.5])
    h0[:, :, 1] = np.diag([2.5, 3.5])
    density[:, :, 0] = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    density[:, :, 1] = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    assert np.isclose(compute_hf_energy(interaction_h, h0, density), 3.25)

    energies = np.asarray([[0.1, 0.2], [1.1, 1.2], [2.1, 2.2], [3.1, 3.2]], dtype=float)
    sigma = np.asarray([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]], dtype=float)
    assert np.isclose(restricted_gap_estimate(energies, -2.0), 0.9)
    assert np.isclose(occupied_sigma_mean(energies, sigma, -2.0), 15.0)
    assert np.isnan(occupied_sigma_mean(energies, sigma, -4.0))
    with pytest.raises(ValueError, match="non-integer occupied-state count"):
        restricted_occupied_state_count(-3.9, 8, 1)
    with pytest.raises(ValueError, match="non-integer per-k occupation"):
        restricted_occupied_bands_per_k(-3.9, 8)

    density8 = np.zeros((8, 8, 1), dtype=np.complex128)
    density8[0, 1, 0] = 3.0 + 4.0j
    density8[0, 4, 0] = 7.0 + 0.0j
    assert np.isclose(offdiag_flavor_norm(density8), 5.0)


def test_hf_energy_uses_julia_transpose_convention_for_offdiagonal_density() -> None:
    interaction_h = np.zeros((2, 2, 1), dtype=np.complex128)
    h0 = np.zeros((2, 2, 1), dtype=np.complex128)
    density = np.zeros((2, 2, 1), dtype=np.complex128)

    interaction_h[:, :, 0] = np.asarray([[0.0, 7.0], [11.0, 0.0]], dtype=np.complex128)
    h0[:, :, 0] = np.asarray([[0.0, 3.0], [5.0, 0.0]], dtype=np.complex128)
    density[:, :, 0] = np.asarray([[0.0, 1.0], [2.0, 0.0]], dtype=np.complex128)

    expected = (7.0 * 1.0 + 11.0 * 2.0) / 2.0 + (3.0 * 1.0 + 5.0 * 2.0)
    assert np.isclose(compute_hf_energy(interaction_h, h0, density), expected)


def test_hf_energy_matches_julia_projector_storage_for_complex_hermitian_blocks() -> None:
    h0 = np.zeros((2, 2, 1), dtype=np.complex128)
    density = np.zeros((2, 2, 1), dtype=np.complex128)
    h0[:, :, 0] = np.asarray([[1.0, 1.0j], [-1.0j, 2.0]], dtype=np.complex128)
    density[:, :, 0] = np.asarray([[3.0, 2.0j], [-2.0j, 4.0]], dtype=np.complex128)

    # Julia B0 uses `tr(H * transpose(P))`, which is 7 here.
    # The ordinary matrix trace `tr(H * P)` would be 15, but that is not the
    # stored-projector contraction used by the benchmarked B0 code.
    assert np.isclose(compute_hf_energy(np.zeros_like(h0), h0, density), 7.0)


def test_build_overlap_block_set_generates_expected_shift_grid() -> None:
    solution = _fake_bm_solution()
    overlap_blocks = build_overlap_block_set(solution, lg=1)

    assert overlap_blocks.shifts == ((0, 0),)
    assert np.allclose(overlap_blocks.gvecs, [0.0 + 0.0j])
    assert overlap_blocks.overlaps[(0, 0)].shape == (solution.nt, solution.nk, solution.nt, solution.nk)
    assert overlap_blocks.diagonal_overlaps[(0, 0)].shape == (solution.nt, solution.nt, solution.nk)
    assert np.isclose(overlap_blocks.hartree_screening[(0, 0)], 0.0)
    assert overlap_blocks.fock_screening[(0, 0)].shape == (solution.nk, solution.nk)


def test_build_interaction_hamiltonian_matches_simple_fock_limit() -> None:
    params = TBGParameters.from_degrees(1.2)
    density = np.zeros((2, 2, 2), dtype=np.complex128)
    density[:, :, 0] = np.diag([0.5, -0.5])
    density[:, :, 1] = np.diag([-0.5, 0.5])

    overlap = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    overlap[:, 0, :, 0] = np.eye(2, dtype=np.complex128)
    overlap[:, 1, :, 1] = np.eye(2, dtype=np.complex128)
    overlap_blocks = HFOverlapBlockSet(
        shifts=((1, 0),),
        gvecs=np.asarray([0.2 + 0.0j], dtype=np.complex128),
        overlaps={(1, 0): overlap},
    )
    lattice_kvec = np.asarray([0.0 + 0.0j, 0.1 + 0.0j], dtype=np.complex128)
    v0 = coulomb_unit(params)

    interaction = build_interaction_hamiltonian(density, overlap_blocks, lattice_kvec, params, v0)

    coeff = v0 * screened_coulomb(0.2 + 0.0j, np.sqrt(abs(params.a1) * abs(params.a2))) / 2.0
    assert np.allclose(interaction[:, :, 0], -coeff * density[:, :, 0].T)
    assert np.allclose(interaction[:, :, 1], -coeff * density[:, :, 1].T)


def test_contract_fock_term_from_overlap_supports_rectangular_target_source_grids() -> None:
    overlap = np.zeros((2, 3, 2, 2), dtype=np.complex128)
    overlap[:, 0, :, 0] = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.complex128)
    overlap[:, 0, :, 1] = np.asarray([[0.5, 0.0], [0.0, 1.5]], dtype=np.complex128)
    overlap[:, 1, :, 0] = np.asarray([[1.0, 0.2], [0.0, 1.0]], dtype=np.complex128)
    overlap[:, 1, :, 1] = np.asarray([[0.8, 0.0], [0.1, 1.1]], dtype=np.complex128)
    overlap[:, 2, :, 0] = np.asarray([[0.6, 0.0], [0.0, 0.9]], dtype=np.complex128)
    overlap[:, 2, :, 1] = np.asarray([[1.2, 0.0], [0.0, 0.7]], dtype=np.complex128)

    density = np.zeros((2, 2, 2), dtype=np.complex128)
    density[:, :, 0] = np.asarray([[0.5, 0.1j], [-0.1j, -0.25]], dtype=np.complex128)
    density[:, :, 1] = np.asarray([[0.2, 0.3], [0.3, -0.4]], dtype=np.complex128)

    coeff_matrix = np.asarray(
        [
            [1.0, 0.5],
            [0.2, 0.8],
            [0.0, 1.1],
        ],
        dtype=float,
    )

    expected = np.zeros((2, 2, 3), dtype=np.complex128)
    for it in range(3):
        for ip in range(2):
            lambda_block = overlap[:, it, :, ip]
            expected[:, :, it] += coeff_matrix[it, ip] * (lambda_block @ density[:, :, ip].T @ lambda_block.conj().T)

    contracted = contract_fock_term_from_overlap(overlap, density, coeff_matrix)
    assert np.allclose(contracted, expected)


def test_oda_parametrization_restricted_handles_flat_direction() -> None:
    params = TBGParameters.from_degrees(1.2)
    state = RestrictedHartreeFockState(
        h0=np.zeros((2, 2, 1), dtype=np.complex128),
        sigma_z=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
        sigma_ztauz=np.zeros((2, 1), dtype=float),
        nu=0.0,
        v0=coulomb_unit(params),
        precision=1e-5,
        n_spin=1,
        n_eta=1,
        n_band=2,
    )
    delta_density = np.zeros_like(state.density)
    overlap_blocks = empty_overlap_block_set()
    lambda_mix = oda_parametrization_restricted(state, delta_density, overlap_blocks, np.asarray([0.0 + 0.0j]), params)
    assert lambda_mix == 0.0


def test_oda_parametrization_restricted_matches_julia_transpose_convention(monkeypatch: pytest.MonkeyPatch) -> None:
    params = TBGParameters.from_degrees(1.2)
    state = RestrictedHartreeFockState(
        h0=np.zeros((2, 2, 1), dtype=np.complex128),
        sigma_z=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
        sigma_ztauz=np.zeros((2, 1), dtype=float),
        nu=0.0,
        v0=coulomb_unit(params),
        precision=1e-5,
        n_spin=1,
        n_eta=1,
        n_band=2,
    )
    state.h0[:, :, 0] = np.asarray([[0.0, -3.0], [-5.0, 0.0]], dtype=np.complex128)
    interaction_h = np.asarray([[0.0, -7.0], [-11.0, 0.0]], dtype=np.complex128)
    state.hamiltonian[:, :, 0] = state.h0[:, :, 0] + interaction_h

    delta_density = np.zeros_like(state.density)
    delta_density[:, :, 0] = np.asarray([[0.0, 1.0], [2.0, 0.0]], dtype=np.complex128)
    delta_h = np.zeros_like(state.density)
    delta_h[:, :, 0] = np.asarray([[0.0, 19.0], [23.0, 0.0]], dtype=np.complex128)

    monkeypatch.setattr(
        "mean_field.systems.tbg.zero_field.hf.build_interaction_hamiltonian",
        lambda *args, **kwargs: delta_h,
    )

    overlap_blocks = empty_overlap_block_set()
    lambda_mix = oda_parametrization_restricted(state, delta_density, overlap_blocks, np.asarray([0.0 + 0.0j]), params)
    assert np.isclose(lambda_mix, 27.5 / 65.0)


def test_restricted_hf_runner_converges_in_zero_interaction_smoke_case() -> None:
    params = TBGParameters.from_degrees(1.2)
    h0 = np.zeros((2, 2, 1), dtype=np.complex128)
    np.fill_diagonal(h0[:, :, 0], [-1.0, 1.0])
    state = RestrictedHartreeFockState(
        h0=h0,
        sigma_z=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
        sigma_ztauz=np.zeros((2, 1), dtype=float),
        nu=0.0,
        v0=coulomb_unit(params),
        precision=1e-8,
        n_spin=1,
        n_eta=1,
        n_band=2,
    )

    run = run_restricted_hartree_fock(
        state,
        empty_overlap_block_set(),
        np.asarray([0.0 + 0.0j], dtype=np.complex128),
        params,
        init_mode="bm",
        max_iter=5,
    )

    assert run.converged
    assert run.exit_reason == "converged"
    assert run.iterations == 1
    assert np.allclose(np.diag(run.state.density[:, :, 0]), [0.5, -0.5])
    assert np.isclose(run.state.mu, 0.0)


def test_restricted_hf_runner_reports_oda_stall_instead_of_false_convergence(monkeypatch) -> None:
    params = TBGParameters.from_degrees(1.2)
    h0 = np.zeros((2, 2, 1), dtype=np.complex128)
    np.fill_diagonal(h0[:, :, 0], [-1.0, 1.0])
    state = RestrictedHartreeFockState(
        h0=h0,
        sigma_z=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
        sigma_ztauz=np.zeros((2, 1), dtype=float),
        nu=0.0,
        v0=coulomb_unit(params),
        precision=1e-8,
        n_spin=1,
        n_eta=1,
        n_band=2,
    )

    monkeypatch.setattr(
        "mean_field.systems.tbg.zero_field.hf.oda_parametrization_restricted",
        lambda *args, **kwargs: 0.0,
    )

    run = run_restricted_hartree_fock(
        state,
        empty_overlap_block_set(),
        np.asarray([0.0 + 0.0j], dtype=np.complex128),
        params,
        init_mode="random",
        seed=1,
        max_iter=5,
    )

    assert not run.converged
    assert run.exit_reason == "oda_stall"
    assert run.iterations == 1
    assert run.iter_err[0] > 0.0


def test_full_hf_runner_matches_julia_mixed_density_convergence_rule(monkeypatch) -> None:
    params = TBGParameters.from_degrees(1.2)
    h0 = np.zeros((2, 2, 1), dtype=np.complex128)
    np.fill_diagonal(h0[:, :, 0], [-1.0, 1.0])
    state = RestrictedHartreeFockState(
        h0=h0,
        sigma_z=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
        sigma_ztauz=np.zeros((2, 1), dtype=float),
        nu=0.0,
        v0=coulomb_unit(params),
        precision=1e-8,
        n_spin=1,
        n_eta=1,
        n_band=2,
    )

    monkeypatch.setattr(
        "mean_field.systems.tbg.zero_field.hf.oda_parametrization_restricted",
        lambda *args, **kwargs: 0.0,
    )

    run = run_full_hartree_fock(
        state,
        empty_overlap_block_set(),
        np.asarray([0.0 + 0.0j], dtype=np.complex128),
        params,
        init_mode="random",
        seed=1,
        max_iter=5,
    )

    assert run.converged
    assert run.exit_reason == "converged"
    assert run.iterations == 1
    assert np.isclose(run.iter_oda[0], 0.0)
    assert np.isclose(run.iter_err[0], 0.0)


def test_restricted_hf_runner_from_bm_solution_accepts_benchmark_alias_modes() -> None:
    solution = _fake_bm_solution()
    run = run_restricted_hf_from_bm_solution(
        solution,
        nu=-2.0,
        init_mode="sp",
        max_iter=2,
    )

    assert run.init_mode == "spindown"
    assert run.iterations >= 1
