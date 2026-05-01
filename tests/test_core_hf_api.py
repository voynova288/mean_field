from __future__ import annotations

import numpy as np

from mean_field.core.hf import (
    FlavorBandData,
    HFOverlapBlockSet,
    ProjectedWavefunctionBasis,
    HartreeFockKernel,
    HartreeFockProblem,
    block_mask,
    build_flavor_band_data,
    build_projected_hf_kernel,
    build_projected_interaction_hamiltonian,
    build_projected_target_hamiltonian,
    calculate_norm_convergence,
    calculate_projected_overlap,
    calculate_projected_overlap_between,
    compute_density_overlap_trace,
    compute_hf_energy,
    contract_fock_term_from_overlap,
    empty_overlap_block_set,
    find_chemical_potential,
    flavor_block_indices,
    flavor_sector_metadata,
    identity_block,
    occupied_state_linear_indices,
    occupied_state_mask,
    project_to_flavor_diagonal,
    run_hartree_fock_problem,
)


def test_core_hf_exports_reusable_flavor_surface() -> None:
    assert FlavorBandData.__name__ == "FlavorBandData"
    assert HartreeFockKernel.__name__ == "HartreeFockKernel"
    assert HartreeFockProblem.__name__ == "HartreeFockProblem"
    assert callable(run_hartree_fock_problem)
    assert ProjectedWavefunctionBasis.__name__ == "ProjectedWavefunctionBasis"
    assert callable(calculate_projected_overlap_between)
    assert callable(build_projected_hf_kernel)
    assert callable(build_projected_interaction_hamiltonian)
    assert callable(contract_fock_term_from_overlap)
    assert empty_overlap_block_set().shifts == ()
    assert flavor_sector_metadata() == (
        ("K_up", "Kprime_up", "K_down", "Kprime_down"),
        ((0, 4), (2, 6), (1, 5), (3, 7)),
    )
    assert flavor_block_indices() == ((0, 4), (2, 6), (1, 5), (3, 7))
    assert np.array_equal(identity_block(2), np.eye(2, dtype=np.complex128))


def test_core_hf_flavor_projection_matches_existing_b0_convention() -> None:
    mask = block_mask()
    assert mask[0, 4]
    assert not mask[0, 1]

    matrix = np.ones((8, 8, 1), dtype=np.complex128)
    projected = project_to_flavor_diagonal(matrix)
    assert projected[0, 4, 0] == 1.0
    assert projected[0, 1, 0] == 0.0


def test_core_hf_band_and_occupation_helpers_keep_fortran_order_rules() -> None:
    hamiltonian = np.zeros((8, 8, 2), dtype=np.complex128)
    np.fill_diagonal(hamiltonian[:, :, 0], np.asarray([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1], dtype=float))
    np.fill_diagonal(hamiltonian[:, :, 1], np.asarray([0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2], dtype=float))

    band_data = build_flavor_band_data(hamiltonian)
    assert band_data.band_labels[0] == "K_up_b1"
    assert band_data.band_labels[-1] == "Kprime_down_b8"

    energies = np.zeros((2, 2), dtype=float)
    assert occupied_state_linear_indices(energies, 2).tolist() == [0, 1]
    assert occupied_state_mask(energies, 2).reshape(-1, order="F").tolist() == [True, True, False, False]
    assert np.isclose(find_chemical_potential(np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float), 0.25), 0.5)


def test_core_hf_norm_convergence_handles_zero_denominator() -> None:
    updated = np.zeros((2, 2, 1), dtype=np.complex128)
    previous = np.zeros_like(updated)
    assert calculate_norm_convergence(updated, previous) == 0.0


def test_core_hf_projected_overlap_uses_flavor_and_spin_diagonal_structure() -> None:
    wavefunctions = np.zeros((4, 1, 2, 2), dtype=np.complex128)
    wavefunctions[:, 0, 0, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    wavefunctions[:, 0, 0, 1] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
    wavefunctions[:, 0, 1, 0] = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)
    wavefunctions[:, 0, 1, 1] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    basis = ProjectedWavefunctionBasis(
        wavefunctions=wavefunctions,
        grid_shape=(1, 1),
        n_spin=2,
        local_basis_size=4,
    )

    overlap_square = calculate_projected_overlap(basis, 0, 0)
    overlap_between = calculate_projected_overlap_between(basis, basis, 0, 0)

    assert overlap_between.shape == (basis.nt, basis.nk, basis.nt, basis.nk)
    assert np.allclose(overlap_square, np.eye(basis.nt * basis.nk, dtype=np.complex128))
    assert np.allclose(overlap_between.reshape(basis.nt * basis.nk, basis.nt * basis.nk, order="F"), overlap_square)

    density = np.zeros((basis.nt, basis.nt, basis.nk), dtype=np.complex128)
    density[:, :, 0] = np.diag([1.0, 2.0, 3.0, 4.0])
    density[:, :, 1] = np.diag([5.0, 6.0, 7.0, 8.0])
    assert compute_density_overlap_trace(density, overlap_between) == 36.0 + 0.0j


def test_core_hf_fock_contraction_numba_path_matches_numpy_path() -> None:
    overlap = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    overlap[:, 0, :, 0] = np.asarray([[1.0, 0.2], [0.0, 0.9]], dtype=np.complex128)
    overlap[:, 0, :, 1] = np.asarray([[0.5, 0.0], [0.1, 1.5]], dtype=np.complex128)
    overlap[:, 1, :, 0] = np.asarray([[1.2, 0.0], [0.3j, 0.7]], dtype=np.complex128)
    overlap[:, 1, :, 1] = np.asarray([[0.8, 0.4], [0.0, 1.1]], dtype=np.complex128)
    density = np.zeros((2, 2, 2), dtype=np.complex128)
    density[:, :, 0] = np.asarray([[0.5, 0.1j], [-0.1j, -0.25]], dtype=np.complex128)
    density[:, :, 1] = np.asarray([[0.2, 0.3], [0.3, -0.4]], dtype=np.complex128)
    coeff = np.asarray([[1.0, 0.5], [0.2, 0.8]], dtype=float)

    numpy_result = contract_fock_term_from_overlap(overlap, density, coeff, use_numba=False)
    optional_numba_result = contract_fock_term_from_overlap(overlap, density, coeff, use_numba=True)

    assert np.allclose(optional_numba_result, numpy_result)


def test_core_hf_projected_interaction_builds_hartree_and_fock_terms_from_precomputed_blocks() -> None:
    density = np.zeros((2, 2, 2), dtype=np.complex128)
    density[:, :, 0] = np.diag([0.5, -0.5])
    density[:, :, 1] = np.diag([-0.5, 0.5])

    overlap = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    overlap[:, 0, :, 0] = np.eye(2, dtype=np.complex128)
    overlap[:, 1, :, 1] = np.eye(2, dtype=np.complex128)
    diagonal = np.diagonal(overlap, axis1=1, axis2=3)
    blocks = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        overlaps={(0, 0): overlap},
        diagonal_overlaps={(0, 0): diagonal},
        hartree_screening={(0, 0): 2.0},
        fock_screening={(0, 0): np.ones((2, 2), dtype=float)},
    )

    interaction = build_projected_interaction_hamiltonian(density, blocks, v0=4.0, use_numba=False)

    hartree_trace = compute_density_overlap_trace(density, overlap, use_numba=False)
    expected = 4.0 * 2.0 / 2.0 * hartree_trace * diagonal
    expected -= contract_fock_term_from_overlap(overlap, density, np.full((2, 2), 2.0), use_numba=False)
    assert np.allclose(interaction, expected)


def test_core_hf_projected_target_hamiltonian_reuses_source_density_on_rectangular_path() -> None:
    base = np.zeros((2, 2, 3), dtype=np.complex128)
    density = np.zeros((2, 2, 2), dtype=np.complex128)
    density[:, :, 0] = np.diag([0.5, -0.5])
    density[:, :, 1] = np.diag([0.25, -0.25])

    source_overlap = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    source_overlap[:, 0, :, 0] = np.eye(2, dtype=np.complex128)
    source_overlap[:, 1, :, 1] = np.eye(2, dtype=np.complex128)
    target_overlap = np.zeros((2, 3, 2, 3), dtype=np.complex128)
    for ik in range(3):
        target_overlap[:, ik, :, ik] = np.eye(2, dtype=np.complex128)
    target_source_overlap = np.zeros((2, 3, 2, 2), dtype=np.complex128)
    for ik_target in range(3):
        for ik_source in range(2):
            target_source_overlap[:, ik_target, :, ik_source] = np.eye(2, dtype=np.complex128)

    source_blocks = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        overlaps={(0, 0): source_overlap},
        diagonal_overlaps={(0, 0): np.diagonal(source_overlap, axis1=1, axis2=3)},
        hartree_screening={(0, 0): 0.0},
    )
    target_blocks = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        overlaps={(0, 0): target_overlap},
        diagonal_overlaps={(0, 0): np.diagonal(target_overlap, axis1=1, axis2=3)},
    )
    target_source_blocks = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        overlaps={(0, 0): target_source_overlap},
        fock_screening={(0, 0): np.ones((3, 2), dtype=float)},
    )

    hamiltonian = build_projected_target_hamiltonian(
        base,
        density,
        source_overlap_blocks=source_blocks,
        target_overlap_blocks=target_blocks,
        target_source_overlap_blocks=target_source_blocks,
        v0=2.0,
        use_numba=False,
    )

    expected_fock = contract_fock_term_from_overlap(
        target_source_overlap,
        density,
        np.ones((3, 2), dtype=float),
        use_numba=False,
    )
    assert np.allclose(hamiltonian, -expected_fock)
    assert np.isclose(compute_hf_energy(np.zeros_like(density), np.zeros_like(density), density), 0.0)
