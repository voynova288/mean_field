from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf import (
    DiskBackedArrayMapping,
    FlavorBandData,
    HFOverlapBlockSet,
    ProjectedWavefunctionBasis,
    HartreeFockKernel,
    apply_random_projector_rotation,
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
    conventional_projector_to_stored,
    empty_overlap_block_set,
    density_from_fixed_sector_occupations,
    estimate_hf_overlap_cache_bytes,
    find_chemical_potential,
    flatten_sector_blocks,
    flavor_block_indices,
    flavor_sector_metadata,
    identity_block,
    occupied_state_linear_indices,
    occupied_state_mask,
    project_to_flavor_diagonal,
    run_hartree_fock_problem,
    sector_block_energies,
    shift_wavefunction_grid,
    should_spill_hf_overlap_cache,
    stored_projector_to_conventional,
    unflatten_sector_blocks,
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


def test_core_hf_sector_layout_and_fixed_occupation_density_are_system_agnostic() -> None:
    blocks = np.zeros((2, 2, 2, 2, 1), dtype=np.complex128)
    for ispin in range(2):
        for ieta in range(2):
            block = np.diag([10.0 * ispin + ieta, 10.0 * ispin + ieta + 1.0]).astype(np.complex128)
            blocks[ispin, ieta, :, :, 0] = block

    flat = flatten_sector_blocks(blocks)
    restored = unflatten_sector_blocks(flat, n_spin=2, n_eta=2, nb=2)

    assert flat.shape == (8, 8, 1)
    assert np.allclose(restored, blocks)

    sector_energies = sector_block_energies(blocks)
    density, energies = density_from_fixed_sector_occupations(
        blocks,
        np.asarray([[1, 2], [0, 1]], dtype=int),
        reference_diagonal=np.asarray([1.0, 0.0]),
    )

    assert sector_energies.shape == (2, 2, 2, 1)
    assert np.allclose(sector_energies, energies)
    assert energies.shape == (2, 2, 2, 1)
    assert np.allclose(density[0, 0, :, :, 0], np.diag([0.0, 0.0]))
    assert np.allclose(density[0, 1, :, :, 0], np.diag([0.0, 1.0]))
    assert np.allclose(density[1, 0, :, :, 0], np.diag([-1.0, 0.0]))


def test_core_hf_stored_projector_convention_is_matrix_axis_transpose() -> None:
    conventional = np.zeros((2, 2, 2), dtype=np.complex128)
    conventional[:, :, 0] = np.asarray([[1.0, 2.0 + 3.0j], [4.0 + 5.0j, 6.0]], dtype=np.complex128)
    conventional[:, :, 1] = np.asarray([[0.0, 7.0 - 2.0j], [8.0 + 1.0j, 9.0]], dtype=np.complex128)

    stored = conventional_projector_to_stored(conventional)

    assert np.allclose(stored[:, :, 0], conventional[:, :, 0].T)
    assert not np.allclose(stored[:, :, 0], np.conj(conventional[:, :, 0]))
    assert np.allclose(stored_projector_to_conventional(stored), conventional)


def test_core_hf_apply_random_projector_rotation_is_deterministic_in_place() -> None:
    density_1 = np.zeros((2, 2, 2), dtype=np.complex128)
    density_2 = np.zeros_like(density_1)
    reference = np.zeros_like(density_1)
    for ik in range(2):
        density_1[:, :, ik] = np.diag([1.0, 0.0])
        density_2[:, :, ik] = np.diag([1.0, 0.0])

    apply_random_projector_rotation(density_1, reference_density=reference, alpha=0.2, seed=7)
    apply_random_projector_rotation(density_2, reference_density=reference, alpha=0.2, seed=7)

    assert not np.allclose(density_1[:, :, 0], np.diag([1.0, 0.0]))
    np.testing.assert_allclose(density_1, density_2)


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


def test_core_hf_shift_wavefunction_grid_supports_zero_fill_and_periodic_modes() -> None:
    grid = np.arange(2 * 3 * 4 * 1, dtype=float).reshape(2, 3, 4, 1).astype(np.complex128)

    shifted = shift_wavefunction_grid(grid, 1, -2, boundary_mode="zero_fill")
    expected = np.zeros_like(grid)
    expected[:, 1:, :2, :] = grid[:, :2, 2:, :]
    np.testing.assert_allclose(shifted, expected)

    periodic = shift_wavefunction_grid(grid, 1, -2, boundary_mode="periodic")
    np.testing.assert_allclose(periodic, np.roll(grid, shift=(1, -2), axis=(1, 2)))


def test_core_hf_projected_overlap_zero_fill_drops_boundary_wraps() -> None:
    source = np.zeros((9, 1, 1, 1), dtype=np.complex128)
    target = np.zeros((9, 1, 1, 1), dtype=np.complex128)

    def index(ix: int, iy: int) -> int:
        return ix + 3 * iy

    source[index(0, 1), 0, 0, 0] = 1.0
    target[index(2, 1), 0, 0, 0] = 1.0

    periodic_target = ProjectedWavefunctionBasis(target, grid_shape=(3, 3), local_basis_size=1)
    periodic_source = ProjectedWavefunctionBasis(source, grid_shape=(3, 3), local_basis_size=1)
    zero_fill_target = ProjectedWavefunctionBasis(
        target,
        grid_shape=(3, 3),
        local_basis_size=1,
        boundary_mode="zero_fill",
    )
    zero_fill_source = ProjectedWavefunctionBasis(
        source,
        grid_shape=(3, 3),
        local_basis_size=1,
        boundary_mode="zero_fill",
    )

    periodic_overlap = calculate_projected_overlap_between(periodic_target, periodic_source, 1, 0)[0, 0, 0, 0]
    zero_fill_overlap = calculate_projected_overlap_between(zero_fill_target, zero_fill_source, 1, 0)[0, 0, 0, 0]

    assert periodic_overlap == 1.0
    assert zero_fill_overlap == 0.0
    with pytest.raises(ValueError, match="boundary_mode mismatch"):
        calculate_projected_overlap_between(periodic_target, zero_fill_source, 1, 0)


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

def test_core_hf_overlap_cache_estimate_matches_dense_shape() -> None:
    estimate = estimate_hf_overlap_cache_bytes(nt=48, nk_target=24 * 24, n_shifts=25)

    assert estimate.overlap_bytes_per_shift == 48 * 576 * 48 * 576 * np.dtype(np.complex128).itemsize
    assert estimate.overlap_bytes_total == 25 * estimate.overlap_bytes_per_shift
    assert estimate.diagonal_bytes_total == 25 * 48 * 48 * 576 * np.dtype(np.complex128).itemsize
    assert estimate.fock_screening_bytes_total == 25 * 576 * 576 * np.dtype(np.float64).itemsize
    assert should_spill_hf_overlap_cache(estimate, memory_limit_bytes=256 * 1024**3, safety_fraction=0.65)
    assert not should_spill_hf_overlap_cache(estimate, memory_limit_bytes=1024 * 1024**3, safety_fraction=0.65)

def test_core_hf_disk_backed_array_mapping_roundtrip_lazy_loads(tmp_path) -> None:
    mapping = DiskBackedArrayMapping(tmp_path)
    key = (-2, 1)
    array = (np.arange(12, dtype=float).reshape(3, 4) + 1j).astype(np.complex128)

    mapping[key] = array
    reloaded = mapping[key]

    assert key in mapping
    assert reloaded.shape == array.shape
    assert reloaded.dtype == array.dtype
    assert np.array_equal(reloaded, array)

    reopened = DiskBackedArrayMapping(tmp_path)
    assert np.array_equal(reopened[key], array)

def test_core_hf_disk_backed_overlap_blocks_match_in_memory_interaction(tmp_path) -> None:
    density = np.zeros((2, 2, 2), dtype=np.complex128)
    density[:, :, 0] = np.diag([0.5, -0.5])
    density[:, :, 1] = np.diag([-0.5, 0.5])

    overlap = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    overlap[:, 0, :, 0] = np.eye(2, dtype=np.complex128)
    overlap[:, 1, :, 1] = np.eye(2, dtype=np.complex128)
    diagonal = np.diagonal(overlap, axis1=1, axis2=3)
    fock = np.ones((2, 2), dtype=float)

    memory_blocks = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        overlaps={(0, 0): overlap},
        diagonal_overlaps={(0, 0): diagonal},
        hartree_screening={(0, 0): 2.0},
        fock_screening={(0, 0): fock},
    )
    overlap_store = DiskBackedArrayMapping(tmp_path / "overlaps")
    diagonal_store = DiskBackedArrayMapping(tmp_path / "diagonal")
    overlap_store[(0, 0)] = overlap
    diagonal_store[(0, 0)] = diagonal
    disk_blocks = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        overlaps=overlap_store,
        diagonal_overlaps=diagonal_store,
        hartree_screening={(0, 0): 2.0},
        fock_screening={(0, 0): fock},
    )

    expected = build_projected_interaction_hamiltonian(density, memory_blocks, v0=4.0, use_numba=False)
    actual = build_projected_interaction_hamiltonian(density, disk_blocks, v0=4.0, use_numba=False)

    assert np.allclose(actual, expected)
