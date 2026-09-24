from __future__ import annotations

import numpy as np
import pytest
from scipy.special import xlogy

from mean_field.core.hf.overlap_cache import (
    DiskBackedArrayMapping,
    estimate_hf_overlap_cache_bytes,
    should_spill_hf_overlap_cache,
)
from mean_field.core.hf.flavors import (
    FlavorBandData,
    block_mask,
    build_flavor_band_data,
    flavor_block_indices,
    flavor_sector_metadata,
    identity_block,
    project_to_flavor_diagonal,
    sector_block_energies,
)
from mean_field.core.hf.overlap import (
    HFOverlapBlockSet,
    ProjectedWavefunctionBasis,
    build_projected_overlap_block_set,
    calculate_projected_overlap,
    calculate_projected_overlap_between,
    compute_density_overlap_trace,
    contract_fock_action_from_overlap,
    contract_fock_term_from_overlap,
    shift_wavefunction_grid,
)
from mean_field.core.hf.problem import (
    HartreeFockKernel,
    HartreeFockProblem,
    run_hartree_fock_problem,
)
from mean_field.core.hf.occupations import (
    apply_random_projector_rotation,
    calculate_norm_convergence,
    conventional_projector_to_stored,
    density_from_fixed_sector_occupations,
    fermionic_density_diagnostics,
    find_chemical_potential,
    global_canonical_occupations,
    global_fermi_occupations,
    helmholtz_free_energy_ev,
    flatten_sector_blocks,
    occupied_state_linear_indices,
    occupied_state_mask,
    stored_projector_to_conventional,
    unflatten_sector_blocks,
)
from mean_field.core.hf.interaction import (
    build_projected_hf_kernel,
    build_projected_interaction_hamiltonian,
    build_projected_target_hamiltonian,
    compute_hf_energy,
    empty_overlap_block_set,
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
    assert callable(contract_fock_action_from_overlap)
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


def test_global_fermi_occupations_fractionally_fills_exact_degeneracy() -> None:
    hamiltonian = np.zeros((2, 2, 2), dtype=np.complex128)
    result = global_fermi_occupations(
        hamiltonian,
        target_particle_number=2.0,
        kbt_ev=5.0e-4,
    )

    assert result.chemical_potential == pytest.approx(0.0, abs=1.0e-14)
    np.testing.assert_allclose(result.occupations, 0.5, atol=1.0e-14)
    for density in result.density_ket:
        np.testing.assert_allclose(density, 0.5 * np.eye(2), atol=1.0e-14)
    assert result.particle_number == pytest.approx(2.0, abs=1.0e-12)
    assert result.particle_residual == pytest.approx(0.0, abs=1.0e-12)
    assert result.entropy_dimensionless == pytest.approx(4.0 * np.log(2.0))
    diagnostics = fermionic_density_diagnostics(result.density_ket)
    assert diagnostics.particle_number == pytest.approx(result.particle_number)
    assert diagnostics.entropy_dimensionless == pytest.approx(
        result.entropy_dimensionless
    )
    assert result.helmholtz_free_energy_ev(1.0) == pytest.approx(
        1.0 - result.kbt_ev * result.entropy_dimensionless
    )


def test_global_occupation_parallel_eigensolver_matches_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    hamiltonian = np.asarray(
        (
            [[-0.8, 0.1 + 0.03j], [0.1 - 0.03j, 0.4]],
            [[-0.5, -0.07j], [0.07j, 0.9]],
            [[-0.2, 0.04 - 0.02j], [0.04 + 0.02j, 1.3]],
        ),
        dtype=np.complex128,
    )
    canonical_serial = global_canonical_occupations(
        hamiltonian,
        total_occupied=3,
        fermi_degeneracy_tolerance=1.0e-12,
        eigensolver_workers=1,
    )
    canonical_parallel = global_canonical_occupations(
        hamiltonian,
        total_occupied=3,
        fermi_degeneracy_tolerance=1.0e-12,
        eigensolver_workers=3,
    )
    np.testing.assert_array_equal(
        canonical_parallel.energies, canonical_serial.energies
    )
    np.testing.assert_array_equal(
        canonical_parallel.projector_ket, canonical_serial.projector_ket
    )
    assert canonical_parallel.chemical_potential == canonical_serial.chemical_potential
    assert canonical_parallel.fermi_gap == canonical_serial.fermi_gap

    fermi_serial = global_fermi_occupations(
        hamiltonian,
        target_particle_number=2.75,
        kbt_ev=0.08,
        eigensolver_workers=1,
    )
    fermi_parallel = global_fermi_occupations(
        hamiltonian,
        target_particle_number=2.75,
        kbt_ev=0.08,
        eigensolver_workers=3,
    )
    np.testing.assert_array_equal(fermi_parallel.energies, fermi_serial.energies)
    np.testing.assert_array_equal(fermi_parallel.density_ket, fermi_serial.density_ket)
    np.testing.assert_array_equal(fermi_parallel.occupations, fermi_serial.occupations)
    assert fermi_parallel.chemical_potential == fermi_serial.chemical_potential
    with pytest.raises(TypeError, match="eigensolver_workers"):
        global_fermi_occupations(
            hamiltonian,
            target_particle_number=2.75,
            kbt_ev=0.08,
            eigensolver_workers=True,
        )
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="one-thread"):
        global_fermi_occupations(
            hamiltonian,
            target_particle_number=2.75,
            kbt_ev=0.08,
            eigensolver_workers=2,
        )


def test_global_fermi_occupations_uses_explicit_k_weight_normalization() -> None:
    temperature = 0.2
    expected_mu = 0.3
    energies = np.asarray([0.0, 1.0])
    weights = np.asarray([2.0, 1.0])
    expected_occupations = 1.0 / (1.0 + np.exp((energies - expected_mu) / temperature))
    target = float(weights @ expected_occupations)
    hamiltonian = energies[:, None, None].astype(np.complex128)
    result = global_fermi_occupations(
        hamiltonian,
        target_particle_number=target,
        kbt_ev=temperature,
        k_weights=weights,
    )
    rescaled = global_fermi_occupations(
        hamiltonian,
        target_particle_number=10.0 * target,
        kbt_ev=temperature,
        k_weights=10.0 * weights,
    )

    np.testing.assert_allclose(result.occupations.ravel(), expected_occupations)
    np.testing.assert_allclose(result.k_weights, weights)
    np.testing.assert_allclose(rescaled.occupations, result.occupations, atol=1.0e-12)
    assert result.chemical_potential == pytest.approx(expected_mu, abs=1.0e-12)
    assert result.particle_number == pytest.approx(target, abs=3.0e-12)


def test_global_fermi_occupations_builds_complex_ket_density_and_is_shift_invariant() -> None:
    temperature = 0.15
    expected_mu = 0.17
    energies = np.asarray([-0.2, 0.8])
    unitary = np.asarray([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)
    expected_occupations = 1.0 / (1.0 + np.exp((energies - expected_mu) / temperature))
    expected_density = (unitary * expected_occupations[None, :]) @ unitary.conj().T
    hamiltonian = ((unitary * energies[None, :]) @ unitary.conj().T)[None, :, :]
    target = float(np.sum(expected_occupations))

    result = global_fermi_occupations(
        hamiltonian,
        target_particle_number=target,
        kbt_ev=temperature,
    )
    shifted = global_fermi_occupations(
        hamiltonian + 1.0e6 * np.eye(2)[None, :, :],
        target_particle_number=target,
        kbt_ev=temperature,
    )

    np.testing.assert_allclose(result.density_ket[0], expected_density, atol=1.0e-12)
    np.testing.assert_allclose(result.density_ket, result.density_ket.conj().transpose(0, 2, 1))
    np.testing.assert_allclose(shifted.occupations, result.occupations, atol=2.0e-10)
    assert result.chemical_potential == pytest.approx(expected_mu, abs=2.0e-12)
    assert shifted.chemical_potential - 1.0e6 == pytest.approx(expected_mu, abs=2.0e-10)


def test_global_fermi_occupations_recovers_gapped_zero_temperature_limit() -> None:
    hamiltonian = np.asarray(
        (np.diag([-2.0, -1.0]), np.diag([1.0, 2.0])),
        dtype=np.complex128,
    )
    result = global_fermi_occupations(
        hamiltonian,
        target_particle_number=2.0,
        kbt_ev=1.0e-3,
    )

    np.testing.assert_allclose(result.density_ket[0], np.eye(2), atol=1.0e-14)
    np.testing.assert_allclose(result.density_ket[1], np.zeros((2, 2)), atol=1.0e-14)
    assert -1.0 < result.chemical_potential < 1.0
    assert result.entropy_dimensionless < 1.0e-12


def test_global_fermi_occupations_handles_empty_and_full_endpoints() -> None:
    hamiltonian = np.asarray((np.diag([-1.0, 2.0]),), dtype=np.complex128)
    empty = global_fermi_occupations(
        hamiltonian,
        target_particle_number=0.0,
        kbt_ev=0.1,
    )
    full = global_fermi_occupations(
        hamiltonian,
        target_particle_number=2.0,
        kbt_ev=0.1,
    )

    assert empty.chemical_potential == float("-inf")
    assert full.chemical_potential == float("inf")
    np.testing.assert_array_equal(empty.density_ket, np.zeros((1, 2, 2)))
    np.testing.assert_array_equal(full.density_ket, np.eye(2)[None, :, :])
    assert empty.entropy_dimensionless == pytest.approx(0.0)
    assert full.entropy_dimensionless == pytest.approx(0.0)


def test_fermionic_density_diagnostics_parallel_matches_serial_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    unitary = np.asarray([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)
    fractional_spectra = np.asarray(
        ([0.05, 0.85], [0.2, 0.65], [0.4, 0.95]), dtype=float
    )
    density = np.asarray(
        [
            (unitary * occupations[None, :]) @ unitary.conj().T
            for occupations in fractional_spectra
        ],
        dtype=np.complex128,
    )
    weights = np.asarray([0.25, 1.5, 0.75], dtype=float)

    serial = fermionic_density_diagnostics(
        density,
        k_weights=weights,
        eigensolver_workers=1,
    )
    parallel = fermionic_density_diagnostics(
        density,
        k_weights=weights,
        eigensolver_workers=3,
    )

    assert np.any(np.abs(density.imag) > 0.0)
    assert parallel.particle_number == serial.particle_number
    assert parallel.entropy_dimensionless == serial.entropy_dimensionless
    assert parallel.eigenvalue_min == serial.eigenvalue_min
    assert parallel.eigenvalue_max == serial.eigenvalue_max
    assert parallel.hermiticity_residual == serial.hermiticity_residual
    np.testing.assert_array_equal(parallel.k_weights, serial.k_weights)

    # Preserve the pre-parallel batched-eigvalsh scalar arithmetic exactly.
    legacy_eigenvalues = np.linalg.eigvalsh(density).T
    legacy_clipped = np.clip(legacy_eigenvalues, 0.0, 1.0)
    legacy_entropy_terms = xlogy(legacy_clipped, legacy_clipped) + xlogy(
        1.0 - legacy_clipped, 1.0 - legacy_clipped
    )
    assert serial.particle_number == float(
        np.einsum("k,bk->", weights, legacy_clipped, optimize=True)
    )
    assert serial.entropy_dimensionless == -float(
        np.einsum("k,bk->", weights, legacy_entropy_terms, optimize=True)
    )


def test_fermionic_density_diagnostics_rejects_invalid_workers_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    density = np.asarray((np.diag([0.25, 0.75]),), dtype=np.complex128)
    with pytest.raises(TypeError, match="eigensolver_workers"):
        fermionic_density_diagnostics(density, eigensolver_workers=True)
    with pytest.raises(TypeError, match="eigensolver_workers"):
        fermionic_density_diagnostics(density, eigensolver_workers=0)

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("BLIS_NUM_THREADS", "1")
    with pytest.raises(RuntimeError, match="one-thread"):
        fermionic_density_diagnostics(density, eigensolver_workers=2)


def test_fermionic_density_diagnostics_and_free_energy_fail_closed() -> None:
    with pytest.raises(ValueError, match="spectrum lies outside"):
        fermionic_density_diagnostics(
            np.asarray((np.diag([-0.01, 1.01]),), dtype=np.complex128),
            spectrum_tolerance=1.0e-4,
        )
    with pytest.raises(ValueError, match="Hermitian"):
        fermionic_density_diagnostics(
            np.asarray(([[0.5, 0.1j], [0.1j, 0.5]],), dtype=np.complex128)
        )
    with pytest.raises(ValueError, match="nonnegative"):
        helmholtz_free_energy_ev(1.0, kbt_ev=-0.1, entropy_dimensionless=0.5)


def test_global_fermi_occupations_rejects_inconsistent_inputs() -> None:
    hamiltonian = np.zeros((2, 1, 1), dtype=np.complex128)
    with pytest.raises(ValueError, match="strictly positive"):
        global_fermi_occupations(
            hamiltonian,
            target_particle_number=1.0,
            kbt_ev=0.0,
        )
    with pytest.raises(ValueError, match="k_weights must have shape"):
        global_fermi_occupations(
            hamiltonian,
            target_particle_number=1.0,
            kbt_ev=0.1,
            k_weights=np.ones(3),
        )
    with pytest.raises(ValueError, match="outside the weighted"):
        global_fermi_occupations(
            hamiltonian,
            target_particle_number=3.0,
            kbt_ev=0.1,
        )
    with pytest.raises(ValueError, match="positive total weight"):
        global_fermi_occupations(
            hamiltonian,
            target_particle_number=0.0,
            kbt_ev=0.1,
            k_weights=np.zeros(2),
        )
    with pytest.raises(ValueError, match="at least one k point and band"):
        global_fermi_occupations(
            np.empty((2, 0, 0), dtype=np.complex128),
            target_particle_number=0.0,
            kbt_ev=0.1,
        )


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


def _explicit_fock_term_from_overlap(overlap: np.ndarray, density: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    nt, nk_target, _nt_rhs, nk_source = overlap.shape
    expected = np.zeros((nt, nt, nk_target), dtype=np.complex128)
    for ik_target in range(nk_target):
        for ik_source in range(nk_source):
            lam = overlap[:, ik_target, :, ik_source]
            expected[:, :, ik_target] += coeff[ik_target, ik_source] * (
                lam @ density[:, :, ik_source].T @ lam.conjugate().T
            )
    return expected


def test_core_hf_fock_contraction_paths_match_explicit_formula() -> None:
    overlap = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    overlap[:, 0, :, 0] = np.asarray([[1.0, 0.2], [0.0, 0.9]], dtype=np.complex128)
    overlap[:, 0, :, 1] = np.asarray([[0.5, 0.0], [0.1, 1.5]], dtype=np.complex128)
    overlap[:, 1, :, 0] = np.asarray([[1.2, 0.0], [0.3j, 0.7]], dtype=np.complex128)
    overlap[:, 1, :, 1] = np.asarray([[0.8, 0.4], [0.0, 1.1]], dtype=np.complex128)
    density = np.zeros((2, 2, 2), dtype=np.complex128)
    density[:, :, 0] = np.asarray([[0.5, 0.1j], [-0.1j, -0.25]], dtype=np.complex128)
    density[:, :, 1] = np.asarray([[0.2, 0.3], [0.3, -0.4]], dtype=np.complex128)
    real_coeff = np.asarray([[1.0, 0.5], [0.2, 0.8]], dtype=float)
    complex_coeff = real_coeff.astype(np.complex128) * (1.0 + 0.25j)

    for coeff in (real_coeff, complex_coeff):
        expected = _explicit_fock_term_from_overlap(overlap, density, coeff)
        numpy_result = contract_fock_term_from_overlap(overlap, density, coeff, use_numba=False)
        optional_numba_result = contract_fock_term_from_overlap(overlap, density, coeff, use_numba=True)

        np.testing.assert_allclose(numpy_result, expected, atol=1.0e-14)
        np.testing.assert_allclose(optional_numba_result, expected, atol=1.0e-14)


def test_core_hf_fock_contraction_supports_different_target_and_source_dimensions() -> None:
    rng = np.random.default_rng(17)
    overlap = rng.normal(size=(3, 2, 2, 2)) + 1.0j * rng.normal(size=(3, 2, 2, 2))
    density = rng.normal(size=(2, 2, 2)) + 1.0j * rng.normal(size=(2, 2, 2))
    coeff = rng.normal(size=(2, 2))

    expected = _explicit_fock_term_from_overlap(overlap, density, coeff)
    numpy_result = contract_fock_term_from_overlap(overlap, density, coeff, use_numba=False)
    optional_numba_result = contract_fock_term_from_overlap(overlap, density, coeff, use_numba=True)

    assert numpy_result.shape == (3, 3, 2)
    np.testing.assert_allclose(numpy_result, expected, atol=1.0e-13)
    np.testing.assert_allclose(optional_numba_result, expected, atol=1.0e-13)


def test_core_hf_fock_action_matches_dense_rectangular_contraction() -> None:
    rng = np.random.default_rng(101)
    overlap = rng.normal(size=(4, 3, 2, 2)) + 1.0j * rng.normal(size=(4, 3, 2, 2))
    density = rng.normal(size=(2, 2, 2)) + 1.0j * rng.normal(size=(2, 2, 2))
    coeff = rng.normal(size=(3, 2)) + 1.0j * rng.normal(size=(3, 2))
    vectors = rng.normal(size=(4, 5, 3)) + 1.0j * rng.normal(size=(4, 5, 3))

    dense_fock = contract_fock_term_from_overlap(overlap, density, coeff, use_numba=False)
    expected = np.einsum("abt,brt->art", dense_fock, vectors, optimize=True)
    actual = contract_fock_action_from_overlap(overlap, density, coeff, vectors)

    assert actual.shape == vectors.shape
    np.testing.assert_allclose(actual, expected, atol=2.0e-13)


def test_core_hf_fock_action_identity_vectors_recover_dense_columns() -> None:
    rng = np.random.default_rng(103)
    overlap = rng.normal(size=(3, 2, 2, 3)) + 1.0j * rng.normal(size=(3, 2, 2, 3))
    density = rng.normal(size=(2, 2, 3)) + 1.0j * rng.normal(size=(2, 2, 3))
    coeff = rng.normal(size=(2, 3)) + 1.0j * rng.normal(size=(2, 3))
    identity_vectors = np.broadcast_to(np.eye(3, dtype=np.complex128)[:, :, None], (3, 3, 2))

    dense_fock = contract_fock_term_from_overlap(overlap, density, coeff, use_numba=False)
    action = contract_fock_action_from_overlap(overlap, density, coeff, identity_vectors)

    np.testing.assert_allclose(action, dense_fock, atol=2.0e-13)


def test_core_hf_fock_action_rejects_invalid_shapes_and_dtypes() -> None:
    overlap = np.zeros((3, 2, 2, 4), dtype=np.complex128)
    density = np.zeros((2, 2, 4), dtype=np.complex128)
    coeff = np.zeros((2, 4), dtype=np.complex128)
    vectors = np.zeros((3, 5, 2), dtype=np.complex128)

    with pytest.raises(ValueError, match="Expected overlap shape"):
        contract_fock_action_from_overlap(overlap[:, :, :, 0], density, coeff, vectors)
    with pytest.raises(ValueError, match="Expected density shape"):
        contract_fock_action_from_overlap(overlap, density[:, :, :3], coeff, vectors)
    with pytest.raises(ValueError, match="Expected coeff_matrix shape"):
        contract_fock_action_from_overlap(overlap, density, coeff[:, :3], vectors)
    with pytest.raises(ValueError, match="Expected vectors shape"):
        contract_fock_action_from_overlap(overlap, density, coeff, vectors[:, :, :1])
    with pytest.raises(TypeError, match="vectors must have a real or complex numeric dtype"):
        contract_fock_action_from_overlap(overlap, density, coeff, vectors.astype(object))


def test_core_hf_fock_action_is_target_source_coordinate_covariant() -> None:
    rng = np.random.default_rng(107)
    nt_target, nk_target = 3, 2
    nt_source, nk_source = 2, 3
    overlap = rng.normal(size=(nt_target, nk_target, nt_source, nk_source)) + 1.0j * rng.normal(
        size=(nt_target, nk_target, nt_source, nk_source)
    )
    density = rng.normal(size=(nt_source, nt_source, nk_source)) + 1.0j * rng.normal(
        size=(nt_source, nt_source, nk_source)
    )
    coeff = rng.normal(size=(nk_target, nk_source)) + 1.0j * rng.normal(size=(nk_target, nk_source))
    vectors = rng.normal(size=(nt_target, 4, nk_target)) + 1.0j * rng.normal(size=(nt_target, 4, nk_target))
    target_q = np.asarray(
        [np.linalg.qr(rng.normal(size=(nt_target, nt_target)) + 1.0j * rng.normal(size=(nt_target, nt_target)))[0]
         for _ in range(nk_target)]
    )
    source_q = np.asarray(
        [np.linalg.qr(rng.normal(size=(nt_source, nt_source)) + 1.0j * rng.normal(size=(nt_source, nt_source)))[0]
         for _ in range(nk_source)]
    )

    rotated_overlap = np.empty_like(overlap)
    for ik_target in range(nk_target):
        for ik_source in range(nk_source):
            rotated_overlap[:, ik_target, :, ik_source] = (
                target_q[ik_target].conjugate().T
                @ overlap[:, ik_target, :, ik_source]
                @ source_q[ik_source]
            )
    rotated_density = np.empty_like(density)
    for ik_source in range(nk_source):
        rotated_density[:, :, ik_source] = (
            source_q[ik_source].T @ density[:, :, ik_source] @ source_q[ik_source].conjugate()
        )
    rotated_vectors = np.empty_like(vectors)
    for ik_target in range(nk_target):
        rotated_vectors[:, :, ik_target] = target_q[ik_target].conjugate().T @ vectors[:, :, ik_target]

    original = contract_fock_action_from_overlap(overlap, density, coeff, vectors)
    rotated = contract_fock_action_from_overlap(rotated_overlap, rotated_density, coeff, rotated_vectors)
    expected = np.empty_like(original)
    for ik_target in range(nk_target):
        expected[:, :, ik_target] = target_q[ik_target].conjugate().T @ original[:, :, ik_target]

    np.testing.assert_allclose(rotated, expected, atol=5.0e-13)


def test_rectangular_overlap_builder_defaults_to_no_diagonal_blocks() -> None:
    target_wavefunctions = np.zeros((3, 2, 1, 1), dtype=np.complex128)
    target_wavefunctions[:, :, 0, 0] = np.eye(3, dtype=np.complex128)[:, :2]
    source_wavefunctions = np.zeros((3, 1, 1, 1), dtype=np.complex128)
    source_wavefunctions[:, 0, 0, 0] = np.eye(3, dtype=np.complex128)[:, 0]
    target = ProjectedWavefunctionBasis(target_wavefunctions, grid_shape=(1, 1))
    source = ProjectedWavefunctionBasis(source_wavefunctions, grid_shape=(1, 1))

    blocks = build_projected_overlap_block_set(
        target,
        source=source,
        shifts=((0, 0),),
    )
    assert blocks.overlaps[(0, 0)].shape == (2, 1, 1, 1)
    assert blocks.diagonal_overlaps == {}
    with pytest.raises(ValueError, match="Diagonal overlaps require identical"):
        build_projected_overlap_block_set(
            target,
            source=source,
            shifts=((0, 0),),
            include_diagonal_overlaps=True,
        )


def test_rectangular_fock_contraction_is_target_source_coordinate_covariant() -> None:
    rng = np.random.default_rng(23)
    overlap = rng.normal(size=(3, 2, 2, 2)) + 1.0j * rng.normal(size=(3, 2, 2, 2))
    density = rng.normal(size=(2, 2, 2)) + 1.0j * rng.normal(size=(2, 2, 2))
    coeff = rng.normal(size=(2, 2))
    target_q, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1.0j * rng.normal(size=(3, 3)))
    source_q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2)))

    rotated_overlap = np.empty_like(overlap)
    for ik_target in range(2):
        for ik_source in range(2):
            rotated_overlap[:, ik_target, :, ik_source] = (
                target_q.conjugate().T
                @ overlap[:, ik_target, :, ik_source]
                @ source_q
            )
    rotated_density = np.asarray(
        [source_q.T @ density[:, :, ik] @ source_q.conjugate() for ik in range(2)]
    ).transpose(1, 2, 0)

    original = contract_fock_term_from_overlap(overlap, density, coeff, use_numba=False)
    rotated = contract_fock_term_from_overlap(
        rotated_overlap,
        rotated_density,
        coeff,
        use_numba=False,
    )
    expected = np.asarray(
        [target_q.conjugate().T @ original[:, :, ik] @ target_q for ik in range(2)]
    ).transpose(1, 2, 0)
    np.testing.assert_allclose(rotated, expected, atol=2.0e-13)


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
    rng = np.random.default_rng(29)
    base = np.zeros((3, 3, 3), dtype=np.complex128)
    density = np.zeros((2, 2, 2), dtype=np.complex128)
    density[:, :, 0] = np.diag([0.5, -0.5])
    density[:, :, 1] = np.diag([0.25, -0.25])

    source_overlap = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    source_overlap[:, 0, :, 0] = np.eye(2, dtype=np.complex128)
    source_overlap[:, 1, :, 1] = np.eye(2, dtype=np.complex128)
    target_overlap = np.zeros((3, 3, 3, 3), dtype=np.complex128)
    for ik in range(3):
        target_overlap[:, ik, :, ik] = np.eye(3, dtype=np.complex128)
    target_source_overlap = rng.normal(size=(3, 3, 2, 2)) + 1.0j * rng.normal(
        size=(3, 3, 2, 2)
    )

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
    assert hamiltonian.shape == (3, 3, 3)
    assert np.allclose(hamiltonian, -expected_fock)
    assert np.isclose(compute_hf_energy(np.zeros_like(density), np.zeros_like(density), density), 0.0)


def test_projected_target_hamiltonian_rejects_mismatched_shift_inventory() -> None:
    density = np.zeros((1, 1, 1), dtype=np.complex128)
    base = np.zeros((1, 1, 1), dtype=np.complex128)
    overlap = np.ones((1, 1, 1, 1), dtype=np.complex128)
    source = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j]),
        overlaps={(0, 0): overlap},
    )
    target = HFOverlapBlockSet(
        shifts=((1, 0),),
        gvecs=np.asarray([1.0 + 0.0j]),
        overlaps={(1, 0): overlap},
    )
    target_source = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j]),
        overlaps={(0, 0): overlap},
    )
    with pytest.raises(ValueError, match="shift order"):
        build_projected_target_hamiltonian(
            base,
            density,
            source_overlap_blocks=source,
            target_overlap_blocks=target,
            target_source_overlap_blocks=target_source,
            v0=1.0,
            use_numba=False,
        )


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
