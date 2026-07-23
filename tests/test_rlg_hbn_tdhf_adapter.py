from __future__ import annotations

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.core.hf import ParticleHolePair, shift_wavefunction_grid, split_pair_indices_by_flavor_channel
from mean_field.systems.RnG_hBN._tdhf_fixed_quotient import (
    RLGhBNTDHFFixedTermEvaluators,
    build_energy_assigned_c3_sewing,
    c3_composed_direct_physical_shell,
    c3_direct_physical_shell,
    c3_reciprocal_index,
    c3_repeated_zone_offset,
    physical_minus_particle_hole,
    populate_shared_b_partner_entries,
    transport_fixed_terms_from_canonical_form_factors,
)
from mean_field.devtools.run_rlg_hbn_tdhf_q0 import _shortcut_decision
from mean_field.systems.RnG_hBN import (
    RLGhBNFiniteQDensityTangent,
    RLGhBNHFInteractionProvenance,
    RLGhBNHartreeFockRun,
    RLGhBNHartreeFockState,
    RLGhBNInteractionParams,
    RLGhBNLayerOverlapBlockSet,
    RLGhBNModel,
    RLGhBNTDHFInteraction,
    RLGhBNTDHFOrbitals,
    RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING,
    RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
    RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
    RLG_HBN_HF_PHYSICAL_SHIFT_POLICY_VERSION,
    RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION,
    RLG_HBN_REMOTE_H0_POLICY_VERSION,
    apply_rlg_hbn_hf_single_representative_response,
    build_rlg_hbn_hf_c3_quotient_interaction_components,
    build_rlg_hbn_hf_c3_quotient_interaction_context,
    build_rlg_hbn_hf_interaction_hamiltonian,
    build_rlg_hbn_hf_problem,
    build_rlg_hbn_lattice,
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_projected_basis,
    build_rlg_hbn_tdhf_c3_quotient_cycle,
    build_rlg_hbn_tdhf_c3_quotient_orbit,
    build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs,
    build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs,
    build_rlg_hbn_tdhf_finite_q_single_representative_matrix_pair_from_pairs,
    build_rlg_hbn_tdhf_interaction,
    build_rlg_hbn_tdhf_orbitals,
    build_rlg_hbn_tdhf_orbitals_from_canonical_hf,
    build_rlg_hbn_tdhf_q_matrices,
    build_rlg_hbn_tdhf_q_matrices_from_canonical_hf,
    build_rlg_hbn_tdhf_q_pairs,
    build_rlg_hbn_tdhf_q0_matrices,
    build_rlg_hbn_tdhf_q0_matrices_from_canonical_hf,
    build_rlg_hbn_tdhf_q0_matrices_from_pairs,
    build_rlg_hbn_tdhf_q0_pairs,
    center_reciprocal_fractional_coordinates,
    finite_q_shift_cartesian_nm_inv,
    initialize_rlg_hbn_density,
    interaction_shifts_for_cutoff,
    load_or_build_projected_basis,
    load_projected_basis_cache,
    mbz_hexagon_vertices_nm_inv,
    required_rlg_hbn_tdhf_finite_q_overlap_shifts,
    rlg_hbn_flavor_occupation_counts_for_init_mode,
    rlg_hbn_hf_run_to_hf_run_result,
    rlg_hbn_reference_density,
    run_rlg_hbn_hartree_fock,
    rlg_hbn_tdhf_finite_q_mode_support,
    validate_rlg_hbn_hf_single_representative_source_closure,
    validate_rlg_hbn_tdhf_canonical_orbital_parity,
)


def _tiny_flavor_polarized_run(
    *,
    k_mesh_size: int = 1,
    mesh_size: int = 1,
    active_conduction_bands: int = 1,
) -> RLGhBNHartreeFockRun:
    model = RLGhBNModel.from_config(
        layer_count=3,
        xi=1,
        theta_deg=0.77,
        displacement_field_mev=24.0,
        shell_count=1,
    )
    interaction = RLGhBNInteractionParams(
        active_valence_bands=0,
        active_conduction_bands=int(active_conduction_bands),
        k_mesh_size=int(k_mesh_size),
        interaction_cutoff_q1=1.0,
        use_screened_basis=False,
    )
    basis_data = build_rlg_hbn_projected_basis(model, interaction, mesh_size=int(mesh_size))
    overlap_blocks = build_rlg_hbn_layer_overlap_blocks(basis_data, shifts=((0, 0), (-1, 0), (1, 0)))
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



def _typed_single_representative_tiny_run(
    *,
    k_mesh_size: int = 1,
    mesh_size: int = 1,
    active_conduction_bands: int = 1,
) -> RLGhBNHartreeFockRun:
    run = _tiny_flavor_polarized_run(
        k_mesh_size=k_mesh_size,
        mesh_size=mesh_size,
        active_conduction_bands=active_conduction_bands,
    )
    physical_shifts = interaction_shifts_for_cutoff(
        run.basis_data.basis_model.lattice,
        run.basis_data.interaction,
    )
    physical_blocks = build_rlg_hbn_layer_overlap_blocks(
        run.basis_data,
        shifts=physical_shifts,
    )
    cache_blocks = build_rlg_hbn_layer_overlap_blocks(
        run.basis_data,
        shifts=((0, 0), (-1, 0), (1, 0)),
    )
    overlap_blocks = RLGhBNLayerOverlapBlockSet(
        shifts=physical_blocks.shifts,
        gvecs=physical_blocks.gvecs,
        layer_overlaps={
            **physical_blocks.layer_overlaps,
            **cache_blocks.layer_overlaps,
        },
        layer_diagonal_overlaps={
            **physical_blocks.layer_diagonal_overlaps,
            **cache_blocks.layer_diagonal_overlaps,
        },
        hartree_layer_coulomb={
            **physical_blocks.hartree_layer_coulomb,
            **cache_blocks.hartree_layer_coulomb,
        },
        fock_layer_coulomb={
            **physical_blocks.fock_layer_coulomb,
            **cache_blocks.fock_layer_coulomb,
        },
    )
    provenance = RLGhBNHFInteractionProvenance(
        convention=(
            RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION
        ),
        quotient_enabled=False,
        beta=1.0,
        physical_shifts=tuple(physical_shifts),
        zero_literal_q0_fock=False,
        basis_periodic_gauge=RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
        basis_periodic_gauge_padding=RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING,
        form_factor_convention=RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
        remote_h0_policy=RLG_HBN_REMOTE_H0_POLICY_VERSION,
        remote_h0_sha256=hashlib.sha256(
            np.ascontiguousarray(
                run.basis_data.fixed_remote_hamiltonian,
                dtype=np.complex128,
            ).view(np.uint8)
        ).hexdigest(),
        physical_shift_policy=RLG_HBN_HF_PHYSICAL_SHIFT_POLICY_VERSION,
    )
    return replace(
        run,
        overlap_blocks=overlap_blocks,
        interaction_provenance=provenance,
    )


def _canonical_ready_tiny_run(*, k_mesh_size: int = 1, mesh_size: int = 1) -> RLGhBNHartreeFockRun:
    """Tiny RLG/hBN run with nondegenerate flavor blocks for canonical TDHF parity tests."""

    run = _tiny_flavor_polarized_run(k_mesh_size=k_mesh_size, mesh_size=mesh_size)
    nt, nk = run.state.nt, run.state.nk
    hamiltonian = np.zeros((nt, nt, nk), dtype=np.complex128)
    energies = np.zeros((nt, nk), dtype=float)
    for ik in range(nk):
        diagonal = np.arange(nt, dtype=float) + 0.125 * float(ik)
        hamiltonian[:, :, ik] = np.diag(diagonal)
        energies[:, ik] = diagonal
    projector = np.zeros_like(hamiltonian)
    projector[0, 0, :] = 1.0
    run.state.h0[:, :, :] = hamiltonian
    run.state.hamiltonian[:, :, :] = hamiltonian
    run.state.energies[:, :] = energies
    run.state.density[:, :, :] = projector - run.state.reference_density
    run.state.mu = 0.5
    run.state.occupation_counts = (1, 0, 0, 0)
    return run


def test_rlg_hbn_basis_cache_roundtrip_preserves_c3_periodic_metadata(tmp_path) -> None:
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
        k_mesh_size=3,
        interaction_cutoff_q1=1.0,
        use_screened_basis=False,
    )
    cached = load_or_build_projected_basis(
        model,
        interaction,
        cache_dir=tmp_path,
        cache_policy="refresh",
        mesh_size=3,
    )
    expected = cached.value
    assert expected.periodic_reciprocal_shifts is not None
    assert expected.c3_fixed_representative_pairs == ((1, 2), (2, 1))

    loaded = load_projected_basis_cache(tmp_path, cached.key)
    assert loaded.periodic_reciprocal_shifts == expected.periodic_reciprocal_shifts
    assert loaded.c3_fixed_representative_pairs == expected.c3_fixed_representative_pairs

    assert cached.path is not None
    (cached.path / "periodic_reciprocal_shifts.npy").unlink()
    (cached.path / "c3_fixed_representative_pairs.npy").unlink()
    derived = load_projected_basis_cache(tmp_path, cached.key)
    assert derived.periodic_reciprocal_shifts == expected.periodic_reciprocal_shifts
    assert derived.c3_fixed_representative_pairs == expected.c3_fixed_representative_pairs


def test_rlg_hbn_hf_c3_quotient_preserves_flavor_seed_spectra() -> None:
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
        k_mesh_size=3,
        interaction_cutoff_q1=1.0,
        use_screened_basis=False,
    )
    basis = build_rlg_hbn_projected_basis(model, interaction, mesh_size=3)
    blocks = build_rlg_hbn_layer_overlap_blocks(basis)
    context = build_rlg_hbn_hf_c3_quotient_interaction_context(basis, blocks)
    reference = rlg_hbn_reference_density(
        basis.nt,
        basis.nk,
        scheme=interaction.scheme,
        active_valence_bands=interaction.active_valence_bands,
        n_spin=basis.basis.n_spin,
        n_eta=basis.basis.n_flavor,
    )
    density = initialize_rlg_hbn_density(
        basis.h0,
        nu=1.0,
        reference_density=reference,
        active_valence_bands=interaction.active_valence_bands,
        init_mode="flavor",
        seed=1,
        n_spin=basis.basis.n_spin,
        n_eta=basis.basis.n_flavor,
        n_band=basis.n_band,
    )
    components = build_rlg_hbn_hf_c3_quotient_interaction_components(
        density,
        context,
        v0=basis.v0,
    )
    fixed = set(basis.c3_fixed_representative_pairs)
    indices = np.arange(basis.nt, dtype=int).reshape(
        (basis.basis.n_spin, basis.basis.n_flavor, basis.n_band),
        order="F",
    )
    for source_pair in ((0, 1), (1, 0), (1, 1)):
        target_pair = (-source_pair[1] % 3, (source_pair[0] - source_pair[1]) % 3)
        if source_pair in fixed or target_pair in fixed:
            continue
        source_index = source_pair[0] * 3 + source_pair[1]
        target_index = target_pair[0] * 3 + target_pair[1]
        for values in (components.hartree, components.fock, components.total):
            for spin in range(basis.basis.n_spin):
                for flavor in range(basis.basis.n_flavor):
                    block_indices = indices[spin, flavor, :]
                    source = values[:, :, source_index][np.ix_(block_indices, block_indices)]
                    target = values[:, :, target_index][np.ix_(block_indices, block_indices)]
                    assert np.max(
                        np.abs(np.linalg.eigvalsh(source) - np.linalg.eigvalsh(target))
                    ) < 1.0e-9


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


def test_rlg_hbn_tdhf_canonical_orbitals_match_legacy_for_flavor_resolved_state() -> None:
    run = _canonical_ready_tiny_run()
    canonical = rlg_hbn_hf_run_to_hf_run_result(run)

    legacy = build_rlg_hbn_tdhf_orbitals(run.state)
    from_result = build_rlg_hbn_tdhf_orbitals_from_canonical_hf(canonical)
    from_state = build_rlg_hbn_tdhf_orbitals_from_canonical_hf(canonical.final_state)
    metrics = validate_rlg_hbn_tdhf_canonical_orbital_parity(run.state, canonical)

    np.testing.assert_allclose(from_result.energies, legacy.energies, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(from_result.eigenvectors, legacy.eigenvectors, rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(from_result.occupied_mask, legacy.occupied_mask)
    np.testing.assert_allclose(from_state.energies, legacy.energies, rtol=0.0, atol=1.0e-12)
    assert metrics["energy_residual"] <= 1.0e-12
    assert metrics["occupied_mask_mismatches"] == 0.0


def test_rlg_hbn_tdhf_q0_matrices_from_canonical_hf_matches_legacy_path() -> None:
    run = _canonical_ready_tiny_run()
    canonical = rlg_hbn_hf_run_to_hf_run_result(run)

    legacy = build_rlg_hbn_tdhf_q0_matrices(run, max_pairs=8)
    bridged = build_rlg_hbn_tdhf_q0_matrices_from_canonical_hf(run, canonical, max_pairs=8)

    np.testing.assert_allclose(bridged.A, legacy.A, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(bridged.B, legacy.B, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(bridged.L, legacy.L, rtol=1.0e-12, atol=1.0e-12)
    assert bridged.structure.ok


def test_rlg_hbn_tdhf_canonical_adapter_rejects_flavor_mixed_hamiltonian() -> None:
    run = _canonical_ready_tiny_run()
    run.state.hamiltonian[0, 1, 0] = 1.0e-4
    run.state.hamiltonian[1, 0, 0] = 1.0e-4
    canonical = rlg_hbn_hf_run_to_hf_run_result(run)

    with pytest.raises(ValueError, match="block-diagonal"):
        build_rlg_hbn_tdhf_orbitals_from_canonical_hf(canonical, occupation_policy="energy_sort")


def test_rlg_hbn_tdhf_orbitals_reject_occupation_counts_for_flavor_mixed_hamiltonian() -> None:
    run = _tiny_flavor_polarized_run()
    run.state.hamiltonian[0, 1, 0] = 1.0e-4
    run.state.hamiltonian[1, 0, 0] = 1.0e-4

    with pytest.raises(ValueError, match="occupation_counts TDHF orbital shortcut"):
        build_rlg_hbn_tdhf_orbitals(run.state)


def test_rlg_hbn_tdhf_finite_q_pairs_shift_particle_momentum_on_mesh() -> None:
    run = _tiny_flavor_polarized_run(k_mesh_size=2, mesh_size=2)
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    q0_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, (0, 0))
    reference_q0_pairs = build_rlg_hbn_tdhf_q0_pairs(orbitals)
    assert [(p.particle, p.hole) for p in q0_pairs] == [(p.particle, p.hole) for p in reference_q0_pairs]

    q_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, (1, 0))
    assert len(q_pairs) == len(reference_q0_pairs)
    for pair in q_pairs:
        _particle_local, particle_k = orbitals.decode_global_index(pair.particle)
        _hole_local, hole_k = orbitals.decode_global_index(pair.hole)
        # k-grid is row-major on a 2x2 mesh; shift (1,0) flips the row index.
        assert particle_k == (hole_k + 2) % 4


def test_rlg_hbn_finite_q_cartesian_coordinates_are_centered_not_ws_folded() -> None:
    lattice = build_rlg_hbn_lattice(theta_deg=0.77, shell_count=2, layer_count=5)
    mesh_shape = (12, 12)

    np.testing.assert_allclose(
        center_reciprocal_fractional_coordinates(np.asarray([11.0 / 12.0, 0.0])),
        [-1.0 / 12.0, 0.0],
    )
    np.testing.assert_allclose(
        finite_q_shift_cartesian_nm_inv(lattice, (11, 0), mesh_shape),
        -np.asarray([lattice.g_m1.real, lattice.g_m1.imag], dtype=float) / 12.0,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        finite_q_shift_cartesian_nm_inv(lattice, (7, 7), mesh_shape),
        -5.0
        * (
            np.asarray([lattice.g_m1.real, lattice.g_m1.imag], dtype=float)
            + np.asarray([lattice.g_m2.real, lattice.g_m2.imag], dtype=float)
        )
        / 12.0,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        finite_q_shift_cartesian_nm_inv(lattice, (11, 0), mesh_shape, centered=False),
        11.0 * np.asarray([lattice.g_m1.real, lattice.g_m1.imag], dtype=float) / 12.0,
        rtol=0.0,
        atol=1.0e-14,
    )

    hexagon = mbz_hexagon_vertices_nm_inv(lattice)
    assert hexagon.shape == (7, 2)
    np.testing.assert_allclose(hexagon[0], hexagon[-1], rtol=0.0, atol=1.0e-14)


def test_rlg_hbn_tdhf_finite_q_required_overlap_shifts_include_wrapped_particle_keys() -> None:
    run = _tiny_flavor_polarized_run(k_mesh_size=2, mesh_size=2)
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    all_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, (1, 0))
    indices = split_pair_indices_by_flavor_channel(all_pairs)["interspin"]
    pairs = tuple(all_pairs[int(index)] for index in indices)

    required = required_rlg_hbn_tdhf_finite_q_overlap_shifts(
        orbitals,
        run.basis_data,
        pairs,
        (1, 0),
        physical_shifts=((0, 0),),
    )

    assert required == ((-1, 0), (0, 0), (1, 0))


def test_rlg_hbn_tdhf_finite_q_exchange_shortcut_smoke_for_flavor_flip_channel() -> None:
    run = _tiny_flavor_polarized_run(k_mesh_size=2, mesh_size=2)
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    all_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, (1, 0))
    indices = split_pair_indices_by_flavor_channel(all_pairs)["interspin"]
    pairs = tuple(all_pairs[int(index)] for index in indices)
    matrices = build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        (1, 0),
        require_complete_umklapp=False,
    )

    assert matrices.A.shape == (len(pairs), len(pairs))
    assert matrices.B.shape == (len(pairs), len(pairs))
    assert np.allclose(matrices.B, 0.0)
    assert np.all(np.isfinite(matrices.A))
    assert matrices.structure.particle_hole_symmetry <= matrices.structure.tolerance


def test_rlg_hbn_tdhf_finite_q_exchange_shortcut_reduces_to_q0_shortcut() -> None:
    run = _tiny_flavor_polarized_run(k_mesh_size=2, mesh_size=2)
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    q0_pairs = build_rlg_hbn_tdhf_q0_pairs(orbitals)
    indices = split_pair_indices_by_flavor_channel(q0_pairs)["interspin"]
    pairs = tuple(q0_pairs[int(index)] for index in indices)
    q0 = build_rlg_hbn_tdhf_q0_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        include_direct_terms=False,
        include_b_terms=False,
        assembly="vectorized",
    )
    finite_q0 = build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        (0, 0),
        require_complete_umklapp=True,
    )

    np.testing.assert_allclose(finite_q0.A, q0.A, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(finite_q0.B, q0.B, rtol=1e-12, atol=1e-12)


def test_rlg_hbn_tdhf_finite_q_intraflavor_reduces_to_q0_full_ab() -> None:
    run = _tiny_flavor_polarized_run(k_mesh_size=2, mesh_size=2, active_conduction_bands=2)
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    q0_pairs = build_rlg_hbn_tdhf_q0_pairs(orbitals)
    indices = split_pair_indices_by_flavor_channel(q0_pairs)["intraflavor"]
    pairs = tuple(q0_pairs[int(index)] for index in indices)
    assert len(pairs) == 4

    q0 = build_rlg_hbn_tdhf_q0_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        include_direct_terms=True,
        include_exchange_terms=True,
        include_b_terms=True,
        assembly="vectorized",
    )
    finite_q0 = build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        (0, 0),
        require_complete_umklapp=True,
    )

    np.testing.assert_allclose(finite_q0.A, q0.A, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(finite_q0.B, q0.B, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(finite_q0.L, q0.L, rtol=1e-12, atol=1e-12)
    assert finite_q0.structure.ok


def test_rlg_hbn_tdhf_finite_q_intraflavor_full_ab_smoke() -> None:
    run = _tiny_flavor_polarized_run(k_mesh_size=2, mesh_size=2, active_conduction_bands=2)
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    all_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, (1, 0))
    indices = split_pair_indices_by_flavor_channel(all_pairs)["intraflavor"]
    pairs = tuple(all_pairs[int(index)] for index in indices)
    matrices = build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        (1, 0),
        physical_shifts=((0, 0),),
    )

    assert matrices.A.shape == (4, 4)
    assert matrices.B.shape == (4, 4)
    assert not np.allclose(matrices.B, 0.0)
    assert matrices.structure.ok


def test_rlg_hbn_tdhf_finite_q_intraflavor_matches_generic_d12_for_no_wrap_synthetic_blocks() -> None:
    """Check nonzero-q intraflavor vectorization against literal D12 matrix elements.

    q=0 reduction tests cannot catch finite-q pair bookkeeping.  This synthetic
    no-wrap case compares the vectorized Eq. D19 builder to the generic TDHF
    interaction callable for the same D12 A/B terms, without relying on any RLG
    basis construction or Slurm-scale calculation.
    """

    rng = np.random.default_rng(123)
    nt = 2
    nk = 4
    n_layer = 2
    eigenvectors = np.zeros((nt, nt, nk), dtype=np.complex128)
    for ik in range(nk):
        q_matrix, _r = np.linalg.qr(rng.normal(size=(nt, nt)) + 1.0j * rng.normal(size=(nt, nt)))
        eigenvectors[:, :, ik] = q_matrix
    orbitals = RLGhBNTDHFOrbitals(
        energies=np.asarray(
            [
                np.linspace(0.0, 0.3, nk),
                2.0 + np.linspace(0.1, 0.4, nk),
            ],
            dtype=float,
        ),
        eigenvectors=eigenvectors,
        occupied_mask=np.asarray([[True] * nk, [False] * nk], dtype=bool),
        mu=1.0,
        n_spin=1,
        n_eta=1,
        n_band=2,
    )
    layer_overlap = rng.normal(size=(n_layer, nt, nk, nt, nk)) + 1.0j * rng.normal(
        size=(n_layer, nt, nk, nt, nk)
    )
    kernel = rng.random(size=(nk, nk, n_layer, n_layer))
    run = SimpleNamespace(
        basis_data=SimpleNamespace(
            nt=nt,
            nk=nk,
            v0=1.7,
            k_grid_frac=np.asarray([(ik / nk, 0.0) for ik in range(nk)], dtype=float),
        ),
        overlap_blocks=SimpleNamespace(
            shifts=((0, 0),),
            layer_overlaps={(0, 0): layer_overlap},
            fock_layer_coulomb={(0, 0): kernel},
        ),
    )
    q_shift = (1, 0)
    pairs = tuple(
        ParticleHolePair(
            particle=orbitals.global_index(1, hole_k + 1),
            hole=orbitals.global_index(0, hole_k),
            particle_momentum=hole_k + 1,
            hole_momentum=hole_k,
            particle_flavor=orbitals.flavor_tag(1),
            hole_flavor=orbitals.flavor_tag(0),
        )
        for hole_k in (1, 2)
    )
    vectorized = build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        q_shift,
        physical_shifts=((0, 0),),
        _build_partner=False,
    )
    interaction = build_rlg_hbn_tdhf_interaction(run, orbitals)
    expected_A = np.zeros_like(vectorized.A)
    expected_B = np.zeros_like(vectorized.B)
    for row, row_pair in enumerate(pairs):
        p_row_local, p_row_plus_k = orbitals.decode_global_index(row_pair.particle)
        _h_row_local, h_row_k = orbitals.decode_global_index(row_pair.hole)
        p_row_minus = orbitals.global_index(p_row_local, h_row_k - 1)
        expected_A[row, row] = orbitals.global_energies[row_pair.particle] - orbitals.global_energies[row_pair.hole]
        for col, col_pair in enumerate(pairs):
            p_col_local, _p_col_plus_k = orbitals.decode_global_index(col_pair.particle)
            _h_col_local, h_col_k = orbitals.decode_global_index(col_pair.hole)
            p_col_minus = orbitals.global_index(p_col_local, h_col_k - 1)
            expected_A[row, col] += interaction(row_pair.particle, col_pair.hole, row_pair.hole, col_pair.particle)
            expected_A[row, col] -= interaction(row_pair.particle, col_pair.hole, col_pair.particle, row_pair.hole)
            expected_B[row, col] += interaction(row_pair.particle, p_col_minus, row_pair.hole, col_pair.hole)
            expected_B[row, col] -= interaction(row_pair.particle, p_col_minus, col_pair.hole, row_pair.hole)

    np.testing.assert_allclose(vectorized.A, expected_A, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vectorized.B, expected_B, rtol=1e-12, atol=1e-12)


def _manual_layer_hf_form_factor_from_projected_wavefunctions(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    *,
    layer: int,
    stored_shift: tuple[int, int],
    target_local: int,
    target_k: int,
    source_local: int,
    source_k: int,
) -> complex:
    """Direct Eq. 18 layer form factor from saved projected wavefunctions.

    This intentionally bypasses `run.overlap_blocks.layer_overlaps`, so it can
    catch mistakes in finite-q wrap-to-stored-shift bookkeeping.
    """

    basis = run.basis_data.basis
    nx, ny = basis.grid_shape
    n_spin = int(orbitals.n_spin)
    n_eta = int(orbitals.n_eta)
    n_band = int(orbitals.n_band)
    layer_indices = np.asarray([2 * int(layer), 2 * int(layer) + 1], dtype=int)
    matrix = np.zeros((orbitals.nt, orbitals.nt), dtype=np.complex128)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            valley = int(run.basis_data.valleys[ieta])
            raw_m = -valley * int(stored_shift[0])
            raw_n = -valley * int(stored_shift[1])
            target_grid = np.asarray(basis.wavefunctions[:, :, ieta, int(target_k)], dtype=np.complex128).reshape(
                basis.local_basis_size,
                nx,
                ny,
                n_band,
                order="F",
            )
            source_grid = np.asarray(basis.wavefunctions[:, :, ieta, int(source_k)], dtype=np.complex128).reshape(
                basis.local_basis_size,
                nx,
                ny,
                n_band,
                order="F",
            )
            shifted_source = shift_wavefunction_grid(
                source_grid,
                -raw_m,
                -raw_n,
                boundary_mode="zero_fill",
                grid_axes=(1, 2),
            )
            target_layer = target_grid[layer_indices, :, :, :].reshape(layer_indices.size * nx * ny, n_band, order="F")
            source_layer = shifted_source[layer_indices, :, :, :].reshape(
                layer_indices.size * nx * ny,
                n_band,
                order="F",
            )
            band_overlap = target_layer.conjugate().T @ source_layer
            for target_band in range(n_band):
                target_index = ispin + n_spin * ieta + n_spin * n_eta * target_band
                for source_band in range(n_band):
                    source_index = ispin + n_spin * ieta + n_spin * n_eta * source_band
                    matrix[target_index, source_index] = band_overlap[target_band, source_band]
    target_vec = orbitals.eigenvectors[:, int(target_local), int(target_k)]
    source_vec = orbitals.eigenvectors[:, int(source_local), int(source_k)]
    return complex(np.vdot(target_vec, matrix @ source_vec))


def test_rlg_hbn_tdhf_finite_q_intraflavor_wraps_match_projected_wavefunction_eq18() -> None:
    """Check finite-q wrap keys against a direct projected-wavefunction Eq. 18 evaluation."""

    run = _tiny_flavor_polarized_run(k_mesh_size=2, mesh_size=2, active_conduction_bands=2)
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    q_shift = (1, 0)
    all_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, q_shift)
    indices = split_pair_indices_by_flavor_channel(all_pairs)["intraflavor"]
    pairs = tuple(all_pairs[int(index)] for index in indices)
    vectorized = build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        q_shift,
        physical_shifts=((0, 0),),
    )

    mesh_shape = (2, 2)
    n_layer = int(run.basis_data.basis_model.params.layer_count)
    scale = float(run.basis_data.v0) / float(run.basis_data.nk)
    p_local = np.empty(len(pairs), dtype=int)
    h_local = np.empty(len(pairs), dtype=int)
    h_k = np.empty(len(pairs), dtype=int)
    p_plus_k = np.empty(len(pairs), dtype=int)
    p_minus_k = np.empty(len(pairs), dtype=int)
    wrap_plus = np.empty((len(pairs), 2), dtype=int)
    wrap_minus = np.empty((len(pairs), 2), dtype=int)
    for index, pair in enumerate(pairs):
        p_local[index], p_plus_k[index] = orbitals.decode_global_index(pair.particle)
        h_local[index], h_k[index] = orbitals.decode_global_index(pair.hole)
        ix = int(h_k[index]) // mesh_shape[1]
        iy = int(h_k[index]) % mesh_shape[1]
        plus_x = ix + q_shift[0]
        minus_x = ix - q_shift[0]
        p_minus_k[index] = (minus_x % mesh_shape[0]) * mesh_shape[1] + iy
        wrap_plus[index] = ((plus_x - (plus_x % mesh_shape[0])) // mesh_shape[0], 0)
        wrap_minus[index] = ((minus_x - (minus_x % mesh_shape[0])) // mesh_shape[0], 0)

    expected_A = np.diag(
        [orbitals.global_energies[pair.particle] - orbitals.global_energies[pair.hole] for pair in pairs]
    ).astype(np.complex128)
    expected_B = np.zeros_like(vectorized.B)
    g0 = (0, 0)
    for row, row_pair in enumerate(pairs):
        plus_key_row = (int(g0[0] + wrap_plus[row, 0]), int(g0[1] + wrap_plus[row, 1]))
        kernel_direct = run.overlap_blocks.fock_layer_coulomb[plus_key_row][p_plus_k[row], h_k[row]]
        for col, _col_pair in enumerate(pairs):
            plus_key_col = (int(g0[0] + wrap_plus[col, 0]), int(g0[1] + wrap_plus[col, 1]))
            minus_key_col = (int(g0[0] - wrap_minus[col, 0]), int(g0[1] - wrap_minus[col, 1]))
            pp_key = (
                int(g0[0] + wrap_plus[row, 0] - wrap_plus[col, 0]),
                int(g0[1] + wrap_plus[row, 1] - wrap_plus[col, 1]),
            )
            kernel_pp = run.overlap_blocks.fock_layer_coulomb[pp_key][p_plus_k[row], p_plus_k[col]]
            kernel_b_exchange = run.overlap_blocks.fock_layer_coulomb[plus_key_row][p_plus_k[row], h_k[col]]
            for layer_left in range(n_layer):
                row_ph = _manual_layer_hf_form_factor_from_projected_wavefunctions(
                    run,
                    orbitals,
                    layer=layer_left,
                    stored_shift=plus_key_row,
                    target_local=p_local[row],
                    target_k=p_plus_k[row],
                    source_local=h_local[row],
                    source_k=h_k[row],
                )
                row_pp = _manual_layer_hf_form_factor_from_projected_wavefunctions(
                    run,
                    orbitals,
                    layer=layer_left,
                    stored_shift=pp_key,
                    target_local=p_local[row],
                    target_k=p_plus_k[row],
                    source_local=p_local[col],
                    source_k=p_plus_k[col],
                )
                row_ph_exchange = _manual_layer_hf_form_factor_from_projected_wavefunctions(
                    run,
                    orbitals,
                    layer=layer_left,
                    stored_shift=plus_key_row,
                    target_local=p_local[row],
                    target_k=p_plus_k[row],
                    source_local=h_local[col],
                    source_k=h_k[col],
                )
                for layer_right in range(n_layer):
                    col_ph = _manual_layer_hf_form_factor_from_projected_wavefunctions(
                        run,
                        orbitals,
                        layer=layer_right,
                        stored_shift=plus_key_col,
                        target_local=p_local[col],
                        target_k=p_plus_k[col],
                        source_local=h_local[col],
                        source_k=h_k[col],
                    )
                    col_hh = _manual_layer_hf_form_factor_from_projected_wavefunctions(
                        run,
                        orbitals,
                        layer=layer_right,
                        stored_shift=g0,
                        target_local=h_local[row],
                        target_k=h_k[row],
                        source_local=h_local[col],
                        source_k=h_k[col],
                    )
                    col_hp_direct = _manual_layer_hf_form_factor_from_projected_wavefunctions(
                        run,
                        orbitals,
                        layer=layer_right,
                        stored_shift=minus_key_col,
                        target_local=h_local[col],
                        target_k=h_k[col],
                        source_local=p_local[col],
                        source_k=p_minus_k[col],
                    )
                    col_hp_exchange = _manual_layer_hf_form_factor_from_projected_wavefunctions(
                        run,
                        orbitals,
                        layer=layer_right,
                        stored_shift=minus_key_col,
                        target_local=h_local[row],
                        target_k=h_k[row],
                        source_local=p_local[col],
                        source_k=p_minus_k[col],
                    )
                    expected_A[row, col] += scale * kernel_direct[layer_left, layer_right] * row_ph * np.conj(col_ph)
                    expected_A[row, col] -= scale * kernel_pp[layer_left, layer_right] * row_pp * np.conj(col_hh)
                    expected_B[row, col] += scale * kernel_direct[layer_left, layer_right] * row_ph * np.conj(col_hp_direct)
                    expected_B[row, col] -= (
                        scale
                        * kernel_b_exchange[layer_left, layer_right]
                        * row_ph_exchange
                        * np.conj(col_hp_exchange)
                    )

    np.testing.assert_allclose(vectorized.A, expected_A, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(vectorized.B, expected_B, rtol=1e-11, atol=1e-11)


def test_rlg_hbn_tdhf_c3_quotient_orbit_runs_reduced_real_path() -> None:
    run = _tiny_flavor_polarized_run(
        k_mesh_size=2,
        mesh_size=2,
        active_conduction_bands=2,
    )
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    result = build_rlg_hbn_tdhf_c3_quotient_orbit(
        run,
        orbitals,
        (1, 0),
        physical_shifts=((0, 0),),
        periodic_gauge_padding=2,
        structure_tolerance=1.0e-8,
    )
    assert result.source_shift == (1, 0)
    assert result.target_shift == (0, 1)
    assert result.source_matrices.L.shape == result.target_matrices.L.shape
    assert result.source_matrices.L.shape[0] == 2 * len(result.source_matrices.pairs)
    assert result.source_matrices.structure.b_symmetric <= 1.0e-10
    assert result.target_matrices.structure.b_symmetric <= 1.0e-10
    assert result.metadata["provider_mode"] == "preassembly_source_form_factor_transport"
    assert result.metadata["sewing_basis_padding"] == 2
    assert result.metadata["source_q_padding"] == 2


def test_rlg_hbn_tdhf_c3_quotient_cycle_closes_reduced_real_path() -> None:
    run = _tiny_flavor_polarized_run(
        k_mesh_size=2,
        mesh_size=2,
        active_conduction_bands=2,
    )
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    cycle = build_rlg_hbn_tdhf_c3_quotient_cycle(
        run,
        orbitals,
        (1, 0),
        physical_shifts=((0, 0),),
        structure_tolerance=1.0e-8,
        closure_tolerance=1.0e-8,
    )
    assert cycle.shifts == ((1, 0), (0, 1), (1, 1))
    assert set(cycle.matrices) == set(cycle.shifts)
    assert cycle.closure_residuals["max"] <= 1.0e-8


def test_rlg_hbn_tdhf_finite_q_matrices_from_canonical_hf_matches_legacy_shortcut() -> None:
    run = _canonical_ready_tiny_run(k_mesh_size=2, mesh_size=2)
    canonical = rlg_hbn_hf_run_to_hf_run_result(run)
    q_shift = (1, 0)
    physical_shifts = ((0, 0),)

    legacy_orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    all_pairs = build_rlg_hbn_tdhf_q_pairs(legacy_orbitals, run.basis_data, q_shift)
    indices = split_pair_indices_by_flavor_channel(all_pairs)["interspin"]
    legacy_pairs = tuple(all_pairs[int(index)] for index in indices)
    legacy = build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs(
        run,
        legacy_orbitals,
        legacy_pairs,
        q_shift,
        require_complete_umklapp=True,
        physical_shifts=physical_shifts,
    )

    bridged = build_rlg_hbn_tdhf_q_matrices_from_canonical_hf(
        run,
        canonical,
        q_shift,
        channel="interspin",
        max_pairs=8,
        require_complete_umklapp=True,
        physical_shifts=physical_shifts,
    )

    assert [(pair.particle, pair.hole) for pair in bridged.pairs] == [
        (pair.particle, pair.hole) for pair in legacy.pairs
    ]
    np.testing.assert_allclose(bridged.A, legacy.A, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bridged.B, legacy.B, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bridged.L, legacy.L, rtol=1e-12, atol=1e-12)
    assert bridged.structure.ok


def test_rlg_hbn_tdhf_finite_q_support_introspection_documents_canonical_scope() -> None:
    supported = rlg_hbn_tdhf_finite_q_mode_support("interspin", canonical_boundary=True)
    assert supported.supported
    assert supported.supported_terms == ("hf_energy_difference", "finite_q_A_exchange")
    assert supported.blockers == ()
    assert "V_hf" in " ".join(supported.evidence)
    assert "canonical boundary" in " ".join(supported.evidence)
    payload = supported.as_dict()
    assert payload["supported"] is True
    assert "finite_q_A_direct" in payload["unsupported_terms"]

    intraflavor = rlg_hbn_tdhf_finite_q_mode_support("intraflavor", canonical_boundary=True)
    assert intraflavor.supported
    assert "finite_q_B_exchange" in intraflavor.supported_terms
    assert "q/-q" in intraflavor.reason
    assert intraflavor.unsupported_terms == ()


def test_rlg_hbn_tdhf_finite_q_canonical_bridge_rejects_flavor_mixed_hamiltonian() -> None:
    run = _canonical_ready_tiny_run(k_mesh_size=2, mesh_size=2)
    run.state.hamiltonian[0, 1, 0] = 1.0e-4
    run.state.hamiltonian[1, 0, 0] = 1.0e-4
    canonical = rlg_hbn_hf_run_to_hf_run_result(run)

    with pytest.raises(ValueError, match="block-diagonal"):
        build_rlg_hbn_tdhf_q_matrices_from_canonical_hf(
            run,
            canonical,
            (1, 0),
            channel="interspin",
            max_pairs=8,
            physical_shifts=((0, 0),),
        )


def test_rlg_hbn_tdhf_q_matrices_reports_precise_blockers_for_unsupported_finite_q_modes() -> None:
    run = _tiny_flavor_polarized_run(k_mesh_size=2, mesh_size=2)
    with pytest.raises(NotImplementedError, match="shortcut_exchange_only=False requests full finite-q direct/B"):
        build_rlg_hbn_tdhf_q_matrices_from_canonical_hf(
            run,
            object(),  # guard must fire before canonical orbital normalization
            (1, 0),
            channel="interspin",
            shortcut_exchange_only=False,
        )
    with pytest.raises(ValueError, match="unknown finite-q channel"):
        build_rlg_hbn_tdhf_q_matrices(run, (1, 0), channel="bogus")  # type: ignore[arg-type]


def test_rlg_hbn_single_representative_active_functional_is_pairing_self_adjoint() -> None:
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
        k_mesh_size=2,
        interaction_cutoff_q1=2.0,
        use_screened_basis=False,
    )
    basis_data = build_rlg_hbn_projected_basis(model, interaction, mesh_size=2)
    blocks = build_rlg_hbn_layer_overlap_blocks(basis_data)
    rng = np.random.default_rng(20260722)

    def hermitian_density() -> np.ndarray:
        raw = rng.normal(size=basis_data.h0.shape) + 1.0j * rng.normal(
            size=basis_data.h0.shape
        )
        return 0.5 * (raw + raw.swapaxes(0, 1).conj())

    left = hermitian_density()
    right = hermitian_density()
    k_left = build_rlg_hbn_hf_interaction_hamiltonian(
        left, blocks, v0=basis_data.v0
    )
    k_right = build_rlg_hbn_hf_interaction_hamiltonian(
        right, blocks, v0=basis_data.v0
    )
    pairing_left = np.einsum("abk,abk->", k_left, right, optimize=True)
    pairing_right = np.einsum("abk,abk->", left, k_right, optimize=True)
    np.testing.assert_allclose(
        pairing_left, pairing_right, rtol=1.0e-11, atol=1.0e-11
    )

    epsilon = 1.0e-6

    def interaction_energy(density: np.ndarray) -> float:
        response = build_rlg_hbn_hf_interaction_hamiltonian(
            density, blocks, v0=basis_data.v0
        )
        return float(
            0.5
            * np.einsum("abk,abk->", response, density, optimize=True).real
            / basis_data.nk
        )

    finite_difference = (
        interaction_energy(left + epsilon * right)
        - interaction_energy(left - epsilon * right)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        finite_difference,
        pairing_left.real / basis_data.nk,
        rtol=2.0e-8,
        atol=2.0e-8,
    )


def test_rlg_hbn_single_representative_runner_selection_is_explicit_and_typed() -> None:
    run = _typed_single_representative_tiny_run()
    fresh = run_rlg_hbn_hartree_fock(
        run.basis_data,
        overlap_blocks=RLGhBNLayerOverlapBlockSet(
            shifts=run.overlap_blocks.shifts,
            gvecs=run.overlap_blocks.gvecs,
            layer_overlaps={
                shift: run.overlap_blocks.layer_overlaps[shift]
                for shift in run.overlap_blocks.shifts
            },
            layer_diagonal_overlaps={
                shift: run.overlap_blocks.layer_diagonal_overlaps[shift]
                for shift in run.overlap_blocks.shifts
            },
            hartree_layer_coulomb={
                shift: run.overlap_blocks.hartree_layer_coulomb[shift]
                for shift in run.overlap_blocks.shifts
            },
            fock_layer_coulomb={
                shift: run.overlap_blocks.fock_layer_coulomb[shift]
                for shift in run.overlap_blocks.shifts
            },
        ),
        nu=1.0,
        init_mode="flavor",
        seed=1,
        max_iter=1,
        precision=1.0e-12,
        c3_quotient_interaction=False,
    )
    assert fresh.interaction_provenance is not None
    assert (
        fresh.interaction_provenance.convention
        == RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION
    )
    assert not fresh.interaction_provenance.quotient_enabled
    assert fresh.interaction_provenance.remote_h0_policy == RLG_HBN_REMOTE_H0_POLICY_VERSION
    assert len(fresh.interaction_provenance.remote_h0_sha256) == 64
    assert (
        fresh.interaction_provenance.physical_shift_policy
        == RLG_HBN_HF_PHYSICAL_SHIFT_POLICY_VERSION
    )


def test_rlg_hbn_hf_single_representative_source_closure_is_fail_closed() -> None:
    run = _typed_single_representative_tiny_run()
    run.state.hamiltonian[:, :, :] = (
        run.state.h0
        + build_rlg_hbn_hf_interaction_hamiltonian(
            run.state.density,
            run.overlap_blocks,
            v0=run.state.v0,
        )
    )
    converged = replace(run, converged=True)
    metrics = validate_rlg_hbn_hf_single_representative_source_closure(
        converged,
        closure_tolerance_mev=1.0e-12,
        stationarity_tolerance_mev=1.0e6,
    )
    assert metrics["hamiltonian_closure_mev"] < 1.0e-12
    assert metrics["h0_basis_residual_mev"] == 0.0

    assert converged.interaction_provenance is not None
    bad_provenance = replace(
        converged.interaction_provenance,
        remote_h0_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="remote_h0_sha256"):
        validate_rlg_hbn_hf_single_representative_source_closure(
            replace(converged, interaction_provenance=bad_provenance),
            stationarity_tolerance_mev=1.0e6,
        )


def test_rlg_hbn_hf_single_representative_q0_response_uses_source_kernel() -> None:
    run = _typed_single_representative_tiny_run()
    rng = np.random.default_rng(17)
    tangent_blocks = rng.normal(size=run.state.density.shape) + 1.0j * rng.normal(
        size=run.state.density.shape
    )
    tangent = RLGhBNFiniteQDensityTangent(
        q_shift=(0, 0),
        target_k=np.arange(run.state.nk, dtype=int),
        source_k=np.arange(run.state.nk, dtype=int),
        blocks=tangent_blocks,
        role="ph",
    )
    response = apply_rlg_hbn_hf_single_representative_response(
        run,
        tangent,
        require_converged=False,
        require_provenance=True,
    )
    expected = build_rlg_hbn_hf_interaction_hamiltonian(
        tangent_blocks,
        run.overlap_blocks,
        v0=run.state.v0,
    )
    np.testing.assert_allclose(response.total, expected, rtol=1.0e-12, atol=1.0e-12)
    assert response.provenance["source_provenance_validated"]


def test_rlg_hbn_tdhf_single_representative_signed_q0_api_is_typed() -> None:
    run = _typed_single_representative_tiny_run()
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    all_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, (0, 0))
    groups = split_pair_indices_by_flavor_channel(all_pairs)
    pairs = tuple(all_pairs[int(index)] for index in groups["intervalley"])

    result = build_rlg_hbn_tdhf_finite_q_single_representative_matrix_pair_from_pairs(
        run,
        orbitals,
        pairs,
        (0, 0),
        channel="intervalley",
        require_provenance=True,
    )

    assert result.q_shift == (0, 0)
    assert result.minus_q_shift == (0, 0)
    assert result.channel == "intervalley"
    np.testing.assert_allclose(result.plus.A, result.minus.A)
    np.testing.assert_allclose(result.plus.B, result.minus.B)
    np.testing.assert_allclose(result.plus.L, result.minus.L)
    assert result.plus.structure.ok


def test_rlg_hbn_tdhf_single_representative_signed_nonzero_q_uses_independent_partner() -> None:
    run = _typed_single_representative_tiny_run(
        k_mesh_size=3,
        mesh_size=3,
        active_conduction_bands=2,
    )
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    all_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, (1, 0))
    groups = split_pair_indices_by_flavor_channel(all_pairs)
    pairs = tuple(all_pairs[int(index)] for index in groups["intraflavor"])

    result = build_rlg_hbn_tdhf_finite_q_single_representative_matrix_pair_from_pairs(
        run,
        orbitals,
        pairs,
        (1, 0),
        channel="intraflavor",
        require_provenance=True,
    )
    n_pairs = len(pairs)
    np.testing.assert_allclose(
        result.plus.L[:n_pairs, :n_pairs], result.plus.A
    )
    np.testing.assert_allclose(
        result.plus.L[:n_pairs, n_pairs:], result.plus.B
    )
    np.testing.assert_allclose(
        result.plus.L[n_pairs:, :n_pairs], -result.minus.B.conj()
    )
    np.testing.assert_allclose(
        result.plus.L[n_pairs:, n_pairs:], -result.minus.A.conj()
    )
    np.testing.assert_allclose(
        result.plus.B, result.minus.B.T, rtol=1.0e-12, atol=1.0e-12
    )
    assert result.plus.structure.ok
    assert result.plus.pairs[0].particle != result.minus.pairs[0].particle

    with pytest.raises(ValueError, match="converged"):
        build_rlg_hbn_tdhf_q_matrices(
            run,
            (1, 0),
            channel="intraflavor",
        )
    dispatched = build_rlg_hbn_tdhf_q_matrices(
        run,
        (1, 0),
        channel="intraflavor",
        require_single_representative_source_closure=False,
    )
    np.testing.assert_allclose(dispatched.A, result.plus.A)
    np.testing.assert_allclose(dispatched.B, result.plus.B)
    np.testing.assert_allclose(dispatched.L, result.plus.L)


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


def test_rlg_hbn_tdhf_c3_repeated_zone_direct_shell_convention() -> None:
    assert c3_reciprocal_index((3, 7)) == (-7, -4)
    assert c3_repeated_zone_offset((3, 7), (5, 8), 12) == (1, 1)
    assert c3_repeated_zone_offset((-3, -7), (-5, -8), 12) == (-1, -1)
    assert c3_repeated_zone_offset((1, 0), (0, 1), 12) == (0, 0)
    shell = ((0, 0), (1, -1), (-2, 1))
    assert c3_direct_physical_shell(shell, repeated_zone_offset=(1, 1)) == (
        (-1, -1),
        (0, 1),
        (-2, -4),
    )
    assert c3_composed_direct_physical_shell(
        shell,
        repeated_zone_offsets=((1, 1), (1, 1)),
    ) == (
        (0, -1),
        (-2, -2),
        (3, 1),
    )
    c3_invariant_shell = (
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, -1),
        (-1, 0),
        (0, -1),
        (1, 1),
    )
    composed = c3_composed_direct_physical_shell(
        c3_invariant_shell,
        repeated_zone_offsets=((1, 1), (1, 1)),
    )
    assert set(composed) == {(m, n - 1) for m, n in c3_invariant_shell}
    with pytest.raises(ValueError, match="not a repeated-zone representative"):
        c3_repeated_zone_offset((3, 7), (5, 7), 12)


def test_rlg_hbn_tdhf_minus_operator_order_has_physical_particle_as_ket() -> None:
    particle, hole = physical_minus_particle_hole(bra="hole", ket="particle")
    assert particle == "particle"
    assert hole == "hole"


def test_rlg_hbn_tdhf_fixed_form_factor_transport_matches_component_identity() -> None:
    rng = np.random.default_rng(1942)
    n = 5
    left = np.zeros((n, n), dtype=np.complex128)
    right = np.zeros((n, n), dtype=np.complex128)
    left[np.arange(n), np.asarray([2, 4, 1, 0, 3])] = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    right[np.arange(n), np.asarray([1, 3, 4, 2, 0])] = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    canonical = {
        name: rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        for name in ("A_direct", "A_exchange", "B_direct", "B_exchange")
    }
    sentinel = 17.0 - 3.0j
    base = {name: np.full((n, n), sentinel, dtype=np.complex128) for name in canonical}
    x_fixed = np.asarray([False, True, False, False, True])
    y_fixed = np.asarray([True, False, False, True, False])
    evaluators = RLGhBNTDHFFixedTermEvaluators(
        a_direct=lambda i, j: canonical["A_direct"][i, j],
        a_exchange=lambda i, j: canonical["A_exchange"][i, j],
        b_direct=lambda i, j: canonical["B_direct"][i, j],
        b_exchange=lambda i, j: canonical["B_exchange"][i, j],
    )
    result = transport_fixed_terms_from_canonical_form_factors(
        base,
        left_transform=left,
        right_transform=right,
        target_x_fixed=x_fixed,
        target_y_fixed=y_fixed,
        evaluators=evaluators,
    )
    expected = {
        "A_direct": left @ canonical["A_direct"] @ left.conj().T,
        "A_exchange": left @ canonical["A_exchange"] @ left.conj().T,
        "B_direct": left @ canonical["B_direct"] @ right.T,
        "B_exchange": left @ canonical["B_exchange"] @ right.T,
    }
    touched_a = x_fixed[:, None] | x_fixed[None, :]
    touched_b = x_fixed[:, None] | y_fixed[None, :]
    for name in ("A_direct", "A_exchange"):
        assert np.allclose(result.terms[name][touched_a], expected[name][touched_a], atol=1.0e-13)
        assert np.all(result.terms[name][~touched_a] == sentinel)
    for name in ("B_direct", "B_exchange"):
        assert np.allclose(result.terms[name][touched_b], expected[name][touched_b], atol=1.0e-13)
        assert np.all(result.terms[name][~touched_b] == sentinel)
    assert result.touched_a_entries == int(np.count_nonzero(touched_a))
    assert result.touched_b_entries == int(np.count_nonzero(touched_b))
    assert result.max_left_support == result.max_right_support == 1


def test_rlg_hbn_tdhf_energy_sewing_assigns_fixed_labels_and_preserves_raw_phases() -> None:
    raw = np.zeros((4, 4), dtype=np.complex128)
    raw[1, 0] = np.exp(0.2j)
    raw[3, 2] = np.exp(-0.4j)
    raw[0, 3] = 2.0j
    raw[2, 1] = -3.0 + 0.0j
    source_fixed = np.asarray([False, True, False, True])
    target_fixed = np.asarray([True, False, True, False])
    sewing = build_energy_assigned_c3_sewing(
        raw,
        source_fixed=source_fixed,
        target_fixed=target_fixed,
        source_energies=np.asarray([10.0, 1.0, 20.0, 2.0]),
        target_energies=np.asarray([2.0, 10.0, 1.0, 20.0]),
    )
    expected = np.zeros((4, 4), dtype=np.complex128)
    expected[1, 0] = np.exp(0.2j)
    expected[3, 2] = np.exp(-0.4j)
    expected[0, 3] = 1.0j
    expected[2, 1] = -1.0 + 0.0j
    np.testing.assert_allclose(sewing.matrix, expected, rtol=0.0, atol=1.0e-15)
    assert sewing.assignment_max_energy_delta == 0.0
    assert sewing.unitarity_residual_max <= 1.0e-15


def test_rlg_hbn_tdhf_shared_b_provider_populates_partner_once() -> None:
    n = 4
    q_direct = np.zeros((n, n), dtype=np.complex128)
    q_exchange = np.zeros_like(q_direct)
    minus_direct = np.zeros_like(q_direct)
    minus_exchange = np.zeros_like(q_direct)
    x_fixed = np.asarray([True, False, False, False])
    y_fixed = np.asarray([False, False, True, False])
    direct_calls: list[tuple[int, int]] = []
    exchange_calls: list[tuple[int, int]] = []

    def direct(i: int, j: int) -> complex:
        direct_calls.append((i, j))
        return complex(10 * i + j, i - j)

    def exchange(i: int, j: int) -> complex:
        exchange_calls.append((i, j))
        return complex(-i, 2 * j)

    count = populate_shared_b_partner_entries(
        q_direct,
        q_exchange,
        minus_direct,
        minus_exchange,
        q_x_fixed=x_fixed,
        q_y_fixed=y_fixed,
        direct_evaluator=direct,
        exchange_evaluator=exchange,
    )
    touched = x_fixed[:, None] | y_fixed[None, :]
    assert count == int(np.count_nonzero(touched))
    assert len(direct_calls) == len(exchange_calls) == count
    assert np.array_equal(q_direct[touched], minus_direct.T[touched])
    assert np.array_equal(q_exchange[touched], minus_exchange.T[touched])
