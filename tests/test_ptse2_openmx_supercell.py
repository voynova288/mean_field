from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.core.hf.overlap import HFOverlapBlockSet
from mean_field.core.hf.diis import PulayDIISOptions
from mean_field.core.hf.interaction import (
    build_projected_interaction_hamiltonian,
    compute_hf_energy,
)
from mean_field.systems.ptse2_openmx.hf import (
    BOHR_INV_TO_NM_INV,
    PtSe2HFConfig,
    PtSe2HFOverlapCache,
    PtSe2ProjectedHFData,
)
from mean_field.core.supercell import IntegerSupercell
from mean_field.systems.ptse2_openmx.supercell import (
    PtSe2SupercellOverlapCache,
    build_ptse2_2x2_literal_physical_q_interaction,
    build_ptse2_fold_map,
    build_ptse2_literal_physical_q_interaction,
    build_ptse2_supercell_hf_data,
    build_ptse2_supercell_zero_temperature_hessian,
    build_ptse2_supercell_overlap_cache,
    embed_ptse2_translation_invariant_density,
    fold_ptse2_primitive_operator,
    initialize_ptse2_2x2_density,
    initialize_ptse2_supercell_density,
    ptse2_2x1_supercell,
    ptse2_2x2_supercell,
    ptse2_sqrt3_supercell,
    ptse2_supercell_hole_fourier,
    ptse2_supercell_label_breaks_translation,
    ptse2_supercell_label_sector,
    ptse2_translation_observables,
    reconstruct_ptse2_supercell_hole_density,
    ptse2_level_shifted_hamiltonian,
    ptse2_stored_density_commutator,
    ptse2_supercell_energy_from_physical_projector,
    ptse2_translation_breaking_orbital_action,
    run_ptse2_supercell_diis_hf,
    run_ptse2_supercell_operator_diis_hf,
    unfold_ptse2_translation_invariant_operator,
)
from mean_field.systems.ptse2_openmx.transfers import (
    build_full_half_open_physical_q_chart,
)


def test_ptse2_2x2_fold_map_is_bijective_and_roundtrips() -> None:
    fold_map = build_ptse2_fold_map(6, 2, ptse2_2x2_supercell())
    assert fold_map.reduced_mesh == 3
    assert fold_map.reduced_nk == 9
    assert fold_map.folded_rank == 8
    np.testing.assert_array_equal(
        np.sort(fold_map.primitive_indices.ravel()), np.arange(36)
    )
    np.testing.assert_array_equal(
        fold_map.fold_representatives,
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    )
    np.testing.assert_array_equal(
        fold_map.primitive_indices,
        (
            (0, 3, 18, 21),
            (1, 4, 19, 22),
            (2, 5, 20, 23),
            (6, 9, 24, 27),
            (7, 10, 25, 28),
            (8, 11, 26, 29),
            (12, 15, 30, 33),
            (13, 16, 31, 34),
            (14, 17, 32, 35),
        ),
    )
    rng = np.random.default_rng(3)
    primitive = np.empty((2, 2, 36), dtype=np.complex128)
    for index in range(36):
        raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        primitive[:, :, index] = 0.5 * (raw + raw.conj().T)
    folded = fold_ptse2_primitive_operator(primitive, fold_map)
    for reduced_index in range(fold_map.reduced_nk):
        for fold_index in range(4):
            primitive_index = int(fold_map.primitive_indices[reduced_index, fold_index])
            for active_index in range(2):
                folded_index = 4 * active_index + fold_index
                assert folded[folded_index, folded_index, reduced_index] == primitive[
                    active_index, active_index, primitive_index
                ]
    restored = unfold_ptse2_translation_invariant_operator(folded, fold_map)
    np.testing.assert_allclose(restored, primitive, rtol=0.0, atol=0.0)
    assert ptse2_translation_observables(
        folded, fold_map
    ).maximum_fold_offdiagonal_abs == 0.0


def test_generic_fold_maps_cover_2x1_and_sqrt3_cells() -> None:
    stripe = build_ptse2_fold_map(12, 2, ptse2_2x1_supercell())
    assert stripe.area_ratio == 2
    assert stripe.reduced_nk == 72
    assert stripe.folded_rank == 4
    np.testing.assert_array_equal(stripe.fold_shifts, ((0, 0), (6, 0)))
    np.testing.assert_array_equal(stripe.fold_representatives, ((0, 0), (1, 0)))
    assert np.all(stripe.reduced_supercell_numerators[:, 0] % 2 == 0)
    assert ptse2_supercell_label_sector((1, 0), stripe) == (1, 0)
    assert ptse2_supercell_label_breaks_translation((1, 0), stripe)
    assert not ptse2_supercell_label_breaks_translation((0, 1), stripe)

    sqrt3 = build_ptse2_fold_map(12, 2, ptse2_sqrt3_supercell())
    assert sqrt3.area_ratio == 3
    assert sqrt3.reduced_nk == 48
    assert sqrt3.folded_rank == 6
    np.testing.assert_array_equal(
        sqrt3.fold_shifts, ((0, 0), (4, 4), (8, 8))
    )
    np.testing.assert_array_equal(
        sqrt3.fold_representatives, ((0, 0), (0, 1), (0, 2))
    )
    assert np.all(
        (sqrt3.reduced_supercell_numerators[:, 0]
         - sqrt3.reduced_supercell_numerators[:, 1])
        % 3
        == 0
    )
    assert ptse2_supercell_label_breaks_translation((1, 0), sqrt3)
    assert not ptse2_supercell_label_breaks_translation((1, 1), sqrt3)


def test_generic_fold_map_rejects_smith_incompatible_mesh() -> None:
    with pytest.raises(ValueError, match="Smith invariants"):
        build_ptse2_fold_map(4, 1, ptse2_sqrt3_supercell())
    with pytest.raises(ValueError, match="positive"):
        build_ptse2_fold_map(4, 1, IntegerSupercell(1, 0, 0, -1))


def test_ptse2_translation_observables_resolve_mod2_sector() -> None:
    fold_map = build_ptse2_fold_map(4, 1, ptse2_2x2_supercell())
    density = np.zeros((4, 4, 4), dtype=np.complex128)
    density[0, 2, :] = 0.25
    density[2, 0, :] = 0.25
    observables = ptse2_translation_observables(density, fold_map)
    assert observables.maximum_fold_offdiagonal_abs == 0.25
    assert observables.sector_frobenius[(1, 0)] > 0.0
    assert observables.sector_frobenius[(0, 1)] == 0.0
    assert observables.sector_frobenius[(1, 1)] == 0.0


def test_ptse2_supercell_hole_fourier_zero_mode_and_reconstruction() -> None:
    fold_map = build_ptse2_fold_map(4, 1, ptse2_2x2_supercell())
    nt = fold_map.folded_rank
    nk = fold_map.reduced_nk
    identity_diagonal = np.repeat(np.eye(nt)[:, :, None], nk, axis=2)
    zero_overlap = np.zeros((nt, nk, nt, nk), dtype=np.complex128)
    blocks = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0j]),
        overlaps={(0, 0): zero_overlap},
        diagonal_overlaps={(0, 0): identity_diagonal},
        hartree_screening={(0, 0): 1.0},
    )
    cache = PtSe2SupercellOverlapCache(
        blocks=blocks,
        fold_map=fold_map,
        pair_masks={(0, 0): np.eye(nk, dtype=bool)},
        channel_q_indices={(0, 0): np.zeros((nk, nk), dtype=np.int64)},
        assignment_counts={(0, 0): np.eye(nk, dtype=np.int64) * 4},
        source_cache="synthetic",
        source_summary="synthetic",
        cache_sha256="synthetic",
        run_id="synthetic",
        hf_authorized=False,
    )
    density = np.repeat((-np.eye(nt))[:, :, None], nk, axis=2)
    coefficients = ptse2_supercell_hole_fourier(density, cache, use_numba=False)
    assert coefficients == {(0, 0): 4.0 + 0.0j}
    _, _, hole = reconstruct_ptse2_supercell_hole_density(
        coefficients, supercell_area_nm2=8.0, grid_size=16
    )
    np.testing.assert_allclose(hole, 0.5, rtol=0.0, atol=1.0e-15)
    _, _, breaking = reconstruct_ptse2_supercell_hole_density(
        coefficients,
        supercell_area_nm2=8.0,
        grid_size=16,
        translation_breaking_only=True,
        fold_map=fold_map,
    )
    np.testing.assert_array_equal(breaking, np.zeros_like(breaking))


def test_embedding_preserves_supercell_filling_trace() -> None:
    fold_map = build_ptse2_fold_map(4, 2, ptse2_2x2_supercell())
    primitive = np.zeros((2, 2, 16), dtype=np.complex128)
    primitive[1, 1, :] = -1.0
    embedded = embed_ptse2_translation_invariant_density(primitive, fold_map)
    measured = np.trace(embedded, axis1=0, axis2=1).real.mean()
    assert measured == -4.0


def _synthetic_primitive_data(
    mesh: int = 4,
    active_rank: int = 1,
) -> PtSe2ProjectedHFData:
    source = SimpleNamespace(
        reciprocal_bohr=np.asarray(
            ((1.2, 0.3, 0.0), (-0.2, 0.9, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        input_sha256="synthetic-source",
    )
    chart = build_full_half_open_physical_q_chart(
        source,
        mesh,
        c3_rotation_fractional=np.eye(3, dtype=np.int64),
    )
    nk = mesh * mesh
    shifts = tuple((int(row[0]), int(row[1])) for row in chart.local_fields)
    overlaps = {}
    diagonal = {}
    hartree = {}
    fock = {}
    masks = {}
    q_tables = {}
    w_by_q = 1.0 + 0.125 * np.arange(chart.physical_q_numerators.shape[0])
    for local_index, shift in enumerate(shifts):
        mask = np.asarray(chart.admitted_mask[local_index], dtype=bool)
        q_table = np.asarray(chart.channel_q_indices[local_index], dtype=np.int64)
        block = np.zeros(
            (active_rank, nk, active_rank, nk), dtype=np.complex128
        )
        for target, source_index in np.argwhere(mask):
            q_index = int(q_table[target, source_index])
            scalar = (
                1.0
                + 0.01 * q_index
                + 0.001j * (int(target) - int(source_index))
            )
            active_matrix = np.eye(active_rank, dtype=np.complex128)
            if active_rank > 1:
                active_matrix[0, 1] = 0.2
                active_matrix[1, 0] = 0.2
            block[:, target, :, source_index] = scalar * active_matrix
        kernel = np.zeros((nk, nk), dtype=np.float64)
        kernel[mask] = w_by_q[q_table[mask]]
        overlaps[shift] = block
        diagonal[shift] = np.diagonal(block, axis1=1, axis2=3)
        fock[shift] = kernel
        masks[shift] = mask
        q_tables[shift] = q_table
        diagonal_mask = np.diag(mask)
        if np.all(diagonal_mask):
            diagonal_q = np.diag(q_table)
            assert np.unique(diagonal_q).size == 1
            hartree[shift] = float(w_by_q[int(diagonal_q[0])])
    primitive_blocks = HFOverlapBlockSet(
        shifts=shifts,
        gvecs=np.asarray(
            [complex(row[0], row[1]) for row in chart.local_fields],
            dtype=np.complex128,
        ),
        overlaps=overlaps,
        diagonal_overlaps=diagonal,
        hartree_screening=hartree,
        fock_screening=fock,
    )
    primitive_cache = PtSe2HFOverlapCache(
        blocks=primitive_blocks,
        pair_masks=masks,
        channel_q_indices=q_tables,
        folding_carries=chart.folding_carries,
        reduced_steps=chart.reduced_steps,
        physical_q_numerators=chart.physical_q_numerators,
        physical_q_bohr_inv=chart.physical_q_bohr_inv,
        source_cache="synthetic-cache",
        source_summary="synthetic-summary",
        cache_sha256="synthetic-sha",
        run_id="synthetic-run",
        active_rank=active_rank,
        input_cache_schema="synthetic/v1",
        eigensystem_manifest_sha256="synthetic-manifest",
        hf_authorized=False,
    )
    config = PtSe2HFConfig(
        filling_nu=-1,
        dielectric_constant=10.0,
        active_rank=active_rank,
        occupied_below_ev=0.0,
        require_hf_authorized=False,
    )
    h0 = np.zeros((active_rank, active_rank, nk), dtype=np.complex128)
    base = np.linspace(-0.2, 0.1, nk)
    for band in range(active_rank):
        h0[band, band, :] = base + 0.4 * band
    initial_density = np.zeros_like(h0)
    if active_rank == 1:
        initial_density[0, 0, :] = -1.0
    else:
        for band in range(1, active_rank):
            initial_density[band, band, :] = -1.0
    return PtSe2ProjectedHFData(
        config=config,
        source=source,
        k_fractional=chart.k_fractional,
        active_energies_ev=np.real(
            np.diagonal(h0, axis1=0, axis2=1).T
        ),
        h0=h0,
        initial_density=initial_density,
        overlap_cache=primitive_cache,
        eigensystem_root="synthetic",
        eigensystem_manifest_sha256="synthetic-manifest",
        edge_summary_sha256="synthetic-edge",
        area_nm2=2.0,
    )


def test_ptse2_2x2_cache_remap_and_primitive_action_parity() -> None:
    primitive_data = _synthetic_primitive_data()
    super_cache = build_ptse2_supercell_overlap_cache(primitive_data, ptse2_2x2_supercell())
    assert sum(int(np.sum(values)) for values in super_cache.assignment_counts.values()) == sum(
        int(np.count_nonzero(mask)) for mask in primitive_data.overlap_cache.pair_masks.values()
    )
    assert all(
        np.all(values[mask] == 4)
        for label, values in super_cache.assignment_counts.items()
        for mask in (super_cache.pair_masks[label],)
    )
    rng = np.random.default_rng(11)
    primitive_density = rng.uniform(-0.8, -0.1, size=(1, 1, 16)).astype(
        np.complex128
    )
    super_density = embed_ptse2_translation_invariant_density(
        primitive_density, super_cache.fold_map
    )
    primitive_action = build_projected_interaction_hamiltonian(
        primitive_density,
        primitive_data.overlap_cache.blocks,
        v0=1.0 / primitive_data.area_nm2,
        use_numba=False,
    )
    super_action = build_projected_interaction_hamiltonian(
        super_density,
        super_cache.blocks,
        v0=1.0 / (4.0 * primitive_data.area_nm2),
        use_numba=False,
    )
    assert ptse2_translation_observables(
        super_action, super_cache.fold_map
    ).maximum_fold_offdiagonal_abs < 1.0e-12
    unfolded_action = unfold_ptse2_translation_invariant_operator(
        super_action, super_cache.fold_map
    )
    np.testing.assert_allclose(unfolded_action, primitive_action, rtol=1.0e-12, atol=1.0e-12)
    super_h0 = fold_ptse2_primitive_operator(primitive_data.h0, super_cache.fold_map)
    primitive_energy = compute_hf_energy(
        primitive_action, primitive_data.h0, primitive_density
    )
    super_energy = compute_hf_energy(super_action, super_h0, super_density)
    np.testing.assert_allclose(super_energy / 4.0, primitive_energy, rtol=1.0e-12, atol=1.0e-12)

    broken_density = np.array(super_density, copy=True)
    broken_density[0, 1, :] += 0.07 + 0.03j
    broken_density[1, 0, :] += 0.07 - 0.03j
    regrouped_broken_action = build_projected_interaction_hamiltonian(
        broken_density,
        super_cache.blocks,
        v0=1.0 / (4.0 * primitive_data.area_nm2),
        use_numba=False,
    )
    literal_broken_action = build_ptse2_2x2_literal_physical_q_interaction(
        broken_density, primitive_data, super_cache.fold_map
    )
    np.testing.assert_allclose(
        regrouped_broken_action,
        literal_broken_action,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("supercell", "area_ratio"),
    ((ptse2_2x1_supercell(), 2), (ptse2_sqrt3_supercell(), 3)),
)
def test_generic_supercell_cache_action_energy_and_literal_parity(
    supercell: IntegerSupercell,
    area_ratio: int,
) -> None:
    primitive_data = _synthetic_primitive_data(mesh=6)
    super_cache = build_ptse2_supercell_overlap_cache(
        primitive_data, supercell
    )
    assert super_cache.fold_map.area_ratio == area_ratio
    assert sum(
        int(np.sum(values))
        for values in super_cache.assignment_counts.values()
    ) == sum(
        int(np.count_nonzero(mask))
        for mask in primitive_data.overlap_cache.pair_masks.values()
    )
    assert all(
        np.all(values[mask] == area_ratio)
        for label, values in super_cache.assignment_counts.items()
        for mask in (super_cache.pair_masks[label],)
    )

    matrix = np.asarray(
        ((supercell.n11, supercell.n12), (supercell.n21, supercell.n22)),
        dtype=np.int64,
    )
    inverse_transpose = np.linalg.inv(matrix).T
    mesh = super_cache.fold_map.primitive_mesh
    reduced_y = super_cache.fold_map.reduced_supercell_numerators
    for label_index, label in enumerate(super_cache.blocks.shifts):
        field_primitive = np.asarray(label, dtype=float) @ inverse_transpose
        field_cart = (
            np.asarray((field_primitive[0], field_primitive[1], 0.0))
            @ primitive_data.source.reciprocal_bohr
            * BOHR_INV_TO_NM_INV
        )
        expected_gvec = complex(field_cart[0], field_cart[1])
        assert abs(super_cache.blocks.gvecs[label_index] - expected_gvec) < 1.0e-12
        mask = super_cache.pair_masks[label]
        q_table = super_cache.channel_q_indices[label]
        for target, source in np.argwhere(mask):
            raw = reduced_y[target] - reduced_y[source]
            dy = (raw + mesh // 2) % mesh - mesh // 2
            q_index = int(q_table[target, source])
            q_num = primitive_data.overlap_cache.physical_q_numerators[
                q_index, :2
            ]
            np.testing.assert_array_equal(
                q_num @ matrix.T,
                dy + mesh * np.asarray(label, dtype=np.int64),
            )
            primitive_cart = (
                np.asarray((q_num[0] / mesh, q_num[1] / mesh, 0.0))
                @ primitive_data.source.reciprocal_bohr
            )
            super_fractional = dy / mesh + np.asarray(label, dtype=float)
            super_cart = (
                np.asarray(
                    (
                        *(super_fractional @ inverse_transpose),
                        0.0,
                    )
                )
                @ primitive_data.source.reciprocal_bohr
            )
            np.testing.assert_allclose(
                super_cart, primitive_cart, rtol=0.0, atol=1.0e-12
            )

    rng = np.random.default_rng(91 + area_ratio)
    primitive_density = rng.uniform(
        -0.9, -0.2, size=(1, 1, primitive_data.h0.shape[2])
    ).astype(np.complex128)
    folded_density = embed_ptse2_translation_invariant_density(
        primitive_density, super_cache.fold_map
    )
    primitive_action = build_projected_interaction_hamiltonian(
        primitive_density,
        primitive_data.overlap_cache.blocks,
        v0=1.0 / primitive_data.area_nm2,
        use_numba=False,
    )
    folded_action = build_projected_interaction_hamiltonian(
        folded_density,
        super_cache.blocks,
        v0=1.0 / (area_ratio * primitive_data.area_nm2),
        use_numba=False,
    )
    unfolded_action = unfold_ptse2_translation_invariant_operator(
        folded_action, super_cache.fold_map
    )
    np.testing.assert_allclose(
        unfolded_action, primitive_action, rtol=1.0e-12, atol=1.0e-12
    )
    folded_h0 = fold_ptse2_primitive_operator(
        primitive_data.h0, super_cache.fold_map
    )
    primitive_energy = compute_hf_energy(
        primitive_action, primitive_data.h0, primitive_density
    )
    folded_energy = compute_hf_energy(
        folded_action, folded_h0, folded_density
    )
    np.testing.assert_allclose(
        folded_energy / area_ratio,
        primitive_energy,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    broken_density = np.array(folded_density, copy=True)
    broken_density[0, 1, :] += 0.04 + 0.02j
    broken_density[1, 0, :] += 0.04 - 0.02j
    regrouped = build_projected_interaction_hamiltonian(
        broken_density,
        super_cache.blocks,
        v0=1.0 / (area_ratio * primitive_data.area_nm2),
        use_numba=False,
    )
    literal = build_ptse2_literal_physical_q_interaction(
        broken_density, primitive_data, super_cache.fold_map
    )
    np.testing.assert_allclose(regrouped, literal, rtol=1.0e-12, atol=1.0e-12)

    hf_data = build_ptse2_supercell_hf_data(primitive_data, supercell)
    assert hf_data.supercell_filling_nu == -area_ratio
    coefficients = ptse2_supercell_hole_fourier(
        hf_data.initial_density, hf_data.overlap_cache, use_numba=False
    )
    assert abs(coefficients[(0, 0)] - area_ratio) < 1.0e-12


@pytest.mark.parametrize(
    ("supercell", "primitive_filling", "supercell_filling", "occupied_per_k"),
    (
        (ptse2_2x1_supercell(), -0.5, -1, 3),
        (ptse2_sqrt3_supercell(), -2.0 / 3.0, -2, 4),
    ),
)
def test_fractional_primitive_fillings_require_integer_supercell_charge(
    supercell: IntegerSupercell,
    primitive_filling: float,
    supercell_filling: int,
    occupied_per_k: int,
) -> None:
    primitive_data = _synthetic_primitive_data(mesh=6, active_rank=2)
    data = build_ptse2_supercell_hf_data(
        primitive_data,
        supercell,
        target_primitive_filling_nu=primitive_filling,
    )
    assert data.supercell_filling_nu == supercell_filling
    projector = data.initial_density + np.eye(data.fold_map.folded_rank)[:, :, None]
    np.testing.assert_allclose(
        np.trace(projector, axis1=0, axis2=1).real.mean(),
        occupied_per_k,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.trace(data.initial_density, axis1=0, axis2=1).real.mean(),
        supercell_filling,
        rtol=0.0,
        atol=1.0e-12,
    )
    coefficients = ptse2_supercell_hole_fourier(
        data.initial_density,
        data.overlap_cache,
        expected_holes_per_supercell=-supercell_filling,
        use_numba=False,
    )
    assert abs(coefficients[(0, 0)] + supercell_filling) < 1.0e-12
    with pytest.raises(ValueError, match="expected filling"):
        ptse2_supercell_hole_fourier(
            data.initial_density,
            data.overlap_cache,
            use_numba=False,
        )


def test_fractional_primitive_filling_rejects_incommensurate_cell() -> None:
    primitive_data = _synthetic_primitive_data(mesh=6, active_rank=2)
    with pytest.raises(ValueError, match="incommensurate"):
        build_ptse2_supercell_hf_data(
            primitive_data,
            ptse2_2x1_supercell(),
            target_primitive_filling_nu=-2.0 / 3.0,
        )


def test_generic_physical_cdw_seeds_are_nontrivial_fixed_filling_projectors() -> None:
    primitive_data = _synthetic_primitive_data(mesh=6, active_rank=2)
    cases = (
        (ptse2_2x1_supercell(), ((1, 0),)),
        (ptse2_sqrt3_supercell(), ((0, 1),)),
    )
    for supercell, sectors in cases:
        data = build_ptse2_supercell_hf_data(primitive_data, supercell)
        density = initialize_ptse2_supercell_density(
            data,
            mode="cdw",
            seed=17,
            amplitude_ev=5.0e-3,
            sectors=sectors,
        )
        observables = ptse2_translation_observables(density, data.fold_map)
        assert observables.absolute_frobenius > 1.0e-8
        assert all(value > 1.0e-8 for value in observables.sector_frobenius.values())
        measured = np.trace(density, axis1=0, axis2=1).real.mean()
        assert abs(measured + data.fold_map.area_ratio) < 1.0e-12
        identity = np.eye(data.fold_map.folded_rank)
        for reduced_index in range(data.fold_map.reduced_nk):
            projector = density[:, :, reduced_index] + identity
            eigenvalues = np.linalg.eigvalsh(projector)
            assert eigenvalues[0] >= -1.0e-12
            assert eigenvalues[-1] <= 1.0 + 1.0e-12
        with pytest.raises(ValueError, match="2x2 initializer"):
            initialize_ptse2_2x2_density(data, mode="bare")
        with pytest.raises(ValueError, match="integer"):
            initialize_ptse2_supercell_density(
                data,
                mode="cdw",
                seed=1.5,
                amplitude_ev=5.0e-3,
                sectors=sectors,
            )

    sqrt3_data = build_ptse2_supercell_hf_data(
        primitive_data, ptse2_sqrt3_supercell()
    )
    with pytest.raises(ValueError, match="must break"):
        initialize_ptse2_supercell_density(
            sqrt3_data,
            mode="cdw",
            seed=3,
            amplitude_ev=5.0e-3,
            sectors=((1, 1),),
        )


def test_ptse2_stored_density_commutator_uses_physical_ket_projector() -> None:
    spinor = np.asarray([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    physical_projector = np.outer(spinor, spinor.conj())
    stored_density = (physical_projector.T - np.eye(2))[:, :, None]
    hamiltonian = np.asarray(
        [[0.3, 0.2 + 0.4j], [0.2 - 0.4j, -0.1]],
        dtype=np.complex128,
    )[:, :, None]

    actual = ptse2_stored_density_commutator(hamiltonian, stored_density)
    expected = (
        hamiltonian[:, :, 0] @ physical_projector
        - physical_projector @ hamiltonian[:, :, 0]
    )
    wrong_stored_orientation = (
        hamiltonian[:, :, 0] @ (stored_density[:, :, 0] + np.eye(2))
        - (stored_density[:, :, 0] + np.eye(2)) @ hamiltonian[:, :, 0]
    )

    np.testing.assert_allclose(actual[:, :, 0], expected, atol=1.0e-14)
    assert np.linalg.norm(actual[:, :, 0] - wrong_stored_orientation) > 0.1
    shifted = ptse2_level_shifted_hamiltonian(
        hamiltonian, stored_density, level_shift_ev=0.15
    )
    shifted_commutator = ptse2_stored_density_commutator(
        shifted, stored_density
    )
    np.testing.assert_allclose(shifted_commutator, actual, atol=1.0e-14)
    np.testing.assert_allclose(
        shifted, np.swapaxes(shifted.conj(), 0, 1), atol=1.0e-14
    )


def test_ptse2_level_shift_raises_only_virtual_energy_for_commuting_projector() -> None:
    physical_projector = np.diag([1.0, 0.0]).astype(np.complex128)
    stored_density = (physical_projector.T - np.eye(2))[:, :, None]
    hamiltonian = np.diag([-0.2, 0.3]).astype(np.complex128)[:, :, None]
    shifted = ptse2_level_shifted_hamiltonian(
        hamiltonian, stored_density, level_shift_ev=0.15
    )
    np.testing.assert_allclose(
        shifted[:, :, 0], np.diag([-0.2, 0.45]), atol=1.0e-14
    )
    np.testing.assert_array_equal(
        ptse2_level_shifted_hamiltonian(
            hamiltonian, stored_density, level_shift_ev=0.0
        ),
        hamiltonian,
    )


def test_ptse2_supercell_diis_adapter_performs_final_raw_replay() -> None:
    primitive_data = _synthetic_primitive_data(mesh=6, active_rank=1)
    data = build_ptse2_supercell_hf_data(
        primitive_data, ptse2_2x1_supercell()
    )
    run = run_ptse2_supercell_diis_hf(
        data,
        initial_density=data.initial_density,
        init_mode="bare",
        seed=1,
        options=PulayDIISOptions(max_history=4),
    )
    assert run.converged
    assert run.state.diagnostics["final_raw_norm"] == 0.0
    assert run.state.diagnostics["diis_restart_count"] == 0.0
    np.testing.assert_array_equal(run.state.density, data.initial_density)

    operator_run = run_ptse2_supercell_operator_diis_hf(
        data,
        initial_density=data.initial_density,
        init_mode="bare",
        seed=1,
        options=PulayDIISOptions(max_history=4),
    )
    assert operator_run.converged
    assert operator_run.state.diagnostics["final_raw_norm"] == 0.0
    assert operator_run.state.diagnostics["diis_commutator_rms_ev"] == 0.0
    np.testing.assert_array_equal(
        operator_run.state.density, data.initial_density
    )


def test_generic_translation_breaking_fourier_filter() -> None:
    for supercell, breaking_label, preserving_label in (
        (ptse2_2x1_supercell(), (1, 0), (0, 1)),
        (ptse2_sqrt3_supercell(), (1, 0), (1, 1)),
    ):
        fold_map = build_ptse2_fold_map(12, 1, supercell)
        coefficients = {
            (0, 0): complex(fold_map.area_ratio),
            breaking_label: 0.2 + 0.1j,
            (-breaking_label[0], -breaking_label[1]): 0.2 - 0.1j,
            preserving_label: 0.3 + 0.0j,
            (-preserving_label[0], -preserving_label[1]): 0.3 + 0.0j,
        }
        _, _, breaking = reconstruct_ptse2_supercell_hole_density(
            coefficients,
            supercell_area_nm2=float(fold_map.area_ratio),
            grid_size=24,
            translation_breaking_only=True,
            fold_map=fold_map,
        )
        assert np.max(np.abs(breaking)) > 0.0
        assert abs(np.mean(breaking)) < 1.0e-14
    with pytest.raises(ValueError, match="explicit fold_map"):
        reconstruct_ptse2_supercell_hole_density(
            {(0, 0): 2.0 + 0.0j},
            supercell_area_nm2=2.0,
            grid_size=8,
            translation_breaking_only=True,
        )


def test_hole_fourier_rekeys_stored_contraction_to_physical_label() -> None:
    fold_map = build_ptse2_fold_map(4, 1, ptse2_2x2_supercell())
    nt = fold_map.folded_rank
    nk = fold_map.reduced_nk
    identity = np.repeat(np.eye(nt)[:, :, None], nk, axis=2)
    plus = np.zeros((nt, nt, nk), dtype=np.complex128)
    plus[0, 1, :] = 1.0 + 2.0j
    minus = np.swapaxes(plus.conj(), 0, 1)
    zero_overlap = np.zeros((nt, nk, nt, nk), dtype=np.complex128)
    blocks = HFOverlapBlockSet(
        shifts=((0, 0), (1, 0), (-1, 0)),
        gvecs=np.asarray([0.0j, 0.5 + 0.0j, -0.5 + 0.0j]),
        overlaps={(0, 0): zero_overlap, (1, 0): zero_overlap, (-1, 0): zero_overlap},
        diagonal_overlaps={(0, 0): identity, (1, 0): plus, (-1, 0): minus},
        hartree_screening={(0, 0): 1.0, (1, 0): 1.0, (-1, 0): 1.0},
    )
    cache = PtSe2SupercellOverlapCache(
        blocks=blocks,
        fold_map=fold_map,
        pair_masks={label: np.eye(nk, dtype=bool) for label in blocks.shifts},
        channel_q_indices={label: np.zeros((nk, nk), dtype=np.int64) for label in blocks.shifts},
        assignment_counts={label: np.eye(nk, dtype=np.int64) * 4 for label in blocks.shifts},
        source_cache="synthetic",
        source_summary="synthetic",
        cache_sha256="synthetic",
        run_id="synthetic",
        hf_authorized=False,
    )
    density = np.repeat(
        (-np.eye(nt, dtype=np.complex128))[:, :, None], nk, axis=2
    )
    density[0, 1, :] = 0.1 + 0.2j
    density[1, 0, :] = 0.1 - 0.2j
    coefficients = ptse2_supercell_hole_fourier(density, cache, use_numba=False)
    raw_plus = -np.einsum("abk,bak->", density, np.conj(plus)) / nk
    assert coefficients[(-1, 0)] == raw_plus
    assert abs(coefficients[(1, 0)] - coefficients[(-1, 0)].conjugate()) < 1.0e-12


def test_ptse2_generic_supercell_zero_temperature_hessian_matches_scalar_energy() -> None:
    """Exercise synthetic algebra without authorizing a PtSe2 provider.

    Production use requires adapter-specific calibration of the per-pair
    hybrid self-adjointness criterion; synthetic tolerances are not authority.
    """

    primitive_data = _synthetic_primitive_data(mesh=6, active_rank=2)
    data = build_ptse2_supercell_hf_data(
        primitive_data, ptse2_2x1_supercell()
    )
    binding = build_ptse2_supercell_zero_temperature_hessian(
        data, data.initial_density
    )
    hessian = binding.hessian
    rng = np.random.default_rng(17)
    direction = rng.normal(size=hessian.shape[0])
    direction /= np.linalg.norm(direction)
    probes = rng.normal(size=(5, hessian.shape[0]))
    report = hessian.verify_self_adjointness(probes)
    assert report.maximum_relative_defect > 1.0e-3
    with pytest.raises(ValueError, match="self-adjointness"):
        hessian.as_linear_operator(
            verification_vectors=probes,
            absolute_tolerance=1.0e-10,
            relative_tolerance=1.0e-10,
        )

    tangent_identity = np.eye(hessian.shape[0])
    breaking_projector = np.column_stack(
        [
            ptse2_translation_breaking_orbital_action(binding, column)
            for column in tangent_identity.T
        ]
    )
    np.testing.assert_allclose(
        breaking_projector, breaking_projector.T, atol=1.0e-12
    )
    np.testing.assert_allclose(
        breaking_projector @ breaking_projector,
        breaking_projector,
        atol=1.0e-12,
    )

    analytic = float(direction @ hessian.energy_hessian_action(direction))

    def energy(amplitude: float) -> float:
        projector = hessian.frame.unitary_projector(
            direction, amplitude=amplitude
        )
        return ptse2_supercell_energy_from_physical_projector(data, projector)

    curvatures = []
    for step in (0.02, 0.01, 0.005):
        curvature = (
            -energy(2.0 * step)
            + 16.0 * energy(step)
            - 30.0 * energy(0.0)
            + 16.0 * energy(-step)
            - energy(-2.0 * step)
        ) / (12.0 * step**2)
        curvatures.append(curvature)
    np.testing.assert_allclose(
        curvatures, analytic, rtol=2.0e-6, atol=2.0e-7
    )
