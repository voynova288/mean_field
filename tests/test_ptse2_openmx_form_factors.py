from __future__ import annotations

import hashlib
import json

import numpy as np

import mean_field.systems.ptse2_openmx as ptse2_api
from mean_field.systems.ptse2_openmx import hf as ptse2_hf_owner
from mean_field.systems.ptse2_openmx import supercell as ptse2_supercell_owner
from mean_field.systems.ptse2_openmx.form_factors import (
    form_factor,
    reverse_form_factor_residual,
)
from mean_field.systems.ptse2_openmx.grid_oracle import (
    ActiveVertexProjectionRequest,
    PAOVertexBlock,
    assemble_reference_subtracted_reverse_pair,
    assemble_reference_subtracted_scalar_vertex,
    assemble_scalar_super_gauge_vertex,
    iter_oracle_shard_q_blocks,
    project_oracle_blocks_active_batch,
    realspace_overlap_to_scalar_blocks,
    read_bound_oracle_metadata,
    read_oracle_metadata,
)
from mean_field.systems.ptse2_openmx.hf import (
    PtSe2HFConfig,
    build_ptse2_hf_overlap_cache,
    ptse2_density_from_fixed_filling,
)
from mean_field.systems.ptse2_openmx.reciprocal import (
    ActiveReciprocalStates,
    OpenMXRadialTransformTable,
    ReciprocalGrid,
    build_active_reciprocal_states,
    generate_kinetic_reciprocal_grid,
    openmx_real_spherical_harmonics,
)
from mean_field.systems.ptse2_openmx.screening import PtSe2DoubleGateScreening
from mean_field.systems.ptse2_openmx.source import (
    OpenMXBasisSpec,
    OpenMXSource,
    assemble_super_gauge_matrix,
)
from mean_field.systems.ptse2_openmx.symmetry import (
    PAOSymmetryTransport,
    assemble_super_gauge_symmetry_sewing,
    load_pao_symmetry_transport_npz,
    write_pao_symmetry_transport_npz,
)
from mean_field.systems.ptse2_openmx.transfers import (
    PhysicalTransferPair,
    build_full_half_open_physical_q_chart,
    build_half_open_nearest_transfer_chart,
    find_q_index,
)


def test_ptse2_package_root_is_narrow_typed_hf_api() -> None:
    expected = {
        "PtSe2HFConfig",
        "PtSe2HFState",
        "PtSe2ProjectedHFData",
        "PtSe2SupercellHFData",
        "build_ptse2_projected_hf_data",
        "build_ptse2_supercell_hf_data",
        "initialize_ptse2_supercell_density",
        "run_ptse2_projected_hf",
        "run_ptse2_supercell_projected_hf",
    }
    assert set(ptse2_api.__all__) == expected
    for name in expected:
        owner = ptse2_supercell_owner if "Supercell" in name or "supercell" in name else ptse2_hf_owner
        assert getattr(ptse2_api, name) is getattr(owner, name)
    assert not hasattr(ptse2_api, "OpenMXSource")
    assert not hasattr(ptse2_api, "solve_ghep_root_csr")


def _one_atom_input(*, position: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str:
    x, y, z = position
    return f"""
Atoms.UnitVectors.Unit Ang
<Atoms.UnitVectors
  5.29177249 0.0 0.0
  0.0 5.29177249 0.0
  0.0 0.0 5.29177249
Atoms.UnitVectors>

Species.Number 1
<Definition.of.Atomic.Species
X X1.0-s1 X_PBE19
Definition.of.Atomic.Species>

Atoms.Number 1
Atoms.SpeciesAndCoordinates.Unit Ang
<Atoms.SpeciesAndCoordinates
1 X {x} {y} {z} 0.0 0.0
Atoms.SpeciesAndCoordinates>

scf.SpinPolarization NC
scf.SpinOrbit.Coupling On
"""


def _two_atom_input() -> str:
    return _one_atom_input().replace(
        "Atoms.Number 1\nAtoms.SpeciesAndCoordinates.Unit Ang\n"
        "<Atoms.SpeciesAndCoordinates\n"
        "1 X 0.0 0.0 0.0 0.0 0.0\n"
        "Atoms.SpeciesAndCoordinates>",
        "Atoms.Number 2\nAtoms.SpeciesAndCoordinates.Unit Ang\n"
        "<Atoms.SpeciesAndCoordinates\n"
        "1 X 0.0 0.0 0.0 0.0 0.0\n"
        "2 X 2.645886245 0.0 0.0 0.0 0.0\n"
        "Atoms.SpeciesAndCoordinates>",
    )


def test_ptse2_fixed_filling_density_uses_identity_reference() -> None:
    hamiltonian = np.zeros((3, 3, 2), dtype=np.complex128)
    hamiltonian[:, :, 0] = np.diag([0.0, 2.0, 4.0])
    hamiltonian[:, :, 1] = np.diag([1.0, 3.0, 5.0])
    neutral = ptse2_density_from_fixed_filling(hamiltonian, 0)
    np.testing.assert_array_equal(neutral.density, np.zeros_like(hamiltonian))
    assert neutral.observables["filling_nu"] == 0.0
    assert neutral.observables["total_occupied"] == 6.0

    one_hole = ptse2_density_from_fixed_filling(hamiltonian, -1)
    expected = np.zeros_like(hamiltonian)
    expected[2, 2, :] = -1.0
    np.testing.assert_allclose(one_hole.density, expected, atol=1.0e-15)
    assert one_hole.observables["filling_nu"] == -1.0
    assert one_hole.observables["total_occupied"] == 4.0
    np.testing.assert_allclose(one_hole.mu, 3.5)
    np.testing.assert_allclose(one_hole.observables["occupation_gap_ev"], 1.0)
    for k_index in range(2):
        stored_projector = one_hole.density[:, :, k_index] + np.eye(3)
        np.testing.assert_allclose(
            stored_projector @ stored_projector,
            stored_projector,
            atol=1.0e-14,
        )
    degenerate = np.zeros((2, 2, 1), dtype=np.complex128)
    with np.testing.assert_raises(ValueError):
        ptse2_density_from_fixed_filling(
            degenerate,
            -1,
            degeneracy_tolerance_ev=1.0e-12,
        )
    ensemble = ptse2_density_from_fixed_filling(
        degenerate,
        -1,
        degeneracy_tolerance_ev=1.0e-12,
        degenerate_occupation_policy="equal_ensemble",
    )
    np.testing.assert_allclose(
        ensemble.density[:, :, 0], -0.5 * np.eye(2), atol=1.0e-15
    )
    assert ensemble.observables["boundary_multiplet_size"] == 2.0
    assert ensemble.observables["boundary_occupation_fraction"] == 0.5
    assert ensemble.observables["degenerate_ensemble_used"] == 1.0


def test_ptse2_fixed_filling_transposes_complex_ket_projector() -> None:
    hamiltonian = np.array(
        [[[0.0], [1.0j]], [[-1.0j], [0.0]]], dtype=np.complex128
    )
    result = ptse2_density_from_fixed_filling(hamiltonian, -1)
    _, vectors = np.linalg.eigh(hamiltonian[:, :, 0])
    ket_delta = vectors[:, :1] @ vectors[:, :1].conj().T - np.eye(2)
    np.testing.assert_allclose(result.density[:, :, 0], ket_delta.T, atol=1.0e-14)
    assert abs(result.observables["filling_nu"] + 1.0) < 1.0e-14


def test_ptse2_hf_config_restricts_valence_integer_fillings() -> None:
    config = PtSe2HFConfig(
        filling_nu=-1,
        dielectric_constant=10.0,
        require_hf_authorized=False,
    )
    assert config.active_rank == 8
    with np.testing.assert_raises(ValueError):
        PtSe2HFConfig(filling_nu=1, dielectric_constant=10.0)
    with np.testing.assert_raises(ValueError):
        PtSe2HFConfig(filling_nu=-0.5, dielectric_constant=10.0)


def test_double_gate_screening_limits_and_units() -> None:
    screening = PtSe2DoubleGateScreening(
        epsilon_top=10.0,
        epsilon_bottom=10.0,
        d_top_nm=20.0,
        d_bottom_nm=20.0,
    )
    expected_zero = 4.0 * np.pi * 1.439964548 / (10.0 / 20.0 + 10.0 / 20.0)
    np.testing.assert_allclose(
        screening.interaction_ev_nm2(0.0), expected_zero, rtol=1.0e-15
    )
    q = 0.3
    expected = 2.0 * np.pi * 1.439964548 / (10.0 * q) * np.tanh(q * 20.0)
    np.testing.assert_allclose(
        screening.interaction_ev_nm2(q), expected, rtol=1.0e-15
    )


def test_ptse2_hf_overlap_bridge_preserves_pair_masks(tmp_path) -> None:
    source = OpenMXSource.from_text(_one_atom_input())
    cache_path = tmp_path / "projected.npz"
    summary_path = tmp_path / "summary.json"
    np.savez(
        cache_path,
        schema=np.asarray("ptse2_full_q_reverse_c3_projected_cache/v1"),
        run_id=np.asarray("synthetic-run"),
        active_rank=np.asarray(2, dtype=np.int64),
        input_cache_schema=np.asarray("synthetic-parent/v1"),
        local_fields=np.array([[0, 0, 0]], dtype=np.int64),
        admitted_mask=np.ones((1, 1, 1), dtype=bool),
        physical_q_numerators=np.zeros((1, 3), dtype=np.int64),
        physical_q_bohr_inv=np.zeros((1, 3)),
        channel_q_indices=np.zeros((1, 1, 1), dtype=np.int64),
        reduced_steps=np.zeros((1, 1, 3), dtype=np.int64),
        folding_carries=np.zeros((1, 1, 3), dtype=np.int64),
        projected_overlaps=np.eye(2, dtype=np.complex128).reshape(1, 2, 1, 2, 1),
    )
    cache_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    summary_path.write_text(
        json.dumps(
            {
                "run_id": "synthetic-run",
                "input_cache_schema": "synthetic-parent/v1",
                "hf_authorized": False,
                "qualification": {"hf_authorized": False},
                "outputs": {"projected_cache_sha256": cache_sha256},
            }
        )
    )
    screening = PtSe2DoubleGateScreening(10.0, 10.0, 20.0, 20.0)
    with np.testing.assert_raises(ValueError):
        build_ptse2_hf_overlap_cache(cache_path, summary_path, source, screening)
    cache = build_ptse2_hf_overlap_cache(
        cache_path,
        summary_path,
        source,
        screening,
        require_hf_authorized=False,
    )
    assert cache.blocks.shifts == ((0, 0),)
    np.testing.assert_allclose(
        cache.blocks.overlaps[(0, 0)],
        np.eye(2, dtype=np.complex128).reshape(2, 1, 2, 1),
    )
    assert cache.pair_masks[(0, 0)][0, 0]
    assert not cache.blocks.overlaps[(0, 0)].flags.writeable
    assert not cache.blocks.fock_screening[(0, 0)].flags.writeable
    assert not cache.hf_authorized


def test_openmx_basis_and_spinor_order() -> None:
    pt = OpenMXBasisSpec.parse("Pt", "Pt7.0-s3p2d2")
    se = OpenMXBasisSpec.parse("Se", "Se7.0-s3p2d1")
    assert pt.scalar_orbital_count == 19
    assert se.scalar_orbital_count == 14
    assert pt.channels()[:4] == (
        (0, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (1, 0, 0),
    )
    assert pt.channels()[3:6] == ((1, 0, 0), (1, 0, 1), (1, 0, 2))


def test_openmx_source_preserves_atom_and_spin_block_order() -> None:
    source = OpenMXSource.from_text(_one_atom_input())
    assert source.scalar_pao_count == 1
    assert source.spinor_pao_count == 2
    assert [entry.spin for entry in source.spinor_pao_map()] == [0, 1]
    np.testing.assert_allclose(
        source.lattice_bohr,
        10.0 * np.eye(3),
        rtol=0.0,
        atol=5.0e-9,
    )
    np.testing.assert_allclose(
        source.lattice_bohr @ source.reciprocal_bohr.T,
        2.0 * np.pi * np.eye(3),
        atol=1.0e-14,
    )


def _synthetic_c3_transport(source: OpenMXSource) -> PAOSymmetryTransport:
    phase = np.exp(-1j * np.pi / 3.0)
    spin_c3 = np.diag([phase, phase.conjugate()]).astype(np.complex128)
    return PAOSymmetryTransport(
        operation_index=1,
        rotation_fractional=np.array(
            [[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=np.int64
        ),
        translation_fractional=np.zeros(3),
        atom_mappings=np.array([0], dtype=np.int64),
        image_shifts=np.zeros((1, 3), dtype=np.int64),
        local_spinor_blocks=(spin_c3,),
        source_json_sha256="synthetic-c3",
        maximum_mapping_residual=0.0,
        maximum_local_unitarity_residual=0.0,
    )


def test_super_gauge_c3_sewing_folds_and_obeys_double_group() -> None:
    source = OpenMXSource.from_text(_one_atom_input())
    transport = _synthetic_c3_transport(source)
    k0 = np.array([1.0 / 12.0, 2.0 / 12.0, 0.0])
    sewing1 = assemble_super_gauge_symmetry_sewing(source, transport, k0)
    sewing2 = assemble_super_gauge_symmetry_sewing(
        source, transport, sewing1.target_k_folded_fractional
    )
    sewing3 = assemble_super_gauge_symmetry_sewing(
        source, transport, sewing2.target_k_folded_fractional
    )
    np.testing.assert_allclose(
        sewing1.target_k_unfolded_fractional,
        [-3.0 / 12.0, 1.0 / 12.0, 0.0],
        atol=1.0e-15,
    )
    np.testing.assert_array_equal(sewing1.target_reciprocal_carry, [-1, 0, 0])
    np.testing.assert_allclose(
        (sewing1.matrix.getH() @ sewing1.matrix).toarray(),
        np.eye(source.spinor_pao_count),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        (sewing3.matrix @ sewing2.matrix @ sewing1.matrix).toarray(),
        -np.eye(source.spinor_pao_count),
        atol=1.0e-14,
    )


def test_super_gauge_fold_phase_with_nonzero_center_and_image() -> None:
    source = OpenMXSource.from_text(
        _one_atom_input(position=(2.645886245, 0.0, 0.0))
    )
    transport = _synthetic_c3_transport(source)
    transport = PAOSymmetryTransport(
        operation_index=transport.operation_index,
        rotation_fractional=transport.rotation_fractional,
        translation_fractional=transport.translation_fractional,
        atom_mappings=transport.atom_mappings,
        image_shifts=np.array([[1, -1, 0]], dtype=np.int64),
        local_spinor_blocks=transport.local_spinor_blocks,
        source_json_sha256=transport.source_json_sha256,
        maximum_mapping_residual=transport.maximum_mapping_residual,
        maximum_local_unitarity_residual=transport.maximum_local_unitarity_residual,
    )
    k = np.array([1.0 / 12.0, 2.0 / 12.0, 0.0])
    sewing = assemble_super_gauge_symmetry_sewing(source, transport, k)
    tau = np.array([0.5, 0.0, 0.0])
    expected_phase = np.exp(
        2j
        * np.pi
        * (
            k @ tau
            - sewing.target_k_unfolded_fractional
            @ (tau + transport.image_shifts[0])
            + sewing.target_reciprocal_carry @ tau
        )
    )
    np.testing.assert_allclose(
        sewing.matrix.toarray(),
        expected_phase * transport.local_spinor_blocks[0],
        atol=1.0e-14,
    )


def test_compact_symmetry_transport_roundtrip(tmp_path) -> None:
    source = OpenMXSource.from_text(_one_atom_input())
    transport = _synthetic_c3_transport(source)
    path = tmp_path / "c3_transport.npz"
    write_pao_symmetry_transport_npz(path, transport)
    loaded = load_pao_symmetry_transport_npz(
        path, source, expected_json_sha256="synthetic-c3"
    )
    np.testing.assert_array_equal(
        loaded.rotation_fractional, transport.rotation_fractional
    )
    np.testing.assert_allclose(
        loaded.local_spinor_blocks[0], transport.local_spinor_blocks[0]
    )


def test_full_half_open_physical_q_chart_exact_closure() -> None:
    source = OpenMXSource.from_text(_one_atom_input())
    rotation = np.array(
        [[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=np.int64
    )
    chart = build_full_half_open_physical_q_chart(
        source, 12, c3_rotation_fractional=rotation
    )
    expected_fields = [
        (-1, 0, 0),
        (-1, 1, 0),
        (0, -1, 0),
        (0, 0, 0),
        (0, 1, 0),
        (1, -1, 0),
        (1, 0, 0),
        (1, 1, 0),
    ]
    assert [tuple(row) for row in chart.local_fields] == expected_fields
    assert chart.admitted_channel_count == 36432
    assert chart.physical_q_numerators.shape == (253, 3)
    np.testing.assert_array_equal(chart.physical_q_numerators[0], [0, 0, 0])
    mask_counts = {
        tuple(field): int(np.count_nonzero(chart.admitted_mask[index]))
        for index, field in enumerate(chart.local_fields)
    }
    assert mask_counts == {
        (-1, 0, 0): 2160,
        (-1, 1, 0): 864,
        (0, -1, 0): 2160,
        (0, 0, 0): 20736,
        (0, 1, 0): 4752,
        (1, -1, 0): 864,
        (1, 0, 0): 4752,
        (1, 1, 0): 144,
    }
    q_counts = np.bincount(
        chart.channel_q_indices[chart.channel_q_indices >= 0], minlength=253
    )
    np.testing.assert_array_equal(q_counts, np.full(253, 144))
    for local_index, target, source_index in np.argwhere(chart.admitted_mask):
        q_index = int(chart.channel_q_indices[local_index, target, source_index])
        reverse_local = int(
            chart.reverse_local_field_indices[local_index, target, source_index]
        )
        reverse_q = int(
            chart.channel_q_indices[reverse_local, source_index, target]
        )
        np.testing.assert_array_equal(
            chart.physical_q_numerators[reverse_q],
            -chart.physical_q_numerators[q_index],
        )


def test_half_open_transfer_chart_closes_reverse_and_c3_orbits() -> None:
    source = OpenMXSource.from_text(_one_atom_input())
    rotation = np.array(
        [[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=np.int64
    )
    chart = build_half_open_nearest_transfer_chart(
        source, 12, c3_rotation_fractional=rotation
    )
    assert len(chart.records) == 3 * 12 * 12
    assert len(chart.c3_images) == len(chart.records)
    q_inventory = set()
    for record in chart.records:
        np.testing.assert_allclose(
            record.pair.reverse.q_bohr_inv,
            -record.pair.forward.q_bohr_inv,
            atol=1.0e-13,
            rtol=0.0,
        )
        q_inventory.add(
            tuple(np.round(record.pair.forward.q_bohr_inv, decimals=14))
        )
        q_inventory.add(
            tuple(np.round(record.pair.reverse.q_bohr_inv, decimals=14))
        )
    assert len(q_inventory) == 6
    assert chart.c3_images[0].record_index == 2 * 12 * 12
    assert chart.c3_images[0].orientation == 1
    assert chart.c3_images[12 * 12].record_index == 11 * 12
    assert chart.c3_images[12 * 12].orientation == -1
    assert chart.c3_images[2 * 12 * 12].record_index == 12 * 12 + 11
    assert chart.c3_images[2 * 12 * 12].orientation == -1
    seam = next(
        record
        for record in chart.records
        if record.source_index == 11 * 12 and record.direction_index == 0
    )
    np.testing.assert_array_equal(seam.pair.forward.folding_carry, [1, 0, 0])
    np.testing.assert_array_equal(seam.pair.reverse.folding_carry, [-1, 0, 0])


def test_super_gauge_assembler_uses_atom_center_phase_at_nonzero_k() -> None:
    source = OpenMXSource.from_text(_two_atom_input())
    blocks = {
        (0, 0, 0): (
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2.0 + 0.0j]),
        )
    }
    matrix = assemble_super_gauge_matrix(blocks, source, [0.25, 0.0, 0.0])
    np.testing.assert_allclose(matrix[0, 1], 2.0 * np.exp(1j * np.pi / 4.0))


def test_sparse_spinor_overlap_converts_to_scalar_atom_blocks() -> None:
    source = OpenMXSource.from_text(_two_atom_input())
    blocks = {
        (1, 0, 0): (
            np.array([0, 2], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
            np.array([2.0 + 0.5j, 2.0 + 0.5j]),
        )
    }
    converted = realspace_overlap_to_scalar_blocks(blocks, source)
    assert len(converted) == 1
    assert converted[0].central_atom == 0
    assert converted[0].neighbor_atom == 1
    assert converted[0].cell == (1, 0, 0)
    np.testing.assert_allclose(converted[0].values, [[2.0 + 0.5j]])
    assert len(
        realspace_overlap_to_scalar_blocks(
            blocks, source, shard_rank=1, shard_count=2
        )
    ) == 0


def test_local_oracle_block_batch_projection_matches_dense_projection() -> None:
    source = OpenMXSource.from_text(_two_atom_input())
    block = PAOVertexBlock(
        central_atom=1,
        neighbor_atom=0,
        cell=(1, 0, 0),
        values=np.array([[2.0 + 0.5j]]),
    )
    target = np.array(
        [
            [1.0, 0.2j],
            [0.3, -0.1j],
            [0.4j, 0.7],
            [0.2, -0.5j],
        ],
        dtype=np.complex128,
    )
    source_coeff = np.array(
        [
            [0.1, 0.6j],
            [0.8, 0.2],
            [0.3j, 0.5],
            [-0.2j, 0.9],
        ],
        dtype=np.complex128,
    )
    request = ActiveVertexProjectionRequest(
        k_target_fractional=np.array([0.25, 0.0, 0.0]),
        k_source_fractional=np.array([0.125, 0.0, 0.0]),
        target_coefficients=target,
        source_coefficients=source_coeff,
    )
    second_request = ActiveVertexProjectionRequest(
        k_target_fractional=np.array([0.375, 0.0, 0.0]),
        k_source_fractional=np.array([0.25, 0.0, 0.0]),
        target_coefficients=1.1 * target,
        source_coefficients=0.9 * source_coeff,
    )
    projected = project_oracle_blocks_active_batch(
        [block], source, [request, second_request]
    )
    scalar = np.zeros((2, 2), dtype=np.complex128)
    target_cart = request.k_target_fractional @ source.reciprocal_angstrom
    source_cart = request.k_source_fractional @ source.reciprocal_angstrom
    r_cart = np.array(block.cell) @ source.lattice_angstrom
    phase = np.exp(
        1j * float(source_cart @ r_cart)
        - 1j * float(target_cart @ source.atoms[1].position_angstrom)
        + 1j * float(source_cart @ source.atoms[0].position_angstrom)
    )
    scalar[1, 0] = phase * block.values[0, 0]
    expected = (
        target[:2].conj().T @ scalar @ source_coeff[:2]
        + target[2:].conj().T @ scalar @ source_coeff[2:]
    )
    np.testing.assert_allclose(projected[0], expected, atol=1.0e-14)
    scalar_second = np.zeros((2, 2), dtype=np.complex128)
    target_cart = second_request.k_target_fractional @ source.reciprocal_angstrom
    source_cart = second_request.k_source_fractional @ source.reciprocal_angstrom
    phase_second = np.exp(
        1j * float(source_cart @ r_cart)
        - 1j * float(target_cart @ source.atoms[1].position_angstrom)
        + 1j * float(source_cart @ source.atoms[0].position_angstrom)
    )
    scalar_second[1, 0] = phase_second * block.values[0, 0]
    expected_second = (
        second_request.target_coefficients[:2].conj().T
        @ scalar_second
        @ second_request.source_coefficients[:2]
        + second_request.target_coefficients[2:].conj().T
        @ scalar_second
        @ second_request.source_coefficients[2:]
    )
    np.testing.assert_allclose(projected[1], expected_second, atol=1.0e-14)


def test_grid_oracle_reader_and_super_gauge_assembly(tmp_path) -> None:
    source = OpenMXSource.from_text(_two_atom_input())
    shard = tmp_path / "oracle.rank00000.bin"
    with shard.open("wb") as handle:
        handle.write(b"OMXPAOVERTEX1\x00\x00\x00")
        np.array([1, 0, 1, 2, 1, 1], dtype="<i4").tofile(handle)
        np.array([[0.0, 0.0, 0.0]], dtype="<f8").tofile(handle)
        np.array([0], dtype="<i4").tofile(handle)
        np.array([1, 2, 0, 0, 0, 1, 1], dtype="<i4").tofile(handle)
        np.array([2.0, 0.0], dtype="<f8").tofile(handle)
    metadata = read_oracle_metadata([shard])
    streamed_channels = list(iter_oracle_shard_q_blocks(metadata, 0))
    assert len(streamed_channels) == 1
    assert streamed_channels[0][0] == 0
    assert len(streamed_channels[0][1]) == 1
    np.testing.assert_allclose(streamed_channels[0][1][0].values, [[2.0]])
    anchor = tmp_path / "anchor.npz"
    anchor.write_bytes(b"synthetic anchor")
    manifest_path = tmp_path / "oracle_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "ptse2_openmx_pao_vertex_oracle/v1",
                "source_id": "synthetic-bound-oracle",
                "source_class": "synthetic_test",
                "source_input_sha256": source.input_sha256,
                "source_structure_digest": source.structure_digest,
                "operator_convention": "absolute_exp_plus_i_Q_dot_r",
                "q_units": "bohr^-1_cartesian",
                "anchor_overlap_sha256": hashlib.sha256(anchor.read_bytes()).hexdigest(),
                "q_vectors_bohr_inv": [[0.0, 0.0, 0.0]],
                "shards": [
                    {
                        "file": shard.name,
                        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    bound = read_bound_oracle_metadata(
        manifest_path, source, anchor, expected_source_class="synthetic_test"
    )
    assert bound.binding_verified
    with np.testing.assert_raises(ValueError):
        assemble_scalar_super_gauge_vertex(
            metadata,
            source,
            0,
            [0.25, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        )
    vertex = assemble_scalar_super_gauge_vertex(
        metadata,
        source,
        0,
        [0.25, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        allow_unbound=True,
    )
    np.testing.assert_allclose(vertex[0, 1], 2.0 * np.exp(1j * np.pi / 4.0))
    bound_vertex = assemble_scalar_super_gauge_vertex(
        bound,
        source,
        0,
        [0.25, 0.0, 0.0],
        [0.25, 0.0, 0.0],
    )
    np.testing.assert_allclose(bound_vertex.toarray(), vertex.toarray())

    overlap_blocks = {
        (0, 0, 0): (
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([3.0 + 0.0j]),
        )
    }
    corrected = assemble_reference_subtracted_scalar_vertex(
        metadata,
        source,
        0,
        0,
        overlap_blocks,
        [0.25, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        allow_unbound=True,
    )
    np.testing.assert_allclose(corrected[0, 1], 3.0 * np.exp(1j * np.pi / 4.0))


def test_reference_subtracted_reverse_projection_retains_raw_pair(tmp_path) -> None:
    source = OpenMXSource.from_text(_one_atom_input())
    shard = tmp_path / "oracle.rank00000.bin"
    b1 = source.reciprocal_bohr[0]
    q_vectors = np.array([[0.0, 0.0, 0.0], b1, -b1])
    values = [2.0 + 0.0j, 4.0 + 1.0j, 3.0 - 2.0j]
    with shard.open("wb") as handle:
        handle.write(b"OMXPAOVERTEX1\x00\x00\x00")
        np.array([1, 0, 1, 1, 3, 1], dtype="<i4").tofile(handle)
        q_vectors.astype("<f8").tofile(handle)
        for q_index, value in enumerate(values):
            np.array([q_index], dtype="<i4").tofile(handle)
            np.array([1, 1, 0, 0, 0, 1, 1], dtype="<i4").tofile(handle)
            np.array([value.real, value.imag], dtype="<f8").tofile(handle)
    metadata = read_oracle_metadata([shard])
    overlap_blocks = {
        (0, 0, 0): (
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([3.0 + 0.0j]),
        )
    }
    transfer_pair = PhysicalTransferPair.create(
        source,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
    )
    pair = assemble_reference_subtracted_reverse_pair(
        metadata,
        source,
        transfer_pair,
        0,
        overlap_blocks,
        allow_unbound=True,
    )
    np.testing.assert_allclose(pair.raw_forward.toarray(), [[5.0 + 1.0j]])
    np.testing.assert_allclose(pair.raw_reverse.toarray(), [[4.0 - 2.0j]])
    np.testing.assert_allclose(pair.projected_forward.toarray(), [[4.5 + 1.5j]])
    np.testing.assert_allclose(
        pair.projected_reverse.toarray(), pair.projected_forward.toarray().conj().T
    )


def test_openmx_real_harmonic_order_on_cartesian_axes() -> None:
    vectors = np.eye(3)
    p = openmx_real_spherical_harmonics(1, vectors)
    coefficient_p = 0.48860251190292
    np.testing.assert_allclose(p, coefficient_p * np.eye(3), atol=1.0e-15)

    d_x = openmx_real_spherical_harmonics(2, np.array([[1.0, 0.0, 0.0]]))[0]
    np.testing.assert_allclose(
        d_x,
        [-0.31539156525252, 0.54627421529604, 0.0, 0.0, 0.0],
        atol=1.0e-15,
    )


def test_radial_transform_table_layout_and_grid_point_interpolation(tmp_path) -> None:
    spec = OpenMXBasisSpec.parse("X", "X1.0-s1p1")
    first = np.array([1.0, 2.0, 3.0, 4.0, 99.0, 98.0])
    second = np.array([5.0, 6.0, 7.0, 8.0, 97.0, 96.0])
    path = tmp_path / "x.ftpao"
    np.concatenate([first, second]).astype(np.float64).tofile(path)
    table = OpenMXRadialTransformTable.from_file(
        path,
        [spec],
        source_id="synthetic-radial-table",
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        energy_cutoff_ry=16.0,
        momentum_grid_size=4,
        padding_values=2,
    )
    np.testing.assert_allclose(table.momentum_grid_bohr_inv, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(table.evaluate("X", 0, 0, np.arange(4.0)), first[:4])
    np.testing.assert_allclose(table.evaluate("X", 1, 0, np.arange(4.0)), second[:4])
    np.testing.assert_allclose(
        table.evaluate("X", 0, 0, np.array([0.25, 1.5, 2.75])),
        [1.25, 2.5, 3.75],
    )
    assert table.evaluate("X", 0, 0, 3.1) == 0.0


def test_kinetic_grid_at_zero_cutoff_contains_only_gamma() -> None:
    source = OpenMXSource.from_text(_one_atom_input())
    grid = generate_kinetic_reciprocal_grid(source, [0.0, 0.0, 0.0], 0.0)
    np.testing.assert_array_equal(grid.integer_indices, [[0, 0, 0]])


def test_half_open_transfer_pair_closes_physical_q_and_carries() -> None:
    source = OpenMXSource.from_text(_one_atom_input())
    pair = PhysicalTransferPair.create(
        source,
        [0.0, 0.0, 0.0],
        [5.0 / 6.0, 0.0, 0.0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 0],
    )
    np.testing.assert_array_equal(pair.forward.integer_shift, [1, 0, 0])
    np.testing.assert_array_equal(pair.reverse.integer_shift, [-1, 0, 0])
    np.testing.assert_allclose(pair.reverse.q_bohr_inv, -pair.forward.q_bohr_inv)
    inventory = np.array([pair.forward.q_bohr_inv, pair.reverse.q_bohr_inv])
    assert find_q_index(inventory, pair.forward) == 0
    with np.testing.assert_raises(ValueError):
        find_q_index(np.zeros((1, 3)), pair.forward)


def test_shifted_form_factor_uses_zero_fill_not_cyclic_roll() -> None:
    grid = ReciprocalGrid(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64))
    reciprocal = np.eye(3)
    source = ActiveReciprocalStates(
        source_id="synthetic-source",
        reciprocal_bohr=reciprocal,
        k_fractional=np.zeros(3),
        grid=grid,
        coefficients=np.array([[[1.0 + 1.0j]], [[0.0]]]),
    )
    target = ActiveReciprocalStates(
        source_id="synthetic-source",
        reciprocal_bohr=reciprocal,
        k_fractional=np.zeros(3),
        grid=grid,
        coefficients=np.array([[[0.0]], [[2.0 - 1.0j]]]),
    )
    result = form_factor(target, source, [1, 0, 0])
    assert result.match.match_count == 1
    np.testing.assert_allclose(result.values, [[(2.0 + 1.0j) * (1.0 + 1.0j)]])
    assert reverse_form_factor_residual(target, source, [1, 0, 0]) == 0.0

    with np.testing.assert_raises(TypeError):
        form_factor(target, source, [1.0, 0.0, 0.0])

    no_wrap = form_factor(target, source, [2, 0, 0])
    assert no_wrap.match.match_count == 0
    np.testing.assert_array_equal(no_wrap.values, np.zeros((1, 1)))


def test_active_projection_uses_super_gauge_center_phase() -> None:
    source = OpenMXSource.from_text(
        _one_atom_input(position=(5.29177249 / 2.0, 0.0, 0.0))
    )
    spec = source.basis_specs[0]
    table = OpenMXRadialTransformTable(
        source_id="synthetic-radial-table",
        artifact_sha256="0" * 64,
        species_order=("X",),
        basis_specs=(spec,),
        momentum_grid_bohr_inv=np.array([0.0, 1.0, 2.0, 3.0]),
        values={("X", 0, 0): np.ones(4)},
        stored_values_per_function=6,
    )
    grid = ReciprocalGrid(np.array([[1, 0, 0]], dtype=np.int64))
    c = np.array([[1.0], [0.0]], dtype=np.complex128)
    active = build_active_reciprocal_states(
        source,
        table,
        [0.25, 0.0, 0.0],
        grid,
        c,
        batch_size=1,
    )
    expected_amplitude = -4.0 * np.pi * 0.282094791773878 / np.sqrt(1000.0)
    np.testing.assert_allclose(active.coefficients[0, 0, 0], expected_amplitude)
    np.testing.assert_allclose(active.coefficients[0, 1, 0], 0.0)
