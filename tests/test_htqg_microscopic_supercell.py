"""T0--T8 draft contracts for HTQG microscopic Batch-A infrastructure.

All tests are algebraic and tiny except the shell-1 T3 and physical T8 gates,
which are marked ``slow`` for Slurm-only execution. The screened interaction,
reference, and filling contract is source-bound to prior reciprocal-space HTQG
HF. The physical T8 gate is implemented but does not gain numerical authority
until its sealed Slurm run passes; the wall profile remains unresolved.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import mean_field.systems.htqg.microscopic_hf as microscopic_hf_module
import mean_field.systems.htqg.microscopic_supercell as microscopic_supercell_module

from mean_field.core.hf.density_vertex import (
    DensityVertexInteractionSpec,
    DensityVertexReferenceSpec,
    PreparedDensityVertexFamily,
    SparseDensityVertex as CoreSparseDensityVertex,
    hartree_fock_action as core_hartree_fock_action,
    hartree_fock_action_and_energy as core_hartree_fock_action_and_energy,
    prepare_density_vertex_family,
)
from mean_field.core.hf.occupations import global_fermi_occupations
from mean_field.core.hf.problem import run_hartree_fock_problem
from mean_field.core.hf.coulomb import screened_coulomb
from mean_field.core.supercell import IntegerSupercell
from mean_field.systems.htqg.hamiltonian import moire_coupling_matrix
from mean_field.systems.htqg.lattice import build_htqg_lattice
from mean_field.systems.htqg.microscopic_hf import (
    MicroscopicHFFinalGateTolerances,
    MicroscopicHFFiniteTConvergence,
    MicroscopicHFState,
    build_batch_a_screened_coulomb_hf_inputs,
    build_global_neutral_h0_reference,
    build_half_filled_h0_reference,
    build_microscopic_hf_problem,
    build_microscopic_hf_source_binding,
    build_screened_coulomb_hf_inputs,
    verify_reciprocal_hf_authority,
)
from mean_field.systems.htqg.microscopic_supercell import (
    BatchAFoldMap,
    InteractionReferences,
    MicroscopicBasisLayout,
    MicroscopicH0Artifact,
    MicroscopicH0Spec,
    MicroscopicInteractionSpec,
    ProfileFourierInput,
    SparseDensityVertex,
    assemble_direct_h0,
    assemble_direct_h0_artifact,
    assemble_direct_h0_dft_oracle,
    assemble_primitive_endpoint_h0,
    batch_a_supercell,
    build_batch_a_fold_map,
    build_batch_a_tr_covariant_support,
    build_periodic_smoothstep_profile,
    build_plane_wave_density_vertices,
    build_primitive_plane_wave_density_vertices,
    build_profile_fourier_multiplier,
    fold_primitive_blocks,
    unfold_translation_invariant_blocks,
    global_canonical_occupations,
    hartree_fock_action,
    hartree_fock_energy,
    identity_density_vertex,
    ket_blocks_to_repository_stored,
    ket_hamiltonian_to_repository_blocks,
    literal_wick_four_index_action,
    literal_wick_four_index_energy,
    primitive_supercell_interaction_replay_scaffold,
    repository_hamiltonian_to_ket_blocks,
    repository_stored_to_ket_blocks,
    reverse_density_vertex,
    uniform_h0_replay_residuals,
    validate_batch_a_plane_wave_density_vertices,
    validate_microscopic_h0_artifact,
    validate_density_vertex_family,
    validate_primitive_supercell_density_vertex_equivalence,
)
from mean_field.systems.htqg.params import HTQGParams


AUTHORITY_FIXTURE = (
    Path(__file__).resolve().parents[1] / "data/htqg_reciprocal_hf_authority"
)


def _authority_root() -> Path:
    return AUTHORITY_FIXTURE


def _fold_map():
    return build_batch_a_fold_map(
        supercell=batch_a_supercell(),
        primitive_mesh=(8, 4),
        reduced_mesh=(2, 4),
    )


def _layout(g_indices: np.ndarray | None = None) -> MicroscopicBasisLayout:
    labels = np.asarray([[0, 0]], dtype=np.int64) if g_indices is None else g_indices
    return MicroscopicBasisLayout(
        g_indices=labels,
        valley_order=(1, -1),
        spin_order=("up", "down"),
        period_cells=4,
    )


def _support(
    layout: MicroscopicBasisLayout,
    fold_map=None,
):
    resolved_fold_map = _fold_map() if fold_map is None else fold_map
    return build_batch_a_tr_covariant_support(
        layout,
        resolved_fold_map,
        provenance="test-only exact Batch-A TR-covariant physical-label support",
    )

def _constant_profile(value: float, *, nsample: int = 16) -> ProfileFourierInput:
    x = np.arange(nsample, dtype=float) * 4.0 / float(nsample)
    values = np.full((2, nsample), float(value), dtype=np.complex128)
    return ProfileFourierInput(
        sample_positions_cells=x,
        values_by_valley=values,
        period_cells=4,
        convention="numpy_fft_exp_minus_iqx",
        provenance=f"test-only constant s={value}; no physical wall authority",
    )


def _nonuniform_profile(*, nsample: int = 32) -> ProfileFourierInput:
    x = np.arange(nsample, dtype=float) * 4.0 / float(nsample)
    s = (
        0.5
        + 0.22 * np.cos(2.0 * np.pi * x / 4.0)
        + 0.11 * np.sin(4.0 * np.pi * x / 4.0)
    )
    return ProfileFourierInput(
        sample_positions_cells=x,
        values_by_valley=np.asarray((s, s), dtype=np.complex128),
        period_cells=4,
        convention="numpy_fft_exp_minus_iqx",
        provenance="test-only nonuniform real profile for direct-DFT oracle",
    )


def _absolute_interaction(*, include_hartree: bool = True, include_fock: bool = True):
    return MicroscopicInteractionSpec(
        include_hartree=include_hartree,
        include_fock=include_fock,
        hartree_reference_policy="absolute",
        fock_reference_policy="absolute",
        zero_mode_policy="include_supplied",
        closure_tolerance=1.0e-13,
        normalization_provenance="test-only supplied scalar vertex weights",
        unresolved_physical_choices=(),
    )


def _absolute_references() -> InteractionReferences:
    return InteractionReferences(
        hartree=None,
        fock=None,
        provenance="test-only absolute-density policy",
    )


def _cross_k_vertex_family() -> tuple[SparseDensityVertex, ...]:
    gamma0 = identity_density_vertex(
        nk=2, dimension=2, weight=0.37, provenance="test-only Gamma0"
    )
    plus = SparseDensityVertex(
        label=(1, 0),
        target_k=np.asarray([1, 1, 0, 0]),
        source_k=np.asarray([0, 0, 1, 1]),
        rows=np.asarray([0, 1, 0, 1]),
        columns=np.asarray([1, 0, 1, 0]),
        values=np.asarray([1.0 + 0.2j, -0.3 + 0.8j, 0.4 - 0.1j, 0.7j]),
        weight=0.19,
        provenance="test-only complex cross-K Gamma_+",
    )
    return (
        gamma0,
        plus,
        reverse_density_vertex(plus, provenance="test-only Gamma_- adjoint"),
    )


def test_t0_reciprocal_hf_authority_hash_binding(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    verify_reciprocal_hf_authority(AUTHORITY_FIXTURE)

    (tmp_path / "SOURCE_MANIFEST.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="authority hash mismatch"):
        verify_reciprocal_hf_authority(tmp_path)

    lattice = build_htqg_lattice(2.25, n_shells=0)
    layout = _layout(lattice.g_indices)
    h0 = np.zeros((_fold_map().reduced_nk, layout.dimension, layout.dimension))
    with pytest.raises(ValueError, match="canonical shell 1, 2, or 3"):
        build_batch_a_screened_coulomb_hf_inputs(
            h0_ket=h0,
            lattice=lattice,
            layout=layout,
            fold_map=_fold_map(),
            support=_support(layout),
            authority_capsule_root=AUTHORITY_FIXTURE,
            max_vertex_entries=40_000,
            provenance="strict-constructor rejection test",
        )

    for authorized_shell in (1, 2, 3):
        authorized_lattice = build_htqg_lattice(
            2.25, n_shells=authorized_shell
        )
        authorized_layout = _layout(authorized_lattice.g_indices)
        authorized_support = _support(authorized_layout)
        with pytest.raises(ValueError, match="UNRESOLVED.*Hartree reference"):
            build_batch_a_screened_coulomb_hf_inputs(
                h0_ket=np.zeros((1, 1, 1), dtype=np.complex128),
                lattice=authorized_lattice,
                layout=authorized_layout,
                fold_map=_fold_map(),
                support=authorized_support,
                authority_capsule_root=AUTHORITY_FIXTURE,
                max_vertex_entries=2_000_000,
                provenance=(
                    f"strict shell-{authorized_shell} unresolved-reference rejection test"
                ),
            )
        with pytest.raises(ValueError, match="UNRESOLVED.*Hartree reference"):
            build_screened_coulomb_hf_inputs(
                h0_ket=np.zeros((1, 1, 1), dtype=np.complex128),
                lattice=authorized_lattice,
                layout=authorized_layout,
                fold_map=_fold_map(),
                support=authorized_support,
                authority_capsule_root=_authority_root(),
                epsilon_r=5.0,
                d_sc_nm=25.0,
                primitive_filling=-3,
                max_vertex_entries=2_000_000,
                provenance=(
                    f"direct shell-{authorized_shell} Batch-A bypass rejection test"
                ),
            )

    shell2_lattice = build_htqg_lattice(2.25, n_shells=2)
    shell2_layout = _layout(shell2_lattice.g_indices)
    shell2_support = _support(shell2_layout)
    noncanonical_indices = np.roll(shell2_lattice.g_indices, 1, axis=0)
    rejected_layouts = (
        (
            replace(shell2_lattice, g_indices=noncanonical_indices),
            _layout(noncanonical_indices),
            shell2_support,
        ),
        (replace(shell2_lattice, n_shells=6), shell2_layout, shell2_support),
    )
    shell6_lattice = build_htqg_lattice(2.25, n_shells=6)
    shell6_layout = _layout(shell6_lattice.g_indices)
    rejected_layouts += (
        (shell6_lattice, shell6_layout, _support(shell6_layout)),
    )
    for rejected_lattice, rejected_layout, rejected_support in rejected_layouts:
        with pytest.raises(ValueError, match="canonical shell 1, 2, or 3"):
            build_batch_a_screened_coulomb_hf_inputs(
                h0_ket=np.zeros((1, 1, 1), dtype=np.complex128),
                lattice=rejected_lattice,
                layout=rejected_layout,
                fold_map=_fold_map(),
                support=rejected_support,
                authority_capsule_root=_authority_root(),
                max_vertex_entries=1,
                provenance="strict noncanonical/shell6 rejection test",
            )


def test_t0_screened_coulomb_reference_and_nu_m3_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lattice = build_htqg_lattice(2.25, n_shells=0)
    layout = _layout(lattice.g_indices)
    fold_map = _fold_map()
    nb = layout.dimension
    h0 = np.zeros((fold_map.reduced_nk, nb, nb), dtype=np.complex128)
    for k in range(fold_map.reduced_nk):
        h0[k] = np.diag(np.arange(nb, dtype=float) + 1.0e-3 * k)

    support = _support(layout, fold_map)
    inputs = build_screened_coulomb_hf_inputs(
        h0_ket=h0,
        lattice=lattice,
        layout=layout,
        fold_map=fold_map,
        support=support,
        authority_capsule_root=_authority_root(),
        epsilon_r=5.0,
        d_sc_nm=25.0,
        primitive_filling=-3,
        max_vertex_entries=40_000,
        provenance="test replay of source-bound reciprocal HTQG HF convention",
    )
    assert inputs.screened_params.finite_zero_limit is True
    assert inputs.neutral_occupied == fold_map.reduced_nk * nb // 2
    assert inputs.total_occupied == inputs.neutral_occupied - 3 * fold_map.primitive_nk
    assert inputs.total_occupied == 416
    assert inputs.interaction_spec.hartree_reference_policy == "subtract_explicit"
    assert inputs.interaction_spec.fock_reference_policy == "absolute"
    assert inputs.interaction_spec.zero_mode_policy == "include_supplied"
    assert inputs.references.fock is None
    assert inputs.references.hartree is not None
    assert len(inputs.support_sha256) == 64
    support_token = f"support_sha256={inputs.support_sha256}"
    assert support.support_sha256 == inputs.support_sha256
    assert all(support_token in vertex.provenance for vertex in inputs.vertices)
    assert support_token in inputs.interaction_spec.normalization_provenance
    assert support_token in inputs.references.provenance
    assert not inputs.vertices[0].values.flags.writeable
    assert inputs.references.hartree is not None
    assert not inputs.references.hartree.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        inputs.vertices[0].values[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        inputs.references.hartree[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        inputs.vertices[0].values.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        inputs.references.hartree.setflags(write=True)
    with pytest.raises(TypeError, match="must be created by"):
        microscopic_hf_module.MicroscopicHFSourceBinding(
            support_sha256="0" * 64,
            h0_sha256="0" * 64,
            vertices_sha256="0" * 64,
            interaction_sha256="0" * 64,
            references_sha256="0" * 64,
            authority_sha256="0" * 64,
            nk=fold_map.reduced_nk,
            dimension=layout.dimension,
        )
    with pytest.raises(TypeError, match="does not permit subclasses"):
        class ForgedSourceBinding(
            microscopic_hf_module.MicroscopicHFSourceBinding
        ):
            pass
    np.testing.assert_allclose(
        np.trace(inputs.references.hartree, axis1=1, axis2=2),
        np.full(fold_map.reduced_nk, nb / 2),
        atol=1.0e-13,
        rtol=0.0,
    )
    assert inputs.supercell_area_nm2 == pytest.approx(
        fold_map.supercell.area_ratio * inputs.primitive_area_nm2
    )
    assert inputs.quadrature_normalization_nm_inv_sq == pytest.approx(
        1.0 / (inputs.primitive_area_nm2 * fold_map.primitive_nk)
    )
    gamma0 = next(vertex for vertex in inputs.vertices if vertex.label == (0, 0))
    expected_v0 = 2.0 * np.pi * 1.439964547 * 25.0 / 5.0
    assert gamma0.weight == pytest.approx(
        expected_v0 / (inputs.supercell_area_nm2 * fold_map.reduced_nk)
    )
    nonzero = next(vertex for vertex in inputs.vertices if vertex.label != (0, 0))
    q1, q2 = nonzero.label
    qvec = (q1 / 8.0) * lattice.b_m1 + (q2 / 4.0) * lattice.b_m2
    expected_vq = (
        2.0
        * np.pi
        * 1.439964547
        / (5.0 * abs(qvec))
        * np.tanh(abs(qvec) * 25.0)
    )
    assert nonzero.weight == pytest.approx(
        expected_vq / (inputs.supercell_area_nm2 * fold_map.reduced_nk)
    )

    forged_binding = object.__new__(
        microscopic_hf_module.MicroscopicHFSourceBinding
    )
    for name in (
        "support_sha256",
        "h0_sha256",
        "vertices_sha256",
        "interaction_sha256",
        "references_sha256",
        "authority_sha256",
        "derivation_authority",
        "occupation_ensemble",
        "kbt_ev",
        "k_weights_sha256",
        "nk",
        "dimension",
        "total_occupied",
    ):
        object.__setattr__(forged_binding, name, getattr(inputs.source_binding, name))
    with pytest.raises(TypeError, match="exact factory-built"):
        build_microscopic_hf_problem(
            vertices=inputs.vertices,
            interaction_spec=inputs.interaction_spec,
            references=inputs.references,
            nk=fold_map.reduced_nk,
            dimension=layout.dimension,
            total_occupied=inputs.total_occupied,
            fermi_degeneracy_tolerance=1.0e-12,
            mixing=0.25,
            source_binding=forged_binding,
        )

    with pytest.raises(ValueError, match="explicit typed source_binding"):
        build_microscopic_hf_problem(
            vertices=inputs.vertices,
            interaction_spec=inputs.interaction_spec,
            references=inputs.references,
            nk=fold_map.reduced_nk,
            dimension=layout.dimension,
            total_occupied=inputs.total_occupied,
            fermi_degeneracy_tolerance=1.0e-12,
            mixing=0.25,
        )
    assert inputs.source_binding.derivation_authority == "identity_only_unverified"
    assert (
        inputs.source_binding.occupation_ensemble
        == "zero_temperature_global_canonical"
    )
    assert inputs.source_binding.kbt_ev is None
    assert inputs.source_binding.k_weights_sha256 is None
    with pytest.raises(ValueError, match="identity-only bindings are test-only"):
        build_microscopic_hf_problem(
            vertices=inputs.vertices,
            interaction_spec=inputs.interaction_spec,
            references=inputs.references,
            nk=fold_map.reduced_nk,
            dimension=layout.dimension,
            total_occupied=inputs.total_occupied,
            fermi_degeneracy_tolerance=1.0e-12,
            mixing=0.25,
            support_sha256=inputs.support_sha256,
            source_binding=inputs.source_binding,
        )
    problem = build_microscopic_hf_problem(
        vertices=inputs.vertices,
        interaction_spec=inputs.interaction_spec,
        references=inputs.references,
        nk=fold_map.reduced_nk,
        dimension=layout.dimension,
        total_occupied=inputs.total_occupied,
        fermi_degeneracy_tolerance=1.0e-12,
        mixing=0.25,
        support_sha256=inputs.support_sha256,
        source_binding=inputs.source_binding,
        allow_identity_only_test_binding=True,
    )
    assert problem.support_sha256 == inputs.support_sha256
    assert problem.source_binding == inputs.source_binding

    bound_weights = np.ones(fold_map.reduced_nk, dtype=float)
    finite_vertices = (gamma0,)
    finite_binding = build_microscopic_hf_source_binding(
        h0_ket=h0,
        vertices=finite_vertices,
        interaction_spec=inputs.interaction_spec,
        references=inputs.references,
        support=support,
        authority_capsule_root=_authority_root(),
        total_occupied=inputs.total_occupied,
        kbt_ev=5.0e-4,
        k_weights=bound_weights,
    )
    assert finite_binding.occupation_ensemble == "finite_temperature_global_fermi"
    assert finite_binding.kbt_ev == 5.0e-4
    assert finite_binding.k_weights_sha256 is not None
    assert len(finite_binding.k_weights_sha256) == 64
    bound_weights[:] = 2.0
    runtime_weights = np.ones(fold_map.reduced_nk, dtype=float)
    finite_problem = build_microscopic_hf_problem(
        vertices=finite_vertices,
        interaction_spec=inputs.interaction_spec,
        references=inputs.references,
        nk=fold_map.reduced_nk,
        dimension=layout.dimension,
        total_occupied=inputs.total_occupied,
        fermi_degeneracy_tolerance=1.0e-12,
        mixing=0.25,
        support_sha256=inputs.support_sha256,
        source_binding=finite_binding,
        kbt_ev=5.0e-4,
        k_weights=runtime_weights,
        final_gate_tolerances=MicroscopicHFFinalGateTolerances(
            particle_number_abs=1.0e-10,
            density_spectrum_abs=1.0e-10,
            commutator_rms_ev=1.0e-8,
            fermi_map_rms=1.0e-7,
        ),
        finite_t_convergence=MicroscopicHFFiniteTConvergence(
            helmholtz_change_abs_ev=1.0e-10,
            required_consecutive_iterations=3,
        ),
        allow_identity_only_test_binding=True,
    )
    runtime_weights[:] = np.nextafter(1.0, 2.0)
    finite_state = MicroscopicHFState.from_ket_h0(
        h0,
        precision=1.0e-8,
        source_binding=finite_binding,
    )
    finite_execution = finite_problem.validate_before_execution(finite_state)
    assert finite_execution is not None
    finite_update = finite_execution.kernel.density_builder(
        ket_hamiltonian_to_repository_blocks(h0)
    )
    expected_finite = global_fermi_occupations(
        h0,
        target_particle_number=float(inputs.total_occupied),
        kbt_ev=5.0e-4,
        k_weights=np.ones(fold_map.reduced_nk),
    )
    np.testing.assert_allclose(
        repository_stored_to_ket_blocks(finite_update.density),
        expected_finite.density_ket,
        atol=1.0e-13,
        rtol=0.0,
    )
    assert finite_update.mu == pytest.approx(
        expected_finite.chemical_potential,
        abs=1.0e-13,
    )
    with pytest.raises(ValueError, match="exact unit k_weights"):
        build_microscopic_hf_source_binding(
            h0_ket=h0,
            vertices=finite_vertices,
            interaction_spec=inputs.interaction_spec,
            references=inputs.references,
            support=support,
            authority_capsule_root=_authority_root(),
            total_occupied=inputs.total_occupied,
            kbt_ev=5.0e-4,
            k_weights=np.full(fold_map.reduced_nk, 2.0),
        )
    with pytest.raises(ValueError, match="occupation ensemble"):
        build_microscopic_hf_problem(
            vertices=inputs.vertices,
            interaction_spec=inputs.interaction_spec,
            references=inputs.references,
            nk=fold_map.reduced_nk,
            dimension=layout.dimension,
            total_occupied=inputs.total_occupied,
            fermi_degeneracy_tolerance=1.0e-12,
            mixing=0.25,
            support_sha256=inputs.support_sha256,
            source_binding=inputs.source_binding,
            kbt_ev=5.0e-4,
            k_weights=np.ones(fold_map.reduced_nk),
            final_gate_tolerances=MicroscopicHFFinalGateTolerances(
                particle_number_abs=1.0e-10,
                density_spectrum_abs=1.0e-10,
                commutator_rms_ev=1.0e-8,
                fermi_map_rms=1.0e-7,
            ),
            finite_t_convergence=MicroscopicHFFiniteTConvergence(
                helmholtz_change_abs_ev=1.0e-10,
                required_consecutive_iterations=3,
            ),
            allow_identity_only_test_binding=True,
        )
    with pytest.raises(ValueError, match="kbt_ev"):
        build_microscopic_hf_problem(
            vertices=finite_vertices,
            interaction_spec=inputs.interaction_spec,
            references=inputs.references,
            nk=fold_map.reduced_nk,
            dimension=layout.dimension,
            total_occupied=inputs.total_occupied,
            fermi_degeneracy_tolerance=1.0e-12,
            mixing=0.25,
            support_sha256=inputs.support_sha256,
            source_binding=finite_binding,
            kbt_ev=4.0e-4,
            k_weights=np.ones(fold_map.reduced_nk),
            final_gate_tolerances=MicroscopicHFFinalGateTolerances(
                particle_number_abs=1.0e-10,
                density_spectrum_abs=1.0e-10,
                commutator_rms_ev=1.0e-8,
                fermi_map_rms=1.0e-7,
            ),
            finite_t_convergence=MicroscopicHFFiniteTConvergence(
                helmholtz_change_abs_ev=1.0e-10,
                required_consecutive_iterations=3,
            ),
            allow_identity_only_test_binding=True,
        )
    with pytest.raises(ValueError, match="k-weight digest"):
        build_microscopic_hf_problem(
            vertices=finite_vertices,
            interaction_spec=inputs.interaction_spec,
            references=inputs.references,
            nk=fold_map.reduced_nk,
            dimension=layout.dimension,
            total_occupied=inputs.total_occupied,
            fermi_degeneracy_tolerance=1.0e-12,
            mixing=0.25,
            support_sha256=inputs.support_sha256,
            source_binding=finite_binding,
            kbt_ev=5.0e-4,
            k_weights=np.full(fold_map.reduced_nk, 2.0),
            final_gate_tolerances=MicroscopicHFFinalGateTolerances(
                particle_number_abs=1.0e-10,
                density_spectrum_abs=1.0e-10,
                commutator_rms_ev=1.0e-8,
                fermi_map_rms=1.0e-7,
            ),
            finite_t_convergence=MicroscopicHFFiniteTConvergence(
                helmholtz_change_abs_ev=1.0e-10,
                required_consecutive_iterations=3,
            ),
            allow_identity_only_test_binding=True,
        )
    import copy
    import pickle

    assert copy.copy(inputs.source_binding) is inputs.source_binding
    assert copy.deepcopy(inputs.source_binding) is inputs.source_binding
    assert copy.copy(problem) is problem
    assert copy.deepcopy(problem) is problem
    with pytest.raises(TypeError, match="does not permit pickling"):
        pickle.dumps(inputs.source_binding)
    with pytest.raises(TypeError, match="does not permit pickling"):
        pickle.dumps(problem)

    import gc

    ephemeral_binding = build_microscopic_hf_source_binding(
        h0_ket=h0,
        vertices=inputs.vertices,
        interaction_spec=inputs.interaction_spec,
        references=inputs.references,
        support=support,
        authority_capsule_root=_authority_root(),
        total_occupied=inputs.total_occupied,
    )
    binding_registry_key = id(ephemeral_binding)
    assert binding_registry_key in (
        microscopic_hf_module._MICROSCOPIC_HF_SOURCE_BINDING_REGISTRY
    )
    del ephemeral_binding
    gc.collect()
    assert binding_registry_key not in (
        microscopic_hf_module._MICROSCOPIC_HF_SOURCE_BINDING_REGISTRY
    )

    ephemeral_problem = build_microscopic_hf_problem(
        vertices=inputs.vertices,
        interaction_spec=inputs.interaction_spec,
        references=inputs.references,
        nk=fold_map.reduced_nk,
        dimension=layout.dimension,
        total_occupied=inputs.total_occupied,
        fermi_degeneracy_tolerance=1.0e-12,
        mixing=0.25,
        source_binding=inputs.source_binding,
        allow_identity_only_test_binding=True,
    )
    problem_registry_key = id(ephemeral_problem)
    assert problem_registry_key in microscopic_hf_module._MICROSCOPIC_HF_PROBLEM_REGISTRY
    del ephemeral_problem
    gc.collect()
    assert problem_registry_key not in microscopic_hf_module._MICROSCOPIC_HF_PROBLEM_REGISTRY

    with pytest.raises(TypeError, match="must be created by"):
        microscopic_hf_module.MicroscopicHartreeFockProblem()
    with pytest.raises(TypeError, match="must be created by"):
        replace(problem)
    with pytest.raises(TypeError, match="does not permit subclasses"):
        class ForgedMicroscopicProblem(
            microscopic_hf_module.MicroscopicHartreeFockProblem
        ):
            pass
    unbound_state = MicroscopicHFState.from_ket_h0(h0, precision=1.0e-12)
    forged_problem = object.__new__(
        microscopic_hf_module.MicroscopicHartreeFockProblem
    )
    with pytest.raises(TypeError, match="exact and factory-built"):
        run_hartree_fock_problem(
            unbound_state,
            forged_problem,
            init_mode="h0_global",
            seed=0,
            max_iter=1,
        )
    mutated_problem = object.__new__(
        microscopic_hf_module.MicroscopicHartreeFockProblem
    )
    for name, value in vars(problem).items():
        object.__setattr__(mutated_problem, name, value)
    object.__setattr__(mutated_problem, "kernel", replace(problem.kernel))
    with pytest.raises(TypeError, match="exact and factory-built"):
        run_hartree_fock_problem(
            unbound_state,
            mutated_problem,
            init_mode="h0_global",
            seed=0,
            max_iter=1,
        )
    bound_preflight_state = MicroscopicHFState.from_ket_h0(
        h0,
        precision=1.0e-12,
        source_binding=inputs.source_binding,
    )
    original_kernel = problem.kernel
    object.__setattr__(problem, "kernel", replace(original_kernel))
    authority_checks: list[Path] = []
    original_authority_verifier = microscopic_hf_module.verify_reciprocal_hf_authority

    def record_authority_check(root: str | Path) -> None:
        authority_checks.append(Path(root))
        original_authority_verifier(root)

    with monkeypatch.context() as patch:
        patch.setattr(
            microscopic_hf_module,
            "verify_reciprocal_hf_authority",
            record_authority_check,
        )
        trusted_execution = problem.validate_before_execution(bound_preflight_state)
    assert authority_checks == [_authority_root().resolve()]
    assert trusted_execution.kernel is not original_kernel
    assert (
        trusted_execution.kernel.interaction_builder
        is original_kernel.interaction_builder
    )
    object.__setattr__(problem, "kernel", original_kernel)
    original_interaction_builder = original_kernel.interaction_builder
    object.__setattr__(
        original_kernel,
        "interaction_builder",
        lambda density: np.full_like(density, np.nan),
    )
    object.__setattr__(original_kernel, "convergence_rule", "mixed")
    replayed_execution = problem.validate_before_execution(bound_preflight_state)
    assert replayed_execution is not trusted_execution
    assert replayed_execution.kernel is not trusted_execution.kernel
    assert replayed_execution.kernel.interaction_builder is original_interaction_builder
    assert replayed_execution.kernel.convergence_rule == "raw"
    object.__setattr__(
        original_kernel, "interaction_builder", original_interaction_builder
    )
    object.__setattr__(original_kernel, "convergence_rule", "raw")

    # A caller can retain and mutate a returned lease; subsequent preflight
    # must still originate from the untouched canonical registry snapshot.
    object.__setattr__(
        trusted_execution.kernel,
        "interaction_builder",
        lambda density: np.full_like(density, np.nan),
    )
    object.__setattr__(trusted_execution.kernel, "convergence_rule", "mixed")
    third_execution = problem.validate_before_execution(bound_preflight_state)
    assert third_execution is not trusted_execution
    assert third_execution is not replayed_execution
    assert third_execution.kernel.interaction_builder is original_interaction_builder
    assert third_execution.kernel.convergence_rule == "raw"
    object.__setattr__(problem, "source_binding", None)
    with pytest.raises(TypeError, match="exact and factory-built"):
        problem.validate_before_execution(bound_preflight_state)
    object.__setattr__(problem, "source_binding", inputs.source_binding)

    binding_mutations = {
        "support_sha256": "0" * 64,
        "h0_sha256": "0" * 64,
        "vertices_sha256": "0" * 64,
        "interaction_sha256": "0" * 64,
        "references_sha256": "0" * 64,
        "authority_sha256": "0" * 64,
        "derivation_authority": "forged_authority",
        "occupation_ensemble": "finite_temperature_global_fermi",
        "kbt_ev": 5.0e-4,
        "k_weights_sha256": "0" * 64,
        "nk": inputs.source_binding.nk + 1,
        "dimension": inputs.source_binding.dimension + 1,
        "total_occupied": inputs.source_binding.total_occupied + 1,
    }
    for name, forged_value in binding_mutations.items():
        original_value = getattr(inputs.source_binding, name)
        object.__setattr__(inputs.source_binding, name, forged_value)
        with pytest.raises(TypeError, match="exact factory-built"):
            problem.validate_before_execution(bound_preflight_state)
        object.__setattr__(inputs.source_binding, name, original_value)
    with pytest.raises(ValueError, match="problem source lineage"):
        run_hartree_fock_problem(
            unbound_state,
            problem,
            init_mode="h0_global",
            seed=0,
            max_iter=1,
        )
    with pytest.raises(ValueError, match="state H0 digest"):
        MicroscopicHFState.from_ket_h0(
            h0 + np.eye(nb, dtype=np.complex128)[None, :, :] * 1.0e-8,
            precision=1.0e-12,
            source_binding=inputs.source_binding,
        )
    bound_state = MicroscopicHFState.from_ket_h0(
        h0, precision=1.0e-12, source_binding=inputs.source_binding
    )
    assert not bound_state.h0.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        bound_state.h0.setflags(write=True)
    with pytest.raises(AttributeError, match="h0 is immutable"):
        bound_state.h0 = bound_state.h0.copy()
    with pytest.raises(AttributeError, match="source_binding is immutable"):
        bound_state.source_binding = None
    tampered_gamma = CoreSparseDensityVertex(
        label=inputs.vertices[0].label,
        target_k=inputs.vertices[0].target_k,
        source_k=inputs.vertices[0].source_k,
        rows=inputs.vertices[0].rows,
        columns=inputs.vertices[0].columns,
        values=inputs.vertices[0].values,
        weight=inputs.vertices[0].weight + 1.0e-12,
        provenance=inputs.vertices[0].provenance,
    )
    with pytest.raises(ValueError, match="vertex digest"):
        build_microscopic_hf_problem(
            vertices=(tampered_gamma, *inputs.vertices[1:]),
            interaction_spec=inputs.interaction_spec,
            references=inputs.references,
            nk=fold_map.reduced_nk,
            dimension=layout.dimension,
            total_occupied=inputs.total_occupied,
            fermi_degeneracy_tolerance=1.0e-12,
            mixing=0.25,
            support_sha256=inputs.support_sha256,
            source_binding=inputs.source_binding,
        )
    with pytest.raises(ValueError, match="interaction digest"):
        build_microscopic_hf_problem(
            vertices=inputs.vertices,
            interaction_spec=replace(inputs.interaction_spec, include_fock=False),
            references=inputs.references,
            nk=fold_map.reduced_nk,
            dimension=layout.dimension,
            total_occupied=inputs.total_occupied,
            fermi_degeneracy_tolerance=1.0e-12,
            mixing=0.25,
            support_sha256=inputs.support_sha256,
            source_binding=inputs.source_binding,
        )
    with pytest.raises(ValueError, match="total_occupied"):
        build_microscopic_hf_problem(
            vertices=inputs.vertices,
            interaction_spec=inputs.interaction_spec,
            references=inputs.references,
            nk=fold_map.reduced_nk,
            dimension=layout.dimension,
            total_occupied=inputs.total_occupied - 1,
            fermi_degeneracy_tolerance=1.0e-12,
            mixing=0.25,
            support_sha256=inputs.support_sha256,
            source_binding=inputs.source_binding,
        )
    with pytest.raises(ValueError, match="reference digest"):
        build_microscopic_hf_problem(
            vertices=inputs.vertices,
            interaction_spec=inputs.interaction_spec,
            references=DensityVertexReferenceSpec(
                hartree=inputs.references.hartree,
                fock=None,
                provenance="test deliberately unbound reference",
            ),
            nk=fold_map.reduced_nk,
            dimension=layout.dimension,
            total_occupied=inputs.total_occupied,
            fermi_degeneracy_tolerance=1.0e-12,
            mixing=0.25,
            support_sha256=inputs.support_sha256,
            source_binding=inputs.source_binding,
        )

    primitive_vertices = build_primitive_plane_wave_density_vertices(
        layout,
        fold_map,
        _support(layout, fold_map),
        b_m1=lattice.b_m1,
        b_m2=lattice.b_m2,
        weight_for_transfer=lambda _label, qvec: (
            inputs.quadrature_normalization_nm_inv_sq
            * screened_coulomb(qvec, inputs.screened_params)
        ),
        provenance="lightweight shell0 primitive/supercell vertex replay",
        max_entries=40_000,
    )
    validate_primitive_supercell_density_vertex_equivalence(
        primitive_vertices,
        inputs.vertices,
        layout=layout,
        fold_map=fold_map,
        tolerance=0.0,
    )


def test_user_authorized_periodic_smoothstep_profile_has_two_c1_walls() -> None:
    profile = build_periodic_smoothstep_profile(
        wall_width_cells=1.0,
        nsample=64,
        provenance="test width parameter and periodic ordering",
    )
    assert not profile.sample_positions_cells.flags.writeable
    assert not profile.values_by_valley.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        profile.sample_positions_cells.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        profile.values_by_valley.setflags(write=True)
    values = profile.values_by_valley.real
    np.testing.assert_array_equal(values[0], values[1])
    assert values[0, 0] == pytest.approx(0.5)
    assert values[0, 16] == pytest.approx(0.0)
    assert values[0, 32] == pytest.approx(0.5)
    assert values[0, 48] == pytest.approx(1.0)
    assert np.min(values) == pytest.approx(0.0)
    assert np.max(values) == pytest.approx(1.0)
    assert "full wall_width_cells=1" in profile.provenance

    with pytest.raises(ValueError, match="N/2=2"):
        build_periodic_smoothstep_profile(
            wall_width_cells=2.0,
            nsample=64,
            provenance="invalid touching walls",
        )
    with pytest.raises(ValueError, match="even exact integer"):
        build_periodic_smoothstep_profile(
            wall_width_cells=1.0,
            nsample=63,
            provenance="invalid odd grid",
        )


def test_scheme_a_reference_uses_global_neutral_mu_and_variable_k_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    nk, nb = 4, 6
    offsets = np.asarray((0.0, 0.37, 7.11, 10.23))
    h0 = np.asarray(
        [np.diag(np.arange(nb, dtype=float) + offset) for offset in offsets],
        dtype=np.complex128,
    )
    result = build_global_neutral_h0_reference(
        h0,
        fermi_degeneracy_tolerance=1.0e-12,
    )
    ranks = np.trace(result.projector_ket, axis1=1, axis2=2).real
    assert result.occupied_count == nk * nb // 2
    assert float(np.sum(ranks)) == pytest.approx(nk * nb / 2)
    assert not np.allclose(ranks, nb / 2, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        result.projector_ket @ result.projector_ket,
        result.projector_ket,
        atol=1.0e-14,
        rtol=0.0,
    )
    assert result.fermi_gap > 0.0
    parallel = build_global_neutral_h0_reference(
        h0,
        fermi_degeneracy_tolerance=1.0e-12,
        eigensolver_workers=2,
    )
    assert np.array_equal(parallel.energies, result.energies)
    assert np.array_equal(parallel.projector_ket, result.projector_ket)
    assert parallel.chemical_potential == result.chemical_potential
    assert parallel.fermi_gap == result.fermi_gap

    degenerate = np.asarray((np.diag([-1.0, 0.0]), np.diag([0.0, 1.0])))
    with pytest.raises(ValueError, match="global Fermi boundary is degenerate"):
        build_global_neutral_h0_reference(
            degenerate,
            fermi_degeneracy_tolerance=1.0e-12,
        )


def test_t0_problem_rejects_invalid_reference_arrays_and_policies() -> None:
    vertices = (
        identity_density_vertex(
            nk=1,
            dimension=2,
            weight=0.0,
            provenance="test-only reference validation",
        ),
    )
    absolute = _absolute_interaction()
    subtract_hartree = replace(
        absolute,
        include_fock=False,
        hartree_reference_policy="subtract_explicit",
    )
    common = dict(
        vertices=vertices,
        nk=1,
        dimension=2,
        total_occupied=1,
        fermi_degeneracy_tolerance=1.0e-12,
        mixing=1.0,
        allow_unbound_test_problem=True,
    )
    with pytest.raises(ValueError, match="requires an explicit reference"):
        build_microscopic_hf_problem(
            interaction_spec=subtract_hartree,
            references=DensityVertexReferenceSpec(
                hartree=None,
                fock=None,
                provenance="test-only missing Hartree reference",
            ),
            **common,
        )
    with pytest.raises(ValueError, match="forbids an explicit reference"):
        build_microscopic_hf_problem(
            interaction_spec=absolute,
            references=DensityVertexReferenceSpec(
                hartree=np.zeros((1, 2, 2), dtype=np.complex128),
                fock=None,
                provenance="test-only forbidden absolute reference",
            ),
            **common,
        )
    invalid_references = (
        (
            np.zeros((1, 1, 1), dtype=np.complex128),
            "finite with shape",
        ),
        (
            np.asarray([[[np.nan, 0.0], [0.0, 0.0]]], dtype=np.complex128),
            "finite with shape",
        ),
        (
            np.asarray([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.complex128),
            "must be Hermitian",
        ),
    )
    for reference, message in invalid_references:
        with pytest.raises(ValueError, match=message):
            build_microscopic_hf_problem(
                interaction_spec=subtract_hartree,
                references=DensityVertexReferenceSpec(
                    hartree=reference,
                    fock=None,
                    provenance="test-only malformed Hartree reference",
                ),
                **common,
            )


def test_t0_problem_detaches_interaction_spec_from_caller_mutation() -> None:
    vertices = (
        identity_density_vertex(
            nk=1,
            dimension=2,
            weight=0.37,
            provenance="test-only interaction snapshot",
        ),
    )
    interaction_spec = _absolute_interaction()
    problem = build_microscopic_hf_problem(
        vertices=vertices,
        interaction_spec=interaction_spec,
        references=_absolute_references(),
        nk=1,
        dimension=2,
        total_occupied=1,
        fermi_degeneracy_tolerance=1.0e-12,
        mixing=1.0,
        allow_unbound_test_problem=True,
    )
    state = MicroscopicHFState.from_ket_h0(
        np.zeros((1, 2, 2), dtype=np.complex128), precision=1.0e-12
    )
    execution = problem.validate_before_execution(state)
    density = np.zeros((2, 2, 1), dtype=np.complex128)
    density[0, 0, 0] = 1.0
    before = execution.kernel.interaction_builder(density)
    object.__setattr__(interaction_spec, "include_fock", False)
    after = execution.kernel.interaction_builder(density)
    assert np.array_equal(after, before)


def test_t0_explicit_neutral_reference_requires_provenance() -> None:
    lattice = build_htqg_lattice(2.25, n_shells=0)
    layout = _layout(lattice.g_indices)
    fold_map = _fold_map()
    support = _support(layout, fold_map)
    nb = layout.dimension
    h0 = np.asarray(
        [np.diag(np.arange(nb, dtype=float) + 2.701 * k) for k in range(8)],
        dtype=np.complex128,
    )
    reference_result = build_global_neutral_h0_reference(
        h0,
        fermi_degeneracy_tolerance=1.0e-12,
    )
    reference = reference_result.projector_ket
    traces = np.trace(reference, axis1=1, axis2=2).real
    assert float(np.sum(traces)) == pytest.approx(fold_map.reduced_nk * nb / 2)
    assert not np.allclose(traces, nb / 2, atol=0.0, rtol=0.0)
    explicit = build_screened_coulomb_hf_inputs(
        h0_ket=h0,
        lattice=lattice,
        layout=layout,
        fold_map=fold_map,
        support=support,
        authority_capsule_root=_authority_root(),
        epsilon_r=5.0,
        d_sc_nm=25.0,
        primitive_filling=-3,
        max_vertex_entries=40_000,
        provenance="test explicit reference path",
        hartree_reference_ket=reference,
        reference_provenance="test exact neutral projector",
    )
    np.testing.assert_array_equal(explicit.references.hartree, reference)
    assert "test exact neutral projector" in explicit.references.provenance
    with pytest.raises(ValueError, match="reference provenance"):
        build_screened_coulomb_hf_inputs(
            h0_ket=h0,
            lattice=lattice,
            layout=layout,
            fold_map=fold_map,
            support=support,
            authority_capsule_root=_authority_root(),
            epsilon_r=5.0,
            d_sc_nm=25.0,
            primitive_filling=-3,
            max_vertex_entries=40_000,
            provenance="test missing explicit reference provenance",
            hartree_reference_ket=reference,
        )

    nonhermitian = reference.copy()
    nonhermitian[:, 0, 1] = 0.1j
    with pytest.raises(ValueError, match="Hermitian"):
        build_screened_coulomb_hf_inputs(
            h0_ket=h0,
            lattice=lattice,
            layout=layout,
            fold_map=fold_map,
            support=support,
            authority_capsule_root=_authority_root(),
            epsilon_r=5.0,
            d_sc_nm=25.0,
            primitive_filling=-3,
            max_vertex_entries=40_000,
            provenance="test non-Hermitian reference rejection",
            hartree_reference_ket=nonhermitian,
            reference_provenance="test non-Hermitian reference",
        )

    occupied_diagonal = np.argwhere(
        np.real(np.diagonal(reference, axis1=1, axis2=2)) > 0.5
    )
    empty_diagonal = np.argwhere(
        np.real(np.diagonal(reference, axis1=1, axis2=2)) < 0.5
    )
    occupied_k, occupied_band = (int(value) for value in occupied_diagonal[0])
    empty_k, empty_band = (int(value) for value in empty_diagonal[0])

    wrong_trace = reference.copy()
    wrong_trace[occupied_k, occupied_band, occupied_band] = 0.0
    with pytest.raises(ValueError, match="neutral trace"):
        build_screened_coulomb_hf_inputs(
            h0_ket=h0,
            lattice=lattice,
            layout=layout,
            fold_map=fold_map,
            support=support,
            authority_capsule_root=_authority_root(),
            epsilon_r=5.0,
            d_sc_nm=25.0,
            primitive_filling=-3,
            max_vertex_entries=40_000,
            provenance="test non-neutral reference rejection",
            hartree_reference_ket=wrong_trace,
            reference_provenance="test wrong-trace projector",
        )

    nonprojector = reference.copy()
    nonprojector[occupied_k, occupied_band, occupied_band] = 0.5
    nonprojector[empty_k, empty_band, empty_band] = 0.5
    with pytest.raises(ValueError, match="idempotent projector"):
        build_screened_coulomb_hf_inputs(
            h0_ket=h0,
            lattice=lattice,
            layout=layout,
            fold_map=fold_map,
            support=support,
            authority_capsule_root=_authority_root(),
            epsilon_r=5.0,
            d_sc_nm=25.0,
            primitive_filling=-3,
            max_vertex_entries=40_000,
            provenance="test nonprojector reference rejection",
            hartree_reference_ket=nonprojector,
            reference_provenance="test trace-correct nonprojector",
        )


def test_t0_contract_rejection_and_global_fermi_degeneracy() -> None:
    with pytest.raises(ValueError, match=r"primitive mesh \(8,4\)"):
        build_batch_a_fold_map(
            supercell=batch_a_supercell(),
            primitive_mesh=(4, 4),
            reduced_mesh=(2, 4),
        )
    with pytest.raises(ValueError, match="physical labels"):
        MicroscopicBasisLayout(
            g_indices=np.asarray([[0, 0]], dtype=np.int64),
            valley_order=(-1, 1),
            spin_order=("up", "down"),
            period_cells=4,
        )

    lattice = build_htqg_lattice(2.25, n_shells=0)
    unresolved = MicroscopicH0Spec(
        lattice=lattice,
        params=HTQGParams.default(),
        fold_map=_fold_map(),
        layout=_layout(lattice.g_indices),
        support=_support(_layout(lattice.g_indices)),
        profile_s_abg=_constant_profile(0.5),
        uniform_replay_tolerance=1.0e-11,
        evidence_paths=("tests/test_htqg_microscopic_supercell.py:T0",),
        unresolved_physical_choices=("wall profile has not been selected",),
    )
    with pytest.raises(ValueError, match="unresolved physical choices"):
        unresolved.validate(require_resolved=True)

    degenerate = np.zeros((2, 2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="Fermi boundary is degenerate"):
        global_canonical_occupations(
            degenerate,
            total_occupied=2,
            fermi_degeneracy_tolerance=0.0,
        )
    globally_filled = global_canonical_occupations(
        np.asarray((np.diag([-3.0, -2.0]), np.diag([1.0, 2.0]))),
        total_occupied=2,
        fermi_degeneracy_tolerance=1.0e-12,
    )
    np.testing.assert_allclose(
        np.trace(globally_filled.projector_ket, axis1=1, axis2=2),
        [2.0, 0.0],
    )


def test_t1_fold_basis_and_density_orientation_roundtrip() -> None:
    fold_map = _fold_map()
    seen = set()
    for r1 in range(2):
        for r2 in range(4):
            for fold in range(4):
                primitive = fold_map.primitive_coordinate((r1, r2), fold)
                assert fold_map.reduced_coordinate_and_fold(primitive) == ((r1, r2), fold)
                seen.add(primitive)
    assert len(seen) == 32

    layout = _layout(np.asarray([[0, 0], [1, -1]], dtype=np.int64))
    for coordinate in (
        (0, 0, 0, 0, 0, 0),
        (3, 1, 3, 1, 1, 1),
        (2, 0, 1, 1, 1, 0),
    ):
        assert layout.unravel_index(layout.flat_index(*coordinate)) == coordinate
    assert layout.legacy_seed_plane_wave_label(fold=3, g=1, valley=0) == (7, -4)
    assert layout.legacy_seed_plane_wave_label(fold=3, g=1, valley=1) == (-1, 4)

    ket = np.asarray(
        [[[0.7, 0.2 + 0.3j], [0.2 - 0.3j, 0.3]]],
        dtype=np.complex128,
    )
    stored = ket_blocks_to_repository_stored(ket)
    assert stored.shape == (2, 2, 1)
    assert stored[0, 1, 0] == ket[0, 1, 0]
    np.testing.assert_array_equal(repository_stored_to_ket_blocks(stored), ket)

    hamiltonian = np.asarray(
        (
            [[-1.0, 0.2j], [-0.2j, 1.0]],
            [[-0.5, 0.1 + 0.3j], [0.1 - 0.3j, 0.5]],
            [[0.0, -0.4j], [0.4j, 2.0]],
        ),
        dtype=np.complex128,
    )
    repository_h = ket_hamiltonian_to_repository_blocks(hamiltonian)
    assert repository_h.shape == (2, 2, 3)
    assert repository_h[0, 1, 0] == 0.2j
    np.testing.assert_array_equal(
        repository_hamiltonian_to_ket_blocks(repository_h), hamiltonian
    )


def test_t1_fold_map_is_immutable_and_rejects_noncanonical_arrays() -> None:
    with pytest.raises(TypeError, match="exact Python integers"):
        build_batch_a_fold_map(
            supercell=IntegerSupercell(4.0, 0, 0, 1),  # type: ignore[arg-type]
            primitive_mesh=(8, 4),
            reduced_mesh=(2, 4),
        )
    fold_map = _fold_map()
    assert not fold_map.reduced_fold_to_primitive.flags.writeable
    assert not fold_map.primitive_to_reduced.flags.writeable
    assert not fold_map.primitive_to_fold.flags.writeable
    for array in (
        fold_map.reduced_fold_to_primitive,
        fold_map.primitive_to_reduced,
        fold_map.primitive_to_fold,
    ):
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
    with pytest.raises(TypeError, match="exact Python integer"):
        fold_map.primitive_coordinate((0, 0), 0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact Python integer"):
        fold_map.reduced_coordinate_and_fold((0.0, 0))  # type: ignore[arg-type]

    class IntLike:
        def __int__(self) -> int:
            return 0

    for invalid in (np.int64(0), False, IntLike()):
        with pytest.raises(TypeError, match="exact Python integer"):
            fold_map.primitive_coordinate((invalid, 0), 0)  # type: ignore[arg-type]
    tampered = fold_map.reduced_fold_to_primitive.copy()
    tampered[0, 0, 0] = (2, 0)
    with pytest.raises(ValueError, match="not the canonical Batch-A fold map"):
        BatchAFoldMap(
            supercell=fold_map.supercell,
            primitive_mesh=fold_map.primitive_mesh,
            reduced_mesh=fold_map.reduced_mesh,
            reduced_fold_to_primitive=tampered,
            primitive_to_reduced=fold_map.primitive_to_reduced,
            primitive_to_fold=fold_map.primitive_to_fold,
        )
    forged = object.__new__(BatchAFoldMap)
    for name in (
        "supercell",
        "primitive_mesh",
        "reduced_mesh",
        "reduced_fold_to_primitive",
        "primitive_to_reduced",
        "primitive_to_fold",
    ):
        object.__setattr__(forged, name, getattr(fold_map, name))
    object.__setattr__(
        forged,
        "reduced_fold_to_primitive",
        np.frombuffer(tampered.tobytes(), dtype=np.int64).reshape(tampered.shape),
    )
    with pytest.raises(ValueError, match="not the canonical Batch-A fold map"):
        unfold_translation_invariant_blocks(
            np.zeros((2, 4, 4, 4), dtype=np.complex128),
            forged,
            offdiagonal_tolerance=0.0,
        )


def test_t1_support_is_translated_fixed_cardinality_and_exactly_tr_sewn() -> None:
    layout = _layout(np.asarray(((-1, 0), (0, 0), (1, 0)), dtype=np.int64))
    support = _support(layout)

    class LayoutSubclass(MicroscopicBasisLayout):
        pass

    subclass_layout = LayoutSubclass(
        g_indices=layout.g_indices,
        valley_order=layout.valley_order,
        spin_order=layout.spin_order,
        period_cells=layout.period_cells,
    )
    with pytest.raises(TypeError, match="exact MicroscopicBasisLayout"):
        build_batch_a_tr_covariant_support(
            subclass_layout,
            _fold_map(),
            provenance="test-only subclass rejection",
        )
    for array in (
        layout.g_indices,
        support.seed_g_indices,
        support.labels,
        support.tr_partner_k,
        support.tr_label_wrap,
        support.tr_pw_permutation,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
    with pytest.raises(TypeError, match="exact Python integer"):
        support.physical_plane_wave_label(
            reduced_k=0.0, valley=0, fold=0, g=0  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="exact Python integer"):
        layout.flat_index(0.0, 0, 0, 0, 0, 0)  # type: ignore[arg-type]

    class IntLike:
        def __int__(self) -> int:
            return 0

    for invalid in (np.int64(0), False, IntLike()):
        with pytest.raises(TypeError, match="exact Python integer"):
            support.physical_plane_wave_label(
                reduced_k=invalid, valley=0, fold=0, g=0  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match="exact Python integer"):
            layout.unravel_index(invalid)  # type: ignore[arg-type]
    mismatched_layout = _layout(
        np.asarray(((-2, 0), (0, 0), (2, 0)), dtype=np.int64)
    )
    with pytest.raises(ValueError, match="support seed does not match"):
        build_profile_fourier_multiplier(
            mismatched_layout,
            support,
            build_periodic_smoothstep_profile(
                wall_width_cells=1.0,
                nsample=64,
                provenance="test mismatched layout/support rejection",
            ),
            reduced_k=0,
        )
    seed = {tuple(int(value) for value in row) for row in layout.g_indices}
    np.testing.assert_array_equal(
        support.tr_partner_k,
        np.asarray((0, 3, 2, 1, 4, 7, 6, 5), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        support.tr_label_wrap,
        np.asarray(
            ((0, 0), (0, 4), (0, 4), (0, 4),
             (1, 0), (1, 4), (1, 4), (1, 4)),
            dtype=np.int64,
        ),
    )
    for reduced_k in range(8):
        partner_k = int(support.tr_partner_k[reduced_k])
        assert int(support.tr_partner_k[partner_k]) == reduced_k
        for valley in range(2):
            source_labels = support.labels[reduced_k, valley].reshape(-1, 2)
            target_labels = support.labels[partner_k, 1 - valley].reshape(-1, 2)
            permutation = support.tr_pw_permutation[reduced_k, valley]
            np.testing.assert_array_equal(
                target_labels[permutation],
                -source_labels - support.tr_label_wrap[reduced_k],
            )
            for fold in range(4):
                translation = support.parent_translation(
                    reduced_k=reduced_k,
                    valley=valley,
                    fold=fold,
                )
                expected = {
                    (g1 + translation[0], g2 + translation[1])
                    for g1, g2 in seed
                }
                observed = {
                    tuple(int(value) for value in row)
                    for row in support.parent_g_indices(
                        reduced_k=reduced_k,
                        valley=valley,
                        fold=fold,
                    )
                }
                assert observed == expected


def test_t1_support_rejects_malformed_fold_labels() -> None:
    layout = _layout()
    support = _support(layout)
    labels = support.labels.copy()
    labels[0, 0, 0, 0, 0] += 9
    with pytest.raises(ValueError, match="fold slots"):
        type(support)(
            seed_g_indices=support.seed_g_indices,
            labels=labels,
            tr_partner_k=support.tr_partner_k,
            tr_label_wrap=support.tr_label_wrap,
            tr_pw_permutation=support.tr_pw_permutation,
            provenance="test-only malformed support rejection",
        )


def test_t1_support_rejects_coherently_reordered_noncanonical_slots() -> None:
    layout = _layout(np.asarray(((-1, 0), (0, 0), (1, 0)), dtype=np.int64))
    support = _support(layout)
    reordered_labels = support.labels[:, :, :, (1, 0, 2), :].copy()
    ng = support.ng
    reordered_permutation = np.empty_like(support.tr_pw_permutation)
    for reduced_k in range(8):
        partner_k = int(support.tr_partner_k[reduced_k])
        for valley in range(2):
            lookup = {
                tuple(int(value) for value in label): slot
                for slot, label in enumerate(
                    reordered_labels[partner_k, 1 - valley].reshape(4 * ng, 2)
                )
            }
            for slot, label in enumerate(
                reordered_labels[reduced_k, valley].reshape(4 * ng, 2)
            ):
                target = tuple(
                    int(value)
                    for value in (-label - support.tr_label_wrap[reduced_k])
                )
                reordered_permutation[reduced_k, valley, slot] = lookup[target]
    with pytest.raises(ValueError, match="exact canonical Batch-A"):
        type(support)(
            seed_g_indices=support.seed_g_indices,
            labels=reordered_labels,
            tr_partner_k=support.tr_partner_k,
            tr_label_wrap=support.tr_label_wrap,
            tr_pw_permutation=reordered_permutation,
            provenance="test-only coherent noncanonical support rejection",
        )


def test_t1_support_rejects_coherently_reordered_reduced_k_slots() -> None:
    layout = _layout(np.asarray(((-1, 0), (0, 0), (1, 0)), dtype=np.int64))
    support = _support(layout)
    new_to_old = np.asarray((0, 2, 1, 3, 4, 5, 6, 7), dtype=np.int64)
    old_to_new = np.argsort(new_to_old)
    reordered_labels = support.labels[new_to_old].copy()
    reordered_wrap = support.tr_label_wrap[new_to_old].copy()
    reordered_partner = np.asarray(
        [old_to_new[support.tr_partner_k[old]] for old in new_to_old],
        dtype=np.int64,
    )
    reordered_permutation = support.tr_pw_permutation[new_to_old].copy()
    with pytest.raises(ValueError, match="exact canonical Batch-A"):
        type(support)(
            seed_g_indices=support.seed_g_indices,
            labels=reordered_labels,
            tr_partner_k=reordered_partner,
            tr_label_wrap=reordered_wrap,
            tr_pw_permutation=reordered_permutation,
            provenance="test-only coherent reduced-k reorder rejection",
        )


def test_t1_support_rejects_coherent_noncanonical_k_valley_labels() -> None:
    layout = _layout(np.asarray(((-1, 0), (0, 0), (1, 0)), dtype=np.int64))
    support = _support(layout)
    labels = support.labels.copy()
    labels[:, 0, :, :, 0] += 4
    labels[:, 1, :, :, 0] -= 4
    with pytest.raises(ValueError, match="exact canonical Batch-A"):
        type(support)(
            seed_g_indices=support.seed_g_indices,
            labels=labels,
            tr_partner_k=support.tr_partner_k,
            tr_label_wrap=support.tr_label_wrap,
            tr_pw_permutation=support.tr_pw_permutation,
            provenance="test-only coherent noncanonical valley-label rejection",
        )


def test_t1_shell0_h0_obeys_full_grid_time_reversal_sewing() -> None:
    lattice = build_htqg_lattice(2.25, n_shells=0)
    layout = _layout(lattice.g_indices)
    support = _support(layout)
    spec = MicroscopicH0Spec(
        lattice=lattice,
        params=HTQGParams.realistic(),
        fold_map=_fold_map(),
        layout=layout,
        support=support,
        profile_s_abg=_nonuniform_profile(nsample=32),
        uniform_replay_tolerance=2.0e-12,
        evidence_paths=("test-only full-grid matrix time-reversal sewing",),
        unresolved_physical_choices=(),
    )
    artifact = assemble_direct_h0_artifact(spec, verify_uniform_replay=True)
    h0 = artifact.h0_ket
    assert validate_microscopic_h0_artifact(artifact) is artifact
    assert artifact.support_sha256 == support.support_sha256
    with pytest.raises(TypeError, match="unexpected keyword argument 'h0_artifact'"):
        build_microscopic_hf_source_binding(
            h0_ket=h0,
            vertices=(
                identity_density_vertex(
                    nk=h0.shape[0],
                    dimension=h0.shape[1],
                    weight=0.0,
                    provenance="test public identity builder",
                ),
            ),
            interaction_spec=_absolute_interaction(),
            references=_absolute_references(),
            support=support,
            authority_capsule_root=_authority_root(),
            total_occupied=1,
            h0_artifact=artifact,
        )
    assert not artifact.h0_ket.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        artifact.h0_ket.setflags(write=True)
    import copy
    import pickle

    assert copy.copy(artifact) is artifact
    assert copy.deepcopy(artifact) is artifact
    with pytest.raises(TypeError, match="does not permit pickling"):
        pickle.dumps(artifact)
    forged_artifact = object.__new__(MicroscopicH0Artifact)
    with pytest.raises(TypeError, match="exact factory-built"):
        validate_microscopic_h0_artifact(forged_artifact)
    original_digest = artifact.h0_sha256
    object.__setattr__(artifact, "h0_sha256", "0" * 64)
    with pytest.raises(TypeError, match="exact factory-built"):
        validate_microscopic_h0_artifact(artifact)
    object.__setattr__(artifact, "h0_sha256", original_digest)
    ng = support.ng
    for reduced_k in range(8):
        target_k = int(support.tr_partner_k[reduced_k])
        target_of_source = np.empty(layout.dimension, dtype=np.int64)
        for source in range(layout.dimension):
            fold, g, layer, sublattice, valley, spin = layout.unravel_index(source)
            target_pw = int(
                support.tr_pw_permutation[reduced_k, valley, fold * ng + g]
            )
            target_fold, target_g = divmod(target_pw, ng)
            target_of_source[source] = layout.flat_index(
                target_fold,
                target_g,
                layer,
                sublattice,
                1 - valley,
                spin,
            )
        np.testing.assert_allclose(
            h0[target_k][np.ix_(target_of_source, target_of_source)],
            h0[reduced_k].conjugate(),
            atol=2.0e-12,
            rtol=0.0,
        )


def test_t1_multig_density_vertices_obey_q_to_minus_q_time_reversal_covariance() -> None:
    lattice = build_htqg_lattice(2.25, n_shells=0)
    layout = _layout(np.asarray(((-1, 0), (0, 0), (1, 0)), dtype=np.int64))
    fold_map = _fold_map()
    support = _support(layout)
    vertices = build_plane_wave_density_vertices(
        layout,
        fold_map,
        support,
        b_m1=lattice.b_m1,
        b_m2=lattice.b_m2,
        weight_for_transfer=lambda _label, qvec: 1.0 + abs(qvec) ** 2,
        provenance="test-only full-grid multi-g density-vertex TR covariance",
        max_entries=400_000,
    )
    by_label = {vertex.label: vertex for vertex in vertices}
    ng = support.ng

    def tr_basis_index(reduced_k: int, basis_index: int) -> tuple[int, int]:
        fold, g, layer, sublattice, valley, spin = layout.unravel_index(basis_index)
        target_pw = int(
            support.tr_pw_permutation[reduced_k, valley, fold * ng + g]
        )
        target_fold, target_g = divmod(target_pw, ng)
        return int(support.tr_partner_k[reduced_k]), layout.flat_index(
            target_fold,
            target_g,
            layer,
            sublattice,
            1 - valley,
            1 - spin,
        )

    for vertex in vertices:
        reverse = by_label[(-vertex.label[0], -vertex.label[1])]
        reverse_records = {
            (int(target_k), int(source_k), int(row), int(column), complex(value))
            for target_k, source_k, row, column, value in zip(
                reverse.target_k,
                reverse.source_k,
                reverse.rows,
                reverse.columns,
                reverse.values,
                strict=True,
            )
        }
        transformed_records = set()
        for target_k, source_k, row, column, value in zip(
            vertex.target_k,
            vertex.source_k,
            vertex.rows,
            vertex.columns,
            vertex.values,
            strict=True,
        ):
            tr_target_k, tr_row = tr_basis_index(int(target_k), int(row))
            tr_source_k, tr_column = tr_basis_index(int(source_k), int(column))
            transformed_records.add(
                (
                    tr_target_k,
                    tr_source_k,
                    tr_row,
                    tr_column,
                    complex(value).conjugate(),
                )
            )
        assert transformed_records == reverse_records
        assert reverse.weight == vertex.weight


def test_t2_profile_fourier_multiplier_uses_physical_labels() -> None:
    layout = _layout(np.asarray([[0, 0], [1, 0]], dtype=np.int64))
    support = _support(layout)
    identity = build_profile_fourier_multiplier(
        layout, support, _constant_profile(1.0, nsample=16), reduced_k=0
    )
    dense_identity = identity.to_dense_for_test(max_dimension=layout.dimension)
    np.testing.assert_allclose(dense_identity, np.eye(layout.dimension), atol=1.0e-15)
    for values in (identity.rows, identity.columns, identity.values):
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)

    nsample = 16
    x = np.arange(nsample, dtype=float) * 4.0 / nsample
    cosine = np.cos(2.0 * np.pi * x / 4.0)
    profile = ProfileFourierInput(
        sample_positions_cells=x,
        values_by_valley=np.asarray((cosine, cosine), dtype=np.complex128),
        period_cells=4,
        convention="numpy_fft_exp_minus_iqx",
        provenance="test-only first harmonic",
    )
    dense = build_profile_fourier_multiplier(
        layout, support, profile, reduced_k=0
    ).to_dense_for_test(max_dimension=layout.dimension)
    # n_out=0+4*1=4 and n_in=3+4*0=3: this is one physical
    # first-harmonic step even though both fold and g labels differ.
    row = layout.flat_index(0, 1, 0, 0, 0, 0)
    col = layout.flat_index(3, 0, 0, 0, 0, 0)
    assert abs(dense[row, col] - 0.5) < 1.0e-14
    np.testing.assert_allclose(dense, dense.conj().T, atol=1.0e-14)


def test_t2_once_projected_t34_keeps_external_intermediate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lattice = build_htqg_lattice(2.25, n_shells=0)
    layout = _layout(lattice.g_indices)
    support = _support(layout)
    params = HTQGParams.default(lambda_mdt_nm=0.0)
    nsample = 16
    sample_index = np.arange(nsample, dtype=float)
    k_profile = np.exp(-2.0j * np.pi * 4.0 * sample_index / nsample)
    profile = ProfileFourierInput(
        sample_positions_cells=sample_index * 4.0 / nsample,
        values_by_valley=np.asarray((k_profile, k_profile.conjugate())),
        period_cells=4,
        convention="numpy_fft_exp_minus_iqx",
        provenance="test-only external T34 intermediate harmonic",
    )

    product = microscopic_supercell_module._projected_profile_t34_product(
        layout=layout,
        support=support,
        profile=profile,
        channel=1,
        reduced_k_index=0,
        reduced_k=0.0j,
        lattice=lattice,
        params=params,
    )
    def forbidden_production_decoder(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("direct-DFT oracle called a production T34 decoder")

    monkeypatch.setattr(
        type(support), "parent_g_label", forbidden_production_decoder
    )
    monkeypatch.setattr(
        microscopic_supercell_module,
        "_one_way_t34_edge_block",
        forbidden_production_decoder,
    )
    monkeypatch.setattr(
        microscopic_supercell_module,
        "_T34_PARENT_INTEGER_SHIFTS",
        (),
    )
    oracle = microscopic_supercell_module._direct_dft_projected_profile_t34_product(
        layout=layout,
        support=support,
        profile=profile,
        channel=1,
        reduced_k_index=0,
        reduced_k=0.0j,
        lattice=lattice,
        params=params,
    )
    np.testing.assert_allclose(product, oracle, atol=2.0e-15, rtol=0.0)

    expected_k = moire_coupling_matrix(1, params)
    expected_k_prime = expected_k.conjugate()
    for valley, expected in ((0, expected_k), (1, expected_k_prime)):
        rows = [layout.flat_index(0, 0, 2, sub, valley, 0) for sub in range(2)]
        columns = [layout.flat_index(0, 0, 3, sub, valley, 0) for sub in range(2)]
        np.testing.assert_allclose(
            product[np.ix_(rows, columns)], expected, atol=2.0e-15, rtol=0.0
        )
    # Channel 1 maps parent g=(0,0) to the virtual g=(1,0), so P A_1 P is
    # exactly zero at shell zero.  A nonzero result proves the implementation
    # evaluates P M_1 A_1 P rather than (P M_1 P)(P A_1 P).
    assert np.max(np.abs(product)) > 0.05


def test_t2_t34_mdt_uses_source_momentum_and_time_reversed_kprime() -> None:
    lattice = build_htqg_lattice(2.25, n_shells=0)
    layout = _layout(lattice.g_indices)
    support = _support(layout)
    params = HTQGParams.realistic(
        lambda_mdt_nm=-0.23,
        include_dirac_rotation=True,
        mdt_momentum="source",
    )
    reduced_k = 0.17 * lattice.b_m1 + 0.11 * lattice.b_m2
    nsample = 16
    sample_index = np.arange(nsample, dtype=float)
    k_profile = np.exp(-2.0j * np.pi * 4.0 * sample_index / nsample)
    profile = ProfileFourierInput(
        sample_positions_cells=sample_index * 4.0 / nsample,
        values_by_valley=np.asarray((k_profile, k_profile.conjugate())),
        period_cells=4,
        convention="numpy_fft_exp_minus_iqx",
        provenance="test-only explicit MDT source-momentum check",
    )
    product = microscopic_supercell_module._projected_profile_t34_product(
        layout=layout,
        support=support,
        profile=profile,
        channel=1,
        reduced_k_index=0,
        reduced_k=complex(reduced_k),
        lattice=lattice,
        params=params,
    )

    qhat = complex(math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0))
    source_momentum_k = complex(reduced_k - 3.0 * lattice.q0)
    source_momentum_kprime_aux = complex(-reduced_k - 3.0 * lattice.q0)
    dot_k = qhat.real * source_momentum_k.real + qhat.imag * source_momentum_k.imag
    dot_kprime = (
        qhat.real * source_momentum_kprime_aux.real
        + qhat.imag * source_momentum_kprime_aux.imag
    )
    base = moire_coupling_matrix(1, params)
    expected = (
        base * (1.0 + params.lambda_mdt_nm * dot_k),
        (base * (1.0 + params.lambda_mdt_nm * dot_kprime)).conjugate(),
    )
    for valley in range(2):
        rows = [layout.flat_index(0, 0, 2, sub, valley, 0) for sub in range(2)]
        columns = [layout.flat_index(0, 0, 3, sub, valley, 0) for sub in range(2)]
        np.testing.assert_allclose(
            product[np.ix_(rows, columns)], expected[valley], atol=2.0e-15, rtol=0.0
        )


def test_t2_virtual_t34_intermediate_has_explicit_alias_boundary() -> None:
    lattice = build_htqg_lattice(2.25, n_shells=0)
    layout = _layout(lattice.g_indices)
    support = _support(layout)
    params = HTQGParams.default()

    def profile(nsample: int) -> ProfileFourierInput:
        sample_index = np.arange(nsample, dtype=float)
        values = np.exp(-2.0j * np.pi * 4.0 * sample_index / nsample)
        return ProfileFourierInput(
            sample_positions_cells=sample_index * 4.0 / nsample,
            values_by_valley=np.asarray((values, values.conjugate())),
            period_cells=4,
            convention="numpy_fft_exp_minus_iqx",
            provenance=f"test-only virtual-edge alias boundary N={nsample}",
        )

    with pytest.raises(ValueError, match="virtual T34 intermediate"):
        microscopic_supercell_module._projected_profile_t34_product(
            layout=layout,
            support=support,
            profile=profile(14),
            channel=1,
            reduced_k_index=0,
            reduced_k=0.0j,
            lattice=lattice,
            params=params,
        )
    product = microscopic_supercell_module._projected_profile_t34_product(
        layout=layout,
        support=support,
        profile=profile(15),
        channel=1,
        reduced_k_index=0,
        reduced_k=0.0j,
        lattice=lattice,
        params=params,
    )
    oracle = microscopic_supercell_module._direct_dft_projected_profile_t34_product(
        layout=layout,
        support=support,
        profile=profile(15),
        channel=1,
        reduced_k_index=0,
        reduced_k=0.0j,
        lattice=lattice,
        params=params,
    )
    np.testing.assert_allclose(product, oracle, atol=2.0e-15, rtol=0.0)


def test_t2_shell0_h0_uses_once_projected_t34_and_replays_endpoints() -> None:
    lattice = build_htqg_lattice(2.25, n_shells=0)
    layout = _layout(lattice.g_indices)
    spec = MicroscopicH0Spec(
        lattice=lattice,
        params=HTQGParams.realistic(),
        fold_map=_fold_map(),
        layout=layout,
        support=_support(layout),
        profile_s_abg=_nonuniform_profile(nsample=32),
        uniform_replay_tolerance=2.0e-12,
        evidence_paths=("test-only shell-zero once-projected T34",),
        unresolved_physical_choices=(),
    )
    residuals = uniform_h0_replay_residuals(spec)
    assert residuals["ABA"] <= spec.uniform_replay_tolerance
    assert residuals["ABG"] <= spec.uniform_replay_tolerance
    h0 = assemble_direct_h0(spec, verify_uniform_replay=True)
    oracle = assemble_direct_h0_dft_oracle(
        spec, ordering="left", max_dimension=256
    )
    wrong_order = assemble_direct_h0_dft_oracle(
        spec, ordering="right", max_dimension=256
    )
    np.testing.assert_allclose(h0, oracle, atol=2.0e-12, rtol=0.0)
    assert np.max(np.abs(h0 - wrong_order)) > 1.0e-7
    np.testing.assert_allclose(h0, h0.conj().transpose(0, 2, 1), atol=2.0e-12)


def test_t2_ordered_profile_product_is_noncommuting_and_valley_resolved() -> None:
    layout = _layout()
    nsample = 16
    x = np.arange(nsample, dtype=float) * 4.0 / nsample
    k_profile = np.exp(2.0j * np.pi * x / 4.0)
    profile = ProfileFourierInput(
        sample_positions_cells=x,
        values_by_valley=np.asarray((k_profile, k_profile.conjugate())),
        period_cells=4,
        convention="numpy_fft_exp_minus_iqx",
        provenance="test-only complex first harmonic and K-prime conjugate",
    )
    multiplier = build_profile_fourier_multiplier(
        layout, _support(layout), profile, reduced_k=0
    )
    dense_m = multiplier.to_dense_for_test(max_dimension=layout.dimension)
    a = np.zeros((layout.dimension, layout.dimension), dtype=np.complex128)
    a[
        layout.flat_index(0, 0, 2, 0, 0, 0),
        layout.flat_index(1, 0, 3, 1, 0, 0),
    ] = 1.0 + 0.4j
    a[
        layout.flat_index(0, 0, 2, 0, 1, 0),
        layout.flat_index(1, 0, 3, 1, 1, 0),
    ] = -0.3 + 0.7j
    ma = multiplier.left_multiply_dense(a)
    np.testing.assert_allclose(ma, dense_m @ a, atol=1.0e-14)
    ordered = ma + ma.conj().T
    wrong_order = a @ dense_m + (a @ dense_m).conj().T
    assert np.max(np.abs(ordered - wrong_order)) > 0.1
    np.testing.assert_allclose(ordered, ordered.conj().T, atol=1.0e-14)

@pytest.mark.slow
@pytest.mark.skipif(
    "SLURM_JOB_ID" not in os.environ,
    reason="shell-1 direct HTQG H0 replay is Slurm-only",
)
def test_t3_shell1_uniform_aba_abg_direct_h0_replay() -> None:
    """Slurm-only: direct ordered ``M A + A^dag M^dag`` endpoint replay."""

    lattice = build_htqg_lattice(2.25, n_shells=1)
    layout = _layout(lattice.g_indices)
    spec = MicroscopicH0Spec(
        lattice=lattice,
        params=HTQGParams.realistic(),
        fold_map=_fold_map(),
        layout=layout,
        support=_support(layout),
        profile_s_abg=_nonuniform_profile(nsample=32),
        uniform_replay_tolerance=2.0e-12,
        evidence_paths=(
            "results/HTQG_Fujimoto2025_hf/active2_aba_abg_boundary_eps5_nu_m3_v1/"
            "source_capsule_v13r3_full_parent_wall_spectrum_solver_qualification_"
            "mesh18_regular6430/wall_operator.py",
        ),
        unresolved_physical_choices=(),
    )
    residuals = uniform_h0_replay_residuals(spec)
    assert residuals["ABA"] <= spec.uniform_replay_tolerance
    assert residuals["ABG"] <= spec.uniform_replay_tolerance
    h0 = assemble_direct_h0(spec, verify_uniform_replay=True)
    oracle = assemble_direct_h0_dft_oracle(
        spec, ordering="left", max_dimension=896
    )
    wrong_order = assemble_direct_h0_dft_oracle(
        spec, ordering="right", max_dimension=896
    )
    np.testing.assert_allclose(h0, oracle, atol=2.0e-12, rtol=0.0)
    assert np.max(np.abs(h0 - wrong_order)) > 1.0e-7
    np.testing.assert_allclose(h0, h0.conj().transpose(0, 2, 1), atol=2.0e-12)


def test_t4_sparse_vertices_require_gamma0_and_reverse_closure() -> None:
    assert SparseDensityVertex is CoreSparseDensityVertex
    assert MicroscopicInteractionSpec is DensityVertexInteractionSpec
    assert InteractionReferences is DensityVertexReferenceSpec

    gamma0 = identity_density_vertex(
        nk=2,
        dimension=2,
        weight=0.4,
        provenance="test Gamma0 coefficient",
    )
    plus = SparseDensityVertex(
        label=(1, 0),
        target_k=np.asarray([1, 1, 0, 0]),
        source_k=np.asarray([0, 0, 1, 1]),
        rows=np.asarray([0, 1, 0, 1]),
        columns=np.asarray([0, 1, 0, 1]),
        values=np.asarray([1.0, 1.0j, 1.0, 1.0j]),
        weight=0.25,
        provenance="test K-permutation transfer",
    )
    minus = reverse_density_vertex(plus, provenance="literal adjoint of test transfer")
    vertices = (gamma0, plus, minus)
    validate_density_vertex_family(
        vertices,
        nk=2,
        dimension=2,
        closure_tolerance=1.0e-13,
        zero_mode_policy="include_supplied",
    )
    with pytest.raises(ValueError, match="reverse label"):
        validate_density_vertex_family(
            (gamma0, plus),
            nk=2,
            dimension=2,
            closure_tolerance=1.0e-13,
            zero_mode_policy="include_supplied",
        )
    with pytest.raises(TypeError, match="two exact integers"):
        SparseDensityVertex(
            label=(1.0, 0),
            target_k=np.asarray([0]),
            source_k=np.asarray([0]),
            rows=np.asarray([0]),
            columns=np.asarray([0]),
            values=np.asarray([1.0]),
            weight=0.1,
            provenance="invalid floating-point label",
        )


def test_t4_prepared_vertex_family_is_factory_only_and_alias_safe() -> None:
    gamma0 = identity_density_vertex(
        nk=1, dimension=2, weight=0.1, provenance="prepared-family test Gamma0"
    )
    prepared = prepare_density_vertex_family(
        (gamma0,),
        nk=1,
        dimension=2,
        closure_tolerance=1.0e-13,
        zero_mode_policy="include_supplied",
    )
    original_values = prepared.vertices[0].values.copy()
    gamma0.values[:] = 7.0
    np.testing.assert_array_equal(prepared.vertices[0].values, original_values)
    with pytest.raises(ValueError):
        prepared.vertices[0].values.setflags(write=True)
    with pytest.raises(TypeError, match="must be created"):
        PreparedDensityVertexFamily(
            vertices=(gamma0,),
            nk=1,
            dimension=2,
            closure_tolerance=1.0e-13,
            zero_mode_policy="include_supplied",
        )
    with pytest.raises(TypeError, match="does not permit subclasses"):
        class _PreparedSubclass(PreparedDensityVertexFamily):
            pass
    import copy
    import pickle

    assert copy.copy(prepared) is prepared
    assert copy.deepcopy(prepared) is prepared
    with pytest.raises(TypeError, match="does not permit pickling"):
        pickle.dumps(prepared)
    forged = object.__new__(PreparedDensityVertexFamily)
    with pytest.raises(TypeError, match="exact and factory-built"):
        core_hartree_fock_action(
            np.eye(2, dtype=np.complex128)[None, :, :],
            forged,
            _absolute_interaction(),
            _absolute_references(),
        )
    projector = np.eye(2, dtype=np.complex128)[None, :, :]
    baseline = core_hartree_fock_action(
        projector,
        prepared,
        _absolute_interaction(),
        _absolute_references(),
    )
    public_vertex = prepared.vertices[0]
    public_group = prepared.fock_groups_by_target[0][0]
    object.__setattr__(public_vertex, "weight", 91.0)
    object.__setattr__(
        public_vertex,
        "values",
        np.full(public_vertex.values.shape, 17.0, dtype=np.complex128),
    )
    object.__setattr__(public_group, "weight", 73.0)
    object.__setattr__(public_group, "entry_indices", np.asarray([1], dtype=np.int64))
    nested_replay = core_hartree_fock_action(
        projector,
        prepared,
        _absolute_interaction(),
        _absolute_references(),
    )
    np.testing.assert_array_equal(nested_replay.hartree, baseline.hartree)
    np.testing.assert_array_equal(nested_replay.fock, baseline.fock)
    np.testing.assert_array_equal(nested_replay.total, baseline.total)
    object.__setattr__(prepared, "vertices", ())
    object.__setattr__(prepared, "dimension", 99)
    replay = core_hartree_fock_action(
        projector,
        prepared,
        _absolute_interaction(),
        _absolute_references(),
    )
    np.testing.assert_array_equal(replay.hartree, baseline.hartree)
    np.testing.assert_array_equal(replay.fock, baseline.fock)
    np.testing.assert_array_equal(replay.total, baseline.total)
    with pytest.raises(TypeError, match="exact positive integer"):
        prepare_density_vertex_family(
            (gamma0,),
            nk=1.5,
            dimension=2,
            closure_tolerance=1.0e-13,
            zero_mode_policy="include_supplied",
        )


def test_t4_rejects_mixed_diagonal_and_offdiagonal_k_mappings() -> None:
    gamma0 = identity_density_vertex(
        nk=2, dimension=2, weight=0.1, provenance="test Gamma0"
    )
    mixed = SparseDensityVertex(
        label=(2, 0),
        target_k=np.asarray([0, 1]),
        source_k=np.asarray([0, 0]),
        rows=np.asarray([0, 0]),
        columns=np.asarray([0, 0]),
        values=np.ones(2),
        weight=0.1,
        provenance="invalid mixed K-diagonal test vertex",
    )
    mixed_reverse = reverse_density_vertex(mixed, provenance="invalid reverse")
    with pytest.raises(ValueError, match="cannot mix diagonal and off-diagonal"):
        validate_density_vertex_family(
            (gamma0, mixed, mixed_reverse),
            nk=2,
            dimension=2,
            closure_tolerance=1.0e-13,
            zero_mode_policy="include_supplied",
        )


def test_t4_physical_plane_wave_vertex_inventory_and_q_units() -> None:
    layout = _layout()
    observed: dict[tuple[int, int], complex] = {}

    def supplied_weight(label: tuple[int, int], qvec: complex) -> float:
        observed[label] = qvec
        return 0.0 if label == (0, 0) else 1.0 / (1.0 + abs(qvec))

    vertices = build_plane_wave_density_vertices(
        layout,
        _fold_map(),
        _support(layout),
        b_m1=8.0 + 0.0j,
        b_m2=4.0j,
        weight_for_transfer=supplied_weight,
        provenance="test-only explicit plane-wave transfer inventory",
        max_entries=40_000,
    )
    by_label = {vertex.label: vertex for vertex in vertices}
    assert observed[(1, 0)] == 1.0 + 0.0j
    assert observed[(0, 1)] == 0.0 + 1.0j
    validate_density_vertex_family(
        vertices,
        nk=8,
        dimension=layout.dimension,
        closure_tolerance=1.0e-13,
        zero_mode_policy="background_removed",
    )
    gamma0_entries = set(
        zip(
            by_label[(0, 0)].target_k.tolist(),
            by_label[(0, 0)].source_k.tolist(),
            by_label[(0, 0)].rows.tolist(),
            by_label[(0, 0)].columns.tolist(),
            strict=True,
        )
    )
    assert len(gamma0_entries) == 8 * layout.dimension
    assert all(
        target == source and row == column
        for target, source, row, column in gamma0_entries
    )

    victim = next(vertex for vertex in vertices if vertex.values.size > 1)
    incomplete = SparseDensityVertex(
        label=victim.label,
        target_k=victim.target_k[:-1],
        source_k=victim.source_k[:-1],
        rows=victim.rows[:-1],
        columns=victim.columns[:-1],
        values=victim.values[:-1],
        weight=victim.weight,
        provenance="test-only omitted retained matrix element",
    )
    incomplete_family = tuple(
        incomplete if vertex.label == victim.label else vertex
        for vertex in vertices
    )
    with pytest.raises(ValueError, match="inventory is incomplete"):
        validate_batch_a_plane_wave_density_vertices(
            incomplete_family, layout, _fold_map(), _support(layout)
        )


def test_t4_physical_vertex_validator_rejects_nonaffine_k_permutation() -> None:
    layout = _layout()
    source = np.arange(8, dtype=np.int64)
    expected_target = np.asarray([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int64)
    invalid = SparseDensityVertex(
        label=(1, 0),
        target_k=np.roll(expected_target, 1),
        source_k=source,
        rows=np.zeros(8, dtype=np.int64),
        columns=np.zeros(8, dtype=np.int64),
        values=np.ones(8, dtype=np.complex128),
        weight=0.2,
        provenance="test-only nonaffine K permutation",
    )
    with pytest.raises(ValueError, match="fixed reduced-K translation"):
        validate_batch_a_plane_wave_density_vertices(
            (invalid,), layout, _fold_map(), _support(layout)
        )


def test_t5_optimized_hartree_fock_matches_literal_four_index_oracle() -> None:
    projector = np.asarray(
        (
            [[0.65, 0.12 + 0.08j], [0.12 - 0.08j, 0.35]],
            [[0.22, -0.07 + 0.04j], [-0.07 - 0.04j, 0.58]],
        ),
        dtype=np.complex128,
    )
    vertices = _cross_k_vertex_family()
    spec = _absolute_interaction()
    references = _absolute_references()
    optimized = hartree_fock_action(projector, vertices, spec, references)
    literal = literal_wick_four_index_action(
        projector,
        vertices,
        spec,
        references,
        max_orbitals=4,
    )
    np.testing.assert_allclose(optimized.hartree, literal.hartree, atol=1.0e-14)
    np.testing.assert_allclose(optimized.fock, literal.fock, atol=1.0e-14)
    optimized_energy = hartree_fock_energy(
        projector, np.zeros_like(projector), vertices, spec, references
    )
    literal_energy = literal_wick_four_index_energy(
        projector, vertices, spec, references, max_orbitals=4
    )
    assert abs(optimized_energy - literal_energy) < 1.0e-14


def test_t5_prepared_grouped_fock_matches_legacy_and_target_parallel_paths() -> None:
    projector = np.asarray(
        (
            [[0.65, 0.12 + 0.08j], [0.12 - 0.08j, 0.35]],
            [[0.22, -0.07 + 0.04j], [-0.07 - 0.04j, 0.58]],
        ),
        dtype=np.complex128,
    )
    h0 = np.asarray(
        (
            [[0.2, 0.03j], [-0.03j, -0.1]],
            [[-0.07, 0.04 + 0.02j], [0.04 - 0.02j, 0.16]],
        ),
        dtype=np.complex128,
    )
    vertices = _cross_k_vertex_family()
    spec = _absolute_interaction()
    references = _absolute_references()
    prepared = prepare_density_vertex_family(
        vertices,
        nk=2,
        dimension=2,
        closure_tolerance=spec.closure_tolerance,
        zero_mode_policy=spec.zero_mode_policy,
    )
    striped = prepare_density_vertex_family(
        vertices,
        nk=2,
        dimension=2,
        closure_tolerance=spec.closure_tolerance,
        zero_mode_policy=spec.zero_mode_policy,
        fock_output_row_stripes=2,
    )

    legacy = core_hartree_fock_action(projector, vertices, spec, references)
    grouped = core_hartree_fock_action(
        projector, prepared, spec, references, target_workers=1
    )
    parallel = core_hartree_fock_action(
        projector, prepared, spec, references, target_workers=2
    )
    striped_serial = core_hartree_fock_action(
        projector,
        striped,
        spec,
        references,
        target_workers=1,
        fock_tile_rows=1,
    )
    striped_parallel = core_hartree_fock_action(
        projector,
        striped,
        spec,
        references,
        target_workers=4,
        fock_tile_rows=1,
    )
    for actual in (grouped, parallel, striped_serial, striped_parallel):
        np.testing.assert_array_equal(actual.hartree, legacy.hartree)
        np.testing.assert_array_equal(actual.fock, legacy.fock)
        np.testing.assert_array_equal(actual.total, legacy.total)

    fused = core_hartree_fock_action_and_energy(
        projector,
        h0,
        prepared,
        spec,
        references,
        target_workers=2,
    )
    np.testing.assert_array_equal(fused.action.hartree, legacy.hartree)
    np.testing.assert_array_equal(fused.action.fock, legacy.fock)
    assert fused.energy == hartree_fock_energy(
        projector, h0, vertices, spec, references
    )
    assert all(
        not group.entry_indices.flags.writeable
        and all(
            not indices.flags.writeable
            for indices in group.left_entry_indices_by_row_stripe
        )
        for groups in striped.fock_groups_by_target
        for group in groups
    )
    assert striped.fock_output_row_stripes == 2
    assert striped.fock_work_unit_count == 4
    total_entries = sum(vertex.values.size for vertex in prepared.vertices)
    assert prepared.storage_nbytes <= 64 * total_entries
    with pytest.raises(TypeError, match="target_workers"):
        core_hartree_fock_action(
            projector, prepared, spec, references, target_workers=True
        )
    for invalid_stripes in (True, 0, 3):
        with pytest.raises(TypeError, match="fock_output_row_stripes"):
            prepare_density_vertex_family(
                vertices,
                nk=2,
                dimension=2,
                closure_tolerance=spec.closure_tolerance,
                zero_mode_policy=spec.zero_mode_policy,
                fock_output_row_stripes=invalid_stripes,
            )


def test_t5_row_stripes_preserve_duplicate_row_accumulation_order() -> None:
    projector = np.asarray(
        [
            [
                [0.7, 0.03 + 0.02j, -0.04j],
                [0.03 - 0.02j, 0.4, 0.01],
                [0.04j, 0.01, 0.2],
            ]
        ],
        dtype=np.complex128,
    )
    gamma0 = identity_density_vertex(
        nk=1, dimension=3, weight=0.31, provenance="test duplicate-row Gamma0"
    )
    plus = SparseDensityVertex(
        label=(1, 0),
        target_k=np.zeros(3, dtype=np.int64),
        source_k=np.zeros(3, dtype=np.int64),
        rows=np.asarray([0, 0, 2]),
        columns=np.asarray([1, 2, 0]),
        values=np.asarray([0.7 + 0.2j, -0.3j, 0.4 - 0.1j]),
        weight=0.23,
        provenance="test duplicate-row Gamma+",
    )
    vertices = (
        gamma0,
        plus,
        reverse_density_vertex(plus, provenance="test duplicate-row Gamma-"),
    )
    spec = _absolute_interaction()
    references = _absolute_references()
    striped = prepare_density_vertex_family(
        vertices,
        nk=1,
        dimension=3,
        closure_tolerance=spec.closure_tolerance,
        zero_mode_policy=spec.zero_mode_policy,
        fock_output_row_stripes=2,
    )
    assert any(
        not group.rows_unique
        for groups in striped.fock_groups_by_target
        for group in groups
    )
    legacy = core_hartree_fock_action(projector, vertices, spec, references)
    for workers in (1, 2, 8):
        actual = core_hartree_fock_action(
            projector,
            striped,
            spec,
            references,
            target_workers=workers,
            fock_tile_rows=1,
        )
        np.testing.assert_array_equal(actual.hartree, legacy.hartree)
        np.testing.assert_array_equal(actual.fock, legacy.fock)
        np.testing.assert_array_equal(actual.total, legacy.total)


def test_t6_self_interaction_cancellation_and_two_spin_hubbard_limit() -> None:
    interaction_strength = 2.3
    vertices = (
        identity_density_vertex(
            nk=1,
            dimension=2,
            weight=interaction_strength,
            provenance="test onsite density-density coefficient U",
        ),
    )
    spec = _absolute_interaction()
    references = _absolute_references()

    orbital = np.asarray([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    one_particle = np.asarray([np.outer(orbital, orbital.conjugate())])
    one_energy = hartree_fock_energy(
        one_particle, np.zeros_like(one_particle), vertices, spec, references
    )
    assert abs(one_energy) < 1.0e-14

    opposite_spins = np.asarray([np.eye(2)], dtype=np.complex128)
    two_energy = hartree_fock_energy(
        opposite_spins, np.zeros_like(opposite_spins), vertices, spec, references
    )
    assert abs(two_energy - interaction_strength) < 1.0e-14


def test_disabled_terms_do_not_resolve_unused_references() -> None:
    projector = np.asarray([np.diag([0.7, 0.3])], dtype=np.complex128)
    vertices = (
        identity_density_vertex(
            nk=1, dimension=2, weight=0.2, provenance="test-only Gamma0"
        ),
    )
    spec = _absolute_interaction(include_hartree=False, include_fock=False)
    unused = np.asarray([[[np.nan]]])
    references = InteractionReferences(
        hartree=unused,
        fock=unused,
        provenance="intentionally invalid unused references",
    )

    action = hartree_fock_action(projector, vertices, spec, references)
    np.testing.assert_array_equal(action.total, np.zeros_like(projector))
    assert hartree_fock_energy(
        projector, np.zeros_like(projector), vertices, spec, references
    ) == 0.0
    literal = literal_wick_four_index_action(
        projector, vertices, spec, references, max_orbitals=2
    )
    np.testing.assert_array_equal(literal.total, np.zeros_like(projector))
    assert literal_wick_four_index_energy(
        projector, vertices, spec, references, max_orbitals=2
    ) == 0.0


def test_t7_energy_directional_derivative_matches_h0_plus_action() -> None:
    projector = np.asarray(
        (
            [[0.63, 0.11 + 0.03j], [0.11 - 0.03j, 0.27]],
            [[0.31, -0.06 + 0.02j], [-0.06 - 0.02j, 0.44]],
        ),
        dtype=np.complex128,
    )
    direction = np.asarray(
        (
            [[0.17, -0.04 + 0.09j], [-0.04 - 0.09j, -0.08]],
            [[-0.05, 0.07 - 0.03j], [0.07 + 0.03j, 0.11]],
        ),
        dtype=np.complex128,
    )
    h0 = np.asarray(
        (
            [[0.2, 0.03j], [-0.03j, -0.1]],
            [[-0.07, 0.04 + 0.02j], [0.04 - 0.02j, 0.16]],
        ),
        dtype=np.complex128,
    )
    reference = np.asarray(
        (np.diag([0.5, 0.5]), np.diag([0.45, 0.55])),
        dtype=np.complex128,
    )
    vertices = _cross_k_vertex_family()
    spec = MicroscopicInteractionSpec(
        include_hartree=True,
        include_fock=True,
        hartree_reference_policy="subtract_explicit",
        fock_reference_policy="absolute",
        zero_mode_policy="include_supplied",
        closure_tolerance=1.0e-13,
        normalization_provenance="test mixed-reference directional derivative",
        unresolved_physical_choices=(),
    )
    references = InteractionReferences(
        hartree=reference,
        fock=None,
        provenance="explicit Hartree reference; absolute Fock",
    )
    action = hartree_fock_action(projector, vertices, spec, references)
    analytic = np.einsum(
        "kab,kba->", h0 + action.total, direction, optimize=True
    ).real
    step = 2.0e-6
    upper = hartree_fock_energy(
        projector + step * direction, h0, vertices, spec, references
    )
    lower = hartree_fock_energy(
        projector - step * direction, h0, vertices, spec, references
    )
    finite_difference = (upper - lower) / (2.0 * step)
    assert abs(finite_difference - analytic) < 2.0e-10


def test_t8_primitive_supercell_interaction_and_replay_scaffold() -> None:
    fold_map = _fold_map()
    primitive_h0 = np.zeros((8, 4, 1, 1), dtype=np.complex128)
    primitive_projector = np.empty_like(primitive_h0)
    for i in range(8):
        for j in range(4):
            primitive_h0[i, j, 0, 0] = 0.01 * (i - j)
            primitive_projector[i, j, 0, 0] = 0.2 + 0.01 * (i + j)

    folded_h0 = fold_primitive_blocks(primitive_h0, fold_map)
    folded_projector = fold_primitive_blocks(primitive_projector, fold_map)
    primitive_blocks = primitive_projector.reshape(32, 1, 1)
    folded_blocks = folded_projector.reshape(8, 4, 4)
    spec = _absolute_interaction()
    references = _absolute_references()
    primitive_vertices = (
        identity_density_vertex(
            nk=32, dimension=1, weight=0.13, provenance="primitive test Gamma0"
        ),
    )
    folded_vertices = (
        identity_density_vertex(
            nk=8, dimension=4, weight=0.13, provenance="folded test Gamma0"
        ),
    )
    primitive_action = hartree_fock_action(
        primitive_blocks, primitive_vertices, spec, references
    ).total.reshape(8, 4, 1, 1)
    folded_action = hartree_fock_action(
        folded_blocks, folded_vertices, spec, references
    ).total.reshape(2, 4, 4, 4)

    scaffold = primitive_supercell_interaction_replay_scaffold(
        primitive_h0=primitive_h0,
        primitive_action=primitive_action,
        folded_h0=folded_h0,
        folded_action=folded_action,
        fold_map=fold_map,
        offdiagonal_tolerance=1.0e-13,
        evidence_paths=("tests/test_htqg_microscopic_supercell.py:T8",),
        uncertainty=(
            "identity-vertex algebra only; physical Coulomb Q inventory and "
            "normalization remain unresolved"
        ),
    )
    assert scaffold.h0_max_abs_residual == 0.0
    assert scaffold.action_max_abs_residual < 1.0e-14

    primitive_energy = hartree_fock_energy(
        primitive_blocks,
        primitive_h0.reshape(32, 1, 1),
        primitive_vertices,
        spec,
        references,
    )
    folded_energy = hartree_fock_energy(
        folded_blocks,
        folded_h0.reshape(8, 4, 4),
        folded_vertices,
        spec,
        references,
    )
    assert abs(primitive_energy - folded_energy) < 1.0e-13


def _fixed_rank_per_k_projector(h0_ket: np.ndarray, rank: int) -> np.ndarray:
    projector = np.zeros_like(h0_ket, dtype=np.complex128)
    for k, block in enumerate(h0_ket):
        _values, vectors = np.linalg.eigh(block)
        occupied = vectors[:, :rank]
        projector[k] = occupied @ occupied.conj().T
    return projector


@pytest.mark.slow
def test_scheme_a_shell1_wall_h0_neutral_reference_diagnostic() -> None:
    """Slurm-only first trial of the user-authorized Scheme-A wall reference."""

    if os.environ.get("SLURM_JOB_ID") is None:
        pytest.skip("physical shell-1 Scheme-A diagnostic is Slurm-only")
    if os.environ.get("HTQG_RUN_SCHEME_A_SHELL1") != "1":
        pytest.skip("set HTQG_RUN_SCHEME_A_SHELL1=1 inside a Slurm allocation")

    lattice = build_htqg_lattice(2.25, n_shells=1)
    params = HTQGParams.realistic()
    fold_map = _fold_map()
    layout = _layout(lattice.g_indices)
    support = _support(layout, fold_map)
    width_normal_over_lm = 0.5
    width_cells = (
        abs(lattice.b_m1)
        * width_normal_over_lm
        * lattice.l_m
        / (2.0 * np.pi)
    )

    h0_by_samples: dict[int, np.ndarray] = {}
    for nsample in (32, 64, 128, 256):
        profile = build_periodic_smoothstep_profile(
            wall_width_cells=width_cells,
            nsample=nsample,
            provenance=(
                "Scheme-A Test001 central-width diagnostic; "
                f"w_normal/L_M={width_normal_over_lm}"
            ),
        )
        h0_by_samples[nsample] = assemble_direct_h0(
            MicroscopicH0Spec(
                lattice=lattice,
                params=params,
                layout=layout,
                fold_map=fold_map,
                support=support,
                profile_s_abg=profile,
                uniform_replay_tolerance=2.0e-12,
                evidence_paths=(
                    "tests/test_htqg_microscopic_supercell.py:"
                    "test_scheme_a_shell1_wall_h0_neutral_reference_diagnostic",
                ),
                unresolved_physical_choices=(),
            ),
            verify_uniform_replay=True,
        )

    residual_32_64 = float(
        np.max(np.abs(h0_by_samples[32] - h0_by_samples[64]))
    )
    residual_64_128 = float(
        np.max(np.abs(h0_by_samples[64] - h0_by_samples[128]))
    )
    residual_128_256 = float(
        np.max(np.abs(h0_by_samples[128] - h0_by_samples[256]))
    )
    assert residual_64_128 < residual_32_64
    assert residual_128_256 < residual_64_128

    references = {
        nsample: build_global_neutral_h0_reference(
            h0_by_samples[nsample],
            fermi_degeneracy_tolerance=1.0e-10,
        )
        for nsample in (64, 128, 256)
    }
    reference = references[256]
    projector = reference.projector_ket
    ranks = np.trace(projector, axis1=1, axis2=2).real
    hermiticity_residual = float(
        np.max(np.abs(projector - projector.conj().transpose(0, 2, 1)))
    )
    idempotency_residual = float(np.max(np.abs(projector @ projector - projector)))
    total_trace = float(np.sum(ranks))
    assert total_trace == pytest.approx(3584.0, abs=1.0e-8)
    assert hermiticity_residual < 1.0e-10
    assert idempotency_residual < 1.0e-9

    projector_convergence: dict[str, dict[str, float]] = {}
    for coarse, fine in ((64, 128), (128, 256)):
        projector_coarse = references[coarse].projector_ket
        projector_fine = references[fine].projector_ket
        overlap = float(
            np.einsum(
                "kij,kji->",
                projector_coarse,
                projector_fine,
                optimize=True,
            ).real
        )
        projector_convergence[f"{coarse}_{fine}"] = {
            "occupied_overlap": overlap,
            "occupied_overlap_deficit": float(3584.0 - overlap),
            "relative_frobenius_difference": float(
                np.linalg.norm(projector_coarse - projector_fine)
                / np.sqrt(2.0 * 3584.0)
            ),
            "max_abs_difference": float(
                np.max(np.abs(projector_coarse - projector_fine))
            ),
        }

    print(
        "SCHEME_A_SHELL1_DIAGNOSTICS="
        + json.dumps(
            {
                "width_normal_over_lm": width_normal_over_lm,
                "wall_width_cells": width_cells,
                "profile_samples": [32, 64, 128, 256],
                "h0_max_abs_delta_32_64_ev": residual_32_64,
                "h0_max_abs_delta_64_128_ev": residual_64_128,
                "h0_max_abs_delta_128_256_ev": residual_128_256,
                "neutral_chemical_potential_ev": reference.chemical_potential,
                "neutral_fermi_gap_ev": reference.fermi_gap,
                "neutral_gap_by_samples_ev": {
                    str(nsample): references[nsample].fermi_gap
                    for nsample in (64, 128, 256)
                },
                "neutral_ranks_by_samples": {
                    str(nsample): np.trace(
                        references[nsample].projector_ket,
                        axis1=1,
                        axis2=2,
                    ).real.tolist()
                    for nsample in (64, 128, 256)
                },
                "neutral_total_trace": total_trace,
                "neutral_reduced_k_ranks": ranks.tolist(),
                "projector_hermiticity_max_abs": hermiticity_residual,
                "projector_idempotency_max_abs": idempotency_residual,
                "projector_convergence": projector_convergence,
            },
            sort_keys=True,
        )
    )


@pytest.mark.slow
def test_scheme_a_shell1_wall_width_scan_diagnostic() -> None:
    """Slurm-only Scheme-A sensitivity over the planned minimal width scan."""

    if os.environ.get("SLURM_JOB_ID") is None:
        pytest.skip("physical shell-1 Scheme-A width scan is Slurm-only")
    if os.environ.get("HTQG_RUN_SCHEME_A_WIDTH_SCAN") != "1":
        pytest.skip("set HTQG_RUN_SCHEME_A_WIDTH_SCAN=1 inside a Slurm allocation")

    lattice = build_htqg_lattice(2.25, n_shells=1)
    params = HTQGParams.realistic()
    fold_map = _fold_map()
    layout = _layout(lattice.g_indices)
    support = _support(layout, fold_map)
    diagnostics: dict[str, object] = {}

    for width_normal_over_lm in (0.35, 0.50, 0.75):
        width_cells = (
            abs(lattice.b_m1)
            * width_normal_over_lm
            * lattice.l_m
            / (2.0 * np.pi)
        )
        by_samples: dict[int, tuple[np.ndarray, object]] = {}
        for nsample in (256, 512, 1024):
            profile = build_periodic_smoothstep_profile(
                wall_width_cells=width_cells,
                nsample=nsample,
                provenance=(
                    "Scheme-A Test003 width/Fourier-convergence diagnostic; "
                    f"w_normal/L_M={width_normal_over_lm}"
                ),
            )
            h0 = assemble_direct_h0(
                MicroscopicH0Spec(
                    lattice=lattice,
                    params=params,
                    layout=layout,
                    fold_map=fold_map,
                    support=support,
                    profile_s_abg=profile,
                    uniform_replay_tolerance=2.0e-12,
                    evidence_paths=(
                        "tests/test_htqg_microscopic_supercell.py:"
                        "test_scheme_a_shell1_wall_width_scan_diagnostic",
                    ),
                    unresolved_physical_choices=(),
                ),
                verify_uniform_replay=True,
            )
            reference = build_global_neutral_h0_reference(
                h0,
                fermi_degeneracy_tolerance=1.0e-10,
            )
            by_samples[nsample] = (h0, reference)

        convergence: dict[str, object] = {}
        h0_deltas: list[float] = []
        for coarse, fine in ((256, 512), (512, 1024)):
            h0_coarse, reference_coarse = by_samples[coarse]
            h0_fine, reference_fine = by_samples[fine]
            projector_coarse = reference_coarse.projector_ket
            projector_fine = reference_fine.projector_ket
            overlap = float(
                np.einsum(
                    "kij,kji->",
                    projector_coarse,
                    projector_fine,
                    optimize=True,
                ).real
            )
            h0_delta = float(np.max(np.abs(h0_coarse - h0_fine)))
            h0_deltas.append(h0_delta)
            convergence[f"{coarse}_{fine}"] = {
                "h0_max_abs_delta_ev": h0_delta,
                "projector_overlap_deficit": float(3584.0 - overlap),
                "projector_relative_frobenius_difference": float(
                    np.linalg.norm(projector_coarse - projector_fine)
                    / np.sqrt(2.0 * 3584.0)
                ),
                "projector_max_abs_difference": float(
                    np.max(np.abs(projector_coarse - projector_fine))
                ),
            }
        assert h0_deltas[1] < h0_deltas[0]

        reference_fine = by_samples[1024][1]
        projector_fine = reference_fine.projector_ket
        ranks = np.trace(projector_fine, axis1=1, axis2=2).real
        total_trace = float(np.sum(ranks))
        assert total_trace == pytest.approx(3584.0, abs=1.0e-8)
        diagnostics[f"{width_normal_over_lm:.2f}"] = {
            "wall_width_cells": width_cells,
            "profile_samples": [256, 512, 1024],
            "neutral_gap_by_samples_ev": {
                str(nsample): by_samples[nsample][1].fermi_gap
                for nsample in (256, 512, 1024)
            },
            "neutral_chemical_potential_1024_ev": (
                reference_fine.chemical_potential
            ),
            "neutral_reduced_k_ranks_1024": ranks.tolist(),
            "convergence": convergence,
        }
        del by_samples

    print(
        "SCHEME_A_WIDTH_SCAN_DIAGNOSTICS="
        + json.dumps(diagnostics, sort_keys=True)
    )


@pytest.mark.slow
def test_scheme_a_uniform_endpoint_covariance_diagnostic() -> None:
    """Slurm-only global-neutral primitive/supercell covariance at ABA and ABG."""

    if os.environ.get("SLURM_JOB_ID") is None:
        pytest.skip("physical shell-1 Scheme-A endpoint diagnostic is Slurm-only")
    if os.environ.get("HTQG_RUN_SCHEME_A_ENDPOINTS") != "1":
        pytest.skip("set HTQG_RUN_SCHEME_A_ENDPOINTS=1 inside a Slurm allocation")

    lattice = build_htqg_lattice(2.25, n_shells=1)
    params = HTQGParams.realistic()
    fold_map = _fold_map()
    layout = _layout(lattice.g_indices)
    support = _support(layout, fold_map)
    primitive_dimension = layout.no_fold_dimension
    diagnostics: dict[str, object] = {}

    for endpoint, profile_value in (("ABA", 0.0), ("ABG", 1.0)):
        primitive_h0 = assemble_primitive_endpoint_h0(
            lattice=lattice,
            params=params,
            layout=layout,
            fold_map=fold_map,
            support=support,
            endpoint=endpoint,
        )
        direct_supercell_h0 = assemble_direct_h0(
            MicroscopicH0Spec(
                lattice=lattice,
                params=params,
                layout=layout,
                fold_map=fold_map,
                support=support,
                profile_s_abg=_constant_profile(profile_value, nsample=32),
                uniform_replay_tolerance=2.0e-12,
                evidence_paths=(
                    "tests/test_htqg_microscopic_supercell.py:"
                    "test_scheme_a_uniform_endpoint_covariance_diagnostic",
                ),
                unresolved_physical_choices=(),
            ),
            verify_uniform_replay=True,
        )
        folded_h0 = fold_primitive_blocks(
            primitive_h0.reshape(8, 4, primitive_dimension, primitive_dimension),
            fold_map,
        ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
        np.testing.assert_allclose(
            direct_supercell_h0,
            folded_h0,
            atol=2.0e-12,
            rtol=0.0,
        )

        primitive_global = build_global_neutral_h0_reference(
            primitive_h0,
            fermi_degeneracy_tolerance=1.0e-10,
        )
        folded_global = fold_primitive_blocks(
            primitive_global.projector_ket.reshape(
                8,
                4,
                primitive_dimension,
                primitive_dimension,
            ),
            fold_map,
        ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
        supercell_global = build_global_neutral_h0_reference(
            direct_supercell_h0,
            fermi_degeneracy_tolerance=1.0e-10,
        )
        covariance_max_abs = float(
            np.max(np.abs(supercell_global.projector_ket - folded_global))
        )
        covariance_relative_frobenius = float(
            np.linalg.norm(supercell_global.projector_ket - folded_global)
            / np.sqrt(2.0 * 3584.0)
        )
        assert covariance_max_abs < 1.0e-9

        historical_primitive = build_half_filled_h0_reference(primitive_h0)
        historical_folded = fold_primitive_blocks(
            historical_primitive.reshape(
                8,
                4,
                primitive_dimension,
                primitive_dimension,
            ),
            fold_map,
        ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
        historical_difference = supercell_global.projector_ket - historical_folded
        diagnostics[endpoint] = {
            "primitive_global_fermi_gap_ev": primitive_global.fermi_gap,
            "supercell_global_fermi_gap_ev": supercell_global.fermi_gap,
            "supercell_global_reduced_k_ranks": np.trace(
                supercell_global.projector_ket,
                axis1=1,
                axis2=2,
            ).real.tolist(),
            "global_primitive_supercell_covariance_max_abs": covariance_max_abs,
            "global_primitive_supercell_covariance_relative_frobenius": (
                covariance_relative_frobenius
            ),
            "scheme_a_vs_historical_fold_max_abs": float(
                np.max(np.abs(historical_difference))
            ),
            "scheme_a_vs_historical_fold_relative_frobenius": float(
                np.linalg.norm(historical_difference) / np.sqrt(2.0 * 3584.0)
            ),
        }

    print(
        "SCHEME_A_ENDPOINT_DIAGNOSTICS="
        + json.dumps(diagnostics, sort_keys=True)
    )


@pytest.mark.slow
def test_scheme_a_endpoint_hartree_counterterm_diagnostic() -> None:
    """Slurm-only size of the endpoint Hartree shift induced by Scheme A."""

    if os.environ.get("SLURM_JOB_ID") is None:
        pytest.skip("physical Scheme-A Hartree diagnostic is Slurm-only")
    if os.environ.get("HTQG_RUN_SCHEME_A_HARTREE_SHIFT") != "1":
        pytest.skip(
            "set HTQG_RUN_SCHEME_A_HARTREE_SHIFT=1 inside a Slurm allocation"
        )

    repository_root = Path(__file__).resolve().parents[1]
    lattice = build_htqg_lattice(2.25, n_shells=1)
    params = HTQGParams.realistic()
    fold_map = _fold_map()
    layout = _layout(lattice.g_indices)
    support = _support(layout, fold_map)
    primitive_dimension = layout.no_fold_dimension
    endpoint_references: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, MicroscopicH0Artifact]
    ] = {}

    for endpoint, profile_value in (("ABA", 0.0), ("ABG", 1.0)):
        primitive_h0 = assemble_primitive_endpoint_h0(
            lattice=lattice,
            params=params,
            layout=layout,
            fold_map=fold_map,
            support=support,
            endpoint=endpoint,
        )
        direct_h0_artifact = assemble_direct_h0_artifact(
            MicroscopicH0Spec(
                lattice=lattice,
                params=params,
                layout=layout,
                fold_map=fold_map,
                support=support,
                profile_s_abg=_constant_profile(profile_value, nsample=32),
                uniform_replay_tolerance=2.0e-12,
                evidence_paths=(
                    "tests/test_htqg_microscopic_supercell.py:"
                    "test_scheme_a_endpoint_hartree_counterterm_diagnostic",
                ),
                unresolved_physical_choices=(),
            ),
            verify_uniform_replay=True,
        )
        direct_supercell_h0 = direct_h0_artifact.h0_ket
        scheme_a = build_global_neutral_h0_reference(
            direct_supercell_h0,
            fermi_degeneracy_tolerance=1.0e-10,
        ).projector_ket
        historical_primitive = build_half_filled_h0_reference(primitive_h0)
        historical_folded = fold_primitive_blocks(
            historical_primitive.reshape(
                8,
                4,
                primitive_dimension,
                primitive_dimension,
            ),
            fold_map,
        ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
        endpoint_references[endpoint] = (
            direct_supercell_h0,
            scheme_a,
            historical_folded,
            direct_h0_artifact,
        )

    inputs = build_batch_a_screened_coulomb_hf_inputs(
        h0_ket=endpoint_references["ABA"][0],
        lattice=lattice,
        layout=layout,
        fold_map=fold_map,
        support=support,
        authority_capsule_root=AUTHORITY_FIXTURE,
        max_vertex_entries=2_000_000,
        provenance="Scheme-A endpoint Hartree-counterterm diagnostic",
        hartree_reference_ket=endpoint_references["ABA"][1],
        reference_provenance="Scheme-A global-neutral ABA H0 projector",
        h0_artifact=endpoint_references["ABA"][3],
        reference_fermi_degeneracy_tolerance=1.0e-10,
        reference_replay_tolerance=2.0e-12,
    )
    assert inputs.source_binding.derivation_authority == (
        "batch_a_direct_h0_artifact"
    )
    prepared = prepare_density_vertex_family(
        inputs.vertices,
        nk=fold_map.reduced_nk,
        dimension=layout.dimension,
        closure_tolerance=inputs.interaction_spec.closure_tolerance,
        zero_mode_policy=inputs.interaction_spec.zero_mode_policy,
    )
    hartree_only_spec = DensityVertexInteractionSpec(
        include_hartree=True,
        include_fock=False,
        hartree_reference_policy="subtract_explicit",
        fock_reference_policy="absolute",
        zero_mode_policy=inputs.interaction_spec.zero_mode_policy,
        closure_tolerance=inputs.interaction_spec.closure_tolerance,
        normalization_provenance=inputs.interaction_spec.normalization_provenance,
        unresolved_physical_choices=(),
    )

    diagnostics: dict[str, object] = {}
    for endpoint, (
        _h0,
        scheme_a,
        historical,
        _artifact,
    ) in endpoint_references.items():
        references = DensityVertexReferenceSpec(
            hartree=scheme_a,
            fock=None,
            provenance=f"Scheme-A global-neutral {endpoint} H0 projector",
        )
        self_action = core_hartree_fock_action(
            scheme_a,
            prepared,
            hartree_only_spec,
            references,
        ).hartree
        induced = core_hartree_fock_action(
            historical,
            prepared,
            hartree_only_spec,
            references,
        ).hartree
        self_max_abs = float(np.max(np.abs(self_action)))
        assert self_max_abs < 1.0e-12
        eigenvalues = np.linalg.eigvalsh(induced)
        diagnostics[endpoint] = {
            "scheme_a_self_hartree_max_abs_ev": self_max_abs,
            "historical_density_minus_scheme_a_reference_hartree_max_abs_ev": (
                float(np.max(np.abs(induced)))
            ),
            "historical_density_minus_scheme_a_reference_hartree_frobenius_rms_ev": (
                float(np.linalg.norm(induced) / np.sqrt(induced.size))
            ),
            "historical_density_minus_scheme_a_reference_hartree_eigenvalue_min_ev": (
                float(np.min(eigenvalues))
            ),
            "historical_density_minus_scheme_a_reference_hartree_eigenvalue_max_ev": (
                float(np.max(eigenvalues))
            ),
        }

    print(
        "SCHEME_A_HARTREE_COUNTERTERM_DIAGNOSTICS="
        + json.dumps(diagnostics, sort_keys=True)
    )


@pytest.mark.slow
def test_scheme_a_shell2_finite_temperature_strict_source_preflight() -> None:
    """Qualify repaired strict shell-2 finite-T source and one interaction action."""

    if os.environ.get("SLURM_JOB_ID") is None:
        pytest.skip("strict shell-2 finite-T qualification is Slurm-only")
    if os.environ.get("HTQG_RUN_SHELL2_STRICT_FINITE_T") != "1":
        pytest.skip(
            "set HTQG_RUN_SHELL2_STRICT_FINITE_T=1 in the sealed capsule"
        )

    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if allocated_cpus < 64:
        pytest.fail(
            "strict shell-2 finite-T qualification requires at least 64 allocated CPUs"
        )
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        if os.environ.get(variable) != "1":
            pytest.fail(f"{variable}=1 is required for nested-parallel safety")

    repository_root = Path(__file__).resolve().parents[1]
    lattice = build_htqg_lattice(2.25, n_shells=2)
    params = HTQGParams.realistic()
    fold_map = _fold_map()
    layout = _layout(lattice.g_indices)
    support = _support(layout, fold_map)
    width_normal_over_lm = 0.5
    width_cells = (
        abs(lattice.b_m1)
        * width_normal_over_lm
        * lattice.l_m
        / (2.0 * np.pi)
    )
    profile = build_periodic_smoothstep_profile(
        wall_width_cells=width_cells,
        nsample=1024,
        provenance=(
            "Scheme-A shell-2 repaired-lineage finite-T qualification; "
            "w_normal/L_M=0.5"
        ),
    )
    h0_artifact = assemble_direct_h0_artifact(
        MicroscopicH0Spec(
            lattice=lattice,
            params=params,
            layout=layout,
            fold_map=fold_map,
            support=support,
            profile_s_abg=profile,
            uniform_replay_tolerance=2.0e-12,
            evidence_paths=(
                "tests/test_htqg_microscopic_supercell.py:"
                "test_scheme_a_shell2_finite_temperature_strict_source_preflight",
            ),
            unresolved_physical_choices=(),
        ),
        verify_uniform_replay=True,
    )
    h0 = h0_artifact.h0_ket
    reference = build_global_neutral_h0_reference(
        h0,
        fermi_degeneracy_tolerance=1.0e-10,
        eigensolver_workers=8,
    )
    k_weights = np.ones(fold_map.reduced_nk, dtype=float)
    inputs = build_batch_a_screened_coulomb_hf_inputs(
        h0_ket=h0,
        lattice=lattice,
        layout=layout,
        fold_map=fold_map,
        support=support,
        authority_capsule_root=AUTHORITY_FIXTURE,
        max_vertex_entries=15_000_000,
        provenance="Batch-A strict shell-2 repaired-lineage finite-T qualification",
        hartree_reference_ket=reference.projector_ket,
        reference_provenance=(
            "Scheme-A one-global-mu zero-temperature wall-H0 neutral projector"
        ),
        h0_artifact=h0_artifact,
        reference_fermi_degeneracy_tolerance=1.0e-10,
        reference_replay_tolerance=2.0e-12,
        kbt_ev=5.0e-4,
        k_weights=k_weights,
    )
    assert inputs.neutral_occupied == 9728
    assert inputs.total_occupied == 9632
    assert inputs.source_binding.derivation_authority == (
        "batch_a_direct_h0_artifact"
    )
    assert inputs.source_binding.occupation_ensemble == (
        "finite_temperature_global_fermi"
    )
    assert inputs.source_binding.kbt_ev == 5.0e-4
    assert inputs.source_binding.k_weights_sha256 is not None

    state = MicroscopicHFState.from_ket_h0(
        h0,
        precision=1.0e-7,
        source_binding=inputs.source_binding,
    )
    problem = build_microscopic_hf_problem(
        vertices=inputs.vertices,
        interaction_spec=inputs.interaction_spec,
        references=inputs.references,
        nk=fold_map.reduced_nk,
        dimension=layout.dimension,
        total_occupied=inputs.total_occupied,
        fermi_degeneracy_tolerance=1.0e-10,
        mixing=0.1,
        support_sha256=inputs.support_sha256,
        source_binding=inputs.source_binding,
        kbt_ev=5.0e-4,
        k_weights=k_weights,
        final_gate_tolerances=MicroscopicHFFinalGateTolerances(
            particle_number_abs=1.0e-9,
            density_spectrum_abs=1.0e-9,
            commutator_rms_ev=1.0e-7,
            fermi_map_rms=1.0e-7,
        ),
        finite_t_convergence=MicroscopicHFFiniteTConvergence(
            helmholtz_change_abs_ev=3.2e-7,
            required_consecutive_iterations=3,
        ),
        interaction_workers=64,
        eigensolver_workers=8,
        fock_tile_rows=256,
        fock_output_row_stripes=8,
    )
    execution = problem.validate_before_execution(state)
    execution.initializer(state, init_mode="h0_global", seed=0)
    density_ket = repository_stored_to_ket_blocks(state.density)
    particle_number = float(
        np.einsum(
            "k,kii->",
            k_weights,
            density_ket,
            optimize=True,
        ).real
    )
    assert particle_number == pytest.approx(9632.0, abs=1.0e-9)

    prepared = prepare_density_vertex_family(
        inputs.vertices,
        nk=fold_map.reduced_nk,
        dimension=layout.dimension,
        closure_tolerance=inputs.interaction_spec.closure_tolerance,
        zero_mode_policy=inputs.interaction_spec.zero_mode_policy,
        fock_output_row_stripes=8,
    )
    prepared_storage_nbytes = prepared.storage_nbytes
    expected_action = core_hartree_fock_action(
        density_ket,
        prepared,
        inputs.interaction_spec,
        inputs.references,
    )
    expected_total = expected_action.total
    del expected_action, prepared
    strict_action = repository_hamiltonian_to_ket_blocks(
        execution.kernel.interaction_builder(state.density)
    )
    np.testing.assert_allclose(
        strict_action,
        expected_total,
        atol=2.0e-11,
        rtol=0.0,
    )
    interaction_action_max_abs = float(
        np.max(np.abs(strict_action), initial=0.0)
    )
    interaction_action_parity_max_abs = float(
        np.max(np.abs(strict_action - expected_total), initial=0.0)
    )
    del strict_action, expected_total, density_ket

    one_step_state = MicroscopicHFState.from_ket_h0(
        h0,
        precision=1.0e-7,
        source_binding=inputs.source_binding,
    )
    one_step_run = run_hartree_fock_problem(
        one_step_state,
        problem,
        init_mode="h0_global",
        seed=0,
        max_iter=1,
    )
    assert not one_step_run.converged
    assert one_step_run.exit_reason == "max_iter"
    assert one_step_run.iterations == 1
    assert np.all(np.isfinite(one_step_run.iter_energy))
    assert math.isfinite(
        one_step_run.state.diagnostics["finite_t_helmholtz_objective_ev"]
    )
    assert one_step_run.state.diagnostics["finite_t_particle_residual_abs"] <= 1.0e-9
    print(
        "SCHEME_A_SHELL2_STRICT_FINITE_T_DIAGNOSTICS="
        + json.dumps(
            {
                "dimension": layout.dimension,
                "n_g": lattice.n_g,
                "neutral_occupied": inputs.neutral_occupied,
                "total_occupied": inputs.total_occupied,
                "particle_number": particle_number,
                "kbt_ev": inputs.source_binding.kbt_ev,
                "k_weights_sha256": inputs.source_binding.k_weights_sha256,
                "h0_sha256": h0_artifact.h0_sha256,
                "support_sha256": inputs.support_sha256,
                "prepared_storage_nbytes": prepared_storage_nbytes,
                "interaction_action_max_abs_ev": interaction_action_max_abs,
                "interaction_action_parity_max_abs_ev": (
                    interaction_action_parity_max_abs
                ),
                "one_step_internal_energy_ev": float(one_step_run.iter_energy[0]),
                "one_step_helmholtz_ev": one_step_run.state.diagnostics[
                    "finite_t_helmholtz_objective_ev"
                ],
                "one_step_particle_residual_abs": one_step_run.state.diagnostics[
                    "finite_t_particle_residual_abs"
                ],
            },
            sort_keys=True,
        )
    )


@pytest.mark.slow
def test_t8_physical_shell1_primitive_supercell_replay_and_uniform_fixed_point() -> None:
    if os.environ.get("SLURM_JOB_ID") is None:
        pytest.skip("physical shell-1 T8 replay is Slurm-only")
    if os.environ.get("HTQG_RUN_PHYSICAL_T8") != "1":
        pytest.skip("set HTQG_RUN_PHYSICAL_T8=1 in the sealed Batch-A capsule")

    repository_root = Path(__file__).resolve().parents[1]
    lattice = build_htqg_lattice(2.25, n_shells=1)
    params = HTQGParams.realistic()
    fold_map = _fold_map()
    layout = _layout(lattice.g_indices)
    support = _support(layout, fold_map)
    primitive_dimension = layout.no_fold_dimension
    h0_primitive = assemble_primitive_endpoint_h0(
        lattice=lattice,
        params=params,
        layout=layout,
        fold_map=fold_map,
        support=support,
        endpoint="ABA",
    )
    h0_supercell = fold_primitive_blocks(
        h0_primitive.reshape(8, 4, primitive_dimension, primitive_dimension),
        fold_map,
    ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
    direct_h0_artifact = assemble_direct_h0_artifact(
        MicroscopicH0Spec(
            lattice=lattice,
            params=params,
            layout=layout,
            fold_map=fold_map,
            support=support,
            profile_s_abg=_constant_profile(0.0, nsample=32),
            uniform_replay_tolerance=2.0e-12,
            evidence_paths=(
                "tests/test_htqg_microscopic_supercell.py:T8 uniform ABA replay",
            ),
            unresolved_physical_choices=(),
        ),
        verify_uniform_replay=True,
    )
    direct_supercell = direct_h0_artifact.h0_ket
    np.testing.assert_allclose(direct_supercell, h0_supercell, atol=2.0e-12, rtol=0.0)

    primitive_reference = build_half_filled_h0_reference(h0_primitive)
    folded_reference = fold_primitive_blocks(
        primitive_reference.reshape(8, 4, primitive_dimension, primitive_dimension),
        fold_map,
    ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
    reduced_sector_reference = build_half_filled_h0_reference(direct_supercell)
    global_reference = build_global_neutral_h0_reference(
        direct_supercell,
        fermi_degeneracy_tolerance=1.0e-10,
    ).projector_ket
    # At a uniform endpoint the translation-invariant folded sectors happen to
    # carry identical half-filled ranks.  Therefore the historical fixed-rank
    # construction exactly replays the required global-neutral projector; the
    # global-mu policy must accept this equality rather than require a mismatch.
    np.testing.assert_allclose(
        reduced_sector_reference, folded_reference, atol=2.0e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        global_reference, folded_reference, atol=2.0e-12, rtol=0.0
    )
    super_inputs = build_batch_a_screened_coulomb_hf_inputs(
        h0_ket=direct_supercell,
        lattice=lattice,
        layout=layout,
        fold_map=fold_map,
        support=support,
        authority_capsule_root=AUTHORITY_FIXTURE,
        max_vertex_entries=2_000_000,
        provenance="Batch-A T8 physical shell1 primitive/supercell replay",
        hartree_reference_ket=folded_reference,
        reference_provenance=(
            "uniform-endpoint fixed-rank fold exactly replaying global neutrality"
        ),
        h0_artifact=direct_h0_artifact,
        reference_fermi_degeneracy_tolerance=1.0e-10,
        reference_replay_tolerance=2.0e-12,
    )
    assert super_inputs.source_binding.derivation_authority == (
        "batch_a_direct_h0_artifact"
    )
    assert super_inputs.references.hartree is not None
    np.testing.assert_allclose(
        super_inputs.references.hartree, folded_reference, atol=0.0, rtol=0.0
    )

    primitive_vertices = build_primitive_plane_wave_density_vertices(
        layout,
        fold_map,
        support,
        b_m1=lattice.b_m1,
        b_m2=lattice.b_m2,
        weight_for_transfer=lambda _label, qvec: (
            super_inputs.quadrature_normalization_nm_inv_sq
            * screened_coulomb(qvec, super_inputs.screened_params)
        ),
        provenance=(
            "Batch-A T8 primitive physical screened Coulomb; "
            f"support_sha256={super_inputs.support_sha256}"
        ),
        max_entries=2_000_000,
    )
    validate_primitive_supercell_density_vertex_equivalence(
        primitive_vertices,
        super_inputs.vertices,
        layout=layout,
        fold_map=fold_map,
        tolerance=0.0,
    )
    primitive_prepared = prepare_density_vertex_family(
        primitive_vertices,
        nk=fold_map.primitive_nk,
        dimension=primitive_dimension,
        closure_tolerance=super_inputs.interaction_spec.closure_tolerance,
        zero_mode_policy=super_inputs.interaction_spec.zero_mode_policy,
    )
    supercell_prepared = prepare_density_vertex_family(
        super_inputs.vertices,
        nk=fold_map.reduced_nk,
        dimension=layout.dimension,
        closure_tolerance=super_inputs.interaction_spec.closure_tolerance,
        zero_mode_policy=super_inputs.interaction_spec.zero_mode_policy,
    )
    primitive_references = DensityVertexReferenceSpec(
        hartree=primitive_reference,
        fock=None,
        provenance=(
            "same finite-cutoff lowest-half primitive H0 reference; "
            f"support_sha256={super_inputs.support_sha256}"
        ),
    )

    # Rank 109 at each primitive k is only the seed. Every update below uses
    # one mesh-wide chemical potential and global total count 3488.
    seed_primitive = _fixed_rank_per_k_projector(h0_primitive, rank=109)
    seed_supercell = fold_primitive_blocks(
        seed_primitive.reshape(8, 4, primitive_dimension, primitive_dimension),
        fold_map,
    ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
    primitive_action = hartree_fock_action(
        seed_primitive,
        primitive_prepared,
        super_inputs.interaction_spec,
        primitive_references,
    )
    supercell_action = hartree_fock_action(
        seed_supercell,
        supercell_prepared,
        super_inputs.interaction_spec,
        super_inputs.references,
    )
    folded_action = fold_primitive_blocks(
        primitive_action.total.reshape(8, 4, primitive_dimension, primitive_dimension),
        fold_map,
    ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
    np.testing.assert_allclose(supercell_action.total, folded_action, atol=2.0e-11, rtol=0.0)
    for primitive_part, supercell_part in (
        (primitive_action.hartree, supercell_action.hartree),
        (primitive_action.fock, supercell_action.fock),
    ):
        folded_part = fold_primitive_blocks(
            primitive_part.reshape(8, 4, primitive_dimension, primitive_dimension),
            fold_map,
        ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
        np.testing.assert_allclose(supercell_part, folded_part, atol=2.0e-11, rtol=0.0)

    # Exercise the strict source-bound supercell execution preflight and its
    # private action snapshot without any test-only binding escape hatch.
    strict_supercell_state = MicroscopicHFState.from_ket_h0(
        direct_supercell,
        precision=1.0e-8,
        source_binding=super_inputs.source_binding,
    )
    strict_supercell_state.density[:, :, :] = ket_blocks_to_repository_stored(
        seed_supercell
    )
    strict_supercell_problem = build_microscopic_hf_problem(
        vertices=super_inputs.vertices,
        interaction_spec=super_inputs.interaction_spec,
        references=super_inputs.references,
        nk=fold_map.reduced_nk,
        dimension=layout.dimension,
        total_occupied=super_inputs.total_occupied,
        fermi_degeneracy_tolerance=1.0e-10,
        mixing=0.25,
        support_sha256=super_inputs.support_sha256,
        source_binding=super_inputs.source_binding,
    )
    strict_execution = strict_supercell_problem.validate_before_execution(
        strict_supercell_state
    )
    strict_action_stored = strict_execution.kernel.interaction_builder(
        strict_supercell_state.density
    )
    np.testing.assert_allclose(
        repository_hamiltonian_to_ket_blocks(strict_action_stored),
        supercell_action.total,
        atol=2.0e-11,
        rtol=0.0,
    )

    primitive_energy = hartree_fock_energy(
        seed_primitive,
        h0_primitive,
        primitive_prepared,
        super_inputs.interaction_spec,
        primitive_references,
    )
    supercell_energy = hartree_fock_energy(
        seed_supercell,
        direct_supercell,
        supercell_prepared,
        super_inputs.interaction_spec,
        super_inputs.references,
    )
    assert abs(primitive_energy - supercell_energy) < 2.0e-9

    primitive_update = global_canonical_occupations(
        h0_primitive + primitive_action.total,
        total_occupied=super_inputs.total_occupied,
        fermi_degeneracy_tolerance=1.0e-10,
    )
    supercell_update = global_canonical_occupations(
        direct_supercell + supercell_action.total,
        total_occupied=super_inputs.total_occupied,
        fermi_degeneracy_tolerance=1.0e-10,
    )
    folded_update = fold_primitive_blocks(
        primitive_update.projector_ket.reshape(
            8, 4, primitive_dimension, primitive_dimension
        ),
        fold_map,
    ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
    np.testing.assert_allclose(
        supercell_update.projector_ket, folded_update, atol=2.0e-9, rtol=0.0
    )
    assert supercell_update.chemical_potential == pytest.approx(
        primitive_update.chemical_potential, abs=2.0e-11
    )

    primitive_binding = build_microscopic_hf_source_binding(
        h0_ket=h0_primitive,
        vertices=primitive_vertices,
        interaction_spec=super_inputs.interaction_spec,
        references=primitive_references,
        support=support,
        authority_capsule_root=_authority_root(),
        total_occupied=super_inputs.total_occupied,
    )
    primitive_state = MicroscopicHFState.from_ket_h0(
        h0_primitive,
        precision=1.0e-8,
        source_binding=primitive_binding,
    )
    primitive_state.density[:, :, :] = ket_blocks_to_repository_stored(seed_primitive)
    primitive_problem = build_microscopic_hf_problem(
        vertices=primitive_vertices,
        interaction_spec=super_inputs.interaction_spec,
        references=primitive_references,
        nk=fold_map.primitive_nk,
        dimension=primitive_dimension,
        total_occupied=super_inputs.total_occupied,
        fermi_degeneracy_tolerance=1.0e-10,
        mixing=0.25,
        support_sha256=super_inputs.support_sha256,
        source_binding=primitive_binding,
        allow_identity_only_test_binding=True,
    )
    primitive_run = run_hartree_fock_problem(
        primitive_state,
        primitive_problem,
        init_mode="preserve",
        seed=0,
        max_iter=320,
    )
    print(
        "T8_SCF_CONVERGENCE_DIAGNOSTICS="
        + json.dumps(
            {
                "converged": primitive_run.converged,
                "exit_reason": primitive_run.exit_reason,
                "iterations": primitive_run.iterations,
                "final_raw_norm": primitive_run.state.diagnostics.get(
                    "final_raw_norm"
                ),
                "selected_norm_tail": primitive_run.iter_err[-16:].tolist(),
                "oda_tail": primitive_run.iter_oda[-16:].tolist(),
            },
            sort_keys=True,
        )
    )
    assert primitive_run.converged, (
        primitive_run.exit_reason,
        primitive_run.state.diagnostics.get("final_raw_norm"),
    )
    fixed_primitive = repository_stored_to_ket_blocks(primitive_run.state.density)
    fixed_supercell = fold_primitive_blocks(
        fixed_primitive.reshape(8, 4, primitive_dimension, primitive_dimension),
        fold_map,
    ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
    fixed_primitive_action = hartree_fock_action(
        fixed_primitive,
        primitive_prepared,
        super_inputs.interaction_spec,
        primitive_references,
    )
    fixed_supercell_action = hartree_fock_action(
        fixed_supercell,
        supercell_prepared,
        super_inputs.interaction_spec,
        super_inputs.references,
    )
    for primitive_part, supercell_part in (
        (fixed_primitive_action.hartree, fixed_supercell_action.hartree),
        (fixed_primitive_action.fock, fixed_supercell_action.fock),
        (fixed_primitive_action.total, fixed_supercell_action.total),
    ):
        folded_part = fold_primitive_blocks(
            primitive_part.reshape(8, 4, primitive_dimension, primitive_dimension),
            fold_map,
        ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
        np.testing.assert_allclose(supercell_part, folded_part, atol=2.0e-10, rtol=0.0)

    fixed_primitive_update = global_canonical_occupations(
        h0_primitive + fixed_primitive_action.total,
        total_occupied=super_inputs.total_occupied,
        fermi_degeneracy_tolerance=1.0e-10,
    )
    fixed_supercell_update = global_canonical_occupations(
        direct_supercell + fixed_supercell_action.total,
        total_occupied=super_inputs.total_occupied,
        fermi_degeneracy_tolerance=1.0e-10,
    )
    folded_fixed_update = fold_primitive_blocks(
        fixed_primitive_update.projector_ket.reshape(
            8, 4, primitive_dimension, primitive_dimension
        ),
        fold_map,
    ).reshape(fold_map.reduced_nk, layout.dimension, layout.dimension)
    np.testing.assert_allclose(
        fixed_supercell_update.projector_ket,
        folded_fixed_update,
        atol=2.0e-9,
        rtol=0.0,
    )
    assert fixed_supercell_update.chemical_potential == pytest.approx(
        fixed_primitive_update.chemical_potential, abs=2.0e-10
    )
    assert fixed_supercell_update.fermi_gap == pytest.approx(
        fixed_primitive_update.fermi_gap, abs=2.0e-10
    )
    fixed_residual = np.linalg.norm(
        fixed_supercell_update.projector_ket - fixed_supercell
    ) / np.linalg.norm(fixed_supercell_update.projector_ket)
    assert fixed_residual <= 2.0e-8
    fixed_primitive_energy = hartree_fock_energy(
        fixed_primitive,
        h0_primitive,
        primitive_prepared,
        super_inputs.interaction_spec,
        primitive_references,
    )
    fixed_supercell_energy = hartree_fock_energy(
        fixed_supercell,
        direct_supercell,
        supercell_prepared,
        super_inputs.interaction_spec,
        super_inputs.references,
    )
    assert abs(fixed_primitive_energy - fixed_supercell_energy) < 2.0e-8

    # The production ensemble uses one mesh-wide chemical potential.  Per-k
    # ranks are diagnostics, not independently constrained occupations.
    counts = np.trace(fixed_primitive, axis1=1, axis2=2).real
    update_counts = np.trace(
        fixed_primitive_update.projector_ket, axis1=1, axis2=2
    ).real
    np.testing.assert_allclose(counts, update_counts, atol=2.0e-8, rtol=0.0)
    assert float(np.sum(update_counts)) == pytest.approx(
        float(super_inputs.total_occupied), abs=2.0e-8
    )
    integer_counts = np.rint(update_counts).astype(np.int64)
    np.testing.assert_allclose(update_counts, integer_counts, atol=2.0e-8, rtol=0.0)

    fixed_primitive_hamiltonian = h0_primitive + fixed_primitive_action.total
    local_eigenvalues = np.linalg.eigvalsh(fixed_primitive_hamiltonian)
    common_fixed_rank_gap = float(
        np.min(local_eigenvalues[:, 109]) - np.max(local_eigenvalues[:, 108])
    )
    global_eigenvalues = np.sort(local_eigenvalues.reshape(-1), kind="stable")
    global_gap = float(
        global_eigenvalues[super_inputs.total_occupied]
        - global_eigenvalues[super_inputs.total_occupied - 1]
    )
    assert fixed_primitive_update.fermi_gap == pytest.approx(global_gap, abs=2.0e-10)
    print(
        "T8_GLOBAL_MU_DIAGNOSTICS="
        + json.dumps(
            {
                "chemical_potential": fixed_primitive_update.chemical_potential,
                "global_fermi_gap": global_gap,
                "common_fixed_109_per_k_gap": common_fixed_rank_gap,
                "primitive_k_occupied_counts": integer_counts.tolist(),
                "total_occupied": int(np.sum(integer_counts)),
            },
            sort_keys=True,
        )
    )


def test_generic_hf_engine_adapter_preserves_hamiltonian_orientation_and_global_mu() -> None:
    h0 = np.asarray(
        (
            [[-3.0, 0.2j], [-0.2j, -2.0]],
            [[-1.0, 0.1 + 0.05j], [0.1 - 0.05j, 0.0]],
            [[1.0, -0.07j], [0.07j, 2.0]],
        ),
        dtype=np.complex128,
    )
    state = MicroscopicHFState.from_ket_h0(h0, precision=1.0e-12)
    vertices = (
        identity_density_vertex(
            nk=3,
            dimension=2,
            weight=0.02,
            provenance="test-only nonzero Gamma0 interaction",
        ),
    )
    spec = _absolute_interaction()
    references = _absolute_references()
    problem = build_microscopic_hf_problem(
        vertices=vertices,
        interaction_spec=spec,
        references=references,
        nk=3,
        dimension=2,
        total_occupied=2,
        fermi_degeneracy_tolerance=1.0e-12,
        mixing=0.35,
        allow_unbound_test_problem=True,
    )
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode="h0_global",
        seed=17,
        max_iter=2,
    )
    assert run.converged
    assert run.state.energies.shape == (2, 3)
    assert run.state.h0[0, 1, 0] == 0.2j
    density = repository_stored_to_ket_blocks(run.state.density)
    np.testing.assert_allclose(
        np.trace(density, axis1=1, axis2=2),
        [2.0, 0.0, 0.0],
        atol=1.0e-14,
    )
    action = hartree_fock_action(density, vertices, spec, references).total
    physical_hamiltonian = repository_hamiltonian_to_ket_blocks(
        run.state.hamiltonian
    )
    np.testing.assert_allclose(physical_hamiltonian, h0 + action, atol=1.0e-14)
    expected_occupation = global_canonical_occupations(
        physical_hamiltonian,
        total_occupied=2,
        fermi_degeneracy_tolerance=1.0e-12,
    )
    assert run.state.mu == pytest.approx(expected_occupation.chemical_potential)
    assert run.state.diagnostics["hf_energy"] == pytest.approx(
        hartree_fock_energy(density, h0, vertices, spec, references)
    )


def _finite_temperature_toy_problem(
    h0: np.ndarray,
    *,
    weights: np.ndarray,
    tolerances: MicroscopicHFFinalGateTolerances,
    kbt_ev: float = 0.2,
    convergence: MicroscopicHFFiniteTConvergence | None = None,
    interaction_workers: int = 1,
    eigensolver_workers: int = 1,
    fock_output_row_stripes: int = 1,
):
    vertices = (
        identity_density_vertex(
            nk=h0.shape[0],
            dimension=h0.shape[1],
            weight=0.0,
            provenance="test-only zero interaction",
        ),
    )
    return build_microscopic_hf_problem(
        vertices=vertices,
        interaction_spec=_absolute_interaction(),
        references=_absolute_references(),
        nk=h0.shape[0],
        dimension=h0.shape[1],
        total_occupied=h0.shape[0],
        fermi_degeneracy_tolerance=1.0e-12,
        mixing=1.0,
        kbt_ev=kbt_ev,
        k_weights=weights,
        final_gate_tolerances=tolerances,
        finite_t_convergence=(
            convergence
            if convergence is not None
            else MicroscopicHFFiniteTConvergence(
                helmholtz_change_abs_ev=1.0e-12,
                required_consecutive_iterations=1,
            )
        ),
        interaction_workers=interaction_workers,
        eigensolver_workers=eigensolver_workers,
        fock_output_row_stripes=fock_output_row_stripes,
        allow_unbound_test_problem=True,
    )


def test_generic_hf_adapter_finite_temperature_free_energy_and_final_gates() -> None:
    block = np.asarray(
        [[-4.0e-4, 1.5e-4j], [-1.5e-4j, 4.0e-4]],
        dtype=np.complex128,
    )
    shifted_block = block + np.diag([3.0e-5, 1.7e-4])
    h0 = np.stack((block, shifted_block))
    weights = np.asarray([0.5, 0.5])
    tolerances = MicroscopicHFFinalGateTolerances(
        particle_number_abs=1.0e-14,
        density_spectrum_abs=1.0e-12,
        commutator_rms_ev=1.0e-12,
        fermi_map_rms=1.0e-12,
    )
    problem = _finite_temperature_toy_problem(
        h0,
        weights=weights,
        tolerances=tolerances,
        kbt_ev=5.0e-4,
        convergence=MicroscopicHFFiniteTConvergence(
            helmholtz_change_abs_ev=1.0e-14,
            required_consecutive_iterations=3,
        ),
    )
    state = MicroscopicHFState.from_ket_h0(h0, precision=1.0e-12)
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode="h0_global",
        seed=0,
        max_iter=4,
    )

    expected = global_fermi_occupations(
        h0,
        target_particle_number=1.0,
        kbt_ev=5.0e-4,
        k_weights=weights,
    )
    density = repository_stored_to_ket_blocks(run.state.density)
    np.testing.assert_allclose(density, expected.density_ket, atol=1.0e-14)
    assert density[0, 0, 1].imag == pytest.approx(expected.density_ket[0, 0, 1].imag)
    assert run.converged
    assert run.iterations == 4
    assert run.state.diagnostics["finite_t_convergence_streak"] == 3.0
    assert run.state.diagnostics["final_accepted"] == 1.0
    assert run.state.diagnostics["finite_t_particle_number"] == pytest.approx(1.0)
    assert run.state.diagnostics["finite_t_particle_residual_abs"] <= 1.0e-14
    assert run.state.diagnostics["finite_t_commutator_rms_ev"] <= 1.0e-12
    assert run.state.diagnostics["finite_t_fermi_map_rms"] <= 1.0e-12

    expected_internal_energy = float(
        np.einsum("k,kij,kji->", weights, h0, density, optimize=True).real
    )
    assert run.iter_energy[0] == pytest.approx(expected_internal_energy)
    assert run.state.diagnostics["hf_energy"] == pytest.approx(
        expected_internal_energy
    )
    expected_free_energy = (
        expected_internal_energy - 5.0e-4 * expected.entropy_dimensionless
    )
    assert run.state.diagnostics["finite_t_helmholtz_objective_ev"] == pytest.approx(
        expected_free_energy
    )
    assert expected_free_energy != pytest.approx(expected_internal_energy)

    short_state = MicroscopicHFState.from_ket_h0(h0, precision=1.0e-12)
    short_run = run_hartree_fock_problem(
        short_state,
        problem,
        init_mode="h0_global",
        seed=1,
        max_iter=3,
    )
    assert not short_run.converged
    assert short_run.exit_reason == "max_iter"
    assert short_run.iterations == 3
    assert short_state.diagnostics["finite_t_convergence_streak"] == 2.0
    assert short_state.diagnostics["final_accepted"] == 0.0

    # The established checkpoint contract preserves the density trajectory but
    # deliberately restarts the finite-T Helmholtz history/streak. Optimized
    # backends must not silently carry closure state absent from checkpoints.
    resumed_state = MicroscopicHFState.from_ket_h0(h0, precision=1.0e-12)
    resumed_state.density[:, :, :] = short_state.density
    resumed = run_hartree_fock_problem(
        resumed_state,
        problem,
        init_mode="preserve",
        seed=1,
        max_iter=4,
    )
    assert resumed.converged
    assert resumed.iterations == 4
    assert resumed.state.diagnostics["finite_t_convergence_streak"] == 3.0
    np.testing.assert_array_equal(resumed.state.density, run.state.density)


def test_microscopic_problem_caches_static_k_major_h0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h0 = np.asarray(
        (
            [[-0.4, 0.02j], [-0.02j, 0.3]],
            [[-0.2, 0.01], [0.01, 0.5]],
        ),
        dtype=np.complex128,
    )
    state = MicroscopicHFState.from_ket_h0(h0, precision=1.0e-12)
    tolerances = MicroscopicHFFinalGateTolerances(
        particle_number_abs=1.0e-12,
        density_spectrum_abs=1.0e-12,
        commutator_rms_ev=1.0e-12,
        fermi_map_rms=1.0e-12,
    )
    problem = _finite_temperature_toy_problem(
        h0,
        weights=np.ones(2),
        tolerances=tolerances,
        convergence=MicroscopicHFFiniteTConvergence(
            helmholtz_change_abs_ev=1.0e-12,
            required_consecutive_iterations=2,
        ),
    )
    original = microscopic_hf_module.repository_hamiltonian_to_ket_blocks
    h0_conversion_count = 0

    def counted_conversion(values: np.ndarray) -> np.ndarray:
        nonlocal h0_conversion_count
        if values is state.h0:
            h0_conversion_count += 1
        return original(values)

    monkeypatch.setattr(
        microscopic_hf_module,
        "repository_hamiltonian_to_ket_blocks",
        counted_conversion,
    )
    run_hartree_fock_problem(
        state,
        problem,
        init_mode="h0_global",
        seed=4,
        max_iter=4,
    )
    assert h0_conversion_count == 1


def test_finite_temperature_parallel_problem_trajectory_matches_serial_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    h0 = np.asarray(
        (
            [[-0.3, 0.02j], [-0.02j, 0.2]],
            [[-0.1, 0.03 + 0.01j], [0.03 - 0.01j, 0.4]],
            [[0.05, -0.02j], [0.02j, 0.7]],
        ),
        dtype=np.complex128,
    )
    weights = np.ones(3)
    tolerances = MicroscopicHFFinalGateTolerances(
        particle_number_abs=1.0e-12,
        density_spectrum_abs=1.0e-12,
        commutator_rms_ev=1.0e-12,
        fermi_map_rms=1.0e-12,
    )
    convergence = MicroscopicHFFiniteTConvergence(
        helmholtz_change_abs_ev=1.0e-12,
        required_consecutive_iterations=2,
    )
    serial_problem = _finite_temperature_toy_problem(
        h0,
        weights=weights,
        tolerances=tolerances,
        convergence=convergence,
    )
    parallel_problem = _finite_temperature_toy_problem(
        h0,
        weights=weights,
        tolerances=tolerances,
        convergence=convergence,
        interaction_workers=6,
        eigensolver_workers=3,
        fock_output_row_stripes=2,
    )
    serial = run_hartree_fock_problem(
        MicroscopicHFState.from_ket_h0(h0, precision=1.0e-12),
        serial_problem,
        init_mode="h0_global",
        seed=9,
        max_iter=4,
    )
    parallel = run_hartree_fock_problem(
        MicroscopicHFState.from_ket_h0(h0, precision=1.0e-12),
        parallel_problem,
        init_mode="h0_global",
        seed=9,
        max_iter=4,
    )
    assert parallel.converged == serial.converged
    assert parallel.iterations == serial.iterations
    np.testing.assert_array_equal(parallel.state.density, serial.state.density)
    np.testing.assert_array_equal(parallel.state.hamiltonian, serial.state.hamiltonian)
    np.testing.assert_array_equal(parallel.state.energies, serial.state.energies)
    np.testing.assert_array_equal(parallel.iter_energy, serial.iter_energy)
    np.testing.assert_array_equal(parallel.iter_err, serial.iter_err)
    assert parallel.state.mu == serial.state.mu
    assert parallel.state.diagnostics == serial.state.diagnostics


def test_generic_hf_adapter_scales_interacting_internal_energy_with_k_weights() -> None:
    h0 = np.asarray((np.diag([-0.7, 0.9]),), dtype=np.complex128)
    density = np.asarray(([[0.8, 0.1j], [-0.1j, 0.2]],), dtype=np.complex128)
    vertices = (
        identity_density_vertex(
            nk=1,
            dimension=2,
            weight=0.3,
            provenance="test-only nonzero interaction",
        ),
    )
    spec = MicroscopicInteractionSpec(
        include_hartree=True,
        include_fock=True,
        hartree_reference_policy="subtract_explicit",
        fock_reference_policy="subtract_explicit",
        zero_mode_policy="include_supplied",
        closure_tolerance=1.0e-13,
        normalization_provenance="test-only supplied scalar vertex weights",
        unresolved_physical_choices=(),
    )
    reference_density = np.asarray((0.1 * np.eye(2),), dtype=np.complex128)
    references = InteractionReferences(
        hartree=reference_density.copy(),
        fock=reference_density.copy(),
        provenance="test-only mutable reference copied by adapter",
    )
    tolerances = MicroscopicHFFinalGateTolerances(
        particle_number_abs=1.0e-10,
        density_spectrum_abs=1.0e-10,
        commutator_rms_ev=1.0e-10,
        fermi_map_rms=1.0e-10,
    )
    problem = build_microscopic_hf_problem(
        vertices=vertices,
        interaction_spec=spec,
        references=references,
        nk=1,
        dimension=2,
        total_occupied=1,
        fermi_degeneracy_tolerance=1.0e-12,
        mixing=1.0,
        kbt_ev=0.2,
        k_weights=np.asarray([0.25]),
        final_gate_tolerances=tolerances,
        finite_t_convergence=MicroscopicHFFiniteTConvergence(
            helmholtz_change_abs_ev=1.0e-10,
            required_consecutive_iterations=1,
        ),
        allow_unbound_test_problem=True,
    )
    stored_h0 = ket_hamiltonian_to_repository_blocks(h0)
    stored_density = ket_blocks_to_repository_stored(density)
    expected = 0.25 * hartree_fock_energy(
        density,
        h0,
        vertices,
        spec,
        references,
    )
    references.hartree[:, :, :] = 0.9 * np.eye(2)
    references.fock[:, :, :] = 0.7 * np.eye(2)
    energy = problem.kernel.energy_functional(
        np.zeros_like(stored_h0), stored_h0, stored_density
    )
    assert energy == pytest.approx(expected)


def test_generic_hf_adapter_finite_temperature_final_gates_are_distinct() -> None:
    h0 = np.asarray((np.diag([-1.0, 1.0]),), dtype=np.complex128)
    tolerances = MicroscopicHFFinalGateTolerances(
        particle_number_abs=1.0e-8,
        density_spectrum_abs=1.0e-8,
        commutator_rms_ev=1.0e-8,
        fermi_map_rms=1.0e-8,
    )
    problem = _finite_temperature_toy_problem(
        h0,
        weights=np.ones(1),
        tolerances=tolerances,
    )
    final_update = problem.kernel.density_builder(
        ket_hamiltonian_to_repository_blocks(h0)
    )
    fermi_candidate = repository_stored_to_ket_blocks(final_update.density)[0]
    candidates = (
        (np.diag([-1.0e-4, 1.0]), "final_density_spectrum"),
        (0.4 * np.eye(2), "final_particle_number"),
        (np.asarray([[0.5, 0.1], [0.1, 0.5]]), "final_commutator"),
        (0.5 * np.eye(2), "final_fermi_map"),
        (fermi_candidate, "final_helmholtz_change"),
    )
    assert problem.kernel.final_acceptance is not None
    for candidate, expected_reason in candidates:
        state = MicroscopicHFState.from_ket_h0(
            h0,
            precision=1.0e-12,
            initial_projector_ket=candidate[None, :, :],
        )
        result = problem.kernel.final_acceptance(state, final_update, 0.0)
        assert not result.accepted
        assert result.reason == expected_reason


def test_generic_hf_adapter_rejects_incomplete_finite_temperature_contract() -> None:
    h0 = np.asarray((np.diag([-1.0, 1.0]),), dtype=np.complex128)
    vertices = (
        identity_density_vertex(
            nk=1,
            dimension=2,
            weight=0.0,
            provenance="test-only zero interaction",
        ),
    )
    common = dict(
        vertices=vertices,
        interaction_spec=_absolute_interaction(),
        references=_absolute_references(),
        nk=1,
        dimension=2,
        total_occupied=1,
        fermi_degeneracy_tolerance=1.0e-12,
        mixing=1.0,
        allow_unbound_test_problem=True,
    )
    tolerances = MicroscopicHFFinalGateTolerances(
        particle_number_abs=1.0e-8,
        density_spectrum_abs=1.0e-8,
        commutator_rms_ev=1.0e-8,
        fermi_map_rms=1.0e-8,
    )
    convergence = MicroscopicHFFiniteTConvergence(
        helmholtz_change_abs_ev=1.0e-8,
        required_consecutive_iterations=2,
    )
    for invalid_stripes in (True, 0, 3):
        with pytest.raises(TypeError, match="fock_output_row_stripes"):
            build_microscopic_hf_problem(
                **common,
                fock_output_row_stripes=invalid_stripes,
            )
    with pytest.raises(ValueError, match="explicit k_weights"):
        build_microscopic_hf_problem(
            **common,
            kbt_ev=0.2,
            final_gate_tolerances=tolerances,
        )
    for nonuniform_weights in (
        np.asarray([0.25, 0.75]),
        np.asarray([1.0, np.nextafter(1.0, 2.0)]),
    ):
        with pytest.raises(ValueError, match="uniform k_weights"):
            build_microscopic_hf_problem(
                **{**common, "nk": 2},
                kbt_ev=0.2,
                k_weights=nonuniform_weights,
                final_gate_tolerances=tolerances,
                finite_t_convergence=convergence,
            )
    with pytest.raises(ValueError, match="explicit finite_t_convergence"):
        build_microscopic_hf_problem(
            **common,
            kbt_ev=0.2,
            k_weights=np.ones(1),
            final_gate_tolerances=tolerances,
        )
    with pytest.raises(ValueError, match="require explicit finite kbt_ev"):
        build_microscopic_hf_problem(
            **common,
            k_weights=np.ones(1),
        )
    with pytest.raises(ValueError, match="finite and strictly positive"):
        MicroscopicHFFinalGateTolerances(
            particle_number_abs=-1.0,
            density_spectrum_abs=1.0e-8,
            commutator_rms_ev=1.0e-8,
            fermi_map_rms=1.0e-8,
        )
    with pytest.raises(TypeError, match="exact integer"):
        MicroscopicHFFiniteTConvergence(
            helmholtz_change_abs_ev=1.0e-8,
            required_consecutive_iterations=True,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        MicroscopicHFFiniteTConvergence(
            helmholtz_change_abs_ev=0.0,
            required_consecutive_iterations=2,
        )
