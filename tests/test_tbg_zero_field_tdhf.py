from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mean_field.systems.tbg.zero_field.companion_tdhf as companion_tdhf_module
from mean_field.core.hf import (
    build_projected_interaction_hamiltonian,
    calculate_norm_convergence,
    classify_tdhf_stability,
    compute_hf_energy,
    solve_tdhf_liouvillian,
)
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field.companion_kivc_seed import (
    KWAN_EQ99_ARXIV,
    KWAN_EQ99_PDF_SHA256,
    KWAN_EQ99_PDF_SOURCE,
    KWAN_EQ99_REFERENCE,
    MAX_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE,
    MAX_VALIDATION_TOLERANCE,
    TBG_ZERO_FIELD_COMPANION_KIVC_BASIS_COVARIANCE_SCOPE,
    TBG_ZERO_FIELD_COMPANION_KIVC_CANONICAL_ORDER,
    TBG_ZERO_FIELD_COMPANION_KIVC_CHERN_SCOPE,
    TBG_ZERO_FIELD_COMPANION_KIVC_COMPANION_MEASURE_TP_SCOPE,
    TBG_ZERO_FIELD_COMPANION_KIVC_EXTERNAL_AUTHORITY_FILES,
    TBG_ZERO_FIELD_COMPANION_KIVC_FRAME_SCOPE,
    TBG_ZERO_FIELD_COMPANION_KIVC_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE,
    TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCOPE,
    TBG_ZERO_FIELD_COMPANION_KIVC_STORED_PROJECTOR_CONVENTION,
    TBG_ZERO_FIELD_COMPANION_KIVC_TP_SCOPE,
    TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE,
    TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256,
    TBG_ZERO_FIELD_COMPANION_MEASURE_TP_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE,
    TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256,
    build_tbg_zero_field_companion_kivc_seed,
    kwan_eq99_kivc_q,
    validate_tbg_zero_field_companion_kivc_external_authorities,
)
from mean_field.systems.tbg.zero_field.companion_hf_action import (
    TBGZeroFieldCompanionHFActionSpec,
    TBGZeroFieldCompanionHFEnergy,
    TBGZeroFieldCompanionHFEvaluationArrayHashes,
    TBG_ZERO_FIELD_COMPANION_CALC_E_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_CALC_FOCK_MATRIX_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_FORM_FACTOR_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_GEN_H_SP_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_GEN_M_TVE_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_HF_ACTION_ARRAY_HASH_SEMANTICS,
    TBG_ZERO_FIELD_COMPANION_HF_ACTION_ENERGY_UNITS,
    TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCOPE,
    TBG_ZERO_FIELD_COMPANION_HF_INPUT_SOURCE_SHA256,
    TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE_SHA256,
    TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE_SHA256,
    calc_E as companion_calc_E,
    calc_fock_matrix as companion_calc_fock_matrix,
    gen_H_SP as companion_gen_H_SP,
    gen_M_tVE as companion_gen_M_tVE,
    gen_form_factors as companion_gen_form_factors,
    gen_full_form_factors as companion_gen_full_form_factors,
    main_program_realify_form as companion_main_program_realify_form,
    prepare_tbg_zero_field_companion_hf_action,
)
from mean_field.systems.tbg.zero_field.companion_hf_scf import (
    TBGZeroFieldCompanionHFQualificationReport,
    TBGZeroFieldCompanionHFQualifierSpec,
    TBGZeroFieldCompanionHFSCFSpec,
    TBG_ZERO_FIELD_COMPANION_AUFBAU_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_AVERAGE_CENTRAL_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_HF_SCF_CONVERGENCE_CONVENTION,
    TBG_ZERO_FIELD_COMPANION_HF_SCF_SCOPE,
    TBG_ZERO_FIELD_COMPANION_HF_SCF_SOURCE_PARITY_EXCEPTION,
    TBG_ZERO_FIELD_COMPANION_MAIN_ODA_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_MAIN_SCF_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_MEASURE_REFERENCE_LINES,
    build_companion_average_central_reference,
    companion_aufbau,
    companion_oda_branch,
    companion_oda_coefficients,
    qualify_tbg_zero_field_companion_hf_diagnostic,
    run_tbg_zero_field_companion_hf_diagnostic,
)
from mean_field.systems.tbg.zero_field.companion_tdhf import (
    Stage7ADiagnosticConsumptionReceipt,
    TBGZeroFieldCompanionTDHFSource,
    TBG_ZERO_FIELD_COMPANION_TDHF_ARCHITECTURE_EXCEPTION,
    TBG_ZERO_FIELD_COMPANION_TDHF_COMMON_SPIN_BASIS_ATOL_EV,
    TBG_ZERO_FIELD_COMPANION_TDHF_COMMON_SPIN_BASIS_SOURCE,
    TBG_ZERO_FIELD_COMPANION_TDHF_DEGENERACY_ATOL_EV,
    TBG_ZERO_FIELD_COMPANION_TDHF_DIAGNOSTIC_CONSUMPTION_SCOPE,
    TBG_ZERO_FIELD_COMPANION_TDHF_EVIDENCE_BUNDLE_SCHEMA,
    TBG_ZERO_FIELD_COMPANION_TDHF_EVIDENCE_BUNDLE_SCHEMA_VERSION,
    TBG_ZERO_FIELD_COMPANION_TDHF_EQ90_SIGN_CONVENTION,
    TBG_ZERO_FIELD_COMPANION_TDHF_MAX_MIXED_AUFBAU_CLOSURE,
    TBG_ZERO_FIELD_COMPANION_TDHF_PAPER_ARXIV,
    TBG_ZERO_FIELD_COMPANION_TDHF_PAPER_EQUATIONS,
    TBG_ZERO_FIELD_COMPANION_TDHF_Q0_RAW_PAIRING_RESIDUAL_ATOL_EV,
    TBG_ZERO_FIELD_COMPANION_TDHF_RAW_EIGENSOLVER_RESIDUAL_ATOL_EV,
    TBG_ZERO_FIELD_COMPANION_TDHF_SCOPE,
    TBG_ZERO_FIELD_COMPANION_TDHF_SELECTED_EIGENSOLVER_RESIDUAL_ATOL_EV,
    TBG_ZERO_FIELD_COMPANION_TDHF_SUMMARY_SCHEMA,
    TBG_ZERO_FIELD_COMPANION_TDHF_SUMMARY_SCHEMA_VERSION,
    _build_tbg_zero_field_companion_single_spin_q0_parity_oracle,
    build_tbg_zero_field_companion_hf_form_factors,
    build_tbg_zero_field_companion_q0_parity_oracle,
    build_tbg_zero_field_companion_signed_q_label,
    build_tbg_zero_field_companion_static_kernel,
    build_tbg_zero_field_companion_tdhf_source_from_in_memory_arrays,
    build_tbg_zero_field_companion_tdhf_source_from_stage6_run,
    build_tbg_zero_field_companion_transition_inventories,
    evaluate_tbg_zero_field_companion_static_matrix_action,
    load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts,
    solve_tbg_zero_field_companion_dense_spectrum,
    tbg_zero_field_companion_signed_spectral_pairing,
)
from mean_field.systems.tbg.zero_field.companion_interaction import (
    TBGZeroFieldCompanionInteractionSpec as TBGZeroFieldCompanionSourceInteractionSpec,
    TBG_ZERO_FIELD_COMPANION_INTERACTION_ARRAY_HASH_SEMANTICS as SOURCE_INTERACTION_ARRAY_HASH_SEMANTICS,
    TBG_ZERO_FIELD_COMPANION_INTERACTION_FINITE_Q_KERNEL_REFERENCE_LINES as SOURCE_INTERACTION_FINITE_Q_KERNEL_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_INTERACTION_Q0_REFERENCE_LINES as SOURCE_INTERACTION_Q0_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_LINES as SOURCE_INTERACTION_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_INTERACTION_SCOPE,
    TBG_ZERO_FIELD_COMPANION_INTERACTION_SOURCE_DSC_M,
    TBG_ZERO_FIELD_COMPANION_INTERACTION_SUPPORT_REFERENCE_LINES as SOURCE_INTERACTION_SUPPORT_REFERENCE_LINES,
    echarge as companion_echarge,
    epsilon0 as companion_epsilon0,
    solve_tbg_zero_field_companion_interaction,
)
from mean_field.systems.tbg.zero_field.companion_single_particle import (
    Beta,
    CCa,
    C2T_symmetry as companion_C2T_symmetry,
    Poisson,
    TBGZeroFieldCompanionSingleParticleArrayHashes,
    TBGZeroFieldCompanionSingleParticleParams,
    TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_SEMANTICS,
    TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE,
    TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256,
    TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE,
    TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE_SHA256,
    TBG_ZERO_FIELD_COMPANION_POINTWISE_GAUGE_WARNING,
    TBG_ZERO_FIELD_COMPANION_RESIDUAL_GAUGE_AMBIGUITY,
    TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCOPE,
    gen_RLVs as gen_companion_RLVs,
    gen_moire_hamiltonian as gen_companion_moire_hamiltonian,
    kD as companion_kD,
    solve_tbg_zero_field_companion_single_particle,
    vhbar as companion_vhbar,
)
from mean_field.systems.tbg.zero_field import (
    BMSolution,
    HFOverlapBlockSet,
    TBGZeroFieldCompanionInteractionSpec,
    TBGZeroFieldCompanionPlaneWaveSpec,
    TBG_ZERO_FIELD_COMPANION_HOPPING_REFERENCE_FUNCTION,
    TBG_ZERO_FIELD_COMPANION_HOPPING_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_FUNCTION,
    TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_REFERENCE_FUNCTION,
    TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_REFERENCE_LINES,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
    RestrictedHartreeFockRun,
    RestrictedHartreeFockState,
    TBGZeroFieldHFRunProvenance,
    TBGZeroFieldHFSourceReceipt,
    TBGZeroFieldInteractionSpec,
    TBGZeroFieldTDHFContext,
    TBGZeroFieldTDHFOccupationResiduals,
    TBGZeroFieldTDHFOrbitals,
    TBGZeroFieldTDHFProvenance,
    TBG_ZERO_FIELD_TDHF_PRODUCTION_MAX_CLOSURE_TOLERANCE_MEV,
    TBG_ZERO_FIELD_TDHF_PRODUCTION_MAX_PROJECTOR_TOLERANCE,
    TBG_ZERO_FIELD_TDHF_PRODUCTION_MAX_STRUCTURE_TOLERANCE,
    TBG_ZERO_FIELD_TDHF_PRODUCTION_MAX_TANGENT_TOLERANCE,
    build_full_density_from_hamiltonian,
    build_overlap_block_set,
    build_tbg_zero_field_companion_interaction_geometry,
    build_tbg_zero_field_companion_plane_wave_geometry,
    build_restricted_density_from_hamiltonian,
    build_tbg_zero_field_half_open_torus_mesh,
    build_tbg_zero_field_diagnostic_hf_source_receipt,
    build_tbg_zero_field_hf_source_receipt,
    build_tbg_zero_field_screened_block_bundle,
    build_tbg_zero_field_tdhf_orbitals,
    build_tbg_zero_field_tdhf_q0_matrices,
    build_tbg_zero_field_tdhf_q0_pairs,
    conventional_projector_to_stored_density,
    conventional_tangent_to_stored_tangent,
    coulomb_unit,
    empty_overlap_block_set,
    screened_coulomb,
    run_full_hartree_fock,
    run_full_hf_from_bm_solution,
    run_restricted_hf_from_bm_solution,
    solve_bm_model,
    solve_bm_model_on_torus,
    stored_density_to_conventional_projector,
    stored_tangent_to_conventional_tangent,
    tbg_zero_field_lattice_kvec_sha256,
    tbg_zero_field_overlap_kernel_inventory_fingerprint,
    validate_tbg_zero_field_tdhf_tangent_columns,
)


_INTERACTION_SPEC = TBGZeroFieldInteractionSpec()


def _provenance(
    label: str,
    receipt: TBGZeroFieldHFSourceReceipt,
    *,
    run_provenance: TBGZeroFieldHFRunProvenance | None = None,
) -> TBGZeroFieldTDHFProvenance:
    return TBGZeroFieldTDHFProvenance(
        hf_run_source=f"synthetic:{label}:hf",
        overlap_blocks_source=f"synthetic:{label}:overlaps",
        interaction_parameters_source=f"synthetic:{label}:interaction",
        reference_density_source=f"synthetic:{label}:centered-half-identity",
        expected_hf_source_receipt_sha256=receipt.fingerprint,
        expected_hf_run_provenance_sha256=(
            None if run_provenance is None else run_provenance.fingerprint
        ),
        hf_mode="full",
    )


def _bm_solution(
    *,
    n_spin: int,
    n_eta: int,
    n_band: int,
    nk: int,
    h0: np.ndarray | None = None,
) -> BMSolution:
    params = TBGParameters.from_degrees(1.05)
    nt = n_spin * n_eta * n_band
    plane_wave_dimension = 4
    lattice_kvec = np.asarray([0.0 + 0.0j, 0.17 + 0.09j], dtype=np.complex128)[:nk]
    spectrum = np.zeros((n_band, n_eta, nk), dtype=float)
    if h0 is not None:
        resolved_h0 = np.asarray(h0, dtype=np.complex128)
        assert resolved_h0.shape == (nt, nt, nk)
        row = 0
        for iband in range(n_band):
            for ieta in range(n_eta):
                spin_values = np.asarray(
                    [resolved_h0[row + ispin, row + ispin, :] for ispin in range(n_spin)]
                )
                np.testing.assert_allclose(spin_values, spin_values[[0], :], rtol=0.0, atol=1.0e-14)
                spectrum[iband, ieta, :] = spin_values[0].real
                row += n_spin
    return BMSolution(
        params=params,
        lattice_kvec=lattice_kvec,
        lg=1,
        nlocal=4,
        n_eta=n_eta,
        n_spin=n_spin,
        nb=n_band,
        hamiltonian=np.zeros(
            (plane_wave_dimension, plane_wave_dimension, n_eta, nk),
            dtype=np.complex128,
        ),
        sigma_z=np.zeros((nt, nt, nk), dtype=np.complex128),
        uk=np.zeros((plane_wave_dimension, n_band, n_eta, nk), dtype=np.complex128),
        spectrum=spectrum,
        gvec=np.asarray([0.0 + 0.0j], dtype=np.complex128),
    )


_TYPED_TORUS_SOLUTION_CACHE: dict[tuple[int, int], BMSolution] = {}
_BM_SOLUTION_ARRAY_FIELDS = (
    "lattice_kvec",
    "hamiltonian",
    "sigma_z",
    "uk",
    "spectrum",
    "gvec",
)

def _typed_torus_solution(mesh_size: int = 1, *, lg: int = 7) -> BMSolution:
    key = (int(mesh_size), int(lg))
    if key not in _TYPED_TORUS_SOLUTION_CACHE:
        base = solve_bm_model_on_torus(
            TBGParameters.from_degrees(1.05),
            mesh_size,
            lg=lg,
            calculate_chern_operator=True,
        )
        for name in _BM_SOLUTION_ARRAY_FIELDS:
            np.asarray(getattr(base, name)).setflags(write=False)
        _TYPED_TORUS_SOLUTION_CACHE[key] = base
    copied = deepcopy(_TYPED_TORUS_SOLUTION_CACHE[key])
    for name in _BM_SOLUTION_ARRAY_FIELDS:
        object.__setattr__(copied, name, np.array(getattr(copied, name), copy=True))
    return copied

def _two_k_source() -> tuple[BMSolution, RestrictedHartreeFockRun]:
    nt = 2
    nk = 2
    beta = 0.6
    v0 = 0.8
    hf_energies = np.asarray([[-1.0, -0.8], [0.7, 1.1]], dtype=float)
    hamiltonian = np.zeros((nt, nt, nk), dtype=np.complex128)
    projector = np.zeros_like(hamiltonian)
    for ik in range(nk):
        hamiltonian[:, :, ik] = np.diag(hf_energies[:, ik])
        projector[:, :, ik] = np.diag([1.0, 0.0])

    overlap = np.zeros((nt, nk, nt, nk), dtype=np.complex128)
    overlap[:, 0, :, 0] = np.diag([1.0, 0.8])
    overlap[:, 1, :, 1] = np.diag([0.9, 1.1])
    overlap[:, 0, :, 1] = np.diag([0.7 + 0.1j, 0.6 - 0.05j])
    overlap[:, 1, :, 0] = overlap[:, 0, :, 1].conjugate().T
    diagonal = np.diagonal(overlap, axis1=1, axis2=3)
    blocks = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        overlaps={(0, 0): overlap},
        diagonal_overlaps={(0, 0): diagonal},
        hartree_screening={(0, 0): 0.3},
        fock_screening={(0, 0): np.asarray([[0.9, 0.4], [0.4, 1.1]], dtype=float)},
    )
    density = conventional_projector_to_stored_density(projector)
    interaction = build_projected_interaction_hamiltonian(
        density,
        blocks,
        v0=v0,
        beta=beta,
    )
    h0 = hamiltonian - interaction
    solution = _bm_solution(n_spin=1, n_eta=1, n_band=2, nk=nk, h0=h0)
    state = RestrictedHartreeFockState(
        h0=h0,
        sigma_z=np.zeros_like(hamiltonian),
        density=density,
        hamiltonian=hamiltonian,
        energies=hf_energies,
        sigma_ztauz=np.zeros((nt, nk), dtype=float),
        nu=0.0,
        v0=v0,
        n_spin=1,
        n_eta=1,
        n_band=2,
        diagnostics={"beta": beta},
        interaction_spec=_INTERACTION_SPEC,
    )
    state.hf_source_receipt = build_tbg_zero_field_diagnostic_hf_source_receipt(
        hf_mode="full",
        beta=beta,
        v0=state.v0,
        lattice_kvec=solution.lattice_kvec,
        overlap_blocks=blocks,
    )
    run = RestrictedHartreeFockRun(
        state=state,
        overlap_blocks=blocks,
        iter_energy=np.asarray([-1.0], dtype=float),
        iter_err=np.asarray([0.0], dtype=float),
        iter_oda=np.asarray([1.0], dtype=float),
        init_mode="synthetic-full",
        seed=0,
        converged=True,
        exit_reason="synthetic-converged",
    )
    return solution, run


def _source_with_overlap_blocks(
    blocks: HFOverlapBlockSet,
) -> tuple[BMSolution, RestrictedHartreeFockRun]:
    solution, run = _two_k_source()
    state = run.state
    beta = float(state.diagnostics["beta"])
    interaction = build_projected_interaction_hamiltonian(
        state.density,
        blocks,
        v0=state.v0,
        beta=beta,
    )
    state.h0 = state.hamiltonian - interaction
    solution = _bm_solution(
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
        nk=state.nk,
        h0=state.h0,
    )
    state.hf_source_receipt = build_tbg_zero_field_diagnostic_hf_source_receipt(
        hf_mode="full",
        beta=beta,
        v0=state.v0,
        lattice_kvec=solution.lattice_kvec,
        overlap_blocks=blocks,
    )
    return solution, replace(run, overlap_blocks=blocks)


def _context_from_source(
    solution: BMSolution,
    run: RestrictedHartreeFockRun,
    *,
    label: str = "two-k",
) -> TBGZeroFieldTDHFContext:
    return TBGZeroFieldTDHFContext(
        grid_solution=solution,
        run=run,
        beta=float(run.state.diagnostics["beta"]),
        provenance=_provenance(label, run.state.hf_source_receipt),
        allow_diagnostic_source=True,
        closure_tolerance=1.0e-12,
    )


def _two_k_context() -> TBGZeroFieldTDHFContext:
    return _context_from_source(*_two_k_source())


def test_source_receipt_fingerprint_binds_v0() -> None:
    _solution, run = _two_k_source()
    receipt = run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    changed_v0_receipt = replace(receipt, v0=receipt.v0 + 1.0)

    assert receipt.schema_version == 5
    assert receipt.interaction_contract == "legacy_untyped_diagnostic"
    assert receipt.interaction_spec_fingerprint is None
    assert changed_v0_receipt.fingerprint != receipt.fingerprint
    assert (
        replace(receipt, overlap_kernel_inventory_sha256="0" * 64).fingerprint
        != receipt.fingerprint
    )


def test_source_receipt_json_metadata_round_trip_reconstructs_and_verifies() -> None:
    _solution, run = _two_k_source()
    receipt = run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    loaded_metadata = json.loads(
        json.dumps(receipt.to_metadata(), sort_keys=True, allow_nan=False)
    )

    reconstructed = TBGZeroFieldHFSourceReceipt.from_metadata(loaded_metadata)
    assert reconstructed == receipt
    assert reconstructed.fingerprint == loaded_metadata["fingerprint"]

    loaded_metadata["v0"] = receipt.v0 + 1.0
    with pytest.raises(ValueError, match="fingerprint does not match"):
        TBGZeroFieldHFSourceReceipt.from_metadata(loaded_metadata)


def test_typed_interaction_maps_physical_25nm_dual_gate_and_finite_q0() -> None:
    spec = TBGZeroFieldInteractionSpec()
    assert spec.screening_lm == pytest.approx(25.0 / (2.0 * 0.246), rel=0.0, abs=0.0)
    assert spec.reference_scheme == "central_average_active_two_band"
    assert spec.transfer_cutoff_policy == "g_label_hex_shell_3"
    assert spec.companion_circular_total_q_cutoff_parity == "not_established"
    assert spec.finite_zero_limit is True
    assert screened_coulomb(
        0.0 + 0.0j,
        spec.screening_lm,
        relative_permittivity=spec.epsr,
        zero_cutoff=spec.zero_cutoff,
        finite_zero_limit=spec.finite_zero_limit,
    ) == pytest.approx(4.0 * np.pi * spec.screening_lm / spec.epsr, abs=1.0e-14)
    assert TBGZeroFieldInteractionSpec.from_json(spec.to_json()) == spec


def test_half_open_10x10_torus_is_unique_fortran_ordered_and_hashed() -> None:
    params = TBGParameters.from_degrees(1.05, w0=80.0, w1=110.0)
    mesh = build_tbg_zero_field_half_open_torus_mesh(params, 10)
    repeated = build_tbg_zero_field_half_open_torus_mesh(params, 10)

    np.testing.assert_array_equal(
        mesh.k_grid_frac[:11],
        np.asarray([[index / 10.0, 0.0] for index in range(10)] + [[0.0, 0.1]]),
    )
    assert mesh.k_grid_frac.shape == (100, 2)
    assert np.unique(mesh.k_grid_frac, axis=0).shape[0] == 100
    assert np.all(mesh.k_grid_frac >= 0.0)
    assert np.all(mesh.k_grid_frac < 1.0)
    assert mesh.fingerprint == repeated.fingerprint
    assert mesh.to_metadata() == repeated.to_metadata()
    assert mesh.to_metadata()["index_order"] == "F"
    assert not mesh.k_grid_frac.flags.writeable
    assert not mesh.kvec.flags.writeable


def test_torus_mesh_copies_inputs_and_rejects_mutation_or_wrong_order() -> None:
    params = TBGParameters.from_degrees(1.05)
    built = build_tbg_zero_field_half_open_torus_mesh(params, 2)
    frac_source = np.array(built.k_grid_frac, copy=True)
    kvec_source = np.array(built.kvec, copy=True)
    mesh = type(built)(
        mesh_size=2,
        g1=params.g1,
        g2=params.g2,
        k_grid_frac=frac_source,
        kvec=kvec_source,
    )
    frac_source[0, 0] = 0.25
    kvec_source[0] = 1.0 + 0.0j
    np.testing.assert_array_equal(mesh.k_grid_frac, built.k_grid_frac)
    np.testing.assert_array_equal(mesh.kvec, built.kvec)
    with pytest.raises(ValueError, match="read-only"):
        mesh.k_grid_frac[0, 0] = 0.25
    with pytest.raises(ValueError, match="Fortran-ordered"):
        type(built)(
            mesh_size=2,
            g1=params.g1,
            g2=params.g2,
            k_grid_frac=built.k_grid_frac[[0, 2, 1, 3]],
            kvec=built.kvec[[0, 2, 1, 3]],
        )
    wrong_kvec = np.array(built.kvec, copy=True)
    wrong_kvec[-1] += 1.0e-12
    with pytest.raises(ValueError, match=r"f1\*g1\+f2\*g2"):
        type(built)(
            mesh_size=2,
            g1=params.g1,
            g2=params.g2,
            k_grid_frac=built.k_grid_frac,
            kvec=wrong_kvec,
        )


def test_solve_bm_model_issues_non_torus_source_attestation() -> None:
    params = TBGParameters.from_degrees(1.05)
    solution = solve_bm_model(
        params,
        np.asarray([0.0 + 0.0j]),
        lg=1,
        calculate_chern_operator=False,
    )
    assert solution.source_attestation is not None
    assert solution.source_attestation.solver_entrypoint == "solve_bm_model"
    assert solution.source_attestation.torus_mesh_fingerprint is None
    solution.validate_source_attestation()


def test_tbg_parameters_derived_arrays_are_owned_and_read_only() -> None:
    params = TBGParameters.from_degrees(1.05, strain=2.0e-4)
    for name in (
        "t0",
        "t1",
        "t2",
        "gauge_shift",
        "rotation_phi",
        "strain_matrix",
    ):
        values = getattr(params, name)
        assert values.flags.owndata
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            values.flat[0] = 0.0


def test_typed_bundle_binds_mesh_shell_and_central_reference() -> None:
    solution = _typed_torus_solution()
    assert solution.lg == 7
    assert solution.source_attestation is not None
    assert solution.source_attestation.solver_entrypoint == "solve_bm_model_on_torus"
    solution.validate_source_attestation(require_torus=True)
    bundle = build_tbg_zero_field_screened_block_bundle(
        solution,
        interaction_spec=_INTERACTION_SPEC,
        overlap_lg=7,
    )
    receipt = build_tbg_zero_field_hf_source_receipt(
        hf_mode="full",
        beta=1.0,
        v0=coulomb_unit(solution.params),
        solution=solution,
        screened_block_bundle=bundle,
    )
    assert receipt.schema_version == 5
    assert receipt.n_band == 2
    assert receipt.reference_projector_dimensions == (8, 8, 1)
    assert receipt.reference_projector_sha256 == bundle.reference_projector_sha256
    assert receipt.overlap_lg == 7
    assert receipt.active_shift_inventory == bundle.active_shifts
    assert receipt.active_shift_inventory_sha256 == bundle.active_shift_inventory_sha256
    assert receipt.screened_block_bundle_sha256 == bundle.fingerprint
    assert bundle.bm_generation_fingerprint == solution.generation_fingerprint
    assert receipt.bm_generation_fingerprint == solution.generation_fingerprint
    assert receipt.mesh_fingerprint == solution.torus_mesh.fingerprint
    assert receipt.companion_circular_total_q_cutoff_parity == "not_established"
    assert not bundle.screened_blocks.gvecs.flags.writeable

    stale_spectrum = np.array(solution.spectrum, copy=True)
    stale_spectrum[0, 0, 0] += 1.0e-9
    stale_solution = replace(solution, spectrum=stale_spectrum)
    with pytest.raises(ValueError, match="source attestation"):
        build_tbg_zero_field_hf_source_receipt(
            hf_mode="full",
            beta=1.0,
            v0=coulomb_unit(stale_solution.params),
            solution=stale_solution,
            screened_block_bundle=bundle,
        )

    with pytest.raises(ValueError, match="insufficient"):
        build_tbg_zero_field_screened_block_bundle(
            solution,
            interaction_spec=_INTERACTION_SPEC,
            overlap_lg=5,
        )
    with pytest.raises(ValueError, match="freezes graphene_a_nm exactly"):
        TBGZeroFieldInteractionSpec(graphene_a_nm=0.245)


@pytest.mark.parametrize("bad_lg", [1, 3, 5], ids=lambda value: f"lg{value}")
def test_typed_bundle_refuses_attested_undersized_bm_sources(bad_lg: int) -> None:
    solution = _typed_torus_solution(lg=bad_lg)
    solution.validate_source_attestation(require_torus=True)
    with pytest.raises(ValueError, match=r"solution\.lg.*odd integer >= 7.*diagnostic-only"):
        build_tbg_zero_field_screened_block_bundle(
            solution,
            interaction_spec=_INTERACTION_SPEC,
            overlap_lg=7,
        )

@pytest.mark.parametrize("bad_lg", [6, 8], ids=lambda value: f"lg{value}")
def test_typed_bundle_refuses_even_bm_sources(bad_lg: int) -> None:
    solution = _typed_torus_solution()
    object.__setattr__(solution, "lg", bad_lg)
    with pytest.raises(ValueError, match=r"solution\.lg.*odd integer >= 7.*diagnostic-only"):
        build_tbg_zero_field_screened_block_bundle(
            solution,
            interaction_spec=_INTERACTION_SPEC,
            overlap_lg=7,
        )

def test_typed_primitive_cell_entrypoints_refuse_fractional_nu() -> None:
    solution = _typed_torus_solution()
    for entrypoint in (
        run_full_hf_from_bm_solution,
        run_restricted_hf_from_bm_solution,
    ):
        with pytest.raises(ValueError, match="separate supercell workflow"):
            entrypoint(
                solution,
                nu=0.25,
                max_iter=0,
                overlap_lg=7,
                interaction_spec=_INTERACTION_SPEC,
            )

def test_typed_bundle_rejects_hand_constructed_and_reference_modified_bm_sources() -> None:
    solved = _typed_torus_solution()
    hand_constructed = BMSolution(
        params=solved.params,
        lattice_kvec=np.array(solved.lattice_kvec, copy=True),
        lg=solved.lg,
        nlocal=solved.nlocal,
        n_eta=solved.n_eta,
        n_spin=solved.n_spin,
        nb=solved.nb,
        hamiltonian=np.array(solved.hamiltonian, copy=True),
        sigma_z=np.array(solved.sigma_z, copy=True),
        uk=np.array(solved.uk, copy=True),
        spectrum=np.array(solved.spectrum, copy=True),
        gvec=np.array(solved.gvec, copy=True),
        sigma_rotation=solved.sigma_rotation,
        calculate_chern_operator=solved.calculate_chern_operator,
        periodic_g_grid=solved.periodic_g_grid,
        torus_mesh=solved.torus_mesh,
    )
    assert hand_constructed.source_attestation is None
    with pytest.raises(ValueError, match="diagnostic-only"):
        build_tbg_zero_field_screened_block_bundle(
            hand_constructed,
            interaction_spec=_INTERACTION_SPEC,
            overlap_lg=7,
        )
    stolen_attestation_clone = replace(
        hand_constructed,
        source_attestation=solved.source_attestation,
    )
    with pytest.raises(ValueError, match="hand-constructed clones are diagnostic-only"):
        build_tbg_zero_field_screened_block_bundle(
            stolen_attestation_clone,
            interaction_spec=_INTERACTION_SPEC,
            overlap_lg=7,
        )

    reference_modified = solved.with_reference_uk(np.array(solved.uk, copy=True))
    assert reference_modified.source_attestation is None
    with pytest.raises(ValueError, match="diagnostic-only"):
        build_tbg_zero_field_screened_block_bundle(
            reference_modified,
            interaction_spec=_INTERACTION_SPEC,
            overlap_lg=7,
        )


def test_bm_source_attestation_rejects_live_array_and_parameter_mutation() -> None:
    array_mutated = _typed_torus_solution()
    array_mutated.hamiltonian[0, 0, 0, 0] += 1.0e-9
    with pytest.raises(ValueError, match="source attestation.*live solver arrays"):
        build_tbg_zero_field_screened_block_bundle(
            array_mutated,
            interaction_spec=_INTERACTION_SPEC,
            overlap_lg=7,
        )

    params_mutated = _typed_torus_solution()
    object.__setattr__(params_mutated.params, "w0", params_mutated.params.w0 + 1.0)
    with pytest.raises(ValueError, match="live independent parameters"):
        build_tbg_zero_field_screened_block_bundle(
            params_mutated,
            interaction_spec=_INTERACTION_SPEC,
            overlap_lg=7,
        )

    flag_source = _typed_torus_solution()
    object.__setattr__(
        flag_source,
        "calculate_chern_operator",
        not flag_source.calculate_chern_operator,
    )
    with pytest.raises(ValueError, match="live dimensions/flags"):
        build_tbg_zero_field_screened_block_bundle(
            flag_source,
            interaction_spec=_INTERACTION_SPEC,
            overlap_lg=7,
        )


    pristine = _typed_torus_solution()
    pristine.validate_source_attestation(require_torus=True)
    assert pristine.lg == 7

def test_typed_screening_rejects_raw_mismatch_and_implicit_legacy_receipt() -> None:
    solution = _bm_solution(n_spin=1, n_eta=1, n_band=1, nk=1)
    with pytest.raises(ValueError, match="Raw screening_lm is rejected"):
        build_overlap_block_set(
            solution,
            lg=1,
            interaction_spec=_INTERACTION_SPEC,
            screening_lm=_INTERACTION_SPEC.screening_lm + 1.0,
        )

    with pytest.raises(TypeError, match="screened_block_bundle"):
        build_tbg_zero_field_hf_source_receipt(
            hf_mode="full",
            beta=1.0,
            v0=1.0,
            solution=solution,
            screened_block_bundle=empty_overlap_block_set(),  # type: ignore[arg-type]
        )
    legacy = build_tbg_zero_field_diagnostic_hf_source_receipt(
        hf_mode="restricted",
        beta=1.0,
        v0=1.0,
        lattice_kvec=solution.lattice_kvec,
        overlap_blocks=empty_overlap_block_set(),
    )
    assert legacy.interaction_contract == "legacy_untyped_diagnostic"
    assert legacy.interaction_spec_fingerprint is None


@pytest.mark.parametrize(
    ("module_name", "runner_name", "expected_mode", "init_mode"),
    [
        (
            "mean_field.systems.tbg.zero_field._hf_full",
            "run_full_hartree_fock",
            "full",
            "flavor",
        ),
        (
            "mean_field.systems.tbg.zero_field._hf_restricted",
            "run_restricted_hartree_fock",
            "restricted",
            "educated",
        ),
    ],
)
def test_hf_runners_persist_the_screened_blocks_used_by_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    runner_name: str,
    expected_mode: str,
    init_mode: str,
) -> None:
    solution = _typed_torus_solution()
    bundle = build_tbg_zero_field_screened_block_bundle(
        solution,
        interaction_spec=_INTERACTION_SPEC,
        overlap_lg=7,
    )
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=5.0e-13)
    state.diagnostics["overlap_lg"] = 7.0
    captured: dict[str, object] = {}
    module = importlib.import_module(module_name)

    def fake_build_projected_hf_kernel(state, overlap_blocks, **kwargs):
        captured["overlap_blocks"] = overlap_blocks
        captured["oda_parameterizer"] = kwargs["oda_parameterizer"]
        kernel = SimpleNamespace(
            density_builder=kwargs["density_builder"],
            energy_functional=kwargs["energy_functional"],
            density_postprocessor=kwargs.get("density_postprocessor"),
            final_state_callback=kwargs["final_state_callback"],
            convergence_metric=kwargs.get("convergence_metric"),
        )
        captured["kernel"] = kernel
        return kernel

    def fake_run_hartree_fock_problem(state, problem, **kwargs):
        assert problem.kernel is captured["kernel"]
        problem.initializer(
            state,
            init_mode=kwargs["init_mode"],
            seed=kwargs["seed"],
        )
        density_update = problem.kernel.density_builder(state.hamiltonian)
        state.energies[:, :] = density_update.energies
        state.mu = float(density_update.mu)
        interaction_h = state.hamiltonian - state.h0
        state.diagnostics["hf_energy"] = float(
            problem.kernel.energy_functional(interaction_h, state.h0, state.density)
        )
        final_raw_density = np.asarray(
            density_update.density,
            dtype=np.complex128,
        ).copy()
        if problem.kernel.density_postprocessor is not None:
            problem.kernel.density_postprocessor(final_raw_density)
        final_metric = (
            calculate_norm_convergence
            if problem.kernel.convergence_metric is None
            else problem.kernel.convergence_metric
        )
        state.diagnostics["final_raw_norm"] = float(
            final_metric(final_raw_density, state.density)
        )
        problem.kernel.final_state_callback(state, density_update)
        return SimpleNamespace(
            state=state,
            iter_energy=np.asarray([], dtype=float),
            iter_err=np.asarray([], dtype=float),
            iter_oda=np.asarray([], dtype=float),
            init_mode=kwargs["init_mode"],
            seed=kwargs["seed"],
            converged=True,
            exit_reason="synthetic-converged",
        )

    monkeypatch.setattr(module, "build_projected_hf_kernel", fake_build_projected_hf_kernel)
    monkeypatch.setattr(module, "run_hartree_fock_problem", fake_run_hartree_fock_problem)
    with pytest.raises(ValueError, match="separate supercell workflow"):
        getattr(module, runner_name)(
            RestrictedHartreeFockState.from_bm_solution(solution, nu=0.25),
            bundle.screened_blocks,
            solution.lattice_kvec,
            solution.params,
            init_mode=init_mode,
            max_iter=0,
            interaction_spec=_INTERACTION_SPEC,
            source_solution=solution,
            screened_block_bundle=bundle,
        )
    with pytest.raises(ValueError, match="arbitrary overlap blocks"):
        getattr(module, runner_name)(
            RestrictedHartreeFockState.from_bm_solution(solution, nu=0.0),
            empty_overlap_block_set(),
            solution.lattice_kvec,
            solution.params,
            init_mode=init_mode,
            max_iter=0,
            interaction_spec=_INTERACTION_SPEC,
        )

    result = getattr(module, runner_name)(
        state,
        bundle.screened_blocks,
        solution.lattice_kvec,
        solution.params,
        init_mode=init_mode,
        max_iter=0,
        interaction_spec=_INTERACTION_SPEC,
        source_solution=solution,
        screened_block_bundle=bundle,
    )

    assert result.state.nu == 0.0
    assert result.screened_block_bundle is bundle
    assert result.overlap_blocks is bundle.screened_blocks
    assert result.overlap_blocks is captured["overlap_blocks"]
    oda_parameterizer = captured["oda_parameterizer"]
    assert callable(oda_parameterizer)
    assert any(
        cell.cell_contents is result.overlap_blocks
        for cell in oda_parameterizer.__closure__ or ()
    )
    assert result.overlap_blocks.diagonal_overlaps
    assert result.overlap_blocks.hartree_screening
    assert result.overlap_blocks.fock_screening
    assert all(
        isinstance(value, (int, float, np.integer, np.floating))
        for value in result.state.diagnostics.values()
    )
    receipt = result.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    assert receipt.hf_mode == expected_mode
    assert receipt.schema_version == 5
    assert receipt.v0 == result.state.v0
    assert receipt.v0 == coulomb_unit(solution.params)
    assert receipt.lattice_kvec_sha256 == tbg_zero_field_lattice_kvec_sha256(
        solution.lattice_kvec
    )
    assert receipt.overlap_kernel_inventory_sha256 == (
        tbg_zero_field_overlap_kernel_inventory_fingerprint(result.overlap_blocks)
    )
    assert receipt.screened_block_bundle_sha256 == bundle.fingerprint
    assert receipt.bm_generation_fingerprint == solution.generation_fingerprint
    assert TBGZeroFieldHFSourceReceipt.from_metadata(receipt.to_metadata()) == receipt
    run_provenance = result.provenance
    assert isinstance(run_provenance, TBGZeroFieldHFRunProvenance)
    assert run_provenance.hf_mode == expected_mode
    assert run_provenance.beta == receipt.beta
    assert run_provenance.nu == result.state.nu
    assert run_provenance.filling == result.state.nu
    assert run_provenance.precision == result.state.precision
    assert (
        run_provenance.oda_stall_threshold
        == result.state.diagnostics["oda_stall_threshold"]
    )
    assert run_provenance.requested_max_iterations == 0
    assert run_provenance.seed == result.seed
    assert run_provenance.normalized_init_mode == result.init_mode
    assert run_provenance.typed_receipt_fingerprint == receipt.fingerprint
    assert run_provenance.interaction_spec_fingerprint == _INTERACTION_SPEC.fingerprint
    assert run_provenance.bm_generation_fingerprint == solution.generation_fingerprint
    assert run_provenance.mesh_fingerprint == solution.torus_mesh.fingerprint
    assert run_provenance.converged is result.converged
    assert run_provenance.exit_reason == result.exit_reason
    metadata = run_provenance.to_metadata()
    assert metadata["schema_version"] == 2
    assert metadata["issuer"] == "TBGZeroFieldFullRestrictedRunner/v2"
    assert metadata["iter_energy_sha256"] == run_provenance.iter_energy_sha256
    assert metadata["iter_err_sha256"] == run_provenance.iter_err_sha256
    assert metadata["iter_oda_sha256"] == run_provenance.iter_oda_sha256
    assert metadata["state_source_sha256"] == run_provenance.state_source_sha256
    assert metadata["fingerprint"] == run_provenance.fingerprint


def test_typed_hf_receipt_and_runners_reject_unbound_v0_and_initial_density() -> None:
    solution = _typed_torus_solution()
    bundle = build_tbg_zero_field_screened_block_bundle(
        solution,
        interaction_spec=_INTERACTION_SPEC,
        overlap_lg=7,
    )
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=0.0)
    with pytest.raises(ValueError, match=r"v0 must equal coulomb_unit\(solution.params\) exactly"):
        build_tbg_zero_field_hf_source_receipt(
            hf_mode="full",
            beta=1.0,
            v0=state.v0 + 1.0e-6,
            solution=solution,
            screened_block_bundle=bundle,
        )

    override = np.zeros_like(state.density)
    for module_name, runner_name, init_mode in (
        ("mean_field.systems.tbg.zero_field._hf_full", "run_full_hartree_fock", "flavor"),
        ("mean_field.systems.tbg.zero_field._hf_restricted", "run_restricted_hartree_fock", "educated"),
    ):
        module = importlib.import_module(module_name)
        with pytest.raises(ValueError, match="typed resume trajectory is not implemented"):
            getattr(module, runner_name)(
                RestrictedHartreeFockState.from_bm_solution(solution, nu=0.0),
                bundle.screened_blocks,
                solution.lattice_kvec,
                solution.params,
                init_mode=init_mode,
                interaction_spec=_INTERACTION_SPEC,
                source_solution=solution,
                screened_block_bundle=bundle,
                initial_density=override,
            )


def test_restricted_complex_two_band_projector_matches_full_interaction_and_energy() -> None:
    hamiltonian = np.asarray(
        [[0.35, 0.62 + 0.47j], [0.62 - 0.47j, -0.28]],
        dtype=np.complex128,
    )[:, :, None]
    sigma_z = np.zeros_like(hamiltonian)
    restricted_density, restricted_energies, _, _ = build_restricted_density_from_hamiltonian(
        hamiltonian,
        sigma_z,
        nu=0.0,
        n_spin=1,
        n_eta=1,
        n_band=2,
    )
    full_density, full_energies, _, _ = build_full_density_from_hamiltonian(
        hamiltonian,
        sigma_z,
        nu=0.0,
    )

    _, eigenvectors = np.linalg.eigh(hamiltonian[:, :, 0])
    occupied = eigenvectors[:, :1]
    conventional_projector = (occupied @ occupied.conj().T)[:, :, None]
    expected_stored = conventional_projector_to_stored_density(conventional_projector)
    wrong_ket_oriented_delta = conventional_projector - 0.5 * np.eye(2, dtype=np.complex128)[:, :, None]
    assert abs(expected_stored[0, 1, 0].imag) > 1.0e-6
    assert not np.allclose(expected_stored, wrong_ket_oriented_delta)
    np.testing.assert_allclose(restricted_density, expected_stored, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(full_density, expected_stored, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(restricted_energies, full_energies, atol=1.0e-14, rtol=0.0)

    overlap_matrix = np.asarray(
        [[1.0 + 0.1j, 0.23 + 0.41j], [-0.17 + 0.29j, 0.74 - 0.08j]],
        dtype=np.complex128,
    )
    overlap = overlap_matrix[:, None, :, None]
    blocks = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j]),
        overlaps={(0, 0): overlap},
        fock_screening={(0, 0): np.asarray([[1.37]], dtype=float)},
    )
    restricted_interaction = build_projected_interaction_hamiltonian(
        restricted_density,
        blocks,
        v0=2.3,
        beta=0.8,
        use_numba=False,
    )
    full_interaction = build_projected_interaction_hamiltonian(
        full_density,
        blocks,
        v0=2.3,
        beta=0.8,
        use_numba=False,
    )
    wrong_interaction = build_projected_interaction_hamiltonian(
        wrong_ket_oriented_delta,
        blocks,
        v0=2.3,
        beta=0.8,
        use_numba=False,
    )
    np.testing.assert_allclose(restricted_interaction, full_interaction, atol=1.0e-14, rtol=0.0)
    assert not np.allclose(full_interaction, wrong_interaction)

    restricted_energy = compute_hf_energy(restricted_interaction, hamiltonian, restricted_density)
    full_energy = compute_hf_energy(full_interaction, hamiltonian, full_density)
    wrong_energy = compute_hf_energy(wrong_interaction, hamiltonian, wrong_ket_oriented_delta)
    assert restricted_energy == pytest.approx(full_energy, abs=1.0e-14, rel=0.0)
    assert not np.isclose(full_energy, wrong_energy, atol=1.0e-8, rtol=0.0)


def test_stored_and_conventional_density_conversions_are_explicit() -> None:
    vector = np.asarray([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    projector = (vector[:, None] @ vector[None, :].conjugate())[:, :, None]
    stored = conventional_projector_to_stored_density(projector)
    np.testing.assert_allclose(stored_density_to_conventional_projector(stored), projector, atol=0.0)

    tangent = np.asarray([[0.0, 0.3 + 0.2j], [-0.1j, 0.0]], dtype=np.complex128)[:, :, None]
    np.testing.assert_allclose(
        stored_tangent_to_conventional_tangent(conventional_tangent_to_stored_tangent(tangent)),
        tangent,
        atol=0.0,
    )


def test_q0_pair_order_is_k_then_hole_then_particle() -> None:
    energies = np.arange(8, dtype=float).reshape((4, 2), order="F")
    occupied = np.asarray(
        [[True, False], [False, True], [True, False], [False, True]],
        dtype=bool,
    )
    projector = np.zeros((4, 4, 2), dtype=np.complex128)
    for ik in range(2):
        projector[:, :, ik] = np.diag(occupied[:, ik].astype(float))
    orbitals = TBGZeroFieldTDHFOrbitals(
        energies=energies,
        eigenvectors=np.repeat(np.eye(4, dtype=np.complex128)[:, :, None], 2, axis=2),
        occupied_mask=occupied,
        conventional_projector=projector,
        mu=0.0,
        occupation_residuals=TBGZeroFieldTDHFOccupationResiduals(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0e-12,
            1.0e-12,
        ),
        source_hamiltonian_sha256="0" * 64,
        source_density_sha256="1" * 64,
    )

    pairs = build_tbg_zero_field_tdhf_q0_pairs(orbitals)
    decoded = [
        (*orbitals.decode_global_index(pair.particle), *orbitals.decode_global_index(pair.hole))
        for pair in pairs
    ]
    assert decoded == [
        (1, 0, 0, 0),
        (3, 0, 0, 0),
        (1, 0, 2, 0),
        (3, 0, 2, 0),
        (0, 1, 1, 1),
        (2, 1, 1, 1),
        (0, 1, 3, 1),
        (2, 1, 3, 1),
    ]


def test_orbitals_use_stored_projector_and_reject_noninteger_occupations() -> None:
    context = _two_k_context()
    orbitals = build_tbg_zero_field_tdhf_orbitals(context.run)
    np.testing.assert_array_equal(orbitals.occupied_mask, [[True, True], [False, False]])
    np.testing.assert_allclose(
        orbitals.conventional_projector,
        stored_density_to_conventional_projector(context.run.state.density),
    )

    context.run.state.density[0, 0, 0] -= 0.2
    with pytest.raises(ValueError, match="0/1 projector"):
        build_tbg_zero_field_tdhf_orbitals(context.run)


def test_vectorized_q0_a_b_structure_and_tangent_column_parity() -> None:
    context = _two_k_context()
    orbitals = build_tbg_zero_field_tdhf_orbitals(context.run)
    matrices = build_tbg_zero_field_tdhf_q0_matrices(context, orbitals)

    assert [(pair.particle_momentum, pair.hole_momentum) for pair in matrices.pairs] == [(0, 0), (1, 1)]
    np.testing.assert_allclose(matrices.A, matrices.A.conjugate().T, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(matrices.B, matrices.B.T, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        matrices.L,
        np.block([[matrices.A, matrices.B], [-matrices.B.conjugate(), -matrices.A.conjugate()]]),
        rtol=0.0,
        atol=0.0,
    )
    assert matrices.structure.ok
    assert (
        orbitals.occupation_residuals.production_max_tolerance
        == TBG_ZERO_FIELD_TDHF_PRODUCTION_MAX_PROJECTOR_TOLERANCE
    )
    assert (
        matrices.production_max_structure_tolerance
        == TBG_ZERO_FIELD_TDHF_PRODUCTION_MAX_STRUCTURE_TOLERANCE
    )
    assert (
        context.authorization_diagnostics["production_max_closure_tolerance_mev"]
        == TBG_ZERO_FIELD_TDHF_PRODUCTION_MAX_CLOSURE_TOLERANCE_MEV
    )

    parity = validate_tbg_zero_field_tdhf_tangent_columns(
        context,
        orbitals,
        matrices,
        columns=(0, 1),
        tolerance=1.0e-12,
    )
    assert parity.ok
    assert parity.max_a_residual <= 1.0e-12
    assert parity.max_b_residual <= 1.0e-12


    assert (
        parity.production_max_tolerance
        == TBG_ZERO_FIELD_TDHF_PRODUCTION_MAX_TANGENT_TOLERANCE
    )

def test_context_rejects_oversized_production_closure_tolerance() -> None:
    solution, run = _two_k_source()
    receipt = run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    with pytest.raises(ValueError, match="closure_tolerance_mev.*production maximum"):
        TBGZeroFieldTDHFContext(
            grid_solution=solution,
            run=run,
            beta=receipt.beta,
            provenance=_provenance("oversized-closure", receipt),
            allow_diagnostic_source=True,
            closure_tolerance=(
                2.0 * TBG_ZERO_FIELD_TDHF_PRODUCTION_MAX_CLOSURE_TOLERANCE_MEV
            ),
        )

def test_orbitals_reject_oversized_production_projector_tolerance() -> None:
    _solution, run = _two_k_source()
    with pytest.raises(ValueError, match="projector_tolerance.*production maximum"):
        build_tbg_zero_field_tdhf_orbitals(
            run,
            projector_tolerance=(
                2.0 * TBG_ZERO_FIELD_TDHF_PRODUCTION_MAX_PROJECTOR_TOLERANCE
            ),
        )

def test_a_b_structure_rejects_oversized_production_tolerance() -> None:
    context = _two_k_context()
    orbitals = build_tbg_zero_field_tdhf_orbitals(context.run)
    with pytest.raises(ValueError, match="structure_tolerance.*production maximum"):
        build_tbg_zero_field_tdhf_q0_matrices(
            context,
            orbitals,
            structure_tolerance=(
                2.0 * TBG_ZERO_FIELD_TDHF_PRODUCTION_MAX_STRUCTURE_TOLERANCE
            ),
        )

def test_tangent_parity_rejects_oversized_production_tolerance() -> None:
    context = _two_k_context()
    orbitals = build_tbg_zero_field_tdhf_orbitals(context.run)
    matrices = build_tbg_zero_field_tdhf_q0_matrices(context, orbitals)
    with pytest.raises(ValueError, match="tangent_tolerance.*production maximum"):
        validate_tbg_zero_field_tdhf_tangent_columns(
            context,
            orbitals,
            matrices,
            columns=(0,),
            tolerance=(
                2.0 * TBG_ZERO_FIELD_TDHF_PRODUCTION_MAX_TANGENT_TOLERANCE
            ),
        )

def test_dense_complex_fock_only_nontrivial_unitary_tangent_parity() -> None:
    nt = 4
    nk = 1
    beta = 0.7
    v0 = 0.9
    projector = np.diag([1.0, 1.0, 0.0, 0.0]).astype(np.complex128)[:, :, None]
    density = conventional_projector_to_stored_density(projector)
    identity2 = np.eye(2, dtype=np.complex128)

    def unitary2(angle: float, phase: float) -> np.ndarray:
        return np.asarray(
            [
                [np.cos(angle), np.sin(angle) * np.exp(1.0j * phase)],
                [-np.sin(angle) * np.exp(-1.0j * phase), np.cos(angle)],
            ],
            dtype=np.complex128,
        )

    cross_block = 0.73 * unitary2(0.41, 0.37)
    dense_hermitian_overlap = np.block(
        [
            [0.61 * identity2, cross_block],
            [cross_block.conjugate().T, -0.27 * identity2],
        ]
    )
    centered_sign = np.diag([1.0, 1.0, -1.0, -1.0])
    overlap_matrices = (
        dense_hermitian_overlap,
        centered_sign @ dense_hermitian_overlap @ centered_sign,
    )
    assert np.max(np.abs(dense_hermitian_overlap.imag)) > 0.1
    shifts = ((0, 0), (1, 0))
    overlaps: dict[tuple[int, int], np.ndarray] = {}
    for shift, matrix in zip(shifts, overlap_matrices, strict=True):
        overlap = np.empty((nt, nk, nt, nk), dtype=np.complex128)
        overlap[:, 0, :, 0] = matrix
        overlaps[shift] = overlap
    blocks = HFOverlapBlockSet(
        shifts=shifts,
        gvecs=np.asarray([0.0 + 0.0j, 0.2 - 0.1j], dtype=np.complex128),
        overlaps=overlaps,
        diagonal_overlaps={},
        hartree_screening={},
        fock_screening={
            shift: np.asarray([[weight]], dtype=float)
            for shift, weight in zip(shifts, (0.8, 0.8), strict=True)
        },
    )
    interaction = build_projected_interaction_hamiltonian(
        density,
        blocks,
        v0=v0,
        beta=beta,
    )
    hamiltonian = np.diag([-0.7, -0.7, 0.9, 0.9]).astype(np.complex128)[:, :, None]
    h0 = hamiltonian - interaction
    np.testing.assert_allclose(
        h0,
        np.einsum("iik,ij->ijk", h0, np.eye(nt)),
        rtol=0.0,
        atol=1.0e-13,
    )
    solution = _bm_solution(n_spin=1, n_eta=1, n_band=nt, nk=nk, h0=h0)
    state = RestrictedHartreeFockState(
        h0=h0,
        sigma_z=np.zeros_like(hamiltonian),
        density=density,
        hamiltonian=hamiltonian,
        energies=np.asarray([[-0.7], [-0.7], [0.9], [0.9]], dtype=float),
        sigma_ztauz=np.zeros((nt, nk), dtype=float),
        nu=0.0,
        v0=v0,
        n_spin=1,
        n_eta=1,
        n_band=nt,
        diagnostics={"beta": beta},
        interaction_spec=_INTERACTION_SPEC,
    )
    state.hf_source_receipt = build_tbg_zero_field_diagnostic_hf_source_receipt(
        hf_mode="full",
        beta=beta,
        v0=state.v0,
        lattice_kvec=solution.lattice_kvec,
        overlap_blocks=blocks,
    )
    run = RestrictedHartreeFockRun(
        state=state,
        overlap_blocks=blocks,
        iter_energy=np.asarray([-1.0], dtype=float),
        iter_err=np.asarray([0.0], dtype=float),
        iter_oda=np.asarray([1.0], dtype=float),
        init_mode="dense-complex-fock-only",
        seed=0,
        converged=True,
        exit_reason="synthetic-converged",
    )
    context = _context_from_source(solution, run, label="dense-complex-fock-only")
    base_orbitals = build_tbg_zero_field_tdhf_orbitals(run)
    nontrivial_unitary = np.block(
        [
            [unitary2(0.33, 0.61), np.zeros((2, 2), dtype=np.complex128)],
            [np.zeros((2, 2), dtype=np.complex128), unitary2(0.27, -0.47)],
        ]
    )[:, :, None]
    assert np.max(np.abs(nontrivial_unitary.imag)) > 0.1
    orbitals = replace(base_orbitals, eigenvectors=nontrivial_unitary)
    matrices = build_tbg_zero_field_tdhf_q0_matrices(context, orbitals)
    parity = validate_tbg_zero_field_tdhf_tangent_columns(
        context,
        orbitals,
        matrices,
        columns=tuple(range(len(matrices.pairs))),
        tolerance=2.0e-12,
    )
    assert parity.ok


def test_flavor_symmetric_toy_has_exact_su2_spin_rotation_goldstone() -> None:
    nt = 2
    nk = 1
    interaction_strength = 1.0
    projector = np.diag([1.0, 0.0]).astype(np.complex128)[:, :, None]
    hamiltonian = np.diag([-0.5, 0.5]).astype(np.complex128)[:, :, None]
    overlap = np.eye(nt, dtype=np.complex128)[:, None, :, None]
    blocks = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        overlaps={(0, 0): overlap},
        diagonal_overlaps={(0, 0): np.eye(nt, dtype=np.complex128)[:, :, None]},
        hartree_screening={(0, 0): 0.7},
        fock_screening={(0, 0): np.ones((1, 1), dtype=float)},
    )
    solution = _bm_solution(n_spin=2, n_eta=1, n_band=1, nk=1)
    state = RestrictedHartreeFockState(
        h0=np.zeros_like(hamiltonian),
        sigma_z=np.zeros_like(hamiltonian),
        density=conventional_projector_to_stored_density(projector),
        hamiltonian=hamiltonian,
        energies=np.asarray([[-0.5], [0.5]], dtype=float),
        sigma_ztauz=np.zeros((nt, nk), dtype=float),
        nu=0.0,
        v0=interaction_strength,
        n_spin=2,
        n_eta=1,
        n_band=1,
        diagnostics={"beta": 1.0},
        interaction_spec=_INTERACTION_SPEC,
    )
    state.hf_source_receipt = build_tbg_zero_field_diagnostic_hf_source_receipt(
        hf_mode="full",
        beta=1.0,
        v0=state.v0,
        lattice_kvec=solution.lattice_kvec,
        overlap_blocks=blocks,
    )
    run = RestrictedHartreeFockRun(
        state=state,
        overlap_blocks=blocks,
        iter_energy=np.asarray([-0.25], dtype=float),
        iter_err=np.asarray([0.0], dtype=float),
        iter_oda=np.asarray([1.0], dtype=float),
        init_mode="spin-polarized-toy",
        seed=0,
        converged=True,
        exit_reason="synthetic-converged",
    )
    context = _context_from_source(solution, run, label="su2-goldstone")

    matrices = build_tbg_zero_field_tdhf_q0_matrices(context)
    assert len(matrices.pairs) == 1
    np.testing.assert_allclose(matrices.A, np.zeros((1, 1)), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(matrices.B, np.zeros((1, 1)), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.linalg.eigvals(matrices.L), np.zeros(2), rtol=0.0, atol=1.0e-14)


def test_context_rejects_unscreened_kernel_inventory() -> None:
    _solution, source_run = _two_k_source()
    blocks = replace(
        source_run.overlap_blocks,
        diagonal_overlaps={},
        hartree_screening={},
        fock_screening={},
    )
    solution, run = _source_with_overlap_blocks(blocks)
    with pytest.raises(ValueError, match="unscreened"):
        _context_from_source(solution, run)


@pytest.mark.parametrize("active_kind", ["hartree", "fock"])
def test_context_accepts_independently_active_exact_functionals(active_kind: str) -> None:
    _solution, source_run = _two_k_source()
    if active_kind == "hartree":
        blocks = replace(source_run.overlap_blocks, fock_screening={})
    else:
        blocks = replace(
            source_run.overlap_blocks,
            diagonal_overlaps={},
            hartree_screening={},
        )
    solution, run = _source_with_overlap_blocks(blocks)
    context = _context_from_source(solution, run, label=f"{active_kind}-only")
    assert set(context.overlap_blocks.hartree_screening) == (
        {(0, 0)} if active_kind == "hartree" else set()
    )
    assert set(context.overlap_blocks.fock_screening) == (
        {(0, 0)} if active_kind == "fock" else set()
    )


def test_context_rejects_bm_h0_mismatch() -> None:
    solution, run = _two_k_source()
    run.state.h0[0, 0, 0] += 1.0e-4
    with pytest.raises(ValueError, match="BM source h0 does not match"):
        _context_from_source(solution, run)


def test_context_requires_explicit_diagnostic_source_override() -> None:
    solution, run = _two_k_source()
    receipt = run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    with pytest.raises(ValueError, match="allow_diagnostic_source=True"):
        TBGZeroFieldTDHFContext(
            grid_solution=solution,
            run=run,
            beta=receipt.beta,
            provenance=_provenance("diagnostic-default-refusal", receipt),
            closure_tolerance=1.0e-12,
        )
    context = TBGZeroFieldTDHFContext(
        grid_solution=solution,
        run=run,
        beta=receipt.beta,
        provenance=_provenance("diagnostic-explicit-override", receipt),
        allow_diagnostic_source=True,
        closure_tolerance=1.0e-12,
    )
    assert context.allow_diagnostic_source is True
    assert context.source_classification == "diagnostic_non_production"
    assert context.is_production_source is False


def _typed_full_source() -> tuple[BMSolution, RestrictedHartreeFockRun]:
    """Use the real full runner; the empty filling is a one-step fixed point."""

    solution = _typed_torus_solution()
    bundle = build_tbg_zero_field_screened_block_bundle(
        solution,
        interaction_spec=_INTERACTION_SPEC,
        overlap_lg=7,
    )
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=-4.0)
    state.diagnostics["overlap_lg"] = 7.0
    run = run_full_hartree_fock(
        state,
        bundle.screened_blocks,
        solution.lattice_kvec,
        solution.params,
        init_mode="bm",
        seed=0,
        beta=1.0,
        max_iter=1,
        oda_stall_threshold=1.0e-3,
        interaction_spec=_INTERACTION_SPEC,
        source_solution=solution,
        screened_block_bundle=bundle,
    )
    assert run.converged
    assert isinstance(run.provenance, TBGZeroFieldHFRunProvenance)
    return solution, run


def _typed_context(
    solution: BMSolution,
    run: RestrictedHartreeFockRun,
    *,
    label: str,
) -> TBGZeroFieldTDHFContext:
    receipt = run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    return TBGZeroFieldTDHFContext(
        grid_solution=solution,
        run=run,
        beta=1.0,
        provenance=_provenance(
            label,
            receipt,
            run_provenance=run.provenance,
        ),
        closure_tolerance=1.0e-12,
    )


def test_typed_context_requires_complete_hf_run_provenance() -> None:
    solution, run = _typed_full_source()
    with pytest.raises(ValueError, match="requires TBGZeroFieldHFRunProvenance"):
        _typed_context(
            solution,
            replace(run, provenance=None, _production_issuer=None),
            label="typed-missing-run-provenance",
        )


@pytest.mark.parametrize(
    "field_name",
    ["density", "hamiltonian", "energies", "mu", "sigma_ztauz", "diagnostics"],
)
def test_typed_context_rejects_pre_context_final_state_mutation(
    field_name: str,
) -> None:
    solution, run = _typed_full_source()
    state = run.state
    if field_name == "mu":
        state.mu += 1.0e-9
    elif field_name == "diagnostics":
        state.diagnostics["final_raw_norm"] += 1.0e-9
    else:
        getattr(state, field_name).flat[0] += 1.0e-9

    with pytest.raises(ValueError, match="final state hash does not match"):
        _typed_context(
            solution,
            run,
            label=f"pre-context-{field_name}-mutation",
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "hf_mode",
        "beta",
        "nu",
        "precision",
        "oda_stall_threshold",
        "requested_max_iterations",
        "seed",
        "normalized_init_mode",
        "typed_receipt_fingerprint",
        "interaction_spec_fingerprint",
        "bm_generation_fingerprint",
        "mesh_fingerprint",
    ],
)
def test_solver_issued_hf_run_provenance_rejects_forged_replace(field_name: str) -> None:
    _solution, run = _typed_full_source()
    provenance = run.provenance
    assert isinstance(provenance, TBGZeroFieldHFRunProvenance)
    mutations = {
        "hf_mode": "restricted",
        "beta": provenance.beta + 0.5,
        "nu": provenance.nu + 0.25,
        "precision": provenance.precision * 2.0,
        "oda_stall_threshold": provenance.oda_stall_threshold * 2.0,
        "requested_max_iterations": provenance.requested_max_iterations + 1,
        "seed": provenance.seed + 1,
        "normalized_init_mode": "vp",
        "typed_receipt_fingerprint": "0" * 64,
        "interaction_spec_fingerprint": "0" * 64,
        "bm_generation_fingerprint": "0" * 64,
        "mesh_fingerprint": "0" * 64,
    }
    with pytest.raises(ValueError, match="solver-issued"):
        replace(provenance, **{field_name: mutations[field_name]})


def test_manual_hf_run_cannot_reuse_solver_issued_provenance() -> None:
    _solution, run = _typed_full_source()
    provenance = run.provenance
    assert isinstance(provenance, TBGZeroFieldHFRunProvenance)
    with pytest.raises(ValueError, match="Manually constructed HF runs"):
        RestrictedHartreeFockRun(
            state=run.state,
            overlap_blocks=run.overlap_blocks,
            screened_block_bundle=run.screened_block_bundle,
            provenance=provenance,
            iter_energy=run.iter_energy,
            iter_err=run.iter_err,
            iter_oda=run.iter_oda,
            init_mode=run.init_mode,
            seed=run.seed,
            converged=run.converged,
            exit_reason=run.exit_reason,
        )

def test_typed_context_live_hash_binds_hf_run_provenance_fingerprint() -> None:
    solution, run = _typed_full_source()
    context = _typed_context(solution, run, label="typed-live-provenance")
    provenance = run.provenance
    assert isinstance(provenance, TBGZeroFieldHFRunProvenance)
    assert context.source_classification == "typed_production"
    assert context.is_production_source is True
    object.__setattr__(provenance, "seed", provenance.seed + 1)
    with pytest.raises(ValueError, match="live HF source changed"):
        context.build_interaction_hamiltonian(np.zeros_like(run.state.density))


@pytest.mark.parametrize("history_name", ["iter_energy", "iter_err", "iter_oda"])
def test_typed_context_live_hash_binds_solver_histories(history_name: str) -> None:
    solution, run = _typed_full_source()
    context = _typed_context(solution, run, label=f"typed-live-{history_name}")
    changed = np.asarray(getattr(run, history_name), dtype=float).copy()
    changed[0] += 1.0
    object.__setattr__(run, history_name, changed)
    with pytest.raises(ValueError, match="live HF source changed"):
        context.build_interaction_hamiltonian(np.zeros_like(run.state.density))

def test_typed_context_recomputes_central_reference_hash() -> None:
    solution, run = _typed_full_source()
    context = _typed_context(solution, run, label="typed-reference")
    assert context.is_production_source is True
    state = run.state
    state.hf_source_receipt = replace(
        state.hf_source_receipt,
        reference_projector_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="reference projector hash mismatch"):
        _typed_context(solution, run, label="typed-reference-tamper")


def test_typed_context_rejects_mismatched_expected_hf_run_fingerprint() -> None:
    solution, run = _typed_full_source()
    receipt = run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    provenance = replace(
        _provenance(
            "wrong-expected-run-fingerprint",
            receipt,
            run_provenance=run.provenance,
        ),
        expected_hf_run_provenance_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="expected HF run provenance fingerprint"):
        TBGZeroFieldTDHFContext(
            grid_solution=solution,
            run=run,
            beta=1.0,
            provenance=provenance,
            closure_tolerance=1.0e-12,
        )

def test_context_requires_receipt_and_matching_provenance_fingerprint() -> None:
    solution, run = _two_k_source()
    receipt = run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    provenance = _provenance("missing-typed-receipt", receipt)
    run.state.hf_source_receipt = None
    with pytest.raises(ValueError, match="TBGZeroFieldHFSourceReceipt"):
        TBGZeroFieldTDHFContext(
            grid_solution=solution,
            run=run,
            beta=receipt.beta,
            provenance=provenance,
            closure_tolerance=1.0e-12,
        )

    solution, run = _two_k_source()
    receipt = run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    provenance = replace(
        _provenance("wrong-expected-fingerprint", receipt),
        expected_hf_source_receipt_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="provenance expected fingerprint"):
        TBGZeroFieldTDHFContext(
            grid_solution=solution,
            run=run,
            beta=receipt.beta,
            provenance=provenance,
            allow_diagnostic_source=True,
            closure_tolerance=1.0e-12,
        )


def test_context_rejects_non_full_hf_mode() -> None:
    solution, run = _two_k_source()
    run.state.hf_source_receipt = replace(
        run.state.hf_source_receipt,
        hf_mode="restricted",
    )
    with pytest.raises(ValueError, match="hf_mode.*full"):
        _context_from_source(solution, run)


def test_context_rejects_wrong_lattice_kvec_source_receipt() -> None:
    solution, run = _two_k_source()
    run.state.hf_source_receipt = replace(
        run.state.hf_source_receipt,
        lattice_kvec_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="lattice_kvec"):
        _context_from_source(solution, run)


def test_context_rejects_source_hamiltonian_closure_failure() -> None:
    solution, run = _two_k_source()
    run.state.hamiltonian[0, 0, 0] += 1.0e-4
    with pytest.raises(ValueError, match="not closed by h0 plus the saved interaction"):
        _context_from_source(solution, run)


def test_context_rejects_pre_context_v0_substitution_even_if_hamiltonian_is_reclosed() -> None:
    solution, run = _two_k_source()
    state = run.state
    state.v0 += 0.125
    state.hamiltonian[:, :, :] = state.h0 + build_projected_interaction_hamiltonian(
        state.density,
        run.overlap_blocks,
        v0=state.v0,
        beta=float(state.diagnostics["beta"]),
    )

    with pytest.raises(ValueError, match=r"receipt v0 does not match run\.state\.v0 exactly"):
        _context_from_source(solution, run, label="pre-context-v0-substitution")


@pytest.mark.parametrize(
    "mutated_source",
    ["hamiltonian", "density", "fock_kernel", "v0"],
)
def test_matrix_and_parity_calls_reject_post_context_live_source_mutation_with_rebuilt_orbitals(
    mutated_source: str,
) -> None:
    context = _two_k_context()
    original_orbitals = build_tbg_zero_field_tdhf_orbitals(context.run)
    matrices = build_tbg_zero_field_tdhf_q0_matrices(context, original_orbitals)
    state = context.run.state
    if mutated_source == "hamiltonian":
        state.hamiltonian[0, 0, 0] += 1.0e-6
    elif mutated_source == "density":
        state.density[0, 0, 0] = -0.5
        state.density[1, 1, 0] = 0.5
    elif mutated_source == "fock_kernel":
        context.overlap_blocks.fock_screening[(0, 0)][0, 0] += 1.0e-6
    else:
        state.v0 += 1.0e-6
    rebuilt_orbitals = build_tbg_zero_field_tdhf_orbitals(context.run)

    with pytest.raises(ValueError, match="live HF source changed"):
        build_tbg_zero_field_tdhf_q0_matrices(context, rebuilt_orbitals)
    with pytest.raises(ValueError, match="live HF source changed"):
        validate_tbg_zero_field_tdhf_tangent_columns(
            context,
            rebuilt_orbitals,
            matrices,
            columns=(0,),
        )


# ---------------------------------------------------------------------------
# Companion-faithful plane-wave and interaction geometry (bookkeeping only)
# ---------------------------------------------------------------------------


def test_companion_actual_unstrained_ng4_ng5_n10_reference_oracle() -> None:
    """Check implementation bookkeeping against an independent Test001 oracle.

    The hard-coded oracle was generated from the pinned reference
    ``singleParticle.gen_RLVs`` output plus literal ``gen_coeff``,
    ``gen_moire_hamiltonian``, and ``gen_interaction`` loops on unstrained
    Test001.  It did not use the new companion-geometry module.  Together with
    the synthetic checks below, this is one actual Test001 oracle plus focused
    cases, not 10 oracle tests.  This is an implementation checkpoint only,
    not validation of HF or TDHF physics.
    """

    assert TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY == "reference/TBG-HF"
    assert TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT == (
        "0d2a3d742aa901fa45ce46690c1385887165f58c"
    )
    assert TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE == "singleParticle.py"
    assert TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256 == (
        "a050fa545c4d399b227a178bcc4705a110bd7962edcb9e1f69e300b5e1a3e43b"
    )
    assert TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_REFERENCE_FUNCTION == "gen_coeff"
    assert TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_REFERENCE_LINES == "132-175"
    assert TBG_ZERO_FIELD_COMPANION_HOPPING_REFERENCE_FUNCTION == (
        "gen_moire_hamiltonian"
    )
    assert TBG_ZERO_FIELD_COMPANION_HOPPING_REFERENCE_LINES == "101-109"
    assert TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_FUNCTION == "gen_interaction"
    assert TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_LINES == "220-255"

    b1 = complex(-1.7320508075688767, 0.0)
    b2 = complex(0.8660254037844384, -1.4999999999999998)
    plane_wave_spec = TBGZeroFieldCompanionPlaneWaveSpec(
        Ng1=4,
        Ng2=4,
        b1=b1,
        b2=b2,
    )
    interaction_spec = TBGZeroFieldCompanionInteractionSpec(
        NG1=5,
        NG2=5,
        b1=b1,
        b2=b2,
    )

    assert plane_wave_spec.b1 == complex(-1.7320508075688767, 0.0)
    assert plane_wave_spec.b2 == complex(0.8660254037844384, -1.4999999999999998)
    assert plane_wave_spec.radius == 4.999999999999999
    assert interaction_spec.b1 == complex(-1.7320508075688767, 0.0)
    assert interaction_spec.b2 == complex(0.8660254037844384, -1.4999999999999998)
    assert interaction_spec.radius == pytest.approx(
        7.4999999999999964,
        rel=0.0,
        abs=1.0e-14,
    )

    expected_plane_geometry = {
        (0, 0): (
            108,
            "a9aea3f1e3b75df38352eae22c6cc70d416c0d65dfc736b5462fee3e7a53b379",
            169,
            72,
            "6335d6c49ab7d2d7db8c6a8bdce8c86f70dac4ed3554fccf60c3bb95f682bb4d",
        ),
        (5, 0): (
            124,
            "02b5c1b816af2bc1974231c74652cf69381012a1134384b237a0a548d7734cec",
            169,
            83,
            "b98f0346050ff8626e1aea55fbb195381eee0593aecdb74ac516a86580236575",
        ),
        (3, 7): (
            118,
            "b2c3661f0fc141e268cc4363cedf555ce33eb39238eb45fac67a02d3cbca4f70",
            169,
            77,
            "a3dd4818d880ed1167a4f4d7bf9984a8a21d9aa36879b8b7d5dd05d0d100e2f6",
        ),
        (9, 9): (
            112,
            "2ae0655bb1dfd53233cd790fbba4e2e11960ef45380a61f431092daa7bd47f17",
            169,
            74,
            "8efebb5ff626a46fb178d8946b74928111e96f3809ed5479420a71cceb97148a",
        ),
    }
    for (ik1, ik2), expected in expected_plane_geometry.items():
        geometry = build_tbg_zero_field_companion_plane_wave_geometry(
            plane_wave_spec,
            N1=10,
            N2=10,
            ik1=ik1,
            ik2=ik2,
            stau=1,
        )
        assert (
            geometry.basis_count,
            geometry.sub_index_sha256,
            geometry.parent_hopping_edge_count,
            geometry.active_hopping_edge_count,
            geometry.hopping_edges_sha256,
        ) == expected

    interaction_geometry = build_tbg_zero_field_companion_interaction_geometry(
        interaction_spec,
        N1=10,
        N2=10,
    )
    assert interaction_geometry.total_count == 10_000
    assert interaction_geometry.active_count == 6_787
    assert interaction_geometry.active_mask_sha256 == (
        "71bef75a6476cb1c80aae0352ab5f1dab2b2569ba484bafab409ff0ed5103b44"
    )
    assert interaction_geometry.labels_sha256 == (
        "060c05bde942742db7b1421833e8ddb78927d1c9ab1a3fe9d5c93c05d27b1aac"
    )


def test_companion_plane_wave_geometry_has_exact_subindex_and_zero_fill_edges() -> None:
    spec = TBGZeroFieldCompanionPlaneWaveSpec(
        Ng1=2,
        Ng2=2,
        b1=1.0 + 0.0j,
        b2=0.0 + 1.0j,
    )
    geometry = build_tbg_zero_field_companion_plane_wave_geometry(
        spec,
        N1=1,
        N2=1,
        ik1=0,
        ik2=0,
        stau=1,
    )

    assert spec.g1_labels == (-2, -1, 0, 1)
    assert spec.g2_labels == (-2, -1, 0, 1)
    assert spec.radius == pytest.approx(4.0 / 3.0, rel=0.0, abs=1.0e-15)
    assert geometry.sub_index == (
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
    )
    assert geometry.basis_count == 16
    assert geometry.parent_hopping_edge_count == 37
    assert geometry.active_hopping_edge_count == 8
    assert tuple(
        (edge.channel, edge.source_label, edge.target_label)
        for edge in geometry.active_hopping_edges
    ) == (
        ("T2", (-1, -1), (0, 0)),
        ("T3", (-1, -1), (-1, 0)),
        ("T1", (-1, 0), (-1, 0)),
        ("T2", (-1, 0), (0, 1)),
        ("T3", (-1, 0), (-1, 1)),
        ("T3", (0, -1), (0, 0)),
        ("T1", (0, 0), (0, 0)),
        ("T3", (0, 0), (0, 1)),
    )
    assert all(
        edge.source_companion_indices[0] % 4 == 2
        and edge.target_companion_indices[0] % 4 == 0
        for edge in geometry.hopping_edges
    )
    assert not any(
        edge.channel == "T2" and (edge.source_label[0] == 1 or edge.source_label[1] == 1)
        for edge in geometry.hopping_edges
    )
    assert not any(
        edge.channel == "T3" and edge.source_label[1] == 1
        for edge in geometry.hopping_edges
    )
    assert spec.to_metadata()["fingerprint"] == spec.fingerprint
    assert geometry.to_metadata()["fingerprint"] == geometry.fingerprint
    assert geometry.to_metadata()["spec_fingerprint"] == spec.fingerprint


def test_companion_plane_wave_radius_preserves_literal_unequal_ng_source_behavior() -> None:
    spec = TBGZeroFieldCompanionPlaneWaveSpec(
        Ng1=2,
        Ng2=1,
        b1=1.0 + 0.0j,
        b2=0.0 + 1.0j,
    )

    assert spec.g1_labels == (-2, -1, 0, 1)
    assert spec.g2_labels == (-1, 0)
    assert spec.radius == pytest.approx(4.0 / 3.0, rel=0.0, abs=1.0e-15)
    assert spec.cutoff_convention == (
        "rectangular_Ng1_Ng2_labels_but_pinned_gen_coeff_radius_uses_Ng1_in_all_four_"
        "RX_RY_terms_including_b2_terms;strict_abs_Q_lt_radius_minus_margin"
    )


def test_companion_plane_wave_geometry_rejects_direct_k_prime_cutoff_reuse() -> None:
    spec = TBGZeroFieldCompanionPlaneWaveSpec(
        Ng1=1,
        Ng2=1,
        b1=1.0 + 0.0j,
        b2=0.0 + 1.0j,
    )

    with pytest.raises(
        ValueError,
        match=(
            "K' must come from the separate companion time-reversal "
            "mesh/wrapped-G construction, not direct cutoff reuse"
        ),
    ):
        build_tbg_zero_field_companion_plane_wave_geometry(
            spec,
            N1=1,
            N2=1,
            ik1=0,
            ik2=0,
            stau=-1,
        )


def test_companion_interaction_geometry_has_exact_nested_mask_and_digest() -> None:
    spec = TBGZeroFieldCompanionInteractionSpec(
        NG1=1,
        NG2=1,
        b1=1.0 + 0.0j,
        b2=0.0 + 1.0j,
    )
    geometry = build_tbg_zero_field_companion_interaction_geometry(spec, N1=2, N2=2)

    assert spec.radius == pytest.approx(1.0, rel=0.0, abs=1.0e-15)
    assert geometry.labels == (
        (0, 0, -1, -1),
        (0, 0, -1, 0),
        (0, 0, 0, -1),
        (0, 0, 0, 0),
        (0, 1, -1, -1),
        (0, 1, -1, 0),
        (0, 1, 0, -1),
        (0, 1, 0, 0),
        (1, 0, -1, -1),
        (1, 0, -1, 0),
        (1, 0, 0, -1),
        (1, 0, 0, 0),
        (1, 1, -1, -1),
        (1, 1, -1, 0),
        (1, 1, 0, -1),
        (1, 1, 0, 0),
    )
    assert geometry.active_indices == (3, 6, 7, 9, 11, 12, 13, 14, 15)
    assert geometry.total_count == 16
    assert geometry.active_count == 9
    assert geometry.active_mask_sha256 == (
        "8ebce8c2e35957ec2f9ae3aac180b563f7d07462a826346b3881f6770097916f"
    )
    assert spec.to_metadata()["fingerprint"] == spec.fingerprint
    assert geometry.to_metadata()["fingerprint"] == geometry.fingerprint
    assert geometry.to_metadata()["spec_fingerprint"] == spec.fingerprint


def test_companion_interaction_radius_preserves_literal_unequal_ng_source_behavior() -> None:
    spec = TBGZeroFieldCompanionInteractionSpec(
        NG1=2,
        NG2=1,
        b1=1.0 + 0.0j,
        b2=0.0 + 1.0j,
    )

    assert spec.G1_labels == (-2, -1, 0, 1)
    assert spec.G2_labels == (-1, 0)
    assert spec.radius == pytest.approx(2.0, rel=0.0, abs=1.0e-15)
    assert spec.cutoff_convention == (
        "rectangular_NG1_NG2_labels_but_pinned_gen_interaction_radius_uses_NG1_for_"
        "both_R1_R2;strict_abs_total_Q_lt_radius_minus_margin"
    )


def test_companion_specs_pin_reference_identity_literals_in_metadata_and_fingerprints() -> None:
    assert TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY == "reference/TBG-HF"
    assert TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT == (
        "0d2a3d742aa901fa45ce46690c1385887165f58c"
    )
    assert TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE == "singleParticle.py"
    assert TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256 == (
        "a050fa545c4d399b227a178bcc4705a110bd7962edcb9e1f69e300b5e1a3e43b"
    )
    assert TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_REFERENCE_FUNCTION == "gen_coeff"
    assert TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_REFERENCE_LINES == "132-175"
    assert TBG_ZERO_FIELD_COMPANION_HOPPING_REFERENCE_FUNCTION == (
        "gen_moire_hamiltonian"
    )
    assert TBG_ZERO_FIELD_COMPANION_HOPPING_REFERENCE_LINES == "101-109"
    assert TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_FUNCTION == "gen_interaction"
    assert TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_LINES == "220-255"

    plane_wave_spec = TBGZeroFieldCompanionPlaneWaveSpec(
        Ng1=1,
        Ng2=1,
        b1=1.0 + 0.0j,
        b2=0.0 + 1.0j,
    )
    interaction_spec = TBGZeroFieldCompanionInteractionSpec(
        NG1=1,
        NG2=1,
        b1=1.0 + 0.0j,
        b2=0.0 + 1.0j,
    )
    shared_identity = {
        "reference_commit": "0d2a3d742aa901fa45ce46690c1385887165f58c",
        "reference_repository": "reference/TBG-HF",
        "reference_source": "singleParticle.py",
        "reference_source_sha256": (
            "a050fa545c4d399b227a178bcc4705a110bd7962edcb9e1f69e300b5e1a3e43b"
        ),
    }

    plane_wave_metadata = plane_wave_spec.to_metadata()
    interaction_metadata = interaction_spec.to_metadata()
    assert {key: plane_wave_metadata[key] for key in shared_identity} == shared_identity
    assert {key: interaction_metadata[key] for key in shared_identity} == shared_identity
    assert plane_wave_metadata["reference_function"] == "gen_coeff"
    assert plane_wave_metadata["reference_lines"] == "132-175"
    assert plane_wave_metadata["hopping_reference_function"] == (
        "gen_moire_hamiltonian"
    )
    assert plane_wave_metadata["hopping_reference_lines"] == "101-109"
    assert interaction_metadata["reference_function"] == "gen_interaction"
    assert interaction_metadata["reference_lines"] == "220-255"

    for metadata, fingerprint in (
        (plane_wave_metadata, plane_wave_spec.fingerprint),
        (interaction_metadata, interaction_spec.fingerprint),
    ):
        fingerprint_payload = dict(metadata)
        assert fingerprint_payload.pop("fingerprint") == fingerprint
        encoded = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        assert hashlib.sha256(encoded).hexdigest() == fingerprint


@pytest.mark.parametrize("bad_integer", [True, 1.0])
def test_companion_geometry_rejects_non_strict_integer_specs(bad_integer: object) -> None:
    with pytest.raises(TypeError, match="bool is not accepted"):
        TBGZeroFieldCompanionPlaneWaveSpec(
            Ng1=bad_integer,  # type: ignore[arg-type]
            Ng2=1,
            b1=1.0 + 0.0j,
            b2=0.0 + 1.0j,
        )
    with pytest.raises(TypeError, match="bool is not accepted"):
        TBGZeroFieldCompanionInteractionSpec(
            NG1=1,
            NG2=bad_integer,  # type: ignore[arg-type]
            b1=1.0 + 0.0j,
            b2=0.0 + 1.0j,
        )


def test_companion_geometry_rejects_bool_mesh_and_valley_indices() -> None:
    plane_wave_spec = TBGZeroFieldCompanionPlaneWaveSpec(
        Ng1=1,
        Ng2=1,
        b1=1.0 + 0.0j,
        b2=0.0 + 1.0j,
    )
    with pytest.raises(TypeError, match="bool is not accepted"):
        build_tbg_zero_field_companion_plane_wave_geometry(
            plane_wave_spec,
            N1=1,
            N2=1,
            ik1=0,
            ik2=0,
            stau=True,  # type: ignore[arg-type]
        )

    interaction_spec = TBGZeroFieldCompanionInteractionSpec(
        NG1=1,
        NG2=1,
        b1=1.0 + 0.0j,
        b2=0.0 + 1.0j,
    )
    with pytest.raises(TypeError, match="bool is not accepted"):
        build_tbg_zero_field_companion_interaction_geometry(
            interaction_spec,
            N1=True,  # type: ignore[arg-type]
            N2=1,
        )


# ---------------------------------------------------------------------------
# Stage-2 source-faithful companion BM single-particle diagnostic
# ---------------------------------------------------------------------------


def _small_companion_single_particle_params(
    *,
    N1: int = 1,
    N2: int = 1,
) -> TBGZeroFieldCompanionSingleParticleParams:
    return TBGZeroFieldCompanionSingleParticleParams(
        N1=N1,
        N2=N2,
        Ng1=2,
        Ng2=2,
        n_active=1,
        theta_deg=1.08,
        wAA_ev=0.07,
        wAB_ev=0.11,
        strain=0.003,
        strain_angle_deg=17.0,
    )


def test_companion_single_particle_params_are_strict_and_pin_source_units() -> None:
    params = TBGZeroFieldCompanionSingleParticleParams()

    assert params.to_companion_input() == {
        "N1": 8,
        "N2": 8,
        "Ng1": 4,
        "Ng2": 4,
        "n_active": 1,
        "theta": 1.08,
        "wAA": 0.07,
        "wAB": 0.11,
        "strain": 0.003,
        "varphi": 0.0,
    }
    assert params.parent_dimension == 256
    assert CCa == 1.42e-10
    assert Poisson == 0.16
    assert Beta == 3.14

    with pytest.raises(TypeError, match="N1 must be a positive integer"):
        replace(params, N1=1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool is not accepted"):
        replace(params, n_active=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="denominator zero"):
        replace(params, theta_deg=0.0)


def test_companion_parent_exact_blocks_indices_boundary_zero_fill_and_hermiticity() -> None:
    params = _small_companion_single_particle_params(N1=2, N2=3)
    geometry = gen_companion_RLVs(params)
    ham = gen_companion_moire_hamiltonian(
        params,
        (1, 2),
        rlv_geometry=geometry,
    )
    spec = TBGZeroFieldCompanionPlaneWaveSpec(
        Ng1=params.Ng1,
        Ng2=params.Ng2,
        b1=geometry.b1_complex,
        b2=geometry.b2_complex,
    )

    assert ham.shape == (params.parent_dimension, params.parent_dimension)
    assert spec.companion_index(-2, -2, 0) == 0
    assert spec.companion_index(-2, -2, 3) == 3
    assert spec.companion_index(-2, -1, 0) == 4
    assert spec.companion_index(-1, -2, 0) == 16
    np.testing.assert_allclose(ham, ham.conj().T, rtol=0.0, atol=0.0)

    omega = np.exp(2.0 * np.pi * 1j / 3.0)
    T1 = np.asarray(
        [[params.wAA_ev, params.wAB_ev], [params.wAB_ev, params.wAA_ev]],
        dtype=np.complex128,
    )
    T2 = np.asarray(
        [
            [params.wAA_ev, params.wAB_ev * omega],
            [params.wAB_ev * np.conj(omega), params.wAA_ev],
        ],
        dtype=np.complex128,
    )
    T3 = np.asarray(
        [
            [params.wAA_ev, params.wAB_ev * np.conj(omega)],
            [params.wAB_ev * omega, params.wAA_ev],
        ],
        dtype=np.complex128,
    )
    source = spec.companion_index(-1, -1, 2)
    target_T1 = spec.companion_index(-1, -1, 0)
    target_T2 = spec.companion_index(0, 0, 0)
    target_T3 = spec.companion_index(-1, 0, 0)
    np.testing.assert_array_equal(ham[target_T1 : target_T1 + 2, source : source + 2], T1)
    np.testing.assert_array_equal(ham[target_T2 : target_T2 + 2, source : source + 2], T2)
    np.testing.assert_array_equal(ham[target_T3 : target_T3 + 2, source : source + 2], T3)

    # Check the literal q -> subtract A -> M ordering and sy-conjugate Dirac block.
    g1, g2 = -1, 0
    kinetic = spec.companion_index(g1, g2, 0)
    kvec = geometry.b1 / params.N1 + 2.0 * (geometry.b2 / params.N2)
    d1 = -geometry.b1 / 3.0 + geometry.b2 / 3.0
    A1 = Beta / 2.0 / CCa * np.asarray(
        [
            geometry.Etens1[0, 0] - geometry.Etens1[1, 1],
            -2.0 * geometry.Etens1[0, 1],
        ]
    )
    q1 = np.dot(
        geometry.M1,
        geometry.ktheta_m_inv * (kvec + g1 * geometry.b1 + g2 * geometry.b2 - d1)
        - A1,
    )
    sx = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy_conjugate = np.asarray([[0.0, 1j], [-1j, 0.0]], dtype=np.complex128)
    expected_kinetic = companion_vhbar * (q1[0] * sx + q1[1] * sy_conjugate)
    np.testing.assert_array_equal(
        ham[kinetic : kinetic + 2, kinetic : kinetic + 2],
        expected_kinetic,
    )

    # At the upper rectangular corner, T2 and T3 are absent: only T1 remains
    # in the directed layer-2 -> layer-1 column block.
    boundary_source = spec.companion_index(params.Ng1 - 1, params.Ng2 - 1, 2)
    for target_g1 in spec.g1_labels:
        for target_g2 in spec.g2_labels:
            target = spec.companion_index(target_g1, target_g2, 0)
            block = ham[target : target + 2, boundary_source : boundary_source + 2]
            expected = (
                T1
                if (target_g1, target_g2) == (params.Ng1 - 1, params.Ng2 - 1)
                else np.zeros((2, 2), dtype=np.complex128)
            )
            np.testing.assert_array_equal(block, expected)


def test_companion_single_particle_uses_central_contiguous_ranks_and_parent_padding() -> None:
    params = _small_companion_single_particle_params()
    rlv_geometry = gen_companion_RLVs(params)
    spec = TBGZeroFieldCompanionPlaneWaveSpec(
        Ng1=params.Ng1,
        Ng2=params.Ng2,
        b1=rlv_geometry.b1_complex,
        b2=rlv_geometry.b2_complex,
    )
    point_geometry = build_tbg_zero_field_companion_plane_wave_geometry(
        spec,
        N1=params.N1,
        N2=params.N2,
        ik1=0,
        ik2=0,
        stau=1,
    )
    ham = gen_companion_moire_hamiltonian(params, (0, 0), rlv_geometry=rlv_geometry)
    sub_index = np.asarray(point_geometry.sub_index, dtype=int)
    eigvals, eigvecs = np.linalg.eigh(ham[sub_index][:, sub_index])
    center = sub_index.size // 2
    ranks = slice(center - params.n_active, center + params.n_active)

    result = solve_tbg_zero_field_companion_single_particle(params)
    np.testing.assert_array_equal(result.sp_energy_ev[0, 0, 0, :], eigvals[ranks])
    parent_coeff = np.transpose(
        result.coeff[0, 0, :, :, 0, :, :],
        (0, 1, 3, 2),
    ).reshape(params.parent_dimension, params.active_band_count)
    overlap = eigvecs[:, ranks].conj().T @ parent_coeff[sub_index, :]
    np.testing.assert_allclose(np.abs(overlap), np.eye(params.active_band_count), atol=1.0e-13)
    outside = np.ones(params.parent_dimension, dtype=bool)
    outside[sub_index] = False
    np.testing.assert_array_equal(
        parent_coeff[outside, :],
        np.zeros_like(parent_coeff[outside, :]),
    )


def test_companion_pointwise_gauge_and_final_C2T_sewing_are_literal() -> None:
    params = _small_companion_single_particle_params(N1=2, N2=3)
    result = solve_tbg_zero_field_companion_single_particle(params)

    reshaped = np.reshape(
        result.coeff,
        (
            params.N1,
            params.N2,
            2 * params.Ng1,
            2 * params.Ng2,
            2,
            params.active_band_count,
            2,
            2,
        ),
    )
    transformed = np.zeros_like(reshaped)
    for sub in range(2):
        transformed[..., sub] = np.conj(reshaped[..., 1 - sub])
    expected = np.einsum(
        "kKgGtals,kKgGtbls->kKtab",
        np.conj(reshaped),
        transformed,
        optimize=True,
    )
    np.testing.assert_array_equal(result.U_C2T, expected)
    np.testing.assert_allclose(
        np.imag(np.diagonal(result.U_C2T[:, :, 0, :, :], axis1=-2, axis2=-1)),
        0.0,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert np.all(
        np.real(np.diagonal(result.U_C2T[:, :, 0, :, :], axis1=-2, axis2=-1))
        >= -2.0e-14
    )
    np.testing.assert_array_equal(result.U_C2T, companion_C2T_symmetry(params, result.coeff))

    assert result.coeff.shape == (2, 3, 4, 4, 2, 2, 4)
    assert result.sp_energy_ev.shape == (2, 3, 2, 2)
    assert result.U_C2T.shape == (2, 3, 2, 2, 2)
    assert not result.coeff.flags.writeable
    assert not result.sp_energy_ev.flags.writeable
    assert not result.U_C2T.flags.writeable
    assert result.geometry_fingerprints.mesh_shape == (2, 3)
    assert result.provenance.scientific_scope == TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCOPE
    assert result.provenance.scientific_scope == (
        "diagnostic_BM_parity_only_not_production_HF_or_TDHF"
    )
    assert result.provenance.rlv_reference_lines == "20-49"
    assert result.provenance.hamiltonian_reference_lines == "51-111"
    assert result.provenance.coefficient_reference_lines == "113-202"
    assert result.provenance.pointwise_C2T_gauge_reference_lines == "177-188"
    assert result.provenance.Kprime_time_reversal_reference_lines == "190-200"
    assert result.provenance.final_C2T_sewing_reference_lines == "265-279"
    metadata = result.to_metadata()
    assert metadata["params"]["default_input_source"] == (
        TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE
    )
    assert metadata["params"]["default_input_source_sha256"] == (
        TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE_SHA256
    )
    assert metadata["pointwise_gauge_warning"] == (
        TBG_ZERO_FIELD_COMPANION_POINTWISE_GAUGE_WARNING
    )
    assert metadata["residual_gauge_ambiguity"] == (
        TBG_ZERO_FIELD_COMPANION_RESIDUAL_GAUGE_AMBIGUITY
    )
    assert metadata["array_hash_semantics"] == (
        "artifact_integrity_only_not_cross_eigensolver_parity"
    )
    assert metadata["array_hashes"]["semantics"] == (
        TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_SEMANTICS
    )
    assert metadata["provenance"]["pointwise_gauge_warning"] == (
        TBG_ZERO_FIELD_COMPANION_POINTWISE_GAUGE_WARNING
    )
    assert metadata["provenance"]["residual_gauge_ambiguity"] == (
        TBG_ZERO_FIELD_COMPANION_RESIDUAL_GAUGE_AMBIGUITY
    )
    assert metadata["provenance"]["array_hash_semantics"] == (
        TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_SEMANTICS
    )


def test_companion_Kprime_uses_unequal_mesh_python_floor_carries_and_zero_fill() -> None:
    params = _small_companion_single_particle_params(N1=2, N2=3)
    result = solve_tbg_zero_field_companion_single_particle(params)
    invalid_boundary_count = 0

    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            source_ik1 = (-ik1) % params.N1
            source_ik2 = (-ik2) % params.N2
            np.testing.assert_array_equal(
                result.sp_energy_ev[ik1, ik2, 1, :],
                result.sp_energy_ev[source_ik1, source_ik2, 0, :],
            )
            carry1 = (-ik1) // params.N1
            carry2 = (-ik2) // params.N2
            for g1 in range(-params.Ng1, params.Ng1):
                for g2 in range(-params.Ng2, params.Ng2):
                    gp1 = -g1 + carry1
                    gp2 = -g2 + carry2
                    target = result.coeff[
                        ik1,
                        ik2,
                        g1 + params.Ng1,
                        g2 + params.Ng2,
                        1,
                        :,
                        :,
                    ]
                    if -params.Ng1 <= gp1 < params.Ng1 and -params.Ng2 <= gp2 < params.Ng2:
                        expected = np.conj(
                            result.coeff[
                                source_ik1,
                                source_ik2,
                                gp1 + params.Ng1,
                                gp2 + params.Ng2,
                                0,
                                :,
                                :,
                            ]
                        )
                    else:
                        invalid_boundary_count += 1
                        expected = np.zeros_like(target)
                    np.testing.assert_array_equal(target, expected)

    assert params.N1 != params.N2
    assert (-1) // params.N1 == -1
    assert (-1) // params.N2 == -1
    assert invalid_boundary_count > 0
    np.testing.assert_array_equal(
        result.coeff[0, :, 0, :, 1, :, :],
        np.zeros_like(result.coeff[0, :, 0, :, 1, :, :]),
    )


# ---------------------------------------------------------------------------
# Pinned-source stage-2 fixture parity (fixture generated out of process)
# ---------------------------------------------------------------------------


_COMPANION_FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parent / "fixtures" / "tbg_companion_single_particle_v1"
)
_COMPANION_FIXTURE_ARRAY_KEYS = {
    "parent_h_0_0",
    "parent_h_1_2",
    "rlv_Etens1",
    "rlv_Etens2",
    "rlv_M1",
    "rlv_M2",
    "rlv_b1",
    "rlv_b2",
    "source_U_C2T",
    "source_coeff",
    "source_sp_energy_ev",
    *(f"sub_index_{ik1}_{ik2}" for ik1 in range(2) for ik2 in range(3)),
}


def _companion_fixture_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _companion_fixture_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _companion_fixture_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@pytest.fixture(scope="module")
def companion_single_particle_source_fixture():
    manifest_path = _COMPANION_FIXTURE_DIRECTORY / "manifest.json"
    generator_path = _COMPANION_FIXTURE_DIRECTORY / "generate_fixture.py"
    assert manifest_path.is_file(), (
        "Pinned companion fixture manifest is absent; explicitly run "
        "tests/fixtures/tbg_companion_single_particle_v1/generate_fixture.py first"
    )
    assert generator_path.is_file(), "Pinned companion fixture generator is absent"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture_path = _COMPANION_FIXTURE_DIRECTORY / manifest["fixture_npz"]
    assert fixture_path.is_file(), (
        "Pinned companion fixture NPZ is absent; explicitly run "
        "tests/fixtures/tbg_companion_single_particle_v1/generate_fixture.py first"
    )
    with np.load(fixture_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    return manifest, arrays, generator_path, fixture_path


def test_companion_source_fixture_manifest_and_payload_are_hash_bound(
    companion_single_particle_source_fixture,
) -> None:
    manifest, arrays, generator_path, fixture_path = companion_single_particle_source_fixture
    assert manifest["fixture_schema"] == (
        "mean_field.tbg.companion_single_particle.source_fixture"
    )
    assert manifest["fixture_schema_version"] == 1
    assert manifest["array_hash_semantics"] == (
        TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_SEMANTICS
    )
    assert manifest["pointwise_gauge_warning"] == (
        TBG_ZERO_FIELD_COMPANION_POINTWISE_GAUGE_WARNING
    )
    assert manifest["residual_gauge_ambiguity"] == (
        TBG_ZERO_FIELD_COMPANION_RESIDUAL_GAUGE_AMBIGUITY
    )

    expected_input = {
        "N1": 2,
        "N2": 3,
        "Ng1": 2,
        "Ng2": 2,
        "n_active": 1,
        "theta": 1.08,
        "wAA": 0.07,
        "wAB": 0.11,
        "strain": 0.003,
        "varphi": 17.0,
    }
    assert manifest["input"] == expected_input
    assert all(manifest["resolved_input"][key] == value for key, value in expected_input.items())
    assert manifest["resolved_input_sha256"] == _companion_fixture_json_sha256(
        manifest["resolved_input"]
    )

    pinned_source = manifest["pinned_source"]
    assert pinned_source["reference_repository"] == TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY
    assert pinned_source["reference_commit"] == TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT
    assert pinned_source["default_input"] == {
        "path": TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE,
        "sha256": "c143c294ad95cf94d91cfbabd0437556e5c2a342850d54484c9b47caaf84b4de",
    }
    assert pinned_source["default_input"]["sha256"] == (
        TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE_SHA256
    )
    assert pinned_source["single_particle"] == {
        "path": (
            f"{TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY}/"
            f"{TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE}"
        ),
        "sha256": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
    }
    assert pinned_source["constants"] == {
        "path": (
            f"{TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY}/"
            f"{TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE}"
        ),
        "sha256": TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256,
    }
    assert pinned_source["source_line_ranges"] == {
        "C2T_symmetry": "265-279",
        "Kprime_time_reversal": "190-200",
        "gen_RLVs": "20-49",
        "gen_coeff": "113-202",
        "gen_moire_hamiltonian": "51-111",
        "pointwise_C2T_gauge": "177-188",
    }

    assert manifest["generator_script"] == generator_path.name
    assert _is_sha256(manifest["generator_script_sha256"])
    assert manifest["generator_script_sha256"] == _companion_fixture_file_sha256(
        generator_path
    )
    assert manifest["fixture_npz"] == fixture_path.name
    assert _is_sha256(manifest["fixture_npz_sha256"])
    assert manifest["fixture_npz_sha256"] == _companion_fixture_file_sha256(fixture_path)

    assert set(manifest["arrays"]) == _COMPANION_FIXTURE_ARRAY_KEYS
    assert set(arrays) == _COMPANION_FIXTURE_ARRAY_KEYS
    for key in sorted(_COMPANION_FIXTURE_ARRAY_KEYS):
        record = manifest["arrays"][key]
        array = arrays[key]
        assert set(record) == {"dtype", "sha256", "shape"}
        assert record["shape"] == list(array.shape)
        assert record["dtype"] == array.dtype.str
        assert _is_sha256(record["sha256"])
        assert record["sha256"] == _companion_fixture_array_sha256(array)

    assert arrays["rlv_M1"].shape == (2, 2)
    assert arrays["rlv_M2"].shape == (2, 2)
    assert arrays["rlv_b1"].shape == (2,)
    assert arrays["rlv_b2"].shape == (2,)
    assert arrays["rlv_Etens1"].shape == (2, 2)
    assert arrays["rlv_Etens2"].shape == (2, 2)
    assert arrays["parent_h_0_0"].shape == (64, 64)
    assert arrays["parent_h_1_2"].shape == (64, 64)
    assert arrays["source_coeff"].shape == (2, 3, 4, 4, 2, 2, 4)
    assert arrays["source_sp_energy_ev"].shape == (2, 3, 2, 2)
    assert arrays["source_U_C2T"].shape == (2, 3, 2, 2, 2)


def test_companion_port_matches_pinned_source_fixture_geometry_hamiltonian_and_energy(
    companion_single_particle_source_fixture,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = (
        companion_single_particle_source_fixture
    )
    params = _small_companion_single_particle_params(N1=2, N2=3)
    geometry = gen_companion_RLVs(params)
    for attribute, fixture_key in (
        ("M1", "rlv_M1"),
        ("M2", "rlv_M2"),
        ("b1", "rlv_b1"),
        ("b2", "rlv_b2"),
        ("Etens1", "rlv_Etens1"),
        ("Etens2", "rlv_Etens2"),
    ):
        np.testing.assert_allclose(
            getattr(geometry, attribute),
            arrays[fixture_key],
            rtol=0.0,
            atol=5.0e-15,
        )

    for ik, fixture_key in (((0, 0), "parent_h_0_0"), ((1, 2), "parent_h_1_2")):
        np.testing.assert_allclose(
            gen_companion_moire_hamiltonian(params, ik, rlv_geometry=geometry),
            arrays[fixture_key],
            rtol=0.0,
            atol=5.0e-13,
        )

    spec = TBGZeroFieldCompanionPlaneWaveSpec(
        Ng1=params.Ng1,
        Ng2=params.Ng2,
        b1=geometry.b1_complex,
        b2=geometry.b2_complex,
    )
    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            point_geometry = build_tbg_zero_field_companion_plane_wave_geometry(
                spec,
                N1=params.N1,
                N2=params.N2,
                ik1=ik1,
                ik2=ik2,
                stau=1,
            )
            np.testing.assert_array_equal(
                np.asarray(point_geometry.sub_index, dtype=np.int64),
                arrays[f"sub_index_{ik1}_{ik2}"],
            )

    result = solve_tbg_zero_field_companion_single_particle(params)
    np.testing.assert_allclose(
        result.sp_energy_ev,
        arrays["source_sp_energy_ev"],
        rtol=0.0,
        atol=5.0e-13,
    )
    # Raw coeff/U_C2T hashes are deliberately not compared across eigensolvers.


def _companion_parent_coefficients(
    coeff: np.ndarray,
    *,
    ik1: int,
    ik2: int,
    tau: int,
) -> np.ndarray:
    return np.transpose(
        coeff[ik1, ik2, :, :, tau, :, :],
        (0, 1, 3, 2),
    ).reshape(64, 2)


def test_companion_port_matches_pinned_source_fixture_gauge_invariants_and_c2t(
    companion_single_particle_source_fixture,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = (
        companion_single_particle_source_fixture
    )
    params = _small_companion_single_particle_params(N1=2, N2=3)
    result = solve_tbg_zero_field_companion_single_particle(params)
    source_coeff = arrays["source_coeff"]
    source_energy = arrays["source_sp_energy_ev"]
    source_U_C2T = arrays["source_U_C2T"]

    np.testing.assert_allclose(
        companion_C2T_symmetry(params, source_coeff),
        source_U_C2T,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        companion_C2T_symmetry(params, result.coeff),
        result.U_C2T,
        rtol=0.0,
        atol=2.0e-14,
    )

    aligned_coeff = np.array(result.coeff, copy=True)
    aligned_points: list[tuple[int, int, int]] = []
    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            for tau in range(2):
                source_parent = _companion_parent_coefficients(
                    source_coeff,
                    ik1=ik1,
                    ik2=ik2,
                    tau=tau,
                )
                port_parent = _companion_parent_coefficients(
                    result.coeff,
                    ik1=ik1,
                    ik2=ik2,
                    tau=tau,
                )
                np.testing.assert_allclose(
                    port_parent @ port_parent.conj().T,
                    source_parent @ source_parent.conj().T,
                    rtol=0.0,
                    atol=5.0e-11,
                )
                overlap = source_parent.conj().T @ port_parent
                np.testing.assert_allclose(
                    np.linalg.svd(overlap, compute_uv=False),
                    np.ones(params.active_band_count),
                    rtol=0.0,
                    atol=5.0e-11,
                )
                np.testing.assert_allclose(
                    np.linalg.svd(result.U_C2T[ik1, ik2, tau], compute_uv=False),
                    np.linalg.svd(source_U_C2T[ik1, ik2, tau], compute_uv=False),
                    rtol=0.0,
                    atol=5.0e-11,
                )

                minimum_gap = float(
                    np.min(np.abs(np.diff(source_energy[ik1, ik2, tau])))
                )
                if minimum_gap <= 1.0e-10:
                    # Exact degeneracies retain U(N), not merely real-sign, freedom.
                    continue
                diagonal = np.diag(overlap)
                off_diagonal = overlap - np.diag(diagonal)
                np.testing.assert_allclose(off_diagonal, 0.0, rtol=0.0, atol=5.0e-10)
                np.testing.assert_allclose(np.abs(diagonal), 1.0, rtol=0.0, atol=5.0e-10)
                np.testing.assert_allclose(
                    np.imag(diagonal),
                    0.0,
                    rtol=0.0,
                    atol=5.0e-10,
                )
                signs = np.where(np.real(diagonal) >= 0.0, 1.0, -1.0)
                aligned_coeff[ik1, ik2, :, :, tau, :, :] *= signs[
                    np.newaxis,
                    np.newaxis,
                    :,
                    np.newaxis,
                ]
                np.testing.assert_allclose(
                    aligned_coeff[ik1, ik2, :, :, tau, :, :],
                    source_coeff[ik1, ik2, :, :, tau, :, :],
                    rtol=0.0,
                    atol=5.0e-10,
                )
                aligned_points.append((ik1, ik2, tau))

    assert aligned_points, "The custom fixture must exercise real-sign coefficient alignment"
    aligned_U_C2T = companion_C2T_symmetry(params, aligned_coeff)
    for ik1, ik2, tau in aligned_points:
        np.testing.assert_allclose(
            aligned_U_C2T[ik1, ik2, tau],
            source_U_C2T[ik1, ik2, tau],
            rtol=0.0,
            atol=5.0e-10,
        )


# ---------------------------------------------------------------------------
# Stage-5 Kwan Eq. (99) source-array-bound ordered-anchor K-IVC diagnostic
# ---------------------------------------------------------------------------


def _tiny_unstrained_companion_stage2(
    *,
    N1: int = 2,
    N2: int = 3,
):
    params = TBGZeroFieldCompanionSingleParticleParams(
        N1=N1,
        N2=N2,
        Ng1=2,
        Ng2=2,
        n_active=1,
        theta_deg=1.05,
        wAA_ev=0.08,
        wAB_ev=0.11,
        strain=0.0,
        strain_angle_deg=0.0,
    )
    return solve_tbg_zero_field_companion_single_particle(params)


def _replace_companion_stage2_coeff(source, coeff: np.ndarray):
    resolved_coeff = np.asarray(coeff, dtype=np.complex128)
    resolved_U_C2T = companion_C2T_symmetry(source.params, resolved_coeff)
    hashes = TBGZeroFieldCompanionSingleParticleArrayHashes.from_arrays(
        coeff=resolved_coeff,
        sp_energy_ev=source.sp_energy_ev,
        U_C2T=resolved_U_C2T,
    )
    return replace(
        source,
        coeff=resolved_coeff,
        U_C2T=resolved_U_C2T,
        array_hashes=hashes,
    )


def _companion_lifted_kivc_projector(source, seed, ik1: int, ik2: int) -> np.ndarray:
    parent_dimension = source.params.parent_dimension
    lifted_frame = np.zeros((2 * parent_dimension, 4), dtype=np.complex128)
    lifted_frame[:parent_dimension, :2] = _companion_parent_coefficients(
        source.coeff,
        ik1=ik1,
        ik2=ik2,
        tau=0,
    )
    lifted_frame[parent_dimension:, 2:] = _companion_parent_coefficients(
        source.coeff,
        ik1=ik1,
        ik2=ik2,
        tau=1,
    )
    return lifted_frame @ seed.P_conventional[ik1, ik2] @ lifted_frame.conj().T


def _kivc_seed_array_mapping(seed) -> dict[str, np.ndarray]:
    return {
        field.name: getattr(seed, field.name)
        for field in fields(seed.array_hashes)
        if field.name != "convention"
    }

def test_kwan_eq99_explicit_matrices_and_valley_major_kronecker_order() -> None:
    expected_phi_zero = np.asarray(
        [
            [0.0, 0.0, 0.0, -1.0j],
            [0.0, 0.0, 1.0j, 0.0],
            [0.0, -1.0j, 0.0, 0.0],
            [1.0j, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    expected_phi_half_pi = np.asarray(
        [
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    np.testing.assert_array_equal(kwan_eq99_kivc_q(0.0), expected_phi_zero)
    np.testing.assert_allclose(
        kwan_eq99_kivc_q(np.pi / 2.0),
        expected_phi_half_pi,
        rtol=0.0,
        atol=1.0e-16,
    )

    tau_x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    np.testing.assert_array_equal(kwan_eq99_kivc_q(0.0), np.kron(tau_x, sigma_y))
    assert not np.array_equal(kwan_eq99_kivc_q(0.0), np.kron(sigma_y, tau_x))


def test_kwan_eq99_seed_actual_unstrained_stage2_invariants_and_scopes() -> None:
    source = _tiny_unstrained_companion_stage2()
    seed = build_tbg_zero_field_companion_kivc_seed(source, phi=0.37)
    params = source.params

    assert seed.provenance.scientific_scope == TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCOPE
    assert KWAN_EQ99_PDF_SHA256 == (
        "2354caaa3c5fddbdc7c5caaacbc9dcfa94c45dfc855d930b10372daabf6fd8a6"
    )
    assert TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256 == (
        "a050fa545c4d399b227a178bcc4705a110bd7962edcb9e1f69e300b5e1a3e43b"
    )
    assert TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256 == (
        "d7c7138ddf2107a71c24194ac70790bd27cdc05297ee9cdc997c1dc3882e5ede"
    )
    assert TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256 == (
        "d1a47420400c3381247f4bc8c2e7700935077536b7782a14e52e1d25a1fd516e"
    )
    assert seed.provenance.source_hashes.paper_pdf == KWAN_EQ99_PDF_SHA256
    assert seed.provenance.source_hashes.single_particle == (
        TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256
    )
    assert seed.provenance.source_hashes.projectors == (
        TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256
    )
    assert seed.provenance.source_hashes.measure == (
        TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256
    )
    assert seed.provenance.paper_arxiv == KWAN_EQ99_ARXIV == "2511.21683v1"
    assert seed.provenance.paper_reference == KWAN_EQ99_REFERENCE
    assert "PDF pages 38-39 (printed 38-39)" in KWAN_EQ99_REFERENCE
    assert "Eq. (99) implementation authority" in KWAN_EQ99_REFERENCE
    assert "Eq. (98) Chern context only" in KWAN_EQ99_REFERENCE
    assert "not implementing Eq. (97) triplet/n_pm" in KWAN_EQ99_REFERENCE
    assert "not implementing Eq. (98) pseudospin equality" in KWAN_EQ99_REFERENCE
    assert seed.P_conventional.shape == (params.N1, params.N2, 4, 4)
    assert seed.P_stored.shape == (params.N1, params.N2, 2, 4, 4)
    assert seed.U_Tp.shape == (params.N1, params.N2, 4, 4)
    assert seed.anchor_indices_K.shape == (params.N1, params.N2, 2)
    assert seed.anchor_relative_magnitudes_K.shape == (params.N1, params.N2, 2)
    assert np.all(
        seed.anchor_relative_magnitudes_K
        >= seed.phase_anchor_min_relative_magnitude
    )
    assert seed.anchor_relative_magnitude_min == float(
        np.min(seed.anchor_relative_magnitudes_K)
    )
    assert len(seed.array_hashes.anchor_indices_K) == 64
    assert len(seed.array_hashes.anchor_relative_magnitudes_K) == 64

    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            mk1 = (-ik1) % params.N1
            mk2 = (-ik2) % params.N2
            np.testing.assert_array_equal(
                seed.W_Kprime[ik1, ik2],
                np.conj(seed.W_K[mk1, mk2]),
            )
            for tau, frame in enumerate(
                (seed.W_K[ik1, ik2], seed.W_Kprime[ik1, ik2])
            ):
                transformed_Z = frame.conj().T @ seed.Z_projected[
                    ik1, ik2, tau
                ] @ frame
                np.testing.assert_allclose(
                    transformed_Z,
                    np.diag(seed.Z_spectra[ik1, ik2, tau]),
                    rtol=0.0,
                    atol=5.0e-10,
                )
            expected_U_Tp = (
                seed.W[ik1, ik2]
                @ np.kron(
                    np.asarray([[0.0, -1.0j], [1.0j, 0.0]]),
                    np.eye(2),
                )
                @ seed.W[mk1, mk2].T
            )
            np.testing.assert_array_equal(seed.U_Tp[ik1, ik2], expected_U_Tp)

    assert np.all(seed.Z_spectra[..., 0] > 0.0)
    assert np.all(seed.Z_spectra[..., 1] < 0.0)
    np.testing.assert_allclose(
        seed.P_conventional @ seed.P_conventional,
        seed.P_conventional,
        rtol=0.0,
        atol=5.0e-10,
    )
    np.testing.assert_array_equal(
        seed.P_stored,
        np.repeat(
            seed.P_conventional.swapaxes(-1, -2)[:, :, None, :, :],
            2,
            axis=2,
        ),
    )
    np.testing.assert_array_equal(seed.P_stored[:, :, 0], seed.P_stored[:, :, 1])
    np.testing.assert_allclose(seed.chern_balance_trace, 0.0, rtol=0.0, atol=5.0e-10)

    # Same projected-Z labels in both valleys are explicitly not the same
    # physical-Chern ordering: Gamma_C eigenvalues reverse in K'.
    np.testing.assert_array_equal(
        np.diag(seed.Gamma_C),
        np.asarray([1.0, -1.0, -1.0, 1.0]),
    )
    assert "same_projected_microscopic_sublattice_Z_order" in (
        TBG_ZERO_FIELD_COMPANION_KIVC_CANONICAL_ORDER
    )
    assert "not_same_physical_Chern_order" in TBG_ZERO_FIELD_COMPANION_KIVC_CANONICAL_ORDER

    assert seed.residuals.source_TR_Z_max_abs < 5.0e-10
    assert seed.residuals.kprime_Z_diagonalization_max_abs < 5.0e-10
    assert seed.residuals.tp_square_max_abs <= MAX_VALIDATION_TOLERANCE
    assert seed.residuals.tp_Q_invariance_max_abs <= MAX_VALIDATION_TOLERANCE
    assert seed.residuals.tp_P_invariance_max_abs <= MAX_VALIDATION_TOLERANCE
    assert seed.residuals.companion_measure_Tp_max_abs <= MAX_VALIDATION_TOLERANCE
    metadata = seed.to_metadata()
    source_metadata = metadata["provenance"]["source_hashes"]
    assert source_metadata["external_authority_files"] == (
        TBG_ZERO_FIELD_COMPANION_KIVC_EXTERNAL_AUTHORITY_FILES
    )
    assert TBG_ZERO_FIELD_COMPANION_KIVC_EXTERNAL_AUTHORITY_FILES == (
        "external_authority_files_not_embedded"
    )
    assert source_metadata["hashes"] == {
        "measure": TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256,
        "paper_pdf": KWAN_EQ99_PDF_SHA256,
        "projectors": TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256,
        "single_particle": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
    }
    assert source_metadata["locator"] == {
        "companion_commit": TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
        "companion_repository": TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
        "measure": TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE,
        "paper_pdf": KWAN_EQ99_PDF_SOURCE,
        "projectors": TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE,
        "single_particle": (
            f"{TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY}/"
            f"{TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE}"
        ),
    }
    assert metadata["frame_scope"] == TBG_ZERO_FIELD_COMPANION_KIVC_FRAME_SCOPE
    assert metadata["frame_scope"] == (
        "source-array-bound ordered phase anchor in exact companion parent order; "
        "active-subspace gauge-covariant; not cross-eigensolver reproducible; "
        "not a global smooth gauge"
    )
    assert metadata["basis_covariance_scope"] == (
        TBG_ZERO_FIELD_COMPANION_KIVC_BASIS_COVARIANCE_SCOPE
    )
    assert metadata["mapped_U_Tp_validation_scope"] == (
        TBG_ZERO_FIELD_COMPANION_KIVC_TP_SCOPE
    )
    assert "algebraic_by_construction" in metadata["mapped_U_Tp_validation_scope"]
    assert metadata["companion_measure_Tp_validation_scope"] == (
        TBG_ZERO_FIELD_COMPANION_KIVC_COMPANION_MEASURE_TP_SCOPE
    )
    assert metadata["provenance"]["companion_measure_source"] == (
        TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE
    )
    assert metadata["provenance"]["companion_measure_Tp_reference_lines"] == (
        TBG_ZERO_FIELD_COMPANION_MEASURE_TP_REFERENCE_LINES
    )
    assert metadata["anchor_relative_magnitude_min"] == (
        seed.anchor_relative_magnitude_min
    )
    assert metadata["phase_anchor_min_relative_magnitude"] == (
        TBG_ZERO_FIELD_COMPANION_KIVC_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE
    )
    assert metadata["phase_anchor_min_relative_magnitude_hard_min"] == (
        TBG_ZERO_FIELD_COMPANION_KIVC_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE
    )
    assert metadata["phase_anchor_min_relative_magnitude_hard_max"] == (
        MAX_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE
    )
    assert metadata["schema_version"] == 2
    assert metadata["validation_tolerance_hard_max"] == MAX_VALIDATION_TOLERANCE
    assert metadata["stage2_array_hashes"]["fingerprint"] == (
        source.array_hashes.fingerprint
    )
    assert metadata["chern_validation_scope"] == TBG_ZERO_FIELD_COMPANION_KIVC_CHERN_SCOPE
    assert "not_FHS_Chern_validation" in metadata["chern_validation_scope"]
    assert "chern_number" not in metadata
    assert metadata["stored_projector_convention"] == (
        TBG_ZERO_FIELD_COMPANION_KIVC_STORED_PROJECTOR_CONVENTION
    )

    for array in (
        seed.Z_projected,
        seed.Z_spectra,
        seed.W_K,
        seed.W_Kprime,
        seed.W,
        seed.anchor_indices_K,
        seed.anchor_relative_magnitudes_K,
        seed.Q_canonical,
        seed.Q_band,
        seed.P_conventional,
        seed.P_stored,
        seed.U_Tp,
        seed.Gamma_C,
        seed.source_TR_Z_residuals,
        seed.kprime_Z_diagonalization_residuals,
        seed.tp_square_residuals,
        seed.tp_Q_invariance_residuals,
        seed.tp_P_invariance_residuals,
        seed.companion_measure_Tp_residuals,
        seed.chern_balance_trace,
    ):
        assert not array.flags.writeable


def test_kwan_eq99_seed_companion_measure_boost0_Tp_residual_is_independent() -> None:
    source = _tiny_unstrained_companion_stage2()
    seed = build_tbg_zero_field_companion_kivc_seed(source, phi=0.29)
    params = source.params

    # Independent transcription of reference/TBG-HF/measure.py lines 5-14,
    # 64-69: reshape valley/band axes, flip k and both valley axes, roll k by
    # (1, 1), conjugate, then negate both off-diagonal valley blocks.
    Pex = np.reshape(seed.P_stored, (params.N1, params.N2, 2, 2, 2, 2, 2))
    expected_T = np.flip(Pex, axis=(0, 1, 3, 5)).copy()
    expected_T = np.roll(expected_T, (1, 1), axis=(0, 1))
    expected_T = np.conj(expected_T)
    expected_Tp = expected_T.copy()
    expected_Tp[:, :, :, 0, :, 1, :] = -expected_T[:, :, :, 0, :, 1, :]
    expected_Tp[:, :, :, 1, :, 0, :] = -expected_T[:, :, :, 1, :, 0, :]
    expected_residuals = np.max(
        np.abs(seed.P_stored - expected_Tp.reshape(seed.P_stored.shape)),
        axis=(2, 3, 4),
    )

    np.testing.assert_array_equal(
        seed.companion_measure_Tp_residuals,
        expected_residuals,
    )
    assert seed.residuals.companion_measure_Tp_max_abs == float(
        np.max(expected_residuals)
    )
    assert seed.residuals.companion_measure_Tp_max_abs <= seed.validation_tolerance
    metadata = seed.to_metadata()
    assert metadata["residuals"]["companion_measure_Tp_max_abs"] == (
        seed.residuals.companion_measure_Tp_max_abs
    )
    assert "independent_companion_measure" in (
        metadata["companion_measure_Tp_validation_scope"]
    )
    assert "algebraic_by_construction" in metadata["mapped_U_Tp_validation_scope"]

def test_kwan_eq99_seed_has_paired_active_subspace_basis_covariance() -> None:
    """Arbitrary paired U(2) is not a valid nondegenerate eigenpair gauge."""

    params = _small_companion_single_particle_params(N1=2, N2=3)
    source = solve_tbg_zero_field_companion_single_particle(params)
    seed = build_tbg_zero_field_companion_kivc_seed(source, phi=0.43)
    assert seed.to_metadata()["basis_covariance_scope"] == (
        "paired active-subspace basis covariance; arbitrary U(2) is not a valid "
        "nondegenerate eigenpair gauge"
    )
    rng = np.random.default_rng(90210)
    K_basis_change = np.empty((params.N1, params.N2, 2, 2), dtype=np.complex128)
    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            raw = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
            unitary, triangular = np.linalg.qr(raw)
            K_basis_change[ik1, ik2] = unitary @ np.diag(
                np.exp(-1.0j * np.angle(np.diag(triangular)))
            )

    changed_coeff = np.array(source.coeff, copy=True)
    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            mk1 = (-ik1) % params.N1
            mk2 = (-ik2) % params.N2
            changed_coeff[ik1, ik2, :, :, 0, :, :] = np.einsum(
                "...as,ab->...bs",
                source.coeff[ik1, ik2, :, :, 0, :, :],
                K_basis_change[ik1, ik2],
            )
            changed_coeff[ik1, ik2, :, :, 1, :, :] = np.einsum(
                "...as,ab->...bs",
                source.coeff[ik1, ik2, :, :, 1, :, :],
                np.conj(K_basis_change[mk1, mk2]),
            )
    changed_source = _replace_companion_stage2_coeff(source, changed_coeff)
    changed_seed = build_tbg_zero_field_companion_kivc_seed(
        changed_source,
        phi=0.43,
    )

    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            np.testing.assert_allclose(
                _companion_lifted_kivc_projector(source, seed, ik1, ik2),
                _companion_lifted_kivc_projector(
                    changed_source,
                    changed_seed,
                    ik1,
                    ik2,
                ),
                rtol=0.0,
                atol=MAX_VALIDATION_TOLERANCE,
            )


def test_kwan_eq99_seed_allows_paired_nondegenerate_U1_phase_covariance() -> None:
    params = _small_companion_single_particle_params(N1=2, N2=3)
    source = solve_tbg_zero_field_companion_single_particle(params)
    seed = build_tbg_zero_field_companion_kivc_seed(source, phi=0.43)
    rng = np.random.default_rng(8675309)
    K_phases = np.exp(
        1.0j * rng.uniform(-np.pi, np.pi, size=(params.N1, params.N2, 2))
    )

    phase_changed_coeff = np.array(source.coeff, copy=True)
    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            mk1 = (-ik1) % params.N1
            mk2 = (-ik2) % params.N2
            phase_changed_coeff[ik1, ik2, :, :, 0, :, :] *= K_phases[
                ik1,
                ik2,
                np.newaxis,
                :,
                np.newaxis,
            ]
            phase_changed_coeff[ik1, ik2, :, :, 1, :, :] *= np.conj(
                K_phases[mk1, mk2, np.newaxis, :, np.newaxis]
            )
    phase_changed_source = _replace_companion_stage2_coeff(
        source,
        phase_changed_coeff,
    )
    phase_changed_seed = build_tbg_zero_field_companion_kivc_seed(
        phase_changed_source,
        phi=0.43,
    )

    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            np.testing.assert_allclose(
                _companion_lifted_kivc_projector(source, seed, ik1, ik2),
                _companion_lifted_kivc_projector(
                    phase_changed_source,
                    phase_changed_seed,
                    ik1,
                    ik2,
                ),
                rtol=0.0,
                atol=MAX_VALIDATION_TOLERANCE,
            )

def test_kwan_eq99_seed_rejects_loose_validation_or_phase_anchor_thresholds() -> None:
    source = solve_tbg_zero_field_companion_single_particle(
        _small_companion_single_particle_params()
    )
    accepted = build_tbg_zero_field_companion_kivc_seed(
        source,
        validation_tolerance=MAX_VALIDATION_TOLERANCE,
    )
    assert accepted.validation_tolerance == MAX_VALIDATION_TOLERANCE == 1.0e-10
    assert accepted.phase_anchor_min_relative_magnitude == (
        TBG_ZERO_FIELD_COMPANION_KIVC_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE
    )
    assert TBG_ZERO_FIELD_COMPANION_KIVC_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE == 1.0e-12
    for rejected in (
        0.0,
        -1.0e-16,
        np.nextafter(MAX_VALIDATION_TOLERANCE, np.inf),
        1.0e-9,
    ):
        with pytest.raises(
            ValueError,
            match=r"validation_tolerance must be > 0 and <= MAX_VALIDATION_TOLERANCE",
        ):
            build_tbg_zero_field_companion_kivc_seed(
                source,
                validation_tolerance=rejected,
            )

    for rejected in (
        0.0,
        np.nextafter(
            TBG_ZERO_FIELD_COMPANION_KIVC_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE,
            0.0,
        ),
        np.nextafter(MAX_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE, np.inf),
    ):
        with pytest.raises(
            ValueError,
            match=r"phase_anchor_min_relative_magnitude must be >=",
        ):
            build_tbg_zero_field_companion_kivc_seed(
                source,
                phase_anchor_min_relative_magnitude=rejected,
            )

def test_kwan_eq99_seed_accepts_equal_max_anchors_and_fails_without_anchor_or_Z_gap() -> None:
    source = solve_tbg_zero_field_companion_single_particle(
        _small_companion_single_particle_params()
    )
    params = source.params

    equal_max_parent = np.zeros((params.parent_dimension, 2), dtype=np.complex128)
    equal_max_parent[0, 0] = equal_max_parent[2, 0] = 1.0 / np.sqrt(2.0)
    equal_max_parent[1, 1] = equal_max_parent[3, 1] = 1.0 / np.sqrt(2.0)
    equal_max_point = equal_max_parent.reshape(
        2 * params.Ng1,
        2 * params.Ng2,
        4,
        2,
    ).transpose(0, 1, 3, 2)
    equal_max_coeff = np.zeros_like(source.coeff)
    equal_max_coeff[0, 0, :, :, 0, :, :] = equal_max_point
    equal_max_coeff[0, 0, :, :, 1, :, :] = equal_max_point
    equal_max_source = _replace_companion_stage2_coeff(source, equal_max_coeff)
    equal_max_seed = build_tbg_zero_field_companion_kivc_seed(
        equal_max_source,
        phase_anchor_min_relative_magnitude=0.5,
    )
    np.testing.assert_array_equal(
        equal_max_seed.anchor_indices_K,
        np.asarray([[[0, 1]]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        equal_max_seed.anchor_relative_magnitudes_K,
        1.0 / np.sqrt(2.0),
        rtol=0.0,
        atol=1.0e-15,
    )
    for canonical_index in range(2):
        lifted = equal_max_parent @ equal_max_seed.W_K[
            0,
            0,
            :,
            canonical_index,
        ]
        anchor = int(equal_max_seed.anchor_indices_K[0, 0, canonical_index])
        qualifying = np.flatnonzero(np.abs(lifted) / np.linalg.norm(lifted) >= 0.5)
        assert anchor == int(qualifying[0])
        assert lifted[anchor].real > 0.0
        assert abs(lifted[anchor].imag) <= 1.0e-15

    assert np.all(equal_max_seed.anchor_relative_magnitudes_K < 0.75)
    with pytest.raises(
        ValueError,
        match="no lifted microscopic phase anchor at or above",
    ):
        build_tbg_zero_field_companion_kivc_seed(
            equal_max_source,
            phase_anchor_min_relative_magnitude=0.75,
        )

    gapless_parent = np.zeros((params.parent_dimension, 2), dtype=np.complex128)
    gapless_parent[0, 0] = gapless_parent[1, 0] = 1.0 / np.sqrt(2.0)
    gapless_parent[2, 1] = gapless_parent[3, 1] = 1.0 / np.sqrt(2.0)
    gapless_point = gapless_parent.reshape(
        2 * params.Ng1,
        2 * params.Ng2,
        4,
        2,
    ).transpose(0, 1, 3, 2)
    gapless_coeff = np.zeros_like(source.coeff)
    gapless_coeff[0, 0, :, :, 0, :, :] = gapless_point
    gapless_coeff[0, 0, :, :, 1, :, :] = gapless_point
    gapless_source = _replace_companion_stage2_coeff(source, gapless_coeff)
    with pytest.raises(ValueError, match="lacks one positive and one negative"):
        build_tbg_zero_field_companion_kivc_seed(gapless_source)


@pytest.mark.parametrize("array_name", ("Z_projected", "Z_spectra", "U_Tp"))
def test_kwan_eq99_seed_semantic_replay_rejects_coherent_array_hash_forgery(
    array_name: str,
) -> None:
    source = _tiny_unstrained_companion_stage2()
    seed = build_tbg_zero_field_companion_kivc_seed(source, phi=0.37)
    forged = np.array(getattr(seed, array_name), copy=True)
    forged.reshape(-1)[0] += 1.0e-6
    arrays = _kivc_seed_array_mapping(seed)
    arrays[array_name] = forged
    refreshed_hashes = type(seed.array_hashes).from_arrays(arrays)
    assert refreshed_hashes != seed.array_hashes

    with pytest.raises(
        ValueError,
        match=rf"{array_name} does not match deterministic Stage-2 semantic replay",
    ):
        replace(
            seed,
            **{array_name: forged, "array_hashes": refreshed_hashes},
        )

@pytest.mark.parametrize(
    ("array_name", "residual_field"),
    (
        ("source_TR_Z_residuals", "source_TR_Z_max_abs"),
        (
            "kprime_Z_diagonalization_residuals",
            "kprime_Z_diagonalization_max_abs",
        ),
        ("tp_square_residuals", "tp_square_max_abs"),
        ("tp_Q_invariance_residuals", "tp_Q_invariance_max_abs"),
        ("tp_P_invariance_residuals", "tp_P_invariance_max_abs"),
        (
            "companion_measure_Tp_residuals",
            "companion_measure_Tp_max_abs",
        ),
        ("chern_balance_trace", "chern_balance_max_abs"),
    ),
)
def test_kwan_eq99_seed_semantic_replay_rejects_coherent_residual_forgery(
    array_name: str,
    residual_field: str,
) -> None:
    source = _tiny_unstrained_companion_stage2()
    seed = build_tbg_zero_field_companion_kivc_seed(source, phi=0.37)
    forged = np.array(getattr(seed, array_name), copy=True)
    target = 0.5 * seed.validation_tolerance
    if getattr(seed.residuals, residual_field) == target:
        target = 0.25 * seed.validation_tolerance
    forged.fill(target)
    arrays = _kivc_seed_array_mapping(seed)
    arrays[array_name] = forged
    refreshed_hashes = type(seed.array_hashes).from_arrays(arrays)
    refreshed_residuals = replace(
        seed.residuals,
        **{residual_field: float(np.max(np.abs(forged)))},
    )
    assert refreshed_hashes != seed.array_hashes
    assert refreshed_residuals != seed.residuals

    with pytest.raises(
        ValueError,
        match=rf"{array_name} does not match deterministic Stage-2 semantic replay",
    ):
        replace(
            seed,
            **{
                array_name: forged,
                "array_hashes": refreshed_hashes,
                "residuals": refreshed_residuals,
            },
        )

def test_kwan_eq99_seed_live_stage2_source_rehash_fails_on_array_drift() -> None:
    source = solve_tbg_zero_field_companion_single_particle(
        _small_companion_single_particle_params()
    )
    seed = build_tbg_zero_field_companion_kivc_seed(source)
    original_value = source.coeff.flat[0].item()
    try:
        source.coeff.setflags(write=True)
        source.coeff.flat[0] = original_value + 1.0e-6
        source.coeff.setflags(write=False)
        with pytest.raises(
            ValueError,
            match=r"single_particle_source\.coeff no longer matches its source hash",
        ):
            _ = seed.fingerprint
    finally:
        source.coeff.setflags(write=True)
        source.coeff.flat[0] = original_value
        source.coeff.setflags(write=False)





# Optional external-authority coverage; the tests above are clean-checkout core.
def test_kwan_eq99_optional_external_authorities_validate_and_detect_drift(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    pdf_path = repository_root / KWAN_EQ99_PDF_SOURCE
    companion_root = repository_root / TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY
    authority_files = (
        pdf_path,
        companion_root / TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
        companion_root / Path(TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE).name,
        companion_root / Path(TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE).name,
    )
    if not all(path.is_file() for path in authority_files) or not (
        companion_root / ".git"
    ).exists():
        pytest.skip("optional ignored external K-IVC authorities are unavailable")

    validated = validate_tbg_zero_field_companion_kivc_external_authorities(
        pdf_path,
        companion_root,
    )
    assert validated.paper_pdf == KWAN_EQ99_PDF_SHA256
    assert validated.single_particle == (
        TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256
    )
    assert validated.projectors == TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256
    assert validated.measure == TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256

    missing_pdf = tmp_path / "missing-authority.pdf"
    with pytest.raises(ValueError, match="Pinned Kwan PDF source is unavailable"):
        validate_tbg_zero_field_companion_kivc_external_authorities(
            missing_pdf,
            companion_root,
        )

    drifted_pdf = tmp_path / pdf_path.name
    drifted_pdf.write_bytes(pdf_path.read_bytes() + b"\nexternal authority drift\n")
    with pytest.raises(ValueError, match="Pinned paper_pdf external authority drift"):
        validate_tbg_zero_field_companion_kivc_external_authorities(
            drifted_pdf,
            companion_root,
        )

# ---------------------------------------------------------------------------
# Stage-3 source-faithful companion interaction diagnostic
# ---------------------------------------------------------------------------

_COMPANION_INTERACTION_FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parent / "fixtures" / "tbg_companion_interaction_v1"
)
_COMPANION_INTERACTION_FIXTURE_ARRAY_KEYS = {"source_intFT"}
_COMPANION_INTERACTION_FIXTURE_MANIFEST_KEYS = {
    "array_hash_convention",
    "array_hash_semantics",
    "arrays",
    "environment",
    "fixture_npz",
    "fixture_npz_sha256",
    "fixture_schema",
    "fixture_schema_version",
    "generator_script",
    "generator_script_sha256",
    "inherited_input",
    "input_overrides",
    "output_units",
    "pinned_source",
    "resolved_input",
    "resolved_input_sha256",
}
_COMPANION_INTERACTION_INPUT_OVERRIDES = {
    "N1": 2,
    "N2": 3,
    "Ng1": 2,
    "Ng2": 2,
    "n_active": 1,
    "theta": 1.08,
    "wAA": 0.07,
    "wAB": 0.11,
    "strain": 0.003,
    "varphi": 17.0,
}
_COMPANION_INTERACTION_INHERITED_INPUT = {
    "NG1": 5,
    "NG2": 5,
    "dsc": 2.5000000000000002e-8,
    "gates": "dual",
    "include_q=0": True,
}
_COMPANION_INTERACTION_SOURCE_LINE_RANGES = {
    "finite_q_kernels": "251-257",
    "gen_interaction": "205-258",
    "q0_branch": "240-250",
    "support_and_allocation": "220-239",
}


@pytest.fixture(scope="module")
def companion_interaction_source_fixture():
    manifest_path = _COMPANION_INTERACTION_FIXTURE_DIRECTORY / "manifest.json"
    generator_path = _COMPANION_INTERACTION_FIXTURE_DIRECTORY / "generate_fixture.py"
    assert manifest_path.is_file(), (
        "Pinned companion interaction manifest is absent; explicitly run "
        "tests/fixtures/tbg_companion_interaction_v1/generate_fixture.py first"
    )
    assert generator_path.is_file(), "Pinned companion interaction generator is absent"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture_path = _COMPANION_INTERACTION_FIXTURE_DIRECTORY / manifest["fixture_npz"]
    assert fixture_path.is_file(), (
        "Pinned companion interaction NPZ is absent; explicitly run "
        "tests/fixtures/tbg_companion_interaction_v1/generate_fixture.py first"
    )
    with np.load(fixture_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    return manifest, arrays, generator_path, fixture_path


def test_companion_interaction_fixture_manifest_and_payload_are_exactly_bound(
    companion_interaction_source_fixture,
) -> None:
    manifest, arrays, generator_path, fixture_path = companion_interaction_source_fixture

    assert set(manifest) == _COMPANION_INTERACTION_FIXTURE_MANIFEST_KEYS
    assert manifest["fixture_schema"] == (
        "mean_field.tbg.companion_interaction.source_fixture"
    )
    assert manifest["fixture_schema_version"] == 1
    assert manifest["array_hash_convention"] == (
        "sha256_little_endian_int64_shape_then_C_order_little_endian_array_bytes"
    )
    assert manifest["array_hash_semantics"] == SOURCE_INTERACTION_ARRAY_HASH_SEMANTICS
    assert manifest["output_units"] == {"source_intFT": "eV"}
    assert manifest["input_overrides"] == _COMPANION_INTERACTION_INPUT_OVERRIDES
    assert set(manifest["input_overrides"]) == set(_COMPANION_INTERACTION_INPUT_OVERRIDES)
    assert manifest["inherited_input"] == _COMPANION_INTERACTION_INHERITED_INPUT
    assert manifest["inherited_input"]["dsc"] == 2.5000000000000002e-8
    assert manifest["inherited_input"]["dsc"] != 25e-9

    expected_resolved_input = {
        **_COMPANION_INTERACTION_INPUT_OVERRIDES,
        **_COMPANION_INTERACTION_INHERITED_INPUT,
    }
    assert manifest["resolved_input"] == expected_resolved_input
    assert manifest["resolved_input_sha256"] == _companion_fixture_json_sha256(
        manifest["resolved_input"]
    )

    pinned_source = manifest["pinned_source"]
    assert set(pinned_source) == {
        "constants",
        "default_input",
        "reference_commit",
        "reference_repository",
        "single_particle",
        "source_line_ranges",
    }
    assert pinned_source["reference_repository"] == (
        TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY
    )
    assert pinned_source["reference_commit"] == TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT
    assert pinned_source["default_input"] == {
        "path": TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE,
        "sha256": TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE_SHA256,
    }
    assert pinned_source["single_particle"] == {
        "path": (
            f"{TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY}/"
            f"{TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE}"
        ),
        "sha256": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
    }
    assert pinned_source["constants"] == {
        "path": (
            f"{TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY}/"
            f"{TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE}"
        ),
        "sha256": TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256,
    }
    assert pinned_source["source_line_ranges"] == (
        _COMPANION_INTERACTION_SOURCE_LINE_RANGES
    )

    assert set(manifest["environment"]) == {
        "numpy",
        "python",
        "zlib_compile",
        "zlib_runtime",
    }
    assert all(
        isinstance(version, str) and version
        for version in manifest["environment"].values()
    )
    assert manifest["generator_script"] == generator_path.name
    assert _is_sha256(manifest["generator_script_sha256"])
    assert manifest["generator_script_sha256"] == _companion_fixture_file_sha256(
        generator_path
    )
    assert manifest["fixture_npz"] == fixture_path.name
    assert _is_sha256(manifest["fixture_npz_sha256"])
    assert manifest["fixture_npz_sha256"] == _companion_fixture_file_sha256(fixture_path)

    assert set(manifest["arrays"]) == _COMPANION_INTERACTION_FIXTURE_ARRAY_KEYS
    assert set(arrays) == _COMPANION_INTERACTION_FIXTURE_ARRAY_KEYS
    record = manifest["arrays"]["source_intFT"]
    array = arrays["source_intFT"]
    assert set(record) == {"dtype", "sha256", "shape"}
    assert record["shape"] == [2, 3, 10, 10] == list(array.shape)
    assert record["dtype"] == "<f8" == array.dtype.str
    assert _is_sha256(record["sha256"])
    assert record["sha256"] == _companion_fixture_array_sha256(array)


def test_companion_source_interaction_spec_freezes_source_fields_and_lines() -> None:
    spec = TBGZeroFieldCompanionSourceInteractionSpec()

    assert TBG_ZERO_FIELD_COMPANION_INTERACTION_SOURCE_DSC_M == (
        2.5000000000000002e-8
    )
    assert TBG_ZERO_FIELD_COMPANION_INTERACTION_SOURCE_DSC_M != 25e-9
    assert spec.to_companion_input() == _COMPANION_INTERACTION_INHERITED_INPUT
    metadata = spec.to_metadata()
    assert metadata["reference_function"] == "gen_interaction"
    assert metadata["reference_lines"] == SOURCE_INTERACTION_REFERENCE_LINES == "205-258"
    assert SOURCE_INTERACTION_SUPPORT_REFERENCE_LINES == "220-239"
    assert SOURCE_INTERACTION_Q0_REFERENCE_LINES == "240-250"
    assert SOURCE_INTERACTION_FINITE_Q_KERNEL_REFERENCE_LINES == "251-257"
    assert metadata["source_units"] == {
        "b1_b2": "dimensionless_in_units_of_2*kD*sin(theta/2)",
        "coulomb_prefactor": "eV*m",
        "dsc": "m",
        "intFT": "eV",
        "physical_q": "m^-1",
        "theta": "degree",
        "total_real_space_area": "m^2",
    }
    assert metadata["fingerprint"] == spec.fingerprint

    with pytest.raises(ValueError, match="freezes NG1=5 and NG2=5"):
        replace(spec, NG1=4)
    with pytest.raises(ValueError, match="dsc_m=2.5000000000000002e-8 m"):
        replace(spec, dsc_m=25e-9)
    with pytest.raises(ValueError, match="gates must be either"):
        replace(spec, gates="none")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="include_q0 must be bool"):
        replace(spec, include_q0=np.bool_(True))  # type: ignore[arg-type]


def test_companion_source_interaction_port_matches_isolated_fixture(
    companion_interaction_source_fixture,
) -> None:
    manifest, arrays, _generator_path, _fixture_path = companion_interaction_source_fixture
    params = _small_companion_single_particle_params(N1=2, N2=3)
    result = solve_tbg_zero_field_companion_interaction(params)

    assert manifest["resolved_input"] == {
        **result.params.to_companion_input(),
        **result.spec.to_companion_input(),
    }
    np.testing.assert_allclose(
        result.intFT_ev,
        arrays["source_intFT"],
        rtol=5.0e-15,
        atol=0.0,
    )
    assert result.intFT_ev.shape == (2, 3, 10, 10)
    assert result.intFT_ev.dtype == np.dtype(np.float64)
    assert not result.intFT_ev.flags.writeable
    assert result.array_hashes.intFT_ev == _companion_fixture_array_sha256(
        result.intFT_ev
    )
    assert result.array_hashes.semantics == SOURCE_INTERACTION_ARRAY_HASH_SEMANTICS
    assert result.provenance.interaction_reference_lines == "205-258"
    assert result.provenance.interaction_support_reference_lines == "220-239"
    assert result.provenance.interaction_q0_reference_lines == "240-250"
    assert result.provenance.interaction_finite_q_kernel_reference_lines == "251-257"
    assert result.provenance.scientific_scope == TBG_ZERO_FIELD_COMPANION_INTERACTION_SCOPE
    assert result.provenance.scientific_scope == (
        "diagnostic_interaction_parity_only_not_production_HF_or_TDHF"
    )

    metadata = result.to_metadata()
    assert metadata["params"] == result.params.to_metadata()
    assert metadata["params_fingerprint"] == result.params.fingerprint
    assert metadata["rlv_geometry"] == result.rlv_geometry.to_metadata()
    assert metadata["rlv_geometry_fingerprint"] == result.rlv_geometry.fingerprint
    assert metadata["fingerprint"] == result.fingerprint


def test_companion_source_interaction_q0_interior_outside_support_and_source_units() -> None:
    params = _small_companion_single_particle_params(N1=2, N2=3)
    dual = solve_tbg_zero_field_companion_interaction(
        params,
        TBGZeroFieldCompanionSourceInteractionSpec(gates="dual", include_q0=True),
    )
    dual_without_q0 = solve_tbg_zero_field_companion_interaction(
        params,
        TBGZeroFieldCompanionSourceInteractionSpec(gates="dual", include_q0=False),
    )
    single = solve_tbg_zero_field_companion_interaction(
        params,
        TBGZeroFieldCompanionSourceInteractionSpec(gates="single", include_q0=True),
    )

    b1 = dual.rlv_geometry.b1
    b2 = dual.rlv_geometry.b2
    q_scale_m_inv = 2.0 * companion_kD * np.sin(params.theta_rad / 2.0)
    reciprocal_cell_area = abs(b1[0] * b2[1] - b1[1] * b2[0])
    area_m2 = (
        params.N1
        * params.N2
        * (4.0 * np.pi**2)
        / reciprocal_cell_area
        / q_scale_m_inv**2
    )
    U_ev_m = companion_echarge**2 / (2.0 * companion_epsilon0) / companion_echarge
    q0_index = (0, 0, dual.spec.NG1, dual.spec.NG2)
    np.testing.assert_allclose(
        dual.intFT_ev[q0_index],
        U_ev_m * dual.spec.dsc_m / area_m2,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        single.intFT_ev[q0_index],
        2.0 * U_ev_m * single.spec.dsc_m / area_m2,
        rtol=2.0e-15,
        atol=0.0,
    )
    assert dual_without_q0.intFT_ev[q0_index] == 0.0

    interior_index = (0, 0, dual.spec.NG1, dual.spec.NG2 + 1)
    modq_m_inv = np.linalg.norm(q_scale_m_inv * b2)
    expected_dual_ev = (
        U_ev_m
        * np.tanh(modq_m_inv * dual.spec.dsc_m)
        / modq_m_inv
        / area_m2
    )
    expected_single_ev = (
        U_ev_m
        * (1.0 - np.exp(-2.0 * single.spec.dsc_m * modq_m_inv))
        / modq_m_inv
        / area_m2
    )
    np.testing.assert_allclose(
        dual.intFT_ev[interior_index],
        expected_dual_ev,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        single.intFT_ev[interior_index],
        expected_single_ev,
        rtol=2.0e-15,
        atol=0.0,
    )
    assert dual_without_q0.intFT_ev[interior_index] == dual.intFT_ev[interior_index]

    R1 = dual.spec.NG1 * np.linalg.norm(
        b1 - b2 * np.dot(b1, b2) / np.dot(b2, b2)
    )
    R2 = dual.spec.NG1 * np.linalg.norm(
        b2 - b1 * np.dot(b1, b2) / np.dot(b1, b1)
    )
    radius = min(R1, R2)
    expected_support = np.zeros(dual.intFT_ev.shape, dtype=bool)
    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            for G1 in range(-dual.spec.NG1, dual.spec.NG1):
                for G2 in range(-dual.spec.NG2, dual.spec.NG2):
                    Q = (
                        ik1 * b1 / params.N1
                        + ik2 * b2 / params.N2
                        + G1 * b1
                        + G2 * b2
                    )
                    expected_support[
                        ik1,
                        ik2,
                        G1 + dual.spec.NG1,
                        G2 + dual.spec.NG2,
                    ] = np.linalg.norm(Q) < radius - 1.0e-5
    np.testing.assert_array_equal(dual.intFT_ev > 0.0, expected_support)
    outside_index = (0, 0, 0, 0)
    assert not expected_support[outside_index]
    assert dual.intFT_ev[outside_index] == 0.0

    metadata = dual.to_metadata()
    assert metadata["coulomb_prefactor_ev_m"] == dual.coulomb_prefactor_ev_m
    assert metadata["physical_q_scale_m_inv"] == dual.physical_q_scale_m_inv
    assert metadata["total_real_space_area_m2"] == dual.total_real_space_area_m2
    assert metadata["source_units"]["intFT"] == "eV"
    assert metadata["source_units"]["physical_q"] == "m^-1"


# ---------------------------------------------------------------------------
# Stage-4 source-faithful companion form-factor/HF-action diagnostic
# ---------------------------------------------------------------------------

_COMPANION_HF_ACTION_FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parent / "fixtures" / "tbg_companion_hf_action_v1"
)
_COMPANION_HF_ACTION_FIXTURE_ARRAY_KEYS = {
    "P",
    "P_ref",
    "source_H_D_ev",
    "source_H_E_ev",
    "source_H_SP_ev",
    "source_M_ev",
    "source_coeff",
    "source_energy_ev",
    "source_form",
    "source_form_branch",
    "source_form_raw",
    "source_raw_intFT_ev",
    "source_screened_intFT_ev",
    "source_sp_energy_ev",
    "source_tVE_ev",
}
_COMPANION_HF_ACTION_INPUT_OVERRIDES = {
    "N1": 2,
    "N2": 3,
    "Ng1": 3,
    "Ng2": 3,
    "n_active": 1,
    "theta": 1.08,
    "wAA": 0.07,
    "wAB": 0.11,
    "strain": 0.0,
    "varphi": 0.0,
}
_COMPANION_HF_ACTION_INHERITED_INTERACTION = {
    "NG1": 5,
    "NG2": 5,
    "dsc": 2.5000000000000002e-8,
    "gates": "dual",
    "include_q=0": True,
}
_COMPANION_HF_ACTION_INHERITED_HF = {
    "epsr": 10,
    "exchange": True,
    "boost1": 0,
    "boost2": 0,
}
_COMPANION_HF_ACTION_SOURCE_LINES = {
    "mainProgram.build_and_realify_form": "33-43",
    "mainProgram.screen_raw_intFT": "31",
    "mainProgram.zero_or_nonzero_boost": "47-54",
    "routines.calc_E": "81-97",
    "routines.calc_fock_matrix": "99-153",
    "routines.gen_H_SP": "6-22",
    "routines.gen_M_tVE": "24-79",
    "singleParticle.gen_form_factors": "389-440",
}


def _companion_stage4_params() -> TBGZeroFieldCompanionSingleParticleParams:
    return TBGZeroFieldCompanionSingleParticleParams(
        N1=2,
        N2=3,
        Ng1=3,
        Ng2=3,
        n_active=1,
        theta_deg=1.08,
        wAA_ev=0.07,
        wAB_ev=0.11,
        strain=0.0,
        strain_angle_deg=0.0,
    )


@pytest.fixture(scope="module")
def companion_hf_action_source_fixture():
    manifest_path = _COMPANION_HF_ACTION_FIXTURE_DIRECTORY / "manifest.json"
    generator_path = _COMPANION_HF_ACTION_FIXTURE_DIRECTORY / "generate_fixture.py"
    assert generator_path.is_file(), "Pinned companion HF-action generator is absent"
    assert manifest_path.is_file(), (
        "Pinned companion HF-action manifest is absent; explicitly run "
        "tests/fixtures/tbg_companion_hf_action_v1/generate_fixture.py first"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture_path = _COMPANION_HF_ACTION_FIXTURE_DIRECTORY / manifest["fixture_npz"]
    assert fixture_path.is_file(), (
        "Pinned companion HF-action NPZ is absent; explicitly run "
        "tests/fixtures/tbg_companion_hf_action_v1/generate_fixture.py first"
    )
    with np.load(fixture_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    return manifest, arrays, generator_path, fixture_path


@pytest.fixture(scope="module")
def companion_stage4_typed_inputs():
    params = _companion_stage4_params()
    single_particle = solve_tbg_zero_field_companion_single_particle(params)
    interaction = solve_tbg_zero_field_companion_interaction(params)
    return single_particle, interaction


def _source_gauge_single_particle(base, arrays):
    coeff = np.asarray(arrays["source_coeff"], dtype=np.complex128)
    sp_energy = np.asarray(arrays["source_sp_energy_ev"], dtype=np.float64)
    U_C2T = companion_C2T_symmetry(base.params, coeff)
    hashes = TBGZeroFieldCompanionSingleParticleArrayHashes.from_arrays(
        coeff=coeff,
        sp_energy_ev=sp_energy,
        U_C2T=U_C2T,
    )
    return replace(
        base,
        coeff=coeff,
        sp_energy_ev=sp_energy,
        U_C2T=U_C2T,
        array_hashes=hashes,
    )


def _companion_prepared_array_hashes(prepared, **overrides):
    arrays = {
        "form_raw": prepared.form_raw,
        "form": prepared.form,
        "screened_intFT_ev": prepared.screened_intFT_ev,
        "M_ev": prepared.M_ev,
        "tVE_ev": prepared.tVE_ev,
        "H_SP_ev": prepared.H_SP_ev,
        "sp_energy_ev": prepared.sp_energy_ev,
    }
    arrays.update(overrides)
    return type(prepared.array_hashes).from_arrays(**arrays)


def _assert_stage4_readonly_bytes_mutation_fails_closed(
    array: np.ndarray,
    callback,
    *,
    match: str,
) -> None:
    """Mutate one live byte, assert failure, then restore exact bytes and flags."""

    assert isinstance(array, np.ndarray)
    assert array.flags.c_contiguous
    original_writeable = array.flags.writeable
    original_bytes = array.view(np.uint8).reshape(-1).copy()
    try:
        array.setflags(write=True)
        byte_view = array.view(np.uint8).reshape(-1)
        byte_view[0] = np.uint8(int(byte_view[0]) ^ 1)
        array.setflags(write=False)
        with pytest.raises(ValueError, match=match):
            callback()
    finally:
        array.setflags(write=True)
        array.view(np.uint8).reshape(-1)[:] = original_bytes
        array.setflags(write=original_writeable)
    np.testing.assert_array_equal(
        array.view(np.uint8).reshape(-1),
        original_bytes,
    )
    assert array.flags.writeable == original_writeable


def _stage4_state_vector(
    coeff: np.ndarray,
    *,
    ik1: int,
    ik2: int,
    tau: int,
    band: int,
) -> np.ndarray:
    return np.asarray(coeff[ik1, ik2, :, :, tau, band, :]).reshape(-1)


def test_companion_hf_action_fixture_manifest_is_source_only_and_hash_bound(
    companion_hf_action_source_fixture,
) -> None:
    manifest, arrays, generator_path, fixture_path = companion_hf_action_source_fixture
    assert manifest["fixture_schema"] == (
        "mean_field.tbg.companion_hf_action.source_fixture"
    )
    assert manifest["fixture_schema_version"] == 1
    assert manifest["array_hash_semantics"] == (
        "source_fixture_integrity_and_same_environment_parity_not_cross_eigensolver_raw_gauge"
    )
    assert manifest["input_overrides"] == _COMPANION_HF_ACTION_INPUT_OVERRIDES
    assert manifest["inherited_interaction_input"] == (
        _COMPANION_HF_ACTION_INHERITED_INTERACTION
    )
    assert manifest["inherited_hf_input"] == _COMPANION_HF_ACTION_INHERITED_HF
    assert manifest["resolved_input_sha256"] == _companion_fixture_json_sha256(
        manifest["resolved_input"]
    )
    assert manifest["projector"]["scope"] == "generic_IVC_algebra_input_not_K_IVC_seed"
    assert "K-IVC" not in manifest["projector"]["definition"]
    assert manifest["projector"]["stored_orientation"] == (
        "P[alpha,beta]=<c_dagger_alpha c_beta>"
    )
    assert manifest["output_units"]["energy"] == TBG_ZERO_FIELD_COMPANION_HF_ACTION_ENERGY_UNITS

    pinned = manifest["pinned_source"]
    assert pinned["reference_repository"] == TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY
    assert pinned["reference_commit"] == TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT
    assert pinned["single_particle"]["sha256"] == TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256
    assert pinned["routines"]["sha256"] == TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE_SHA256
    assert pinned["main_program"]["sha256"] == (
        TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE_SHA256
    )
    assert pinned["hf_input"]["sha256"] == TBG_ZERO_FIELD_COMPANION_HF_INPUT_SOURCE_SHA256
    assert pinned["source_line_ranges"] == _COMPANION_HF_ACTION_SOURCE_LINES

    assert set(manifest["environment"]) == {
        "byteorder",
        "numpy",
        "platform",
        "python",
        "zlib_compile",
        "zlib_runtime",
    }
    assert all(isinstance(value, str) and value for value in manifest["environment"].values())
    assert manifest["generator_script"] == generator_path.name
    assert manifest["generator_script_sha256"] == _companion_fixture_file_sha256(
        generator_path
    )
    assert manifest["fixture_npz"] == fixture_path.name
    assert manifest["fixture_npz_sha256"] == _companion_fixture_file_sha256(fixture_path)
    assert set(manifest["arrays"]) == _COMPANION_HF_ACTION_FIXTURE_ARRAY_KEYS
    assert set(arrays) == _COMPANION_HF_ACTION_FIXTURE_ARRAY_KEYS
    for key in sorted(arrays):
        record = manifest["arrays"][key]
        assert record["shape"] == list(arrays[key].shape)
        assert record["dtype"] == arrays[key].dtype.str
        assert record["sha256"] == _companion_fixture_array_sha256(arrays[key])

    assert arrays["source_coeff"].shape == (2, 3, 6, 6, 2, 2, 4)
    assert arrays["source_form_raw"].shape == (2, 3, 2, 3, 10, 10, 2, 2, 2)
    assert arrays["source_M_ev"].shape == arrays["source_form_raw"].shape
    assert arrays["source_tVE_ev"].shape == (24, 24, 4)
    assert arrays["source_H_SP_ev"].shape == (2, 3, 2, 4, 4)
    assert arrays["source_energy_ev"].shape == (4,)
    assert arrays["source_form_branch"].shape == ()
    assert arrays["source_form_branch"].item() == manifest["form_branch"] == "real"
    assert manifest["form_real_threshold"] == 1.0e-9
    assert manifest["form_raw_max_abs_imag"] == float(
        np.max(np.abs(np.imag(arrays["source_form_raw"])))
    )
    assert manifest["form_raw_max_abs_imag"] <= manifest["form_real_threshold"]
    assert arrays["source_form_raw"].dtype == np.dtype("<c16")
    for name in ("source_form", "source_M_ev", "source_tVE_ev"):
        assert arrays[name].dtype == np.dtype("<f8")
    np.testing.assert_array_equal(
        arrays["source_form"],
        np.real(arrays["source_form_raw"]),
    )


def test_companion_hf_action_source_lines_spec_and_provenance_are_pinned(
    companion_stage4_typed_inputs,
) -> None:
    single_particle, interaction = companion_stage4_typed_inputs
    spec = TBGZeroFieldCompanionHFActionSpec()
    prepared = prepare_tbg_zero_field_companion_hf_action(
        single_particle,
        interaction,
        spec=spec,
    )
    assert spec.to_companion_input() == {
        "epsr": 10.0,
        "exchange": True,
        "boost1": 0,
        "boost2": 0,
    }
    assert prepared.provenance.form_factor_reference_lines == (
        TBG_ZERO_FIELD_COMPANION_FORM_FACTOR_REFERENCE_LINES
    )
    assert TBG_ZERO_FIELD_COMPANION_FORM_FACTOR_REFERENCE_LINES == "389-440"
    assert prepared.provenance.gen_H_SP_reference_lines == (
        TBG_ZERO_FIELD_COMPANION_GEN_H_SP_REFERENCE_LINES
    )
    assert TBG_ZERO_FIELD_COMPANION_GEN_H_SP_REFERENCE_LINES == "6-22"
    assert prepared.provenance.gen_M_tVE_reference_lines == (
        TBG_ZERO_FIELD_COMPANION_GEN_M_TVE_REFERENCE_LINES
    )
    assert TBG_ZERO_FIELD_COMPANION_GEN_M_TVE_REFERENCE_LINES == "24-79"
    assert prepared.provenance.calc_E_reference_lines == (
        TBG_ZERO_FIELD_COMPANION_CALC_E_REFERENCE_LINES
    )
    assert TBG_ZERO_FIELD_COMPANION_CALC_E_REFERENCE_LINES == "81-97"
    assert prepared.provenance.calc_fock_matrix_reference_lines == (
        TBG_ZERO_FIELD_COMPANION_CALC_FOCK_MATRIX_REFERENCE_LINES
    )
    assert TBG_ZERO_FIELD_COMPANION_CALC_FOCK_MATRIX_REFERENCE_LINES == "99-153"
    assert prepared.provenance.scientific_scope == TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCOPE
    assert "not_production" in prepared.provenance.scientific_scope
    assert prepared.array_hashes.semantics == (
        TBG_ZERO_FIELD_COMPANION_HF_ACTION_ARRAY_HASH_SEMANTICS
    )
    assert prepared.single_particle_source is single_particle
    assert prepared.interaction_source is interaction
    assert prepared.single_particle_fingerprint == single_particle.fingerprint
    assert prepared.interaction_fingerprint == interaction.fingerprint
    metadata = prepared.to_metadata()
    assert metadata["single_particle_source_fingerprint"] == single_particle.fingerprint
    assert metadata["interaction_source_fingerprint"] == interaction.fingerprint


def test_companion_prepared_rejects_source_receipt_and_param_substitution(
    companion_stage4_typed_inputs,
) -> None:
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        single_particle,
        interaction,
    )
    with pytest.raises(TypeError, match="single_particle_source must be"):
        replace(prepared, single_particle_source=object())
    with pytest.raises(TypeError, match="interaction_source must be"):
        replace(prepared, interaction_source=object())
    with pytest.raises(ValueError, match="single_particle_fingerprint does not match"):
        replace(prepared, single_particle_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="interaction_fingerprint does not match"):
        replace(prepared, interaction_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="params differ from single_particle_source"):
        replace(
            prepared,
            params=replace(prepared.params, theta_deg=prepared.params.theta_deg + 0.01),
        )


def test_companion_main_program_realify_form_keeps_tiny_synthetic_complex_branch() -> None:
    form_raw = np.asarray([1.0 + 2.0e-9j], dtype=np.complex128)
    form, branch, max_abs_imag = companion_main_program_realify_form(form_raw)
    assert branch == "complex"
    assert max_abs_imag == 2.0e-9
    assert form.dtype == np.dtype(np.complex128)
    np.testing.assert_array_equal(form, form_raw)


def test_companion_form_factor_saturates_guard_and_keeps_roll_zero_fill_carries() -> None:
    params = _companion_stage4_params()
    shape = (params.N1, params.N2, 2 * params.Ng1, 2 * params.Ng2, 4)
    c = np.zeros(shape, dtype=np.complex128)
    cp = np.zeros(shape, dtype=np.complex128)
    c[0, 0, 0, 0, 0] = 1.0
    cp[0, 0, 0, 0, 0] = 1.0
    form = companion_gen_form_factors(params, c, cp, NG1=5, NG2=5)
    assert form.shape == (2, 3, 2, 3, 10, 10)
    assert np.count_nonzero(form) > 0
    # A periodic parent roll would preserve this norm for every reciprocal shift;
    # the pinned zero-fill boundary does not.
    assert np.count_nonzero(form[0, 0, 0, 0]) < form.shape[4] * form.shape[5]
    with pytest.raises(ValueError, match=r"NG1 <= 2\*Ng1-1"):
        companion_gen_form_factors(params, c, cp, NG1=6, NG2=5)
    with pytest.raises(ValueError, match=r"NG2 <= 2\*Ng2-1"):
        companion_gen_form_factors(params, c, cp, NG1=5, NG2=6)


def test_companion_hf_action_direct_port_matches_source_fixture_in_source_gauge(
    companion_hf_action_source_fixture,
) -> None:
    manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    params = _companion_stage4_params()
    form_raw = companion_gen_full_form_factors(
        params,
        arrays["source_coeff"],
        NG1=5,
        NG2=5,
    )
    np.testing.assert_allclose(form_raw, arrays["source_form_raw"], rtol=0.0, atol=2.0e-14)
    form, branch, max_abs_imag = companion_main_program_realify_form(form_raw)
    assert branch == manifest["form_branch"] == arrays["source_form_branch"].item() == "real"
    assert max_abs_imag <= 1.0e-9
    np.testing.assert_allclose(
        max_abs_imag,
        manifest["form_raw_max_abs_imag"],
        rtol=0.0,
        atol=2.0e-14,
    )
    assert form_raw.dtype == np.dtype(np.complex128)
    assert form.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(form, np.real(form_raw))
    np.testing.assert_allclose(form, arrays["source_form"], rtol=0.0, atol=2.0e-14)

    M, tVE = companion_gen_M_tVE(
        params,
        form_raw,
        arrays["source_screened_intFT_ev"],
        exchange=True,
    )
    assert M.dtype == np.dtype(np.float64)
    assert tVE.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(M, arrays["source_M_ev"], rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(tVE, arrays["source_tVE_ev"], rtol=0.0, atol=2.0e-13)
    H_SP = companion_gen_H_SP(params, arrays["source_sp_energy_ev"])
    np.testing.assert_array_equal(H_SP, arrays["source_H_SP_ev"])
    action = companion_calc_fock_matrix(
        params,
        arrays["P"] - arrays["P_ref"],
        form,
        M,
        tVE,
    )
    np.testing.assert_allclose(action.H_D_ev, arrays["source_H_D_ev"], rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(action.H_E_ev, arrays["source_H_E_ev"], rtol=0.0, atol=2.0e-13)
    energy = companion_calc_E(
        params,
        arrays["P"],
        arrays["P_ref"],
        arrays["source_sp_energy_ev"],
        action,
    )
    np.testing.assert_allclose(energy.components_ev, arrays["source_energy_ev"], rtol=0.0, atol=2.0e-13)
    assert energy.units == "finite_system_eV_not_per_moire_cell"


def test_companion_stage2_coefficients_are_real_sign_alignable_and_actions_covary(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    params = single_particle.params
    bands = params.active_band_count
    gauge = np.empty((params.N1, params.N2, 2 * bands), dtype=np.complex128)
    aligned = np.array(single_particle.coeff, copy=True)
    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            for tau in range(2):
                for band in range(bands):
                    source_state = _stage4_state_vector(
                        arrays["source_coeff"],
                        ik1=ik1,
                        ik2=ik2,
                        tau=tau,
                        band=band,
                    )
                    port_state = _stage4_state_vector(
                        single_particle.coeff,
                        ik1=ik1,
                        ik2=ik2,
                        tau=tau,
                        band=band,
                    )
                    overlap = np.vdot(source_state, port_state)
                    np.testing.assert_allclose(abs(overlap), 1.0, rtol=0.0, atol=5.0e-11)
                    phase = overlap / abs(overlap)
                    assert abs(phase.imag) < 5.0e-10
                    np.testing.assert_allclose(abs(phase.real), 1.0, rtol=0.0, atol=5.0e-10)
                    aligned[ik1, ik2, :, :, tau, band, :] *= np.conj(phase)
                    gauge[ik1, ik2, tau * bands + band] = phase
    np.testing.assert_allclose(aligned, arrays["source_coeff"], rtol=0.0, atol=5.0e-11)

    source_single_particle = _source_gauge_single_particle(single_particle, arrays)
    source_prepared = prepare_tbg_zero_field_companion_hf_action(
        source_single_particle,
        interaction,
    )
    port_prepared = prepare_tbg_zero_field_companion_hf_action(
        single_particle,
        interaction,
    )
    source_evaluation = source_prepared.evaluate(arrays["P"], arrays["P_ref"])
    D_left = gauge[:, :, None, :, None]
    D_right_conj = np.conj(gauge[:, :, None, None, :])
    P_port = D_left * arrays["P"] * D_right_conj
    P_ref_port = D_left * arrays["P_ref"] * D_right_conj
    port_evaluation = port_prepared.evaluate(P_port, P_ref_port)
    expected_H_D_port = np.conj(D_left) * source_evaluation.H_D_ev * np.conj(D_right_conj)
    expected_H_E_port = np.conj(D_left) * source_evaluation.H_E_ev * np.conj(D_right_conj)
    np.testing.assert_allclose(port_evaluation.H_D_ev, expected_H_D_port, rtol=0.0, atol=2.0e-10)
    np.testing.assert_allclose(port_evaluation.H_E_ev, expected_H_E_port, rtol=0.0, atol=2.0e-10)
    np.testing.assert_allclose(
        port_evaluation.energy_components_ev,
        source_evaluation.energy_components_ev,
        rtol=0.0,
        atol=2.0e-10,
    )


def test_companion_prepared_source_gauge_matches_fixture_and_is_immutable_hashed(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    source_single_particle = _source_gauge_single_particle(single_particle, arrays)
    prepared = prepare_tbg_zero_field_companion_hf_action(
        source_single_particle,
        interaction,
    )
    np.testing.assert_allclose(prepared.form_raw, arrays["source_form_raw"], rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(prepared.form, arrays["source_form"], rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        prepared.screened_intFT_ev,
        arrays["source_screened_intFT_ev"],
        rtol=5.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(prepared.M_ev, arrays["source_M_ev"], rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(prepared.tVE_ev, arrays["source_tVE_ev"], rtol=0.0, atol=2.0e-13)
    assert prepared.form_branch == arrays["source_form_branch"].item() == "real"
    assert prepared.form_raw.dtype == np.dtype(np.complex128)
    assert prepared.form.dtype == np.dtype(np.float64)
    assert prepared.M_ev.dtype == np.dtype(np.float64)
    assert prepared.tVE_ev.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(prepared.form, np.real(prepared.form_raw))
    assert prepared.spec.epsr == 10.0
    np.testing.assert_array_equal(
        prepared.screened_intFT_ev,
        np.asarray(interaction.intFT_ev / prepared.spec.epsr),
    )
    assert prepared.raw_interaction_array_sha256 == interaction.array_hashes.intFT_ev
    assert prepared.residuals.screening_roundtrip_max_abs_ev == float(
        np.max(
            np.abs(
                prepared.screened_intFT_ev * prepared.spec.epsr
                - interaction.intFT_ev
            )
        )
    )
    for array in (
        prepared.form_raw,
        prepared.form,
        prepared.screened_intFT_ev,
        prepared.M_ev,
        prepared.tVE_ev,
        prepared.H_SP_ev,
        prepared.sp_energy_ev,
    ):
        assert not array.flags.writeable
    assert prepared.array_hashes.form_raw == _companion_fixture_array_sha256(
        np.asarray(prepared.form_raw, dtype=np.dtype("<c16"))
    )
    assert prepared.memory_estimate.form_elements == prepared.form.size
    assert prepared.memory_estimate.tVE_elements == prepared.tVE_ev.size
    assert prepared.memory_estimate.hamiltonian_elements == prepared.H_SP_ev.size
    assert prepared.memory_estimate.prepared_arrays_bytes == sum(
        array.nbytes
        for array in (
            prepared.form_raw,
            prepared.form,
            prepared.screened_intFT_ev,
            prepared.M_ev,
            prepared.tVE_ev,
            prepared.H_SP_ev,
            prepared.sp_energy_ev,
        )
    )
    assert prepared.memory_estimate.raw_form_bytes == 450 * 1024
    assert prepared.memory_estimate.tVE_bytes in (18 * 1024, 36 * 1024)
    assert prepared.to_metadata()["fingerprint"] == prepared.fingerprint


def test_companion_screening_is_exact_source_division_and_raw_hash(
    companion_stage4_typed_inputs,
) -> None:
    single_particle, interaction = companion_stage4_typed_inputs
    spec = TBGZeroFieldCompanionHFActionSpec(epsr=7.3)
    prepared = prepare_tbg_zero_field_companion_hf_action(
        single_particle,
        interaction,
        spec=spec,
    )
    np.testing.assert_array_equal(
        prepared.screened_intFT_ev,
        np.asarray(interaction.intFT_ev / spec.epsr),
    )
    assert prepared.raw_interaction_array_sha256 == interaction.array_hashes.intFT_ev
    assert prepared.residuals.screening_roundtrip_max_abs_ev == float(
        np.max(
            np.abs(
                prepared.screened_intFT_ev * spec.epsr
                - interaction.intFT_ev
            )
        )
    )
    with pytest.raises(ValueError, match="residuals do not match"):
        replace(
            prepared,
            residuals=replace(
                prepared.residuals,
                screening_roundtrip_max_abs_ev=(
                    prepared.residuals.screening_roundtrip_max_abs_ev + 1.0e-18
                ),
            ),
        )


def test_companion_prepared_rejects_mutated_typed_source_arrays(
    companion_stage4_typed_inputs,
) -> None:
    single_particle, interaction = companion_stage4_typed_inputs

    tampered_single_particle = replace(single_particle)
    tampered_single_particle.coeff.setflags(write=True)
    tampered_single_particle.coeff.flat[0] += 1.0e-6
    tampered_single_particle.coeff.setflags(write=False)
    with pytest.raises(
        ValueError,
        match=r"single_particle_source\.coeff no longer matches its source hash",
    ):
        prepare_tbg_zero_field_companion_hf_action(
            tampered_single_particle,
            interaction,
        )

    tampered_interaction = replace(interaction)
    tampered_interaction.intFT_ev.setflags(write=True)
    tampered_interaction.intFT_ev.flat[0] += 1.0e-6
    tampered_interaction.intFT_ev.setflags(write=False)
    with pytest.raises(
        ValueError,
        match=r"interaction_source\.intFT_ev no longer matches its source hash",
    ):
        prepare_tbg_zero_field_companion_hf_action(
            single_particle,
            tampered_interaction,
        )


@pytest.mark.parametrize(
    ("source_array_name", "entrypoint"),
    (
        ("coeff", "fingerprint"),
        ("sp_energy_ev", "metadata"),
        ("U_C2T", "evaluate"),
    ),
)
def test_companion_prepared_live_validation_rehashes_post_prepare_stage2_arrays(
    companion_stage4_typed_inputs,
    source_array_name,
    entrypoint,
) -> None:
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        single_particle,
        interaction,
    )
    source_array = getattr(prepared.single_particle_source, source_array_name)
    original_value = source_array.flat[0].item()
    original_writeable = source_array.flags.writeable
    try:
        source_array.setflags(write=True)
        source_array.flat[0] = original_value + 1.0e-6
        source_array.setflags(write=False)
        with pytest.raises(
            ValueError,
            match=rf"single_particle_source\.{source_array_name} no longer matches its source hash",
        ):
            if entrypoint == "fingerprint":
                _ = prepared.fingerprint
            elif entrypoint == "metadata":
                prepared.to_metadata()
            else:
                zeros = np.zeros_like(prepared.H_SP_ev, dtype=np.complex128)
                prepared.evaluate(zeros, zeros)
    finally:
        source_array.setflags(write=True)
        source_array.flat[0] = original_value
        source_array.setflags(write=original_writeable)


def test_companion_prepared_live_validation_rehashes_post_prepare_stage3_source(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, arrays),
        interaction,
    )
    evaluation = prepared.evaluate(arrays["P"], arrays["P_ref"])
    source_array = prepared.interaction_source.intFT_ev
    original_value = source_array.flat[0].item()
    original_writeable = source_array.flags.writeable
    expected_error = (
        r"interaction_source\.intFT_ev no longer matches its source hash"
    )
    try:
        source_array.setflags(write=True)
        source_array.flat[0] = original_value + 1.0e-6
        source_array.setflags(write=False)
        with pytest.raises(ValueError, match=expected_error):
            _ = prepared.fingerprint
        with pytest.raises(ValueError, match=expected_error):
            prepared.to_metadata()
        with pytest.raises(ValueError, match=expected_error):
            prepared.evaluate(arrays["P"], arrays["P_ref"])
        with pytest.raises(ValueError, match=expected_error):
            replace(evaluation)
    finally:
        source_array.setflags(write=True)
        source_array.flat[0] = original_value
        source_array.setflags(write=original_writeable)


@pytest.mark.parametrize(
    ("array_name", "entrypoint"),
    (
        ("form_raw", "fingerprint"),
        ("form", "metadata"),
        ("screened_intFT_ev", "evaluate"),
        ("M_ev", "evaluation"),
        ("tVE_ev", "fingerprint"),
        ("H_SP_ev", "metadata"),
        ("sp_energy_ev", "evaluate"),
    ),
)
def test_companion_prepared_live_arrays_fail_closed_after_setflags_mutation(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
    array_name,
    entrypoint,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, arrays),
        interaction,
    )
    existing_evaluation = (
        prepared.evaluate(arrays["P"], arrays["P_ref"])
        if entrypoint == "evaluation"
        else None
    )
    callbacks = {
        "fingerprint": lambda: prepared.fingerprint,
        "metadata": prepared.to_metadata,
        "evaluate": lambda: prepared.evaluate(arrays["P"], arrays["P_ref"]),
        "evaluation": lambda: existing_evaluation.fingerprint,
    }
    _assert_stage4_readonly_bytes_mutation_fails_closed(
        getattr(prepared, array_name),
        callbacks[entrypoint],
        match="prepared array_hashes no longer match live prepared arrays",
    )


@pytest.mark.parametrize(
    ("array_name", "entrypoint"),
    (
        ("density_delta", "action"),
        ("H_D_ev", "evaluation"),
        ("H_E_ev", "action"),
        ("H_interaction_ev", "evaluation"),
    ),
)
def test_companion_action_live_arrays_fail_closed_after_setflags_mutation(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
    array_name,
    entrypoint,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, arrays),
        interaction,
    )
    evaluation = prepared.evaluate(arrays["P"], arrays["P_ref"])
    callbacks = {
        "action": lambda: evaluation.action.fingerprint,
        "evaluation": lambda: evaluation.fingerprint,
    }
    _assert_stage4_readonly_bytes_mutation_fails_closed(
        getattr(evaluation.action, array_name),
        callbacks[entrypoint],
        match="action array_hashes no longer match live companion HF-action arrays",
    )


@pytest.mark.parametrize(
    "array_name",
    ("projector", "reference", "density_delta", "H_SP_ev", "H_total_ev"),
)
def test_companion_evaluation_live_arrays_fail_closed_after_setflags_mutation(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
    array_name,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, arrays),
        interaction,
    )
    evaluation = prepared.evaluate(arrays["P"], arrays["P_ref"])
    _assert_stage4_readonly_bytes_mutation_fails_closed(
        getattr(evaluation, array_name),
        lambda: evaluation.fingerprint,
        match="evaluation array_hashes no longer match live evaluation arrays",
    )


@pytest.mark.parametrize("entrypoint", ("energy", "evaluation"))
def test_companion_energy_components_fail_closed_after_setflags_mutation(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
    entrypoint,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, arrays),
        interaction,
    )
    evaluation = prepared.evaluate(arrays["P"], arrays["P_ref"])
    callbacks = {
        "energy": lambda: evaluation.energy.fingerprint,
        "evaluation": lambda: evaluation.fingerprint,
    }
    expected_error = (
        "energy.components_ev no longer matches its construction hash"
        if entrypoint == "energy"
        else "evaluation array_hashes no longer match live evaluation arrays"
    )
    _assert_stage4_readonly_bytes_mutation_fails_closed(
        evaluation.energy.components_ev,
        callbacks[entrypoint],
        match=expected_error,
    )


def test_companion_prepared_rejects_coherent_derived_array_substitutions(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, arrays),
        interaction,
    )

    altered_sp_energy = np.asarray(prepared.sp_energy_ev + 1.0e-6)
    altered_sp_H_SP = companion_gen_H_SP(prepared.params, altered_sp_energy)
    with pytest.raises(ValueError, match="sp_energy_ev does not exactly equal"):
        replace(
            prepared,
            sp_energy_ev=altered_sp_energy,
            H_SP_ev=altered_sp_H_SP,
            array_hashes=_companion_prepared_array_hashes(
                prepared,
                sp_energy_ev=altered_sp_energy,
                H_SP_ev=altered_sp_H_SP,
            ),
        )

    altered_form_raw = np.array(prepared.form_raw, copy=True)
    altered_form_raw[0, 0, 0, 0, 0, 0, 0, 0, 0] += 1.0e-6
    with pytest.raises(ValueError, match="form_raw does not match recomputation"):
        replace(
            prepared,
            form_raw=altered_form_raw,
            array_hashes=_companion_prepared_array_hashes(
                prepared,
                form_raw=altered_form_raw,
            ),
        )

    altered_form = np.array(prepared.form, copy=True)
    altered_form[0, 0, 0, 0, 0, 0, 0, 0, 0] += 1.0e-6
    with pytest.raises(ValueError, match="form does not exactly match"):
        replace(
            prepared,
            form=altered_form,
            array_hashes=_companion_prepared_array_hashes(
                prepared,
                form=altered_form,
            ),
        )

    with pytest.raises(ValueError, match="form_branch does not match"):
        replace(
            prepared,
            form_branch="complex",
            residuals=replace(prepared.residuals, form_branch="complex"),
            memory_estimate=type(prepared.memory_estimate).from_shapes(
                prepared.params,
                NG1=prepared.interaction_NG1,
                NG2=prepared.interaction_NG2,
                form_branch="complex",
                exchange=prepared.spec.exchange,
            ),
        )

    altered_H_SP = np.array(prepared.H_SP_ev, copy=True)
    altered_H_SP[0, 0, 0, 0, 0] += 1.0e-6
    with pytest.raises(ValueError, match="H_SP_ev does not exactly equal gen_H_SP"):
        replace(
            prepared,
            H_SP_ev=altered_H_SP,
            array_hashes=_companion_prepared_array_hashes(
                prepared,
                H_SP_ev=altered_H_SP,
            ),
        )

    altered_M = np.array(prepared.M_ev, copy=True)
    altered_M[0, 0, 0, 0, 0, 0, 0, 0, 0] += 1.0e-6
    with pytest.raises(ValueError, match="M_ev does not exactly equal gen_M_tVE"):
        replace(
            prepared,
            M_ev=altered_M,
            array_hashes=_companion_prepared_array_hashes(
                prepared,
                M_ev=altered_M,
            ),
        )

    altered_tVE = np.array(prepared.tVE_ev, copy=True)
    altered_tVE[0, 0, 0] += 1.0e-6
    with pytest.raises(ValueError, match="tVE_ev does not exactly equal gen_M_tVE"):
        replace(
            prepared,
            tVE_ev=altered_tVE,
            array_hashes=_companion_prepared_array_hashes(
                prepared,
                tVE_ev=altered_tVE,
            ),
        )

    altered_screened_intFT = np.asarray(prepared.screened_intFT_ev * 1.125)
    altered_screened_M, altered_screened_tVE = companion_gen_M_tVE(
        prepared.params,
        prepared.form,
        altered_screened_intFT,
        exchange=prepared.spec.exchange,
    )
    with pytest.raises(
        ValueError,
        match="screened_intFT_ev does not exactly equal interaction_source",
    ):
        replace(
            prepared,
            screened_intFT_ev=altered_screened_intFT,
            M_ev=altered_screened_M,
            tVE_ev=altered_screened_tVE,
            array_hashes=_companion_prepared_array_hashes(
                prepared,
                screened_intFT_ev=altered_screened_intFT,
                M_ev=altered_screened_M,
                tVE_ev=altered_screened_tVE,
            ),
        )


def test_companion_prepared_rejects_raw_hash_substitution(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, arrays),
        interaction,
    )
    with pytest.raises(ValueError, match="raw_interaction_array_sha256 does not match"):
        replace(prepared, raw_interaction_array_sha256="0" * 64)


def test_companion_action_zero_linearity_hermiticity_spin_and_manual_element(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, arrays),
        interaction,
    )
    delta = arrays["P"] - arrays["P_ref"]
    zero = companion_calc_fock_matrix(
        prepared.params,
        np.zeros_like(delta),
        prepared.form,
        prepared.M_ev,
        prepared.tVE_ev,
    )
    np.testing.assert_array_equal(zero.H_D_ev, np.zeros_like(zero.H_D_ev))
    np.testing.assert_array_equal(zero.H_E_ev, np.zeros_like(zero.H_E_ev))
    action = companion_calc_fock_matrix(
        prepared.params,
        delta,
        prepared.form,
        prepared.M_ev,
        prepared.tVE_ev,
    )
    scaled = companion_calc_fock_matrix(
        prepared.params,
        0.375 * delta,
        prepared.form,
        prepared.M_ev,
        prepared.tVE_ev,
    )
    np.testing.assert_allclose(scaled.H_D_ev, 0.375 * action.H_D_ev, rtol=2.0e-14, atol=2.0e-15)
    np.testing.assert_allclose(scaled.H_E_ev, 0.375 * action.H_E_ev, rtol=2.0e-14, atol=2.0e-15)
    np.testing.assert_allclose(action.H_interaction_ev, action.H_interaction_ev.conj().swapaxes(-1, -2), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(action.H_D_ev[:, :, 0], action.H_D_ev[:, :, 1], rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(action.H_E_ev[:, :, 0], action.H_E_ev[:, :, 1], rtol=0.0, atol=1.0e-13)

    # Independent single-element Hartree sum in the exact stored orientation.
    split = delta.reshape(2, 3, 2, 2, 2, 2, 2)
    tau, a, b = 0, 0, 1
    expected = 0.0j
    for iG1 in range(10):
        for iG2 in range(10):
            source_sum = 0.0j
            for ik1 in range(2):
                for ik2 in range(3):
                    for spin in range(2):
                        for source_tau in range(2):
                            for aa in range(2):
                                for bb in range(2):
                                    source_sum += (
                                        prepared.M_ev[
                                            ik1,
                                            ik2,
                                            0,
                                            0,
                                            iG1,
                                            iG2,
                                            source_tau,
                                            aa,
                                            bb,
                                        ]
                                        * split[
                                            ik1,
                                            ik2,
                                            spin,
                                            source_tau,
                                            bb,
                                            source_tau,
                                            aa,
                                        ]
                                    )
            expected += prepared.form[0, 0, 0, 0, iG1, iG2, tau, a, b] * source_sum
    np.testing.assert_allclose(action.H_D_ev[0, 0, 0, tau * 2 + a, tau * 2 + b], expected, rtol=0.0, atol=2.0e-13)
    index = (1, 2, 1, 1, 4, 6, 1, 0, 1)
    np.testing.assert_allclose(
        prepared.M_ev[index],
        prepared.screened_intFT_ev[index[2], index[3], index[4], index[5]]
        * np.conj(prepared.form[index]),
        rtol=0.0,
        atol=0.0,
    )


def test_companion_energy_uses_stored_orientation_and_finite_system_ev(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, arrays),
        interaction,
    )
    evaluation = prepared.evaluate(arrays["P"], arrays["P_ref"])
    delta = arrays["P"] - arrays["P_ref"]
    Psplit = arrays["P"].reshape(2, 3, 2, 2, 2, 2, 2)
    manual_kinetic = np.einsum(
        "kKta,kKstata->",
        arrays["source_sp_energy_ev"],
        Psplit,
        optimize=True,
    )
    manual_D = 0.5 * np.einsum(
        "kpsAB,kpsAB->",
        evaluation.H_D_ev,
        delta,
        optimize=True,
    )
    manual_E = 0.5 * np.einsum(
        "kpsAB,kpsAB->",
        evaluation.H_E_ev,
        delta,
        optimize=True,
    )
    np.testing.assert_allclose(
        evaluation.energy_components_ev,
        np.real([manual_kinetic + manual_D + manual_E, manual_kinetic, manual_D, manual_E]),
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(evaluation.energy_components_ev, arrays["source_energy_ev"], rtol=0.0, atol=2.0e-13)
    assert evaluation.energy.units == TBG_ZERO_FIELD_COMPANION_HF_ACTION_ENERGY_UNITS
    assert evaluation.prepared is prepared
    assert evaluation.H_SP_ev is prepared.H_SP_ev
    assert not evaluation.energy_components_ev.flags.writeable
    assert not evaluation.H_SP_ev.flags.writeable
    assert not evaluation.H_total_ev.flags.writeable
    assert evaluation.array_hashes.H_SP_ev == prepared.array_hashes.H_SP_ev
    assert evaluation.array_hashes.density_delta == evaluation.action.array_hashes.density_delta
    assert evaluation.energy.projector_sha256 == evaluation.array_hashes.projector
    assert evaluation.energy.reference_sha256 == evaluation.array_hashes.reference
    assert evaluation.residuals.density_subtraction_max_abs == 0.0
    assert evaluation.residuals.H_total_closure_max_abs_ev == 0.0
    assert evaluation.residuals.energy_action_binding_residual == 0.0


def test_companion_evaluation_rejects_tampered_residuals_total_and_mixed_pair(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, arrays),
        interaction,
    )
    evaluation = prepared.evaluate(arrays["P"], arrays["P_ref"])

    for residual_name in (
        "projector_hermiticity_max_abs",
        "reference_hermiticity_max_abs",
        "density_subtraction_max_abs",
        "H_total_closure_max_abs_ev",
        "energy_action_binding_residual",
    ):
        with pytest.raises(ValueError, match="residuals do not match"):
            replace(
                evaluation,
                residuals=replace(
                    evaluation.residuals,
                    **{
                        residual_name: (
                            getattr(evaluation.residuals, residual_name) + 1.0e-15
                        )
                    },
                ),
            )

    tampered_H_total = np.array(evaluation.H_total_ev, copy=True)
    tampered_H_total[0, 0, 0, 0, 0] += 1.0e-6
    tampered_hashes = TBGZeroFieldCompanionHFEvaluationArrayHashes.from_arrays(
        projector=evaluation.projector,
        reference=evaluation.reference,
        density_delta=evaluation.density_delta,
        H_SP_ev=evaluation.H_SP_ev,
        H_total_ev=tampered_H_total,
        energy_components_ev=evaluation.energy.components_ev,
    )
    with pytest.raises(ValueError, match="H_total_ev must equal H_SP_ev"):
        replace(
            evaluation,
            H_total_ev=tampered_H_total,
            array_hashes=tampered_hashes,
        )

    common_shift = np.zeros_like(evaluation.projector)
    common_shift[..., 1, 1] = 0.125
    mixed_projector = evaluation.projector + common_shift
    mixed_reference = evaluation.reference + common_shift
    np.testing.assert_array_equal(
        mixed_projector - mixed_reference,
        evaluation.density_delta,
    )
    mixed_hashes = TBGZeroFieldCompanionHFEvaluationArrayHashes.from_arrays(
        projector=mixed_projector,
        reference=mixed_reference,
        density_delta=evaluation.density_delta,
        H_SP_ev=evaluation.H_SP_ev,
        H_total_ev=evaluation.H_total_ev,
        energy_components_ev=evaluation.energy.components_ev,
    )
    with pytest.raises(ValueError, match="energy projector hash"):
        replace(
            evaluation,
            projector=mixed_projector,
            reference=mixed_reference,
            array_hashes=mixed_hashes,
        )
    mixed_projector_bound_energy = replace(
        evaluation.energy,
        projector_sha256=mixed_hashes.projector,
    )
    with pytest.raises(ValueError, match="energy reference hash"):
        replace(
            evaluation,
            projector=mixed_projector,
            reference=mixed_reference,
            energy=mixed_projector_bound_energy,
            array_hashes=mixed_hashes,
        )

    with pytest.raises(ValueError, match="energy is not bound"):
        replace(
            evaluation,
            energy=replace(evaluation.energy, action_fingerprint="0" * 64),
        )


def test_companion_evaluation_rejects_coherent_alternate_action_substitution(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, arrays),
        interaction,
    )
    evaluation = prepared.evaluate(arrays["P"], arrays["P_ref"])

    alternate_H_D = np.array(evaluation.action.H_D_ev, copy=True)
    alternate_H_D[..., 0, 0] += 1.0e-6
    alternate_H_E = np.array(evaluation.action.H_E_ev, copy=True)
    alternate_H_interaction = np.asarray(alternate_H_D + alternate_H_E)
    alternate_action_residuals = type(evaluation.action.residuals)(
        density_hermiticity_max_abs=float(
            np.max(
                np.abs(
                    evaluation.density_delta
                    - evaluation.density_delta.conj().swapaxes(-1, -2)
                )
            )
        ),
        H_D_hermiticity_max_abs_ev=float(
            np.max(np.abs(alternate_H_D - alternate_H_D.conj().swapaxes(-1, -2)))
        ),
        H_E_hermiticity_max_abs_ev=float(
            np.max(np.abs(alternate_H_E - alternate_H_E.conj().swapaxes(-1, -2)))
        ),
        H_interaction_hermiticity_max_abs_ev=float(
            np.max(
                np.abs(
                    alternate_H_interaction
                    - alternate_H_interaction.conj().swapaxes(-1, -2)
                )
            )
        ),
    )
    alternate_action_hashes = type(evaluation.action.array_hashes).from_arrays(
        density_delta=evaluation.density_delta,
        H_D_ev=alternate_H_D,
        H_E_ev=alternate_H_E,
        H_interaction_ev=alternate_H_interaction,
    )
    alternate_action = replace(
        evaluation.action,
        H_D_ev=alternate_H_D,
        H_E_ev=alternate_H_E,
        H_interaction_ev=alternate_H_interaction,
        residuals=alternate_action_residuals,
        array_hashes=alternate_action_hashes,
    )
    alternate_energy = companion_calc_E(
        prepared.params,
        evaluation.projector,
        evaluation.reference,
        prepared.sp_energy_ev,
        alternate_action,
    )
    alternate_H_total = np.asarray(
        prepared.H_SP_ev + alternate_action.H_interaction_ev
    )
    alternate_evaluation_residuals = replace(
        evaluation.residuals,
        H_total_hermiticity_max_abs_ev=float(
            np.max(
                np.abs(
                    alternate_H_total
                    - alternate_H_total.conj().swapaxes(-1, -2)
                )
            )
        ),
        H_total_closure_max_abs_ev=float(
            np.max(
                np.abs(
                    alternate_H_total
                    - (prepared.H_SP_ev + alternate_action.H_interaction_ev)
                )
            )
        ),
        total_energy_imag_residual_ev=alternate_energy.total_imag_residual_ev,
        energy_action_binding_residual=float(
            alternate_energy.action_fingerprint != alternate_action.fingerprint
        ),
    )
    alternate_evaluation_hashes = (
        TBGZeroFieldCompanionHFEvaluationArrayHashes.from_arrays(
            projector=evaluation.projector,
            reference=evaluation.reference,
            density_delta=evaluation.density_delta,
            H_SP_ev=prepared.H_SP_ev,
            H_total_ev=alternate_H_total,
            energy_components_ev=alternate_energy.components_ev,
        )
    )

    assert alternate_energy.action_fingerprint == alternate_action.fingerprint
    np.testing.assert_array_equal(
        alternate_H_total,
        prepared.H_SP_ev + alternate_action.H_interaction_ev,
    )
    with pytest.raises(
        ValueError,
        match=r"action\.H_D_ev does not exactly equal the canonical calc_fock_matrix output",
    ):
        replace(
            evaluation,
            action=alternate_action,
            energy=alternate_energy,
            H_total_ev=alternate_H_total,
            residuals=alternate_evaluation_residuals,
            array_hashes=alternate_evaluation_hashes,
        )


def test_companion_evaluation_recomputes_energy_and_rejects_stale_prepared(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, arrays),
        interaction,
    )
    evaluation = prepared.evaluate(arrays["P"], arrays["P_ref"])

    coherent_components = np.array(evaluation.energy.components_ev, copy=True)
    coherent_components[1] += 1.0e-6
    coherent_components[0] = float(
        coherent_components[1]
        + coherent_components[2]
        + coherent_components[3]
    )
    coherent_energy = replace(
        evaluation.energy,
        components_ev=coherent_components,
    )
    coherent_hashes = TBGZeroFieldCompanionHFEvaluationArrayHashes.from_arrays(
        projector=evaluation.projector,
        reference=evaluation.reference,
        density_delta=evaluation.density_delta,
        H_SP_ev=evaluation.H_SP_ev,
        H_total_ev=evaluation.H_total_ev,
        energy_components_ev=coherent_components,
    )
    with pytest.raises(ValueError, match="energy components do not match recomputed calc_E"):
        replace(
            evaluation,
            energy=coherent_energy,
            array_hashes=coherent_hashes,
        )
    with pytest.raises(ValueError, match="max_component_imag_residual_ev"):
        replace(
            evaluation,
            energy=replace(
                evaluation.energy,
                max_component_imag_residual_ev=(
                    evaluation.energy.max_component_imag_residual_ev + 1.0e-15
                ),
            ),
        )

    stale_sp_energy = np.asarray(prepared.sp_energy_ev + 1.0e-4)
    stale_H_SP = companion_gen_H_SP(prepared.params, stale_sp_energy)
    stale_single_particle_hashes = (
        TBGZeroFieldCompanionSingleParticleArrayHashes.from_arrays(
            coeff=prepared.single_particle_source.coeff,
            sp_energy_ev=stale_sp_energy,
            U_C2T=prepared.single_particle_source.U_C2T,
        )
    )
    stale_single_particle_source = replace(
        prepared.single_particle_source,
        sp_energy_ev=stale_sp_energy,
        array_hashes=stale_single_particle_hashes,
    )
    stale_prepared = replace(
        prepared,
        single_particle_source=stale_single_particle_source,
        single_particle_fingerprint=stale_single_particle_source.fingerprint,
        sp_energy_ev=stale_sp_energy,
        H_SP_ev=stale_H_SP,
        array_hashes=_companion_prepared_array_hashes(
            prepared,
            sp_energy_ev=stale_sp_energy,
            H_SP_ev=stale_H_SP,
        ),
    )
    with pytest.raises(TypeError, match="prepared must be"):
        replace(evaluation, prepared=object())
    with pytest.raises(ValueError, match="prepared_fingerprint does not match"):
        replace(evaluation, prepared=stale_prepared)
    with pytest.raises(ValueError, match="H_SP_ev must exactly equal prepared.H_SP_ev"):
        replace(
            evaluation,
            prepared=stale_prepared,
            prepared_fingerprint=stale_prepared.fingerprint,
        )

    stale_H_total = np.asarray(stale_H_SP + evaluation.action.H_interaction_ev)
    stale_hashes = TBGZeroFieldCompanionHFEvaluationArrayHashes.from_arrays(
        projector=evaluation.projector,
        reference=evaluation.reference,
        density_delta=evaluation.density_delta,
        H_SP_ev=stale_H_SP,
        H_total_ev=stale_H_total,
        energy_components_ev=evaluation.energy.components_ev,
    )
    with pytest.raises(ValueError, match="energy components do not match recomputed calc_E"):
        replace(
            evaluation,
            prepared=stale_prepared,
            prepared_fingerprint=stale_prepared.fingerprint,
            H_SP_ev=stale_H_SP,
            H_total_ev=stale_H_total,
            array_hashes=stale_hashes,
        )


def test_companion_zero_exchange_is_exact_and_preserves_hartree(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> None:
    _manifest, arrays, _generator_path, _fixture_path = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    source_single_particle = _source_gauge_single_particle(single_particle, arrays)
    with_exchange = prepare_tbg_zero_field_companion_hf_action(
        source_single_particle,
        interaction,
    )
    without_exchange = prepare_tbg_zero_field_companion_hf_action(
        source_single_particle,
        interaction,
        spec=TBGZeroFieldCompanionHFActionSpec(exchange=False),
    )
    np.testing.assert_array_equal(
        without_exchange.tVE_ev,
        np.zeros(without_exchange.tVE_ev.shape, dtype=np.complex128),
    )
    assert without_exchange.tVE_ev.dtype == np.dtype(np.complex128)
    evaluation = without_exchange.evaluate(arrays["P"], arrays["P_ref"])
    full_evaluation = with_exchange.evaluate(arrays["P"], arrays["P_ref"])
    np.testing.assert_array_equal(evaluation.H_E_ev, np.zeros_like(evaluation.H_E_ev))
    np.testing.assert_array_equal(evaluation.H_D_ev, full_evaluation.H_D_ev)
    assert evaluation.energy.exchange_ev == 0.0


def test_companion_hf_action_guards_shape_nonfinite_boost_hermiticity_and_energy() -> None:
    spec = TBGZeroFieldCompanionHFActionSpec()
    with pytest.raises(ValueError, match="finite and positive"):
        replace(spec, epsr=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        replace(spec, epsr=np.nan)
    with pytest.raises(TypeError, match="exchange must be bool"):
        replace(spec, exchange=np.bool_(True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boost1=boost2=0"):
        replace(spec, boost1=1)

    params = _companion_stage4_params()
    form = np.zeros((2, 3, 2, 3, 10, 10, 2, 2, 2), dtype=np.complex128)
    intFT = np.zeros((2, 3, 10, 10), dtype=float)
    M, tVE = companion_gen_M_tVE(params, form, intFT)
    good_delta = np.zeros((2, 3, 2, 4, 4), dtype=np.complex128)
    with pytest.raises(ValueError, match="must have shape"):
        companion_calc_fock_matrix(params, good_delta[..., :3, :3], form, M, tVE)
    bad_form = form.copy()
    bad_form[0, 0, 0, 0, 0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        companion_gen_M_tVE(params, bad_form, intFT)
    nonhermitian = good_delta.copy()
    nonhermitian[0, 0, 0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="materially non-Hermitian"):
        companion_calc_fock_matrix(params, nonhermitian, form, M, tVE)
    with pytest.raises(ValueError, match="total must equal kinetic"):
        TBGZeroFieldCompanionHFEnergy(
            components_ev=np.asarray([1.0e-6, 0.0, 0.0, 0.0]),
            total_imag_residual_ev=0.0,
            max_component_imag_residual_ev=0.0,
            projector_sha256="0" * 64,
            reference_sha256="0" * 64,
            action_fingerprint="0" * 64,
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        TBGZeroFieldCompanionHFEnergy(
            components_ev=np.zeros(4),
            total_imag_residual_ev=np.inf,
            max_component_imag_residual_ev=np.inf,
            projector_sha256="0" * 64,
            reference_sha256="0" * 64,
            action_fingerprint="0" * 64,
        )
    with pytest.raises(ValueError, match="material imaginary defect"):
        TBGZeroFieldCompanionHFEnergy(
            components_ev=np.zeros(4),
            total_imag_residual_ev=2.0e-9,
            max_component_imag_residual_ev=2.0e-9,
            projector_sha256="0" * 64,
            reference_sha256="0" * 64,
            action_fingerprint="0" * 64,
        )


# ---------------------------------------------------------------------------
# Stage-6A source-faithful companion SCF diagnostic
# ---------------------------------------------------------------------------

_COMPANION_HF_SCF_FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parent / "fixtures" / "tbg_companion_hf_scf_v1"
)


_COMPANION_HF_SCF_FIXTURE_MANIFEST_KEYS = {
    "array_hash_convention",
    "array_hash_semantics",
    "arrays",
    "environment",
    "fixture_npz",
    "fixture_npz_sha256",
    "fixture_schema",
    "fixture_schema_version",
    "generator_script",
    "generator_script_sha256",
    "pinned_source",
    "scf_input",
    "scf_input_sha256",
    "scope",
    "stage4_fixture",
    "stored_projector_orientation",
    "units",
}
_COMPANION_HF_SCF_ITERATION_ARRAY_SUFFIXES = {
    "P_old",
    "H_D_old_ev",
    "H_E_old_ev",
    "H_old_ev",
    "P_raw",
    "eigenvalues_ev",
    "fill_indices",
    "H_D_dP_ev",
    "H_E_dP_ev",
    "P_mixed",
    "H_D_mixed_ev",
    "H_E_mixed_ev",
    "difference",
    "coefficients",
    "branch",
    "positive_linear",
    "energy_ev",
}
_COMPANION_HF_SCF_FIXTURE_ARRAY_KEYS = {
    "initial_P",
    "P_ref",
    "history_differences",
    "history_coefficients",
    "history_energies_ev",
    "history_branches",
    "history_positive_linear",
    "final_P_mixed",
    "final_H_D_ev",
    "final_H_E_ev",
    "final_H_ev",
    "final_P_raw",
    "final_eigenvalues_ev",
    "final_fill_indices",
    "final_energy_ev",
    "final_closure_difference",
    "final_source_energy_ev",
    "final_source_gap_ev",
    "final_source_difference",
    "final_source_local_occupations",
    "final_source_flavor_occupations",
    "final_source_valley_polarization",
    "final_source_ivc",
    "final_source_spin_polarization",
    "final_source_tp_break",
    *(
        f"iter_{iteration:03d}_{suffix}"
        for iteration in range(4)
        for suffix in _COMPANION_HF_SCF_ITERATION_ARRAY_SUFFIXES
    ),
}
_COMPANION_HF_SCF_SOURCE_LINE_RANGES = {
    "routines.calc_E": TBG_ZERO_FIELD_COMPANION_CALC_E_REFERENCE_LINES,
    "routines.calc_fock_matrix": (
        TBG_ZERO_FIELD_COMPANION_CALC_FOCK_MATRIX_REFERENCE_LINES
    ),
    "routines.aufbau": TBG_ZERO_FIELD_COMPANION_AUFBAU_REFERENCE_LINES,
    "mainProgram.SCF_and_final_rebuild": (
        TBG_ZERO_FIELD_COMPANION_MAIN_SCF_REFERENCE_LINES
    ),
    "mainProgram.ODA": TBG_ZERO_FIELD_COMPANION_MAIN_ODA_REFERENCE_LINES,
    "projectors.average_central": (
        TBG_ZERO_FIELD_COMPANION_AVERAGE_CENTRAL_REFERENCE_LINES
    ),
    "measure.boost0_Tp_and_observables": (
        TBG_ZERO_FIELD_COMPANION_MEASURE_REFERENCE_LINES
    ),
}
_COMPANION_HF_SCF_STAGE4_RESOLVED_INPUT_SHA256 = (
    "c7922c6e78d8bf23eb633877b8655d9aa71634b3ba24fe8595ddf6ff17496881"
)

# Every fixture array must be consumed by the parity test or documented here
# with a scientific reason why no portable numerical comparison exists. Raw
# eigensolver vectors are gauge-nonportable and are therefore not stored at all.
_COMPANION_HF_SCF_NONCOMPARABLE_ARRAYS: dict[str, str] = {}

@pytest.fixture(scope="module")
def companion_hf_scf_source_fixture():
    generator_path = _COMPANION_HF_SCF_FIXTURE_DIRECTORY / "generate_fixture.py"
    manifest_path = _COMPANION_HF_SCF_FIXTURE_DIRECTORY / "manifest.json"
    assert generator_path.is_file(), "Pinned companion Stage6A generator is absent"
    assert manifest_path.is_file(), (
        "Pinned companion Stage6A manifest is absent; explicitly run "
        "tests/fixtures/tbg_companion_hf_scf_v1/generate_fixture.py first"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture_path = _COMPANION_HF_SCF_FIXTURE_DIRECTORY / manifest["fixture_npz"]
    assert fixture_path.is_file(), (
        "Pinned companion Stage6A NPZ is absent; explicitly run the source-only "
        "fixture generator first"
    )
    assert manifest["fixture_npz_sha256"] == _companion_fixture_file_sha256(
        fixture_path
    )
    assert manifest["generator_script_sha256"] == _companion_fixture_file_sha256(
        generator_path
    )
    with np.load(fixture_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    assert set(arrays) == set(manifest["arrays"])
    for key, array in arrays.items():
        record = manifest["arrays"][key]
        assert record["shape"] == list(array.shape)
        assert record["dtype"] == array.dtype.str
        assert record["sha256"] == _companion_fixture_array_sha256(array)
    return manifest, arrays


def test_companion_hf_scf_source_fixture_manifest_is_source_only_and_pinned(
    companion_hf_scf_source_fixture,
) -> None:
    manifest, arrays = companion_hf_scf_source_fixture
    assert set(manifest) == _COMPANION_HF_SCF_FIXTURE_MANIFEST_KEYS
    assert set(manifest["arrays"]) == _COMPANION_HF_SCF_FIXTURE_ARRAY_KEYS
    assert set(arrays) == _COMPANION_HF_SCF_FIXTURE_ARRAY_KEYS
    for record in manifest["arrays"].values():
        assert set(record) == {"dtype", "sha256", "shape"}
    assert set(manifest["environment"]) == {
        "byteorder",
        "numpy",
        "platform",
        "python",
        "zlib_compile",
        "zlib_runtime",
    }
    assert all(
        isinstance(value, str) and value
        for value in manifest["environment"].values()
    )
    assert manifest["array_hash_convention"] == (
        "sha256_little_endian_int64_shape_then_C_order_little_endian_array_bytes"
    )
    assert manifest["array_hash_semantics"] == (
        "source_fixture_integrity_and_same_environment_parity_not_production_result"
    )
    assert manifest["fixture_npz"] == "fixture.npz"
    assert manifest["generator_script"] == "generate_fixture.py"
    assert manifest["stored_projector_orientation"] == (
        "P[alpha,beta]=<c_dagger_alpha c_beta>"
    )
    assert set(manifest["units"]) == {
        "Hamiltonian_action_eigenvalues_energy",
        "energy",
    }
    assert manifest["units"] == {
        "Hamiltonian_action_eigenvalues_energy": "eV",
        "energy": "finite_system_eV_not_per_moire_cell",
    }

    pinned = manifest["pinned_source"]
    assert set(pinned) == {
        "main_program",
        "measure",
        "projectors",
        "reference_commit",
        "reference_repository",
        "routines",
        "source_line_ranges",
    }
    assert pinned["reference_repository"] == TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY
    assert pinned["reference_commit"] == TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT
    assert pinned["routines"] == {
        "path": "reference/TBG-HF/routines.py",
        "sha256": TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE_SHA256,
    }
    assert pinned["main_program"] == {
        "path": "reference/TBG-HF/mainProgram.py",
        "sha256": TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE_SHA256,
    }
    assert pinned["projectors"] == {
        "path": TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE,
        "sha256": TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256,
    }
    assert pinned["measure"] == {
        "path": TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE,
        "sha256": TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256,
    }
    for source_record in (
        pinned["routines"],
        pinned["main_program"],
        pinned["projectors"],
        pinned["measure"],
    ):
        assert set(source_record) == {"path", "sha256"}
        assert _is_sha256(source_record["sha256"])
    assert set(pinned["source_line_ranges"]) == set(
        _COMPANION_HF_SCF_SOURCE_LINE_RANGES
    )
    assert pinned["source_line_ranges"] == _COMPANION_HF_SCF_SOURCE_LINE_RANGES

    assert set(manifest["scf_input"]) == {
        "filling",
        "HF_itermax",
        "HF_itermin",
        "HF_tolerance",
        "HF_type",
        "ODA_branch_threshold",
    }
    assert manifest["scf_input_sha256"] == _companion_fixture_json_sha256(
        manifest["scf_input"]
    )

    stage4_fixture = manifest["stage4_fixture"]
    assert set(stage4_fixture) == {
        "array_sha256",
        "directory",
        "fixture_npz_sha256",
        "generator_sha256",
        "resolved_input_sha256",
    }
    assert stage4_fixture["directory"] == (
        "tests/fixtures/tbg_companion_hf_action_v1"
    )
    stage4_manifest_path = _COMPANION_HF_ACTION_FIXTURE_DIRECTORY / "manifest.json"
    stage4_manifest = json.loads(stage4_manifest_path.read_text(encoding="utf-8"))
    expected_stage4_input = {
        **_COMPANION_HF_ACTION_INPUT_OVERRIDES,
        **_COMPANION_HF_ACTION_INHERITED_INTERACTION,
        **_COMPANION_HF_ACTION_INHERITED_HF,
    }
    assert set(stage4_manifest["resolved_input"]) == set(expected_stage4_input)
    assert stage4_manifest["resolved_input"] == expected_stage4_input
    recomputed_stage4_digest = _companion_fixture_json_sha256(
        stage4_manifest["resolved_input"]
    )
    assert recomputed_stage4_digest == (
        _COMPANION_HF_SCF_STAGE4_RESOLVED_INPUT_SHA256
    )
    assert stage4_manifest["resolved_input_sha256"] == recomputed_stage4_digest
    assert stage4_fixture["resolved_input_sha256"] == recomputed_stage4_digest
    assert stage4_fixture["fixture_npz_sha256"] == (
        stage4_manifest["fixture_npz_sha256"]
    )
    assert stage4_fixture["generator_sha256"] == (
        stage4_manifest["generator_script_sha256"]
    )
    assert set(stage4_fixture["array_sha256"]) == (
        _COMPANION_HF_ACTION_FIXTURE_ARRAY_KEYS
    )
    assert stage4_fixture["array_sha256"] == {
        name: stage4_manifest["arrays"][name]["sha256"]
        for name in _COMPANION_HF_ACTION_FIXTURE_ARRAY_KEYS
    }

    assert manifest["fixture_schema"] == (
        "mean_field.tbg.companion_hf_scf.source_fixture"
    )
    assert manifest["fixture_schema_version"] == 1
    assert manifest["scope"] == (
        "four_iteration_source_oracle_not_production_HF_TDHF_or_Fig8"
    )
    assert manifest["scf_input"] == {
        "filling": 0,
        "HF_itermax": 4,
        "HF_itermin": 20,
        "HF_tolerance": 1.0e-8,
        "HF_type": "ODA",
        "ODA_branch_threshold": 1.0e-12,
    }
    pinned = manifest["pinned_source"]
    assert pinned["routines"]["sha256"] == TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE_SHA256
    assert pinned["main_program"]["sha256"] == (
        TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE_SHA256
    )
    assert pinned["projectors"]["sha256"] == TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256
    assert pinned["measure"]["sha256"] == TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256
    assert pinned["source_line_ranges"]["routines.aufbau"] == (
        TBG_ZERO_FIELD_COMPANION_AUFBAU_REFERENCE_LINES
    )
    assert pinned["source_line_ranges"]["mainProgram.SCF_and_final_rebuild"] == (
        TBG_ZERO_FIELD_COMPANION_MAIN_SCF_REFERENCE_LINES
    )
    assert pinned["source_line_ranges"]["mainProgram.ODA"] == (
        TBG_ZERO_FIELD_COMPANION_MAIN_ODA_REFERENCE_LINES
    )
    assert pinned["source_line_ranges"]["projectors.average_central"] == (
        TBG_ZERO_FIELD_COMPANION_AVERAGE_CENTRAL_REFERENCE_LINES
    )
    assert TBG_ZERO_FIELD_COMPANION_MEASURE_REFERENCE_LINES == "5-14,35-72"
    assert pinned["source_line_ranges"]["measure.boost0_Tp_and_observables"] == (
        TBG_ZERO_FIELD_COMPANION_MEASURE_REFERENCE_LINES
    )
    assert not any(name.endswith("_eigenvectors") for name in arrays)
    assert arrays["history_differences"].shape == (4,)
    assert arrays["history_coefficients"].shape == (4, 6)
    assert arrays["history_energies_ev"].shape == (4, 4)
    assert arrays["history_positive_linear"].dtype == np.dtype(np.bool_)
    assert arrays["final_source_local_occupations"].shape == (2, 3, 2)
    assert arrays["final_source_flavor_occupations"].shape == (2, 2)
    for name in (
        "final_source_energy_ev",
        "final_source_gap_ev",
        "final_source_difference",
        "final_source_valley_polarization",
        "final_source_ivc",
        "final_source_spin_polarization",
        "final_source_tp_break",
    ):
        assert arrays[name].shape == ()


def test_companion_hf_scf_system_local_source_parity_exception_is_explicit() -> None:
    exception = TBG_ZERO_FIELD_COMPANION_HF_SCF_SOURCE_PARITY_EXCEPTION
    assert "generic_core_engine_is_non_equivalent" in exception
    assert "ODA_branch_convergence_norm_index_and_finalization" in exception
    assert "not_reusable_framework_fork" in exception
    assert "not_frontdoor_production_TDHF_or_Fig8" in exception

def test_companion_hf_scf_four_iteration_source_fixture_parity(
    companion_hf_scf_source_fixture,
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> None:
    _manifest, raw_oracle = companion_hf_scf_source_fixture
    consumed_fixture_arrays: set[str] = set()

    class _TrackedFixtureArrays(dict[str, np.ndarray]):
        def __getitem__(self, key: str) -> np.ndarray:
            consumed_fixture_arrays.add(key)
            return super().__getitem__(key)

    oracle = _TrackedFixtureArrays(raw_oracle)
    _stage4_manifest, stage4, _generator, _fixture = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, stage4),
        interaction,
    )
    spec = TBGZeroFieldCompanionHFSCFSpec(HF_itermax=4, HF_itermin=20)
    run = run_tbg_zero_field_companion_hf_diagnostic(
        prepared,
        stage4["P"],
        stage4["P_ref"],
        spec,
    )

    np.testing.assert_array_equal(run.initial_projector, oracle["initial_P"])
    np.testing.assert_array_equal(run.reference, oracle["P_ref"])
    np.testing.assert_allclose(
        run.history.differences,
        oracle["history_differences"],
        rtol=2.0e-12,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        np.column_stack(
            (
                run.history.c1,
                run.history.c01,
                run.history.c11,
                run.history.lin,
                run.history.quad,
                run.history.mixing_lambda,
            )
        ),
        oracle["history_coefficients"],
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        run.history.energies_ev,
        oracle["history_energies_ev"],
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    assert run.history.branches == tuple(oracle["history_branches"].tolist())
    np.testing.assert_array_equal(
        run.history.positive_linear,
        oracle["history_positive_linear"],
    )

    old = np.array(stage4["P"], copy=True)
    reference = np.asarray(stage4["P_ref"])
    Nk = prepared.params.N1 * prepared.params.N2
    for iteration in range(4):
        prefix = f"iter_{iteration:03d}"
        old_evaluation = prepared.evaluate(old, reference)
        aufbau = companion_aufbau(
            prepared,
            old_evaluation.H_total_ev,
            filling=spec.filling,
        )
        difference = float(np.linalg.norm(old - aufbau.projector) / Nk)
        oda = companion_oda_coefficients(
            prepared,
            old,
            aufbau.projector,
            reference,
        )
        mixed = np.asarray(
            (1.0 - oda.mixing_lambda) * old
            + oda.mixing_lambda * aufbau.projector
        )
        mixed_evaluation = prepared.evaluate(mixed, reference)

        np.testing.assert_allclose(
            old,
            oracle[f"{prefix}_P_old"],
            rtol=2.0e-11,
            atol=2.0e-12,
        )
        for actual, suffix in (
            (old_evaluation.action.H_D_ev, "H_D_old_ev"),
            (old_evaluation.action.H_E_ev, "H_E_old_ev"),
            (old_evaluation.H_total_ev, "H_old_ev"),
            (aufbau.projector, "P_raw"),
            (aufbau.eigenvalues_ev, "eigenvalues_ev"),
            (oda.dP_action.H_D_ev, "H_D_dP_ev"),
            (oda.dP_action.H_E_ev, "H_E_dP_ev"),
            (mixed, "P_mixed"),
            (mixed_evaluation.action.H_D_ev, "H_D_mixed_ev"),
            (mixed_evaluation.action.H_E_ev, "H_E_mixed_ev"),
            (mixed_evaluation.energy.components_ev, "energy_ev"),
        ):
            np.testing.assert_allclose(
                actual,
                oracle[f"{prefix}_{suffix}"],
                rtol=2.0e-11,
                atol=2.0e-12,
            )
        np.testing.assert_array_equal(
            aufbau.fill_indices,
            oracle[f"{prefix}_fill_indices"],
        )
        np.testing.assert_allclose(
            difference,
            oracle[f"{prefix}_difference"],
            rtol=2.0e-12,
            atol=2.0e-13,
        )
        actual_coefficients = np.asarray(
            [oda.c1, oda.c01, oda.c11, oda.lin, oda.quad, oda.mixing_lambda]
        )
        np.testing.assert_allclose(
            actual_coefficients,
            oracle[f"{prefix}_coefficients"],
            rtol=2.0e-11,
            atol=2.0e-12,
        )
        assert oda.branch == oracle[f"{prefix}_branch"].item()
        assert oda.positive_linear == bool(
            oracle[f"{prefix}_positive_linear"].item()
        )

        assert run.history.old_projector_hashes[iteration] == (
            _companion_fixture_array_sha256(old)
        )
        assert run.history.raw_projector_hashes[iteration] == (
            _companion_fixture_array_sha256(aufbau.projector)
        )
        assert run.history.mixed_projector_hashes[iteration] == (
            _companion_fixture_array_sha256(mixed)
        )
        assert run.history.hamiltonian_hashes[iteration] == (
            _companion_fixture_array_sha256(old_evaluation.H_total_ev)
        )
        assert run.history.old_action_fingerprints[iteration] == (
            old_evaluation.action.fingerprint
        )
        assert run.history.dP_action_fingerprints[iteration] == (
            oda.dP_action.fingerprint
        )
        assert run.history.energy_fingerprints[iteration] == (
            mixed_evaluation.energy.fingerprint
        )
        old = np.array(mixed, copy=True)

    for actual, key in (
        (run.final_projector_mixed, "final_P_mixed"),
        (run.final_evaluation.action.H_D_ev, "final_H_D_ev"),
        (run.final_evaluation.action.H_E_ev, "final_H_E_ev"),
        (run.final_evaluation.H_total_ev, "final_H_ev"),
        (run.final_aufbau.projector, "final_P_raw"),
        (run.final_aufbau.eigenvalues_ev, "final_eigenvalues_ev"),
        (run.final_evaluation.energy.components_ev, "final_energy_ev"),
    ):
        np.testing.assert_allclose(
            actual,
            oracle[key],
            rtol=2.0e-11,
            atol=2.0e-12,
        )
    np.testing.assert_array_equal(
        run.final_aufbau.fill_indices,
        oracle["final_fill_indices"],
    )
    np.testing.assert_allclose(
        run.closure_difference,
        oracle["final_closure_difference"],
        rtol=2.0e-12,
        atol=2.0e-13,
    )

    report = qualify_tbg_zero_field_companion_hf_diagnostic(run)
    np.testing.assert_allclose(
        run.final_evaluation.energy.components_ev[0],
        oracle["final_source_energy_ev"],
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        report.gap_ev,
        oracle["final_source_gap_ev"],
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        run.closure_difference,
        oracle["final_source_difference"],
        rtol=2.0e-12,
        atol=2.0e-13,
    )
    np.testing.assert_array_equal(
        report.local_occupations,
        oracle["final_source_local_occupations"],
    )
    for actual, key in (
        (report.flavor_occupations, "final_source_flavor_occupations"),
        (report.valley_polarization, "final_source_valley_polarization"),
        (report.source_ivc, "final_source_ivc"),
        (report.spin_polarization, "final_source_spin_polarization"),
        (report.tp_break, "final_source_tp_break"),
    ):
        np.testing.assert_allclose(actual, oracle[key], rtol=2.0e-11, atol=2.0e-12)
    np.testing.assert_allclose(
        report.spin_block_rms_difference,
        np.linalg.norm(
            oracle["final_P_mixed"][:, :, 0]
            - oracle["final_P_mixed"][:, :, 1]
        )
        / np.sqrt(Nk),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    projector_hermiticity = max(
        float(np.max(np.abs(array - np.swapaxes(array.conj(), -1, -2))))
        for array in (
            oracle["final_P_mixed"],
            oracle["P_ref"],
            oracle["final_P_raw"],
        )
    )
    hamiltonian_hermiticity_ev = max(
        float(np.max(np.abs(array - np.swapaxes(array.conj(), -1, -2))))
        for array in (
            oracle["final_H_ev"],
            oracle["final_H_D_ev"],
            oracle["final_H_E_ev"],
        )
    )
    np.testing.assert_allclose(
        report.maximum_projector_hermiticity_residual,
        projector_hermiticity,
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        report.maximum_hamiltonian_hermiticity_residual_ev,
        hamiltonian_hermiticity_ev,
        rtol=2.0e-11,
        atol=2.0e-12,
    )

    noncomparable = set(_COMPANION_HF_SCF_NONCOMPARABLE_ARRAYS)
    assert consumed_fixture_arrays.isdisjoint(noncomparable)
    assert set(oracle) == consumed_fixture_arrays | noncomparable, {
        "unconsumed": sorted(set(oracle) - consumed_fixture_arrays - noncomparable),
        "unknown_allowlist": sorted(noncomparable - set(oracle)),
    }


def test_companion_average_central_reference_and_aufbau_stored_orientation(
    companion_stage4_typed_inputs,
) -> None:
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(single_particle, interaction)
    reference = build_companion_average_central_reference(prepared)
    expected = 0.5 * np.eye(4, dtype=np.complex128)
    np.testing.assert_array_equal(reference, np.broadcast_to(expected, reference.shape))
    assert not reference.flags.writeable

    shape = prepared.H_SP_ev.shape
    H = np.zeros(shape, dtype=np.complex128)
    for sector, matrix in enumerate(H.reshape((-1, 4, 4))):
        matrix[:] = np.diag(1000.0 + 10.0 * sector + np.arange(4))
    unitary = np.asarray(
        [[1.0, 1.0j], [1.0j, 1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    target_spectrum = np.asarray([0.0, 200.0])
    H[0, 0, 0, :2, :2] = unitary @ np.diag(target_spectrum) @ unitary.conj().T
    for sector in range(1, 6):
        H.reshape((-1, 4, 4))[sector, 0, 0] = float(sector)
    result = companion_aufbau(prepared, H, filling=-3)
    vector = result.eigenvectors[0, :, 0]
    assert abs(result.projector[0, 0, 0, 0, 1]) > 0.0
    np.testing.assert_allclose(
        result.projector[0, 0, 0, 0, 1],
        np.conj(vector[0]) * vector[1],
        rtol=0.0,
        atol=1.0e-15,
    )


def test_companion_aufbau_uses_global_c_order_fill(
    companion_stage4_typed_inputs,
) -> None:
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(single_particle, interaction)
    H = np.zeros_like(prepared.H_SP_ev, dtype=np.complex128)
    flat_matrices = H.reshape((-1, 4, 4), order="C")
    for sector, matrix in enumerate(flat_matrices):
        matrix[:] = np.diag(4 * sector + np.arange(4, dtype=float))
    result = companion_aufbau(prepared, H, filling=0)
    expected = np.arange(result.electron_count, dtype=np.int64)
    np.testing.assert_array_equal(result.fill_indices, expected)
    assert result.array_hashes.fill_indices == _companion_fixture_array_sha256(expected)


@pytest.mark.parametrize(
    ("lin", "quad", "expected_lambda", "expected_branch", "positive"),
    (
        (2.0e-3, -100.0, 1.0, "positive_linear", True),
        (-2.0, 1.0, 1.0, "endpoint_quad", False),
        (5.0e-13, 1.0, 1.0, "linear_near_zero", False),
        (-2.0, 2.0, 0.5, "interior", False),
    ),
)
def test_companion_oda_every_source_branch_is_ordered_exactly(
    lin,
    quad,
    expected_lambda,
    expected_branch,
    positive,
) -> None:
    mixing_lambda, branch, positive_linear = companion_oda_branch(lin, quad)
    assert mixing_lambda == expected_lambda
    assert branch == expected_branch
    assert positive_linear is positive


def test_companion_scf_convergence_is_zero_based_strict_and_pre_mixing(
    companion_stage4_typed_inputs,
) -> None:
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(single_particle, interaction)
    reference = build_companion_average_central_reference(prepared)
    run = run_tbg_zero_field_companion_hf_diagnostic(
        prepared,
        reference,
        reference,
        TBGZeroFieldCompanionHFSCFSpec(
            HF_itermax=3,
            HF_itermin=0,
            tolerance=1.0e9,
        ),
    )
    assert run.converged
    assert run.convergence_iteration == 1
    np.testing.assert_array_equal(run.history.iterations, np.asarray([0, 1]))
    assert np.all(run.history.differences < run.spec.tolerance)
    assert TBG_ZERO_FIELD_COMPANION_HF_SCF_CONVERGENCE_CONVENTION.startswith(
        "zero_based_iteration"
    )


def test_companion_scf_accepts_direct_stage5_seed_and_reports_only(
    companion_stage4_typed_inputs,
) -> None:
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(single_particle, interaction)
    seed = build_tbg_zero_field_companion_kivc_seed(single_particle)
    reference = build_companion_average_central_reference(prepared)
    run = run_tbg_zero_field_companion_hf_diagnostic(
        prepared,
        seed,
        reference,
        TBGZeroFieldCompanionHFSCFSpec(HF_itermax=1, HF_itermin=20),
    )
    assert run.initial_source == "stage5_kivc_seed"
    assert run.stage5_seed is seed
    assert run.stage5_seed_fingerprint == seed.fingerprint
    np.testing.assert_array_equal(run.initial_projector, seed.P_stored)
    report = qualify_tbg_zero_field_companion_hf_diagnostic(
        run,
        TBGZeroFieldCompanionHFQualifierSpec(),
    )
    assert report.scientific_scope == TBG_ZERO_FIELD_COMPANION_HF_SCF_SCOPE
    assert "converged" in dict(report.checks)
    assert report.stage5_eq99_projection_magnitude is not None
    assert "production" in report.scientific_scope
    assert "not_production" in report.scientific_scope


def test_companion_scf_run_fails_closed_on_prepared_and_source_mutation(
    companion_stage4_typed_inputs,
) -> None:
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(single_particle, interaction)
    reference = build_companion_average_central_reference(prepared)
    run = run_tbg_zero_field_companion_hf_diagnostic(
        prepared,
        reference,
        reference,
        TBGZeroFieldCompanionHFSCFSpec(HF_itermax=1, HF_itermin=20),
    )
    _assert_stage4_readonly_bytes_mutation_fails_closed(
        prepared.H_SP_ev,
        lambda: run.fingerprint,
        match="prepared array_hashes no longer match live prepared arrays",
    )
    _assert_stage4_readonly_bytes_mutation_fails_closed(
        prepared.single_particle_source.coeff,
        lambda: run.fingerprint,
        match=r"single_particle_source\.coeff no longer matches its source hash",
    )
    _assert_stage4_readonly_bytes_mutation_fails_closed(
        run.history.differences,
        lambda: run.fingerprint,
        match="history array_hashes no longer match live arrays",
    )
    _assert_stage4_readonly_bytes_mutation_fails_closed(
        run.final_projector_mixed,
        lambda: run.fingerprint,
        match="run array_hashes no longer match live arrays",
    )

def test_companion_scf_run_replay_rejects_coherent_history_forgery(
    companion_stage4_typed_inputs,
) -> None:
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(single_particle, interaction)
    reference = build_companion_average_central_reference(prepared)
    run = run_tbg_zero_field_companion_hf_diagnostic(
        prepared,
        reference,
        reference,
        TBGZeroFieldCompanionHFSCFSpec(HF_itermax=1, HF_itermin=20),
    )

    forged_differences = np.array(run.history.differences, copy=True)
    forged_differences[0] += 1.0e-6
    history_arrays = {
        "iterations": run.history.iterations,
        "differences": forged_differences,
        "c1": run.history.c1,
        "c01": run.history.c01,
        "c11": run.history.c11,
        "lin": run.history.lin,
        "quad": run.history.quad,
        "mixing_lambda": run.history.mixing_lambda,
        "positive_linear": run.history.positive_linear,
        "energies_ev": run.history.energies_ev,
    }
    forged_history = replace(
        run.history,
        differences=forged_differences,
        array_hashes=type(run.history.array_hashes).from_arrays(history_arrays),
    )
    with pytest.raises(
        ValueError,
        match=r"history\.differences does not match deterministic trajectory replay",
    ):
        replace(run, history=forged_history)

    forged_receipts = replace(
        run.history,
        raw_projector_hashes=("0" * 64,),
    )
    with pytest.raises(
        ValueError,
        match=r"history\.raw_projector_hashes do not match deterministic trajectory replay",
    ):
        replace(run, history=forged_receipts)

def test_companion_qualification_report_recomputes_every_live_field(
    companion_stage4_typed_inputs,
) -> None:
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(single_particle, interaction)
    reference = build_companion_average_central_reference(prepared)
    run = run_tbg_zero_field_companion_hf_diagnostic(
        prepared,
        reference,
        reference,
        TBGZeroFieldCompanionHFSCFSpec(HF_itermax=1, HF_itermin=20),
    )
    qualifier_spec = TBGZeroFieldCompanionHFQualifierSpec()
    report = qualify_tbg_zero_field_companion_hf_diagnostic(run, qualifier_spec)
    assert isinstance(report, TBGZeroFieldCompanionHFQualificationReport)
    assert qualifier_spec.projector_hermiticity_threshold == 1.0e-9
    assert qualifier_spec.hamiltonian_hermiticity_threshold_ev == 1.0e-9
    assert qualifier_spec.spin_block_rms_tolerance == 1.0e-8
    np.testing.assert_allclose(
        report.spin_block_rms_difference,
        np.linalg.norm(
            run.final_projector_mixed[:, :, 0]
            - run.final_projector_mixed[:, :, 1]
        )
        / np.sqrt(prepared.params.N1 * prepared.params.N2),
        rtol=0.0,
        atol=0.0,
    )

    spec_metadata = qualifier_spec.to_metadata()
    assert spec_metadata == {
        name: getattr(qualifier_spec, name)
        for name in qualifier_spec.__dataclass_fields__
    } | {"fingerprint": qualifier_spec.fingerprint}
    metadata = report.to_metadata()
    assert metadata["spec"] == spec_metadata
    assert metadata["spec_fingerprint"] == qualifier_spec.fingerprint
    assert metadata["local_occupations_sha256"] == (
        report.local_occupations_sha256
    )
    assert metadata["flavor_occupations_sha256"] == (
        report.flavor_occupations_sha256
    )
    assert metadata["checks"] == dict(report.checks)
    assert metadata["metrics"] == {
        "expected_occupied_count": report.expected_occupied_count,
        "gap_ev": report.gap_ev,
        "fermi_tie": report.fermi_tie,
        "maximum_hamiltonian_hermiticity_residual_ev": (
            report.maximum_hamiltonian_hermiticity_residual_ev
        ),
        "maximum_projector_hermiticity_residual": (
            report.maximum_projector_hermiticity_residual
        ),
        "nu": report.nu,
        "nu_residual": report.nu_residual,
        "occupied_count": report.occupied_count,
        "source_ivc": report.source_ivc,
        "spin_block_rms_difference": report.spin_block_rms_difference,
        "spin_polarization": report.spin_polarization,
        "stage5_eq99_projection_magnitude": (
            report.stage5_eq99_projection_magnitude
        ),
        "tp_break": report.tp_break,
        "valley_polarization": report.valley_polarization,
    }

    for name in (
        "maximum_projector_hermiticity_residual",
        "maximum_hamiltonian_hermiticity_residual_ev",
        "gap_ev",
        "nu",
        "nu_residual",
        "source_ivc",
        "tp_break",
        "valley_polarization",
        "spin_polarization",
        "spin_block_rms_difference",
    ):
        with pytest.raises(ValueError, match=rf"qualification {name} does not match"):
            replace(report, **{name: getattr(report, name) + 1.0e-6})
    with pytest.raises(ValueError, match="qualification occupied_count does not match"):
        replace(report, occupied_count=report.occupied_count + 1)
    with pytest.raises(ValueError, match="qualification fermi_tie does not match"):
        replace(report, fermi_tie=not report.fermi_tie)
    with pytest.raises(
        ValueError,
        match="qualification stage5_eq99_projection_magnitude does not match",
    ):
        replace(report, stage5_eq99_projection_magnitude=0.5)

    with pytest.raises(ValueError, match="qualification passed flag does not match"):
        replace(report, passed=not report.passed)
    forged_checks = tuple((name, True) for name, _value in report.checks)
    assert forged_checks != report.checks
    with pytest.raises(ValueError, match="qualification checks do not match live run"):
        replace(report, passed=report.passed, checks=forged_checks)

    direct_fields = {
        name: getattr(report, name) for name in report.__dataclass_fields__
    }
    direct_fields["expected_occupied_count"] = report.expected_occupied_count + 1
    with pytest.raises(
        ValueError,
        match="qualification expected_occupied_count does not match live run",
    ):
        TBGZeroFieldCompanionHFQualificationReport(**direct_fields)

    forged_local = np.array(report.local_occupations, copy=True)
    forged_local[0, 0, 0] += 1
    coherent_occupation_fields = {
        name: getattr(report, name) for name in report.__dataclass_fields__
    }
    coherent_occupation_fields.update(
        local_occupations=forged_local,
        local_occupations_sha256=_companion_fixture_array_sha256(forged_local),
    )
    with pytest.raises(
        ValueError,
        match="qualification local_occupations do not match live run",
    ):
        TBGZeroFieldCompanionHFQualificationReport(**coherent_occupation_fields)


# ---------------------------------------------------------------------------
# Stage-7A diagnostic-only companion finite-q TDHF core
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def companion_stage7a_source(
    companion_hf_action_source_fixture,
    companion_stage4_typed_inputs,
) -> TBGZeroFieldCompanionTDHFSource:
    """Tiny source-gauge Stage4 plus typed Stage6 fixed point; not production."""

    _manifest, stage4, _generator, _fixture = companion_hf_action_source_fixture
    single_particle, interaction = companion_stage4_typed_inputs
    prepared = prepare_tbg_zero_field_companion_hf_action(
        _source_gauge_single_particle(single_particle, stage4),
        interaction,
    )
    stationary = companion_aufbau(prepared, prepared.H_SP_ev, filling=0)
    run = run_tbg_zero_field_companion_hf_diagnostic(
        prepared,
        stationary.projector,
        stationary.projector,
        TBGZeroFieldCompanionHFSCFSpec(
            filling=0,
            HF_itermax=3,
            HF_itermin=0,
            tolerance=1.0e-8,
        ),
    )
    assert run.converged
    assert run.closure_difference <= (
        TBG_ZERO_FIELD_COMPANION_TDHF_MAX_MIXED_AUFBAU_CLOSURE
    )
    return build_tbg_zero_field_companion_tdhf_source_from_stage6_run(run)


def _stage7a_in_memory_source(
    source: TBGZeroFieldCompanionTDHFSource,
    *,
    eigenvectors: np.ndarray | None = None,
    H_total_ev: np.ndarray | None = None,
) -> TBGZeroFieldCompanionTDHFSource:
    return build_tbg_zero_field_companion_tdhf_source_from_in_memory_arrays(
        source.prepared,
        final_projector_mixed=source.final_projector_mixed,
        reference=source.reference,
        H_total_ev=(source.H_total_ev if H_total_ev is None else H_total_ev),
        eigenvalues_ev=source.eigenvalues_ev,
        eigenvectors=(source.eigenvectors if eigenvectors is None else eigenvectors),
        fill_indices=source.fill_indices,
        occupations=source.occupations,
        final_projector_aufbau=source.final_projector_aufbau,
        filling=source.filling,
        electron_count=source.electron_count,
    )

def _stage7a_job_state_payload(
    source: TBGZeroFieldCompanionTDHFSource,
) -> dict[str, np.ndarray]:
    dimension = source.dimension
    return {
        "final_H_total_ev": source.H_total_ev,
        "final_fill_indices": source.fill_indices,
        "final_hf_eigenvalues_ev": np.reshape(
            source.eigenvalues_ev,
            (-1,),
            order="C",
        ),
        "final_hf_eigenvectors": np.reshape(
            source.eigenvectors,
            (-1, dimension, dimension),
            order="C",
        ),
        "final_projector_aufbau_stored": source.final_projector_aufbau,
        "final_projector_mixed_stored": source.final_projector_mixed,
        "history_iterations": np.asarray([0], dtype=np.int64),
        "reference_projector_stored": source.reference,
    }


def _stage7a_file_record(root: Path, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _stage7a_write_job_artifacts(
    root: Path,
    source: TBGZeroFieldCompanionTDHFSource,
    *,
    payload: dict[str, np.ndarray] | None = None,
    declared_array_keys: list[str] | None = None,
    summary_schema: str = TBG_ZERO_FIELD_COMPANION_TDHF_SUMMARY_SCHEMA,
    summary_status: str = "pass",
    summary_scope: str = (
        "N10_companion_HF_prerequisite_diagnostic_"
        "not_production_not_TDHF_not_Fig8_reproduction"
    ),
    summary_limitations: list[str] | None = None,
    target_filling: int | None = None,
    prepared_fingerprint: str | None = None,
    evidence_status: str = "pass",
    evidence_job_id: str = "201962",
) -> SimpleNamespace:
    output = root / "output"
    controller = root / "controller"
    output.mkdir(parents=True)
    controller.mkdir(parents=True)
    state_path = output / "hfdiag_state_201962.npz"
    state_payload = _stage7a_job_state_payload(source) if payload is None else payload
    np.savez(state_path, **state_payload)
    state_bytes = state_path.read_bytes()
    state_sha = hashlib.sha256(state_bytes).hexdigest()
    source_commit = "1" * 40
    source_commit_path = root / "SOURCE_COMMIT.txt"
    source_commit_path.write_text(source_commit + "\n", encoding="utf-8")
    manifest_path = root / "STATIC_SHA256SUMS.txt"
    manifest_path.write_text("job-style fixture manifest\n", encoding="utf-8")
    limitations = (
        [
            "No TDHF implementation or Kwan Fig.8(a) spectrum was computed.",
            "The NPZ is a diagnostic snapshot, not restart authority.",
        ]
        if summary_limitations is None
        else summary_limitations
    )
    summary = {
        "schema": summary_schema,
        "schema_version": TBG_ZERO_FIELD_COMPANION_TDHF_SUMMARY_SCHEMA_VERSION,
        "status": summary_status,
        "scope": summary_scope,
        "source_commit": source_commit,
        "job": {
            "job_id": "201962",
            "allocation_attestation": {"job_id": "201962", "status": "pass"},
        },
        "target": {"filling": source.filling if target_filling is None else target_filling},
        "limitations": limitations,
        "fingerprints": {
            "prepared_hf_action": (
                source.prepared_fingerprint
                if prepared_fingerprint is None
                else prepared_fingerprint
            ),
            "scf_run": "2" * 64,
        },
        "state": {
            "array_keys": (
                sorted(state_payload)
                if declared_array_keys is None
                else declared_array_keys
            ),
            "path": str(state_path.resolve()),
            "scope": (
                "diagnostic_complete_HF_state_not_restart_authority_"
                "not_TDHF_source"
            ),
            "sha256": state_sha,
            "size_bytes": len(state_bytes),
        },
    }
    summary_path = output / "hfdiag_summary_201962.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    records = {
        "source_commit": _stage7a_file_record(root, source_commit_path),
        "state": _stage7a_file_record(root, state_path),
        "static_source_manifest": _stage7a_file_record(root, manifest_path),
        "summary": _stage7a_file_record(root, summary_path),
    }
    evidence = {
        "schema": TBG_ZERO_FIELD_COMPANION_TDHF_EVIDENCE_BUNDLE_SCHEMA,
        "schema_version": (
            TBG_ZERO_FIELD_COMPANION_TDHF_EVIDENCE_BUNDLE_SCHEMA_VERSION
        ),
        "status": evidence_status,
        "scope": (
            "qualified_single_seed_companion_HF_prerequisite_diagnostic_"
            "not_production_not_TDHF_not_Fig8"
        ),
        "job_id": evidence_job_id,
        "limitations": [
            "state is diagnostic snapshot, not restart authority",
            "no global-ground-state, TDHF, or Fig8 claim",
        ],
        "records": records,
    }
    evidence_path = controller / "evidence_bundle_201962.json"
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return SimpleNamespace(
        state=state_path,
        summary=summary_path,
        evidence=evidence_path,
        manifest=manifest_path,
        state_sha256=state_sha,
        summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
    )



def test_companion_stage7a_scope_authority_and_no_package_export(
    companion_stage7a_source,
) -> None:
    source = companion_stage7a_source
    assert TBG_ZERO_FIELD_COMPANION_TDHF_SCOPE == (
        "Stage7A_diagnostic_only_not_generic_TDHF_authority_production_or_Fig8"
    )
    assert TBG_ZERO_FIELD_COMPANION_TDHF_PAPER_ARXIV == "2511.21683v1"
    assert TBG_ZERO_FIELD_COMPANION_TDHF_PAPER_EQUATIONS == (
        15,
        16,
        17,
        18,
        64,
        82,
        83,
        84,
        88,
        89,
        90,
    )
    assert "system_local_q_label_only_binds_companion_form_and_reciprocal_carry" in (
        TBG_ZERO_FIELD_COMPANION_TDHF_ARCHITECTURE_EXCEPTION
    )
    assert "generic_eigensolver_metric_normalization_and_stability_from_core_hf" in (
        TBG_ZERO_FIELD_COMPANION_TDHF_ARCHITECTURE_EXCEPTION
    )
    assert "not_package_export_not_production_not_Fig8" in (
        TBG_ZERO_FIELD_COMPANION_TDHF_ARCHITECTURE_EXCEPTION
    )
    front_door = importlib.import_module("mean_field.systems.tbg.zero_field")
    for name in (
        "TBGZeroFieldCompanionTDHFSource",
        "build_tbg_zero_field_companion_static_kernel",
        "evaluate_tbg_zero_field_companion_static_matrix_action",
        "load_tbg_zero_field_companion_tdhf_source_from_checkpoint_npz",
        "load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts",
        "solve_tbg_zero_field_companion_dense_spectrum",
    ):
        assert not hasattr(front_door, name)
    assert source.stage6_run is not None
    assert source.prepared.form is source.stage6_run.prepared.form
    assert source.prepared.screened_intFT_ev is (
        source.stage6_run.prepared.screened_intFT_ev
    )
    assert source.residuals.positive_gap_ev > 0.0
    assert source.residuals.mixed_aufbau_closure <= 1.0e-8
    assert source.residuals.spin_occupied_subspace_max_abs <= 1.0e-10
    assert source.residuals.common_spin_basis_eigenpair_max_abs_ev <= (
        TBG_ZERO_FIELD_COMPANION_TDHF_COMMON_SPIN_BASIS_ATOL_EV
    )
    assert "exactly identical by construction" in (
        TBG_ZERO_FIELD_COMPANION_TDHF_COMMON_SPIN_BASIS_SOURCE
    )
    assert source.diagnostic_consumption_receipt is None
    residual_metadata = source.residuals.to_metadata()
    assert set(residual_metadata) == {
        field.name for field in fields(type(source.residuals))
    }
    json.dumps(residual_metadata, sort_keys=True, allow_nan=False)


def test_companion_stage7a_raw_endpoint_q_labels_keep_distinct_provenance(
    companion_stage7a_source,
) -> None:
    source = companion_stage7a_source
    minus_endpoint = build_tbg_zero_field_companion_signed_q_label(source, (-1, 0))
    plus_endpoint = build_tbg_zero_field_companion_signed_q_label(source, (1, 0))
    assert minus_endpoint.raw != plus_endpoint.raw
    assert minus_endpoint.canonical_delta == plus_endpoint.canonical_delta == (1, 0)
    assert minus_endpoint.reciprocal_carry == (-1, 0)
    assert plus_endpoint.reciprocal_carry == (0, 0)
    assert minus_endpoint.fingerprint != plus_endpoint.fingerprint
    assert minus_endpoint.leg_carry(0, 0) == (-1, 0)
    assert plus_endpoint.leg_carry(1, 0) == (1, 0)

    minus_form = build_tbg_zero_field_companion_hf_form_factors(
        source,
        minus_endpoint,
    )
    plus_form = build_tbg_zero_field_companion_hf_form_factors(source, plus_endpoint)
    np.testing.assert_array_equal(minus_form.values, plus_form.values)
    assert not np.array_equal(minus_form.raw_g_labels, plus_form.raw_g_labels)
    assert minus_form.fingerprint != plus_form.fingerprint


def test_companion_stage7a_eq16_boundary_carry_and_hash_receipts(
    companion_stage7a_source,
) -> None:
    source = companion_stage7a_source
    q = build_tbg_zero_field_companion_signed_q_label(source, (1, 2))
    receipt = build_tbg_zero_field_companion_hf_form_factors(source, q)
    assert q.canonical_delta == (1, 2)
    assert ((-q.canonical_delta[0]) // q.N1, (-q.canonical_delta[1]) // q.N2) == (
        -1,
        -1,
    )
    expected_active = int(
        np.count_nonzero(
            source.prepared.screened_intFT_ev[
                q.canonical_delta[0],
                q.canonical_delta[1],
            ]
        )
    )
    assert receipt.interaction_active_count == expected_active
    assert receipt.inverse_mapped_count == expected_active
    assert receipt.inverse_missing_count == 0
    assert receipt.inverse_weight_max_abs_ev <= 1.0e-14
    assert receipt.eq16_matrix_comparison_count == expected_active * q.N1 * q.N2
    assert receipt.eq16_carry_max_abs <= 1.0e-10
    assert receipt.values_sha256 == _companion_fixture_array_sha256(receipt.values)
    assert receipt.source_g_labels_sha256 == _companion_fixture_array_sha256(
        receipt.source_g_labels
    )
    assert receipt.raw_g_labels_sha256 == _companion_fixture_array_sha256(
        receipt.raw_g_labels
    )

def test_companion_stage7a_eq16_incomplete_interaction_support_fails_closed(
    companion_stage7a_source,
    monkeypatch,
) -> None:
    source = companion_stage7a_source
    q = build_tbg_zero_field_companion_signed_q_label(source, (1, 2))
    minus_q = build_tbg_zero_field_companion_signed_q_label(
        source,
        (-q.raw[0], -q.raw[1]),
    )
    original_weights = companion_tdhf_module._interaction_weights
    labels = companion_tdhf_module._source_g_labels(source)
    label_to_index = {g: index for index, g in enumerate(labels)}
    source_weights = original_weights(source, q)
    inverse_weights = original_weights(source, minus_q)
    inverse_carry = (
        (-q.canonical_delta[0]) // q.N1,
        (-q.canonical_delta[1]) // q.N2,
    )
    selected_pair = None
    for source_index in np.flatnonzero(source_weights != 0.0):
        g1, g2 = labels[int(source_index)]
        inverse_index = label_to_index.get(
            (inverse_carry[0] - g1, inverse_carry[1] - g2)
        )
        if inverse_index is not None and inverse_weights[inverse_index] != 0.0:
            selected_pair = (int(source_index), inverse_index)
            break
    assert selected_pair is not None, (
        "fixture must contain an Eq. (16)-mapped interaction-active transfer"
    )
    selected_source_index, selected_inverse_index = selected_pair
    assert source_weights[selected_source_index] != 0.0
    assert inverse_weights[selected_inverse_index] != 0.0

    def incomplete_inverse_weights(source_arg, q_arg):
        weights = np.array(original_weights(source_arg, q_arg), copy=True)
        if q_arg.raw == minus_q.raw:
            weights[selected_inverse_index] = 0.0
        return weights

    monkeypatch.setattr(
        companion_tdhf_module,
        "_interaction_weights",
        incomplete_inverse_weights,
    )
    with pytest.raises(ValueError, match="inverse support is incomplete"):
        build_tbg_zero_field_companion_hf_form_factors(source, q)


def test_companion_stage7a_transition_roles_q_maps_and_generic_static_structure(
    companion_stage7a_source,
) -> None:
    source = companion_stage7a_source
    inventories = build_tbg_zero_field_companion_transition_inventories(source, (1, 0))
    q_inventory = inventories.q
    minus_inventory = inventories.minus_q
    assert TBG_ZERO_FIELD_COMPANION_TDHF_EQ90_SIGN_CONVENTION == (
        "paper_occupation_sign=n_mu(k+q)-n_nu(k)=-core_metric_sign;"
        "Eq90_is_K_phi=paper_occupation_sign*eta*omega*phi;"
        "L=J_core*K;lambda_L=-eta*omega"
    )
    assert all(pair.role == "ph" for pair in q_inventory.pairs[: q_inventory.ph_count])
    assert all(
        (pair.core_metric_sign, pair.paper_occupation_sign) == (1, -1)
        for pair in q_inventory.pairs[: q_inventory.ph_count]
    )
    assert all(pair.role == "hp" for pair in q_inventory.pairs[q_inventory.ph_count :])
    assert all(
        (pair.core_metric_sign, pair.paper_occupation_sign) == (-1, 1)
        for pair in q_inventory.pairs[q_inventory.ph_count :]
    )
    occupations = source.canonical_occupations
    for pair in q_inventory.pairs:
        assert pair.paper_occupation_sign == (
            int(
                occupations[
                    pair.k_target[0],
                    pair.k_target[1],
                    pair.mu_target,
                ]
            )
            - int(
                occupations[
                    pair.k_source[0],
                    pair.k_source[1],
                    pair.nu_source,
                ]
            )
        )
        assert pair.paper_occupation_sign == -pair.core_metric_sign
    for index, mapped in enumerate(q_inventory.conjugate_indices_at_minus_q):
        assert minus_inventory.conjugate_indices_at_minus_q[int(mapped)] == index
        pair = q_inventory.pairs[index]
        conjugate = minus_inventory.pairs[int(mapped)]
        assert conjugate.k_source == pair.k_target
        assert conjugate.k_target == pair.k_source
        assert conjugate.mu_target == pair.nu_source
        assert conjugate.nu_source == pair.mu_target

    q_kernel = build_tbg_zero_field_companion_static_kernel(
        source,
        (1, 0),
        spin_sector="triplet",
    )
    minus_kernel = build_tbg_zero_field_companion_static_kernel(
        source,
        (-1, 0),
        spin_sector="triplet",
    )
    assert q_kernel.direct_multiplier == minus_kernel.direct_multiplier == 0
    assert q_kernel.residuals.K_hermiticity_max_abs_ev <= 1.0e-10
    assert q_kernel.residuals.L_pseudo_hermiticity_max_abs_ev <= 1.0e-10
    np.testing.assert_array_equal(
        q_kernel.core_metric,
        np.asarray(
            [pair.core_metric_sign for pair in q_kernel.inventory.pairs],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(
        q_kernel.L_ev,
        q_kernel.core_metric[:, None] * q_kernel.K_ev,
    )

    assert TBG_ZERO_FIELD_COMPANION_TDHF_DEGENERACY_ATOL_EV == 1.0e-10
    assert TBG_ZERO_FIELD_COMPANION_TDHF_RAW_EIGENSOLVER_RESIDUAL_ATOL_EV == 1.0e-9
    assert (
        TBG_ZERO_FIELD_COMPANION_TDHF_SELECTED_EIGENSOLVER_RESIDUAL_ATOL_EV
        == 1.0e-9
    )
    assert TBG_ZERO_FIELD_COMPANION_TDHF_Q0_RAW_PAIRING_RESIDUAL_ATOL_EV == 1.0e-9
    q_spectrum = solve_tbg_zero_field_companion_dense_spectrum(q_kernel)
    minus_spectrum = solve_tbg_zero_field_companion_dense_spectrum(minus_kernel)
    n_pairs = q_kernel.inventory.ph_count
    core_spectrum = solve_tdhf_liouvillian(
        q_kernel.L_ev,
        n_pairs=n_pairs,
        energy_tol=(
            companion_tdhf_module.TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV
        ),
        imag_tol=(
            companion_tdhf_module.TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV
        ),
        norm_tol=(
            companion_tdhf_module.TBG_ZERO_FIELD_COMPANION_TDHF_METRIC_CLASSIFICATION_ATOL
        ),
        degeneracy_tol=TBG_ZERO_FIELD_COMPANION_TDHF_DEGENERACY_ATOL_EV,
    )
    np.testing.assert_allclose(
        q_spectrum.raw_eigenvalues_ev,
        core_spectrum.raw_eigenvalues,
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        q_spectrum.raw_J_metric_norms,
        core_spectrum.raw_eta_norms,
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        q_spectrum.raw_eigensolver_residuals_ev,
        core_spectrum.raw_residuals,
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        q_spectrum.selected_right_eigenvectors,
        core_spectrum.amplitudes,
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        q_spectrum.selected_eigensolver_residuals_ev,
        core_spectrum.residuals,
        rtol=0.0,
        atol=1.0e-13,
    )
    assert q_spectrum.selected_right_eigenvectors.shape == (
        q_spectrum.selected_eigenvalues_ev.size,
        len(q_kernel.inventory.pairs),
    )
    np.testing.assert_array_equal(
        q_spectrum.paper_eta_omega_ev,
        -q_spectrum.selected_eigenvalues_ev,
    )
    expected_classification = classify_tdhf_stability(
        core_spectrum.raw_eigenvalues,
        core_spectrum.energies,
        n_pairs=n_pairs,
        structure_ok=True,
        imag_tol=(
            companion_tdhf_module.TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV
        ),
        energy_tol=(
            companion_tdhf_module.TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV
        ),
    )
    assert q_spectrum.anomaly_classification == expected_classification
    assert q_spectrum.diagnostic_passed
    assert q_spectrum.diagnostic_failure_reasons == ()
    assert q_spectrum.raw_eigensolver_residual_max_ev <= (
        TBG_ZERO_FIELD_COMPANION_TDHF_RAW_EIGENSOLVER_RESIDUAL_ATOL_EV
    )
    assert len(q_spectrum.raw_eigenvalues_ev) == len(q_kernel.inventory.pairs)
    assert q_spectrum.raw_eigenvalues_ev.dtype == np.dtype(np.complex128)
    classified = set(q_spectrum.complex_indices.tolist())
    classified.update(q_spectrum.negative_real_indices.tolist())
    classified.update(q_spectrum.static_real_indices.tolist())
    classified.update(
        np.flatnonzero(
            (np.abs(q_spectrum.raw_eigenvalues_ev.imag) <= 1.0e-10)
            & (q_spectrum.raw_eigenvalues_ev.real > 1.0e-10)
        ).tolist()
    )
    assert classified == set(range(len(q_spectrum.raw_eigenvalues_ev)))
    pairing = tbg_zero_field_companion_signed_spectral_pairing(
        q_spectrum,
        minus_spectrum,
    )
    assert pairing.q_raw == (1, 0)
    assert pairing.minus_q_raw == (-1, 0)
    assert pairing.max_abs_ev <= 1.0e-9


def test_companion_stage7a_signed_spectral_pairing_preserves_duplicate_multiplicity(
) -> None:
    q_kernel = SimpleNamespace(
        inventory=SimpleNamespace(q=SimpleNamespace(raw=(1, 0))),
        source_fingerprint="synthetic-stage7a-source",
        spin_sector="triplet",
    )
    minus_kernel = SimpleNamespace(
        inventory=SimpleNamespace(q=SimpleNamespace(raw=(-1, 0))),
        source_fingerprint="synthetic-stage7a-source",
        spin_sector="triplet",
    )
    q_spectrum = SimpleNamespace(
        kernel=q_kernel,
        raw_eigenvalues_ev=np.asarray([1.0, 1.0, 2.0], dtype=np.complex128),
    )
    minus_spectrum = SimpleNamespace(
        kernel=minus_kernel,
        raw_eigenvalues_ev=np.asarray([-1.0, -2.0, -2.0], dtype=np.complex128),
    )

    pairing = tbg_zero_field_companion_signed_spectral_pairing(
        q_spectrum,
        minus_spectrum,
    )
    assert pairing.q_to_minus_q_max_abs_ev == 1.0
    assert pairing.minus_q_to_q_max_abs_ev == 1.0
    assert pairing.max_abs_ev == 1.0

    unequal_minus_spectrum = SimpleNamespace(
        kernel=minus_kernel,
        raw_eigenvalues_ev=np.asarray([-1.0, -2.0], dtype=np.complex128),
    )
    with pytest.raises(ValueError, match="must have the same size"):
        tbg_zero_field_companion_signed_spectral_pairing(
            q_spectrum,
            unequal_minus_spectrum,
        )


def test_companion_stage7a_dense_spectrum_fails_result_on_material_raw_residual(
    companion_stage7a_source,
    monkeypatch,
) -> None:
    kernel = build_tbg_zero_field_companion_static_kernel(
        companion_stage7a_source,
        (1, 0),
        spin_sector="triplet",
    )
    core_solver = companion_tdhf_module.solve_tdhf_liouvillian

    def solver_with_material_raw_residual(*args, **kwargs):
        spectrum = core_solver(*args, **kwargs)
        return replace(
            spectrum,
            raw_residuals=np.full_like(
                spectrum.raw_residuals,
                2.0 * TBG_ZERO_FIELD_COMPANION_TDHF_RAW_EIGENSOLVER_RESIDUAL_ATOL_EV,
            ),
        )

    monkeypatch.setattr(
        companion_tdhf_module,
        "solve_tdhf_liouvillian",
        solver_with_material_raw_residual,
    )
    result = solve_tbg_zero_field_companion_dense_spectrum(kernel)
    assert not result.diagnostic_passed
    assert result.diagnostic_failure_reasons == (
        "max_raw_eigensolver_residual_exceeds_1e-9_eV",
    )
    assert result.raw_eigensolver_residual_max_ev > (
        TBG_ZERO_FIELD_COMPANION_TDHF_RAW_EIGENSOLVER_RESIDUAL_ATOL_EV
    )
    assert result.raw_eigenvalues_ev.size == len(kernel.inventory.pairs)
    assert result.raw_J_metric_norms.size == len(kernel.inventory.pairs)
    assert result.raw_eigensolver_residuals_ev.size == len(kernel.inventory.pairs)


def test_companion_stage7a_dense_spectrum_fails_when_selected_residual_fails(
    companion_stage7a_source,
    monkeypatch,
) -> None:
    kernel = build_tbg_zero_field_companion_static_kernel(
        companion_stage7a_source,
        (1, 0),
        spin_sector="triplet",
    )
    core_solver = companion_tdhf_module.solve_tdhf_liouvillian

    def solver_with_material_selected_residual(*args, **kwargs):
        assert len(args) == 1
        assert args[0] is kernel.L_ev
        assert kwargs == {
            "n_pairs": kernel.inventory.ph_count,
            "energy_tol": (
                companion_tdhf_module.TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV
            ),
            "imag_tol": (
                companion_tdhf_module.TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV
            ),
            "norm_tol": (
                companion_tdhf_module.TBG_ZERO_FIELD_COMPANION_TDHF_METRIC_CLASSIFICATION_ATOL
            ),
            "degeneracy_tol": TBG_ZERO_FIELD_COMPANION_TDHF_DEGENERACY_ATOL_EV,
        }
        spectrum = core_solver(*args, **kwargs)
        assert np.all(np.isfinite(spectrum.raw_residuals))
        assert np.max(np.abs(spectrum.raw_residuals)) <= (
            TBG_ZERO_FIELD_COMPANION_TDHF_RAW_EIGENSOLVER_RESIDUAL_ATOL_EV
        )
        material_selected_residual = (
            2.0
            * TBG_ZERO_FIELD_COMPANION_TDHF_SELECTED_EIGENSOLVER_RESIDUAL_ATOL_EV
        )
        synthetic_X = np.zeros((1, spectrum.X.shape[1]), dtype=np.complex128)
        synthetic_X[0, 0] = 1.0
        return replace(
            spectrum,
            eigenvalues=np.asarray([0.0 + 0.0j], dtype=np.complex128),
            energies=np.asarray([0.0], dtype=np.float64),
            X=synthetic_X,
            Y=np.zeros_like(synthetic_X),
            eta_norms=np.asarray([1.0], dtype=np.float64),
            residuals=np.asarray([material_selected_residual], dtype=np.float64),
            selected_indices=np.asarray([0], dtype=np.int64),
        )

    monkeypatch.setattr(
        companion_tdhf_module,
        "solve_tdhf_liouvillian",
        solver_with_material_selected_residual,
    )
    result = solve_tbg_zero_field_companion_dense_spectrum(kernel)
    assert not result.diagnostic_passed
    assert result.diagnostic_failure_reasons == (
        "max_selected_normalized_mode_eigensolver_residual_exceeds_1e-9_eV",
    )
    assert result.raw_eigensolver_residual_max_ev <= (
        TBG_ZERO_FIELD_COMPANION_TDHF_RAW_EIGENSOLVER_RESIDUAL_ATOL_EV
    )
    assert np.max(np.abs(result.selected_eigensolver_residuals_ev)) > (
        TBG_ZERO_FIELD_COMPANION_TDHF_SELECTED_EIGENSOLVER_RESIDUAL_ATOL_EV
    )

def test_companion_stage7a_dense_spectrum_q0_gates_same_matrix_pairing(
    companion_stage7a_source,
    monkeypatch,
) -> None:
    kernel = build_tbg_zero_field_companion_static_kernel(
        companion_stage7a_source,
        (0, 0),
        spin_sector="triplet",
    )
    core_solver = companion_tdhf_module.solve_tdhf_liouvillian
    material_pairing_residual = (
        2.0 * TBG_ZERO_FIELD_COMPANION_TDHF_Q0_RAW_PAIRING_RESIDUAL_ATOL_EV
    )

    def solver_with_material_pairing_residual(*args, **kwargs):
        spectrum = core_solver(*args, **kwargs)
        return replace(spectrum, pairing_residual=material_pairing_residual)

    monkeypatch.setattr(
        companion_tdhf_module,
        "solve_tdhf_liouvillian",
        solver_with_material_pairing_residual,
    )
    result = solve_tbg_zero_field_companion_dense_spectrum(kernel)
    assert kernel.inventory.q.raw == (0, 0)
    assert not result.diagnostic_passed
    assert result.diagnostic_failure_reasons == (
        "q0_same_matrix_raw_pairing_residual_exceeds_1e-9_eV",
    )
    assert result.raw_pairing_residual_ev == material_pairing_residual

def test_companion_stage7a_dense_spectrum_generic_q_does_not_gate_same_matrix_pairing_or_anomaly(
    companion_stage7a_source,
    monkeypatch,
) -> None:
    kernel = build_tbg_zero_field_companion_static_kernel(
        companion_stage7a_source,
        (1, 0),
        spin_sector="triplet",
    )
    core_solver = companion_tdhf_module.solve_tdhf_liouvillian
    core_classifier = companion_tdhf_module.classify_tdhf_stability
    material_pairing_residual = (
        2.0 * TBG_ZERO_FIELD_COMPANION_TDHF_Q0_RAW_PAIRING_RESIDUAL_ATOL_EV
    )

    def solver_with_material_pairing_residual(*args, **kwargs):
        spectrum = core_solver(*args, **kwargs)
        return replace(spectrum, pairing_residual=material_pairing_residual)

    def classify_as_unstable(*args, **kwargs):
        classification = core_classifier(*args, **kwargs)
        return replace(
            classification,
            stable=False,
            complex_eigenvalues=True,
            complex_count=max(1, classification.complex_count),
            reason="test_control_flow_unstable",
        )

    monkeypatch.setattr(
        companion_tdhf_module,
        "solve_tdhf_liouvillian",
        solver_with_material_pairing_residual,
    )
    monkeypatch.setattr(
        companion_tdhf_module,
        "classify_tdhf_stability",
        classify_as_unstable,
    )
    result = solve_tbg_zero_field_companion_dense_spectrum(kernel)
    assert kernel.inventory.q.raw == (1, 0)
    assert result.raw_pairing_residual_ev == material_pairing_residual
    assert not result.anomaly_classification.stable
    assert result.diagnostic_passed
    assert result.diagnostic_failure_reasons == ()

def test_companion_stage7a_finite_q_direct_action_matches_each_dense_eq90_term(
    companion_stage7a_source,
) -> None:
    source = companion_stage7a_source
    kernel = build_tbg_zero_field_companion_static_kernel(
        source,
        (1, 0),
        spin_sector="singlet",
    )
    rng = np.random.default_rng(7301)
    vector = rng.standard_normal(len(kernel.inventory.pairs)) + 1.0j * rng.standard_normal(
        len(kernel.inventory.pairs)
    )
    direct = evaluate_tbg_zero_field_companion_static_matrix_action(
        source,
        (1, 0),
        vector,
        spin_sector="singlet",
    )
    assert direct.inventory.pairs == kernel.inventory.pairs
    assert np.max(np.abs(kernel.hartree_ev)) > 0.0
    assert np.max(np.abs(kernel.exchange_ev)) > 0.0
    for direct_term, dense_term in (
        (direct.bare_action_ev, kernel.bare_ev),
        (direct.hartree_action_ev, kernel.hartree_ev),
        (direct.exchange_action_ev, kernel.exchange_ev),
        (direct.K_action_ev, kernel.K_ev),
    ):
        np.testing.assert_allclose(
            direct_term,
            dense_term @ vector,
            rtol=0.0,
            atol=1.0e-12,
        )

def test_companion_stage7a_q0_scalar_response_all_columns_factor_one_triplet_singlet(
    companion_stage7a_source,
) -> None:
    source = companion_stage7a_source
    single_spin = _build_tbg_zero_field_companion_single_spin_q0_parity_oracle(source)
    triplet = build_tbg_zero_field_companion_q0_parity_oracle(
        source,
        spin_sector="triplet",
    )
    singlet = build_tbg_zero_field_companion_q0_parity_oracle(
        source,
        spin_sector="singlet",
    )
    assert single_spin.direct_multiplier == 1
    assert triplet.direct_multiplier == 0
    assert singlet.direct_multiplier == 2
    assert single_spin.columns == triplet.columns == singlet.columns
    for oracle in (single_spin, triplet, singlet):
        assert oracle.residuals.H_D_A_max_abs_ev <= 1.0e-12
        assert oracle.residuals.H_E_A_max_abs_ev <= 1.0e-12
        assert oracle.residuals.H_D_B_max_abs_ev <= 1.0e-12
        assert oracle.residuals.H_E_B_max_abs_ev <= 1.0e-12
        assert oracle.residuals.max_abs_ev <= 1.0e-12


def test_companion_stage7a_random_pointwise_u1_gauge_covariance_is_entrywise(
    companion_stage7a_source,
) -> None:
    source = companion_stage7a_source
    rng = np.random.default_rng(7319)
    state_phases = np.exp(
        1.0j
        * rng.uniform(
            -np.pi,
            np.pi,
            size=(
                source.prepared.params.N1,
                source.prepared.params.N2,
                source.dimension,
            ),
        )
    )
    gauged_vectors = source.eigenvectors * state_phases[:, :, None, None, :]
    gauged_source = _stage7a_in_memory_source(source, eigenvectors=gauged_vectors)
    assert gauged_source.source_kind == "in_memory_diagnostic_arrays"
    assert gauged_source.source_artifact_sha256 is None
    assert gauged_source.diagnostic_consumption_receipt is None
    base_kernel = build_tbg_zero_field_companion_static_kernel(
        source,
        (1, 0),
        spin_sector="singlet",
    )
    gauged_kernel = build_tbg_zero_field_companion_static_kernel(
        gauged_source,
        (1, 0),
        spin_sector="singlet",
    )
    assert gauged_kernel.inventory.pairs == base_kernel.inventory.pairs
    transition_phases = np.asarray(
        [
            np.conj(
                state_phases[
                    pair.k_source[0],
                    pair.k_source[1],
                    pair.nu_source,
                ]
            )
            * state_phases[
                pair.k_target[0],
                pair.k_target[1],
                pair.mu_target,
            ]
            for pair in base_kernel.inventory.pairs
        ],
        dtype=np.complex128,
    )
    for base_term, gauged_term in (
        (base_kernel.hartree_ev, gauged_kernel.hartree_ev),
        (base_kernel.exchange_ev, gauged_kernel.exchange_ev),
        (base_kernel.K_ev, gauged_kernel.K_ev),
    ):
        expected = (
            np.conj(transition_phases)[:, None]
            * base_term
            * transition_phases[None, :]
        )
        np.testing.assert_allclose(gauged_term, expected, rtol=0.0, atol=2.0e-12)


def test_companion_stage7a_job_artifact_loader_binds_receipt_and_real_keys(
    companion_stage7a_source,
    tmp_path,
) -> None:
    source = companion_stage7a_source
    artifacts = _stage7a_write_job_artifacts(tmp_path / "job201962", source)
    loaded = load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
        source.prepared,
        artifacts.state,
        artifacts.summary,
        artifacts.evidence,
    )
    assert loaded.source_kind == "stage6_diagnostic_artifacts"
    assert loaded.source_artifact_sha256 == artifacts.state_sha256
    assert loaded.source_summary_sha256 == artifacts.summary_sha256
    assert loaded.array_hashes == source.array_hashes
    np.testing.assert_array_equal(loaded.occupations, source.occupations)
    receipt = loaded.diagnostic_consumption_receipt
    assert isinstance(receipt, Stage7ADiagnosticConsumptionReceipt)
    assert receipt.consumption_scope == (
        TBG_ZERO_FIELD_COMPANION_TDHF_DIAGNOSTIC_CONSUMPTION_SCOPE
    )
    assert receipt.summary_schema == (
        "mean_field.tbg.kwan2511_fig8a.stage6_hf_diagnostic"
    )
    assert receipt.summary_status == receipt.evidence_status == "pass"
    assert receipt.evidence_schema == (
        "mean_field.tbg.kwan2511.stage6_hfdiag_evidence_bundle"
    )
    assert receipt.job_id == "201962"
    assert receipt.source_commit == "1" * 40
    assert receipt.prepared_fingerprint == source.prepared_fingerprint
    assert receipt.state_sha256 == artifacts.state_sha256
    assert receipt.summary_sha256 == artifacts.summary_sha256
    assert receipt.evidence_bundle_sha256 == artifacts.evidence_sha256
    assert tuple(record.name for record in receipt.records) == (
        "source_commit",
        "state",
        "static_source_manifest",
        "summary",
    )
    assert "not_TDHF" in receipt.summary_scope
    assert "not_restart_authority" in receipt.state_scope
    assert "not_TDHF" in receipt.evidence_scope
    assert any("No TDHF implementation" in x for x in receipt.summary_limitations)
    assert len(receipt.fingerprint) == len(loaded.fingerprint) == 64

def test_companion_stage7a_job_artifact_loader_hashes_every_record_fail_closed(
    companion_stage7a_source,
    tmp_path,
) -> None:
    source = companion_stage7a_source
    manifest_artifacts = _stage7a_write_job_artifacts(
        tmp_path / "manifest_tamper",
        source,
    )
    manifest_artifacts.manifest.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="static_source_manifest SHA-256"):
        load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
            source.prepared,
            manifest_artifacts.state,
            manifest_artifacts.summary,
            manifest_artifacts.evidence,
        )

    state_artifacts = _stage7a_write_job_artifacts(tmp_path / "state_tamper", source)
    state_artifacts.state.write_bytes(state_artifacts.state.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="evidence record state SHA-256"):
        load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
            source.prepared,
            state_artifacts.state,
            state_artifacts.summary,
            state_artifacts.evidence,
        )

def test_companion_stage7a_job_schema_scope_inventory_and_shape_fail_closed(
    companion_stage7a_source,
    tmp_path,
) -> None:
    source = companion_stage7a_source

    bad_schema = _stage7a_write_job_artifacts(
        tmp_path / "bad_schema",
        source,
        summary_schema="wrong.schema",
    )
    with pytest.raises(ValueError, match="summary schema"):
        load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
            source.prepared,
            bad_schema.state,
            bad_schema.summary,
            bad_schema.evidence,
        )

    bad_status = _stage7a_write_job_artifacts(
        tmp_path / "bad_status",
        source,
        summary_status="scientific_gate_failed",
    )
    with pytest.raises(ValueError, match="summary status must be pass"):
        load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
            source.prepared,
            bad_status.state,
            bad_status.summary,
            bad_status.evidence,
        )

    bad_evidence_status = _stage7a_write_job_artifacts(
        tmp_path / "bad_evidence_status",
        source,
        evidence_status="fail",
    )
    with pytest.raises(ValueError, match="evidence bundle status must be pass"):
        load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
            source.prepared,
            bad_evidence_status.state,
            bad_evidence_status.summary,
            bad_evidence_status.evidence,
        )

    bad_job_id = _stage7a_write_job_artifacts(
        tmp_path / "bad_job_id",
        source,
        evidence_job_id="201963",
    )
    with pytest.raises(ValueError, match="summary and evidence bundle job ids differ"):
        load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
            source.prepared,
            bad_job_id.state,
            bad_job_id.summary,
            bad_job_id.evidence,
        )

    bad_scope = _stage7a_write_job_artifacts(
        tmp_path / "bad_scope",
        source,
        summary_scope="diagnostic_without_explicit_limit",
    )
    with pytest.raises(ValueError, match="scope must explicitly remain not_TDHF"):
        load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
            source.prepared,
            bad_scope.state,
            bad_scope.summary,
            bad_scope.evidence,
        )

    bad_limitation = _stage7a_write_job_artifacts(
        tmp_path / "bad_limitation",
        source,
        summary_limitations=["No production claim."],
    )
    with pytest.raises(ValueError, match="explicitly state no TDHF implementation"):
        load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
            source.prepared,
            bad_limitation.state,
            bad_limitation.summary,
            bad_limitation.evidence,
        )

    payload = _stage7a_job_state_payload(source)
    inventory_mismatch = _stage7a_write_job_artifacts(
        tmp_path / "inventory_mismatch",
        source,
        payload=payload,
        declared_array_keys=sorted((*payload, "summary_only_array")),
    )
    with pytest.raises(ValueError, match="inventory does not exactly equal"):
        load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
            source.prepared,
            inventory_mismatch.state,
            inventory_mismatch.summary,
            inventory_mismatch.evidence,
        )

    missing_required_payload = {
        name: value
        for name, value in _stage7a_job_state_payload(source).items()
        if name != "final_H_total_ev"
    }
    missing_required = _stage7a_write_job_artifacts(
        tmp_path / "missing_required",
        source,
        payload=missing_required_payload,
    )
    with pytest.raises(ValueError, match="omit required Stage6 arrays"):
        load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
            source.prepared,
            missing_required.state,
            missing_required.summary,
            missing_required.evidence,
        )

    bad_shape_payload = _stage7a_job_state_payload(source)
    bad_shape_payload["final_hf_eigenvectors"] = bad_shape_payload[
        "final_hf_eigenvectors"
    ][..., :-1]
    bad_shape = _stage7a_write_job_artifacts(
        tmp_path / "bad_shape",
        source,
        payload=bad_shape_payload,
    )
    with pytest.raises(ValueError, match="final_hf_eigenvectors must have exact shape"):
        load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
            source.prepared,
            bad_shape.state,
            bad_shape.summary,
            bad_shape.evidence,
        )

    wrong_prepared = _stage7a_write_job_artifacts(
        tmp_path / "wrong_prepared",
        source,
        prepared_fingerprint="0" * 64,
    )
    with pytest.raises(ValueError, match="prepared_hf_action fingerprint"):
        load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
            source.prepared,
            wrong_prepared.state,
            wrong_prepared.summary,
            wrong_prepared.evidence,
        )

def test_companion_stage7a_actual_job201962_artifacts_when_present() -> None:
    root = Path(
        "/data/home/ziyuzhu/.runs/"
        "Mean_Field_1683426_tbg_kwan_stage6_hfdiag_v2_20260802"
    )
    state_path = root / "output/hfdiag_state_201962.npz"
    summary_path = root / "output/hfdiag_summary_201962.json"
    evidence_path = root / "controller/evidence_bundle_201962.json"
    if not all(path.is_file() for path in (state_path, summary_path, evidence_path)):
        pytest.skip("immutable job201962 Stage6 artifact triplet is not present")

    params = TBGZeroFieldCompanionSingleParticleParams(
        N1=10,
        N2=10,
        Ng1=4,
        Ng2=4,
        n_active=1,
        theta_deg=1.05,
        wAA_ev=0.08,
        wAB_ev=0.11,
        strain=0.0,
        strain_angle_deg=0.0,
    )
    single_particle = solve_tbg_zero_field_companion_single_particle(params)
    interaction = solve_tbg_zero_field_companion_interaction(
        params,
        TBGZeroFieldCompanionSourceInteractionSpec(),
        rlv_geometry=single_particle.rlv_geometry,
    )
    prepared = prepare_tbg_zero_field_companion_hf_action(
        single_particle,
        interaction,
        spec=TBGZeroFieldCompanionHFActionSpec(epsr=10.0, exchange=True),
    )
    assert prepared.fingerprint == (
        "19a0f744a1532c10f4eba901f7fbf875923d75d29b8e3f23c63dbb786dbb1882"
    )
    loaded = load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
        prepared,
        state_path,
        summary_path,
        evidence_path,
    )
    receipt = loaded.diagnostic_consumption_receipt
    assert isinstance(receipt, Stage7ADiagnosticConsumptionReceipt)
    assert receipt.job_id == "201962"
    assert receipt.source_commit == "1683426959a009a2e31631b786199ecc13b6249d"
    assert receipt.prepared_fingerprint == prepared.fingerprint
    assert receipt.state_sha256 == (
        "3c5bc2ce20851c4ddb0c04ed0b7ba88f93d5a4cd91e37af022b3db023ee79c0c"
    )
    assert receipt.state_size_bytes == 158852
    assert receipt.summary_sha256 == (
        "485e236e1460aa9b3588b6a49cfae80ee7b2224b711909b08402adc3b4d94775"
    )
    assert receipt.evidence_bundle_sha256 == (
        "719ffaac4205573e078c1e2aa11c05d99db1e8888aff9d6b23a34b00c0a370ff"
    )
    summary_fingerprints = dict(receipt.summary_fingerprints)
    assert summary_fingerprints["prepared_hf_action"] == prepared.fingerprint
    assert summary_fingerprints["scf_run"] == (
        "d3f499eb4de2c41c947772f00159d26cac5e24cb34528a77f76a45393e858b35"
    )
    record_hashes = {record.name: record.sha256 for record in receipt.records}
    assert record_hashes["state"] == receipt.state_sha256
    assert record_hashes["summary"] == receipt.summary_sha256
    assert record_hashes["source_commit"] == (
        "77fc66ce832d11f2e973398423cc110e1e3790336926a13771ffface254114fc"
    )
    assert loaded.residuals.checkpoint_eigenpair_max_abs_ev <= 1.0e-12
    assert loaded.residuals.eigenvector_unitarity_max_abs <= 2.0e-15
    assert loaded.residuals.common_spin_basis_eigenpair_max_abs_ev <= 1.0e-10
    assert loaded.residuals.mixed_aufbau_closure == pytest.approx(
        8.5339254776197e-9,
        rel=0.0,
        abs=1.0e-15,
    )
    assert loaded.residuals.positive_gap_ev == pytest.approx(
        0.024959582701741298,
        rel=0.0,
        abs=1.0e-14,
    )
    assert len(loaded.array_hashes.fingerprint) == len(receipt.fingerprint) == 64
    assert len(loaded.fingerprint) == 64


def test_companion_stage7a_source_and_in_memory_mutations_fail_closed(
    companion_stage7a_source,
) -> None:
    source = companion_stage7a_source
    _assert_stage4_readonly_bytes_mutation_fails_closed(
        source.eigenvectors,
        lambda: source.fingerprint,
        match="Stage7A source array hashes no longer match live arrays",
    )
    _assert_stage4_readonly_bytes_mutation_fails_closed(
        source.prepared.form,
        lambda: source.fingerprint,
        match="prepared array_hashes no longer match live prepared arrays",
    )

    altered_H = np.array(source.H_total_ev, copy=True)
    altered_H[0, 0, 0, 0, 0] += 1.0e-6
    with pytest.raises(ValueError, match="H_total_ev does not match live prepared.evaluate"):
        _stage7a_in_memory_source(source, H_total_ev=altered_H)
