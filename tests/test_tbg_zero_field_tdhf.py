from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.core.hf import (
    build_projected_interaction_hamiltonian,
    calculate_norm_convergence,
    compute_hf_energy,
)
from mean_field.systems.tbg.params import TBGParameters
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
