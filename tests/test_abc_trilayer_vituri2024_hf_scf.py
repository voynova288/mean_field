"""Reduced faithful tests for the independent Vituri homogeneous HF source."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mean_field.core.hf import DensityUpdateResult
from mean_field.systems.abc_trilayer.vituri2024_hf import (
    vituri2024_native_density_to_conventional_k_diagonal,
)
import mean_field.systems.abc_trilayer.vituri2024_hf_scf as hf_scf
from mean_field.systems.abc_trilayer.vituri2024_hf_scf import (
    VITURI2024_DELTA1_EV,
    VITURI2024_HF_SCF_AUTHORITY,
    VITURI2024_TOTAL_HOLE_DENSITY_CM2,
    Vituri2024CartesianHFSpec,
    analyze_vituri2024_global_aufbau_boundary,
    Vituri2024InitialFockBoundaryScanChoice,
    Vituri2024InitialFockBoundarySelection,
    build_vituri2024_cartesian_mesh,
    make_vituri2024_hf_problem,
    make_vituri2024_hf_state,
    prepare_vituri2024_homogeneous_hf,
    run_vituri2024_hf_seed,
    scan_vituri2024_initial_fock_aufbau_boundaries,
)


def _prepared():
    return prepare_vituri2024_homogeneous_hf(
        Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=1)
    )


def test_cartesian_spec_closes_density_area_spacing_and_exact_mesh() -> None:
    spec = Vituri2024CartesianHFSpec(mesh_size=5, holes_per_valley=2)
    mesh, labels = build_vituri2024_cartesian_mesh(spec)
    assert spec.nk == 25
    assert spec.total_holes == 4
    assert spec.total_electrons == 96
    assert spec.delta1_ev == VITURI2024_DELTA1_EV
    assert spec.actual_total_hole_density_cm2 == pytest.approx(
        VITURI2024_TOTAL_HOLE_DENSITY_CM2
    )
    assert spec.delta_k_inverse_angstrom == pytest.approx(
        2.0 * np.pi / np.sqrt(spec.area_angstrom_squared)
    )
    assert mesh.shape == (25, 2)
    assert labels.shape == (25, 2)
    assert len({tuple(value) for value in labels.tolist()}) == 25
    assert np.array_equal(mesh, labels * spec.delta_k_inverse_angstrom)
    assert not mesh.flags.writeable
    assert spec.production_ready is False
    assert spec.paper_reproduction_verified is False


def test_global_aufbau_boundary_analyzer_uses_global_rank_and_full_shell() -> None:
    fock = np.zeros((2, 2, 2), dtype=np.complex128)
    # Both occupied states are at k=0; a per-k rank-one policy is not equivalent.
    fock[:, :, 0] = np.diag([0.0, 0.5])
    fock[:, :, 1] = np.diag([2.0, 3.0])
    analysis = analyze_vituri2024_global_aufbau_boundary(
        fock,
        total_occupied=2,
        minimum_gap_to_eigensolver_residual_ratio=1.0e3,
    )
    assert analysis.boundary_gap_ev == 1.5
    assert analysis.shell_multiplicity == 1
    assert analysis.occupied_in_boundary_shell == 1
    assert analysis.closed_global_aufbau_boundary
    assert analysis.fock_energy_scale_ev == 3.0
    assert analysis.effective_eigensolver_residual_floor_ev == pytest.approx(
        np.finfo(np.float64).eps * 3.0
    )
    assert analysis.gap_to_effective_eigensolver_residual_ratio == pytest.approx(
        1.5 / analysis.effective_eigensolver_residual_floor_ev
    )

    degenerate = fock.copy()
    degenerate[:, :, 0] = np.diag([0.0, 1.0])
    degenerate[:, :, 1] = np.diag([1.0, 3.0])
    complete = analyze_vituri2024_global_aufbau_boundary(
        degenerate,
        total_occupied=3,
        minimum_gap_to_eigensolver_residual_ratio=1.0e3,
    )
    assert complete.boundary_gap_ev == 2.0
    assert complete.shell_multiplicity == 2
    assert complete.occupied_in_boundary_shell == 2
    assert complete.closed_global_aufbau_boundary
    partial = analyze_vituri2024_global_aufbau_boundary(
        degenerate,
        total_occupied=2,
        minimum_gap_to_eigensolver_residual_ratio=1.0e3,
    )
    assert partial.boundary_gap_ev == 0.0
    assert partial.shell_multiplicity == 2
    assert partial.occupied_in_boundary_shell == 1
    assert not partial.closed_global_aufbau_boundary
    with pytest.raises(ValueError, match="analysis drifted"):
        replace(partial, closed_global_aufbau_boundary=True)


def test_global_aufbau_boundary_uses_actual_residual_when_it_dominates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_eigh = hf_scf.np.linalg.eigh

    def biased_eigh(matrix):
        values, vectors = original_eigh(matrix)
        values = values.copy()
        values[0] += 1.0e-8
        return values, vectors

    monkeypatch.setattr(hf_scf.np.linalg, "eigh", biased_eigh)
    fock = np.zeros((2, 2, 2), dtype=np.complex128)
    rotation = np.asarray([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2.0)
    fock[:, :, 0] = rotation @ np.diag([0.0, 0.5]) @ rotation.T
    fock[:, :, 1] = np.diag([2.0, 3.0])
    analysis = analyze_vituri2024_global_aufbau_boundary(
        fock,
        total_occupied=2,
        minimum_gap_to_eigensolver_residual_ratio=1.0e3,
    )
    assert analysis.maximum_eigensolver_residual_ev == pytest.approx(1.0e-8)
    assert analysis.effective_eigensolver_residual_floor_ev == pytest.approx(1.0e-8)
    assert analysis.effective_eigensolver_residual_floor_ev > (
        np.finfo(np.float64).eps * analysis.fock_energy_scale_ev
    )
    small_gap = fock.copy()
    small_gap[:, :, 0] = np.diag([0.0, 5.0e-9])
    below_floor = analyze_vituri2024_global_aufbau_boundary(
        small_gap,
        total_occupied=1,
        minimum_gap_to_eigensolver_residual_ratio=1.0,
    )
    assert below_floor.boundary_gap_ev < (
        below_floor.effective_eigensolver_residual_floor_ev
    )
    assert not below_floor.closed_global_aufbau_boundary
    with pytest.raises(ValueError, match=">=1"):
        analyze_vituri2024_global_aufbau_boundary(
            fock,
            total_occupied=2,
            minimum_gap_to_eigensolver_residual_ratio=0.5,
        )


def test_initial_fock_boundary_scan_is_branch_conditioned_and_fail_closed() -> None:
    target = Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=2)
    choice = Vituri2024InitialFockBoundaryScanChoice(
        mesh_size=3,
        target_holes_per_valley=2,
        scan_min_holes_per_valley=1,
        scan_max_holes_per_valley=4,
    )
    assert choice.target_axial_cutoff_a0 == target.axial_k_cutoff_a0
    assert choice.initial_branch_rng_used is False
    selection = scan_vituri2024_initial_fock_aufbau_boundaries(choice)
    assert tuple(record.holes_per_valley for record in selection.records) == (1, 2, 3, 4)
    assert selection.selected.holes_per_valley == 2
    assert selection.target_record == selection.selected
    assert selection.target_admitted is True
    assert selection.fallback_used is False
    assert selection.selected.analysis.closed_global_aufbau_boundary
    assert selection.selected.initial_branch_mode == "half_metal_sz_plus"
    assert selection.selected.initial_branch_rng_used is False
    assert type(selection.selected.analysis.closed_global_aufbau_boundary) is bool
    assert (
        selection.selected.analysis.gap_to_effective_eigensolver_residual_ratio
        >= 1.0e6
    )
    assert selection.selection_key == (
        0.0,
        0.0,
        -selection.selected.analysis.boundary_gap_ev,
        2,
    )
    assert selection.branch_conditioned_regulator_admission_only is True
    assert selection.scf_stationarity_established is False
    assert selection.finite_domain_cutoff_converged is False
    assert selection.global_ground_state_proved is False
    assert all(
        record.spec_fingerprint
        == Vituri2024CartesianHFSpec(
            mesh_size=3, holes_per_valley=record.holes_per_valley
        ).fingerprint
        for record in selection.records
    )
    selection.validate_live_state(recompute_branch_bindings=True)
    bad_records = list(selection.records)
    bad_records[0] = replace(bad_records[0], initial_fock_sha256="0" * 64)
    with pytest.raises(ValueError, match="binding fingerprint drifted"):
        replace(selection, records=tuple(bad_records))
    with pytest.raises(TypeError, match="exact tuple"):
        replace(selection, records=list(selection.records))  # type: ignore[arg-type]


def test_initial_fock_boundary_scan_rejects_live_drift_and_no_candidate() -> None:
    with pytest.raises(ValueError, match=">=1"):
        Vituri2024InitialFockBoundaryScanChoice(
            mesh_size=3,
            target_holes_per_valley=2,
            scan_min_holes_per_valley=1,
            scan_max_holes_per_valley=4,
            minimum_gap_to_eigensolver_residual_ratio=0.5,
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        Vituri2024InitialFockBoundaryScanChoice(
            mesh_size=3,
            target_holes_per_valley=2,
            scan_min_holes_per_valley=1,
            scan_max_holes_per_valley=4,
            initial_branch_mode="half_metal_sx",  # type: ignore[call-arg]
        )
    choice = Vituri2024InitialFockBoundaryScanChoice(
        mesh_size=3,
        target_holes_per_valley=2,
        scan_min_holes_per_valley=1,
        scan_max_holes_per_valley=4,
    )
    object.__setattr__(choice, "target_axial_cutoff_a0", 0.3)
    with pytest.raises(ValueError, match="choice drifted"):
        scan_vituri2024_initial_fock_aufbau_boundaries(choice)
    blocked_fallback = Vituri2024InitialFockBoundaryScanChoice(
        mesh_size=3,
        target_holes_per_valley=3,
        scan_min_holes_per_valley=2,
        scan_max_holes_per_valley=4,
        minimum_gap_to_eigensolver_residual_ratio=5.0e13,
    )
    with pytest.raises(ValueError, match="fallback is disabled"):
        scan_vituri2024_initial_fock_aufbau_boundaries(blocked_fallback)
    fallback = Vituri2024InitialFockBoundaryScanChoice(
        mesh_size=3,
        target_holes_per_valley=3,
        scan_min_holes_per_valley=2,
        scan_max_holes_per_valley=4,
        minimum_gap_to_eigensolver_residual_ratio=5.0e13,
        allow_fallback=True,
    )
    fallback_selection = scan_vituri2024_initial_fock_aufbau_boundaries(fallback)
    target_record = next(
        record
        for record in fallback_selection.records
        if record.holes_per_valley == fallback.target_holes_per_valley
    )
    assert not target_record.analysis.closed_global_aufbau_boundary
    assert fallback_selection.target_record == target_record
    assert fallback_selection.target_admitted is False
    assert fallback_selection.fallback_used is True
    assert fallback_selection.selected.holes_per_valley in (2, 4)
    assert fallback_selection.selected.analysis.closed_global_aufbau_boundary

    impossible = Vituri2024InitialFockBoundaryScanChoice(
        mesh_size=3,
        target_holes_per_valley=2,
        scan_min_holes_per_valley=1,
        scan_max_holes_per_valley=4,
        minimum_gap_to_eigensolver_residual_ratio=1.0e30,
    )
    with pytest.raises(ValueError, match="no admitted candidate"):
        scan_vituri2024_initial_fock_aufbau_boundaries(impossible)


def test_prepared_h0_and_functional_share_mesh_gauge_and_active_band() -> None:
    prepared = _prepared()
    assert prepared.authority == VITURI2024_HF_SCF_AUTHORITY
    assert prepared.functional.nk == prepared.spec.nk
    assert np.array_equal(prepared.functional.ordered_mesh, prepared.ordered_mesh)
    assert np.array_equal(
        prepared.functional.active_band_states, prepared.active_band_states
    )
    for flavor, valley_index in ((0, 0), (1, 0), (2, 1), (3, 1)):
        assert np.array_equal(
            prepared.h0_native[flavor, flavor, :],
            prepared.active_band_energies_by_valley[valley_index],
        )
    off_diagonal = prepared.h0_native.copy()
    for index in range(4):
        off_diagonal[index, index, :] = 0.0
    assert np.count_nonzero(off_diagonal) == 0
    assert prepared.minimum_lower_gap_ev > 0.0
    assert prepared.minimum_upper_gap_ev > 0.0
    assert prepared.source_closure_established is False
    assert prepared.production_ready is False
    with pytest.raises(ValueError, match="(semantic binding|fingerprint) mismatch"):
        type(prepared)(
            spec=Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=2),
            ordered_mesh=prepared.ordered_mesh,
            integer_mesh_labels=prepared.integer_mesh_labels,
            active_band_states=prepared.active_band_states,
            active_band_energies_by_valley=prepared.active_band_energies_by_valley,
            h0_native=prepared.h0_native,
            functional=prepared.functional,
            fixed_density_scf_choice=prepared.fixed_density_scf_choice,
            minimum_lower_gap_ev=prepared.minimum_lower_gap_ev,
            minimum_upper_gap_ev=prepared.minimum_upper_gap_ev,
            fingerprint=prepared.fingerprint,
        )


def test_global_aufbau_has_exact_rank_and_native_orientation() -> None:
    prepared = _prepared()
    problem = make_vituri2024_hf_problem(prepared)
    state = make_vituri2024_hf_state(prepared)
    problem.initializer(state, init_mode="half_metal_sy", seed=404)
    interaction = problem.kernel.interaction_builder(state.density)
    update = problem.kernel.density_builder(state.h0 + interaction)
    assert isinstance(update, DensityUpdateResult)
    conventional = vituri2024_native_density_to_conventional_k_diagonal(
        update.density
    )
    rank = sum(
        np.trace(conventional[:, :, momentum]).real
        for momentum in range(prepared.spec.nk)
    )
    assert rank == pytest.approx(prepared.spec.total_electrons, abs=1.0e-12)
    assert update.observables["occupied_count"] == prepared.spec.total_electrons
    assert np.max(
        np.abs(update.density - update.density.swapaxes(0, 1).conj())
    ) < 1.0e-13
    assert np.isfinite(update.mu)
    assert update.observables["finite_size_fermi_gap_ev"] > 1.0e-12


def test_engine_energy_callback_matches_same_scalar_functional() -> None:
    prepared = _prepared()
    problem = make_vituri2024_hf_problem(prepared)
    state = make_vituri2024_hf_state(prepared)
    problem.initializer(state, init_mode="half_metal_sy", seed=404)
    interaction = prepared.functional.interaction_action(state.density)
    engine_energy = problem.kernel.energy_functional(
        interaction, state.h0, state.density
    )
    scalar_energy = prepared.functional.energy(state.density)
    assert engine_energy == pytest.approx(scalar_energy, abs=2.0e-12)
    direction = problem.kernel.density_builder(state.h0 + interaction).density - state.density
    assert np.max(
        np.abs(
            problem.kernel.oda_delta_interaction_builder(direction)
            - prepared.functional.fock_derivative(state.density, direction)
        )
    ) < 2.0e-12


def test_reduced_half_metal_seed_runs_through_generic_oda_and_is_authority_limited() -> None:
    prepared = _prepared()
    result = run_vituri2024_hf_seed(
        prepared,
        seed_mode="half_metal_sz_plus",
        seed=101,
        max_iter=80,
        oda_stall_threshold=1.0e-12,
    )
    assert result.run.converged
    assert result.run.exit_reason == "converged"
    assert result.diagnostics.total_holes == pytest.approx(
        prepared.spec.total_holes, abs=1.0e-7
    )
    assert result.diagnostics.projector_idempotency_residual < 1.0e-8
    assert result.diagnostics.stationary
    assert result.diagnostics.valid_homogeneous_half_metal_candidate
    assert result.diagnostics.intervalley_incoherent
    assert np.isfinite(result.final_energy_ev)
    assert result.production_ready is False
    assert result.paper_reproduction_verified is False


def test_spec_and_fixed_density_choice_reject_live_mutation() -> None:
    prepared = _prepared()
    object.__setattr__(prepared.spec, "precision", 1.0)
    with pytest.raises(ValueError, match="spec live state drifted"):
        make_vituri2024_hf_problem(prepared)
    prepared = _prepared()
    object.__setattr__(
        prepared.fixed_density_scf_choice,
        "absolute_paper_energy_authorized",
        True,
    )
    with pytest.raises(ValueError, match="authority was inflated"):
        make_vituri2024_hf_problem(prepared)


def test_intervalley_coherence_diagnostic_cannot_cancel_between_momenta() -> None:
    prepared = _prepared()
    state = make_vituri2024_hf_state(prepared)
    conventional = np.repeat(
        np.eye(4, dtype=np.complex128)[:, :, None], prepared.spec.nk, axis=2
    )
    amplitude = 0.1
    conventional[0, 2, 0] = amplitude
    conventional[2, 0, 0] = amplitude
    conventional[0, 2, 1] = -amplitude
    conventional[2, 0, 1] = -amplitude
    state.density[:, :, :] = conventional.swapaxes(0, 1)
    state.hamiltonian[:, :, :] = state.h0
    state.energies[:, :] = np.arange(4 * prepared.spec.nk).reshape(
        (4, prepared.spec.nk), order="C"
    )
    state.diagnostics["final_raw_norm"] = 0.0
    fake_run = hf_scf.HartreeFockRun(
        state=state,
        iter_energy=np.asarray([0.0]),
        iter_err=np.asarray([0.0]),
        iter_oda=np.asarray([1.0]),
        init_mode="ivc_x",
        seed=1,
        converged=True,
        exit_reason="converged",
    )
    diagnostic = hf_scf.diagnose_vituri2024_half_metal(prepared, fake_run)
    assert np.isclose(np.sum(conventional[0, 2, :]), 0.0)
    assert diagnostic.intervalley_coherence_frobenius > 0.0
    assert diagnostic.intervalley_coherence_max_per_k == pytest.approx(amplitude)
    assert not diagnostic.intervalley_incoherent
    assert not diagnostic.valid_homogeneous_half_metal_candidate


def test_spec_rejects_nonphysical_integer_arithmetic() -> None:
    with pytest.raises(ValueError, match="odd integer"):
        Vituri2024CartesianHFSpec(mesh_size=4, holes_per_valley=1)
    with pytest.raises(ValueError, match="1<=H_v<=Nk"):
        Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=10)
    branch_specific = prepare_vituri2024_homogeneous_hf(
        Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=5)
    )
    branch_problem = make_vituri2024_hf_problem(branch_specific)
    branch_state = make_vituri2024_hf_state(branch_specific)
    branch_problem.initializer(branch_state, init_mode="half_metal_sz_plus", seed=0)
    with pytest.raises(ValueError, match="IVC seed cannot realize"):
        branch_problem.initializer(branch_state, init_mode="ivc_x", seed=0)
    with pytest.raises(TypeError, match="integer"):
        Vituri2024CartesianHFSpec(mesh_size=True, holes_per_valley=1)
