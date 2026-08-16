"""Reduced faithful tests for the independent Vituri homogeneous HF source."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from mean_field.core.hf import (
    DensityUpdateResult,
    HartreeFockStepResult,
    StateBoundPreviousDensityBuilder,
    run_hartree_fock_problem,
)
from mean_field.systems.abc_trilayer.vituri2024_hf import (
    vituri2024_native_density_to_conventional_k_diagonal,
)
import mean_field.systems.abc_trilayer.vituri2024_hf_scf as hf_scf
from mean_field.systems.abc_trilayer.vituri2024_hf_scf import (
    VITURI2024_DELTA1_EV,
    VITURI2024_HF_SCF_AUTHORITY,
    VITURI2024_TOTAL_HOLE_DENSITY_CM2,
    Vituri2024CartesianHFSpec,
    Vituri2024ExplicitShellBranchPath,
    analyze_vituri2024_global_aufbau_boundary,
    Vituri2024InitialFockBoundaryScanChoice,
    Vituri2024InitialFockBoundarySelection,
    Vituri2024MaximumOverlapAufbauChoice,
    build_vituri2024_cartesian_mesh,
    make_vituri2024_explicit_shell_branch_choices,
    make_vituri2024_hf_maximum_overlap_problem,
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


def _explicit_branch_step(
    *,
    state,
    update: DensityUpdateResult,
    previous_density: np.ndarray,
    total_hamiltonian: np.ndarray,
    iteration: int,
    oda_lambda: float = 1.0,
) -> HartreeFockStepResult:
    mixed_density = (
        oda_lambda * update.density
        + (1.0 - oda_lambda) * previous_density
    )
    state.density[:, :, :] = mixed_density
    return HartreeFockStepResult(
        iteration=iteration,
        previous_density=previous_density.copy(),
        interaction_h=np.zeros_like(total_hamiltonian),
        total_hamiltonian=total_hamiltonian.copy(),
        density_update=update,
        mixed_density=mixed_density.copy(),
        oda_lambda=oda_lambda,
        norm_raw=0.0,
        norm_mixed=0.0,
        norm_selected=0.0,
        energy=0.0,
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
    target_spec = Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=2)
    continuous_cutoff = target_spec.axial_k_cutoff_a0 * np.sqrt(2.0 / 2.4)
    nearest_choice = Vituri2024InitialFockBoundaryScanChoice(
        mesh_size=3,
        target_holes_per_valley=2,
        scan_min_holes_per_valley=1,
        scan_max_holes_per_valley=4,
        target_holes_policy="nearest_physical_cutoff",
        target_axial_cutoff_a0=continuous_cutoff,
    )
    assert nearest_choice.target_axial_cutoff_a0 == continuous_cutoff
    assert nearest_choice.target_axial_cutoff_a0 != target_spec.axial_k_cutoff_a0
    with pytest.raises(ValueError, match="not the nearest integer"):
        Vituri2024InitialFockBoundaryScanChoice(
            mesh_size=3,
            target_holes_per_valley=3,
            scan_min_holes_per_valley=1,
            scan_max_holes_per_valley=4,
            target_holes_policy="nearest_physical_cutoff",
            target_axial_cutoff_a0=continuous_cutoff,
        )
    with pytest.raises(ValueError, match="derives target cutoff"):
        Vituri2024InitialFockBoundaryScanChoice(
            mesh_size=3,
            target_holes_per_valley=2,
            scan_min_holes_per_valley=1,
            scan_max_holes_per_valley=4,
            target_axial_cutoff_a0=0.2,
        )
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


def test_maximum_overlap_problem_preserves_open_gap_aufbau_exactly() -> None:
    prepared = _prepared()
    baseline = make_vituri2024_hf_problem(prepared)
    state = make_vituri2024_hf_state(prepared)
    continuation = make_vituri2024_hf_maximum_overlap_problem(prepared, state)
    assert isinstance(
        continuation.kernel.density_builder,
        StateBoundPreviousDensityBuilder,
    )
    assert len(continuation.kernel.density_builder.policy_fingerprint) == 64
    with pytest.raises(ValueError, match="different HF state"):
        continuation.initializer(
            make_vituri2024_hf_state(prepared),
            init_mode="half_metal_sz_plus",
            seed=0,
        )
    baseline.initializer(state, init_mode="half_metal_sz_plus", seed=0)
    fock = state.h0 + baseline.kernel.interaction_builder(state.density)
    baseline_update = baseline.kernel.density_builder(fock)
    continued_update = continuation.kernel.density_builder(fock)
    assert np.array_equal(continued_update.density, baseline_update.density)
    assert np.array_equal(continued_update.energies, baseline_update.energies)
    assert continued_update.mu == baseline_update.mu
    assert continued_update.observables == baseline_update.observables
    assert continuation.kernel.final_state_callback is not None
    continuation.kernel.final_state_callback(state, continued_update)


def test_maximum_overlap_problem_runs_open_gap_final_recomputation() -> None:
    prepared = _prepared()
    state = make_vituri2024_hf_state(prepared)
    problem = make_vituri2024_hf_maximum_overlap_problem(prepared, state)
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode="half_metal_sz_plus",
        seed=0,
        max_iter=1,
        oda_stall_threshold=1.0e-8,
        max_oda_lambda=1.0,
    )
    assert run.iterations == 1
    assert np.isfinite(run.state.diagnostics["final_raw_norm"])


def test_maximum_overlap_problem_selects_unique_exact_shell_branch() -> None:
    prepared = _prepared()
    state = make_vituri2024_hf_state(prepared)
    problem = make_vituri2024_hf_maximum_overlap_problem(prepared, state)
    assert isinstance(
        problem.kernel.density_builder,
        StateBoundPreviousDensityBuilder,
    )
    nk = prepared.spec.nk
    values = np.empty((4, nk), dtype=np.float64)
    for momentum in range(nk):
        values[:, momentum] = np.asarray(
            [momentum, 20 + momentum, 40 + momentum, 60 + momentum],
            dtype=np.float64,
        )
    values[3, 6] = 100.0
    values[3, 7] = 100.0
    values[3, 8] = 200.0
    fock = np.zeros((4, 4, nk), dtype=np.complex128)
    for momentum in range(nk):
        fock[:, :, momentum] = np.diag(values[:, momentum])

    previous = np.zeros_like(fock)
    previous[:3, :3, :] = np.eye(3, dtype=np.complex128)[:, :, None]
    for momentum in range(6):
        previous[3, 3, momentum] = 1.0
    previous[3, 3, 6] = 1.0
    state.density[:, :, :] = previous.swapaxes(0, 1)
    update = problem.kernel.density_builder(fock)
    conventional = vituri2024_native_density_to_conventional_k_diagonal(
        update.density
    )
    assert conventional[3, 3, 6] == pytest.approx(1.0)
    assert conventional[3, 3, 7] == pytest.approx(0.0)
    assert update.observables["degenerate_shell_multiplicity"] == 2.0
    assert update.observables["degenerate_shell_selected_rank"] == 1.0
    assert update.observables["maximum_overlap_cutoff_gap"] == pytest.approx(1.0)
    assert max(
        np.max(
            np.abs(
                conventional[:, :, momentum] @ conventional[:, :, momentum]
                - conventional[:, :, momentum]
            )
        )
        for momentum in range(nk)
    ) < 1.0e-13
    particle_number = sum(
        np.trace(conventional[:, :, momentum]).real for momentum in range(nk)
    )
    assert particle_number == pytest.approx(prepared.spec.total_electrons)
    assert max(
        np.max(
            np.abs(
                fock[:, :, momentum] @ conventional[:, :, momentum]
                - conventional[:, :, momentum] @ fock[:, :, momentum]
            )
        )
        for momentum in range(nk)
    ) < 1.0e-13
    linearized_energy = float(
        np.einsum("abk,bak->", fock, conventional, optimize=False).real
    )
    expected_energy = float(
        np.sum(np.sort(values.reshape(-1, order="C"))[: prepared.spec.total_electrons])
    )
    assert linearized_energy == pytest.approx(expected_energy, abs=1.0e-12)


def test_maximum_overlap_problem_rejects_unresolved_overlap_tie() -> None:
    prepared = _prepared()
    choice = Vituri2024MaximumOverlapAufbauChoice()
    state = make_vituri2024_hf_state(prepared)
    problem = make_vituri2024_hf_maximum_overlap_problem(
        prepared, state, choice
    )
    assert isinstance(
        problem.kernel.density_builder,
        StateBoundPreviousDensityBuilder,
    )
    nk = prepared.spec.nk
    values = np.empty((4, nk), dtype=np.float64)
    for momentum in range(nk):
        values[:, momentum] = np.asarray(
            [momentum, 20 + momentum, 40 + momentum, 60 + momentum],
            dtype=np.float64,
        )
    values[3, 6] = values[3, 7] = 100.0
    values[3, 8] = 200.0
    fock = np.zeros((4, 4, nk), dtype=np.complex128)
    for momentum in range(nk):
        fock[:, :, momentum] = np.diag(values[:, momentum])
    previous = np.zeros_like(fock)
    previous[:3, :3, :] = np.eye(3, dtype=np.complex128)[:, :, None]
    for momentum in range(6):
        previous[3, 3, momentum] = 1.0
    previous[3, 3, 6] = 0.5
    previous[3, 3, 7] = 0.5
    state.density[:, :, :] = previous.swapaxes(0, 1)
    with pytest.raises(ValueError, match="branch fanout"):
        problem.kernel.density_builder(fock)
    state.density[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        problem.kernel.density_builder(fock)
    state.density[:, :, :] = previous.swapaxes(0, 1)
    split_fock = fock.copy()
    split_fock[3, 3, 7] += 5.0e-13
    with pytest.raises(ValueError, match="unresolved but not an exact energy tie"):
        problem.kernel.density_builder(split_fock)
    with pytest.raises(ValueError, match="locked"):
        Vituri2024MaximumOverlapAufbauChoice(
            energy_shell_tolerance_ev=5.0e-13
        )
    assert choice.author_exact_numerical_policy is False


def test_explicit_shell_branch_fanout_is_exhaustive_and_trigger_bound() -> None:
    prepared = _prepared()
    nk = prepared.spec.nk
    values = np.empty((4, nk), dtype=np.float64)
    for momentum in range(nk):
        values[:, momentum] = np.asarray(
            [momentum, 20 + momentum, 40 + momentum, 60 + momentum],
            dtype=np.float64,
        )
    values[3, 6] = values[3, 7] = 100.0
    values[3, 8] = 200.0
    fock = np.zeros((4, 4, nk), dtype=np.complex128)
    for momentum in range(nk):
        fock[:, :, momentum] = np.diag(values[:, momentum])
    previous = np.zeros_like(fock)
    previous[:3, :3, :] = np.eye(3, dtype=np.complex128)[:, :, None]
    for momentum in range(6):
        previous[3, 3, momentum] = 1.0
    previous[3, 3, 6] = previous[3, 3, 7] = 0.5
    previous_native = previous.swapaxes(0, 1)
    shell = (3 * nk + 6, 3 * nk + 7)
    branches = make_vituri2024_explicit_shell_branch_choices(
        trigger_fock_sha256=hf_scf._array_sha256(fock),
        trigger_previous_density_sha256=hf_scf._array_sha256(previous_native),
        shell_flat_indices=shell,
        selected_rank=1,
    )
    assert len(branches) == 2
    assert len({branch.branch_set_fingerprint for branch in branches}) == 1
    assert tuple(branch.selected_shell_flat_indices for branch in branches) == (
        (shell[0],),
        (shell[1],),
    )
    six = make_vituri2024_explicit_shell_branch_choices(
        trigger_fock_sha256="1" * 64,
        trigger_previous_density_sha256="2" * 64,
        shell_flat_indices=(10, 11, 12, 13),
        selected_rank=2,
    )
    assert len(six) == 6
    assert tuple(branch.branch_index for branch in six) == tuple(range(6))
    assert len({branch.branch_set_fingerprint for branch in six}) == 1
    second_generation = make_vituri2024_explicit_shell_branch_choices(
        trigger_fock_sha256="3" * 64,
        trigger_previous_density_sha256="4" * 64,
        shell_flat_indices=(20, 21),
        selected_rank=1,
    )[1]
    golden_path = Vituri2024ExplicitShellBranchPath(
        branches=(six[0], second_generation)
    )
    golden_choice = Vituri2024MaximumOverlapAufbauChoice(
        unresolved_overlap_policy="exact_triggered_coordinate_branch_path",
        explicit_branch_path=golden_path,
    )
    assert six[0].fingerprint == (
        "9bf91aa48a5a185fefa032f1b81981fc0a551987dc5e722de9e2c8e4e6cc36c8"
    )
    assert second_generation.fingerprint == (
        "8d06f241f1aa43d065504ea687de9264dd97418e80d5261ffd8d951bf6a5525c"
    )
    assert golden_path.fingerprint == (
        "8ecd0499eddec543802c230da9bdfe665060ce25c2abb96540dcac23a936111d"
    )
    assert golden_choice.fingerprint == (
        "1aa66c7ad65588f650bda791e3965cc594fa73fae7f53c6fce68d1333a9c6976"
    )
    positional_legacy_choice = Vituri2024MaximumOverlapAufbauChoice(
        hf_scf.VITURI2024_DEFAULT_AUFBAU_GAP_TOLERANCE_EV,
        1.0e-12,
        "current_oda_mixed_density",
        "exact_triggered_coordinate_branch",
        branches[0],
        True,
        False,
    )
    assert positional_legacy_choice.explicit_branch is branches[0]
    assert positional_legacy_choice.explicit_branch_path is None
    with pytest.raises(ValueError, match="exhaust"):
        replace(branches[0], branch_count=1)
    with pytest.raises(ValueError, match="canonical"):
        replace(branches[0], branch_index=1)
    with pytest.raises(ValueError, match="declared limit"):
        make_vituri2024_explicit_shell_branch_choices(
            trigger_fock_sha256="1" * 64,
            trigger_previous_density_sha256="2" * 64,
            shell_flat_indices=tuple(range(10)),
            selected_rank=5,
        )

    drift_choice = Vituri2024MaximumOverlapAufbauChoice(
        unresolved_overlap_policy="exact_triggered_coordinate_branch",
        explicit_branch=branches[1],
    )
    drift_state = make_vituri2024_hf_state(prepared)
    drift_state.density[:, :, :] = previous_native
    drift_problem = make_vituri2024_hf_maximum_overlap_problem(
        prepared, drift_state, drift_choice
    )
    object.__setattr__(drift_choice, "overlap_gap_tolerance", 2.0e-12)
    with pytest.raises(ValueError, match="choice fingerprint drifted"):
        drift_problem.kernel.density_builder(fock)

    callback_branch = replace(branches[1])
    callback_choice = Vituri2024MaximumOverlapAufbauChoice(
        unresolved_overlap_policy="exact_triggered_coordinate_branch",
        explicit_branch=callback_branch,
    )
    callback_state = make_vituri2024_hf_state(prepared)
    callback_state.density[:, :, :] = previous_native
    callback_problem = make_vituri2024_hf_maximum_overlap_problem(
        prepared, callback_state, callback_choice
    )
    callback_update = callback_problem.kernel.density_builder(fock)
    callback_step = _explicit_branch_step(
        state=callback_state,
        update=callback_update,
        previous_density=previous_native,
        total_hamiltonian=fock,
        iteration=3,
    )
    object.__setattr__(callback_branch, "fingerprint", "0" * 64)
    assert callback_problem.kernel.step_callback is not None
    with pytest.raises(ValueError, match="branch fingerprint drifted"):
        callback_problem.kernel.step_callback(callback_state, callback_step)

    state = make_vituri2024_hf_state(prepared)
    state.density[:, :, :] = previous_native
    choice = Vituri2024MaximumOverlapAufbauChoice(
        unresolved_overlap_policy="exact_triggered_coordinate_branch",
        explicit_branch=branches[0],
    )
    problem = make_vituri2024_hf_maximum_overlap_problem(
        prepared, state, choice
    )
    assert problem.kernel.density_builder.policy_fingerprint == choice.fingerprint
    drifted_fock = fock.copy()
    drifted_fock[0, 0, 0] += 1.0e-6
    with pytest.raises(ValueError, match="Fock trigger mismatch"):
        problem.kernel.density_builder(drifted_fock)
    update = problem.kernel.density_builder(fock)
    conventional = vituri2024_native_density_to_conventional_k_diagonal(
        update.density
    )
    assert conventional[3, 3, 6] == pytest.approx(1.0)
    assert conventional[3, 3, 7] == pytest.approx(0.0)
    assert update.observables["explicit_coordinate_branch_used"] == 1.0
    assert update.observables["explicit_coordinate_branch_index"] == 0.0
    assert update.observables["explicit_coordinate_branch_generation"] == 0.0
    assert update.observables["explicit_coordinate_branch_path_length"] == 1.0
    with pytest.raises(RuntimeError, match="not audited"):
        problem.kernel.density_builder(fock)
    assert problem.kernel.step_callback is not None
    assert problem.kernel.final_state_callback is not None
    different_state = make_vituri2024_hf_state(prepared)
    different_step = _explicit_branch_step(
        state=different_state,
        update=update,
        previous_density=previous_native,
        total_hamiltonian=fock,
        iteration=3,
    )
    with pytest.raises(RuntimeError, match="different HF state"):
        problem.kernel.step_callback(different_state, different_step)
    forged_update = replace(update, density=update.density.copy())
    forged_step = _explicit_branch_step(
        state=state,
        update=forged_update,
        previous_density=previous_native,
        total_hamiltonian=fock,
        iteration=3,
    )
    with pytest.raises(RuntimeError, match="builder receipt"):
        problem.kernel.step_callback(state, forged_step)
    valid_step = _explicit_branch_step(
        state=state,
        update=update,
        previous_density=previous_native,
        total_hamiltonian=fock,
        iteration=3,
    )
    problem.kernel.step_callback(state, valid_step)
    assert state.diagnostics["explicit_coordinate_branch_used"] == 1.0
    assert state.diagnostics["explicit_coordinate_branch_index"] == 0.0
    assert state.diagnostics["explicit_coordinate_branch_count"] == 2.0
    assert state.diagnostics["explicit_coordinate_branch_iteration"] == 3.0
    assert state.diagnostics["explicit_coordinate_branch_use_count"] == 1.0
    assert state.diagnostics["explicit_coordinate_branch_path_length"] == 1.0

    repeated_update = problem.kernel.density_builder(fock)
    assert repeated_update.observables["explicit_coordinate_branch_used"] == 0.0
    assert np.array_equal(repeated_update.density, update.density)
    state.diagnostics["explicit_coordinate_branch_last_index"] = 99.0
    with pytest.raises(RuntimeError, match="first/last diagnostics drifted"):
        problem.kernel.final_state_callback(state, repeated_update)
    state.diagnostics["explicit_coordinate_branch_last_index"] = 0.0
    problem.kernel.final_state_callback(state, repeated_update)


def test_final_explicit_branch_use_poisoned_after_rejection() -> None:
    prepared = _prepared()
    nk = prepared.spec.nk
    values = np.empty((4, nk), dtype=np.float64)
    for momentum in range(nk):
        values[:, momentum] = np.asarray(
            [momentum, 20 + momentum, 40 + momentum, 60 + momentum],
            dtype=np.float64,
        )
    values[3, 6] = values[3, 7] = 100.0
    values[3, 8] = 200.0
    fock = np.zeros((4, 4, nk), dtype=np.complex128)
    for momentum in range(nk):
        fock[:, :, momentum] = np.diag(values[:, momentum])
    previous = np.zeros_like(fock)
    previous[:3, :3, :] = np.eye(3, dtype=np.complex128)[:, :, None]
    for momentum in range(6):
        previous[3, 3, momentum] = 1.0
    previous[3, 3, 6] = previous[3, 3, 7] = 0.5
    previous_native = previous.swapaxes(0, 1)
    branch = make_vituri2024_explicit_shell_branch_choices(
        trigger_fock_sha256=hf_scf._array_sha256(fock),
        trigger_previous_density_sha256=hf_scf._array_sha256(previous_native),
        shell_flat_indices=(3 * nk + 6, 3 * nk + 7),
        selected_rank=1,
    )[0]
    choice = Vituri2024MaximumOverlapAufbauChoice(
        unresolved_overlap_policy="exact_triggered_coordinate_branch",
        explicit_branch=branch,
    )
    state = make_vituri2024_hf_state(prepared)
    state.density[:, :, :] = previous_native
    problem = make_vituri2024_hf_maximum_overlap_problem(prepared, state, choice)
    update = problem.kernel.density_builder(fock)
    assert problem.kernel.final_state_callback is not None
    with pytest.raises(RuntimeError, match="new branch fanout"):
        problem.kernel.final_state_callback(state, update)
    with pytest.raises(RuntimeError, match="terminally rejected"):
        problem.kernel.density_builder(fock)
    step = _explicit_branch_step(
        state=state,
        update=update,
        previous_density=previous_native,
        total_hamiltonian=fock,
        iteration=3,
    )
    assert problem.kernel.step_callback is not None
    with pytest.raises(RuntimeError, match="terminally rejected"):
        problem.kernel.step_callback(state, step)


def test_exact_triggered_branch_path_consumes_two_generations_in_order() -> None:
    prepared = _prepared()
    nk = prepared.spec.nk

    first_values = np.empty((4, nk), dtype=np.float64)
    for momentum in range(nk):
        first_values[:, momentum] = np.asarray(
            [momentum, 20 + momentum, 40 + momentum, 60 + momentum],
            dtype=np.float64,
        )
    for momentum in range(5, 9):
        first_values[3, momentum] = 100.0
    first_fock = np.zeros((4, 4, nk), dtype=np.complex128)
    for momentum in range(nk):
        first_fock[:, :, momentum] = np.diag(first_values[:, momentum])

    initial = np.zeros_like(first_fock)
    initial[:3, :3, :] = np.eye(3, dtype=np.complex128)[:, :, None]
    for momentum in range(5):
        initial[3, 3, momentum] = 1.0
    for momentum in range(5, 9):
        initial[3, 3, momentum] = 0.5
    initial_native = initial.swapaxes(0, 1)
    first_shell = tuple(3 * nk + momentum for momentum in range(5, 9))
    first_branches = make_vituri2024_explicit_shell_branch_choices(
        trigger_fock_sha256=hf_scf._array_sha256(first_fock),
        trigger_previous_density_sha256=hf_scf._array_sha256(initial_native),
        shell_flat_indices=first_shell,
        selected_rank=2,
    )
    first_branch = first_branches[0]

    temporary_state = make_vituri2024_hf_state(prepared)
    temporary_state.density[:, :, :] = initial_native
    temporary_choice = Vituri2024MaximumOverlapAufbauChoice(
        unresolved_overlap_policy="exact_triggered_coordinate_branch",
        explicit_branch=first_branch,
    )
    temporary_problem = make_vituri2024_hf_maximum_overlap_problem(
        prepared, temporary_state, temporary_choice
    )
    first_update_reference = temporary_problem.kernel.density_builder(first_fock)

    second_values = np.empty((4, nk), dtype=np.float64)
    for momentum in range(nk):
        second_values[:, momentum] = np.asarray(
            [momentum, 20 + momentum, 40 + momentum, 200 + momentum],
            dtype=np.float64,
        )
    for momentum in range(5):
        second_values[3, momentum] = 60.0 + momentum
    second_values[3, 7] = 65.0
    second_values[3, 5] = second_values[3, 6] = 100.0
    second_fock = np.zeros((4, 4, nk), dtype=np.complex128)
    for momentum in range(nk):
        second_fock[:, :, momentum] = np.diag(second_values[:, momentum])
    second_shell = tuple(3 * nk + momentum for momentum in (5, 6))
    second_branches = make_vituri2024_explicit_shell_branch_choices(
        trigger_fock_sha256=hf_scf._array_sha256(second_fock),
        trigger_previous_density_sha256=hf_scf._array_sha256(
            first_update_reference.density
        ),
        shell_flat_indices=second_shell,
        selected_rank=1,
    )
    path = Vituri2024ExplicitShellBranchPath(
        branches=(first_branch, second_branches[0])
    )
    reversed_path = Vituri2024ExplicitShellBranchPath(
        branches=(second_branches[0], first_branch)
    )
    assert len(path.fingerprint) == 64
    assert reversed_path.fingerprint != path.fingerprint
    drifted_path = Vituri2024ExplicitShellBranchPath(branches=path.branches)
    object.__setattr__(drifted_path, "fingerprint", "0" * 64)
    with pytest.raises(ValueError, match="path fingerprint drifted"):
        drifted_path.validate_live_state()
    with pytest.raises(ValueError, match="repeats an exact trigger"):
        Vituri2024ExplicitShellBranchPath(branches=(first_branch, first_branch))
    with pytest.raises(ValueError, match="mutually exclusive"):
        Vituri2024MaximumOverlapAufbauChoice(
            unresolved_overlap_policy="exact_triggered_coordinate_branch_path",
            explicit_branch=first_branch,
            explicit_branch_path=path,
        )

    state = make_vituri2024_hf_state(prepared)
    state.density[:, :, :] = initial_native
    choice = Vituri2024MaximumOverlapAufbauChoice(
        unresolved_overlap_policy="exact_triggered_coordinate_branch_path",
        explicit_branch_path=path,
    )
    problem = make_vituri2024_hf_maximum_overlap_problem(prepared, state, choice)
    assert problem.kernel.step_callback is not None

    first_update = problem.kernel.density_builder(first_fock)
    assert np.array_equal(first_update.density, first_update_reference.density)
    assert first_update.observables["explicit_coordinate_branch_generation"] == 0.0
    first_step = _explicit_branch_step(
        state=state,
        update=first_update,
        previous_density=initial_native,
        total_hamiltonian=first_fock,
        iteration=3,
    )
    state.diagnostics["explicit_coordinate_branch_use_count"] = 1.0
    with pytest.raises(RuntimeError, match="stale or preseeded"):
        problem.kernel.step_callback(state, first_step)
    state.diagnostics.clear()
    problem.kernel.step_callback(state, first_step)
    assert state.diagnostics["explicit_coordinate_branch_use_count"] == 1.0

    drifted_second_fock = second_fock.copy()
    drifted_second_fock[0, 0, 0] += 1.0e-6
    with pytest.raises(ValueError, match="Fock trigger mismatch"):
        problem.kernel.density_builder(drifted_second_fock)

    second_update = problem.kernel.density_builder(second_fock)
    second_conventional = vituri2024_native_density_to_conventional_k_diagonal(
        second_update.density
    )
    assert second_update.observables["explicit_coordinate_branch_generation"] == 1.0
    assert second_update.observables["explicit_coordinate_branch_path_length"] == 2.0
    assert second_conventional[3, 3, 5] == pytest.approx(1.0)
    assert second_conventional[3, 3, 6] == pytest.approx(0.0)
    second_step = _explicit_branch_step(
        state=state,
        update=second_update,
        previous_density=first_update.density,
        total_hamiltonian=second_fock,
        iteration=5,
    )
    problem.kernel.step_callback(state, second_step)
    assert state.diagnostics["explicit_coordinate_branch_use_count"] == 2.0
    assert state.diagnostics["explicit_coordinate_branch_last_generation"] == 1.0
    assert state.diagnostics[
        "explicit_coordinate_branch_generation_0_iteration"
    ] == 3.0
    assert state.diagnostics[
        "explicit_coordinate_branch_generation_1_iteration"
    ] == 5.0
    state.density[:, :, :] = first_update.density
    with pytest.raises(ValueError, match="branch path is exhausted"):
        problem.kernel.density_builder(second_fock)
    with pytest.raises(RuntimeError, match="terminally rejected"):
        problem.kernel.step_callback(state, second_step)


def test_four_state_rank_two_coordinate_fanout_runs_all_six_ground_states() -> None:
    prepared = _prepared()
    nk = prepared.spec.nk
    values = np.empty((4, nk), dtype=np.float64)
    for momentum in range(nk):
        values[:, momentum] = np.asarray(
            [momentum, 20 + momentum, 40 + momentum, 60 + momentum],
            dtype=np.float64,
        )
    for momentum in range(5, 9):
        values[3, momentum] = 100.0
    fock = np.zeros((4, 4, nk), dtype=np.complex128)
    for momentum in range(nk):
        fock[:, :, momentum] = np.diag(values[:, momentum])
    previous = np.zeros_like(fock)
    previous[:3, :3, :] = np.eye(3, dtype=np.complex128)[:, :, None]
    for momentum in range(5):
        previous[3, 3, momentum] = 1.0
    for momentum in range(5, 9):
        previous[3, 3, momentum] = 0.5
    previous_native = previous.swapaxes(0, 1)
    shell = tuple(3 * nk + momentum for momentum in range(5, 9))
    branches = make_vituri2024_explicit_shell_branch_choices(
        trigger_fock_sha256=hf_scf._array_sha256(fock),
        trigger_previous_density_sha256=hf_scf._array_sha256(previous_native),
        shell_flat_indices=shell,
        selected_rank=2,
    )
    density_hashes: set[str] = set()
    linearized_energies: list[float] = []
    for branch in branches:
        state = make_vituri2024_hf_state(prepared)
        state.density[:, :, :] = previous_native
        choice = Vituri2024MaximumOverlapAufbauChoice(
            unresolved_overlap_policy="exact_triggered_coordinate_branch",
            explicit_branch=branch,
        )
        problem = make_vituri2024_hf_maximum_overlap_problem(
            prepared, state, choice
        )
        update = problem.kernel.density_builder(fock)
        conventional = vituri2024_native_density_to_conventional_k_diagonal(
            update.density
        )
        selected = {
            flat_index - 3 * nk for flat_index in branch.selected_shell_flat_indices
        }
        assert {
            momentum
            for momentum in range(5, 9)
            if conventional[3, 3, momentum].real > 0.5
        } == selected
        assert update.observables["explicit_coordinate_branch_used"] == 1.0
        assert update.observables["explicit_coordinate_branch_index"] == float(
            branch.branch_index
        )
        density_hashes.add(hf_scf._array_sha256(update.density))
        linearized_energies.append(
            float(
                np.einsum(
                    "abk,bak->",
                    fock,
                    conventional,
                    optimize=False,
                ).real
            )
        )
    assert len(branches) == len(density_hashes) == 6
    assert max(linearized_energies) - min(linearized_energies) == pytest.approx(
        0.0, abs=1.0e-12
    )


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
