"""Reduced faithful tests for the independent Vituri homogeneous HF source."""

from __future__ import annotations

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
    build_vituri2024_cartesian_mesh,
    make_vituri2024_hf_problem,
    make_vituri2024_hf_state,
    prepare_vituri2024_homogeneous_hf,
    run_vituri2024_hf_seed,
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
    with pytest.raises(ValueError, match="incompatible"):
        Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=5)
    with pytest.raises(TypeError, match="integer"):
        Vituri2024CartesianHFSpec(mesh_size=True, holes_per_valley=1)
