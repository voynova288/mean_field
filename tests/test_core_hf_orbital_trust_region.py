from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf.orbital_trust_region import (
    FixedRankHFEvaluation,
    OrbitalTrustRegionOptions,
    run_orbital_trust_region,
    solve_steihaug_trust_region,
)


def test_steihaug_rejects_zero_gradient_without_curvature_direction() -> None:
    with pytest.raises(ValueError, match="external curvature direction"):
        solve_steihaug_trust_region(
            np.zeros(2),
            lambda vector: np.asarray([-vector[0], vector[1]]),
            radius=0.3,
            maximum_iterations=10,
            residual_tolerance=1.0e-12,
            curvature_tolerance_ev=1.0e-14,
            model_reduction_tolerance_ev=1.0e-16,
        )


def test_steihaug_uses_boundary_on_negative_curvature() -> None:
    gradient = np.asarray([1.0, 0.0])
    hessian = np.diag([-1.0, 2.0])
    result = solve_steihaug_trust_region(
        gradient,
        lambda vector: hessian @ vector,
        radius=0.3,
        maximum_iterations=10,
        residual_tolerance=1.0e-12,
        curvature_tolerance_ev=1.0e-14,
        model_reduction_tolerance_ev=1.0e-16,
    )
    assert result.negative_curvature
    assert result.reached_boundary
    np.testing.assert_allclose(result.step, [-0.3, 0.0], atol=1.0e-14)
    assert result.predicted_reduction > 0.0


def test_orbital_trust_region_minimizes_complex_two_level_projector() -> None:
    hamiltonian = np.diag([-1.0, 1.0]).astype(np.complex128)[:, :, None]
    theta = 0.4
    phase = np.exp(0.3j)
    occupied = np.asarray([np.cos(theta), phase * np.sin(theta)])
    virtual = np.asarray([-phase.conjugate() * np.sin(theta), np.cos(theta)])
    basis = np.column_stack([occupied, virtual]).astype(np.complex128)[:, :, None]
    projector = (occupied[:, None] @ occupied.conj()[None, :])[:, :, None]

    def evaluate(value: np.ndarray) -> FixedRankHFEvaluation:
        energy = float(np.trace(hamiltonian[:, :, 0] @ value[:, :, 0]).real)
        return FixedRankHFEvaluation(
            energy=energy,
            hamiltonian=hamiltonian,
            hamiltonian_response=lambda delta: np.zeros_like(delta),
        )

    initial_energy = evaluate(projector).energy
    run = run_orbital_trust_region(
        projector,
        evaluate,
        occupied_per_k=1,
        initial_basis=basis,
        options=OrbitalTrustRegionOptions(
            max_iterations=30,
            gradient_tolerance_ev=1.0e-10,
            initial_radius=0.2,
            maximum_radius=0.5,
            cg_max_iterations=10,
        ),
    )
    assert run.stationary_candidate
    assert run.exit_reason == "second_order_stationary_candidate"
    assert run.termination.lowest_curvature_ev is not None
    assert run.termination.lowest_curvature_ev > 0.0
    assert not run.physical_convergence_established
    assert run.evaluation.energy < initial_energy
    np.testing.assert_allclose(run.evaluation.energy, -1.0, atol=1.0e-12)
    np.testing.assert_allclose(
        run.physical_projector[:, :, 0], np.diag([1.0, 0.0]), atol=1.0e-10
    )
    assert all(step.energy_after <= step.energy_before for step in run.history)


def test_orbital_trust_region_escapes_zero_gradient_saddle() -> None:
    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)[:, :, None]
    projector = np.diag([1.0, 0.0]).astype(np.complex128)[:, :, None]
    basis = np.eye(2, dtype=np.complex128)[:, :, None]

    def evaluate(value: np.ndarray) -> FixedRankHFEvaluation:
        return FixedRankHFEvaluation(
            energy=float(np.trace(hamiltonian[:, :, 0] @ value[:, :, 0]).real),
            hamiltonian=hamiltonian,
            hamiltonian_response=lambda delta: np.zeros_like(delta),
        )

    run = run_orbital_trust_region(
        projector,
        evaluate,
        occupied_per_k=1,
        initial_basis=basis,
        options=OrbitalTrustRegionOptions(
            max_iterations=30,
            gradient_tolerance_ev=1.0e-10,
            stationarity_max_tolerance_ev=1.0e-10,
            initial_radius=0.2,
            maximum_radius=0.5,
            stationary_curvature_tolerance_ev=1.0e-10,
            stationary_curvature_residual_tolerance_ev=1.0e-10,
        ),
    )
    assert run.stationary_candidate
    assert run.exit_reason == "second_order_stationary_candidate"
    assert run.history[0].cg_exit_reason == "stationary_negative_curvature_boundary"
    assert run.history[0].accepted
    np.testing.assert_allclose(run.evaluation.energy, -1.0, atol=1.0e-12)
    np.testing.assert_allclose(
        run.physical_projector[:, :, 0], np.diag([0.0, 1.0]), atol=1.0e-10
    )
    assert run.termination.lowest_curvature_ev is not None
    assert run.termination.lowest_curvature_ev > 0.0


def test_orbital_trust_region_records_maximum_iteration_termination() -> None:
    hamiltonian = np.diag([-1.0, 1.0]).astype(np.complex128)[:, :, None]
    occupied = np.asarray([np.cos(0.3), np.sin(0.3)], dtype=np.complex128)
    projector = (occupied[:, None] @ occupied.conj()[None, :])[:, :, None]

    def evaluate(value: np.ndarray) -> FixedRankHFEvaluation:
        return FixedRankHFEvaluation(
            energy=float(np.trace(hamiltonian[:, :, 0] @ value[:, :, 0]).real),
            hamiltonian=hamiltonian,
            hamiltonian_response=lambda delta: np.zeros_like(delta),
        )

    run = run_orbital_trust_region(
        projector,
        evaluate,
        occupied_per_k=1,
        options=OrbitalTrustRegionOptions(max_iterations=0),
    )
    assert not run.history
    assert run.exit_reason == "maximum_iterations"
    assert run.termination.iteration == 0
    assert run.termination.gradient_norm_ev > 0.0
    assert run.termination.energy == run.evaluation.energy


@pytest.mark.parametrize(
    "options",
    [
        OrbitalTrustRegionOptions(initial_radius=0.0),
        OrbitalTrustRegionOptions(minimum_radius=0.3, initial_radius=0.2),
        OrbitalTrustRegionOptions(maximum_radius=np.inf),
        OrbitalTrustRegionOptions(cg_relative_tolerance_maximum=1.1),
        OrbitalTrustRegionOptions(stationary_curvature_residual_tolerance_ev=-1.0),
        OrbitalTrustRegionOptions(self_adjoint_absolute_tolerance_ev=np.nan),
        OrbitalTrustRegionOptions(max_iterations=1.5),  # type: ignore[arg-type]
    ],
)
def test_orbital_trust_region_rejects_inconsistent_options(
    options: OrbitalTrustRegionOptions,
) -> None:
    hamiltonian = np.diag([-1.0, 1.0]).astype(np.complex128)[:, :, None]
    projector = np.diag([1.0, 0.0]).astype(np.complex128)[:, :, None]

    def evaluate(value: np.ndarray) -> FixedRankHFEvaluation:
        return FixedRankHFEvaluation(
            energy=float(np.trace(hamiltonian[:, :, 0] @ value[:, :, 0]).real),
            hamiltonian=hamiltonian,
            hamiltonian_response=lambda delta: np.zeros_like(delta),
        )

    with pytest.raises(ValueError, match="options are inconsistent"):
        run_orbital_trust_region(
            projector,
            evaluate,
            occupied_per_k=1,
            options=options,
        )


def test_invalid_trial_contracts_radius_and_terminal_state_is_consistent() -> None:
    hamiltonian = np.diag([-1.0, 1.0]).astype(np.complex128)[:, :, None]
    theta = 0.3
    occupied = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.complex128)
    projector = (occupied[:, None] @ occupied.conj()[None, :])[:, :, None]

    def evaluate(value: np.ndarray) -> FixedRankHFEvaluation:
        return FixedRankHFEvaluation(
            energy=float(np.trace(hamiltonian[:, :, 0] @ value[:, :, 0]).real),
            hamiltonian=hamiltonian,
            hamiltonian_response=lambda delta: np.zeros_like(delta),
        )

    initial_energy = evaluate(projector).energy
    run = run_orbital_trust_region(
        projector,
        evaluate,
        occupied_per_k=1,
        options=OrbitalTrustRegionOptions(
            max_iterations=2,
            initial_radius=0.2,
            minimum_radius=0.1,
            shrink_factor=0.25,
        ),
        trial_validator=lambda trial, evaluation: False,
    )
    assert len(run.history) == 1
    assert not run.history[0].trial_valid
    assert not run.history[0].accepted
    assert run.history[0].trust_radius_after == pytest.approx(0.05)
    assert run.exit_reason == "radius_exhausted"
    assert run.termination.iteration == 1
    assert run.termination.energy == pytest.approx(initial_energy)
    assert run.termination.gradient_norm_ev == pytest.approx(
        run.history[0].gradient_norm_ev
    )
    np.testing.assert_allclose(run.physical_projector, projector, atol=1.0e-14)


def test_stationary_curvature_uses_matrix_free_path_above_dense_threshold() -> None:
    diagonal = np.linspace(-2.0, 2.0, 8)
    hamiltonian = np.diag(diagonal).astype(np.complex128)[:, :, None]
    projector = np.diag([1.0] * 4 + [0.0] * 4).astype(np.complex128)[:, :, None]

    def evaluate(value: np.ndarray) -> FixedRankHFEvaluation:
        return FixedRankHFEvaluation(
            energy=float(np.trace(hamiltonian[:, :, 0] @ value[:, :, 0]).real),
            hamiltonian=hamiltonian,
            hamiltonian_response=lambda delta: np.zeros_like(delta),
        )

    run = run_orbital_trust_region(
        projector,
        evaluate,
        occupied_per_k=4,
        options=OrbitalTrustRegionOptions(
            max_iterations=0,
            stationary_curvature_residual_tolerance_ev=1.0e-10,
        ),
    )
    assert run.exit_reason == "second_order_stationary_candidate"
    assert run.termination.lowest_curvature_ev == pytest.approx(
        2.0 * (diagonal[4] - diagonal[3]), abs=1.0e-10
    )
    assert run.termination.lowest_curvature_residual_ev is not None
    assert run.termination.lowest_curvature_residual_ev < 1.0e-10
