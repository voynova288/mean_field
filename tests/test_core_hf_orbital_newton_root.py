from __future__ import annotations

from collections.abc import Callable
import inspect

import numpy as np
import pytest

from mean_field.core.hf.orbital_trust_region import FixedRankHFEvaluation
from mean_field.core.hf.orbital_newton_root import (
    OrbitalNewtonRootOptions,
    run_orbital_newton_root as _run_orbital_newton_root,
)
from mean_field.core.hf.zero_temperature_stability import build_zero_temperature_orbital_hessian
from mean_field.core.hf.orbital_newton_root import _probe_self_adjointness


def _accept_all(
    _projector_value: np.ndarray, _evaluation: FixedRankHFEvaluation
) -> bool:
    return True


def run_orbital_newton_root(*args: object, **kwargs: object):
    kwargs.setdefault("trial_validator", _accept_all)
    return _run_orbital_newton_root(*args, **kwargs)  # type: ignore[arg-type]


def _projector(occupied: np.ndarray) -> np.ndarray:
    vector = np.asarray(occupied, dtype=np.complex128)
    vector = vector / np.linalg.norm(vector)
    return np.outer(vector, vector.conj())[:, :, None]


def _constant_h_evaluator(
    hamiltonian: np.ndarray,
    *,
    seen: list[np.ndarray] | None = None,
) -> Callable[[np.ndarray], FixedRankHFEvaluation]:
    hvalue = np.asarray(hamiltonian, dtype=np.complex128)

    def evaluate(projector: np.ndarray) -> FixedRankHFEvaluation:
        if seen is not None:
            seen.append(projector.copy())
        return FixedRankHFEvaluation(
            energy=float(np.trace(hvalue[:, :, 0] @ projector[:, :, 0]).real),
            hamiltonian=hvalue,
            hamiltonian_response=lambda tangent: np.zeros_like(tangent),
        )

    return evaluate


def _tight_options(**overrides: object) -> OrbitalNewtonRootOptions:
    values: dict[str, object] = {
        "residual_max_tolerance_ev": 1.0e-12,
        "residual_rms_tolerance_ev": 1.0e-12,
        "step_radius": 0.5,
        "max_outer_iterations": 20,
    }
    values.update(overrides)
    return OrbitalNewtonRootOptions(**values)  # type: ignore[arg-type]


def test_validator_is_required_and_initially_stationary_invalid_point_fails() -> None:
    hamiltonian = np.diag([-1.0, 1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([1.0, 0.0])
    evaluate = _constant_h_evaluator(hamiltonian)

    with pytest.raises(TypeError):
        _run_orbital_newton_root(
            initial,
            evaluate,
            occupied_per_k=1,
            options=_tight_options(),
        )
    with pytest.raises(ValueError, match="explicit callable"):
        _run_orbital_newton_root(
            initial,
            evaluate,
            occupied_per_k=1,
            trial_validator=None,  # type: ignore[arg-type]
            options=_tight_options(),
        )

    invalid_hessian_validator_calls = 0
    unauthorized_response_calls = 0

    def invalid_hessian_validator(
        _projector_value: np.ndarray, _evaluation: FixedRankHFEvaluation
    ) -> bool:
        nonlocal invalid_hessian_validator_calls
        invalid_hessian_validator_calls += 1
        return False

    def unauthorized_response(tangent: np.ndarray) -> np.ndarray:
        nonlocal unauthorized_response_calls
        unauthorized_response_calls += 1
        raise AssertionError("initial rejection must not construct a Hessian")

    rejected = _run_orbital_newton_root(
        initial,
        lambda _projector_value: FixedRankHFEvaluation(
            energy=0.0,
            # Deliberately incompatible with the projector: reaching Hessian
            # construction would raise instead of returning the receipt.
            hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
            hamiltonian_response=unauthorized_response,
        ),
        occupied_per_k=1,
        trial_validator=invalid_hessian_validator,
        options=_tight_options(),
    )
    assert invalid_hessian_validator_calls == 1
    assert unauthorized_response_calls == 0
    assert rejected.exit_reason == "initial_validator_rejected"
    assert rejected.history == ()
    assert rejected.hessian_actions == 0
    assert rejected.residual.size == 0
    assert np.isnan(rejected.residual_max_abs_ev)

    def raising_validator(
        _projector_value: np.ndarray, _evaluation: FixedRankHFEvaluation
    ) -> bool:
        raise RuntimeError("initial validation unavailable")

    failed = _run_orbital_newton_root(
        initial,
        lambda _projector_value: FixedRankHFEvaluation(
            energy=0.0,
            hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
            hamiltonian_response=unauthorized_response,
        ),
        occupied_per_k=1,
        trial_validator=raising_validator,
        options=_tight_options(),
    )
    assert unauthorized_response_calls == 0
    assert failed.exit_reason == "initial_validator_failed"
    assert failed.initial_validator_accepted is None
    assert failed.history == ()
    assert failed.hessian_actions == 0

    run = _run_orbital_newton_root(
        initial,
        evaluate,
        occupied_per_k=1,
        trial_validator=lambda _point, _evaluation: False,
        options=_tight_options(),
    )

    assert not run.converged
    assert run.exit_reason == "initial_validator_rejected"
    assert run.initial_validator_accepted is False
    assert run.final_validator_accepted is None
    assert np.isnan(run.residual_max_abs_ev)


def test_final_validator_can_reject_a_numerically_stationary_point() -> None:
    hamiltonian = np.diag([-1.0, 1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([1.0, 0.0])
    calls = 0

    def validator(
        _projector_value: np.ndarray, _evaluation: FixedRankHFEvaluation
    ) -> bool:
        nonlocal calls
        calls += 1
        return calls == 1

    run = _run_orbital_newton_root(
        initial,
        _constant_h_evaluator(hamiltonian),
        occupied_per_k=1,
        trial_validator=validator,
        options=_tight_options(),
    )

    assert calls == 2
    assert not run.converged
    assert run.exit_reason == "final_validator_rejected"
    assert run.initial_validator_accepted is True
    assert run.final_validator_accepted is False


def test_final_validator_exception_fails_closed() -> None:
    hamiltonian = np.diag([-1.0, 1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([1.0, 0.0])
    calls = 0

    def validator(
        _projector_value: np.ndarray, _evaluation: FixedRankHFEvaluation
    ) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("final validation failed")
        return True

    run = _run_orbital_newton_root(
        initial,
        _constant_h_evaluator(hamiltonian),
        occupied_per_k=1,
        trial_validator=validator,
        options=_tight_options(),
    )

    assert calls == 2
    assert not run.converged
    assert run.exit_reason == "final_validator_failed"
    assert run.initial_validator_accepted is True
    assert run.final_validator_accepted is None


def test_indefinite_saddle_root_converges_even_though_energy_increases() -> None:
    # The target occupied level lies between two virtual levels, so its
    # orbital Hessian has both signs.  Approaching it from the lower level
    # raises energy and therefore distinguishes root globalization from
    # minimization.
    hamiltonian = np.diag([0.0, -1.0, 1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([np.cos(0.2), np.sin(0.2), 0.0])
    evaluate = _constant_h_evaluator(hamiltonian)

    run = run_orbital_newton_root(
        initial,
        evaluate,
        occupied_per_k=1,
        options=_tight_options(),
    )

    assert run.converged
    assert run.exit_reason == "converged"
    assert run.initial_validator_accepted is True
    assert run.final_validator_accepted is True
    assert run.evaluation.energy > evaluate(initial).energy
    assert any(
        trial.accepted and trial.energy_trial > trial.energy_before
        for trial in run.trial_history
    )
    np.testing.assert_allclose(
        run.physical_projector[:, :, 0], np.diag([1.0, 0.0, 0.0]), atol=1.0e-11
    )

    final_hessian = build_zero_temperature_orbital_hessian(
        run.physical_projector,
        run.evaluation.hamiltonian,
        run.evaluation.hamiltonian_response,
        occupied_per_k=1,
        tangent_basis=run.tangent_basis,
    )
    dense = np.column_stack(
        [
            final_hessian.energy_hessian_action(np.eye(4)[:, index])
            for index in range(4)
        ]
    )
    eigenvalues = np.linalg.eigvalsh(dense)
    assert eigenvalues[0] < 0.0 < eigenvalues[-1]
    for trial in run.trial_history:
        if trial.accepted:
            assert trial.validator_accepted
            assert trial.sufficient_decrease
            assert trial.residual_merit_trial is not None
            assert trial.residual_merit_trial < trial.residual_merit_before


def test_nonzero_exact_hamiltonian_response_matches_residual_jacobian() -> None:
    h0 = np.diag([0.0, -1.0, 1.0]).astype(np.complex128)[:, :, None]
    coupling = 3.7
    projector = _projector([np.cos(0.2), np.sin(0.2), 0.0])
    hamiltonian = h0 + coupling * projector
    hessian = build_zero_temperature_orbital_hessian(
        projector,
        hamiltonian,
        lambda tangent: coupling * tangent,
        occupied_per_k=1,
    )
    direction = np.asarray(
        [[[0.3 + 0.2j]], [[-0.4 + 0.1j]]], dtype=np.complex128
    )
    analytic = hessian.gradient_jacobian_action(direction)
    packed_direction = hessian.frame.pack_weighted_complex(direction)

    def residual_at(amplitude: float) -> np.ndarray:
        basis = hessian.frame.unitary_basis(
            packed_direction, amplitude=amplitude
        )
        occupied = basis[:, :1, 0]
        trial_projector = (occupied @ occupied.conj().T)[:, :, None]
        trial_hamiltonian = h0 + coupling * trial_projector
        rotated = (
            basis[:, :, 0].conj().T
            @ trial_hamiltonian[:, :, 0]
            @ basis[:, :, 0]
        )
        return rotated[1:, :1, None]

    epsilon = 2.0e-6
    finite_difference = (
        residual_at(epsilon) - residual_at(-epsilon)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-10, atol=2.0e-10)


def test_self_adjointness_hybrid_bound_is_evaluated_per_probe_pair() -> None:
    matrix = np.asarray([[1000.0, 1.0], [0.0, 0.001]])
    absolute_tolerance = 0.1
    relative_tolerance = 0.0031622776601683794
    report = _probe_self_adjointness(
        lambda vector: matrix @ vector,
        size=2,
        probe_count=5,
        seed=19,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        scale_floor=1.0e-3,
    )

    # The independent maxima violate their respective tolerances on different
    # pairs.  Every individual pair nevertheless satisfies abs + rel*scale.
    assert report.maximum_absolute_defect > absolute_tolerance
    assert report.maximum_relative_defect > relative_tolerance
    assert report.maximum_hybrid_ratio <= 1.0


def test_non_self_adjoint_response_fails_before_minres() -> None:
    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([np.cos(0.2), np.sin(0.2)])

    def evaluate(projector: np.ndarray) -> FixedRankHFEvaluation:
        def non_self_adjoint_response(tangent: np.ndarray) -> np.ndarray:
            output = np.zeros_like(tangent)
            value = float(np.asarray(tangent)[1, 0, 0].imag)
            output[1, 0, 0] = value
            output[0, 1, 0] = value
            return output

        return FixedRankHFEvaluation(
            energy=float(
                np.trace(hamiltonian[:, :, 0] @ projector[:, :, 0]).real
            ),
            hamiltonian=hamiltonian,
            hamiltonian_response=non_self_adjoint_response,
        )

    options = _tight_options(
        self_adjoint_probe_count=5,
        self_adjoint_seed=19,
        self_adjoint_absolute_tolerance_ev=1.0e-13,
        self_adjoint_relative_tolerance=1.0e-13,
    )
    run = run_orbital_newton_root(
        initial,
        evaluate,
        occupied_per_k=1,
        options=options,
    )

    assert not run.converged
    assert run.exit_reason == "jacobian_not_self_adjoint"
    assert run.hessian_actions == options.self_adjoint_probe_count
    assert len(run.history) == 1
    iteration = run.history[0]
    assert iteration.minres_iterations == 0
    assert iteration.minres_info == -1
    assert iteration.self_adjoint_absolute_defect_ev is not None
    assert iteration.self_adjoint_absolute_defect_ev > (
        options.self_adjoint_absolute_tolerance_ev
    )
    assert iteration.self_adjoint_relative_defect is not None
    assert iteration.self_adjoint_maximum_hybrid_ratio is not None
    assert iteration.self_adjoint_maximum_hybrid_ratio > 1.0
    assert run.self_adjoint_absolute_defect_ev == pytest.approx(
        iteration.self_adjoint_absolute_defect_ev
    )
    assert run.self_adjoint_relative_defect == pytest.approx(
        iteration.self_adjoint_relative_defect
    )
    assert run.self_adjoint_maximum_hybrid_ratio == pytest.approx(
        iteration.self_adjoint_maximum_hybrid_ratio
    )


def test_validator_rejection_backtracks_from_same_recentered_chart() -> None:
    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([np.cos(0.2), np.sin(0.2)])
    validator_calls = 0

    def validator(
        _projector_value: np.ndarray, _evaluation: FixedRankHFEvaluation
    ) -> bool:
        nonlocal validator_calls
        validator_calls += 1
        return validator_calls != 2

    run = run_orbital_newton_root(
        initial,
        _constant_h_evaluator(hamiltonian),
        occupied_per_k=1,
        trial_validator=validator,
        options=_tight_options(backtrack_factor=0.5),
    )

    assert run.converged
    first, second = run.trial_history[:2]
    assert first.iteration == second.iteration == 1
    assert not first.validator_accepted
    assert not first.accepted
    assert second.validator_accepted and second.accepted
    assert second.step_scale == pytest.approx(0.5 * first.step_scale)
    assert second.step_norm == pytest.approx(0.5 * first.step_norm)
    assert run.history[0].backtracks == 1
    assert run.history[0].accepted


def test_hard_budgets_have_deterministic_fail_closed_exit_reasons() -> None:
    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([np.cos(0.2), np.sin(0.2)])
    evaluate = _constant_h_evaluator(hamiltonian)

    cases = (
        (
            _tight_options(max_outer_iterations=0),
            "maximum_outer_iterations",
            (0, 0, 1),
        ),
        (
            _tight_options(max_projector_evaluations=1),
            "projector_evaluation_budget_exhausted",
            (0, 0, 1),
        ),
        (
            _tight_options(max_hessian_actions=0),
            "hessian_action_budget_exhausted",
            (0, 0, 1),
        ),
    )
    for options, reason, expected_counts in cases:
        run = run_orbital_newton_root(
            initial,
            evaluate,
            occupied_per_k=1,
            options=options,
        )
        assert not run.converged
        assert run.exit_reason == reason
        assert (
            run.outer_iterations,
            run.hessian_actions,
            run.projector_evaluations,
        ) == expected_counts


def test_hessian_action_budget_stops_minres_without_overshoot() -> None:
    hamiltonian = np.diag([-2.0, 0.3, 3.0]).astype(np.complex128)[:, :, None]
    initial = _projector([0.8, 0.5, np.sqrt(0.11)])
    run = run_orbital_newton_root(
        initial,
        _constant_h_evaluator(hamiltonian),
        occupied_per_k=1,
        options=_tight_options(max_hessian_actions=1),
    )

    assert not run.converged
    assert run.exit_reason == "hessian_action_budget_exhausted"
    assert run.hessian_actions == 1
    assert run.projector_evaluations == 1
    assert len(run.history) == 1
    assert not run.history[0].accepted


def test_self_adjoint_probes_are_budgeted_before_minres() -> None:
    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([np.cos(0.2), np.sin(0.2)])
    options = _tight_options(
        self_adjoint_probe_count=3,
        max_hessian_actions=3,
    )

    run = run_orbital_newton_root(
        initial,
        _constant_h_evaluator(hamiltonian),
        occupied_per_k=1,
        options=options,
    )

    assert run.exit_reason == "hessian_action_budget_exhausted"
    assert run.hessian_actions == options.self_adjoint_probe_count
    assert run.history[0].minres_iterations == 0
    assert run.history[0].self_adjoint_absolute_defect_ev is not None
    assert run.history[0].self_adjoint_relative_defect is not None
    assert run.history[0].self_adjoint_maximum_hybrid_ratio is not None


def test_both_complex_residual_max_and_rms_are_required_for_convergence() -> None:
    hamiltonian = np.diag([0.0, 1.0, 2.0]).astype(np.complex128)[:, :, None]
    initial = _projector([np.cos(0.2), np.sin(0.2), 0.0])
    evaluate = _constant_h_evaluator(hamiltonian)

    rms_blocks = run_orbital_newton_root(
        initial,
        evaluate,
        occupied_per_k=1,
        options=_tight_options(
            max_outer_iterations=0,
            residual_max_tolerance_ev=1.0,
            residual_rms_tolerance_ev=1.0e-3,
        ),
    )
    max_blocks = run_orbital_newton_root(
        initial,
        evaluate,
        occupied_per_k=1,
        options=_tight_options(
            max_outer_iterations=0,
            residual_max_tolerance_ev=1.0e-3,
            residual_rms_tolerance_ev=1.0,
        ),
    )

    assert rms_blocks.residual_max_abs_ev < 1.0
    assert rms_blocks.residual_rms_ev > 1.0e-3
    assert not rms_blocks.converged
    assert max_blocks.residual_rms_ev < 1.0
    assert max_blocks.residual_max_abs_ev > 1.0e-3
    assert not max_blocks.converged
    assert rms_blocks.exit_reason == max_blocks.exit_reason == "maximum_outer_iterations"


def test_exact_unitary_trials_preserve_rank_idempotency_and_recenter_history() -> None:
    phase = np.exp(0.37j)
    hamiltonian = np.diag([1.0, -1.0, 2.0]).astype(np.complex128)[:, :, None]
    initial = _projector([np.cos(0.25), phase * np.sin(0.25), 0.0])
    seen: list[np.ndarray] = []

    run = run_orbital_newton_root(
        initial,
        _constant_h_evaluator(hamiltonian, seen=seen),
        occupied_per_k=1,
        options=_tight_options(),
    )

    assert run.converged
    assert len(seen) == run.projector_evaluations
    for projector in seen:
        block = projector[:, :, 0]
        np.testing.assert_allclose(block, block.conj().T, atol=2.0e-14)
        np.testing.assert_allclose(block @ block, block, atol=2.0e-14)
        assert np.trace(block).real == pytest.approx(1.0, abs=2.0e-14)
        assert np.linalg.matrix_rank(block, tol=1.0e-12) == 1
    occupied = run.tangent_basis[:, :1, 0]
    np.testing.assert_allclose(
        occupied @ occupied.conj().T,
        run.physical_projector[:, :, 0],
        atol=2.0e-14,
    )
    accepted_trials = [trial for trial in run.trial_history if trial.accepted]
    assert len(accepted_trials) == len(run.history)
    for previous, following in zip(accepted_trials, run.history[1:]):
        assert following.residual_rms_before_ev == pytest.approx(
            previous.residual_rms_trial_ev
        )
        assert following.residual_max_before_ev == pytest.approx(
            previous.residual_max_trial_ev
        )
    assert run.cumulative_distance == pytest.approx(
        sum(step.accepted_step_norm for step in run.history)
    )


def test_step_radius_and_cumulative_distance_are_hard_limits() -> None:
    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([np.cos(0.3), np.sin(0.3)])
    run = run_orbital_newton_root(
        initial,
        _constant_h_evaluator(hamiltonian),
        occupied_per_k=1,
        options=_tight_options(
            step_radius=0.03,
            maximum_cumulative_distance=0.06,
        ),
    )

    assert not run.converged
    assert run.exit_reason == "cumulative_distance_exhausted"
    assert run.cumulative_distance <= 0.06 + 2.0e-15
    assert all(step.accepted_step_norm <= 0.03 + 2.0e-15 for step in run.history)


def test_optional_coarse_initial_guess_preserves_signed_root_equation() -> None:
    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([np.cos(0.2), np.sin(0.2)])
    initial_calls = 0

    def initial_correction(
        _frame: object,
        rhs: np.ndarray,
        _jacobian_action: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        nonlocal initial_calls
        initial_calls += 1
        np.testing.assert_allclose(
            _jacobian_action(np.zeros_like(rhs)), np.zeros_like(rhs)
        )
        return 0.05 * rhs

    run = run_orbital_newton_root(
        initial,
        _constant_h_evaluator(hamiltonian),
        occupied_per_k=1,
        initial_correction=initial_correction,
        options=_tight_options(),
    )

    assert run.converged
    assert initial_calls == run.outer_iterations
    assert "preconditioner_factory" not in inspect.signature(
        _run_orbital_newton_root
    ).parameters
    assert run.hessian_actions == sum(step.hessian_actions for step in run.history)
    assert all(
        step.self_adjoint_absolute_defect_ev is not None
        and step.self_adjoint_relative_defect is not None
        and step.self_adjoint_maximum_hybrid_ratio is not None
        and step.hessian_actions
        >= _tight_options().self_adjoint_probe_count + 1
        for step in run.history
    )
    np.testing.assert_allclose(
        run.physical_projector[:, :, 0], np.diag([1.0, 0.0]), atol=1.0e-11
    )


def test_external_hook_and_validator_failures_return_receipts() -> None:
    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([np.cos(0.2), np.sin(0.2)])
    evaluate = _constant_h_evaluator(hamiltonian)

    def broken_initial(
        _frame: object,
        _rhs: np.ndarray,
        _action: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        raise RuntimeError("coarse setup failed")

    coarse_failure = run_orbital_newton_root(
        initial,
        evaluate,
        occupied_per_k=1,
        initial_correction=broken_initial,
        options=_tight_options(),
    )
    assert not coarse_failure.converged
    assert coarse_failure.exit_reason == "coarse_hook_failed"
    assert coarse_failure.history[-1].exit_reason == "coarse_hook_failed"

    validator_calls = 0

    def broken_validator(
        _projector_value: np.ndarray, _evaluation: FixedRankHFEvaluation
    ) -> bool:
        nonlocal validator_calls
        validator_calls += 1
        if validator_calls == 1:
            return True
        raise RuntimeError("validator failed")

    validator_failure = run_orbital_newton_root(
        initial,
        evaluate,
        occupied_per_k=1,
        trial_validator=broken_validator,
        options=_tight_options(),
    )
    assert not validator_failure.converged
    assert validator_failure.exit_reason == "trial_validator_failed"
    assert validator_failure.trial_history[-1].failure_reason == "trial_validator_failed"
    assert validator_failure.trial_history[-1].validator_accepted is None


def test_coarse_setup_hessian_actions_obey_the_hard_budget() -> None:
    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([np.cos(0.2), np.sin(0.2)])

    def consume_action_budget(
        _frame: object,
        rhs: np.ndarray,
        action: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        action(np.zeros_like(rhs))
        return np.zeros_like(rhs)

    run = run_orbital_newton_root(
        initial,
        _constant_h_evaluator(hamiltonian),
        occupied_per_k=1,
        initial_correction=consume_action_budget,
        options=_tight_options(max_hessian_actions=1),
    )
    assert not run.converged
    assert run.exit_reason == "hessian_action_budget_exhausted"
    assert run.hessian_actions == 1
    assert run.projector_evaluations == 1


def test_return_arrays_are_read_only_and_evaluator_buffers_do_not_alias() -> None:
    base_hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)[:, :, None]
    shared_hamiltonian = base_hamiltonian.copy()
    initial = _projector([np.cos(0.3), np.sin(0.3)])
    initial_before = initial.copy()
    evaluated_projectors: list[np.ndarray] = []
    evaluator_calls = 0

    def evaluate(projector: np.ndarray) -> FixedRankHFEvaluation:
        nonlocal evaluator_calls
        evaluator_calls += 1
        evaluated_projectors.append(projector)
        if evaluator_calls <= 2:
            shared_hamiltonian[...] = base_hamiltonian
        else:
            shared_hamiltonian[...] = 99.0
        return FixedRankHFEvaluation(
            energy=float(
                np.trace(
                    shared_hamiltonian[:, :, 0] @ projector[:, :, 0]
                ).real
            ),
            hamiltonian=shared_hamiltonian,
            hamiltonian_response=lambda tangent: np.zeros_like(tangent),
            payload={"opaque": shared_hamiltonian},
        )

    validator_calls = 0

    def validator(
        _projector_value: np.ndarray, _evaluation: FixedRankHFEvaluation
    ) -> bool:
        nonlocal validator_calls
        validator_calls += 1
        if validator_calls == 3:
            raise RuntimeError("stop after mutating the rejected evaluator buffer")
        return True

    run = _run_orbital_newton_root(
        initial,
        evaluate,
        occupied_per_k=1,
        trial_validator=validator,
        options=_tight_options(step_radius=0.03),
    )

    assert run.exit_reason == "trial_validator_failed"
    assert evaluator_calls == 3
    np.testing.assert_allclose(run.evaluation.hamiltonian, base_hamiltonian)
    assert not np.shares_memory(run.evaluation.hamiltonian, shared_hamiltonian)
    assert not np.shares_memory(run.physical_projector, initial)
    assert not np.shares_memory(run.physical_projector, evaluated_projectors[1])

    returned_arrays = (
        run.physical_projector,
        run.tangent_basis,
        run.residual,
        run.evaluation.hamiltonian,
    )
    assert all(not array.flags.writeable for array in returned_arrays)
    for array in returned_arrays:
        with pytest.raises(ValueError):
            array.flat[0] = 0.0

    initial[...] = 0.0
    shared_hamiltonian[...] = -123.0
    evaluated_projectors[1][...] = 0.0
    np.testing.assert_allclose(run.evaluation.hamiltonian, base_hamiltonian)
    assert not np.allclose(run.physical_projector, 0.0)
    np.testing.assert_allclose(
        initial_before[:, :, 0] @ initial_before[:, :, 0],
        initial_before[:, :, 0],
    )
    assert isinstance(run.evaluation.payload, dict)
    assert run.evaluation.payload["opaque"] is shared_hamiltonian


def test_histories_are_deterministic() -> None:
    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)[:, :, None]
    initial = _projector([np.cos(0.2), np.sin(0.2)])

    def solve():
        return run_orbital_newton_root(
            initial,
            _constant_h_evaluator(hamiltonian),
            occupied_per_k=1,
            options=_tight_options(),
        )

    first = solve()
    second = solve()
    assert first.history == second.history
    assert first.trial_history == second.trial_history
    assert first.exit_reason == second.exit_reason == "converged"
