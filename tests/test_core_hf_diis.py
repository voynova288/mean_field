from __future__ import annotations

import numpy as np

from mean_field.core.hf.diis import (
    FixedPointEvaluation,
    PulayDIISOptions,
    PulayOperatorEvaluation,
    run_pulay_diis_fixed_point,
    run_pulay_operator_diis,
)


def _validate_trace_one_hermitian(value: np.ndarray) -> None:
    assert value.shape == (2, 2, 1)
    np.testing.assert_allclose(value, np.swapaxes(value.conj(), 0, 1), atol=1.0e-12)
    np.testing.assert_allclose(np.trace(value[:, :, 0]), 1.0, atol=1.0e-12)


def test_pulay_diis_constant_map_uses_authoritative_final_replay() -> None:
    initial = np.zeros((2, 2, 1), dtype=np.complex128)
    initial[0, 0, 0] = 1.0
    target = np.zeros_like(initial)
    target[1, 1, 0] = 1.0
    calls: list[np.ndarray] = []

    def evaluate(density: np.ndarray) -> FixedPointEvaluation[int]:
        calls.append(np.array(density, copy=True))
        return FixedPointEvaluation(output=target, payload=len(calls))

    run = run_pulay_diis_fixed_point(
        initial,
        evaluate,
        precision=1.0e-12,
        max_iter=5,
        validate_output=_validate_trace_one_hermitian,
        validate_iterate=_validate_trace_one_hermitian,
    )

    assert run.converged
    assert run.exit_reason == "converged"
    np.testing.assert_array_equal(run.density, target)
    assert run.final_residual_norm == 0.0
    assert len(calls) == 3
    assert run.steps[-1].step_kind == "converged"


def test_pulay_diis_solves_period_two_map_without_mixed_norm_false_positive() -> None:
    initial = np.zeros((2, 2, 1), dtype=np.complex128)
    initial[0, 0, 0] = 1.0
    identity = np.eye(2, dtype=np.complex128)[:, :, None]

    def evaluate(density: np.ndarray) -> FixedPointEvaluation[None]:
        return FixedPointEvaluation(output=identity - density, payload=None)

    run = run_pulay_diis_fixed_point(
        initial,
        evaluate,
        precision=1.0e-10,
        max_iter=8,
        options=PulayDIISOptions(
            regularization=1.0e-14,
            max_regularization=1.0e-8,
            trust_radius=10.0,
        ),
        validate_output=_validate_trace_one_hermitian,
        validate_iterate=_validate_trace_one_hermitian,
    )

    assert run.converged
    np.testing.assert_allclose(run.density, 0.5 * identity, atol=1.0e-10)
    assert run.final_residual_norm <= 1.0e-10
    assert any(step.step_kind.startswith("diis") for step in run.steps)


def test_pulay_operator_diis_solves_period_two_map_by_extrapolating_operator() -> None:
    initial = np.zeros((1, 1, 1), dtype=np.complex128)
    calls = 0

    def evaluate(density: np.ndarray) -> PulayOperatorEvaluation[int]:
        nonlocal calls
        calls += 1
        hamiltonian = 1.0 - density
        mapped = np.array(hamiltonian, copy=True)
        return PulayOperatorEvaluation(
            hamiltonian=hamiltonian,
            mapped_density=mapped,
            error=mapped - density,
            payload=calls,
        )

    run = run_pulay_operator_diis(
        initial,
        evaluate,
        lambda hamiltonian: np.array(hamiltonian, copy=True),
        precision=1.0e-10,
        max_iter=8,
        convergence_metric=lambda updated, previous: float(
            np.linalg.norm(updated - previous)
        ),
        options=PulayDIISOptions(
            regularization=1.0e-14,
            max_regularization=1.0e-8,
        ),
    )

    assert run.converged
    np.testing.assert_allclose(run.density, 0.5, atol=1.0e-10)
    assert run.final_residual_norm <= 1.0e-10
    assert any(step.step_kind == "operator_diis" for step in run.steps)
    assert calls >= 4


def test_pulay_operator_diis_matches_independent_complex_commutator_system() -> None:
    sigma_x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sigma_z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    h0 = sigma_x + 0.3 * sigma_y + 0.2 * sigma_z
    coupling = 2.0
    initial = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)[:, :, None]
    mapped_hamiltonians: list[np.ndarray] = []

    def ground_projector(hamiltonian: np.ndarray) -> np.ndarray:
        _, vectors = np.linalg.eigh(hamiltonian[:, :, 0])
        vector = vectors[:, 0]
        return np.outer(vector, vector.conj())[:, :, None]

    def evaluate(density: np.ndarray) -> PulayOperatorEvaluation[None]:
        hamiltonian = h0[:, :, None] + coupling * density
        mapped = ground_projector(hamiltonian)
        error = np.empty_like(hamiltonian)
        error[:, :, 0] = (
            hamiltonian[:, :, 0] @ density[:, :, 0]
            - density[:, :, 0] @ hamiltonian[:, :, 0]
        )
        return PulayOperatorEvaluation(
            hamiltonian=hamiltonian,
            mapped_density=mapped,
            error=error,
            payload=None,
        )

    def map_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
        mapped_hamiltonians.append(np.array(hamiltonian, copy=True))
        return ground_projector(hamiltonian)

    regularization = 1.0e-6
    run_pulay_operator_diis(
        initial,
        evaluate,
        map_hamiltonian,
        precision=1.0e-14,
        max_iter=2,
        options=PulayDIISOptions(
            regularization=regularization,
            max_regularization=regularization,
            coefficient_l1_limit=100.0,
            restart_growth_factor=1000.0,
        ),
    )

    evaluation0 = evaluate(initial)
    density1 = evaluation0.mapped_density
    evaluation1 = evaluate(density1)
    errors = [evaluation0.error, evaluation1.error]
    gram = np.asarray(
        [[float(np.vdot(left, right).real) for right in errors] for left in errors]
    )
    scale = max(float(np.max(np.diag(gram))), 1.0e-300)
    augmented = np.empty((3, 3), dtype=float)
    augmented[:2, :2] = gram / scale + regularization * np.eye(2)
    augmented[:2, 2] = 1.0
    augmented[2, :2] = 1.0
    augmented[2, 2] = 0.0
    coefficients = np.linalg.solve(augmented, np.asarray([0.0, 0.0, 1.0]))[:2]
    expected = (
        coefficients[0] * evaluation0.hamiltonian
        + coefficients[1] * evaluation1.hamiltonian
    )

    assert len(mapped_hamiltonians) == 1
    np.testing.assert_allclose(mapped_hamiltonians[0], expected, atol=1.0e-12)
    np.testing.assert_allclose(np.sum(coefficients), 1.0, atol=1.0e-14)


def test_pulay_diis_restarts_when_coefficients_are_excessive() -> None:
    initial = np.zeros((2, 2, 1), dtype=np.complex128)
    initial[0, 0, 0] = 1.0
    target = 0.5 * np.eye(2, dtype=np.complex128)[:, :, None]

    def evaluate(density: np.ndarray) -> FixedPointEvaluation[None]:
        return FixedPointEvaluation(
            output=target + 0.99 * (density - target),
            payload=None,
        )

    run = run_pulay_diis_fixed_point(
        initial,
        evaluate,
        precision=1.0e-14,
        max_iter=3,
        options=PulayDIISOptions(
            regularization=1.0e-16,
            max_regularization=1.0e-12,
            coefficient_l1_limit=10.0,
            trust_radius=10.0,
        ),
        validate_output=_validate_trace_one_hermitian,
        validate_iterate=_validate_trace_one_hermitian,
    )

    assert not run.converged
    assert run.restart_count >= 1
    assert any(step.step_kind == "restart_picard" for step in run.steps)
    _validate_trace_one_hermitian(run.density)
