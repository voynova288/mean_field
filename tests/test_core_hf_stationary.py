from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf.stationary import (
    StationarySolveConfig,
    finite_difference_jacobian_vector,
    pack_hermitian_matrix_field,
    solve_stationary_residual,
    unpack_hermitian_matrix_field,
)
from mean_field.core.hf.symmetry import (
    antiunitary_matrix_field_residual,
    antiunitary_transform_matrix_field,
    project_antiunitary_matrix_field,
    validate_partner_involution,
)


def test_hermitian_matrix_field_coordinate_roundtrip() -> None:
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(4, 4, 5)) + 1j * rng.normal(size=(4, 4, 5))
    field = raw + np.swapaxes(raw.conj(), 0, 1)
    vector = pack_hermitian_matrix_field(field)
    restored = unpack_hermitian_matrix_field(vector, dimension=4, nk=5)
    assert vector.shape == (4 * 4 * 5,)
    assert np.max(np.abs(restored - field)) == 0.0


def test_antiunitary_matrix_field_projector_is_exact_and_idempotent() -> None:
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(4, 4, 5)) + 1j * rng.normal(size=(4, 4, 5))
    field = raw + np.swapaxes(raw.conj(), 0, 1)
    partners = np.asarray([4, 3, 2, 1, 0], dtype=np.int64)
    sy = np.asarray([[0.0, -1j], [1j, 0.0]])
    unitary = np.kron(1j * sy, np.eye(2))
    assert np.max(np.abs(unitary @ unitary.conj() + np.eye(4))) == 0.0
    projected = project_antiunitary_matrix_field(
        field,
        partner_indices=partners,
        unitary=unitary,
    )
    projected_twice = project_antiunitary_matrix_field(
        projected,
        partner_indices=partners,
        unitary=unitary,
    )
    assert np.max(np.abs(projected_twice - projected)) < 2.0e-15
    assert antiunitary_matrix_field_residual(
        projected,
        partner_indices=partners,
        unitary=unitary,
    ) < 2.0e-15
    transformed = antiunitary_transform_matrix_field(
        field,
        partner_indices=partners,
        unitary=unitary,
    )
    transformed_twice = antiunitary_transform_matrix_field(
        transformed,
        partner_indices=partners,
        unitary=unitary,
    )
    assert np.max(np.abs(transformed_twice - field)) < 2.0e-15
    with pytest.raises(ValueError, match="involution"):
        validate_partner_involution(np.asarray([0, 0]))


def test_stationary_solver_converges_to_saddle_that_energy_descent_leaves() -> None:
    # E(x,y)=0.5*(x-1)^2-0.5*(y+2)^2 has a stationary saddle at (1,-2).
    def gradient(vector: np.ndarray) -> np.ndarray:
        return np.asarray([vector[0] - 1.0, -(vector[1] + 2.0)])

    initial = np.asarray([1.2, -1.8])
    descent = initial.copy()
    for _ in range(8):
        descent -= 0.2 * gradient(descent)
    assert abs(descent[1] + 2.0) > abs(initial[1] + 2.0)

    result = solve_stationary_residual(
        gradient,
        initial,
        config=StationarySolveConfig(
            residual_rms_tolerance=1.0e-12,
            residual_max_tolerance=1.0e-12,
            anderson_max_iterations=0,
            anderson_memory=1,
            krylov_max_iterations=30,
        ),
    )
    assert result.converged
    assert result.residual_max < 1.0e-12
    assert result.vector == pytest.approx([1.0, -2.0], abs=1.0e-12)


def test_finite_difference_jacobian_vector_matches_linear_map() -> None:
    matrix = np.asarray([[2.0, -0.3], [0.7, -1.2]])
    point = np.asarray([0.4, -0.8])
    direction = np.asarray([0.2, 0.9])
    observed = finite_difference_jacobian_vector(
        lambda vector: matrix @ vector,
        point,
        direction,
    )
    assert observed == pytest.approx(matrix @ direction, rel=2.0e-10, abs=2.0e-10)
