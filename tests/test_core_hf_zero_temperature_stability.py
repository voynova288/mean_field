from __future__ import annotations

import numpy as np

from mean_field.core.hf.zero_temperature_stability import (
    build_penalized_projected_hessian_operator,
    build_zero_temperature_orbital_hessian,
    probe_projected_hessian_operator,
    solve_lowest_projected_hessian_eigenpairs,
)


def test_zero_temperature_hessian_full_ab_weighting_and_scalar_curvature() -> None:
    nk = 2
    weights = np.asarray([0.25, 0.75], dtype=np.float64)
    projector = np.repeat(
        np.diag([1.0, 0.0]).astype(np.complex128)[:, :, None], nk, axis=2
    )
    h0 = np.repeat(
        np.diag([-0.4, 0.6]).astype(np.complex128)[:, :, None], nk, axis=2
    )
    sigma_x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    coupling = 0.3

    def response(delta_projector: np.ndarray) -> np.ndarray:
        delta_q = sum(
            weights[k] * np.trace(sigma_x @ delta_projector[:, :, k]).real
            for k in range(nk)
        )
        return np.repeat(
            (coupling * delta_q * sigma_x)[:, :, None], nk, axis=2
        )

    hessian = build_zero_temperature_orbital_hessian(
        projector,
        h0,
        response,
        occupied_per_k=1,
        k_weights=weights,
        stationarity_tolerance_ev=0.0,
    )
    identity = np.eye(hessian.shape[0])
    dense = np.column_stack(
        [hessian.energy_hessian_action(identity[:, index]) for index in range(4)]
    )
    weight_root = np.sqrt(weights)
    expected = 2.0 * np.eye(4)
    expected[:2, :2] += 4.0 * coupling * np.outer(weight_root, weight_root)

    assert hessian.stationarity_max_abs_ev == 0.0
    np.testing.assert_allclose(dense, expected, atol=1.0e-13)
    np.testing.assert_allclose(dense, dense.T, atol=1.0e-13)

    probes = np.asarray(
        [
            [1.0, 2.0, -0.5, 0.3],
            [-0.4, 0.8, 1.2, -0.7],
            [0.2, -1.1, 0.6, 1.5],
        ],
        dtype=np.float64,
    )
    report = hessian.verify_self_adjointness(probes)
    assert report.pair_count == 3
    assert report.maximum_absolute_defect < 1.0e-14
    operator = hessian.as_linear_operator(
        verification_vectors=probes,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
    )
    np.testing.assert_allclose(operator @ probes[0], dense @ probes[0], atol=1.0e-13)

    direction = np.asarray([0.7, -0.2, 0.4, -0.5], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    coordinates = hessian.frame.unpack_weighted_real(direction)
    tangent = hessian.frame.tangent_projector(coordinates)
    step_for_tangent = 1.0e-6
    centered_tangent = (
        hessian.frame.unitary_projector(direction, amplitude=step_for_tangent)
        - hessian.frame.unitary_projector(direction, amplitude=-step_for_tangent)
    ) / (2.0 * step_for_tangent)
    np.testing.assert_allclose(centered_tangent, tangent, atol=2.0e-10)

    def energy(amplitude: float) -> float:
        rotated = hessian.frame.unitary_projector(direction, amplitude=amplitude)
        one_body = sum(
            weights[k] * np.trace(h0[:, :, k] @ rotated[:, :, k]).real
            for k in range(nk)
        )
        order = sum(
            weights[k] * np.trace(sigma_x @ rotated[:, :, k]).real
            for k in range(nk)
        )
        return float(one_body + 0.5 * coupling * order**2)

    step = 1.0e-2
    curvature = (
        -energy(2.0 * step)
        + 16.0 * energy(step)
        - 30.0 * energy(0.0)
        + 16.0 * energy(-step)
        - energy(-2.0 * step)
    ) / (12.0 * step**2)
    expected_curvature = float(direction @ dense @ direction)
    np.testing.assert_allclose(curvature, expected_curvature, rtol=0.0, atol=1.0e-7)


class _DenseRealHessian:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.shape = self.matrix.shape

    def energy_hessian_action(self, vector: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(vector, dtype=np.float64)


def test_penalized_projected_hessian_avoids_complement_zeros() -> None:
    hessian = _DenseRealHessian(np.diag([1.0, 2.0, -4.0]))
    projector = np.diag([1.0, 1.0, 0.0])
    projector_action = lambda vector: projector @ vector
    bare_projected = projector @ hessian.matrix @ projector
    assert np.linalg.eigvalsh(bare_projected)[0] == 0.0

    probes = np.asarray(
        [[1.0, 2.0, 3.0], [-0.5, 0.7, 1.1], [0.2, -1.0, 0.4]]
    )
    report = probe_projected_hessian_operator(
        hessian, projector_action, probes
    )
    assert report.projector_idempotency_max == 0.0
    assert report.sector_commutator_max_ev == 0.0
    penalized = build_penalized_projected_hessian_operator(
        hessian, projector_action, penalty_ev=10.0
    )
    result = solve_lowest_projected_hessian_eigenpairs(
        penalized,
        count=2,
        seed=7,
        ncv=3,
        tolerance=1.0e-12,
        max_iterations=100,
    )
    np.testing.assert_allclose(result.eigenvalues_ev, [1.0, 2.0], atol=1.0e-12)
    np.testing.assert_allclose(result.sector_leakage_norms, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.full_ritz_residuals_ev, 0.0, atol=1.0e-11)
    assert np.all(result.penalty_distances_ev > 7.0)


def test_projected_hessian_probe_detects_noncommuting_sector() -> None:
    hessian = _DenseRealHessian(
        np.asarray([[1.0, 0.4, 0.0], [0.4, 2.0, 0.0], [0.0, 0.0, 3.0]])
    )
    projector = np.diag([1.0, 0.0, 0.0])
    probes = np.asarray(
        [[1.0, 2.0, 0.5], [-0.3, 0.7, 1.0], [0.8, -1.1, 0.2]]
    )
    report = probe_projected_hessian_operator(
        hessian, lambda vector: projector @ vector, probes
    )
    assert report.projector_idempotency_max == 0.0
    assert report.hessian_self_adjoint_max_ev < 1.0e-14
    assert report.sector_commutator_max_ev > 0.1
    assert report.cross_sector_coupling_max_ev > 0.01


def test_zero_temperature_orbital_hessian_rejects_fractional_projector() -> None:
    projector = np.diag([0.75, 0.25]).astype(np.complex128)[:, :, None]
    hamiltonian = np.diag([-1.0, 1.0]).astype(np.complex128)[:, :, None]
    try:
        build_zero_temperature_orbital_hessian(
            projector,
            hamiltonian,
            lambda delta: np.zeros_like(delta),
            occupied_per_k=1,
        )
    except ValueError as error:
        assert "idempotent" in str(error)
    else:
        raise AssertionError("fractional projector was accepted")
