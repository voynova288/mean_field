"""Focused tests for the system-agnostic ragged candidate Hessian action."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mean_field.core.hf import (
    ZeroTemperatureRaggedOrbitalHessian,
    build_zero_temperature_ragged_orbital_hessian,
)


@dataclass(frozen=True)
class _Toy:
    hessian: ZeroTemperatureRaggedOrbitalHessian
    one_body: np.ndarray
    full_fock: np.ndarray
    observables: np.ndarray
    coupling: float
    source_signal: float
    active_couplings: tuple[np.ndarray, np.ndarray]

    def energy(self, projectors: np.ndarray) -> float:
        weights = self.hessian.block_weights
        one_body_energy = sum(
            weights[block]
            * np.trace(self.one_body[:, :, block] @ projectors[:, :, block]).real
            for block in range(self.hessian.nblock)
        )
        signal = sum(
            weights[block]
            * np.trace(
                self.observables[:, :, block] @ projectors[:, :, block]
            ).real
            for block in range(self.hessian.nblock)
        )
        return float(one_body_energy + 0.5 * self.coupling * signal**2)


def _unitary(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, _ = np.linalg.qr(raw)
    return q


def _toy() -> _Toy:
    """Two partial unequal-weight blocks from one explicit scalar functional."""

    n = 3
    counts = (0, 1, 2, n)
    nblock = len(counts)
    weights = np.array([2.3, 3.7, 1.9, 5.1])
    basis = np.stack([_unitary(11 + block, n) for block in range(nblock)], axis=2)

    projectors = np.empty((n, n, nblock), dtype=np.complex128)
    full_fock = np.empty_like(projectors)
    observables = np.zeros_like(projectors)
    spectra = (
        np.array([0.2, 0.8, 1.4]),
        np.array([-0.4, 0.5, 1.1]),
        np.array([-1.0, -0.2, 0.7]),
        np.array([-1.2, -0.7, -0.1]),
    )
    for block, occupied in enumerate(counts):
        unitary = basis[:, :, block]
        occupied_basis = unitary[:, :occupied]
        projectors[:, :, block] = occupied_basis @ occupied_basis.conj().T
        full_fock[:, :, block] = (
            unitary @ np.diag(spectra[block]) @ unitary.conj().T
        )

    coupling_1 = np.array([0.4 + 0.3j, -0.2 + 0.5j])
    observable_1 = np.diag([0.35, 0.0, 0.0]).astype(np.complex128)
    observable_1[1:, 0] = coupling_1
    observable_1[0, 1:] = coupling_1.conj()
    observables[:, :, 1] = (
        basis[:, :, 1] @ observable_1 @ basis[:, :, 1].conj().T
    )

    coupling_2 = np.array([0.3 - 0.4j, -0.25 + 0.2j])
    observable_2 = np.diag([-0.1, 0.28, 0.0]).astype(np.complex128)
    observable_2[2, :2] = coupling_2
    observable_2[:2, 2] = coupling_2.conj()
    observables[:, :, 2] = (
        basis[:, :, 2] @ observable_2 @ basis[:, :, 2].conj().T
    )

    source_signal = float(
        sum(
            weights[block]
            * np.trace(
                observables[:, :, block] @ projectors[:, :, block]
            ).real
            for block in range(nblock)
        )
    )
    coupling = 0.63
    source_mean_field = coupling * source_signal * observables
    one_body = full_fock - source_mean_field

    def fock_derivative(tangent: np.ndarray) -> np.ndarray:
        signal = sum(
            weights[block]
            * np.trace(observables[:, :, block] @ tangent[:, :, block]).real
            for block in range(nblock)
        )
        return coupling * signal * observables

    hessian = build_zero_temperature_ragged_orbital_hessian(
        projectors,
        full_fock,
        basis,
        counts,
        fock_derivative,
        block_weights=weights,
    )
    return _Toy(
        hessian,
        one_body,
        full_fock,
        observables,
        coupling,
        source_signal,
        (coupling_1, coupling_2),
    )


def _active_blocks(
    x_1: np.ndarray, x_2: np.ndarray
) -> tuple[np.ndarray, ...]:
    return (
        np.empty((3, 0), dtype=np.complex128),
        np.asarray(x_1, dtype=np.complex128).reshape(2, 1),
        np.asarray(x_2, dtype=np.complex128).reshape(1, 2),
        np.empty((0, 3), dtype=np.complex128),
    )


def _direction() -> tuple[np.ndarray, ...]:
    return _active_blocks(
        np.array([0.31 - 0.27j, -0.18 + 0.41j]),
        np.array([0.16 + 0.22j, -0.37 + 0.09j]),
    )


def test_ragged_zero_two_partial_full_layout_and_exact_retraction() -> None:
    toy = _toy()
    hessian = toy.hessian
    assert hessian.occupied_counts == (0, 1, 2, 3)
    assert tuple(layout.shape for layout in hessian.layouts) == (
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
    )
    assert np.array_equal(hessian.complex_offsets, np.array([0, 0, 2, 4, 4]))
    assert hessian.complex_dimension == 4
    assert hessian.real_dimension == 8
    assert np.array_equal(hessian.block_weights, np.array([2.3, 3.7, 1.9, 5.1]))
    assert not np.isclose(np.sum(hessian.block_weights), 1.0)

    # The constructor receives full F[P0], not h0.  The nonzero source
    # mean-field field cancels an off-diagonal term in h0 at stationarity.
    assert abs(toy.source_signal) > 0.1
    source_mean_field = toy.coupling * toy.source_signal * toy.observables
    assert np.linalg.norm(source_mean_field) > 0.1
    assert np.allclose(toy.one_body + source_mean_field, toy.full_fock)
    assert not np.allclose(toy.one_body, toy.full_fock)

    blocks = _direction()
    packed = hessian.pack_real(blocks)
    unpacked = hessian.unpack_real(packed)
    for expected, actual in zip(blocks, unpacked):
        assert np.array_equal(expected, actual)

    tangent = hessian.tangent(packed)
    for block in range(hessian.nblock):
        assert np.allclose(tangent[:, :, block], tangent[:, :, block].conj().T)
    assert np.linalg.norm(tangent[:, :, 0]) == 0.0
    assert np.linalg.norm(tangent[:, :, 3]) == 0.0

    step = 1.0e-6
    centered_tangent = (
        hessian.retract(packed, step) - hessian.retract(packed, -step)
    ) / (2.0 * step)
    assert np.allclose(centered_tangent, tangent, atol=3.0e-10, rtol=3.0e-10)

    retracted = hessian.retract(packed, 0.37)
    for block, count in enumerate(hessian.occupied_counts):
        projector = retracted[:, :, block]
        assert np.allclose(projector, projector.conj().T, atol=2.0e-14)
        assert np.allclose(projector @ projector, projector, atol=4.0e-14)
        assert np.isclose(np.trace(projector), count, atol=4.0e-14)


def test_reciprocal_cross_block_weighted_identity_and_complex_action() -> None:
    toy = _toy()
    hessian = toy.hessian
    blocks = _direction()
    direction = hessian.pack_real(blocks)

    x_1 = blocks[1]
    x_2 = blocks[2]
    c_1, c_2 = toy.active_couplings
    signal = 2.0 * (
        hessian.block_weights[1] * np.vdot(c_1, x_1[:, 0]).real
        + hessian.block_weights[2] * np.vdot(c_2, x_2[0, :]).real
    )
    expected_1 = np.array([0.9, 1.5])[:, None] * x_1
    expected_1 += toy.coupling * signal * c_1[:, None]
    expected_2 = np.array([1.7, 0.9])[None, :] * x_2
    expected_2 += toy.coupling * signal * c_2[None, :]

    action = hessian.complex_action(direction)
    assert np.allclose(action[1], expected_1, atol=3.0e-14, rtol=3.0e-14)
    assert np.allclose(action[2], expected_2, atol=3.0e-14, rtol=3.0e-14)

    weighted_real_action = hessian.unpack_real(
        hessian.candidate_linear_operator @ direction
    )
    assert np.allclose(
        weighted_real_action[1],
        2.0 * hessian.block_weights[1] * expected_1,
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    assert np.allclose(
        weighted_real_action[2],
        2.0 * hessian.block_weights[2] * expected_2,
        atol=3.0e-13,
        rtol=3.0e-13,
    )

    other = hessian.pack_real(
        _active_blocks(
            np.array([-0.21 + 0.13j, 0.07 - 0.19j]),
            np.array([0.24 - 0.16j, 0.11 + 0.28j]),
        )
    )
    d = hessian.tangent(direction)
    d_tilde = hessian.tangent(other)
    df_d = hessian.fock_derivative(d)
    df_d_tilde = hessian.fock_derivative(d_tilde)
    left = sum(
        hessian.block_weights[b] * np.trace(d[:, :, b] @ df_d_tilde[:, :, b])
        for b in range(hessian.nblock)
    )
    right = sum(
        hessian.block_weights[b] * np.trace(d_tilde[:, :, b] @ df_d[:, :, b])
        for b in range(hessian.nblock)
    )
    assert abs(left.imag) < 2.0e-15
    assert abs(right.imag) < 2.0e-15
    assert np.isclose(left.real, right.real, atol=2.0e-14, rtol=2.0e-14)


def test_complete_dense_scalar_curvature_oracle_and_block_gauge_covariance() -> None:
    """Compare every candidate column with an independent scalar oracle."""

    rng = np.random.default_rng(20260829)
    n = 5
    counts = (2, 3)
    weights = np.array([1.7, 2.6])
    basis = np.stack((_unitary(101, n), _unitary(102, n)), axis=2)
    projectors = np.empty((n, n, 2), dtype=np.complex128)
    full_fock = np.empty_like(projectors)

    foo_blocks = (
        np.array([[-1.3, 0.17 + 0.23j], [0.17 - 0.23j, -0.7]]),
        np.array(
            [
                [-1.5, 0.11 - 0.19j, -0.08 + 0.14j],
                [0.11 + 0.19j, -0.9, 0.21 + 0.16j],
                [-0.08 - 0.14j, 0.21 - 0.16j, -0.4],
            ]
        ),
    )
    fvv_blocks = (
        np.array(
            [
                [0.4, -0.13 + 0.18j, 0.09 - 0.12j],
                [-0.13 - 0.18j, 0.9, 0.16 + 0.22j],
                [0.09 + 0.12j, 0.16 - 0.22j, 1.5],
            ]
        ),
        np.array([[0.6, -0.15 - 0.24j], [-0.15 + 0.24j, 1.2]]),
    )
    for block, occupied in enumerate(counts):
        unitary = basis[:, :, block]
        occupied_basis = unitary[:, :occupied]
        projectors[:, :, block] = occupied_basis @ occupied_basis.conj().T
        fock_in_orbital_gauge = np.zeros((n, n), dtype=np.complex128)
        fock_in_orbital_gauge[:occupied, :occupied] = foo_blocks[block]
        fock_in_orbital_gauge[occupied:, occupied:] = fvv_blocks[block]
        full_fock[:, :, block] = unitary @ fock_in_orbital_gauge @ unitary.conj().T
        assert np.linalg.norm(np.triu(foo_blocks[block], 1).imag) > 0.1
        assert np.linalg.norm(np.triu(fvv_blocks[block], 1).imag) > 0.1

    # Orthonormal real basis for every Hermitian matrix degree of freedom.
    hermitian_basis: list[np.ndarray] = []
    for row in range(n):
        element = np.zeros((n, n), dtype=np.complex128)
        element[row, row] = 1.0
        hermitian_basis.append(element)
    for row in range(n):
        for column in range(row + 1, n):
            symmetric = np.zeros((n, n), dtype=np.complex128)
            symmetric[row, column] = symmetric[column, row] = 1.0 / np.sqrt(2.0)
            hermitian_basis.append(symmetric)
            imaginary = np.zeros((n, n), dtype=np.complex128)
            imaginary[row, column] = -1j / np.sqrt(2.0)
            imaginary[column, row] = 1j / np.sqrt(2.0)
            hermitian_basis.append(imaginary)
    assert len(hermitian_basis) == n * n

    def matrix_coordinates(matrices: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.trace(element @ matrices[:, :, block]).real
                for block in range(2)
                for element in hermitian_basis
            ]
        )

    def matrices_from_coordinates(coordinates: np.ndarray) -> np.ndarray:
        result = np.zeros((n, n, 2), dtype=np.complex128)
        cursor = 0
        for block in range(2):
            for element in hermitian_basis:
                result[:, :, block] += coordinates[cursor] * element
                cursor += 1
        return result

    matrix_dimension = 2 * n * n
    raw = rng.standard_normal((matrix_dimension, matrix_dimension))
    response_kernel = 0.08 * (raw + raw.T) / np.sqrt(matrix_dimension)
    response_kernel += np.diag(np.linspace(0.3, 0.9, matrix_dimension))
    assert np.linalg.matrix_rank(response_kernel) == matrix_dimension
    assert np.linalg.norm(response_kernel[: n * n, n * n :]) > 0.1
    coordinate_weights = np.repeat(weights, n * n)

    def reciprocal_real_linear_response(tangent: np.ndarray) -> np.ndarray:
        tangent_coordinates = matrix_coordinates(tangent)
        response_coordinates = response_kernel @ tangent_coordinates / coordinate_weights
        return matrices_from_coordinates(response_coordinates)

    # Exact scalar functional: E(P)=sum_b w_b Tr(h0_b P_b)+p(P)^T C p(P)/2.
    source_response = reciprocal_real_linear_response(projectors)
    one_body = full_fock - source_response
    np.testing.assert_allclose(one_body + source_response, full_fock, atol=2.0e-14)
    for _ in range(2):
        left_matrices = matrices_from_coordinates(rng.standard_normal(matrix_dimension))
        right_matrices = matrices_from_coordinates(rng.standard_normal(matrix_dimension))
        left_response = reciprocal_real_linear_response(left_matrices)
        right_response = reciprocal_real_linear_response(right_matrices)
        left_pairing = sum(
            weights[b] * np.trace(left_matrices[:, :, b] @ right_response[:, :, b]).real
            for b in range(2)
        )
        right_pairing = sum(
            weights[b] * np.trace(right_matrices[:, :, b] @ left_response[:, :, b]).real
            for b in range(2)
        )
        assert left_pairing == pytest.approx(right_pairing, abs=2.0e-13)

    shapes = tuple((n - occupied, occupied) for occupied in counts)
    complex_dimension = sum(virtual * occupied for virtual, occupied in shapes)
    real_dimension = 2 * complex_dimension

    def unpack_independently(real_coordinates: np.ndarray) -> tuple[np.ndarray, ...]:
        complex_coordinates = (
            real_coordinates[:complex_dimension]
            + 1j * real_coordinates[complex_dimension:]
        )
        blocks: list[np.ndarray] = []
        cursor = 0
        for shape in shapes:
            size = shape[0] * shape[1]
            blocks.append(complex_coordinates[cursor : cursor + size].reshape(shape))
            cursor += size
        return tuple(blocks)

    def pack_independently(blocks: tuple[np.ndarray, ...]) -> np.ndarray:
        flat = np.concatenate(tuple(block.reshape(-1) for block in blocks))
        return np.concatenate((flat.real, flat.imag))

    def commutator(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left @ right - right @ left

    def direction_geometry(
        real_coordinates: np.ndarray, orbital_basis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        generators = np.zeros_like(projectors)
        tangents = np.zeros_like(projectors)
        for block, (occupied, x_block) in enumerate(
            zip(counts, unpack_independently(real_coordinates))
        ):
            generator_in_orbital_gauge = np.zeros((n, n), dtype=np.complex128)
            generator_in_orbital_gauge[occupied:, :occupied] = x_block
            generator_in_orbital_gauge[:occupied, occupied:] = -x_block.conj().T
            unitary = orbital_basis[:, :, block]
            generator = unitary @ generator_in_orbital_gauge @ unitary.conj().T
            generators[:, :, block] = generator
            tangents[:, :, block] = commutator(generator, projectors[:, :, block])
        return generators, tangents

    def scalar_curvature_polarization(
        left_coordinates: np.ndarray,
        right_coordinates: np.ndarray,
        orbital_basis: np.ndarray,
    ) -> float:
        left_generators, left_tangents = direction_geometry(
            left_coordinates, orbital_basis
        )
        right_generators, right_tangents = direction_geometry(
            right_coordinates, orbital_basis
        )
        geometric = 0.0
        for block in range(2):
            mixed_second_projector = 0.5 * (
                commutator(
                    left_generators[:, :, block], right_tangents[:, :, block]
                )
                + commutator(
                    right_generators[:, :, block], left_tangents[:, :, block]
                )
            )
            geometric += weights[block] * np.trace(
                full_fock[:, :, block] @ mixed_second_projector
            ).real
        interaction = (
            matrix_coordinates(left_tangents)
            @ response_kernel
            @ matrix_coordinates(right_tangents)
        )
        return float(geometric + interaction)

    def independent_dense_oracle(orbital_basis: np.ndarray) -> np.ndarray:
        coordinate_basis = np.eye(real_dimension)
        return np.array(
            [
                [
                    scalar_curvature_polarization(
                        coordinate_basis[:, row],
                        coordinate_basis[:, column],
                        orbital_basis,
                    )
                    for column in range(real_dimension)
                ]
                for row in range(real_dimension)
            ]
        )

    candidate = ZeroTemperatureRaggedOrbitalHessian(
        projectors,
        full_fock,
        basis,
        counts,
        reciprocal_real_linear_response,
        block_weights=weights,
    )
    coordinate_basis = np.eye(real_dimension)
    actual_dense = np.column_stack(
        [candidate.matvec(coordinate_basis[:, column]) for column in range(real_dimension)]
    )
    oracle_dense = independent_dense_oracle(basis)
    for column in range(real_dimension):
        np.testing.assert_allclose(
            actual_dense[:, column],
            oracle_dense[:, column],
            atol=8.0e-13,
            rtol=8.0e-13,
        )
    np.testing.assert_allclose(oracle_dense, oracle_dense.T, atol=5.0e-13)

    left_coordinates = rng.standard_normal(real_dimension)
    right_coordinates = rng.standard_normal(real_dimension)
    expected_polarization = scalar_curvature_polarization(
        left_coordinates, right_coordinates, basis
    )
    assert left_coordinates @ actual_dense @ right_coordinates == pytest.approx(
        expected_polarization, abs=2.0e-11, rel=2.0e-12
    )

    rotations: list[tuple[np.ndarray, np.ndarray]] = []
    rotated_basis = np.empty_like(basis)
    for block, occupied in enumerate(counts):
        occupied_rotation = _unitary(201 + block, occupied)
        virtual_rotation = _unitary(211 + block, n - occupied)
        rotations.append((occupied_rotation, virtual_rotation))
        gauge_rotation = np.zeros((n, n), dtype=np.complex128)
        gauge_rotation[:occupied, :occupied] = occupied_rotation
        gauge_rotation[occupied:, occupied:] = virtual_rotation
        rotated_basis[:, :, block] = basis[:, :, block] @ gauge_rotation

    coordinate_rotation = np.empty((real_dimension, real_dimension))
    for column in range(real_dimension):
        old_blocks = unpack_independently(coordinate_basis[:, column])
        new_blocks = tuple(
            virtual_rotation.conj().T @ old_block @ occupied_rotation
            for old_block, (occupied_rotation, virtual_rotation) in zip(
                old_blocks, rotations
            )
        )
        coordinate_rotation[:, column] = pack_independently(new_blocks)
    np.testing.assert_allclose(
        coordinate_rotation.T @ coordinate_rotation,
        np.eye(real_dimension),
        atol=7.0e-14,
    )

    rotated_candidate = ZeroTemperatureRaggedOrbitalHessian(
        projectors,
        full_fock,
        rotated_basis,
        counts,
        reciprocal_real_linear_response,
        block_weights=weights,
    )
    rotated_actual_dense = np.column_stack(
        [
            rotated_candidate.matvec(coordinate_basis[:, column])
            for column in range(real_dimension)
        ]
    )
    rotated_oracle_dense = independent_dense_oracle(rotated_basis)
    np.testing.assert_allclose(
        rotated_actual_dense, rotated_oracle_dense, atol=9.0e-13, rtol=9.0e-13
    )
    np.testing.assert_allclose(
        rotated_actual_dense,
        coordinate_rotation @ actual_dense @ coordinate_rotation.T,
        atol=1.2e-12,
        rtol=1.2e-12,
    )
    _, old_tangent = direction_geometry(left_coordinates, basis)
    _, new_tangent = direction_geometry(
        coordinate_rotation @ left_coordinates, rotated_basis
    )
    np.testing.assert_allclose(new_tangent, old_tangent, atol=7.0e-14)

def test_candidate_scope_and_bilinear_diagnostics_cannot_imply_proof() -> None:
    hessian = _toy().hessian
    assert hessian.scope == "fixed-per-block-rank, block-preserving candidate"
    assert not hessian.tests_inter_block_occupation_transfers
    assert not hessian.tests_aufbau_ordering
    assert hessian.requires_separate_occupation_gap_gate
    assert hessian.candidate_linear_operator is hessian.linear_operator
    assert hessian.candidate_linear_operator is hessian.operator
    assert hessian.linear_operator.shape == (8, 8)
    assert hessian.linear_operator.dtype == np.dtype(np.float64)
    with pytest.raises(NotImplementedError):
        hessian.linear_operator.rmatvec(np.ones(hessian.real_dimension))

    report_1 = hessian.diagnose_bilinear_symmetry(
        seed=20260828, probe_count=8, atol=2.0e-12, rtol=2.0e-12
    )
    report_2 = hessian.diagnose_bilinear_symmetry(
        seed=20260828, probe_count=8, atol=2.0e-12, rtol=2.0e-12
    )
    assert report_1 == report_2
    assert not report_1.inconclusive
    assert report_1.all_evaluated_pairs_symmetric
    assert not report_1.asymmetry_detected
    assert report_1.outcome == "sampled_pairs_symmetric_not_proof"
    assert report_1.maximum_residual < 2.0e-13

    zero_probe = hessian.diagnose_bilinear_symmetry(probe_count=0)
    assert zero_probe.retained_real_dimension > 0
    assert zero_probe.inconclusive
    assert not zero_probe.all_evaluated_pairs_symmetric
    assert not zero_probe.asymmetry_detected
    assert zero_probe.outcome == "inconclusive"


def test_nonreciprocal_cross_block_callback_detected_by_bilinear_diagnostic() -> None:
    toy = _toy()
    hessian = toy.hessian

    def nonreciprocal_response(tangent: np.ndarray) -> np.ndarray:
        result = np.zeros_like(tangent)
        signal_from_block_2 = np.trace(
            toy.observables[:, :, 2] @ tangent[:, :, 2]
        ).real
        result[:, :, 1] = signal_from_block_2 * toy.observables[:, :, 1]
        return result

    candidate = ZeroTemperatureRaggedOrbitalHessian(
        hessian.projectors,
        hessian.hamiltonians,
        hessian.orbital_basis,
        hessian.occupied_counts,
        nonreciprocal_response,
        block_weights=hessian.block_weights,
    )
    report = candidate.diagnose_bilinear_symmetry(
        seed=17, probe_count=12, atol=1.0e-13, rtol=1.0e-13
    )
    assert not report.inconclusive
    assert not report.all_evaluated_pairs_symmetric
    assert report.asymmetry_detected
    assert report.outcome == "sampled_bilinear_asymmetry_detected"
    assert report.maximum_residual > 1.0e-4


def test_stationarity_gate_is_block_gauge_invariant_near_threshold() -> None:
    n = 4
    occupied = 2
    amplitude = 2.0e-8
    projectors = np.diag([1.0, 1.0, 0.0, 0.0]).astype(np.complex128)[:, :, None]
    hamiltonian = np.zeros((n, n), dtype=np.complex128)
    gradient = np.zeros((n - occupied, occupied), dtype=np.complex128)
    gradient[0, 0] = amplitude
    hamiltonian[occupied:, :occupied] = gradient
    hamiltonian[:occupied, occupied:] = gradient.conj().T
    hamiltonians = hamiltonian[:, :, None]

    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    identity_basis = np.eye(n, dtype=np.complex128)
    rotated_basis = np.zeros((n, n), dtype=np.complex128)
    rotated_basis[:occupied, :occupied] = hadamard
    rotated_basis[occupied:, occupied:] = hadamard

    def zero_response(tangent: np.ndarray) -> np.ndarray:
        return np.zeros_like(tangent)

    expected_residual = np.linalg.norm(gradient, ord="fro")
    expected_hamiltonian_scale = np.linalg.norm(hamiltonian, ord="fro")
    accepting_rtol = 1.001 * expected_residual / expected_hamiltonian_scale
    accepted = []
    for basis in (identity_basis, rotated_basis):
        accepted.append(
            ZeroTemperatureRaggedOrbitalHessian(
                projectors,
                hamiltonians,
                basis[:, :, None],
                (occupied,),
                zero_response,
                stationarity_atol=0.0,
                stationarity_rtol=accepting_rtol,
            )
        )
    for candidate in accepted:
        assert candidate.stationarity_residuals[0] == pytest.approx(
            expected_residual, rel=2.0e-15, abs=2.0e-24
        )
        assert candidate.stationarity_tolerances[0] == pytest.approx(
            1.001 * expected_residual, rel=2.0e-15, abs=2.0e-24
        )

    rejecting_rtol = 0.999 * expected_residual / expected_hamiltonian_scale
    for basis in (identity_basis, rotated_basis):
        with pytest.raises(ValueError, match="not orbital-stationary"):
            ZeroTemperatureRaggedOrbitalHessian(
                projectors,
                hamiltonians,
                basis[:, :, None],
                (occupied,),
                zero_response,
                stationarity_atol=0.0,
                stationarity_rtol=rejecting_rtol,
            )


def test_geometry_and_hamiltonian_scales_are_separate_and_shift_invariant() -> None:
    toy = _toy()
    hessian = toy.hessian
    identity_blocks = np.repeat(
        np.eye(hessian.n, dtype=np.complex128)[:, :, None],
        hessian.nblock,
        axis=2,
    )
    shifted_hamiltonians = hessian.hamiltonians + 1.0e6 * identity_blocks
    shifted = ZeroTemperatureRaggedOrbitalHessian(
        hessian.projectors,
        shifted_hamiltonians,
        hessian.orbital_basis,
        hessian.occupied_counts,
        hessian.fock_derivative,
        block_weights=hessian.block_weights,
    )
    assert np.allclose(
        shifted.stationarity_tolerances,
        hessian.stationarity_tolerances,
        atol=2.0e-18,
        rtol=2.0e-10,
    )
    assert np.allclose(
        shifted.stationarity_residuals,
        hessian.stationarity_residuals,
        atol=2.0e-10,
        rtol=0.0,
    )

    # A huge Hamiltonian identity offset must not loosen dimensionless
    # projector/basis checks.
    bad_projector = np.array(hessian.projectors, copy=True)
    bad_projector[0, 1, 1] += 1.0e-5j
    with pytest.raises(ValueError, match="projectors.*not Hermitian"):
        ZeroTemperatureRaggedOrbitalHessian(
            bad_projector,
            hessian.hamiltonians + 1.0e12 * identity_blocks,
            hessian.orbital_basis,
            hessian.occupied_counts,
            hessian.fock_derivative,
            block_weights=hessian.block_weights,
        )


def test_source_stationarity_and_response_hermiticity_validation() -> None:
    toy = _toy()
    hessian = toy.hessian

    nonstationary = np.array(hessian.hamiltonians, copy=True)
    unitary = hessian.orbital_basis[:, :, 1]
    bad_basis_hamiltonian = unitary.conj().T @ nonstationary[:, :, 1] @ unitary
    bad_basis_hamiltonian[1, 0] = 1.0e-4 + 2.0e-4j
    bad_basis_hamiltonian[0, 1] = bad_basis_hamiltonian[1, 0].conjugate()
    nonstationary[:, :, 1] = unitary @ bad_basis_hamiltonian @ unitary.conj().T
    with pytest.raises(ValueError, match="not orbital-stationary"):
        ZeroTemperatureRaggedOrbitalHessian(
            hessian.projectors,
            nonstationary,
            hessian.orbital_basis,
            hessian.occupied_counts,
            hessian.fock_derivative,
            block_weights=hessian.block_weights,
        )

    nonhermitian_hamiltonian = np.array(hessian.hamiltonians, copy=True)
    nonhermitian_hamiltonian[0, 1, 0] += 1.0e-4j
    with pytest.raises(ValueError, match="hamiltonians.*not Hermitian"):
        ZeroTemperatureRaggedOrbitalHessian(
            hessian.projectors,
            nonhermitian_hamiltonian,
            hessian.orbital_basis,
            hessian.occupied_counts,
            hessian.fock_derivative,
            block_weights=hessian.block_weights,
        )

    def nonhermitian_response(tangent: np.ndarray) -> np.ndarray:
        result = np.zeros_like(tangent)
        result[0, 1, 1] = 1.0
        return result

    bad_response = ZeroTemperatureRaggedOrbitalHessian(
        hessian.projectors,
        hessian.hamiltonians,
        hessian.orbital_basis,
        hessian.occupied_counts,
        nonhermitian_response,
        block_weights=hessian.block_weights,
    )
    direction = hessian.pack_real(_direction())
    with pytest.raises(
        ValueError, match="fock_derivative output block 1 is not Hermitian"
    ):
        bad_response.matvec(direction)


def test_diagnostic_five_point_normalizes_direction_and_matches_same_functional() -> None:
    toy = _toy()
    hessian = toy.hessian
    direction = hessian.pack_real(
        _active_blocks(
            np.array([0.23 - 0.17j, -0.11 + 0.29j]),
            np.array([0.14 + 0.08j, -0.26 + 0.18j]),
        )
    )

    check = hessian.check_five_point_curvature(
        direction,
        toy.energy,
        step=2.0e-3,
        curvature_atol=2.0e-8,
        curvature_rtol=2.0e-8,
        stationarity_atol=2.0e-9,
    )
    scaled_check = hessian.check_five_point_curvature(
        7.0 * direction,
        toy.energy,
        step=2.0e-3,
        curvature_atol=2.0e-8,
        curvature_rtol=2.0e-8,
        stationarity_atol=2.0e-9,
    )
    assert check.diagnostic_only
    assert np.isclose(check.input_direction_norm, np.linalg.norm(direction))
    assert np.isclose(check.evaluated_direction_norm, 1.0)
    assert np.isclose(scaled_check.input_direction_norm, 7.0 * np.linalg.norm(direction))
    assert np.isclose(scaled_check.evaluated_direction_norm, 1.0)
    assert check.energies_minus_2h_to_plus_2h == pytest.approx(
        scaled_check.energies_minus_2h_to_plus_2h, abs=2.0e-14, rel=2.0e-14
    )
    assert check.predicted_curvature == pytest.approx(
        scaled_check.predicted_curvature, abs=2.0e-14, rel=2.0e-14
    )
    assert check.passed
    assert abs(check.stationarity_derivative) < 2.0e-10
    assert check.curvature_residual < 2.0e-8


def test_energy_diagnostics_rescale_homogeneously_from_ev_to_mev() -> None:
    toy = _toy()
    source = toy.hessian
    unit_scales = {"eV": 1.0e-3, "meV": 1.0}
    candidates: dict[str, ZeroTemperatureRaggedOrbitalHessian] = {}
    for unit, scale in unit_scales.items():
        def scaled_response(
            tangent: np.ndarray, *, _scale: float = scale
        ) -> np.ndarray:
            return _scale * source.fock_derivative(tangent)

        candidates[unit] = ZeroTemperatureRaggedOrbitalHessian(
            source.projectors,
            scale * source.hamiltonians,
            source.orbital_basis,
            source.occupied_counts,
            scaled_response,
            block_weights=source.block_weights,
            hamiltonian_atol=scale * 2.0e-10,
            stationarity_atol=scale * 2.0e-10,
        )

    bilinear = {
        unit: candidate.diagnose_bilinear_symmetry(
            seed=314159,
            probe_count=6,
            atol=unit_scales[unit] * 2.0e-12,
            rtol=2.0e-7,
        )
        for unit, candidate in candidates.items()
    }
    assert bilinear["eV"].all_evaluated_pairs_symmetric
    assert bilinear["meV"].all_evaluated_pairs_symmetric
    for ev_probe, mev_probe in zip(
        bilinear["eV"].probes, bilinear["meV"].probes
    ):
        for field in ("left_right", "right_left", "residual", "tolerance"):
            assert getattr(mev_probe, field) == pytest.approx(
                1000.0 * getattr(ev_probe, field), rel=3.0e-10, abs=2.0e-15
            )
        assert mev_probe.passed == ev_probe.passed

    direction = source.pack_real(_direction())
    curvature = {
        unit: candidate.check_five_point_curvature(
            direction,
            lambda projectors, _scale=unit_scales[unit]: (
                _scale * toy.energy(projectors)
            ),
            step=2.0e-3,
            curvature_atol=unit_scales[unit] * 2.0e-8,
            curvature_rtol=2.0e-8,
            stationarity_atol=unit_scales[unit] * 2.0e-9,
        )
        for unit, candidate in candidates.items()
    }
    assert curvature["eV"].passed
    assert curvature["meV"].passed
    # The cancellation-dominated raw stationarity and curvature residuals
    # need not rescale bitwise; the physical values, tolerances, and verdict do.
    for field in (
        "predicted_curvature",
        "finite_difference_curvature",
        "curvature_tolerance",
        "stationarity_tolerance",
    ):
        assert getattr(curvature["meV"], field) == pytest.approx(
            1000.0 * getattr(curvature["eV"], field),
            rel=3.0e-7,
            abs=2.0e-12,
        )
    assert curvature["meV"].energies_minus_2h_to_plus_2h == pytest.approx(
        tuple(
            1000.0 * value
            for value in curvature["eV"].energies_minus_2h_to_plus_2h
        ),
        rel=3.0e-13,
        abs=2.0e-14,
    )

    eps = np.finfo(float).eps
    for unit, candidate in candidates.items():
        real_energy = 2.0 * unit_scales[unit]
        complex_energy = complex(real_energy, 128.0 * eps * abs(real_energy))
        with pytest.raises(ValueError, match="non-real scalar"):
            candidate.check_five_point_curvature(
                direction,
                lambda projectors, _value=complex_energy: _value,
                step=2.0e-3,
            )


@pytest.mark.parametrize(
    "bad_energy",
    [
        complex(np.nan, 0.0),
        complex(0.0, np.nan),
        complex(np.inf, 0.0),
        complex(0.0, np.inf),
    ],
)
def test_five_point_hard_rejects_nonfinite_real_or_imag_energy(
    bad_energy: complex,
) -> None:
    hessian = _toy().hessian
    direction = hessian.pack_real(_direction())
    with pytest.raises(ValueError, match="non-finite real or imaginary"):
        hessian.check_five_point_curvature(
            direction,
            lambda projectors: bad_energy,
            step=1.0e-3,
        )


def test_raw_unweighted_default_matches_explicit_all_one_repository_abi() -> None:
    toy = _toy()
    source = toy.hessian
    observables = toy.observables

    def raw_unweighted_response(tangent: np.ndarray) -> np.ndarray:
        signal = sum(
            np.trace(observables[:, :, block] @ tangent[:, :, block]).real
            for block in range(source.nblock)
        )
        return toy.coupling * signal * observables

    default = ZeroTemperatureRaggedOrbitalHessian(
        source.projectors,
        source.hamiltonians,
        source.orbital_basis,
        source.occupied_counts,
        raw_unweighted_response,
    )
    explicit = ZeroTemperatureRaggedOrbitalHessian(
        source.projectors,
        source.hamiltonians,
        source.orbital_basis,
        source.occupied_counts,
        raw_unweighted_response,
        block_weights=np.ones(source.nblock),
    )
    direction = default.pack_real(_direction())
    assert np.array_equal(default.block_weights, np.ones(source.nblock))
    assert np.allclose(default.matvec(direction), explicit.matvec(direction))
    assert np.allclose(
        default.matvec(direction),
        default.pack_real(tuple(2.0 * block for block in default.complex_action(direction))),
    )


def test_inverted_full_empty_zero_dimension_means_no_retained_directions() -> None:
    n = 2
    basis = np.repeat(np.eye(n, dtype=np.complex128)[:, :, None], 2, axis=2)
    projectors = np.stack(
        (np.zeros((n, n), dtype=np.complex128), np.eye(n, dtype=np.complex128)),
        axis=2,
    )
    # Deliberately inverted global Aufbau order: moving an electron from the
    # full block to the empty block lowers the one-body energy.  The fixed-rank
    # block-preserving candidate cannot retain that transfer direction.
    hamiltonians = np.stack(
        (np.diag([-10.0, -9.0]), np.diag([9.0, 10.0])), axis=2
    )
    occupation_transfer_delta = hamiltonians[0, 0, 0] - hamiltonians[1, 1, 1]
    assert occupation_transfer_delta < 0.0

    def zero_response(tangent: np.ndarray) -> np.ndarray:
        assert tangent.shape == projectors.shape
        assert np.count_nonzero(tangent) == 0
        return np.zeros_like(tangent)

    candidate = ZeroTemperatureRaggedOrbitalHessian(
        projectors,
        hamiltonians,
        basis,
        (0, n),
        zero_response,
    )
    empty = np.empty(0, dtype=np.float64)
    assert candidate.complex_dimension == 0
    assert candidate.real_dimension == 0
    assert candidate.candidate_linear_operator.shape == (0, 0)
    assert candidate.pack_real(
        (np.empty((2, 0)), np.empty((0, 2)))
    ).shape == (0,)
    assert candidate.unpack_real(empty)[0].shape == (2, 0)
    assert candidate.unpack_real(empty)[1].shape == (0, 2)
    assert candidate.matvec(empty).shape == (0,)
    assert candidate.candidate_linear_operator.matvec(empty).shape == (0,)
    diagnostic = candidate.diagnose_bilinear_symmetry(probe_count=3)
    assert diagnostic.retained_real_dimension == 0
    assert diagnostic.inconclusive
    assert not diagnostic.all_evaluated_pairs_symmetric
    assert not diagnostic.asymmetry_detected
    assert diagnostic.outcome == "inconclusive"
    assert diagnostic.probes == ()
    assert candidate.requires_separate_occupation_gap_gate
    assert np.allclose(candidate.retract(empty, 0.7), projectors)


def test_rejects_invalid_ragged_counts_and_nonpositive_raw_weights() -> None:
    hessian = _toy().hessian
    with pytest.raises(ValueError, match=r"lie in \[0, 3\]"):
        ZeroTemperatureRaggedOrbitalHessian(
            hessian.projectors,
            hessian.hamiltonians,
            hessian.orbital_basis,
            (0, 1, 2, 4),
            hessian.fock_derivative,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        ZeroTemperatureRaggedOrbitalHessian(
            hessian.projectors,
            hessian.hamiltonians,
            hessian.orbital_basis,
            hessian.occupied_counts,
            hessian.fock_derivative,
            block_weights=(1.0, 0.0, 2.0, 3.0),
        )
