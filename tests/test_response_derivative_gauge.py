from __future__ import annotations

import numpy as np

from analysis.response_derivative_gauge import (
    projected_basis_connection,
    projected_basis_covariant_hamiltonian_derivative,
    projected_basis_covariant_velocity,
    random_block_unitary,
    trace_subspace,
)


def _plane_rotation(size: int, first: int, second: int, angle: float, angle_derivative: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a real plane rotation and its exact derivative."""

    rotation = np.eye(size, dtype=np.complex128)
    derivative = np.zeros((size, size), dtype=np.complex128)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation[first, first] = cosine
    rotation[first, second] = -sine
    rotation[second, first] = sine
    rotation[second, second] = cosine
    derivative[first, first] = -angle_derivative * sine
    derivative[first, second] = -angle_derivative * cosine
    derivative[second, first] = angle_derivative * cosine
    derivative[second, second] = -angle_derivative * sine
    return rotation, derivative


def test_projected_basis_covariant_velocity_matches_projected_full_velocity_for_moving_rectangular_basis():
    """Exact finite-dimensional P2.2 identity, including its sign.

    The three-column projected basis moves inside a four-dimensional full
    space and also undergoes a k-dependent internal frame rotation.  The full
    Hamiltonian is exactly block invariant in that moving projected subspace.
    """

    k = 0.37
    outer, d_outer = _plane_rotation(4, 0, 3, angle=0.41 * k, angle_derivative=0.41)
    inner, d_inner = _plane_rotation(3, 0, 1, angle=-0.63 * k, angle_derivative=-0.63)
    outer_projected = outer[:, :3]
    d_outer_projected = d_outer[:, :3]
    basis = outer_projected @ inner
    basis_derivative = d_outer_projected @ inner + outer_projected @ d_inner
    complement = outer[:, 3:4]
    complement_derivative = d_outer[:, 3:4]

    h_projected_input = np.asarray(
        [
            [0.20, 0.13 + 0.07j, -0.04j],
            [0.13 - 0.07j, -0.35, 0.09],
            [0.04j, 0.09, 0.82],
        ],
        dtype=np.complex128,
    )
    dh_projected_input = np.asarray(
        [
            [0.11, -0.03j, 0.02],
            [0.03j, -0.07, 0.05 + 0.01j],
            [0.02, 0.05 - 0.01j, 0.04],
        ],
        dtype=np.complex128,
    )
    complement_energy = 2.7

    full_hamiltonian = (
        basis @ h_projected_input @ basis.conjugate().T
        + complement_energy * complement @ complement.conjugate().T
    )
    full_hamiltonian_derivative = (
        basis_derivative @ h_projected_input @ basis.conjugate().T
        + basis @ dh_projected_input @ basis.conjugate().T
        + basis @ h_projected_input @ basis_derivative.conjugate().T
        + complement_energy * complement_derivative @ complement.conjugate().T
        + complement_energy * complement @ complement_derivative.conjugate().T
    )

    h_projected = basis.conjugate().T @ full_hamiltonian @ basis
    dh_projected = (
        basis_derivative.conjugate().T @ full_hamiltonian @ basis
        + basis.conjugate().T @ full_hamiltonian_derivative @ basis
        + basis.conjugate().T @ full_hamiltonian @ basis_derivative
    )[None, :, :]
    connection = projected_basis_connection(basis, basis_derivative[None, :, :])
    covariant_derivative = projected_basis_covariant_hamiltonian_derivative(
        h_projected,
        dh_projected,
        connection,
    )
    projected_full_derivative = (basis.conjugate().T @ full_hamiltonian_derivative @ basis)[None, :, :]

    np.testing.assert_allclose(basis.conjugate().T @ basis, np.eye(3), rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(connection, connection.swapaxes(1, 2).conjugate(), rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(covariant_derivative, projected_full_derivative, rtol=0.0, atol=3.0e-15)

    hbar = 2.5
    np.testing.assert_allclose(
        projected_basis_covariant_velocity(h_projected, dh_projected, connection, hbar=hbar),
        projected_full_derivative / hbar,
        rtol=0.0,
        atol=2.0e-15,
    )

    commutator = connection @ h_projected - h_projected @ connection
    wrong_sign_derivative = dh_projected + 1.0j * commutator
    assert np.linalg.norm(wrong_sign_derivative - projected_full_derivative) > 1.0e-2
    assert np.linalg.norm(dh_projected - projected_full_derivative) > 1.0e-2


def test_projected_basis_covariant_derivative_has_exact_degenerate_subspace_un_covariance():
    """A k-dependent random U(2) rotation is allowed only in an exact doublet."""

    h_projected = np.diag(np.asarray([-0.4, -0.4, 1.3], dtype=float)).astype(np.complex128)
    dh_projected = np.asarray(
        [
            [0.17, 0.06 + 0.02j, -0.08j],
            [0.06 - 0.02j, -0.03, 0.11],
            [0.08j, 0.11, 0.09],
        ],
        dtype=np.complex128,
    )
    connection_generator = np.asarray(
        [
            [0.21, 0.04j, 0.13],
            [-0.04j, -0.16, -0.07j],
            [0.13, 0.07j, 0.05],
        ],
        dtype=np.complex128,
    )
    basis = np.eye(3, dtype=np.complex128)
    basis_derivative = -1.0j * connection_generator
    connection = projected_basis_connection(basis, basis_derivative[None, :, :])
    covariant_derivative = projected_basis_covariant_hamiltonian_derivative(
        h_projected,
        dh_projected[None, :, :],
        connection,
    )

    groups = [(0, 2), (2, 3)]
    gauge = random_block_unitary(groups, 3, rng=20260713)
    gauge_generator = np.asarray(
        [
            [0.31, 0.09 - 0.04j, 0.0],
            [0.09 + 0.04j, -0.12, 0.0],
            [0.0, 0.0, 0.23],
        ],
        dtype=np.complex128,
    )
    gauge_derivative = -1.0j * gauge @ gauge_generator

    gauged_basis = basis @ gauge
    gauged_basis_derivative = basis_derivative @ gauge + basis @ gauge_derivative
    gauged_hamiltonian = gauge.conjugate().T @ h_projected @ gauge
    gauged_hamiltonian_derivative = (
        gauge_derivative.conjugate().T @ h_projected @ gauge
        + gauge.conjugate().T @ dh_projected @ gauge
        + gauge.conjugate().T @ h_projected @ gauge_derivative
    )[None, :, :]
    gauged_connection = projected_basis_connection(gauged_basis, gauged_basis_derivative[None, :, :])
    gauged_covariant_derivative = projected_basis_covariant_hamiltonian_derivative(
        gauged_hamiltonian,
        gauged_hamiltonian_derivative,
        gauged_connection,
    )

    expected_connection = (
        gauge.conjugate().T @ connection[0] @ gauge
        + 1.0j * gauge.conjugate().T @ gauge_derivative
    )[None, :, :]
    expected_covariant_derivative = (
        gauge.conjugate().T @ covariant_derivative[0] @ gauge
    )[None, :, :]
    np.testing.assert_allclose(gauged_connection, expected_connection, rtol=0.0, atol=3.0e-15)
    np.testing.assert_allclose(gauged_covariant_derivative, expected_covariant_derivative, rtol=0.0, atol=4.0e-15)
    np.testing.assert_allclose(
        trace_subspace(gauged_covariant_derivative[0], groups[0]),
        trace_subspace(covariant_derivative[0], groups[0]),
        rtol=0.0,
        atol=3.0e-15,
    )
