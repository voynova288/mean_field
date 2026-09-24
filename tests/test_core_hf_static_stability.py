from __future__ import annotations

from io import BytesIO
import json

import numpy as np
import pytest

from mean_field.core.hf.static_stability import (
    FermiFrechetResponse,
    SqrtUnderflowSandwichBound,
    StaticStabilityOperators,
    WeightedKMatrixSpace,
    build_static_stability_operators,
    compute_sqrt_underflow_sandwich_bound,
)


def _random_unitary(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trial = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(trial)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return q * phases.conj()[None, :]


def _response_from_energies(
    energies: np.ndarray,
    *,
    thermal_energy: float,
    mu: float = 0.0,
    eigenvectors: np.ndarray | None = None,
) -> FermiFrechetResponse:
    values = np.asarray(energies, dtype=float)
    if values.ndim != 2:
        raise ValueError("test energies must have shape (n, nk)")
    n, nk = values.shape
    if eigenvectors is None:
        vectors = np.repeat(
            np.eye(n, dtype=np.complex128)[:, :, None], nk, axis=2
        )
    else:
        vectors = np.asarray(eigenvectors, dtype=np.complex128)
    hamiltonian = np.einsum(
        "api,pi,bpi->abi",
        vectors,
        values,
        vectors.conj(),
        optimize=True,
    )
    return FermiFrechetResponse.from_eigensystem(
        hamiltonian,
        vectors,
        values,
        mu=mu,
        thermal_energy=thermal_energy,
    )


def _dense(linear_operator: object) -> np.ndarray:
    dimension = int(linear_operator.shape[0])
    identity = np.eye(dimension, dtype=np.complex128)
    return np.column_stack(
        [linear_operator.matvec(identity[:, column]) for column in range(dimension)]
    )


def _dagger(values: np.ndarray) -> np.ndarray:
    return np.swapaxes(np.asarray(values).conj(), 0, 1)


def test_scalar_and_two_level_fermi_frechet_sign_oracles() -> None:
    theta = 2.0
    scalar = _response_from_energies(
        np.asarray([[0.0]]), thermal_energy=theta
    )
    direction = np.asarray([[[3.0 - 2.0j]]], dtype=np.complex128)
    np.testing.assert_allclose(
        scalar.divided_differences[0, 0, 0],
        -1.0 / (4.0 * theta),
        rtol=0.0,
        atol=2e-16,
    )
    np.testing.assert_allclose(
        scalar.apply(direction),
        -direction / (4.0 * theta),
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        scalar.apply_negative_sqrt(direction),
        direction / np.sqrt(4.0 * theta),
        rtol=0.0,
        atol=2e-15,
    )

    coupling = 1.7
    operators = build_static_stability_operators(
        scalar,
        lambda value: -coupling * value,
        k_weights=np.asarray([0.3]),
    )
    expected_growth = coupling / (4.0 * theta)
    np.testing.assert_allclose(
        operators.apply_jacobian(direction),
        expected_growth * direction,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        operators.apply_self_adjoint(direction),
        expected_growth * direction,
        rtol=0.0,
        atol=2e-15,
    )

    two_level = _response_from_energies(
        np.asarray([[-1.0], [1.0]]), thermal_energy=0.5
    )
    f_minus = 1.0 / (1.0 + np.exp(-2.0))
    f_plus = 1.0 / (1.0 + np.exp(2.0))
    expected_off_diagonal = (f_minus - f_plus) / (-2.0)
    assert expected_off_diagonal < 0.0
    np.testing.assert_allclose(
        two_level.divided_differences[0, 1, 0],
        expected_off_diagonal,
        rtol=2e-15,
        atol=2e-15,
    )


def test_fermi_response_fails_closed_on_supplied_eigensystem_and_scale() -> None:
    energies = np.asarray([[0.0], [1.0]])
    vectors = np.eye(2, dtype=np.complex128)[:, :, None]
    hamiltonian = np.diag([0.0, 1.0]).astype(np.complex128)[:, :, None]

    nonhermitian = hamiltonian.copy()
    nonhermitian[0, 1, 0] = 1e-4j
    with pytest.raises(ValueError, match="not Hermitian"):
        FermiFrechetResponse.from_eigensystem(
            nonhermitian,
            vectors,
            energies,
            mu=0.0,
            thermal_energy=0.2,
        )

    nonunitary = vectors.copy()
    nonunitary[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="not unitary"):
        FermiFrechetResponse.from_eigensystem(
            hamiltonian,
            nonunitary,
            energies,
            mu=0.0,
            thermal_energy=0.2,
        )

    wrong_energies = energies.copy()
    wrong_energies[1, 0] += 1e-4
    with pytest.raises(ValueError, match="eigenpair residual"):
        FermiFrechetResponse.from_eigensystem(
            hamiltonian,
            vectors,
            wrong_energies,
            mu=0.0,
            thermal_energy=0.2,
        )

    nonfinite = hamiltonian.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must all be finite"):
        FermiFrechetResponse.from_eigensystem(
            nonfinite,
            vectors,
            energies,
            mu=0.0,
            thermal_energy=0.2,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        FermiFrechetResponse.from_eigensystem(
            hamiltonian,
            vectors,
            energies,
            mu=0.0,
            thermal_energy=0.0,
        )


def test_exact_degenerate_block_is_unitary_covariant_for_chi_and_sqrt() -> None:
    n = 3
    nk = 2
    energies = np.asarray(
        [[-0.4, 0.2], [-0.4, 0.2], [1.1, 1.4]], dtype=float
    )
    vectors = np.stack(
        [_random_unitary(n, seed=31 + ik) for ik in range(nk)], axis=2
    )
    response = _response_from_energies(
        energies,
        thermal_energy=0.37,
        mu=0.1,
        eigenvectors=vectors,
    )

    rotated = vectors.copy()
    for ik in range(nk):
        block_rotation = np.eye(n, dtype=np.complex128)
        block_rotation[:2, :2] = _random_unitary(2, seed=51 + ik)
        rotated[:, :, ik] = vectors[:, :, ik] @ block_rotation
    rotated_response = _response_from_energies(
        energies,
        thermal_energy=0.37,
        mu=0.1,
        eigenvectors=rotated,
    )

    rng = np.random.default_rng(71)
    value = rng.normal(size=(n, n, nk)) + 1j * rng.normal(size=(n, n, nk))
    np.testing.assert_allclose(
        rotated_response.apply(value),
        response.apply(value),
        rtol=5e-12,
        atol=5e-12,
    )
    np.testing.assert_allclose(
        rotated_response.apply_negative_sqrt(value),
        response.apply_negative_sqrt(value),
        rtol=5e-12,
        atol=5e-12,
    )


def test_near_degeneracy_is_continuous_without_a_gap_cutoff() -> None:
    theta = 0.7
    splitting = 2.0e-13
    response = _response_from_energies(
        np.asarray(
            [[-0.5 * splitting], [0.5 * splitting], [1.2]], dtype=float
        ),
        thermal_energy=theta,
    )
    expected_at_midpoint = 1.0 / (4.0 * theta)
    coefficient = response.negative_divided_differences[0, 1, 0]
    assert np.isfinite(coefficient)
    assert coefficient > 0.0
    np.testing.assert_allclose(
        coefficient,
        expected_at_midpoint,
        rtol=2e-13,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        response.negative_divided_differences,
        np.swapaxes(response.negative_divided_differences, 0, 1),
        rtol=0.0,
        atol=0.0,
    )


def test_log_domain_z_plus_minus_1000_preserves_representable_sqrt() -> None:
    response = _response_from_energies(
        np.asarray([[1000.0], [1000.0], [-1000.0], [-1000.0]]),
        thermal_energy=1.0,
    )
    assert np.all(np.isfinite(response.log_negative_divided_differences))
    assert np.all(np.isfinite(response.divided_differences))
    assert np.all(np.isfinite(response.sqrt_negative_divided_differences))

    # Same-side derivatives exp(-1000) underflow in L, but their square roots
    # exp(-500) are still representable and must be built independently.
    assert response.divided_differences[0, 1, 0] == 0.0
    assert response.divided_differences[2, 3, 0] == 0.0
    assert response.sqrt_negative_divided_differences[0, 1, 0] > 0.0
    assert response.sqrt_negative_divided_differences[2, 3, 0] > 0.0
    assert response.divided_difference_underflow_count > 0
    assert response.negative_sqrt_underflow_count == 0

    # Across the Fermi surface the divided difference is exactly the large-z
    # limit -(1-0)/(2000*theta).
    np.testing.assert_allclose(
        response.divided_differences[0, 2, 0],
        -1.0 / 2000.0,
        rtol=3e-12,
        atol=2e-15,
    )


def test_forced_sqrt_omission_dense_sandwich_spectral_norm_is_bounded() -> None:
    response = _response_from_energies(
        np.asarray([[-0.4], [0.9]]), thermal_energy=0.6, mu=0.1
    )
    exact_sqrt = response.sqrt_negative_divided_differences.copy()
    omitted_mask = np.zeros_like(exact_sqrt, dtype=bool)
    omitted_mask[0, 1, 0] = True
    omitted_mask[1, 0, 0] = True
    truncated_sqrt = exact_sqrt.copy()
    truncated_sqrt[omitted_mask] = 0.0
    truncated_sqrt.setflags(write=False)
    # Force a controlled, representable omission so the exact and truncated
    # dense sandwiches can both be formed in float64 for this algebraic oracle.
    object.__setattr__(
        response, "sqrt_negative_divided_differences", truncated_sqrt
    )

    dimension = exact_sqrt.size
    rng = np.random.default_rng(79)
    trial = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    b_euclidean = 0.5 * (trial + trial.conj().T)
    b_frobenius = float(np.linalg.norm(b_euclidean, ord="fro"))
    bound = compute_sqrt_underflow_sandwich_bound(
        response, interaction_frobenius_norm=b_frobenius
    )
    assert isinstance(bound, SqrtUnderflowSandwichBound)
    np.testing.assert_array_equal(bound.omitted_mask, omitted_mask)
    np.testing.assert_array_equal(
        bound.omitted_indices, np.argwhere(omitted_mask)
    )
    np.testing.assert_array_equal(
        bound.omitted_logs,
        response.log_negative_divided_differences[omitted_mask],
    )
    assert not bound.omitted_mask.flags.writeable
    assert not bound.omitted_indices.flags.writeable
    assert not bound.omitted_logs.flags.writeable

    exact_diagonal = np.ravel(exact_sqrt, order="C")
    truncated_diagonal = np.ravel(truncated_sqrt, order="C")
    exact_c = (
        exact_diagonal[:, None]
        * b_euclidean
        * exact_diagonal[None, :]
    )
    truncated_c = (
        truncated_diagonal[:, None]
        * b_euclidean
        * truncated_diagonal[None, :]
    )
    spectral_error = float(
        np.linalg.norm(exact_c - truncated_c, ord=2)
    )
    assert spectral_error <= bound.delta
    expected_log_delta = (
        np.log(2.0)
        + 0.5 * (bound.ell_max + bound.ell_Z_max)
        + np.log(b_frobenius)
    )
    np.testing.assert_allclose(
        bound.log_delta, expected_log_delta, rtol=0.0, atol=2e-15
    )

def test_sqrt_underflow_bound_keeps_extreme_logs_and_maximum_omission() -> None:
    response = _response_from_energies(
        np.asarray([[0.0, 1581.0, 1603.0]]),
        thermal_energy=1.0,
    )
    assert response.negative_sqrt_underflow_count == 2
    with np.errstate(over="raise", under="raise", invalid="raise"):
        bound = compute_sqrt_underflow_sandwich_bound(
            response,
            log_interaction_frobenius_norm=-900.0,
        )

    np.testing.assert_array_equal(
        bound.omitted_indices,
        np.asarray([[0, 0, 1], [0, 0, 2]], dtype=np.intp),
    )
    np.testing.assert_allclose(
        bound.omitted_logs,
        np.asarray([-1581.0, -1603.0]),
        rtol=0.0,
        atol=2e-12,
    )
    assert bound.ell_Z_max == float(np.max(bound.omitted_logs))
    assert bound.ell_Z_max > float(np.min(bound.omitted_logs))
    assert bound.log_Bfro == -900.0
    assert np.isfinite(bound.log_delta)
    assert bound.log_delta == pytest.approx(
        np.log(2.0)
        + 0.5 * (bound.ell_max + bound.ell_Z_max)
        + bound.log_Bfro
    )
    assert bound.delta == 0.0

def test_sqrt_underflow_bound_json_roundtrip_for_positive_omission() -> None:
    response = _response_from_energies(
        np.asarray([[0.0, 1581.0, 1603.0]]),
        thermal_energy=1.0,
    )
    bound = compute_sqrt_underflow_sandwich_bound(
        response,
        log_interaction_frobenius_norm=-900.0,
    )

    record = bound.to_json_dict()
    assert record["ell_Z_max"] == bound.ell_Z_max
    assert record["ell_Z_max_state"] == "finite"
    assert record["log_delta_uf"] == bound.log_delta_uf
    assert record["log_delta_uf_state"] == "finite"
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_sqrt_underflow_bound_no_omission_json_and_npz_roundtrip() -> None:
    response = _response_from_energies(
        np.asarray([[0.0]]), thermal_energy=1.0
    )
    bound = compute_sqrt_underflow_sandwich_bound(
        response,
        log_interaction_frobenius_norm=-900.0,
    )
    assert bound.omitted_count == 0
    assert bound.omitted_indices.shape == (0, 3)
    assert bound.omitted_logs.shape == (0,)
    assert bound.ell_Z_max == -np.inf
    assert bound.log_delta == -np.inf
    assert bound.log_delta_uf == -np.inf
    assert bound.delta == 0.0

    record = bound.to_json_dict()
    assert record["ell_Z_max"] is None
    assert record["ell_Z_max_state"] == "negative_infinity_no_omission"
    assert record["log_delta_uf"] is None
    assert record["log_delta_uf_state"] == "negative_infinity_no_omission"
    assert json.loads(json.dumps(record, allow_nan=False)) == record

    archive_buffer = BytesIO()
    np.savez(
        archive_buffer,
        ell_Z_max=np.asarray(bound.ell_Z_max),
        log_delta_uf=np.asarray(bound.log_delta_uf),
    )
    with np.load(BytesIO(archive_buffer.getvalue()), allow_pickle=False) as archive:
        assert float(np.asarray(archive["ell_Z_max"]).item()) == -np.inf
        assert float(np.asarray(archive["log_delta_uf"]).item()) == -np.inf

def test_weighted_matrix_space_is_exact_unequal_weight_euclidean_isometry() -> None:
    weights = np.asarray([0.07, 0.9, 3.4])
    space = WeightedKMatrixSpace(matrix_dimension=2, k_weights=weights)
    rng = np.random.default_rng(83)
    left = rng.normal(size=space.matrix_shape) + 1j * rng.normal(
        size=space.matrix_shape
    )
    right = rng.normal(size=space.matrix_shape) + 1j * rng.normal(
        size=space.matrix_shape
    )

    expected_flat = np.ravel(
        left * np.sqrt(weights)[None, None, :], order="C"
    )
    np.testing.assert_array_equal(space.to_euclidean(left), expected_flat)
    np.testing.assert_allclose(
        space.from_euclidean(space.to_euclidean(left)),
        left,
        rtol=0.0,
        atol=2e-16,
    )
    expected_inner = np.einsum(
        "i,adi,adi->", weights, left.conj(), right, optimize=True
    )
    np.testing.assert_allclose(
        np.vdot(space.to_euclidean(left), space.to_euclidean(right)),
        expected_inner,
        rtol=2e-15,
        atol=2e-15,
    )


def test_chi_and_negative_sqrt_are_weighted_self_adjoint_and_dagger_covariant() -> None:
    n = 3
    nk = 2
    energies = np.asarray(
        [[-0.8, -0.3], [0.1, 0.4], [1.3, 1.7]], dtype=float
    )
    vectors = np.stack(
        [_random_unitary(n, seed=101 + ik) for ik in range(nk)], axis=2
    )
    response = _response_from_energies(
        energies,
        thermal_energy=0.41,
        mu=0.05,
        eigenvectors=vectors,
    )
    space = WeightedKMatrixSpace(
        matrix_dimension=n, k_weights=np.asarray([0.13, 1.7])
    )
    rng = np.random.default_rng(109)
    left = rng.normal(size=(n, n, nk)) + 1j * rng.normal(size=(n, n, nk))
    right = rng.normal(size=(n, n, nk)) + 1j * rng.normal(size=(n, n, nk))

    for action in (response.apply, response.apply_negative_sqrt):
        lhs = space.inner(left, action(right))
        rhs = space.inner(action(left), right)
        np.testing.assert_allclose(lhs, rhs, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(
            action(_dagger(right)),
            _dagger(action(right)),
            rtol=2e-12,
            atol=2e-12,
        )


def test_global_canonical_control_removes_only_the_number_direction() -> None:
    response = _response_from_energies(
        np.asarray([[-0.6, -0.2], [0.7, 1.1]]),
        thermal_energy=0.3,
        mu=0.1,
    )
    weights = np.asarray([0.2, 1.3])
    identity = np.repeat(np.eye(2)[:, :, None], 2, axis=2).astype(np.complex128)
    constant = 0.37 - 0.12j
    np.testing.assert_allclose(
        response.canonical_delta_mu(constant * identity, weights),
        constant,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        response.apply_canonical(constant * identity, weights),
        0.0,
        rtol=0.0,
        atol=2e-14,
    )

    operators = build_static_stability_operators(
        response,
        lambda value: -value,
        k_weights=weights,
        number_constraint="global",
    )
    assert isinstance(operators, StaticStabilityOperators)
    assert operators.constraint_vector_euclidean is not None
    constraint = operators.constraint_vector_euclidean
    np.testing.assert_allclose(
        operators.C.matvec(constraint), 0.0, rtol=0.0, atol=2e-14
    )

    rng = np.random.default_rng(127)
    value = rng.normal(size=response.matrix_shape) + 1j * rng.normal(
        size=response.matrix_shape
    )
    output = operators.apply_jacobian(value)
    number_response = np.einsum(
        "i,aai->", weights, output, optimize=True
    )
    np.testing.assert_allclose(number_response, 0.0, rtol=0.0, atol=2e-13)


def test_dense_j_and_c_have_matching_nonzero_spectra_and_c_is_self_adjoint() -> None:
    n = 2
    nk = 2
    energies = np.asarray([[-0.9, -0.25], [0.45, 1.2]])
    vectors = np.stack(
        [_random_unitary(n, seed=149 + ik) for ik in range(nk)], axis=2
    )
    response = _response_from_energies(
        energies,
        thermal_energy=0.36,
        mu=0.08,
        eigenvectors=vectors,
    )
    weights = np.asarray([0.11, 1.9])
    space = WeightedKMatrixSpace(matrix_dimension=n, k_weights=weights)

    rng = np.random.default_rng(157)
    trial = rng.normal(size=(space.size, space.size)) + 1j * rng.normal(
        size=(space.size, space.size)
    )
    fock_euclidean = 0.12 * (trial + trial.conj().T)

    def fock_action(value: np.ndarray) -> np.ndarray:
        return space.from_euclidean(
            fock_euclidean @ space.to_euclidean(value)
        )

    operators = build_static_stability_operators(
        response,
        fock_action,
        k_weights=weights,
        number_constraint="none",
    )
    dense_j = _dense(operators.J)
    dense_c = _dense(operators.C)
    self_adjoint_residual = np.max(np.abs(dense_c - dense_c.conj().T))
    assert self_adjoint_residual <= 1e-10

    j_eigenvalues = np.linalg.eigvals(dense_j)
    c_eigenvalues = np.linalg.eigvalsh(dense_c)
    assert np.max(np.abs(j_eigenvalues.imag)) <= (
        1e-10 * max(1.0, float(np.max(np.abs(j_eigenvalues))))
    )
    np.testing.assert_allclose(
        np.sort(j_eigenvalues.real),
        np.sort(c_eigenvalues),
        rtol=2e-9,
        atol=2e-11,
    )
