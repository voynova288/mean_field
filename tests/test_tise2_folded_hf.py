from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf.occupations import conventional_projector_to_stored
from mean_field.systems.tise2.folded_hf import (
    DOUBLED_Q_LABELS,
    ELECTRONS_PER_REDUCED_MOMENTUM,
    FOLDED_DIMENSION,
    FoldedHFConfig,
    FoldedTiSe2Geometry,
    Q_FRACTIONAL,
    _entropy_gradient_stored,
    all_hermitian_direction_derivative_oracle,
    build_folded_hf_state,
    diagonalize_folded_finite_temperature,
    estimate_folded_coulomb_peak_bytes,
    folded_fock_action,
    folded_h0,
    folded_hartree_action,
    folded_interaction_action,
    folded_reference_relative_energy,
    folded_reference_relative_free_energy,
    global_primitive_h0,
    physical_momentum_routes,
    physical_q_vectors,
    precompute_folded_coulomb_kernel,
    run_folded_hf,
)
from mean_field.systems.tise2.hf import (
    COULOMB_EV_ANGSTROM,
    MonneyLocalMesh,
    build_monney_local_mesh,
)
from mean_field.systems.tise2.monney import MonneyTiSe2Parameters


def _one_point_problem(epsilon_r: float = 5.0):
    mesh = build_monney_local_mesh(
        inplane_shells=0,
        inplane_spacing_Ainv=0.04,
        z_shells=0,
        z_spacing_Ainv=0.05,
    )
    q_vectors = physical_q_vectors()
    coulomb = precompute_folded_coulomb_kernel(
        mesh,
        q_vectors_Ainv=q_vectors,
        epsilon_r=epsilon_r,
        max_dense_memory_gb=0.01,
    )
    return mesh, q_vectors, coulomb


def _random_hermitian(nk: int, seed: int = 4, scale: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(FOLDED_DIMENSION, FOLDED_DIMENSION, nk))
    raw = raw + 1j * rng.normal(size=raw.shape)
    return scale * (raw + raw.conj().swapaxes(0, 1))


def _literal_route_is_retained(s: int, t: int, u: int, v: int) -> bool:
    residual = (
        DOUBLED_Q_LABELS[s]
        + DOUBLED_Q_LABELS[u]
        - DOUBLED_Q_LABELS[t]
        - DOUBLED_Q_LABELS[v]
    )
    return bool(np.array_equal(residual, np.zeros(3, dtype=np.int64)))


def _literal_wick_action(
    full_projector: np.ndarray,
    density_delta: np.ndarray,
    *,
    mesh: MonneyLocalMesh,
    q_vectors: np.ndarray,
    epsilon_r: float,
    self_cell_ev: float,
) -> np.ndarray:
    """Independent literal four-index, all-k density-vertex/Wick contraction.

    This deliberately does not call a production route builder or consume a
    production shifted/Hartree table.  It enumerates all sector, internal, and
    source/target-k indices, including the literal source quadrature weight.
    """

    prefactor = 4.0 * np.pi * COULOMB_EV_ANGSTROM / epsilon_r

    def weighted_fock_coulomb(transfer: np.ndarray, source_k: int) -> float:
        q2 = float(np.dot(transfer, transfer))
        if q2 <= 64.0 * np.finfo(float).eps:
            return self_cell_ev
        return float(mesh.weights_Ainv3[source_k]) * prefactor / q2

    def hartree_coulomb(transfer: np.ndarray) -> float:
        q2 = float(np.dot(transfer, transfer))
        if q2 <= 64.0 * np.finfo(float).eps:
            return 0.0  # neutral q=0 background
        return prefactor / q2

    sigma = np.zeros_like(full_projector)
    for s in range(4):
        for t in range(4):
            hartree_coefficient = hartree_coulomb(q_vectors[s] - q_vectors[t])
            for a in range(4):
                for b in range(4):
                    target_row = 4 * s + a
                    target_col = 4 * t + b
                    for target_k in range(mesh.nk):
                        value = 0.0j
                        for u in range(4):
                            for v in range(4):
                                if not _literal_route_is_retained(s, t, u, v):
                                    continue
                                for source_k in range(mesh.nk):
                                    transfer = (
                                        mesh.points_Ainv[target_k]
                                        - mesh.points_Ainv[source_k]
                                        + q_vectors[s]
                                        - q_vectors[v]
                                    )
                                    value -= weighted_fock_coulomb(
                                        transfer, source_k
                                    ) * full_projector[
                                        4 * u + b, 4 * v + a, source_k
                                    ]
                                    if a == b:
                                        for internal in range(4):
                                            value += (
                                                hartree_coefficient
                                                * mesh.weights_Ainv3[source_k]
                                                * density_delta[
                                                    4 * u + internal,
                                                    4 * v + internal,
                                                    source_k,
                                                ]
                                            )
                        sigma[target_row, target_col, target_k] = value
    return sigma


def test_tise2_package_exports_unrestricted_folded_api() -> None:
    from mean_field.systems import tise2

    assert tise2.FoldedHFConfig is FoldedHFConfig
    assert tise2.build_folded_hf_state is build_folded_hf_state
    assert tise2.run_folded_hf is run_folded_hf
    assert tise2.folded_h0 is folded_h0


def test_physical_q_vectors_are_the_declared_three_l_points() -> None:
    geometry = FoldedTiSe2Geometry(a_angstrom=3.54, c_angstrom=6.008)
    reciprocal = geometry.reciprocal_basis_Ainv
    q_vectors = physical_q_vectors(geometry)

    assert q_vectors.shape == (4, 3)
    assert np.array_equal(q_vectors[0], np.zeros(3))
    assert np.array_equal(DOUBLED_Q_LABELS, np.rint(2.0 * Q_FRACTIONAL).astype(int))
    assert np.allclose(q_vectors, Q_FRACTIONAL @ reciprocal)
    assert np.allclose(
        np.linalg.norm(q_vectors[1:, :2], axis=1),
        2.0 * np.pi / (np.sqrt(3.0) * geometry.a_angstrom),
    )
    assert np.allclose(q_vectors[1:, 2], np.pi / geometry.c_angstrom)


def test_global_h0_has_the_requested_gamma_and_l_centers() -> None:
    params = MonneyTiSe2Parameters()
    q_vectors = physical_q_vectors()
    h0 = global_primitive_h0(q_vectors, params=params, q_vectors_Ainv=q_vectors)

    assert h0.shape == (4, 4, 4)
    diagonal_h0 = np.diagonal(h0, axis1=0, axis2=1).T[:, None, :] * np.eye(4)[:, :, None]
    assert np.allclose(h0 - diagonal_h0, 0.0)
    assert np.isclose(h0[0, 0, 0], params.epsilon_v0_ev + params.t_v_ev)
    expected_c = params.epsilon_c0_ev + params.t_c_ev
    for internal in range(1, 4):
        assert np.isclose(h0[internal, internal, internal], expected_c)


def test_folded_h0_is_the_direct_sum_of_four_complete_primitive_blocks() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.007, -0.011, 0.013]], dtype=float)
    q_vectors = physical_q_vectors()
    folded = folded_h0(points, q_vectors_Ainv=q_vectors)

    assert folded.shape == (16, 16, 2)
    for sector in range(4):
        expected = global_primitive_h0(points + q_vectors[sector], q_vectors_Ainv=q_vectors)
        block = slice(4 * sector, 4 * (sector + 1))
        assert np.allclose(folded[block, block], expected)
        outside = folded[block].copy()
        outside[:, block, :] = 0.0
        assert np.array_equal(outside, np.zeros_like(outside))


def test_route_inventory_uses_exact_fixed_representative_g0_conservation() -> None:
    routes = physical_momentum_routes()
    assert len(routes) == 4
    assert all(
        len(routes[s][t]) == (4 if s == t else 1)
        for s in range(4)
        for t in range(4)
    )
    assert sum(len(routes[s][t]) for s in range(4) for t in range(4)) == 28

    for s in range(4):
        for t in range(4):
            for u, v in routes[s][t]:
                residual = DOUBLED_Q_LABELS[s] + DOUBLED_Q_LABELS[u]
                residual -= DOUBLED_Q_LABELS[t] + DOUBLED_Q_LABELS[v]
                assert np.array_equal(residual, np.zeros(3, dtype=np.int64))

    # This quartet is conserved only modulo a primitive reciprocal vector and
    # must remain excluded until a G-resolved Umklapp inventory is supplied.
    modulo_g_only = (0, 1, 0, 1)
    s, t, u, v = modulo_g_only
    residual = DOUBLED_Q_LABELS[s] + DOUBLED_Q_LABELS[u]
    residual -= DOUBLED_Q_LABELS[t] + DOUBLED_Q_LABELS[v]
    assert np.all(np.remainder(residual, 2) == 0)
    assert np.any(residual != 0)
    assert (u, v) not in routes[s][t]


def test_full_fock_and_hartree_match_independent_literal_wick_oracle() -> None:
    mesh, q_vectors, coulomb = _one_point_problem()
    assert coulomb.route_policy == "fixed_representative_g0"
    assert coulomb.route_count == 28
    full_projector = 0.35 * np.eye(16, dtype=np.complex128)[:, :, None]
    full_projector += _random_hermitian(1, seed=9, scale=0.002)
    density_delta = _random_hermitian(1, seed=17, scale=0.003)

    actual = folded_hartree_action(density_delta, coulomb=coulomb)
    actual += folded_fock_action(full_projector, coulomb=coulomb)
    expected = _literal_wick_action(
        full_projector,
        density_delta,
        mesh=mesh,
        q_vectors=q_vectors,
        epsilon_r=coulomb.epsilon_r,
        self_cell_ev=coulomb.self_cell_ev,
    )

    assert np.allclose(actual, expected, rtol=3.0e-14, atol=3.0e-14)
    assert np.max(np.abs(actual - actual.conj().swapaxes(0, 1))) < 3.0e-13


def test_every_sector_block_impulse_reaches_every_allowed_output_block() -> None:
    mesh, _q_vectors, coulomb = _one_point_problem()
    for u in range(4):
        for v in range(4):
            impulse = np.zeros((16, 16, mesh.nk), dtype=np.complex128)
            source = slice(4 * u, 4 * (u + 1))
            source_col = slice(4 * v, 4 * (v + 1))
            impulse[source, source_col, 0] = (1.0 + 0.37j) * np.eye(4)
            impulse[source_col, source, 0] = (1.0 - 0.37j) * np.eye(4)
            if u == v:
                impulse[source, source, 0] = np.eye(4)
            sigma = folded_fock_action(impulse, coulomb=coulomb)

            for s in range(4):
                for t in range(4):
                    allowed = _literal_route_is_retained(s, t, u, v) or _literal_route_is_retained(
                        s, t, v, u
                    )
                    block = sigma[4 * s : 4 * (s + 1), 4 * t : 4 * (t + 1), 0]
                    if allowed:
                        assert np.max(np.abs(block)) > 0.0


def test_global_finite_temperature_occupation_is_not_rank_four_per_k() -> None:
    hamiltonian = np.zeros((16, 16, 2), dtype=np.complex128)
    hamiltonian[:, :, 0] = np.diag(np.linspace(-2.0, -1.0, 16))
    hamiltonian[:, :, 1] = np.diag(np.linspace(1.0, 2.0, 16))
    spectrum = diagonalize_folded_finite_temperature(
        hamiltonian,
        kbt_ev=0.01,
        k_weights=np.asarray([0.5, 0.5]),
    )

    local_traces = np.einsum("aak->k", spectrum.projector_stored).real
    assert np.isclose(np.mean(local_traces), ELECTRONS_PER_REDUCED_MOMENTUM, atol=2.0e-11)
    assert local_traces[0] > 7.9
    assert local_traces[1] < 1.0e-20
    assert not np.allclose(local_traces, 4.0)
    assert spectrum.energies_ev.shape == (16, 2)
    assert spectrum.eigenvectors.shape == (2, 16, 16)
    assert spectrum.occupations.shape == (16, 2)
    assert np.allclose(
        spectrum.eigenvectors.conj().transpose(0, 2, 1) @ spectrum.eigenvectors,
        np.eye(16)[None, :, :],
        atol=2.0e-14,
    )


@pytest.mark.parametrize("workers", [True, 0, -1, 1.5])
def test_folded_config_rejects_nonpositive_or_nonexact_workers(workers: object) -> None:
    with pytest.raises(TypeError, match="exact positive integer"):
        FoldedHFConfig(epsilon_r=5.0, eigensolver_workers=workers)  # type: ignore[arg-type]


def test_parallel_folded_diagonalization_requires_one_thread_numerical_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hamiltonian = np.zeros((16, 16, 2), dtype=np.complex128)
    hamiltonian[:, :, 0] = np.diag(np.linspace(-1.7, 1.3, 16))
    hamiltonian[:, :, 1] = np.diag(np.linspace(-1.2, 1.8, 16))

    assert FoldedHFConfig(epsilon_r=5.0, eigensolver_workers=64).eigensolver_workers == 64
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with pytest.raises(RuntimeError, match="one-thread"):
        diagonalize_folded_finite_temperature(
            hamiltonian, kbt_ev=0.02, eigensolver_workers=2
        )

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="MKL_NUM_THREADS"):
        diagonalize_folded_finite_temperature(
            hamiltonian, kbt_ev=0.02, eigensolver_workers=2
        )

    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("BLIS_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="BLIS_NUM_THREADS"):
        diagonalize_folded_finite_temperature(
            hamiltonian, kbt_ev=0.02, eigensolver_workers=2
        )


def test_parallel_folded_diagonalization_is_deterministic_and_keeps_all_16_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("BLIS_NUM_THREADS", raising=False)
    hamiltonian = np.zeros((16, 16, 2), dtype=np.complex128)
    hamiltonian[:, :, 0] = np.diag(np.linspace(-1.7, 1.3, 16))
    hamiltonian[:, :, 1] = np.diag(np.linspace(-1.2, 1.8, 16))

    first = diagonalize_folded_finite_temperature(
        hamiltonian, kbt_ev=0.02, eigensolver_workers=64
    )
    second = diagonalize_folded_finite_temperature(
        hamiltonian, kbt_ev=0.02, eigensolver_workers=64
    )

    assert first.energies_ev.shape == (16, 2)
    assert first.eigenvectors.shape == (2, 16, 16)
    assert np.array_equal(first.energies_ev, second.energies_ev)
    assert np.array_equal(first.eigenvectors, second.eigenvectors)
    assert np.array_equal(first.projector_stored, second.projector_stored)


def test_asymmetric_two_k_full_action_matches_literal_weighted_wick_oracle() -> None:
    mesh = MonneyLocalMesh(
        points_Ainv=np.asarray(
            [[0.0, 0.0, 0.0], [0.013, -0.007, 0.009]], dtype=np.float64
        ),
        weights_Ainv3=np.asarray([1.3e-6, 3.7e-6], dtype=np.float64),
        labels=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
        inplane_shells=1,
        inplane_spacing_Ainv=0.04,
        z_shells=0,
        z_spacing_Ainv=0.05,
        cell_volume_Ainv3=2.0e-4,
    )
    q_vectors = physical_q_vectors()
    coulomb = precompute_folded_coulomb_kernel(
        mesh,
        q_vectors_Ainv=q_vectors,
        epsilon_r=6.0,
        max_dense_memory_gb=0.01,
    )
    full_projector = 0.35 * np.eye(16, dtype=np.complex128)[:, :, None]
    full_projector = np.broadcast_to(full_projector, (16, 16, 2)).copy()
    full_projector += _random_hermitian(2, seed=71, scale=0.002)
    density_delta = _random_hermitian(2, seed=72, scale=0.003)

    actual = folded_hartree_action(density_delta, coulomb=coulomb)
    actual += folded_fock_action(full_projector, coulomb=coulomb)
    expected = _literal_wick_action(
        full_projector,
        density_delta,
        mesh=mesh,
        q_vectors=q_vectors,
        epsilon_r=coulomb.epsilon_r,
        self_cell_ev=coulomb.self_cell_ev,
    )

    assert not np.isclose(
        coulomb.shifted_weighted_ev[0, 0, 0, 1],
        coulomb.shifted_weighted_ev[0, 0, 1, 0],
    )
    assert np.allclose(actual, expected, rtol=7.0e-14, atol=7.0e-14)
    assert np.max(np.abs(actual - actual.conj().swapaxes(0, 1))) < 5.0e-13


def test_full_fock_affine_identity_and_physical_hamiltonian_decomposition() -> None:
    mesh, _q_vectors, coulomb = _one_point_problem()
    reference = 0.4 * np.eye(16, dtype=np.complex128)[:, :, None]
    reference += _random_hermitian(mesh.nk, seed=31, scale=0.001)
    density_delta = _random_hermitian(mesh.nk, seed=32, scale=0.002)
    bare_h0 = _random_hermitian(mesh.nk, seed=33, scale=0.01)

    fock_full = folded_fock_action(reference + density_delta, coulomb=coulomb)
    fock_reference = folded_fock_action(reference, coulomb=coulomb)
    fock_delta = folded_fock_action(density_delta, coulomb=coulomb)
    assert np.allclose(fock_full, fock_reference + fock_delta, rtol=3.0e-14, atol=3.0e-14)

    physical = bare_h0 + folded_hartree_action(density_delta, coulomb=coulomb) + fock_full
    h0_eff = bare_h0 + fock_reference
    affine = h0_eff + folded_interaction_action(density_delta, coulomb=coulomb)
    assert np.allclose(physical, affine, rtol=3.0e-14, atol=3.0e-14)


def test_full_relative_free_energy_passes_every_hermitian_direction() -> None:
    mesh, _q_vectors, coulomb = _one_point_problem()
    reference = 0.5 * np.eye(16, dtype=np.complex128)[:, :, None]
    density_delta = _random_hermitian(mesh.nk, seed=41, scale=0.001)
    h0_eff = _random_hermitian(mesh.nk, seed=42, scale=0.01)
    kbt_ev = 0.02

    result = all_hermitian_direction_derivative_oracle(
        density_delta,
        coulomb=coulomb,
        h0_effective_ev=h0_eff,
        reference_projector_stored=reference,
        kbt_ev=kbt_ev,
        finite_difference_step=1.0e-6,
        absolute_tolerance=8.0e-10,
        relative_tolerance=5.0e-7,
    )

    assert result.direction_count == mesh.nk * FOLDED_DIMENSION**2
    assert result.passed
    zero = np.zeros_like(density_delta)
    assert folded_reference_relative_energy(zero, h0_eff, zero, mesh=mesh) == 0.0
    assert (
        folded_reference_relative_free_energy(
            zero,
            h0_eff,
            zero,
            reference_projector_stored=reference,
            mesh=mesh,
            kbt_ev=kbt_ev,
        )
        == 0.0
    )


def test_parallel_free_energy_entropy_restores_physical_weight_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = MonneyLocalMesh(
        points_Ainv=np.asarray(
            [[0.0, 0.0, 0.0], [0.013, -0.007, 0.009]], dtype=np.float64
        ),
        weights_Ainv3=np.asarray([1.3e-6, 3.7e-6], dtype=np.float64),
        labels=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
        inplane_shells=1,
        inplane_spacing_Ainv=0.04,
        z_shells=0,
        z_spacing_Ainv=0.05,
        cell_volume_Ainv3=2.0e-4,
    )
    reference = np.repeat(
        (0.5 * np.eye(16, dtype=np.complex128))[:, :, None], 2, axis=2
    )
    projector = reference.copy()
    projector[:, :, 0] = 0.2 * np.eye(16)
    projector[:, :, 1] = 0.7 * np.eye(16)
    density_delta = projector - reference
    zero = np.zeros_like(reference)
    kbt_ev = 0.02

    entropy_terms = np.asarray(
        [
            16.0 * (value * np.log(value) + (1.0 - value) * np.log(1.0 - value))
            for value in (0.2, 0.7)
        ]
    )
    reference_term = 16.0 * np.log(0.5)
    expected = kbt_ev * float(
        np.dot(mesh.weights_Ainv3, entropy_terms - reference_term)
    )
    serial = folded_reference_relative_free_energy(
        zero,
        zero,
        density_delta,
        reference_projector_stored=reference,
        mesh=mesh,
        kbt_ev=kbt_ev,
    )

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("BLIS_NUM_THREADS", raising=False)
    parallel = folded_reference_relative_free_energy(
        zero,
        zero,
        density_delta,
        reference_projector_stored=reference,
        mesh=mesh,
        kbt_ev=kbt_ev,
        eigensolver_workers=2,
    )

    assert serial == pytest.approx(expected, rel=2.0e-15, abs=1.0e-30)
    assert parallel == serial


def test_entropy_gradient_uses_stored_orientation_for_complex_direction() -> None:
    mesh, _q_vectors, _coulomb = _one_point_problem()
    reference = 0.5 * np.eye(FOLDED_DIMENSION, dtype=np.complex128)[:, :, None]
    reference[0, 1, 0] = 0.1j
    reference[1, 0, 0] = -0.1j
    direction = np.zeros_like(reference)
    direction[0, 1, 0] = 1.0j / np.sqrt(2.0)
    direction[1, 0, 0] = -1.0j / np.sqrt(2.0)
    gradient = _entropy_gradient_stored(reference, mesh)

    step = 1.0e-6
    zero = np.zeros_like(reference)
    plus = folded_reference_relative_free_energy(
        zero,
        zero,
        step * direction,
        reference_projector_stored=reference,
        mesh=mesh,
        kbt_ev=1.0,
    )
    minus = folded_reference_relative_free_energy(
        zero,
        zero,
        -step * direction,
        reference_projector_stored=reference,
        mesh=mesh,
        kbt_ev=1.0,
    )
    numerical = (plus - minus) / (2.0 * step)
    analytic = np.einsum(
        "abk,abk,k->", gradient, direction, mesh.weights_Ainv3, optimize=True
    ).real

    assert abs(numerical) > 1.0e-8
    assert numerical == pytest.approx(analytic, rel=2.0e-9, abs=2.0e-13)


def test_reference_fock_produces_a_nontrivial_first_physical_action() -> None:
    config = FoldedHFConfig(
        epsilon_r=8.0,
        inplane_shells=0,
        z_shells=0,
        max_dense_memory_gb=0.01,
    )
    state = build_folded_hf_state(config)
    assert config.route_policy == "fixed_representative_g0"
    assert state.coulomb.route_policy == config.route_policy
    first = diagonalize_folded_finite_temperature(
        state.h0_effective_ev,
        kbt_ev=config.kbt_ev,
        k_weights=state.mesh.normalized_weights,
    )
    density_delta = first.projector_stored - state.reference_projector_stored
    density_action = folded_interaction_action(density_delta, coulomb=state.coulomb)

    assert np.max(np.abs(state.reference_fock_ev)) > 0.0
    assert np.max(np.abs(density_delta)) > 1.0e-12
    assert np.max(np.abs(density_action)) > 1.0e-12
    physical = state.bare_h0_ev + folded_hartree_action(
        density_delta, coulomb=state.coulomb
    ) + folded_fock_action(first.projector_stored, coulomb=state.coulomb)
    assert np.allclose(physical, state.h0_effective_ev + density_action)
    assert not np.allclose(physical, state.h0_effective_ev)


def test_supplied_q_vectors_must_match_the_doubled_label_route_predicate() -> None:
    mesh, q_vectors, _coulomb = _one_point_problem()
    mismatched = q_vectors.copy()
    mismatched[3] = mismatched[1] + mismatched[2]

    with pytest.raises(ValueError, match="zero-residual routes.*DOUBLED_Q_LABELS"):
        precompute_folded_coulomb_kernel(
            mesh,
            q_vectors_Ainv=mismatched,
            epsilon_r=5.0,
            max_dense_memory_gb=0.01,
        )


def test_one_point_run_folded_hf_closes_final_state_contracts() -> None:
    config = FoldedHFConfig(
        epsilon_r=100.0,
        temperature_K=300.0,
        inplane_shells=0,
        z_shells=0,
        precision=5.0e-9,
        max_iter=20,
        max_dense_memory_gb=0.01,
    )
    state = build_folded_hf_state(config)
    result = run_folded_hf(state, init_mode="normal")

    assert result.run.converged
    assert result.run.exit_reason == "converged"
    assert result.run.iter_oda.size == result.run.iter_energy.size
    assert np.all(np.isfinite(result.run.iter_oda))
    assert np.all((0.0 <= result.run.iter_oda) & (result.run.iter_oda <= 1.0))
    assert np.any(result.run.iter_oda > 0.0)
    energy_history = np.append(
        result.run.iter_energy, result.reference_relative_free_energy_ev_A3
    )
    energy_scale = max(
        float(np.max(np.abs(energy_history))), np.finfo(np.float64).tiny
    )
    energy_roundoff = 4096.0 * np.finfo(np.float64).eps * energy_scale
    assert np.all(np.diff(energy_history) <= energy_roundoff)

    assert np.allclose(
        result.h0_effective_ev,
        result.bare_h0_ev + result.reference_fock_ev,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.allclose(
        result.interaction_h_ev,
        result.density_hartree_ev + result.density_fock_ev,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.allclose(
        result.total_hamiltonian_ev,
        result.h0_effective_ev + result.interaction_h_ev,
        rtol=0.0,
        atol=2.0e-15,
    )
    physical_hamiltonian = result.bare_h0_ev + folded_hartree_action(
        result.density_delta_stored, coulomb=state.coulomb
    ) + folded_fock_action(result.mixed_projector_stored, coulomb=state.coulomb)
    assert np.allclose(
        result.total_hamiltonian_ev, physical_hamiltonian, rtol=2.0e-13, atol=2.0e-13
    )
    assert np.array_equal(result.run.state.hamiltonian, result.total_hamiltonian_ev)

    assert np.array_equal(
        result.mixed_projector_stored,
        result.reference_projector_stored + result.density_delta_stored,
    )
    assert np.array_equal(
        result.raw_update_density_delta_stored,
        result.raw_update_projector_stored - result.reference_projector_stored,
    )
    assert result.projector_stored is result.raw_update_projector_stored
    assert np.max(
        np.abs(result.raw_update_projector_stored - result.mixed_projector_stored)
    ) <= result.final_raw_norm + 2.0e-15

    weights = state.mesh.normalized_weights
    raw_number = np.einsum(
        "aak,k->", result.raw_update_projector_stored, weights, optimize=True
    ).real
    mixed_number = np.einsum(
        "aak,k->", result.mixed_projector_stored, weights, optimize=True
    ).real
    assert np.isclose(raw_number, config.electrons_per_reduced_momentum, atol=2.0e-11)
    assert np.isclose(mixed_number, config.electrons_per_reduced_momentum, atol=2.0e-11)

    assert result.energies_ev.shape == (FOLDED_DIMENSION, state.nk)
    for k_index in range(state.nk):
        vectors = result.eigenvectors[k_index]
        residual = (
            result.total_hamiltonian_ev[:, :, k_index] @ vectors
            - vectors * result.energies_ev[:, k_index][None, :]
        )
        assert np.max(np.abs(residual)) <= 2.0e-12
    density_ket = np.empty(
        (state.nk, FOLDED_DIMENSION, FOLDED_DIMENSION), dtype=np.complex128
    )
    for k_index in range(state.nk):
        vectors = result.eigenvectors[k_index]
        density_ket[k_index] = (
            vectors * result.occupations[:, k_index][None, :]
        ) @ vectors.conj().T
    reconstructed = conventional_projector_to_stored(np.moveaxis(density_ket, 0, 2))
    assert np.allclose(
        result.projector_stored, reconstructed, rtol=0.0, atol=2.0e-13
    )

    direct_free_energy = folded_reference_relative_free_energy(
        result.interaction_h_ev,
        result.h0_effective_ev,
        result.density_delta_stored,
        reference_projector_stored=result.reference_projector_stored,
        mesh=state.mesh,
        kbt_ev=config.kbt_ev,
    )
    assert result.reference_relative_free_energy_ev_A3 == pytest.approx(
        direct_free_energy, rel=5.0e-13, abs=1.0e-28
    )
    assert state.diagnostics["hf_energy"] == pytest.approx(
        direct_free_energy, rel=5.0e-13, abs=1.0e-28
    )


def test_precompute_rejects_a_cap_below_conservative_peak_estimate() -> None:
    mesh = build_monney_local_mesh(
        inplane_shells=0,
        inplane_spacing_Ainv=0.04,
        z_shells=0,
        z_spacing_Ainv=0.05,
    )
    estimate = estimate_folded_coulomb_peak_bytes(mesh)
    cap_gb = (estimate - 1) / 1.0e9
    with pytest.raises(MemoryError, match="conservative peak-memory cap"):
        precompute_folded_coulomb_kernel(
            mesh,
            q_vectors_Ainv=physical_q_vectors(),
            epsilon_r=5.0,
            max_dense_memory_gb=cap_gb,
        )
