from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf.occupations import conventional_projector_to_stored
from mean_field.systems.tise2.folded_hf import (
    DOUBLED_Q_LABELS,
    FoldedTiSe2Geometry,
    physical_momentum_routes,
    physical_q_vectors,
    precompute_folded_coulomb_kernel,
)
from mean_field.systems.tise2.hf import COULOMB_EV_ANGSTROM, MonneyLocalMesh
from mean_field.systems.tise2.local_full_hf import (
    ELECTRONS_PER_LOCAL_MOMENTUM,
    LOCAL_DIMENSION,
    LocalFullHFConfig,
    _finite_temperature_oda_parameter,
    build_local_full_hf_state,
    build_local_full_seed_field,
    diagonalize_local_full_finite_temperature,
    initialize_local_full_hf_state,
    local_full_fock_action,
    local_full_h0,
    local_full_hartree_action,
    local_full_interaction_action,
    local_full_interaction_energy,
    local_full_reference_relative_energy,
    local_full_reference_relative_free_energy,
    run_local_full_hf,
)
from mean_field.systems.tise2.monney import (
    MonneyTiSe2Parameters,
    monney_bare_hamiltonian,
)


def _asymmetric_two_k_problem(epsilon_r: float = 7.0):
    points = np.asarray(
        [[0.013, -0.007, 0.011], [-0.019, 0.005, -0.017]], dtype=float
    )
    weights = np.asarray([0.7e-4, 1.3e-4], dtype=float)
    mesh = MonneyLocalMesh(
        points_Ainv=points,
        weights_Ainv3=weights,
        labels=np.asarray([[0, 0, 0], [1, -1, 1]], dtype=np.int64),
        inplane_shells=1,
        inplane_spacing_Ainv=0.04,
        z_shells=1,
        z_spacing_Ainv=0.05,
        cell_volume_Ainv3=1.0e-4,
    )
    q_vectors = physical_q_vectors()
    coulomb = precompute_folded_coulomb_kernel(
        mesh,
        q_vectors_Ainv=q_vectors,
        epsilon_r=epsilon_r,
        max_dense_memory_gb=0.01,
    )
    return mesh, q_vectors, coulomb


def _random_hermitian(nk: int, *, seed: int, scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(LOCAL_DIMENSION, LOCAL_DIMENSION, nk))
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


def _literal_local_wick_actions(
    density_delta: np.ndarray,
    *,
    mesh: MonneyLocalMesh,
    q_vectors: np.ndarray,
    epsilon_r: float,
    self_cell_ev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent all-index two-k Wick contraction with no route helper."""

    prefactor = 4.0 * np.pi * COULOMB_EV_ANGSTROM / epsilon_r
    fock = np.zeros_like(density_delta)
    hartree = np.zeros_like(density_delta)
    for s in range(LOCAL_DIMENSION):
        for t in range(LOCAL_DIMENSION):
            q_h = q_vectors[s] - q_vectors[t]
            q_h2 = float(np.dot(q_h, q_h))
            hartree_coefficient = 0.0 if q_h2 <= 64.0 * np.finfo(float).eps else prefactor / q_h2
            for target_k in range(mesh.nk):
                for u in range(LOCAL_DIMENSION):
                    for v in range(LOCAL_DIMENSION):
                        if not _literal_route_is_retained(s, t, u, v):
                            continue
                        for source_k in range(mesh.nk):
                            transfer = (
                                mesh.points_Ainv[target_k]
                                - mesh.points_Ainv[source_k]
                                + q_vectors[s]
                                - q_vectors[v]
                            )
                            q2 = float(np.dot(transfer, transfer))
                            weighted_fock = (
                                self_cell_ev
                                if q2 <= 64.0 * np.finfo(float).eps
                                else mesh.weights_Ainv3[source_k] * prefactor / q2
                            )
                            fock[s, t, target_k] -= (
                                weighted_fock * density_delta[u, v, source_k]
                            )
                            hartree[s, t, target_k] += (
                                hartree_coefficient
                                * mesh.weights_Ainv3[source_k]
                                * density_delta[u, v, source_k]
                            )
    return hartree, fock


def _hermitian_directions(nk: int):
    normalization = 1.0 / np.sqrt(2.0)
    for k_index in range(nk):
        for row in range(LOCAL_DIMENSION):
            direction = np.zeros((LOCAL_DIMENSION, LOCAL_DIMENSION, nk), complex)
            direction[row, row, k_index] = 1.0
            yield direction
        for row in range(LOCAL_DIMENSION):
            for col in range(row + 1, LOCAL_DIMENSION):
                real_direction = np.zeros(
                    (LOCAL_DIMENSION, LOCAL_DIMENSION, nk), complex
                )
                real_direction[row, col, k_index] = normalization
                real_direction[col, row, k_index] = normalization
                yield real_direction
                imaginary_direction = np.zeros_like(real_direction)
                imaginary_direction[row, col, k_index] = 1j * normalization
                imaginary_direction[col, row, k_index] = -1j * normalization
                yield imaginary_direction


def _entropy_gradient_stored(projector_stored: np.ndarray) -> np.ndarray:
    result = np.empty_like(projector_stored)
    for k_index in range(projector_stored.shape[2]):
        values, vectors = np.linalg.eigh(projector_stored[:, :, k_index])
        logit = np.log(values) - np.log1p(-values)
        result[:, :, k_index] = (
            (vectors * logit[None, :]) @ vectors.conj().T
        ).T
    return result


def test_tise2_package_exports_local_unrestricted_api() -> None:
    from mean_field.systems import tise2

    assert tise2.LocalFullHFConfig is LocalFullHFConfig
    assert tise2.build_local_full_hf_state is build_local_full_hf_state
    assert tise2.run_local_full_hf is run_local_full_hf
    assert tise2.local_full_h0 is local_full_h0


def test_local_h0_is_exact_monney_bare_hamiltonian_on_common_p_basis() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.017, -0.009, 0.021]], dtype=float
    )
    params = MonneyTiSe2Parameters()
    assert np.array_equal(
        local_full_h0(points, params=params),
        monney_bare_hamiltonian(points, params=params),
    )


def test_exact_g0_inventory_has_28_routes() -> None:
    routes = physical_momentum_routes()
    assert sum(len(block) for row in routes for block in row) == 28
    assert all(
        len(routes[s][t]) == (4 if s == t else 1)
        for s in range(LOCAL_DIMENSION)
        for t in range(LOCAL_DIMENSION)
    )
    for s in range(LOCAL_DIMENSION):
        for t in range(LOCAL_DIMENSION):
            for u, v in routes[s][t]:
                assert _literal_route_is_retained(s, t, u, v)


def test_asymmetric_two_k_actions_match_independent_literal_wick_oracle() -> None:
    mesh, q_vectors, coulomb = _asymmetric_two_k_problem()
    density = _random_hermitian(mesh.nk, seed=13, scale=0.004)
    expected_hartree, expected_fock = _literal_local_wick_actions(
        density,
        mesh=mesh,
        q_vectors=q_vectors,
        epsilon_r=coulomb.epsilon_r,
        self_cell_ev=coulomb.self_cell_ev,
    )

    actual_hartree = local_full_hartree_action(density, coulomb=coulomb)
    actual_fock = local_full_fock_action(density, coulomb=coulomb)
    np.testing.assert_allclose(actual_hartree, expected_hartree, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(actual_fock, expected_fock, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(
        local_full_interaction_action(density, coulomb=coulomb),
        expected_hartree + expected_fock,
        rtol=3e-13,
        atol=3e-13,
    )


def test_asymmetric_two_k_serial_parallel_interaction_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, _q_vectors, coulomb = _asymmetric_two_k_problem(epsilon_r=19.0)
    density = _random_hermitian(mesh.nk, seed=20260919, scale=0.006)

    serial_fock = local_full_fock_action(density, coulomb=coulomb)
    serial_interaction = local_full_interaction_action(density, coulomb=coulomb)
    serial_energy = local_full_interaction_energy(density, coulomb=coulomb)

    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    parallel_fock = local_full_fock_action(density, coulomb=coulomb, workers=3)
    parallel_interaction = local_full_interaction_action(
        density, coulomb=coulomb, workers=3
    )
    parallel_energy = local_full_interaction_energy(
        density, coulomb=coulomb, workers=3
    )

    np.testing.assert_array_equal(parallel_fock, serial_fock)
    np.testing.assert_array_equal(parallel_interaction, serial_interaction)
    assert parallel_energy == serial_energy


@pytest.mark.parametrize(
    "name",
    (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ),
)
def test_interaction_workers_require_one_thread_numerical_environment(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        monkeypatch.setenv(variable, "1")
    monkeypatch.setenv(name, "2")
    config = LocalFullHFConfig(
        epsilon_r=100.0,
        inplane_shells=0,
        z_shells=0,
        max_dense_memory_gb=0.01,
        interaction_workers=2,
        eigensolver_workers=1,
    )
    with pytest.raises(RuntimeError, match=name):
        build_local_full_hf_state(config)


@pytest.mark.parametrize("workers", (0, -1, True, 1.5))
def test_config_rejects_invalid_interaction_workers(workers: object) -> None:
    with pytest.raises(TypeError, match="interaction_workers"):
        LocalFullHFConfig(
            epsilon_r=100.0,
            interaction_workers=workers,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("mixing_policy", ("linear", "", None))
def test_config_rejects_invalid_mixing_policy(mixing_policy: object) -> None:
    with pytest.raises(ValueError, match="mixing_policy"):
        LocalFullHFConfig(
            epsilon_r=100.0,
            mixing_policy=mixing_policy,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "fixed_mixing", (0.0, -0.01, 1.01, np.nan, np.inf, -np.inf)
)
def test_config_rejects_invalid_fixed_mixing(fixed_mixing: float) -> None:
    with pytest.raises(ValueError, match="fixed_mixing"):
        LocalFullHFConfig(epsilon_r=100.0, fixed_mixing=fixed_mixing)


def test_isolated_imaginary_route_fixes_stored_density_orientation() -> None:
    mesh, _q_vectors, coulomb = _asymmetric_two_k_problem()
    routes = physical_momentum_routes()
    assert routes[0][1] == ((1, 0),)

    amplitude = 0.017
    density = np.zeros((LOCAL_DIMENSION, LOCAL_DIMENSION, mesh.nk), complex)
    density[1, 0, 0] = 1j * amplitude
    density[0, 1, 0] = -1j * amplitude

    hartree = local_full_hartree_action(density, coulomb=coulomb)
    fock = local_full_fock_action(density, coulomb=coulomb)
    expected_hartree_01 = (
        coulomb.hartree_ev[0, 1]
        * mesh.weights_Ainv3[0]
        * density[1, 0, 0]
    )
    expected_fock_01 = (
        -coulomb.shifted_weighted_ev[0, 0, :, 0] * density[1, 0, 0]
    )

    np.testing.assert_allclose(
        hartree[0, 1], expected_hartree_01, rtol=0.0, atol=2e-15
    )
    np.testing.assert_allclose(fock[0, 1], expected_fock_01, rtol=0.0, atol=2e-15)
    assert np.max(np.abs(fock[0, 1] + expected_fock_01)) > 1.0e-6


def test_q0_hartree_is_removed_finite_q_retained_and_reference_zero_is_zero() -> None:
    mesh, _q_vectors, coulomb = _asymmetric_two_k_problem()
    assert np.array_equal(np.diag(coulomb.hartree_ev), np.zeros(LOCAL_DIMENSION))
    assert np.all(coulomb.hartree_ev[~np.eye(LOCAL_DIMENSION, dtype=bool)] > 0.0)

    zero = np.zeros((LOCAL_DIMENSION, LOCAL_DIMENSION, mesh.nk), complex)
    assert np.array_equal(local_full_hartree_action(zero, coulomb=coulomb), zero)
    assert np.array_equal(local_full_fock_action(zero, coulomb=coulomb), zero)
    assert np.array_equal(local_full_interaction_action(zero, coulomb=coulomb), zero)


def test_every_external_4x4_block_is_reachable_without_a_mask() -> None:
    mesh, _q_vectors, coulomb = _asymmetric_two_k_problem()
    routes = physical_momentum_routes()
    for s in range(LOCAL_DIMENSION):
        for t in range(LOCAL_DIMENSION):
            u, v = routes[s][t][0]
            density = np.zeros((LOCAL_DIMENSION, LOCAL_DIMENSION, mesh.nk), complex)
            density[u, v, 0] = 0.013 + (0.007j if u != v else 0.0j)
            density[v, u, 0] = density[u, v, 0].conjugate()
            sigma = local_full_fock_action(density, coulomb=coulomb)
            assert np.max(np.abs(sigma[s, t])) > 0.0, (s, t, u, v)


def test_global_finite_temperature_filling_targets_one_on_average() -> None:
    hamiltonian = np.zeros((LOCAL_DIMENSION, LOCAL_DIMENSION, 2), complex)
    hamiltonian[:, :, 0] = np.diag([-1.0, -0.8, 0.5, 0.7])
    hamiltonian[:, :, 1] = np.diag([0.1, 0.2, 0.3, 0.4])
    weights = np.asarray([0.2, 0.8])
    spectrum = diagonalize_local_full_finite_temperature(
        hamiltonian,
        kbt_ev=0.08,
        k_weights=weights,
        particle_tolerance=1.0e-13,
    )

    traces = np.trace(spectrum.projector_stored, axis1=0, axis2=1).real
    assert weights @ traces == pytest.approx(ELECTRONS_PER_LOCAL_MOMENTUM, abs=2e-12)
    assert traces[0] != pytest.approx(traces[1], abs=1e-3)
    assert spectrum.energies_ev.shape == (LOCAL_DIMENSION, 2)
    assert spectrum.eigenvectors.shape == (2, LOCAL_DIMENSION, LOCAL_DIMENSION)
    assert spectrum.occupations.shape == (LOCAL_DIMENSION, 2)


def test_full_hermitian_energy_and_free_energy_derivatives_include_complex_directions() -> None:
    mesh, _q_vectors, coulomb = _asymmetric_two_k_problem(epsilon_r=31.0)
    h0 = monney_bare_hamiltonian(mesh.points_Ainv)
    reference = np.repeat(
        (0.25 * np.eye(LOCAL_DIMENSION, dtype=complex))[:, :, None], mesh.nk, axis=2
    )
    density = _random_hermitian(mesh.nk, seed=22, scale=0.002)
    interaction = local_full_interaction_action(density, coulomb=coulomb)
    projector = reference + density
    assert np.min(np.linalg.eigvalsh(np.moveaxis(projector, 2, 0))) > 0.0
    assert np.max(np.linalg.eigvalsh(np.moveaxis(projector, 2, 0))) < 1.0

    kbt_ev = 0.07
    internal_gradient = h0 + interaction
    free_gradient = internal_gradient + kbt_ev * _entropy_gradient_stored(projector)
    step = 2.0e-6
    for direction in _hermitian_directions(mesh.nk):
        direction_interaction = local_full_interaction_action(direction, coulomb=coulomb)

        def internal_at(sign: float) -> float:
            trial = density + sign * step * direction
            trial_interaction = interaction + sign * step * direction_interaction
            return local_full_reference_relative_energy(
                trial_interaction, h0, trial, mesh=mesh
            )

        def free_at(sign: float) -> float:
            trial = density + sign * step * direction
            trial_interaction = interaction + sign * step * direction_interaction
            return local_full_reference_relative_free_energy(
                trial_interaction,
                h0,
                trial,
                reference_projector_stored=reference,
                mesh=mesh,
                kbt_ev=kbt_ev,
            )

        numerical_internal = (internal_at(1.0) - internal_at(-1.0)) / (2.0 * step)
        numerical_free = (free_at(1.0) - free_at(-1.0)) / (2.0 * step)
        analytic_internal = np.einsum(
            "abk,abk,k->", internal_gradient, direction, mesh.weights_Ainv3
        )
        analytic_free = np.einsum(
            "abk,abk,k->", free_gradient, direction, mesh.weights_Ainv3
        )
        assert abs(analytic_internal.imag) < 2e-12
        assert abs(analytic_free.imag) < 2e-12
        assert numerical_internal == pytest.approx(analytic_internal.real, rel=2e-8, abs=2e-10)
        assert numerical_free == pytest.approx(analytic_free.real, rel=3e-7, abs=3e-10)


def test_seed_fields_are_unconstrained_and_random_full_activates_all_blocks() -> None:
    symmetric = build_local_full_seed_field(
        2, init_mode="symmetric_exciton", amplitude_ev=1.0e-3, seed=0
    )
    assert np.all(symmetric[0, 1:, :] == -1.0e-3)
    assert np.all(symmetric[1:, 0, :] == -1.0e-3)
    assert np.array_equal(symmetric, symmetric.conj().swapaxes(0, 1))

    amplitude = 1.0e-3
    random = build_local_full_seed_field(
        2, init_mode="random_full", amplitude_ev=amplitude, seed=11
    )
    assert np.array_equal(random, random.conj().swapaxes(0, 1))
    assert np.all(np.abs(random) > 0.0)
    assert np.array_equal(random[:, :, 0], random[:, :, 1])
    assert np.max(np.abs(random)) == pytest.approx(amplitude, rel=0.0, abs=1e-18)

    config = LocalFullHFConfig(
        epsilon_r=100.0,
        temperature_K=300.0,
        inplane_shells=0,
        z_shells=0,
        max_dense_memory_gb=0.01,
    )
    state = build_local_full_hf_state(config)
    initialize_local_full_hf_state(state, init_mode="random_full", seed=11)
    assert np.all(np.abs(state.initial_seed_field_ev) > 0.0)
    assert np.all(np.abs(state.density) > 0.0)
    assert np.max(np.abs(state.density - state.density.conj().swapaxes(0, 1))) < 1e-14


def test_seeded_initializer_removes_seed_and_closes_physical_hamiltonian() -> None:
    config = LocalFullHFConfig(
        epsilon_r=100.0,
        temperature_K=300.0,
        inplane_shells=0,
        z_shells=0,
        max_dense_memory_gb=0.01,
    )
    state = build_local_full_hf_state(config)
    initialize_local_full_hf_state(state, init_mode="random_full", seed=29)

    seed_spectrum = diagonalize_local_full_finite_temperature(
        state.h0 + state.initial_seed_field_ev,
        kbt_ev=state.config.kbt_ev,
        k_weights=state.mesh.normalized_weights,
    )
    expected_density = (
        seed_spectrum.projector_stored - state.reference_projector_stored
    )
    expected_hamiltonian = state.h0 + local_full_interaction_action(
        expected_density, coulomb=state.coulomb
    )
    physical_spectrum = diagonalize_local_full_finite_temperature(
        expected_hamiltonian,
        kbt_ev=state.config.kbt_ev,
        k_weights=state.mesh.normalized_weights,
    )

    np.testing.assert_allclose(state.density, expected_density, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        state.hamiltonian, expected_hamiltonian, rtol=0.0, atol=2e-15
    )
    np.testing.assert_allclose(
        state.energies, physical_spectrum.energies_ev, rtol=0.0, atol=2e-15
    )
    assert state.mu == pytest.approx(
        physical_spectrum.chemical_potential_ev, rel=0.0, abs=2e-15
    )
    physical_raw_density = (
        physical_spectrum.projector_stored - state.reference_projector_stored
    )
    assert np.max(np.abs(state.density - physical_raw_density)) > 1.0e-12

    state.hamiltonian.fill(np.nan)
    initialize_local_full_hf_state(state, init_mode="normal", seed=29)
    assert np.array_equal(state.density, np.zeros_like(state.density))
    assert np.array_equal(state.hamiltonian, state.h0)


def test_outer_eigensolver_workers_require_one_thread_blas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hamiltonian = np.repeat(
        np.diag([-0.2, 0.0, 0.2, 0.4]).astype(complex)[None, :, :], 2, axis=0
    )
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with pytest.raises(RuntimeError, match="one-thread"):
        diagonalize_local_full_finite_temperature(
            np.moveaxis(hamiltonian, 0, 2),
            kbt_ev=0.05,
            eigensolver_workers=2,
        )


def test_one_point_seeded_interacting_oda_step_closes_mixed_raw_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check one physical ODA step without weakening strict SCF convergence gates."""

    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    config = LocalFullHFConfig(
        epsilon_r=100.0,
        temperature_K=300.0,
        inplane_shells=0,
        z_shells=0,
        max_dense_memory_gb=0.01,
        seed_field_ev=5.0e-4,
        interaction_workers=2,
        oda_grid_points=17,
    )
    state = build_local_full_hf_state(config)
    initialize_local_full_hf_state(state, init_mode="random_full", seed=41)

    previous_density = state.density.copy()
    previous_interaction = local_full_interaction_action(
        previous_density, coulomb=state.coulomb
    )
    np.testing.assert_allclose(
        state.hamiltonian, state.h0 + previous_interaction, rtol=0.0, atol=2e-15
    )
    assert np.max(np.abs(previous_interaction)) > 0.0

    previous_raw = diagonalize_local_full_finite_temperature(
        state.hamiltonian,
        kbt_ev=state.config.kbt_ev,
        k_weights=state.mesh.normalized_weights,
    )
    raw_density = (
        previous_raw.projector_stored - state.reference_projector_stored
    )
    delta_density = raw_density - previous_density
    oda_lambda = _finite_temperature_oda_parameter(state, delta_density)
    assert 0.0 <= oda_lambda <= 1.0

    delta_interaction = local_full_interaction_action(
        delta_density, coulomb=state.coulomb
    )

    def free_energy_at(value: float) -> float:
        density = previous_density + value * delta_density
        interaction = previous_interaction + value * delta_interaction
        return local_full_reference_relative_free_energy(
            interaction,
            state.h0,
            density,
            reference_projector_stored=state.reference_projector_stored,
            mesh=state.mesh,
            kbt_ev=state.config.kbt_ev,
        )

    endpoint_energies = (free_energy_at(0.0), free_energy_at(1.0))
    mixed_free_energy = free_energy_at(oda_lambda)
    energy_tolerance = 2.0e-13 * max(
        1.0, abs(endpoint_energies[0]), abs(endpoint_energies[1])
    )
    assert mixed_free_energy <= min(endpoint_energies) + energy_tolerance
    assert mixed_free_energy <= endpoint_energies[0] + energy_tolerance

    mixed_density = previous_density + oda_lambda * delta_density
    mixed_hartree = local_full_hartree_action(mixed_density, coulomb=state.coulomb)
    mixed_fock = local_full_fock_action(mixed_density, coulomb=state.coulomb)
    mixed_interaction = mixed_hartree + mixed_fock
    final_hamiltonian = state.h0 + mixed_interaction
    final_raw = diagonalize_local_full_finite_temperature(
        final_hamiltonian,
        kbt_ev=state.config.kbt_ev,
        k_weights=state.mesh.normalized_weights,
    )

    np.testing.assert_allclose(
        mixed_interaction,
        previous_interaction + oda_lambda * delta_interaction,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        final_hamiltonian,
        state.h0
        + local_full_hartree_action(mixed_density, coulomb=state.coulomb)
        + local_full_fock_action(mixed_density, coulomb=state.coulomb),
        rtol=0.0,
        atol=2e-15,
    )
    for k_index in range(state.nk):
        vectors = final_raw.eigenvectors[k_index]
        residual = (
            final_hamiltonian[:, :, k_index] @ vectors
            - vectors * final_raw.energies_ev[:, k_index][None, :]
        )
        assert np.max(np.abs(residual)) < 2e-12
    raw_density_ket = (
        final_raw.eigenvectors * final_raw.occupations.T[:, None, :]
    ) @ final_raw.eigenvectors.conj().transpose(0, 2, 1)
    reconstructed_raw = conventional_projector_to_stored(
        np.moveaxis(raw_density_ket, 0, 2)
    )
    np.testing.assert_allclose(
        final_raw.projector_stored, reconstructed_raw, rtol=0.0, atol=2e-13
    )

    independently_evaluated_mixed_free_energy = (
        local_full_reference_relative_free_energy(
            mixed_interaction,
            state.h0,
            mixed_density,
            reference_projector_stored=state.reference_projector_stored,
            mesh=state.mesh,
            kbt_ev=state.config.kbt_ev,
        )
    )
    assert mixed_free_energy == pytest.approx(
        independently_evaluated_mixed_free_energy, rel=0.0, abs=2e-15
    )
    final_raw_density = (
        final_raw.projector_stored - state.reference_projector_stored
    )
    assert np.max(np.abs(mixed_density - final_raw_density)) > 1.0e-12


def test_one_point_seeded_fixed_mixing_run_records_exact_steps_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    config = LocalFullHFConfig(
        epsilon_r=100.0,
        temperature_K=300.0,
        inplane_shells=0,
        z_shells=0,
        max_iter=200,
        max_dense_memory_gb=0.01,
        seed_field_ev=5.0e-4,
        interaction_workers=2,
        mixing_policy="fixed",
        fixed_mixing=0.20,
    )
    state = build_local_full_hf_state(config)
    result = run_local_full_hf(state, init_mode="random_full", seed=41)

    assert result.run.converged
    assert result.run.exit_reason == "converged"
    # The generic engine chooses and records mixing before testing raw convergence,
    # so even an already-converged first raw update has one fixed-policy entry.
    assert result.run.iterations >= 1
    np.testing.assert_array_equal(
        result.run.iter_oda,
        np.full(result.run.iterations, config.fixed_mixing, dtype=float),
    )
    assert result.final_raw_norm <= config.precision

    expected_interaction = local_full_interaction_action(
        result.density_delta_stored,
        coulomb=state.coulomb,
        workers=config.interaction_workers,
    )
    np.testing.assert_allclose(
        result.interaction_h_ev, expected_interaction, rtol=0.0, atol=2e-15
    )
    np.testing.assert_allclose(
        result.total_hamiltonian_ev,
        result.h0_ev + expected_interaction,
        rtol=0.0,
        atol=2e-15,
    )
    closed = diagonalize_local_full_finite_temperature(
        result.total_hamiltonian_ev,
        kbt_ev=config.kbt_ev,
        k_weights=state.mesh.normalized_weights,
        eigensolver_workers=config.eigensolver_workers,
    )
    np.testing.assert_allclose(
        result.raw_update_projector_stored,
        closed.projector_stored,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_array_equal(
        result.mixed_projector_stored,
        result.reference_projector_stored + result.density_delta_stored,
    )


def test_one_point_normal_run_closes_eigensystem_and_projectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    config = LocalFullHFConfig(
        epsilon_r=100.0,
        temperature_K=300.0,
        inplane_shells=0,
        z_shells=0,
        precision=5.0e-10,
        max_iter=4,
        max_dense_memory_gb=0.01,
        interaction_workers=2,
    )
    state = build_local_full_hf_state(config)
    expected_h0 = monney_bare_hamiltonian(state.mesh.points_Ainv, params=state.params)
    assert np.array_equal(state.h0, expected_h0)

    result = run_local_full_hf(state, init_mode="normal")
    assert result.run.converged
    assert result.run.exit_reason == "converged"
    assert result.energies_ev.shape == (LOCAL_DIMENSION, state.nk)
    assert result.eigenvectors.shape == (state.nk, LOCAL_DIMENSION, LOCAL_DIMENSION)
    assert result.occupations.shape == (LOCAL_DIMENSION, state.nk)
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
    ) <= result.final_raw_norm + 2e-15

    for k_index in range(state.nk):
        vectors = result.eigenvectors[k_index]
        residual = (
            result.total_hamiltonian_ev[:, :, k_index] @ vectors
            - vectors * result.energies_ev[:, k_index][None, :]
        )
        assert np.max(np.abs(residual)) < 2e-12
    density_ket = (
        result.eigenvectors * result.occupations.T[:, None, :]
    ) @ result.eigenvectors.conj().transpose(0, 2, 1)
    reconstructed = conventional_projector_to_stored(np.moveaxis(density_ket, 0, 2))
    np.testing.assert_allclose(result.projector_stored, reconstructed, atol=2e-13, rtol=0.0)

    raw_number = np.einsum(
        "aak,k->", result.raw_update_projector_stored, state.mesh.normalized_weights
    ).real
    mixed_number = np.einsum(
        "aak,k->", result.mixed_projector_stored, state.mesh.normalized_weights
    ).real
    assert raw_number == pytest.approx(ELECTRONS_PER_LOCAL_MOMENTUM, abs=2e-11)
    assert mixed_number == pytest.approx(ELECTRONS_PER_LOCAL_MOMENTUM, abs=2e-11)
    assert np.max(np.abs(result.interaction_h_ev)) == pytest.approx(0.0, abs=2e-13)
    assert np.array_equal(result.total_hamiltonian_ev, result.h0_ev)
