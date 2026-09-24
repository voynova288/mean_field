from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf.occupations import conventional_projector_to_stored
from mean_field.core.hf.problem import HartreeFockProblem
from mean_field.systems.tise2.folded_hf import precompute_folded_coulomb_kernel
from mean_field.systems.tise2.local_full_hf import (
    build_local_full_seed_field,
    diagonalize_local_full_finite_temperature,
    local_full_reference_relative_free_energy,
)
from mean_field.systems.tise2.local_ws_fft import (
    local_ws_fft_fock_action,
    local_ws_fft_hartree_action,
    local_ws_fft_interaction_action,
)
from mean_field.systems.tise2.local_ws_scf import (
    LocalWSSCFConfig,
    build_local_ws_scf_problem,
    build_local_ws_scf_state,
    initialize_local_ws_scf_state,
    local_ws_block_diagnostics,
    run_local_ws_scf,
)


def _config(**overrides) -> LocalWSSCFConfig:
    values = dict(
        M=0,
        L=0,
        epsilon_r=1.0e6,
        temperature_K=300.0,
        precision=1.0e-10,
        max_iter=8,
        fixed_mixing=0.25,
        seed_field_ev=5.0e-4,
        fft_workers=1,
        eigensolver_workers=1,
        max_fft_memory_gb=0.01,
    )
    values.update(overrides)
    return LocalWSSCFConfig(**values)


def _random_hermitian(nk: int, *, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(4, 4, nk)) + 1j * rng.normal(size=(4, 4, nk))
    return scale * 0.5 * (raw + raw.conj().swapaxes(0, 1))


def _entropy_gradient_stored(projector_stored: np.ndarray) -> np.ndarray:
    result = np.empty_like(projector_stored)
    for k_index in range(projector_stored.shape[2]):
        values, vectors = np.linalg.eigh(projector_stored[:, :, k_index])
        logit = np.log(values) - np.log1p(-values)
        result[:, :, k_index] = (
            (vectors * logit[None, :]) @ vectors.conj().T
        ).T
    return result


def test_d0_closes_normal_state_and_reference_relative_free_energy() -> None:
    state = build_local_ws_scf_state(_config())
    zero = np.zeros_like(state.density)
    interaction = local_ws_fft_interaction_action(
        zero, kernel=state.normal_state.kernel
    )
    np.testing.assert_array_equal(interaction, np.zeros_like(interaction))
    assert local_full_reference_relative_free_energy(
        interaction,
        state.h0,
        zero,
        reference_projector_stored=state.reference_projector_stored,
        mesh=state.normal_state.mesh,
        kbt_ev=state.normal_state.kbt_ev,
    ) == pytest.approx(0.0, rel=0.0, abs=2e-15)

    initialize_local_ws_scf_state(state, init_mode="normal")
    np.testing.assert_array_equal(state.density, zero)
    np.testing.assert_array_equal(state.hamiltonian, state.h0)


def test_seeded_initializer_removes_field_from_physical_hamiltonian() -> None:
    state = build_local_ws_scf_state(_config())
    initialize_local_ws_scf_state(state, init_mode="random_full", seed=29)

    assert np.all(np.abs(state.initial_seed_field_ev) > 0.0)
    interaction = local_ws_fft_interaction_action(
        state.density, kernel=state.normal_state.kernel
    )
    np.testing.assert_allclose(
        state.hamiltonian, state.h0 + interaction, rtol=0.0, atol=2e-15
    )
    assert np.max(
        np.abs(state.hamiltonian - (state.h0 + state.initial_seed_field_ev))
    ) > 1.0e-7

    physical = diagonalize_local_full_finite_temperature(
        state.hamiltonian,
        kbt_ev=state.normal_state.kbt_ev,
        k_weights=state.normal_state.mesh.normalized_weights,
    )
    np.testing.assert_allclose(state.energies, physical.energies_ev, atol=2e-15)
    assert state.mu == pytest.approx(physical.chemical_potential_ev, abs=2e-15)


def test_runner_calls_generic_hartree_fock_problem_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mean_field.systems.tise2.local_ws_scf as module

    calls: list[tuple[object, object]] = []
    generic_runner = module.run_hartree_fock_problem

    def recording_runner(state, problem, **kwargs):
        calls.append((state, problem))
        return generic_runner(state, problem, **kwargs)

    monkeypatch.setattr(module, "run_hartree_fock_problem", recording_runner)
    state = build_local_ws_scf_state(_config())
    result = module.run_local_ws_scf(state, init_mode="normal")

    assert len(calls) == 1
    assert calls[0][0] is state
    assert isinstance(calls[0][1], HartreeFockProblem)
    assert result.run.converged


def test_fixed_point_acceptance_complete_eigensystem_and_mixing_semantics() -> None:
    state = build_local_ws_scf_state(_config())
    result = run_local_ws_scf(state, init_mode="normal")

    assert result.run.converged
    assert result.run.exit_reason == "converged"
    assert state.diagnostics["final_accepted"] == 1.0
    assert result.final_raw_norm <= state.precision
    np.testing.assert_array_equal(
        result.run.iter_oda,
        np.full(result.run.iterations, state.config.fixed_mixing),
    )
    assert result.energies_ev.shape == (4, state.nk)
    assert result.eigenvectors.shape == (state.nk, 4, 4)
    assert result.occupations.shape == (4, state.nk)
    assert result.projector_stored is result.raw_update_projector_stored
    np.testing.assert_array_equal(
        result.mixed_projector_stored,
        result.reference_projector_stored + result.density_delta_stored,
    )
    np.testing.assert_array_equal(
        result.raw_update_density_delta_stored,
        result.raw_update_projector_stored - result.reference_projector_stored,
    )
    for k_index in range(state.nk):
        vectors = result.eigenvectors[k_index]
        residual = (
            result.total_hamiltonian_ev[:, :, k_index] @ vectors
            - vectors * result.energies_ev[:, k_index][None, :]
        )
        assert np.max(np.abs(residual)) < 2.0e-12
    density_ket = (
        result.eigenvectors * result.occupations.T[:, None, :]
    ) @ result.eigenvectors.conj().transpose(0, 2, 1)
    reconstructed_raw = conventional_projector_to_stored(
        np.moveaxis(density_ket, 0, 2)
    )
    np.testing.assert_allclose(
        result.raw_update_projector_stored,
        reconstructed_raw,
        rtol=0.0,
        atol=2.0e-13,
    )

    # A deliberately loose one-step acceptance makes raw-vs-mixed semantics
    # observable without pretending this is a production convergence setting.
    loose = build_local_ws_scf_state(
        _config(precision=1.0, max_iter=1, fixed_mixing=0.25)
    )
    one_step = run_local_ws_scf(loose, init_mode="random_full", seed=7)
    assert one_step.run.converged
    assert np.max(
        np.abs(
            one_step.raw_update_density_delta_stored
            - one_step.density_delta_stored
        )
    ) > 1.0e-12
    expected_interaction = local_ws_fft_interaction_action(
        one_step.density_delta_stored, kernel=loose.normal_state.kernel
    )
    np.testing.assert_allclose(
        one_step.interaction_h_ev, expected_interaction, rtol=0.0, atol=2e-15
    )


def test_ws_free_energy_derivative_matches_physical_hamiltonian() -> None:
    state = build_local_ws_scf_state(
        _config(epsilon_r=31.0, temperature_K=1000.0)
    )
    density = _random_hermitian(state.nk, seed=31, scale=1.0e-4)
    direction = _random_hermitian(state.nk, seed=32)
    for k_index in range(state.nk):
        direction[:, :, k_index] -= (
            np.trace(direction[:, :, k_index]) / 4.0
        ) * np.eye(4)
    interaction = local_ws_fft_interaction_action(
        density, kernel=state.normal_state.kernel
    )
    direction_interaction = local_ws_fft_interaction_action(
        direction, kernel=state.normal_state.kernel
    )
    projector = state.reference_projector_stored + density
    eigenvalues = np.linalg.eigvalsh(np.moveaxis(projector, 2, 0))
    assert np.min(eigenvalues) > 0.0
    assert np.max(eigenvalues) < 1.0

    step = 2.0e-7

    def free_energy(sign: float) -> float:
        trial_density = density + sign * step * direction
        trial_interaction = interaction + sign * step * direction_interaction
        return local_full_reference_relative_free_energy(
            trial_interaction,
            state.h0,
            trial_density,
            reference_projector_stored=state.reference_projector_stored,
            mesh=state.normal_state.mesh,
            kbt_ev=state.normal_state.kbt_ev,
        )

    numerical = (free_energy(1.0) - free_energy(-1.0)) / (2.0 * step)
    gradient = (
        state.h0
        + interaction
        + state.normal_state.kbt_ev * _entropy_gradient_stored(projector)
    )
    analytic = np.einsum(
        "abk,abk,k->",
        gradient,
        direction,
        state.normal_state.mesh.weights_Ainv3,
        optimize=True,
    )
    assert abs(analytic.imag) < 2.0e-12
    assert numerical == pytest.approx(analytic.real, rel=4.0e-7, abs=4.0e-10)


def test_explicit_continuation_initializer_preserves_supplied_mixed_density() -> None:
    source = build_local_ws_scf_state(_config())
    field = build_local_full_seed_field(
        source.nk,
        init_mode="symmetric_exciton",
        amplitude_ev=source.config.seed_field_ev,
        seed=0,
    )
    seeded = diagonalize_local_full_finite_temperature(
        source.h0 + field,
        kbt_ev=source.normal_state.kbt_ev,
        k_weights=source.normal_state.mesh.normalized_weights,
    )
    continuation = seeded.projector_stored - source.reference_projector_stored

    state = build_local_ws_scf_state(_config())
    initialize_local_ws_scf_state(
        state,
        init_mode="continuation",
        continuation_density_delta_stored=continuation,
    )
    np.testing.assert_array_equal(state.density, continuation)
    np.testing.assert_array_equal(
        state.initial_seed_field_ev, np.zeros_like(state.initial_seed_field_ev)
    )
    expected_interaction = local_ws_fft_interaction_action(
        continuation, kernel=state.normal_state.kernel
    )
    np.testing.assert_allclose(
        state.hamiltonian, state.h0 + expected_interaction, rtol=0.0, atol=2e-15
    )
    assert state.diagnostics["continuation_initializer"] == 1.0

    with pytest.raises(ValueError, match="requires an explicit continuation"):
        initialize_local_ws_scf_state(state, init_mode="continuation")


def test_unrestricted_problem_has_no_block_masks_and_reports_all_blocks() -> None:
    state = build_local_ws_scf_state(_config())
    problem = build_local_ws_scf_problem(state)
    assert problem.kernel.hamiltonian_postprocessor is None
    assert problem.kernel.density_postprocessor is None

    initialize_local_ws_scf_state(state, init_mode="random_full", seed=11)
    assert np.all(np.abs(state.initial_seed_field_ev) > 0.0)
    hartree = local_ws_fft_hartree_action(
        state.density, kernel=state.normal_state.kernel
    )
    fock = local_ws_fft_fock_action(
        state.density, kernel=state.normal_state.kernel
    )
    diagnostics = local_ws_block_diagnostics(
        state.density,
        density_hartree_ev=hartree,
        density_fock_ev=fock,
        state=state,
    )
    for value in (
        diagnostics.density_average_stored,
        diagnostics.density_rms,
        diagnostics.hartree_average_ev,
        diagnostics.hartree_rms_ev,
        diagnostics.fock_average_ev,
        diagnostics.fock_rms_ev,
        diagnostics.self_energy_average_ev,
        diagnostics.self_energy_rms_ev,
        diagnostics.order_magnitude,
    ):
        assert value.shape == (4, 4)
    assert diagnostics.exciton_order_stored.shape == (3,)
    assert diagnostics.exciton_self_energy_ev.shape == (3,)
    np.testing.assert_array_equal(
        diagnostics.order_magnitude,
        np.abs(diagnostics.density_average_stored),
    )


def test_problem_fft_action_matches_direct_and_dense_actions() -> None:
    state = build_local_ws_scf_state(
        _config(M=1, epsilon_r=17.0, max_fft_memory_gb=0.05)
    )
    density = _random_hermitian(state.nk, seed=53, scale=0.01)
    problem = build_local_ws_scf_problem(state)
    from_problem = problem.kernel.interaction_builder(density)
    direct = local_ws_fft_interaction_action(
        density, kernel=state.normal_state.kernel
    )
    dense = precompute_folded_coulomb_kernel(
        state.normal_state.mesh,
        q_vectors_Ainv=state.normal_state.q_vectors_Ainv,
        epsilon_r=state.config.epsilon_r,
        max_dense_memory_gb=0.05,
    )
    dense_expected = np.zeros_like(density)
    routes = (
        (((0, 0), (1, 1), (2, 2), (3, 3)), ((1, 0),), ((2, 0),), ((3, 0),)),
        (((0, 1),), ((0, 0), (1, 1), (2, 2), (3, 3)), ((2, 1),), ((3, 1),)),
        (((0, 2),), ((1, 2),), ((0, 0), (1, 1), (2, 2), (3, 3)), ((3, 2),)),
        (((0, 3),), ((1, 3),), ((2, 3),), ((0, 0), (1, 1), (2, 2), (3, 3))),
    )
    for sector in range(4):
        for target in range(4):
            for row, col in routes[sector][target]:
                dense_expected[sector, target] -= (
                    dense.shifted_weighted_ev[sector, col] @ density[row, col]
                )
                dense_expected[sector, target] += dense.hartree_ev[
                    sector, target
                ] * np.einsum(
                    "p,p->",
                    density[row, col],
                    state.normal_state.mesh.weights_Ainv3,
                    optimize=True,
                )
    np.testing.assert_array_equal(from_problem, direct)
    np.testing.assert_allclose(from_problem, dense_expected, rtol=3e-13, atol=3e-13)
