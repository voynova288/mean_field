from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mean_field.core.hf import (
    DensityUpdateResult,
    HartreeFockKernel,
    HartreeFockProblem,
    compute_oda_parameter,
    run_hartree_fock_iterations,
    run_hartree_fock_problem,
)


@dataclass
class _DummyState:
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    mu: float = float("nan")
    precision: float = 1e-8
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


def test_core_hf_engine_oda_parameter_matches_julia_transpose_convention() -> None:
    state = _DummyState(
        h0=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
    )
    state.h0[:, :, 0] = np.asarray([[0.0, -3.0], [-5.0, 0.0]], dtype=np.complex128)
    interaction_h = np.asarray([[0.0, -7.0], [-11.0, 0.0]], dtype=np.complex128)
    state.hamiltonian[:, :, 0] = state.h0[:, :, 0] + interaction_h

    delta_density = np.zeros_like(state.density)
    delta_density[:, :, 0] = np.asarray([[0.0, 1.0], [2.0, 0.0]], dtype=np.complex128)
    delta_h = np.zeros_like(state.density)
    delta_h[:, :, 0] = np.asarray([[0.0, 19.0], [23.0, 0.0]], dtype=np.complex128)

    lambda_mix = compute_oda_parameter(
        state,
        delta_density,
        interaction_builder=lambda density: delta_h,
    )
    assert np.isclose(lambda_mix, 27.5 / 65.0)


def test_core_hf_engine_oda_parameter_matches_julia_complex_projector_storage_convention() -> None:
    state = _DummyState(
        h0=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
    )
    state.h0[:, :, 0] = np.asarray([[-1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    state.hamiltonian[:, :, 0] = state.h0[:, :, 0]

    delta_density = np.zeros_like(state.density)
    delta_density[:, :, 0] = np.asarray([[3.0, 2.0j], [-2.0j, 4.0]], dtype=np.complex128)
    delta_h = np.zeros_like(state.density)
    delta_h[:, :, 0] = np.asarray([[1.0, 1.0j], [-1.0j, 2.0]], dtype=np.complex128)

    lambda_mix = compute_oda_parameter(
        state,
        delta_density,
        interaction_builder=lambda density: delta_h,
    )

    # Julia B0 computes `a = tr(transpose(delta_P) * delta_H) = 7`
    # and `b = tr(transpose(delta_P) * H0) = -3` for these blocks.
    assert np.isclose(lambda_mix, 3.0 / 7.0)


def test_core_hf_engine_default_convergence_rule_is_raw_to_avoid_false_oda_convergence() -> None:
    state = _DummyState(
        h0=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
    )
    target_density = np.zeros_like(state.density)
    target_density[:, :, 0] = np.asarray([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
    target_energies = np.asarray([[-1.0], [1.0]], dtype=float)

    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=1,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=lambda hamiltonian: DensityUpdateResult(
            density=target_density,
            energies=target_energies,
            mu=0.0,
        ),
        energy_functional=lambda interaction_h, h0, density: 0.0,
        oda_parameterizer=lambda state_obj, delta_density: 0.0,
        max_iter=3,
    )

    assert not run.converged
    assert run.exit_reason == "oda_stall"
    assert run.iterations == 1
    assert run.iter_err[0] > 0.0


def test_core_hf_engine_mixed_convergence_requires_raw_convergence_before_oda_stall() -> None:
    state = _DummyState(
        h0=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
    )
    target_density = np.zeros_like(state.density)
    target_density[:, :, 0] = np.asarray([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
    target_energies = np.asarray([[-1.0], [1.0]], dtype=float)

    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=1,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=lambda hamiltonian: DensityUpdateResult(
            density=target_density,
            energies=target_energies,
            mu=0.0,
        ),
        energy_functional=lambda interaction_h, h0, density: 0.0,
        oda_parameterizer=lambda state_obj, delta_density: 0.0,
        convergence_rule="mixed",
        max_iter=3,
    )

    assert not run.converged
    assert run.exit_reason == "oda_stall"
    assert run.iterations == 1
    assert np.isclose(run.iter_err[0], 0.0)
    assert np.isclose(run.iter_oda[0], 0.0)
    assert np.allclose(state.density, 0.0)
    assert np.isclose(state.diagnostics["final_raw_norm"], 1.0)


def test_core_hf_engine_finalizes_hamiltonian_from_final_mixed_density() -> None:
    state = _DummyState(
        h0=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
    )
    state.h0[:, :, 0] = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    target_density = np.zeros_like(state.density)
    target_density[:, :, 0] = np.asarray([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)

    def density_builder(hamiltonian: np.ndarray) -> DensityUpdateResult:
        return DensityUpdateResult(
            density=target_density,
            energies=np.diag(hamiltonian[:, :, 0]).real[:, None],
            mu=float(np.trace(hamiltonian[:, :, 0]).real),
        )

    final_energies: list[np.ndarray] = []
    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=1,
        interaction_builder=lambda density: 2.0 * density,
        density_builder=density_builder,
        energy_functional=lambda interaction_h, h0, density: float(np.linalg.norm(h0 + interaction_h)),
        oda_parameterizer=lambda state_obj, delta_density: 0.5,
        convergence_rule="raw",
        max_iter=1,
        final_state_callback=lambda state_obj, update: final_energies.append(update.energies.copy()),
    )

    expected_density = 0.5 * target_density
    expected_hamiltonian = state.h0 + 2.0 * expected_density

    assert run.exit_reason == "max_iter"
    assert np.allclose(state.density, expected_density)
    assert np.allclose(state.hamiltonian, expected_hamiltonian)
    assert np.allclose(state.energies, np.diag(expected_hamiltonian[:, :, 0]).real[:, None])
    assert np.allclose(final_energies[0], state.energies)


def test_core_hf_engine_raw_convergence_rule_matches_restricted_hf_behavior() -> None:
    state = _DummyState(
        h0=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
    )
    target_density = np.zeros_like(state.density)
    target_density[:, :, 0] = np.asarray([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
    target_energies = np.asarray([[-1.0], [1.0]], dtype=float)

    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=1,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=lambda hamiltonian: DensityUpdateResult(
            density=target_density,
            energies=target_energies,
            mu=0.0,
        ),
        energy_functional=lambda interaction_h, h0, density: 0.0,
        oda_parameterizer=lambda state_obj, delta_density: 0.0,
        convergence_rule="raw",
        max_iter=3,
    )

    assert not run.converged
    assert run.exit_reason == "oda_stall"
    assert run.iterations == 1
    assert run.iter_err[0] > 0.0


def test_core_hf_engine_applies_postprocessors_and_step_callback() -> None:
    state = _DummyState(
        h0=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
    )

    callback_trace: list[tuple[int, float]] = []

    def density_builder(hamiltonian: np.ndarray) -> DensityUpdateResult:
        density = np.zeros_like(hamiltonian)
        density[:, :, 0] = np.asarray([[0.5, 2.0], [3.0, -0.5]], dtype=np.complex128)
        return DensityUpdateResult(
            density=density,
            energies=np.asarray([[-1.0], [1.0]], dtype=float),
            mu=0.0,
            observables={"sigma_ztauz": np.asarray([[1.0], [-1.0]], dtype=float)},
        )

    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=7,
        interaction_builder=lambda density: np.ones_like(density),
        density_builder=density_builder,
        energy_functional=lambda interaction_h, h0, density: float(np.linalg.norm(interaction_h)),
        density_postprocessor=lambda density: density.__setitem__((0, 1, 0), 0.0),
        hamiltonian_postprocessor=lambda hamiltonian: hamiltonian.__setitem__((0, 1, 0), 0.0),
        step_callback=lambda state_obj, step: callback_trace.append((step.iteration, step.energy)),
        max_iter=1,
    )

    assert run.iterations == 1
    assert callback_trace == [(1, float(np.linalg.norm(np.ones_like(state.h0))))]
    assert state.hamiltonian[0, 1, 0] == 0.0
    assert state.density[0, 1, 0] == 0.0


def test_core_hf_problem_bundles_initializer_interaction_and_projected_density_solver() -> None:
    state = _DummyState(
        h0=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
    )

    initialized = {"value": False}

    def initializer(state_obj: _DummyState, *, init_mode: str, seed: int) -> None:
        initialized["value"] = True
        assert init_mode == "generic"
        assert seed == 3
        state_obj.density[:, :, 0] = np.asarray([[0.25, 0.0], [0.0, -0.25]], dtype=np.complex128)

    problem = HartreeFockProblem(
        initializer=initializer,
        kernel=HartreeFockKernel(
            interaction_builder=lambda density: np.zeros_like(density),
            density_builder=lambda hamiltonian: DensityUpdateResult(
                density=np.asarray([[[0.5], [0.0]], [[0.0], [-0.5]]], dtype=np.complex128),
                energies=np.asarray([[-1.0], [1.0]], dtype=float),
                mu=0.0,
            ),
            energy_functional=lambda interaction_h, h0, density: float(np.trace(density[:, :, 0]).real),
            convergence_rule="mixed",
        ),
    )

    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode="generic",
        seed=3,
        max_iter=1,
    )

    assert initialized["value"]
    assert run.iterations == 1
    assert np.allclose(state.density[:, :, 0], np.asarray([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128))
