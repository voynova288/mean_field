from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from mean_field.core.hf.engine import (
    DensityUpdateResult,
    FinalAcceptanceResult,
    HartreeFockStepResult,
    InteractionEnergyResult,
    compute_oda_parameter,
    run_fixed_mixing_scf,
    run_hartree_fock_iterations,
)
from mean_field.core.hf.problem import (
    HartreeFockKernel,
    HartreeFockProblem,
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


def test_core_hf_fixed_mixing_scf_matches_legacy_raw_norm_and_final_recompute() -> None:
    h0 = np.zeros((1, 1, 1), dtype=np.complex128)
    initial = np.zeros_like(h0)
    calls = {"interaction": 0}

    def interaction_builder(density: np.ndarray) -> np.ndarray:
        calls["interaction"] += 1
        return 0.25 * density + 1.0

    def density_builder(hamiltonian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        density = 0.5 * np.asarray(hamiltonian, dtype=np.complex128)
        energies = np.real(np.diagonal(hamiltonian[:, :, 0]))[:, None]
        return density, energies

    run = run_fixed_mixing_scf(
        h0,
        initial,
        interaction_builder=interaction_builder,
        density_builder=density_builder,
        max_iter=2,
        mixing=0.5,
        precision=1.0e-14,
    )

    # Iteration 1: raw=0.5, mixed=0.25. Iteration 2: interaction=1.0625,
    # raw density=0.53125, mixed=0.390625. Final interaction is recomputed
    # from the final mixed density, not reused from iteration 2.
    assert run.iterations == 2
    assert not run.converged
    assert np.allclose(run.density, 0.390625)
    assert np.allclose(run.interaction_h, 1.09765625)
    assert np.allclose(run.energies, [[1.09765625]])
    assert calls["interaction"] == 3


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


def test_core_hf_engine_cached_oda_matches_uncached_one_step_hamiltonian_and_energy() -> None:
    def make_state() -> _DummyState:
        state = _DummyState(
            h0=np.zeros((2, 2, 1), dtype=np.complex128),
            density=np.zeros((2, 2, 1), dtype=np.complex128),
            hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
            energies=np.zeros((2, 1), dtype=float),
            precision=1.0e-12,
        )
        state.h0[:, :, 0] = np.asarray([[-1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        return state

    target_density = np.zeros((2, 2, 1), dtype=np.complex128)
    target_density[:, :, 0] = np.asarray([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)

    def density_builder(hamiltonian: np.ndarray) -> DensityUpdateResult:
        return DensityUpdateResult(
            density=target_density,
            energies=np.diag(hamiltonian[:, :, 0]).real[:, None],
            mu=0.0,
        )

    def run_case(*, cached: bool) -> tuple[_DummyState, object, int]:
        state = make_state()
        calls = {"interaction": 0}

        def interaction_builder(density: np.ndarray) -> np.ndarray:
            calls["interaction"] += 1
            return 2.0 * density

        kwargs = {
            "state": state,
            "init_mode": "test",
            "seed": 1,
            "interaction_builder": interaction_builder,
            "density_builder": density_builder,
            "energy_functional": lambda interaction_h, h0, density: float(compute_energy_norm(interaction_h, h0, density)),
            "convergence_rule": "raw",
            "max_iter": 2,
        }
        if cached:
            kwargs["oda_delta_interaction_builder"] = interaction_builder
        else:
            kwargs["oda_parameterizer"] = lambda state_obj, delta_density: compute_oda_parameter(
                state_obj,
                delta_density,
                interaction_builder=interaction_builder,
            )
        run = run_hartree_fock_iterations(**kwargs)
        return state, run, calls["interaction"]

    def compute_energy_norm(interaction_h: np.ndarray, h0: np.ndarray, density: np.ndarray) -> float:
        return float(np.linalg.norm(h0 + interaction_h) + np.linalg.norm(density))

    uncached_state, uncached_run, uncached_calls = run_case(cached=False)
    cached_state, cached_run, cached_calls = run_case(cached=True)

    assert np.allclose(cached_run.iter_energy, uncached_run.iter_energy)
    assert np.allclose(cached_run.iter_err, uncached_run.iter_err)
    assert np.allclose(cached_run.iter_oda, uncached_run.iter_oda)
    assert np.allclose(cached_state.density, uncached_state.density)
    assert np.allclose(cached_state.hamiltonian, uncached_state.hamiltonian)
    assert cached_calls < uncached_calls


def test_core_hf_engine_fused_interaction_energy_builder_preserves_trajectory() -> None:
    def make_state() -> _DummyState:
        return _DummyState(
            h0=np.asarray([[[0.25 + 0.0j]]]),
            density=np.asarray([[[0.1 + 0.0j]]]),
            hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
            energies=np.zeros((1, 1), dtype=float),
            precision=1.0e-14,
        )

    target_density = np.asarray([[[0.4 + 0.0j]]])

    def density_builder(hamiltonian: np.ndarray) -> DensityUpdateResult:
        return DensityUpdateResult(
            density=target_density,
            energies=hamiltonian.real.reshape(1, 1),
            mu=float(hamiltonian[0, 0, 0].real),
        )

    def interaction(density: np.ndarray) -> np.ndarray:
        return 2.0 * density + 0.125

    def energy(interaction_h: np.ndarray, h0: np.ndarray, density: np.ndarray) -> float:
        return float(np.einsum("abk,abk->", h0 + 0.5 * interaction_h, density).real)

    legacy_state = make_state()
    legacy = run_hartree_fock_iterations(
        legacy_state,
        init_mode="test",
        seed=4,
        interaction_builder=interaction,
        density_builder=density_builder,
        energy_functional=energy,
        oda_parameterizer=lambda _state, _delta: 0.25,
        max_iter=2,
    )

    calls = {"joint": 0, "fallback": 0}

    def joint_builder(density: np.ndarray, h0: np.ndarray) -> InteractionEnergyResult:
        calls["joint"] += 1
        interaction_h = interaction(density)
        return InteractionEnergyResult(
            interaction_h=interaction_h,
            energy=energy(interaction_h, h0, density),
        )

    def forbidden_fallback(
        interaction_h: np.ndarray, h0: np.ndarray, density: np.ndarray
    ) -> float:
        del interaction_h, h0, density
        calls["fallback"] += 1
        raise AssertionError("fused path must not invoke the scalar fallback")

    fused_state = make_state()
    fused = run_hartree_fock_iterations(
        fused_state,
        init_mode="test",
        seed=4,
        interaction_builder=interaction,
        interaction_energy_builder=joint_builder,
        density_builder=density_builder,
        energy_functional=forbidden_fallback,
        oda_parameterizer=lambda _state, _delta: 0.25,
        max_iter=2,
    )

    np.testing.assert_array_equal(fused.iter_energy, legacy.iter_energy)
    np.testing.assert_array_equal(fused.iter_err, legacy.iter_err)
    np.testing.assert_array_equal(fused.iter_oda, legacy.iter_oda)
    np.testing.assert_array_equal(fused_state.density, legacy_state.density)
    np.testing.assert_array_equal(fused_state.hamiltonian, legacy_state.hamiltonian)
    assert fused_state.mu == legacy_state.mu
    assert calls == {"joint": fused.iterations + 1, "fallback": 0}


def test_core_hf_engine_fused_builder_disables_interaction_only_oda_cache() -> None:
    state = _DummyState(
        h0=np.asarray([[[0.25 + 0.0j]]]),
        density=np.asarray([[[0.1 + 0.0j]]]),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
        precision=0.0,
    )
    calls = {"joint": 0, "fallback": 0}

    def joint_builder(density: np.ndarray, h0: np.ndarray) -> InteractionEnergyResult:
        calls["joint"] += 1
        interaction_h = 2.0 * density + 0.125
        energy = float(
            np.einsum("abk,abk->", h0 + 0.5 * interaction_h, density).real
        )
        return InteractionEnergyResult(interaction_h=interaction_h, energy=energy)

    def density_builder(hamiltonian: np.ndarray) -> DensityUpdateResult:
        return DensityUpdateResult(
            density=np.asarray([[[0.4 + 0.0j]]]),
            energies=hamiltonian.real.reshape(1, 1),
            mu=float(hamiltonian[0, 0, 0].real),
        )

    def forbidden_fallback(
        interaction_h: np.ndarray, h0: np.ndarray, density: np.ndarray
    ) -> float:
        del interaction_h, h0, density
        calls["fallback"] += 1
        raise AssertionError("fused energy cannot use an interaction-only cache")

    run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=5,
        interaction_builder=lambda density: 2.0 * density + 0.125,
        interaction_energy_builder=joint_builder,
        density_builder=density_builder,
        energy_functional=forbidden_fallback,
        oda_delta_interaction_builder=lambda delta: 2.0 * delta,
        max_iter=1,
    )
    assert calls == {"joint": 2, "fallback": 0}


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


def test_core_hf_engine_mixed_convergence_rule_accepts_mixed_density_fixed_point() -> None:
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
        final_state_callback=lambda state_obj, update: (
            setattr(state_obj, "precision", 2.0),
            state_obj.diagnostics.__setitem__("final_raw_norm", -1.0),
        ),
        convergence_rule="mixed",
        max_iter=3,
    )

    assert not run.converged
    assert run.exit_reason == "final_raw_residual"
    assert run.iterations == 1
    assert np.isclose(run.iter_err[0], 0.0)
    assert np.isclose(run.iter_oda[0], 0.0)
    assert np.allclose(state.density, 0.0)
    assert np.isclose(state.diagnostics["final_raw_norm"], 1.0)
    assert state.diagnostics["final_accepted"] == 0.0
    assert state.precision == pytest.approx(1.0e-8)


def test_core_hf_engine_final_acceptance_can_reject_otherwise_fixed_point() -> None:
    state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
    )
    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=1,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=lambda hamiltonian: DensityUpdateResult(
            density=np.zeros_like(hamiltonian),
            energies=np.asarray([[0.0]], dtype=float),
            mu=0.0,
        ),
        energy_functional=lambda interaction_h, h0, density: 0.0,
        final_acceptance=lambda state_obj, update, final_norm: FinalAcceptanceResult(
            accepted=False,
            reason="final_particle_number",
            diagnostics={"particle_residual": 0.25},
        ),
        max_iter=1,
    )

    assert not run.converged
    assert run.exit_reason == "final_particle_number"
    assert state.diagnostics["final_raw_norm"] == pytest.approx(0.0)
    assert state.diagnostics["particle_residual"] == pytest.approx(0.25)
    assert state.diagnostics["final_accepted"] == 0.0


def test_core_hf_engine_restores_engine_owned_diagnostics_after_callbacks() -> None:
    state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
    )

    def mutate_diagnostics(
        state_obj: _DummyState,
        update: DensityUpdateResult,
        final_norm: float,
    ) -> FinalAcceptanceResult:
        state_obj.diagnostics["final_raw_norm"] = -1.0
        state_obj.diagnostics["hf_energy"] = -2.0
        state_obj.diagnostics["iterations"] = -3.0
        return FinalAcceptanceResult(accepted=True)

    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=1,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=lambda hamiltonian: DensityUpdateResult(
            density=np.zeros_like(hamiltonian),
            energies=np.asarray([[0.0]], dtype=float),
            mu=0.0,
        ),
        energy_functional=lambda interaction_h, h0, density: 0.0,
        final_acceptance=mutate_diagnostics,
        max_iter=1,
    )

    assert run.converged
    assert state.diagnostics["final_raw_norm"] == pytest.approx(0.0)
    assert state.diagnostics["hf_energy"] == pytest.approx(0.0)
    assert state.diagnostics["iterations"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("acceptance", "error"),
    [
        (FinalAcceptanceResult(accepted="false"), "exact Boolean"),
        (
            FinalAcceptanceResult(accepted=False, reason=object()),
            "nonempty exact string",
        ),
        (
            FinalAcceptanceResult(accepted=False, reason="converged"),
            "cannot use reason='converged'",
        ),
        (
            FinalAcceptanceResult(
                accepted=True,
                diagnostics={"final_raw_norm": 0.0},
            ),
            "engine-owned key",
        ),
    ],
)
def test_core_hf_engine_rejects_unsafe_final_acceptance_results(
    acceptance: FinalAcceptanceResult,
    error: str,
) -> None:
    state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
    )
    with pytest.raises((TypeError, ValueError), match=error):
        run_hartree_fock_iterations(
            state,
            init_mode="test",
            seed=1,
            interaction_builder=lambda density: np.zeros_like(density),
            density_builder=lambda hamiltonian: DensityUpdateResult(
                density=np.zeros_like(hamiltonian),
                energies=np.asarray([[0.0]], dtype=float),
                mu=0.0,
            ),
            energy_functional=lambda interaction_h, h0, density: 0.0,
            final_acceptance=lambda state_obj, update, final_norm: acceptance,
            max_iter=1,
        )


def test_core_hf_engine_accepts_custom_convergence_metric() -> None:
    state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
        precision=0.2,
    )
    target_density = np.ones_like(state.density)
    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=1,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=lambda hamiltonian: DensityUpdateResult(
            density=target_density,
            energies=np.asarray([[0.0]], dtype=float),
            mu=0.0,
        ),
        energy_functional=lambda interaction_h, h0, density: 0.0,
        oda_parameterizer=lambda state_obj, delta_density: 1.0,
        convergence_metric=lambda updated, previous: float(np.linalg.norm(updated - previous) / 10.0),
        max_iter=2,
    )
    assert run.converged
    assert np.allclose(run.iter_err, [0.1])
    assert np.isclose(state.diagnostics["final_raw_norm"], 0.0)


def test_core_hf_engine_rejects_invalid_custom_convergence_metric() -> None:
    state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
    )
    with pytest.raises(ValueError, match="finite nonnegative"):
        run_hartree_fock_iterations(
            state,
            init_mode="test",
            seed=1,
            interaction_builder=lambda density: np.zeros_like(density),
            density_builder=lambda hamiltonian: DensityUpdateResult(
                density=np.ones_like(state.density),
                energies=np.asarray([[0.0]], dtype=float),
                mu=0.0,
            ),
            energy_functional=lambda interaction_h, h0, density: 0.0,
            convergence_metric=lambda updated, previous: float("nan"),
            max_iter=1,
        )


def test_core_hf_engine_can_cap_oda_lambda_for_damped_continuation() -> None:
    state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
        precision=1.0e-12,
    )
    target_density = np.ones_like(state.density)

    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=1,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=lambda hamiltonian: DensityUpdateResult(
            density=target_density,
            energies=np.asarray([[0.0]], dtype=float),
            mu=0.0,
        ),
        energy_functional=lambda interaction_h, h0, density: 0.0,
        oda_parameterizer=lambda state_obj, delta_density: 1.0,
        max_oda_lambda=0.25,
        max_iter=1,
    )

    assert run.exit_reason == "max_iter"
    assert np.allclose(run.iter_oda, [0.25])
    assert np.allclose(state.density, 0.25)


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


def test_core_hf_engine_protects_precision_and_diagnostics_from_step_callback() -> None:
    state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
    )

    def mutate_step(state_obj: _DummyState, step: object) -> None:
        state_obj.precision = 2.0
        state_obj.diagnostics["iterations"] = 999.0
        state_obj.diagnostics["oda_parameter"] = -7.0

    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=1,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=lambda hamiltonian: DensityUpdateResult(
            density=np.zeros_like(hamiltonian),
            energies=np.asarray([[0.0]], dtype=float),
            mu=0.0,
        ),
        energy_functional=lambda interaction_h, h0, density: 0.0,
        step_callback=mutate_step,
        max_iter=1,
    )

    assert run.converged
    assert state.precision == pytest.approx(1.0e-8)
    assert state.diagnostics["iterations"] == pytest.approx(1.0)
    assert state.diagnostics["oda_parameter"] == pytest.approx(1.0)


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


def test_core_hf_engine_iteration_convergence_gate_delays_density_convergence() -> None:
    state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
    )
    trace: list[tuple[str, int, bool | None]] = []

    def step_callback(
        state_obj: _DummyState, step: HartreeFockStepResult
    ) -> None:
        del state_obj
        trace.append(("step", step.iteration, None))

    def convergence_gate(
        state_obj: _DummyState,
        step: HartreeFockStepResult,
        density_converged: bool,
    ) -> bool:
        del state_obj
        trace.append(("gate", step.iteration, density_converged))
        return step.iteration >= 3

    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=1,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=lambda hamiltonian: DensityUpdateResult(
            density=np.zeros_like(hamiltonian),
            energies=np.asarray([[0.0]], dtype=float),
            mu=0.0,
        ),
        energy_functional=lambda interaction_h, h0, density: 0.0,
        oda_parameterizer=lambda state_obj, delta_density: 0.0,
        step_callback=step_callback,
        iteration_convergence_gate=convergence_gate,
        max_iter=4,
    )

    assert run.converged
    assert run.iterations == 3
    assert trace == [
        ("step", 1, None),
        ("gate", 1, True),
        ("step", 2, None),
        ("gate", 2, True),
        ("step", 3, None),
        ("gate", 3, True),
    ]


def test_core_hf_engine_iteration_gate_does_not_hide_zero_mixing_stall() -> None:
    state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
    )
    target_density = np.ones_like(state.density)
    gate_inputs: list[bool] = []

    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=1,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=lambda hamiltonian: DensityUpdateResult(
            density=target_density,
            energies=np.asarray([[0.0]], dtype=float),
            mu=0.0,
        ),
        energy_functional=lambda interaction_h, h0, density: 0.0,
        oda_parameterizer=lambda state_obj, delta_density: 0.0,
        iteration_convergence_gate=lambda state_obj, step, passed: (
            gate_inputs.append(passed) or False
        ),
        convergence_rule="mixed",
        max_iter=3,
    )

    assert not run.converged
    assert run.exit_reason == "oda_stall"
    assert run.iterations == 1
    assert gate_inputs == [True]
    assert run.state.diagnostics["final_raw_norm"] > 0.0


def test_core_hf_engine_iteration_convergence_gate_is_protected_and_exact_bool() -> None:
    state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
    )
    calls = {"count": 0}

    def protected_gate(
        state_obj: _DummyState,
        step: HartreeFockStepResult,
        density_converged: bool,
    ) -> bool:
        assert density_converged
        calls["count"] += 1
        state_obj.precision = 2.0
        state_obj.diagnostics["iterations"] = 999.0
        return calls["count"] >= 2

    run = run_hartree_fock_iterations(
        state,
        init_mode="test",
        seed=1,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=lambda hamiltonian: DensityUpdateResult(
            density=np.zeros_like(hamiltonian),
            energies=np.asarray([[0.0]], dtype=float),
            mu=0.0,
        ),
        energy_functional=lambda interaction_h, h0, density: 0.0,
        iteration_convergence_gate=protected_gate,
        max_iter=2,
    )
    assert run.converged
    assert run.iterations == 2
    assert state.precision == pytest.approx(1.0e-8)
    assert state.diagnostics["iterations"] == pytest.approx(2.0)

    invalid_state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
    )
    with pytest.raises(TypeError, match="exact Boolean"):
        run_hartree_fock_iterations(
            invalid_state,
            init_mode="test",
            seed=1,
            interaction_builder=lambda density: np.zeros_like(density),
            density_builder=lambda hamiltonian: DensityUpdateResult(
                density=np.zeros_like(hamiltonian),
                energies=np.asarray([[0.0]], dtype=float),
                mu=0.0,
            ),
            energy_functional=lambda interaction_h, h0, density: 0.0,
            iteration_convergence_gate=lambda state_obj, step, passed: np.bool_(
                True
            ),
            max_iter=1,
        )


def test_core_hf_problem_bundles_initializer_interaction_and_projected_density_solver() -> None:
    state = _DummyState(
        h0=np.zeros((2, 2, 1), dtype=np.complex128),
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
    )

    initialized = {"value": False}
    gate_calls: list[bool] = []

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
            iteration_convergence_gate=lambda state_obj, step, passed: (
                gate_calls.append(passed) or True
            ),
            final_acceptance=lambda state_obj, update, final_norm: FinalAcceptanceResult(
                accepted=True,
                diagnostics={"problem_final_gate": 1.0},
            ),
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
    assert gate_calls == [False]
    assert run.iterations == 1
    assert state.diagnostics["problem_final_gate"] == 1.0
    assert np.allclose(state.density[:, :, 0], np.asarray([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128))


def test_core_hf_problem_preflight_can_supply_private_execution_snapshot() -> None:
    state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
    )
    called = {"trusted": False, "public": False}

    def trusted_initializer(
        state_obj: _DummyState, *, init_mode: str, seed: int
    ) -> None:
        del state_obj, init_mode, seed
        called["trusted"] = True

    trusted = HartreeFockProblem(
        initializer=trusted_initializer,
        kernel=HartreeFockKernel(
            interaction_builder=lambda density: np.zeros_like(density),
            density_builder=lambda hamiltonian: DensityUpdateResult(
                density=np.zeros_like(hamiltonian),
                energies=np.asarray([[0.0]], dtype=float),
                mu=0.0,
            ),
            energy_functional=lambda interaction_h, h0, density: 0.0,
        ),
    )

    class SnapshotProblem(HartreeFockProblem):
        def validate_before_execution(
            self, state_obj: _DummyState
        ) -> HartreeFockProblem:
            del state_obj
            return trusted

    def public_initializer(
        state_obj: _DummyState, *, init_mode: str, seed: int
    ) -> None:
        del state_obj, init_mode, seed
        called["public"] = True
        raise AssertionError("caller-visible initializer must not execute")

    public = SnapshotProblem(
        initializer=public_initializer,
        kernel=HartreeFockKernel(
            interaction_builder=lambda density: (_ for _ in ()).throw(
                AssertionError("caller-visible kernel must not execute")
            ),
            density_builder=lambda hamiltonian: (_ for _ in ()).throw(
                AssertionError("caller-visible kernel must not execute")
            ),
            energy_functional=lambda interaction_h, h0, density: (_ for _ in ()).throw(
                AssertionError("caller-visible kernel must not execute")
            ),
        ),
    )
    object.__setattr__(
        public,
        "validate_before_execution",
        lambda state_obj: (_ for _ in ()).throw(
            AssertionError("instance-level preflight override must not execute")
        ),
    )
    run = run_hartree_fock_problem(
        state,
        public,
        init_mode="snapshot",
        seed=0,
        max_iter=1,
    )
    assert run.iterations == 1
    assert called == {"trusted": True, "public": False}


def test_core_hf_problem_preflight_rejects_invalid_snapshot_type() -> None:
    state = _DummyState(
        h0=np.zeros((1, 1, 1), dtype=np.complex128),
        density=np.zeros((1, 1, 1), dtype=np.complex128),
        hamiltonian=np.zeros((1, 1, 1), dtype=np.complex128),
        energies=np.zeros((1, 1), dtype=float),
    )

    class InvalidSnapshotProblem(HartreeFockProblem):
        def validate_before_execution(self, state_obj: _DummyState) -> object:
            del state_obj
            return object()

    problem = InvalidSnapshotProblem(
        initializer=lambda state_obj, *, init_mode, seed: None,
        kernel=HartreeFockKernel(
            interaction_builder=lambda density: np.zeros_like(density),
            density_builder=lambda hamiltonian: DensityUpdateResult(
                density=np.zeros_like(hamiltonian),
                energies=np.asarray([[0.0]], dtype=float),
                mu=0.0,
            ),
            energy_functional=lambda interaction_h, h0, density: 0.0,
        ),
    )
    with pytest.raises(TypeError, match="must return HartreeFockProblem or None"):
        run_hartree_fock_problem(
            state,
            problem,
            init_mode="invalid-snapshot",
            seed=0,
            max_iter=1,
        )
