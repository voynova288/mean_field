from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf.stationary import (
    StationarySolveConfig,
    pack_hermitian_matrix_field,
)
from mean_field.systems.inas_gasb.xue2018_hf import (
    build_xue2018_hf_state,
    run_xue2018_hf,
    xue2018_interaction_action,
)
from mean_field.systems.inas_gasb.xue2018_stationary import (
    Xue2018StationaryConfig,
    project_xue2018_neutral_density_delta,
    solve_xue2018_stationary_root,
    xue2018_state_at_parameters,
    xue2018_stationary_density_map,
    xue2018_stationary_residual_vector,
    xue2018_trs_tangent_projector_vector,
    xue2018_trsb_tangent_projector_vector,
    xue2018_unrestricted_density_map_vector,
)
from mean_field.systems.inas_gasb.xue2018_symmetry import (
    project_xue2018_full_trs,
    xue2018_opposite_k_indices,
    xue2018_time_reversal_unitary,
    xue2018_trs_residual,
)


def _small_state():
    return build_xue2018_hf_state(
        eg_ry=-0.5,
        hybridization_ab_ry=0.22,
        kmax_ab_inv=1.0,
        points_per_axis=3,
        precision=1.0e-8,
        q0_kernel_backend="dense",
    )


def test_xue2018_time_reversal_algebra_mesh_and_h0_covariance() -> None:
    state = _small_state()
    unitary = xue2018_time_reversal_unitary()
    assert np.max(np.abs(unitary @ unitary.conj() + np.eye(4))) == 0.0
    partners = xue2018_opposite_k_indices(state.mesh)
    assert np.array_equal(partners[partners], np.arange(state.nk))
    assert xue2018_trs_residual(state.h0, state.mesh) < 2.0e-15
    assert xue2018_trs_residual(state.reference_density, state.mesh) < 2.0e-15


def test_xue2018_full_trs_projector_and_interaction_covariance() -> None:
    state = _small_state()
    rng = np.random.default_rng(31)
    raw = rng.normal(size=state.h0.shape) + 1j * rng.normal(size=state.h0.shape)
    hermitian = raw + np.swapaxes(raw.conj(), 0, 1)
    density = project_xue2018_full_trs(hermitian, state.mesh)
    weights = np.asarray(state.mesh.weights_ab2)
    total = np.einsum("aak,k->", density, weights, optimize=True).real
    density -= (
        total / (density.shape[0] * np.sum(weights))
        * np.eye(density.shape[0], dtype=np.complex128)[:, :, None]
    )
    projected_twice = project_xue2018_full_trs(density, state.mesh)
    assert np.max(np.abs(projected_twice - density)) < 2.0e-15
    assert xue2018_trs_residual(density, state.mesh) < 2.0e-15
    interaction = xue2018_interaction_action(state, density)
    assert xue2018_trs_residual(interaction, state.mesh) < 2.0e-13


def test_xue2018_finite_temperature_map_closes_half_filling_and_trs() -> None:
    state = _small_state()
    mapped = xue2018_stationary_density_map(
        state,
        np.zeros_like(state.density),
        thermal_energy_ry=1.0e-2,
    )
    assert abs(mapped.number_residual) < 2.0e-13
    for ik in range(state.nk):
        eigenvalues = np.linalg.eigvalsh(mapped.absolute_density[:, :, ik])
        assert np.min(eigenvalues) >= -1.0e-14
        assert np.max(eigenvalues) <= 1.0 + 1.0e-14
    assert mapped.trs_leakage_max < 2.0e-13
    assert xue2018_trs_residual(mapped.density_delta_raw, state.mesh) < 2.0e-13


def test_xue2018_stationary_solver_accepts_known_normal_fixed_point() -> None:
    state = _small_state()
    oda = run_xue2018_hf(
        state,
        init_mode="normal",
        max_iter=1000,
        max_oda_lambda=0.5,
        oda_stall_threshold=1.0e-14,
    )
    assert oda.run.converged
    stationary = solve_xue2018_stationary_root(
        state,
        oda.run.state.density,
        config=Xue2018StationaryConfig(
            thermal_energy_ry=0.0,
            root=StationarySolveConfig(
                residual_rms_tolerance=1.0e-10,
                residual_max_tolerance=1.0e-9,
                anderson_max_iterations=30,
                anderson_memory=1,
                krylov_max_iterations=30,
            ),
            full_residual_rms_tolerance=1.0e-10,
            full_residual_max_tolerance=1.0e-9,
        ),
    )
    assert stationary.converged
    assert stationary.full_residual_rms < 1.0e-10
    assert stationary.full_residual_max < 1.0e-9
    assert stationary.trs_error < 1.0e-12
    assert abs(stationary.number_residual) < 1.0e-12
    vector = pack_hermitian_matrix_field(stationary.density_delta)
    mapped = xue2018_unrestricted_density_map_vector(
        state,
        vector,
        thermal_energy_ry=0.0,
    )
    assert np.max(np.abs(mapped - vector)) < 1.0e-9


def test_xue2018_parameter_family_reuses_regulator_and_returns_full_residual() -> None:
    state = _small_state()
    changed = xue2018_state_at_parameters(
        state,
        eg_ry=-0.5,
        hybridization_ab_ry=0.23,
    )
    assert changed.q0_kernel is state.q0_kernel
    assert changed.mesh is state.mesh
    assert np.max(np.abs(changed.h0 - state.h0)) > 0.0
    vector = pack_hermitian_matrix_field(np.zeros_like(state.density))
    residual = xue2018_stationary_residual_vector(
        changed,
        vector,
        thermal_energy_ry=1.0e-2,
    )
    assert residual.shape == vector.shape
    assert np.all(np.isfinite(residual))


def test_xue2018_trs_and_trsb_tangent_projectors_decompose_field() -> None:
    state = _small_state()
    rng = np.random.default_rng(71)
    raw = rng.normal(size=state.h0.shape) + 1j * rng.normal(size=state.h0.shape)
    field = raw + np.swapaxes(raw.conj(), 0, 1)
    vector = pack_hermitian_matrix_field(field)
    trs = xue2018_trs_tangent_projector_vector(state, vector)
    trsb = xue2018_trsb_tangent_projector_vector(state, vector)
    neutral = pack_hermitian_matrix_field(
        project_xue2018_neutral_density_delta(state, field)
    )
    assert trs + trsb == pytest.approx(neutral, abs=2.0e-13)
    assert xue2018_trs_tangent_projector_vector(state, trs) == pytest.approx(trs, abs=2.0e-13)
    assert xue2018_trsb_tangent_projector_vector(state, trsb) == pytest.approx(trsb, abs=2.0e-13)


def test_xue2018_stationary_map_rejects_negative_temperature() -> None:
    state = _small_state()
    with pytest.raises(ValueError, match="nonnegative"):
        xue2018_stationary_density_map(
            state,
            np.zeros_like(state.density),
            thermal_energy_ry=-1.0e-3,
        )
