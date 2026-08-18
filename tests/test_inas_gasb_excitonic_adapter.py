from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mean_field.systems.inas_gasb import (
    Kane4Bundle,
    MatrixEIConfig,
    ProjectedFockOperator,
    fixed_mu_fermi_density,
    solve_reference_subtracted_matrix_ei,
    weighted_fermi_density,
)


def _bundle_from_h0(h0: np.ndarray, weights: np.ndarray) -> Kane4Bundle:
    dimension, _, nk = h0.shape
    if dimension != 4:
        raise ValueError("test bundle requires four active states")
    phi = np.zeros((nk, 2, 4, 4), dtype=np.complex128)
    phi[:, 0, :, :] = np.eye(4)[None, :, :] / np.sqrt(2.0)
    phi[:, 1, :, :] = np.eye(4)[None, :, :] / np.sqrt(2.0)
    return Kane4Bundle(
        k_cart_nm_inv=np.column_stack(
            [np.arange(nk, dtype=float) * 0.01, np.zeros(nk)]
        ),
        weights_nm2=np.asarray(weights, dtype=float),
        z_nm=np.array([-0.5, 0.5]),
        z_weights_nm=np.ones(2),
        h0_mev=np.asarray(h0, dtype=np.complex128),
        micro_wavefunctions=phi,
        provenance={"source": "generic-excitonic-adapter-test"},
    )


def _zero_fock(bundle: Kane4Bundle) -> ProjectedFockOperator:
    green = np.zeros((bundle.nk, bundle.nk, bundle.nz, bundle.nz), dtype=float)
    return ProjectedFockOperator.from_bundle(
        bundle,
        green,
        self_cell_description="analytic zero-interaction adapter test",
    )


def test_legacy_fermi_wrappers_preserve_units_keys_and_fixed_mu_covariance() -> None:
    nk = 3
    energies = np.array(
        [
            [-2.0, -1.8, -1.6],
            [-0.7, -0.5, -0.3],
            [0.4, 0.6, 0.8],
            [1.5, 1.7, 1.9],
        ]
    )
    h0 = np.zeros((4, 4, nk), dtype=np.complex128)
    for ik in range(nk):
        h0[:, :, ik] = np.diag(energies[:, ik])
    weights = np.array([0.03, 0.07, 0.11])
    temperature = 0.4
    mu = 0.15
    update = fixed_mu_fermi_density(
        h0, weights, temperature_K=temperature, mu_mev=mu
    )
    expected = 1.0 / (
        1.0 + np.exp((energies - mu) / (0.08617333262 * temperature))
    )
    for ik in range(nk):
        np.testing.assert_allclose(update.density[:, :, ik], np.diag(expected[:, ik]))
    assert update.mu == mu
    assert update.observables["active_number_density_nm2"] == pytest.approx(
        np.einsum("k,nk->", weights, expected)
    )
    assert "number_residual_nm2" not in update.observables

    shift = 7.25
    shifted = fixed_mu_fermi_density(
        h0 + shift * np.eye(4)[:, :, None],
        weights,
        temperature_K=temperature,
        mu_mev=mu + shift,
    )
    np.testing.assert_allclose(shifted.density, update.density, atol=2.0e-13)
    np.testing.assert_allclose(shifted.energies, update.energies + shift, atol=2.0e-13)

    weighted = weighted_fermi_density(
        h0,
        weights,
        temperature_K=0.2,
        target_occupation_per_k=2.0,
    )
    assert weighted.observables["achieved_occupation_per_k"] == pytest.approx(2.0)
    assert abs(weighted.observables["number_residual_nm2"]) < 1.0e-12


def test_kane_adapter_preserves_fixed_mu_reference_and_checkpoint_callbacks() -> None:
    nk = 4
    h0 = np.zeros((4, 4, nk), dtype=np.complex128)
    for ik in range(nk):
        h0[:, :, ik] = np.diag([-2.0 + 0.1 * ik, -1.0, 1.0, 2.0 + 0.2 * ik])
    weights = np.array([0.03, 0.07, 0.11, 0.19])
    fixed_mu = -1.25
    bundle = _bundle_from_h0(h0, weights)
    steps: list[int] = []
    finals: list[np.ndarray] = []

    result = solve_reference_subtracted_matrix_ei(
        bundle,
        _zero_fock(bundle),
        config=MatrixEIConfig(
            temperature_K=0.2,
            mixing=0.3,
            precision=1.0e-11,
            max_iter=10,
            constraint_policy="kane_poisson_fixed_mu",
            fixed_mu_mev=fixed_mu,
        ),
        step_callback=lambda _state, step: steps.append(step.iteration),
        final_state_callback=lambda _state, update: finals.append(update.density.copy()),
    )
    assert result.run.converged
    assert result.reference_mu_mev == fixed_mu
    assert result.mu_mev == fixed_mu
    assert result.run.state.diagnostics["fixed_mu_mev"] == fixed_mu
    assert abs(result.run.state.diagnostics["reference_occupation_per_k"] - 2.0) > 0.1
    np.testing.assert_allclose(result.density_delta, 0.0, atol=1.0e-13)
    assert abs(result.canonical_free_energy_difference_mev_nm2) < 1.0e-13
    assert abs(result.grand_potential_difference_mev_nm2) < 1.0e-13
    assert steps == [1]
    assert len(finals) == 1
    np.testing.assert_allclose(finals[0], result.density_delta, atol=1.0e-14)
    assert "fixed Kane-Poisson ordinary-electron mu" in result.classification


def test_kane_adapter_keeps_e1_empty_h1_filled_normal_ordering() -> None:
    nk = 3
    h0 = np.repeat(
        np.diag([-2.0, -1.0, 1.0, 2.0]).astype(complex)[:, :, None],
        nk,
        axis=2,
    )
    weights = np.array([0.03, 0.07, 0.11])
    bundle = _bundle_from_h0(h0, weights)
    result = solve_reference_subtracted_matrix_ei(
        bundle,
        _zero_fock(bundle),
        config=MatrixEIConfig(
            temperature_K=0.02,
            mixing=0.3,
            precision=1.0e-11,
            max_iter=10,
            constraint_policy="kane_poisson_fixed_mu",
            fixed_mu_mev=0.0,
            normal_ordering_reference_policy="electron_hole_vacuum",
        ),
    )
    expected_vacuum = np.repeat(
        bundle.basis.h1_electron_projector[:, :, None], nk, axis=2
    )
    expected_physical = np.repeat(
        bundle.basis.electron_projector[:, :, None], nk, axis=2
    )
    np.testing.assert_allclose(result.reference_density, expected_vacuum, atol=1.0e-14)
    np.testing.assert_allclose(result.noninteracting_density, expected_physical, atol=1.0e-14)
    np.testing.assert_allclose(
        result.density_delta, expected_physical - expected_vacuum, atol=1.0e-14
    )
    np.testing.assert_allclose(result.total_density, expected_physical, atol=1.0e-14)
    assert abs(result.run.state.diagnostics["active_number_change_nm2"]) < 1.0e-13
    assert "E1-empty/H1-filled normal ordering" in result.classification


def test_nonzero_fock_adapter_matches_pre_refactor_trajectory_oracle() -> None:
    nk = 3
    weights = np.array([0.03, 0.07, 0.11])
    h0 = np.empty((4, 4, nk), dtype=complex)
    for ik in range(nk):
        h0[:, :, ik] = np.array(
            [
                [-0.35 + 0.04 * ik, 0.01j, 0.025, -0.008j],
                [-0.01j, -0.30 + 0.03 * ik, 0.006j, 0.021],
                [0.025, -0.006j, 0.22 + 0.02 * ik, -0.012j],
                [0.008j, 0.021, 0.012j, 0.27 + 0.01 * ik],
            ]
        )
    bundle = _bundle_from_h0(h0, weights)
    green = np.empty((nk, nk, 2, 2), dtype=float)
    for i in range(nk):
        for j in range(nk):
            green[i, j] = 0.9 / (1 + abs(i - j)) * np.array(
                [[1.0, 0.55], [0.55, 1.0]]
            )
    operator = ProjectedFockOperator.from_bundle(
        bundle,
        green,
        self_cell_description="old-new parity",
    )
    result = solve_reference_subtracted_matrix_ei(
        bundle,
        operator,
        config=MatrixEIConfig(
            temperature_K=0.25,
            mixing=0.3,
            precision=1.0e-11,
            max_iter=500,
            constraint_policy="kane_poisson_fixed_mu",
            fixed_mu_mev=0.0,
            normal_ordering_reference_policy="electron_hole_vacuum",
        ),
    )

    fixture_root = Path(__file__).parent / "fixtures" / "excitonic"
    metadata = json.loads(
        (fixture_root / "inas_gasb_old_adapter_nonzero_fock.json").read_text()
    )
    fixture_path = fixture_root / metadata["fixture_npz"]
    generator_path = fixture_root / Path(metadata["generator"]["path"]).name
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == metadata["fixture_sha256"]
    assert (
        hashlib.sha256(generator_path.read_bytes()).hexdigest()
        == metadata["generator"]["sha256"]
    )
    snapshot_root = fixture_root / metadata["pre_refactor_source_snapshot"]
    for source_path, expected_sha in metadata["pre_refactor_sources"].items():
        snapshot = snapshot_root / Path(source_path).name
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected_sha
    assert result.run.converged
    assert result.run.exit_reason == metadata["expected"]["exit_reason"]
    assert result.run.iterations == metadata["expected"]["iterations"]

    current = {
        "reference_density": result.reference_density,
        "noninteracting_density": result.noninteracting_density,
        "density_delta": result.density_delta,
        "total_density": result.total_density,
        "sigma_fock": result.sigma_fock_mev,
        "hamiltonian": result.hamiltonian_mev,
        "energies": result.energies_mev,
        "iter_err": result.run.iter_err,
        "iter_energy": result.run.iter_energy,
        "iter_oda": result.run.iter_oda,
        "scalars": np.array(
            [
                result.reference_mu_mev,
                result.mu_mev,
                result.one_body_internal_energy_density_mev_nm2,
                result.fock_internal_energy_density_mev_nm2,
                result.total_internal_energy_density_mev_nm2,
                result.entropy_difference_mev_per_K_nm2,
                result.canonical_free_energy_difference_mev_nm2,
                result.grand_potential_difference_mev_nm2,
                result.run.state.diagnostics["final_raw_norm"],
            ]
        ),
    }
    with np.load(fixture_path, allow_pickle=False) as expected:
        assert set(expected.files) == set(current)
        for name, values in current.items():
            np.testing.assert_allclose(values, expected[name], rtol=0.0, atol=2.0e-14)
