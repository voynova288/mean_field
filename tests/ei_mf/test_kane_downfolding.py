from __future__ import annotations

import numpy as np
import pytest

from ei_mf.kane_downfolding import (
    e1_h1_block_diagnostics,
    intersubspace_dhdk_diagnostics,
    intersubspace_operator_diagnostics,
    lowdin_effective_hamiltonian,
)


def _tilted_subspaces() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.eye(8, 4, dtype=complex)
    angles = np.array([0.1, 0.2, 0.3, 0.4])
    active = np.zeros((8, 4), dtype=complex)
    for i, theta in enumerate(angles):
        active[i, i] = np.cos(theta)
        active[4 + i, i] = np.sin(theta)
    energies = np.array([-2.0, -1.8, 1.7, 2.1])
    return reference, active, energies


def test_lowdin_downfolding_retains_spectrum_and_reports_leakage() -> None:
    reference, active, energies = _tilted_subspaces()
    result = lowdin_effective_hamiltonian(reference, active, energies)
    assert np.allclose(np.linalg.eigvalsh(result.h_eff_mev), energies)
    assert np.allclose(result.principal_cos2, np.cos([0.4, 0.3, 0.2, 0.1]) ** 2)
    assert result.hermiticity_error_mev < 1e-12
    assert result.spectrum_error_mev < 1e-12


def test_lowdin_downfolding_is_invariant_to_active_phases_and_permutation() -> None:
    reference, active, energies = _tilted_subspaces()
    baseline = lowdin_effective_hamiltonian(reference, active, energies)
    permutation = np.array([2, 0, 3, 1])
    phases = np.exp(1j * np.array([0.3, -0.7, 1.1, 0.2]))
    transformed = active[:, permutation] * phases[None, :]
    result = lowdin_effective_hamiltonian(
        reference, transformed, energies[permutation]
    )
    assert np.allclose(result.h_eff_mev, baseline.h_eff_mev, atol=1e-12)


def test_e1_h1_diagnostics_are_invariant_to_pair_basis_rotations() -> None:
    h = np.array(
        [
            [-2.0, 0.1, 0.3, 0.2j],
            [0.1, -1.8, -0.2j, 0.25],
            [0.3, 0.2j, 1.7, -0.1],
            [-0.2j, 0.25, -0.1, 2.1],
        ],
        dtype=complex,
    )
    qe, _ = np.linalg.qr(np.array([[1.0, 1.0j], [0.2j, 1.0]], dtype=complex))
    qh, _ = np.linalg.qr(np.array([[1.0j, 0.3], [1.0, -0.2j]], dtype=complex))
    rotation = np.block([[qe, np.zeros((2, 2))], [np.zeros((2, 2)), qh]])
    baseline = e1_h1_block_diagnostics(h)
    rotated = e1_h1_block_diagnostics(rotation.conj().T @ h @ rotation)
    for key in baseline:
        assert np.isclose(rotated[key], baseline[key])


def test_intersubspace_operator_is_invariant_to_internal_rotations() -> None:
    lower = np.eye(5, 2, dtype=complex)
    upper = np.eye(5, dtype=complex)[:, 2:5]
    operator = np.array(
        [
            [0, 0, 1, 2j, 0],
            [0, 0, -1j, 0, 3],
            [1, 1j, 0, 0, 0],
            [-2j, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
        ],
        dtype=complex,
    )
    baseline = intersubspace_operator_diagnostics(lower, upper, operator)
    ql, _ = np.linalg.qr(np.array([[1, 1j], [0.3j, 1]], dtype=complex))
    qu, _ = np.linalg.qr(
        np.array([[1, 0.2j, 0.1], [0.3, 1, -0.2j], [0.1j, 0.4, 1]], dtype=complex)
    )
    rotated = intersubspace_operator_diagnostics(lower@ql, upper@qu, operator)
    for key in baseline:
        assert np.isclose(rotated[key], baseline[key], atol=1e-12)


def test_intersubspace_dhdk_norm_and_basis_invariance() -> None:
    h = np.diag([-2.0, -1.0, 1.0, 2.0]).astype(complex)
    coupling = np.diag([3.0, 4.0]).astype(complex)
    derivative = np.block(
        [[np.zeros((2, 2)), coupling], [coupling.conj().T, np.zeros((2, 2))]]
    )
    baseline = intersubspace_dhdk_diagnostics(h, derivative)
    assert np.isclose(baseline["direct_gap_meV"], 2.0)
    assert np.isclose(baseline["intersubspace_frobenius_meV_nm"], 5.0)
    assert np.isclose(
        baseline["intersubspace_rms_per_lower_state_meV_nm"], 5.0 / np.sqrt(2.0)
    )
    assert np.isclose(baseline["intersubspace_max_singular_meV_nm"], 4.0)

    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    rotated = intersubspace_dhdk_diagnostics(
        q.conj().T @ h @ q,
        q.conj().T @ derivative @ q,
    )
    for key in baseline:
        assert np.isclose(rotated[key], baseline[key], atol=1e-12)


def test_lowdin_downfolding_rejects_rank_deficient_overlap() -> None:
    reference, active, energies = _tilted_subspaces()
    active[:, -1] = 0.0
    active[7, -1] = 1.0
    with pytest.raises(ValueError, match="rank-deficient"):
        lowdin_effective_hamiltonian(reference, active, energies)
