from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.inas_gasb.normal_reference import (
    ChemicalPotentialRootTolerances,
    ChemicalPotentialShiftCovariance,
    KaneNormalReferenceContract,
    build_charge_neutral_kane_reference,
    chemical_potential_shift_covariance,
    evaluate_kane_state_at_fixed_mu,
    kane_window_source_fingerprint,
    normal_reference_fingerprint,
)


def _unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    trial = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(trial)
    phase = np.diag(r)
    return q * np.where(np.abs(phase) > 0.0, phase.conj() / np.abs(phase), 1.0)[None, :]


def _orthonormal_kane_window(
    *,
    nk: int = 3,
    nz: int = 2,
    nband: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(417)
    z_weights = np.array([0.4, 0.6], dtype=float)
    assert z_weights.size == nz
    metric_sqrt = np.repeat(np.sqrt(z_weights), 8)
    frames = np.empty((nk, nz, 8, nband), dtype=np.complex128)
    hamiltonian = np.empty((nband, nband, nk), dtype=np.complex128)
    base = np.linspace(-4.0, 4.0, nband)
    for ik in range(nk):
        trial = rng.normal(size=(8 * nz, nband)) + 1j * rng.normal(size=(8 * nz, nband))
        weighted_q, _r = np.linalg.qr(trial)
        physical_q = weighted_q / metric_sqrt[:, None]
        frames[ik] = physical_q.reshape(nz, 8, nband)
        rotation = _unitary(nband, rng)
        diagonal = np.diag(base + 0.17 * ik * np.arange(nband))
        hamiltonian[:, :, ik] = rotation.conj().T @ diagonal @ rotation
    k_weights = np.array([0.03, 0.07, 0.11], dtype=float)
    return hamiltonian, k_weights, frames, z_weights


def _contract(
    hamiltonian: np.ndarray,
    k_weights: np.ndarray,
    frames: np.ndarray,
    z_weights: np.ndarray,
    **changes: object,
) -> KaneNormalReferenceContract:
    values: dict[str, object] = {
        "source_fingerprint": kane_window_source_fingerprint(
            hamiltonian, k_weights, frames, z_weights
        ),
        "selected_band_indices": (-3, -2, -1, 1, 2, 3),
        "parent_hilbert_dimension": 16,
        "calculation_split_mev": 0.0,
        "energy_zero_label": "synthetic Kane source gauge",
        "electrostatic_ensemble": "diagnostic",
        "fixed_gate_source_complete": False,
        "band_window_converged": False,
        "momentum_window_converged": False,
        "z_mesh_converged": False,
        "poisson_residual_mev": None,
        "radial_only": True,
    }
    values.update(changes)
    return KaneNormalReferenceContract(**values)


def _direct_microscopic_carriers(
    frames: np.ndarray,
    z_weights: np.ndarray,
    density: np.ndarray,
    k_weights: np.ndarray,
) -> tuple[float, float]:
    electron = 0.0
    hole = 0.0
    identity = np.eye(density.shape[0])[:, :, None]
    for ik in range(density.shape[2]):
        occupied_orbital = np.einsum(
            "zma,ab,znb->zmn",
            frames[ik],
            density[:, :, ik],
            frames[ik].conj(),
            optimize=True,
        )
        vacant_orbital = np.einsum(
            "zma,ab,znb->zmn",
            frames[ik],
            (identity - density)[:, :, ik],
            frames[ik].conj(),
            optimize=True,
        )
        electron += k_weights[ik] * np.einsum(
            "z,zmm->", z_weights, occupied_orbital[:, :2, :2], optimize=True
        ).real
        hole += k_weights[ik] * np.einsum(
            "z,zmm->", z_weights, vacant_orbital[:, 2:, 2:], optimize=True
        ).real
    return float(electron), float(hole)


def test_multiband_kane_reference_solves_note6_neutrality_and_number_identity() -> None:
    hamiltonian, k_weights, frames, z_weights = _orthonormal_kane_window()
    reference = build_charge_neutral_kane_reference(
        hamiltonian,
        k_weights,
        frames,
        z_weights,
        contract=_contract(hamiltonian, k_weights, frames, z_weights),
        temperature_K=0.2,
    )
    assert reference.numerically_certified
    assert reference.claim_scope == "window_restricted_numerical_charge_neutrality_root"
    assert abs(reference.electron_density_nm2 - reference.hole_density_nm2) < 1e-12
    assert abs(reference.number_residual_nm2) < reference.numerical_residual_tolerance_nm2
    assert abs(reference.number_residual_nm2 - reference.neutrality_residual_nm2) < 1e-13
    assert reference.bracket_residuals_nm2[0] < 0.0 < reference.bracket_residuals_nm2[1]
    assert reference.root_converged and reference.root_iterations > 0
    direct_e, direct_h = _direct_microscopic_carriers(
        frames, z_weights, reference.density, k_weights
    )
    np.testing.assert_allclose(reference.electron_density_nm2, direct_e, atol=1e-12)
    np.testing.assert_allclose(reference.hole_density_nm2, direct_h, atol=1e-12)
    assert len(normal_reference_fingerprint(reference)) == 64


def test_hole_density_uses_identity_complement_for_near_orthonormal_frame() -> None:
    hamiltonian, k_weights, frames, z_weights = _orthonormal_kane_window()
    perturbed = frames.copy()
    perturbed[:, :, :, 0] *= np.sqrt(1.0 + 5.0e-9)
    reference = build_charge_neutral_kane_reference(
        hamiltonian,
        k_weights,
        perturbed,
        z_weights,
        contract=_contract(hamiltonian, k_weights, perturbed, z_weights),
        temperature_K=0.2,
    )
    assert 1e-9 < reference.orbital_projector_completeness_error < 1e-8
    direct_e, direct_h = _direct_microscopic_carriers(
        perturbed, z_weights, reference.density, k_weights
    )
    np.testing.assert_allclose(reference.electron_density_nm2, direct_e, atol=1e-12)
    np.testing.assert_allclose(reference.hole_density_nm2, direct_h, atol=1e-12)
    assert abs(reference.number_residual_nm2 - reference.neutrality_residual_nm2) < 1e-13


def test_kane_reference_is_covariant_under_window_basis_rotations() -> None:
    hamiltonian, k_weights, frames, z_weights = _orthonormal_kane_window()
    contract = _contract(hamiltonian, k_weights, frames, z_weights)
    baseline = build_charge_neutral_kane_reference(
        hamiltonian,
        k_weights,
        frames,
        z_weights,
        contract=contract,
        temperature_K=0.2,
    )
    rng = np.random.default_rng(991)
    rotated_h = np.empty_like(hamiltonian)
    rotated_phi = np.empty_like(frames)
    rotations = []
    for ik in range(hamiltonian.shape[2]):
        gauge = _unitary(hamiltonian.shape[0], rng)
        rotations.append(gauge)
        rotated_h[:, :, ik] = gauge.conj().T @ hamiltonian[:, :, ik] @ gauge
        rotated_phi[ik] = np.einsum("zma,ab->zmb", frames[ik], gauge)
    rotated = build_charge_neutral_kane_reference(
        rotated_h,
        k_weights,
        rotated_phi,
        z_weights,
        contract=_contract(rotated_h, k_weights, rotated_phi, z_weights),
        temperature_K=0.2,
    )
    np.testing.assert_allclose(rotated.mu_mev, baseline.mu_mev, atol=1e-12)
    np.testing.assert_allclose(
        rotated.electron_density_nm2, baseline.electron_density_nm2, atol=1e-12
    )
    np.testing.assert_allclose(rotated.hole_density_nm2, baseline.hole_density_nm2, atol=1e-12)
    for ik, gauge in enumerate(rotations):
        expected = gauge.conj().T @ baseline.density[:, :, ik] @ gauge
        np.testing.assert_allclose(rotated.density[:, :, ik], expected, atol=2e-12)


def test_kane_reference_chemical_potential_is_energy_zero_covariant() -> None:
    hamiltonian, k_weights, frames, z_weights = _orthonormal_kane_window()
    contract = _contract(hamiltonian, k_weights, frames, z_weights)
    diagnostics = chemical_potential_shift_covariance(
        hamiltonian,
        k_weights,
        frames,
        z_weights,
        contract=contract,
        temperature_K=0.2,
        energy_shift_mev=7.25,
    )
    assert diagnostics.passed
    assert len(diagnostics.baseline_reference_fingerprint) == 64
    assert len(diagnostics.shifted_source_fingerprint) == 64
    assert diagnostics.mu_shift_error_mev < 1e-12
    assert diagnostics.density_max_error < 1e-12
    with pytest.raises(ValueError, match="too small"):
        chemical_potential_shift_covariance(
            hamiltonian,
            k_weights,
            frames,
            z_weights,
            contract=contract,
            temperature_K=0.2,
            energy_shift_mev=0.0,
        )


def test_carrier_covariance_bound_uses_observable_scale_not_error_scale() -> None:
    tolerances = ChemicalPotentialRootTolerances(covariance_rtol=1e-6)
    common = {
        "baseline_reference_fingerprint": "a" * 64,
        "shifted_source_fingerprint": "b" * 64,
        "tolerances": tolerances,
        "energy_shift_mev": 2.0,
        "baseline_mu_mev": 1.0,
        "shifted_mu_mev": 3.0,
        "baseline_electron_density_nm2": 100.0,
        "shifted_electron_density_nm2": 100.00005,
        "baseline_hole_density_nm2": 80.0,
        "shifted_hole_density_nm2": 80.00004,
        "mu_shift_error_mev": 0.0,
        "energy_shift_error_mev": 0.0,
        "density_max_error": 0.0,
    }
    passing = ChemicalPotentialShiftCovariance(
        **common,
        electron_density_error_nm2=5e-5,
        hole_density_error_nm2=4e-5,
    )
    assert passing.passed
    failing = ChemicalPotentialShiftCovariance(
        **common,
        electron_density_error_nm2=2e-4,
        hole_density_error_nm2=2e-4,
    )
    assert not failing.passed


def test_fixed_gate_reservoir_uses_supplied_mu_and_never_root_solves_it() -> None:
    hamiltonian, k_weights, frames, z_weights = _orthonormal_kane_window()
    fixed_contract = _contract(
        hamiltonian,
        k_weights,
        frames,
        z_weights,
        electrostatic_ensemble="fixed_gate_reservoir",
        fixed_gate_source_complete=True,
    )
    with pytest.raises(ValueError, match="supplied mu"):
        build_charge_neutral_kane_reference(
            hamiltonian,
            k_weights,
            frames,
            z_weights,
            contract=fixed_contract,
            temperature_K=0.2,
        )
    state = evaluate_kane_state_at_fixed_mu(
        hamiltonian,
        k_weights,
        frames,
        z_weights,
        contract=fixed_contract,
        temperature_K=0.2,
        reservoir_mu_mev=1.234,
    )
    assert state.reservoir_mu_mev == 1.234
    assert state.claim_scope == "fixed_reservoir_mu_charge_evaluation_not_cnp_root"


def test_contract_reports_blockers_but_does_not_self_authorize_physics() -> None:
    hamiltonian, k_weights, frames, z_weights = _orthonormal_kane_window()
    diagnostic = _contract(
        hamiltonian, k_weights, frames, z_weights, calculation_split_mev=0.01
    )
    blockers = diagnostic.credibility_blockers()
    assert "calculation_split_is_nonzero" in blockers
    assert "electrostatic_ensemble_is_diagnostic" in blockers
    assert "band_window_is_not_converged" in blockers
    assert "momentum_window_is_not_converged" in blockers
    assert "z_mesh_is_not_converged" in blockers
    assert "source_is_radial_only" in blockers
    assert "poisson_residual_is_missing" in blockers

    declared_complete = _contract(
        hamiltonian,
        k_weights,
        frames,
        z_weights,
        electrostatic_ensemble="fixed_gate_reservoir",
        fixed_gate_source_complete=True,
        band_window_converged=True,
        momentum_window_converged=True,
        z_mesh_converged=True,
        poisson_residual_mev=1e-8,
        radial_only=False,
    )
    assert declared_complete.credibility_blockers() == ()
    assert not hasattr(declared_complete, "physical_chemical_potential_authorized")


def test_normal_reference_rejects_unbound_or_invalid_source_data() -> None:
    hamiltonian, k_weights, frames, z_weights = _orthonormal_kane_window()
    contract = _contract(hamiltonian, k_weights, frames, z_weights)
    bad = frames.copy()
    bad[:, :, :, 0] *= 2.0
    with pytest.raises(ValueError, match="does not bind"):
        build_charge_neutral_kane_reference(
            hamiltonian,
            k_weights,
            bad,
            z_weights,
            contract=contract,
            temperature_K=0.2,
        )
    bound_bad = _contract(hamiltonian, k_weights, bad, z_weights)
    with pytest.raises(ValueError, match="not orthonormal"):
        build_charge_neutral_kane_reference(
            hamiltonian,
            k_weights,
            bad,
            z_weights,
            contract=bound_bad,
            temperature_K=0.2,
        )
    with pytest.raises(ValueError, match="shape"):
        truncated = frames[:, :, :7, :]
        build_charge_neutral_kane_reference(
            hamiltonian,
            k_weights,
            truncated,
            z_weights,
            contract=_contract(hamiltonian, k_weights, truncated, z_weights),
            temperature_K=0.2,
        )


def test_normal_reference_contract_and_tolerances_reject_false_metadata() -> None:
    hamiltonian, k_weights, frames, z_weights = _orthonormal_kane_window()
    with pytest.raises(ValueError, match="smaller"):
        _contract(hamiltonian, k_weights, frames, z_weights, parent_hilbert_dimension=5)
    with pytest.raises(ValueError, match="unique nonzero"):
        _contract(
            hamiltonian,
            k_weights,
            frames,
            z_weights,
            selected_band_indices=(-2, -1, 1, 1, 2, 3),
        )
    with pytest.raises(ValueError, match="energy_zero_label"):
        _contract(hamiltonian, k_weights, frames, z_weights, energy_zero_label="")
    with pytest.raises(ValueError, match="orbital labels"):
        _contract(
            hamiltonian,
            k_weights,
            frames,
            z_weights,
            kane_orbital_labels=tuple(reversed(KaneNormalReferenceContract.__dataclass_fields__["kane_orbital_labels"].default)),
        )
    with pytest.raises(ValueError, match="float64 minimum"):
        ChemicalPotentialRootTolerances(rtol=1e-20)
    with pytest.raises(ValueError, match="smaller than one"):
        ChemicalPotentialRootTolerances(covariance_rtol=1.0)
