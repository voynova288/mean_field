"""Lightweight Test001 checks for the isolated Vituri-2024 adapter."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mean_field.systems.abc_trilayer.vituri2024 import (
    ARXIV_PDF_SHA256,
    ARXIV_SOURCE_SHA256,
    BASIS,
    PDF_AUTHORITY_PATH,
    SM_TEX_AUTHORITY_PATH,
    SM_TEX_SHA256,
    SPIN_STIFFNESS_CHECKPOINTS,
    VITURI2024_PARAMETERS,
    VITURI2024_SPEC,
    UnresolvedVituriAuthorityError,
    Vituri2024Parameters,
    c3_basis_operator,
    six_band_hamiltonian,
    state_projector,
    third_lowest_active_band,
)
from mean_field.systems.abc_trilayer.vituri2024 import (
    _require_local_third_band_nondegeneracy,
)


def _rotate_c3(k: np.ndarray) -> np.ndarray:
    angle = 2.0 * np.pi / 3.0
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return rotation @ k


def test001_exact_ham6_entries_and_hermiticity() -> None:
    p = VITURI2024_PARAMETERS
    k = np.array([0.017, -0.023])
    tau = -1
    Delta1 = 0.026
    pi = tau * k[0] + 1j * k[1]
    v0 = np.sqrt(3.0) * p.a0 * p.gamma0 / 2.0
    v3 = np.sqrt(3.0) * p.a0 * p.gamma3 / 2.0
    v4 = np.sqrt(3.0) * p.a0 * p.gamma4 / 2.0

    expected = np.array(
        [
            [
                Delta1 + p.Delta2 + p.delta,
                p.gamma2 / 2.0,
                v0 * pi.conjugate(),
                v4 * pi.conjugate(),
                v3 * pi,
                0.0,
            ],
            [
                p.gamma2 / 2.0,
                p.Delta2 - Delta1 + p.delta,
                0.0,
                v3 * pi.conjugate(),
                v4 * pi,
                v0 * pi,
            ],
            [
                v0 * pi,
                0.0,
                Delta1 + p.Delta2,
                p.gamma1,
                v4 * pi.conjugate(),
                0.0,
            ],
            [
                v4 * pi,
                v3 * pi,
                p.gamma1,
                -2.0 * p.Delta2,
                v0 * pi.conjugate(),
                v4 * pi.conjugate(),
            ],
            [
                v3 * pi.conjugate(),
                v4 * pi.conjugate(),
                v4 * pi,
                v0 * pi,
                -2.0 * p.Delta2,
                p.gamma1,
            ],
            [
                0.0,
                v0 * pi.conjugate(),
                0.0,
                v4 * pi,
                p.gamma1,
                p.Delta2 - Delta1,
            ],
        ],
        dtype=np.complex128,
    )
    actual = six_band_hamiltonian(k, tau, Delta1)

    assert BASIS == ("A1", "B3", "B1", "A2", "B2", "A3")
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual, actual.conjugate().T, rtol=0.0, atol=0.0)


def test001_time_reversal_and_c3_spectral_covariance() -> None:
    k = np.array([0.031, -0.014])
    Delta1 = 0.019

    for tau in (-1, 1):
        h = six_band_hamiltonian(k, tau, Delta1)
        time_reversed = six_band_hamiltonian(-k, -tau, Delta1)
        np.testing.assert_allclose(time_reversed, h.conjugate(), rtol=0.0, atol=0.0)

        rotated = six_band_hamiltonian(_rotate_c3(k), tau, Delta1)
        c3 = c3_basis_operator(tau)
        np.testing.assert_allclose(
            rotated,
            c3 @ h @ c3.conjugate().T,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        projector = third_lowest_active_band(k, tau, Delta1).projector
        rotated_projector = third_lowest_active_band(
            _rotate_c3(k), tau, Delta1
        ).projector
        np.testing.assert_allclose(
            rotated_projector,
            c3 @ projector @ c3.conjugate().T,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        time_reversed_projector = third_lowest_active_band(
            -k, -tau, Delta1
        ).projector
        np.testing.assert_allclose(
            time_reversed_projector,
            projector.conjugate(),
            rtol=2.0e-13,
            atol=2.0e-13,
        )


def test001_third_band_residual_projector_and_gauge_invariance() -> None:
    k = np.array([0.021, 0.013])
    tau = 1
    Delta1 = 0.024
    h = six_band_hamiltonian(k, tau, Delta1)
    solution = third_lowest_active_band(k, tau, Delta1)

    assert solution.band_index_zero_based == 2
    assert solution.lower_gap > 0.0
    assert solution.upper_gap > 0.0
    assert solution.energy == pytest.approx(np.linalg.eigvalsh(h)[2], abs=1.0e-15)
    np.testing.assert_allclose(
        h @ solution.eigenvector,
        solution.energy * solution.eigenvector,
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        solution.projector.conjugate().T,
        solution.projector,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        solution.projector @ solution.projector,
        solution.projector,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.trace(solution.projector).real == pytest.approx(1.0, abs=2.0e-15)

    phase_changed = np.exp(0.731j) * solution.eigenvector
    rescaled = (2.4 - 0.7j) * solution.eigenvector
    np.testing.assert_allclose(
        state_projector(phase_changed), solution.projector, rtol=0.0, atol=2.0e-15
    )
    np.testing.assert_allclose(
        state_projector(rescaled), solution.projector, rtol=0.0, atol=2.0e-15
    )


def test001_strict_inputs_and_locked_parameters() -> None:
    for bad_valley in (0, 2, -2, 1.0, True, np.nan, "1"):
        with pytest.raises(ValueError):
            six_band_hamiltonian([0.0, 0.0], bad_valley, 0.01)  # type: ignore[arg-type]

    for bad_k in (
        [0.0],
        [0.0, 0.0, 0.0],
        [np.nan, 0.0],
        [0.0, np.inf],
        [0.0 + 0.0j, 0.0 + 0.0j],
    ):
        with pytest.raises(ValueError):
            six_band_hamiltonian(bad_k, 1, 0.01)

    for bad_delta1 in (np.nan, np.inf, -np.inf, 0.01 + 0.0j, [0.01]):
        with pytest.raises(ValueError):
            six_band_hamiltonian([0.0, 0.0], 1, bad_delta1)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        state_projector(np.zeros(6))
    with pytest.raises(ValueError):
        state_projector(np.full(6, np.nan))

    parameters = Vituri2024Parameters()
    assert parameters == Vituri2024Parameters(
        a0=2.46,
        gamma0=3.1,
        gamma1=0.38,
        gamma2=-0.022,
        gamma3=-0.29,
        gamma4=-0.21,
        delta=-0.0105,
        Delta2=-0.0023,
    )
    with pytest.raises(FrozenInstanceError):
        parameters.gamma0 = 0.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        Vituri2024Parameters(gamma0=np.nan)
    with pytest.raises(ValueError, match="paper-direct parameters"):
        Vituri2024Parameters(gamma0=3.0)
    with pytest.raises(RuntimeError, match="locally nondegenerate"):
        _require_local_third_band_nondegeneracy(
            [-2.0, -1.0, 0.0, 0.0, 1.0, 2.0]
        )


def test001_pinned_authority_and_fail_closed_unresolved_list() -> None:
    assert ARXIV_SOURCE_SHA256 == (
        "c01d805c463e388989370a202f04f4f27ceb38a668294e1959e172c8fc9932f9"
    )
    assert ARXIV_PDF_SHA256 == (
        "ec761a2b494a8e5983ff3fb6cfb842e114526cc0ba8b3e7cdc7c128f5d204bc8"
    )
    assert SM_TEX_SHA256 == (
        "f2847fa3dc14590f4157dd82ac6983ace39328a620f55cf75d4db51f1a43be45"
    )
    assert SM_TEX_AUTHORITY_PATH == "SM.tex"
    assert PDF_AUTHORITY_PATH == "reference/2408.10309v1.pdf"
    assert "/tmp/" not in SM_TEX_AUTHORITY_PATH
    assert "/tmp/" not in PDF_AUTHORITY_PATH

    expected_unresolved = {
        "paper_gauge_basis_index_conflict",
        "spin_stiffness_density_normalization",
        "momentum_axis_and_valley_center_convention",
        "gate_distance_d",
        "uv_domain_and_cutoff",
        "interaction_q0_policy",
        "figure_q0_momentum",
        "mesh_and_quadrature",
        "ensemble_and_source",
        "cdw_harmonic_cutoff_and_q_scan",
        "scf_policy",
        "exact_tdhf_q_tolerances_and_provider",
    }
    assert set(VITURI2024_SPEC.unresolved_keys) == expected_unresolved
    assert VITURI2024_SPEC.paper_gauge_imposed is False
    assert VITURI2024_SPEC.production_ready is False
    conflict = VITURI2024_SPEC.unresolved_authority[0]
    assert conflict.key == "paper_gauge_basis_index_conflict"
    assert "B3" in conflict.detail
    assert "index 2" in conflict.detail
    assert "psi_6" in conflict.detail
    assert "U_{6,3}" in conflict.detail
    with pytest.raises(UnresolvedVituriAuthorityError):
        VITURI2024_SPEC.require_resolved()
    with pytest.raises(ValueError):
        type(VITURI2024_SPEC)(unresolved_authority=())
    with pytest.raises(ValueError):
        type(VITURI2024_SPEC)(paper_gauge_imposed=True)

    checkpoints = {item.quantity: item for item in SPIN_STIFFNESS_CHECKPOINTS}
    assert checkpoints["2 rho_s / |n|"].value == 780.0
    assert checkpoints["2 rho_s / |n|"].unit == "meV*a0^2"
    assert checkpoints["rho_s"].value == 0.28
    assert checkpoints["rho_s"].unit == "meV"
    assert checkpoints["2 rho_s / |n|"].qualifier == (
        "paper_approximate_not_acceptance_threshold"
    )
    assert checkpoints["rho_s"].qualifier == (
        "source_reported_density_normalization_conflict_not_threshold"
    )
    with pytest.raises(ValueError, match="checkpoints may not be changed"):
        type(VITURI2024_SPEC)(spin_stiffness_checkpoints=())
