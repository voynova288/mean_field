from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from mean_field.systems.inas_gasb.axial_fock import (
    AxialAveragedProjectedFockOperator,
    AxialE1H1CoherenceFockSuperoperator,
    AxialProjectedFockOperator,
    co_rotating_mode_alias,
    precompute_axial_harmonic_exchange_tensor_mev_nm2,
    precompute_axial_harmonic_exchange_tensors_mev_nm2,
)
from mean_field.systems.inas_gasb.axial import (
    AxialRotationSpec,
    axial_radial_time_reversal_error,
    axial_radial_time_reversal_residuals,
    expand_axial_radial_bundle,
    expand_axial_radial_matrices,
    project_axial_radial_time_reversal_matrices,
)
from mean_field.systems.inas_gasb.angular_fock import (
    CoRotatingHarmonicFockOperator,
    PolarHarmonicProjectedFockOperator,
)
from mean_field.systems.inas_gasb.conventions import (
    E1H1BasisSpec,
    KANE8_JZ,
    kane8_time_reversal_unitary,
)
from mean_field.systems.inas_gasb.kane4_bundle import Kane4Bundle
from mean_field.systems.inas_gasb.projected_fock import (
    ProjectedFockOperator,
    apply_e1h1_coherence_exchange_superoperator,
    e1h1_coherence_exchange_tensors_mev_nm2,
    hermitian_density_from_e1h1_coherence,
    uniform_dielectric_green_on_mesh_mev_nm2,
)
from mean_field.systems.inas_gasb.matrix_ei import (
    MatrixEIConfig,
    solve_reference_subtracted_matrix_ei,
)
def _detuned_bundle(bundle: Kane4Bundle, delta_tau_z_mev: float) -> Kane4Bundle:
    h0 = np.asarray(bundle.h0_mev, dtype=np.complex128).copy()
    h0 += 0.5 * float(delta_tau_z_mev) * bundle.basis.tau_z[:, :, None]
    return replace(bundle, h0_mev=h0)


def _dense_radial_tr_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trial = rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
    spec = AxialRotationSpec()
    micro_tr = kane8_time_reversal_unitary()
    active_tr = np.kron(
        np.eye(2), np.asarray([[0.0, -1.0], [1.0, 0.0]])
    ).astype(np.complex128)
    r_pi = np.diag(np.exp(-1j * KANE8_JZ * np.pi))
    u_pi = np.diag(np.exp(-1j * spec.active_jz * np.pi))

    def radial_tr(values: np.ndarray) -> np.ndarray:
        return (
            r_pi.conj().T
            @ micro_tr
            @ values.conj()
            @ active_tr.conj().T
            @ u_pi
        )

    projected = 0.5 * (trial + radial_tr(trial))
    overlap = projected.conj().T @ projected
    eigenvalues, vectors = np.linalg.eigh(overlap)
    frame = projected @ (
        (vectors * (1.0 / np.sqrt(eigenvalues))[None, :]) @ vectors.conj().T
    )
    np.testing.assert_allclose(radial_tr(frame), frame, atol=2e-12)
    return frame


def _radial_bundle() -> Kane4Bundle:
    nr = 2
    dense_frame = _dense_radial_tr_frame(seed=37)
    frame = np.repeat(dense_frame[None, None, :, :], nr, axis=0)
    h0 = np.repeat(
        np.diag([-1.2, -1.2, 0.8, 0.8])[:, :, None], nr, axis=2
    ).astype(np.complex128)
    tr = np.kron(
        np.eye(2), np.asarray([[0.0, -1.0], [1.0, 0.0]])
    ).astype(np.complex128)
    return Kane4Bundle(
        k_cart_nm_inv=np.array([[0.02, 0.0], [0.04, 0.0]]),
        weights_nm2=np.array([2.0e-4, 2.0e-4]),
        z_nm=np.array([0.0, 1.0]),
        z_weights_nm=np.array([0.5, 0.5]),
        h0_mev=h0,
        micro_wavefunctions=np.repeat(frame, 2, axis=1),
        time_reversal_unitary=tr,
        provenance={
            "source": "analytic-axial-fock-test",
            "radial_only": True,
            "nr": nr,
            "k_max_nm_inv": 0.05,
        },
    )


def _random_hermitian(n: int, nk: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, n, nk)) + 1j * rng.normal(
        size=(n, n, nk)
    )
    return 0.5 * (values + np.swapaxes(values.conj(), 0, 1))


def _random_complex(n: int, nk: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n, nk)) + 1j * rng.normal(size=(n, n, nk))


def _dagger_radial(values: np.ndarray) -> np.ndarray:
    return np.swapaxes(np.asarray(values).conj(), 0, 1)


def _active_rotation(angle: float) -> np.ndarray:
    spec = AxialRotationSpec()
    return np.diag(
        np.exp(
            1j
            * int(spec.exponent_sign)
            * np.asarray(spec.active_jz, dtype=float)
            * float(angle)
        )
    )


def _expand_co_rotating_physical_pair(
    mode_matrix: np.ndarray, *, mode: int, nphi: int
) -> tuple[np.ndarray, np.ndarray]:
    nactive, _, nr = mode_matrix.shape
    density_lab = np.empty(
        (nactive, nactive, nr * nphi), dtype=np.complex128
    )
    density_co_rotating = np.empty(
        (nphi, nactive, nactive, nr), dtype=np.complex128
    )
    angles = 2.0 * np.pi * np.arange(nphi, dtype=float) / float(nphi)
    for ir in range(nr):
        for iangle, angle in enumerate(angles):
            phase_component = np.exp(1j * mode * angle) * mode_matrix[:, :, ir]
            co_rotating = phase_component + phase_component.conj().T
            active_u = _active_rotation(float(angle))
            density_co_rotating[iangle, :, :, ir] = co_rotating
            density_lab[:, :, ir * nphi + iangle] = (
                active_u @ co_rotating @ active_u.conj().T
            )
    return density_lab, density_co_rotating


def _co_rotating_from_lab(
    matrices_lab: np.ndarray, *, nr: int, nphi: int
) -> np.ndarray:
    nactive = matrices_lab.shape[0]
    output = np.empty(
        (nphi, nactive, nactive, nr), dtype=np.complex128
    )
    angles = 2.0 * np.pi * np.arange(nphi, dtype=float) / float(nphi)
    for ir in range(nr):
        for iangle, angle in enumerate(angles):
            active_u = _active_rotation(float(angle))
            matrix_lab = matrices_lab[:, :, ir * nphi + iangle]
            output[iangle, :, :, ir] = (
                active_u.conj().T @ matrix_lab @ active_u
            )
    return output


@pytest.fixture(scope="module")
def harmonic_pair_oracle() -> dict[str, object]:
    radial = _radial_bundle()
    nphi = 6
    mode = 1
    epsilon_r = 15.0
    plus = CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
        radial, mode=mode, nphi=nphi, epsilon_r=epsilon_r
    )
    minus = CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
        radial, mode=-mode, nphi=nphi, epsilon_r=epsilon_r
    )
    full_bundle = expand_axial_radial_bundle(radial, nphi=nphi)
    green = uniform_dielectric_green_on_mesh_mev_nm2(
        full_bundle.k_cart_nm_inv,
        full_bundle.weights_nm2,
        full_bundle.z_nm,
        epsilon_r=epsilon_r,
    )
    full = ProjectedFockOperator.from_bundle(
        full_bundle,
        green,
        self_cell_description="full-2D oracle for co-rotating harmonics",
        precompute_exchange_tensor=True,
    )
    mode_matrix = _random_complex(4, radial.nk, seed=59)
    density_full, density_co_rotating = _expand_co_rotating_physical_pair(
        mode_matrix, mode=mode, nphi=nphi
    )
    return {
        "radial": radial,
        "nphi": nphi,
        "mode": mode,
        "plus": plus,
        "minus": minus,
        "full_bundle": full_bundle,
        "full": full,
        "mode_matrix": mode_matrix,
        "density_full": density_full,
        "density_co_rotating": density_co_rotating,
    }


def test_axial_fock_rejects_noncanonical_e1h1_partition() -> None:
    radial = replace(
        _radial_bundle(),
        basis=E1H1BasisSpec(
            electron_indices=(0, 2),
            hole_indices=(1, 3),
        ),
    )
    with pytest.raises(ValueError, match="canonical labels and E1"):
        AxialProjectedFockOperator.from_bundle_uniform_dielectric(
            radial,
            nphi=4,
            epsilon_r=15.0,
        )


def test_coherence_factory_rejects_stale_parent_attestation() -> None:
    parent = AxialProjectedFockOperator.from_bundle_uniform_dielectric(
        _radial_bundle(),
        nphi=4,
        epsilon_r=15.0,
    )
    object.__setattr__(parent, "epsilon_r", 16.0)
    with pytest.raises(ValueError, match="attestation is stale or invalid"):
        AxialE1H1CoherenceFockSuperoperator.from_axial_fock_operator(parent)


def test_e1h1_coherence_superoperator_matches_full_parent_and_impulses() -> None:
    radial = _radial_bundle()
    parent = AxialProjectedFockOperator.from_bundle_uniform_dielectric(
        radial,
        nphi=4,
        epsilon_r=15.0,
    )
    superoperator = AxialE1H1CoherenceFockSuperoperator.from_axial_fock_operator(
        parent
    )
    coherence = _random_complex(2, radial.nk, seed=137)
    density = hermitian_density_from_e1h1_coherence(coherence)
    sigma_full = parent(density)
    sigma_eh = sigma_full[np.ix_((0, 1), (2, 3), range(radial.nk))]

    np.testing.assert_allclose(superoperator(coherence), sigma_eh, atol=2e-13)
    np.testing.assert_allclose(
        sigma_full[np.ix_((2, 3), (0, 1), range(radial.nk))],
        np.swapaxes(sigma_eh.conj(), 0, 1),
        atol=2e-13,
    )
    superoperator.validate_against_parent(parent)
    assert superoperator.direct_impulse_error(parent) < 2e-13
    assert np.isfinite(superoperator.conjugate_to_direct_norm_ratio())


def test_e1h1_coherence_superoperator_scalar_and_conjugate_limits() -> None:
    nr = 2
    weights = np.asarray((0.2, 0.3))
    kernel = np.asarray(((1.0, 0.4), (0.6, 1.2)))
    coherence = _random_complex(2, nr, seed=139)
    tensor = np.zeros((nr, nr, 4, 4, 4, 4), dtype=np.complex128)
    for output_e in range(2):
        for output_h in range(2):
            tensor[:, :, output_e, output_h + 2, output_e, output_h + 2] = kernel
    direct, conjugate = e1h1_coherence_exchange_tensors_mev_nm2(tensor)
    actual = apply_e1h1_coherence_exchange_superoperator(
        coherence, direct, conjugate, weights
    )
    expected = -np.einsum("ij,abj,j->abi", kernel, coherence, weights)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    assert np.max(np.abs(conjugate)) == 0.0

    conjugate_tensor = np.zeros_like(tensor)
    for output_e in range(2):
        for output_h in range(2):
            conjugate_tensor[
                :, :, output_e, output_h + 2, output_h + 2, output_e
            ] = kernel
    direct, conjugate = e1h1_coherence_exchange_tensors_mev_nm2(
        conjugate_tensor
    )
    actual = apply_e1h1_coherence_exchange_superoperator(
        coherence, direct, conjugate, weights
    )
    expected = -np.einsum("ij,abj,j->abi", kernel, coherence.conj(), weights)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    assert np.max(np.abs(direct)) == 0.0


def test_e1h1_coherence_superoperator_is_block_gauge_covariant() -> None:
    radial = _radial_bundle()
    parent = AxialProjectedFockOperator.from_bundle_uniform_dielectric(
        radial,
        nphi=4,
        epsilon_r=15.0,
    )
    tensor = np.asarray(parent.exchange_tensor_mev_nm2)
    weights = np.asarray(parent.k_weights_nm2)
    nr = radial.nk
    gauge = np.zeros((nr, 4, 4), dtype=np.complex128)
    for ik in range(nr):
        electron, _ = np.linalg.qr(
            np.random.default_rng(151 + ik).normal(size=(2, 2))
            + 1j * np.random.default_rng(161 + ik).normal(size=(2, 2))
        )
        hole, _ = np.linalg.qr(
            np.random.default_rng(171 + ik).normal(size=(2, 2))
            + 1j * np.random.default_rng(181 + ik).normal(size=(2, 2))
        )
        gauge[ik, :2, :2] = electron
        gauge[ik, 2:, 2:] = hole
    transformed_tensor = np.einsum(
        "iaA,ijadbc,idD,jbB,jcC->ijADBC",
        gauge.conj(),
        tensor,
        gauge,
        gauge,
        gauge.conj(),
        optimize=True,
    )
    coherence = _random_complex(2, nr, seed=191)
    transformed_coherence = np.empty_like(coherence)
    for ik in range(nr):
        transformed_coherence[:, :, ik] = (
            gauge[ik, :2, :2].conj().T
            @ coherence[:, :, ik]
            @ gauge[ik, 2:, 2:]
        )
    direct, conjugate = e1h1_coherence_exchange_tensors_mev_nm2(tensor)
    transformed_direct, transformed_conjugate = (
        e1h1_coherence_exchange_tensors_mev_nm2(transformed_tensor)
    )
    output = apply_e1h1_coherence_exchange_superoperator(
        coherence, direct, conjugate, weights
    )
    transformed_output = apply_e1h1_coherence_exchange_superoperator(
        transformed_coherence,
        transformed_direct,
        transformed_conjugate,
        weights,
    )
    expected = np.empty_like(output)
    for ik in range(nr):
        expected[:, :, ik] = (
            gauge[ik, :2, :2].conj().T
            @ output[:, :, ik]
            @ gauge[ik, 2:, 2:]
        )
    np.testing.assert_allclose(transformed_output, expected, atol=3e-13)


def test_axial_radial_time_reversal_projection_and_source_sewing() -> None:
    bundle = _radial_bundle()
    residuals = axial_radial_time_reversal_residuals(bundle)
    assert residuals["h0_radial_time_reversal_error_mev"] < 1e-12
    assert residuals["frame_radial_time_reversal_error_nm_minus_half"] < 1e-12
    assert residuals["h0_radial_projector_error_mev"] < 1e-12

    matrices = _random_hermitian(4, bundle.nk, seed=41)
    projected = project_axial_radial_time_reversal_matrices(matrices, bundle)
    projected_twice = project_axial_radial_time_reversal_matrices(
        projected, bundle
    )
    np.testing.assert_allclose(projected_twice, projected, atol=2e-13)
    assert axial_radial_time_reversal_error(projected, bundle) < 2e-13


def test_axial_reduced_fock_matches_full_2d_action_and_energy() -> None:
    radial = _radial_bundle()
    nphi = 4
    reduced = AxialProjectedFockOperator.from_bundle_uniform_dielectric(
        radial, nphi=nphi, epsilon_r=15.0
    )
    full_bundle = expand_axial_radial_bundle(radial, nphi=nphi)
    green = uniform_dielectric_green_on_mesh_mev_nm2(
        full_bundle.k_cart_nm_inv,
        full_bundle.weights_nm2,
        full_bundle.z_nm,
        epsilon_r=15.0,
    )
    full = ProjectedFockOperator.from_bundle(
        full_bundle,
        green,
        self_cell_description="full-2D oracle for axial reduction",
        precompute_exchange_tensor=True,
    )

    density_radial = _random_hermitian(4, radial.nk, seed=43)
    assert np.linalg.norm(density_radial[:2, 2:]) > 0.1
    density_full = expand_axial_radial_matrices(density_radial, full_bundle)
    sigma_reduced = reduced(density_radial)
    sigma_full = full(density_full)
    phi0 = np.arange(radial.nk) * nphi
    np.testing.assert_allclose(
        sigma_reduced, sigma_full[:, :, phi0], rtol=2e-11, atol=2e-11
    )
    np.testing.assert_allclose(
        expand_axial_radial_matrices(sigma_reduced, full_bundle),
        sigma_full,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        reduced.energy_density_mev_nm2(density_radial),
        full.energy_density_mev_nm2(density_full),
        rtol=2e-11,
        atol=2e-11,
    )


def test_axial_averaged_polar_fock_keeps_only_co_rotating_m0() -> None:
    radial = _radial_bundle()
    nphi = 4
    polar = expand_axial_radial_bundle(radial, nphi=nphi)
    radial_operator = AxialProjectedFockOperator.from_bundle_uniform_dielectric(
        radial, nphi=nphi, epsilon_r=15.0
    )
    operator = AxialAveragedProjectedFockOperator.from_radial_operator(
        radial,
        polar,
        radial_operator,
        nphi=nphi,
    )
    density_radial = _random_hermitian(4, radial.nk, seed=101)
    density_polar = expand_axial_radial_matrices(density_radial, polar)
    expected_sigma = expand_axial_radial_matrices(
        radial_operator(density_radial), polar
    )
    np.testing.assert_allclose(operator(density_polar), expected_sigma, atol=2e-12)
    np.testing.assert_allclose(
        operator.energy_density_mev_nm2(density_polar),
        radial_operator.energy_density_mev_nm2(density_radial),
        atol=2e-12,
    )

    anisotropic = density_polar.reshape(4, 4, radial.nk, nphi).copy()
    perturbation = operator.expand_radial_matrices(
        _random_hermitian(4, radial.nk, seed=103)
    ).reshape(4, 4, radial.nk, nphi)
    for iphi in range(nphi):
        anisotropic[:, :, :, iphi] += (
            np.cos(2.0 * np.pi * iphi / nphi)
            * perturbation[:, :, :, iphi]
        )
    anisotropic = anisotropic.reshape(4, 4, radial.nk * nphi)
    np.testing.assert_allclose(operator(anisotropic), expected_sigma, atol=2e-12)
    operator.validate_against_bundle(polar)
    normal = solve_reference_subtracted_matrix_ei(
        polar,
        operator,
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.2,
            precision=1e-10,
            max_iter=5,
            search_mode="normal_reference",
            reference_policy="normal_ordered_exchange_only",
            normal_ordering_reference_policy="noninteracting_fermi_state",
        ),
    )
    assert normal.run.converged
    np.testing.assert_allclose(normal.density_delta, 0.0, atol=1e-14)
    with pytest.raises(ValueError, match="fingerprints differ"):
        operator.validate_against_bundle(_detuned_bundle(polar, 0.1))


def test_matrix_ei_accepts_certified_axial_reduced_fock_operator() -> None:
    radial = _radial_bundle()
    operator = AxialProjectedFockOperator.from_bundle_uniform_dielectric(
        radial, nphi=4, epsilon_r=15.0
    )
    result = solve_reference_subtracted_matrix_ei(
        radial,
        operator,
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.1,
            target_occupation_per_k=2.0,
            precision=1e-10,
            max_iter=5,
            search_mode="normal_reference",
            reference_policy="normal_ordered_exchange_only",
            normal_ordering_reference_policy="noninteracting_fermi_state",
        ),
        density_symmetrizer=lambda density: (
            project_axial_radial_time_reversal_matrices(density, radial)
        ),
        symmetry_constraint_label="axial-radial+time-reversal",
    )
    assert result.run.converged
    np.testing.assert_allclose(result.density_delta, 0.0, atol=1e-14)
    assert result.canonical_free_energy_difference_mev_nm2 == 0.0


def test_axial_reduced_fock_energy_derivative_tr_and_provenance() -> None:
    radial = _radial_bundle()
    with pytest.raises(ValueError, match="even integer"):
        AxialProjectedFockOperator.from_bundle_uniform_dielectric(
            radial, nphi=4.5, epsilon_r=14.5
        )
    operator = AxialProjectedFockOperator.from_bundle_uniform_dielectric(
        radial, nphi=6, epsilon_r=14.5
    )
    rehydrated = AxialProjectedFockOperator.from_precomputed_uniform_dielectric(
        radial,
        operator.exchange_tensor_mev_nm2,
        nphi=6,
        epsilon_r=14.5,
        expected_action_fingerprint=operator.fingerprint(),
    )
    assert rehydrated.fingerprint() == operator.fingerprint()
    np.testing.assert_array_equal(
        rehydrated.exchange_tensor_mev_nm2,
        operator.exchange_tensor_mev_nm2,
    )
    density = project_axial_radial_time_reversal_matrices(
        _random_hermitian(4, radial.nk, seed=47), radial
    )
    direction = project_axial_radial_time_reversal_matrices(
        _random_hermitian(4, radial.nk, seed=53), radial
    )
    sigma = operator(density)
    assert axial_radial_time_reversal_error(sigma, radial) < 2e-10
    analytic = np.einsum(
        "k,abk,bak->",
        radial.weights_nm2,
        sigma,
        direction,
        optimize=True,
    ).real
    step = 1e-7
    finite = (
        operator.energy_density_mev_nm2(density + step * direction)
        - operator.energy_density_mev_nm2(density - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(finite, analytic, rtol=2e-7, atol=2e-7)
    assert not operator.exchange_tensor_mev_nm2.flags.writeable
    assert not operator.k_weights_nm2.flags.writeable
    operator.validate_against_bundle(radial)
    with pytest.raises(ValueError, match="builder attestation"):
        replace(
            operator,
            exchange_tensor_mev_nm2=2.0 * operator.exchange_tensor_mev_nm2,
        )
    with pytest.raises(ValueError, match="independently pinned"):
        AxialProjectedFockOperator.from_precomputed_uniform_dielectric(
            radial,
            2.0 * operator.exchange_tensor_mev_nm2,
            nphi=6,
            epsilon_r=14.5,
            expected_action_fingerprint=operator.fingerprint(),
        )
    with pytest.raises(ValueError, match="fingerprints differ"):
        operator.validate_against_bundle(_detuned_bundle(radial, 0.2))


def test_co_rotating_harmonic_m0_is_existing_axial_action_bitwise() -> None:
    radial = _radial_bundle()
    nphi = 4
    axial = AxialProjectedFockOperator.from_bundle_uniform_dielectric(
        radial, nphi=nphi, epsilon_r=15.0
    )
    harmonic = CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
        radial, mode=0, nphi=nphi, epsilon_r=15.0
    )
    np.testing.assert_array_equal(
        harmonic.exchange_tensor_mev_nm2,
        axial.exchange_tensor_mev_nm2,
    )
    frozen = np.ascontiguousarray(axial.exchange_tensor_mev_nm2)
    frozen_digest = hashlib.sha256()
    frozen_digest.update(str(frozen.dtype).encode())
    frozen_digest.update(str(frozen.shape).encode())
    frozen_digest.update(frozen.view(np.uint8))
    assert frozen_digest.hexdigest() == (
        "2a33ff9843cd49551d1da1789deb6f1de56d25c7d37bcde4a0024a516f39a08e"
    )
    assert axial.fingerprint() == (
        "92cd26493d3790731e472505bc39ca53a9e167c1dce159941a47d6245d66374c"
    )
    density = _random_hermitian(4, radial.nk, seed=61)
    np.testing.assert_array_equal(harmonic(density), axial(density))
    assert harmonic.mode == 0
    assert harmonic.mode_mod_nphi == 0
    assert harmonic.is_self_conjugate_mode
    assert co_rotating_mode_alias(-1, nphi) == nphi - 1
    assert co_rotating_mode_alias(1 + nphi, nphi) == 1


def test_multi_harmonic_builder_and_polar_action_match_full_2d() -> None:
    radial = _radial_bundle()
    nphi = 4
    epsilon_r = 15.0
    requested_modes = (0, 1, 2, -1)
    tensors = precompute_axial_harmonic_exchange_tensors_mev_nm2(
        radial,
        modes=requested_modes,
        nphi=nphi,
        epsilon_r=epsilon_r,
    )
    harmonics = []
    for mode in requested_modes:
        single = precompute_axial_harmonic_exchange_tensor_mev_nm2(
            radial,
            mode=mode,
            nphi=nphi,
            epsilon_r=epsilon_r,
        )
        np.testing.assert_array_equal(tensors[mode], single)
        direct = CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
            radial,
            mode=mode,
            nphi=nphi,
            epsilon_r=epsilon_r,
        )
        cached = CoRotatingHarmonicFockOperator.from_precomputed_uniform_dielectric(
            radial,
            tensors[mode],
            mode=mode,
            nphi=nphi,
            epsilon_r=epsilon_r,
        )
        assert cached.fingerprint() == direct.fingerprint()
        harmonics.append(cached)

    polar_bundle = expand_axial_radial_bundle(radial, nphi=nphi)
    polar = PolarHarmonicProjectedFockOperator.from_harmonic_operators(
        radial,
        polar_bundle,
        tuple(harmonics),
        target_nphi=nphi,
    )
    green = uniform_dielectric_green_on_mesh_mev_nm2(
        polar_bundle.k_cart_nm_inv,
        polar_bundle.weights_nm2,
        polar_bundle.z_nm,
        epsilon_r=epsilon_r,
    )
    full = ProjectedFockOperator.from_bundle(
        polar_bundle,
        green,
        self_cell_description="full-2D all-harmonic polar oracle",
        precompute_exchange_tensor=True,
    )
    density = _random_hermitian(4, polar_bundle.nk, seed=107)
    np.testing.assert_allclose(
        polar(density), full(density), rtol=4e-11, atol=4e-11
    )
    np.testing.assert_allclose(
        polar.energy_density_mev_nm2(density),
        full.energy_density_mev_nm2(density),
        rtol=4e-11,
        atol=4e-11,
    )
    normal = solve_reference_subtracted_matrix_ei(
        polar_bundle,
        polar,
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.2,
            precision=1e-10,
            max_iter=5,
            search_mode="normal_reference",
            reference_policy="normal_ordered_exchange_only",
            normal_ordering_reference_policy="noninteracting_fermi_state",
        ),
    )
    assert normal.run.converged


def test_polar_target_nyquist_uses_both_finer_interaction_modes() -> None:
    radial = _radial_bundle()
    target_nphi = 4
    interaction_nphi = 8
    modes = (0, 1, -1, 2, -2)
    tensors = precompute_axial_harmonic_exchange_tensors_mev_nm2(
        radial,
        modes=modes,
        nphi=interaction_nphi,
        epsilon_r=15.0,
    )
    harmonics = tuple(
        CoRotatingHarmonicFockOperator.from_precomputed_uniform_dielectric(
            radial,
            tensors[mode],
            mode=mode,
            nphi=interaction_nphi,
            epsilon_r=15.0,
        )
        for mode in modes
    )
    target = expand_axial_radial_bundle(radial, nphi=target_nphi)
    with pytest.raises(ValueError, match="Nyquist requires"):
        PolarHarmonicProjectedFockOperator.from_harmonic_operators(
            radial,
            target,
            harmonics[:-1],
            target_nphi=target_nphi,
        )
    polar = PolarHarmonicProjectedFockOperator.from_harmonic_operators(
        radial,
        target,
        harmonics,
        target_nphi=target_nphi,
    )
    assert polar.target_mode_aliases == (0, 1, 2, 3)
    assert polar.operator_target_mode_aliases.count(2) == 2

    nyquist_matrix = _random_hermitian(4, radial.nk, seed=109)
    density_modes = np.zeros(
        (4, 4, radial.nk, target_nphi), dtype=np.complex128
    )
    density_modes[:, :, :, target_nphi // 2] = nyquist_matrix
    density = polar.reconstruct_modes(density_modes)
    sigma = polar(density)
    np.testing.assert_allclose(
        sigma, np.swapaxes(sigma.conj(), 0, 1), rtol=2e-11, atol=2e-11
    )
    sigma_modes = polar.co_rotating_modes(sigma)
    plus = next(operator for operator in harmonics if operator.mode == 2)
    minus = next(operator for operator in harmonics if operator.mode == -2)
    expected_nyquist = 0.5 * (
        plus(nyquist_matrix) + minus(nyquist_matrix)
    )
    np.testing.assert_allclose(
        sigma_modes[:, :, :, target_nphi // 2],
        expected_nyquist,
        rtol=3e-11,
        atol=3e-11,
    )


def test_co_rotating_physical_pair_matches_full_2d_at_every_angle_and_fft(
    harmonic_pair_oracle: dict[str, object],
) -> None:
    radial = harmonic_pair_oracle["radial"]
    nphi = harmonic_pair_oracle["nphi"]
    mode = harmonic_pair_oracle["mode"]
    plus = harmonic_pair_oracle["plus"]
    minus = harmonic_pair_oracle["minus"]
    full = harmonic_pair_oracle["full"]
    mode_matrix = harmonic_pair_oracle["mode_matrix"]
    density_full = harmonic_pair_oracle["density_full"]

    sigma_plus = plus(mode_matrix)
    sigma_minus = minus(_dagger_radial(mode_matrix))
    sigma_full = full(density_full)
    sigma_co_rotating = _co_rotating_from_lab(
        sigma_full, nr=radial.nk, nphi=nphi
    )
    angles = 2.0 * np.pi * np.arange(nphi, dtype=float) / float(nphi)
    for iangle, angle in enumerate(angles):
        expected = (
            np.exp(1j * mode * angle) * sigma_plus
            + np.exp(-1j * mode * angle) * sigma_minus
        )
        np.testing.assert_allclose(
            sigma_co_rotating[iangle],
            expected,
            rtol=3e-11,
            atol=3e-11,
        )

    sigma_modes = np.fft.fft(sigma_co_rotating, axis=0) / float(nphi)
    np.testing.assert_allclose(
        sigma_modes[plus.mode_mod_nphi],
        sigma_plus,
        rtol=3e-11,
        atol=3e-11,
    )
    np.testing.assert_allclose(
        sigma_modes[minus.mode_mod_nphi],
        sigma_minus,
        rtol=3e-11,
        atol=3e-11,
    )


def test_co_rotating_physical_pair_full_reduced_fock_energy_parity(
    harmonic_pair_oracle: dict[str, object],
) -> None:
    radial = harmonic_pair_oracle["radial"]
    plus = harmonic_pair_oracle["plus"]
    minus = harmonic_pair_oracle["minus"]
    full = harmonic_pair_oracle["full"]
    mode_matrix = harmonic_pair_oracle["mode_matrix"]
    density_full = harmonic_pair_oracle["density_full"]

    mode_matrix_dagger = _dagger_radial(mode_matrix)
    sigma_plus = plus(mode_matrix)
    sigma_minus = minus(mode_matrix_dagger)
    reduced_energy = 0.5 * (
        np.einsum(
            "i,abi,bai->",
            radial.weights_nm2,
            sigma_plus,
            mode_matrix_dagger,
            optimize=True,
        )
        + np.einsum(
            "i,abi,bai->",
            radial.weights_nm2,
            sigma_minus,
            mode_matrix,
            optimize=True,
        )
    )
    assert abs(reduced_energy.imag) < 2e-11
    np.testing.assert_allclose(
        reduced_energy.real,
        full.energy_density_mev_nm2(density_full),
        rtol=3e-11,
        atol=2e-11,
    )


def test_co_rotating_mode_pair_hermiticity_and_weighted_self_adjointness(
    harmonic_pair_oracle: dict[str, object],
) -> None:
    radial = harmonic_pair_oracle["radial"]
    plus = harmonic_pair_oracle["plus"]
    minus = harmonic_pair_oracle["minus"]
    mode_matrix = harmonic_pair_oracle["mode_matrix"]

    np.testing.assert_allclose(
        plus.exchange_tensor_mev_nm2,
        minus.exchange_tensor_mev_nm2.conj().transpose(0, 1, 3, 2, 5, 4),
        rtol=2e-11,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        minus(_dagger_radial(mode_matrix)),
        _dagger_radial(plus(mode_matrix)),
        rtol=2e-11,
        atol=2e-11,
    )

    left_mode = _random_complex(4, radial.nk, seed=67)
    right_mode = _random_complex(4, radial.nk, seed=71)
    lhs = np.einsum(
        "i,abi,abi->",
        radial.weights_nm2,
        left_mode.conj(),
        plus(right_mode),
        optimize=True,
    )
    rhs = np.einsum(
        "i,abi,abi->",
        radial.weights_nm2,
        plus(left_mode).conj(),
        right_mode,
        optimize=True,
    )
    np.testing.assert_allclose(lhs, rhs, rtol=2e-11, atol=2e-11)


def test_co_rotating_nyquist_mode_matches_full_2d_action_and_energy() -> None:
    radial = _radial_bundle()
    nphi = 4
    mode = nphi // 2
    epsilon_r = 15.0
    harmonic = CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
        radial, mode=mode, nphi=nphi, epsilon_r=epsilon_r
    )
    alias = CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
        radial, mode=-mode, nphi=nphi, epsilon_r=epsilon_r
    )
    assert harmonic.is_self_conjugate_mode
    assert alias.mode_mod_nphi == harmonic.mode_mod_nphi
    assert alias.fingerprint() == harmonic.fingerprint()

    full_bundle = expand_axial_radial_bundle(radial, nphi=nphi)
    green = uniform_dielectric_green_on_mesh_mev_nm2(
        full_bundle.k_cart_nm_inv,
        full_bundle.weights_nm2,
        full_bundle.z_nm,
        epsilon_r=epsilon_r,
    )
    full = ProjectedFockOperator.from_bundle(
        full_bundle,
        green,
        self_cell_description="full-2D Nyquist harmonic oracle",
        precompute_exchange_tensor=True,
    )
    mode_matrix = _random_hermitian(4, radial.nk, seed=73)
    angles = 2.0 * np.pi * np.arange(nphi, dtype=float) / float(nphi)
    density_full = np.empty((4, 4, radial.nk * nphi), dtype=np.complex128)
    for ir in range(radial.nk):
        for iangle, angle in enumerate(angles):
            active_u = _active_rotation(float(angle))
            co_rotating = np.exp(1j * mode * angle) * mode_matrix[:, :, ir]
            np.testing.assert_allclose(co_rotating, co_rotating.conj().T, atol=1e-13)
            density_full[:, :, ir * nphi + iangle] = (
                active_u @ co_rotating @ active_u.conj().T
            )

    sigma_mode = harmonic(mode_matrix)
    np.testing.assert_allclose(
        sigma_mode, _dagger_radial(sigma_mode), rtol=2e-11, atol=2e-11
    )
    sigma_full = full(density_full)
    sigma_co_rotating = _co_rotating_from_lab(
        sigma_full, nr=radial.nk, nphi=nphi
    )
    sigma_modes = np.fft.fft(sigma_co_rotating, axis=0) / float(nphi)
    np.testing.assert_allclose(
        sigma_modes[mode], sigma_mode, rtol=3e-11, atol=3e-11
    )
    for iangle, angle in enumerate(angles):
        np.testing.assert_allclose(
            sigma_co_rotating[iangle],
            np.exp(1j * mode * angle) * sigma_mode,
            rtol=3e-11,
            atol=3e-11,
        )
    reduced_energy = 0.5 * np.einsum(
        "i,abi,bai->",
        radial.weights_nm2,
        sigma_mode,
        mode_matrix,
        optimize=True,
    ).real
    np.testing.assert_allclose(
        reduced_energy,
        full.energy_density_mev_nm2(density_full),
        rtol=3e-11,
        atol=2e-11,
    )


def test_co_rotating_negative_mode_is_stable_at_production_nphi() -> None:
    radial = _radial_bundle()
    nphi = 1200
    plus = CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
        radial, mode=1, nphi=nphi, epsilon_r=15.0
    )
    minus = CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
        radial, mode=-1, nphi=nphi, epsilon_r=15.0
    )
    np.testing.assert_allclose(
        plus.exchange_tensor_mev_nm2,
        minus.exchange_tensor_mev_nm2.conj().transpose(0, 1, 3, 2, 5, 4),
        rtol=2e-10,
        atol=2e-10,
    )
    assert np.all(np.isfinite(minus.exchange_tensor_mev_nm2))


def test_co_rotating_harmonic_validation_aliases_and_attestation(
    harmonic_pair_oracle: dict[str, object],
) -> None:
    radial = harmonic_pair_oracle["radial"]
    nphi = harmonic_pair_oracle["nphi"]
    plus = harmonic_pair_oracle["plus"]
    minus = harmonic_pair_oracle["minus"]
    mode_matrix = harmonic_pair_oracle["mode_matrix"]

    assert plus.mode == 1
    assert plus.mode_mod_nphi == 1
    assert minus.mode == -1
    assert minus.mode_mod_nphi == nphi - 1
    assert plus.conjugate_mode_mod_nphi == minus.mode_mod_nphi
    assert not plus.is_self_conjugate_mode
    alias = CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
        radial, mode=1 + nphi, nphi=nphi, epsilon_r=15.0
    )
    assert alias.mode == 1 + nphi
    assert alias.mode_mod_nphi == plus.mode_mod_nphi
    assert alias.fingerprint() == plus.fingerprint()
    np.testing.assert_array_equal(
        alias.exchange_tensor_mev_nm2,
        plus.exchange_tensor_mev_nm2,
    )

    # A finite non-Hermitian D_m is valid; only a physical +/-m pair is
    # constrained by Hermiticity.
    assert np.max(np.abs(mode_matrix - _dagger_radial(mode_matrix))) > 0.1
    assert np.all(np.isfinite(plus(mode_matrix)))
    with pytest.raises(ValueError, match="integer"):
        CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
            radial, mode=1.0, nphi=nphi, epsilon_r=15.0
        )
    with pytest.raises(ValueError, match="even integer"):
        CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
            radial, mode=1, nphi=3, epsilon_r=15.0
        )
    invalid_density = mode_matrix.copy()
    invalid_density[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        plus(invalid_density)

    invalid_provenance = replace(
        radial,
        provenance={**radial.provenance, "radial_only": False},
    )
    with pytest.raises(ValueError, match="radial-only"):
        CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
            invalid_provenance,
            mode=1,
            nphi=nphi,
            epsilon_r=15.0,
        )
    with pytest.raises(ValueError, match="mode alias"):
        replace(plus, mode_mod_nphi=2)
    with pytest.raises(ValueError, match="finite"):
        replace(plus, epsilon_r=np.nan)
    with pytest.raises(ValueError, match="builder attestation"):
        replace(
            plus,
            exchange_tensor_mev_nm2=2.0 * plus.exchange_tensor_mev_nm2,
        )

    assert not plus.exchange_tensor_mev_nm2.flags.writeable
    assert not plus.k_weights_nm2.flags.writeable
    plus.validate_against_bundle(radial)
    with pytest.raises(ValueError, match="fingerprints differ"):
        plus.validate_against_bundle(_detuned_bundle(radial, 0.2))
