from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import ellipkm1

from ei_mf.coulomb import (
    RadialScreeningSpec,
    angular_averaged_k_dependent_density_kernels,
    angular_averaged_kernel,
    angular_averaged_kernel_cell_integrated,
    angular_averaged_screened_kernel_cell_integrated,
)
from ei_mf.grids import radial_grid
from ei_mf.units import COULOMB_MEV_NM
from ei_mf.params import EIParams
from ei_mf.wavefunction_form_factor import KaneLayerProfiles, KaneMomentumLayerProfiles


def test_interlayer_kernel_finite_positive_symmetric_and_parameter_trends() -> None:
    k, _w, _edges = radial_grid(0.04, 12)
    params = EIParams(nk=12, nphi=32, k_max_nm_inv=0.04, max_iter=5)
    K = angular_averaged_kernel(k, params, kind="interlayer", block_rows=4)
    assert np.all(np.isfinite(K))
    assert np.all(K >= 0.0)
    assert np.max(np.abs(K - K.T)) < 1e-10 * np.max(K)

    K_large_d = angular_averaged_kernel(k, params.updated(d_eh_nm=20.0), kind="interlayer", block_rows=4)
    K_large_eps = angular_averaged_kernel(k, params.updated(eps_r=30.0), kind="interlayer", block_rows=4)
    assert np.all(K_large_d <= K + 1e-12)
    assert np.all(K_large_eps <= K + 1e-12)


def test_absolute_fourier_prefactor_and_radial_measure_oracles() -> None:
    """Guard the 2D Fourier and integration conventions against 2*pi drift."""
    k = np.array([0.0, 0.01, 0.02, 0.03])
    params = EIParams(nk=4, nphi=64, k_max_nm_inv=0.04, eps_r=15.0, d_eh_nm=10.0)
    K = angular_averaged_kernel(k, params, kind="interlayer", block_rows=4)
    q = k[2]
    expected = 2.0 * np.pi * COULOMB_MEV_NM / params.eps_r / q * np.exp(-q * params.d_eh_nm)
    assert np.isclose(K[0, 2], expected, rtol=2e-14, atol=0.0)

    _grid, weights, _edges = radial_grid(0.12, 37)
    # int_{|k|<kmax} d^2k/(2*pi)^2 = kmax^2/(4*pi)
    assert np.isclose(np.sum(weights), 0.12**2 / (4.0 * np.pi), rtol=2e-15, atol=0.0)


def test_cell_integrated_kernel_removes_qfloor_and_nphi_from_singular_part() -> None:
    k, weights, _edges = radial_grid(0.04, 16)
    params = EIParams(nk=16, nphi=32, k_max_nm_inv=0.04, eps_r=15.0, d_eh_nm=0.0)
    kernel32 = angular_averaged_kernel_cell_integrated(
        k, weights, params, kind="intralayer"
    )
    kernel256 = angular_averaged_kernel_cell_integrated(
        k,
        weights,
        params.updated(nphi=256, q_floor_nm_inv=1e-2),
        kind="intralayer",
    )
    assert np.array_equal(kernel32, kernel256)
    assert np.all(np.isfinite(kernel32))
    assert np.all(kernel32 > 0.0)
    assert np.array_equal(kernel32, kernel32.T)

    i, j = 4, 11
    denom = k[i] + k[j]
    complementary = ((k[i] - k[j]) / denom) ** 2
    expected = (
        COULOMB_MEV_NM
        / params.eps_r
        * 4.0
        * ellipkm1(complementary)
        / denom
    )
    assert np.isclose(kernel32[i, j], expected, rtol=2e-14, atol=0.0)


def test_cell_integrated_interlayer_regular_correction_converges() -> None:
    k, weights, _edges = radial_grid(0.05, 18)
    params = EIParams(nk=18, nphi=64, k_max_nm_inv=0.05, eps_r=15.0, d_eh_nm=10.0)
    intralayer = angular_averaged_kernel_cell_integrated(
        k, weights, params, kind="intralayer"
    )
    inter64 = angular_averaged_kernel_cell_integrated(
        k, weights, params, kind="interlayer", nphi_regular=64
    )
    inter128 = angular_averaged_kernel_cell_integrated(
        k, weights, params, kind="interlayer", nphi_regular=128
    )
    assert np.all(inter64 <= intralayer)
    assert np.all(inter64 > 0.0)
    assert np.max(np.abs(inter64 - inter64.T)) == 0.0
    assert np.max(np.abs(inter64 - inter128)) < 2e-7 * np.max(inter128)

    coincident = angular_averaged_kernel_cell_integrated(
        k, weights, params.updated(d_eh_nm=0.0), kind="interlayer"
    )
    assert np.array_equal(coincident, intralayer)


def test_cell_integrated_diagonal_matches_adaptive_annular_oracle() -> None:
    k, weights, edges = radial_grid(0.05, 10)
    index = 4
    k0 = float(k[index])
    lo = float(edges[index])
    hi = float(edges[index + 1])
    weight = float(weights[index])
    params = EIParams(nk=10, nphi=96, k_max_nm_inv=0.05, eps_r=15.0, d_eh_nm=10.0)
    prefactor = COULOMB_MEV_NM / params.eps_r

    def singular_angular(kp: float) -> float:
        denom = k0 + kp
        complementary = ((k0 - kp) / denom) ** 2
        return float(4.0 * ellipkm1(complementary) / denom)

    xphi, wphi = leggauss(512)
    phi = np.pi * (xphi + 1.0)
    angular_weights = np.pi * wphi

    def regular_angular(kp: float) -> float:
        q = np.sqrt(
            np.maximum(k0 * k0 + kp * kp - 2.0 * k0 * kp * np.cos(phi), 0.0)
        )
        regular = np.empty_like(q)
        mask = q > 1e-13
        regular[mask] = np.expm1(-params.d_eh_nm * q[mask]) / q[mask]
        regular[~mask] = -params.d_eh_nm
        return float(np.dot(angular_weights, regular))

    def radial_cell(angular) -> float:
        value = 0.0
        for left, right in ((lo, k0), (k0, hi)):
            value += float(
                quad(
                    lambda kp: kp * angular(kp) / (2.0 * np.pi),
                    left,
                    right,
                    epsabs=1e-9,
                    epsrel=1e-10,
                    limit=300,
                )[0]
            )
        return value / weight

    expected_intra = prefactor * radial_cell(singular_angular)
    expected_inter = expected_intra + prefactor * radial_cell(regular_angular)
    intra = angular_averaged_kernel_cell_integrated(
        k, weights, params, kind="intralayer", diagonal_order=256
    )
    inter = angular_averaged_kernel_cell_integrated(
        k,
        weights,
        params,
        kind="interlayer",
        nphi_regular=256,
        diagonal_order=256,
        regular_cell_order=128,
    )
    assert np.isclose(intra[index, index], expected_intra, rtol=2.0e-6, atol=1e-8)
    assert np.isclose(inter[index, index], expected_inter, rtol=2.5e-6, atol=1e-8)

    refined = angular_averaged_kernel_cell_integrated(
        k,
        weights,
        params,
        kind="interlayer",
        nphi_regular=384,
        diagonal_order=384,
        regular_cell_order=192,
    )
    assert np.isclose(refined[index, index], expected_inter, rtol=1.2e-6, atol=1e-8)
    assert np.isclose(inter[index, index], refined[index, index], rtol=1.5e-6, atol=1e-8)


def test_delta_profile_reproduces_point_layer_kernel() -> None:
    k, _w, _edges = radial_grid(0.04, 12)
    params = EIParams(nk=12, nphi=32, k_max_nm_inv=0.04, d_eh_nm=10.0)
    z = np.arange(-2.0, 13.0)
    rho_e = np.zeros_like(z)
    rho_h = np.zeros_like(z)
    rho_e[np.where(z == 0.0)[0][0]] = 1.0
    rho_h[np.where(z == 10.0)[0][0]] = 1.0
    profiles = KaneLayerProfiles(z, rho_e, rho_h)
    point = angular_averaged_kernel(k, params, kind="interlayer", block_rows=4)
    projected = angular_averaged_kernel(
        k, params, kind="interlayer", block_rows=4, form_factor_profiles=profiles
    )
    assert np.allclose(projected, point, rtol=2e-7, atol=1e-10)

    point_cell = angular_averaged_kernel_cell_integrated(
        k, _w, params, kind="interlayer", nphi_regular=64
    )
    projected_cell = angular_averaged_kernel_cell_integrated(
        k,
        _w,
        params,
        kind="interlayer",
        nphi_regular=64,
        form_factor_profiles=profiles,
    )
    assert np.allclose(projected_cell, point_cell, rtol=3e-7, atol=1e-8)

    projected_intra = angular_averaged_kernel_cell_integrated(
        k,
        _w,
        params,
        kind="intralayer",
        nphi_regular=64,
        form_factor_profiles=profiles,
    )
    point_intra = angular_averaged_kernel_cell_integrated(
        k, _w, params, kind="intralayer", nphi_regular=64
    )
    assert np.allclose(projected_intra, point_intra, rtol=3e-7, atol=1e-8)


def test_constant_momentum_profiles_reduce_to_fixed_profile_kernels() -> None:
    source_k = np.linspace(0.0, 0.04, 21)
    z = np.linspace(-4.0, 4.0, 33)
    dz = z[1] - z[0]
    rho_e = np.exp(-0.5 * ((z + 0.8) / 0.7) ** 2)
    rho_h = np.exp(-0.5 * ((z - 0.9) / 0.5) ** 2)
    rho_e /= np.sum(rho_e) * dz
    rho_h /= np.sum(rho_h) * dz
    momentum_profiles = KaneMomentumLayerProfiles(
        source_k,
        z,
        np.repeat(rho_e[None, :], source_k.size, axis=0),
        np.repeat(rho_h[None, :], source_k.size, axis=0),
        np.tile(np.array([0.8, 0.7, 0.1, 0.0]), (source_k.size, 1)),
        np.ones(source_k.size - 1),
        np.ones(source_k.size - 1),
        np.ones(source_k.size - 1),
    )
    k, weights, _edges = radial_grid(0.04, 12)
    params = EIParams(nk=12, nphi=64, k_max_nm_inv=0.04, d_eh_nm=10.0)
    kernels = angular_averaged_k_dependent_density_kernels(
        k,
        weights,
        params,
        momentum_profiles,
        nphi_regular=64,
        q_samples=513,
        diagonal_order=256,
        regular_cell_order=64,
    )
    fixed = KaneLayerProfiles(z, rho_e, rho_h)
    expected_eh = angular_averaged_kernel_cell_integrated(
        k,
        weights,
        params,
        kind="interlayer",
        nphi_regular=64,
        form_factor_profiles=fixed,
    )
    expected_ee = angular_averaged_kernel_cell_integrated(
        k,
        weights,
        params,
        kind="intralayer",
        nphi_regular=64,
        form_factor_profiles=fixed,
    )
    hole_as_electron = KaneLayerProfiles(z, rho_h, rho_h)
    expected_hh = angular_averaged_kernel_cell_integrated(
        k,
        weights,
        params,
        kind="intralayer",
        nphi_regular=64,
        form_factor_profiles=hole_as_electron,
    )
    assert np.allclose(kernels.K_eh_mev_nm2, expected_eh, rtol=2e-5, atol=1e-7)
    assert np.allclose(kernels.K_ee_mev_nm2, expected_ee, rtol=2e-5, atol=1e-7)
    assert np.allclose(kernels.K_hh_mev_nm2, expected_hh, rtol=2e-5, atol=1e-7)
    assert np.allclose(
        kernels.K_aa_balanced_mev_nm2,
        0.5 * (expected_ee + expected_hh),
        rtol=2e-5,
        atol=1e-7,
    )


def test_k_dependent_density_kernels_are_symmetric_and_change_with_profiles() -> None:
    source_k = np.linspace(0.0, 0.04, 17)
    z = np.linspace(-4.0, 4.0, 25)
    dz = z[1] - z[0]
    electron = []
    hole = []
    for fraction in source_k / source_k[-1]:
        rho_e = np.exp(-0.5 * ((z + 1.0 - 0.6 * fraction) / 0.7) ** 2)
        rho_h = np.exp(-0.5 * ((z - 1.0 + 0.4 * fraction) / 0.6) ** 2)
        electron.append(rho_e / (np.sum(rho_e) * dz))
        hole.append(rho_h / (np.sum(rho_h) * dz))
    profiles = KaneMomentumLayerProfiles(
        source_k,
        z,
        np.asarray(electron),
        np.asarray(hole),
        np.tile(np.array([0.8, 0.7, 0.1, 0.0]), (source_k.size, 1)),
        np.ones(source_k.size - 1),
        np.ones(source_k.size - 1),
        np.ones(source_k.size - 1),
    )
    k, weights, _edges = radial_grid(0.04, 10)
    params = EIParams(nk=10, nphi=64, k_max_nm_inv=0.04)
    dynamic = angular_averaged_k_dependent_density_kernels(
        k,
        weights,
        params,
        profiles,
        nphi_regular=64,
        q_samples=513,
        regular_cell_order=48,
    )
    frozen = angular_averaged_k_dependent_density_kernels(
        k,
        weights,
        params,
        profiles,
        frozen_k_nm_inv=0.02,
        nphi_regular=64,
        q_samples=513,
        regular_cell_order=48,
    )
    for kernel in (
        dynamic.K_eh_mev_nm2,
        dynamic.K_ee_mev_nm2,
        dynamic.K_hh_mev_nm2,
        dynamic.K_aa_balanced_mev_nm2,
    ):
        assert np.array_equal(kernel, kernel.T)
        assert np.all(kernel > 0.0)
    assert np.max(np.abs(dynamic.K_eh_mev_nm2 - frozen.K_eh_mev_nm2)) > 1e-3


def test_screening_specs_have_correct_q_zero_limits() -> None:
    q = np.array([0.0, 0.03, 0.1])
    single = RadialScreeningSpec("single_metal_gate", gate_distance_nm=12.0)
    dual = RadialScreeningSpec(
        "symmetric_dual_metal_gate", gate_distance_nm=12.0
    )
    tf = RadialScreeningSpec("thomas_fermi", q_tf_nm_inv=0.025)
    single_values = single.inverse_momentum_nm(q)
    dual_values = dual.inverse_momentum_nm(q)
    tf_values = tf.inverse_momentum_nm(q)
    assert np.isclose(single_values[0], 24.0)
    assert np.isclose(dual_values[0], 12.0)
    assert np.isclose(tf_values[0], 40.0)
    assert np.allclose(
        single_values[1:],
        (1.0 - np.exp(-24.0 * q[1:])) / q[1:],
    )
    assert np.allclose(dual_values[1:], np.tanh(12.0 * q[1:]) / q[1:])
    assert np.allclose(tf_values[1:], 1.0 / (q[1:] + 0.025))


def test_screened_cell_kernels_are_symmetric_positive_and_follow_screening_strength() -> None:
    k, weights, _edges = radial_grid(0.08, 18)
    params = EIParams(nk=18, nphi=96, k_max_nm_inv=0.08, d_eh_nm=10.0)

    single_near = angular_averaged_screened_kernel_cell_integrated(
        k,
        weights,
        params,
        RadialScreeningSpec("single_metal_gate", gate_distance_nm=5.0),
        kind="intralayer",
        nphi=96,
        radial_cell_order=64,
    )
    single_far = angular_averaged_screened_kernel_cell_integrated(
        k,
        weights,
        params,
        RadialScreeningSpec("single_metal_gate", gate_distance_nm=50.0),
        kind="intralayer",
        nphi=96,
        radial_cell_order=64,
    )
    tf_weak = angular_averaged_screened_kernel_cell_integrated(
        k,
        weights,
        params,
        RadialScreeningSpec("thomas_fermi", q_tf_nm_inv=0.005),
        kind="intralayer",
        nphi=96,
        radial_cell_order=64,
    )
    tf_strong = angular_averaged_screened_kernel_cell_integrated(
        k,
        weights,
        params,
        RadialScreeningSpec("thomas_fermi", q_tf_nm_inv=0.05),
        kind="intralayer",
        nphi=96,
        radial_cell_order=64,
    )
    interlayer = angular_averaged_screened_kernel_cell_integrated(
        k,
        weights,
        params,
        RadialScreeningSpec("single_metal_gate", gate_distance_nm=50.0),
        kind="interlayer",
        nphi=96,
        radial_cell_order=64,
    )
    for kernel in (single_near, single_far, tf_weak, tf_strong, interlayer):
        assert np.array_equal(kernel, kernel.T)
        assert np.all(np.isfinite(kernel))
        assert np.all(kernel > 0.0)
    assert np.all(single_near < single_far)
    assert np.all(tf_strong < tf_weak)
    assert np.all(interlayer <= single_far)


def test_screened_diagonal_annular_cell_quadrature_refines() -> None:
    k, weights, _edges = radial_grid(0.1, 20)
    params = EIParams(nk=20, nphi=128, k_max_nm_inv=0.1)
    screening = RadialScreeningSpec(
        "symmetric_dual_metal_gate", gate_distance_nm=20.0
    )
    coarse = angular_averaged_screened_kernel_cell_integrated(
        k,
        weights,
        params,
        screening,
        kind="intralayer",
        nphi=128,
        radial_cell_order=48,
    )
    fine = angular_averaged_screened_kernel_cell_integrated(
        k,
        weights,
        params,
        screening,
        kind="intralayer",
        nphi=192,
        radial_cell_order=96,
    )
    refined = angular_averaged_screened_kernel_cell_integrated(
        k,
        weights,
        params,
        screening,
        kind="intralayer",
        nphi=256,
        radial_cell_order=160,
    )
    scale = np.max(np.diag(refined))
    coarse_error = np.max(np.abs(np.diag(coarse) - np.diag(refined))) / scale
    fine_error = np.max(np.abs(np.diag(fine) - np.diag(refined))) / scale
    assert coarse_error < 1e-12
    assert fine_error < 1e-12


def test_screened_kernel_matches_independent_absolute_offdiagonal_and_cell_oracles() -> None:
    k, weights, _edges = radial_grid(0.06, 12)
    params = EIParams(nk=12, nphi=128, k_max_nm_inv=0.06, d_eh_nm=10.0)
    screening = RadialScreeningSpec("single_metal_gate", gate_distance_nm=18.0)
    kernel = angular_averaged_screened_kernel_cell_integrated(
        k,
        weights,
        params,
        screening,
        kind="interlayer",
        nphi=160,
        radial_cell_order=96,
    )
    prefactor = COULOMB_MEV_NM / params.eps_r

    x_oracle, w_oracle = leggauss(1024)
    phi_oracle = np.pi * (x_oracle + 1.0)
    angular_oracle_weights = np.pi * w_oracle

    def angular(k0: float, kp: float) -> float:
        q = np.sqrt(
            np.maximum(
                k0 * k0 + kp * kp - 2.0 * k0 * kp * np.cos(phi_oracle),
                0.0,
            )
        )
        inverse_q = -np.expm1(-2.0 * 18.0 * q) / q
        values = inverse_q * np.exp(-params.d_eh_nm * q)
        return float(np.dot(angular_oracle_weights, values))

    i, j = 2, 8
    expected_offdiagonal = prefactor * angular(float(k[i]), float(k[j]))
    assert np.isclose(kernel[i, j], expected_offdiagonal, rtol=2e-10, atol=1e-8)

    area = np.concatenate(([0.0], np.cumsum(weights)))
    edges = np.sqrt(4.0 * np.pi * area)
    index = 5
    k0 = float(k[index])
    lo, hi = float(edges[index]), float(edges[index + 1])

    x_radial, w_radial = leggauss(256)
    cell_integral = 0.0
    for left, right in ((lo, k0), (k0, hi)):
        kp_nodes = 0.5 * (right - left) * x_radial + 0.5 * (left + right)
        kp_weights = 0.5 * (right - left) * w_radial
        values = np.asarray(
            [kp / (2.0 * np.pi) * angular(k0, float(kp)) for kp in kp_nodes]
        )
        cell_integral += float(np.dot(kp_weights, values))
    expected_diagonal = prefactor / weights[index] * cell_integral
    assert np.isclose(kernel[index, index], expected_diagonal, rtol=2e-8, atol=1e-7)


def test_screened_kernel_rejects_nonfinite_and_underresolved_contracts() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        EIParams(eps_r=np.inf).validate()
    with pytest.raises(ValueError, match="finite and non-negative"):
        EIParams(d_eh_nm=np.inf).validate()
    k, weights, _edges = radial_grid(0.1, 12)
    params = EIParams(nk=12, nphi=32, k_max_nm_inv=0.1)
    with pytest.raises(ValueError, match="under-resolved"):
        angular_averaged_screened_kernel_cell_integrated(
            k,
            weights,
            params,
            RadialScreeningSpec("thomas_fermi", q_tf_nm_inv=1e-10),
            nphi=32,
            radial_cell_order=16,
        )
    with pytest.raises(ValueError, match="under-resolved"):
        angular_averaged_screened_kernel_cell_integrated(
            k,
            weights,
            params,
            RadialScreeningSpec("single_metal_gate", gate_distance_nm=1e9),
            nphi=32,
            radial_cell_order=16,
        )
