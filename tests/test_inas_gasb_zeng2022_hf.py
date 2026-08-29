from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.inas_gasb.xue2018 import xue2018_standard_parameters
from mean_field.systems.inas_gasb.zeng2022 import Zeng2022Parameters, ZengSlabBasis
from mean_field.systems.inas_gasb.zeng2022_hf import (
    precompute_q0_coulomb_kernel,
    precompute_q0_toeplitz_coulomb_kernel,
    q0_coulomb_kernel_row_with_integrated_cell,
    q0_fock_at_k_from_integrated_cell_row,
    q0_fock_from_precomputed_kernel,
    q0_fock_from_toeplitz_kernel,
    q0_interaction_from_precomputed_kernel,
    q0_interaction_from_toeplitz_kernel,
    rectangular_coulomb_self_cell_average,
    rectangular_coulomb_singular_cell_average,
    uniform_midpoint_kappa_mesh,
    zeng2022_fock_direct,
    zeng2022_hartree_direct,
    zeng2022_interaction_direct,
    zeng2022_interaction_energy_density,
)


def _random_neutral_hermitian_density(
    basis: ZengSlabBasis,
    mesh,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(basis.dimension, basis.dimension, mesh.nk))
    raw = raw + 1j * rng.normal(size=raw.shape)
    density = 0.01 * (raw + np.swapaxes(raw.conj(), 0, 1))
    weights = np.asarray(mesh.weights_ab2)
    total = 0.0
    for slab in basis.slab_indices:
        for band in ("c", "v"):
            for spin in ("up", "down"):
                index = basis.index(band, spin, slab)
                total += float(np.dot(weights, density[index, index, :].real))
    repair_index = basis.index("v", "up", basis.slab_indices[0])
    density[repair_index, repair_index, :] -= total / np.sum(weights)
    return density


def test_rectangular_coulomb_self_cell_is_integrated_not_q_floor() -> None:
    intra, inter = rectangular_coulomb_self_cell_average(
        delta_kx_ab_inv=0.2,
        delta_ky_ab_inv=0.3,
        d_over_ab=0.3,
        quadrature_order=96,
    )
    a, b = 0.1, 0.15
    exact_integral = 4.0 * (a * np.arcsinh(b / a) + b * np.arcsinh(a / b))
    exact_average = 4.0 * np.pi * exact_integral / (0.2 * 0.3)
    assert intra == pytest.approx(exact_average, rel=1e-14)
    assert 0.0 < inter < intra
    _, inter_high = rectangular_coulomb_self_cell_average(
        delta_kx_ab_inv=0.2,
        delta_ky_ab_inv=0.3,
        d_over_ab=0.3,
        quadrature_order=160,
    )
    assert inter == pytest.approx(inter_high, rel=2e-13)


def test_shifted_singular_cell_recovers_center_and_offset_symmetry() -> None:
    centered = rectangular_coulomb_self_cell_average(
        delta_kx_ab_inv=0.2,
        delta_ky_ab_inv=0.3,
        d_over_ab=0.4,
    )
    shifted_center = rectangular_coulomb_singular_cell_average(
        delta_kx_ab_inv=0.2,
        delta_ky_ab_inv=0.3,
        query_offset_x_ab_inv=0.0,
        query_offset_y_ab_inv=0.0,
        d_over_ab=0.4,
    )
    assert shifted_center == pytest.approx(centered, rel=2.0e-14)
    positive = rectangular_coulomb_singular_cell_average(
        delta_kx_ab_inv=0.2,
        delta_ky_ab_inv=0.3,
        query_offset_x_ab_inv=0.07,
        query_offset_y_ab_inv=-0.04,
        d_over_ab=0.4,
    )
    reflected = rectangular_coulomb_singular_cell_average(
        delta_kx_ab_inv=0.2,
        delta_ky_ab_inv=0.3,
        query_offset_x_ab_inv=-0.07,
        query_offset_y_ab_inv=0.04,
        d_over_ab=0.4,
    )
    assert positive == pytest.approx(reflected, rel=2.0e-14)
    positive_high_order = rectangular_coulomb_singular_cell_average(
        delta_kx_ab_inv=0.2,
        delta_ky_ab_inv=0.3,
        query_offset_x_ab_inv=0.07,
        query_offset_y_ab_inv=-0.04,
        d_over_ab=0.4,
        quadrature_order=160,
    )
    assert positive == pytest.approx(positive_high_order, rel=3.0e-13)
    unscreened = rectangular_coulomb_singular_cell_average(
        delta_kx_ab_inv=0.2,
        delta_ky_ab_inv=0.3,
        query_offset_x_ab_inv=0.07,
        query_offset_y_ab_inv=-0.04,
        d_over_ab=0.0,
    )
    assert unscreened[0] == pytest.approx(unscreened[1], rel=2.0e-14)


def test_uniform_hartree_reproduces_eq9_capacitor_splitting() -> None:
    mesh = uniform_midpoint_kappa_mesh(
        kx_bounds_ab_inv=(-0.5, 0.5),
        ky_bounds_ab_inv=(-0.5, 0.5),
        nkx=1,
        nky=1,
    )
    basis = ZengSlabBasis((0,))
    params = xue2018_standard_parameters(eg_ry=0.1, hybridization_ab_ry=0.2)
    density = np.zeros((4, 4, 1), dtype=np.complex128)
    amplitude = 0.3
    for spin in ("up", "down"):
        density[basis.index("c", spin, 0), basis.index("c", spin, 0), 0] = amplitude
        density[basis.index("v", spin, 0), basis.index("v", spin, 0), 0] = -amplitude
    hartree = zeng2022_hartree_direct(density, basis=basis, mesh=mesh, params=params)
    exciton_density = 2.0 * mesh.weights_ab2[0] * amplitude
    expected_c = 4.0 * np.pi * params.d_over_ab * exciton_density
    for spin in ("up", "down"):
        c = basis.index("c", spin, 0)
        v = basis.index("v", spin, 0)
        assert hartree[c, c, 0] == pytest.approx(expected_c)
        assert hartree[v, v, 0] == pytest.approx(-expected_c)
    bad = density.copy()
    bad[basis.index("c", "up", 0), basis.index("c", "up", 0), 0] += 0.01
    with pytest.raises(ValueError, match="requires neutral"):
        zeng2022_hartree_direct(bad, basis=basis, mesh=mesh, params=params)


def test_direct_fock_reference_zero_and_q0_self_action() -> None:
    mesh = uniform_midpoint_kappa_mesh(
        kx_bounds_ab_inv=(-0.4, 0.4),
        ky_bounds_ab_inv=(-0.3, 0.3),
        nkx=1,
        nky=1,
    )
    basis = ZengSlabBasis((0,))
    params = xue2018_standard_parameters(eg_ry=0.1, hybridization_ab_ry=0.2)
    zero = np.zeros((4, 4, 1), dtype=np.complex128)
    assert np.max(np.abs(zeng2022_fock_direct(zero, basis=basis, mesh=mesh, params=params))) == 0.0
    density = zero.copy()
    index = basis.index("c", "up", 0)
    density[index, index, 0] = 0.2
    fock = zeng2022_fock_direct(density, basis=basis, mesh=mesh, params=params)
    self_intra, _ = rectangular_coulomb_self_cell_average(
        delta_kx_ab_inv=0.8,
        delta_ky_ab_inv=0.6,
        d_over_ab=params.d_over_ab,
    )
    assert fock[index, index, 0] == pytest.approx(-mesh.weights_ab2[0] * self_intra * 0.2)


def test_direct_interaction_is_hermitian_and_has_correct_energy_derivative() -> None:
    mesh = uniform_midpoint_kappa_mesh(
        kx_bounds_ab_inv=(-0.6, 0.6),
        ky_bounds_ab_inv=(-0.5, 0.5),
        nkx=3,
        nky=3,
    )
    basis = ZengSlabBasis((0,))
    params = xue2018_standard_parameters(eg_ry=-0.5, hybridization_ab_ry=0.2)
    density = _random_neutral_hermitian_density(basis, mesh, seed=7)
    direction = _random_neutral_hermitian_density(basis, mesh, seed=19)
    sigma = zeng2022_interaction_direct(density, basis=basis, mesh=mesh, params=params)
    assert np.max(np.abs(sigma - np.swapaxes(sigma.conj(), 0, 1))) < 2e-13
    predicted = np.einsum(
        "abk,bak,k->", sigma, direction, mesh.weights_ab2, optimize=True
    ).real
    step = 2.0e-6
    plus = density + step * direction
    minus = density - step * direction
    e_plus = zeng2022_interaction_energy_density(
        plus,
        zeng2022_interaction_direct(plus, basis=basis, mesh=mesh, params=params),
        mesh=mesh,
    )
    e_minus = zeng2022_interaction_energy_density(
        minus,
        zeng2022_interaction_direct(minus, basis=basis, mesh=mesh, params=params),
        mesh=mesh,
    )
    observed = (e_plus - e_minus) / (2.0 * step)
    assert observed == pytest.approx(predicted, rel=2e-8, abs=2e-10)


def test_precomputed_q0_kernel_matches_direct_oracle_exactly() -> None:
    mesh = uniform_midpoint_kappa_mesh(
        kx_bounds_ab_inv=(-0.7, 0.7),
        ky_bounds_ab_inv=(-0.5, 0.5),
        nkx=3,
        nky=3,
    )
    basis = ZengSlabBasis((0,))
    params = xue2018_standard_parameters(eg_ry=-0.5, hybridization_ab_ry=0.2)
    density = _random_neutral_hermitian_density(basis, mesh, seed=47)
    kernel = precompute_q0_coulomb_kernel(mesh, d_over_ab=params.d_over_ab)
    direct_fock = zeng2022_fock_direct(density, basis=basis, mesh=mesh, params=params)
    fast_fock = q0_fock_from_precomputed_kernel(
        density, basis=basis, mesh=mesh, kernel=kernel
    )
    assert np.max(np.abs(fast_fock - direct_fock)) < 2e-15
    direct = zeng2022_interaction_direct(density, basis=basis, mesh=mesh, params=params)
    fast = q0_interaction_from_precomputed_kernel(
        density, basis=basis, mesh=mesh, params=params, kernel=kernel
    )
    assert np.max(np.abs(fast - direct)) < 2e-15
    toeplitz = precompute_q0_toeplitz_coulomb_kernel(
        mesh, d_over_ab=params.d_over_ab
    )
    fft_fock = q0_fock_from_toeplitz_kernel(
        density, basis=basis, mesh=mesh, kernel=toeplitz
    )
    assert np.max(np.abs(fft_fock - direct_fock)) < 2e-14
    fft_action = q0_interaction_from_toeplitz_kernel(
        density, basis=basis, mesh=mesh, params=params, kernel=toeplitz
    )
    assert np.max(np.abs(fft_action - direct)) < 2e-14


def test_single_k_integrated_row_matches_saved_grid_operator_at_mesh_centers() -> None:
    mesh = uniform_midpoint_kappa_mesh(
        kx_bounds_ab_inv=(-0.7, 0.7),
        ky_bounds_ab_inv=(-0.5, 0.5),
        nkx=3,
        nky=3,
    )
    basis = ZengSlabBasis((0,))
    params = xue2018_standard_parameters(eg_ry=-0.5, hybridization_ab_ry=0.2)
    density = _random_neutral_hermitian_density(basis, mesh, seed=73)
    kernel = precompute_q0_coulomb_kernel(mesh, d_over_ab=params.d_over_ab)
    full_fock = q0_fock_from_precomputed_kernel(
        density,
        basis=basis,
        mesh=mesh,
        kernel=kernel,
    )
    for index in (0, 4, 8):
        intra, inter = q0_coulomb_kernel_row_with_integrated_cell(
            mesh.points_ab_inv[index],
            mesh=mesh,
            d_over_ab=params.d_over_ab,
            singular_cell_index=index,
        )
        assert intra == pytest.approx(kernel.intra_ry_ab2[index], abs=2.0e-13)
        assert inter == pytest.approx(kernel.inter_ry_ab2[index], abs=2.0e-13)
        single = q0_fock_at_k_from_integrated_cell_row(
            density,
            basis=basis,
            mesh=mesh,
            intra_row_ry_ab2=intra,
            inter_row_ry_ab2=inter,
        )
        assert single == pytest.approx(full_fock[:, :, index], abs=2.0e-14)


def test_finite_q_slab_action_preserves_hermiticity() -> None:
    mesh = uniform_midpoint_kappa_mesh(
        kx_bounds_ab_inv=(-0.3, 0.3),
        ky_bounds_ab_inv=(-0.2, 0.2),
        nkx=2,
        nky=2,
    )
    basis = ZengSlabBasis((-1, 0, 1))
    params = Zeng2022Parameters(eg_ry=0.1, hybridization_ab_ry=0.3, q_ab_inv=0.6)
    density = _random_neutral_hermitian_density(basis, mesh, seed=31)
    sigma = zeng2022_interaction_direct(density, basis=basis, mesh=mesh, params=params)
    assert np.max(np.abs(sigma - np.swapaxes(sigma.conj(), 0, 1))) < 5e-13
