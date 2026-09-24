from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from ei_mf.zhu1995 import (
    density_from_rs,
    mapped_beta_grid,
    solve_zhu1995_fixed_mu,
    solve_zhu1995_fixed_rs,
    zhu1995_kernels,
)


def test_mapped_beta_grid_radial_measure_oracle() -> None:
    grid = mapped_beta_grid(120)
    for decay in (0.7, 1.0, 2.5):
        numeric = np.dot(grid.weights, np.exp(-decay * grid.k))
        exact = 1.0 / (2.0 * np.pi * decay**2)
        assert np.isclose(numeric, exact, rtol=2e-9, atol=2e-11)


def test_finite_mapped_beta_grid_matches_hard_cutoff_measure() -> None:
    k_max = 1.75
    grid = mapped_beta_grid(160, k_max=k_max)
    assert grid.k[-1] < k_max
    assert np.isclose(np.tan(grid.beta_edges[-1]), k_max, rtol=0.0, atol=2e-15)
    assert np.isclose(
        np.sum(grid.weights),
        k_max**2 / (4.0 * np.pi),
        rtol=2e-14,
        atol=2e-15,
    )
    for decay in (0.7, 1.0, 2.5):
        numeric = np.dot(grid.weights, np.exp(-decay * grid.k))
        exact = (
            1.0 - np.exp(-decay * k_max) * (1.0 + decay * k_max)
        ) / (2.0 * np.pi * decay**2)
        assert np.isclose(numeric, exact, rtol=2e-14, atol=2e-15)


def test_zhu_density_definition_and_spinless_kf() -> None:
    rs = 2.66
    n = density_from_rs(rs)
    kf = 2.0 / rs
    assert np.isclose(n, 1.0 / (np.pi * rs**2))
    assert np.isclose(kf**2 / (4.0 * np.pi), n)


def test_fixed_mu_replays_fixed_density_root_at_its_mu() -> None:
    fixed_density = solve_zhu1995_fixed_rs(
        2.66,
        1.0,
        nbeta=32,
        nphi_regular=96,
        tol=3e-8,
        max_iter=700,
    )
    assert fixed_density.converged
    fixed_mu = solve_zhu1995_fixed_mu(
        fixed_density.mu,
        1.0,
        nbeta=32,
        nphi_regular=96,
        initial=fixed_density,
        tol=3e-8,
        max_iter=700,
    )
    assert fixed_mu.converged
    assert fixed_mu.residual < 3e-8
    assert np.isclose(fixed_mu.density, fixed_density.density, rtol=2e-7)
    assert np.allclose(fixed_mu.Delta, fixed_density.Delta, rtol=2e-7, atol=2e-8)
    assert np.allclose(fixed_mu.xi, fixed_density.xi, rtol=2e-7, atol=2e-8)


def test_intralayer_angular_kernel_matches_direct_integral_off_diagonal() -> None:
    grid = mapped_beta_grid(24)
    Kaa, Kab0 = zhu1995_kernels(grid, 0.0, nphi_regular=80)
    i, j = 6, 13
    ki, kj = grid.k[i], grid.k[j]
    direct = quad(
        lambda phi: 1.0 / np.sqrt(ki**2 + kj**2 - 2.0 * ki * kj * np.cos(phi)),
        0.0,
        2.0 * np.pi,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    assert np.isclose(Kaa[i, j], direct, rtol=2e-12, atol=2e-12)
    assert np.array_equal(Kaa, Kab0)


def test_interlayer_diagonal_regular_correction_is_cell_averaged() -> None:
    grid = mapped_beta_grid(20)
    Kaa, Kab = zhu1995_kernels(grid, 1.0, nphi_regular=160)
    i = 8
    ki = grid.k[i]

    def angular_regular(beta: float) -> float:
        kp = np.tan(beta)
        def integrand(phi: float) -> float:
            q = np.sqrt(max(ki**2 + kp**2 - 2.0 * ki * kp * np.cos(phi), 0.0))
            return -1.0 if q < 1e-12 else np.expm1(-q) / q
        angular = quad(integrand, 0.0, 2.0 * np.pi, epsabs=1e-7, epsrel=1e-7)[0]
        return kp / np.cos(beta) ** 2 * angular / (2.0 * np.pi)

    direct = quad(
        angular_regular,
        grid.beta_edges[i],
        grid.beta_edges[i + 1],
        points=[grid.beta[i]],
        epsabs=2e-8,
        epsrel=2e-8,
    )[0]
    matrix_cell = grid.weights[i] * (Kab[i, i] - Kaa[i, i])
    assert np.isclose(matrix_cell, direct, rtol=3e-5, atol=3e-7)


def test_interlayer_kernel_is_weaker_and_has_same_singular_part() -> None:
    grid = mapped_beta_grid(24)
    Kaa, Kab = zhu1995_kernels(grid, 1.0, nphi_regular=160)
    assert np.all(Kab < Kaa)
    assert np.all(np.isfinite(Kaa))
    assert np.all(np.isfinite(Kab))
    # At large momentum transfer exp(-q d) suppresses the interlayer kernel.
    assert Kab[0, -1] / Kaa[0, -1] < 1e-3
