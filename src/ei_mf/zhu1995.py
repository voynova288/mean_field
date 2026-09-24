"""Native-unit benchmark for Zhu et al., Phys. Rev. Lett. 74, 1633 (1995).

The paper uses spinless effective atomic units, ``epsilon_k=k^2``, momentum
mapping ``k=tan(beta)``, ``V_ee(q)=2*pi/q``, and
``V_eh(q)=2*pi*exp(-q*d)/q``.  This module is an independent benchmark of the
radial Coulomb/BCS core; it is not a fit to Du et al. digitized curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy.optimize import root
from scipy.special import ellipkm1

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Zhu1995Grid:
    beta: Array
    k: Array
    weights: Array
    beta_edges: Array


@dataclass(frozen=True)
class Zhu1995Result:
    grid: Zhu1995Grid
    rs: float
    d: float
    mu: float
    Delta: Array
    xi: Array
    E_pair: Array
    density: float
    density_target: float
    energy_per_pair_eq7: float
    energy_per_pair_with_unneutralized_hartree: float
    mu_with_unneutralized_hartree: float
    residual: float
    gap_residual: float
    xi_residual: float
    density_relative_residual: float
    converged: bool
    iterations: int

    @property
    def kF_spinless(self) -> float:
        return 2.0 / self.rs

    @property
    def Delta_max(self) -> float:
        return float(np.max(np.abs(self.Delta)))

    @property
    def E_min(self) -> float:
        return float(np.min(self.E_pair))


@dataclass(frozen=True)
class Zhu1995FixedMuResult:
    """Paired root of the thesis equations at an externally fixed ``mu``."""

    grid: Zhu1995Grid
    d: float
    mu: float
    Delta: Array
    xi: Array
    E_pair: Array
    density: float
    residual: float
    gap_residual: float
    xi_residual: float
    converged: bool
    iterations: int

    @property
    def Delta_max(self) -> float:
        return float(np.max(np.abs(self.Delta)))

    @property
    def E_min(self) -> float:
        return float(np.min(self.E_pair))


def mapped_beta_grid(
    nbeta: int,
    *,
    k_max: float | None = None,
) -> Zhu1995Grid:
    """Return a mapped radial Gauss-Legendre grid.

    The source-faithful default uses ``beta in (0,pi/2)`` and
    ``k=tan(beta)``, so ``weights`` implement
    ``int_0^infty k dk/(2*pi)``. A finite positive ``k_max`` instead maps
    ``beta in (0,atan(k_max))``. The latter is an explicitly forensic hard
    momentum cutoff for testing an undocumented author-side convention; it is
    not a convergence replacement for the infinite-domain benchmark.
    """

    if nbeta < 8:
        raise ValueError("nbeta must be at least 8")
    if k_max is not None and (not np.isfinite(k_max) or k_max <= 0.0):
        raise ValueError("k_max must be finite and positive when supplied")
    beta_max = 0.5 * np.pi if k_max is None else float(np.arctan(k_max))
    x, wx = leggauss(int(nbeta))
    beta = 0.5 * beta_max * (x + 1.0)
    wb = 0.5 * beta_max * wx
    k = np.tan(beta)
    sec2 = 1.0 / np.cos(beta) ** 2
    weights = wb * k * sec2 / (2.0 * np.pi)
    edges = np.empty(nbeta + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (beta[:-1] + beta[1:])
    edges[0] = max(0.0, beta[0] - 0.5 * (beta[1] - beta[0]))
    edges[-1] = beta_max
    return Zhu1995Grid(beta, k, weights, edges)


def density_from_rs(rs: float) -> float:
    """Spinless pair density ``n=1/(pi*rs^2)`` in effective a.u."""

    if rs <= 0.0:
        raise ValueError("rs must be positive")
    return 1.0 / (np.pi * float(rs) ** 2)


def _angular_coulomb_intralayer(k: Array) -> Array:
    """Return ``(1/2pi) int dphi [2pi/q]`` off the diagonal."""

    ki = k[:, None]
    kj = k[None, :]
    denom = ki + kj
    complementary = ((ki - kj) / denom) ** 2
    out = 4.0 * ellipkm1(complementary) / denom
    np.fill_diagonal(out, np.nan)
    return out


def _diagonal_cell_average(k0: float, beta_lo: float, beta_hi: float) -> float:
    """Cell-integrate the logarithmic singularity with split Gauss quadrature."""

    beta0 = float(np.arctan(k0))
    x, wx = leggauss(96)

    def segment(a: float, b: float) -> float:
        if b <= a:
            return 0.0
        beta = 0.5 * (b - a) * x + 0.5 * (a + b)
        wb = 0.5 * (b - a) * wx
        kp = np.tan(beta)
        denom = k0 + kp
        complementary = ((k0 - kp) / denom) ** 2
        angular = 4.0 * ellipkm1(complementary) / denom
        integrand = kp / np.cos(beta) ** 2 * angular / (2.0 * np.pi)
        return float(np.dot(wb, integrand))

    return segment(float(beta_lo), beta0) + segment(beta0, float(beta_hi))


def zhu1995_kernels(
    grid: Zhu1995Grid,
    d: float,
    *,
    nphi_regular: int = 240,
) -> tuple[Array, Array]:
    """Build intralayer/interlayer angular kernels with diagonal cell averaging.

    The logarithmic ``1/q`` singularity is integrated over each diagonal beta
    cell.  The finite interlayer correction is evaluated as
    ``int dphi [exp(-q*d)-1]/q``, whose ``q->0`` limit is ``-d``.
    """

    if d < 0.0:
        raise ValueError("d must be non-negative")
    if nphi_regular < 32:
        raise ValueError("nphi_regular must be at least 32")
    k = grid.k
    Kaa = _angular_coulomb_intralayer(k)
    for i, (ki, blo, bhi, wi) in enumerate(
        zip(k, grid.beta_edges[:-1], grid.beta_edges[1:], grid.weights)
    ):
        Kaa[i, i] = _diagonal_cell_average(float(ki), float(blo), float(bhi)) / float(wi)

    if d == 0.0:
        return Kaa, Kaa.copy()

    x, wx = leggauss(int(nphi_regular))
    phi = np.pi * (x + 1.0)
    wphi = np.pi * wx
    def regular_correction(ki: float, kp: Array) -> Array:
        q = np.sqrt(
            np.maximum(
                ki * ki + kp[:, None] ** 2 - 2.0 * ki * kp[:, None] * np.cos(phi),
                0.0,
            )
        )
        regular = np.empty_like(q)
        mask = q > 1e-13
        regular[mask] = np.expm1(-d * q[mask]) / q[mask]
        regular[~mask] = -d
        return regular @ wphi

    correction = np.empty_like(Kaa)
    xcell, wcell = leggauss(48)
    for i, ki in enumerate(k):
        correction[i] = regular_correction(float(ki), k)
        beta0 = float(grid.beta[i])
        cell_integral = 0.0
        for a, b in ((float(grid.beta_edges[i]), beta0), (beta0, float(grid.beta_edges[i + 1]))):
            if b <= a:
                continue
            beta_cell = 0.5 * (b - a) * xcell + 0.5 * (a + b)
            wb_cell = 0.5 * (b - a) * wcell
            kp_cell = np.tan(beta_cell)
            radial = kp_cell / np.cos(beta_cell) ** 2 / (2.0 * np.pi)
            cell_integral += float(
                np.dot(wb_cell, radial * regular_correction(float(ki), kp_cell))
            )
        correction[i, i] = cell_integral / float(grid.weights[i])
    return Kaa, Kaa + correction


def zhu1995_residual(
    x: Array,
    grid: Zhu1995Grid,
    Kaa: Array,
    Kab: Array,
    density_target: float,
    *,
    energy_floor: float = 1e-12,
) -> Array:
    """Coupled Eqs. (4)-(6) plus the fixed-density condition."""

    n = grid.k.size
    Delta = np.asarray(x[:n], dtype=np.float64)
    xi = np.asarray(x[n : 2 * n], dtype=np.float64)
    mu = float(x[-1])
    E = np.maximum(np.sqrt(Delta * Delta + xi * xi), energy_floor)
    Delta_rhs = Kab @ (grid.weights * Delta / E)
    xi_rhs = grid.k * grid.k - mu - Kaa @ (grid.weights * (1.0 - xi / E))
    density = float(np.dot(grid.weights, 0.5 * (1.0 - xi / E)))
    return np.concatenate(
        [Delta - Delta_rhs, xi - xi_rhs, [(density - density_target) / density_target]]
    )


def zhu1995_fixed_mu_residual(
    x: Array,
    grid: Zhu1995Grid,
    Kaa: Array,
    Kab: Array,
    mu: float,
    *,
    energy_floor: float = 1e-12,
) -> Array:
    """Coupled thesis gap/exchange equations at externally fixed ``mu``."""

    n = grid.k.size
    Delta = np.asarray(x[:n], dtype=np.float64)
    xi = np.asarray(x[n : 2 * n], dtype=np.float64)
    E = np.maximum(np.sqrt(Delta * Delta + xi * xi), energy_floor)
    Delta_rhs = Kab @ (grid.weights * Delta / E)
    xi_rhs = grid.k * grid.k - float(mu) - Kaa @ (
        grid.weights * (1.0 - xi / E)
    )
    return np.concatenate([Delta - Delta_rhs, xi - xi_rhs])


def solve_zhu1995_fixed_mu(
    mu: float,
    d: float,
    *,
    nbeta: int = 120,
    nphi_regular: int = 240,
    initial: Zhu1995Result | Zhu1995FixedMuResult | None = None,
    tol: float = 2e-8,
    max_iter: int = 900,
    k_max: float | None = None,
) -> Zhu1995FixedMuResult:
    """Solve the source thesis equations with fixed pair chemical potential.

    The density is an output. A finite ``k_max`` remains a forensic hidden
    convention rather than a controlled ultraviolet completion.
    """

    if not np.isfinite(mu):
        raise ValueError("mu must be finite")
    grid = mapped_beta_grid(nbeta, k_max=k_max)
    Kaa, Kab = zhu1995_kernels(grid, d, nphi_regular=nphi_regular)
    if initial is None:
        width = 0.75
        Delta0 = 0.5 / (1.0 + (grid.k / width) ** 2) ** 1.5
        xi0 = grid.k * grid.k - float(mu)
    else:
        Delta0 = np.interp(
            grid.k,
            initial.grid.k,
            initial.Delta,
            left=initial.Delta[0],
            right=0.0,
        )
        xi0 = np.interp(
            grid.k,
            initial.grid.k,
            initial.xi,
            left=initial.xi[0],
            right=grid.k[-1] ** 2 - float(mu),
        )
    x0 = np.concatenate([Delta0, xi0])
    residual = lambda x: zhu1995_fixed_mu_residual(
        x, grid, Kaa, Kab, float(mu)
    )
    initial_residual = float(np.max(np.abs(residual(x0))))
    if initial_residual <= tol:
        solution_x = x0.copy()
        solution_success = True
        solution_iterations = 0
    else:
        solution = root(
            residual,
            x0,
            method="krylov",
            options={"fatol": float(tol), "maxiter": int(max_iter), "disp": False},
        )
        solution_x = np.asarray(solution.x, dtype=np.float64)
        solution_success = bool(solution.success)
        solution_iterations = int(getattr(solution, "nit", max_iter))
    Delta = np.asarray(solution_x[:nbeta], dtype=np.float64)
    xi = np.asarray(solution_x[nbeta : 2 * nbeta], dtype=np.float64)
    if float(np.dot(grid.weights, Delta)) < 0.0:
        Delta = -Delta
        solution_x[:nbeta] = Delta
    E = np.sqrt(Delta * Delta + xi * xi)
    occupation = 0.5 * (1.0 - xi / np.maximum(E, 1e-12))
    density = float(np.dot(grid.weights, occupation))
    residual_vector = residual(solution_x)
    gap_residual = float(np.max(np.abs(residual_vector[:nbeta])))
    xi_residual = float(np.max(np.abs(residual_vector[nbeta : 2 * nbeta])))
    equation_residual = max(gap_residual, xi_residual)
    return Zhu1995FixedMuResult(
        grid=grid,
        d=float(d),
        mu=float(mu),
        Delta=Delta,
        xi=xi,
        E_pair=E,
        density=density,
        residual=equation_residual,
        gap_residual=gap_residual,
        xi_residual=xi_residual,
        converged=solution_success and equation_residual <= tol,
        iterations=solution_iterations,
    )


def solve_zhu1995_fixed_rs(
    rs: float,
    d: float,
    *,
    nbeta: int = 120,
    nphi_regular: int = 240,
    initial: Zhu1995Result | None = None,
    tol: float = 2e-8,
    max_iter: int = 900,
    k_max: float | None = None,
) -> Zhu1995Result:
    """Solve the Zhu 1995 zero-temperature equations at fixed ``rs`` and ``d``.

    ``k_max=None`` retains the source benchmark's infinite compactified
    domain. A finite value is a forensic hard-cutoff hypothesis and must not
    be presented as a converged physical regulator.
    """

    grid = mapped_beta_grid(nbeta, k_max=k_max)
    Kaa, Kab = zhu1995_kernels(grid, d, nphi_regular=nphi_regular)
    target = density_from_rs(rs)
    kF = 2.0 / float(rs)
    if initial is None:
        width = max(0.5, kF)
        Delta0 = 0.5 / (1.0 + (grid.k / width) ** 2) ** 1.5
        xi0 = grid.k * grid.k - kF * kF
        mu0 = kF * kF - 4.0 * kF / np.pi
    else:
        Delta0 = np.interp(grid.k, initial.grid.k, initial.Delta, left=initial.Delta[0], right=0.0)
        xi0 = np.interp(
            grid.k,
            initial.grid.k,
            initial.xi,
            left=initial.xi[0],
            right=grid.k[-1] ** 2 - initial.mu,
        )
        mu0 = initial.mu
    x0 = np.concatenate([Delta0, xi0, [mu0]])
    residual = lambda x: zhu1995_residual(x, grid, Kaa, Kab, target)
    solution = root(
        residual,
        x0,
        method="krylov",
        options={"fatol": float(tol), "maxiter": int(max_iter), "disp": False},
    )
    Delta = np.asarray(solution.x[:nbeta], dtype=np.float64)
    xi = np.asarray(solution.x[nbeta : 2 * nbeta], dtype=np.float64)
    if float(np.dot(grid.weights, Delta)) < 0.0:
        Delta = -Delta
        solution.x[:nbeta] = Delta
    mu = float(solution.x[-1])
    E = np.sqrt(Delta * Delta + xi * xi)
    occupation = 0.5 * (1.0 - xi / np.maximum(E, 1e-12))
    density = float(np.dot(grid.weights, occupation))
    integrand = 0.5 * (
        (grid.k * grid.k + mu + xi) * occupation
        - Delta * (Delta / np.maximum(E, 1e-12)) / 2.0
    )
    energy_density = float(np.dot(grid.weights, integrand))
    residual_vector = residual(solution.x)
    gap_residual = float(np.max(np.abs(residual_vector[:nbeta])))
    xi_residual = float(np.max(np.abs(residual_vector[nbeta : 2 * nbeta])))
    density_relative_residual = float(abs(residual_vector[-1]))
    equation_residual = max(gap_residual, xi_residual, density_relative_residual)
    energy_per_pair_eq7 = energy_density / density
    return Zhu1995Result(
        grid=grid,
        rs=float(rs),
        d=float(d),
        mu=mu,
        Delta=Delta,
        xi=xi,
        E_pair=E,
        density=density,
        density_target=target,
        energy_per_pair_eq7=energy_per_pair_eq7,
        energy_per_pair_with_unneutralized_hartree=(
            energy_per_pair_eq7 + 2.0 * float(d) / float(rs) ** 2
        ),
        mu_with_unneutralized_hartree=mu + 4.0 * float(d) / float(rs) ** 2,
        residual=equation_residual,
        gap_residual=gap_residual,
        xi_residual=xi_residual,
        density_relative_residual=density_relative_residual,
        converged=bool(solution.success) and equation_residual <= tol,
        iterations=int(getattr(solution, "nit", max_iter)),
    )
