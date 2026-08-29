"""Direct Hartree/Fock oracles for the Xue/Zeng folded BHZ model.

The contractions follow Eqs. (5) and (8) of Zeng--Xue--MacDonald 2022 and
reduce to Eqs. (7)--(10) of Xue--MacDonald 2018 for one slab at Q=0. These
implementations prioritize auditable index routing over production speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.signal import fftconvolve
from numpy.typing import NDArray

from .zeng2022 import Zeng2022Parameters, ZengSlabBasis

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Q0CoulombKernel:
    """Precomputed one-slab Q=0 Coulomb matrices, target k by source k."""

    intra_ry_ab2: FloatArray
    inter_ry_ab2: FloatArray

    def validate(self, nk: int) -> None:
        intra = np.asarray(self.intra_ry_ab2, dtype=np.float64)
        inter = np.asarray(self.inter_ry_ab2, dtype=np.float64)
        if intra.shape != (nk, nk) or inter.shape != (nk, nk):
            raise ValueError("Q0 Coulomb kernels must have shape (nk,nk)")
        if not np.all(np.isfinite(intra)) or not np.all(np.isfinite(inter)):
            raise ValueError("Q0 Coulomb kernels must be finite")
        if np.any(intra < 0.0) or np.any(inter < 0.0):
            raise ValueError("Q0 Coulomb kernels must be nonnegative")
        off_diagonal = ~np.eye(nk, dtype=bool)
        if np.any(intra[off_diagonal] <= 0.0) or np.any(inter[off_diagonal] <= 0.0):
            raise ValueError("off-diagonal Q0 Coulomb kernels must be positive")
        if np.max(np.abs(intra - intra.T)) > 1.0e-12:
            raise ValueError("intralayer Q0 kernel must be symmetric")
        if np.max(np.abs(inter - inter.T)) > 1.0e-12:
            raise ValueError("interlayer Q0 kernel must be symmetric")


@dataclass(frozen=True)
class Q0ToeplitzCoulombKernel:
    """Open-boundary uniform-grid Coulomb kernels indexed by momentum offset."""

    intra_offsets_ry_ab2: FloatArray
    inter_offsets_ry_ab2: FloatArray
    mesh_shape: tuple[int, int]

    def validate(self) -> None:
        expected = (2 * self.mesh_shape[0] - 1, 2 * self.mesh_shape[1] - 1)
        intra = np.asarray(self.intra_offsets_ry_ab2, dtype=np.float64)
        inter = np.asarray(self.inter_offsets_ry_ab2, dtype=np.float64)
        if intra.shape != expected or inter.shape != expected:
            raise ValueError(f"Toeplitz kernels must have shape {expected}")
        if np.any(~np.isfinite(intra)) or np.any(~np.isfinite(inter)):
            raise ValueError("Toeplitz Coulomb kernels must be finite")
        if np.any(intra < 0.0) or np.any(inter < 0.0):
            raise ValueError("Toeplitz Coulomb kernels must be nonnegative")


@dataclass(frozen=True)
class UniformKappaMesh:
    """Uniform cell-centered Cartesian quadrature for d^2k/(2*pi)^2."""

    points_ab_inv: FloatArray
    weights_ab2: FloatArray
    cell_widths_ab_inv: tuple[float, float]
    shape: tuple[int, int]

    def validate(self) -> None:
        points = np.asarray(self.points_ab_inv, dtype=np.float64)
        weights = np.asarray(self.weights_ab2, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points_ab_inv must have shape (nk,2)")
        if weights.shape != (points.shape[0],):
            raise ValueError("weights_ab2 must have shape (nk,)")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
            raise ValueError("mesh points and weights must be finite")
        if np.any(weights <= 0.0):
            raise ValueError("mesh weights must be positive")
        dx, dy = map(float, self.cell_widths_ab_inv)
        if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
            raise ValueError("cell widths must be finite and positive")
        if self.shape[0] * self.shape[1] != points.shape[0]:
            raise ValueError("mesh shape does not match point count")
        expected_weight = dx * dy / (2.0 * np.pi) ** 2
        if not np.allclose(weights, expected_weight, rtol=1e-13, atol=1e-15):
            raise ValueError("weights do not equal d^2k/(2*pi)^2 for the stated cell")

    @property
    def nk(self) -> int:
        return int(np.asarray(self.points_ab_inv).shape[0])


def uniform_midpoint_kappa_mesh(
    *,
    kx_bounds_ab_inv: tuple[float, float],
    ky_bounds_ab_inv: tuple[float, float],
    nkx: int,
    nky: int,
) -> UniformKappaMesh:
    """Build an explicit uniform midpoint mesh on a rectangular k domain."""

    if int(nkx) != nkx or int(nky) != nky or nkx < 1 or nky < 1:
        raise ValueError("nkx and nky must be positive integers")
    kx_min, kx_max = map(float, kx_bounds_ab_inv)
    ky_min, ky_max = map(float, ky_bounds_ab_inv)
    if not (np.isfinite(kx_min) and np.isfinite(kx_max) and kx_min < kx_max):
        raise ValueError("invalid kx bounds")
    if not (np.isfinite(ky_min) and np.isfinite(ky_max) and ky_min < ky_max):
        raise ValueError("invalid ky bounds")
    dx = (kx_max - kx_min) / int(nkx)
    dy = (ky_max - ky_min) / int(nky)
    kx = kx_min + (np.arange(int(nkx), dtype=np.float64) + 0.5) * dx
    ky = ky_min + (np.arange(int(nky), dtype=np.float64) + 0.5) * dy
    grid_x, grid_y = np.meshgrid(kx, ky, indexing="ij")
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    weights = np.full(points.shape[0], dx * dy / (2.0 * np.pi) ** 2, dtype=np.float64)
    mesh = UniformKappaMesh(
        points_ab_inv=points,
        weights_ab2=weights,
        cell_widths_ab_inv=(dx, dy),
        shape=(int(nkx), int(nky)),
    )
    mesh.validate()
    return mesh


def rectangular_coulomb_singular_cell_average(
    *,
    delta_kx_ab_inv: float,
    delta_ky_ab_inv: float,
    query_offset_x_ab_inv: float,
    query_offset_y_ab_inv: float,
    d_over_ab: float,
    quadrature_order: int = 96,
) -> tuple[float, float]:
    """Average the Coulomb kernels over a rectangle containing the query.

    ``query_offset_*`` gives the query momentum relative to the source-cell
    center. Splitting the shifted rectangle into four origin-corner
    rectangles removes the integrable singularity in polar coordinates.
    """

    dx = float(delta_kx_ab_inv)
    dy = float(delta_ky_ab_inv)
    ox = float(query_offset_x_ab_inv)
    oy = float(query_offset_y_ab_inv)
    d = float(d_over_ab)
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ValueError("cell widths must be finite and positive")
    if not np.isfinite(ox) or not np.isfinite(oy):
        raise ValueError("query offsets must be finite")
    tolerance = 1.0e-13 * max(1.0, dx, dy)
    if abs(ox) > 0.5 * dx + tolerance or abs(oy) > 0.5 * dy + tolerance:
        raise ValueError("query must lie inside the selected source cell")
    if not np.isfinite(d) or d < 0.0:
        raise ValueError("d_over_ab must be finite and nonnegative")
    if int(quadrature_order) != quadrature_order or quadrature_order < 8:
        raise ValueError("quadrature_order must be an integer >= 8")

    x_extents = (max(0.0, 0.5 * dx + ox), max(0.0, 0.5 * dx - ox))
    y_extents = (max(0.0, 0.5 * dy + oy), max(0.0, 0.5 * dy - oy))
    nodes, weights = leggauss(int(quadrature_order))
    intra_integral = 0.0
    inter_integral = 0.0
    for x_extent in x_extents:
        for y_extent in y_extents:
            if x_extent == 0.0 or y_extent == 0.0:
                continue
            intra_integral += x_extent * np.arcsinh(y_extent / x_extent)
            intra_integral += y_extent * np.arcsinh(x_extent / y_extent)
            theta_switch = float(np.arctan2(y_extent, x_extent))
            for lo, hi, boundary in (
                (0.0, theta_switch, "x"),
                (theta_switch, 0.5 * np.pi, "y"),
            ):
                if hi == lo:
                    continue
                theta = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
                if boundary == "x":
                    radial_max = x_extent / np.cos(theta)
                else:
                    radial_max = y_extent / np.sin(theta)
                if d == 0.0:
                    radial_integral = radial_max
                else:
                    radial_integral = -np.expm1(-d * radial_max) / d
                inter_integral += float(
                    0.5 * (hi - lo) * np.dot(weights, radial_integral)
                )
    area = dx * dy
    intra_average = 4.0 * np.pi * intra_integral / area
    inter_average = 4.0 * np.pi * inter_integral / area
    return float(intra_average), float(inter_average)


def rectangular_coulomb_self_cell_average(
    *,
    delta_kx_ab_inv: float,
    delta_ky_ab_inv: float,
    d_over_ab: float,
    quadrature_order: int = 96,
) -> tuple[float, float]:
    """Average ``(4*pi/q, 4*pi*exp(-qd)/q)`` over a centered rectangle.

    Polar integration removes the integrable q=0 singularity analytically.
    The intralayer integral is evaluated in closed form; the interlayer radial
    integral is analytic and only its smooth angular integral is quadratured.
    """

    return rectangular_coulomb_singular_cell_average(
        delta_kx_ab_inv=delta_kx_ab_inv,
        delta_ky_ab_inv=delta_ky_ab_inv,
        query_offset_x_ab_inv=0.0,
        query_offset_y_ab_inv=0.0,
        d_over_ab=d_over_ab,
        quadrature_order=quadrature_order,
    )


def precompute_q0_coulomb_kernel(
    mesh: UniformKappaMesh,
    *,
    d_over_ab: float,
    self_cell_quadrature_order: int = 96,
) -> Q0CoulombKernel:
    """Precompute the exact discrete one-slab kernels used by the direct oracle."""

    mesh.validate()
    d = float(d_over_ab)
    if not np.isfinite(d) or d < 0.0:
        raise ValueError("d_over_ab must be finite and nonnegative")
    points = np.asarray(mesh.points_ab_inv)
    delta = points[None, :, :] - points[:, None, :]
    q = np.linalg.norm(delta, axis=2)
    diagonal = np.eye(mesh.nk, dtype=bool)
    if np.any((q <= 1.0e-14) & ~diagonal):
        raise ValueError("Q0 mesh contains duplicate momentum points")
    intra = np.empty((mesh.nk, mesh.nk), dtype=np.float64)
    inter = np.empty_like(intra)
    off_diagonal = ~diagonal
    intra[off_diagonal] = 4.0 * np.pi / q[off_diagonal]
    inter[off_diagonal] = intra[off_diagonal] * np.exp(-q[off_diagonal] * d)
    self_intra, self_inter = rectangular_coulomb_self_cell_average(
        delta_kx_ab_inv=mesh.cell_widths_ab_inv[0],
        delta_ky_ab_inv=mesh.cell_widths_ab_inv[1],
        d_over_ab=d,
        quadrature_order=self_cell_quadrature_order,
    )
    intra[diagonal] = self_intra
    inter[diagonal] = self_inter
    kernel = Q0CoulombKernel(intra_ry_ab2=intra, inter_ry_ab2=inter)
    kernel.validate(mesh.nk)
    return kernel


def q0_coulomb_kernel_row_with_integrated_cell(
    query_kappa_ab_inv: FloatArray,
    *,
    mesh: UniformKappaMesh,
    d_over_ab: float,
    singular_cell_index: int,
    singular_cell_quadrature_order: int = 96,
) -> tuple[FloatArray, FloatArray]:
    """Build a midpoint kernel row with one shifted singular cell integrated.

    This is the Nyström extension of the saved-grid regulator: all nonsingular
    source cells retain midpoint evaluation, while the explicitly selected
    cell is integrated with constant density over its rectangle. At a mesh
    center the row exactly matches :func:`precompute_q0_coulomb_kernel`.
    """

    mesh.validate()
    query = np.asarray(query_kappa_ab_inv, dtype=np.float64)
    if query.shape != (2,) or not np.all(np.isfinite(query)):
        raise ValueError("query_kappa_ab_inv must be a finite length-two vector")
    index = int(singular_cell_index)
    if index != singular_cell_index or not 0 <= index < mesh.nk:
        raise ValueError("singular_cell_index is outside the mesh")
    d = float(d_over_ab)
    if not np.isfinite(d) or d < 0.0:
        raise ValueError("d_over_ab must be finite and nonnegative")
    points = np.asarray(mesh.points_ab_inv, dtype=np.float64)
    q = np.linalg.norm(points - query[None, :], axis=1)
    mask = np.arange(mesh.nk) != index
    if np.any(q[mask] <= 1.0e-14):
        raise ValueError("query coincides with a nonselected source cell")
    intra = np.empty(mesh.nk, dtype=np.float64)
    inter = np.empty(mesh.nk, dtype=np.float64)
    intra[mask] = 4.0 * np.pi / q[mask]
    inter[mask] = intra[mask] * np.exp(-q[mask] * d)
    offset = query - points[index]
    local_intra, local_inter = rectangular_coulomb_singular_cell_average(
        delta_kx_ab_inv=mesh.cell_widths_ab_inv[0],
        delta_ky_ab_inv=mesh.cell_widths_ab_inv[1],
        query_offset_x_ab_inv=float(offset[0]),
        query_offset_y_ab_inv=float(offset[1]),
        d_over_ab=d,
        quadrature_order=singular_cell_quadrature_order,
    )
    intra[index] = local_intra
    inter[index] = local_inter
    return intra, inter


def q0_fock_at_k_from_integrated_cell_row(
    density_delta: ComplexArray,
    *,
    basis: ZengSlabBasis,
    mesh: UniformKappaMesh,
    intra_row_ry_ab2: FloatArray,
    inter_row_ry_ab2: FloatArray,
) -> ComplexArray:
    """Evaluate the Q=0 Fock self-energy at one query momentum."""

    density = _validate_density(density_delta, basis=basis, mesh=mesh)
    if basis.slab_indices != (0,):
        raise ValueError("single-k Q0 Fock requires the one-slab basis")
    intra = np.asarray(intra_row_ry_ab2, dtype=np.float64)
    inter = np.asarray(inter_row_ry_ab2, dtype=np.float64)
    if intra.shape != (mesh.nk,) or inter.shape != (mesh.nk,):
        raise ValueError("single-k kernel rows must have shape (nk,)")
    if np.any(~np.isfinite(intra)) or np.any(~np.isfinite(inter)):
        raise ValueError("single-k kernel rows must be finite")
    weighted_density = density * np.asarray(mesh.weights_ab2)[None, None, :]
    result = np.empty((basis.dimension, basis.dimension), dtype=np.complex128)
    for row in range(basis.dimension):
        row_band, _row_spin, _row_slab = basis.label(row)
        for col in range(basis.dimension):
            col_band, _col_spin, _col_slab = basis.label(col)
            kernel = intra if row_band == col_band else inter
            result[row, col] = -np.dot(kernel, weighted_density[row, col, :])
    return result


def q0_fock_from_precomputed_kernel(
    density_delta: ComplexArray,
    *,
    basis: ZengSlabBasis,
    mesh: UniformKappaMesh,
    kernel: Q0CoulombKernel,
) -> ComplexArray:
    """Apply the one-slab Q=0 Fock action by exact matrix contractions."""

    density = _validate_density(density_delta, basis=basis, mesh=mesh)
    if basis.slab_indices != (0,):
        raise ValueError("precomputed Q0 Fock requires the one-slab basis")
    kernel.validate(mesh.nk)
    weighted_density = density * np.asarray(mesh.weights_ab2)[None, None, :]
    result = np.empty_like(density)
    for row in range(basis.dimension):
        row_band, _row_spin, _row_slab = basis.label(row)
        for col in range(basis.dimension):
            col_band, _col_spin, _col_slab = basis.label(col)
            matrix = kernel.intra_ry_ab2 if row_band == col_band else kernel.inter_ry_ab2
            result[row, col, :] = -(matrix @ weighted_density[row, col, :])
    return result


def q0_interaction_from_precomputed_kernel(
    density_delta: ComplexArray,
    *,
    basis: ZengSlabBasis,
    mesh: UniformKappaMesh,
    params: Zeng2022Parameters,
    kernel: Q0CoulombKernel,
    neutrality_tolerance: float = 1.0e-10,
) -> ComplexArray:
    """Return Q=0 Hartree plus the precomputed exact-discrete Fock action."""

    if params.q_ab_inv != 0.0:
        raise ValueError("precomputed Q0 interaction requires q_ab_inv=0")
    hartree = zeng2022_hartree_direct(
        density_delta,
        basis=basis,
        mesh=mesh,
        params=params,
        neutrality_tolerance=neutrality_tolerance,
    )
    fock = q0_fock_from_precomputed_kernel(
        density_delta,
        basis=basis,
        mesh=mesh,
        kernel=kernel,
    )
    return hartree + fock


def precompute_q0_toeplitz_coulomb_kernel(
    mesh: UniformKappaMesh,
    *,
    d_over_ab: float,
    self_cell_quadrature_order: int = 96,
    omit_self_cell_diagnostic: bool = False,
) -> Q0ToeplitzCoulombKernel:
    """Precompute open-boundary difference kernels for FFT convolution."""

    mesh.validate()
    d = float(d_over_ab)
    if not np.isfinite(d) or d < 0.0:
        raise ValueError("d_over_ab must be finite and nonnegative")
    nx, ny = mesh.shape
    dx, dy = mesh.cell_widths_ab_inv
    ix = np.arange(-(nx - 1), nx, dtype=np.float64) * dx
    iy = np.arange(-(ny - 1), ny, dtype=np.float64) * dy
    qx, qy = np.meshgrid(ix, iy, indexing="ij")
    q = np.hypot(qx, qy)
    center = (nx - 1, ny - 1)
    nonzero = q > 0.0
    intra = np.empty_like(q)
    inter = np.empty_like(q)
    intra[nonzero] = 4.0 * np.pi / q[nonzero]
    inter[nonzero] = intra[nonzero] * np.exp(-q[nonzero] * d)
    if omit_self_cell_diagnostic:
        intra[center] = 0.0
        inter[center] = 0.0
    else:
        self_intra, self_inter = rectangular_coulomb_self_cell_average(
            delta_kx_ab_inv=dx,
            delta_ky_ab_inv=dy,
            d_over_ab=d,
            quadrature_order=self_cell_quadrature_order,
        )
        intra[center] = self_intra
        inter[center] = self_inter
    kernel = Q0ToeplitzCoulombKernel(
        intra_offsets_ry_ab2=intra,
        inter_offsets_ry_ab2=inter,
        mesh_shape=mesh.shape,
    )
    kernel.validate()
    return kernel


def q0_fock_from_toeplitz_kernel(
    density_delta: ComplexArray,
    *,
    basis: ZengSlabBasis,
    mesh: UniformKappaMesh,
    kernel: Q0ToeplitzCoulombKernel,
) -> ComplexArray:
    """Apply the exact open-boundary discrete Q=0 Fock convolution by FFT."""

    density = _validate_density(density_delta, basis=basis, mesh=mesh)
    if basis.slab_indices != (0,):
        raise ValueError("Toeplitz Q0 Fock requires the one-slab basis")
    kernel.validate()
    if kernel.mesh_shape != mesh.shape:
        raise ValueError("Toeplitz kernel mesh shape mismatch")
    weight = float(np.asarray(mesh.weights_ab2)[0])
    if not np.allclose(mesh.weights_ab2, weight, rtol=1e-13, atol=1e-15):
        raise ValueError("Toeplitz Q0 Fock requires uniform weights")
    result = np.empty_like(density)
    for row in range(basis.dimension):
        row_band, _row_spin, _row_slab = basis.label(row)
        for col in range(basis.dimension):
            col_band, _col_spin, _col_slab = basis.label(col)
            offsets = (
                kernel.intra_offsets_ry_ab2
                if row_band == col_band
                else kernel.inter_offsets_ry_ab2
            )
            source = density[row, col, :].reshape(mesh.shape)
            result[row, col, :] = (
                -weight * fftconvolve(source, offsets, mode="same")
            ).ravel()
    return result


def q0_interaction_from_toeplitz_kernel(
    density_delta: ComplexArray,
    *,
    basis: ZengSlabBasis,
    mesh: UniformKappaMesh,
    params: Zeng2022Parameters,
    kernel: Q0ToeplitzCoulombKernel,
    neutrality_tolerance: float = 1.0e-10,
) -> ComplexArray:
    """Return Q=0 Hartree plus exact-discrete Toeplitz-FFT Fock."""

    if params.q_ab_inv != 0.0:
        raise ValueError("Toeplitz Q0 interaction requires q_ab_inv=0")
    return zeng2022_hartree_direct(
        density_delta,
        basis=basis,
        mesh=mesh,
        params=params,
        neutrality_tolerance=neutrality_tolerance,
    ) + q0_fock_from_toeplitz_kernel(
        density_delta,
        basis=basis,
        mesh=mesh,
        kernel=kernel,
    )


def _validate_density(
    density_delta: ComplexArray,
    *,
    basis: ZengSlabBasis,
    mesh: UniformKappaMesh,
) -> ComplexArray:
    mesh.validate()
    density = np.asarray(density_delta, dtype=np.complex128)
    expected = (basis.dimension, basis.dimension, mesh.nk)
    if density.shape != expected:
        raise ValueError(f"density_delta shape {density.shape} does not match {expected}")
    if not np.all(np.isfinite(density)):
        raise ValueError("density_delta must be finite")
    return density


def zeng2022_hartree_direct(
    density_delta: ComplexArray,
    *,
    basis: ZengSlabBasis,
    mesh: UniformKappaMesh,
    params: Zeng2022Parameters,
    neutrality_tolerance: float = 1.0e-10,
) -> ComplexArray:
    """Literal finite-regulator Hartree action from Eq. (5)."""

    density = _validate_density(density_delta, basis=basis, mesh=mesh)
    params.validate()
    if len(basis.slab_indices) > 1 and params.q_ab_inv <= 0.0:
        raise ValueError("multiple folded slabs require Q>0; Q=0 uses one unfurled slab")
    result = np.zeros_like(density)
    weights = np.asarray(mesh.weights_ab2)
    slabs = basis.slab_indices
    slab_set = set(slabs)
    transfers = range(slabs[0] - slabs[-1], slabs[-1] - slabs[0] + 1)

    for transfer in transfers:
        harmonics: dict[str, complex] = {"c": 0.0j, "v": 0.0j}
        for band in ("c", "v"):
            value = 0.0j
            for spin in ("up", "down"):
                for source_slab in slabs:
                    shifted_slab = source_slab + transfer
                    if shifted_slab not in slab_set:
                        continue
                    row = basis.index(band, spin, shifted_slab)
                    col = basis.index(band, spin, source_slab)
                    value += np.dot(weights, density[row, col, :])
            harmonics[band] = complex(value)

        if transfer == 0:
            total = harmonics["c"] + harmonics["v"]
            scale = max(1.0, abs(harmonics["c"]), abs(harmonics["v"]))
            if abs(total) > float(neutrality_tolerance) * scale:
                raise ValueError(
                    "uniform Hartree q=0 requires neutral reference-subtracted density: "
                    f"n_c+n_v={total!r}"
                )
            potential = {
                "c": 4.0 * np.pi * params.d_over_ab * harmonics["c"],
                "v": -4.0 * np.pi * params.d_over_ab * harmonics["c"],
            }
        else:
            q = abs(transfer * params.q_ab_inv)
            intra = 4.0 * np.pi / q
            inter = intra * np.exp(-q * params.d_over_ab)
            potential = {
                "c": intra * harmonics["c"] + inter * harmonics["v"],
                "v": inter * harmonics["c"] + intra * harmonics["v"],
            }

        for source_slab in slabs:
            target_slab = source_slab + transfer
            if target_slab not in slab_set:
                continue
            for band in ("c", "v"):
                for spin in ("up", "down"):
                    row = basis.index(band, spin, target_slab)
                    col = basis.index(band, spin, source_slab)
                    result[row, col, :] = potential[band]
    return result


def zeng2022_fock_direct(
    density_delta: ComplexArray,
    *,
    basis: ZengSlabBasis,
    mesh: UniformKappaMesh,
    params: Zeng2022Parameters,
    self_cell_quadrature_order: int = 96,
) -> ComplexArray:
    """Literal finite-regulator Fock action from Eq. (8)."""

    density = _validate_density(density_delta, basis=basis, mesh=mesh)
    params.validate()
    if len(basis.slab_indices) > 1 and params.q_ab_inv <= 0.0:
        raise ValueError("multiple folded slabs require Q>0; Q=0 uses one unfurled slab")
    points = np.asarray(mesh.points_ab_inv)
    weights = np.asarray(mesh.weights_ab2)
    self_intra, self_inter = rectangular_coulomb_self_cell_average(
        delta_kx_ab_inv=mesh.cell_widths_ab_inv[0],
        delta_ky_ab_inv=mesh.cell_widths_ab_inv[1],
        d_over_ab=params.d_over_ab,
        quadrature_order=self_cell_quadrature_order,
    )
    result = np.zeros_like(density)
    slabs = basis.slab_indices
    slab_set = set(slabs)

    for ik, target_kappa in enumerate(points):
        for target_slab in slabs:
            for target_row_slab in slabs:
                transfer = target_row_slab - target_slab
                for band in ("c", "v"):
                    for spin in ("up", "down"):
                        col = basis.index(band, spin, target_slab)
                        for row_band in ("c", "v"):
                            same_layer = row_band == band
                            for row_spin in ("up", "down"):
                                row = basis.index(row_band, row_spin, target_row_slab)
                                value = 0.0j
                                for source_slab in slabs:
                                    shifted_slab = source_slab + transfer
                                    if shifted_slab not in slab_set:
                                        continue
                                    density_row = basis.index(row_band, row_spin, shifted_slab)
                                    density_col = basis.index(band, spin, source_slab)
                                    delta = points - target_kappa
                                    delta = delta.copy()
                                    delta[:, 0] += (source_slab - target_slab) * params.q_ab_inv
                                    q = np.linalg.norm(delta, axis=1)
                                    kernel = np.empty(mesh.nk, dtype=np.float64)
                                    zero_mask = q <= 1.0e-14
                                    nonzero = ~zero_mask
                                    kernel[nonzero] = 4.0 * np.pi / q[nonzero]
                                    if not same_layer:
                                        kernel[nonzero] *= np.exp(-q[nonzero] * params.d_over_ab)
                                    kernel[zero_mask] = self_intra if same_layer else self_inter
                                    value -= np.dot(
                                        weights * kernel,
                                        density[density_row, density_col, :],
                                    )
                                result[row, col, ik] = value
    return result


def zeng2022_interaction_direct(
    density_delta: ComplexArray,
    *,
    basis: ZengSlabBasis,
    mesh: UniformKappaMesh,
    params: Zeng2022Parameters,
    neutrality_tolerance: float = 1.0e-10,
    self_cell_quadrature_order: int = 96,
) -> ComplexArray:
    """Return ``Sigma_H[D]+Sigma_F[D]`` from the direct source equations."""

    hartree = zeng2022_hartree_direct(
        density_delta,
        basis=basis,
        mesh=mesh,
        params=params,
        neutrality_tolerance=neutrality_tolerance,
    )
    fock = zeng2022_fock_direct(
        density_delta,
        basis=basis,
        mesh=mesh,
        params=params,
        self_cell_quadrature_order=self_cell_quadrature_order,
    )
    return hartree + fock


def zeng2022_interaction_energy_density(
    density_delta: ComplexArray,
    interaction_h: ComplexArray,
    *,
    mesh: UniformKappaMesh,
) -> float:
    """Return ``0.5*integral Tr(Sigma[D] D)`` in ``Ry*/a_B*^2``."""

    density = np.asarray(density_delta, dtype=np.complex128)
    sigma = np.asarray(interaction_h, dtype=np.complex128)
    if density.shape != sigma.shape or density.ndim != 3 or density.shape[2] != mesh.nk:
        raise ValueError("density and interaction_h must have matching (dim,dim,nk) shapes")
    value = 0.5 * np.einsum(
        "abk,bak,k->", sigma, density, np.asarray(mesh.weights_ab2), optimize=True
    )
    if abs(value.imag) > 1.0e-10 * max(1.0, abs(value.real)):
        raise ValueError(f"interaction energy is not real: {value!r}")
    return float(value.real)


__all__ = [
    "Q0CoulombKernel",
    "Q0ToeplitzCoulombKernel",
    "UniformKappaMesh",
    "precompute_q0_coulomb_kernel",
    "precompute_q0_toeplitz_coulomb_kernel",
    "q0_coulomb_kernel_row_with_integrated_cell",
    "q0_fock_at_k_from_integrated_cell_row",
    "q0_fock_from_toeplitz_kernel",
    "q0_interaction_from_toeplitz_kernel",
    "q0_fock_from_precomputed_kernel",
    "q0_interaction_from_precomputed_kernel",
    "rectangular_coulomb_self_cell_average",
    "rectangular_coulomb_singular_cell_average",
    "uniform_midpoint_kappa_mesh",
    "zeng2022_fock_direct",
    "zeng2022_hartree_direct",
    "zeng2022_interaction_direct",
    "zeng2022_interaction_energy_density",
]
