"""Source-bound Gaussian-Wannier Keldysh inputs for TPT monolayer HF.

This module owns the physical approximation used to turn matching Wannier90
centres and spreads into an orbital-resolved density--density interaction.  It
does not claim that the isotropic Gaussian densities are the unavailable real
Wannier charge densities.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Iterable

import numpy as np
from numpy.polynomial.legendre import leggauss

from mean_field.core.hf.real_space_density_density import (
    PeriodicDensityDensityKernel,
)

COULOMB_EV_ANGSTROM = 14.3996454784255
TPT_PUBLISHED_ALPHA_2D_ANGSTROM = 17.854
_FINAL_WF_PATTERN = re.compile(
    r"^\s*WF centre and spread\s+(\d+)\s+\(\s*"
    r"([-+0-9.Ee]+)\s*,\s*([-+0-9.Ee]+)\s*,\s*([-+0-9.Ee]+)\s*\)\s*"
    r"([-+0-9.Ee]+)\s*$"
)


@dataclass(frozen=True)
class WannierGaussianDensities:
    """Ordered centres and isotropic Gaussian widths derived from W90 output."""

    centres_cartesian_angstrom: np.ndarray
    total_spreads_angstrom2: np.ndarray
    gaussian_widths_angstrom: np.ndarray

    def __post_init__(self) -> None:
        centres = np.asarray(self.centres_cartesian_angstrom, dtype=float)
        spreads = np.asarray(self.total_spreads_angstrom2, dtype=float)
        widths = np.asarray(self.gaussian_widths_angstrom, dtype=float)
        if centres.ndim != 2 or centres.shape[1] != 3:
            raise ValueError("centres must have shape (n_orbital,3)")
        if spreads.shape != (centres.shape[0],) or widths.shape != spreads.shape:
            raise ValueError("spread and width arrays must match the centre inventory")
        if (
            centres.shape[0] == 0
            or not np.all(np.isfinite(centres))
            or not np.all(np.isfinite(spreads))
            or not np.all(np.isfinite(widths))
            or np.any(spreads <= 0.0)
            or np.any(widths <= 0.0)
        ):
            raise ValueError("Wannier Gaussian data must be finite and positive")
        if not np.allclose(widths * widths, spreads, rtol=2.0e-14, atol=2.0e-14):
            raise ValueError("Gaussian width convention requires ell=sqrt(total spread)")
        centres = np.ascontiguousarray(centres)
        spreads = np.ascontiguousarray(spreads)
        widths = np.ascontiguousarray(widths)
        centres.setflags(write=False)
        spreads.setflags(write=False)
        widths.setflags(write=False)
        object.__setattr__(self, "centres_cartesian_angstrom", centres)
        object.__setattr__(self, "total_spreads_angstrom2", spreads)
        object.__setattr__(self, "gaussian_widths_angstrom", widths)

    @property
    def n_orbital(self) -> int:
        return int(self.total_spreads_angstrom2.size)


@dataclass(frozen=True)
class WannierHRData:
    """Degeneracy-normalized Wannier90 real-space Hamiltonian blocks."""

    r_vectors: np.ndarray
    h_r_ev: np.ndarray
    degeneracies: np.ndarray

    def __post_init__(self) -> None:
        r_vectors = np.asarray(self.r_vectors, dtype=np.int64)
        h_r = np.asarray(self.h_r_ev, dtype=np.complex128)
        degeneracies = np.asarray(self.degeneracies, dtype=np.int64)
        if r_vectors.ndim != 2 or r_vectors.shape[1] != 3:
            raise ValueError("r_vectors must have shape (nr,3)")
        nr = r_vectors.shape[0]
        if h_r.ndim != 3 or h_r.shape[0] != nr or h_r.shape[1] != h_r.shape[2]:
            raise ValueError("h_r_ev must have shape (nr,nw,nw)")
        if degeneracies.shape != (nr,) or np.any(degeneracies <= 0):
            raise ValueError("degeneracies must be positive with shape (nr,)")
        if not np.all(np.isfinite(h_r)):
            raise ValueError("h_r_ev must be finite")
        r_vectors = np.ascontiguousarray(r_vectors)
        h_r = np.ascontiguousarray(h_r)
        degeneracies = np.ascontiguousarray(degeneracies)
        r_vectors.setflags(write=False)
        h_r.setflags(write=False)
        degeneracies.setflags(write=False)
        object.__setattr__(self, "r_vectors", r_vectors)
        object.__setattr__(self, "h_r_ev", h_r)
        object.__setattr__(self, "degeneracies", degeneracies)

    @property
    def n_wannier(self) -> int:
        return int(self.h_r_ev.shape[1])


@dataclass(frozen=True)
class KeldyshKernelBuild:
    """Built HF kernel plus numerical provenance needed for convergence gates."""

    kernel: PeriodicDensityDensityKernel
    reciprocal_vectors_angstrom_inv: np.ndarray
    reciprocal_shell_indices: np.ndarray
    zero_cell_matrix_ev: np.ndarray
    metadata: dict[str, int | float | str]


def parse_final_wannier_gaussian_densities(
    wout_path: str | Path,
    *,
    expected_num_wann: int,
) -> WannierGaussianDensities:
    """Parse the final ordered W90 centres/spreads and set ``ell=sqrt(Omega)``."""

    path = Path(wout_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    final_positions = [index for index, line in enumerate(lines) if line.strip() == "Final State"]
    if not final_positions:
        raise ValueError(f"no final Wannier state found in {path}")
    records: list[tuple[int, tuple[float, float, float], float]] = []
    for line in lines[final_positions[-1] + 1 :]:
        match = _FINAL_WF_PATTERN.match(line)
        if match is None:
            if records:
                break
            continue
        records.append(
            (
                int(match.group(1)),
                (float(match.group(2)), float(match.group(3)), float(match.group(4))),
                float(match.group(5)),
            )
        )
    expected = int(expected_num_wann)
    if expected <= 0 or len(records) != expected:
        raise ValueError(
            f"expected {expected} final Wannier records in {path}, found {len(records)}"
        )
    indices = [record[0] for record in records]
    if indices != list(range(1, expected + 1)):
        raise ValueError("final Wannier records are not complete and ordered")
    centres = np.asarray([record[1] for record in records], dtype=float)
    spreads = np.asarray([record[2] for record in records], dtype=float)
    return WannierGaussianDensities(
        centres_cartesian_angstrom=centres,
        total_spreads_angstrom2=spreads,
        gaussian_widths_angstrom=np.sqrt(spreads),
    )


def parse_wannier90_hr(path: str | Path) -> WannierHRData:
    """Parse a standard Wannier90 HR text file into normalized matrix blocks."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8", errors="strict") as handle:
        _comment = handle.readline()
        n_wannier = int(handle.readline().strip())
        nrpts = int(handle.readline().strip())
        if n_wannier <= 0 or nrpts <= 0:
            raise ValueError("HR dimensions must be positive")
        degeneracies: list[int] = []
        while len(degeneracies) < nrpts:
            line = handle.readline()
            if not line:
                raise ValueError("truncated HR degeneracy table")
            degeneracies.extend(int(value) for value in line.split())
        if len(degeneracies) != nrpts:
            raise ValueError("HR degeneracy table has extra entries")

        r_vectors = np.empty((nrpts, 3), dtype=np.int64)
        h_r = np.zeros((nrpts, n_wannier, n_wannier), dtype=np.complex128)
        records_per_r = n_wannier * n_wannier
        seen_r: set[tuple[int, int, int]] = set()
        for r_index in range(nrpts):
            reference_r: tuple[int, int, int] | None = None
            seen_matrix = np.zeros((n_wannier, n_wannier), dtype=bool)
            for _ in range(records_per_r):
                line = handle.readline()
                if not line:
                    raise ValueError("truncated HR matrix table")
                fields = line.split()
                if len(fields) != 7:
                    raise ValueError(f"invalid HR record: {line.rstrip()!r}")
                r_value = (int(fields[0]), int(fields[1]), int(fields[2]))
                if reference_r is None:
                    reference_r = r_value
                    r_vectors[r_index] = r_value
                elif r_value != reference_r:
                    raise ValueError("HR R block changed before all matrix entries were read")
                row, column = int(fields[3]) - 1, int(fields[4]) - 1
                if not (0 <= row < n_wannier and 0 <= column < n_wannier):
                    raise ValueError("HR orbital index is out of range")
                if seen_matrix[row, column]:
                    raise ValueError("duplicate matrix element in an HR R block")
                seen_matrix[row, column] = True
                h_r[r_index, row, column] = complex(float(fields[5]), float(fields[6]))
            if reference_r is None or reference_r in seen_r or not np.all(seen_matrix):
                raise ValueError("HR R-block inventory is duplicate or incomplete")
            seen_r.add(reference_r)
        if any(line.strip() for line in handle):
            raise ValueError("unexpected nonempty records after the HR matrix table")
    degeneracy_array = np.asarray(degeneracies, dtype=np.int64)
    h_r /= degeneracy_array[:, None, None]
    r_lookup = {tuple(vector): index for index, vector in enumerate(r_vectors)}
    scale = max(1.0, float(np.max(np.abs(h_r), initial=0.0)))
    for r_value, r_index in r_lookup.items():
        partner = tuple(-component for component in r_value)
        if partner not in r_lookup:
            raise ValueError(f"HR inventory is missing Hermitian partner R={partner}")
        residual = float(
            np.max(
                np.abs(h_r[r_index] - h_r[r_lookup[partner]].conj().T),
                initial=0.0,
            )
        )
        if residual > 2.0e-10 * scale:
            raise ValueError(
                f"HR violates H(R)=H(-R)^dagger at R={r_value}: {residual:.6e}"
            )
    return WannierHRData(
        r_vectors=r_vectors,
        h_r_ev=h_r,
        degeneracies=degeneracy_array,
    )


def regular_fractional_mesh(mesh_shape: tuple[int, int]) -> np.ndarray:
    """Return periodic fractional nodes ordered with the second index fastest."""

    n1, n2 = int(mesh_shape[0]), int(mesh_shape[1])
    if n1 <= 0 or n2 <= 0:
        raise ValueError("mesh_shape entries must be positive")
    first = np.arange(n1, dtype=float) / n1
    second = np.arange(n2, dtype=float) / n2
    return np.asarray(
        [(x, y, 0.0) for x in first for y in second], dtype=float
    )


def evaluate_wannier_hr_on_mesh(
    hr: WannierHRData,
    fractional_kpoints: np.ndarray,
    *,
    phase_chunk_size: int = 256,
) -> np.ndarray:
    """Evaluate exact plain HR Fourier blocks and return ``H[a,b,k]``."""

    kpoints = np.asarray(fractional_kpoints, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3 or not np.all(np.isfinite(kpoints)):
        raise ValueError("fractional_kpoints must be finite with shape (nk,3)")
    if type(phase_chunk_size) is not int or phase_chunk_size <= 0:
        raise TypeError("phase_chunk_size must be an exact positive integer")
    nk = kpoints.shape[0]
    nw = hr.n_wannier
    h_ket = np.empty((nk, nw, nw), dtype=np.complex128)
    for start in range(0, nk, phase_chunk_size):
        stop = min(start + phase_chunk_size, nk)
        phases = np.exp(
            2.0j * np.pi * (kpoints[start:stop] @ hr.r_vectors.T)
        )
        h_ket[start:stop] = np.einsum(
            "kr,rab->kab", phases, hr.h_r_ev, optimize=True
        )
    residual = float(
        np.max(np.abs(h_ket - h_ket.conj().transpose(0, 2, 1)), initial=0.0)
    )
    scale = max(1.0, float(np.max(np.abs(h_ket), initial=0.0)))
    if residual > 2.0e-10 * scale:
        raise ValueError(f"HR Fourier Hamiltonian is not Hermitian: residual={residual:.6e}")
    h_ket = 0.5 * (h_ket + h_ket.conj().transpose(0, 2, 1))
    return np.ascontiguousarray(h_ket.transpose(1, 2, 0))


def build_spin_doubled_h0(scalar_h0_abk: np.ndarray) -> np.ndarray:
    """Duplicate a scalar Hamiltonian into contiguous up/down basis blocks."""

    scalar = np.asarray(scalar_h0_abk, dtype=np.complex128)
    if scalar.ndim != 3 or scalar.shape[0] != scalar.shape[1]:
        raise ValueError("scalar_h0_abk must have shape (n,n,nk)")
    nw, _, nk = scalar.shape
    result = np.zeros((2 * nw, 2 * nw, nk), dtype=np.complex128)
    result[:nw, :nw] = scalar
    result[nw:, nw:] = scalar
    return result


def _validate_parallel_workers(workers: int) -> None:
    if type(workers) is not int or workers <= 0:
        raise TypeError("workers must be an exact positive integer")
    if workers == 1:
        return
    invalid = {
        name: os.environ.get(name)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        if os.environ.get(name) != "1"
    }
    if invalid:
        raise RuntimeError(
            "parallel Keldysh kernel construction requires one-thread BLAS/OpenMP; "
            f"invalid environment: {invalid}"
        )


def _reciprocal_rows(lattice_2d_angstrom: np.ndarray) -> np.ndarray:
    lattice = np.asarray(lattice_2d_angstrom, dtype=float)
    if lattice.shape != (2, 2) or not np.all(np.isfinite(lattice)):
        raise ValueError("lattice_2d_angstrom must be finite with shape (2,2)")
    area = float(abs(np.linalg.det(lattice)))
    if area <= 0.0:
        raise ValueError("lattice area must be positive")
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    dot = float(np.dot(reciprocal[0], reciprocal[1]))
    scale = float(np.linalg.norm(reciprocal[0]) * np.linalg.norm(reciprocal[1]))
    if abs(dot) > 2.0e-12 * scale:
        raise ValueError("v1 singular-cell quadrature requires an orthogonal 2D lattice")
    return reciprocal


def _centered_mesh_coefficients(size: int) -> np.ndarray:
    indices = np.arange(size, dtype=np.int64)
    return np.where(indices <= size // 2, indices, indices - size)


def _orbital_lambda(
    q_vectors: np.ndarray,
    centres_xy: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    q = np.asarray(q_vectors, dtype=float)
    q2 = np.einsum("gd,gd->g", q, q, optimize=True)
    phase = np.exp(-1.0j * (q @ centres_xy.T))
    envelope = np.exp(-0.25 * q2[:, None] * widths[None, :] ** 2)
    return phase * envelope


def _weighted_orbital_gram(
    q_vectors: np.ndarray,
    scalar_weights: np.ndarray,
    centres_xy: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(scalar_weights, dtype=float)
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("Gram weights must be finite and nonnegative")
    lambdas = _orbital_lambda(q_vectors, centres_xy, widths)
    weighted = np.sqrt(weights)[:, None] * lambdas
    # With the archived HR convention H(k)=sum_R H(R) exp(+i k.R),
    # U_ab(q)=sum_R V_ab(R) exp(+i q.R) carries the centre phase
    # exp(+i Q.(tau_a-tau_b)) = lambda_a(Q)^* lambda_b(Q).
    return np.asarray(weighted.conj().T @ weighted, dtype=np.complex128)


def _zero_cell_gaussian_keldysh_matrix(
    *,
    dx: float,
    dy: float,
    cell_area_angstrom2: float,
    centres_xy: np.ndarray,
    widths: np.ndarray,
    alpha_2d_angstrom: float,
    kappa: float,
    angular_order: int,
    radial_order: int,
    coulomb_ev_angstrom: float,
) -> np.ndarray:
    """Average the integrable ``q=0`` cell with orbital form factors."""

    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("reciprocal cell widths must be positive")
    if angular_order < 8 or radial_order < 4:
        raise ValueError("singular-cell quadrature orders are too small")
    theta_nodes, theta_weights = leggauss(int(angular_order))
    radial_nodes, radial_weights = leggauss(int(radial_order))
    half_x, half_y = 0.5 * dx, 0.5 * dy
    corner_angle = math.atan2(half_y, half_x)
    # The radial boundary switches between an x edge and a y edge at each
    # corner ray.  Splitting there avoids applying one high-order rule across
    # the resulting derivative discontinuity in r_max(theta).
    angular_breaks = np.asarray(
        sorted(
            {
                0.0,
                corner_angle,
                0.5 * np.pi,
                np.pi - corner_angle,
                np.pi,
                np.pi + corner_angle,
                1.5 * np.pi,
                2.0 * np.pi - corner_angle,
                2.0 * np.pi,
            }
        ),
        dtype=float,
    )
    vectors: list[tuple[float, float]] = []
    weights_all: list[float] = []
    prefactor = 2.0 * np.pi * float(coulomb_ev_angstrom) / float(kappa)
    normalization = float(cell_area_angstrom2 * dx * dy)
    for angular_start, angular_stop in zip(
        angular_breaks[:-1], angular_breaks[1:], strict=True
    ):
        theta = 0.5 * (angular_stop - angular_start) * theta_nodes
        theta += 0.5 * (angular_stop + angular_start)
        theta_weight = 0.5 * (angular_stop - angular_start) * theta_weights
        for angle, weight_theta in zip(theta, theta_weight, strict=True):
            cosine, sine = math.cos(float(angle)), math.sin(float(angle))
            x_limit = math.inf if abs(cosine) < 1.0e-15 else half_x / abs(cosine)
            y_limit = math.inf if abs(sine) < 1.0e-15 else half_y / abs(sine)
            maximum_radius = min(x_limit, y_limit)
            radii = 0.5 * maximum_radius * (radial_nodes + 1.0)
            radial_quadrature = 0.5 * maximum_radius * radial_weights
            for radius, weight_radius in zip(radii, radial_quadrature, strict=True):
                vectors.append((float(radius * cosine), float(radius * sine)))
                weights_all.append(
                    float(
                        weight_theta
                        * weight_radius
                        * prefactor
                        / (1.0 + alpha_2d_angstrom * radius)
                        / normalization
                    )
                )
    matrix = _weighted_orbital_gram(
        np.asarray(vectors, dtype=float),
        np.asarray(weights_all, dtype=float),
        centres_xy,
        widths,
    )
    return 0.5 * (matrix + matrix.conj().T)


def rectangular_gaussian_keldysh_zero_cell(
    *,
    lattice_2d_angstrom: np.ndarray,
    wannier: WannierGaussianDensities,
    mesh_shape: tuple[int, int],
    alpha_2d_angstrom: float = TPT_PUBLISHED_ALPHA_2D_ANGSTROM,
    kappa: float = 1.0,
    angular_order: int = 64,
    radial_order: int = 24,
    coulomb_ev_angstrom: float = COULOMB_EV_ANGSTROM,
) -> np.ndarray:
    """Return the orbital-resolved average over the exact rectangular q=0 cell."""

    if abs(float(kappa) - 1.0) > 1.0e-14:
        raise ValueError("TPT v1 zero-cell authority requires kappa=1")
    n1, n2 = int(mesh_shape[0]), int(mesh_shape[1])
    if n1 <= 1 or n2 <= 1:
        raise ValueError("mesh_shape entries must exceed one")
    lattice = np.asarray(lattice_2d_angstrom, dtype=float)
    reciprocal = _reciprocal_rows(lattice)
    return _zero_cell_gaussian_keldysh_matrix(
        dx=float(np.linalg.norm(reciprocal[0]) / n1),
        dy=float(np.linalg.norm(reciprocal[1]) / n2),
        cell_area_angstrom2=float(abs(np.linalg.det(lattice))),
        centres_xy=np.asarray(wannier.centres_cartesian_angstrom[:, :2], dtype=float),
        widths=np.asarray(wannier.gaussian_widths_angstrom, dtype=float),
        alpha_2d_angstrom=float(alpha_2d_angstrom),
        kappa=float(kappa),
        angular_order=int(angular_order),
        radial_order=int(radial_order),
        coulomb_ev_angstrom=float(coulomb_ev_angstrom),
    )


def build_gaussian_keldysh_density_density_kernel(
    *,
    lattice_2d_angstrom: np.ndarray,
    wannier: WannierGaussianDensities,
    mesh_shape: tuple[int, int],
    alpha_2d_angstrom: float = TPT_PUBLISHED_ALPHA_2D_ANGSTROM,
    kappa: float = 1.0,
    spin_blocks: int = 2,
    gaussian_tail_tolerance: float = 1.0e-11,
    zero_cell_angular_order: int = 64,
    zero_cell_radial_order: int = 24,
    shell_covariance_tolerance_ev: float = 1.0e-8,
    workers: int = 1,
    coulomb_ev_angstrom: float = COULOMB_EV_ANGSTROM,
) -> KeldyshKernelBuild:
    """Build the local-field Gaussian Keldysh kernel for periodic Q=0 HF."""

    _validate_parallel_workers(workers)
    n1, n2 = int(mesh_shape[0]), int(mesh_shape[1])
    if n1 <= 1 or n2 <= 1:
        raise ValueError("mesh_shape entries must exceed one")
    alpha = float(alpha_2d_angstrom)
    dielectric = float(kappa)
    tail = float(gaussian_tail_tolerance)
    if alpha <= 0.0 or dielectric <= 0.0:
        raise ValueError("alpha_2d_angstrom and kappa must be positive")
    if abs(dielectric - 1.0) > 1.0e-14:
        raise ValueError(
            "TPT v1 is source-authorized only for the paper's free-standing "
            "interaction (kappa=1); an environmental rescaling needs a new contract"
        )
    covariance_tolerance = float(shell_covariance_tolerance_ev)
    if not np.isfinite(covariance_tolerance) or covariance_tolerance <= 0.0:
        raise ValueError("shell_covariance_tolerance_ev must be finite and positive")
    if not (0.0 < tail < 1.0):
        raise ValueError("gaussian_tail_tolerance must lie in (0,1)")
    if type(spin_blocks) is not int or spin_blocks <= 0:
        raise TypeError("spin_blocks must be an exact positive integer")

    lattice = np.asarray(lattice_2d_angstrom, dtype=float)
    reciprocal = _reciprocal_rows(lattice)
    cell_area = float(abs(np.linalg.det(lattice)))
    centres_xy = np.asarray(wannier.centres_cartesian_angstrom[:, :2], dtype=float)
    widths = np.asarray(wannier.gaussian_widths_angstrom, dtype=float)
    minimum_width = float(np.min(widths))
    q_cutoff = math.sqrt(2.0 * math.log(1.0 / tail)) / minimum_width
    q_bz_radius = 0.5 * float(
        np.linalg.norm(reciprocal[0]) + np.linalg.norm(reciprocal[1])
    )
    shell_bounds = [
        int(math.ceil((q_cutoff + q_bz_radius) / np.linalg.norm(vector))) + 1
        for vector in reciprocal
    ]
    shell_indices = np.asarray(
        [
            (first, second)
            for first in range(-shell_bounds[0], shell_bounds[0] + 1)
            for second in range(-shell_bounds[1], shell_bounds[1] + 1)
        ],
        dtype=np.int64,
    )
    shell_vectors = shell_indices @ reciprocal

    first_coefficients = _centered_mesh_coefficients(n1)
    second_coefficients = _centered_mesh_coefficients(n2)
    transfers = [
        first * reciprocal[0] / n1 + second * reciprocal[1] / n2
        for first in first_coefficients
        for second in second_coefficients
    ]
    dx = float(np.linalg.norm(reciprocal[0]) / n1)
    dy = float(np.linalg.norm(reciprocal[1]) / n2)
    zero_cell = _zero_cell_gaussian_keldysh_matrix(
        dx=dx,
        dy=dy,
        cell_area_angstrom2=cell_area,
        centres_xy=centres_xy,
        widths=widths,
        alpha_2d_angstrom=alpha,
        kappa=dielectric,
        angular_order=int(zero_cell_angular_order),
        radial_order=int(zero_cell_radial_order),
        coulomb_ev_angstrom=float(coulomb_ev_angstrom),
    )
    prefactor = 2.0 * np.pi * float(coulomb_ev_angstrom) / dielectric / cell_area

    def build_transfer(item: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
        flat_index, transfer = item
        q_vectors = shell_vectors + transfer[None, :]
        norms = np.linalg.norm(q_vectors, axis=1)
        nonzero = norms > 1.0e-13
        q_selected = q_vectors[nonzero]
        norms_selected = norms[nonzero]
        scalar = prefactor / (
            norms_selected * (1.0 + alpha * norms_selected)
        )
        matrix = _weighted_orbital_gram(
            q_selected, scalar, centres_xy, widths
        )
        if flat_index == 0:
            matrix += zero_cell
        return flat_index, 0.5 * (matrix + matrix.conj().T)

    orbital_count = wannier.n_orbital
    kernel_flat = np.empty(
        (n1 * n2, orbital_count, orbital_count), dtype=np.complex128
    )
    indexed_transfers = list(enumerate(transfers))
    if workers == 1:
        iterator: Iterable[tuple[int, np.ndarray]] = map(
            build_transfer, indexed_transfers
        )
        for flat_index, matrix in iterator:
            kernel_flat[flat_index] = matrix
    else:
        with ThreadPoolExecutor(max_workers=min(workers, n1 * n2)) as executor:
            for flat_index, matrix in executor.map(build_transfer, indexed_transfers):
                kernel_flat[flat_index] = matrix

    kernel_q = kernel_flat.reshape(n1, n2, orbital_count, orbital_count).transpose(
        2, 3, 0, 1
    )
    inverse_i = (-np.arange(n1)) % n1
    inverse_j = (-np.arange(n2)) % n2
    inverse = kernel_q[:, :, inverse_i][:, :, :, inverse_j]
    raw_inversion_residual = float(
        np.max(np.abs(kernel_q - inverse.conj()), initial=0.0)
    )
    raw_orbital_hermiticity_residual = float(
        np.max(
            np.abs(kernel_q - kernel_q.conj().swapaxes(0, 1)), initial=0.0
        )
    )
    if max(raw_inversion_residual, raw_orbital_hermiticity_residual) > covariance_tolerance:
        raise ValueError(
            "reciprocal shell failed pre-symmetrization covariance gate: "
            f"inversion={raw_inversion_residual:.6e} eV, "
            f"orbital_hermiticity={raw_orbital_hermiticity_residual:.6e} eV"
        )
    kernel_q = 0.5 * (kernel_q + inverse.conj())
    kernel_q = 0.5 * (kernel_q + kernel_q.conj().swapaxes(0, 1))

    nonzero_shell = np.any(shell_indices != 0, axis=1)
    g_vectors = shell_vectors[nonzero_shell]
    g_norms = np.linalg.norm(g_vectors, axis=1)
    hartree_scalar = prefactor / (g_norms * (1.0 + alpha * g_norms))
    hartree_complex = _weighted_orbital_gram(
        g_vectors, hartree_scalar, centres_xy, widths
    )
    hartree = 0.5 * (hartree_complex.real + hartree_complex.real.T)

    periodic_kernel = PeriodicDensityDensityKernel(
        mesh_shape=(n1, n2),
        fock_kernel_q=kernel_q,
        hartree_kernel=hartree,
        spin_blocks=spin_blocks,
        require_neutral_density=True,
        neutrality_tolerance=1.0e-10,
    )
    maximum_shell_norm = float(np.max(np.linalg.norm(shell_vectors, axis=1)))
    metadata: dict[str, int | float | str] = {
        "schema": "tpt-gaussian-keldysh-kernel-v1",
        "alpha_2d_angstrom": alpha,
        "kappa": dielectric,
        "coulomb_ev_angstrom": float(coulomb_ev_angstrom),
        "cell_area_angstrom2": cell_area,
        "mesh_n1": n1,
        "mesh_n2": n2,
        "n_spatial_orbital": orbital_count,
        "spin_blocks": spin_blocks,
        "gaussian_width_convention": "ell=sqrt(final_total_wannier_spread)",
        "gaussian_tail_tolerance": tail,
        "q_cutoff_estimate_angstrom_inv": q_cutoff,
        "shell_bound_first": shell_bounds[0],
        "shell_bound_second": shell_bounds[1],
        "shell_vector_count": int(shell_indices.shape[0]),
        "maximum_shell_norm_angstrom_inv": maximum_shell_norm,
        "zero_cell_angular_order": int(zero_cell_angular_order),
        "zero_cell_radial_order": int(zero_cell_radial_order),
        "shell_covariance_tolerance_ev": covariance_tolerance,
        "raw_inversion_residual_ev": raw_inversion_residual,
        "raw_orbital_hermiticity_residual_ev": raw_orbital_hermiticity_residual,
        "workers": workers,
        "hartree_macroscopic_g0": "removed_neutral_background",
        "fock_g0": "rectangular_cell_polar_gauss_quadrature",
    }
    return KeldyshKernelBuild(
        kernel=periodic_kernel,
        reciprocal_vectors_angstrom_inv=np.ascontiguousarray(reciprocal),
        reciprocal_shell_indices=np.ascontiguousarray(shell_indices),
        zero_cell_matrix_ev=np.ascontiguousarray(zero_cell),
        metadata=metadata,
    )


__all__ = [
    "COULOMB_EV_ANGSTROM",
    "KeldyshKernelBuild",
    "TPT_PUBLISHED_ALPHA_2D_ANGSTROM",
    "WannierGaussianDensities",
    "WannierHRData",
    "build_gaussian_keldysh_density_density_kernel",
    "build_spin_doubled_h0",
    "evaluate_wannier_hr_on_mesh",
    "parse_final_wannier_gaussian_densities",
    "parse_wannier90_hr",
    "rectangular_gaussian_keldysh_zero_cell",
    "regular_fractional_mesh",
]
