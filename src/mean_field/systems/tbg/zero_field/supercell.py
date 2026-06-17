from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.linalg import eigh

from ....core.lattice import KPath, LatticeGrid
from ....core.supercell import (
    IntegerSupercell,
    fixed_sector_occupation_counts,
    folded_band_count,
    primitive_filling_from_occupation_counts,
)
from ....core.hf import (
    density_from_fixed_sector_occupations as _core_density_from_fixed_sector_occupations,
    sector_block_energies,
    shift_wavefunction_grid,
)
from ..params import TBGParameters
from .hf import coulomb_unit, screened_coulomb
from .model import _construct_diagonal_block, build_sigma_z_from_uk
from .path import build_kpath_from_nodes


GRAPHENE_LATTICE_A_ANGSTROM = 2.46


@dataclass(frozen=True)
class MoireSupercell(IntegerSupercell):
    """TBG-facing wrapper around the generic integer supercell convention."""

    def reciprocal_vectors(self, params: TBGParameters) -> tuple[complex, complex]:
        return super().reciprocal_vectors(params.g1, params.g2)


def zhang_sqrt3_tripled_supercell() -> MoireSupercell:
    """The ``sqrt(3) x sqrt(3)`` tripled cell used for nu=8/3 in Zhang Fig. 10."""

    return MoireSupercell(n11=1, n12=1, n21=-1, n22=2)


@dataclass(frozen=True)
class SupercellBMSolution:
    params: TBGParameters
    supercell: MoireSupercell
    lattice_kvec: np.ndarray
    lg: int
    nlocal: int
    n_eta: int
    n_spin: int
    nb: int
    hamiltonian: np.ndarray
    sigma_z: np.ndarray
    uk: np.ndarray
    spectrum: np.ndarray
    gvec: np.ndarray
    super_g1: complex
    super_g2: complex
    boundary_mode: str = "zero_fill"

    @property
    def nk(self) -> int:
        return int(self.lattice_kvec.size)

    @property
    def nt(self) -> int:
        return int(self.n_spin * self.n_eta * self.nb)

    @property
    def basis_dimension(self) -> int:
        return int(self.nlocal * self.lg * self.lg)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (int(self.lg), int(self.lg))

    def flattened_energies(self) -> np.ndarray:
        data = np.zeros((self.nt, self.nk), dtype=float)
        row = 0
        for ib in range(self.nb):
            for ieta in range(self.n_eta):
                for ispin in range(self.n_spin):
                    data[row, :] = self.spectrum[ib, ieta, :]
                    row += 1
        return data


def build_supercell_uniform_lattice(
    params: TBGParameters,
    supercell: MoireSupercell,
    mesh: int,
    *,
    endpoint: bool = False,
) -> LatticeGrid:
    """Uniform mesh in the supercell Brillouin zone.

    ``endpoint=False`` gives an exact ``mesh x mesh`` Monkhorst-style tile,
    matching the wording of Zhang's 12x12 tripled-supercell k mesh.
    """

    mesh = int(mesh)
    if mesh <= 0:
        raise ValueError(f"mesh must be positive, got {mesh}")
    super_g1, super_g2 = supercell.reciprocal_vectors(params)
    if endpoint:
        frac = np.arange(mesh + 1, dtype=float) / float(mesh)
        lk = mesh
    else:
        frac = np.arange(mesh, dtype=float) / float(mesh)
        lk = mesh - 1
    kvec = np.ravel(frac[:, None] * super_g1 + frac[None, :] * super_g2, order="F")
    return LatticeGrid(
        k1=frac.copy(),
        k2=frac.copy(),
        kvec=np.asarray(kvec, dtype=np.complex128),
        nk=int(kvec.size),
        lk=int(lk),
        flag_inv=bool(endpoint),
    )


def build_supercell_gamma_m_k_gamma_kprime_path(
    params: TBGParameters,
    supercell: MoireSupercell,
    points_per_segment: int,
) -> KPath:
    super_g1, super_g2 = supercell.reciprocal_vectors(params)
    gamma = 0.0 + 0.0j
    m_point = (super_g1 + super_g2) / 2.0
    k_point = (2.0 * super_g1 + super_g2) / 3.0
    kprime_point = (super_g1 + 2.0 * super_g2) / 3.0
    return build_kpath_from_nodes(
        [gamma, m_point, k_point, gamma, kprime_point],
        ("Gamma_s", "M_s", "K_s", "Gamma_s", "Kprime_s"),
        int(points_per_segment),
    )


@dataclass(frozen=True)
class SupercellSCFGridPathSamples:
    """Exact SCF-grid samples lying on the Zhang Fig. 10 high-symmetry path."""

    kdist: np.ndarray
    grid_indices: np.ndarray
    frac_coords: np.ndarray
    segment_indices: np.ndarray
    node_kdist: np.ndarray
    labels: tuple[str, ...]
    exact_node_hit_mask: np.ndarray
    exact_tolerance: float

    @property
    def unique_grid_count(self) -> int:
        return int(np.unique(self.grid_indices).size)

    @property
    def exact_node_hit_count(self) -> int:
        return int(np.count_nonzero(self.exact_node_hit_mask))

    @property
    def segment_counts(self) -> tuple[int, ...]:
        n_segments = max(len(self.labels) - 1, 0)
        counts = np.bincount(self.segment_indices.astype(int), minlength=n_segments)
        return tuple(int(value) for value in counts[:n_segments])


def extract_supercell_gamma_m_k_gamma_kprime_scf_grid_path(
    grid: LatticeGrid,
    *,
    super_g1: complex,
    super_g2: complex,
    exact_tolerance: float = 1.0e-10,
) -> SupercellSCFGridPathSamples:
    """Extract exact SCF-grid points on ``Gamma_s-M_s-K_s-Gamma_s-K'_s``.

    This implements the benchmark policy that SCF-grid diagnostics should use
    points from the converged SCF mesh itself rather than a dense post-SCF path
    reconstruction.  Adjacent segment endpoints are deliberately kept as
    duplicate rows so that plotting can draw each segment independently and make
    missing node hits visible.
    """

    k1 = np.asarray(grid.k1, dtype=float)
    k2 = np.asarray(grid.k2, dtype=float)
    if k1.ndim != 1 or k2.ndim != 1 or int(grid.kvec.size) != int(k1.size * k2.size):
        raise ValueError(
            f"Expected tensor-product supercell grid with len(k1)*len(k2)=nk, "
            f"got {k1.shape}, {k2.shape}, nk={grid.kvec.size}"
        )

    labels = ("Gamma_s", "M_s", "K_s", "Gamma_s", "Kprime_s")
    node_frac = np.asarray(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [2.0 / 3.0, 1.0 / 3.0],
            [0.0, 0.0],
            [1.0 / 3.0, 2.0 / 3.0],
        ],
        dtype=float,
    )
    node_kvec = node_frac[:, 0] * complex(super_g1) + node_frac[:, 1] * complex(super_g2)
    node_kdist = np.zeros(node_frac.shape[0], dtype=float)
    for inode in range(1, node_frac.shape[0]):
        node_kdist[inode] = node_kdist[inode - 1] + float(abs(node_kvec[inode] - node_kvec[inode - 1]))

    tol = float(exact_tolerance)
    selected: list[tuple[float, int, float, float, int]] = []
    for iseg in range(node_frac.shape[0] - 1):
        start = node_frac[iseg]
        end = node_frac[iseg + 1]
        segment = end - start
        segment_norm2 = float(np.dot(segment, segment))
        if segment_norm2 <= 0.0:
            continue
        segment_length = float(abs(node_kvec[iseg + 1] - node_kvec[iseg]))
        segment_rows: list[tuple[float, int, float, float, int]] = []
        for iy, fy in enumerate(k2):
            for ix, fx in enumerate(k1):
                point = np.asarray([float(fx), float(fy)], dtype=float)
                diff = point - start
                cross = float(diff[0] * segment[1] - diff[1] * segment[0])
                t = float(np.dot(diff, segment) / segment_norm2)
                if abs(cross) > tol or t < -tol or t > 1.0 + tol:
                    continue
                projection = start + min(1.0, max(0.0, t)) * segment
                if float(np.linalg.norm(point - projection)) > tol:
                    continue
                index = int(ix + k1.size * iy)
                kdist = float(node_kdist[iseg] + min(1.0, max(0.0, t)) * segment_length)
                segment_rows.append((kdist, index, float(fx), float(fy), int(iseg)))
        segment_rows.sort(key=lambda row: (row[0], row[1]))
        selected.extend(segment_rows)

    if not selected:
        raise ValueError("No exact SCF-grid points were found on the requested supercell path")

    kvec_grid = np.asarray(grid.kvec, dtype=np.complex128)
    node_tol = max(abs(complex(super_g1)), abs(complex(super_g2)), 1.0) * tol
    node_hits = np.asarray([np.min(np.abs(kvec_grid - complex(node))) <= node_tol for node in node_kvec], dtype=bool)

    return SupercellSCFGridPathSamples(
        kdist=np.asarray([row[0] for row in selected], dtype=float),
        grid_indices=np.asarray([row[1] for row in selected], dtype=int),
        frac_coords=np.asarray([[row[2], row[3]] for row in selected], dtype=float),
        segment_indices=np.asarray([row[4] for row in selected], dtype=int),
        node_kdist=np.asarray(node_kdist, dtype=float),
        labels=labels,
        exact_node_hit_mask=node_hits,
        exact_tolerance=float(exact_tolerance),
    )


def _supercell_gvecs(super_g1: complex, super_g2: complex, lg: int) -> np.ndarray:
    if lg <= 0 or lg % 2 == 0:
        raise ValueError(f"Expected positive odd lg, got {lg}")
    coords = np.arange(-(lg // 2), lg // 2 + 1, dtype=int)
    return np.ravel(coords[:, None] * super_g1 + coords[None, :] * super_g2, order="F").astype(np.complex128)


def _zero_fill_tunnel(params: TBGParameters, supercell: MoireSupercell, lg: int, zeta: int) -> np.ndarray:
    dim = 4 * lg * lg
    t12 = np.zeros((dim, dim), dtype=np.complex128)

    if zeta == 1:
        t0, t1, t2 = params.t0, params.t1, params.t2
    elif zeta == -1:
        t0, t1, t2 = params.t0, params.t2, params.t1
    else:
        raise ValueError(f"Unexpected valley label: {zeta}")

    primitive_neighbors = (
        (zeta, -zeta, t2),
        (0, -zeta, t1),
        (zeta, 0, t0),
    )

    def flat(ix: int, iy: int) -> int:
        return int(ix) + int(lg) * int(iy)

    def in_bounds(ix: int, iy: int) -> bool:
        return 0 <= int(ix) < int(lg) and 0 <= int(iy) < int(lg)

    for iy in range(lg):
        for ix in range(lg):
            here = flat(ix, iy)
            left = 4 * here
            for pdx, pdy, tunnel in primitive_neighbors:
                sdx, sdy = supercell.primitive_shift_to_supercell(int(pdx), int(pdy))
                nx = ix + sdx
                ny = iy + sdy
                if not in_bounds(nx, ny):
                    continue
                right = 4 * flat(nx, ny)
                t12[left + 2 : left + 4, right : right + 2] = tunnel
                t12[right : right + 2, left + 2 : left + 4] = tunnel
    return t12


def solve_supercell_bm_model(
    params: TBGParameters,
    lattice_kvec: np.ndarray,
    *,
    supercell: MoireSupercell | None = None,
    lg: int = 11,
    sigma_rotation: bool = True,
    calculate_chern_operator: bool = False,
) -> SupercellBMSolution:
    """Solve the BM model in an enlarged moire reciprocal basis.

    The number of retained active bands per spin/valley is ``2 * det(supercell)``:
    two primitive flat bands folded into the enlarged-cell Brillouin zone.
    """

    supercell = zhang_sqrt3_tripled_supercell() if supercell is None else supercell
    n_eta, n_spin, nlocal = 2, 2, 4
    nb = folded_band_count(2, supercell.area_ratio)
    nk = int(np.asarray(lattice_kvec).size)
    dim = nlocal * int(lg) * int(lg)
    if nb <= 0 or nb > dim:
        raise ValueError(f"Invalid number of active bands nb={nb} for basis dimension {dim}")
    super_g1, super_g2 = supercell.reciprocal_vectors(params)
    gvec = _supercell_gvecs(super_g1, super_g2, int(lg))

    hamiltonian = np.zeros((dim, dim, n_eta, nk), dtype=np.complex128)
    spectrum = np.zeros((nb, n_eta, nk), dtype=float)
    uk = np.zeros((dim, nb, n_eta, nk), dtype=np.complex128)
    sigma_z = np.zeros((n_spin * n_eta * nb, n_spin * n_eta * nb, nk), dtype=np.complex128)

    tunnel = {1: _zero_fill_tunnel(params, supercell, int(lg), 1), -1: _zero_fill_tunnel(params, supercell, int(lg), -1)}
    start = dim // 2 - nb // 2
    stop = start + nb - 1

    for ieta, zeta in enumerate((1, -1)):
        valley_tunnel = tunnel[zeta]
        for ik, kval in enumerate(np.asarray(lattice_kvec, dtype=np.complex128)):
            h0 = _construct_diagonal_block(params, gvec, int(lg), complex(kval), zeta, sigma_rotation)
            h = h0 + valley_tunnel - params.chemical_potential * np.eye(dim, dtype=np.complex128)
            hamiltonian[:, :, ieta, ik] = h
            evals, evecs = eigh(h, subset_by_index=[start, stop], driver="evr")
            spectrum[:, ieta, ik] = evals
            uk[:, :, ieta, ik] = evecs

    if calculate_chern_operator:
        sigma_z[:, :, :] = build_sigma_z_from_uk(uk, lg=int(lg), n_spin=n_spin)

    return SupercellBMSolution(
        params=params,
        supercell=supercell,
        lattice_kvec=np.asarray(lattice_kvec, dtype=np.complex128),
        lg=int(lg),
        nlocal=nlocal,
        n_eta=n_eta,
        n_spin=n_spin,
        nb=nb,
        hamiltonian=hamiltonian,
        sigma_z=sigma_z,
        uk=uk,
        spectrum=spectrum,
        gvec=gvec,
        super_g1=complex(super_g1),
        super_g2=complex(super_g2),
        boundary_mode="zero_fill",
    )


def reciprocal_shift_labels(lg: int) -> tuple[int, ...]:
    if lg <= 0 or lg % 2 == 0:
        raise ValueError(f"Expected positive odd lg, got {lg}")
    half_width = (int(lg) - 1) // 2
    return tuple(range(-half_width, half_width + 1))


def supercell_interaction_shifts(
    supercell_solution: SupercellBMSolution,
    interaction_lg: int,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    labels = reciprocal_shift_labels(int(interaction_lg))
    shifts = tuple((m, n) for n in labels for m in labels)
    gvecs = np.asarray(
        [m * supercell_solution.super_g1 + n * supercell_solution.super_g2 for m, n in shifts],
        dtype=np.complex128,
    )
    return shifts, gvecs


def screening_lm_from_ds_angstrom(ds_angstrom: float) -> float:
    """Return the code's ``lm`` so tanh uses ``|q| * ds/a``."""

    return float(ds_angstrom) / GRAPHENE_LATTICE_A_ANGSTROM / 2.0


def supercell_coulomb_unit(params: TBGParameters, supercell: MoireSupercell) -> float:
    """Coulomb energy unit for the enlarged real-space supercell area."""

    return coulomb_unit(params) / float(supercell.area_ratio)


def screened_coulomb_matrix(
    qvals: np.ndarray,
    lm: float,
    *,
    relative_permittivity: float,
    finite_zero_limit: bool = True,
    zero_cutoff: float = 1e-6,
) -> np.ndarray:
    q_abs = np.abs(np.asarray(qvals, dtype=np.complex128))
    values = np.zeros_like(q_abs, dtype=float)
    if finite_zero_limit:
        values[q_abs < zero_cutoff] = 2.0 * np.pi * 2.0 * float(lm) / float(relative_permittivity)
    mask = q_abs >= zero_cutoff
    if np.any(mask):
        values[mask] = 2.0 * np.pi / (float(relative_permittivity) * q_abs[mask]) * np.tanh(q_abs[mask] * 2.0 * float(lm))
    return values


def _shift_wavefunction_grid(values: np.ndarray, dm: int, dn: int) -> np.ndarray:
    return shift_wavefunction_grid(values, dm, dn, boundary_mode="zero_fill", grid_axes=(1, 2))


def compact_overlap_between(
    target: SupercellBMSolution,
    source: SupercellBMSolution,
    shift: tuple[int, int],
    *,
    valley_index: int,
) -> np.ndarray:
    if target.lg != source.lg or target.nlocal != source.nlocal or target.nb != source.nb:
        raise ValueError("target/source supercell bases must have matching lg, nlocal, and nb")
    if target.supercell != source.supercell:
        raise ValueError("target/source supercell conventions differ")
    if valley_index < 0 or valley_index >= target.n_eta:
        raise ValueError(f"valley_index={valley_index} outside [0, {target.n_eta})")

    nb = int(target.nb)
    nx = ny = int(target.lg)
    target_cols = nb * int(target.nk)
    source_cols = nb * int(source.nk)
    ul = target.uk[:, :, valley_index, :].reshape(target.basis_dimension, target_cols, order="F")
    ur_grid = source.uk[:, :, valley_index, :].reshape(source.nlocal, nx, ny, source_cols, order="F")
    shifted = _shift_wavefunction_grid(ur_grid, -int(shift[0]), -int(shift[1])).reshape(
        source.basis_dimension,
        source_cols,
        order="F",
    )
    return ul.conj().T @ shifted


def compact_diagonal_overlap(
    solution: SupercellBMSolution,
    shift: tuple[int, int],
    *,
    valley_index: int,
) -> np.ndarray:
    if valley_index < 0 or valley_index >= solution.n_eta:
        raise ValueError(f"valley_index={valley_index} outside [0, {solution.n_eta})")
    nb = int(solution.nb)
    nx = ny = int(solution.lg)
    w_grid = solution.uk[:, :, valley_index, :].reshape(solution.nlocal, nx, ny, nb, solution.nk, order="F")
    shifted = _shift_wavefunction_grid(w_grid, -int(shift[0]), -int(shift[1]))
    return np.einsum("lxyak,lxybk->abk", np.conj(w_grid), shifted, optimize=True)


def precompute_diagonal_overlaps(
    solution: SupercellBMSolution,
    shifts: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for shift in shifts:
        out[tuple(shift)] = np.asarray(
            [compact_diagonal_overlap(solution, tuple(shift), valley_index=ieta) for ieta in range(solution.n_eta)],
            dtype=np.complex128,
        )
    return out


def zero_reference_density_blocks(solution: SupercellBMSolution) -> np.ndarray:
    density = np.zeros((solution.n_spin, solution.n_eta, solution.nb, solution.nb, solution.nk), dtype=np.complex128)
    eye = np.eye(solution.nb, dtype=np.complex128)
    density[:, :, :, :, :] = -0.5 * eye[None, None, :, :, None]
    return density


def diagonal_h0_blocks(solution: SupercellBMSolution) -> np.ndarray:
    h0 = np.zeros((solution.n_spin, solution.n_eta, solution.nb, solution.nb, solution.nk), dtype=np.complex128)
    for ispin in range(solution.n_spin):
        for ieta in range(solution.n_eta):
            for ik in range(solution.nk):
                np.fill_diagonal(h0[ispin, ieta, :, :, ik], solution.spectrum[:, ieta, ik])
    return h0


def blocks_to_full(blocks: np.ndarray) -> np.ndarray:
    arr = np.asarray(blocks, dtype=np.complex128)
    if arr.ndim != 5:
        raise ValueError(f"Expected blocks shape (spin, valley, band, band, k), got {arr.shape}")
    n_spin, n_eta, nb, nb_rhs, nk = arr.shape
    if nb != nb_rhs:
        raise ValueError(f"Expected square band blocks, got {arr.shape}")
    nt = n_spin * n_eta * nb
    full = np.zeros((nt, nt, nk), dtype=np.complex128)
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, nb), order="F")
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            inds = idx[ispin, ieta, :]
            full[np.ix_(inds, inds, np.arange(nk))] = arr[ispin, ieta]
    return full


def density_trace_for_shift(density_blocks: np.ndarray, diagonal_by_valley: np.ndarray) -> complex:
    density = np.asarray(density_blocks, dtype=np.complex128)
    diagonal = np.asarray(diagonal_by_valley, dtype=np.complex128)
    if diagonal.shape != (density.shape[1], density.shape[2], density.shape[3], density.shape[4]):
        raise ValueError(f"Diagonal overlap shape {diagonal.shape} incompatible with density {density.shape}")
    total = 0.0 + 0.0j
    for ispin in range(density.shape[0]):
        for ieta in range(density.shape[1]):
            total += np.einsum("abk,bak->", density[ispin, ieta], np.conj(diagonal[ieta]), optimize=True)
    return complex(total)


def build_hartree_blocks_from_diagonals(
    density_source_blocks: np.ndarray,
    *,
    source_diagonals: dict[tuple[int, int], np.ndarray],
    target_diagonals: dict[tuple[int, int], np.ndarray],
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    target_nk: int,
    v0: float,
    screening_lm: float,
    relative_permittivity: float,
    beta: float = 1.0,
    finite_zero_limit: bool = True,
) -> np.ndarray:
    density = np.asarray(density_source_blocks, dtype=np.complex128)
    n_spin, n_eta, nb, _nb_rhs, source_nk = density.shape
    hartree = np.zeros((n_spin, n_eta, nb, nb, int(target_nk)), dtype=np.complex128)
    scale = float(beta) * float(v0) / float(source_nk)
    for shift, gvec in zip(shifts, gvecs, strict=True):
        shift = tuple(shift)
        source_diag = source_diagonals[shift]
        target_diag = target_diagonals[shift]
        trace = density_trace_for_shift(density, source_diag)
        kernel = screened_coulomb(
            complex(gvec),
            float(screening_lm),
            relative_permittivity=float(relative_permittivity),
            finite_zero_limit=bool(finite_zero_limit),
        )
        coeff = scale * float(kernel) * trace
        if coeff == 0.0:
            continue
        for ispin in range(n_spin):
            for ieta in range(n_eta):
                hartree[ispin, ieta] += coeff * target_diag[ieta]
    return hartree


def _contract_block_fock(lambda_compact: np.ndarray, density: np.ndarray, coeff_matrix: np.ndarray) -> np.ndarray:
    nb, _, nk_source = density.shape
    nk_target = coeff_matrix.shape[0]
    if coeff_matrix.shape[1] != nk_source:
        raise ValueError(f"coeff_matrix shape {coeff_matrix.shape} incompatible with source nk={nk_source}")
    lam = np.asarray(lambda_compact, dtype=np.complex128).reshape(nb, nk_target, nb, nk_source, order="F")
    lambda_blocks = np.transpose(lam, (1, 3, 0, 2))
    density_t = np.transpose(np.asarray(density, dtype=np.complex128), (2, 1, 0))
    intermediate = np.einsum("tsac,scd->tsad", lambda_blocks, density_t, optimize=True)
    fock = np.einsum("ts,tsad,tsbd->tab", coeff_matrix, intermediate, np.conj(lambda_blocks), optimize=True)
    return np.transpose(fock, (1, 2, 0))


def build_remote_interaction_blocks(
    target: SupercellBMSolution,
    source: SupercellBMSolution,
    density_source_blocks: np.ndarray,
    *,
    source_diagonals: dict[tuple[int, int], np.ndarray],
    target_diagonals: dict[tuple[int, int], np.ndarray],
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    v0: float,
    screening_lm: float,
    relative_permittivity: float,
    beta: float = 1.0,
    finite_zero_limit: bool = True,
    include_hartree: bool = True,
    include_fock: bool = True,
    progress_prefix: str | None = None,
) -> np.ndarray:
    density = np.asarray(density_source_blocks, dtype=np.complex128)
    n_spin, n_eta, nb, _nb_rhs, source_nk = density.shape
    if nb != target.nb or nb != source.nb:
        raise ValueError("Density band dimension must match target/source active band count")
    out = np.zeros((n_spin, n_eta, nb, nb, target.nk), dtype=np.complex128)
    scale = float(beta) * float(v0) / float(source_nk)

    if include_hartree:
        out += build_hartree_blocks_from_diagonals(
            density,
            source_diagonals=source_diagonals,
            target_diagonals=target_diagonals,
            shifts=shifts,
            gvecs=gvecs,
            target_nk=target.nk,
            v0=v0,
            screening_lm=screening_lm,
            relative_permittivity=relative_permittivity,
            beta=beta,
            finite_zero_limit=finite_zero_limit,
        )

    if not include_fock:
        return out

    for ishift, (shift, gvec) in enumerate(zip(shifts, gvecs, strict=True), start=1):
        if progress_prefix and (ishift == 1 or ishift == len(shifts) or ishift % 10 == 0):
            print(f"{progress_prefix} fock shift {ishift}/{len(shifts)} shift={shift}", flush=True)
        coeff_matrix = scale * screened_coulomb_matrix(
            source.lattice_kvec[None, :] - target.lattice_kvec[:, None] + complex(gvec),
            float(screening_lm),
            relative_permittivity=float(relative_permittivity),
            finite_zero_limit=bool(finite_zero_limit),
        )
        if not np.any(coeff_matrix):
            continue
        for ieta in range(n_eta):
            lam = compact_overlap_between(target, source, tuple(shift), valley_index=ieta)
            for ispin in range(n_spin):
                out[ispin, ieta] -= _contract_block_fock(lam, density[ispin, ieta], coeff_matrix)
    return out


def occupation_counts_svp_8over3(nb: int) -> np.ndarray:
    """Zhang Fig. 9(a)-style SVP occupation pattern for primitive nu=8/3.

    Sectors are indexed as ``(spin, valley)`` with spin 0/1 = up/down and
    valley 0/1 = K/K'.  The K' spin-up sector is partially filled with two
    out of six tripled-cell bands; the other three sectors are full.
    """

    nb = int(nb)
    if nb != 6:
        raise ValueError(
            "occupation_counts_svp_8over3 is specific to Zhang's sqrt(3) x sqrt(3) tripled cell, "
            f"which folds two primitive flat bands into nb=6; got nb={nb}"
        )
    return fixed_sector_occupation_counts(
        n_spin=2,
        n_eta=2,
        default_count=nb,
        overrides={(0, 1): 2},
        n_band=nb,
    )


def filling_from_occupation_counts(occupation_counts: np.ndarray, *, nb: int, area_ratio: int) -> float:
    return primitive_filling_from_occupation_counts(
        occupation_counts,
        reference_diagonal=0.5,
        n_band=int(nb),
        area_ratio=int(area_ratio),
    )


def density_from_fixed_sector_occupations(
    h_blocks: np.ndarray,
    occupation_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _core_density_from_fixed_sector_occupations(
        h_blocks,
        occupation_counts,
        reference_diagonal=0.5,
    )


def random_density_blocks(
    *,
    n_spin: int,
    n_eta: int,
    nb: int,
    nk: int,
    occupation_counts: np.ndarray,
    seed: int = 1,
) -> np.ndarray:
    """Random flavor-diagonal centered density with fixed sector occupations."""

    occ = np.asarray(occupation_counts, dtype=int)
    if occ.shape != (int(n_spin), int(n_eta)):
        raise ValueError(f"occupation_counts shape {occ.shape} incompatible with {(n_spin, n_eta)}")
    rng = np.random.default_rng(int(seed))
    density = np.zeros((int(n_spin), int(n_eta), int(nb), int(nb), int(nk)), dtype=np.complex128)
    eye = np.eye(int(nb), dtype=np.complex128)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            n_occ = int(occ[ispin, ieta])
            for ik in range(int(nk)):
                if n_occ == 0:
                    density[ispin, ieta, :, :, ik] = -0.5 * eye
                elif n_occ == int(nb):
                    density[ispin, ieta, :, :, ik] = 0.5 * eye
                else:
                    sampled = rng.standard_normal((int(nb), int(nb))) + 1j * rng.standard_normal((int(nb), int(nb)))
                    hermitian = sampled + sampled.conjugate().T
                    _evals, evecs = np.linalg.eigh(hermitian)
                    occ_vecs = evecs[:, :n_occ]
                    density[ispin, ieta, :, :, ik] = occ_vecs @ occ_vecs.conjugate().T - 0.5 * eye
    return density


def block_density_norm(updated: np.ndarray, previous: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(previous) - np.asarray(updated)))
    denominator = float(np.linalg.norm(np.asarray(updated)))
    if denominator < 1e-15:
        return 0.0 if numerator < 1e-15 else float("inf")
    return numerator / denominator


def run_hartree_only_scf(
    h0_blocks: np.ndarray,
    *,
    occupation_counts: np.ndarray,
    source_diagonals: dict[tuple[int, int], np.ndarray],
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    v0: float,
    screening_lm: float,
    relative_permittivity: float,
    beta: float = 1.0,
    finite_zero_limit: bool = True,
    max_iter: int = 80,
    mixing: float = 0.5,
    precision: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | bool | str]]:
    if not (0.0 < float(mixing) <= 1.0):
        raise ValueError(f"mixing must lie in (0,1], got {mixing}")
    h0 = np.asarray(h0_blocks, dtype=np.complex128)
    density, energies = density_from_fixed_sector_occupations(h0, occupation_counts)
    final_hartree = np.zeros_like(h0)
    final_norm = float("inf")
    converged = False
    iterations = 0
    for iteration in range(1, int(max_iter) + 1):
        hartree = build_hartree_blocks_from_diagonals(
            density,
            source_diagonals=source_diagonals,
            target_diagonals=source_diagonals,
            shifts=shifts,
            gvecs=gvecs,
            target_nk=h0.shape[-1],
            v0=v0,
            screening_lm=screening_lm,
            relative_permittivity=relative_permittivity,
            beta=beta,
            finite_zero_limit=finite_zero_limit,
        )
        trial_density, trial_energies = density_from_fixed_sector_occupations(h0 + hartree, occupation_counts)
        mixed_density = float(mixing) * trial_density + (1.0 - float(mixing)) * density
        final_norm = block_density_norm(trial_density, density)
        density = mixed_density
        energies = trial_energies
        final_hartree = hartree
        iterations = iteration
        print(f"[hartree-scf] iter={iteration} raw_norm={final_norm:.6e}", flush=True)
        if final_norm <= float(precision):
            converged = True
            break

    final_hartree = build_hartree_blocks_from_diagonals(
        density,
        source_diagonals=source_diagonals,
        target_diagonals=source_diagonals,
        shifts=shifts,
        gvecs=gvecs,
        target_nk=h0.shape[-1],
        v0=v0,
        screening_lm=screening_lm,
        relative_permittivity=relative_permittivity,
        beta=beta,
        finite_zero_limit=finite_zero_limit,
    )
    _final_density, final_energies = density_from_fixed_sector_occupations(h0 + final_hartree, occupation_counts)
    info: dict[str, float | int | bool | str] = {
        "mode": "hartree_only",
        "iterations": int(iterations),
        "converged": bool(converged),
        "final_raw_norm": float(final_norm),
        "mixing": float(mixing),
        "precision": float(precision),
    }
    return density, final_hartree, final_energies, info


def run_hartree_fock_scf(
    solution: SupercellBMSolution,
    h0_blocks: np.ndarray,
    *,
    occupation_counts: np.ndarray,
    source_diagonals: dict[tuple[int, int], np.ndarray],
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    v0: float,
    screening_lm: float,
    relative_permittivity: float,
    beta: float = 1.0,
    finite_zero_limit: bool = True,
    max_iter: int = 80,
    mixing: float = 0.5,
    precision: float = 1e-6,
    initial_density_blocks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | bool | str]]:
    """Restricted flavor-diagonal active-flat Hartree-Fock SCF.

    This is the density generator needed for Zhang Fig. 10: the plotted
    Hamiltonian contains only BM + remote + active Hartree, but the charge
    density should come from the density-wave HF state, where active Fock is
    what selects a nontrivial folded-cell density in the SVP sector.
    """

    if not (0.0 < float(mixing) <= 1.0):
        raise ValueError(f"mixing must lie in (0,1], got {mixing}")
    h0 = np.asarray(h0_blocks, dtype=np.complex128)
    if initial_density_blocks is None:
        density, energies = density_from_fixed_sector_occupations(h0, occupation_counts)
        init_mode = "bm"
    else:
        density = np.asarray(initial_density_blocks, dtype=np.complex128).copy()
        if density.shape != h0.shape:
            raise ValueError(f"initial_density_blocks shape {density.shape} does not match h0 shape {h0.shape}")
        _density_check, energies = density_from_fixed_sector_occupations(h0, occupation_counts)
        init_mode = "provided"
    final_interaction = np.zeros_like(h0)
    final_norm = float("inf")
    converged = False
    iterations = 0
    for iteration in range(1, int(max_iter) + 1):
        interaction = build_remote_interaction_blocks(
            solution,
            solution,
            density,
            source_diagonals=source_diagonals,
            target_diagonals=source_diagonals,
            shifts=shifts,
            gvecs=gvecs,
            v0=v0,
            screening_lm=screening_lm,
            relative_permittivity=relative_permittivity,
            beta=beta,
            finite_zero_limit=finite_zero_limit,
            include_hartree=True,
            include_fock=True,
            progress_prefix=None,
        )
        trial_density, trial_energies = density_from_fixed_sector_occupations(h0 + interaction, occupation_counts)
        mixed_density = float(mixing) * trial_density + (1.0 - float(mixing)) * density
        final_norm = block_density_norm(trial_density, density)
        density = mixed_density
        energies = trial_energies
        final_interaction = interaction
        iterations = iteration
        print(f"[active-hf-scf] iter={iteration} raw_norm={final_norm:.6e}", flush=True)
        if final_norm <= float(precision):
            converged = True
            break

    final_interaction = build_remote_interaction_blocks(
        solution,
        solution,
        density,
        source_diagonals=source_diagonals,
        target_diagonals=source_diagonals,
        shifts=shifts,
        gvecs=gvecs,
        v0=v0,
        screening_lm=screening_lm,
        relative_permittivity=relative_permittivity,
        beta=beta,
        finite_zero_limit=finite_zero_limit,
        include_hartree=True,
        include_fock=True,
        progress_prefix=None,
    )
    _final_density, final_energies = density_from_fixed_sector_occupations(h0 + final_interaction, occupation_counts)
    info: dict[str, float | int | bool | str] = {
        "mode": "hartree_fock_density_then_hartree_plot",
        "iterations": int(iterations),
        "converged": bool(converged),
        "final_raw_norm": float(final_norm),
        "mixing": float(mixing),
        "precision": float(precision),
        "init_mode": init_mode,
        "final_interaction_norm_mev": float(np.linalg.norm(final_interaction)),
    }
    return density, final_interaction, final_energies, info


def sector_labels(n_spin: int = 2, n_eta: int = 2) -> tuple[str, ...]:
    spin_labels = ["up", "down"] + [f"spin_{idx + 1}" for idx in range(2, n_spin)]
    valley_labels = ["K", "Kprime"] + [f"eta_{idx + 1}" for idx in range(2, n_eta)]
    return tuple(f"{valley_labels[ieta]}_{spin_labels[ispin]}" for ispin in range(n_spin) for ieta in range(n_eta))


def sector_index_from_label(label: str, *, n_spin: int = 2, n_eta: int = 2) -> tuple[int, int]:
    labels = sector_labels(n_spin=n_spin, n_eta=n_eta)
    normalized = str(label).strip().replace("'", "prime").replace("↑", "up").replace("↓", "down")
    aliases = {
        "K_up": "K_up",
        "K_down": "K_down",
        "Kprime_up": "Kprime_up",
        "Kprime_down": "Kprime_down",
        "Kp_up": "Kprime_up",
        "Kp_down": "Kprime_down",
        "partial": "Kprime_up",
    }
    canonical = aliases.get(normalized, normalized)
    if canonical not in labels:
        raise ValueError(f"Unknown sector label {label!r}; choices: {', '.join(labels)} plus partial")
    idx = labels.index(canonical)
    return idx // n_eta, idx % n_eta


def path_sector_energies(h_blocks: np.ndarray) -> np.ndarray:
    return sector_block_energies(h_blocks)


def max_hermitian_error(blocks: np.ndarray) -> float:
    h = np.asarray(blocks, dtype=np.complex128)
    return float(np.max(np.abs(h - np.swapaxes(h.conjugate(), 2, 3))))


def complex_to_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]
