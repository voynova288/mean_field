from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
from scipy.linalg import eigh

from ...core.hf import (
    DensityUpdateResult,
    HFOverlapBlockSet,
    ProjectedWavefunctionBasis,
    build_projected_interaction_hamiltonian,
    build_projected_target_hamiltonian,
    calculate_norm_convergence,
    calculate_projected_overlap_between,
    compute_hf_energy,
    density_from_fixed_sector_occupations as _core_density_from_fixed_sector_occupations,
    diagonal_overlap_blocks,
    flat_sector_indices as _core_flat_sector_indices,
    flatten_sector_blocks as _core_flatten_sector_blocks,
    real_space_cell_area_nm2_from_reciprocal,
    run_hartree_fock_iterations,
    screened_coulomb,
    screened_coulomb_matrix,
    sector_block_energies,
    shift_wavefunction_grid,
    unflatten_sector_blocks as _core_unflatten_sector_blocks,
    unflatten_sector_energies as _core_unflatten_sector_energies,
)
from .core_lattice import KPath, cumulative_distance
from .hamiltonian import build_diagonal_block, build_hamiltonian, dirac_block
from .lattice import TMBGLattice
from .model import TMBGModel
from .params import TMBGParameters


@dataclass(frozen=True)
class PolshynDoubledCell:
    """Area-2 rectangular cell used for the Polshyn tMBG SBCI.

    In the primitive tMBG convention ``b1=G_M1`` and ``b2=G_M2``.  The doubled
    cell keeps the y-translation and doubles the other primitive direction:

    ``B1 = b1/2`` and ``B2 = b2 - b1/2``.

    The CDW wavevector is therefore ``Q = B1``.
    """

    n11: int = 2
    n12: int = 1
    n21: int = 0
    n22: int = 1

    @property
    def area_ratio(self) -> int:
        return int(self.n11 * self.n22 - self.n12 * self.n21)

    def reciprocal_vectors(self, lattice: TMBGLattice) -> tuple[complex, complex]:
        b1 = complex(lattice.g_m1)
        b2 = complex(lattice.g_m2)
        return b1 / 2.0, b2 - b1 / 2.0

    def primitive_to_supercell_coords(self, n1: int, n2: int, fold: int = 0) -> tuple[int, int]:
        return (int(2 * n1 + n2 + fold), int(n2))

    def as_dict(self) -> dict[str, int]:
        return {
            "n11": int(self.n11),
            "n12": int(self.n12),
            "n21": int(self.n21),
            "n22": int(self.n22),
            "area_ratio": int(self.area_ratio),
        }


def polshyn_doubled_cell() -> PolshynDoubledCell:
    return PolshynDoubledCell()


@dataclass(frozen=True)
class PolshynFillingSummary:
    projected_indices: tuple[int, ...]
    target_band_index: int
    target_primitive_position: int
    target_fold_indices: tuple[int, int]
    nb: int
    area_ratio: int
    reference_diagonal: np.ndarray
    occupation_counts: np.ndarray
    primitive_nu: float
    matches_expected_filling: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "projected_indices": [int(value) for value in self.projected_indices],
            "target_band_index": int(self.target_band_index),
            "target_primitive_position": int(self.target_primitive_position),
            "target_fold_indices": [int(value) for value in self.target_fold_indices],
            "nb": int(self.nb),
            "area_ratio": int(self.area_ratio),
            "reference_diagonal": [float(value) for value in self.reference_diagonal],
            "occupation_counts": self.occupation_counts.astype(int).tolist(),
            "primitive_nu": float(self.primitive_nu),
            "matches_expected_filling": bool(self.matches_expected_filling),
        }


@dataclass(frozen=True)
class PolshynProjectedBasis:
    model: TMBGModel
    supercell: PolshynDoubledCell
    kvec: np.ndarray
    k_grid_frac: np.ndarray | None
    projected_indices: tuple[int, ...]
    target_band_index: int
    wavefunctions: np.ndarray
    h0_blocks: np.ndarray
    reference_diagonal: np.ndarray
    super_b1: complex
    super_b2: complex
    embedding_shape: tuple[int, int]
    embedding_origin: tuple[int, int]
    embedding_positions: dict[tuple[int, int, int], tuple[int, int]]

    @property
    def nk(self) -> int:
        return int(self.kvec.size)

    @property
    def n_eta(self) -> int:
        return int(self.wavefunctions.shape[2])

    @property
    def n_spin(self) -> int:
        return int(self.h0_blocks.shape[0])

    @property
    def nb(self) -> int:
        return int(self.wavefunctions.shape[1])

    @property
    def basis_dimension(self) -> int:
        return int(self.wavefunctions.shape[0])

    @property
    def local_basis_size(self) -> int:
        return 6


@dataclass
class PolshynWangHFState:
    """Minimal mutable state for the generic Wang/Xiaoyu HF iteration engine."""

    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    mu: float
    precision: float
    v0: float
    diagnostics: dict[str, float]

    @property
    def nk(self) -> int:
        return int(self.density.shape[2])


def build_doubled_uniform_grid(
    lattice: TMBGLattice,
    mesh: int,
    *,
    supercell: PolshynDoubledCell | None = None,
    endpoint: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    supercell = polshyn_doubled_cell() if supercell is None else supercell
    b1, b2 = supercell.reciprocal_vectors(lattice)
    mesh = int(mesh)
    if mesh <= 0:
        raise ValueError(f"mesh must be positive, got {mesh}")
    if endpoint:
        frac = np.linspace(0.0, 1.0, mesh, dtype=float)
    else:
        frac = np.arange(mesh, dtype=float) / float(mesh)
    f1, f2 = np.meshgrid(frac, frac, indexing="ij")
    kvec = f1 * b1 + f2 * b2
    frac_grid = np.stack([f1, f2], axis=-1)
    return np.asarray(frac_grid, dtype=float), np.asarray(kvec.reshape(-1), dtype=np.complex128)


def build_polshyn_s1a_path(lattice: TMBGLattice, points_per_segment: int) -> KPath:
    """Path for Supplementary Fig. S1a: Gamma-K_-^M-M-K_+^M-Gamma-M."""

    gamma = 0.0 + 0.0j
    return _build_path(
        (
            gamma,
            complex(lattice.k_m),
            complex(lattice.m_m),
            complex(lattice.kprime_m),
            gamma,
            complex(lattice.m_m),
        ),
        ("Gamma", "Kminus", "M", "Kplus", "Gamma", "M"),
        points_per_segment,
    )


def build_polshyn_kx0_path(lattice: TMBGLattice, points: int, *, supercell: PolshynDoubledCell | None = None) -> KPath:
    """Rectangular doubled-BZ line used in S1(b,c): kx=0, ky a_M in [-pi, pi].

    Keep the momenta in a centered supercell reciprocal gauge instead of
    wrapping them into the half-open SCF tile.  The HF Fock sum is truncated to a
    finite set of supercell reciprocal shifts; wrapping the line by a reciprocal
    vector changes the truncated set of q-vectors and produces artificial
    folded/wavy bands.  The projected-basis builder already sews primitive
    momenta back to a centered primitive gauge, so the continuous centered line
    is the correct plotting gauge.
    """

    supercell = polshyn_doubled_cell() if supercell is None else supercell
    _b1, b2 = supercell.reciprocal_vectors(lattice)
    points = int(points)
    if points < 2:
        raise ValueError("points must be at least 2")
    frac_centered = np.linspace(-0.5, 0.5, points, dtype=float)
    kvec = frac_centered * b2
    return KPath(
        kvec=np.asarray(kvec, dtype=np.complex128),
        kdist=np.asarray((frac_centered + 0.5) * abs(b2), dtype=float),
        labels=("-pi", "0", "pi"),
        node_indices=(1, int((points + 1) // 2), points),
    )


def _build_path(nodes: tuple[complex, ...], labels: tuple[str, ...], points_per_segment: int) -> KPath:
    if len(nodes) != len(labels):
        raise ValueError("nodes and labels must have the same length")
    kvec: list[complex] = [complex(nodes[0])]
    node_indices = [1]
    for start, stop in zip(nodes[:-1], nodes[1:], strict=True):
        step = (complex(stop) - complex(start)) / float(points_per_segment)
        for idx in range(1, int(points_per_segment) + 1):
            kvec.append(complex(start + idx * step))
        node_indices.append(len(kvec))
    array = np.asarray(kvec, dtype=np.complex128)
    return KPath(kvec=array, kdist=cumulative_distance(array), labels=labels, node_indices=tuple(node_indices))


def _embedding_table(
    lattice: TMBGLattice,
    supercell: PolshynDoubledCell,
    *,
    primitive_padding: int = 2,
) -> tuple[tuple[int, int], tuple[int, int], dict[tuple[int, int, int], tuple[int, int]]]:
    coords: list[tuple[int, int, int, int]] = []
    primitive_indices: set[tuple[int, int]] = set()
    pad = int(primitive_padding)
    for n1, n2 in np.asarray(lattice.g_indices, dtype=int):
        for dn1 in range(-pad, pad + 1):
            for dn2 in range(-pad, pad + 1):
                primitive_indices.add((int(n1) + int(dn1), int(n2) + int(dn2)))
    for n1, n2 in sorted(primitive_indices):
        for fold in (0, 1):
            sx, sy = supercell.primitive_to_supercell_coords(int(n1), int(n2), fold)
            coords.append((int(n1), int(n2), int(fold), int(sx), int(sy)))
    min_x = min(item[3] for item in coords)
    max_x = max(item[3] for item in coords)
    min_y = min(item[4] for item in coords)
    max_y = max(item[4] for item in coords)
    shape = (int(max_x - min_x + 1), int(max_y - min_y + 1))
    origin = (int(min_x), int(min_y))
    positions = {
        (n1, n2, fold): (int(sx - min_x), int(sy - min_y))
        for n1, n2, fold, sx, sy in coords
    }
    return shape, origin, positions


def _primitive_fractional_coords(lattice: TMBGLattice, k_tilde: complex) -> tuple[float, float]:
    matrix = np.asarray(
        [
            [float(complex(lattice.g_m1).real), float(complex(lattice.g_m2).real)],
            [float(complex(lattice.g_m1).imag), float(complex(lattice.g_m2).imag)],
        ],
        dtype=float,
    )
    rhs = np.asarray([float(complex(k_tilde).real), float(complex(k_tilde).imag)], dtype=float)
    coeff = np.linalg.solve(matrix, rhs)
    return float(coeff[0]), float(coeff[1])


def _centered_primitive_reduction(lattice: TMBGLattice, k_tilde: complex) -> tuple[complex, tuple[int, int]]:
    """Reduce a primitive momentum to the centered primitive reciprocal cell.

    The continuum Hamiltonian is periodic under primitive reciprocal vectors only
    up to a plane-wave-index sewing transformation.  For folded supercell bands
    one must diagonalize a canonical primitive representative and then embed the
    sewn eigenvector at the original folded momentum; otherwise finite-cutoff
    extended-zone evaluations create spurious folded branches.
    """

    f1, f2 = _primitive_fractional_coords(lattice, complex(k_tilde))
    s1 = int(np.floor(f1 + 0.5))
    s2 = int(np.floor(f2 + 0.5))
    reduced = complex(k_tilde - s1 * lattice.g_m1 - s2 * lattice.g_m2)
    return reduced, (s1, s2)


def _solve_projected_primitive(
    model: TMBGModel,
    k_tilde: complex,
    *,
    valley: int,
    projected_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    low = int(min(projected_indices))
    high = int(max(projected_indices))
    hmat = build_hamiltonian(complex(k_tilde), model.lattice, model.params, valley=int(valley))
    evals, evecs = eigh(hmat, subset_by_index=(low, high), driver="evr")
    local = {int(index): int(index - low) for index in range(low, high + 1)}
    cols = [local[int(index)] for index in projected_indices]
    return np.asarray(evals[cols], dtype=float), np.asarray(evecs[:, cols], dtype=np.complex128)


def reference_diagonal_for_projected_indices(projected_indices: tuple[int, ...], target_band_index: int) -> np.ndarray:
    """Reference density for Polshyn's conduction-band filling convention.

    The experimental/paper filling ``nu=7/2`` counts electrons added into the
    target conduction C=2 band.  Therefore the target band reference is empty,
    not half-filled as in charge-neutral two-flat-band TBG conventions.  Remote
    bands below the target are part of the subtraction-method sea and are filled
    in the reference; remote bands above the target are empty.
    """

    values: list[float] = []
    for index in projected_indices:
        if int(index) < int(target_band_index):
            ref = 1.0
        else:
            ref = 0.0
        values.extend([ref, ref])
    return np.asarray(values, dtype=float)


def occupation_counts_nu_7over2(projected_indices: tuple[int, ...], target_band_index: int) -> np.ndarray:
    lower_count = sum(1 for index in projected_indices if int(index) < int(target_band_index))
    target_slots = sum(1 for index in projected_indices if int(index) == int(target_band_index)) * 2
    if target_slots != 2:
        raise ValueError("Expected exactly one primitive target band, folded into two supercell bands")
    full = 2 * int(lower_count) + 2
    partial = 2 * int(lower_count) + 1
    occ = np.full((2, 2), int(full), dtype=int)
    occ[0, 0] = int(partial)  # spin up, valley K+
    return occ


def primitive_nu_from_counts(occupation_counts: np.ndarray, reference_diagonal: np.ndarray, *, area_ratio: int) -> float:
    reference_total = 2 * 2 * float(np.sum(reference_diagonal))
    occupied_total = float(np.sum(np.asarray(occupation_counts, dtype=int)))
    return (occupied_total - reference_total) / float(area_ratio)



def polshyn_nu_7over2_filling_summary(
    projected_indices: tuple[int, ...],
    *,
    target_band_index: int,
    area_ratio: int = 2,
) -> PolshynFillingSummary:
    indices = tuple(int(index) for index in projected_indices)
    target = int(target_band_index)
    if target not in indices:
        raise ValueError(f"target_band_index={target} is not present in projected_indices={indices}")
    target_position = indices.index(target)
    target_fold_indices = (2 * target_position, 2 * target_position + 1)
    reference = reference_diagonal_for_projected_indices(indices, target)
    counts = occupation_counts_nu_7over2(indices, target)
    primitive_nu = primitive_nu_from_counts(counts, reference, area_ratio=int(area_ratio))
    return PolshynFillingSummary(
        projected_indices=indices,
        target_band_index=target,
        target_primitive_position=int(target_position),
        target_fold_indices=target_fold_indices,
        nb=2 * len(indices),
        area_ratio=int(area_ratio),
        reference_diagonal=reference,
        occupation_counts=counts,
        primitive_nu=float(primitive_nu),
        matches_expected_filling=bool(np.isclose(primitive_nu, 3.5, atol=1.0e-12)),
    )

def build_polshyn_projected_basis(
    model: TMBGModel,
    kvec: np.ndarray,
    *,
    projected_indices: tuple[int, ...],
    target_band_index: int,
    supercell: PolshynDoubledCell | None = None,
    k_grid_frac: np.ndarray | None = None,
) -> PolshynProjectedBasis:
    supercell = polshyn_doubled_cell() if supercell is None else supercell
    projected_indices = tuple(int(index) for index in projected_indices)
    if int(target_band_index) not in projected_indices:
        raise ValueError(f"target_band_index={target_band_index} not in projected_indices={projected_indices}")
    kvec = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    lattice = model.lattice
    super_b1, super_b2 = supercell.reciprocal_vectors(lattice)
    grid_shape, origin, positions = _embedding_table(lattice, supercell)
    nx, ny = grid_shape
    n_primitive = len(projected_indices)
    nb = 2 * n_primitive
    embedded = np.zeros((6, nx, ny, nb, 2, kvec.size), dtype=np.complex128)
    h0_valley = np.zeros((nb, nb, 2, kvec.size), dtype=np.complex128)

    for ieta, valley in enumerate((1, -1)):
        for ik, kval in enumerate(kvec):
            for fold in (0, 1):
                primitive_k_full = complex(kval + fold * super_b1)
                primitive_k, primitive_shift = _centered_primitive_reduction(lattice, primitive_k_full)
                evals, evecs = _solve_projected_primitive(
                    model,
                    primitive_k,
                    valley=valley,
                    projected_indices=projected_indices,
                )
                shift_n1, shift_n2 = primitive_shift
                for iprim, _band_index in enumerate(projected_indices):
                    out_band = 2 * iprim + fold
                    h0_valley[out_band, out_band, ieta, ik] = float(evals[iprim])
                    for source_g_index, pair in enumerate(np.asarray(lattice.g_indices, dtype=int)):
                        embed_key = (int(pair[0]) - shift_n1, int(pair[1]) - shift_n2, int(fold))
                        target_position = positions.get(embed_key)
                        if target_position is None:
                            continue
                        ix, iy = target_position
                        start = 6 * source_g_index
                        embedded[:, ix, iy, out_band, ieta, ik] = evecs[start : start + 6, iprim]

    wavefunctions = embedded.reshape((6 * nx * ny, nb, 2, kvec.size), order="F")
    h0_blocks = np.zeros((2, 2, nb, nb, kvec.size), dtype=np.complex128)
    for ispin in range(2):
        for ieta in range(2):
            h0_blocks[ispin, ieta] = h0_valley[:, :, ieta, :]

    return PolshynProjectedBasis(
        model=model,
        supercell=supercell,
        kvec=kvec,
        k_grid_frac=None if k_grid_frac is None else np.asarray(k_grid_frac, dtype=float),
        projected_indices=projected_indices,
        target_band_index=int(target_band_index),
        wavefunctions=wavefunctions,
        h0_blocks=h0_blocks,
        reference_diagonal=reference_diagonal_for_projected_indices(projected_indices, int(target_band_index)),
        super_b1=complex(super_b1),
        super_b2=complex(super_b2),
        embedding_shape=grid_shape,
        embedding_origin=origin,
        embedding_positions=positions,
    )


def _shift_wavefunction_grid(values: np.ndarray, dm: int, dn: int) -> np.ndarray:
    return shift_wavefunction_grid(values, dm, dn, boundary_mode="zero_fill", grid_axes=(1, 2))


def compact_overlap_between(
    target: PolshynProjectedBasis,
    source: PolshynProjectedBasis,
    shift: tuple[int, int],
    *,
    valley_index: int,
) -> np.ndarray:
    if target.nb != source.nb or target.embedding_shape != source.embedding_shape:
        raise ValueError("target/source basis mismatch")
    nb = int(target.nb)
    nx, ny = target.embedding_shape
    target_cols = nb * target.nk
    source_cols = nb * source.nk
    ul = target.wavefunctions[:, :, valley_index, :].reshape(target.basis_dimension, target_cols, order="F")
    ur_grid = source.wavefunctions[:, :, valley_index, :].reshape(source.local_basis_size, nx, ny, source_cols, order="F")
    shifted = _shift_wavefunction_grid(ur_grid, -int(shift[0]), -int(shift[1])).reshape(
        source.basis_dimension,
        source_cols,
        order="F",
    )
    return ul.conj().T @ shifted


def compact_diagonal_overlap(
    basis: PolshynProjectedBasis,
    shift: tuple[int, int],
    *,
    valley_index: int,
) -> np.ndarray:
    nb = int(basis.nb)
    nx, ny = basis.embedding_shape
    w_grid = basis.wavefunctions[:, :, valley_index, :].reshape(basis.local_basis_size, nx, ny, nb, basis.nk, order="F")
    shifted = _shift_wavefunction_grid(w_grid, -int(shift[0]), -int(shift[1]))
    return np.einsum("lxyak,lxybk->abk", np.conj(w_grid), shifted, optimize=True)


def reciprocal_shift_labels(g_shells: int) -> tuple[int, ...]:
    if int(g_shells) < 0:
        raise ValueError("g_shells must be non-negative")
    return tuple(range(-int(g_shells), int(g_shells) + 1))


def supercell_interaction_shifts(basis: PolshynProjectedBasis, g_shells: int) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    labels = reciprocal_shift_labels(g_shells)
    shifts = tuple((m, n) for n in labels for m in labels)
    gvecs = np.asarray([m * basis.super_b1 + n * basis.super_b2 for m, n in shifts], dtype=np.complex128)
    return shifts, gvecs


def precompute_diagonal_overlaps(
    basis: PolshynProjectedBasis,
    shifts: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for shift in shifts:
        out[tuple(shift)] = np.asarray(
            [compact_diagonal_overlap(basis, tuple(shift), valley_index=ieta) for ieta in range(basis.n_eta)],
            dtype=np.complex128,
        )
    return out


def precompute_compact_overlaps(
    target: PolshynProjectedBasis,
    source: PolshynProjectedBasis,
    shifts: Iterable[tuple[int, int]],
    *,
    progress_prefix: str | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    shift_tuple = tuple(tuple(shift) for shift in shifts)
    for ishift, shift in enumerate(shift_tuple, start=1):
        if progress_prefix and (ishift == 1 or ishift == len(shift_tuple) or ishift % 10 == 0):
            print(f"{progress_prefix} overlap {ishift}/{len(shift_tuple)} shift={shift}", flush=True)
        out[shift] = np.asarray(
            [compact_overlap_between(target, source, shift, valley_index=ieta) for ieta in range(source.n_eta)],
            dtype=np.complex128,
        )
    return out


def density_trace_for_shift(density_blocks: np.ndarray, diagonal_by_valley: np.ndarray) -> complex:
    density = np.asarray(density_blocks, dtype=np.complex128)
    diagonal = np.asarray(diagonal_by_valley, dtype=np.complex128)
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
    epsilon_r: float,
    d_sc_nm: float,
) -> np.ndarray:
    density = np.asarray(density_source_blocks, dtype=np.complex128)
    n_spin, n_eta, nb, _nb_rhs, source_nk = density.shape
    out = np.zeros((n_spin, n_eta, nb, nb, int(target_nk)), dtype=np.complex128)
    scale = float(v0) / float(source_nk)
    for shift, gvec in zip(shifts, gvecs, strict=True):
        trace = density_trace_for_shift(density, source_diagonals[tuple(shift)])
        kernel = screened_coulomb(complex(gvec), epsilon_r=float(epsilon_r), d_sc_nm=float(d_sc_nm))
        coeff = scale * float(kernel) * trace
        if coeff == 0.0:
            continue
        target_diag = target_diagonals[tuple(shift)]
        for ispin in range(n_spin):
            for ieta in range(n_eta):
                out[ispin, ieta] += coeff * target_diag[ieta]
    return out


def _contract_block_fock(lambda_compact: np.ndarray, density: np.ndarray, coeff_matrix: np.ndarray) -> np.ndarray:
    nb, _, nk_source = density.shape
    nk_target = coeff_matrix.shape[0]
    lam = np.asarray(lambda_compact, dtype=np.complex128).reshape(nb, nk_target, nb, nk_source, order="F")
    lambda_blocks = np.transpose(lam, (1, 3, 0, 2))
    density_t = np.transpose(np.asarray(density, dtype=np.complex128), (2, 1, 0))
    intermediate = np.einsum("tsac,scd->tsad", lambda_blocks, density_t, optimize=True)
    fock = np.einsum("ts,tsad,tsbd->tab", coeff_matrix, intermediate, np.conj(lambda_blocks), optimize=True)
    return np.transpose(fock, (1, 2, 0))


def build_interaction_blocks(
    target: PolshynProjectedBasis,
    source: PolshynProjectedBasis,
    density_source_blocks: np.ndarray,
    *,
    source_diagonals: dict[tuple[int, int], np.ndarray],
    target_diagonals: dict[tuple[int, int], np.ndarray],
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    v0: float,
    epsilon_r: float,
    d_sc_nm: float,
    include_hartree: bool = True,
    include_fock: bool = True,
    compact_overlaps: dict[tuple[int, int], np.ndarray] | None = None,
    progress_prefix: str | None = None,
) -> np.ndarray:
    density = np.asarray(density_source_blocks, dtype=np.complex128)
    out = np.zeros((source.n_spin, source.n_eta, source.nb, source.nb, target.nk), dtype=np.complex128)
    if include_hartree:
        out += build_hartree_blocks_from_diagonals(
            density,
            source_diagonals=source_diagonals,
            target_diagonals=target_diagonals,
            shifts=shifts,
            gvecs=gvecs,
            target_nk=target.nk,
            v0=v0,
            epsilon_r=epsilon_r,
            d_sc_nm=d_sc_nm,
        )
    if not include_fock:
        return out
    scale = float(v0) / float(source.nk)
    for ishift, (shift, gvec) in enumerate(zip(shifts, gvecs, strict=True), start=1):
        if progress_prefix and (ishift == 1 or ishift == len(shifts) or ishift % 10 == 0):
            print(f"{progress_prefix} fock shift {ishift}/{len(shifts)} shift={shift}", flush=True)
        coeff_matrix = scale * screened_coulomb_matrix(
            source.kvec[None, :] - target.kvec[:, None] + complex(gvec),
            epsilon_r=float(epsilon_r),
            d_sc_nm=float(d_sc_nm),
        )
        overlap_by_valley = None if compact_overlaps is None else compact_overlaps.get(tuple(shift))
        for ieta in range(source.n_eta):
            lam = (
                compact_overlap_between(target, source, tuple(shift), valley_index=ieta)
                if overlap_by_valley is None
                else overlap_by_valley[ieta]
            )
            for ispin in range(source.n_spin):
                out[ispin, ieta] -= _contract_block_fock(lam, density[ispin, ieta], coeff_matrix)
    return out


def density_from_fixed_sector_occupations(
    h_blocks: np.ndarray,
    occupation_counts: np.ndarray,
    reference_diagonal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _core_density_from_fixed_sector_occupations(
        h_blocks,
        occupation_counts,
        reference_diagonal=reference_diagonal,
    )

def cdw_density_blocks(
    *,
    projected_indices: tuple[int, ...],
    target_band_index: int,
    n_spin: int,
    n_eta: int,
    nb: int,
    nk: int,
    reference_diagonal: np.ndarray,
) -> np.ndarray:
    """Maximal translation-breaking initializer for the K+ spin-up target band."""

    projected_indices = tuple(int(index) for index in projected_indices)
    target_primitive_pos = projected_indices.index(int(target_band_index))
    target_fold_indices = (2 * target_primitive_pos, 2 * target_primitive_pos + 1)
    reference = np.diag(np.asarray(reference_diagonal, dtype=float)).astype(np.complex128)
    density = np.zeros((int(n_spin), int(n_eta), int(nb), int(nb), int(nk)), dtype=np.complex128)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            projector = np.zeros((int(nb), int(nb)), dtype=np.complex128)
            # Lower remote bands are filled in the reference and stay filled in the initializer.
            for iprim, band_index in enumerate(projected_indices):
                if int(band_index) < int(target_band_index):
                    projector[2 * iprim, 2 * iprim] = 1.0
                    projector[2 * iprim + 1, 2 * iprim + 1] = 1.0
            if ispin == 0 and ieta == 0:
                i0, i1 = target_fold_indices
                projector[i0, i0] = 0.5
                projector[i1, i1] = 0.5
                projector[i0, i1] = 0.5
                projector[i1, i0] = 0.5
            else:
                i0, i1 = target_fold_indices
                projector[i0, i0] = 1.0
                projector[i1, i1] = 1.0
            for ik in range(int(nk)):
                density[ispin, ieta, :, :, ik] = projector - reference
    return density


def random_density_blocks(
    *,
    n_spin: int,
    n_eta: int,
    nb: int,
    nk: int,
    occupation_counts: np.ndarray,
    reference_diagonal: np.ndarray,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    occ = np.asarray(occupation_counts, dtype=int)
    reference = np.diag(np.asarray(reference_diagonal, dtype=float)).astype(np.complex128)
    density = np.zeros((int(n_spin), int(n_eta), int(nb), int(nb), int(nk)), dtype=np.complex128)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            n_occ = int(occ[ispin, ieta])
            for ik in range(int(nk)):
                if n_occ == 0:
                    projector = np.zeros((nb, nb), dtype=np.complex128)
                elif n_occ == int(nb):
                    projector = np.eye(int(nb), dtype=np.complex128)
                else:
                    sampled = rng.standard_normal((int(nb), int(nb))) + 1j * rng.standard_normal((int(nb), int(nb)))
                    hermitian = sampled + sampled.conjugate().T
                    _evals, evecs = np.linalg.eigh(hermitian)
                    vecs = evecs[:, :n_occ]
                    projector = vecs @ vecs.conjugate().T
                density[ispin, ieta, :, :, ik] = projector - reference
    return density


def _lowest_band_projector(hmat: np.ndarray, n_occ: int, *, degeneracy_tol: float = 1.0e-12) -> np.ndarray:
    """Projector onto the ``n_occ`` lowest eigenvectors of a local Hermitian block."""

    h = np.asarray(hmat, dtype=np.complex128)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"Expected a square local block, got {h.shape}")
    n_occ = int(n_occ)
    if n_occ < 0 or n_occ > h.shape[0]:
        raise ValueError(f"n_occ={n_occ} incompatible with local block size {h.shape[0]}")
    h = 0.5 * (h + h.conjugate().T)
    if n_occ == 0:
        return np.zeros_like(h)
    if n_occ == h.shape[0]:
        return np.eye(h.shape[0], dtype=np.complex128)
    evals, evecs = np.linalg.eigh(h)
    if float(evals[n_occ] - evals[n_occ - 1]) < float(degeneracy_tol):
        # A Fermi-level degeneracy makes the reference gauge ambiguous.  Use the
        # deterministic eigh basis but keep this path explicit for diagnostics.
        pass
    vecs = evecs[:, :n_occ]
    return vecs @ vecs.conjugate().T


def _single_layer_valence_projector(h2: np.ndarray, *, degeneracy_tol: float = 1.0e-12) -> np.ndarray:
    """Charge-neutral valence projector for one decoupled Dirac layer."""

    h = np.asarray(h2, dtype=np.complex128)
    if h.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 Dirac block, got {h.shape}")
    h = 0.5 * (h + h.conjugate().T)
    evals, evecs = np.linalg.eigh(h)
    if float(np.max(evals) - np.min(evals)) < float(degeneracy_tol):
        # Exactly at a massless Dirac point the CNP projector is gauge ambiguous;
        # use the rotationally invariant half-filled limit.
        return 0.5 * np.eye(2, dtype=np.complex128)
    vec = evecs[:, [0]]
    return vec @ vec.conjugate().T


def _decoupled_layers_cnp_block(
    basis: PolshynProjectedBasis,
    k_super: complex,
    *,
    n1: int,
    n2: int,
    fold: int,
    valley: int,
    p0_reference: str = "decoupled-layers",
) -> np.ndarray:
    """Local six-orbital P0 block for the subtraction-method reference.

    This implements the local CNP Slater determinant used by the subtraction
    reference, expressed on the same folded supercell plane-wave embedding as
    the Polshyn basis.

    ``p0_reference='decoupled-layers'`` is the literal Soejima TBG-style choice:
    three independent monolayer Dirac seas.  ``'bernal-bilayer'`` keeps the
    untwisted bottom-middle Bernal block (and layer potentials) while switching
    off only the moire top-middle tunnelling; this is a tMBG-specific diagnostic.
    """

    lattice = basis.model.lattice
    params = basis.model.params
    k_site = complex(k_super + int(fold) * basis.super_b1 + int(n1) * lattice.g_m1 + int(n2) * lattice.g_m2)
    mode = str(p0_reference).strip().lower().replace("_", "-")
    if mode in {"bernal-bilayer", "untwisted", "untwisted-bernal"}:
        k_tilde = complex(k_super + int(fold) * basis.super_b1)
        gvec = complex(int(n1) * lattice.g_m1 + int(n2) * lattice.g_m2)
        return _lowest_band_projector(
            build_diagonal_block(k_tilde, gvec, lattice, params, int(valley)),
            3,
        )
    if mode != "decoupled-layers":
        raise ValueError(f"Unsupported p0_reference={p0_reference!r}")
    k_bottom = complex(k_site - int(valley) * lattice.k_m)
    k_top = complex(k_site - int(valley) * lattice.kprime_m)
    h_bottom = dirac_block(k_bottom, -lattice.theta_rad / 2.0, params.vf, int(valley))
    h_middle = dirac_block(k_bottom, -lattice.theta_rad / 2.0, params.vf, int(valley))
    h_top = dirac_block(k_top, lattice.theta_rad / 2.0, params.vf, int(valley))
    out = np.zeros((6, 6), dtype=np.complex128)
    out[0:2, 0:2] = _single_layer_valence_projector(h_bottom)
    out[2:4, 2:4] = _single_layer_valence_projector(h_middle)
    out[4:6, 4:6] = _single_layer_valence_projector(h_top)
    return out


def projected_decoupled_cnp_density_blocks(
    basis: PolshynProjectedBasis,
    *,
    p0_reference: str = "decoupled-layers",
) -> np.ndarray:
    """Project the decoupled-layer CNP reference ``P0`` into a Polshyn basis.

    The returned array uses the conventional Hermitian density convention with
    shape ``(spin, valley, band, band, k)``.  It is intended for first-pass
    subtraction-method diagnostics.  The full Soejima subtraction also contains
    active-remote off-block pieces; this active-projected P0 term is the part
    that can be represented inside the current projected HF basis.
    """

    nb = int(basis.nb)
    out = np.zeros((basis.n_spin, basis.n_eta, nb, nb, basis.nk), dtype=np.complex128)
    nx, ny = basis.embedding_shape
    valleys = (1, -1)
    position_items = tuple(basis.embedding_positions.items())
    for ieta, valley in enumerate(valleys):
        for ik, kval in enumerate(basis.kvec):
            w_grid = basis.wavefunctions[:, :, ieta, ik].reshape(basis.local_basis_size, nx, ny, nb, order="F")
            p0_band = np.zeros((nb, nb), dtype=np.complex128)
            for (n1, n2, fold), (ix, iy) in position_items:
                u_site = w_grid[:, int(ix), int(iy), :]
                if not np.any(u_site):
                    continue
                p0_local = _decoupled_layers_cnp_block(
                    basis,
                    complex(kval),
                    n1=int(n1),
                    n2=int(n2),
                    fold=int(fold),
                    valley=int(valley),
                    p0_reference=p0_reference,
                )
                p0_band += u_site.conjugate().T @ p0_local @ u_site
            p0_band = 0.5 * (p0_band + p0_band.conjugate().T)
            for ispin in range(basis.n_spin):
                out[ispin, ieta, :, :, ik] = p0_band
    return out


def reference_projector_blocks(basis: PolshynProjectedBasis) -> np.ndarray:
    """Conventional projector corresponding to ``basis.reference_diagonal``."""

    ref = np.diag(np.asarray(basis.reference_diagonal, dtype=float)).astype(np.complex128)
    out = np.zeros((basis.n_spin, basis.n_eta, basis.nb, basis.nb, basis.nk), dtype=np.complex128)
    for ispin in range(basis.n_spin):
        for ieta in range(basis.n_eta):
            for ik in range(basis.nk):
                out[ispin, ieta, :, :, ik] = ref
    return out


def projected_p0_subtraction_density_blocks(
    basis: PolshynProjectedBasis,
    *,
    include_active_reference: bool = True,
    p0_reference: str = "decoupled-layers",
) -> tuple[np.ndarray, dict[str, float]]:
    """Density whose HF potential is added to ``h0`` for projected-P0 subtraction.

    The runner evolves the interaction using ``P - P_ref``.  To recover a
    physical mean-field Hamiltonian ``h_ren + HF[P]`` in that convention, the
    static ``h0`` correction is represented as ``HF[-P0_projected + P_ref]``.
    """

    p0 = projected_decoupled_cnp_density_blocks(basis, p0_reference=p0_reference)
    density = -p0
    if include_active_reference:
        density = density + reference_projector_blocks(basis)
    trace_p0 = np.trace(p0, axis1=2, axis2=3).real
    trace_density = np.trace(density, axis1=2, axis2=3).real
    diagnostics = {
        "projected_p0_trace_mean": float(np.mean(trace_p0)),
        "projected_p0_trace_min": float(np.min(trace_p0)),
        "projected_p0_trace_max": float(np.max(trace_p0)),
        "subtraction_density_trace_mean": float(np.mean(trace_density)),
        "subtraction_density_trace_min": float(np.min(trace_density)),
        "subtraction_density_trace_max": float(np.max(trace_density)),
        "include_active_reference": float(1.0 if include_active_reference else 0.0),
    }
    return np.asarray(density, dtype=np.complex128), diagnostics


def _p0_times_wavefunction_grid(
    basis: PolshynProjectedBasis,
    *,
    valley_index: int,
    p0_reference: str = "decoupled-layers",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``P0 U`` and ``U^† P0 U`` for all k in the folded basis."""

    ieta = int(valley_index)
    valley = (1, -1)[ieta]
    nb = int(basis.nb)
    nx, ny = basis.embedding_shape
    u_grid = basis.wavefunctions[:, :, ieta, :].reshape(
        basis.local_basis_size,
        nx,
        ny,
        nb,
        basis.nk,
        order="F",
    )
    p0u_grid = np.zeros_like(u_grid)
    p0_band = np.zeros((nb, nb, basis.nk), dtype=np.complex128)
    position_items = tuple(basis.embedding_positions.items())
    for ik, kval in enumerate(basis.kvec):
        for (n1, n2, fold), (ix, iy) in position_items:
            u_site = u_grid[:, int(ix), int(iy), :, ik]
            if not np.any(u_site):
                continue
            p0_local = _decoupled_layers_cnp_block(
                basis,
                complex(kval),
                n1=int(n1),
                n2=int(n2),
                fold=int(fold),
                valley=int(valley),
                p0_reference=p0_reference,
            )
            p0u_grid[:, int(ix), int(iy), :, ik] = p0_local @ u_site
        p0_band[:, :, ik] = np.einsum(
            "lxyb,lxya->ab",
            np.conj(u_grid[:, :, :, :, ik]),
            p0u_grid[:, :, :, :, ik],
            optimize=True,
        )
        p0_band[:, :, ik] = 0.5 * (p0_band[:, :, ik] + p0_band[:, :, ik].conjugate().T)
    return p0u_grid, p0_band


def _compact_overlap_to_source_grid(
    target: PolshynProjectedBasis,
    source_grid: np.ndarray,
    shift: tuple[int, int],
    *,
    valley_index: int,
) -> np.ndarray:
    """Overlap ``U_target^† T_shift X_source`` for a source grid ``X``."""

    nb = int(target.nb)
    nx, ny = target.embedding_shape
    source = np.asarray(source_grid, dtype=np.complex128)
    expected = (target.local_basis_size, nx, ny, nb, target.nk)
    if source.shape != expected:
        raise ValueError(f"Expected source_grid shape {expected}, got {source.shape}")
    target_cols = nb * target.nk
    ul = target.wavefunctions[:, :, valley_index, :].reshape(target.basis_dimension, target_cols, order="F")
    source_cols = nb * target.nk
    shifted = _shift_wavefunction_grid(
        source.reshape(target.local_basis_size, nx, ny, source_cols, order="F"),
        -int(shift[0]),
        -int(shift[1]),
    ).reshape(target.basis_dimension, source_cols, order="F")
    return ul.conj().T @ shifted


def _compact_block(compact: np.ndarray, nb: int, kt: int, ks: int) -> np.ndarray:
    row = slice(int(kt) * int(nb), (int(kt) + 1) * int(nb))
    col = slice(int(ks) * int(nb), (int(ks) + 1) * int(nb))
    return np.asarray(compact[row, col], dtype=np.complex128)


def build_full_p0_subtraction_h0_correction(
    basis: PolshynProjectedBasis,
    *,
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    v0: float,
    epsilon_r: float,
    d_sc_nm: float,
    zero_hartree_q0: bool = True,
    include_active_reference: bool = True,
    p0_reference: str = "decoupled-layers",
    hartree_scale: float = 1.0,
    fock_scale: float = 1.0,
    progress_prefix: str | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Full active-remote P0 subtraction correction for the Polshyn projected basis.

    Soejima/Parker/Zaletel's subtraction one-body term is linear in
    ``P_r - P0``.  If ``Q_A`` is the active projected subspace and
    ``P_r = (1-Q_A) P0 (1-Q_A)``, the source density that must be represented
    in the current ``P-P_ref`` SCF convention is

        ``D = P_ref + P_r - P0 = P_ref - Q_A P0 - P0 Q_A + Q_A P0 Q_A``.

    Unlike ``projected_p0_subtraction_density_blocks`` this keeps the
    active-remote off-block pieces.  The contractions are evaluated in the
    plane-wave/orbital embedding without explicitly constructing remote bands.
    """

    shift_tuple = tuple((int(m), int(n)) for m, n in shifts)
    gvec_array = np.asarray(gvecs, dtype=np.complex128)
    if len(shift_tuple) != int(gvec_array.size):
        raise ValueError("shifts and gvecs must have the same length")
    nb = int(basis.nb)
    nk = int(basis.nk)
    ref = np.diag(np.asarray(basis.reference_diagonal, dtype=float)).astype(np.complex128)
    ref_for_m = ref if bool(include_active_reference) else np.zeros_like(ref)
    scale = float(v0) / float(nk)
    correction = np.zeros_like(basis.h0_blocks)

    p0u_by_valley: list[np.ndarray] = []
    p0_band_by_valley: list[np.ndarray] = []
    m_by_valley: list[np.ndarray] = []
    for ieta in range(basis.n_eta):
        p0u_grid, p0_band = _p0_times_wavefunction_grid(basis, valley_index=ieta, p0_reference=p0_reference)
        p0u_by_valley.append(p0u_grid)
        p0_band_by_valley.append(p0_band)
        m_blocks = np.zeros_like(p0_band)
        for ik in range(nk):
            # M in D = U M U^† - U(P0 U)^† - (P0 U)U^†.
            m_blocks[:, :, ik] = ref_for_m + p0_band[:, :, ik]
        m_by_valley.append(m_blocks)

    hartree_norm = 0.0
    fock_norm = 0.0
    trace_abs_max = 0.0
    for ishift, (shift, gvec) in enumerate(zip(shift_tuple, gvec_array, strict=True), start=1):
        if progress_prefix and (ishift == 1 or ishift == len(shift_tuple) or ishift % 10 == 0):
            print(f"{progress_prefix} full-p0 shift {ishift}/{len(shift_tuple)} shift={shift}", flush=True)
        target_diagonal = np.asarray(
            [compact_diagonal_overlap(basis, shift, valley_index=ieta) for ieta in range(basis.n_eta)],
            dtype=np.complex128,
        )
        k_p0 = [
            _compact_overlap_to_source_grid(basis, p0u_by_valley[ieta], shift, valley_index=ieta)
            for ieta in range(basis.n_eta)
        ]
        k_p0_dagger_shift = [
            _compact_overlap_to_source_grid(basis, p0u_by_valley[ieta], (-shift[0], -shift[1]), valley_index=ieta)
            for ieta in range(basis.n_eta)
        ]

        trace_total = 0.0 + 0.0j
        for ispin in range(basis.n_spin):
            for ieta in range(basis.n_eta):
                diag = target_diagonal[ieta]
                for ik in range(nk):
                    lam = diag[:, :, ik]
                    m_block = m_by_valley[ieta][:, :, ik]
                    k_same = _compact_block(k_p0[ieta], nb, ik, ik)
                    j_same = _compact_block(k_p0_dagger_shift[ieta], nb, ik, ik)
                    trace = np.einsum("ab,ab->", m_block, np.conj(lam), optimize=True)
                    trace -= np.conj(np.trace(k_same))
                    trace -= np.trace(j_same)
                    trace_total += trace
        trace_abs_max = max(trace_abs_max, float(abs(trace_total)))
        hartree_kernel = float(hartree_scale) * screened_coulomb(complex(gvec), epsilon_r=float(epsilon_r), d_sc_nm=float(d_sc_nm))
        if bool(zero_hartree_q0) and shift == (0, 0):
            hartree_kernel = 0.0
        hartree_piece = np.zeros_like(correction)
        if float(hartree_kernel) != 0.0:
            coeff_h = scale * float(hartree_kernel) * trace_total
            for ispin in range(basis.n_spin):
                for ieta in range(basis.n_eta):
                    hartree_piece[ispin, ieta] += coeff_h * target_diagonal[ieta]
            correction += hartree_piece
            hartree_norm += float(np.linalg.norm(hartree_piece))

        fock_kernel = float(fock_scale) * screened_coulomb_matrix(
            basis.kvec[None, :] - basis.kvec[:, None] + complex(gvec),
            epsilon_r=float(epsilon_r),
            d_sc_nm=float(d_sc_nm),
        )
        for ieta in range(basis.n_eta):
            lam_compact = compact_overlap_between(basis, basis, shift, valley_index=ieta)
            k_compact = k_p0[ieta]
            for ispin in range(basis.n_spin):
                for kt in range(nk):
                    fock_block = np.zeros((nb, nb), dtype=np.complex128)
                    for ks in range(nk):
                        coeff = scale * float(fock_kernel[kt, ks])
                        if coeff == 0.0:
                            continue
                        lam = _compact_block(lam_compact, nb, kt, ks)
                        kblk = _compact_block(k_compact, nb, kt, ks)
                        m_block = m_by_valley[ieta][:, :, ks]
                        density_projected = lam @ m_block @ lam.conjugate().T
                        density_projected -= lam @ kblk.conjugate().T
                        density_projected -= kblk @ lam.conjugate().T
                        fock_block -= coeff * density_projected
                    correction[ispin, ieta, :, :, kt] += fock_block
                    fock_norm += float(np.linalg.norm(fock_block))
    correction = 0.5 * (correction + np.swapaxes(correction.conjugate(), 2, 3))
    p0_traces = [np.trace(p0_band, axis1=0, axis2=1).real for p0_band in p0_band_by_valley]
    p0_trace_all = np.concatenate([arr.reshape(-1) for arr in p0_traces])
    diagnostics = {
        "mode": 1.0,
        "projected_p0_trace_mean": float(np.mean(p0_trace_all)),
        "projected_p0_trace_min": float(np.min(p0_trace_all)),
        "projected_p0_trace_max": float(np.max(p0_trace_all)),
        "hartree_accumulated_norm_ev": float(hartree_norm),
        "fock_accumulated_norm_ev": float(fock_norm),
        "source_trace_abs_max": float(trace_abs_max),
        "h0_correction_norm_ev": float(np.linalg.norm(correction)),
        "h0_correction_max_abs_mev": float(1000.0 * np.max(np.abs(correction))),
        "zero_hartree_q0": float(1.0 if zero_hartree_q0 else 0.0),
        "include_active_reference": float(1.0 if include_active_reference else 0.0),
        "hartree_scale": float(hartree_scale),
        "fock_scale": float(fock_scale),
    }
    return correction, diagnostics


def basis_with_h0_correction(basis: PolshynProjectedBasis, correction_blocks: np.ndarray) -> PolshynProjectedBasis:
    correction = np.asarray(correction_blocks, dtype=np.complex128)
    if correction.shape != basis.h0_blocks.shape:
        raise ValueError(f"h0 correction shape {correction.shape} incompatible with {basis.h0_blocks.shape}")
    corrected = np.asarray(basis.h0_blocks, dtype=np.complex128) + correction
    corrected = 0.5 * (corrected + np.swapaxes(corrected.conjugate(), 2, 3))
    return replace(basis, h0_blocks=corrected)


def wang_stored_density_from_sector_blocks(density_blocks: np.ndarray) -> np.ndarray:
    """Convert conventional sector density blocks to Wang/Xiaoyu stored layout."""

    return flatten_sector_blocks(np.conj(np.asarray(density_blocks, dtype=np.complex128)))


def scaled_overlap_blocks(
    overlap_blocks: HFOverlapBlockSet,
    *,
    hartree_scale: float = 1.0,
    fock_scale: float = 1.0,
) -> HFOverlapBlockSet:
    """Return an overlap table with Hartree/Fock kernels rescaled for diagnostics."""

    return HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening={tuple(shift): float(hartree_scale) * float(value) for shift, value in overlap_blocks.hartree_screening.items()},
        fock_screening={tuple(shift): float(fock_scale) * np.asarray(value, dtype=float) for shift, value in overlap_blocks.fock_screening.items()},
    )


def overlap_blocks_with_hartree_q0_zeroed(overlap_blocks: HFOverlapBlockSet) -> HFOverlapBlockSet:
    """Return an overlap table with only the uniform Hartree shift removed."""

    hartree = {
        tuple(shift): (0.0 if tuple(shift) == (0, 0) else float(value))
        for shift, value in overlap_blocks.hartree_screening.items()
    }
    return HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening=hartree,
        fock_screening=overlap_blocks.fock_screening,
    )


def wang_interaction_blocks_from_sector_density(
    density_blocks: np.ndarray,
    basis: PolshynProjectedBasis,
    overlap_blocks: HFOverlapBlockSet,
    *,
    v0: float,
) -> np.ndarray:
    """Evaluate a generic Wang/Xiaoyu HF potential from conventional sector blocks."""

    flat_density = wang_stored_density_from_sector_blocks(density_blocks)
    flat_h = build_projected_interaction_hamiltonian(flat_density, overlap_blocks, v0=float(v0), beta=1.0)
    return unflatten_sector_blocks(flat_h, n_spin=basis.n_spin, n_eta=basis.n_eta, nb=basis.nb)


def block_density_norm(updated: np.ndarray, previous: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(updated) - np.asarray(previous)))
    denominator = float(np.linalg.norm(np.asarray(updated)))
    if denominator < 1e-15:
        return 0.0 if numerator < 1e-15 else float("inf")
    return numerator / denominator


def run_projected_hf_scf(
    basis: PolshynProjectedBasis,
    *,
    occupation_counts: np.ndarray,
    source_diagonals: dict[tuple[int, int], np.ndarray],
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    v0: float,
    epsilon_r: float,
    d_sc_nm: float,
    max_iter: int = 80,
    mixing: float = 0.5,
    precision: float = 1e-6,
    initial_density_blocks: np.ndarray | None = None,
    compact_overlaps: dict[tuple[int, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | bool | str]]:
    if initial_density_blocks is None:
        density, energies = density_from_fixed_sector_occupations(basis.h0_blocks, occupation_counts, basis.reference_diagonal)
        init_mode = "bm"
    else:
        density = np.asarray(initial_density_blocks, dtype=np.complex128).copy()
        _density_check, energies = density_from_fixed_sector_occupations(basis.h0_blocks, occupation_counts, basis.reference_diagonal)
        init_mode = "provided"
    final_interaction = np.zeros_like(basis.h0_blocks)
    final_norm = float("inf")
    converged = False
    iterations = 0
    for iteration in range(1, int(max_iter) + 1):
        interaction = build_interaction_blocks(
            basis,
            basis,
            density,
            source_diagonals=source_diagonals,
            target_diagonals=source_diagonals,
            shifts=shifts,
            gvecs=gvecs,
            v0=v0,
            epsilon_r=epsilon_r,
            d_sc_nm=d_sc_nm,
            include_hartree=True,
            include_fock=True,
            compact_overlaps=compact_overlaps,
        )
        trial_density, trial_energies = density_from_fixed_sector_occupations(
            basis.h0_blocks + interaction,
            occupation_counts,
            basis.reference_diagonal,
        )
        final_norm = block_density_norm(trial_density, density)
        density = float(mixing) * trial_density + (1.0 - float(mixing)) * density
        energies = trial_energies
        final_interaction = interaction
        iterations = iteration
        print(f"[polshyn-hf-scf] iter={iteration} raw_norm={final_norm:.6e}", flush=True)
        if final_norm <= float(precision):
            converged = True
            break
    final_interaction = build_interaction_blocks(
        basis,
        basis,
        density,
        source_diagonals=source_diagonals,
        target_diagonals=source_diagonals,
        shifts=shifts,
        gvecs=gvecs,
        v0=v0,
        epsilon_r=epsilon_r,
        d_sc_nm=d_sc_nm,
        include_hartree=True,
        include_fock=True,
        compact_overlaps=compact_overlaps,
    )
    _final_density, final_energies = density_from_fixed_sector_occupations(
        basis.h0_blocks + final_interaction,
        occupation_counts,
        basis.reference_diagonal,
    )
    info: dict[str, float | int | bool | str] = {
        "mode": "polshyn_projected_hf",
        "iterations": int(iterations),
        "converged": bool(converged),
        "final_raw_norm": float(final_norm),
        "mixing": float(mixing),
        "precision": float(precision),
        "init_mode": init_mode,
        "final_interaction_norm_ev": float(np.linalg.norm(final_interaction)),
    }
    return density, final_interaction, final_energies, info


def wang_projected_wavefunction_basis(basis: PolshynProjectedBasis) -> ProjectedWavefunctionBasis:
    """View a Polshyn projected basis in the generic Wang/Xiaoyu HF layout."""

    return ProjectedWavefunctionBasis(
        wavefunctions=np.asarray(basis.wavefunctions, dtype=np.complex128),
        grid_shape=tuple(basis.embedding_shape),
        n_spin=int(basis.n_spin),
        local_basis_size=int(basis.local_basis_size),
        name="polshyn_doubled",
        boundary_mode="zero_fill",
    )


def _flat_sector_indices(n_spin: int, n_eta: int, nb: int, ispin: int, ieta: int) -> np.ndarray:
    return _core_flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)

def flatten_sector_blocks(blocks: np.ndarray) -> np.ndarray:
    """Flatten (spin, valley, band, band, k) blocks to Wang nt x nt x nk layout."""

    return _core_flatten_sector_blocks(blocks)

def unflatten_sector_blocks(flat: np.ndarray, *, n_spin: int, n_eta: int, nb: int) -> np.ndarray:
    return _core_unflatten_sector_blocks(flat, n_spin=n_spin, n_eta=n_eta, nb=nb)

def unflatten_sector_energies(flat_energies: np.ndarray, *, n_spin: int, n_eta: int, nb: int) -> np.ndarray:
    return _core_unflatten_sector_energies(flat_energies, n_spin=n_spin, n_eta=n_eta, nb=nb)

def wang_density_from_fixed_sector_occupations(
    hamiltonian_flat: np.ndarray,
    occupation_counts: np.ndarray,
    reference_diagonal: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    nb: int,
) -> DensityUpdateResult:
    """Fixed-sector density update in Wang/Xiaoyu's stored-projector convention.

    The generic validated HF kernel stores the transpose/conjugate projector
    convention used by the original Wang/Xiaoyu code, namely ``P_store = P*``.
    This differs from the legacy Polshyn helper, which stored the conventional
    density matrix.  Keeping this builder separate prevents convention mixing.
    """

    h = np.asarray(hamiltonian_flat, dtype=np.complex128)
    nt, nt_rhs, nk = h.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square flattened Hamiltonian, got {h.shape}")
    if nt != int(n_spin) * int(n_eta) * int(nb):
        raise ValueError(f"Flattened dimension {nt} incompatible with {(n_spin, n_eta, nb)}")
    occ = np.asarray(occupation_counts, dtype=int)
    if occ.shape != (int(n_spin), int(n_eta)):
        raise ValueError(f"occupation_counts shape {occ.shape} incompatible with {(n_spin, n_eta)}")
    reference = np.asarray(reference_diagonal, dtype=float)
    if reference.shape != (int(nb),):
        raise ValueError(f"reference_diagonal shape {reference.shape} incompatible with nb={nb}")
    ref_mat = np.diag(reference).astype(np.complex128)
    density = np.zeros_like(h)
    energies = np.zeros((nt, int(nk)), dtype=float)
    sector_energies = np.zeros((int(n_spin), int(n_eta), int(nb), int(nk)), dtype=float)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            idx = _flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)
            n_occ = int(occ[ispin, ieta])
            for ik in range(int(nk)):
                block_h = h[idx[:, None], idx[None, :], ik]
                block_h = 0.5 * (block_h + block_h.conjugate().T)
                evals, evecs = np.linalg.eigh(block_h)
                energies[idx, ik] = evals
                sector_energies[ispin, ieta, :, ik] = evals
                if n_occ == 0:
                    projector = np.zeros((int(nb), int(nb)), dtype=np.complex128)
                elif n_occ == int(nb):
                    projector = np.eye(int(nb), dtype=np.complex128)
                else:
                    vecs = evecs[:, :n_occ]
                    projector = vecs.conj() @ vecs.T
                density[idx[:, None], idx[None, :], ik] = projector - ref_mat
    mu = estimate_fermi_level_from_sector_energies(sector_energies, occ)
    return DensityUpdateResult(density=density, energies=energies, mu=float(mu))


def build_wang_overlap_blocks(
    target: PolshynProjectedBasis,
    source: PolshynProjectedBasis,
    shifts: Iterable[tuple[int, int]],
    gvecs: np.ndarray,
    *,
    epsilon_r: float,
    d_sc_nm: float,
    include_hartree: bool = True,
    include_fock: bool = True,
    progress_prefix: str | None = None,
) -> HFOverlapBlockSet:
    """Build dense generic HF overlap blocks for the Wang/Xiaoyu engine."""

    target_core = wang_projected_wavefunction_basis(target)
    source_core = wang_projected_wavefunction_basis(source)
    shift_tuple = tuple(tuple(shift) for shift in shifts)
    gvec_array = np.asarray(gvecs, dtype=np.complex128)
    overlaps: dict[tuple[int, int], np.ndarray] = {}
    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for ishift, (shift, gvec) in enumerate(zip(shift_tuple, gvec_array, strict=True), start=1):
        if progress_prefix and (ishift == 1 or ishift == len(shift_tuple) or ishift % 10 == 0):
            print(f"{progress_prefix} wang overlap {ishift}/{len(shift_tuple)} shift={shift}", flush=True)
        overlap = calculate_projected_overlap_between(target_core, source_core, int(shift[0]), int(shift[1]))
        overlaps[shift] = overlap
        if target.nk == source.nk:
            diagonal_overlaps[shift] = diagonal_overlap_blocks(overlap, nt=target_core.nt, nk=target.nk)
        if include_hartree:
            hartree_screening[shift] = screened_coulomb(complex(gvec), epsilon_r=float(epsilon_r), d_sc_nm=float(d_sc_nm))
        if include_fock:
            fock_screening[shift] = screened_coulomb_matrix(
                source.kvec[None, :] - target.kvec[:, None] + complex(gvec),
                epsilon_r=float(epsilon_r),
                d_sc_nm=float(d_sc_nm),
            )
    return HFOverlapBlockSet(
        shifts=shift_tuple,
        gvecs=gvec_array,
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def run_projected_hf_scf_wang(
    basis: PolshynProjectedBasis,
    *,
    occupation_counts: np.ndarray,
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    v0: float,
    epsilon_r: float,
    d_sc_nm: float,
    max_iter: int = 80,
    precision: float = 1e-6,
    initial_density_blocks: np.ndarray | None = None,
    oda_stall_threshold: float = 1.0e-4,
    progress_prefix: str | None = None,
    overlap_blocks: HFOverlapBlockSet | None = None,
    seed: int = 0,
    hartree_scale: float = 1.0,
    fock_scale: float = 1.0,
    zero_hartree_q0: bool = False,
) -> tuple[PolshynWangHFState, HFOverlapBlockSet, dict[str, float | int | bool | str]]:
    """Run Polshyn projected HF through the generic Wang/Xiaoyu ODA engine."""

    if overlap_blocks is None:
        overlap_blocks = build_wang_overlap_blocks(
            basis,
            basis,
            shifts,
            gvecs,
            epsilon_r=epsilon_r,
            d_sc_nm=d_sc_nm,
            include_hartree=True,
            include_fock=True,
            progress_prefix=progress_prefix,
        )
    if float(hartree_scale) != 1.0 or float(fock_scale) != 1.0:
        overlap_blocks = scaled_overlap_blocks(
            overlap_blocks,
            hartree_scale=float(hartree_scale),
            fock_scale=float(fock_scale),
        )
    if bool(zero_hartree_q0):
        overlap_blocks = overlap_blocks_with_hartree_q0_zeroed(overlap_blocks)
    h0_flat = flatten_sector_blocks(basis.h0_blocks)
    if initial_density_blocks is None:
        init_update = wang_density_from_fixed_sector_occupations(
            h0_flat,
            occupation_counts,
            basis.reference_diagonal,
            n_spin=basis.n_spin,
            n_eta=basis.n_eta,
            nb=basis.nb,
        )
        density_flat = init_update.density
        energies = init_update.energies
        mu = init_update.mu
        init_mode = "bm_wang"
    else:
        # Existing initializers create conventional Hermitian density matrices;
        # Wang/Xiaoyu's kernel stores P* instead.
        density_flat = flatten_sector_blocks(np.conj(np.asarray(initial_density_blocks, dtype=np.complex128)))
        init_update = wang_density_from_fixed_sector_occupations(
            h0_flat,
            occupation_counts,
            basis.reference_diagonal,
            n_spin=basis.n_spin,
            n_eta=basis.n_eta,
            nb=basis.nb,
        )
        energies = init_update.energies
        mu = init_update.mu
        init_mode = "provided_wang"

    state = PolshynWangHFState(
        h0=h0_flat.copy(),
        density=density_flat.copy(),
        hamiltonian=h0_flat.copy(),
        energies=np.asarray(energies, dtype=float).copy(),
        mu=float(mu),
        precision=float(precision),
        v0=float(v0),
        diagnostics={},
    )

    def interaction_builder(density_flat_in: np.ndarray) -> np.ndarray:
        return build_projected_interaction_hamiltonian(
            density_flat_in,
            overlap_blocks,
            v0=float(v0),
            beta=1.0,
        )

    def density_builder(hamiltonian_flat: np.ndarray) -> DensityUpdateResult:
        return wang_density_from_fixed_sector_occupations(
            hamiltonian_flat,
            occupation_counts,
            basis.reference_diagonal,
            n_spin=basis.n_spin,
            n_eta=basis.n_eta,
            nb=basis.nb,
        )

    run = run_hartree_fock_iterations(
        state,
        init_mode=init_mode,
        seed=int(seed),
        interaction_builder=interaction_builder,
        density_builder=density_builder,
        energy_functional=compute_hf_energy,
        oda_delta_interaction_builder=interaction_builder,
        convergence_rule="mixed",
        max_iter=int(max_iter),
        oda_stall_threshold=float(oda_stall_threshold),
    )
    info: dict[str, float | int | bool | str] = {
        "mode": "polshyn_projected_hf_wang",
        "iterations": int(run.iterations),
        "converged": bool(run.converged),
        "exit_reason": str(run.exit_reason),
        "final_raw_norm": float(state.diagnostics.get("final_raw_norm", float("nan"))),
        "init_mode": init_mode,
        "precision": float(precision),
        "oda_stall_threshold": float(oda_stall_threshold),
        "final_interaction_norm_ev": float(np.linalg.norm(state.hamiltonian - state.h0)),
        "hf_energy": float(state.diagnostics.get("hf_energy", float("nan"))),
        "hartree_scale": float(hartree_scale),
        "fock_scale": float(fock_scale),
        "zero_hartree_q0": bool(zero_hartree_q0),
    }
    if run.iter_oda.size:
        info["last_oda_lambda"] = float(run.iter_oda[-1])
        info["min_oda_lambda"] = float(np.min(run.iter_oda))
    return state, overlap_blocks, info


def wang_sector_density_blocks(state: PolshynWangHFState, basis: PolshynProjectedBasis) -> np.ndarray:
    """Return conventional sector density blocks from a Wang/Xiaoyu HF state."""

    return np.conj(unflatten_sector_blocks(state.density, n_spin=basis.n_spin, n_eta=basis.n_eta, nb=basis.nb))


def wang_sector_hamiltonian_blocks(state: PolshynWangHFState, basis: PolshynProjectedBasis) -> np.ndarray:
    return unflatten_sector_blocks(state.hamiltonian, n_spin=basis.n_spin, n_eta=basis.n_eta, nb=basis.nb)


def wang_sector_energy_blocks(state: PolshynWangHFState, basis: PolshynProjectedBasis) -> np.ndarray:
    return unflatten_sector_energies(state.energies, n_spin=basis.n_spin, n_eta=basis.n_eta, nb=basis.nb)


def wang_target_hamiltonian(
    target: PolshynProjectedBasis,
    source: PolshynProjectedBasis,
    source_state: PolshynWangHFState,
    source_overlap_blocks: HFOverlapBlockSet,
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    *,
    v0: float,
    epsilon_r: float,
    d_sc_nm: float,
    progress_prefix: str | None = None,
) -> tuple[np.ndarray, HFOverlapBlockSet, HFOverlapBlockSet]:
    """Evaluate a Wang/Xiaoyu HF Hamiltonian on a target basis from a source density."""

    target_overlap_blocks = build_wang_overlap_blocks(
        target,
        target,
        shifts,
        gvecs,
        epsilon_r=epsilon_r,
        d_sc_nm=d_sc_nm,
        include_hartree=True,
        include_fock=False,
        progress_prefix=None if progress_prefix is None else f"{progress_prefix} target",
    )
    target_source_overlap_blocks = build_wang_overlap_blocks(
        target,
        source,
        shifts,
        gvecs,
        epsilon_r=epsilon_r,
        d_sc_nm=d_sc_nm,
        include_hartree=False,
        include_fock=True,
        progress_prefix=None if progress_prefix is None else f"{progress_prefix} target-source",
    )
    h0_target = flatten_sector_blocks(target.h0_blocks)
    h_target = build_projected_target_hamiltonian(
        h0_target,
        source_state.density,
        source_overlap_blocks=source_overlap_blocks,
        target_overlap_blocks=target_overlap_blocks,
        target_source_overlap_blocks=target_source_overlap_blocks,
        v0=float(v0),
        beta=1.0,
    )
    return h_target, target_overlap_blocks, target_source_overlap_blocks


def wang_sector_energies_from_flat_hamiltonian(
    hamiltonian_flat: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    nb: int,
) -> np.ndarray:
    h = np.asarray(hamiltonian_flat, dtype=np.complex128)
    nk = h.shape[2]
    out = np.zeros((int(n_spin), int(n_eta), int(nb), int(nk)), dtype=float)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            idx = _flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)
            for ik in range(int(nk)):
                block_h = h[idx[:, None], idx[None, :], ik]
                block_h = 0.5 * (block_h + block_h.conjugate().T)
                out[ispin, ieta, :, ik] = np.linalg.eigvalsh(block_h)
    return out


def translation_order_parameters(
    density_blocks: np.ndarray,
    *,
    projected_indices: tuple[int, ...],
    target_band_index: int,
    spin_index: int = 0,
    valley_index: int = 0,
) -> dict[str, np.ndarray | float]:
    """Fold-off-diagonal CDW order diagnostic for the doubled supercell.

    In the area-2 folded basis, fold 0 and fold 1 differ by the paper's
    translation-breaking wavevector Q=B1.  The quantity analogous to Polshyn
    Eq. (1) is therefore the norm of density-matrix elements connecting even
    and odd folded copies at the same supercell k.  A single target-band state
    (|fold0>+|fold1>)/sqrt(2) has |rho_01|=1/2, so the reported ``*_x2``
    values are normalized to have maximal target-band order near one.
    """

    density = np.asarray(density_blocks, dtype=np.complex128)
    projected_indices = tuple(int(index) for index in projected_indices)
    target_pos = projected_indices.index(int(target_band_index))
    fold0 = np.asarray([2 * iprim for iprim in range(len(projected_indices))], dtype=int)
    fold1 = fold0 + 1
    sector = density[int(spin_index), int(valley_index)]
    target_raw = np.abs(sector[2 * target_pos, 2 * target_pos + 1, :])
    all_raw = np.sqrt(np.sum(np.abs(sector[np.ix_(fold0, fold1, np.arange(sector.shape[-1]))]) ** 2, axis=(0, 1)))
    return {
        "target_raw": np.asarray(target_raw, dtype=float),
        "all_raw": np.asarray(all_raw, dtype=float),
        "target_x2": np.asarray(2.0 * target_raw, dtype=float),
        "all_x2": np.asarray(2.0 * all_raw, dtype=float),
        "target_x2_min": float(np.min(2.0 * target_raw)),
        "target_x2_mean": float(np.mean(2.0 * target_raw)),
        "target_x2_max": float(np.max(2.0 * target_raw)),
        "all_x2_min": float(np.min(2.0 * all_raw)),
        "all_x2_mean": float(np.mean(2.0 * all_raw)),
        "all_x2_max": float(np.max(2.0 * all_raw)),
    }


def path_sector_energies(h_blocks: np.ndarray) -> np.ndarray:
    return sector_block_energies(h_blocks)

def estimate_fermi_level_from_sector_energies(energies: np.ndarray, occupation_counts: np.ndarray) -> float:
    vals = np.asarray(energies, dtype=float)
    occ = np.asarray(occupation_counts, dtype=int)
    occupied_max: list[float] = []
    empty_min: list[float] = []
    for ispin in range(vals.shape[0]):
        for ieta in range(vals.shape[1]):
            n_occ = int(occ[ispin, ieta])
            if n_occ > 0:
                occupied_max.append(float(np.max(vals[ispin, ieta, n_occ - 1, :])))
            if n_occ < vals.shape[2]:
                empty_min.append(float(np.min(vals[ispin, ieta, n_occ, :])))
    if occupied_max and empty_min:
        return 0.5 * (max(occupied_max) + min(empty_min))
    if occupied_max:
        return max(occupied_max)
    if empty_min:
        return min(empty_min)
    return 0.0


def moire_cell_area_nm2(lattice: TMBGLattice, *, area_ratio: int = 1) -> float:
    primitive_area = real_space_cell_area_nm2_from_reciprocal(lattice.g_m1, lattice.g_m2)
    return float(area_ratio) * float(primitive_area)


def max_hermitian_error(blocks: np.ndarray) -> float:
    arr = np.asarray(blocks, dtype=np.complex128)
    return float(np.max(np.abs(arr - np.swapaxes(arr.conjugate(), 2, 3))))
