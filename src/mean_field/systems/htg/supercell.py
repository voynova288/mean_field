from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import math

import numpy as np

from ...core.hf import (
    DensityUpdateResult,
    HFOverlapBlockSet,
    HartreeFockProblem,
    HartreeFockRun,
    ProjectedWavefunctionBasis,
    build_projected_hf_kernel,
    build_projected_hf_problem,
    build_projected_interaction_hamiltonian,
    build_projected_target_hamiltonian,
    calculate_projected_overlap_between,
    compute_hf_energy,
    find_chemical_potential,
    occupied_state_mask,
    random_unitary_from_hermitian,
    real_space_cell_area_nm2_from_reciprocal,
    run_hartree_fock_problem,
    screened_coulomb_matrix,
)
from ...core.supercell import IntegerSupercell, folded_band_count, occupied_count_from_primitive_filling
from .hamiltonian import centered_band_indices
from .lattice import HTGLattice, KPath, build_kpath_from_nodes
from .mean_field_adapter import (
    _hybrid_projected_basis_at_k,
    _layer_potential_operator,
    centered_projection_band_indices,
    hermitian_residual,
    projector_idempotency_residual,
    reciprocal_shift_labels,
)
from .model import HTGModel
from .params import InteractionParams
from .topology import sublattice_sigma_z


@dataclass(frozen=True)
class HTGSupercell(IntegerSupercell):
    """Integer supercell for folded-BZ HTG projected Hartree-Fock."""

    def reciprocal_vectors(self, lattice: HTGLattice) -> tuple[complex, complex]:
        return super().reciprocal_vectors(lattice.b_m1, lattice.b_m2)


@dataclass(frozen=True)
class HTGSupercellProjectedBasisData:
    model: HTGModel
    interaction: InteractionParams
    supercell: HTGSupercell
    mesh_size: int
    kvec: np.ndarray
    k_grid_frac: np.ndarray | None
    basis: ProjectedWavefunctionBasis
    h0: np.ndarray
    sigma_z: np.ndarray
    band_sigma_z: np.ndarray
    primitive_projected_indices: tuple[int, ...]
    primitive_band_count: int
    fold_representatives: tuple[tuple[int, int], ...]
    reference_diagonal: np.ndarray
    super_g1: complex
    super_g2: complex
    reciprocal_grid_shape: tuple[int, int]
    reciprocal_grid_origin: tuple[int, int]
    moire_supercell_area_nm2: float

    @property
    def nk(self) -> int:
        return int(self.kvec.size)

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nb(self) -> int:
        return int(self.basis.n_band)


@dataclass
class HTGSupercellHartreeFockState:
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    nu: float
    reference_diagonal: np.ndarray
    v0: float
    mu: float = float("nan")
    precision: float = 1.0e-6
    n_spin: int = 2
    n_eta: int = 2
    n_band: int = 12
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @classmethod
    def from_projected_basis(
        cls,
        basis_data: HTGSupercellProjectedBasisData,
        *,
        nu: float,
        precision: float = 1.0e-6,
    ) -> "HTGSupercellHartreeFockState":
        h0 = np.asarray(basis_data.h0, dtype=np.complex128).copy()
        nt, _, nk = h0.shape
        return cls(
            h0=h0,
            density=np.zeros((nt, nt, nk), dtype=np.complex128),
            hamiltonian=h0.copy(),
            energies=np.zeros((nt, nk), dtype=float),
            nu=float(nu),
            reference_diagonal=np.asarray(basis_data.reference_diagonal, dtype=float).copy(),
            v0=1.0 / float(basis_data.moire_supercell_area_nm2),
            precision=float(precision),
            n_spin=int(basis_data.basis.n_spin),
            n_eta=int(basis_data.basis.n_flavor),
            n_band=int(basis_data.basis.n_band),
        )


@dataclass(frozen=True)
class HTGSupercellHartreeFockRun(HartreeFockRun):
    state: HTGSupercellHartreeFockState
    overlap_blocks: HFOverlapBlockSet
    basis_data: HTGSupercellProjectedBasisData


@dataclass(frozen=True)
class HTGSupercellGroundStateScan:
    runs: tuple[HTGSupercellHartreeFockRun, ...]

    @property
    def best_run(self) -> HTGSupercellHartreeFockRun:
        if not self.runs:
            raise ValueError("No HTG supercell HF runs are available")
        return min(self.runs, key=lambda run: float(run.state.diagnostics.get("hf_energy", np.inf)))


@dataclass(frozen=True)
class HTGSupercellPathResult:
    path: KPath
    hamiltonian: np.ndarray
    energies: np.ndarray
    mu: float
    nu: float
    init_mode: str
    seed: int
    exit_reason: str
    points_per_segment: int


@dataclass(frozen=True)
class HTGSupercellSCFGridPathSamples:
    """Exact saved SCF-grid samples lying on a folded-BZ path."""

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

@dataclass(frozen=True)
class HTGSupercellHFWavefunctionGrid:
    """Full physical wavefunction mesh reconstructed from a supercell HF Hamiltonian."""

    wavefunctions: np.ndarray
    energies: np.ndarray
    k_grid_frac: np.ndarray
    band_indices: tuple[int, ...]
    basis_data: HTGSupercellProjectedBasisData

def htg_tripled_fractional_supercell() -> HTGSupercell:
    """Area-3 sqrt(3) x sqrt(3) cell for one-third/two-third fillings."""

    return HTGSupercell(n11=1, n12=1, n21=-1, n22=2)

def htg_doubled_fractional_supercell() -> HTGSupercell:
    """Area-2 doubled cell for half filling, matching the Polshyn-style convention."""

    return HTGSupercell(n11=2, n12=1, n21=0, n22=1)

def htg_common_area6_fractional_supercell() -> HTGSupercell:
    """Legacy common area-6 rectangular cell; use only when an area-6 cell is intended."""

    return HTGSupercell(n11=3, n12=0, n21=0, n22=2)

def htg_default_fractional_supercell() -> HTGSupercell:
    """Backward-compatible area-6 helper.

    This is not the default for new fractional HTG runs.  Use
    :func:`htg_minimal_fractional_supercell` when the filling is known, because
    nu=3+1/2 should use an area-2 cell while nu=3+1/3 and 3+2/3 should use an
    area-3 cell.
    """

    return htg_common_area6_fractional_supercell()

def htg_minimal_fractional_supercell(primitive_nu: float, *, max_denominator: int = 12) -> HTGSupercell:
    """Choose the minimal intended HTG folded cell from the filling denominator."""

    fraction = Fraction(float(primitive_nu)).limit_denominator(int(max_denominator))
    denominator = int(fraction.denominator)
    if denominator == 1:
        return HTGSupercell(n11=1, n12=0, n21=0, n22=1)
    if denominator == 2:
        return htg_doubled_fractional_supercell()
    if denominator == 3:
        return htg_tripled_fractional_supercell()
    if denominator == 6:
        return htg_common_area6_fractional_supercell()
    raise ValueError(
        f"No built-in HTG supercell for primitive_nu={primitive_nu} (denominator {denominator}); "
        "pass an explicit supercell."
    )


def _integer_matrix(supercell: HTGSupercell) -> np.ndarray:
    return np.asarray(
        [[int(supercell.n11), int(supercell.n12)], [int(supercell.n21), int(supercell.n22)]],
        dtype=int,
    )


def _equivalent_mod_primitive(diff: tuple[int, int], supercell: HTGSupercell, *, atol: float = 1.0e-9) -> bool:
    matrix = _integer_matrix(supercell).astype(float)
    coeff = np.linalg.solve(matrix, np.asarray(diff, dtype=float))
    return bool(np.all(np.abs(coeff - np.rint(coeff)) <= float(atol)))


def supercell_fold_representatives(supercell: HTGSupercell) -> tuple[tuple[int, int], ...]:
    """Coset representatives of supercell reciprocal vectors modulo primitive ones."""

    if int(supercell.n12) == 0 and int(supercell.n21) == 0:
        return tuple(
            (ix, iy)
            for ix in range(int(supercell.n11))
            for iy in range(int(supercell.n22))
        )

    target_count = int(supercell.area_ratio)
    reps: list[tuple[int, int]] = []
    bound = max(target_count, abs(supercell.n11), abs(supercell.n12), abs(supercell.n21), abs(supercell.n22), 1)
    for radius in range(0, 4 * bound + 8):
        for sx in range(-radius, radius + 1):
            for sy in range(-radius, radius + 1):
                candidate = (int(sx), int(sy))
                if any(_equivalent_mod_primitive((candidate[0] - old[0], candidate[1] - old[1]), supercell) for old in reps):
                    continue
                reps.append(candidate)
                if len(reps) == target_count:
                    reps.sort(key=lambda item: (item[0] * item[0] + item[1] * item[1], item[0], item[1]))
                    return tuple(reps)
    raise ValueError(f"Could not construct {target_count} fold representatives for {supercell}")


def build_htg_supercell_uniform_grid(
    lattice: HTGLattice,
    supercell: HTGSupercell,
    mesh: int,
    *,
    endpoint: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    super_g1, super_g2 = supercell.reciprocal_vectors(lattice)
    mesh = int(mesh)
    if mesh <= 0:
        raise ValueError(f"mesh must be positive, got {mesh}")
    if endpoint:
        frac = np.linspace(0.0, 1.0, mesh, dtype=float)
    else:
        frac = np.arange(mesh, dtype=float) / float(mesh)
    f1, f2 = np.meshgrid(frac, frac, indexing="ij")
    kvec = f1 * super_g1 + f2 * super_g2
    frac_grid = np.stack([f1, f2], axis=-1)
    return np.asarray(frac_grid, dtype=float), np.asarray(kvec.reshape(-1), dtype=np.complex128)


def extract_htg_supercell_scf_grid_path(
    k_grid_frac: np.ndarray,
    *,
    super_g1: complex,
    super_g2: complex,
    node_frac: Sequence[Sequence[float]],
    labels: Sequence[str],
    exact_tolerance: float = 1.0e-10,
) -> HTGSupercellSCFGridPathSamples:
    """Extract exact saved SCF-grid points on a path in supercell fractional coordinates.

    The returned indices refer to the saved SCF arrays flattened in the same C
    order used by :func:`build_htg_supercell_uniform_grid`.  This function does
    not evaluate any off-grid Hamiltonian.
    """

    grid = np.asarray(k_grid_frac, dtype=float)
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError(f"Expected k_grid_frac shape (n1, n2, 2), got {grid.shape}")
    nodes = np.asarray(node_frac, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"Expected node_frac shape (nnode, 2), got {nodes.shape}")
    resolved_labels = tuple(str(label) for label in labels)
    if len(resolved_labels) != nodes.shape[0]:
        raise ValueError("labels and node_frac must have the same length")

    node_kvec = nodes[:, 0] * complex(super_g1) + nodes[:, 1] * complex(super_g2)
    node_kdist = np.zeros(nodes.shape[0], dtype=float)
    for inode in range(1, nodes.shape[0]):
        node_kdist[inode] = node_kdist[inode - 1] + float(abs(node_kvec[inode] - node_kvec[inode - 1]))

    tol = float(exact_tolerance)
    n1, n2 = int(grid.shape[0]), int(grid.shape[1])
    selected: list[tuple[float, int, float, float, int]] = []
    for iseg in range(nodes.shape[0] - 1):
        start = nodes[iseg]
        end = nodes[iseg + 1]
        segment = end - start
        segment_norm2 = float(np.dot(segment, segment))
        if segment_norm2 <= 0.0:
            continue
        segment_length = float(abs(node_kvec[iseg + 1] - node_kvec[iseg]))
        rows: list[tuple[float, int, float, float, int]] = []
        for ix in range(n1):
            for iy in range(n2):
                point = np.asarray(grid[ix, iy], dtype=float)
                diff = point - start
                cross = float(diff[0] * segment[1] - diff[1] * segment[0])
                t = float(np.dot(diff, segment) / segment_norm2)
                if abs(cross) > tol or t < -tol or t > 1.0 + tol:
                    continue
                projection = start + min(1.0, max(0.0, t)) * segment
                if float(np.linalg.norm(point - projection)) > tol:
                    continue
                index = int(ix * n2 + iy)
                kdist = float(node_kdist[iseg] + min(1.0, max(0.0, t)) * segment_length)
                rows.append((kdist, index, float(point[0]), float(point[1]), int(iseg)))
        rows.sort(key=lambda row: (row[0], row[1]))
        selected.extend(rows)

    if not selected:
        raise ValueError("No exact SCF-grid points were found on the requested HTG supercell path")

    flat_grid = grid.reshape((-1, 2))
    node_hits = np.asarray(
        [np.min(np.linalg.norm(flat_grid - node[None, :], axis=1)) <= tol for node in nodes],
        dtype=bool,
    )
    return HTGSupercellSCFGridPathSamples(
        kdist=np.asarray([row[0] for row in selected], dtype=float),
        grid_indices=np.asarray([row[1] for row in selected], dtype=int),
        frac_coords=np.asarray([[row[2], row[3]] for row in selected], dtype=float),
        segment_indices=np.asarray([row[4] for row in selected], dtype=int),
        node_kdist=np.asarray(node_kdist, dtype=float),
        labels=resolved_labels,
        exact_node_hit_mask=node_hits,
        exact_tolerance=float(exact_tolerance),
    )

def extract_htg_supercell_inspection_scf_grid_path(
    basis_data: HTGSupercellProjectedBasisData,
    *,
    exact_tolerance: float = 1.0e-10,
) -> HTGSupercellSCFGridPathSamples:
    """Exact SCF-grid diagnostic path kept inside the saved folded-BZ tile."""

    if basis_data.k_grid_frac is None:
        raise ValueError("basis_data does not contain a tensor-product SCF grid")
    node_frac = ((0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0), (0.5, 0.0), (0.0, 0.0))
    labels = ("Gamma_s", "kappa_s", "M_s", "Gamma_s")
    return extract_htg_supercell_scf_grid_path(
        basis_data.k_grid_frac,
        super_g1=basis_data.super_g1,
        super_g2=basis_data.super_g2,
        node_frac=node_frac,
        labels=labels,
        exact_tolerance=exact_tolerance,
    )

def htg_supercell_full_boundary_sewing_transform(basis_data: HTGSupercellProjectedBasisData, dm: int, dn: int):
    """Boundary sewing for reconstructed full HTG supercell wavefunctions.

    This is a system adapter: it expresses how the plane-wave embedding grid is
    relabelled when the supercell crystal momentum crosses a reciprocal-periodic
    boundary.  The Berry-link and Chern formulas are delegated to
    :mod:`analysis.topology`.
    """

    n_spin = int(basis_data.basis.n_spin)
    n_eta = int(basis_data.basis.n_flavor)
    local_size = int(basis_data.basis.local_basis_size)
    nx, ny = (int(value) for value in basis_data.basis.grid_shape)
    shift_x = int(dm)
    shift_y = int(dn)

    def apply(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.complex128)
        has_columns = array.ndim == 2
        if has_columns:
            n_column = int(array.shape[1])
            reshaped = array.reshape((n_spin, n_eta, local_size, nx, ny, n_column))
            out = np.zeros_like(reshaped)
        else:
            reshaped = array.reshape((n_spin, n_eta, local_size, nx, ny))
            out = np.zeros_like(reshaped)
        for ix in range(nx):
            source_x = ix + shift_x
            if source_x < 0 or source_x >= nx:
                continue
            for iy in range(ny):
                source_y = iy + shift_y
                if source_y < 0 or source_y >= ny:
                    continue
                if has_columns:
                    out[:, :, :, ix, iy, :] = reshaped[:, :, :, source_x, source_y, :]
                else:
                    out[:, :, :, ix, iy] = reshaped[:, :, :, source_x, source_y]
        return out.reshape(array.shape)

    return apply

def htg_supercell_full_boundary_sewing_transforms(basis_data: HTGSupercellProjectedBasisData):
    return (
        htg_supercell_full_boundary_sewing_transform(basis_data, 1, 0),
        htg_supercell_full_boundary_sewing_transform(basis_data, 0, 1),
    )

def _htg_supercell_full_wavefunction_from_coefficients(
    basis_data: HTGSupercellProjectedBasisData,
    coefficients: np.ndarray,
    ik: int,
) -> np.ndarray:
    n_spin = int(basis_data.basis.n_spin)
    n_eta = int(basis_data.basis.n_flavor)
    n_band = int(basis_data.basis.n_band)
    basis_dimension = int(basis_data.basis.basis_dimension)
    local_size = int(basis_data.basis.local_basis_size)
    nx, ny = basis_data.basis.grid_shape
    coeff = np.asarray(coefficients, dtype=np.complex128).reshape((n_spin, n_eta, n_band), order="F")
    out = np.zeros((n_spin, n_eta, basis_dimension), dtype=np.complex128)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            out[ispin, ieta, :] = basis_data.basis.wavefunctions[:, :, ieta, int(ik)] @ coeff[ispin, ieta, :]
    return out.reshape((n_spin, n_eta, local_size, int(nx), int(ny))).reshape(-1)

def build_htg_supercell_hf_wavefunction_grid(
    hamiltonian: np.ndarray,
    basis_data: HTGSupercellProjectedBasisData,
    *,
    band_indices: Iterable[int],
) -> HTGSupercellHFWavefunctionGrid:
    """Reconstruct full wavefunction columns for selected HF bands on the SCF grid."""

    if basis_data.k_grid_frac is None:
        raise ValueError("basis_data must contain a tensor-product SCF k-grid for topology")
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = hamiltonian.shape
    if nt != nt_rhs or nt != basis_data.nt or nk != basis_data.nk:
        raise ValueError(f"Hamiltonian shape {hamiltonian.shape} is incompatible with basis_data nt/nk {(basis_data.nt, basis_data.nk)}")
    bands = tuple(int(index) for index in band_indices)
    if not bands:
        raise ValueError("At least one HF band index is required")
    if min(bands) < 0 or max(bands) >= nt:
        raise ValueError(f"HF band indices {bands} outside [0, {nt})")
    k_grid = np.asarray(basis_data.k_grid_frac, dtype=float)
    mesh1, mesh2 = int(k_grid.shape[0]), int(k_grid.shape[1])
    if int(mesh1 * mesh2) != nk:
        raise ValueError(f"SCF k-grid shape {k_grid.shape} is incompatible with nk={nk}")
    full_dim = int(basis_data.basis.n_spin * basis_data.basis.n_flavor * basis_data.basis.basis_dimension)
    wavefunctions = np.zeros((mesh1, mesh2, full_dim, len(bands)), dtype=np.complex128)
    energies = np.zeros((len(bands), mesh1, mesh2), dtype=float)
    for ix in range(mesh1):
        for iy in range(mesh2):
            ik = int(ix * mesh2 + iy)
            evals, evecs = np.linalg.eigh(hamiltonian[:, :, ik])
            for out_index, band_index in enumerate(bands):
                vector = _htg_supercell_full_wavefunction_from_coefficients(basis_data, evecs[:, band_index], ik)
                norm = float(np.vdot(vector, vector).real)
                wavefunctions[ix, iy, :, out_index] = vector / np.sqrt(max(norm, 1.0e-300))
                energies[out_index, ix, iy] = float(evals[band_index])
    return HTGSupercellHFWavefunctionGrid(
        wavefunctions=wavefunctions,
        energies=energies,
        k_grid_frac=k_grid,
        band_indices=bands,
        basis_data=basis_data,
    )

def _primitive_fractional_coords(lattice: HTGLattice, k_tilde: complex) -> tuple[float, float]:
    matrix = np.asarray(
        [
            [float(complex(lattice.b_m1).real), float(complex(lattice.b_m2).real)],
            [float(complex(lattice.b_m1).imag), float(complex(lattice.b_m2).imag)],
        ],
        dtype=float,
    )
    rhs = np.asarray([float(complex(k_tilde).real), float(complex(k_tilde).imag)], dtype=float)
    coeff = np.linalg.solve(matrix, rhs)
    return float(coeff[0]), float(coeff[1])


def _centered_primitive_reduction(lattice: HTGLattice, k_tilde: complex) -> tuple[complex, tuple[int, int]]:
    f1, f2 = _primitive_fractional_coords(lattice, complex(k_tilde))
    s1 = int(np.floor(f1 + 0.5))
    s2 = int(np.floor(f2 + 0.5))
    reduced = complex(k_tilde - s1 * lattice.b_m1 - s2 * lattice.b_m2)
    return reduced, (s1, s2)


def _supercell_embedding_table(
    lattice: HTGLattice,
    supercell: HTGSupercell,
    fold_reps: tuple[tuple[int, int], ...],
    *,
    primitive_shift_padding: int = 2,
) -> tuple[tuple[int, int], tuple[int, int], dict[tuple[int, int], tuple[int, int]]]:
    coords: set[tuple[int, int]] = set()
    pad = int(primitive_shift_padding)
    for n1, n2 in np.asarray(lattice.g_indices, dtype=int):
        for s1 in range(-pad, pad + 1):
            for s2 in range(-pad, pad + 1):
                primitive_coords = supercell.primitive_shift_to_supercell(int(n1) - s1, int(n2) - s2)
                for fx, fy in fold_reps:
                    coords.add((int(fx + primitive_coords[0]), int(fy + primitive_coords[1])))
    if not coords:
        raise ValueError("empty HTG supercell embedding table")
    min_x = min(item[0] for item in coords)
    max_x = max(item[0] for item in coords)
    min_y = min(item[1] for item in coords)
    max_y = max(item[1] for item in coords)
    shape = (int(max_x - min_x + 1), int(max_y - min_y + 1))
    origin = (int(min_x), int(min_y))
    positions = {(sx, sy): (int(sx - min_x), int(sy - min_y)) for sx, sy in coords}
    return shape, origin, positions


def htg_supercell_reference_diagonal(primitive_projected_band_count: int, area_ratio: int) -> np.ndarray:
    primitive_projected_band_count = int(primitive_projected_band_count)
    area_ratio = int(area_ratio)
    if primitive_projected_band_count < 2 or primitive_projected_band_count % 2 != 0:
        raise ValueError(f"primitive_projected_band_count must be an even integer >= 2, got {primitive_projected_band_count}")
    lower_count = (primitive_projected_band_count - 2) // 2
    primitive_reference = np.zeros(primitive_projected_band_count, dtype=float)
    primitive_reference[:lower_count] = 1.0
    primitive_reference[lower_count : lower_count + 2] = 0.5
    return np.repeat(primitive_reference, area_ratio).astype(float)


def htg_supercell_occupied_count_per_k(
    primitive_nu: float,
    *,
    reference_diagonal: np.ndarray,
    area_ratio: int,
    n_sector: int = 4,
    atol: float = 1.0e-9,
) -> int:
    return occupied_count_from_primitive_filling(
        float(primitive_nu),
        reference_diagonal=np.asarray(reference_diagonal, dtype=float),
        area_ratio=int(area_ratio),
        n_sector=int(n_sector),
        atol=float(atol),
    )


def htg_supercell_filling_from_density(
    density: np.ndarray,
    *,
    reference_diagonal: np.ndarray,
    area_ratio: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> float:
    density = np.asarray(density, dtype=np.complex128)
    nt, _, nk = density.shape
    projector = density + _supercell_reference_density_blocks(
        nt,
        nk,
        reference_diagonal=reference_diagonal,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    particles_per_k = float(np.trace(projector, axis1=0, axis2=1).real.sum()) / float(nk)
    reference_total = float(n_spin) * float(n_eta) * float(np.sum(np.asarray(reference_diagonal, dtype=float)))
    return float((particles_per_k - reference_total) / float(area_ratio))


def _supercell_reference_density_blocks(
    nt: int,
    nk: int,
    *,
    reference_diagonal: np.ndarray,
    n_spin: int = 2,
    n_eta: int = 2,
) -> np.ndarray:
    reference = np.asarray(reference_diagonal, dtype=float).reshape(-1)
    n_band = int(reference.size)
    if int(nt) != int(n_spin) * int(n_eta) * n_band:
        raise ValueError(f"nt={nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")
    idx = np.arange(int(nt), dtype=int).reshape((int(n_spin), int(n_eta), n_band), order="F")
    blocks = np.zeros((int(nt), int(nt), int(nk)), dtype=np.complex128)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            for iband in range(n_band):
                row = int(idx[ispin, ieta, iband])
                blocks[row, row, :] = float(reference[iband])
    return blocks


def _build_htg_supercell_projected_basis_from_kvec(
    model: HTGModel,
    interaction: InteractionParams,
    kvec: np.ndarray,
    *,
    supercell: HTGSupercell,
    mesh_size: int,
    k_grid_frac: np.ndarray | None,
    projected_band_count: int = 2,
) -> HTGSupercellProjectedBasisData:
    lattice = model.lattice
    super_g1, super_g2 = supercell.reciprocal_vectors(lattice)
    fold_reps = supercell_fold_representatives(supercell)
    area_ratio = int(supercell.area_ratio)
    if len(fold_reps) != area_ratio:
        raise ValueError(f"Expected {area_ratio} fold representatives, got {len(fold_reps)}")
    central_pair_raw = centered_band_indices(lattice.matrix_dim, 2)
    central_pair = (int(central_pair_raw[0]), int(central_pair_raw[1]))
    projected_indices = centered_projection_band_indices(lattice.matrix_dim, projected_band_count)
    primitive_band_count = len(projected_indices)
    folded_band_count_expected = folded_band_count(primitive_band_count, area_ratio)
    kvec = np.asarray(kvec, dtype=np.complex128).reshape(-1)

    grid_shape, origin, positions = _supercell_embedding_table(lattice, supercell, fold_reps)
    nx, ny = grid_shape
    embedded = np.zeros((6, nx, ny, folded_band_count_expected, 2, kvec.size), dtype=np.complex128)
    h_projected = np.zeros((folded_band_count_expected, folded_band_count_expected, 2, kvec.size), dtype=np.complex128)
    sigma_projected = np.zeros_like(h_projected)
    band_sigma_z = np.zeros((folded_band_count_expected, 2, kvec.size), dtype=float)
    sigma_z_operator = sublattice_sigma_z(lattice)
    layer_potential = _layer_potential_operator(lattice, interaction.U_ev)

    for iflavor, valley in enumerate((1, -1)):
        for ik, kval in enumerate(kvec):
            for ifold, (fold_x, fold_y) in enumerate(fold_reps):
                primitive_k_full = complex(kval + fold_x * super_g1 + fold_y * super_g2)
                primitive_k, primitive_shift = _centered_primitive_reduction(lattice, primitive_k_full)
                wavefunctions, h_block, sigma_block, sigma_values = _hybrid_projected_basis_at_k(
                    primitive_k,
                    lattice,
                    model.params,
                    interaction,
                    valley=valley,
                    projected_indices=projected_indices,
                    central_pair=central_pair,
                    sigma_z_operator=sigma_z_operator,
                    layer_potential=layer_potential,
                )
                shift_n1, shift_n2 = primitive_shift
                primitive_shift_sc = supercell.primitive_shift_to_supercell(-shift_n1, -shift_n2)
                for iprim, _band_index in enumerate(projected_indices):
                    out_band = iprim * area_ratio + ifold
                    h_projected[out_band, out_band, iflavor, ik] = h_block[iprim, iprim]
                    band_sigma_z[out_band, iflavor, ik] = float(np.real(sigma_values[iprim]))
                    for jprim in range(primitive_band_count):
                        out_band_rhs = jprim * area_ratio + ifold
                        h_projected[out_band, out_band_rhs, iflavor, ik] = h_block[iprim, jprim]
                        sigma_projected[out_band, out_band_rhs, iflavor, ik] = sigma_block[iprim, jprim]
                    for source_g_index, pair in enumerate(np.asarray(lattice.g_indices, dtype=int)):
                        pair_shift_sc = supercell.primitive_shift_to_supercell(int(pair[0]), int(pair[1]))
                        sx = int(fold_x + primitive_shift_sc[0] + pair_shift_sc[0])
                        sy = int(fold_y + primitive_shift_sc[1] + pair_shift_sc[1])
                        target_position = positions.get((sx, sy))
                        if target_position is None:
                            continue
                        ix, iy = target_position
                        start = 6 * source_g_index
                        embedded[:, ix, iy, out_band, iflavor, ik] = wavefunctions[start : start + 6, iprim]

    wavefunction_array = embedded.reshape((6 * nx * ny, folded_band_count_expected, 2, kvec.size), order="F")
    basis = ProjectedWavefunctionBasis(
        wavefunctions=wavefunction_array,
        grid_shape=grid_shape,
        n_spin=2,
        local_basis_size=6,
        name="htg_supercell_folded_chern_sublattice",
        boundary_mode="zero_fill",
    )

    h0 = np.zeros((basis.nt, basis.nt, basis.nk), dtype=np.complex128)
    sigma_z = np.zeros_like(h0)
    idx = np.arange(basis.nt, dtype=int).reshape((2, 2, folded_band_count_expected), order="F")
    for ik in range(basis.nk):
        for ispin in range(2):
            for iflavor in range(2):
                block_indices = np.asarray(idx[ispin, iflavor, :], dtype=int)
                h0[:, :, ik][np.ix_(block_indices, block_indices)] = h_projected[:, :, iflavor, ik]
                sigma_z[:, :, ik][np.ix_(block_indices, block_indices)] = sigma_projected[:, :, iflavor, ik]

    reference_diagonal = htg_supercell_reference_diagonal(primitive_band_count, area_ratio)
    primitive_area = real_space_cell_area_nm2_from_reciprocal(lattice.b_m1, lattice.b_m2)
    return HTGSupercellProjectedBasisData(
        model=model,
        interaction=interaction,
        supercell=supercell,
        mesh_size=int(mesh_size),
        kvec=kvec,
        k_grid_frac=None if k_grid_frac is None else np.asarray(k_grid_frac, dtype=float),
        basis=basis,
        h0=h0,
        sigma_z=sigma_z,
        band_sigma_z=band_sigma_z,
        primitive_projected_indices=projected_indices,
        primitive_band_count=primitive_band_count,
        fold_representatives=fold_reps,
        reference_diagonal=reference_diagonal,
        super_g1=complex(super_g1),
        super_g2=complex(super_g2),
        reciprocal_grid_shape=grid_shape,
        reciprocal_grid_origin=origin,
        moire_supercell_area_nm2=float(primitive_area) * float(area_ratio),
    )


def build_htg_supercell_projected_basis(
    model: HTGModel,
    interaction: InteractionParams | None = None,
    *,
    supercell: HTGSupercell | None = None,
    mesh_size: int | None = None,
    projected_band_count: int = 2,
) -> HTGSupercellProjectedBasisData:
    resolved_interaction = interaction if interaction is not None else InteractionParams()
    resolved_supercell = htg_default_fractional_supercell() if supercell is None else supercell
    resolved_mesh = resolved_interaction.n_k if mesh_size is None else int(mesh_size)
    if resolved_mesh <= 0:
        raise ValueError("mesh_size must be positive")
    k_grid_frac, kvec_grid = build_htg_supercell_uniform_grid(model.lattice, resolved_supercell, resolved_mesh, endpoint=False)
    return _build_htg_supercell_projected_basis_from_kvec(
        model,
        resolved_interaction,
        np.asarray(kvec_grid, dtype=np.complex128),
        supercell=resolved_supercell,
        mesh_size=resolved_mesh,
        k_grid_frac=k_grid_frac,
        projected_band_count=projected_band_count,
    )


def build_htg_supercell_projected_basis_for_kvec(
    model: HTGModel,
    interaction: InteractionParams,
    kvec: np.ndarray,
    *,
    supercell: HTGSupercell,
    projected_band_count: int = 2,
) -> HTGSupercellProjectedBasisData:
    kvec_array = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    return _build_htg_supercell_projected_basis_from_kvec(
        model,
        interaction,
        kvec_array,
        supercell=supercell,
        mesh_size=0,
        k_grid_frac=None,
        projected_band_count=projected_band_count,
    )


def build_htg_supercell_overlap_blocks(
    basis_data: HTGSupercellProjectedBasisData,
    *,
    g_shells: int | None = None,
) -> HFOverlapBlockSet:
    return build_htg_supercell_overlap_blocks_between(basis_data, basis_data, g_shells=g_shells, include_hartree=True)


def build_htg_supercell_overlap_blocks_between(
    target_basis_data: HTGSupercellProjectedBasisData,
    source_basis_data: HTGSupercellProjectedBasisData,
    *,
    g_shells: int | None = None,
    include_hartree: bool = True,
) -> HFOverlapBlockSet:
    if target_basis_data.basis.grid_shape != source_basis_data.basis.grid_shape:
        raise ValueError("Target and source HTG supercell bases must use the same reciprocal embedding grid")
    interaction = target_basis_data.interaction
    resolved_shells = interaction.g_shells if g_shells is None else int(g_shells)
    labels = reciprocal_shift_labels(resolved_shells)
    shifts = tuple((m, n) for m in labels for n in labels)
    gvecs = np.asarray(
        [m * target_basis_data.super_g1 + n * target_basis_data.super_g2 for m, n in shifts],
        dtype=np.complex128,
    )
    overlaps = {
        shift: calculate_projected_overlap_between(
            target_basis_data.basis,
            source_basis_data.basis,
            shift[0],
            shift[1],
        )
        for shift in shifts
    }
    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for shift, gvec in zip(shifts, gvecs, strict=True):
        if target_basis_data.nk == source_basis_data.nk:
            diagonal_overlaps[shift] = np.diagonal(overlaps[shift], axis1=1, axis2=3)
        if include_hartree:
            hartree_screening[shift] = float(screened_coulomb_matrix(np.asarray(gvec), interaction))
        qvals = source_basis_data.kvec[None, :] - target_basis_data.kvec[:, None] + complex(gvec)
        fock_screening[shift] = screened_coulomb_matrix(qvals, interaction)
    return HFOverlapBlockSet(
        shifts=shifts,
        gvecs=gvecs,
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def _density_from_hamiltonian(
    hamiltonian: np.ndarray,
    *,
    primitive_nu: float,
    reference_diagonal: np.ndarray,
    area_ratio: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> DensityUpdateResult:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = hamiltonian.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square Hamiltonian blocks, got {hamiltonian.shape}")
    reference_density = _supercell_reference_density_blocks(
        nt,
        nk,
        reference_diagonal=reference_diagonal,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    n_occ = htg_supercell_occupied_count_per_k(
        primitive_nu,
        reference_diagonal=reference_diagonal,
        area_ratio=area_ratio,
        n_sector=int(n_spin) * int(n_eta),
    )
    density = np.zeros_like(hamiltonian)
    energies = np.zeros((nt, nk), dtype=float)
    occ_mask = np.zeros((nt, nk), dtype=bool)
    for ik in range(nk):
        eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik])
        energies[:, ik] = eigvals
        if n_occ > 0:
            occupied_vecs = eigvecs[:, :n_occ]
            density[:, :, ik] = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density[:, :, ik]
            occ_mask[:n_occ, ik] = True
        else:
            density[:, :, ik] = -reference_density[:, :, ik]
    mu = find_chemical_potential(energies, float(n_occ) / float(nt))
    return DensityUpdateResult(
        density=density,
        energies=energies,
        mu=float(mu),
        observables={"occupation_mask": occ_mask},
    )


@dataclass(frozen=True)
class HTGSupercellDensityBuilder:
    primitive_nu: float
    reference_diagonal: np.ndarray
    area_ratio: int
    n_spin: int = 2
    n_eta: int = 2

    def __call__(self, hamiltonian: np.ndarray) -> DensityUpdateResult:
        return _density_from_hamiltonian(
            hamiltonian,
            primitive_nu=self.primitive_nu,
            reference_diagonal=self.reference_diagonal,
            area_ratio=self.area_ratio,
            n_spin=self.n_spin,
            n_eta=self.n_eta,
        )


@dataclass(frozen=True)
class HTGSupercellInitializer:
    initial_density: np.ndarray | None = None

    def __call__(self, state: HTGSupercellHartreeFockState, *, init_mode: str, seed: int) -> None:
        if self.initial_density is not None:
            density = np.asarray(self.initial_density, dtype=np.complex128)
            if density.shape != state.density.shape:
                raise ValueError(f"Expected initial_density shape {state.density.shape}, got {density.shape}")
            state.density[:, :, :] = density
        else:
            state.density[:, :, :] = initialize_htg_supercell_density(state, init_mode=init_mode, seed=seed)
        _update_supercell_diagnostics_from_density(state)


def initialize_htg_supercell_density(
    state: HTGSupercellHartreeFockState,
    *,
    init_mode: str = "bm",
    seed: int = 1,
) -> np.ndarray:
    mode = str(init_mode).strip().lower().replace("-", "_")
    primitive_band_count = _infer_primitive_band_count_from_reference(state.reference_diagonal)
    area_ratio = int(state.n_band // primitive_band_count)
    if mode in {"bm", "noninteracting", "fermi"}:
        return _density_from_hamiltonian(
            state.h0,
            primitive_nu=state.nu,
            reference_diagonal=state.reference_diagonal,
            area_ratio=area_ratio,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
        ).density

    rng = np.random.default_rng(seed)
    nt, _, nk = state.h0.shape
    reference_density = _supercell_reference_density_blocks(
        nt,
        nk,
        reference_diagonal=state.reference_diagonal,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )
    n_occ = htg_supercell_occupied_count_per_k(
        state.nu,
        reference_diagonal=state.reference_diagonal,
        area_ratio=area_ratio,
        n_sector=state.n_spin * state.n_eta,
    )
    if mode in {"random", "diag_random"}:
        density = np.zeros_like(state.h0)
        for ik in range(nk):
            unitary = random_unitary_from_hermitian(nt, rng)
            occupied_vecs = unitary[:, :n_occ]
            density[:, :, ik] = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density[:, :, ik]
        return density
    if mode in {"perturbed", "cdw", "fold_random"}:
        perturbed = np.asarray(state.h0, dtype=np.complex128).copy()
        scale = 1.0e-3
        for ik in range(nk):
            noise = rng.standard_normal((nt, nt)) + 1j * rng.standard_normal((nt, nt))
            noise = 0.5 * (noise + noise.conjugate().T)
            perturbed[:, :, ik] += scale * noise
        return _density_from_hamiltonian(
            perturbed,
            primitive_nu=state.nu,
            reference_diagonal=state.reference_diagonal,
            area_ratio=area_ratio,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
        ).density
    raise ValueError(f"Unsupported HTG supercell init_mode={init_mode!r}; use bm, perturbed, or random")


def _infer_primitive_band_count_from_reference(reference_diagonal: np.ndarray) -> int:
    # The current HTG supercell adapter folds each primitive projected band into
    # the same number of copies and stores them contiguously.  The default and
    # production path use the central two-band window; use the shortest even
    # primitive window consistent with a repeated reference pattern.
    reference = np.asarray(reference_diagonal, dtype=float).reshape(-1)
    for primitive_count in range(2, reference.size + 1, 2):
        if reference.size % primitive_count != 0:
            continue
        repeats = reference.size // primitive_count
        compressed = reference[::repeats]
        if np.allclose(reference, np.repeat(compressed, repeats)):
            return int(primitive_count)
    raise ValueError("Could not infer primitive projected band count from supercell reference")


def _supercell_gap_from_mask(energies: np.ndarray, occupation_mask: np.ndarray) -> float:
    energies = np.asarray(energies, dtype=float)
    occupied = np.asarray(occupation_mask, dtype=bool)
    if not np.any(occupied) or np.all(occupied):
        return float("nan")
    return float(np.min(energies[~occupied]) - np.max(energies[occupied]))


def _update_supercell_diagnostics_from_density(state: HTGSupercellHartreeFockState) -> None:
    primitive_band_count = _infer_primitive_band_count_from_reference(state.reference_diagonal)
    area_ratio = int(state.n_band // primitive_band_count)
    state.diagnostics["filling"] = htg_supercell_filling_from_density(
        state.density,
        reference_diagonal=state.reference_diagonal,
        area_ratio=area_ratio,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )
    projector = state.density + _supercell_reference_density_blocks(
        state.nt,
        state.nk,
        reference_diagonal=state.reference_diagonal,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )
    residual = 0.0
    for ik in range(state.nk):
        block = projector[:, :, ik]
        residual = max(residual, float(np.max(np.abs(block @ block - block))))
    state.diagnostics["projector_idempotency_residual"] = float(residual)


def _update_supercell_density_update_state(state: HTGSupercellHartreeFockState, density_update: DensityUpdateResult) -> None:
    _update_supercell_diagnostics_from_density(state)
    occupation_mask = density_update.observables.get("occupation_mask")
    if occupation_mask is not None:
        state.diagnostics["hf_gap"] = _supercell_gap_from_mask(state.energies, np.asarray(occupation_mask, dtype=bool))
    state.diagnostics["hamiltonian_hermitian_residual"] = hermitian_residual(state.hamiltonian)


def _update_supercell_step_state(state: HTGSupercellHartreeFockState, step) -> None:
    _update_supercell_density_update_state(state, step.density_update)


def build_htg_supercell_hf_kernel(
    state: HTGSupercellHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    *,
    beta: float = 1.0,
    use_numba: bool | None = None,
):
    primitive_band_count = _infer_primitive_band_count_from_reference(state.reference_diagonal)
    area_ratio = int(state.n_band // primitive_band_count)
    return build_projected_hf_kernel(
        state,
        overlap_blocks,
        density_builder=HTGSupercellDensityBuilder(
            state.nu,
            reference_diagonal=state.reference_diagonal,
            area_ratio=area_ratio,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
        ),
        energy_functional=compute_hf_energy,
        oda_parameterizer="default",
        step_callback=_update_supercell_step_state,
        final_state_callback=_update_supercell_density_update_state,
        convergence_rule="raw",
        v0=state.v0,
        beta=beta,
        use_numba=use_numba,
    )


def build_htg_supercell_hf_problem(
    state: HTGSupercellHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    *,
    beta: float = 1.0,
    initial_density: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> HartreeFockProblem:
    return build_projected_hf_problem(
        initializer=HTGSupercellInitializer(initial_density=initial_density),
        kernel=build_htg_supercell_hf_kernel(state, overlap_blocks, beta=beta, use_numba=use_numba),
    )


def run_htg_supercell_hf(
    model: HTGModel,
    interaction: InteractionParams | None = None,
    *,
    primitive_nu: float,
    supercell: HTGSupercell | None = None,
    init_mode: str = "perturbed",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 300,
    precision: float = 1.0e-6,
    oda_stall_threshold: float = 1.0e-3,
    mesh_size: int | None = None,
    g_shells: int | None = None,
    projected_band_count: int = 2,
    initial_density: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> HTGSupercellHartreeFockRun:
    resolved_interaction = interaction if interaction is not None else InteractionParams()
    resolved_supercell = htg_minimal_fractional_supercell(primitive_nu) if supercell is None else supercell
    basis_data = build_htg_supercell_projected_basis(
        model,
        resolved_interaction,
        supercell=resolved_supercell,
        mesh_size=mesh_size,
        projected_band_count=projected_band_count,
    )
    # Validate the requested filling before any SCF work.
    htg_supercell_occupied_count_per_k(
        primitive_nu,
        reference_diagonal=basis_data.reference_diagonal,
        area_ratio=basis_data.supercell.area_ratio,
        n_sector=basis_data.basis.n_spin * basis_data.basis.n_flavor,
    )
    state = HTGSupercellHartreeFockState.from_projected_basis(basis_data, nu=primitive_nu, precision=precision)
    overlap_blocks = build_htg_supercell_overlap_blocks(basis_data, g_shells=g_shells)
    problem = build_htg_supercell_hf_problem(
        state,
        overlap_blocks,
        beta=beta,
        initial_density=initial_density,
        use_numba=use_numba,
    )
    base_run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=str(init_mode),
        seed=int(seed),
        max_iter=int(max_iter),
        oda_stall_threshold=float(oda_stall_threshold),
    )
    return HTGSupercellHartreeFockRun(
        state=state,
        overlap_blocks=overlap_blocks,
        basis_data=basis_data,
        iter_energy=base_run.iter_energy,
        iter_err=base_run.iter_err,
        iter_oda=base_run.iter_oda,
        init_mode=base_run.init_mode,
        seed=base_run.seed,
        converged=base_run.converged,
        exit_reason=base_run.exit_reason,
    )


def scan_htg_supercell_ground_state(
    model: HTGModel,
    interaction: InteractionParams | None = None,
    *,
    primitive_nu: float,
    supercell: HTGSupercell | None = None,
    init_modes: Iterable[str] = ("perturbed", "random", "bm"),
    seeds: Iterable[int] = (1, 2, 3),
    beta: float = 1.0,
    max_iter: int = 300,
    precision: float = 1.0e-6,
    oda_stall_threshold: float = 1.0e-3,
    mesh_size: int | None = None,
    g_shells: int | None = None,
    projected_band_count: int = 2,
    use_numba: bool | None = None,
) -> HTGSupercellGroundStateScan:
    resolved_interaction = interaction if interaction is not None else InteractionParams()
    resolved_supercell = htg_minimal_fractional_supercell(primitive_nu) if supercell is None else supercell
    basis_data = build_htg_supercell_projected_basis(
        model,
        resolved_interaction,
        supercell=resolved_supercell,
        mesh_size=mesh_size,
        projected_band_count=projected_band_count,
    )
    htg_supercell_occupied_count_per_k(
        primitive_nu,
        reference_diagonal=basis_data.reference_diagonal,
        area_ratio=basis_data.supercell.area_ratio,
        n_sector=basis_data.basis.n_spin * basis_data.basis.n_flavor,
    )
    overlap_blocks = build_htg_supercell_overlap_blocks(basis_data, g_shells=g_shells)
    runs: list[HTGSupercellHartreeFockRun] = []
    for init_mode in init_modes:
        for seed in seeds:
            state = HTGSupercellHartreeFockState.from_projected_basis(
                basis_data,
                nu=primitive_nu,
                precision=precision,
            )
            problem = build_htg_supercell_hf_problem(state, overlap_blocks, beta=beta, use_numba=use_numba)
            base_run = run_hartree_fock_problem(
                state,
                problem,
                init_mode=str(init_mode),
                seed=int(seed),
                max_iter=int(max_iter),
                oda_stall_threshold=float(oda_stall_threshold),
            )
            runs.append(
                HTGSupercellHartreeFockRun(
                    state=state,
                    overlap_blocks=overlap_blocks,
                    basis_data=basis_data,
                    iter_energy=base_run.iter_energy,
                    iter_err=base_run.iter_err,
                    iter_oda=base_run.iter_oda,
                    init_mode=base_run.init_mode,
                    seed=base_run.seed,
                    converged=base_run.converged,
                    exit_reason=base_run.exit_reason,
                )
            )
    return HTGSupercellGroundStateScan(runs=tuple(runs))


def build_htg_supercell_gamma_path(
    basis_data: HTGSupercellProjectedBasisData,
    points_per_segment: int = 80,
) -> KPath:
    """Continuous HTG paper-style supercell path.

    This is an off-grid reconstruction path.  Mean-field band figures should use
    exact SCF-grid samples by default; call this only for explicitly diagnostic
    off-grid reconstruction.
    """

    gamma = 0.0 + 0.0j
    kappa = (basis_data.super_g1 + basis_data.super_g2) / 3.0
    kappa_prime_edge = -(basis_data.super_g1 + basis_data.super_g2) / 3.0 + basis_data.super_g1
    m_point = basis_data.super_g1 / 2.0
    gamma_across_m = gamma + basis_data.super_g1
    return build_kpath_from_nodes(
        (gamma, kappa, kappa_prime_edge, gamma, m_point, gamma_across_m),
        ("Gamma_s", "kappa_s", "kappa_prime_s", "Gamma_s", "M_s", "Gamma_s+G1"),
        int(points_per_segment),
    )


def evaluate_htg_supercell_hf_path(
    hf_run: HTGSupercellHartreeFockRun,
    *,
    path: KPath | None = None,
    points_per_segment: int = 80,
    g_shells: int | None = None,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> HTGSupercellPathResult:
    source_basis_data = hf_run.basis_data
    resolved_path = build_htg_supercell_gamma_path(source_basis_data, points_per_segment=points_per_segment) if path is None else path
    resolved_g_shells = source_basis_data.interaction.g_shells if g_shells is None else int(g_shells)
    path_basis_data = build_htg_supercell_projected_basis_for_kvec(
        source_basis_data.model,
        source_basis_data.interaction,
        resolved_path.kvec,
        supercell=source_basis_data.supercell,
        projected_band_count=source_basis_data.primitive_band_count,
    )
    target_overlap_blocks = build_htg_supercell_overlap_blocks(path_basis_data, g_shells=resolved_g_shells)
    target_source_overlap_blocks = build_htg_supercell_overlap_blocks_between(
        path_basis_data,
        source_basis_data,
        g_shells=resolved_g_shells,
        include_hartree=False,
    )
    h_path = build_projected_target_hamiltonian(
        path_basis_data.h0,
        hf_run.state.density,
        source_overlap_blocks=hf_run.overlap_blocks,
        target_overlap_blocks=target_overlap_blocks,
        target_source_overlap_blocks=target_source_overlap_blocks,
        v0=hf_run.state.v0,
        beta=beta,
        use_numba=use_numba,
    )
    energies = np.zeros((resolved_path.kvec.size, h_path.shape[0]), dtype=float)
    for ik in range(resolved_path.kvec.size):
        energies[ik, :] = np.linalg.eigvalsh(h_path[:, :, ik])
    return HTGSupercellPathResult(
        path=resolved_path,
        hamiltonian=h_path,
        energies=energies,
        mu=hf_run.state.mu,
        nu=hf_run.state.nu,
        init_mode=hf_run.init_mode,
        seed=hf_run.seed,
        exit_reason=hf_run.exit_reason,
        points_per_segment=int(points_per_segment),
    )


def save_htg_supercell_run_npz(path: str, run: HTGSupercellHartreeFockRun) -> None:
    np.savez_compressed(
        path,
        density=run.state.density,
        hamiltonian=run.state.hamiltonian,
        h0=run.state.h0,
        energies=run.state.energies,
        kvec=run.basis_data.kvec,
        k_grid_frac=np.asarray([]) if run.basis_data.k_grid_frac is None else run.basis_data.k_grid_frac,
        iter_energy=run.iter_energy,
        iter_err=run.iter_err,
        iter_oda=run.iter_oda,
        reference_diagonal=run.state.reference_diagonal,
        fold_representatives=np.asarray(run.basis_data.fold_representatives, dtype=int),
        supercell_matrix=np.asarray(
            [
                [run.basis_data.supercell.n11, run.basis_data.supercell.n12],
                [run.basis_data.supercell.n21, run.basis_data.supercell.n22],
            ],
            dtype=int,
        ),
        primitive_nu=float(run.state.nu),
        init_mode=str(run.init_mode),
        seed=int(run.seed),
        converged=bool(run.converged),
        exit_reason=str(run.exit_reason),
        diagnostics=np.asarray([run.state.diagnostics], dtype=object),
    )


def save_htg_supercell_path_npz(path: str, result: HTGSupercellPathResult) -> None:
    np.savez_compressed(
        path,
        kvec=result.path.kvec,
        kdist=result.path.kdist,
        labels=np.asarray(result.path.labels, dtype=object),
        node_indices=np.asarray(result.path.node_indices, dtype=int),
        hamiltonian=result.hamiltonian,
        energies=result.energies,
        mu=float(result.mu),
        primitive_nu=float(result.nu),
        init_mode=str(result.init_mode),
        seed=int(result.seed),
        exit_reason=str(result.exit_reason),
    )
