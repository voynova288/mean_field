from __future__ import annotations

from ._supercell_shared import *  # noqa: F401,F403
from ._supercell_types import *  # noqa: F401,F403

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
    boundary.  Berry-link/Chern helpers are archived out of the tracked public
    surface for now; this transform remains available for any future reviewed
    topology API.
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

__all__ = [name for name in globals() if not name.startswith('__')]
