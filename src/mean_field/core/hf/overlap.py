from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

try:  # pragma: no cover - exercised only when the optional perf extra exists.
    from numba import njit, prange
except Exception:  # pragma: no cover - import failure is the supported fallback.
    njit = None
    prange = range


@dataclass(frozen=True)
class ProjectedWavefunctionBasis:
    """Band-projected wavefunctions arranged on a reciprocal-lattice shift grid."""

    wavefunctions: np.ndarray
    grid_shape: tuple[int, int]
    n_spin: int = 1
    local_basis_size: int | None = None
    name: str = "projected"
    boundary_mode: str = "periodic"

    def __post_init__(self) -> None:
        wavefunctions = np.asarray(self.wavefunctions, dtype=np.complex128)
        if wavefunctions.ndim != 4:
            raise ValueError(f"Expected wavefunctions shape (basis, band, flavor, k), got {wavefunctions.shape}")
        grid_shape = tuple(int(v) for v in self.grid_shape)
        if len(grid_shape) != 2 or grid_shape[0] <= 0 or grid_shape[1] <= 0:
            raise ValueError(f"Expected a positive two-dimensional grid_shape, got {self.grid_shape}")
        grid_size = grid_shape[0] * grid_shape[1]
        local_basis_size = self.local_basis_size
        if local_basis_size is None:
            if wavefunctions.shape[0] % grid_size != 0:
                raise ValueError(
                    f"Wavefunction basis dimension {wavefunctions.shape[0]} is not divisible by grid size {grid_size}"
                )
            local_basis_size = wavefunctions.shape[0] // grid_size
        local_basis_size = int(local_basis_size)
        if local_basis_size <= 0:
            raise ValueError(f"Expected positive local_basis_size, got {local_basis_size}")
        if wavefunctions.shape[0] != local_basis_size * grid_size:
            raise ValueError(
                f"Expected basis dimension {local_basis_size * grid_size} from local_basis_size={local_basis_size} "
                f"and grid_shape={grid_shape}, got {wavefunctions.shape[0]}"
            )
        n_spin = int(self.n_spin)
        if n_spin <= 0:
            raise ValueError(f"Expected positive n_spin, got {self.n_spin}")
        boundary_mode = _normalize_boundary_mode(self.boundary_mode)

        object.__setattr__(self, "wavefunctions", wavefunctions)
        object.__setattr__(self, "grid_shape", grid_shape)
        object.__setattr__(self, "local_basis_size", local_basis_size)
        object.__setattr__(self, "n_spin", n_spin)
        object.__setattr__(self, "boundary_mode", boundary_mode)

    @property
    def n_band(self) -> int:
        return int(self.wavefunctions.shape[1])

    @property
    def n_flavor(self) -> int:
        return int(self.wavefunctions.shape[2])

    @property
    def nk(self) -> int:
        return int(self.wavefunctions.shape[3])

    @property
    def nt(self) -> int:
        return int(self.n_spin * self.n_flavor * self.n_band)

    @property
    def basis_dimension(self) -> int:
        return int(self.wavefunctions.shape[0])


@dataclass(frozen=True)
class HFOverlapBlockSet:
    shifts: tuple[tuple[int, int], ...]
    gvecs: np.ndarray
    overlaps: dict[tuple[int, int], np.ndarray]
    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    hartree_screening: dict[tuple[int, int], float] = field(default_factory=dict)
    fock_screening: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class OverlapDiagnostics:
    theta_deg: float
    lattice_kind: str
    valley_label: str
    m: int
    n: int
    fro_norm: float
    max_abs: float
    trace_real: float
    trace_imag: float
    entry_11_real: float
    entry_11_imag: float
    entry_mid_real: float
    entry_mid_imag: float


def _numba_enabled(use_numba: bool | None) -> bool:
    if use_numba is False:
        return False
    disabled = os.environ.get("MEAN_FIELD_HF_DISABLE_NUMBA", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return False
    return njit is not None


def _normalize_boundary_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"periodic", "hf_periodic", "wrap", "wrapped"}:
        return "periodic"
    if normalized in {"zero_fill", "zerofill", "zhang_zero_fill", "finite_cutoff"}:
        return "zero_fill"
    raise ValueError(f"Unsupported projected-overlap boundary_mode: {mode!r}")


def shift_wavefunction_grid(
    values: np.ndarray,
    dm: int,
    dn: int,
    *,
    boundary_mode: str = "zero_fill",
    grid_axes: tuple[int, int] = (1, 2),
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex128)
    mode = _normalize_boundary_mode(boundary_mode)
    dm = int(dm)
    dn = int(dn)
    axis_m, axis_n = (int(grid_axes[0]), int(grid_axes[1]))
    if axis_m == axis_n:
        raise ValueError(f"grid_axes must be distinct, got {grid_axes}")
    if arr.ndim <= max(axis_m, axis_n) or min(axis_m, axis_n) < 0:
        raise ValueError(f"grid_axes={grid_axes} are incompatible with shape {arr.shape}")
    if mode == "periodic":
        return np.roll(arr, shift=(dm, dn), axis=(axis_m, axis_n))

    nx, ny = arr.shape[axis_m], arr.shape[axis_n]
    out = np.zeros_like(arr)
    if abs(dm) >= nx or abs(dn) >= ny:
        return out

    if dm >= 0:
        dst_x = slice(dm, nx)
        src_x = slice(0, nx - dm)
    else:
        dst_x = slice(0, nx + dm)
        src_x = slice(-dm, nx)
    if dn >= 0:
        dst_y = slice(dn, ny)
        src_y = slice(0, ny - dn)
    else:
        dst_y = slice(0, ny + dn)
        src_y = slice(-dn, ny)

    src = [slice(None)] * arr.ndim
    dst = [slice(None)] * arr.ndim
    src[axis_m] = src_x
    src[axis_n] = src_y
    dst[axis_m] = dst_x
    dst[axis_n] = dst_y
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _shift_wavefunction_grid(values: np.ndarray, dm: int, dn: int, *, boundary_mode: str) -> np.ndarray:
    return shift_wavefunction_grid(values, dm, dn, boundary_mode=boundary_mode, grid_axes=(1, 2))


def validate_projected_basis_compatibility(target: ProjectedWavefunctionBasis, source: ProjectedWavefunctionBasis) -> None:
    if target.local_basis_size != source.local_basis_size:
        raise ValueError(f"local_basis_size mismatch: {target.local_basis_size} != {source.local_basis_size}")
    if target.n_band != source.n_band:
        raise ValueError(f"n_band mismatch: {target.n_band} != {source.n_band}")
    if target.n_flavor != source.n_flavor:
        raise ValueError(f"n_flavor mismatch: {target.n_flavor} != {source.n_flavor}")
    if target.n_spin != source.n_spin:
        raise ValueError(f"n_spin mismatch: {target.n_spin} != {source.n_spin}")
    if target.grid_shape != source.grid_shape:
        raise ValueError(f"grid_shape mismatch: {target.grid_shape} != {source.grid_shape}")
    if target.boundary_mode != source.boundary_mode:
        raise ValueError(f"boundary_mode mismatch: {target.boundary_mode} != {source.boundary_mode}")


def calculate_projected_overlap_compact(
    basis: ProjectedWavefunctionBasis,
    m: int,
    n: int,
    *,
    flavor_index: int = 0,
) -> np.ndarray:
    if flavor_index < 0 or flavor_index >= basis.n_flavor:
        raise ValueError(f"flavor_index={flavor_index} is outside [0, {basis.n_flavor})")

    nx, ny = basis.grid_shape
    nk_band = basis.n_band * basis.nk
    ul = basis.wavefunctions[:, :, flavor_index, :].reshape(basis.basis_dimension, nk_band, order="F")
    shifted = _shift_wavefunction_grid(
        ul.reshape(basis.local_basis_size, nx, ny, nk_band, order="F"),
        -int(m),
        -int(n),
        boundary_mode=basis.boundary_mode,
    ).reshape(basis.basis_dimension, nk_band, order="F")
    return ul.conj().T @ shifted


def calculate_projected_overlap_between(
    target: ProjectedWavefunctionBasis,
    source: ProjectedWavefunctionBasis,
    m: int,
    n: int,
) -> np.ndarray:
    validate_projected_basis_compatibility(target, source)

    nx, ny = target.grid_shape
    target_band_k = target.n_band * target.nk
    source_band_k = source.n_band * source.nk
    overlap_blocks = np.zeros(
        (
            target.n_spin,
            target.n_flavor,
            target_band_k,
            target.n_spin,
            target.n_flavor,
            source_band_k,
        ),
        dtype=np.complex128,
        order="F",
    )

    for iflavor in range(target.n_flavor):
        ul = target.wavefunctions[:, :, iflavor, :].reshape(target.basis_dimension, target_band_k, order="F")
        ur = source.wavefunctions[:, :, iflavor, :].reshape(source.basis_dimension, source_band_k, order="F")
        shifted = _shift_wavefunction_grid(
            ur.reshape(source.local_basis_size, nx, ny, source_band_k, order="F"),
            -int(m),
            -int(n),
            boundary_mode=target.boundary_mode,
        ).reshape(source.basis_dimension, source_band_k, order="F")
        lambda_kp = ul.conj().T @ shifted
        for ispin in range(target.n_spin):
            overlap_blocks[ispin, iflavor, :, ispin, iflavor, :] = lambda_kp

    return overlap_blocks.reshape((target.nt, target.nk, source.nt, source.nk), order="F")


def calculate_projected_overlap(basis: ProjectedWavefunctionBasis, m: int, n: int) -> np.ndarray:
    overlap = calculate_projected_overlap_between(basis, basis, m, n)
    return overlap.reshape(basis.nt * basis.nk, basis.nt * basis.nk, order="F")


def diagonal_overlap_blocks(overlap: np.ndarray, *, nt: int | None = None, nk: int | None = None) -> np.ndarray:
    if overlap.ndim != 4:
        raise ValueError(f"Expected overlap shape (nt, nk_target, nt, nk_source), got {overlap.shape}")
    nt_expected = overlap.shape[0] if nt is None else int(nt)
    nk_expected = overlap.shape[1] if nk is None else int(nk)
    if overlap.shape != (nt_expected, nk_expected, nt_expected, nk_expected):
        raise ValueError(f"Expected overlap shape {(nt_expected, nk_expected, nt_expected, nk_expected)}, got {overlap.shape}")
    return np.diagonal(overlap, axis1=1, axis2=3)


def compute_density_overlap_trace(density: np.ndarray, overlap: np.ndarray, *, use_numba: bool | None = None) -> complex:
    nt, _, nk = density.shape
    diagonal_overlap = diagonal_overlap_blocks(overlap, nt=nt, nk=nk)
    return compute_density_overlap_trace_from_diagonal(density, diagonal_overlap, use_numba=use_numba)


def compute_density_overlap_trace_from_diagonal(
    density: np.ndarray,
    diagonal_overlap: np.ndarray,
    *,
    use_numba: bool | None = None,
) -> complex:
    density = np.asarray(density, dtype=np.complex128)
    diagonal_overlap = np.asarray(diagonal_overlap, dtype=np.complex128)
    nt, nt_rhs, nk = density.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {density.shape}")
    if diagonal_overlap.shape != (nt, nt, nk):
        raise ValueError(f"Expected diagonal_overlap shape {(nt, nt, nk)}, got {diagonal_overlap.shape}")

    if _numba_enabled(use_numba):
        return complex(_compute_density_overlap_trace_from_diagonal_numba(density, diagonal_overlap))

    total = np.einsum("abk,bak->", density, np.conj(diagonal_overlap), optimize=True)
    return complex(total)


def contract_fock_term_from_overlap(
    overlap: np.ndarray,
    density: np.ndarray,
    coeff_matrix: np.ndarray,
    *,
    use_numba: bool | None = None,
) -> np.ndarray:
    overlap = np.asarray(overlap, dtype=np.complex128)
    density = np.asarray(density, dtype=np.complex128)
    coeff_matrix = np.asarray(coeff_matrix)
    nt, nk_target, nt_rhs, nk_source = overlap.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square flavor dimensions in overlap, got {overlap.shape}")
    if density.shape != (nt, nt, nk_source):
        raise ValueError(f"Expected density shape {(nt, nt, nk_source)}, got {density.shape}")
    if coeff_matrix.shape != (nk_target, nk_source):
        raise ValueError(f"Expected coeff_matrix shape {(nk_target, nk_source)}, got {coeff_matrix.shape}")

    if _numba_enabled(use_numba):
        numba_coeff_matrix = _coerce_fock_coeff_matrix_for_numba(coeff_matrix)
        return _contract_fock_term_from_overlap_numba(overlap, density, numba_coeff_matrix)

    lambda_blocks = np.transpose(overlap, (1, 3, 0, 2))
    density_t = np.transpose(density, (2, 1, 0))
    intermediate = np.einsum("tsac,scd->tsad", lambda_blocks, density_t, optimize=True)
    fock = np.einsum("ts,tsad,tsbd->tab", coeff_matrix, intermediate, np.conj(lambda_blocks), optimize=True)
    return np.transpose(fock, (1, 2, 0))


def _coerce_fock_coeff_matrix_for_numba(coeff_matrix: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(coeff_matrix):
        return np.asarray(coeff_matrix, dtype=np.complex128)
    return np.asarray(coeff_matrix, dtype=np.float64)


def summarize_overlap(
    theta_deg: float,
    lattice_kind: str,
    overlap: np.ndarray,
    m: int,
    n: int,
    *,
    valley_label: str = "K",
) -> OverlapDiagnostics:
    mid = overlap.shape[0] // 2
    return OverlapDiagnostics(
        theta_deg=theta_deg,
        lattice_kind=lattice_kind,
        valley_label=valley_label,
        m=int(m),
        n=int(n),
        fro_norm=float(np.linalg.norm(overlap)),
        max_abs=float(np.max(np.abs(overlap))),
        trace_real=float(np.trace(overlap).real),
        trace_imag=float(np.trace(overlap).imag),
        entry_11_real=float(overlap[0, 0].real),
        entry_11_imag=float(overlap[0, 0].imag),
        entry_mid_real=float(overlap[mid, mid].real),
        entry_mid_imag=float(overlap[mid, mid].imag),
    )


if njit is not None:  # pragma: no branch

    @njit(cache=True, fastmath=True)
    def _compute_density_overlap_trace_from_diagonal_numba(density: np.ndarray, diagonal_overlap: np.ndarray) -> complex:
        total = 0.0 + 0.0j
        nt = density.shape[0]
        nk = density.shape[2]
        for ik in range(nk):
            for a in range(nt):
                for b in range(nt):
                    total += density[a, b, ik] * np.conjugate(diagonal_overlap[b, a, ik])
        return total

    @njit(cache=True, fastmath=True, parallel=True)
    def _contract_fock_term_from_overlap_numba(
        overlap: np.ndarray,
        density: np.ndarray,
        coeff_matrix: np.ndarray,
    ) -> np.ndarray:
        nt = overlap.shape[0]
        nk_target = overlap.shape[1]
        nk_source = overlap.shape[3]
        out = np.zeros((nt, nt, nk_target), dtype=np.complex128)
        for ik_target in prange(nk_target):
            for ik_source in range(nk_source):
                coeff = coeff_matrix[ik_target, ik_source]
                if coeff == 0.0:
                    continue
                for a in range(nt):
                    for b in range(nt):
                        total = 0.0 + 0.0j
                        for c in range(nt):
                            left = overlap[a, ik_target, c, ik_source]
                            if left == 0.0:
                                continue
                            for d in range(nt):
                                right = overlap[b, ik_target, d, ik_source]
                                if right == 0.0:
                                    continue
                                total += left * density[d, c, ik_source] * np.conjugate(right)
                        out[a, b, ik_target] += coeff * total
        return out

else:

    def _compute_density_overlap_trace_from_diagonal_numba(density: np.ndarray, diagonal_overlap: np.ndarray) -> complex:
        raise RuntimeError("numba is not available")

    def _contract_fock_term_from_overlap_numba(
        overlap: np.ndarray,
        density: np.ndarray,
        coeff_matrix: np.ndarray,
    ) -> np.ndarray:
        raise RuntimeError("numba is not available")
