"""Source-bound loader for the 40-atom/320-Wannier TPT bulk model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

TPT_BULK_HR_SHA256 = "583ddf207e81231dfc3641f4a9895bdc146e1d4e093bb9469776337f8b312b14"
# The accepted text HR prints real and imaginary coefficients to six decimal
# places in eV.  Exact-grid comparisons to the binary checkpoint Hamiltonian
# must therefore not impose binary-roundoff agreement on this quantized export.
TPT_BULK_HR_TEXT_COEFFICIENT_RESOLUTION_EV = 1.0e-6
# This is the previously sealed serialized-HR replay tolerance used by the
# accepted projector-refinement lineage, not a post-hoc fit to this bridge.
TPT_BULK_HR_CHECKPOINT_ACTIVE_ENERGY_TOLERANCE_EV = 5.0e-5
TPT_BULK_HR_CHECKPOINT_ACTIVE_SUBSPACE_MIN_SV = 1.0 - 1.0e-5
TPT_BULK_NUM_WANN = 320
TPT_BULK_NUM_RPTS = 325
TPT_BULK_SOURCE_MESH = (12, 4, 4)


@dataclass(frozen=True)
class TPTBulkSource:
    root: Path
    lattice_angstrom: np.ndarray
    species: tuple[str, ...]
    counts: tuple[int, ...]
    positions_fractional: np.ndarray
    hr_sha256: str

    @property
    def hr_path(self) -> Path:
        return self.root / "wannier90_hr.dat"

    @property
    def inplane_area_angstrom2(self) -> float:
        return float(np.linalg.norm(np.cross(self.lattice_angstrom[0], self.lattice_angstrom[1])))

    @property
    def cell_volume_angstrom3(self) -> float:
        return float(abs(np.linalg.det(self.lattice_angstrom)))

    @property
    def normal_repeat_angstrom(self) -> float:
        return float(self.cell_volume_angstrom3 / self.inplane_area_angstrom2)

    @property
    def reciprocal_rows_angstrom_inv(self) -> np.ndarray:
        return 2.0 * np.pi * np.linalg.inv(self.lattice_angstrom).T


@dataclass(frozen=True)
class TPTBulkActiveEigensystem:
    source: TPTBulkSource
    mesh: tuple[int, int, int]
    active_ranks_1based: tuple[int, int]
    k_fractional: np.ndarray
    active_energies_ev: np.ndarray
    active_eigenvectors: np.ndarray
    max_antihermitian_before_symmetrization_ev: float
    max_eigenpair_residual_ev: float

    @property
    def nk(self) -> int:
        return int(self.k_fractional.shape[0])

    @property
    def active_rank(self) -> int:
        return int(self.active_energies_ev.shape[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_tpt_bulk_source(
    root: str | Path,
    *,
    expected_hr_sha256: str = TPT_BULK_HR_SHA256,
) -> TPTBulkSource:
    source_root = Path(root).expanduser().resolve()
    poscar_path = source_root / "POSCAR"
    hr_path = source_root / "wannier90_hr.dat"
    if not poscar_path.is_file() or not hr_path.is_file():
        raise FileNotFoundError("TPT bulk source must contain POSCAR and wannier90_hr.dat")

    lines = _read_nonempty_lines(poscar_path)
    scale = float(lines[1])
    lattice = scale * np.asarray([[float(value) for value in lines[index].split()[:3]] for index in range(2, 5)])
    species = tuple(lines[5].split())
    counts = tuple(int(value) for value in lines[6].split())
    if species != ("Ta", "Te", "Pd") or counts != (8, 20, 12):
        raise ValueError(f"unexpected TPT bulk species/counts: {species}/{counts}")
    coordinate_mode = lines[7].lower()
    if not coordinate_mode.startswith("d"):
        raise ValueError("TPT bulk POSCAR must use direct coordinates")
    natom = sum(counts)
    positions = np.asarray(
        [[float(value) for value in lines[8 + index].split()[:3]] for index in range(natom)],
        dtype=np.float64,
    )
    if positions.shape != (40, 3):
        raise ValueError(f"unexpected TPT bulk position shape {positions.shape}")
    hr_sha256 = _sha256(hr_path)
    if expected_hr_sha256 and hr_sha256 != expected_hr_sha256:
        raise ValueError(
            "TPT bulk HR hash mismatch: "
            f"expected {expected_hr_sha256}, got {hr_sha256}"
        )
    return TPTBulkSource(
        root=source_root,
        lattice_angstrom=lattice,
        species=species,
        counts=counts,
        positions_fractional=positions,
        hr_sha256=hr_sha256,
    )


def _parse_hr_mod_grid(
    path: Path,
    mesh: tuple[int, int, int],
    *,
    expected_sha256: str,
) -> tuple[np.ndarray, int, int]:
    with path.open("rb", buffering=0) as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "TPT bulk HR changed before eigensystem construction: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        handle.seek(0)
        handle.readline()
        num_wann = int(handle.readline())
        nrpts = int(handle.readline())
        if num_wann != TPT_BULK_NUM_WANN or nrpts != TPT_BULK_NUM_RPTS:
            raise ValueError(
                f"unexpected TPT bulk HR dimensions: num_wann={num_wann}, nrpts={nrpts}"
            )
        degeneracies: list[int] = []
        while len(degeneracies) < nrpts:
            degeneracies.extend(int(item) for item in handle.readline().split())
        if len(degeneracies) != nrpts or any(value <= 0 for value in degeneracies):
            raise ValueError("invalid TPT bulk HR degeneracy list")

        hr_mod = np.zeros((*mesh, num_wann, num_wann), dtype=np.complex128)
        expected_m = np.tile(np.arange(1, num_wann + 1, dtype=np.int64), num_wann)
        expected_n = np.repeat(np.arange(1, num_wann + 1, dtype=np.int64), num_wann)
        values_per_block = num_wann * num_wann * 7
        seen_r: set[tuple[int, int, int]] = set()
        for ir in range(nrpts):
            raw = np.fromfile(handle, dtype=np.float64, count=values_per_block, sep=" ")
            if raw.size != values_per_block:
                raise ValueError(f"truncated TPT bulk HR block {ir}: {raw.size}/{values_per_block}")
            rows = raw.reshape(num_wann * num_wann, 7)
            integer_columns = np.rint(rows[:, :5]).astype(np.int64)
            if not np.array_equal(rows[:, :5], integer_columns.astype(np.float64)):
                raise ValueError(f"noninteger R/m/n field in TPT bulk HR block {ir}")
            r_columns = integer_columns[:, :3]
            if not np.all(r_columns == r_columns[0]):
                raise ValueError(f"nonconstant R in TPT bulk HR block {ir}")
            r = tuple(int(value) for value in r_columns[0])
            if r in seen_r:
                raise ValueError(f"duplicate TPT bulk HR representative {r}")
            seen_r.add(r)
            m = integer_columns[:, 3]
            n = integer_columns[:, 4]
            if not np.array_equal(m, expected_m) or not np.array_equal(n, expected_n):
                raise ValueError(f"unexpected matrix-index order in TPT bulk HR block {ir}")
            coefficients = rows[:, 5] + 1j * rows[:, 6]
            block = np.empty((num_wann, num_wann), dtype=np.complex128)
            block[m - 1, n - 1] = coefficients
            target = tuple(r[axis] % mesh[axis] for axis in range(3))
            hr_mod[target] += block / float(degeneracies[ir])
        if handle.read().strip():
            raise ValueError("unexpected trailing non-whitespace TPT bulk HR content")
    return hr_mod, num_wann, nrpts


def _signed_mesh_coordinate(index: int, size: int) -> float:
    value = float(index) / float(size)
    return value - 1.0 if value > 0.5 else value


def _uniform_k_fractional(mesh: tuple[int, int, int]) -> np.ndarray:
    """Wannier90 source ordering: first reciprocal coordinate varies fastest."""

    nx, ny, nz = mesh
    return np.asarray(
        [
            (
                _signed_mesh_coordinate(ix, nx),
                _signed_mesh_coordinate(iy, ny),
                _signed_mesh_coordinate(iz, nz),
            )
            for iz in range(nz)
            for iy in range(ny)
            for ix in range(nx)
        ],
        dtype=np.float64,
    )


def build_tpt_bulk_active_eigensystem(
    source: TPTBulkSource,
    *,
    mesh: Iterable[int],
    active_ranks_1based: tuple[int, int] = (233, 248),
) -> TPTBulkActiveEigensystem:
    resolved_mesh = tuple(int(value) for value in mesh)
    if len(resolved_mesh) != 3 or any(value <= 0 for value in resolved_mesh):
        raise ValueError(f"mesh must contain three positive integers, got {resolved_mesh}")
    rank_first, rank_last = (int(value) for value in active_ranks_1based)
    if not 1 <= rank_first <= rank_last <= TPT_BULK_NUM_WANN:
        raise ValueError(f"invalid active ranks {active_ranks_1based}")

    hr_mod, num_wann, _nrpts = _parse_hr_mod_grid(
        source.hr_path,
        resolved_mesh,
        expected_sha256=source.hr_sha256,
    )
    hk_grid = np.fft.ifftn(hr_mod, axes=(0, 1, 2))
    hk_grid *= int(np.prod(resolved_mesh))
    del hr_mod

    k_fractional = _uniform_k_fractional(resolved_mesh)
    nk = int(k_fractional.shape[0])
    active_rank = rank_last - rank_first + 1
    active_energies = np.empty((active_rank, nk), dtype=np.float64)
    active_vectors = np.empty((num_wann, active_rank, nk), dtype=np.complex128)
    max_antihermitian = 0.0
    max_eigenpair_residual = 0.0
    first = rank_first - 1
    stop = rank_last
    # The FFT grid is indexed (ix,iy,iz), whereas the accepted Wannier90
    # k-point list stores ix fastest, then iy, then iz.
    hk_wannier_order = np.transpose(hk_grid, (2, 1, 0, 3, 4)).reshape(
        (nk, num_wann, num_wann)
    )
    for ik, matrix_view in enumerate(hk_wannier_order):
        anti = float(np.max(np.abs(matrix_view - matrix_view.conj().T)))
        max_antihermitian = max(max_antihermitian, anti)
        matrix = 0.5 * (matrix_view + matrix_view.conj().T)
        values, vectors = np.linalg.eigh(matrix)
        selected_values = values[first:stop]
        selected_vectors = vectors[:, first:stop]
        residual = float(
            np.max(
                np.abs(
                    matrix @ selected_vectors
                    - selected_vectors * selected_values[None, :]
                )
            )
        )
        max_eigenpair_residual = max(max_eigenpair_residual, residual)
        active_energies[:, ik] = selected_values
        active_vectors[:, :, ik] = selected_vectors
    del hk_grid
    if max_antihermitian > 1.0e-4:
        raise ValueError(
            "TPT bulk H(k) Hermiticity gate failed before symmetrization: "
            f"{max_antihermitian:.6e} eV"
        )
    if max_eigenpair_residual > 1.0e-9:
        raise ValueError(
            f"TPT bulk active eigenpair residual too large: {max_eigenpair_residual:.6e} eV"
        )
    return TPTBulkActiveEigensystem(
        source=source,
        mesh=resolved_mesh,
        active_ranks_1based=(rank_first, rank_last),
        k_fractional=k_fractional,
        active_energies_ev=active_energies,
        active_eigenvectors=active_vectors,
        max_antihermitian_before_symmetrization_ev=max_antihermitian,
        max_eigenpair_residual_ev=max_eigenpair_residual,
    )


__all__ = [
    "TPT_BULK_HR_CHECKPOINT_ACTIVE_ENERGY_TOLERANCE_EV",
    "TPT_BULK_HR_CHECKPOINT_ACTIVE_SUBSPACE_MIN_SV",
    "TPT_BULK_HR_SHA256",
    "TPT_BULK_HR_TEXT_COEFFICIENT_RESOLUTION_EV",
    "TPT_BULK_NUM_RPTS",
    "TPT_BULK_NUM_WANN",
    "TPT_BULK_SOURCE_MESH",
    "TPTBulkActiveEigensystem",
    "TPTBulkSource",
    "build_tpt_bulk_active_eigensystem",
    "load_tpt_bulk_source",
]
