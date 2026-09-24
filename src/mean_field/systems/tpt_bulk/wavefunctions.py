"""Same-parent Wannier90/WAVECAR wavefunction utilities for TPT bulk.

The accepted noninteracting Hamiltonian is represented in the 320-Wannier
basis, whereas microscopic density vertices must be contracted from the
same-parent spinor plane-wave coefficients.  This module supplies the exact
bridge

    DFT bands --U_dis(k) U(k)--> Wannier gauge --E_active(k)--> active states

and a streaming WAVECAR reader.  It deliberately does not invent PAW
augmentation: direct WAVECAR contractions are pseudo-wavefunction matrix
elements and cannot satisfy the all-electron ``rho(k,Q=0)=I`` production gate
unless a separately validated PAW correction is added.

The unformatted checkpoint layout follows Wannier90 3.1
``src/w90chk2chk.F90``.  The plane-wave ordering follows VASP/WaveTrans and is
cross-checkable against :class:`pymatgen.io.vasp.outputs.Wavecar` without
loading the full 23 GB WAVECAR into memory.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
import threading
from typing import BinaryIO, Iterable, Iterator, Sequence

import numpy as np

Array = np.ndarray

TPT_BULK_WAVECAR_SHA256 = (
    "7e7394f9323fb9c8efbbdabe1431a0df326149c1ec0f74817c474e12f3b6f7c9"
)
TPT_BULK_WANNIER90_CHK_SHA256 = (
    "5614848ae7f54b0c7294eab1ecd1502149204bf3827681efca60393503366598"
)
TPT_BULK_WANNIER90_EIG_SHA256 = (
    "f3fc1dc6c16e540b2463cb7e2560abe4981c7ae5f680e36af8ed5ae75dcd2328"
)
TPT_BULK_WANNIER90_AMN_SHA256 = (
    "e6a5642281489d6986d0102c6a5babad8219be77da6b07d9694891c0c558786d"
)
TPT_BULK_WANNIER90_MMN_SHA256 = (
    "ee898bd6c43c30f20f414c2798aaedecb887df8825018c7464667a067f02a740"
)
TPT_BULK_POTCAR_SHA256 = (
    "5c74d51a0724e9b2f514b3d99398c9acbb18fd0848e77ae5eb3a57b1e5be3cb5"
)
TPT_BULK_POSCAR_SHA256 = (
    "8d98fc676d1c01c954c5cf39980abb165f5d79cd1313e80bee197b5de2dd8eb9"
)


def _sha256_open_file(handle: BinaryIO) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    handle.seek(0)
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return digest


@dataclass(frozen=True)
class _FortranRecord:
    payload_offset: int
    size_bytes: int


@dataclass(frozen=True)
class Wannier90CheckpointMetadata:
    """Metadata and record locations in an unformatted Wannier90 checkpoint."""

    header: str
    byte_order: str
    sha256: str | None
    num_bands: int
    exclude_bands_1based: tuple[int, ...]
    real_lattice_angstrom: Array
    reciprocal_lattice_angstrom_inv: Array
    num_kpts: int
    mp_grid: tuple[int, int, int]
    kpoints_fractional: Array
    nntot: int
    num_wann: int
    checkpoint: str
    have_disentangled: bool
    omega_invariant: float | None
    lwindow: Array
    ndimwin: Array


class Wannier90CheckpointReader:
    """Streaming reader for a Wannier90 unformatted ``.chk`` file.

    The two large gauge records are read one k point at a time.  This avoids
    materializing the 2.8 GB checkpoint or converting it to a much larger
    formatted file.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self._handle = self.path.open("rb", buffering=0)
        try:
            self._byte_order = self._detect_byte_order()
            self._records = self._index_records()
            digest = None
            if expected_sha256 is not None:
                expected = _require_sha256(
                    expected_sha256, name="expected checkpoint SHA-256"
                )
                digest = _sha256_open_file(self._handle)
                if digest != expected:
                    raise ValueError(
                        "Wannier90 checkpoint hash mismatch: "
                        f"expected {expected}, got {digest}"
                    )
            self.metadata = self._parse_metadata(digest)
        except Exception:
            self._handle.close()
            raise

    def __enter__(self) -> "Wannier90CheckpointReader":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def _detect_byte_order(self) -> str:
        self._handle.seek(0)
        marker = self._handle.read(4)
        if len(marker) != 4:
            raise ValueError("truncated Wannier90 checkpoint marker")
        little = struct.unpack("<i", marker)[0]
        big = struct.unpack(">i", marker)[0]
        if little == 33:
            return "<"
        if big == 33:
            return ">"
        raise ValueError("unrecognized Wannier90 checkpoint record markers")

    def _index_records(self) -> tuple[_FortranRecord, ...]:
        handle = self._handle
        handle.seek(0, 2)
        file_size = handle.tell()
        handle.seek(0)
        records: list[_FortranRecord] = []
        integer = self._byte_order + "i"
        while handle.tell() < file_size:
            prefix = handle.read(4)
            if len(prefix) != 4:
                raise ValueError("truncated Wannier90 checkpoint prefix marker")
            size = struct.unpack(integer, prefix)[0]
            if size < 0:
                raise ValueError("negative Wannier90 checkpoint record size")
            payload_offset = handle.tell()
            if payload_offset + size + 4 > file_size:
                raise ValueError("Wannier90 checkpoint record exceeds file size")
            handle.seek(size, 1)
            suffix = handle.read(4)
            if len(suffix) != 4 or struct.unpack(integer, suffix)[0] != size:
                raise ValueError("Wannier90 checkpoint record markers disagree")
            records.append(_FortranRecord(payload_offset, size))
        if handle.tell() != file_size:
            raise ValueError("Wannier90 checkpoint has trailing partial content")
        return tuple(records)

    def _read_record_bytes(self, index: int) -> bytes:
        record = self._records[index]
        self._handle.seek(record.payload_offset)
        data = self._handle.read(record.size_bytes)
        if len(data) != record.size_bytes:
            raise ValueError(f"truncated Wannier90 checkpoint record {index}")
        return data

    def _read_int_record(self, index: int) -> Array:
        return np.frombuffer(
            self._read_record_bytes(index), dtype=self._byte_order + "i4"
        ).copy()

    def _read_float_record(self, index: int) -> Array:
        return np.frombuffer(
            self._read_record_bytes(index), dtype=self._byte_order + "f8"
        ).copy()

    def _parse_metadata(self, digest: str | None) -> Wannier90CheckpointMetadata:
        if len(self._records) < 18:
            raise ValueError("Wannier90 checkpoint lacks required gauge records")
        header = self._read_record_bytes(0).decode("ascii").rstrip()
        num_bands_values = self._read_int_record(1)
        exclude_count_values = self._read_int_record(2)
        if num_bands_values.size != 1 or exclude_count_values.size != 1:
            raise ValueError("invalid Wannier90 checkpoint scalar records")
        num_bands = int(num_bands_values[0])
        exclude_count = int(exclude_count_values[0])
        excluded = self._read_int_record(3)
        if exclude_count < 0 or excluded.size != exclude_count:
            raise ValueError("invalid excluded-band record in Wannier90 checkpoint")

        real_lattice_raw = self._read_float_record(4)
        reciprocal_raw = self._read_float_record(5)
        if real_lattice_raw.size != 9 or reciprocal_raw.size != 9:
            raise ValueError("invalid lattice records in Wannier90 checkpoint")
        real_lattice = real_lattice_raw.reshape((3, 3), order="F")
        reciprocal = reciprocal_raw.reshape((3, 3), order="F")

        num_kpts_values = self._read_int_record(6)
        mp_values = self._read_int_record(7)
        if num_kpts_values.size != 1 or mp_values.size != 3:
            raise ValueError("invalid k-mesh records in Wannier90 checkpoint")
        num_kpts = int(num_kpts_values[0])
        mp_grid = tuple(int(value) for value in mp_values)
        if num_kpts <= 0 or any(value <= 0 for value in mp_grid):
            raise ValueError("Wannier90 checkpoint has a nonpositive k mesh")
        kpoints_raw = self._read_float_record(8)
        if kpoints_raw.size != 3 * num_kpts:
            raise ValueError("invalid k-point record in Wannier90 checkpoint")
        kpoints = kpoints_raw.reshape((3, num_kpts), order="F").T.copy()

        nntot_values = self._read_int_record(9)
        num_wann_values = self._read_int_record(10)
        if nntot_values.size != 1 or num_wann_values.size != 1:
            raise ValueError("invalid Wannier90 dimension records")
        nntot = int(nntot_values[0])
        num_wann = int(num_wann_values[0])
        if nntot <= 0 or num_wann <= 0 or num_wann > num_bands:
            raise ValueError("invalid Wannier90 checkpoint dimensions")
        checkpoint = self._read_record_bytes(11).decode("ascii").strip()
        disentangled_values = self._read_int_record(12)
        if disentangled_values.size != 1:
            raise ValueError("invalid disentanglement flag record")
        have_disentangled = bool(disentangled_values[0])
        if not have_disentangled:
            raise ValueError(
                "this TPT bridge requires the executed disentangled Wannier gauge"
            )

        omega_values = self._read_float_record(13)
        if omega_values.size != 1:
            raise ValueError("invalid omega-invariant record")
        lwindow_raw = self._read_int_record(14)
        if lwindow_raw.size != num_bands * num_kpts:
            raise ValueError("invalid lwindow record")
        # GNU Fortran commonly writes TRUE as 1, while Intel Fortran writes
        # it as -1 in unformatted logical records.  Accept only those two
        # compiler representations plus zero for FALSE.
        if np.any((lwindow_raw != 0) & (lwindow_raw != 1) & (lwindow_raw != -1)):
            raise ValueError("lwindow contains nonlogical values")
        lwindow = lwindow_raw.reshape((num_bands, num_kpts), order="F").astype(
            bool, copy=False
        )
        ndimwin = self._read_int_record(15).astype(np.int64, copy=False)
        if ndimwin.size != num_kpts:
            raise ValueError("invalid ndimwin record")
        counts = lwindow.sum(axis=0, dtype=np.int64)
        if not np.array_equal(counts, ndimwin):
            raise ValueError("lwindow counts disagree with ndimwin")
        if np.any(ndimwin < num_wann) or np.any(ndimwin > num_bands):
            raise ValueError("outer-window dimensions cannot contain num_wann states")

        expected_u_opt = num_bands * num_wann * num_kpts * 16
        expected_u = num_wann * num_wann * num_kpts * 16
        if self._records[16].size_bytes != expected_u_opt:
            raise ValueError("unexpected U_matrix_opt checkpoint record size")
        if self._records[17].size_bytes != expected_u:
            raise ValueError("unexpected U_matrix checkpoint record size")

        return Wannier90CheckpointMetadata(
            header=header,
            byte_order=self._byte_order,
            sha256=digest,
            num_bands=num_bands,
            exclude_bands_1based=tuple(int(value) for value in excluded),
            real_lattice_angstrom=real_lattice,
            reciprocal_lattice_angstrom_inv=reciprocal,
            num_kpts=num_kpts,
            mp_grid=mp_grid,
            kpoints_fractional=kpoints,
            nntot=nntot,
            num_wann=num_wann,
            checkpoint=checkpoint,
            have_disentangled=True,
            omega_invariant=float(omega_values[0]),
            lwindow=lwindow,
            ndimwin=ndimwin,
        )

    def _read_complex_k_block(
        self,
        *,
        record_index: int,
        shape: tuple[int, int],
        kpoint_index: int,
    ) -> Array:
        ik = int(kpoint_index)
        if not 0 <= ik < self.metadata.num_kpts:
            raise IndexError("checkpoint k-point index is out of range")
        count = int(np.prod(shape))
        record = self._records[record_index]
        self._handle.seek(record.payload_offset + ik * count * 16)
        data = np.fromfile(
            self._handle,
            dtype=self._byte_order + "c16",
            count=count,
        )
        if data.size != count:
            raise ValueError("truncated Wannier90 checkpoint gauge block")
        return data.reshape(shape, order="F")

    def read_u_matrix_opt(self, kpoint_index: int) -> Array:
        """Return the stored compact outer-window map for one k point."""

        return self._read_complex_k_block(
            record_index=16,
            shape=(self.metadata.num_bands, self.metadata.num_wann),
            kpoint_index=kpoint_index,
        )

    def read_u_matrix(self, kpoint_index: int) -> Array:
        """Return the final ``num_wann x num_wann`` Wannier rotation."""

        return self._read_complex_k_block(
            record_index=17,
            shape=(self.metadata.num_wann, self.metadata.num_wann),
            kpoint_index=kpoint_index,
        )

    def dft_to_wannier_gauge(self, kpoint_index: int) -> Array:
        """Return the full-band DFT-to-Wannier gauge ``U_dis U``.

        Wannier90 stores ``U_matrix_opt`` compactly in rows
        ``1:ndimwin(k)``.  ``lwindow`` identifies which original DFT bands
        those rows represent.
        """

        ik = int(kpoint_index)
        outer = np.flatnonzero(self.metadata.lwindow[:, ik])
        ndim = int(self.metadata.ndimwin[ik])
        u_opt = self.read_u_matrix_opt(ik)[:ndim, :]
        u = self.read_u_matrix(ik)
        compact = u_opt @ u
        gauge = np.zeros(
            (self.metadata.num_bands, self.metadata.num_wann),
            dtype=np.complex128,
        )
        gauge[outer, :] = compact
        return gauge


def read_wannier90_eig(
    path: str | Path,
    *,
    num_bands: int,
    num_kpts: int,
    expected_sha256: str | None = None,
) -> Array:
    """Read ``wannier90.eig`` as ``eigenvalues[k, band]`` in eV."""

    eig_path = Path(path).expanduser().resolve()
    with eig_path.open("rb") as handle:
        if expected_sha256 is not None:
            expected = _require_sha256(expected_sha256, name="expected eig SHA-256")
            actual = _sha256_open_file(handle)
            if actual != expected:
                raise ValueError(
                    f"wannier90.eig hash mismatch: expected {expected}, got {actual}"
                )
    raw = np.fromfile(eig_path, dtype=np.float64, sep=" ")
    expected_values = int(num_bands) * int(num_kpts) * 3
    if raw.size != expected_values:
        raise ValueError(
            f"wannier90.eig has {raw.size} values, expected {expected_values}"
        )
    rows = raw.reshape((num_kpts, num_bands, 3))
    expected_bands = np.arange(1, num_bands + 1, dtype=np.float64)[None, :]
    expected_k = np.arange(1, num_kpts + 1, dtype=np.float64)[:, None]
    if not np.array_equal(rows[:, :, 0], np.broadcast_to(expected_bands, rows[:, :, 0].shape)):
        raise ValueError("wannier90.eig band indices are not ordered band-fastest")
    if not np.array_equal(rows[:, :, 1], np.broadcast_to(expected_k, rows[:, :, 1].shape)):
        raise ValueError("wannier90.eig k-point indices are not ordered k-slowest")
    energies = np.asarray(rows[:, :, 2], dtype=np.float64)
    if not np.all(np.isfinite(energies)):
        raise ValueError("wannier90.eig contains nonfinite energies")
    return energies


@dataclass(frozen=True)
class Wannier90MmnBlock:
    """One raw DFT-band neighbour-overlap block from ``wannier90.mmn``."""

    source_k_0based: int
    target_k_0based: int
    target_cell_shift_integer: tuple[int, int, int]
    overlap_source_bra_target_ket: Array


def iter_wannier90_mmn_blocks(
    path: str | Path,
    *,
    expected_num_bands: int | None = None,
    expected_num_kpts: int | None = None,
    expected_nntot: int | None = None,
    expected_sha256: str | None = None,
) -> Iterator[Wannier90MmnBlock]:
    """Yield raw ``M_mn(k,b)=<u_mk|u_n,k+b>`` blocks sequentially.

    The 15 GB TPT ``.mmn`` is intentionally streamed.  Callers may stop after
    the six blocks for one source k point without indexing or loading the full
    file.  Matrix ordering follows Wannier90 ``src/overlap.F90``: source-bra
    band ``m`` varies fastest inside target-ket band ``n``.
    """

    mmn_path = Path(path).expanduser().resolve()
    with mmn_path.open("rb", buffering=0) as handle:
        if expected_sha256 is not None:
            actual_sha256 = _sha256_open_file(handle)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "wannier90.mmn hash mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        comment = handle.readline()
        if not comment:
            raise ValueError("empty wannier90.mmn")
        dimensions_line = handle.readline()
        try:
            dimensions = tuple(int(value) for value in dimensions_line.split())
        except ValueError as exc:
            raise ValueError("invalid wannier90.mmn dimension line") from exc
        if len(dimensions) != 3:
            raise ValueError("wannier90.mmn dimension line must contain three integers")
        num_bands, num_kpts, nntot = dimensions
        for actual, expected, name in (
            (num_bands, expected_num_bands, "num_bands"),
            (num_kpts, expected_num_kpts, "num_kpts"),
            (nntot, expected_nntot, "nntot"),
        ):
            if expected is not None and actual != int(expected):
                raise ValueError(
                    f"wannier90.mmn {name} mismatch: expected {expected}, got {actual}"
                )
        if num_bands <= 0 or num_kpts <= 0 or nntot <= 0:
            raise ValueError("wannier90.mmn dimensions must be positive")
        for block_index in range(num_kpts * nntot):
            header = handle.readline()
            while header and not header.strip():
                header = handle.readline()
            if not header:
                raise ValueError(f"truncated wannier90.mmn block header {block_index}")
            try:
                values = tuple(int(value) for value in header.split())
            except ValueError as exc:
                raise ValueError(
                    f"invalid wannier90.mmn block header {block_index}"
                ) from exc
            if len(values) != 5:
                raise ValueError(
                    f"wannier90.mmn block header {block_index} must contain five integers"
                )
            source_1based, target_1based, gx, gy, gz = values
            count = 2 * num_bands * num_bands
            raw = np.fromfile(handle, dtype=np.float64, count=count, sep=" ")
            if raw.size != count:
                raise ValueError(f"truncated wannier90.mmn matrix block {block_index}")
            matrix = (
                raw.reshape((num_bands, num_bands, 2), order="C")[:, :, 0]
                + 1j
                * raw.reshape((num_bands, num_bands, 2), order="C")[:, :, 1]
            )
            # The text loop is n outer, m inner.  A C reshape above is
            # [n,m], hence transpose to the documented [m,n] matrix.
            matrix = matrix.T.copy()
            if not (1 <= source_1based <= num_kpts and 1 <= target_1based <= num_kpts):
                raise ValueError("wannier90.mmn k-point index is out of range")
            yield Wannier90MmnBlock(
                source_k_0based=source_1based - 1,
                target_k_0based=target_1based - 1,
                target_cell_shift_integer=(gx, gy, gz),
                overlap_source_bra_target_ket=matrix,
            )
        trailing = handle.read()
        if trailing.strip():
            raise ValueError("wannier90.mmn contains trailing content after declared blocks")


@dataclass(frozen=True)
class WavecarMetadata:
    """Lightweight metadata from a direct-access VASP WAVECAR."""

    record_length_bytes: int
    spin_channels: int
    format_tag: int
    num_kpts: int
    num_bands: int
    encut_ev: float
    lattice_angstrom: Array
    reciprocal_lattice_angstrom_inv: Array
    volume_angstrom3: float
    efermi_ev: float
    kpoints_fractional: Array
    plane_wave_coefficients_per_k: Array
    g_vectors_per_k: Array
    eigenvalues_ev: Array
    occupations: Array
    coefficient_dtype: np.dtype


class WavecarSpinorReader:
    """Streaming reader for a noncollinear VASP WAVECAR.

    Only requested bands are read.  The reader never stores all 600 bands or
    all 192 k points at once.
    """

    _C = 0.262465831  # VASP-compatible 2m/hbar^2 in eV^-1 Angstrom^-2

    def __init__(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self._handle = self.path.open("rb", buffering=0)
        try:
            digest = None
            if expected_sha256 is not None:
                expected = _require_sha256(
                    expected_sha256, name="expected WAVECAR SHA-256"
                )
                digest = _sha256_open_file(self._handle)
                if digest != expected:
                    raise ValueError(
                        f"WAVECAR hash mismatch: expected {expected}, got {digest}"
                    )
            self.sha256 = digest
            self.metadata = self._parse_metadata()
        except Exception:
            self._handle.close()
            raise

    def __enter__(self) -> "WavecarSpinorReader":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def _read_record_doubles(self, record_index: int, count: int) -> Array:
        self._handle.seek(int(record_index) * self._record_length)
        values = np.fromfile(self._handle, dtype="<f8", count=int(count))
        if values.size != count:
            raise ValueError(f"truncated WAVECAR direct-access record {record_index}")
        return values

    @staticmethod
    def _reciprocal_rows(lattice: Array) -> tuple[Array, float]:
        volume = float(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError("WAVECAR lattice must be right-handed with positive volume")
        reciprocal = np.asarray(
            [
                np.cross(lattice[1], lattice[2]),
                np.cross(lattice[2], lattice[0]),
                np.cross(lattice[0], lattice[1]),
            ]
        )
        reciprocal *= 2.0 * np.pi / volume
        return reciprocal, volume

    def _generate_nbmax(self, reciprocal: Array, encut_ev: float) -> Array:
        bmag = np.linalg.norm(reciprocal, axis=1)
        if np.any(bmag <= 0.0):
            raise ValueError("WAVECAR reciprocal lattice is singular")
        b = reciprocal
        phi12 = np.arccos(np.dot(b[0], b[1]) / (bmag[0] * bmag[1]))
        sphi123 = np.dot(b[2], np.cross(b[0], b[1])) / (
            bmag[2] * np.linalg.norm(np.cross(b[0], b[1]))
        )
        first = np.sqrt(encut_ev * self._C) / bmag
        first[0] /= abs(np.sin(phi12))
        first[1] /= abs(np.sin(phi12))
        first[2] /= abs(sphi123)
        first += 1.0

        phi13 = np.arccos(np.dot(b[0], b[2]) / (bmag[0] * bmag[2]))
        sphi123 = np.dot(b[1], np.cross(b[0], b[2])) / (
            bmag[1] * np.linalg.norm(np.cross(b[0], b[2]))
        )
        second = np.sqrt(encut_ev * self._C) / bmag
        second[0] /= abs(np.sin(phi13))
        second[1] /= abs(sphi123)
        second[2] /= abs(np.sin(phi13))
        second += 1.0

        phi23 = np.arccos(np.dot(b[1], b[2]) / (bmag[1] * bmag[2]))
        sphi123 = np.dot(b[0], np.cross(b[1], b[2])) / (
            bmag[0] * np.linalg.norm(np.cross(b[1], b[2]))
        )
        third = np.sqrt(encut_ev * self._C) / bmag
        third[0] /= abs(sphi123)
        third[1] /= abs(np.sin(phi23))
        third[2] /= abs(np.sin(phi23))
        third += 1.0
        return np.max([first, second, third], axis=0).astype(np.int64)

    @staticmethod
    def _generate_g_vectors(
        kpoint_fractional: Array,
        *,
        reciprocal_rows: Array,
        encut_ev: float,
        nbmax: Array,
        constant_c: float,
    ) -> Array:
        vectors: list[tuple[int, int, int]] = []
        for i in range(2 * int(nbmax[2]) + 1):
            i3 = i - 2 * int(nbmax[2]) - 1 if i > int(nbmax[2]) else i
            for j in range(2 * int(nbmax[1]) + 1):
                j2 = j - 2 * int(nbmax[1]) - 1 if j > int(nbmax[1]) else j
                for k in range(2 * int(nbmax[0]) + 1):
                    k1 = k - 2 * int(nbmax[0]) - 1 if k > int(nbmax[0]) else k
                    candidate = np.asarray((k1, j2, i3), dtype=np.float64)
                    cartesian = (candidate + kpoint_fractional) @ reciprocal_rows
                    energy = float(np.dot(cartesian, cartesian) / constant_c)
                    if encut_ev > energy:
                        vectors.append((k1, j2, i3))
        return np.asarray(vectors, dtype=np.int64)

    def _parse_metadata(self) -> WavecarMetadata:
        self._handle.seek(0)
        header = np.fromfile(self._handle, dtype="<f8", count=3)
        if header.size != 3:
            raise ValueError("truncated WAVECAR first record")
        record_length, spin_channels, format_tag = (int(value) for value in header)
        if record_length <= 0 or record_length % 8 != 0:
            raise ValueError("invalid WAVECAR direct-access record length")
        if spin_channels != 1:
            raise ValueError("TPT spinor WAVECAR must have one noncollinear spin channel")
        if format_tag not in (45200, 45210):
            raise ValueError("unsupported WAVECAR coefficient precision/tag")
        self._record_length = record_length
        second = self._read_record_doubles(1, 13)
        num_kpts, num_bands = int(second[0]), int(second[1])
        encut_ev = float(second[2])
        lattice = np.asarray(second[3:12], dtype=np.float64).reshape((3, 3))
        efermi = float(second[12])
        if num_kpts <= 0 or num_bands <= 0 or encut_ev <= 0.0:
            raise ValueError("invalid WAVECAR dimensions or energy cutoff")
        reciprocal, volume = self._reciprocal_rows(lattice)
        nbmax = self._generate_nbmax(reciprocal, encut_ev)

        kpoints = np.empty((num_kpts, 3), dtype=np.float64)
        ncoeff = np.empty(num_kpts, dtype=np.int64)
        ng = np.empty(num_kpts, dtype=np.int64)
        eigenvalues = np.empty((num_kpts, num_bands), dtype=np.float64)
        occupations = np.empty((num_kpts, num_bands), dtype=np.float64)
        for ik in range(num_kpts):
            record = 2 + ik * (num_bands + 1)
            raw = self._read_record_doubles(record, 4 + 3 * num_bands)
            ncoeff[ik] = int(raw[0])
            kpoints[ik] = raw[1:4]
            bands = raw[4:].reshape((num_bands, 3))
            eigenvalues[ik] = bands[:, 0]
            occupations[ik] = bands[:, 2]
            generated = self._generate_g_vectors(
                kpoints[ik],
                reciprocal_rows=reciprocal,
                encut_ev=encut_ev,
                nbmax=nbmax,
                constant_c=self._C,
            )
            ng[ik] = generated.shape[0]
            if ncoeff[ik] != 2 * ng[ik]:
                raise ValueError(
                    "WAVECAR is not the expected noncollinear spinor layout at "
                    f"k={ik}: coefficients={ncoeff[ik]}, G={ng[ik]}"
                )
        coefficient_dtype = np.dtype("<c8" if format_tag == 45200 else "<c16")
        return WavecarMetadata(
            record_length_bytes=record_length,
            spin_channels=spin_channels,
            format_tag=format_tag,
            num_kpts=num_kpts,
            num_bands=num_bands,
            encut_ev=encut_ev,
            lattice_angstrom=lattice,
            reciprocal_lattice_angstrom_inv=reciprocal,
            volume_angstrom3=volume,
            efermi_ev=efermi,
            kpoints_fractional=kpoints,
            plane_wave_coefficients_per_k=ncoeff,
            g_vectors_per_k=ng,
            eigenvalues_ev=eigenvalues,
            occupations=occupations,
            coefficient_dtype=coefficient_dtype,
        )

    def g_vectors(self, kpoint_index: int) -> Array:
        ik = int(kpoint_index)
        if not 0 <= ik < self.metadata.num_kpts:
            raise IndexError("WAVECAR k-point index is out of range")
        nbmax = self._generate_nbmax(
            self.metadata.reciprocal_lattice_angstrom_inv,
            self.metadata.encut_ev,
        )
        vectors = self._generate_g_vectors(
            self.metadata.kpoints_fractional[ik],
            reciprocal_rows=self.metadata.reciprocal_lattice_angstrom_inv,
            encut_ev=self.metadata.encut_ev,
            nbmax=nbmax,
            constant_c=self._C,
        )
        if vectors.shape != (int(self.metadata.g_vectors_per_k[ik]), 3):
            raise RuntimeError("regenerated WAVECAR G-vector inventory changed")
        return vectors

    def read_spinor_coefficients(
        self,
        kpoint_index: int,
        band_indices_0based: Sequence[int],
    ) -> Array:
        """Return selected coefficients as ``[band, spinor, G]``."""

        ik = int(kpoint_index)
        if not 0 <= ik < self.metadata.num_kpts:
            raise IndexError("WAVECAR k-point index is out of range")
        bands = tuple(int(value) for value in band_indices_0based)
        if not bands:
            raise ValueError("at least one WAVECAR band must be requested")
        if len(set(bands)) != len(bands):
            raise ValueError("duplicate WAVECAR band indices are not permitted")
        if any(value < 0 or value >= self.metadata.num_bands for value in bands):
            raise IndexError("WAVECAR band index is out of range")
        ng = int(self.metadata.g_vectors_per_k[ik])
        ncoeff = 2 * ng
        coefficient_size = self.metadata.coefficient_dtype.itemsize
        if ncoeff * coefficient_size > self.metadata.record_length_bytes:
            raise ValueError("WAVECAR coefficient record exceeds direct-access length")
        output = np.empty(
            (len(bands), 2, ng), dtype=self.metadata.coefficient_dtype
        )
        first_band_record = 3 + ik * (self.metadata.num_bands + 1)
        for output_index, band in enumerate(bands):
            self._handle.seek(
                (first_band_record + band) * self.metadata.record_length_bytes
            )
            values = np.fromfile(
                self._handle,
                dtype=self.metadata.coefficient_dtype,
                count=ncoeff,
            )
            if values.size != ncoeff:
                raise ValueError("truncated WAVECAR coefficient record")
            output[output_index] = values.reshape((2, ng))
        if not np.all(np.isfinite(output)):
            raise ValueError("WAVECAR coefficient record contains nonfinite values")
        return output


def write_projected_spinor_wavecar(
    source_path: str | Path,
    output_path: str | Path,
    *,
    dft_to_active_gauge_by_k: Array,
    active_eigenvalues_ev: Array,
    active_occupations: Array,
    expected_source_sha256: str | None = None,
    output_coefficient_dtype: str = "<c16",
    projection_workers: int = 1,
) -> dict[str, object]:
    """Write a restart-only WAVECAR containing linear-combination active states.

    This is a basis transformation of the pseudo-wavefunction coefficients,
    not a density-vertex approximation.  When the resulting WAVECAR is passed
    back through VASP's ``PEAD_CALC_OVERLAP(..., LQIJB=.TRUE.)`` path, PAW
    projector overlaps and augmentation are recomputed for the transformed
    states.  Linearity then gives the same active-space PAW matrix as
    ``A_source^dagger M_DFT A_target``, up to WAVECAR coefficient precision.
    """

    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing WAVECAR: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with WavecarSpinorReader(
        source_path,
        expected_sha256=expected_source_sha256,
    ) as source:
        metadata = source.metadata
        gauge = np.asarray(dft_to_active_gauge_by_k, dtype=np.complex128)
        energies = np.asarray(active_eigenvalues_ev, dtype=np.float64)
        occupations = np.asarray(active_occupations, dtype=np.float64)
        if gauge.ndim != 3:
            raise ValueError("active gauge must have shape (k,dft_band,active_band)")
        num_kpts, num_dft_bands, num_active = gauge.shape
        if (num_kpts, num_dft_bands) != (
            metadata.num_kpts,
            metadata.num_bands,
        ):
            raise ValueError("active gauge does not match source WAVECAR dimensions")
        if num_active <= 0:
            raise ValueError("projected WAVECAR must contain active bands")
        if energies.shape != (num_kpts, num_active):
            raise ValueError("active eigenvalues do not match active gauge")
        if occupations.shape != (num_kpts, num_active):
            raise ValueError("active occupations do not match active gauge")
        if not all(
            np.all(np.isfinite(array)) for array in (gauge, energies, occupations)
        ):
            raise ValueError("projected WAVECAR inputs must be finite")
        if np.any(occupations < 0.0) or np.any(occupations > 1.0):
            raise ValueError("spinor active occupations must lie in [0,1]")
        semiunitarity = np.max(
            np.abs(
                np.einsum("kba,kbc->kac", gauge.conj(), gauge, optimize=True)
                - np.eye(num_active)[None, :, :]
            )
        )
        if float(semiunitarity) > 1.0e-10:
            raise ValueError("active gauge is not semiunitary")

        requested_dtype = np.dtype(output_coefficient_dtype)
        if requested_dtype not in (np.dtype("<c8"), np.dtype("<c16")):
            raise ValueError("output WAVECAR coefficients must be <c8 or <c16")
        source_dtype = metadata.coefficient_dtype
        precision_ratio = requested_dtype.itemsize // source_dtype.itemsize
        if requested_dtype.itemsize < source_dtype.itemsize:
            precision_ratio = 1
        record_length = metadata.record_length_bytes * precision_ratio
        coefficient_dtype = requested_dtype
        format_tag = 45200 if coefficient_dtype == np.dtype("<c8") else 45210
        final_record_count = 2 + num_kpts * (num_active + 1)
        workers = int(projection_workers)
        if not 1 <= workers <= num_kpts:
            raise ValueError("projection_workers must lie in [1,num_kpoints]")
        supports = tuple(
            np.flatnonzero(np.any(gauge[ik] != 0.0, axis=1))
            for ik in range(num_kpts)
        )
        if any(selected.size == 0 for selected in supports):
            raise ValueError("active gauge has empty support at one or more k points")
        initial_source_stat = source.path.stat()
        thread_state = threading.local()
        worker_readers: list[WavecarSpinorReader] = []
        worker_readers_lock = threading.Lock()

        def projected_coefficients(ik: int) -> Array:
            if workers == 1:
                reader = source
            else:
                reader = getattr(thread_state, "reader", None)
                if reader is None:
                    reader = WavecarSpinorReader(source.path)
                    if (
                        reader.metadata.num_kpts != metadata.num_kpts
                        or reader.metadata.num_bands != metadata.num_bands
                        or reader.metadata.record_length_bytes
                        != metadata.record_length_bytes
                    ):
                        reader.close()
                        raise RuntimeError("parallel WAVECAR reader metadata changed")
                    thread_state.reader = reader
                    with worker_readers_lock:
                        worker_readers.append(reader)
            selected = supports[ik]
            coefficients = reader.read_spinor_coefficients(ik, selected)
            active_coefficients = project_spinor_plane_wave_coefficients(
                coefficients,
                gauge[ik, selected],
            ).astype(coefficient_dtype, copy=False)
            expected_shape = (
                num_active,
                2,
                int(metadata.g_vectors_per_k[ik]),
            )
            if active_coefficients.shape != expected_shape:
                raise RuntimeError("projected coefficient shape changed")
            return active_coefficients

        executor: ThreadPoolExecutor | None = None
        try:
            if workers > 1:
                executor = ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="wavecar-project",
                )
                futures = {
                    ik: executor.submit(projected_coefficients, ik)
                    for ik in range(min(workers, num_kpts))
                }
            else:
                futures = {}
            with destination.open("xb", buffering=0) as handle:
                def write_record(record_index: int, values: Array) -> None:
                    raw = np.ascontiguousarray(values)
                    if raw.nbytes > record_length:
                        raise ValueError("projected WAVECAR record exceeds source RECL")
                    handle.seek(record_index * record_length)
                    raw.tofile(handle)

                write_record(
                    0,
                    np.asarray(
                        (record_length, metadata.spin_channels, format_tag),
                        dtype="<f8",
                    ),
                )
                write_record(
                    1,
                    np.concatenate(
                        (
                            np.asarray(
                                (num_kpts, num_active, metadata.encut_ev),
                                dtype="<f8",
                            ),
                            metadata.lattice_angstrom.reshape(-1).astype("<f8"),
                            np.asarray((metadata.efermi_ev,), dtype="<f8"),
                        )
                    ),
                )
                for ik in range(num_kpts):
                    if executor is None:
                        active_coefficients = projected_coefficients(ik)
                    else:
                        active_coefficients = futures.pop(ik).result()
                        next_ik = ik + workers
                        if next_ik < num_kpts:
                            futures[next_ik] = executor.submit(
                                projected_coefficients,
                                next_ik,
                            )
                    ncoeff = int(metadata.plane_wave_coefficients_per_k[ik])
                    band_metadata = np.empty((num_active, 3), dtype="<f8")
                    band_metadata[:, 0] = energies[ik]
                    band_metadata[:, 1] = 0.0
                    band_metadata[:, 2] = occupations[ik]
                    k_record = 2 + ik * (num_active + 1)
                    write_record(
                        k_record,
                        np.concatenate(
                            (
                                np.asarray((ncoeff,), dtype="<f8"),
                                metadata.kpoints_fractional[ik].astype("<f8"),
                                band_metadata.reshape(-1),
                            )
                        ),
                    )
                    for ia in range(num_active):
                        write_record(
                            k_record + 1 + ia,
                            active_coefficients[ia].reshape(-1),
                        )
                handle.truncate(final_record_count * record_length)
            final_source_stat = source.path.stat()
            stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if any(
                getattr(initial_source_stat, field) != getattr(final_source_stat, field)
                for field in stat_fields
            ):
                raise RuntimeError("source WAVECAR changed during active projection")
            destination.chmod(0o444)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            for reader in worker_readers:
                reader.close()
    return {
        "path": str(destination),
        "record_length_bytes": record_length,
        "format_tag": format_tag,
        "coefficient_dtype": str(coefficient_dtype),
        "num_kpoints": num_kpts,
        "source_num_bands": num_dft_bands,
        "active_num_bands": num_active,
        "projection_workers": workers,
        "max_gauge_semiunitarity_residual": float(semiunitarity),
        "size_bytes": destination.stat().st_size,
    }


def project_spinor_plane_wave_coefficients(
    dft_coefficients: Array,
    dft_to_active_gauge: Array,
) -> Array:
    """Project DFT spinor coefficients into active states.

    Parameters
    ----------
    dft_coefficients
        ``[band, spinor, G]`` coefficients for the selected DFT bands.
    dft_to_active_gauge
        ``[band, active]`` columns in exactly the same selected-band order.
    """

    coefficients = np.asarray(dft_coefficients)
    gauge = np.asarray(dft_to_active_gauge, dtype=np.complex128)
    if coefficients.ndim != 3 or coefficients.shape[1] != 2:
        raise ValueError("dft_coefficients must have shape (band,2,G)")
    if gauge.ndim != 2 or gauge.shape[0] != coefficients.shape[0]:
        raise ValueError("DFT coefficient and active-gauge band axes disagree")
    if gauge.shape[1] <= 0:
        raise ValueError("active gauge must contain at least one state")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(gauge)):
        raise ValueError("plane-wave coefficients and gauge must be finite")
    return np.einsum(
        "bsg,ba->asg",
        coefficients,
        gauge,
        optimize=True,
        dtype=np.complex128,
    )


def contract_spinor_plane_wave_density_vertex(
    *,
    source_coefficients: Array,
    source_g_integer: Array,
    target_coefficients: Array,
    target_g_integer: Array,
    reciprocal_shift_integer: Iterable[int],
) -> Array:
    r"""Contract one pseudo-wavefunction density vertex.

    With

    ``psi_nk = sum_{g,s} C_nks(g) exp(i(k+g).r) |s>``

    and ``k+q = k_target + K`` for integer reciprocal ``K``, the requested
    ``Q=q+G`` matrix element matches target coefficients at
    ``g_target = g_source + G + K``.  The caller supplies ``G+K`` as
    ``reciprocal_shift_integer``.
    """

    source = np.asarray(source_coefficients, dtype=np.complex128)
    target = np.asarray(target_coefficients, dtype=np.complex128)
    source_g = np.asarray(source_g_integer)
    target_g = np.asarray(target_g_integer)
    shift_raw = np.asarray(tuple(reciprocal_shift_integer))
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("active coefficients must have shape (band,spinor,G)")
    if source.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("TPT active coefficients must contain two spinor components")
    if source_g.dtype.kind not in "iu" or target_g.dtype.kind not in "iu":
        raise TypeError("plane-wave G labels must be exact integer arrays")
    if shift_raw.dtype.kind not in "iu" or shift_raw.shape != (3,):
        raise TypeError("reciprocal shift must contain three exact integers")
    source_g = source_g.astype(np.int64, copy=False)
    target_g = target_g.astype(np.int64, copy=False)
    shift = shift_raw.astype(np.int64, copy=False)
    if source_g.shape != (source.shape[2], 3):
        raise ValueError("source G inventory does not match source coefficients")
    if target_g.shape != (target.shape[2], 3):
        raise ValueError("target G inventory does not match target coefficients")
    if np.unique(source_g, axis=0).shape[0] != source_g.shape[0]:
        raise ValueError("source G inventory contains duplicates")
    if np.unique(target_g, axis=0).shape[0] != target_g.shape[0]:
        raise ValueError("target G inventory contains duplicates")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("active plane-wave coefficients must be finite")

    target_lookup = {tuple(row): index for index, row in enumerate(target_g.tolist())}
    source_indices: list[int] = []
    target_indices: list[int] = []
    for source_index, row in enumerate(source_g):
        target_index = target_lookup.get(tuple((row + shift).tolist()))
        if target_index is not None:
            source_indices.append(source_index)
            target_indices.append(target_index)
    if not source_indices:
        return np.zeros((target.shape[0], source.shape[0]), dtype=np.complex128)
    source_selected = source[:, :, np.asarray(source_indices, dtype=np.int64)]
    target_selected = target[:, :, np.asarray(target_indices, dtype=np.int64)]
    return np.einsum(
        "msg,nsg->mn",
        target_selected.conj(),
        source_selected,
        optimize=True,
    )


def reciprocal_wrap_for_mesh_transfer(
    *,
    source_k_fractional: Sequence[float],
    target_k_fractional: Sequence[float],
    q_mesh_shift: Sequence[int],
    mesh: Sequence[int],
    tolerance: float = 1.0e-10,
) -> Array:
    """Return integer ``K=k+q-k_target`` for a discrete mesh transfer."""

    source = np.asarray(source_k_fractional, dtype=np.float64)
    target = np.asarray(target_k_fractional, dtype=np.float64)
    shift = np.asarray(q_mesh_shift)
    mesh_raw = np.asarray(mesh)
    if source.shape != (3,) or target.shape != (3,):
        raise ValueError("source and target k points must be three-vectors")
    if shift.shape != (3,) or shift.dtype.kind not in "iu":
        raise TypeError("q_mesh_shift must contain three exact integers")
    if mesh_raw.shape != (3,) or mesh_raw.dtype.kind not in "iu":
        raise TypeError("mesh must contain three exact integers")
    mesh_int = mesh_raw.astype(np.int64, copy=False)
    if np.any(mesh_int <= 0):
        raise ValueError("mesh components must be positive")
    raw = source + shift.astype(np.float64) / mesh_int - target
    nearest = np.rint(raw)
    if float(np.max(np.abs(raw - nearest))) > float(tolerance):
        raise ValueError("source, q, and target do not close modulo a reciprocal vector")
    return nearest.astype(np.int64)


__all__ = [
    "TPT_BULK_POSCAR_SHA256",
    "TPT_BULK_POTCAR_SHA256",
    "TPT_BULK_WANNIER90_AMN_SHA256",
    "TPT_BULK_WANNIER90_CHK_SHA256",
    "TPT_BULK_WANNIER90_EIG_SHA256",
    "TPT_BULK_WANNIER90_MMN_SHA256",
    "TPT_BULK_WAVECAR_SHA256",
    "Wannier90CheckpointMetadata",
    "Wannier90MmnBlock",
    "Wannier90CheckpointReader",
    "WavecarMetadata",
    "WavecarSpinorReader",
    "contract_spinor_plane_wave_density_vertex",
    "iter_wannier90_mmn_blocks",
    "project_spinor_plane_wave_coefficients",
    "read_wannier90_eig",
    "reciprocal_wrap_for_mesh_transfer",
    "write_projected_spinor_wavecar",
]
