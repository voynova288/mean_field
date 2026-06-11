from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from mean_field.core.io import write_json_artifact

ArrayKey = tuple[int, int]


def format_bytes(num_bytes: int | float) -> str:
    """Return a compact binary-size string for diagnostics."""

    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.3g} {unit}"
        value /= 1024.0
    return f"{value:.3g} TiB"


@dataclass(frozen=True)
class HFOverlapMemoryEstimate:
    """Memory/storage estimate for dense projected-HF overlap tables.

    A dense overlap for one reciprocal shift has shape
    ``(nt, nk_target, nt, nk_source)`` and usually dominates memory.  The
    diagonal Hartree helper and Fock screening matrix are included because they
    are stored alongside the overlap table, but they are normally much smaller.
    """

    nt: int
    nk_target: int
    nk_source: int
    n_shifts: int
    overlap_dtype: str
    diagonal_dtype: str
    fock_dtype: str
    overlap_bytes_per_shift: int
    overlap_bytes_total: int
    diagonal_bytes_total: int
    fock_screening_bytes_total: int

    @property
    def cached_bytes_total(self) -> int:
        return int(self.overlap_bytes_total + self.diagonal_bytes_total + self.fock_screening_bytes_total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nt": int(self.nt),
            "nk_target": int(self.nk_target),
            "nk_source": int(self.nk_source),
            "n_shifts": int(self.n_shifts),
            "overlap_dtype": self.overlap_dtype,
            "diagonal_dtype": self.diagonal_dtype,
            "fock_dtype": self.fock_dtype,
            "overlap_bytes_per_shift": int(self.overlap_bytes_per_shift),
            "overlap_bytes_total": int(self.overlap_bytes_total),
            "diagonal_bytes_total": int(self.diagonal_bytes_total),
            "fock_screening_bytes_total": int(self.fock_screening_bytes_total),
            "cached_bytes_total": int(self.cached_bytes_total),
            "overlap_per_shift_human": format_bytes(self.overlap_bytes_per_shift),
            "overlap_total_human": format_bytes(self.overlap_bytes_total),
            "cached_total_human": format_bytes(self.cached_bytes_total),
        }


def estimate_hf_overlap_cache_bytes(
    *,
    nt: int,
    nk_target: int,
    n_shifts: int,
    nk_source: int | None = None,
    overlap_dtype: np.dtype | type = np.complex128,
    diagonal_dtype: np.dtype | type = np.complex128,
    fock_dtype: np.dtype | type = np.float64,
    include_diagonal: bool = True,
    include_fock_screening: bool = True,
) -> HFOverlapMemoryEstimate:
    """Estimate bytes for dense projected-HF overlap storage.

    Parameters are deliberately system-agnostic: ``nt`` is the flattened
    spin/flavor/band dimension, ``nk_target`` and ``nk_source`` are the number
    of target/source k-points, and ``n_shifts`` is the number of reciprocal
    vectors retained in the interaction cutoff.
    """

    nt_i = int(nt)
    nk_t = int(nk_target)
    nk_s = nk_t if nk_source is None else int(nk_source)
    n_shift_i = int(n_shifts)
    if nt_i <= 0 or nk_t <= 0 or nk_s <= 0 or n_shift_i < 0:
        raise ValueError(
            "nt, nk_target, nk_source must be positive and n_shifts non-negative; "
            f"got nt={nt_i}, nk_target={nk_t}, nk_source={nk_s}, n_shifts={n_shift_i}"
        )
    overlap_dt = np.dtype(overlap_dtype)
    diagonal_dt = np.dtype(diagonal_dtype)
    fock_dt = np.dtype(fock_dtype)
    overlap_one = int(nt_i * nk_t * nt_i * nk_s * overlap_dt.itemsize)
    diagonal_total = int(n_shift_i * nt_i * nt_i * nk_t * diagonal_dt.itemsize) if include_diagonal else 0
    fock_total = int(n_shift_i * nk_t * nk_s * fock_dt.itemsize) if include_fock_screening else 0
    return HFOverlapMemoryEstimate(
        nt=nt_i,
        nk_target=nk_t,
        nk_source=nk_s,
        n_shifts=n_shift_i,
        overlap_dtype=str(overlap_dt),
        diagonal_dtype=str(diagonal_dt),
        fock_dtype=str(fock_dt),
        overlap_bytes_per_shift=overlap_one,
        overlap_bytes_total=int(n_shift_i * overlap_one),
        diagonal_bytes_total=diagonal_total,
        fock_screening_bytes_total=fock_total,
    )


def _parse_slurm_memory_mb(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def available_memory_bytes() -> int | None:
    """Best-effort available memory estimate for cache-placement decisions.

    Preference order:
    1. Slurm memory request if available.
    2. Linux ``/proc/meminfo`` ``MemAvailable``.
    3. POSIX physical-page count.

    The result is a planning heuristic, not a hard allocation guarantee.
    """

    slurm_node_mb = _parse_slurm_memory_mb(os.environ.get("SLURM_MEM_PER_NODE"))
    if slurm_node_mb is not None:
        return int(slurm_node_mb * 1024 * 1024)
    slurm_cpu_mb = _parse_slurm_memory_mb(os.environ.get("SLURM_MEM_PER_CPU"))
    if slurm_cpu_mb is not None:
        cpus = (
            _parse_slurm_memory_mb(os.environ.get("SLURM_CPUS_PER_TASK"))
            or _parse_slurm_memory_mb(os.environ.get("SLURM_CPUS_ON_NODE"))
            or 1
        )
        return int(slurm_cpu_mb * cpus * 1024 * 1024)

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
        except OSError:
            pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        return None
    return None


def should_spill_hf_overlap_cache(
    estimate: HFOverlapMemoryEstimate,
    *,
    memory_limit_bytes: int | None = None,
    safety_fraction: float = 0.65,
    reuse_requested: bool = False,
) -> bool:
    """Return whether dense overlaps should be stored on disk instead of RAM."""

    if reuse_requested:
        return True
    limit = available_memory_bytes() if memory_limit_bytes is None else int(memory_limit_bytes)
    if limit is None or limit <= 0:
        return False
    fraction = float(safety_fraction)
    if fraction <= 0.0:
        raise ValueError(f"safety_fraction must be positive, got {safety_fraction}")
    return int(estimate.overlap_bytes_total) > int(fraction * limit)


def _key_to_name(key: ArrayKey) -> str:
    return f"shift_{int(key[0]):+d}_{int(key[1]):+d}.npy".replace("+", "p").replace("-", "m")


def _name_to_key(name: str) -> ArrayKey:
    stem = Path(name).stem
    if not stem.startswith("shift_"):
        raise ValueError(f"Invalid cached-array file name {name!r}")
    parts = stem.removeprefix("shift_").split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid cached-array file name {name!r}")

    def parse(part: str) -> int:
        if part.startswith("p"):
            return int(part[1:])
        if part.startswith("m"):
            return -int(part[1:])
        return int(part)

    return parse(parts[0]), parse(parts[1])


class DiskBackedArrayMapping(MutableMapping[ArrayKey, np.ndarray]):
    """Dictionary-like ``.npy`` array store with lazy ``np.load(..., mmap_mode)``.

    Values are not retained in Python memory.  ``__getitem__`` returns a new
    memory-mapped array view each time, so HF kernels can process one reciprocal
    shift at a time while keeping a normal ``mapping[shift]`` interface.
    """

    def __init__(self, root: str | os.PathLike[str], *, mmap_mode: str | None = "r") -> None:
        self.root = Path(root)
        self.mmap_mode = mmap_mode
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self._entries: dict[ArrayKey, dict[str, Any]] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self._manifest_path.exists():
            return
        payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        entries: dict[ArrayKey, dict[str, Any]] = {}
        for item in payload.get("arrays", []):
            key = tuple(int(x) for x in item["key"])
            if len(key) != 2:
                raise ValueError(f"Invalid array key in {self._manifest_path}: {key}")
            entries[(key[0], key[1])] = dict(item)
        self._entries = entries

    def _write_manifest(self) -> None:
        arrays = []
        for key in sorted(self._entries):
            item = dict(self._entries[key])
            item["key"] = [int(key[0]), int(key[1])]
            arrays.append(item)
        write_json_artifact({"arrays": arrays}, self._manifest_path)

    def path_for_key(self, key: ArrayKey) -> Path:
        return self.root / _key_to_name((int(key[0]), int(key[1])))

    def store(self, key: ArrayKey, value: np.ndarray, *, overwrite: bool = True) -> Path:
        normalized = (int(key[0]), int(key[1]))
        path = self.path_for_key(normalized)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        arr = np.asarray(value)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as handle:
            np.save(handle, arr, allow_pickle=False)
        os.replace(tmp, path)
        self._entries[normalized] = {
            "key": [int(normalized[0]), int(normalized[1])],
            "file": path.name,
            "shape": [int(x) for x in arr.shape],
            "dtype": str(arr.dtype),
            "nbytes": int(arr.nbytes),
        }
        self._write_manifest()
        return path

    def __getitem__(self, key: ArrayKey) -> np.ndarray:
        normalized = (int(key[0]), int(key[1]))
        item = self._entries[normalized]
        path = self.root / str(item["file"])
        return np.load(path, mmap_mode=self.mmap_mode, allow_pickle=False)

    def __setitem__(self, key: ArrayKey, value: np.ndarray) -> None:
        self.store(key, value, overwrite=True)

    def __delitem__(self, key: ArrayKey) -> None:
        normalized = (int(key[0]), int(key[1]))
        item = self._entries.pop(normalized)
        path = self.root / str(item["file"])
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        self._write_manifest()

    def __iter__(self) -> Iterator[ArrayKey]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        try:
            k0, k1 = key  # type: ignore[misc]
            normalized = (int(k0), int(k1))
        except Exception:
            return False
        return normalized in self._entries

    @classmethod
    def from_existing_files(cls, root: str | os.PathLike[str], *, mmap_mode: str | None = "r") -> "DiskBackedArrayMapping":
        mapping = cls(root, mmap_mode=mmap_mode)
        if mapping._entries:
            return mapping
        for path in sorted(mapping.root.glob("shift_*.npy")):
            key = _name_to_key(path.name)
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            mapping._entries[key] = {
                "key": [int(key[0]), int(key[1])],
                "file": path.name,
                "shape": [int(x) for x in arr.shape],
                "dtype": str(arr.dtype),
                "nbytes": int(arr.nbytes),
            }
        mapping._write_manifest()
        return mapping
