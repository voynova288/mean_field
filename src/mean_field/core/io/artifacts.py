from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any, BinaryIO, Callable

import numpy as np


def file_object_sha256(
    handle: BinaryIO,
    *,
    chunk_bytes: int = 1024 * 1024,
) -> str:
    """Hash one already-open seekable file and rewind it to byte zero."""

    if (
        isinstance(chunk_bytes, bool)
        or int(chunk_bytes) != chunk_bytes
        or chunk_bytes <= 0
    ):
        raise ValueError("chunk_bytes must be a positive integer")
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(int(chunk_bytes)), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def file_sha256(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading large artifacts at once."""

    with Path(path).open("rb") as handle:
        return file_object_sha256(handle, chunk_bytes=chunk_bytes)


def _file_snapshot(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@contextmanager
def open_stable_binary_file(path: str | Path) -> Iterator[BinaryIO]:
    """Open one inode and reject in-place changes or path replacement on exit."""

    source = Path(path)
    with source.open("rb") as handle:
        initial = os.fstat(handle.fileno())
        yield handle
        final = os.fstat(handle.fileno())
        if _file_snapshot(final) != _file_snapshot(initial):
            raise RuntimeError(f"artifact changed while open: {source}")
    if _file_snapshot(source.stat()) != _file_snapshot(initial):
        raise RuntimeError(f"artifact path changed while open: {source}")


@dataclass(frozen=True)
class ArtifactArrayInfo:
    key: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "shape": [int(v) for v in self.shape],
            "dtype": self.dtype,
            "nbytes": int(self.nbytes),
        }


@dataclass(frozen=True)
class NpzArtifactSummary:
    path: Path
    arrays: tuple[ArtifactArrayInfo, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.arrays)

    def array(self, key: str) -> ArtifactArrayInfo:
        for item in self.arrays:
            if item.key == key:
                return item
        raise KeyError(key)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "keys": list(self.keys),
            "arrays": [item.to_dict() for item in self.arrays],
        }


def _unique_temporary_path(output: Path) -> Path:
    return output.with_name(
        f".{output.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )


def unique_staging_path(destination: str | Path) -> Path:
    """Return an unused same-directory path for one private staged artifact."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        candidate = output.with_name(
            f".{output.name}.stage-{os.getpid()}-{secrets.token_hex(8)}"
        )
        if not candidate.exists():
            return candidate

def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def publish_staged_file_noreplace(
    staged_path: str | Path,
    destination: str | Path,
) -> tuple[int, int]:
    """Atomically hard-link a validated same-directory file without replacement."""

    staged = Path(staged_path)
    output = Path(destination)
    if staged.parent.resolve() != output.parent.resolve():
        raise ValueError("staged artifact and destination must share one directory")
    with staged.open("rb") as handle:
        os.fsync(handle.fileno())
        staged_identity = os.fstat(handle.fileno())
        identity = (staged_identity.st_dev, staged_identity.st_ino)
        os.link(staged, output, follow_symlinks=False)
        final_identity = os.fstat(handle.fileno())
        published_identity = os.stat(output, follow_symlinks=False)
        snapshot = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
        )
        if not (
            snapshot(staged_identity)
            == snapshot(final_identity)
            == snapshot(published_identity)
        ):
            raise RuntimeError("published artifact changed the staged inode")
    staged.unlink()
    _fsync_directory(output.parent)
    return identity

def write_text_artifact(text: str, path: str | Path) -> Path:
    """Atomically write a small text artifact and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_temporary_path(output)
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(output)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return output


def parse_json_artifact(text: str | bytes) -> object:
    """Parse strict JSON bytes/text, rejecting duplicate keys and constants."""

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"nonstandard JSON constant {value!r}")

    return json.loads(
        text,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def read_json_artifact(path: str | Path) -> object:
    """Read strict JSON, rejecting duplicate keys and nonstandard constants."""

    return parse_json_artifact(Path(path).read_bytes())


def write_json_artifact(
    payload: object,
    path: str | Path,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    default: Callable[[object], object] | None = None,
    allow_nan: bool = False,
) -> Path:
    """Atomically write a JSON artifact with the repository's stable defaults.

    ``default`` is forwarded to ``json.dumps`` so system adapters can preserve
    local encoders for paths, NumPy scalars, complex numbers, or other metadata
    without reimplementing atomic file replacement.  Public artifacts use strict
    JSON by default and therefore reject ``NaN``/``Infinity`` tokens unless a
    caller explicitly opts in via ``allow_nan=True``.
    """

    return write_text_artifact(
        json.dumps(payload, indent=indent, sort_keys=sort_keys, default=default, allow_nan=allow_nan) + "\n",
        path,
    )


def write_npz_artifact(
    arrays: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    compressed: bool = False,
) -> Path:
    """Atomically write an NPZ artifact and return its path.

    Object-dtype arrays are rejected so all public NPZ artifacts remain readable
    with ``np.load(..., allow_pickle=False)``.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"NPZ artifact keys must be non-empty strings, got {key!r}")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"NPZ artifact array {key!r} has object dtype and would require pickle")
        payload[key] = array
    tmp = _unique_temporary_path(output)
    try:
        with tmp.open("wb") as handle:
            if compressed:
                np.savez_compressed(handle, **payload)
            else:
                np.savez(handle, **payload)
        tmp.replace(output)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return output


def summarize_npz_artifact(path: str | Path) -> NpzArtifactSummary:
    artifact_path = Path(path)
    with np.load(artifact_path, allow_pickle=False) as payload:
        arrays = tuple(
            ArtifactArrayInfo(
                key=str(key),
                shape=tuple(int(v) for v in payload[key].shape),
                dtype=str(payload[key].dtype),
                nbytes=int(payload[key].nbytes),
            )
            for key in payload.files
        )
    return NpzArtifactSummary(path=artifact_path, arrays=arrays)


def read_npz_scalar(payload: Any, key: str, default: object | None = None) -> object:
    """Return a Python scalar/string from a loaded ``np.load`` payload.

    The helper intentionally accepts either an ``NpzFile`` or a plain mapping so
    archive readers can share the same scalar-normalization logic in tests.
    """

    if key not in payload:
        return default
    value = payload[key]
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


__all__ = [
    "ArtifactArrayInfo",
    "NpzArtifactSummary",
    "file_object_sha256",
    "file_sha256",
    "open_stable_binary_file",
    "parse_json_artifact",
    "publish_staged_file_noreplace",
    "read_json_artifact",
    "read_npz_scalar",
    "summarize_npz_artifact",
    "unique_staging_path",
    "write_json_artifact",
    "write_npz_artifact",
    "write_text_artifact",
]
