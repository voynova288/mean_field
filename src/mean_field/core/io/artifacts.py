from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


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


def write_text_artifact(text: str, path: str | Path) -> Path:
    """Atomically write a small text artifact and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(output)
    return output


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
    tmp = output.with_name(output.name + ".tmp")
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
    "read_npz_scalar",
    "summarize_npz_artifact",
    "write_json_artifact",
    "write_npz_artifact",
    "write_text_artifact",
]
