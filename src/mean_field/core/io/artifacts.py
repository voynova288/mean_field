from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

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
) -> Path:
    """Atomically write a JSON artifact with the repository's stable defaults."""

    return write_text_artifact(
        json.dumps(payload, indent=indent, sort_keys=sort_keys) + "\n",
        path,
    )


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
    "write_text_artifact",
]
