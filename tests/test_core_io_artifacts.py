from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mean_field.core.io import (
    read_npz_scalar,
    summarize_npz_artifact,
    write_json_artifact,
    write_text_artifact,
)


def test_npz_artifact_summary_and_scalar_reader(tmp_path) -> None:
    path = tmp_path / "artifact.npz"
    np.savez(path, scalar=np.asarray(3.5), vector=np.asarray([1, 2, 3], dtype=np.int64))

    summary = summarize_npz_artifact(path)

    assert summary.keys == ("scalar", "vector")
    assert summary.array("vector").shape == (3,)
    assert summary.array("vector").dtype == "int64"
    assert summary.to_dict()["path"] == str(path)

    with np.load(path, allow_pickle=False) as payload:
        assert read_npz_scalar(payload, "scalar") == 3.5
        assert read_npz_scalar(payload, "vector") == [1, 2, 3]
        assert read_npz_scalar(payload, "missing", default="fallback") == "fallback"


def test_write_json_and_text_artifacts_are_stable_and_create_parents(tmp_path) -> None:
    json_path = write_json_artifact({"b": 2, "a": 1}, tmp_path / "nested" / "payload.json")
    text_path = write_text_artifact("hello\n", tmp_path / "nested" / "note.md")

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert json_path.read_text(encoding="utf-8").splitlines()[1].strip() == '"a": 1,'
    assert text_path.read_text(encoding="utf-8") == "hello\n"
    assert not (tmp_path / "nested" / "payload.json.tmp").exists()


def test_write_json_artifact_accepts_custom_encoder(tmp_path) -> None:
    def encode(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, complex):
            return [float(value.real), float(value.imag)]
        raise TypeError(f"Cannot encode {type(value).__name__}")

    path = write_json_artifact(
        {"path": tmp_path, "scalar": np.int64(7), "z": 1.5 - 2.0j},
        tmp_path / "encoded" / "payload.json",
        default=encode,
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "path": str(tmp_path),
        "scalar": 7,
        "z": [1.5, -2.0],
    }
