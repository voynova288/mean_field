from __future__ import annotations

import json

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
