from __future__ import annotations

import json

import pytest

from mean_field.workflows import WorkflowJobSpec, WorkflowManifest, write_workflow_manifest


def test_workflow_manifest_validates_dependencies_and_writes_json(tmp_path) -> None:
    manifest = WorkflowManifest(
        name="demo",
        jobs=(
            WorkflowJobSpec(name="hf", command=("python", "run.py"), output_dir=tmp_path / "hf"),
            WorkflowJobSpec(name="chern", command=("python", "chern.py"), dependencies=("hf",)),
        ),
        root=tmp_path,
        metadata={"system": "toy"},
    )
    path = write_workflow_manifest(manifest, tmp_path / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["name"] == "demo"
    assert payload["jobs"][1]["dependencies"] == ["hf"]
    assert payload["metadata"] == {"system": "toy"}


def test_workflow_manifest_rejects_unknown_dependency() -> None:
    with pytest.raises(ValueError, match="unknown jobs"):
        WorkflowManifest(
            name="bad",
            jobs=(WorkflowJobSpec(name="post", command=("true",), dependencies=("missing",)),),
        )
