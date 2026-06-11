from __future__ import annotations

import json

from mean_field.devtools.merge_tbg_crpa_chunks import (
    _merge_workflow_manifest,
    _merge_workflow_state,
    _write_merge_workflow_artifacts,
)


def test_crpa_merge_workflow_manifest_tracks_external_chunks(tmp_path) -> None:
    chunks = (tmp_path / "chunk_0", tmp_path / "chunk_1")
    manifest = _merge_workflow_manifest(tmp_path / "merged", chunks)

    assert manifest.name == "tbg_crpa_merge"
    assert [job.name for job in manifest.jobs] == ["input_chunk_0", "input_chunk_1", "merge"]
    assert manifest.jobs[-1].dependencies == ("input_chunk_0", "input_chunk_1")
    assert manifest.jobs[-1].metadata["chunk_count"] == 2
    assert manifest.jobs[-1].command.count("--chunk") == 2


def test_crpa_merge_workflow_artifacts_roundtrip(tmp_path) -> None:
    chunks = (tmp_path / "chunk_0", tmp_path / "chunk_1")
    manifest = _merge_workflow_manifest(tmp_path / "merged", chunks)
    state = _merge_workflow_state(manifest, "running", message="unit test")

    _write_merge_workflow_artifacts(tmp_path / "merged", manifest, state)

    manifest_payload = json.loads((tmp_path / "merged" / "workflow_manifest.json").read_text(encoding="utf-8"))
    state_payload = json.loads((tmp_path / "merged" / "workflow_run_state.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "merged" / "workflow_run_state.md").read_text(encoding="utf-8")

    assert manifest_payload["jobs"][-1]["name"] == "merge"
    assert {job["name"]: job["status"] for job in state_payload["jobs"]} == {
        "input_chunk_0": "succeeded",
        "input_chunk_1": "succeeded",
        "merge": "running",
    }
    assert "[running] merge" in markdown
