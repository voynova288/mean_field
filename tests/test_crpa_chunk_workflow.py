from __future__ import annotations

import json

from mean_field.devtools.run_tbg_crpa_chunk import (
    _crpa_chunk_workflow_manifest,
    _crpa_chunk_workflow_state,
    _write_crpa_chunk_workflow_artifacts,
    build_parser,
)


def test_crpa_chunk_workflow_manifest_records_command_and_metadata(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--bm-solution",
            str(tmp_path / "bm_solution.npz"),
            "--q-lg",
            "3",
            "--q-range",
            "0:2",
            "--epsilon-bn",
            "4.5",
            "--output-dir",
            str(tmp_path / "chunk_0"),
        ]
    )

    manifest = _crpa_chunk_workflow_manifest(args)

    assert manifest.name == "tbg_crpa_chunk"
    assert manifest.jobs[0].name == "crpa_chunk"
    assert manifest.jobs[0].metadata["q_range"] == "0:2"
    assert "--q-range" in manifest.jobs[0].command
    assert str(tmp_path / "bm_solution.npz") in manifest.jobs[0].command


def test_crpa_chunk_workflow_artifacts_roundtrip(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--bm-solution",
            str(tmp_path / "bm_solution.npz"),
            "--q-lg",
            "3",
            "--chunk-index",
            "1",
            "--chunk-count",
            "4",
            "--output-dir",
            str(tmp_path / "chunk_1"),
        ]
    )
    manifest = _crpa_chunk_workflow_manifest(args)
    state = _crpa_chunk_workflow_state(manifest, "running", message="unit test")

    _write_crpa_chunk_workflow_artifacts(tmp_path / "chunk_1", manifest, state)

    manifest_payload = json.loads((tmp_path / "chunk_1" / "workflow_manifest.json").read_text(encoding="utf-8"))
    state_payload = json.loads((tmp_path / "chunk_1" / "workflow_run_state.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "chunk_1" / "workflow_run_state.md").read_text(encoding="utf-8")

    assert manifest_payload["jobs"][0]["name"] == "crpa_chunk"
    assert state_payload["jobs"][0]["status"] == "running"
    assert "[running] crpa_chunk" in markdown
