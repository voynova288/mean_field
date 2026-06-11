from __future__ import annotations

import json

import pytest

from mean_field.workflows import (
    WorkflowJobSpec,
    WorkflowJobState,
    WorkflowManifest,
    WorkflowRunState,
    blocked_workflow_jobs,
    ready_workflow_jobs,
    write_workflow_manifest,
    write_workflow_run_state,
)


def _demo_manifest(tmp_path) -> WorkflowManifest:
    return WorkflowManifest(
        name="demo",
        jobs=(
            WorkflowJobSpec(name="hf", command=("python", "run.py"), output_dir=tmp_path / "hf"),
            WorkflowJobSpec(name="chern", command=("python", "chern.py"), dependencies=("hf",)),
            WorkflowJobSpec(name="plot", command=("python", "plot.py"), dependencies=("chern",)),
        ),
        root=tmp_path,
        metadata={"system": "toy"},
    )


def test_workflow_manifest_validates_dependencies_and_writes_json(tmp_path) -> None:
    manifest = _demo_manifest(tmp_path)

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


def test_workflow_state_reports_failures_and_writes_json(tmp_path) -> None:
    state = WorkflowRunState(
        name="demo",
        jobs=(
            WorkflowJobState(name="hf", status="succeeded", return_code=0),
            WorkflowJobState(name="chern", status="failed", return_code=1, message="near-zero link"),
        ),
        metadata={"attempt": 2},
    )

    assert state.failed_jobs()[0].name == "chern"
    assert "near-zero link" in state.to_markdown()

    path = write_workflow_run_state(state, tmp_path / "state.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["failed_jobs"] == ["chern"]
    assert payload["jobs"][0]["return_code"] == 0


def test_ready_workflow_jobs_use_successful_dependencies(tmp_path) -> None:
    manifest = _demo_manifest(tmp_path)
    state = WorkflowRunState(name="demo", jobs=(WorkflowJobState(name="hf", status="succeeded"),))

    ready = ready_workflow_jobs(manifest, state)

    assert [job.name for job in ready] == ["chern"]


def test_blocked_workflow_jobs_report_failed_direct_dependencies(tmp_path) -> None:
    manifest = _demo_manifest(tmp_path)
    states = {
        "hf": WorkflowJobState(name="hf", status="failed", message="SCF did not converge"),
    }

    blocked = blocked_workflow_jobs(manifest, states)

    assert [job.name for job in blocked] == ["chern"]


def test_workflow_job_state_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="Unsupported workflow job status"):
        WorkflowJobState(name="bad", status="done")  # type: ignore[arg-type]
