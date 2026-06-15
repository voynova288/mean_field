from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal, Mapping

from mean_field.core.io import write_json_artifact

WorkflowJobStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]
TERMINAL_WORKFLOW_STATUSES: tuple[WorkflowJobStatus, ...] = ("succeeded", "failed", "skipped")
SUCCESS_WORKFLOW_STATUSES: tuple[WorkflowJobStatus, ...] = ("succeeded",)


_SLURM_METADATA_KEYS: dict[str, str] = {
    "SLURM_JOB_ID": "job_id",
    "SLURM_JOB_NAME": "job_name",
    "SLURM_JOB_PARTITION": "partition",
    "SLURM_CLUSTER_NAME": "cluster_name",
    "SLURM_SUBMIT_DIR": "submit_dir",
    "SLURM_CPUS_PER_TASK": "cpus_per_task",
    "SLURM_NTASKS": "ntasks",
    "SLURM_JOB_NODELIST": "node_list",
    "SLURM_ARRAY_JOB_ID": "array_job_id",
    "SLURM_ARRAY_TASK_ID": "array_task_id",
    "SLURM_ARRAY_TASK_COUNT": "array_task_count",
    "SLURM_ARRAY_TASK_MIN": "array_task_min",
    "SLURM_ARRAY_TASK_MAX": "array_task_max",
}


def collect_slurm_metadata(env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Return Slurm identifiers from the current environment, if present.

    The workflow layer remains scheduler-neutral, but this helper gives long
    task entrypoints one common place to capture the Slurm job/array IDs that
    are needed for later handoff, resume, or ``sacct``/``squeue`` inspection.
    Values are intentionally serialized as strings to preserve Slurm's exact
    formatting.
    """

    source = os.environ if env is None else env
    metadata = {
        payload_name: value
        for env_name, payload_name in _SLURM_METADATA_KEYS.items()
        if (value := source.get(env_name)) not in {None, ""}
    }
    if not metadata:
        return {}
    return {"scheduler": "slurm", **metadata}


@dataclass(frozen=True)
class WorkflowJobSpec:
    """Serializable description of one workflow job or postprocess stage."""

    name: str
    command: tuple[str, ...]
    output_dir: Path | None = None
    dependencies: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("WorkflowJobSpec name must be non-empty")
        if not self.command:
            raise ValueError(f"WorkflowJobSpec {self.name!r} must have a non-empty command")
        object.__setattr__(self, "command", tuple(str(part) for part in self.command))
        object.__setattr__(self, "dependencies", tuple(str(dep) for dep in self.dependencies))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": str(self.name),
            "command": list(self.command),
            "output_dir": None if self.output_dir is None else str(self.output_dir),
            "dependencies": list(self.dependencies),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WorkflowManifest:
    """Small durable manifest for config-driven workflow orchestration."""

    name: str
    jobs: tuple[WorkflowJobSpec, ...]
    root: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("WorkflowManifest name must be non-empty")
        names = [job.name for job in self.jobs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate workflow job names: {duplicates}")
        known = set(names)
        missing = sorted({dep for job in self.jobs for dep in job.dependencies if dep not in known})
        if missing:
            raise ValueError(f"Workflow dependencies reference unknown jobs: {missing}")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": str(self.name),
            "root": None if self.root is None else str(self.root),
            "metadata": dict(self.metadata),
            "jobs": [job.to_dict() for job in self.jobs],
        }


@dataclass(frozen=True)
class WorkflowJobState:
    """Serializable status record for one workflow job.

    This is intentionally scheduler-agnostic: Slurm IDs, output paths, or error
    summaries can live in ``metadata`` while the common workflow layer only
    reasons about status and dependencies.
    """

    name: str
    status: WorkflowJobStatus = "pending"
    return_code: int | None = None
    message: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("WorkflowJobState name must be non-empty")
        if self.status not in {"pending", "running", "succeeded", "failed", "skipped"}:
            raise ValueError(f"Unsupported workflow job status: {self.status!r}")
        if self.return_code is not None:
            object.__setattr__(self, "return_code", int(self.return_code))

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_WORKFLOW_STATUSES

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": str(self.name),
            "status": str(self.status),
            "return_code": self.return_code,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WorkflowRunState:
    """Minimal scheduler-neutral workflow state for resume/failure reports."""

    name: str
    jobs: tuple[WorkflowJobState, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("WorkflowRunState name must be non-empty")
        names = [job.name for job in self.jobs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate workflow state names: {duplicates}")

    def by_name(self) -> dict[str, WorkflowJobState]:
        return {job.name: job for job in self.jobs}

    def failed_jobs(self) -> tuple[WorkflowJobState, ...]:
        return tuple(job for job in self.jobs if job.failed)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": str(self.name),
            "metadata": dict(self.metadata),
            "jobs": [job.to_dict() for job in self.jobs],
            "failed_jobs": [job.name for job in self.failed_jobs()],
        }

    def to_markdown(self) -> str:
        lines = [f"# Workflow state: {self.name}", ""]
        slurm_metadata = self.metadata.get("slurm")
        if isinstance(slurm_metadata, dict) and slurm_metadata.get("job_id"):
            lines.append(f"- slurm_job_id: {slurm_metadata['job_id']}")
            if slurm_metadata.get("array_task_id"):
                lines.append(f"- slurm_array_task_id: {slurm_metadata['array_task_id']}")
            lines.append("")
        for job in self.jobs:
            suffix = f" ({job.message})" if job.message else ""
            lines.append(f"- [{job.status}] {job.name}{suffix}")
        failures = self.failed_jobs()
        lines.append("")
        lines.append(f"- failures: {len(failures)}")
        if failures:
            lines.append("- failed jobs: " + ", ".join(job.name for job in failures))
        return "\n".join(lines)


def _state_map(states: Mapping[str, WorkflowJobState] | WorkflowRunState | tuple[WorkflowJobState, ...]) -> dict[str, WorkflowJobState]:
    if isinstance(states, WorkflowRunState):
        return states.by_name()
    if isinstance(states, Mapping):
        return dict(states)
    return {state.name: state for state in states}


def ready_workflow_jobs(
    manifest: WorkflowManifest,
    states: Mapping[str, WorkflowJobState] | WorkflowRunState | tuple[WorkflowJobState, ...] = (),
    *,
    success_statuses: tuple[WorkflowJobStatus, ...] = SUCCESS_WORKFLOW_STATUSES,
) -> tuple[WorkflowJobSpec, ...]:
    """Return jobs not yet started whose dependencies have succeeded."""

    by_name = _state_map(states)
    successes = set(success_statuses)
    ready: list[WorkflowJobSpec] = []
    for job in manifest.jobs:
        current = by_name.get(job.name)
        if current is not None and current.status != "pending":
            continue
        if all((state := by_name.get(dep)) is not None and state.status in successes for dep in job.dependencies):
            ready.append(job)
    return tuple(ready)


def blocked_workflow_jobs(
    manifest: WorkflowManifest,
    states: Mapping[str, WorkflowJobState] | WorkflowRunState | tuple[WorkflowJobState, ...] = (),
) -> tuple[WorkflowJobSpec, ...]:
    """Return pending jobs with at least one failed or skipped direct dependency."""

    by_name = _state_map(states)
    blockers = {"failed", "skipped"}
    blocked: list[WorkflowJobSpec] = []
    for job in manifest.jobs:
        current = by_name.get(job.name)
        if current is not None and current.status != "pending":
            continue
        if any((state := by_name.get(dep)) is not None and state.status in blockers for dep in job.dependencies):
            blocked.append(job)
    return tuple(blocked)


def write_workflow_manifest(manifest: WorkflowManifest, path: str | Path) -> Path:
    return write_json_artifact(manifest.to_dict(), path)


def write_workflow_run_state(state: WorkflowRunState, path: str | Path) -> Path:
    return write_json_artifact(state.to_dict(), path)


__all__ = [
    "SUCCESS_WORKFLOW_STATUSES",
    "TERMINAL_WORKFLOW_STATUSES",
    "WorkflowJobSpec",
    "WorkflowJobState",
    "WorkflowJobStatus",
    "WorkflowManifest",
    "WorkflowRunState",
    "collect_slurm_metadata",
    "blocked_workflow_jobs",
    "ready_workflow_jobs",
    "write_workflow_manifest",
    "write_workflow_run_state",
]
