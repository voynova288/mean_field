from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Literal, Mapping

WorkflowJobStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]
TERMINAL_WORKFLOW_STATUSES: tuple[WorkflowJobStatus, ...] = ("succeeded", "failed", "skipped")
SUCCESS_WORKFLOW_STATUSES: tuple[WorkflowJobStatus, ...] = ("succeeded",)


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


def _write_json_atomically(payload: dict[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    return output


def write_workflow_manifest(manifest: WorkflowManifest, path: str | Path) -> Path:
    return _write_json_atomically(manifest.to_dict(), path)


def write_workflow_run_state(state: WorkflowRunState, path: str | Path) -> Path:
    return _write_json_atomically(state.to_dict(), path)


__all__ = [
    "SUCCESS_WORKFLOW_STATUSES",
    "TERMINAL_WORKFLOW_STATUSES",
    "WorkflowJobSpec",
    "WorkflowJobState",
    "WorkflowJobStatus",
    "WorkflowManifest",
    "WorkflowRunState",
    "blocked_workflow_jobs",
    "ready_workflow_jobs",
    "write_workflow_manifest",
    "write_workflow_run_state",
]
