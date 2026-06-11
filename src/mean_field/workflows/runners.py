from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json


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


def write_workflow_manifest(manifest: WorkflowManifest, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    return output


__all__ = ["WorkflowJobSpec", "WorkflowManifest", "write_workflow_manifest"]
