from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mean_field.core.io import write_json_artifact


@dataclass(frozen=True)
class ConventionBundle:
    energy_unit: str = "meV"
    length_unit: str = "nm"
    momentum_unit: str = "nm^-1"
    density_convention: str = "stored_delta"
    density_axis_order: str = "abk"
    hamiltonian_axis_order: str = "abk"
    wavefunction_axis_order: str = "k_basis_band"
    valley_labels: dict[str, int] = field(default_factory=lambda: {"K": 1, "Kprime": -1})
    spin_labels: dict[str, int] = field(default_factory=lambda: {"up": 0, "down": 1})
    gauge: str = "system_defined"

    def to_dict(self) -> dict[str, object]:
        return {
            "energy_unit": self.energy_unit,
            "length_unit": self.length_unit,
            "momentum_unit": self.momentum_unit,
            "density_convention": self.density_convention,
            "density_axis_order": self.density_axis_order,
            "hamiltonian_axis_order": self.hamiltonian_axis_order,
            "wavefunction_axis_order": self.wavefunction_axis_order,
            "valley_labels": dict(self.valley_labels),
            "spin_labels": dict(self.spin_labels),
            "gauge": self.gauge,
        }


@dataclass(frozen=True)
class ModelRecord:
    system_name: str
    params: dict[str, object] = field(default_factory=dict)
    lattice: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "system_name": str(self.system_name),
            "params": dict(self.params),
            "lattice": dict(self.lattice),
        }


@dataclass(frozen=True)
class ArtifactManifest:
    root: Path
    model: ModelRecord | None = None
    conventions: ConventionBundle = field(default_factory=ConventionBundle)
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "model": None if self.model is None else self.model.to_dict(),
            "conventions": self.conventions.to_dict(),
            "files": dict(self.files),
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else Path(self.root) / "manifest.json"
        return write_json_artifact(self.to_dict(), target)


@dataclass(frozen=True)
class ResultDirectory:
    root: Path
    manifest: dict[str, Any]
    model: dict[str, Any] | None = None
    conventions: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    import json

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_result(path: str | Path) -> ResultDirectory:
    root = Path(path)
    manifest = _read_json_if_present(root / "manifest.json") or {}
    return ResultDirectory(
        root=root,
        manifest=manifest,
        model=_read_json_if_present(root / "model.json"),
        conventions=_read_json_if_present(root / "conventions.json"),
        validation=_read_json_if_present(root / "validation.json"),
    )


def required_artifact_files() -> tuple[str, ...]:
    return (
        "manifest.json",
        "model.json",
        "config.yaml",
        "conventions.json",
        "environment.json",
        "validation.json",
        "observables.json",
    )


__all__ = [
    "ArtifactManifest",
    "ConventionBundle",
    "ModelRecord",
    "ResultDirectory",
    "load_result",
    "required_artifact_files",
]
