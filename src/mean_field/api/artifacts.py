from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mean_field.core.io import summarize_npz_artifact, write_json_artifact


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
    model: ModelRecord | Mapping[str, object] | None = None
    conventions: ConventionBundle | Mapping[str, object] = field(default_factory=ConventionBundle)
    files: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "model": None if self.model is None else _to_plain_dict(self.model),
            "conventions": _to_plain_dict(self.conventions),
            "files": dict(self.files),
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else Path(self.root) / "manifest.json"
        return write_json_artifact(self.to_dict(), target, default=_json_default)


@dataclass(frozen=True)
class ResultDirectory:
    root: Path
    manifest: dict[str, Any]
    model: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    conventions: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    observables: dict[str, Any] | None = None
    canonical_hf_run_result: dict[str, Any] | None = None


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_plain_dict(value: object) -> dict[str, object]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())  # type: ignore[union-attr]
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected mapping-like payload, got {type(value).__name__}")


def _model_payload(system_name: str, model: ModelRecord | Mapping[str, object] | None) -> dict[str, object]:
    if model is None:
        return ModelRecord(system_name=system_name).to_dict()
    payload = _to_plain_dict(model)
    payload.setdefault("system_name", str(system_name))
    payload.setdefault("params", {})
    payload.setdefault("lattice", {})
    return payload


def _convention_payload(conventions: ConventionBundle | Mapping[str, object] | None) -> dict[str, object]:
    payload = ConventionBundle().to_dict()
    if conventions is not None:
        payload.update(_to_plain_dict(conventions))
    return payload


def _relative_artifact_path(root: Path, path: str | Path) -> str:
    artifact_path = Path(path)
    if not artifact_path.is_absolute():
        artifact_path = root / artifact_path
    try:
        return artifact_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(artifact_path)


def write_contract_artifacts(
    root: str | Path,
    *,
    workflow: str,
    system_name: str,
    model: ModelRecord | Mapping[str, object] | None = None,
    config: Mapping[str, object] | None = None,
    conventions: ConventionBundle | Mapping[str, object] | None = None,
    environment: Mapping[str, object] | None = None,
    validation: Mapping[str, object] | None = None,
    observables: Mapping[str, object] | None = None,
    files: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    array_files: tuple[str | Path, ...] = (),
) -> dict[str, Path]:
    """Write the public workflow-result contract sidecars.

    The helper is intentionally metadata-only: existing NPZ/TSV/plot artifacts
    are referenced and summarized but never rewritten.
    """

    result_root = Path(root)
    result_root.mkdir(parents=True, exist_ok=True)

    model_data = _model_payload(system_name, model)
    convention_data = _convention_payload(conventions)
    config_data = dict(config or {})
    environment_data = dict(environment or {})
    validation_data = dict(validation or {})
    observables_data = dict(observables or {})

    output_paths: dict[str, Path] = {}
    output_paths["model.json"] = write_json_artifact(model_data, result_root / "model.json", default=_json_default)
    output_paths["config.yaml"] = write_json_artifact(config_data, result_root / "config.yaml", default=_json_default)
    output_paths["conventions.json"] = write_json_artifact(
        convention_data,
        result_root / "conventions.json",
        default=_json_default,
    )
    output_paths["environment.json"] = write_json_artifact(
        environment_data,
        result_root / "environment.json",
        default=_json_default,
    )
    output_paths["validation.json"] = write_json_artifact(
        validation_data,
        result_root / "validation.json",
        default=_json_default,
    )
    output_paths["observables.json"] = write_json_artifact(
        observables_data,
        result_root / "observables.json",
        default=_json_default,
    )

    manifest_files: dict[str, object] = {
        "model": "model.json",
        "config": "config.yaml",
        "conventions": "conventions.json",
        "environment": "environment.json",
        "validation": "validation.json",
        "observables": "observables.json",
    }
    manifest_files.update(dict(files or {}))

    array_summaries = []
    for array_file in array_files:
        relative_path = _relative_artifact_path(result_root, array_file)
        manifest_files.setdefault(Path(relative_path).stem, relative_path)
        summary_path = result_root / relative_path if not Path(relative_path).is_absolute() else Path(relative_path)
        array_summaries.append(summarize_npz_artifact(summary_path).to_dict())

    manifest_metadata: dict[str, object] = {
        "schema_version": 1,
        "workflow": str(workflow),
        "system_name": str(system_name),
    }
    if array_summaries:
        manifest_metadata["array_summaries"] = array_summaries
    manifest_metadata.update(dict(metadata or {}))

    manifest = ArtifactManifest(
        root=result_root,
        model=model_data,
        conventions=convention_data,
        files=manifest_files,
        metadata=manifest_metadata,
    )
    output_paths["manifest.json"] = manifest.save(result_root / "manifest.json")
    return output_paths


def update_artifact_manifest(
    root: str | Path,
    *,
    files: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    array_files: tuple[str | Path, ...] = (),
) -> Path:
    """Update only ``manifest.json`` with additional files/metadata.

    This is for derived postprocessing products that share an existing result
    root.  It preserves the existing contract sidecars and does not rewrite
    ``model.json``, ``config.yaml``, ``conventions.json``, ``environment.json``,
    ``validation.json``, or ``observables.json``.
    """

    result_root = Path(root)
    result_root.mkdir(parents=True, exist_ok=True)
    manifest_path = result_root / "manifest.json"
    existing = _read_json_if_present(manifest_path) or {}

    manifest_files = dict(existing.get("files", {})) if isinstance(existing.get("files", {}), Mapping) else {}
    manifest_files.update(dict(files or {}))

    manifest_metadata = dict(existing.get("metadata", {})) if isinstance(existing.get("metadata", {}), Mapping) else {}
    manifest_metadata.update(dict(metadata or {}))

    if array_files:
        array_summaries = list(manifest_metadata.get("array_summaries", []))
        for array_file in array_files:
            relative_path = _relative_artifact_path(result_root, array_file)
            manifest_files.setdefault(Path(relative_path).stem, relative_path)
            summary_path = result_root / relative_path if not Path(relative_path).is_absolute() else Path(relative_path)
            array_summaries.append(summarize_npz_artifact(summary_path).to_dict())
        manifest_metadata["array_summaries"] = array_summaries

    updated = {
        "root": str(existing.get("root", str(result_root))),
        "model": existing.get("model"),
        "conventions": existing.get("conventions", ConventionBundle().to_dict()),
        "files": manifest_files,
        "metadata": manifest_metadata,
    }
    return write_json_artifact(updated, manifest_path, default=_json_default)


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    import json

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_manifest_json_sidecar(root: Path, manifest: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    files = manifest.get("files", {})
    if not isinstance(files, Mapping) or key not in files:
        return None
    sidecar = Path(str(files[key]))
    if not sidecar.is_absolute():
        sidecar = root / sidecar
    return _read_json_if_present(sidecar)


def load_result(path: str | Path) -> ResultDirectory:
    root = Path(path)
    manifest = _read_json_if_present(root / "manifest.json") or {}
    return ResultDirectory(
        root=root,
        manifest=manifest,
        model=_read_json_if_present(root / "model.json"),
        config=_read_json_if_present(root / "config.yaml"),
        conventions=_read_json_if_present(root / "conventions.json"),
        validation=_read_json_if_present(root / "validation.json"),
        environment=_read_json_if_present(root / "environment.json"),
        observables=_read_json_if_present(root / "observables.json"),
        canonical_hf_run_result=_read_manifest_json_sidecar(root, manifest, "canonical_hf_run_result"),
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
    "update_artifact_manifest",
    "write_contract_artifacts",
]
