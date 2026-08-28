"""Fail-closed validation for tracked Xue 2018 Fig. 2 report artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Xue2018ArtifactValidation:
    dataset_id: str
    full_path_branch_id: str
    strong_anchor_branch_id: str
    stationary_branch_id: str
    checked_paths: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _resolve_relative(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"artifact path must be report-root relative: {relative!r}")
    resolved = root / candidate
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def validate_xue2018_report_artifacts(report_root: str | Path) -> Xue2018ArtifactValidation:
    """Validate one manifest-bound Xue report/JSON/PNG lineage.

    The validator intentionally does not infer a branch from figure pixels or
    report prose.  One manifest must bind every input and output by SHA-256,
    while the report and branch-data JSON must carry the same immutable
    dataset and branch identifiers.
    """

    root = Path(report_root)
    manifest_path = root / "data/xue2018_fig2_artifact_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "xue2018-fig2-artifact-manifest-v1":
        raise ValueError("unsupported Xue artifact manifest schema")
    if manifest.get("mixed_branch_curve_forbidden") is not True:
        raise ValueError("manifest must forbid mixed-branch curves")
    dataset_id = str(manifest.get("dataset_id", ""))
    if not dataset_id:
        raise ValueError("manifest dataset_id must be nonempty")

    checked: list[str] = []
    for section in ("data", "figures"):
        entries = manifest.get(section)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"manifest {section} must be a nonempty list")
        for entry in entries:
            relative = str(entry.get("path", ""))
            path = _resolve_relative(root, relative)
            expected = str(entry.get("sha256", ""))
            if len(expected) != 64 or _sha256(path) != expected:
                raise ValueError(f"SHA-256 mismatch for {relative}")
            if section == "figures" and entry.get("dataset_id") != dataset_id:
                raise ValueError(f"figure {relative} has the wrong dataset_id")
            checked.append(relative)

    branch_data_path = root / "data/xue2018_fig2_branch_data.json"
    branch_data = _load_json(branch_data_path)
    if branch_data.get("dataset_id") != dataset_id:
        raise ValueError("branch-data and manifest dataset_id differ")
    full_id = str(branch_data["full_path_dataset"]["branch_id"])
    strong_id = str(branch_data["strong_trs_anchor_dataset"]["branch_id"])
    stationary_id = str(branch_data["stationary_trs_branch_dataset"]["branch_id"])
    if manifest.get("full_path_branch_id") != full_id:
        raise ValueError("full-path branch_id differs between JSON and manifest")
    if manifest.get("strong_anchor_branch_id") != strong_id:
        raise ValueError("strong-anchor branch_id differs between JSON and manifest")
    if manifest.get("stationary_branch_id") != stationary_id:
        raise ValueError("stationary branch_id differs between JSON and manifest")
    if branch_data["full_path_dataset"].get("status") != "stale_historical_branch_artifact":
        raise ValueError("old full-path TRS curve must remain explicitly stale")

    report_relative = str(manifest.get("report", ""))
    report_path = _resolve_relative(root, report_relative)
    report = report_path.read_text()
    required_text = (
        dataset_id,
        full_id,
        strong_id,
        stationary_id,
        manifest["figures"][0]["path"],
    )
    missing = [item for item in required_text if item not in report]
    if missing:
        raise ValueError(f"report is not bound to manifest identifiers: {missing}")
    checked.append(report_relative)
    return Xue2018ArtifactValidation(
        dataset_id=dataset_id,
        full_path_branch_id=full_id,
        strong_anchor_branch_id=strong_id,
        stationary_branch_id=stationary_id,
        checked_paths=tuple(checked),
    )


__all__ = ["Xue2018ArtifactValidation", "validate_xue2018_report_artifacts"]
