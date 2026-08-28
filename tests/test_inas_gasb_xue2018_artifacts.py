from __future__ import annotations

import json
from pathlib import Path

import pytest

from mean_field.systems.inas_gasb.xue2018_artifacts import (
    validate_xue2018_report_artifacts,
)


REPORT_ROOT = Path(__file__).resolve().parents[1] / "reports"


def test_xue2018_report_json_png_share_one_manifest_lineage() -> None:
    result = validate_xue2018_report_artifacts(REPORT_ROOT)
    assert result.dataset_id == "xue2018-fig2-branch-lineage-audit-v1"
    assert result.full_path_branch_id == "xue2018-trs-weak-seed-normal-attractor-fullpath-v1"
    assert result.strong_anchor_branch_id == "xue2018-trs-strong-attractor-p24-p26-v1"
    assert "figures/xue2018_fig2_branch_lineage_audit.png" in result.checked_paths


def test_xue2018_artifact_validator_rejects_mixed_branch_id(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    for source in REPORT_ROOT.rglob("*"):
        if source.is_file():
            target = root / source.relative_to(REPORT_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    manifest_path = root / "data/xue2018_fig2_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["strong_anchor_branch_id"] = "wrong-branch"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="strong-anchor branch_id"):
        validate_xue2018_report_artifacts(root)
