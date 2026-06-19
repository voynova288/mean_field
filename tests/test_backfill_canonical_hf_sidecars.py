from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mean_field.devtools.backfill_canonical_hf_sidecars import (
    build_parser,
    execute_backfill_writes,
    plan_backfill_writes,
    scan_backfill_candidates,
)


def _write_manifest(root: Path, *, system_name: str, workflow: str, files: dict[str, object] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "model": {"system_name": system_name, "params": {}, "lattice": {}},
                "conventions": {},
                "files": {} if files is None else dict(files),
                "metadata": {"schema_version": 1, "workflow": workflow, "system_name": system_name},
            }
        ),
        encoding="utf-8",
    )

def _write_minimal_eligible_rlg_hbn_archive(tmp_path: Path) -> Path:
    panel_root = tmp_path / "panel_A"
    panel_root.mkdir(parents=True)
    cache_dir = tmp_path / "cache"
    basis_key = "basis_key"
    overlap_key = "overlap_key"
    (cache_dir / "basis" / basis_key).mkdir(parents=True)
    (cache_dir / "basis" / basis_key / "manifest.json").write_text("{}", encoding="utf-8")
    (cache_dir / "overlap" / overlap_key).mkdir(parents=True)
    (cache_dir / "overlap" / overlap_key / "manifest.json").write_text("{}", encoding="utf-8")
    archive_path = panel_root / "hf_ground_state.npz"
    matrix = np.zeros((1, 1, 1), dtype=np.complex128)
    np.savez(
        archive_path,
        density=matrix,
        hamiltonian=matrix,
        h0=matrix,
        energies_mev=np.zeros((1, 1), dtype=float),
        reference_density=matrix,
        cache_key_basis=np.asarray(basis_key),
        cache_key_overlap=np.asarray(overlap_key),
        cache_dir=np.asarray(str(cache_dir)),
        zero_literal_q0_fock=np.asarray(False),
    )
    return archive_path


def test_backfill_scanner_is_dry_run_by_default() -> None:
    args = build_parser().parse_args([])

    assert args.dry_run is True
    assert args.write is False
    assert args.target_root is None
    assert args.allow_target_root == []
    assert args.overwrite is False
    assert args.report_json is None
    assert args.report_md is None


def test_scanner_recognizes_already_canonical_sidecar_without_writing(tmp_path: Path) -> None:
    result_root = tmp_path / "canonical_result"
    _write_manifest(
        result_root,
        system_name="tdbg",
        workflow="tdbg.projected_hf",
        files={"canonical_hf_run_result": "canonical_hf_run_result.json"},
    )
    sidecar = {
        "schema_version": 1,
        "contract_type": "mean_field.core.contracts.HFRunResult",
        "final_state": {},
    }
    (result_root / "canonical_hf_run_result.json").write_text(json.dumps(sidecar), encoding="utf-8")

    records = scan_backfill_candidates([tmp_path], include_archives=False)

    assert len(records) == 1
    assert records[0].decision == "already_canonical"
    assert records[0].would_write is False
    assert not (result_root / "canonical_hf_arrays.npz").exists()


def test_scanner_marks_tdbg_historical_root_as_needing_archive_loader(tmp_path: Path) -> None:
    result_root = tmp_path / "tdbg_historical"
    _write_manifest(
        result_root,
        system_name="tdbg",
        workflow="tdbg.projected_hf",
        files={"hf_state": "hf_state.npz", "projected_hf_summary": "projected_hf_summary.json"},
    )

    records = scan_backfill_candidates([tmp_path], include_archives=False)

    assert len(records) == 1
    record = records[0]
    assert record.decision == "requires_archive_loader"
    assert record.can_backfill_now is False
    assert record.would_write is False
    assert any("TDBGProjectedHFData" in blocker for blocker in record.blockers)
    assert not (result_root / "canonical_hf_run_result.json").exists()


def test_scanner_identifies_rlg_hbn_archive_with_existing_loader_inputs(tmp_path: Path) -> None:
    archive_path = _write_minimal_eligible_rlg_hbn_archive(tmp_path)
    panel_root = archive_path.parent

    records = scan_backfill_candidates([archive_path])

    assert len(records) == 1
    record = records[0]
    assert record.decision == "eligible_with_existing_archive_loader"
    assert record.can_backfill_now is True
    assert record.would_write is False
    assert "load_rlg_hbn_tdhf_run_from_archive" in " ".join(record.adapters)
    assert not (panel_root / "canonical_hf_run_result.json").exists()

def test_write_plan_rejects_target_outside_allowlist(tmp_path: Path) -> None:
    archive_path = _write_minimal_eligible_rlg_hbn_archive(tmp_path)
    records = scan_backfill_candidates([archive_path])

    with pytest.raises(ValueError, match="outside the explicit allowlist"):
        plan_backfill_writes(
            records,
            roots=[archive_path],
            target_root=tmp_path / "stage",
            allow_target_roots=[tmp_path / "other_allowed_root"],
        )

def test_write_mode_stages_synthetic_sidecar_without_mutating_source(tmp_path: Path) -> None:
    archive_path = _write_minimal_eligible_rlg_hbn_archive(tmp_path)
    panel_root = archive_path.parent
    records = scan_backfill_candidates([archive_path])
    calls: dict[str, object] = {}

    def fake_loader(path: str | Path) -> object:
        calls["loader_path"] = str(path)
        return {"fake_run": True}

    def fake_adapter(run: object, *, archive_manifest: dict[str, object]) -> object:
        calls["archive_manifest"] = dict(archive_manifest)
        return {"fake_canonical_run_result": run}

    def fake_sidecar(_canonical_run_result: object) -> dict[str, object]:
        return {
            "schema_version": 1,
            "contract_type": "mean_field.core.contracts.HFRunResult",
            "final_state": {"contract_type": "mean_field.core.contracts.HFState"},
        }

    target_root = tmp_path / "staged_backfill"
    payload = execute_backfill_writes(
        records,
        roots=[archive_path],
        target_root=target_root,
        allow_target_roots=[tmp_path],
        rlg_hbn_loader=fake_loader,
        rlg_hbn_adapter=fake_adapter,
        canonical_sidecar_builder=fake_sidecar,
    )

    assert payload["dry_run"] is False
    assert payload["historical_results_mutated"] is False
    assert payload["summary"]["written_count"] == 1
    assert calls["loader_path"] == str(archive_path)
    assert calls["archive_manifest"]["historical_results_mutated"] is False
    assert not (panel_root / "canonical_hf_run_result.json").exists()

    entry = payload["entries"][0]
    planned_files = entry["planned_files"]
    sidecar_path = Path(planned_files["canonical_hf_run_result"])
    audit_path = Path(planned_files["audit"])
    patch_path = Path(planned_files["manifest_patch"])
    assert sidecar_path.is_file()
    assert audit_path.is_file()
    assert patch_path.is_file()
    assert (target_root / "backfill_write_manifest.json").is_file()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["contract_type"] == "mean_field.core.contracts.HFRunResult"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["historical_results_mutated"] is False
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    assert patch["mutate_historical_root"] is False

def test_write_mode_refuses_existing_staged_sidecar_without_overwrite(tmp_path: Path) -> None:
    archive_path = _write_minimal_eligible_rlg_hbn_archive(tmp_path)
    records = scan_backfill_candidates([archive_path])
    target_root = tmp_path / "staged_backfill"
    plan = plan_backfill_writes(
        records,
        roots=[archive_path],
        target_root=target_root,
        allow_target_roots=[tmp_path],
    )
    sidecar_path = Path(plan["entries"][0]["planned_files"]["canonical_hf_run_result"])
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text("{}", encoding="utf-8")

    def fail_loader(_path: str | Path) -> object:  # pragma: no cover - should not be called.
        raise AssertionError("existing output should be skipped before loader runs")

    payload = execute_backfill_writes(
        records,
        roots=[archive_path],
        target_root=target_root,
        allow_target_roots=[tmp_path],
        rlg_hbn_loader=fail_loader,
    )

    assert payload["summary"]["written_count"] == 0
    assert payload["entries"][0]["status"] == "skipped_existing_output"
    assert "--overwrite" in payload["entries"][0]["skipped_reason"]
