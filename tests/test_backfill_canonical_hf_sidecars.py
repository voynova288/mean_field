from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mean_field.devtools.backfill_canonical_hf_sidecars import build_parser, scan_backfill_candidates


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


def test_backfill_scanner_is_dry_run_by_default() -> None:
    args = build_parser().parse_args([])

    assert args.dry_run is True
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

    records = scan_backfill_candidates([archive_path])

    assert len(records) == 1
    record = records[0]
    assert record.decision == "eligible_with_existing_archive_loader"
    assert record.can_backfill_now is True
    assert record.would_write is False
    assert "load_rlg_hbn_tdhf_run_from_archive" in " ".join(record.adapters)
    assert not (panel_root / "canonical_hf_run_result.json").exists()
