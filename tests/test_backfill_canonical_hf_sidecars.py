from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mean_field.devtools.backfill_canonical_hf_sidecars import (
    build_parser,
    execute_backfill_writes,
    inventory_payload,
    main,
    plan_backfill_writes,
    render_markdown_inventory,
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
    compatible_cache_manifest = json.dumps(
        {
            "extra": {
                "basis_periodic_gauge": "centered_cell_reciprocal_relabel_pad1_v2",
                "form_factor_convention": "physical_q_plus_g_valley_signed_raw_shift_v2",
            }
        }
    )
    (cache_dir / "basis" / basis_key).mkdir(parents=True)
    (cache_dir / "basis" / basis_key / "manifest.json").write_text(compatible_cache_manifest, encoding="utf-8")
    (cache_dir / "overlap" / overlap_key).mkdir(parents=True)
    (cache_dir / "overlap" / overlap_key / "manifest.json").write_text(compatible_cache_manifest, encoding="utf-8")
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
    assert record.metadata["archive_format_contract"]["raw_object"].endswith("TDBGProjectedHFResult")
    assert not (result_root / "canonical_hf_run_result.json").exists()


def test_scanner_reports_tdbg_archive_missing_exact_raw_fields(tmp_path: Path) -> None:
    result_root = tmp_path / "tdbg_archive_case"
    result_root.mkdir()
    matrix = np.zeros((2, 2, 1), dtype=np.complex128)
    np.savez(
        result_root / "hf_state.npz",
        density=matrix,
        hamiltonian=matrix,
        h0=matrix,
        energies=np.zeros((2, 1), dtype=float),
        k_grid_frac=np.zeros((1, 1, 2), dtype=float),
        kvec_nm_inv=np.zeros((1, 2), dtype=float),
        band_indices=np.asarray([0, 1], dtype=int),
        reference_density=matrix,
    )
    (result_root / "state_labels.json").write_text("[]", encoding="utf-8")
    (result_root / "projected_hf_summary.json").write_text("{}", encoding="utf-8")

    records = scan_backfill_candidates([result_root])

    assert len(records) == 1
    record = records[0]
    assert record.kind == "tdbg_archive"
    assert record.decision == "requires_archive_loader"
    assert any("missing key `mu`" in blocker for blocker in record.blockers)
    assert any("projected-basis micro_wavefunctions" in blocker for blocker in record.blockers)
    assert record.metadata["archive_format_contract"]["state_npz_required_keys"]
    assert not (result_root / "canonical_hf_run_result.json").exists()


def test_scanner_reports_htg_primitive_archive_missing_projected_basis(tmp_path: Path) -> None:
    result_root = tmp_path / "HTG_primitive_archive_case"
    result_root.mkdir()
    matrix = np.zeros((4, 4, 1), dtype=np.complex128)
    np.savez(
        result_root / "hf_ground_state.npz",
        density=matrix,
        hamiltonian=matrix,
        h0=matrix,
        energies_ev=np.zeros((4, 1), dtype=float),
        kvec_nm_inv=np.zeros((1, 2), dtype=float),
        k_grid_frac=np.zeros((1, 1, 2), dtype=float),
        iter_energy_ev=np.zeros(1, dtype=float),
        iter_err=np.zeros(1, dtype=float),
        iter_oda=np.zeros(1, dtype=float),
    )
    (result_root / "hf_params.json").write_text("{}", encoding="utf-8")

    records = scan_backfill_candidates([result_root])

    assert len(records) == 1
    record = records[0]
    assert record.kind == "htg_primitive_archive"
    assert record.system_name == "htg"
    assert record.decision == "requires_archive_loader"
    assert any("missing key `mu`" in blocker for blocker in record.blockers)
    assert any("HTGProjectedBasisData.basis.wavefunctions" in blocker for blocker in record.blockers)
    assert "htg_hf_run_to_hf_run_result" in " ".join(record.adapters)


def test_scanner_reports_htg_supercell_archive_missing_projected_basis(tmp_path: Path) -> None:
    result_root = tmp_path / "HTG_supercell_archive_case"
    result_root.mkdir()
    matrix = np.zeros((4, 4, 1), dtype=np.complex128)
    np.savez(
        result_root / "hf_supercell_ground_state.npz",
        density=matrix,
        hamiltonian=matrix,
        h0=matrix,
        energies=np.zeros((4, 1), dtype=float),
        kvec=np.zeros(1, dtype=np.complex128),
        k_grid_frac=np.zeros((1, 1, 2), dtype=float),
        iter_energy=np.zeros(1, dtype=float),
        iter_err=np.zeros(1, dtype=float),
        iter_oda=np.zeros(1, dtype=float),
        reference_diagonal=np.zeros(1, dtype=float),
        fold_representatives=np.zeros((1, 2), dtype=int),
        supercell_matrix=np.eye(2, dtype=int),
        primitive_nu=np.asarray(3.5),
        init_mode=np.asarray("bm"),
        seed=np.asarray(1),
        converged=np.asarray(True),
        exit_reason=np.asarray("converged"),
    )
    (result_root / "summary.json").write_text("{}", encoding="utf-8")

    records = scan_backfill_candidates([result_root])

    assert len(records) == 1
    record = records[0]
    assert record.kind == "htg_supercell_archive"
    assert record.system_name == "htg_supercell"
    assert record.decision == "requires_archive_loader"
    assert any("missing key `mu`" in blocker for blocker in record.blockers)
    assert any("HTGSupercellProjectedBasisData.basis.wavefunctions" in blocker for blocker in record.blockers)
    assert "htg_supercell_hf_run_to_hf_run_result" in " ".join(record.adapters)


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

def test_scanner_rejects_rlg_hbn_archive_with_stale_cache_manifest(tmp_path: Path) -> None:
    archive_path = _write_minimal_eligible_rlg_hbn_archive(tmp_path)
    with np.load(archive_path, allow_pickle=False) as data:
        cache_dir = Path(str(np.asarray(data["cache_dir"]).reshape(-1)[0]))
        basis_key = str(np.asarray(data["cache_key_basis"]).reshape(-1)[0])
    (cache_dir / "basis" / basis_key / "manifest.json").write_text(
        json.dumps({"extra": {"basis_periodic_gauge": "centered_cell_reciprocal_relabel_pad1_v2"}}),
        encoding="utf-8",
    )

    records = scan_backfill_candidates([archive_path])

    assert len(records) == 1
    record = records[0]
    assert record.decision == "missing_loader_inputs"
    assert record.can_backfill_now is False
    assert any("form_factor_convention" in blocker for blocker in record.blockers)

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


def _write_minimal_eligible_htg_primitive_archive(tmp_path: Path) -> Path:
    root = tmp_path / "htg_full_archive"
    root.mkdir()
    nt = 8
    nk = 1
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    density = np.zeros_like(h0)
    sigma_z = np.zeros_like(h0)
    model_params = {
        "theta_deg": 1.0,
        "n_shells": 0,
        "params": {
            "graphene_lattice_constant_nm": 0.246,
            "fermi_velocity_m_per_s": 1.03e6,
            "w_ev": 0.105,
            "kappa": 0.7,
            "zeta_rad": None,
            "model_name": "default",
        },
    }
    interaction_params = {
        "epsilon_r": 8.0,
        "d_sc_nm": 25.0,
        "U_ev": 0.0,
        "subtraction": "average",
        "n_k": 1,
        "g_shells": 0,
        "finite_zero_limit": False,
        "zero_cutoff_nm_inv": 1.0e-12,
    }
    np.savez(
        root / "hf_ground_state.npz",
        density=density,
        hamiltonian=h0,
        h0=h0,
        energies_ev=np.zeros((nt, nk), dtype=float),
        kvec_nm_inv=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        k_grid_frac=np.zeros((1, 2), dtype=float),
        iter_energy_ev=np.asarray([0.0], dtype=float),
        iter_err=np.asarray([0.0], dtype=float),
        iter_oda=np.asarray([1.0], dtype=float),
        mu=np.asarray(0.0),
        nu=np.asarray(0.0),
        precision=np.asarray(1.0e-8),
        v0=np.asarray(1.0),
        sigma_z=sigma_z,
        converged=np.asarray(True),
        exit_reason=np.asarray("converged"),
        init_mode=np.asarray("archive"),
        seed=np.asarray(7),
    )
    np.savez(
        root / "hf_projected_basis.npz",
        wavefunctions=np.zeros((6, 2, 2, 1), dtype=np.complex128),
        projected_band_indices=np.asarray([0, 1], dtype=int),
        central_band_indices=np.asarray([0, 1], dtype=int),
        band_sigma_z=np.zeros((2, 1), dtype=float),
        reciprocal_grid_shape=np.asarray([1, 1], dtype=int),
        reciprocal_grid_origin=np.asarray([0, 0], dtype=int),
        moire_cell_area_nm2=np.asarray(1.0),
        model_params=np.asarray(json.dumps(model_params)),
        interaction_params=np.asarray(json.dumps(interaction_params)),
    )
    return root / "hf_ground_state.npz"


def test_write_mode_stages_htg_primitive_full_archive_with_existing_loader(tmp_path: Path) -> None:
    archive_path = _write_minimal_eligible_htg_primitive_archive(tmp_path)
    records = scan_backfill_candidates([archive_path])
    assert len(records) == 1
    record = records[0]
    assert record.kind == "htg_primitive_archive"
    assert record.decision == "eligible_with_existing_archive_loader"
    assert record.can_backfill_now is True
    assert "load_htg_hf_run_from_archive" in " ".join(record.adapters)

    target_root = tmp_path / "staged_backfill"
    payload = execute_backfill_writes(
        records,
        roots=[archive_path],
        target_root=target_root,
        allow_target_roots=[tmp_path],
    )
    assert payload["historical_results_mutated"] is False
    assert payload["summary"]["written_count"] == 1
    sidecar_path = Path(payload["entries"][0]["planned_files"]["canonical_hf_run_result"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["contract_type"] == "mean_field.core.contracts.HFRunResult"
    assert sidecar["final_state"]["density"]["metadata"]["adapter"] == "mean_field.systems.htg.mean_field_adapter"


def _write_minimal_eligible_tdbg_archive(tmp_path: Path) -> Path:
    root = tmp_path / "tdbg_full_archive"
    root.mkdir()
    nt = 2
    nk = 1
    matrix = np.zeros((nt, nt, nk), dtype=np.complex128)
    np.savez(
        root / "hf_state.npz",
        density=matrix,
        hamiltonian=matrix,
        h0=matrix,
        energies=np.zeros((nt, nk), dtype=float),
        k_grid_frac=np.zeros((nk, 2), dtype=float),
        kvec_nm_inv=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        band_indices=np.asarray([0, 1], dtype=int),
        reference_density=matrix,
        mu=np.asarray(0.0),
        iter_energy=np.asarray([0.0], dtype=float),
        iter_err=np.asarray([0.0], dtype=float),
        iter_oda=np.asarray([1.0], dtype=float),
        n_occupied_per_k=np.asarray(1),
        lower_band_count=np.asarray(0),
    )
    np.savez(
        root / "projected_basis.npz",
        wavefunctions=np.zeros((nt, nk, 1, 4), dtype=np.complex128),
        moire_area_nm2=np.asarray(1.0),
        shifts=np.zeros((0, 2), dtype=int),
        shift_gvecs=np.zeros((0,), dtype=np.complex128),
        shift_srcmaps=np.zeros((0, 1), dtype=int),
        valley_params=np.asarray(json.dumps({})),
    )
    (root / "state_labels.json").write_text(
        json.dumps(
            [
                {"index": 0, "spin": "up", "valley": 1, "band_position": 0, "band_index": 0},
                {"index": 1, "spin": "up", "valley": 1, "band_position": 1, "band_index": 1},
            ]
        ),
        encoding="utf-8",
    )
    (root / "projected_hf_summary.json").write_text(
        json.dumps(
            {
                "init_mode": "archive",
                "seed": 2,
                "converged": True,
                "exit_reason": "converged",
                "order_parameters": {},
                "energy_components_ev": {},
            }
        ),
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "theta_deg": 1.38,
                "cut": 1.0,
                "mesh_size": 1,
                "paper_ud_ev": 0.09,
                "stacking": "AB-BA",
                "window": {"name": "two_flat", "band_indices": None},
                "filling": 1,
                "interaction": {},
                "precision": 1.0e-7,
                "max_iter": 1,
            }
        ),
        encoding="utf-8",
    )
    (root / "model.json").write_text(json.dumps({"theta_deg": 1.38, "cut": 1.0}), encoding="utf-8")
    return root / "hf_state.npz"


def test_scanner_identifies_tdbg_full_archive_with_existing_loader(tmp_path: Path) -> None:
    archive_path = _write_minimal_eligible_tdbg_archive(tmp_path)
    records = scan_backfill_candidates([archive_path])
    assert len(records) == 1
    record = records[0]
    assert record.kind == "tdbg_archive"
    assert record.decision == "eligible_with_existing_archive_loader"
    assert record.can_backfill_now is True
    assert "load_tdbg_projected_hf_result_from_archive" in " ".join(record.adapters)


def test_backfill_inventory_report_renders_empty_dry_run(tmp_path: Path) -> None:
    records = scan_backfill_candidates([tmp_path / "empty_results"])
    payload = inventory_payload(records, roots=[tmp_path / "empty_results"])
    markdown = render_markdown_inventory(payload)
    assert "Historical Canonical HF Sidecar Backfill Dry Run" in markdown
    assert "candidate_count" in markdown
    assert "TDBGProjectedHFResult" in markdown
    assert payload["historical_results_mutated"] is False


def test_backfill_cli_accepts_positional_root_with_no_archives(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    status = main(
        [
            str(tmp_path / "empty_results"),
            "--no-archives",
            "--report-json",
            str(tmp_path / "inventory.json"),
            "--report-md",
            str(tmp_path / "inventory.md"),
        ]
    )
    captured = capsys.readouterr()
    assert status == 0
    assert "candidate_count: `0`" in captured.out
    assert (tmp_path / "inventory.json").is_file()
    assert (tmp_path / "inventory.md").is_file()


def test_backfill_cli_write_path_returns_zero_for_staged_full_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive_path = _write_minimal_eligible_htg_primitive_archive(tmp_path)
    target_root = tmp_path / "cli_staged"
    status = main(
        [
            str(archive_path),
            "--write",
            "--target-root",
            str(target_root),
            "--allow-target-root",
            str(tmp_path),
            "--report-json",
            str(target_root / "inventory.json"),
            "--report-md",
            str(target_root / "inventory.md"),
        ]
    )
    captured = capsys.readouterr()
    assert status == 0
    assert "written_count: `1`" in captured.out
    with (target_root / "inventory.json").open(encoding="utf-8") as fh:
        payload = json.load(fh)
    assert Path(payload["write_plan"]["write_manifest_path"]).is_file()
    assert (target_root / "inventory.json").is_file()
