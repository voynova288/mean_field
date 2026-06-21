from __future__ import annotations

from ._shared import *  # noqa: F401,F403
from ._scan import _mapping

def _safe_slug(text: str, *, fallback: str = "candidate") -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:80] if slug else fallback

def _stage_name(record: BackfillCandidate, index: int) -> str:
    identity = record.archive_path or record.manifest_path or record.root
    digest = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()[:12]
    root_name = _safe_slug(Path(record.root).name, fallback="root")
    system = _safe_slug(record.system_name, fallback="system")
    decision = _safe_slug(record.decision, fallback="decision")
    return f"{index:05d}_{system}_{decision}_{root_name}_{digest}"

def _resolve_allowlisted_target_root(target_root: str | Path, allow_target_roots: Sequence[str | Path]) -> Path:
    if not allow_target_roots:
        raise ValueError("write-mode target_root requires at least one explicit allow_target_root")
    resolved_target = Path(target_root).expanduser().resolve()
    for raw_allowed in allow_target_roots:
        allowed = Path(raw_allowed).expanduser().resolve()
        try:
            resolved_target.relative_to(allowed)
        except ValueError:
            continue
        return resolved_target
    allowed_text = ", ".join(str(Path(item).expanduser().resolve()) for item in allow_target_roots)
    raise ValueError(f"target_root is outside the explicit allowlist: target={resolved_target}; allowlist=[{allowed_text}]")

def _require_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} would escape target_root: {resolved} is not under {root}") from exc
    return resolved

def _planned_stage_files(stage_dir: Path) -> dict[str, str]:
    return {
        _CANONICAL_HF_SIDECAR_KEY: str(stage_dir / _CANONICAL_HF_SIDECAR_FILE),
        "manifest_patch": str(stage_dir / _MANIFEST_PATCH_FILE),
        "audit": str(stage_dir / _BACKFILL_AUDIT_FILE),
    }

def _manifest_patch_for_record(record: BackfillCandidate) -> dict[str, object]:
    return {
        "schema_version": 1,
        "patch_type": "canonical_hf_sidecar_manifest_entry",
        "historical_root": record.target_root,
        "historical_manifest_path": record.manifest_path,
        "historical_archive_path": record.archive_path,
        "mutate_historical_root": False,
        "files_patch": {_CANONICAL_HF_SIDECAR_KEY: _CANONICAL_HF_SIDECAR_FILE},
        "metadata_patch": {
            "canonical_hf_run_result": {
                "schema_version": 1,
                "contract_type": "mean_field.core.contracts.HFRunResult",
                "source": "staged_backfill_not_applied_to_historical_results",
            }
        },
        "blocker_before_historical_apply": (
            "This task only stages sidecars and audit material. Applying this patch to a historical result root "
            "requires a separate approval/policy step and overwrite audit."
        ),
    }

def _candidate_is_write_eligible(record: BackfillCandidate) -> bool:
    return bool(
        record.can_backfill_now
        and record.kind == "rlg_hbn_archive"
        and record.decision == "eligible_with_existing_archive_loader"
        and record.archive_path
    )

def plan_backfill_writes(
    records: Sequence[BackfillCandidate],
    *,
    roots: Sequence[str | Path],
    target_root: str | Path,
    allow_target_roots: Sequence[str | Path],
    overwrite: bool = False,
) -> dict[str, object]:
    """Return an audited staging/write plan without mutating any files.

    The plan intentionally maps each eligible historical candidate to a fresh
    subdirectory below ``target_root``.  Original result roots are only recorded
    as evidence; they are never used as output destinations.
    """

    resolved_target = _resolve_allowlisted_target_root(target_root, allow_target_roots)
    entries: list[dict[str, object]] = []
    for index, record in enumerate(records):
        stage_dir = _require_within(resolved_target / _stage_name(record, index), resolved_target, label="stage_dir")
        planned_files = _planned_stage_files(stage_dir)
        for key, path_text in planned_files.items():
            _require_within(Path(path_text), resolved_target, label=f"planned_file[{key}]")

        status = "planned" if _candidate_is_write_eligible(record) else "skipped"
        skipped_reason = None if status == "planned" else record.reason
        if status == "planned" and not bool(overwrite):
            existing = [path for path in planned_files.values() if Path(path).exists()]
            if existing:
                status = "skipped_existing_output"
                skipped_reason = "planned staging output already exists; pass --overwrite to replace it"
        entry = {
            "index": int(index),
            "status": status,
            "skipped_reason": skipped_reason,
            "source": record.to_dict(),
            "target_dir": str(stage_dir),
            "planned_files": planned_files,
            "written_files": [],
            "historical_results_mutated": False,
            "manifest_patch": _manifest_patch_for_record(record),
            "evidence": list(record.evidence),
            "blockers": list(record.blockers),
            "uncertainty": list(record.uncertainty),
        }
        entries.append(entry)

    status_counts = Counter(str(entry["status"]) for entry in entries)
    manifest_path = _require_within(resolved_target / _WRITE_MANIFEST_FILE, resolved_target, label="write_manifest")
    return {
        "schema_version": 1,
        "mode": "plan",
        "dry_run": True,
        "historical_results_mutated": False,
        "target_root": str(resolved_target),
        "allow_target_roots": [str(Path(root).expanduser().resolve()) for root in allow_target_roots],
        "overwrite": bool(overwrite),
        "roots": [str(Path(root).expanduser()) for root in roots],
        "write_manifest_path": str(manifest_path),
        "summary": {
            "entry_count": len(entries),
            "status_counts": dict(sorted(status_counts.items())),
            "planned_count": int(status_counts.get("planned", 0)),
            "skipped_count": sum(int(count) for status, count in status_counts.items() if status.startswith("skipped")),
            "written_count": 0,
            "write_error_count": 0,
        },
        "entries": entries,
    }

def _default_rlg_hbn_loader(archive_path: str | Path) -> object:
    module = import_module("mean_field.systems.RnG_hBN.tdhf")
    return module.load_rlg_hbn_tdhf_run_from_archive(archive_path)

def _default_rlg_hbn_adapter(run: object, *, archive_manifest: dict[str, object]) -> object:
    module = import_module("mean_field.systems.RnG_hBN.hf_contracts")
    return module.rlg_hbn_hf_run_to_hf_run_result(run, archive_manifest=archive_manifest)  # type: ignore[arg-type]

def _default_canonical_sidecar_builder(canonical_run_result: object) -> dict[str, object]:
    from mean_field.api.hf import _canonical_hf_run_result_sidecar

    return _canonical_hf_run_result_sidecar(canonical_run_result)

def _write_json_no_overwrite(payload: object, path: str | Path, *, overwrite: bool) -> Path:
    target = Path(path)
    if target.exists() and not bool(overwrite):
        raise FileExistsError(f"refusing to overwrite existing backfill output: {target}")
    return write_json_artifact(payload, target)

def execute_backfill_writes(
    records: Sequence[BackfillCandidate],
    *,
    roots: Sequence[str | Path],
    target_root: str | Path,
    allow_target_roots: Sequence[str | Path],
    overwrite: bool = False,
    rlg_hbn_loader: Callable[[str | Path], object] | None = None,
    rlg_hbn_adapter: Callable[..., object] | None = None,
    canonical_sidecar_builder: Callable[[object], dict[str, object]] | None = None,
) -> dict[str, object]:
    """Materialize eligible RLG/hBN canonical sidecars into a staging root.

    This is an explicit write path, but it is *not* a historical mutation path:
    outputs are written only below the allowlisted ``target_root``.  Ineligible
    candidates are recorded with skipped reasons.  The real historical result or
    archive directories are left untouched.
    """

    payload = plan_backfill_writes(
        records,
        roots=roots,
        target_root=target_root,
        allow_target_roots=allow_target_roots,
        overwrite=overwrite,
    )
    manifest_path = Path(str(payload["write_manifest_path"]))
    if manifest_path.exists() and not bool(overwrite):
        raise FileExistsError(f"refusing to overwrite existing write manifest: {manifest_path}")

    loader = _default_rlg_hbn_loader if rlg_hbn_loader is None else rlg_hbn_loader
    adapter = _default_rlg_hbn_adapter if rlg_hbn_adapter is None else rlg_hbn_adapter
    sidecar_builder = _default_canonical_sidecar_builder if canonical_sidecar_builder is None else canonical_sidecar_builder

    entries = list(payload["entries"])
    for entry in entries:
        if entry.get("status") != "planned":
            continue
        source = _mapping(entry.get("source"))
        archive_path = source.get("archive_path")
        if not isinstance(archive_path, str) or not archive_path:
            entry["status"] = "write_error"
            entry["error"] = "planned entry is missing archive_path"
            continue
        archive_manifest = {
            "source_archive": archive_path,
            "source_root": source.get("root"),
            "backfill_mode": "staged_explicit_write",
            "historical_results_mutated": False,
        }
        planned_files = _mapping(entry.get("planned_files"))
        resolved_target = Path(str(payload["target_root"]))
        for key, path_text in planned_files.items():
            _require_within(Path(str(path_text)), resolved_target, label=f"execute_planned_file[{key}]")
        try:
            run = loader(archive_path)
            canonical_run_result = adapter(run, archive_manifest=archive_manifest)
            sidecar = sidecar_builder(canonical_run_result)
            if sidecar.get("schema_version") != 1 or sidecar.get("contract_type") != "mean_field.core.contracts.HFRunResult":
                raise ValueError("canonical sidecar builder returned an invalid HFRunResult sidecar header")
            if not isinstance(sidecar.get("final_state"), Mapping):
                raise ValueError("canonical sidecar builder returned an invalid final_state payload")

            sidecar_path = str(planned_files[_CANONICAL_HF_SIDECAR_KEY])
            manifest_patch_path = str(planned_files["manifest_patch"])
            audit_path = str(planned_files["audit"])
            _write_json_no_overwrite(sidecar, sidecar_path, overwrite=overwrite)
            _write_json_no_overwrite(entry["manifest_patch"], manifest_patch_path, overwrite=overwrite)
            audit_payload = {
                "schema_version": 1,
                "status": "written",
                "historical_results_mutated": False,
                "source": source,
                "archive_manifest": archive_manifest,
                "planned_files": planned_files,
                "written_files": [sidecar_path, manifest_patch_path, audit_path],
                "manifest_patch": entry["manifest_patch"],
                "evidence": entry.get("evidence", []),
                "uncertainty": entry.get("uncertainty", []),
            }
            _write_json_no_overwrite(audit_payload, audit_path, overwrite=overwrite)
            entry["status"] = "written"
            entry["written_files"] = [sidecar_path, manifest_patch_path, audit_path]
            entry["audit_manifest"] = audit_payload
        except Exception as exc:  # pragma: no cover - exact archive-loader failures are environment-specific.
            entry["status"] = "write_error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["written_files"] = list(entry.get("written_files", []))

    status_counts = Counter(str(entry["status"]) for entry in entries)
    payload["mode"] = "write"
    payload["dry_run"] = False
    payload["summary"] = {
        "entry_count": len(entries),
        "status_counts": dict(sorted(status_counts.items())),
        "planned_count": 0,
        "skipped_count": sum(int(count) for status, count in status_counts.items() if status.startswith("skipped")),
        "written_count": int(status_counts.get("written", 0)),
        "write_error_count": int(status_counts.get("write_error", 0)),
    }
    payload["entries"] = entries
    _write_json_no_overwrite(payload, manifest_path, overwrite=overwrite)
    return payload
