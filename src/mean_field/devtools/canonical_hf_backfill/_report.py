from __future__ import annotations

from ._shared import *  # noqa: F401,F403

def backfill_strategy() -> dict[str, object]:
    return {
        "default_mode": "dry-run by default; explicit --write is staging-only and never mutates scanned historical roots",
        "safe_write_policy": [
            "Never fabricate canonical HF physics from summary-only artifacts.",
            "Write-mode requires --write, --target-root, and at least one --allow-target-root allowlist entry.",
            "Only eligible RLG/hBN archives are materialized through the existing archive loader and canonical adapter.",
            "TDBG/HTG remain blocked until archive loaders restore their raw run objects without recomputing physics.",
            "TDBG/HTG diagnostics list the precise missing raw files/fields so future runs can save a loader-compatible archive contract.",
            "Staged writes produce sidecars, manifest patches, and audit manifests under the caller-specified target root only.",
        ],
        "systems": [
            {
                "system": "tdbg",
                "current_status": "eligible_when_full_raw_state_basis_and_metadata_archives_are_present",
                "existing_loader": _TDBG_ARCHIVE_LOADER,
                "existing_adapter": _TDBG_ADAPTER,
                "safe_now": True,
                "blocker": "Summary-only TDBG archives remain blocked; full raw state, projected-basis micro-wavefunctions, labels, run history, and exact config/model fields are required.",
                "archive_format_contract": _contract_metadata(
                    raw_object="mean_field.systems.tdbg.projected_hf_state.TDBGProjectedHFResult",
                    state_keys=_TDBG_HF_STATE_CONTRACT_KEYS,
                    basis_keys=_TDBG_PROJECTED_BASIS_CONTRACT_KEYS,
                ),
            },
            {
                "system": "htg",
                "current_status": "eligible_when_full_raw_state_basis_and_metadata_archives_are_present",
                "existing_loader": _HTG_PRIMITIVE_ARCHIVE_LOADER,
                "existing_adapter": _HTG_PRIMITIVE_ADAPTER,
                "safe_now": True,
                "blocker": "Summary-only primitive HTG archives remain blocked; full raw state, projected-basis wavefunctions, and exact model/interaction metadata are required.",
                "archive_format_contract": _contract_metadata(
                    raw_object="mean_field.systems.htg.mean_field_adapter.HTGHartreeFockRun",
                    state_keys=_HTG_PRIMITIVE_STATE_CONTRACT_KEYS,
                    basis_keys=_HTG_PRIMITIVE_BASIS_CONTRACT_KEYS,
                ),
            },
            {
                "system": "htg_supercell",
                "current_status": "eligible_when_full_raw_state_basis_and_metadata_archives_are_present",
                "existing_loader": _HTG_ARCHIVE_LOADER,
                "existing_adapter": _HTG_ADAPTER,
                "safe_now": True,
                "blocker": "Summary-only HTG supercell archives remain blocked; full raw state, projected-basis wavefunctions, and exact model/interaction metadata are required.",
                "archive_format_contract": _contract_metadata(
                    raw_object="mean_field.systems.htg.supercell.HTGSupercellHartreeFockRun",
                    state_keys=_HTG_SUPERCELL_STATE_CONTRACT_KEYS,
                    basis_keys=_HTG_SUPERCELL_BASIS_CONTRACT_KEYS,
                ),
            },
            {
                "system": "rlg_hbn / RnG_hBN",
                "current_status": "eligible_when_hf_ground_state_archive_and_cache_entries_are_present",
                "existing_loader": _RLG_HBN_ARCHIVE_LOADER,
                "existing_adapter": _RLG_HBN_ADAPTER,
                "safe_now": True,
                "blocker": "Historical mutation remains blocked; current write support stages eligible sidecars under an explicit allowlisted target root only.",
            },
        ],
    }


def inventory_payload(
    records: Sequence[BackfillCandidate],
    *,
    roots: Sequence[str | Path],
    dry_run: bool = True,
    write_plan: Mapping[str, object] | None = None,
) -> dict[str, object]:
    decision_counts = Counter(record.decision for record in records)
    system_counts = Counter(record.system_name for record in records)
    payload: dict[str, object] = {
        "schema_version": 1,
        "dry_run": bool(dry_run),
        "would_write_anything": False if write_plan is None else not bool(write_plan.get("dry_run", True)),
        "historical_results_mutated": False,
        "roots": [str(Path(root).expanduser()) for root in roots],
        "summary": {
            "candidate_count": len(records),
            "decision_counts": dict(sorted(decision_counts.items())),
            "system_counts": dict(sorted(system_counts.items())),
            "can_backfill_now_count": sum(1 for record in records if record.can_backfill_now),
        },
        "strategy": backfill_strategy(),
        "records": [record.to_dict() for record in records],
    }
    if write_plan is not None:
        payload["write_plan"] = dict(write_plan)
    return payload


def render_markdown_inventory(payload: Mapping[str, object]) -> str:
    summary = _mapping(payload.get("summary"))
    strategy = _mapping(payload.get("strategy"))
    title = "# Historical Canonical HF Sidecar Backfill Write Audit" if not bool(payload.get("dry_run", True)) else "# Historical Canonical HF Sidecar Backfill Dry Run"
    lines = [
        title,
        "",
        f"- dry_run: `{payload.get('dry_run')}`",
        f"- would_write_anything: `{payload.get('would_write_anything')}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        f"- can_backfill_now_count: `{summary.get('can_backfill_now_count', 0)}`",
        "",
        "## Strategy",
        "",
        f"- default_mode: {strategy.get('default_mode', '')}",
        "- load path: use `load_result(...)` for metadata-only result roots; use existing archive loaders only when raw archives exist.",
        "- safety: no bulk writes to `results/`; no heavy compute; no cRPA.",
        "",
        "### System status",
        "",
    ]
    for system in strategy.get("systems", []):
        if not isinstance(system, Mapping):
            continue
        lines.extend(
            [
                f"- `{system.get('system')}`: `{system.get('current_status')}` (safe_now=`{system.get('safe_now')}`)",
                f"  - blocker: {system.get('blocker')}",
            ]
        )
        contract = _mapping(system.get("archive_format_contract"))
        if contract:
            lines.append(f"  - archive raw object: `{contract.get('raw_object')}`")
            lines.append(f"  - state NPZ keys: `{', '.join(str(key) for key in contract.get('state_npz_required_keys', []))}`")
            lines.append(f"  - projected-basis NPZ keys: `{', '.join(str(key) for key in contract.get('projected_basis_npz_required_keys', []))}`")
    lines.extend(["", "## Decision counts", ""])
    for key, value in _mapping(summary.get("decision_counts")).items():
        lines.append(f"- `{key}`: `{value}`")
    write_plan = _mapping(payload.get("write_plan"))
    if write_plan:
        write_summary = _mapping(write_plan.get("summary"))
        lines.extend(
            [
                "",
                "## Write/staging plan",
                "",
                f"- mode: `{write_plan.get('mode')}`",
                f"- target_root: `{write_plan.get('target_root')}`",
                f"- write_manifest_path: `{write_plan.get('write_manifest_path')}`",
                f"- historical_results_mutated: `{write_plan.get('historical_results_mutated')}`",
                f"- planned_count: `{write_summary.get('planned_count', 0)}`",
                f"- written_count: `{write_summary.get('written_count', 0)}`",
                f"- skipped_count: `{write_summary.get('skipped_count', 0)}`",
                f"- write_error_count: `{write_summary.get('write_error_count', 0)}`",
                "",
            ]
        )
        for entry in write_plan.get("entries", []):
            if not isinstance(entry, Mapping):
                continue
            lines.extend(
                [
                    f"### write entry {entry.get('index')} — {entry.get('status')}",
                    f"- target_dir: `{entry.get('target_dir')}`",
                    f"- skipped_reason: {entry.get('skipped_reason')}",
                ]
            )
            planned_files = _mapping(entry.get("planned_files"))
            if planned_files:
                lines.append("- planned_files:")
                lines.extend(f"  - `{key}`: `{value}`" for key, value in planned_files.items())
            written_files = entry.get("written_files", [])
            if written_files:
                lines.append("- written_files:")
                lines.extend(f"  - `{value}`" for value in written_files)
            if entry.get("error"):
                lines.append(f"- error: {entry.get('error')}")
            lines.append("")
    lines.extend(["", "## Records", ""])
    for item in payload.get("records", []):
        if not isinstance(item, Mapping):
            continue
        evidence = item.get("evidence", [])
        blockers = item.get("blockers", [])
        uncertainty = item.get("uncertainty", [])
        lines.extend(
            [
                f"### {item.get('decision')} — {item.get('root')}",
                "",
                f"- kind: `{item.get('kind')}`",
                f"- system: `{item.get('system_name')}`",
                f"- workflow: `{item.get('workflow')}`",
                f"- can_backfill_now: `{item.get('can_backfill_now')}`",
                f"- would_write: `{item.get('would_write')}`",
                f"- reason: {item.get('reason')}",
            ]
        )
        if evidence:
            lines.append("- evidence:")
            lines.extend(f"  - {entry}" for entry in evidence)
        if blockers:
            lines.append("- blockers:")
            lines.extend(f"  - {entry}" for entry in blockers)
        if uncertainty:
            lines.append("- uncertainty:")
            lines.extend(f"  - {entry}" for entry in uncertainty)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
