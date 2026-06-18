from __future__ import annotations

"""Dry-run inventory helper for historical canonical HF sidecar backfills.

This module deliberately does not mutate historical result directories.  It
uses :func:`mean_field.api.load_result` for metadata-only result inspection and
only opens recognized RLG/hBN ``hf_ground_state.npz`` archives far enough to
inspect their key list and small scalar cache metadata.  It never reconstructs
HF states, reruns SCF, diagonalizes grids, computes cRPA, or writes into
``results/``.
"""

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from mean_field.api import load_result
from mean_field.core.io import write_json_artifact, write_text_artifact

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_ROOT = REPO_ROOT / "results"
_CANONICAL_HF_SIDECAR_KEY = "canonical_hf_run_result"
_CANONICAL_HF_SIDECAR_FILE = "canonical_hf_run_result.json"

_TDBG_ADAPTER = "mean_field.systems.tdbg.projected_hf_contracts.tdbg_projected_hf_result_to_hf_run_result"
_HTG_ADAPTER = "mean_field.systems.htg.supercell_contracts.htg_supercell_hf_run_to_hf_run_result"
_RLG_HBN_ARCHIVE_LOADER = "mean_field.systems.RnG_hBN.tdhf.load_rlg_hbn_tdhf_run_from_archive"
_RLG_HBN_ADAPTER = "mean_field.systems.RnG_hBN.hf_contracts.rlg_hbn_hf_run_to_hf_run_result"

_RLG_HBN_REQUIRED_ARCHIVE_KEYS = frozenset(
    {
        "density",
        "hamiltonian",
        "h0",
        "energies_mev",
        "reference_density",
        "cache_key_basis",
        "cache_key_overlap",
        "cache_dir",
    }
)


@dataclass(frozen=True)
class BackfillCandidate:
    """One scanned historical candidate.

    ``can_backfill_now`` means the existing repository has enough *loader and
    adapter surface* to reconstruct a canonical contract object without heavy
    compute.  It does not mean this dry-run helper will write anything.
    """

    kind: str
    root: str
    system_name: str
    workflow: str
    decision: str
    can_backfill_now: bool
    would_write: bool
    reason: str
    manifest_path: str | None = None
    archive_path: str | None = None
    target_root: str | None = None
    evidence: tuple[str, ...] = ()
    adapters: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "root": self.root,
            "system_name": self.system_name,
            "workflow": self.workflow,
            "decision": self.decision,
            "can_backfill_now": bool(self.can_backfill_now),
            "would_write": bool(self.would_write),
            "reason": self.reason,
            "manifest_path": self.manifest_path,
            "archive_path": self.archive_path,
            "target_root": self.target_root,
            "evidence": list(self.evidence),
            "adapters": list(self.adapters),
            "blockers": list(self.blockers),
            "uncertainty": list(self.uncertainty),
            "metadata": dict(self.metadata),
        }


def _json_load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_or_empty(value: object) -> str:
    return "" if value is None else str(value)


def _normal_system_name(system_name: str, workflow: str = "") -> str:
    text = f"{system_name} {workflow}".lower().replace("-", "_")
    if "tdbg" in text:
        return "tdbg"
    if "htg_supercell" in text or ("htg" in text and "supercell" in text):
        return "htg_supercell"
    if "rlg_hbn" in text or "rng_hbn" in text or "rng/hbn" in text or "rlg/hbn" in text:
        return "rlg_hbn"
    return system_name.lower()


def _manifest_sidecar_path(root: Path, files: Mapping[str, Any], key: str) -> Path | None:
    raw = files.get(key)
    if not isinstance(raw, str):
        return None
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return root / relative


def _manifest_candidate_roots(root: Path, max_candidates: int) -> list[Path]:
    candidates: list[Path] = []
    if root.is_file():
        if root.name == "manifest.json":
            return [root.parent]
        return []
    if not root.exists():
        return []
    if (root / "manifest.json").is_file():
        candidates.append(root)
    for manifest_path in sorted(root.rglob("manifest.json")):
        parent = manifest_path.parent
        if parent not in candidates:
            candidates.append(parent)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _rlg_hbn_archive_candidates(root: Path, max_candidates: int) -> list[Path]:
    if root.is_file():
        return [root] if root.name in {"hf_ground_state.npz", "hf_run_state.npz"} else []
    if not root.exists():
        return []
    candidates = list(sorted(root.rglob("hf_ground_state.npz")))
    if len(candidates) >= max_candidates:
        return candidates[:max_candidates]
    # Individual run archives are useful inventory evidence, but ground-state
    # archives remain the preferred historical backfill target.
    for archive_path in sorted(root.rglob("hf_run_state.npz")):
        if archive_path not in candidates:
            candidates.append(archive_path)
        if len(candidates) >= max_candidates:
            break
    return candidates[:max_candidates]


def _classify_manifest_root(root: Path) -> BackfillCandidate:
    manifest_path = root / "manifest.json"
    load_error: str | None = None
    result = None
    try:
        result = load_result(root)
        manifest = result.manifest
        model = result.model
    except Exception as exc:  # pragma: no cover - exercised by malformed local artifacts.
        load_error = f"{type(exc).__name__}: {exc}"
        try:
            manifest = _json_load(manifest_path)
        except Exception as manifest_exc:
            return BackfillCandidate(
                kind="result_manifest",
                root=str(root),
                manifest_path=str(manifest_path),
                system_name="unknown",
                workflow="unknown",
                decision="scan_error",
                can_backfill_now=False,
                would_write=False,
                reason=f"load_result and direct manifest read failed: {type(manifest_exc).__name__}: {manifest_exc}",
                blockers=(load_error,),
                uncertainty=("The result root may be malformed or not a mean-field artifact root.",),
            )
        model = _mapping(manifest.get("model"))

    metadata = _mapping(manifest.get("metadata"))
    files = _mapping(manifest.get("files"))
    model_mapping = _mapping(model)
    system_name = _string_or_empty(metadata.get("system_name") or model_mapping.get("system_name") or "unknown")
    workflow = _string_or_empty(metadata.get("workflow") or "unknown")
    normal_system = _normal_system_name(system_name, workflow)
    evidence = [
        "metadata-only scan via mean_field.api.load_result",
        f"manifest={manifest_path}",
    ]
    if load_error is not None:
        evidence.append(f"load_result_error={load_error}")

    if result is not None and result.canonical_hf_run_result is not None:
        sidecar_path = _manifest_sidecar_path(root, files, _CANONICAL_HF_SIDECAR_KEY)
        return BackfillCandidate(
            kind="result_manifest",
            root=str(root),
            manifest_path=str(manifest_path),
            target_root=str(root),
            system_name=system_name,
            workflow=workflow,
            decision="already_canonical",
            can_backfill_now=False,
            would_write=False,
            reason="manifest already references a valid canonical_hf_run_result sidecar; no historical backfill is needed",
            evidence=tuple(evidence + ([f"sidecar={sidecar_path}"] if sidecar_path is not None else [])),
        )

    if _CANONICAL_HF_SIDECAR_KEY in files:
        return BackfillCandidate(
            kind="result_manifest",
            root=str(root),
            manifest_path=str(manifest_path),
            target_root=str(root),
            system_name=system_name,
            workflow=workflow,
            decision="broken_canonical_reference",
            can_backfill_now=False,
            would_write=False,
            reason="manifest mentions canonical_hf_run_result but load_result did not load a valid sidecar",
            evidence=tuple(evidence),
            blockers=("repair requires explicit inspection; dry-run scanner will not rewrite manifests",),
            uncertainty=("The sidecar may be missing, invalid, or outside the result root.",),
        )

    orphan_sidecar = root / _CANONICAL_HF_SIDECAR_FILE
    if orphan_sidecar.is_file():
        return BackfillCandidate(
            kind="result_manifest",
            root=str(root),
            manifest_path=str(manifest_path),
            target_root=str(root),
            system_name=system_name,
            workflow=workflow,
            decision="orphan_canonical_sidecar",
            can_backfill_now=False,
            would_write=False,
            reason="canonical_hf_run_result.json exists but is not referenced by manifest.json",
            evidence=tuple(evidence + [f"orphan_sidecar={orphan_sidecar}"]),
            blockers=("manifest update would mutate a historical result and is intentionally not done by dry-run",),
        )

    if normal_system == "tdbg":
        blockers = (
            "historical TDBG hf_state.npz stores final matrices but not the full TDBGProjectedHFData object",
            "canonical ProjectedBasis requires micro_wavefunctions and labels from the raw projected basis",
            "rebuilding missing basis data would require a system archive loader or fresh diagonalization, which this helper will not do",
        )
        return BackfillCandidate(
            kind="result_manifest",
            root=str(root),
            manifest_path=str(manifest_path),
            target_root=str(root),
            system_name=system_name,
            workflow=workflow,
            decision="requires_archive_loader",
            can_backfill_now=False,
            would_write=False,
            reason="TDBG has an in-memory canonical adapter but historical result roots do not contain a loadable TDBGProjectedHFResult contract object",
            evidence=tuple(evidence + [f"files={sorted(str(key) for key in files)}"]),
            adapters=(_TDBG_ADAPTER,),
            blockers=blockers,
            uncertainty=("Safe backfill needs an archive loader that restores TDBGProjectedHFResult without recomputing physics.",),
        )

    if normal_system == "htg_supercell":
        blockers = (
            "saved HTG supercell NPZ archives do not by themselves restore HTGSupercellProjectedBasisData.basis.wavefunctions",
            "canonical ProjectedBasis requires model/interaction metadata and micro_wavefunctions from the raw run object",
            "no repository loader currently restores HTGSupercellHartreeFockRun from historical output roots",
        )
        return BackfillCandidate(
            kind="result_manifest",
            root=str(root),
            manifest_path=str(manifest_path),
            target_root=str(root),
            system_name=system_name,
            workflow=workflow,
            decision="requires_archive_loader",
            can_backfill_now=False,
            would_write=False,
            reason="HTG supercell has an in-memory canonical adapter but needs a raw-run archive loader before historical sidecars can be generated safely",
            evidence=tuple(evidence + [f"files={sorted(str(key) for key in files)}"]),
            adapters=(_HTG_ADAPTER,),
            blockers=blockers,
            uncertainty=("Safe backfill needs an archive loader that restores HTGSupercellHartreeFockRun without recomputing physics.",),
        )

    if normal_system == "rlg_hbn":
        return BackfillCandidate(
            kind="result_manifest",
            root=str(root),
            manifest_path=str(manifest_path),
            target_root=str(root),
            system_name=system_name,
            workflow=workflow,
            decision="scan_archives_for_eligible_panels",
            can_backfill_now=False,
            would_write=False,
            reason="RLG/hBN paper-HF manifests are workflow/container roots; individual hf_ground_state.npz archives determine canonical backfill eligibility",
            evidence=tuple(evidence + [f"files={sorted(str(key) for key in files)}"]),
            adapters=(_RLG_HBN_ARCHIVE_LOADER, _RLG_HBN_ADAPTER),
            uncertainty=("Sidecar placement for multi-panel workflow roots must be explicit before any write-mode is implemented.",),
        )

    if "hf" in workflow.lower() or "hartree" in workflow.lower():
        decision = "unsupported_hf_workflow"
        reason = "no historical canonical sidecar backfill rule is registered for this HF workflow"
    else:
        decision = "not_hf_result"
        reason = "manifest does not look like an HF result root targeted by canonical HF sidecar backfill"
    return BackfillCandidate(
        kind="result_manifest",
        root=str(root),
        manifest_path=str(manifest_path),
        target_root=str(root),
        system_name=system_name,
        workflow=workflow,
        decision=decision,
        can_backfill_now=False,
        would_write=False,
        reason=reason,
        evidence=tuple(evidence + [f"files={sorted(str(key) for key in files)}"]),
        uncertainty=("Only TDBG, HTG supercell, and RLG/hBN historical rules were audited in this task.",),
    )


def _npz_string(data: Mapping[str, np.ndarray], key: str) -> str:
    if key not in data:
        return ""
    value = np.asarray(data[key])
    if value.size == 0:
        return ""
    return str(value.reshape(-1)[0])


def _npz_bool(data: Mapping[str, np.ndarray], key: str, *, default: bool = False) -> bool:
    if key not in data:
        return bool(default)
    value = np.asarray(data[key]).reshape(-1)
    if value.size == 0:
        return bool(default)
    item = value[0]
    if isinstance(item, np.bool_ | bool):
        return bool(item)
    if isinstance(item, np.integer | int):
        return bool(int(item))
    return str(item).strip().lower() not in {"", "0", "false", "no", "off"}


def _classify_rlg_hbn_archive(archive_path: Path) -> BackfillCandidate:
    evidence = [
        f"archive={archive_path}",
        "header/scalar-only NPZ inspection; arrays are not materialized for physics",
    ]
    try:
        with np.load(archive_path, allow_pickle=False) as payload:
            keys = frozenset(str(key) for key in payload.files)
            archive_scalars = {key: np.asarray(payload[key]) for key in (keys & {"cache_key_basis", "cache_key_overlap", "cache_dir", "zero_literal_q0_fock"})}
    except Exception as exc:
        return BackfillCandidate(
            kind="rlg_hbn_archive",
            root=str(archive_path.parent),
            archive_path=str(archive_path),
            target_root=str(archive_path.parent),
            system_name="rlg_hbn",
            workflow="rlg_hbn.paper_hf.archive",
            decision="scan_error",
            can_backfill_now=False,
            would_write=False,
            reason=f"could not inspect RLG/hBN archive: {type(exc).__name__}: {exc}",
            evidence=tuple(evidence),
            blockers=("archive could not be opened with np.load(..., allow_pickle=False)",),
        )

    missing_keys = sorted(_RLG_HBN_REQUIRED_ARCHIVE_KEYS.difference(keys))
    if missing_keys:
        return BackfillCandidate(
            kind="rlg_hbn_archive",
            root=str(archive_path.parent),
            archive_path=str(archive_path),
            target_root=str(archive_path.parent),
            system_name="rlg_hbn",
            workflow="rlg_hbn.paper_hf.archive",
            decision="incomplete_archive",
            can_backfill_now=False,
            would_write=False,
            reason="RLG/hBN archive is missing keys required by the existing archive loader",
            evidence=tuple(evidence + [f"archive_keys={sorted(keys)}"]),
            adapters=(_RLG_HBN_ARCHIVE_LOADER, _RLG_HBN_ADAPTER),
            blockers=tuple(f"missing archive key: {key}" for key in missing_keys),
        )

    if _npz_bool(archive_scalars, "zero_literal_q0_fock", default=False):
        return BackfillCandidate(
            kind="rlg_hbn_archive",
            root=str(archive_path.parent),
            archive_path=str(archive_path),
            target_root=str(archive_path.parent),
            system_name="rlg_hbn",
            workflow="rlg_hbn.paper_hf.archive",
            decision="ineligible_zero_literal_q0_fock",
            can_backfill_now=False,
            would_write=False,
            reason="existing RLG/hBN archive loader intentionally rejects zero_literal_q0_fock archives",
            evidence=tuple(evidence),
            adapters=(_RLG_HBN_ARCHIVE_LOADER, _RLG_HBN_ADAPTER),
            blockers=("zero_literal_q0_fock=1 archives are not TDHF/canonical-postprocessing compatible",),
        )

    cache_dir_text = _npz_string(archive_scalars, "cache_dir")
    basis_key = _npz_string(archive_scalars, "cache_key_basis")
    overlap_key = _npz_string(archive_scalars, "cache_key_overlap")
    blockers: list[str] = []
    cache_metadata: dict[str, object] = {
        "cache_dir": cache_dir_text,
        "cache_key_basis": basis_key,
        "cache_key_overlap": overlap_key,
    }
    if not cache_dir_text:
        blockers.append("cache_dir scalar is empty")
    if not basis_key:
        blockers.append("cache_key_basis scalar is empty")
    if not overlap_key:
        blockers.append("cache_key_overlap scalar is empty")

    if cache_dir_text and basis_key:
        basis_cache = Path(cache_dir_text).expanduser() / "basis" / basis_key
        cache_metadata["basis_cache"] = str(basis_cache)
        if not (basis_cache / "manifest.json").is_file():
            blockers.append(f"basis cache manifest not found: {basis_cache / 'manifest.json'}")
    if cache_dir_text and overlap_key:
        overlap_cache = Path(cache_dir_text).expanduser() / "overlap" / overlap_key
        cache_metadata["overlap_cache"] = str(overlap_cache)
        if not (overlap_cache / "manifest.json").is_file():
            blockers.append(f"overlap cache manifest not found: {overlap_cache / 'manifest.json'}")

    if blockers:
        return BackfillCandidate(
            kind="rlg_hbn_archive",
            root=str(archive_path.parent),
            archive_path=str(archive_path),
            target_root=str(archive_path.parent),
            system_name="rlg_hbn",
            workflow="rlg_hbn.paper_hf.archive",
            decision="missing_loader_inputs",
            can_backfill_now=False,
            would_write=False,
            reason="RLG/hBN archive has the right shape for the existing loader but required cache inputs are unavailable",
            evidence=tuple(evidence),
            adapters=(_RLG_HBN_ARCHIVE_LOADER, _RLG_HBN_ADAPTER),
            blockers=tuple(blockers),
            metadata=cache_metadata,
        )

    return BackfillCandidate(
        kind="rlg_hbn_archive",
        root=str(archive_path.parent),
        archive_path=str(archive_path),
        target_root=str(archive_path.parent),
        system_name="rlg_hbn",
        workflow="rlg_hbn.paper_hf.archive",
        decision="eligible_with_existing_archive_loader",
        can_backfill_now=True,
        would_write=False,
        reason="RLG/hBN archive records cache_dir/cache keys and can be reconstructed by the existing archive loader without rerunning SCF",
        evidence=tuple(evidence + ["load_rlg_hbn_tdhf_run_from_archive + rlg_hbn_hf_run_to_hf_run_result is the existing safe reconstruction path"]),
        adapters=(_RLG_HBN_ARCHIVE_LOADER, _RLG_HBN_ADAPTER),
        uncertainty=(
            "This dry-run scanner does not load the full basis/overlap arrays; full reconstruction should be verified in an explicit write/staging workflow.",
            "Historical sidecar target placement for panel archives must be approved before mutation is implemented.",
        ),
        metadata=cache_metadata,
    )


def scan_backfill_candidates(
    roots: Sequence[str | Path],
    *,
    include_archives: bool = True,
    max_candidates: int = 10000,
) -> list[BackfillCandidate]:
    """Return dry-run canonical HF sidecar backfill candidates.

    The scan is metadata-only except for recognized RLG/hBN NPZ header/scalar
    inspection.  It never writes into result directories.
    """

    records: list[BackfillCandidate] = []
    seen_manifest_roots: set[Path] = set()
    seen_archives: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        for candidate_root in _manifest_candidate_roots(root, max_candidates=max_candidates):
            resolved = candidate_root.resolve()
            if resolved in seen_manifest_roots:
                continue
            seen_manifest_roots.add(resolved)
            records.append(_classify_manifest_root(candidate_root))
        if include_archives:
            for archive_path in _rlg_hbn_archive_candidates(root, max_candidates=max_candidates):
                resolved = archive_path.resolve()
                if resolved in seen_archives:
                    continue
                seen_archives.add(resolved)
                records.append(_classify_rlg_hbn_archive(archive_path))
    return records


def backfill_strategy() -> dict[str, object]:
    return {
        "default_mode": "dry-run only; do not mutate historical results by default",
        "safe_write_policy": [
            "Never fabricate canonical HF physics from summary-only artifacts.",
            "Only write after an explicit future flag and after reconstructing a typed HFRunResult from a raw in-memory object or audited archive loader.",
            "Prefer staging/patch report first; historical results/ must not receive bulk writes.",
        ],
        "systems": [
            {
                "system": "tdbg",
                "current_status": "needs_archive_loader_for_historical_roots",
                "existing_adapter": _TDBG_ADAPTER,
                "safe_now": False,
                "blocker": "Historical hf_state.npz roots do not contain full TDBGProjectedHFResult/TDBGProjectedHFData, especially micro_wavefunctions and labels.",
            },
            {
                "system": "htg_supercell",
                "current_status": "needs_archive_loader_for_historical_roots",
                "existing_adapter": _HTG_ADAPTER,
                "safe_now": False,
                "blocker": "Saved run NPZ is not a full HTGSupercellHartreeFockRun archive with basis wavefunctions/model/interaction objects.",
            },
            {
                "system": "rlg_hbn / RnG_hBN",
                "current_status": "eligible_when_hf_ground_state_archive_and_cache_entries_are_present",
                "existing_loader": _RLG_HBN_ARCHIVE_LOADER,
                "existing_adapter": _RLG_HBN_ADAPTER,
                "safe_now": True,
                "blocker": "Write-mode target placement for multi-panel historical workflow roots is not implemented in this dry-run helper.",
            },
        ],
    }


def inventory_payload(
    records: Sequence[BackfillCandidate],
    *,
    roots: Sequence[str | Path],
    dry_run: bool = True,
) -> dict[str, object]:
    decision_counts = Counter(record.decision for record in records)
    system_counts = Counter(record.system_name for record in records)
    return {
        "schema_version": 1,
        "dry_run": bool(dry_run),
        "would_write_anything": False,
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


def render_markdown_inventory(payload: Mapping[str, object]) -> str:
    summary = _mapping(payload.get("summary"))
    strategy = _mapping(payload.get("strategy"))
    lines = [
        "# Historical Canonical HF Sidecar Backfill Dry Run",
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
    lines.extend(["", "## Decision counts", ""])
    for key, value in _mapping(summary.get("decision_counts")).items():
        lines.append(f"- `{key}`: `{value}`")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-only inventory for historical canonical HF sidecar backfill eligibility."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[DEFAULT_RESULT_ROOT],
        help="Result roots to scan. Defaults to repository results/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Default and only mutation mode: scan/report without modifying historical results.",
    )
    parser.add_argument("--no-archives", action="store_true", help="Skip RLG/hBN hf_ground_state.npz archive inventory.")
    parser.add_argument("--max-candidates", type=int, default=10000, help="Safety cap per root and candidate kind.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional explicit JSON report path.")
    parser.add_argument("--report-md", type=Path, default=None, help="Optional explicit Markdown report path.")
    parser.add_argument(
        "--fail-on-ineligible",
        action="store_true",
        help="Return nonzero if any scanned candidate is not already canonical or eligible via an existing loader.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = list(args.roots)
    records = scan_backfill_candidates(
        roots,
        include_archives=not bool(args.no_archives),
        max_candidates=int(args.max_candidates),
    )
    payload = inventory_payload(records, roots=roots, dry_run=True)
    markdown = render_markdown_inventory(payload)
    if args.report_json is not None:
        write_json_artifact(payload, args.report_json)
    if args.report_md is not None:
        write_text_artifact(markdown, args.report_md)
    print(markdown, end="")
    if bool(args.fail_on_ineligible):
        allowed = {"already_canonical", "eligible_with_existing_archive_loader"}
        if any(record.decision not in allowed for record in records):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
