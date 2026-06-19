from __future__ import annotations

"""Dry-run-first helper for historical canonical HF sidecar backfills.

This module deliberately does not mutate historical result directories by
default.  It uses :func:`mean_field.api.load_result` for metadata-only result
inspection and only opens recognized RLG/hBN ``hf_ground_state.npz`` archives
far enough to inspect their key list and small scalar cache metadata.  The
opt-in write path is staging-only: it requires ``--write``, a caller-specified
``--target-root``, and an explicit target allowlist; it never writes into scanned
historical roots.  It never reruns SCF, diagonalizes grids, computes cRPA, or
writes into ``results/`` unless the caller explicitly stages somewhere under an
allowlisted target outside the historical tree.
"""

import argparse
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
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
_WRITE_MANIFEST_FILE = "backfill_write_manifest.json"
_BACKFILL_AUDIT_FILE = "canonical_hf_backfill_audit.json"
_MANIFEST_PATCH_FILE = "canonical_hf_manifest_patch.json"

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
    from mean_field.systems.RnG_hBN.tdhf import load_rlg_hbn_tdhf_run_from_archive

    return load_rlg_hbn_tdhf_run_from_archive(archive_path)

def _default_rlg_hbn_adapter(run: object, *, archive_manifest: dict[str, object]) -> object:
    from mean_field.systems.RnG_hBN.hf_contracts import rlg_hbn_hf_run_to_hf_run_result

    return rlg_hbn_hf_run_to_hf_run_result(run, archive_manifest=archive_manifest)  # type: ignore[arg-type]

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

def backfill_strategy() -> dict[str, object]:
    return {
        "default_mode": "dry-run by default; explicit --write is staging-only and never mutates scanned historical roots",
        "safe_write_policy": [
            "Never fabricate canonical HF physics from summary-only artifacts.",
            "Write-mode requires --write, --target-root, and at least one --allow-target-root allowlist entry.",
            "Only eligible RLG/hBN archives are materialized through the existing archive loader and canonical adapter.",
            "TDBG/HTG remain blocked until archive loaders restore their raw run objects without recomputing physics.",
            "Staged writes produce sidecars, manifest patches, and audit manifests under the caller-specified target root only.",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-first inventory and allowlisted staging helper for historical canonical HF sidecar backfill eligibility."
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
        help="Default mode: scan/report without writing staged sidecars or modifying historical results.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Explicitly materialize eligible sidecars into --target-root staging directories; scanned historical roots are never mutated.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=None,
        help="Caller-specified staging root for --write outputs or dry-run write plans.",
    )
    parser.add_argument(
        "--allow-target-root",
        type=Path,
        action="append",
        default=[],
        help="Allowlist parent for --target-root. Required with --target-root/--write; repeatable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing files in the staging target root. Default refuses overwrites.",
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.write and args.target_root is None:
        parser.error("--write requires --target-root")
    if args.target_root is not None and not args.allow_target_root:
        parser.error("--target-root requires at least one --allow-target-root")

    roots = list(args.roots)
    records = scan_backfill_candidates(
        roots,
        include_archives=not bool(args.no_archives),
        max_candidates=int(args.max_candidates),
    )
    write_plan: dict[str, object] | None = None
    if args.target_root is not None:
        if bool(args.write):
            write_plan = execute_backfill_writes(
                records,
                roots=roots,
                target_root=args.target_root,
                allow_target_roots=args.allow_target_root,
                overwrite=bool(args.overwrite),
            )
        else:
            write_plan = plan_backfill_writes(
                records,
                roots=roots,
                target_root=args.target_root,
                allow_target_roots=args.allow_target_root,
                overwrite=bool(args.overwrite),
            )
    payload = inventory_payload(records, roots=roots, dry_run=not bool(args.write), write_plan=write_plan)
    markdown = render_markdown_inventory(payload)
    if args.report_json is not None:
        write_json_artifact(payload, args.report_json)
    if args.report_md is not None:
        write_text_artifact(markdown, args.report_md)
    print(markdown, end="")
    if write_plan is not None and not bool(write_plan.get("dry_run", True)):
        write_summary = _mapping(write_plan.get("summary"))
        if int(write_summary.get("write_error_count", 0)):
            return 3
    if bool(args.fail_on_ineligible):
        allowed = {"already_canonical", "eligible_with_existing_archive_loader"}
        if any(record.decision not in allowed for record in records):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
