from __future__ import annotations

from ._shared import *  # noqa: F401,F403
from ._scan_utils import *  # noqa: F401,F403
from ._scan_contracts import *  # noqa: F401,F403

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
        blockers, archive_metadata = _tdbg_contract_blockers(root, files)
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
            uncertainty=(
                "Safe backfill needs an archive loader that restores TDBGProjectedHFResult without recomputing physics.",
                "Existing summaries/matrices are insufficient when projected micro-wavefunctions or run-history scalars are absent.",
            ),
            metadata=archive_metadata,
        )

    if normal_system == "htg":
        blockers, archive_metadata = _htg_primitive_contract_blockers(root, files)
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
            reason="HTG primitive has an in-memory canonical adapter but needs a raw-run archive loader before historical sidecars can be generated safely",
            evidence=tuple(evidence + [f"files={sorted(str(key) for key in files)}"]),
            adapters=(_HTG_PRIMITIVE_ADAPTER,),
            blockers=blockers,
            uncertainty=(
                "Safe backfill needs an archive loader that restores HTGHartreeFockRun without recomputing physics.",
                "Existing state archives are insufficient when projected micro-wavefunctions/model/interaction metadata are absent.",
            ),
            metadata=archive_metadata,
        )

    if normal_system == "htg_supercell":
        blockers, archive_metadata = _htg_supercell_contract_blockers(root, files)
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
            uncertainty=(
                "Safe backfill needs an archive loader that restores HTGSupercellHartreeFockRun without recomputing physics.",
                "Existing state archives are insufficient when projected micro-wavefunctions/model/interaction metadata are absent.",
            ),
            metadata=archive_metadata,
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


def _classify_tdbg_archive(archive_path: Path) -> BackfillCandidate:
    root = archive_path.parent
    blockers, archive_metadata = _tdbg_contract_blockers(root, {"hf_state": archive_path.name})
    evidence = [
        f"archive={archive_path}",
        "header-only TDBG archive inspection; arrays are not materialized for physics",
    ]
    eligible = len(blockers) == 0
    return BackfillCandidate(
        kind="tdbg_archive",
        root=str(root),
        archive_path=str(archive_path),
        target_root=str(root),
        system_name="tdbg",
        workflow="tdbg.projected_hf.archive",
        decision="eligible_with_existing_archive_loader" if eligible else "requires_archive_loader",
        can_backfill_now=eligible,
        would_write=False,
        reason=(
            "TDBG projected-HF archive satisfies the complete raw TDBGProjectedHFResult loader contract"
            if eligible
            else "TDBG projected-HF archive cannot be backfilled until it satisfies the exact TDBGProjectedHFResult raw-object loader contract"
        ),
        evidence=tuple(evidence),
        adapters=(_TDBG_ARCHIVE_LOADER, _TDBG_ADAPTER),
        blockers=blockers,
        uncertainty=(
            "Do not fabricate projected micro-wavefunctions or raw run history from final matrices.",
            "Eligible records are still staged-only; applying sidecars to historical roots requires a separate approval step.",
        ),
        metadata=archive_metadata,
    )


def _classify_htg_archive(archive_path: Path) -> BackfillCandidate:
    root = archive_path.parent
    keys, key_error = _npz_key_set(archive_path)
    is_supercell = archive_path.name.startswith("hf_supercell") or (keys is not None and "supercell_matrix" in keys)
    if is_supercell:
        blockers, archive_metadata = _htg_supercell_contract_blockers(root, {"hf_supercell_ground_state": archive_path.name}, archive_name=archive_path.name)
        system_name = "htg_supercell"
        workflow = "htg.supercell_hf.archive"
        adapter = _HTG_ADAPTER
        loader = _HTG_ARCHIVE_LOADER
        raw_name = "HTGSupercellHartreeFockRun"
    else:
        blockers, archive_metadata = _htg_primitive_contract_blockers(root, {"hf_ground_state": archive_path.name}, archive_name=archive_path.name)
        system_name = "htg"
        workflow = "htg.primitive_hf.archive"
        adapter = _HTG_PRIMITIVE_ADAPTER
        loader = _HTG_PRIMITIVE_ARCHIVE_LOADER
        raw_name = "HTGHartreeFockRun"
    evidence = [
        f"archive={archive_path}",
        "header-only HTG archive inspection; arrays are not materialized for physics",
    ]
    if key_error is not None:
        evidence.append(f"npz_header_error={key_error}")
    eligible = len(blockers) == 0
    return BackfillCandidate(
        kind="htg_supercell_archive" if is_supercell else "htg_primitive_archive",
        root=str(root),
        archive_path=str(archive_path),
        target_root=str(root),
        system_name=system_name,
        workflow=workflow,
        decision="eligible_with_existing_archive_loader" if eligible else "requires_archive_loader",
        can_backfill_now=eligible,
        would_write=False,
        reason=(
            f"HTG archive satisfies the complete raw {raw_name} loader contract"
            if eligible
            else f"HTG archive cannot be backfilled until it satisfies the exact {raw_name} raw-object loader contract"
        ),
        evidence=tuple(evidence),
        adapters=(loader, adapter),
        blockers=blockers,
        uncertainty=(
            "Do not fabricate projected micro-wavefunctions/model/interaction metadata from final matrices.",
            "Eligible records are still staged-only; applying sidecars to historical roots requires a separate approval step.",
        ),
        metadata=archive_metadata,
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
        else:
            blockers.extend(_rlg_hbn_cache_manifest_blockers(basis_cache, cache_kind="basis"))
    if cache_dir_text and overlap_key:
        overlap_cache = Path(cache_dir_text).expanduser() / "overlap" / overlap_key
        cache_metadata["overlap_cache"] = str(overlap_cache)
        if not (overlap_cache / "manifest.json").is_file():
            blockers.append(f"overlap cache manifest not found: {overlap_cache / 'manifest.json'}")
        else:
            blockers.extend(_rlg_hbn_cache_manifest_blockers(overlap_cache, cache_kind="overlap"))

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

__all__ = [name for name in globals() if not name.startswith('__')]
