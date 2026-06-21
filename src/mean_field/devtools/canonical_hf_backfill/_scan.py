from __future__ import annotations

from ._shared import *  # noqa: F401,F403

def _json_load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _rlg_hbn_cache_manifest_blockers(cache_path: Path, *, cache_kind: str) -> tuple[str, ...]:
    """Return loader-compatibility blockers from a RLG/hBN cache manifest."""

    manifest_path = cache_path / "manifest.json"
    try:
        manifest = _json_load(manifest_path)
    except Exception as exc:  # pragma: no cover - exact filesystem failures vary.
        return (
            f"{cache_kind} cache manifest could not be read: {manifest_path}: "
            f"{type(exc).__name__}: {exc}",
        )
    extra = manifest.get("extra")
    if not isinstance(extra, Mapping):
        return (f"{cache_kind} cache manifest has invalid/missing extra metadata: {manifest_path}",)

    blockers: list[str] = []
    basis_periodic_gauge = extra.get("basis_periodic_gauge")
    if basis_periodic_gauge != _RLG_HBN_EXPECTED_BASIS_PERIODIC_GAUGE:
        blockers.append(
            f"{cache_kind} cache {cache_path} uses incompatible basis_periodic_gauge "
            f"{basis_periodic_gauge!r}; expected {_RLG_HBN_EXPECTED_BASIS_PERIODIC_GAUGE!r}"
        )
    form_factor_convention = extra.get("form_factor_convention")
    if form_factor_convention != _RLG_HBN_EXPECTED_FORM_FACTOR_CONVENTION:
        blockers.append(
            f"{cache_kind} cache {cache_path} uses incompatible form_factor_convention "
            f"{form_factor_convention!r}; expected {_RLG_HBN_EXPECTED_FORM_FACTOR_CONVENTION!r}"
        )
    return tuple(blockers)


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


def _artifact_path(root: Path, files: Mapping[str, Any], key: str, default_name: str) -> Path:
    return _manifest_sidecar_path(root, files, key) or (root / default_name)


def _npz_key_set(path: Path) -> tuple[frozenset[str] | None, str | None]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return frozenset(str(key) for key in payload.files), None
    except Exception as exc:  # pragma: no cover - depends on malformed local artifacts.
        return None, f"{type(exc).__name__}: {exc}"


def _missing_file_blocker(path: Path, *, role: str) -> str:
    return f"missing raw file for {role}: {path}"


def _missing_key_blockers(path: Path, missing_keys: Iterable[str], *, role: str) -> tuple[str, ...]:
    return tuple(f"{path} missing key `{key}` required for {role}" for key in sorted(missing_keys))


def _path_text_mentions(path: Path, token: str) -> bool:
    return token.lower() in str(path).lower().replace("-", "_")


def _contract_metadata(*, raw_object: str, state_keys: Iterable[str], basis_keys: Iterable[str]) -> dict[str, object]:
    return {
        "raw_object": raw_object,
        "state_npz_required_keys": sorted(str(key) for key in state_keys),
        "projected_basis_npz_required_keys": sorted(str(key) for key in basis_keys),
        "loader_policy": "metadata-only/raw-archive loader must materialize these fields without SCF, diagonalization, topology, cRPA, or fabricated wavefunctions",
    }


def _tdbg_contract_blockers(root: Path, files: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, object]]:
    state_path = _artifact_path(root, files, "hf_state", "hf_state.npz")
    labels_path = _artifact_path(root, files, "state_labels", "state_labels.json")
    summary_path = _artifact_path(root, files, "projected_hf_summary", "projected_hf_summary.json")
    basis_path = _artifact_path(root, files, "projected_basis", "projected_basis.npz")
    metadata: dict[str, object] = {
        "archive_format_contract": _contract_metadata(
            raw_object="mean_field.systems.tdbg.projected_hf_state.TDBGProjectedHFResult",
            state_keys=_TDBG_HF_STATE_CONTRACT_KEYS,
            basis_keys=_TDBG_PROJECTED_BASIS_CONTRACT_KEYS,
        ),
        "expected_files": {
            "hf_state": str(state_path),
            "state_labels": str(labels_path),
            "projected_hf_summary": str(summary_path),
            "projected_basis": str(basis_path),
        },
    }
    blockers: list[str] = []
    if not state_path.is_file():
        blockers.append(_missing_file_blocker(state_path, role="TDBGProjectedHFState arrays and run history"))
    else:
        keys, error = _npz_key_set(state_path)
        if keys is None:
            blockers.append(f"could not inspect TDBG hf_state archive {state_path}: {error}")
        else:
            metadata["hf_state_keys"] = sorted(keys)
            blockers.extend(_missing_key_blockers(state_path, _TDBG_HF_STATE_CONTRACT_KEYS.difference(keys), role="TDBGProjectedHFResult canonical adapter"))
    if not labels_path.is_file():
        blockers.append(_missing_file_blocker(labels_path, role="TDBGProjectedHFData.labels / TDBGStateLabel records"))
    if not summary_path.is_file():
        blockers.append(_missing_file_blocker(summary_path, role="TDBG run metadata, order parameters, energy components"))
    if not basis_path.is_file():
        blockers.append(
            _missing_file_blocker(
                basis_path,
                role="TDBGProjectedHFData projected-basis micro_wavefunctions, shifts, moire area, and valley parameters",
            )
        )
    else:
        keys, error = _npz_key_set(basis_path)
        if keys is None:
            blockers.append(f"could not inspect TDBG projected-basis archive {basis_path}: {error}")
        else:
            metadata["projected_basis_keys"] = sorted(keys)
            blockers.extend(_missing_key_blockers(basis_path, _TDBG_PROJECTED_BASIS_CONTRACT_KEYS.difference(keys), role="TDBGProjectedHFData exact projected basis"))
    if not (root / "model.json").is_file() and not _mapping(files).get("model"):
        blockers.append("missing raw field: TDBGProjectedHFData.model / model.json with exact TDBGModel parameters")
    if not (root / "config.json").is_file() and not _mapping(files).get("config"):
        blockers.append("missing raw field: TDBGProjectedHFData.config / config.json with exact TDBGProjectedHFConfig")
    return tuple(blockers), metadata


def _htg_primitive_contract_blockers(root: Path, files: Mapping[str, Any], *, archive_name: str = "hf_ground_state.npz") -> tuple[tuple[str, ...], dict[str, object]]:
    state_path = _artifact_path(root, files, "hf_ground_state", archive_name)
    basis_path = _artifact_path(root, files, "projected_basis", "hf_projected_basis.npz")
    params_path = _artifact_path(root, files, "hf_params", "hf_params.json")
    metadata: dict[str, object] = {
        "archive_format_contract": _contract_metadata(
            raw_object="mean_field.systems.htg.mean_field_adapter.HTGHartreeFockRun",
            state_keys=_HTG_PRIMITIVE_STATE_CONTRACT_KEYS,
            basis_keys=_HTG_PRIMITIVE_BASIS_CONTRACT_KEYS,
        ),
        "expected_files": {
            "hf_ground_state": str(state_path),
            "hf_projected_basis": str(basis_path),
            "hf_params": str(params_path),
        },
    }
    blockers: list[str] = []
    if not state_path.is_file():
        blockers.append(_missing_file_blocker(state_path, role="HTGHartreeFockState arrays and run history"))
    else:
        keys, error = _npz_key_set(state_path)
        if keys is None:
            blockers.append(f"could not inspect HTG primitive state archive {state_path}: {error}")
        else:
            metadata["hf_ground_state_keys"] = sorted(keys)
            blockers.extend(_missing_key_blockers(state_path, _HTG_PRIMITIVE_STATE_CONTRACT_KEYS.difference(keys), role="HTGHartreeFockRun canonical adapter"))
    if not basis_path.is_file():
        blockers.append(
            _missing_file_blocker(
                basis_path,
                role="HTGProjectedBasisData.basis.wavefunctions, band labels, sigma_z, lattice metadata, and interaction metadata",
            )
        )
    else:
        keys, error = _npz_key_set(basis_path)
        if keys is None:
            blockers.append(f"could not inspect HTG primitive projected-basis archive {basis_path}: {error}")
        else:
            metadata["projected_basis_keys"] = sorted(keys)
            blockers.extend(_missing_key_blockers(basis_path, _HTG_PRIMITIVE_BASIS_CONTRACT_KEYS.difference(keys), role="HTGProjectedBasisData exact projected basis"))
    if not params_path.is_file() and not ("model_params" in metadata.get("projected_basis_keys", []) and "interaction_params" in metadata.get("projected_basis_keys", [])):
        blockers.append(_missing_file_blocker(params_path, role="HTG model/interaction/config metadata"))
    return tuple(blockers), metadata


def _htg_supercell_contract_blockers(root: Path, files: Mapping[str, Any], *, archive_name: str = "hf_supercell_ground_state.npz") -> tuple[tuple[str, ...], dict[str, object]]:
    state_path = _artifact_path(root, files, "hf_supercell_ground_state", archive_name)
    basis_path = _artifact_path(root, files, "projected_basis", "hf_supercell_projected_basis.npz")
    summary_path = _artifact_path(root, files, "summary", "summary.json")
    metadata: dict[str, object] = {
        "archive_format_contract": _contract_metadata(
            raw_object="mean_field.systems.htg.supercell.HTGSupercellHartreeFockRun",
            state_keys=_HTG_SUPERCELL_STATE_CONTRACT_KEYS,
            basis_keys=_HTG_SUPERCELL_BASIS_CONTRACT_KEYS,
        ),
        "expected_files": {
            "hf_supercell_ground_state": str(state_path),
            "hf_supercell_projected_basis": str(basis_path),
            "summary": str(summary_path),
        },
    }
    blockers: list[str] = []
    if not state_path.is_file():
        blockers.append(_missing_file_blocker(state_path, role="HTGSupercellHartreeFockState arrays and run history"))
    else:
        keys, error = _npz_key_set(state_path)
        if keys is None:
            blockers.append(f"could not inspect HTG supercell state archive {state_path}: {error}")
        else:
            metadata["hf_supercell_ground_state_keys"] = sorted(keys)
            blockers.extend(_missing_key_blockers(state_path, _HTG_SUPERCELL_STATE_CONTRACT_KEYS.difference(keys), role="HTGSupercellHartreeFockRun canonical adapter"))
    if not basis_path.is_file():
        blockers.append(
            _missing_file_blocker(
                basis_path,
                role="HTGSupercellProjectedBasisData.basis.wavefunctions, fold/band metadata, lattice metadata, and interaction metadata",
            )
        )
    else:
        keys, error = _npz_key_set(basis_path)
        if keys is None:
            blockers.append(f"could not inspect HTG supercell projected-basis archive {basis_path}: {error}")
        else:
            metadata["projected_basis_keys"] = sorted(keys)
            blockers.extend(_missing_key_blockers(basis_path, _HTG_SUPERCELL_BASIS_CONTRACT_KEYS.difference(keys), role="HTGSupercellProjectedBasisData exact projected basis"))
    if not summary_path.is_file() and not ("model_params" in metadata.get("projected_basis_keys", []) and "interaction_params" in metadata.get("projected_basis_keys", [])):
        blockers.append(_missing_file_blocker(summary_path, role="HTG supercell run summary/model metadata"))
    return tuple(blockers), metadata


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


def _tdbg_archive_candidates(root: Path, max_candidates: int) -> list[Path]:
    if not _path_text_mentions(root, "tdbg"):
        return []
    if root.is_file():
        return [root] if root.name == "hf_state.npz" else []
    if not root.exists():
        return []
    return list(sorted(root.rglob("hf_state.npz")))[:max_candidates]


def _htg_archive_candidates(root: Path, max_candidates: int) -> list[Path]:
    if not _path_text_mentions(root, "htg"):
        return []
    archive_names = {
        "hf_ground_state.npz",
        "hf_supercell_ground_state.npz",
        "hf_supercell_ground_state_best.npz",
        "hf_supercell_ground_state_best_copy_of_candidate.npz",
    }
    if root.is_file():
        return [root] if root.name in archive_names else []
    if not root.exists():
        return []
    candidates: list[Path] = []
    for name in sorted(archive_names):
        for archive_path in sorted(root.rglob(name)):
            if archive_path not in candidates:
                candidates.append(archive_path)
            if len(candidates) >= max_candidates:
                return candidates[:max_candidates]
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


def scan_backfill_candidates(
    roots: Sequence[str | Path],
    *,
    include_archives: bool = True,
    max_candidates: int = 10000,
) -> list[BackfillCandidate]:
    """Return dry-run canonical HF sidecar backfill candidates.

    The scan is metadata-only except for recognized TDBG/HTG/RLG-hBN NPZ
    header/scalar inspection.  It never writes into result directories.
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
            for archive_path in _tdbg_archive_candidates(root, max_candidates=max_candidates):
                resolved = archive_path.resolve()
                if resolved in seen_archives or archive_path.parent.resolve() in seen_manifest_roots:
                    continue
                seen_archives.add(resolved)
                records.append(_classify_tdbg_archive(archive_path))
            for archive_path in _htg_archive_candidates(root, max_candidates=max_candidates):
                resolved = archive_path.resolve()
                if resolved in seen_archives or archive_path.parent.resolve() in seen_manifest_roots:
                    continue
                seen_archives.add(resolved)
                records.append(_classify_htg_archive(archive_path))
            for archive_path in _rlg_hbn_archive_candidates(root, max_candidates=max_candidates):
                resolved = archive_path.resolve()
                if resolved in seen_archives or archive_path.parent.resolve() in seen_manifest_roots:
                    continue
                seen_archives.add(resolved)
                records.append(_classify_rlg_hbn_archive(archive_path))
    return records
