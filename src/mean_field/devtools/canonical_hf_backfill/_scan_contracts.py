from __future__ import annotations

from ._shared import *  # noqa: F401,F403
from ._scan_utils import *  # noqa: F401,F403

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

__all__ = [name for name in globals() if not name.startswith('__')]
