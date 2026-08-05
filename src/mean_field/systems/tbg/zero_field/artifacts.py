from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ....api.artifacts import ModelRecord, write_contract_artifacts
from ....core.hf import HFOverlapBlockSet
from ....core.io import write_npz_artifact
from ..params import TBGParameters
from ._hf_basis_overlap import (
    RestrictedHartreeFockState,
    TBGZeroFieldHFSourceReceipt,
    _tbg_zero_field_hf_history_sha256,
    _tbg_zero_field_hf_state_source_sha256,
    reciprocal_shift_labels,
    tbg_zero_field_active_shift_inventory_sha256,
    tbg_zero_field_lattice_kvec_sha256,
    tbg_zero_field_overlap_kernel_inventory_fingerprint,
    tbg_zero_field_reference_projector_sha256,
)
from .hf_contracts import validate_tbg_zero_field_typed_hf_run_source
from .interaction import (
    TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1,
    TBGZeroFieldInteractionSpec,
)
from .model import (
    TBG_ZERO_FIELD_B0_COORDINATE_CONVENTION,
    TBG_ZERO_FIELD_PHYSICAL_COORDINATE_CONVENTION,
    TBGZeroFieldTorusMesh,
    _b0_real_to_nm,
    _b0_reciprocal_to_nm_inv,
    tbg_zero_field_bm_generation_fingerprint,
)

_CONTRACT_FILENAMES = (
    "manifest.json",
    "model.json",
    "config.yaml",
    "conventions.json",
    "environment.json",
    "validation.json",
    "observables.json",
)
TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA = "mean_field.tbg.zero_field.validated_complete_hf_state_archive"
TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA_VERSION = 1
TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_FILENAME = "validated_complete_hf_state_archive.npz"
TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY = "validated_complete_hf_state_archive_npz"
TBG_ZERO_FIELD_STORED_DENSITY_DEFINITION = (
    "D_stored[a,b]=<c_a† c_b>-0.5*delta_ab"
)
_TYPED_B0_HF_SIDECAR_ARTIFACT_KEYS = frozenset(
    {
        "advisor_selection_txt",
        TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY,
        "path_limitation_txt",
        "runtime_summary_txt",
        "scf_band_plot_pdf",
        "scf_band_plot_png",
        "scf_path_tsv",
    }
)


def complex_to_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _resolve_artifact_input_path(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (root / path).resolve()


def _relative_artifact_files(root: Path, artifact_paths: Mapping[str, str | Path] | None) -> dict[str, str]:
    files: dict[str, str] = {}
    for key, raw_path in dict(artifact_paths or {}).items():
        path = Path(raw_path)
        if not path.is_absolute():
            files[str(key)] = path.as_posix()
            continue
        try:
            files[str(key)] = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            files[str(key)] = str(path)
    return files


def _ensure_contract_sidecars_absent(root: Path, *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [name for name in _CONTRACT_FILENAMES if (root / name).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing TBG zero-field contract sidecars in {root}: {existing}. "
            "Pass overwrite=True only when replacing this workflow's sidecars intentionally."
        )


def _dataclass_payload(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if is_dataclass(value):
        return dict(asdict(value))
    if isinstance(value, Mapping):
        return dict(value)
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def _json_safe(value: object) -> object:
    if isinstance(value, complex):
        return complex_to_pair(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _runtime_environment_payload(runtime: object) -> dict[str, object]:
    environment = getattr(runtime, "environment", None)
    payload = {str(key): _json_safe(value) for key, value in _dataclass_payload(environment).items()}
    for key in (
        "start_time",
        "end_time",
        "bm_elapsed_sec",
        "hf_elapsed_sec",
        "path_elapsed_sec",
        "grid_elapsed_sec",
        "total_elapsed_sec",
    ):
        if hasattr(runtime, key):
            payload[key] = _json_safe(getattr(runtime, key))
    return payload


def _tbg_model_record(params: TBGParameters, *, system_name: str = "tbg", extra: Mapping[str, object] | None = None) -> ModelRecord:
    param_payload = {
        "dtheta_rad": float(params.dtheta_rad),
        "convention": str(params.convention),
        "vf": float(params.vf),
        "chemical_potential": float(params.chemical_potential),
        "w0": float(params.w0),
        "w1": float(params.w1),
        "delta": float(params.delta),
        "strain": float(params.strain),
        "strain_angle_rad": float(params.strain_angle_rad),
        "poisson": float(params.poisson),
        "beta_g": float(params.beta_g),
        "alpha": float(params.alpha),
        "deformation_potential": float(params.deformation_potential),
    }
    param_payload.update(dict(extra or {}))
    lattice = {
        "coordinate_convention": TBG_ZERO_FIELD_B0_COORDINATE_CONVENTION,
        "physical_coordinate_convention": TBG_ZERO_FIELD_PHYSICAL_COORDINATE_CONVENTION,
        "graphene_a_nm": TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1,
        "g1_b0_code_pair": complex_to_pair(params.g1),
        "g2_b0_code_pair": complex_to_pair(params.g2),
        "a1_b0_code_pair": complex_to_pair(params.a1),
        "a2_b0_code_pair": complex_to_pair(params.a2),
        "kt_b0_code_pair": complex_to_pair(params.kt),
        "kb_point_b0_code_pair": complex_to_pair(params.kb_point),
        "g1_nm_inv_pair": complex_to_pair(_b0_reciprocal_to_nm_inv(params.g1)),
        "g2_nm_inv_pair": complex_to_pair(_b0_reciprocal_to_nm_inv(params.g2)),
        "a1_nm_pair": complex_to_pair(_b0_real_to_nm(params.a1)),
        "a2_nm_pair": complex_to_pair(_b0_real_to_nm(params.a2)),
        "theta12_rad": float(params.theta12),
        "kt_nm_inv_pair": complex_to_pair(_b0_reciprocal_to_nm_inv(params.kt)),
        "kb_point_nm_inv_pair": complex_to_pair(_b0_reciprocal_to_nm_inv(params.kb_point)),
    }
    return ModelRecord(system_name=system_name, params=param_payload, lattice=lattice)


def _bm_solution_summary(solution: object | None) -> dict[str, object] | None:
    if solution is None:
        return None
    spectrum = np.asarray(getattr(solution, "spectrum"), dtype=float)
    return {
        "lg": int(getattr(solution, "lg")),
        "nk": int(getattr(solution, "nk")),
        "nt": int(getattr(solution, "nt")),
        "n_eta": int(getattr(solution, "n_eta")),
        "n_spin": int(getattr(solution, "n_spin")),
        "nb": int(getattr(solution, "nb")),
        "periodic_g_grid": bool(getattr(solution, "periodic_g_grid", True)),
        "spectrum_shape": [int(value) for value in spectrum.shape],
        "energy_min_mev": float(np.min(spectrum)) if spectrum.size else float("nan"),
        "energy_max_mev": float(np.max(spectrum)) if spectrum.size else float("nan"),
    }


def _kpath_payload(path: object) -> dict[str, object]:
    kvec = np.asarray(getattr(path, "kvec"), dtype=np.complex128)
    kdist = np.asarray(getattr(path, "kdist"), dtype=float)
    return {
        "point_count": int(kvec.size),
        "labels": list(getattr(path, "labels")),
        "node_indices": [int(value) for value in getattr(path, "node_indices")],
        "kdist_min": float(np.min(kdist)) if kdist.size else 0.0,
        "kdist_max": float(np.max(kdist)) if kdist.size else 0.0,
    }


def _bm_unstrained_validation_payload(result: object) -> dict[str, object]:
    parity = getattr(result, "parity")
    runtime_parity = getattr(result, "runtime_parity", None)
    payload: dict[str, object] = {
        "status": "recorded",
        "parity": {
            "kdist_max_abs_diff": float(parity.kdist_max_abs_diff),
            "max_abs_band_diff_mev": float(parity.max_abs_band_diff_mev),
            "rms_band_diff_mev": float(parity.rms_band_diff_mev),
            "mean_abs_band_diff_mev": float(parity.mean_abs_band_diff_mev),
            "k_middle_gap_diff_mev": float(parity.k_middle_gap_diff_mev),
            "valence_bandwidth_diff_mev": _optional_float(parity.valence_bandwidth_diff_mev),
            "conduction_bandwidth_diff_mev": _optional_float(parity.conduction_bandwidth_diff_mev),
        },
    }
    if runtime_parity is not None:
        payload["runtime_parity"] = _json_safe(_dataclass_payload(runtime_parity))
    return payload


def _bm_unstrained_observables(result: object) -> dict[str, object]:
    run = getattr(result, "run")
    return {
        "theta_deg": float(getattr(getattr(result, "reference"), "theta_deg")),
        "k_middle_gap_mev": float(run.k_middle_gap_mev),
        "valence_bandwidth_mev": _optional_float(run.valence_bandwidth_mev),
        "conduction_bandwidth_mev": _optional_float(run.conduction_bandwidth_mev),
        "path": _kpath_payload(run.path),
        "path_solution": _bm_solution_summary(run.path_solution),
        "grid_solution": _bm_solution_summary(run.grid_solution),
    }


def _hf_state_shapes(state: object) -> dict[str, object]:
    return {
        "density": [int(value) for value in np.asarray(getattr(state, "density")).shape],
        "hamiltonian": [int(value) for value in np.asarray(getattr(state, "hamiltonian")).shape],
        "h0": [int(value) for value in np.asarray(getattr(state, "h0")).shape],
        "energies": [int(value) for value in np.asarray(getattr(state, "energies")).shape],
    }


def _diagnostics_payload(diagnostics: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in diagnostics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[str(key)] = _json_safe(value)
    return payload


def _is_typed_hf_run(hf_run: object) -> bool:
    state = getattr(hf_run, "state")
    receipt = getattr(state, "hf_source_receipt", None)
    return (
        getattr(state, "interaction_spec", None) is not None
        or getattr(hf_run, "screened_block_bundle", None) is not None
        or getattr(hf_run, "provenance", None) is not None
        or getattr(receipt, "interaction_contract", None) == "typed"
    )

def _validated_typed_b0_hf_source(result: object):
    hf_run = getattr(result, "hf_run")
    if not _is_typed_hf_run(hf_run):
        return None
    grid_solution = getattr(result, "grid_solution", None)
    if grid_solution is None:
        raise ValueError(
            "Typed TBG zero-field artifact export requires the exact grid_solution"
        )
    return validate_tbg_zero_field_typed_hf_run_source(hf_run, grid_solution)

@dataclass(frozen=True)
class _ExactSavedSCFPathCoverage:
    exact_point_count: int
    distinct_coordinate_count: int
    represented_segment_indices: tuple[int, ...]
    segment_distinct_coordinate_counts: tuple[int, ...]
    missing_interior_node_labels: tuple[str, ...]
    path_sample_indices: np.ndarray
    grid_indices: np.ndarray
    distance_to_path: np.ndarray
    meaningful: bool


def _exact_saved_scf_path_coverage(
    result: object,
    *,
    path_tolerance: float = 1.0e-12,
) -> _ExactSavedSCFPathCoverage:
    """Recompute the saved-grid-only path coverage directly from a result."""

    path = getattr(result, "path")
    grid_solution = getattr(result, "grid_solution")
    path_kvec = np.asarray(getattr(path, "kvec"), dtype=np.complex128).reshape(-1)
    grid_kvec = np.asarray(
        getattr(grid_solution, "lattice_kvec"), dtype=np.complex128
    ).reshape(-1)
    if path_kvec.size == 0 or grid_kvec.size == 0:
        raise ValueError("Exact saved-SCF path coverage requires non-empty path and grid coordinates")

    distance_matrix = np.abs(path_kvec[:, None] - grid_kvec[None, :])
    nearest_grid_indices = np.argmin(distance_matrix, axis=1).astype(int)
    nearest_distances = distance_matrix[
        np.arange(path_kvec.size), nearest_grid_indices
    ]
    path_sample_indices = np.flatnonzero(
        nearest_distances <= float(path_tolerance)
    ).astype(int)
    grid_indices = nearest_grid_indices[path_sample_indices].astype(int)
    exact_grid_kvec = np.asarray(grid_kvec[grid_indices], dtype=np.complex128)
    coordinate_keys = tuple(
        (float(value.real), float(value.imag)) for value in exact_grid_kvec
    )

    node_indices = tuple(int(index) - 1 for index in getattr(path, "node_indices"))
    segment_sets: list[set[tuple[float, float]]] = [
        set() for _ in range(max(len(node_indices) - 1, 0))
    ]
    for path_index, coordinate_key in zip(
        path_sample_indices, coordinate_keys, strict=True
    ):
        for segment_index, (start, stop) in enumerate(
            zip(node_indices[:-1], node_indices[1:], strict=True)
        ):
            if start <= int(path_index) <= stop:
                segment_sets[segment_index].add(coordinate_key)
    segment_counts = tuple(len(values) for values in segment_sets)
    represented_segments = tuple(
        index for index, count in enumerate(segment_counts) if count >= 2
    )
    selected_indices = set(int(value) for value in path_sample_indices)
    missing_interior_labels = tuple(
        str(label)
        for label, index in zip(
            getattr(path, "labels")[1:-1],
            node_indices[1:-1],
            strict=True,
        )
        if index not in selected_indices
    )
    distinct_count = len(set(coordinate_keys))
    meaningful = (
        distinct_count >= 3
        and len(represented_segments) >= 2
        and not missing_interior_labels
    )
    return _ExactSavedSCFPathCoverage(
        exact_point_count=int(path_sample_indices.size),
        distinct_coordinate_count=int(distinct_count),
        represented_segment_indices=represented_segments,
        segment_distinct_coordinate_counts=segment_counts,
        missing_interior_node_labels=missing_interior_labels,
        path_sample_indices=path_sample_indices,
        grid_indices=grid_indices,
        distance_to_path=np.asarray(
            nearest_distances[path_sample_indices], dtype=float
        ),
        meaningful=bool(meaningful),
    )


@dataclass(frozen=True)
class TBGZeroFieldValidatedCompleteHFStateArchive:
    """A validated complete HF state archive with no resume authority."""

    path: Path
    params: TBGParameters
    state: RestrictedHartreeFockState
    iter_energy: np.ndarray
    iter_err: np.ndarray
    iter_oda: np.ndarray
    mesh: TBGZeroFieldTorusMesh
    physical_kvec_nm_inv: np.ndarray
    bm_uk: np.ndarray
    bm_spectrum: np.ndarray
    bm_gvec: np.ndarray
    screened_blocks: HFOverlapBlockSet
    interaction_spec: TBGZeroFieldInteractionSpec
    receipt: TBGZeroFieldHFSourceReceipt
    bundle_metadata: Mapping[str, object]
    source_attestation_metadata: Mapping[str, object]
    provenance_metadata: Mapping[str, object]

_COMPLETE_HF_STATE_ARCHIVE_ARRAY_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "density_delta_definition",
        "params_json",
        "interaction_spec_json",
        "hf_source_receipt_json",
        "screened_block_bundle_json",
        "bm_source_attestation_json",
        "hf_run_provenance_json",
        "mesh_json",
        "state_diagnostics_json",
        "bm_solution_sha256",
        "bm_generation_fingerprint",
        "bm_source_attestation_fingerprint",
        "bm_hamiltonian_sha256",
        "bm_sigma_z_sha256",
        "bm_uk_sha256",
        "bm_spectrum_sha256",
        "bm_gvec_sha256",
        "bm_kvec_sha256",
        "screened_block_bundle_sha256",
        "hf_source_receipt_sha256",
        "hf_state_source_sha256",
        "state_h0",
        "state_sigma_z",
        "state_density",
        "state_hamiltonian",
        "state_energies",
        "state_sigma_ztauz",
        "state_mu",
        "state_nu",
        "state_v0",
        "state_precision",
        "state_dimensions",
        "iter_energy",
        "iter_err",
        "iter_oda",
        "mesh_k_grid_frac",
        "mesh_kvec_b0",
        "mesh_kvec_nm_inv",
        "bm_uk",
        "bm_spectrum",
        "bm_gvec",
        "screened_shifts",
        "screened_gvecs",
        "screened_overlaps",
        "screened_active_shifts",
        "screened_diagonal_overlaps",
        "screened_hartree_screening",
        "screened_fock_screening",
    }
)


def _strict_json_text(value: object) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json_payload(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256_text(value: object, *, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    return digest


def _canonical_source_array_sha256(values: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _update_labeled_bytes(digest: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, byteorder="little", signed=False))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
    digest.update(payload)


def _archived_bm_solution_sha256(
    *,
    mesh_fingerprint: str,
    source_attestation_fingerprint: str,
    generation_fingerprint: str,
) -> str:
    digest = hashlib.sha256()
    _update_labeled_bytes(digest, "domain", b"TBGZeroFieldBMSolution/v3")
    _update_labeled_bytes(digest, "mesh_fingerprint", mesh_fingerprint.encode("ascii"))
    _update_labeled_bytes(
        digest,
        "source_attestation_fingerprint",
        source_attestation_fingerprint.encode("ascii"),
    )
    _update_labeled_bytes(
        digest,
        "bm_generation_fingerprint",
        generation_fingerprint.encode("ascii"),
    )
    return digest.hexdigest()


def _npz_scalar(arrays: Mapping[str, np.ndarray], key: str) -> object:
    value = np.asarray(arrays[key])
    if value.shape != ():
        raise ValueError(f"Validated complete HF state archive field {key!r} must be a scalar, got {value.shape}")
    return value.item()


def _npz_json_mapping(arrays: Mapping[str, np.ndarray], key: str) -> dict[str, object]:
    raw = _npz_scalar(arrays, key)
    if not isinstance(raw, str):
        raise ValueError(f"Validated complete HF state archive field {key!r} must be a JSON string")
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError(f"Validated complete HF state archive field {key!r} must encode a JSON object")
    return dict(decoded)


def _source_array_record(
    source_metadata: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    sources = source_metadata.get("array_sources")
    if not isinstance(sources, Mapping):
        raise ValueError("BM source attestation array_sources must be a mapping")
    record = sources.get(name)
    if not isinstance(record, Mapping) or set(record) != {"shape", "sha256"}:
        raise ValueError(f"BM source attestation array record {name!r} is malformed")
    shape = record.get("shape")
    if not isinstance(shape, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in shape
    ):
        raise ValueError(f"BM source attestation shape for {name!r} is malformed")
    return {"shape": list(shape), "sha256": _validate_sha256_text(record.get("sha256"), name=f"{name}.sha256")}


def _validate_archived_source_array(
    arrays: Mapping[str, np.ndarray],
    source_metadata: Mapping[str, object],
    *,
    source_name: str,
    archive_key: str,
    dtype: str,
) -> None:
    record = _source_array_record(source_metadata, source_name)
    values = np.asarray(arrays[archive_key])
    if list(values.shape) != record["shape"]:
        raise ValueError(
            f"Archived BM {source_name} shape does not match source attestation"
        )
    actual = _canonical_source_array_sha256(values, dtype=dtype)
    if actual != record["sha256"]:
        raise ValueError(
            f"Archived BM {source_name} hash does not match source attestation"
        )
    explicit = _validate_sha256_text(
        _npz_scalar(arrays, f"bm_{source_name}_sha256"),
        name=f"bm_{source_name}_sha256",
    )
    if explicit != actual:
        raise ValueError(f"Archived explicit BM {source_name} hash is inconsistent")


def _validate_source_attestation_metadata(
    source_metadata: Mapping[str, object],
) -> str:
    expected_keys = {
        "array_sources",
        "calculate_chern_operator",
        "dimensions",
        "fingerprint",
        "params_independent_fingerprint",
        "periodic_g_grid",
        "sigma_rotation",
        "solver_entrypoint",
        "solver_schema",
        "solver_schema_version",
        "torus_mesh_fingerprint",
    }
    if set(source_metadata) != expected_keys:
        raise ValueError("BM source attestation metadata keys differ from schema-v1")
    dimensions = source_metadata.get("dimensions")
    expected_dimension_keys = {"lg", "n_eta", "n_spin", "nb", "nk", "nlocal"}
    if not isinstance(dimensions, Mapping) or set(dimensions) != expected_dimension_keys:
        raise ValueError("BM source attestation dimensions differ from schema-v1")
    resolved_dimensions: dict[str, int] = {}
    for name in expected_dimension_keys:
        value = dimensions[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"BM source attestation dimension {name!r} must be positive")
        resolved_dimensions[name] = int(value)
    for name in ("calculate_chern_operator", "periodic_g_grid", "sigma_rotation"):
        if not isinstance(source_metadata.get(name), bool):
            raise ValueError(f"BM source attestation flag {name!r} must be bool")
    if source_metadata.get("solver_entrypoint") != "solve_bm_model_on_torus":
        raise ValueError("Validated complete HF state archive requires the torus BM solver entrypoint")
    if source_metadata.get("solver_schema") != "mean_field.tbg.zero_field.bm_solver":
        raise ValueError("Unsupported BM source-attestation schema")
    if source_metadata.get("solver_schema_version") != 1:
        raise ValueError("Unsupported BM source-attestation schema version")
    records = {
        name: _source_array_record(source_metadata, name)
        for name in ("gvec", "hamiltonian", "kvec", "sigma_z", "spectrum", "uk")
    }
    dim = resolved_dimensions["nlocal"] * resolved_dimensions["lg"] ** 2
    nt = (
        resolved_dimensions["n_spin"]
        * resolved_dimensions["n_eta"]
        * resolved_dimensions["nb"]
    )
    expected_shapes = {
        "gvec": [resolved_dimensions["lg"] ** 2],
        "hamiltonian": [dim, dim, resolved_dimensions["n_eta"], resolved_dimensions["nk"]],
        "kvec": [resolved_dimensions["nk"]],
        "sigma_z": [nt, nt, resolved_dimensions["nk"]],
        "spectrum": [resolved_dimensions["nb"], resolved_dimensions["n_eta"], resolved_dimensions["nk"]],
        "uk": [dim, resolved_dimensions["nb"], resolved_dimensions["n_eta"], resolved_dimensions["nk"]],
    }
    if any(records[name]["shape"] != shape for name, shape in expected_shapes.items()):
        raise ValueError("BM source-attestation array shapes do not match dimensions")
    supplied = _validate_sha256_text(
        source_metadata.get("fingerprint"),
        name="bm_source_attestation.fingerprint",
    )
    payload = dict(source_metadata)
    payload.pop("fingerprint")
    if _sha256_json_payload(payload) != supplied:
        raise ValueError("BM source attestation fingerprint does not match its metadata")
    return supplied


def _validate_bundle_metadata(bundle_metadata: Mapping[str, object]) -> str:
    expected_keys = {
        "active_shift_inventory",
        "active_shift_inventory_sha256",
        "bm_generation_fingerprint",
        "bm_solution_sha256",
        "companion_circular_total_q_cutoff_parity",
        "fingerprint",
        "interaction_spec_fingerprint",
        "mesh_fingerprint",
        "n_band",
        "overlap_kernel_inventory_sha256",
        "overlap_lg",
        "reference_projector_convention",
        "reference_projector_dimensions",
        "reference_projector_sha256",
        "schema",
        "schema_version",
        "transfer_cutoff_policy",
    }
    if set(bundle_metadata) != expected_keys:
        raise ValueError("Screened-block bundle metadata keys differ from schema-v1")
    supplied = _validate_sha256_text(
        bundle_metadata.get("fingerprint"),
        name="screened_block_bundle.fingerprint",
    )
    payload = dict(bundle_metadata)
    payload.pop("fingerprint")
    if _sha256_json_payload(payload) != supplied:
        raise ValueError("Screened-block bundle fingerprint does not match its metadata")
    return supplied


def _validate_provenance_metadata(provenance: Mapping[str, object]) -> str:
    expected_keys = {
        "beta",
        "bm_generation_fingerprint",
        "converged",
        "exit_reason",
        "filling",
        "fingerprint",
        "hf_mode",
        "issuer",
        "interaction_spec_fingerprint",
        "iter_energy_sha256",
        "iter_err_sha256",
        "iter_oda_sha256",
        "mesh_fingerprint",
        "normalized_init_mode",
        "nu",
        "oda_stall_threshold",
        "precision",
        "requested_max_iterations",
        "schema",
        "schema_version",
        "seed",
        "state_source_sha256",
        "typed_receipt_fingerprint",
    }
    if set(provenance) != expected_keys:
        raise ValueError("Typed HF run provenance metadata keys differ from schema-v1")
    supplied = _validate_sha256_text(
        provenance.get("fingerprint"),
        name="hf_run_provenance.fingerprint",
    )
    payload = dict(provenance)
    payload.pop("fingerprint")
    if _sha256_json_payload(payload) != supplied:
        raise ValueError("Typed HF run provenance fingerprint does not match its metadata")
    return supplied


def load_tbg_zero_field_complete_hf_state_archive_npz(
    path: str | Path,
) -> TBGZeroFieldValidatedCompleteHFStateArchive:
    """Validate and reconstruct a complete HF state archive without resume authority."""

    archive_path = Path(path)
    with np.load(archive_path, allow_pickle=False) as payload:
        if set(payload.files) != _COMPLETE_HF_STATE_ARCHIVE_ARRAY_KEYS:
            missing = sorted(_COMPLETE_HF_STATE_ARCHIVE_ARRAY_KEYS - set(payload.files))
            extra = sorted(set(payload.files) - _COMPLETE_HF_STATE_ARCHIVE_ARRAY_KEYS)
            raise ValueError(
                "Validated complete HF state archive NPZ keys differ from schema-v1: "
                f"missing={missing}, extra={extra}"
            )
        arrays = {str(key): np.array(payload[key], copy=True) for key in payload.files}

    complex_keys = {
        "bm_gvec",
        "bm_uk",
        "mesh_kvec_b0",
        "mesh_kvec_nm_inv",
        "screened_diagonal_overlaps",
        "screened_gvecs",
        "screened_overlaps",
        "state_density",
        "state_h0",
        "state_hamiltonian",
        "state_sigma_z",
    }
    float_keys = {
        "bm_spectrum",
        "iter_energy",
        "iter_err",
        "iter_oda",
        "mesh_k_grid_frac",
        "screened_fock_screening",
        "screened_hartree_screening",
        "state_energies",
        "state_mu",
        "state_nu",
        "state_precision",
        "state_sigma_ztauz",
        "state_v0",
    }
    int_keys = {
        "schema_version",
        "screened_active_shifts",
        "screened_shifts",
        "state_dimensions",
    }
    for key, expected_dtype in (
        *((key, np.dtype("<c16")) for key in complex_keys),
        *((key, np.dtype("<f8")) for key in float_keys),
        *((key, np.dtype("<i8")) for key in int_keys),
    ):
        if arrays[key].dtype != expected_dtype:
            raise ValueError(
                f"Validated complete HF state archive field {key!r} has dtype {arrays[key].dtype}, "
                f"expected {expected_dtype}"
            )

    if _npz_scalar(arrays, "schema") != TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA:
        raise ValueError("Unsupported typed TBG zero-field complete HF state archive schema")
    if int(_npz_scalar(arrays, "schema_version")) != TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA_VERSION:
        raise ValueError("Unsupported typed TBG zero-field complete HF state archive schema version")
    if _npz_scalar(arrays, "density_delta_definition") != TBG_ZERO_FIELD_STORED_DENSITY_DEFINITION:
        raise ValueError("Validated complete HF state archive density convention is not the stored B0 convention")

    params_payload = _npz_json_mapping(arrays, "params_json")
    params = TBGParameters(**params_payload)  # type: ignore[arg-type]
    interaction_metadata = _npz_json_mapping(arrays, "interaction_spec_json")
    interaction_spec = TBGZeroFieldInteractionSpec.from_metadata(interaction_metadata)
    receipt_metadata = _npz_json_mapping(arrays, "hf_source_receipt_json")
    receipt = TBGZeroFieldHFSourceReceipt.from_metadata(receipt_metadata)
    bundle_metadata = _npz_json_mapping(arrays, "screened_block_bundle_json")
    bundle_fingerprint = _validate_bundle_metadata(bundle_metadata)
    source_metadata = _npz_json_mapping(arrays, "bm_source_attestation_json")
    source_fingerprint = _validate_source_attestation_metadata(source_metadata)
    provenance_metadata = _npz_json_mapping(arrays, "hf_run_provenance_json")
    _validate_provenance_metadata(provenance_metadata)
    mesh_metadata = _npz_json_mapping(arrays, "mesh_json")
    diagnostics = _npz_json_mapping(arrays, "state_diagnostics_json")

    mesh_shape = mesh_metadata.get("mesh_shape")
    g1_pair = mesh_metadata.get("g1")
    g2_pair = mesh_metadata.get("g2")
    if (
        not isinstance(mesh_shape, list)
        or len(mesh_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in mesh_shape
        )
        or not isinstance(g1_pair, list)
        or len(g1_pair) != 2
        or not isinstance(g2_pair, list)
        or len(g2_pair) != 2
    ):
        raise ValueError("Validated complete HF state archive mesh metadata is malformed")
    n1, n2 = int(mesh_shape[0]), int(mesh_shape[1])
    mesh = TBGZeroFieldTorusMesh(
        mesh_size=n1 if n1 == n2 else (n1, n2),
        g1=complex(float(g1_pair[0]), float(g1_pair[1])),
        g2=complex(float(g2_pair[0]), float(g2_pair[1])),
        k_grid_frac=np.asarray(arrays["mesh_k_grid_frac"], dtype=np.float64),
        kvec=np.asarray(arrays["mesh_kvec_b0"], dtype=np.complex128),
        schema=str(mesh_metadata.get("schema")),
        schema_version=int(mesh_metadata.get("schema_version", -1)),
        index_order=str(mesh_metadata.get("index_order")),
        fractional_domain=str(mesh_metadata.get("fractional_domain")),
    )
    if mesh.to_metadata() != mesh_metadata:
        raise ValueError("Validated complete HF state archive mesh metadata does not match archived arrays")
    if mesh.g1 != params.g1 or mesh.g2 != params.g2:
        raise ValueError("Validated complete HF state archive mesh reciprocal vectors do not match parameters")
    physical_kvec = np.asarray(arrays["mesh_kvec_nm_inv"], dtype=np.complex128)
    expected_physical_kvec = mesh.kvec / TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1
    if not np.array_equal(physical_kvec, expected_physical_kvec):
        raise ValueError("Validated complete HF state archive physical k-vectors do not match raw B0 k-vectors")

    dimensions = source_metadata.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("BM source attestation dimensions must be a mapping")
    if source_metadata.get("params_independent_fingerprint") != params.independent_fingerprint:
        raise ValueError("Archived TBG parameters do not match BM source attestation")
    if source_metadata.get("torus_mesh_fingerprint") != mesh.fingerprint:
        raise ValueError("Archived mesh does not match BM source attestation")
    generation_fingerprint = tbg_zero_field_bm_generation_fingerprint(
        params,
        lg=int(dimensions.get("lg", 0)),
        periodic_g_grid=bool(source_metadata.get("periodic_g_grid")),
        sigma_rotation=bool(source_metadata.get("sigma_rotation")),
        calculate_chern_operator=bool(source_metadata.get("calculate_chern_operator")),
        torus_mesh_fingerprint=mesh.fingerprint,
    )
    if generation_fingerprint != _npz_scalar(arrays, "bm_generation_fingerprint"):
        raise ValueError("Archived BM generation fingerprint is inconsistent")
    if generation_fingerprint != bundle_metadata.get("bm_generation_fingerprint"):
        raise ValueError("Archived BM generation does not match screened bundle")

    _validate_archived_source_array(
        arrays,
        source_metadata,
        source_name="sigma_z",
        archive_key="state_sigma_z",
        dtype="<c16",
    )
    _validate_archived_source_array(
        arrays,
        source_metadata,
        source_name="uk",
        archive_key="bm_uk",
        dtype="<c16",
    )
    _validate_archived_source_array(
        arrays,
        source_metadata,
        source_name="spectrum",
        archive_key="bm_spectrum",
        dtype="<f8",
    )
    _validate_archived_source_array(
        arrays,
        source_metadata,
        source_name="gvec",
        archive_key="bm_gvec",
        dtype="<c16",
    )
    _validate_archived_source_array(
        arrays,
        source_metadata,
        source_name="kvec",
        archive_key="mesh_kvec_b0",
        dtype="<c16",
    )
    hamiltonian_record = _source_array_record(source_metadata, "hamiltonian")
    if _validate_sha256_text(
        _npz_scalar(arrays, "bm_hamiltonian_sha256"),
        name="bm_hamiltonian_sha256",
    ) != hamiltonian_record["sha256"]:
        raise ValueError("Archived BM Hamiltonian source hash is inconsistent")
    if source_fingerprint != _npz_scalar(arrays, "bm_source_attestation_fingerprint"):
        raise ValueError("Archived BM source-attestation fingerprint is inconsistent")

    bm_solution_sha256 = _archived_bm_solution_sha256(
        mesh_fingerprint=mesh.fingerprint,
        source_attestation_fingerprint=source_fingerprint,
        generation_fingerprint=generation_fingerprint,
    )
    if bm_solution_sha256 != _npz_scalar(arrays, "bm_solution_sha256"):
        raise ValueError("Archived BM solution hash chain is inconsistent")
    if bm_solution_sha256 != bundle_metadata.get("bm_solution_sha256"):
        raise ValueError("Archived BM source does not match screened bundle")

    state_dimensions = np.asarray(arrays["state_dimensions"])
    if state_dimensions.shape != (3,) or state_dimensions.dtype != np.dtype("<i8"):
        raise ValueError("Validated complete HF state archive state_dimensions must be canonical int64[3]")
    n_spin, n_eta, n_band = (int(value) for value in state_dimensions)
    state = RestrictedHartreeFockState(
        h0=np.asarray(arrays["state_h0"], dtype=np.complex128),
        sigma_z=np.asarray(arrays["state_sigma_z"], dtype=np.complex128),
        density=np.asarray(arrays["state_density"], dtype=np.complex128),
        hamiltonian=np.asarray(arrays["state_hamiltonian"], dtype=np.complex128),
        energies=np.asarray(arrays["state_energies"], dtype=np.float64),
        sigma_ztauz=np.asarray(arrays["state_sigma_ztauz"], dtype=np.float64),
        mu=float(_npz_scalar(arrays, "state_mu")),
        nu=float(_npz_scalar(arrays, "state_nu")),
        v0=float(_npz_scalar(arrays, "state_v0")),
        precision=float(_npz_scalar(arrays, "state_precision")),
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
        diagnostics={str(key): value for key, value in diagnostics.items()},
        hf_source_receipt=receipt,
        interaction_spec=interaction_spec,
    )
    expected_nt = n_spin * n_eta * n_band
    expected_matrix_shape = (expected_nt, expected_nt, mesh.nk)
    if state.nk != mesh.nk or state.nt != expected_nt:
        raise ValueError("Validated complete HF state archive state and half-open mesh dimensions differ")
    for name, values in (
        ("h0", state.h0),
        ("sigma_z", state.sigma_z),
        ("density", state.density),
        ("hamiltonian", state.hamiltonian),
    ):
        if np.asarray(values).shape != expected_matrix_shape:
            raise ValueError(f"Validated complete HF state archive {name} shape is inconsistent")
    if state.energies.shape != (expected_nt, mesh.nk):
        raise ValueError("Validated complete HF state archive energies shape is inconsistent")
    if state.sigma_ztauz.shape != (expected_nt, mesh.nk):
        raise ValueError("Validated complete HF state archive sigma_ztauz shape is inconsistent")
    if (
        int(dimensions.get("n_spin", 0)),
        int(dimensions.get("n_eta", 0)),
        int(dimensions.get("nb", 0)),
        int(dimensions.get("nk", 0)),
    ) != (n_spin, n_eta, n_band, mesh.nk):
        raise ValueError("Validated complete HF state archive state dimensions do not match BM source")
    bm_spectrum = np.asarray(arrays["bm_spectrum"], dtype=np.float64)
    if bm_spectrum.shape != (n_band, n_eta, state.nk):
        raise ValueError("Validated complete HF state archive BM spectrum shape does not match state dimensions")
    flattened = np.zeros((state.nt, state.nk), dtype=np.float64)
    row = 0
    for ib in range(n_band):
        for ieta in range(n_eta):
            for _ispin in range(n_spin):
                flattened[row, :] = bm_spectrum[ib, ieta, :]
                row += 1
    expected_h0 = np.zeros_like(state.h0)
    for ik in range(state.nk):
        np.fill_diagonal(expected_h0[:, :, ik], flattened[:, ik])
    if not np.array_equal(state.h0, expected_h0):
        raise ValueError("Validated complete HF state archive h0 does not match archived BM spectrum")

    shifts_array = np.asarray(arrays["screened_shifts"])
    active_array = np.asarray(arrays["screened_active_shifts"])
    if shifts_array.ndim != 2 or shifts_array.shape[1:] != (2,) or shifts_array.dtype != np.dtype("<i8"):
        raise ValueError("Validated complete HF state archive screened_shifts must be canonical int64[:,2]")
    if active_array.ndim != 2 or active_array.shape[1:] != (2,) or active_array.dtype != np.dtype("<i8"):
        raise ValueError("Validated complete HF state archive screened_active_shifts must be canonical int64[:,2]")
    shifts = tuple((int(value[0]), int(value[1])) for value in shifts_array)
    active_shifts = tuple((int(value[0]), int(value[1])) for value in active_array)
    labels = reciprocal_shift_labels(int(bundle_metadata.get("overlap_lg", 0)))
    expected_shifts = tuple((m, n) for n in labels for m in labels)
    if shifts != expected_shifts:
        raise ValueError("Validated complete HF state archive ordered screened shifts do not match overlap_lg")
    metadata_active = tuple(
        (int(value[0]), int(value[1]))
        for value in bundle_metadata.get("active_shift_inventory", [])  # type: ignore[union-attr]
    )
    if active_shifts != metadata_active:
        raise ValueError("Validated complete HF state archive active shifts do not match bundle metadata")
    overlaps_array = np.asarray(arrays["screened_overlaps"], dtype=np.complex128)
    diagonal_array = np.asarray(arrays["screened_diagonal_overlaps"], dtype=np.complex128)
    hartree_array = np.asarray(arrays["screened_hartree_screening"], dtype=np.float64)
    fock_array = np.asarray(arrays["screened_fock_screening"], dtype=np.float64)
    if overlaps_array.shape[0] != len(shifts):
        raise ValueError("Validated complete HF state archive overlap stack does not match ordered shifts")
    if any(values.shape[0] != len(active_shifts) for values in (diagonal_array, hartree_array, fock_array)):
        raise ValueError("Validated complete HF state archive screened active arrays do not match active shifts")
    screened_blocks = HFOverlapBlockSet(
        shifts=shifts,
        gvecs=np.asarray(arrays["screened_gvecs"], dtype=np.complex128),
        overlaps={shift: overlaps_array[index] for index, shift in enumerate(shifts)},
        diagonal_overlaps={
            shift: diagonal_array[index] for index, shift in enumerate(active_shifts)
        },
        hartree_screening={
            shift: float(hartree_array[index]) for index, shift in enumerate(active_shifts)
        },
        fock_screening={
            shift: fock_array[index] for index, shift in enumerate(active_shifts)
        },
    )
    expected_screened_gvecs = np.asarray(
        [m * params.g1 + n * params.g2 for m, n in shifts],
        dtype=np.complex128,
    )
    if not np.array_equal(screened_blocks.gvecs, expected_screened_gvecs):
        raise ValueError("Validated complete HF state archive screened g-vectors do not match shifts/parameters")
    kernel_fingerprint = tbg_zero_field_overlap_kernel_inventory_fingerprint(screened_blocks)
    if kernel_fingerprint != bundle_metadata.get("overlap_kernel_inventory_sha256"):
        raise ValueError("Validated complete HF state archive screened block inventory hash mismatch")
    active_hash = tbg_zero_field_active_shift_inventory_sha256(active_shifts)
    if active_hash != bundle_metadata.get("active_shift_inventory_sha256"):
        raise ValueError("Validated complete HF state archive active shift inventory hash mismatch")
    reference = np.repeat(
        (0.5 * np.eye(state.nt, dtype=np.complex128))[:, :, None],
        state.nk,
        axis=2,
    )
    if tbg_zero_field_reference_projector_sha256(reference) != bundle_metadata.get(
        "reference_projector_sha256"
    ):
        raise ValueError("Validated complete HF state archive reference-projector hash mismatch")

    if bundle_metadata.get("interaction_spec_fingerprint") != interaction_spec.fingerprint:
        raise ValueError("Archived screened bundle does not match interaction specification")
    if bundle_metadata.get("mesh_fingerprint") != mesh.fingerprint:
        raise ValueError("Archived screened bundle does not match half-open mesh")
    if int(bundle_metadata.get("n_band", 0)) != n_band:
        raise ValueError("Archived screened bundle does not match active band count")
    if tuple(bundle_metadata.get("reference_projector_dimensions", ())) != tuple(reference.shape):
        raise ValueError("Archived screened bundle reference dimensions are inconsistent")
    if bundle_fingerprint != _npz_scalar(arrays, "screened_block_bundle_sha256"):
        raise ValueError("Archived screened-bundle fingerprint is inconsistent")
    if receipt.screened_block_bundle_sha256 != bundle_fingerprint:
        raise ValueError("Archived HF receipt does not match screened bundle")
    if receipt.fingerprint != _npz_scalar(arrays, "hf_source_receipt_sha256"):
        raise ValueError("Archived HF receipt fingerprint is inconsistent")
    if receipt.interaction_spec_fingerprint != interaction_spec.fingerprint:
        raise ValueError("Archived HF receipt does not match interaction specification")
    if receipt.bm_solution_sha256 != bm_solution_sha256:
        raise ValueError("Archived HF receipt does not match BM source")
    if receipt.bm_generation_fingerprint != generation_fingerprint:
        raise ValueError("Archived HF receipt does not match BM generation")
    if receipt.active_shift_inventory != active_shifts:
        raise ValueError("Archived HF receipt does not match active screened shifts")
    if receipt.active_shift_inventory_sha256 != active_hash:
        raise ValueError("Archived HF receipt active-shift hash is inconsistent")
    if receipt.reference_projector_dimensions != tuple(reference.shape):
        raise ValueError("Archived HF receipt reference dimensions are inconsistent")
    if receipt.reference_projector_sha256 != bundle_metadata.get("reference_projector_sha256"):
        raise ValueError("Archived HF receipt reference hash is inconsistent")
    if receipt.mesh_fingerprint != mesh.fingerprint:
        raise ValueError("Archived HF receipt does not match half-open mesh")
    if receipt.lattice_kvec_sha256 != tbg_zero_field_lattice_kvec_sha256(mesh.kvec):
        raise ValueError("Archived HF receipt lattice hash does not match mesh")
    if receipt.overlap_kernel_inventory_sha256 != kernel_fingerprint:
        raise ValueError("Archived HF receipt does not match screened block arrays")
    if receipt.v0 != state.v0:
        raise ValueError("Archived HF receipt does not match final Coulomb unit")

    if provenance_metadata.get("interaction_spec_fingerprint") != interaction_spec.fingerprint:
        raise ValueError("Validated complete HF state archive provenance does not match interaction specification")
    if provenance_metadata.get("bm_generation_fingerprint") != generation_fingerprint:
        raise ValueError("Validated complete HF state archive provenance does not match BM generation")
    if provenance_metadata.get("mesh_fingerprint") != mesh.fingerprint:
        raise ValueError("Validated complete HF state archive provenance does not match half-open mesh")
    if float(provenance_metadata.get("beta", np.nan)) != receipt.beta:
        raise ValueError("Validated complete HF state archive provenance does not match receipt beta")

    iter_energy = np.asarray(arrays["iter_energy"], dtype=np.float64)
    iter_err = np.asarray(arrays["iter_err"], dtype=np.float64)
    iter_oda = np.asarray(arrays["iter_oda"], dtype=np.float64)
    for name, values in (
        ("iter_energy", iter_energy),
        ("iter_err", iter_err),
        ("iter_oda", iter_oda),
    ):
        actual = _tbg_zero_field_hf_history_sha256(values, name=name)
        if actual != provenance_metadata.get(f"{name}_sha256"):
            raise ValueError(f"Validated complete HF state archive {name} hash mismatch")
    state_hash = _tbg_zero_field_hf_state_source_sha256(state)
    if state_hash != provenance_metadata.get("state_source_sha256"):
        raise ValueError("Validated complete HF state archive final-state hash mismatch")
    if state_hash != _npz_scalar(arrays, "hf_state_source_sha256"):
        raise ValueError("Validated complete HF state archive explicit final-state hash is inconsistent")
    if receipt.fingerprint != provenance_metadata.get("typed_receipt_fingerprint"):
        raise ValueError("Validated complete HF state archive provenance does not match source receipt")
    if float(provenance_metadata.get("nu", np.nan)) != state.nu:
        raise ValueError("Validated complete HF state archive provenance does not match final filling")
    if float(provenance_metadata.get("precision", np.nan)) != state.precision:
        raise ValueError("Validated complete HF state archive provenance does not match final precision")

    return TBGZeroFieldValidatedCompleteHFStateArchive(
        path=archive_path,
        params=params,
        state=state,
        iter_energy=iter_energy,
        iter_err=iter_err,
        iter_oda=iter_oda,
        mesh=mesh,
        physical_kvec_nm_inv=physical_kvec,
        bm_uk=np.asarray(arrays["bm_uk"], dtype=np.complex128),
        bm_spectrum=bm_spectrum,
        bm_gvec=np.asarray(arrays["bm_gvec"], dtype=np.complex128),
        screened_blocks=screened_blocks,
        interaction_spec=interaction_spec,
        receipt=receipt,
        bundle_metadata=bundle_metadata,
        source_attestation_metadata=source_metadata,
        provenance_metadata=provenance_metadata,
    )


def write_tbg_zero_field_complete_hf_state_archive_npz(
    path: str | Path,
    result: object,
) -> Path:
    """Atomically publish a validated complete HF state archive/checkpoint; no resume authority is implied."""

    typed_source = _validated_typed_b0_hf_source(result)
    if typed_source is None:
        raise ValueError("Complete TBG zero-field HF state archive writer rejects legacy/untyped runs")
    interaction_spec, bundle, receipt = typed_source
    hf_run = getattr(result, "hf_run")
    state = hf_run.state
    provenance = hf_run.provenance
    grid_solution = getattr(result, "grid_solution")
    mesh = grid_solution.torus_mesh
    source_attestation = grid_solution.source_attestation
    if not isinstance(mesh, TBGZeroFieldTorusMesh) or source_attestation is None or provenance is None:
        raise ValueError("Complete TBG zero-field HF state archive requires mesh, source attestation, and provenance")

    blocks = bundle.screened_blocks
    shifts = tuple(blocks.shifts)
    active_shifts = tuple(bundle.active_shifts)
    source_metadata = source_attestation.to_metadata()
    source_records = source_metadata["array_sources"]
    if not isinstance(source_records, Mapping):
        raise ValueError("Typed BM source attestation array_sources is malformed")
    arrays = {
        "schema": np.asarray(TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA),
        "schema_version": np.asarray(TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA_VERSION, dtype="<i8"),
        "density_delta_definition": np.asarray(TBG_ZERO_FIELD_STORED_DENSITY_DEFINITION),
        "params_json": np.asarray(_strict_json_text(grid_solution.params.to_summary_dict())),
        "interaction_spec_json": np.asarray(_strict_json_text(interaction_spec.to_metadata())),
        "hf_source_receipt_json": np.asarray(_strict_json_text(receipt.to_metadata())),
        "screened_block_bundle_json": np.asarray(_strict_json_text(bundle.to_metadata())),
        "bm_source_attestation_json": np.asarray(_strict_json_text(source_metadata)),
        "hf_run_provenance_json": np.asarray(_strict_json_text(provenance.to_metadata())),
        "mesh_json": np.asarray(_strict_json_text(mesh.to_metadata())),
        "state_diagnostics_json": np.asarray(_strict_json_text(state.diagnostics)),
        "bm_solution_sha256": np.asarray(bundle.bm_solution_sha256),
        "bm_generation_fingerprint": np.asarray(bundle.bm_generation_fingerprint),
        "bm_source_attestation_fingerprint": np.asarray(source_attestation.fingerprint),
        "bm_hamiltonian_sha256": np.asarray(source_records["hamiltonian"]["sha256"]),
        "bm_sigma_z_sha256": np.asarray(source_records["sigma_z"]["sha256"]),
        "bm_uk_sha256": np.asarray(source_records["uk"]["sha256"]),
        "bm_spectrum_sha256": np.asarray(source_records["spectrum"]["sha256"]),
        "bm_gvec_sha256": np.asarray(source_records["gvec"]["sha256"]),
        "bm_kvec_sha256": np.asarray(source_records["kvec"]["sha256"]),
        "screened_block_bundle_sha256": np.asarray(bundle.fingerprint),
        "hf_source_receipt_sha256": np.asarray(receipt.fingerprint),
        "hf_state_source_sha256": np.asarray(provenance.state_source_sha256),
        "state_h0": np.asarray(state.h0, dtype="<c16"),
        "state_sigma_z": np.asarray(state.sigma_z, dtype="<c16"),
        "state_density": np.asarray(state.density, dtype="<c16"),
        "state_hamiltonian": np.asarray(state.hamiltonian, dtype="<c16"),
        "state_energies": np.asarray(state.energies, dtype="<f8"),
        "state_sigma_ztauz": np.asarray(state.sigma_ztauz, dtype="<f8"),
        "state_mu": np.asarray(state.mu, dtype="<f8"),
        "state_nu": np.asarray(state.nu, dtype="<f8"),
        "state_v0": np.asarray(state.v0, dtype="<f8"),
        "state_precision": np.asarray(state.precision, dtype="<f8"),
        "state_dimensions": np.asarray(
            [state.n_spin, state.n_eta, state.n_band], dtype="<i8"
        ),
        "iter_energy": np.asarray(hf_run.iter_energy, dtype="<f8"),
        "iter_err": np.asarray(hf_run.iter_err, dtype="<f8"),
        "iter_oda": np.asarray(hf_run.iter_oda, dtype="<f8"),
        "mesh_k_grid_frac": np.asarray(mesh.k_grid_frac, dtype="<f8"),
        "mesh_kvec_b0": np.asarray(mesh.kvec, dtype="<c16"),
        "mesh_kvec_nm_inv": np.asarray(
            mesh.kvec / TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1,
            dtype="<c16",
        ),
        "bm_uk": np.asarray(grid_solution.uk, dtype="<c16"),
        "bm_spectrum": np.asarray(grid_solution.spectrum, dtype="<f8"),
        "bm_gvec": np.asarray(grid_solution.gvec, dtype="<c16"),
        "screened_shifts": np.asarray(shifts, dtype="<i8"),
        "screened_gvecs": np.asarray(blocks.gvecs, dtype="<c16"),
        "screened_overlaps": np.stack(
            [np.asarray(blocks.overlaps[shift], dtype="<c16") for shift in shifts],
            axis=0,
        ),
        "screened_active_shifts": np.asarray(active_shifts, dtype="<i8"),
        "screened_diagonal_overlaps": np.stack(
            [
                np.asarray(blocks.diagonal_overlaps[shift], dtype="<c16")
                for shift in active_shifts
            ],
            axis=0,
        ),
        "screened_hartree_screening": np.asarray(
            [blocks.hartree_screening[shift] for shift in active_shifts],
            dtype="<f8",
        ),
        "screened_fock_screening": np.stack(
            [
                np.asarray(blocks.fock_screening[shift], dtype="<f8")
                for shift in active_shifts
            ],
            axis=0,
        ),
    }

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".validated.tmp")
    staging.unlink(missing_ok=True)
    try:
        write_npz_artifact(arrays, staging, compressed=True)
        loaded = load_tbg_zero_field_complete_hf_state_archive_npz(staging)
        if (
            loaded.receipt.fingerprint != receipt.fingerprint
            or loaded.bundle_metadata["fingerprint"] != bundle.fingerprint
            or loaded.source_attestation_metadata["fingerprint"]
            != source_attestation.fingerprint
        ):
            raise ValueError("Validated complete HF state archive round trip changed source identities")
        staging.replace(output)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    return output


def _validate_typed_b0_hf_artifact_paths(
    artifact_paths: Mapping[str, str | Path] | None,
    *,
    coverage: _ExactSavedSCFPathCoverage,
) -> None:
    paths = {str(key): value for key, value in dict(artifact_paths or {}).items()}
    unsupported = sorted(
        key for key in paths if key not in _TYPED_B0_HF_SIDECAR_ARTIFACT_KEYS
    )
    if unsupported:
        raise ValueError(
            "Typed TBG zero-field sidecars reject off-grid/path/parity artifact "
            f"keys; unsupported keys: {unsupported}"
        )
    required = {
        "advisor_selection_txt",
        TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY,
        "runtime_summary_txt",
        "scf_path_tsv",
    }
    missing = sorted(required - set(paths))
    if missing:
        raise ValueError(
            "Typed TBG zero-field sidecars require exact-SCF evidence artifacts; "
            f"missing keys: {missing}"
        )
    plot_keys = {"scf_band_plot_pdf", "scf_band_plot_png"}
    present_plot_keys = plot_keys & set(paths)
    if present_plot_keys and present_plot_keys != plot_keys:
        raise ValueError("Typed exact-SCF plots must provide both PNG and PDF artifacts")
    if present_plot_keys and not coverage.meaningful:
        raise ValueError(
            "Typed exact-SCF sidecars reject fabricated PNG/PDF plot keys because "
            "the recomputed coverage gate failed"
        )
    if present_plot_keys and "path_limitation_txt" in paths:
        raise ValueError("Typed exact-SCF artifacts cannot claim both a plot and a path limitation")
    if not present_plot_keys and "path_limitation_txt" not in paths:
        raise ValueError("Typed exact-SCF artifacts without a plot require a limitation report")


def _resolve_typed_b0_hf_artifact_paths(
    root: Path,
    artifact_paths: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    root_resolved = root.resolve()
    resolved: dict[str, Path] = {}
    for key, raw_path in dict(artifact_paths or {}).items():
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            rooted_candidate = root_resolved / candidate
            candidate = rooted_candidate if rooted_candidate.exists() else candidate
        try:
            path = candidate.resolve(strict=True)
            path.relative_to(root_resolved)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"Typed artifact {key!r} must reference an existing file under output root {root_resolved}"
            ) from exc
        if not path.is_file():
            raise ValueError(
                f"Typed artifact {key!r} must reference a regular file under output root {root_resolved}"
            )
        resolved[str(key)] = path
    return resolved


def _validate_exact_saved_scf_path_tsv(
    root: Path,
    result: object,
    coverage: _ExactSavedSCFPathCoverage,
    path: Path,
) -> dict[str, object]:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Exact saved-SCF path TSV must be UTF-8") from exc
    lines = text.splitlines()
    if not lines:
        raise ValueError("Exact saved-SCF path TSV is empty")
    header = lines[0].split("\t")
    expected_prefix = [
        "path_index",
        "path_k_dist",
        "k_dist",
        "distance_to_path",
        "path_kx",
        "path_ky",
        "projected_kx",
        "projected_ky",
        "grid_index",
        "grid_kx",
        "grid_ky",
    ]
    if header[: len(expected_prefix)] != expected_prefix:
        raise ValueError("Exact saved-SCF path TSV header does not match the typed lineage schema")

    rows = [line.split("\t") for line in lines[1:]]
    if len(rows) != coverage.exact_point_count:
        raise ValueError(
            "Exact saved-SCF path TSV row count does not match coverage recomputed from result"
        )
    state = getattr(getattr(result, "hf_run"), "state")
    state_energies = np.asarray(getattr(state, "energies"), dtype=float)
    path_spec = getattr(result, "path")
    path_kvec = np.asarray(getattr(path_spec, "kvec"), dtype=np.complex128)
    path_kdist = np.asarray(getattr(path_spec, "kdist"), dtype=float)
    grid_kvec = np.asarray(
        getattr(getattr(result, "grid_solution"), "lattice_kvec"),
        dtype=np.complex128,
    )
    expected_column_count = len(expected_prefix) + int(state_energies.shape[0])
    for row_number, (row, path_index, grid_index, distance) in enumerate(
        zip(
            rows,
            coverage.path_sample_indices,
            coverage.grid_indices,
            coverage.distance_to_path,
            strict=True,
        ),
        start=2,
    ):
        if len(row) != expected_column_count or len(header) != expected_column_count:
            raise ValueError(
                f"Exact saved-SCF path TSV column count is inconsistent at row {row_number}"
            )
        path_coordinate = path_kvec[int(path_index)]
        grid_coordinate = grid_kvec[int(grid_index)]
        expected_row_prefix = [
            str(int(path_index) + 1),
            f"{float(path_kdist[int(path_index)]):.16f}",
            f"{float(path_kdist[int(path_index)]):.16f}",
            f"{float(distance):.16f}",
            f"{float(path_coordinate.real):.16f}",
            f"{float(path_coordinate.imag):.16f}",
            f"{float(path_coordinate.real):.16f}",
            f"{float(path_coordinate.imag):.16f}",
            str(int(grid_index) + 1),
            f"{float(grid_coordinate.real):.16f}",
            f"{float(grid_coordinate.imag):.16f}",
        ]
        if row[: len(expected_prefix)] != expected_row_prefix:
            raise ValueError(
                f"Exact saved-SCF path TSV lineage differs from result at row {row_number}"
            )
        try:
            observed_energies = np.asarray(
                [float(value) for value in row[len(expected_prefix) :]], dtype=float
            )
        except ValueError as exc:
            raise ValueError(
                f"Exact saved-SCF path TSV has invalid band data at row {row_number}"
            ) from exc
        expected_energies = np.sort(state_energies[:, int(grid_index)])
        if not np.allclose(
            observed_energies,
            expected_energies,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError(
                f"Exact saved-SCF path TSV band data differs from saved HF state at row {row_number}"
            )

    relative_path = path.relative_to(root.resolve()).as_posix()
    return {
        "artifact_key": "scf_path_tsv",
        "relative_path": relative_path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": int(len(data)),
        "row_count": int(len(rows)),
    }


def _exact_saved_scf_path_coverage_payload(
    coverage: _ExactSavedSCFPathCoverage,
    *,
    tsv_lineage: Mapping[str, object],
) -> dict[str, object]:
    return {
        "source": "recomputed_from_result_path_and_saved_scf_grid",
        "exact_point_count": int(coverage.exact_point_count),
        "distinct_coordinate_count": int(coverage.distinct_coordinate_count),
        "represented_segment_count": int(len(coverage.represented_segment_indices)),
        "represented_segment_indices_zero_based": [
            int(index) for index in coverage.represented_segment_indices
        ],
        "segment_distinct_coordinate_counts": [
            int(count) for count in coverage.segment_distinct_coordinate_counts
        ],
        "missing_interior_node_labels": list(coverage.missing_interior_node_labels),
        "coverage_gate": {
            "minimum_distinct_coordinates": 3,
            "minimum_represented_segments": 2,
            "requires_all_interior_nodes": True,
            "passed": bool(coverage.meaningful),
        },
        "tsv_lineage": dict(tsv_lineage),
    }

def _reject_typed_b0_hf_suite_outputs(suite_result: object) -> None:
    typed_case_ids: list[str] = []
    for result in tuple(getattr(suite_result, "case_results")):
        if _validated_typed_b0_hf_source(result) is None:
            continue
        typed_case_ids.append(str(getattr(result, "case").benchmark_id))
    if typed_case_ids:
        raise ValueError(
            "Typed TBG zero-field suite/summary writers reject off-grid/path/parity "
            f"data; write exact saved-SCF-grid case artifacts instead: {typed_case_ids}"
        )

def _typed_runtime_environment_payload(runtime: object) -> dict[str, object]:
    environment = getattr(runtime, "environment", None)
    payload = {
        str(key): _json_safe(value)
        for key, value in _dataclass_payload(environment).items()
    }
    for key in ("start_time", "end_time", "bm_elapsed_sec", "hf_elapsed_sec"):
        if hasattr(runtime, key):
            payload[key] = _json_safe(getattr(runtime, key))
    return payload

def _b0_reported_grid_descriptor(result: object) -> dict[str, object]:
    """Return legacy square ``lk`` or an explicit rectangular mesh shape."""

    grid_solution = getattr(result, "grid_solution", None)
    mesh = None if grid_solution is None else getattr(grid_solution, "torus_mesh", None)
    if mesh is None:
        return {"lk": int(getattr(result, "path_result").lk)}
    n1, n2 = (int(value) for value in mesh.mesh_shape)
    if n1 == n2:
        return {"lk": n1}
    return {"mesh_shape": [n1, n2]}

def _b0_hf_validation_payload(
    result: object,
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
    coverage: _ExactSavedSCFPathCoverage | None = None,
    tsv_lineage: Mapping[str, object] | None = None,
) -> dict[str, object]:
    hf_run = getattr(result, "hf_run")
    payload: dict[str, object] = {
        "status": "converged" if bool(hf_run.converged) else "not_converged",
        "converged": bool(hf_run.converged),
        "exit_reason": str(hf_run.exit_reason),
        "iterations": int(hf_run.iterations),
    }
    if _is_typed_hf_run(hf_run):
        _validated_typed_b0_hf_source(result)
        if coverage is None or tsv_lineage is None:
            raise ValueError(
                "Typed HF validation payload requires recomputed coverage and TSV lineage"
            )
        plot_written = {
            "scf_band_plot_pdf",
            "scf_band_plot_png",
        } <= set(dict(artifact_paths or {}))
        payload.update(
            {
                "artifact_mode": "typed_exact_saved_scf_grid_only",
                "path_status": (
                    "exact_saved_scf_plot_written"
                    if plot_written
                    else "limitation_report_only"
                ),
                "exact_saved_scf_path": _exact_saved_scf_path_coverage_payload(
                    coverage,
                    tsv_lineage=tsv_lineage,
                ),
                "advisor_status": "unavailable",
                "complete_hf_state_archive_status": "validated",
                "typed_resume_trajectory": "not_implemented_fail_closed",
            }
        )
        return payload

    parity = getattr(result, "parity")
    payload["parity"] = {
        "kdist_max_abs_diff": float(parity.kdist_max_abs_diff),
        "max_abs_band_diff_mev": float(parity.max_abs_band_diff_mev),
        "rms_band_diff_mev": float(parity.rms_band_diff_mev),
        "mean_abs_band_diff_mev": float(parity.mean_abs_band_diff_mev),
        "energy_sorting": str(parity.energy_sorting),
    }
    runtime_parity = getattr(result, "runtime_parity", None)
    if runtime_parity is not None:
        payload["runtime_parity"] = _json_safe(_dataclass_payload(runtime_parity))
    return payload


def _b0_hf_observables(
    result: object,
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
    coverage: _ExactSavedSCFPathCoverage | None = None,
    tsv_lineage: Mapping[str, object] | None = None,
) -> dict[str, object]:
    case = getattr(result, "case")
    hf_run = getattr(result, "hf_run")
    state = hf_run.state
    typed_source = _validated_typed_b0_hf_source(result)
    if typed_source is not None:
        if coverage is None or tsv_lineage is None:
            raise ValueError(
                "Typed HF observables require recomputed coverage and TSV lineage"
            )
        interaction_spec, screened_block_bundle, receipt = typed_source
        provenance = hf_run.provenance
        grid_solution = getattr(result, "grid_solution")
        plot_written = {
            "scf_band_plot_pdf",
            "scf_band_plot_png",
        } <= set(dict(artifact_paths or {}))
        return {
            "benchmark_id": str(case.benchmark_id),
            "theta_deg": float(grid_solution.params.dtheta_rad * 180.0 / np.pi),
            "nu": float(provenance.nu),
            "mu_mev": float(state.mu),
            "hf_mode": str(provenance.hf_mode),
            "init_mode": str(provenance.normalized_init_mode),
            "normalized_init_mode": str(provenance.normalized_init_mode),
            "seed": int(provenance.seed),
            "iterations": int(hf_run.iterations),
            "exit_reason": str(hf_run.exit_reason),
            "converged": bool(hf_run.converged),
            "diagnostics": _diagnostics_payload(getattr(state, "diagnostics", {})),
            "artifact_mode": "typed_exact_saved_scf_grid_only",
            "path": {
                "status": (
                    "exact_saved_scf_plot_written"
                    if plot_written
                    else "limitation_report_only"
                ),
                "evidence": "exact matched saved-SCF points TSV",
                "coverage": _exact_saved_scf_path_coverage_payload(
                    coverage,
                    tsv_lineage=tsv_lineage,
                ),
            },
            "advisor": {"status": "unavailable"},
            "complete_hf_state_archive": {
                "status": "validated",
                "description": "validated complete HF state archive/checkpoint",
                "resume_authority": "none",
                "typed_resume_trajectory": "not_implemented_fail_closed",
                "schema": TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA,
                "schema_version": TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA_VERSION,
            },
            "bands": {
                "source": "saved_scf_grid_hamiltonian",
                "energy_shape": [
                    int(value) for value in np.asarray(state.energies).shape
                ],
            },
            "state_shapes": _hf_state_shapes(state),
            "hf_source_receipt": receipt.to_metadata(),
            "hf_run_provenance": provenance.to_metadata(),
            "interaction_spec": interaction_spec.to_metadata(),
            "screened_block_bundle": screened_block_bundle.to_metadata(),
        }

    path_result = getattr(result, "path_result")
    band_data = getattr(path_result, "band_data")
    payload: dict[str, object] = {
        "benchmark_id": str(case.benchmark_id),
        "theta_deg": float(case.theta_deg),
        "nu": float(path_result.nu),
        "mu_mev": float(state.mu),
        "init_mode": str(path_result.init_mode),
        "normalized_init_mode": str(path_result.normalized_init_mode),
        "seed": int(path_result.seed),
        "iterations": int(hf_run.iterations),
        "exit_reason": str(hf_run.exit_reason),
        "converged": bool(hf_run.converged),
        "diagnostics": _diagnostics_payload(getattr(state, "diagnostics", {})),
        "path": _kpath_payload(path_result.path),
        "bands": {
            "labels": list(getattr(band_data, "band_labels")),
            "energy_shape": [int(value) for value in np.asarray(getattr(band_data, "energies")).shape],
            "mean_weights_shape": [int(value) for value in np.asarray(getattr(band_data, "mean_weights")).shape],
        },
        "state_shapes": _hf_state_shapes(state),
    }


    receipt = getattr(state, "hf_source_receipt", None)
    if receipt is not None:
        if not isinstance(receipt, TBGZeroFieldHFSourceReceipt):
            raise TypeError("hf_source_receipt must be a TBGZeroFieldHFSourceReceipt")
        payload["hf_source_receipt"] = receipt.to_metadata()
    return payload


def write_bm_unstrained_benchmark_contract_sidecars(
    output_dir: str | Path,
    result: object,
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write public contract sidecars for a zero-field BM benchmark result.

    This is metadata-only: it references existing TSV/plot/text artifacts and
    summarizes in-memory result shapes/scalars, but never writes numerical
    arrays or reruns BM/HF computations.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _ensure_contract_sidecars_absent(root, overwrite=overwrite)
    run = getattr(result, "run")
    reference = getattr(result, "reference")
    files = _relative_artifact_files(root, artifact_paths)
    model = _tbg_model_record(
        run.params,
        extra={
            "theta_deg": float(reference.theta_deg),
            "lg": int(getattr(run.path_solution, "lg")),
            "periodic_g_grid": bool(getattr(run.path_solution, "periodic_g_grid", True)),
        },
    )
    return write_contract_artifacts(
        root,
        workflow="tbg.zero_field.bm_unstrained_benchmark",
        system_name="tbg",
        model=model,
        config={
            "implementation": "python_b0",
            "runner_kind": "bm_unstrained_benchmark",
            "theta_deg": float(reference.theta_deg),
            "lg": int(getattr(run.path_solution, "lg")),
            "grid_lk": None if run.grid_solution is None else int(round(np.sqrt(int(getattr(run.grid_solution, "nk"))) - 1)),
            "points_per_segment": None,
            "reference_path_tsv": str(getattr(reference, "path_tsv_path", "")),
        },
        conventions={
            "energy_unit": "meV",
            "momentum_unit": "dimensionless_b0_code",
            "length_unit": "a_graphene_units",
            "coordinate_convention": TBG_ZERO_FIELD_B0_COORDINATE_CONVENTION,
            "physical_coordinate_convention": TBG_ZERO_FIELD_PHYSICAL_COORDINATE_CONVENTION,
            "graphene_a_nm": TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1,
            "density_convention": "not_applicable",
            "wavefunction_axis_order": "basis_band_valley_k",
            "hamiltonian_axis_order": "basis_basis_valley_k",
            "gauge": "tbg_zero_field_bm_c2t_symmetrized_system_defined",
        },
        environment=_runtime_environment_payload(run.runtime),
        validation=_bm_unstrained_validation_payload(result),
        observables=_bm_unstrained_observables(result),
        files=files,
        metadata={
            "runner_kind": "bm_unstrained_benchmark",
            "adapter": "mean_field.systems.tbg.zero_field.artifacts",
        },
        array_files=(),
    )


def _b0_hf_suite_case_summary(result: object) -> dict[str, object]:
    case = getattr(result, "case")
    hf_run = getattr(result, "hf_run")
    parity = getattr(result, "parity")
    state = hf_run.state
    return {
        "benchmark_id": str(case.benchmark_id),
        "theta_deg": float(case.theta_deg),
        "nu": int(case.nu),
        "init_mode": str(getattr(result, "path_result").init_mode),
        "normalized_init_mode": str(getattr(result, "path_result").normalized_init_mode),
        "seed": int(getattr(result, "path_result").seed),
        **_b0_reported_grid_descriptor(result),
        "lg": int(getattr(result, "path_result").lg),
        "converged": bool(hf_run.converged),
        "exit_reason": str(hf_run.exit_reason),
        "iterations": int(hf_run.iterations),
        "mu_mev": float(state.mu),
        "max_abs_band_diff_mev": float(parity.max_abs_band_diff_mev),
        "kdist_max_abs_diff": float(parity.kdist_max_abs_diff),
        "runtime_total_elapsed_sec": float(getattr(result, "runtime").total_elapsed_sec),
    }


def _b0_hf_suite_validation_payload(suite_result: object) -> dict[str, object]:
    case_results = tuple(getattr(suite_result, "case_results"))
    converged_count = sum(1 for result in case_results if bool(getattr(result, "hf_run").converged))
    return {
        "status": "all_converged" if converged_count == len(case_results) else "not_all_converged",
        "case_count": int(len(case_results)),
        "converged_count": int(converged_count),
        "max_kdist_max_abs_diff": 0.0
        if not case_results
        else float(max(getattr(result, "parity").kdist_max_abs_diff for result in case_results)),
        "max_abs_band_diff_mev": 0.0
        if not case_results
        else float(max(getattr(result, "parity").max_abs_band_diff_mev for result in case_results)),
    }


def _b0_hf_suite_observables(suite_result: object) -> dict[str, object]:
    case_results = tuple(getattr(suite_result, "case_results"))
    return {
        "case_count": int(len(case_results)),
        "total_elapsed_sec": float(sum(getattr(result, "runtime").total_elapsed_sec for result in case_results)),
        "case_results": [_b0_hf_suite_case_summary(result) for result in case_results],
    }


def write_b0_hf_benchmark_contract_sidecars(
    output_dir: str | Path,
    result: object,
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write public contract sidecars for a zero-field B0 HF benchmark result.

    Legacy sidecars remain metadata-only. Typed production sidecars require and
    validate the separately written complete HF state archive, recompute exact
    saved-path coverage, and bind the matching TSV lineage without rerunning HF.
    The archive is a checkpoint for analysis, not authority to resume a typed
    solver trajectory.
    """

    typed_source = _validated_typed_b0_hf_source(result)
    root = Path(output_dir)
    _ensure_contract_sidecars_absent(root, overwrite=overwrite)
    root.mkdir(parents=True, exist_ok=True)
    case = getattr(result, "case")
    hf_run = getattr(result, "hf_run")
    coverage: _ExactSavedSCFPathCoverage | None = None
    tsv_lineage: Mapping[str, object] | None = None
    if typed_source is not None:
        coverage = _exact_saved_scf_path_coverage(result)
        _validate_typed_b0_hf_artifact_paths(
            artifact_paths,
            coverage=coverage,
        )
        normalized_artifact_paths = _resolve_typed_b0_hf_artifact_paths(
            root,
            artifact_paths,
        )
        tsv_lineage = _validate_exact_saved_scf_path_tsv(
            root,
            result,
            coverage,
            normalized_artifact_paths["scf_path_tsv"],
        )
    else:
        normalized_artifact_paths = {
            str(key): _resolve_artifact_input_path(root, value)
            for key, value in dict(artifact_paths or {}).items()
        }
    files = _relative_artifact_files(root, normalized_artifact_paths)
    array_files: tuple[str | Path, ...] = ()
    if typed_source is not None:
        complete_archive_path = normalized_artifact_paths[
            TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY
        ]
        complete_hf_state_archive = load_tbg_zero_field_complete_hf_state_archive_npz(complete_archive_path)
        _interaction_spec, live_bundle, live_receipt = typed_source
        grid_solution = getattr(result, "grid_solution")
        source_attestation = grid_solution.source_attestation
        if (
            complete_hf_state_archive.receipt.fingerprint != live_receipt.fingerprint
            or complete_hf_state_archive.bundle_metadata["fingerprint"] != live_bundle.fingerprint
            or source_attestation is None
            or complete_hf_state_archive.source_attestation_metadata["fingerprint"]
            != source_attestation.fingerprint
        ):
            raise ValueError("Validated complete HF state archive does not match the live artifact source")
        array_files = (complete_archive_path,)
        interaction_spec, screened_block_bundle, _receipt = typed_source
        provenance = hf_run.provenance
        grid_solution = getattr(result, "grid_solution")
        theta_deg = float(grid_solution.params.dtheta_rad * 180.0 / np.pi)
        model = _tbg_model_record(
            grid_solution.params,
            extra={
                "theta_deg": theta_deg,
                "nu": float(provenance.nu),
                "hf_mode": str(provenance.hf_mode),
                **_b0_reported_grid_descriptor(result),
                "lg": int(grid_solution.lg),
                "overlap_lg": int(screened_block_bundle.overlap_lg),
                "beta": float(provenance.beta),
            },
        )
        config_payload = {
            "implementation": "python_b0",
            "runner_kind": "b0_hf_benchmark",
            "artifact_mode": "typed_exact_saved_scf_grid_only",
            "interaction_contract": "typed",
            "benchmark_id": str(case.benchmark_id),
            "theta_deg": theta_deg,
            "nu": float(provenance.nu),
            "hf_mode": str(provenance.hf_mode),
            "init_mode": str(provenance.normalized_init_mode),
            "normalized_init_mode": str(provenance.normalized_init_mode),
            "seed": int(provenance.seed),
            **_b0_reported_grid_descriptor(result),
            "lg": int(grid_solution.lg),
            "overlap_lg": int(screened_block_bundle.overlap_lg),
            "beta": float(provenance.beta),
            "relative_permittivity": float(interaction_spec.epsr),
            "dsc_nm": float(interaction_spec.dsc_nm),
            "finite_zero_limit": bool(interaction_spec.finite_zero_limit),
            "zero_cutoff": float(interaction_spec.zero_cutoff),
            "include_interaction": True,
            "precision": float(provenance.precision),
            "requested_max_iterations": int(provenance.requested_max_iterations),
        }
        environment = _typed_runtime_environment_payload(result.runtime)
    else:
        path_result = getattr(result, "path_result")
        model = _tbg_model_record(
            result.params,
            extra={
                "theta_deg": float(case.theta_deg),
                "nu": int(case.nu),
                **_b0_reported_grid_descriptor(result),
                "lg": int(path_result.lg),
                "overlap_lg": None if path_result.overlap_lg is None else int(path_result.overlap_lg),
                "beta": float(path_result.beta),
            },
        )
        config_payload = {
            "implementation": "python_b0",
            "runner_kind": "b0_hf_benchmark",
            "benchmark_id": str(case.benchmark_id),
            "theta_deg": float(case.theta_deg),
            "nu": int(case.nu),
            "init_mode": str(path_result.init_mode),
            "normalized_init_mode": str(path_result.normalized_init_mode),
            "seed": int(path_result.seed),
            **_b0_reported_grid_descriptor(result),
            "lg": int(path_result.lg),
            "points_per_segment": int(path_result.points_per_segment),
            "overlap_lg": None if path_result.overlap_lg is None else int(path_result.overlap_lg),
            "beta": float(path_result.beta),
            "relative_permittivity": float(path_result.relative_permittivity),
            "screening_lm": _optional_float(path_result.screening_lm),
            "finite_zero_limit": bool(path_result.finite_zero_limit),
            "zero_cutoff": float(path_result.zero_cutoff),
            "include_interaction": bool(path_result.include_interaction),
            "precision": float(hf_run.state.precision),
            "initial_density_override_path": None
            if getattr(result, "initial_density_override_path", None) is None
            else str(result.initial_density_override_path),
        }
        environment = _runtime_environment_payload(result.runtime)

    return write_contract_artifacts(
        root,
        workflow="tbg.zero_field.b0_hf_benchmark",
        system_name="tbg",
        model=model,
        config=config_payload,
        conventions={
            "energy_unit": "meV",
            "momentum_unit": "dimensionless_b0_code",
            "length_unit": "a_graphene_units",
            "coordinate_convention": TBG_ZERO_FIELD_B0_COORDINATE_CONVENTION,
            "physical_coordinate_convention": TBG_ZERO_FIELD_PHYSICAL_COORDINATE_CONVENTION,
            "graphene_a_nm": TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1,
            "density_convention": "stored_delta",
            "density_axis_order": "abk",
            "hamiltonian_axis_order": "abk",
            "wavefunction_axis_order": "basis_band_valley_k",
            "gauge": "tbg_zero_field_b0_projected_system_defined",
        },
        environment=environment,
        validation=_b0_hf_validation_payload(
            result,
            artifact_paths=artifact_paths,
            coverage=coverage,
            tsv_lineage=tsv_lineage,
        ),
        observables=_b0_hf_observables(
            result,
            artifact_paths=artifact_paths,
            coverage=coverage,
            tsv_lineage=tsv_lineage,
        ),
        files=files,
        metadata={
            "runner_kind": "b0_hf_benchmark",
            "benchmark_id": str(case.benchmark_id),
            "adapter": "mean_field.systems.tbg.zero_field.artifacts",
            **(
                {
                    "validated_complete_hf_state_archive": {
                        "artifact_key": TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY,
                        "schema": TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA,
                        "schema_version": TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA_VERSION,
                        "description": "validated complete HF state archive/checkpoint",
                        "validation": "strict_round_trip_rehash",
                        "resume_authority": "none",
                        "typed_resume_trajectory": "not_implemented_fail_closed",
                        "evidence_paths": [
                            "src/mean_field/systems/tbg/zero_field/model.py::BMSolution.source_attestation",
                            "src/mean_field/systems/tbg/zero_field/_hf_basis_overlap.py::TBGZeroFieldHFSourceReceipt",
                            "src/mean_field/systems/tbg/zero_field/_hf_basis_overlap.py::TBGZeroFieldScreenedBlockBundle",
                        ],
                        "uncertainty": {
                            "companion_circular_total_q_cutoff_parity": (
                                "not_established"
                            ),
                            "off_grid_reconstruction": "not_included",
                            "paper_figure_claim": "none",
                        },
                    }
                }
                if typed_source is not None
                else {}
            ),
        },
        array_files=array_files,
    )


def write_b0_hf_suite_contract_sidecars(
    output_dir: str | Path,
    suite_result: object,
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write public contract sidecars for a zero-field B0 HF benchmark suite.

    The suite sidecar is metadata-only. It summarizes case-level scalar metrics
    and references existing suite/case artifacts without writing numerical
    arrays or rerunning BM/HF.
    """

    _reject_typed_b0_hf_suite_outputs(suite_result)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _ensure_contract_sidecars_absent(root, overwrite=overwrite)
    case_results = tuple(getattr(suite_result, "case_results"))
    config_cases = [
        {
            "benchmark_id": str(getattr(result, "case").benchmark_id),
            "theta_deg": float(getattr(result, "case").theta_deg),
            "nu": int(getattr(result, "case").nu),
            "init_mode": str(getattr(result, "path_result").init_mode),
            "seed": int(getattr(result, "path_result").seed),
            **_b0_reported_grid_descriptor(result),
            "lg": int(getattr(result, "path_result").lg),
        }
        for result in case_results
    ]
    return write_contract_artifacts(
        root,
        workflow="tbg.zero_field.b0_hf_suite",
        system_name="tbg",
        model=ModelRecord(system_name="tbg", params={"case_count": int(len(case_results))}),
        config={
            "implementation": "python_b0",
            "runner_kind": "b0_hf_suite",
            "benchmark_ids": [item["benchmark_id"] for item in config_cases],
            "cases": config_cases,
        },
        conventions={
            "energy_unit": "meV",
            "momentum_unit": "dimensionless_b0_code",
            "length_unit": "a_graphene_units",
            "coordinate_convention": TBG_ZERO_FIELD_B0_COORDINATE_CONVENTION,
            "physical_coordinate_convention": TBG_ZERO_FIELD_PHYSICAL_COORDINATE_CONVENTION,
            "graphene_a_nm": TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1,
            "density_convention": "stored_delta",
            "density_axis_order": "abk",
            "hamiltonian_axis_order": "abk",
            "wavefunction_axis_order": "basis_band_valley_k",
            "gauge": "tbg_zero_field_b0_projected_system_defined",
        },
        environment={},
        validation=_b0_hf_suite_validation_payload(suite_result),
        observables=_b0_hf_suite_observables(suite_result),
        files=_relative_artifact_files(root, artifact_paths),
        metadata={
            "runner_kind": "b0_hf_suite",
            "adapter": "mean_field.systems.tbg.zero_field.artifacts",
        },
        array_files=(),
    )


__all__ = [
    "TBGZeroFieldValidatedCompleteHFStateArchive",
    "TBG_ZERO_FIELD_STORED_DENSITY_DEFINITION",
    "TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY",
    "TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_FILENAME",
    "TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA",
    "TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA_VERSION",
    "complex_to_pair",
    "load_tbg_zero_field_complete_hf_state_archive_npz",
    "write_b0_hf_benchmark_contract_sidecars",
    "write_b0_hf_suite_contract_sidecars",
    "write_bm_unstrained_benchmark_contract_sidecars",
    "write_tbg_zero_field_complete_hf_state_archive_npz",
]
