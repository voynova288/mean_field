"""Strict complete-state archive for typed TBG zero-field HF runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mean_field.core.hf.overlap import HFOverlapBlockSet
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
from mean_field.systems.tbg.zero_field._hf_basis_overlap import (
    RestrictedHartreeFockRun,
    restricted_filling,
)
from mean_field.systems.tbg.zero_field._hf_full import normalize_full_init_mode
from mean_field.systems.tbg.zero_field._hf_restricted import normalize_restricted_init_mode
from .hf_contracts import validate_tbg_zero_field_typed_hf_run_source
from .interaction import (
    TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1,
    TBGZeroFieldInteractionSpec,
)
from .model import (
    BMSolution,
    TBGZeroFieldTorusMesh,
    tbg_zero_field_bm_generation_fingerprint,
)

TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA = (
    "mean_field.tbg.zero_field.validated_complete_hf_state_archive"
)
TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA_VERSION = 1
TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_FILENAME = (
    "validated_complete_hf_state_archive.npz"
)
TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY = (
    "validated_complete_hf_state_archive_npz"
)
TBG_ZERO_FIELD_STORED_DENSITY_DEFINITION = (
    "D_stored[a,b]=<c_a† c_b>-0.5*delta_ab"
)


def complex_to_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]


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
    if provenance.get("schema") != "mean_field.tbg.zero_field.hf_run_provenance":
        raise ValueError("Unsupported typed HF run provenance schema")
    schema_version = provenance.get("schema_version")
    if type(schema_version) is not int or schema_version != 2:
        raise ValueError("Unsupported typed HF run provenance schema version")
    if provenance.get("issuer") != "TBGZeroFieldFullRestrictedRunner/v2":
        raise ValueError("Typed HF run provenance issuer is not authoritative")
    if provenance.get("hf_mode") not in {"full", "restricted"}:
        raise ValueError("Typed HF run provenance hf_mode is invalid")
    for name in ("beta", "nu", "filling", "precision", "oda_stall_threshold"):
        value = provenance.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(float(value))
        ):
            raise ValueError(f"Typed HF run provenance {name} must be finite real")
    if float(provenance["filling"]) != float(provenance["nu"]):
        raise ValueError("Typed HF run provenance filling does not equal nu")
    if float(provenance["precision"]) <= 0.0:
        raise ValueError("Typed HF run provenance precision must be positive")
    if float(provenance["oda_stall_threshold"]) <= 0.0:
        raise ValueError("Typed HF run provenance oda_stall_threshold must be positive")
    requested = provenance.get("requested_max_iterations")
    if not isinstance(requested, int) or isinstance(requested, bool) or requested < 0:
        raise ValueError("Typed HF run provenance requested_max_iterations is invalid")
    seed = provenance.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Typed HF run provenance seed must be an integer")
    if not isinstance(provenance.get("converged"), bool):
        raise ValueError("Typed HF run provenance converged must be bool")
    init_mode = provenance.get("normalized_init_mode")
    if not isinstance(init_mode, str) or not init_mode:
        raise ValueError("Typed HF run provenance normalized_init_mode is invalid")
    try:
        normalized_mode = (
            normalize_full_init_mode(init_mode)
            if provenance["hf_mode"] == "full"
            else normalize_restricted_init_mode(init_mode)
        )
    except ValueError as exc:
        raise ValueError(
            "Typed HF run provenance normalized_init_mode is invalid for hf_mode"
        ) from exc
    if normalized_mode != init_mode:
        raise ValueError("Typed HF run provenance normalized_init_mode is not canonical")
    exit_reason = provenance.get("exit_reason")
    if (
        not isinstance(exit_reason, str)
        or not exit_reason
        or exit_reason != exit_reason.strip()
    ):
        raise ValueError("Typed HF run provenance exit_reason is invalid")
    for name in (
        "typed_receipt_fingerprint",
        "interaction_spec_fingerprint",
        "bm_generation_fingerprint",
        "mesh_fingerprint",
        "iter_energy_sha256",
        "iter_err_sha256",
        "iter_oda_sha256",
        "state_source_sha256",
    ):
        _validate_sha256_text(provenance.get(name), name=f"hf_run_provenance.{name}")
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
    for provenance_name, diagnostics_name in (
        ("beta", "beta"),
        ("oda_stall_threshold", "oda_stall_threshold"),
        ("requested_max_iterations", "requested_max_iterations"),
    ):
        diagnostic_value = diagnostics.get(diagnostics_name)
        if (
            not isinstance(diagnostic_value, (int, float))
            or isinstance(diagnostic_value, bool)
            or not np.isfinite(float(diagnostic_value))
            or float(diagnostic_value) != float(provenance_metadata[provenance_name])
        ):
            raise ValueError(
                "Validated complete HF state archive provenance "
                f"{provenance_name} does not match state diagnostics"
            )

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
    diagnostic_filling = diagnostics.get("filling")
    density_filling = restricted_filling(state.density)
    if (
        not isinstance(diagnostic_filling, (int, float))
        or isinstance(diagnostic_filling, bool)
        or not np.isfinite(float(diagnostic_filling))
        or not np.isclose(
            float(diagnostic_filling), density_filling, rtol=1.0e-12, atol=1.0e-12
        )
    ):
        raise ValueError(
            "Validated complete HF state archive diagnostic filling does not "
            "match density-derived filling"
        )
    if not np.isclose(density_filling, state.nu, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(
            "Validated complete HF state archive density-derived filling does "
            "not match requested nu"
        )
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
    if provenance_metadata.get("hf_mode") != receipt.hf_mode:
        raise ValueError("Validated complete HF state archive provenance does not match receipt hf_mode")

    iter_energy = np.asarray(arrays["iter_energy"], dtype=np.float64)
    iter_err = np.asarray(arrays["iter_err"], dtype=np.float64)
    iter_oda = np.asarray(arrays["iter_oda"], dtype=np.float64)
    if not (iter_energy.shape == iter_err.shape == iter_oda.shape):
        raise ValueError("Validated complete HF state archive history shapes differ")
    if int(provenance_metadata["requested_max_iterations"]) < int(iter_energy.size):
        raise ValueError("Validated complete HF state archive history exceeds requested iterations")
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
    *,
    hf_run: RestrictedHartreeFockRun,
    grid_solution: BMSolution,
) -> Path:
    """Atomically publish a validated complete HF state archive.

    The archive preserves complete arrays and source lineage but grants no
    trajectory-resume authority.  Both typed inputs are explicit; benchmark
    wrapper objects are intentionally unsupported.
    """

    interaction_spec, bundle, receipt = validate_tbg_zero_field_typed_hf_run_source(
        hf_run, grid_solution
    )
    state = hf_run.state
    provenance = hf_run.provenance
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

__all__ = [
    "TBGZeroFieldValidatedCompleteHFStateArchive",
    "TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY",
    "TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_FILENAME",
    "TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA",
    "TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA_VERSION",
    "TBG_ZERO_FIELD_STORED_DENSITY_DEFINITION",
    "load_tbg_zero_field_complete_hf_state_archive_npz",
    "write_tbg_zero_field_complete_hf_state_archive_npz",
]
