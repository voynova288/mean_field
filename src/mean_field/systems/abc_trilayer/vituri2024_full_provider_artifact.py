"""Immutable candidate-artifact envelope for the Vituri full provider.

This module binds canonical JSON metadata and deterministic NPZ bytes to one
replay payload, one explicit conventional-density reference ``R``, interaction
and area choices, q=0/background declarations, and the existing absolute
full-functional replay bridge.  It verifies artifact integrity and candidate
array parity only.  It never establishes source closure, normal-order or q=0
background authority, stationarity, TDHF/scalar-Hessian authority, production
readiness, or paper reproduction.
"""

from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass, field, fields, is_dataclass, replace
from hashlib import sha256
import base64
import hmac
import io
import json
import math
import os
from pathlib import Path
import stat
from typing import Final, Literal
import zipfile

import numpy as np

from .vituri2024_hf_preflight import (
    ACTIVE_BAND_STATES_GAUGE_SCOPE,
    ACTIVE_BAND_STATES_LAYOUT,
    ACTIVE_BAND_STATES_VALLEY_ORDER,
    CANONICAL_BASIS_KIND,
    FOCK_DECOMPOSITION_CONVENTION,
    INTERNAL_FLAVOR_ORDER,
    ORBITAL_INDEX_DESCRIPTOR_LABEL,
    ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT,
    ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL,
    REPLAY_ORBITAL_ORDER,
    REPLAY_PAYLOAD_SCHEMA_FINGERPRINT,
    REPLAY_RESIDUAL_NORM,
)
from .vituri2024_hf_replay import (
    Vituri2024HalfMetalHFReplayPayload,
    canonical_array_sha256,
    canonical_orbital_order_sha256,
)
from .vituri2024_interaction import (
    Vituri2024InteractionBinding,
    Vituri2024InteractionChoiceReceipt,
)
from .vituri2024_tdhf_full_provider_bridge import (
    Vituri2024FullFunctionalReplayBridge,
    build_vituri2024_full_functional_replay_bridge,
)

Array = np.ndarray
InteractionInput = Vituri2024InteractionChoiceReceipt | Vituri2024InteractionBinding
ArtifactKind = Literal["synthetic_fixture", "provider_candidate"]
NormalReferenceKind = Literal[
    "paper_projected_R0_representative_only",
    "provider_supplied_explicit_R_unqualified",
]
Q0BackgroundStatus = Literal["absent", "declared_evidence_bound_not_executable"]

VITURI2024_FULL_PROVIDER_ARTIFACT_SCHEMA: Final[str] = (
    "vituri2024_full_provider_candidate_artifact.v1"
)
VITURI2024_FULL_PROVIDER_ARTIFACT_JSON_PROFILE: Final[str] = (
    "utf8_sorted_keys_compact_no_nan_newline_v1"
)
VITURI2024_FULL_PROVIDER_ARTIFACT_NPZ_PROFILE: Final[str] = (
    "zip_stored_fixed1980_posix0444_sorted_npy_v1"
)
VITURI2024_FULL_PROVIDER_ARTIFACT_RELATIONSHIP: Final[str] = (
    "lineage_bound_full_provider_candidate_artifact"
)
VITURI2024_FULL_PROVIDER_ARTIFACT_PROVIDER_IDENTITY_RECIPE: Final[str] = (
    "target_free_provider_execution_identity.v1"
)
VITURI2024_FULL_PROVIDER_ARTIFACT_SOURCE_STATE_RECIPE: Final[str] = (
    "vituri2024_half_metal_replay_source_state.v1"
)
VITURI2024_FULL_PROVIDER_ARTIFACT_RUNTIME_SCOPE: Final[str] = (
    "checkout_path_bound_code_manifests"
)
VITURI2024_FULL_PROVIDER_ARTIFACT_TARGET_EXCLUSION_SCOPE: Final[str] = (
    "saved_interaction_h_fock_and_energies_excluded_directly_while_source_commit_"
    "and_loader_lineage_remain_in_execution_metadata"
)
VITURI2024_FULL_PROVIDER_ARTIFACT_TOLERANCE: Final[float] = 1.0e-10
VITURI2024_FULL_PROVIDER_ARTIFACT_MAX_MANIFEST_BYTES: Final[int] = 4 * 1024 * 1024
VITURI2024_FULL_PROVIDER_ARTIFACT_MAX_NPZ_BYTES: Final[int] = 4 * 1024 * 1024 * 1024
VITURI2024_FULL_PROVIDER_ARTIFACT_ARRAY_ORDER: Final[tuple[str, ...]] = (
    "active_band_states",
    "energies",
    "fock",
    "h0",
    "interaction_h",
    "mesh",
    "normal_order_reference_full",
    "occupations",
    "projector",
)
VITURI2024_FULL_PROVIDER_ARTIFACT_AUTHORITY: Final[str] = (
    "immutable_bytes_schema_and_absolute_candidate_array_parity_only_no_source_"
    "normal_order_q0_stationarity_tdhf_slurm_production_or_paper_authority"
)
if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
    raise RuntimeError("Vituri immutable artifact loading requires O_NOFOLLOW/O_DIRECTORY")

_LOADED_ARTIFACT_TOKEN = object()

VITURI2024_FULL_PROVIDER_ARTIFACT_INTERACTION_KEYS: Final[tuple[str, ...]] = (
    "gate_distance_angstrom",
    "coulomb_e2_ev_angstrom",
    "q0_evaluation",
    "provider_sha256",
    "source_sha256",
    "authority_kind",
    "source_text",
    "epsilon",
    "q_tf_per_a0",
    "q_tf_inverse_angstrom",
    "paper_direct_claim_allowed",
    "establishes_hf_q0_background",
)


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    payload = (
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    )
    return sha256(payload).hexdigest()


def _fingerprint(value: object) -> str:
    def stable(item: object) -> object:
        if isinstance(item, np.ndarray):
            return {
                "dtype": item.dtype.str,
                "shape": list(item.shape),
                "sha256": _array_sha256(item),
            }
        if is_dataclass(item) and not isinstance(item, type):
            return {entry.name: stable(getattr(item, entry.name)) for entry in fields(item)}
        if isinstance(item, dict):
            return {str(key): stable(value) for key, value in sorted(item.items())}
        if isinstance(item, (tuple, list)):
            return [stable(value) for value in item]
        if isinstance(item, np.generic):
            return stable(item.item())
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        raise TypeError(f"unsupported artifact fingerprint type {type(item).__name__}")

    return sha256(
        json.dumps(stable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be explicit nonempty text")
    return value


def _canonical_json_bytes(document: dict[str, object]) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _json_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant {value!r} is forbidden")


def _parse_canonical_json(raw: bytes) -> dict[str, object]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("artifact manifest BOM is forbidden")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact manifest is not strict UTF-8 JSON") from error
    if type(document) is not dict:
        raise ValueError("artifact manifest root must be an object")
    if _canonical_json_bytes(document) != raw:
        raise ValueError("artifact manifest bytes are not canonical JSON")
    return document


def _exact_keys(value: object, expected: tuple[str, ...], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"{label} must contain the exact key inventory {expected}")
    return value


def _require_exact_typed_equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(f"{label} JSON type drifted")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{label} key inventory drifted")
        for key in expected:
            _require_exact_typed_equal(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label} list length drifted")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _require_exact_typed_equal(left, right, f"{label}[{index}]")
        return
    if actual != expected:
        raise ValueError(f"{label} value drifted")


def _strict_int(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value < 1):
        raise ValueError(f"{label} must be an exact {'positive ' if positive else ''}integer")
    return value


def _strict_float(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"{label} must be an exact finite {'positive ' if positive else ''}float")
    return value


def vituri2024_full_provider_target_free_identity_fingerprint() -> str:
    """Return the fixed callback execution namespace, independent of provider data.

    Provider implementation/source identities remain in the target-bearing
    artifact lineage.  They are deliberately excluded from executable callback
    inputs because arbitrary digests can encode saved targets transitively.
    """

    return _fingerprint(
        {
            "recipe": VITURI2024_FULL_PROVIDER_ARTIFACT_PROVIDER_IDENTITY_RECIPE,
            "callback_surface": "canonical_vituri2024_full_provider_E_F_dF",
            "provider_specific_lineage_excluded": True,
            "excluded": (
                "provider_name",
                "provider_implementation_bytes",
                "source_commit",
                "replay_loader_implementation_fingerprint",
                "source_artifact_sha256",
                "source_state_sha256",
                "interaction_h",
                "fock",
                "energies",
            ),
        }
    )


def _orbital_descriptor_fingerprint(mesh_sha256: str, orbital_sha256: str) -> str:
    return _fingerprint(
        {
            "descriptor_label": ORBITAL_INDEX_DESCRIPTOR_LABEL,
            "schema_label": ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL,
            "schema_fingerprint": ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT,
            "ordered_orbitals_sha256": orbital_sha256,
            "ordered_momentum_mesh_sha256": mesh_sha256,
            "internal_flavor_order": INTERNAL_FLAVOR_ORDER,
            "orbital_order": REPLAY_ORBITAL_ORDER,
        }
    )


def vituri2024_full_provider_artifact_source_state_fingerprint(
    *,
    payload: Vituri2024HalfMetalHFReplayPayload,
    geometry_receipt_fingerprint: str,
    ensemble_receipt_fingerprint: str,
) -> str:
    if type(payload) is not Vituri2024HalfMetalHFReplayPayload:
        raise TypeError("source-state reconstruction requires exact replay payload")
    _sha256(geometry_receipt_fingerprint, "geometry receipt")
    _sha256(ensemble_receipt_fingerprint, "ensemble receipt")
    mesh_hash = canonical_array_sha256(payload.mesh)
    orbital_hash = canonical_orbital_order_sha256(payload.mesh)
    return _fingerprint(
        {
            "ordered_orbitals_sha256": orbital_hash,
            "ordered_orbitals_descriptor_label": ORBITAL_INDEX_DESCRIPTOR_LABEL,
            "ordered_orbitals_schema_label": ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL,
            "ordered_orbitals_schema_fingerprint": ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT,
            "ordered_orbitals_descriptor_fingerprint": _orbital_descriptor_fingerprint(
                mesh_hash, orbital_hash
            ),
            "ordered_energies_sha256": canonical_array_sha256(payload.energies),
            "ordered_occupations_sha256": canonical_array_sha256(payload.occupations),
            "ordered_projector_sha256": canonical_array_sha256(payload.projector),
            "ordered_fock_sha256": canonical_array_sha256(payload.fock),
            "h0_sha256": canonical_array_sha256(payload.h0),
            "interaction_h_sha256": canonical_array_sha256(payload.interaction_h),
            "active_band_states_sha256": canonical_array_sha256(
                payload.active_band_states
            ),
            "active_band_states_layout": ACTIVE_BAND_STATES_LAYOUT,
            "active_band_states_valley_order": ACTIVE_BAND_STATES_VALLEY_ORDER,
            "active_band_states_gauge_scope": ACTIVE_BAND_STATES_GAUGE_SCOPE,
            "replay_loader_implementation_fingerprint": payload.replay_loader_implementation_fingerprint,
            "replay_payload_schema_fingerprint": payload.replay_payload_schema_fingerprint,
            "canonical_basis_kind": CANONICAL_BASIS_KIND,
            "residual_norm": REPLAY_RESIDUAL_NORM,
            "fock_decomposition_convention": FOCK_DECOMPOSITION_CONVENTION,
            "geometry_receipt_fingerprint": geometry_receipt_fingerprint,
            "ensemble_receipt_fingerprint": ensemble_receipt_fingerprint,
            "source_commit": payload.source_commit,
            "source_artifact_sha256": payload.source_artifact_sha256,
        }
    )


def _normal_reference_fingerprint(
    *,
    reference: Array,
    kind: NormalReferenceKind,
    evidence_sha256: str,
) -> str:
    return _fingerprint(
        {
            "array_sha256": _array_sha256(reference),
            "density_convention": "P_ij=<c_j_dagger c_i>; Q=P-R",
            "layout": "flat=flavor*Nk+k",
            "semantic_kind": kind,
            "evidence_sha256": evidence_sha256,
            "targets_and_source_state_excluded": (
                "source_artifact_sha256",
                "source_state_sha256",
                "interaction_h",
                "fock",
                "energies",
            ),
            "authority": "candidate_subtraction_reference_only",
        }
    )


def vituri2024_full_provider_saved_target_excluding_source_input_fingerprint(
    payload: Vituri2024HalfMetalHFReplayPayload,
) -> str:
    """Mirror the bridge source-input scope with saved arrays excluded directly."""

    if type(payload) is not Vituri2024HalfMetalHFReplayPayload:
        raise TypeError("source-input fingerprint requires exact replay payload")
    return _fingerprint(
        {
            "provider_fingerprint": payload.provider_fingerprint,
            "source_commit": payload.source_commit,
            "replay_loader_implementation_fingerprint": (
                payload.replay_loader_implementation_fingerprint
            ),
            "replay_payload_schema_fingerprint": (
                payload.replay_payload_schema_fingerprint
            ),
            "mesh": _array_sha256(payload.mesh),
            "active_band_states": _array_sha256(payload.active_band_states),
            "h0": _array_sha256(payload.h0),
            "projector": _array_sha256(payload.projector),
            "occupations": _array_sha256(payload.occupations),
            "targets_excluded": ("interaction_h", "fock", "energies"),
        }
    )


def _q0_policy_fingerprint(
    *, interaction: Vituri2024InteractionChoiceReceipt, status: Q0BackgroundStatus,
    evidence_sha256: str,
) -> str:
    return _fingerprint(
        {
            "kernel_q0_evaluation": interaction.q0_evaluation,
            "hf_background_status": status,
            "evidence_sha256": evidence_sha256,
            "kernel_limit_is_not_hf_background": True,
            "hf_background_authority_established": False,
        }
    )


def _npy_bytes(array: Array) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def _deterministic_npz(arrays: dict[str, Array]) -> tuple[bytes, dict[str, bytes]]:
    if tuple(sorted(arrays)) != VITURI2024_FULL_PROVIDER_ARTIFACT_ARRAY_ORDER:
        raise ValueError("artifact serializer array inventory is not canonical")
    archive = io.BytesIO()
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_STORED) as handle:
        handle.comment = b""
        for name in VITURI2024_FULL_PROVIDER_ARTIFACT_ARRAY_ORDER:
            member = _npy_bytes(arrays[name])
            members[name] = member
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o444) << 16
            info.extra = b""
            info.comment = b""
            handle.writestr(info, member)
    return archive.getvalue(), members


@dataclass(frozen=True, slots=True)
class Vituri2024FullProviderArtifactExpectation:
    manifest_file_name: str
    arrays_file_name: str
    manifest_sha256: str
    manifest_size_bytes: int
    arrays_sha256: str
    arrays_size_bytes: int
    artifact_root_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate_fields(check_root=False)
        object.__setattr__(self, "artifact_root_sha256", self._current_root())

    def _current_root(self) -> str:
        return _fingerprint(
            {
                "schema": VITURI2024_FULL_PROVIDER_ARTIFACT_SCHEMA,
                "manifest_file_name": self.manifest_file_name,
                "manifest_sha256": self.manifest_sha256,
                "manifest_size_bytes": self.manifest_size_bytes,
                "arrays_file_name": self.arrays_file_name,
                "arrays_sha256": self.arrays_sha256,
                "arrays_size_bytes": self.arrays_size_bytes,
            }
        )

    def _validate_fields(self, *, check_root: bool) -> None:
        for name in ("manifest_file_name", "arrays_file_name"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or not value
                or Path(value).name != value
                or value in (".", "..")
            ):
                raise ValueError(f"artifact expectation {name} must be one basename")
        _sha256(self.manifest_sha256, "manifest bytes")
        _sha256(self.arrays_sha256, "arrays bytes")
        _strict_int(self.manifest_size_bytes, "manifest size", positive=True)
        _strict_int(self.arrays_size_bytes, "arrays size", positive=True)
        if check_root and self._current_root() != self.artifact_root_sha256:
            raise ValueError("detached artifact expectation trust root drifted")

    def validate_live_state(self) -> None:
        self._validate_fields(check_root=True)

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.artifact_root_sha256


@dataclass(frozen=True, slots=True)
class Vituri2024SerializedFullProviderArtifact:
    manifest_bytes: bytes
    arrays_bytes: bytes
    expectation: Vituri2024FullProviderArtifactExpectation
    bridge_fingerprint: str
    source_input_fingerprint: str
    authority: str = field(default=VITURI2024_FULL_PROVIDER_ARTIFACT_AUTHORITY, init=False)
    production_ready: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if type(self.manifest_bytes) is not bytes or type(self.arrays_bytes) is not bytes:
            raise TypeError("serialized artifact requires exact bytes")
        if sha256(self.manifest_bytes).hexdigest() != self.expectation.manifest_sha256:
            raise ValueError("serialized manifest bytes/expectation mismatch")
        if sha256(self.arrays_bytes).hexdigest() != self.expectation.arrays_sha256:
            raise ValueError("serialized arrays bytes/expectation mismatch")
        _sha256(self.bridge_fingerprint, "serialized bridge")
        _sha256(self.source_input_fingerprint, "serialized source inputs")
        if self.production_ready is not False:
            raise ValueError("serialized candidate artifact cannot be production ready")


@dataclass(frozen=True, slots=True)
class Vituri2024FullProviderArtifactDiagnostics:
    fock_decomposition_max_abs_ev: float
    active_state_norm_max_abs: float
    projector_hermiticity_max_abs: float
    projector_idempotency_max_abs: float
    projector_vs_occupations_max_abs: float
    h0_hermiticity_max_abs_ev: float
    interaction_h_hermiticity_max_abs_ev: float
    fock_hermiticity_max_abs_ev: float
    fock_projector_commutator_max_abs_ev: float
    fock_vs_ordered_energies_max_abs_ev: float
    tolerance: float = field(default=VITURI2024_FULL_PROVIDER_ARTIFACT_TOLERANCE, init=False)
    passed: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        for item in fields(self):
            if item.name in ("passed",):
                continue
            value = getattr(self, item.name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"artifact diagnostic {item.name} must be finite/nonnegative")
        if self.tolerance != VITURI2024_FULL_PROVIDER_ARTIFACT_TOLERANCE:
            raise ValueError("artifact diagnostic tolerance drifted")
        if self.passed is not True:
            raise ValueError("failed diagnostics cannot enter a loaded artifact")
        for item in fields(self):
            if item.name.endswith("max_abs") or "max_abs_" in item.name:
                if getattr(self, item.name) > self.tolerance:
                    raise ValueError(f"artifact diagnostic {item.name} exceeds tolerance")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def _max_abs(value: Array) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _payload_diagnostics(payload: Vituri2024HalfMetalHFReplayPayload) -> Vituri2024FullProviderArtifactDiagnostics:
    product = np.einsum("abk,bck->ack", payload.fock, payload.projector, optimize=False)
    reverse = np.einsum("abk,bck->ack", payload.projector, payload.fock, optimize=False)
    p2 = np.einsum("abk,bck->ack", payload.projector, payload.projector, optimize=False)
    diagonal_occupations = np.zeros_like(payload.projector)
    diagonal_energies = np.zeros_like(payload.fock)
    indices = np.arange(len(INTERNAL_FLAVOR_ORDER))
    diagonal_occupations[indices, indices, :] = payload.occupations
    diagonal_energies[indices, indices, :] = payload.energies
    return Vituri2024FullProviderArtifactDiagnostics(
        fock_decomposition_max_abs_ev=_max_abs(payload.fock - payload.h0 - payload.interaction_h),
        active_state_norm_max_abs=_max_abs(
            np.sum(np.abs(payload.active_band_states) ** 2, axis=1) - 1.0
        ),
        projector_hermiticity_max_abs=_max_abs(
            payload.projector - payload.projector.swapaxes(0, 1).conj()
        ),
        projector_idempotency_max_abs=_max_abs(p2 - payload.projector),
        projector_vs_occupations_max_abs=_max_abs(payload.projector - diagonal_occupations),
        h0_hermiticity_max_abs_ev=_max_abs(payload.h0 - payload.h0.swapaxes(0, 1).conj()),
        interaction_h_hermiticity_max_abs_ev=_max_abs(
            payload.interaction_h - payload.interaction_h.swapaxes(0, 1).conj()
        ),
        fock_hermiticity_max_abs_ev=_max_abs(payload.fock - payload.fock.swapaxes(0, 1).conj()),
        fock_projector_commutator_max_abs_ev=_max_abs(product - reverse),
        fock_vs_ordered_energies_max_abs_ev=_max_abs(payload.fock - diagonal_energies),
    )


def _interaction_receipt(interaction: InteractionInput) -> Vituri2024InteractionChoiceReceipt:
    if type(interaction) is Vituri2024InteractionChoiceReceipt:
        return interaction
    if type(interaction) is Vituri2024InteractionBinding:
        return interaction.receipt
    raise TypeError("artifact interaction must be exact receipt or binding")


def _reference_array(reference: object, dimension: int) -> Array:
    if (
        not isinstance(reference, np.ndarray)
        or reference.dtype != np.dtype(np.complex128)
        or reference.shape != (dimension, dimension)
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError("artifact normal reference must be finite complex128 (4Nk,4Nk)")
    result = np.frombuffer(reference.tobytes(order="C"), dtype=np.complex128).reshape(
        dimension, dimension
    )
    result.setflags(write=False)
    if _max_abs(result - result.conj().T) > VITURI2024_FULL_PROVIDER_ARTIFACT_TOLERANCE:
        raise ValueError("artifact normal reference is not Hermitian")
    eigenvalues = np.linalg.eigvalsh(result)
    if float(np.min(eigenvalues)) < -VITURI2024_FULL_PROVIDER_ARTIFACT_TOLERANCE or float(
        np.max(eigenvalues)
    ) > 1.0 + VITURI2024_FULL_PROVIDER_ARTIFACT_TOLERANCE:
        raise ValueError("artifact normal reference must satisfy 0<=R<=I")
    return result


def serialize_vituri2024_full_provider_artifact_candidate(
    *,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
    normal_order_reference_full: Array,
    area_angstrom_squared: float,
    interaction: InteractionInput,
    provider_name: str,
    provider_implementation_bytes: bytes,
    geometry_receipt_fingerprint: str,
    ensemble_receipt_fingerprint: str,
    selected_branch_label: str,
    selected_spin: int,
    branch_table_sha256: str,
    normal_reference_kind: NormalReferenceKind,
    normal_reference_evidence_text: str,
    area_evidence_text: str,
    q0_background_status: Q0BackgroundStatus,
    q0_background_evidence_text: str,
    provenance: str,
    artifact_kind: ArtifactKind,
    manifest_file_name: str = "vituri2024_full_provider_manifest.json",
    arrays_file_name: str = "vituri2024_full_provider_arrays.npz",
) -> Vituri2024SerializedFullProviderArtifact:
    if type(source_payload) is not Vituri2024HalfMetalHFReplayPayload:
        raise TypeError("artifact serializer requires exact replay payload")
    if artifact_kind not in ("synthetic_fixture", "provider_candidate"):
        raise ValueError("artifact kind is invalid")
    if normal_reference_kind not in (
        "paper_projected_R0_representative_only",
        "provider_supplied_explicit_R_unqualified",
    ):
        raise ValueError("artifact normal-reference kind is invalid")
    if q0_background_status not in ("absent", "declared_evidence_bound_not_executable"):
        raise ValueError("artifact q0-background status is invalid")
    if type(selected_spin) is not int or selected_spin not in (-1, 1):
        raise ValueError("artifact selected spin must be exact -1 or +1")
    if type(provider_implementation_bytes) is not bytes or not provider_implementation_bytes:
        raise TypeError("artifact provider implementation must be exact nonempty bytes")
    provider_code_sha256 = sha256(provider_implementation_bytes).hexdigest()
    for value, label in (
        (geometry_receipt_fingerprint, "geometry receipt"),
        (ensemble_receipt_fingerprint, "ensemble receipt"),
        (branch_table_sha256, "branch table"),
    ):
        _sha256(value, label)
    for value, label in (
        (provider_name, "provider name"),
        (selected_branch_label, "selected branch"),
        (normal_reference_evidence_text, "normal-reference evidence"),
        (area_evidence_text, "area evidence"),
        (q0_background_evidence_text, "q0-background evidence"),
        (provenance, "artifact provenance"),
    ):
        _strict_text(value, label)
    area = _strict_float(area_angstrom_squared, "artifact area", positive=True)
    interaction_receipt = _interaction_receipt(interaction)
    expected_provider = vituri2024_full_provider_target_free_identity_fingerprint()
    if not hmac.compare_digest(source_payload.provider_fingerprint, expected_provider):
        raise ValueError("payload provider fingerprint is not target-free by artifact recipe")
    expected_source_state = vituri2024_full_provider_artifact_source_state_fingerprint(
        payload=source_payload,
        geometry_receipt_fingerprint=geometry_receipt_fingerprint,
        ensemble_receipt_fingerprint=ensemble_receipt_fingerprint,
    )
    if not hmac.compare_digest(source_payload.source_state_sha256, expected_source_state):
        raise ValueError("payload source-state fingerprint is not reconstructible")
    if source_payload.replay_payload_schema_fingerprint != REPLAY_PAYLOAD_SCHEMA_FINGERPRINT:
        raise ValueError("artifact payload replay schema is not canonical")
    diagnostics = _payload_diagnostics(source_payload)
    nk = int(source_payload.mesh.shape[0])
    reference = _reference_array(normal_order_reference_full, 4 * nk)
    if normal_reference_kind == "paper_projected_R0_representative_only" and np.any(
        reference != 0.0
    ):
        raise ValueError("paper projected-Hamiltonian reference kind requires exact R=0")
    reference_evidence_sha = sha256(normal_reference_evidence_text.encode()).hexdigest()
    area_evidence_sha = sha256(area_evidence_text.encode()).hexdigest()
    q0_evidence_sha = sha256(q0_background_evidence_text.encode()).hexdigest()
    reference_fingerprint = _normal_reference_fingerprint(
        reference=reference,
        kind=normal_reference_kind,
        evidence_sha256=reference_evidence_sha,
    )
    q0_policy_fingerprint = _q0_policy_fingerprint(
        interaction=interaction_receipt,
        status=q0_background_status,
        evidence_sha256=q0_evidence_sha,
    )
    bridge = build_vituri2024_full_functional_replay_bridge(
        source_payload=source_payload,
        normal_order_reference_full=reference,
        area_angstrom_squared=area,
        interaction=interaction_receipt,
        normal_order_reference_fingerprint=reference_fingerprint,
        reference_policy_evidence_sha256=reference_evidence_sha,
        q0_policy_fingerprint=q0_policy_fingerprint,
        q0_background_evidence_sha256=q0_evidence_sha,
        provenance=provenance,
    )
    arrays = {
        "active_band_states": source_payload.active_band_states,
        "energies": source_payload.energies,
        "fock": source_payload.fock,
        "h0": source_payload.h0,
        "interaction_h": source_payload.interaction_h,
        "mesh": source_payload.mesh,
        "normal_order_reference_full": reference,
        "occupations": source_payload.occupations,
        "projector": source_payload.projector,
    }
    arrays_bytes, member_bytes = _deterministic_npz(arrays)
    descriptors = [
        {
            "name": name,
            "member": f"{name}.npy",
            "dtype": arrays[name].dtype.str,
            "shape": list(arrays[name].shape),
            "nbytes": int(arrays[name].nbytes),
            "array_sha256": _array_sha256(arrays[name]),
            "canonical_array_sha256": canonical_array_sha256(arrays[name]),
            "member_sha256": sha256(member_bytes[name]).hexdigest(),
        }
        for name in VITURI2024_FULL_PROVIDER_ARTIFACT_ARRAY_ORDER
    ]
    interaction_document = asdict(interaction_receipt)
    manifest: dict[str, object] = {
        "schema": VITURI2024_FULL_PROVIDER_ARTIFACT_SCHEMA,
        "artifact_kind": artifact_kind,
        "artifact_relationship": VITURI2024_FULL_PROVIDER_ARTIFACT_RELATIONSHIP,
        "json_profile": VITURI2024_FULL_PROVIDER_ARTIFACT_JSON_PROFILE,
        "npz_profile": VITURI2024_FULL_PROVIDER_ARTIFACT_NPZ_PROFILE,
        "runtime_identity_scope": VITURI2024_FULL_PROVIDER_ARTIFACT_RUNTIME_SCOPE,
        "files": {
            "arrays_file_name": arrays_file_name,
            "arrays_sha256": sha256(arrays_bytes).hexdigest(),
            "arrays_size_bytes": len(arrays_bytes),
        },
        "arrays": descriptors,
        "payload_metadata": {
            name: getattr(source_payload, name)
            for name in (
                "provider_fingerprint",
                "source_commit",
                "source_artifact_sha256",
                "spec_fingerprint",
                "source_state_sha256",
                "replay_loader_implementation_fingerprint",
                "replay_payload_schema_fingerprint",
            )
        },
        "provider_identity": {
            "recipe": VITURI2024_FULL_PROVIDER_ARTIFACT_PROVIDER_IDENTITY_RECIPE,
            "provider_name": provider_name,
            "provider_implementation_base64": base64.b64encode(
                provider_implementation_bytes
            ).decode("ascii"),
            "provider_code_sha256": provider_code_sha256,
            "fingerprint": expected_provider,
            "target_arrays_excluded": ["interaction_h", "fock", "energies"],
        },
        "source_lineage": {
            "source_state_recipe": VITURI2024_FULL_PROVIDER_ARTIFACT_SOURCE_STATE_RECIPE,
            "geometry_receipt_fingerprint": geometry_receipt_fingerprint,
            "ensemble_receipt_fingerprint": ensemble_receipt_fingerprint,
            "selected_branch_label": selected_branch_label,
            "selected_spin": selected_spin,
            "branch_table_sha256": branch_table_sha256,
            "branch_evidence_status": "declared_digest_only_not_verified",
            "lineage_status": "declared_and_content_bound_not_source_closed",
        },
        "conventions": {
            "density": "P_ij=<c_j_dagger c_i>; Q=P-R",
            "stored_density_conversion": "P_ab=rho_ba",
            "operator_conversion": "h0_and_F_blocks_not_transposed",
            "orbital_order": "flat=flavor*Nk+k",
            "mesh_units": "inverse_angstrom",
            "energy_units": "eV",
            "area_units": "angstrom_squared",
            "reference_units": "dimensionless",
            "interaction_area_normalization": "divide_by_area_exactly_once",
            "callback_energy": "raw_total_energy_unweighted_full_trace",
            "active_states": "exact_source_gauge_bytes_no_rediagonalization",
            "momentum_policy": "literal_zero_no_tolerance_no_wrap_no_reciprocal_carry",
        },
        "normal_reference": {
            "array_name": "normal_order_reference_full",
            "kind": normal_reference_kind,
            "role": "candidate_functional_subtraction_R",
            "evidence_text": normal_reference_evidence_text,
            "evidence_sha256": reference_evidence_sha,
            "fingerprint": reference_fingerprint,
            "bytes_bound": True,
            "normal_order_authority_established": False,
            "physical_neutral_reference_identified": False,
            "replay_source_normal_reference_authority": False,
        },
        "area": {
            "value": area,
            "units": "angstrom_squared",
            "normalization": "interaction_divided_by_area_exactly_once",
            "evidence_text": area_evidence_text,
            "evidence_sha256": area_evidence_sha,
            "source_geometry_authority_established": False,
        },
        "interaction": {
            "receipt": interaction_document,
            "fingerprint": interaction_receipt.fingerprint,
        },
        "q0": {
            "kernel_evaluation": interaction_receipt.q0_evaluation,
            "background_status": q0_background_status,
            "evidence_text": q0_background_evidence_text,
            "evidence_sha256": q0_evidence_sha,
            "policy_fingerprint": q0_policy_fingerprint,
            "kernel_limit_is_not_hf_background": True,
            "background_authority_established": False,
        },
        "diagnostics": asdict(diagnostics),
        "runtime_expectations": {
            "target_bearing_payload_fingerprint": bridge.source_payload_fingerprint,
            "target_bearing_supplied_array_receipt_fingerprint": (
                bridge.array_consistency.fingerprint
            ),
            "target_bearing_bridge_fingerprint": bridge.fingerprint,
            "lineage_bound_kernel_fingerprint": bridge.kernel.fingerprint,
            "kernel_scope": (
                "lineage_bound_potentially_target_transitive_not_target_independence_proof"
            ),
            "saved_target_array_excluding_source_input_fingerprint": (
                bridge.source_input_fingerprint
            ),
            "source_input_scope": VITURI2024_FULL_PROVIDER_ARTIFACT_TARGET_EXCLUSION_SCOPE,
            "callback_execution_input_fingerprint": (
                bridge.inputs.value("execution_input_fingerprint")
            ),
            "callback_input_manifest_fingerprint": bridge.inputs.fingerprint,
            "binding_fingerprint": bridge.binding.fingerprint,
            "interaction_fingerprint": interaction_receipt.fingerprint,
        },
        "provenance": provenance,
        "authority": {
            "artifact_bytes_and_schema_verified": False,
            "provider_candidate": artifact_kind == "provider_candidate",
            "source_closure_established": False,
            "source_generation_functional_established": False,
            "source_stationarity_established": False,
            "normal_order_authority_established": False,
            "q0_background_authority_established": False,
            "tdhf_hessian_match": False,
            "eligible_for_slurm_qualification": False,
            "production_ready": False,
            "paper_reproduction_verified": False,
        },
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    expectation = Vituri2024FullProviderArtifactExpectation(
        manifest_file_name=manifest_file_name,
        arrays_file_name=arrays_file_name,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        manifest_size_bytes=len(manifest_bytes),
        arrays_sha256=sha256(arrays_bytes).hexdigest(),
        arrays_size_bytes=len(arrays_bytes),
    )
    return Vituri2024SerializedFullProviderArtifact(
        manifest_bytes=manifest_bytes,
        arrays_bytes=arrays_bytes,
        expectation=expectation,
        bridge_fingerprint=bridge.fingerprint,
        source_input_fingerprint=bridge.source_input_fingerprint,
    )


def _open_directory_without_symlinks(path: Path) -> int:
    if not path.is_absolute():
        raise ValueError("artifact paths must be absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            if part in ("", ".", ".."):
                raise ValueError("artifact directory contains forbidden component")
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_at(parent: Path, name: str, expected_size: int, maximum: int) -> tuple[bytes, os.stat_result]:
    directory_fd = _open_directory_without_symlinks(parent)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"artifact file {name!r} is not regular")
            if before.st_size != expected_size or before.st_size < 1 or before.st_size > maximum:
                raise ValueError(f"artifact file {name!r} size differs from detached expectation")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"artifact file {name!r} ended early")
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError(f"artifact file {name!r} changed while reading")
            return b"".join(chunks), before
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_fd)


def _decode_npz(raw: bytes, descriptors: list[object]) -> dict[str, Array]:
    expected_members = tuple(f"{name}.npy" for name in VITURI2024_FULL_PROVIDER_ARTIFACT_ARRAY_ORDER)
    with zipfile.ZipFile(io.BytesIO(raw), mode="r") as archive:
        infos = archive.infolist()
        if tuple(info.filename for info in infos) != expected_members:
            raise ValueError("artifact NPZ member inventory/order is not canonical")
        if archive.comment != b"":
            raise ValueError("artifact NPZ comment is forbidden")
        member_bytes: dict[str, bytes] = {}
        for info in infos:
            if (
                info.compress_type != zipfile.ZIP_STORED
                or info.date_time != (1980, 1, 1, 0, 0, 0)
                or info.extra != b""
                or info.comment != b""
                or info.create_system != 3
                or (info.external_attr >> 16) != (stat.S_IFREG | 0o444)
            ):
                raise ValueError("artifact NPZ member metadata is not canonical")
            member_bytes[info.filename[:-4]] = archive.read(info)
    if type(descriptors) is not list or len(descriptors) != len(expected_members):
        raise ValueError("artifact array descriptors are incomplete")
    arrays: dict[str, Array] = {}
    for expected_name, raw_descriptor in zip(
        VITURI2024_FULL_PROVIDER_ARTIFACT_ARRAY_ORDER, descriptors, strict=True
    ):
        descriptor = _exact_keys(
            raw_descriptor,
            (
                "name",
                "member",
                "dtype",
                "shape",
                "nbytes",
                "array_sha256",
                "canonical_array_sha256",
                "member_sha256",
            ),
            f"array descriptor {expected_name}",
        )
        if descriptor["name"] != expected_name or descriptor["member"] != f"{expected_name}.npy":
            raise ValueError("artifact array descriptor order/name drifted")
        member = member_bytes[expected_name]
        if not hmac.compare_digest(sha256(member).hexdigest(), _sha256(descriptor["member_sha256"], "member hash")):
            raise ValueError("artifact NPY member hash mismatch")
        try:
            value = np.load(io.BytesIO(member), allow_pickle=False)
        except (ValueError, TypeError) as error:
            raise ValueError("artifact NPY member is invalid or object-bearing") from error
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ValueError("artifact NPY member must be a plain non-object array")
        if _npy_bytes(value) != member:
            raise ValueError("artifact NPY member bytes are not canonical")
        if (
            type(descriptor["dtype"]) is not str
            or value.dtype.str != descriptor["dtype"]
            or type(descriptor["shape"]) is not list
            or any(type(item) is not int for item in descriptor["shape"])
            or list(value.shape) != descriptor["shape"]
            or value.nbytes != _strict_int(descriptor["nbytes"], "array nbytes", positive=True)
            or not np.all(np.isfinite(value))
            or not hmac.compare_digest(_array_sha256(value), _sha256(descriptor["array_sha256"], "array hash"))
            or not hmac.compare_digest(
                canonical_array_sha256(value),
                _sha256(descriptor["canonical_array_sha256"], "canonical array hash"),
            )
        ):
            raise ValueError(f"artifact array {expected_name} descriptor/data mismatch")
        immutable = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
        immutable.setflags(write=False)
        arrays[expected_name] = immutable
    regenerated, _ = _deterministic_npz(arrays)
    if regenerated != raw:
        raise ValueError("artifact NPZ bytes are not deterministic canonical bytes")
    return arrays


@dataclass(frozen=True, slots=True)
class Vituri2024LoadedFullProviderArtifact:
    _factory_token: InitVar[object]
    expectation: Vituri2024FullProviderArtifactExpectation
    source_payload: Vituri2024HalfMetalHFReplayPayload
    normal_order_reference_full: Array
    interaction: Vituri2024InteractionChoiceReceipt
    bridge: Vituri2024FullFunctionalReplayBridge
    diagnostics: Vituri2024FullProviderArtifactDiagnostics
    manifest_sha256: str
    arrays_sha256: str
    artifact_kind: ArtifactKind
    selected_branch_label: str
    selected_spin: int
    construction_fingerprint: str = field(init=False)
    status: str = field(
        default="immutable_candidate_bytes_loaded_absolute_bridge_passed_not_source_closed",
        init=False,
    )
    authority: str = field(default=VITURI2024_FULL_PROVIDER_ARTIFACT_AUTHORITY, init=False)
    artifact_bytes_and_schema_verified: bool = field(default=True, init=False)
    payload_reconstructed: bool = field(default=True, init=False)
    normal_reference_bytes_bound: bool = field(default=True, init=False)
    source_lineage_declared_and_content_bound_not_source_closed: bool = field(default=True, init=False)
    provider_candidate: bool = field(init=False)
    source_closure_established: bool = field(default=False, init=False)
    source_generation_functional_established: bool = field(default=False, init=False)
    source_stationarity_established: bool = field(default=False, init=False)
    normal_order_authority_established: bool = field(default=False, init=False)
    q0_background_authority_established: bool = field(default=False, init=False)
    tdhf_hessian_match: bool = field(default=False, init=False)
    eligible_for_slurm_qualification: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _LOADED_ARTIFACT_TOKEN:
            raise TypeError("loaded full-provider artifact is loader-factory-only")
        object.__setattr__(self, "provider_candidate", self.artifact_kind == "provider_candidate")
        self._validate_fields(check_construction=False)
        object.__setattr__(self, "construction_fingerprint", self._current_fingerprint())

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                "expectation": self.expectation.fingerprint,
                "payload": self.bridge.source_payload_fingerprint,
                "reference": _array_sha256(self.normal_order_reference_full),
                "interaction": self.interaction.fingerprint,
                "bridge": self.bridge.fingerprint,
                "diagnostics": self.diagnostics.fingerprint,
                "manifest_sha256": self.manifest_sha256,
                "arrays_sha256": self.arrays_sha256,
                "artifact_kind": self.artifact_kind,
                "selected_branch_label": self.selected_branch_label,
                "selected_spin": self.selected_spin,
                "status": self.status,
                "authority": self.authority,
                "authority_flags": tuple(
                    getattr(self, item.name)
                    for item in fields(self)
                    if item.name.endswith("established")
                    or item.name
                    in (
                        "artifact_bytes_and_schema_verified",
                        "payload_reconstructed",
                        "normal_reference_bytes_bound",
                        "source_lineage_declared_and_content_bound_not_source_closed",
                        "provider_candidate",
                        "tdhf_hessian_match",
                        "eligible_for_slurm_qualification",
                        "production_ready",
                        "paper_reproduction_verified",
                    )
                ),
            }
        )

    def _validate_fields(self, *, check_construction: bool) -> None:
        if type(self.expectation) is not Vituri2024FullProviderArtifactExpectation:
            raise TypeError("loaded artifact requires detached expectation")
        if type(self.source_payload) is not Vituri2024HalfMetalHFReplayPayload:
            raise TypeError("loaded artifact requires exact replay payload")
        if type(self.interaction) is not Vituri2024InteractionChoiceReceipt:
            raise TypeError("loaded artifact requires exact interaction receipt")
        if type(self.bridge) is not Vituri2024FullFunctionalReplayBridge:
            raise TypeError("loaded artifact requires exact replay bridge")
        if type(self.diagnostics) is not Vituri2024FullProviderArtifactDiagnostics:
            raise TypeError("loaded artifact requires exact diagnostics")
        self.bridge.validate_live_state()
        live_payload_fingerprint = _fingerprint(
            {
                item.name: (
                    _array_sha256(getattr(self.source_payload, item.name))
                    if isinstance(getattr(self.source_payload, item.name), np.ndarray)
                    else getattr(self.source_payload, item.name)
                )
                for item in fields(self.source_payload)
            }
        )
        if live_payload_fingerprint != self.bridge.source_payload_fingerprint:
            raise ValueError("loaded artifact live payload/bridge binding drifted")
        if _payload_diagnostics(self.source_payload).fingerprint != self.diagnostics.fingerprint:
            raise ValueError("loaded artifact live payload diagnostics drifted")
        _sha256(self.manifest_sha256, "loaded manifest")
        _sha256(self.arrays_sha256, "loaded arrays")
        if (
            not hmac.compare_digest(self.manifest_sha256, self.expectation.manifest_sha256)
            or not hmac.compare_digest(self.arrays_sha256, self.expectation.arrays_sha256)
            or _array_sha256(self.normal_order_reference_full)
            != self.bridge.normal_order_reference_array_sha256
        ):
            raise ValueError("loaded artifact bytes/reference binding drifted")
        positives = (
            self.artifact_bytes_and_schema_verified,
            self.payload_reconstructed,
            self.normal_reference_bytes_bound,
            self.source_lineage_declared_and_content_bound_not_source_closed,
            self.provider_candidate is (self.artifact_kind == "provider_candidate"),
        )
        negatives = (
            self.source_closure_established,
            self.source_generation_functional_established,
            self.source_stationarity_established,
            self.normal_order_authority_established,
            self.q0_background_authority_established,
            self.tdhf_hessian_match,
            self.eligible_for_slurm_qualification,
            self.production_ready,
            self.paper_reproduction_verified,
        )
        if not all(item is True for item in positives) or not all(item is False for item in negatives):
            raise ValueError("loaded artifact authority was inflated")
        if check_construction and self._current_fingerprint() != self.construction_fingerprint:
            raise ValueError("loaded artifact construction drifted")

    def validate_live_state(self) -> None:
        self._validate_fields(check_construction=True)

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.construction_fingerprint


def load_vituri2024_full_provider_artifact(
    *, manifest_path: str | Path, arrays_path: str | Path,
    expected: Vituri2024FullProviderArtifactExpectation,
) -> Vituri2024LoadedFullProviderArtifact:
    if type(expected) is not Vituri2024FullProviderArtifactExpectation:
        raise TypeError("artifact loader requires exact detached expectation")
    expected.validate_live_state()
    manifest = Path(manifest_path)
    arrays_file = Path(arrays_path)
    if manifest.parent != arrays_file.parent:
        raise ValueError("artifact manifest and arrays must share one directory")
    if manifest.name != expected.manifest_file_name or arrays_file.name != expected.arrays_file_name:
        raise ValueError("artifact file names differ from detached expectation")
    manifest_raw, manifest_stat = _read_regular_at(
        manifest.parent,
        manifest.name,
        expected.manifest_size_bytes,
        VITURI2024_FULL_PROVIDER_ARTIFACT_MAX_MANIFEST_BYTES,
    )
    arrays_raw, arrays_stat = _read_regular_at(
        arrays_file.parent,
        arrays_file.name,
        expected.arrays_size_bytes,
        VITURI2024_FULL_PROVIDER_ARTIFACT_MAX_NPZ_BYTES,
    )
    if (manifest_stat.st_dev, manifest_stat.st_ino) == (arrays_stat.st_dev, arrays_stat.st_ino):
        raise ValueError("artifact manifest/arrays cannot be hardlink aliases")
    if not hmac.compare_digest(sha256(manifest_raw).hexdigest(), expected.manifest_sha256):
        raise ValueError("artifact manifest differs from detached expectation")
    if not hmac.compare_digest(sha256(arrays_raw).hexdigest(), expected.arrays_sha256):
        raise ValueError("artifact arrays differ from detached expectation")
    document = _parse_canonical_json(manifest_raw)
    root = _exact_keys(
        document,
        (
            "schema", "artifact_kind", "artifact_relationship", "json_profile",
            "npz_profile", "runtime_identity_scope", "files", "arrays",
            "payload_metadata", "provider_identity", "source_lineage", "conventions",
            "normal_reference", "area", "interaction", "q0", "diagnostics",
            "runtime_expectations", "provenance", "authority",
        ),
        "artifact manifest",
    )
    fixed = (
        (root["schema"], VITURI2024_FULL_PROVIDER_ARTIFACT_SCHEMA),
        (root["artifact_relationship"], VITURI2024_FULL_PROVIDER_ARTIFACT_RELATIONSHIP),
        (root["json_profile"], VITURI2024_FULL_PROVIDER_ARTIFACT_JSON_PROFILE),
        (root["npz_profile"], VITURI2024_FULL_PROVIDER_ARTIFACT_NPZ_PROFILE),
        (root["runtime_identity_scope"], VITURI2024_FULL_PROVIDER_ARTIFACT_RUNTIME_SCOPE),
    )
    if any(actual != wanted for actual, wanted in fixed):
        raise ValueError("artifact manifest fixed schema/profile field drifted")
    artifact_kind = root["artifact_kind"]
    if artifact_kind not in ("synthetic_fixture", "provider_candidate"):
        raise ValueError("artifact manifest kind is invalid")
    files_document = _exact_keys(
        root["files"],
        ("arrays_file_name", "arrays_sha256", "arrays_size_bytes"),
        "artifact files",
    )
    if (
        files_document["arrays_file_name"] != expected.arrays_file_name
        or files_document["arrays_sha256"] != expected.arrays_sha256
        or _strict_int(files_document["arrays_size_bytes"], "manifest arrays size", positive=True)
        != expected.arrays_size_bytes
    ):
        raise ValueError("artifact manifest arrays file binding differs from expectation")
    arrays = _decode_npz(arrays_raw, root["arrays"])
    metadata = _exact_keys(
        root["payload_metadata"],
        (
            "provider_fingerprint", "source_commit", "source_artifact_sha256",
            "spec_fingerprint", "source_state_sha256",
            "replay_loader_implementation_fingerprint", "replay_payload_schema_fingerprint",
        ),
        "payload metadata",
    )
    payload = Vituri2024HalfMetalHFReplayPayload(
        **metadata,
        mesh=arrays["mesh"],
        active_band_states=arrays["active_band_states"],
        h0=arrays["h0"],
        interaction_h=arrays["interaction_h"],
        fock=arrays["fock"],
        projector=arrays["projector"],
        energies=arrays["energies"],
        occupations=arrays["occupations"],
    )
    provider_document = _exact_keys(
        root["provider_identity"],
        (
            "recipe", "provider_name", "provider_implementation_base64",
            "provider_code_sha256", "fingerprint", "target_arrays_excluded",
        ),
        "provider identity",
    )
    _strict_text(provider_document["provider_name"], "artifact provider name")
    if provider_document["recipe"] != VITURI2024_FULL_PROVIDER_ARTIFACT_PROVIDER_IDENTITY_RECIPE or provider_document["target_arrays_excluded"] != ["interaction_h", "fock", "energies"]:
        raise ValueError("artifact provider identity recipe drifted")
    try:
        provider_implementation_bytes = base64.b64decode(
            provider_document["provider_implementation_base64"], validate=True
        )
    except (TypeError, ValueError) as error:
        raise ValueError("artifact provider implementation base64 is invalid") from error
    if sha256(provider_implementation_bytes).hexdigest() != provider_document["provider_code_sha256"]:
        raise ValueError("artifact provider implementation content/hash mismatch")
    provider_fingerprint = vituri2024_full_provider_target_free_identity_fingerprint()
    if provider_fingerprint != provider_document["fingerprint"] or provider_fingerprint != payload.provider_fingerprint:
        raise ValueError("artifact target-free provider identity does not reconstruct")
    lineage = _exact_keys(
        root["source_lineage"],
        (
            "source_state_recipe", "geometry_receipt_fingerprint", "ensemble_receipt_fingerprint",
            "selected_branch_label", "selected_spin", "branch_table_sha256",
            "branch_evidence_status", "lineage_status",
        ),
        "source lineage",
    )
    if (
        lineage["source_state_recipe"] != VITURI2024_FULL_PROVIDER_ARTIFACT_SOURCE_STATE_RECIPE
        or lineage["branch_evidence_status"] != "declared_digest_only_not_verified"
        or lineage["lineage_status"] != "declared_and_content_bound_not_source_closed"
    ):
        raise ValueError("artifact source-lineage status drifted")
    _sha256(lineage["geometry_receipt_fingerprint"], "geometry receipt")
    _sha256(lineage["ensemble_receipt_fingerprint"], "ensemble receipt")
    _sha256(lineage["branch_table_sha256"], "branch table")
    if type(lineage["selected_spin"]) is not int or lineage["selected_spin"] not in (-1, 1):
        raise ValueError("artifact selected spin is invalid")
    _strict_text(lineage["selected_branch_label"], "selected branch")
    reconstructed_state = vituri2024_full_provider_artifact_source_state_fingerprint(
        payload=payload,
        geometry_receipt_fingerprint=lineage["geometry_receipt_fingerprint"],
        ensemble_receipt_fingerprint=lineage["ensemble_receipt_fingerprint"],
    )
    if reconstructed_state != payload.source_state_sha256:
        raise ValueError("artifact source-state context does not reconstruct")
    expected_conventions = {
        "density": "P_ij=<c_j_dagger c_i>; Q=P-R",
        "stored_density_conversion": "P_ab=rho_ba",
        "operator_conversion": "h0_and_F_blocks_not_transposed",
        "orbital_order": "flat=flavor*Nk+k",
        "mesh_units": "inverse_angstrom",
        "energy_units": "eV",
        "area_units": "angstrom_squared",
        "reference_units": "dimensionless",
        "interaction_area_normalization": "divide_by_area_exactly_once",
        "callback_energy": "raw_total_energy_unweighted_full_trace",
        "active_states": "exact_source_gauge_bytes_no_rediagonalization",
        "momentum_policy": "literal_zero_no_tolerance_no_wrap_no_reciprocal_carry",
    }
    if root["conventions"] != expected_conventions:
        raise ValueError("artifact physical convention manifest drifted")
    reference_document = _exact_keys(
        root["normal_reference"],
        (
            "array_name", "kind", "role", "evidence_text", "evidence_sha256",
            "fingerprint", "bytes_bound", "normal_order_authority_established",
            "physical_neutral_reference_identified", "replay_source_normal_reference_authority",
        ),
        "normal reference",
    )
    if reference_document["array_name"] != "normal_order_reference_full" or reference_document["role"] != "candidate_functional_subtraction_R" or reference_document["bytes_bound"] is not True or any(reference_document[name] is not False for name in ("normal_order_authority_established", "physical_neutral_reference_identified", "replay_source_normal_reference_authority")):
        raise ValueError("artifact normal-reference authority/status drifted")
    reference_kind = reference_document["kind"]
    if reference_kind not in ("paper_projected_R0_representative_only", "provider_supplied_explicit_R_unqualified"):
        raise ValueError("artifact normal-reference kind is invalid")
    reference_evidence = _strict_text(reference_document["evidence_text"], "reference evidence")
    reference_evidence_sha = sha256(reference_evidence.encode()).hexdigest()
    if reference_evidence_sha != reference_document["evidence_sha256"]:
        raise ValueError("artifact normal-reference evidence content/hash mismatch")
    reference = _reference_array(arrays["normal_order_reference_full"], 4 * payload.mesh.shape[0])
    if reference_kind == "paper_projected_R0_representative_only" and np.any(reference != 0.0):
        raise ValueError("artifact paper R0 kind contains nonzero R")
    reference_fingerprint = _normal_reference_fingerprint(
        reference=reference,
        kind=reference_kind,
        evidence_sha256=reference_evidence_sha,
    )
    if reference_fingerprint != reference_document["fingerprint"]:
        raise ValueError("artifact normal-reference fingerprint does not reconstruct")
    area_document = _exact_keys(
        root["area"],
        ("value", "units", "normalization", "evidence_text", "evidence_sha256", "source_geometry_authority_established"),
        "area",
    )
    area = _strict_float(area_document["value"], "artifact area", positive=True)
    area_evidence = _strict_text(area_document["evidence_text"], "area evidence")
    if area_document["units"] != "angstrom_squared" or area_document["normalization"] != "interaction_divided_by_area_exactly_once" or area_document["source_geometry_authority_established"] is not False or sha256(area_evidence.encode()).hexdigest() != area_document["evidence_sha256"]:
        raise ValueError("artifact area semantics/evidence drifted")
    interaction_document = _exact_keys(root["interaction"], ("receipt", "fingerprint"), "interaction")
    receipt_data = _exact_keys(
        interaction_document["receipt"],
        VITURI2024_FULL_PROVIDER_ARTIFACT_INTERACTION_KEYS,
        "interaction receipt",
    )
    interaction_receipt = Vituri2024InteractionChoiceReceipt(
        gate_distance_angstrom=receipt_data["gate_distance_angstrom"],
        coulomb_e2_ev_angstrom=receipt_data["coulomb_e2_ev_angstrom"],
        q0_evaluation=receipt_data["q0_evaluation"],
        provider_sha256=receipt_data["provider_sha256"],
        source_sha256=receipt_data["source_sha256"],
        authority_kind=receipt_data["authority_kind"],
        source_text=receipt_data["source_text"],
    )
    _require_exact_typed_equal(
        receipt_data, asdict(interaction_receipt), "interaction receipt"
    )
    if interaction_receipt.fingerprint != interaction_document["fingerprint"]:
        raise ValueError("artifact interaction receipt fingerprint does not reconstruct")
    q0_document = _exact_keys(
        root["q0"],
        ("kernel_evaluation", "background_status", "evidence_text", "evidence_sha256", "policy_fingerprint", "kernel_limit_is_not_hf_background", "background_authority_established"),
        "q0",
    )
    if q0_document["background_status"] not in ("absent", "declared_evidence_bound_not_executable") or q0_document["kernel_evaluation"] != interaction_receipt.q0_evaluation or q0_document["kernel_limit_is_not_hf_background"] is not True or q0_document["background_authority_established"] is not False:
        raise ValueError("artifact q0 status/authority drifted")
    q0_evidence = _strict_text(q0_document["evidence_text"], "q0 evidence")
    q0_evidence_sha = sha256(q0_evidence.encode()).hexdigest()
    q0_policy = _q0_policy_fingerprint(
        interaction=interaction_receipt,
        status=q0_document["background_status"],
        evidence_sha256=q0_evidence_sha,
    )
    if q0_evidence_sha != q0_document["evidence_sha256"] or q0_policy != q0_document["policy_fingerprint"]:
        raise ValueError("artifact q0 evidence/policy fingerprint does not reconstruct")
    diagnostics = _payload_diagnostics(payload)
    _require_exact_typed_equal(
        root["diagnostics"], asdict(diagnostics), "payload diagnostics"
    )
    provenance = _strict_text(root["provenance"], "artifact provenance")
    bridge = build_vituri2024_full_functional_replay_bridge(
        source_payload=payload,
        normal_order_reference_full=reference,
        area_angstrom_squared=area,
        interaction=interaction_receipt,
        normal_order_reference_fingerprint=reference_fingerprint,
        reference_policy_evidence_sha256=reference_evidence_sha,
        q0_policy_fingerprint=q0_policy,
        q0_background_evidence_sha256=q0_evidence_sha,
        provenance=provenance,
    )
    runtime = _exact_keys(
        root["runtime_expectations"],
        (
            "target_bearing_payload_fingerprint", "target_bearing_supplied_array_receipt_fingerprint",
            "target_bearing_bridge_fingerprint", "lineage_bound_kernel_fingerprint", "kernel_scope",
            "saved_target_array_excluding_source_input_fingerprint", "source_input_scope",
            "callback_execution_input_fingerprint", "callback_input_manifest_fingerprint",
            "binding_fingerprint", "interaction_fingerprint",
        ),
        "runtime expectations",
    )
    computed_runtime = {
        "target_bearing_payload_fingerprint": bridge.source_payload_fingerprint,
        "target_bearing_supplied_array_receipt_fingerprint": bridge.array_consistency.fingerprint,
        "target_bearing_bridge_fingerprint": bridge.fingerprint,
        "lineage_bound_kernel_fingerprint": bridge.kernel.fingerprint,
        "kernel_scope": "lineage_bound_potentially_target_transitive_not_target_independence_proof",
        "saved_target_array_excluding_source_input_fingerprint": bridge.source_input_fingerprint,
        "source_input_scope": VITURI2024_FULL_PROVIDER_ARTIFACT_TARGET_EXCLUSION_SCOPE,
        "callback_execution_input_fingerprint": bridge.inputs.value("execution_input_fingerprint"),
        "callback_input_manifest_fingerprint": bridge.inputs.fingerprint,
        "binding_fingerprint": bridge.binding.fingerprint,
        "interaction_fingerprint": interaction_receipt.fingerprint,
    }
    for name, value in computed_runtime.items():
        if type(value) is str and len(value) == 64:
            if type(runtime[name]) is not str or not hmac.compare_digest(value, runtime[name]):
                raise ValueError(f"artifact runtime fingerprint {name} drifted")
        elif runtime[name] != value:
            raise ValueError(f"artifact runtime scope {name} drifted")
    authority = _exact_keys(
        root["authority"],
        (
            "artifact_bytes_and_schema_verified", "provider_candidate", "source_closure_established",
            "source_generation_functional_established", "source_stationarity_established",
            "normal_order_authority_established", "q0_background_authority_established",
            "tdhf_hessian_match", "eligible_for_slurm_qualification", "production_ready",
            "paper_reproduction_verified",
        ),
        "authority",
    )
    expected_provider_candidate = artifact_kind == "provider_candidate"
    if (
        authority["artifact_bytes_and_schema_verified"] is not False
        or authority["provider_candidate"] is not expected_provider_candidate
        or any(
            authority[name] is not False
            for name in authority
            if name not in ("artifact_bytes_and_schema_verified", "provider_candidate")
        )
    ):
        raise ValueError("artifact manifest authority was inflated")
    return Vituri2024LoadedFullProviderArtifact(
        _factory_token=_LOADED_ARTIFACT_TOKEN,
        expectation=expected,
        source_payload=payload,
        normal_order_reference_full=reference,
        interaction=interaction_receipt,
        bridge=bridge,
        diagnostics=diagnostics,
        manifest_sha256=sha256(manifest_raw).hexdigest(),
        arrays_sha256=sha256(arrays_raw).hexdigest(),
        artifact_kind=artifact_kind,
        selected_branch_label=lineage["selected_branch_label"],
        selected_spin=lineage["selected_spin"],
    )


__all__ = [
    "VITURI2024_FULL_PROVIDER_ARTIFACT_ARRAY_ORDER",
    "VITURI2024_FULL_PROVIDER_ARTIFACT_AUTHORITY",
    "VITURI2024_FULL_PROVIDER_ARTIFACT_JSON_PROFILE",
    "VITURI2024_FULL_PROVIDER_ARTIFACT_NPZ_PROFILE",
    "VITURI2024_FULL_PROVIDER_ARTIFACT_PROVIDER_IDENTITY_RECIPE",
    "VITURI2024_FULL_PROVIDER_ARTIFACT_RELATIONSHIP",
    "VITURI2024_FULL_PROVIDER_ARTIFACT_RUNTIME_SCOPE",
    "VITURI2024_FULL_PROVIDER_ARTIFACT_SCHEMA",
    "VITURI2024_FULL_PROVIDER_ARTIFACT_SOURCE_STATE_RECIPE",
    "Vituri2024FullProviderArtifactDiagnostics",
    "Vituri2024FullProviderArtifactExpectation",
    "Vituri2024LoadedFullProviderArtifact",
    "Vituri2024SerializedFullProviderArtifact",
    "load_vituri2024_full_provider_artifact",
    "serialize_vituri2024_full_provider_artifact_candidate",
    "vituri2024_full_provider_artifact_source_state_fingerprint",
    "vituri2024_full_provider_saved_target_excluding_source_input_fingerprint",
    "vituri2024_full_provider_target_free_identity_fingerprint",
]
