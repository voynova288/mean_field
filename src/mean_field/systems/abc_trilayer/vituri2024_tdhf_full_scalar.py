"""Fail-closed Vituri full-projector scalar-functional candidate adapter.

This module only binds a candidate to the exact current Vituri readiness chain
and to the generic conventional dense ABI.  Static success means
``candidate_bound_not_executed``.  A later generic consistency receipt may be
consumed, but neither path compares TDHF A/B or ``H+`` nor promotes
scalar-Hessian, production, readiness, or paper authority.  Synthetic evidence
is factory-only and never Slurm eligible; this repository supplies no concrete
immutable artifact-authority callback.

The replay payload stores ``rho_ab=<c_a^dagger c_b>`` while the generic ABI uses
``P_ij=<c_j^dagger c_i>``.  Therefore each payload projector block is
transposed into ``P``.  The payload Fock matrix is the one-body operator matrix
entering ``Tr(F P)`` and is embedded without transpose.  This storage duality,
the flavor-major flat map ``flat=flavor*Nk+k``, and raw total-energy/full-trace
normalization are fingerprinted below.  Native replay energy is per-k, so the
raw total multiplies by exactly ``Nk`` and never silently divides by area.
Callback/dependency/byte snapshots remain a trusted-provider boundary, not a
hostile-code or global-completeness proof.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Callable, Final, Literal, Sequence

import numpy as np

from mean_field.core.hf.tdhf_scalar_functional import (
    TDHF_FULL_PROJECTOR_ENERGY_NORMALIZATION,
    TDHFFullProjectorFunctionalApproval,
    TDHFFullProjectorFunctionalBinding,
    TDHFFullProjectorFunctionalEvidenceReceipt,
    TDHFFullProjectorSpace,
    TDHFFullProjectorValidationPlan,
    TDHFScalarFunctionalInputsManifest,
    make_tdhf_full_projector_functional_approval,
)

from .vituri2024_hf_pocket_replay import Vituri2024PocketRefinementPrerequisites
from .vituri2024_hf_preflight import (
    INTERNAL_FLAVOR_ORDER,
    REPLAY_ARRAY_CONVERSION,
    REPLAY_ARRAY_LAYOUT,
    REPLAY_ORBITAL_ORDER,
)
from .vituri2024_hf_replay import (
    Vituri2024HalfMetalHFReplayPayload,
    canonical_array_sha256,
    canonical_orbital_order_sha256,
)
from .vituri2024_tdhf import Vituri2024TDHFSignedQAssemblyReceipt
from .vituri2024_tdhf_scalar import Vituri2024TDHFScalarReadinessReceipt

Array = np.ndarray
CandidateEvidenceKind = Literal["synthetic_fixture", "immutable_provider_artifact"]
EvidenceArtifactRelationship = Literal[
    "source_artifact", "lineage_bound_evidence_artifact"
]

VITURI2024_FULL_SCALAR_API_VERSION: Final[str] = (
    "vituri2024_tdhf_full_projector_scalar_candidate.v1"
)
VITURI2024_FULL_SCALAR_PREFLIGHT_STATUS: Final[str] = (
    "candidate_bound_not_executed"
)
VITURI2024_FULL_SCALAR_STORAGE_DUALITY: Final[str] = (
    "payload_rho_ab=<c_a^dagger c_b>; conventional_P_ij=payload_rho_ji; "
    "payload_Fock_ij_is_operator_matrix; flat=flavor*Nk+k"
)
VITURI2024_FULL_SCALAR_RUNTIME_LAYOUT: Final[str] = (
    "arbitrary_full_dense_conventional_N_by_N"
)
VITURI2024_FULL_SCALAR_AFFINE_SUPPORT: Final[str] = (
    "arbitrary_full_dense_affine_hermitian"
)
VITURI2024_FULL_SCALAR_EXACT_UNITARY_SUPPORT: Final[str] = (
    "preregistered_projector_values_with_executed_energy_and_fock_evidence"
)
VITURI2024_FULL_SCALAR_ONE_BODY_CONSTRUCTION: Final[str] = (
    "provider_native_full_h0_no_fitted_counterterm"
)
VITURI2024_FULL_SCALAR_TOTAL_ENERGY_NORMALIZATION: Final[str] = (
    "raw_total_energy_unweighted_full_trace"
)

VITURI2024_FULL_SCALAR_PHYSICAL_INPUT_NAMES: Final[tuple[str, ...]] = (
    "area_angstrom_squared",
    "form_factor_manifest_sha256",
    "h0_full",
    "interaction_kernel_manifest_sha256",
    "normal_order_reference_fingerprint",
    "ordered_mesh",
    "q0_policy_fingerprint",
    "source_projector_full",
)
VITURI2024_FULL_SCALAR_OFF_K_TOLERANCE: Final[float] = (
    64.0 * np.finfo(np.float64).eps
)

_PREFLIGHT_TOKEN = object()
_QUALIFICATION_TOKEN = object()
_EVIDENCE_TOKEN = object()
_NORMALIZATION_TOKEN = object()


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


def _stable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": value.shape,
            "sha256": _array_sha256(value),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _stable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {
            str(key): _stable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, np.generic):
        return _stable(value.item())
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _fingerprint(value: object) -> str:
    return sha256(
        json.dumps(
            _stable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be explicit nonempty text")
    return value


def _readonly_matrix(value: object, *, label: str, dimension: int) -> Array:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.complex128):
        raise TypeError(f"{label} must be an exact complex128 numpy array")
    if value.shape != (dimension, dimension) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite with full shape {(dimension, dimension)}")
    residual = float(np.max(np.abs(value - value.conj().T)))
    scale = max(1.0, float(np.max(np.abs(value))))
    if residual > 64.0 * np.finfo(np.float64).eps * scale:
        raise ValueError(f"{label} must be Hermitian")
    result = np.frombuffer(value.tobytes(order="C"), dtype=np.complex128).reshape(
        value.shape
    )
    result.setflags(write=False)
    return result


def _layout_adapter_fingerprint(nk: int, mesh_sha256: str) -> str:
    return _fingerprint(
        {
            "api_version": VITURI2024_FULL_SCALAR_API_VERSION,
            "number_of_flavors": len(INTERNAL_FLAVOR_ORDER),
            "nk": nk,
            "dimension": len(INTERNAL_FLAVOR_ORDER) * nk,
            "axis_order": ("flavor", "k"),
            "flat_map": "flat=flavor*Nk+k",
            "payload_array_layout": REPLAY_ARRAY_LAYOUT,
            "payload_array_conversion": REPLAY_ARRAY_CONVERSION,
            "payload_orbital_order": REPLAY_ORBITAL_ORDER,
            "ordered_mesh_sha256": mesh_sha256,
            "storage_duality": VITURI2024_FULL_SCALAR_STORAGE_DUALITY,
        }
    )


def _readonly_native_blocks(value: object, *, label: str) -> Array:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.complex128):
        raise TypeError(f"{label} must be an exact complex128 numpy array")
    if (
        value.ndim != 3
        or value.shape[:2]
        != (len(INTERNAL_FLAVOR_ORDER), len(INTERNAL_FLAVOR_ORDER))
        or value.shape[2] < 1
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{label} must have finite shape (4,4,Nk), Nk>0")
    residual = float(np.max(np.abs(value - value.swapaxes(0, 1).conj())))
    scale = max(1.0, float(np.max(np.abs(value))))
    if residual > 64.0 * np.finfo(np.float64).eps * scale:
        raise ValueError(f"{label} blocks must be Hermitian")
    result = np.frombuffer(value.tobytes(order="C"), dtype=np.complex128).reshape(
        value.shape
    )
    result.setflags(write=False)
    return result


def vituri2024_payload_operator_to_full_dense(stored_operator: Array) -> Array:
    """Embed native k-diagonal operator blocks without transpose."""

    blocks = _readonly_native_blocks(stored_operator, label="payload operator")
    nk = blocks.shape[2]
    result = np.zeros((len(INTERNAL_FLAVOR_ORDER) * nk,) * 2, dtype=np.complex128)
    offsets = np.arange(len(INTERNAL_FLAVOR_ORDER), dtype=int) * nk
    for k_index in range(nk):
        flat = offsets + k_index
        result[np.ix_(flat, flat)] = blocks[:, :, k_index]
    result.setflags(write=False)
    return result


def vituri2024_payload_density_to_full_projector(stored_density: Array) -> Array:
    """Embed ``rho_ab=<c_a^dagger c_b>`` as conventional ``P_ij=rho_ji``."""

    blocks = _readonly_native_blocks(stored_density, label="payload stored density")
    nk = blocks.shape[2]
    result = np.zeros((len(INTERNAL_FLAVOR_ORDER) * nk,) * 2, dtype=np.complex128)
    offsets = np.arange(len(INTERNAL_FLAVOR_ORDER), dtype=int) * nk
    for k_index in range(nk):
        flat = offsets + k_index
        result[np.ix_(flat, flat)] = blocks[:, :, k_index].T
    result.setflags(write=False)
    return result


def _full_dense_to_payload_blocks(
    value: Array, *, density: bool, label: str
) -> Array:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.complex128):
        raise TypeError(f"{label} must be an exact complex128 numpy array")
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{label} must be square")
    flavors = len(INTERNAL_FLAVOR_ORDER)
    if value.shape[0] % flavors != 0:
        raise ValueError(f"{label} dimension must equal 4*Nk")
    nk = value.shape[0] // flavors
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite")
    residual = float(np.max(np.abs(value - value.conj().T)))
    scale = max(1.0, float(np.max(np.abs(value))))
    if residual > 64.0 * np.finfo(np.float64).eps * scale:
        raise ValueError(f"{label} must be Hermitian")
    blocks = np.zeros((flavors, flavors, nk), dtype=np.complex128)
    reconstructed = np.zeros_like(value)
    offsets = np.arange(flavors, dtype=int) * nk
    for k_index in range(nk):
        flat = offsets + k_index
        block = value[np.ix_(flat, flat)]
        blocks[:, :, k_index] = block.T if density else block
        reconstructed[np.ix_(flat, flat)] = block
    if np.max(np.abs(value - reconstructed)) > VITURI2024_FULL_SCALAR_OFF_K_TOLERANCE:
        raise ValueError(f"{label} has off-k entries and has no native k-diagonal inverse")
    blocks.setflags(write=False)
    return blocks


def vituri2024_full_projector_to_payload_density(projector: Array) -> Array:
    """Inverse of the native-density embedding for k-diagonal projectors."""

    return _full_dense_to_payload_blocks(
        projector, density=True, label="full conventional projector"
    )


def vituri2024_full_operator_to_payload_k_diagonal(operator: Array) -> Array:
    """Inverse of the native-operator embedding for k-diagonal operators."""

    return _full_dense_to_payload_blocks(
        operator, density=False, label="full conventional operator"
    )


@dataclass(frozen=True, slots=True)
class Vituri2024FullScalarNormalizationReceipt:
    _factory_token: InitVar[object]
    nk: int
    native_pairing_per_k: complex
    raw_full_trace_pairing: complex
    raw_total_factor: int
    orientation_residual: float
    no_area_division: bool
    operator_fingerprint: str
    stored_density_fingerprint: str
    full_operator_fingerprint: str
    full_projector_fingerprint: str

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _NORMALIZATION_TOKEN:
            raise TypeError("normalization receipt requires the private factory token")
        if type(self.nk) is not int or self.nk < 1:
            raise ValueError("normalization receipt Nk must be positive")
        if type(self.raw_total_factor) is not int or self.raw_total_factor != self.nk:
            raise ValueError("native per-k to raw-total factor must be exactly Nk")
        for value in (
            self.native_pairing_per_k.real,
            self.native_pairing_per_k.imag,
            self.raw_full_trace_pairing.real,
            self.raw_full_trace_pairing.imag,
            self.orientation_residual,
        ):
            if not math.isfinite(value):
                raise ValueError("normalization receipt values must be finite")
        if self.orientation_residual < 0.0:
            raise ValueError("normalization residual must be nonnegative")
        if self.no_area_division is not True:
            raise ValueError("full-projector raw total must not divide by area")
        for name in (
            "operator_fingerprint",
            "stored_density_fingerprint",
            "full_operator_fingerprint",
            "full_projector_fingerprint",
        ):
            _sha256(getattr(self, name), f"normalization {name}")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def validate_vituri2024_native_to_raw_total_factor(
    receipt: Vituri2024FullScalarNormalizationReceipt,
    factor: int,
) -> complex:
    """Reject omitted or doubled native-per-k to raw-total normalization."""

    if type(receipt) is not Vituri2024FullScalarNormalizationReceipt:
        raise TypeError("normalization-factor validation requires a factory receipt")
    if type(factor) is not int or factor != receipt.nk:
        raise ValueError("native per-k to raw-total normalization factor must be exactly Nk")
    return receipt.native_pairing_per_k * factor


def certify_vituri2024_full_scalar_orientation_and_normalization(
    stored_operator: Array,
    stored_density: Array,
) -> Vituri2024FullScalarNormalizationReceipt:
    """Prove native ``sum(F_ab rho_ab)/Nk`` equals ``Tr(F_full P_full)/Nk``."""

    operator = _readonly_native_blocks(stored_operator, label="pairing operator")
    density = _readonly_native_blocks(stored_density, label="pairing stored density")
    if operator.shape != density.shape:
        raise ValueError("pairing operator/density native shapes differ")
    nk = operator.shape[2]
    native_raw = complex(np.einsum("abk,abk->", operator, density, optimize=False))
    full_operator = vituri2024_payload_operator_to_full_dense(operator)
    full_projector = vituri2024_payload_density_to_full_projector(density)
    full_trace = complex(
        np.einsum("ij,ji->", full_operator, full_projector, optimize=False)
    )
    residual = abs(native_raw - full_trace)
    scale = max(1.0, abs(native_raw), abs(full_trace))
    if residual > 64.0 * np.finfo(np.float64).eps * scale:
        raise ValueError("native/full complex-offdiagonal orientation oracle failed")
    return Vituri2024FullScalarNormalizationReceipt(
        _factory_token=_NORMALIZATION_TOKEN,
        nk=nk,
        native_pairing_per_k=native_raw / nk,
        raw_full_trace_pairing=full_trace,
        raw_total_factor=nk,
        orientation_residual=residual,
        no_area_division=True,
        operator_fingerprint=_array_sha256(operator),
        stored_density_fingerprint=_array_sha256(density),
        full_operator_fingerprint=_array_sha256(full_operator),
        full_projector_fingerprint=_array_sha256(full_projector),
    )


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFFullScalarSource:
    """Full dense conventional source matrices derived from one replay payload."""

    space: TDHFFullProjectorSpace
    source_projector: Array
    source_h0: Array
    source_fock: Array
    ordered_mesh: Array
    normalization_receipt: Vituri2024FullScalarNormalizationReceipt
    nk: int
    ordered_mesh_sha256: str
    storage_duality: str = field(
        default=VITURI2024_FULL_SCALAR_STORAGE_DUALITY, init=False
    )

    def __post_init__(self) -> None:
        if type(self.space) is not TDHFFullProjectorSpace:
            raise TypeError("Vituri full source requires exact full-projector space")
        if type(self.nk) is not int or self.nk < 1:
            raise ValueError("Vituri full source Nk must be positive")
        if self.space.dimension != len(INTERNAL_FLAVOR_ORDER) * self.nk:
            raise ValueError("Vituri full source dimension must be exactly 4*Nk")
        _sha256(self.ordered_mesh_sha256, "ordered mesh fingerprint")
        p0 = _readonly_matrix(
            self.source_projector,
            label="Vituri conventional full P0",
            dimension=self.space.dimension,
        )
        h0 = _readonly_matrix(
            self.source_h0,
            label="Vituri conventional full h0",
            dimension=self.space.dimension,
        )
        f0 = _readonly_matrix(
            self.source_fock,
            label="Vituri conventional full F0",
            dimension=self.space.dimension,
        )
        if np.max(np.abs(p0 @ p0 - p0)) > 5.0e-10:
            raise ValueError("Vituri conventional full P0 is not a projector")
        if (
            not isinstance(self.ordered_mesh, np.ndarray)
            or self.ordered_mesh.dtype != np.dtype(np.float64)
            or self.ordered_mesh.shape != (self.nk, 2)
            or not np.all(np.isfinite(self.ordered_mesh))
        ):
            raise ValueError("Vituri ordered mesh must be exact finite float64 (Nk,2)")
        mesh = np.frombuffer(
            self.ordered_mesh.tobytes(order="C"), dtype=np.float64
        ).reshape(self.ordered_mesh.shape)
        mesh.setflags(write=False)
        if canonical_array_sha256(mesh) != self.ordered_mesh_sha256:
            raise ValueError("Vituri ordered mesh bytes/fingerprint mismatch")
        if type(self.normalization_receipt) is not Vituri2024FullScalarNormalizationReceipt:
            raise TypeError("Vituri source requires a factory normalization receipt")
        if (
            self.normalization_receipt.nk != self.nk
            or self.normalization_receipt.full_projector_fingerprint
            != _array_sha256(p0)
            or self.normalization_receipt.full_operator_fingerprint
            != _array_sha256(f0)
        ):
            raise ValueError("Vituri source normalization receipt is stale")
        object.__setattr__(self, "source_projector", p0)
        object.__setattr__(self, "source_h0", h0)
        object.__setattr__(self, "source_fock", f0)
        object.__setattr__(self, "ordered_mesh", mesh)
        if self.storage_duality != VITURI2024_FULL_SCALAR_STORAGE_DUALITY:
            raise ValueError("Vituri payload/conventional storage duality changed")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def vituri2024_tdhf_full_scalar_source_from_payload(
    payload: Vituri2024HalfMetalHFReplayPayload,
) -> Vituri2024TDHFFullScalarSource:
    """Embed exact ``(4,4,Nk)`` replay arrays into conventional ``(4Nk)^2`` matrices."""

    if type(payload) is not Vituri2024HalfMetalHFReplayPayload:
        raise TypeError("full scalar source requires exact Vituri replay payload")
    nk = int(payload.mesh.shape[0])
    flavors = len(INTERNAL_FLAVOR_ORDER)
    dimension = flavors * nk
    mesh_sha = canonical_array_sha256(payload.mesh)
    layout_sha = _layout_adapter_fingerprint(nk, mesh_sha)
    space = TDHFFullProjectorSpace(
        dimension=dimension,
        axis_sizes=(flavors, nk),
        axis_order=("flavor", "k"),
        orbital_order_fingerprint=canonical_orbital_order_sha256(payload.mesh),
        layout_adapter_fingerprint=layout_sha,
    )
    p0 = vituri2024_payload_density_to_full_projector(payload.projector)
    h0 = vituri2024_payload_operator_to_full_dense(payload.h0)
    f0 = vituri2024_payload_operator_to_full_dense(payload.fock)
    normalization = certify_vituri2024_full_scalar_orientation_and_normalization(
        payload.fock,
        payload.projector,
    )
    return Vituri2024TDHFFullScalarSource(
        space=space,
        source_projector=p0,
        source_h0=h0,
        source_fock=f0,
        ordered_mesh=payload.mesh,
        normalization_receipt=normalization,
        nk=nk,
        ordered_mesh_sha256=mesh_sha,
    )


def _candidate_callback_dependency_inventory_fingerprint(
    binding: TDHFFullProjectorFunctionalBinding,
) -> str:
    return _fingerprint(
        tuple(
            (
                kernel.manifest.fingerprint,
                tuple(item.manifest.fingerprint for item in kernel.dependencies),
            )
            for kernel in (binding.energy, binding.fock, binding.fock_derivative)
        )
    )


def _code_snapshot_fingerprint(value: object, label: str) -> str:
    if isinstance(value, ModuleType):
        module = value
        qualname = module.__name__
    elif isinstance(value, FunctionType):
        module = inspect.getmodule(value)
        if module is None:
            raise ValueError(f"{label} has no importable module")
        qualname = value.__qualname__
    else:
        raise TypeError(f"{label} must be a Python function or module")
    module_file = inspect.getsourcefile(module) or getattr(module, "__file__", None)
    if not module_file:
        raise ValueError(f"{label} module source file is unavailable")
    try:
        module_bytes = Path(module_file).read_bytes()
        source = inspect.getsource(value).encode()
    except (OSError, TypeError) as error:
        raise ValueError(f"{label} source snapshot is unavailable") from error
    return _fingerprint(
        {
            "qualname": qualname,
            "module_file": str(module_file),
            "module_sha256": sha256(module_bytes).hexdigest(),
            "source_sha256": sha256(source).hexdigest(),
        }
    )


@dataclass(frozen=True, slots=True)
class Vituri2024FullScalarImmutableVerificationRequest:
    lineage_fingerprint: str
    provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    source_state_sha256: str
    selected_branch_label: str
    evidence_artifact_sha256: str
    artifact_relationship: EvidenceArtifactRelationship
    binding_fingerprint: str
    generic_input_manifest_fingerprint: str
    physical_input_inventory_fingerprint: str
    callback_dependency_inventory_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "lineage_fingerprint",
            "provider_fingerprint",
            "source_artifact_sha256",
            "source_state_sha256",
            "evidence_artifact_sha256",
            "binding_fingerprint",
            "generic_input_manifest_fingerprint",
            "physical_input_inventory_fingerprint",
            "callback_dependency_inventory_fingerprint",
        ):
            _sha256(getattr(self, name), f"immutable request {name}")
        if (
            type(self.source_commit) is not str
            or len(self.source_commit) not in (40, 64)
            or any(character not in "0123456789abcdef" for character in self.source_commit)
        ):
            raise ValueError("immutable request source commit is invalid")
        _text(self.selected_branch_label, "immutable request selected branch")
        if self.artifact_relationship == "source_artifact":
            if self.evidence_artifact_sha256 != self.source_artifact_sha256:
                raise ValueError(
                    "source-artifact evidence SHA must equal source_artifact_sha256"
                )
        elif self.artifact_relationship == "lineage_bound_evidence_artifact":
            if self.evidence_artifact_sha256 == self.source_artifact_sha256:
                raise ValueError(
                    "non-source evidence relationship cannot label the source artifact"
                )
        else:
            raise ValueError("immutable request artifact relationship is invalid")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class Vituri2024FullScalarImmutableVerificationSnapshot:
    """Bytes/manifests returned by one trusted artifact authority callback."""

    artifact_bytes: bytes
    implementation_bytes: bytes
    artifact_manifest_sha256: str
    implementation_manifest_sha256: str
    authority_snapshot_fingerprint: str
    request_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.artifact_bytes) is not bytes or not self.artifact_bytes:
            raise TypeError("immutable verifier must return nonempty artifact bytes")
        if type(self.implementation_bytes) is not bytes or not self.implementation_bytes:
            raise TypeError("immutable verifier must return nonempty implementation bytes")
        for name in (
            "artifact_manifest_sha256",
            "implementation_manifest_sha256",
            "authority_snapshot_fingerprint",
            "request_fingerprint",
        ):
            _sha256(getattr(self, name), f"immutable snapshot {name}")


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFFullScalarCandidateEvidence:
    """Factory-only evidence; immutable success requires an executed authority."""

    _factory_token: InitVar[object]
    evidence_kind: CandidateEvidenceKind
    evidence_artifact_sha256: str
    lineage_fingerprint: str
    immutable_verification_request_fingerprint: str
    provider_fingerprint: str
    source_commit: str
    implementation_archive_sha256: str
    binding_fingerprint: str
    generic_input_manifest_fingerprint: str
    physical_input_inventory_fingerprint: str
    callback_dependency_inventory_fingerprint: str
    artifact_manifest_sha256: str | None
    implementation_manifest_sha256: str | None
    verifier_callback_fingerprint: str | None
    verifier_dependency_inventory_fingerprint: str | None
    authority_snapshot_fingerprint: str | None
    immutable_artifact_verified: bool
    provenance: str

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _EVIDENCE_TOKEN:
            raise TypeError("candidate evidence is factory-only")
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        if self.evidence_kind not in ("synthetic_fixture", "immutable_provider_artifact"):
            raise ValueError("candidate evidence kind is invalid")
        for name in (
            "evidence_artifact_sha256",
            "lineage_fingerprint",
            "immutable_verification_request_fingerprint",
            "provider_fingerprint",
            "implementation_archive_sha256",
            "binding_fingerprint",
            "generic_input_manifest_fingerprint",
            "physical_input_inventory_fingerprint",
            "callback_dependency_inventory_fingerprint",
        ):
            _sha256(getattr(self, name), f"candidate {name}")
        if (
            type(self.source_commit) is not str
            or len(self.source_commit) not in (40, 64)
            or any(character not in "0123456789abcdef" for character in self.source_commit)
        ):
            raise ValueError("candidate source commit must be a lowercase commit digest")
        optional = (
            self.artifact_manifest_sha256,
            self.implementation_manifest_sha256,
            self.verifier_callback_fingerprint,
            self.verifier_dependency_inventory_fingerprint,
            self.authority_snapshot_fingerprint,
        )
        if self.evidence_kind == "synthetic_fixture":
            if any(item is not None for item in optional) or self.immutable_artifact_verified:
                raise ValueError("synthetic evidence cannot carry immutable authority fields")
        else:
            if any(item is None for item in optional):
                raise ValueError("immutable evidence lacks verifier/manifest snapshots")
            for item in optional:
                assert item is not None
                _sha256(item, "immutable evidence snapshot fingerprint")
            if self.immutable_artifact_verified is not True:
                raise ValueError("immutable evidence must come from an executed verifier")
        _text(self.provenance, "candidate evidence provenance")

    @property
    def slurm_evidence_eligible(self) -> bool:
        self._validate_consistency()
        return bool(
            self.evidence_kind == "immutable_provider_artifact"
            and self.immutable_artifact_verified
        )

    @property
    def fingerprint(self) -> str:
        self._validate_consistency()
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFFullScalarLineage:
    readiness_fingerprint: str
    source_payload_fingerprint: str
    source_payload_manifest_sha256: str
    provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    source_state_sha256: str
    selected_branch_label: str
    selected_spin: int
    finite_area_receipt_fingerprint: str
    area_angstrom_squared: float
    delta1_ev: float
    interaction_receipt_fingerprint: str
    interaction_kernel_manifest_sha256: str
    form_factor_manifest_sha256: str
    q0_policy_fingerprint: str
    normal_order_reference_fingerprint: str
    assembly_context_fingerprint: str
    ordered_mesh_sha256: str
    mesh_shape: tuple[int, int]
    mesh_order: str
    payload_array_layout: str
    payload_array_conversion: str
    payload_orbital_order: str
    replay_loader_implementation_fingerprint: str
    replay_payload_schema_fingerprint: str
    source_h0_full_sha256: str
    source_projector_full_sha256: str
    layout_adapter_fingerprint: str
    storage_duality: str

    def __post_init__(self) -> None:
        for name in (
            "readiness_fingerprint",
            "source_payload_fingerprint",
            "source_payload_manifest_sha256",
            "provider_fingerprint",
            "source_artifact_sha256",
            "spec_fingerprint",
            "source_state_sha256",
            "finite_area_receipt_fingerprint",
            "interaction_receipt_fingerprint",
            "interaction_kernel_manifest_sha256",
            "form_factor_manifest_sha256",
            "q0_policy_fingerprint",
            "normal_order_reference_fingerprint",
            "assembly_context_fingerprint",
            "ordered_mesh_sha256",
            "replay_loader_implementation_fingerprint",
            "replay_payload_schema_fingerprint",
            "source_h0_full_sha256",
            "source_projector_full_sha256",
            "layout_adapter_fingerprint",
        ):
            _sha256(getattr(self, name), f"candidate lineage {name}")
        if (
            type(self.source_commit) is not str
            or len(self.source_commit) not in (40, 64)
            or any(character not in "0123456789abcdef" for character in self.source_commit)
        ):
            raise ValueError("candidate lineage source commit is invalid")
        _text(self.selected_branch_label, "candidate selected branch")
        if type(self.selected_spin) is not int or self.selected_spin not in (-1, 1):
            raise ValueError("candidate selected spin must be exactly -1 or +1")
        area = float(self.area_angstrom_squared)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("candidate lineage area must be finite and positive")
        delta1 = float(self.delta1_ev)
        if not math.isfinite(delta1):
            raise ValueError("candidate lineage Delta1 must be finite")
        object.__setattr__(self, "area_angstrom_squared", area)
        object.__setattr__(self, "delta1_ev", delta1)
        if (
            tuple(self.mesh_shape) != self.mesh_shape
            or len(self.mesh_shape) != 2
            or math.prod(self.mesh_shape) * len(INTERNAL_FLAVOR_ORDER) <= 0
            or self.mesh_order != "row_major_cartesian_k"
            or self.payload_array_layout != REPLAY_ARRAY_LAYOUT
            or self.payload_array_conversion != REPLAY_ARRAY_CONVERSION
            or self.payload_orbital_order != REPLAY_ORBITAL_ORDER
        ):
            raise ValueError("candidate lineage mesh/order/layout contract changed")
        if self.storage_duality != VITURI2024_FULL_SCALAR_STORAGE_DUALITY:
            raise ValueError("candidate lineage storage duality changed")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def _physical_input_inventory_fingerprint(
    inputs: TDHFScalarFunctionalInputsManifest,
) -> str:
    return _fingerprint(tuple((item.name, item.fingerprint) for item in inputs.entries))


def _make_immutable_verification_request(
    *,
    lineage: Vituri2024TDHFFullScalarLineage,
    inputs: TDHFScalarFunctionalInputsManifest,
    binding: TDHFFullProjectorFunctionalBinding,
    evidence_artifact_sha256: str,
) -> Vituri2024FullScalarImmutableVerificationRequest:
    """Derive the exact artifact/source/state/branch verification request."""

    if type(lineage) is not Vituri2024TDHFFullScalarLineage:
        raise TypeError("immutable request requires exact Vituri lineage")
    if type(inputs) is not TDHFScalarFunctionalInputsManifest:
        raise TypeError("immutable request requires exact generic inputs")
    if type(binding) is not TDHFFullProjectorFunctionalBinding:
        raise TypeError("immutable request requires exact callback binding")
    artifact_sha = _sha256(
        evidence_artifact_sha256, "immutable request evidence artifact"
    )
    inputs.validate_live_state()
    binding.validate_live_state()
    relationship: EvidenceArtifactRelationship = (
        "source_artifact"
        if artifact_sha == lineage.source_artifact_sha256
        else "lineage_bound_evidence_artifact"
    )
    return Vituri2024FullScalarImmutableVerificationRequest(
        lineage_fingerprint=lineage.fingerprint,
        provider_fingerprint=lineage.provider_fingerprint,
        source_commit=lineage.source_commit,
        source_artifact_sha256=lineage.source_artifact_sha256,
        source_state_sha256=lineage.source_state_sha256,
        selected_branch_label=lineage.selected_branch_label,
        evidence_artifact_sha256=artifact_sha,
        artifact_relationship=relationship,
        binding_fingerprint=binding.fingerprint,
        generic_input_manifest_fingerprint=inputs.fingerprint,
        physical_input_inventory_fingerprint=(
            _physical_input_inventory_fingerprint(inputs)
        ),
        callback_dependency_inventory_fingerprint=(
            _candidate_callback_dependency_inventory_fingerprint(binding)
        ),
    )


def make_vituri2024_tdhf_full_scalar_synthetic_evidence(
    *,
    lineage: Vituri2024TDHFFullScalarLineage,
    inputs: TDHFScalarFunctionalInputsManifest,
    binding: TDHFFullProjectorFunctionalBinding,
    evidence_artifact_bytes: bytes,
    provenance: str,
) -> Vituri2024TDHFFullScalarCandidateEvidence:
    """Create test-only evidence; this factory can never create Slurm authority."""

    if type(lineage) is not Vituri2024TDHFFullScalarLineage:
        raise TypeError("synthetic evidence requires exact Vituri lineage")
    if type(inputs) is not TDHFScalarFunctionalInputsManifest:
        raise TypeError("synthetic evidence requires exact generic inputs")
    if type(binding) is not TDHFFullProjectorFunctionalBinding:
        raise TypeError("synthetic evidence requires exact callback binding")
    if type(evidence_artifact_bytes) is not bytes or not evidence_artifact_bytes:
        raise TypeError("synthetic evidence artifact bytes must be nonempty bytes")
    inputs.validate_live_state()
    binding.validate_live_state()
    callback_inventory = _candidate_callback_dependency_inventory_fingerprint(binding)
    physical_inventory = _physical_input_inventory_fingerprint(inputs)
    artifact_sha = sha256(evidence_artifact_bytes).hexdigest()
    request = _make_immutable_verification_request(
        lineage=lineage,
        inputs=inputs,
        binding=binding,
        evidence_artifact_sha256=artifact_sha,
    )
    implementation = _fingerprint(
        {
            "binding": binding.fingerprint,
            "callback_dependency_inventory": callback_inventory,
            "generic_input_manifest": inputs.fingerprint,
            "physical_input_inventory": physical_inventory,
        }
    )
    return Vituri2024TDHFFullScalarCandidateEvidence(
        _factory_token=_EVIDENCE_TOKEN,
        evidence_kind="synthetic_fixture",
        evidence_artifact_sha256=artifact_sha,
        lineage_fingerprint=lineage.fingerprint,
        immutable_verification_request_fingerprint=request.fingerprint,
        provider_fingerprint=lineage.provider_fingerprint,
        source_commit=lineage.source_commit,
        implementation_archive_sha256=implementation,
        binding_fingerprint=binding.fingerprint,
        generic_input_manifest_fingerprint=inputs.fingerprint,
        physical_input_inventory_fingerprint=physical_inventory,
        callback_dependency_inventory_fingerprint=callback_inventory,
        artifact_manifest_sha256=None,
        implementation_manifest_sha256=None,
        verifier_callback_fingerprint=None,
        verifier_dependency_inventory_fingerprint=None,
        authority_snapshot_fingerprint=None,
        immutable_artifact_verified=False,
        provenance=provenance,
    )


def make_vituri2024_tdhf_full_scalar_immutable_evidence(
    *,
    lineage: Vituri2024TDHFFullScalarLineage,
    inputs: TDHFScalarFunctionalInputsManifest,
    binding: TDHFFullProjectorFunctionalBinding,
    verifier: Callable[[Vituri2024FullScalarImmutableVerificationRequest], object],
    verifier_dependencies: Sequence[object],
    provenance: str,
    evidence_artifact_sha256: str | None = None,
) -> Vituri2024TDHFFullScalarCandidateEvidence:
    """Execute one trusted authority callback and hash returned artifact bytes.

    This is a trusted-provider boundary, not a hostile-code or global-completeness
    proof.  The repository supplies no concrete immutable authority callback.
    """

    if type(lineage) is not Vituri2024TDHFFullScalarLineage:
        raise TypeError("immutable evidence requires exact Vituri lineage")
    if type(inputs) is not TDHFScalarFunctionalInputsManifest:
        raise TypeError("immutable evidence requires exact generic inputs")
    if type(binding) is not TDHFFullProjectorFunctionalBinding:
        raise TypeError("immutable evidence requires exact callback binding")
    if not inspect.isfunction(verifier) or verifier.__closure__:
        raise TypeError("immutable authority verifier must be a closure-free Python function")
    parameters = tuple(inspect.signature(verifier).parameters.values())
    if (
        len(parameters) != 1
        or parameters[0].name != "request"
        or parameters[0].default is not inspect.Parameter.empty
    ):
        raise TypeError("immutable authority verifier signature must be exactly (request)")
    inputs.validate_live_state()
    binding.validate_live_state()
    callback_inventory = _candidate_callback_dependency_inventory_fingerprint(binding)
    physical_inventory = _physical_input_inventory_fingerprint(inputs)
    expected_artifact_sha = (
        lineage.source_artifact_sha256
        if evidence_artifact_sha256 is None
        else _sha256(evidence_artifact_sha256, "immutable evidence artifact")
    )
    request = _make_immutable_verification_request(
        lineage=lineage,
        inputs=inputs,
        binding=binding,
        evidence_artifact_sha256=expected_artifact_sha,
    )
    verifier_fingerprint = _code_snapshot_fingerprint(verifier, "immutable verifier")
    dependency_fingerprints = tuple(
        _code_snapshot_fingerprint(item, "immutable verifier dependency")
        for item in verifier_dependencies
    )
    if len(set(dependency_fingerprints)) != len(dependency_fingerprints):
        raise ValueError("immutable verifier dependency snapshots must be unique")
    dependency_inventory = _fingerprint(dependency_fingerprints)
    inputs_before = inputs.fingerprint
    binding_before = binding.fingerprint
    snapshot = verifier(request)
    inputs.validate_live_state()
    binding.validate_live_state()
    if inputs.fingerprint != inputs_before or binding.fingerprint != binding_before:
        raise ValueError("immutable verifier mutated generic inputs/callback binding")
    if _code_snapshot_fingerprint(verifier, "immutable verifier") != verifier_fingerprint:
        raise ValueError("immutable verifier source snapshot drifted during execution")
    if tuple(
        _code_snapshot_fingerprint(item, "immutable verifier dependency")
        for item in verifier_dependencies
    ) != dependency_fingerprints:
        raise ValueError("immutable verifier dependency snapshots drifted during execution")
    if type(snapshot) is not Vituri2024FullScalarImmutableVerificationSnapshot:
        raise TypeError("immutable verifier returned the wrong snapshot type")
    assert isinstance(snapshot, Vituri2024FullScalarImmutableVerificationSnapshot)
    if snapshot.request_fingerprint != request.fingerprint:
        raise ValueError("immutable verifier snapshot/request binding mismatch")
    artifact_sha = sha256(snapshot.artifact_bytes).hexdigest()
    if artifact_sha != request.evidence_artifact_sha256:
        raise ValueError("immutable verifier artifact bytes differ from the requested artifact SHA")
    if (
        request.artifact_relationship == "source_artifact"
        and artifact_sha != lineage.source_artifact_sha256
    ):
        raise ValueError("immutable source-artifact evidence differs from lineage source SHA")
    if request.artifact_relationship == "lineage_bound_evidence_artifact" and (
        request.source_artifact_sha256,
        request.source_state_sha256,
        request.selected_branch_label,
    ) != (
        lineage.source_artifact_sha256,
        lineage.source_state_sha256,
        lineage.selected_branch_label,
    ):
        raise ValueError("immutable non-source artifact request lost exact source lineage")
    implementation_sha = sha256(snapshot.implementation_bytes).hexdigest()
    return Vituri2024TDHFFullScalarCandidateEvidence(
        _factory_token=_EVIDENCE_TOKEN,
        evidence_kind="immutable_provider_artifact",
        evidence_artifact_sha256=artifact_sha,
        lineage_fingerprint=lineage.fingerprint,
        immutable_verification_request_fingerprint=request.fingerprint,
        provider_fingerprint=lineage.provider_fingerprint,
        source_commit=lineage.source_commit,
        implementation_archive_sha256=implementation_sha,
        binding_fingerprint=binding.fingerprint,
        generic_input_manifest_fingerprint=inputs.fingerprint,
        physical_input_inventory_fingerprint=physical_inventory,
        callback_dependency_inventory_fingerprint=callback_inventory,
        artifact_manifest_sha256=snapshot.artifact_manifest_sha256,
        implementation_manifest_sha256=snapshot.implementation_manifest_sha256,
        verifier_callback_fingerprint=verifier_fingerprint,
        verifier_dependency_inventory_fingerprint=dependency_inventory,
        authority_snapshot_fingerprint=snapshot.authority_snapshot_fingerprint,
        immutable_artifact_verified=True,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFFullScalarSupport:
    runtime_layout: str
    runtime_density_shape: tuple[int, ...]
    runtime_fock_shape: tuple[int, ...]
    affine_support: str
    arbitrary_full_dense_affine_hermitian: bool
    arbitrary_off_k: bool
    arbitrary_cross_flavor: bool
    arbitrary_complex_imaginary: bool

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFFullScalarCandidate:
    evidence: Vituri2024TDHFFullScalarCandidateEvidence
    lineage: Vituri2024TDHFFullScalarLineage
    space: TDHFFullProjectorSpace
    inputs: TDHFScalarFunctionalInputsManifest
    binding: TDHFFullProjectorFunctionalBinding
    validation_plan: TDHFFullProjectorValidationPlan
    support: Vituri2024TDHFFullScalarSupport
    orbital_ids: tuple[int, ...]
    one_body_construction: str
    total_energy_normalization: str
    provenance: str

    def __post_init__(self) -> None:
        exact_types = (
            (self.evidence, Vituri2024TDHFFullScalarCandidateEvidence, "evidence"),
            (self.lineage, Vituri2024TDHFFullScalarLineage, "lineage"),
            (self.space, TDHFFullProjectorSpace, "space"),
            (self.inputs, TDHFScalarFunctionalInputsManifest, "inputs"),
            (self.binding, TDHFFullProjectorFunctionalBinding, "binding"),
            (self.validation_plan, TDHFFullProjectorValidationPlan, "validation plan"),
            (self.support, Vituri2024TDHFFullScalarSupport, "support"),
        )
        for value, expected, label in exact_types:
            if type(value) is not expected:
                raise TypeError(f"candidate {label} requires exact type")
        self.evidence._validate_consistency()
        object.__setattr__(self, "orbital_ids", tuple(self.orbital_ids))
        _text(self.one_body_construction, "candidate one-body construction")
        _text(self.total_energy_normalization, "candidate energy normalization")
        _text(self.provenance, "candidate provenance")

    @property
    def fingerprint(self) -> str:
        self.inputs.validate_live_state()
        self.binding.validate_live_state()
        return _fingerprint(
            {
                "evidence": self.evidence.fingerprint,
                "lineage": self.lineage.fingerprint,
                "space": self.space.fingerprint,
                "inputs": self.inputs.fingerprint,
                "binding": self.binding.fingerprint,
                "validation_plan": self.validation_plan.fingerprint,
                "support": self.support.fingerprint,
                "orbital_ids": self.orbital_ids,
                "one_body_construction": self.one_body_construction,
                "total_energy_normalization": self.total_energy_normalization,
                "provenance": self.provenance,
            }
        )


def make_vituri2024_tdhf_full_scalar_lineage(
    *,
    readiness: Vituri2024TDHFScalarReadinessReceipt,
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
) -> Vituri2024TDHFFullScalarLineage:
    if type(readiness) is not Vituri2024TDHFScalarReadinessReceipt:
        raise TypeError("candidate lineage requires exact Vituri readiness")
    if type(prerequisites) is not Vituri2024PocketRefinementPrerequisites:
        raise TypeError("candidate lineage requires exact Vituri prerequisites")
    if type(source_payload) is not Vituri2024HalfMetalHFReplayPayload:
        raise TypeError("candidate lineage requires exact Vituri replay payload")
    spec = prerequisites.binding.spec
    spec.require_receipt_set_complete()
    assert spec.geometry is not None
    assert spec.ensemble is not None
    assert spec.shared_functional is not None
    assert spec.attested_source is not None
    source = vituri2024_tdhf_full_scalar_source_from_payload(source_payload)
    return Vituri2024TDHFFullScalarLineage(
        readiness_fingerprint=readiness.fingerprint,
        source_payload_fingerprint=_fingerprint(source_payload),
        source_payload_manifest_sha256=(
            prerequisites.array_replay_receipt.hashes.payload_manifest_sha256
        ),
        provider_fingerprint=spec.attested_source.provider_fingerprint,
        source_commit=spec.attested_source.source_commit,
        source_artifact_sha256=spec.attested_source.source_artifact_sha256,
        spec_fingerprint=spec.fingerprint,
        source_state_sha256=spec.attested_source.source_state_sha256,
        selected_branch_label=spec.attested_source.selected_branch_label,
        selected_spin=spec.attested_source.selected_spin,
        finite_area_receipt_fingerprint=spec.geometry.finite_area_receipt_fingerprint,
        area_angstrom_squared=spec.geometry.area_angstrom_squared,
        delta1_ev=readiness.delta1_ev,
        interaction_receipt_fingerprint=(
            spec.shared_functional.interaction_receipt_fingerprint
        ),
        interaction_kernel_manifest_sha256=(
            readiness.interaction_receipt_fingerprint
        ),
        form_factor_manifest_sha256=(
            spec.shared_functional.interaction_form_factor.fingerprint
        ),
        q0_policy_fingerprint=spec.ensemble.q0_policy_fingerprint,
        normal_order_reference_fingerprint=(
            spec.ensemble.normal_order_reference_fingerprint
        ),
        assembly_context_fingerprint=readiness.assembly_context_fingerprint,
        ordered_mesh_sha256=source.ordered_mesh_sha256,
        mesh_shape=spec.geometry.mesh_shape,
        mesh_order=spec.geometry.mesh_order,
        payload_array_layout=REPLAY_ARRAY_LAYOUT,
        payload_array_conversion=REPLAY_ARRAY_CONVERSION,
        payload_orbital_order=REPLAY_ORBITAL_ORDER,
        replay_loader_implementation_fingerprint=(
            source_payload.replay_loader_implementation_fingerprint
        ),
        replay_payload_schema_fingerprint=(
            source_payload.replay_payload_schema_fingerprint
        ),
        source_h0_full_sha256=_array_sha256(source.source_h0),
        source_projector_full_sha256=_array_sha256(source.source_projector),
        layout_adapter_fingerprint=source.space.layout_adapter_fingerprint,
        storage_duality=VITURI2024_FULL_SCALAR_STORAGE_DUALITY,
    )


def _input_name_is_forbidden(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    exact = {
        "a",
        "b",
        "h",
        "h_plus",
        "h_minus",
        "a_plus",
        "a_minus",
        "b_plus_minus",
        "b_minus_plus",
        "wbar",
        "target_h",
        "target_h_plus",
    }
    return (
        normalized in exact
        or "f0" in normalized
        or "source_fock" in normalized
        or "counterterm" in normalized
        or "wbar" in normalized
        or "target" in normalized
        or "finite_q_hessian" in normalized
        or normalized.startswith("tdhf_")
    )


def _validate_implementation_surface(candidate: Vituri2024TDHFFullScalarCandidate) -> None:
    restricted_marker = "vituri2024_tdhf_restricted_scalar"
    kernels = (
        candidate.binding.energy,
        candidate.binding.fock,
        candidate.binding.fock_derivative,
    )
    for kernel in kernels:
        manifests = (kernel.manifest.callback,) + tuple(
            item.manifest for item in kernel.dependencies
        )
        for manifest in manifests:
            if restricted_marker in manifest.module_name or restricted_marker in manifest.module_file:
                raise ValueError("full candidate implementation/dependency points to restricted oracle")
        peer_names = {
            peer.callback.__name__ for peer in kernels if peer is not kernel
        }
        if peer_names.intersection(kernel.callback.__code__.co_names):
            raise ValueError("full candidate callback directly delegates to a peer callback")
    derivative = candidate.binding.fock_derivative.callback
    try:
        source = inspect.getsource(derivative).lower()
    except (OSError, TypeError) as error:
        raise ValueError("candidate dF source is unavailable") from error
    names = tuple(str(item).lower() for item in derivative.__code__.co_names)
    if "finite_q_hessian" in source or any("finite_q_hessian" in item for item in names):
        raise ValueError("candidate dF delegates to a finite-q Hessian entrypoint")
    if restricted_marker in source:
        raise ValueError("candidate dF delegates to the restricted scalar module")


def _probe_support(plan: TDHFFullProjectorValidationPlan, nk: int) -> tuple[bool, bool, bool]:
    off_k = False
    cross_flavor = False
    imaginary = False
    threshold = 64.0 * np.finfo(np.float64).eps
    for direction in plan.directions:
        rows, columns = np.nonzero(np.abs(direction.matrix) > threshold)
        for row, column in zip(rows.tolist(), columns.tolist()):
            if row == column:
                continue
            off_k = off_k or (row % nk != column % nk)
            cross_flavor = cross_flavor or (row // nk != column // nk)
            imaginary = imaginary or abs(direction.matrix[row, column].imag) > threshold
    return off_k, cross_flavor, imaginary


def _revalidate_chain(
    *,
    readiness: Vituri2024TDHFScalarReadinessReceipt,
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
    assembly_receipt: Vituri2024TDHFSignedQAssemblyReceipt,
) -> tuple[Vituri2024TDHFFullScalarSource, Vituri2024TDHFFullScalarLineage]:
    exact = (
        (readiness, Vituri2024TDHFScalarReadinessReceipt, "readiness"),
        (prerequisites, Vituri2024PocketRefinementPrerequisites, "prerequisites"),
        (source_payload, Vituri2024HalfMetalHFReplayPayload, "source payload"),
        (assembly_receipt, Vituri2024TDHFSignedQAssemblyReceipt, "assembly receipt"),
    )
    for value, expected, label in exact:
        if type(value) is not expected:
            raise TypeError(f"full candidate requires exact Vituri {label}")
    readiness._validate_consistency()
    assembly_receipt._validate_live_state()
    binding = prerequisites.binding
    spec = binding.spec
    spec.require_receipt_set_complete()
    assert spec.geometry is not None
    assert spec.ensemble is not None
    assert spec.shared_functional is not None
    assert spec.attested_source is not None
    source_receipt = spec.attested_source
    # Re-run non-executing binding/prerequisite constructors to reject mixed A/B
    # or stale nested chains rather than trusting their outer dataclass labels.
    type(binding)(spec, binding.provider)
    type(prerequisites)(
        binding,
        prerequisites.array_replay_receipt,
        prerequisites.scf_replay_approval,
        prerequisites.scf_replay_receipt,
    )
    payload_fingerprint = _fingerprint(source_payload)
    array_hashes = prerequisites.array_replay_receipt.hashes
    plus_context = assembly_receipt.signed_pair.plus_context
    checks = (
        (readiness.assembly_receipt_fingerprint, assembly_receipt.fingerprint, "readiness/assembly"),
        (readiness.source_payload_fingerprint, payload_fingerprint, "readiness/source payload"),
        (
            readiness.source_payload_manifest_sha256,
            array_hashes.payload_manifest_sha256,
            "readiness/payload manifest",
        ),
        (readiness.provider_fingerprint, source_receipt.provider_fingerprint, "readiness/provider"),
        (readiness.source_artifact_sha256, source_receipt.source_artifact_sha256, "readiness/artifact"),
        (readiness.spec_fingerprint, spec.fingerprint, "readiness/spec"),
        (readiness.source_state_sha256, source_receipt.source_state_sha256, "readiness/state"),
        (readiness.selected_branch_label, source_receipt.selected_branch_label, "readiness/branch"),
        (readiness.selected_spin, source_receipt.selected_spin, "readiness/spin"),
        (source_payload.provider_fingerprint, source_receipt.provider_fingerprint, "payload/provider"),
        (source_payload.source_commit, source_receipt.source_commit, "payload/commit"),
        (source_payload.source_artifact_sha256, source_receipt.source_artifact_sha256, "payload/artifact"),
        (source_payload.spec_fingerprint, spec.fingerprint, "payload/spec"),
        (source_payload.source_state_sha256, source_receipt.source_state_sha256, "payload/state"),
        (
            source_payload.replay_loader_implementation_fingerprint,
            source_receipt.replay_loader_implementation_fingerprint,
            "payload/loader",
        ),
        (
            source_payload.replay_payload_schema_fingerprint,
            source_receipt.replay_payload_schema_fingerprint,
            "payload/schema",
        ),
        (readiness.finite_area, plus_context.area, "readiness/context area object"),
        (
            readiness.finite_area_receipt_fingerprint,
            plus_context.area.fingerprint,
            "readiness/context area fingerprint",
        ),
        (
            readiness.finite_area_receipt_fingerprint,
            spec.geometry.finite_area_receipt_fingerprint,
            "readiness/spec area fingerprint",
        ),
        (readiness.area_angstrom_squared, plus_context.area.area_angstrom_squared, "readiness/context area value"),
        (readiness.area_angstrom_squared, spec.geometry.area_angstrom_squared, "readiness/spec area value"),
        (readiness.delta1_ev, plus_context.delta1_ev, "readiness/context Delta1"),
        (readiness.delta1_ev, spec.geometry.delta1_mev / 1000.0, "readiness/spec Delta1"),
        (
            readiness.interaction_receipt_fingerprint,
            plus_context.interaction_receipt_fingerprint,
            "readiness/context interaction",
        ),
        (
            readiness.interaction_receipt_fingerprint,
            spec.shared_functional.interaction_receipt_fingerprint,
            "readiness/spec interaction",
        ),
        (
            readiness.assembly_context_fingerprint,
            plus_context.assembly_context_fingerprint,
            "readiness/assembly context",
        ),
        (readiness.source_fingerprint, assembly_receipt.source_fingerprint, "readiness/assembly source"),
        (
            readiness.assembly_interaction_fingerprint,
            assembly_receipt.interaction_fingerprint,
            "readiness/assembly interaction",
        ),
        (canonical_array_sha256(source_payload.mesh), array_hashes.ordered_momentum_mesh_sha256, "payload mesh bytes"),
        (canonical_orbital_order_sha256(source_payload.mesh), array_hashes.ordered_orbitals_sha256, "payload orbital order"),
        (canonical_array_sha256(source_payload.active_band_states), array_hashes.active_band_states_sha256, "payload active states"),
        (canonical_array_sha256(source_payload.energies), array_hashes.ordered_energies_sha256, "payload energies"),
        (canonical_array_sha256(source_payload.occupations), array_hashes.ordered_occupations_sha256, "payload occupations"),
        (canonical_array_sha256(source_payload.projector), array_hashes.ordered_projector_sha256, "payload projector"),
        (canonical_array_sha256(source_payload.fock), array_hashes.ordered_fock_sha256, "payload Fock"),
        (canonical_array_sha256(source_payload.h0), array_hashes.h0_sha256, "payload h0"),
        (canonical_array_sha256(source_payload.interaction_h), array_hashes.interaction_h_sha256, "payload interaction"),
        (source_payload.source_state_sha256, array_hashes.reconstructed_source_state_sha256, "payload reconstructed state"),
        (canonical_array_sha256(source_payload.mesh), spec.geometry.ordered_momentum_mesh_sha256, "payload/spec mesh"),
        (source_payload.mesh.shape[0], spec.geometry.mesh_point_count, "payload/spec Nk"),
        (spec.geometry.array_layout, REPLAY_ARRAY_LAYOUT, "spec array layout"),
        (spec.geometry.array_conversion, REPLAY_ARRAY_CONVERSION, "spec array conversion"),
        (prerequisites.array_replay_receipt.orbital_order, REPLAY_ORBITAL_ORDER, "receipt orbital order"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"full candidate mixed/stale chain rejected at {label}")
    source = vituri2024_tdhf_full_scalar_source_from_payload(source_payload)
    lineage = make_vituri2024_tdhf_full_scalar_lineage(
        readiness=readiness,
        prerequisites=prerequisites,
        source_payload=source_payload,
    )
    return source, lineage


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFFullScalarPreflightReceipt:
    _factory_token: InitVar[object]
    api_version: str
    status: str
    candidate_fingerprint: str
    readiness_fingerprint: str
    source_payload_fingerprint: str
    assembly_receipt_fingerprint: str
    lineage_fingerprint: str
    space_fingerprint: str
    source_projector_fingerprint: str
    source_h0_fingerprint: str
    source_fock_fingerprint: str
    source_payload_manifest_sha256: str
    normalization_receipt: Vituri2024FullScalarNormalizationReceipt
    normalization_receipt_fingerprint: str
    native_to_raw_total_factor: int
    generic_approval: TDHFFullProjectorFunctionalApproval
    generic_approval_fingerprint: str
    evidence_kind: str
    candidate_bound: bool
    generic_validation_executed: bool
    synthetic_fixture: bool
    immutable_provider_candidate: bool
    eligible_for_slurm_qualification: bool
    tdhf_hessian_match: bool = field(default=False, init=False)
    scalar_hessian_match: bool = field(default=False, init=False)
    static_hessian_authority_promoted: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _PREFLIGHT_TOKEN:
            raise TypeError("Vituri full-scalar preflight requires private factory token")
        if self.api_version != VITURI2024_FULL_SCALAR_API_VERSION:
            raise ValueError("Vituri full-scalar preflight API changed")
        if self.status != VITURI2024_FULL_SCALAR_PREFLIGHT_STATUS:
            raise ValueError("static preflight status must remain candidate_bound_not_executed")
        for name in (
            "candidate_fingerprint",
            "readiness_fingerprint",
            "source_payload_fingerprint",
            "assembly_receipt_fingerprint",
            "lineage_fingerprint",
            "space_fingerprint",
            "source_projector_fingerprint",
            "source_h0_fingerprint",
            "source_fock_fingerprint",
            "source_payload_manifest_sha256",
            "normalization_receipt_fingerprint",
            "generic_approval_fingerprint",
        ):
            _sha256(getattr(self, name), name)
        if type(self.normalization_receipt) is not Vituri2024FullScalarNormalizationReceipt:
            raise TypeError("preflight requires a factory normalization receipt")
        if (
            self.normalization_receipt.fingerprint
            != self.normalization_receipt_fingerprint
            or self.native_to_raw_total_factor != self.normalization_receipt.nk
        ):
            raise ValueError("preflight native/raw-total normalization receipt drifted")
        if type(self.generic_approval) is not TDHFFullProjectorFunctionalApproval:
            raise TypeError("preflight requires exact generic detached approval")
        if self.generic_approval.fingerprint != self.generic_approval_fingerprint:
            raise ValueError("preflight generic approval fingerprint mismatch")
        if (
            self.candidate_bound is not True
            or self.generic_validation_executed is not False
            or self.eligible_for_slurm_qualification is not False
            or self.synthetic_fixture != (self.evidence_kind == "synthetic_fixture")
            or self.immutable_provider_candidate
            != (self.evidence_kind == "immutable_provider_artifact")
        ):
            raise ValueError("static preflight execution/eligibility status drifted")
        if any(
            (
                self.tdhf_hessian_match,
                self.scalar_hessian_match,
                self.static_hessian_authority_promoted,
                self.production_ready,
                self.paper_reproduction_verified,
            )
        ):
            raise ValueError("static candidate preflight cannot promote authority")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def preflight_vituri2024_tdhf_full_scalar_candidate(
    *,
    readiness: Vituri2024TDHFScalarReadinessReceipt,
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
    assembly_receipt: Vituri2024TDHFSignedQAssemblyReceipt,
    candidate: Vituri2024TDHFFullScalarCandidate,
) -> Vituri2024TDHFFullScalarPreflightReceipt:
    """Bind a full candidate without executing E, F, or dF."""

    if type(candidate) is not Vituri2024TDHFFullScalarCandidate:
        raise TypeError("preflight requires exact typed full-scalar candidate")
    source, lineage = _revalidate_chain(
        readiness=readiness,
        prerequisites=prerequisites,
        source_payload=source_payload,
        assembly_receipt=assembly_receipt,
    )
    candidate.evidence._validate_consistency()
    if candidate.evidence.lineage_fingerprint != lineage.fingerprint:
        raise ValueError("candidate evidence exact lineage fingerprint drift")
    if candidate.evidence.provider_fingerprint != lineage.provider_fingerprint:
        raise ValueError("candidate evidence provider fingerprint drift")
    if candidate.evidence.source_commit != lineage.source_commit:
        raise ValueError("candidate evidence source commit drift")
    if candidate.lineage != lineage:
        raise ValueError("candidate source/state/branch/area/interaction/q0/normal-order/layout lineage drift")
    if candidate.space.fingerprint != source.space.fingerprint:
        raise ValueError("candidate full-space dimension/layout fingerprint drift")
    expected_dimension = len(INTERNAL_FLAVOR_ORDER) * source.nk
    transition_union = len(assembly_receipt.orbital_id_map)
    if candidate.space.dimension != expected_dimension:
        if candidate.space.dimension == transition_union:
            raise ValueError("restricted assembly.orbital_id_map transition union is not full space")
        raise ValueError("candidate dimension must be exactly 4*Nk full orbital space")
    if candidate.orbital_ids != tuple(range(expected_dimension)):
        raise ValueError("candidate orbital IDs must cover exact full flavor*Nk order")
    if candidate.validation_plan.space.fingerprint != source.space.fingerprint:
        raise ValueError("candidate validation plan uses stale/full-layout-incompatible space")
    if candidate.validation_plan.probe_scope != "explicit_bound_probes":
        raise ValueError("production-sized Vituri plans require explicit bound probes")
    if candidate.validation_plan.require_informative_df is not True:
        raise ValueError("Vituri validation requires mandatory dF informativeness")
    if not candidate.validation_plan.unitary_projector_probes:
        raise ValueError("Vituri validation requires preregistered exact-unitary projector values")
    if not np.array_equal(candidate.validation_plan.source_projector, source.source_projector):
        raise ValueError("candidate conventional full P0 differs from exact replay payload")
    support = candidate.support
    expected_shape = (expected_dimension, expected_dimension)
    required_support = (
        support.runtime_layout == VITURI2024_FULL_SCALAR_RUNTIME_LAYOUT,
        support.runtime_density_shape == expected_shape,
        support.runtime_fock_shape == expected_shape,
        support.affine_support == VITURI2024_FULL_SCALAR_AFFINE_SUPPORT,
        support.arbitrary_full_dense_affine_hermitian is True,
        support.arbitrary_off_k is True,
        support.arbitrary_cross_flavor is True,
        support.arbitrary_complex_imaginary is True,
    )
    if not all(required_support):
        raise ValueError(
            "candidate lacks arbitrary full dense affine/off-k/cross-flavor/imaginary support"
        )
    off_k, cross_flavor, imaginary = _probe_support(candidate.validation_plan, source.nk)
    if not (off_k and cross_flavor and imaginary):
        raise ValueError("explicit probes do not cover off-k/cross-flavor/imaginary full-dense support")
    if candidate.one_body_construction != VITURI2024_FULL_SCALAR_ONE_BODY_CONSTRUCTION:
        raise ValueError("candidate one-body term uses forbidden F0-dF[P0] counterterm/fitting")
    if candidate.total_energy_normalization != VITURI2024_FULL_SCALAR_TOTAL_ENERGY_NORMALIZATION:
        raise ValueError("candidate must use raw total energy and unweighted full trace")
    if candidate.validation_plan.energy_normalization != TDHF_FULL_PROJECTOR_ENERGY_NORMALIZATION:
        raise ValueError("candidate generic plan energy normalization drift")
    input_names = tuple(item.name for item in candidate.inputs.entries)
    if input_names != VITURI2024_FULL_SCALAR_PHYSICAL_INPUT_NAMES:
        forbidden = tuple(name for name in input_names if _input_name_is_forbidden(name))
        detail = ", ".join(forbidden or input_names)
        raise ValueError(
            "candidate physical inputs must equal the required allowlist; "
            "F0/source_fock/A/B/H+/wbar/target/counterterm inputs are forbidden: "
            + detail
        )
    required_input_values: dict[str, object] = {
        "area_angstrom_squared": lineage.area_angstrom_squared,
        "form_factor_manifest_sha256": lineage.form_factor_manifest_sha256,
        "h0_full": source.source_h0,
        "interaction_kernel_manifest_sha256": (
            lineage.interaction_kernel_manifest_sha256
        ),
        "normal_order_reference_fingerprint": (
            lineage.normal_order_reference_fingerprint
        ),
        "ordered_mesh": source.ordered_mesh,
        "q0_policy_fingerprint": lineage.q0_policy_fingerprint,
        "source_projector_full": source.source_projector,
    }
    for name, expected in required_input_values.items():
        try:
            actual = candidate.inputs.value(name)
        except KeyError as error:
            raise ValueError(f"candidate required physical input {name!r} is missing") from error
        if isinstance(expected, np.ndarray):
            if not isinstance(actual, np.ndarray) or not np.array_equal(actual, expected):
                raise ValueError(f"candidate physical input {name!r} bytes are stale")
        elif actual != expected:
            raise ValueError(f"candidate physical input {name!r} is stale")
    if _array_sha256(candidate.inputs.array("h0_full")) != lineage.source_h0_full_sha256:
        raise ValueError("candidate h0_full does not exactly embed payload h0 bytes")
    if (
        _array_sha256(candidate.inputs.array("source_projector_full"))
        != lineage.source_projector_full_sha256
    ):
        raise ValueError("candidate source P0 does not exactly embed payload projector bytes")
    evidence_bindings = (
        (
            candidate.evidence.binding_fingerprint,
            candidate.binding.fingerprint,
            "callback binding",
        ),
        (
            candidate.evidence.generic_input_manifest_fingerprint,
            candidate.inputs.fingerprint,
            "generic input manifest",
        ),
        (
            candidate.evidence.physical_input_inventory_fingerprint,
            _physical_input_inventory_fingerprint(candidate.inputs),
            "physical input inventory",
        ),
        (
            candidate.evidence.callback_dependency_inventory_fingerprint,
            _candidate_callback_dependency_inventory_fingerprint(candidate.binding),
            "callback/dependency inventory",
        ),
    )
    for actual, expected, label in evidence_bindings:
        if actual != expected:
            raise ValueError(f"candidate evidence {label} drift")
    exact_request = _make_immutable_verification_request(
        lineage=lineage,
        inputs=candidate.inputs,
        binding=candidate.binding,
        evidence_artifact_sha256=candidate.evidence.evidence_artifact_sha256,
    )
    if (
        candidate.evidence.immutable_verification_request_fingerprint
        != exact_request.fingerprint
    ):
        raise ValueError(
            "candidate evidence immutable verification request fingerprint drift"
        )
    _validate_implementation_surface(candidate)
    approval = make_tdhf_full_projector_functional_approval(
        space=candidate.space,
        inputs=candidate.inputs,
        binding=candidate.binding,
        plan=candidate.validation_plan,
        provenance=(
            "Vituri static candidate preflight detached before any E/F/dF callback; "
            "no TDHF H+ comparison or authority promotion."
        ),
    )
    return Vituri2024TDHFFullScalarPreflightReceipt(
        _factory_token=_PREFLIGHT_TOKEN,
        api_version=VITURI2024_FULL_SCALAR_API_VERSION,
        status=VITURI2024_FULL_SCALAR_PREFLIGHT_STATUS,
        candidate_fingerprint=candidate.fingerprint,
        readiness_fingerprint=readiness.fingerprint,
        source_payload_fingerprint=_fingerprint(source_payload),
        assembly_receipt_fingerprint=assembly_receipt.fingerprint,
        lineage_fingerprint=lineage.fingerprint,
        space_fingerprint=source.space.fingerprint,
        source_projector_fingerprint=_array_sha256(source.source_projector),
        source_h0_fingerprint=_array_sha256(source.source_h0),
        source_fock_fingerprint=_array_sha256(source.source_fock),
        source_payload_manifest_sha256=lineage.source_payload_manifest_sha256,
        normalization_receipt=source.normalization_receipt,
        normalization_receipt_fingerprint=source.normalization_receipt.fingerprint,
        native_to_raw_total_factor=source.nk,
        generic_approval=approval,
        generic_approval_fingerprint=approval.fingerprint,
        evidence_kind=candidate.evidence.evidence_kind,
        candidate_bound=True,
        generic_validation_executed=False,
        synthetic_fixture=candidate.evidence.evidence_kind == "synthetic_fixture",
        immutable_provider_candidate=(
            candidate.evidence.evidence_kind == "immutable_provider_artifact"
        ),
        eligible_for_slurm_qualification=False,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFFullScalarQualificationReceipt:
    _factory_token: InitVar[object]
    api_version: str
    preflight_fingerprint: str
    candidate_fingerprint: str
    generic_receipt_fingerprint: str
    generic_registered_probe_consistency_consumed: bool
    generic_full_projector_consistency_consumed: bool
    dF_response_informativeness_consumed: bool
    exact_unitary_execution_consumed: bool
    evidence_kind: str
    immutable_evidence_verified: bool
    eligible_for_slurm_qualification: bool
    readiness_established: bool = field(default=False, init=False)
    tdhf_hessian_match: bool = field(default=False, init=False)
    scalar_hessian_match: bool = field(default=False, init=False)
    static_hessian_authority_promoted: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _QUALIFICATION_TOKEN:
            raise TypeError("Vituri full-scalar qualification requires private factory token")
        if self.api_version != VITURI2024_FULL_SCALAR_API_VERSION:
            raise ValueError("Vituri full-scalar qualification API changed")
        for name in (
            "preflight_fingerprint",
            "candidate_fingerprint",
            "generic_receipt_fingerprint",
        ):
            _sha256(getattr(self, name), name)
        if self.generic_registered_probe_consistency_consumed is not True:
            raise ValueError("qualification must consume registered-probe consistency")
        if self.generic_full_projector_consistency_consumed is not False:
            raise ValueError("incomplete Vituri probes cannot claim full-projector consistency")
        if (
            self.dF_response_informativeness_consumed is not True
            or self.exact_unitary_execution_consumed is not True
        ):
            raise ValueError("qualification must consume informative dF and executed unitary E/F")
        expected_eligibility = bool(
            self.evidence_kind == "immutable_provider_artifact"
            and self.immutable_evidence_verified
        )
        if self.eligible_for_slurm_qualification != expected_eligibility:
            raise ValueError("Slurm eligibility requires factory-verified immutable evidence")
        if any(
            (
                self.readiness_established,
                self.tdhf_hessian_match,
                self.scalar_hessian_match,
                self.static_hessian_authority_promoted,
                self.production_ready,
                self.paper_reproduction_verified,
            )
        ):
            raise ValueError("candidate qualification cannot establish readiness/authority")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def consume_vituri2024_tdhf_full_scalar_validation(
    *,
    preflight: Vituri2024TDHFFullScalarPreflightReceipt,
    readiness: Vituri2024TDHFScalarReadinessReceipt,
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
    assembly_receipt: Vituri2024TDHFSignedQAssemblyReceipt,
    candidate: Vituri2024TDHFFullScalarCandidate,
    generic_receipt: TDHFFullProjectorFunctionalEvidenceReceipt,
) -> Vituri2024TDHFFullScalarQualificationReceipt:
    """Consume generic consistency while retaining every authority flag false."""

    if type(preflight) is not Vituri2024TDHFFullScalarPreflightReceipt:
        raise TypeError("qualification requires exact static preflight")
    if type(generic_receipt) is not TDHFFullProjectorFunctionalEvidenceReceipt:
        raise TypeError("qualification requires exact generic full-projector receipt")
    rebound = preflight_vituri2024_tdhf_full_scalar_candidate(
        readiness=readiness,
        prerequisites=prerequisites,
        source_payload=source_payload,
        assembly_receipt=assembly_receipt,
        candidate=candidate,
    )
    if rebound.fingerprint != preflight.fingerprint:
        raise ValueError("candidate/preflight rebind drifted before qualification")
    qualification_request = _make_immutable_verification_request(
        lineage=candidate.lineage,
        inputs=candidate.inputs,
        binding=candidate.binding,
        evidence_artifact_sha256=candidate.evidence.evidence_artifact_sha256,
    )
    if candidate.evidence.lineage_fingerprint != rebound.lineage_fingerprint:
        raise ValueError("qualification exact lineage fingerprint drift")
    if (
        candidate.evidence.immutable_verification_request_fingerprint
        != qualification_request.fingerprint
    ):
        raise ValueError("qualification immutable verification request drift")
    checks = (
        (
            generic_receipt.approval_fingerprint,
            preflight.generic_approval_fingerprint,
            "generic approval",
        ),
        (generic_receipt.space_fingerprint, preflight.space_fingerprint, "generic space"),
        (
            generic_receipt.source_projector_fingerprint,
            preflight.source_projector_fingerprint,
            "generic source projector",
        ),
        (
            generic_receipt.inputs_fingerprint_after,
            candidate.inputs.fingerprint,
            "generic inputs",
        ),
        (
            generic_receipt.binding_fingerprint_after,
            candidate.binding.fingerprint,
            "generic binding",
        ),
        (
            generic_receipt.plan_fingerprint,
            candidate.validation_plan.fingerprint,
            "generic plan",
        ),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"qualification {label} fingerprint drift")
    if generic_receipt.registered_probe_functional_consistency is not True:
        raise ValueError("generic registered-probe consistency is not positive")
    if generic_receipt.full_projector_functional_consistency is not False:
        raise ValueError("incomplete Vituri N^2 inventory falsely claims full consistency")
    if (
        generic_receipt.dF_response_informativeness_required is not True
        or generic_receipt.dF_response_informative is not True
    ):
        raise ValueError("Vituri mandatory dF response informativeness was not executed")
    if (
        generic_receipt.exact_unitary_projector_probes_executed is not True
        or generic_receipt.exact_unitary_projector_probe_count
        != len(candidate.validation_plan.unitary_projector_probes)
    ):
        raise ValueError("Vituri preregistered exact-unitary E/F probes were not executed")
    if (
        candidate.evidence.evidence_kind == "immutable_provider_artifact"
        and generic_receipt.source_fock_fingerprint != preflight.source_fock_fingerprint
    ):
        raise ValueError("immutable provider F(P0) differs from exact payload Fock")
    return Vituri2024TDHFFullScalarQualificationReceipt(
        _factory_token=_QUALIFICATION_TOKEN,
        api_version=VITURI2024_FULL_SCALAR_API_VERSION,
        preflight_fingerprint=preflight.fingerprint,
        candidate_fingerprint=candidate.fingerprint,
        generic_receipt_fingerprint=generic_receipt.fingerprint,
        generic_registered_probe_consistency_consumed=True,
        generic_full_projector_consistency_consumed=False,
        dF_response_informativeness_consumed=True,
        exact_unitary_execution_consumed=True,
        evidence_kind=candidate.evidence.evidence_kind,
        immutable_evidence_verified=candidate.evidence.immutable_artifact_verified,
        eligible_for_slurm_qualification=candidate.evidence.slurm_evidence_eligible,
    )


__all__ = [
    "VITURI2024_FULL_SCALAR_AFFINE_SUPPORT",
    "VITURI2024_FULL_SCALAR_API_VERSION",
    "VITURI2024_FULL_SCALAR_EXACT_UNITARY_SUPPORT",
    "VITURI2024_FULL_SCALAR_ONE_BODY_CONSTRUCTION",
    "VITURI2024_FULL_SCALAR_PHYSICAL_INPUT_NAMES",
    "VITURI2024_FULL_SCALAR_PREFLIGHT_STATUS",
    "VITURI2024_FULL_SCALAR_RUNTIME_LAYOUT",
    "VITURI2024_FULL_SCALAR_STORAGE_DUALITY",
    "VITURI2024_FULL_SCALAR_TOTAL_ENERGY_NORMALIZATION",
    "Vituri2024FullScalarImmutableVerificationRequest",
    "Vituri2024FullScalarImmutableVerificationSnapshot",
    "Vituri2024FullScalarNormalizationReceipt",
    "Vituri2024TDHFFullScalarCandidate",
    "Vituri2024TDHFFullScalarCandidateEvidence",
    "Vituri2024TDHFFullScalarLineage",
    "Vituri2024TDHFFullScalarPreflightReceipt",
    "Vituri2024TDHFFullScalarQualificationReceipt",
    "Vituri2024TDHFFullScalarSource",
    "Vituri2024TDHFFullScalarSupport",
    "certify_vituri2024_full_scalar_orientation_and_normalization",
    "consume_vituri2024_tdhf_full_scalar_validation",
    "make_vituri2024_tdhf_full_scalar_immutable_evidence",
    "make_vituri2024_tdhf_full_scalar_lineage",
    "make_vituri2024_tdhf_full_scalar_synthetic_evidence",
    "preflight_vituri2024_tdhf_full_scalar_candidate",
    "vituri2024_full_operator_to_payload_k_diagonal",
    "vituri2024_full_projector_to_payload_density",
    "vituri2024_payload_density_to_full_projector",
    "vituri2024_payload_operator_to_full_dense",
    "validate_vituri2024_native_to_raw_total_factor",
    "vituri2024_tdhf_full_scalar_source_from_payload",
]
