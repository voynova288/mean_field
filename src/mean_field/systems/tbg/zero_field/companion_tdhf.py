"""Stage7A diagnostic-only finite-q TDHF for the pinned companion TBG lane.

Physics authority is Kwan et al., arXiv:2511.21683v1, Eqs. (15)-(18),
(64), (82)-(84), and (88)-(90).  This module consumes an immutable
:class:`TBGZeroFieldCompanionPreparedHFAction` and qualified Stage6 final
arrays.  In particular, it uses ``prepared.form`` and
``prepared.screened_intFT_ev`` directly: no additional ``1/Nk``, ``1/2``,
area, dielectric, beta, or other normalization is introduced.

System-local architecture exception
-----------------------------------
The companion q label remains system-local only because its reciprocal carry
is inseparable from the pinned companion form-factor support.  Static-kernel
assembly therefore stays here, while eigenproblem solution, degenerate metric
normalization, and stability classification come from ``mean_field.core.hf``.
This file is not exported from ``mean_field.systems.tbg.zero_field``; it
defines no runner, plotter, artifact writer, package front door, production
spectrum, or Fig. 8 claim.  Generic finite-q TDHF authority is explicitly out
of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Final, Literal, Sequence

import numpy as np

from mean_field.core.hf import (
    ParticleHolePair,
    TDHFGenericSignedQ,
    TDHFGenericSignedQSector,
    TDHFNambuSewing,
    TDHFSelfConjugateQ,
    TDHFSelfConjugateQSector,
    TDHFSignedQBlocks,
    TDHFTypedSector,
    build_tdhf_self_conjugate_matrices,
    build_tdhf_signed_q_matrices,
    classify_tdhf_signed_q,
    fingerprint_tdhf_pairs,
    TDHFStabilityClassification,
    classify_tdhf_stability,
    signed_q_particle_hole_assignment_residual,
    solve_tdhf_liouvillian,
)

from .companion_hf_action import (
    TBGZeroFieldCompanionPreparedHFAction,
    calc_fock_matrix,
)
from .companion_hf_scf import (
    TBGZeroFieldCompanionHFSCFRun,
    companion_aufbau,
)

TBG_ZERO_FIELD_COMPANION_TDHF_SCHEMA: Final[str] = (
    "mean_field.tbg.zero_field.companion_tdhf"
)
TBG_ZERO_FIELD_COMPANION_TDHF_SCHEMA_VERSION: Final[int] = 1
TBG_ZERO_FIELD_COMPANION_TDHF_SCOPE: Final[str] = (
    "Stage7A_diagnostic_only_not_generic_TDHF_authority_production_or_Fig8"
)
TBG_ZERO_FIELD_COMPANION_TDHF_ARCHITECTURE_EXCEPTION: Final[str] = (
    "system_local_q_label_only_binds_companion_form_and_reciprocal_carry;"
    "static_kernel_preserves_source_form_screened_intFT_and_projector_orientation;"
    "generic_eigensolver_metric_normalization_and_stability_from_core_hf;"
    "not_package_export_not_production_not_Fig8"
)
TBG_ZERO_FIELD_COMPANION_TDHF_PAPER_ARXIV: Final[str] = "2511.21683v1"
TBG_ZERO_FIELD_COMPANION_TDHF_PAPER_EQUATIONS: Final[tuple[int, ...]] = (
    15,
    16,
    17,
    18,
    64,
    82,
    83,
    84,
    88,
    89,
    90,
)
TBG_ZERO_FIELD_COMPANION_TDHF_ENERGY_UNITS: Final[str] = "eV"
TBG_ZERO_FIELD_COMPANION_TDHF_SUMMARY_SCHEMA: Final[str] = (
    "mean_field.tbg.kwan2511_fig8a.stage6_hf_diagnostic"
)
TBG_ZERO_FIELD_COMPANION_TDHF_SUMMARY_SCHEMA_VERSION: Final[int] = 1
TBG_ZERO_FIELD_COMPANION_TDHF_EVIDENCE_BUNDLE_SCHEMA: Final[str] = (
    "mean_field.tbg.kwan2511.stage6_hfdiag_evidence_bundle"
)
TBG_ZERO_FIELD_COMPANION_TDHF_EVIDENCE_BUNDLE_SCHEMA_VERSION: Final[int] = 1
TBG_ZERO_FIELD_COMPANION_TDHF_DIAGNOSTIC_CONSUMPTION_SCOPE: Final[str] = (
    "Stage7A_diagnostic_consumption_only_not_source_authority_"
    "not_restart_authority_not_production_authority_not_Fig8_authority"
)
TBG_ZERO_FIELD_COMPANION_TDHF_JOB_REQUIRED_ARRAY_KEYS: Final[tuple[str, ...]] = (
    "final_projector_mixed_stored",
    "reference_projector_stored",
    "final_H_total_ev",
    "final_hf_eigenvalues_ev",
    "final_hf_eigenvectors",
    "final_fill_indices",
    "final_projector_aufbau_stored",
)
TBG_ZERO_FIELD_COMPANION_TDHF_EQ90_SIGN_CONVENTION: Final[str] = (
    "paper_occupation_sign=n_mu(k+q)-n_nu(k)=-core_metric_sign;"
    "Eq90_is_K_phi=paper_occupation_sign*eta*omega*phi;"
    "L=J_core*K;lambda_L=-eta*omega"
)
TBG_ZERO_FIELD_COMPANION_TDHF_MAX_MIXED_AUFBAU_CLOSURE: Final[float] = 1.0e-8
TBG_ZERO_FIELD_COMPANION_TDHF_SOURCE_ARRAY_ATOL: Final[float] = 1.0e-12
TBG_ZERO_FIELD_COMPANION_TDHF_EIGENSYSTEM_ATOL: Final[float] = 1.0e-10
TBG_ZERO_FIELD_COMPANION_TDHF_SPIN_EQUIVALENCE_ATOL: Final[float] = 1.0e-10
TBG_ZERO_FIELD_COMPANION_TDHF_COMMON_SPIN_BASIS_ATOL_EV: Final[float] = 1.0e-10
TBG_ZERO_FIELD_COMPANION_TDHF_COMMON_SPIN_BASIS_SOURCE: Final[str] = (
    "Stage6 companion spin Hamiltonian blocks are exactly identical by construction;"
    "therefore spin-0 eigenvectors and eigenvalues must solve the spin-1 block"
)
TBG_ZERO_FIELD_COMPANION_TDHF_STATIC_STRUCTURE_ATOL_EV: Final[float] = 1.0e-10
TBG_ZERO_FIELD_COMPANION_TDHF_EQ16_ATOL: Final[float] = 1.0e-10
TBG_ZERO_FIELD_COMPANION_TDHF_EQ16_WEIGHT_ATOL_EV: Final[float] = 1.0e-14
TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV: Final[float] = 1.0e-10
TBG_ZERO_FIELD_COMPANION_TDHF_METRIC_CLASSIFICATION_ATOL: Final[float] = 1.0e-10
TBG_ZERO_FIELD_COMPANION_TDHF_DEGENERACY_ATOL_EV: Final[float] = 1.0e-10
TBG_ZERO_FIELD_COMPANION_TDHF_RAW_EIGENSOLVER_RESIDUAL_ATOL_EV: Final[float] = 1.0e-9
TBG_ZERO_FIELD_COMPANION_TDHF_SELECTED_EIGENSOLVER_RESIDUAL_ATOL_EV: Final[float] = (
    1.0e-9
)
TBG_ZERO_FIELD_COMPANION_TDHF_Q0_RAW_PAIRING_RESIDUAL_ATOL_EV: Final[float] = (
    1.0e-9
)
TBG_ZERO_FIELD_COMPANION_TDHF_ARRAY_HASH_CONVENTION: Final[str] = (
    "sha256_little_endian_int64_shape_then_C_order_canonical_array_bytes"
)

SpinSector = Literal["triplet", "singlet"]
TransitionRole = Literal["ph", "hp"]
SourceKind = Literal[
    "typed_stage6_run",
    "stage6_diagnostic_artifacts",
    "in_memory_diagnostic_arrays",
]


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer (bool is not accepted)")
    return int(value)


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 hexadecimal string")
    resolved = value.strip().lower()
    if len(resolved) != 64 or any(c not in "0123456789abcdef" for c in resolved):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    return resolved


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_array(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind == "c":
        dtype = np.dtype("<c16")
    elif source.dtype.kind == "f":
        dtype = np.dtype("<f8")
    elif source.dtype.kind in "iu":
        dtype = np.dtype("<i8")
    elif source.dtype.kind == "b":
        dtype = np.dtype("?")
    else:
        raise TypeError(f"Unsupported canonical array dtype {source.dtype}")
    return np.ascontiguousarray(source, dtype=dtype)


def _array_sha256(values: np.ndarray) -> str:
    array = _canonical_array(values)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _readonly_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> np.ndarray:
    array = np.array(values, dtype=dtype, order="C", copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _validate_live_array(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must remain a numpy.ndarray")
    expected_dtype = np.dtype(dtype)
    if values.shape != shape:
        raise ValueError(f"{name} must retain shape {shape}, got {values.shape}")
    if values.dtype != expected_dtype:
        raise ValueError(
            f"{name} must retain dtype {expected_dtype.str}, got {values.dtype.str}"
        )
    if not values.flags.c_contiguous:
        raise ValueError(f"{name} must remain C-contiguous")
    if values.flags.writeable:
        raise ValueError(f"{name} must remain read-only")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _max_abs(values: np.ndarray) -> float:
    array = np.asarray(values)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def _max_hermiticity_residual(values: np.ndarray) -> float:
    array = np.asarray(values)
    return _max_abs(array - np.swapaxes(array.conj(), -1, -2))


def _source_shapes(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    params = prepared.params
    dimension = 4 * params.n_active
    hamiltonian = (params.N1, params.N2, 2, dimension, dimension)
    spectrum = (params.N1, params.N2, 2, dimension)
    vectors = (params.N1, params.N2, 2, dimension, dimension)
    return hamiltonian, spectrum, vectors


def _electron_count(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
    filling: object,
) -> tuple[int, int]:
    resolved_filling = _strict_int(filling, name="filling")
    params = prepared.params
    dimension = 4 * params.n_active
    count = params.N1 * params.N2 * (dimension + resolved_filling)
    total = params.N1 * params.N2 * 2 * dimension
    if count <= 0 or count >= total:
        raise ValueError(
            "Stage7A requires both occupied and unoccupied states; "
            f"filling={resolved_filling} gives {count}/{total} occupied"
        )
    return resolved_filling, count


def _occupations_from_fill_indices(
    fill_indices: np.ndarray,
    *,
    spectrum_shape: tuple[int, ...],
) -> np.ndarray:
    indices = np.asarray(fill_indices, dtype=np.int64)
    size = math.prod(spectrum_shape)
    if indices.ndim != 1:
        raise ValueError("fill_indices must be one-dimensional")
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError("fill_indices contain an out-of-range flat state index")
    if np.unique(indices).size != indices.size:
        raise ValueError("fill_indices must contain unique flat state indices")
    occupations = np.zeros(size, dtype=np.int8)
    occupations[indices] = 1
    return np.reshape(occupations, spectrum_shape, order="C")


def _validate_git_commit(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a 40-character hexadecimal Git commit")
    resolved = value.strip().lower()
    if len(resolved) != 40 or any(c not in "0123456789abcdef" for c in resolved):
        raise ValueError(f"{name} must be a 40-character hexadecimal Git commit")
    return resolved


def _validate_job_id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.isdigit() or not value:
        raise ValueError(f"{name} must be a nonempty decimal string")
    return value


@dataclass(frozen=True, slots=True)
class Stage7ADiagnosticEvidenceRecord:
    """One evidence-bundle record whose path, size, and bytes were rechecked."""

    name: str
    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("evidence record name must be a nonempty string")
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ValueError("evidence record relative_path must be a nonempty string")
        relative = Path(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("evidence record path must stay relative to the bundle root")
        object.__setattr__(
            self,
            "sha256",
            _validate_sha256(self.sha256, name=f"evidence record {self.name} sha256"),
        )
        size = _strict_int(self.size_bytes, name=f"evidence record {self.name} size_bytes")
        if size < 0:
            raise ValueError("evidence record size_bytes must be nonnegative")
        object.__setattr__(self, "size_bytes", size)

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class Stage7ADiagnosticConsumptionReceipt:
    """Hash-bound permission to consume Stage6 bytes for one diagnostic only.

    This receipt does not grant source, restart, production, or Fig. 8 authority.
    It preserves the original Stage6 summary/state/evidence scopes and limitations
    rather than widening them when Stage7A consumes the arrays.
    """

    consumption_scope: str
    state_path: str
    state_sha256: str
    state_size_bytes: int
    summary_path: str
    summary_sha256: str
    summary_schema: str
    summary_schema_version: int
    summary_status: str
    summary_scope: str
    summary_limitations: tuple[str, ...]
    state_scope: str
    evidence_bundle_path: str
    evidence_bundle_sha256: str
    evidence_bundle_size_bytes: int
    evidence_schema: str
    evidence_schema_version: int
    evidence_status: str
    evidence_scope: str
    evidence_limitations: tuple[str, ...]
    job_id: str
    source_commit: str
    prepared_fingerprint: str
    summary_fingerprints: tuple[tuple[str, str], ...]
    records: tuple[Stage7ADiagnosticEvidenceRecord, ...]

    def __post_init__(self) -> None:
        if self.consumption_scope != (
            TBG_ZERO_FIELD_COMPANION_TDHF_DIAGNOSTIC_CONSUMPTION_SCOPE
        ):
            raise ValueError("Stage7A receipt must remain diagnostic-consumption-only")
        for name in ("state_path", "summary_path", "evidence_bundle_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        for name in ("state_sha256", "summary_sha256", "evidence_bundle_sha256"):
            object.__setattr__(
                self,
                name,
                _validate_sha256(getattr(self, name), name=name),
            )
        for name in ("state_size_bytes", "evidence_bundle_size_bytes"):
            size = _strict_int(getattr(self, name), name=name)
            if size <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, size)
        if self.summary_schema != TBG_ZERO_FIELD_COMPANION_TDHF_SUMMARY_SCHEMA:
            raise ValueError("receipt summary schema is not the pinned Stage6 schema")
        if _strict_int(self.summary_schema_version, name="summary_schema_version") != 1:
            raise ValueError("receipt summary schema version is unsupported")
        if self.summary_status != "pass":
            raise ValueError("receipt summary status must be pass")
        if self.evidence_schema != TBG_ZERO_FIELD_COMPANION_TDHF_EVIDENCE_BUNDLE_SCHEMA:
            raise ValueError("receipt evidence schema is not the pinned Stage6 schema")
        if _strict_int(self.evidence_schema_version, name="evidence_schema_version") != 1:
            raise ValueError("receipt evidence schema version is unsupported")
        if self.evidence_status != "pass":
            raise ValueError("receipt evidence status must be pass")
        for name in (
            "summary_scope",
            "state_scope",
            "evidence_scope",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must preserve a nonempty original scope")
        for name in ("summary_limitations", "evidence_limitations"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values or not all(
                isinstance(value, str) and value for value in values
            ):
                raise ValueError(f"{name} must preserve nonempty original limitations")
        object.__setattr__(self, "job_id", _validate_job_id(self.job_id, name="job_id"))
        object.__setattr__(
            self,
            "source_commit",
            _validate_git_commit(self.source_commit, name="source_commit"),
        )
        object.__setattr__(
            self,
            "prepared_fingerprint",
            _validate_sha256(self.prepared_fingerprint, name="prepared_fingerprint"),
        )
        fingerprint_names: set[str] = set()
        validated_fingerprints: list[tuple[str, str]] = []
        for name, digest in self.summary_fingerprints:
            if not isinstance(name, str) or not name or name in fingerprint_names:
                raise ValueError("summary fingerprint names must be unique nonempty strings")
            fingerprint_names.add(name)
            validated_fingerprints.append(
                (name, _validate_sha256(digest, name=f"summary fingerprint {name}"))
            )
        object.__setattr__(self, "summary_fingerprints", tuple(validated_fingerprints))
        if dict(self.summary_fingerprints).get("prepared_hf_action") != (
            self.prepared_fingerprint
        ):
            raise ValueError("receipt prepared fingerprint is not summary-bound")
        if not self.records or tuple(record.name for record in self.records) != tuple(
            sorted(record.name for record in self.records)
        ):
            raise ValueError("evidence records must be nonempty and sorted by unique name")
        if len({record.name for record in self.records}) != len(self.records):
            raise ValueError("evidence record names must be unique")

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "consumption_scope": self.consumption_scope,
                "evidence_bundle_path": self.evidence_bundle_path,
                "evidence_bundle_sha256": self.evidence_bundle_sha256,
                "evidence_bundle_size_bytes": self.evidence_bundle_size_bytes,
                "evidence_limitations": list(self.evidence_limitations),
                "evidence_schema": self.evidence_schema,
                "evidence_schema_version": self.evidence_schema_version,
                "evidence_scope": self.evidence_scope,
                "evidence_status": self.evidence_status,
                "job_id": self.job_id,
                "prepared_fingerprint": self.prepared_fingerprint,
                "records": [record.to_metadata() for record in self.records],
                "source_commit": self.source_commit,
                "state_path": self.state_path,
                "state_scope": self.state_scope,
                "state_sha256": self.state_sha256,
                "state_size_bytes": self.state_size_bytes,
                "summary_fingerprints": dict(self.summary_fingerprints),
                "summary_limitations": list(self.summary_limitations),
                "summary_path": self.summary_path,
                "summary_schema": self.summary_schema,
                "summary_schema_version": self.summary_schema_version,
                "summary_scope": self.summary_scope,
                "summary_sha256": self.summary_sha256,
                "summary_status": self.summary_status,
            }
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionTDHFSourceArrayHashes:
    final_projector_mixed: str
    reference: str
    H_total_ev: str
    eigenvalues_ev: str
    eigenvectors: str
    fill_indices: str
    occupations: str
    final_projector_aufbau: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _validate_sha256(getattr(self, name), name=name),
            )

    @classmethod
    def from_arrays(
        cls,
        *,
        final_projector_mixed: np.ndarray,
        reference: np.ndarray,
        H_total_ev: np.ndarray,
        eigenvalues_ev: np.ndarray,
        eigenvectors: np.ndarray,
        fill_indices: np.ndarray,
        occupations: np.ndarray,
        final_projector_aufbau: np.ndarray,
    ) -> "TBGZeroFieldCompanionTDHFSourceArrayHashes":
        return cls(
            final_projector_mixed=_array_sha256(final_projector_mixed),
            reference=_array_sha256(reference),
            H_total_ev=_array_sha256(H_total_ev),
            eigenvalues_ev=_array_sha256(eigenvalues_ev),
            eigenvectors=_array_sha256(eigenvectors),
            fill_indices=_array_sha256(fill_indices),
            occupations=_array_sha256(occupations),
            final_projector_aufbau=_array_sha256(final_projector_aufbau),
        )

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionTDHFSourceResiduals:
    live_evaluation_H_total_max_abs_ev: float
    eigensolver_eigenvalue_max_abs_ev: float
    checkpoint_eigenpair_max_abs_ev: float
    eigenvector_unitarity_max_abs: float
    aufbau_projector_max_abs: float
    live_aufbau_projector_max_abs: float
    mixed_aufbau_closure: float
    spin_hamiltonian_max_abs_ev: float
    common_spin_basis_eigenpair_max_abs_ev: float
    spin_mixed_projector_max_abs: float
    spin_aufbau_projector_max_abs: float
    spin_occupied_subspace_max_abs: float
    positive_gap_ev: float

    def to_metadata(self) -> dict[str, float]:
        """Serialize every source gate; no residual is omitted from provenance."""

        return {
            name: float(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionTDHFSource:
    """Strict Stage4/6-bound source for one normalized spin channel.

    Spin zero supplies the canonical eigengauge.  Spin one is checked through
    its Hamiltonian and occupied projectors/subspaces, never through entrywise
    equality of eigenvectors inside degenerate subspaces.
    """

    prepared: TBGZeroFieldCompanionPreparedHFAction
    prepared_fingerprint: str
    source_kind: SourceKind
    source_artifact_sha256: str | None
    filling: int
    electron_count: int
    final_projector_mixed: np.ndarray
    reference: np.ndarray
    H_total_ev: np.ndarray
    eigenvalues_ev: np.ndarray
    eigenvectors: np.ndarray
    fill_indices: np.ndarray
    occupations: np.ndarray
    final_projector_aufbau: np.ndarray
    source_summary_sha256: str | None = None
    diagnostic_consumption_receipt: Stage7ADiagnosticConsumptionReceipt | None = None
    stage6_run: TBGZeroFieldCompanionHFSCFRun | None = None
    stage6_run_fingerprint: str | None = None
    array_hashes: TBGZeroFieldCompanionTDHFSourceArrayHashes = field(init=False)
    residuals: TBGZeroFieldCompanionTDHFSourceResiduals = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, TBGZeroFieldCompanionPreparedHFAction):
            raise TypeError("prepared must be TBGZeroFieldCompanionPreparedHFAction")
        live_prepared = self.prepared.fingerprint
        object.__setattr__(
            self,
            "prepared_fingerprint",
            _validate_sha256(self.prepared_fingerprint, name="prepared_fingerprint"),
        )
        if self.prepared_fingerprint != live_prepared:
            raise ValueError("prepared_fingerprint does not match live prepared input")
        if self.source_kind not in (
            "typed_stage6_run",
            "stage6_diagnostic_artifacts",
            "in_memory_diagnostic_arrays",
        ):
            raise ValueError("unsupported Stage7A source_kind")
        if self.source_kind == "in_memory_diagnostic_arrays":
            if self.source_artifact_sha256 is not None:
                raise ValueError("in-memory diagnostic arrays cannot carry an artifact SHA")
        else:
            object.__setattr__(
                self,
                "source_artifact_sha256",
                _validate_sha256(
                    self.source_artifact_sha256,
                    name="source_artifact_sha256",
                ),
            )
        if self.source_summary_sha256 is not None:
            object.__setattr__(
                self,
                "source_summary_sha256",
                _validate_sha256(
                    self.source_summary_sha256,
                    name="source_summary_sha256",
                ),
            )
            if self.source_kind != "stage6_diagnostic_artifacts":
                raise ValueError(
                    "only stage6_diagnostic_artifacts may carry a source summary SHA"
                )
        if self.source_kind == "stage6_diagnostic_artifacts":
            if not isinstance(
                self.diagnostic_consumption_receipt,
                Stage7ADiagnosticConsumptionReceipt,
            ):
                raise TypeError(
                    "stage6_diagnostic_artifacts require a diagnostic consumption receipt"
                )
            receipt = self.diagnostic_consumption_receipt
            if receipt.prepared_fingerprint != self.prepared_fingerprint:
                raise ValueError("diagnostic receipt does not bind the live prepared input")
            if receipt.state_sha256 != self.source_artifact_sha256:
                raise ValueError("diagnostic receipt state SHA does not bind source arrays")
            if receipt.summary_sha256 != self.source_summary_sha256:
                raise ValueError("diagnostic receipt summary SHA does not bind source arrays")
        elif self.diagnostic_consumption_receipt is not None:
            raise ValueError(
                "only path-loaded Stage6 diagnostic artifacts may carry a consumption receipt"
            )
        filling, count = _electron_count(self.prepared, self.filling)
        if _strict_int(self.electron_count, name="electron_count") != count:
            raise ValueError("electron_count does not match the Stage6 filling formula")
        object.__setattr__(self, "filling", filling)
        object.__setattr__(self, "electron_count", count)

        hshape, eshape, vshape = _source_shapes(self.prepared)
        arrays = {
            "final_projector_mixed": _readonly_array(
                self.final_projector_mixed,
                name="final_projector_mixed",
                shape=hshape,
                dtype=np.complex128,
            ),
            "reference": _readonly_array(
                self.reference,
                name="reference",
                shape=hshape,
                dtype=np.complex128,
            ),
            "H_total_ev": _readonly_array(
                self.H_total_ev,
                name="H_total_ev",
                shape=hshape,
                dtype=np.complex128,
            ),
            "eigenvalues_ev": _readonly_array(
                self.eigenvalues_ev,
                name="eigenvalues_ev",
                shape=eshape,
                dtype=np.float64,
            ),
            "eigenvectors": _readonly_array(
                self.eigenvectors,
                name="eigenvectors",
                shape=vshape,
                dtype=np.complex128,
            ),
            "fill_indices": _readonly_array(
                self.fill_indices,
                name="fill_indices",
                shape=(count,),
                dtype=np.int64,
            ),
            "occupations": _readonly_array(
                self.occupations,
                name="occupations",
                shape=eshape,
                dtype=np.int8,
            ),
            "final_projector_aufbau": _readonly_array(
                self.final_projector_aufbau,
                name="final_projector_aufbau",
                shape=hshape,
                dtype=np.complex128,
            ),
        }
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

        if self.source_kind == "typed_stage6_run":
            if not isinstance(self.stage6_run, TBGZeroFieldCompanionHFSCFRun):
                raise TypeError("typed_stage6_run source requires a typed Stage6 run")
            run_fingerprint = self.stage6_run.fingerprint
            if self.stage6_run_fingerprint is None:
                object.__setattr__(self, "stage6_run_fingerprint", run_fingerprint)
            else:
                object.__setattr__(
                    self,
                    "stage6_run_fingerprint",
                    _validate_sha256(
                        self.stage6_run_fingerprint,
                        name="stage6_run_fingerprint",
                    ),
                )
            if self.stage6_run_fingerprint != run_fingerprint:
                raise ValueError("stage6_run_fingerprint does not match the live run")
            if self.source_artifact_sha256 != run_fingerprint:
                raise ValueError(
                    "typed Stage6 source_artifact_sha256 must be its live run fingerprint"
                )
        elif self.stage6_run is not None or self.stage6_run_fingerprint is not None:
            raise ValueError("array-backed Stage7A source must not carry a Stage6 run")

        hashes = TBGZeroFieldCompanionTDHFSourceArrayHashes.from_arrays(**arrays)
        object.__setattr__(self, "array_hashes", hashes)
        residuals = self._compute_validation(arrays)
        object.__setattr__(self, "residuals", residuals)
        self._enforce_validation(residuals)

    @property
    def dimension(self) -> int:
        return 4 * self.prepared.params.n_active

    @property
    def canonical_eigenvalues_ev(self) -> np.ndarray:
        return self.eigenvalues_ev[:, :, 0]

    @property
    def canonical_eigenvectors(self) -> np.ndarray:
        return self.eigenvectors[:, :, 0]

    @property
    def canonical_occupations(self) -> np.ndarray:
        return self.occupations[:, :, 0]

    @property
    def gap_ev(self) -> float:
        return self.residuals.positive_gap_ev

    def _live_arrays(self) -> dict[str, np.ndarray]:
        hshape, eshape, vshape = _source_shapes(self.prepared)
        return {
            "final_projector_mixed": _validate_live_array(
                self.final_projector_mixed,
                name="source.final_projector_mixed",
                shape=hshape,
                dtype=np.complex128,
            ),
            "reference": _validate_live_array(
                self.reference,
                name="source.reference",
                shape=hshape,
                dtype=np.complex128,
            ),
            "H_total_ev": _validate_live_array(
                self.H_total_ev,
                name="source.H_total_ev",
                shape=hshape,
                dtype=np.complex128,
            ),
            "eigenvalues_ev": _validate_live_array(
                self.eigenvalues_ev,
                name="source.eigenvalues_ev",
                shape=eshape,
                dtype=np.float64,
            ),
            "eigenvectors": _validate_live_array(
                self.eigenvectors,
                name="source.eigenvectors",
                shape=vshape,
                dtype=np.complex128,
            ),
            "fill_indices": _validate_live_array(
                self.fill_indices,
                name="source.fill_indices",
                shape=(self.electron_count,),
                dtype=np.int64,
            ),
            "occupations": _validate_live_array(
                self.occupations,
                name="source.occupations",
                shape=eshape,
                dtype=np.int8,
            ),
            "final_projector_aufbau": _validate_live_array(
                self.final_projector_aufbau,
                name="source.final_projector_aufbau",
                shape=hshape,
                dtype=np.complex128,
            ),
        }

    def _validate_stage6_binding(self, arrays: dict[str, np.ndarray]) -> None:
        if self.source_kind != "typed_stage6_run":
            return
        assert self.stage6_run is not None
        live_run_fingerprint = self.stage6_run.fingerprint
        if live_run_fingerprint != self.stage6_run_fingerprint:
            raise ValueError("Stage6 run binding drifted")
        if live_run_fingerprint != self.source_artifact_sha256:
            raise ValueError("typed Stage6 source artifact SHA drifted")
        run = self.stage6_run
        run_arrays = {
            "final_projector_mixed": run.final_projector_mixed,
            "reference": run.reference,
            "H_total_ev": run.final_evaluation.H_total_ev,
            "eigenvalues_ev": np.reshape(
                run.final_aufbau.eigenvalues_ev,
                arrays["eigenvalues_ev"].shape,
                order="C",
            ),
            "eigenvectors": np.reshape(
                run.final_aufbau.eigenvectors,
                arrays["eigenvectors"].shape,
                order="C",
            ),
            "fill_indices": run.final_aufbau.fill_indices,
            "occupations": _occupations_from_fill_indices(
                run.final_aufbau.fill_indices,
                spectrum_shape=arrays["occupations"].shape,
            ),
            "final_projector_aufbau": run.final_aufbau.projector,
        }
        for name, expected in run_arrays.items():
            if not np.array_equal(arrays[name], expected):
                raise ValueError(f"source {name} no longer equals the bound Stage6 run")
        if not run.converged:
            raise ValueError("typed Stage6 run must have source-converged")

    def _compute_validation(
        self,
        arrays: dict[str, np.ndarray],
    ) -> TBGZeroFieldCompanionTDHFSourceResiduals:
        self._validate_stage6_binding(arrays)
        evaluation = self.prepared.evaluate(
            arrays["final_projector_mixed"],
            arrays["reference"],
        )
        evaluation_residual = _max_abs(
            arrays["H_total_ev"] - evaluation.H_total_ev
        )
        if evaluation_residual > TBG_ZERO_FIELD_COMPANION_TDHF_SOURCE_ARRAY_ATOL:
            raise ValueError("H_total_ev does not match live prepared.evaluate")

        hshape, eshape, _vshape = _source_shapes(self.prepared)
        dimension = hshape[-1]
        matrices = np.reshape(
            arrays["H_total_ev"],
            (-1, dimension, dimension),
            order="C",
        )
        reference_eigenvalues = np.empty((matrices.shape[0], dimension), dtype=np.float64)
        for sector, matrix in enumerate(matrices):
            reference_eigenvalues[sector] = np.linalg.eigvalsh(matrix)
        reference_eigenvalues = np.reshape(reference_eigenvalues, eshape, order="C")
        eigensolver_residual = _max_abs(
            arrays["eigenvalues_ev"] - reference_eigenvalues
        )

        vectors = np.reshape(
            arrays["eigenvectors"],
            (-1, dimension, dimension),
            order="C",
        )
        values = np.reshape(arrays["eigenvalues_ev"], (-1, dimension), order="C")
        eigenpair_residual = _max_abs(
            matrices @ vectors - vectors * values[:, None, :]
        )
        identity = np.eye(dimension, dtype=np.complex128)
        unitarity_residual = _max_abs(
            np.swapaxes(vectors.conj(), -1, -2) @ vectors - identity
        )

        expected_fill = np.argsort(
            arrays["eigenvalues_ev"].reshape(-1, order="C")
        )[: self.electron_count].astype(np.int64, copy=False)
        if not np.array_equal(arrays["fill_indices"], expected_fill):
            raise ValueError("fill_indices do not equal the exact Stage6 C-order Aufbau fill")
        expected_occupations = _occupations_from_fill_indices(
            expected_fill,
            spectrum_shape=eshape,
        )
        if not np.array_equal(arrays["occupations"], expected_occupations):
            raise ValueError("occupations do not equal the exact fill-index inventory")
        if np.any((arrays["occupations"] != 0) & (arrays["occupations"] != 1)):
            raise ValueError("occupations must be exactly zero or one")

        fill = arrays["occupations"].astype(np.float64, copy=False)
        expected_projector = np.einsum(
            "kKsan,kKsbn,kKsn->kKsab",
            arrays["eigenvectors"].conj(),
            arrays["eigenvectors"],
            fill,
            optimize=True,
        )
        aufbau_projector_residual = _max_abs(
            arrays["final_projector_aufbau"] - expected_projector
        )
        live_aufbau = companion_aufbau(
            self.prepared,
            arrays["H_total_ev"],
            filling=self.filling,
        )
        if not np.array_equal(live_aufbau.fill_indices, arrays["fill_indices"]):
            raise ValueError("checkpoint fill_indices differ from live companion_aufbau")
        if not np.allclose(
            np.reshape(live_aufbau.eigenvalues_ev, eshape, order="C"),
            arrays["eigenvalues_ev"],
            rtol=0.0,
            atol=TBG_ZERO_FIELD_COMPANION_TDHF_SOURCE_ARRAY_ATOL,
        ):
            raise ValueError("checkpoint eigenvalues differ from live companion_aufbau")
        live_aufbau_projector_residual = _max_abs(
            arrays["final_projector_aufbau"] - live_aufbau.projector
        )

        Nk = self.prepared.params.N1 * self.prepared.params.N2
        closure = float(
            np.linalg.norm(
                arrays["final_projector_mixed"]
                - arrays["final_projector_aufbau"]
            )
            / Nk
        )
        spin_hamiltonian = _max_abs(
            arrays["H_total_ev"][:, :, 0] - arrays["H_total_ev"][:, :, 1]
        )
        # Stage6 constructs identical spin Hamiltonian blocks.  Stage7A uses one
        # common spin-0 eigengauge, so check that basis directly in spin block 1.
        common_spin_basis_eigenpair = _max_abs(
            arrays["H_total_ev"][:, :, 1] @ arrays["eigenvectors"][:, :, 0]
            - arrays["eigenvectors"][:, :, 0]
            * arrays["eigenvalues_ev"][:, :, 0, None, :]
        )
        spin_mixed = _max_abs(
            arrays["final_projector_mixed"][:, :, 0]
            - arrays["final_projector_mixed"][:, :, 1]
        )
        spin_aufbau = _max_abs(
            arrays["final_projector_aufbau"][:, :, 0]
            - arrays["final_projector_aufbau"][:, :, 1]
        )
        occupied_subspaces = np.einsum(
            "kKsan,kKsbn,kKsn->kKsab",
            arrays["eigenvectors"],
            arrays["eigenvectors"].conj(),
            fill,
            optimize=True,
        )
        spin_subspace = _max_abs(
            occupied_subspaces[:, :, 0] - occupied_subspaces[:, :, 1]
        )
        if not np.array_equal(
            arrays["occupations"][:, :, 0],
            arrays["occupations"][:, :, 1],
        ):
            raise ValueError("spin channels do not have the same occupation inventory")
        expected_per_spin = self.electron_count // 2
        if self.electron_count % 2:
            raise ValueError("normalized spin sectors require an even electron count")
        for spin in range(2):
            if int(np.sum(arrays["occupations"][:, :, spin])) != expected_per_spin:
                raise ValueError("spin channel occupied count is not exactly half the total")

        flat_values = arrays["eigenvalues_ev"].reshape(-1, order="C")
        flat_occupations = arrays["occupations"].reshape(-1, order="C").astype(bool)
        gap = float(
            np.min(flat_values[~flat_occupations])
            - np.max(flat_values[flat_occupations])
        )
        return TBGZeroFieldCompanionTDHFSourceResiduals(
            live_evaluation_H_total_max_abs_ev=evaluation_residual,
            eigensolver_eigenvalue_max_abs_ev=eigensolver_residual,
            checkpoint_eigenpair_max_abs_ev=eigenpair_residual,
            eigenvector_unitarity_max_abs=unitarity_residual,
            aufbau_projector_max_abs=aufbau_projector_residual,
            live_aufbau_projector_max_abs=live_aufbau_projector_residual,
            mixed_aufbau_closure=closure,
            spin_hamiltonian_max_abs_ev=spin_hamiltonian,
            common_spin_basis_eigenpair_max_abs_ev=common_spin_basis_eigenpair,
            spin_mixed_projector_max_abs=spin_mixed,
            spin_aufbau_projector_max_abs=spin_aufbau,
            spin_occupied_subspace_max_abs=spin_subspace,
            positive_gap_ev=gap,
        )

    @staticmethod
    def _enforce_validation(
        residuals: TBGZeroFieldCompanionTDHFSourceResiduals,
    ) -> None:
        if residuals.live_evaluation_H_total_max_abs_ev > (
            TBG_ZERO_FIELD_COMPANION_TDHF_SOURCE_ARRAY_ATOL
        ):
            raise ValueError("H_total_ev does not match live prepared.evaluate")
        if residuals.eigensolver_eigenvalue_max_abs_ev > (
            TBG_ZERO_FIELD_COMPANION_TDHF_SOURCE_ARRAY_ATOL
        ):
            raise ValueError("checkpoint eigenvalues do not match the live eigensolver")
        if residuals.checkpoint_eigenpair_max_abs_ev > (
            TBG_ZERO_FIELD_COMPANION_TDHF_EIGENSYSTEM_ATOL
        ):
            raise ValueError("checkpoint eigenvectors do not close the eigensystem")
        if residuals.eigenvector_unitarity_max_abs > (
            TBG_ZERO_FIELD_COMPANION_TDHF_EIGENSYSTEM_ATOL
        ):
            raise ValueError("checkpoint eigenvectors are not unitary")
        if residuals.aufbau_projector_max_abs > (
            TBG_ZERO_FIELD_COMPANION_TDHF_EIGENSYSTEM_ATOL
        ):
            raise ValueError("checkpoint Aufbau projector does not match occupations")
        if residuals.live_aufbau_projector_max_abs > (
            TBG_ZERO_FIELD_COMPANION_TDHF_EIGENSYSTEM_ATOL
        ):
            raise ValueError("checkpoint Aufbau projector differs from live Aufbau")
        if residuals.mixed_aufbau_closure > (
            TBG_ZERO_FIELD_COMPANION_TDHF_MAX_MIXED_AUFBAU_CLOSURE
        ):
            raise ValueError("mixed-vs-Aufbau closure exceeds 1e-8")
        for name in (
            "spin_hamiltonian_max_abs_ev",
            "spin_mixed_projector_max_abs",
            "spin_aufbau_projector_max_abs",
            "spin_occupied_subspace_max_abs",
        ):
            if getattr(residuals, name) > (
                TBG_ZERO_FIELD_COMPANION_TDHF_SPIN_EQUIVALENCE_ATOL
            ):
                raise ValueError(f"common spin gauge rejected by {name}")
        if residuals.common_spin_basis_eigenpair_max_abs_ev > (
            TBG_ZERO_FIELD_COMPANION_TDHF_COMMON_SPIN_BASIS_ATOL_EV
        ):
            raise ValueError(
                "common spin-0 eigenbasis does not solve the identical spin-1 "
                "Hamiltonian block within 1e-10 eV"
            )
        if not residuals.positive_gap_ev > 0.0:
            raise ValueError("Stage7A source requires a strictly positive HF gap")

    def _validate_live_state(self) -> None:
        if self.prepared.fingerprint != self.prepared_fingerprint:
            raise ValueError("Stage7A prepared binding drifted")
        arrays = self._live_arrays()
        actual_hashes = TBGZeroFieldCompanionTDHFSourceArrayHashes.from_arrays(**arrays)
        if actual_hashes != self.array_hashes:
            raise ValueError("Stage7A source array hashes no longer match live arrays")
        actual_residuals = self._compute_validation(arrays)
        if actual_residuals != self.residuals:
            raise ValueError("Stage7A source residual receipts no longer match live arrays")
        self._enforce_validation(actual_residuals)

    @property
    def fingerprint(self) -> str:
        self._validate_live_state()
        return _json_sha256(
            {
                "array_hashes": self.array_hashes.fingerprint,
                "common_spin_basis_source": (
                    TBG_ZERO_FIELD_COMPANION_TDHF_COMMON_SPIN_BASIS_SOURCE
                ),
                "diagnostic_consumption_receipt": (
                    None
                    if self.diagnostic_consumption_receipt is None
                    else self.diagnostic_consumption_receipt.fingerprint
                ),
                "electron_count": self.electron_count,
                "filling": self.filling,
                "paper_arxiv": TBG_ZERO_FIELD_COMPANION_TDHF_PAPER_ARXIV,
                "paper_equations": list(TBG_ZERO_FIELD_COMPANION_TDHF_PAPER_EQUATIONS),
                "prepared_fingerprint": self.prepared_fingerprint,
                "schema": TBG_ZERO_FIELD_COMPANION_TDHF_SCHEMA,
                "schema_version": TBG_ZERO_FIELD_COMPANION_TDHF_SCHEMA_VERSION,
                "scope": TBG_ZERO_FIELD_COMPANION_TDHF_SCOPE,
                "source_artifact_sha256": self.source_artifact_sha256,
                "source_residuals": self.residuals.to_metadata(),
                "source_kind": self.source_kind,
                "source_summary_sha256": self.source_summary_sha256,
                "stage6_run_fingerprint": self.stage6_run_fingerprint,
            }
        )


def build_tbg_zero_field_companion_tdhf_source_from_stage6_run(
    run: TBGZeroFieldCompanionHFSCFRun,
) -> TBGZeroFieldCompanionTDHFSource:
    """Bind Stage7A to the immutable final arrays of a typed Stage6 run."""

    if not isinstance(run, TBGZeroFieldCompanionHFSCFRun):
        raise TypeError("run must be TBGZeroFieldCompanionHFSCFRun")
    run_fingerprint = run.fingerprint
    prepared = run.prepared
    _hshape, eshape, vshape = _source_shapes(prepared)
    occupations = _occupations_from_fill_indices(
        run.final_aufbau.fill_indices,
        spectrum_shape=eshape,
    )
    return TBGZeroFieldCompanionTDHFSource(
        prepared=prepared,
        prepared_fingerprint=prepared.fingerprint,
        source_kind="typed_stage6_run",
        source_artifact_sha256=run_fingerprint,
        filling=run.spec.filling,
        electron_count=run.final_aufbau.electron_count,
        final_projector_mixed=run.final_projector_mixed,
        reference=run.reference,
        H_total_ev=run.final_evaluation.H_total_ev,
        eigenvalues_ev=np.reshape(run.final_aufbau.eigenvalues_ev, eshape, order="C"),
        eigenvectors=np.reshape(run.final_aufbau.eigenvectors, vshape, order="C"),
        fill_indices=run.final_aufbau.fill_indices,
        occupations=occupations,
        final_projector_aufbau=run.final_aufbau.projector,
        stage6_run=run,
        stage6_run_fingerprint=run_fingerprint,
    )


def build_tbg_zero_field_companion_tdhf_source_from_in_memory_arrays(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
    *,
    final_projector_mixed: np.ndarray,
    reference: np.ndarray,
    H_total_ev: np.ndarray,
    eigenvalues_ev: np.ndarray,
    eigenvectors: np.ndarray,
    fill_indices: np.ndarray,
    occupations: np.ndarray,
    final_projector_aufbau: np.ndarray,
    filling: int,
    electron_count: int,
) -> TBGZeroFieldCompanionTDHFSource:
    """Build an in-memory diagnostic source with no external artifact authority."""

    if not isinstance(prepared, TBGZeroFieldCompanionPreparedHFAction):
        raise TypeError("prepared must be TBGZeroFieldCompanionPreparedHFAction")
    arrays = {
        "final_projector_mixed": final_projector_mixed,
        "reference": reference,
        "H_total_ev": H_total_ev,
        "eigenvalues_ev": eigenvalues_ev,
        "eigenvectors": eigenvectors,
        "fill_indices": fill_indices,
        "occupations": occupations,
        "final_projector_aufbau": final_projector_aufbau,
    }
    return TBGZeroFieldCompanionTDHFSource(
        prepared=prepared,
        prepared_fingerprint=prepared.fingerprint,
        source_kind="in_memory_diagnostic_arrays",
        source_artifact_sha256=None,
        filling=filling,
        electron_count=electron_count,
        **arrays,
    )

def _read_hashed_file(
    path_value: str | Path,
    *,
    name: str,
) -> tuple[Path, bytes, str, int]:
    if not isinstance(path_value, (str, Path)):
        raise TypeError(f"{name} must be a path string or pathlib.Path")
    path = Path(path_value).expanduser().resolve()
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{name} cannot be read") from exc
    if not payload:
        raise ValueError(f"{name} must not be empty")
    return path, payload, hashlib.sha256(payload).hexdigest(), len(payload)


def _decode_json_object(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return decoded


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be a nonempty list of nonempty strings")
    return tuple(value)


def _validated_evidence_records(
    evidence: dict[str, object],
    *,
    evidence_path: Path,
) -> tuple[
    tuple[Stage7ADiagnosticEvidenceRecord, ...],
    dict[str, Path],
    dict[str, bytes],
]:
    records_value = evidence.get("records")
    if not isinstance(records_value, dict) or not records_value:
        raise ValueError("evidence bundle records must be a nonempty object")
    bundle_root = evidence_path.parent.parent.resolve()
    receipts: list[Stage7ADiagnosticEvidenceRecord] = []
    resolved_paths: dict[str, Path] = {}
    record_bytes: dict[str, bytes] = {}
    for name in sorted(records_value):
        if not isinstance(name, str) or not name:
            raise ValueError("evidence record names must be nonempty strings")
        value = records_value[name]
        if not isinstance(value, dict) or set(value) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"evidence record {name} has an invalid exact key inventory")
        relative_path = value["path"]
        if not isinstance(relative_path, str):
            raise ValueError(f"evidence record {name} path must be a string")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"evidence record {name} path escapes the bundle root")
        record_path = (bundle_root / relative).resolve()
        try:
            record_path.relative_to(bundle_root)
        except ValueError as exc:
            raise ValueError(f"evidence record {name} path escapes the bundle root") from exc
        _, payload, actual_sha, actual_size = _read_hashed_file(
            record_path,
            name=f"evidence record {name}",
        )
        expected_sha = _validate_sha256(
            value["sha256"],
            name=f"evidence record {name} sha256",
        )
        expected_size = _strict_int(
            value["size_bytes"],
            name=f"evidence record {name} size_bytes",
        )
        if actual_sha != expected_sha:
            raise ValueError(f"evidence record {name} SHA-256 does not match its bytes")
        if actual_size != expected_size:
            raise ValueError(f"evidence record {name} size does not match its bytes")
        receipts.append(
            Stage7ADiagnosticEvidenceRecord(
                name=name,
                relative_path=relative_path,
                sha256=actual_sha,
                size_bytes=actual_size,
            )
        )
        resolved_paths[name] = record_path
        record_bytes[name] = payload
    for required in ("state", "summary", "source_commit"):
        if required not in resolved_paths:
            raise ValueError(f"evidence bundle is missing required record {required}")
    return tuple(receipts), resolved_paths, record_bytes


def load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
    state_path: str | Path,
    summary_path: str | Path,
    evidence_bundle_path: str | Path,
) -> TBGZeroFieldCompanionTDHFSource:
    """Consume an exact Stage6 job-style diagnostic bundle for Stage7A only.

    Every input and every evidence-bundle record is hashed inside this loader.
    The receipt preserves the source artifact's diagnostic-only scope; it is not
    source, restart, production, or Kwan Fig. 8 authority.
    """

    if not isinstance(prepared, TBGZeroFieldCompanionPreparedHFAction):
        raise TypeError("prepared must be TBGZeroFieldCompanionPreparedHFAction")
    state_file, state_bytes, state_sha, state_size = _read_hashed_file(
        state_path,
        name="Stage6 state_path",
    )
    summary_file, summary_bytes, summary_sha, _summary_size = _read_hashed_file(
        summary_path,
        name="Stage6 summary_path",
    )
    evidence_file, evidence_bytes, evidence_sha, evidence_size = _read_hashed_file(
        evidence_bundle_path,
        name="Stage6 evidence_bundle_path",
    )
    summary = _decode_json_object(summary_bytes, name="Stage6 summary")
    evidence = _decode_json_object(evidence_bytes, name="Stage6 evidence bundle")

    if summary.get("schema") != TBG_ZERO_FIELD_COMPANION_TDHF_SUMMARY_SCHEMA:
        raise ValueError(
            "summary schema is not "
            "mean_field.tbg.kwan2511_fig8a.stage6_hf_diagnostic"
        )
    summary_version = _strict_int(
        summary.get("schema_version"),
        name="summary schema_version",
    )
    if summary_version != TBG_ZERO_FIELD_COMPANION_TDHF_SUMMARY_SCHEMA_VERSION:
        raise ValueError("summary schema_version is unsupported")
    if summary.get("status") != "pass":
        raise ValueError("summary status must be pass")
    summary_scope = summary.get("scope")
    if not isinstance(summary_scope, str) or "not_TDHF" not in summary_scope:
        raise ValueError("summary scope must explicitly remain not_TDHF")
    summary_limitations = _string_tuple(
        summary.get("limitations"),
        name="summary limitations",
    )
    if not any("No TDHF implementation" in value for value in summary_limitations):
        raise ValueError("summary limitations must explicitly state no TDHF implementation")

    if set(evidence) != {
        "schema",
        "schema_version",
        "status",
        "scope",
        "job_id",
        "limitations",
        "records",
    }:
        raise ValueError("evidence bundle has an invalid exact key inventory")
    if evidence["schema"] != TBG_ZERO_FIELD_COMPANION_TDHF_EVIDENCE_BUNDLE_SCHEMA:
        raise ValueError("evidence bundle schema is unsupported")
    evidence_version = _strict_int(
        evidence["schema_version"],
        name="evidence schema_version",
    )
    if evidence_version != TBG_ZERO_FIELD_COMPANION_TDHF_EVIDENCE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("evidence bundle schema_version is unsupported")
    if evidence["status"] != "pass":
        raise ValueError("evidence bundle status must be pass")
    evidence_scope = evidence["scope"]
    if not isinstance(evidence_scope, str) or "not_TDHF" not in evidence_scope:
        raise ValueError("evidence scope must explicitly remain not_TDHF")
    evidence_limitations = _string_tuple(
        evidence["limitations"],
        name="evidence limitations",
    )
    if not any(
        "TDHF" in value and ("no " in value.lower() or "not " in value.lower())
        for value in evidence_limitations
    ):
        raise ValueError("evidence limitations must explicitly disclaim TDHF authority")
    job_id = _validate_job_id(evidence["job_id"], name="evidence job_id")

    records, record_paths, record_bytes = _validated_evidence_records(
        evidence,
        evidence_path=evidence_file,
    )
    if record_paths["state"] != state_file:
        raise ValueError("evidence state record does not bind the supplied state_path")
    if record_paths["summary"] != summary_file:
        raise ValueError("evidence summary record does not bind the supplied summary_path")
    record_map = {record.name: record for record in records}
    if (
        record_map["state"].sha256 != state_sha
        or record_map["state"].size_bytes != state_size
    ):
        raise ValueError("evidence state record does not bind state bytes and size")
    if record_map["summary"].sha256 != summary_sha:
        raise ValueError("evidence summary record does not bind summary bytes")

    summary_job = summary.get("job")
    if not isinstance(summary_job, dict):
        raise ValueError("summary job must be an object")
    summary_job_id = _validate_job_id(summary_job.get("job_id"), name="summary job_id")
    if summary_job_id != job_id:
        raise ValueError("summary and evidence bundle job ids differ")
    allocation = summary_job.get("allocation_attestation")
    if not isinstance(allocation, dict) or _validate_job_id(
        allocation.get("job_id"),
        name="summary allocation job_id",
    ) != job_id:
        raise ValueError("summary allocation attestation does not bind the job id")

    source_commit = _validate_git_commit(
        summary.get("source_commit"),
        name="summary source_commit",
    )
    try:
        record_source_commit = record_bytes["source_commit"].decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("source_commit record must be UTF-8") from exc
    if _validate_git_commit(record_source_commit, name="record source_commit") != source_commit:
        raise ValueError("summary source_commit differs from the evidence record")

    fingerprints = summary.get("fingerprints")
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError("summary fingerprints must be a nonempty object")
    summary_fingerprints = tuple(
        sorted(
            (
                name,
                _validate_sha256(value, name=f"summary fingerprint {name}"),
            )
            for name, value in fingerprints.items()
            if isinstance(name, str) and name
        )
    )
    if len(summary_fingerprints) != len(fingerprints):
        raise ValueError("summary fingerprint names must be nonempty strings")
    prepared_fingerprint = prepared.fingerprint
    if dict(summary_fingerprints).get("prepared_hf_action") != prepared_fingerprint:
        raise ValueError("summary prepared_hf_action fingerprint does not match prepared")

    state = summary.get("state")
    if not isinstance(state, dict):
        raise ValueError("summary state must be an object")
    for key in ("array_keys", "path", "scope", "sha256", "size_bytes"):
        if key not in state:
            raise ValueError(f"summary state is missing {key}")
    if _validate_sha256(state["sha256"], name="summary state sha256") != state_sha:
        raise ValueError("summary state SHA-256 does not bind the supplied NPZ bytes")
    if _strict_int(state["size_bytes"], name="summary state size_bytes") != state_size:
        raise ValueError("summary state size does not bind the supplied NPZ bytes")
    state_declared_path = state["path"]
    if (
        not isinstance(state_declared_path, str)
        or Path(state_declared_path).expanduser().resolve() != state_file
    ):
        raise ValueError("summary state path does not bind the supplied state_path")
    state_scope = state["scope"]
    if not isinstance(state_scope, str) or (
        "not_restart_authority" not in state_scope
        or "not_TDHF_source" not in state_scope
    ):
        raise ValueError("summary state scope must remain not restart and not TDHF source")
    array_keys_value = state["array_keys"]
    if not isinstance(array_keys_value, list) or not all(
        isinstance(name, str) and name for name in array_keys_value
    ):
        raise ValueError("summary state.array_keys must be a list of nonempty strings")
    array_keys = tuple(array_keys_value)
    if array_keys != tuple(sorted(set(array_keys))):
        raise ValueError("summary state.array_keys must be sorted and unique")
    missing_required = set(
        TBG_ZERO_FIELD_COMPANION_TDHF_JOB_REQUIRED_ARRAY_KEYS
    ).difference(array_keys)
    if missing_required:
        raise ValueError(
            "summary state.array_keys omit required Stage6 arrays: "
            + ", ".join(sorted(missing_required))
        )

    target = summary.get("target")
    if not isinstance(target, dict):
        raise ValueError("summary target must be an object")
    filling, electron_count = _electron_count(prepared, target.get("filling"))
    hshape, eshape, vshape = _source_shapes(prepared)
    flat_sector_count = math.prod(eshape[:-1])
    raw_specs = {
        "final_projector_mixed_stored": (hshape, np.dtype(np.complex128)),
        "reference_projector_stored": (hshape, np.dtype(np.complex128)),
        "final_H_total_ev": (hshape, np.dtype(np.complex128)),
        "final_hf_eigenvalues_ev": ((math.prod(eshape),), np.dtype(np.float64)),
        "final_hf_eigenvectors": (
            (flat_sector_count, vshape[-2], vshape[-1]),
            np.dtype(np.complex128),
        ),
        "final_fill_indices": ((electron_count,), np.dtype(np.int64)),
        "final_projector_aufbau_stored": (hshape, np.dtype(np.complex128)),
    }
    try:
        with np.load(io.BytesIO(state_bytes), allow_pickle=False) as archive:
            if tuple(sorted(archive.files)) != array_keys:
                raise ValueError(
                    "Stage6 state NPZ inventory does not exactly equal "
                    "summary state.array_keys"
                )
            loaded = {name: np.asarray(archive[name]) for name in archive.files}
            for name, array in loaded.items():
                if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                    raise ValueError(f"Stage6 state array {name} contains nonfinite values")
            raw_arrays: dict[str, np.ndarray] = {}
            for name, (shape, dtype) in raw_specs.items():
                array = loaded[name]
                if array.shape != shape:
                    raise ValueError(
                        f"Stage6 state {name} must have exact shape {shape}, "
                        f"got {array.shape}"
                    )
                if array.dtype != dtype:
                    raise ValueError(
                        f"Stage6 state {name} must have exact dtype {dtype.str}, "
                        f"got {array.dtype.str}"
                    )
                raw_arrays[name] = np.array(array, copy=True, order="C")
    except (OSError, EOFError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Stage6 state"):
            raise
        raise ValueError("state_path is not a valid allow_pickle=False Stage6 NPZ") from exc

    fill_indices = raw_arrays["final_fill_indices"]
    occupations = _occupations_from_fill_indices(
        fill_indices,
        spectrum_shape=eshape,
    )
    arrays = {
        "final_projector_mixed": raw_arrays["final_projector_mixed_stored"],
        "reference": raw_arrays["reference_projector_stored"],
        "H_total_ev": raw_arrays["final_H_total_ev"],
        "eigenvalues_ev": np.reshape(
            raw_arrays["final_hf_eigenvalues_ev"],
            eshape,
            order="C",
        ),
        "eigenvectors": np.reshape(
            raw_arrays["final_hf_eigenvectors"],
            vshape,
            order="C",
        ),
        "fill_indices": fill_indices,
        "occupations": occupations,
        "final_projector_aufbau": raw_arrays["final_projector_aufbau_stored"],
    }
    receipt = Stage7ADiagnosticConsumptionReceipt(
        consumption_scope=TBG_ZERO_FIELD_COMPANION_TDHF_DIAGNOSTIC_CONSUMPTION_SCOPE,
        state_path=str(state_file),
        state_sha256=state_sha,
        state_size_bytes=state_size,
        summary_path=str(summary_file),
        summary_sha256=summary_sha,
        summary_schema=str(summary["schema"]),
        summary_schema_version=summary_version,
        summary_status=str(summary["status"]),
        summary_scope=summary_scope,
        summary_limitations=summary_limitations,
        state_scope=state_scope,
        evidence_bundle_path=str(evidence_file),
        evidence_bundle_sha256=evidence_sha,
        evidence_bundle_size_bytes=evidence_size,
        evidence_schema=str(evidence["schema"]),
        evidence_schema_version=evidence_version,
        evidence_status=str(evidence["status"]),
        evidence_scope=evidence_scope,
        evidence_limitations=evidence_limitations,
        job_id=job_id,
        source_commit=source_commit,
        prepared_fingerprint=prepared_fingerprint,
        summary_fingerprints=summary_fingerprints,
        records=records,
    )
    return TBGZeroFieldCompanionTDHFSource(
        prepared=prepared,
        prepared_fingerprint=prepared_fingerprint,
        source_kind="stage6_diagnostic_artifacts",
        source_artifact_sha256=state_sha,
        source_summary_sha256=summary_sha,
        diagnostic_consumption_receipt=receipt,
        filling=filling,
        electron_count=electron_count,
        **arrays,
    )



def load_tbg_zero_field_companion_tdhf_source_from_checkpoint_npz(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
    checkpoint_path: str | Path,
    summary_path: str | Path,
    evidence_bundle_path: str | Path,
) -> TBGZeroFieldCompanionTDHFSource:
    """Public legacy spelling for the exact three-path Stage6 artifact loader.

    Unlike the removed synthetic-checkpoint implementation, this spelling still
    requires the real summary and evidence bundle and computes every hash inside
    :func:`load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts`.
    """

    return load_tbg_zero_field_companion_tdhf_source_from_stage6_artifacts(
        prepared,
        checkpoint_path,
        summary_path,
        evidence_bundle_path,
    )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionSignedQLabel:
    """Exact signed raw q provenance and its torus/carry decomposition."""

    source_fingerprint: str
    N1: int
    N2: int
    raw: tuple[int, int]
    canonical_delta: tuple[int, int]
    reciprocal_carry: tuple[int, int]
    target_indices: tuple[tuple[int, int], ...]
    leg_carries: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_fingerprint",
            _validate_sha256(self.source_fingerprint, name="source_fingerprint"),
        )
        N1 = _strict_int(self.N1, name="N1")
        N2 = _strict_int(self.N2, name="N2")
        if N1 <= 0 or N2 <= 0:
            raise ValueError("N1 and N2 must be positive")
        raw = tuple(_strict_int(x, name="raw component") for x in self.raw)
        if len(raw) != 2:
            raise ValueError("raw q must contain exactly two integers")
        expected_delta = (raw[0] % N1, raw[1] % N2)
        expected_carry = (raw[0] // N1, raw[1] // N2)
        if tuple(self.canonical_delta) != expected_delta:
            raise ValueError("canonical_delta does not equal raw q modulo the mesh")
        if tuple(self.reciprocal_carry) != expected_carry:
            raise ValueError("reciprocal_carry does not equal raw q floor division")
        expected_targets: list[tuple[int, int]] = []
        expected_legs: list[tuple[int, int]] = []
        for k1 in range(N1):
            for k2 in range(N2):
                expected_targets.append(((k1 + raw[0]) % N1, (k2 + raw[1]) % N2))
                expected_legs.append(((k1 + raw[0]) // N1, (k2 + raw[1]) // N2))
        if tuple(self.target_indices) != tuple(expected_targets):
            raise ValueError("target_indices do not match raw q")
        if tuple(self.leg_carries) != tuple(expected_legs):
            raise ValueError("leg_carries do not match raw q floor division")
        object.__setattr__(self, "N1", N1)
        object.__setattr__(self, "N2", N2)
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "canonical_delta", expected_delta)
        object.__setattr__(self, "reciprocal_carry", expected_carry)
        object.__setattr__(self, "target_indices", tuple(expected_targets))
        object.__setattr__(self, "leg_carries", tuple(expected_legs))

    def target(self, k1: int, k2: int) -> tuple[int, int]:
        i1 = _strict_int(k1, name="k1")
        i2 = _strict_int(k2, name="k2")
        if not 0 <= i1 < self.N1 or not 0 <= i2 < self.N2:
            raise ValueError("k index is outside the source mesh")
        return self.target_indices[i1 * self.N2 + i2]

    def leg_carry(self, k1: int, k2: int) -> tuple[int, int]:
        i1 = _strict_int(k1, name="k1")
        i2 = _strict_int(k2, name="k2")
        if not 0 <= i1 < self.N1 or not 0 <= i2 < self.N2:
            raise ValueError("k index is outside the source mesh")
        return self.leg_carries[i1 * self.N2 + i2]

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "canonical_delta": list(self.canonical_delta),
                "leg_carries": [list(x) for x in self.leg_carries],
                "mesh": [self.N1, self.N2],
                "raw": list(self.raw),
                "reciprocal_carry": list(self.reciprocal_carry),
                "source_fingerprint": self.source_fingerprint,
                "target_indices": [list(x) for x in self.target_indices],
            }
        )


def _signed_q_label(
    source_fingerprint: str,
    N1: int,
    N2: int,
    raw: Sequence[int],
) -> TBGZeroFieldCompanionSignedQLabel:
    if isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise TypeError("raw q must be a length-two integer sequence")
    l1 = _strict_int(raw[0], name="raw q l1")
    l2 = _strict_int(raw[1], name="raw q l2")
    targets: list[tuple[int, int]] = []
    carries: list[tuple[int, int]] = []
    for k1 in range(N1):
        for k2 in range(N2):
            targets.append(((k1 + l1) % N1, (k2 + l2) % N2))
            carries.append(((k1 + l1) // N1, (k2 + l2) // N2))
    return TBGZeroFieldCompanionSignedQLabel(
        source_fingerprint=source_fingerprint,
        N1=N1,
        N2=N2,
        raw=(l1, l2),
        canonical_delta=(l1 % N1, l2 % N2),
        reciprocal_carry=(l1 // N1, l2 // N2),
        target_indices=tuple(targets),
        leg_carries=tuple(carries),
    )


def build_tbg_zero_field_companion_signed_q_label(
    source: TBGZeroFieldCompanionTDHFSource,
    raw: Sequence[int],
) -> TBGZeroFieldCompanionSignedQLabel:
    source_fingerprint = source.fingerprint
    params = source.prepared.params
    return _signed_q_label(source_fingerprint, params.N1, params.N2, raw)


def _minus_q_label(
    q: TBGZeroFieldCompanionSignedQLabel,
) -> TBGZeroFieldCompanionSignedQLabel:
    return _signed_q_label(q.source_fingerprint, q.N1, q.N2, (-q.raw[0], -q.raw[1]))


def _source_g_labels(
    source: TBGZeroFieldCompanionTDHFSource,
) -> tuple[tuple[int, int], ...]:
    prepared = source.prepared
    return tuple(
        (g1, g2)
        for g1 in range(-prepared.interaction_NG1, prepared.interaction_NG1)
        for g2 in range(-prepared.interaction_NG2, prepared.interaction_NG2)
    )


def _interaction_weights(
    source: TBGZeroFieldCompanionTDHFSource,
    q: TBGZeroFieldCompanionSignedQLabel,
) -> np.ndarray:
    d1, d2 = q.canonical_delta
    prepared = source.prepared
    return np.reshape(
        prepared.screened_intFT_ev[d1, d2],
        (-1,),
        order="C",
    )


def _band_form_matrix(
    source: TBGZeroFieldCompanionTDHFSource,
    *,
    k1: int,
    k2: int,
    delta: tuple[int, int],
    g: tuple[int, int],
) -> np.ndarray:
    prepared = source.prepared
    d1, d2 = delta
    g1, g2 = g
    iG1 = g1 + prepared.interaction_NG1
    iG2 = g2 + prepared.interaction_NG2
    if not 0 <= iG1 < 2 * prepared.interaction_NG1:
        raise ValueError("G1 is outside full prepared.form support")
    if not 0 <= iG2 < 2 * prepared.interaction_NG2:
        raise ValueError("G2 is outside full prepared.form support")
    band_count = prepared.params.active_band_count
    dimension = 2 * band_count
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for valley in range(2):
        block = prepared.form[k1, k2, d1, d2, iG1, iG2, valley]
        start = valley * band_count
        matrix[start : start + band_count, start : start + band_count] = block
    return matrix


def _compute_hf_form_factor_values(
    source: TBGZeroFieldCompanionTDHFSource,
    q: TBGZeroFieldCompanionSignedQLabel,
) -> np.ndarray:
    if q.source_fingerprint != source.fingerprint:
        raise ValueError("q label is not bound to the live Stage7A source")
    params = source.prepared.params
    labels = _source_g_labels(source)
    dimension = source.dimension
    values = np.empty(
        (params.N1, params.N2, len(labels), dimension, dimension),
        dtype=np.complex128,
    )
    C = source.canonical_eigenvectors
    for k1 in range(params.N1):
        for k2 in range(params.N2):
            t1, t2 = q.target(k1, k2)
            for iG, g in enumerate(labels):
                form = _band_form_matrix(
                    source,
                    k1=k1,
                    k2=k2,
                    delta=q.canonical_delta,
                    g=g,
                )
                values[k1, k2, iG] = C[k1, k2].conj().T @ form @ C[t1, t2]
    return values


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFFormFactorReceipt:
    q: TBGZeroFieldCompanionSignedQLabel
    source_g_labels: np.ndarray
    raw_g_labels: np.ndarray
    values: np.ndarray
    values_sha256: str
    source_g_labels_sha256: str
    raw_g_labels_sha256: str
    interaction_active_count: int
    inverse_mapped_count: int
    inverse_missing_count: int
    inverse_weight_max_abs_ev: float
    eq16_matrix_comparison_count: int
    eq16_carry_max_abs: float

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "eq16_carry_max_abs": self.eq16_carry_max_abs,
                "eq16_matrix_comparison_count": self.eq16_matrix_comparison_count,
                "interaction_active_count": self.interaction_active_count,
                "inverse_mapped_count": self.inverse_mapped_count,
                "inverse_missing_count": self.inverse_missing_count,
                "inverse_weight_max_abs_ev": self.inverse_weight_max_abs_ev,
                "q": self.q.fingerprint,
                "raw_g_labels_sha256": self.raw_g_labels_sha256,
                "source_g_labels_sha256": self.source_g_labels_sha256,
                "values_sha256": self.values_sha256,
            }
        )


def build_tbg_zero_field_companion_hf_form_factors(
    source: TBGZeroFieldCompanionTDHFSource,
    q: TBGZeroFieldCompanionSignedQLabel | Sequence[int],
) -> TBGZeroFieldCompanionHFFormFactorReceipt:
    """Build Eq. (15) HF form factors and audit the exact Eq. (16) carry."""

    label = (
        q
        if isinstance(q, TBGZeroFieldCompanionSignedQLabel)
        else build_tbg_zero_field_companion_signed_q_label(source, q)
    )
    if label.source_fingerprint != source.fingerprint:
        raise ValueError("q label is not bound to the live Stage7A source")
    labels_tuple = _source_g_labels(source)
    source_labels = np.asarray(labels_tuple, dtype=np.int64)
    raw_labels = source_labels - np.asarray(label.reciprocal_carry, dtype=np.int64)
    values = _compute_hf_form_factor_values(source, label)
    minus = _minus_q_label(label)
    minus_values = _compute_hf_form_factor_values(source, minus)
    label_to_index = {g: index for index, g in enumerate(labels_tuple)}
    if len(label_to_index) != len(labels_tuple):
        raise ValueError("Eq. (16) source G support is not unique")
    d1, d2 = label.canonical_delta
    inverse_delta = ((-d1) % label.N1, (-d2) % label.N2)
    if minus.canonical_delta != inverse_delta:
        raise ValueError("inverse q label does not have the Eq. (16) canonical delta")
    inverse_carry = ((-d1) // label.N1, (-d2) // label.N2)
    W = _interaction_weights(source, label)
    Wminus = _interaction_weights(source, minus)
    expected_weight_shape = (len(labels_tuple),)
    if W.shape != expected_weight_shape or Wminus.shape != expected_weight_shape:
        raise ValueError("Eq. (16) interaction-weight inventory does not match G support")
    active = tuple(int(index) for index in np.flatnonzero(W != 0.0))
    active_minus = set(int(index) for index in np.flatnonzero(Wminus != 0.0))
    if not active:
        raise ValueError("Eq. (16) has no nonzero interaction-active transfers")
    inverse_indices: list[int] = []
    missing_count = 0
    weight_residual = 0.0
    for iG in active:
        g1, g2 = labels_tuple[iG]
        gminus = (inverse_carry[0] - g1, inverse_carry[1] - g2)
        jG = label_to_index.get(gminus)
        if jG is None or jG not in active_minus:
            missing_count += 1
            continue
        inverse_indices.append(jG)
        weight_residual = max(weight_residual, float(abs(W[iG] - Wminus[jG])))
    if len(set(inverse_indices)) != len(inverse_indices):
        raise ValueError("Eq. (16) inverse transfer map is not one-to-one")
    missing_count += len(active_minus.difference(inverse_indices))
    if missing_count != 0 or len(inverse_indices) != len(active):
        raise ValueError(
            "Eq. (16) interaction-active inverse support is incomplete: "
            f"active={len(active)}, mapped={len(inverse_indices)}, "
            f"missing={missing_count}"
        )
    if weight_residual > TBG_ZERO_FIELD_COMPANION_TDHF_EQ16_WEIGHT_ATOL_EV:
        raise ValueError(
            "Eq. (16) inverse interaction weights differ by more than 1e-14 eV: "
            f"{weight_residual:.6e} eV"
        )

    residual = 0.0
    comparison_count = 0
    for k1 in range(label.N1):
        for k2 in range(label.N2):
            t1, t2 = label.target(k1, k2)
            for iG, jG in zip(active, inverse_indices, strict=True):
                comparison_count += 1
                residual = max(
                    residual,
                    _max_abs(
                        values[k1, k2, iG]
                        - minus_values[t1, t2, jG].conj().T
                    ),
                )
    expected_comparisons = len(active) * label.N1 * label.N2
    if comparison_count != expected_comparisons:
        raise ValueError("Eq. (16) did not compare every active transfer at every k")
    if residual > TBG_ZERO_FIELD_COMPANION_TDHF_EQ16_ATOL:
        raise ValueError(f"Eq. (16) carry residual is material: {residual:.6e}")
    for array in (source_labels, raw_labels, values):
        array.setflags(write=False)
    return TBGZeroFieldCompanionHFFormFactorReceipt(
        q=label,
        source_g_labels=source_labels,
        raw_g_labels=raw_labels,
        values=values,
        values_sha256=_array_sha256(values),
        source_g_labels_sha256=_array_sha256(source_labels),
        raw_g_labels_sha256=_array_sha256(raw_labels),
        interaction_active_count=len(active),
        inverse_mapped_count=len(inverse_indices),
        inverse_missing_count=missing_count,
        inverse_weight_max_abs_ev=weight_residual,
        eq16_matrix_comparison_count=comparison_count,
        eq16_carry_max_abs=residual,
    )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionTransition:
    """Eq. (90) transition with both paper and core sign conventions.

    The paper occupation factor is ``n_mu(k+q)-n_nu(k)``.  Stage7A stores
    ``core_metric_sign`` with the opposite sign so that ``L = J_core K`` and
    an Eq. (90) mode labeled by ``eta, omega`` has ``lambda_L = -eta*omega``.
    """

    k_source: tuple[int, int]
    k_target: tuple[int, int]
    mu_target: int
    nu_source: int
    role: TransitionRole
    core_metric_sign: int
    paper_occupation_sign: int

    def __post_init__(self) -> None:
        if self.role not in ("ph", "hp"):
            raise ValueError("transition role must be ph or hp")
        expected_core = 1 if self.role == "ph" else -1
        if _strict_int(self.core_metric_sign, name="core_metric_sign") != expected_core:
            raise ValueError("core_metric_sign does not match transition role")
        if _strict_int(
            self.paper_occupation_sign,
            name="paper_occupation_sign",
        ) != -expected_core:
            raise ValueError("paper_occupation_sign must equal -core_metric_sign")


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionTransitionInventory:
    source_fingerprint: str
    q: TBGZeroFieldCompanionSignedQLabel
    pairs: tuple[TBGZeroFieldCompanionTransition, ...]
    ph_count: int
    conjugate_indices_at_minus_q: np.ndarray
    inventory_sha256: str

    @property
    def hp_count(self) -> int:
        return len(self.pairs) - self.ph_count

    @property
    def core_metric(self) -> np.ndarray:
        """Diagonal of J_core; its negative is the Eq. (90) occupation sign."""

        result = np.asarray(
            [pair.core_metric_sign for pair in self.pairs],
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "conjugate_indices_sha256": _array_sha256(
                    self.conjugate_indices_at_minus_q
                ),
                "inventory_sha256": self.inventory_sha256,
                "ph_count": self.ph_count,
                "q": self.q.fingerprint,
                "source_fingerprint": self.source_fingerprint,
            }
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionTransitionInventoryPair:
    q: TBGZeroFieldCompanionTransitionInventory
    minus_q: TBGZeroFieldCompanionTransitionInventory


def _ph_inventory(
    source: TBGZeroFieldCompanionTDHFSource,
    q: TBGZeroFieldCompanionSignedQLabel,
) -> tuple[TBGZeroFieldCompanionTransition, ...]:
    occupations = source.canonical_occupations
    dimension = source.dimension
    result: list[TBGZeroFieldCompanionTransition] = []
    for k1 in range(q.N1):
        for k2 in range(q.N2):
            t1, t2 = q.target(k1, k2)
            for mu in range(dimension):
                if occupations[t1, t2, mu] != 0:
                    continue
                for nu in range(dimension):
                    if occupations[k1, k2, nu] == 1:
                        result.append(
                            TBGZeroFieldCompanionTransition(
                                k_source=(k1, k2),
                                k_target=(t1, t2),
                                mu_target=mu,
                                nu_source=nu,
                                role="ph",
                                core_metric_sign=1,
                                paper_occupation_sign=-1,
                            )
                        )
    return tuple(result)


def _hp_from_minus_ph(
    minus_ph: tuple[TBGZeroFieldCompanionTransition, ...],
) -> tuple[TBGZeroFieldCompanionTransition, ...]:
    return tuple(
        TBGZeroFieldCompanionTransition(
            k_source=pair.k_target,
            k_target=pair.k_source,
            mu_target=pair.nu_source,
            nu_source=pair.mu_target,
            role="hp",
            core_metric_sign=-1,
            paper_occupation_sign=1,
        )
        for pair in minus_ph
    )


def _inventory_digest(
    pairs: tuple[TBGZeroFieldCompanionTransition, ...],
) -> str:
    return _json_sha256(
        [
            {
                "k_source": list(pair.k_source),
                "k_target": list(pair.k_target),
                "core_metric_sign": pair.core_metric_sign,
                "mu_target": pair.mu_target,
                "paper_occupation_sign": pair.paper_occupation_sign,
                "nu_source": pair.nu_source,
                "role": pair.role,
            }
            for pair in pairs
        ]
    )


def build_tbg_zero_field_companion_transition_inventories(
    source: TBGZeroFieldCompanionTDHFSource,
    q: TBGZeroFieldCompanionSignedQLabel | Sequence[int],
) -> TBGZeroFieldCompanionTransitionInventoryPair:
    """Build ph first, then hp by explicit conjugation of the -q ph list."""

    source_fingerprint = source.fingerprint
    label = (
        q
        if isinstance(q, TBGZeroFieldCompanionSignedQLabel)
        else build_tbg_zero_field_companion_signed_q_label(source, q)
    )
    if label.source_fingerprint != source_fingerprint:
        raise ValueError("q label is not bound to the live Stage7A source")
    minus = _minus_q_label(label)
    ph_q = _ph_inventory(source, label)
    ph_minus = _ph_inventory(source, minus)
    hp_q = _hp_from_minus_ph(ph_minus)
    hp_minus = _hp_from_minus_ph(ph_q)
    pairs_q = ph_q + hp_q
    pairs_minus = ph_minus + hp_minus
    if not ph_q or not ph_minus:
        raise ValueError("finite-q transition inventory has no particle-hole pairs")
    map_q = np.asarray(
        [
            *(len(ph_minus) + index for index in range(len(ph_q))),
            *(index for index in range(len(ph_minus))),
        ],
        dtype=np.int64,
    )
    map_minus = np.asarray(
        [
            *(len(ph_q) + index for index in range(len(ph_minus))),
            *(index for index in range(len(ph_q))),
        ],
        dtype=np.int64,
    )
    map_q.setflags(write=False)
    map_minus.setflags(write=False)
    inventory_q = TBGZeroFieldCompanionTransitionInventory(
        source_fingerprint=source_fingerprint,
        q=label,
        pairs=pairs_q,
        ph_count=len(ph_q),
        conjugate_indices_at_minus_q=map_q,
        inventory_sha256=_inventory_digest(pairs_q),
    )
    inventory_minus = TBGZeroFieldCompanionTransitionInventory(
        source_fingerprint=source_fingerprint,
        q=minus,
        pairs=pairs_minus,
        ph_count=len(ph_minus),
        conjugate_indices_at_minus_q=map_minus,
        inventory_sha256=_inventory_digest(pairs_minus),
    )
    for index, mapped in enumerate(map_q):
        if map_minus[int(mapped)] != index:
            raise ValueError("q/-q transition conjugation maps do not close")
    return TBGZeroFieldCompanionTransitionInventoryPair(
        q=inventory_q,
        minus_q=inventory_minus,
    )


def _normalized_direct_multiplier(sector: SpinSector) -> int:
    if sector == "triplet":
        return 0
    if sector == "singlet":
        return 2
    raise ValueError("spin sector must be 'triplet' or 'singlet'")


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionStaticKernelResiduals:
    K_hermiticity_max_abs_ev: float
    L_pseudo_hermiticity_max_abs_ev: float
    eq16_carry_max_abs: float


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionStaticKernel:
    source: TBGZeroFieldCompanionTDHFSource
    source_fingerprint: str
    inventory: TBGZeroFieldCompanionTransitionInventory
    spin_sector: Literal["triplet", "singlet", "single_spin_internal"]
    direct_multiplier: int
    bare_ev: np.ndarray
    hartree_ev: np.ndarray
    exchange_ev: np.ndarray
    K_ev: np.ndarray
    core_metric: np.ndarray
    L_ev: np.ndarray
    residuals: TBGZeroFieldCompanionStaticKernelResiduals
    array_hashes: tuple[tuple[str, str], ...]

    def _validate_live_state(self) -> None:
        if self.source.fingerprint != self.source_fingerprint:
            raise ValueError("static kernel source binding drifted")
        size = len(self.inventory.pairs)
        arrays = {
            "bare_ev": _validate_live_array(
                self.bare_ev,
                name="kernel.bare_ev",
                shape=(size, size),
                dtype=np.complex128,
            ),
            "hartree_ev": _validate_live_array(
                self.hartree_ev,
                name="kernel.hartree_ev",
                shape=(size, size),
                dtype=np.complex128,
            ),
            "exchange_ev": _validate_live_array(
                self.exchange_ev,
                name="kernel.exchange_ev",
                shape=(size, size),
                dtype=np.complex128,
            ),
            "K_ev": _validate_live_array(
                self.K_ev,
                name="kernel.K_ev",
                shape=(size, size),
                dtype=np.complex128,
            ),
            "core_metric": _validate_live_array(
                self.core_metric,
                name="kernel.core_metric",
                shape=(size,),
                dtype=np.float64,
            ),
            "L_ev": _validate_live_array(
                self.L_ev,
                name="kernel.L_ev",
                shape=(size, size),
                dtype=np.complex128,
            ),
        }
        actual_hashes = tuple((name, _array_sha256(array)) for name, array in arrays.items())
        if actual_hashes != self.array_hashes:
            raise ValueError("static kernel hashes no longer match live arrays")
        if not np.array_equal(
            arrays["K_ev"],
            arrays["bare_ev"] + arrays["hartree_ev"] + arrays["exchange_ev"],
        ):
            raise ValueError("K no longer equals bare + Hartree + exchange")
        if not np.array_equal(
            arrays["L_ev"],
            arrays["core_metric"][:, None] * arrays["K_ev"],
        ):
            raise ValueError("L no longer equals J_core K")

    @property
    def fingerprint(self) -> str:
        self._validate_live_state()
        return _json_sha256(
            {
                "array_hashes": dict(self.array_hashes),
                "direct_multiplier": self.direct_multiplier,
                "inventory": self.inventory.fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "spin_sector": self.spin_sector,
            }
        )


def _build_static_kernel(
    source: TBGZeroFieldCompanionTDHFSource,
    q: TBGZeroFieldCompanionSignedQLabel | Sequence[int],
    *,
    direct_multiplier: int,
    spin_sector: Literal["triplet", "singlet", "single_spin_internal"],
) -> TBGZeroFieldCompanionStaticKernel:
    multiplier = _strict_int(direct_multiplier, name="direct_multiplier")
    if multiplier not in (0, 1, 2):
        raise ValueError("direct_multiplier must be exactly 0, 1, or 2")
    if spin_sector != "single_spin_internal" and multiplier not in (0, 2):
        raise ValueError("public normalized spin sectors support only direct factor 0 or 2")
    inventory_pair = build_tbg_zero_field_companion_transition_inventories(source, q)
    inventory = inventory_pair.q
    label = inventory.q
    form_receipt = build_tbg_zero_field_companion_hf_form_factors(source, label)
    lambda_q = form_receipt.values
    Wq = _interaction_weights(source, label)
    pairs = inventory.pairs
    size = len(pairs)
    hartree = np.empty((size, size), dtype=np.complex128)
    exchange = np.empty((size, size), dtype=np.complex128)
    bare = np.zeros((size, size), dtype=np.complex128)
    energies = source.canonical_eigenvalues_ev

    lambda_cache: dict[tuple[int, int], np.ndarray] = {label.raw: lambda_q}
    weight_cache: dict[tuple[int, int], np.ndarray] = {label.raw: Wq}

    def lambda_for(raw: tuple[int, int]) -> np.ndarray:
        cached = lambda_cache.get(raw)
        if cached is None:
            delta_label = _signed_q_label(
                source.fingerprint,
                label.N1,
                label.N2,
                raw,
            )
            cached = _compute_hf_form_factor_values(source, delta_label)
            lambda_cache[raw] = cached
            weight_cache[raw] = _interaction_weights(source, delta_label)
        return cached

    for i, out in enumerate(pairs):
        ok1, ok2 = out.k_source
        ot1, ot2 = out.k_target
        bare[i, i] = abs(
            energies[ot1, ot2, out.mu_target]
            - energies[ok1, ok2, out.nu_source]
        )
        lambda_out = lambda_q[ok1, ok2]
        for j, incoming in enumerate(pairs):
            ik1, ik2 = incoming.k_source
            lambda_in = lambda_q[ik1, ik2]
            hartree[i, j] = multiplier * np.sum(
                Wq
                * np.conj(lambda_out[:, out.nu_source, out.mu_target])
                * lambda_in[:, incoming.nu_source, incoming.mu_target]
            )
            raw_delta = (ik1 - ok1, ik2 - ok2)
            lambda_delta = lambda_for(raw_delta)
            Wdelta = weight_cache[raw_delta]
            exchange[i, j] = -np.sum(
                Wdelta
                * lambda_delta[ot1, ot2, :, out.mu_target, incoming.mu_target]
                * np.conj(
                    lambda_delta[
                        ok1,
                        ok2,
                        :,
                        out.nu_source,
                        incoming.nu_source,
                    ]
                )
            )

    K = bare + hartree + exchange
    core_metric = inventory.core_metric
    L = core_metric[:, None] * K
    K_residual = _max_hermiticity_residual(K)
    Jmatrix = np.diag(core_metric.astype(np.complex128))
    pseudo_residual = _max_abs(L.conj().T @ Jmatrix - Jmatrix @ L)
    residuals = TBGZeroFieldCompanionStaticKernelResiduals(
        K_hermiticity_max_abs_ev=K_residual,
        L_pseudo_hermiticity_max_abs_ev=pseudo_residual,
        eq16_carry_max_abs=form_receipt.eq16_carry_max_abs,
    )
    if K_residual > TBG_ZERO_FIELD_COMPANION_TDHF_STATIC_STRUCTURE_ATOL_EV:
        raise ValueError(f"Eq. (90) static Hessian K is not Hermitian: {K_residual:.6e} eV")
    if pseudo_residual > TBG_ZERO_FIELD_COMPANION_TDHF_STATIC_STRUCTURE_ATOL_EV:
        raise ValueError(
            f"Eq. (90) L is not J-pseudo-Hermitian: {pseudo_residual:.6e} eV"
        )
    arrays = {
        "bare_ev": bare,
        "hartree_ev": hartree,
        "exchange_ev": exchange,
        "K_ev": K,
        "core_metric": core_metric,
        "L_ev": L,
    }
    readonly: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        resolved = np.array(array, copy=True, order="C")
        resolved.setflags(write=False)
        readonly[name] = resolved
    hashes = tuple((name, _array_sha256(array)) for name, array in readonly.items())
    return TBGZeroFieldCompanionStaticKernel(
        source=source,
        source_fingerprint=source.fingerprint,
        inventory=inventory,
        spin_sector=spin_sector,
        direct_multiplier=multiplier,
        bare_ev=readonly["bare_ev"],
        hartree_ev=readonly["hartree_ev"],
        exchange_ev=readonly["exchange_ev"],
        K_ev=readonly["K_ev"],
        core_metric=readonly["core_metric"],
        L_ev=readonly["L_ev"],
        residuals=residuals,
        array_hashes=hashes,
    )


def build_tbg_zero_field_companion_static_kernel(
    source: TBGZeroFieldCompanionTDHFSource,
    q: TBGZeroFieldCompanionSignedQLabel | Sequence[int],
    *,
    spin_sector: SpinSector,
) -> TBGZeroFieldCompanionStaticKernel:
    """Build the public normalized triplet (0) or singlet (2) Eq. (90) kernel."""

    multiplier = _normalized_direct_multiplier(spin_sector)
    return _build_static_kernel(
        source,
        q,
        direct_multiplier=multiplier,
        spin_sector=spin_sector,
    )


def _companion_transition_to_core_pair(
    source: TBGZeroFieldCompanionTDHFSource,
    transition: TBGZeroFieldCompanionTransition,
) -> ParticleHolePair:
    params = source.prepared.params
    internal_dimension = 2 * params.active_band_count
    source_k = transition.k_source[0] * params.N2 + transition.k_source[1]
    target_k = transition.k_target[0] * params.N2 + transition.k_target[1]
    return ParticleHolePair(
        particle=target_k * internal_dimension + transition.mu_target,
        hole=source_k * internal_dimension + transition.nu_source,
        particle_momentum=transition.k_target,
        hole_momentum=transition.k_source,
        particle_flavor=(transition.mu_target, transition.role),
        hole_flavor=(transition.nu_source, transition.role),
    )


def build_tbg_zero_field_companion_typed_sector_from_kernels(
    plus: TBGZeroFieldCompanionStaticKernel,
    minus: TBGZeroFieldCompanionStaticKernel,
) -> TDHFTypedSector:
    """Convert independently assembled companion kernels to the core API."""

    plus._validate_live_state()
    minus._validate_live_state()
    if plus.source_fingerprint != minus.source_fingerprint:
        raise ValueError("signed companion kernels have different sources")
    if plus.spin_sector != minus.spin_sector:
        raise ValueError("signed companion kernels have different spin sectors")
    expected_minus = _minus_q_label(plus.inventory.q)
    if minus.inventory.q.raw != expected_minus.raw:
        raise ValueError("minus kernel is not the independently assembled -q partner")
    if _inventory_digest(plus.inventory.pairs) != plus.inventory.inventory_sha256:
        raise ValueError("plus companion transition inventory digest mismatch")
    if _inventory_digest(minus.inventory.pairs) != minus.inventory.inventory_sha256:
        raise ValueError("minus companion transition inventory digest mismatch")
    n_plus = plus.inventory.ph_count
    n_minus = minus.inventory.ph_count
    if len(plus.inventory.pairs) != n_plus + n_minus:
        raise ValueError("plus companion inventory does not have ph/hp signed blocks")
    if len(minus.inventory.pairs) != n_minus + n_plus:
        raise ValueError("minus companion inventory does not have ph/hp signed blocks")
    plus_pairs = tuple(
        _companion_transition_to_core_pair(plus.source, transition)
        for transition in plus.inventory.pairs[:n_plus]
    )
    minus_pairs = tuple(
        _companion_transition_to_core_pair(minus.source, transition)
        for transition in minus.inventory.pairs[:n_minus]
    )
    blocks = TDHFSignedQBlocks(
        plus_pairs=plus_pairs,
        minus_pairs=minus_pairs,
        A_plus=np.asarray(plus.K_ev[:n_plus, :n_plus]),
        B_plus_minus=np.asarray(plus.K_ev[:n_plus, n_plus:]),
        A_minus=np.asarray(minus.K_ev[:n_minus, :n_minus]),
        B_minus_plus=np.asarray(minus.K_ev[:n_minus, n_minus:]),
    )
    map_plus = np.asarray(
        plus.inventory.conjugate_indices_at_minus_q, dtype=np.int64
    )
    map_minus = np.asarray(
        minus.inventory.conjugate_indices_at_minus_q, dtype=np.int64
    )
    total = n_plus + n_minus
    if map_plus.shape != (total,) or map_minus.shape != (total,):
        raise ValueError("companion signed transition maps have wrong shape")
    if (
        np.any(map_plus < 0)
        or np.any(map_plus >= total)
        or np.any(map_minus < 0)
        or np.any(map_minus >= total)
    ):
        raise ValueError("companion signed transition map is out of range")
    for index, mapped in enumerate(map_plus):
        mapped_index = int(mapped)
        if int(map_minus[mapped_index]) != index:
            raise ValueError("companion signed transition maps do not close")
        transition = plus.inventory.pairs[index]
        conjugate = minus.inventory.pairs[mapped_index]
        if (
            conjugate.k_source != transition.k_target
            or conjugate.k_target != transition.k_source
            or conjugate.mu_target != transition.nu_source
            or conjugate.nu_source != transition.mu_target
            or conjugate.core_metric_sign != -transition.core_metric_sign
        ):
            raise ValueError("companion signed transition map identity mismatch")
    plus_to_minus = np.zeros((total, total), dtype=np.complex128)
    minus_to_plus = np.zeros((total, total), dtype=np.complex128)
    plus_to_minus[map_plus, np.arange(total)] = 1.0
    minus_to_plus[map_minus, np.arange(total)] = 1.0
    closure = float(
        max(
            np.max(
                np.abs(minus_to_plus @ np.conj(plus_to_minus) - np.eye(total)),
                initial=0.0,
            ),
            np.max(
                np.abs(plus_to_minus @ np.conj(minus_to_plus) - np.eye(total)),
                initial=0.0,
            ),
        )
    )
    sewing = TDHFNambuSewing(
        plus_to_minus=plus_to_minus,
        minus_to_plus=minus_to_plus,
        source_fingerprint=plus.source_fingerprint,
        plus_pairs_fingerprint=fingerprint_tdhf_pairs(plus_pairs),
        minus_pairs_fingerprint=fingerprint_tdhf_pairs(minus_pairs),
        construction=(
            "tbg_companion_explicit_transition_conjugation_map_v1;"
            f"plus_inventory={plus.inventory.fingerprint};"
            f"minus_inventory={minus.inventory.fingerprint};"
            f"plus_map={_array_sha256(map_plus)};minus_map={_array_sha256(map_minus)}"
        ),
        closure_residual=closure,
    )
    q_kind = classify_tdhf_signed_q(
        plus_raw=plus.inventory.q.raw,
        minus_raw=minus.inventory.q.raw,
        plus_canonical=plus.inventory.q.canonical_delta,
        minus_canonical=minus.inventory.q.canonical_delta,
        provenance=(
            "companion_raw_q_floor_carry_and_transition_inventory_v1;"
            f"plus={plus.inventory.fingerprint};minus={minus.inventory.fingerprint}"
        ),
    )
    interaction_fingerprint = _json_sha256(
        {
            "prepared": plus.source.prepared.fingerprint,
            "spin_sector": plus.spin_sector,
            "direct_multiplier": plus.direct_multiplier,
        }
    )
    response_scope = (
        "tbg_companion_kwan_eq90_scalar_hessian_diagnostic_only_v1"
    )
    if isinstance(q_kind, TDHFGenericSignedQ):
        sector: TDHFTypedSector = TDHFGenericSignedQSector(
            q=q_kind,
            blocks=blocks,
            sewing=sewing,
            source_fingerprint=plus.source_fingerprint,
            interaction_fingerprint=interaction_fingerprint,
            response_scope=response_scope,
            static_hessian_authority="scalar_hessian",
        )
        core = build_tdhf_signed_q_matrices(
            blocks, sewing, structure_tolerance=1.0e-10
        )
        if np.max(np.abs(core.H_plus - plus.K_ev), initial=0.0) > 1.0e-12:
            raise ValueError("core H(q) does not reproduce companion K(q)")
        if np.max(np.abs(core.H_minus - minus.K_ev), initial=0.0) > 1.0e-12:
            raise ValueError("core H(-q) does not reproduce companion K(-q)")
        return sector
    is_raw_boundary_alias = q_kind.plus_raw != q_kind.minus_raw
    if is_raw_boundary_alias and not np.array_equal(plus.K_ev, minus.K_ev):
        raise ValueError(
            "companion exact-M raw kernels differ; canonical scalar sewing is not certified"
        )
    canonical_sewing_provenance = (
        "independent_raw_exact_boundary_kernels_byte_identical_plus_branch_v1"
        if is_raw_boundary_alias
        else "literal_q0_companion_inventory_v1"
    )
    sector = TDHFSelfConjugateQSector(
        q=q_kind,
        canonical_pairs=plus_pairs,
        A=np.asarray(plus.K_ev[:n_plus, :n_plus]),
        B=np.asarray(plus.K_ev[:n_plus, n_plus:]),
        source_fingerprint=plus.source_fingerprint,
        interaction_fingerprint=interaction_fingerprint,
        response_scope=response_scope,
        static_hessian_authority="scalar_hessian",
        canonical_sewing_provenance=canonical_sewing_provenance,
        raw_signed_diagnostic_blocks=blocks if is_raw_boundary_alias else None,
        raw_signed_diagnostic_sewing=sewing if is_raw_boundary_alias else None,
    )
    core_self = build_tdhf_self_conjugate_matrices(
        sector, structure_tolerance=1.0e-10
    )
    if np.max(np.abs(core_self.H - plus.K_ev), initial=0.0) > 1.0e-12:
        raise ValueError("core self-conjugate H does not reproduce q0 companion K")
    return sector


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionTDHFTypedProvider:
    """Diagnostic-only companion adapter for ``mean_field.api.run_tdhf``."""

    source: TBGZeroFieldCompanionTDHFSource

    def build_tdhf_sector(self, config: object, **kwargs: object) -> TDHFTypedSector:
        if kwargs:
            raise TypeError(f"unexpected companion TDHF adapter kwargs: {sorted(kwargs)}")
        q_sector = getattr(config, "q_sector", None)
        channel = getattr(config, "channel", None)
        if not isinstance(q_sector, tuple) or len(q_sector) != 2:
            raise TypeError("companion typed TDHF requires an integer tuple q_sector")
        if channel not in ("triplet", "singlet"):
            raise ValueError("companion typed TDHF channel must be triplet or singlet")
        label = build_tbg_zero_field_companion_signed_q_label(self.source, q_sector)
        minus = _minus_q_label(label)
        plus_kernel = build_tbg_zero_field_companion_static_kernel(
            self.source, label, spin_sector=channel
        )
        minus_kernel = build_tbg_zero_field_companion_static_kernel(
            self.source, minus, spin_sector=channel
        )
        return build_tbg_zero_field_companion_typed_sector_from_kernels(
            plus_kernel, minus_kernel
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionDirectMatrixAction:
    """Independent loop evaluation of the three Eq. (90) operator terms."""

    source_fingerprint: str
    inventory: TBGZeroFieldCompanionTransitionInventory
    spin_sector: SpinSector
    transition_vector: np.ndarray
    bare_action_ev: np.ndarray
    hartree_action_ev: np.ndarray
    exchange_action_ev: np.ndarray
    K_action_ev: np.ndarray

def evaluate_tbg_zero_field_companion_static_matrix_action(
    source: TBGZeroFieldCompanionTDHFSource,
    q: TBGZeroFieldCompanionSignedQLabel | Sequence[int],
    transition_vector: Sequence[complex] | np.ndarray,
    *,
    spin_sector: SpinSector,
) -> TBGZeroFieldCompanionDirectMatrixAction:
    """Apply Eq. (90) directly with paper-index loops, never via a dense K.

    In particular, the first exchange vertex is evaluated on the output
    ``k+q`` fiber, while its second vertex is evaluated on output ``k``.
    This evaluator intentionally does not call the dense-kernel builder.
    """

    multiplier = _normalized_direct_multiplier(spin_sector)
    inventory = build_tbg_zero_field_companion_transition_inventories(source, q).q
    label = inventory.q
    form_receipt = build_tbg_zero_field_companion_hf_form_factors(source, label)
    lambda_q = form_receipt.values
    Wq = _interaction_weights(source, label)
    pairs = inventory.pairs
    vector = np.array(transition_vector, dtype=np.complex128, copy=True, order="C")
    if vector.shape != (len(pairs),):
        raise ValueError(
            f"transition_vector must have shape {(len(pairs),)}, got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("transition_vector must contain only finite values")

    bare_action = np.zeros(vector.shape, dtype=np.complex128)
    hartree_action = np.zeros(vector.shape, dtype=np.complex128)
    exchange_action = np.zeros(vector.shape, dtype=np.complex128)
    energies = source.canonical_eigenvalues_ev
    lambda_cache: dict[tuple[int, int], np.ndarray] = {label.raw: lambda_q}
    weight_cache: dict[tuple[int, int], np.ndarray] = {label.raw: Wq}

    def lambda_for(raw: tuple[int, int]) -> np.ndarray:
        cached = lambda_cache.get(raw)
        if cached is None:
            delta_label = _signed_q_label(
                source.fingerprint,
                label.N1,
                label.N2,
                raw,
            )
            cached = _compute_hf_form_factor_values(source, delta_label)
            lambda_cache[raw] = cached
            weight_cache[raw] = _interaction_weights(source, delta_label)
        return cached

    for output_index, output in enumerate(pairs):
        ok1, ok2 = output.k_source
        ot1, ot2 = output.k_target
        bare_action[output_index] = (
            abs(
                energies[ot1, ot2, output.mu_target]
                - energies[ok1, ok2, output.nu_source]
            )
            * vector[output_index]
        )
        for input_index, incoming in enumerate(pairs):
            ik1, ik2 = incoming.k_source
            amplitude = vector[input_index]
            for iG in range(Wq.size):
                hartree_action[output_index] += (
                    multiplier
                    * Wq[iG]
                    * np.conj(
                        lambda_q[
                            ok1,
                            ok2,
                            iG,
                            output.nu_source,
                            output.mu_target,
                        ]
                    )
                    * lambda_q[
                        ik1,
                        ik2,
                        iG,
                        incoming.nu_source,
                        incoming.mu_target,
                    ]
                    * amplitude
                )
            raw_delta = (ik1 - ok1, ik2 - ok2)
            lambda_delta = lambda_for(raw_delta)
            Wdelta = weight_cache[raw_delta]
            for iG in range(Wdelta.size):
                exchange_action[output_index] -= (
                    Wdelta[iG]
                    * lambda_delta[
                        ot1,
                        ot2,
                        iG,
                        output.mu_target,
                        incoming.mu_target,
                    ]
                    * np.conj(
                        lambda_delta[
                            ok1,
                            ok2,
                            iG,
                            output.nu_source,
                            incoming.nu_source,
                        ]
                    )
                    * amplitude
                )

    K_action = bare_action + hartree_action + exchange_action
    arrays = (vector, bare_action, hartree_action, exchange_action, K_action)
    for array in arrays:
        array.setflags(write=False)
    return TBGZeroFieldCompanionDirectMatrixAction(
        source_fingerprint=source.fingerprint,
        inventory=inventory,
        spin_sector=spin_sector,
        transition_vector=vector,
        bare_action_ev=bare_action,
        hartree_action_ev=hartree_action,
        exchange_action_ev=exchange_action,
        K_action_ev=K_action,
    )

@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionDenseSpectrum:
    """System binding around the generic core TDHF eigensolver result."""

    kernel: TBGZeroFieldCompanionStaticKernel
    kernel_fingerprint: str
    raw_eigenvalues_ev: np.ndarray
    raw_J_metric_norms: np.ndarray
    raw_eigensolver_residuals_ev: np.ndarray
    selected_eigenvalues_ev: np.ndarray
    selected_energies_ev: np.ndarray
    selected_right_eigenvectors: np.ndarray
    selected_J_metric_norms: np.ndarray
    selected_eigensolver_residuals_ev: np.ndarray
    selected_raw_indices: np.ndarray
    complex_indices: np.ndarray
    negative_real_indices: np.ndarray
    static_real_indices: np.ndarray
    raw_pairing_residual_ev: float
    raw_eigensolver_residual_max_ev: float
    anomaly_classification: TDHFStabilityClassification
    diagnostic_passed: bool
    diagnostic_failure_reasons: tuple[str, ...]

    @property
    def paper_eta_omega_ev(self) -> np.ndarray:
        """Eq. (90) quantity, using ``lambda_L = -eta*omega``."""

        values = np.array(-self.selected_eigenvalues_ev, copy=True)
        values.setflags(write=False)
        return values

    @property
    def fingerprint(self) -> str:
        if self.kernel.fingerprint != self.kernel_fingerprint:
            raise ValueError("dense spectrum kernel binding drifted")
        stability = self.anomaly_classification
        return _json_sha256(
            {
                "anomaly_classification": {
                    "complex_count": stability.complex_count,
                    "complex_eigenvalues": stability.complex_eigenvalues,
                    "energy_tol": stability.energy_tol,
                    "imag_tol": stability.imag_tol,
                    "lowest_energy": stability.lowest_energy,
                    "missing_positive_metric_modes": (
                        stability.missing_positive_metric_modes
                    ),
                    "n_pairs": stability.n_pairs,
                    "negative_selected_energy": stability.negative_selected_energy,
                    "nonfinite_eigenvalues": stability.nonfinite_eigenvalues,
                    "nonfinite_selected_energies": (
                        stability.nonfinite_selected_energies
                    ),
                    "reason": stability.reason,
                    "selected_count": stability.selected_count,
                    "stable": stability.stable,
                    "structure_ok": stability.structure_ok,
                    "zero_mode_branches": stability.zero_mode_branches,
                },
                "complex_indices": _array_sha256(self.complex_indices),
                "diagnostic_failure_reasons": list(self.diagnostic_failure_reasons),
                "diagnostic_passed": self.diagnostic_passed,
                "kernel_fingerprint": self.kernel_fingerprint,
                "negative_real_indices": _array_sha256(self.negative_real_indices),
                "raw_J_metric_norms": _array_sha256(self.raw_J_metric_norms),
                "raw_eigenvalues_ev": _array_sha256(self.raw_eigenvalues_ev),
                "raw_eigensolver_residual_max_ev": (
                    self.raw_eigensolver_residual_max_ev
                ),
                "raw_eigensolver_residuals_ev": _array_sha256(
                    self.raw_eigensolver_residuals_ev
                ),
                "raw_pairing_residual_ev": self.raw_pairing_residual_ev,
                "selected_J_metric_norms": _array_sha256(
                    self.selected_J_metric_norms
                ),
                "selected_eigenvalues_ev": _array_sha256(
                    self.selected_eigenvalues_ev
                ),
                "selected_energies_ev": _array_sha256(self.selected_energies_ev),
                "selected_eigensolver_residuals_ev": _array_sha256(
                    self.selected_eigensolver_residuals_ev
                ),
                "selected_raw_indices": _array_sha256(self.selected_raw_indices),
                "selected_right_eigenvectors": _array_sha256(
                    self.selected_right_eigenvectors
                ),
                "static_real_indices": _array_sha256(self.static_real_indices),
            }
        )


def solve_tbg_zero_field_companion_dense_spectrum(
    kernel: TBGZeroFieldCompanionStaticKernel,
) -> TBGZeroFieldCompanionDenseSpectrum:
    """Solve ``L=J_core K`` through the reusable core TDHF eigensolver.

    This wrapper only binds the companion kernel and maps the core eigenvalue to
    the paper's ``eta*omega`` sign.  Eigenproblem solution and degenerate metric
    normalization are not reimplemented here.  Numerical diagnostic validity is
    independent of physical stability classification.  Same-matrix raw spectral
    pairing is authoritative only for the canonical self-conjugate ``q=(0, 0)``;
    generic signed q sectors require the separate q/-q comparison.
    """

    if not isinstance(kernel, TBGZeroFieldCompanionStaticKernel):
        raise TypeError("kernel must be TBGZeroFieldCompanionStaticKernel")
    if kernel.direct_multiplier not in (0, 2) or kernel.spin_sector not in (
        "triplet",
        "singlet",
    ):
        raise ValueError("public spectra support only normalized direct factors 0 or 2")
    n_pairs = kernel.inventory.ph_count
    if len(kernel.inventory.pairs) != 2 * n_pairs:
        raise ValueError("companion ph/hp inventory does not match core TDHF ordering")
    kernel_fingerprint = kernel.fingerprint
    core_spectrum = solve_tdhf_liouvillian(
        kernel.L_ev,
        n_pairs=n_pairs,
        energy_tol=TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV,
        imag_tol=TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV,
        norm_tol=TBG_ZERO_FIELD_COMPANION_TDHF_METRIC_CLASSIFICATION_ATOL,
        degeneracy_tol=TBG_ZERO_FIELD_COMPANION_TDHF_DEGENERACY_ATOL_EV,
    )
    raw_eigenvalues = np.array(core_spectrum.raw_eigenvalues, copy=True)
    raw_metric_norms = np.array(core_spectrum.raw_eta_norms, copy=True)
    raw_residuals = np.array(core_spectrum.raw_residuals, copy=True)
    selected_eigenvalues = np.array(core_spectrum.eigenvalues, copy=True)
    selected_energies = np.array(core_spectrum.energies, copy=True)
    selected_vectors = np.array(core_spectrum.amplitudes, copy=True)
    selected_metric_norms = np.array(core_spectrum.eta_norms, copy=True)
    selected_residuals = np.array(core_spectrum.residuals, copy=True)
    selected_indices = np.array(core_spectrum.selected_indices, dtype=np.int64, copy=True)

    real_mask = np.abs(np.imag(raw_eigenvalues)) <= (
        TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV
    )
    complex_indices = np.flatnonzero(~real_mask).astype(np.int64)
    negative_indices = np.flatnonzero(
        real_mask
        & (
            np.real(raw_eigenvalues)
            < -TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV
        )
    ).astype(np.int64)
    static_indices = np.flatnonzero(
        real_mask
        & (
            np.abs(np.real(raw_eigenvalues))
            <= TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV
        )
    ).astype(np.int64)
    structure_ok = bool(
        kernel.residuals.K_hermiticity_max_abs_ev
        <= TBG_ZERO_FIELD_COMPANION_TDHF_STATIC_STRUCTURE_ATOL_EV
        and kernel.residuals.L_pseudo_hermiticity_max_abs_ev
        <= TBG_ZERO_FIELD_COMPANION_TDHF_STATIC_STRUCTURE_ATOL_EV
    )
    anomaly_classification = classify_tdhf_stability(
        raw_eigenvalues,
        selected_energies,
        n_pairs=n_pairs,
        structure_ok=structure_ok,
        imag_tol=TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV,
        energy_tol=TBG_ZERO_FIELD_COMPANION_TDHF_EIGEN_CLASSIFICATION_ATOL_EV,
    )
    raw_residual_max = _max_abs(raw_residuals)
    selected_residual_max = _max_abs(selected_residuals)
    raw_pairing_residual = float(core_spectrum.pairing_residual)
    # These are solver-validity gates only; an unstable spectrum may still be a
    # numerically valid diagnostic represented by ``anomaly_classification``.
    failure_reasons: list[str] = []
    if not np.all(np.isfinite(raw_residuals)):
        failure_reasons.append("raw_eigensolver_residuals_are_nonfinite")
    elif raw_residual_max > (
        TBG_ZERO_FIELD_COMPANION_TDHF_RAW_EIGENSOLVER_RESIDUAL_ATOL_EV
    ):
        failure_reasons.append(
            "max_raw_eigensolver_residual_exceeds_1e-9_eV"
        )
    if not np.all(np.isfinite(selected_residuals)):
        failure_reasons.append(
            "selected_normalized_mode_eigensolver_residuals_are_nonfinite"
        )
    elif selected_residual_max > (
        TBG_ZERO_FIELD_COMPANION_TDHF_SELECTED_EIGENSOLVER_RESIDUAL_ATOL_EV
    ):
        failure_reasons.append(
            "max_selected_normalized_mode_eigensolver_residual_exceeds_1e-9_eV"
        )
    if kernel.inventory.q.raw == (0, 0) and (
        not np.isfinite(raw_pairing_residual)
        or raw_pairing_residual
        > TBG_ZERO_FIELD_COMPANION_TDHF_Q0_RAW_PAIRING_RESIDUAL_ATOL_EV
    ):
        failure_reasons.append(
            "q0_same_matrix_raw_pairing_residual_exceeds_1e-9_eV"
        )
    arrays = (
        raw_eigenvalues,
        raw_metric_norms,
        raw_residuals,
        selected_eigenvalues,
        selected_energies,
        selected_vectors,
        selected_metric_norms,
        selected_residuals,
        selected_indices,
        complex_indices,
        negative_indices,
        static_indices,
    )
    for array in arrays:
        array.setflags(write=False)
    return TBGZeroFieldCompanionDenseSpectrum(
        kernel=kernel,
        kernel_fingerprint=kernel_fingerprint,
        raw_eigenvalues_ev=raw_eigenvalues,
        raw_J_metric_norms=raw_metric_norms,
        raw_eigensolver_residuals_ev=raw_residuals,
        selected_eigenvalues_ev=selected_eigenvalues,
        selected_energies_ev=selected_energies,
        selected_right_eigenvectors=selected_vectors,
        selected_J_metric_norms=selected_metric_norms,
        selected_eigensolver_residuals_ev=selected_residuals,
        selected_raw_indices=selected_indices,
        complex_indices=complex_indices,
        negative_real_indices=negative_indices,
        static_real_indices=static_indices,
        raw_pairing_residual_ev=raw_pairing_residual,
        raw_eigensolver_residual_max_ev=raw_residual_max,
        anomaly_classification=anomaly_classification,
        diagnostic_passed=not failure_reasons,
        diagnostic_failure_reasons=tuple(failure_reasons),
    )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionSpectralPairingResidual:
    q_raw: tuple[int, int]
    minus_q_raw: tuple[int, int]
    q_to_minus_q_max_abs_ev: float
    minus_q_to_q_max_abs_ev: float
    max_abs_ev: float


def tbg_zero_field_companion_signed_spectral_pairing(
    q_spectrum: TBGZeroFieldCompanionDenseSpectrum,
    minus_q_spectrum: TBGZeroFieldCompanionDenseSpectrum,
) -> TBGZeroFieldCompanionSpectralPairingResidual:
    """Compare the complete signed spectra as omega(q)=-omega(-q)*."""

    q_raw = q_spectrum.kernel.inventory.q.raw
    minus_raw = minus_q_spectrum.kernel.inventory.q.raw
    if minus_raw != (-q_raw[0], -q_raw[1]):
        raise ValueError("spectra are not labeled by exact signed q and -q")
    if q_spectrum.kernel.source_fingerprint != minus_q_spectrum.kernel.source_fingerprint:
        raise ValueError("q/-q spectra do not share a Stage7A source")
    if q_spectrum.kernel.spin_sector != minus_q_spectrum.kernel.spin_sector:
        raise ValueError("q/-q spectra do not share a normalized spin sector")
    assignment_residual = signed_q_particle_hole_assignment_residual(
        q_spectrum.raw_eigenvalues_ev,
        minus_q_spectrum.raw_eigenvalues_ev,
    )
    return TBGZeroFieldCompanionSpectralPairingResidual(
        q_raw=q_raw,
        minus_q_raw=minus_raw,
        q_to_minus_q_max_abs_ev=assignment_residual,
        minus_q_to_q_max_abs_ev=assignment_residual,
        max_abs_ev=assignment_residual,
    )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionQ0ParityResiduals:
    H_D_A_max_abs_ev: float
    H_E_A_max_abs_ev: float
    H_D_B_max_abs_ev: float
    H_E_B_max_abs_ev: float

    @property
    def max_abs_ev(self) -> float:
        return max(
            self.H_D_A_max_abs_ev,
            self.H_E_A_max_abs_ev,
            self.H_D_B_max_abs_ev,
            self.H_E_B_max_abs_ev,
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionQ0ParityOracle:
    source_fingerprint: str
    spin_sector: Literal["triplet", "singlet", "single_spin_internal"]
    direct_multiplier: int
    columns: tuple[int, ...]
    response_A_H_D_ev: np.ndarray
    response_A_H_E_ev: np.ndarray
    response_B_H_D_ev: np.ndarray
    response_B_H_E_ev: np.ndarray
    kernel_A_H_D_ev: np.ndarray
    kernel_A_H_E_ev: np.ndarray
    kernel_B_H_D_ev: np.ndarray
    kernel_B_H_E_ev: np.ndarray
    residuals: TBGZeroFieldCompanionQ0ParityResiduals


def _spin_weights(
    sector: Literal["triplet", "singlet", "single_spin_internal"],
) -> tuple[np.ndarray, int]:
    if sector == "single_spin_internal":
        return np.asarray([1.0, 0.0], dtype=np.float64), 1
    if sector == "triplet":
        return np.asarray([1.0, -1.0], dtype=np.float64) / np.sqrt(2.0), 0
    if sector == "singlet":
        return np.asarray([1.0, 1.0], dtype=np.float64) / np.sqrt(2.0), 2
    raise ValueError("unsupported q=0 spin sector")


def _build_q0_parity_oracle(
    source: TBGZeroFieldCompanionTDHFSource,
    *,
    sector: Literal["triplet", "singlet", "single_spin_internal"],
    columns: Sequence[int] | None = None,
) -> TBGZeroFieldCompanionQ0ParityOracle:
    weights, multiplier = _spin_weights(sector)
    kernel = _build_static_kernel(
        source,
        (0, 0),
        direct_multiplier=multiplier,
        spin_sector=sector,
    )
    nph = kernel.inventory.ph_count
    resolved_columns = (
        tuple(range(nph))
        if columns is None
        else tuple(_strict_int(column, name="column") for column in columns)
    )
    if len(set(resolved_columns)) != len(resolved_columns):
        raise ValueError("q=0 parity columns must be unique")
    if any(column < 0 or column >= nph for column in resolved_columns):
        raise ValueError("q=0 parity column is outside the ph inventory")
    nrows = nph
    ncolumns = len(resolved_columns)
    response_A_D = np.empty((nrows, ncolumns), dtype=np.complex128)
    response_A_E = np.empty((nrows, ncolumns), dtype=np.complex128)
    response_B_D = np.empty((nrows, ncolumns), dtype=np.complex128)
    response_B_E = np.empty((nrows, ncolumns), dtype=np.complex128)
    C = source.canonical_eigenvectors
    hshape, _eshape, _vshape = _source_shapes(source.prepared)
    zero = np.zeros(hshape, dtype=np.complex128)

    def project_response(deltaH: np.ndarray, row: TBGZeroFieldCompanionTransition) -> complex:
        k1, k2 = row.k_source
        value = 0.0j
        for spin in range(2):
            transformed = C[k1, k2].conj().T @ deltaH[k1, k2, spin] @ C[k1, k2]
            value += weights[spin] * transformed[row.mu_target, row.nu_source]
        return complex(value)

    for output_column, input_index in enumerate(resolved_columns):
        pair = kernel.inventory.pairs[input_index]
        k1, k2 = pair.k_source
        T = np.zeros((source.dimension, source.dimension), dtype=np.complex128)
        # Stored HF tangent required by the Stage6 convention: T = E_hp.
        T[pair.nu_source, pair.mu_target] = 1.0
        stored_T = C[k1, k2].conj() @ T @ C[k1, k2].T
        tangent_T = np.zeros(hshape, dtype=np.complex128)
        for spin in range(2):
            tangent_T[k1, k2, spin] = weights[spin] * stored_T
        tangent_X = tangent_T + np.swapaxes(tangent_T.conj(), -1, -2)
        tangent_Y = 1.0j * (
            tangent_T - np.swapaxes(tangent_T.conj(), -1, -2)
        )
        # Both source entry points are exercised deliberately.  evaluate performs
        # the same canonical calc_fock_matrix action and binds it to prepared;
        # the Y quadrature calls calc_fock_matrix directly.
        action_X = source.prepared.evaluate(tangent_X, zero).action
        action_Y = calc_fock_matrix(
            source.prepared.params,
            tangent_Y,
            source.prepared.form,
            source.prepared.M_ev,
            source.prepared.tVE_ev,
        )
        F_T_D = (action_X.H_D_ev - 1.0j * action_Y.H_D_ev) / 2.0
        F_T_E = (action_X.H_E_ev - 1.0j * action_Y.H_E_ev) / 2.0
        F_Tdag_D = (action_X.H_D_ev + 1.0j * action_Y.H_D_ev) / 2.0
        F_Tdag_E = (action_X.H_E_ev + 1.0j * action_Y.H_E_ev) / 2.0
        for row_index, row in enumerate(kernel.inventory.pairs[:nph]):
            response_A_D[row_index, output_column] = project_response(F_T_D, row)
            response_A_E[row_index, output_column] = project_response(F_T_E, row)
            response_B_D[row_index, output_column] = project_response(F_Tdag_D, row)
            response_B_E[row_index, output_column] = project_response(F_Tdag_E, row)

    hp_columns = np.asarray([nph + column for column in resolved_columns], dtype=np.int64)
    ph_rows = np.arange(nph, dtype=np.int64)
    ph_columns = np.asarray(resolved_columns, dtype=np.int64)
    kernel_A_D = kernel.hartree_ev[np.ix_(ph_rows, ph_columns)]
    kernel_A_E = kernel.exchange_ev[np.ix_(ph_rows, ph_columns)]
    kernel_B_D = kernel.hartree_ev[np.ix_(ph_rows, hp_columns)]
    kernel_B_E = kernel.exchange_ev[np.ix_(ph_rows, hp_columns)]
    residuals = TBGZeroFieldCompanionQ0ParityResiduals(
        H_D_A_max_abs_ev=_max_abs(response_A_D - kernel_A_D),
        H_E_A_max_abs_ev=_max_abs(response_A_E - kernel_A_E),
        H_D_B_max_abs_ev=_max_abs(response_B_D - kernel_B_D),
        H_E_B_max_abs_ev=_max_abs(response_B_E - kernel_B_E),
    )
    arrays = (
        response_A_D,
        response_A_E,
        response_B_D,
        response_B_E,
        kernel_A_D,
        kernel_A_E,
        kernel_B_D,
        kernel_B_E,
    )
    for array in arrays:
        array.setflags(write=False)
    return TBGZeroFieldCompanionQ0ParityOracle(
        source_fingerprint=source.fingerprint,
        spin_sector=sector,
        direct_multiplier=multiplier,
        columns=resolved_columns,
        response_A_H_D_ev=response_A_D,
        response_A_H_E_ev=response_A_E,
        response_B_H_D_ev=response_B_D,
        response_B_H_E_ev=response_B_E,
        kernel_A_H_D_ev=kernel_A_D,
        kernel_A_H_E_ev=kernel_A_E,
        kernel_B_H_D_ev=kernel_B_D,
        kernel_B_H_E_ev=kernel_B_E,
        residuals=residuals,
    )


def _build_tbg_zero_field_companion_single_spin_q0_parity_oracle(
    source: TBGZeroFieldCompanionTDHFSource,
    *,
    columns: Sequence[int] | None = None,
) -> TBGZeroFieldCompanionQ0ParityOracle:
    """Private factor-one reference; not a public normalized spectrum sector."""

    return _build_q0_parity_oracle(
        source,
        sector="single_spin_internal",
        columns=columns,
    )


def build_tbg_zero_field_companion_q0_parity_oracle(
    source: TBGZeroFieldCompanionTDHFSource,
    *,
    spin_sector: SpinSector,
    columns: Sequence[int] | None = None,
) -> TBGZeroFieldCompanionQ0ParityOracle:
    """Audit all selected q=0 A/B columns in normalized triplet or singlet."""

    _normalized_direct_multiplier(spin_sector)
    return _build_q0_parity_oracle(
        source,
        sector=spin_sector,
        columns=columns,
    )
