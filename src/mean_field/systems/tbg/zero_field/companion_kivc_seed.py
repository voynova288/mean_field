"""Kwan Eq. (99) K-IVC seed diagnostic built from typed Stage-2 data.

This module constructs a source-array-bound ordered-phase-anchor diagnostic
projector in the companion active-subspace coordinates.  The anchored frame is
bound to the exact Stage-2 array hashes; it is active-subspace gauge-covariant
but neither a cross-eigensolver frame nor a global smooth gauge.  The module is
intentionally not exported by the zero-field TBG
package front door and is not connected to an HF runner or production path.
The construction uses Kwan et al. Eq. (99), while the companion checkout
supplies only the Stage-2 active-subspace coordinates and stored-projector
convention.  It is therefore not a source-faithful companion K-IVC
implementation.

The canonical basis is valley-major ``(K Z+, K Z-, K' Z+, K' Z-)``.  The two
valleys use the same projected microscopic-sublattice ``Z+,-`` order, not the
same physical-Chern order.  The mapped ``U_Tp`` checks follow from the
constructed frames algebraically; they are not an independent microscopic
sewing validation.  A separate stored-projector ``Tp`` residual applies the
exact boost-zero transform from ``reference/TBG-HF/measure.py``.  The
Chern-balance trace uses Eq. (98) only as Chern-label context and is not an FHS
Chern calculation or an implementation of the Eq. (98) pseudospin equality.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import subprocess
from typing import Final

import numpy as np

from .companion_geometry import (
    TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
)
from .companion_single_particle import (
    TBGZeroFieldCompanionSingleParticleArrayHashes,
    TBGZeroFieldCompanionSingleParticleResult,
)

TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCHEMA: Final[str] = (
    "mean_field.tbg.zero_field.companion_kivc_seed"
)
TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCHEMA_VERSION: Final[int] = 2
TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCOPE: Final[str] = (
    "paper_eq99_source_array_bound_ordered_phase_anchor_diagnostic_not_companion_"
    "source_parity_not_HF_not_FHS_not_production_result"
)
TBG_ZERO_FIELD_COMPANION_KIVC_FRAME_SCOPE: Final[str] = (
    "source-array-bound ordered phase anchor in exact companion parent order; "
    "active-subspace gauge-covariant; not cross-eigensolver reproducible; "
    "not a global smooth gauge"
)
TBG_ZERO_FIELD_COMPANION_KIVC_BASIS_COVARIANCE_SCOPE: Final[str] = (
    "paired active-subspace basis covariance; arbitrary U(2) is not a valid "
    "nondegenerate eigenpair gauge"
)
TBG_ZERO_FIELD_COMPANION_KIVC_CANONICAL_ORDER: Final[str] = (
    "valley_major:(K_Zplus,K_Zminus,Kprime_Zplus,Kprime_Zminus);"
    "same_projected_microscopic_sublattice_Z_order_in_both_valleys;"
    "not_same_physical_Chern_order"
)
TBG_ZERO_FIELD_COMPANION_KIVC_STORED_PROJECTOR_CONVENTION: Final[str] = (
    "P_stored[ik1,ik2,spin,a,b]=P_conventional[ik1,ik2,b,a]="
    "<c_a^dagger_c_b>;transpose_matrix_axes_only;duplicated_two_spins"
)
TBG_ZERO_FIELD_COMPANION_KIVC_TP_SCOPE: Final[str] = (
    "mapped_U_Tp_checks_algebraic_by_construction_not_independent_microscopic_sewing"
)
TBG_ZERO_FIELD_COMPANION_KIVC_COMPANION_MEASURE_TP_SCOPE: Final[str] = (
    "independent_companion_measure_boost0_stored_projector_Tp_residual"
)
TBG_ZERO_FIELD_COMPANION_KIVC_CHERN_SCOPE: Final[str] = (
    "Eq_98_Chern_context_only_canonical_Gamma_C_balance_trace;"
    "not_Eq_98_pseudospin_equality;not_FHS_Chern_validation"
)
TBG_ZERO_FIELD_COMPANION_KIVC_ARRAY_HASH_CONVENTION: Final[str] = (
    "sha256_little_endian_int64_shape_then_C_order_canonical_bytes"
)
TBG_ZERO_FIELD_COMPANION_KIVC_EXTERNAL_AUTHORITY_FILES: Final[str] = (
    "external_authority_files_not_embedded"
)

TBG_ZERO_FIELD_COMPANION_KIVC_SIGN_GAP_TOLERANCE: Final[float] = 1.0e-8
# Scan the lifted column in exact companion parent order and anchor its phase to
# the first component meeting this source-array-bound relative-magnitude floor.
# A caller may tighten the floor but may neither loosen it nor exceed the
# mathematical upper bound for a component divided by the vector norm.
TBG_ZERO_FIELD_COMPANION_KIVC_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE: Final[float] = (
    1.0e-12
)
MAX_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE: Final[float] = 1.0
MAX_VALIDATION_TOLERANCE: Final[float] = 1.0e-10
TBG_ZERO_FIELD_COMPANION_KIVC_VALIDATION_TOLERANCE: Final[float] = (
    MAX_VALIDATION_TOLERANCE
)

KWAN_EQ99_PDF_SOURCE: Final[str] = "reference/2511.21683v1.pdf"
KWAN_EQ99_PDF_SHA256: Final[str] = (
    "2354caaa3c5fddbdc7c5caaacbc9dcfa94c45dfc855d930b10372daabf6fd8a6"
)
KWAN_EQ99_ARXIV: Final[str] = "2511.21683v1"
KWAN_EQ99_REFERENCE: Final[str] = (
    "arXiv:2511.21683v1 PDF pages 38-39 (printed 38-39); "
    "Eq. (99) implementation authority; Eq. (98) Chern context only; "
    "not implementing Eq. (97) triplet/n_pm; "
    "not implementing Eq. (98) pseudospin equality"
)
TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE: Final[str] = (
    "reference/TBG-HF/projectors.py"
)
TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256: Final[str] = (
    "d7c7138ddf2107a71c24194ac70790bd27cdc05297ee9cdc997c1dc3882e5ede"
)
TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE: Final[str] = "reference/TBG-HF/measure.py"
TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256: Final[str] = (
    "d1a47420400c3381247f4bc8c2e7700935077536b7782a14e52e1d25a1fd516e"
)
TBG_ZERO_FIELD_COMPANION_MEASURE_TP_REFERENCE_LINES: Final[str] = "5-14,64-69"

_TAU_Y: Final[np.ndarray] = np.asarray(
    [[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128
)
_SIGMA_Y: Final[np.ndarray] = np.asarray(
    [[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128
)
_TAU_Z: Final[np.ndarray] = np.diag([1.0, -1.0]).astype(np.complex128)
_SIGMA_Z: Final[np.ndarray] = np.diag([1.0, -1.0]).astype(np.complex128)
_TP_CANONICAL: Final[np.ndarray] = np.kron(
    _TAU_Y, np.eye(2, dtype=np.complex128)
)
_GAMMA_CHERN_CANONICAL: Final[np.ndarray] = np.kron(_TAU_Z, _SIGMA_Z)


def _finite_real(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar (bool is not accepted)")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if nonnegative and resolved < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return resolved


def _validate_phase_anchor_min_relative_magnitude(value: object) -> float:
    name = "phase_anchor_min_relative_magnitude"
    resolved = _finite_real(value, name=name, nonnegative=True)
    hard_minimum = (
        TBG_ZERO_FIELD_COMPANION_KIVC_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE
    )
    if not hard_minimum <= resolved <= MAX_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE:
        raise ValueError(
            f"{name} must be >= "
            "TBG_ZERO_FIELD_COMPANION_KIVC_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE "
            f"({hard_minimum:.1e}) and <= "
            "MAX_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE "
            f"({MAX_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE:.1e})"
        )
    return resolved

def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path, *, label: str) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Pinned {label} source is unavailable: {path}") from exc
    return hashlib.sha256(data).hexdigest()


def _canonical_array_sha256(values: np.ndarray) -> str:
    source = np.asarray(values)
    if source.dtype.kind == "c":
        dtype = np.dtype("<c16")
    elif source.dtype.kind == "f":
        dtype = np.dtype("<f8")
    elif source.dtype.kind in "iu":
        dtype = np.dtype("<i8")
    else:
        raise TypeError(f"Unsupported array dtype for canonical hashing: {source.dtype}")
    array = np.ascontiguousarray(source, dtype=dtype)
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


def _minus_mesh_index(ik1: int, ik2: int, N1: int, N2: int) -> tuple[int, int]:
    return (-ik1) % N1, (-ik2) % N2


def kwan_eq99_kivc_q(phi: float) -> np.ndarray:
    """Return Eq. (99) in valley-major, same-``Z`` canonical order."""

    resolved_phi = _finite_real(phi, name="phi")
    tau_phi = np.asarray(
        [
            [0.0, np.exp(-1.0j * resolved_phi)],
            [np.exp(1.0j * resolved_phi), 0.0],
        ],
        dtype=np.complex128,
    )
    result = np.ascontiguousarray(np.kron(tau_phi, _SIGMA_Y))
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionKIVCSourceHashes:
    """Immutable hashes for optional, non-embedded external authorities."""

    paper_pdf: str
    single_particle: str
    projectors: str
    measure: str

    @classmethod
    def from_pinned_metadata(cls) -> "TBGZeroFieldCompanionKIVCSourceHashes":
        """Construct from tracked constants without reading external files."""

        return cls(
            paper_pdf=KWAN_EQ99_PDF_SHA256,
            single_particle=TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
            projectors=TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256,
            measure=TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256,
        )

    @staticmethod
    def _pinned_hashes() -> dict[str, str]:
        return {
            "measure": TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256,
            "paper_pdf": KWAN_EQ99_PDF_SHA256,
            "projectors": TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256,
            "single_particle": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
        }

    @staticmethod
    def _pinned_locator() -> dict[str, str]:
        return {
            "companion_commit": TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
            "companion_repository": TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
            "measure": TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE,
            "paper_pdf": KWAN_EQ99_PDF_SOURCE,
            "projectors": TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE,
            "single_particle": (
                f"{TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY}/"
                f"{TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE}"
            ),
        }

    def __post_init__(self) -> None:
        self.validate_pinned_metadata()

    def validate_pinned_metadata(self) -> None:
        for name, digest in self._pinned_hashes().items():
            actual = getattr(self, name)
            if actual != digest:
                raise ValueError(
                    f"Pinned {name} source drift: expected {digest}, got {actual}"
                )

    @property
    def fingerprint(self) -> str:
        self.validate_pinned_metadata()
        return _json_sha256(
            {
                "external_authority_files": (
                    TBG_ZERO_FIELD_COMPANION_KIVC_EXTERNAL_AUTHORITY_FILES
                ),
                "hashes": self._pinned_hashes(),
                "locator": self._pinned_locator(),
            }
        )

    def to_metadata(self) -> dict[str, object]:
        self.validate_pinned_metadata()
        return {
            "external_authority_files": (
                TBG_ZERO_FIELD_COMPANION_KIVC_EXTERNAL_AUTHORITY_FILES
            ),
            "fingerprint": self.fingerprint,
            "hashes": self._pinned_hashes(),
            "locator": self._pinned_locator(),
        }


def validate_tbg_zero_field_companion_kivc_external_authorities(
    pdf_path: str | Path,
    companion_root: str | Path,
) -> TBGZeroFieldCompanionKIVCSourceHashes:
    """Explicitly verify supplied non-embedded paper and companion authorities.

    This opt-in validator never treats missing files, an unavailable nested Git
    checkout, or source drift as optional.  Normal seed construction does not
    call it and remains self-contained in a tracked checkout.
    """

    resolved_pdf_path = Path(pdf_path)
    resolved_companion_root = Path(companion_root)
    if not resolved_companion_root.is_dir():
        raise ValueError(
            "Pinned companion authority root is unavailable: "
            f"{resolved_companion_root}"
        )
    if not (resolved_companion_root / ".git").exists():
        raise ValueError(
            "Pinned companion nested Git metadata is unavailable at "
            f"{resolved_companion_root}"
        )

    expected = TBGZeroFieldCompanionKIVCSourceHashes.from_pinned_metadata()
    supplied = {
        "paper_pdf": _file_sha256(resolved_pdf_path, label="Kwan PDF"),
        "single_particle": _file_sha256(
            resolved_companion_root / TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
            label="companion singleParticle.py",
        ),
        "projectors": _file_sha256(
            resolved_companion_root
            / Path(TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE).name,
            label="companion projectors.py",
        ),
        "measure": _file_sha256(
            resolved_companion_root
            / Path(TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE).name,
            label="companion measure.py",
        ),
    }
    for name, actual in supplied.items():
        pinned = getattr(expected, name)
        if actual != pinned:
            raise ValueError(
                f"Pinned {name} external authority drift: "
                f"expected {pinned}, got {actual}"
            )

    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(resolved_companion_root),
                "rev-parse",
                "--verify",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "Pinned companion nested commit is unavailable at "
            f"{resolved_companion_root}"
        ) from exc
    live_commit = completed.stdout.strip()
    if live_commit != TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT:
        raise ValueError(
            "Pinned companion nested commit drift: expected "
            f"{TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT}, got {live_commit}"
        )
    return expected


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionKIVCSeedProvenance:
    phi: float
    stage2_fingerprint: str
    stage2_array_hashes_fingerprint: str
    source_hashes: TBGZeroFieldCompanionKIVCSourceHashes
    paper_source: str = KWAN_EQ99_PDF_SOURCE
    paper_reference: str = KWAN_EQ99_REFERENCE
    paper_arxiv: str = KWAN_EQ99_ARXIV
    companion_repository: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY
    companion_commit: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT
    companion_single_particle_source: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE
    companion_projectors_source: str = TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE
    canonical_order: str = TBG_ZERO_FIELD_COMPANION_KIVC_CANONICAL_ORDER
    stored_projector_convention: str = (
        TBG_ZERO_FIELD_COMPANION_KIVC_STORED_PROJECTOR_CONVENTION
    )
    frame_scope: str = TBG_ZERO_FIELD_COMPANION_KIVC_FRAME_SCOPE
    basis_covariance_scope: str = TBG_ZERO_FIELD_COMPANION_KIVC_BASIS_COVARIANCE_SCOPE
    mapped_U_Tp_validation_scope: str = TBG_ZERO_FIELD_COMPANION_KIVC_TP_SCOPE
    companion_measure_Tp_validation_scope: str = (
        TBG_ZERO_FIELD_COMPANION_KIVC_COMPANION_MEASURE_TP_SCOPE
    )
    companion_measure_source: str = TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE
    companion_measure_Tp_reference_lines: str = (
        TBG_ZERO_FIELD_COMPANION_MEASURE_TP_REFERENCE_LINES
    )
    chern_validation_scope: str = TBG_ZERO_FIELD_COMPANION_KIVC_CHERN_SCOPE
    scientific_scope: str = TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(self, "phi", _finite_real(self.phi, name="phi"))
        if not isinstance(self.source_hashes, TBGZeroFieldCompanionKIVCSourceHashes):
            raise TypeError("source_hashes must be typed K-IVC source hashes")
        expected = {
            "paper_source": KWAN_EQ99_PDF_SOURCE,
            "paper_reference": KWAN_EQ99_REFERENCE,
            "paper_arxiv": KWAN_EQ99_ARXIV,
            "companion_repository": TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
            "companion_commit": TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
            "companion_single_particle_source": (
                TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE
            ),
            "companion_projectors_source": (
                TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE
            ),
            "canonical_order": TBG_ZERO_FIELD_COMPANION_KIVC_CANONICAL_ORDER,
            "stored_projector_convention": (
                TBG_ZERO_FIELD_COMPANION_KIVC_STORED_PROJECTOR_CONVENTION
            ),
            "frame_scope": TBG_ZERO_FIELD_COMPANION_KIVC_FRAME_SCOPE,
            "basis_covariance_scope": (
                TBG_ZERO_FIELD_COMPANION_KIVC_BASIS_COVARIANCE_SCOPE
            ),
            "mapped_U_Tp_validation_scope": TBG_ZERO_FIELD_COMPANION_KIVC_TP_SCOPE,
            "companion_measure_Tp_validation_scope": (
                TBG_ZERO_FIELD_COMPANION_KIVC_COMPANION_MEASURE_TP_SCOPE
            ),
            "companion_measure_source": TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE,
            "companion_measure_Tp_reference_lines": (
                TBG_ZERO_FIELD_COMPANION_MEASURE_TP_REFERENCE_LINES
            ),
            "chern_validation_scope": TBG_ZERO_FIELD_COMPANION_KIVC_CHERN_SCOPE,
            "scientific_scope": TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCOPE,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"Unsupported K-IVC provenance field {name}")
        for name in ("stage2_fingerprint", "stage2_array_hashes_fingerprint"):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        self.source_hashes.validate_pinned_metadata()

    @property
    def fingerprint(self) -> str:
        self.source_hashes.validate_pinned_metadata()
        return _json_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "basis_covariance_scope": self.basis_covariance_scope,
            "canonical_order": self.canonical_order,
            "chern_validation_scope": self.chern_validation_scope,
            "companion_commit": self.companion_commit,
            "companion_measure_source": self.companion_measure_source,
            "companion_measure_Tp_reference_lines": (
                self.companion_measure_Tp_reference_lines
            ),
            "companion_measure_Tp_validation_scope": (
                self.companion_measure_Tp_validation_scope
            ),
            "companion_projectors_source": self.companion_projectors_source,
            "companion_repository": self.companion_repository,
            "companion_single_particle_source": self.companion_single_particle_source,
            "frame_scope": self.frame_scope,
            "mapped_U_Tp_validation_scope": self.mapped_U_Tp_validation_scope,
            "paper_arxiv": self.paper_arxiv,
            "paper_reference": self.paper_reference,
            "paper_source": self.paper_source,
            "phi": self.phi,
            "scientific_scope": self.scientific_scope,
            "source_hashes_fingerprint": self.source_hashes.fingerprint,
            "stage2_array_hashes_fingerprint": self.stage2_array_hashes_fingerprint,
            "stage2_fingerprint": self.stage2_fingerprint,
            "stored_projector_convention": self.stored_projector_convention,
        }

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        payload["source_hashes"] = self.source_hashes.to_metadata()
        return payload


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionKIVCSeedArrayHashes:
    Z_projected: str
    Z_spectra: str
    W_K: str
    W_Kprime: str
    W: str
    anchor_indices_K: str
    anchor_relative_magnitudes_K: str
    Q_canonical: str
    Q_band: str
    P_conventional: str
    P_stored: str
    U_Tp: str
    Gamma_C: str
    source_TR_Z_residuals: str
    kprime_Z_diagonalization_residuals: str
    tp_square_residuals: str
    tp_Q_invariance_residuals: str
    tp_P_invariance_residuals: str
    companion_measure_Tp_residuals: str
    chern_balance_trace: str
    convention: str = TBG_ZERO_FIELD_COMPANION_KIVC_ARRAY_HASH_CONVENTION

    @classmethod
    def from_arrays(
        cls,
        arrays: dict[str, np.ndarray],
    ) -> "TBGZeroFieldCompanionKIVCSeedArrayHashes":
        names = tuple(field.name for field in fields(cls) if field.name != "convention")
        return cls(**{name: _canonical_array_sha256(arrays[name]) for name in names})

    def __post_init__(self) -> None:
        if self.convention != TBG_ZERO_FIELD_COMPANION_KIVC_ARRAY_HASH_CONVENTION:
            raise ValueError("Unsupported K-IVC array-hash convention")
        for field in fields(self):
            if field.name == "convention":
                continue
            digest = getattr(self, field.name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{field.name} must be a lowercase SHA-256 digest")

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {field.name: getattr(self, field.name) for field in fields(self)}
        )

    def to_metadata(self) -> dict[str, str]:
        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionKIVCSeedResiduals:
    Z_hermiticity_max_abs: float
    frame_unitarity_max_abs: float
    source_TR_Z_max_abs: float
    kprime_Z_diagonalization_max_abs: float
    Q_hermiticity_max_abs: float
    Q_involution_max_abs: float
    P_hermiticity_max_abs: float
    P_idempotency_max_abs: float
    P_trace_max_abs: float
    P_rank_spectrum_max_abs: float
    stored_transpose_max_abs: float
    spin_singlet_max_abs: float
    tp_square_max_abs: float
    tp_Q_invariance_max_abs: float
    tp_P_invariance_max_abs: float
    companion_measure_Tp_max_abs: float
    chern_balance_max_abs: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = _finite_real(getattr(self, field.name), name=field.name, nonnegative=True)
            object.__setattr__(self, field.name, value)

    def to_metadata(self) -> dict[str, float]:
        return {field.name: float(getattr(self, field.name)) for field in fields(self)}


def _parent_coefficients(
    source: TBGZeroFieldCompanionSingleParticleResult,
    ik1: int,
    ik2: int,
    tau: int,
) -> np.ndarray:
    params = source.params
    return np.transpose(
        source.coeff[ik1, ik2, :, :, tau, :, :],
        (0, 1, 3, 2),
    ).reshape(params.parent_dimension, params.active_band_count)


def _validate_stage2_source(
    source: TBGZeroFieldCompanionSingleParticleResult,
) -> TBGZeroFieldCompanionSingleParticleArrayHashes:
    if not isinstance(source, TBGZeroFieldCompanionSingleParticleResult):
        raise TypeError(
            "single_particle must be TBGZeroFieldCompanionSingleParticleResult"
        )
    if source.params.active_band_count != 2:
        raise ValueError("Kwan Eq. (99) diagnostic requires exactly two active bands")
    actual = TBGZeroFieldCompanionSingleParticleArrayHashes.from_arrays(
        coeff=source.coeff,
        sp_energy_ev=source.sp_energy_ev,
        U_C2T=source.U_C2T,
    )
    for name in ("coeff", "sp_energy_ev", "U_C2T"):
        if getattr(actual, name) != getattr(source.array_hashes, name):
            raise ValueError(
                f"single_particle_source.{name} no longer matches its source hash"
            )
    return actual


def _first_phase_anchor(
    lifted: np.ndarray,
    *,
    min_relative_magnitude: float,
    label: str,
) -> tuple[int, float, float]:
    """Return the first qualifying component in companion parent order."""

    lifted_norm = float(np.linalg.norm(lifted))
    if not math.isfinite(lifted_norm) or lifted_norm == 0.0:
        raise ValueError(f"{label} has no nonzero lifted microscopic phase anchor")
    magnitudes = np.abs(lifted)
    relative_magnitudes = magnitudes / lifted_norm
    for anchor, relative_magnitude in enumerate(relative_magnitudes):
        if float(relative_magnitude) >= min_relative_magnitude:
            return anchor, float(relative_magnitude), float(magnitudes[anchor])
    raise ValueError(
        f"{label} has no lifted microscopic phase anchor at or above "
        f"relative magnitude {min_relative_magnitude:.3e}; maximum is "
        f"{float(np.max(relative_magnitudes)):.3e}"
    )

def _phase_frame_column(
    column: np.ndarray,
    parent_coefficients: np.ndarray,
    *,
    min_relative_magnitude: float,
    label: str,
) -> tuple[np.ndarray, int, float]:
    # `_parent_coefficients` flattens in exact companion order: species fastest,
    # then g2, then g1.  A first-match scan therefore needs no maximum or tie.
    lifted = parent_coefficients @ column
    anchor, anchor_relative_magnitude, anchor_magnitude = _first_phase_anchor(
        lifted,
        min_relative_magnitude=min_relative_magnitude,
        label=label,
    )
    phased = np.asarray(column, dtype=np.complex128) * np.conj(
        lifted[anchor] / anchor_magnitude
    )
    anchor_value = (parent_coefficients @ phased)[anchor]
    imaginary_tolerance = max(
        1.0e-12 * anchor_magnitude,
        64.0 * np.finfo(np.float64).eps * float(np.linalg.norm(lifted)),
    )
    if (
        anchor_value.real <= 0.0
        or abs(anchor_value.imag) > imaginary_tolerance
    ):
        raise ValueError(f"{label} phase anchor could not be made positive-real")
    return phased, anchor, anchor_relative_magnitude


def _block_frames(W_K: np.ndarray, W_Kprime: np.ndarray) -> np.ndarray:
    N1, N2 = W_K.shape[:2]
    W = np.zeros((N1, N2, 4, 4), dtype=np.complex128)
    W[:, :, :2, :2] = W_K
    W[:, :, 2:, 2:] = W_Kprime
    return W


def _companion_measure_boost0_Tp(P_stored: np.ndarray) -> np.ndarray:
    """Apply ``measure.py`` lines 5-14 and 64-69 with both boosts zero."""

    N1, N2 = P_stored.shape[:2]
    Pex = np.reshape(P_stored, (N1, N2, 2, 2, 2, 2, 2)).copy()
    P_T = np.flip(Pex, axis=(0, 1, 3, 5)).copy()
    P_T = np.roll(P_T, (1, 1), axis=(0, 1))
    P_T = np.conj(P_T)
    P_Tp = P_T.copy()
    P_Tp[:, :, :, 0, :, 1, :] = -P_T[:, :, :, 0, :, 1, :]
    P_Tp[:, :, :, 1, :, 0, :] = -P_T[:, :, :, 1, :, 0, :]
    return np.reshape(P_Tp, P_stored.shape)

def _diagnostic_residuals(arrays: dict[str, np.ndarray]) -> TBGZeroFieldCompanionKIVCSeedResiduals:
    Z = arrays["Z_projected"]
    W = arrays["W"]
    Q = arrays["Q_band"]
    P = arrays["P_conventional"]
    P_stored = arrays["P_stored"]
    identity4 = np.eye(4, dtype=np.complex128)
    expected_rank_spectrum = np.asarray([0.0, 0.0, 1.0, 1.0])
    rank_residual = 0.0
    for matrix in P.reshape((-1, 4, 4)):
        rank_residual = max(
            rank_residual,
            _max_abs(np.linalg.eigvalsh(matrix) - expected_rank_spectrum),
        )
    return TBGZeroFieldCompanionKIVCSeedResiduals(
        Z_hermiticity_max_abs=_max_hermiticity_residual(Z),
        frame_unitarity_max_abs=_max_abs(
            np.swapaxes(W.conj(), -1, -2) @ W - identity4
        ),
        source_TR_Z_max_abs=_max_abs(arrays["source_TR_Z_residuals"]),
        kprime_Z_diagonalization_max_abs=_max_abs(
            arrays["kprime_Z_diagonalization_residuals"]
        ),
        Q_hermiticity_max_abs=_max_hermiticity_residual(Q),
        Q_involution_max_abs=_max_abs(Q @ Q - identity4),
        P_hermiticity_max_abs=max(
            _max_hermiticity_residual(P),
            _max_hermiticity_residual(P_stored),
        ),
        P_idempotency_max_abs=max(
            _max_abs(P @ P - P),
            _max_abs(P_stored @ P_stored - P_stored),
        ),
        P_trace_max_abs=max(
            _max_abs(np.trace(P, axis1=-2, axis2=-1) - 2.0),
            _max_abs(np.trace(P_stored, axis1=-2, axis2=-1) - 2.0),
        ),
        P_rank_spectrum_max_abs=rank_residual,
        stored_transpose_max_abs=_max_abs(
            P_stored - np.swapaxes(P, -1, -2)[:, :, None, :, :]
        ),
        spin_singlet_max_abs=_max_abs(P_stored[:, :, 0] - P_stored[:, :, 1]),
        tp_square_max_abs=_max_abs(arrays["tp_square_residuals"]),
        tp_Q_invariance_max_abs=_max_abs(arrays["tp_Q_invariance_residuals"]),
        tp_P_invariance_max_abs=_max_abs(arrays["tp_P_invariance_residuals"]),
        companion_measure_Tp_max_abs=_max_abs(
            arrays["companion_measure_Tp_residuals"]
        ),
        chern_balance_max_abs=_max_abs(arrays["chern_balance_trace"]),
    )


_RESULT_ARRAY_LAYOUTS = {
    "Z_projected": (lambda N1, N2: (N1, N2, 2, 2, 2), np.complex128),
    "Z_spectra": (lambda N1, N2: (N1, N2, 2, 2), np.float64),
    "W_K": (lambda N1, N2: (N1, N2, 2, 2), np.complex128),
    "W_Kprime": (lambda N1, N2: (N1, N2, 2, 2), np.complex128),
    "W": (lambda N1, N2: (N1, N2, 4, 4), np.complex128),
    "anchor_indices_K": (lambda N1, N2: (N1, N2, 2), np.int64),
    "anchor_relative_magnitudes_K": (
        lambda N1, N2: (N1, N2, 2),
        np.float64,
    ),
    "Q_canonical": (lambda N1, N2: (4, 4), np.complex128),
    "Q_band": (lambda N1, N2: (N1, N2, 4, 4), np.complex128),
    "P_conventional": (lambda N1, N2: (N1, N2, 4, 4), np.complex128),
    "P_stored": (lambda N1, N2: (N1, N2, 2, 4, 4), np.complex128),
    "U_Tp": (lambda N1, N2: (N1, N2, 4, 4), np.complex128),
    "Gamma_C": (lambda N1, N2: (4, 4), np.complex128),
    "source_TR_Z_residuals": (lambda N1, N2: (N1, N2), np.float64),
    "kprime_Z_diagonalization_residuals": (
        lambda N1, N2: (N1, N2),
        np.float64,
    ),
    "tp_square_residuals": (lambda N1, N2: (N1, N2), np.float64),
    "tp_Q_invariance_residuals": (lambda N1, N2: (N1, N2), np.float64),
    "tp_P_invariance_residuals": (lambda N1, N2: (N1, N2), np.float64),
    "companion_measure_Tp_residuals": (
        lambda N1, N2: (N1, N2),
        np.float64,
    ),
    "chern_balance_trace": (lambda N1, N2: (N1, N2), np.complex128),
}


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionKIVCSeedResult:
    single_particle_source: TBGZeroFieldCompanionSingleParticleResult
    phi: float
    sign_gap_tolerance: float
    phase_anchor_min_relative_magnitude: float
    anchor_relative_magnitude_min: float
    validation_tolerance: float
    Z_projected: np.ndarray
    Z_spectra: np.ndarray
    W_K: np.ndarray
    W_Kprime: np.ndarray
    W: np.ndarray
    anchor_indices_K: np.ndarray
    anchor_relative_magnitudes_K: np.ndarray
    Q_canonical: np.ndarray
    Q_band: np.ndarray
    P_conventional: np.ndarray
    P_stored: np.ndarray
    U_Tp: np.ndarray
    Gamma_C: np.ndarray
    source_TR_Z_residuals: np.ndarray
    kprime_Z_diagonalization_residuals: np.ndarray
    tp_square_residuals: np.ndarray
    tp_Q_invariance_residuals: np.ndarray
    tp_P_invariance_residuals: np.ndarray
    companion_measure_Tp_residuals: np.ndarray
    chern_balance_trace: np.ndarray
    residuals: TBGZeroFieldCompanionKIVCSeedResiduals
    provenance: TBGZeroFieldCompanionKIVCSeedProvenance
    array_hashes: TBGZeroFieldCompanionKIVCSeedArrayHashes

    def __post_init__(self) -> None:
        _validate_stage2_source(self.single_particle_source)
        object.__setattr__(self, "phi", _finite_real(self.phi, name="phi"))
        object.__setattr__(
            self,
            "sign_gap_tolerance",
            _finite_real(
                self.sign_gap_tolerance,
                name="sign_gap_tolerance",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "phase_anchor_min_relative_magnitude",
            _validate_phase_anchor_min_relative_magnitude(
                self.phase_anchor_min_relative_magnitude
            ),
        )
        object.__setattr__(
            self,
            "anchor_relative_magnitude_min",
            _finite_real(
                self.anchor_relative_magnitude_min,
                name="anchor_relative_magnitude_min",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "validation_tolerance",
            _finite_real(self.validation_tolerance, name="validation_tolerance"),
        )
        if self.sign_gap_tolerance == 0.0:
            raise ValueError("sign_gap_tolerance must be positive")
        if not 0.0 < self.validation_tolerance <= MAX_VALIDATION_TOLERANCE:
            raise ValueError(
                "validation_tolerance must be > 0 and <= "
                f"MAX_VALIDATION_TOLERANCE ({MAX_VALIDATION_TOLERANCE:.1e})"
            )
        if not isinstance(self.residuals, TBGZeroFieldCompanionKIVCSeedResiduals):
            raise TypeError("residuals must be typed K-IVC residuals")
        if not isinstance(self.provenance, TBGZeroFieldCompanionKIVCSeedProvenance):
            raise TypeError("provenance must be typed K-IVC provenance")
        if not isinstance(self.array_hashes, TBGZeroFieldCompanionKIVCSeedArrayHashes):
            raise TypeError("array_hashes must be typed K-IVC array hashes")
        if self.provenance.phi != self.phi:
            raise ValueError("provenance phi differs from result phi")

        N1 = self.single_particle_source.params.N1
        N2 = self.single_particle_source.params.N2
        arrays: dict[str, np.ndarray] = {}
        for name, (shape_builder, dtype) in _RESULT_ARRAY_LAYOUTS.items():
            array = _readonly_array(
                getattr(self, name),
                name=name,
                shape=shape_builder(N1, N2),
                dtype=dtype,
            )
            arrays[name] = array
            object.__setattr__(self, name, array)
        if TBGZeroFieldCompanionKIVCSeedArrayHashes.from_arrays(arrays) != self.array_hashes:
            raise ValueError("array_hashes do not match K-IVC diagnostic arrays")
        actual_minimum_magnitude = float(
            np.min(arrays["anchor_relative_magnitudes_K"])
        )
        if self.anchor_relative_magnitude_min != actual_minimum_magnitude:
            raise ValueError(
                "anchor_relative_magnitude_min does not match anchor magnitude array"
            )
        self._validate_live_state()

    def _live_arrays(self) -> dict[str, np.ndarray]:
        N1 = self.single_particle_source.params.N1
        N2 = self.single_particle_source.params.N2
        return {
            name: _validate_live_array(
                getattr(self, name),
                name=f"K-IVC result.{name}",
                shape=shape_builder(N1, N2),
                dtype=dtype,
            )
            for name, (shape_builder, dtype) in _RESULT_ARRAY_LAYOUTS.items()
        }

    def _validate_live_state(self) -> dict[str, np.ndarray]:
        source_hashes = _validate_stage2_source(self.single_particle_source)
        if source_hashes.fingerprint != self.provenance.stage2_array_hashes_fingerprint:
            raise ValueError("Stage-2 array hash fingerprint drifted after K-IVC build")
        if self.single_particle_source.fingerprint != self.provenance.stage2_fingerprint:
            raise ValueError("Stage-2 fingerprint drifted after K-IVC build")
        self.provenance.source_hashes.validate_pinned_metadata()
        arrays = self._live_arrays()
        if TBGZeroFieldCompanionKIVCSeedArrayHashes.from_arrays(arrays) != self.array_hashes:
            raise ValueError("K-IVC arrays no longer match their live hashes")

        (
            reconstructed_arrays,
            reconstructed_residuals,
            reconstructed_hashes,
            reconstructed_anchor_minimum,
        ) = _reconstruct_kivc_seed_arrays(
            self.single_particle_source,
            self.phi,
            self.sign_gap_tolerance,
            self.phase_anchor_min_relative_magnitude,
            self.validation_tolerance,
        )
        for name in _RESULT_ARRAY_LAYOUTS:
            actual = arrays[name]
            expected = reconstructed_arrays[name]
            if actual.dtype.kind in "iu":
                matches_replay = np.array_equal(actual, expected)
            else:
                replay_tolerance = (
                    64.0
                    * np.finfo(np.float64).eps
                    * max(1.0, _max_abs(expected))
                )
                matches_replay = _max_abs(actual - expected) <= replay_tolerance
            if not matches_replay:
                raise ValueError(
                    f"{name} does not match deterministic Stage-2 semantic replay"
                )
        for field in fields(self.array_hashes):
            if field.name == "convention":
                continue
            if getattr(self.array_hashes, field.name) != getattr(
                reconstructed_hashes,
                field.name,
            ):
                raise ValueError(
                    f"{field.name} hash does not match deterministic Stage-2 "
                    "semantic replay"
                )
        if self.residuals != reconstructed_residuals:
            raise ValueError(
                "residuals do not match deterministic Stage-2 semantic replay"
            )
        if self.anchor_relative_magnitude_min != reconstructed_anchor_minimum:
            raise ValueError(
                "anchor_relative_magnitude_min does not match deterministic "
                "Stage-2 semantic replay"
            )

        tolerance = self.validation_tolerance
        if not np.array_equal(arrays["Q_canonical"], kwan_eq99_kivc_q(self.phi)):
            raise ValueError("Q_canonical no longer equals Kwan Eq. (99)")
        if not np.array_equal(arrays["Gamma_C"], _GAMMA_CHERN_CANONICAL):
            raise ValueError("Gamma_C no longer equals tau_z kron sigma_z")
        expected_W = _block_frames(arrays["W_K"], arrays["W_Kprime"])
        if not np.array_equal(arrays["W"], expected_W):
            raise ValueError("W no longer equals blockdiag(W_K,W_Kprime)")

        N1 = self.single_particle_source.params.N1
        N2 = self.single_particle_source.params.N2
        for ik1 in range(N1):
            for ik2 in range(N2):
                mk1, mk2 = _minus_mesh_index(ik1, ik2, N1, N2)
                if not np.array_equal(
                    arrays["W_Kprime"][ik1, ik2],
                    np.conj(arrays["W_K"][mk1, mk2]),
                ):
                    raise ValueError("W_Kprime no longer obeys exact source TR pairing")
        expected_Q_band = (
            arrays["W"]
            @ arrays["Q_canonical"]
            @ np.swapaxes(arrays["W"].conj(), -1, -2)
        )
        if _max_abs(arrays["Q_band"] - expected_Q_band) > tolerance:
            raise ValueError("Q_band does not close from W Q_canonical W^dagger")
        expected_P = 0.5 * (
            np.eye(4, dtype=np.complex128) + arrays["Q_band"]
        )
        if _max_abs(arrays["P_conventional"] - expected_P) > tolerance:
            raise ValueError("P_conventional does not equal (I+Q_band)/2")
        expected_stored = np.repeat(
            np.swapaxes(expected_P, -1, -2)[:, :, None, :, :],
            2,
            axis=2,
        )
        if not np.array_equal(arrays["P_stored"], expected_stored):
            raise ValueError("P_stored is not the two-spin matrix-axis transpose")
        expected_companion_Tp = _companion_measure_boost0_Tp(arrays["P_stored"])
        expected_companion_Tp_residuals = np.max(
            np.abs(arrays["P_stored"] - expected_companion_Tp),
            axis=(2, 3, 4),
        )
        if not np.array_equal(
            arrays["companion_measure_Tp_residuals"],
            expected_companion_Tp_residuals,
        ):
            raise ValueError(
                "companion_measure_Tp_residuals do not match the pinned boost0 "
                "stored-projector transform"
            )
        if np.any(
            arrays["anchor_relative_magnitudes_K"]
            < self.phase_anchor_min_relative_magnitude
        ):
            raise ValueError(
                "anchor relative magnitude fell below the configured phase-anchor floor"
            )
        for ik1 in range(N1):
            for ik2 in range(N2):
                C_K = _parent_coefficients(
                    self.single_particle_source,
                    ik1,
                    ik2,
                    0,
                )
                for canonical_index in range(2):
                    label = f"K frame ({ik1},{ik2}) anchor receipt"
                    lifted = C_K @ arrays["W_K"][
                        ik1,
                        ik2,
                        :,
                        canonical_index,
                    ]
                    expected_index, expected_relative_magnitude, anchor_magnitude = (
                        _first_phase_anchor(
                            lifted,
                            min_relative_magnitude=(
                                self.phase_anchor_min_relative_magnitude
                            ),
                            label=label,
                        )
                    )
                    if (
                        int(arrays["anchor_indices_K"][ik1, ik2, canonical_index])
                        != expected_index
                    ):
                        raise ValueError(
                            "anchor_indices_K no longer records the first qualifying "
                            "companion-parent component"
                        )
                    stored_relative_magnitude = float(
                        arrays["anchor_relative_magnitudes_K"][
                            ik1,
                            ik2,
                            canonical_index,
                        ]
                    )
                    if (
                        abs(
                            stored_relative_magnitude
                            - expected_relative_magnitude
                        )
                        > tolerance
                    ):
                        raise ValueError(
                            "anchor_relative_magnitudes_K no longer matches the "
                            "lifted frame"
                        )
                    anchor_value = lifted[expected_index]
                    imaginary_tolerance = max(
                        1.0e-12 * anchor_magnitude,
                        64.0
                        * np.finfo(np.float64).eps
                        * float(np.linalg.norm(lifted)),
                    )
                    if (
                        anchor_value.real <= 0.0
                        or abs(anchor_value.imag) > imaginary_tolerance
                    ):
                        raise ValueError(
                            "ordered phase anchor is no longer positive-real"
                        )
        if self.anchor_relative_magnitude_min != float(
            np.min(arrays["anchor_relative_magnitudes_K"])
        ):
            raise ValueError(
                "anchor_relative_magnitude_min no longer matches live magnitudes"
            )

        recomputed_residuals = _diagnostic_residuals(arrays)
        if recomputed_residuals != self.residuals:
            raise ValueError("residuals no longer match live K-IVC arrays")
        threshold_fields = (
            "Z_hermiticity_max_abs",
            "frame_unitarity_max_abs",
            "source_TR_Z_max_abs",
            "kprime_Z_diagonalization_max_abs",
            "Q_hermiticity_max_abs",
            "Q_involution_max_abs",
            "P_hermiticity_max_abs",
            "P_idempotency_max_abs",
            "P_trace_max_abs",
            "P_rank_spectrum_max_abs",
            "stored_transpose_max_abs",
            "spin_singlet_max_abs",
            "tp_square_max_abs",
            "tp_Q_invariance_max_abs",
            "tp_P_invariance_max_abs",
            "companion_measure_Tp_max_abs",
            "chern_balance_max_abs",
        )
        for name in threshold_fields:
            if getattr(self.residuals, name) > tolerance:
                raise ValueError(
                    f"K-IVC diagnostic {name} exceeds validation tolerance"
                )
        if np.any(arrays["Z_spectra"][..., 0] <= self.sign_gap_tolerance):
            raise ValueError("Z+ spectrum lost its required positive sign gap")
        if np.any(arrays["Z_spectra"][..., 1] >= -self.sign_gap_tolerance):
            raise ValueError("Z- spectrum lost its required negative sign gap")
        return arrays

    @property
    def fingerprint(self) -> str:
        self._validate_live_state()
        return _json_sha256(
            {
                "array_hashes": self.array_hashes.fingerprint,
                "anchor_relative_magnitude_min": (
                    self.anchor_relative_magnitude_min
                ),
                "phase_anchor_min_relative_magnitude": (
                    self.phase_anchor_min_relative_magnitude
                ),
                "phi": self.phi,
                "provenance": self.provenance.fingerprint,
                "residuals": self.residuals.to_metadata(),
                "schema": TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCHEMA,
                "schema_version": TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCHEMA_VERSION,
                "scope": TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCOPE,
                "sign_gap_tolerance": self.sign_gap_tolerance,
                "validation_tolerance": self.validation_tolerance,
            }
        )

    def to_metadata(self) -> dict[str, object]:
        fingerprint = self.fingerprint
        return {
            "array_hashes": self.array_hashes.to_metadata(),
            "basis_covariance_scope": (
                TBG_ZERO_FIELD_COMPANION_KIVC_BASIS_COVARIANCE_SCOPE
            ),
            "canonical_order": TBG_ZERO_FIELD_COMPANION_KIVC_CANONICAL_ORDER,
            "chern_validation_scope": TBG_ZERO_FIELD_COMPANION_KIVC_CHERN_SCOPE,
            "companion_measure_Tp_validation_scope": (
                TBG_ZERO_FIELD_COMPANION_KIVC_COMPANION_MEASURE_TP_SCOPE
            ),
            "fingerprint": fingerprint,
            "frame_scope": TBG_ZERO_FIELD_COMPANION_KIVC_FRAME_SCOPE,
            "mapped_U_Tp_validation_scope": TBG_ZERO_FIELD_COMPANION_KIVC_TP_SCOPE,
            "anchor_relative_magnitude_min": self.anchor_relative_magnitude_min,
            "phase_anchor_min_relative_magnitude": (
                self.phase_anchor_min_relative_magnitude
            ),
            "phase_anchor_min_relative_magnitude_hard_max": (
                MAX_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE
            ),
            "phase_anchor_min_relative_magnitude_hard_min": (
                TBG_ZERO_FIELD_COMPANION_KIVC_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE
            ),
            "phi": self.phi,
            "provenance": self.provenance.to_metadata(),
            "stage2_array_hashes": {
                "U_C2T": self.single_particle_source.array_hashes.U_C2T,
                "coeff": self.single_particle_source.array_hashes.coeff,
                "fingerprint": self.single_particle_source.array_hashes.fingerprint,
                "sp_energy_ev": self.single_particle_source.array_hashes.sp_energy_ev,
            },
            "residuals": self.residuals.to_metadata(),
            "schema": TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCHEMA,
            "schema_version": TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCHEMA_VERSION,
            "scope": TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCOPE,
            "sign_gap_tolerance": self.sign_gap_tolerance,
            "stored_projector_convention": (
                TBG_ZERO_FIELD_COMPANION_KIVC_STORED_PROJECTOR_CONVENTION
            ),
            "validation_tolerance": self.validation_tolerance,
            "validation_tolerance_hard_max": MAX_VALIDATION_TOLERANCE,
        }


def _reconstruct_kivc_seed_arrays(
    single_particle: TBGZeroFieldCompanionSingleParticleResult,
    phi: object,
    sign_gap: object,
    anchor_threshold: object,
    validation_tol: object,
) -> tuple[
    dict[str, np.ndarray],
    TBGZeroFieldCompanionKIVCSeedResiduals,
    TBGZeroFieldCompanionKIVCSeedArrayHashes,
    float,
]:
    """Purely reconstruct every retained Stage-5 array and numerical receipt."""

    _validate_stage2_source(single_particle)
    resolved_phi = _finite_real(phi, name="phi")
    sign_tolerance = _finite_real(
        sign_gap,
        name="sign_gap_tolerance",
        nonnegative=True,
    )
    anchor_min_relative_magnitude = (
        _validate_phase_anchor_min_relative_magnitude(anchor_threshold)
    )
    numerical_tolerance = _finite_real(
        validation_tol,
        name="validation_tolerance",
    )
    if sign_tolerance == 0.0:
        raise ValueError("sign_gap_tolerance must be positive")
    if not 0.0 < numerical_tolerance <= MAX_VALIDATION_TOLERANCE:
        raise ValueError(
            "validation_tolerance must be > 0 and <= "
            f"MAX_VALIDATION_TOLERANCE ({MAX_VALIDATION_TOLERANCE:.1e})"
        )

    params = single_particle.params
    N1, N2 = params.N1, params.N2
    microscopic_Z = np.tile(
        np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64),
        4 * params.Ng1 * params.Ng2,
    )
    Z_projected = np.empty((N1, N2, 2, 2, 2), dtype=np.complex128)
    for ik1 in range(N1):
        for ik2 in range(N2):
            for tau in range(2):
                C = _parent_coefficients(single_particle, ik1, ik2, tau)
                Z_projected[ik1, ik2, tau] = C.conj().T @ (
                    microscopic_Z[:, None] * C
                )
    if not np.all(np.isfinite(Z_projected)):
        raise ValueError("Projected microscopic-sublattice Z contains nonfinite values")
    if _max_hermiticity_residual(Z_projected) > numerical_tolerance:
        raise ValueError("Projected microscopic-sublattice Z is non-Hermitian")

    Z_spectra = np.empty((N1, N2, 2, 2), dtype=np.float64)
    W_K = np.empty((N1, N2, 2, 2), dtype=np.complex128)
    anchor_indices_K = np.empty((N1, N2, 2), dtype=np.int64)
    anchor_relative_magnitudes_K = np.empty(
        (N1, N2, 2),
        dtype=np.float64,
    )
    for ik1 in range(N1):
        for ik2 in range(N2):
            eigenvalues, eigenvectors = np.linalg.eigh(Z_projected[ik1, ik2, 0])
            negative = float(eigenvalues[0])
            positive = float(eigenvalues[1])
            if positive <= sign_tolerance or negative >= -sign_tolerance:
                raise ValueError(
                    "Projected K-valley Z lacks one positive and one negative "
                    f"eigenvalue at ({ik1},{ik2}): {eigenvalues.tolist()}"
                )
            C_K = _parent_coefficients(single_particle, ik1, ik2, 0)
            for canonical_index, eigen_index in enumerate((1, 0)):
                phased, anchor, anchor_relative_magnitude = _phase_frame_column(
                    eigenvectors[:, eigen_index],
                    C_K,
                    min_relative_magnitude=anchor_min_relative_magnitude,
                    label=(
                        f"K frame ({ik1},{ik2}) "
                        f"{'Z+' if canonical_index == 0 else 'Z-'}"
                    ),
                )
                W_K[ik1, ik2, :, canonical_index] = phased
                anchor_indices_K[ik1, ik2, canonical_index] = anchor
                anchor_relative_magnitudes_K[
                    ik1,
                    ik2,
                    canonical_index,
                ] = anchor_relative_magnitude
            Z_spectra[ik1, ik2, 0] = (positive, negative)

    W_Kprime = np.empty_like(W_K)
    source_TR_Z_residuals = np.empty((N1, N2), dtype=np.float64)
    kprime_Z_diagonalization_residuals = np.empty((N1, N2), dtype=np.float64)
    for ik1 in range(N1):
        for ik2 in range(N2):
            mk1, mk2 = _minus_mesh_index(ik1, ik2, N1, N2)
            W_Kprime[ik1, ik2] = np.conj(W_K[mk1, mk2])
            Z_Kprime = Z_projected[ik1, ik2, 1]
            source_TR_Z_residuals[ik1, ik2] = _max_abs(
                Z_Kprime - np.conj(Z_projected[mk1, mk2, 0])
            )
            transformed = W_Kprime[ik1, ik2].conj().T @ Z_Kprime @ W_Kprime[
                ik1, ik2
            ]
            independent_eigenvalues = np.linalg.eigvalsh(Z_Kprime)
            expected_order = independent_eigenvalues[[1, 0]]
            diagonalization_residual = transformed - np.diag(expected_order)
            kprime_Z_diagonalization_residuals[ik1, ik2] = _max_abs(
                diagonalization_residual
            )
            if (
                source_TR_Z_residuals[ik1, ik2] > numerical_tolerance
                or kprime_Z_diagonalization_residuals[ik1, ik2]
                > numerical_tolerance
            ):
                raise ValueError(
                    "Exact source-TR K' frame does not independently diagonalize "
                    f"projected Z_Kprime at ({ik1},{ik2})"
                )
            positive, negative = float(expected_order[0]), float(expected_order[1])
            if positive <= sign_tolerance or negative >= -sign_tolerance:
                raise ValueError(
                    "Projected K'-valley Z lacks the required positive/negative sign gap"
                )
            Z_spectra[ik1, ik2, 1] = (positive, negative)

    W = _block_frames(W_K, W_Kprime)
    identity4 = np.eye(4, dtype=np.complex128)
    if _max_abs(np.swapaxes(W.conj(), -1, -2) @ W - identity4) > numerical_tolerance:
        raise ValueError("Canonical projected-Z frames are not unitary")

    Q_canonical = np.array(kwan_eq99_kivc_q(resolved_phi), copy=True)
    Q_band = W @ Q_canonical @ np.swapaxes(W.conj(), -1, -2)
    P_conventional = 0.5 * (identity4 + Q_band)
    P_stored = np.repeat(
        np.swapaxes(P_conventional, -1, -2)[:, :, None, :, :],
        2,
        axis=2,
    )

    U_Tp = np.empty((N1, N2, 4, 4), dtype=np.complex128)
    for ik1 in range(N1):
        for ik2 in range(N2):
            mk1, mk2 = _minus_mesh_index(ik1, ik2, N1, N2)
            U_Tp[ik1, ik2] = W[ik1, ik2] @ _TP_CANONICAL @ W[mk1, mk2].T

    tp_square_residuals = np.empty((N1, N2), dtype=np.float64)
    tp_Q_invariance_residuals = np.empty((N1, N2), dtype=np.float64)
    tp_P_invariance_residuals = np.empty((N1, N2), dtype=np.float64)
    companion_measure_Tp = _companion_measure_boost0_Tp(P_stored)
    companion_measure_Tp_residuals = np.max(
        np.abs(P_stored - companion_measure_Tp),
        axis=(2, 3, 4),
    )
    chern_balance_trace = np.empty((N1, N2), dtype=np.complex128)
    for ik1 in range(N1):
        for ik2 in range(N2):
            mk1, mk2 = _minus_mesh_index(ik1, ik2, N1, N2)
            U = U_Tp[ik1, ik2]
            tp_square_residuals[ik1, ik2] = _max_abs(
                U @ np.conj(U_Tp[mk1, mk2]) + identity4
            )
            tp_Q_invariance_residuals[ik1, ik2] = _max_abs(
                U @ np.conj(Q_band[mk1, mk2]) @ U.conj().T - Q_band[ik1, ik2]
            )
            tp_P_invariance_residuals[ik1, ik2] = _max_abs(
                U
                @ np.conj(P_conventional[mk1, mk2])
                @ U.conj().T
                - P_conventional[ik1, ik2]
            )
            canonical_projector = (
                W[ik1, ik2].conj().T
                @ P_conventional[ik1, ik2]
                @ W[ik1, ik2]
            )
            chern_balance_trace[ik1, ik2] = np.trace(
                _GAMMA_CHERN_CANONICAL @ canonical_projector
            )

    arrays = {
        "Z_projected": Z_projected,
        "Z_spectra": Z_spectra,
        "W_K": W_K,
        "W_Kprime": W_Kprime,
        "W": W,
        "anchor_indices_K": anchor_indices_K,
        "anchor_relative_magnitudes_K": anchor_relative_magnitudes_K,
        "Q_canonical": Q_canonical,
        "Q_band": Q_band,
        "P_conventional": P_conventional,
        "P_stored": P_stored,
        "U_Tp": U_Tp,
        "Gamma_C": np.array(_GAMMA_CHERN_CANONICAL, copy=True),
        "source_TR_Z_residuals": source_TR_Z_residuals,
        "kprime_Z_diagonalization_residuals": (
            kprime_Z_diagonalization_residuals
        ),
        "tp_square_residuals": tp_square_residuals,
        "tp_Q_invariance_residuals": tp_Q_invariance_residuals,
        "tp_P_invariance_residuals": tp_P_invariance_residuals,
        "companion_measure_Tp_residuals": companion_measure_Tp_residuals,
        "chern_balance_trace": chern_balance_trace,
    }
    if set(arrays) != set(_RESULT_ARRAY_LAYOUTS):
        raise RuntimeError("Internal K-IVC reconstruction array inventory drifted")
    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("K-IVC diagnostic produced nonfinite arrays")
    residuals = _diagnostic_residuals(arrays)
    for field in fields(residuals):
        if getattr(residuals, field.name) > numerical_tolerance:
            raise ValueError(
                f"K-IVC diagnostic {field.name} exceeds validation tolerance: "
                f"{getattr(residuals, field.name):.3e}"
            )

    array_hashes = TBGZeroFieldCompanionKIVCSeedArrayHashes.from_arrays(arrays)
    anchor_relative_magnitude_min = float(
        np.min(anchor_relative_magnitudes_K)
    )
    return arrays, residuals, array_hashes, anchor_relative_magnitude_min

def build_tbg_zero_field_companion_kivc_seed(
    single_particle: TBGZeroFieldCompanionSingleParticleResult,
    *,
    phi: float = 0.0,
    sign_gap_tolerance: float = TBG_ZERO_FIELD_COMPANION_KIVC_SIGN_GAP_TOLERANCE,
    phase_anchor_min_relative_magnitude: float = (
        TBG_ZERO_FIELD_COMPANION_KIVC_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE
    ),
    validation_tolerance: float = TBG_ZERO_FIELD_COMPANION_KIVC_VALIDATION_TOLERANCE,
) -> TBGZeroFieldCompanionKIVCSeedResult:
    """Build the isolated Eq. (99) diagnostic without invoking HF or FHS."""

    source_array_hashes = _validate_stage2_source(single_particle)
    resolved_phi = _finite_real(phi, name="phi")
    sign_tolerance = _finite_real(
        sign_gap_tolerance,
        name="sign_gap_tolerance",
        nonnegative=True,
    )
    anchor_min_relative_magnitude = (
        _validate_phase_anchor_min_relative_magnitude(
            phase_anchor_min_relative_magnitude
        )
    )
    numerical_tolerance = _finite_real(
        validation_tolerance,
        name="validation_tolerance",
    )
    if sign_tolerance == 0.0:
        raise ValueError("sign_gap_tolerance must be positive")
    if not 0.0 < numerical_tolerance <= MAX_VALIDATION_TOLERANCE:
        raise ValueError(
            "validation_tolerance must be > 0 and <= "
            f"MAX_VALIDATION_TOLERANCE ({MAX_VALIDATION_TOLERANCE:.1e})"
        )

    (
        arrays,
        residuals,
        array_hashes,
        anchor_relative_magnitude_min,
    ) = _reconstruct_kivc_seed_arrays(
        single_particle,
        resolved_phi,
        sign_tolerance,
        anchor_min_relative_magnitude,
        numerical_tolerance,
    )
    source_hashes = TBGZeroFieldCompanionKIVCSourceHashes.from_pinned_metadata()
    provenance = TBGZeroFieldCompanionKIVCSeedProvenance(
        phi=resolved_phi,
        stage2_fingerprint=single_particle.fingerprint,
        stage2_array_hashes_fingerprint=source_array_hashes.fingerprint,
        source_hashes=source_hashes,
    )
    return TBGZeroFieldCompanionKIVCSeedResult(
        single_particle_source=single_particle,
        phi=resolved_phi,
        sign_gap_tolerance=sign_tolerance,
        phase_anchor_min_relative_magnitude=anchor_min_relative_magnitude,
        anchor_relative_magnitude_min=anchor_relative_magnitude_min,
        validation_tolerance=numerical_tolerance,
        residuals=residuals,
        provenance=provenance,
        array_hashes=array_hashes,
        **arrays,
    )


__all__ = [
    "KWAN_EQ99_ARXIV",
    "KWAN_EQ99_PDF_SHA256",
    "KWAN_EQ99_PDF_SOURCE",
    "KWAN_EQ99_REFERENCE",
    "MAX_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE",
    "MAX_VALIDATION_TOLERANCE",
    "TBGZeroFieldCompanionKIVCSeedArrayHashes",
    "TBGZeroFieldCompanionKIVCSeedProvenance",
    "TBGZeroFieldCompanionKIVCSeedResiduals",
    "TBGZeroFieldCompanionKIVCSeedResult",
    "TBGZeroFieldCompanionKIVCSourceHashes",
    "TBG_ZERO_FIELD_COMPANION_KIVC_ARRAY_HASH_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_KIVC_BASIS_COVARIANCE_SCOPE",
    "TBG_ZERO_FIELD_COMPANION_KIVC_CANONICAL_ORDER",
    "TBG_ZERO_FIELD_COMPANION_KIVC_CHERN_SCOPE",
    "TBG_ZERO_FIELD_COMPANION_KIVC_COMPANION_MEASURE_TP_SCOPE",
    "TBG_ZERO_FIELD_COMPANION_KIVC_EXTERNAL_AUTHORITY_FILES",
    "TBG_ZERO_FIELD_COMPANION_KIVC_FRAME_SCOPE",
    "TBG_ZERO_FIELD_COMPANION_KIVC_PHASE_ANCHOR_MIN_RELATIVE_MAGNITUDE",
    "TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCHEMA",
    "TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCHEMA_VERSION",
    "TBG_ZERO_FIELD_COMPANION_KIVC_SEED_SCOPE",
    "TBG_ZERO_FIELD_COMPANION_KIVC_SIGN_GAP_TOLERANCE",
    "TBG_ZERO_FIELD_COMPANION_KIVC_STORED_PROJECTOR_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_KIVC_TP_SCOPE",
    "TBG_ZERO_FIELD_COMPANION_KIVC_VALIDATION_TOLERANCE",
    "TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE",
    "TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256",
    "TBG_ZERO_FIELD_COMPANION_MEASURE_TP_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE",
    "TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256",
    "build_tbg_zero_field_companion_kivc_seed",
    "kwan_eq99_kivc_q",
    "validate_tbg_zero_field_companion_kivc_external_authorities",
]
