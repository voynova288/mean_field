"""Source-faithful companion interaction diagnostic for zero-field TBG.

This module is an isolated port of ``reference/TBG-HF/singleParticle.py``
``gen_interaction`` lines 205--258.  It consumes the companion single-particle
parameters and reciprocal-lattice geometry, retains the source's SI/eV units,
and returns the source ``intFT`` tensor as read-only ``intFT_ev``.  It is not a
production interaction, HF, or TDHF input and is intentionally not exported
from :mod:`mean_field.systems.tbg.zero_field`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Final, Literal

import numpy as np

from .companion_geometry import (
    TBGZeroFieldCompanionInteractionGeometry,
    TBGZeroFieldCompanionInteractionSpec as TBGZeroFieldCompanionInteractionGeometrySpec,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
    build_tbg_zero_field_companion_interaction_geometry,
)
from .companion_single_particle import (
    TBGZeroFieldCompanionRLVGeometry,
    TBGZeroFieldCompanionSingleParticleParams,
    TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE,
    TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256,
    gen_RLVs,
    kD,
)

# Literal pinned ``reference/TBG-HF/constants.py`` values.
echarge: Final[float] = 1.602176634e-19  # C
epsilon0: Final[float] = 8.854e-12  # F m^-1

TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA: Final[str] = (
    "mean_field.tbg.zero_field.companion_interaction"
)
TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA_VERSION: Final[int] = 1
TBG_ZERO_FIELD_COMPANION_INTERACTION_SCOPE: Final[str] = (
    "diagnostic_interaction_parity_only_not_production_HF_or_TDHF"
)
TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_FUNCTION: Final[str] = "gen_interaction"
TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_LINES: Final[str] = "205-258"
TBG_ZERO_FIELD_COMPANION_INTERACTION_SUPPORT_REFERENCE_LINES: Final[str] = "220-239"
TBG_ZERO_FIELD_COMPANION_INTERACTION_Q0_REFERENCE_LINES: Final[str] = "240-250"
TBG_ZERO_FIELD_COMPANION_INTERACTION_FINITE_Q_KERNEL_REFERENCE_LINES: Final[str] = (
    "251-257"
)
TBG_ZERO_FIELD_COMPANION_INTERACTION_SOURCE_DSC_M: Final[float] = (
    2.5000000000000002e-8
)
TBG_ZERO_FIELD_COMPANION_INTERACTION_ARRAY_HASH_CONVENTION: Final[str] = (
    "sha256_little_endian_int64_shape_then_C_order_canonical_bytes"
)
TBG_ZERO_FIELD_COMPANION_INTERACTION_ARRAY_HASH_SEMANTICS: Final[str] = (
    "artifact_integrity_only_not_cross_platform_float_parity"
)
TBG_ZERO_FIELD_COMPANION_INTERACTION_SOURCE_UNITS: Final[dict[str, str]] = {
    "b1_b2": "dimensionless_in_units_of_2*kD*sin(theta/2)",
    "coulomb_prefactor": "eV*m",
    "dsc": "m",
    "intFT": "eV",
    "physical_q": "m^-1",
    "theta": "degree",
    "total_real_space_area": "m^2",
}
TBG_ZERO_FIELD_COMPANION_INTERACTION_CUTOFF_CONVENTION: Final[str] = (
    "G1_in_range(-NG1,NG1);G2_in_range(-NG2,NG2);"
    "R=min(NG1*perp_norm(b1,b2),NG1*perp_norm(b2,b1));"
    "strict_norm_total_Q_lt_R_minus_1e-5"
)
TBG_ZERO_FIELD_COMPANION_INTERACTION_KERNEL_CONVENTION: Final[str] = (
    "q=(2*kD*sin(theta/2))*Q;U=echarge^2/(2*epsilon0)/echarge;"
    "dual=U*tanh(norm(q)*dsc)/norm(q)/area;"
    "single=U*(1-exp(-2*dsc*norm(q)))/norm(q)/area;"
    "finite_q0_only_when_include_q0"
)


def _strict_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be a positive integer (bool is not accepted), got {value!r}"
        )
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return resolved


def _finite_positive_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(
            f"{name} must be a finite positive real scalar (bool is not accepted), "
            f"got {value!r}"
        )
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return resolved


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 hexadecimal string")
    resolved = value.strip().lower()
    if len(resolved) != 64 or any(character not in "0123456789abcdef" for character in resolved):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    return resolved


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype("<f8")))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionInteractionSpec:
    """Pinned source interaction fields in their original units.

    Schema v1 freezes ``NG1=NG2=5`` and the exact source float
    ``dsc=2.5000000000000002e-8 m`` (physically 25 nm).  Gate geometry and
    source q=0 inclusion remain explicit diagnostic choices.
    """

    NG1: int = 5
    NG2: int = 5
    dsc_m: float = TBG_ZERO_FIELD_COMPANION_INTERACTION_SOURCE_DSC_M
    gates: Literal["dual", "single"] = "dual"
    include_q0: bool = True

    def __post_init__(self) -> None:
        NG1 = _strict_positive_int(self.NG1, name="NG1")
        NG2 = _strict_positive_int(self.NG2, name="NG2")
        dsc_m = _finite_positive_real(self.dsc_m, name="dsc_m")
        if NG1 != 5 or NG2 != 5:
            raise ValueError("Companion interaction schema-v1 freezes NG1=5 and NG2=5")
        if dsc_m != TBG_ZERO_FIELD_COMPANION_INTERACTION_SOURCE_DSC_M:
            raise ValueError(
                "Companion interaction schema-v1 freezes "
                "dsc_m=2.5000000000000002e-8 m"
            )
        if self.gates not in ("dual", "single"):
            raise ValueError("gates must be either 'dual' or 'single'")
        if not isinstance(self.include_q0, bool):
            raise TypeError("include_q0 must be bool")
        object.__setattr__(self, "NG1", NG1)
        object.__setattr__(self, "NG2", NG2)
        object.__setattr__(self, "dsc_m", dsc_m)

    def to_companion_input(self) -> dict[str, int | float | str | bool]:
        """Return the exact source dictionary fields represented by this spec."""

        return {
            "NG1": self.NG1,
            "NG2": self.NG2,
            "dsc": self.dsc_m,
            "gates": self.gates,
            "include_q=0": self.include_q0,
        }

    def _payload(self) -> dict[str, object]:
        return {
            "cutoff_convention": TBG_ZERO_FIELD_COMPANION_INTERACTION_CUTOFF_CONVENTION,
            "input": self.to_companion_input(),
            "kernel_convention": TBG_ZERO_FIELD_COMPANION_INTERACTION_KERNEL_CONVENTION,
            "reference_commit": TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
            "reference_function": TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_FUNCTION,
            "reference_lines": TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_LINES,
            "reference_repository": TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
            "reference_source": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
            "reference_source_sha256": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
            "schema": TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA,
            "schema_version": TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA_VERSION,
            "source_units": dict(TBG_ZERO_FIELD_COMPANION_INTERACTION_SOURCE_UNITS),
        }

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self._payload())

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionInteractionArrayHashes:
    """Canonical integrity hash for the deterministic interaction tensor."""

    intFT_ev: str
    convention: str = TBG_ZERO_FIELD_COMPANION_INTERACTION_ARRAY_HASH_CONVENTION
    semantics: str = TBG_ZERO_FIELD_COMPANION_INTERACTION_ARRAY_HASH_SEMANTICS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intFT_ev",
            _validate_sha256(self.intFT_ev, name="intFT_ev"),
        )
        if self.convention != TBG_ZERO_FIELD_COMPANION_INTERACTION_ARRAY_HASH_CONVENTION:
            raise ValueError("Unsupported companion interaction array-hash convention")
        if self.semantics != TBG_ZERO_FIELD_COMPANION_INTERACTION_ARRAY_HASH_SEMANTICS:
            raise ValueError("Unsupported companion interaction array-hash semantics")

    @classmethod
    def from_array(cls, intFT_ev: np.ndarray) -> TBGZeroFieldCompanionInteractionArrayHashes:
        return cls(intFT_ev=_canonical_array_sha256(intFT_ev))

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "convention": self.convention,
                "intFT_ev": self.intFT_ev,
                "semantics": self.semantics,
            }
        )

    def to_metadata(self) -> dict[str, str]:
        return {
            "convention": self.convention,
            "fingerprint": self.fingerprint,
            "intFT_ev": self.intFT_ev,
            "semantics": self.semantics,
        }


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionInteractionProvenance:
    """Pinned source identity and deliberately diagnostic-only scope."""

    reference_repository: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY
    reference_commit: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT
    reference_source: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE
    reference_source_sha256: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256
    constants_source: str = TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE
    constants_source_sha256: str = TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256
    rlv_reference_lines: str = "20-49"
    interaction_reference_function: str = TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_FUNCTION
    interaction_reference_lines: str = TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_LINES
    interaction_support_reference_lines: str = (
        TBG_ZERO_FIELD_COMPANION_INTERACTION_SUPPORT_REFERENCE_LINES
    )
    interaction_q0_reference_lines: str = (
        TBG_ZERO_FIELD_COMPANION_INTERACTION_Q0_REFERENCE_LINES
    )
    interaction_finite_q_kernel_reference_lines: str = (
        TBG_ZERO_FIELD_COMPANION_INTERACTION_FINITE_Q_KERNEL_REFERENCE_LINES
    )
    array_hash_semantics: str = TBG_ZERO_FIELD_COMPANION_INTERACTION_ARRAY_HASH_SEMANTICS
    scientific_scope: str = TBG_ZERO_FIELD_COMPANION_INTERACTION_SCOPE

    def __post_init__(self) -> None:
        expected = {
            "reference_repository": TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
            "reference_commit": TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
            "reference_source": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
            "reference_source_sha256": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
            "constants_source": TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE,
            "constants_source_sha256": TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256,
            "rlv_reference_lines": "20-49",
            "interaction_reference_function": (
                TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_FUNCTION
            ),
            "interaction_reference_lines": TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_LINES,
            "interaction_support_reference_lines": (
                TBG_ZERO_FIELD_COMPANION_INTERACTION_SUPPORT_REFERENCE_LINES
            ),
            "interaction_q0_reference_lines": (
                TBG_ZERO_FIELD_COMPANION_INTERACTION_Q0_REFERENCE_LINES
            ),
            "interaction_finite_q_kernel_reference_lines": (
                TBG_ZERO_FIELD_COMPANION_INTERACTION_FINITE_Q_KERNEL_REFERENCE_LINES
            ),
            "array_hash_semantics": TBG_ZERO_FIELD_COMPANION_INTERACTION_ARRAY_HASH_SEMANTICS,
            "scientific_scope": TBG_ZERO_FIELD_COMPANION_INTERACTION_SCOPE,
        }
        for name, pinned in expected.items():
            if getattr(self, name) != pinned:
                raise ValueError(f"{name} differs from the pinned companion provenance")

    def to_metadata(self) -> dict[str, str]:
        return {
            "array_hash_semantics": self.array_hash_semantics,
            "constants_source": self.constants_source,
            "constants_source_sha256": self.constants_source_sha256,
            "interaction_finite_q_kernel_reference_lines": (
                self.interaction_finite_q_kernel_reference_lines
            ),
            "interaction_q0_reference_lines": self.interaction_q0_reference_lines,
            "interaction_reference_function": self.interaction_reference_function,
            "interaction_reference_lines": self.interaction_reference_lines,
            "interaction_support_reference_lines": (
                self.interaction_support_reference_lines
            ),
            "reference_commit": self.reference_commit,
            "reference_repository": self.reference_repository,
            "reference_source": self.reference_source,
            "reference_source_sha256": self.reference_source_sha256,
            "rlv_reference_lines": self.rlv_reference_lines,
            "scientific_scope": self.scientific_scope,
        }

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self.to_metadata())


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionInteractionResult:
    """Read-only source ``intFT`` plus geometry and provenance receipts."""

    params: TBGZeroFieldCompanionSingleParticleParams
    spec: TBGZeroFieldCompanionInteractionSpec
    intFT_ev: np.ndarray
    rlv_geometry: TBGZeroFieldCompanionRLVGeometry
    interaction_geometry: TBGZeroFieldCompanionInteractionGeometry
    provenance: TBGZeroFieldCompanionInteractionProvenance
    array_hashes: TBGZeroFieldCompanionInteractionArrayHashes

    def __post_init__(self) -> None:
        if not isinstance(self.params, TBGZeroFieldCompanionSingleParticleParams):
            raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
        if not isinstance(self.spec, TBGZeroFieldCompanionInteractionSpec):
            raise TypeError("spec must be TBGZeroFieldCompanionInteractionSpec")
        if not isinstance(self.rlv_geometry, TBGZeroFieldCompanionRLVGeometry):
            raise TypeError("rlv_geometry must be TBGZeroFieldCompanionRLVGeometry")
        if self.rlv_geometry.params_fingerprint != self.params.fingerprint:
            raise ValueError("rlv_geometry is not bound to params")
        if not isinstance(
            self.interaction_geometry,
            TBGZeroFieldCompanionInteractionGeometry,
        ):
            raise TypeError("interaction_geometry must be typed companion geometry")
        geometry_spec = self.interaction_geometry.spec
        if self.interaction_geometry.N1 != self.params.N1 or self.interaction_geometry.N2 != self.params.N2:
            raise ValueError("interaction_geometry mesh does not match params")
        if geometry_spec.NG1 != self.spec.NG1 or geometry_spec.NG2 != self.spec.NG2:
            raise ValueError("interaction_geometry cutoffs do not match spec")
        if geometry_spec.margin != 1.0e-5:
            raise ValueError("interaction_geometry must use the source margin 1e-5")
        if geometry_spec.b1 != self.rlv_geometry.b1_complex or geometry_spec.b2 != self.rlv_geometry.b2_complex:
            raise ValueError("interaction_geometry reciprocal vectors do not match rlv_geometry")
        if not isinstance(self.provenance, TBGZeroFieldCompanionInteractionProvenance):
            raise TypeError("provenance must be typed companion interaction provenance")
        if not isinstance(self.array_hashes, TBGZeroFieldCompanionInteractionArrayHashes):
            raise TypeError("array_hashes must be typed companion interaction hashes")

        expected_shape = (
            self.params.N1,
            self.params.N2,
            2 * self.spec.NG1,
            2 * self.spec.NG2,
        )
        array = np.array(self.intFT_ev, dtype=np.float64, order="C", copy=True)
        if array.shape != expected_shape:
            raise ValueError(f"intFT_ev must have shape {expected_shape}, got {array.shape}")
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError("intFT_ev must contain only finite nonnegative values")
        if TBGZeroFieldCompanionInteractionArrayHashes.from_array(array) != self.array_hashes:
            raise ValueError("array_hashes do not match intFT_ev")
        array.setflags(write=False)
        object.__setattr__(self, "intFT_ev", array)

    @property
    def physical_q_scale_m_inv(self) -> float:
        return float(2.0 * kD * np.sin(self.params.theta_rad / 2.0))

    @property
    def coulomb_prefactor_ev_m(self) -> float:
        return float(echarge**2 / (2.0 * epsilon0) / echarge)

    @property
    def total_real_space_area_m2(self) -> float:
        b1 = self.rlv_geometry.b1
        b2 = self.rlv_geometry.b2
        reciprocal_cell_area = float(abs(b1[0] * b2[1] - b1[1] * b2[0]))
        return float(
            self.params.N1
            * self.params.N2
            * (4.0 * np.pi**2)
            / reciprocal_cell_area
            / self.physical_q_scale_m_inv**2
        )

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "array_hashes_fingerprint": self.array_hashes.fingerprint,
                "interaction_geometry_fingerprint": self.interaction_geometry.fingerprint,
                "params_fingerprint": self.params.fingerprint,
                "provenance_fingerprint": self.provenance.fingerprint,
                "rlv_geometry_fingerprint": self.rlv_geometry.fingerprint,
                "schema": TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA,
                "schema_version": TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA_VERSION,
                "scope": TBG_ZERO_FIELD_COMPANION_INTERACTION_SCOPE,
                "spec_fingerprint": self.spec.fingerprint,
            }
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "array_hash_semantics": TBG_ZERO_FIELD_COMPANION_INTERACTION_ARRAY_HASH_SEMANTICS,
            "array_hashes": self.array_hashes.to_metadata(),
            "coulomb_prefactor_ev_m": self.coulomb_prefactor_ev_m,
            "fingerprint": self.fingerprint,
            "interaction_geometry": self.interaction_geometry.to_metadata(),
            "params": self.params.to_metadata(),
            "params_fingerprint": self.params.fingerprint,
            "physical_q_scale_m_inv": self.physical_q_scale_m_inv,
            "provenance": self.provenance.to_metadata(),
            "rlv_geometry": self.rlv_geometry.to_metadata(),
            "rlv_geometry_fingerprint": self.rlv_geometry.fingerprint,
            "schema": TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA,
            "schema_version": TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA_VERSION,
            "scope": TBG_ZERO_FIELD_COMPANION_INTERACTION_SCOPE,
            "source_units": dict(TBG_ZERO_FIELD_COMPANION_INTERACTION_SOURCE_UNITS),
            "spec": self.spec.to_metadata(),
            "total_real_space_area_m2": self.total_real_space_area_m2,
        }


def _resolve_geometry(
    params: TBGZeroFieldCompanionSingleParticleParams,
    spec: TBGZeroFieldCompanionInteractionSpec,
    rlv_geometry: TBGZeroFieldCompanionRLVGeometry | None,
) -> tuple[TBGZeroFieldCompanionRLVGeometry, TBGZeroFieldCompanionInteractionGeometry]:
    if not isinstance(params, TBGZeroFieldCompanionSingleParticleParams):
        raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
    if not isinstance(spec, TBGZeroFieldCompanionInteractionSpec):
        raise TypeError("spec must be TBGZeroFieldCompanionInteractionSpec")
    resolved_rlv = gen_RLVs(params) if rlv_geometry is None else rlv_geometry
    if not isinstance(resolved_rlv, TBGZeroFieldCompanionRLVGeometry):
        raise TypeError("rlv_geometry must be TBGZeroFieldCompanionRLVGeometry")
    if resolved_rlv.params_fingerprint != params.fingerprint:
        raise ValueError("rlv_geometry is not bound to params")
    geometry_spec = TBGZeroFieldCompanionInteractionGeometrySpec(
        NG1=spec.NG1,
        NG2=spec.NG2,
        b1=resolved_rlv.b1_complex,
        b2=resolved_rlv.b2_complex,
        margin=1.0e-5,
    )
    interaction_geometry = build_tbg_zero_field_companion_interaction_geometry(
        geometry_spec,
        N1=params.N1,
        N2=params.N2,
    )
    return resolved_rlv, interaction_geometry


def _evaluate_gen_interaction(
    params: TBGZeroFieldCompanionSingleParticleParams,
    spec: TBGZeroFieldCompanionInteractionSpec,
    rlv_geometry: TBGZeroFieldCompanionRLVGeometry,
    interaction_geometry: TBGZeroFieldCompanionInteractionGeometry,
) -> np.ndarray:
    """Evaluate pinned ``gen_interaction`` with its source ordering and units."""

    b1 = rlv_geometry.b1
    b2 = rlv_geometry.b2
    q_scale_m_inv = kD * 2.0 * np.sin(params.theta_rad / 2.0)
    reciprocal_cell_area = np.abs(b1[0] * b2[1] - b1[1] * b2[0])
    area_m2 = (
        params.N1
        * params.N2
        * (4.0 * np.pi**2)
        / reciprocal_cell_area
        / q_scale_m_inv**2
    )
    U_ev_m = echarge**2 / (2.0 * epsilon0) / echarge

    # Keep the source's literal NG1 use in both perpendicular cutoff radii.
    R1 = spec.NG1 * np.linalg.norm(
        b1 - b2 * np.dot(b1, b2) / np.dot(b2, b2)
    )
    R2 = spec.NG1 * np.linalg.norm(
        b2 - b1 * np.dot(b1, b2) / np.dot(b1, b1)
    )
    radius = np.min((R1, R2))

    intFT_ev = np.zeros(
        (params.N1, params.N2, 2 * spec.NG1, 2 * spec.NG2),
        dtype=np.float64,
    )
    for entry in interaction_geometry.entries:
        # Preserve source operation ordering rather than reconstructing Q from
        # the geometry's complex bookkeeping value.
        Q = (
            entry.ik1 * b1 / params.N1
            + entry.ik2 * b2 / params.N2
            + entry.G1 * b1
            + entry.G2 * b2
        )
        if not np.linalg.norm(Q) < radius - 0.00001:
            continue
        target = (
            entry.ik1,
            entry.ik2,
            entry.G1 + spec.NG1,
            entry.G2 + spec.NG2,
        )
        if entry.label == (0, 0, 0, 0):
            if spec.include_q0:
                q0_factor = 1.0 if spec.gates == "dual" else 2.0
                intFT_ev[target] = q0_factor * U_ev_m * spec.dsc_m / area_m2
            continue

        modq_m_inv = np.linalg.norm(q_scale_m_inv * Q)
        if spec.gates == "dual":
            intFT_ev[target] = (
                U_ev_m * np.tanh(modq_m_inv * spec.dsc_m) / modq_m_inv / area_m2
            )
        else:
            intFT_ev[target] = (
                U_ev_m
                * (1.0 - np.exp(-2.0 * spec.dsc_m * modq_m_inv))
                / modq_m_inv
                / area_m2
            )
    return intFT_ev


def gen_interaction(
    params: TBGZeroFieldCompanionSingleParticleParams,
    spec: TBGZeroFieldCompanionInteractionSpec | None = None,
    *,
    rlv_geometry: TBGZeroFieldCompanionRLVGeometry | None = None,
) -> np.ndarray:
    """Return read-only ``intFT_ev`` from the isolated source-faithful port."""

    resolved_spec = TBGZeroFieldCompanionInteractionSpec() if spec is None else spec
    resolved_rlv, interaction_geometry = _resolve_geometry(
        params,
        resolved_spec,
        rlv_geometry,
    )
    intFT_ev = _evaluate_gen_interaction(
        params,
        resolved_spec,
        resolved_rlv,
        interaction_geometry,
    )
    intFT_ev.setflags(write=False)
    return intFT_ev


def solve_tbg_zero_field_companion_interaction(
    params: TBGZeroFieldCompanionSingleParticleParams,
    spec: TBGZeroFieldCompanionInteractionSpec | None = None,
    *,
    rlv_geometry: TBGZeroFieldCompanionRLVGeometry | None = None,
) -> TBGZeroFieldCompanionInteractionResult:
    """Build the diagnostic tensor and bind it to geometry/provenance hashes."""

    resolved_spec = TBGZeroFieldCompanionInteractionSpec() if spec is None else spec
    resolved_rlv, interaction_geometry = _resolve_geometry(
        params,
        resolved_spec,
        rlv_geometry,
    )
    intFT_ev = _evaluate_gen_interaction(
        params,
        resolved_spec,
        resolved_rlv,
        interaction_geometry,
    )
    return TBGZeroFieldCompanionInteractionResult(
        params=params,
        spec=resolved_spec,
        intFT_ev=intFT_ev,
        rlv_geometry=resolved_rlv,
        interaction_geometry=interaction_geometry,
        provenance=TBGZeroFieldCompanionInteractionProvenance(),
        array_hashes=TBGZeroFieldCompanionInteractionArrayHashes.from_array(intFT_ev),
    )


def build_tbg_zero_field_companion_interaction(
    params: TBGZeroFieldCompanionSingleParticleParams,
    spec: TBGZeroFieldCompanionInteractionSpec | None = None,
    *,
    rlv_geometry: TBGZeroFieldCompanionRLVGeometry | None = None,
) -> TBGZeroFieldCompanionInteractionResult:
    """Named builder alias for the diagnostic source-faithful interaction."""

    return solve_tbg_zero_field_companion_interaction(
        params,
        spec,
        rlv_geometry=rlv_geometry,
    )


__all__ = [
    "TBGZeroFieldCompanionInteractionArrayHashes",
    "TBGZeroFieldCompanionInteractionProvenance",
    "TBGZeroFieldCompanionInteractionResult",
    "TBGZeroFieldCompanionInteractionSpec",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_ARRAY_HASH_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_ARRAY_HASH_SEMANTICS",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_CUTOFF_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_FINITE_Q_KERNEL_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_KERNEL_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_Q0_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_FUNCTION",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_SOURCE_DSC_M",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_SUPPORT_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA_VERSION",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_SCOPE",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_SOURCE_UNITS",
    "build_tbg_zero_field_companion_interaction",
    "echarge",
    "epsilon0",
    "gen_interaction",
    "solve_tbg_zero_field_companion_interaction",
]
