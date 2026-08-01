"""Source-faithful companion BM single-particle diagnostic.

This module is a direct, typed port of the pinned ``reference/TBG-HF``
``singleParticle.py`` single-particle path.  It exists only for BM parity
checks.  Its arrays are not production BM, HF, or TDHF inputs, and this module
is intentionally not exported from :mod:`mean_field.systems.tbg.zero_field`.

The implementation keeps the companion's eV/metre units, rectangular parent
basis, K-only diagonalization, pointwise nondegenerate C2T phase choice,
spinless time-reversal construction of K', and final C2T sewing literally.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Final, Sequence

import numpy as np

from .companion_geometry import (
    TBGZeroFieldCompanionPlaneWaveSpec,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
    build_tbg_zero_field_companion_plane_wave_geometry,
)

# Literal companion constants.py values and units.
CCa: Final[float] = 1.42e-10  # m
Poisson: Final[float] = 0.16
Beta: Final[float] = 3.14
vkD: Final[float] = 9.905  # eV
kD: Final[float] = 4.0 * np.pi / (3.0 * np.sqrt(3.0) * CCa)  # m^-1
vhbar: Final[float] = vkD / kD  # eV m

TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE: Final[str] = "constants.py"
TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256: Final[str] = (
    "8d25bcccd54e41207788ff4a9e1b934a50347fabdad46ba44408fc535573ec62"
)
TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE: Final[str] = (
    "reference/TBG-HF/int_input.json"
)
TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE_SHA256: Final[str] = (
    "c143c294ad95cf94d91cfbabd0437556e5c2a342850d54484c9b47caaf84b4de"
)
TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCHEMA: Final[str] = (
    "mean_field.tbg.zero_field.companion_single_particle"
)
TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCHEMA_VERSION: Final[int] = 1
TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCOPE: Final[str] = (
    "diagnostic_BM_parity_only_not_production_HF_or_TDHF"
)
TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_CONVENTION: Final[str] = (
    "sha256_little_endian_int64_shape_then_C_order_canonical_bytes"
)
TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_SEMANTICS: Final[str] = (
    "artifact_integrity_only_not_cross_eigensolver_parity"
)
TBG_ZERO_FIELD_COMPANION_PARENT_ORDER: Final[str] = (
    "species_fast_then_g2_then_g1;species=1A,1B,2A,2B"
)
TBG_ZERO_FIELD_COMPANION_POINTWISE_GAUGE_WARNING: Final[str] = (
    "pinned_singleParticle.py_lines_177-188_apply_a_pointwise_nondegenerate_"
    "C2T_phase_choice_only;insufficient_inside_degenerate_subspaces"
)
TBG_ZERO_FIELD_COMPANION_RESIDUAL_GAUGE_AMBIGUITY: Final[str] = (
    "residual_real_sign_per_nondegenerate_state_and_U(N)_rotation_within_"
    "degenerate_subspaces"
)

_OMEGA: Final[complex] = complex(np.exp(2.0 * np.pi * 1j / 3.0))
_OMEGAC: Final[complex] = complex(np.exp(-2.0 * np.pi * 1j / 3.0))
_SX: Final[np.ndarray] = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
# The companion uses syc, the complex conjugate of the usual sigma_y.
_SYC: Final[np.ndarray] = np.asarray([[0.0, 1j], [-1j, 0.0]], dtype=np.complex128)
_BG1: Final[np.ndarray] = 0.5 * np.asarray([3.0, np.sqrt(3.0)], dtype=np.float64)
_BG2: Final[np.ndarray] = 0.5 * np.asarray([3.0, -np.sqrt(3.0)], dtype=np.float64)


def _strict_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer (bool is not accepted), got {value!r}")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return resolved


def _strict_index(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer (bool is not accepted), got {value!r}")
    return int(value)


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar (bool is not accepted), got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite, got {value!r}")
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


def _canonical_array_sha256(values: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _vector_to_complex(vector: np.ndarray) -> complex:
    resolved = np.asarray(vector, dtype=np.float64)
    if resolved.shape != (2,):
        raise ValueError(f"Expected a Cartesian two-vector, got shape {resolved.shape}")
    return complex(float(resolved[0]), float(resolved[1]))


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionSingleParticleParams:
    """Strict companion input fields in the original units.

    Angles are in degrees, tunnelling amplitudes are in eV, and ``strain`` is
    dimensionless.  Defaults are the pinned companion ``int_input.json``.
    """

    N1: int = 8
    N2: int = 8
    Ng1: int = 4
    Ng2: int = 4
    n_active: int = 1
    theta_deg: float = 1.08
    wAA_ev: float = 0.07
    wAB_ev: float = 0.11
    strain: float = 0.003
    strain_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        for name in ("N1", "N2", "Ng1", "Ng2", "n_active"):
            object.__setattr__(self, name, _strict_positive_int(getattr(self, name), name=name))
        for name in ("theta_deg", "wAA_ev", "wAB_ev", "strain", "strain_angle_deg"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name=name))
        if np.sin(self.theta_rad / 2.0) == 0.0:
            raise ValueError("theta_deg makes the companion 2*sin(theta/2) denominator zero")

    @property
    def theta_rad(self) -> float:
        return float(np.pi / 180.0 * self.theta_deg)

    @property
    def strain_angle_rad(self) -> float:
        return float(np.pi / 180.0 * self.strain_angle_deg)

    @property
    def active_band_count(self) -> int:
        return 2 * self.n_active

    @property
    def parent_dimension(self) -> int:
        return 16 * self.Ng1 * self.Ng2

    def to_companion_input(self) -> dict[str, int | float]:
        """Return only the source dictionary keys used by this stage."""

        return {
            "N1": self.N1,
            "N2": self.N2,
            "Ng1": self.Ng1,
            "Ng2": self.Ng2,
            "n_active": self.n_active,
            "theta": self.theta_deg,
            "wAA": self.wAA_ev,
            "wAB": self.wAB_ev,
            "strain": self.strain,
            "varphi": self.strain_angle_deg,
        }

    def _payload(self) -> dict[str, object]:
        return {
            "constants": {
                "Beta": Beta,
                "CCa_m": CCa,
                "Poisson": Poisson,
                "kD_m_inv": kD,
                "vhbar_ev_m": vhbar,
                "vkD_ev": vkD,
            },
            "default_input_source": TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE,
            "default_input_source_sha256": (
                TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE_SHA256
            ),
            "input": self.to_companion_input(),
            "parent_dimension": self.parent_dimension,
            "parent_order": TBG_ZERO_FIELD_COMPANION_PARENT_ORDER,
            "reference_commit": TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
            "reference_repository": TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
            "reference_source": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
            "reference_source_sha256": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
            "schema": TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCHEMA,
            "schema_version": TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCHEMA_VERSION,
            "scope": TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCOPE,
        }

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self._payload())

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionRLVGeometry:
    """Literal ``gen_RLVs`` output plus the source ktheta/vhbar units."""

    params_fingerprint: str
    M1: np.ndarray
    M2: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    Etens1: np.ndarray
    Etens2: np.ndarray
    ktheta_m_inv: float
    vhbar_ev_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "params_fingerprint",
            _validate_sha256(self.params_fingerprint, name="params_fingerprint"),
        )
        for name, shape in (
            ("M1", (2, 2)),
            ("M2", (2, 2)),
            ("b1", (2,)),
            ("b2", (2,)),
            ("Etens1", (2, 2)),
            ("Etens2", (2, 2)),
        ):
            array = np.array(getattr(self, name), dtype=np.float64, order="C", copy=True)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        resolved_ktheta = _finite_real(self.ktheta_m_inv, name="ktheta_m_inv")
        resolved_vhbar = _finite_real(self.vhbar_ev_m, name="vhbar_ev_m")
        if resolved_ktheta == 0.0 or resolved_vhbar == 0.0:
            raise ValueError("ktheta_m_inv and vhbar_ev_m must be nonzero")
        object.__setattr__(self, "ktheta_m_inv", resolved_ktheta)
        object.__setattr__(self, "vhbar_ev_m", resolved_vhbar)

    @property
    def b1_complex(self) -> complex:
        return _vector_to_complex(self.b1)

    @property
    def b2_complex(self) -> complex:
        return _vector_to_complex(self.b2)

    def _payload(self) -> dict[str, object]:
        return {
            "Etens1_sha256": _canonical_array_sha256(self.Etens1, dtype="<f8"),
            "Etens2_sha256": _canonical_array_sha256(self.Etens2, dtype="<f8"),
            "M1_sha256": _canonical_array_sha256(self.M1, dtype="<f8"),
            "M2_sha256": _canonical_array_sha256(self.M2, dtype="<f8"),
            "b1": _complex_pair(self.b1_complex),
            "b1_sha256": _canonical_array_sha256(self.b1, dtype="<f8"),
            "b2": _complex_pair(self.b2_complex),
            "b2_sha256": _canonical_array_sha256(self.b2, dtype="<f8"),
            "ktheta_m_inv": self.ktheta_m_inv,
            "params_fingerprint": self.params_fingerprint,
            "reference_function": "gen_RLVs",
            "reference_lines": "20-49",
            "vhbar_ev_m": self.vhbar_ev_m,
        }

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self._payload())

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionGeometryFingerprints:
    """Immutable reciprocal and pointwise circular-geometry identities."""

    rlv_geometry: str
    plane_wave_spec: str
    K_point_geometry: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rlv_geometry",
            _validate_sha256(self.rlv_geometry, name="rlv_geometry"),
        )
        object.__setattr__(
            self,
            "plane_wave_spec",
            _validate_sha256(self.plane_wave_spec, name="plane_wave_spec"),
        )
        rows = tuple(tuple(row) for row in self.K_point_geometry)
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("K_point_geometry must be a nonempty rectangular tuple")
        for ik1, row in enumerate(rows):
            for ik2, digest in enumerate(row):
                _validate_sha256(digest, name=f"K_point_geometry[{ik1}][{ik2}]")
        object.__setattr__(self, "K_point_geometry", rows)

    @property
    def mesh_shape(self) -> tuple[int, int]:
        return (len(self.K_point_geometry), len(self.K_point_geometry[0]))

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "K_point_geometry": [list(row) for row in self.K_point_geometry],
                "mesh_shape": list(self.mesh_shape),
                "plane_wave_spec": self.plane_wave_spec,
                "rlv_geometry": self.rlv_geometry,
            }
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "K_point_geometry": [list(row) for row in self.K_point_geometry],
            "fingerprint": self.fingerprint,
            "mesh_shape": list(self.mesh_shape),
            "plane_wave_spec": self.plane_wave_spec,
            "rlv_geometry": self.rlv_geometry,
        }


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionSingleParticleArrayHashes:
    """Canonical artifact-integrity hashes, not cross-eigensolver parity data."""

    coeff: str
    sp_energy_ev: str
    U_C2T: str
    convention: str = TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_CONVENTION
    semantics: str = TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_SEMANTICS

    def __post_init__(self) -> None:
        for name in ("coeff", "sp_energy_ev", "U_C2T"):
            object.__setattr__(self, name, _validate_sha256(getattr(self, name), name=name))
        if self.convention != TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_CONVENTION:
            raise ValueError("Unsupported companion array-hash convention")
        if self.semantics != TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_SEMANTICS:
            raise ValueError("Unsupported companion array-hash semantics")

    @classmethod
    def from_arrays(
        cls,
        *,
        coeff: np.ndarray,
        sp_energy_ev: np.ndarray,
        U_C2T: np.ndarray,
    ) -> TBGZeroFieldCompanionSingleParticleArrayHashes:
        return cls(
            coeff=_canonical_array_sha256(coeff, dtype="<c16"),
            sp_energy_ev=_canonical_array_sha256(sp_energy_ev, dtype="<f8"),
            U_C2T=_canonical_array_sha256(U_C2T, dtype="<c16"),
        )

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "U_C2T": self.U_C2T,
                "coeff": self.coeff,
                "convention": self.convention,
                "semantics": self.semantics,
                "sp_energy_ev": self.sp_energy_ev,
            }
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionSingleParticleProvenance:
    """Pinned source identity and deliberately narrow scientific scope."""

    reference_repository: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY
    reference_commit: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT
    reference_source: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE
    reference_source_sha256: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256
    constants_source: str = TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE
    constants_source_sha256: str = TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256
    default_input_source: str = TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE
    default_input_source_sha256: str = TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE_SHA256
    rlv_reference_lines: str = "20-49"
    hamiltonian_reference_lines: str = "51-111"
    coefficient_reference_lines: str = "113-202"
    pointwise_C2T_gauge_reference_lines: str = "177-188"
    Kprime_time_reversal_reference_lines: str = "190-200"
    final_C2T_sewing_reference_lines: str = "265-279"
    pointwise_gauge_warning: str = TBG_ZERO_FIELD_COMPANION_POINTWISE_GAUGE_WARNING
    residual_gauge_ambiguity: str = TBG_ZERO_FIELD_COMPANION_RESIDUAL_GAUGE_AMBIGUITY
    array_hash_semantics: str = TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_SEMANTICS
    scientific_scope: str = TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCOPE

    def __post_init__(self) -> None:
        expected = {
            "reference_repository": TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
            "reference_commit": TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
            "reference_source": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
            "reference_source_sha256": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
            "constants_source": TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE,
            "constants_source_sha256": TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256,
            "default_input_source": TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE,
            "default_input_source_sha256": (
                TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE_SHA256
            ),
            "rlv_reference_lines": "20-49",
            "hamiltonian_reference_lines": "51-111",
            "coefficient_reference_lines": "113-202",
            "pointwise_C2T_gauge_reference_lines": "177-188",
            "Kprime_time_reversal_reference_lines": "190-200",
            "final_C2T_sewing_reference_lines": "265-279",
            "pointwise_gauge_warning": TBG_ZERO_FIELD_COMPANION_POINTWISE_GAUGE_WARNING,
            "residual_gauge_ambiguity": TBG_ZERO_FIELD_COMPANION_RESIDUAL_GAUGE_AMBIGUITY,
            "array_hash_semantics": TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_SEMANTICS,
            "scientific_scope": TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCOPE,
        }
        for name, pinned in expected.items():
            if getattr(self, name) != pinned:
                raise ValueError(f"{name} differs from the pinned companion provenance")

    def to_metadata(self) -> dict[str, str]:
        return {
            "Kprime_time_reversal_reference_lines": self.Kprime_time_reversal_reference_lines,
            "coefficient_reference_lines": self.coefficient_reference_lines,
            "array_hash_semantics": self.array_hash_semantics,
            "constants_source": self.constants_source,
            "constants_source_sha256": self.constants_source_sha256,
            "default_input_source": self.default_input_source,
            "default_input_source_sha256": self.default_input_source_sha256,
            "final_C2T_sewing_reference_lines": self.final_C2T_sewing_reference_lines,
            "hamiltonian_reference_lines": self.hamiltonian_reference_lines,
            "pointwise_C2T_gauge_reference_lines": self.pointwise_C2T_gauge_reference_lines,
            "pointwise_gauge_warning": self.pointwise_gauge_warning,
            "reference_commit": self.reference_commit,
            "reference_repository": self.reference_repository,
            "reference_source": self.reference_source,
            "reference_source_sha256": self.reference_source_sha256,
            "residual_gauge_ambiguity": self.residual_gauge_ambiguity,
            "rlv_reference_lines": self.rlv_reference_lines,
            "scientific_scope": self.scientific_scope,
        }

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self.to_metadata())


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionSingleParticleResult:
    """Read-only-array result for diagnostic companion BM parity only."""

    params: TBGZeroFieldCompanionSingleParticleParams
    coeff: np.ndarray
    sp_energy_ev: np.ndarray
    U_C2T: np.ndarray
    rlv_geometry: TBGZeroFieldCompanionRLVGeometry
    geometry_fingerprints: TBGZeroFieldCompanionGeometryFingerprints
    provenance: TBGZeroFieldCompanionSingleParticleProvenance
    array_hashes: TBGZeroFieldCompanionSingleParticleArrayHashes

    def __post_init__(self) -> None:
        if not isinstance(self.params, TBGZeroFieldCompanionSingleParticleParams):
            raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
        if not isinstance(self.rlv_geometry, TBGZeroFieldCompanionRLVGeometry):
            raise TypeError("rlv_geometry must be TBGZeroFieldCompanionRLVGeometry")
        if self.rlv_geometry.params_fingerprint != self.params.fingerprint:
            raise ValueError("rlv_geometry is not bound to params")
        if not isinstance(self.geometry_fingerprints, TBGZeroFieldCompanionGeometryFingerprints):
            raise TypeError("geometry_fingerprints must be typed companion geometry fingerprints")
        if self.geometry_fingerprints.mesh_shape != (self.params.N1, self.params.N2):
            raise ValueError("geometry_fingerprints mesh shape does not match params")
        if self.geometry_fingerprints.rlv_geometry != self.rlv_geometry.fingerprint:
            raise ValueError("geometry_fingerprints do not match rlv_geometry")
        if not isinstance(self.provenance, TBGZeroFieldCompanionSingleParticleProvenance):
            raise TypeError("provenance must be typed companion single-particle provenance")
        if not isinstance(self.array_hashes, TBGZeroFieldCompanionSingleParticleArrayHashes):
            raise TypeError("array_hashes must be typed companion array hashes")

        nb = self.params.active_band_count
        expected_shapes = {
            "coeff": (
                self.params.N1,
                self.params.N2,
                2 * self.params.Ng1,
                2 * self.params.Ng2,
                2,
                nb,
                4,
            ),
            "sp_energy_ev": (self.params.N1, self.params.N2, 2, nb),
            "U_C2T": (self.params.N1, self.params.N2, 2, nb, nb),
        }
        arrays: dict[str, np.ndarray] = {}
        for name, dtype in (("coeff", np.complex128), ("sp_energy_ev", np.float64), ("U_C2T", np.complex128)):
            array = np.array(getattr(self, name), dtype=dtype, order="C", copy=True)
            if array.shape != expected_shapes[name]:
                raise ValueError(f"{name} must have shape {expected_shapes[name]}, got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            arrays[name] = array

        actual_hashes = TBGZeroFieldCompanionSingleParticleArrayHashes.from_arrays(
            coeff=arrays["coeff"],
            sp_energy_ev=arrays["sp_energy_ev"],
            U_C2T=arrays["U_C2T"],
        )
        if actual_hashes != self.array_hashes:
            raise ValueError("array_hashes do not match the returned arrays")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "array_hash_semantics": TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_SEMANTICS,
                "array_hashes_fingerprint": self.array_hashes.fingerprint,
                "geometry_fingerprints": self.geometry_fingerprints.fingerprint,
                "params_fingerprint": self.params.fingerprint,
                "pointwise_gauge_warning": TBG_ZERO_FIELD_COMPANION_POINTWISE_GAUGE_WARNING,
                "provenance_fingerprint": self.provenance.fingerprint,
                "residual_gauge_ambiguity": (
                    TBG_ZERO_FIELD_COMPANION_RESIDUAL_GAUGE_AMBIGUITY
                ),
                "schema": TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCHEMA,
                "schema_version": TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCHEMA_VERSION,
                "scope": TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCOPE,
            }
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "array_hashes": {
                "U_C2T": self.array_hashes.U_C2T,
                "coeff": self.array_hashes.coeff,
                "convention": self.array_hashes.convention,
                "fingerprint": self.array_hashes.fingerprint,
                "semantics": self.array_hashes.semantics,
                "sp_energy_ev": self.array_hashes.sp_energy_ev,
            },
            "array_hash_semantics": TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_SEMANTICS,
            "fingerprint": self.fingerprint,
            "geometry_fingerprints": self.geometry_fingerprints.to_metadata(),
            "params": self.params.to_metadata(),
            "pointwise_gauge_warning": TBG_ZERO_FIELD_COMPANION_POINTWISE_GAUGE_WARNING,
            "provenance": self.provenance.to_metadata(),
            "residual_gauge_ambiguity": TBG_ZERO_FIELD_COMPANION_RESIDUAL_GAUGE_AMBIGUITY,
            "schema": TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCHEMA,
            "schema_version": TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCHEMA_VERSION,
            "scope": TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCOPE,
        }


def _rot(alpha: float) -> np.ndarray:
    return np.asarray(
        [[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]],
        dtype=np.float64,
    )


def gen_RLVs(
    params: TBGZeroFieldCompanionSingleParticleParams,
) -> TBGZeroFieldCompanionRLVGeometry:
    """Port companion ``gen_RLVs`` exactly, including its scaled b vectors."""

    if not isinstance(params, TBGZeroFieldCompanionSingleParticleParams):
        raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
    theta = np.pi / 180.0 * params.theta_deg
    varphi = np.pi / 180.0 * params.strain_angle_deg
    strain = params.strain

    Etens1 = _rot(-varphi) @ np.diag([-strain / 2.0, Poisson * strain / 2.0]) @ _rot(varphi)
    Etens2 = -Etens1
    M1 = _rot(theta / 2.0) + Etens1
    M2 = _rot(-theta / 2.0) + Etens2

    denominator = 2.0 * np.sin(theta / 2.0)
    try:
        inverse_difference = np.linalg.inv(M1) - np.linalg.inv(M2)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Companion twist/strain matrix is singular") from exc
    b1 = np.dot(inverse_difference, _BG2 - _BG1) / denominator
    b2 = np.dot(inverse_difference, _BG1) / denominator
    ktheta = 2.0 * kD * np.sin(theta / 2.0)

    return TBGZeroFieldCompanionRLVGeometry(
        params_fingerprint=params.fingerprint,
        M1=M1,
        M2=M2,
        b1=b1,
        b2=b2,
        Etens1=Etens1,
        Etens2=Etens2,
        ktheta_m_inv=float(ktheta),
        vhbar_ev_m=float(vhbar),
    )


def _resolve_ik(
    params: TBGZeroFieldCompanionSingleParticleParams,
    ik: Sequence[int],
) -> tuple[int, int]:
    if isinstance(ik, (str, bytes)) or len(ik) != 2:
        raise TypeError("ik must be a length-two integer sequence")
    ik1 = _strict_index(ik[0], name="ik1")
    ik2 = _strict_index(ik[1], name="ik2")
    if not 0 <= ik1 < params.N1:
        raise ValueError(f"ik1={ik1} must satisfy 0 <= ik1 < N1={params.N1}")
    if not 0 <= ik2 < params.N2:
        raise ValueError(f"ik2={ik2} must satisfy 0 <= ik2 < N2={params.N2}")
    return ik1, ik2


def gen_moire_hamiltonian(
    params: TBGZeroFieldCompanionSingleParticleParams,
    ik: Sequence[int],
    *,
    rlv_geometry: TBGZeroFieldCompanionRLVGeometry | None = None,
) -> np.ndarray:
    """Return the literal companion K-valley rectangular parent in eV.

    The dimension is ``16*Ng1*Ng2`` (256 for the companion's Ng1=Ng2=4),
    ordered species-fast, then g2, then g1.  T2/T3 are zero-filled at the
    rectangular parent boundary before the Hermitian conjugate is added.
    """

    if not isinstance(params, TBGZeroFieldCompanionSingleParticleParams):
        raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
    ik1, ik2 = _resolve_ik(params, ik)
    geometry = gen_RLVs(params) if rlv_geometry is None else rlv_geometry
    if not isinstance(geometry, TBGZeroFieldCompanionRLVGeometry):
        raise TypeError("rlv_geometry must be TBGZeroFieldCompanionRLVGeometry")
    if geometry.params_fingerprint != params.fingerprint:
        raise ValueError("rlv_geometry is not bound to params")

    b1 = geometry.b1
    b2 = geometry.b2
    b1s = b1 / params.N1
    b2s = b2 / params.N2
    k = ik1 * b1s + ik2 * b2s
    T1 = np.asarray(
        [[params.wAA_ev, params.wAB_ev], [params.wAB_ev, params.wAA_ev]],
        dtype=np.complex128,
    )
    T2 = np.asarray(
        [
            [params.wAA_ev, params.wAB_ev * _OMEGA],
            [params.wAB_ev * _OMEGAC, params.wAA_ev],
        ],
        dtype=np.complex128,
    )
    T3 = np.asarray(
        [
            [params.wAA_ev, params.wAB_ev * _OMEGAC],
            [params.wAB_ev * _OMEGA, params.wAA_ev],
        ],
        dtype=np.complex128,
    )

    Hkin = np.zeros((params.parent_dimension, params.parent_dimension), dtype=np.complex128)
    Hhop = np.zeros_like(Hkin)
    for g1 in range(-params.Ng1, params.Ng1):
        for g2 in range(-params.Ng2, params.Ng2):
            indexg = 4 * ((g1 + params.Ng1) * (2 * params.Ng2) + (g2 + params.Ng2))

            d1 = -b1 / 3.0 + b2 / 3.0
            A1 = Beta / 2.0 / CCa * np.asarray(
                [
                    geometry.Etens1[0, 0] - geometry.Etens1[1, 1],
                    -2.0 * geometry.Etens1[0, 1],
                ]
            )
            q1 = geometry.ktheta_m_inv * (k + g1 * b1 + g2 * b2 - d1) - A1
            q1 = np.dot(geometry.M1, q1)
            t1 = geometry.vhbar_ev_m * (q1[0] * _SX + q1[1] * _SYC)

            d2 = -2.0 * b1 / 3.0 - b2 / 3.0
            A2 = Beta / 2.0 / CCa * np.asarray(
                [
                    geometry.Etens2[0, 0] - geometry.Etens2[1, 1],
                    -2.0 * geometry.Etens2[0, 1],
                ]
            )
            q2 = geometry.ktheta_m_inv * (k + g1 * b1 + g2 * b2 - d2) - A2
            q2 = np.dot(geometry.M2, q2)
            t2 = geometry.vhbar_ev_m * (q2[0] * _SX + q2[1] * _SYC)

            Hkin[indexg : indexg + 2, indexg : indexg + 2] = t1
            Hkin[indexg + 2 : indexg + 4, indexg + 2 : indexg + 4] = t2
            Hhop[indexg : indexg + 2, indexg + 2 : indexg + 4] = T1
            if g1 < params.Ng1 - 1 and g2 < params.Ng2 - 1:
                indexp = indexg + 4 + 8 * params.Ng2
                Hhop[indexp : indexp + 2, indexg + 2 : indexg + 4] = T2
            if g2 < params.Ng2 - 1:
                indexp = indexg + 4
                Hhop[indexp : indexp + 2, indexg + 2 : indexg + 4] = T3

    return Hkin + Hhop + np.conjugate(np.transpose(Hhop))


def _apply_pointwise_C2T_gauge(
    coeff: np.ndarray,
    params: TBGZeroFieldCompanionSingleParticleParams,
) -> None:
    """Apply literal companion lines 177-188 to the directly solved K valley."""

    nb = params.active_band_count
    tau = 0
    coeff_K = np.reshape(
        coeff[:, :, :, :, tau, :, :],
        (params.N1, params.N2, 2 * params.Ng1, 2 * params.Ng2, nb, 2, 2),
    )
    coeff_C2T = np.zeros_like(coeff_K, dtype=complex)
    for sub in range(2):
        coeff_C2T[..., sub] = np.conj(coeff_K[..., 1 - sub])
    U_C2T = np.einsum(
        "kKgGals,kKgGbls->kKab",
        np.conj(coeff_K),
        coeff_C2T,
        optimize=True,
    )

    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            for a in range(nb):
                sum_phase = -np.angle(U_C2T[ik1, ik2, a, a])
                coeff[ik1, ik2, :, :, tau, a, :] = (
                    np.exp(-1j * 0.5 * sum_phase)
                    * coeff[ik1, ik2, :, :, tau, a, :].copy()
                )


def _fill_Kprime_by_time_reversal(
    coeff: np.ndarray,
    sp_energy_ev: np.ndarray,
    params: TBGZeroFieldCompanionSingleParticleParams,
) -> None:
    """Apply literal lines 190-200, including Python floor carries and zeros."""

    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            sp_energy_ev[ik1, ik2, 1, :] = sp_energy_ev[
                (-ik1) % params.N1,
                (-ik2) % params.N2,
                0,
                :,
            ]
            for g1 in range(-params.Ng1, params.Ng1):
                for g2 in range(-params.Ng2, params.Ng2):
                    gp1 = -g1 + (-ik1) // params.N1
                    gp2 = -g2 + (-ik2) // params.N2
                    if (
                        gp1 >= -params.Ng1
                        and gp2 >= -params.Ng2
                        and gp1 < params.Ng1
                        and gp2 < params.Ng2
                    ):
                        coeff[
                            ik1,
                            ik2,
                            g1 + params.Ng1,
                            g2 + params.Ng2,
                            1,
                            :,
                            :,
                        ] = np.conj(
                            coeff[
                                (-ik1) % params.N1,
                                (-ik2) % params.N2,
                                gp1 + params.Ng1,
                                gp2 + params.Ng2,
                                0,
                                :,
                                :,
                            ]
                        )


def C2T_symmetry(
    params: TBGZeroFieldCompanionSingleParticleParams,
    coeff: np.ndarray,
) -> np.ndarray:
    """Return literal companion lines 265-279 final C2T sewing matrices."""

    if not isinstance(params, TBGZeroFieldCompanionSingleParticleParams):
        raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
    nb = params.active_band_count
    expected_shape = (
        params.N1,
        params.N2,
        2 * params.Ng1,
        2 * params.Ng2,
        2,
        nb,
        4,
    )
    resolved = np.asarray(coeff, dtype=np.complex128)
    if resolved.shape != expected_shape:
        raise ValueError(f"coeff must have shape {expected_shape}, got {resolved.shape}")
    reshaped = np.reshape(
        resolved,
        (params.N1, params.N2, 2 * params.Ng1, 2 * params.Ng2, 2, nb, 2, 2),
    )
    coeff_C2T = np.zeros_like(reshaped, dtype=complex)
    for sub in range(2):
        coeff_C2T[..., sub] = np.conj(reshaped[..., 1 - sub])
    return np.einsum(
        "kKgGtals,kKgGtbls->kKtab",
        np.conj(reshaped),
        coeff_C2T,
        optimize=True,
    )


def solve_tbg_zero_field_companion_single_particle(
    params: TBGZeroFieldCompanionSingleParticleParams,
) -> TBGZeroFieldCompanionSingleParticleResult:
    """Solve the pinned companion single-particle chain without HF integration."""

    if not isinstance(params, TBGZeroFieldCompanionSingleParticleParams):
        raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
    rlv_geometry = gen_RLVs(params)
    plane_wave_spec = TBGZeroFieldCompanionPlaneWaveSpec(
        Ng1=params.Ng1,
        Ng2=params.Ng2,
        b1=rlv_geometry.b1_complex,
        b2=rlv_geometry.b2_complex,
    )
    nb = params.active_band_count
    coeff = np.zeros(
        (
            params.N1,
            params.N2,
            2 * params.Ng1,
            2 * params.Ng2,
            2,
            nb,
            4,
        ),
        dtype=np.complex128,
    )
    sp_energy_ev = np.zeros((params.N1, params.N2, 2, nb), dtype=np.float64)
    point_geometry_fingerprints: list[list[str]] = [
        ["" for _ in range(params.N2)] for _ in range(params.N1)
    ]

    tau = 0
    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            ham = gen_moire_hamiltonian(
                params,
                (ik1, ik2),
                rlv_geometry=rlv_geometry,
            )
            point_geometry = build_tbg_zero_field_companion_plane_wave_geometry(
                plane_wave_spec,
                N1=params.N1,
                N2=params.N2,
                ik1=ik1,
                ik2=ik2,
                stau=1,
            )
            point_geometry_fingerprints[ik1][ik2] = point_geometry.fingerprint
            sub_index = np.asarray(point_geometry.sub_index, dtype=int)
            cbandnum = np.size(sub_index) // 2
            rank_start = cbandnum - params.n_active
            rank_stop = cbandnum + params.n_active
            if rank_start < 0 or rank_stop > np.size(sub_index):
                raise ValueError(
                    "n_active does not fit the central contiguous rank window at "
                    f"K mesh point ({ik1},{ik2}): subspace dimension={np.size(sub_index)}"
                )
            sub_ham = ham[sub_index][:, sub_index]
            eigvals, eigvecs = np.linalg.eigh(sub_ham)
            sp_energy_ev[ik1, ik2, tau, :] = eigvals[rank_start:rank_stop]
            for index, value in enumerate(sub_index):
                species = int(value % 4)
                g2 = int((value // 4) % (2 * params.Ng2) - params.Ng2)
                g1 = int(value // (4 * 2 * params.Ng2) - params.Ng1)
                coeff[
                    ik1,
                    ik2,
                    g1 + params.Ng1,
                    g2 + params.Ng2,
                    tau,
                    :,
                    species,
                ] = eigvecs[index, rank_start:rank_stop]

    _apply_pointwise_C2T_gauge(coeff, params)
    _fill_Kprime_by_time_reversal(coeff, sp_energy_ev, params)
    U_C2T = C2T_symmetry(params, coeff)

    geometry_fingerprints = TBGZeroFieldCompanionGeometryFingerprints(
        rlv_geometry=rlv_geometry.fingerprint,
        plane_wave_spec=plane_wave_spec.fingerprint,
        K_point_geometry=tuple(tuple(row) for row in point_geometry_fingerprints),
    )
    provenance = TBGZeroFieldCompanionSingleParticleProvenance()
    array_hashes = TBGZeroFieldCompanionSingleParticleArrayHashes.from_arrays(
        coeff=coeff,
        sp_energy_ev=sp_energy_ev,
        U_C2T=U_C2T,
    )
    return TBGZeroFieldCompanionSingleParticleResult(
        params=params,
        coeff=coeff,
        sp_energy_ev=sp_energy_ev,
        U_C2T=U_C2T,
        rlv_geometry=rlv_geometry,
        geometry_fingerprints=geometry_fingerprints,
        provenance=provenance,
        array_hashes=array_hashes,
    )


def build_tbg_zero_field_companion_single_particle(
    params: TBGZeroFieldCompanionSingleParticleParams,
) -> TBGZeroFieldCompanionSingleParticleResult:
    """Named builder alias for the diagnostic source-faithful solve."""

    return solve_tbg_zero_field_companion_single_particle(params)


__all__ = [
    "Beta",
    "CCa",
    "C2T_symmetry",
    "Poisson",
    "TBGZeroFieldCompanionGeometryFingerprints",
    "TBGZeroFieldCompanionRLVGeometry",
    "TBGZeroFieldCompanionSingleParticleArrayHashes",
    "TBGZeroFieldCompanionSingleParticleParams",
    "TBGZeroFieldCompanionSingleParticleProvenance",
    "TBGZeroFieldCompanionSingleParticleResult",
    "TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_ARRAY_HASH_SEMANTICS",
    "TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE",
    "TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256",
    "TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE",
    "TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE_SHA256",
    "TBG_ZERO_FIELD_COMPANION_PARENT_ORDER",
    "TBG_ZERO_FIELD_COMPANION_POINTWISE_GAUGE_WARNING",
    "TBG_ZERO_FIELD_COMPANION_RESIDUAL_GAUGE_AMBIGUITY",
    "TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCHEMA",
    "TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCHEMA_VERSION",
    "TBG_ZERO_FIELD_COMPANION_SINGLE_PARTICLE_SCOPE",
    "build_tbg_zero_field_companion_single_particle",
    "gen_RLVs",
    "gen_moire_hamiltonian",
    "kD",
    "solve_tbg_zero_field_companion_single_particle",
    "vhbar",
    "vkD",
]
