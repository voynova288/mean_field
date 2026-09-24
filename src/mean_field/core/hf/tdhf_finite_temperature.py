"""Finite-temperature collisionless TDHF transition coordinates.

The finite-temperature inventory is energy oriented. For each endpoint record,
``particle`` is the higher-energy full-HF quasiparticle and ``hole`` is the
lower-energy quasiparticle. These names label ``X_ph`` and do not assert the
zero-temperature unoccupied/occupied classification.

For ``d_i = f(e_h) - f(e_p) > 0`` and physical density amplitudes ``z_i``,

    (L_z z)_i = gap_i z_i + d_i sum_j K_ij z_j.

The manifest fixes that each derivative block ``K`` contains its source-column
quadrature weight exactly once. In the coordinate
``x_i = sqrt(w_i / d_i) z_i``,

    A = diag(gap) + diag(sqrt(w*d)) K_A diag(sqrt(d/w)),
    B =             diag(sqrt(w*d)) K_B diag(sqrt(d/w)).

No sign, half factor, extra quadrature factor, symmetrization, or Hermitization
is added. ``sqrt(d)`` is evaluated independently as ``exp(log(d)/2)`` so it
can remain representable after the stored float64 ``d`` has underflowed.

This module certifies only the algebraic assembly above. It does **not**
validate that a production ``F0/P0`` reference is stationary or that the
finite-temperature occupations and interaction derivative obey the required
thermal closure. A production system adapter must establish and bind those
physical conditions before treating a converted typed sector as authoritative.
No stationary-reference loader is provided here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Literal

import numpy as np

from .tdhf_signed import (
    TDHFGenericSignedQ,
    TDHFGenericSignedQSector,
    TDHFNambuSewing,
    TDHFSelfConjugateQ,
    TDHFSelfConjugateQSector,
    TDHFSignedQBlocks,
    TDHFStaticHessianAuthority,
    fingerprint_tdhf_pairs,
)

_INTERACTION_BLOCK_CONVENTION = (
    "physical_density_derivative_source_weight_once_v1"
)
_MANIFEST_FINGERPRINT_VERSION = "finite_temperature_tdhf_manifest_v1"
_TRANSITION_FINGERPRINT_VERSION = "finite_temperature_tdhf_transition_weights_v3"
_ASSEMBLY_FINGERPRINT_VERSION = "finite_temperature_tdhf_assembly_v3"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


_FACTORY_ONLY_ERROR = "must be created by its public factory"

def _reject_public_record_init(record: str) -> None:
    raise TypeError(f"{record} {_FACTORY_ONLY_ERROR}")


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values), initial=0.0))


def _strict_real_scalar(value: Any, *, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if np.iscomplexobj(raw) and bool(np.imag(raw) != 0.0):
        raise ValueError(f"{name} must be real; imaginary parts are forbidden")
    result = float(np.real(raw))
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_real_array(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw) and bool(np.any(np.imag(raw) != 0.0)):
        raise ValueError(f"{name} must be real; imaginary parts are forbidden")
    result = np.asarray(np.real(raw), dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly_float_vector(values: Sequence[float]) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _nonnegative_tolerance(value: Any, *, name: str) -> float:
    result = _strict_real_scalar(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _fingerprint64(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _freeze_metadata(value: Any, *, name: str) -> Any:
    """Normalize deeply immutable JSON-like tuple/scalar metadata."""

    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} floating metadata must be finite")
        return result
    if isinstance(value, tuple):
        return tuple(
            _freeze_metadata(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{name} must be deeply immutable JSON-like tuple/scalar metadata; "
        "mutable containers and arbitrary objects are rejected"
    )


def _readonly_complex_matrix(
    value: Any,
    *,
    name: str,
    shape: tuple[int, int],
) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a rectangular numeric matrix") from error
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    if not (
        np.all(np.isfinite(matrix.real)) and np.all(np.isfinite(matrix.imag))
    ):
        raise ValueError(f"{name} must be finite")
    result = np.array(matrix, dtype=np.complex128, copy=True, order="C")
    result.setflags(write=False)
    return result


def _fermi_occupations(dimensionless_energies: np.ndarray) -> np.ndarray:
    """Evaluate ``1/(1+exp(x))`` without overflowing ``exp(x)``."""

    return np.exp(-np.logaddexp(0.0, dimensionless_energies))


def _log_fermi_difference(
    dimensionless_hole_energy: float,
    dimensionless_particle_energy: float,
    log_dimensionless_gap: float,
) -> float:
    """Return ``log(f_h-f_p)`` without subtracting close occupations."""

    log_f_h = -float(np.logaddexp(0.0, dimensionless_hole_energy))
    log_one_minus_f_p = -float(
        np.logaddexp(0.0, -dimensionless_particle_energy)
    )
    if log_dimensionless_gap < -36.0:
        log_gap_factor = log_dimensionless_gap
    elif log_dimensionless_gap > 36.0:
        log_gap_factor = 0.0
    else:
        dimensionless_gap = float(np.exp(log_dimensionless_gap))
        log_gap_factor = float(np.log(-np.expm1(-dimensionless_gap)))
    return log_f_h + log_one_minus_f_p + log_gap_factor


def _update_float64(digest: Any, values: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _update_complex128(digest: Any, values: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype="<c16"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())


@dataclass(frozen=True)
class EnergyOrientedTransitionEndpoints:
    """One ordered higher-energy/lower-energy full-HF transition label."""

    particle_index: int
    hole_index: int
    particle_momentum: Any | None = None
    hole_momentum: Any | None = None
    particle_flavor: Any | None = None
    hole_flavor: Any | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("particle_index", self.particle_index),
            ("hole_index", self.hole_index),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer full-HF eigenstate index")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(value))
        if self.particle_index == self.hole_index:
            raise ValueError("energy-oriented transition endpoints must be distinct")
        for name in (
            "particle_momentum",
            "hole_momentum",
            "particle_flavor",
            "hole_flavor",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_metadata(getattr(self, name), name=name),
            )

    @property
    def particle(self) -> int:
        """Higher-energy endpoint, exposed through the neutral label protocol."""

        return self.particle_index

    @property
    def hole(self) -> int:
        """Lower-energy endpoint, exposed through the neutral label protocol."""

        return self.hole_index


@dataclass(init=False, frozen=True)
class FiniteTemperatureTDHFManifest:
    """Factory-only physical/provenance contract shared by thermal sectors."""

    source_eigenbasis_fingerprint: str
    interaction_derivative_fingerprint: str
    quadrature_convention: str
    response_scope: str
    interaction_block_convention: Literal[
        "physical_density_derivative_source_weight_once_v1"
    ]
    fingerprint: str = field(init=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _reject_public_record_init("FiniteTemperatureTDHFManifest")

    def _compute_fingerprint(self) -> str:
        payload = {
            "source_eigenbasis_fingerprint": self.source_eigenbasis_fingerprint,
            "interaction_derivative_fingerprint": (
                self.interaction_derivative_fingerprint
            ),
            "quadrature_convention": self.quadrature_convention,
            "response_scope": self.response_scope,
            "interaction_block_convention": self.interaction_block_convention,
        }
        digest = hashlib.sha256()
        digest.update(_MANIFEST_FINGERPRINT_VERSION.encode("ascii"))
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()


def _validated_manifest_fields(
    value: FiniteTemperatureTDHFManifest,
) -> tuple[str, str, str, str]:
    source = _fingerprint64(
        value.source_eigenbasis_fingerprint,
        name="source_eigenbasis_fingerprint",
    )
    interaction = _fingerprint64(
        value.interaction_derivative_fingerprint,
        name="interaction_derivative_fingerprint",
    )
    quadrature = _nonempty_text(
        value.quadrature_convention, name="quadrature_convention"
    )
    response_scope = _nonempty_text(value.response_scope, name="response_scope")
    if value.interaction_block_convention != _INTERACTION_BLOCK_CONVENTION:
        raise ValueError(
            "interaction_block_convention must be "
            f"{_INTERACTION_BLOCK_CONVENTION!r}"
        )
    return source, interaction, quadrature, response_scope

def _build_finite_temperature_tdhf_manifest(
    *,
    source_eigenbasis_fingerprint: str,
    interaction_derivative_fingerprint: str,
    quadrature_convention: str,
    response_scope: str,
    interaction_block_convention: Literal[
        "physical_density_derivative_source_weight_once_v1"
    ],
) -> FiniteTemperatureTDHFManifest:
    result = object.__new__(FiniteTemperatureTDHFManifest)
    object.__setattr__(
        result, "source_eigenbasis_fingerprint", source_eigenbasis_fingerprint
    )
    object.__setattr__(
        result,
        "interaction_derivative_fingerprint",
        interaction_derivative_fingerprint,
    )
    object.__setattr__(result, "quadrature_convention", quadrature_convention)
    object.__setattr__(result, "response_scope", response_scope)
    object.__setattr__(
        result, "interaction_block_convention", interaction_block_convention
    )
    source, interaction, quadrature, scope = _validated_manifest_fields(result)
    object.__setattr__(result, "source_eigenbasis_fingerprint", source)
    object.__setattr__(result, "interaction_derivative_fingerprint", interaction)
    object.__setattr__(result, "quadrature_convention", quadrature)
    object.__setattr__(result, "response_scope", scope)
    object.__setattr__(result, "fingerprint", result._compute_fingerprint())
    return _validate_manifest(result)

def make_finite_temperature_tdhf_manifest(
    *,
    source_eigenbasis_fingerprint: str,
    interaction_derivative_fingerprint: str,
    quadrature_convention: str,
    response_scope: str,
    interaction_block_convention: Literal[
        "physical_density_derivative_source_weight_once_v1"
    ] = _INTERACTION_BLOCK_CONVENTION,
) -> FiniteTemperatureTDHFManifest:
    """Create the strict manifest required by all finite-T transition weights."""

    return _build_finite_temperature_tdhf_manifest(
        source_eigenbasis_fingerprint=source_eigenbasis_fingerprint,
        interaction_derivative_fingerprint=interaction_derivative_fingerprint,
        quadrature_convention=quadrature_convention,
        response_scope=response_scope,
        interaction_block_convention=interaction_block_convention,
    )


def _validate_manifest(value: Any) -> FiniteTemperatureTDHFManifest:
    if not isinstance(value, FiniteTemperatureTDHFManifest):
        raise TypeError("manifest must be FiniteTemperatureTDHFManifest")
    expected = _validated_manifest_fields(value)
    actual = (
        value.source_eigenbasis_fingerprint,
        value.interaction_derivative_fingerprint,
        value.quadrature_convention,
        value.response_scope,
    )
    if actual != expected or value.fingerprint != value._compute_fingerprint():
        raise ValueError("finite-temperature TDHF manifest fingerprint is inconsistent")
    return value


@dataclass(init=False, frozen=True)
class _DirectedThermalTransition:
    endpoints: EnergyOrientedTransitionEndpoints
    gap: float
    hole_occupation: float
    particle_occupation: float
    log_d: float
    d: float
    sqrt_d: float
    weight: float

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _reject_public_record_init("thermal transition")

def _validated_directed_transition_fields(
    value: _DirectedThermalTransition,
) -> tuple[EnergyOrientedTransitionEndpoints, float, float, float, float, float, float, float]:
    if not isinstance(value.endpoints, EnergyOrientedTransitionEndpoints):
        raise TypeError("endpoints must be EnergyOrientedTransitionEndpoints")
    gap = _strict_real_scalar(value.gap, name="gap")
    if gap <= 0.0:
        raise ValueError("transition gap must be strictly positive")
    hole_occupation = _strict_real_scalar(
        value.hole_occupation, name="hole_occupation"
    )
    particle_occupation = _strict_real_scalar(
        value.particle_occupation, name="particle_occupation"
    )
    for name, occupation in (
        ("hole_occupation", hole_occupation),
        ("particle_occupation", particle_occupation),
    ):
        if not 0.0 <= occupation <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if hole_occupation < particle_occupation:
        raise ValueError(
            "lower-energy occupation must not be below higher-energy occupation"
        )
    log_d = _strict_real_scalar(value.log_d, name="log_d")
    if log_d > 0.0:
        raise ValueError("log_d must be non-positive")
    d = _strict_real_scalar(value.d, name="d")
    if d < 0.0:
        raise ValueError("d must be non-negative")
    sqrt_d = _strict_real_scalar(value.sqrt_d, name="sqrt_d")
    if sqrt_d <= 0.0:
        raise ValueError("sqrt_d must be strictly positive")
    weight = _strict_real_scalar(value.weight, name="weight")
    if weight <= 0.0:
        raise ValueError("transition quadrature weight must be positive")
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        expected_d = float(np.exp(log_d))
        expected_sqrt_d = float(np.exp(0.5 * log_d))
    if d != expected_d:
        raise ValueError("d is inconsistent with log_d")
    if sqrt_d != expected_sqrt_d:
        raise ValueError("sqrt_d is inconsistent with log_d")
    return (
        value.endpoints,
        gap,
        hole_occupation,
        particle_occupation,
        log_d,
        d,
        sqrt_d,
        weight,
    )

def _build_directed_thermal_transition(
    *,
    endpoints: EnergyOrientedTransitionEndpoints,
    gap: float,
    hole_occupation: float,
    particle_occupation: float,
    log_d: float,
    d: float,
    sqrt_d: float,
    weight: float,
) -> _DirectedThermalTransition:
    result = object.__new__(_DirectedThermalTransition)
    for name, value in (
        ("endpoints", endpoints),
        ("gap", gap),
        ("hole_occupation", hole_occupation),
        ("particle_occupation", particle_occupation),
        ("log_d", log_d),
        ("d", d),
        ("sqrt_d", sqrt_d),
        ("weight", weight),
    ):
        object.__setattr__(result, name, value)
    validated = _validated_directed_transition_fields(result)
    for name, value in zip(
        (
            "endpoints",
            "gap",
            "hole_occupation",
            "particle_occupation",
            "log_d",
            "d",
            "sqrt_d",
            "weight",
        ),
        validated,
        strict=True,
    ):
        object.__setattr__(result, name, value)
    return _validate_directed_thermal_transition(result)

def _validate_directed_thermal_transition(
    value: Any,
) -> _DirectedThermalTransition:
    if not isinstance(value, _DirectedThermalTransition):
        raise TypeError("transition must be a factory-built thermal transition")
    actual = (
        value.endpoints,
        value.gap,
        value.hole_occupation,
        value.particle_occupation,
        value.log_d,
        value.d,
        value.sqrt_d,
        value.weight,
    )
    if actual != _validated_directed_transition_fields(value):
        raise ValueError("thermal transition fields are not canonical")
    return value


@dataclass(init=False, frozen=True)
class FiniteTemperatureTransitionWeights:
    """Factory-only immutable transition inventory bound to one manifest."""

    manifest: FiniteTemperatureTDHFManifest
    transition_scope: str
    mu: float
    thermal_energy: float
    transitions: tuple[_DirectedThermalTransition, ...]
    fingerprint: str = field(init=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _reject_public_record_init("FiniteTemperatureTransitionWeights")

    def _compute_fingerprint(self) -> str:
        endpoint_records = [
            {
                "particle": transition.endpoints.particle,
                "hole": transition.endpoints.hole,
                "particle_momentum": transition.endpoints.particle_momentum,
                "hole_momentum": transition.endpoints.hole_momentum,
                "particle_flavor": transition.endpoints.particle_flavor,
                "hole_flavor": transition.endpoints.hole_flavor,
            }
            for transition in self.transitions
        ]
        digest = hashlib.sha256()
        digest.update(_TRANSITION_FINGERPRINT_VERSION.encode("ascii"))
        digest.update(
            json.dumps(
                {
                    "manifest_fingerprint": self.manifest.fingerprint,
                    "transition_scope": self.transition_scope,
                    "mu": self.mu,
                    "thermal_energy": self.thermal_energy,
                    "endpoints": endpoint_records,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for values in (
            self.gaps,
            self.hole_occupations,
            self.particle_occupations,
            self.log_d,
            self.d,
            self.sqrt_d,
            self.weights,
        ):
            _update_float64(digest, values)
        return digest.hexdigest()

    @property
    def endpoints(self) -> tuple[EnergyOrientedTransitionEndpoints, ...]:
        return tuple(transition.endpoints for transition in self.transitions)

    @property
    def gaps(self) -> np.ndarray:
        return _readonly_float_vector([item.gap for item in self.transitions])

    @property
    def hole_occupations(self) -> np.ndarray:
        return _readonly_float_vector(
            [item.hole_occupation for item in self.transitions]
        )

    @property
    def particle_occupations(self) -> np.ndarray:
        return _readonly_float_vector(
            [item.particle_occupation for item in self.transitions]
        )

    @property
    def log_d(self) -> np.ndarray:
        return _readonly_float_vector([item.log_d for item in self.transitions])

    @property
    def d(self) -> np.ndarray:
        return _readonly_float_vector([item.d for item in self.transitions])

    @property
    def sqrt_d(self) -> np.ndarray:
        return _readonly_float_vector([item.sqrt_d for item in self.transitions])

    @property
    def weights(self) -> np.ndarray:
        return _readonly_float_vector([item.weight for item in self.transitions])

    def physical_to_thermal_euclidean(self, z: np.ndarray) -> np.ndarray:
        """Map physical density amplitudes by ``x=sqrt(w/d) z``."""

        values = self._validated_amplitudes(z, name="z")
        scale = np.sqrt(self.weights) / self.sqrt_d
        if not np.all(np.isfinite(scale)):
            raise ValueError("physical-to-Euclidean thermal scale is non-finite")
        output = scale * values
        if not (
            np.all(np.isfinite(output.real)) and np.all(np.isfinite(output.imag))
        ):
            raise ValueError("physical-to-Euclidean amplitudes are non-finite")
        return output

    def thermal_euclidean_to_physical(self, x: np.ndarray) -> np.ndarray:
        """Map thermal Euclidean amplitudes by ``z=sqrt(d/w) x``."""

        values = self._validated_amplitudes(x, name="x")
        scale = self.sqrt_d / np.sqrt(self.weights)
        if not np.all(np.isfinite(scale)) or np.any(scale == 0.0):
            raise ValueError(
                "Euclidean-to-physical thermal scale underflowed or is non-finite"
            )
        output = scale * values
        if not (
            np.all(np.isfinite(output.real)) and np.all(np.isfinite(output.imag))
        ):
            raise ValueError("Euclidean-to-physical amplitudes are non-finite")
        return output

    def _validated_amplitudes(self, value: np.ndarray, *, name: str) -> np.ndarray:
        amplitudes = np.asarray(value, dtype=np.complex128)
        expected = (len(self.transitions),)
        if amplitudes.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if not (
            np.all(np.isfinite(amplitudes.real))
            and np.all(np.isfinite(amplitudes.imag))
        ):
            raise ValueError(f"{name} must be finite")
        return amplitudes

    def assembly_left_scale(self) -> np.ndarray:
        """Return ``sqrt(w*d)`` without reading the underflow-prone ``d``."""

        return self._positive_log_scale(
            0.5 * (np.log(self.weights) + self.log_d),
            name="sqrt(w*d)",
        )

    def assembly_right_scale(self) -> np.ndarray:
        """Return ``sqrt(d/w)`` without reading the underflow-prone ``d``."""

        return self._positive_log_scale(
            0.5 * (self.log_d - np.log(self.weights)),
            name="sqrt(d/w)",
        )

    @staticmethod
    def _positive_log_scale(log_values: np.ndarray, *, name: str) -> np.ndarray:
        with np.errstate(under="ignore", over="ignore", invalid="ignore"):
            values = np.exp(log_values)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{name} scaling underflowed or is non-finite")
        values.setflags(write=False)
        return values


def _validated_transition_weight_fields(
    value: FiniteTemperatureTransitionWeights,
) -> tuple[
    FiniteTemperatureTDHFManifest,
    str,
    float,
    float,
    tuple[_DirectedThermalTransition, ...],
]:
    manifest = _validate_manifest(value.manifest)
    scope = _nonempty_text(value.transition_scope, name="transition_scope")
    mu = _strict_real_scalar(value.mu, name="mu")
    thermal_energy = _strict_real_scalar(
        value.thermal_energy, name="thermal_energy"
    )
    if thermal_energy <= 0.0:
        raise ValueError("thermal_energy must be strictly positive")
    transitions = tuple(value.transitions)
    if not transitions:
        raise ValueError("finite-temperature TDHF requires nonempty transitions")
    for transition in transitions:
        _validate_directed_thermal_transition(transition)
    endpoint_indices = [
        (transition.endpoints.particle, transition.endpoints.hole)
        for transition in transitions
    ]
    if len(set(endpoint_indices)) != len(endpoint_indices):
        raise ValueError("duplicate ordered energy-oriented endpoint indices")
    return manifest, scope, mu, thermal_energy, transitions

def _build_finite_temperature_transition_weight_record(
    *,
    manifest: FiniteTemperatureTDHFManifest,
    transition_scope: str,
    mu: float,
    thermal_energy: float,
    transitions: tuple[_DirectedThermalTransition, ...],
) -> FiniteTemperatureTransitionWeights:
    result = object.__new__(FiniteTemperatureTransitionWeights)
    for name, value in (
        ("manifest", manifest),
        ("transition_scope", transition_scope),
        ("mu", mu),
        ("thermal_energy", thermal_energy),
        ("transitions", transitions),
    ):
        object.__setattr__(result, name, value)
    validated = _validated_transition_weight_fields(result)
    for name, value in zip(
        ("manifest", "transition_scope", "mu", "thermal_energy", "transitions"),
        validated,
        strict=True,
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "fingerprint", result._compute_fingerprint())
    return _validate_transition_weights(result, name="transition_weights")

def _validate_transition_weights(
    value: Any,
    *,
    name: str,
) -> FiniteTemperatureTransitionWeights:
    if not isinstance(value, FiniteTemperatureTransitionWeights):
        raise TypeError(f"{name} must be FiniteTemperatureTransitionWeights")
    expected = _validated_transition_weight_fields(value)
    actual = (
        value.manifest,
        value.transition_scope,
        value.mu,
        value.thermal_energy,
        value.transitions,
    )
    if actual != expected or value.fingerprint != value._compute_fingerprint():
        raise ValueError(f"{name} fingerprint is inconsistent")
    return value


def build_finite_temperature_transition_weights(
    endpoints: Sequence[EnergyOrientedTransitionEndpoints],
    hf_energies: np.ndarray,
    *,
    mu: float,
    thermal_energy: float,
    transition_quadrature_weights: np.ndarray,
    manifest: FiniteTemperatureTDHFManifest,
    transition_scope: str,
    supplied_occupations: np.ndarray | None = None,
    occupation_atol: float = 5.0e-15,
    occupation_rtol: float = 5.0e-13,
) -> FiniteTemperatureTransitionWeights:
    """Build stable thermal weights from a complete full-HF spectrum."""

    manifest_value = _validate_manifest(manifest)
    scope = _nonempty_text(transition_scope, name="transition_scope")
    endpoint_values = tuple(endpoints)
    if not endpoint_values:
        raise ValueError("finite-temperature TDHF requires nonempty endpoints")
    if not all(
        isinstance(item, EnergyOrientedTransitionEndpoints)
        for item in endpoint_values
    ):
        raise TypeError("endpoints must contain EnergyOrientedTransitionEndpoints")
    endpoint_indices = [(item.particle, item.hole) for item in endpoint_values]
    if len(set(endpoint_indices)) != len(endpoint_indices):
        raise ValueError("duplicate ordered energy-oriented endpoint indices")

    energies = _strict_real_array(hf_energies, name="hf_energies")
    if energies.ndim != 1 or energies.size == 0:
        raise ValueError("hf_energies must be a nonempty full-HF one-dimensional array")
    mu_value = _strict_real_scalar(mu, name="mu")
    thermal_value = _strict_real_scalar(thermal_energy, name="thermal_energy")
    if thermal_value <= 0.0:
        raise ValueError("thermal_energy must be strictly positive")

    weights = _strict_real_array(
        transition_quadrature_weights,
        name="transition_quadrature_weights",
    )
    if weights.shape != (len(endpoint_values),):
        raise ValueError(
            "transition_quadrature_weights must have one entry per endpoint record"
        )
    if np.any(weights <= 0.0):
        raise ValueError("transition quadrature weights must be finite and positive")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        dimensionless = (energies - mu_value) / thermal_value
    if not np.all(np.isfinite(dimensionless)):
        raise ValueError(
            "thermal_energy is too small to form finite dimensionless HF energies"
        )
    occupations = _fermi_occupations(dimensionless)
    if supplied_occupations is not None:
        supplied = _strict_real_array(
            supplied_occupations, name="supplied_occupations"
        )
        if supplied.shape != energies.shape:
            raise ValueError("supplied_occupations must match full hf_energies")
        if np.any(supplied < 0.0) or np.any(supplied > 1.0):
            raise ValueError("supplied_occupations must lie in [0, 1]")
        atol = _nonnegative_tolerance(occupation_atol, name="occupation_atol")
        rtol = _nonnegative_tolerance(occupation_rtol, name="occupation_rtol")
        mismatch = np.abs(supplied - occupations) > (
            atol + rtol * np.abs(occupations)
        )
        if np.any(mismatch):
            maximum = float(np.max(np.abs(supplied - occupations)))
            raise ValueError(
                "supplied occupation mismatch with finite-temperature Fermi values "
                f"(max_abs={maximum:.6e})"
            )

    transitions: list[_DirectedThermalTransition] = []
    n_quasiparticles = int(energies.size)
    for index, (item, weight) in enumerate(
        zip(endpoint_values, weights, strict=True)
    ):
        particle = item.particle
        hole = item.hole
        if particle >= n_quasiparticles or hole >= n_quasiparticles:
            raise IndexError("transition endpoint is outside the full HF eigensystem")
        gap = float(energies[particle] - energies[hole])
        if not np.isfinite(gap) or gap <= 0.0:
            raise ValueError(
                "every energy-oriented transition requires eps_particle > eps_hole; "
                f"endpoint index {index} has gap {gap!r}"
            )
        log_dimensionless_gap = float(np.log(gap) - np.log(thermal_value))
        log_d = _log_fermi_difference(
            float(dimensionless[hole]),
            float(dimensionless[particle]),
            log_dimensionless_gap,
        )
        with np.errstate(under="ignore", over="ignore", invalid="ignore"):
            d = float(np.exp(log_d))
            sqrt_d = float(np.exp(0.5 * log_d))
        if not np.isfinite(sqrt_d) or sqrt_d <= 0.0:
            raise ValueError(
                "sqrt(d) underflowed or is non-finite for energy-oriented "
                f"transition {index}; log_d={log_d!r}"
            )
        transitions.append(
            _build_directed_thermal_transition(
                endpoints=item,
                gap=gap,
                hole_occupation=float(occupations[hole]),
                particle_occupation=float(occupations[particle]),
                log_d=log_d,
                d=d,
                sqrt_d=sqrt_d,
                weight=float(weight),
            )
        )

    return _build_finite_temperature_transition_weight_record(
        manifest=manifest_value,
        transition_scope=scope,
        mu=mu_value,
        thermal_energy=thermal_value,
        transitions=tuple(transitions),
    )


@dataclass(frozen=True)
class FiniteTemperatureSelfConjugateStructureResiduals:
    A_hermitian: float
    B_symmetric: float
    tolerance: float

    @property
    def ok(self) -> bool:
        return bool(
            np.isfinite(self.A_hermitian)
            and np.isfinite(self.B_symmetric)
            and self.A_hermitian <= self.tolerance
            and self.B_symmetric <= self.tolerance
        )


@dataclass(frozen=True)
class FiniteTemperatureSignedQStructureResiduals:
    A_plus_hermitian: float
    A_minus_hermitian: float
    B_partner_transpose: float
    tolerance: float

    @property
    def ok(self) -> bool:
        values = (
            self.A_plus_hermitian,
            self.A_minus_hermitian,
            self.B_partner_transpose,
        )
        return bool(
            all(np.isfinite(value) and value <= self.tolerance for value in values)
        )


def _assemble_interaction_block(
    row_weights: FiniteTemperatureTransitionWeights,
    derivative: Any,
    column_weights: FiniteTemperatureTransitionWeights,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _readonly_complex_matrix(
        derivative,
        name=name,
        shape=(len(row_weights.transitions), len(column_weights.transitions)),
    )
    output = (
        row_weights.assembly_left_scale()[:, None]
        * matrix
        * column_weights.assembly_right_scale()[None, :]
    )
    if not (
        np.all(np.isfinite(output.real)) and np.all(np.isfinite(output.imag))
    ):
        raise ValueError(f"assembled {name} block is non-finite")
    result = np.array(output, dtype=np.complex128, copy=True, order="C")
    result.setflags(write=False)
    return matrix, result


def _assembly_fingerprint(
    *,
    kind: str,
    manifest_fingerprint: str,
    transition_fingerprints: tuple[str, ...],
    structure_tolerance: float,
    derivative_blocks: tuple[np.ndarray, ...],
    assembled_blocks: tuple[np.ndarray, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(_ASSEMBLY_FINGERPRINT_VERSION.encode("ascii"))
    digest.update(
        json.dumps(
            {
                "kind": kind,
                "manifest_fingerprint": manifest_fingerprint,
                "transition_fingerprints": transition_fingerprints,
                "structure_tolerance": structure_tolerance,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for block in (*derivative_blocks, *assembled_blocks):
        _update_complex128(digest, block)
    return digest.hexdigest()


def _typed_response_scope(
    manifest_scope: str,
    transition_scopes: tuple[tuple[str, str], ...],
) -> str:
    return json.dumps(
        {
            "manifest_response_scope": manifest_scope,
            "transition_scopes": [
                {"lane": lane, "scope": scope}
                for lane, scope in transition_scopes
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

@dataclass(init=False, frozen=True)
class FiniteTemperatureTDHFSelfConjugateAssembly:
    """Factory-only self-conjugate thermal A/B algebraic result."""

    transition_weights: FiniteTemperatureTransitionWeights
    _K_A: np.ndarray = field(repr=False, compare=False)
    _K_B: np.ndarray = field(repr=False, compare=False)
    structure_tolerance: float = 1.0e-10
    A: np.ndarray = field(init=False)
    B: np.ndarray = field(init=False)
    assembly_fingerprint: str = field(init=False)
    structure: FiniteTemperatureSelfConjugateStructureResiduals = field(init=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _reject_public_record_init("FiniteTemperatureTDHFSelfConjugateAssembly")

    @property
    def manifest(self) -> FiniteTemperatureTDHFManifest:
        return self.transition_weights.manifest

    def to_typed_self_conjugate_sector(
        self,
        q: TDHFSelfConjugateQ,
        canonical_sewing_provenance: str,
        static_hessian_authority: TDHFStaticHessianAuthority = "not_established",
    ) -> TDHFSelfConjugateQSector:
        """Bind this algebraic result into the generic self-conjugate API.

        This conversion does not establish ``F0/P0`` stationarity or thermal
        closure; a production adapter must validate those physical conditions.
        """

        _validate_self_conjugate_assembly(self)
        if not isinstance(q, TDHFSelfConjugateQ):
            raise TypeError("q must be TDHFSelfConjugateQ")
        return TDHFSelfConjugateQSector(
            q=q,
            canonical_pairs=self.transition_weights.endpoints,
            A=self.A,
            B=self.B,
            source_fingerprint=self.manifest.source_eigenbasis_fingerprint,
            interaction_fingerprint=self.assembly_fingerprint,
            response_scope=_typed_response_scope(
                self.manifest.response_scope,
                (("self_conjugate", self.transition_weights.transition_scope),),
            ),
            static_hessian_authority=static_hessian_authority,
            canonical_sewing_provenance=canonical_sewing_provenance,
        )

def _self_conjugate_assembly_fields(
    transition_weights: FiniteTemperatureTransitionWeights,
    K_A: Any,
    K_B: Any,
    structure_tolerance: Any,
) -> tuple[
    FiniteTemperatureTransitionWeights,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    FiniteTemperatureSelfConjugateStructureResiduals,
    str,
]:
    weights = _validate_transition_weights(
        transition_weights, name="transition_weights"
    )
    tolerance = _nonnegative_tolerance(
        structure_tolerance, name="structure_tolerance"
    )
    raw_a, interaction_a = _assemble_interaction_block(
        weights, K_A, weights, name="K_A"
    )
    raw_b, b = _assemble_interaction_block(weights, K_B, weights, name="K_B")
    a = np.array(
        interaction_a + np.diag(weights.gaps),
        dtype=np.complex128,
        copy=True,
        order="C",
    )
    a.setflags(write=False)
    structure = FiniteTemperatureSelfConjugateStructureResiduals(
        A_hermitian=_max_abs(a - np.conj(a.T)),
        B_symmetric=_max_abs(b - b.T),
        tolerance=tolerance,
    )
    fingerprint = _assembly_fingerprint(
        kind="self_conjugate",
        manifest_fingerprint=weights.manifest.fingerprint,
        transition_fingerprints=(weights.fingerprint,),
        structure_tolerance=tolerance,
        derivative_blocks=(raw_a, raw_b),
        assembled_blocks=(a, b),
    )
    return weights, raw_a, raw_b, tolerance, a, b, structure, fingerprint

def _build_self_conjugate_assembly(
    *,
    transition_weights: FiniteTemperatureTransitionWeights,
    K_A: Any,
    K_B: Any,
    structure_tolerance: Any,
) -> FiniteTemperatureTDHFSelfConjugateAssembly:
    result = object.__new__(FiniteTemperatureTDHFSelfConjugateAssembly)
    for name, value in (
        ("transition_weights", transition_weights),
        ("_K_A", K_A),
        ("_K_B", K_B),
        ("structure_tolerance", structure_tolerance),
    ):
        object.__setattr__(result, name, value)
    values = _self_conjugate_assembly_fields(
        result.transition_weights,
        result._K_A,
        result._K_B,
        result.structure_tolerance,
    )
    for name, value in zip(
        (
            "transition_weights",
            "_K_A",
            "_K_B",
            "structure_tolerance",
            "A",
            "B",
            "structure",
            "assembly_fingerprint",
        ),
        values,
        strict=True,
    ):
        object.__setattr__(result, name, value)
    return _validate_self_conjugate_assembly(result)

def _validate_self_conjugate_assembly(
    value: Any,
) -> FiniteTemperatureTDHFSelfConjugateAssembly:
    if not isinstance(value, FiniteTemperatureTDHFSelfConjugateAssembly):
        raise TypeError(
            "assembly must be FiniteTemperatureTDHFSelfConjugateAssembly"
        )
    expected = _self_conjugate_assembly_fields(
        value.transition_weights,
        value._K_A,
        value._K_B,
        value.structure_tolerance,
    )
    if (
        value.transition_weights is not expected[0]
        or value.structure_tolerance != expected[3]
        or value.structure != expected[6]
        or value.assembly_fingerprint != expected[7]
        or not np.array_equal(value._K_A, expected[1])
        or not np.array_equal(value._K_B, expected[2])
        or not np.array_equal(value.A, expected[4])
        or not np.array_equal(value.B, expected[5])
    ):
        raise ValueError("self-conjugate assembly fingerprint is inconsistent")
    return value


@dataclass(init=False, frozen=True)
class FiniteTemperatureTDHFSignedQAssembly:
    """Factory-only signed thermal blocks with bound endpoint labels."""

    plus_transition_weights: FiniteTemperatureTransitionWeights
    minus_transition_weights: FiniteTemperatureTransitionWeights
    _K_A_plus: np.ndarray = field(repr=False, compare=False)
    _K_B_plus_minus: np.ndarray = field(repr=False, compare=False)
    _K_A_minus: np.ndarray = field(repr=False, compare=False)
    _K_B_minus_plus: np.ndarray = field(repr=False, compare=False)
    structure_tolerance: float = 1.0e-10
    A_plus: np.ndarray = field(init=False)
    B_plus_minus: np.ndarray = field(init=False)
    A_minus: np.ndarray = field(init=False)
    B_minus_plus: np.ndarray = field(init=False)
    assembly_fingerprint: str = field(init=False)
    structure: FiniteTemperatureSignedQStructureResiduals = field(init=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _reject_public_record_init("FiniteTemperatureTDHFSignedQAssembly")

    @property
    def manifest(self) -> FiniteTemperatureTDHFManifest:
        return self.plus_transition_weights.manifest

    @property
    def plus_endpoints(self) -> tuple[EnergyOrientedTransitionEndpoints, ...]:
        return self.plus_transition_weights.endpoints

    @property
    def minus_endpoints(self) -> tuple[EnergyOrientedTransitionEndpoints, ...]:
        return self.minus_transition_weights.endpoints

    def _to_nonauthoritative_signed_q_blocks(self) -> TDHFSignedQBlocks:
        """Return raw algebraic blocks without provenance or authority binding."""

        return TDHFSignedQBlocks(
            plus_pairs=self.plus_endpoints,
            minus_pairs=self.minus_endpoints,
            A_plus=self.A_plus,
            B_plus_minus=self.B_plus_minus,
            A_minus=self.A_minus,
            B_minus_plus=self.B_minus_plus,
        )

    def to_typed_generic_sector(
        self,
        q: TDHFGenericSignedQ,
        sewing: TDHFNambuSewing,
        static_hessian_authority: TDHFStaticHessianAuthority = "not_established",
    ) -> TDHFGenericSignedQSector:
        """Return a source/pair/K-byte-bound typed generic signed sector.

        This conversion does not establish ``F0/P0`` stationarity or thermal
        closure; a production adapter must validate those physical conditions.
        """

        _validate_signed_q_assembly(self)
        if not isinstance(q, TDHFGenericSignedQ):
            raise TypeError("q must be TDHFGenericSignedQ")
        if not isinstance(sewing, TDHFNambuSewing):
            raise TypeError("sewing must be TDHFNambuSewing")
        source = self.manifest.source_eigenbasis_fingerprint
        if sewing.source_fingerprint != source:
            raise ValueError("Nambu sewing source fingerprint mismatch")
        expected_plus = fingerprint_tdhf_pairs(self.plus_endpoints)
        expected_minus = fingerprint_tdhf_pairs(self.minus_endpoints)
        if (
            sewing.plus_pairs_fingerprint != expected_plus
            or sewing.minus_pairs_fingerprint != expected_minus
        ):
            raise ValueError("Nambu sewing pair fingerprints mismatch")
        return TDHFGenericSignedQSector(
            q=q,
            blocks=self._to_nonauthoritative_signed_q_blocks(),
            sewing=sewing,
            source_fingerprint=source,
            interaction_fingerprint=self.assembly_fingerprint,
            response_scope=_typed_response_scope(
                self.manifest.response_scope,
                (
                    ("+q", self.plus_transition_weights.transition_scope),
                    ("-q", self.minus_transition_weights.transition_scope),
                ),
            ),
            static_hessian_authority=static_hessian_authority,
        )

def _signed_q_assembly_fields(
    plus_transition_weights: FiniteTemperatureTransitionWeights,
    minus_transition_weights: FiniteTemperatureTransitionWeights,
    K_A_plus: Any,
    K_B_plus_minus: Any,
    K_A_minus: Any,
    K_B_minus_plus: Any,
    structure_tolerance: Any,
) -> tuple[Any, ...]:
    plus = _validate_transition_weights(
        plus_transition_weights, name="plus_transition_weights"
    )
    minus = _validate_transition_weights(
        minus_transition_weights, name="minus_transition_weights"
    )
    if plus.manifest.fingerprint != minus.manifest.fingerprint:
        raise ValueError(
            "signed-q transition inventories must share the exact same "
            "finite-temperature TDHF manifest fingerprint"
        )
    if plus.mu != minus.mu:
        raise ValueError("signed-q transition inventories must use the same mu")
    if plus.thermal_energy != minus.thermal_energy:
        raise ValueError(
            "signed-q transition inventories must use the same thermal_energy"
        )
    tolerance = _nonnegative_tolerance(
        structure_tolerance, name="structure_tolerance"
    )
    raw_ap, ap_interaction = _assemble_interaction_block(
        plus, K_A_plus, plus, name="K_A_plus"
    )
    raw_bp, bp = _assemble_interaction_block(
        plus, K_B_plus_minus, minus, name="K_B_plus_minus"
    )
    raw_am, am_interaction = _assemble_interaction_block(
        minus, K_A_minus, minus, name="K_A_minus"
    )
    raw_bm, bm = _assemble_interaction_block(
        minus, K_B_minus_plus, plus, name="K_B_minus_plus"
    )
    ap = np.array(
        ap_interaction + np.diag(plus.gaps),
        dtype=np.complex128,
        copy=True,
        order="C",
    )
    am = np.array(
        am_interaction + np.diag(minus.gaps),
        dtype=np.complex128,
        copy=True,
        order="C",
    )
    ap.setflags(write=False)
    am.setflags(write=False)
    structure = FiniteTemperatureSignedQStructureResiduals(
        A_plus_hermitian=_max_abs(ap - np.conj(ap.T)),
        A_minus_hermitian=_max_abs(am - np.conj(am.T)),
        B_partner_transpose=_max_abs(bp - bm.T),
        tolerance=tolerance,
    )
    fingerprint = _assembly_fingerprint(
        kind="signed_q",
        manifest_fingerprint=plus.manifest.fingerprint,
        transition_fingerprints=(plus.fingerprint, minus.fingerprint),
        structure_tolerance=tolerance,
        derivative_blocks=(raw_ap, raw_bp, raw_am, raw_bm),
        assembled_blocks=(ap, bp, am, bm),
    )
    return (
        plus,
        minus,
        raw_ap,
        raw_bp,
        raw_am,
        raw_bm,
        tolerance,
        ap,
        bp,
        am,
        bm,
        structure,
        fingerprint,
    )

def _build_signed_q_assembly(
    *,
    plus_transition_weights: FiniteTemperatureTransitionWeights,
    minus_transition_weights: FiniteTemperatureTransitionWeights,
    K_A_plus: Any,
    K_B_plus_minus: Any,
    K_A_minus: Any,
    K_B_minus_plus: Any,
    structure_tolerance: Any,
) -> FiniteTemperatureTDHFSignedQAssembly:
    result = object.__new__(FiniteTemperatureTDHFSignedQAssembly)
    for name, value in (
        ("plus_transition_weights", plus_transition_weights),
        ("minus_transition_weights", minus_transition_weights),
        ("_K_A_plus", K_A_plus),
        ("_K_B_plus_minus", K_B_plus_minus),
        ("_K_A_minus", K_A_minus),
        ("_K_B_minus_plus", K_B_minus_plus),
        ("structure_tolerance", structure_tolerance),
    ):
        object.__setattr__(result, name, value)
    values = _signed_q_assembly_fields(
        result.plus_transition_weights,
        result.minus_transition_weights,
        result._K_A_plus,
        result._K_B_plus_minus,
        result._K_A_minus,
        result._K_B_minus_plus,
        result.structure_tolerance,
    )
    for name, value in zip(
        (
            "plus_transition_weights",
            "minus_transition_weights",
            "_K_A_plus",
            "_K_B_plus_minus",
            "_K_A_minus",
            "_K_B_minus_plus",
            "structure_tolerance",
            "A_plus",
            "B_plus_minus",
            "A_minus",
            "B_minus_plus",
            "structure",
            "assembly_fingerprint",
        ),
        values,
        strict=True,
    ):
        object.__setattr__(result, name, value)
    return _validate_signed_q_assembly(result)

def _validate_signed_q_assembly(
    value: Any,
) -> FiniteTemperatureTDHFSignedQAssembly:
    if not isinstance(value, FiniteTemperatureTDHFSignedQAssembly):
        raise TypeError("assembly must be FiniteTemperatureTDHFSignedQAssembly")
    expected = _signed_q_assembly_fields(
        value.plus_transition_weights,
        value.minus_transition_weights,
        value._K_A_plus,
        value._K_B_plus_minus,
        value._K_A_minus,
        value._K_B_minus_plus,
        value.structure_tolerance,
    )
    if (
        value.plus_transition_weights is not expected[0]
        or value.minus_transition_weights is not expected[1]
        or value.structure_tolerance != expected[6]
        or value.structure != expected[11]
        or value.assembly_fingerprint != expected[12]
        or not np.array_equal(value._K_A_plus, expected[2])
        or not np.array_equal(value._K_B_plus_minus, expected[3])
        or not np.array_equal(value._K_A_minus, expected[4])
        or not np.array_equal(value._K_B_minus_plus, expected[5])
        or not np.array_equal(value.A_plus, expected[7])
        or not np.array_equal(value.B_plus_minus, expected[8])
        or not np.array_equal(value.A_minus, expected[9])
        or not np.array_equal(value.B_minus_plus, expected[10])
    ):
        raise ValueError("signed-q assembly fingerprint is inconsistent")
    return value


def assemble_finite_temperature_self_conjugate_tdhf(
    transition_weights: FiniteTemperatureTransitionWeights,
    K_A: np.ndarray,
    K_B: np.ndarray,
    *,
    structure_tolerance: float = 1.0e-10,
    raise_on_structure_error: bool = False,
) -> FiniteTemperatureTDHFSelfConjugateAssembly:
    """Assemble one self-conjugate sector under the bound manifest contract."""

    result = _build_self_conjugate_assembly(
        transition_weights=transition_weights,
        K_A=K_A,
        K_B=K_B,
        structure_tolerance=structure_tolerance,
    )
    if raise_on_structure_error and not result.structure.ok:
        raise ValueError(
            "finite-temperature self-conjugate TDHF structure gate failed: "
            f"{result.structure}"
        )
    return result


def assemble_finite_temperature_signed_q_tdhf(
    plus_transition_weights: FiniteTemperatureTransitionWeights,
    minus_transition_weights: FiniteTemperatureTransitionWeights,
    K_A_plus: np.ndarray,
    K_B_plus_minus: np.ndarray,
    K_A_minus: np.ndarray,
    K_B_minus_plus: np.ndarray,
    *,
    structure_tolerance: float = 1.0e-10,
    raise_on_structure_error: bool = False,
) -> FiniteTemperatureTDHFSignedQAssembly:
    """Assemble independent ``+q/-q`` blocks under one exact manifest."""

    result = _build_signed_q_assembly(
        plus_transition_weights=plus_transition_weights,
        minus_transition_weights=minus_transition_weights,
        K_A_plus=K_A_plus,
        K_B_plus_minus=K_B_plus_minus,
        K_A_minus=K_A_minus,
        K_B_minus_plus=K_B_minus_plus,
        structure_tolerance=structure_tolerance,
    )
    if raise_on_structure_error and not result.structure.ok:
        raise ValueError(
            "finite-temperature signed-q TDHF structure gate failed: "
            f"{result.structure}"
        )
    return result


__all__ = [
    "EnergyOrientedTransitionEndpoints",
    "FiniteTemperatureSelfConjugateStructureResiduals",
    "FiniteTemperatureSignedQStructureResiduals",
    "FiniteTemperatureTDHFManifest",
    "FiniteTemperatureTDHFSelfConjugateAssembly",
    "FiniteTemperatureTDHFSignedQAssembly",
    "FiniteTemperatureTransitionWeights",
    "assemble_finite_temperature_self_conjugate_tdhf",
    "assemble_finite_temperature_signed_q_tdhf",
    "build_finite_temperature_transition_weights",
    "make_finite_temperature_tdhf_manifest",
]
