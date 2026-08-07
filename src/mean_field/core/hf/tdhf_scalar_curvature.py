"""Approval-gated scalar-Hessian certificates for generic signed-q TDHF.

The energy callback is trusted scientific code.  This module binds its source,
implementation snapshot, and caller-declared immutable inputs, but it is not a
sandbox: closure cells, defaults, globals, and hostile providers are outside
the authority of this contract.

The raw mathematical identity is ``E''(0) = 2 Re(v^dagger H_plus v)`` with
``v=(x,y*)``.  Its whole-Hessian authority additionally requires independent
five-point stationarity gates on the complete real tangent inventory
``{e_i, i e_i}``; curvature directions are never used as a stationarity span.
Energy normalization is a reporting-only division performed after the raw
comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from hashlib import sha256
import inspect
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.linalg import expm

from .tdhf_signed import (
    TDHFGenericSignedQ,
    TDHFGenericSignedQSector,
    TDHFNambuSewing,
    TDHFSignedQBlocks,
    TDHFSignedQMatrices,
    build_tdhf_signed_q_matrices,
    fingerprint_tdhf_matrix,
    fingerprint_tdhf_pairs,
    fingerprint_tdhf_sector,
)

Array = np.ndarray
EnergyNormalization = Literal["total", "per_k", "per_area"]

_CALLBACK_TRUST = (
    "trusted_adapter: source/implementation/declared immutable inputs are bound, "
    "but closure/default/global-state semantics and hostile-provider resistance are not proved"
)
_NORMALIZATION_STATEMENT = (
    "The denominator is a caller-attested reporting conversion only. Changing it changes "
    "reported curvature units and the convention fingerprint, not the raw mathematical "
    "comparison. This certificate does not establish Nk, finite-area, quadrature, or "
    "energy-density physics."
)
_SCALAR_CURVATURE_FACTORY_TOKEN = object()

# Locked v1 policy.  Approvals may tighten the tolerance ceilings, but the
# roundoff multiplier is an exact fixed value rather than a tunable tolerance.
TDHF_SCALAR_CURVATURE_V1_TANGENT_TOLERANCE_MAXIMUM = 1.0e-8
TDHF_SCALAR_CURVATURE_V1_DIRECTION_TOLERANCE_MAXIMUM = 1.0e-10
TDHF_SCALAR_CURVATURE_V1_STATIONARITY_ABSOLUTE_MAXIMUM = 1.0e-5
TDHF_SCALAR_CURVATURE_V1_STATIONARITY_RELATIVE_MAXIMUM = 1.0e-5
TDHF_SCALAR_CURVATURE_V1_CURVATURE_ABSOLUTE_MAXIMUM = 1.0e-5
TDHF_SCALAR_CURVATURE_V1_CURVATURE_RELATIVE_MAXIMUM = 1.0e-5
TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER = 256.0
# Compatibility name only; validation requires equality, not merely <=.
TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER_MAXIMUM = (
    TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER
)
TDHF_SCALAR_CURVATURE_V1_PROJECTOR_TOLERANCE_MAXIMUM = 1.0e-8
TDHF_SCALAR_CURVATURE_V1_MATRIX_ABSOLUTE_MAXIMUM = 1.0e-5
TDHF_SCALAR_CURVATURE_V1_MATRIX_RELATIVE_MAXIMUM = 1.0e-5
# ``h`` is the dimensionless angle multiplying the anti-Hermitian generator.
TDHF_SCALAR_CURVATURE_V1_STEP_MINIMUM = 1.0e-4
TDHF_SCALAR_CURVATURE_V1_STEP_MAXIMUM = 5.0e-2
# These ceilings have the callback's raw energy/radian and energy/radian^2
# units.  They prevent a large energy zero or cancellation at small h from
# turning the roundoff estimate itself into a vacuous acceptance tolerance.
TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM = 1.0e-8
TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_CURVATURE_ALLOWANCE_MAXIMUM = 1.0e-7


def _readonly_complex(value: Any, *, ndim: int | None = None) -> Array:
    array = np.array(value, dtype=np.complex128, copy=True, order="C")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected a rank-{ndim} complex array, got rank {array.ndim}")
    array.setflags(write=False)
    return array


def _sha_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    payload = (
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    )
    return sha256(payload).hexdigest()


def _stable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": value.shape, "sha256": _sha_array(value)}
    if is_dataclass(value):
        return {item.name: _stable_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_value(item) for item in value]
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        _stable_value(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(payload).hexdigest()


def _validate_fingerprint(name: str, value: str) -> None:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 fingerprint")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} is not hexadecimal") from error


def _finite_real(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not bool")
    result = float(value)
    if not np.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _bounded_tolerance(name: str, value: Any, maximum: float, *, positive: bool = False) -> float:
    result = _finite_real(name, value, nonnegative=True)
    if positive and result == 0.0:
        raise ValueError(f"{name} must be positive")
    if result > maximum:
        raise ValueError(
            f"{name}={result} exceeds locked v1 maximum {maximum}; vacuous approvals are forbidden"
        )
    return result


@dataclass(frozen=True, slots=True)
class TDHFTransitionTangentBasis:
    """Ordered, disjoint ``T=(1-P0)TP0`` tangents bound to one sector source."""

    source_projector: Array
    plus_tangents: tuple[Array, ...]
    minus_tangents: tuple[Array, ...]
    source_fingerprint: str
    plus_pairs_fingerprint: str
    minus_pairs_fingerprint: str
    validation_tolerance: float = 2.0e-11

    def __post_init__(self) -> None:
        projector = _readonly_complex(self.source_projector, ndim=2)
        plus = tuple(_readonly_complex(item, ndim=2) for item in self.plus_tangents)
        minus = tuple(_readonly_complex(item, ndim=2) for item in self.minus_tangents)
        object.__setattr__(self, "source_projector", projector)
        object.__setattr__(self, "plus_tangents", plus)
        object.__setattr__(self, "minus_tangents", minus)
        if projector.shape[0] != projector.shape[1]:
            raise ValueError("source_projector must be square")
        if not plus or not minus:
            raise ValueError("the independent +q and -q tangent lanes must both be nonempty")
        for name in ("source_fingerprint", "plus_pairs_fingerprint", "minus_pairs_fingerprint"):
            _validate_fingerprint(name, getattr(self, name))
        if self.plus_pairs_fingerprint == self.minus_pairs_fingerprint:
            raise ValueError("+q and -q ordered pair bases must be distinct")
        tolerance = _bounded_tolerance(
            "validation_tolerance",
            self.validation_tolerance,
            TDHF_SCALAR_CURVATURE_V1_TANGENT_TOLERANCE_MAXIMUM,
            positive=True,
        )
        object.__setattr__(self, "validation_tolerance", tolerance)
        self.validate()

    @property
    def source_projector_fingerprint(self) -> str:
        return fingerprint_tdhf_matrix(self.source_projector)

    @property
    def plus_tangent_fingerprints(self) -> tuple[str, ...]:
        return tuple(fingerprint_tdhf_matrix(item) for item in self.plus_tangents)

    @property
    def minus_tangent_fingerprints(self) -> tuple[str, ...]:
        return tuple(fingerprint_tdhf_matrix(item) for item in self.minus_tangents)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "source_fingerprint": self.source_fingerprint,
                "source_projector_fingerprint": self.source_projector_fingerprint,
                "plus_pairs_fingerprint": self.plus_pairs_fingerprint,
                "minus_pairs_fingerprint": self.minus_pairs_fingerprint,
                "plus_tangent_fingerprints": self.plus_tangent_fingerprints,
                "minus_tangent_fingerprints": self.minus_tangent_fingerprints,
                "validation_tolerance": self.validation_tolerance,
            }
        )

    def validate(self) -> None:
        p0 = self.source_projector
        tolerance = self.validation_tolerance
        if not np.allclose(p0, p0.conj().T, atol=tolerance, rtol=0.0):
            raise ValueError("source projector is not Hermitian")
        if not np.allclose(p0 @ p0, p0, atol=tolerance, rtol=0.0):
            raise ValueError("source projector is not idempotent")
        q0 = np.eye(p0.shape[0], dtype=np.complex128) - p0
        tangents = self.plus_tangents + self.minus_tangents
        for index, tangent in enumerate(tangents):
            if tangent.shape != p0.shape:
                raise ValueError(f"tangent {index} shape does not match the source projector")
            if not np.allclose(tangent, q0 @ tangent @ p0, atol=tolerance, rtol=0.0):
                raise ValueError(f"tangent {index} does not satisfy T=(1-P0) T P0")
        gram = np.asarray(
            [[np.vdot(left, right) for right in tangents] for left in tangents],
            dtype=np.complex128,
        )
        if not np.allclose(gram, np.eye(len(tangents)), atol=tolerance, rtol=0.0):
            raise ValueError("the combined +q/-q tangent basis is not HS-orthonormal/disjoint")


@dataclass(frozen=True, slots=True)
class TDHFPhysicalDirection:
    """A normalized physical direction; this is an ordinary positive norm."""

    label: str
    x: Array
    y: Array
    normalization_tolerance: float = 2.0e-12

    def __post_init__(self) -> None:
        x = _readonly_complex(self.x, ndim=1)
        y = _readonly_complex(self.y, ndim=1)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        if type(self.label) is not str or not self.label:
            raise ValueError("direction label must be nonempty")
        tolerance = _bounded_tolerance(
            "normalization_tolerance",
            self.normalization_tolerance,
            TDHF_SCALAR_CURVATURE_V1_DIRECTION_TOLERANCE_MAXIMUM,
            positive=True,
        )
        object.__setattr__(self, "normalization_tolerance", tolerance)
        norm2 = float(np.vdot(x, x).real + np.vdot(y, y).real)
        if abs(norm2 - 1.0) > tolerance:
            raise ValueError(f"physical direction must obey ||x||^2+||y||^2=1, got {norm2}")

    @property
    def vector(self) -> Array:
        """Return the signed-Hessian coordinate ``v=(x,y*)``."""

        return _readonly_complex(np.concatenate((self.x, self.y.conj())), ndim=1)

    @property
    def fingerprint(self) -> str:
        return _fingerprint((self.label, self.x, self.y))


def _physical_direction_from_v(label: str, vector: Array, n_plus: int) -> TDHFPhysicalDirection:
    vector = np.asarray(vector, dtype=np.complex128)
    return TDHFPhysicalDirection(
        label=label,
        x=vector[:n_plus],
        # The lower signed-Hessian coordinate is y*, not y.
        y=vector[n_plus:].conj(),
    )


def canonical_tdhf_stationarity_directions(
    n_plus: int, n_minus: int
) -> tuple[TDHFPhysicalDirection, ...]:
    """Return the canonical real-tangent ``{e_i, i e_i}`` inventory.

    Stationarity is a real-linear condition on a complex ``d``-coordinate
    tangent space, so all ``2d`` real quadratures are mandatory.  In the lower
    lane the signed-Hessian coordinate is ``v_lower=y*``; consequently
    ``i e_i`` maps to ``y_i=-i`` rather than ``+i``.
    """

    if type(n_plus) is not int or type(n_minus) is not int:
        raise TypeError("canonical stationarity lane dimensions must be exact integers")
    if n_plus < 1 or n_minus < 1:
        raise ValueError("canonical stationarity lanes must both be nonempty")
    dimension = n_plus + n_minus
    directions: list[TDHFPhysicalDirection] = []
    for index in range(dimension):
        real = np.zeros(dimension, dtype=np.complex128)
        real[index] = 1.0
        directions.append(
            _physical_direction_from_v(f"stationarity.real[{index}]", real, n_plus)
        )
        imaginary = np.zeros(dimension, dtype=np.complex128)
        imaginary[index] = 1.0j
        directions.append(
            _physical_direction_from_v(
                f"stationarity.imag[{index}]", imaginary, n_plus
            )
        )
    result = tuple(directions)
    if len(result) != 2 * dimension:
        raise RuntimeError("internal canonical 2d stationarity construction failed")
    return result


def canonical_tdhf_scalar_directions(
    n_plus: int, n_minus: int
) -> tuple[TDHFPhysicalDirection, ...]:
    """Return the canonical informationally-complete ``d^2`` inventory.

    Exact order and labels are:
    ``diag[i]`` for ``i=0..d-1``, then for lexicographic ``i<j`` the adjacent
    pair ``real[i,j]``, ``imag[i,j]``.  The associated signed-Hessian vectors
    are ``e_i``, ``(e_i+e_j)/sqrt(2)``, and
    ``(e_i+i e_j)/sqrt(2)`` respectively, mapped back through ``v=(x,y*)``.
    """

    if type(n_plus) is not int or type(n_minus) is not int:
        raise TypeError("canonical direction lane dimensions must be exact integers")
    if n_plus < 1 or n_minus < 1:
        raise ValueError("canonical direction lanes must both be nonempty")
    dimension = n_plus + n_minus
    directions: list[TDHFPhysicalDirection] = []
    for index in range(dimension):
        vector = np.zeros(dimension, dtype=np.complex128)
        vector[index] = 1.0
        directions.append(_physical_direction_from_v(f"diag[{index}]", vector, n_plus))
    root_two = np.sqrt(2.0)
    for left in range(dimension):
        for right in range(left + 1, dimension):
            real = np.zeros(dimension, dtype=np.complex128)
            real[left] = real[right] = 1.0 / root_two
            directions.append(
                _physical_direction_from_v(f"real[{left},{right}]", real, n_plus)
            )
            imag = np.zeros(dimension, dtype=np.complex128)
            imag[left] = 1.0 / root_two
            imag[right] = 1.0j / root_two
            directions.append(
                _physical_direction_from_v(f"imag[{left},{right}]", imag, n_plus)
            )
    result = tuple(directions)
    if len(result) != dimension * dimension:
        raise RuntimeError("internal canonical d^2 direction construction failed")
    return result


def _direction_inventory_fingerprint(
    directions: Sequence[TDHFPhysicalDirection],
    n_plus: int,
    n_minus: int,
    *,
    purpose: Literal["curvature", "stationarity"] = "curvature",
) -> str:
    return _fingerprint(
        {
            "coordinate": "v=(x,y*)",
            "purpose": purpose,
            "n_plus": n_plus,
            "n_minus": n_minus,
            "labels": tuple(item.label for item in directions),
            "fingerprints": tuple(item.fingerprint for item in directions),
        }
    )


@dataclass(frozen=True, slots=True)
class TDHFEnergyConvention:
    """Caller-attested conversion from raw callback energy to reported units."""

    normalization: EnergyNormalization
    denominator: float
    energy_units: str
    curvature_units: str
    denominator_source: str
    caller_attested_reporting_only: bool = field(default=True, init=False)
    normalization_physics_certified: bool = field(default=False, init=False)
    semantics: str = field(default=_NORMALIZATION_STATEMENT, init=False)

    def __post_init__(self) -> None:
        if self.normalization not in ("total", "per_k", "per_area"):
            raise ValueError("normalization must be total, per_k, or per_area")
        denominator = _finite_real("energy denominator", self.denominator)
        if denominator <= 0.0:
            raise ValueError("energy denominator must be positive")
        object.__setattr__(self, "denominator", denominator)
        if self.normalization == "total" and denominator != 1.0:
            raise ValueError("total-energy reporting requires denominator=1")
        if any(
            type(value) is not str or not value
            for value in (self.energy_units, self.curvature_units, self.denominator_source)
        ):
            raise ValueError("energy convention units and denominator_source must be explicit")
        if (
            self.caller_attested_reporting_only is not True
            or self.normalization_physics_certified is not False
            or self.semantics != _NORMALIZATION_STATEMENT
        ):
            raise ValueError("energy convention authority semantics changed")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class TDHFScalarCurvatureTolerances:
    stationarity_absolute: float
    stationarity_relative: float
    curvature_absolute: float
    curvature_relative: float
    roundoff_multiplier: float = TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER
    projector_tolerance: float = 5.0e-11
    matrix_absolute: float = 1.0e-7
    matrix_relative: float = 1.0e-6

    def __post_init__(self) -> None:
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        maxima = {
            "stationarity_absolute": TDHF_SCALAR_CURVATURE_V1_STATIONARITY_ABSOLUTE_MAXIMUM,
            "stationarity_relative": TDHF_SCALAR_CURVATURE_V1_STATIONARITY_RELATIVE_MAXIMUM,
            "curvature_absolute": TDHF_SCALAR_CURVATURE_V1_CURVATURE_ABSOLUTE_MAXIMUM,
            "curvature_relative": TDHF_SCALAR_CURVATURE_V1_CURVATURE_RELATIVE_MAXIMUM,
            "projector_tolerance": TDHF_SCALAR_CURVATURE_V1_PROJECTOR_TOLERANCE_MAXIMUM,
            "matrix_absolute": TDHF_SCALAR_CURVATURE_V1_MATRIX_ABSOLUTE_MAXIMUM,
            "matrix_relative": TDHF_SCALAR_CURVATURE_V1_MATRIX_RELATIVE_MAXIMUM,
        }
        roundoff_multiplier = _finite_real(
            "roundoff_multiplier", self.roundoff_multiplier
        )
        if roundoff_multiplier != TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER:
            raise ValueError(
                "roundoff_multiplier must equal the exact locked v1 value "
                f"{TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER}"
            )
        object.__setattr__(self, "roundoff_multiplier", roundoff_multiplier)
        for item in fields(self):
            if item.name == "roundoff_multiplier":
                continue
            value = _bounded_tolerance(
                item.name,
                getattr(self, item.name),
                maxima[item.name],
                positive=item.name in ("projector_tolerance",),
            )
            object.__setattr__(self, item.name, value)


@dataclass(frozen=True, slots=True)
class TDHFScalarCurvatureStepLadder:
    """Preregistered strictly decreasing five-point steps and bounded tolerances."""

    steps: tuple[float, ...]
    tolerances: TDHFScalarCurvatureTolerances
    registration_label: str

    def __post_init__(self) -> None:
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        if type(self.tolerances) is not TDHFScalarCurvatureTolerances:
            raise TypeError("step ladder requires exact scalar-curvature tolerances")
        self.tolerances._validate_consistency()
        steps = tuple(_finite_real("finite-difference step", step) for step in self.steps)
        object.__setattr__(self, "steps", steps)
        if len(steps) < 2:
            raise ValueError("a preregistered ladder needs at least two steps")
        if any(
            step < TDHF_SCALAR_CURVATURE_V1_STEP_MINIMUM
            or step > TDHF_SCALAR_CURVATURE_V1_STEP_MAXIMUM
            for step in steps
        ):
            raise ValueError(
                "finite-difference steps must lie within the locked v1 dimensionless "
                f"unitary-angle range [{TDHF_SCALAR_CURVATURE_V1_STEP_MINIMUM}, "
                f"{TDHF_SCALAR_CURVATURE_V1_STEP_MAXIMUM}]"
            )
        if any(left <= right for left, right in zip(steps, steps[1:])):
            raise ValueError("finite-difference steps must be strictly decreasing")
        if type(self.registration_label) is not str or not self.registration_label:
            raise ValueError("registration_label must be nonempty")

    @property
    def fingerprint(self) -> str:
        self._validate_consistency()
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class TDHFScalarCallbackProvenance:
    module: str
    qualname: str
    module_path: str | None
    module_sha256: str | None
    callback_source_sha256: str | None
    callback_code_sha256: str
    trust_model: str = _CALLBACK_TRUST
    trusted_callback: bool = True
    hostile_provider_proof: bool = False

    def __post_init__(self) -> None:
        if any(type(value) is not str or not value for value in (self.module, self.qualname)):
            raise ValueError("callback module and qualname must be explicit")
        for name in ("module_sha256", "callback_source_sha256", "callback_code_sha256"):
            value = getattr(self, name)
            if value is not None:
                _validate_fingerprint(name, value)
        if self.module_path is not None and (
            type(self.module_path) is not str or not self.module_path
        ):
            raise ValueError("callback module_path must be nonempty when present")
        if (
            self.trust_model != _CALLBACK_TRUST
            or self.trusted_callback is not True
            or self.hostile_provider_proof is not False
        ):
            raise ValueError("callback trust semantics changed")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def snapshot_tdhf_scalar_callback(
    callback: Callable[[Array], float],
) -> TDHFScalarCallbackProvenance:
    """Snapshot callback file/source/code without invoking the callback."""

    if not callable(callback):
        raise TypeError("energy_callback must be callable")
    module = inspect.getmodule(callback)
    module_name = getattr(module, "__name__", getattr(callback, "__module__", "<unknown>"))
    qualname = getattr(
        callback, "__qualname__", getattr(callback, "__name__", type(callback).__qualname__)
    )
    path_text: str | None = None
    module_digest: str | None = None
    source_digest: str | None = None
    try:
        path = inspect.getsourcefile(callback)
        if path is not None:
            resolved = Path(path).resolve()
            path_text = str(resolved)
            module_digest = sha256(resolved.read_bytes()).hexdigest()
    except (OSError, TypeError):
        pass
    try:
        source_digest = sha256(inspect.getsource(callback).encode()).hexdigest()
    except (OSError, TypeError):
        pass
    code = getattr(callback, "__code__", None)
    if code is None:
        call = getattr(callback, "__call__", None)
        code = getattr(call, "__code__", None)
    if code is None:
        code_payload = repr(type(callback)).encode()
    else:
        code_payload = (
            code.co_code
            + repr(code.co_consts).encode()
            + repr(code.co_names).encode()
            + repr(code.co_varnames).encode()
        )
    return TDHFScalarCallbackProvenance(
        module=module_name,
        qualname=qualname,
        module_path=path_text,
        module_sha256=module_digest,
        callback_source_sha256=source_digest,
        callback_code_sha256=sha256(code_payload).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class TDHFScalarFunctionalManifest:
    """Immutable honest-caller declaration of the scalar functional and inputs."""

    source_functional_fingerprint: str
    immutable_callback_input_fingerprint: str
    implementation_fingerprint: str
    provenance: str

    def __post_init__(self) -> None:
        for name in (
            "source_functional_fingerprint",
            "immutable_callback_input_fingerprint",
            "implementation_fingerprint",
        ):
            _validate_fingerprint(name, getattr(self, name))
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("scalar functional manifest provenance must be explicit")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def make_tdhf_scalar_functional_manifest(
    *,
    energy_callback: Callable[[Array], float],
    source_functional_fingerprint: str,
    immutable_callback_input_fingerprint: str,
    provenance: str,
) -> TDHFScalarFunctionalManifest:
    """Bind a callback implementation snapshot to declared source/input hashes."""

    callback = snapshot_tdhf_scalar_callback(energy_callback)
    return TDHFScalarFunctionalManifest(
        source_functional_fingerprint=source_functional_fingerprint,
        immutable_callback_input_fingerprint=immutable_callback_input_fingerprint,
        implementation_fingerprint=callback.fingerprint,
        provenance=provenance,
    )


def _assemble_h_plus(
    sector: TDHFGenericSignedQSector,
) -> tuple[TDHFSignedQMatrices, Array, tuple[Any, ...], tuple[Any, ...]]:
    if type(sector) is not TDHFGenericSignedQSector:
        raise TypeError("sector must be the exact TDHFGenericSignedQSector type")
    if type(sector.q) is not TDHFGenericSignedQ:
        raise TypeError("generic signed sector has a non-exact q type")
    if type(sector.blocks) is not TDHFSignedQBlocks:
        raise TypeError("generic signed sector has a non-exact blocks type")
    if type(sector.sewing) is not TDHFNambuSewing:
        raise TypeError("generic signed sector has a non-exact sewing type")
    signed = build_tdhf_signed_q_matrices(
        sector.blocks, sector.sewing, raise_on_structure_error=True
    )
    if type(signed) is not TDHFSignedQMatrices or not signed.structure.ok:
        raise ValueError("generic signed-q structure was not established")
    h_plus = _readonly_complex(signed.H_plus, ndim=2)
    return signed, h_plus, sector.blocks.plus_pairs, sector.blocks.minus_pairs


@dataclass(frozen=True, slots=True)
class TDHFScalarCurvatureApproval:
    """Detached authority boundary registered before any callback evaluation."""

    api_version: str
    sector_fingerprint: str
    sector_source_fingerprint: str
    source_projector_fingerprint: str
    tangent_basis_fingerprint: str
    plus_tangent_fingerprints: tuple[str, ...]
    minus_tangent_fingerprints: tuple[str, ...]
    interaction_fingerprint: str
    plus_pairs_fingerprint: str
    minus_pairs_fingerprint: str
    h_plus_fingerprint: str
    functional_manifest: TDHFScalarFunctionalManifest
    functional_manifest_fingerprint: str
    callback: TDHFScalarCallbackProvenance
    callback_provenance_fingerprint: str
    convention: TDHFEnergyConvention
    convention_fingerprint: str
    stationarity_directions: tuple[TDHFPhysicalDirection, ...]
    stationarity_direction_labels: tuple[str, ...]
    stationarity_direction_fingerprints: tuple[str, ...]
    stationarity_direction_inventory_fingerprint: str
    canonical_stationarity_complete_inventory: bool
    directions: tuple[TDHFPhysicalDirection, ...]
    direction_labels: tuple[str, ...]
    direction_fingerprints: tuple[str, ...]
    direction_inventory_fingerprint: str
    canonical_complete_inventory: bool
    step_ladder: TDHFScalarCurvatureStepLadder
    ladder_fingerprint: str
    provenance: str

    def __post_init__(self) -> None:
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        if self.api_version != "tdhf_scalar_curvature_approval.v2":
            raise ValueError("scalar-curvature approval API version changed")
        for name in (
            "sector_fingerprint",
            "sector_source_fingerprint",
            "source_projector_fingerprint",
            "tangent_basis_fingerprint",
            "interaction_fingerprint",
            "plus_pairs_fingerprint",
            "minus_pairs_fingerprint",
            "h_plus_fingerprint",
            "functional_manifest_fingerprint",
            "callback_provenance_fingerprint",
            "convention_fingerprint",
            "stationarity_direction_inventory_fingerprint",
            "direction_inventory_fingerprint",
            "ladder_fingerprint",
        ):
            _validate_fingerprint(name, getattr(self, name))
        for name in ("plus_tangent_fingerprints", "minus_tangent_fingerprints"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise TypeError(f"{name} must be a nonempty tuple")
            for index, value in enumerate(values):
                _validate_fingerprint(f"{name}[{index}]", value)
        if type(self.functional_manifest) is not TDHFScalarFunctionalManifest:
            raise TypeError("approval requires the exact scalar functional manifest")
        if self.functional_manifest_fingerprint != self.functional_manifest.fingerprint:
            raise ValueError("approval scalar functional manifest fingerprint mismatch")
        if type(self.callback) is not TDHFScalarCallbackProvenance:
            raise TypeError("approval callback provenance has the wrong exact type")
        if self.callback_provenance_fingerprint != self.callback.fingerprint:
            raise ValueError("approval callback provenance fingerprint mismatch")
        if self.functional_manifest.implementation_fingerprint != self.callback.fingerprint:
            raise ValueError("functional manifest implementation does not match callback snapshot")
        if type(self.convention) is not TDHFEnergyConvention:
            raise TypeError("approval energy convention has the wrong exact type")
        if self.convention_fingerprint != self.convention.fingerprint:
            raise ValueError("approval energy convention fingerprint mismatch")
        if type(self.step_ladder) is not TDHFScalarCurvatureStepLadder:
            raise TypeError("approval step ladder has the wrong exact type")
        if self.ladder_fingerprint != self.step_ladder.fingerprint:
            raise ValueError("approval step-ladder fingerprint mismatch")
        if type(self.stationarity_directions) is not tuple or not self.stationarity_directions:
            raise TypeError("approval stationarity directions must be a nonempty tuple")
        if any(
            type(item) is not TDHFPhysicalDirection
            for item in self.stationarity_directions
        ):
            raise TypeError("approval stationarity directions have a non-exact nested type")
        stationarity_labels = tuple(item.label for item in self.stationarity_directions)
        stationarity_fingerprints = tuple(
            item.fingerprint for item in self.stationarity_directions
        )
        if (
            stationarity_labels != self.stationarity_direction_labels
            or stationarity_fingerprints != self.stationarity_direction_fingerprints
        ):
            raise ValueError("approval stationarity label/fingerprint inventory drifted")
        if len(set(stationarity_labels)) != len(stationarity_labels):
            raise ValueError("approval stationarity direction labels must be unique")
        if type(self.directions) is not tuple or not self.directions:
            raise TypeError("approval curvature directions must be a nonempty tuple")
        if any(type(item) is not TDHFPhysicalDirection for item in self.directions):
            raise TypeError("approval curvature directions have a non-exact nested type")
        labels = tuple(item.label for item in self.directions)
        fingerprints = tuple(item.fingerprint for item in self.directions)
        if labels != self.direction_labels or fingerprints != self.direction_fingerprints:
            raise ValueError("approval curvature label/fingerprint inventory drifted")
        if len(set(labels)) != len(labels):
            raise ValueError("approval curvature direction labels must be unique")
        n_plus = len(self.plus_tangent_fingerprints)
        n_minus = len(self.minus_tangent_fingerprints)
        if any(
            item.x.shape != (n_plus,) or item.y.shape != (n_minus,)
            for item in self.stationarity_directions + self.directions
        ):
            raise ValueError("approval direction dimensions do not match tangent lanes")
        expected_stationarity_inventory = _direction_inventory_fingerprint(
            self.stationarity_directions,
            n_plus,
            n_minus,
            purpose="stationarity",
        )
        if (
            self.stationarity_direction_inventory_fingerprint
            != expected_stationarity_inventory
        ):
            raise ValueError("approval stationarity direction inventory fingerprint mismatch")
        canonical_stationarity = canonical_tdhf_stationarity_directions(
            n_plus, n_minus
        )
        expected_stationarity_complete = stationarity_fingerprints == tuple(
            item.fingerprint for item in canonical_stationarity
        )
        if (
            self.canonical_stationarity_complete_inventory
            is not expected_stationarity_complete
            or not expected_stationarity_complete
        ):
            raise ValueError(
                "approval must bind the exact canonical stationarity-complete inventory"
            )
        expected_inventory = _direction_inventory_fingerprint(
            self.directions, n_plus, n_minus, purpose="curvature"
        )
        if self.direction_inventory_fingerprint != expected_inventory:
            raise ValueError("approval curvature direction inventory fingerprint mismatch")
        canonical = canonical_tdhf_scalar_directions(n_plus, n_minus)
        expected_complete = fingerprints == tuple(item.fingerprint for item in canonical)
        if self.canonical_complete_inventory is not expected_complete:
            raise ValueError("approval canonical-complete curvature flag is inconsistent")
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("detached approval provenance must be explicit")

    @property
    def fingerprint(self) -> str:
        self._validate_consistency()
        return _fingerprint(self)


def _validate_sector_basis_bindings(
    sector: TDHFGenericSignedQSector,
    tangent_basis: TDHFTransitionTangentBasis,
    interaction_fingerprint: str,
) -> tuple[Array, str, str]:
    _validate_fingerprint("interaction_fingerprint", interaction_fingerprint)
    _validate_fingerprint("sector.source_fingerprint", sector.source_fingerprint)
    _validate_fingerprint("sector.interaction_fingerprint", sector.interaction_fingerprint)
    if sector.interaction_fingerprint != interaction_fingerprint:
        raise ValueError("interaction fingerprint does not match the generic signed sector")
    if tangent_basis.source_fingerprint != sector.source_fingerprint:
        raise ValueError("tangent basis source_fingerprint does not match the signed sector")
    tangent_basis.validate()
    signed, h_plus, plus_pairs, minus_pairs = _assemble_h_plus(sector)
    plus_pairs_fingerprint = fingerprint_tdhf_pairs(plus_pairs)
    minus_pairs_fingerprint = fingerprint_tdhf_pairs(minus_pairs)
    sewing = sector.sewing
    if sewing.source_fingerprint != sector.source_fingerprint:
        raise ValueError("Nambu sewing source fingerprint does not match the sector")
    if sewing.plus_pairs_fingerprint != plus_pairs_fingerprint:
        raise ValueError("Nambu sewing +q pair fingerprint is stale")
    if sewing.minus_pairs_fingerprint != minus_pairs_fingerprint:
        raise ValueError("Nambu sewing -q pair fingerprint is stale")
    if tangent_basis.plus_pairs_fingerprint != plus_pairs_fingerprint:
        raise ValueError("ordered +q tangent basis does not match the sector pair fingerprint")
    if tangent_basis.minus_pairs_fingerprint != minus_pairs_fingerprint:
        raise ValueError("ordered -q tangent basis does not match the sector pair fingerprint")
    n_plus_tangents = len(tangent_basis.plus_tangents)
    n_minus_tangents = len(tangent_basis.minus_tangents)
    n_plus_pairs = len(plus_pairs)
    n_minus_pairs = len(minus_pairs)
    if (n_plus_tangents, n_minus_tangents) != (n_plus_pairs, n_minus_pairs):
        raise ValueError(
            "tangent lane cardinalities do not match sector pair counts: "
            f"+q tangents={n_plus_tangents}, +q pairs={n_plus_pairs}; "
            f"-q tangents={n_minus_tangents}, -q pairs={n_minus_pairs}"
        )
    if not signed.structure.ok:
        raise ValueError("typed signed-q structure gate did not pass")
    dimension = n_plus_tangents + n_minus_tangents
    if h_plus.shape != (dimension, dimension):
        raise ValueError("H_plus dimension does not match the combined tangent basis")
    return h_plus, plus_pairs_fingerprint, minus_pairs_fingerprint


def make_tdhf_scalar_curvature_approval(
    *,
    sector: TDHFGenericSignedQSector,
    tangent_basis: TDHFTransitionTangentBasis,
    directions: Sequence[TDHFPhysicalDirection],
    energy_callback: Callable[[Array], float],
    functional_manifest: TDHFScalarFunctionalManifest,
    energy_convention: TDHFEnergyConvention,
    step_ladder: TDHFScalarCurvatureStepLadder,
    interaction_fingerprint: str,
    provenance: str,
) -> TDHFScalarCurvatureApproval:
    """Create a detached approval without evaluating ``energy_callback``."""

    if type(sector) is not TDHFGenericSignedQSector:
        raise TypeError("sector must be the exact TDHFGenericSignedQSector type")
    if type(tangent_basis) is not TDHFTransitionTangentBasis:
        raise TypeError("tangent_basis must be the exact tangent basis type")
    if type(functional_manifest) is not TDHFScalarFunctionalManifest:
        raise TypeError("functional_manifest must be the exact manifest type")
    if type(energy_convention) is not TDHFEnergyConvention:
        raise TypeError("energy_convention must be the exact convention type")
    if type(step_ladder) is not TDHFScalarCurvatureStepLadder:
        raise TypeError("step_ladder must be the exact ladder type")
    directions = tuple(directions)
    if not directions or any(type(item) is not TDHFPhysicalDirection for item in directions):
        raise TypeError("at least one exact preregistered physical direction is required")
    h_plus, plus_pairs_fingerprint, minus_pairs_fingerprint = (
        _validate_sector_basis_bindings(sector, tangent_basis, interaction_fingerprint)
    )
    callback = snapshot_tdhf_scalar_callback(energy_callback)
    if functional_manifest.implementation_fingerprint != callback.fingerprint:
        raise ValueError("functional manifest implementation does not match callback snapshot")
    n_plus = len(tangent_basis.plus_tangents)
    n_minus = len(tangent_basis.minus_tangents)
    canonical_stationarity = canonical_tdhf_stationarity_directions(n_plus, n_minus)
    canonical = canonical_tdhf_scalar_directions(n_plus, n_minus)
    fingerprints = tuple(item.fingerprint for item in directions)
    return TDHFScalarCurvatureApproval(
        api_version="tdhf_scalar_curvature_approval.v2",
        sector_fingerprint=fingerprint_tdhf_sector(sector),
        sector_source_fingerprint=sector.source_fingerprint,
        source_projector_fingerprint=tangent_basis.source_projector_fingerprint,
        tangent_basis_fingerprint=tangent_basis.fingerprint,
        plus_tangent_fingerprints=tangent_basis.plus_tangent_fingerprints,
        minus_tangent_fingerprints=tangent_basis.minus_tangent_fingerprints,
        interaction_fingerprint=interaction_fingerprint,
        plus_pairs_fingerprint=plus_pairs_fingerprint,
        minus_pairs_fingerprint=minus_pairs_fingerprint,
        h_plus_fingerprint=fingerprint_tdhf_matrix(h_plus),
        functional_manifest=functional_manifest,
        functional_manifest_fingerprint=functional_manifest.fingerprint,
        callback=callback,
        callback_provenance_fingerprint=callback.fingerprint,
        convention=energy_convention,
        convention_fingerprint=energy_convention.fingerprint,
        stationarity_directions=canonical_stationarity,
        stationarity_direction_labels=tuple(
            item.label for item in canonical_stationarity
        ),
        stationarity_direction_fingerprints=tuple(
            item.fingerprint for item in canonical_stationarity
        ),
        stationarity_direction_inventory_fingerprint=_direction_inventory_fingerprint(
            canonical_stationarity,
            n_plus,
            n_minus,
            purpose="stationarity",
        ),
        canonical_stationarity_complete_inventory=True,
        directions=directions,
        direction_labels=tuple(item.label for item in directions),
        direction_fingerprints=fingerprints,
        direction_inventory_fingerprint=_direction_inventory_fingerprint(
            directions, n_plus, n_minus, purpose="curvature"
        ),
        canonical_complete_inventory=(
            fingerprints == tuple(item.fingerprint for item in canonical)
        ),
        step_ladder=step_ladder,
        ladder_fingerprint=step_ladder.fingerprint,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class TDHFScalarStationarityStepEvidence:
    """Factory evidence for one five-point first-derivative stationarity gate."""

    step: float
    raw_energies_at_minus2_minus1_zero_plus1_plus2: tuple[
        float, float, float, float, float
    ]
    raw_first_derivative: float
    raw_second_derivative_scale_probe: float
    stationarity_residual: float
    stationarity_roundoff_allowance: float
    curvature_roundoff_nonvacuity_probe: float
    stationarity_bound: float
    projector_max_hermiticity_residual: float
    projector_max_idempotency_residual: float
    projector_max_trace_residual: float
    stationarity_passed: bool

    def __post_init__(self) -> None:
        energies = self.raw_energies_at_minus2_minus1_zero_plus1_plus2
        if type(energies) is not tuple or len(energies) != 5:
            raise TypeError("stationarity evidence requires the exact five raw callback energies")
        for index, value in enumerate(energies):
            _finite_real(f"stationarity raw energy {index}", value)
        for name in (
            "step",
            "raw_first_derivative",
            "raw_second_derivative_scale_probe",
        ):
            _finite_real(name, getattr(self, name))
        if self.step <= 0.0:
            raise ValueError("stationarity evidence step must be positive")
        for name in (
            "stationarity_residual",
            "stationarity_roundoff_allowance",
            "curvature_roundoff_nonvacuity_probe",
            "stationarity_bound",
            "projector_max_hermiticity_residual",
            "projector_max_idempotency_residual",
            "projector_max_trace_residual",
        ):
            _finite_real(name, getattr(self, name), nonnegative=True)
        if (
            self.stationarity_roundoff_allowance
            > TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM
            or self.curvature_roundoff_nonvacuity_probe
            > TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_CURVATURE_ALLOWANCE_MAXIMUM
        ):
            raise ValueError(
                "stationarity evidence exceeds a locked v1 roundoff non-vacuity ceiling"
            )
        if self.stationarity_residual != abs(self.raw_first_derivative):
            raise ValueError("stationarity residual does not equal the raw first derivative")
        if type(self.stationarity_passed) is not bool:
            raise TypeError("stationarity pass flag must be an exact boolean")
        expected = bool(self.stationarity_residual <= self.stationarity_bound)
        if self.stationarity_passed is not expected:
            raise ValueError("stationarity pass flag is inconsistent")
        if not self.stationarity_passed:
            raise ValueError("factory stationarity evidence must pass")


@dataclass(frozen=True, slots=True)
class TDHFScalarStationarityDirectionEvidence:
    label: str
    direction_fingerprint: str
    generator_antihermiticity_residual: float
    steps: tuple[TDHFScalarStationarityStepEvidence, ...]
    raw_first_derivative_plateau: tuple[float, ...]
    all_registered_steps_passed: bool

    def __post_init__(self) -> None:
        if type(self.label) is not str or not self.label:
            raise ValueError("stationarity direction evidence label must be nonempty")
        _validate_fingerprint("stationarity direction_fingerprint", self.direction_fingerprint)
        _finite_real(
            "stationarity generator_antihermiticity_residual",
            self.generator_antihermiticity_residual,
            nonnegative=True,
        )
        if type(self.steps) is not tuple or not self.steps or any(
            type(item) is not TDHFScalarStationarityStepEvidence for item in self.steps
        ):
            raise TypeError("every stationarity direction requires exact step evidence")
        if self.raw_first_derivative_plateau != tuple(
            item.raw_first_derivative for item in self.steps
        ):
            raise ValueError("stationarity first-derivative plateau does not match evidence")
        expected = all(item.stationarity_passed for item in self.steps)
        if type(self.all_registered_steps_passed) is not bool:
            raise TypeError("stationarity all-steps flag must be an exact boolean")
        if self.all_registered_steps_passed is not expected or not expected:
            raise ValueError("stationarity direction did not pass every registered step")


@dataclass(frozen=True, slots=True)
class TDHFScalarCurvatureStepEvidence:
    step: float
    raw_energies_at_minus2_minus1_zero_plus1_plus2: tuple[
        float, float, float, float, float
    ]
    raw_first_derivative: float
    raw_second_derivative: float
    measured_raw_quadratic: float
    reported_second_derivative: float
    raw_target_curvature: float
    reported_target_curvature: float
    stationarity_residual: float
    curvature_residual: float
    stationarity_roundoff_allowance: float
    curvature_roundoff_allowance: float
    stationarity_bound: float
    curvature_bound: float
    projector_max_hermiticity_residual: float
    projector_max_idempotency_residual: float
    projector_max_trace_residual: float
    stationarity_passed: bool
    curvature_passed: bool

    def __post_init__(self) -> None:
        if type(self.raw_energies_at_minus2_minus1_zero_plus1_plus2) is not tuple or len(
            self.raw_energies_at_minus2_minus1_zero_plus1_plus2
        ) != 5:
            raise TypeError("step evidence requires the exact five raw callback energies")
        for index, value in enumerate(self.raw_energies_at_minus2_minus1_zero_plus1_plus2):
            _finite_real(f"raw energy {index}", value)
        for name in (
            "step",
            "raw_first_derivative",
            "raw_second_derivative",
            "measured_raw_quadratic",
            "reported_second_derivative",
            "raw_target_curvature",
            "reported_target_curvature",
        ):
            _finite_real(name, getattr(self, name))
        if self.step <= 0.0:
            raise ValueError("step evidence step must be positive")
        if self.measured_raw_quadratic != self.raw_second_derivative / 2.0:
            raise ValueError("measured raw quadratic must be raw E''/2")
        for name in (
            "stationarity_residual",
            "curvature_residual",
            "stationarity_roundoff_allowance",
            "curvature_roundoff_allowance",
            "stationarity_bound",
            "curvature_bound",
            "projector_max_hermiticity_residual",
            "projector_max_idempotency_residual",
            "projector_max_trace_residual",
        ):
            _finite_real(name, getattr(self, name), nonnegative=True)
        if (
            self.stationarity_roundoff_allowance
            > TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM
            or self.curvature_roundoff_allowance
            > TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_CURVATURE_ALLOWANCE_MAXIMUM
        ):
            raise ValueError("step evidence exceeds a locked v1 roundoff non-vacuity ceiling")
        if type(self.stationarity_passed) is not bool or type(self.curvature_passed) is not bool:
            raise TypeError("step pass flags must be exact booleans")
        if self.stationarity_residual != abs(self.raw_first_derivative):
            raise ValueError("stationarity residual does not equal the raw first derivative")
        if self.curvature_residual != abs(self.raw_second_derivative - self.raw_target_curvature):
            raise ValueError("curvature residual does not equal the raw mathematical mismatch")
        stationarity_expected = bool(
            self.stationarity_residual <= self.stationarity_bound
        )
        curvature_expected = bool(self.curvature_residual <= self.curvature_bound)
        if self.stationarity_passed is not stationarity_expected:
            raise ValueError("stationarity pass flag is inconsistent")
        if self.curvature_passed is not curvature_expected:
            raise ValueError("curvature pass flag is inconsistent")
        if not self.stationarity_passed or not self.curvature_passed:
            raise ValueError("factory step evidence must pass both mandatory gates")


@dataclass(frozen=True, slots=True)
class TDHFScalarCurvatureDirectionEvidence:
    label: str
    direction_fingerprint: str
    target_quadratic_form: float
    raw_target_curvature: float
    reported_target_curvature: float
    generator_antihermiticity_residual: float
    steps: tuple[TDHFScalarCurvatureStepEvidence, ...]
    raw_curvature_plateau: tuple[float, ...]
    reported_curvature_plateau: tuple[float, ...]
    all_registered_steps_passed: bool

    def __post_init__(self) -> None:
        if type(self.label) is not str or not self.label:
            raise ValueError("direction evidence label must be nonempty")
        _validate_fingerprint("direction_fingerprint", self.direction_fingerprint)
        for name in ("target_quadratic_form", "raw_target_curvature", "reported_target_curvature"):
            _finite_real(name, getattr(self, name))
        _finite_real(
            "generator_antihermiticity_residual",
            self.generator_antihermiticity_residual,
            nonnegative=True,
        )
        if type(self.steps) is not tuple or not self.steps or any(
            type(item) is not TDHFScalarCurvatureStepEvidence for item in self.steps
        ):
            raise TypeError("every direction requires nonempty exact step evidence")
        if self.raw_curvature_plateau != tuple(item.raw_second_derivative for item in self.steps):
            raise ValueError("raw curvature plateau does not match the step evidence")
        if self.reported_curvature_plateau != tuple(
            item.reported_second_derivative for item in self.steps
        ):
            raise ValueError("reported curvature plateau does not match the step evidence")
        expected = all(item.stationarity_passed and item.curvature_passed for item in self.steps)
        if type(self.all_registered_steps_passed) is not bool:
            raise TypeError("direction all-steps flag must be an exact boolean")
        if self.all_registered_steps_passed is not expected or not expected:
            raise ValueError("direction evidence does not pass every registered step")


@dataclass(frozen=True, slots=True)
class TDHFScalarHessianReconstructionEvidence:
    step: float
    reconstructed_hessian: Array
    reconstructed_hessian_fingerprint: str
    max_abs_residual: float
    matrix_bound: float
    reconstruction_passed: bool

    def __post_init__(self) -> None:
        matrix = _readonly_complex(self.reconstructed_hessian, ndim=2)
        object.__setattr__(self, "reconstructed_hessian", matrix)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("reconstructed Hessian must be square")
        _validate_fingerprint(
            "reconstructed_hessian_fingerprint", self.reconstructed_hessian_fingerprint
        )
        if self.reconstructed_hessian_fingerprint != fingerprint_tdhf_matrix(matrix):
            raise ValueError("reconstructed Hessian fingerprint mismatch")
        _finite_real("reconstruction step", self.step)
        _finite_real("matrix max-abs residual", self.max_abs_residual, nonnegative=True)
        _finite_real("matrix bound", self.matrix_bound, nonnegative=True)
        if not np.array_equal(matrix, matrix.conj().T):
            raise ValueError("reconstructed Hessian is not exactly Hermitian")
        if type(self.reconstruction_passed) is not bool:
            raise TypeError("matrix reconstruction pass flag must be an exact boolean")
        reconstruction_expected = bool(self.max_abs_residual <= self.matrix_bound)
        if self.reconstruction_passed is not reconstruction_expected:
            raise ValueError("matrix reconstruction pass flag is inconsistent")
        if not self.reconstruction_passed:
            raise ValueError("factory reconstruction evidence must pass")


@dataclass(frozen=True, slots=True)
class TDHFScalarCurvatureFactoryStatus:
    _factory_token: InitVar[object]
    scalar_curvature_executed: bool
    stationarity_complete_all_passed: bool
    registered_direction_curvatures_match: bool
    mathematical_scalar_hessian_match: bool
    mathematical_scalar_curvature_match: bool
    static_hessian_authority_promoted: bool
    promotion_eligible: bool
    authority: str

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _SCALAR_CURVATURE_FACTORY_TOKEN:
            raise TypeError("scalar-curvature status requires the private factory token")
        if self.scalar_curvature_executed is not True:
            raise ValueError("scalar-curvature execution status changed")
        if self.stationarity_complete_all_passed is not True:
            raise ValueError("stationarity-complete execution status changed")
        if self.registered_direction_curvatures_match is not True:
            raise ValueError("registered direction curvature status changed")
        if self.mathematical_scalar_curvature_match is not self.mathematical_scalar_hessian_match:
            raise ValueError("legacy mathematical match alias must mean whole-Hessian match")
        expected_authority = (
            "raw_mathematical_scalar_hessian_match"
            if self.mathematical_scalar_hessian_match
            else "registered_raw_direction_curvatures_only"
        )
        if (
            self.static_hessian_authority_promoted is not False
            or self.promotion_eligible is not False
            or self.authority != expected_authority
        ):
            raise ValueError("scalar-curvature status authority or promotion semantics changed")


@dataclass(frozen=True, slots=True)
class TDHFScalarCurvatureCertificate:
    _factory_token: InitVar[object]
    api_version: str
    status: TDHFScalarCurvatureFactoryStatus
    approval: TDHFScalarCurvatureApproval
    approval_fingerprint: str
    sector_fingerprint: str
    sector_source_fingerprint: str
    source_projector_fingerprint: str
    tangent_basis_fingerprint: str
    plus_tangent_fingerprints: tuple[str, ...]
    minus_tangent_fingerprints: tuple[str, ...]
    interaction_fingerprint: str
    plus_pairs_fingerprint: str
    minus_pairs_fingerprint: str
    h_plus_fingerprint: str
    functional_manifest: TDHFScalarFunctionalManifest
    functional_manifest_fingerprint: str
    callback: TDHFScalarCallbackProvenance
    callback_provenance_fingerprint: str
    convention: TDHFEnergyConvention
    convention_fingerprint: str
    ladder_fingerprint: str
    stationarity_direction_inventory_fingerprint: str
    direction_inventory_fingerprint: str
    stationarity_evidence: tuple[TDHFScalarStationarityDirectionEvidence, ...]
    direction_evidence: tuple[TDHFScalarCurvatureDirectionEvidence, ...]
    reconstruction_evidence: tuple[TDHFScalarHessianReconstructionEvidence, ...]
    reconstructed_hessian_fingerprint: str | None
    reconstructed_hessian_max_abs_residual: float | None
    reconstructed_hessian_bound: float | None
    scalar_curvature_executed: bool
    stationarity_complete_all_passed: bool
    registered_direction_curvatures_match: bool
    mathematical_scalar_hessian_match: bool
    mathematical_scalar_curvature_match: bool
    static_hessian_authority_promoted: bool
    promotion_eligible: bool
    normalization_physics_certified: bool
    normalization_statement: str
    trusted_callback: bool
    hostile_provider_proof: bool

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _SCALAR_CURVATURE_FACTORY_TOKEN:
            raise TypeError("scalar-curvature certificate requires the private factory token")
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        if self.api_version != "tdhf_scalar_curvature.v4":
            raise ValueError("scalar-curvature certificate API version changed")
        if type(self.status) is not TDHFScalarCurvatureFactoryStatus:
            raise TypeError("certificate status must be the exact factory status type")
        if type(self.approval) is not TDHFScalarCurvatureApproval:
            raise TypeError("certificate approval has the wrong exact type")
        if self.approval_fingerprint != self.approval.fingerprint:
            raise ValueError("certificate detached approval fingerprint mismatch")
        for name in (
            "sector_fingerprint",
            "sector_source_fingerprint",
            "source_projector_fingerprint",
            "tangent_basis_fingerprint",
            "interaction_fingerprint",
            "plus_pairs_fingerprint",
            "minus_pairs_fingerprint",
            "h_plus_fingerprint",
            "functional_manifest_fingerprint",
            "callback_provenance_fingerprint",
            "convention_fingerprint",
            "ladder_fingerprint",
            "stationarity_direction_inventory_fingerprint",
            "direction_inventory_fingerprint",
        ):
            _validate_fingerprint(name, getattr(self, name))
        approval_bindings = {
            "sector_fingerprint": self.sector_fingerprint,
            "sector_source_fingerprint": self.sector_source_fingerprint,
            "source_projector_fingerprint": self.source_projector_fingerprint,
            "tangent_basis_fingerprint": self.tangent_basis_fingerprint,
            "plus_tangent_fingerprints": self.plus_tangent_fingerprints,
            "minus_tangent_fingerprints": self.minus_tangent_fingerprints,
            "interaction_fingerprint": self.interaction_fingerprint,
            "plus_pairs_fingerprint": self.plus_pairs_fingerprint,
            "minus_pairs_fingerprint": self.minus_pairs_fingerprint,
            "h_plus_fingerprint": self.h_plus_fingerprint,
            "functional_manifest_fingerprint": self.functional_manifest_fingerprint,
            "callback_provenance_fingerprint": self.callback_provenance_fingerprint,
            "convention_fingerprint": self.convention_fingerprint,
            "ladder_fingerprint": self.ladder_fingerprint,
            "stationarity_direction_inventory_fingerprint": (
                self.stationarity_direction_inventory_fingerprint
            ),
            "direction_inventory_fingerprint": self.direction_inventory_fingerprint,
        }
        for name, value in approval_bindings.items():
            if getattr(self.approval, name) != value:
                raise ValueError(f"certificate no longer matches approval field {name}")
        if self.functional_manifest != self.approval.functional_manifest:
            raise ValueError("certificate must retain the exact approved functional manifest")
        if self.callback != self.approval.callback:
            raise ValueError("certificate callback snapshot differs from approval")
        if self.convention != self.approval.convention:
            raise ValueError("certificate must retain the exact approved energy convention")
        if type(self.stationarity_evidence) is not tuple or not self.stationarity_evidence:
            raise TypeError("certificate requires nonempty stationarity evidence")
        if any(
            type(item) is not TDHFScalarStationarityDirectionEvidence
            for item in self.stationarity_evidence
        ):
            raise TypeError("certificate stationarity evidence has a non-exact nested type")
        if tuple(item.label for item in self.stationarity_evidence) != (
            self.approval.stationarity_direction_labels
        ):
            raise ValueError("certificate stationarity evidence order differs from approval")
        if tuple(item.direction_fingerprint for item in self.stationarity_evidence) != (
            self.approval.stationarity_direction_fingerprints
        ):
            raise ValueError(
                "certificate stationarity evidence fingerprints differ from approval"
            )
        for direction in self.stationarity_evidence:
            if tuple(item.step for item in direction.steps) != (
                self.approval.step_ladder.steps
            ):
                raise ValueError("stationarity step inventory differs from approval")
        expected_stationarity_complete = bool(
            self.approval.canonical_stationarity_complete_inventory
            and len(self.stationarity_evidence)
            == len(self.approval.stationarity_directions)
            and all(
                item.all_registered_steps_passed for item in self.stationarity_evidence
            )
        )
        if (
            self.stationarity_complete_all_passed
            is not expected_stationarity_complete
            or not expected_stationarity_complete
        ):
            raise ValueError("certificate lacks stationarity-complete all-pass evidence")
        if type(self.direction_evidence) is not tuple or not self.direction_evidence:
            raise TypeError("certificate requires nonempty curvature direction evidence")
        if any(
            type(item) is not TDHFScalarCurvatureDirectionEvidence
            for item in self.direction_evidence
        ):
            raise TypeError("certificate direction evidence has a non-exact nested type")
        if tuple(item.label for item in self.direction_evidence) != self.approval.direction_labels:
            raise ValueError("certificate direction evidence order differs from approval")
        if tuple(item.direction_fingerprint for item in self.direction_evidence) != (
            self.approval.direction_fingerprints
        ):
            raise ValueError("certificate direction evidence fingerprints differ from approval")
        denominator = self.convention.denominator
        for direction in self.direction_evidence:
            if direction.raw_target_curvature != 2.0 * direction.target_quadratic_form:
                raise ValueError("raw target curvature lost the mandatory factor two")
            if direction.reported_target_curvature != direction.raw_target_curvature / denominator:
                raise ValueError("reported target curvature is not raw target/denominator")
            if tuple(item.step for item in direction.steps) != self.approval.step_ladder.steps:
                raise ValueError("certificate step inventory differs from approval")
            for step in direction.steps:
                if step.raw_target_curvature != direction.raw_target_curvature:
                    raise ValueError("step and direction raw target curvatures disagree")
                if step.reported_target_curvature != direction.reported_target_curvature:
                    raise ValueError("step and direction reported target curvatures disagree")
                if step.reported_second_derivative != step.raw_second_derivative / denominator:
                    raise ValueError("reported callback curvature is not raw/denominator")
        curvature_complete_inventory = self.approval.canonical_complete_inventory
        if curvature_complete_inventory:
            if len(self.reconstruction_evidence) != len(self.approval.step_ladder.steps):
                raise ValueError("complete inventory requires one reconstruction at every step")
            if any(
                type(item) is not TDHFScalarHessianReconstructionEvidence
                for item in self.reconstruction_evidence
            ):
                raise TypeError("certificate reconstruction evidence has a non-exact type")
            if tuple(item.step for item in self.reconstruction_evidence) != (
                self.approval.step_ladder.steps
            ):
                raise ValueError("reconstruction step inventory differs from approval")
            finest = self.reconstruction_evidence[-1]
            if (
                self.reconstructed_hessian_fingerprint
                != finest.reconstructed_hessian_fingerprint
                or self.reconstructed_hessian_max_abs_residual != finest.max_abs_residual
                or self.reconstructed_hessian_bound != finest.matrix_bound
            ):
                raise ValueError("stored reconstructed-matrix summary differs from finest step")
        else:
            if self.reconstruction_evidence:
                raise ValueError("incomplete directions cannot carry whole-Hessian reconstruction")
            if any(
                value is not None
                for value in (
                    self.reconstructed_hessian_fingerprint,
                    self.reconstructed_hessian_max_abs_residual,
                    self.reconstructed_hessian_bound,
                )
            ):
                raise ValueError("incomplete directions cannot carry a reconstructed-matrix claim")
        curvature_reconstruction_complete = bool(
            curvature_complete_inventory
            and len(self.reconstruction_evidence)
            == len(self.approval.step_ladder.steps)
            and all(item.reconstruction_passed for item in self.reconstruction_evidence)
        )
        whole_match = bool(
            expected_stationarity_complete and curvature_reconstruction_complete
        )
        top = (
            self.scalar_curvature_executed,
            self.stationarity_complete_all_passed,
            self.registered_direction_curvatures_match,
            self.mathematical_scalar_hessian_match,
            self.mathematical_scalar_curvature_match,
            self.static_hessian_authority_promoted,
            self.promotion_eligible,
        )
        status = (
            self.status.scalar_curvature_executed,
            self.status.stationarity_complete_all_passed,
            self.status.registered_direction_curvatures_match,
            self.status.mathematical_scalar_hessian_match,
            self.status.mathematical_scalar_curvature_match,
            self.status.static_hessian_authority_promoted,
            self.status.promotion_eligible,
        )
        expected = (True, True, True, whole_match, whole_match, False, False)
        if top != status or top != expected:
            raise ValueError("certificate top-level and status conclusions disagree")
        if (
            self.normalization_physics_certified is not False
            or self.normalization_statement != _NORMALIZATION_STATEMENT
            or self.convention.normalization_physics_certified is not False
        ):
            raise ValueError("certificate cannot claim normalization physics authority")
        if self.trusted_callback is not True or self.hostile_provider_proof is not False:
            raise ValueError("certificate callback trust limitations changed")

    @property
    def fingerprint(self) -> str:
        self._validate_consistency()
        return _fingerprint(self)


# Short aliases keep the immutable public API readable.
TransitionTangentBasis = TDHFTransitionTangentBasis
PhysicalDirection = TDHFPhysicalDirection
EnergyConvention = TDHFEnergyConvention
ScalarCurvatureTolerances = TDHFScalarCurvatureTolerances
ScalarCurvatureStepLadder = TDHFScalarCurvatureStepLadder
ScalarCurvatureApproval = TDHFScalarCurvatureApproval
ScalarCurvatureCertificate = TDHFScalarCurvatureCertificate


def _exact_projector_path(
    basis: TDHFTransitionTangentBasis,
    direction: TDHFPhysicalDirection,
    parameter: float,
) -> tuple[Array, Array]:
    if type(basis) is not TDHFTransitionTangentBasis:
        raise TypeError("basis must be the exact transition tangent basis type")
    if type(direction) is not TDHFPhysicalDirection:
        raise TypeError("direction must be the exact physical direction type")
    if direction.x.shape != (len(basis.plus_tangents),):
        raise ValueError("x dimension does not match the ordered +q tangent basis")
    if direction.y.shape != (len(basis.minus_tangents),):
        raise ValueError("y dimension does not match the ordered -q tangent basis")
    z = np.zeros_like(basis.source_projector)
    for coefficient, tangent in zip(direction.x, basis.plus_tangents):
        z += coefficient * tangent
    for coefficient, tangent in zip(direction.y, basis.minus_tangents):
        z += coefficient * tangent
    generator = z - z.conj().T
    unitary = expm(float(parameter) * generator)
    projector = unitary @ basis.source_projector @ unitary.conj().T
    return projector, generator


def exact_tdhf_projector_path(
    basis: TDHFTransitionTangentBasis,
    direction: TDHFPhysicalDirection,
    parameter: float,
) -> Array:
    """Return ``exp(tK) P0 exp(-tK)`` for diagnostics and exact oracles."""

    return _exact_projector_path(basis, direction, parameter)[0]


def _projector_residuals(
    projector: Array, reference_trace: complex
) -> tuple[float, float, float]:
    return (
        float(np.linalg.norm(projector - projector.conj().T, ord="fro")),
        float(np.linalg.norm(projector @ projector - projector, ord="fro")),
        float(abs(np.trace(projector) - reference_trace)),
    )


def _real_energy(callback: Callable[[Array], float], projector: Array) -> float:
    owner = np.array(projector, dtype=np.complex128, copy=True, order="C")
    owner.setflags(write=False)
    callback_input = owner.view()
    callback_input.setflags(write=False)
    before = fingerprint_tdhf_matrix(owner)
    try:
        value = callback(callback_input)
    except Exception as error:
        after = fingerprint_tdhf_matrix(owner)
        if after != before or "read-only" in str(error).lower() or "writeable" in str(error).lower():
            raise ValueError("energy callback attempted to mutate its read-only projector input") from error
        raise
    after = fingerprint_tdhf_matrix(owner)
    if after != before:
        raise ValueError("energy callback mutated its projector input")
    scalar = complex(np.asarray(value).item())
    if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
        raise ValueError("energy callback returned a non-finite scalar")
    if abs(scalar.imag) > 64.0 * np.finfo(float).eps * max(1.0, abs(scalar.real)):
        raise ValueError("energy callback must return a real expectation value")
    return float(scalar.real)


def _reconstruct_hermitian_from_canonical_quadratics(
    quadratics: Sequence[float], dimension: int
) -> Array:
    if len(quadratics) != dimension * dimension:
        raise ValueError("Hermitian reconstruction requires the exact canonical d^2 quadratics")
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    diagonal = np.asarray(quadratics[:dimension], dtype=np.float64)
    result[np.diag_indices(dimension)] = diagonal
    cursor = dimension
    for left in range(dimension):
        for right in range(left + 1, dimension):
            real_mix = float(quadratics[cursor])
            imag_mix = float(quadratics[cursor + 1])
            cursor += 2
            average = 0.5 * (diagonal[left] + diagonal[right])
            real_part = real_mix - average
            # For v=(e_i+i e_j)/sqrt(2), q=average-Im(H_ij).
            imag_part = average - imag_mix
            value = complex(real_part, imag_part)
            result[left, right] = value
            result[right, left] = value.conjugate()
    if cursor != len(quadratics):
        raise RuntimeError("internal canonical reconstruction inventory mismatch")
    return _readonly_complex(result, ndim=2)


def _rederive_approval(
    approval: TDHFScalarCurvatureApproval,
    sector: TDHFGenericSignedQSector,
    tangent_basis: TDHFTransitionTangentBasis,
    energy_callback: Callable[[Array], float],
    functional_manifest: TDHFScalarFunctionalManifest,
) -> tuple[Array, TDHFScalarCallbackProvenance]:
    """Re-derive every approval binding before the first callback call."""

    approval._validate_consistency()
    if type(functional_manifest) is not TDHFScalarFunctionalManifest:
        raise TypeError("functional_manifest must be the exact manifest type")
    h_plus, plus_pairs_fingerprint, minus_pairs_fingerprint = (
        _validate_sector_basis_bindings(
            sector, tangent_basis, approval.interaction_fingerprint
        )
    )
    callback = snapshot_tdhf_scalar_callback(energy_callback)
    current = {
        "sector_fingerprint": fingerprint_tdhf_sector(sector),
        "sector_source_fingerprint": sector.source_fingerprint,
        "source_projector_fingerprint": tangent_basis.source_projector_fingerprint,
        "tangent_basis_fingerprint": tangent_basis.fingerprint,
        "plus_tangent_fingerprints": tangent_basis.plus_tangent_fingerprints,
        "minus_tangent_fingerprints": tangent_basis.minus_tangent_fingerprints,
        "interaction_fingerprint": sector.interaction_fingerprint,
        "plus_pairs_fingerprint": plus_pairs_fingerprint,
        "minus_pairs_fingerprint": minus_pairs_fingerprint,
        "h_plus_fingerprint": fingerprint_tdhf_matrix(h_plus),
        "functional_manifest_fingerprint": functional_manifest.fingerprint,
        "callback_provenance_fingerprint": callback.fingerprint,
        "convention_fingerprint": approval.convention.fingerprint,
        "stationarity_direction_inventory_fingerprint": (
            _direction_inventory_fingerprint(
                approval.stationarity_directions,
                len(tangent_basis.plus_tangents),
                len(tangent_basis.minus_tangents),
                purpose="stationarity",
            )
        ),
        "direction_inventory_fingerprint": _direction_inventory_fingerprint(
            approval.directions,
            len(tangent_basis.plus_tangents),
            len(tangent_basis.minus_tangents),
            purpose="curvature",
        ),
        "ladder_fingerprint": approval.step_ladder.fingerprint,
    }
    for name, value in current.items():
        if getattr(approval, name) != value:
            raise ValueError(f"detached scalar-curvature approval is stale: {name} mismatch")
    if functional_manifest != approval.functional_manifest:
        raise ValueError("detached scalar-curvature approval has a cross-functional manifest")
    if callback != approval.callback:
        raise ValueError("energy callback source/code snapshot differs from detached approval")
    if functional_manifest.implementation_fingerprint != callback.fingerprint:
        raise ValueError("functional manifest implementation differs from current callback")
    return h_plus, callback


def certify_tdhf_scalar_curvature(
    *,
    approval: TDHFScalarCurvatureApproval,
    sector: TDHFGenericSignedQSector,
    tangent_basis: TDHFTransitionTangentBasis,
    energy_callback: Callable[[Array], float],
    functional_manifest: TDHFScalarFunctionalManifest,
) -> TDHFScalarCurvatureCertificate:
    """Certify complete stationarity and approved raw curvature directions.

    Whole-Hessian authority requires both canonical ``2d`` stationarity and
    canonical ``d^2`` curvature reconstruction at every registered step.
    """

    if type(approval) is not TDHFScalarCurvatureApproval:
        raise TypeError("certify requires the exact detached TDHFScalarCurvatureApproval")
    if type(sector) is not TDHFGenericSignedQSector:
        raise TypeError("sector must be the exact TDHFGenericSignedQSector type")
    if type(tangent_basis) is not TDHFTransitionTangentBasis:
        raise TypeError("tangent_basis must be the exact tangent basis type")
    if not callable(energy_callback):
        raise TypeError("energy_callback must be callable")

    # This is the authority gate: all checks complete before _real_energy is reachable.
    h_plus, callback_before = _rederive_approval(
        approval, sector, tangent_basis, energy_callback, functional_manifest
    )
    tolerance = approval.step_ladder.tolerances
    if not np.allclose(
        h_plus, h_plus.conj().T, atol=tolerance.projector_tolerance, rtol=0.0
    ):
        raise ValueError("the exact generic sector H_plus is not Hermitian")

    eps = np.finfo(float).eps
    reference_trace = np.trace(tangent_basis.source_projector)
    stationarity_evidence: list[TDHFScalarStationarityDirectionEvidence] = []
    stationarity_failures: list[str] = []
    denominator = approval.convention.denominator

    # Stationarity is real-linear on the complex tangent space.  This separate
    # canonical {e_i, i e_i} pass is the authority gate; the d^2 curvature
    # inventory below is not used as a surrogate span check.
    for direction in approval.stationarity_directions:
        _, generator = _exact_projector_path(tangent_basis, direction, 0.0)
        antihermiticity = float(
            np.linalg.norm(generator + generator.conj().T, ord="fro")
        )
        if antihermiticity > tolerance.projector_tolerance:
            raise ValueError(
                f"stationarity direction {direction.label!r} generator is not anti-Hermitian"
            )

        step_evidence: list[TDHFScalarStationarityStepEvidence] = []
        for step in approval.step_ladder.steps:
            residuals: list[tuple[float, float, float]] = []
            energies: list[float] = []
            for multiplier in (-2.0, -1.0, 0.0, 1.0, 2.0):
                projector, _ = _exact_projector_path(
                    tangent_basis, direction, multiplier * step
                )
                projector_residual = _projector_residuals(projector, reference_trace)
                if max(projector_residual) > tolerance.projector_tolerance:
                    raise ValueError(
                        "exact stationarity projector path failed at "
                        f"direction={direction.label}, h={step}, "
                        f"multiplier={multiplier}: residuals={projector_residual}"
                    )
                residuals.append(projector_residual)
                energies.append(_real_energy(energy_callback, projector))
            fm2, fm1, f0, fp1, fp2 = energies
            raw_first = (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * step)
            raw_second_probe = (
                -fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2
            ) / (12.0 * step * step)
            energy_scale = max(1.0, *(abs(value) for value in energies))
            stationarity_roundoff_allowance = (
                tolerance.roundoff_multiplier * eps * energy_scale / step
            )
            curvature_roundoff_probe = (
                tolerance.roundoff_multiplier * eps * energy_scale / (step * step)
            )
            if (
                stationarity_roundoff_allowance
                > TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM
                or curvature_roundoff_probe
                > TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_CURVATURE_ALLOWANCE_MAXIMUM
            ):
                raise ValueError(
                    "scalar-stationarity certification rejected a vacuous derived "
                    f"roundoff allowance at direction={direction.label}, h={step}: "
                    f"stationarity {stationarity_roundoff_allowance:.6e}/"
                    f"{TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM:.6e}, "
                    f"curvature probe {curvature_roundoff_probe:.6e}/"
                    f"{TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_CURVATURE_ALLOWANCE_MAXIMUM:.6e}"
                )
            stationarity_scale = max(1.0, abs(raw_second_probe) * step)
            stationarity_bound = (
                tolerance.stationarity_absolute
                + tolerance.stationarity_relative * stationarity_scale
                + stationarity_roundoff_allowance
            )
            stationarity_residual = abs(raw_first)
            stationarity_passed = stationarity_residual <= stationarity_bound
            if not stationarity_passed:
                stationarity_failures.append(
                    f"{direction.label}@h={step}: stationarity "
                    f"{stationarity_residual:.6e}/{stationarity_bound:.6e}"
                )
            else:
                step_evidence.append(
                    TDHFScalarStationarityStepEvidence(
                        step=step,
                        raw_energies_at_minus2_minus1_zero_plus1_plus2=tuple(energies),  # type: ignore[arg-type]
                        raw_first_derivative=raw_first,
                        raw_second_derivative_scale_probe=raw_second_probe,
                        stationarity_residual=stationarity_residual,
                        stationarity_roundoff_allowance=(
                            stationarity_roundoff_allowance
                        ),
                        curvature_roundoff_nonvacuity_probe=curvature_roundoff_probe,
                        stationarity_bound=stationarity_bound,
                        projector_max_hermiticity_residual=max(
                            item[0] for item in residuals
                        ),
                        projector_max_idempotency_residual=max(
                            item[1] for item in residuals
                        ),
                        projector_max_trace_residual=max(item[2] for item in residuals),
                        stationarity_passed=True,
                    )
                )
        if len(step_evidence) == len(approval.step_ladder.steps):
            stationarity_evidence.append(
                TDHFScalarStationarityDirectionEvidence(
                    label=direction.label,
                    direction_fingerprint=direction.fingerprint,
                    generator_antihermiticity_residual=antihermiticity,
                    steps=tuple(step_evidence),
                    raw_first_derivative_plateau=tuple(
                        item.raw_first_derivative for item in step_evidence
                    ),
                    all_registered_steps_passed=True,
                )
            )

    if stationarity_failures:
        raise ValueError(
            "scalar-stationarity certification failed; all 2d canonical directions "
            "and all steps are mandatory: "
            + "; ".join(stationarity_failures)
        )

    direction_evidence: list[TDHFScalarCurvatureDirectionEvidence] = []
    failures: list[str] = []
    for direction in approval.directions:
        vector = direction.vector
        quadratic_complex = complex(np.vdot(vector, h_plus @ vector))
        if abs(quadratic_complex.imag) > 64.0 * eps * max(
            1.0, abs(quadratic_complex.real)
        ):
            raise ValueError(f"direction {direction.label!r} has a non-real H_plus form")
        quadratic = float(quadratic_complex.real)
        raw_target = 2.0 * quadratic
        reported_target = raw_target / denominator
        _, generator = _exact_projector_path(tangent_basis, direction, 0.0)
        antihermiticity = float(np.linalg.norm(generator + generator.conj().T, ord="fro"))
        if antihermiticity > tolerance.projector_tolerance:
            raise ValueError(f"direction {direction.label!r} generator is not anti-Hermitian")

        step_evidence: list[TDHFScalarCurvatureStepEvidence] = []
        for step in approval.step_ladder.steps:
            residuals: list[tuple[float, float, float]] = []
            energies: list[float] = []
            for multiplier in (-2.0, -1.0, 0.0, 1.0, 2.0):
                projector, _ = _exact_projector_path(
                    tangent_basis, direction, multiplier * step
                )
                projector_residual = _projector_residuals(projector, reference_trace)
                if max(projector_residual) > tolerance.projector_tolerance:
                    raise ValueError(
                        f"exact projector path failed at direction={direction.label}, "
                        f"h={step}, multiplier={multiplier}: residuals={projector_residual}"
                    )
                residuals.append(projector_residual)
                energies.append(_real_energy(energy_callback, projector))
            fm2, fm1, f0, fp1, fp2 = energies
            raw_first = (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * step)
            raw_second = (
                -fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2
            ) / (12.0 * step * step)
            reported_second = raw_second / denominator
            energy_scale = max(1.0, *(abs(value) for value in energies))
            stationarity_roundoff_allowance = (
                tolerance.roundoff_multiplier * eps * energy_scale / step
            )
            curvature_roundoff_allowance = (
                tolerance.roundoff_multiplier * eps * energy_scale / (step * step)
            )
            if (
                stationarity_roundoff_allowance
                > TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM
                or curvature_roundoff_allowance
                > TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_CURVATURE_ALLOWANCE_MAXIMUM
            ):
                raise ValueError(
                    "scalar-curvature certification rejected a vacuous derived roundoff "
                    f"allowance at direction={direction.label}, h={step}: stationarity "
                    f"{stationarity_roundoff_allowance:.6e}/"
                    f"{TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM:.6e}, "
                    f"curvature {curvature_roundoff_allowance:.6e}/"
                    f"{TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_CURVATURE_ALLOWANCE_MAXIMUM:.6e}"
                )
            stationarity_scale = max(1.0, abs(raw_second) * step)
            stationarity_bound = (
                tolerance.stationarity_absolute
                + tolerance.stationarity_relative * stationarity_scale
                + stationarity_roundoff_allowance
            )
            curvature_scale = max(1.0, abs(raw_target), abs(raw_second))
            curvature_bound = (
                tolerance.curvature_absolute
                + tolerance.curvature_relative * curvature_scale
                + curvature_roundoff_allowance
            )
            stationarity_residual = abs(raw_first)
            curvature_residual = abs(raw_second - raw_target)
            stationarity_passed = stationarity_residual <= stationarity_bound
            curvature_passed = curvature_residual <= curvature_bound
            if not stationarity_passed or not curvature_passed:
                failures.append(
                    f"{direction.label}@h={step}: stationarity "
                    f"{stationarity_residual:.6e}/{stationarity_bound:.6e}, raw curvature "
                    f"{curvature_residual:.6e}/{curvature_bound:.6e}"
                )
            if stationarity_passed and curvature_passed:
                step_evidence.append(
                    TDHFScalarCurvatureStepEvidence(
                        step=step,
                        raw_energies_at_minus2_minus1_zero_plus1_plus2=tuple(energies),  # type: ignore[arg-type]
                        raw_first_derivative=raw_first,
                        raw_second_derivative=raw_second,
                        measured_raw_quadratic=raw_second / 2.0,
                        reported_second_derivative=reported_second,
                        raw_target_curvature=raw_target,
                        reported_target_curvature=reported_target,
                        stationarity_residual=stationarity_residual,
                        curvature_residual=curvature_residual,
                        stationarity_roundoff_allowance=stationarity_roundoff_allowance,
                        curvature_roundoff_allowance=curvature_roundoff_allowance,
                        stationarity_bound=stationarity_bound,
                        curvature_bound=curvature_bound,
                        projector_max_hermiticity_residual=max(item[0] for item in residuals),
                        projector_max_idempotency_residual=max(item[1] for item in residuals),
                        projector_max_trace_residual=max(item[2] for item in residuals),
                        stationarity_passed=True,
                        curvature_passed=True,
                    )
                )
        if len(step_evidence) == len(approval.step_ladder.steps):
            direction_evidence.append(
                TDHFScalarCurvatureDirectionEvidence(
                    label=direction.label,
                    direction_fingerprint=direction.fingerprint,
                    target_quadratic_form=quadratic,
                    raw_target_curvature=raw_target,
                    reported_target_curvature=reported_target,
                    generator_antihermiticity_residual=antihermiticity,
                    steps=tuple(step_evidence),
                    raw_curvature_plateau=tuple(
                        item.raw_second_derivative for item in step_evidence
                    ),
                    reported_curvature_plateau=tuple(
                        item.reported_second_derivative for item in step_evidence
                    ),
                    all_registered_steps_passed=True,
                )
            )

    if failures:
        raise ValueError(
            "scalar-curvature certification failed; all steps are mandatory: "
            + "; ".join(failures)
        )

    reconstruction_evidence: list[TDHFScalarHessianReconstructionEvidence] = []
    if approval.canonical_complete_inventory:
        dimension = h_plus.shape[0]
        for step_index, step in enumerate(approval.step_ladder.steps):
            measured = tuple(
                item.steps[step_index].measured_raw_quadratic
                for item in direction_evidence
            )
            reconstructed = _reconstruct_hermitian_from_canonical_quadratics(
                measured, dimension
            )
            residual = float(np.max(np.abs(reconstructed - h_plus)))
            scale = max(
                1.0,
                float(np.max(np.abs(reconstructed))),
                float(np.max(np.abs(h_plus))),
            )
            bound = tolerance.matrix_absolute + tolerance.matrix_relative * scale
            if residual > bound:
                raise ValueError(
                    "scalar-Hessian matrix reconstruction failed at "
                    f"h={step}: max-abs residual {residual:.6e}/{bound:.6e}"
                )
            reconstruction_evidence.append(
                TDHFScalarHessianReconstructionEvidence(
                    step=step,
                    reconstructed_hessian=reconstructed,
                    reconstructed_hessian_fingerprint=fingerprint_tdhf_matrix(reconstructed),
                    max_abs_residual=residual,
                    matrix_bound=bound,
                    reconstruction_passed=True,
                )
            )

    callback_after = snapshot_tdhf_scalar_callback(energy_callback)
    if callback_after != callback_before:
        raise ValueError("energy callback file/source/code snapshot changed during certification")
    if functional_manifest.implementation_fingerprint != callback_after.fingerprint:
        raise ValueError("functional manifest implementation changed during certification")

    stationarity_complete = bool(
        approval.canonical_stationarity_complete_inventory
        and len(stationarity_evidence) == len(approval.stationarity_directions)
        and all(item.all_registered_steps_passed for item in stationarity_evidence)
    )
    curvature_reconstruction_complete = bool(
        approval.canonical_complete_inventory
        and len(reconstruction_evidence) == len(approval.step_ladder.steps)
        and all(item.reconstruction_passed for item in reconstruction_evidence)
    )
    whole = bool(stationarity_complete and curvature_reconstruction_complete)
    authority = (
        "raw_mathematical_scalar_hessian_match"
        if whole
        else "registered_raw_direction_curvatures_only"
    )
    status = TDHFScalarCurvatureFactoryStatus(
        _factory_token=_SCALAR_CURVATURE_FACTORY_TOKEN,
        scalar_curvature_executed=True,
        stationarity_complete_all_passed=stationarity_complete,
        registered_direction_curvatures_match=True,
        mathematical_scalar_hessian_match=whole,
        mathematical_scalar_curvature_match=whole,
        static_hessian_authority_promoted=False,
        promotion_eligible=False,
        authority=authority,
    )
    finest = reconstruction_evidence[-1] if reconstruction_evidence else None
    return TDHFScalarCurvatureCertificate(
        _factory_token=_SCALAR_CURVATURE_FACTORY_TOKEN,
        api_version="tdhf_scalar_curvature.v4",
        status=status,
        approval=approval,
        approval_fingerprint=approval.fingerprint,
        sector_fingerprint=approval.sector_fingerprint,
        sector_source_fingerprint=approval.sector_source_fingerprint,
        source_projector_fingerprint=approval.source_projector_fingerprint,
        tangent_basis_fingerprint=approval.tangent_basis_fingerprint,
        plus_tangent_fingerprints=approval.plus_tangent_fingerprints,
        minus_tangent_fingerprints=approval.minus_tangent_fingerprints,
        interaction_fingerprint=approval.interaction_fingerprint,
        plus_pairs_fingerprint=approval.plus_pairs_fingerprint,
        minus_pairs_fingerprint=approval.minus_pairs_fingerprint,
        h_plus_fingerprint=approval.h_plus_fingerprint,
        functional_manifest=approval.functional_manifest,
        functional_manifest_fingerprint=approval.functional_manifest_fingerprint,
        callback=callback_before,
        callback_provenance_fingerprint=callback_before.fingerprint,
        convention=approval.convention,
        convention_fingerprint=approval.convention_fingerprint,
        ladder_fingerprint=approval.ladder_fingerprint,
        stationarity_direction_inventory_fingerprint=(
            approval.stationarity_direction_inventory_fingerprint
        ),
        direction_inventory_fingerprint=approval.direction_inventory_fingerprint,
        stationarity_evidence=tuple(stationarity_evidence),
        direction_evidence=tuple(direction_evidence),
        reconstruction_evidence=tuple(reconstruction_evidence),
        reconstructed_hessian_fingerprint=(
            None if finest is None else finest.reconstructed_hessian_fingerprint
        ),
        reconstructed_hessian_max_abs_residual=(
            None if finest is None else finest.max_abs_residual
        ),
        reconstructed_hessian_bound=None if finest is None else finest.matrix_bound,
        scalar_curvature_executed=True,
        stationarity_complete_all_passed=stationarity_complete,
        registered_direction_curvatures_match=True,
        mathematical_scalar_hessian_match=whole,
        mathematical_scalar_curvature_match=whole,
        static_hessian_authority_promoted=False,
        promotion_eligible=False,
        normalization_physics_certified=False,
        normalization_statement=_NORMALIZATION_STATEMENT,
        trusted_callback=True,
        hostile_provider_proof=False,
    )


__all__ = [
    "EnergyConvention",
    "PhysicalDirection",
    "ScalarCurvatureApproval",
    "ScalarCurvatureCertificate",
    "ScalarCurvatureStepLadder",
    "ScalarCurvatureTolerances",
    "TDHFEnergyConvention",
    "TDHFPhysicalDirection",
    "TDHFScalarCallbackProvenance",
    "TDHFScalarCurvatureApproval",
    "TDHFScalarCurvatureCertificate",
    "TDHFScalarCurvatureDirectionEvidence",
    "TDHFScalarCurvatureFactoryStatus",
    "TDHFScalarCurvatureStepEvidence",
    "TDHFScalarCurvatureStepLadder",
    "TDHFScalarCurvatureTolerances",
    "TDHFScalarFunctionalManifest",
    "TDHFScalarHessianReconstructionEvidence",
    "TDHFScalarStationarityDirectionEvidence",
    "TDHFScalarStationarityStepEvidence",
    "TDHFTransitionTangentBasis",
    "TDHF_SCALAR_CURVATURE_V1_CURVATURE_ABSOLUTE_MAXIMUM",
    "TDHF_SCALAR_CURVATURE_V1_CURVATURE_RELATIVE_MAXIMUM",
    "TDHF_SCALAR_CURVATURE_V1_DIRECTION_TOLERANCE_MAXIMUM",
    "TDHF_SCALAR_CURVATURE_V1_MATRIX_ABSOLUTE_MAXIMUM",
    "TDHF_SCALAR_CURVATURE_V1_MATRIX_RELATIVE_MAXIMUM",
    "TDHF_SCALAR_CURVATURE_V1_PROJECTOR_TOLERANCE_MAXIMUM",
    "TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_CURVATURE_ALLOWANCE_MAXIMUM",
    "TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER",
    "TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER_MAXIMUM",
    "TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM",
    "TDHF_SCALAR_CURVATURE_V1_STATIONARITY_ABSOLUTE_MAXIMUM",
    "TDHF_SCALAR_CURVATURE_V1_STATIONARITY_RELATIVE_MAXIMUM",
    "TDHF_SCALAR_CURVATURE_V1_STEP_MAXIMUM",
    "TDHF_SCALAR_CURVATURE_V1_STEP_MINIMUM",
    "TDHF_SCALAR_CURVATURE_V1_TANGENT_TOLERANCE_MAXIMUM",
    "TransitionTangentBasis",
    "canonical_tdhf_scalar_directions",
    "canonical_tdhf_stationarity_directions",
    "certify_tdhf_scalar_curvature",
    "exact_tdhf_projector_path",
    "make_tdhf_scalar_curvature_approval",
    "make_tdhf_scalar_functional_manifest",
    "snapshot_tdhf_scalar_callback",
]
