"""Conventional dense full-projector scalar-functional ABI v1.

The mathematical boundary is a full ``(N,N)`` ``complex128`` Hermitian
projector/density ``P`` with ``P_ij=<c_j^dagger c_i>``.  For Hermitian affine
probes ``D`` this module validates the unweighted, raw-total-energy identities

``dE[P+tD]/dt = Tr(F[P]D)`` and ``dF[P+tD]/dt = dF[P,D]``.

The v1 qualification is deliberately narrower than TDHF authority.  It proves
consistency of three separately bound trusted-provider kernels on a detached,
preregistered probe inventory.  Registered-probe consistency is distinct from
``full_projector_functional_consistency``, which is true only for the exact
internally generated normalized ``N^2`` Hermitian inventory at supported small
N.  Explicit exact-projector probes record actual E/F execution; they are not a
support Boolean.  The ABI neither compares a TDHF A/B Hessian nor promotes
static-Hessian, production, or paper authority.  Source/code/dependency
snapshots and Python call tracing detect ordinary drift and direct delegation;
they are a trusted-provider boundary, not a sandbox, hostile-code proof, or
global completeness proof.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from hashlib import sha256
import inspect
import json
import marshal
import math
from pathlib import Path
import sys
from types import FunctionType, ModuleType
from typing import Any, Callable, Final, Literal, Mapping, Sequence

import numpy as np

from .tdhf_scalar_curvature import (
    TDHF_SCALAR_CURVATURE_V1_CURVATURE_ABSOLUTE_MAXIMUM,
    TDHF_SCALAR_CURVATURE_V1_CURVATURE_RELATIVE_MAXIMUM,
    TDHF_SCALAR_CURVATURE_V1_MATRIX_ABSOLUTE_MAXIMUM,
    TDHF_SCALAR_CURVATURE_V1_MATRIX_RELATIVE_MAXIMUM,
    TDHF_SCALAR_CURVATURE_V1_PROJECTOR_TOLERANCE_MAXIMUM,
    TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_CURVATURE_ALLOWANCE_MAXIMUM,
    TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER,
    TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM,
    TDHF_SCALAR_CURVATURE_V1_STATIONARITY_ABSOLUTE_MAXIMUM,
    TDHF_SCALAR_CURVATURE_V1_STATIONARITY_RELATIVE_MAXIMUM,
    TDHF_SCALAR_CURVATURE_V1_STEP_MAXIMUM,
    TDHF_SCALAR_CURVATURE_V1_STEP_MINIMUM,
)

Array = np.ndarray
KernelRole = Literal["energy", "fock", "fock_derivative"]

TDHF_FULL_PROJECTOR_SCALAR_FUNCTIONAL_API_VERSION: Final[str] = (
    "tdhf_full_projector_scalar_functional.v1"
)
TDHF_FULL_PROJECTOR_CONVENTION: Final[str] = (
    "conventional_dense_complex128_P_ij=<c_j^dagger c_i>"
)
TDHF_FULL_PROJECTOR_PAIRING: Final[str] = "raw_unweighted_full_trace_Tr(left@right)"
TDHF_FULL_PROJECTOR_ENERGY_NORMALIZATION: Final[str] = "raw_total_energy"
TDHF_FULL_PROJECTOR_SMALL_BASIS_MAXIMUM_DIMENSION: Final[int] = 8
TDHF_FULL_PROJECTOR_DIRECTION_NORMALIZATION_TOLERANCE: Final[float] = (
    64.0 * np.finfo(np.float64).eps
)
TDHF_FULL_PROJECTOR_DIRECTION_SIGNAL_MINIMUM: Final[float] = 1.0e-12
TDHF_FULL_PROJECTOR_DF_RESPONSE_MINIMUM: Final[float] = 1.0e-10

_APPROVAL_TOKEN = object()
_RECEIPT_TOKEN = object()


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
    if isinstance(value, FunctionType):
        return {
            "module": value.__module__,
            "qualname": value.__qualname__,
            "code_sha256": sha256(marshal.dumps(value.__code__)).hexdigest(),
        }
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


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a strict real scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _bounded(value: object, label: str, maximum: float, *, positive: bool = False) -> float:
    result = _finite(value, label, positive=positive)
    if result < 0.0 or result > maximum:
        raise ValueError(
            f"{label}={result} exceeds locked v1 range [0,{maximum}]; "
            "vacuous bounds are forbidden"
        )
    return result


def _readonly_exact_complex(
    value: object,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
    hermitian: bool = False,
) -> Array:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a numpy.ndarray")
    if value.dtype != np.dtype(np.complex128):
        raise TypeError(f"{label} dtype must be exactly complex128")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{label} shape must be exactly {shape}, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must contain only finite values")
    if hermitian:
        residual = float(np.max(np.abs(value - value.conj().T))) if value.size else 0.0
        scale = max(1.0, float(np.max(np.abs(value))) if value.size else 0.0)
        if residual > 64.0 * np.finfo(np.float64).eps * scale:
            raise ValueError(f"{label} must be Hermitian")
    result = np.frombuffer(value.tobytes(order="C"), dtype=np.complex128).reshape(
        value.shape
    )
    result.setflags(write=False)
    return result


def _immutable_value(value: object, label: str) -> object:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"{label} rejects object arrays")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{label} array must be finite")
        result = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(
            value.shape
        )
        result.setflags(write=False)
        return result
    if isinstance(value, np.generic):
        return _immutable_value(value.item(), label)
    if isinstance(value, tuple):
        return tuple(_immutable_value(item, label) for item in value)
    if isinstance(value, list) or isinstance(value, dict) or isinstance(value, set):
        raise TypeError(f"{label} rejects mutable containers")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if isinstance(value, complex) and not (
        math.isfinite(value.real) and math.isfinite(value.imag)
    ):
        raise ValueError(f"{label} must be finite")
    if isinstance(value, (str, int, float, complex, bool)) or value is None:
        return value
    raise TypeError(f"{label} has unsupported immutable input type {type(value).__name__}")


def _max_abs(value: object) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return 0.0
    result = float(np.max(np.abs(array)))
    if not math.isfinite(result):
        raise ValueError("nonfinite residual encountered")
    return result


def _frobenius_norm(value: object) -> float:
    result = float(np.linalg.norm(np.asarray(value), ord="fro"))
    if not math.isfinite(result):
        raise ValueError("nonfinite Frobenius norm encountered")
    return result


def _trace_pairing(left: Array, right: Array) -> complex:
    """Literal unweighted full trace; never a per-k/per-area reduction."""

    return complex(np.einsum("ij,ji->", left, right, optimize=False))


def _bound(absolute: float, relative: float, *values: object) -> float:
    return absolute + relative * max(1.0, *(_max_abs(value) for value in values))


@dataclass(frozen=True, slots=True)
class TDHFFullProjectorSpace:
    """Immutable description of one dense conventional orbital space."""

    dimension: int
    axis_sizes: tuple[int, ...]
    axis_order: tuple[str, ...]
    orbital_order_fingerprint: str
    layout_adapter_fingerprint: str
    convention: str = field(default=TDHF_FULL_PROJECTOR_CONVENTION, init=False)

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("full-projector dimension must be a positive integer")
        sizes = tuple(self.axis_sizes)
        order = tuple(self.axis_order)
        if (
            not sizes
            or any(type(value) is not int or value < 1 for value in sizes)
            or math.prod(sizes) != self.dimension
        ):
            raise ValueError("axis sizes must be positive and multiply to dimension")
        if (
            len(order) != len(sizes)
            or len(set(order)) != len(order)
            or any(type(value) is not str or not value for value in order)
        ):
            raise ValueError("axis order must contain one unique label per axis")
        object.__setattr__(self, "axis_sizes", sizes)
        object.__setattr__(self, "axis_order", order)
        _sha256(self.orbital_order_fingerprint, "orbital-order fingerprint")
        _sha256(self.layout_adapter_fingerprint, "layout-adapter fingerprint")
        if self.convention != TDHF_FULL_PROJECTOR_CONVENTION:
            raise ValueError("full-projector conventional dense ABI changed")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class TDHFScalarFunctionalInput:
    """One explicitly named immutable callback input."""

    name: str
    value: object

    def __post_init__(self) -> None:
        _text(self.name, "functional input name")
        object.__setattr__(
            self, "value", _immutable_value(self.value, f"functional input {self.name!r}")
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class TDHFScalarFunctionalInputsManifest:
    """Explicit values passed as the first argument to all three kernels."""

    entries: tuple[TDHFScalarFunctionalInput, ...]
    source_fingerprint: str
    provenance: str
    abi_version: str = field(
        default=TDHF_FULL_PROJECTOR_SCALAR_FUNCTIONAL_API_VERSION, init=False
    )

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries or any(type(item) is not TDHFScalarFunctionalInput for item in entries):
            raise TypeError("functional inputs require exact typed entries")
        names = tuple(item.name for item in entries)
        if len(set(names)) != len(names) or names != tuple(sorted(names)):
            raise ValueError("functional input names must be unique and sorted")
        arrays = [item.value for item in entries if isinstance(item.value, np.ndarray)]
        for index, left in enumerate(arrays):
            if left.flags.writeable:
                raise ValueError("functional input arrays must be read-only")
            for right in arrays[index + 1 :]:
                if np.shares_memory(left, right):
                    raise ValueError("functional input arrays must not alias")
        object.__setattr__(self, "entries", entries)
        _sha256(self.source_fingerprint, "functional-input source fingerprint")
        _text(self.provenance, "functional-input provenance")
        if self.abi_version != TDHF_FULL_PROJECTOR_SCALAR_FUNCTIONAL_API_VERSION:
            raise ValueError("functional-input ABI version changed")

    def value(self, name: str) -> object:
        for item in self.entries:
            if item.name == name:
                return item.value
        raise KeyError(name)

    def array(self, name: str) -> Array:
        value = self.value(name)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"functional input {name!r} is not an array")
        return value

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return _fingerprint(self)

    def validate_live_state(self) -> None:
        for item in self.entries:
            if isinstance(item.value, np.ndarray):
                if item.value.flags.writeable or not np.all(np.isfinite(item.value)):
                    raise ValueError(f"functional input {item.name!r} became writable/nonfinite")
        # Reconstructing exact entries checks names, aliases, and ABI again.
        TDHFScalarFunctionalInputsManifest(
            entries=self.entries,
            source_fingerprint=self.source_fingerprint,
            provenance=self.provenance,
        )


def make_tdhf_scalar_functional_inputs_manifest(
    values: Mapping[str, object], *, source_fingerprint: str, provenance: str
) -> TDHFScalarFunctionalInputsManifest:
    """Copy explicit values after rejecting caller-visible array aliases."""

    if not isinstance(values, Mapping) or not values:
        raise TypeError("functional input values must be a nonempty mapping")
    raw_arrays = [value for value in values.values() if isinstance(value, np.ndarray)]
    for index, left in enumerate(raw_arrays):
        for right in raw_arrays[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError("functional input arrays must not alias")
    entries = tuple(
        TDHFScalarFunctionalInput(name, value)
        for name, value in sorted(values.items(), key=lambda pair: pair[0])
    )
    return TDHFScalarFunctionalInputsManifest(
        entries=entries,
        source_fingerprint=source_fingerprint,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class TDHFScalarCodeManifest:
    object_kind: str
    module_name: str
    qualname: str
    module_file: str
    module_fingerprint: str
    source_fingerprint: str
    code_fingerprint: str | None

    def __post_init__(self) -> None:
        for name in ("object_kind", "module_name", "qualname", "module_file"):
            _text(getattr(self, name), f"code manifest {name}")
        _sha256(self.module_fingerprint, "module fingerprint")
        _sha256(self.source_fingerprint, "source fingerprint")
        if self.code_fingerprint is not None:
            _sha256(self.code_fingerprint, "code fingerprint")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def _code_manifest(value: object, label: str) -> TDHFScalarCodeManifest:
    if isinstance(value, ModuleType):
        module = value
        module_name = module.__name__
        qualname = module_name
        object_kind = "module"
        code_fingerprint = None
    elif inspect.isfunction(value):
        module = inspect.getmodule(value)
        if module is None:
            raise ValueError(f"{label} has no importable module")
        module_name = value.__module__
        qualname = value.__qualname__
        object_kind = "python_function"
        code_fingerprint = sha256(marshal.dumps(value.__code__)).hexdigest()
    else:
        raise TypeError(f"{label} must be a Python function or module")
    module_file_value = inspect.getsourcefile(module) or getattr(module, "__file__", None)
    if not module_file_value:
        raise ValueError(f"{label} module has no source file")
    module_file = str(Path(module_file_value).resolve())
    try:
        module_bytes = Path(module_file).read_bytes()
        source = inspect.getsource(value)
    except (OSError, TypeError) as error:
        raise ValueError(f"{label} source snapshot is unavailable") from error
    return TDHFScalarCodeManifest(
        object_kind=object_kind,
        module_name=module_name,
        qualname=qualname,
        module_file=module_file,
        module_fingerprint=sha256(module_bytes).hexdigest(),
        source_fingerprint=sha256(source.encode()).hexdigest(),
        code_fingerprint=code_fingerprint,
    )


def _validate_callback_signature(callback: Callable[..., object], role: KernelRole) -> None:
    if not inspect.isfunction(callback):
        raise TypeError(f"{role} callback must be a plain Python function")
    if callback.__closure__:
        raise ValueError(f"{role} callback closures are forbidden")
    if callback.__defaults__ is not None or callback.__kwdefaults__:
        raise ValueError(f"{role} callback defaults are forbidden")
    required = ("inputs", "P") if role != "fock_derivative" else ("inputs", "P", "D")
    parameters = tuple(inspect.signature(callback).parameters.values())
    if tuple(item.name for item in parameters) != required or any(
        item.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        or item.default is not inspect.Parameter.empty
        for item in parameters
    ):
        rendered = ",".join(required)
        raise TypeError(f"{role} callback signature must be exactly ({rendered})")


@dataclass(frozen=True, slots=True)
class TDHFScalarDependencyBinding:
    dependency: object
    manifest: TDHFScalarCodeManifest

    def __post_init__(self) -> None:
        if type(self.manifest) is not TDHFScalarCodeManifest:
            raise TypeError("dependency binding requires an exact code manifest")
        self.validate_live_state()

    def validate_live_state(self) -> None:
        current = _code_manifest(self.dependency, "kernel dependency")
        if current != self.manifest:
            raise ValueError("kernel dependency source/code/module fingerprint drifted")

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.manifest.fingerprint


@dataclass(frozen=True, slots=True)
class TDHFScalarKernelManifest:
    role: KernelRole
    callback: TDHFScalarCodeManifest
    dependency_fingerprints: tuple[str, ...]
    provenance: str

    def __post_init__(self) -> None:
        if self.role not in ("energy", "fock", "fock_derivative"):
            raise ValueError("invalid scalar kernel role")
        if type(self.callback) is not TDHFScalarCodeManifest:
            raise TypeError("kernel callback manifest must have exact type")
        dependencies = tuple(self.dependency_fingerprints)
        for value in dependencies:
            _sha256(value, "kernel dependency fingerprint")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("kernel dependencies must be unique")
        object.__setattr__(self, "dependency_fingerprints", dependencies)
        _text(self.provenance, "kernel provenance")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class TDHFScalarKernelBinding:
    callback: Callable[..., object]
    dependencies: tuple[TDHFScalarDependencyBinding, ...]
    manifest: TDHFScalarKernelManifest

    def __post_init__(self) -> None:
        if type(self.manifest) is not TDHFScalarKernelManifest:
            raise TypeError("kernel binding requires an exact manifest")
        _validate_callback_signature(self.callback, self.manifest.role)
        dependencies = tuple(self.dependencies)
        if any(type(item) is not TDHFScalarDependencyBinding for item in dependencies):
            raise TypeError("kernel dependencies require exact typed bindings")
        object.__setattr__(self, "dependencies", dependencies)
        self.validate_live_state()

    def validate_live_state(self) -> None:
        _validate_callback_signature(self.callback, self.manifest.role)
        current = _code_manifest(self.callback, f"{self.manifest.role} callback")
        if current != self.manifest.callback:
            raise ValueError(
                f"{self.manifest.role} callback source/code/module fingerprint drifted"
            )
        for item in self.dependencies:
            item.validate_live_state()
        if tuple(item.manifest.fingerprint for item in self.dependencies) != (
            self.manifest.dependency_fingerprints
        ):
            raise ValueError(f"{self.manifest.role} dependency manifest drifted")

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.manifest.fingerprint


def bind_tdhf_scalar_kernel(
    *,
    role: KernelRole,
    callback: Callable[..., object],
    dependencies: Sequence[object],
    provenance: str,
) -> TDHFScalarKernelBinding:
    """Bind callback, module, source, code, and concrete dependency snapshots."""

    _validate_callback_signature(callback, role)
    dependency_bindings = tuple(
        TDHFScalarDependencyBinding(item, _code_manifest(item, "kernel dependency"))
        for item in dependencies
    )
    manifests = tuple(item.manifest.fingerprint for item in dependency_bindings)
    manifest = TDHFScalarKernelManifest(
        role=role,
        callback=_code_manifest(callback, f"{role} callback"),
        dependency_fingerprints=manifests,
        provenance=provenance,
    )
    return TDHFScalarKernelBinding(callback, dependency_bindings, manifest)


@dataclass(frozen=True, slots=True)
class TDHFFullProjectorFunctionalBinding:
    """Three distinct kernels plus configured trace-forbidden entrypoints."""

    energy: TDHFScalarKernelBinding
    fock: TDHFScalarKernelBinding
    fock_derivative: TDHFScalarKernelBinding
    forbidden_entrypoints: tuple[object, ...] = ()
    trust_statement: str = field(
        default=(
            "trusted_provider_source_code_dependency_binding_and_python_trace; "
            "not_hostile_code_proof"
        ),
        init=False,
    )

    def __post_init__(self) -> None:
        kernels = (self.energy, self.fock, self.fock_derivative)
        if any(type(item) is not TDHFScalarKernelBinding for item in kernels):
            raise TypeError("functional binding requires exact kernel bindings")
        if tuple(item.manifest.role for item in kernels) != (
            "energy",
            "fock",
            "fock_derivative",
        ):
            raise ValueError("functional kernel roles/order changed")
        callbacks = tuple(item.callback for item in kernels)
        if len({id(item) for item in callbacks}) != 3:
            raise ValueError("energy/F/dF callbacks must be distinct objects")
        codes = tuple(item.__code__ for item in callbacks)
        if len({id(item) for item in codes}) != 3:
            raise ValueError("energy/F/dF callbacks must use distinct code objects")
        forbidden = tuple(self.forbidden_entrypoints)
        for item in forbidden:
            if not inspect.isfunction(item):
                raise TypeError("trace-forbidden entrypoints must be Python functions")
        object.__setattr__(self, "forbidden_entrypoints", forbidden)
        self.validate_live_state()

    def validate_live_state(self) -> None:
        for item in (self.energy, self.fock, self.fock_derivative):
            item.validate_live_state()
        callbacks = (self.energy.callback, self.fock.callback, self.fock_derivative.callback)
        if len({id(item) for item in callbacks}) != 3 or len(
            {id(item.__code__) for item in callbacks}
        ) != 3:
            raise ValueError("kernel callback/code identity drifted")

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return _fingerprint(
            {
                "kernels": tuple(
                    item.manifest.fingerprint
                    for item in (self.energy, self.fock, self.fock_derivative)
                ),
                "forbidden": tuple(
                    _code_manifest(item, "forbidden entrypoint").fingerprint
                    for item in self.forbidden_entrypoints
                ),
                "trust_statement": self.trust_statement,
            }
        )


@dataclass(frozen=True, slots=True)
class TDHFFullProjectorDirection:
    """One Hermitian affine direction normalized by the ABI itself."""

    label: str
    matrix: Array

    def __post_init__(self) -> None:
        _text(self.label, "direction label")
        value = _readonly_exact_complex(
            self.matrix, label=f"direction {self.label!r}", hermitian=True
        )
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("direction must be a square Hermitian matrix")
        signal = _frobenius_norm(value)
        if signal < TDHF_FULL_PROJECTOR_DIRECTION_SIGNAL_MINIMUM:
            raise ValueError(
                "direction Frobenius signal is zero/tiny before normalization: "
                f"{signal:.6e} < {TDHF_FULL_PROJECTOR_DIRECTION_SIGNAL_MINIMUM:.6e}"
            )
        normalized = np.asarray(value / signal, dtype=np.complex128)
        normalized_norm = _frobenius_norm(normalized)
        if (
            abs(normalized_norm - 1.0)
            > TDHF_FULL_PROJECTOR_DIRECTION_NORMALIZATION_TOLERANCE
        ):
            raise ValueError("direction failed locked unit-Frobenius normalization")
        object.__setattr__(
            self,
            "matrix",
            _readonly_exact_complex(
                normalized,
                label=f"normalized direction {self.label!r}",
                hermitian=True,
            ),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def _direction_inventory_fingerprint(
    directions: Sequence[TDHFFullProjectorDirection],
) -> str:
    return _fingerprint(
        tuple((item.label, _array_sha256(item.matrix)) for item in directions)
    )


def deterministic_complete_hermitian_basis(
    dimension: int,
) -> tuple[TDHFFullProjectorDirection, ...]:
    """Return the deterministic ``N^2`` real Hermitian basis for small tests.

    Production-sized spaces must preregister explicit physically bound probes;
    this helper fails above the locked small-test dimension.
    """

    if type(dimension) is not int or not (
        1 <= dimension <= TDHF_FULL_PROJECTOR_SMALL_BASIS_MAXIMUM_DIMENSION
    ):
        raise ValueError(
            "deterministic complete Hermitian basis is restricted to small tests "
            f"with N<={TDHF_FULL_PROJECTOR_SMALL_BASIS_MAXIMUM_DIMENSION}"
        )
    result: list[TDHFFullProjectorDirection] = []
    for index in range(dimension):
        value = np.zeros((dimension, dimension), dtype=np.complex128)
        value[index, index] = 1.0
        result.append(TDHFFullProjectorDirection(f"diag[{index}]", value))
    scale = 1.0 / math.sqrt(2.0)
    for left in range(dimension):
        for right in range(left + 1, dimension):
            real = np.zeros((dimension, dimension), dtype=np.complex128)
            real[left, right] = scale
            real[right, left] = scale
            result.append(
                TDHFFullProjectorDirection(f"real[{left},{right}]", real)
            )
            imaginary = np.zeros((dimension, dimension), dtype=np.complex128)
            imaginary[left, right] = -1j * scale
            imaginary[right, left] = 1j * scale
            result.append(
                TDHFFullProjectorDirection(f"imag[{left},{right}]", imaginary)
            )
    return tuple(result)


def complete_hermitian_basis_inventory_fingerprint(dimension: int) -> str:
    """Return the internally generated canonical normalized ``N^2`` inventory."""

    return _direction_inventory_fingerprint(
        deterministic_complete_hermitian_basis(dimension)
    )


@dataclass(frozen=True, slots=True)
class TDHFFullProjectorUnitaryProbe:
    """One preregistered same-trace exact-projector E/F execution probe."""

    label: str
    projector: Array
    source_projector_fingerprint: str
    projector_fingerprint: str = field(init=False)
    trace_fingerprint: str = field(init=False)
    idempotency_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _text(self.label, "unitary-projector probe label")
        _sha256(
            self.source_projector_fingerprint,
            "unitary-projector source fingerprint",
        )
        projector = _readonly_exact_complex(
            self.projector,
            label=f"unitary-projector probe {self.label!r}",
            hermitian=True,
        )
        if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
            raise ValueError("unitary-projector probe must be square")
        object.__setattr__(self, "projector", projector)
        object.__setattr__(self, "projector_fingerprint", _array_sha256(projector))
        object.__setattr__(
            self,
            "trace_fingerprint",
            _fingerprint(complex(np.trace(projector))),
        )
        object.__setattr__(
            self,
            "idempotency_fingerprint",
            _array_sha256(projector @ projector - projector),
        )

    def _validate_live_state(self) -> None:
        if (
            _array_sha256(self.projector) != self.projector_fingerprint
            or _fingerprint(complex(np.trace(self.projector)))
            != self.trace_fingerprint
            or _array_sha256(self.projector @ self.projector - self.projector)
            != self.idempotency_fingerprint
        ):
            raise ValueError("unitary-projector probe value/hash fields drifted")

    @property
    def fingerprint(self) -> str:
        self._validate_live_state()
        return _fingerprint(self)


def make_tdhf_full_projector_unitary_probe(
    *, label: str, source_projector: Array, projector: Array
) -> TDHFFullProjectorUnitaryProbe:
    """Bind an explicit projector value to the exact source-projector bytes."""

    source = _readonly_exact_complex(
        source_projector, label="unitary-probe source projector", hermitian=True
    )
    return TDHFFullProjectorUnitaryProbe(
        label=label,
        projector=projector,
        source_projector_fingerprint=_array_sha256(source),
    )


@dataclass(frozen=True, slots=True)
class TDHFFullProjectorValidationTolerances:
    gradient_absolute: float = 2.0e-8
    gradient_relative: float = 2.0e-8
    derivative_absolute: float = 2.0e-8
    derivative_relative: float = 2.0e-8
    exact_absolute: float = 2.0e-8
    exact_relative: float = 2.0e-8
    stationarity_absolute: float = 2.0e-8
    stationarity_relative: float = 2.0e-8
    self_adjoint_absolute: float = 2.0e-8
    self_adjoint_relative: float = 2.0e-8
    projector_tolerance: float = 5.0e-11
    roundoff_multiplier: float = TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER

    def __post_init__(self) -> None:
        maxima = {
            "gradient_absolute": TDHF_SCALAR_CURVATURE_V1_STATIONARITY_ABSOLUTE_MAXIMUM,
            "gradient_relative": TDHF_SCALAR_CURVATURE_V1_STATIONARITY_RELATIVE_MAXIMUM,
            "derivative_absolute": TDHF_SCALAR_CURVATURE_V1_CURVATURE_ABSOLUTE_MAXIMUM,
            "derivative_relative": TDHF_SCALAR_CURVATURE_V1_CURVATURE_RELATIVE_MAXIMUM,
            "exact_absolute": TDHF_SCALAR_CURVATURE_V1_CURVATURE_ABSOLUTE_MAXIMUM,
            "exact_relative": TDHF_SCALAR_CURVATURE_V1_CURVATURE_RELATIVE_MAXIMUM,
            "stationarity_absolute": TDHF_SCALAR_CURVATURE_V1_STATIONARITY_ABSOLUTE_MAXIMUM,
            "stationarity_relative": TDHF_SCALAR_CURVATURE_V1_STATIONARITY_RELATIVE_MAXIMUM,
            "self_adjoint_absolute": TDHF_SCALAR_CURVATURE_V1_MATRIX_ABSOLUTE_MAXIMUM,
            "self_adjoint_relative": TDHF_SCALAR_CURVATURE_V1_MATRIX_RELATIVE_MAXIMUM,
            "projector_tolerance": TDHF_SCALAR_CURVATURE_V1_PROJECTOR_TOLERANCE_MAXIMUM,
        }
        for name, maximum in maxima.items():
            object.__setattr__(
                self,
                name,
                _bounded(
                    getattr(self, name),
                    name,
                    maximum,
                    positive=name == "projector_tolerance",
                ),
            )
        multiplier = _finite(self.roundoff_multiplier, "roundoff multiplier")
        if multiplier != TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER:
            raise ValueError("roundoff multiplier must equal the locked scalar-curvature v1 value")
        object.__setattr__(self, "roundoff_multiplier", multiplier)


@dataclass(frozen=True, slots=True)
class TDHFFullProjectorValidationPlan:
    """Detached all-step affine validation inventory."""

    space: TDHFFullProjectorSpace
    source_projector: Array
    directions: tuple[TDHFFullProjectorDirection, ...]
    steps: tuple[float, ...]
    tolerances: TDHFFullProjectorValidationTolerances
    registration_label: str
    probe_scope: Literal["complete_small_test_basis", "explicit_bound_probes"]
    require_informative_df: bool = False
    unitary_projector_probes: tuple[TDHFFullProjectorUnitaryProbe, ...] = ()
    energy_normalization: str = field(
        default=TDHF_FULL_PROJECTOR_ENERGY_NORMALIZATION, init=False
    )
    pairing: str = field(default=TDHF_FULL_PROJECTOR_PAIRING, init=False)

    def __post_init__(self) -> None:
        if type(self.space) is not TDHFFullProjectorSpace:
            raise TypeError("validation plan requires an exact full-projector space")
        projector = _readonly_exact_complex(
            self.source_projector,
            label="source projector",
            shape=(self.space.dimension, self.space.dimension),
            hermitian=True,
        )
        object.__setattr__(self, "source_projector", projector)
        tolerance = self.tolerances.projector_tolerance
        if _max_abs(projector @ projector - projector) > tolerance:
            raise ValueError("source projector is not idempotent")
        directions = tuple(self.directions)
        if not directions or any(type(item) is not TDHFFullProjectorDirection for item in directions):
            raise TypeError("validation plan requires explicit typed directions")
        if len({item.label for item in directions}) != len(directions):
            raise ValueError("direction labels must be unique")
        if any(item.matrix.shape != projector.shape for item in directions):
            raise ValueError("every direction must have the full dense source shape")
        if any(
            abs(_frobenius_norm(item.matrix) - 1.0)
            > TDHF_FULL_PROJECTOR_DIRECTION_NORMALIZATION_TOLERANCE
            for item in directions
        ):
            raise ValueError("every registered direction must have locked unit Frobenius norm")
        object.__setattr__(self, "directions", directions)
        steps = tuple(_finite(item, "finite-difference step", positive=True) for item in self.steps)
        if len(steps) < 2:
            raise ValueError("preregistered validation needs at least two steps")
        if any(
            item < TDHF_SCALAR_CURVATURE_V1_STEP_MINIMUM
            or item > TDHF_SCALAR_CURVATURE_V1_STEP_MAXIMUM
            for item in steps
        ):
            raise ValueError(
                "finite-difference steps must use the locked scalar-curvature v1 range"
            )
        if any(left <= right for left, right in zip(steps, steps[1:])):
            raise ValueError("finite-difference steps must be strictly decreasing")
        object.__setattr__(self, "steps", steps)
        if type(self.tolerances) is not TDHFFullProjectorValidationTolerances:
            raise TypeError("validation plan requires exact typed tolerances")
        self.tolerances.__post_init__()
        _text(self.registration_label, "validation registration label")
        if self.probe_scope not in ("complete_small_test_basis", "explicit_bound_probes"):
            raise ValueError("invalid full-projector probe scope")
        if type(self.require_informative_df) is not bool:
            raise TypeError("dF informativeness requirement must be a strict bool")
        unitary_probes = tuple(self.unitary_projector_probes)
        if any(type(item) is not TDHFFullProjectorUnitaryProbe for item in unitary_probes):
            raise TypeError("unitary projector probes require exact typed values")
        if len({item.label for item in unitary_probes}) != len(unitary_probes):
            raise ValueError("unitary projector probe labels must be unique")
        source_fingerprint = _array_sha256(projector)
        source_trace = complex(np.trace(projector))
        for probe in unitary_probes:
            probe._validate_live_state()
            if probe.projector.shape != projector.shape:
                raise ValueError("unitary projector probe shape differs from source")
            if probe.source_projector_fingerprint != source_fingerprint:
                raise ValueError("unitary projector probe source fingerprint is stale")
            if probe.projector_fingerprint == source_fingerprint:
                raise ValueError("unitary projector probe must differ from its source")
            if _max_abs(probe.projector @ probe.projector - probe.projector) > tolerance:
                raise ValueError("unitary projector probe is not idempotent")
            if abs(complex(np.trace(probe.projector)) - source_trace) > tolerance:
                raise ValueError("unitary projector probe trace differs from source")
        object.__setattr__(self, "unitary_projector_probes", unitary_probes)
        complete = self._directions_are_complete()
        if self.probe_scope == "complete_small_test_basis" and not complete:
            raise ValueError(
                "complete_small_test_basis must equal the internally generated "
                "normalized Hermitian inventory"
            )
        if self.probe_scope == "explicit_bound_probes" and complete:
            raise ValueError("a complete small basis must use complete_small_test_basis scope")
        if (
            self.energy_normalization != TDHF_FULL_PROJECTOR_ENERGY_NORMALIZATION
            or self.pairing != TDHF_FULL_PROJECTOR_PAIRING
        ):
            raise ValueError("raw total-energy/full-trace convention changed")

    def _directions_are_complete(self) -> bool:
        dimension = self.space.dimension
        if len(self.directions) != dimension * dimension:
            return False
        if dimension > TDHF_FULL_PROJECTOR_SMALL_BASIS_MAXIMUM_DIMENSION:
            return False
        return self.direction_inventory_fingerprint == (
            self.complete_basis_inventory_fingerprint
        )

    @property
    def direction_inventory_fingerprint(self) -> str:
        return _direction_inventory_fingerprint(self.directions)

    @property
    def complete_basis_inventory_fingerprint(self) -> str | None:
        if self.space.dimension > TDHF_FULL_PROJECTOR_SMALL_BASIS_MAXIMUM_DIMENSION:
            return None
        return complete_hermitian_basis_inventory_fingerprint(self.space.dimension)

    @property
    def directions_are_complete(self) -> bool:
        return self._directions_are_complete()

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class TDHFFullProjectorFunctionalApproval:
    _factory_token: InitVar[object]
    api_version: str
    space_fingerprint: str
    inputs_fingerprint: str
    binding_fingerprint: str
    plan_fingerprint: str
    kernel_manifest_fingerprints: tuple[str, str, str]
    callback_code_fingerprints: tuple[str, str, str]
    source_projector_fingerprint: str
    provenance: str
    detached_before_callback_calls: bool

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _APPROVAL_TOKEN:
            raise TypeError("full-projector approval requires the private factory token")
        if self.api_version != TDHF_FULL_PROJECTOR_SCALAR_FUNCTIONAL_API_VERSION:
            raise ValueError("full-projector approval ABI changed")
        for name in (
            "space_fingerprint",
            "inputs_fingerprint",
            "binding_fingerprint",
            "plan_fingerprint",
            "source_projector_fingerprint",
        ):
            _sha256(getattr(self, name), name)
        for value in self.kernel_manifest_fingerprints + self.callback_code_fingerprints:
            _sha256(value, "approval kernel/code fingerprint")
        if len(set(self.callback_code_fingerprints)) != 3:
            raise ValueError("approval callback code fingerprints must be distinct")
        _text(self.provenance, "approval provenance")
        if self.detached_before_callback_calls is not True:
            raise ValueError("approval must be detached before callback calls")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def make_tdhf_full_projector_functional_approval(
    *,
    space: TDHFFullProjectorSpace,
    inputs: TDHFScalarFunctionalInputsManifest,
    binding: TDHFFullProjectorFunctionalBinding,
    plan: TDHFFullProjectorValidationPlan,
    provenance: str,
) -> TDHFFullProjectorFunctionalApproval:
    """Bind the complete preregistration without executing any callback."""

    if type(space) is not TDHFFullProjectorSpace:
        raise TypeError("approval requires exact full-projector space")
    if type(inputs) is not TDHFScalarFunctionalInputsManifest:
        raise TypeError("approval requires exact functional inputs")
    if type(binding) is not TDHFFullProjectorFunctionalBinding:
        raise TypeError("approval requires exact functional binding")
    if type(plan) is not TDHFFullProjectorValidationPlan:
        raise TypeError("approval requires exact validation plan")
    if plan.space.fingerprint != space.fingerprint:
        raise ValueError("approval plan/space mismatch")
    inputs.validate_live_state()
    binding.validate_live_state()
    kernels = (binding.energy, binding.fock, binding.fock_derivative)
    code_fingerprints = tuple(
        item.manifest.callback.code_fingerprint for item in kernels
    )
    if any(item is None for item in code_fingerprints):
        raise ValueError("callback code fingerprint is missing")
    return TDHFFullProjectorFunctionalApproval(
        _factory_token=_APPROVAL_TOKEN,
        api_version=TDHF_FULL_PROJECTOR_SCALAR_FUNCTIONAL_API_VERSION,
        space_fingerprint=space.fingerprint,
        inputs_fingerprint=inputs.fingerprint,
        binding_fingerprint=binding.fingerprint,
        plan_fingerprint=plan.fingerprint,
        kernel_manifest_fingerprints=tuple(
            item.manifest.fingerprint for item in kernels
        ),
        callback_code_fingerprints=code_fingerprints,  # type: ignore[arg-type]
        source_projector_fingerprint=_array_sha256(plan.source_projector),
        provenance=provenance,
        detached_before_callback_calls=True,
    )


@contextmanager
def _reject_traced_calls(forbidden: Mapping[object, str]):
    previous = sys.gettrace()

    def tracer(frame: Any, event: str, arg: object) -> Any:
        del arg
        if event == "call" and frame.f_code in forbidden:
            raise RuntimeError(
                "trusted-provider trace rejected direct peer/forbidden callback delegation: "
                + forbidden[frame.f_code]
            )
        return tracer

    sys.settrace(tracer)
    try:
        yield
    finally:
        sys.settrace(previous)


@dataclass(slots=True)
class _Executor:
    inputs: TDHFScalarFunctionalInputsManifest
    binding: TDHFFullProjectorFunctionalBinding
    dimension: int
    invocation_counts: dict[str, int] = field(
        default_factory=lambda: {"energy": 0, "fock": 0, "fock_derivative": 0}
    )

    def _call(self, role: KernelRole, P: Array, D: Array | None = None) -> object:
        self.inputs.validate_live_state()
        self.binding.validate_live_state()
        kernel = getattr(self.binding, role)
        input_before = self.inputs.fingerprint
        binding_before = self.binding.fingerprint
        p = _readonly_exact_complex(
            P,
            label=f"{role} P argument",
            shape=(self.dimension, self.dimension),
            hermitian=True,
        )
        p_before = _array_sha256(p)
        d = None
        d_before = None
        if D is not None:
            d = _readonly_exact_complex(
                D,
                label=f"{role} D argument",
                shape=(self.dimension, self.dimension),
                hermitian=True,
            )
            d_before = _array_sha256(d)
        callbacks = (
            self.binding.energy.callback,
            self.binding.fock.callback,
            self.binding.fock_derivative.callback,
        )
        forbidden: dict[object, str] = {
            item.__code__: f"peer callback {item.__module__}.{item.__qualname__}"
            for item in callbacks
            if item is not kernel.callback
        }
        forbidden.update(
            {
                item.__code__: f"configured forbidden entrypoint {item.__module__}.{item.__qualname__}"
                for item in self.binding.forbidden_entrypoints
            }
        )
        with _reject_traced_calls(forbidden):
            if role == "fock_derivative":
                assert d is not None
                result = kernel.callback(self.inputs, p, d)
            else:
                result = kernel.callback(self.inputs, p)
        self.invocation_counts[role] += 1
        if p.flags.writeable or _array_sha256(p) != p_before:
            raise ValueError(f"{role} callback made P writable or mutated it")
        if d is not None and (d.flags.writeable or _array_sha256(d) != d_before):
            raise ValueError(f"{role} callback made D writable or mutated it")
        self.inputs.validate_live_state()
        self.binding.validate_live_state()
        if self.inputs.fingerprint != input_before:
            raise ValueError(f"{role} callback mutated explicit functional inputs")
        if self.binding.fingerprint != binding_before:
            raise ValueError(f"{role} callback/module/source/code/dependency drifted")
        if isinstance(result, np.ndarray):
            aliases = (p,) + ((d,) if d is not None else ()) + tuple(
                item.value
                for item in self.inputs.entries
                if isinstance(item.value, np.ndarray)
            )
            if any(np.shares_memory(result, item) for item in aliases):
                raise ValueError(f"{role} callback output aliases an input/P/D array")
        return result

    def energy(self, P: Array) -> float:
        result = self._call("energy", P)
        if isinstance(result, (bool, np.bool_, complex, np.complexfloating)):
            raise TypeError("energy callback must return a strict real scalar")
        value = float(result)
        if not math.isfinite(value):
            raise ValueError("energy callback returned nonfinite data")
        return value

    def matrix(self, role: Literal["fock", "fock_derivative"], P: Array, D: Array | None = None) -> Array:
        result = self._call(role, P, D)
        return _readonly_exact_complex(
            result,
            label=f"{role} callback output",
            shape=(self.dimension, self.dimension),
            hermitian=True,
        )


@dataclass(frozen=True, slots=True)
class TDHFFullProjectorStepEvidence:
    direction_label: str
    step: float
    energies_minus2_minus1_zero_plus1_plus2: tuple[float, float, float, float, float]
    energy_first_derivative: float
    fock_pairing: float
    energy_to_fock_residual: float
    energy_second_derivative: float
    derivative_pairing: float
    energy_second_to_derivative_residual: float
    fock_to_derivative_residual: float
    exact_affine_fock_residual: float
    exact_quadratic_energy_residual: float
    first_roundoff_allowance: float
    second_roundoff_allowance: float
    passed: bool

    def __post_init__(self) -> None:
        _text(self.direction_label, "step-evidence direction")
        _finite(self.step, "step-evidence step", positive=True)
        for value in self.energies_minus2_minus1_zero_plus1_plus2:
            _finite(value, "step evidence energy")
        for item in fields(self):
            if item.name in (
                "direction_label",
                "energies_minus2_minus1_zero_plus1_plus2",
                "passed",
            ):
                continue
            _finite(getattr(self, item.name), f"step evidence {item.name}")
        if self.passed is not True:
            raise ValueError("failed local step evidence cannot enter a receipt")


@dataclass(frozen=True, slots=True)
class TDHFFullProjectorAnchorEvidence:
    probe_label: str
    anchor_label: str
    step: float
    maximum_dF_anchor_residual: float
    passed: bool

    def __post_init__(self) -> None:
        _text(self.probe_label, "anchor probe label")
        _text(self.anchor_label, "anchor label")
        _finite(self.step, "anchor step", positive=True)
        _finite(self.maximum_dF_anchor_residual, "anchor dF residual")
        if self.passed is not True:
            raise ValueError("failed anchor evidence cannot enter a receipt")


@dataclass(frozen=True, slots=True)
class TDHFFullProjectorPairingEvidence:
    left_label: str
    right_label: str
    left_dF_right: complex
    right_dF_left: complex
    imaginary_residual: float
    reciprocity_residual: float
    passed: bool

    def __post_init__(self) -> None:
        _text(self.left_label, "pairing left label")
        _text(self.right_label, "pairing right label")
        if not all(
            math.isfinite(value)
            for value in (
                self.left_dF_right.real,
                self.left_dF_right.imag,
                self.right_dF_left.real,
                self.right_dF_left.imag,
                self.imaginary_residual,
                self.reciprocity_residual,
            )
        ):
            raise ValueError("pairing evidence must be finite")
        if self.passed is not True:
            raise ValueError("failed pairing evidence cannot enter a receipt")


@dataclass(frozen=True, slots=True)
class TDHFFullProjectorUnitaryProbeEvidence:
    label: str
    probe_fingerprint: str
    source_projector_fingerprint: str
    projector_fingerprint: str
    trace_fingerprint: str
    idempotency_fingerprint: str
    energy: float
    fock_fingerprint: str
    energy_executed: bool
    fock_executed: bool

    def __post_init__(self) -> None:
        _text(self.label, "unitary evidence label")
        for name in (
            "probe_fingerprint",
            "source_projector_fingerprint",
            "projector_fingerprint",
            "trace_fingerprint",
            "idempotency_fingerprint",
            "fock_fingerprint",
        ):
            _sha256(getattr(self, name), f"unitary evidence {name}")
        _finite(self.energy, "unitary evidence energy")
        if (self.energy_executed, self.fock_executed) != (True, True):
            raise ValueError("unitary evidence requires executed E and F callbacks")


@dataclass(frozen=True, slots=True)
class TDHFFullProjectorFunctionalEvidenceReceipt:
    _factory_token: InitVar[object]
    api_version: str
    approval_fingerprint: str
    space_fingerprint: str
    inputs_fingerprint_before: str
    inputs_fingerprint_after: str
    binding_fingerprint_before: str
    binding_fingerprint_after: str
    plan_fingerprint: str
    source_projector_fingerprint: str
    source_fock: Array
    source_fock_fingerprint: str
    source_commutator_residual: float
    source_qfp_residual: float
    direction_inventory_fingerprint: str
    complete_basis_inventory_fingerprint: str | None
    step_evidence: tuple[TDHFFullProjectorStepEvidence, ...]
    anchor_evidence: tuple[TDHFFullProjectorAnchorEvidence, ...]
    pairing_evidence: tuple[TDHFFullProjectorPairingEvidence, ...]
    unitary_probe_evidence: tuple[TDHFFullProjectorUnitaryProbeEvidence, ...]
    callback_invocation_counts: tuple[tuple[str, int], ...]
    maximum_dF_response_frobenius_norm: float
    maximum_taylor_fock_signal_frobenius_norm: float
    maximum_taylor_signal_to_roundoff_ratio: float
    all_registered_steps_passed: bool
    source_stationarity_verified: bool
    dF_anchor_independence_verified: bool
    dF_real_self_adjoint_verified: bool
    dF_response_informativeness_required: bool
    dF_response_informative: bool
    callback_trace_verified: bool
    callback_source_code_dependency_stable: bool
    registered_probe_functional_consistency: bool
    full_projector_functional_consistency: bool
    directions_are_complete: bool
    exact_unitary_projector_probes_executed: bool
    exact_unitary_projector_probe_count: int
    exact_unitary_projector_probe_inventory_fingerprint: str | None
    tdhf_hessian_match: bool = field(default=False, init=False)
    static_hessian_authority_promoted: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)
    trust_statement: str = field(
        default="trusted_provider_not_hostile_proof", init=False
    )

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _RECEIPT_TOKEN:
            raise TypeError("full-projector receipt requires the private factory token")
        if self.api_version != TDHF_FULL_PROJECTOR_SCALAR_FUNCTIONAL_API_VERSION:
            raise ValueError("full-projector receipt ABI changed")
        for name in (
            "approval_fingerprint",
            "space_fingerprint",
            "inputs_fingerprint_before",
            "inputs_fingerprint_after",
            "binding_fingerprint_before",
            "binding_fingerprint_after",
            "plan_fingerprint",
            "source_projector_fingerprint",
            "source_fock_fingerprint",
        ):
            _sha256(getattr(self, name), name)
        if self.inputs_fingerprint_before != self.inputs_fingerprint_after:
            raise ValueError("receipt input fingerprints changed")
        if self.binding_fingerprint_before != self.binding_fingerprint_after:
            raise ValueError("receipt callback/source/code/dependency fingerprints changed")
        fock = _readonly_exact_complex(
            self.source_fock, label="receipt source Fock", hermitian=True
        )
        object.__setattr__(self, "source_fock", fock)
        if _array_sha256(fock) != self.source_fock_fingerprint:
            raise ValueError("receipt source-Fock fingerprint mismatch")
        for name in (
            "source_commutator_residual",
            "source_qfp_residual",
            "maximum_dF_response_frobenius_norm",
            "maximum_taylor_fock_signal_frobenius_norm",
            "maximum_taylor_signal_to_roundoff_ratio",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        _sha256(self.direction_inventory_fingerprint, "direction inventory fingerprint")
        if self.complete_basis_inventory_fingerprint is not None:
            _sha256(
                self.complete_basis_inventory_fingerprint,
                "complete-basis inventory fingerprint",
            )
        if not self.step_evidence or not self.anchor_evidence or not self.pairing_evidence:
            raise ValueError("receipt affine evidence inventories must be nonempty")
        if any(
            type(item) is not TDHFFullProjectorUnitaryProbeEvidence
            for item in self.unitary_probe_evidence
        ):
            raise TypeError("receipt unitary evidence uses a non-exact type")
        if type(self.exact_unitary_projector_probe_count) is not int:
            raise TypeError("unitary probe count must be a strict integer")
        if self.exact_unitary_projector_probe_count != len(self.unitary_probe_evidence):
            raise ValueError("unitary probe count/evidence inventory mismatch")
        executed = bool(self.unitary_probe_evidence)
        if self.exact_unitary_projector_probes_executed is not executed:
            raise ValueError("unitary support must be derived from executed E/F evidence")
        expected_unitary_inventory = (
            _fingerprint(tuple(item.probe_fingerprint for item in self.unitary_probe_evidence))
            if executed
            else None
        )
        if self.exact_unitary_projector_probe_inventory_fingerprint != expected_unitary_inventory:
            raise ValueError("unitary execution inventory fingerprint mismatch")
        status = (
            self.all_registered_steps_passed,
            self.source_stationarity_verified,
            self.dF_anchor_independence_verified,
            self.dF_real_self_adjoint_verified,
            self.callback_trace_verified,
            self.callback_source_code_dependency_stable,
            self.registered_probe_functional_consistency,
        )
        if status != (True,) * 7:
            raise ValueError("registered-probe receipt lost a mandatory consistency gate")
        if self.dF_response_informativeness_required and not self.dF_response_informative:
            raise ValueError("required dF response informativeness is false")
        expected_full = bool(
            self.registered_probe_functional_consistency
            and self.directions_are_complete
            and self.complete_basis_inventory_fingerprint
            == self.direction_inventory_fingerprint
        )
        if self.full_projector_functional_consistency is not expected_full:
            raise ValueError(
                "full-projector consistency must be derived only from the exact "
                "complete normalized N^2 inventory"
            )
        if any(
            (
                self.tdhf_hessian_match,
                self.static_hessian_authority_promoted,
                self.production_ready,
                self.paper_reproduction_verified,
            )
        ):
            raise ValueError("functional consistency receipt cannot promote authority")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def validate_tdhf_full_projector_functional(
    *,
    approval: TDHFFullProjectorFunctionalApproval,
    space: TDHFFullProjectorSpace,
    inputs: TDHFScalarFunctionalInputsManifest,
    binding: TDHFFullProjectorFunctionalBinding,
    plan: TDHFFullProjectorValidationPlan,
) -> TDHFFullProjectorFunctionalEvidenceReceipt:
    """Execute every preregistered all-step gate and return a factory receipt."""

    if type(approval) is not TDHFFullProjectorFunctionalApproval:
        raise TypeError("validation requires an exact detached approval")
    if type(space) is not TDHFFullProjectorSpace:
        raise TypeError("validation requires exact full-projector space")
    if type(inputs) is not TDHFScalarFunctionalInputsManifest:
        raise TypeError("validation requires exact functional inputs")
    if type(binding) is not TDHFFullProjectorFunctionalBinding:
        raise TypeError("validation requires exact functional binding")
    if type(plan) is not TDHFFullProjectorValidationPlan:
        raise TypeError("validation requires exact validation plan")
    inputs.validate_live_state()
    binding.validate_live_state()
    expected = (
        (approval.space_fingerprint, space.fingerprint, "space"),
        (approval.inputs_fingerprint, inputs.fingerprint, "input manifest"),
        (approval.binding_fingerprint, binding.fingerprint, "callback/source binding"),
        (approval.plan_fingerprint, plan.fingerprint, "validation plan"),
        (
            approval.source_projector_fingerprint,
            _array_sha256(plan.source_projector),
            "source projector",
        ),
    )
    for actual, current, label in expected:
        if actual != current:
            raise ValueError(f"detached approval {label} fingerprint is stale")
    kernels = (binding.energy, binding.fock, binding.fock_derivative)
    if approval.kernel_manifest_fingerprints != tuple(
        item.manifest.fingerprint for item in kernels
    ):
        raise ValueError("detached approval kernel manifest is stale")
    if approval.callback_code_fingerprints != tuple(
        item.manifest.callback.code_fingerprint for item in kernels
    ):
        raise ValueError("detached approval callback code is stale")

    inputs_before = inputs.fingerprint
    binding_before = binding.fingerprint
    executor = _Executor(inputs, binding, space.dimension)
    p0 = plan.source_projector
    f0 = executor.matrix("fock", p0)
    identity = np.eye(space.dimension, dtype=np.complex128)
    commutator = _max_abs(f0 @ p0 - p0 @ f0)
    qfp = _max_abs((identity - p0) @ f0 @ p0)
    stationarity_bound = _bound(
        plan.tolerances.stationarity_absolute,
        plan.tolerances.stationarity_relative,
        f0,
    )
    if commutator > stationarity_bound or qfp > stationarity_bound:
        raise ValueError(
            "full source stationarity [F(P0),P0]/QFP failed: "
            f"commutator={commutator:.6e}, qfp={qfp:.6e}, "
            f"bound={stationarity_bound:.6e}"
        )

    dF0 = {
        direction.label: executor.matrix(
            "fock_derivative", p0, direction.matrix
        )
        for direction in plan.directions
    }
    maximum_dF_response = max(_frobenius_norm(item) for item in dF0.values())
    pairing_records: list[TDHFFullProjectorPairingEvidence] = []
    for left in plan.directions:
        for right in plan.directions:
            left_right = _trace_pairing(left.matrix, dF0[right.label])
            right_left = _trace_pairing(right.matrix, dF0[left.label])
            imaginary = max(abs(left_right.imag), abs(right_left.imag))
            reciprocity = abs(left_right.real - right_left.real)
            bound = _bound(
                plan.tolerances.self_adjoint_absolute,
                plan.tolerances.self_adjoint_relative,
                left_right,
                right_left,
            )
            if imaginary > bound or reciprocity > bound:
                raise ValueError(
                    "dF real self-adjoint pairing failed for "
                    f"{left.label}/{right.label}: imaginary={imaginary:.6e}, "
                    f"reciprocity={reciprocity:.6e}, bound={bound:.6e}"
                )
            pairing_records.append(
                TDHFFullProjectorPairingEvidence(
                    left_label=left.label,
                    right_label=right.label,
                    left_dF_right=left_right,
                    right_dF_left=right_left,
                    imaginary_residual=imaginary,
                    reciprocity_residual=reciprocity,
                    passed=True,
                )
            )

    e0 = executor.energy(p0)
    step_records: list[TDHFFullProjectorStepEvidence] = []
    anchor_records: list[TDHFFullProjectorAnchorEvidence] = []
    eps = np.finfo(np.float64).eps
    maximum_taylor_signal = 0.0
    maximum_taylor_signal_to_roundoff_ratio = 0.0
    for direction in plan.directions:
        D = direction.matrix
        dF = dF0[direction.label]
        fock_pairing_complex = _trace_pairing(f0, D)
        derivative_pairing_complex = _trace_pairing(D, dF)
        pairing_imag_bound = _bound(
            plan.tolerances.self_adjoint_absolute,
            plan.tolerances.self_adjoint_relative,
            fock_pairing_complex,
            derivative_pairing_complex,
        )
        if max(abs(fock_pairing_complex.imag), abs(derivative_pairing_complex.imag)) > pairing_imag_bound:
            raise ValueError("raw full-trace E/F/dF pairing is not real")
        fock_pairing = float(fock_pairing_complex.real)
        derivative_pairing = float(derivative_pairing_complex.real)
        for h in plan.steps:
            projectors = tuple(p0 + multiplier * h * D for multiplier in (-2, -1, 0, 1, 2))
            energies = tuple(executor.energy(item) for item in projectors)
            focks = tuple(executor.matrix("fock", item) for item in projectors)
            for shifted_fock in focks:
                signal = _frobenius_norm(shifted_fock - f0)
                roundoff_floor = (
                    plan.tolerances.roundoff_multiplier
                    * eps
                    * max(1.0, _frobenius_norm(f0), _frobenius_norm(shifted_fock))
                )
                maximum_taylor_signal = max(maximum_taylor_signal, signal)
                maximum_taylor_signal_to_roundoff_ratio = max(
                    maximum_taylor_signal_to_roundoff_ratio,
                    signal / roundoff_floor,
                )
            em2, em1, ezero, ep1, ep2 = energies
            first = (em2 - 8.0 * em1 + 8.0 * ep1 - ep2) / (12.0 * h)
            second = (
                -ep2 + 16.0 * ep1 - 30.0 * ezero + 16.0 * em1 - em2
            ) / (12.0 * h * h)
            fock_fd = (
                focks[0] - 8.0 * focks[1] + 8.0 * focks[3] - focks[4]
            ) / (12.0 * h)
            e_f_residual = abs(first - fock_pairing)
            e_second_residual = abs(second - derivative_pairing)
            f_df_residual = _max_abs(fock_fd - dF)
            exact_fock_residual = 0.0
            exact_energy_residual = 0.0
            for multiplier, energy, fock in zip((-2, -1, 0, 1, 2), energies, focks):
                t = multiplier * h
                exact_fock_residual = max(
                    exact_fock_residual, _max_abs(fock - (f0 + t * dF))
                )
                expected_energy = e0 + t * fock_pairing + 0.5 * t * t * derivative_pairing
                exact_energy_residual = max(
                    exact_energy_residual, abs(energy - expected_energy)
                )
            energy_scale = max(1.0, *(abs(item) for item in energies))
            fock_scale = max(1.0, *(_max_abs(item) for item in focks))
            first_roundoff = (
                plan.tolerances.roundoff_multiplier * eps * energy_scale / h
            )
            second_roundoff = (
                plan.tolerances.roundoff_multiplier * eps * energy_scale / (h * h)
            )
            fock_roundoff = (
                plan.tolerances.roundoff_multiplier * eps * fock_scale / h
            )
            if (
                first_roundoff
                > TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM
                or fock_roundoff
                > TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM
                or second_roundoff
                > TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_CURVATURE_ALLOWANCE_MAXIMUM
            ):
                raise ValueError(
                    "roundoff allowance is vacuous (tiny step or huge energy/Fock offset)"
                )
            gradient_bound = _bound(
                plan.tolerances.gradient_absolute,
                plan.tolerances.gradient_relative,
                first,
                fock_pairing,
            ) + first_roundoff
            derivative_bound = _bound(
                plan.tolerances.derivative_absolute,
                plan.tolerances.derivative_relative,
                fock_fd,
                dF,
                second,
                derivative_pairing,
            ) + max(second_roundoff, fock_roundoff)
            exact_fock_bound = _bound(
                plan.tolerances.exact_absolute,
                plan.tolerances.exact_relative,
                *focks,
                f0,
                dF,
            )
            exact_energy_bound = _bound(
                plan.tolerances.exact_absolute,
                plan.tolerances.exact_relative,
                *energies,
                e0,
                fock_pairing,
                derivative_pairing,
            ) + second_roundoff * h * h
            if (
                e_f_residual > gradient_bound
                or e_second_residual > derivative_bound
                or f_df_residual > derivative_bound
                or exact_fock_residual > exact_fock_bound
                or exact_energy_residual > exact_energy_bound
            ):
                raise ValueError(
                    "registered full-projector E/F/dF affine/quadratic gate failed for "
                    f"{direction.label} at h={h}: "
                    f"E->F={e_f_residual:.6e}, E''->dF={e_second_residual:.6e}, "
                    f"F->dF={f_df_residual:.6e}, affineF={exact_fock_residual:.6e}, "
                    f"quadraticE={exact_energy_residual:.6e}"
                )
            step_records.append(
                TDHFFullProjectorStepEvidence(
                    direction_label=direction.label,
                    step=h,
                    energies_minus2_minus1_zero_plus1_plus2=energies,
                    energy_first_derivative=first,
                    fock_pairing=fock_pairing,
                    energy_to_fock_residual=e_f_residual,
                    energy_second_derivative=second,
                    derivative_pairing=derivative_pairing,
                    energy_second_to_derivative_residual=e_second_residual,
                    fock_to_derivative_residual=f_df_residual,
                    exact_affine_fock_residual=exact_fock_residual,
                    exact_quadratic_energy_residual=exact_energy_residual,
                    first_roundoff_allowance=max(first_roundoff, fock_roundoff),
                    second_roundoff_allowance=second_roundoff,
                    passed=True,
                )
            )
            for anchor in plan.directions:
                residual = 0.0
                for multiplier in (-2, -1, 1, 2):
                    shifted = p0 + multiplier * h * anchor.matrix
                    shifted_dF = executor.matrix("fock_derivative", shifted, D)
                    residual = max(residual, _max_abs(shifted_dF - dF))
                anchor_bound = _bound(
                    plan.tolerances.exact_absolute,
                    plan.tolerances.exact_relative,
                    dF,
                )
                if residual > anchor_bound:
                    raise ValueError(
                        "dF anchor independence failed for probe/anchor "
                        f"{direction.label}/{anchor.label} at h={h}"
                    )
                anchor_records.append(
                    TDHFFullProjectorAnchorEvidence(
                        probe_label=direction.label,
                        anchor_label=anchor.label,
                        step=h,
                        maximum_dF_anchor_residual=residual,
                        passed=True,
                    )
                )

    dF_response_informative = bool(
        maximum_dF_response >= TDHF_FULL_PROJECTOR_DF_RESPONSE_MINIMUM
        and maximum_taylor_signal_to_roundoff_ratio > 1.0
    )
    if plan.require_informative_df and not dF_response_informative:
        raise ValueError(
            "required dF response is uninformative: "
            f"max ||dF[D]||_F={maximum_dF_response:.6e}, "
            "locked minimum="
            f"{TDHF_FULL_PROJECTOR_DF_RESPONSE_MINIMUM:.6e}, "
            "max Taylor/roundoff="
            f"{maximum_taylor_signal_to_roundoff_ratio:.6e}"
        )

    unitary_records: list[TDHFFullProjectorUnitaryProbeEvidence] = []
    for probe in plan.unitary_projector_probes:
        energy = executor.energy(probe.projector)
        fock = executor.matrix("fock", probe.projector)
        unitary_records.append(
            TDHFFullProjectorUnitaryProbeEvidence(
                label=probe.label,
                probe_fingerprint=probe.fingerprint,
                source_projector_fingerprint=probe.source_projector_fingerprint,
                projector_fingerprint=probe.projector_fingerprint,
                trace_fingerprint=probe.trace_fingerprint,
                idempotency_fingerprint=probe.idempotency_fingerprint,
                energy=energy,
                fock_fingerprint=_array_sha256(fock),
                energy_executed=True,
                fock_executed=True,
            )
        )

    inputs_after = inputs.fingerprint
    binding_after = binding.fingerprint
    if inputs_before != inputs_after or binding_before != binding_after:
        raise ValueError("callback/module/source/code/dependency/input state changed")
    complete_inventory = plan.complete_basis_inventory_fingerprint
    full_consistency = bool(
        plan.directions_are_complete
        and complete_inventory == plan.direction_inventory_fingerprint
    )
    unitary_inventory = (
        _fingerprint(tuple(item.probe_fingerprint for item in unitary_records))
        if unitary_records
        else None
    )
    return TDHFFullProjectorFunctionalEvidenceReceipt(
        _factory_token=_RECEIPT_TOKEN,
        api_version=TDHF_FULL_PROJECTOR_SCALAR_FUNCTIONAL_API_VERSION,
        approval_fingerprint=approval.fingerprint,
        space_fingerprint=space.fingerprint,
        inputs_fingerprint_before=inputs_before,
        inputs_fingerprint_after=inputs_after,
        binding_fingerprint_before=binding_before,
        binding_fingerprint_after=binding_after,
        plan_fingerprint=plan.fingerprint,
        source_projector_fingerprint=_array_sha256(p0),
        source_fock=f0,
        source_fock_fingerprint=_array_sha256(f0),
        source_commutator_residual=commutator,
        source_qfp_residual=qfp,
        direction_inventory_fingerprint=plan.direction_inventory_fingerprint,
        complete_basis_inventory_fingerprint=complete_inventory,
        step_evidence=tuple(step_records),
        anchor_evidence=tuple(anchor_records),
        pairing_evidence=tuple(pairing_records),
        unitary_probe_evidence=tuple(unitary_records),
        callback_invocation_counts=tuple(sorted(executor.invocation_counts.items())),
        maximum_dF_response_frobenius_norm=maximum_dF_response,
        maximum_taylor_fock_signal_frobenius_norm=maximum_taylor_signal,
        maximum_taylor_signal_to_roundoff_ratio=(
            maximum_taylor_signal_to_roundoff_ratio
        ),
        all_registered_steps_passed=True,
        source_stationarity_verified=True,
        dF_anchor_independence_verified=True,
        dF_real_self_adjoint_verified=True,
        dF_response_informativeness_required=plan.require_informative_df,
        dF_response_informative=dF_response_informative,
        callback_trace_verified=True,
        callback_source_code_dependency_stable=True,
        registered_probe_functional_consistency=True,
        full_projector_functional_consistency=full_consistency,
        directions_are_complete=plan.directions_are_complete,
        exact_unitary_projector_probes_executed=bool(unitary_records),
        exact_unitary_projector_probe_count=len(unitary_records),
        exact_unitary_projector_probe_inventory_fingerprint=unitary_inventory,
    )


__all__ = [
    "TDHF_FULL_PROJECTOR_CONVENTION",
    "TDHF_FULL_PROJECTOR_DF_RESPONSE_MINIMUM",
    "TDHF_FULL_PROJECTOR_DIRECTION_NORMALIZATION_TOLERANCE",
    "TDHF_FULL_PROJECTOR_DIRECTION_SIGNAL_MINIMUM",
    "TDHF_FULL_PROJECTOR_ENERGY_NORMALIZATION",
    "TDHF_FULL_PROJECTOR_PAIRING",
    "TDHF_FULL_PROJECTOR_SCALAR_FUNCTIONAL_API_VERSION",
    "TDHF_FULL_PROJECTOR_SMALL_BASIS_MAXIMUM_DIMENSION",
    "TDHFFullProjectorAnchorEvidence",
    "TDHFFullProjectorDirection",
    "TDHFFullProjectorFunctionalApproval",
    "TDHFFullProjectorFunctionalBinding",
    "TDHFFullProjectorFunctionalEvidenceReceipt",
    "TDHFFullProjectorPairingEvidence",
    "TDHFFullProjectorSpace",
    "TDHFFullProjectorStepEvidence",
    "TDHFFullProjectorUnitaryProbe",
    "TDHFFullProjectorUnitaryProbeEvidence",
    "TDHFFullProjectorValidationPlan",
    "TDHFFullProjectorValidationTolerances",
    "TDHFScalarCodeManifest",
    "TDHFScalarDependencyBinding",
    "TDHFScalarFunctionalInput",
    "TDHFScalarFunctionalInputsManifest",
    "TDHFScalarKernelBinding",
    "TDHFScalarKernelManifest",
    "bind_tdhf_scalar_kernel",
    "complete_hermitian_basis_inventory_fingerprint",
    "deterministic_complete_hermitian_basis",
    "make_tdhf_full_projector_functional_approval",
    "make_tdhf_full_projector_unitary_probe",
    "make_tdhf_scalar_functional_inputs_manifest",
    "validate_tdhf_full_projector_functional",
]
