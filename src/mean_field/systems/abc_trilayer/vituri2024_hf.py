"""Translational equal-weight finite-domain HF quadrature for Vituri-2024.

This module is the production-scalable specialization of
:mod:`vituri2024_tdhf_full_functional` to translation-preserving densities

``P[(flavor,k),(flavor',k')] = delta[k,k'] * rho[flavor,flavor',k]``.

The stored native convention is ``rho_ab(k)=<c_a^dagger c_b>``.  Therefore the
conventional full density block is ``P_ab(k)=rho_ba(k)``.  The interaction is
divided by the finite area exactly once.  Momentum is a literal finite-domain
coordinate: there is no torus wrap, reciprocal carry, tolerance inclusion,
averaging, or post-Hermitization.

The specialization is an algebraic candidate for independent reproduction.  It
includes the analytic q=0 direct term only to remain exactly comparable with
the existing full oracle; it is not yet an authorized physical Hartree
background.  Consequently it must not be used for branch-energy comparison or
SCF before a separate fixed-density background choice is derived and tested.
It does not establish the paper's missing UV domain, gate distance, q=0 Hartree
background, SCF branch, production result, or paper reproduction.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from hashlib import sha256
import inspect
import json
import math
from numbers import Real
from types import CodeType
from pathlib import Path
from typing import Final

import numpy as np

from .vituri2024_hf_preflight import (
    ACTIVE_BAND_STATES_VALLEY_ORDER,
    INTERNAL_FLAVOR_ORDER,
)
from .vituri2024_interaction import (
    InteractionInput,
    Vituri2024InteractionBinding,
    Vituri2024InteractionChoiceReceipt,
    vituri2024_vtf,
)
from .vituri2024_tdhf_full_scalar import (
    vituri2024_full_operator_to_payload_k_diagonal,
    vituri2024_full_projector_to_payload_density,
    vituri2024_payload_density_to_full_projector,
    vituri2024_payload_operator_to_full_dense,
)

Array = np.ndarray

VITURI2024_TRANSLATIONAL_HF_API_VERSION: Final[str] = (
    "vituri2024_translational_equal_weight_finite_domain_hf.v1"
)
VITURI2024_TRANSLATIONAL_HF_AUTHORITY: Final[str] = (
    "translation_preserving_specialization_of_projected_H_functional_"
    "not_source_uv_q0_scf_production_or_paper_qualified"
)
VITURI2024_TRANSLATIONAL_HF_CONVENTION: Final[str] = (
    "stored_rho_abk=<c_a_dagger c_b>; conventional_P_abk=rho_bak; Q=P-R; "
    "finite_domain_no_wrap; interaction_divided_by_area_once"
)
VITURI2024_TRANSLATIONAL_HF_STRUCTURE_TOLERANCE: Final[float] = 5.0e-11
VITURI2024_TRANSLATIONAL_Q0_POLICY: Final[str] = (
    "retain_finite_dual_gate_q0_direct_and_exchange_explicitly_no_identity_quotient"
)
VITURI2024_TRANSLATIONAL_MESH_POLICY: Final[str] = (
    "explicit_equal_weight_finite_domain_quadrature_no_wrap_no_reciprocal_carry_"
    "not_finite_box_geometry_closed"
)

_Q0_TOKEN = object()
_MESH_TOKEN = object()


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
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _readonly(value: Array, *, dtype: np.dtype | None = None) -> Array:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    result.setflags(write=False)
    return result


def _max_abs(value: object) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _canonical_live_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": value.shape,
            "sha256": _array_sha256(value),
        }
    if isinstance(value, np.generic):
        return _canonical_live_value(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, tuple):
        return ["tuple", *(_canonical_live_value(item) for item in value)]
    if isinstance(value, list):
        return ["list", *(_canonical_live_value(item) for item in value)]
    if isinstance(value, dict):
        return {
            str(key): _canonical_live_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonical_live_value(item) for item in value]
        return [
            type(value).__name__,
            *sorted(
                items,
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":"), allow_nan=False
                ),
            ),
        ]
    if inspect.isfunction(value):
        return {
            "function_module": value.__module__,
            "function_qualname": value.__qualname__,
            "code": _stable_code_record(value.__code__),
        }
    if callable(value) and type(value).__module__.startswith("numpy"):
        return {
            "callable_type_module": type(value).__module__,
            "callable_type_qualname": type(value).__qualname__,
            "callable_module": getattr(value, "__module__", ""),
            "callable_name": getattr(value, "__name__", ""),
        }
    if isinstance(value, type):
        return {"type_module": value.__module__, "type_qualname": value.__qualname__}
    state = getattr(value, "__dict__", None)
    if state in (None, {}):
        return {
            "singleton_type_module": type(value).__module__,
            "singleton_type_qualname": type(value).__qualname__,
        }
    raise TypeError(
        "unsupported nondeterministic translational live fingerprint value "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _canonical_code_constant(value: object) -> object:
    if isinstance(value, CodeType):
        return _stable_code_record(value)
    if isinstance(value, bytes):
        return {"bytes_sha256": sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, tuple):
        return ["tuple", *(_canonical_code_constant(item) for item in value)]
    if isinstance(value, frozenset):
        items = [_canonical_code_constant(item) for item in value]
        return [
            "frozenset",
            *sorted(
                items,
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":"), allow_nan=False
                ),
            ),
        ]
    if value is Ellipsis:
        return {"singleton": "builtins.Ellipsis"}
    if value is NotImplemented:
        return {"singleton": "builtins.NotImplemented"}
    return _canonical_live_value(value)


def _stable_code_record(code: CodeType) -> object:
    """Return interpreter-specialization-independent Python code semantics."""

    return {
        "co_code_sha256": sha256(bytes(code.co_code)).hexdigest(),
        "co_consts": tuple(_canonical_code_constant(item) for item in code.co_consts),
        "co_names": code.co_names,
        "co_varnames": code.co_varnames,
        "co_freevars": code.co_freevars,
        "co_cellvars": code.co_cellvars,
        "co_argcount": code.co_argcount,
        "co_posonlyargcount": code.co_posonlyargcount,
        "co_kwonlyargcount": code.co_kwonlyargcount,
        "co_nlocals": code.co_nlocals,
        "co_stacksize": code.co_stacksize,
        "co_flags": code.co_flags,
        "co_exceptiontable_sha256": sha256(
            bytes(getattr(code, "co_exceptiontable", b""))
        ).hexdigest(),
    }


def _live_callable_fingerprint(value: object) -> str:
    if not inspect.isfunction(value):
        raise TypeError("translational live binding must be a function")
    closure = value.__closure__
    closure_values = () if closure is None else tuple(
        _canonical_live_value(cell.cell_contents) for cell in closure
    )
    return _fingerprint(
        {
            "module": value.__module__,
            "qualname": value.__qualname__,
            "code": _stable_code_record(value.__code__),
            "defaults": _canonical_live_value(value.__defaults__),
            "kwdefaults": _canonical_live_value(value.__kwdefaults__),
            "closure": closure_values,
        }
    )


def _module_live_inventory(module_name: str) -> tuple[tuple[str, str], ...]:
    module = __import__(module_name, fromlist=["*"])
    records: list[tuple[str, str]] = []
    for object_name, value in sorted(vars(module).items()):
        if inspect.isfunction(value) and value.__module__ == module_name:
            records.append((object_name, _live_callable_fingerprint(value)))
        elif inspect.isclass(value) and value.__module__ == module_name:
            for member_name, raw_member in sorted(vars(value).items()):
                if inspect.isfunction(raw_member):
                    members = (raw_member,)
                elif isinstance(raw_member, (staticmethod, classmethod)):
                    members = (raw_member.__func__,)
                elif isinstance(raw_member, property):
                    members = tuple(
                        item
                        for item in (raw_member.fget, raw_member.fset, raw_member.fdel)
                        if item is not None
                    )
                else:
                    members = ()
                for member in members:
                    records.append(
                        (
                            f"{object_name}.{member_name}",
                            _live_callable_fingerprint(member),
                        )
                    )
    return tuple(records)


def _module_semantic_inventory(module_name: str) -> tuple[tuple[str, object], ...]:
    module = __import__(module_name, fromlist=["*"])
    supported = (str, int, float, complex, bool, tuple, list, dict, np.ndarray, np.generic)
    return tuple(
        (name, _canonical_live_value(value))
        for name, value in sorted(vars(module).items())
        if name.isupper() and isinstance(value, supported)
    )


def _implementation_fingerprint() -> str:
    module_functions = tuple(
        (name, value)
        for name, value in sorted(globals().items())
        if inspect.isfunction(value) and value.__module__ == __name__
    )
    runtime_bindings = (
        ("vituri2024_vtf", vituri2024_vtf),
        ("payload_density_to_full", vituri2024_payload_density_to_full_projector),
        ("payload_operator_to_full", vituri2024_payload_operator_to_full_dense),
        ("full_projector_to_payload", vituri2024_full_projector_to_payload_density),
        ("full_operator_to_payload", vituri2024_full_operator_to_payload_k_diagonal),
    )
    class_functions = tuple(
        (class_name, name, member)
        for class_name, class_type in (
            ("Vituri2024TranslationalQ0ReproductionChoice", Vituri2024TranslationalQ0ReproductionChoice),
            ("Vituri2024FiniteDomainMeshReceipt", Vituri2024FiniteDomainMeshReceipt),
            ("Vituri2024TranslationalHFFunctional", Vituri2024TranslationalHFFunctional),
        )
        for name, raw_member in sorted(vars(class_type).items())
        for member in (
            tuple(
                item
                for item in (raw_member.fget, raw_member.fset, raw_member.fdel)
                if item is not None
            )
            if isinstance(raw_member, property)
            else (raw_member,)
        )
        if inspect.isfunction(member)
    )
    semantic_constants = tuple(
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("VITURI2024_TRANSLATIONAL_")
        and isinstance(value, (str, int, float, bool, tuple))
    )
    modules = {
        function.__module__
        for function in (
            vituri2024_vtf,
            vituri2024_payload_density_to_full_projector,
            vituri2024_payload_operator_to_full_dense,
            vituri2024_full_projector_to_payload_density,
            vituri2024_full_operator_to_payload_k_diagonal,
        )
    }
    modules.add(__name__)
    source_records = []
    dependency_live_inventories = []
    dependency_semantic_inventories = []
    for module_name in sorted(modules):
        module = __import__(module_name, fromlist=["*"])
        source_file = inspect.getsourcefile(module)
        if source_file is None:
            raise RuntimeError(f"cannot locate translational dependency {module_name}")
        source_records.append(
            (module_name, sha256(Path(source_file).read_bytes()).hexdigest())
        )
        dependency_live_inventories.append(
            (module_name, _module_live_inventory(module_name))
        )
        dependency_semantic_inventories.append(
            (module_name, _module_semantic_inventory(module_name))
        )
    return _fingerprint(
        {
            "module_sources": tuple(source_records),
            "dependency_live_inventories": tuple(dependency_live_inventories),
            "dependency_semantic_inventories": tuple(
                dependency_semantic_inventories
            ),
            "module_functions": tuple(
                (name, _live_callable_fingerprint(function))
                for name, function in module_functions
                if name not in ("_implementation_fingerprint",)
            ),
            "runtime_bindings": tuple(
                (
                    name,
                    function.__module__,
                    function.__qualname__,
                    _live_callable_fingerprint(function),
                )
                for name, function in runtime_bindings
            ),
            "class_functions": tuple(
                (class_name, name, _live_callable_fingerprint(function))
                for class_name, name, function in class_functions
            ),
            "flavor_order": INTERNAL_FLAVOR_ORDER,
            "valley_order": ACTIVE_BAND_STATES_VALLEY_ORDER,
            "semantic_constants": semantic_constants,
            "structure_tolerance": VITURI2024_TRANSLATIONAL_HF_STRUCTURE_TOLERANCE,
        }
    )


@dataclass(frozen=True, slots=True)
class Vituri2024TranslationalQ0ReproductionChoice:
    _factory_token: InitVar[object]
    evidence: str
    fingerprint: str = field(init=False)
    policy: str = field(default=VITURI2024_TRANSLATIONAL_Q0_POLICY, init=False)
    kernel_limit_retained: bool = field(default=True, init=False)
    identity_quotient_authorized: bool = field(default=False, init=False)
    establishes_paper_or_source_q0_background: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _Q0_TOKEN:
            raise TypeError("translational q0 choice is factory-only")
        self._validate_fields(check_fingerprint=False)
        object.__setattr__(self, "fingerprint", self._current_fingerprint())

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                "policy": self.policy,
                "evidence": self.evidence,
                "kernel_limit_retained": self.kernel_limit_retained,
                "identity_quotient_authorized": self.identity_quotient_authorized,
                "establishes_paper_or_source_q0_background": (
                    self.establishes_paper_or_source_q0_background
                ),
            }
        )

    def _validate_fields(self, *, check_fingerprint: bool) -> None:
        if type(self.evidence) is not str or not self.evidence.strip():
            raise ValueError("translational q0 choice needs explicit evidence")
        locked = (
            self.policy == VITURI2024_TRANSLATIONAL_Q0_POLICY,
            self.kernel_limit_retained is True,
            self.identity_quotient_authorized is False,
            self.establishes_paper_or_source_q0_background is False,
        )
        if not all(locked):
            raise ValueError("translational q0 authority was inflated")
        if check_fingerprint and self._current_fingerprint() != self.fingerprint:
            raise ValueError("translational q0 choice fingerprint drifted")

    def validate_live_state(self) -> None:
        self._validate_fields(check_fingerprint=True)


def make_vituri2024_translational_q0_reproduction_choice(
    *, evidence: str,
) -> Vituri2024TranslationalQ0ReproductionChoice:
    return Vituri2024TranslationalQ0ReproductionChoice(
        _factory_token=_Q0_TOKEN, evidence=evidence
    )


@dataclass(frozen=True, slots=True)
class Vituri2024FiniteDomainMeshReceipt:
    _factory_token: InitVar[object]
    ordered_mesh: Array
    area_angstrom_squared: float
    provenance: str
    nk: int = field(init=False)
    uniform_weight_inverse_angstrom_squared: float = field(init=False)
    fingerprint: str = field(init=False)
    policy: str = field(default=VITURI2024_TRANSLATIONAL_MESH_POLICY, init=False)
    production_mesh_convergence_established: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _MESH_TOKEN:
            raise TypeError("translational mesh receipt is factory-only")
        if (
            not isinstance(self.ordered_mesh, np.ndarray)
            or self.ordered_mesh.dtype != np.dtype(np.float64)
            or self.ordered_mesh.ndim != 2
            or self.ordered_mesh.shape[1] != 2
            or self.ordered_mesh.shape[0] < 1
            or not np.all(np.isfinite(self.ordered_mesh))
        ):
            raise ValueError("mesh receipt requires finite float64 (Nk,2)")
        if isinstance(self.area_angstrom_squared, (bool, np.bool_)) or not isinstance(
            self.area_angstrom_squared, Real
        ):
            raise TypeError("mesh receipt area must be a strict real scalar")
        area = float(self.area_angstrom_squared)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("mesh receipt area must be positive")
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("mesh receipt provenance must be explicit")
        mesh = _readonly(self.ordered_mesh, dtype=np.dtype(np.float64))
        nk = int(mesh.shape[0])
        object.__setattr__(self, "ordered_mesh", mesh)
        object.__setattr__(self, "area_angstrom_squared", area)
        object.__setattr__(self, "nk", nk)
        object.__setattr__(self, "uniform_weight_inverse_angstrom_squared", 1.0 / area)
        object.__setattr__(self, "fingerprint", self._current_fingerprint())
        self.validate_live_state()

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                "mesh": _array_sha256(self.ordered_mesh),
                "area": self.area_angstrom_squared,
                "weight": self.uniform_weight_inverse_angstrom_squared,
                "policy": self.policy,
                "provenance": self.provenance,
                "production_mesh_convergence_established": (
                    self.production_mesh_convergence_established
                ),
            }
        )

    def validate_live_state(self) -> None:
        if (
            not isinstance(self.ordered_mesh, np.ndarray)
            or self.ordered_mesh.dtype != np.dtype(np.float64)
            or self.ordered_mesh.ndim != 2
            or self.ordered_mesh.shape != (self.nk, 2)
            or self.ordered_mesh.flags.writeable
            or not np.all(np.isfinite(self.ordered_mesh))
            or type(self.area_angstrom_squared) is not float
            or not math.isfinite(self.area_angstrom_squared)
            or self.area_angstrom_squared <= 0.0
            or self.uniform_weight_inverse_angstrom_squared
            != 1.0 / self.area_angstrom_squared
            or self.policy != VITURI2024_TRANSLATIONAL_MESH_POLICY
            or self.production_mesh_convergence_established is not False
            or type(self.provenance) is not str
            or not self.provenance.strip()
        ):
            raise ValueError("translational mesh receipt live state drifted")
        if self._current_fingerprint() != self.fingerprint:
            raise ValueError("translational mesh receipt fingerprint drifted")


def make_vituri2024_finite_domain_mesh_receipt(
    *, ordered_mesh: Array, area_angstrom_squared: float, provenance: str,
) -> Vituri2024FiniteDomainMeshReceipt:
    return Vituri2024FiniteDomainMeshReceipt(
        _factory_token=_MESH_TOKEN,
        ordered_mesh=ordered_mesh,
        area_angstrom_squared=area_angstrom_squared,
        provenance=provenance,
    )


def _resolve_interaction(
    interaction: InteractionInput,
) -> tuple[Vituri2024InteractionChoiceReceipt, str]:
    if type(interaction) is Vituri2024InteractionChoiceReceipt:
        receipt = interaction
        fingerprint = receipt.fingerprint
    elif type(interaction) is Vituri2024InteractionBinding:
        receipt = interaction.receipt
        if (
            interaction.receipt_fingerprint != receipt.fingerprint
            or interaction.paper_direct_claim_allowed is not False
            or interaction.establishes_hf_q0_background is not False
        ):
            raise ValueError("translational interaction binding is stale or inflated")
        fingerprint = interaction.receipt_fingerprint
    else:
        raise TypeError("translational interaction must be exact receipt or binding")
    clean = Vituri2024InteractionChoiceReceipt(
        gate_distance_angstrom=receipt.gate_distance_angstrom,
        coulomb_e2_ev_angstrom=receipt.coulomb_e2_ev_angstrom,
        q0_evaluation=receipt.q0_evaluation,
        provider_sha256=receipt.provider_sha256,
        source_sha256=receipt.source_sha256,
        authority_kind=receipt.authority_kind,
        source_text=receipt.source_text,
    )
    if clean != receipt or clean.q0_evaluation != "analytic_kernel_limit_only":
        raise ValueError("translational HF requires a finite analytic q=0 kernel")
    return clean, fingerprint


def vituri2024_native_density_to_conventional_k_diagonal(density: Array) -> Array:
    """Map stored ``rho_ab=<c_a†c_b>`` to conventional ``P_ab=<c_b†c_a>``."""

    if (
        not isinstance(density, np.ndarray)
        or density.dtype != np.dtype(np.complex128)
        or density.ndim != 3
        or density.shape[:2]
        != (len(INTERNAL_FLAVOR_ORDER), len(INTERNAL_FLAVOR_ORDER))
        or density.shape[2] < 1
        or not np.all(np.isfinite(density))
    ):
        raise ValueError("native density must be finite complex128 (4,4,Nk)")
    residual = _max_abs(density - density.swapaxes(0, 1).conj())
    if residual > 64.0 * np.finfo(np.float64).eps * max(1.0, _max_abs(density)):
        raise ValueError("native density blocks must be Hermitian")
    return _readonly(density.swapaxes(0, 1), dtype=np.dtype(np.complex128))


def vituri2024_conventional_k_diagonal_to_native_density(projector: Array) -> Array:
    """Map conventional k-diagonal ``P`` to stored native density blocks."""

    return vituri2024_native_density_to_conventional_k_diagonal(projector)


def vituri2024_native_operator_to_conventional_k_diagonal(operator: Array) -> Array:
    """Validate an operator block array; operators are not transposed."""

    if (
        not isinstance(operator, np.ndarray)
        or operator.dtype != np.dtype(np.complex128)
        or operator.ndim != 3
        or operator.shape[:2]
        != (len(INTERNAL_FLAVOR_ORDER), len(INTERNAL_FLAVOR_ORDER))
        or operator.shape[2] < 1
        or not np.all(np.isfinite(operator))
    ):
        raise ValueError("native operator must be finite complex128 (4,4,Nk)")
    residual = _max_abs(operator - operator.swapaxes(0, 1).conj())
    if residual > 64.0 * np.finfo(np.float64).eps * max(1.0, _max_abs(operator)):
        raise ValueError("native operator blocks must be Hermitian")
    return _readonly(operator, dtype=np.dtype(np.complex128))


def vituri2024_translational_interaction_action_conventional(
    density: Array,
    *,
    form_factors_by_flavor: Array,
    kernel_by_mesh_pair: Array,
    area_angstrom_squared: float,
    _density_validator=vituri2024_native_operator_to_conventional_k_diagonal,
    _zeros_like=np.zeros_like,
    _diagonal=np.diagonal,
    _sum=np.sum,
    _einsum=np.einsum,
    _max_abs_function=_max_abs,
    _readonly_function=_readonly,
    _structure_tolerance: float = VITURI2024_TRANSLATIONAL_HF_STRUCTURE_TOLERANCE,
) -> Array:
    """Apply the validated translational direct/exchange action in O(Nk^2)."""

    nk = int(form_factors_by_flavor.shape[1])
    clean = _density_validator(density)
    if clean.shape != (4, 4, nk):
        raise ValueError("translational action density Nk mismatch")
    flavors = len(INTERNAL_FLAVOR_ORDER)
    result = _zeros_like(clean)
    direct_scalar = 0.0 + 0.0j
    for flavor in range(flavors):
        direct_scalar += _sum(
            _diagonal(form_factors_by_flavor[flavor])
            * clean[flavor, flavor, :]
        )
    direct_scalar *= kernel_by_mesh_pair[0, 0]
    for flavor in range(flavors):
        result[flavor, flavor, :] += (
            _diagonal(form_factors_by_flavor[flavor]) * direct_scalar
        )
    for left_flavor in range(flavors):
        left_form_factor = form_factors_by_flavor[left_flavor]
        for right_flavor in range(flavors):
            result[left_flavor, right_flavor, :] -= _einsum(
                "mr,mr,rm,r->m",
                kernel_by_mesh_pair,
                left_form_factor,
                form_factors_by_flavor[right_flavor],
                clean[left_flavor, right_flavor, :],
                optimize=True,
            )
    result *= 1.0 / area_angstrom_squared
    residual = _max_abs_function(result - result.swapaxes(0, 1).conj())
    if residual > _structure_tolerance * max(
        1.0, _max_abs_function(result)
    ):
        raise ValueError("translational interaction action is not Hermitian")
    return _readonly_function(result, dtype=np.dtype(np.complex128))


@dataclass(frozen=True, slots=True)
class Vituri2024TranslationalHFFunctional:
    """Scalable translation-preserving E/F/dF specialization."""

    ordered_mesh: Array
    active_band_states: Array
    h0_native: Array
    normal_order_reference_native: Array
    mesh_receipt: Vituri2024FiniteDomainMeshReceipt
    interaction: InteractionInput
    normal_order_reference_fingerprint: str
    q0_choice: Vituri2024TranslationalQ0ReproductionChoice
    provenance: str
    normal_order_reference_conventional: Array = field(init=False, repr=False)
    form_factors_by_flavor: Array = field(init=False, repr=False)
    kernel_by_mesh_pair: Array = field(init=False, repr=False)
    interaction_receipt: Vituri2024InteractionChoiceReceipt = field(init=False)
    interaction_fingerprint: str = field(init=False)
    nk: int = field(init=False)
    implementation_fingerprint: str = field(init=False)
    construction_fingerprint: str = field(init=False)
    api_version: str = field(default=VITURI2024_TRANSLATIONAL_HF_API_VERSION, init=False)
    authority: str = field(default=VITURI2024_TRANSLATIONAL_HF_AUTHORITY, init=False)
    convention: str = field(default=VITURI2024_TRANSLATIONAL_HF_CONVENTION, init=False)
    source_stationarity_established: bool = field(default=False, init=False)
    q0_background_authority_established: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.ordered_mesh, np.ndarray)
            or self.ordered_mesh.dtype != np.dtype(np.float64)
            or self.ordered_mesh.ndim != 2
            or self.ordered_mesh.shape[1] != 2
            or self.ordered_mesh.shape[0] < 1
            or not np.all(np.isfinite(self.ordered_mesh))
        ):
            raise ValueError("translational mesh must be finite float64 (Nk,2)")
        mesh = _readonly(self.ordered_mesh, dtype=np.dtype(np.float64))
        nk = int(mesh.shape[0])
        if len({(float(k[0]), float(k[1])) for k in mesh}) != nk:
            raise ValueError("translational mesh contains duplicate exact coordinates")
        for m in range(nk):
            for r in range(nk):
                residual = (
                    float(mesh[m, 0] + mesh[r, 0] - mesh[r, 0] - mesh[m, 0]),
                    float(mesh[m, 1] + mesh[r, 1] - mesh[r, 1] - mesh[m, 1]),
                )
                if residual != (0.0, 0.0):
                    raise ValueError(
                        "translational mesh fails literal quartet conservation in "
                        "the full-oracle arithmetic order"
                    )
        if (
            not isinstance(self.active_band_states, np.ndarray)
            or self.active_band_states.dtype != np.dtype(np.complex128)
            or self.active_band_states.shape
            != (len(ACTIVE_BAND_STATES_VALLEY_ORDER), 6, nk)
            or not np.all(np.isfinite(self.active_band_states))
        ):
            raise ValueError("active states must be finite complex128 (2,6,Nk)")
        states = _readonly(self.active_band_states, dtype=np.dtype(np.complex128))
        norm_residual = _max_abs(np.sum(np.abs(states) ** 2, axis=1) - 1.0)
        if norm_residual > 5.0e-12:
            raise ValueError("translational active states are not normalized")
        h0 = vituri2024_native_operator_to_conventional_k_diagonal(self.h0_native)
        if (
            not isinstance(self.normal_order_reference_native, np.ndarray)
            or self.normal_order_reference_native.dtype != np.dtype(np.complex128)
        ):
            raise TypeError("translational native reference must be exact complex128")
        reference_native = _readonly(self.normal_order_reference_native)
        reference = vituri2024_native_density_to_conventional_k_diagonal(
            reference_native
        )
        if h0.shape[2] != nk or reference.shape[2] != nk:
            raise ValueError("translational h0/reference Nk mismatch")
        for momentum in range(nk):
            eigenvalues = np.linalg.eigvalsh(reference[:, :, momentum])
            if eigenvalues[0] < -5.0e-12 or eigenvalues[-1] > 1.0 + 5.0e-12:
                raise ValueError("translational reference must satisfy 0<=R<=I")
        if type(self.mesh_receipt) is not Vituri2024FiniteDomainMeshReceipt:
            raise TypeError("translational HF requires an exact mesh receipt")
        if (
            self.mesh_receipt.nk != nk
            or not np.array_equal(self.mesh_receipt.ordered_mesh, mesh)
        ):
            raise ValueError("translational mesh/receipt binding drifted")
        if type(self.q0_choice) is not Vituri2024TranslationalQ0ReproductionChoice:
            raise TypeError("translational HF requires an exact q0 reproduction choice")
        interaction, interaction_fingerprint = _resolve_interaction(self.interaction)
        _sha256(self.normal_order_reference_fingerprint, "normal reference")
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("translational provenance must be explicit")
        valley_index = {
            valley: index
            for index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER)
        }
        form_by_valley = np.einsum(
            "vcm,vcn->vmn", states.conj(), states, optimize=False
        )
        form_factors = np.stack(
            [
                form_by_valley[valley_index[valley]]
                for valley, _spin in INTERNAL_FLAVOR_ORDER
            ],
            axis=0,
        ).astype(np.complex128, copy=False)
        if _max_abs(form_factors - form_factors.swapaxes(1, 2).conj()) > 5.0e-12:
            raise ValueError("translational form factors violate pair conjugation")
        kernel = np.empty((nk, nk), dtype=np.float64)
        for left in range(nk):
            for right in range(nk):
                transfer = mesh[left] - mesh[right]
                kernel[left, right] = vituri2024_vtf(
                    float(np.hypot(transfer[0], transfer[1])), interaction
                )
        object.__setattr__(self, "ordered_mesh", mesh)
        object.__setattr__(self, "active_band_states", states)
        object.__setattr__(self, "h0_native", h0)
        object.__setattr__(self, "normal_order_reference_native", reference_native)
        object.__setattr__(self, "normal_order_reference_conventional", reference)
        object.__setattr__(self, "interaction_receipt", interaction)
        object.__setattr__(self, "interaction_fingerprint", interaction_fingerprint)
        object.__setattr__(self, "form_factors_by_flavor", _readonly(form_factors))
        object.__setattr__(self, "kernel_by_mesh_pair", _readonly(kernel))
        object.__setattr__(self, "nk", nk)
        object.__setattr__(self, "implementation_fingerprint", _implementation_fingerprint())
        object.__setattr__(self, "construction_fingerprint", self._current_fingerprint())
        self.validate_live_state()

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": self.api_version,
                "authority": self.authority,
                "convention": self.convention,
                "mesh": _array_sha256(self.ordered_mesh),
                "states": _array_sha256(self.active_band_states),
                "h0": _array_sha256(self.h0_native),
                "reference_native": _array_sha256(self.normal_order_reference_native),
                "reference_conventional": _array_sha256(
                    self.normal_order_reference_conventional
                ),
                "mesh_receipt": self.mesh_receipt.fingerprint,
                "interaction": self.interaction_fingerprint,
                "interaction_input_kind": type(self.interaction).__name__,
                "interaction_binding_flags": (
                    (
                        self.interaction.paper_direct_claim_allowed,
                        self.interaction.establishes_hf_q0_background,
                    )
                    if type(self.interaction) is Vituri2024InteractionBinding
                    else (False, False)
                ),
                "normal_reference_fingerprint": self.normal_order_reference_fingerprint,
                "q0_choice": self.q0_choice.fingerprint,
                "form_factors": _array_sha256(self.form_factors_by_flavor),
                "kernel": _array_sha256(self.kernel_by_mesh_pair),
                "implementation": self.implementation_fingerprint,
                "provenance": self.provenance,
                "authority_flags": (
                    self.source_stationarity_established,
                    self.q0_background_authority_established,
                    self.production_ready,
                    self.paper_reproduction_verified,
                ),
            }
        )

    def validate_live_state(self) -> None:
        arrays = (
            (self.ordered_mesh, np.dtype(np.float64), (self.nk, 2), "mesh"),
            (
                self.active_band_states,
                np.dtype(np.complex128),
                (2, 6, self.nk),
                "states",
            ),
            (self.h0_native, np.dtype(np.complex128), (4, 4, self.nk), "h0"),
            (
                self.normal_order_reference_native,
                np.dtype(np.complex128),
                (4, 4, self.nk),
                "native reference",
            ),
            (
                self.normal_order_reference_conventional,
                np.dtype(np.complex128),
                (4, 4, self.nk),
                "conventional reference",
            ),
            (
                self.form_factors_by_flavor,
                np.dtype(np.complex128),
                (4, self.nk, self.nk),
                "form factors",
            ),
            (
                self.kernel_by_mesh_pair,
                np.dtype(np.float64),
                (self.nk, self.nk),
                "kernel",
            ),
        )
        for value, dtype, shape, label in arrays:
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != dtype
                or value.shape != shape
                or value.flags.writeable
                or not np.all(np.isfinite(value))
            ):
                raise ValueError(f"translational live {label} drifted")
        if type(self.mesh_receipt) is Vituri2024FiniteDomainMeshReceipt:
            self.mesh_receipt.validate_live_state()
        if type(self.q0_choice) is Vituri2024TranslationalQ0ReproductionChoice:
            self.q0_choice.validate_live_state()
        if (
            type(self.mesh_receipt) is not Vituri2024FiniteDomainMeshReceipt
            or self.mesh_receipt.nk != self.nk
            or not np.array_equal(self.mesh_receipt.ordered_mesh, self.ordered_mesh)
        ):
            raise ValueError("translational live mesh receipt drifted")
        if type(self.q0_choice) is not Vituri2024TranslationalQ0ReproductionChoice:
            raise ValueError("translational live q0 choice drifted")
        resolved_interaction, resolved_fingerprint = _resolve_interaction(self.interaction)
        if (
            self.interaction_receipt.fingerprint != self.interaction_fingerprint
            or resolved_interaction != self.interaction_receipt
            or resolved_fingerprint != self.interaction_fingerprint
        ):
            raise ValueError("translational interaction receipt drifted")
        if _implementation_fingerprint() != self.implementation_fingerprint:
            raise ValueError("translational HF implementation drifted")
        locked = (
            type(self.nk) is int,
            self.nk == int(self.ordered_mesh.shape[0]),
            self.api_version == VITURI2024_TRANSLATIONAL_HF_API_VERSION,
            self.authority == VITURI2024_TRANSLATIONAL_HF_AUTHORITY,
            self.convention == VITURI2024_TRANSLATIONAL_HF_CONVENTION,
            self.source_stationarity_established is False,
            self.q0_background_authority_established is False,
            self.q0_choice.establishes_paper_or_source_q0_background is False,
            self.mesh_receipt.production_mesh_convergence_established is False,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
        )
        if not all(locked):
            raise ValueError("translational HF authority was inflated")
        if hasattr(self, "construction_fingerprint") and (
            self._current_fingerprint() != self.construction_fingerprint
        ):
            raise ValueError("translational HF construction drifted")

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.construction_fingerprint

    def _interaction_action_conventional_validated(self, density: Array) -> Array:
        """Hot-loop action after one explicit trusted boundary validation."""

        return vituri2024_translational_interaction_action_conventional(
            density,
            form_factors_by_flavor=self.form_factors_by_flavor,
            kernel_by_mesh_pair=self.kernel_by_mesh_pair,
            area_angstrom_squared=self.mesh_receipt.area_angstrom_squared,
        )

    def interaction_action_conventional(self, density: Array) -> Array:
        """Return k-diagonal conventional ``Sigma[density]`` in O(Nk^2)."""

        self.validate_live_state()
        return self._interaction_action_conventional_validated(density)

    def interaction_action(self, native_density: Array) -> Array:
        self.validate_live_state()
        conventional = vituri2024_native_density_to_conventional_k_diagonal(
            native_density
        )
        return self._interaction_action_conventional_validated(conventional)

    def make_validated_interaction_action(self):
        """Validate once and return the SCF hot-loop linear action closure."""

        self.validate_live_state()

        action_implementation = vituri2024_translational_interaction_action_conventional
        density_converter = vituri2024_native_density_to_conventional_k_diagonal
        form_factors = self.form_factors_by_flavor
        kernel = self.kernel_by_mesh_pair
        area = self.mesh_receipt.area_angstrom_squared

        def action(native_density: Array) -> Array:
            conventional = density_converter(native_density)
            return action_implementation(
                conventional,
                form_factors_by_flavor=form_factors,
                kernel_by_mesh_pair=kernel,
                area_angstrom_squared=area,
            )

        return action

    def energy(self, native_density: Array) -> float:
        conventional = vituri2024_native_density_to_conventional_k_diagonal(
            native_density
        )
        if conventional.shape != (4, 4, self.nk):
            raise ValueError("translational energy density Nk mismatch")
        difference = conventional - self.normal_order_reference_conventional
        self.validate_live_state()
        interaction = self._interaction_action_conventional_validated(difference)
        one_body = np.einsum(
            "abk,bak->", self.h0_native, conventional, optimize=False
        )
        interaction_energy = 0.5 * np.einsum(
            "abk,bak->", interaction, difference, optimize=False
        )
        total = complex(one_body + interaction_energy)
        if abs(total.imag) > VITURI2024_TRANSLATIONAL_HF_STRUCTURE_TOLERANCE * max(
            1.0, abs(total), abs(one_body), abs(interaction_energy)
        ):
            raise ValueError("translational scalar energy is materially complex")
        return float(total.real)

    def fock(self, native_density: Array) -> Array:
        conventional = vituri2024_native_density_to_conventional_k_diagonal(
            native_density
        )
        if conventional.shape != (4, 4, self.nk):
            raise ValueError("translational Fock density Nk mismatch")
        self.validate_live_state()
        interaction = self._interaction_action_conventional_validated(
            conventional - self.normal_order_reference_conventional
        )
        result = self.h0_native + interaction
        return vituri2024_native_operator_to_conventional_k_diagonal(result)

    def fock_derivative(self, native_density: Array, native_direction: Array) -> Array:
        anchor = vituri2024_native_density_to_conventional_k_diagonal(native_density)
        conventional_direction = vituri2024_native_density_to_conventional_k_diagonal(
            native_direction
        )
        if anchor.shape != (4, 4, self.nk) or conventional_direction.shape != (
            4, 4, self.nk
        ):
            raise ValueError("translational dF anchor/direction Nk mismatch")
        self.validate_live_state()
        return self._interaction_action_conventional_validated(conventional_direction)


__all__ = [
    "VITURI2024_TRANSLATIONAL_HF_API_VERSION",
    "VITURI2024_TRANSLATIONAL_HF_AUTHORITY",
    "VITURI2024_TRANSLATIONAL_HF_CONVENTION",
    "VITURI2024_TRANSLATIONAL_MESH_POLICY",
    "VITURI2024_TRANSLATIONAL_Q0_POLICY",
    "Vituri2024FiniteDomainMeshReceipt",
    "Vituri2024TranslationalHFFunctional",
    "Vituri2024TranslationalQ0ReproductionChoice",
    "make_vituri2024_finite_domain_mesh_receipt",
    "make_vituri2024_translational_q0_reproduction_choice",
    "vituri2024_conventional_k_diagonal_to_native_density",
    "vituri2024_native_density_to_conventional_k_diagonal",
    "vituri2024_native_operator_to_conventional_k_diagonal",
    "vituri2024_translational_interaction_action_conventional",
]
