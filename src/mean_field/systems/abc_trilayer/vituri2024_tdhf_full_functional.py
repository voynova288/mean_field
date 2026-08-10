"""Factorized full-projector Vituri-2024 scalar functional kernel.

This module implements the projected-Hamiltonian Wick functional on the full
active-band space, without allocating the dense ``(4*Nk)^4`` vertex tensor.
The conventional density convention is

``P_ij = <c_j^dagger c_i>``.

For one explicit, fixed normal-order reference ``R`` in the same source gauge,

``Q = P - R``
``Sigma[X]_ij = sum_bg wbar[i,b,g,j] X[g,b]``
``E[P] = Tr(h0 P) + 1/2 Tr(Q Sigma[Q])``
``F[P] = h0 + Sigma[Q]``
``dF[P;D] = Sigma[D]``.

Here ``wbar`` is the antisymmetrized projected vertex divided by the finite
area exactly once.  The action uses the exact local continuum condition
``k_alpha+k_beta-k_gamma-k_delta == (0,0)`` in the same arithmetic order as
:mod:`vituri2024_vertex`; it performs no tolerance inclusion, reciprocal-torus
wrap, carry, averaging, symmetrization, or Hermitization.

The active-band spinors are explicit source-gauge inputs.  They are not
recomputed by diagonalizing the six-band model, because mixing an independently
chosen k-dependent phase gauge with saved source projectors would invalidate a
full scalar closure test.

A constructed kernel is actual projected-Hamiltonian algebra, but not source
or paper authority.  In particular, an analytic ``V_TF(0)`` value does not
establish the HF q=0 background, and a caller-supplied reference matrix is not
an immutable normal-order authority.  Source closure therefore requires a
separate fail-closed comparison to saved ``interaction_h`` and full Fock arrays.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from hashlib import sha256
import importlib
import inspect
import json
import marshal
import math
from pathlib import Path
import sys
from typing import Final

import numpy as np

from .vituri2024 import SM_TEX_SHA256
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
from .vituri2024_vertex import (
    VERTEX_AUTHORITY,
    vituri2024_antisymmetrized_projected_vertex,
)

Array = np.ndarray

VITURI2024_FULL_FUNCTIONAL_API_VERSION: Final[str] = (
    "vituri2024_full_projected_functional.v1"
)
VITURI2024_FULL_FUNCTIONAL_AUTHORITY: Final[str] = (
    "projected_H_factorized_functional_kernel_not_source_qualified"
)
VITURI2024_FULL_FUNCTIONAL_SUPPLIED_ARRAY_AUTHORITY: Final[str] = (
    "caller_supplied_array_consistency_only_no_source_stationarity_q0_normal_order_"
    "full_projector_tdhf_scalar_hessian_production_or_paper_authority"
)
VITURI2024_FULL_FUNCTIONAL_CONVENTION: Final[str] = (
    "conventional_dense_P_ij=<c_j^dagger c_i>; Q=P-R; "
    "Sigma[X]_ij=sum_bg wbar[i,b,g,j]X[g,b]"
)
VITURI2024_FULL_FUNCTIONAL_EXACT_LOCAL_MASK: Final[str] = (
    "literal_float64_k_alpha_plus_k_beta_minus_k_gamma_minus_k_delta_eq_zero; "
    "no_tolerance_no_torus_no_carry"
)
VITURI2024_FULL_FUNCTIONAL_Q0_STATUS: Final[str] = (
    "analytic_kernel_value_consumed_but_hf_background_authority_not_established"
)
VITURI2024_FULL_FUNCTIONAL_GAUGE_STATUS: Final[str] = (
    "explicit_source_active_band_states_no_independent_rediagonalization"
)
VITURI2024_FULL_FUNCTIONAL_ENERGY_EQUATION: Final[str] = (
    "E[P]=Tr(h0 P)+1/2 Tr((P-R) Sigma[P-R])"
)
VITURI2024_FULL_FUNCTIONAL_FOCK_EQUATION: Final[str] = (
    "F[P]=h0+Sigma[P-R]"
)
VITURI2024_FULL_FUNCTIONAL_DF_EQUATION: Final[str] = "dF[P;D]=Sigma[D]"
VITURI2024_FULL_FUNCTIONAL_STRUCTURE_TOLERANCE: Final[float] = 5.0e-11
VITURI2024_FULL_FUNCTIONAL_SUPPLIED_ARRAY_TOLERANCE: Final[float] = 1.0e-10

_SUPPLIED_ARRAY_TOKEN = object()


def _locked_structure_tolerance() -> float:
    if VITURI2024_FULL_FUNCTIONAL_STRUCTURE_TOLERANCE != 5.0e-11:
        raise RuntimeError("full functional structure tolerance drifted from locked v1")
    return 5.0e-11


def _locked_supplied_array_tolerance() -> float:
    if VITURI2024_FULL_FUNCTIONAL_SUPPLIED_ARRAY_TOLERANCE != 1.0e-10:
        raise RuntimeError("supplied-array tolerance drifted from locked v1")
    return 1.0e-10


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


def _max_abs(value: object) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return 0.0
    result = float(np.max(np.abs(array)))
    if not math.isfinite(result):
        raise ValueError("nonfinite full-functional residual")
    return result


def _readonly_exact_array(
    value: object,
    *,
    label: str,
    dtype: np.dtype,
    shape: tuple[int, ...] | None = None,
) -> Array:
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise TypeError(f"{label} must be an exact {dtype} numpy array")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{label} shape must be exactly {shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite")
    result = np.frombuffer(value.tobytes(order="C"), dtype=dtype).reshape(value.shape)
    result.setflags(write=False)
    return result


def _readonly_hermitian(
    value: object, *, label: str, dimension: int | None = None
) -> Array:
    shape = None if dimension is None else (dimension, dimension)
    result = _readonly_exact_array(
        value, label=label, dtype=np.dtype(np.complex128), shape=shape
    )
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{label} must be square")
    residual = _max_abs(result - result.conj().T)
    scale = max(1.0, _max_abs(result))
    if residual > 64.0 * np.finfo(np.float64).eps * scale:
        raise ValueError(f"{label} must be Hermitian")
    return result


def _readonly_reference(
    value: object, *, label: str, dimension: int
) -> Array:
    result = _readonly_hermitian(value, label=label, dimension=dimension)
    eigenvalues = np.linalg.eigvalsh(result)
    tolerance = 5.0e-12
    if float(eigenvalues[0]) < -tolerance or float(eigenvalues[-1]) > 1.0 + tolerance:
        raise ValueError(f"{label} must be a representable density with 0<=R<=I")
    return result


def _bytes_backed(value: Array, *, dtype: np.dtype | None = None) -> Array:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    result.setflags(write=False)
    return result


def _readonly_result(value: Array) -> Array:
    return _bytes_backed(value, dtype=np.dtype(np.complex128))


def _implementation_source_fingerprints() -> tuple[tuple[str, str], ...]:
    module_names = (
        __name__,
        vituri2024_vtf.__module__,
        vituri2024_antisymmetrized_projected_vertex.__module__,
        "mean_field.systems.abc_trilayer.vituri2024_hf_preflight",
        "mean_field.systems.abc_trilayer.vituri2024",
    )
    result: list[tuple[str, str]] = []
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        source_file = inspect.getsourcefile(module)
        if source_file is None:
            raise ValueError(f"implementation module {module_name} has no source file")
        raw = Path(source_file).read_bytes()
        if not raw:
            raise ValueError(f"implementation module {module_name} source is empty")
        result.append((module_name, sha256(raw).hexdigest()))
    return tuple(result)


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
            "code_sha256": sha256(marshal.dumps(value.__code__)).hexdigest(),
        }
    if isinstance(value, type):
        return {"type_module": value.__module__, "type_qualname": value.__qualname__}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_live_value(getattr(value, item.name))
            for item in fields(value)
        }
    state = getattr(value, "__dict__", None)
    if state in (None, {}):
        return {
            "singleton_type_module": type(value).__module__,
            "singleton_type_qualname": type(value).__qualname__,
        }
    raise TypeError(
        "unsupported nondeterministic live fingerprint value "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _function_live_fingerprint(function: object) -> str:
    closure = getattr(function, "__closure__", None)
    closure_values = () if closure is None else tuple(
        _canonical_live_value(cell.cell_contents) for cell in closure
    )
    return _fingerprint(
        {
            "code": sha256(
                marshal.dumps(function.__code__)  # type: ignore[attr-defined]
            ).hexdigest(),
            "defaults": _canonical_live_value(
                getattr(function, "__defaults__", None)
            ),
            "kwdefaults": _canonical_live_value(
                getattr(function, "__kwdefaults__", None)
            ),
            "closure": closure_values,
        }
    )


def _module_live_code_inventory() -> tuple[tuple[str, str], ...]:
    records: list[tuple[str, str]] = []
    for module_name, _source_sha256 in _implementation_source_fingerprints():
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        for object_name, value in sorted(vars(module).items()):
            if inspect.isfunction(value) and value.__module__ == module_name:
                records.append(
                    (
                        f"{module_name}:{object_name}",
                        _function_live_fingerprint(value),
                    )
                )
            elif inspect.isclass(value) and value.__module__ == module_name:
                for member_name, member in sorted(vars(value).items()):
                    functions: tuple[object, ...]
                    if inspect.isfunction(member):
                        functions = (member,)
                    elif isinstance(member, (staticmethod, classmethod)):
                        functions = (member.__func__,)
                    elif isinstance(member, property):
                        functions = tuple(
                            item
                            for item in (member.fget, member.fset, member.fdel)
                            if item is not None
                        )
                    else:
                        functions = ()
                    for function in functions:
                        records.append(
                            (
                                f"{module_name}:{object_name}.{member_name}",
                                _function_live_fingerprint(function),
                            )
                        )
    return tuple(records)


def _module_semantic_constant_inventory() -> tuple[tuple[str, object], ...]:
    records: list[tuple[str, object]] = []
    supported = (str, int, float, complex, bool, tuple, list, dict, np.ndarray, np.generic)
    for module_name, _source_sha256 in _implementation_source_fingerprints():
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        for name, value in sorted(vars(module).items()):
            if not name.isupper():
                continue
            if isinstance(value, supported) or (
                is_dataclass(value) and not isinstance(value, type)
            ):
                records.append((f"{module_name}:{name}", _stable(value)))
    return tuple(records)


def _runtime_binding_inventory() -> tuple[tuple[str, str, str], ...]:
    bindings = (
        ("vituri2024_vtf", vituri2024_vtf),
        (
            "vituri2024_antisymmetrized_projected_vertex",
            vituri2024_antisymmetrized_projected_vertex,
        ),
        ("Vituri2024InteractionChoiceReceipt", Vituri2024InteractionChoiceReceipt),
        ("Vituri2024InteractionBinding", Vituri2024InteractionBinding),
    )
    return tuple(
        (
            name,
            getattr(value, "__module__", ""),
            getattr(value, "__qualname__", ""),
        )
        for name, value in bindings
    )


def _kernel_implementation_fingerprint(kernel_type: type) -> str:
    del kernel_type
    return _fingerprint(
        {
            "module_sources": _implementation_source_fingerprints(),
            "live_code_inventory": _module_live_code_inventory(),
            "semantic_constant_inventory": _module_semantic_constant_inventory(),
            "runtime_binding_inventory": _runtime_binding_inventory(),
            "flavor_order": INTERNAL_FLAVOR_ORDER,
            "valley_order": ACTIVE_BAND_STATES_VALLEY_ORDER,
            "structure_tolerance": _locked_structure_tolerance(),
            "supplied_array_tolerance": _locked_supplied_array_tolerance(),
            "vertex_authority": VERTEX_AUTHORITY,
        }
    )


def _resolve_interaction(
    interaction: InteractionInput,
) -> tuple[Vituri2024InteractionChoiceReceipt, str]:
    if type(interaction) is Vituri2024InteractionChoiceReceipt:
        receipt = interaction
        fingerprint = receipt.fingerprint
    elif type(interaction) is Vituri2024InteractionBinding:
        receipt = interaction.receipt
        if interaction.receipt_fingerprint != receipt.fingerprint:
            raise ValueError("interaction binding fingerprint mismatch")
        fingerprint = interaction.receipt_fingerprint
    else:
        raise TypeError("interaction must be an exact Vituri receipt or binding")
    clean = Vituri2024InteractionChoiceReceipt(
        gate_distance_angstrom=receipt.gate_distance_angstrom,
        coulomb_e2_ev_angstrom=receipt.coulomb_e2_ev_angstrom,
        q0_evaluation=receipt.q0_evaluation,
        provider_sha256=receipt.provider_sha256,
        source_sha256=receipt.source_sha256,
        authority_kind=receipt.authority_kind,
        source_text=receipt.source_text,
    )
    if clean != receipt or clean.source_sha256 != SM_TEX_SHA256:
        raise ValueError("interaction receipt is stale or not the Vituri source")
    if clean.q0_evaluation != "analytic_kernel_limit_only":
        raise ValueError(
            "full-projector functional requires an explicit finite q=0 kernel value; "
            "the interaction receipt currently rejects q=0"
        )
    return clean, fingerprint


def _exact_local_mask(mesh: Array) -> Array:
    """Build the literal quartet predicate in vertex arithmetic order."""

    nk = mesh.shape[0]
    result = np.zeros((nk, nk, nk, nk), dtype=np.bool_)
    for alpha in range(nk):
        for beta in range(nk):
            for gamma in range(nk):
                for delta in range(nk):
                    residual = (
                        mesh[alpha, 0]
                        + mesh[beta, 0]
                        - mesh[gamma, 0]
                        - mesh[delta, 0],
                        mesh[alpha, 1]
                        + mesh[beta, 1]
                        - mesh[gamma, 1]
                        - mesh[delta, 1],
                    )
                    result[alpha, beta, gamma, delta] = residual == (0.0, 0.0)
    # The actual antisymmetrized vertex requires every selected quartet to stay
    # exact-conserving under both bra/ket swaps and pair Hermitian reversal.
    if not np.array_equal(result, result.swapaxes(0, 1)):
        raise ValueError("exact local mesh mask is not bra-swap closed")
    if not np.array_equal(result, result.swapaxes(2, 3)):
        raise ValueError("exact local mesh mask is not ket-swap closed")
    if not np.array_equal(result, result.transpose(3, 2, 1, 0)):
        raise ValueError("exact local mesh mask is not pair-Hermitian closed")
    return _bytes_backed(result, dtype=np.dtype(np.bool_))


def vituri2024_full_projected_interaction_action(
    density: Array,
    *,
    form_factors_by_flavor: Array,
    interaction_kernel_by_mesh_pair: Array,
    exact_local_mask: Array,
    area_angstrom_squared: float,
) -> Array:
    """Apply the factorized direct/exchange vertex to a conventional density."""

    if not isinstance(form_factors_by_flavor, np.ndarray) or (
        form_factors_by_flavor.dtype != np.dtype(np.complex128)
        or form_factors_by_flavor.ndim != 3
        or form_factors_by_flavor.shape[0] != len(INTERNAL_FLAVOR_ORDER)
        or form_factors_by_flavor.shape[1] != form_factors_by_flavor.shape[2]
        or form_factors_by_flavor.shape[1] < 1
        or not np.all(np.isfinite(form_factors_by_flavor))
    ):
        raise ValueError(
            "factorized form factors must have finite complex128 shape (4,Nk,Nk)"
        )
    nk = int(form_factors_by_flavor.shape[1])
    dimension = len(INTERNAL_FLAVOR_ORDER) * nk
    clean = _readonly_hermitian(
        density,
        label="full conventional density action input",
        dimension=dimension,
    )
    if not isinstance(interaction_kernel_by_mesh_pair, np.ndarray) or (
        interaction_kernel_by_mesh_pair.dtype != np.dtype(np.float64)
        or interaction_kernel_by_mesh_pair.shape != (nk, nk)
        or not np.all(np.isfinite(interaction_kernel_by_mesh_pair))
        or np.any(interaction_kernel_by_mesh_pair <= 0.0)
    ):
        raise ValueError(
            "factorized interaction kernel must be finite positive float64 (Nk,Nk)"
        )
    if not isinstance(exact_local_mask, np.ndarray) or (
        exact_local_mask.dtype != np.dtype(np.bool_)
        or exact_local_mask.shape != (nk, nk, nk, nk)
    ):
        raise ValueError(
            "factorized exact-local mask must have bool shape (Nk,Nk,Nk,Nk)"
        )
    if isinstance(area_angstrom_squared, (bool, np.bool_)):
        raise TypeError("factorized finite area must be a strict real scalar")
    area = float(area_angstrom_squared)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("factorized finite area must be positive")
    if _max_abs(
        form_factors_by_flavor
        - form_factors_by_flavor.swapaxes(1, 2).conj()
    ) > 5.0e-12:
        raise ValueError("factorized form factors violate pair conjugation")
    if _max_abs(
        interaction_kernel_by_mesh_pair - interaction_kernel_by_mesh_pair.T
    ) > 64.0 * np.finfo(np.float64).eps * max(
        1.0, _max_abs(interaction_kernel_by_mesh_pair)
    ):
        raise ValueError("factorized interaction kernel is not symmetric")
    if (
        not np.array_equal(exact_local_mask, exact_local_mask.swapaxes(0, 1))
        or not np.array_equal(exact_local_mask, exact_local_mask.swapaxes(2, 3))
        or not np.array_equal(
            exact_local_mask, exact_local_mask.transpose(3, 2, 1, 0)
        )
    ):
        raise ValueError("factorized exact-local mask symmetry closure failed")

    flavors = len(INTERNAL_FLAVOR_ORDER)
    x = clean.reshape(flavors, nk, flavors, nk)
    charge = np.zeros((nk, nk), dtype=np.complex128)
    for flavor in range(flavors):
        charge += (
            form_factors_by_flavor[flavor]
            * x[flavor, :, flavor, :].T
        )
    potential = np.einsum(
        "mprn,pr,pr->mn",
        exact_local_mask,
        interaction_kernel_by_mesh_pair,
        charge,
        optimize=True,
    )
    result = np.zeros_like(x)
    for flavor in range(flavors):
        result[flavor, :, flavor, :] = (
            form_factors_by_flavor[flavor] * potential
        )
    for left_flavor in range(flavors):
        left_form_factor = form_factors_by_flavor[left_flavor]
        for right_flavor in range(flavors):
            right_form_factor = form_factors_by_flavor[right_flavor]
            density_block = x[left_flavor, :, right_flavor, :]
            result[left_flavor, :, right_flavor, :] -= np.einsum(
                "mprn,pn,mr,pn,rp->mn",
                exact_local_mask,
                interaction_kernel_by_mesh_pair,
                left_form_factor,
                right_form_factor,
                density_block,
                optimize=True,
            )
    result = result.reshape(dimension, dimension)
    result /= area
    residual = _max_abs(result - result.conj().T)
    scale = max(1.0, _max_abs(result))
    if residual > _locked_structure_tolerance() * scale:
        raise ValueError(
            "factorized Vituri interaction action is not Hermitian; "
            "mask/form-factor/source-gauge repair is prohibited"
        )
    return _readonly_result(result)


@dataclass(frozen=True, slots=True)
class Vituri2024FullProjectedFunctionalKernel:
    """Factorized ``Sigma`` and conventional full-projector ``E/F/dF``.

    ``normal_order_reference`` is an explicit physical input, not the source
    projector and not a target-fitted counterterm.  This class validates its
    bytes but cannot establish their external authority.
    """

    ordered_mesh: Array
    active_band_states: Array
    h0_full: Array
    normal_order_reference: Array
    area_angstrom_squared: float
    interaction: InteractionInput
    normal_order_reference_fingerprint: str
    q0_policy_fingerprint: str
    source_artifact_sha256: str
    provenance: str
    interaction_receipt: Vituri2024InteractionChoiceReceipt = field(init=False)
    interaction_fingerprint: str = field(init=False)
    form_factors_by_flavor: Array = field(init=False, repr=False)
    kernel_by_mesh_pair: Array = field(init=False, repr=False)
    exact_local_mask: Array = field(init=False, repr=False)
    nk: int = field(init=False)
    dimension: int = field(init=False)
    component_fingerprints: tuple[tuple[str, str], ...] = field(init=False)
    implementation_source_fingerprints: tuple[tuple[str, str], ...] = field(
        init=False
    )
    implementation_fingerprint: str = field(init=False)
    kernel_fingerprint: str = field(init=False)
    api_version: str = field(
        default=VITURI2024_FULL_FUNCTIONAL_API_VERSION, init=False
    )
    authority: str = field(default=VITURI2024_FULL_FUNCTIONAL_AUTHORITY, init=False)
    vertex_authority: str = field(default=VERTEX_AUTHORITY, init=False)
    orbital_order: str = field(default="flavor_major_then_k", init=False)
    convention: str = field(
        default=VITURI2024_FULL_FUNCTIONAL_CONVENTION, init=False
    )
    exact_local_mask_policy: str = field(
        default=VITURI2024_FULL_FUNCTIONAL_EXACT_LOCAL_MASK, init=False
    )
    q0_status: str = field(
        default=VITURI2024_FULL_FUNCTIONAL_Q0_STATUS, init=False
    )
    gauge_status: str = field(
        default=VITURI2024_FULL_FUNCTIONAL_GAUGE_STATUS, init=False
    )
    source_closure_established: bool = field(default=False, init=False)
    source_stationarity_established: bool = field(default=False, init=False)
    q0_background_authority_established: bool = field(default=False, init=False)
    normal_order_authority_established: bool = field(default=False, init=False)
    full_projector_functional_consistency: bool = field(default=False, init=False)
    tdhf_hessian_match: bool = field(default=False, init=False)
    scalar_hessian_authority_promoted: bool = field(default=False, init=False)
    immutable_source_authority: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _locked_structure_tolerance()
        _locked_supplied_array_tolerance()
        mesh = _readonly_exact_array(
            self.ordered_mesh,
            label="ordered mesh",
            dtype=np.dtype(np.float64),
        )
        if mesh.ndim != 2 or mesh.shape[1] != 2 or mesh.shape[0] < 1:
            raise ValueError("ordered mesh must have exact shape (Nk,2), Nk>0")
        nk = int(mesh.shape[0])
        states = _readonly_exact_array(
            self.active_band_states,
            label="source-gauge active-band states",
            dtype=np.dtype(np.complex128),
            shape=(len(ACTIVE_BAND_STATES_VALLEY_ORDER), 6, nk),
        )
        norms = np.sum(np.abs(states) ** 2, axis=1)
        if _max_abs(norms - 1.0) > 5.0e-12:
            raise ValueError("every source-gauge active-band state must be normalized")
        dimension = len(INTERNAL_FLAVOR_ORDER) * nk
        h0 = _readonly_hermitian(
            self.h0_full, label="full one-body h0", dimension=dimension
        )
        reference = _readonly_reference(
            self.normal_order_reference,
            label="full normal-order reference R",
            dimension=dimension,
        )
        if isinstance(self.area_angstrom_squared, (bool, np.bool_)):
            raise TypeError("finite area must be a strict real scalar")
        area = float(self.area_angstrom_squared)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("finite area must be positive")
        interaction, interaction_fingerprint = _resolve_interaction(self.interaction)
        _sha256(
            self.normal_order_reference_fingerprint,
            "normal-order reference fingerprint",
        )
        _sha256(self.q0_policy_fingerprint, "q0 policy fingerprint")
        _sha256(self.source_artifact_sha256, "source artifact fingerprint")
        _text(self.provenance, "full functional provenance")

        valley_index = {
            valley: index
            for index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER)
        }
        form_factors_by_valley = np.einsum(
            "vcm,vcn->vmn", states.conj(), states, optimize=False
        )
        form_factors = np.stack(
            [
                form_factors_by_valley[valley_index[valley]]
                for valley, _spin in INTERNAL_FLAVOR_ORDER
            ],
            axis=0,
        ).astype(np.complex128, copy=False)
        form_factor_pair_residual = _max_abs(
            form_factors - form_factors.swapaxes(1, 2).conj()
        )
        if form_factor_pair_residual > 5.0e-12:
            raise ValueError("source-gauge form factors violate pair conjugation")
        form_factors = _bytes_backed(
            form_factors, dtype=np.dtype(np.complex128)
        )

        interaction_kernel = np.empty((nk, nk), dtype=np.float64)
        for left in range(nk):
            for right in range(nk):
                transfer_x = float(mesh[left, 0] - mesh[right, 0])
                transfer_y = float(mesh[left, 1] - mesh[right, 1])
                interaction_kernel[left, right] = vituri2024_vtf(
                    math.hypot(transfer_x, transfer_y), interaction
                )
        kernel_residual = _max_abs(interaction_kernel - interaction_kernel.T)
        if kernel_residual > 64.0 * np.finfo(np.float64).eps * max(
            1.0, _max_abs(interaction_kernel)
        ):
            raise ValueError("mesh-pair interaction kernel is not symmetric")
        if np.any(interaction_kernel <= 0.0):
            raise ValueError("Vituri VTF mesh-pair kernel must be strictly positive")
        interaction_kernel = _bytes_backed(
            interaction_kernel, dtype=np.dtype(np.float64)
        )
        local_mask = _exact_local_mask(mesh)

        object.__setattr__(self, "ordered_mesh", mesh)
        object.__setattr__(self, "active_band_states", states)
        object.__setattr__(self, "h0_full", h0)
        object.__setattr__(self, "normal_order_reference", reference)
        object.__setattr__(self, "area_angstrom_squared", area)
        object.__setattr__(self, "interaction_receipt", interaction)
        object.__setattr__(self, "interaction_fingerprint", interaction_fingerprint)
        object.__setattr__(self, "form_factors_by_flavor", form_factors)
        object.__setattr__(self, "kernel_by_mesh_pair", interaction_kernel)
        object.__setattr__(self, "exact_local_mask", local_mask)
        object.__setattr__(self, "nk", nk)
        object.__setattr__(self, "dimension", dimension)
        component_fingerprints = tuple(
            (name, _array_sha256(value))
            for name, value in (
                ("ordered_mesh", mesh),
                ("active_band_states", states),
                ("h0_full", h0),
                ("normal_order_reference", reference),
                ("form_factors_by_flavor", form_factors),
                ("kernel_by_mesh_pair", interaction_kernel),
                ("exact_local_mask", local_mask),
            )
        )
        implementation_sources = _implementation_source_fingerprints()
        implementation_fingerprint = _kernel_implementation_fingerprint(type(self))
        object.__setattr__(self, "component_fingerprints", component_fingerprints)
        object.__setattr__(
            self, "implementation_source_fingerprints", implementation_sources
        )
        object.__setattr__(
            self, "implementation_fingerprint", implementation_fingerprint
        )
        locked = (
            self.api_version == VITURI2024_FULL_FUNCTIONAL_API_VERSION,
            self.authority == VITURI2024_FULL_FUNCTIONAL_AUTHORITY,
            self.vertex_authority == VERTEX_AUTHORITY,
            self.orbital_order == "flavor_major_then_k",
            self.convention == VITURI2024_FULL_FUNCTIONAL_CONVENTION,
            self.exact_local_mask_policy == VITURI2024_FULL_FUNCTIONAL_EXACT_LOCAL_MASK,
            self.q0_status == VITURI2024_FULL_FUNCTIONAL_Q0_STATUS,
            self.gauge_status == VITURI2024_FULL_FUNCTIONAL_GAUGE_STATUS,
            self.source_closure_established is False,
            self.source_stationarity_established is False,
            self.q0_background_authority_established is False,
            self.normal_order_authority_established is False,
            self.full_projector_functional_consistency is False,
            self.tdhf_hessian_match is False,
            self.scalar_hessian_authority_promoted is False,
            self.immutable_source_authority is False,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
        )
        if not all(locked):
            raise ValueError("full functional scope or authority was inflated")
        object.__setattr__(
            self, "kernel_fingerprint", self._expected_kernel_fingerprint()
        )

    def _authority_fields_locked(self) -> bool:
        return all(
            (
                self.api_version == VITURI2024_FULL_FUNCTIONAL_API_VERSION,
                self.authority == VITURI2024_FULL_FUNCTIONAL_AUTHORITY,
                self.vertex_authority == VERTEX_AUTHORITY,
                self.orbital_order == "flavor_major_then_k",
                self.convention == VITURI2024_FULL_FUNCTIONAL_CONVENTION,
                self.exact_local_mask_policy
                == VITURI2024_FULL_FUNCTIONAL_EXACT_LOCAL_MASK,
                self.q0_status == VITURI2024_FULL_FUNCTIONAL_Q0_STATUS,
                self.gauge_status == VITURI2024_FULL_FUNCTIONAL_GAUGE_STATUS,
                self.source_closure_established is False,
                self.source_stationarity_established is False,
                self.q0_background_authority_established is False,
                self.normal_order_authority_established is False,
                self.full_projector_functional_consistency is False,
                self.tdhf_hessian_match is False,
                self.scalar_hessian_authority_promoted is False,
                self.immutable_source_authority is False,
                self.production_ready is False,
                self.paper_reproduction_verified is False,
            )
        )

    def _expected_kernel_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": self.api_version,
                "mesh": self.ordered_mesh,
                "active_band_states": self.active_band_states,
                "h0_full": self.h0_full,
                "normal_order_reference": self.normal_order_reference,
                "area_angstrom_squared": self.area_angstrom_squared,
                "interaction_fingerprint": self.interaction_fingerprint,
                "normal_order_reference_fingerprint": (
                    self.normal_order_reference_fingerprint
                ),
                "q0_policy_fingerprint": self.q0_policy_fingerprint,
                "source_artifact_sha256": self.source_artifact_sha256,
                "form_factors_by_flavor": self.form_factors_by_flavor,
                "kernel_by_mesh_pair": self.kernel_by_mesh_pair,
                "exact_local_mask": self.exact_local_mask,
                "component_fingerprints": self.component_fingerprints,
                "implementation_source_fingerprints": (
                    self.implementation_source_fingerprints
                ),
                "implementation_fingerprint": self.implementation_fingerprint,
                "energy_equation": VITURI2024_FULL_FUNCTIONAL_ENERGY_EQUATION,
                "fock_equation": VITURI2024_FULL_FUNCTIONAL_FOCK_EQUATION,
                "df_equation": VITURI2024_FULL_FUNCTIONAL_DF_EQUATION,
                "vertex_authority": self.vertex_authority,
                "orbital_order": self.orbital_order,
                "structure_tolerance": _locked_structure_tolerance(),
                "supplied_array_tolerance": _locked_supplied_array_tolerance(),
                "provenance": self.provenance,
                "authority": self.authority,
            }
        )

    def validate_live_state(self) -> None:
        arrays = {
            "ordered_mesh": self.ordered_mesh,
            "active_band_states": self.active_band_states,
            "h0_full": self.h0_full,
            "normal_order_reference": self.normal_order_reference,
            "form_factors_by_flavor": self.form_factors_by_flavor,
            "kernel_by_mesh_pair": self.kernel_by_mesh_pair,
            "exact_local_mask": self.exact_local_mask,
        }
        if tuple((name, _array_sha256(arrays[name])) for name in arrays) != (
            self.component_fingerprints
        ):
            raise ValueError("full functional component bytes drifted")
        if any(value.flags.writeable or value.flags.owndata for value in arrays.values()):
            raise ValueError("full functional arrays must remain immutable bytes-backed views")
        if not self._authority_fields_locked():
            raise ValueError("full functional scalar/authority fields drifted")
        if (
            type(self.nk) is not int
            or self.nk != self.ordered_mesh.shape[0]
            or type(self.dimension) is not int
            or self.dimension != len(INTERNAL_FLAVOR_ORDER) * self.nk
        ):
            raise ValueError("full functional dimension/layout drifted")
        if not math.isfinite(self.area_angstrom_squared) or self.area_angstrom_squared <= 0.0:
            raise ValueError("full functional area drifted")
        _sha256(
            self.normal_order_reference_fingerprint,
            "live normal-order reference fingerprint",
        )
        _sha256(self.q0_policy_fingerprint, "live q0 policy fingerprint")
        _sha256(self.source_artifact_sha256, "live source artifact fingerprint")
        _text(self.provenance, "live full functional provenance")
        clean_interaction, clean_interaction_fingerprint = _resolve_interaction(
            self.interaction
        )
        if (
            clean_interaction != self.interaction_receipt
            or clean_interaction_fingerprint != self.interaction_fingerprint
            or self.interaction_receipt.fingerprint != self.interaction_fingerprint
        ):
            raise ValueError("full functional interaction receipt drifted")
        current_sources = _implementation_source_fingerprints()
        if current_sources != self.implementation_source_fingerprints:
            raise ValueError("full functional implementation source bytes drifted")
        if _kernel_implementation_fingerprint(type(self)) != self.implementation_fingerprint:
            raise ValueError("full functional implementation code drifted")
        if self._expected_kernel_fingerprint() != self.kernel_fingerprint:
            raise ValueError("full functional kernel fingerprint drifted")

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.kernel_fingerprint

    def _validate_density(self, value: object, label: str) -> Array:
        return _readonly_hermitian(value, label=label, dimension=self.dimension)

    def interaction_action(self, density: Array) -> Array:
        """Return ``Sigma[density]`` without post-Hermitization."""

        self.validate_live_state()
        return vituri2024_full_projected_interaction_action(
            density,
            form_factors_by_flavor=self.form_factors_by_flavor,
            interaction_kernel_by_mesh_pair=self.kernel_by_mesh_pair,
            exact_local_mask=self.exact_local_mask,
            area_angstrom_squared=self.area_angstrom_squared,
        )

    def energy(self, projector: Array) -> float:
        clean = self._validate_density(projector, "full conventional P")
        difference = clean - self.normal_order_reference
        interaction = self.interaction_action(difference)
        one_body = complex(np.einsum("ij,ji->", self.h0_full, clean, optimize=False))
        interaction_energy = 0.5 * complex(
            np.einsum("ij,ji->", difference, interaction, optimize=False)
        )
        total = one_body + interaction_energy
        imaginary_bound = _locked_structure_tolerance() * max(
            1.0, abs(total), abs(one_body), abs(interaction_energy)
        )
        if abs(total.imag) > imaginary_bound:
            raise ValueError("full projected scalar energy has a material imaginary part")
        return float(total.real)

    def fock(self, projector: Array) -> Array:
        clean = self._validate_density(projector, "full conventional P")
        result = self.h0_full + self.interaction_action(
            clean - self.normal_order_reference
        )
        return _readonly_result(result)

    def fock_derivative(self, projector: Array, direction: Array) -> Array:
        self._validate_density(projector, "full conventional P anchor")
        clean_direction = self._validate_density(
            direction, "full conventional Hermitian direction D"
        )
        return self.interaction_action(clean_direction)


@dataclass(frozen=True, slots=True)
class Vituri2024FullProjectedSuppliedArrayConsistencyReceipt:
    _factory_token: InitVar[object]
    kernel_fingerprint: str
    source_projector_sha256: str
    source_projector_rank: int
    supplied_interaction_h_sha256: str
    supplied_fock_sha256: str
    computed_interaction_h_sha256: str
    computed_fock_sha256: str
    maximum_interaction_h_residual_ev: float
    maximum_fock_residual_ev: float
    maximum_supplied_decomposition_residual_ev: float
    tolerance_ev: float
    passed: bool
    authority: str = field(
        default=VITURI2024_FULL_FUNCTIONAL_SUPPLIED_ARRAY_AUTHORITY, init=False
    )
    source_stationarity_established: bool = field(default=False, init=False)
    q0_background_authority_established: bool = field(default=False, init=False)
    normal_order_authority_established: bool = field(default=False, init=False)
    full_projector_functional_consistency: bool = field(default=False, init=False)
    tdhf_hessian_match: bool = field(default=False, init=False)
    scalar_hessian_authority_promoted: bool = field(default=False, init=False)
    immutable_source_authority: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _SUPPLIED_ARRAY_TOKEN:
            raise TypeError("supplied-array receipt requires the private factory token")
        for name in (
            "kernel_fingerprint",
            "source_projector_sha256",
            "supplied_interaction_h_sha256",
            "supplied_fock_sha256",
            "computed_interaction_h_sha256",
            "computed_fock_sha256",
        ):
            _sha256(getattr(self, name), f"supplied-array {name}")
        if type(self.source_projector_rank) is not int or self.source_projector_rank < 0:
            raise ValueError("supplied-array source projector rank must be nonnegative")
        for name in (
            "maximum_interaction_h_residual_ev",
            "maximum_fock_residual_ev",
            "maximum_supplied_decomposition_residual_ev",
            "tolerance_ev",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"supplied-array {name} must be finite and nonnegative")
        if self.tolerance_ev != _locked_supplied_array_tolerance():
            raise ValueError("supplied-array tolerance changed")
        if self.passed is not True:
            raise ValueError("failed supplied-array comparison cannot create a receipt")
        locked = (
            self.authority == VITURI2024_FULL_FUNCTIONAL_SUPPLIED_ARRAY_AUTHORITY,
            self.source_stationarity_established is False,
            self.q0_background_authority_established is False,
            self.normal_order_authority_established is False,
            self.full_projector_functional_consistency is False,
            self.tdhf_hessian_match is False,
            self.scalar_hessian_authority_promoted is False,
            self.immutable_source_authority is False,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
        )
        if not all(locked):
            raise ValueError("supplied-array consistency authority was inflated")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


def validate_vituri2024_full_projected_supplied_arrays(
    *,
    kernel: Vituri2024FullProjectedFunctionalKernel,
    source_projector: Array,
    supplied_interaction_h: Array,
    supplied_fock: Array,
) -> Vituri2024FullProjectedSuppliedArrayConsistencyReceipt:
    """Check caller-supplied arrays without claiming source provenance.

    This check does not prove that ``h0_full`` was not fitted before kernel
    construction.  Real source closure requires a later immutable replay and
    normal-reference authority binding.
    """

    if type(kernel) is not Vituri2024FullProjectedFunctionalKernel:
        raise TypeError("supplied-array check requires an exact full functional kernel")
    kernel.validate_live_state()
    p0 = kernel._validate_density(source_projector, "supplied conventional P0")
    idempotency = _max_abs(p0 @ p0 - p0)
    trace = complex(np.trace(p0))
    rank = int(round(trace.real))
    if idempotency > _locked_supplied_array_tolerance():
        raise ValueError("supplied source P0 is not idempotent")
    if (
        rank < 0
        or rank > kernel.dimension
        or abs(trace.imag) > 1.0e-12
        or abs(trace.real - rank) > 1.0e-10
    ):
        raise ValueError("supplied source P0 does not have a valid integer real trace")
    interaction_h = kernel._validate_density(
        supplied_interaction_h, "supplied full interaction_h"
    )
    fock = kernel._validate_density(supplied_fock, "supplied full Fock")
    computed_interaction = kernel.interaction_action(
        p0 - kernel.normal_order_reference
    )
    computed_fock = kernel.fock(p0)
    interaction_residual = _max_abs(computed_interaction - interaction_h)
    fock_residual = _max_abs(computed_fock - fock)
    decomposition_residual = _max_abs(fock - kernel.h0_full - interaction_h)
    tolerance = _locked_supplied_array_tolerance()
    maximum = max(interaction_residual, fock_residual, decomposition_residual)
    if maximum > tolerance:
        raise ValueError(
            "Vituri full projected supplied-array consistency failed: "
            f"interaction={interaction_residual:.6e} eV, "
            f"fock={fock_residual:.6e} eV, "
            f"decomposition={decomposition_residual:.6e} eV, "
            f"tolerance={tolerance:.6e} eV"
        )
    return Vituri2024FullProjectedSuppliedArrayConsistencyReceipt(
        _factory_token=_SUPPLIED_ARRAY_TOKEN,
        kernel_fingerprint=kernel.fingerprint,
        source_projector_sha256=_array_sha256(p0),
        source_projector_rank=rank,
        supplied_interaction_h_sha256=_array_sha256(interaction_h),
        supplied_fock_sha256=_array_sha256(fock),
        computed_interaction_h_sha256=_array_sha256(computed_interaction),
        computed_fock_sha256=_array_sha256(computed_fock),
        maximum_interaction_h_residual_ev=interaction_residual,
        maximum_fock_residual_ev=fock_residual,
        maximum_supplied_decomposition_residual_ev=decomposition_residual,
        tolerance_ev=tolerance,
        passed=True,
    )


__all__ = [
    "VITURI2024_FULL_FUNCTIONAL_API_VERSION",
    "VITURI2024_FULL_FUNCTIONAL_AUTHORITY",
    "VITURI2024_FULL_FUNCTIONAL_CONVENTION",
    "VITURI2024_FULL_FUNCTIONAL_DF_EQUATION",
    "VITURI2024_FULL_FUNCTIONAL_ENERGY_EQUATION",
    "VITURI2024_FULL_FUNCTIONAL_EXACT_LOCAL_MASK",
    "VITURI2024_FULL_FUNCTIONAL_FOCK_EQUATION",
    "VITURI2024_FULL_FUNCTIONAL_GAUGE_STATUS",
    "VITURI2024_FULL_FUNCTIONAL_Q0_STATUS",
    "VITURI2024_FULL_FUNCTIONAL_SUPPLIED_ARRAY_AUTHORITY",
    "VITURI2024_FULL_FUNCTIONAL_SUPPLIED_ARRAY_TOLERANCE",
    "Vituri2024FullProjectedFunctionalKernel",
    "Vituri2024FullProjectedSuppliedArrayConsistencyReceipt",
    "validate_vituri2024_full_projected_supplied_arrays",
    "vituri2024_full_projected_interaction_action",
]
