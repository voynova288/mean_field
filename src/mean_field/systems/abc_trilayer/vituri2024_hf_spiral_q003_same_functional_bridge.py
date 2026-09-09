"""Candidate-only q003 bridge from the production action to one scalar functional.

The bridge implements the operator-level checks requested by
``reports/vituri2024_q003_same_functional_scalar_hessian_authority_gap_20260909.md``.
It binds both pinned q003 sources to the live selected-spin inventories
and checks the one-body identity explicitly:

``D_prod = diag(F_prod)_p - diag(F_prod)_h``
``D_F    = diag(F_exact)_p - diag(F_exact)_h``.

The independently reconstructed exact-integer Fock must be in the registered
particle/hole basis within a fixed recorded tolerance.  The interaction
identity is structural, not a numerical action comparison: the fingerprint
covers the production direct rank-one and exchange ``-M^dagger C_K M`` path,
the paired injection/extraction and core matvec, and the independent
exact-integer ``Sigma``/scalar functions.

A successful result is only a candidate structural receipt.  This module has
no matrix action, LinearOperator, eigensolver, stability, production, or paper
promotion surface.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
from typing import Final

import numpy as np

from . import vituri2024_hf_spiral_full_hessian as _hessian_module
from . import vituri2024_hf_spiral_full_response as _response_module
from . import vituri2024_hf_spiral_full_stability as _inventory_module
from . import vituri2024_tdhf_exact_integer_signed_scalar as _scalar_module
from . import vituri2024_hf_spiral_q003_source_bound_reciprocity as _reciprocity_module
from ...core.hf import zero_temperature_sector_stability as _core_module
from .vituri2024_hf_spiral_full_hessian import (
    Vituri2024HFSpiralFullHessianContext,
)
from .vituri2024_hf_spiral_q003_source_bound_reciprocity import (
    Vituri2024Q003SourceBoundReciprocityCertificate,
    Vituri2024Q003SourceBoundReciprocityGroup,
)

Array = np.ndarray

VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_API_VERSION: Final[str] = (
    "vituri2024_q003_selected_spin_same_functional_structural_bridge.v1"
)
VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_SCOPE: Final[str] = (
    "both_pinned_q003_sources_full_selected_spin_fixed_rank_inventory_candidate"
)
VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_AUTHORITY: Final[str] = (
    "candidate_structural_receipt_only_without_scalar_hessian_eigensolver_"
    "stability_scientific_production_or_paper_authority"
)
VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_IDENTITY: Final[str] = (
    "candidate_derivation_of_L_prod_equals_Sigma_from_matching_direct_rank_one_"
    "and_exchange_minus_Mdagger_CK_M_factors_support_flavor_blocks_area_and_"
    "no_wrap_orientation_pending_detached_review"
)
VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_DERIVATION: Final[
    tuple[str, ...]
] = (
    "common_inputs=selected_spinors,exact_integer_labels,fft_plan,real_even_kernel,area",
    "direct=F_a(k+d,k)K(-d)sum_cp[F_c(p,p+d)W_cc(p)]/area",
    "exchange=-sum_p[V(p-k)<u_a(k+d)|u_a(p+d)><u_b(p)|u_b(k)>W_ab(p)]/area",
    "support=centered_integer_particle_minus_hole_no_wrap",
    "flavor_blocks=chi0:(0,0),(1,1);chi+2:(1,0);chi-2:(0,1)",
    "paired_geometry=W_d=Z_d[x]+Z_-d[y]^dagger_with_both_signed_lanes",
    "comparison=structural_only_no_numerical_L_prod_Sigma_action_assumption",
)
VITURI2024_Q003_ONE_BODY_BASIS_TOLERANCE_EV: Final[float] = 1.0e-12
VITURI2024_Q003_ONE_BODY_DIAGONAL_TOLERANCE_EV: Final[float] = 1.0e-12
VITURI2024_Q003_ONE_BODY_ACTION_TOLERANCE_EV: Final[float] = 2.0e-12
VITURI2024_Q003_SOURCE_FOCK_CLOSURE_TOLERANCE_EV: Final[float] = 2.0e-10
VITURI2024_Q003_SELECTED_OCCUPIED_COUNT: Final[int] = 11_852
VITURI2024_Q003_SELECTED_VIRTUAL_COUNT: Final[int] = 1_270
VITURI2024_Q003_NONEMPTY_SECTOR_COUNT: Final[int] = 38_259
VITURI2024_Q003_CANONICAL_ORBIT_COUNT: Final[int] = 21_170
VITURI2024_Q003_COMPLEX_DIMENSION: Final[int] = 15_052_040
VITURI2024_Q003_REAL_DIMENSION: Final[int] = 30_104_080

_PINNED_LINEAGE_SHA256: Final[tuple[tuple[str, str], ...]] = (
    (
        "stationarity_477106",
        "b2793903720e9b1550855dcaa1c84d60d08a8cebe84970a058e598c5b8880934",
    ),
    (
        "source_fock_closure_487653",
        "24855c67a6997bad2c10963ef585d5d7199609e1a72b88598f2f285d3aaaa1a5",
    ),
    (
        "exact_unitary_step_ladder_496930",
        "22715b1594c424b9d95f9a3aba2511a65946491ca1a75d8bb6138b5db399eae8",
    ),
)
_PINNED_SOURCE_DENSITY_GROUP_SHA256: Final[tuple[tuple[str, str], ...]] = (
    (
        "3307",
        "3307efd3434cef054b37545d778a23f2a6fb8684f599b9e7d0e0cbea8ee36288",
    ),
    (
        "40fd",
        "40fd7da28d7d32ad8034352bbd1171b873ceac9c268f5395f908f9a9f686462d",
    ),
)
_PINNED_INDEPENDENT_SOURCE_FOCK_SHA256: Final[tuple[tuple[str, str], ...]] = (
    (
        "3307",
        "f630d2a53fa3ff83a621e560a4c16c22e1f39ea4ae8a951084fe06e67b49f13a",
    ),
    (
        "40fd",
        "ebd55e7bf03e680d554e3843188d8bd031059c273889342846d5a6be93358c62",
    ),
)

def _make_factory_boundary():
    token = object()

    def require(candidate: object, role: str) -> None:
        if candidate is not token:
            raise TypeError(f"{role} are factory-only")

    def construct(owner: type, /, **kwargs: object):
        return owner(_factory_token=token, **kwargs)

    return require, construct


_REQUIRE_FACTORY_TOKEN, _construct_with_factory_token = _make_factory_boundary()
_SHA256 = sha256
_JSON_LOADS = json.loads

# Public construction captures these bridge-local aliases and every mutable
# library attribute dereferenced by a bridge helper.  Checking only derived
# owner/binding tables is insufficient: rebinding an alias to a forwarding
# proxy (or rebinding a primitive together with its bridge cache) could
# otherwise make the post-guard implementation execute a different object.
_IMPORT_BRIDGE_LOCAL_ALIAS_BINDINGS = (
    ("_response_module", _response_module),
    ("_hessian_module", _hessian_module),
    ("_inventory_module", _inventory_module),
    ("_core_module", _core_module),
    ("_scalar_module", _scalar_module),
    ("_reciprocity_module", _reciprocity_module),
    ("np", np),
    ("json", json),
    ("inspect", inspect),
    ("Path", Path),
    ("sys", sys),
    ("sha256", sha256),
    ("Array", Array),
    ("Vituri2024HFSpiralFullHessianContext", Vituri2024HFSpiralFullHessianContext),
    (
        "Vituri2024Q003SourceBoundReciprocityCertificate",
        Vituri2024Q003SourceBoundReciprocityCertificate,
    ),
    (
        "Vituri2024Q003SourceBoundReciprocityGroup",
        Vituri2024Q003SourceBoundReciprocityGroup,
    ),
)
_IMPORT_BRIDGE_PRIMITIVE_BINDINGS = (
    (np, "__version__", np.__version__),
    (np, "ndarray", np.ndarray),
    (np, "ascontiguousarray", np.ascontiguousarray),
    (np, "dtype", np.dtype),
    (np, "complex128", np.complex128),
    (np, "uint8", np.uint8),
    (np, "frombuffer", np.frombuffer),
    (np, "all", np.all),
    (np, "isfinite", np.isfinite),
    (np, "array_equal", np.array_equal),
    (np, "count_nonzero", np.count_nonzero),
    (np, "asarray", np.asarray),
    (np, "int64", np.int64),
    (np, "bool_", np.bool_),
    (np, "float64", np.float64),
    (np, "max", np.max),
    (np, "min", np.min),
    (np, "array", np.array),
    (np, "flatnonzero", np.flatnonzero),
    (np, "ix_", np.ix_),
    (np, "linalg", np.linalg),
    (np.linalg, "norm", np.linalg.norm),
    (np, "abs", np.abs),
    (np, "arange", np.arange),
    (np, "stack", np.stack),
    (json, "loads", json.loads),
    (json, "dumps", json.dumps),
    (inspect, "getsource", inspect.getsource),
    (inspect, "getsourcefile", inspect.getsourcefile),
    (Path, "resolve", Path.resolve),
    (Path, "read_bytes", Path.read_bytes),
    (sys, "modules", sys.modules),
)

def _module_own_callables(module: object) -> tuple[tuple[str, object], ...]:
    module_name = getattr(module, "__name__", "")
    return tuple(
        (name, value)
        for name, value in sorted(vars(module).items())
        if callable(value) and getattr(value, "__module__", "") == module_name
    )


def _class_descriptors(value: type) -> tuple[tuple[str, tuple[object, ...]], ...]:
    records: list[tuple[str, tuple[object, ...]]] = []
    for name, descriptor in sorted(vars(value).items()):
        if isinstance(descriptor, property):
            records.append((name, (descriptor.fget, descriptor.fset, descriptor.fdel)))
        elif isinstance(descriptor, (staticmethod, classmethod)):
            records.append((name, (descriptor.__func__,)))
        elif callable(descriptor):
            records.append((name, (descriptor,)))
    return tuple(records)


def _module_class_bindings(
    module: object,
) -> tuple[tuple[type, tuple[tuple[str, tuple[object, ...]], ...]], ...]:
    module_name = getattr(module, "__name__", "")
    return tuple(
        (value, _class_descriptors(value))
        for _name, value in sorted(vars(module).items())
        if isinstance(value, type) and getattr(value, "__module__", "") == module_name
    )


_NUMPY_RUNTIME_OWNER_MODULES = (
    _response_module,
    _hessian_module,
    _inventory_module,
    _core_module,
    _scalar_module,
)
_RUNTIME_OWNER_MODULES = (
    *_NUMPY_RUNTIME_OWNER_MODULES,
    _reciprocity_module,
)
# The source-bound owner is part of the live closure, not merely a provider of
# two imported class aliases.  Its complete callable/class-descriptor inventory
# transitively pins the certificate validator's bridge-visible call graph,
# including _validate_live_bindings and Certificate._payload.
_IMPORT_OWNER_CALLABLE_BINDINGS = tuple(
    (module, _module_own_callables(module)) for module in _RUNTIME_OWNER_MODULES
)
_IMPORT_OWNER_CLASS_BINDINGS = tuple(
    (module, _module_class_bindings(module)) for module in _RUNTIME_OWNER_MODULES
)
# Snapshot every reciprocity-module constant consulted by the certificate/group
# live-validation path, including constants checked transitively by the old
# module's own guard.  The bridge must not rely on that guard alone: a caller
# can rebind a live constant together with the old module's mutable import-time
# comparison value.  These original objects are also captured by the public
# builder closure below.
_RECIPROCITY_LIVE_CONSTANT_NAMES: Final[tuple[str, ...]] = (
    "ARTIFACT_ROLES",
    "REVIEWED_CAPSULE_MEMBERS",
    "_PINNED_ARTIFACT_SHA256",
    "_PINNED_MEMBER_SHA256",
    "_PINNED_SOURCE_COMMIT",
    "_PINNED_HISTORICAL_IMPLEMENTATION",
    "_PINNED_GROUPS",
    "_PINNED_CERTIFICATE_GROUP_FINGERPRINTS",
    "_REVIEW_GATES",
    "_STRUCTURAL_GATES",
    "_FALSE_AUTHORITIES",
    "PINNED_Q003_RECIPROCITY_CERTIFIER_SHA256",
    "PINNED_Q003_RECIPROCITY_CERTIFIER_REVIEW_SHA256",
)
_IMPORT_RECIPROCITY_LIVE_CONSTANT_BINDINGS = tuple(
    (name, getattr(_reciprocity_module, name))
    for name in _RECIPROCITY_LIVE_CONSTANT_NAMES
)

_IMPORT_RUNTIME_BINDINGS = (
    (
        _reciprocity_module,
        "Vituri2024Q003SourceBoundReciprocityCertificate",
        Vituri2024Q003SourceBoundReciprocityCertificate,
    ),
    (
        _reciprocity_module,
        "Vituri2024Q003SourceBoundReciprocityGroup",
        Vituri2024Q003SourceBoundReciprocityGroup,
    ),
    (
        Vituri2024Q003SourceBoundReciprocityCertificate,
        "validate_live_state",
        Vituri2024Q003SourceBoundReciprocityCertificate.validate_live_state,
    ),
    (_response_module, "_FFT2", _response_module._FFT2),
    (_response_module, "_IFFT2", _response_module._IFFT2),
    (_response_module, "_IMPORT_FFT2", _response_module._IMPORT_FFT2),
    (_response_module, "_IMPORT_IFFT2", _response_module._IMPORT_IFFT2),
    (_scalar_module, "_FFT2", _scalar_module._FFT2),
    (_scalar_module, "_IFFT2", _scalar_module._IFFT2),
    (_scalar_module, "_EXPM", _scalar_module._EXPM),
    (_scalar_module, "_IMPORT_FFT2", _scalar_module._IMPORT_FFT2),
    (_scalar_module, "_IMPORT_IFFT2", _scalar_module._IMPORT_IFFT2),
    (_scalar_module, "_IMPORT_EXPM", _scalar_module._IMPORT_EXPM),
    (
        _scalar_module,
        "Vituri2024SquareCartesianFFTPlan",
        _scalar_module.Vituri2024SquareCartesianFFTPlan,
    ),
    (_scalar_module, "_IMPORT_FFT_PLAN_TYPE", _scalar_module._IMPORT_FFT_PLAN_TYPE),
    (_hessian_module, "OrbitalTransitionLane", _hessian_module.OrbitalTransitionLane),
    (
        _hessian_module,
        "PairedSectorOrbitalHessian",
        _hessian_module.PairedSectorOrbitalHessian,
    ),
    (
        _hessian_module,
        "PairedOrbitalTransitionFrame",
        _hessian_module.PairedOrbitalTransitionFrame,
    ),
    *((module, "np", np) for module in _NUMPY_RUNTIME_OWNER_MODULES),
)

# These are the formula owners whose exact live objects and source are covered.
_FORMULA_BINDING_SPECS: Final[tuple[tuple[str, object, str], ...]] = (
    ("response_direct_rank_one", _response_module, "_direct_rank_one_numerator"),
    (
        "response_exchange_mdagger_ck_m",
        _response_module,
        "_accumulate_exchange_mdagger_ck_m_numerator",
    ),
    (
        "response_support",
        _response_module,
        "_support_indices",
    ),
    (
        "response_flavor_blocks",
        _response_module,
        "_allowed_flavor_blocks",
    ),
    (
        "response_direct_wrapper",
        _response_module.Vituri2024HFSpiralSignedDisplacementResponse,
        "_direct_response",
    ),
    (
        "response_validate_signed_block",
        _response_module.Vituri2024HFSpiralSignedDisplacementResponse,
        "_validate_signed_block",
    ),
    (
        "response_fft_action",
        _response_module.Vituri2024HFSpiralSignedDisplacementResponse,
        "_apply_fft_validated",
    ),
    (
        "response_validated_action_call",
        _response_module.Vituri2024HFSpiralValidatedSignedDisplacementFFTAction,
        "__call__",
    ),
    (
        "response_prepare_action",
        _response_module.Vituri2024HFSpiralValidatedResponseActionFactory,
        "prepare_fft_action",
    ),
    (
        "hessian_paired_injection",
        _hessian_module._VituriPairedInteractionCallback,
        "__call__",
    ),
    (
        "hessian_transition_embedding",
        _hessian_module.Vituri2024HFSpiralFullHessianContext,
        "_build_embedding",
    ),
    (
        "hessian_orbit_builder",
        _hessian_module.Vituri2024HFSpiralFullHessianContext,
        "build_orbit_hessian",
    ),
    (
        "core_unpack_real",
        _core_module.PairedOrbitalTransitionFrame,
        "unpack_real",
    ),
    (
        "core_pack_complex",
        _core_module.PairedOrbitalTransitionFrame,
        "pack_complex",
    ),
    (
        "core_lane_one_body_gaps",
        _core_module.OrbitalTransitionLane,
        "__post_init__",
    ),
    (
        "core_paired_gradient_action",
        _core_module.PairedSectorOrbitalHessian,
        "complex_gradient_jacobian_action",
    ),
    (
        "core_paired_matvec",
        _core_module.PairedSectorOrbitalHessian,
        "matvec",
    ),
    (
        "inventory_iter_sectors",
        _inventory_module.Vituri2024HFSpiralFullSectorInventory,
        "iter_sector_keys",
    ),
    (
        "inventory_iter_orbits",
        _inventory_module.Vituri2024HFSpiralFullSectorInventory,
        "iter_conjugate_orbits",
    ),
    (
        "exact_integer_support",
        _scalar_module,
        "_support",
    ),
    (
        "exact_integer_flavor_blocks",
        _scalar_module,
        "_allowed_blocks",
    ),
    (
        "exact_integer_sigma_action",
        _scalar_module,
        "vituri2024_exact_integer_signed_scalar_fft_action",
    ),
    (
        "exact_integer_source_fock",
        _scalar_module,
        "vituri2024_exact_integer_source_fock_fft",
    ),
    (
        "exact_integer_interaction_trace",
        _scalar_module,
        "vituri2024_exact_integer_paired_interaction_trace",
    ),
    (
        "exact_integer_scalar_curvature",
        _scalar_module,
        "vituri2024_exact_integer_orbital_scalar_curvature",
    ),
    (
        "exact_integer_unitary_scalar",
        _scalar_module,
        "vituri2024_exact_integer_unitary_relative_energy",
    ),
)
_IMPORT_FORMULA_BINDINGS = tuple(
    (label, owner, attribute, getattr(owner, attribute))
    for label, owner, attribute in _FORMULA_BINDING_SPECS
)


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _fingerprint(payload: object) -> str:
    return _SHA256(_canonical(payload)).hexdigest()


def _array_sha256(value: Array) -> str:
    array = np.ascontiguousarray(value)
    return _SHA256(
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    ).hexdigest()


def _source_sha256(value: object) -> str:
    return _SHA256(inspect.getsource(value).encode()).hexdigest()


def _module_file_sha256(module: object) -> str:
    source_file = inspect.getsourcefile(module)
    if source_file is None:
        raise RuntimeError("same-functional formula owner has no source file")
    return _SHA256(Path(source_file).resolve().read_bytes()).hexdigest()


def _validate_formula_bindings(
    _reciprocity_constant_snapshot: tuple[tuple[str, object], ...] = (
        _IMPORT_RECIPROCITY_LIVE_CONSTANT_BINDINGS
    ),
) -> None:
    if (
        _IMPORT_RECIPROCITY_LIVE_CONSTANT_BINDINGS
        is not _reciprocity_constant_snapshot
    ):
        raise RuntimeError("same-functional reciprocity constant table drifted")
    for name, expected in _reciprocity_constant_snapshot:
        if getattr(_reciprocity_module, name, None) is not expected:
            raise RuntimeError(
                "same-functional reciprocity live constant binding drifted: "
                f"{name}"
            )
    for module, expected in _IMPORT_OWNER_CALLABLE_BINDINGS:
        current = _module_own_callables(module)
        if len(current) != len(expected) or any(
            current_name != expected_name or current_value is not expected_value
            for (current_name, current_value), (expected_name, expected_value) in zip(
                current, expected, strict=True
            )
        ):
            raise RuntimeError("same-functional owner callable inventory drifted")
    for module, expected_classes in _IMPORT_OWNER_CLASS_BINDINGS:
        current_classes = _module_class_bindings(module)
        if len(current_classes) != len(expected_classes):
            raise RuntimeError("same-functional owner class inventory drifted")
        for (current_class, current_descriptors), (
            expected_class,
            expected_descriptors,
        ) in zip(current_classes, expected_classes, strict=True):
            if current_class is not expected_class or len(current_descriptors) != len(
                expected_descriptors
            ):
                raise RuntimeError("same-functional owner class binding drifted")
            for (current_name, current_values), (
                expected_name,
                expected_values,
            ) in zip(current_descriptors, expected_descriptors, strict=True):
                if current_name != expected_name or len(current_values) != len(
                    expected_values
                ) or any(
                    current_value is not expected_value
                    for current_value, expected_value in zip(
                        current_values, expected_values, strict=True
                    )
                ):
                    raise RuntimeError(
                        "same-functional owner class descriptor binding drifted"
                    )
    if _FORMULA_BINDING_SPECS != tuple(
        (label, owner, attribute)
        for label, owner, attribute, _value in _IMPORT_FORMULA_BINDINGS
    ):
        raise RuntimeError("same-functional formula binding inventory drifted")
    for label, owner, attribute, expected in _IMPORT_FORMULA_BINDINGS:
        if getattr(owner, attribute) is not expected:
            raise RuntimeError(
                f"same-functional formula function binding drifted: {label}"
            )
    for owner, attribute, expected in _IMPORT_RUNTIME_BINDINGS:
        if getattr(owner, attribute, None) is not expected:
            raise RuntimeError(
                "same-functional imported runtime binding drifted: "
                f"{getattr(owner, '__name__', type(owner).__name__)}.{attribute}"
            )
    if (
        _hessian_module.OrbitalTransitionLane
        is not _core_module.OrbitalTransitionLane
        or _hessian_module.PairedSectorOrbitalHessian
        is not _core_module.PairedSectorOrbitalHessian
        or _hessian_module.PairedOrbitalTransitionFrame
        is not _core_module.PairedOrbitalTransitionFrame
    ):
        raise RuntimeError("same-functional imported runtime binding drifted")
    _scalar_module.vituri2024_exact_integer_signed_scalar_implementation_fingerprint()
    (
        _reciprocity_module
        .vituri2024_q003_source_bound_reciprocity_implementation_fingerprint()
    )


def _formula_source_records() -> tuple[tuple[str, str, str, str], ...]:
    _validate_formula_bindings()
    return tuple(
        (
            label,
            getattr(value, "__module__", ""),
            getattr(value, "__qualname__", ""),
            _source_sha256(value),
        )
        for label, _owner, _attribute, value in _IMPORT_FORMULA_BINDINGS
    )


def _current_formula_implementation_fingerprint() -> str:
    records = _formula_source_records()
    return _fingerprint(
        {
            "api_version": VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_API_VERSION,
            "interaction_identity": VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_IDENTITY,
            "module_files": tuple(
                sorted(
                    (
                        getattr(module, "__name__", ""),
                        _module_file_sha256(module),
                    )
                    for module in (
                        _response_module,
                        _hessian_module,
                        _inventory_module,
                        _core_module,
                        _scalar_module,
                        sys.modules[__name__],
                    )
                )
            ),
            "formula_sources": records,
            "exact_integer_scalar_implementation": (
                _scalar_module.vituri2024_exact_integer_signed_scalar_implementation_fingerprint()
            ),
            "numpy_version": np.__version__,
        }
    )


_IMPORT_FORMULA_IMPLEMENTATION_FINGERPRINT = (
    _current_formula_implementation_fingerprint()
)


def vituri2024_q003_same_functional_formula_implementation_fingerprint() -> str:
    """Return the closed production/scalar formula fingerprint."""

    current = _current_formula_implementation_fingerprint()
    if current != _IMPORT_FORMULA_IMPLEMENTATION_FINGERPRINT:
        raise RuntimeError("same-functional formula implementation drifted")
    return current


def _strict_attestation_bytes(value: object, role: str) -> bytes:
    if type(value) is not bytes:
        raise TypeError(f"{role} must be immutable exact bytes")
    expected = dict(_PINNED_LINEAGE_SHA256)[role]
    if _SHA256(value).hexdigest() != expected:
        raise ValueError(f"{role} does not match its pinned q003 lineage bytes")
    return value


def _validate_lineage_attestations(
    *,
    stationarity_attestation: object,
    source_fock_closure_attestation: object,
    exact_unitary_step_ladder_attestation: object,
) -> tuple[tuple[str, str], ...]:
    values = (
        ("stationarity_477106", stationarity_attestation),
        ("source_fock_closure_487653", source_fock_closure_attestation),
        ("exact_unitary_step_ladder_496930", exact_unitary_step_ladder_attestation),
    )
    checked = tuple(
        (role, _strict_attestation_bytes(value, role)) for role, value in values
    )
    stationarity, closure, ladder = tuple(
        _JSON_LOADS(value.decode("utf-8")) for _role, value in checked
    )
    expected_density = dict(_PINNED_SOURCE_DENSITY_GROUP_SHA256)
    stationary_density = {
        item["final_density_sha256"]
        for item in stationarity["normal_endpoints"]["groups"]
    }
    closure_groups = {
        item["label"]: item
        for item in closure["fixed_source_fock_closure"]["groups"]
    }
    ladder_groups = {
        item["label"]: item for item in ladder["inventory"]["density_groups"]
    }
    semantic_binding = (
        stationarity.get("schema")
        == "mean_field.vituri2024.fig2_q003_dense_oracle_replay_477106_attestation.v1",
        stationarity["execution"]["all_normal_endpoints_stationary"] is True,
        stationarity["normal_endpoints"]["stationary_group_count"] == 2,
        stationary_density == set(expected_density.values()),
        closure.get("schema")
        == "mean_field.vituri2024.fig2_q003_source_fock_closure_attestation.v1",
        closure["fixed_source_fock_closure"]["both_fixed_q003_sources_closed"] is True,
        set(closure_groups) == set(expected_density),
        all(
            closure_groups[label]["source_density_group_sha256"] == density
            and closure_groups[label]["independent_fock_array_sha256"]
            == dict(_PINNED_INDEPENDENT_SOURCE_FOCK_SHA256)[label]
            for label, density in expected_density.items()
        ),
        ladder.get("schema")
        == "mean_field.vituri2024.fig2_q003_exact_unitary_step_ladder_attestation.v1",
        ladder["inventory"]["all_evaluated_without_postselection"] is True,
        ladder["inventory"]["group_count"] == 2,
        set(ladder_groups) == set(expected_density),
        all(
            ladder_groups[label]["source_density_group_sha256"] == density
            for label, density in expected_density.items()
        ),
        ladder["narrow_verdict"]["universal_same_functional_scalar_hessian_established"]
        is False,
    )
    if not all(semantic_binding):
        raise ValueError("q003 lineage attestations are not semantically cross-bound")
    return tuple((role, _SHA256(value).hexdigest()) for role, value in checked)


def _select_pinned_group(
    certificate: Vituri2024Q003SourceBoundReciprocityCertificate,
    group_label: object,
) -> Vituri2024Q003SourceBoundReciprocityGroup:
    if type(certificate) is not Vituri2024Q003SourceBoundReciprocityCertificate:
        raise TypeError("bridge requires the exact pinned q003 reciprocity certificate")
    certificate.validate_live_state()
    if type(group_label) is not str or group_label not in ("3307", "40fd"):
        raise ValueError("q003 source group must be exactly '3307' or '40fd'")
    matches = tuple(
        group for group in certificate.groups if group.group_label == group_label
    )
    if len(matches) != 1:
        raise ValueError("q003 reciprocity certificate group inventory is incomplete")
    return matches[0]


def _validate_independent_source_fock(
    value: object,
    *,
    nk: int,
    group_label: str,
) -> Array:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.complex128)
        or value.shape != (4, 4, nk)
        or value.flags.writeable
        or not value.flags.c_contiguous
    ):
        raise ValueError(
            "independent source Fock must be finite immutable C-contiguous "
            f"complex128 (4,4,{nk})"
        )
    # A read-only ndarray can still alias caller-owned writable storage.  Copy
    # first into an immutable bytes owner, then perform every finite/hash/use
    # operation on that detached snapshot.
    snapshot = np.frombuffer(value.tobytes(order="C"), dtype=np.complex128).reshape(
        (4, 4, nk)
    )
    snapshot.setflags(write=False)
    if not np.all(np.isfinite(snapshot)):
        raise ValueError(
            "independent source Fock must be finite immutable C-contiguous "
            f"complex128 (4,4,{nk})"
        )
    expected = dict(_PINNED_INDEPENDENT_SOURCE_FOCK_SHA256)[group_label]
    if _array_sha256(snapshot) != expected:
        raise ValueError("independent source Fock is not the pinned q003 source Fock")
    return snapshot


def _validate_context_source_binding(
    context: object,
    group: Vituri2024Q003SourceBoundReciprocityGroup,
) -> Vituri2024HFSpiralFullHessianContext:
    if type(context) is not Vituri2024HFSpiralFullHessianContext:
        raise TypeError("bridge requires the exact live full-Hessian context")
    context.validate_live_state()
    preparation = context.inventory.restricted_preparation
    if (
        context.context_fingerprint != group.context_fingerprint
        or context.response.response_fingerprint != group.response_fingerprint
        or context.inventory.inventory_fingerprint != group.inventory_fingerprint
        or preparation.receipt.density_native_sha256 != group.density_sha256
        or preparation.receipt.fresh_hamiltonian_conventional_sha256
        != group.fresh_hamiltonian_sha256
    ):
        raise ValueError("live context is not exactly bound to the pinned q003 source")
    return context


def _validate_q003_inventory_counts(context: object) -> tuple[int, int, int]:
    if type(context) is not Vituri2024HFSpiralFullHessianContext:
        raise TypeError("inventory check requires the exact full-Hessian context")
    inventory = context.inventory
    inventory.validate_live_state()
    signed_sector_count = 0
    independently_summed_complex_dimension = 0
    for key in inventory.iter_sector_keys(include_zero_dimension=True):
        dimension = inventory.sector_complex_dimension(key)
        independently_summed_complex_dimension += dimension
        if dimension:
            signed_sector_count += 1
    canonical_orbit_count = 0
    canonical_orbit_complex_dimension = 0
    for orbit in inventory.iter_conjugate_orbits(include_zero_dimension=False):
        canonical_orbit_count += 1
        canonical_orbit_complex_dimension += orbit.complex_dimension
    expected = (
        inventory.selected_occupied_count == VITURI2024_Q003_SELECTED_OCCUPIED_COUNT,
        inventory.selected_virtual_count == VITURI2024_Q003_SELECTED_VIRTUAL_COUNT,
        inventory.nonempty_sector_count == VITURI2024_Q003_NONEMPTY_SECTOR_COUNT,
        signed_sector_count == VITURI2024_Q003_NONEMPTY_SECTOR_COUNT,
        inventory.nonempty_conjugate_orbit_count
        == VITURI2024_Q003_CANONICAL_ORBIT_COUNT,
        canonical_orbit_count == VITURI2024_Q003_CANONICAL_ORBIT_COUNT,
        inventory.complex_dimension == VITURI2024_Q003_COMPLEX_DIMENSION,
        independently_summed_complex_dimension
        == VITURI2024_Q003_COMPLEX_DIMENSION,
        canonical_orbit_complex_dimension == VITURI2024_Q003_COMPLEX_DIMENSION,
        inventory.real_dimension == VITURI2024_Q003_REAL_DIMENSION,
        inventory.complex_dimension
        == inventory.selected_occupied_count * inventory.selected_virtual_count,
    )
    if not all(expected):
        raise ValueError("incomplete q003 selected-spin fixed-rank inventory")
    return (
        signed_sector_count,
        canonical_orbit_count,
        independently_summed_complex_dimension,
    )


def _compare_full_transition_geometry(
    context: Vituri2024HFSpiralFullHessianContext,
) -> tuple[int, int, int, int]:
    """Compare support, flavor blocks, and dimensions on the signed domain.

    Every nonempty canonical orbit and both of its signed lanes are traversed.
    Support equality is evaluated once per displacement, and transition
    dimensions are counted vectorially from the occupation masks.  This check
    does not build transition embeddings or establish transition ordering,
    ``J``, or ``J^dagger``.
    """

    context.validate_live_state()
    inventory = context.inventory
    plan = context.response.fft_plan
    occupations = inventory.selected_occupations
    ordered_pair_dimensions_by_displacement: dict[tuple[int, int], Array] = {}
    support_mismatch_count = 0
    flavor_block_mismatch_count = 0
    transition_dimension_mismatch_count = 0
    summed_transition_dimension = 0
    signed_lane_count = 0
    for orbit in inventory.iter_conjugate_orbits(include_zero_dimension=False):
        for key in (orbit.first, orbit.second):
            dimension = inventory.sector_complex_dimension(key)
            if dimension == 0:
                continue
            signed_lane_count += 1
            displacement = (key.displacement_x, key.displacement_y)
            if displacement not in ordered_pair_dimensions_by_displacement:
                production_bases, production_targets = (
                    _response_module._support_indices(inventory, key)
                )
                scalar_bases, scalar_targets = _scalar_module._support(
                    plan, displacement
                )
                if not (
                    np.array_equal(production_bases, scalar_bases)
                    and np.array_equal(production_targets, scalar_targets)
                ):
                    support_mismatch_count += 1
                hole_occupied = occupations[:, scalar_bases]
                particle_virtual = ~occupations[:, scalar_targets]
                ordered_pair_dimensions_by_displacement[displacement] = (
                    np.count_nonzero(
                        particle_virtual[:, None, :]
                        & hole_occupied[None, :, :],
                        axis=2,
                    )
                )
            production_blocks = _response_module._allowed_flavor_blocks(key)
            scalar_blocks = _scalar_module._allowed_blocks(key.valley_charge)
            if production_blocks != scalar_blocks:
                flavor_block_mismatch_count += 1
            block_indices = np.asarray(scalar_blocks, dtype=np.int64)
            ordered_pair_dimensions = (
                ordered_pair_dimensions_by_displacement[displacement]
            )
            expected_dimension = int(
                ordered_pair_dimensions[
                    block_indices[:, 0], block_indices[:, 1]
                ].sum(dtype=np.int64)
            )
            summed_transition_dimension += expected_dimension
            if expected_dimension != dimension:
                transition_dimension_mismatch_count += 1
    if (
        support_mismatch_count
        or flavor_block_mismatch_count
        or transition_dimension_mismatch_count
        or signed_lane_count != VITURI2024_Q003_NONEMPTY_SECTOR_COUNT
        or summed_transition_dimension != VITURI2024_Q003_COMPLEX_DIMENSION
    ):
        raise ValueError(
            "production/scalar transition-domain comparison disagrees on the "
            "full q003 inventory: "
            f"support_mismatches={support_mismatch_count}, "
            f"flavor_block_mismatches={flavor_block_mismatch_count}, "
            f"transition_dimension_mismatches={transition_dimension_mismatch_count}, "
            f"signed_lanes={signed_lane_count}, "
            f"summed_dimension={summed_transition_dimension}"
        )
    return (
        signed_lane_count,
        support_mismatch_count,
        flavor_block_mismatch_count,
        transition_dimension_mismatch_count,
    )


def _maximum_transition_gap_residual(
    diagonal_residual: Array,
    occupations: Array,
) -> float:
    flattened = np.asarray(diagonal_residual, dtype=np.float64).reshape(-1)
    occupied = np.asarray(occupations, dtype=np.bool_).reshape(-1)
    virtual_values = flattened[~occupied]
    occupied_values = flattened[occupied]
    if virtual_values.size == 0 or occupied_values.size == 0:
        raise ValueError("one-body action comparison requires occupied and virtual states")
    return float(
        max(
            abs(float(np.max(virtual_values)) - float(np.min(occupied_values))),
            abs(float(np.min(virtual_values)) - float(np.max(occupied_values))),
        )
    )


def _one_body_superoperator_difference_bound(
    selected_exact: Array,
    production_diagonal: Array,
    occupations: Array,
) -> float:
    """Bound ``||(F_v-D_v)X-X(F_o-D_o)||_F / ||X||_F`` exactly enough.

    The selected Fock is k-block diagonal.  The occupied and virtual restrictions
    are therefore direct sums of at-most-2x2 blocks, whose spectral norms can be
    maximized without allocating the 15,052,040-dimensional tangent action.
    """

    maximum_virtual_norm = 0.0
    maximum_occupied_norm = 0.0
    for momentum in range(selected_exact.shape[-1]):
        delta = np.array(selected_exact[:, :, momentum], copy=True)
        delta[0, 0] -= production_diagonal[0, momentum]
        delta[1, 1] -= production_diagonal[1, momentum]
        occupation = occupations[:, momentum]
        occupied_indices = np.flatnonzero(occupation)
        virtual_indices = np.flatnonzero(~occupation)
        if occupied_indices.size:
            occupied_block = delta[np.ix_(occupied_indices, occupied_indices)]
            maximum_occupied_norm = max(
                maximum_occupied_norm,
                float(np.linalg.norm(occupied_block, ord=2)),
            )
        if virtual_indices.size:
            virtual_block = delta[np.ix_(virtual_indices, virtual_indices)]
            maximum_virtual_norm = max(
                maximum_virtual_norm,
                float(np.linalg.norm(virtual_block, ord=2)),
            )
    return maximum_virtual_norm + maximum_occupied_norm


def _compare_one_body_operators(
    context: Vituri2024HFSpiralFullHessianContext,
    independent_source_fock: Array,
) -> tuple[float, float, float, float, float]:
    preparation = context.inventory.restricted_preparation
    production_full = preparation.fresh_hamiltonian_conventional
    closure_residual = float(
        np.max(np.abs(independent_source_fock - production_full), initial=0.0)
    )
    if closure_residual > VITURI2024_Q003_SOURCE_FOCK_CLOSURE_TOLERANCE_EV:
        raise ValueError("independent and production source Fock matrices disagree")

    selected = np.asarray(
        preparation.receipt.selected_flavor_indices, dtype=np.int64
    )
    momenta = np.arange(context.nk, dtype=np.int64)
    selected_exact = independent_source_fock[np.ix_(selected, selected, momenta)]
    off_diagonal = np.array(selected_exact, copy=True)
    off_diagonal[0, 0] = 0.0
    off_diagonal[1, 1] = 0.0
    off_diagonal_residual = float(np.max(np.abs(off_diagonal), initial=0.0))
    diagonal_imaginary_residual = float(
        np.max(
            np.abs(
                np.stack((selected_exact[0, 0].imag, selected_exact[1, 1].imag))
            ),
            initial=0.0,
        )
    )
    basis_residual = max(off_diagonal_residual, diagonal_imaginary_residual)
    if basis_residual > VITURI2024_Q003_ONE_BODY_BASIS_TOLERANCE_EV:
        raise ValueError(
            "independent source Fock is not diagonal in the registered tangent basis"
        )

    d_f_diagonal = np.stack(
        (selected_exact[0, 0].real, selected_exact[1, 1].real)
    )
    diagonal_residual_array = d_f_diagonal - context.selected_fock_diagonal_ev
    diagonal_residual = float(
        np.max(np.abs(diagonal_residual_array), initial=0.0)
    )
    if diagonal_residual > VITURI2024_Q003_ONE_BODY_DIAGONAL_TOLERANCE_EV:
        raise ValueError("D_prod and D_F source diagonals disagree")
    diagonal_gap_residual = _maximum_transition_gap_residual(
        diagonal_residual_array,
        context.inventory.selected_occupations,
    )
    action_residual = _one_body_superoperator_difference_bound(
        selected_exact,
        context.selected_fock_diagonal_ev,
        context.inventory.selected_occupations,
    )
    if (
        diagonal_gap_residual > VITURI2024_Q003_ONE_BODY_ACTION_TOLERANCE_EV
        or action_residual > VITURI2024_Q003_ONE_BODY_ACTION_TOLERANCE_EV
    ):
        raise ValueError("D_prod and D_F disagree on the full transition inventory")
    return (
        closure_residual,
        off_diagonal_residual,
        diagonal_imaginary_residual,
        diagonal_residual,
        action_residual,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024Q003SameFunctionalSourceRecord:
    """Array-free structural record for exactly one pinned q003 source."""

    _factory_token: InitVar[object]
    group_label: str
    source_group_fingerprint: str
    context_fingerprint: str
    response_fingerprint: str
    inventory_fingerprint: str
    independent_source_fock_sha256: str
    source_fock_closure_residual_ev: float
    selected_fock_offdiagonal_residual_ev: float
    selected_fock_diagonal_imaginary_residual_ev: float
    maximum_d_prod_d_f_diagonal_residual_ev: float
    maximum_d_prod_d_f_action_bound_ev: float
    nonempty_sector_count: int
    canonical_orbit_count: int
    complex_dimension: int
    real_dimension: int
    signed_lane_count_checked: int
    support_mismatch_count: int
    flavor_block_mismatch_count: int
    transition_dimension_mismatch_count: int
    fingerprint: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _REQUIRE_FACTORY_TOKEN(_factory_token, "same-functional source records")
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("_factory_token", "fingerprint")
        }
        object.__setattr__(self, "fingerprint", _fingerprint(payload))


@dataclass(frozen=True, slots=True)
class Vituri2024Q003SameFunctionalStructuralReceipt:
    """Factory-only two-source candidate receipt with no operator references."""

    _factory_token: InitVar[object]
    _reciprocity_certificate: Vituri2024Q003SourceBoundReciprocityCertificate = field(
        repr=False, compare=False
    )
    sources: tuple[Vituri2024Q003SameFunctionalSourceRecord, ...]
    reciprocity_certificate_fingerprint: str
    formula_implementation_fingerprint: str
    formula_source_records: tuple[tuple[str, str, str, str], ...]
    lineage_sha256: tuple[tuple[str, str], ...]
    receipt_fingerprint: str = field(init=False)
    api_version: str = field(
        default=VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_API_VERSION, init=False
    )
    scope: str = field(default=VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_SCOPE, init=False)
    authority: str = field(
        default=VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_AUTHORITY, init=False
    )
    interaction_identity: str = field(
        default=VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_IDENTITY, init=False
    )
    interaction_derivation: tuple[str, ...] = field(
        default=VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_DERIVATION,
        init=False,
    )
    hessian_identity: str = field(
        default="candidate_derivation_pending_detached_review:H_prod_to_2Jdagger(D_F+Sigma)J_to_H_E",
        init=False,
    )
    normalization: str = field(
        default="raw_total_energy_no_Nk_division_no_extra_interaction_half_real_factor_two",
        init=False,
    )
    one_body_basis_tolerance_ev: float = field(
        default=VITURI2024_Q003_ONE_BODY_BASIS_TOLERANCE_EV, init=False
    )
    one_body_diagonal_tolerance_ev: float = field(
        default=VITURI2024_Q003_ONE_BODY_DIAGONAL_TOLERANCE_EV, init=False
    )
    one_body_action_tolerance_ev: float = field(
        default=VITURI2024_Q003_ONE_BODY_ACTION_TOLERANCE_EV, init=False
    )
    source_fock_closure_tolerance_ev: float = field(
        default=VITURI2024_Q003_SOURCE_FOCK_CLOSURE_TOLERANCE_EV, init=False
    )
    candidate_only: bool = field(default=True, init=False)
    candidate_structural_receipt_established: bool = field(default=True, init=False)
    exact_live_source_binding: bool = field(default=True, init=False)
    both_q003_sources_covered: bool = field(default=True, init=False)
    d_prod_equals_d_f_within_recorded_tolerance: bool = field(default=True, init=False)
    candidate_l_prod_sigma_structural_derivation_recorded: bool = field(
        default=True, init=False
    )
    l_prod_equals_sigma_structurally_derived: bool = field(default=False, init=False)
    l_prod_sigma_numerical_equality_assumed: bool = field(default=False, init=False)
    full_selected_spin_fixed_rank_inventory_covered: bool = field(default=True, init=False)
    candidate_transition_adjoint_derivation_recorded: bool = field(
        default=True, init=False
    )
    transition_injection_extraction_adjoint_derived: bool = field(
        default=False, init=False
    )
    literal_float_full_functional_parity_established: bool = field(
        default=False, init=False
    )
    both_signed_lanes_retained: bool = field(default=True, init=False)
    raw_total_energy_units: bool = field(default=True, init=False)
    extra_interaction_half_factor: bool = field(default=False, init=False)
    nk_division: bool = field(default=False, init=False)
    real_hessian_factor_two: bool = field(default=True, init=False)
    no_postselection: bool = field(default=True, init=False)
    source_fock_closure_lineage_bound: bool = field(default=True, init=False)
    stationarity_lineage_bound: bool = field(default=True, init=False)
    fixed_probe_step_ladder_lineage_bound: bool = field(default=True, init=False)
    algebraic_reciprocity_certificate_bound: bool = field(default=True, init=False)
    stationarity_established: bool = field(default=False, init=False)
    full_inventory_exact_unitary_scalar_curvature_established: bool = field(
        default=False, init=False
    )
    scalar_hessian_authority_established: bool = field(default=False, init=False)
    linear_operator_authorized: bool = field(default=False, init=False)
    hermitian_eigensolver_authorized: bool = field(default=False, init=False)
    full_local_stability_established: bool = field(default=False, init=False)
    scientific_authority_promoted: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _REQUIRE_FACTORY_TOKEN(_factory_token, "same-functional structural receipts")
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("_factory_token", "_reciprocity_certificate", "receipt_fingerprint")
        }
        payload["sources"] = tuple(
            {
                name: getattr(source, name)
                for name in source.__dataclass_fields__
                if name != "_factory_token"
            }
            for source in self.sources
        )
        object.__setattr__(self, "receipt_fingerprint", _fingerprint(payload))
        self.validate_live_state()

    def validate_live_state(self) -> None:
        _validate_bridge_bindings()
        self._reciprocity_certificate.validate_live_state()
        expected_sources = _formula_source_records()
        source_groups = {group.group_label: group for group in self._reciprocity_certificate.groups}
        numeric_and_inventory: list[bool] = []
        if tuple(source.group_label for source in self.sources) != ("3307", "40fd"):
            raise ValueError("same-functional receipt must contain both q003 sources exactly once")
        for source in self.sources:
            group = source_groups[source.group_label]
            source_payload = {
                name: getattr(source, name)
                for name in source.__dataclass_fields__
                if name not in ("_factory_token", "fingerprint")
            }
            numeric_and_inventory.extend(
                (
                    source.fingerprint == _fingerprint(source_payload),
                    source.source_group_fingerprint == group.fingerprint,
                    source.context_fingerprint == group.context_fingerprint,
                    source.response_fingerprint == group.response_fingerprint,
                    source.inventory_fingerprint == group.inventory_fingerprint,
                    source.independent_source_fock_sha256
                    == dict(_PINNED_INDEPENDENT_SOURCE_FOCK_SHA256)[source.group_label],
                    source.source_fock_closure_residual_ev
                    <= self.source_fock_closure_tolerance_ev,
                    max(
                        source.selected_fock_offdiagonal_residual_ev,
                        source.selected_fock_diagonal_imaginary_residual_ev,
                    )
                    <= self.one_body_basis_tolerance_ev,
                    source.maximum_d_prod_d_f_diagonal_residual_ev
                    <= self.one_body_diagonal_tolerance_ev,
                    source.maximum_d_prod_d_f_action_bound_ev
                    <= self.one_body_action_tolerance_ev,
                    source.nonempty_sector_count == VITURI2024_Q003_NONEMPTY_SECTOR_COUNT,
                    source.canonical_orbit_count == VITURI2024_Q003_CANONICAL_ORBIT_COUNT,
                    source.complex_dimension == VITURI2024_Q003_COMPLEX_DIMENSION,
                    source.real_dimension == VITURI2024_Q003_REAL_DIMENSION,
                    source.signed_lane_count_checked == VITURI2024_Q003_NONEMPTY_SECTOR_COUNT,
                    source.support_mismatch_count == 0,
                    source.flavor_block_mismatch_count == 0,
                    source.transition_dimension_mismatch_count == 0,
                )
            )
        locked = (
            self.reciprocity_certificate_fingerprint == self._reciprocity_certificate.fingerprint,
            self.formula_implementation_fingerprint
            == vituri2024_q003_same_functional_formula_implementation_fingerprint(),
            self.formula_source_records == expected_sources,
            self.lineage_sha256 == _PINNED_LINEAGE_SHA256,
            self.api_version == VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_API_VERSION,
            self.scope == VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_SCOPE,
            self.authority == VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_AUTHORITY,
            self.interaction_identity == VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_IDENTITY,
            self.interaction_derivation
            == VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_DERIVATION,
            self.hessian_identity
            == "candidate_derivation_pending_detached_review:H_prod_to_2Jdagger(D_F+Sigma)J_to_H_E",
            self.normalization
            == "raw_total_energy_no_Nk_division_no_extra_interaction_half_real_factor_two",
            self.one_body_basis_tolerance_ev
            == VITURI2024_Q003_ONE_BODY_BASIS_TOLERANCE_EV,
            self.one_body_diagonal_tolerance_ev
            == VITURI2024_Q003_ONE_BODY_DIAGONAL_TOLERANCE_EV,
            self.one_body_action_tolerance_ev
            == VITURI2024_Q003_ONE_BODY_ACTION_TOLERANCE_EV,
            self.source_fock_closure_tolerance_ev
            == VITURI2024_Q003_SOURCE_FOCK_CLOSURE_TOLERANCE_EV,
            self.candidate_only is True,
            self.candidate_structural_receipt_established is True,
            self.exact_live_source_binding is True,
            self.both_q003_sources_covered is True,
            self.d_prod_equals_d_f_within_recorded_tolerance is True,
            self.candidate_l_prod_sigma_structural_derivation_recorded is True,
            self.l_prod_equals_sigma_structurally_derived is False,
            self.l_prod_sigma_numerical_equality_assumed is False,
            self.full_selected_spin_fixed_rank_inventory_covered is True,
            self.candidate_transition_adjoint_derivation_recorded is True,
            self.transition_injection_extraction_adjoint_derived is False,
            self.literal_float_full_functional_parity_established is False,
            self.both_signed_lanes_retained is True,
            self.raw_total_energy_units is True,
            self.extra_interaction_half_factor is False,
            self.nk_division is False,
            self.real_hessian_factor_two is True,
            self.no_postselection is True,
            self.source_fock_closure_lineage_bound is True,
            self.stationarity_lineage_bound is True,
            self.fixed_probe_step_ladder_lineage_bound is True,
            self.algebraic_reciprocity_certificate_bound is True,
            self.stationarity_established is False,
            self.full_inventory_exact_unitary_scalar_curvature_established is False,
            self.scalar_hessian_authority_established is False,
            self.linear_operator_authorized is False,
            self.hermitian_eigensolver_authorized is False,
            self.full_local_stability_established is False,
            self.scientific_authority_promoted is False,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
            *numeric_and_inventory,
        )
        if not all(locked):
            raise ValueError("same-functional structural receipt binding drifted")
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("_factory_token", "_reciprocity_certificate", "receipt_fingerprint")
        }
        payload["sources"] = tuple(
            {
                name: getattr(source, name)
                for name in source.__dataclass_fields__
                if name != "_factory_token"
            }
            for source in self.sources
        )
        if self.receipt_fingerprint != _fingerprint(payload):
            raise ValueError("same-functional structural receipt fingerprint drifted")


def _build_source_record(
    *,
    context: Vituri2024HFSpiralFullHessianContext,
    source_fock: Array,
    group: Vituri2024Q003SourceBoundReciprocityGroup,
    factory_constructor: object,
) -> Vituri2024Q003SameFunctionalSourceRecord:
    live_context = _validate_context_source_binding(context, group)
    signed_sector_count, orbit_count, complex_dimension = (
        _validate_q003_inventory_counts(live_context)
    )
    if (
        signed_sector_count != group.nonempty_sector_count
        or orbit_count != group.canonical_orbit_count
    ):
        raise ValueError("live q003 inventory disagrees with reciprocity certificate")
    signed_lanes, support_mismatches, flavor_mismatches, dimension_mismatches = (
        _compare_full_transition_geometry(live_context)
    )
    closure, offdiagonal, diagonal_imaginary, diagonal, action_bound = (
        _compare_one_body_operators(live_context, source_fock)
    )
    if not callable(factory_constructor):
        raise TypeError("same-functional source factory constructor is unavailable")
    return factory_constructor(
        Vituri2024Q003SameFunctionalSourceRecord,
        group_label=group.group_label,
        source_group_fingerprint=group.fingerprint,
        context_fingerprint=live_context.context_fingerprint,
        response_fingerprint=live_context.response.response_fingerprint,
        inventory_fingerprint=live_context.inventory.inventory_fingerprint,
        independent_source_fock_sha256=_array_sha256(source_fock),
        source_fock_closure_residual_ev=closure,
        selected_fock_offdiagonal_residual_ev=offdiagonal,
        selected_fock_diagonal_imaginary_residual_ev=diagonal_imaginary,
        maximum_d_prod_d_f_diagonal_residual_ev=diagonal,
        maximum_d_prod_d_f_action_bound_ev=action_bound,
        nonempty_sector_count=signed_sector_count,
        canonical_orbit_count=orbit_count,
        complex_dimension=complex_dimension,
        real_dimension=live_context.inventory.real_dimension,
        signed_lane_count_checked=signed_lanes,
        support_mismatch_count=support_mismatches,
        flavor_block_mismatch_count=flavor_mismatches,
        transition_dimension_mismatch_count=dimension_mismatches,
    )


def _build_vituri2024_q003_same_functional_structural_receipt_impl(
    *,
    context_3307: Vituri2024HFSpiralFullHessianContext,
    independent_source_fock_3307: Array,
    context_40fd: Vituri2024HFSpiralFullHessianContext,
    independent_source_fock_40fd: Array,
    reciprocity_certificate: Vituri2024Q003SourceBoundReciprocityCertificate,
    stationarity_attestation: bytes,
    source_fock_closure_attestation: bytes,
    exact_unitary_step_ladder_attestation: bytes,
    _factory_constructor: object,
) -> Vituri2024Q003SameFunctionalStructuralReceipt:
    lineage = _validate_lineage_attestations(
        stationarity_attestation=stationarity_attestation,
        source_fock_closure_attestation=source_fock_closure_attestation,
        exact_unitary_step_ladder_attestation=exact_unitary_step_ladder_attestation,
    )
    _validate_bridge_bindings()
    for context in (context_3307, context_40fd):
        if type(context) is not Vituri2024HFSpiralFullHessianContext:
            raise TypeError("bridge requires exact live full-Hessian contexts")
    fock_3307 = _validate_independent_source_fock(
        independent_source_fock_3307, nk=context_3307.nk, group_label="3307"
    )
    fock_40fd = _validate_independent_source_fock(
        independent_source_fock_40fd, nk=context_40fd.nk, group_label="40fd"
    )
    group_3307 = _select_pinned_group(reciprocity_certificate, "3307")
    group_40fd = _select_pinned_group(reciprocity_certificate, "40fd")
    sources = (
        _build_source_record(
            context=context_3307,
            source_fock=fock_3307,
            group=group_3307,
            factory_constructor=_factory_constructor,
        ),
        _build_source_record(
            context=context_40fd,
            source_fock=fock_40fd,
            group=group_40fd,
            factory_constructor=_factory_constructor,
        ),
    )
    if not callable(_factory_constructor):
        raise TypeError("same-functional receipt factory constructor is unavailable")
    receipt = _factory_constructor(
        Vituri2024Q003SameFunctionalStructuralReceipt,
        _reciprocity_certificate=reciprocity_certificate,
        sources=sources,
        reciprocity_certificate_fingerprint=reciprocity_certificate.fingerprint,
        formula_implementation_fingerprint=(
            vituri2024_q003_same_functional_formula_implementation_fingerprint()
        ),
        formula_source_records=_formula_source_records(),
        lineage_sha256=lineage,
    )
    _validate_bridge_bindings()
    return receipt


_RECEIPT_METHOD_SPECS = (
    (Vituri2024Q003SameFunctionalSourceRecord, "__post_init__"),
    (Vituri2024Q003SameFunctionalStructuralReceipt, "__post_init__"),
    (Vituri2024Q003SameFunctionalStructuralReceipt, "validate_live_state"),
)
_IMPORT_RECEIPT_METHOD_BINDINGS = tuple(
    (owner, name, getattr(owner, name)) for owner, name in _RECEIPT_METHOD_SPECS
)


def _validate_bridge_bindings() -> None:
    _validate_formula_bindings()
    for name, expected in _IMPORT_BRIDGE_LOCAL_ALIAS_BINDINGS:
        if globals().get(name) is not expected:
            raise RuntimeError(
                f"same-functional bridge local alias drifted: {name}"
            )
    for owner, attribute, expected in _IMPORT_BRIDGE_PRIMITIVE_BINDINGS:
        if getattr(owner, attribute, None) is not expected:
            raise RuntimeError(
                "same-functional bridge primitive binding drifted: "
                f"{getattr(owner, '__name__', type(owner).__name__)}.{attribute}"
            )
    for name, expected in _IMPORT_BRIDGE_BINDINGS:
        if globals().get(name) is not expected:
            raise RuntimeError(f"same-functional bridge function binding drifted: {name}")
    for owner, name, expected in _IMPORT_RECEIPT_METHOD_BINDINGS:
        if getattr(owner, name) is not expected:
            raise RuntimeError(
                f"same-functional receipt method binding drifted: {owner.__name__}.{name}"
            )
    if _SHA256 is not sha256 or json.loads is not _JSON_LOADS:
        raise RuntimeError("same-functional bridge primitive binding drifted")
    if (
        _PINNED_LINEAGE_SHA256 != _IMPORT_PINNED_LINEAGE_SHA256
        or _PINNED_SOURCE_DENSITY_GROUP_SHA256
        != _IMPORT_PINNED_SOURCE_DENSITY_GROUP_SHA256
        or _PINNED_INDEPENDENT_SOURCE_FOCK_SHA256
        != _IMPORT_PINNED_INDEPENDENT_SOURCE_FOCK_SHA256
        or _FORMULA_BINDING_SPECS != _IMPORT_FORMULA_BINDING_SPECS
    ):
        raise RuntimeError("same-functional bridge constant binding drifted")


_IMPORT_BRIDGE_BINDINGS = tuple(
    (name, globals()[name])
    for name in (
        "_make_factory_boundary",
        "_REQUIRE_FACTORY_TOKEN",
        "_module_own_callables",
        "_class_descriptors",
        "_module_class_bindings",
        "_canonical",
        "_fingerprint",
        "_array_sha256",
        "_source_sha256",
        "_module_file_sha256",
        "_validate_formula_bindings",
        "_formula_source_records",
        "_current_formula_implementation_fingerprint",
        "vituri2024_q003_same_functional_formula_implementation_fingerprint",
        "_strict_attestation_bytes",
        "_validate_lineage_attestations",
        "_select_pinned_group",
        "_validate_independent_source_fock",
        "_validate_context_source_binding",
        "_validate_q003_inventory_counts",
        "_compare_full_transition_geometry",
        "_maximum_transition_gap_residual",
        "_one_body_superoperator_difference_bound",
        "_compare_one_body_operators",
        "Vituri2024Q003SameFunctionalSourceRecord",
        "Vituri2024Q003SameFunctionalStructuralReceipt",
        "_build_source_record",
        "_build_vituri2024_q003_same_functional_structural_receipt_impl",
        "_validate_bridge_bindings",
    )
)
_IMPORT_PINNED_LINEAGE_SHA256 = _PINNED_LINEAGE_SHA256
_IMPORT_PINNED_SOURCE_DENSITY_GROUP_SHA256 = _PINNED_SOURCE_DENSITY_GROUP_SHA256
_IMPORT_PINNED_INDEPENDENT_SOURCE_FOCK_SHA256 = (
    _PINNED_INDEPENDENT_SOURCE_FOCK_SHA256
)
_IMPORT_FORMULA_BINDING_SPECS = _FORMULA_BINDING_SPECS


def _make_public_builder(
    implementation: object,
    guard: object,
    binding_snapshot: tuple[tuple[str, object], ...],
    local_alias_snapshot: tuple[tuple[str, object], ...],
    primitive_binding_snapshot: tuple[tuple[object, str, object], ...],
    lineage_snapshot: tuple[tuple[str, str], ...],
    density_group_snapshot: tuple[tuple[str, str], ...],
    fock_snapshot: tuple[tuple[str, str], ...],
    formula_binding_snapshot: tuple[tuple[str, object, str], ...],
    runtime_binding_snapshot: tuple[tuple[object, str, object], ...],
    formula_implementation_fingerprint_snapshot: str,
    owner_callable_snapshot: tuple[tuple[object, tuple[tuple[str, object], ...]], ...],
    owner_class_snapshot: tuple[
        tuple[object, tuple[tuple[type, tuple[tuple[str, tuple[object, ...]], ...]], ...]],
        ...,
    ],
    receipt_method_snapshot: tuple[tuple[type, str, object], ...],
    reciprocity_constant_snapshot: tuple[tuple[str, object], ...],
    sha256_snapshot: object,
    json_loads_snapshot: object,
    factory_constructor: object,
    namespace: dict[str, object],
):
    if (
        not callable(implementation)
        or not callable(guard)
        or not callable(factory_constructor)
    ):
        raise TypeError("same-functional builder closure inputs must be callable")

    def immutable_guard() -> None:
        if "globals" in namespace:
            raise RuntimeError("same-functional bridge shadows the globals builtin")
        if namespace.get("_IMPORT_BRIDGE_LOCAL_ALIAS_BINDINGS") is not local_alias_snapshot:
            raise RuntimeError("same-functional bridge local alias table drifted")
        for name, expected in local_alias_snapshot:
            if namespace.get(name) is not expected:
                raise RuntimeError(
                    f"same-functional bridge local alias drifted: {name}"
                )
        if namespace.get("_IMPORT_BRIDGE_PRIMITIVE_BINDINGS") is not primitive_binding_snapshot:
            raise RuntimeError("same-functional bridge primitive table drifted")
        for owner, attribute, expected in primitive_binding_snapshot:
            if getattr(owner, attribute, None) is not expected:
                raise RuntimeError(
                    "same-functional bridge primitive binding drifted: "
                    f"{getattr(owner, '__name__', type(owner).__name__)}.{attribute}"
                )
        if namespace.get("_IMPORT_BRIDGE_BINDINGS") is not binding_snapshot:
            raise RuntimeError("same-functional bridge binding table drifted")
        if namespace.get("_PINNED_LINEAGE_SHA256") != lineage_snapshot:
            raise RuntimeError("same-functional lineage constants drifted")
        if namespace.get("_PINNED_SOURCE_DENSITY_GROUP_SHA256") != density_group_snapshot:
            raise RuntimeError("same-functional source-density constants drifted")
        if namespace.get("_PINNED_INDEPENDENT_SOURCE_FOCK_SHA256") != fock_snapshot:
            raise RuntimeError("same-functional source-Fock constants drifted")
        if namespace.get("_SHA256") is not sha256_snapshot:
            raise RuntimeError("same-functional SHA256 primitive drifted")
        if namespace.get("sha256") is not sha256_snapshot:
            raise RuntimeError("same-functional SHA256 import drifted")
        if namespace.get("_JSON_LOADS") is not json_loads_snapshot:
            raise RuntimeError("same-functional JSON primitive drifted")
        if namespace.get("_FORMULA_BINDING_SPECS") is not formula_binding_snapshot:
            raise RuntimeError("same-functional formula binding specs drifted")
        if namespace.get("_IMPORT_RUNTIME_BINDINGS") is not runtime_binding_snapshot:
            raise RuntimeError("same-functional runtime binding table drifted")
        if namespace.get(
            "_IMPORT_FORMULA_IMPLEMENTATION_FINGERPRINT"
        ) != formula_implementation_fingerprint_snapshot:
            raise RuntimeError(
                "same-functional import formula implementation fingerprint drifted"
            )
        for owner, attribute, expected in runtime_binding_snapshot:
            if getattr(owner, attribute, None) is not expected:
                raise RuntimeError(
                    "same-functional imported runtime binding drifted: "
                    f"{getattr(owner, '__name__', type(owner).__name__)}.{attribute}"
                )
        if namespace.get("_IMPORT_OWNER_CALLABLE_BINDINGS") is not owner_callable_snapshot:
            raise RuntimeError("same-functional owner callable table drifted")
        if namespace.get("_IMPORT_OWNER_CLASS_BINDINGS") is not owner_class_snapshot:
            raise RuntimeError("same-functional owner class table drifted")
        if namespace.get("_IMPORT_RECEIPT_METHOD_BINDINGS") is not receipt_method_snapshot:
            raise RuntimeError("same-functional receipt method table drifted")
        if namespace.get(
            "_IMPORT_RECIPROCITY_LIVE_CONSTANT_BINDINGS"
        ) is not reciprocity_constant_snapshot:
            raise RuntimeError("same-functional reciprocity constant table drifted")
        for name, expected in reciprocity_constant_snapshot:
            if getattr(_reciprocity_module, name, None) is not expected:
                raise RuntimeError(
                    "same-functional reciprocity live constant binding drifted: "
                    f"{name}"
                )
        for owner, name, expected in receipt_method_snapshot:
            if getattr(owner, name) is not expected:
                raise RuntimeError(
                    f"same-functional receipt method binding drifted: {owner.__name__}.{name}"
                )
        if namespace.get(
            "build_vituri2024_q003_same_functional_structural_receipt"
        ) is not public_builder:
            raise RuntimeError("same-functional public builder binding drifted")
        for name, expected in binding_snapshot:
            if namespace.get(name) is not expected:
                raise RuntimeError(
                    f"same-functional bridge function binding drifted: {name}"
                )
        guard()

    def public_builder(
        *,
        context_3307: Vituri2024HFSpiralFullHessianContext,
        independent_source_fock_3307: Array,
        context_40fd: Vituri2024HFSpiralFullHessianContext,
        independent_source_fock_40fd: Array,
        reciprocity_certificate: Vituri2024Q003SourceBoundReciprocityCertificate,
        stationarity_attestation: bytes,
        source_fock_closure_attestation: bytes,
        exact_unitary_step_ladder_attestation: bytes,
    ) -> Vituri2024Q003SameFunctionalStructuralReceipt:
        immutable_guard()
        result = implementation(
            context_3307=context_3307,
            independent_source_fock_3307=independent_source_fock_3307,
            context_40fd=context_40fd,
            independent_source_fock_40fd=independent_source_fock_40fd,
            reciprocity_certificate=reciprocity_certificate,
            stationarity_attestation=stationarity_attestation,
            source_fock_closure_attestation=source_fock_closure_attestation,
            exact_unitary_step_ladder_attestation=(
                exact_unitary_step_ladder_attestation
            ),
            _factory_constructor=factory_constructor,
        )
        immutable_guard()
        return result

    return public_builder


build_vituri2024_q003_same_functional_structural_receipt = _make_public_builder(
    _build_vituri2024_q003_same_functional_structural_receipt_impl,
    _validate_bridge_bindings,
    _IMPORT_BRIDGE_BINDINGS,
    _IMPORT_BRIDGE_LOCAL_ALIAS_BINDINGS,
    _IMPORT_BRIDGE_PRIMITIVE_BINDINGS,
    _PINNED_LINEAGE_SHA256,
    _PINNED_SOURCE_DENSITY_GROUP_SHA256,
    _PINNED_INDEPENDENT_SOURCE_FOCK_SHA256,
    _FORMULA_BINDING_SPECS,
    _IMPORT_RUNTIME_BINDINGS,
    _IMPORT_FORMULA_IMPLEMENTATION_FINGERPRINT,
    _IMPORT_OWNER_CALLABLE_BINDINGS,
    _IMPORT_OWNER_CLASS_BINDINGS,
    _IMPORT_RECEIPT_METHOD_BINDINGS,
    _IMPORT_RECIPROCITY_LIVE_CONSTANT_BINDINGS,
    _SHA256,
    _JSON_LOADS,
    _construct_with_factory_token,
    globals(),
)
del _construct_with_factory_token


__all__ = [
    "VITURI2024_Q003_CANONICAL_ORBIT_COUNT",
    "VITURI2024_Q003_COMPLEX_DIMENSION",
    "VITURI2024_Q003_NONEMPTY_SECTOR_COUNT",
    "VITURI2024_Q003_ONE_BODY_ACTION_TOLERANCE_EV",
    "VITURI2024_Q003_ONE_BODY_BASIS_TOLERANCE_EV",
    "VITURI2024_Q003_ONE_BODY_DIAGONAL_TOLERANCE_EV",
    "VITURI2024_Q003_REAL_DIMENSION",
    "VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_API_VERSION",
    "VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_AUTHORITY",
    "VITURI2024_Q003_SAME_FUNCTIONAL_BRIDGE_SCOPE",
    "VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_DERIVATION",
    "VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_IDENTITY",
    "VITURI2024_Q003_SELECTED_OCCUPIED_COUNT",
    "VITURI2024_Q003_SELECTED_VIRTUAL_COUNT",
    "VITURI2024_Q003_SOURCE_FOCK_CLOSURE_TOLERANCE_EV",
    "build_vituri2024_q003_same_functional_structural_receipt",
    "vituri2024_q003_same_functional_formula_implementation_fingerprint",
]
