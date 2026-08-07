"""Actual-Vituri restricted finite-orbital scalar algebra oracle.

This module is deliberately narrower than a physical full-space scalar
provider.  It takes the exact orbitals already present in one validated
Vituri C9 signed-q assembly, constructs the antisymmetrized local vertex on
that finite orbital set, and checks the resulting Wick functional against an
independent fixed-particle-number bitstring Hamiltonian.  No missing quartet
is tolerance-included and no tensor entry or A/B lane is copied, averaged,
symmetrized, Hermitized, or repaired.

The permanent authority is
``restricted_finite_orbital_algebra_oracle_only``.  A passing receipt proves
finite-orbital algebra parity with the actual projected vertex and all four
actual C9 lanes.  It does not establish a real full provider, source scalar,
global static-Hessian authority, production readiness, or paper parity.
"""

from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass, field, fields, is_dataclass
from hashlib import sha256
from itertools import combinations, product
import json
import math
from typing import Callable, Final, Literal

import numpy as np

from mean_field.core.hf.tdhf_scalar_curvature import (
    TDHFEnergyConvention,
    TDHFScalarCurvatureApproval,
    TDHFScalarCurvatureCertificate,
    TDHFScalarCurvatureStepLadder,
    TDHFScalarCurvatureTolerances,
    TDHFScalarFunctionalManifest,
    TDHFTransitionTangentBasis,
    canonical_tdhf_scalar_directions,
    certify_tdhf_scalar_curvature,
    make_tdhf_scalar_curvature_approval,
    make_tdhf_scalar_functional_manifest,
)
from mean_field.core.hf.tdhf_signed import (
    fingerprint_tdhf_matrix,
    fingerprint_tdhf_pairs,
)

from .vituri2024 import SM_TEX_SHA256
from .vituri2024_hf_replay import Vituri2024HalfMetalHFReplayPayload
from .vituri2024_tdhf import (
    Vituri2024TDHFAssemblyContext,
    Vituri2024TDHFSignedQAssemblyReceipt,
)
from .vituri2024_tdhf_scalar import Vituri2024TDHFScalarReadinessReceipt
from .vituri2024_vertex import (
    Vituri2024FourPointKinematicsReceipt,
    Vituri2024Orbital,
    vituri2024_antisymmetrized_projected_vertex,
)

Array = np.ndarray

VITURI2024_RESTRICTED_SCALAR_AUTHORITY: Final[str] = (
    "restricted_finite_orbital_algebra_oracle_only"
)
VITURI2024_RESTRICTED_SCALAR_MAX_ORBITALS: Final[int] = 8
VITURI2024_RESTRICTED_SCALAR_ALGEBRA_ABSOLUTE_TOLERANCE: Final[float] = 2.0e-10
VITURI2024_RESTRICTED_SCALAR_ALGEBRA_RELATIVE_TOLERANCE: Final[float] = 2.0e-10
VITURI2024_RESTRICTED_SCALAR_WICK_ABSOLUTE_TOLERANCE: Final[float] = 2.0e-10
VITURI2024_RESTRICTED_SCALAR_WICK_RELATIVE_TOLERANCE: Final[float] = 2.0e-10

_SIGMA_EQUATION = "Sigma[P]_ij=sum_bg wbar[i,b,g,j] P[g,b]"
_ENERGY_EQUATION = (
    "E[P]=Tr(hP)+1/2*einsum('abgd,da,gb',wbar,P,P)"
)
_LITERAL_HAMILTONIAN_EQUATION = (
    "H=sum_ij h_ij c_i^dagger c_j + "
    "1/4 sum_abgd wbar_ab;gd c_a^dagger c_b^dagger c_g c_d; "
    "action_order=d,g,create_b,create_a"
)
_DF_COLUMN_TABLE = (
    "x=e: (A_plus-A0, B_minus_plus); "
    "x=i*e: (i*(A_plus-A0), -i*B_minus_plus); "
    "y=e: (B_plus_minus, A_minus-A0); "
    "y=i*e: (-i*B_plus_minus, i*(A_minus-A0)); "
    "lower canonical v=i*e uses y=-i*e"
)
_APPROVAL_TOKEN = object()
_RECEIPT_TOKEN = object()


def _readonly_complex(value: object, *, ndim: int | None = None) -> Array:
    result = np.array(value, dtype=np.complex128, copy=True, order="C")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"expected rank-{ndim} complex array")
    result.setflags(write=False)
    return result


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
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):
        return _stable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _fingerprint(value: object) -> str:
    return sha256(
        json.dumps(_stable(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _max_abs(value: object) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return 0.0
    result = float(np.max(np.abs(array)))
    if not math.isfinite(result):
        raise ValueError("restricted scalar residual is not finite")
    return result


def _comparison_bound(expected: object) -> float:
    return (
        VITURI2024_RESTRICTED_SCALAR_ALGEBRA_ABSOLUTE_TOLERANCE
        + VITURI2024_RESTRICTED_SCALAR_ALGEBRA_RELATIVE_TOLERANCE
        * max(1.0, _max_abs(expected))
    )


def _wick_bound(left: float, right: float) -> float:
    return (
        VITURI2024_RESTRICTED_SCALAR_WICK_ABSOLUTE_TOLERANCE
        + VITURI2024_RESTRICTED_SCALAR_WICK_RELATIVE_TOLERANCE
        * max(1.0, abs(left), abs(right))
    )


def _require_entrywise(actual: object, expected: object, *, label: str) -> float:
    actual_array = np.asarray(actual, dtype=np.complex128)
    expected_array = np.asarray(expected, dtype=np.complex128)
    if actual_array.shape != expected_array.shape:
        raise ValueError(f"{label} shape mismatch")
    residual = _max_abs(actual_array - expected_array)
    bound = _comparison_bound(expected_array)
    if residual > bound:
        raise ValueError(
            f"{label} entrywise mismatch: residual={residual:.6e}, "
            f"bound={bound:.6e}"
        )
    return residual


def _source_payload_fingerprint(payload: Vituri2024HalfMetalHFReplayPayload) -> str:
    # This intentionally mirrors the readiness receipt's deterministic
    # dataclass/array projection so a live payload cannot drift after readiness.
    return _fingerprint(payload)


def _enforce_orbital_cap(number_of_orbitals: int) -> None:
    """Fail before any N^4 tensor or fixed-Ne Fock-space allocation."""

    if type(number_of_orbitals) is not int or number_of_orbitals < 1:
        raise ValueError("restricted scalar orbital count must be a positive integer")
    if number_of_orbitals > VITURI2024_RESTRICTED_SCALAR_MAX_ORBITALS:
        raise ValueError(
            "restricted scalar orbital cap exceeded before N^4/Fock allocation: "
            f"{number_of_orbitals}>{VITURI2024_RESTRICTED_SCALAR_MAX_ORBITALS}"
        )


@dataclass(frozen=True, slots=True)
class Vituri2024RestrictedScalarOrbitalCrosswalk:
    orbital_id: int
    orbital: Vituri2024Orbital
    payload_flat_orbital_index: int
    mesh_index: int
    flavor_index: int
    energy_ev: float
    occupation: int

    def __post_init__(self) -> None:
        if type(self.orbital_id) is not int or self.orbital_id < 0:
            raise ValueError("crosswalk orbital_id must be a nonnegative integer")
        if type(self.orbital) is not Vituri2024Orbital:
            raise TypeError("crosswalk orbital must be an exact Vituri2024Orbital")
        for name in ("payload_flat_orbital_index", "mesh_index", "flavor_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"crosswalk {name} must be a nonnegative integer")
        if not math.isfinite(float(self.energy_ev)):
            raise ValueError("crosswalk energy must be finite")
        if type(self.occupation) is not int or self.occupation not in (0, 1):
            raise ValueError("crosswalk occupation must be exact integer 0 or 1")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class Vituri2024RestrictedScalarVertexBinding:
    quartet: tuple[int, int, int, int]
    kinematics_fingerprint: str
    vertex_fingerprint: str
    vertex_value_ev_angstrom_squared: complex
    wbar_value_ev: complex

    def __post_init__(self) -> None:
        if (
            type(self.quartet) is not tuple
            or len(self.quartet) != 4
            or any(type(index) is not int or index < 0 for index in self.quartet)
        ):
            raise ValueError("vertex binding quartet must contain four orbital IDs")
        _validate_sha256(self.kinematics_fingerprint, label="kinematics fingerprint")
        _validate_sha256(self.vertex_fingerprint, label="vertex fingerprint")
        for name in ("vertex_value_ev_angstrom_squared", "wbar_value_ev"):
            value = complex(getattr(self, name))
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                raise ValueError(f"{name} must be finite")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class Vituri2024RestrictedScalarDFColumnEvidence:
    label: str
    input_lane: Literal["x", "y"]
    input_index: int
    coefficient: complex
    expected_column: Array
    actual_column: Array
    max_abs_residual: float

    def __post_init__(self) -> None:
        if type(self.label) is not str or not self.label:
            raise ValueError("dF evidence label must be explicit")
        if self.input_lane not in ("x", "y"):
            raise ValueError("dF evidence lane must be x or y")
        if type(self.input_index) is not int or self.input_index < 0:
            raise ValueError("dF input index must be nonnegative")
        expected = _readonly_complex(self.expected_column, ndim=1)
        actual = _readonly_complex(self.actual_column, ndim=1)
        object.__setattr__(self, "expected_column", expected)
        object.__setattr__(self, "actual_column", actual)
        residual = _max_abs(actual - expected)
        if residual != float(self.max_abs_residual):
            raise ValueError("dF evidence residual is inconsistent")
        if residual > _comparison_bound(expected):
            raise ValueError("dF physical-column evidence does not pass")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class Vituri2024RestrictedScalarResiduals:
    tensor_bra_antisymmetry: float
    tensor_ket_antisymmetry: float
    tensor_pair_hermiticity: float
    fock_counterterm_closure: float
    c9_A_plus: float
    c9_B_plus_minus: float
    c9_A_minus: float
    c9_B_minus_plus: float
    dF_physical_columns: float
    literal_hamiltonian_hermiticity: float
    double_commutator_A_plus: float
    double_commutator_B_plus_minus: float
    double_commutator_A_minus: float
    double_commutator_B_minus_plus: float
    wick_literal_p0: float
    wick_literal_all_stencils: float
    wick_energy_imaginary_part: float
    generic_stationarity: float
    generic_curvature: float
    generic_hessian_reconstruction: float

    def __post_init__(self) -> None:
        for item in fields(self):
            value = float(getattr(self, item.name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"residual {item.name} must be finite and nonnegative")
            object.__setattr__(self, item.name, value)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True, slots=True)
class _DeterministicResiduals:
    tensor_bra_antisymmetry: float
    tensor_ket_antisymmetry: float
    tensor_pair_hermiticity: float
    fock_counterterm_closure: float
    c9_A_plus: float
    c9_B_plus_minus: float
    c9_A_minus: float
    c9_B_minus_plus: float
    dF_physical_columns: float
    literal_hamiltonian_hermiticity: float
    double_commutator_A_plus: float
    double_commutator_B_plus_minus: float
    double_commutator_A_minus: float
    double_commutator_B_minus_plus: float


@dataclass(slots=True)
class _WickLiteralTracker:
    residuals: list[float] = field(default_factory=list)
    imaginary_residuals: list[float] = field(default_factory=list)
    projector_fingerprints: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _PreparedOracle:
    context: Vituri2024TDHFAssemblyContext
    orbital_crosswalk: tuple[Vituri2024RestrictedScalarOrbitalCrosswalk, ...]
    wbar: Array
    conserving_vertices: tuple[Vituri2024RestrictedScalarVertexBinding, ...]
    p0: Array
    f0: Array
    sigma_p0: Array
    h: Array
    scalar_A_plus: Array
    scalar_B_plus_minus: Array
    scalar_A_minus: Array
    scalar_B_minus_plus: Array
    dF_evidence: tuple[Vituri2024RestrictedScalarDFColumnEvidence, ...]
    literal_states: tuple[int, ...]
    literal_reference_index: int
    literal_one_body_hamiltonian: Array
    literal_interaction_hamiltonian: Array
    literal_hamiltonian: Array
    tangent_basis: TDHFTransitionTangentBasis
    callback: Callable[[Array], float]
    tracker: _WickLiteralTracker
    functional_manifest: TDHFScalarFunctionalManifest
    generic_approval: TDHFScalarCurvatureApproval
    deterministic_residuals: _DeterministicResiduals
    deterministic_manifest_sha256: str


def _validate_inputs(
    readiness: Vituri2024TDHFScalarReadinessReceipt,
    assembly: Vituri2024TDHFSignedQAssemblyReceipt,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
) -> tuple[str, str, str]:
    if type(readiness) is not Vituri2024TDHFScalarReadinessReceipt:
        raise TypeError("restricted scalar requires exact factory readiness")
    if type(assembly) is not Vituri2024TDHFSignedQAssemblyReceipt:
        raise TypeError("restricted scalar requires exact Vituri signed-q assembly")
    if type(source_payload) is not Vituri2024HalfMetalHFReplayPayload:
        raise TypeError("restricted scalar requires exact Vituri replay payload")

    # The cap is intentionally checked before any path below can allocate the
    # N^4 tensor or fixed-Ne Fock matrices.
    _enforce_orbital_cap(len(assembly.orbital_id_map))

    readiness_fingerprint = readiness.fingerprint
    assembly_fingerprint = assembly.fingerprint
    payload_fingerprint = _source_payload_fingerprint(source_payload)
    checks = (
        (
            readiness.assembly_receipt_fingerprint,
            assembly_fingerprint,
            "readiness/assembly fingerprint",
        ),
        (
            readiness.source_payload_fingerprint,
            payload_fingerprint,
            "readiness/source payload fingerprint",
        ),
        (
            readiness.source_fingerprint,
            assembly.source_fingerprint,
            "readiness/assembly source",
        ),
        (
            readiness.assembly_context_fingerprint,
            assembly.assembly_context_fingerprint,
            "readiness/assembly context",
        ),
        (
            readiness.area_angstrom_squared,
            assembly.signed_pair.plus_context.area.area_angstrom_squared,
            "readiness/context area",
        ),
        (
            readiness.delta1_ev,
            assembly.signed_pair.plus_context.delta1_ev,
            "readiness/context Delta1",
        ),
        (
            readiness.interaction_receipt_fingerprint,
            assembly.signed_pair.plus_context.interaction_receipt_fingerprint,
            "readiness/context interaction",
        ),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"restricted scalar {label} drift")
    authority_lock = (
        readiness.predecessor_chain_bound is True,
        readiness.payload_transition_binding_verified is True,
        readiness.assembly_context_bound is True,
        readiness.projected_signed_q_structure_verified is True,
        readiness.scalar_curvature_executed is False,
        readiness.mathematical_scalar_curvature_match is False,
        readiness.static_hessian_authority_promoted is False,
        readiness.promotion_eligible is False,
        readiness.original_sector_authority == "projected_signed_ab",
        readiness.static_hessian_authority == "not_established",
        assembly.static_hessian_authority == "projected_signed_ab",
        assembly.sector.static_hessian_authority == "projected_signed_ab",
    )
    if not all(authority_lock):
        raise ValueError("restricted scalar readiness/original authority drift")
    if assembly.signed_pair.plus_context != assembly.signed_pair.minus_context:
        raise ValueError("restricted scalar requires one exact shared assembly context")
    return readiness_fingerprint, assembly_fingerprint, payload_fingerprint


def _build_crosswalk(
    readiness: Vituri2024TDHFScalarReadinessReceipt,
    assembly: Vituri2024TDHFSignedQAssemblyReceipt,
    payload: Vituri2024HalfMetalHFReplayPayload,
) -> tuple[Vituri2024RestrictedScalarOrbitalCrosswalk, ...]:
    orbital_map = assembly.orbital_id_map
    expected_ids = tuple(range(len(orbital_map)))
    if tuple(index for index, _ in orbital_map) != expected_ids:
        raise ValueError("assembly.orbital_id_map is not exact contiguous ID order")
    by_id: dict[int, Vituri2024RestrictedScalarOrbitalCrosswalk] = {}
    binding_index = 0
    lane_specs = (
        (
            assembly.signed_pair.plus_inventory.transitions,
            assembly.blocks.plus_pairs,
        ),
        (
            assembly.signed_pair.minus_inventory.transitions,
            assembly.blocks.minus_pairs,
        ),
    )
    nk = int(payload.mesh.shape[0])
    for transitions, pairs in lane_specs:
        for transition, pair in zip(transitions, pairs):
            binding = readiness.transition_source_bindings[binding_index]
            binding_index += 1
            records = (
                (
                    pair.particle,
                    transition.particle,
                    binding.particle_flat_orbital_index,
                    binding.particle_mesh_index,
                    binding.particle_flavor_index,
                    binding.particle_energy_ev,
                    binding.particle_occupation,
                ),
                (
                    pair.hole,
                    transition.hole,
                    binding.hole_flat_orbital_index,
                    binding.hole_mesh_index,
                    binding.hole_flavor_index,
                    binding.hole_energy_ev,
                    binding.hole_occupation,
                ),
            )
            for record in records:
                orbital_id, orbital, flat, mesh, flavor, energy, occupation = record
                if flat != flavor * nk + mesh:
                    raise ValueError("payload flat-orbital crosswalk drift")
                if payload.energies[flavor, mesh] != energy:
                    raise ValueError("payload energy drift in orbital crosswalk")
                if int(payload.occupations[flavor, mesh]) != occupation:
                    raise ValueError("payload occupation drift in orbital crosswalk")
                candidate = Vituri2024RestrictedScalarOrbitalCrosswalk(
                    orbital_id=orbital_id,
                    orbital=orbital,
                    payload_flat_orbital_index=flat,
                    mesh_index=mesh,
                    flavor_index=flavor,
                    energy_ev=energy,
                    occupation=occupation,
                )
                previous = by_id.get(orbital_id)
                if previous is not None and previous != candidate:
                    raise ValueError("inconsistent duplicate orbital crosswalk")
                by_id[orbital_id] = candidate
    if binding_index != len(readiness.transition_source_bindings):
        raise ValueError("transition-source binding count drift")
    if tuple(sorted(by_id)) != expected_ids:
        raise ValueError("orbital crosswalk does not cover assembly.orbital_id_map")
    result = tuple(by_id[index] for index in expected_ids)
    if tuple((item.orbital_id, item.orbital) for item in result) != orbital_map:
        raise ValueError("orbital crosswalk order differs from assembly.orbital_id_map")
    return result


def _exact_local_conserving(
    alpha: Vituri2024Orbital,
    beta: Vituri2024Orbital,
    gamma: Vituri2024Orbital,
    delta: Vituri2024Orbital,
) -> bool:
    momenta = tuple(
        item.momentum_inverse_angstrom for item in (alpha, beta, gamma, delta)
    )
    residual = (
        momenta[0][0] + momenta[1][0] - momenta[2][0] - momenta[3][0],
        momenta[0][1] + momenta[1][1] - momenta[2][1] - momenta[3][1],
    )
    return residual == (0.0, 0.0)


def _build_raw_tensor(
    context: Vituri2024TDHFAssemblyContext,
    crosswalk: tuple[Vituri2024RestrictedScalarOrbitalCrosswalk, ...],
) -> tuple[Array, tuple[Vituri2024RestrictedScalarVertexBinding, ...]]:
    number = len(crosswalk)
    _enforce_orbital_cap(number)
    # This is the first N^4 allocation in the oracle.
    wbar = np.zeros((number, number, number, number), dtype=np.complex128)
    bindings: list[Vituri2024RestrictedScalarVertexBinding] = []
    orbitals = tuple(item.orbital for item in crosswalk)
    for alpha, beta, gamma, delta in product(range(number), repeat=4):
        quartet = (
            orbitals[alpha],
            orbitals[beta],
            orbitals[gamma],
            orbitals[delta],
        )
        if not _exact_local_conserving(*quartet):
            # Exact zero: no vertex call, no diagnostic-tolerance inclusion.
            continue
        kinematics = Vituri2024FourPointKinematicsReceipt(
            alpha=quartet[0],
            beta=quartet[1],
            gamma=quartet[2],
            delta=quartet[3],
            momentum_tolerance_inverse_angstrom=0.0,
            provider_sha256=context.kinematics_provider_sha256,
            derivation_source_sm_sha256=SM_TEX_SHA256,
            source_text=context.kinematics_source_text,
        )
        kinematics.require_conserving()
        vertex = vituri2024_antisymmetrized_projected_vertex(
            kinematics, context.Delta1, context.interaction
        )
        value = vertex.value_ev_angstrom_squared / context.area.area_angstrom_squared
        wbar[alpha, beta, gamma, delta] = value
        bindings.append(
            Vituri2024RestrictedScalarVertexBinding(
                quartet=(alpha, beta, gamma, delta),
                kinematics_fingerprint=kinematics.fingerprint,
                vertex_fingerprint=vertex.fingerprint,
                vertex_value_ev_angstrom_squared=vertex.value_ev_angstrom_squared,
                wbar_value_ev=value,
            )
        )
    return _readonly_complex(wbar, ndim=4), tuple(bindings)


def _validate_raw_tensor_symmetries(wbar: Array) -> tuple[float, float, float]:
    bra = _max_abs(wbar + np.swapaxes(wbar, 0, 1))
    ket = _max_abs(wbar + np.swapaxes(wbar, 2, 3))
    pair = _max_abs(wbar - wbar.transpose(3, 2, 1, 0).conj())
    bound = _comparison_bound(wbar)
    failures = tuple(
        (name, value)
        for name, value in (
            ("bra antisymmetry", bra),
            ("ket antisymmetry", ket),
            ("pair Hermiticity", pair),
        )
        if value > bound
    )
    if failures:
        details = ", ".join(f"{name}={value:.6e}" for name, value in failures)
        raise ValueError(
            "raw Vituri wbar tensor symmetry failure; tensor repair is prohibited: "
            + details
        )
    return bra, ket, pair


def _sigma(wbar: Array, density: Array) -> Array:
    return np.einsum("ibgj,gb->ij", wbar, density, optimize=False)


def _wick_energy(h: Array, wbar: Array, density: Array) -> complex:
    return complex(
        np.trace(h @ density)
        + 0.5
        * np.einsum(
            "abgd,da,gb", wbar, density, density, optimize=False
        )
    )


def _expected_scalar_blocks(
    assembly: Vituri2024TDHFSignedQAssemblyReceipt,
    f0: Array,
    wbar: Array,
) -> tuple[Array, Array, Array, Array]:
    def a_block(pairs: tuple[object, ...]) -> Array:
        result = np.zeros((len(pairs), len(pairs)), dtype=np.complex128)
        for row, left in enumerate(pairs):
            for column, right in enumerate(pairs):
                a, A = left.particle, left.hole  # type: ignore[attr-defined]
                b, B = right.particle, right.hole  # type: ignore[attr-defined]
                gap = (
                    f0[a, a] - f0[A, A]
                    if a == b and A == B
                    else 0.0j
                )
                result[row, column] = gap - wbar[a, B, A, b]
        return _readonly_complex(result, ndim=2)

    def b_block(rows: tuple[object, ...], columns: tuple[object, ...]) -> Array:
        result = np.zeros((len(rows), len(columns)), dtype=np.complex128)
        for row, left in enumerate(rows):
            for column, right in enumerate(columns):
                a, A = left.particle, left.hole  # type: ignore[attr-defined]
                b, B = right.particle, right.hole  # type: ignore[attr-defined]
                result[row, column] = -wbar[a, b, A, B]
        return _readonly_complex(result, ndim=2)

    plus = assembly.blocks.plus_pairs
    minus = assembly.blocks.minus_pairs
    return (
        a_block(plus),
        b_block(plus, minus),
        a_block(minus),
        b_block(minus, plus),
    )


def _gap_matrix(pairs: tuple[object, ...], f0: Array) -> Array:
    result = np.zeros((len(pairs), len(pairs)), dtype=np.complex128)
    for index, pair in enumerate(pairs):
        result[index, index] = (
            f0[pair.particle, pair.particle]  # type: ignore[attr-defined]
            - f0[pair.hole, pair.hole]  # type: ignore[attr-defined]
        )
    return result


def _extract_ph_column(matrix: Array, plus: tuple[object, ...], minus: tuple[object, ...]) -> Array:
    return np.asarray(
        [
            matrix[pair.particle, pair.hole]  # type: ignore[attr-defined]
            for pair in plus + minus
        ],
        dtype=np.complex128,
    )


def _build_dF_evidence(
    assembly: Vituri2024TDHFSignedQAssemblyReceipt,
    wbar: Array,
    f0: Array,
    scalar_blocks: tuple[Array, Array, Array, Array],
) -> tuple[Vituri2024RestrictedScalarDFColumnEvidence, ...]:
    plus = assembly.blocks.plus_pairs
    minus = assembly.blocks.minus_pairs
    A_plus, B_plus_minus, A_minus, B_minus_plus = scalar_blocks
    A_plus_interaction = A_plus - _gap_matrix(plus, f0)
    A_minus_interaction = A_minus - _gap_matrix(minus, f0)
    number = wbar.shape[0]
    evidence: list[Vituri2024RestrictedScalarDFColumnEvidence] = []

    def append(
        *,
        label: str,
        lane: Literal["x", "y"],
        index: int,
        coefficient: complex,
        pair: object,
        expected: Array,
    ) -> None:
        density = np.zeros((number, number), dtype=np.complex128)
        particle = pair.particle  # type: ignore[attr-defined]
        hole = pair.hole  # type: ignore[attr-defined]
        density[particle, hole] = coefficient
        density[hole, particle] = coefficient.conjugate()
        actual = _extract_ph_column(_sigma(wbar, density), plus, minus)
        residual = _require_entrywise(actual, expected, label=f"dF {label}")
        evidence.append(
            Vituri2024RestrictedScalarDFColumnEvidence(
                label=label,
                input_lane=lane,
                input_index=index,
                coefficient=coefficient,
                expected_column=expected,
                actual_column=actual,
                max_abs_residual=residual,
            )
        )

    for index, pair in enumerate(plus):
        base = np.concatenate(
            (A_plus_interaction[:, index], B_minus_plus[:, index])
        )
        append(
            label=f"x.real[{index}]",
            lane="x",
            index=index,
            coefficient=1.0 + 0.0j,
            pair=pair,
            expected=base,
        )
        imag_expected = np.concatenate(
            (1.0j * A_plus_interaction[:, index], -1.0j * B_minus_plus[:, index])
        )
        append(
            label=f"x.imag[{index}]",
            lane="x",
            index=index,
            coefficient=1.0j,
            pair=pair,
            expected=imag_expected,
        )
    for index, pair in enumerate(minus):
        real_expected = np.concatenate(
            (B_plus_minus[:, index], A_minus_interaction[:, index])
        )
        imag_expected = np.concatenate(
            (-1.0j * B_plus_minus[:, index], 1.0j * A_minus_interaction[:, index])
        )
        append(
            label=f"y.real[{index}]",
            lane="y",
            index=index,
            coefficient=1.0 + 0.0j,
            pair=pair,
            expected=real_expected,
        )
        append(
            label=f"y.imag[{index}]",
            lane="y",
            index=index,
            coefficient=1.0j,
            pair=pair,
            expected=imag_expected,
        )
        # The generic signed-Hessian coordinate is v=(x,y*).  Its canonical
        # lower imaginary vector v_lower=+i therefore means physical y=-i.
        canonical_expected = np.concatenate(
            (1.0j * B_plus_minus[:, index], -1.0j * A_minus_interaction[:, index])
        )
        append(
            label=f"lower_canonical_v.imag[{index}]",
            lane="y",
            index=index,
            coefficient=-1.0j,
            pair=pair,
            expected=canonical_expected,
        )
    return tuple(evidence)


def _annihilate(state: int, orbital: int) -> tuple[int, int] | None:
    if not state & (1 << orbital):
        return None
    sign = -1 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return sign, state ^ (1 << orbital)


def _create(state: int, orbital: int) -> tuple[int, int] | None:
    if state & (1 << orbital):
        return None
    sign = -1 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return sign, state | (1 << orbital)


def _one_body_operator(
    states: tuple[int, ...], particle: int, hole: int
) -> Array:
    state_index = {state: index for index, state in enumerate(states)}
    result = np.zeros((len(states), len(states)), dtype=np.complex128)
    for column, state in enumerate(states):
        first = _annihilate(state, hole)
        if first is None:
            continue
        sign1, intermediate = first
        second = _create(intermediate, particle)
        if second is None:
            continue
        sign2, final = second
        row = state_index.get(final)
        if row is not None:
            result[row, column] = sign1 * sign2
    return result


def _quartic_operator(
    states: tuple[int, ...], alpha: int, beta: int, gamma: int, delta: int
) -> Array:
    """Literal c_a^dagger c_b^dagger c_g c_d action: d,g,b-create,a-create."""

    state_index = {state: index for index, state in enumerate(states)}
    result = np.zeros((len(states), len(states)), dtype=np.complex128)
    for column, state in enumerate(states):
        action1 = _annihilate(state, delta)
        if action1 is None:
            continue
        sign1, state1 = action1
        action2 = _annihilate(state1, gamma)
        if action2 is None:
            continue
        sign2, state2 = action2
        action3 = _create(state2, beta)
        if action3 is None:
            continue
        sign3, state3 = action3
        action4 = _create(state3, alpha)
        if action4 is None:
            continue
        sign4, final = action4
        row = state_index.get(final)
        if row is not None:
            result[row, column] = sign1 * sign2 * sign3 * sign4
    return result


def _build_literal_hamiltonian(
    h: Array, wbar: Array, p0: Array
) -> tuple[tuple[int, ...], int, Array, Array, Array]:
    number = h.shape[0]
    _enforce_orbital_cap(number)
    occupations = tuple(int(round(value.real)) for value in np.diag(p0))
    number_particles = sum(occupations)
    if not 0 < number_particles < number:
        raise ValueError("restricted literal oracle requires 0<Ne<N orbitals")
    states = tuple(
        sum(1 << orbital for orbital in occupied)
        for occupied in combinations(range(number), number_particles)
    )
    # The fixed-Ne Fock allocation occurs only after the cap above.
    one_body = np.zeros((len(states), len(states)), dtype=np.complex128)
    interaction = np.zeros_like(one_body)
    one_body_operators = {
        (i, j): _one_body_operator(states, i, j)
        for i in range(number)
        for j in range(number)
    }
    for i, j in product(range(number), repeat=2):
        if h[i, j] != 0.0j:
            one_body += h[i, j] * one_body_operators[i, j]
    for alpha, beta, gamma, delta in product(range(number), repeat=4):
        coefficient = wbar[alpha, beta, gamma, delta]
        if coefficient != 0.0j:
            interaction += 0.25 * coefficient * _quartic_operator(
                states, alpha, beta, gamma, delta
            )
    reference_state = sum(
        1 << index for index, occupation in enumerate(occupations) if occupation
    )
    try:
        reference_index = states.index(reference_state)
    except ValueError as error:
        raise ValueError("P0 occupation bitstring is absent from fixed-Ne basis") from error
    return (
        states,
        reference_index,
        _readonly_complex(one_body, ndim=2),
        _readonly_complex(interaction, ndim=2),
        _readonly_complex(one_body + interaction, ndim=2),
    )


def _commutator(left: Array, right: Array) -> Array:
    return left @ right - right @ left


def _literal_double_commutator_blocks(
    assembly: Vituri2024TDHFSignedQAssemblyReceipt,
    states: tuple[int, ...],
    reference_index: int,
    hamiltonian: Array,
) -> tuple[Array, Array, Array, Array]:
    def operators(pairs: tuple[object, ...]) -> tuple[Array, ...]:
        return tuple(
            _one_body_operator(states, pair.particle, pair.hole)  # type: ignore[attr-defined]
            for pair in pairs
        )

    plus = operators(assembly.blocks.plus_pairs)
    minus = operators(assembly.blocks.minus_pairs)

    def expectation(operator: Array) -> complex:
        return complex(operator[reference_index, reference_index])

    def a_block(items: tuple[Array, ...]) -> Array:
        return _readonly_complex(
            [
                [
                    expectation(
                        _commutator(
                            left.conj().T,
                            _commutator(hamiltonian, right),
                        )
                    )
                    for right in items
                ]
                for left in items
            ],
            ndim=2,
        )

    def b_block(rows: tuple[Array, ...], columns: tuple[Array, ...]) -> Array:
        return _readonly_complex(
            [
                [
                    expectation(
                        _commutator(_commutator(hamiltonian, left), right)
                    ).conjugate()
                    for right in columns
                ]
                for left in rows
            ],
            ndim=2,
        )

    return a_block(plus), b_block(plus, minus), a_block(minus), b_block(minus, plus)


def _slater_vector(
    projector: Array, states: tuple[int, ...], number_particles: int
) -> Array:
    hermiticity = _max_abs(projector - projector.conj().T)
    idempotency = _max_abs(projector @ projector - projector)
    trace_residual = abs(complex(np.trace(projector)) - number_particles)
    bound = _comparison_bound(projector)
    if max(hermiticity, idempotency, trace_residual) > bound:
        raise ValueError("literal Slater expectation requires an exact projector")
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    occupied = eigenvectors[:, np.argsort(eigenvalues)[-number_particles:]]
    amplitudes = np.asarray(
        [
            np.linalg.det(
                occupied[
                    [index for index in range(projector.shape[0]) if state & (1 << index)],
                    :,
                ]
            )
            for state in states
        ],
        dtype=np.complex128,
    )
    norm = float(np.linalg.norm(amplitudes))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("literal Slater determinant amplitudes have invalid norm")
    return amplitudes / norm


def _literal_slater_energy(
    projector: Array,
    states: tuple[int, ...],
    number_particles: int,
    hamiltonian: Array,
) -> float:
    state = _slater_vector(projector, states, number_particles)
    value = complex(np.vdot(state, hamiltonian @ state))
    imaginary = abs(value.imag)
    if imaginary > _wick_bound(value.real, 0.0):
        raise ValueError("literal Slater energy is not real")
    return float(value.real)


def _make_energy_callback(
    h: Array,
    wbar: Array,
    states: tuple[int, ...],
    number_particles: int,
    literal_hamiltonian: Array,
) -> tuple[Callable[[Array], float], _WickLiteralTracker]:
    tracker = _WickLiteralTracker()

    def restricted_wick_literal_energy(projector: Array) -> float:
        wick = _wick_energy(h, wbar, projector)
        literal = _literal_slater_energy(
            projector, states, number_particles, literal_hamiltonian
        )
        residual = abs(wick.real - literal)
        imaginary = abs(wick.imag)
        if residual > _wick_bound(wick.real, literal):
            raise ValueError(
                "Wick/literal Slater scalar mismatch: "
                f"residual={residual:.6e}"
            )
        if imaginary > _wick_bound(wick.real, literal):
            raise ValueError("Wick scalar has a non-real component")
        tracker.residuals.append(residual)
        tracker.imaginary_residuals.append(imaginary)
        tracker.projector_fingerprints.append(fingerprint_tdhf_matrix(projector))
        return float(wick.real)

    return restricted_wick_literal_energy, tracker


def _build_tangent_basis(
    assembly: Vituri2024TDHFSignedQAssemblyReceipt, p0: Array
) -> TDHFTransitionTangentBasis:
    number = p0.shape[0]

    def tangent(pair: object) -> Array:
        result = np.zeros((number, number), dtype=np.complex128)
        result[pair.particle, pair.hole] = 1.0  # type: ignore[attr-defined]
        return result

    return TDHFTransitionTangentBasis(
        source_projector=p0,
        plus_tangents=tuple(tangent(pair) for pair in assembly.blocks.plus_pairs),
        minus_tangents=tuple(tangent(pair) for pair in assembly.blocks.minus_pairs),
        source_fingerprint=assembly.sector.source_fingerprint,
        plus_pairs_fingerprint=fingerprint_tdhf_pairs(assembly.blocks.plus_pairs),
        minus_pairs_fingerprint=fingerprint_tdhf_pairs(assembly.blocks.minus_pairs),
    )


def _scalar_step_ladder() -> TDHFScalarCurvatureStepLadder:
    return TDHFScalarCurvatureStepLadder(
        steps=(2.0e-2, 1.0e-2, 5.0e-3),
        tolerances=TDHFScalarCurvatureTolerances(
            stationarity_absolute=3.0e-8,
            stationarity_relative=1.0e-10,
            curvature_absolute=2.0e-8,
            curvature_relative=3.0e-7,
            roundoff_multiplier=256.0,
            projector_tolerance=8.0e-11,
            matrix_absolute=1.0e-7,
            matrix_relative=1.0e-6,
        ),
        registration_label="actual-vituri-restricted-finite-orbital-v1",
    )


def _prepare_oracle(
    readiness: Vituri2024TDHFScalarReadinessReceipt,
    assembly: Vituri2024TDHFSignedQAssemblyReceipt,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
    *,
    provenance: str,
) -> _PreparedOracle:
    readiness_fingerprint, assembly_fingerprint, payload_fingerprint = (
        _validate_inputs(readiness, assembly, source_payload)
    )
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("restricted scalar detached approval provenance is required")
    context = assembly.signed_pair.plus_context
    crosswalk = _build_crosswalk(readiness, assembly, source_payload)
    wbar, conserving_vertices = _build_raw_tensor(context, crosswalk)
    tensor_bra, tensor_ket, tensor_pair = _validate_raw_tensor_symmetries(wbar)

    occupations = np.asarray([item.occupation for item in crosswalk], dtype=float)
    energies = np.asarray([item.energy_ev for item in crosswalk], dtype=float)
    p0 = _readonly_complex(np.diag(occupations), ndim=2)
    f0 = _readonly_complex(np.diag(energies), ndim=2)
    sigma_p0 = _readonly_complex(_sigma(wbar, p0), ndim=2)
    h = _readonly_complex(f0 - sigma_p0, ndim=2)
    fock_counterterm = _max_abs(h + _sigma(wbar, p0) - f0)
    if fock_counterterm > _comparison_bound(f0):
        raise ValueError("h=F0-Sigma[P0] counterterm closure failed")

    scalar_blocks = _expected_scalar_blocks(assembly, f0, wbar)
    A_plus, B_plus_minus, A_minus, B_minus_plus = scalar_blocks
    c9_residuals = (
        _require_entrywise(
            assembly.blocks.A_plus, A_plus, label="actual C9 A_plus"
        ),
        _require_entrywise(
            assembly.blocks.B_plus_minus,
            B_plus_minus,
            label="actual C9 B_plus_minus",
        ),
        _require_entrywise(
            assembly.blocks.A_minus, A_minus, label="actual C9 A_minus"
        ),
        _require_entrywise(
            assembly.blocks.B_minus_plus,
            B_minus_plus,
            label="actual C9 B_minus_plus",
        ),
    )
    dF_evidence = _build_dF_evidence(assembly, wbar, f0, scalar_blocks)
    dF_residual = max(item.max_abs_residual for item in dF_evidence)

    (
        literal_states,
        literal_reference_index,
        literal_one_body,
        literal_interaction,
        literal_hamiltonian,
    ) = _build_literal_hamiltonian(h, wbar, p0)
    literal_hermiticity = _max_abs(
        literal_hamiltonian - literal_hamiltonian.conj().T
    )
    if literal_hermiticity > _comparison_bound(literal_hamiltonian):
        raise ValueError("literal fixed-Ne Hamiltonian is not Hermitian")
    double_blocks = _literal_double_commutator_blocks(
        assembly,
        literal_states,
        literal_reference_index,
        literal_hamiltonian,
    )
    double_residuals = tuple(
        _require_entrywise(actual, expected, label=f"literal double commutator {label}")
        for actual, expected, label in zip(
            double_blocks,
            (
                assembly.blocks.A_plus,
                assembly.blocks.B_plus_minus,
                assembly.blocks.A_minus,
                assembly.blocks.B_minus_plus,
            ),
            ("A_plus", "B_plus_minus", "A_minus", "B_minus_plus"),
        )
    )

    tangent_basis = _build_tangent_basis(assembly, p0)
    callback, tracker = _make_energy_callback(
        h,
        wbar,
        literal_states,
        int(np.trace(p0).real),
        literal_hamiltonian,
    )
    immutable_input_fingerprint = _fingerprint(
        {
            "wbar": _array_sha256(wbar),
            "p0": _array_sha256(p0),
            "f0": _array_sha256(f0),
            "sigma_p0": _array_sha256(sigma_p0),
            "h": _array_sha256(h),
            "literal_states": literal_states,
            "literal_hamiltonian": _array_sha256(literal_hamiltonian),
            "sigma_equation": _SIGMA_EQUATION,
            "energy_equation": _ENERGY_EQUATION,
            "literal_equation": _LITERAL_HAMILTONIAN_EQUATION,
        }
    )
    source_functional_fingerprint = _fingerprint(
        {
            "scope": VITURI2024_RESTRICTED_SCALAR_AUTHORITY,
            "readiness": readiness_fingerprint,
            "assembly": assembly_fingerprint,
            "payload": payload_fingerprint,
            "actual_vertex": True,
            "actual_C9": True,
        }
    )
    functional_manifest = make_tdhf_scalar_functional_manifest(
        energy_callback=callback,
        source_functional_fingerprint=source_functional_fingerprint,
        immutable_callback_input_fingerprint=immutable_input_fingerprint,
        provenance=(
            "System-local Wick scalar and independent fixed-Ne literal Hamiltonian; "
            "restricted to exact assembly.orbital_id_map."
        ),
    )
    directions = canonical_tdhf_scalar_directions(
        len(tangent_basis.plus_tangents), len(tangent_basis.minus_tangents)
    )
    generic_approval = make_tdhf_scalar_curvature_approval(
        sector=assembly.sector,
        tangent_basis=tangent_basis,
        directions=directions,
        energy_callback=callback,
        functional_manifest=functional_manifest,
        energy_convention=TDHFEnergyConvention(
            normalization="total",
            denominator=1.0,
            energy_units="eV_on_restricted_finite_orbital_space",
            curvature_units="eV_per_dimensionless_unitary_angle_squared",
            denominator_source="literal total finite-orbital scalar; no reporting division",
        ),
        step_ladder=_scalar_step_ladder(),
        interaction_fingerprint=assembly.sector.interaction_fingerprint,
        provenance=provenance,
    )
    deterministic_residuals = _DeterministicResiduals(
        tensor_bra_antisymmetry=tensor_bra,
        tensor_ket_antisymmetry=tensor_ket,
        tensor_pair_hermiticity=tensor_pair,
        fock_counterterm_closure=fock_counterterm,
        c9_A_plus=c9_residuals[0],
        c9_B_plus_minus=c9_residuals[1],
        c9_A_minus=c9_residuals[2],
        c9_B_minus_plus=c9_residuals[3],
        dF_physical_columns=dF_residual,
        literal_hamiltonian_hermiticity=literal_hermiticity,
        double_commutator_A_plus=double_residuals[0],
        double_commutator_B_plus_minus=double_residuals[1],
        double_commutator_A_minus=double_residuals[2],
        double_commutator_B_minus_plus=double_residuals[3],
    )
    deterministic_manifest_sha256 = _fingerprint(
        {
            "readiness_fingerprint": readiness_fingerprint,
            "assembly_fingerprint": assembly_fingerprint,
            "payload_fingerprint": payload_fingerprint,
            "context_fingerprint": context.fingerprint,
            "orbital_id_map": assembly.orbital_id_map,
            "crosswalk": tuple(item.fingerprint for item in crosswalk),
            "wbar": _array_sha256(wbar),
            "conserving_vertices": tuple(
                item.fingerprint for item in conserving_vertices
            ),
            "p0": _array_sha256(p0),
            "f0": _array_sha256(f0),
            "sigma_p0": _array_sha256(sigma_p0),
            "h": _array_sha256(h),
            "scalar_blocks": tuple(_array_sha256(item) for item in scalar_blocks),
            "dF_evidence": tuple(item.fingerprint for item in dF_evidence),
            "literal_states": literal_states,
            "literal_reference_index": literal_reference_index,
            "literal_one_body": _array_sha256(literal_one_body),
            "literal_interaction": _array_sha256(literal_interaction),
            "literal_hamiltonian": _array_sha256(literal_hamiltonian),
            "tangent_basis": tangent_basis.fingerprint,
            "functional_manifest": functional_manifest.fingerprint,
            "generic_approval": generic_approval.fingerprint,
            "deterministic_residuals": asdict(deterministic_residuals),
            "authority": VITURI2024_RESTRICTED_SCALAR_AUTHORITY,
        }
    )
    if tracker.residuals or tracker.projector_fingerprints:
        raise RuntimeError("approval preparation invoked scalar energy unexpectedly")
    return _PreparedOracle(
        context=context,
        orbital_crosswalk=crosswalk,
        wbar=wbar,
        conserving_vertices=conserving_vertices,
        p0=p0,
        f0=f0,
        sigma_p0=sigma_p0,
        h=h,
        scalar_A_plus=A_plus,
        scalar_B_plus_minus=B_plus_minus,
        scalar_A_minus=A_minus,
        scalar_B_minus_plus=B_minus_plus,
        dF_evidence=dF_evidence,
        literal_states=literal_states,
        literal_reference_index=literal_reference_index,
        literal_one_body_hamiltonian=literal_one_body,
        literal_interaction_hamiltonian=literal_interaction,
        literal_hamiltonian=literal_hamiltonian,
        tangent_basis=tangent_basis,
        callback=callback,
        tracker=tracker,
        functional_manifest=functional_manifest,
        generic_approval=generic_approval,
        deterministic_residuals=deterministic_residuals,
        deterministic_manifest_sha256=deterministic_manifest_sha256,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFRestrictedScalarApproval:
    """Detached system approval; construction performs no scalar-energy call."""

    _factory_token: InitVar[object]
    api_version: str
    readiness: Vituri2024TDHFScalarReadinessReceipt
    assembly_receipt: Vituri2024TDHFSignedQAssemblyReceipt
    source_payload: Vituri2024HalfMetalHFReplayPayload
    readiness_fingerprint: str
    assembly_receipt_fingerprint: str
    source_payload_fingerprint: str
    context_fingerprint: str
    orbital_id_map: tuple[tuple[int, Vituri2024Orbital], ...]
    deterministic_manifest_sha256: str
    generic_approval: TDHFScalarCurvatureApproval
    generic_approval_fingerprint: str
    provenance: str
    scalar_energy_evaluated: bool
    authority: str
    approval_fingerprint: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _APPROVAL_TOKEN:
            raise TypeError("restricted scalar approval requires the private factory token")
        self._validate_consistency(check_fingerprint=False)
        object.__setattr__(self, "approval_fingerprint", self._expected_fingerprint())

    def _expected_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": self.api_version,
                "readiness_fingerprint": self.readiness_fingerprint,
                "assembly_receipt_fingerprint": self.assembly_receipt_fingerprint,
                "source_payload_fingerprint": self.source_payload_fingerprint,
                "context_fingerprint": self.context_fingerprint,
                "orbital_id_map": self.orbital_id_map,
                "deterministic_manifest_sha256": self.deterministic_manifest_sha256,
                "generic_approval_fingerprint": self.generic_approval_fingerprint,
                "provenance": self.provenance,
                "scalar_energy_evaluated": self.scalar_energy_evaluated,
                "authority": self.authority,
            }
        )

    def _validate_consistency(self, *, check_fingerprint: bool = True) -> None:
        if self.api_version != "vituri2024_tdhf_restricted_scalar_approval.v1":
            raise ValueError("restricted scalar approval API version drift")
        live_readiness, live_assembly, live_payload = _validate_inputs(
            self.readiness, self.assembly_receipt, self.source_payload
        )
        checks = (
            (self.readiness_fingerprint, live_readiness, "readiness"),
            (self.assembly_receipt_fingerprint, live_assembly, "assembly"),
            (self.source_payload_fingerprint, live_payload, "source payload"),
            (
                self.context_fingerprint,
                self.assembly_receipt.signed_pair.plus_context.fingerprint,
                "context",
            ),
            (self.orbital_id_map, self.assembly_receipt.orbital_id_map, "orbital map"),
            (
                self.generic_approval_fingerprint,
                self.generic_approval.fingerprint,
                "generic approval",
            ),
        )
        for actual, expected, label in checks:
            if actual != expected:
                raise ValueError(f"stale restricted scalar approval: {label} mismatch")
        _validate_sha256(
            self.deterministic_manifest_sha256,
            label="deterministic approval manifest",
        )
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("restricted scalar approval provenance is missing")
        if (
            self.scalar_energy_evaluated is not False
            or self.authority != VITURI2024_RESTRICTED_SCALAR_AUTHORITY
        ):
            raise ValueError("restricted scalar approval phase/authority drift")
        if check_fingerprint and self.approval_fingerprint != self._expected_fingerprint():
            raise ValueError("stale restricted scalar approval manifest")

    @property
    def fingerprint(self) -> str:
        self._validate_consistency()
        return self.approval_fingerprint


def make_vituri2024_tdhf_restricted_scalar_approval(
    *,
    readiness: Vituri2024TDHFScalarReadinessReceipt,
    assembly_receipt: Vituri2024TDHFSignedQAssemblyReceipt,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
    provenance: str,
) -> Vituri2024TDHFRestrictedScalarApproval:
    """Construct and hash deterministic inputs without evaluating scalar E[P]."""

    prepared = _prepare_oracle(
        readiness,
        assembly_receipt,
        source_payload,
        provenance=provenance,
    )
    return Vituri2024TDHFRestrictedScalarApproval(
        _factory_token=_APPROVAL_TOKEN,
        api_version="vituri2024_tdhf_restricted_scalar_approval.v1",
        readiness=readiness,
        assembly_receipt=assembly_receipt,
        source_payload=source_payload,
        readiness_fingerprint=readiness.fingerprint,
        assembly_receipt_fingerprint=assembly_receipt.fingerprint,
        source_payload_fingerprint=_source_payload_fingerprint(source_payload),
        context_fingerprint=prepared.context.fingerprint,
        orbital_id_map=assembly_receipt.orbital_id_map,
        deterministic_manifest_sha256=prepared.deterministic_manifest_sha256,
        generic_approval=prepared.generic_approval,
        generic_approval_fingerprint=prepared.generic_approval.fingerprint,
        provenance=provenance,
        scalar_energy_evaluated=False,
        authority=VITURI2024_RESTRICTED_SCALAR_AUTHORITY,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFRestrictedScalarReceipt:
    """Factory-only finite-orbital algebra receipt with permanent narrow scope."""

    _factory_token: InitVar[object]
    api_version: str
    approval: Vituri2024TDHFRestrictedScalarApproval
    approval_fingerprint: str
    readiness: Vituri2024TDHFScalarReadinessReceipt
    readiness_fingerprint: str
    assembly_receipt: Vituri2024TDHFSignedQAssemblyReceipt
    assembly_receipt_fingerprint: str
    source_payload: Vituri2024HalfMetalHFReplayPayload
    source_payload_fingerprint: str
    context: Vituri2024TDHFAssemblyContext
    context_fingerprint: str
    orbital_id_map: tuple[tuple[int, Vituri2024Orbital], ...]
    orbital_crosswalk: tuple[Vituri2024RestrictedScalarOrbitalCrosswalk, ...]
    orbital_crosswalk_fingerprint: str
    wbar_tensor_ev: Array
    wbar_tensor_fingerprint: str
    conserving_vertices: tuple[Vituri2024RestrictedScalarVertexBinding, ...]
    conserving_vertices_fingerprint: str
    p0: Array
    p0_fingerprint: str
    f0_ev: Array
    f0_fingerprint: str
    sigma_p0_ev: Array
    sigma_p0_fingerprint: str
    h_ev: Array
    h_fingerprint: str
    scalar_A_plus_ev: Array
    scalar_B_plus_minus_ev: Array
    scalar_A_minus_ev: Array
    scalar_B_minus_plus_ev: Array
    dF_column_evidence: tuple[Vituri2024RestrictedScalarDFColumnEvidence, ...]
    dF_column_evidence_fingerprint: str
    literal_states: tuple[int, ...]
    literal_reference_index: int
    literal_one_body_hamiltonian_ev: Array
    literal_one_body_hamiltonian_fingerprint: str
    literal_interaction_hamiltonian_ev: Array
    literal_interaction_hamiltonian_fingerprint: str
    literal_hamiltonian_ev: Array
    literal_hamiltonian_fingerprint: str
    tangent_basis: TDHFTransitionTangentBasis
    tangent_basis_fingerprint: str
    generic_approval: TDHFScalarCurvatureApproval
    generic_approval_fingerprint: str
    generic_certificate: TDHFScalarCurvatureCertificate
    generic_certificate_fingerprint: str
    residuals: Vituri2024RestrictedScalarResiduals
    residuals_fingerprint: str
    stencil_projector_fingerprints: tuple[str, ...]
    scalar_energy_call_count: int
    expected_scalar_energy_call_count: int
    sigma_equation: str = field(default=_SIGMA_EQUATION, init=False)
    energy_equation: str = field(default=_ENERGY_EQUATION, init=False)
    literal_hamiltonian_equation: str = field(
        default=_LITERAL_HAMILTONIAN_EQUATION, init=False
    )
    dF_column_table: str = field(default=_DF_COLUMN_TABLE, init=False)
    authority: str = field(
        default=VITURI2024_RESTRICTED_SCALAR_AUTHORITY, init=False
    )
    actual_vertex_compared: bool = field(default=True, init=False)
    actual_c9_compared: bool = field(default=True, init=False)
    real_full_provider: bool = field(default=False, init=False)
    source_scalar: bool = field(default=False, init=False)
    global_static_hessian_authority: bool = field(default=False, init=False)
    authority_promoted: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_numerical_parity: bool = field(default=False, init=False)
    receipt_fingerprint: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _RECEIPT_TOKEN:
            raise TypeError("restricted scalar receipt requires the private factory token")
        array_ranks = (
            ("wbar_tensor_ev", 4),
            ("p0", 2),
            ("f0_ev", 2),
            ("sigma_p0_ev", 2),
            ("h_ev", 2),
            ("scalar_A_plus_ev", 2),
            ("scalar_B_plus_minus_ev", 2),
            ("scalar_A_minus_ev", 2),
            ("scalar_B_minus_plus_ev", 2),
            ("literal_one_body_hamiltonian_ev", 2),
            ("literal_interaction_hamiltonian_ev", 2),
            ("literal_hamiltonian_ev", 2),
        )
        for name, ndim in array_ranks:
            object.__setattr__(
                self, name, _readonly_complex(getattr(self, name), ndim=ndim)
            )
        self._validate_consistency(check_fingerprint=False)
        object.__setattr__(self, "receipt_fingerprint", self._expected_fingerprint())

    def _expected_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": self.api_version,
                "approval_fingerprint": self.approval_fingerprint,
                "readiness_fingerprint": self.readiness_fingerprint,
                "assembly_receipt_fingerprint": self.assembly_receipt_fingerprint,
                "source_payload_fingerprint": self.source_payload_fingerprint,
                "context_fingerprint": self.context_fingerprint,
                "orbital_id_map": self.orbital_id_map,
                "orbital_crosswalk_fingerprint": self.orbital_crosswalk_fingerprint,
                "wbar_tensor_fingerprint": self.wbar_tensor_fingerprint,
                "conserving_vertices_fingerprint": self.conserving_vertices_fingerprint,
                "p0_fingerprint": self.p0_fingerprint,
                "f0_fingerprint": self.f0_fingerprint,
                "sigma_p0_fingerprint": self.sigma_p0_fingerprint,
                "h_fingerprint": self.h_fingerprint,
                "scalar_blocks": tuple(
                    _array_sha256(item)
                    for item in (
                        self.scalar_A_plus_ev,
                        self.scalar_B_plus_minus_ev,
                        self.scalar_A_minus_ev,
                        self.scalar_B_minus_plus_ev,
                    )
                ),
                "dF_column_evidence_fingerprint": self.dF_column_evidence_fingerprint,
                "literal_states": self.literal_states,
                "literal_reference_index": self.literal_reference_index,
                "literal_one_body_hamiltonian_fingerprint": (
                    self.literal_one_body_hamiltonian_fingerprint
                ),
                "literal_interaction_hamiltonian_fingerprint": (
                    self.literal_interaction_hamiltonian_fingerprint
                ),
                "literal_hamiltonian_fingerprint": self.literal_hamiltonian_fingerprint,
                "tangent_basis_fingerprint": self.tangent_basis_fingerprint,
                "generic_approval_fingerprint": self.generic_approval_fingerprint,
                "generic_certificate_fingerprint": self.generic_certificate_fingerprint,
                "residuals_fingerprint": self.residuals_fingerprint,
                "stencil_projector_fingerprints": self.stencil_projector_fingerprints,
                "scalar_energy_call_count": self.scalar_energy_call_count,
                "expected_scalar_energy_call_count": self.expected_scalar_energy_call_count,
                "authority": self.authority,
                "actual_vertex_compared": self.actual_vertex_compared,
                "actual_c9_compared": self.actual_c9_compared,
                "authority_locks": (
                    self.real_full_provider,
                    self.source_scalar,
                    self.global_static_hessian_authority,
                    self.authority_promoted,
                    self.production_ready,
                    self.paper_numerical_parity,
                ),
            }
        )

    def _validate_consistency(self, *, check_fingerprint: bool = True) -> None:
        if self.api_version != "vituri2024_tdhf_restricted_scalar.v1":
            raise ValueError("restricted scalar receipt API version drift")
        if type(self.approval) is not Vituri2024TDHFRestrictedScalarApproval:
            raise TypeError("receipt approval has the wrong exact type")
        if self.approval_fingerprint != self.approval.fingerprint:
            raise ValueError("receipt detached approval fingerprint mismatch")
        live_readiness, live_assembly, live_payload = _validate_inputs(
            self.readiness, self.assembly_receipt, self.source_payload
        )
        identity_locks = (
            self.readiness is self.approval.readiness,
            self.assembly_receipt is self.approval.assembly_receipt,
            self.source_payload is self.approval.source_payload,
            self.context is self.assembly_receipt.signed_pair.plus_context,
            self.generic_approval is self.approval.generic_approval,
            self.generic_certificate.approval is self.generic_approval,
        )
        if not all(identity_locks):
            raise ValueError("receipt lost exact approval/input object bindings")
        literal_shape = (len(self.literal_states), len(self.literal_states))
        for name in (
            "literal_one_body_hamiltonian_ev",
            "literal_interaction_hamiltonian_ev",
            "literal_hamiltonian_ev",
        ):
            array = getattr(self, name)
            if type(array) is not np.ndarray:
                raise TypeError(f"restricted scalar receipt {name} must be an array")
            if array.shape != literal_shape:
                raise ValueError(f"restricted scalar receipt {name} shape drift")
            if array.flags.writeable:
                raise ValueError(f"restricted scalar receipt {name} must be read-only")

        checks = (
            (self.readiness_fingerprint, live_readiness, "readiness"),
            (self.assembly_receipt_fingerprint, live_assembly, "assembly"),
            (self.source_payload_fingerprint, live_payload, "source payload"),
            (self.context_fingerprint, self.context.fingerprint, "context"),
            (self.orbital_id_map, self.assembly_receipt.orbital_id_map, "orbital map"),
            (
                self.orbital_crosswalk_fingerprint,
                _fingerprint(tuple(item.fingerprint for item in self.orbital_crosswalk)),
                "orbital crosswalk",
            ),
            (self.wbar_tensor_fingerprint, _array_sha256(self.wbar_tensor_ev), "wbar"),
            (
                self.conserving_vertices_fingerprint,
                _fingerprint(tuple(item.fingerprint for item in self.conserving_vertices)),
                "conserving vertices",
            ),
            (self.p0_fingerprint, _array_sha256(self.p0), "P0"),
            (self.f0_fingerprint, _array_sha256(self.f0_ev), "F0"),
            (self.sigma_p0_fingerprint, _array_sha256(self.sigma_p0_ev), "Sigma[P0]"),
            (self.h_fingerprint, _array_sha256(self.h_ev), "h"),
            (
                self.dF_column_evidence_fingerprint,
                _fingerprint(tuple(item.fingerprint for item in self.dF_column_evidence)),
                "dF columns",
            ),
            (
                self.literal_one_body_hamiltonian_fingerprint,
                _array_sha256(self.literal_one_body_hamiltonian_ev),
                "literal one-body Hamiltonian",
            ),
            (
                self.literal_interaction_hamiltonian_fingerprint,
                _array_sha256(self.literal_interaction_hamiltonian_ev),
                "literal interaction Hamiltonian",
            ),
            (
                self.literal_hamiltonian_fingerprint,
                _array_sha256(self.literal_hamiltonian_ev),
                "literal total Hamiltonian",
            ),
            (self.tangent_basis_fingerprint, self.tangent_basis.fingerprint, "tangent basis"),
            (
                self.generic_approval_fingerprint,
                self.generic_approval.fingerprint,
                "generic approval",
            ),
            (
                self.generic_certificate_fingerprint,
                self.generic_certificate.fingerprint,
                "generic certificate",
            ),
            (self.residuals_fingerprint, self.residuals.fingerprint, "residuals"),
        )
        for actual, expected, label in checks:
            if actual != expected:
                raise ValueError(f"restricted scalar receipt {label} drift")
        if not np.array_equal(
            self.literal_hamiltonian_ev,
            self.literal_one_body_hamiltonian_ev
            + self.literal_interaction_hamiltonian_ev,
        ):
            raise ValueError(
                "restricted scalar receipt literal Hamiltonian decomposition mismatch"
            )
        certificate = self.generic_certificate
        if not (
            certificate.scalar_curvature_executed
            and certificate.stationarity_complete_all_passed
            and certificate.registered_direction_curvatures_match
            and certificate.mathematical_scalar_hessian_match
            and certificate.mathematical_scalar_curvature_match
            and not certificate.static_hessian_authority_promoted
            and not certificate.promotion_eligible
        ):
            raise ValueError("generic scalar certificate conclusions drift")
        if (
            type(self.scalar_energy_call_count) is not int
            or self.scalar_energy_call_count != self.expected_scalar_energy_call_count
            or len(self.stencil_projector_fingerprints) != self.scalar_energy_call_count
        ):
            raise ValueError("scalar-energy registered-stencil call inventory drift")
        authority_locks = (
            self.sigma_equation == _SIGMA_EQUATION,
            self.energy_equation == _ENERGY_EQUATION,
            self.literal_hamiltonian_equation == _LITERAL_HAMILTONIAN_EQUATION,
            self.dF_column_table == _DF_COLUMN_TABLE,
            self.authority == VITURI2024_RESTRICTED_SCALAR_AUTHORITY,
            self.actual_vertex_compared is True,
            self.actual_c9_compared is True,
            self.real_full_provider is False,
            self.source_scalar is False,
            self.global_static_hessian_authority is False,
            self.authority_promoted is False,
            self.production_ready is False,
            self.paper_numerical_parity is False,
            self.assembly_receipt.sector.static_hessian_authority
            == "projected_signed_ab",
        )
        if not all(authority_locks):
            raise ValueError("restricted scalar receipt authority was inflated or mutated")
        if check_fingerprint and self.receipt_fingerprint != self._expected_fingerprint():
            raise ValueError("restricted scalar receipt fingerprint mismatch")

    @property
    def fingerprint(self) -> str:
        self._validate_consistency()
        return self.receipt_fingerprint


def certify_vituri2024_tdhf_restricted_scalar(
    *,
    approval: Vituri2024TDHFRestrictedScalarApproval,
    readiness: Vituri2024TDHFScalarReadinessReceipt,
    assembly_receipt: Vituri2024TDHFSignedQAssemblyReceipt,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
) -> Vituri2024TDHFRestrictedScalarReceipt:
    """Execute Wick/literal and exact-unitary checks after detached approval."""

    if type(approval) is not Vituri2024TDHFRestrictedScalarApproval:
        raise TypeError("certification requires exact detached restricted approval")
    approval._validate_consistency()
    if not (
        readiness is approval.readiness
        and assembly_receipt is approval.assembly_receipt
        and source_payload is approval.source_payload
    ):
        raise ValueError("certification inputs are not the exact detached-approved objects")

    original_sector_authority = assembly_receipt.sector.static_hessian_authority
    prepared = _prepare_oracle(
        readiness,
        assembly_receipt,
        source_payload,
        provenance=approval.provenance,
    )
    if prepared.deterministic_manifest_sha256 != approval.deterministic_manifest_sha256:
        raise ValueError("stale restricted scalar deterministic approval manifest")
    if prepared.generic_approval.fingerprint != approval.generic_approval_fingerprint:
        raise ValueError("stale restricted scalar generic approval/manifest")
    if prepared.functional_manifest != approval.generic_approval.functional_manifest:
        raise ValueError("stale restricted scalar functional manifest")

    # The first scalar call is an explicit P0 Wick/literal check.  Every
    # subsequent call comes from the generic factory's registered five-point
    # stationarity or curvature stencil.
    prepared.callback(prepared.p0)
    certificate = certify_tdhf_scalar_curvature(
        approval=approval.generic_approval,
        sector=assembly_receipt.sector,
        tangent_basis=prepared.tangent_basis,
        energy_callback=prepared.callback,
        functional_manifest=prepared.functional_manifest,
    )
    dimension = len(prepared.tangent_basis.plus_tangents) + len(
        prepared.tangent_basis.minus_tangents
    )
    expected_calls = 1 + (2 * dimension + dimension * dimension) * len(
        approval.generic_approval.step_ladder.steps
    ) * 5
    if len(prepared.tracker.residuals) != expected_calls:
        raise ValueError("generic registered-stencil scalar call count mismatch")
    if assembly_receipt.sector.static_hessian_authority != original_sector_authority:
        raise ValueError("restricted oracle mutated original sector authority")

    stationarity_residual = max(
        step.stationarity_residual
        for direction in certificate.stationarity_evidence
        for step in direction.steps
    )
    curvature_residual = max(
        step.curvature_residual
        for direction in certificate.direction_evidence
        for step in direction.steps
    )
    reconstruction_residual = max(
        item.max_abs_residual for item in certificate.reconstruction_evidence
    )
    deterministic = prepared.deterministic_residuals
    residuals = Vituri2024RestrictedScalarResiduals(
        **asdict(deterministic),
        wick_literal_p0=prepared.tracker.residuals[0],
        wick_literal_all_stencils=max(prepared.tracker.residuals),
        wick_energy_imaginary_part=max(prepared.tracker.imaginary_residuals),
        generic_stationarity=stationarity_residual,
        generic_curvature=curvature_residual,
        generic_hessian_reconstruction=reconstruction_residual,
    )
    return Vituri2024TDHFRestrictedScalarReceipt(
        _factory_token=_RECEIPT_TOKEN,
        api_version="vituri2024_tdhf_restricted_scalar.v1",
        approval=approval,
        approval_fingerprint=approval.fingerprint,
        readiness=readiness,
        readiness_fingerprint=readiness.fingerprint,
        assembly_receipt=assembly_receipt,
        assembly_receipt_fingerprint=assembly_receipt.fingerprint,
        source_payload=source_payload,
        source_payload_fingerprint=_source_payload_fingerprint(source_payload),
        context=prepared.context,
        context_fingerprint=prepared.context.fingerprint,
        orbital_id_map=assembly_receipt.orbital_id_map,
        orbital_crosswalk=prepared.orbital_crosswalk,
        orbital_crosswalk_fingerprint=_fingerprint(
            tuple(item.fingerprint for item in prepared.orbital_crosswalk)
        ),
        wbar_tensor_ev=prepared.wbar,
        wbar_tensor_fingerprint=_array_sha256(prepared.wbar),
        conserving_vertices=prepared.conserving_vertices,
        conserving_vertices_fingerprint=_fingerprint(
            tuple(item.fingerprint for item in prepared.conserving_vertices)
        ),
        p0=prepared.p0,
        p0_fingerprint=_array_sha256(prepared.p0),
        f0_ev=prepared.f0,
        f0_fingerprint=_array_sha256(prepared.f0),
        sigma_p0_ev=prepared.sigma_p0,
        sigma_p0_fingerprint=_array_sha256(prepared.sigma_p0),
        h_ev=prepared.h,
        h_fingerprint=_array_sha256(prepared.h),
        scalar_A_plus_ev=prepared.scalar_A_plus,
        scalar_B_plus_minus_ev=prepared.scalar_B_plus_minus,
        scalar_A_minus_ev=prepared.scalar_A_minus,
        scalar_B_minus_plus_ev=prepared.scalar_B_minus_plus,
        dF_column_evidence=prepared.dF_evidence,
        dF_column_evidence_fingerprint=_fingerprint(
            tuple(item.fingerprint for item in prepared.dF_evidence)
        ),
        literal_states=prepared.literal_states,
        literal_reference_index=prepared.literal_reference_index,
        literal_one_body_hamiltonian_ev=prepared.literal_one_body_hamiltonian,
        literal_one_body_hamiltonian_fingerprint=_array_sha256(
            prepared.literal_one_body_hamiltonian
        ),
        literal_interaction_hamiltonian_ev=prepared.literal_interaction_hamiltonian,
        literal_interaction_hamiltonian_fingerprint=_array_sha256(
            prepared.literal_interaction_hamiltonian
        ),
        literal_hamiltonian_ev=prepared.literal_hamiltonian,
        literal_hamiltonian_fingerprint=_array_sha256(prepared.literal_hamiltonian),
        tangent_basis=prepared.tangent_basis,
        tangent_basis_fingerprint=prepared.tangent_basis.fingerprint,
        generic_approval=approval.generic_approval,
        generic_approval_fingerprint=approval.generic_approval.fingerprint,
        generic_certificate=certificate,
        generic_certificate_fingerprint=certificate.fingerprint,
        residuals=residuals,
        residuals_fingerprint=residuals.fingerprint,
        stencil_projector_fingerprints=tuple(
            prepared.tracker.projector_fingerprints
        ),
        scalar_energy_call_count=len(prepared.tracker.residuals),
        expected_scalar_energy_call_count=expected_calls,
    )


__all__ = [
    "VITURI2024_RESTRICTED_SCALAR_ALGEBRA_ABSOLUTE_TOLERANCE",
    "VITURI2024_RESTRICTED_SCALAR_ALGEBRA_RELATIVE_TOLERANCE",
    "VITURI2024_RESTRICTED_SCALAR_AUTHORITY",
    "VITURI2024_RESTRICTED_SCALAR_MAX_ORBITALS",
    "VITURI2024_RESTRICTED_SCALAR_WICK_ABSOLUTE_TOLERANCE",
    "VITURI2024_RESTRICTED_SCALAR_WICK_RELATIVE_TOLERANCE",
    "Vituri2024RestrictedScalarDFColumnEvidence",
    "Vituri2024RestrictedScalarOrbitalCrosswalk",
    "Vituri2024RestrictedScalarResiduals",
    "Vituri2024RestrictedScalarVertexBinding",
    "Vituri2024TDHFRestrictedScalarApproval",
    "Vituri2024TDHFRestrictedScalarReceipt",
    "certify_vituri2024_tdhf_restricted_scalar",
    "make_vituri2024_tdhf_restricted_scalar_approval",
]
