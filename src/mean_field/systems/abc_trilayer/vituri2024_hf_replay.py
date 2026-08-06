"""Array-only replay gate for the Vituri-2024 half-metal HF source.

This module loads one hash-bound, already-computed source payload and checks its
canonical array layout, receipt hashes, diagonal half-metal structure, and
finite-domain base-mesh pocket evidence.  It does not run SCF iterations,
replay the branch table or pocket refinement, call the shared E/dE/dP/d2E
functional chain, or establish scientific execution or paper reproduction.
"""
from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass, field
import hashlib
import json
import math
from numbers import Real
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .vituri2024_hf_preflight import (
    ACTIVE_BAND_STATES_GAUGE_SCOPE,
    ACTIVE_BAND_STATES_LAYOUT,
    ACTIVE_BAND_STATES_VALLEY_ORDER,
    CANONICAL_BASIS_KIND,
    FOCK_DECOMPOSITION_CONVENTION,
    INTERNAL_FLAVOR_ORDER,
    ORBITAL_INDEX_DESCRIPTOR_LABEL,
    ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT,
    ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL,
    REPLAY_ARRAY_CONVERSION,
    REPLAY_ARRAY_LAYOUT,
    REPLAY_ORBITAL_ORDER,
    REPLAY_PAYLOAD_SCHEMA_FINGERPRINT,
    REPLAY_RESIDUAL_NORM,
    VITURI2024_BASE_PROVIDER_METADATA_FIELDS,
    Vituri2024HalfMetalHFProviderBinding,
    Vituri2024HalfMetalHFProviderProtocol,
    _orbital_index_descriptor_fingerprint,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _require_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase 40- or 64-character commit")
    return value


def _require_finite_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a strict real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result

def _require_positive_real(value: object, label: str) -> float:
    result = _require_finite_real(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result

def canonical_array_sha256(array: np.ndarray) -> str:
    """Hash one array's canonical C-order shape, dtype, and element bytes."""

    if not isinstance(array, np.ndarray):
        raise TypeError("canonical array hashing requires a numpy.ndarray")
    if array.dtype.hasobject:
        raise TypeError("canonical array hashing rejects object dtypes")
    header = json.dumps(
        {
            "schema": "vituri2024_canonical_array_v1",
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "byte_order": "C",
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, byteorder="big", signed=False))
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_orbital_order_sha256(mesh: FloatArray) -> str:
    """Hash the orbital *index descriptor*, not an active-band state array."""

    if not isinstance(mesh, np.ndarray):
        raise TypeError("orbital-order hashing requires a numpy.ndarray mesh")
    if mesh.dtype != np.dtype(np.float64):
        raise TypeError("orbital-order mesh dtype must be exactly float64")
    if mesh.ndim != 2 or mesh.shape[1] != 2 or mesh.shape[0] < 1:
        raise ValueError("orbital-order mesh must have shape (Nk,2) with Nk positive")
    if not np.all(np.isfinite(mesh)):
        raise ValueError("orbital-order mesh must be finite")
    return _fingerprint(
        {
            "descriptor_label": ORBITAL_INDEX_DESCRIPTOR_LABEL,
            "schema_label": ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL,
            "schema_fingerprint": ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT,
            "internal_flavor_order": INTERNAL_FLAVOR_ORDER,
            "orbital_order": REPLAY_ORBITAL_ORDER,
            "nk": int(mesh.shape[0]),
            "ordered_momentum_mesh_sha256": canonical_array_sha256(mesh),
        }
    )


def _immutable_finite_array(
    value: object,
    *,
    label: str,
    dtype: np.dtype[object],
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a numpy.ndarray")
    if value.dtype != dtype:
        raise TypeError(f"{label} dtype must be exactly {dtype.name}")
    if value.shape != shape:
        raise ValueError(f"{label} shape must be exactly {shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must contain only finite values")
    immutable = np.frombuffer(value.tobytes(order="C"), dtype=dtype).reshape(shape)
    immutable.flags.writeable = False
    return immutable


@dataclass(frozen=True, slots=True)
class Vituri2024HalfMetalHFReplayPayload:
    """Provider-loaded canonical arrays with immutable copied storage."""

    provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    source_state_sha256: str
    replay_loader_implementation_fingerprint: str
    replay_payload_schema_fingerprint: str
    mesh: FloatArray
    active_band_states: ComplexArray
    h0: ComplexArray
    interaction_h: ComplexArray
    fock: ComplexArray
    projector: ComplexArray
    energies: FloatArray
    occupations: IntegerArray

    def __post_init__(self) -> None:
        _require_sha256(self.provider_fingerprint, "payload provider fingerprint")
        _require_commit(self.source_commit, "payload source commit")
        _require_sha256(self.source_artifact_sha256, "payload source artifact")
        _require_sha256(self.spec_fingerprint, "payload spec fingerprint")
        _require_sha256(self.source_state_sha256, "payload source state")
        _require_sha256(
            self.replay_loader_implementation_fingerprint,
            "payload replay-loader implementation",
        )
        _require_sha256(
            self.replay_payload_schema_fingerprint,
            "payload replay schema",
        )
        if self.replay_payload_schema_fingerprint != REPLAY_PAYLOAD_SCHEMA_FINGERPRINT:
            raise ValueError("payload replay schema fingerprint mismatch")

        if not isinstance(self.mesh, np.ndarray) or self.mesh.ndim != 2:
            raise ValueError("mesh shape must be exactly (Nk,2)")
        if self.mesh.shape[1:] != (2,) or self.mesh.shape[0] < 1:
            raise ValueError("mesh shape must be exactly (Nk,2) with Nk positive")
        nk = int(self.mesh.shape[0])
        mesh = _immutable_finite_array(
            self.mesh,
            label="mesh",
            dtype=np.dtype(np.float64),
            shape=(nk, 2),
        )
        matrix_shape = (len(INTERNAL_FLAVOR_ORDER), len(INTERNAL_FLAVOR_ORDER), nk)
        band_shape = (len(INTERNAL_FLAVOR_ORDER), nk)
        active_band_states = _immutable_finite_array(
            self.active_band_states,
            label="active_band_states",
            dtype=np.dtype(np.complex128),
            shape=(len(ACTIVE_BAND_STATES_VALLEY_ORDER), 6, nk),
        )
        h0 = _immutable_finite_array(
            self.h0,
            label="h0",
            dtype=np.dtype(np.complex128),
            shape=matrix_shape,
        )
        interaction_h = _immutable_finite_array(
            self.interaction_h,
            label="interaction_h",
            dtype=np.dtype(np.complex128),
            shape=matrix_shape,
        )
        fock = _immutable_finite_array(
            self.fock,
            label="fock",
            dtype=np.dtype(np.complex128),
            shape=matrix_shape,
        )
        projector = _immutable_finite_array(
            self.projector,
            label="projector",
            dtype=np.dtype(np.complex128),
            shape=matrix_shape,
        )
        energies = _immutable_finite_array(
            self.energies,
            label="energies",
            dtype=np.dtype(np.float64),
            shape=band_shape,
        )
        occupations = _immutable_finite_array(
            self.occupations,
            label="occupations",
            dtype=np.dtype(np.int64),
            shape=band_shape,
        )
        if not np.all((occupations == 0) | (occupations == 1)):
            raise ValueError("occupations must contain strict integer 0/1 values")
        for name, array in (
            ("mesh", mesh),
            ("active_band_states", active_band_states),
            ("h0", h0),
            ("interaction_h", interaction_h),
            ("fock", fock),
            ("projector", projector),
            ("energies", energies),
            ("occupations", occupations),
        ):
            object.__setattr__(self, name, array)


@runtime_checkable
class Vituri2024HalfMetalHFReplayProviderProtocol(
    Vituri2024HalfMetalHFProviderProtocol, Protocol
):
    """Metadata-complete preflight provider plus one array-replay loader."""

    replay_loader_implementation_fingerprint: str
    replay_payload_schema_fingerprint: str

    def load_half_metal_replay_payload(
        self, source_artifact_sha256: str
    ) -> Vituri2024HalfMetalHFReplayPayload: ...


@dataclass(frozen=True, slots=True)
class Vituri2024HalfMetalHFReplayHashes:
    ordered_momentum_mesh_sha256: str
    ordered_orbitals_sha256: str
    ordered_orbitals_descriptor_fingerprint: str
    active_band_states_sha256: str
    ordered_energies_sha256: str
    ordered_occupations_sha256: str
    ordered_projector_sha256: str
    ordered_fock_sha256: str
    h0_sha256: str
    interaction_h_sha256: str
    reconstructed_source_state_sha256: str
    payload_manifest_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_sha256(value, name.replace("_", " "))


@dataclass(frozen=True, slots=True)
class Vituri2024HalfMetalHFReplayResiduals:
    residual_norm: Literal["entrywise_max_abs"]
    fock_decomposition_max_abs_ev: float
    fock_decomposition_tolerance_ev: float
    h0_hermiticity_max_abs_ev: float
    h0_hermiticity_tolerance_ev: float
    interaction_h_hermiticity_max_abs_ev: float
    interaction_h_hermiticity_tolerance_ev: float
    active_band_state_norm_max_abs: float
    active_band_state_norm_tolerance: float
    projector_hermiticity_max_abs: float
    projector_hermiticity_tolerance: float
    projector_idempotency_max_abs: float
    projector_idempotency_tolerance: float
    fock_hermiticity_max_abs_ev: float
    fock_hermiticity_tolerance_ev: float
    fock_projector_commutator_max_abs_ev: float
    fock_projector_commutator_tolerance_ev: float
    projector_vs_occupation_max_abs: float
    projector_vs_occupation_tolerance: float
    canonical_basis_diagonal_closure_max_abs_ev: float
    canonical_basis_diagonal_closure_tolerance_ev: float
    aufbau_gap_ev: float
    aufbau_violation_ev: float
    aufbau_tolerance_ev: float
    chemical_mu_occupation_violation_ev: float
    chemical_mu_occupation_tolerance_ev: float

    def __post_init__(self) -> None:
        if self.residual_norm != REPLAY_RESIDUAL_NORM:
            raise ValueError("replay residual norm contract mismatch")
        pairs = (
            (self.fock_decomposition_max_abs_ev, self.fock_decomposition_tolerance_ev, "Fock decomposition"),
            (self.h0_hermiticity_max_abs_ev, self.h0_hermiticity_tolerance_ev, "h0 Hermiticity"),
            (self.interaction_h_hermiticity_max_abs_ev, self.interaction_h_hermiticity_tolerance_ev, "interaction_h Hermiticity"),
            (self.active_band_state_norm_max_abs, self.active_band_state_norm_tolerance, "active-band state norm"),
            (self.projector_hermiticity_max_abs, self.projector_hermiticity_tolerance, "projector Hermiticity"),
            (self.projector_idempotency_max_abs, self.projector_idempotency_tolerance, "projector idempotency"),
            (self.fock_hermiticity_max_abs_ev, self.fock_hermiticity_tolerance_ev, "Fock Hermiticity"),
            (self.fock_projector_commutator_max_abs_ev, self.fock_projector_commutator_tolerance_ev, "Fock/projector commutator"),
            (self.projector_vs_occupation_max_abs, self.projector_vs_occupation_tolerance, "projector/occupation closure"),
            (self.canonical_basis_diagonal_closure_max_abs_ev, self.canonical_basis_diagonal_closure_tolerance_ev, "canonical-basis diagonal closure"),
            (self.aufbau_violation_ev, self.aufbau_tolerance_ev, "Aufbau occupation"),
            (self.chemical_mu_occupation_violation_ev, self.chemical_mu_occupation_tolerance_ev, "chemical-potential occupation closure"),
        )
        for residual, tolerance, label in pairs:
            clean_tolerance = _require_positive_real(
                tolerance, f"{label} tolerance"
            )
            _require_within(residual, clean_tolerance, label)
        _require_finite_real(self.aufbau_gap_ev, "Aufbau gap")


@dataclass(frozen=True, slots=True)
class Vituri2024HalfMetalHFBasePocketReplayEvidence:
    valley: Literal[-1, 1]
    selected_spin: Literal[-1, 1]
    hole_component_count: int
    component_cardinalities: tuple[int, ...]
    hole_state_count: int
    adjacency_convention: Literal["four_neighbor_finite_domain_no_wrap"]
    attested_pocket_receipt_fingerprint: str

    def __post_init__(self) -> None:
        if (
            type(self.valley) is not int
            or self.valley not in (-1, 1)
            or type(self.selected_spin) is not int
            or self.selected_spin not in (-1, 1)
        ):
            raise ValueError("pocket evidence valley/spin must be exactly -1 or +1")
        if type(self.hole_component_count) is not int or self.hole_component_count < 1:
            raise TypeError("pocket component count must be a positive strict integer")
        cardinalities = tuple(self.component_cardinalities)
        if (
            len(cardinalities) != self.hole_component_count
            or any(type(value) is not int or value < 1 for value in cardinalities)
            or cardinalities != tuple(sorted(cardinalities, reverse=True))
        ):
            raise ValueError("pocket component cardinalities are invalid")
        if type(self.hole_state_count) is not int or self.hole_state_count != sum(cardinalities):
            raise ValueError("pocket hole-state count does not close")
        if self.adjacency_convention != "four_neighbor_finite_domain_no_wrap":
            raise ValueError("pocket adjacency convention mismatch")
        _require_sha256(
            self.attested_pocket_receipt_fingerprint,
            "attested pocket receipt fingerprint",
        )
        object.__setattr__(self, "component_cardinalities", cardinalities)


@dataclass(frozen=True, slots=True)
class Vituri2024HalfMetalHFReplayOccupationEvidence:
    chemical_potential_ev: float
    selected_spin: Literal[-1, 1]
    selected_spin_occupied_state_count: int
    selected_spin_unoccupied_state_count: int
    selected_spin_band_min_ev: float
    selected_spin_band_max_ev: float
    valley_minus_hole_count: int
    valley_plus_hole_count: int
    selected_spin_hole_count: int
    opposite_spin_hole_count: int
    measured_density_cm2: float
    target_density_cm2: float
    density_residual_cm2: float
    density_tolerance_cm2: float

    def __post_init__(self) -> None:
        for name in (
            "chemical_potential_ev",
            "selected_spin_band_min_ev",
            "selected_spin_band_max_ev",
            "measured_density_cm2",
            "target_density_cm2",
            "density_residual_cm2",
            "density_tolerance_cm2",
        ):
            _require_finite_real(getattr(self, name), name)
        if type(self.selected_spin) is not int or self.selected_spin not in (-1, 1):
            raise ValueError("occupation evidence selected spin must be -1 or +1")
        count_names = (
            "selected_spin_occupied_state_count",
            "selected_spin_unoccupied_state_count",
            "valley_minus_hole_count",
            "valley_plus_hole_count",
            "selected_spin_hole_count",
            "opposite_spin_hole_count",
        )
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in count_names):
            raise TypeError("occupation evidence counts must be non-negative strict integers")
        if self.selected_spin_occupied_state_count < 1 or self.selected_spin_unoccupied_state_count < 1:
            raise ValueError("occupation evidence requires occupied and unoccupied states")
        if (
            self.selected_spin_hole_count
            != self.valley_minus_hole_count + self.valley_plus_hole_count
            or self.selected_spin_unoccupied_state_count
            != self.selected_spin_hole_count
            or self.valley_minus_hole_count != self.valley_plus_hole_count
            or self.opposite_spin_hole_count != 0
        ):
            raise ValueError("occupation evidence valley-hole counts do not close")
        if self.selected_spin_band_min_ev >= self.selected_spin_band_max_ev:
            raise ValueError("occupation evidence band extrema are not ordered")
        if self.density_tolerance_cm2 <= 0.0 or self.density_residual_cm2 < 0.0:
            raise ValueError("occupation evidence density residual/tolerance is invalid")
        if self.density_residual_cm2 > self.density_tolerance_cm2:
            raise ValueError("occupation evidence density residual exceeds tolerance")
        _require_scalar_match(
            self.density_residual_cm2,
            abs(self.measured_density_cm2 - self.target_density_cm2),
            "density residual",
        )


_REPLAY_FACTORY_TOKEN = object()

@dataclass(frozen=True, slots=True)
class Vituri2024HalfMetalHFReplayStatus:
    """Successful array replay, constructible only by the replay factory."""

    _factory_token: InitVar[object]
    arrays_loaded: Literal[True] = field(default=True, init=False)
    array_hashes_verified: Literal[True] = field(default=True, init=False)
    source_structure_verified: Literal[True] = field(default=True, init=False)
    provider_methods_executed: tuple[Literal["load_half_metal_replay_payload"], ...] = (
        field(default=("load_half_metal_replay_payload",), init=False)
    )
    scf_trajectory_replayed: Literal[False] = field(default=False, init=False)
    branch_table_replayed: Literal[False] = field(default=False, init=False)
    pocket_refinement_replayed: Literal[False] = field(default=False, init=False)
    functional_chain_replayed: Literal[False] = field(default=False, init=False)
    scientific_execution_verified: Literal[False] = field(default=False, init=False)
    paper_reproduction_verified: Literal[False] = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _REPLAY_FACTORY_TOKEN:
            raise TypeError(
                "replay success status requires the private replay factory token"
            )
        if not (
            self.arrays_loaded
            and self.array_hashes_verified
            and self.source_structure_verified
        ):
            raise ValueError("successful replay status lost a positive replay gate")
        if self.provider_methods_executed != ("load_half_metal_replay_payload",):
            raise ValueError("successful replay status has an invalid method inventory")
        if any(
            (
                self.scf_trajectory_replayed,
                self.branch_table_replayed,
                self.pocket_refinement_replayed,
                self.functional_chain_replayed,
                self.scientific_execution_verified,
                self.paper_reproduction_verified,
            )
        ):
            raise ValueError("array-only replay status cannot claim scientific execution")


@dataclass(frozen=True, slots=True)
class Vituri2024HalfMetalHFReplayReceipt:
    """Hash- and evidence-bound result constructible only by replay success."""

    provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    attested_source_receipt_fingerprint: str
    replay_loader_implementation_fingerprint: str
    replay_payload_schema_fingerprint: str
    internal_flavor_order: tuple[tuple[int, int], ...]
    array_layout: Literal["internal_flavor_internal_flavor_k_final"]
    array_conversion: Literal["identity_no_transpose"]
    orbital_order: Literal["flavor_major_then_k"]
    ordered_orbitals_descriptor_label: Literal["orbital_index_descriptor"]
    ordered_orbitals_schema_label: Literal[
        "vituri2024_orbital_index_descriptor_v1"
    ]
    ordered_orbitals_schema_fingerprint: str
    active_band_states_layout: Literal["valley_six_band_component_k_final"]
    active_band_states_valley_order: tuple[int, int]
    active_band_states_gauge_scope: Literal[
        "gauge_covariant_source_data_not_paper_gauge"
    ]
    canonical_basis_kind: Literal[
        "uniform_half_metal_flavor_momentum_diagonal"
    ]
    residual_norm: Literal["entrywise_max_abs"]
    fock_decomposition_convention: Literal[
        "fock_equals_h0_plus_interaction_h"
    ]
    hashes: Vituri2024HalfMetalHFReplayHashes
    residuals: Vituri2024HalfMetalHFReplayResiduals
    occupation_evidence: Vituri2024HalfMetalHFReplayOccupationEvidence
    base_pocket_evidence: tuple[
        Vituri2024HalfMetalHFBasePocketReplayEvidence,
        Vituri2024HalfMetalHFBasePocketReplayEvidence,
    ]
    status: Vituri2024HalfMetalHFReplayStatus
    _factory_token: InitVar[object]

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _REPLAY_FACTORY_TOKEN:
            raise TypeError("replay receipt requires the private replay factory token")
        for value, label in (
            (self.provider_fingerprint, "receipt provider fingerprint"),
            (self.source_artifact_sha256, "receipt source artifact"),
            (self.spec_fingerprint, "receipt spec fingerprint"),
            (self.attested_source_receipt_fingerprint, "receipt attested source"),
            (
                self.replay_loader_implementation_fingerprint,
                "receipt replay loader",
            ),
            (self.replay_payload_schema_fingerprint, "receipt replay schema"),
            (self.ordered_orbitals_schema_fingerprint, "receipt orbital schema"),
        ):
            _require_sha256(value, label)
        _require_commit(self.source_commit, "receipt source commit")
        if (
            type(self.internal_flavor_order) is not tuple
            or tuple(self.internal_flavor_order) != INTERNAL_FLAVOR_ORDER
            or any(
                type(flavor) is not tuple
                or any(type(value) is not int for value in flavor)
                for flavor in self.internal_flavor_order
            )
            or self.array_layout != REPLAY_ARRAY_LAYOUT
            or self.array_conversion != REPLAY_ARRAY_CONVERSION
            or self.orbital_order != REPLAY_ORBITAL_ORDER
            or self.ordered_orbitals_descriptor_label
            != ORBITAL_INDEX_DESCRIPTOR_LABEL
            or self.ordered_orbitals_schema_label
            != ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL
            or self.ordered_orbitals_schema_fingerprint
            != ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT
            or self.active_band_states_layout != ACTIVE_BAND_STATES_LAYOUT
            or type(self.active_band_states_valley_order) is not tuple
            or tuple(self.active_band_states_valley_order)
            != ACTIVE_BAND_STATES_VALLEY_ORDER
            or any(
                type(value) is not int
                for value in self.active_band_states_valley_order
            )
            or self.active_band_states_gauge_scope
            != ACTIVE_BAND_STATES_GAUGE_SCOPE
            or self.canonical_basis_kind != CANONICAL_BASIS_KIND
            or self.residual_norm != REPLAY_RESIDUAL_NORM
            or self.fock_decomposition_convention
            != FOCK_DECOMPOSITION_CONVENTION
            or self.replay_payload_schema_fingerprint
            != REPLAY_PAYLOAD_SCHEMA_FINGERPRINT
        ):
            raise ValueError("replay receipt schema/layout/basis contract mismatch")
        if type(self.hashes) is not Vituri2024HalfMetalHFReplayHashes:
            raise TypeError("replay receipt requires typed hash evidence")
        if type(self.residuals) is not Vituri2024HalfMetalHFReplayResiduals:
            raise TypeError("replay receipt requires typed residual evidence")
        if (
            type(self.occupation_evidence)
            is not Vituri2024HalfMetalHFReplayOccupationEvidence
        ):
            raise TypeError("replay receipt requires typed occupation evidence")
        pockets = tuple(self.base_pocket_evidence)
        if (
            len(pockets) != 2
            or any(
                type(item) is not Vituri2024HalfMetalHFBasePocketReplayEvidence
                for item in pockets
            )
            or tuple(item.valley for item in pockets) != (-1, 1)
            or pockets[0].selected_spin != pockets[1].selected_spin
        ):
            raise ValueError("replay receipt requires ordered bilateral pocket evidence")
        if (
            self.residuals.residual_norm != self.residual_norm
            or pockets[0].selected_spin
            != self.occupation_evidence.selected_spin
        ):
            raise ValueError("replay receipt nested evidence does not close")
        if type(self.status) is not Vituri2024HalfMetalHFReplayStatus:
            raise TypeError("replay receipt requires factory-created success status")
        object.__setattr__(self, "base_pocket_evidence", pockets)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


def _max_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array)))


def _require_within(actual: float, tolerance: float, label: str) -> None:
    clean_actual = _require_finite_real(actual, f"{label} residual")
    clean_tolerance = _require_positive_real(tolerance, f"{label} tolerance")
    if clean_actual < 0.0:
        raise ValueError(f"{label} residual must be finite and non-negative")
    if clean_actual > clean_tolerance:
        raise ValueError(f"{label} residual exceeds its replay tolerance")


def _require_receipt_match(actual: float, expected: float, label: str) -> None:
    scale = max(abs(actual), abs(expected))
    if not math.isclose(
        actual,
        expected,
        rel_tol=1.0e-12,
        abs_tol=64.0 * math.ulp(scale),
    ):
        raise ValueError(f"{label} residual does not match the attested receipt")


def _require_scalar_match(actual: float, expected: float, label: str) -> None:
    scale = max(abs(actual), abs(expected), 1.0)
    if not math.isclose(
        actual,
        expected,
        rel_tol=1.0e-12,
        abs_tol=64.0 * math.ulp(scale),
    ):
        raise ValueError(f"recomputed {label} does not match the attested receipt")


def _matrix_product(left: ComplexArray, right: ComplexArray) -> ComplexArray:
    return np.einsum("aik,ibk->abk", left, right, optimize=False)


def _diagonal_matrix(values: np.ndarray) -> ComplexArray:
    result = np.zeros(
        (len(INTERNAL_FLAVOR_ORDER), len(INTERNAL_FLAVOR_ORDER), values.shape[1]),
        dtype=np.complex128,
    )
    diagonal = np.arange(len(INTERNAL_FLAVOR_ORDER))
    result[diagonal, diagonal, :] = values
    return result


def _component_cardinalities(mask: NDArray[np.bool_], shape: tuple[int, int]) -> tuple[int, ...]:
    if mask.shape != (shape[0] * shape[1],):
        raise ValueError("hole mask does not match the declared base mesh shape")
    visited = np.zeros(mask.shape, dtype=np.bool_)
    cardinalities: list[int] = []
    width = shape[1]
    for start in np.flatnonzero(mask):
        start_index = int(start)
        if visited[start_index]:
            continue
        stack = [start_index]
        visited[start_index] = True
        count = 0
        while stack:
            index = stack.pop()
            count += 1
            row, column = divmod(index, width)
            neighbors: list[int] = []
            if row > 0:
                neighbors.append(index - width)
            if row + 1 < shape[0]:
                neighbors.append(index + width)
            if column > 0:
                neighbors.append(index - 1)
            if column + 1 < width:
                neighbors.append(index + 1)
            for neighbor in neighbors:
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        cardinalities.append(count)
    return tuple(sorted(cardinalities, reverse=True))


_REPLAY_PROVIDER_METADATA_FIELDS = VITURI2024_BASE_PROVIDER_METADATA_FIELDS + (
    "replay_loader_implementation_fingerprint",
    "replay_payload_schema_fingerprint",
)

def _provider_metadata_snapshot(provider: object) -> dict[str, object]:
    return {
        name: getattr(provider, name) for name in _REPLAY_PROVIDER_METADATA_FIELDS
    }


def _validate_preload_provider_snapshot(
    binding: Vituri2024HalfMetalHFProviderBinding,
    provider_snapshot: dict[str, object],
) -> None:
    if type(provider_snapshot) is not dict:
        raise TypeError("preload provider snapshot must be a strict dictionary")
    required_fields = frozenset(_REPLAY_PROVIDER_METADATA_FIELDS)
    actual_fields = frozenset(provider_snapshot)
    if actual_fields != required_fields:
        missing = sorted(required_fields - actual_fields)
        unexpected = sorted(actual_fields - required_fields)
        raise ValueError(
            "preload provider snapshot fields mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    spec = binding.spec
    assert spec.geometry is not None
    assert spec.ensemble is not None
    assert spec.scf_policy is not None
    assert spec.shared_functional is not None
    assert spec.attested_source is not None
    expected = {
        "provider_fingerprint": (
            spec.geometry.provider_fingerprint,
            "provider identity",
        ),
        "source_commit": (spec.shared_functional.source_commit, "source commit"),
        "source_artifact_sha256": (
            spec.shared_functional.source_artifact_sha256,
            "source artifact",
        ),
        "spec_fingerprint": (spec.fingerprint, "spec fingerprint"),
        "geometry_receipt_fingerprint": (
            spec.geometry.fingerprint,
            "geometry receipt fingerprint",
        ),
        "ensemble_receipt_fingerprint": (
            spec.ensemble.fingerprint,
            "ensemble receipt fingerprint",
        ),
        "scf_policy_receipt_fingerprint": (
            spec.scf_policy.fingerprint,
            "SCF-policy receipt fingerprint",
        ),
        "shared_functional_receipt_fingerprint": (
            spec.shared_functional.fingerprint,
            "shared-functional receipt fingerprint",
        ),
        "attested_source_receipt_fingerprint": (
            spec.attested_source.fingerprint,
            "attested-source receipt fingerprint",
        ),
        "finite_area_receipt_fingerprint": (
            spec.geometry.finite_area_receipt_fingerprint,
            "finite-area receipt fingerprint",
        ),
        "interaction_receipt_fingerprint": (
            spec.shared_functional.interaction_receipt_fingerprint,
            "interaction receipt fingerprint",
        ),
        "normal_order_reference_fingerprint": (
            spec.ensemble.normal_order_reference_fingerprint,
            "normal-order reference fingerprint",
        ),
        "q0_policy_fingerprint": (
            spec.ensemble.q0_policy_fingerprint,
            "q0 policy fingerprint",
        ),
        "source_state_sha256": (
            spec.attested_source.source_state_sha256,
            "source-state fingerprint",
        ),
        "scalar_energy_implementation_fingerprint": (
            spec.shared_functional.scalar_energy.implementation_fingerprint,
            "scalar-energy implementation fingerprint",
        ),
        "fock_derivative_implementation_fingerprint": (
            spec.shared_functional.fock_derivative.implementation_fingerprint,
            "Fock-derivative implementation fingerprint",
        ),
        "finite_q_hessian_implementation_fingerprint": (
            spec.shared_functional.finite_q_hessian.implementation_fingerprint,
            "finite-q-Hessian implementation fingerprint",
        ),
        "interaction_form_factor_implementation_fingerprint": (
            spec.shared_functional.interaction_form_factor.implementation_fingerprint,
            "interaction/form-factor implementation fingerprint",
        ),
        "replay_loader_implementation_fingerprint": (
            spec.attested_source.replay_loader_implementation_fingerprint,
            "loader implementation fingerprint",
        ),
        "replay_payload_schema_fingerprint": (
            spec.attested_source.replay_payload_schema_fingerprint,
            "payload schema fingerprint",
        ),
    }
    if tuple(expected) != _REPLAY_PROVIDER_METADATA_FIELDS:
        raise RuntimeError("internal preload provider snapshot mapping is incomplete")

    for field_name, (required, label) in expected.items():
        actual = provider_snapshot[field_name]
        if field_name == "source_commit":
            _require_commit(actual, f"preload provider snapshot {label}")
        else:
            _require_sha256(actual, f"preload provider snapshot {label}")
        if actual != required:
            raise ValueError(f"preload provider snapshot {label} mismatch")


def _validate_payload_against_provider_snapshot(
    payload: Vituri2024HalfMetalHFReplayPayload,
    provider_snapshot: dict[str, object],
) -> None:
    provider_values = provider_snapshot
    expected_payload = (
        (
            payload.provider_fingerprint,
            provider_values["provider_fingerprint"],
            "provider fingerprint",
        ),
        (
            payload.source_commit,
            provider_values["source_commit"],
            "source commit",
        ),
        (
            payload.source_artifact_sha256,
            provider_values["source_artifact_sha256"],
            "source artifact",
        ),
        (
            payload.spec_fingerprint,
            provider_values["spec_fingerprint"],
            "spec fingerprint",
        ),
        (
            payload.source_state_sha256,
            provider_values["source_state_sha256"],
            "source-state fingerprint",
        ),
        (
            payload.replay_loader_implementation_fingerprint,
            provider_values["replay_loader_implementation_fingerprint"],
            "loader implementation fingerprint",
        ),
        (
            payload.replay_payload_schema_fingerprint,
            provider_values["replay_payload_schema_fingerprint"],
            "payload schema fingerprint",
        ),
    )
    for actual, expected, label in expected_payload:
        if actual != expected:
            raise ValueError(f"replay payload {label} mismatch")


def replay_vituri2024_half_metal_hf_arrays(
    binding: Vituri2024HalfMetalHFProviderBinding,
) -> Vituri2024HalfMetalHFReplayReceipt:
    """Load and verify one provider source payload without replaying HF physics."""

    if type(binding) is not Vituri2024HalfMetalHFProviderBinding:
        raise TypeError("array replay requires a typed complete provider binding")
    binding.spec.require_receipt_set_complete()
    provider = binding.provider
    if not isinstance(provider, Vituri2024HalfMetalHFReplayProviderProtocol):
        raise TypeError("provider is missing the runtime-checkable replay loader protocol")
    loader = getattr(provider, "load_half_metal_replay_payload", None)
    if not callable(loader):
        raise TypeError("provider replay loader must be callable")

    spec = binding.spec
    assert spec.geometry is not None
    assert spec.ensemble is not None
    assert spec.scf_policy is not None
    assert spec.shared_functional is not None
    assert spec.attested_source is not None
    source = spec.attested_source

    # Re-run the complete base binding, then independently pin the captured
    # preload snapshot to every base-binding and replay-loader/schema field.
    Vituri2024HalfMetalHFProviderBinding(spec, provider)
    provider_snapshot_before = _provider_metadata_snapshot(provider)
    _validate_preload_provider_snapshot(binding, provider_snapshot_before)
    provider_values_before = MappingProxyType(provider_snapshot_before)

    payload = loader(source.source_artifact_sha256)
    if type(payload) is not Vituri2024HalfMetalHFReplayPayload:
        raise TypeError("replay loader must return Vituri2024HalfMetalHFReplayPayload")

    # A mutable provider cannot drift any base-binding or loader field while
    # serving the payload, even if the returned payload is internally valid.
    provider_snapshot_after = _provider_metadata_snapshot(provider)
    if provider_snapshot_after != provider_snapshot_before:
        changed = tuple(
            name
            for name in _REPLAY_PROVIDER_METADATA_FIELDS
            if provider_snapshot_before[name] != provider_snapshot_after[name]
        )
        raise ValueError(
            "replay provider metadata mutated during loader call: "
            + ", ".join(changed)
        )
    Vituri2024HalfMetalHFProviderBinding(spec, provider)
    _validate_payload_against_provider_snapshot(payload, provider_snapshot_before)

    nk = spec.geometry.mesh_point_count
    if payload.mesh.shape != (nk, 2):
        raise ValueError("payload mesh shape does not match geometry (Nk,2)")
    if payload.active_band_states.shape != (2, 6, nk):
        raise ValueError(
            "payload active_band_states shape does not match (valley,6,Nk)"
        )
    if payload.h0.shape != (4, 4, nk) or payload.interaction_h.shape != (4, 4, nk):
        raise ValueError("payload Hamiltonian shapes do not match geometry (4,4,Nk)")
    if payload.fock.shape != (4, 4, nk) or payload.projector.shape != (4, 4, nk):
        raise ValueError("payload Fock/projector shapes do not match geometry (4,4,Nk)")
    if payload.energies.shape != (4, nk) or payload.occupations.shape != (4, nk):
        raise ValueError("payload energy/occupation shapes do not match geometry (4,Nk)")

    mesh_hash = canonical_array_sha256(payload.mesh)
    orbital_hash = canonical_orbital_order_sha256(payload.mesh)
    orbital_descriptor_fingerprint = _orbital_index_descriptor_fingerprint(
        orbital_hash, mesh_hash
    )
    active_band_states_hash = canonical_array_sha256(payload.active_band_states)
    energy_hash = canonical_array_sha256(payload.energies)
    occupation_hash = canonical_array_sha256(payload.occupations)
    projector_hash = canonical_array_sha256(payload.projector)
    fock_hash = canonical_array_sha256(payload.fock)
    h0_hash = canonical_array_sha256(payload.h0)
    interaction_h_hash = canonical_array_sha256(payload.interaction_h)
    expected_hashes = (
        (mesh_hash, spec.geometry.ordered_momentum_mesh_sha256, "ordered momentum mesh"),
        (
            orbital_hash,
            source.ordered_orbitals_sha256,
            "ordered orbital-index descriptor",
        ),
        (
            orbital_descriptor_fingerprint,
            source.ordered_orbitals_descriptor_fingerprint,
            "ordered-orbitals descriptor fingerprint",
        ),
        (
            active_band_states_hash,
            source.active_band_states_sha256,
            "active-band states",
        ),
        (energy_hash, source.ordered_energies_sha256, "ordered energies"),
        (occupation_hash, source.ordered_occupations_sha256, "ordered occupations"),
        (projector_hash, source.ordered_projector_sha256, "ordered projector"),
        (fock_hash, source.ordered_fock_sha256, "ordered Fock"),
        (h0_hash, source.h0_sha256, "ordered h0"),
        (
            interaction_h_hash,
            source.interaction_h_sha256,
            "ordered interaction_h",
        ),
    )
    for actual, expected, label in expected_hashes:
        if actual != expected:
            raise ValueError(f"canonical {label} hash mismatch")
    reconstructed_source_state = _fingerprint(
        {
            "ordered_orbitals_sha256": orbital_hash,
            "ordered_orbitals_descriptor_label": (
                ORBITAL_INDEX_DESCRIPTOR_LABEL
            ),
            "ordered_orbitals_schema_label": (
                ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL
            ),
            "ordered_orbitals_schema_fingerprint": (
                ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT
            ),
            "ordered_orbitals_descriptor_fingerprint": (
                orbital_descriptor_fingerprint
            ),
            "ordered_energies_sha256": energy_hash,
            "ordered_occupations_sha256": occupation_hash,
            "ordered_projector_sha256": projector_hash,
            "ordered_fock_sha256": fock_hash,
            "h0_sha256": h0_hash,
            "interaction_h_sha256": interaction_h_hash,
            "active_band_states_sha256": active_band_states_hash,
            "active_band_states_layout": source.active_band_states_layout,
            "active_band_states_valley_order": (
                source.active_band_states_valley_order
            ),
            "active_band_states_gauge_scope": (
                source.active_band_states_gauge_scope
            ),
            "replay_loader_implementation_fingerprint": (
                source.replay_loader_implementation_fingerprint
            ),
            "replay_payload_schema_fingerprint": (
                source.replay_payload_schema_fingerprint
            ),
            "canonical_basis_kind": CANONICAL_BASIS_KIND,
            "residual_norm": REPLAY_RESIDUAL_NORM,
            "fock_decomposition_convention": FOCK_DECOMPOSITION_CONVENTION,
            "geometry_receipt_fingerprint": spec.geometry.fingerprint,
            "ensemble_receipt_fingerprint": spec.ensemble.fingerprint,
            "source_commit": source.source_commit,
            "source_artifact_sha256": source.source_artifact_sha256,
        }
    )
    if reconstructed_source_state != source.source_state_sha256:
        raise ValueError("reconstructed source-state context hash mismatch")
    if payload.source_state_sha256 != reconstructed_source_state:
        raise ValueError("replay payload source-state fingerprint mismatch")

    decomposition = _max_abs(payload.fock - (payload.h0 + payload.interaction_h))
    h0_hermiticity = _max_abs(
        payload.h0 - np.swapaxes(payload.h0.conj(), 0, 1)
    )
    interaction_h_hermiticity = _max_abs(
        payload.interaction_h
        - np.swapaxes(payload.interaction_h.conj(), 0, 1)
    )
    active_band_state_norm = _max_abs(
        np.sum(np.abs(payload.active_band_states) ** 2, axis=1) - 1.0
    )
    projector_hermiticity = _max_abs(
        payload.projector - np.swapaxes(payload.projector.conj(), 0, 1)
    )
    projector_idempotency = _max_abs(
        _matrix_product(payload.projector, payload.projector) - payload.projector
    )
    fock_hermiticity = _max_abs(
        payload.fock - np.swapaxes(payload.fock.conj(), 0, 1)
    )
    commutator = _max_abs(
        _matrix_product(payload.fock, payload.projector)
        - _matrix_product(payload.projector, payload.fock)
    )
    occupation_diagonal = _diagonal_matrix(payload.occupations)
    energy_diagonal = _diagonal_matrix(payload.energies)
    projector_vs_occupation = _max_abs(payload.projector - occupation_diagonal)
    canonical_basis_diagonal_closure = _max_abs(payload.fock - energy_diagonal)

    dedicated_residuals = (
        (
            decomposition,
            source.fock_decomposition_residual_ev,
            source.fock_decomposition_tolerance_ev,
            "Fock decomposition",
        ),
        (
            h0_hermiticity,
            source.h0_hermiticity_residual_ev,
            source.h0_hermiticity_tolerance_ev,
            "h0 Hermiticity",
        ),
        (
            interaction_h_hermiticity,
            source.interaction_h_hermiticity_residual_ev,
            source.interaction_h_hermiticity_tolerance_ev,
            "interaction_h Hermiticity",
        ),
        (
            active_band_state_norm,
            source.active_band_state_norm_residual,
            source.active_band_state_norm_tolerance,
            "active-band state norm",
        ),
        (
            projector_hermiticity,
            source.projector_hermiticity_residual,
            source.projector_hermiticity_tolerance,
            "projector Hermiticity",
        ),
        (
            projector_idempotency,
            source.projector_idempotency_residual,
            source.projector_idempotency_tolerance,
            "projector idempotency",
        ),
        (
            fock_hermiticity,
            source.fock_hermiticity_residual_ev,
            source.fock_hermiticity_tolerance_ev,
            "Fock Hermiticity",
        ),
        (
            commutator,
            source.fock_projector_commutator_residual_ev,
            source.stationarity_tolerance_ev,
            "Fock/projector commutator",
        ),
        (
            projector_vs_occupation,
            source.projector_vs_occupation_residual,
            source.projector_vs_occupation_tolerance,
            "projector/occupation diagonal closure",
        ),
        (
            canonical_basis_diagonal_closure,
            source.fock_vs_diagonal_energy_residual_ev,
            source.fock_vs_diagonal_energy_tolerance_ev,
            "canonical-basis diagonal closure",
        ),
    )
    for actual, attested, tolerance, label in dedicated_residuals:
        _require_within(actual, tolerance, label)
        _require_receipt_match(actual, attested, label)

    occupied = payload.energies[payload.occupations == 1]
    unoccupied = payload.energies[payload.occupations == 0]
    if occupied.size < 1 or unoccupied.size < 1:
        raise ValueError("Aufbau replay requires both occupied and unoccupied orbitals")
    max_occupied = float(np.max(occupied))
    min_unoccupied = float(np.min(unoccupied))
    aufbau_gap = min_unoccupied - max_occupied
    aufbau_violation = max(0.0, -aufbau_gap)
    _require_within(aufbau_violation, source.aufbau_tolerance_ev, "Aufbau occupation")
    _require_receipt_match(
        aufbau_gap,
        source.aufbau_min_unoccupied_minus_max_occupied_ev,
        "Aufbau gap",
    )
    _require_receipt_match(
        aufbau_violation,
        source.aufbau_occupation_violation_ev,
        "Aufbau occupation",
    )
    mu = source.chemical_potential_ev
    mu_violation = max(0.0, max_occupied - mu, mu - min_unoccupied)
    _require_within(
        mu_violation,
        source.chemical_mu_occupation_tolerance_ev,
        "chemical-potential occupation closure",
    )
    _require_receipt_match(
        mu_violation,
        source.chemical_mu_occupation_residual_ev,
        "chemical-potential occupation closure",
    )

    selected_spin = source.selected_spin
    selected_indices = tuple(
        index
        for index, (_, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin == selected_spin
    )
    opposite_indices = tuple(
        index
        for index, (_, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin != selected_spin
    )
    selected_occupations = payload.occupations[np.asarray(selected_indices), :]
    selected_energies = payload.energies[np.asarray(selected_indices), :]
    selected_occupied_count = int(np.count_nonzero(selected_occupations == 1))
    selected_unoccupied_count = int(np.count_nonzero(selected_occupations == 0))
    selected_minimum = float(np.min(selected_energies))
    selected_maximum = float(np.max(selected_energies))
    metallicity = source.metallicity_evidence
    if selected_occupied_count != metallicity.selected_spin_occupied_state_count:
        raise ValueError("selected-spin occupied metallicity count mismatch")
    if selected_unoccupied_count != metallicity.selected_spin_unoccupied_state_count:
        raise ValueError("selected-spin unoccupied metallicity count mismatch")
    _require_scalar_match(
        selected_minimum, metallicity.selected_spin_band_min_ev, "selected-spin band minimum"
    )
    _require_scalar_match(
        selected_maximum, metallicity.selected_spin_band_max_ev, "selected-spin band maximum"
    )
    if not (
        selected_minimum < mu - metallicity.metallicity_tolerance_ev
        and selected_maximum > mu + metallicity.metallicity_tolerance_ev
    ):
        raise ValueError("selected-spin metallicity does not straddle chemical potential")

    valley_holes: dict[int, int] = {}
    for valley in (-1, 1):
        flavor_index = INTERNAL_FLAVOR_ORDER.index((valley, selected_spin))
        valley_holes[valley] = int(np.count_nonzero(payload.occupations[flavor_index] == 0))
    opposite_holes = int(
        np.count_nonzero(payload.occupations[np.asarray(opposite_indices), :] == 0)
    )
    if valley_holes[-1] != source.valley_minus_hole_count:
        raise ValueError("selected-spin valley -1 hole-count mismatch")
    if valley_holes[1] != source.valley_plus_hole_count:
        raise ValueError("selected-spin valley +1 hole-count mismatch")
    if valley_holes[-1] != valley_holes[1]:
        raise ValueError("selected-spin replay does not have equal valley hole counts")
    selected_holes = valley_holes[-1] + valley_holes[1]
    if selected_holes != source.selected_spin_hole_count:
        raise ValueError("selected-spin total hole-count mismatch")
    if opposite_holes != source.opposite_spin_hole_count:
        raise ValueError("opposite-spin hole-count mismatch")
    measured_density = -float(selected_holes) / source.area_angstrom_squared * 1.0e16
    density_residual = abs(measured_density - source.target_density_cm2)
    _require_within(density_residual, source.density_tolerance_cm2, "density closure")
    _require_scalar_match(measured_density, source.measured_density_cm2, "measured density")
    _require_receipt_match(density_residual, source.density_residual_cm2, "density closure")

    base_pockets: list[Vituri2024HalfMetalHFBasePocketReplayEvidence] = []
    for pocket_receipt in source.pocket_evidence:
        flavor_index = INTERNAL_FLAVOR_ORDER.index(
            (pocket_receipt.valley, selected_spin)
        )
        hole_mask = payload.occupations[flavor_index] == 0
        cardinalities = _component_cardinalities(hole_mask, spec.geometry.mesh_shape)
        hole_count = int(np.count_nonzero(hole_mask))
        if len(cardinalities) != pocket_receipt.hole_component_count:
            raise ValueError(
                f"valley {pocket_receipt.valley:+d} base-pocket component-count mismatch"
            )
        if hole_count != pocket_receipt.hole_state_count:
            raise ValueError(
                f"valley {pocket_receipt.valley:+d} base-pocket cardinality mismatch"
            )
        base_pockets.append(
            Vituri2024HalfMetalHFBasePocketReplayEvidence(
                valley=pocket_receipt.valley,
                selected_spin=selected_spin,
                hole_component_count=len(cardinalities),
                component_cardinalities=cardinalities,
                hole_state_count=hole_count,
                adjacency_convention="four_neighbor_finite_domain_no_wrap",
                attested_pocket_receipt_fingerprint=pocket_receipt.fingerprint,
            )
        )

    payload_manifest_hash = _fingerprint(
        {
            "provider_fingerprint": payload.provider_fingerprint,
            "source_commit": payload.source_commit,
            "source_artifact_sha256": payload.source_artifact_sha256,
            "spec_fingerprint": payload.spec_fingerprint,
            "source_state_sha256": reconstructed_source_state,
            "replay_loader_implementation_fingerprint": (
                payload.replay_loader_implementation_fingerprint
            ),
            "replay_payload_schema_fingerprint": (
                payload.replay_payload_schema_fingerprint
            ),
            "array_layout": REPLAY_ARRAY_LAYOUT,
            "array_conversion": REPLAY_ARRAY_CONVERSION,
            "orbital_order": REPLAY_ORBITAL_ORDER,
            "internal_flavor_order": INTERNAL_FLAVOR_ORDER,
            "ordered_orbitals_descriptor_label": (
                ORBITAL_INDEX_DESCRIPTOR_LABEL
            ),
            "ordered_orbitals_schema_label": (
                ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL
            ),
            "ordered_orbitals_schema_fingerprint": (
                ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT
            ),
            "active_band_states_layout": ACTIVE_BAND_STATES_LAYOUT,
            "active_band_states_valley_order": ACTIVE_BAND_STATES_VALLEY_ORDER,
            "active_band_states_gauge_scope": ACTIVE_BAND_STATES_GAUGE_SCOPE,
            "canonical_basis_kind": CANONICAL_BASIS_KIND,
            "residual_norm": REPLAY_RESIDUAL_NORM,
            "fock_decomposition_convention": FOCK_DECOMPOSITION_CONVENTION,
            "ordered_momentum_mesh_sha256": mesh_hash,
            "ordered_orbitals_sha256": orbital_hash,
            "ordered_orbitals_descriptor_fingerprint": (
                orbital_descriptor_fingerprint
            ),
            "active_band_states_sha256": active_band_states_hash,
            "ordered_energies_sha256": energy_hash,
            "ordered_occupations_sha256": occupation_hash,
            "ordered_projector_sha256": projector_hash,
            "ordered_fock_sha256": fock_hash,
            "h0_sha256": h0_hash,
            "interaction_h_sha256": interaction_h_hash,
        }
    )
    hashes = Vituri2024HalfMetalHFReplayHashes(
        ordered_momentum_mesh_sha256=mesh_hash,
        ordered_orbitals_sha256=orbital_hash,
        ordered_orbitals_descriptor_fingerprint=(
            orbital_descriptor_fingerprint
        ),
        active_band_states_sha256=active_band_states_hash,
        ordered_energies_sha256=energy_hash,
        ordered_occupations_sha256=occupation_hash,
        ordered_projector_sha256=projector_hash,
        ordered_fock_sha256=fock_hash,
        h0_sha256=h0_hash,
        interaction_h_sha256=interaction_h_hash,
        reconstructed_source_state_sha256=reconstructed_source_state,
        payload_manifest_sha256=payload_manifest_hash,
    )
    residuals = Vituri2024HalfMetalHFReplayResiduals(
        residual_norm=REPLAY_RESIDUAL_NORM,
        fock_decomposition_max_abs_ev=decomposition,
        fock_decomposition_tolerance_ev=source.fock_decomposition_tolerance_ev,
        h0_hermiticity_max_abs_ev=h0_hermiticity,
        h0_hermiticity_tolerance_ev=source.h0_hermiticity_tolerance_ev,
        interaction_h_hermiticity_max_abs_ev=interaction_h_hermiticity,
        interaction_h_hermiticity_tolerance_ev=(
            source.interaction_h_hermiticity_tolerance_ev
        ),
        active_band_state_norm_max_abs=active_band_state_norm,
        active_band_state_norm_tolerance=source.active_band_state_norm_tolerance,
        projector_hermiticity_max_abs=projector_hermiticity,
        projector_hermiticity_tolerance=source.projector_hermiticity_tolerance,
        projector_idempotency_max_abs=projector_idempotency,
        projector_idempotency_tolerance=source.projector_idempotency_tolerance,
        fock_hermiticity_max_abs_ev=fock_hermiticity,
        fock_hermiticity_tolerance_ev=source.fock_hermiticity_tolerance_ev,
        fock_projector_commutator_max_abs_ev=commutator,
        fock_projector_commutator_tolerance_ev=source.stationarity_tolerance_ev,
        projector_vs_occupation_max_abs=projector_vs_occupation,
        projector_vs_occupation_tolerance=(
            source.projector_vs_occupation_tolerance
        ),
        canonical_basis_diagonal_closure_max_abs_ev=(
            canonical_basis_diagonal_closure
        ),
        canonical_basis_diagonal_closure_tolerance_ev=(
            source.fock_vs_diagonal_energy_tolerance_ev
        ),
        aufbau_gap_ev=aufbau_gap,
        aufbau_violation_ev=aufbau_violation,
        aufbau_tolerance_ev=source.aufbau_tolerance_ev,
        chemical_mu_occupation_violation_ev=mu_violation,
        chemical_mu_occupation_tolerance_ev=(
            source.chemical_mu_occupation_tolerance_ev
        ),
    )
    occupation_evidence = Vituri2024HalfMetalHFReplayOccupationEvidence(
        chemical_potential_ev=mu,
        selected_spin=selected_spin,
        selected_spin_occupied_state_count=selected_occupied_count,
        selected_spin_unoccupied_state_count=selected_unoccupied_count,
        selected_spin_band_min_ev=selected_minimum,
        selected_spin_band_max_ev=selected_maximum,
        valley_minus_hole_count=valley_holes[-1],
        valley_plus_hole_count=valley_holes[1],
        selected_spin_hole_count=selected_holes,
        opposite_spin_hole_count=opposite_holes,
        measured_density_cm2=measured_density,
        target_density_cm2=source.target_density_cm2,
        density_residual_cm2=density_residual,
        density_tolerance_cm2=source.density_tolerance_cm2,
    )
    return Vituri2024HalfMetalHFReplayReceipt(
        provider_fingerprint=provider_values_before["provider_fingerprint"],
        source_commit=provider_values_before["source_commit"],
        source_artifact_sha256=provider_values_before["source_artifact_sha256"],
        spec_fingerprint=provider_values_before["spec_fingerprint"],
        attested_source_receipt_fingerprint=source.fingerprint,
        replay_loader_implementation_fingerprint=(
            source.replay_loader_implementation_fingerprint
        ),
        replay_payload_schema_fingerprint=(
            source.replay_payload_schema_fingerprint
        ),
        internal_flavor_order=INTERNAL_FLAVOR_ORDER,
        array_layout=REPLAY_ARRAY_LAYOUT,
        array_conversion=REPLAY_ARRAY_CONVERSION,
        orbital_order=REPLAY_ORBITAL_ORDER,
        ordered_orbitals_descriptor_label=ORBITAL_INDEX_DESCRIPTOR_LABEL,
        ordered_orbitals_schema_label=ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL,
        ordered_orbitals_schema_fingerprint=(
            ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT
        ),
        active_band_states_layout=ACTIVE_BAND_STATES_LAYOUT,
        active_band_states_valley_order=ACTIVE_BAND_STATES_VALLEY_ORDER,
        active_band_states_gauge_scope=ACTIVE_BAND_STATES_GAUGE_SCOPE,
        canonical_basis_kind=CANONICAL_BASIS_KIND,
        residual_norm=REPLAY_RESIDUAL_NORM,
        fock_decomposition_convention=FOCK_DECOMPOSITION_CONVENTION,
        hashes=hashes,
        residuals=residuals,
        occupation_evidence=occupation_evidence,
        base_pocket_evidence=(base_pockets[0], base_pockets[1]),
        status=Vituri2024HalfMetalHFReplayStatus(
            _factory_token=_REPLAY_FACTORY_TOKEN
        ),
        _factory_token=_REPLAY_FACTORY_TOKEN,
    )


__all__ = [
    "ACTIVE_BAND_STATES_GAUGE_SCOPE",
    "ACTIVE_BAND_STATES_LAYOUT",
    "ACTIVE_BAND_STATES_VALLEY_ORDER",
    "CANONICAL_BASIS_KIND",
    "FOCK_DECOMPOSITION_CONVENTION",
    "INTERNAL_FLAVOR_ORDER",
    "ORBITAL_INDEX_DESCRIPTOR_LABEL",
    "ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT",
    "ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL",
    "REPLAY_ARRAY_CONVERSION",
    "REPLAY_ARRAY_LAYOUT",
    "REPLAY_ORBITAL_ORDER",
    "REPLAY_PAYLOAD_SCHEMA_FINGERPRINT",
    "REPLAY_RESIDUAL_NORM",
    "Vituri2024HalfMetalHFBasePocketReplayEvidence",
    "Vituri2024HalfMetalHFReplayHashes",
    "Vituri2024HalfMetalHFReplayOccupationEvidence",
    "Vituri2024HalfMetalHFReplayPayload",
    "Vituri2024HalfMetalHFReplayProviderProtocol",
    "Vituri2024HalfMetalHFReplayReceipt",
    "Vituri2024HalfMetalHFReplayResiduals",
    "Vituri2024HalfMetalHFReplayStatus",
    "canonical_array_sha256",
    "canonical_orbital_order_sha256",
    "replay_vituri2024_half_metal_hf_arrays",
]
