"""Detached, authority-safe local functional probes for Vituri-2024 HF.

This module replays only pre-registered local affine ``E -> F`` identities and
finite-domain signed-q ``F -> dF`` probes.  The direct displaced-Fock evidence
uses nonce-bound, typed dependency traces.  Those traces are honest-provider
self-reports: they detect truthful delegation and accidental dependency drift,
but they are not a proof against hostile provider code.

A successful receipt is local only.  It does not verify a global functional
chain, an SCF trajectory, branch selection, pocket refinement, a reciprocal
q torus, scientific execution, or paper reproduction.
"""
from __future__ import annotations

import ast
from dataclasses import InitVar, asdict, dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .vituri2024_hf_preflight import (
    FINITE_Q_HESSIAN_NORMALIZATION,
    FIXED_DENSITY_DIRECTION_CONVENTION,
    FOCK_FIRST_DERIVATIVE_NORMALIZATION,
    FOCK_OUTPUT_CONVENTION,
    STORED_DENSITY_PAIRING,
    VITURI2024_BASE_PROVIDER_METADATA_FIELDS,
    Vituri2024HalfMetalHFProviderBinding,
    Vituri2024HalfMetalHFSpec,
)
from .vituri2024_hf_replay import (
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
    REPLAY_RESIDUAL_NORM,
    Vituri2024HalfMetalHFReplayPayload,
    Vituri2024HalfMetalHFReplayProviderProtocol,
    Vituri2024HalfMetalHFReplayReceipt,
    canonical_array_sha256,
    replay_vituri2024_half_metal_hf_arrays,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

FUNCTIONAL_REPLAY_SCOPE = "vituri2024_local_registered_functional_probes_v2"
FUNCTIONAL_REPLAY_PAYLOAD_SCHEMA_LABEL = "vituri2024_functional_probe_payload_v2"
FUNCTIONAL_REPLAY_ABI_LABEL = "vituri2024_functional_replay_abi_v2"
SIGNED_Q_SIGN_ORDER: tuple[str, str] = ("+q", "-q")
SIGNED_Q_CREATION_ANNIHILATION_CONVENTION = (
    "lane_sign_is_creation_momentum_minus_annihilation_momentum"
)
SIGNED_Q_CHART_KIND = (
    "complexified_independent_signed_blocks_finite_domain_no_wrap_no_carry"
)
DIRECT_DISPLACED_FOCK_CONSTRUCTION = (
    "source_closed_direct_interaction_and_full_fock_builders_no_hessian_call"
)
AFFINE_ANCHOR_LABELS: tuple[str, ...] = (
    "source",
    "imaginary_offdiagonal_coherence_shift",
    "dense_complex_hermitian_shift",
)
Q0_PROBE_LABELS: tuple[str, ...] = (
    "diagonal_flavor",
    "momentum_redistribution",
    "real_coherence",
    "imaginary_coherence",
    "deterministic_dense_mixed",
)
Q_CHART_LABELS: tuple[str, str] = (
    "horizontal_nonzero_no_wrap",
    "vertical_nonzero_no_wrap",
)
SIGNED_Q_PROBE_LABELS: tuple[str, ...] = tuple(
    f"{q_label}:{probe_label}"
    for q_label in Q_CHART_LABELS
    for probe_label in (
        "plus_only_boundary_edge",
        "minus_only_boundary_edge",
        "mixed_interior",
    )
)

_REGISTERED_ABSOLUTE_TOLERANCE = 1.0e-9
_REGISTERED_RELATIVE_TOLERANCE = 1.0e-8
_REGISTERED_ROUNDOFF_ULPS = 512.0
_REGISTERED_SLOPE_STABILITY_TOLERANCE = 5.0e-8
_REGISTERED_INFORMATIVENESS_FLOOR = 1.0e-10
_REGISTERED_SOURCE_MESH_Q_COORDINATE_TOLERANCE_INVERSE_ANGSTROM = 1.0e-12


def functional_replay_module_ast_manifest_sha256(source: str | None = None) -> str:
    """Hash the canonical full-module AST, excluding source locations/comments."""

    module_source = (
        Path(__file__).read_text(encoding="utf-8") if source is None else source
    )
    if not isinstance(module_source, str):
        raise TypeError("functional replay module source must be text")
    tree = ast.parse(module_source, filename="vituri2024_hf_functional_replay.py")
    canonical_ast = ast.dump(
        tree,
        annotate_fields=True,
        include_attributes=False,
    )
    return hashlib.sha256(canonical_ast.encode("utf-8")).hexdigest()


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _commit(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase 40- or 64-character commit")
    return value


def _strict_int(value: object, label: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a strict integer")
    return int(value)


def _finite(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a strict real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _immutable_array(
    value: object,
    *,
    label: str,
    dtype: np.dtype[object],
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a numpy.ndarray")
    if value.dtype != dtype:
        raise TypeError(f"{label} dtype must be exactly {dtype.name}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{label} shape must be exactly {shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must contain only finite values")
    result = np.frombuffer(value.tobytes(order="C"), dtype=dtype).reshape(value.shape)
    result.flags.writeable = False
    return result


def _array_manifest(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": canonical_array_sha256(array),
    }


def affine_anchor_inventory_sha256(
    labels: tuple[str, ...], offsets: ComplexArray
) -> str:
    if type(labels) is not tuple or labels != AFFINE_ANCHOR_LABELS:
        raise ValueError("affine anchor labels/inventory changed")
    if not isinstance(offsets, np.ndarray):
        raise TypeError("affine anchor offsets must be a numpy.ndarray")
    return _fingerprint(
        {
            "schema": "vituri2024_affine_anchor_inventory_v1",
            "labels": labels,
            "density_direction_convention": FIXED_DENSITY_DIRECTION_CONVENTION,
            "offsets": _array_manifest(offsets),
        }
    )


def q0_probe_inventory_sha256(
    labels: tuple[str, ...], directions: ComplexArray
) -> str:
    if type(labels) is not tuple or labels != Q0_PROBE_LABELS:
        raise ValueError("q0 probe labels/inventory changed")
    if not isinstance(directions, np.ndarray):
        raise TypeError("q0 directions must be a numpy.ndarray")
    return _fingerprint(
        {
            "schema": "vituri2024_q0_probe_inventory_v1",
            "labels": labels,
            "normalization": FOCK_FIRST_DERIVATIVE_NORMALIZATION,
            "directions": _array_manifest(directions),
        }
    )


def signed_q_probe_inventory_sha256(
    labels: tuple[str, ...], probes: ComplexArray, q_probe_indices: IntegerArray
) -> str:
    if type(labels) is not tuple or labels != SIGNED_Q_PROBE_LABELS:
        raise ValueError("signed-q probe labels/inventory changed")
    if not isinstance(probes, np.ndarray) or not isinstance(
        q_probe_indices, np.ndarray
    ):
        raise TypeError("signed-q inventory arrays must be numpy.ndarray objects")
    return _fingerprint(
        {
            "schema": "vituri2024_signed_q_probe_inventory_v2",
            "labels": labels,
            "sign_order": SIGNED_Q_SIGN_ORDER,
            "normalization": FINITE_Q_HESSIAN_NORMALIZATION,
            "probes": _array_manifest(probes),
            "q_probe_indices": _array_manifest(q_probe_indices),
        }
    )


@dataclass(frozen=True, slots=True)
class Vituri2024SignedQProbeChart:
    """One typed nonzero finite-domain q chart with independent signed lanes."""

    q_probe_index: int
    q_label: str
    mesh_shape: tuple[int, int]
    mesh_displacement: IntegerArray
    cartesian_q: FloatArray
    source_k_indices: IntegerArray
    target_maps: IntegerArray
    validity_masks: BoolArray
    reverse_edge_map: IntegerArray
    sign_order: tuple[str, str] = SIGNED_Q_SIGN_ORDER
    creation_annihilation_convention: str = (
        SIGNED_Q_CREATION_ANNIHILATION_CONVENTION
    )
    chart_kind: str = SIGNED_Q_CHART_KIND

    def __post_init__(self) -> None:
        index = _strict_int(self.q_probe_index, "signed-q probe index")
        if index < 0:
            raise ValueError("signed-q probe index must be non-negative")
        _text(self.q_label, "signed-q label")
        if (
            type(self.mesh_shape) is not tuple
            or len(self.mesh_shape) != 2
            or any(type(value) is not int or value < 2 for value in self.mesh_shape)
        ):
            raise ValueError("signed-q mesh shape must contain two dimensions >=2")
        nk = self.mesh_shape[0] * self.mesh_shape[1]
        displacement = _immutable_array(
            self.mesh_displacement,
            label="signed-q integer mesh displacement",
            dtype=np.dtype(np.int64),
            shape=(2,),
        )
        if not np.any(displacement != 0):
            raise ValueError("signed-q chart excludes q=0")
        cartesian_q = _immutable_array(
            self.cartesian_q,
            label="signed-q Cartesian q",
            dtype=np.dtype(np.float64),
            shape=(2,),
        )
        if not np.any(cartesian_q != 0.0):
            raise ValueError("signed-q Cartesian q must be nonzero")
        source = _immutable_array(
            self.source_k_indices,
            label="signed-q source-k inventory",
            dtype=np.dtype(np.int64),
            shape=(nk,),
        )
        targets = _immutable_array(
            self.target_maps,
            label="signed-q target maps",
            dtype=np.dtype(np.int64),
            shape=(2, nk),
        )
        masks = _immutable_array(
            self.validity_masks,
            label="signed-q validity masks",
            dtype=np.dtype(np.bool_),
            shape=(2, nk),
        )
        reverse = _immutable_array(
            self.reverse_edge_map,
            label="signed-q reverse-edge map",
            dtype=np.dtype(np.int64),
            shape=(2, nk),
        )
        if not np.array_equal(source, np.arange(nk, dtype=np.int64)):
            raise ValueError("signed-q source-k inventory must be exact row-major arange")
        rows, columns = self.mesh_shape
        expected_targets = np.full((2, nk), -1, dtype=np.int64)
        expected_masks = np.zeros((2, nk), dtype=np.bool_)
        delta_row, delta_column = (int(displacement[0]), int(displacement[1]))
        for sign_index, multiplier in enumerate((1, -1)):
            for source_index in range(nk):
                row, column = divmod(source_index, columns)
                target_row = row + multiplier * delta_row
                target_column = column + multiplier * delta_column
                valid = 0 <= target_row < rows and 0 <= target_column < columns
                expected_masks[sign_index, source_index] = valid
                if valid:
                    expected_targets[sign_index, source_index] = (
                        target_row * columns + target_column
                    )
        if not np.array_equal(masks, expected_masks) or not np.array_equal(
            targets, expected_targets
        ):
            raise ValueError("signed-q maps must use exact no-wrap/no-carry targets")
        if not np.array_equal(reverse, expected_targets):
            raise ValueError("signed-q reverse-edge map does not bind the target edge")
        if not np.all(np.any(masks, axis=1)):
            raise ValueError("signed-q chart requires valid edges in both lanes")
        for sign_index in range(2):
            opposite = 1 - sign_index
            for source_index in np.flatnonzero(masks[sign_index]):
                target_index = int(targets[sign_index, source_index])
                if (
                    not masks[opposite, target_index]
                    or int(targets[opposite, target_index]) != int(source_index)
                ):
                    raise ValueError("signed-q reverse-edge involution does not close")
        both_valid = masks[0] & masks[1]
        if np.any(targets[0, both_valid] == targets[1, both_valid]):
            raise ValueError("signed-q chart excludes self-conjugate q aliases")
        if self.sign_order != SIGNED_Q_SIGN_ORDER:
            raise ValueError("signed-q sign order must be exactly (+q,-q)")
        if (
            self.creation_annihilation_convention
            != SIGNED_Q_CREATION_ANNIHILATION_CONVENTION
            or self.chart_kind != SIGNED_Q_CHART_KIND
        ):
            raise ValueError("signed-q chart convention changed")
        object.__setattr__(self, "q_probe_index", index)
        for name, array in (
            ("mesh_displacement", displacement),
            ("cartesian_q", cartesian_q),
            ("source_k_indices", source),
            ("target_maps", targets),
            ("validity_masks", masks),
            ("reverse_edge_map", reverse),
        ):
            object.__setattr__(self, name, array)

    @property
    def fingerprint(self) -> str:
        return _single_q_chart_fingerprint(self)


def _single_q_chart_fingerprint(chart: Vituri2024SignedQProbeChart) -> str:
    return _fingerprint(
        {
            "schema": "vituri2024_signed_q_chart_v2",
            "q_probe_index": chart.q_probe_index,
            "q_label": chart.q_label,
            "mesh_shape": chart.mesh_shape,
            "mesh_displacement": _array_manifest(chart.mesh_displacement),
            "cartesian_q": _array_manifest(chart.cartesian_q),
            "source_k_indices": _array_manifest(chart.source_k_indices),
            "target_maps": _array_manifest(chart.target_maps),
            "validity_masks": _array_manifest(chart.validity_masks),
            "reverse_edge_map": _array_manifest(chart.reverse_edge_map),
            "sign_order": chart.sign_order,
            "creation_annihilation_convention": chart.creation_annihilation_convention,
            "chart_kind": chart.chart_kind,
        }
    )


def signed_q_chart_inventory_sha256(
    charts: Vituri2024SignedQProbeChart | tuple[Vituri2024SignedQProbeChart, ...],
) -> str:
    """Hash one chart for compatibility or the complete registered chart tuple."""

    if type(charts) is Vituri2024SignedQProbeChart:
        return _single_q_chart_fingerprint(charts)
    if type(charts) is not tuple or not charts or any(
        type(chart) is not Vituri2024SignedQProbeChart for chart in charts
    ):
        raise TypeError("q chart inventory requires typed signed-q chart(s)")
    return _fingerprint(
        {
            "schema": "vituri2024_signed_q_chart_inventory_v2",
            "chart_fingerprints": tuple(chart.fingerprint for chart in charts),
        }
    )


FUNCTIONAL_REPLAY_PAYLOAD_SCHEMA_FINGERPRINT = _fingerprint(
    {
        "schema_label": FUNCTIONAL_REPLAY_PAYLOAD_SCHEMA_LABEL,
        "anchor_layout": "anchor_internal_flavor_internal_flavor_k",
        "q0_layout": "probe_internal_flavor_internal_flavor_k",
        "signed_q_layout": "probe_sign_internal_flavor_internal_flavor_k",
        "anchor_labels": AFFINE_ANCHOR_LABELS,
        "q0_labels": Q0_PROBE_LABELS,
        "q_chart_labels": Q_CHART_LABELS,
        "signed_q_labels": SIGNED_Q_PROBE_LABELS,
        "sign_order": SIGNED_Q_SIGN_ORDER,
        "chart_kind": SIGNED_Q_CHART_KIND,
        "normalizations": (
            FOCK_FIRST_DERIVATIVE_NORMALIZATION,
            FINITE_Q_HESSIAN_NORMALIZATION,
        ),
    }
)
FUNCTIONAL_REPLAY_ABI_FINGERPRINT = _fingerprint(
    {
        "abi_label": FUNCTIONAL_REPLAY_ABI_LABEL,
        "provider_methods": (
            "load_half_metal_replay_payload(source_artifact_sha256)",
            "load_functional_replay_payload(source_artifact_sha256)",
            "evaluate_scalar_energy(interaction_h,h0,density)",
            "evaluate_fock_derivative(density)",
            "evaluate_finite_q_hessian(perturbation,q_index,chart_arrays)",
            "evaluate_displaced_fock(density,displacement,q_index,chart_arrays,nonce)->typed_response_trace",
        ),
        "fock_output": FOCK_OUTPUT_CONVENTION,
        "stored_density_pairing": STORED_DENSITY_PAIRING,
        "density_direction_convention": FIXED_DENSITY_DIRECTION_CONVENTION,
        "sign_order": SIGNED_Q_SIGN_ORDER,
        "creation_annihilation_convention": SIGNED_Q_CREATION_ANNIHILATION_CONVENTION,
        "honest_provider_trace_not_hostile_code_proof": True,
        "no_q_lane_inference_or_symmetrization": True,
    }
)
FUNCTIONAL_REPLAY_VERIFIER_IMPLEMENTATION_SCHEMA_FINGERPRINT = _fingerprint(
    {
        "schema": "vituri2024_functional_replay_verifier_implementation_schema_v3",
        "source_q_binding": {
            "chart_mesh_shape": "exactly_spec.geometry.mesh_shape",
            "edge_coordinate_identity": (
                "source_payload.mesh[target]-source_payload.mesh[source]"
                "==lane_sign_times_cartesian_q"
            ),
            "coordinate_units": "inverse_angstrom",
            "coordinate_tolerance_field": (
                "choice.source_mesh_q_coordinate_tolerance_inverse_angstrom"
            ),
            "coordinate_residual_norm": "entrywise_max_abs",
            "valid_edges_only": True,
        },
        "source_anchor_bounds": {
            "fock_terms": "max_entry_abs_F_eval_plus_abs_F_saved",
            "interaction_terms": (
                "max_entry_abs_F_eval_plus_abs_h0_plus_abs_interaction_saved"
            ),
            "energy_terms": "abs_E_eval_plus_abs_E_selected",
            "operation_counts": {"fock": 2, "interaction": 3, "energy": 2},
            "bound_formula": (
                "absolute_tolerance+relative_tolerance*max(result_scale,floor)"
                "+gamma(operation_count+roundoff_ulps)*max(termwise,floor)"
            ),
            "separate_result_scale_termwise_operation_roundoff_bound": True,
        },
        "detached_approval_bindings": (
            "choice_fingerprint",
            "verifier_implementation_schema_fingerprint",
        ),
    }
)


def direct_builder_dependency_archive_fingerprint(
    *,
    source_commit: str,
    source_artifact_sha256: str,
    direct_displaced_fock_implementation_fingerprint: str,
    interaction_builder_implementation_fingerprint: str,
    full_fock_builder_implementation_fingerprint: str,
) -> str:
    return _fingerprint(
        {
            "schema": "vituri2024_direct_builder_dependency_archive_v1",
            "source_commit": _commit(source_commit, "direct archive source commit"),
            "source_artifact_sha256": _sha256(
                source_artifact_sha256, "direct archive source artifact"
            ),
            "direct_displaced_fock_implementation_fingerprint": _sha256(
                direct_displaced_fock_implementation_fingerprint,
                "direct displaced-Fock implementation",
            ),
            "interaction_builder_implementation_fingerprint": _sha256(
                interaction_builder_implementation_fingerprint,
                "direct interaction builder implementation",
            ),
            "full_fock_builder_implementation_fingerprint": _sha256(
                full_fock_builder_implementation_fingerprint,
                "direct full-Fock builder implementation",
            ),
            "construction": DIRECT_DISPLACED_FOCK_CONSTRUCTION,
            "required_trace": {
                "interaction_builder_call_count_min": 1,
                "full_fock_builder_call_count_min": 1,
                "finite_q_hessian_call_count_exact": 0,
                "target_and_reverse_maps_read": True,
                "caller_nonce_bound": True,
            },
        }
    )


def functional_provider_fingerprint(
    *,
    base_provider_fingerprint: str,
    functional_replay_abi_fingerprint: str,
    functional_replay_payload_schema_fingerprint: str,
    functional_probe_loader_implementation_fingerprint: str,
    direct_displaced_fock_implementation_fingerprint: str,
    direct_builder_dependency_archive_fingerprint: str,
) -> str:
    return _fingerprint(
        {
            "base_provider_fingerprint": _sha256(
                base_provider_fingerprint, "base provider fingerprint"
            ),
            "functional_replay_abi_fingerprint": _sha256(
                functional_replay_abi_fingerprint, "functional replay ABI"
            ),
            "functional_replay_payload_schema_fingerprint": _sha256(
                functional_replay_payload_schema_fingerprint,
                "functional replay payload schema",
            ),
            "functional_probe_loader_implementation_fingerprint": _sha256(
                functional_probe_loader_implementation_fingerprint,
                "functional probe loader implementation",
            ),
            "direct_displaced_fock_implementation_fingerprint": _sha256(
                direct_displaced_fock_implementation_fingerprint,
                "direct displaced-Fock implementation",
            ),
            "direct_builder_dependency_archive_fingerprint": _sha256(
                direct_builder_dependency_archive_fingerprint,
                "direct builder dependency archive",
            ),
        }
    )


def expected_array_payload_manifest_sha256(spec: Vituri2024HalfMetalHFSpec) -> str:
    """Derive the array manifest before execution from attested receipt identities."""

    if type(spec) is not Vituri2024HalfMetalHFSpec:
        raise TypeError("expected array manifest requires a typed HF spec")
    spec.require_receipt_set_complete()
    assert spec.geometry is not None
    assert spec.attested_source is not None
    source = spec.attested_source
    return _fingerprint(
        {
            "provider_fingerprint": source.provider_fingerprint,
            "source_commit": source.source_commit,
            "source_artifact_sha256": source.source_artifact_sha256,
            "spec_fingerprint": spec.fingerprint,
            "source_state_sha256": source.source_state_sha256,
            "replay_loader_implementation_fingerprint": (
                source.replay_loader_implementation_fingerprint
            ),
            "replay_payload_schema_fingerprint": source.replay_payload_schema_fingerprint,
            "array_layout": REPLAY_ARRAY_LAYOUT,
            "array_conversion": REPLAY_ARRAY_CONVERSION,
            "orbital_order": REPLAY_ORBITAL_ORDER,
            "internal_flavor_order": INTERNAL_FLAVOR_ORDER,
            "ordered_orbitals_descriptor_label": ORBITAL_INDEX_DESCRIPTOR_LABEL,
            "ordered_orbitals_schema_label": ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL,
            "ordered_orbitals_schema_fingerprint": (
                ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT
            ),
            "active_band_states_layout": ACTIVE_BAND_STATES_LAYOUT,
            "active_band_states_valley_order": ACTIVE_BAND_STATES_VALLEY_ORDER,
            "active_band_states_gauge_scope": ACTIVE_BAND_STATES_GAUGE_SCOPE,
            "canonical_basis_kind": CANONICAL_BASIS_KIND,
            "residual_norm": REPLAY_RESIDUAL_NORM,
            "fock_decomposition_convention": FOCK_DECOMPOSITION_CONVENTION,
            "ordered_momentum_mesh_sha256": source.ordered_momentum_mesh_sha256,
            "ordered_orbitals_sha256": source.ordered_orbitals_sha256,
            "ordered_orbitals_descriptor_fingerprint": (
                source.ordered_orbitals_descriptor_fingerprint
            ),
            "active_band_states_sha256": source.active_band_states_sha256,
            "ordered_energies_sha256": source.ordered_energies_sha256,
            "ordered_occupations_sha256": source.ordered_occupations_sha256,
            "ordered_projector_sha256": source.ordered_projector_sha256,
            "ordered_fock_sha256": source.ordered_fock_sha256,
            "h0_sha256": source.h0_sha256,
            "interaction_h_sha256": source.interaction_h_sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayPayload:
    """Immutable source-bound registration of anchors, probes, and q charts."""

    provider_fingerprint: str
    functional_provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    source_state_sha256: str
    functional_probe_loader_implementation_fingerprint: str
    functional_replay_payload_schema_fingerprint: str
    affine_anchor_labels: tuple[str, ...]
    affine_anchor_offsets: ComplexArray
    q0_labels: tuple[str, ...]
    q0_directions: ComplexArray
    signed_q_labels: tuple[str, ...]
    signed_q_probes: ComplexArray
    signed_q_probe_indices: IntegerArray
    q_charts: tuple[Vituri2024SignedQProbeChart, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.provider_fingerprint, "payload base provider"),
            (self.functional_provider_fingerprint, "payload functional provider"),
            (self.source_artifact_sha256, "payload source artifact"),
            (self.spec_fingerprint, "payload spec"),
            (self.source_state_sha256, "payload source state"),
            (
                self.functional_probe_loader_implementation_fingerprint,
                "payload functional probe loader",
            ),
            (
                self.functional_replay_payload_schema_fingerprint,
                "payload functional schema",
            ),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "payload source commit")
        if (
            self.functional_replay_payload_schema_fingerprint
            != FUNCTIONAL_REPLAY_PAYLOAD_SCHEMA_FINGERPRINT
        ):
            raise ValueError("functional replay payload schema fingerprint mismatch")
        if type(self.q_charts) is not tuple or len(self.q_charts) != len(Q_CHART_LABELS):
            raise ValueError("functional replay payload requires exactly two q charts")
        if any(type(chart) is not Vituri2024SignedQProbeChart for chart in self.q_charts):
            raise TypeError("functional replay payload requires typed signed-q charts")
        if tuple(chart.q_probe_index for chart in self.q_charts) != tuple(
            range(len(Q_CHART_LABELS))
        ) or tuple(chart.q_label for chart in self.q_charts) != Q_CHART_LABELS:
            raise ValueError("q chart indices/labels changed")
        if len({chart.fingerprint for chart in self.q_charts}) != len(self.q_charts):
            raise ValueError("q charts must be distinct")
        if np.array_equal(
            self.q_charts[0].mesh_displacement,
            self.q_charts[1].mesh_displacement,
        ) or np.array_equal(
            self.q_charts[0].cartesian_q,
            self.q_charts[1].cartesian_q,
        ):
            raise ValueError("registered q charts require distinct displacement and Cartesian q")
        mesh_shapes = {chart.mesh_shape for chart in self.q_charts}
        if len(mesh_shapes) != 1:
            raise ValueError("all q charts must bind one source mesh shape")
        nk = self.q_charts[0].mesh_shape[0] * self.q_charts[0].mesh_shape[1]

        if self.affine_anchor_labels != AFFINE_ANCHOR_LABELS:
            raise ValueError("affine anchor labels/inventory changed")
        anchors = _immutable_array(
            self.affine_anchor_offsets,
            label="affine anchor offsets",
            dtype=np.dtype(np.complex128),
            shape=(len(AFFINE_ANCHOR_LABELS), 4, 4, nk),
        )
        if not np.array_equal(anchors[0], np.zeros_like(anchors[0])):
            raise ValueError("source affine anchor offset must be exactly zero")
        for anchor_index, offset in enumerate(anchors):
            if not np.array_equal(offset, np.swapaxes(offset.conj(), 0, 1)):
                raise ValueError(f"affine anchor {anchor_index} is not exactly Hermitian")
            trace = np.sum(np.trace(offset, axis1=0, axis2=1))
            scale = max(float(np.sum(np.abs(offset))), 1.0)
            if abs(trace) > 64.0 * np.finfo(np.float64).eps * scale:
                raise ValueError(f"affine anchor {anchor_index} changes total density")
        imaginary = anchors[1]
        diagonal_mask = np.eye(4, dtype=np.bool_)[:, :, None].repeat(nk, axis=2)
        if (
            np.any(imaginary[diagonal_mask] != 0.0)
            or not np.any(np.imag(imaginary) != 0.0)
            or np.any(np.real(imaginary) != 0.0)
        ):
            raise ValueError("imaginary off-diagonal affine anchor semantics changed")
        dense = anchors[2]
        if not (
            np.any(dense[diagonal_mask] != 0.0)
            and np.any(dense[~diagonal_mask] != 0.0)
            and np.any(np.real(dense) != 0.0)
            and np.any(np.imag(dense) != 0.0)
        ):
            raise ValueError("dense complex Hermitian affine anchor semantics changed")

        if self.q0_labels != Q0_PROBE_LABELS:
            raise ValueError("q0 probe labels/inventory changed")
        q0 = _immutable_array(
            self.q0_directions,
            label="q0 density directions",
            dtype=np.dtype(np.complex128),
            shape=(len(Q0_PROBE_LABELS), 4, 4, nk),
        )
        for probe_index, direction in enumerate(q0):
            if not np.array_equal(direction, np.swapaxes(direction.conj(), 0, 1)):
                raise ValueError(f"q0 direction {probe_index} is not exactly Hermitian")
            trace = np.sum(np.trace(direction, axis1=0, axis2=1))
            scale = max(float(np.sum(np.abs(direction))), 1.0)
            if abs(trace) > 64.0 * np.finfo(np.float64).eps * scale:
                raise ValueError(f"q0 direction {probe_index} does not have total trace zero")
            norm = float(np.sqrt(np.sum(np.abs(direction) ** 2)))
            if not math.isclose(norm, 1.0, rel_tol=2.0e-15, abs_tol=2.0e-15):
                raise ValueError(f"q0 direction {probe_index} Frobenius norm is not one")
        off_diagonal_mask = ~diagonal_mask
        if np.any(q0[0][off_diagonal_mask] != 0.0):
            raise ValueError("diagonal-flavor q0 probe semantics changed")
        nonzero = np.argwhere(np.abs(q0[1]) > 0.0)
        if (
            np.any(q0[1][off_diagonal_mask] != 0.0)
            or len(nonzero) < 2
            or len(set(int(item[0]) for item in nonzero)) != 1
            or len(set(int(item[2]) for item in nonzero)) < 2
        ):
            raise ValueError("momentum-redistribution q0 probe semantics changed")
        if (
            np.any(q0[2][diagonal_mask] != 0.0)
            or not np.any(q0[2] != 0.0)
            or np.any(np.imag(q0[2]) != 0.0)
        ):
            raise ValueError("real-coherence q0 probe semantics changed")
        if (
            np.any(q0[3][diagonal_mask] != 0.0)
            or not np.any(q0[3] != 0.0)
            or np.any(np.real(q0[3]) != 0.0)
        ):
            raise ValueError("imaginary-coherence q0 probe semantics changed")
        if not (
            np.any(q0[4][diagonal_mask] != 0.0)
            and np.any(q0[4][off_diagonal_mask] != 0.0)
            and np.any(np.real(q0[4]) != 0.0)
            and np.any(np.imag(q0[4]) != 0.0)
        ):
            raise ValueError("deterministic dense-mixed q0 probe semantics changed")

        if self.signed_q_labels != SIGNED_Q_PROBE_LABELS:
            raise ValueError("signed-q probe labels/inventory changed")
        signed = _immutable_array(
            self.signed_q_probes,
            label="signed-q probes",
            dtype=np.dtype(np.complex128),
            shape=(len(SIGNED_Q_PROBE_LABELS), 2, 4, 4, nk),
        )
        indices = _immutable_array(
            self.signed_q_probe_indices,
            label="signed-q probe indices",
            dtype=np.dtype(np.int64),
            shape=(len(SIGNED_Q_PROBE_LABELS),),
        )
        expected_indices = np.repeat(
            np.arange(len(Q_CHART_LABELS), dtype=np.int64), 3
        )
        if not np.array_equal(indices, expected_indices):
            raise ValueError("signed-q probes must cover both q indices with three probes each")
        for probe_index, probe in enumerate(signed):
            chart = self.q_charts[int(indices[probe_index])]
            norm = float(np.sqrt(np.sum(np.abs(probe) ** 2)))
            if not math.isclose(norm, 1.0, rel_tol=2.0e-15, abs_tol=2.0e-15):
                raise ValueError(
                    f"signed-q probe {probe_index} packed-pair Frobenius norm is not one"
                )
            for sign_index in range(2):
                invalid = ~chart.validity_masks[sign_index]
                if not np.array_equal(
                    probe[sign_index, :, :, invalid],
                    np.zeros_like(probe[sign_index, :, :, invalid]),
                ):
                    raise ValueError("signed-q probe has nonzero invalid-edge slots")
            local_kind = probe_index % 3
            lane_norms = np.sqrt(np.sum(np.abs(probe) ** 2, axis=(1, 2, 3)))
            plus_boundary = chart.validity_masks[0] & ~chart.validity_masks[1]
            minus_boundary = chart.validity_masks[1] & ~chart.validity_masks[0]
            interior = chart.validity_masks[0] & chart.validity_masks[1]
            if local_kind == 0:
                if not (lane_norms[0] > 0.0 and lane_norms[1] == 0.0):
                    raise ValueError("plus-only boundary probe lane semantics changed")
                if not np.any(np.abs(probe[0, :, :, plus_boundary]) > 0.0) or np.any(
                    probe[0, :, :, ~plus_boundary] != 0.0
                ):
                    raise ValueError("plus-only probe must occupy plus-valid/minus-invalid edges")
            elif local_kind == 1:
                if not (lane_norms[0] == 0.0 and lane_norms[1] > 0.0):
                    raise ValueError("minus-only boundary probe lane semantics changed")
                if not np.any(np.abs(probe[1, :, :, minus_boundary]) > 0.0) or np.any(
                    probe[1, :, :, ~minus_boundary] != 0.0
                ):
                    raise ValueError("minus-only probe must occupy minus-valid/plus-invalid edges")
            else:
                if not np.all(lane_norms > 0.0):
                    raise ValueError("mixed interior probe requires both independent lanes")
                if not np.any(np.abs(probe[:, :, :, interior]) > 0.0) or np.any(
                    probe[:, :, :, ~interior] != 0.0
                ):
                    raise ValueError("mixed probe must occupy only two-sided interior edges")
        object.__setattr__(self, "affine_anchor_offsets", anchors)
        object.__setattr__(self, "q0_directions", q0)
        object.__setattr__(self, "signed_q_probes", signed)
        object.__setattr__(self, "signed_q_probe_indices", indices)

    @property
    def affine_anchor_inventory_sha256(self) -> str:
        return affine_anchor_inventory_sha256(
            self.affine_anchor_labels, self.affine_anchor_offsets
        )

    @property
    def q0_probe_inventory_sha256(self) -> str:
        return q0_probe_inventory_sha256(self.q0_labels, self.q0_directions)

    @property
    def signed_q_probe_inventory_sha256(self) -> str:
        return signed_q_probe_inventory_sha256(
            self.signed_q_labels, self.signed_q_probes, self.signed_q_probe_indices
        )

    @property
    def q_probe_inventory_sha256(self) -> str:
        return signed_q_chart_inventory_sha256(self.q_charts)

    @property
    def manifest_sha256(self) -> str:
        return _fingerprint(
            {
                "schema": self.functional_replay_payload_schema_fingerprint,
                "affine_anchor_inventory_sha256": (
                    self.affine_anchor_inventory_sha256
                ),
                "q0_probe_inventory_sha256": self.q0_probe_inventory_sha256,
                "signed_q_probe_inventory_sha256": (
                    self.signed_q_probe_inventory_sha256
                ),
                "q_probe_inventory_sha256": self.q_probe_inventory_sha256,
                "source_state_sha256": self.source_state_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayChoice:
    absolute_tolerance: float = _REGISTERED_ABSOLUTE_TOLERANCE
    relative_tolerance: float = _REGISTERED_RELATIVE_TOLERANCE
    roundoff_ulps: float = _REGISTERED_ROUNDOFF_ULPS
    slope_stability_tolerance: float = _REGISTERED_SLOPE_STABILITY_TOLERANCE
    informativeness_floor: float = _REGISTERED_INFORMATIVENESS_FLOOR
    source_mesh_q_coordinate_tolerance_inverse_angstrom: float = (
        _REGISTERED_SOURCE_MESH_Q_COORDINATE_TOLERANCE_INVERSE_ANGSTROM
    )
    aggregation: Literal["all_local_gates_before_aggregate"] = (
        "all_local_gates_before_aggregate"
    )
    fock_output: Literal["full_fock_h0_plus_interaction"] = FOCK_OUTPUT_CONVENTION
    stored_density_pairing: Literal[
        "real_bilinear_sum_abk_no_conjugation_over_nk"
    ] = STORED_DENSITY_PAIRING
    density_direction_convention: Literal["fixed_density_affine_directions"] = (
        FIXED_DENSITY_DIRECTION_CONVENTION
    )

    def __post_init__(self) -> None:
        locked = (
            (self.absolute_tolerance, _REGISTERED_ABSOLUTE_TOLERANCE),
            (self.relative_tolerance, _REGISTERED_RELATIVE_TOLERANCE),
            (self.roundoff_ulps, _REGISTERED_ROUNDOFF_ULPS),
            (self.slope_stability_tolerance, _REGISTERED_SLOPE_STABILITY_TOLERANCE),
            (self.informativeness_floor, _REGISTERED_INFORMATIVENESS_FLOOR),
            (
                self.source_mesh_q_coordinate_tolerance_inverse_angstrom,
                _REGISTERED_SOURCE_MESH_Q_COORDINATE_TOLERANCE_INVERSE_ANGSTROM,
            ),
        )
        if any(
            _positive(actual, "functional replay tolerance") != expected
            for actual, expected in locked
        ):
            raise ValueError("functional replay registered tolerances changed")
        if (
            self.aggregation != "all_local_gates_before_aggregate"
            or self.fock_output != FOCK_OUTPUT_CONVENTION
            or self.stored_density_pairing != STORED_DENSITY_PAIRING
            or self.density_direction_convention != FIXED_DENSITY_DIRECTION_CONVENTION
        ):
            raise ValueError("functional replay choice/conventions changed")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayApproval:
    """Detached preregistration created before any provider method is called."""

    choice_fingerprint: str
    verifier_implementation_schema_fingerprint: str
    verifier_module_ast_manifest_sha256: str
    functional_provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    source_state_sha256: str
    expected_array_payload_manifest_sha256: str
    affine_anchor_inventory_sha256: str
    q0_probe_inventory_sha256: str
    signed_q_probe_inventory_sha256: str
    q_probe_inventory_sha256: str
    fock_step_ladder: tuple[float, ...]
    hessian_step_ladder: tuple[float, ...]
    direct_displaced_fock_implementation_fingerprint: str
    direct_builder_dependency_archive_fingerprint: str
    provenance: str
    scope: str = FUNCTIONAL_REPLAY_SCOPE
    approval_precedes_execution: Literal[True] = field(default=True, init=False)

    def __post_init__(self) -> None:
        _commit(self.source_commit, "approval source commit")
        for value, label in (
            (self.choice_fingerprint, "approval choice"),
            (
                self.verifier_implementation_schema_fingerprint,
                "approval verifier implementation/schema",
            ),
            (
                self.verifier_module_ast_manifest_sha256,
                "approval verifier module AST/source manifest",
            ),
            (self.functional_provider_fingerprint, "approval functional provider"),
            (self.source_artifact_sha256, "approval source artifact"),
            (self.spec_fingerprint, "approval spec"),
            (self.source_state_sha256, "approval source state"),
            (self.expected_array_payload_manifest_sha256, "approval array manifest"),
            (self.affine_anchor_inventory_sha256, "approval anchor inventory"),
            (self.q0_probe_inventory_sha256, "approval q0 inventory"),
            (self.signed_q_probe_inventory_sha256, "approval signed-q inventory"),
            (self.q_probe_inventory_sha256, "approval q chart inventory"),
            (
                self.direct_displaced_fock_implementation_fingerprint,
                "approval direct implementation",
            ),
            (
                self.direct_builder_dependency_archive_fingerprint,
                "approval direct dependency archive",
            ),
        ):
            _sha256(value, label)
        _text(self.provenance, "detached approval provenance")
        if (
            self.verifier_implementation_schema_fingerprint
            != FUNCTIONAL_REPLAY_VERIFIER_IMPLEMENTATION_SCHEMA_FINGERPRINT
        ):
            raise ValueError("detached approval verifier implementation/schema changed")
        if (
            self.verifier_module_ast_manifest_sha256
            != functional_replay_module_ast_manifest_sha256()
        ):
            raise ValueError("detached approval verifier AST/source manifest changed")
        if self.scope != FUNCTIONAL_REPLAY_SCOPE or not self.approval_precedes_execution:
            raise ValueError("detached approval scope/order changed")
        for name in ("fock_step_ladder", "hessian_step_ladder"):
            steps = tuple(_positive(value, name) for value in getattr(self, name))
            if len(steps) < 3 or any(
                left <= right for left, right in zip(steps, steps[1:])
            ):
                raise ValueError(f"{name} must contain >=3 strictly decreasing steps")
            object.__setattr__(self, name, steps)

    @property
    def manifest_sha256(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayContract:
    """Detached contract; construction has no replay-receipt dependency."""

    choice: Vituri2024FunctionalReplayChoice
    choice_fingerprint: str
    verifier_implementation_schema_fingerprint: str
    verifier_module_ast_manifest_sha256: str
    provider_fingerprint: str
    functional_provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    source_state_sha256: str
    geometry_receipt_fingerprint: str
    ensemble_receipt_fingerprint: str
    normal_order_reference_fingerprint: str
    q0_policy_fingerprint: str
    interaction_receipt_fingerprint: str
    shared_functional_receipt_fingerprint: str
    attested_source_receipt_fingerprint: str
    expected_array_payload_manifest_sha256: str
    affine_anchor_inventory_sha256: str
    q0_probe_inventory_sha256: str
    signed_q_probe_inventory_sha256: str
    q_probe_inventory_sha256: str
    fock_step_ladder: tuple[float, ...]
    hessian_step_ladder: tuple[float, ...]
    replay_loader_implementation_fingerprint: str
    functional_probe_loader_implementation_fingerprint: str
    functional_replay_payload_schema_fingerprint: str
    functional_replay_abi_fingerprint: str
    direct_displaced_fock_implementation_fingerprint: str
    direct_interaction_builder_implementation_fingerprint: str
    direct_full_fock_builder_implementation_fingerprint: str
    direct_builder_dependency_archive_fingerprint: str
    detached_approval_manifest_sha256: str
    detached_approval_provenance: str
    scope: str = FUNCTIONAL_REPLAY_SCOPE
    approval_precedes_execution: Literal[True] = field(default=True, init=False)

    def __post_init__(self) -> None:
        if type(self.choice) is not Vituri2024FunctionalReplayChoice:
            raise TypeError("functional replay contract requires a typed choice")
        if self.scope != FUNCTIONAL_REPLAY_SCOPE or not self.approval_precedes_execution:
            raise ValueError("functional replay contract scope/approval order changed")
        _commit(self.source_commit, "contract source commit")
        for value, label in (
            (self.choice_fingerprint, "contract choice"),
            (
                self.verifier_implementation_schema_fingerprint,
                "contract verifier implementation/schema",
            ),
            (
                self.verifier_module_ast_manifest_sha256,
                "contract verifier module AST/source manifest",
            ),
            (self.provider_fingerprint, "contract base provider"),
            (self.functional_provider_fingerprint, "contract functional provider"),
            (self.source_artifact_sha256, "contract source artifact"),
            (self.spec_fingerprint, "contract spec"),
            (self.source_state_sha256, "contract source state"),
            (self.geometry_receipt_fingerprint, "contract geometry"),
            (self.ensemble_receipt_fingerprint, "contract ensemble"),
            (self.normal_order_reference_fingerprint, "contract reference"),
            (self.q0_policy_fingerprint, "contract q0 policy"),
            (self.interaction_receipt_fingerprint, "contract interaction"),
            (self.shared_functional_receipt_fingerprint, "contract shared functional"),
            (self.attested_source_receipt_fingerprint, "contract attested source"),
            (self.expected_array_payload_manifest_sha256, "contract array manifest"),
            (self.affine_anchor_inventory_sha256, "contract anchor inventory"),
            (self.q0_probe_inventory_sha256, "contract q0 inventory"),
            (self.signed_q_probe_inventory_sha256, "contract signed-q inventory"),
            (self.q_probe_inventory_sha256, "contract q chart inventory"),
            (self.replay_loader_implementation_fingerprint, "contract array loader"),
            (
                self.functional_probe_loader_implementation_fingerprint,
                "contract probe loader",
            ),
            (self.functional_replay_payload_schema_fingerprint, "contract schema"),
            (self.functional_replay_abi_fingerprint, "contract ABI"),
            (
                self.direct_displaced_fock_implementation_fingerprint,
                "contract direct implementation",
            ),
            (
                self.direct_interaction_builder_implementation_fingerprint,
                "contract direct interaction builder",
            ),
            (
                self.direct_full_fock_builder_implementation_fingerprint,
                "contract direct full-Fock builder",
            ),
            (
                self.direct_builder_dependency_archive_fingerprint,
                "contract direct dependency archive",
            ),
            (self.detached_approval_manifest_sha256, "contract detached approval"),
        ):
            _sha256(value, label)
        _text(self.detached_approval_provenance, "contract approval provenance")
        if self.choice_fingerprint != self.choice.fingerprint:
            raise ValueError("functional replay contract choice fingerprint mismatch")
        if (
            self.verifier_implementation_schema_fingerprint
            != FUNCTIONAL_REPLAY_VERIFIER_IMPLEMENTATION_SCHEMA_FINGERPRINT
        ):
            raise ValueError("functional replay contract verifier implementation/schema changed")
        if (
            self.verifier_module_ast_manifest_sha256
            != functional_replay_module_ast_manifest_sha256()
        ):
            raise ValueError("functional replay contract verifier AST/source manifest changed")
        if (
            self.functional_replay_payload_schema_fingerprint
            != FUNCTIONAL_REPLAY_PAYLOAD_SCHEMA_FINGERPRINT
            or self.functional_replay_abi_fingerprint != FUNCTIONAL_REPLAY_ABI_FINGERPRINT
        ):
            raise ValueError("functional replay contract ABI/schema changed")
        expected_archive = direct_builder_dependency_archive_fingerprint(
            source_commit=self.source_commit,
            source_artifact_sha256=self.source_artifact_sha256,
            direct_displaced_fock_implementation_fingerprint=(
                self.direct_displaced_fock_implementation_fingerprint
            ),
            interaction_builder_implementation_fingerprint=(
                self.direct_interaction_builder_implementation_fingerprint
            ),
            full_fock_builder_implementation_fingerprint=(
                self.direct_full_fock_builder_implementation_fingerprint
            ),
        )
        if self.direct_builder_dependency_archive_fingerprint != expected_archive:
            raise ValueError("direct-builder dependency archive fingerprint mismatch")
        expected_provider = functional_provider_fingerprint(
            base_provider_fingerprint=self.provider_fingerprint,
            functional_replay_abi_fingerprint=self.functional_replay_abi_fingerprint,
            functional_replay_payload_schema_fingerprint=(
                self.functional_replay_payload_schema_fingerprint
            ),
            functional_probe_loader_implementation_fingerprint=(
                self.functional_probe_loader_implementation_fingerprint
            ),
            direct_displaced_fock_implementation_fingerprint=(
                self.direct_displaced_fock_implementation_fingerprint
            ),
            direct_builder_dependency_archive_fingerprint=(
                self.direct_builder_dependency_archive_fingerprint
            ),
        )
        if self.functional_provider_fingerprint != expected_provider:
            raise ValueError("derived functional provider fingerprint mismatch")
        for name in ("fock_step_ladder", "hessian_step_ladder"):
            steps = tuple(_positive(value, name) for value in getattr(self, name))
            if len(steps) < 3 or any(
                left <= right for left, right in zip(steps, steps[1:])
            ):
                raise ValueError(f"{name} must contain >=3 strictly decreasing steps")
            object.__setattr__(self, name, steps)
        approval = Vituri2024FunctionalReplayApproval(
            choice_fingerprint=self.choice_fingerprint,
            verifier_implementation_schema_fingerprint=(
                self.verifier_implementation_schema_fingerprint
            ),
            verifier_module_ast_manifest_sha256=(
                self.verifier_module_ast_manifest_sha256
            ),
            functional_provider_fingerprint=self.functional_provider_fingerprint,
            source_commit=self.source_commit,
            source_artifact_sha256=self.source_artifact_sha256,
            spec_fingerprint=self.spec_fingerprint,
            source_state_sha256=self.source_state_sha256,
            expected_array_payload_manifest_sha256=(
                self.expected_array_payload_manifest_sha256
            ),
            affine_anchor_inventory_sha256=self.affine_anchor_inventory_sha256,
            q0_probe_inventory_sha256=self.q0_probe_inventory_sha256,
            signed_q_probe_inventory_sha256=self.signed_q_probe_inventory_sha256,
            q_probe_inventory_sha256=self.q_probe_inventory_sha256,
            fock_step_ladder=self.fock_step_ladder,
            hessian_step_ladder=self.hessian_step_ladder,
            direct_displaced_fock_implementation_fingerprint=(
                self.direct_displaced_fock_implementation_fingerprint
            ),
            direct_builder_dependency_archive_fingerprint=(
                self.direct_builder_dependency_archive_fingerprint
            ),
            provenance=self.detached_approval_provenance,
        )
        if approval.manifest_sha256 != self.detached_approval_manifest_sha256:
            raise ValueError("detached approval manifest mismatch")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024DirectBuilderDependencyTrace:
    caller_nonce_sha256: str
    q_probe_index: int
    q_label: str
    interaction_builder_implementation_fingerprint: str
    full_fock_builder_implementation_fingerprint: str
    target_maps_sha256: str
    reverse_edge_map_sha256: str
    target_map_read_count: int
    reverse_edge_map_read_count: int
    interaction_builder_call_count: int
    full_fock_builder_call_count: int
    finite_q_hessian_call_count: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.caller_nonce_sha256, "direct trace caller nonce"),
            (
                self.interaction_builder_implementation_fingerprint,
                "direct trace interaction builder",
            ),
            (
                self.full_fock_builder_implementation_fingerprint,
                "direct trace full-Fock builder",
            ),
            (self.target_maps_sha256, "direct trace target maps"),
            (self.reverse_edge_map_sha256, "direct trace reverse map"),
        ):
            _sha256(value, label)
        index = _strict_int(self.q_probe_index, "direct trace q index")
        if index < 0:
            raise ValueError("direct trace q index must be non-negative")
        _text(self.q_label, "direct trace q label")
        for name in (
            "target_map_read_count",
            "reverse_edge_map_read_count",
            "interaction_builder_call_count",
            "full_fock_builder_call_count",
            "finite_q_hessian_call_count",
        ):
            count = _strict_int(getattr(self, name), f"direct trace {name}")
            if count < 0:
                raise ValueError(f"direct trace {name} must be non-negative")
            object.__setattr__(self, name, count)
        object.__setattr__(self, "q_probe_index", index)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024DirectDisplacedFockResponse:
    caller_nonce_sha256: str
    response: ComplexArray
    dependency_trace: Vituri2024DirectBuilderDependencyTrace
    response_token_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.caller_nonce_sha256, "direct response caller nonce")
        if type(self.dependency_trace) is not Vituri2024DirectBuilderDependencyTrace:
            raise TypeError("direct response requires a typed dependency trace")
        if self.dependency_trace.caller_nonce_sha256 != self.caller_nonce_sha256:
            raise ValueError("direct response trace is not bound to caller nonce")
        response = _immutable_array(
            self.response,
            label="direct displaced-Fock response",
            dtype=np.dtype(np.complex128),
        )
        token = _fingerprint(
            {
                "schema": "vituri2024_direct_displaced_fock_response_v1",
                "caller_nonce_sha256": self.caller_nonce_sha256,
                "response_sha256": canonical_array_sha256(response),
                "dependency_trace_sha256": self.dependency_trace.fingerprint,
            }
        )
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "response_token_sha256", token)


@runtime_checkable
class Vituri2024FunctionalReplayProviderProtocol(
    Vituri2024HalfMetalHFReplayProviderProtocol, Protocol
):
    functional_provider_fingerprint: str
    functional_probe_loader_implementation_fingerprint: str
    functional_replay_payload_schema_fingerprint: str
    functional_replay_abi_fingerprint: str
    direct_displaced_fock_implementation_fingerprint: str
    direct_interaction_builder_implementation_fingerprint: str
    direct_full_fock_builder_implementation_fingerprint: str
    direct_builder_dependency_archive_fingerprint: str
    direct_displaced_fock_construction: str
    fock_output: str
    stored_density_pairing: str
    density_direction_convention: str

    def load_functional_replay_payload(
        self, source_artifact_sha256: str
    ) -> Vituri2024FunctionalReplayPayload: ...

    def evaluate_finite_q_hessian(
        self,
        perturbation: ComplexArray,
        *,
        q_probe_index: int,
        mesh_displacement: IntegerArray,
        cartesian_q: FloatArray,
        target_maps: IntegerArray,
        reverse_edge_map: IntegerArray,
    ) -> ComplexArray: ...

    def evaluate_displaced_fock(
        self,
        density: ComplexArray,
        signed_q_displacement: ComplexArray,
        *,
        q_probe_index: int,
        mesh_displacement: IntegerArray,
        cartesian_q: FloatArray,
        target_maps: IntegerArray,
        reverse_edge_map: IntegerArray,
        caller_nonce: str,
    ) -> Vituri2024DirectDisplacedFockResponse: ...


FUNCTIONAL_REPLAY_PROVIDER_METADATA_FIELDS = (
    VITURI2024_BASE_PROVIDER_METADATA_FIELDS
    + (
        "replay_loader_implementation_fingerprint",
        "replay_payload_schema_fingerprint",
        "functional_provider_fingerprint",
        "functional_probe_loader_implementation_fingerprint",
        "functional_replay_payload_schema_fingerprint",
        "functional_replay_abi_fingerprint",
        "direct_displaced_fock_implementation_fingerprint",
        "direct_interaction_builder_implementation_fingerprint",
        "direct_full_fock_builder_implementation_fingerprint",
        "direct_builder_dependency_archive_fingerprint",
        "direct_displaced_fock_construction",
        "fock_output",
        "stored_density_pairing",
        "density_direction_convention",
    )
)


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayCallRecord:
    sequence_index: int
    method: str
    argument_hashes: tuple[str, ...]
    keyword_fingerprint: str
    output_sha256: str
    dependency_trace_sha256: str | None = None

    def __post_init__(self) -> None:
        index = _strict_int(self.sequence_index, "call sequence index")
        if index < 0:
            raise ValueError("call sequence index must be non-negative")
        _text(self.method, "call method")
        if type(self.argument_hashes) is not tuple:
            raise TypeError("call argument hashes must be a tuple")
        for value in self.argument_hashes:
            _sha256(value, "call argument hash")
        _sha256(self.keyword_fingerprint, "call keyword fingerprint")
        _sha256(self.output_sha256, "call output hash")
        if self.dependency_trace_sha256 is not None:
            _sha256(self.dependency_trace_sha256, "call dependency trace")
        object.__setattr__(self, "sequence_index", index)


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayAnchorCheck:
    fock_entrywise_residual: float
    fock_entrywise_result_scale: float
    fock_entrywise_termwise_magnitude: float
    fock_entrywise_operation_count: int
    fock_entrywise_roundoff_contribution: float
    fock_entrywise_registered_bound: float
    interaction_entrywise_residual: float
    interaction_entrywise_result_scale: float
    interaction_entrywise_termwise_magnitude: float
    interaction_entrywise_operation_count: int
    interaction_entrywise_roundoff_contribution: float
    interaction_entrywise_registered_bound: float
    scalar_energy_residual: float
    scalar_energy_result_scale: float
    scalar_energy_termwise_magnitude: float
    scalar_energy_operation_count: int
    scalar_energy_roundoff_contribution: float
    scalar_energy_registered_bound: float

    def __post_init__(self) -> None:
        for name in (
            "fock_entrywise_operation_count",
            "interaction_entrywise_operation_count",
            "scalar_energy_operation_count",
        ):
            count = _strict_int(getattr(self, name), f"source anchor {name}")
            if count < 1:
                raise ValueError(f"source anchor {name} must be positive")
            object.__setattr__(self, name, count)
        for name in asdict(self):
            if not name.endswith("operation_count"):
                _nonnegative(getattr(self, name), f"source anchor {name}")
        if (
            self.fock_entrywise_termwise_magnitude
            < self.fock_entrywise_result_scale
            or self.interaction_entrywise_termwise_magnitude
            < self.interaction_entrywise_result_scale
            or self.scalar_energy_termwise_magnitude
            < self.scalar_energy_result_scale
        ):
            raise ValueError("source anchor termwise magnitude is below its result scale")
        if (
            self.fock_entrywise_residual
            > self.fock_entrywise_registered_bound
            or self.interaction_entrywise_residual
            > self.interaction_entrywise_registered_bound
            or self.scalar_energy_residual > self.scalar_energy_registered_bound
        ):
            raise ValueError("source anchor exceeds a local registered bound")


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayScalarStep:
    anchor_index: int
    anchor_label: str
    probe_index: int
    probe_label: str
    step: float
    central_slope: float
    fock_pairing: float
    residual: float
    local_scale: float
    energy_termwise_abs_sum: float
    pairing_termwise_abs_sum: float
    operation_count: int
    roundoff_bound: float
    registered_bound: float

    def __post_init__(self) -> None:
        for name in ("anchor_index", "probe_index", "operation_count"):
            value = _strict_int(getattr(self, name), f"scalar-step {name}")
            if value < 0:
                raise ValueError(f"scalar-step {name} must be non-negative")
            object.__setattr__(self, name, value)
        _text(self.anchor_label, "scalar-step anchor label")
        _text(self.probe_label, "scalar-step probe label")
        _positive(self.step, "scalar-step finite-difference step")
        for name in (
            "central_slope",
            "fock_pairing",
            "residual",
            "local_scale",
            "energy_termwise_abs_sum",
            "pairing_termwise_abs_sum",
            "roundoff_bound",
            "registered_bound",
        ):
            value = _finite(getattr(self, name), f"scalar-step {name}")
            if name not in ("central_slope", "fock_pairing") and value < 0.0:
                raise ValueError(f"scalar-step {name} must be non-negative")
        if self.residual > self.registered_bound:
            raise ValueError("scalar E->F record exceeds its local registered bound")


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayScalarLocalGate:
    anchor_index: int
    anchor_label: str
    probe_index: int
    probe_label: str
    informativeness_abs: float
    stability_max_abs: float
    stability_scale: float
    stability_roundoff_bound: float
    stability_registered_bound: float

    def __post_init__(self) -> None:
        _strict_int(self.anchor_index, "scalar local anchor index")
        _strict_int(self.probe_index, "scalar local probe index")
        _text(self.anchor_label, "scalar local anchor label")
        _text(self.probe_label, "scalar local probe label")
        for name in (
            "informativeness_abs",
            "stability_max_abs",
            "stability_scale",
            "stability_roundoff_bound",
            "stability_registered_bound",
        ):
            _nonnegative(getattr(self, name), f"scalar local {name}")
        if self.stability_max_abs > self.stability_registered_bound:
            raise ValueError("scalar E->F local slope stability gate failed")


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayMatrixStep:
    q_probe_index: int
    q_label: str
    probe_index: int
    probe_label: str
    response_sign_index: Literal[0, 1]
    step: float
    derivative_norm: float
    hessian_norm: float
    residual_norm: float
    local_scale: float
    difference_termwise_abs_sum: float
    operation_count: int
    roundoff_bound: float
    registered_bound: float

    def __post_init__(self) -> None:
        for name in ("q_probe_index", "probe_index", "operation_count"):
            value = _strict_int(getattr(self, name), f"matrix-step {name}")
            if value < 0:
                raise ValueError(f"matrix-step {name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.response_sign_index not in (0, 1):
            raise ValueError("matrix-step response sign index must be 0 or 1")
        _text(self.q_label, "matrix-step q label")
        _text(self.probe_label, "matrix-step probe label")
        _positive(self.step, "matrix-step finite-difference step")
        for name in (
            "derivative_norm",
            "hessian_norm",
            "residual_norm",
            "local_scale",
            "difference_termwise_abs_sum",
            "roundoff_bound",
            "registered_bound",
        ):
            _nonnegative(getattr(self, name), f"matrix-step {name}")
        if self.residual_norm > self.registered_bound:
            raise ValueError("matrix F->dF record exceeds its local registered bound")


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayMatrixLocalGate:
    q_probe_index: int
    q_label: str
    probe_index: int
    probe_label: str
    response_sign_index: Literal[0, 1]
    informativeness_norm: float
    stability_max_frobenius: float
    stability_scale: float
    stability_roundoff_bound: float
    stability_registered_bound: float

    def __post_init__(self) -> None:
        _strict_int(self.q_probe_index, "matrix local q index")
        _strict_int(self.probe_index, "matrix local probe index")
        if self.response_sign_index not in (0, 1):
            raise ValueError("matrix local response sign index must be 0 or 1")
        _text(self.q_label, "matrix local q label")
        _text(self.probe_label, "matrix local probe label")
        for name in (
            "informativeness_norm",
            "stability_max_frobenius",
            "stability_scale",
            "stability_roundoff_bound",
            "stability_registered_bound",
        ):
            _nonnegative(getattr(self, name), f"matrix local {name}")
        if self.stability_max_frobenius > self.stability_registered_bound:
            raise ValueError("matrix F->dF local slope stability gate failed")


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayReciprocity:
    q_probe_index: int
    q_label: str
    left_probe_index: int
    right_probe_index: int
    left_right_pairing: float
    right_left_pairing: float
    left_right_termwise_abs_sum: float
    right_left_termwise_abs_sum: float
    operation_count: int
    residual: float
    local_scale: float
    roundoff_bound: float
    registered_bound: float

    def __post_init__(self) -> None:
        q_index = _strict_int(self.q_probe_index, "reciprocity q index")
        left = _strict_int(self.left_probe_index, "reciprocity left index")
        right = _strict_int(self.right_probe_index, "reciprocity right index")
        count = _strict_int(self.operation_count, "reciprocity operation count")
        if q_index < 0 or left >= right or count < 1:
            raise ValueError("reciprocity index/count ordering is invalid")
        _text(self.q_label, "reciprocity q label")
        for name in (
            "left_right_pairing",
            "right_left_pairing",
            "left_right_termwise_abs_sum",
            "right_left_termwise_abs_sum",
            "residual",
            "local_scale",
            "roundoff_bound",
            "registered_bound",
        ):
            value = _finite(getattr(self, name), f"reciprocity {name}")
            if name not in ("left_right_pairing", "right_left_pairing") and value < 0.0:
                raise ValueError(f"reciprocity {name} must be non-negative")
        if self.residual > self.registered_bound:
            raise ValueError("bilinear Hessian reciprocity exceeds local bound")


_FUNCTIONAL_REPLAY_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayStatus:
    _factory_token: InitVar[object]
    local_registered_functional_probes_replayed: Literal[True] = field(
        default=True, init=False
    )
    array_replay_verified: Literal[True] = field(default=True, init=False)
    all_local_registered_gates_passed: Literal[True] = field(default=True, init=False)
    approval_precedes_execution: Literal[True] = field(default=True, init=False)
    scope: str = field(default=FUNCTIONAL_REPLAY_SCOPE, init=False)
    affine_anchor_count: int = field(default=len(AFFINE_ANCHOR_LABELS), init=False)
    q0_probe_count: int = field(default=len(Q0_PROBE_LABELS), init=False)
    signed_q_probe_count: int = field(default=len(SIGNED_Q_PROBE_LABELS), init=False)
    q_chart_count: int = field(default=len(Q_CHART_LABELS), init=False)
    global_functional_chain_verified: Literal[False] = field(default=False, init=False)
    scf_trajectory_replayed: Literal[False] = field(default=False, init=False)
    branch_table_replayed: Literal[False] = field(default=False, init=False)
    pocket_refinement_replayed: Literal[False] = field(default=False, init=False)
    scientific_execution_verified: Literal[False] = field(default=False, init=False)
    paper_reproduction_verified: Literal[False] = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _FUNCTIONAL_REPLAY_FACTORY_TOKEN:
            raise TypeError("functional replay success requires the private factory token")
        if not (
            self.local_registered_functional_probes_replayed
            and self.array_replay_verified
            and self.all_local_registered_gates_passed
            and self.approval_precedes_execution
        ):
            raise ValueError("functional replay success lost a required local gate")
        if any(
            (
                self.global_functional_chain_verified,
                self.scf_trajectory_replayed,
                self.branch_table_replayed,
                self.pocket_refinement_replayed,
                self.scientific_execution_verified,
                self.paper_reproduction_verified,
            )
        ):
            raise ValueError("local functional replay cannot claim broader execution")


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalReplayReceipt:
    contract_fingerprint: str
    choice_fingerprint: str
    verifier_implementation_schema_fingerprint: str
    verifier_module_ast_manifest_sha256: str
    functional_provider_fingerprint: str
    detached_approval_manifest_sha256: str
    detached_approval_provenance: str
    approval_precedes_execution: Literal[True]
    expected_array_payload_manifest_sha256: str
    array_replay_receipt_fingerprint: str
    array_replay_payload_manifest_sha256: str
    functional_payload_manifest_sha256: str
    affine_anchor_inventory_sha256: str
    q0_probe_inventory_sha256: str
    signed_q_probe_inventory_sha256: str
    q_probe_inventory_sha256: str
    direct_dependency_trace_transcript_sha256: str
    scope: str
    affine_anchor_count: int
    q0_probe_count: int
    signed_q_probe_count: int
    q_chart_count: int
    source_anchor: Vituri2024FunctionalReplayAnchorCheck
    scalar_steps: tuple[Vituri2024FunctionalReplayScalarStep, ...]
    scalar_local_gates: tuple[Vituri2024FunctionalReplayScalarLocalGate, ...]
    matrix_steps: tuple[Vituri2024FunctionalReplayMatrixStep, ...]
    matrix_local_gates: tuple[Vituri2024FunctionalReplayMatrixLocalGate, ...]
    reciprocity: tuple[Vituri2024FunctionalReplayReciprocity, ...]
    scalar_slope_stability_max_abs: float
    matrix_slope_stability_max_frobenius: float
    scalar_informativeness_max_abs: float
    matrix_informativeness_max_frobenius: float
    call_records: tuple[Vituri2024FunctionalReplayCallRecord, ...]
    transcript_sha256: str
    status: Vituri2024FunctionalReplayStatus
    _factory_token: InitVar[object]

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _FUNCTIONAL_REPLAY_FACTORY_TOKEN:
            raise TypeError("functional replay receipt requires the private factory token")
        for value, label in (
            (self.contract_fingerprint, "receipt contract"),
            (self.choice_fingerprint, "receipt choice"),
            (
                self.verifier_implementation_schema_fingerprint,
                "receipt verifier implementation/schema",
            ),
            (
                self.verifier_module_ast_manifest_sha256,
                "receipt verifier module AST/source manifest",
            ),
            (self.functional_provider_fingerprint, "receipt functional provider"),
            (self.detached_approval_manifest_sha256, "receipt approval manifest"),
            (self.expected_array_payload_manifest_sha256, "receipt expected array manifest"),
            (self.array_replay_receipt_fingerprint, "receipt array replay"),
            (self.array_replay_payload_manifest_sha256, "receipt array manifest"),
            (self.functional_payload_manifest_sha256, "receipt probe payload"),
            (self.affine_anchor_inventory_sha256, "receipt anchor inventory"),
            (self.q0_probe_inventory_sha256, "receipt q0 inventory"),
            (self.signed_q_probe_inventory_sha256, "receipt signed-q inventory"),
            (self.q_probe_inventory_sha256, "receipt q chart inventory"),
            (
                self.direct_dependency_trace_transcript_sha256,
                "receipt direct dependency trace transcript",
            ),
            (self.transcript_sha256, "receipt transcript"),
        ):
            _sha256(value, label)
        _text(self.detached_approval_provenance, "receipt approval provenance")
        if (
            self.verifier_implementation_schema_fingerprint
            != FUNCTIONAL_REPLAY_VERIFIER_IMPLEMENTATION_SCHEMA_FINGERPRINT
        ):
            raise ValueError("receipt verifier implementation/schema changed")
        if (
            self.verifier_module_ast_manifest_sha256
            != functional_replay_module_ast_manifest_sha256()
        ):
            raise ValueError("receipt verifier AST/source manifest changed")
        if not self.approval_precedes_execution or self.scope != FUNCTIONAL_REPLAY_SCOPE:
            raise ValueError("receipt approval order/scope changed")
        expected_counts = (
            len(AFFINE_ANCHOR_LABELS),
            len(Q0_PROBE_LABELS),
            len(SIGNED_Q_PROBE_LABELS),
            len(Q_CHART_LABELS),
        )
        actual_counts = tuple(
            _strict_int(value, "receipt inventory count")
            for value in (
                self.affine_anchor_count,
                self.q0_probe_count,
                self.signed_q_probe_count,
                self.q_chart_count,
            )
        )
        if actual_counts != expected_counts:
            raise ValueError("receipt exact registered inventory counts changed")
        if self.expected_array_payload_manifest_sha256 != self.array_replay_payload_manifest_sha256:
            raise ValueError("receipt expected/actual array manifests differ")
        if type(self.source_anchor) is not Vituri2024FunctionalReplayAnchorCheck:
            raise TypeError("receipt requires a typed source anchor check")
        inventories: tuple[tuple[object, ...], ...] = (
            tuple(self.scalar_steps),
            tuple(self.scalar_local_gates),
            tuple(self.matrix_steps),
            tuple(self.matrix_local_gates),
            tuple(self.reciprocity),
            tuple(self.call_records),
        )
        types = (
            Vituri2024FunctionalReplayScalarStep,
            Vituri2024FunctionalReplayScalarLocalGate,
            Vituri2024FunctionalReplayMatrixStep,
            Vituri2024FunctionalReplayMatrixLocalGate,
            Vituri2024FunctionalReplayReciprocity,
            Vituri2024FunctionalReplayCallRecord,
        )
        if any(not items or any(type(item) is not item_type for item in items)
               for items, item_type in zip(inventories, types)):
            raise TypeError("functional replay receipt evidence inventory is incomplete")
        calls = inventories[-1]
        if tuple(item.sequence_index for item in calls) != tuple(range(len(calls))):
            raise ValueError("functional replay call transcript sequence is not contiguous")
        if self.transcript_sha256 != _fingerprint([asdict(item) for item in calls]):
            raise ValueError("functional replay transcript hash mismatch")
        trace_hashes = tuple(
            item.dependency_trace_sha256
            for item in calls
            if item.dependency_trace_sha256 is not None
        )
        if self.direct_dependency_trace_transcript_sha256 != _fingerprint(trace_hashes):
            raise ValueError("direct dependency trace transcript hash mismatch")
        for name in (
            "scalar_slope_stability_max_abs",
            "matrix_slope_stability_max_frobenius",
            "scalar_informativeness_max_abs",
            "matrix_informativeness_max_frobenius",
        ):
            _nonnegative(getattr(self, name), f"receipt {name}")
        if type(self.status) is not Vituri2024FunctionalReplayStatus:
            raise TypeError("functional replay receipt requires factory-created status")
        for name, items in zip(
            (
                "scalar_steps",
                "scalar_local_gates",
                "matrix_steps",
                "matrix_local_gates",
                "reciprocity",
                "call_records",
            ),
            inventories,
        ):
            object.__setattr__(self, name, items)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


def _provider_snapshot(provider: object) -> dict[str, object]:
    return {
        name: getattr(provider, name)
        for name in FUNCTIONAL_REPLAY_PROVIDER_METADATA_FIELDS
    }


def _validate_snapshot(
    binding: Vituri2024HalfMetalHFProviderBinding,
    contract: Vituri2024FunctionalReplayContract,
    snapshot: dict[str, object],
) -> None:
    if type(snapshot) is not dict or tuple(snapshot) != FUNCTIONAL_REPLAY_PROVIDER_METADATA_FIELDS:
        raise ValueError("functional provider metadata snapshot fields/order mismatch")
    spec = binding.spec
    assert spec.geometry is not None
    assert spec.ensemble is not None
    assert spec.scf_policy is not None
    assert spec.shared_functional is not None
    assert spec.attested_source is not None
    expected: dict[str, object] = {
        "provider_fingerprint": contract.provider_fingerprint,
        "source_commit": contract.source_commit,
        "source_artifact_sha256": contract.source_artifact_sha256,
        "spec_fingerprint": contract.spec_fingerprint,
        "geometry_receipt_fingerprint": contract.geometry_receipt_fingerprint,
        "ensemble_receipt_fingerprint": contract.ensemble_receipt_fingerprint,
        "scf_policy_receipt_fingerprint": spec.scf_policy.fingerprint,
        "shared_functional_receipt_fingerprint": contract.shared_functional_receipt_fingerprint,
        "attested_source_receipt_fingerprint": contract.attested_source_receipt_fingerprint,
        "finite_area_receipt_fingerprint": spec.geometry.finite_area_receipt_fingerprint,
        "interaction_receipt_fingerprint": contract.interaction_receipt_fingerprint,
        "normal_order_reference_fingerprint": contract.normal_order_reference_fingerprint,
        "q0_policy_fingerprint": contract.q0_policy_fingerprint,
        "source_state_sha256": contract.source_state_sha256,
        "scalar_energy_implementation_fingerprint": (
            spec.shared_functional.scalar_energy.implementation_fingerprint
        ),
        "fock_derivative_implementation_fingerprint": (
            spec.shared_functional.fock_derivative.implementation_fingerprint
        ),
        "finite_q_hessian_implementation_fingerprint": (
            spec.shared_functional.finite_q_hessian.implementation_fingerprint
        ),
        "interaction_form_factor_implementation_fingerprint": (
            spec.shared_functional.interaction_form_factor.implementation_fingerprint
        ),
        "replay_loader_implementation_fingerprint": (
            contract.replay_loader_implementation_fingerprint
        ),
        "replay_payload_schema_fingerprint": source_schema(spec),
        "functional_provider_fingerprint": contract.functional_provider_fingerprint,
        "functional_probe_loader_implementation_fingerprint": (
            contract.functional_probe_loader_implementation_fingerprint
        ),
        "functional_replay_payload_schema_fingerprint": (
            contract.functional_replay_payload_schema_fingerprint
        ),
        "functional_replay_abi_fingerprint": contract.functional_replay_abi_fingerprint,
        "direct_displaced_fock_implementation_fingerprint": (
            contract.direct_displaced_fock_implementation_fingerprint
        ),
        "direct_interaction_builder_implementation_fingerprint": (
            contract.direct_interaction_builder_implementation_fingerprint
        ),
        "direct_full_fock_builder_implementation_fingerprint": (
            contract.direct_full_fock_builder_implementation_fingerprint
        ),
        "direct_builder_dependency_archive_fingerprint": (
            contract.direct_builder_dependency_archive_fingerprint
        ),
        "direct_displaced_fock_construction": DIRECT_DISPLACED_FOCK_CONSTRUCTION,
        "fock_output": FOCK_OUTPUT_CONVENTION,
        "stored_density_pairing": STORED_DENSITY_PAIRING,
        "density_direction_convention": FIXED_DENSITY_DIRECTION_CONVENTION,
    }
    if tuple(expected) != FUNCTIONAL_REPLAY_PROVIDER_METADATA_FIELDS:
        raise RuntimeError("internal functional provider snapshot mapping is incomplete")
    for name, required in expected.items():
        actual = snapshot[name]
        if name == "source_commit":
            _commit(actual, f"provider snapshot {name}")
        elif name in (
            "direct_displaced_fock_construction",
            "fock_output",
            "stored_density_pairing",
            "density_direction_convention",
        ):
            _text(actual, f"provider snapshot {name}")
        else:
            _sha256(actual, f"provider snapshot {name}")
        if actual != required:
            raise ValueError(f"functional provider snapshot {name} mismatch")
    if (
        snapshot["direct_displaced_fock_implementation_fingerprint"]
        == snapshot["finite_q_hessian_implementation_fingerprint"]
    ):
        raise ValueError("direct displaced-Fock and finite-q-Hessian fingerprints must differ")


def source_schema(spec: Vituri2024HalfMetalHFSpec) -> str:
    assert spec.attested_source is not None
    return spec.attested_source.replay_payload_schema_fingerprint


def _assert_snapshot_unchanged(
    provider: object, baseline: dict[str, object], label: str
) -> None:
    after = _provider_snapshot(provider)
    if after != baseline:
        changed = tuple(
            name
            for name in FUNCTIONAL_REPLAY_PROVIDER_METADATA_FIELDS
            if after[name] != baseline[name]
        )
        raise ValueError(
            f"functional provider metadata mutated during {label}: " + ", ".join(changed)
        )


def _argument_state(array: np.ndarray) -> tuple[str, tuple[int, ...], str, bool]:
    return (
        canonical_array_sha256(array),
        array.shape,
        array.dtype.str,
        bool(array.flags.writeable),
    )


def _copy_call_argument(array: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{label} must be an array")
    return np.array(array, dtype=array.dtype, order="C", copy=True)


def _output_array(value: object, *, label: str, shape: tuple[int, ...]) -> ComplexArray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must return a numpy.ndarray")
    if value.dtype != np.dtype(np.complex128):
        raise TypeError(f"{label} output dtype must be exactly complex128")
    if value.shape != shape:
        raise ValueError(f"{label} output shape must be exactly {shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} output must be finite")
    result = np.array(value, dtype=np.complex128, order="C", copy=True)
    result.flags.writeable = False
    return result


class _CallRecorder:
    def __init__(self, provider: object, baseline: dict[str, object]) -> None:
        self.provider = provider
        self.baseline = baseline
        self.records: list[Vituri2024FunctionalReplayCallRecord] = []

    def _append(
        self,
        method: str,
        argument_hashes: tuple[str, ...],
        keywords: dict[str, object],
        output_hash: str,
        trace_hash: str | None = None,
    ) -> None:
        keyword_payload = {
            name: (_array_manifest(value) if isinstance(value, np.ndarray) else value)
            for name, value in keywords.items()
        }
        self.records.append(
            Vituri2024FunctionalReplayCallRecord(
                sequence_index=len(self.records),
                method=method,
                argument_hashes=argument_hashes,
                keyword_fingerprint=_fingerprint(keyword_payload),
                output_sha256=output_hash,
                dependency_trace_sha256=trace_hash,
            )
        )

    def load_arrays(self, source_artifact_sha256: str) -> Vituri2024HalfMetalHFReplayPayload:
        _assert_snapshot_unchanged(self.provider, self.baseline, "before array payload loader")
        argument_hash = _fingerprint({"source_artifact_sha256": source_artifact_sha256})
        value = self.provider.load_half_metal_replay_payload(source_artifact_sha256)
        _assert_snapshot_unchanged(self.provider, self.baseline, "array payload loader")
        if type(value) is not Vituri2024HalfMetalHFReplayPayload:
            raise TypeError("array payload loader returned the wrong typed payload")
        output_hash = _fingerprint(
            {
                "mesh": canonical_array_sha256(value.mesh),
                "active_band_states": canonical_array_sha256(value.active_band_states),
                "h0": canonical_array_sha256(value.h0),
                "interaction_h": canonical_array_sha256(value.interaction_h),
                "fock": canonical_array_sha256(value.fock),
                "projector": canonical_array_sha256(value.projector),
                "energies": canonical_array_sha256(value.energies),
                "occupations": canonical_array_sha256(value.occupations),
            }
        )
        self._append("load_half_metal_replay_payload", (argument_hash,), {}, output_hash)
        return value

    def load_probes(self, source_artifact_sha256: str) -> Vituri2024FunctionalReplayPayload:
        _assert_snapshot_unchanged(self.provider, self.baseline, "before functional probe loader")
        argument_hash = _fingerprint({"source_artifact_sha256": source_artifact_sha256})
        value = self.provider.load_functional_replay_payload(source_artifact_sha256)
        _assert_snapshot_unchanged(self.provider, self.baseline, "functional probe loader")
        if type(value) is not Vituri2024FunctionalReplayPayload:
            raise TypeError("functional probe loader returned the wrong typed payload")
        self._append("load_functional_replay_payload", (argument_hash,), {}, value.manifest_sha256)
        return value

    def array_call(
        self,
        method: str,
        arrays: tuple[np.ndarray, ...],
        *,
        shape: tuple[int, ...],
        keywords: dict[str, object] | None = None,
    ) -> ComplexArray:
        _assert_snapshot_unchanged(self.provider, self.baseline, f"before {method}")
        call_arrays = tuple(_copy_call_argument(array, f"{method} argument") for array in arrays)
        states = tuple(_argument_state(array) for array in call_arrays)
        clean_keywords = {} if keywords is None else dict(keywords)
        function = getattr(self.provider, method)
        value = function(*call_arrays, **clean_keywords)
        for array, state in zip(call_arrays, states):
            if _argument_state(array) != state:
                raise ValueError(f"provider mutated an input during {method}")
        _assert_snapshot_unchanged(self.provider, self.baseline, method)
        output = _output_array(value, label=method, shape=shape)
        self._append(
            method,
            tuple(state[0] for state in states),
            clean_keywords,
            canonical_array_sha256(output),
        )
        return output

    def scalar_call(self, interaction_h: np.ndarray, h0: np.ndarray, density: np.ndarray) -> float:
        _assert_snapshot_unchanged(self.provider, self.baseline, "before evaluate_scalar_energy")
        arrays = tuple(
            _copy_call_argument(array, "evaluate_scalar_energy argument")
            for array in (interaction_h, h0, density)
        )
        states = tuple(_argument_state(array) for array in arrays)
        value = self.provider.evaluate_scalar_energy(*arrays)
        for array, state in zip(arrays, states):
            if _argument_state(array) != state:
                raise ValueError("provider mutated an input during evaluate_scalar_energy")
        _assert_snapshot_unchanged(self.provider, self.baseline, "evaluate_scalar_energy")
        result = _finite(value, "evaluate_scalar_energy output")
        output_hash = _fingerprint({"schema": "vituri2024_scalar_output_v1", "value": result})
        self._append(
            "evaluate_scalar_energy", tuple(state[0] for state in states), {}, output_hash
        )
        return result

    def hessian_call(self, perturbation: np.ndarray, chart: Vituri2024SignedQProbeChart) -> ComplexArray:
        keywords: dict[str, object] = {
            "q_probe_index": chart.q_probe_index,
            "mesh_displacement": _copy_call_argument(chart.mesh_displacement, "hessian mesh displacement"),
            "cartesian_q": _copy_call_argument(chart.cartesian_q, "hessian Cartesian q"),
            "target_maps": _copy_call_argument(chart.target_maps, "hessian target maps"),
            "reverse_edge_map": _copy_call_argument(chart.reverse_edge_map, "hessian reverse map"),
        }
        keyword_states = {
            name: _argument_state(value)
            for name, value in keywords.items()
            if isinstance(value, np.ndarray)
        }
        output = self.array_call(
            "evaluate_finite_q_hessian",
            (perturbation,),
            shape=perturbation.shape,
            keywords=keywords,
        )
        for name, state in keyword_states.items():
            if _argument_state(keywords[name]) != state:  # type: ignore[arg-type]
                raise ValueError("provider mutated a Hessian chart input")
        return output

    def direct_call(
        self,
        density: np.ndarray,
        displacement: np.ndarray,
        chart: Vituri2024SignedQProbeChart,
    ) -> Vituri2024DirectDisplacedFockResponse:
        method = "evaluate_displaced_fock"
        _assert_snapshot_unchanged(self.provider, self.baseline, f"before {method}")
        call_arrays = tuple(
            _copy_call_argument(array, f"{method} argument")
            for array in (
                density,
                displacement,
                chart.mesh_displacement,
                chart.cartesian_q,
                chart.target_maps,
                chart.reverse_edge_map,
            )
        )
        states = tuple(_argument_state(array) for array in call_arrays)
        nonce = _fingerprint(
            {
                "schema": "vituri2024_direct_call_nonce_v1",
                "sequence_index": len(self.records),
                "q_probe_index": chart.q_probe_index,
                "argument_hashes": tuple(state[0] for state in states),
            }
        )
        value = self.provider.evaluate_displaced_fock(
            call_arrays[0],
            call_arrays[1],
            q_probe_index=chart.q_probe_index,
            mesh_displacement=call_arrays[2],
            cartesian_q=call_arrays[3],
            target_maps=call_arrays[4],
            reverse_edge_map=call_arrays[5],
            caller_nonce=nonce,
        )
        for array, state in zip(call_arrays, states):
            if _argument_state(array) != state:
                raise ValueError(f"provider mutated an input during {method}")
        _assert_snapshot_unchanged(self.provider, self.baseline, method)
        if type(value) is not Vituri2024DirectDisplacedFockResponse:
            raise TypeError("direct displaced-Fock call must return a typed response")
        if value.caller_nonce_sha256 != nonce:
            raise ValueError("direct displaced-Fock response caller nonce mismatch")
        output = _output_array(value.response, label=method, shape=displacement.shape)
        trace = value.dependency_trace
        expected_read_count = int(chart.target_maps.size)
        expected_trace = (
            (trace.q_probe_index, chart.q_probe_index, "q index"),
            (trace.q_label, chart.q_label, "q label"),
            (
                trace.interaction_builder_implementation_fingerprint,
                self.baseline["direct_interaction_builder_implementation_fingerprint"],
                "interaction builder fingerprint",
            ),
            (
                trace.full_fock_builder_implementation_fingerprint,
                self.baseline["direct_full_fock_builder_implementation_fingerprint"],
                "full-Fock builder fingerprint",
            ),
            (trace.target_maps_sha256, states[4][0], "target-map fingerprint"),
            (trace.reverse_edge_map_sha256, states[5][0], "reverse-map fingerprint"),
            (trace.target_map_read_count, expected_read_count, "target-map read count"),
            (
                trace.reverse_edge_map_read_count,
                expected_read_count,
                "reverse-map read count",
            ),
            (trace.finite_q_hessian_call_count, 0, "finite-q-Hessian call count"),
        )
        for actual, required, label in expected_trace:
            if actual != required:
                raise ValueError(f"direct dependency trace {label} mismatch")
        if trace.interaction_builder_call_count < 1:
            raise ValueError("direct dependency trace interaction-builder call count must be >=1")
        if trace.full_fock_builder_call_count < 1:
            raise ValueError("direct dependency trace full-Fock builder call count must be >=1")
        if value.response_token_sha256 != _fingerprint(
            {
                "schema": "vituri2024_direct_displaced_fock_response_v1",
                "caller_nonce_sha256": nonce,
                "response_sha256": canonical_array_sha256(output),
                "dependency_trace_sha256": trace.fingerprint,
            }
        ):
            raise ValueError("direct displaced-Fock response token mismatch")
        keywords = {
            "q_probe_index": chart.q_probe_index,
            "caller_nonce": nonce,
            "q_label": chart.q_label,
        }
        self._append(
            method,
            tuple(state[0] for state in states),
            keywords,
            value.response_token_sha256,
            trace.fingerprint,
        )
        return value


def _pairing(left: np.ndarray, right: np.ndarray, nk: int) -> float:
    return float(np.real(np.sum(left * right)) / nk)


def _pairing_diagnostics(left: np.ndarray, right: np.ndarray, nk: int) -> tuple[float, float, int]:
    value = _pairing(left, right, nk)
    termwise = float(np.sum(np.abs(left * right)) / nk)
    operation_count = int(4 * left.size + 1)
    return value, termwise, operation_count


def _frobenius(array: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.abs(array) ** 2)))


def _max_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array)))


def _gamma(operation_count: int) -> float:
    epsilon = np.finfo(np.float64).eps
    product = operation_count * epsilon
    if product >= 0.5:
        raise ValueError("roundoff operation count is outside the registered regime")
    return product / (1.0 - product)


def _termwise_roundoff(choice: Vituri2024FunctionalReplayChoice, termwise: float, count: int) -> float:
    effective_count = count + int(choice.roundoff_ulps)
    return _gamma(effective_count) * max(termwise, choice.informativeness_floor)


def _entrywise_bound(
    choice: Vituri2024FunctionalReplayChoice,
    *,
    result_scale: float,
    termwise_magnitude: float,
    operation_count: int,
) -> tuple[float, float]:
    clean_result_scale = _nonnegative(result_scale, "entrywise result scale")
    local_scale = max(clean_result_scale, choice.informativeness_floor)
    termwise = _nonnegative(termwise_magnitude, "entrywise termwise magnitude")
    count = _strict_int(operation_count, "entrywise operation count")
    if termwise < clean_result_scale:
        raise ValueError("entrywise termwise magnitude is below its result scale")
    if count < 1:
        raise ValueError("entrywise operation count must be positive")
    roundoff = _termwise_roundoff(choice, termwise, count)
    return (
        roundoff,
        choice.absolute_tolerance
        + choice.relative_tolerance * local_scale
        + roundoff,
    )


def _validate_source_q_charts(
    spec: Vituri2024HalfMetalHFSpec,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
    probe_payload: Vituri2024FunctionalReplayPayload,
    choice: Vituri2024FunctionalReplayChoice,
) -> None:
    assert spec.geometry is not None
    expected_mesh_shape = spec.geometry.mesh_shape
    tolerance = choice.source_mesh_q_coordinate_tolerance_inverse_angstrom
    for chart in probe_payload.q_charts:
        if chart.mesh_shape != expected_mesh_shape:
            raise ValueError(
                "signed-q chart mesh_shape does not equal spec.geometry.mesh_shape"
            )
        for sign_index, multiplier in enumerate((1.0, -1.0)):
            valid = chart.validity_masks[sign_index]
            source_indices = chart.source_k_indices[valid]
            target_indices = chart.target_maps[sign_index, valid]
            actual_q = (
                source_payload.mesh[target_indices]
                - source_payload.mesh[source_indices]
            )
            expected_q = multiplier * chart.cartesian_q
            if np.any(np.abs(actual_q - expected_q[None, :]) > tolerance):
                raise ValueError(
                    "signed-q Cartesian q does not match source momentum mesh edges "
                    "within the registered inverse-angstrom coordinate tolerance"
                )

def _require_exact_invalid_zero(
    array: np.ndarray, chart: Vituri2024SignedQProbeChart, label: str
) -> None:
    for sign_index in range(2):
        invalid = ~chart.validity_masks[sign_index]
        values = array[sign_index, :, :, invalid]
        if not np.array_equal(values, np.zeros_like(values)):
            raise ValueError(f"{label} has nonzero invalid-edge output")


def _validate_contract_against_binding(
    binding: Vituri2024HalfMetalHFProviderBinding,
    contract: Vituri2024FunctionalReplayContract,
) -> None:
    if type(binding) is not Vituri2024HalfMetalHFProviderBinding:
        raise TypeError("functional replay requires a typed complete provider binding")
    if type(contract) is not Vituri2024FunctionalReplayContract:
        raise TypeError("functional replay requires a typed detached contract")
    if contract.choice_fingerprint != contract.choice.fingerprint:
        raise ValueError("functional replay contract choice fingerprint mismatch")
    if (
        contract.verifier_implementation_schema_fingerprint
        != FUNCTIONAL_REPLAY_VERIFIER_IMPLEMENTATION_SCHEMA_FINGERPRINT
    ):
        raise ValueError("functional replay contract verifier implementation/schema changed")
    spec = binding.spec
    spec.require_receipt_set_complete()
    assert spec.geometry is not None
    assert spec.ensemble is not None
    assert spec.shared_functional is not None
    assert spec.attested_source is not None
    expected = (
        (contract.provider_fingerprint, spec.geometry.provider_fingerprint, "provider"),
        (contract.source_commit, spec.shared_functional.source_commit, "source commit"),
        (
            contract.source_artifact_sha256,
            spec.shared_functional.source_artifact_sha256,
            "source artifact",
        ),
        (contract.spec_fingerprint, spec.fingerprint, "spec"),
        (contract.source_state_sha256, spec.attested_source.source_state_sha256, "source state"),
        (contract.geometry_receipt_fingerprint, spec.geometry.fingerprint, "geometry"),
        (contract.ensemble_receipt_fingerprint, spec.ensemble.fingerprint, "ensemble"),
        (
            contract.normal_order_reference_fingerprint,
            spec.ensemble.normal_order_reference_fingerprint,
            "normal-order reference",
        ),
        (contract.q0_policy_fingerprint, spec.ensemble.q0_policy_fingerprint, "q0 policy"),
        (
            contract.interaction_receipt_fingerprint,
            spec.shared_functional.interaction_receipt_fingerprint,
            "interaction",
        ),
        (
            contract.shared_functional_receipt_fingerprint,
            spec.shared_functional.fingerprint,
            "shared functional",
        ),
        (
            contract.attested_source_receipt_fingerprint,
            spec.attested_source.fingerprint,
            "attested source",
        ),
        (
            contract.expected_array_payload_manifest_sha256,
            expected_array_payload_manifest_sha256(spec),
            "expected array payload manifest",
        ),
        (
            contract.q0_probe_inventory_sha256,
            spec.shared_functional.fock_finite_difference.perturbation_inventory_sha256,
            "q0 finite-difference probe inventory",
        ),
        (
            contract.signed_q_probe_inventory_sha256,
            spec.shared_functional.hessian_finite_difference.perturbation_inventory_sha256,
            "finite-q Hessian probe inventory",
        ),
        (
            contract.q_probe_inventory_sha256,
            spec.shared_functional.hessian_finite_difference.q_probe_inventory_sha256,
            "finite-q Hessian q inventory",
        ),
        (
            contract.fock_step_ladder,
            spec.shared_functional.fock_finite_difference.finite_difference_step_ladder,
            "Fock finite-difference steps",
        ),
        (
            contract.hessian_step_ladder,
            spec.shared_functional.hessian_finite_difference.finite_difference_step_ladder,
            "Hessian finite-difference steps",
        ),
        (
            contract.replay_loader_implementation_fingerprint,
            spec.attested_source.replay_loader_implementation_fingerprint,
            "array replay loader",
        ),
    )
    for actual, required, label in expected:
        if actual != required:
            raise ValueError(f"functional replay contract/{label} mismatch")
    if (
        contract.direct_displaced_fock_implementation_fingerprint
        == spec.shared_functional.finite_q_hessian.implementation_fingerprint
    ):
        raise ValueError("direct displaced-Fock implementation fingerprint equals Hessian")
    if (
        spec.shared_functional.fock_output != contract.choice.fock_output
        or spec.shared_functional.stored_density_pairing != contract.choice.stored_density_pairing
        or spec.shared_functional.density_direction_convention
        != contract.choice.density_direction_convention
    ):
        raise ValueError("functional replay contract/shared-functional conventions mismatch")


def _validate_array_reload(
    payload: Vituri2024HalfMetalHFReplayPayload,
    receipt: Vituri2024HalfMetalHFReplayReceipt,
) -> None:
    expected = (
        (canonical_array_sha256(payload.mesh), receipt.hashes.ordered_momentum_mesh_sha256, "mesh"),
        (
            canonical_array_sha256(payload.active_band_states),
            receipt.hashes.active_band_states_sha256,
            "active-band states",
        ),
        (canonical_array_sha256(payload.h0), receipt.hashes.h0_sha256, "h0"),
        (
            canonical_array_sha256(payload.interaction_h),
            receipt.hashes.interaction_h_sha256,
            "interaction_h",
        ),
        (canonical_array_sha256(payload.fock), receipt.hashes.ordered_fock_sha256, "Fock"),
        (
            canonical_array_sha256(payload.projector),
            receipt.hashes.ordered_projector_sha256,
            "projector",
        ),
        (
            canonical_array_sha256(payload.energies),
            receipt.hashes.ordered_energies_sha256,
            "energies",
        ),
        (
            canonical_array_sha256(payload.occupations),
            receipt.hashes.ordered_occupations_sha256,
            "occupations",
        ),
    )
    for actual, required, label in expected:
        if actual != required:
            raise ValueError(f"functional replay reloaded canonical {label} hash mismatch")


def _expected_active_response_signs(probe_index: int) -> tuple[int, ...]:
    local_kind = probe_index % 3
    if local_kind == 0:
        return (1,)
    if local_kind == 1:
        return (0,)
    return (0, 1)


def replay_vituri2024_half_metal_hf_functional(
    binding: Vituri2024HalfMetalHFProviderBinding,
    contract: Vituri2024FunctionalReplayContract,
) -> Vituri2024FunctionalReplayReceipt:
    """Execute all detached, pre-registered local probes and no broader run."""

    _validate_contract_against_binding(binding, contract)
    provider = binding.provider
    if not isinstance(provider, Vituri2024FunctionalReplayProviderProtocol):
        raise TypeError("provider is missing the runtime functional replay protocol")
    for method_name in (
        "load_half_metal_replay_payload",
        "load_functional_replay_payload",
        "evaluate_scalar_energy",
        "evaluate_fock_derivative",
        "evaluate_finite_q_hessian",
        "evaluate_displaced_fock",
    ):
        if not callable(getattr(provider, method_name, None)):
            raise TypeError(f"functional replay provider {method_name} must be callable")

    baseline = _provider_snapshot(provider)
    _validate_snapshot(binding, contract, baseline)
    frozen_baseline = MappingProxyType(dict(baseline))
    recorder = _CallRecorder(provider, baseline)

    verifier_module_ast_manifest_sha256 = (
        functional_replay_module_ast_manifest_sha256()
    )
    if (
        contract.verifier_module_ast_manifest_sha256
        != verifier_module_ast_manifest_sha256
    ):
        raise ValueError("functional replay verifier AST/source manifest changed")
    array_receipt = replay_vituri2024_half_metal_hf_arrays(binding)
    _assert_snapshot_unchanged(provider, baseline, "existing array replay")
    if (
        array_receipt.hashes.payload_manifest_sha256
        != contract.expected_array_payload_manifest_sha256
    ):
        raise ValueError("array replay payload differs from detached expected manifest")
    array_loader_output_hash = _fingerprint(
        {
            "mesh": array_receipt.hashes.ordered_momentum_mesh_sha256,
            "active_band_states": array_receipt.hashes.active_band_states_sha256,
            "h0": array_receipt.hashes.h0_sha256,
            "interaction_h": array_receipt.hashes.interaction_h_sha256,
            "fock": array_receipt.hashes.ordered_fock_sha256,
            "projector": array_receipt.hashes.ordered_projector_sha256,
            "energies": array_receipt.hashes.ordered_energies_sha256,
            "occupations": array_receipt.hashes.ordered_occupations_sha256,
        }
    )
    recorder._append(
        "load_half_metal_replay_payload",
        (_fingerprint({"source_artifact_sha256": contract.source_artifact_sha256}),),
        {},
        array_loader_output_hash,
    )
    source_payload = recorder.load_arrays(contract.source_artifact_sha256)
    _validate_array_reload(source_payload, array_receipt)
    if recorder.records[0].output_sha256 != recorder.records[1].output_sha256:
        raise ValueError("reloaded array payload is not deterministic")
    probe_payload = recorder.load_probes(contract.source_artifact_sha256)

    expected_payload_identity = (
        (probe_payload.provider_fingerprint, frozen_baseline["provider_fingerprint"], "base provider"),
        (
            probe_payload.functional_provider_fingerprint,
            frozen_baseline["functional_provider_fingerprint"],
            "functional provider",
        ),
        (probe_payload.source_commit, frozen_baseline["source_commit"], "source commit"),
        (
            probe_payload.source_artifact_sha256,
            frozen_baseline["source_artifact_sha256"],
            "source artifact",
        ),
        (probe_payload.spec_fingerprint, frozen_baseline["spec_fingerprint"], "spec"),
        (
            probe_payload.source_state_sha256,
            frozen_baseline["source_state_sha256"],
            "source state",
        ),
        (
            probe_payload.functional_probe_loader_implementation_fingerprint,
            frozen_baseline["functional_probe_loader_implementation_fingerprint"],
            "probe loader",
        ),
        (
            probe_payload.functional_replay_payload_schema_fingerprint,
            frozen_baseline["functional_replay_payload_schema_fingerprint"],
            "probe schema",
        ),
        (
            probe_payload.affine_anchor_inventory_sha256,
            contract.affine_anchor_inventory_sha256,
            "affine anchor inventory",
        ),
        (
            probe_payload.q0_probe_inventory_sha256,
            contract.q0_probe_inventory_sha256,
            "q0 probe inventory",
        ),
        (
            probe_payload.signed_q_probe_inventory_sha256,
            contract.signed_q_probe_inventory_sha256,
            "signed-q probe inventory",
        ),
        (
            probe_payload.q_probe_inventory_sha256,
            contract.q_probe_inventory_sha256,
            "q inventory",
        ),
    )
    for actual, required, label in expected_payload_identity:
        if actual != required:
            raise ValueError(f"functional replay payload {label} mismatch")

    _validate_source_q_charts(
        binding.spec, source_payload, probe_payload, contract.choice
    )

    nk = source_payload.projector.shape[2]
    matrix_shape = source_payload.projector.shape
    signed_shape = (2, 4, 4, nk)

    # Source-anchor roundoff uses explicit unsuppressed term magnitudes.  The
    # result scales remain separate and control only the registered relative term.
    fock0 = recorder.array_call(
        "evaluate_fock_derivative", (source_payload.projector,), shape=matrix_shape
    )
    fock_residual = _max_abs(fock0 - source_payload.fock)
    fock_result_scale = max(_max_abs(fock0), _max_abs(source_payload.fock))
    fock_termwise = _max_abs(np.abs(fock0) + np.abs(source_payload.fock))
    fock_operation_count = 2
    fock_roundoff, fock_bound = _entrywise_bound(
        contract.choice,
        result_scale=fock_result_scale,
        termwise_magnitude=fock_termwise,
        operation_count=fock_operation_count,
    )
    interaction0 = fock0 - source_payload.h0
    interaction_residual = _max_abs(interaction0 - source_payload.interaction_h)
    interaction_result_scale = max(
        _max_abs(interaction0), _max_abs(source_payload.interaction_h)
    )
    interaction_termwise = _max_abs(
        np.abs(fock0) + np.abs(source_payload.h0) + np.abs(source_payload.interaction_h)
    )
    interaction_operation_count = 3
    interaction_roundoff, interaction_bound = _entrywise_bound(
        contract.choice,
        result_scale=interaction_result_scale,
        termwise_magnitude=interaction_termwise,
        operation_count=interaction_operation_count,
    )
    selected_energy = binding.spec.attested_source.selected_branch_energy_ev  # type: ignore[union-attr]
    energy0 = recorder.scalar_call(interaction0, source_payload.h0, source_payload.projector)
    energy_residual = abs(energy0 - selected_energy)
    energy_result_scale = max(abs(energy0), abs(selected_energy))
    energy_termwise = abs(energy0) + abs(selected_energy)
    energy_operation_count = 2
    energy_roundoff, energy_bound = _entrywise_bound(
        contract.choice,
        result_scale=energy_result_scale,
        termwise_magnitude=energy_termwise,
        operation_count=energy_operation_count,
    )
    source_anchor = Vituri2024FunctionalReplayAnchorCheck(
        fock_entrywise_residual=fock_residual,
        fock_entrywise_result_scale=fock_result_scale,
        fock_entrywise_termwise_magnitude=fock_termwise,
        fock_entrywise_operation_count=fock_operation_count,
        fock_entrywise_roundoff_contribution=fock_roundoff,
        fock_entrywise_registered_bound=fock_bound,
        interaction_entrywise_residual=interaction_residual,
        interaction_entrywise_result_scale=interaction_result_scale,
        interaction_entrywise_termwise_magnitude=interaction_termwise,
        interaction_entrywise_operation_count=interaction_operation_count,
        interaction_entrywise_roundoff_contribution=interaction_roundoff,
        interaction_entrywise_registered_bound=interaction_bound,
        scalar_energy_residual=energy_residual,
        scalar_energy_result_scale=energy_result_scale,
        scalar_energy_termwise_magnitude=energy_termwise,
        scalar_energy_operation_count=energy_operation_count,
        scalar_energy_roundoff_contribution=energy_roundoff,
        scalar_energy_registered_bound=energy_bound,
    )

    # Complete scalar E->F phase runs before any finite-q provider call.  Thus
    # conjugation/transpose canaries must fail through scalar records first.
    scalar_records: list[Vituri2024FunctionalReplayScalarStep] = []
    scalar_gates: list[Vituri2024FunctionalReplayScalarLocalGate] = []
    informative_by_probe = [False] * len(Q0_PROBE_LABELS)
    for anchor_index, (anchor_label, offset) in enumerate(
        zip(probe_payload.affine_anchor_labels, probe_payload.affine_anchor_offsets)
    ):
        anchor_density = source_payload.projector + offset
        anchor_fock = recorder.array_call(
            "evaluate_fock_derivative", (anchor_density,), shape=matrix_shape
        )
        anchor_interaction = anchor_fock - source_payload.h0
        recorder.scalar_call(anchor_interaction, source_payload.h0, anchor_density)
        for probe_index, (probe_label, direction) in enumerate(
            zip(probe_payload.q0_labels, probe_payload.q0_directions)
        ):
            target, pairing_terms, pairing_ops = _pairing_diagnostics(
                anchor_fock, direction, nk
            )
            informativeness = abs(target)
            if informativeness >= contract.choice.informativeness_floor:
                informative_by_probe[probe_index] = True
            local_steps: list[Vituri2024FunctionalReplayScalarStep] = []
            slopes: list[float] = []
            for step in contract.fock_step_ladder:
                density_plus = anchor_density + step * direction
                density_minus = anchor_density - step * direction
                fock_plus = recorder.array_call(
                    "evaluate_fock_derivative", (density_plus,), shape=matrix_shape
                )
                fock_minus = recorder.array_call(
                    "evaluate_fock_derivative", (density_minus,), shape=matrix_shape
                )
                energy_plus = recorder.scalar_call(
                    fock_plus - source_payload.h0, source_payload.h0, density_plus
                )
                energy_minus = recorder.scalar_call(
                    fock_minus - source_payload.h0, source_payload.h0, density_minus
                )
                central = (energy_plus - energy_minus) / (2.0 * step)
                residual = abs(central - target)
                energy_terms = (abs(energy_plus) + abs(energy_minus)) / (2.0 * step)
                energy_ops = 3
                energy_roundoff = _termwise_roundoff(
                    contract.choice, energy_terms, energy_ops
                )
                pairing_roundoff = _termwise_roundoff(
                    contract.choice, pairing_terms, pairing_ops
                )
                roundoff_bound = energy_roundoff + pairing_roundoff
                local_scale = max(
                    abs(central),
                    abs(target),
                    contract.choice.informativeness_floor,
                )
                bound = (
                    contract.choice.absolute_tolerance
                    + contract.choice.relative_tolerance * local_scale
                    + roundoff_bound
                )
                record = Vituri2024FunctionalReplayScalarStep(
                    anchor_index=anchor_index,
                    anchor_label=anchor_label,
                    probe_index=probe_index,
                    probe_label=probe_label,
                    step=step,
                    central_slope=central,
                    fock_pairing=target,
                    residual=residual,
                    local_scale=local_scale,
                    energy_termwise_abs_sum=energy_terms,
                    pairing_termwise_abs_sum=pairing_terms,
                    operation_count=energy_ops + pairing_ops,
                    roundoff_bound=roundoff_bound,
                    registered_bound=bound,
                )
                local_steps.append(record)
                slopes.append(central)
            stability = max(
                abs(left - right)
                for left_index, left in enumerate(slopes)
                for right in slopes[left_index + 1 :]
            )
            stability_scale = max(
                informativeness, contract.choice.informativeness_floor
            )
            stability_roundoff = 2.0 * max(
                record.roundoff_bound for record in local_steps
            )
            stability_bound = (
                contract.choice.slope_stability_tolerance * stability_scale
                + stability_roundoff
            )
            scalar_records.extend(local_steps)
            scalar_gates.append(
                Vituri2024FunctionalReplayScalarLocalGate(
                    anchor_index=anchor_index,
                    anchor_label=anchor_label,
                    probe_index=probe_index,
                    probe_label=probe_label,
                    informativeness_abs=informativeness,
                    stability_max_abs=stability,
                    stability_scale=stability_scale,
                    stability_roundoff_bound=stability_roundoff,
                    stability_registered_bound=stability_bound,
                )
            )
    if not all(informative_by_probe):
        missing = tuple(
            Q0_PROBE_LABELS[index]
            for index, informative in enumerate(informative_by_probe)
            if not informative
        )
        raise ValueError(
            "every q0 direction must be informative at a registered affine anchor: "
            + ", ".join(missing)
        )

    matrix_records: list[Vituri2024FunctionalReplayMatrixStep] = []
    matrix_gates: list[Vituri2024FunctionalReplayMatrixLocalGate] = []
    reciprocity_records: list[Vituri2024FunctionalReplayReciprocity] = []
    hessian_actions: dict[int, ComplexArray] = {}
    for probe_index, (probe_label, packed, q_index_raw) in enumerate(
        zip(
            probe_payload.signed_q_labels,
            probe_payload.signed_q_probes,
            probe_payload.signed_q_probe_indices,
        )
    ):
        q_index = int(q_index_raw)
        chart = probe_payload.q_charts[q_index]
        hessian = recorder.hessian_call(packed, chart)
        _require_exact_invalid_zero(hessian, chart, "finite-q Hessian")
        hessian_actions[probe_index] = hessian
        zero = np.zeros(signed_shape, dtype=np.complex128)
        base_response = recorder.direct_call(source_payload.projector, zero, chart)
        _require_exact_invalid_zero(base_response.response, chart, "direct base response")
        active_signs = _expected_active_response_signs(probe_index)
        derivatives_by_sign: dict[int, list[np.ndarray]] = {
            sign_index: [] for sign_index in active_signs
        }
        steps_by_sign: dict[int, list[Vituri2024FunctionalReplayMatrixStep]] = {
            sign_index: [] for sign_index in active_signs
        }
        for sign_index in active_signs:
            if _frobenius(hessian[sign_index]) < contract.choice.informativeness_floor:
                raise ValueError(
                    "finite-q Hessian active response lane is not locally informative"
                )
        for sign_index in set((0, 1)) - set(active_signs):
            if not np.array_equal(
                hessian[sign_index], np.zeros_like(hessian[sign_index])
            ):
                raise ValueError("finite-q Hessian populated an inactive response lane")
        for step in contract.hessian_step_ladder:
            plus = recorder.direct_call(source_payload.projector, step * packed, chart)
            minus = recorder.direct_call(source_payload.projector, -step * packed, chart)
            _require_exact_invalid_zero(plus.response, chart, "direct plus displaced-Fock")
            _require_exact_invalid_zero(minus.response, chart, "direct minus displaced-Fock")
            derivative = (plus.response - minus.response) / (2.0 * step)
            for sign_index in active_signs:
                derivative_sign = derivative[sign_index]
                hessian_sign = hessian[sign_index]
                residual = _frobenius(derivative_sign - hessian_sign)
                difference_terms = float(
                    np.sum(
                        np.abs(plus.response[sign_index])
                        + np.abs(minus.response[sign_index])
                    )
                    / (2.0 * step)
                )
                operation_count = int(4 * derivative_sign.size + 1)
                roundoff_bound = _termwise_roundoff(
                    contract.choice, difference_terms, operation_count
                )
                local_scale = max(
                    _frobenius(derivative_sign),
                    _frobenius(hessian_sign),
                    contract.choice.informativeness_floor,
                )
                bound = (
                    contract.choice.absolute_tolerance
                    + contract.choice.relative_tolerance * local_scale
                    + roundoff_bound
                )
                record = Vituri2024FunctionalReplayMatrixStep(
                    q_probe_index=q_index,
                    q_label=chart.q_label,
                    probe_index=probe_index,
                    probe_label=probe_label,
                    response_sign_index=sign_index,  # type: ignore[arg-type]
                    step=step,
                    derivative_norm=_frobenius(derivative_sign),
                    hessian_norm=_frobenius(hessian_sign),
                    residual_norm=residual,
                    local_scale=local_scale,
                    difference_termwise_abs_sum=difference_terms,
                    operation_count=operation_count,
                    roundoff_bound=roundoff_bound,
                    registered_bound=bound,
                )
                steps_by_sign[sign_index].append(record)
                frozen = np.array(derivative_sign, dtype=np.complex128, copy=True)
                frozen.flags.writeable = False
                derivatives_by_sign[sign_index].append(frozen)
        for sign_index in active_signs:
            local_derivatives = derivatives_by_sign[sign_index]
            stability = max(
                _frobenius(left - right)
                for left_index, left in enumerate(local_derivatives)
                for right in local_derivatives[left_index + 1 :]
            )
            informativeness = _frobenius(hessian[sign_index])
            stability_scale = max(
                informativeness, contract.choice.informativeness_floor
            )
            stability_roundoff = 2.0 * max(
                record.roundoff_bound for record in steps_by_sign[sign_index]
            )
            stability_bound = (
                contract.choice.slope_stability_tolerance * stability_scale
                + stability_roundoff
            )
            matrix_records.extend(steps_by_sign[sign_index])
            matrix_gates.append(
                Vituri2024FunctionalReplayMatrixLocalGate(
                    q_probe_index=q_index,
                    q_label=chart.q_label,
                    probe_index=probe_index,
                    probe_label=probe_label,
                    response_sign_index=sign_index,  # type: ignore[arg-type]
                    informativeness_norm=informativeness,
                    stability_max_frobenius=stability,
                    stability_scale=stability_scale,
                    stability_roundoff_bound=stability_roundoff,
                    stability_registered_bound=stability_bound,
                )
            )

    for q_index, chart in enumerate(probe_payload.q_charts):
        probe_indices = tuple(
            index
            for index, registered_q in enumerate(probe_payload.signed_q_probe_indices)
            if int(registered_q) == q_index
        )
        for position, left_index in enumerate(probe_indices):
            for right_index in probe_indices[position + 1 :]:
                left_right, left_terms, left_ops = _pairing_diagnostics(
                    probe_payload.signed_q_probes[left_index],
                    hessian_actions[right_index],
                    nk,
                )
                right_left, right_terms, right_ops = _pairing_diagnostics(
                    probe_payload.signed_q_probes[right_index],
                    hessian_actions[left_index],
                    nk,
                )
                residual = abs(left_right - right_left)
                operation_count = left_ops + right_ops + 1
                local_scale = max(
                    left_terms,
                    right_terms,
                    contract.choice.informativeness_floor,
                )
                roundoff_bound = _termwise_roundoff(
                    contract.choice, left_terms, left_ops
                ) + _termwise_roundoff(contract.choice, right_terms, right_ops)
                bound = (
                    contract.choice.absolute_tolerance
                    + contract.choice.relative_tolerance * local_scale
                    + roundoff_bound
                )
                reciprocity_records.append(
                    Vituri2024FunctionalReplayReciprocity(
                        q_probe_index=q_index,
                        q_label=chart.q_label,
                        left_probe_index=left_index,
                        right_probe_index=right_index,
                        left_right_pairing=left_right,
                        right_left_pairing=right_left,
                        left_right_termwise_abs_sum=left_terms,
                        right_left_termwise_abs_sum=right_terms,
                        operation_count=operation_count,
                        residual=residual,
                        local_scale=local_scale,
                        roundoff_bound=roundoff_bound,
                        registered_bound=bound,
                    )
                )

    _assert_snapshot_unchanged(provider, baseline, "complete functional replay")
    call_records = tuple(recorder.records)
    transcript = _fingerprint([asdict(item) for item in call_records])
    trace_hashes = tuple(
        item.dependency_trace_sha256
        for item in call_records
        if item.dependency_trace_sha256 is not None
    )
    scalar_stability = max(item.stability_max_abs for item in scalar_gates)
    matrix_stability = max(item.stability_max_frobenius for item in matrix_gates)
    scalar_info = max(item.informativeness_abs for item in scalar_gates)
    matrix_info = max(item.informativeness_norm for item in matrix_gates)
    return Vituri2024FunctionalReplayReceipt(
        contract_fingerprint=contract.fingerprint,
        choice_fingerprint=contract.choice_fingerprint,
        verifier_implementation_schema_fingerprint=(
            contract.verifier_implementation_schema_fingerprint
        ),
        verifier_module_ast_manifest_sha256=(
            verifier_module_ast_manifest_sha256
        ),
        functional_provider_fingerprint=frozen_baseline["functional_provider_fingerprint"],
        detached_approval_manifest_sha256=contract.detached_approval_manifest_sha256,
        detached_approval_provenance=contract.detached_approval_provenance,
        approval_precedes_execution=True,
        expected_array_payload_manifest_sha256=(
            contract.expected_array_payload_manifest_sha256
        ),
        array_replay_receipt_fingerprint=array_receipt.fingerprint,
        array_replay_payload_manifest_sha256=array_receipt.hashes.payload_manifest_sha256,
        functional_payload_manifest_sha256=probe_payload.manifest_sha256,
        affine_anchor_inventory_sha256=probe_payload.affine_anchor_inventory_sha256,
        q0_probe_inventory_sha256=probe_payload.q0_probe_inventory_sha256,
        signed_q_probe_inventory_sha256=probe_payload.signed_q_probe_inventory_sha256,
        q_probe_inventory_sha256=probe_payload.q_probe_inventory_sha256,
        direct_dependency_trace_transcript_sha256=_fingerprint(trace_hashes),
        scope=FUNCTIONAL_REPLAY_SCOPE,
        affine_anchor_count=len(AFFINE_ANCHOR_LABELS),
        q0_probe_count=len(Q0_PROBE_LABELS),
        signed_q_probe_count=len(SIGNED_Q_PROBE_LABELS),
        q_chart_count=len(Q_CHART_LABELS),
        source_anchor=source_anchor,
        scalar_steps=tuple(scalar_records),
        scalar_local_gates=tuple(scalar_gates),
        matrix_steps=tuple(matrix_records),
        matrix_local_gates=tuple(matrix_gates),
        reciprocity=tuple(reciprocity_records),
        scalar_slope_stability_max_abs=scalar_stability,
        matrix_slope_stability_max_frobenius=matrix_stability,
        scalar_informativeness_max_abs=scalar_info,
        matrix_informativeness_max_frobenius=matrix_info,
        call_records=call_records,
        transcript_sha256=transcript,
        status=Vituri2024FunctionalReplayStatus(
            _factory_token=_FUNCTIONAL_REPLAY_FACTORY_TOKEN
        ),
        _factory_token=_FUNCTIONAL_REPLAY_FACTORY_TOKEN,
    )


__all__ = [
    "AFFINE_ANCHOR_LABELS",
    "DIRECT_DISPLACED_FOCK_CONSTRUCTION",
    "FUNCTIONAL_REPLAY_ABI_FINGERPRINT",
    "FUNCTIONAL_REPLAY_PAYLOAD_SCHEMA_FINGERPRINT",
    "FUNCTIONAL_REPLAY_PROVIDER_METADATA_FIELDS",
    "FUNCTIONAL_REPLAY_SCOPE",
    "FUNCTIONAL_REPLAY_VERIFIER_IMPLEMENTATION_SCHEMA_FINGERPRINT",
    "Q0_PROBE_LABELS",
    "Q_CHART_LABELS",
    "SIGNED_Q_CHART_KIND",
    "SIGNED_Q_CREATION_ANNIHILATION_CONVENTION",
    "SIGNED_Q_PROBE_LABELS",
    "SIGNED_Q_SIGN_ORDER",
    "Vituri2024DirectBuilderDependencyTrace",
    "Vituri2024DirectDisplacedFockResponse",
    "Vituri2024FunctionalReplayAnchorCheck",
    "Vituri2024FunctionalReplayApproval",
    "Vituri2024FunctionalReplayCallRecord",
    "Vituri2024FunctionalReplayChoice",
    "Vituri2024FunctionalReplayContract",
    "Vituri2024FunctionalReplayMatrixLocalGate",
    "Vituri2024FunctionalReplayMatrixStep",
    "Vituri2024FunctionalReplayPayload",
    "Vituri2024FunctionalReplayProviderProtocol",
    "Vituri2024FunctionalReplayReceipt",
    "Vituri2024FunctionalReplayReciprocity",
    "Vituri2024FunctionalReplayScalarLocalGate",
    "Vituri2024FunctionalReplayScalarStep",
    "Vituri2024FunctionalReplayStatus",
    "Vituri2024SignedQProbeChart",
    "affine_anchor_inventory_sha256",
    "direct_builder_dependency_archive_fingerprint",
    "expected_array_payload_manifest_sha256",
    "functional_provider_fingerprint",
    "functional_replay_module_ast_manifest_sha256",
    "q0_probe_inventory_sha256",
    "replay_vituri2024_half_metal_hf_functional",
    "signed_q_chart_inventory_sha256",
    "signed_q_probe_inventory_sha256",
]
