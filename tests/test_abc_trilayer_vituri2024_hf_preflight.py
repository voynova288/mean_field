"""Lightweight contract tests for the receipt-only Vituri HF preflight."""
import ast
from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

import mean_field.systems.abc_trilayer as abc
import mean_field.systems.abc_trilayer.vituri2024_hf_preflight as preflight
import mean_field.systems.abc_trilayer.vituri2024_hf_functional_replay as functional
import mean_field.systems.abc_trilayer.vituri2024_hf_replay as replay
import mean_field.systems.abc_trilayer.vituri2024_hf_scf_replay as scf_replay
from mean_field.core.hf import DensityUpdateResult, HartreeFockKernel, HartreeFockProblem

_SHA = {str(index): str(index) * 64 for index in range(10)}
_COMMIT = "a" * 40
_SOURCE_ARTIFACT = "b" * 64
_PROVIDER_FINGERPRINT = _SHA["8"]
_REPLAY_LOADER_IMPLEMENTATION_FINGERPRINT = _SHA["5"]
_FUNCTIONAL_PROBE_LOADER_IMPLEMENTATION_FINGERPRINT = _SHA["6"]
_DIRECT_DISPLACED_FOCK_IMPLEMENTATION_FINGERPRINT = _SHA["7"]
_DIRECT_INTERACTION_BUILDER_IMPLEMENTATION_FINGERPRINT = _SHA["8"]
_DIRECT_FULL_FOCK_BUILDER_IMPLEMENTATION_FINGERPRINT = _SHA["9"]
_FUNCTIONAL_G = 0.125
_FUNCTIONAL_GQ = 0.375 + 0.125j
_DIRECT_DENSITY_SENSITIVITY = 0.03125

def _canonical_replay_arrays() -> dict[str, np.ndarray]:
    mesh = np.asarray(
        [(0.01 * row, 0.01 * column) for row in range(4) for column in range(5)],
        dtype=np.float64,
    )
    energies = np.empty((4, 20), dtype=np.float64)
    for flavor in range(4):
        energies[flavor] = -0.05 + 0.001 * np.arange(20) + 0.001 * flavor
    occupations = np.ones((4, 20), dtype=np.int64)
    for flavor, k_index in ((1, 0), (3, 19)):
        occupations[flavor, k_index] = 0
        energies[flavor, k_index] = 0.01
    fock = np.zeros((4, 4, 20), dtype=np.complex128)
    projector = np.zeros_like(fock)
    diagonal = np.arange(4)
    fock[diagonal, diagonal, :] = energies
    projector[diagonal, diagonal, :] = occupations
    # The canonical synthetic functional is genuinely density dependent:
    # F(D)=h0+gD and the saved interaction is its value gP at the anchor.
    interaction_h = _FUNCTIONAL_G * projector
    h0 = fock - interaction_h
    # Register the canonical synthetic source after the same final H0+gP
    # arithmetic performed by HEAD, so exact source hashes are meaningful.
    fock = h0 + interaction_h
    energies = np.real(np.diagonal(fock, axis1=0, axis2=1).T).copy()
    active_band_states = np.zeros((2, 6, 20), dtype=np.complex128)
    for valley_index in range(2):
        for k_index in range(20):
            first = (k_index + valley_index) % 6
            second = (first + 1) % 6
            active_band_states[valley_index, first, k_index] = 0.5 + 0.5j
            active_band_states[valley_index, second, k_index] = 0.5 - 0.5j
    return {
        "mesh": mesh,
        "active_band_states": active_band_states,
        "h0": h0,
        "interaction_h": interaction_h,
        "fock": fock,
        "projector": projector,
        "energies": energies,
        "occupations": occupations,
    }


def _canonical_hashes(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        "orbitals": abc.canonical_orbital_order_sha256(arrays["mesh"]),
        "active_band_states": abc.canonical_array_sha256(
            arrays["active_band_states"]
        ),
        "h0": abc.canonical_array_sha256(arrays["h0"]),
        "interaction_h": abc.canonical_array_sha256(arrays["interaction_h"]),
        "energies": abc.canonical_array_sha256(arrays["energies"]),
        "occupations": abc.canonical_array_sha256(arrays["occupations"]),
        "projector": abc.canonical_array_sha256(arrays["projector"]),
        "fock": abc.canonical_array_sha256(arrays["fock"]),
    }

_REPLAY_ARRAYS = _canonical_replay_arrays()
_ARRAY_HASHES = _canonical_hashes(_REPLAY_ARRAYS)
_ORDERED_MESH_HASH = abc.canonical_array_sha256(_REPLAY_ARRAYS["mesh"])
_BASE_FOCK_DECOMPOSITION_RESIDUAL = float(
    np.max(
        np.abs(
            _REPLAY_ARRAYS["fock"]
            - (_REPLAY_ARRAYS["h0"] + _REPLAY_ARRAYS["interaction_h"])
        )
    )
)
_BASE_ACTIVE_BAND_STATE_NORM_RESIDUAL = float(
    np.max(
        np.abs(
            np.sum(np.abs(_REPLAY_ARRAYS["active_band_states"]) ** 2, axis=1)
            - 1.0
        )
    )
)
_BASE_AUFBAU_GAP = float(
    np.min(_REPLAY_ARRAYS["energies"][_REPLAY_ARRAYS["occupations"] == 0])
    - np.max(_REPLAY_ARRAYS["energies"][_REPLAY_ARRAYS["occupations"] == 1])
)


def _fp(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _semantically_mutated_functional_source(source: str) -> str:
    tree = ast.parse(source)
    targets = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_source_q_charts"
    ]
    assert len(targets) == 1
    targets[0].body.insert(
        0,
        ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                args=[ast.Constant(value="semantic q-validation mutation")],
                keywords=[],
            ),
            cause=None,
        ),
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _normalized_complex(array: np.ndarray) -> np.ndarray:
    norm = float(np.sqrt(np.sum(np.abs(array) ** 2)))
    assert norm > 0.0
    return np.asarray(array / norm, dtype=np.complex128)


def _q_chart(
    q_probe_index: int,
    q_label: str,
    displacement: tuple[int, int],
    cartesian_q: tuple[float, float],
) -> abc.Vituri2024SignedQProbeChart:
    rows, columns = 4, 5
    nk = rows * columns
    source = np.arange(nk, dtype=np.int64)
    targets = np.full((2, nk), -1, dtype=np.int64)
    masks = np.zeros((2, nk), dtype=np.bool_)
    for sign_index, multiplier in enumerate((1, -1)):
        for source_index in range(nk):
            row, column = divmod(source_index, columns)
            target_row = row + multiplier * displacement[0]
            target_column = column + multiplier * displacement[1]
            if 0 <= target_row < rows and 0 <= target_column < columns:
                masks[sign_index, source_index] = True
                targets[sign_index, source_index] = (
                    target_row * columns + target_column
                )
    return abc.Vituri2024SignedQProbeChart(
        q_probe_index=q_probe_index,
        q_label=q_label,
        mesh_shape=(rows, columns),
        mesh_displacement=np.asarray(displacement, dtype=np.int64),
        cartesian_q=np.asarray(cartesian_q, dtype=np.float64),
        source_k_indices=source,
        target_maps=targets,
        validity_masks=masks,
        reverse_edge_map=targets.copy(),
    )


def _functional_probe_arrays() -> tuple[
    tuple[str, ...],
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    tuple[abc.Vituri2024SignedQProbeChart, ...],
]:
    nk = 20
    q0 = np.zeros((5, 4, 4, nk), dtype=np.complex128)
    q0[0, 0, 0, 0] = 1.0
    q0[0, 1, 1, 0] = -1.0
    q0[1, 2, 2, 1] = 1.0
    q0[1, 2, 2, 2] = -1.0
    q0[2, 0, 1, 3] = q0[2, 1, 0, 3] = 1.0
    q0[3, 0, 1, 4] = 1.0j
    q0[3, 1, 0, 4] = -1.0j
    for k_index in range(nk):
        raw = np.empty((4, 4), dtype=np.complex128)
        for row in range(4):
            for column in range(4):
                raw[row, column] = complex(
                    1 + row + 2 * column + k_index,
                    row - column + 2 * k_index,
                )
        hermitian = raw + raw.conj().T
        q0[4, :, :, k_index] = hermitian
    trace_shift = np.sum(np.trace(q0[4], axis1=0, axis2=1)) / (4 * nk)
    for k_index in range(nk):
        q0[4, :, :, k_index] -= trace_shift * np.eye(4)
    for probe_index in range(q0.shape[0]):
        q0[probe_index] = _normalized_complex(q0[probe_index])

    anchors = np.zeros((3, 4, 4, nk), dtype=np.complex128)
    anchors[1] = 0.25 * q0[3]
    anchors[2] = 0.2 * q0[4]

    charts = (
        _q_chart(0, abc.Q_CHART_LABELS[0], (1, 0), (0.01, 0.0)),
        _q_chart(1, abc.Q_CHART_LABELS[1], (0, 1), (0.0, 0.01)),
    )
    signed = np.zeros((6, 2, 4, 4, nk), dtype=np.complex128)
    for q_index, chart in enumerate(charts):
        plus_boundary = np.flatnonzero(
            chart.validity_masks[0] & ~chart.validity_masks[1]
        )
        minus_boundary = np.flatnonzero(
            chart.validity_masks[1] & ~chart.validity_masks[0]
        )
        interior = np.flatnonzero(
            chart.validity_masks[0] & chart.validity_masks[1]
        )
        base = 3 * q_index
        for offset, k_index in enumerate(plus_boundary):
            signed[base, 0, 0, 1, k_index] = complex(1 + offset, 2 - offset)
        for offset, k_index in enumerate(minus_boundary):
            signed[base + 1, 1, 2, 3, k_index] = complex(2 - offset, 1 + offset)
        for offset, k_index in enumerate(interior):
            signed[base + 2, 0, 0, 2, k_index] = complex(1 + offset, -2)
            signed[base + 2, 1, 1, 3, k_index] = complex(-1, 2 + offset)
    for probe_index in range(signed.shape[0]):
        signed[probe_index] = _normalized_complex(signed[probe_index])
    q_indices = np.repeat(np.arange(2, dtype=np.int64), 3)
    return (
        abc.AFFINE_ANCHOR_LABELS,
        anchors,
        abc.Q0_PROBE_LABELS,
        q0,
        abc.SIGNED_Q_PROBE_LABELS,
        signed,
        q_indices,
        charts,
    )


(
    _AFFINE_ANCHOR_LABELS,
    _AFFINE_ANCHOR_OFFSETS,
    _Q0_LABELS,
    _Q0_DIRECTIONS,
    _SIGNED_Q_LABELS,
    _SIGNED_Q_PROBES,
    _SIGNED_Q_PROBE_INDICES,
    _SIGNED_Q_CHARTS,
) = _functional_probe_arrays()
_AFFINE_ANCHOR_INVENTORY_SHA256 = abc.affine_anchor_inventory_sha256(
    _AFFINE_ANCHOR_LABELS, _AFFINE_ANCHOR_OFFSETS
)
_Q0_PROBE_INVENTORY_SHA256 = abc.q0_probe_inventory_sha256(
    _Q0_LABELS, _Q0_DIRECTIONS
)
_SIGNED_Q_PROBE_INVENTORY_SHA256 = abc.signed_q_probe_inventory_sha256(
    _SIGNED_Q_LABELS, _SIGNED_Q_PROBES, _SIGNED_Q_PROBE_INDICES
)
_Q_CHART_INVENTORY_SHA256 = abc.signed_q_chart_inventory_sha256(_SIGNED_Q_CHARTS)
_DIRECT_BUILDER_DEPENDENCY_ARCHIVE_FINGERPRINT = (
    abc.direct_builder_dependency_archive_fingerprint(
        source_commit=_COMMIT,
        source_artifact_sha256=_SOURCE_ARTIFACT,
        direct_displaced_fock_implementation_fingerprint=(
            _DIRECT_DISPLACED_FOCK_IMPLEMENTATION_FINGERPRINT
        ),
        interaction_builder_implementation_fingerprint=(
            _DIRECT_INTERACTION_BUILDER_IMPLEMENTATION_FINGERPRINT
        ),
        full_fock_builder_implementation_fingerprint=(
            _DIRECT_FULL_FOCK_BUILDER_IMPLEMENTATION_FINGERPRINT
        ),
    )
)


def _area(
    area_angstrom_squared: float = 20_000.0,
) -> abc.Vituri2024FiniteAreaReceipt:
    return abc.Vituri2024FiniteAreaReceipt(
        area_angstrom_squared=area_angstrom_squared,
        provider_sha256=_SHA["1"],
        source_text="Synthetic finite area; no paper or production authority.",
    )


def _interaction() -> abc.Vituri2024InteractionChoiceReceipt:
    return abc.Vituri2024InteractionChoiceReceipt(
        gate_distance_angstrom=250.0,
        coulomb_e2_ev_angstrom=14.3996454784255,
        q0_evaluation="analytic_kernel_limit_only",
        provider_sha256=_SHA["2"],
        source_sha256=abc.SM_TEX_SHA256,
        authority_kind="reproduction_choice",
        source_text="Synthetic analytic interaction choice; not an HF q0 prescription.",
    )


def _geometry(**overrides: object) -> abc.Vituri2024HFGeometryReceipt:
    values: dict[str, object] = {
        "area_angstrom_squared": _area().area_angstrom_squared,
        "finite_area_receipt_fingerprint": _area().fingerprint,
        "mesh_shape": (4, 5),
        "mesh_point_count": 20,
        "core_state_nk": 20,
        "per_valley_k_count": 20,
        "valley_representation": "internal_flavor_axis",
        "spin_count": 2,
        "total_active_state_count": 80,
        "selected_spin_state_count": 40,
        "internal_flavor_order": abc.INTERNAL_FLAVOR_ORDER,
        "array_layout": abc.REPLAY_ARRAY_LAYOUT,
        "array_conversion": abc.REPLAY_ARRAY_CONVERSION,
        "ordered_momentum_mesh_sha256": _ORDERED_MESH_HASH,
        "mesh_order": "row_major_cartesian_k",
        "momentum_units": "inverse_angstrom",
        "quadrature_rule": "uniform_finite_volume_state_sum",
        "state_sum_weight": 1.0,
        "state_sum_weight_units": "dimensionless",
        "state_sum_weight_sum": 20.0,
        "state_sum_weight_sum_residual": 0.0,
        "state_sum_weight_sum_tolerance": 1.0e-12,
        "state_sum_weight_sum_evidence_sha256": _SHA["4"],
        "reciprocal_basis_sha256": _SHA["5"],
        "reciprocal_basis_convention": "columns_b1_b2_cartesian",
        "axis_origin_convention": "kx_crystal_axis_origin_at_valley_center",
        "boundary_policy": "finite_domain_no_wrap",
        "torus_policy": "not_a_reciprocal_torus",
        "reciprocal_carry_policy": "no_reciprocal_carry",
        "uv_cutoff_inverse_angstrom": 0.2,
        "delta1_mev": 28.0,
        "active_band_index": 2,
        "valleys": (-1, 1),
        "domain_minimum_third_band_direct_gap_ev": 0.01,
        "domain_gap_tolerance_ev": 1.0e-6,
        "domain_gap_point_count": 40,
        "domain_gap_evidence_sha256": _SHA["6"],
        "provider_fingerprint": _PROVIDER_FINGERPRINT,
        "source_commit": _COMMIT,
        "source_artifact_sha256": _SOURCE_ARTIFACT,
        "authority_kind": "reproduction_choice",
        "provenance": "Synthetic finite-domain geometry; not paper-direct.",
    }
    values.update(overrides)
    values.setdefault(
        "domain_gap_context_fingerprint",
        _fp(
            {
                "ordered_momentum_mesh_sha256": values["ordered_momentum_mesh_sha256"],
                "mesh_point_count": values["mesh_point_count"],
                "core_state_nk": values["core_state_nk"],
                "per_valley_k_count": values["per_valley_k_count"],
                "valley_representation": values["valley_representation"],
                "internal_flavor_order": values["internal_flavor_order"],
                "array_layout": values["array_layout"],
                "array_conversion": values["array_conversion"],
                "delta1_mev": values["delta1_mev"],
                "active_band_index": values["active_band_index"],
                "valleys": values["valleys"],
                "domain_gap_point_count": values["domain_gap_point_count"],
                "domain_minimum_third_band_direct_gap_ev": values[
                    "domain_minimum_third_band_direct_gap_ev"
                ],
                "domain_gap_tolerance_ev": values["domain_gap_tolerance_ev"],
                "domain_gap_evidence_sha256": values["domain_gap_evidence_sha256"],
                "source_commit": values["source_commit"],
                "source_artifact_sha256": values["source_artifact_sha256"],
            }
        ),
    )
    return abc.Vituri2024HFGeometryReceipt(**values)  # type: ignore[arg-type]


def _ensemble(**overrides: object) -> abc.Vituri2024HFEnsembleReceipt:
    values: dict[str, object] = {
        "ensemble": "fixed_density",
        "target_density_cm2": -1.0e12,
        "density_tolerance_cm2": 1.0e5,
        "delta1_mev": 28.0,
        "electron_hole_counting": "holes_negative_relative_to_neutral_reference",
        "normal_order_reference_kind": "provider_neutral_active_band_reference",
        "normal_order_reference_evidence_sha256": _SHA["8"],
        "q0_neutralizing_background_kind": (
            "remove_uniform_hartree_charge_against_normal_order_reference"
        ),
        "q0_background_evidence_sha256": _SHA["9"],
        "interaction_analytic_kernel_q0_policy": "analytic_kernel_limit_only",
        "interaction_receipt_fingerprint": _interaction().fingerprint,
        "chemical_potential_policy": "solve_global_mu_for_exact_fixed_state_count",
        "occupation_policy": "zero_temperature_stable_aufbau_exact_state_count",
        "branch_thermodynamic_functional": "fixed_density_canonical_energy_ev",
        "provider_fingerprint": _PROVIDER_FINGERPRINT,
        "source_commit": _COMMIT,
        "source_artifact_sha256": _SOURCE_ARTIFACT,
        "authority_kind": "reproduction_choice",
        "provenance": "Synthetic fixed-density canonical ensemble.",
    }
    values.update(overrides)
    values.setdefault(
        "normal_order_reference_fingerprint",
        _fp(
            {
                "normal_order_reference_kind": values["normal_order_reference_kind"],
                "normal_order_reference_evidence_sha256": values[
                    "normal_order_reference_evidence_sha256"
                ],
                "source_commit": values["source_commit"],
                "source_artifact_sha256": values["source_artifact_sha256"],
            }
        ),
    )
    values.setdefault(
        "q0_policy_fingerprint",
        _fp(
            {
                "q0_neutralizing_background_kind": values[
                    "q0_neutralizing_background_kind"
                ],
                "q0_background_evidence_sha256": values[
                    "q0_background_evidence_sha256"
                ],
                "interaction_analytic_kernel_q0_policy": values[
                    "interaction_analytic_kernel_q0_policy"
                ],
                "source_commit": values["source_commit"],
                "source_artifact_sha256": values["source_artifact_sha256"],
            }
        ),
    )
    return abc.Vituri2024HFEnsembleReceipt(**values)  # type: ignore[arg-type]


def _seeds() -> tuple[abc.Vituri2024SCFSeedReceipt, ...]:
    return (
        abc.Vituri2024SCFSeedReceipt("spin_plus", 101, "spin_polarized_plus"),
        abc.Vituri2024SCFSeedReceipt("spin_minus", 202, "spin_polarized_minus"),
        abc.Vituri2024SCFSeedReceipt("random_broken", 303, "random_hermitian"),
    )


def _callbacks() -> tuple[abc.Vituri2024SCFCallbackReceipt, ...]:
    callable_roles = {
        "initializer",
        "interaction_builder",
        "density_builder",
        "energy_functional",
        "oda_delta_interaction_builder",
    }
    roles = (
        "initializer",
        "interaction_builder",
        "density_builder",
        "energy_functional",
        "oda_parameterizer",
        "oda_delta_interaction_builder",
        "hamiltonian_postprocessor",
        "density_postprocessor",
        "step_callback",
        "final_state_callback",
    )
    return tuple(
        abc.Vituri2024SCFCallbackReceipt(
            role=role,  # type: ignore[arg-type]
            implementation_kind="callable" if role in callable_roles else "none",
            implementation_fingerprint=_fp(
                {"role": role, "kind": "callable" if role in callable_roles else "none"}
            ),
        )
        for role in roles
    )


def _scf(**overrides: object) -> abc.Vituri2024HFSCFPolicyReceipt:
    values: dict[str, object] = {
        "convergence_rule": "raw",
        "convergence_metric_identity": (
            "mean_field.core.hf.occupations.calculate_norm_convergence"
        ),
        "convergence_metric_normalization": (
            "frobenius_updated_minus_previous_over_frobenius_updated"
        ),
        "precision": 1.0e-9,
        "branch_energy_tolerance_ev": 2.0e-9,
        "stationarity_tolerance_ev": 1.0e-8,
        "max_iter": 500,
        "oda_stall_threshold": 1.0e-3,
        "max_oda_lambda": 1.0,
        "seed_records": _seeds(),
        "callback_receipts": _callbacks(),
        "branch_energy_comparison_policy": (
            "compare_canonical_energy_of_all_attested_converged_branches"
        ),
        "transfer_learning_policy": (
            "include_hash_bound_neighbor_sources_from_both_density_sides"
        ),
        "restart_checkpoint_policy": (
            "hash_bound_atomic_checkpoint_with_exact_policy_restart"
        ),
        "checkpoint_interval": 10,
        "uniform_weight_representation": (
            "implicit_dimensionless_unit_weight_per_finite_mesh_state"
        ),
        "final_exit_semantics": "core_exit_reason_plus_recomputed_final_raw_metric",
        "provider_fingerprint": _PROVIDER_FINGERPRINT,
        "source_commit": _COMMIT,
        "source_artifact_sha256": _SOURCE_ARTIFACT,
        "authority_kind": "reproduction_choice",
        "provenance": "Synthetic exact core SCF policy absent from the paper.",
    }
    values.update(overrides)
    return abc.Vituri2024HFSCFPolicyReceipt(**values)  # type: ignore[arg-type]


def _component(role: str, symbol: str, digest: str) -> abc.Vituri2024FunctionalComponentReceipt:
    return abc.Vituri2024FunctionalComponentReceipt(
        role=role,  # type: ignore[arg-type]
        symbol=symbol,
        implementation_fingerprint=digest,
        source_commit=_COMMIT,
        source_artifact_sha256=_SOURCE_ARTIFACT,
    )


def _ordered_orbitals_descriptor_fingerprint(
    geometry: abc.Vituri2024HFGeometryReceipt,
    hashes: dict[str, str],
) -> str:
    return _fp(
        {
            "descriptor_label": abc.ORBITAL_INDEX_DESCRIPTOR_LABEL,
            "schema_label": abc.ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL,
            "schema_fingerprint": (
                abc.ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT
            ),
            "ordered_orbitals_sha256": hashes["orbitals"],
            "ordered_momentum_mesh_sha256": (
                geometry.ordered_momentum_mesh_sha256
            ),
            "internal_flavor_order": abc.INTERNAL_FLAVOR_ORDER,
            "orbital_order": abc.REPLAY_ORBITAL_ORDER,
        }
    )

def _source_state_hash(
    geometry: abc.Vituri2024HFGeometryReceipt,
    ensemble: abc.Vituri2024HFEnsembleReceipt,
    array_hashes: dict[str, str] | None = None,
) -> str:
    hashes = _ARRAY_HASHES if array_hashes is None else array_hashes
    return _fp(
        {
            "ordered_orbitals_sha256": hashes["orbitals"],
            "ordered_orbitals_descriptor_label": (
                abc.ORBITAL_INDEX_DESCRIPTOR_LABEL
            ),
            "ordered_orbitals_schema_label": (
                abc.ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL
            ),
            "ordered_orbitals_schema_fingerprint": (
                abc.ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT
            ),
            "ordered_orbitals_descriptor_fingerprint": (
                _ordered_orbitals_descriptor_fingerprint(geometry, hashes)
            ),
            "ordered_energies_sha256": hashes["energies"],
            "ordered_occupations_sha256": hashes["occupations"],
            "ordered_projector_sha256": hashes["projector"],
            "ordered_fock_sha256": hashes["fock"],
            "h0_sha256": hashes["h0"],
            "interaction_h_sha256": hashes["interaction_h"],
            "active_band_states_sha256": hashes["active_band_states"],
            "active_band_states_layout": abc.ACTIVE_BAND_STATES_LAYOUT,
            "active_band_states_valley_order": (
                abc.ACTIVE_BAND_STATES_VALLEY_ORDER
            ),
            "active_band_states_gauge_scope": (
                abc.ACTIVE_BAND_STATES_GAUGE_SCOPE
            ),
            "replay_loader_implementation_fingerprint": (
                _REPLAY_LOADER_IMPLEMENTATION_FINGERPRINT
            ),
            "replay_payload_schema_fingerprint": (
                abc.REPLAY_PAYLOAD_SCHEMA_FINGERPRINT
            ),
            "canonical_basis_kind": abc.CANONICAL_BASIS_KIND,
            "residual_norm": abc.REPLAY_RESIDUAL_NORM,
            "fock_decomposition_convention": (
                abc.FOCK_DECOMPOSITION_CONVENTION
            ),
            "geometry_receipt_fingerprint": geometry.fingerprint,
            "ensemble_receipt_fingerprint": ensemble.fingerprint,
            "source_commit": _COMMIT,
            "source_artifact_sha256": _SOURCE_ARTIFACT,
        }
    )


def _fd(
    kind: str,
    left_implementation: str,
    right_implementation: str,
    geometry: abc.Vituri2024HFGeometryReceipt,
    ensemble: abc.Vituri2024HFEnsembleReceipt,
    *,
    array_hashes: dict[str, str] | None = None,
    **overrides: object,
) -> abc.Vituri2024FiniteDifferenceEvidenceReceipt:
    comparison_identity = (
        "scalar_energy_vs_fock_derivative"
        if kind == "fock_first_derivative"
        else "fock_derivative_vs_finite_q_hessian"
    )
    values: dict[str, object] = {
        "validation_kind": kind,
        "residual": 1.0e-10,
        "tolerance": 1.0e-8,
        "source_state_sha256": _source_state_hash(
            geometry, ensemble, array_hashes
        ),
        "geometry_receipt_fingerprint": geometry.fingerprint,
        "ensemble_receipt_fingerprint": ensemble.fingerprint,
        "perturbation_inventory_sha256": (
            _SIGNED_Q_PROBE_INVENTORY_SHA256
            if kind == "finite_q_hessian"
            else _Q0_PROBE_INVENTORY_SHA256
        ),
        "perturbation_normalization": (
            abc.FINITE_Q_HESSIAN_NORMALIZATION
            if kind == "finite_q_hessian"
            else abc.FOCK_FIRST_DERIVATIVE_NORMALIZATION
        ),
        "matrix_norm": "frobenius",
        "fock_output": abc.FOCK_OUTPUT_CONVENTION,
        "stored_density_pairing": abc.STORED_DENSITY_PAIRING,
        "density_direction_convention": (
            abc.FIXED_DENSITY_DIRECTION_CONVENTION
        ),
        "q_probe_inventory_sha256": (
            _Q_CHART_INVENTORY_SHA256
            if kind == "finite_q_hessian"
            else None
        ),
        "finite_difference_step_ladder": (1.0e-3, 3.0e-4, 1.0e-4),
        "comparison_identity": comparison_identity,
        "left_implementation_fingerprint": left_implementation,
        "right_implementation_fingerprint": right_implementation,
        "evidence_artifact_sha256": _SHA["9"],
        "source_commit": _COMMIT,
        "source_artifact_sha256": _SOURCE_ARTIFACT,
        "provenance": "Synthetic finite-difference evidence; not replayed.",
    }
    values.update(overrides)
    values.setdefault(
        "evidence_context_fingerprint",
        _fp(
            {
                "validation_kind": values["validation_kind"],
                "residual": values["residual"],
                "tolerance": values["tolerance"],
                "source_state_sha256": values["source_state_sha256"],
                "geometry_receipt_fingerprint": values[
                    "geometry_receipt_fingerprint"
                ],
                "ensemble_receipt_fingerprint": values[
                    "ensemble_receipt_fingerprint"
                ],
                "perturbation_inventory_sha256": values[
                    "perturbation_inventory_sha256"
                ],
                "perturbation_normalization": values["perturbation_normalization"],
                "matrix_norm": values["matrix_norm"],
                "fock_output": values["fock_output"],
                "stored_density_pairing": values["stored_density_pairing"],
                "density_direction_convention": values[
                    "density_direction_convention"
                ],
                "q_probe_inventory_sha256": values["q_probe_inventory_sha256"],
                "finite_difference_step_ladder": values[
                    "finite_difference_step_ladder"
                ],
                "comparison_identity": values["comparison_identity"],
                "left_implementation_fingerprint": values[
                    "left_implementation_fingerprint"
                ],
                "right_implementation_fingerprint": values[
                    "right_implementation_fingerprint"
                ],
                "evidence_artifact_sha256": values["evidence_artifact_sha256"],
                "source_commit": values["source_commit"],
                "source_artifact_sha256": values["source_artifact_sha256"],
            }
        ),
    )
    return abc.Vituri2024FiniteDifferenceEvidenceReceipt(**values)  # type: ignore[arg-type]


def _shared(
    geometry: abc.Vituri2024HFGeometryReceipt,
    ensemble: abc.Vituri2024HFEnsembleReceipt,
    *,
    array_hashes: dict[str, str] | None = None,
    q_probe_inventory_sha256: str = _Q_CHART_INVENTORY_SHA256,
    **overrides: object,
) -> abc.Vituri2024SharedFunctionalReceipt:
    scalar = _component("scalar_energy", "canonical_energy", _SHA["1"])
    fock = _component("fock_derivative", "fock", _SHA["2"])
    hessian = _component("finite_q_hessian", "hessian_q", _SHA["3"])
    interaction = _component(
        "interaction_form_factor", "interaction_and_form_factor", _SHA["4"]
    )
    values: dict[str, object] = {
        "source_commit": _COMMIT,
        "source_artifact_sha256": _SOURCE_ARTIFACT,
        "provider_fingerprint": _PROVIDER_FINGERPRINT,
        "geometry_receipt_fingerprint": geometry.fingerprint,
        "ensemble_receipt_fingerprint": ensemble.fingerprint,
        "normal_order_reference_fingerprint": ensemble.normal_order_reference_fingerprint,
        "q0_policy_fingerprint": ensemble.q0_policy_fingerprint,
        "scalar_energy": scalar,
        "fock_derivative": fock,
        "finite_q_hessian": hessian,
        "interaction_form_factor": interaction,
        "interaction_receipt_fingerprint": _interaction().fingerprint,
        "fock_finite_difference": _fd(
            "fock_first_derivative",
            scalar.implementation_fingerprint,
            fock.implementation_fingerprint,
            geometry,
            ensemble,
            array_hashes=array_hashes,
        ),
        "hessian_finite_difference": _fd(
            "finite_q_hessian",
            fock.implementation_fingerprint,
            hessian.implementation_fingerprint,
            geometry,
            ensemble,
            array_hashes=array_hashes,
            q_probe_inventory_sha256=q_probe_inventory_sha256,
        ),
        "fock_output": abc.FOCK_OUTPUT_CONVENTION,
        "stored_density_pairing": abc.STORED_DENSITY_PAIRING,
        "density_direction_convention": (
            abc.FIXED_DENSITY_DIRECTION_CONVENTION
        ),
        "authority_kind": "independent_provider_explicit",
        "provenance": "Synthetic one-source functional receipt.",
    }
    values.update(overrides)
    return abc.Vituri2024SharedFunctionalReceipt(**values)  # type: ignore[arg-type]


def _metallicity(
    geometry: abc.Vituri2024HFGeometryReceipt,
    ensemble: abc.Vituri2024HFEnsembleReceipt,
    *,
    array_hashes: dict[str, str] | None = None,
    **overrides: object,
) -> abc.Vituri2024MetallicityEvidenceReceipt:
    hashes = _ARRAY_HASHES if array_hashes is None else array_hashes
    values: dict[str, object] = {
        "source_state_sha256": _source_state_hash(geometry, ensemble, hashes),
        "geometry_receipt_fingerprint": geometry.fingerprint,
        "ordered_momentum_mesh_sha256": geometry.ordered_momentum_mesh_sha256,
        "ordered_energies_sha256": hashes["energies"],
        "ordered_occupations_sha256": hashes["occupations"],
        "selected_spin": 1,
        "chemical_potential_ev": -0.02,
        "selected_spin_band_min_ev": float(
            np.min(_REPLAY_ARRAYS["energies"][[1, 3], :])
        ),
        "selected_spin_band_max_ev": float(
            np.max(_REPLAY_ARRAYS["energies"][[1, 3], :])
        ),
        "selected_spin_occupied_state_count": 38,
        "selected_spin_unoccupied_state_count": 2,
        "metallicity_tolerance_ev": 1.0e-4,
        "evidence_sha256": _SHA["6"],
        "source_commit": _COMMIT,
        "source_artifact_sha256": _SOURCE_ARTIFACT,
    }
    values.update(overrides)
    values.setdefault(
        "context_fingerprint",
        _fp(
            {
                "source_state_sha256": values["source_state_sha256"],
                "geometry_receipt_fingerprint": values[
                    "geometry_receipt_fingerprint"
                ],
                "ordered_momentum_mesh_sha256": values[
                    "ordered_momentum_mesh_sha256"
                ],
                "ordered_energies_sha256": values["ordered_energies_sha256"],
                "ordered_occupations_sha256": values[
                    "ordered_occupations_sha256"
                ],
                "selected_spin": values["selected_spin"],
                "chemical_potential_ev": values["chemical_potential_ev"],
                "selected_spin_band_min_ev": values["selected_spin_band_min_ev"],
                "selected_spin_band_max_ev": values["selected_spin_band_max_ev"],
                "selected_spin_occupied_state_count": values[
                    "selected_spin_occupied_state_count"
                ],
                "selected_spin_unoccupied_state_count": values[
                    "selected_spin_unoccupied_state_count"
                ],
                "metallicity_tolerance_ev": values["metallicity_tolerance_ev"],
                "evidence_sha256": values["evidence_sha256"],
                "source_commit": values["source_commit"],
                "source_artifact_sha256": values["source_artifact_sha256"],
            }
        ),
    )
    return abc.Vituri2024MetallicityEvidenceReceipt(**values)  # type: ignore[arg-type]


def _pocket(
    valley: int,
    geometry: abc.Vituri2024HFGeometryReceipt,
    ensemble: abc.Vituri2024HFEnsembleReceipt,
    *,
    array_hashes: dict[str, str] | None = None,
    **overrides: object,
) -> abc.Vituri2024ValleyPocketEvidenceReceipt:
    hashes = _ARRAY_HASHES if array_hashes is None else array_hashes
    values: dict[str, object] = {
        "valley": valley,
        "source_state_sha256": _source_state_hash(geometry, ensemble, hashes),
        "geometry_receipt_fingerprint": geometry.fingerprint,
        "ordered_momentum_mesh_sha256": geometry.ordered_momentum_mesh_sha256,
        "ordered_occupations_sha256": hashes["occupations"],
        "selected_spin": 1,
        "hole_component_count": 1,
        "hole_state_count": 1,
        "adjacency_convention": "four_neighbor_finite_domain_no_wrap",
        "base_mesh_point_count": geometry.mesh_point_count,
        "component_evidence_sha256": _SHA["7"],
        "refinement_mesh_sha256": _SHA["8"],
        "refinement_point_count": 80,
        "refinement_evidence_sha256": _SHA["9"],
        "lifshitz_margin_ev": 2.0e-3,
        "lifshitz_tolerance_ev": 1.0e-4,
        "source_commit": _COMMIT,
        "source_artifact_sha256": _SOURCE_ARTIFACT,
    }
    values.update(overrides)
    values.setdefault(
        "context_fingerprint",
        _fp(
            {
                "valley": values["valley"],
                "source_state_sha256": values["source_state_sha256"],
                "geometry_receipt_fingerprint": values[
                    "geometry_receipt_fingerprint"
                ],
                "ordered_momentum_mesh_sha256": values[
                    "ordered_momentum_mesh_sha256"
                ],
                "ordered_occupations_sha256": values[
                    "ordered_occupations_sha256"
                ],
                "selected_spin": values["selected_spin"],
                "hole_component_count": values["hole_component_count"],
                "hole_state_count": values["hole_state_count"],
                "adjacency_convention": values["adjacency_convention"],
                "base_mesh_point_count": values["base_mesh_point_count"],
                "component_evidence_sha256": values[
                    "component_evidence_sha256"
                ],
                "refinement_mesh_sha256": values["refinement_mesh_sha256"],
                "refinement_point_count": values["refinement_point_count"],
                "refinement_evidence_sha256": values[
                    "refinement_evidence_sha256"
                ],
                "lifshitz_margin_ev": values["lifshitz_margin_ev"],
                "lifshitz_tolerance_ev": values["lifshitz_tolerance_ev"],
                "source_commit": values["source_commit"],
                "source_artifact_sha256": values["source_artifact_sha256"],
            }
        ),
    )
    return abc.Vituri2024ValleyPocketEvidenceReceipt(**values)  # type: ignore[arg-type]


def _branch(
    seed: abc.Vituri2024SCFSeedReceipt,
    energy: float,
    *,
    convergence_rule: str = "raw",
    **overrides: object,
) -> abc.Vituri2024BranchEnergyReceipt:
    values: dict[str, object] = {
        "seed": seed,
        "attested_exit_reason": "converged",
        "iterations": 12,
        "terminal_norm_raw": 2.0e-11,
        "terminal_norm_mixed": 1.0e-11,
        "terminal_norm_selected": (
            2.0e-11 if convergence_rule == "raw" else 1.0e-11
        ),
        "terminal_oda_lambda": 0.4,
        "final_replay_raw_metric": 1.0e-11,
        "canonical_energy_ev": energy,
    }
    values.update(overrides)
    return abc.Vituri2024BranchEnergyReceipt(**values)  # type: ignore[arg-type]


def _branches(
    convergence_rule: str = "raw",
) -> tuple[abc.Vituri2024BranchEnergyReceipt, ...]:
    seeds = _seeds()
    return tuple(
        _branch(seed, energy, convergence_rule=convergence_rule)
        for seed, energy in zip(seeds, (-2.0, -1.9, -1.8))
    )


def _branch_table_bytes(
    records: tuple[abc.Vituri2024BranchEnergyReceipt, ...],
) -> bytes:
    """Independent canonical source-table serialization used by fixtures."""

    return json.dumps(
        [asdict(item) for item in records],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

def _source(
    geometry: abc.Vituri2024HFGeometryReceipt,
    ensemble: abc.Vituri2024HFEnsembleReceipt,
    scf: abc.Vituri2024HFSCFPolicyReceipt,
    shared: abc.Vituri2024SharedFunctionalReceipt,
    *,
    array_hashes: dict[str, str] | None = None,
    **overrides: object,
) -> abc.Vituri2024AttestedHalfMetalSourceReceipt:
    hashes = _ARRAY_HASHES if array_hashes is None else array_hashes
    records = _branches(scf.convergence_rule)
    values: dict[str, object] = {
        "source_commit": _COMMIT,
        "source_artifact_sha256": _SOURCE_ARTIFACT,
        "provider_fingerprint": _PROVIDER_FINGERPRINT,
        "source_state_sha256": _source_state_hash(geometry, ensemble, hashes),
        "ordered_orbitals_sha256": hashes["orbitals"],
        "ordered_orbitals_descriptor_label": (
            abc.ORBITAL_INDEX_DESCRIPTOR_LABEL
        ),
        "ordered_orbitals_schema_label": (
            abc.ORBITAL_INDEX_DESCRIPTOR_SCHEMA_LABEL
        ),
        "ordered_orbitals_schema_fingerprint": (
            abc.ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT
        ),
        "ordered_orbitals_descriptor_fingerprint": (
            _ordered_orbitals_descriptor_fingerprint(geometry, hashes)
        ),
        "ordered_energies_sha256": hashes["energies"],
        "ordered_occupations_sha256": hashes["occupations"],
        "ordered_projector_sha256": hashes["projector"],
        "ordered_fock_sha256": hashes["fock"],
        "h0_sha256": hashes["h0"],
        "interaction_h_sha256": hashes["interaction_h"],
        "active_band_states_sha256": hashes["active_band_states"],
        "active_band_states_layout": abc.ACTIVE_BAND_STATES_LAYOUT,
        "active_band_states_valley_order": (
            abc.ACTIVE_BAND_STATES_VALLEY_ORDER
        ),
        "active_band_states_gauge_scope": (
            abc.ACTIVE_BAND_STATES_GAUGE_SCOPE
        ),
        "replay_loader_implementation_fingerprint": (
            _REPLAY_LOADER_IMPLEMENTATION_FINGERPRINT
        ),
        "replay_payload_schema_fingerprint": (
            abc.REPLAY_PAYLOAD_SCHEMA_FINGERPRINT
        ),
        "geometry_receipt_fingerprint": geometry.fingerprint,
        "ensemble_receipt_fingerprint": ensemble.fingerprint,
        "scf_policy_receipt_fingerprint": scf.fingerprint,
        "shared_functional_receipt_fingerprint": shared.fingerprint,
        "area_angstrom_squared": geometry.area_angstrom_squared,
        "finite_area_receipt_fingerprint": geometry.finite_area_receipt_fingerprint,
        "ordered_momentum_mesh_sha256": geometry.ordered_momentum_mesh_sha256,
        "target_density_cm2": -1.0e12,
        "measured_density_cm2": -1.0e12,
        "density_residual_cm2": 0.0,
        "density_tolerance_cm2": 1.0e5,
        "chemical_potential_ev": -0.02,
        "attested_exit_reason": "converged",
        "final_replay_raw_metric": 1.0e-11,
        "final_replay_raw_precision": 1.0e-9,
        "canonical_basis_kind": abc.CANONICAL_BASIS_KIND,
        "residual_norm": abc.REPLAY_RESIDUAL_NORM,
        "fock_decomposition_convention": (
            abc.FOCK_DECOMPOSITION_CONVENTION
        ),
        "fock_decomposition_residual_ev": (
            _BASE_FOCK_DECOMPOSITION_RESIDUAL
        ),
        "fock_decomposition_tolerance_ev": 1.0e-12,
        "h0_hermiticity_residual_ev": 0.0,
        "h0_hermiticity_tolerance_ev": 1.0e-12,
        "interaction_h_hermiticity_residual_ev": 0.0,
        "interaction_h_hermiticity_tolerance_ev": 1.0e-12,
        "active_band_state_norm_residual": (
            _BASE_ACTIVE_BAND_STATE_NORM_RESIDUAL
        ),
        "active_band_state_norm_tolerance": 1.0e-12,
        "fock_projector_commutator_residual_ev": 0.0,
        "stationarity_tolerance_ev": 1.0e-8,
        "projector_idempotency_residual": 0.0,
        "projector_idempotency_tolerance": 1.0e-8,
        "projector_hermiticity_residual": 0.0,
        "projector_hermiticity_tolerance": 1.0e-9,
        "fock_hermiticity_residual_ev": 0.0,
        "fock_hermiticity_tolerance_ev": 1.0e-9,
        "projector_vs_occupation_residual": 0.0,
        "projector_vs_occupation_tolerance": 1.0e-12,
        "fock_vs_diagonal_energy_residual_ev": 0.0,
        "fock_vs_diagonal_energy_tolerance_ev": 1.0e-12,
        "aufbau_min_unoccupied_minus_max_occupied_ev": _BASE_AUFBAU_GAP,
        "aufbau_occupation_violation_ev": 0.0,
        "aufbau_tolerance_ev": 1.0e-9,
        "chemical_mu_occupation_residual_ev": 0.0,
        "chemical_mu_occupation_tolerance_ev": 1.0e-9,
        "selected_spin": 1,
        "valley_plus_hole_count": 1,
        "valley_minus_hole_count": 1,
        "selected_spin_hole_count": 2,
        "opposite_spin_hole_count": 0,
        "metallicity_evidence": _metallicity(
            geometry, ensemble, array_hashes=hashes
        ),
        "pocket_evidence": (
            _pocket(-1, geometry, ensemble, array_hashes=hashes),
            _pocket(1, geometry, ensemble, array_hashes=hashes),
        ),
        "branch_comparison_evidence_sha256": _SHA["6"],
        "branch_energy_table_sha256": hashlib.sha256(
            _branch_table_bytes(records)
        ).hexdigest(),
        "branch_records": records,
        "branch_energy_functional_fingerprint": shared.scalar_energy.fingerprint,
        "selected_branch_label": "spin_plus",
        "selected_branch_energy_ev": -2.0,
        "minimum_compared_branch_energy_ev": -2.0,
        "branch_energy_residual_ev": 0.0,
        "branch_energy_tolerance_ev": 2.0e-9,
        "provenance": "Synthetic receipt-level source; arrays are not loaded.",
    }
    values.update(overrides)
    values.setdefault(
        "branch_table_context_fingerprint",
        _fp(
            {
                "branch_energy_table_sha256": values["branch_energy_table_sha256"],
                "branch_comparison_evidence_sha256": values[
                    "branch_comparison_evidence_sha256"
                ],
                "branch_records": [asdict(item) for item in values["branch_records"]],
                "branch_energy_functional_fingerprint": values[
                    "branch_energy_functional_fingerprint"
                ],
                "source_state_sha256": values["source_state_sha256"],
                "source_commit": values["source_commit"],
                "source_artifact_sha256": values["source_artifact_sha256"],
            }
        ),
    )
    return abc.Vituri2024AttestedHalfMetalSourceReceipt(**values)  # type: ignore[arg-type]


def _complete() -> abc.Vituri2024HalfMetalHFSpec:
    geometry, ensemble, scf = _geometry(), _ensemble(), _scf()
    shared = _shared(geometry, ensemble)
    source = _source(geometry, ensemble, scf, shared)
    return abc.Vituri2024HalfMetalHFSpec(
        geometry=geometry,
        ensemble=ensemble,
        scf_policy=scf,
        shared_functional=shared,
        attested_source=source,
    )


def _array_copy(**updates: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {name: value.copy() for name, value in _REPLAY_ARRAYS.items()}
    arrays.update(updates)
    return arrays


def _spec_for_arrays(
    arrays: dict[str, np.ndarray],
    *,
    geometry_overrides: dict[str, object] | None = None,
    source_overrides: dict[str, object] | None = None,
    q_probe_inventory_sha256: str = _Q_CHART_INVENTORY_SHA256,
) -> abc.Vituri2024HalfMetalHFSpec:
    hashes = _canonical_hashes(arrays)
    geometry_values: dict[str, object] = {
        "ordered_momentum_mesh_sha256": abc.canonical_array_sha256(arrays["mesh"])
    }
    if geometry_overrides:
        geometry_values.update(geometry_overrides)
    geometry = _geometry(**geometry_values)
    ensemble, scf = _ensemble(), _scf()
    shared = _shared(
        geometry,
        ensemble,
        array_hashes=hashes,
        q_probe_inventory_sha256=q_probe_inventory_sha256,
    )
    source = _source(
        geometry,
        ensemble,
        scf,
        shared,
        array_hashes=hashes,
        **({} if source_overrides is None else source_overrides),
    )
    return abc.Vituri2024HalfMetalHFSpec(
        geometry=geometry,
        ensemble=ensemble,
        scf_policy=scf,
        shared_functional=shared,
        attested_source=source,
    )


def _payload(
    spec: abc.Vituri2024HalfMetalHFSpec,
    arrays: dict[str, np.ndarray] | None = None,
    **identity_overrides: object,
) -> abc.Vituri2024HalfMetalHFReplayPayload:
    assert spec.attested_source
    source = spec.attested_source
    values: dict[str, object] = {
        "provider_fingerprint": source.provider_fingerprint,
        "source_commit": source.source_commit,
        "source_artifact_sha256": source.source_artifact_sha256,
        "spec_fingerprint": spec.fingerprint,
        "source_state_sha256": source.source_state_sha256,
        "replay_loader_implementation_fingerprint": (
            source.replay_loader_implementation_fingerprint
        ),
        "replay_payload_schema_fingerprint": (
            source.replay_payload_schema_fingerprint
        ),
        **(_REPLAY_ARRAYS if arrays is None else arrays),
    }
    values.update(identity_overrides)
    return abc.Vituri2024HalfMetalHFReplayPayload(**values)  # type: ignore[arg-type]


class _Provider:
    def __init__(
        self,
        spec: abc.Vituri2024HalfMetalHFSpec,
        payload: abc.Vituri2024HalfMetalHFReplayPayload | None = None,
        functional_payload: abc.Vituri2024FunctionalReplayPayload | None = None,
    ) -> None:
        assert spec.geometry and spec.ensemble and spec.scf_policy
        assert spec.shared_functional and spec.attested_source
        shared = spec.shared_functional
        self.provider_fingerprint = _PROVIDER_FINGERPRINT
        self.source_commit = shared.source_commit
        self.source_artifact_sha256 = shared.source_artifact_sha256
        self.spec_fingerprint = spec.fingerprint
        self.geometry_receipt_fingerprint = spec.geometry.fingerprint
        self.ensemble_receipt_fingerprint = spec.ensemble.fingerprint
        self.scf_policy_receipt_fingerprint = spec.scf_policy.fingerprint
        self.shared_functional_receipt_fingerprint = shared.fingerprint
        self.attested_source_receipt_fingerprint = spec.attested_source.fingerprint
        self.finite_area_receipt_fingerprint = spec.geometry.finite_area_receipt_fingerprint
        self.interaction_receipt_fingerprint = shared.interaction_receipt_fingerprint
        self.normal_order_reference_fingerprint = (
            spec.ensemble.normal_order_reference_fingerprint
        )
        self.q0_policy_fingerprint = spec.ensemble.q0_policy_fingerprint
        self.source_state_sha256 = spec.attested_source.source_state_sha256
        self.replay_loader_implementation_fingerprint = (
            spec.attested_source.replay_loader_implementation_fingerprint
        )
        self.replay_payload_schema_fingerprint = (
            spec.attested_source.replay_payload_schema_fingerprint
        )
        self.scalar_energy_implementation_fingerprint = (
            shared.scalar_energy.implementation_fingerprint
        )
        self.fock_derivative_implementation_fingerprint = (
            shared.fock_derivative.implementation_fingerprint
        )
        self.finite_q_hessian_implementation_fingerprint = (
            shared.finite_q_hessian.implementation_fingerprint
        )
        self.interaction_form_factor_implementation_fingerprint = (
            shared.interaction_form_factor.implementation_fingerprint
        )
        self.functional_probe_loader_implementation_fingerprint = (
            _FUNCTIONAL_PROBE_LOADER_IMPLEMENTATION_FINGERPRINT
        )
        self.functional_replay_payload_schema_fingerprint = (
            abc.FUNCTIONAL_REPLAY_PAYLOAD_SCHEMA_FINGERPRINT
        )
        self.functional_replay_abi_fingerprint = (
            abc.FUNCTIONAL_REPLAY_ABI_FINGERPRINT
        )
        self.direct_displaced_fock_implementation_fingerprint = (
            _DIRECT_DISPLACED_FOCK_IMPLEMENTATION_FINGERPRINT
        )
        self.direct_interaction_builder_implementation_fingerprint = (
            _DIRECT_INTERACTION_BUILDER_IMPLEMENTATION_FINGERPRINT
        )
        self.direct_full_fock_builder_implementation_fingerprint = (
            _DIRECT_FULL_FOCK_BUILDER_IMPLEMENTATION_FINGERPRINT
        )
        self.direct_builder_dependency_archive_fingerprint = (
            _DIRECT_BUILDER_DEPENDENCY_ARCHIVE_FINGERPRINT
        )
        self.functional_provider_fingerprint = abc.functional_provider_fingerprint(
            base_provider_fingerprint=self.provider_fingerprint,
            functional_replay_abi_fingerprint=(
                self.functional_replay_abi_fingerprint
            ),
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
        self.direct_displaced_fock_construction = (
            abc.DIRECT_DISPLACED_FOCK_CONSTRUCTION
        )
        self.fock_output = abc.FOCK_OUTPUT_CONVENTION
        self.stored_density_pairing = abc.STORED_DENSITY_PAIRING
        self.density_direction_convention = (
            abc.FIXED_DENSITY_DIRECTION_CONVENTION
        )
        self.replay_payload = payload
        self.functional_payload = functional_payload
        self.loader_mutation: tuple[str, object] | None = None
        self.functional_loader_mutation: tuple[str, object] | None = None
        self.call_mutation: tuple[str, object] | None = None
        self.mutate_input_method: str | None = None
        self.replay_loader_calls: list[str] = []
        self.functional_loader_calls: list[str] = []
        self.scalar_energy_calls = 0
        self.fock_derivative_calls = 0
        self.direct_displaced_calls = 0
        self.hessian_calls = 0
        if payload is not None:
            nk = payload.projector.shape[2]
            raw_anchor = float(
                np.real(
                    np.sum(payload.h0 * payload.projector)
                    + 0.5 * np.sum(
                        (_FUNCTIONAL_G * payload.projector) * payload.projector
                    )
                )
                / nk
            )
            self.energy_offset = (
                spec.attested_source.selected_branch_energy_ev - raw_anchor
            )
        else:
            self.energy_offset = 0.0

    def _maybe_mutate_metadata(self) -> None:
        if self.call_mutation is not None:
            setattr(self, self.call_mutation[0], self.call_mutation[1])

    def evaluate_scalar_energy(
        self,
        interaction_h: preflight.ComplexArray,
        h0: preflight.ComplexArray,
        density: preflight.ComplexArray,
    ) -> float:
        if self.mutate_input_method == "evaluate_scalar_energy":
            density[0, 0, 0] += 1.0
        self._maybe_mutate_metadata()
        self.scalar_energy_calls += 1
        nk = density.shape[2]
        # Independent scalar implementation.  On the proper affine call path
        # interaction_h=gD, this is offset+[sum(h0 D)+g/2 sum(D D)]/Nk.
        return float(
            self.energy_offset
            + np.real(
                np.sum(h0 * density)
                + 0.5 * np.sum(interaction_h * density)
            )
            / nk
        )

    def evaluate_fock_derivative(
        self, density: preflight.ComplexArray
    ) -> preflight.ComplexArray:
        if self.replay_payload is None:
            raise AssertionError("Fock evaluation payload was not configured")
        if self.mutate_input_method == "evaluate_fock_derivative":
            density[0, 0, 0] += 1.0
        self._maybe_mutate_metadata()
        self.fock_derivative_calls += 1
        # Kept separate from evaluate_scalar_energy by construction.
        return np.asarray(
            self.replay_payload.h0 + _FUNCTIONAL_G * density,
            dtype=np.complex128,
        )

    def _validate_q_call(
        self,
        q_probe_index: int,
        mesh_displacement: np.ndarray,
        cartesian_q: np.ndarray,
        target_maps: np.ndarray,
        reverse_edge_map: np.ndarray,
    ) -> abc.Vituri2024SignedQProbeChart:
        if self.functional_payload is None:
            raise AssertionError("functional probe payload was not configured")
        if q_probe_index not in range(len(self.functional_payload.q_charts)):
            raise AssertionError("finite-q probe index was not configured")
        chart = self.functional_payload.q_charts[q_probe_index]
        for actual, required, label in (
            (mesh_displacement, chart.mesh_displacement, "mesh displacement"),
            (cartesian_q, chart.cartesian_q, "Cartesian q"),
            (target_maps, chart.target_maps, "target maps"),
            (reverse_edge_map, chart.reverse_edge_map, "reverse map"),
        ):
            if not np.array_equal(actual, required):
                raise ValueError(f"synthetic q call {label} mismatch")
        return chart

    def evaluate_finite_q_hessian(
        self,
        perturbation: preflight.ComplexArray,
        *,
        q_probe_index: int,
        mesh_displacement: np.ndarray,
        cartesian_q: np.ndarray,
        target_maps: np.ndarray,
        reverse_edge_map: np.ndarray,
    ) -> preflight.ComplexArray:
        chart = self._validate_q_call(
            q_probe_index,
            mesh_displacement,
            cartesian_q,
            target_maps,
            reverse_edge_map,
        )
        if self.mutate_input_method == "evaluate_finite_q_hessian":
            perturbation[0, 0, 0, 0] += 1.0
        self._maybe_mutate_metadata()
        self.hessian_calls += 1
        # Independent analytic Hessian implementation: explicit cross-sign
        # routing from the supplied maps, without any direct-builder call.
        result = np.zeros_like(perturbation)
        for sign_index in range(2):
            opposite = 1 - sign_index
            for source_index in range(target_maps.shape[1]):
                target = int(target_maps[sign_index, source_index])
                reverse = int(reverse_edge_map[sign_index, source_index])
                if target >= 0:
                    if reverse != target:
                        raise ValueError("analytic Hessian reverse routing mismatch")
                    result[sign_index, :, :, source_index] = (
                        _FUNCTIONAL_GQ
                        * perturbation[opposite, :, :, target]
                    )
        for sign_index in range(2):
            result[sign_index, :, :, ~chart.validity_masks[sign_index]] = 0.0
        return result

    def evaluate_displaced_fock(
        self,
        density: preflight.ComplexArray,
        signed_q_displacement: preflight.ComplexArray,
        *,
        q_probe_index: int,
        mesh_displacement: np.ndarray,
        cartesian_q: np.ndarray,
        target_maps: np.ndarray,
        reverse_edge_map: np.ndarray,
        caller_nonce: str,
    ) -> abc.Vituri2024DirectDisplacedFockResponse:
        chart = self._validate_q_call(
            q_probe_index,
            mesh_displacement,
            cartesian_q,
            target_maps,
            reverse_edge_map,
        )
        if self.replay_payload is None:
            raise AssertionError("direct displaced-Fock source was not configured")
        if self.mutate_input_method == "evaluate_displaced_fock":
            signed_q_displacement[0, 0, 0, 0] += 1.0
        self._maybe_mutate_metadata()
        self.direct_displaced_calls += 1

        # Source-closed direct interaction/full-Fock builders.  The base
        # density is materially consumed; changing it changes the direct
        # response.  This path never calls evaluate_finite_q_hessian.
        interaction_builder_calls = 1
        base_interaction = _FUNCTIONAL_G * density
        full_fock_builder_calls = 1
        full_fock = self.replay_payload.h0 + base_interaction
        reference_full_fock = (
            self.replay_payload.h0
            + _FUNCTIONAL_G * self.replay_payload.projector
        )
        density_marker = float(
            np.real(full_fock[0, 0, 0] - reference_full_fock[0, 0, 0])
            / _FUNCTIONAL_G
        )
        coefficient = _FUNCTIONAL_GQ + _DIRECT_DENSITY_SENSITIVITY * density_marker
        result = np.zeros_like(signed_q_displacement)
        target_reads = 0
        reverse_reads = 0
        # Deliberately algorithmically separate from the scalar-loop analytic
        # Hessian above: each direct lane gathers its routed opposite-sign
        # source in one vectorized operation and never calls the Hessian.
        for sign_index in range(2):
            opposite = 1 - sign_index
            target_lane = target_maps[sign_index]
            reverse_lane = reverse_edge_map[sign_index]
            target_reads += target_lane.size
            reverse_reads += reverse_lane.size
            valid = target_lane >= 0
            if not np.array_equal(target_lane[valid], reverse_lane[valid]):
                raise ValueError("direct builder reverse routing mismatch")
            routes = target_lane[valid]
            result[sign_index][:, :, valid] = coefficient * np.take(
                signed_q_displacement[opposite], routes, axis=2
            )
        for sign_index in range(2):
            result[sign_index, :, :, ~chart.validity_masks[sign_index]] = 0.0
        trace = abc.Vituri2024DirectBuilderDependencyTrace(
            caller_nonce_sha256=caller_nonce,
            q_probe_index=q_probe_index,
            q_label=chart.q_label,
            interaction_builder_implementation_fingerprint=(
                self.direct_interaction_builder_implementation_fingerprint
            ),
            full_fock_builder_implementation_fingerprint=(
                self.direct_full_fock_builder_implementation_fingerprint
            ),
            target_maps_sha256=abc.canonical_array_sha256(target_maps),
            reverse_edge_map_sha256=abc.canonical_array_sha256(reverse_edge_map),
            target_map_read_count=target_reads,
            reverse_edge_map_read_count=reverse_reads,
            interaction_builder_call_count=interaction_builder_calls,
            full_fock_builder_call_count=full_fock_builder_calls,
            finite_q_hessian_call_count=0,
        )
        return abc.Vituri2024DirectDisplacedFockResponse(
            caller_nonce_sha256=caller_nonce,
            response=result,
            dependency_trace=trace,
        )

    def load_attested_source_arrays(
        self, source_artifact_sha256: str
    ) -> abc.Vituri2024AttestedHalfMetalSourceArrays:
        raise AssertionError("metadata attestation must not execute providers")

    def load_half_metal_replay_payload(
        self, source_artifact_sha256: str
    ) -> abc.Vituri2024HalfMetalHFReplayPayload:
        self.replay_loader_calls.append(source_artifact_sha256)
        if self.replay_payload is None:
            raise AssertionError("array replay payload was not configured")
        if self.loader_mutation is not None:
            setattr(self, self.loader_mutation[0], self.loader_mutation[1])
        return self.replay_payload

    def load_functional_replay_payload(
        self, source_artifact_sha256: str
    ) -> abc.Vituri2024FunctionalReplayPayload:
        self.functional_loader_calls.append(source_artifact_sha256)
        if self.functional_payload is None:
            raise AssertionError("functional replay payload was not configured")
        if self.functional_loader_mutation is not None:
            setattr(
                self,
                self.functional_loader_mutation[0],
                self.functional_loader_mutation[1],
            )
        return self.functional_payload


class _BaseBindingSnapshotSubstitutionProvider(_Provider):
    @staticmethod
    def snapshot_base_values(
        spec: abc.Vituri2024HalfMetalHFSpec,
    ) -> dict[str, str]:
        assert spec.attested_source is not None
        values = {
            name: ("c" * 40 if name == "source_commit" else _SHA["9"])
            for name in preflight.VITURI2024_BASE_PROVIDER_METADATA_FIELDS
        }
        values["source_state_sha256"] = spec.attested_source.source_state_sha256
        values["replay_loader_implementation_fingerprint"] = (
            spec.attested_source.replay_loader_implementation_fingerprint
        )
        values["replay_payload_schema_fingerprint"] = (
            spec.attested_source.replay_payload_schema_fingerprint
        )
        return values

    def __init__(
        self,
        spec: abc.Vituri2024HalfMetalHFSpec,
        payload: abc.Vituri2024HalfMetalHFReplayPayload,
    ) -> None:
        self._snapshot_base_values = type(self).snapshot_base_values(spec)
        self.snapshot_reads: list[str] = []
        super().__init__(spec, payload)

    def __getattribute__(self, name: str) -> object:
        snapshot_values = object.__getattribute__(self, "_snapshot_base_values")
        if name in snapshot_values:
            frame = inspect.currentframe()
            try:
                while frame is not None:
                    if (
                        frame.f_code.co_name == "_provider_metadata_snapshot"
                        and frame.f_globals.get("__name__") == replay.__name__
                    ):
                        object.__getattribute__(self, "snapshot_reads").append(name)
                        return snapshot_values[name]
                    frame = frame.f_back
            finally:
                del frame
        return super().__getattribute__(name)


class _PostValidationAccessDriftProvider(_Provider):
    _LATE_IDENTITY_VALUES = {
        "provider_fingerprint": _SHA["9"],
        "source_commit": "c" * 40,
        "source_artifact_sha256": _SHA["7"],
        "spec_fingerprint": _SHA["6"],
    }

    def __init__(
        self,
        spec: abc.Vituri2024HalfMetalHFSpec,
        payload: abc.Vituri2024HalfMetalHFReplayPayload,
    ) -> None:
        self._post_validation_drift_armed = False
        self.relevant_reads_after_validation: list[str] = []
        super().__init__(spec, payload)

    def __getattribute__(self, name: str) -> object:
        late_values = type(self)._LATE_IDENTITY_VALUES
        if name in late_values and object.__getattribute__(
            self, "_post_validation_drift_armed"
        ):
            object.__getattribute__(
                self, "relevant_reads_after_validation"
            ).append(name)
            return late_values[name]
        return super().__getattribute__(name)

    def arm_post_validation_drift(self) -> None:
        self._post_validation_drift_armed = True


def _replay_case(
    arrays: dict[str, np.ndarray] | None = None,
    *,
    geometry_overrides: dict[str, object] | None = None,
    source_overrides: dict[str, object] | None = None,
) -> tuple[
    abc.Vituri2024HalfMetalHFProviderBinding,
    abc.Vituri2024HalfMetalHFReplayPayload,
    _Provider,
]:
    actual_arrays = _array_copy() if arrays is None else arrays
    spec = _spec_for_arrays(
        actual_arrays,
        geometry_overrides=geometry_overrides,
        source_overrides=source_overrides,
    )
    payload = _payload(spec, actual_arrays)
    provider = _Provider(spec, payload)
    return abc.Vituri2024HalfMetalHFProviderBinding(spec, provider), payload, provider


def _functional_payload(
    spec: abc.Vituri2024HalfMetalHFSpec,
    q_charts: tuple[abc.Vituri2024SignedQProbeChart, ...] = _SIGNED_Q_CHARTS,
) -> abc.Vituri2024FunctionalReplayPayload:
    functional_provider = abc.functional_provider_fingerprint(
        base_provider_fingerprint=_PROVIDER_FINGERPRINT,
        functional_replay_abi_fingerprint=abc.FUNCTIONAL_REPLAY_ABI_FINGERPRINT,
        functional_replay_payload_schema_fingerprint=(
            abc.FUNCTIONAL_REPLAY_PAYLOAD_SCHEMA_FINGERPRINT
        ),
        functional_probe_loader_implementation_fingerprint=(
            _FUNCTIONAL_PROBE_LOADER_IMPLEMENTATION_FINGERPRINT
        ),
        direct_displaced_fock_implementation_fingerprint=(
            _DIRECT_DISPLACED_FOCK_IMPLEMENTATION_FINGERPRINT
        ),
        direct_builder_dependency_archive_fingerprint=(
            _DIRECT_BUILDER_DEPENDENCY_ARCHIVE_FINGERPRINT
        ),
    )
    assert spec.attested_source is not None
    return abc.Vituri2024FunctionalReplayPayload(
        provider_fingerprint=_PROVIDER_FINGERPRINT,
        functional_provider_fingerprint=functional_provider,
        source_commit=_COMMIT,
        source_artifact_sha256=_SOURCE_ARTIFACT,
        spec_fingerprint=spec.fingerprint,
        source_state_sha256=spec.attested_source.source_state_sha256,
        functional_probe_loader_implementation_fingerprint=(
            _FUNCTIONAL_PROBE_LOADER_IMPLEMENTATION_FINGERPRINT
        ),
        functional_replay_payload_schema_fingerprint=(
            abc.FUNCTIONAL_REPLAY_PAYLOAD_SCHEMA_FINGERPRINT
        ),
        affine_anchor_labels=_AFFINE_ANCHOR_LABELS,
        affine_anchor_offsets=_AFFINE_ANCHOR_OFFSETS,
        q0_labels=_Q0_LABELS,
        q0_directions=_Q0_DIRECTIONS,
        signed_q_labels=_SIGNED_Q_LABELS,
        signed_q_probes=_SIGNED_Q_PROBES,
        signed_q_probe_indices=_SIGNED_Q_PROBE_INDICES,
        q_charts=q_charts,
    )


def _functional_case(
    *,
    q_charts: tuple[abc.Vituri2024SignedQProbeChart, ...] = _SIGNED_Q_CHARTS,
    geometry_overrides: dict[str, object] | None = None,
) -> tuple[
    abc.Vituri2024HalfMetalHFProviderBinding,
    abc.Vituri2024FunctionalReplayContract,
    _Provider,
]:
    arrays = _array_copy()
    q_inventory = abc.signed_q_chart_inventory_sha256(q_charts)
    spec = _spec_for_arrays(
        arrays,
        geometry_overrides=geometry_overrides,
        q_probe_inventory_sha256=q_inventory,
    )
    array_payload = _payload(spec, arrays)
    probes = _functional_payload(spec, q_charts)
    provider = _Provider(spec, array_payload, probes)
    binding = abc.Vituri2024HalfMetalHFProviderBinding(spec, provider)
    assert spec.geometry and spec.ensemble and spec.shared_functional
    assert spec.attested_source
    expected_manifest = abc.expected_array_payload_manifest_sha256(spec)
    choice = abc.Vituri2024FunctionalReplayChoice()
    verifier_ast_manifest = abc.functional_replay_module_ast_manifest_sha256()
    approval = abc.Vituri2024FunctionalReplayApproval(
        choice_fingerprint=choice.fingerprint,
        verifier_implementation_schema_fingerprint=(
            abc.FUNCTIONAL_REPLAY_VERIFIER_IMPLEMENTATION_SCHEMA_FINGERPRINT
        ),
        verifier_module_ast_manifest_sha256=verifier_ast_manifest,
        functional_provider_fingerprint=provider.functional_provider_fingerprint,
        source_commit=provider.source_commit,
        source_artifact_sha256=provider.source_artifact_sha256,
        spec_fingerprint=spec.fingerprint,
        source_state_sha256=spec.attested_source.source_state_sha256,
        expected_array_payload_manifest_sha256=expected_manifest,
        affine_anchor_inventory_sha256=_AFFINE_ANCHOR_INVENTORY_SHA256,
        q0_probe_inventory_sha256=_Q0_PROBE_INVENTORY_SHA256,
        signed_q_probe_inventory_sha256=_SIGNED_Q_PROBE_INVENTORY_SHA256,
        q_probe_inventory_sha256=q_inventory,
        fock_step_ladder=(
            spec.shared_functional.fock_finite_difference.finite_difference_step_ladder
        ),
        hessian_step_ladder=(
            spec.shared_functional.hessian_finite_difference.finite_difference_step_ladder
        ),
        direct_displaced_fock_implementation_fingerprint=(
            provider.direct_displaced_fock_implementation_fingerprint
        ),
        direct_builder_dependency_archive_fingerprint=(
            provider.direct_builder_dependency_archive_fingerprint
        ),
        provenance="Detached synthetic approval constructed before provider calls.",
    )
    assert provider.replay_loader_calls == []
    assert provider.functional_loader_calls == []
    assert provider.scalar_energy_calls == 0
    assert provider.fock_derivative_calls == 0
    assert provider.hessian_calls == 0
    assert provider.direct_displaced_calls == 0
    contract = abc.Vituri2024FunctionalReplayContract(
        choice=choice,
        choice_fingerprint=choice.fingerprint,
        verifier_implementation_schema_fingerprint=(
            abc.FUNCTIONAL_REPLAY_VERIFIER_IMPLEMENTATION_SCHEMA_FINGERPRINT
        ),
        verifier_module_ast_manifest_sha256=(
            approval.verifier_module_ast_manifest_sha256
        ),
        provider_fingerprint=provider.provider_fingerprint,
        functional_provider_fingerprint=provider.functional_provider_fingerprint,
        source_commit=provider.source_commit,
        source_artifact_sha256=provider.source_artifact_sha256,
        spec_fingerprint=spec.fingerprint,
        source_state_sha256=spec.attested_source.source_state_sha256,
        geometry_receipt_fingerprint=spec.geometry.fingerprint,
        ensemble_receipt_fingerprint=spec.ensemble.fingerprint,
        normal_order_reference_fingerprint=(
            spec.ensemble.normal_order_reference_fingerprint
        ),
        q0_policy_fingerprint=spec.ensemble.q0_policy_fingerprint,
        interaction_receipt_fingerprint=(
            spec.shared_functional.interaction_receipt_fingerprint
        ),
        shared_functional_receipt_fingerprint=spec.shared_functional.fingerprint,
        attested_source_receipt_fingerprint=spec.attested_source.fingerprint,
        expected_array_payload_manifest_sha256=expected_manifest,
        affine_anchor_inventory_sha256=_AFFINE_ANCHOR_INVENTORY_SHA256,
        q0_probe_inventory_sha256=_Q0_PROBE_INVENTORY_SHA256,
        signed_q_probe_inventory_sha256=(
            _SIGNED_Q_PROBE_INVENTORY_SHA256
        ),
        q_probe_inventory_sha256=q_inventory,
        fock_step_ladder=(
            spec.shared_functional.fock_finite_difference.finite_difference_step_ladder
        ),
        hessian_step_ladder=(
            spec.shared_functional.hessian_finite_difference.finite_difference_step_ladder
        ),
        replay_loader_implementation_fingerprint=(
            provider.replay_loader_implementation_fingerprint
        ),
        functional_probe_loader_implementation_fingerprint=(
            provider.functional_probe_loader_implementation_fingerprint
        ),
        functional_replay_payload_schema_fingerprint=(
            provider.functional_replay_payload_schema_fingerprint
        ),
        functional_replay_abi_fingerprint=(
            provider.functional_replay_abi_fingerprint
        ),
        direct_displaced_fock_implementation_fingerprint=(
            provider.direct_displaced_fock_implementation_fingerprint
        ),
        direct_interaction_builder_implementation_fingerprint=(
            provider.direct_interaction_builder_implementation_fingerprint
        ),
        direct_full_fock_builder_implementation_fingerprint=(
            provider.direct_full_fock_builder_implementation_fingerprint
        ),
        direct_builder_dependency_archive_fingerprint=(
            provider.direct_builder_dependency_archive_fingerprint
        ),
        detached_approval_manifest_sha256=approval.manifest_sha256,
        detached_approval_provenance=approval.provenance,
    )
    assert provider.replay_loader_calls == []
    assert provider.functional_loader_calls == []
    assert provider.scalar_energy_calls == 0
    assert provider.fock_derivative_calls == 0
    assert provider.hessian_calls == 0
    assert provider.direct_displaced_calls == 0
    return binding, contract, provider


def test_paper_facts_are_exact_and_qualified() -> None:
    target = abc.Vituri2024HalfMetalPaperTarget()
    assert target.half_metal_spin_polarized
    assert target.one_hole_pocket_per_valley
    assert target.intervalley_hund_omitted
    assert target.independent_valley_spin_rotations
    assert target.fig3a_density_cm2 == -1.0e12
    assert target.fig3bd_density_range_cm2 == (-1.1e12, -1.0e12)
    assert target.unrestricted_hf and target.optimal_damping
    assert target.many_broken_symmetry_initial_conditions
    assert target.direct_branch_energy_comparison and target.transfer_learning_near_transitions
    assert not target.exact_numerical_scf_policy_reported
    assert target.fig2_delta1_mev == 28.0
    assert target.fig3_delta1_mev is None
    assert "at odds with experiment" in target.hole_pocket_qualifier
    assert "model simplification" in target.hund_qualifier
    assert "unresolved" in target.fig3_delta1_qualifier
    with pytest.raises(ValueError, match="paper-direct facts"):
        abc.Vituri2024HalfMetalPaperTarget(fig3_delta1_mev=28.0)


def test_receipt_set_complete_never_resolves_execution_replay() -> None:
    default = abc.Vituri2024HalfMetalHFSpec.paper_default()
    assert default.missing_receipts == (
        "geometry",
        "ensemble",
        "scf_policy",
        "shared_functional",
        "attested_source",
    )
    assert not default.receipt_set_complete
    assert default.unresolved_authorities[-1] == "execution_replay"
    with pytest.raises(RuntimeError, match="receipt set is incomplete"):
        default.require_receipt_set_complete()

    complete = _complete()
    complete.require_receipt_set_complete()
    assert complete.receipt_set_complete
    assert complete.missing_receipts == ()
    assert complete.unresolved_authorities == ("execution_replay",)
    status = complete.status
    assert status.receipt_set_complete
    assert status.scientific_execution_verified is False
    assert status.arrays_recomputed is False
    assert status.provider_methods_executed is False
    assert status.paper_reproduction_verified is False
    assert complete.paper_direct_claim_allowed is False
    assert not hasattr(complete, "metadata_resolved")


def test_geometry_closes_k_mesh_internal_flavors_state_counts_and_no_wrap() -> None:
    receipt = _geometry()
    nk = receipt.mesh_point_count
    assert receipt.area_angstrom_squared == 20_000.0
    assert receipt.finite_area_receipt_fingerprint == _area().fingerprint
    assert receipt.mesh_order == "row_major_cartesian_k"
    assert receipt.core_state_nk == nk
    assert receipt.per_valley_k_count == nk
    assert receipt.valley_representation == "internal_flavor_axis"
    assert receipt.spin_count == 2
    assert receipt.total_active_state_count == 4 * nk
    assert receipt.selected_spin_state_count == 2 * nk
    assert receipt.internal_flavor_order == abc.INTERNAL_FLAVOR_ORDER
    assert receipt.array_layout == abc.REPLAY_ARRAY_LAYOUT
    assert receipt.array_conversion == abc.REPLAY_ARRAY_CONVERSION
    assert receipt.state_sum_weight == 1.0
    assert receipt.state_sum_weight_sum == nk
    assert receipt.state_sum_weight_sum_residual == 0.0
    assert (
        receipt.boundary_policy,
        receipt.torus_policy,
        receipt.reciprocal_carry_policy,
    ) == ("finite_domain_no_wrap", "not_a_reciprocal_torus", "no_reciprocal_carry")
    with pytest.raises(ValueError, match="state counts"):
        _geometry(core_state_nk=2 * nk)
    with pytest.raises(ValueError, match="state counts"):
        _geometry(total_active_state_count=2 * nk)
    with pytest.raises(ValueError, match="k-only mesh"):
        _geometry(mesh_order="valley_then_row_major_cartesian_k")
    with pytest.raises(ValueError, match="weight exactly 1"):
        _geometry(state_sum_weight=0.5)
    with pytest.raises(ValueError, match="residual does not match"):
        _geometry(state_sum_weight_sum=19.0)
    with pytest.raises(ValueError, match="k-only mesh"):
        _geometry(quadrature_rule="arbitrary_weighted_rule")
    with pytest.raises(ValueError, match="boundary/torus/carry"):
        _geometry(torus_policy="periodic_torus")


def test_geometry_gap_context_binds_two_valleys_times_nk_and_source() -> None:
    receipt = _geometry()
    assert receipt.active_band_index == 2
    assert receipt.valleys == (-1, 1)
    assert receipt.domain_gap_point_count == 2 * receipt.mesh_point_count
    with pytest.raises(ValueError, match="domain-gap context fingerprint"):
        replace(receipt, delta1_mev=27.0)
    with pytest.raises(ValueError, match="domain-gap context fingerprint"):
        replace(receipt, source_commit="c" * 40)
    with pytest.raises(ValueError, match="active band index 2"):
        _geometry(active_band_index=1)
    with pytest.raises(ValueError, match=r"2\*Nk"):
        _geometry(domain_gap_point_count=receipt.mesh_point_count)


def test_ensemble_is_fixed_density_canonical_with_typed_reference_and_q0() -> None:
    receipt = _ensemble()
    assert receipt.ensemble == "fixed_density"
    assert receipt.branch_thermodynamic_functional == "fixed_density_canonical_energy_ev"
    assert receipt.delta1_authority == "reproduction_choice"
    assert receipt.paper_direct_claim_allowed is False
    assert receipt.q0_neutralizing_background_kind != (
        receipt.interaction_analytic_kernel_q0_policy
    )
    with pytest.raises(ValueError, match="fixed-density canonical only"):
        _ensemble(ensemble="grand_canonical")
    with pytest.raises(ValueError, match="locked typed policies"):
        _ensemble(occupation_policy="fractional_fermi_dirac")
    with pytest.raises(ValueError, match="normal-order reference fingerprint"):
        replace(receipt, normal_order_reference_evidence_sha256=_SHA["7"])
    with pytest.raises(ValueError, match="hole density"):
        _ensemble(target_density_cm2=1.0e12)


def test_scf_policy_binds_exact_core_args_callbacks_and_integer_seed_pairs() -> None:
    receipt = _scf()
    assert receipt.core_module == "mean_field.core.hf"
    assert receipt.core_entrypoint == "run_hartree_fock_problem"
    assert receipt.convergence_rule == "raw"
    assert receipt.convergence_metric_identity.endswith("calculate_norm_convergence")
    assert receipt.convergence_metric_normalization == (
        "frobenius_updated_minus_previous_over_frobenius_updated"
    )
    assert receipt.precision == 1.0e-9
    assert receipt.branch_energy_tolerance_ev == 2.0e-9
    assert receipt.max_iter == 500
    assert receipt.oda_stall_threshold == 1.0e-3
    assert receipt.max_oda_lambda == 1.0
    assert [
        (item.seed_label, item.seed_value, item.init_mode)
        for item in receipt.seed_records
    ] == [
        ("spin_plus", 101, "spin_polarized_plus"),
        ("spin_minus", 202, "spin_polarized_minus"),
        ("random_broken", 303, "random_hermitian"),
    ]
    assert receipt.uniform_weight_representation.startswith("implicit_dimensionless_unit")
    assert receipt.final_exit_semantics == "core_exit_reason_plus_recomputed_final_raw_metric"
    assert receipt.forks_solver is False
    assert receipt.weighted_quadrature_claimed is False
    with pytest.raises(TypeError, match="strict integer"):
        abc.Vituri2024SCFSeedReceipt("bad", 1.5, "random")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hook order"):
        _scf(callback_receipts=tuple(reversed(_callbacks())))
    with pytest.raises(ValueError, match="exact generic core"):
        _scf(core_entrypoint="local_scf_fork")


def test_shared_functional_binds_e_f_and_f_h_fd_comparison_pairs() -> None:
    geometry, ensemble = _geometry(), _ensemble()
    shared = _shared(geometry, ensemble)
    assert shared.geometry_receipt_fingerprint == geometry.fingerprint
    assert shared.ensemble_receipt_fingerprint == ensemble.fingerprint
    assert shared.normal_order_reference_fingerprint == (
        ensemble.normal_order_reference_fingerprint
    )
    assert shared.q0_policy_fingerprint == ensemble.q0_policy_fingerprint
    assert shared.fock_finite_difference.source_state_sha256 == shared.source_state_sha256
    assert shared.fock_finite_difference.comparison_identity == (
        "scalar_energy_vs_fock_derivative"
    )
    assert (
        shared.fock_finite_difference.left_implementation_fingerprint,
        shared.fock_finite_difference.right_implementation_fingerprint,
    ) == (
        shared.scalar_energy.implementation_fingerprint,
        shared.fock_derivative.implementation_fingerprint,
    )
    assert shared.hessian_finite_difference.comparison_identity == (
        "fock_derivative_vs_finite_q_hessian"
    )
    assert (
        shared.hessian_finite_difference.left_implementation_fingerprint,
        shared.hessian_finite_difference.right_implementation_fingerprint,
    ) == (
        shared.fock_derivative.implementation_fingerprint,
        shared.finite_q_hessian.implementation_fingerprint,
    )
    assert (
        shared.fock_finite_difference.perturbation_normalization
        == "unit_frobenius_hermitian_trace_zero_density_direction"
    )
    assert shared.fock_finite_difference.q_probe_inventory_sha256 is None
    assert (
        shared.hessian_finite_difference.perturbation_normalization
        == "unit_frobenius_complexified_independent_signed_q_block_pair"
    )
    assert (
        shared.hessian_finite_difference.q_probe_inventory_sha256
        == _Q_CHART_INVENTORY_SHA256
    )
    assert len(shared.hessian_finite_difference.finite_difference_step_ladder) == 3
    assert shared.hessian_finite_difference.matrix_norm == "frobenius"
    assert shared.fock_output == "full_fock_h0_plus_interaction"
    assert shared.stored_density_pairing == (
        "real_bilinear_sum_abk_no_conjugation_over_nk"
    )
    assert shared.density_direction_convention == "fixed_density_affine_directions"
    assert "projector_tangent" not in shared.fock_finite_difference.perturbation_normalization

    alien_component = replace(shared.fock_derivative, source_commit="c" * 40)
    with pytest.raises(ValueError, match=r"one source artifact\+commit"):
        _shared(geometry, ensemble, fock_derivative=alien_component)
    alien_source_fd = _fd(
        "fock_first_derivative",
        shared.scalar_energy.implementation_fingerprint,
        shared.fock_derivative.implementation_fingerprint,
        geometry,
        ensemble,
        source_commit="c" * 40,
    )
    with pytest.raises(ValueError, match="evidence source"):
        _shared(geometry, ensemble, fock_finite_difference=alien_source_fd)
    alien_pair_fd = _fd(
        "fock_first_derivative",
        _SHA["9"],
        shared.fock_derivative.implementation_fingerprint,
        geometry,
        ensemble,
    )
    with pytest.raises(ValueError, match="implementation pair"):
        _shared(geometry, ensemble, fock_finite_difference=alien_pair_fd)
    alien_hessian_pair_fd = _fd(
        "finite_q_hessian",
        shared.scalar_energy.implementation_fingerprint,
        shared.finite_q_hessian.implementation_fingerprint,
        geometry,
        ensemble,
    )
    with pytest.raises(ValueError, match="implementation pair"):
        _shared(
            geometry,
            ensemble,
            hessian_finite_difference=alien_hessian_pair_fd,
        )
    with pytest.raises(ValueError, match="comparison identity"):
        _fd(
            "fock_first_derivative",
            shared.scalar_energy.implementation_fingerprint,
            shared.fock_derivative.implementation_fingerprint,
            geometry,
            ensemble,
            comparison_identity="fock_derivative_vs_finite_q_hessian",
        )
    with pytest.raises(ValueError, match="q/probe inventory"):
        _fd(
            "finite_q_hessian",
            shared.fock_derivative.implementation_fingerprint,
            shared.finite_q_hessian.implementation_fingerprint,
            geometry,
            ensemble,
            q_probe_inventory_sha256=None,
        )
    with pytest.raises(ValueError, match="evidence context fingerprint"):
        replace(shared.hessian_finite_difference, source_state_sha256=_SHA["5"])


def test_attested_source_closes_density_state_counts_pockets_and_selected_exit() -> None:
    spec = _complete()
    assert spec.geometry and spec.scf_policy and spec.attested_source
    source = spec.attested_source
    assert isinstance(source.selected_spin_hole_count, int)
    assert source.valley_plus_hole_count == source.valley_minus_hole_count == 1
    assert source.selected_spin_hole_count == 2
    assert source.target_density_cm2 == (
        -source.selected_spin_hole_count
        / spec.geometry.area_angstrom_squared
        * 1.0e16
    )
    metallicity = source.metallicity_evidence
    assert source.chemical_potential_ev == metallicity.chemical_potential_ev
    assert metallicity.selected_spin_unoccupied_state_count == (
        source.selected_spin_hole_count
    )
    assert (
        metallicity.selected_spin_occupied_state_count
        + metallicity.selected_spin_unoccupied_state_count
        == spec.geometry.selected_spin_state_count
    )
    assert source.attested_exit_reason == "converged"
    assert source.final_replay_raw_metric <= spec.scf_policy.precision
    selected_row = next(
        item
        for item in source.branch_records
        if item.seed.seed_label == source.selected_branch_label
    )
    assert selected_row.attested_exit_reason == "converged"
    assert selected_row.final_replay_raw_metric == source.final_replay_raw_metric
    assert selected_row.final_replay_raw_metric != selected_row.terminal_norm_raw
    assert tuple(item.valley for item in source.pocket_evidence) == (-1, 1)
    assert tuple(item.hole_state_count for item in source.pocket_evidence) == (1, 1)
    assert all(item.hole_component_count == 1 for item in source.pocket_evidence)
    assert all(
        item.lifshitz_margin_ev > item.lifshitz_tolerance_ev
        for item in source.pocket_evidence
    )
    assert metallicity.selected_spin_band_min_ev < source.chemical_potential_ev
    assert metallicity.selected_spin_band_max_ev > source.chemical_potential_ev
    assert tuple(item.seed for item in source.branch_records) == spec.scf_policy.seed_records
    assert source.branch_energy_functional_fingerprint == (
        spec.shared_functional.scalar_energy.fingerprint  # type: ignore[union-attr]
    )
    assert not hasattr(source, "converged")


def test_attested_source_rejects_hole_pocket_metallicity_and_replay_drift() -> None:
    geometry, ensemble, scf = _geometry(), _ensemble(), _scf()
    shared = _shared(geometry, ensemble)
    with pytest.raises(TypeError, match="strict integer"):
        _source(
            geometry,
            ensemble,
            scf,
            shared,
            selected_spin_hole_count=2.0,
        )
    with pytest.raises(ValueError, match="equal nonzero"):
        _source(
            geometry,
            ensemble,
            scf,
            shared,
            valley_plus_hole_count=2,
            selected_spin_hole_count=3,
        )
    with pytest.raises(ValueError, match="exactly one connected"):
        _pocket(-1, geometry, ensemble, hole_component_count=2)
    with pytest.raises(ValueError, match="pocket hole-state count"):
        _source(
            geometry,
            ensemble,
            scf,
            shared,
            pocket_evidence=(
                _pocket(-1, geometry, ensemble, hole_state_count=2),
                _pocket(1, geometry, ensemble),
            ),
        )
    with pytest.raises(ValueError, match="metallicity unoccupied count"):
        _source(
            geometry,
            ensemble,
            scf,
            shared,
            metallicity_evidence=_metallicity(
                geometry,
                ensemble,
                selected_spin_occupied_state_count=37,
                selected_spin_unoccupied_state_count=3,
            ),
        )
    with pytest.raises(ValueError, match="Lifshitz margin"):
        _pocket(-1, geometry, ensemble, lifshitz_margin_ev=1.0e-5)
    with pytest.raises(ValueError, match="straddle"):
        _metallicity(geometry, ensemble, selected_spin_band_max_ev=-0.02)
    with pytest.raises(ValueError, match="selected source exit must be converged"):
        _source(
            geometry,
            ensemble,
            scf,
            shared,
            attested_exit_reason="max_iter",
        )
    with pytest.raises(ValueError, match="final replay exceeds"):
        _source(
            geometry,
            ensemble,
            scf,
            shared,
            final_replay_raw_metric=1.0e-4,
        )


def test_aggregate_rejects_density_identity_and_crossed_source_fingerprints() -> None:
    geometry, ensemble, scf = _geometry(), _ensemble(), _scf()
    shared = _shared(geometry, ensemble)
    source = _source(geometry, ensemble, scf, shared)
    with pytest.raises(ValueError, match="density identity"):
        abc.Vituri2024HalfMetalHFSpec(
            geometry=geometry,
            ensemble=ensemble,
            scf_policy=scf,
            shared_functional=shared,
            attested_source=_source(
                geometry,
                ensemble,
                scf,
                shared,
                valley_plus_hole_count=2,
                valley_minus_hole_count=2,
                selected_spin_hole_count=4,
                metallicity_evidence=_metallicity(
                    geometry,
                    ensemble,
                    selected_spin_occupied_state_count=36,
                    selected_spin_unoccupied_state_count=4,
                ),
                pocket_evidence=(
                    _pocket(-1, geometry, ensemble, hole_state_count=2),
                    _pocket(1, geometry, ensemble, hole_state_count=2),
                ),
            ),
        )
    with pytest.raises(ValueError, match="shared functional fingerprint mismatch"):
        abc.Vituri2024HalfMetalHFSpec(
            geometry=geometry,
            ensemble=ensemble,
            scf_policy=scf,
            shared_functional=shared,
            attested_source=replace(
                source, shared_functional_receipt_fingerprint=_SHA["9"]
            ),
        )
    crossed_ensemble = _ensemble(
        normal_order_reference_evidence_sha256=_SHA["7"]
    )
    with pytest.raises(ValueError, match=r"source artifact\+commit|ensemble mismatch"):
        abc.Vituri2024HalfMetalHFSpec(
            geometry=geometry,
            ensemble=crossed_ensemble,
            scf_policy=scf,
            shared_functional=shared,
        )
    crossed_geometry = _geometry(source_commit="c" * 40)
    with pytest.raises(ValueError, match=r"source artifact\+commit identities"):
        abc.Vituri2024HalfMetalHFSpec(
            geometry=crossed_geometry,
            ensemble=ensemble,
        )


def test_aggregate_rejects_selected_spin_state_total_drift() -> None:
    geometry, ensemble, scf = _geometry(), _ensemble(), _scf()
    shared = _shared(geometry, ensemble)
    source = _source(
        geometry,
        ensemble,
        scf,
        shared,
        metallicity_evidence=_metallicity(
            geometry,
            ensemble,
            selected_spin_occupied_state_count=37,
            selected_spin_unoccupied_state_count=2,
        ),
    )
    with pytest.raises(ValueError, match="state-count closure"):
        abc.Vituri2024HalfMetalHFSpec(
            geometry=geometry,
            ensemble=ensemble,
            scf_policy=scf,
            shared_functional=shared,
            attested_source=source,
        )


def test_branch_rows_bind_exact_core_exit_semantics_and_convergence_rule() -> None:
    geometry, ensemble = _geometry(), _ensemble()
    shared = _shared(geometry, ensemble)

    def build_spec(
        scf: abc.Vituri2024HFSCFPolicyReceipt,
        rows: tuple[abc.Vituri2024BranchEnergyReceipt, ...],
    ) -> abc.Vituri2024HalfMetalHFSpec:
        return abc.Vituri2024HalfMetalHFSpec(
            geometry=geometry,
            ensemble=ensemble,
            scf_policy=scf,
            shared_functional=shared,
            attested_source=_source(
                geometry,
                ensemble,
                scf,
                shared,
                branch_records=rows,
            ),
        )

    raw_scf = _scf()
    raw_rows = list(_branches())
    build_spec(raw_scf, tuple(raw_rows))

    stall_row = replace(
        raw_rows[2],
        attested_exit_reason="oda_stall",
        iterations=40,
        terminal_norm_raw=2.0e-5,
        terminal_norm_mixed=1.0e-5,
        terminal_norm_selected=2.0e-5,
        terminal_oda_lambda=5.0e-4,
        final_replay_raw_metric=1.0e-5,
    )
    build_spec(raw_scf, tuple((*raw_rows[:2], stall_row)))
    max_iter_row = replace(
        stall_row,
        attested_exit_reason="max_iter",
        iterations=raw_scf.max_iter,
        terminal_oda_lambda=raw_scf.oda_stall_threshold,
    )
    build_spec(raw_scf, tuple((*raw_rows[:2], max_iter_row)))
    precision_boundary_row = replace(
        raw_rows[2],
        terminal_norm_raw=raw_scf.precision,
        terminal_norm_selected=raw_scf.precision,
    )
    build_spec(raw_scf, tuple((*raw_rows[:2], precision_boundary_row)))

    with pytest.raises(ValueError, match="converged exit exceeds"):
        build_spec(
            raw_scf,
            tuple(
                (*raw_rows[:2], replace(raw_rows[2], terminal_norm_raw=2.0e-5,
                                         terminal_norm_selected=2.0e-5))
            ),
        )
    with pytest.raises(ValueError, match="non-converged exit"):
        build_spec(
            raw_scf,
            tuple(
                (
                    *raw_rows[:2],
                    replace(
                        stall_row,
                        terminal_norm_raw=raw_scf.precision,
                        terminal_norm_selected=raw_scf.precision,
                    ),
                )
            ),
        )
    with pytest.raises(ValueError, match="stall threshold"):
        build_spec(
            raw_scf,
            tuple(
                (
                    *raw_rows[:2],
                    replace(
                        stall_row,
                        terminal_oda_lambda=raw_scf.oda_stall_threshold,
                    ),
                )
            ),
        )
    with pytest.raises(ValueError, match="max-iteration exit"):
        build_spec(
            raw_scf,
            tuple((*raw_rows[:2], replace(max_iter_row, iterations=499))),
        )
    with pytest.raises(ValueError, match="max-iteration exit"):
        build_spec(
            raw_scf,
            tuple(
                (*raw_rows[:2], replace(max_iter_row, terminal_oda_lambda=5.0e-4))
            ),
        )
    with pytest.raises(ValueError, match=r"outside \[1,max_iter\]"):
        build_spec(
            raw_scf,
            tuple((*raw_rows[:2], replace(raw_rows[2], iterations=501))),
        )
    with pytest.raises(ValueError, match="iterations must be positive"):
        replace(raw_rows[2], iterations=0)

    mixed_scf = _scf(convergence_rule="mixed")
    mixed_rows = list(_branches("mixed"))
    build_spec(mixed_scf, tuple(mixed_rows))
    with pytest.raises(ValueError, match="convergence_rule"):
        build_spec(
            mixed_scf,
            tuple(
                (
                    *mixed_rows[:2],
                    replace(
                        mixed_rows[2],
                        terminal_norm_selected=mixed_rows[2].terminal_norm_raw,
                    ),
                )
            ),
        )


def test_selected_branch_must_be_converged_and_bind_distinct_final_replay() -> None:
    geometry, ensemble, scf = _geometry(), _ensemble(), _scf()
    shared = _shared(geometry, ensemble)
    rows = list(_branches())
    rows[0] = replace(
        rows[0],
        attested_exit_reason="oda_stall",
        terminal_norm_raw=2.0e-5,
        terminal_norm_mixed=1.0e-5,
        terminal_norm_selected=2.0e-5,
        terminal_oda_lambda=5.0e-4,
        final_replay_raw_metric=1.0e-5,
    )
    with pytest.raises(ValueError, match="selected branch row/source exit"):
        _source(
            geometry,
            ensemble,
            scf,
            shared,
            branch_records=tuple(rows),
        )


def test_crossed_pocket_source_and_branch_seed_inventory_fail_closed() -> None:
    geometry, ensemble, scf = _geometry(), _ensemble(), _scf()
    shared = _shared(geometry, ensemble)
    crossed_pocket = _pocket(
        -1,
        geometry,
        ensemble,
        ordered_occupations_sha256=_SHA["8"],
    )
    source = _source(
        geometry,
        ensemble,
        scf,
        shared,
        pocket_evidence=(crossed_pocket, _pocket(1, geometry, ensemble)),
    )
    with pytest.raises(ValueError, match="pocket evidence/source/geometry"):
        abc.Vituri2024HalfMetalHFSpec(
            geometry=geometry,
            ensemble=ensemble,
            scf_policy=scf,
            shared_functional=shared,
            attested_source=source,
        )
    changed_seed = abc.Vituri2024SCFSeedReceipt("spin_minus", 999, "spin_polarized_minus")
    rows = list(_branches())
    rows[1] = replace(rows[1], seed=changed_seed)
    changed_source = _source(
        geometry,
        ensemble,
        scf,
        shared,
        branch_records=tuple(rows),
    )
    with pytest.raises(ValueError, match="exact SCF seed inventory"):
        abc.Vituri2024HalfMetalHFSpec(
            geometry=geometry,
            ensemble=ensemble,
            scf_policy=scf,
            shared_functional=shared,
            attested_source=changed_source,
        )


def test_provider_identity_closes_across_all_receipts_and_binding() -> None:
    spec = _complete()
    assert spec.geometry and spec.ensemble and spec.scf_policy
    assert spec.shared_functional and spec.attested_source
    receipt_names = (
        "geometry",
        "ensemble",
        "scf_policy",
        "shared_functional",
        "attested_source",
    )
    receipts = {name: getattr(spec, name) for name in receipt_names}
    assert {receipt.provider_fingerprint for receipt in receipts.values()} == {
        _PROVIDER_FINGERPRINT
    }
    for drift_name in receipt_names:
        drifted = dict(receipts)
        drifted[drift_name] = replace(
            drifted[drift_name], provider_fingerprint=_SHA["9"]
        )
        with pytest.raises(ValueError, match="provider fingerprints"):
            abc.Vituri2024HalfMetalHFSpec(**drifted)

    provider = _Provider(spec)
    binding = abc.Vituri2024HalfMetalHFProviderBinding(spec, provider)
    assert binding.status.metadata_status == "provider_metadata_attested"
    assert binding.status.binding_execution_verified is False
    assert binding.status.scientific_execution_verified is False
    assert binding.status.arrays_recomputed is False
    assert binding.status.provider_methods_executed is False
    assert binding.status.paper_reproduction_verified is False
    assert not hasattr(binding, "provider_bound")
    assert not hasattr(binding, "executable_ready")

    missing = _Provider(spec)
    missing.evaluate_finite_q_hessian = None  # type: ignore[method-assign]
    with pytest.raises(TypeError, match="typed methods|must be callable"):
        abc.Vituri2024HalfMetalHFProviderBinding(spec, missing)
    mismatch = _Provider(spec)
    mismatch.q0_policy_fingerprint = _SHA["9"]
    with pytest.raises(ValueError, match="q0 policy fingerprint mismatch"):
        abc.Vituri2024HalfMetalHFProviderBinding(spec, mismatch)
    provider_drift = _Provider(spec)
    provider_drift.provider_fingerprint = _SHA["9"]
    with pytest.raises(ValueError, match="provider identity fingerprint mismatch"):
        abc.Vituri2024HalfMetalHFProviderBinding(spec, provider_drift)


def test_protocol_annotations_exports_and_no_solver_execution_surface() -> None:
    for name in preflight.__all__:
        assert getattr(abc, name) is getattr(preflight, name)
    annotations = inspect.get_annotations(
        preflight.Vituri2024HalfMetalHFProviderProtocol.evaluate_fock_derivative
    )
    assert annotations["density"] == "ComplexArray"
    assert annotations["return"] == "ComplexArray"
    source = inspect.getsource(preflight)
    benchmark_reference = (
        Path(__file__).resolve().parents[1]
        / "reference"
        / "TDHF_MOIRE_BENCHMARKS.md"
    ).read_text(encoding="utf-8")
    for forbidden_status in (
        "metadata_resolved",
        "provider_bound",
        "executable_ready",
    ):
        assert forbidden_status not in source
        assert forbidden_status not in benchmark_reference
    assert "np.linalg" not in source
    assert "scipy" not in source
    assert "run_hartree_fock_iterations(" not in source
    assert "run_hartree_fock_problem(" not in source
    assert not hasattr(preflight, "run_hartree_fock")


def test_synthetic_area_density_hole_identity_is_physically_consistent() -> None:
    area_angstrom_squared = _area().area_angstrom_squared
    selected_spin_holes = 2
    density_cm2 = -selected_spin_holes / area_angstrom_squared * 1.0e16
    assert density_cm2 == -1.0e12
    assert np.isfinite(density_cm2)


def test_array_replay_success_binds_hashes_structure_and_partial_status() -> None:
    binding, payload, provider = _replay_case()
    receipt = abc.replay_vituri2024_half_metal_hf_arrays(binding)
    assert provider.replay_loader_calls == [_SOURCE_ARTIFACT]
    assert receipt.internal_flavor_order == ((-1, -1), (-1, 1), (1, -1), (1, 1))
    assert receipt.array_layout == "internal_flavor_internal_flavor_k_final"
    assert receipt.array_conversion == "identity_no_transpose"
    assert receipt.orbital_order == "flavor_major_then_k"
    assert (
        receipt.ordered_orbitals_descriptor_label
        == "orbital_index_descriptor"
    )
    assert receipt.active_band_states_gauge_scope.endswith("not_paper_gauge")
    assert receipt.canonical_basis_kind == abc.CANONICAL_BASIS_KIND
    assert receipt.residual_norm == "entrywise_max_abs"
    assert receipt.hashes.ordered_momentum_mesh_sha256 == _ORDERED_MESH_HASH
    assert receipt.hashes.ordered_orbitals_sha256 == _ARRAY_HASHES["orbitals"]
    assert (
        receipt.hashes.active_band_states_sha256
        == _ARRAY_HASHES["active_band_states"]
    )
    assert receipt.hashes.h0_sha256 == _ARRAY_HASHES["h0"]
    assert receipt.hashes.interaction_h_sha256 == _ARRAY_HASHES["interaction_h"]
    assert (
        receipt.replay_loader_implementation_fingerprint
        == _REPLAY_LOADER_IMPLEMENTATION_FINGERPRINT
    )
    assert receipt.hashes.reconstructed_source_state_sha256 == payload.source_state_sha256
    assert len(receipt.fingerprint) == 64
    assert json.dumps(asdict(receipt), sort_keys=True, allow_nan=False)
    assert receipt.fingerprint == receipt.fingerprint
    assert (
        receipt.residuals.fock_decomposition_max_abs_ev
        == _BASE_FOCK_DECOMPOSITION_RESIDUAL
    )
    assert (
        receipt.residuals.active_band_state_norm_max_abs
        == _BASE_ACTIVE_BAND_STATE_NORM_RESIDUAL
    )
    assert all(value == 0.0 for value in (
        receipt.residuals.h0_hermiticity_max_abs_ev,
        receipt.residuals.interaction_h_hermiticity_max_abs_ev,
        receipt.residuals.projector_hermiticity_max_abs,
        receipt.residuals.projector_idempotency_max_abs,
        receipt.residuals.fock_hermiticity_max_abs_ev,
        receipt.residuals.fock_projector_commutator_max_abs_ev,
        receipt.residuals.projector_vs_occupation_max_abs,
        receipt.residuals.canonical_basis_diagonal_closure_max_abs_ev,
        receipt.residuals.aufbau_violation_ev,
        receipt.residuals.chemical_mu_occupation_violation_ev,
    ))
    evidence = receipt.occupation_evidence
    assert (evidence.valley_minus_hole_count, evidence.valley_plus_hole_count) == (1, 1)
    assert (evidence.selected_spin_occupied_state_count,
            evidence.selected_spin_unoccupied_state_count) == (38, 2)
    assert evidence.opposite_spin_hole_count == 0
    assert evidence.measured_density_cm2 == evidence.target_density_cm2 == -1.0e12
    assert tuple(item.component_cardinalities for item in receipt.base_pocket_evidence) == (
        (1,),
        (1,),
    )
    status = receipt.status
    assert status.arrays_loaded and status.array_hashes_verified
    assert status.source_structure_verified
    assert status.provider_methods_executed == ("load_half_metal_replay_payload",)
    assert status.scf_trajectory_replayed is False
    assert status.branch_table_replayed is False
    assert status.pocket_refinement_replayed is False
    assert status.functional_chain_replayed is False
    assert status.scientific_execution_verified is False
    assert status.paper_reproduction_verified is False
    for absent_status in ("executable_ready", "provider_bound", "metadata_resolved"):
        assert not hasattr(status, absent_status)
        assert not hasattr(receipt, absent_status)


def test_canonical_hashes_bind_shape_dtype_bytes_and_orbital_order() -> None:
    array = np.arange(6, dtype=np.float64).reshape(2, 3)
    golden = "a1762f487b340b78e2ac9770b45e3e09ee79c1fb4b8132320c389538d0e59598"
    assert abc.canonical_array_sha256(array) == golden
    independent_header = (
        b'{"byte_order":"C","dtype":"<f8","schema":'
        b'"vituri2024_canonical_array_v1","shape":[2,3]}'
    )
    independent_digest = hashlib.sha256()
    independent_digest.update(struct.pack(">Q", len(independent_header)))
    independent_digest.update(independent_header)
    independent_digest.update(struct.pack("<6d", *range(6)))
    assert independent_digest.hexdigest() == golden
    assert abc.canonical_array_sha256(array) == abc.canonical_array_sha256(array.copy())
    assert abc.canonical_array_sha256(array) != abc.canonical_array_sha256(array.T)
    assert abc.canonical_array_sha256(array) != abc.canonical_array_sha256(
        array.astype(np.complex128)
    )
    changed = array.copy()
    changed[0, 0] = 1.0
    assert abc.canonical_array_sha256(array) != abc.canonical_array_sha256(changed)
    changed_mesh = _REPLAY_ARRAYS["mesh"].copy()
    changed_mesh[0, 0] += 1.0e-12
    assert abc.canonical_orbital_order_sha256(changed_mesh) != _ARRAY_HASHES["orbitals"]


def test_layout_conversion_and_orbital_descriptor_contracts_are_explicit() -> None:
    geometry = _geometry()
    assert geometry.internal_flavor_order == abc.INTERNAL_FLAVOR_ORDER
    assert geometry.array_layout == abc.REPLAY_ARRAY_LAYOUT
    assert geometry.array_conversion == abc.REPLAY_ARRAY_CONVERSION
    with pytest.raises(ValueError, match="internal-flavor array layout"):
        _geometry(array_layout="core_state_k_then_internal_valley_then_spin")
    with pytest.raises(ValueError, match="internal-flavor array layout"):
        _geometry(array_conversion="transpose_k_first")

    spec = _complete()
    assert spec.attested_source
    source = spec.attested_source
    assert source.ordered_orbitals_descriptor_label == "orbital_index_descriptor"
    assert (
        source.ordered_orbitals_schema_fingerprint
        == abc.ORBITAL_INDEX_DESCRIPTOR_SCHEMA_FINGERPRINT
    )
    with pytest.raises(ValueError, match="descriptor/schema contract"):
        replace(
            source,
            ordered_orbitals_descriptor_label="active_band_state_array",
        )


@pytest.mark.parametrize(
    ("array_name", "error"),
    (
        ("fock", "canonical ordered Fock hash mismatch"),
        ("h0", "canonical ordered h0 hash mismatch"),
        ("interaction_h", "canonical ordered interaction_h hash mismatch"),
        ("active_band_states", "canonical active-band states hash mismatch"),
    ),
)
def test_array_replay_detects_mutation_against_attested_hashes(
    array_name: str, error: str
) -> None:
    spec = _complete()
    arrays = _array_copy()
    arrays[array_name].flat[0] += 1.0e-6
    payload = _payload(spec, arrays)
    provider = _Provider(spec, payload)
    binding = abc.Vituri2024HalfMetalHFProviderBinding(spec, provider)
    with pytest.raises(ValueError, match=error):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)


def test_attested_source_hash_context_binds_h0_interaction_and_active_states() -> None:
    spec = _complete()
    assert spec.geometry and spec.ensemble and spec.scf_policy
    assert spec.shared_functional and spec.attested_source
    for field_name in ("h0_sha256", "interaction_h_sha256", "active_band_states_sha256"):
        with pytest.raises(ValueError, match="source-state hash"):
            replace(spec.attested_source, **{field_name: _SHA["9"]})


def test_replay_payload_rejects_nonfinite_shape_dtype_and_occupations() -> None:
    spec = _complete()
    bad = _array_copy()
    bad["mesh"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _payload(spec, bad)
    bad = _array_copy(h0=np.zeros((4, 4, 19), dtype=np.complex128))
    with pytest.raises(ValueError, match="h0 shape"):
        _payload(spec, bad)
    bad = _array_copy(
        active_band_states=np.zeros((2, 6, 19), dtype=np.complex128)
    )
    with pytest.raises(ValueError, match="active_band_states shape"):
        _payload(spec, bad)
    bad = _array_copy(energies=_REPLAY_ARRAYS["energies"].astype(np.float32))
    with pytest.raises(TypeError, match="energies dtype"):
        _payload(spec, bad)
    bad = _array_copy(occupations=_REPLAY_ARRAYS["occupations"].astype(np.int32))
    with pytest.raises(TypeError, match="occupations dtype"):
        _payload(spec, bad)
    occupations = _REPLAY_ARRAYS["occupations"].copy()
    occupations[0, 0] = 2
    bad = _array_copy(occupations=occupations)
    with pytest.raises(ValueError, match="integer 0/1"):
        _payload(spec, bad)


def test_replay_payload_arrays_are_immutable_owned_copies() -> None:
    arrays = _array_copy()
    spec = _spec_for_arrays(arrays)
    payload = _payload(spec, arrays)
    arrays["mesh"][0, 0] = 99.0
    arrays["fock"][0, 0, 0] = 99.0
    assert payload.mesh[0, 0] != 99.0
    assert payload.fock[0, 0, 0] != 99.0
    for array in (
        payload.mesh,
        payload.active_band_states,
        payload.h0,
        payload.interaction_h,
        payload.fock,
        payload.projector,
        payload.energies,
        payload.occupations,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.flat[0] = 0
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_array_replay_gates_attested_fock_decomposition() -> None:
    arrays = _array_copy()
    arrays["fock"][0, 0, 0] += 5.0e-13
    binding, _, _ = _replay_case(arrays)
    with pytest.raises(ValueError, match="Fock decomposition residual"):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)


@pytest.mark.parametrize(
    ("residual_field", "error"),
    (
        ("fock_decomposition_residual_ev", "Fock decomposition residual"),
        ("h0_hermiticity_residual_ev", "h0 Hermiticity residual"),
        (
            "interaction_h_hermiticity_residual_ev",
            "interaction_h Hermiticity residual",
        ),
        ("active_band_state_norm_residual", "active-band state norm residual"),
        (
            "projector_vs_occupation_residual",
            "projector/occupation diagonal closure residual",
        ),
        (
            "fock_vs_diagonal_energy_residual_ev",
            "canonical-basis diagonal closure residual",
        ),
    ),
)
def test_array_replay_matches_each_dedicated_attested_residual(
    residual_field: str, error: str
) -> None:
    binding, _, _ = _replay_case(
        source_overrides={residual_field: 5.0e-13}
    )
    with pytest.raises(ValueError, match=error):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)


def test_array_replay_gates_active_band_state_norm() -> None:
    arrays = _array_copy()
    arrays["active_band_states"][0, :, 0] *= 1.01
    binding, _, _ = _replay_case(arrays)
    with pytest.raises(ValueError, match="active-band state norm"):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)


def test_array_replay_gates_matrix_and_diagonal_eigen_residuals() -> None:
    cases: list[tuple[str, dict[str, np.ndarray], str]] = []

    arrays = _array_copy()
    arrays["projector"][0, 1, 0] = 1.0e-4
    cases.append(("P Hermiticity", arrays, "projector Hermiticity"))

    arrays = _array_copy()
    arrays["projector"][0, 0, 0] = 0.5
    cases.append(("P idempotency", arrays, "projector idempotency"))

    arrays = _array_copy()
    arrays["h0"][0, 1, 1] = 1.0e-4
    arrays["fock"][0, 1, 1] = 1.0e-4
    cases.append(("h0 Hermiticity", arrays, "h0 Hermiticity"))

    arrays = _array_copy()
    arrays["interaction_h"][0, 1, 1] = 1.0e-4
    arrays["fock"][0, 1, 1] = 1.0e-4
    cases.append(("interaction_h Hermiticity", arrays, "interaction_h Hermiticity"))

    arrays = _array_copy()
    arrays["projector"][0:2, 0:2, 0] = np.asarray(
        [[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128
    )
    cases.append(("commutator", arrays, "Fock/projector commutator"))

    arrays = _array_copy()
    arrays["projector"][0, 0, 0] = 0.0
    arrays["projector"][1, 1, 0] = 1.0
    cases.append(("P diagonal", arrays, "projector/occupation diagonal"))

    arrays = _array_copy()
    arrays["fock"][0, 1, 1] = 1.0e-5
    arrays["fock"][1, 0, 1] = 1.0e-5
    arrays["h0"][0, 1, 1] = 1.0e-5
    arrays["h0"][1, 0, 1] = 1.0e-5
    cases.append(
        ("canonical diagonal", arrays, "canonical-basis diagonal closure")
    )

    for _label, arrays, expected_error in cases:
        binding, _, _ = _replay_case(arrays)
        with pytest.raises(ValueError, match=expected_error):
            abc.replay_vituri2024_half_metal_hf_arrays(binding)


def test_array_replay_compares_residual_value_not_only_tolerance() -> None:
    arrays = _array_copy()
    arrays["projector"][0, 1, 0] = 1.0e-10
    binding, _, _ = _replay_case(arrays)
    with pytest.raises(ValueError, match="does not match the attested receipt"):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)


def test_array_replay_gates_valley_spin_and_chemical_mu_closure() -> None:
    valley_arrays = _array_copy()
    valley_arrays["occupations"][1, 2] = 0
    valley_arrays["projector"][1, 1, 2] = 0.0
    valley_arrays["energies"][1, 2] = 0.01
    valley_arrays["fock"][1, 1, 2] = 0.01
    valley_arrays["h0"][1, 1, 2] = (
        0.01 - valley_arrays["interaction_h"][1, 1, 2]
    )
    valley_arrays["fock"][1, 1, 2] = (
        valley_arrays["h0"][1, 1, 2]
        + valley_arrays["interaction_h"][1, 1, 2]
    )
    valley_arrays["energies"][1, 2] = valley_arrays["fock"][1, 1, 2].real
    valley_arrays["occupations"][3, 19] = 1
    valley_arrays["projector"][3, 3, 19] = 1.0
    valley_arrays["energies"][3, 19] = -0.028
    valley_arrays["fock"][3, 3, 19] = -0.028
    valley_arrays["h0"][3, 3, 19] = (
        -0.028 - valley_arrays["interaction_h"][3, 3, 19]
    )
    valley_arrays["fock"][3, 3, 19] = (
        valley_arrays["h0"][3, 3, 19]
        + valley_arrays["interaction_h"][3, 3, 19]
    )
    valley_arrays["energies"][3, 19] = valley_arrays["fock"][3, 3, 19].real
    valley_gap = float(
        np.min(valley_arrays["energies"][valley_arrays["occupations"] == 0])
        - np.max(valley_arrays["energies"][valley_arrays["occupations"] == 1])
    )
    binding, _, _ = _replay_case(
        valley_arrays,
        source_overrides={
            "aufbau_min_unoccupied_minus_max_occupied_ev": valley_gap
        },
    )
    with pytest.raises(ValueError, match="valley -1 hole-count mismatch"):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)

    spin_arrays = _array_copy()
    spin_arrays["occupations"][0, 0] = 0
    spin_arrays["projector"][0, 0, 0] = 0.0
    spin_arrays["energies"][0, 0] = 0.01
    spin_arrays["fock"][0, 0, 0] = 0.01
    spin_arrays["h0"][0, 0, 0] = (
        0.01 - spin_arrays["interaction_h"][0, 0, 0]
    )
    spin_arrays["fock"][0, 0, 0] = (
        spin_arrays["h0"][0, 0, 0]
        + spin_arrays["interaction_h"][0, 0, 0]
    )
    spin_arrays["energies"][0, 0] = spin_arrays["fock"][0, 0, 0].real
    binding, _, _ = _replay_case(spin_arrays)
    with pytest.raises(ValueError, match="opposite-spin hole-count mismatch"):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)

    mu_arrays = _array_copy()
    mu_arrays["energies"][0, 0] = -0.015
    mu_arrays["fock"][0, 0, 0] = -0.015
    mu_arrays["h0"][0, 0, 0] = (
        -0.015 - mu_arrays["interaction_h"][0, 0, 0]
    )
    mu_arrays["fock"][0, 0, 0] = (
        mu_arrays["h0"][0, 0, 0]
        + mu_arrays["interaction_h"][0, 0, 0]
    )
    mu_arrays["energies"][0, 0] = mu_arrays["fock"][0, 0, 0].real
    mu_gap = float(
        np.min(mu_arrays["energies"][mu_arrays["occupations"] == 0])
        - np.max(mu_arrays["energies"][mu_arrays["occupations"] == 1])
    )
    binding, _, _ = _replay_case(
        mu_arrays,
        source_overrides={"aufbau_min_unoccupied_minus_max_occupied_ev": mu_gap},
    )
    with pytest.raises(ValueError, match="chemical-potential occupation closure"):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)


def test_array_replay_rejects_disconnected_base_pocket_without_refinement_claim() -> None:
    arrays = _array_copy()
    for flavor, k_index in ((1, 2), (3, 18)):
        arrays["occupations"][flavor, k_index] = 0
        arrays["projector"][flavor, flavor, k_index] = 0.0
        arrays["energies"][flavor, k_index] = 0.01
        arrays["fock"][flavor, flavor, k_index] = 0.01
        arrays["h0"][flavor, flavor, k_index] = (
            0.01 - arrays["interaction_h"][flavor, flavor, k_index]
        )
        arrays["fock"][flavor, flavor, k_index] = (
            arrays["h0"][flavor, flavor, k_index]
            + arrays["interaction_h"][flavor, flavor, k_index]
        )
        arrays["energies"][flavor, k_index] = arrays["fock"][
            flavor, flavor, k_index
        ].real
    hashes = _canonical_hashes(arrays)
    area = 40_000.0
    geometry = _geometry(
        area_angstrom_squared=area,
        finite_area_receipt_fingerprint=_area(area).fingerprint,
        ordered_momentum_mesh_sha256=abc.canonical_array_sha256(arrays["mesh"]),
    )
    ensemble, scf = _ensemble(), _scf()
    shared = _shared(geometry, ensemble, array_hashes=hashes)
    metallicity = _metallicity(
        geometry,
        ensemble,
        array_hashes=hashes,
        selected_spin_occupied_state_count=36,
        selected_spin_unoccupied_state_count=4,
    )
    pockets = (
        _pocket(-1, geometry, ensemble, array_hashes=hashes, hole_state_count=2),
        _pocket(1, geometry, ensemble, array_hashes=hashes, hole_state_count=2),
    )
    source = _source(
        geometry,
        ensemble,
        scf,
        shared,
        array_hashes=hashes,
        valley_minus_hole_count=2,
        valley_plus_hole_count=2,
        selected_spin_hole_count=4,
        metallicity_evidence=metallicity,
        pocket_evidence=pockets,
    )
    spec = abc.Vituri2024HalfMetalHFSpec(
        geometry=geometry,
        ensemble=ensemble,
        scf_policy=scf,
        shared_functional=shared,
        attested_source=source,
    )
    payload = _payload(spec, arrays)
    provider = _Provider(spec, payload)
    binding = abc.Vituri2024HalfMetalHFProviderBinding(spec, provider)
    with pytest.raises(ValueError, match="base-pocket component-count mismatch"):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)
    assert provider.replay_loader_calls == [_SOURCE_ARTIFACT]


@pytest.mark.parametrize(
    ("attribute", "replacement", "error"),
    (
        ("provider_fingerprint", _SHA["9"], "provider identity fingerprint mismatch"),
        ("source_commit", "c" * 40, "source artifact\\+commit mismatch"),
        ("source_artifact_sha256", _SHA["9"], "source artifact\\+commit mismatch"),
        ("spec_fingerprint", _SHA["9"], "provider/spec fingerprint mismatch"),
        ("source_state_sha256", _SHA["9"], "provider/source state fingerprint mismatch"),
        (
            "replay_loader_implementation_fingerprint",
            _SHA["9"],
            "loader implementation fingerprint mismatch",
        ),
    ),
)
def test_array_replay_rechecks_mutable_provider_identity_before_loading(
    attribute: str, replacement: str, error: str
) -> None:
    binding, _, provider = _replay_case()
    setattr(provider, attribute, replacement)
    with pytest.raises(ValueError, match=error):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)
    assert provider.replay_loader_calls == []


@pytest.mark.parametrize(
    "attribute",
    replay._REPLAY_PROVIDER_METADATA_FIELDS,
)
def test_preload_snapshot_validator_pins_every_provider_field(
    attribute: str,
) -> None:
    binding, _, provider = _replay_case()
    snapshot = replay._provider_metadata_snapshot(provider)
    snapshot[attribute] = "c" * 40 if attribute == "source_commit" else _SHA["9"]

    with pytest.raises(ValueError, match="preload provider snapshot"):
        replay._validate_preload_provider_snapshot(binding, snapshot)

    assert provider.replay_loader_calls == []


def test_array_replay_rejects_snapshot_substitution_after_valid_base_binding() -> None:
    spec = _complete()
    snapshot_values = _BaseBindingSnapshotSubstitutionProvider.snapshot_base_values(spec)
    payload = _payload(
        spec,
        provider_fingerprint=snapshot_values["provider_fingerprint"],
        source_commit=snapshot_values["source_commit"],
        source_artifact_sha256=snapshot_values["source_artifact_sha256"],
        spec_fingerprint=snapshot_values["spec_fingerprint"],
        source_state_sha256=snapshot_values["source_state_sha256"],
    )
    provider = _BaseBindingSnapshotSubstitutionProvider(spec, payload)
    binding = abc.Vituri2024HalfMetalHFProviderBinding(spec, provider)

    with pytest.raises(ValueError, match="preload provider snapshot provider identity mismatch"):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)

    assert provider.replay_loader_calls == []
    assert tuple(provider.snapshot_reads) == replay._REPLAY_PROVIDER_METADATA_FIELDS


@pytest.mark.parametrize(
    "attribute",
    replay._REPLAY_PROVIDER_METADATA_FIELDS,
)
def test_array_replay_rejects_provider_metadata_drift_during_loader(
    attribute: str,
) -> None:
    assert replay._REPLAY_PROVIDER_METADATA_FIELDS == (
        preflight.VITURI2024_BASE_PROVIDER_METADATA_FIELDS
        + (
            "replay_loader_implementation_fingerprint",
            "replay_payload_schema_fingerprint",
        )
    )
    binding, _, provider = _replay_case()
    provider.loader_mutation = (attribute, "mutated_during_loader")
    with pytest.raises(ValueError, match="metadata mutated during loader call"):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)
    assert provider.replay_loader_calls == [_SOURCE_ARTIFACT]


def test_array_replay_receipt_identity_uses_only_preload_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _complete()
    payload = _payload(spec)
    provider = _PostValidationAccessDriftProvider(spec, payload)
    binding = abc.Vituri2024HalfMetalHFProviderBinding(spec, provider)
    validate_identity = replay._validate_payload_against_provider_snapshot

    def validate_identity_then_arm_drift(
        checked_payload: abc.Vituri2024HalfMetalHFReplayPayload,
        provider_snapshot: dict[str, object],
    ) -> None:
        validate_identity(checked_payload, provider_snapshot)
        provider.arm_post_validation_drift()

    monkeypatch.setattr(
        replay,
        "_validate_payload_against_provider_snapshot",
        validate_identity_then_arm_drift,
    )
    receipt = abc.replay_vituri2024_half_metal_hf_arrays(binding)

    assert (
        receipt.provider_fingerprint,
        receipt.source_commit,
        receipt.source_artifact_sha256,
        receipt.spec_fingerprint,
    ) == (
        _PROVIDER_FINGERPRINT,
        _COMMIT,
        _SOURCE_ARTIFACT,
        spec.fingerprint,
    )
    assert provider.relevant_reads_after_validation == []


@pytest.mark.parametrize(
    ("identity_override", "error"),
    (
        ({"provider_fingerprint": _SHA["9"]}, "payload provider fingerprint mismatch"),
        ({"source_commit": "c" * 40}, "payload source commit mismatch"),
        ({"source_artifact_sha256": _SHA["9"]}, "payload source artifact mismatch"),
        ({"spec_fingerprint": _SHA["9"]}, "payload spec fingerprint mismatch"),
        ({"source_state_sha256": _SHA["9"]}, "payload source-state fingerprint mismatch"),
        (
            {"replay_loader_implementation_fingerprint": _SHA["9"]},
            "payload loader implementation fingerprint mismatch",
        ),
    ),
)
def test_array_replay_rejects_payload_identity_mismatch(
    identity_override: dict[str, object], error: str
) -> None:
    spec = _complete()
    payload = _payload(spec, _array_copy(), **identity_override)
    provider = _Provider(spec, payload)
    binding = abc.Vituri2024HalfMetalHFProviderBinding(spec, provider)
    with pytest.raises(ValueError, match=error):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)


def test_replay_evidence_dataclasses_validate_and_success_is_factory_only() -> None:
    receipt = abc.replay_vituri2024_half_metal_hf_arrays(_replay_case()[0])
    with pytest.raises(ValueError, match="lowercase SHA256"):
        replace(receipt.hashes, h0_sha256="bad")
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(receipt.residuals, active_band_state_norm_max_abs=-1.0)
    with pytest.raises(ValueError, match="valley-hole counts"):
        replace(receipt.occupation_evidence, selected_spin_hole_count=3)
    with pytest.raises(ValueError, match="component cardinalities"):
        replace(receipt.base_pocket_evidence[0], component_cardinalities=(1, 1))

    with pytest.raises(TypeError, match="factory_token"):
        abc.Vituri2024HalfMetalHFReplayStatus()  # type: ignore[call-arg]
    receipt_kwargs = {
        name: getattr(receipt, name)
        for name in inspect.signature(
            abc.Vituri2024HalfMetalHFReplayReceipt
        ).parameters
        if name != "_factory_token"
    }
    with pytest.raises(TypeError, match="factory_token"):
        abc.Vituri2024HalfMetalHFReplayReceipt(**receipt_kwargs)  # type: ignore[arg-type]
    serialized = asdict(receipt)
    assert "_factory_token" not in serialized
    assert "_factory_token" not in serialized["status"]


def test_replay_protocol_exports_and_requires_callable_loader() -> None:
    for name in replay.__all__:
        assert getattr(abc, name) is getattr(replay, name)
    spec = _complete()
    provider = _Provider(spec)
    provider.load_half_metal_replay_payload = None  # type: ignore[method-assign]
    binding = abc.Vituri2024HalfMetalHFProviderBinding(spec, provider)
    with pytest.raises(TypeError, match="replay loader protocol|must be callable"):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)
    source = inspect.getsource(replay)
    assert "run_hartree_fock_problem(" not in source
    assert "evaluate_scalar_energy(" not in source
    assert "evaluate_fock_derivative(" not in source
    assert "evaluate_finite_q_hessian(" not in source


def test_functional_replay_executes_all_registered_probes_with_narrow_claims() -> None:
    binding, contract, provider = _functional_case()
    assert provider.replay_loader_calls == []
    assert provider.functional_loader_calls == []
    assert provider.scalar_energy_calls == provider.fock_derivative_calls == 0
    assert provider.hessian_calls == provider.direct_displaced_calls == 0

    receipt = abc.replay_vituri2024_half_metal_hf_functional(binding, contract)

    assert receipt.contract_fingerprint == contract.fingerprint
    assert receipt.choice_fingerprint == contract.choice.fingerprint
    assert receipt.verifier_implementation_schema_fingerprint == (
        abc.FUNCTIONAL_REPLAY_VERIFIER_IMPLEMENTATION_SCHEMA_FINGERPRINT
    )
    assert receipt.verifier_module_ast_manifest_sha256 == (
        contract.verifier_module_ast_manifest_sha256
    )
    assert receipt.verifier_module_ast_manifest_sha256 == (
        abc.functional_replay_module_ast_manifest_sha256()
    )
    assert receipt.detached_approval_manifest_sha256 == (
        contract.detached_approval_manifest_sha256
    )
    assert receipt.approval_precedes_execution is True
    assert receipt.expected_array_payload_manifest_sha256 == (
        receipt.array_replay_payload_manifest_sha256
    )
    assert receipt.affine_anchor_inventory_sha256 == (
        _AFFINE_ANCHOR_INVENTORY_SHA256
    )
    assert receipt.q0_probe_inventory_sha256 == _Q0_PROBE_INVENTORY_SHA256
    assert receipt.signed_q_probe_inventory_sha256 == (
        _SIGNED_Q_PROBE_INVENTORY_SHA256
    )
    assert receipt.q_probe_inventory_sha256 == _Q_CHART_INVENTORY_SHA256
    assert receipt.scope == abc.FUNCTIONAL_REPLAY_SCOPE
    assert (
        receipt.affine_anchor_count,
        receipt.q0_probe_count,
        receipt.signed_q_probe_count,
        receipt.q_chart_count,
    ) == (3, 5, 6, 2)
    assert receipt.source_anchor.fock_entrywise_residual <= (
        receipt.source_anchor.fock_entrywise_registered_bound
    )
    assert receipt.source_anchor.interaction_entrywise_residual <= (
        receipt.source_anchor.interaction_entrywise_registered_bound
    )
    assert receipt.source_anchor.scalar_energy_residual <= (
        receipt.source_anchor.scalar_energy_registered_bound
    )
    assert receipt.source_anchor.fock_entrywise_operation_count == 2
    assert receipt.source_anchor.interaction_entrywise_operation_count == 3
    assert receipt.source_anchor.scalar_energy_operation_count == 2
    assert receipt.source_anchor.fock_entrywise_termwise_magnitude >= (
        receipt.source_anchor.fock_entrywise_result_scale
    )
    assert receipt.source_anchor.interaction_entrywise_termwise_magnitude >= (
        receipt.source_anchor.interaction_entrywise_result_scale
    )
    assert receipt.source_anchor.scalar_energy_termwise_magnitude >= (
        receipt.source_anchor.scalar_energy_result_scale
    )
    assert len(receipt.scalar_steps) == 3 * 5 * 3
    assert len(receipt.scalar_local_gates) == 3 * 5
    for probe_index in range(5):
        assert max(
            item.informativeness_abs
            for item in receipt.scalar_local_gates
            if item.probe_index == probe_index
        ) >= contract.choice.informativeness_floor
    assert len(receipt.matrix_steps) == 2 * (1 + 1 + 2) * 3
    assert len(receipt.matrix_local_gates) == 2 * (1 + 1 + 2)
    assert len(receipt.reciprocity) == 2 * 3
    assert all(item.residual <= item.registered_bound for item in receipt.scalar_steps)
    assert all(
        item.stability_max_abs <= item.stability_registered_bound
        for item in receipt.scalar_local_gates
    )
    assert all(
        item.residual_norm <= item.registered_bound for item in receipt.matrix_steps
    )
    assert all(
        item.stability_max_frobenius <= item.stability_registered_bound
        for item in receipt.matrix_local_gates
    )
    assert all(item.residual <= item.registered_bound for item in receipt.reciprocity)
    assert all(item.operation_count > 0 for item in receipt.scalar_steps)
    assert all(item.operation_count > 0 for item in receipt.matrix_steps)
    assert all(item.operation_count > 0 for item in receipt.reciprocity)
    assert provider.hessian_calls == 6
    assert provider.direct_displaced_calls == 6 * (1 + 2 * 3)
    assert len(receipt.call_records) == 239
    assert receipt.call_records[0].method == receipt.call_records[1].method
    assert receipt.call_records[0].argument_hashes == receipt.call_records[1].argument_hashes
    assert receipt.call_records[0].output_sha256 == receipt.call_records[1].output_sha256
    assert receipt.transcript_sha256 == _fp([asdict(item) for item in receipt.call_records])
    direct_records = [
        item for item in receipt.call_records
        if item.method == "evaluate_displaced_fock"
    ]
    assert direct_records and all(
        item.dependency_trace_sha256 is not None for item in direct_records
    )
    status = receipt.status
    assert status.local_registered_functional_probes_replayed is True
    assert status.global_functional_chain_verified is False
    assert status.scope == abc.FUNCTIONAL_REPLAY_SCOPE
    assert (
        status.affine_anchor_count,
        status.q0_probe_count,
        status.signed_q_probe_count,
        status.q_chart_count,
    ) == (3, 5, 6, 2)
    assert status.scf_trajectory_replayed is False
    assert status.branch_table_replayed is False
    assert status.pocket_refinement_replayed is False
    assert status.scientific_execution_verified is False
    assert status.paper_reproduction_verified is False
    assert "functional_chain_replayed" not in asdict(status)
    assert "projector_tangent" not in inspect.getsource(functional)
    assert "np.vdot" not in inspect.getsource(functional._pairing)


def test_functional_replay_ast_manifest_hashes_the_canonical_full_module() -> None:
    source = Path(functional.__file__).read_text(encoding="utf-8")
    canonical_ast = ast.dump(
        ast.parse(source),
        annotate_fields=True,
        include_attributes=False,
    )
    independently_generated = hashlib.sha256(
        canonical_ast.encode("utf-8")
    ).hexdigest()
    assert functional.functional_replay_module_ast_manifest_sha256() == (
        independently_generated
    )
    assert functional.functional_replay_module_ast_manifest_sha256(source) == (
        independently_generated
    )
    reformatted_source = ast.unparse(ast.parse(source))
    assert functional.functional_replay_module_ast_manifest_sha256(
        reformatted_source
    ) == independently_generated
    mutated_source = _semantically_mutated_functional_source(source)
    assert functional.functional_replay_module_ast_manifest_sha256(
        mutated_source
    ) != independently_generated


def test_functional_replay_rejects_stale_ast_manifest_before_provider_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, contract, provider = _functional_case()
    source = Path(functional.__file__).read_text(encoding="utf-8")
    mutated_source = _semantically_mutated_functional_source(source)
    mutated_path = tmp_path / "vituri2024_hf_functional_replay.py"
    mutated_path.write_text(mutated_source, encoding="utf-8")
    assert functional.functional_replay_module_ast_manifest_sha256(
        mutated_source
    ) != contract.verifier_module_ast_manifest_sha256

    monkeypatch.setattr(functional, "__file__", str(mutated_path))
    with pytest.raises(ValueError, match="verifier AST/source manifest"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)

    assert provider.replay_loader_calls == []
    assert provider.functional_loader_calls == []
    assert provider.scalar_energy_calls == 0
    assert provider.fock_derivative_calls == 0
    assert provider.hessian_calls == 0
    assert provider.direct_displaced_calls == 0


def test_functional_probe_and_q_inventory_hashes_have_independent_goldens() -> None:
    assert abc.FUNCTIONAL_REPLAY_VERIFIER_IMPLEMENTATION_SCHEMA_FINGERPRINT == (
        "93eb57bbe188f99307f8d9bedb78af2104704bafbb0c31887a8b6639aa52e034"
    )
    assert abc.Vituri2024FunctionalReplayChoice().fingerprint == (
        "9bf4d931ee68125e94d5de3e75de5b1e738c78b810913e2aada1f478344acc3b"
    )
    assert _AFFINE_ANCHOR_INVENTORY_SHA256 == (
        "f2ec507092d6c3b0859620ad93b4314555f57c48b2f623d6fd46c2b22118abdc"
    )
    assert _Q0_PROBE_INVENTORY_SHA256 == (
        "f844b2c5745702ff9b8b46f36db3ac02e4e9d0ecdfef912c15852ffb86418f3a"
    )
    assert _SIGNED_Q_PROBE_INVENTORY_SHA256 == (
        "9f0a754ecdc30716be23bd58b5e16a18a81e7ee8b7a67c03c97bf2490ffb8a74"
    )
    assert _Q_CHART_INVENTORY_SHA256 == (
        "49326562e251ae128b5315ce17255f5687d5a2ab733d037f6097b89d48376891"
    )
    assert tuple(chart.q_probe_index for chart in _SIGNED_Q_CHARTS) == (0, 1)
    assert tuple(chart.q_label for chart in _SIGNED_Q_CHARTS) == abc.Q_CHART_LABELS
    assert not np.array_equal(
        _SIGNED_Q_CHARTS[0].mesh_displacement,
        _SIGNED_Q_CHARTS[1].mesh_displacement,
    )
    assert np.array_equal(_SIGNED_Q_PROBE_INDICES, np.asarray((0, 0, 0, 1, 1, 1)))
    for q_index, chart in enumerate(_SIGNED_Q_CHARTS):
        base = 3 * q_index
        plus_boundary = chart.validity_masks[0] & ~chart.validity_masks[1]
        minus_boundary = chart.validity_masks[1] & ~chart.validity_masks[0]
        interior = chart.validity_masks[0] & chart.validity_masks[1]
        assert np.any(_SIGNED_Q_PROBES[base, 0, :, :, plus_boundary] != 0.0)
        assert np.all(_SIGNED_Q_PROBES[base, 1] == 0.0)
        assert np.any(_SIGNED_Q_PROBES[base + 1, 1, :, :, minus_boundary] != 0.0)
        assert np.all(_SIGNED_Q_PROBES[base + 1, 0] == 0.0)
        assert np.any(_SIGNED_Q_PROBES[base + 2, :, :, :, interior] != 0.0)
        assert np.allclose(
            np.sqrt(np.sum(np.abs(_SIGNED_Q_PROBES[base : base + 3]) ** 2, axis=(1, 2, 3, 4))),
            1.0,
        )


def test_functional_replay_binds_q_charts_to_geometry_and_source_mesh() -> None:
    binding, contract, provider = _functional_case(
        geometry_overrides={"mesh_shape": (5, 4)}
    )
    with pytest.raises(
        ValueError,
        match=r"mesh_shape does not equal spec\.geometry\.mesh_shape",
    ):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)
    assert provider.fock_derivative_calls == 0
    assert provider.hessian_calls == provider.direct_displaced_calls == 0

    # The synthetic provider's q action has one common coefficient and accepts
    # whichever registered Cartesian q it is given.  A self-consistent but
    # physically wrong chart registration must therefore be stopped by source
    # mesh coordinates, not by provider q dependence.
    wrong_q_charts = (
        replace(
            _SIGNED_Q_CHARTS[0],
            cartesian_q=np.asarray((0.02, -0.01), dtype=np.float64),
        ),
        replace(
            _SIGNED_Q_CHARTS[1],
            cartesian_q=np.asarray((-0.015, 0.025), dtype=np.float64),
        ),
    )
    binding, contract, provider = _functional_case(q_charts=wrong_q_charts)
    with pytest.raises(ValueError, match="does not match source momentum mesh edges"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)
    assert provider.fock_derivative_calls == 0
    assert provider.hessian_calls == provider.direct_displaced_calls == 0


def test_detached_approval_binds_choice_conventions_and_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, contract, _ = _functional_case()

    tolerance_constants = (
        ("absolute_tolerance", "_REGISTERED_ABSOLUTE_TOLERANCE"),
        ("relative_tolerance", "_REGISTERED_RELATIVE_TOLERANCE"),
        ("roundoff_ulps", "_REGISTERED_ROUNDOFF_ULPS"),
        (
            "slope_stability_tolerance",
            "_REGISTERED_SLOPE_STABILITY_TOLERANCE",
        ),
        ("informativeness_floor", "_REGISTERED_INFORMATIVENESS_FLOOR"),
        (
            "source_mesh_q_coordinate_tolerance_inverse_angstrom",
            "_REGISTERED_SOURCE_MESH_Q_COORDINATE_TOLERANCE_INVERSE_ANGSTROM",
        ),
    )
    for field_name, constant_name in tolerance_constants:
        with monkeypatch.context() as patch:
            changed_tolerance = 2.0 * float(getattr(contract.choice, field_name))
            patch.setattr(functional, constant_name, changed_tolerance)
            changed_choice = replace(
                contract.choice, **{field_name: changed_tolerance}
            )
            with pytest.raises(ValueError, match="detached approval manifest"):
                replace(
                    contract,
                    choice=changed_choice,
                    choice_fingerprint=changed_choice.fingerprint,
                )

    convention_fields = (
        ("fock_output", "FOCK_OUTPUT_CONVENTION", "changed_full_fock"),
        (
            "stored_density_pairing",
            "STORED_DENSITY_PAIRING",
            "changed_stored_density_pairing",
        ),
        (
            "density_direction_convention",
            "FIXED_DENSITY_DIRECTION_CONVENTION",
            "changed_density_direction",
        ),
    )
    for field_name, constant_name, changed_convention in convention_fields:
        with monkeypatch.context() as patch:
            patch.setattr(functional, constant_name, changed_convention)
            changed_choice = replace(
                contract.choice,
                **{field_name: changed_convention},
            )
            with pytest.raises(ValueError, match="detached approval manifest"):
                replace(
                    contract,
                    choice=changed_choice,
                    choice_fingerprint=changed_choice.fingerprint,
                )

    with monkeypatch.context() as patch:
        changed_verifier = _SHA["4"]
        patch.setattr(
            functional,
            "FUNCTIONAL_REPLAY_VERIFIER_IMPLEMENTATION_SCHEMA_FINGERPRINT",
            changed_verifier,
        )
        with pytest.raises(ValueError, match="detached approval manifest"):
            replace(
                contract,
                verifier_implementation_schema_fingerprint=changed_verifier,
            )


def test_source_anchor_high_cancellation_uses_termwise_roundoff_bound() -> None:
    choice = abc.Vituri2024FunctionalReplayChoice()
    result_scale = 1.0
    termwise_magnitude = 2.0e8
    operation_count = 2
    roundoff, registered_bound = functional._entrywise_bound(
        choice,
        result_scale=result_scale,
        termwise_magnitude=termwise_magnitude,
        operation_count=operation_count,
    )
    old_result_only_bound = (
        choice.absolute_tolerance
        + choice.relative_tolerance * result_scale
        + functional._termwise_roundoff(choice, result_scale, operation_count)
    )
    cancellation_residual = 1.0e-7
    assert old_result_only_bound < cancellation_residual <= registered_bound

    small_roundoff, small_bound = functional._entrywise_bound(
        choice,
        result_scale=1.0,
        termwise_magnitude=2.0,
        operation_count=1,
    )
    evidence = abc.Vituri2024FunctionalReplayAnchorCheck(
        fock_entrywise_residual=cancellation_residual,
        fock_entrywise_result_scale=result_scale,
        fock_entrywise_termwise_magnitude=termwise_magnitude,
        fock_entrywise_operation_count=operation_count,
        fock_entrywise_roundoff_contribution=roundoff,
        fock_entrywise_registered_bound=registered_bound,
        interaction_entrywise_residual=0.0,
        interaction_entrywise_result_scale=1.0,
        interaction_entrywise_termwise_magnitude=2.0,
        interaction_entrywise_operation_count=1,
        interaction_entrywise_roundoff_contribution=small_roundoff,
        interaction_entrywise_registered_bound=small_bound,
        scalar_energy_residual=0.0,
        scalar_energy_result_scale=1.0,
        scalar_energy_termwise_magnitude=2.0,
        scalar_energy_operation_count=1,
        scalar_energy_roundoff_contribution=small_roundoff,
        scalar_energy_registered_bound=small_bound,
    )
    assert evidence.fock_entrywise_roundoff_contribution == roundoff
    assert evidence.fock_entrywise_termwise_magnitude == termwise_magnitude


def test_functional_replay_rejects_provider_and_input_mutation() -> None:
    binding, contract, provider = _functional_case()
    provider.call_mutation = ("q0_policy_fingerprint", _SHA["9"])
    with pytest.raises(ValueError, match="metadata mutated"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)

    binding, contract, provider = _functional_case()
    provider.mutate_input_method = "evaluate_fock_derivative"
    with pytest.raises(ValueError, match="mutated an input"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)


def test_functional_replay_rejects_fixed_interaction_and_interaction_only_fock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, contract, provider = _functional_case()
    assert provider.replay_payload is not None
    fixed_interaction = provider.replay_payload.interaction_h.copy()
    original_scalar = provider.evaluate_scalar_energy

    def fixed_interaction_scalar(
        interaction_h: np.ndarray, h0: np.ndarray, density: np.ndarray
    ) -> float:
        return original_scalar(fixed_interaction, h0, density)

    monkeypatch.setattr(provider, "evaluate_scalar_energy", fixed_interaction_scalar)
    with pytest.raises(ValueError, match="scalar E->F record"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)

    binding, contract, provider = _functional_case()

    def interaction_only_fock(density: np.ndarray) -> np.ndarray:
        return np.asarray(_FUNCTIONAL_G * density, dtype=np.complex128)

    monkeypatch.setattr(provider, "evaluate_fock_derivative", interaction_only_fock)
    with pytest.raises(ValueError, match="anchor"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)


@pytest.mark.parametrize(
    "fault", ("no_cross_route", "copy_plus", "average", "conjugate_minus")
)
def test_functional_replay_rejects_signed_lane_repairs_or_inference(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    binding, contract, provider = _functional_case()
    direct = provider.evaluate_displaced_fock

    def faulty_direct(
        density: np.ndarray,
        displacement: np.ndarray,
        **kwargs: object,
    ) -> abc.Vituri2024DirectDisplacedFockResponse:
        typed = direct(density, displacement, **kwargs)  # type: ignore[arg-type]
        result = typed.response.copy()
        if fault == "no_cross_route":
            result = np.asarray(_FUNCTIONAL_GQ * displacement)
        elif fault == "copy_plus":
            result[1] = result[0]
        elif fault == "average":
            average = 0.5 * (result[0] + result[1])
            result[0] = average
            result[1] = average
        else:
            result[1] = result[0].conj()
        return abc.Vituri2024DirectDisplacedFockResponse(
            caller_nonce_sha256=typed.caller_nonce_sha256,
            response=result,
            dependency_trace=typed.dependency_trace,
        )

    monkeypatch.setattr(provider, "evaluate_displaced_fock", faulty_direct)
    with pytest.raises(
        ValueError,
        match="matrix F->dF record|inactive response lane|invalid-edge",
    ):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)


def test_functional_replay_rejects_wrap_invalid_and_best_step_cherry_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, contract, provider = _functional_case()
    direct = provider.evaluate_displaced_fock

    def wrap_invalid(
        density: np.ndarray,
        displacement: np.ndarray,
        **kwargs: object,
    ) -> abc.Vituri2024DirectDisplacedFockResponse:
        typed = direct(density, displacement, **kwargs)  # type: ignore[arg-type]
        result = typed.response.copy()
        if np.any(displacement != 0.0):
            target_maps = kwargs["target_maps"]
            assert isinstance(target_maps, np.ndarray)
            invalid_index = int(np.flatnonzero(target_maps[0] < 0)[0])
            result[0, 0, 0, invalid_index] = 1.0
        return abc.Vituri2024DirectDisplacedFockResponse(
            caller_nonce_sha256=typed.caller_nonce_sha256,
            response=result,
            dependency_trace=typed.dependency_trace,
        )

    monkeypatch.setattr(provider, "evaluate_displaced_fock", wrap_invalid)
    with pytest.raises(ValueError, match="invalid-edge"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)

    binding, contract, provider = _functional_case()
    direct = provider.evaluate_displaced_fock

    def one_good_step_only(
        density: np.ndarray,
        displacement: np.ndarray,
        **kwargs: object,
    ) -> abc.Vituri2024DirectDisplacedFockResponse:
        typed = direct(density, displacement, **kwargs)  # type: ignore[arg-type]
        result = typed.response + 0.2 * displacement * np.abs(displacement) ** 2
        return abc.Vituri2024DirectDisplacedFockResponse(
            caller_nonce_sha256=typed.caller_nonce_sha256,
            response=np.asarray(result, dtype=np.complex128),
            dependency_trace=typed.dependency_trace,
        )

    monkeypatch.setattr(provider, "evaluate_displaced_fock", one_good_step_only)
    with pytest.raises(ValueError, match="matrix F->dF record"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)


def test_functional_payload_and_contract_reject_altered_registration() -> None:
    spec = _complete()
    payload = _functional_payload(spec)
    with pytest.raises(ValueError, match="packed-pair Frobenius norm"):
        changed = _SIGNED_Q_PROBES.copy()
        changed[2] *= 2.0
        replace(payload, signed_q_probes=changed)

    bad_targets = _SIGNED_Q_CHARTS[0].target_maps.copy()
    bad_targets[0, 15] = 0
    with pytest.raises(ValueError, match="no-wrap/no-carry"):
        replace(_SIGNED_Q_CHARTS[0], target_maps=bad_targets)

    binding, contract, provider = _functional_case()
    with pytest.raises(ValueError, match="detached approval manifest"):
        replace(contract, fock_step_ladder=(1.0e-2, 1.0e-3, 1.0e-4))
    collision_direct = provider.finite_q_hessian_implementation_fingerprint
    collision_archive = abc.direct_builder_dependency_archive_fingerprint(
        source_commit=contract.source_commit,
        source_artifact_sha256=contract.source_artifact_sha256,
        direct_displaced_fock_implementation_fingerprint=collision_direct,
        interaction_builder_implementation_fingerprint=(
            contract.direct_interaction_builder_implementation_fingerprint
        ),
        full_fock_builder_implementation_fingerprint=(
            contract.direct_full_fock_builder_implementation_fingerprint
        ),
    )
    collision_provider = abc.functional_provider_fingerprint(
        base_provider_fingerprint=contract.provider_fingerprint,
        functional_replay_abi_fingerprint=contract.functional_replay_abi_fingerprint,
        functional_replay_payload_schema_fingerprint=(
            contract.functional_replay_payload_schema_fingerprint
        ),
        functional_probe_loader_implementation_fingerprint=(
            contract.functional_probe_loader_implementation_fingerprint
        ),
        direct_displaced_fock_implementation_fingerprint=collision_direct,
        direct_builder_dependency_archive_fingerprint=collision_archive,
    )
    collision_approval = abc.Vituri2024FunctionalReplayApproval(
        choice_fingerprint=contract.choice_fingerprint,
        verifier_implementation_schema_fingerprint=(
            contract.verifier_implementation_schema_fingerprint
        ),
        verifier_module_ast_manifest_sha256=(
            contract.verifier_module_ast_manifest_sha256
        ),
        functional_provider_fingerprint=collision_provider,
        source_commit=contract.source_commit,
        source_artifact_sha256=contract.source_artifact_sha256,
        spec_fingerprint=contract.spec_fingerprint,
        source_state_sha256=contract.source_state_sha256,
        expected_array_payload_manifest_sha256=(
            contract.expected_array_payload_manifest_sha256
        ),
        affine_anchor_inventory_sha256=contract.affine_anchor_inventory_sha256,
        q0_probe_inventory_sha256=contract.q0_probe_inventory_sha256,
        signed_q_probe_inventory_sha256=contract.signed_q_probe_inventory_sha256,
        q_probe_inventory_sha256=contract.q_probe_inventory_sha256,
        fock_step_ladder=contract.fock_step_ladder,
        hessian_step_ladder=contract.hessian_step_ladder,
        direct_displaced_fock_implementation_fingerprint=collision_direct,
        direct_builder_dependency_archive_fingerprint=collision_archive,
        provenance=contract.detached_approval_provenance,
    )
    contract_kwargs = {
        name: getattr(contract, name)
        for name in inspect.signature(abc.Vituri2024FunctionalReplayContract).parameters
    }
    contract_kwargs.update(
        functional_provider_fingerprint=collision_provider,
        direct_displaced_fock_implementation_fingerprint=collision_direct,
        direct_builder_dependency_archive_fingerprint=collision_archive,
        detached_approval_manifest_sha256=collision_approval.manifest_sha256,
    )
    collision_contract = abc.Vituri2024FunctionalReplayContract(**contract_kwargs)
    with pytest.raises(ValueError, match="fingerprint equals Hessian"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, collision_contract)
    with pytest.raises(ValueError, match="dependency archive"):
        replace(contract, direct_builder_dependency_archive_fingerprint=_SHA["4"])
    with pytest.raises(ValueError, match="detached approval manifest"):
        replace(contract, detached_approval_manifest_sha256=_SHA["4"])


def test_functional_replay_rejects_truthful_unchanged_metadata_hessian_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, contract, provider = _functional_case()

    def delegating_direct(
        density: np.ndarray,
        displacement: np.ndarray,
        **kwargs: object,
    ) -> abc.Vituri2024DirectDisplacedFockResponse:
        result = provider.evaluate_finite_q_hessian(
            displacement,
            q_probe_index=int(kwargs["q_probe_index"]),
            mesh_displacement=kwargs["mesh_displacement"],  # type: ignore[arg-type]
            cartesian_q=kwargs["cartesian_q"],  # type: ignore[arg-type]
            target_maps=kwargs["target_maps"],  # type: ignore[arg-type]
            reverse_edge_map=kwargs["reverse_edge_map"],  # type: ignore[arg-type]
        )
        target_maps = kwargs["target_maps"]
        reverse_map = kwargs["reverse_edge_map"]
        assert isinstance(target_maps, np.ndarray)
        assert isinstance(reverse_map, np.ndarray)
        trace = abc.Vituri2024DirectBuilderDependencyTrace(
            caller_nonce_sha256=str(kwargs["caller_nonce"]),
            q_probe_index=int(kwargs["q_probe_index"]),
            q_label=_SIGNED_Q_CHARTS[int(kwargs["q_probe_index"])].q_label,
            interaction_builder_implementation_fingerprint=(
                provider.direct_interaction_builder_implementation_fingerprint
            ),
            full_fock_builder_implementation_fingerprint=(
                provider.direct_full_fock_builder_implementation_fingerprint
            ),
            target_maps_sha256=abc.canonical_array_sha256(target_maps),
            reverse_edge_map_sha256=abc.canonical_array_sha256(reverse_map),
            target_map_read_count=target_maps.size,
            reverse_edge_map_read_count=reverse_map.size,
            interaction_builder_call_count=0,
            full_fock_builder_call_count=0,
            finite_q_hessian_call_count=1,
        )
        return abc.Vituri2024DirectDisplacedFockResponse(
            caller_nonce_sha256=str(kwargs["caller_nonce"]),
            response=result,
            dependency_trace=trace,
        )

    monkeypatch.setattr(provider, "evaluate_displaced_fock", delegating_direct)
    with pytest.raises(
        ValueError, match="finite-q-Hessian call count mismatch"
    ):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)
    assert provider.hessian_calls == 2  # one registered call plus delegated direct call
    assert provider.direct_displaced_fock_construction == (
        abc.DIRECT_DISPLACED_FOCK_CONSTRUCTION
    )


@pytest.mark.parametrize(
    "attribute",
    functional.FUNCTIONAL_REPLAY_PROVIDER_METADATA_FIELDS,
)
def test_functional_replay_snapshots_every_provider_metadata_field(
    attribute: str,
) -> None:
    binding, contract, provider = _functional_case()
    provider.functional_loader_mutation = (attribute, "mutated_during_probe_load")
    with pytest.raises(ValueError, match="metadata mutated during functional probe loader"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)


def test_functional_replay_rejects_altered_probe_payload_and_schema() -> None:
    binding, contract, provider = _functional_case()
    assert provider.functional_payload is not None
    changed_q0 = provider.functional_payload.q0_directions.copy()
    changed_q0[0] *= -1.0
    provider.functional_payload = replace(
        provider.functional_payload, q0_directions=changed_q0
    )
    with pytest.raises(ValueError, match="q0 probe inventory mismatch"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)

    with pytest.raises(ValueError, match="schema fingerprint mismatch"):
        replace(
            provider.functional_payload,
            functional_replay_payload_schema_fingerprint=_SHA["9"],
        )
    with pytest.raises(ValueError, match="derived functional provider"):
        replace(contract, functional_provider_fingerprint=_SHA["9"])


def test_synthetic_direct_and_hessian_consume_two_chart_routing_inputs() -> None:
    _, _, provider = _functional_case()
    assert provider.replay_payload is not None
    probe = _SIGNED_Q_PROBES[0]
    chart = _SIGNED_Q_CHARTS[0]
    nonce = _SHA["4"]
    direct = provider.evaluate_displaced_fock(
        provider.replay_payload.projector,
        probe,
        q_probe_index=chart.q_probe_index,
        mesh_displacement=chart.mesh_displacement,
        cartesian_q=chart.cartesian_q,
        target_maps=chart.target_maps,
        reverse_edge_map=chart.reverse_edge_map,
        caller_nonce=nonce,
    )
    changed_density = provider.replay_payload.projector.copy()
    changed_density[0, 0, 0] += 0.5
    changed = provider.evaluate_displaced_fock(
        changed_density,
        probe,
        q_probe_index=chart.q_probe_index,
        mesh_displacement=chart.mesh_displacement,
        cartesian_q=chart.cartesian_q,
        target_maps=chart.target_maps,
        reverse_edge_map=chart.reverse_edge_map,
        caller_nonce=_SHA["5"],
    )
    assert not np.array_equal(direct.response, changed.response)
    assert not np.array_equal(direct.response, probe)

    bad_target = chart.target_maps.copy()
    bad_target[0, 0] = -1
    bad_reverse = chart.reverse_edge_map.copy()
    bad_reverse[0, 0] = -1
    for method in (
        provider.evaluate_finite_q_hessian,
        provider.evaluate_displaced_fock,
    ):
        common: dict[str, object] = {
            "q_probe_index": chart.q_probe_index,
            "mesh_displacement": chart.mesh_displacement,
            "cartesian_q": chart.cartesian_q,
            "target_maps": bad_target,
            "reverse_edge_map": chart.reverse_edge_map,
        }
        if method == provider.evaluate_displaced_fock:
            common["caller_nonce"] = _SHA["6"]
            arguments = (provider.replay_payload.projector, probe)
        else:
            arguments = (probe,)
        with pytest.raises(ValueError, match="target maps"):
            method(*arguments, **common)  # type: ignore[arg-type]
        common["target_maps"] = chart.target_maps
        common["reverse_edge_map"] = bad_reverse
        with pytest.raises(ValueError, match="reverse map"):
            method(*arguments, **common)  # type: ignore[arg-type]

    direct_kwargs = {
        "q_probe_index": chart.q_probe_index,
        "mesh_displacement": chart.mesh_displacement,
        "cartesian_q": chart.cartesian_q,
        "target_maps": chart.target_maps,
        "reverse_edge_map": chart.reverse_edge_map,
        "caller_nonce": _SHA["7"],
    }
    for name, altered, message in (
        ("q_probe_index", 1, "mesh displacement"),
        ("mesh_displacement", np.asarray((0, 1), dtype=np.int64), "mesh displacement"),
        ("cartesian_q", chart.cartesian_q + 0.01, "Cartesian q"),
    ):
        kwargs = dict(direct_kwargs)
        kwargs[name] = altered
        with pytest.raises((ValueError, AssertionError), match=message):
            provider.evaluate_displaced_fock(
                provider.replay_payload.projector, probe, **kwargs  # type: ignore[arg-type]
            )


def test_weak_matrix_channel_instability_is_not_hidden_by_global_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, contract, provider = _functional_case()
    original_hessian = provider.evaluate_finite_q_hessian
    original_direct = provider.evaluate_displaced_fock
    weak_scale = 4.0e-10

    def weak_hessian(perturbation: np.ndarray, **kwargs: object) -> np.ndarray:
        result = original_hessian(perturbation, **kwargs)  # type: ignore[arg-type]
        if int(kwargs["q_probe_index"]) == 1:
            result = weak_scale * result
        return result

    def weak_unstable_direct(
        density: np.ndarray,
        displacement: np.ndarray,
        **kwargs: object,
    ) -> abc.Vituri2024DirectDisplacedFockResponse:
        typed = original_direct(density, displacement, **kwargs)  # type: ignore[arg-type]
        result = typed.response.copy()
        if int(kwargs["q_probe_index"]) == 1:
            step_norm = float(np.sqrt(np.sum(np.abs(displacement) ** 2)))
            result *= weak_scale * (1.0 + 0.5 * step_norm / 1.0e-3)
        return abc.Vituri2024DirectDisplacedFockResponse(
            caller_nonce_sha256=typed.caller_nonce_sha256,
            response=result,
            dependency_trace=typed.dependency_trace,
        )

    monkeypatch.setattr(provider, "evaluate_finite_q_hessian", weak_hessian)
    monkeypatch.setattr(provider, "evaluate_displaced_fock", weak_unstable_direct)
    with pytest.raises(ValueError, match="local slope stability gate failed"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)


@pytest.mark.parametrize("pairing_fault", ("conjugating_dot", "transpose"))
def test_functional_replay_rejects_pairing_canaries_in_scalar_records_before_q(
    monkeypatch: pytest.MonkeyPatch, pairing_fault: str
) -> None:
    if pairing_fault == "conjugating_dot":
        def faulty_pairing(left: np.ndarray, right: np.ndarray, nk: int) -> float:
            return float(np.real(np.vdot(left, right)) / nk)
    else:
        def faulty_pairing(left: np.ndarray, right: np.ndarray, nk: int) -> float:
            return float(
                np.real(np.sum(np.swapaxes(left, 0, 1) * right)) / nk
            )

    monkeypatch.setattr(functional, "_pairing", faulty_pairing)
    binding, contract, provider = _functional_case()
    with pytest.raises(ValueError, match="scalar E->F record"):
        abc.replay_vituri2024_half_metal_hf_functional(binding, contract)
    assert provider.hessian_calls == 0
    assert provider.direct_displaced_calls == 0


def test_fd_evidence_rejects_crossed_normalizations_and_convention_drift() -> None:
    geometry, ensemble = _geometry(), _ensemble()
    shared = _shared(geometry, ensemble)
    with pytest.raises(ValueError, match="normalization/norm contradicts"):
        _fd(
            "fock_first_derivative",
            shared.scalar_energy.implementation_fingerprint,
            shared.fock_derivative.implementation_fingerprint,
            geometry,
            ensemble,
            perturbation_normalization=abc.FINITE_Q_HESSIAN_NORMALIZATION,
        )
    with pytest.raises(ValueError, match="normalization/norm contradicts"):
        _fd(
            "finite_q_hessian",
            shared.fock_derivative.implementation_fingerprint,
            shared.finite_q_hessian.implementation_fingerprint,
            geometry,
            ensemble,
            perturbation_normalization=abc.FOCK_FIRST_DERIVATIVE_NORMALIZATION,
        )
    with pytest.raises(ValueError, match="shared-functional conventions"):
        _fd(
            "fock_first_derivative",
            shared.scalar_energy.implementation_fingerprint,
            shared.fock_derivative.implementation_fingerprint,
            geometry,
            ensemble,
            fock_output="interaction_only",
        )


def test_functional_success_objects_are_factory_token_gated() -> None:
    with pytest.raises(TypeError, match="factory token"):
        abc.Vituri2024FunctionalReplayStatus(_factory_token=object())
    receipt = abc.replay_vituri2024_half_metal_hf_functional(*_functional_case()[:2])
    receipt_kwargs = {
        name: getattr(receipt, name)
        for name in inspect.signature(
            abc.Vituri2024FunctionalReplayReceipt
        ).parameters
        if name != "_factory_token"
    }
    with pytest.raises(TypeError, match="_factory_token"):
        abc.Vituri2024FunctionalReplayReceipt(**receipt_kwargs)  # type: ignore[arg-type]
    annotations = inspect.get_annotations(
        functional.Vituri2024FunctionalReplayProviderProtocol.evaluate_displaced_fock
    )
    assert annotations["density"] == "ComplexArray"
    assert annotations["signed_q_displacement"] == "ComplexArray"
    assert annotations["return"] == "Vituri2024DirectDisplacedFockResponse"
    for name in functional.__all__:
        assert getattr(abc, name) is getattr(functional, name)


# Uninterrupted SCF replay fixtures.  The trajectory oracle below implements
# the two affine fixed-point steps directly and never calls core/replay code.
_SCF_ARCHIVE_LOADER_FP = _fp({"scf": "archive_loader_v1"})
_SCF_STATE_BUILDER_FP = _fp({"scf": "state_builder_v1"})
_SCF_PROBLEM_BUILDER_FP = _fp({"scf": "problem_builder_v1"})
_SCF_ADAPTER_SCHEMA_FP = _fp({"scf": "adapter_schema_v1"})
_SCF_OBSERVABLES_EMPTY_FP = _fp({})
_SCF_DIAGNOSTICS_EMPTY_FP = _fp({})


class _SyntheticSCFState:
    def __init__(self, h0: np.ndarray, precision: float, *, copy_arrays: bool = True) -> None:
        copy = np.array if copy_arrays else np.asarray
        self.h0 = copy(h0, dtype=np.complex128)
        self.density = np.zeros_like(self.h0)
        self.hamiltonian = np.zeros_like(self.h0)
        self.energies = np.zeros((self.h0.shape[0], self.h0.shape[2]), dtype=np.float64)
        self.mu = 0.0
        self.precision = precision
        self.diagnostics: dict[str, float] = {}

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


def _manual_oda_lambda(
    density: np.ndarray,
    h0: np.ndarray,
    interaction_h: np.ndarray,
    delta_density: np.ndarray,
    delta_h: np.ndarray,
) -> float:
    nk = density.shape[2]
    a = float(np.real(np.sum(delta_density * delta_h)) / nk)
    b = float(
        np.real(
            np.sum(delta_density * h0)
            + 0.5 * np.sum(delta_density * interaction_h)
            + 0.5 * np.sum(density * delta_h)
        )
        / nk
    )
    if abs(a) < 1.0e-15:
        return 1.0 if b < 0.0 else 0.0
    lambda0 = -b / a
    if a > 0.0:
        if lambda0 <= 0.0:
            return 0.0
        if lambda0 < 1.0:
            return float(lambda0)
        return 1.0
    if lambda0 <= 0.5:
        return 1.0
    return 0.0


def _manual_norm(updated: np.ndarray, previous: np.ndarray) -> float:
    denominator = float(np.linalg.norm(updated))
    numerator = float(np.linalg.norm(previous - updated))
    return 0.0 if denominator < 1.0e-15 and numerator < 1.0e-15 else numerator / denominator


def _scf_targets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        seed.seed_label: (
            np.roll(_REPLAY_ARRAYS["projector"], index, axis=2).copy(),
            np.roll(_REPLAY_ARRAYS["energies"], index, axis=1).copy(),
        )
        for index, seed in enumerate(_seeds())
    }


def _scf_branch_records(
    energies: tuple[float, float, float] = (-2.0, -1.9, -1.8),
    *,
    convergence_rule: str = "raw",
    second_exit_reason: str = "converged",
) -> tuple[abc.Vituri2024BranchEnergyReceipt, ...]:
    return tuple(
        _branch(
            seed,
            energy,
            convergence_rule=convergence_rule,
            attested_exit_reason=(
                second_exit_reason if seed.seed_label == "spin_minus" else "converged"
            ),
            iterations=(
                500
                if seed.seed_label == "spin_minus" and second_exit_reason == "max_iter"
                else 2
            ),
            terminal_norm_raw=(
                1.0e-4
                if seed.seed_label == "spin_minus" and second_exit_reason != "converged"
                else 0.0
            ),
            terminal_norm_mixed=(
                1.0e-4
                if seed.seed_label == "spin_minus" and second_exit_reason != "converged"
                else 0.0
            ),
            terminal_norm_selected=(
                1.0e-4
                if seed.seed_label == "spin_minus" and second_exit_reason != "converged"
                else 0.0
            ),
            terminal_oda_lambda=(
                1.0e-3
                if seed.seed_label == "spin_minus" and second_exit_reason == "max_iter"
                else 0.0
            ),
            final_replay_raw_metric=0.0,
        )
        for seed, energy in zip(_seeds(), energies)
    )


def _scf_spec(
    branch_energies: tuple[float, float, float] = (-2.0, -1.9, -1.8),
    *,
    convergence_rule: str = "raw",
    second_exit_reason: str = "converged",
) -> abc.Vituri2024HalfMetalHFSpec:
    geometry, ensemble, policy = (
        _geometry(),
        _ensemble(),
        _scf(convergence_rule=convergence_rule),
    )
    shared = _shared(geometry, ensemble)
    records = _scf_branch_records(
        branch_energies,
        convergence_rule=convergence_rule,
        second_exit_reason=second_exit_reason,
    )
    table_bytes = _branch_table_bytes(records)
    source = _source(
        geometry,
        ensemble,
        policy,
        shared,
        branch_records=records,
        branch_energy_table_sha256=hashlib.sha256(table_bytes).hexdigest(),
        selected_branch_energy_ev=branch_energies[0],
        minimum_compared_branch_energy_ev=min(branch_energies),
        branch_energy_residual_ev=abs(branch_energies[0] - min(branch_energies)),
        final_replay_raw_metric=0.0,
    )
    return abc.Vituri2024HalfMetalHFSpec(
        geometry=geometry,
        ensemble=ensemble,
        scf_policy=policy,
        shared_functional=shared,
        attested_source=source,
    )


def _manual_energy(
    interaction_h: np.ndarray,
    h0: np.ndarray,
    density: np.ndarray,
    offset: float,
) -> float:
    return float(
        offset
        + np.real(
            np.sum(h0 * density) + 0.5 * np.sum(interaction_h * density)
        )
        / density.shape[2]
    )


def _manual_trajectory(
    seed: abc.Vituri2024SCFSeedReceipt,
    target: np.ndarray,
    energies: np.ndarray,
    branch_energy: float,
) -> abc.Vituri2024SCFSeedTrajectoryArchive:
    h0 = _REPLAY_ARRAYS["h0"].copy()
    zeros_matrix = np.zeros_like(h0)
    zeros_energy = np.zeros_like(energies)
    pre = abc.Vituri2024SCFStateSnapshot(
        h0=h0,
        density=zeros_matrix,
        hamiltonian=zeros_matrix,
        energies=zeros_energy,
        mu=0.0,
        precision=1.0e-9,
        diagnostics_manifest_sha256=_SCF_DIAGNOSTICS_EMPTY_FP,
    )
    density0 = 0.5 * target
    post = abc.Vituri2024SCFStateSnapshot(
        h0=h0,
        density=density0,
        hamiltonian=zeros_matrix,
        energies=zeros_energy,
        mu=0.0,
        precision=1.0e-9,
        diagnostics_manifest_sha256=_SCF_DIAGNOSTICS_EMPTY_FP,
    )
    final_interaction = _FUNCTIONAL_G * target
    raw_final_energy = float(
        np.real(
            np.sum(h0 * target) + 0.5 * np.sum(final_interaction * target)
        )
        / target.shape[2]
    )
    offset = branch_energy - raw_final_energy

    interaction1 = _FUNCTIONAL_G * density0
    delta1 = target - density0
    delta_h1 = _FUNCTIONAL_G * delta1
    lambda1 = _manual_oda_lambda(density0, h0, interaction1, delta1, delta_h1)
    assert lambda1 == 1.0
    mixed1 = lambda1 * target + (1.0 - lambda1) * density0
    total1 = h0 + interaction1
    step1_energy = _manual_energy(interaction1, h0, density0, offset)
    step1 = abc.Vituri2024SCFStepArchive(
        iteration=1,
        previous_density=density0,
        interaction_h=interaction1,
        total_hamiltonian=total1,
        raw_density=target,
        raw_energies=energies,
        raw_mu=-0.02,
        density_update_observables_sha256=_SCF_OBSERVABLES_EMPTY_FP,
        mixed_density=mixed1,
        state_density=mixed1,
        state_hamiltonian=total1,
        state_energies=energies,
        state_mu=-0.02,
        state_diagnostics_manifest_sha256=_fp(
            {"hf_energy": step1_energy, "oda_parameter": lambda1, "iterations": 1.0}
        ),
        delta_interaction_h=delta_h1,
        oda_lambda=lambda1,
        norm_raw=_manual_norm(target, density0),
        norm_mixed=_manual_norm(mixed1, density0),
        norm_selected=_manual_norm(target, density0),
        energy=step1_energy,
        interaction_h_from_cache=False,
    )

    interaction2 = interaction1 + lambda1 * delta_h1
    delta2 = target - mixed1
    delta_h2 = _FUNCTIONAL_G * delta2
    lambda2 = _manual_oda_lambda(mixed1, h0, interaction2, delta2, delta_h2)
    assert lambda2 == 0.0
    mixed2 = lambda2 * target + (1.0 - lambda2) * mixed1
    total2 = h0 + interaction2
    step2_energy = _manual_energy(interaction2, h0, mixed1, offset)
    step2 = abc.Vituri2024SCFStepArchive(
        iteration=2,
        previous_density=mixed1,
        interaction_h=interaction2,
        total_hamiltonian=total2,
        raw_density=target,
        raw_energies=energies,
        raw_mu=-0.02,
        density_update_observables_sha256=_SCF_OBSERVABLES_EMPTY_FP,
        mixed_density=mixed2,
        state_density=mixed2,
        state_hamiltonian=total2,
        state_energies=energies,
        state_mu=-0.02,
        state_diagnostics_manifest_sha256=_fp(
            {"hf_energy": step2_energy, "oda_parameter": lambda2, "iterations": 2.0}
        ),
        delta_interaction_h=delta_h2,
        oda_lambda=lambda2,
        norm_raw=_manual_norm(target, mixed1),
        norm_mixed=_manual_norm(mixed2, mixed1),
        norm_selected=_manual_norm(target, mixed1),
        energy=step2_energy,
        interaction_h_from_cache=True,
    )
    final_energy = _manual_energy(interaction2, h0, mixed2, offset)
    final_raw_norm = _manual_norm(target, mixed2)
    final = abc.Vituri2024SCFFinalRecomputationArchive(
        h0=h0,
        state_density=mixed2,
        effective_interaction_h=interaction2,
        total_hamiltonian=total2,
        raw_density=target,
        energies=energies,
        mu=-0.02,
        energy=final_energy,
        raw_norm=final_raw_norm,
        density_update_observables_sha256=_SCF_OBSERVABLES_EMPTY_FP,
        state_diagnostics_manifest_sha256=_fp(
            {
                "hf_energy": final_energy,
                "oda_parameter": lambda2,
                "iterations": 2.0,
                "final_raw_norm": final_raw_norm,
            }
        ),
    )
    transfer = (
        abc.Vituri2024SCFTransferSourceReceipt(
            "lower_density",
            f"{seed.seed_label}_lower",
            _COMMIT,
            _fp({"seed": seed.seed_label, "side": "lower", "artifact": True}),
            _fp({"seed": seed.seed_label, "side": "lower", "state": True}),
        ),
        abc.Vituri2024SCFTransferSourceReceipt(
            "higher_density",
            f"{seed.seed_label}_higher",
            _COMMIT,
            _fp({"seed": seed.seed_label, "side": "higher", "artifact": True}),
            _fp({"seed": seed.seed_label, "side": "higher", "state": True}),
        ),
    )
    callback_sequence = (
        "initializer",
        "interaction_builder",
        "energy_functional",
        "density_builder",
        "oda_delta_interaction_builder",
        "energy_functional",
        "density_builder",
        "oda_delta_interaction_builder",
        "density_builder",
        "energy_functional",
    )
    return abc.Vituri2024SCFSeedTrajectoryArchive(
        seed=seed,
        transfer_source_receipts=transfer,
        pre_init=pre,
        post_init=post,
        steps=(step1, step2),
        final_recomputation=final,
        callback_sequence=callback_sequence,
        exit_reason="converged",
        converged=True,
        iterations=2,
    )


def _scf_metadata(spec: abc.Vituri2024HalfMetalHFSpec) -> dict[str, str]:
    assert spec.shared_functional is not None
    dependency = abc.scf_dependency_archive_fingerprint(
        source_commit=_COMMIT,
        source_artifact_sha256=_SOURCE_ARTIFACT,
        state_builder_implementation_fingerprint=_SCF_STATE_BUILDER_FP,
        problem_builder_implementation_fingerprint=_SCF_PROBLEM_BUILDER_FP,
        scf_adapter_schema_fingerprint=_SCF_ADAPTER_SCHEMA_FP,
        scf_adapter_abi_fingerprint=abc.SCF_REPLAY_ADAPTER_ABI_FINGERPRINT,
    )
    functional_provider = abc.functional_provider_fingerprint(
        base_provider_fingerprint=_PROVIDER_FINGERPRINT,
        functional_replay_abi_fingerprint=abc.FUNCTIONAL_REPLAY_ABI_FINGERPRINT,
        functional_replay_payload_schema_fingerprint=(
            abc.FUNCTIONAL_REPLAY_PAYLOAD_SCHEMA_FINGERPRINT
        ),
        functional_probe_loader_implementation_fingerprint=(
            _FUNCTIONAL_PROBE_LOADER_IMPLEMENTATION_FINGERPRINT
        ),
        direct_displaced_fock_implementation_fingerprint=(
            _DIRECT_DISPLACED_FOCK_IMPLEMENTATION_FINGERPRINT
        ),
        direct_builder_dependency_archive_fingerprint=(
            _DIRECT_BUILDER_DEPENDENCY_ARCHIVE_FINGERPRINT
        ),
    )
    return {
        "state_builder_implementation_fingerprint": _SCF_STATE_BUILDER_FP,
        "problem_builder_implementation_fingerprint": _SCF_PROBLEM_BUILDER_FP,
        "scf_adapter_schema_fingerprint": _SCF_ADAPTER_SCHEMA_FP,
        "scf_adapter_abi_fingerprint": abc.SCF_REPLAY_ADAPTER_ABI_FINGERPRINT,
        "scf_dependency_archive_fingerprint": dependency,
        "scf_provider_fingerprint": abc.scf_provider_fingerprint(
            functional_provider_fingerprint=functional_provider,
            state_builder_implementation_fingerprint=_SCF_STATE_BUILDER_FP,
            problem_builder_implementation_fingerprint=_SCF_PROBLEM_BUILDER_FP,
            scf_adapter_schema_fingerprint=_SCF_ADAPTER_SCHEMA_FP,
            scf_adapter_abi_fingerprint=abc.SCF_REPLAY_ADAPTER_ABI_FINGERPRINT,
            scf_dependency_archive_fingerprint=dependency,
        ),
    }


def _manual_scf_archive(
    spec: abc.Vituri2024HalfMetalHFSpec,
) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
    assert spec.attested_source is not None
    records = spec.attested_source.branch_records
    targets = _scf_targets()
    trajectories = tuple(
        _manual_trajectory(
            row.seed,
            targets[row.seed.seed_label][0],
            targets[row.seed.seed_label][1],
            row.canonical_energy_ev,
        )
        for row in records
    )
    selected = abc.Vituri2024SCFSelectedSource(
        selected_branch_label="spin_plus",
        source_state_sha256=spec.attested_source.source_state_sha256,
        h0=_REPLAY_ARRAYS["h0"],
        effective_interaction_h=_REPLAY_ARRAYS["interaction_h"],
        fock=_REPLAY_ARRAYS["fock"],
        projector=_REPLAY_ARRAYS["projector"],
        energies=_REPLAY_ARRAYS["energies"],
        mu=-0.02,
        registered_hashes=(
            ("h0", _ARRAY_HASHES["h0"]),
            ("effective_interaction_h", _ARRAY_HASHES["interaction_h"]),
            ("fock", _ARRAY_HASHES["fock"]),
            ("projector", _ARRAY_HASHES["projector"]),
            ("energies", _ARRAY_HASHES["energies"]),
        ),
    )
    table_bytes = _branch_table_bytes(records)
    archive_authority_fingerprint = abc.scf_archive_authority_fingerprint(
        source_artifact_sha256=_SOURCE_ARTIFACT,
        archive_loader_implementation_fingerprint=_SCF_ARCHIVE_LOADER_FP,
    )
    return abc.Vituri2024ImmutableHistoricalSCFArchive(
        archive_authority_fingerprint=archive_authority_fingerprint,
        source_commit=_COMMIT,
        source_artifact_sha256=_SOURCE_ARTIFACT,
        spec_fingerprint=spec.fingerprint,
        archive_loader_implementation_fingerprint=_SCF_ARCHIVE_LOADER_FP,
        archive_schema_fingerprint=abc.SCF_REPLAY_ARCHIVE_SCHEMA_FINGERPRINT,
        generation_phase=abc.SCF_REPLAY_ARCHIVE_GENERATION_PHASE,
        seed_trajectories=trajectories,
        branch_records=records,
        original_branch_table_bytes=table_bytes,
        original_branch_table_sha256=hashlib.sha256(table_bytes).hexdigest(),
        selected_branch_label="spin_plus",
        selected_source=selected,
    )


class _SCFArchiveAuthority:
    def __init__(self, archive: abc.Vituri2024ImmutableHistoricalSCFArchive) -> None:
        self.archive_authority_fingerprint = archive.archive_authority_fingerprint
        self.source_artifact_sha256 = archive.source_artifact_sha256
        self.archive_loader_implementation_fingerprint = (
            archive.archive_loader_implementation_fingerprint
        )
        self.archive_schema_fingerprint = archive.archive_schema_fingerprint
        self.archive = archive
        self.calls: list[str] = []
        self.metadata_mutation: tuple[str, object] | None = None

    def load_immutable_scf_archive(
        self, source_artifact_sha256: str
    ) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
        self.calls.append("load_archive")
        assert source_artifact_sha256 == _SOURCE_ARTIFACT
        if self.metadata_mutation is not None:
            setattr(self, *self.metadata_mutation)
        return self.archive


class _SCFProvider(_Provider):
    def __init__(
        self,
        spec: abc.Vituri2024HalfMetalHFSpec,
    ) -> None:
        super().__init__(spec, _payload(spec, _array_copy()))
        for name, value in _scf_metadata(spec).items():
            setattr(self, name, value)
        self.scf_calls: list[str] = []
        self.live_interaction_corruption = False
        self.mutate_diagnostics = False
        self.non_none_step_callback = False
        self.non_none_final_callback = False
        self.uninspectable_callback_role: str | None = None
        self.share_live_states = False
        self._first_live_state: _SyntheticSCFState | None = None
        self._targets = _scf_targets()
        self._branch_energy = {
            row.seed.seed_label: row.canonical_energy_ev
            for row in spec.attested_source.branch_records
        }
        assert spec.scf_policy is not None
        self._convergence_rule = spec.scf_policy.convergence_rule

    def build_fresh_scf_state(
        self, seed: abc.Vituri2024SCFSeedReceipt
    ) -> _SyntheticSCFState:
        self.scf_calls.append(f"build_state:{seed.seed_label}")
        state = _SyntheticSCFState(_REPLAY_ARRAYS["h0"], 1.0e-9)
        if self.share_live_states and self._first_live_state is not None:
            state.h0 = self._first_live_state.h0
            state.density = self._first_live_state.density
        if self._first_live_state is None:
            self._first_live_state = state
        return state

    def build_scf_problem(
        self, state: object, seed: abc.Vituri2024SCFSeedReceipt
    ) -> HartreeFockProblem:
        self.scf_calls.append(f"build_problem:{seed.seed_label}")
        target, energies = self._targets[seed.seed_label]
        branch_energy = self._branch_energy[seed.seed_label]
        final_interaction = _FUNCTIONAL_G * target
        raw_final = float(
            np.real(
                np.sum(_REPLAY_ARRAYS["h0"] * target)
                + 0.5 * np.sum(final_interaction * target)
            )
            / target.shape[2]
        )
        self.energy_offset = branch_energy - raw_final

        def initializer(actual_state: _SyntheticSCFState, *, init_mode: str, seed: int) -> None:
            assert init_mode == next(item.init_mode for item in _seeds() if item.seed_value == seed)
            actual_state.density[:, :, :] = 0.5 * target

        def interaction_builder(density: np.ndarray) -> np.ndarray:
            result = _FUNCTIONAL_G * density
            if self.live_interaction_corruption:
                result = result.copy()
                result[0, 0, 0] += 1.0e-4
            return np.asarray(result, dtype=np.complex128)

        def density_builder(hamiltonian: np.ndarray) -> DensityUpdateResult:
            return DensityUpdateResult(
                density=target.copy(),
                energies=energies.copy(),
                mu=-0.02,
            )

        def energy_functional(
            interaction_h: np.ndarray, h0: np.ndarray, density: np.ndarray
        ) -> float:
            value = self.evaluate_scalar_energy(interaction_h, h0, density)
            if self.mutate_diagnostics:
                state.diagnostics["provider_callback_mutation"] = 1.0
            return value

        def delta_interaction_builder(delta_density: np.ndarray) -> np.ndarray:
            return np.asarray(_FUNCTIONAL_G * delta_density, dtype=np.complex128)

        callbacks = {
            "initializer": initializer,
            "interaction_builder": interaction_builder,
            "density_builder": density_builder,
            "energy_functional": energy_functional,
            "oda_delta_interaction_builder": delta_interaction_builder,
        }
        if self.uninspectable_callback_role is not None:
            callbacks[self.uninspectable_callback_role] = len
        step_callback = (lambda actual_state, step: None) if self.non_none_step_callback else None
        final_state_callback = (
            (lambda actual_state, update: None) if self.non_none_final_callback else None
        )
        return HartreeFockProblem(
            initializer=callbacks["initializer"],  # type: ignore[arg-type]
            kernel=HartreeFockKernel(
                interaction_builder=callbacks["interaction_builder"],  # type: ignore[arg-type]
                density_builder=callbacks["density_builder"],  # type: ignore[arg-type]
                energy_functional=callbacks["energy_functional"],  # type: ignore[arg-type]
                oda_delta_interaction_builder=callbacks[
                    "oda_delta_interaction_builder"
                ],  # type: ignore[arg-type]
                step_callback=step_callback,
                final_state_callback=final_state_callback,
                convergence_rule=self._convergence_rule,
            ),
        )


def _scf_case(
    *,
    branch_energies: tuple[float, float, float] = (-2.0, -1.9, -1.8),
    convergence_rule: str = "raw",
    second_exit_reason: str = "converged",
    archive_transform: object | None = None,
) -> tuple[
    abc.Vituri2024HalfMetalHFProviderBinding,
    _SCFArchiveAuthority,
    abc.Vituri2024SCFReplayApproval,
    _SCFProvider,
    abc.Vituri2024ImmutableHistoricalSCFArchive,
]:
    spec = _scf_spec(
        branch_energies,
        convergence_rule=convergence_rule,
        second_exit_reason=second_exit_reason,
    )
    archive = _manual_scf_archive(spec)
    if archive_transform is not None:
        archive = archive_transform(archive)  # type: ignore[operator]
    review_provider = _SCFProvider(spec)
    review_seed = spec.scf_policy.seed_records[0]
    review_state = review_provider.build_fresh_scf_state(review_seed)
    reviewed_callback_manifests = abc.vituri2024_scf_problem_callback_manifests(
        review_provider.build_scf_problem(review_state, review_seed)
    )
    provider = _SCFProvider(spec)
    authority = _SCFArchiveAuthority(archive)
    binding = abc.Vituri2024HalfMetalHFProviderBinding(spec, provider)
    approval = abc.make_vituri2024_scf_replay_approval(
        binding,
        authority,
        expected_archive_manifest_sha256=abc.scf_archive_manifest_sha256(archive),
        expected_branch_table_sha256=archive.original_branch_table_sha256,
        problem_callback_manifests=reviewed_callback_manifests,
        provenance="Detached synthetic manual-oracle approval; not real Vituri execution.",
    )
    assert authority.calls == []
    assert provider.scf_calls == []
    return binding, authority, approval, provider, archive


def _replace_first_step(
    archive: abc.Vituri2024ImmutableHistoricalSCFArchive,
    field_name: str,
) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
    trajectory = archive.seed_trajectories[0]
    step = trajectory.steps[0]
    value = getattr(step, field_name)
    if isinstance(value, np.ndarray):
        changed = value.copy()
        changed.flat[0] += 1.0e-5
    elif field_name == "interaction_h_from_cache":
        changed = not value
    elif field_name == "density_update_observables_sha256":
        changed = _SHA["9"]
    else:
        changed = float(value) + 1.0e-5
    changed_step = replace(step, **{field_name: changed})
    changed_trajectory = replace(
        trajectory,
        steps=(changed_step, *trajectory.steps[1:]),
    )
    return replace(
        archive,
        seed_trajectories=(changed_trajectory, *archive.seed_trajectories[1:]),
    )


def test_scf_replay_runs_all_three_seeds_through_actual_core_and_closes_status() -> None:
    binding, authority, approval, provider, archive = _scf_case()

    receipt = abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)

    assert receipt.seed_order == ("spin_plus", "spin_minus", "random_broken")
    assert receipt.replayed_branch_energies_ev == pytest.approx((-2.0, -1.9, -1.8))
    assert receipt.tolerance_degenerate_minimum_labels == ("spin_plus",)
    assert receipt.selected_branch_residual_ev <= 1.0e-12
    assert receipt.archive_manifest_sha256 == abc.scf_archive_manifest_sha256(archive)
    assert receipt.core_provenance_mode == approval.core_provenance_mode
    assert (
        receipt.core_baseline_commit_authority
        == approval.core_baseline_commit_authority
        == scf_replay._CORE_BASELINE_COMMIT_AUTHORITY
    )
    assert receipt.archive_authority_outer_call_sequence == ("load_immutable_scf_archive",)
    assert receipt.effective_tolerances == abc.default_vituri2024_scf_replay_tolerances()
    assert authority.calls == ["load_archive"]
    assert provider.scf_calls == [
        "build_state:spin_plus",
        "build_problem:spin_plus",
        "build_state:spin_minus",
        "build_problem:spin_minus",
        "build_state:random_broken",
        "build_problem:random_broken",
    ]
    assert all(item.steps[1].interaction_h_from_cache for item in archive.seed_trajectories)
    archived_array = archive.seed_trajectories[0].steps[0].previous_density
    assert not archived_array.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        archived_array[0, 0, 0] = 1.0
    status = receipt.status
    assert (
        receipt.evidence_model
        == status.evidence_model
        == "trusted_live_provider_distinct_archive_object"
    )
    assert not any(
        (
            receipt.archive_data_independence_verified,
            receipt.hostile_provider_resistance_verified,
            receipt.live_builder_dependency_state_independently_pinned,
            status.archive_data_independence_verified,
            status.hostile_provider_resistance_verified,
            status.live_builder_dependency_state_independently_pinned,
        )
    )
    assert status.uninterrupted_registered_seed_trajectories_replayed
    assert status.all_attested_seed_branches_replayed
    assert status.branch_table_replayed
    assert status.selected_final_source_reproduced
    assert not any(
        (
            status.global_ground_state_verified,
            status.transfer_learning_physics_verified,
            status.checkpoint_snapshot_hash_verified,
            status.atomic_checkpoint_publication_verified,
            status.exact_restart_verified,
            status.interrupted_vs_uninterrupted_trajectory_equivalent,
            status.scientific_execution_verified,
            status.paper_reproduction_verified,
        )
    )
    audit = receipt.restart_capability_audit
    assert not audit.public_continuation_api_available
    assert not audit.cached_interaction_h_publicly_exposed
    assert not audit.rng_state_captured
    assert not audit.callback_state_captured
    assert not audit.exact_restart_verified
    assert "cached_interaction_h" in audit.blocker
    receipt_kwargs = {
        name: getattr(receipt, name)
        for name in inspect.signature(abc.Vituri2024SCFReplayReceipt).parameters
        if name != "_factory_token"
    }
    with pytest.raises((TypeError, ValueError), match="factory|_factory_token"):
        abc.Vituri2024SCFReplayReceipt(**receipt_kwargs)  # type: ignore[arg-type]
    assert provider.scalar_energy_calls == 9


def test_scf_replay_rejects_alias_and_same_fingerprint_authorities() -> None:
    binding, authority, approval, provider, _ = _scf_case()
    with pytest.raises(TypeError, match="distinct objects"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, provider, approval)  # type: ignore[arg-type]
    assert authority.calls == []
    assert provider.scf_calls == []

    authority.archive_authority_fingerprint = provider.scf_provider_fingerprint
    with pytest.raises(ValueError, match="same authority fingerprint"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)
    assert authority.calls == []
    assert provider.scf_calls == []

def test_scf_replay_same_class_same_code_archive_copy_canary_stays_limited() -> None:
    binding, authority, approval, provider, archive = _scf_case()
    assert type(provider) is _SCFProvider

    # This approved same-class/same-code provider receives detached archive
    # copies through unmanifested instance state.  The trusted-provider replay
    # can pass and must report the limitation rather than claim rejection.
    provider._targets = {
        item.seed.seed_label: (
            np.array(item.final_recomputation.raw_density, copy=True),
            np.array(item.final_recomputation.energies, copy=True),
        )
        for item in archive.seed_trajectories
    }
    assert all(
        not np.shares_memory(target, item.final_recomputation.raw_density)
        and not np.shares_memory(energies, item.final_recomputation.energies)
        for (target, energies), item in zip(
            provider._targets.values(), archive.seed_trajectories
        )
    )
    assert (
        abc.vituri2024_scf_callable_manifest(
            "build_fresh_scf_state", provider.build_fresh_scf_state
        ),
        abc.vituri2024_scf_callable_manifest(
            "build_scf_problem", provider.build_scf_problem
        ),
    ) == approval.live_builder_manifests

    receipt = abc.replay_vituri2024_half_metal_hf_scf(
        binding, authority, approval
    )

    assert receipt.status.uninterrupted_registered_seed_trajectories_replayed
    assert receipt.status.selected_final_source_reproduced
    assert (
        receipt.evidence_model
        == receipt.status.evidence_model
        == "trusted_live_provider_distinct_archive_object"
    )
    assert not any(
        (
            receipt.archive_data_independence_verified,
            receipt.hostile_provider_resistance_verified,
            receipt.live_builder_dependency_state_independently_pinned,
            receipt.status.archive_data_independence_verified,
            receipt.status.hostile_provider_resistance_verified,
            receipt.status.live_builder_dependency_state_independently_pinned,
            receipt.status.exact_restart_verified,
            receipt.status.scientific_execution_verified,
            receipt.status.paper_reproduction_verified,
        )
    )

def test_scf_replay_uses_registered_mixed_convergence_rule() -> None:
    binding, authority, approval, _, archive = _scf_case(convergence_rule="mixed")
    receipt = abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)
    assert receipt.seed_order == tuple(
        item.seed.seed_label for item in archive.seed_trajectories
    )
    assert all(item.steps[-1].norm_mixed == 0.0 for item in archive.seed_trajectories)


@pytest.mark.parametrize(
    "field_name",
    (
        "previous_density",
        "interaction_h",
        "total_hamiltonian",
        "raw_density",
        "raw_energies",
        "raw_mu",
        "density_update_observables_sha256",
        "mixed_density",
        "state_density",
        "state_hamiltonian",
        "state_energies",
        "state_mu",
        "delta_interaction_h",
        "oda_lambda",
        "norm_raw",
        "norm_mixed",
        "norm_selected",
        "energy",
        "interaction_h_from_cache",
    ),
)
def test_scf_replay_rejects_every_archived_step_field_corruption(
    field_name: str,
) -> None:
    binding, authority, approval, provider, _ = _scf_case(
        archive_transform=lambda archive: _replace_first_step(archive, field_name)
    )
    with pytest.raises(ValueError, match="SCF|step|cache|observables|tolerance"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)
    assert authority.calls == ["load_archive"]
    assert provider.scf_calls[:2] == [
        "build_state:spin_plus",
        "build_problem:spin_plus",
    ]


def test_scf_replay_rejects_step_iteration_none_delta_and_callback_sequence() -> None:
    archive = _manual_scf_archive(_scf_spec())
    trajectory = archive.seed_trajectories[0]
    with pytest.raises(ValueError, match="consecutive"):
        replace(
            trajectory,
            steps=(replace(trajectory.steps[0], iteration=2), *trajectory.steps[1:]),
        )

    def none_delta(
        source: abc.Vituri2024ImmutableHistoricalSCFArchive,
    ) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
        first = source.seed_trajectories[0]
        changed_step = replace(first.steps[0], delta_interaction_h=None)
        return replace(
            source,
            seed_trajectories=(
                replace(first, steps=(changed_step, *first.steps[1:])),
                *source.seed_trajectories[1:],
            ),
        )

    binding, authority, approval, _, _ = _scf_case(archive_transform=none_delta)
    with pytest.raises(ValueError, match="None/present"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)

    def callback_drift(
        source: abc.Vituri2024ImmutableHistoricalSCFArchive,
    ) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
        first = source.seed_trajectories[0]
        return replace(
            source,
            seed_trajectories=(
                replace(first, callback_sequence=first.callback_sequence + ("step_callback",)),
                *source.seed_trajectories[1:],
            ),
        )

    binding, authority, approval, _, _ = _scf_case(archive_transform=callback_drift)
    with pytest.raises(ValueError, match="callback sequence"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)


def test_scf_replay_rejects_pre_and_post_initializer_snapshot_corruption() -> None:
    for snapshot_name in ("pre_init", "post_init"):
        def corrupt(
            source: abc.Vituri2024ImmutableHistoricalSCFArchive,
            name: str = snapshot_name,
        ) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
            first = source.seed_trajectories[0]
            snapshot = getattr(first, name)
            density = snapshot.density.copy()
            density[0, 0, 0] += 1.0e-5
            return replace(
                source,
                seed_trajectories=(
                    replace(first, **{name: replace(snapshot, density=density)}),
                    *source.seed_trajectories[1:],
                ),
            )

        binding, authority, approval, _, _ = _scf_case(archive_transform=corrupt)
        with pytest.raises(ValueError, match=rf"{snapshot_name}\.density"):
            abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)


def test_scf_replay_distinguishes_final_recomputation_from_last_step() -> None:
    def corrupt_final(
        archive: abc.Vituri2024ImmutableHistoricalSCFArchive,
    ) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
        trajectory = archive.seed_trajectories[0]
        changed = trajectory.final_recomputation.raw_density.copy()
        changed[0, 0, 0] += 1.0e-5
        return replace(
            archive,
            seed_trajectories=(
                replace(
                    trajectory,
                    final_recomputation=replace(
                        trajectory.final_recomputation,
                        raw_density=changed,
                    ),
                ),
                *archive.seed_trajectories[1:],
            ),
        )

    binding, authority, approval, _, _ = _scf_case(archive_transform=corrupt_final)
    with pytest.raises(ValueError, match="final.raw_density"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)


def test_scf_replay_rejects_seed_reorder_missing_transfer_and_nonconverged_drift() -> None:
    def reorder(
        archive: abc.Vituri2024ImmutableHistoricalSCFArchive,
    ) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
        order = (1, 0, 2)
        return replace(
            archive,
            seed_trajectories=tuple(archive.seed_trajectories[index] for index in order),
            branch_records=tuple(archive.branch_records[index] for index in order),
        )

    binding, authority, approval, provider, _ = _scf_case(archive_transform=reorder)
    with pytest.raises(ValueError, match="seed order"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)
    assert authority.calls == ["load_archive"]
    assert provider.scf_calls == []

    def missing(
        archive: abc.Vituri2024ImmutableHistoricalSCFArchive,
    ) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
        return replace(
            archive,
            seed_trajectories=archive.seed_trajectories[:-1],
            branch_records=archive.branch_records[:-1],
        )

    binding, authority, approval, provider, _ = _scf_case(archive_transform=missing)
    with pytest.raises(ValueError, match="seed order|inventory|branch rows"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)
    assert authority.calls == ["load_archive"]
    assert provider.scf_calls == []

    trajectory = _manual_scf_archive(_scf_spec()).seed_trajectories[0]
    with pytest.raises((TypeError, ValueError), match="two typed|two-sided"):
        replace(trajectory, transfer_source_receipts=trajectory.transfer_source_receipts[:1])

    def false_nonconverged(
        archive: abc.Vituri2024ImmutableHistoricalSCFArchive,
    ) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
        first = replace(
            archive.seed_trajectories[0], exit_reason="max_iter", converged=False
        )
        return replace(
            archive,
            seed_trajectories=(first, *archive.seed_trajectories[1:]),
        )

    binding, authority, approval, _, _ = _scf_case(archive_transform=false_nonconverged)
    with pytest.raises(ValueError, match="iteration/exit/converged"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)


def test_scf_replay_rejects_callback_provider_input_storage_and_generation_drift() -> None:
    binding, authority, approval, provider, _ = _scf_case()
    authority.metadata_mutation = ("archive_authority_fingerprint", _SHA["9"])
    with pytest.raises(ValueError, match="archive-authority metadata mutated"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)
    assert authority.calls == ["load_archive"]
    assert provider.scf_calls == []

    binding, authority, approval, provider, _ = _scf_case()
    provider.uninspectable_callback_role = "energy_functional"
    with pytest.raises(RuntimeError, match="not inspectable"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)

    binding, authority, approval, provider, _ = _scf_case()
    provider.call_mutation = ("provider_fingerprint", _SHA["9"])
    with pytest.raises(ValueError, match="metadata mutated"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)

    binding, authority, approval, provider, _ = _scf_case()
    provider.mutate_input_method = "evaluate_scalar_energy"
    with pytest.raises(ValueError, match="mutated verifier input"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)

    binding, authority, approval, provider, _ = _scf_case()
    provider.share_live_states = True
    with pytest.raises(ValueError, match="live SCF states share storage"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)

    def share_archive_seed_storage(
        archive: abc.Vituri2024ImmutableHistoricalSCFArchive,
    ) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
        shared = archive.seed_trajectories[0].steps[0].previous_density
        object.__setattr__(
            archive.seed_trajectories[1].steps[0], "previous_density", shared
        )
        return archive

    binding, authority, approval, provider, _ = _scf_case(
        archive_transform=share_archive_seed_storage
    )
    with pytest.raises(ValueError, match="shared array storage"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)
    assert authority.calls == ["load_archive"]
    assert provider.scf_calls == []

    binding, authority, approval, provider, archive = _scf_case()
    object.__setattr__(archive, "generation_phase", "generated_after_builders")
    approval = replace(
        approval,
        expected_archive_manifest_sha256=abc.scf_archive_manifest_sha256(archive),
    )
    with pytest.raises(ValueError, match="generated after"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)


def test_scf_replay_rejects_live_vs_archive_and_selected_source_corruption() -> None:
    binding, authority, approval, provider, _ = _scf_case()
    provider.live_interaction_corruption = True
    with pytest.raises(ValueError, match=r"step.interaction_h|tolerance|state\.diagnostics"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)

    def corrupt_selected(
        archive: abc.Vituri2024ImmutableHistoricalSCFArchive,
    ) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
        selected = archive.selected_source
        changed = selected.h0.copy()
        changed[0, 0, 0] += 1.0e-5
        changed_hashes = list(selected.registered_hashes)
        changed_hashes[0] = ("h0", abc.canonical_array_sha256(changed))
        return replace(
            archive,
            selected_source=replace(
                selected,
                h0=changed,
                registered_hashes=tuple(changed_hashes),
            ),
        )

    binding, authority, approval, provider, _ = _scf_case(archive_transform=corrupt_selected)
    with pytest.raises(ValueError, match="selected-source hashes"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)
    assert authority.calls == ["load_archive"]
    assert provider.scf_calls == []


def test_scf_replay_reports_tolerance_ties_without_unique_ground_state_claim() -> None:
    binding, authority, approval, _, _ = _scf_case(branch_energies=(-2.0, -2.0 + 1.0e-9, -1.8))
    receipt = abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)
    assert receipt.tolerance_degenerate_minimum_labels == ("spin_plus", "spin_minus")
    assert not receipt.unique_ground_state_claimed
    assert not receipt.status.global_ground_state_verified


def test_scf_replay_locked_tolerances_and_selected_hash_flags_are_not_weakenable() -> None:
    assert tuple(inspect.signature(abc.default_vituri2024_scf_replay_tolerances).parameters) == ()
    with pytest.raises(TypeError):
        abc.default_vituri2024_scf_replay_tolerances(absolute=1.0)  # type: ignore[call-arg]
    assert "tolerances" not in inspect.signature(
        abc.make_vituri2024_scf_replay_approval
    ).parameters

    _, _, approval, _, _ = _scf_case()
    huge = list(approval.tolerances)
    huge[0] = replace(huge[0], absolute=1.0e100, relative=1.0e100)
    with pytest.raises(ValueError, match="locked v1 tolerance"):
        replace(approval, tolerances=tuple(huge))

    disabled = list(approval.tolerances)
    selected_index = next(
        index
        for index, item in enumerate(disabled)
        if item.field_name == "selected.h0"
    )
    disabled[selected_index] = replace(
        disabled[selected_index], require_canonical_hash=False
    )
    with pytest.raises(ValueError, match="locked v1 tolerance"):
        replace(approval, tolerances=tuple(disabled))

def test_scf_replay_rejects_provider_step_final_callbacks_and_diagnostics_mutation() -> None:
    for flag in ("non_none_step_callback", "non_none_final_callback"):
        binding, authority, approval, provider, _ = _scf_case()
        setattr(provider, flag, True)
        with pytest.raises(ValueError, match="callback changed from None|requires provider"):
            abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)

    binding, authority, approval, provider, _ = _scf_case()
    provider.mutate_diagnostics = True
    with pytest.raises(ValueError, match=r"state\.diagnostics manifest"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)

def test_scf_branch_row_exit_mismatch_is_rejected_in_both_directions() -> None:
    # Row says converged while detached trajectory says nonconverged.
    def trajectory_nonconverged(
        archive: abc.Vituri2024ImmutableHistoricalSCFArchive,
    ) -> abc.Vituri2024ImmutableHistoricalSCFArchive:
        first = replace(
            archive.seed_trajectories[0], exit_reason="max_iter", converged=False
        )
        return replace(archive, seed_trajectories=(first, *archive.seed_trajectories[1:]))

    binding, authority, approval, _, _ = _scf_case(
        archive_transform=trajectory_nonconverged
    )
    with pytest.raises(ValueError, match="iteration/exit/converged"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)

    # Row says nonconverged while detached trajectory and actual run converge.
    binding, authority, approval, _, _ = _scf_case(second_exit_reason="max_iter")
    with pytest.raises(ValueError, match="branch exit reason"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)

def _write_scf_core_source_export(export_root: Path) -> None:
    repository_root = scf_replay._repository_root()
    for relative_path in scf_replay._CORE_SOURCE_EXPECTATIONS:
        destination = export_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repository_root / relative_path).read_bytes())


def test_scf_no_git_source_export_accepts_only_pinned_core_and_binds_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    export_root = tmp_path / "immutable-source-export"
    _write_scf_core_source_export(export_root)
    monkeypatch.setattr(scf_replay, "_repository_root", lambda: export_root)

    def unexpected_git(*args: object, **kwargs: object) -> object:
        raise AssertionError("no-git source-export mode must not invoke git")

    monkeypatch.setattr(subprocess, "run", unexpected_git)
    provenance = scf_replay.verified_vituri2024_core_provenance()

    assert provenance.provenance_mode == "pinned_hash_verified_source_export"
    assert provenance.baseline_commit == scf_replay.VITURI2024_SCF_BASELINE_COMMIT
    assert (
        provenance.baseline_commit_authority
        == scf_replay._CORE_BASELINE_COMMIT_AUTHORITY
    )
    assert not any(
        (
            provenance.repository_checks_available,
            provenance.repository_ancestry_verified,
            provenance.repository_head_core_verified,
            provenance.repository_index_core_verified,
            provenance.repository_worktree_core_verified,
        )
    )
    assert {
        item.relative_path: (
            item.source_bytes_sha256,
            item.canonical_ast_sha256,
        )
        for item in provenance.source_manifests
    } == dict(scf_replay._CORE_SOURCE_EXPECTATIONS)
    assert {
        item.symbol: (
            item.module,
            item.qualname,
            item.signature,
            item.canonical_function_ast_sha256,
        )
        for item in provenance.callable_identities
    } == dict(scf_replay._CORE_CALLABLE_EXPECTATIONS)
    assert (
        provenance.package_version,
        provenance.python_version,
        provenance.python_implementation,
        provenance.numpy_version,
    ) == (
        scf_replay._package_version(),
        scf_replay.platform.python_version(),
        scf_replay.platform.python_implementation(),
        np.__version__,
    )

    binding, authority, approval, _, _ = _scf_case()
    receipt = abc.replay_vituri2024_half_metal_hf_scf(
        binding, authority, approval
    )
    assert (
        approval.core_provenance_mode
        == receipt.core_provenance_mode
        == "pinned_hash_verified_source_export"
    )
    assert (
        approval.core_baseline_commit_authority
        == receipt.core_baseline_commit_authority
        == scf_replay._CORE_BASELINE_COMMIT_AUTHORITY
    )
    assert approval.core_provenance_fingerprint == receipt.core_provenance_fingerprint

    with pytest.raises(ValueError, match="core provenance mode"):
        abc.replay_vituri2024_half_metal_hf_scf(
            binding,
            authority,
            replace(
                approval,
                core_provenance_mode="git_ancestor_head_index_worktree_verified",
            ),
        )
    with pytest.raises(ValueError, match="package/Python/NumPy runtime drift"):
        abc.replay_vituri2024_half_metal_hf_scf(
            binding,
            authority,
            replace(approval, package_version=approval.package_version + "+drift"),
        )


def test_scf_no_git_source_export_rejects_altered_bound_core(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    export_root = tmp_path / "altered-source-export"
    _write_scf_core_source_export(export_root)
    target_path = export_root / "src/mean_field/core/hf/engine.py"
    target_path.write_bytes(
        target_path.read_bytes()
        + b"\n\ndef _unauthorized_exported_core_change():\n    return True\n"
    )
    monkeypatch.setattr(scf_replay, "_repository_root", lambda: export_root)

    with pytest.raises(RuntimeError, match="source-export core source manifest mismatch"):
        scf_replay.verified_vituri2024_core_provenance()


def _install_scf_descendant_git_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    changed_location: str | None = None,
    ancestor: bool = True,
) -> tuple[str, list[tuple[str, ...]]]:
    root = scf_replay._repository_root()
    descendant_head = "f" * 40
    assert descendant_head != scf_replay.VITURI2024_SCF_BASELINE_COMMIT
    sources = {
        relative_path: (root / relative_path).read_bytes()
        for relative_path in scf_replay._CORE_SOURCE_EXPECTATIONS
    }
    target_path = next(iter(sources))
    changed_source = sources[target_path] + b"\ndef _descendant_core_drift():\n    return 1\n"
    calls: list[tuple[str, ...]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool = False,
    ) -> subprocess.CompletedProcess[object]:
        assert cwd == root
        assert capture_output
        args = tuple(command[1:])
        calls.append(args)
        if args == ("rev-parse", "--verify", "HEAD^{commit}"):
            return subprocess.CompletedProcess(
                command, 0, stdout=descendant_head + "\n", stderr=""
            )
        if args == (
            "merge-base",
            "--is-ancestor",
            scf_replay.VITURI2024_SCF_BASELINE_COMMIT,
            descendant_head,
        ):
            return subprocess.CompletedProcess(
                command, 0 if ancestor else 1, stdout="", stderr=""
            )
        if len(args) == 2 and args[0] == "show":
            revision, relative_path = args[1].split(":", 1)
            if revision == scf_replay.VITURI2024_SCF_BASELINE_COMMIT:
                location = "baseline"
            elif revision == descendant_head:
                location = "HEAD"
            elif revision == "":
                location = "index"
            else:  # pragma: no cover - assertion aid for command drift
                raise AssertionError(f"unexpected git object: {args[1]}")
            raw = sources[relative_path]
            if location == changed_location and relative_path == target_path:
                raw = changed_source
            return subprocess.CompletedProcess(command, 0, stdout=raw, stderr=b"")
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    if changed_location == "working-tree":
        original_read_bytes = Path.read_bytes

        def changed_worktree(path: Path) -> bytes:
            if path == root / target_path:
                return changed_source
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", changed_worktree)
    return descendant_head, calls


def test_scf_git_core_baseline_allows_descendant_head_with_unchanged_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descendant_head, calls = _install_scf_descendant_git_probe(monkeypatch)

    scf_replay._verify_git_core_baseline(scf_replay._repository_root())

    assert (
        "merge-base",
        "--is-ancestor",
        scf_replay.VITURI2024_SCF_BASELINE_COMMIT,
        descendant_head,
    ) in calls
    for relative_path in scf_replay._CORE_SOURCE_EXPECTATIONS:
        assert (
            "show",
            f"{scf_replay.VITURI2024_SCF_BASELINE_COMMIT}:{relative_path}",
        ) in calls
        assert ("show", f"{descendant_head}:{relative_path}") in calls
        assert ("show", f":{relative_path}") in calls


def test_scf_git_core_baseline_requires_baseline_ancestor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_scf_descendant_git_probe(monkeypatch, ancestor=False)
    with pytest.raises(RuntimeError, match="baseline commit is not an ancestor"):
        scf_replay._verify_git_core_baseline(scf_replay._repository_root())


@pytest.mark.parametrize("changed_location", ("HEAD", "index", "working-tree"))
def test_scf_git_core_baseline_rejects_core_change_at_every_layer(
    monkeypatch: pytest.MonkeyPatch,
    changed_location: str,
) -> None:
    _install_scf_descendant_git_probe(
        monkeypatch,
        changed_location=changed_location,
    )
    with pytest.raises(
        RuntimeError,
        match=rf"{changed_location} core source manifest mismatch",
    ):
        scf_replay._verify_git_core_baseline(scf_replay._repository_root())


def test_scf_replay_core_and_verifier_provenance_fail_before_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, authority, approval, provider, _ = _scf_case()
    original = Path.read_bytes

    def semantically_dirty_core(path: Path) -> bytes:
        raw = original(path)
        if path.name == "engine.py" and path.parent.name == "hf":
            return raw.replace(b"exit_reason = \"max_iter\"", b"exit_reason = \"oda_stall\"", 1)
        return raw

    monkeypatch.setattr(Path, "read_bytes", semantically_dirty_core)
    with pytest.raises(RuntimeError, match="core source manifest mismatch"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)
    assert provider.scf_calls == []
    monkeypatch.undo()

    binding, authority, approval, provider, _ = _scf_case()
    monkeypatch.setattr(
        scf_replay._hf_problem,
        "run_hartree_fock_problem",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(RuntimeError, match="runtime callable identity drift"):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)
    assert provider.scf_calls == []


def test_scf_replay_problem_iteration_alias_fails_before_authority_or_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, authority, approval, provider, _ = _scf_case()
    assert (
        scf_replay._hf_problem.run_hartree_fock_iterations
        is scf_replay._hf_engine.run_hartree_fock_iterations
    )
    monkeypatch.setattr(
        scf_replay._hf_problem,
        "run_hartree_fock_iterations",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(
        RuntimeError,
        match="problem-module SCF iteration alias runtime identity",
    ):
        abc.replay_vituri2024_half_metal_hf_scf(binding, authority, approval)

    assert authority.calls == []
    assert provider.scf_calls == []


def test_scf_replay_ast_archive_schema_exports_and_success_factory_guards() -> None:
    source = Path(scf_replay.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_archive_against_source"
    )
    target.body.insert(0, ast.Pass())
    ast.fix_missing_locations(tree)
    mutated = ast.unparse(tree)
    assert (
        abc.scf_replay_module_ast_manifest_sha256(mutated)
        != abc.scf_replay_module_ast_manifest_sha256(source)
    )
    assert "current_replay_receipt" not in {
        item.name
        for item in inspect.signature(
            abc.Vituri2024ImmutableHistoricalSCFArchive
        ).parameters.values()
    }
    assert "transcript" not in inspect.signature(
        abc.Vituri2024ImmutableHistoricalSCFArchive
    ).parameters
    with pytest.raises((TypeError, ValueError), match="factory"):
        abc.Vituri2024SCFReplayStatus(_factory_token=object())
    with pytest.raises(TypeError):
        abc.Vituri2024RestartCapabilityAudit(exact_restart_verified=True)  # type: ignore[call-arg]
    for name in scf_replay.__all__:
        assert getattr(abc, name) is getattr(scf_replay, name)
