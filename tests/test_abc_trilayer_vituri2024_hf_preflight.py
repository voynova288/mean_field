"""Lightweight contract tests for the receipt-only Vituri HF preflight."""
from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path
import struct

import numpy as np
import pytest

import mean_field.systems.abc_trilayer as abc
import mean_field.systems.abc_trilayer.vituri2024_hf_preflight as preflight
import mean_field.systems.abc_trilayer.vituri2024_hf_replay as replay

_SHA = {str(index): str(index) * 64 for index in range(10)}
_COMMIT = "a" * 40
_SOURCE_ARTIFACT = "b" * 64
_PROVIDER_FINGERPRINT = _SHA["8"]
_REPLAY_LOADER_IMPLEMENTATION_FINGERPRINT = _SHA["5"]

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
    interaction_h = np.zeros_like(fock)
    for flavor in range(4):
        interaction_h[flavor, flavor, :] = (flavor + 1) / 256.0
    h0 = fock - interaction_h
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
        "perturbation_inventory_sha256": _SHA["7"],
        "perturbation_normalization": (
            "unit_frobenius_norm_hermitian_projector_tangent"
        ),
        "matrix_norm": "frobenius",
        "q_probe_inventory_sha256": _SHA["8"] if kind == "finite_q_hessian" else None,
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
        "branch_energy_table_sha256": _SHA["7"],
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
) -> abc.Vituri2024HalfMetalHFSpec:
    hashes = _canonical_hashes(arrays)
    geometry_values: dict[str, object] = {
        "ordered_momentum_mesh_sha256": abc.canonical_array_sha256(arrays["mesh"])
    }
    if geometry_overrides:
        geometry_values.update(geometry_overrides)
    geometry = _geometry(**geometry_values)
    ensemble, scf = _ensemble(), _scf()
    shared = _shared(geometry, ensemble, array_hashes=hashes)
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
        self.loader_mutation: tuple[str, object] | None = None
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
        self.replay_payload = payload
        self.replay_loader_calls: list[str] = []

    def evaluate_scalar_energy(
        self,
        interaction_h: preflight.ComplexArray,
        h0: preflight.ComplexArray,
        density: preflight.ComplexArray,
    ) -> float:
        raise AssertionError("metadata attestation must not execute providers")

    def evaluate_fock_derivative(
        self, density: preflight.ComplexArray
    ) -> preflight.ComplexArray:
        raise AssertionError("metadata attestation must not execute providers")

    def evaluate_finite_q_hessian(
        self, perturbation: preflight.ComplexArray, *, q_probe_index: int
    ) -> preflight.ComplexArray:
        raise AssertionError("metadata attestation must not execute providers")

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
    assert shared.hessian_finite_difference.q_probe_inventory_sha256 == _SHA["8"]
    assert len(shared.hessian_finite_difference.finite_difference_step_ladder) == 3
    assert shared.hessian_finite_difference.matrix_norm == "frobenius"

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
    valley_arrays["occupations"][3, 19] = 1
    valley_arrays["projector"][3, 3, 19] = 1.0
    valley_arrays["energies"][3, 19] = -0.028
    valley_arrays["fock"][3, 3, 19] = -0.028
    valley_arrays["h0"][3, 3, 19] = (
        -0.028 - valley_arrays["interaction_h"][3, 3, 19]
    )
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
    binding, _, _ = _replay_case(spin_arrays)
    with pytest.raises(ValueError, match="opposite-spin hole-count mismatch"):
        abc.replay_vituri2024_half_metal_hf_arrays(binding)

    mu_arrays = _array_copy()
    mu_arrays["energies"][0, 0] = -0.015
    mu_arrays["fock"][0, 0, 0] = -0.015
    mu_arrays["h0"][0, 0, 0] = (
        -0.015 - mu_arrays["interaction_h"][0, 0, 0]
    )
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
