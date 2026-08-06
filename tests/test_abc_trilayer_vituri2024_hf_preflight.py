"""Lightweight contract tests for the receipt-only Vituri HF preflight."""
from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import mean_field.systems.abc_trilayer as abc
import mean_field.systems.abc_trilayer.vituri2024_hf_preflight as preflight

_SHA = {str(index): str(index) * 64 for index in range(10)}
_COMMIT = "a" * 40
_SOURCE_ARTIFACT = "b" * 64
_PROVIDER_FINGERPRINT = _SHA["8"]
_ARRAY_HASHES = {
    "orbitals": _SHA["1"],
    "energies": _SHA["2"],
    "occupations": _SHA["3"],
    "projector": _SHA["4"],
    "fock": _SHA["5"],
}


def _fp(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _area() -> abc.Vituri2024FiniteAreaReceipt:
    return abc.Vituri2024FiniteAreaReceipt(
        area_angstrom_squared=20_000.0,
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
        "array_layout": "core_state_k_then_internal_valley_then_spin",
        "ordered_momentum_mesh_sha256": _SHA["3"],
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
                "array_layout": values["array_layout"],
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


def _source_state_hash(
    geometry: abc.Vituri2024HFGeometryReceipt,
    ensemble: abc.Vituri2024HFEnsembleReceipt,
) -> str:
    return _fp(
        {
            "ordered_orbitals_sha256": _ARRAY_HASHES["orbitals"],
            "ordered_energies_sha256": _ARRAY_HASHES["energies"],
            "ordered_occupations_sha256": _ARRAY_HASHES["occupations"],
            "ordered_projector_sha256": _ARRAY_HASHES["projector"],
            "ordered_fock_sha256": _ARRAY_HASHES["fock"],
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
        "source_state_sha256": _source_state_hash(geometry, ensemble),
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
        ),
        "hessian_finite_difference": _fd(
            "finite_q_hessian",
            fock.implementation_fingerprint,
            hessian.implementation_fingerprint,
            geometry,
            ensemble,
        ),
        "authority_kind": "independent_provider_explicit",
        "provenance": "Synthetic one-source functional receipt.",
    }
    values.update(overrides)
    return abc.Vituri2024SharedFunctionalReceipt(**values)  # type: ignore[arg-type]


def _metallicity(
    geometry: abc.Vituri2024HFGeometryReceipt,
    ensemble: abc.Vituri2024HFEnsembleReceipt,
    **overrides: object,
) -> abc.Vituri2024MetallicityEvidenceReceipt:
    values: dict[str, object] = {
        "source_state_sha256": _source_state_hash(geometry, ensemble),
        "geometry_receipt_fingerprint": geometry.fingerprint,
        "ordered_momentum_mesh_sha256": geometry.ordered_momentum_mesh_sha256,
        "ordered_energies_sha256": _ARRAY_HASHES["energies"],
        "ordered_occupations_sha256": _ARRAY_HASHES["occupations"],
        "selected_spin": 1,
        "chemical_potential_ev": -0.02,
        "selected_spin_band_min_ev": -0.05,
        "selected_spin_band_max_ev": 0.01,
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
    **overrides: object,
) -> abc.Vituri2024ValleyPocketEvidenceReceipt:
    values: dict[str, object] = {
        "valley": valley,
        "source_state_sha256": _source_state_hash(geometry, ensemble),
        "geometry_receipt_fingerprint": geometry.fingerprint,
        "ordered_momentum_mesh_sha256": geometry.ordered_momentum_mesh_sha256,
        "ordered_occupations_sha256": _ARRAY_HASHES["occupations"],
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
    **overrides: object,
) -> abc.Vituri2024AttestedHalfMetalSourceReceipt:
    records = _branches(scf.convergence_rule)
    values: dict[str, object] = {
        "source_commit": _COMMIT,
        "source_artifact_sha256": _SOURCE_ARTIFACT,
        "provider_fingerprint": _PROVIDER_FINGERPRINT,
        "source_state_sha256": _source_state_hash(geometry, ensemble),
        "ordered_orbitals_sha256": _ARRAY_HASHES["orbitals"],
        "ordered_energies_sha256": _ARRAY_HASHES["energies"],
        "ordered_occupations_sha256": _ARRAY_HASHES["occupations"],
        "ordered_projector_sha256": _ARRAY_HASHES["projector"],
        "ordered_fock_sha256": _ARRAY_HASHES["fock"],
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
        "fock_projector_commutator_residual_ev": 1.0e-10,
        "stationarity_tolerance_ev": 1.0e-8,
        "projector_idempotency_residual": 1.0e-11,
        "projector_idempotency_tolerance": 1.0e-8,
        "projector_hermiticity_residual": 1.0e-12,
        "projector_hermiticity_tolerance": 1.0e-9,
        "fock_hermiticity_residual_ev": 1.0e-12,
        "fock_hermiticity_tolerance_ev": 1.0e-9,
        "aufbau_min_unoccupied_minus_max_occupied_ev": 1.0e-4,
        "aufbau_occupation_violation_ev": 0.0,
        "aufbau_tolerance_ev": 1.0e-9,
        "selected_spin": 1,
        "valley_plus_hole_count": 1,
        "valley_minus_hole_count": 1,
        "selected_spin_hole_count": 2,
        "opposite_spin_hole_count": 0,
        "metallicity_evidence": _metallicity(geometry, ensemble),
        "pocket_evidence": (
            _pocket(-1, geometry, ensemble),
            _pocket(1, geometry, ensemble),
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


class _Provider:
    def __init__(self, spec: abc.Vituri2024HalfMetalHFSpec) -> None:
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
    assert receipt.array_layout == "core_state_k_then_internal_valley_then_spin"
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
