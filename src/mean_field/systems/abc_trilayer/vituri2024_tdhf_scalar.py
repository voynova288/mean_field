"""Fail-closed Vituri-2024 scalar-readiness receipts.

Readiness binds the exact replay payload and predecessor chain to source-derived
C9 transitions and projected signed-A/B assembly.  It does not execute or
claim a scalar functional, scalar curvature, or static-Hessian authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, fields, is_dataclass
from hashlib import sha256
import json
from typing import Literal
import numpy as np

from mean_field.core.hf.tdhf_signed import (
    TDHFGenericSignedQSector,
    build_tdhf_signed_q_matrices,
    fingerprint_tdhf_matrix,
    fingerprint_tdhf_pairs,
    fingerprint_tdhf_sector,
)

from .vituri2024_hf_functional_replay import (
    AFFINE_ANCHOR_LABELS,
    FUNCTIONAL_REPLAY_SCOPE,
    Q0_PROBE_LABELS,
    Q_CHART_LABELS,
    SIGNED_Q_PROBE_LABELS,
    Vituri2024FunctionalReplayContract,
    Vituri2024FunctionalReplayReceipt,
    Vituri2024FunctionalReplayStatus,
    expected_array_payload_manifest_sha256,
)
from .vituri2024_hf_pocket_replay import (
    POCKET_REFINEMENT_EVIDENCE_MODEL,
    VITURI2024_POCKET_REFINEMENT_REPLAY_SCOPE,
    Vituri2024PocketRefinementPrerequisites,
    Vituri2024PocketRefinementReplayApproval,
    Vituri2024PocketRefinementReplayReceipt,
    Vituri2024PocketRefinementReplayStatus,
    canonical_half_metal_hf_replay_receipt_fingerprint,
    canonical_scf_replay_receipt_fingerprint,
)
from .vituri2024_hf_preflight import (
    INTERNAL_FLAVOR_ORDER,
    Vituri2024HalfMetalHFProviderBinding,
)
from .vituri2024_hf_replay import (
    Vituri2024HalfMetalHFReplayPayload,
    Vituri2024HalfMetalHFReplayReceipt,
    Vituri2024HalfMetalHFReplayStatus,
    canonical_array_sha256,
    canonical_orbital_order_sha256,
)
from .vituri2024_interaction import Vituri2024InteractionChoiceReceipt
from .vituri2024_rpa import Vituri2024FiniteAreaReceipt
from .vituri2024_hf_scf_replay import (
    VITURI2024_SCF_REPLAY_SCOPE,
    Vituri2024SCFReplayApproval,
    Vituri2024SCFReplayReceipt,
    Vituri2024SCFReplayStatus,
)
from .vituri2024_tdhf import (
    Vituri2024TDHFAssemblyContext,
    Vituri2024TDHFSignedQAssemblyReceipt,
    vituri2024_tdhf_interaction_fingerprint,
)

# Planner-facing compatibility name; the concrete type keeps its historical
# SignedQ spelling.
Vituri2024TDHFAssemblyReceipt = Vituri2024TDHFSignedQAssemblyReceipt

VITURI2024_SCALAR_READINESS_AUTHORITY = "projected_signed_ab"
VITURI2024_SCALAR_STATIC_HESSIAN_AUTHORITY = "not_established"
VITURI2024_SCALAR_READINESS_LOCKED_STRUCTURE_THRESHOLD = 1.0e-10
VITURI2024_SCALAR_BLOCKERS = (
    "no full finite-q exact-unitary scalar E[P] provider",
    "no complete real tangent basis or canonical d^2 scalar inventory",
    "finite area and kinematics provider remain caller-attested",
    "no real scalar-curvature artifact",
    "no refined-SCF stationarity or continuum-convergence certification",
    "no finite reciprocal-torus authority",
)
_ESTABLISHED_EVIDENCE = (
    "exact typed replay payload and array/functional/SCF/pocket predecessor chain are bound",
    "every ordered C9 transition is re-derived from exact payload mesh/flavor/energy/occupation arrays",
    "Delta1/area/interaction/kinematics assembly context is rebound without cross-lane drift",
    "C9 signed-q structure and pair/sector/matrix bytes are independently rechecked under the locked threshold",
)
_READINESS_FACTORY_TOKEN = object()

_ARRAY_REPLAY_STATUS = {
    "arrays_loaded": True,
    "array_hashes_verified": True,
    "source_structure_verified": True,
    "provider_methods_executed": ("load_half_metal_replay_payload",),
    "scf_trajectory_replayed": False,
    "branch_table_replayed": False,
    "pocket_refinement_replayed": False,
    "functional_chain_replayed": False,
    "scientific_execution_verified": False,
    "paper_reproduction_verified": False,
}
_FUNCTIONAL_REPLAY_STATUS = {
    "local_registered_functional_probes_replayed": True,
    "array_replay_verified": True,
    "all_local_registered_gates_passed": True,
    "approval_precedes_execution": True,
    "scope": FUNCTIONAL_REPLAY_SCOPE,
    "affine_anchor_count": len(AFFINE_ANCHOR_LABELS),
    "q0_probe_count": len(Q0_PROBE_LABELS),
    "signed_q_probe_count": len(SIGNED_Q_PROBE_LABELS),
    "q_chart_count": len(Q_CHART_LABELS),
    "global_functional_chain_verified": False,
    "scf_trajectory_replayed": False,
    "branch_table_replayed": False,
    "pocket_refinement_replayed": False,
    "scientific_execution_verified": False,
    "paper_reproduction_verified": False,
}
_SCF_REPLAY_EVIDENCE_MODEL = "trusted_live_provider_distinct_archive_object"
_SCF_REPLAY_STATUS = {
    "evidence_model": _SCF_REPLAY_EVIDENCE_MODEL,
    "archive_data_independence_verified": False,
    "hostile_provider_resistance_verified": False,
    "live_builder_dependency_state_independently_pinned": False,
    "uninterrupted_registered_seed_trajectories_replayed": True,
    "all_attested_seed_branches_replayed": True,
    "branch_table_replayed": True,
    "selected_final_source_reproduced": True,
    "global_ground_state_verified": False,
    "transfer_learning_physics_verified": False,
    "checkpoint_snapshot_hash_verified": False,
    "atomic_checkpoint_publication_verified": False,
    "exact_restart_verified": False,
    "interrupted_vs_uninterrupted_trajectory_equivalent": False,
    "scientific_execution_verified": False,
    "paper_reproduction_verified": False,
}
_POCKET_REPLAY_STATUS = {
    "evidence_model": POCKET_REFINEMENT_EVIDENCE_MODEL,
    "array_replay_prerequisite_bound": True,
    "scf_selected_source_prerequisite_bound": True,
    "detached_refinement_archive_loaded": True,
    "structured_refinement_mesh_registered": True,
    "live_frozen_selected_source_evaluated": True,
    "refined_occupations_recomputed": True,
    "bilateral_hole_topology_recomputed": True,
    "discrete_lifshitz_margin_recomputed": True,
    "pocket_refinement_replayed": True,
    "real_vituri_artifact_replayed": False,
    "archive_live_computational_independence_verified": False,
    "hostile_provider_resistance_verified": False,
    "hidden_live_dependency_state_excluded": False,
    "refined_scf_executed": False,
    "refined_fixed_density_resolved": False,
    "continuum_pocket_stability_verified": False,
    "refinement_convergence_verified": False,
    "global_ground_state_verified": False,
    "scientific_execution_verified": False,
    "paper_reproduction_verified": False,
    "tdhf_readiness_verified": False,
}


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    payload = (
        str(value.dtype).encode()
        + b"\0"
        + json.dumps(value.shape).encode()
        + b"\0"
        + value.view(np.uint8).tobytes()
    )
    return sha256(payload).hexdigest()


def _stable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": value.shape,
            "sha256": _array_sha256(value),
        }
    if is_dataclass(value):
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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _receipt_fingerprint(value: object) -> str:
    payload = json.dumps(_stable(value), sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def _validate_fingerprint(name: str, value: str) -> None:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 fingerprint")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} is not hexadecimal") from error


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Vituri scalar readiness {label} mismatch")

def _max_signed_structure_residual(structure: object) -> float:
    names = (
        "A_plus_hermitian",
        "A_minus_hermitian",
        "B_partner_transpose",
        "H_plus_hermitian",
        "H_minus_hermitian",
        "L_plus_pseudo_hermitian",
        "L_minus_pseudo_hermitian",
        "signed_liouvillian_covariance",
        "reverse_signed_liouvillian_covariance",
        "sewing_closure",
        "sewing_metric_anticovariance",
    )
    values = tuple(float(getattr(structure, name)) for name in names)
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Vituri scalar readiness structure residual is invalid")
    return max(values)


def _validate_exact_status(
    status: object,
    status_type: type[object],
    expected: dict[str, object],
    label: str,
) -> None:
    if type(status) is not status_type:
        raise TypeError(f"Vituri scalar readiness requires the exact {label} status type")
    field_inventory = tuple(item.name for item in fields(status))
    if field_inventory != tuple(expected):
        raise RuntimeError(
            f"Vituri scalar readiness {label} status field inventory drifted"
        )
    for name, expected_value in expected.items():
        actual = getattr(status, name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(
                f"Vituri scalar readiness {label} status field {name} mismatch"
            )


def _validate_replay_statuses(
    array_receipt: Vituri2024HalfMetalHFReplayReceipt,
    functional_contract: Vituri2024FunctionalReplayContract,
    functional_receipt: Vituri2024FunctionalReplayReceipt,
    scf_approval: Vituri2024SCFReplayApproval,
    scf_receipt: Vituri2024SCFReplayReceipt,
    pocket_approval: Vituri2024PocketRefinementReplayApproval,
    pocket_receipt: Vituri2024PocketRefinementReplayReceipt,
) -> None:
    _validate_exact_status(
        array_receipt.status,
        Vituri2024HalfMetalHFReplayStatus,
        _ARRAY_REPLAY_STATUS,
        "array replay",
    )
    _validate_exact_status(
        functional_receipt.status,
        Vituri2024FunctionalReplayStatus,
        _FUNCTIONAL_REPLAY_STATUS,
        "functional replay",
    )
    _validate_exact_status(
        scf_receipt.status,
        Vituri2024SCFReplayStatus,
        _SCF_REPLAY_STATUS,
        "SCF replay",
    )
    _validate_exact_status(
        pocket_receipt.status,
        Vituri2024PocketRefinementReplayStatus,
        _POCKET_REPLAY_STATUS,
        "pocket replay",
    )

    functional_status = functional_receipt.status
    functional_top_level = (
        functional_contract.scope,
        functional_receipt.scope,
        functional_status.scope,
        functional_contract.approval_precedes_execution,
        functional_receipt.approval_precedes_execution,
        functional_status.approval_precedes_execution,
        functional_receipt.affine_anchor_count,
        functional_status.affine_anchor_count,
        functional_receipt.q0_probe_count,
        functional_status.q0_probe_count,
        functional_receipt.signed_q_probe_count,
        functional_status.signed_q_probe_count,
        functional_receipt.q_chart_count,
        functional_status.q_chart_count,
    )
    expected_functional_top_level = (
        FUNCTIONAL_REPLAY_SCOPE,
        FUNCTIONAL_REPLAY_SCOPE,
        FUNCTIONAL_REPLAY_SCOPE,
        True,
        True,
        True,
        len(AFFINE_ANCHOR_LABELS),
        len(AFFINE_ANCHOR_LABELS),
        len(Q0_PROBE_LABELS),
        len(Q0_PROBE_LABELS),
        len(SIGNED_Q_PROBE_LABELS),
        len(SIGNED_Q_PROBE_LABELS),
        len(Q_CHART_LABELS),
        len(Q_CHART_LABELS),
    )
    _require_equal(
        functional_top_level,
        expected_functional_top_level,
        "functional status/receipt scope and count inventory",
    )
    # Every chart has three registered probe labels, while its mixed probe has
    # two independent signed-lane local gates rather than one.
    signed_q_local_gate_count = len(SIGNED_Q_PROBE_LABELS) + len(Q_CHART_LABELS)
    _require_equal(
        (
            len(functional_receipt.scalar_steps),
            len(functional_receipt.scalar_local_gates),
            len(functional_receipt.matrix_steps),
            len(functional_receipt.matrix_local_gates),
            len(functional_receipt.reciprocity),
        ),
        (
            len(AFFINE_ANCHOR_LABELS)
            * len(Q0_PROBE_LABELS)
            * len(functional_contract.fock_step_ladder),
            len(AFFINE_ANCHOR_LABELS) * len(Q0_PROBE_LABELS),
            signed_q_local_gate_count
            * len(functional_contract.hessian_step_ladder),
            signed_q_local_gate_count,
            len(Q_CHART_LABELS) * len(AFFINE_ANCHOR_LABELS),
        ),
        "functional exact evidence inventory counts",
    )

    scf_status = scf_receipt.status
    _require_equal(scf_approval.scope, VITURI2024_SCF_REPLAY_SCOPE, "SCF approval scope")
    _require_equal(
        (
            scf_receipt.evidence_model,
            scf_receipt.archive_data_independence_verified,
            scf_receipt.hostile_provider_resistance_verified,
            scf_receipt.live_builder_dependency_state_independently_pinned,
        ),
        (
            scf_status.evidence_model,
            scf_status.archive_data_independence_verified,
            scf_status.hostile_provider_resistance_verified,
            scf_status.live_builder_dependency_state_independently_pinned,
        ),
        "SCF status/receipt evidence model",
    )
    expected_seed_order = tuple(
        item.seed_label for item in scf_approval.exact_seed_inventory
    )
    _require_equal(scf_receipt.seed_order, expected_seed_order, "SCF seed inventory")
    _require_equal(
        scf_receipt.archive_authority_outer_call_sequence,
        ("load_immutable_scf_archive",),
        "SCF archive-authority call inventory",
    )
    _require_equal(
        scf_receipt.provider_outer_call_sequence,
        tuple(
            call
            for seed_label in expected_seed_order
            for call in (
                f"build_fresh_scf_state:{seed_label}",
                f"build_scf_problem:{seed_label}",
            )
        ),
        "SCF live-provider call inventory",
    )
    seed_count = len(expected_seed_order)
    _require_equal(
        len(scf_receipt.replayed_branch_energies_ev),
        seed_count,
        "SCF replayed-seed energy count",
    )
    comparison_counts = (
        scf_receipt.canonical_hash_comparison_count,
        scf_receipt.canonical_hash_equal_count,
        scf_receipt.bitwise_comparison_count,
        scf_receipt.bitwise_equal_count,
    )
    if any(type(value) is not int for value in comparison_counts):
        raise ValueError("Vituri scalar readiness SCF comparison counts are invalid")
    hash_count, hash_equal, bitwise_count, bitwise_equal = comparison_counts
    if (
        hash_count <= 0
        or bitwise_count != hash_count
        or not 0 <= hash_equal <= hash_count
        or not 0 <= bitwise_equal <= bitwise_count
    ):
        raise ValueError(
            "Vituri scalar readiness SCF replay comparison/equality counts mismatch"
        )
    _require_equal(
        scf_receipt.unique_ground_state_claimed,
        scf_status.global_ground_state_verified,
        "SCF global-ground-state status/receipt",
    )
    _require_equal(
        scf_receipt.restart_capability_audit.exact_restart_verified,
        scf_status.exact_restart_verified,
        "SCF exact-restart status/receipt",
    )

    pocket_status = pocket_receipt.status
    _require_equal(
        pocket_approval.scope,
        VITURI2024_POCKET_REFINEMENT_REPLAY_SCOPE,
        "pocket approval scope",
    )
    _require_equal(
        (
            pocket_receipt.evidence_model,
            pocket_receipt.archive_live_computational_independence_verified,
            pocket_receipt.hostile_provider_resistance_verified,
            pocket_receipt.hidden_live_dependency_state_excluded,
        ),
        (
            pocket_status.evidence_model,
            pocket_status.archive_live_computational_independence_verified,
            pocket_status.hostile_provider_resistance_verified,
            pocket_status.hidden_live_dependency_state_excluded,
        ),
        "pocket status/receipt evidence model",
    )
    _require_equal(
        pocket_receipt.refined_point_count,
        pocket_approval.preflight_refinement_point_count,
        "pocket refined-point count",
    )
    _require_equal(
        (
            pocket_receipt.archive_authority_outer_call_sequence,
            pocket_receipt.live_provider_outer_call_sequence,
        ),
        (
            ("load_immutable_pocket_refinement_archive",),
            ("evaluate_frozen_selected_hf_source",),
        ),
        "pocket archive/live call inventories",
    )
    _require_equal(
        (
            len(pocket_receipt.archive_topology),
            len(pocket_receipt.live_topology),
            len(pocket_receipt.lifshitz_evidence),
            len(pocket_receipt.refinement_evidence_sha256),
        ),
        (2, 2, 2, 2),
        "pocket bilateral evidence inventory counts",
    )


def _validate_exact_types(
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
    functional_contract: Vituri2024FunctionalReplayContract,
    functional_receipt: Vituri2024FunctionalReplayReceipt,
    pocket_approval: Vituri2024PocketRefinementReplayApproval,
    pocket_receipt: Vituri2024PocketRefinementReplayReceipt,
    assembly_receipt: Vituri2024TDHFSignedQAssemblyReceipt,
) -> None:
    expected = (
        (prerequisites, Vituri2024PocketRefinementPrerequisites, "prerequisites"),
        (source_payload, Vituri2024HalfMetalHFReplayPayload, "source payload"),
        (functional_contract, Vituri2024FunctionalReplayContract, "functional contract"),
        (functional_receipt, Vituri2024FunctionalReplayReceipt, "functional receipt"),
        (pocket_approval, Vituri2024PocketRefinementReplayApproval, "pocket approval"),
        (pocket_receipt, Vituri2024PocketRefinementReplayReceipt, "pocket receipt"),
        (assembly_receipt, Vituri2024TDHFSignedQAssemblyReceipt, "assembly receipt"),
    )
    for value, value_type, label in expected:
        if type(value) is not value_type:
            raise TypeError(f"Vituri scalar readiness requires the exact typed {label}")
    if type(prerequisites.binding) is not Vituri2024HalfMetalHFProviderBinding:
        raise TypeError("Vituri scalar readiness requires the exact provider binding")
    if type(prerequisites.array_replay_receipt) is not Vituri2024HalfMetalHFReplayReceipt:
        raise TypeError("Vituri scalar readiness requires the factory array receipt")
    if type(prerequisites.scf_replay_approval) is not Vituri2024SCFReplayApproval:
        raise TypeError("Vituri scalar readiness requires the exact SCF approval")
    if type(prerequisites.scf_replay_receipt) is not Vituri2024SCFReplayReceipt:
        raise TypeError("Vituri scalar readiness requires the factory SCF receipt")


def _revalidate_predecessors(
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    functional_contract: Vituri2024FunctionalReplayContract,
    functional_receipt: Vituri2024FunctionalReplayReceipt,
    pocket_approval: Vituri2024PocketRefinementReplayApproval,
    pocket_receipt: Vituri2024PocketRefinementReplayReceipt,
) -> None:
    binding = prerequisites.binding
    spec = binding.spec
    provider = binding.provider
    _validate_replay_statuses(
        prerequisites.array_replay_receipt,
        functional_contract,
        functional_receipt,
        prerequisites.scf_replay_approval,
        prerequisites.scf_replay_receipt,
        pocket_approval,
        pocket_receipt,
    )
    # Re-run exact typed constructors that do not execute provider methods.
    Vituri2024HalfMetalHFProviderBinding(spec, provider)
    Vituri2024PocketRefinementPrerequisites(
        binding,
        prerequisites.array_replay_receipt,
        prerequisites.scf_replay_approval,
        prerequisites.scf_replay_receipt,
    )
    spec.require_receipt_set_complete()
    assert spec.geometry is not None
    assert spec.ensemble is not None
    assert spec.scf_policy is not None
    assert spec.shared_functional is not None
    assert spec.attested_source is not None
    geometry = spec.geometry
    ensemble = spec.ensemble
    shared = spec.shared_functional
    source = spec.attested_source
    array = prerequisites.array_replay_receipt
    scf_approval = prerequisites.scf_replay_approval
    scf_receipt = prerequisites.scf_replay_receipt

    # Reconstruct source/spec/state/provider identity at every predecessor.
    predecessor_checks = (
        (array.provider_fingerprint, source.provider_fingerprint, "array provider"),
        (array.source_commit, source.source_commit, "array source commit"),
        (array.source_artifact_sha256, source.source_artifact_sha256, "array source artifact"),
        (array.spec_fingerprint, spec.fingerprint, "array spec"),
        (
            array.attested_source_receipt_fingerprint,
            source.fingerprint,
            "array attested source receipt",
        ),
        (scf_approval.provider_fingerprint, source.provider_fingerprint, "SCF provider"),
        (scf_approval.source_commit, source.source_commit, "SCF source commit"),
        (
            scf_approval.source_artifact_sha256,
            source.source_artifact_sha256,
            "SCF source artifact",
        ),
        (scf_approval.spec_fingerprint, spec.fingerprint, "SCF spec"),
        (scf_approval.source_state_sha256, source.source_state_sha256, "SCF source state"),
        (
            scf_approval.shared_functional_fingerprint,
            shared.fingerprint,
            "SCF shared functional",
        ),
        (
            scf_receipt.approval_fingerprint,
            scf_approval.fingerprint,
            "SCF receipt approval",
        ),
        (
            scf_receipt.selected_branch_label,
            source.selected_branch_label,
            "SCF selected branch",
        ),
    )
    for actual, expected, label in predecessor_checks:
        _require_equal(actual, expected, label)

    contract_checks = (
        (functional_contract.choice_fingerprint, functional_contract.choice.fingerprint, "functional choice"),
        (functional_contract.provider_fingerprint, source.provider_fingerprint, "functional provider"),
        (
            functional_contract.functional_provider_fingerprint,
            provider.functional_provider_fingerprint,
            "functional derived provider",
        ),
        (functional_contract.source_commit, source.source_commit, "functional source commit"),
        (
            functional_contract.source_artifact_sha256,
            source.source_artifact_sha256,
            "functional source artifact",
        ),
        (functional_contract.spec_fingerprint, spec.fingerprint, "functional spec"),
        (
            functional_contract.source_state_sha256,
            source.source_state_sha256,
            "functional source state",
        ),
        (
            functional_contract.geometry_receipt_fingerprint,
            geometry.fingerprint,
            "functional geometry",
        ),
        (
            functional_contract.ensemble_receipt_fingerprint,
            ensemble.fingerprint,
            "functional ensemble",
        ),
        (
            functional_contract.normal_order_reference_fingerprint,
            ensemble.normal_order_reference_fingerprint,
            "functional normal-order reference",
        ),
        (
            functional_contract.q0_policy_fingerprint,
            ensemble.q0_policy_fingerprint,
            "functional q0 policy",
        ),
        (
            functional_contract.interaction_receipt_fingerprint,
            shared.interaction_receipt_fingerprint,
            "functional interaction receipt",
        ),
        (
            functional_contract.shared_functional_receipt_fingerprint,
            shared.fingerprint,
            "functional shared receipt",
        ),
        (
            functional_contract.attested_source_receipt_fingerprint,
            source.fingerprint,
            "functional attested source",
        ),
        (
            functional_contract.expected_array_payload_manifest_sha256,
            expected_array_payload_manifest_sha256(spec),
            "functional expected array manifest",
        ),
        (
            functional_contract.q0_probe_inventory_sha256,
            shared.fock_finite_difference.perturbation_inventory_sha256,
            "functional q0 inventory",
        ),
        (
            functional_contract.signed_q_probe_inventory_sha256,
            shared.hessian_finite_difference.perturbation_inventory_sha256,
            "functional signed-q inventory",
        ),
        (
            functional_contract.q_probe_inventory_sha256,
            shared.hessian_finite_difference.q_probe_inventory_sha256,
            "functional q inventory",
        ),
        (
            functional_contract.fock_step_ladder,
            shared.fock_finite_difference.finite_difference_step_ladder,
            "functional Fock ladder",
        ),
        (
            functional_contract.hessian_step_ladder,
            shared.hessian_finite_difference.finite_difference_step_ladder,
            "functional Hessian ladder",
        ),
        (
            functional_contract.replay_loader_implementation_fingerprint,
            source.replay_loader_implementation_fingerprint,
            "functional array loader",
        ),
    )
    provider_contract_fields = (
        "functional_probe_loader_implementation_fingerprint",
        "functional_replay_payload_schema_fingerprint",
        "functional_replay_abi_fingerprint",
        "direct_displaced_fock_implementation_fingerprint",
        "direct_interaction_builder_implementation_fingerprint",
        "direct_full_fock_builder_implementation_fingerprint",
        "direct_builder_dependency_archive_fingerprint",
    )
    for actual, expected, label in contract_checks:
        _require_equal(actual, expected, label)
    for name in provider_contract_fields:
        _require_equal(
            getattr(functional_contract, name),
            getattr(provider, name),
            f"functional contract provider field {name}",
        )

    array_fingerprint = canonical_half_metal_hf_replay_receipt_fingerprint(array)
    functional_receipt_checks = (
        (
            functional_receipt.contract_fingerprint,
            functional_contract.fingerprint,
            "functional receipt contract",
        ),
        (
            functional_receipt.choice_fingerprint,
            functional_contract.choice_fingerprint,
            "functional receipt choice",
        ),
        (
            functional_receipt.functional_provider_fingerprint,
            functional_contract.functional_provider_fingerprint,
            "functional receipt provider",
        ),
        (
            functional_receipt.expected_array_payload_manifest_sha256,
            functional_contract.expected_array_payload_manifest_sha256,
            "functional receipt expected array manifest",
        ),
        (
            functional_receipt.array_replay_receipt_fingerprint,
            array_fingerprint,
            "functional receipt array prerequisite",
        ),
        (
            functional_receipt.array_replay_payload_manifest_sha256,
            array.hashes.payload_manifest_sha256,
            "functional receipt array payload",
        ),
        (
            functional_receipt.affine_anchor_inventory_sha256,
            functional_contract.affine_anchor_inventory_sha256,
            "functional receipt affine inventory",
        ),
        (
            functional_receipt.q0_probe_inventory_sha256,
            functional_contract.q0_probe_inventory_sha256,
            "functional receipt q0 inventory",
        ),
        (
            functional_receipt.signed_q_probe_inventory_sha256,
            functional_contract.signed_q_probe_inventory_sha256,
            "functional receipt signed-q inventory",
        ),
        (
            functional_receipt.q_probe_inventory_sha256,
            functional_contract.q_probe_inventory_sha256,
            "functional receipt q inventory",
        ),
    )
    for actual, expected, label in functional_receipt_checks:
        _require_equal(actual, expected, label)

    scf_fingerprint = canonical_scf_replay_receipt_fingerprint(scf_receipt)
    pocket_approval_checks = (
        (pocket_approval.array_replay_receipt_fingerprint, array_fingerprint, "pocket approval array"),
        (pocket_approval.scf_replay_approval_fingerprint, scf_approval.fingerprint, "pocket approval SCF approval"),
        (pocket_approval.scf_replay_receipt_fingerprint, scf_fingerprint, "pocket approval SCF receipt"),
        (pocket_approval.source_commit, source.source_commit, "pocket approval source commit"),
        (pocket_approval.source_artifact_sha256, source.source_artifact_sha256, "pocket approval source artifact"),
        (pocket_approval.spec_fingerprint, spec.fingerprint, "pocket approval spec"),
        (pocket_approval.source_state_sha256, source.source_state_sha256, "pocket approval source state"),
        (pocket_approval.selected_branch_label, source.selected_branch_label, "pocket approval branch"),
        (pocket_approval.selected_spin, source.selected_spin, "pocket approval spin"),
    )
    for actual, expected, label in pocket_approval_checks:
        _require_equal(actual, expected, label)

    pocket_receipt_checks = (
        (pocket_receipt.approval_fingerprint, pocket_approval.fingerprint, "pocket receipt approval"),
        (pocket_receipt.array_replay_receipt_fingerprint, array_fingerprint, "pocket receipt array"),
        (pocket_receipt.scf_replay_approval_fingerprint, scf_approval.fingerprint, "pocket receipt SCF approval"),
        (pocket_receipt.scf_replay_receipt_fingerprint, scf_fingerprint, "pocket receipt SCF receipt"),
        (pocket_receipt.source_commit, source.source_commit, "pocket receipt source commit"),
        (pocket_receipt.source_artifact_sha256, source.source_artifact_sha256, "pocket receipt source artifact"),
        (pocket_receipt.spec_fingerprint, spec.fingerprint, "pocket receipt spec"),
        (pocket_receipt.source_state_sha256, source.source_state_sha256, "pocket receipt source state"),
        (pocket_receipt.selected_branch_label, source.selected_branch_label, "pocket receipt branch"),
        (pocket_receipt.selected_spin, source.selected_spin, "pocket receipt spin"),
    )
    for actual, expected, label in pocket_receipt_checks:
        _require_equal(actual, expected, label)


def _revalidate_source_payload(
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    payload: Vituri2024HalfMetalHFReplayPayload,
) -> tuple[str, str]:
    """Bind the exact immutable payload bytes to the factory array receipt."""

    spec = prerequisites.binding.spec
    assert spec.attested_source is not None
    source = spec.attested_source
    receipt = prerequisites.array_replay_receipt
    identity_checks = (
        (payload.provider_fingerprint, source.provider_fingerprint, "payload provider"),
        (payload.source_commit, source.source_commit, "payload source commit"),
        (payload.source_artifact_sha256, source.source_artifact_sha256, "payload source artifact"),
        (payload.spec_fingerprint, spec.fingerprint, "payload spec"),
        (payload.source_state_sha256, source.source_state_sha256, "payload source state"),
        (payload.replay_loader_implementation_fingerprint, source.replay_loader_implementation_fingerprint, "payload replay loader"),
        (payload.replay_payload_schema_fingerprint, source.replay_payload_schema_fingerprint, "payload schema"),
    )
    for actual, expected, label in identity_checks:
        _require_equal(actual, expected, label)

    hashes = receipt.hashes
    array_checks = (
        (canonical_array_sha256(payload.mesh), hashes.ordered_momentum_mesh_sha256, "payload mesh/receipt"),
        (canonical_orbital_order_sha256(payload.mesh), hashes.ordered_orbitals_sha256, "payload orbital order/receipt"),
        (canonical_array_sha256(payload.active_band_states), hashes.active_band_states_sha256, "payload active states/receipt"),
        (canonical_array_sha256(payload.energies), hashes.ordered_energies_sha256, "payload energies/receipt"),
        (canonical_array_sha256(payload.occupations), hashes.ordered_occupations_sha256, "payload occupations/receipt"),
        (canonical_array_sha256(payload.projector), hashes.ordered_projector_sha256, "payload projector/receipt"),
        (canonical_array_sha256(payload.fock), hashes.ordered_fock_sha256, "payload Fock/receipt"),
        (canonical_array_sha256(payload.h0), hashes.h0_sha256, "payload h0/receipt"),
        (canonical_array_sha256(payload.interaction_h), hashes.interaction_h_sha256, "payload interaction_h/receipt"),
        (payload.source_state_sha256, hashes.reconstructed_source_state_sha256, "payload reconstructed source state"),
    )
    for actual, expected, label in array_checks:
        _require_equal(actual, expected, label)
    payload_fingerprint = _receipt_fingerprint(payload)
    _validate_fingerprint("source_payload_fingerprint", payload_fingerprint)
    _validate_fingerprint("payload_manifest_sha256", hashes.payload_manifest_sha256)
    return payload_fingerprint, hashes.payload_manifest_sha256


def _mesh_index(payload: Vituri2024HalfMetalHFReplayPayload, momentum: tuple[float, float]) -> int:
    target = np.asarray(momentum, dtype=np.float64)
    matches = np.flatnonzero(np.all(payload.mesh == target[None, :], axis=1))
    if matches.size != 1:
        raise ValueError(
            "Vituri scalar readiness transition momentum has no unique exact payload mesh index"
        )
    return int(matches[0])


def _flavor_index(valley: int, spin: int) -> int:
    try:
        return INTERNAL_FLAVOR_ORDER.index((valley, spin))
    except ValueError as error:
        raise ValueError(
            "Vituri scalar readiness transition flavor is absent from exact payload order"
        ) from error


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFTransitionSourceBinding:
    """One ordered assembly transition re-derived from the exact replay payload."""

    lane: Literal["plus", "minus"]
    ordered_transition_index: int
    transition_fingerprint: str
    core_pair_fingerprint: str
    particle_mesh_index: int
    hole_mesh_index: int
    particle_flavor_index: int
    hole_flavor_index: int
    particle_flat_orbital_index: int
    hole_flat_orbital_index: int
    particle_energy_ev: float
    hole_energy_ev: float
    particle_occupation: int
    hole_occupation: int
    source_artifact_sha256: str
    source_state_sha256: str
    selected_branch_label: str
    pair_to_tangent_ready: bool

    def __post_init__(self) -> None:
        if self.lane not in ("plus", "minus"):
            raise ValueError("transition-source lane must be plus or minus")
        for name in (
            "transition_fingerprint",
            "core_pair_fingerprint",
            "source_artifact_sha256",
            "source_state_sha256",
        ):
            _validate_fingerprint(name, getattr(self, name))
        for name in (
            "ordered_transition_index",
            "particle_mesh_index",
            "hole_mesh_index",
            "particle_flavor_index",
            "hole_flavor_index",
            "particle_flat_orbital_index",
            "hole_flat_orbital_index",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative exact integer")
        if not np.isfinite(self.particle_energy_ev) or not np.isfinite(self.hole_energy_ev):
            raise ValueError("transition-source energies must be finite")
        if (self.particle_occupation, self.hole_occupation) != (0, 1):
            raise ValueError("transition-source binding requires particle occ0 and hole occ1")
        if type(self.selected_branch_label) is not str or not self.selected_branch_label:
            raise ValueError("transition-source selected branch must be explicit")
        if self.pair_to_tangent_ready is not True:
            raise ValueError("transition-source pair-to-tangent readiness was not established")

    @property
    def fingerprint(self) -> str:
        return _receipt_fingerprint(self)


def _transition_source_bindings(
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    payload: Vituri2024HalfMetalHFReplayPayload,
    assembly: Vituri2024TDHFSignedQAssemblyReceipt,
) -> tuple[Vituri2024TDHFTransitionSourceBinding, ...]:
    spec = prerequisites.binding.spec
    assert spec.attested_source is not None
    source = spec.attested_source
    nk = int(payload.mesh.shape[0])
    evidence: list[Vituri2024TDHFTransitionSourceBinding] = []
    lane_specs = (
        ("plus", assembly.signed_pair.plus_inventory.transitions, assembly.blocks.plus_pairs),
        ("minus", assembly.signed_pair.minus_inventory.transitions, assembly.blocks.minus_pairs),
    )
    for lane, transitions, pairs in lane_specs:
        if len(transitions) != len(pairs):
            raise ValueError("Vituri scalar readiness transition/core-pair count mismatch")
        for index, (transition, pair) in enumerate(zip(transitions, pairs)):
            particle_mesh = _mesh_index(payload, transition.particle.momentum_inverse_angstrom)
            hole_mesh = _mesh_index(payload, transition.hole.momentum_inverse_angstrom)
            particle_flavor = _flavor_index(
                transition.particle.flavor.valley, transition.particle.flavor.spin
            )
            hole_flavor = _flavor_index(
                transition.hole.flavor.valley, transition.hole.flavor.spin
            )
            particle_energy = float(payload.energies[particle_flavor, particle_mesh])
            hole_energy = float(payload.energies[hole_flavor, hole_mesh])
            particle_occupation = int(payload.occupations[particle_flavor, particle_mesh])
            hole_occupation = int(payload.occupations[hole_flavor, hole_mesh])
            checks = (
                (transition.particle_energy_ev, particle_energy, f"{lane} transition particle energy"),
                (transition.hole_energy_ev, hole_energy, f"{lane} transition hole energy"),
                (transition.particle_occupation, particle_occupation, f"{lane} transition particle occupation"),
                (transition.hole_occupation, hole_occupation, f"{lane} transition hole occupation"),
                (transition.source_artifact_sha256, source.source_artifact_sha256, f"{lane} transition source artifact"),
            )
            for actual, expected, label in checks:
                _require_equal(actual, expected, label)
            evidence.append(
                Vituri2024TDHFTransitionSourceBinding(
                    lane=lane,  # type: ignore[arg-type]
                    ordered_transition_index=index,
                    transition_fingerprint=transition.fingerprint,
                    core_pair_fingerprint=fingerprint_tdhf_pairs((pair,)),
                    particle_mesh_index=particle_mesh,
                    hole_mesh_index=hole_mesh,
                    particle_flavor_index=particle_flavor,
                    hole_flavor_index=hole_flavor,
                    particle_flat_orbital_index=particle_flavor * nk + particle_mesh,
                    hole_flat_orbital_index=hole_flavor * nk + hole_mesh,
                    particle_energy_ev=particle_energy,
                    hole_energy_ev=hole_energy,
                    particle_occupation=particle_occupation,
                    hole_occupation=hole_occupation,
                    source_artifact_sha256=source.source_artifact_sha256,
                    source_state_sha256=source.source_state_sha256,
                    selected_branch_label=source.selected_branch_label,
                    pair_to_tangent_ready=True,
                )
            )
    return tuple(evidence)


def _revalidate_assembly(
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    payload: Vituri2024HalfMetalHFReplayPayload,
    assembly: Vituri2024TDHFSignedQAssemblyReceipt,
) -> tuple[object, ...]:
    spec = prerequisites.binding.spec
    assert spec.geometry is not None
    assert spec.shared_functional is not None
    assert spec.attested_source is not None
    geometry = spec.geometry
    shared = spec.shared_functional
    source = spec.attested_source

    # The property replays the assembly receipt's complete live-state checks.
    assembly_fingerprint = assembly.fingerprint
    _validate_fingerprint("assembly_fingerprint", assembly_fingerprint)
    if (
        not np.isfinite(assembly.structure_tolerance)
        or assembly.structure_tolerance < 0.0
    ):
        raise ValueError("assembly structure_tolerance must be finite and nonnegative")
    if (
        assembly.structure_tolerance
        > VITURI2024_SCALAR_READINESS_LOCKED_STRUCTURE_THRESHOLD
    ):
        raise ValueError(
            "assembly structure_tolerance exceeds the locked Vituri scalar-readiness "
            f"threshold {VITURI2024_SCALAR_READINESS_LOCKED_STRUCTURE_THRESHOLD}"
        )
    if type(assembly.sector) is not TDHFGenericSignedQSector:
        raise TypeError("Vituri scalar readiness requires an exact generic signed sector")
    if assembly.sector.static_hessian_authority != "projected_signed_ab":
        raise ValueError("assembly original static-Hessian authority was inflated")
    if assembly.static_hessian_authority != "projected_signed_ab":
        raise ValueError("assembly receipt original authority was inflated")
    if assembly.post_symmetrized or assembly.post_hermitized:
        raise ValueError("assembly was post-repaired")

    transitions = (
        assembly.signed_pair.plus_inventory.transitions
        + assembly.signed_pair.minus_inventory.transitions
    )
    if not transitions:
        raise ValueError("assembly transition inventories are empty")
    for transition in transitions:
        if transition.source_artifact_sha256 != source.source_artifact_sha256:
            raise ValueError("assembly nested transition source_artifact does not match HF source artifact")
    plus_context = assembly.signed_pair.plus_context
    minus_context = assembly.signed_pair.minus_context
    if plus_context.area is not minus_context.area:
        raise ValueError("assembly signed lanes do not retain the exact same finite-area object")
    if plus_context.interaction is not minus_context.interaction:
        raise ValueError("assembly signed lanes do not retain the exact same interaction receipt")
    if type(plus_context.area) is not Vituri2024FiniteAreaReceipt:
        raise TypeError("Vituri scalar readiness requires the exact finite-area receipt type")
    if type(plus_context.interaction) is not Vituri2024InteractionChoiceReceipt:
        raise TypeError("Vituri scalar readiness requires the exact interaction receipt type")
    required_delta1 = geometry.delta1_mev * 1.0e-3
    for lane, context in (("plus", plus_context), ("minus", minus_context)):
        _require_equal(context.Delta1, required_delta1, f"assembly {lane} Delta1")
        _require_equal(
            context.area.area_angstrom_squared,
            geometry.area_angstrom_squared,
            f"assembly {lane} finite-area value",
        )
        _require_equal(
            context.area.fingerprint,
            geometry.finite_area_receipt_fingerprint,
            f"assembly {lane} finite-area fingerprint",
        )
        _require_equal(
            context.interaction_receipt_fingerprint,
            shared.interaction_receipt_fingerprint,
            f"assembly {lane} interaction receipt",
        )
        _require_equal(
            context.assembly_context_fingerprint,
            assembly.assembly_context_fingerprint,
            f"assembly {lane} full context fingerprint",
        )
    _require_equal(
        plus_context.kinematics_provider_sha256,
        minus_context.kinematics_provider_sha256,
        "assembly kinematics provider cross-lane",
    )
    _require_equal(
        plus_context.kinematics_source_text,
        minus_context.kinematics_source_text,
        "assembly kinematics source cross-lane",
    )
    transition_bindings = _transition_source_bindings(
        prerequisites, payload, assembly
    )

    plus_pairs_fingerprint = fingerprint_tdhf_pairs(assembly.blocks.plus_pairs)
    minus_pairs_fingerprint = fingerprint_tdhf_pairs(assembly.blocks.minus_pairs)
    _require_equal(
        assembly.plus_pairs_fingerprint,
        plus_pairs_fingerprint,
        "assembly +q pair fingerprint",
    )
    _require_equal(
        assembly.minus_pairs_fingerprint,
        minus_pairs_fingerprint,
        "assembly -q pair fingerprint",
    )
    sector_fingerprint = fingerprint_tdhf_sector(assembly.sector)
    _require_equal(
        assembly.sector_fingerprint,
        sector_fingerprint,
        "assembly sector fingerprint",
    )
    matrix_fingerprints = (
        fingerprint_tdhf_matrix(assembly.blocks.A_plus),
        fingerprint_tdhf_matrix(assembly.blocks.B_plus_minus),
        fingerprint_tdhf_matrix(assembly.blocks.A_minus),
        fingerprint_tdhf_matrix(assembly.blocks.B_minus_plus),
    )
    recorded_matrix_fingerprints = (
        assembly.A_plus_matrix_fingerprint,
        assembly.B_plus_minus_matrix_fingerprint,
        assembly.A_minus_matrix_fingerprint,
        assembly.B_minus_plus_matrix_fingerprint,
    )
    _require_equal(
        recorded_matrix_fingerprints,
        matrix_fingerprints,
        "assembly matrix fingerprints",
    )
    _require_equal(
        assembly.sector.source_fingerprint,
        assembly.source_fingerprint,
        "assembly sector source fingerprint",
    )
    _require_equal(
        assembly.sector.sewing.source_fingerprint,
        assembly.source_fingerprint,
        "assembly sewing source fingerprint",
    )
    _require_equal(
        assembly.sector.sewing.plus_pairs_fingerprint,
        plus_pairs_fingerprint,
        "assembly sewing +q pairs",
    )
    _require_equal(
        assembly.sector.sewing.minus_pairs_fingerprint,
        minus_pairs_fingerprint,
        "assembly sewing -q pairs",
    )
    signed = build_tdhf_signed_q_matrices(
        assembly.blocks,
        assembly.sector.sewing,
        structure_tolerance=VITURI2024_SCALAR_READINESS_LOCKED_STRUCTURE_THRESHOLD,
        raise_on_structure_error=True,
    )
    if (
        signed.structure.tolerance
        != VITURI2024_SCALAR_READINESS_LOCKED_STRUCTURE_THRESHOLD
        or not signed.structure.ok
    ):
        raise ValueError("assembly projected signed-q structure is not positive")
    max_structure_residual = _max_signed_structure_residual(signed.structure)
    if max_structure_residual > VITURI2024_SCALAR_READINESS_LOCKED_STRUCTURE_THRESHOLD:
        raise ValueError("assembly projected signed-q structure exceeds the locked threshold")
    h_plus_fingerprint = fingerprint_tdhf_matrix(signed.H_plus)
    canonical_interaction_fingerprint = vituri2024_tdhf_interaction_fingerprint(
        plus_context
    )
    _require_equal(
        assembly.interaction_fingerprint,
        canonical_interaction_fingerprint,
        "assembly canonical interaction fingerprint",
    )
    _require_equal(
        assembly.sector.interaction_fingerprint,
        canonical_interaction_fingerprint,
        "assembly sector canonical interaction fingerprint",
    )
    return (
        assembly_fingerprint,
        plus_pairs_fingerprint,
        minus_pairs_fingerprint,
        sector_fingerprint,
        matrix_fingerprints,
        h_plus_fingerprint,
        max_structure_residual,
        transition_bindings,
        plus_context.area,
        plus_context.interaction,
        required_delta1,
        plus_context.kinematics_provider_sha256,
        plus_context.kinematics_source_text,
        plus_context.assembly_context_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFScalarReadinessReceipt:
    _factory_token: InitVar[object]
    api_version: str
    exact_array_replay_fingerprint: str
    source_payload_fingerprint: str
    source_payload_manifest_sha256: str
    functional_contract_fingerprint: str
    functional_receipt_fingerprint: str
    scf_approval_fingerprint: str
    scf_receipt_fingerprint: str
    pocket_approval_fingerprint: str
    pocket_receipt_fingerprint: str
    assembly_receipt_fingerprint: str
    provider_fingerprint: str
    source_artifact_sha256: str
    source_fingerprint: str
    spec_fingerprint: str
    source_state_sha256: str
    selected_branch_label: str
    selected_spin: int
    finite_area: Vituri2024FiniteAreaReceipt
    finite_area_receipt_fingerprint: str
    area_angstrom_squared: float
    delta1_ev: float
    interaction_receipt: Vituri2024InteractionChoiceReceipt
    interaction_receipt_fingerprint: str
    assembly_interaction_fingerprint: str
    kinematics_provider_sha256: str
    kinematics_source_text: str
    assembly_context_fingerprint: str
    transition_source_bindings: tuple[Vituri2024TDHFTransitionSourceBinding, ...]
    transition_source_bindings_fingerprint: str
    plus_pairs_fingerprint: str
    minus_pairs_fingerprint: str
    sector_fingerprint: str
    A_plus_matrix_fingerprint: str
    B_plus_minus_matrix_fingerprint: str
    A_minus_matrix_fingerprint: str
    B_minus_plus_matrix_fingerprint: str
    h_plus_fingerprint: str
    max_structure_residual: float
    locked_structure_threshold: float
    original_sector_authority: str
    static_hessian_authority: str
    established_evidence: tuple[str, ...]
    exact_blockers: tuple[str, ...]
    predecessor_chain_bound: bool
    payload_transition_binding_verified: bool
    assembly_context_bound: bool
    projected_signed_q_structure_verified: bool
    scalar_curvature_executed: bool
    mathematical_scalar_curvature_match: bool
    static_hessian_authority_promoted: bool
    promotion_eligible: bool

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _READINESS_FACTORY_TOKEN:
            raise TypeError("Vituri scalar readiness requires the private factory token")
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        if self.api_version != "vituri2024_tdhf_scalar_readiness.v4":
            raise ValueError("Vituri scalar readiness API version changed")
        for name in (
            "exact_array_replay_fingerprint",
            "source_payload_fingerprint",
            "source_payload_manifest_sha256",
            "functional_contract_fingerprint",
            "functional_receipt_fingerprint",
            "scf_approval_fingerprint",
            "scf_receipt_fingerprint",
            "pocket_approval_fingerprint",
            "pocket_receipt_fingerprint",
            "assembly_receipt_fingerprint",
            "provider_fingerprint",
            "source_artifact_sha256",
            "source_fingerprint",
            "spec_fingerprint",
            "source_state_sha256",
            "finite_area_receipt_fingerprint",
            "interaction_receipt_fingerprint",
            "assembly_interaction_fingerprint",
            "kinematics_provider_sha256",
            "assembly_context_fingerprint",
            "transition_source_bindings_fingerprint",
            "plus_pairs_fingerprint",
            "minus_pairs_fingerprint",
            "sector_fingerprint",
            "A_plus_matrix_fingerprint",
            "B_plus_minus_matrix_fingerprint",
            "A_minus_matrix_fingerprint",
            "B_minus_plus_matrix_fingerprint",
            "h_plus_fingerprint",
        ):
            _validate_fingerprint(name, getattr(self, name))
        if (
            type(self.locked_structure_threshold) is not float
            or self.locked_structure_threshold
            != VITURI2024_SCALAR_READINESS_LOCKED_STRUCTURE_THRESHOLD
        ):
            raise ValueError("readiness locked structure threshold changed")
        if (
            type(self.max_structure_residual) is not float
            or not np.isfinite(self.max_structure_residual)
            or self.max_structure_residual < 0.0
            or self.max_structure_residual > self.locked_structure_threshold
        ):
            raise ValueError("readiness max structure residual exceeds the locked threshold")
        if type(self.selected_branch_label) is not str or not self.selected_branch_label:
            raise ValueError("readiness selected branch must be explicit")
        if type(self.selected_spin) is not int or self.selected_spin not in (-1, 1):
            raise ValueError("readiness selected spin must be exactly -1 or +1")
        if type(self.finite_area) is not Vituri2024FiniteAreaReceipt:
            raise TypeError("readiness requires the exact finite-area receipt")
        if self.finite_area.fingerprint != self.finite_area_receipt_fingerprint:
            raise ValueError("readiness finite-area object/fingerprint mismatch")
        if (
            not np.isfinite(self.area_angstrom_squared)
            or self.area_angstrom_squared <= 0.0
            or self.area_angstrom_squared != self.finite_area.area_angstrom_squared
        ):
            raise ValueError("readiness finite-area object/value mismatch")
        if not np.isfinite(self.delta1_ev):
            raise ValueError("readiness Delta1 must be finite")
        if type(self.interaction_receipt) is not Vituri2024InteractionChoiceReceipt:
            raise TypeError("readiness requires the exact interaction receipt")
        if self.interaction_receipt.fingerprint != self.interaction_receipt_fingerprint:
            raise ValueError("readiness interaction object/fingerprint mismatch")
        if type(self.kinematics_source_text) is not str or not self.kinematics_source_text:
            raise ValueError("readiness kinematics source text must be explicit")
        reconstructed_context = Vituri2024TDHFAssemblyContext(
            area=self.finite_area,
            Delta1=self.delta1_ev,
            interaction=self.interaction_receipt,
            kinematics_provider_sha256=self.kinematics_provider_sha256,
            kinematics_source_text=self.kinematics_source_text,
        )
        if reconstructed_context.fingerprint != self.assembly_context_fingerprint:
            raise ValueError("readiness area/Delta1/interaction/kinematics context drifted")
        if type(self.transition_source_bindings) is not tuple or not self.transition_source_bindings:
            raise TypeError("readiness requires ordered transition-source binding evidence")
        if any(
            type(item) is not Vituri2024TDHFTransitionSourceBinding
            for item in self.transition_source_bindings
        ):
            raise TypeError("readiness transition-source evidence has a non-exact type")
        if self.transition_source_bindings_fingerprint != _receipt_fingerprint(
            self.transition_source_bindings
        ):
            raise ValueError("readiness transition-source binding fingerprint mismatch")
        if any(
            item.source_artifact_sha256 != self.source_artifact_sha256
            or item.source_state_sha256 != self.source_state_sha256
            or item.selected_branch_label != self.selected_branch_label
            or not item.pair_to_tangent_ready
            for item in self.transition_source_bindings
        ):
            raise ValueError("readiness transition-source lineage drifted")
        if self.original_sector_authority != VITURI2024_SCALAR_READINESS_AUTHORITY:
            raise ValueError("readiness original sector authority changed")
        if self.static_hessian_authority != VITURI2024_SCALAR_STATIC_HESSIAN_AUTHORITY:
            raise ValueError("readiness static-Hessian authority was inflated")
        if self.established_evidence != _ESTABLISHED_EVIDENCE:
            raise ValueError("readiness established-evidence inventory changed")
        if self.exact_blockers != VITURI2024_SCALAR_BLOCKERS:
            raise ValueError("readiness blocker inventory changed")
        status = (
            self.predecessor_chain_bound,
            self.payload_transition_binding_verified,
            self.assembly_context_bound,
            self.projected_signed_q_structure_verified,
            self.scalar_curvature_executed,
            self.mathematical_scalar_curvature_match,
            self.static_hessian_authority_promoted,
            self.promotion_eligible,
        )
        if status != (True, True, True, True, False, False, False, False):
            raise ValueError(
                "readiness may be positive only for predecessor/payload/context/projected structure"
            )

    @property
    def fingerprint(self) -> str:
        self._validate_consistency()
        return _receipt_fingerprint(self)



def build_vituri2024_tdhf_scalar_readiness(
    *,
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
    functional_contract: Vituri2024FunctionalReplayContract,
    functional_receipt: Vituri2024FunctionalReplayReceipt,
    pocket_approval: Vituri2024PocketRefinementReplayApproval,
    pocket_receipt: Vituri2024PocketRefinementReplayReceipt,
    assembly_receipt: Vituri2024TDHFSignedQAssemblyReceipt,
) -> Vituri2024TDHFScalarReadinessReceipt:
    """Reconstruct and bind the exact current Vituri chain without promotion."""

    _validate_exact_types(
        prerequisites,
        source_payload,
        functional_contract,
        functional_receipt,
        pocket_approval,
        pocket_receipt,
        assembly_receipt,
    )
    _revalidate_predecessors(
        prerequisites,
        functional_contract,
        functional_receipt,
        pocket_approval,
        pocket_receipt,
    )
    source_payload_fingerprint, source_payload_manifest_sha256 = (
        _revalidate_source_payload(prerequisites, source_payload)
    )
    _require_equal(
        pocket_receipt.base_point_count,
        int(source_payload.mesh.shape[0]),
        "pocket/base-payload point count",
    )
    (
        assembly_fingerprint,
        plus_pairs_fingerprint,
        minus_pairs_fingerprint,
        sector_fingerprint,
        matrix_fingerprints,
        h_plus_fingerprint,
        max_structure_residual,
        transition_source_bindings,
        finite_area,
        interaction_receipt,
        delta1_ev,
        kinematics_provider_sha256,
        kinematics_source_text,
        assembly_context_fingerprint,
    ) = _revalidate_assembly(prerequisites, source_payload, assembly_receipt)

    spec = prerequisites.binding.spec
    assert spec.geometry is not None
    assert spec.shared_functional is not None
    assert spec.attested_source is not None
    source = spec.attested_source
    return Vituri2024TDHFScalarReadinessReceipt(
        _factory_token=_READINESS_FACTORY_TOKEN,
        api_version="vituri2024_tdhf_scalar_readiness.v4",
        exact_array_replay_fingerprint=(
            canonical_half_metal_hf_replay_receipt_fingerprint(
                prerequisites.array_replay_receipt
            )
        ),
        source_payload_fingerprint=source_payload_fingerprint,
        source_payload_manifest_sha256=source_payload_manifest_sha256,
        functional_contract_fingerprint=functional_contract.fingerprint,
        functional_receipt_fingerprint=functional_receipt.fingerprint,
        scf_approval_fingerprint=prerequisites.scf_replay_approval.fingerprint,
        scf_receipt_fingerprint=canonical_scf_replay_receipt_fingerprint(
            prerequisites.scf_replay_receipt
        ),
        pocket_approval_fingerprint=pocket_approval.fingerprint,
        pocket_receipt_fingerprint=pocket_receipt.fingerprint,
        assembly_receipt_fingerprint=assembly_fingerprint,
        provider_fingerprint=source.provider_fingerprint,
        source_artifact_sha256=source.source_artifact_sha256,
        source_fingerprint=assembly_receipt.sector.source_fingerprint,
        spec_fingerprint=spec.fingerprint,
        source_state_sha256=source.source_state_sha256,
        selected_branch_label=source.selected_branch_label,
        selected_spin=source.selected_spin,
        finite_area=finite_area,
        finite_area_receipt_fingerprint=spec.geometry.finite_area_receipt_fingerprint,
        area_angstrom_squared=spec.geometry.area_angstrom_squared,
        delta1_ev=delta1_ev,
        interaction_receipt=interaction_receipt,
        interaction_receipt_fingerprint=(
            spec.shared_functional.interaction_receipt_fingerprint
        ),
        assembly_interaction_fingerprint=assembly_receipt.interaction_fingerprint,
        kinematics_provider_sha256=kinematics_provider_sha256,
        kinematics_source_text=kinematics_source_text,
        assembly_context_fingerprint=assembly_context_fingerprint,
        transition_source_bindings=transition_source_bindings,
        transition_source_bindings_fingerprint=_receipt_fingerprint(
            transition_source_bindings
        ),
        plus_pairs_fingerprint=plus_pairs_fingerprint,
        minus_pairs_fingerprint=minus_pairs_fingerprint,
        sector_fingerprint=sector_fingerprint,
        A_plus_matrix_fingerprint=matrix_fingerprints[0],
        B_plus_minus_matrix_fingerprint=matrix_fingerprints[1],
        A_minus_matrix_fingerprint=matrix_fingerprints[2],
        B_minus_plus_matrix_fingerprint=matrix_fingerprints[3],
        h_plus_fingerprint=h_plus_fingerprint,
        max_structure_residual=max_structure_residual,
        locked_structure_threshold=(
            VITURI2024_SCALAR_READINESS_LOCKED_STRUCTURE_THRESHOLD
        ),
        original_sector_authority=VITURI2024_SCALAR_READINESS_AUTHORITY,
        static_hessian_authority=VITURI2024_SCALAR_STATIC_HESSIAN_AUTHORITY,
        established_evidence=_ESTABLISHED_EVIDENCE,
        exact_blockers=VITURI2024_SCALAR_BLOCKERS,
        predecessor_chain_bound=True,
        payload_transition_binding_verified=True,
        assembly_context_bound=True,
        projected_signed_q_structure_verified=True,
        scalar_curvature_executed=False,
        mathematical_scalar_curvature_match=False,
        static_hessian_authority_promoted=False,
        promotion_eligible=False,
    )


build_vituri2024_tdhf_scalar_factory_readiness = (
    build_vituri2024_tdhf_scalar_readiness
)


__all__ = [
    "VITURI2024_SCALAR_BLOCKERS",
    "VITURI2024_SCALAR_READINESS_AUTHORITY",
    "VITURI2024_SCALAR_READINESS_LOCKED_STRUCTURE_THRESHOLD",
    "VITURI2024_SCALAR_STATIC_HESSIAN_AUTHORITY",
    "Vituri2024TDHFAssemblyReceipt",
    "Vituri2024TDHFScalarReadinessReceipt",
    "Vituri2024TDHFTransitionSourceBinding",
    "build_vituri2024_tdhf_scalar_factory_readiness",
    "build_vituri2024_tdhf_scalar_readiness",
]
