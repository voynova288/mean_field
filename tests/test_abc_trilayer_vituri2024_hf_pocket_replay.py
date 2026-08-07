"""Synthetic contract tests for frozen-source Vituri pocket refinement replay."""
from __future__ import annotations

from dataclasses import fields as dataclass_fields, replace
import inspect
from pathlib import Path

import numpy as np
import pytest

import mean_field.systems.abc_trilayer as abc
import mean_field.systems.abc_trilayer.vituri2024_hf_pocket_replay as pocket
import test_abc_trilayer_vituri2024_hf_preflight as fixtures


class _PocketProvider(fixtures._SCFProvider):
    def __init__(
        self,
        spec: abc.Vituri2024HalfMetalHFSpec,
        refined_fields: abc.Vituri2024ArchivedPocketRefinementFields,
    ) -> None:
        super().__init__(spec)
        self._refined_fields = refined_fields
        self.pocket_calls: list[str] = []
        self.last_request: abc.Vituri2024FrozenHFRefinementRequest | None = None
        self.live_delta: tuple[str, int, complex] | None = None
        self.identity_override: tuple[str, str] | None = None
        self.metadata_mutation: tuple[str, str] | None = None
        self.assert_request_is_archive_free = True
        self.refinement_evaluator_implementation_fingerprint = (
            abc.vituri2024_pocket_callable_manifest(
                "evaluate_frozen_selected_hf_source",
                self.evaluate_frozen_selected_hf_source,
            ).fingerprint
        )
        self.refinement_request_schema_fingerprint = (
            abc.POCKET_REFINEMENT_REQUEST_SCHEMA_FINGERPRINT
        )
        self.refinement_evaluation_schema_fingerprint = (
            abc.POCKET_REFINEMENT_EVALUATION_SCHEMA_FINGERPRINT
        )
        self.pocket_refinement_provider_fingerprint = (
            abc.pocket_refinement_provider_fingerprint(
                base_provider_fingerprint=self.provider_fingerprint,
                evaluator_implementation_fingerprint=(
                    self.refinement_evaluator_implementation_fingerprint
                ),
            )
        )

    def evaluate_frozen_selected_hf_source(
        self, request: abc.Vituri2024FrozenHFRefinementRequest
    ) -> abc.Vituri2024FrozenHFRefinementEvaluation:
        self.pocket_calls.append("evaluate")
        self.last_request = request
        if self.assert_request_is_archive_free:
            request_names = {item.name for item in dataclass_fields(request)}
            assert not any(
                "archive" in name
                or "expected" in name
                or "margin" in name
                or "topology" in name
                for name in request_names
            )
        h0 = self._refined_fields.h0.copy()
        interaction = self._refined_fields.interaction_h.copy()
        fock = self._refined_fields.fock.copy()
        if self.live_delta is not None:
            field_name, flat_index, delta = self.live_delta
            target = {"h0": h0, "interaction_h": interaction, "fock": fock}[field_name]
            target.flat[flat_index] += delta
            if field_name == "h0":
                fock.flat[flat_index] += delta
            elif field_name == "interaction_h":
                fock.flat[flat_index] += delta
            elif field_name == "fock":
                h0.flat[flat_index] += delta
        identity = {
            "pocket_refinement_provider_fingerprint": (
                self.pocket_refinement_provider_fingerprint
            ),
            "evaluator_implementation_fingerprint": (
                self.refinement_evaluator_implementation_fingerprint
            ),
            "evaluation_schema_fingerprint": (
                self.refinement_evaluation_schema_fingerprint
            ),
            "request_fingerprint": request.fingerprint,
            "source_commit": request.source_commit,
            "source_artifact_sha256": request.source_artifact_sha256,
            "spec_fingerprint": request.spec_fingerprint,
            "source_state_sha256": request.source_state_sha256,
            "selected_branch_label": request.selected_branch_label,
        }
        if self.identity_override is not None:
            identity[self.identity_override[0]] = self.identity_override[1]
        result = abc.Vituri2024FrozenHFRefinementEvaluation(
            **identity,
            h0=h0,
            interaction_h=interaction,
            fock=fock,
        )
        if self.metadata_mutation is not None:
            setattr(self, *self.metadata_mutation)
        return result


class _PocketArchiveAuthority:
    def __init__(
        self,
        spec: abc.Vituri2024HalfMetalHFSpec,
        mesh: abc.Vituri2024NestedNoWrapRefinementMesh,
        refined_fields: abc.Vituri2024ArchivedPocketRefinementFields,
    ) -> None:
        assert spec.attested_source
        source = spec.attested_source
        self.source_commit = source.source_commit
        self.source_artifact_sha256 = source.source_artifact_sha256
        self.spec_fingerprint = spec.fingerprint
        self.source_state_sha256 = source.source_state_sha256
        self.archive_schema_fingerprint = (
            abc.POCKET_REFINEMENT_ARCHIVE_SCHEMA_FINGERPRINT
        )
        self.archive_loader_implementation_fingerprint = (
            abc.vituri2024_pocket_callable_manifest(
                "load_immutable_pocket_refinement_archive",
                self.load_immutable_pocket_refinement_archive,
            ).fingerprint
        )
        self.archive_authority_fingerprint = (
            abc.pocket_refinement_archive_authority_fingerprint(
                source_commit=self.source_commit,
                source_artifact_sha256=self.source_artifact_sha256,
                spec_fingerprint=self.spec_fingerprint,
                source_state_sha256=self.source_state_sha256,
                archive_loader_implementation_fingerprint=(
                    self.archive_loader_implementation_fingerprint
                ),
            )
        )
        self.archive = abc.Vituri2024ImmutablePocketRefinementArchive(
            archive_authority_fingerprint=self.archive_authority_fingerprint,
            source_commit=self.source_commit,
            source_artifact_sha256=self.source_artifact_sha256,
            spec_fingerprint=self.spec_fingerprint,
            source_state_sha256=self.source_state_sha256,
            selected_branch_label=source.selected_branch_label,
            selected_spin=source.selected_spin,
            chemical_potential_ev=source.chemical_potential_ev,
            archive_loader_implementation_fingerprint=(
                self.archive_loader_implementation_fingerprint
            ),
            archive_schema_fingerprint=self.archive_schema_fingerprint,
            generation_phase=abc.POCKET_REFINEMENT_ARCHIVE_GENERATION_PHASE,
            mesh=mesh,
            fields=refined_fields,
        )
        self.calls: list[str] = []
        self.metadata_mutation: tuple[str, str] | None = None

    def load_immutable_pocket_refinement_archive(
        self, source_artifact_sha256: str
    ) -> abc.Vituri2024ImmutablePocketRefinementArchive:
        self.calls.append("load")
        assert source_artifact_sha256 == self.source_artifact_sha256
        result = self.archive
        if self.metadata_mutation is not None:
            setattr(self, *self.metadata_mutation)
        return result


def _case(
    *,
    archive_fields: abc.Vituri2024ArchivedPocketRefinementFields | None = None,
    branch_energies: tuple[float, float, float] = (-2.0, -1.9, -1.8),
) -> tuple[
    abc.Vituri2024PocketRefinementPrerequisites,
    _PocketArchiveAuthority,
    abc.Vituri2024PocketRefinementReplayApproval,
    _PocketProvider,
    abc.Vituri2024NestedNoWrapRefinementMesh,
    abc.Vituri2024ArchivedPocketRefinementFields,
]:
    spec = fixtures._scf_spec(branch_energies)
    assert spec.geometry and spec.ensemble and spec.scf_policy
    mesh, fields, _, _, _ = fixtures._synthetic_refinement_inputs(
        spec.geometry,
        spec.ensemble,
        fixtures._ARRAY_HASHES,
        -1,
    )
    if archive_fields is not None:
        fields = archive_fields
    scf_archive = fixtures._manual_scf_archive(spec)

    review_fields = abc.Vituri2024ArchivedPocketRefinementFields(
        fields.h0.copy(), fields.interaction_h.copy(), fields.fock.copy()
    )
    review_provider = _PocketProvider(spec, review_fields)
    review_seed = spec.scf_policy.seed_records[0]
    review_state = review_provider.build_fresh_scf_state(review_seed)
    callback_manifests = abc.vituri2024_scf_problem_callback_manifests(
        review_provider.build_scf_problem(review_state, review_seed)
    )

    provider_fields = abc.Vituri2024ArchivedPocketRefinementFields(
        fields.h0.copy(), fields.interaction_h.copy(), fields.fock.copy()
    )
    provider = _PocketProvider(spec, provider_fields)
    scf_authority = fixtures._SCFArchiveAuthority(scf_archive)
    binding = abc.Vituri2024HalfMetalHFProviderBinding(spec, provider)
    array_receipt = abc.replay_vituri2024_half_metal_hf_arrays(binding)
    scf_approval = abc.make_vituri2024_scf_replay_approval(
        binding,
        scf_authority,
        expected_archive_manifest_sha256=abc.scf_archive_manifest_sha256(scf_archive),
        expected_branch_table_sha256=scf_archive.original_branch_table_sha256,
        problem_callback_manifests=callback_manifests,
        provenance="Detached synthetic SCF prerequisite approval.",
    )
    scf_receipt = abc.replay_vituri2024_half_metal_hf_scf(
        binding, scf_authority, scf_approval
    )
    prerequisites = abc.Vituri2024PocketRefinementPrerequisites(
        binding, array_receipt, scf_approval, scf_receipt
    )
    authority = _PocketArchiveAuthority(spec, mesh, fields)
    approval = abc.make_vituri2024_pocket_refinement_replay_approval(
        prerequisites,
        authority,
        expected_archive_manifest_sha256=(
            abc.pocket_refinement_archive_manifest_sha256(authority.archive)
        ),
        provenance="Detached 7x9 synthetic pocket approval; not real Vituri evidence.",
    )
    assert authority.calls == []
    assert provider.pocket_calls == []
    return prerequisites, authority, approval, provider, mesh, fields


def _copy_fields(
    fields: abc.Vituri2024ArchivedPocketRefinementFields,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return fields.h0.copy(), fields.interaction_h.copy(), fields.fock.copy()


def _replace_snapshot_value(
    snapshot: tuple[tuple[str, str], ...], name: str, value: str
) -> tuple[tuple[str, str], ...]:
    assert name in dict(snapshot)
    return tuple((key, value if key == name else current) for key, current in snapshot)


def test_pocket_replay_success_binds_receipts_hashes_counts_and_narrow_status() -> None:
    prerequisites, authority, approval, provider, mesh, _ = _case()

    receipt = abc.replay_vituri2024_half_metal_hf_pocket_refinement(
        prerequisites, authority, approval
    )

    assert authority.calls == ["load"]
    assert provider.pocket_calls == ["evaluate"]
    assert provider.last_request is not None
    assert not np.shares_memory(
        provider._refined_fields.fock, authority.archive.fields.fock
    )
    assert not authority.archive.fields.fock.flags.writeable
    assert not provider.last_request.mesh.refined_mesh.flags.writeable
    assert receipt.base_point_count == 20
    assert receipt.refined_point_count == 63
    assert mesh.base_shape == (4, 5)
    assert mesh.refined_shape == (7, 9)
    assert mesh.subdivision_factors == (2, 2)
    assert receipt.refinement_mesh_sha256 == abc.canonical_array_sha256(
        mesh.refined_mesh
    )
    assert receipt.array_replay_receipt_fingerprint == (
        abc.canonical_half_metal_hf_replay_receipt_fingerprint(
            prerequisites.array_replay_receipt
        )
    )
    assert receipt.scf_replay_approval_fingerprint == (
        prerequisites.scf_replay_approval.fingerprint
    )
    assert receipt.scf_replay_receipt_fingerprint == (
        abc.canonical_scf_replay_receipt_fingerprint(
            prerequisites.scf_replay_receipt
        )
    )
    assert receipt.embedded_base_hashes.h0_sha256 == receipt.base_hashes.h0_sha256
    assert receipt.embedded_base_hashes.occupations_sha256 == (
        receipt.base_hashes.occupations_sha256
    )
    assert receipt.archive_field_hashes == receipt.live_field_hashes
    assert receipt.archive_live_field_max_abs_residual_ev == 0.0
    assert tuple(item.valley for item in receipt.archive_topology) == (-1, 1)
    assert all(item.signature.accepted for item in receipt.archive_topology)
    assert all(item.signature.accepted for item in receipt.live_topology)
    assert receipt.refinement_evidence_sha256 == tuple(
        item.refinement_evidence_sha256
        for item in prerequisites.binding.spec.attested_source.pocket_evidence  # type: ignore[union-attr]
    )
    for evidence in receipt.lifshitz_evidence:
        assert evidence.certified_margin_ev > 0.0
        assert evidence.archive.raw_margin_ev > (
            evidence.archive.minimum_absolute_energy_distance_to_mu_ev
        )
    status = receipt.status
    assert (
        receipt.evidence_model
        == status.evidence_model
        == "trusted_live_selected_source_evaluator_distinct_refinement_archive_object"
    )
    assert all(
        (
            status.array_replay_prerequisite_bound,
            status.scf_selected_source_prerequisite_bound,
            status.detached_refinement_archive_loaded,
            status.structured_refinement_mesh_registered,
            status.live_frozen_selected_source_evaluated,
            status.refined_occupations_recomputed,
            status.bilateral_hole_topology_recomputed,
            status.discrete_lifshitz_margin_recomputed,
            status.pocket_refinement_replayed,
        )
    )
    assert not any(
        (
            receipt.archive_live_computational_independence_verified,
            receipt.hostile_provider_resistance_verified,
            receipt.hidden_live_dependency_state_excluded,
            status.real_vituri_artifact_replayed,
            status.archive_live_computational_independence_verified,
            status.hostile_provider_resistance_verified,
            status.hidden_live_dependency_state_excluded,
            status.refined_scf_executed,
            status.refined_fixed_density_resolved,
            status.continuum_pocket_stability_verified,
            status.refinement_convergence_verified,
            status.global_ground_state_verified,
            status.scientific_execution_verified,
            status.paper_reproduction_verified,
            status.tdhf_readiness_verified,
        )
    )


def test_scf_approval_identity_rejects_cross_source_same_branch() -> None:
    current, authority, _, provider, _, _ = _case()
    foreign, _, _, _, _, _ = _case(
        branch_energies=(-2.1, -1.9, -1.8)
    )
    current_source = current.binding.spec.attested_source
    foreign_source = foreign.binding.spec.attested_source
    assert current_source is not None and foreign_source is not None
    assert current_source.selected_branch_label == foreign_source.selected_branch_label
    assert current_source.fingerprint != foreign_source.fingerprint
    assert current.binding.spec.fingerprint != foreign.binding.spec.fingerprint

    with pytest.raises(ValueError, match="SCF approval.*(spec|attested source)"):
        abc.Vituri2024PocketRefinementPrerequisites(
            current.binding,
            current.array_replay_receipt,
            foreign.scf_replay_approval,
            foreign.scf_replay_receipt,
        )
    assert authority.calls == []
    assert provider.pocket_calls == []


def test_replay_ingress_reconstructs_every_derivable_approval_field() -> None:
    prerequisites, authority, approval, provider, _, _ = _case()
    valid_sha = "f" * 64
    valid_commit = "e" * 40
    provenance = approval.prerequisite_provenance
    source_manifests = provenance.source_manifests
    mutated_provenance = replace(
        provenance,
        source_manifests=(
            replace(source_manifests[0], source_bytes_sha256=valid_sha),
            *source_manifests[1:],
        ),
    )
    mutated_base_hashes = replace(approval.base_hashes, h0_sha256=valid_sha)
    mutated_archive_snapshot = _replace_snapshot_value(
        approval.archive_authority_metadata_snapshot,
        "archive_authority_fingerprint",
        valid_sha,
    )
    mutated_provider_snapshot = _replace_snapshot_value(
        approval.live_provider_metadata_snapshot,
        "pocket_refinement_provider_fingerprint",
        valid_sha,
    )
    mutators = {
        "scope": lambda: replace(approval, scope="other_scope"),
        "prerequisite_provenance": lambda: replace(
            approval, prerequisite_provenance=mutated_provenance
        ),
        "verifier_module_ast_manifest_sha256": lambda: replace(
            approval, verifier_module_ast_manifest_sha256=valid_sha
        ),
        "array_replay_receipt_fingerprint": lambda: replace(
            approval, array_replay_receipt_fingerprint=valid_sha
        ),
        "scf_replay_approval_fingerprint": lambda: replace(
            approval, scf_replay_approval_fingerprint=valid_sha
        ),
        "scf_replay_receipt_fingerprint": lambda: replace(
            approval, scf_replay_receipt_fingerprint=valid_sha
        ),
        "source_commit": lambda: replace(approval, source_commit=valid_commit),
        "source_artifact_sha256": lambda: replace(
            approval, source_artifact_sha256=valid_sha
        ),
        "spec_fingerprint": lambda: replace(approval, spec_fingerprint=valid_sha),
        "source_state_sha256": lambda: replace(
            approval, source_state_sha256=valid_sha
        ),
        "selected_branch_label": lambda: replace(
            approval, selected_branch_label="other_same_shape_branch"
        ),
        "selected_spin": lambda: replace(
            approval, selected_spin=-approval.selected_spin
        ),
        "base_hashes": lambda: replace(approval, base_hashes=mutated_base_hashes),
        "scf_contract_fingerprint": lambda: replace(
            approval, scf_contract_fingerprint=valid_sha
        ),
        "scf_archive_manifest_sha256": lambda: replace(
            approval, scf_archive_manifest_sha256=valid_sha
        ),
        "scf_core_provenance_fingerprint": lambda: replace(
            approval, scf_core_provenance_fingerprint=valid_sha
        ),
        "scf_selected_source_status": lambda: replace(
            approval, scf_selected_source_status="other_status"
        ),
        "ordered_preflight_pocket_receipt_fingerprints": lambda: replace(
            approval,
            ordered_preflight_pocket_receipt_fingerprints=(
                valid_sha,
                approval.ordered_preflight_pocket_receipt_fingerprints[1],
            ),
        ),
        "preflight_refinement_mesh_sha256": lambda: replace(
            approval, preflight_refinement_mesh_sha256=valid_sha
        ),
        "preflight_refinement_point_count": lambda: replace(
            approval,
            preflight_refinement_point_count=(
                approval.preflight_refinement_point_count + 1
            ),
        ),
        "preflight_refinement_evidence_sha256": lambda: replace(
            approval,
            preflight_refinement_evidence_sha256=(
                valid_sha,
                approval.preflight_refinement_evidence_sha256[1],
            ),
        ),
        "preflight_raw_lifshitz_margins_ev": lambda: replace(
            approval,
            preflight_raw_lifshitz_margins_ev=(
                approval.preflight_raw_lifshitz_margins_ev[0] + 1.0e-6,
                approval.preflight_raw_lifshitz_margins_ev[1],
            ),
        ),
        "preflight_lifshitz_uncertainties_ev": lambda: replace(
            approval,
            preflight_lifshitz_uncertainties_ev=(
                approval.preflight_lifshitz_uncertainties_ev[0] + 1.0e-6,
                approval.preflight_lifshitz_uncertainties_ev[1],
            ),
        ),
        "archive_authority_metadata_snapshot": lambda: replace(
            approval,
            archive_authority_metadata_snapshot=mutated_archive_snapshot,
        ),
        "archive_loader_callable_manifest": lambda: replace(
            approval,
            archive_loader_callable_manifest=replace(
                approval.archive_loader_callable_manifest,
                source_sha256=valid_sha,
            ),
        ),
        "live_provider_metadata_snapshot": lambda: replace(
            approval, live_provider_metadata_snapshot=mutated_provider_snapshot
        ),
        "live_evaluator_callable_manifest": lambda: replace(
            approval,
            live_evaluator_callable_manifest=replace(
                approval.live_evaluator_callable_manifest,
                code_sha256=valid_sha,
            ),
        ),
    }
    derivable_fields = {
        item.name for item in dataclass_fields(approval)
    } - {"expected_archive_manifest_sha256", "detached_approval_provenance"}
    assert set(mutators) == derivable_fields

    constructor_locked = {
        "scope",
        "verifier_module_ast_manifest_sha256",
        "scf_selected_source_status",
    }
    for name, mutate in mutators.items():
        try:
            mutated = mutate()
        except (TypeError, ValueError):
            assert name in constructor_locked
        else:
            with pytest.raises(ValueError, match=f"pocket approval field mismatch: {name}"):
                abc.replay_vituri2024_half_metal_hf_pocket_refinement(
                    prerequisites, authority, mutated
                )
        assert authority.calls == []
        assert provider.pocket_calls == []


def test_only_external_approval_values_are_archive_manifest_and_provenance() -> None:
    prerequisites, authority, approval, provider, _, _ = _case()
    bad_manifest = replace(approval, expected_archive_manifest_sha256="f" * 64)
    with pytest.raises(ValueError, match="detached pocket archive manifest"):
        abc.replay_vituri2024_half_metal_hf_pocket_refinement(
            prerequisites, authority, bad_manifest
        )
    assert authority.calls == ["load"]
    assert provider.pocket_calls == []

    prerequisites, authority, approval, provider, _, _ = _case()
    changed_provenance = replace(
        approval,
        detached_approval_provenance=(
            "Different non-scientific review note; all scientific fields unchanged."
        ),
    )
    receipt = abc.replay_vituri2024_half_metal_hf_pocket_refinement(
        prerequisites, authority, changed_provenance
    )
    assert receipt.status.pocket_refinement_replayed
    assert authority.calls == ["load"]
    assert provider.pocket_calls == ["evaluate"]


def test_nested_mesh_canaries_reject_factor_shape_order_affinity_and_embedding() -> None:
    _, _, _, _, mesh, _ = _case()
    assert np.array_equal(
        mesh.base_embedding_indices,
        np.asarray(
            [(2 * row) * 9 + 2 * column for row in range(4) for column in range(5)],
            dtype=np.int64,
        ),
    )
    assert np.allclose(mesh.refined_mesh[mesh.base_embedding_indices], mesh.base_mesh)
    with pytest.raises((TypeError, ValueError), match="subdivision"):
        abc.Vituri2024NestedNoWrapRefinementMesh(
            (4, 5), (2.0, 2), mesh.base_mesh, mesh.refined_mesh  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="not both one"):
        abc.Vituri2024NestedNoWrapRefinementMesh(
            (4, 5), (1, 1), mesh.base_mesh, mesh.base_mesh
        )
    reordered = mesh.refined_mesh.copy()
    reordered[[1, 2]] = reordered[[2, 1]]
    with pytest.raises(ValueError, match="row-major affine"):
        abc.Vituri2024NestedNoWrapRefinementMesh(
            (4, 5), (2, 2), mesh.base_mesh, reordered
        )
    nonaffine = mesh.refined_mesh.copy()
    nonaffine[10, 0] += 1.0e-7
    with pytest.raises(ValueError, match="affine"):
        abc.Vituri2024NestedNoWrapRefinementMesh(
            (4, 5), (2, 2), mesh.base_mesh, nonaffine
        )
    bad_base = mesh.base_mesh.copy()
    bad_base[7, 0] += 1.0e-7
    with pytest.raises(ValueError, match="base row-major affine"):
        abc.Vituri2024NestedNoWrapRefinementMesh(
            (4, 5), (2, 2), bad_base, mesh.refined_mesh
        )


def test_digital_topology_canaries_are_no_wrap_and_use_four_eight_duality() -> None:
    disconnected = np.zeros((5, 5), dtype=np.bool_)
    disconnected[2, 1] = disconnected[2, 3] = True
    signature = abc.vituri2024_pocket_topology_signature(disconnected)
    assert signature.hole_component_count == 2
    assert not signature.accepted

    diagonal = np.zeros((5, 5), dtype=np.bool_)
    diagonal[1, 1] = diagonal[2, 2] = True
    assert abc.vituri2024_pocket_topology_signature(diagonal).hole_component_count == 2

    opposite_edges = np.zeros((5, 5), dtype=np.bool_)
    opposite_edges[2, 0] = opposite_edges[2, -1] = True
    no_wrap = abc.vituri2024_pocket_topology_signature(opposite_edges)
    assert no_wrap.hole_component_count == 2
    assert no_wrap.boundary_hole_state_count == 2

    boundary = np.zeros((5, 5), dtype=np.bool_)
    boundary[0, 2] = True
    assert not abc.vituri2024_pocket_topology_signature(boundary).accepted

    annulus = np.zeros((5, 5), dtype=np.bool_)
    annulus[1:4, 1] = True
    annulus[1:4, 3] = True
    annulus[1, 1:4] = True
    annulus[3, 1:4] = True
    annular_signature = abc.vituri2024_pocket_topology_signature(annulus)
    assert annular_signature.hole_component_count == 1
    assert annular_signature.boundary_hole_state_count == 0
    assert annular_signature.enclosed_complement_component_count == 1
    assert not annular_signature.accepted

    compact = np.zeros((5, 5), dtype=np.bool_)
    compact[2, 2] = compact[2, 3] = True
    assert abc.vituri2024_pocket_topology_signature(compact).accepted


def test_lifshitz_groups_degenerate_levels_and_ignores_nonlimiting_cardinality() -> None:
    prerequisites, authority, approval, _, mesh, fields = _case()
    receipt = abc.replay_vituri2024_half_metal_hf_pocket_refinement(
        prerequisites, authority, approval
    )
    for evidence in receipt.lifshitz_evidence:
        assert (
            evidence.archive.minimum_absolute_energy_distance_to_mu_ev
            == pytest.approx(1.0e-3)
        )
        assert evidence.archive.raw_margin_ev > 7.999e-3

    h0, interaction, fock = _copy_fields(fields)
    # Make the center and one connected neighbor a degenerate maximal group.
    for flavor, center in ((1, 2 * 9 + 4), (3, 4 * 9 + 4)):
        neighbor = center + 1
        delta = fock[flavor, flavor, center] - fock[flavor, flavor, neighbor]
        h0[flavor, flavor, neighbor] += delta
        fock[flavor, flavor, neighbor] += delta
    degenerate = abc.Vituri2024ArchivedPocketRefinementFields(h0, interaction, fock)
    spec = prerequisites.binding.spec
    assert spec.attested_source
    _, _, _, evidence, _, _ = abc.vituri2024_refinement_evidence_sha256(
        valley=-1,
        selected_spin=1,
        source_commit=spec.attested_source.source_commit,
        source_artifact_sha256=spec.attested_source.source_artifact_sha256,
        source_state_sha256=spec.attested_source.source_state_sha256,
        selected_branch_label=spec.attested_source.selected_branch_label,
        base_hashes=prerequisites.base_hashes,
        mesh=mesh,
        archive_fields=degenerate,
        live_fields=degenerate,
        chemical_potential_ev=spec.attested_source.chemical_potential_ev,
        locked_threshold_uncertainty_ev=1.0e-4,
    )
    assert evidence.archive.upper_critical_level_multiplicity == 2

    flavor = abc.INTERNAL_FLAVOR_ORDER.index((-1, 1))
    energies = np.diagonal(fields.fock, axis1=0, axis2=1).T.real[flavor]
    with pytest.raises(ValueError, match="raw Lifshitz margin|threshold uncertainty"):
        pocket._lifshitz_lane(
            np.asarray(energies, dtype=np.float64),
            mu=-0.02,
            uncertainty=0.01,
            shape=mesh.refined_shape,
        )


def test_rejects_opposite_spin_embedded_base_and_live_archive_corruption() -> None:
    prerequisites, authority, approval, provider, mesh, _ = _case()
    provider.live_delta = ("h0", (1 * 4 + 1) * 63 + 1, 1.0e-6)
    with pytest.raises(ValueError, match="archive/live refined h0"):
        abc.replay_vituri2024_half_metal_hf_pocket_refinement(
            prerequisites, authority, approval
        )

    prerequisites, authority, approval, provider, _, _ = _case()
    # Diagonal flat index for flavor 1, embedded base point 7.
    embedded = int(mesh.base_embedding_indices[7])
    provider.live_delta = ("h0", (1 * 4 + 1) * 63 + embedded, 1.0e-6)
    with pytest.raises(ValueError, match="embedded base"):
        abc.replay_vituri2024_half_metal_hf_pocket_refinement(
            prerequisites, authority, approval
        )

    _, _, _, _, mesh, fields = _case()
    h0, interaction, fock = _copy_fields(fields)
    opposite_flavor = abc.INTERNAL_FLAVOR_ORDER.index((-1, -1))
    point = 1 * 9 + 4
    delta = 0.08
    h0[opposite_flavor, opposite_flavor, point] += delta
    fock[opposite_flavor, opposite_flavor, point] += delta
    opposite_fields = abc.Vituri2024ArchivedPocketRefinementFields(
        h0, interaction, fock
    )
    prerequisites, authority, approval, _, _, _ = _case(
        archive_fields=opposite_fields
    )
    with pytest.raises(ValueError, match="opposite-spin refined holes"):
        abc.replay_vituri2024_half_metal_hf_pocket_refinement(
            prerequisites, authority, approval
        )

    h0, interaction, fock = _copy_fields(fields)
    h0[1, 1, 3 * 9 + 4] += 0.02
    fock[1, 1, 3 * 9 + 4] += 0.02
    corrupt_archive_fields = abc.Vituri2024ArchivedPocketRefinementFields(
        h0, interaction, fock
    )
    prerequisites, authority, approval, _, _, _ = _case(
        archive_fields=corrupt_archive_fields
    )
    with pytest.raises(ValueError, match="refinement evidence hash|topology"):
        abc.replay_vituri2024_half_metal_hf_pocket_refinement(
            prerequisites, authority, approval
        )


def test_rejects_arbitrary_valid_provider_and_archive_fingerprint_schemas() -> None:
    prerequisites, authority, approval, provider, _, _ = _case()
    valid_sha = "f" * 64
    valid_commit = "e" * 40
    provider_mutations = (
        ("source_commit", valid_commit),
        ("source_artifact_sha256", valid_sha),
        ("spec_fingerprint", valid_sha),
        ("source_state_sha256", valid_sha),
        ("replay_loader_implementation_fingerprint", valid_sha),
        ("replay_payload_schema_fingerprint", valid_sha),
        ("pocket_refinement_provider_fingerprint", valid_sha),
        ("refinement_evaluator_implementation_fingerprint", valid_sha),
        ("refinement_request_schema_fingerprint", valid_sha),
        ("refinement_evaluation_schema_fingerprint", valid_sha),
    )
    for name, value in provider_mutations:
        original = getattr(provider, name)
        setattr(provider, name, value)
        forged_approval = replace(
            approval,
            live_provider_metadata_snapshot=_replace_snapshot_value(
                approval.live_provider_metadata_snapshot, name, value
            ),
        )
        with pytest.raises(ValueError):
            abc.replay_vituri2024_half_metal_hf_pocket_refinement(
                prerequisites, authority, forged_approval
            )
        assert authority.calls == []
        assert provider.pocket_calls == []
        setattr(provider, name, original)

    archive_mutations = (
        ("archive_authority_fingerprint", valid_sha),
        ("source_commit", valid_commit),
        ("source_artifact_sha256", valid_sha),
        ("spec_fingerprint", valid_sha),
        ("source_state_sha256", valid_sha),
        ("archive_loader_implementation_fingerprint", valid_sha),
        ("archive_schema_fingerprint", valid_sha),
    )
    for name, value in archive_mutations:
        original = getattr(authority, name)
        setattr(authority, name, value)
        forged_approval = replace(
            approval,
            archive_authority_metadata_snapshot=_replace_snapshot_value(
                approval.archive_authority_metadata_snapshot, name, value
            ),
        )
        with pytest.raises(ValueError, match="archive authority semantic|helper-derived"):
            abc.replay_vituri2024_half_metal_hf_pocket_refinement(
                prerequisites, authority, forged_approval
            )
        assert authority.calls == []
        assert provider.pocket_calls == []
        setattr(authority, name, original)


def test_rejects_same_authority_metadata_identity_and_ast_drift_before_or_after_calls() -> None:
    prerequisites, authority, approval, provider, _, _ = _case()
    # Make the live-provider object satisfy both runtime protocols; identity
    # must still be rejected before either delegated method can be called.
    for name in abc.POCKET_REFINEMENT_ARCHIVE_AUTHORITY_METADATA_FIELDS:
        setattr(provider, name, getattr(authority, name))
    provider.load_immutable_pocket_refinement_archive = (  # type: ignore[attr-defined]
        authority.load_immutable_pocket_refinement_archive
    )
    with pytest.raises(ValueError, match="distinct objects"):
        abc.make_vituri2024_pocket_refinement_replay_approval(
            prerequisites,
            provider,  # type: ignore[arg-type]
            expected_archive_manifest_sha256=approval.expected_archive_manifest_sha256,
            provenance="invalid same object",
        )
    assert authority.calls == []

    prerequisites, authority, approval, provider, _, _ = _case()
    authority.metadata_mutation = ("source_state_sha256", "9" * 64)
    with pytest.raises(ValueError, match="metadata/callable mutated"):
        abc.replay_vituri2024_half_metal_hf_pocket_refinement(
            prerequisites, authority, approval
        )
    assert provider.pocket_calls == []

    prerequisites, authority, approval, provider, _, _ = _case()
    provider.metadata_mutation = ("source_state_sha256", "9" * 64)
    with pytest.raises(ValueError, match="metadata/callable mutated"):
        abc.replay_vituri2024_half_metal_hf_pocket_refinement(
            prerequisites, authority, approval
        )

    prerequisites, authority, approval, provider, _, _ = _case()
    provider.metadata_mutation = ("scf_provider_fingerprint", "9" * 64)
    with pytest.raises(ValueError, match="SCF approval SCF provider"):
        abc.replay_vituri2024_half_metal_hf_pocket_refinement(
            prerequisites, authority, approval
        )

    prerequisites, authority, approval, provider, _, _ = _case()
    provider.identity_override = ("request_fingerprint", "9" * 64)
    with pytest.raises(ValueError, match="result request mismatch"):
        abc.replay_vituri2024_half_metal_hf_pocket_refinement(
            prerequisites, authority, approval
        )

    source = inspect.getsource(pocket)
    mutated = source.replace(
        '"finite_grid_threshold_topology_margin_not_nearest_level_"',
        '"semantically_mutated_threshold_topology_margin_"',
        1,
    )
    assert abc.pocket_refinement_replay_module_ast_manifest_sha256(mutated) != (
        abc.pocket_refinement_replay_module_ast_manifest_sha256(source)
    )
    with pytest.raises(ValueError, match="full-module AST"):
        replace(approval, verifier_module_ast_manifest_sha256="9" * 64)


def test_ae6_prerequisite_provenance_supports_pinned_no_git_source_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    export_root = tmp_path
    repository_root = pocket._repository_root()
    for relative_path in pocket._PREREQUISITE_SOURCE_EXPECTATIONS:
        destination = export_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repository_root / relative_path).read_bytes())
    monkeypatch.setattr(pocket, "_repository_root", lambda: export_root)

    def unexpected_git(*args: object, **kwargs: object) -> object:
        raise AssertionError("source-export mode must not invoke git")

    monkeypatch.setattr(pocket, "_git", unexpected_git)
    provenance = abc.verified_vituri2024_pocket_prerequisite_provenance()
    assert provenance.provenance_mode == "pinned_hash_verified_source_export"
    assert provenance.baseline_commit == (
        abc.VITURI2024_POCKET_REPLAY_PREREQUISITE_BASELINE_COMMIT
    )
    assert not any(
        (
            provenance.repository_checks_available,
            provenance.repository_ancestry_verified,
            provenance.repository_head_sources_verified,
            provenance.repository_index_sources_verified,
            provenance.repository_worktree_sources_verified,
        )
    )
    assert {
        item.relative_path: (
            item.source_bytes_sha256,
            item.canonical_ast_sha256,
        )
        for item in provenance.source_manifests
    } == dict(pocket._PREREQUISITE_SOURCE_EXPECTATIONS)


def test_factory_gates_and_same_code_hidden_archive_copy_limitation_are_explicit() -> None:
    with pytest.raises(ValueError, match="factory-only"):
        abc.Vituri2024PocketRefinementReplayStatus()

    prerequisites, authority, approval, provider, _, _ = _case()
    # The trusted provider deliberately reads a hidden copy of archive fields.
    # This passes parity, demonstrating why independence/hostile flags stay false.
    provider._refined_fields = authority.archive.fields
    receipt = abc.replay_vituri2024_half_metal_hf_pocket_refinement(
        prerequisites, authority, approval
    )
    assert receipt.status.pocket_refinement_replayed
    assert not receipt.archive_live_computational_independence_verified
    assert not receipt.hostile_provider_resistance_verified
    assert not receipt.hidden_live_dependency_state_excluded
    assert not np.shares_memory(
        authority.archive.fields.fock,
        provider._refined_fields.fock.copy(),
    )

    receipt_kwargs = {
        name: getattr(receipt, name)
        for name in inspect.signature(
            abc.Vituri2024PocketRefinementReplayReceipt
        ).parameters
        if name != "_factory_token"
    }
    with pytest.raises(ValueError, match="factory-only"):
        abc.Vituri2024PocketRefinementReplayReceipt(**receipt_kwargs)  # type: ignore[arg-type]

    for name in pocket.__all__:
        assert getattr(abc, name) is getattr(pocket, name)
