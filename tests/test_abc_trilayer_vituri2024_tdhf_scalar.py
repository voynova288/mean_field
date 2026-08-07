"""Exact-chain tests for Vituri scalar readiness without a fabricated scalar receipt."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from mean_field.core.hf import fingerprint_tdhf_sector
import mean_field.systems.abc_trilayer as abc
from mean_field.systems.abc_trilayer.vituri2024_tdhf_scalar import (
    VITURI2024_SCALAR_BLOCKERS,
    VITURI2024_SCALAR_READINESS_LOCKED_STRUCTURE_THRESHOLD,
    Vituri2024TDHFScalarReadinessReceipt,
    Vituri2024TDHFTransitionSourceBinding,
    build_vituri2024_tdhf_scalar_readiness,
)
import test_abc_trilayer_vituri2024_hf_pocket_replay as pocket_fixtures
import test_abc_trilayer_vituri2024_hf_preflight as preflight_fixtures
import test_abc_trilayer_vituri2024_tdhf as tdhf_fixtures


def _functional_contract_for_pocket_provider(
    prerequisites: abc.Vituri2024PocketRefinementPrerequisites,
    provider: pocket_fixtures._PocketProvider,
) -> abc.Vituri2024FunctionalReplayContract:
    # Reuse only the detached registration inventory/provenance surface.  Every
    # source/spec/state/provider field is rebound to the exact pocket chain.
    _, template, _ = preflight_fixtures._functional_case()
    spec = prerequisites.binding.spec
    assert spec.geometry is not None
    assert spec.ensemble is not None
    assert spec.shared_functional is not None
    assert spec.attested_source is not None
    shared = spec.shared_functional
    source = spec.attested_source
    updates = {
        "provider_fingerprint": provider.provider_fingerprint,
        "functional_provider_fingerprint": provider.functional_provider_fingerprint,
        "source_commit": source.source_commit,
        "source_artifact_sha256": source.source_artifact_sha256,
        "spec_fingerprint": spec.fingerprint,
        "source_state_sha256": source.source_state_sha256,
        "geometry_receipt_fingerprint": spec.geometry.fingerprint,
        "ensemble_receipt_fingerprint": spec.ensemble.fingerprint,
        "normal_order_reference_fingerprint": (
            spec.ensemble.normal_order_reference_fingerprint
        ),
        "q0_policy_fingerprint": spec.ensemble.q0_policy_fingerprint,
        "interaction_receipt_fingerprint": shared.interaction_receipt_fingerprint,
        "shared_functional_receipt_fingerprint": shared.fingerprint,
        "attested_source_receipt_fingerprint": source.fingerprint,
        "expected_array_payload_manifest_sha256": (
            abc.expected_array_payload_manifest_sha256(spec)
        ),
        "q0_probe_inventory_sha256": (
            shared.fock_finite_difference.perturbation_inventory_sha256
        ),
        "signed_q_probe_inventory_sha256": (
            shared.hessian_finite_difference.perturbation_inventory_sha256
        ),
        "q_probe_inventory_sha256": (
            shared.hessian_finite_difference.q_probe_inventory_sha256
        ),
        "fock_step_ladder": (
            shared.fock_finite_difference.finite_difference_step_ladder
        ),
        "hessian_step_ladder": (
            shared.hessian_finite_difference.finite_difference_step_ladder
        ),
        "replay_loader_implementation_fingerprint": (
            source.replay_loader_implementation_fingerprint
        ),
        "functional_probe_loader_implementation_fingerprint": (
            provider.functional_probe_loader_implementation_fingerprint
        ),
        "functional_replay_payload_schema_fingerprint": (
            provider.functional_replay_payload_schema_fingerprint
        ),
        "functional_replay_abi_fingerprint": (
            provider.functional_replay_abi_fingerprint
        ),
        "direct_displaced_fock_implementation_fingerprint": (
            provider.direct_displaced_fock_implementation_fingerprint
        ),
        "direct_interaction_builder_implementation_fingerprint": (
            provider.direct_interaction_builder_implementation_fingerprint
        ),
        "direct_full_fock_builder_implementation_fingerprint": (
            provider.direct_full_fock_builder_implementation_fingerprint
        ),
        "direct_builder_dependency_archive_fingerprint": (
            provider.direct_builder_dependency_archive_fingerprint
        ),
    }
    approval = abc.Vituri2024FunctionalReplayApproval(
        choice_fingerprint=template.choice_fingerprint,
        verifier_implementation_schema_fingerprint=(
            template.verifier_implementation_schema_fingerprint
        ),
        verifier_module_ast_manifest_sha256=(
            template.verifier_module_ast_manifest_sha256
        ),
        functional_provider_fingerprint=provider.functional_provider_fingerprint,
        source_commit=source.source_commit,
        source_artifact_sha256=source.source_artifact_sha256,
        spec_fingerprint=spec.fingerprint,
        source_state_sha256=source.source_state_sha256,
        expected_array_payload_manifest_sha256=updates[
            "expected_array_payload_manifest_sha256"
        ],
        affine_anchor_inventory_sha256=template.affine_anchor_inventory_sha256,
        q0_probe_inventory_sha256=updates["q0_probe_inventory_sha256"],
        signed_q_probe_inventory_sha256=updates[
            "signed_q_probe_inventory_sha256"
        ],
        q_probe_inventory_sha256=updates["q_probe_inventory_sha256"],
        fock_step_ladder=updates["fock_step_ladder"],
        hessian_step_ladder=updates["hessian_step_ladder"],
        direct_displaced_fock_implementation_fingerprint=updates[
            "direct_displaced_fock_implementation_fingerprint"
        ],
        direct_builder_dependency_archive_fingerprint=updates[
            "direct_builder_dependency_archive_fingerprint"
        ],
        provenance=template.detached_approval_provenance,
    )
    updates["detached_approval_manifest_sha256"] = approval.manifest_sha256
    return replace(template, **updates)


def _configure_functional_payload(
    prerequisites: abc.Vituri2024PocketRefinementPrerequisites,
    provider: pocket_fixtures._PocketProvider,
) -> None:
    spec = prerequisites.binding.spec
    assert spec.attested_source is not None
    provider.functional_payload = preflight_fixtures._functional_payload(spec)
    payload = provider.replay_payload
    assert payload is not None
    raw_anchor = float(
        np.real(
            np.sum(payload.h0 * payload.projector)
            + 0.5
            * np.sum(
                (preflight_fixtures._FUNCTIONAL_G * payload.projector)
                * payload.projector
            )
        )
        / payload.projector.shape[2]
    )
    # SCF replay legitimately exercised the same provider first.  Restore the
    # selected-source scalar offset before the independent functional replay.
    provider.energy_offset = (
        spec.attested_source.selected_branch_energy_ev - raw_anchor
    )


_KINEMATICS_SHA = "7" * 64
_KINEMATICS_TEXT = "Synthetic payload-indexed C9 local quartet; no torus/carry authority."
_TRANSITION_SOURCE_TEXT = "Exact synthetic replay payload transition source."


def _payload_transition(
    payload: abc.Vituri2024HalfMetalHFReplayPayload,
    *,
    particle_k: int,
    hole_k: int,
    flavor_index: int,
    source_artifact_sha256: str,
    particle_momentum: tuple[float, float] | None = None,
    hole_momentum: tuple[float, float] | None = None,
    particle_energy_delta: float = 0.0,
) -> abc.Vituri2024DiagonalHFTransitionReceipt:
    valley, spin = abc.INTERNAL_FLAVOR_ORDER[flavor_index]
    flavor = abc.Vituri2024Flavor(valley=valley, spin=spin)
    particle = tuple(float(value) for value in payload.mesh[particle_k])
    hole = tuple(float(value) for value in payload.mesh[hole_k])
    return abc.Vituri2024DiagonalHFTransitionReceipt(
        particle=abc.Vituri2024Orbital(
            flavor=flavor,
            momentum_inverse_angstrom=(
                particle if particle_momentum is None else particle_momentum
            ),
        ),
        hole=abc.Vituri2024Orbital(
            flavor=flavor,
            momentum_inverse_angstrom=hole if hole_momentum is None else hole_momentum,
        ),
        particle_energy_ev=(
            float(payload.energies[flavor_index, particle_k])
            + particle_energy_delta
        ),
        hole_energy_ev=float(payload.energies[flavor_index, hole_k]),
        source_artifact_sha256=source_artifact_sha256,
        source_text=_TRANSITION_SOURCE_TEXT,
    )


def _assembly(
    spec: abc.Vituri2024HalfMetalHFSpec,
    payload: abc.Vituri2024HalfMetalHFReplayPayload,
    *,
    source_artifact_sha256: str | None = None,
    area: abc.Vituri2024FiniteAreaReceipt | None = None,
    interaction: abc.Vituri2024InteractionChoiceReceipt | None = None,
    delta1_ev: float | None = None,
    kinematics_sha: str = _KINEMATICS_SHA,
    kinematics_text: str = _KINEMATICS_TEXT,
    particle_energy_delta: float = 0.0,
    plus_flavor_index: int = 1,
    momentum_drift: float = 0.0,
    structure_tolerance: float = 1.0e-10,
) -> abc.Vituri2024TDHFSignedQAssemblyReceipt:
    assert spec.geometry is not None
    assert spec.attested_source is not None
    source_sha = (
        spec.attested_source.source_artifact_sha256
        if source_artifact_sha256 is None
        else source_artifact_sha256
    )
    area_receipt = preflight_fixtures._area() if area is None else area
    interaction_receipt = (
        preflight_fixtures._interaction() if interaction is None else interaction
    )
    context = abc.Vituri2024TDHFAssemblyContext(
        area=area_receipt,
        Delta1=(
            spec.geometry.delta1_mev * 1.0e-3
            if delta1_ev is None
            else delta1_ev
        ),
        interaction=interaction_receipt,
        kinematics_provider_sha256=kinematics_sha,
        kinematics_source_text=kinematics_text,
    )
    plus_particle = tuple(float(value) for value in payload.mesh[7])
    plus_hole = tuple(float(value) for value in payload.mesh[6])
    minus_particle = tuple(float(value) for value in payload.mesh[12])
    minus_hole = tuple(float(value) for value in payload.mesh[13])
    if momentum_drift:
        plus_particle = (plus_particle[0] + momentum_drift, plus_particle[1])
        minus_hole = (minus_hole[0] + momentum_drift, minus_hole[1])
    plus = _payload_transition(
        payload,
        particle_k=7,
        hole_k=6,
        flavor_index=plus_flavor_index,
        source_artifact_sha256=source_sha,
        particle_momentum=plus_particle,
        hole_momentum=plus_hole,
        particle_energy_delta=particle_energy_delta,
    )
    minus = _payload_transition(
        payload,
        particle_k=12,
        hole_k=13,
        flavor_index=3,
        source_artifact_sha256=source_sha,
        particle_momentum=minus_particle,
        hole_momentum=minus_hole,
    )
    q = tuple(
        plus.particle.momentum_inverse_angstrom[index]
        - plus.hole.momentum_inverse_angstrom[index]
        for index in range(2)
    )
    minus_q = tuple(
        minus.particle.momentum_inverse_angstrom[index]
        - minus.hole.momentum_inverse_angstrom[index]
        for index in range(2)
    )
    assert minus_q == (-q[0], -q[1])
    signed_pair = abc.Vituri2024SignedQTransitionInventoryPair(
        plus_inventory=abc.Vituri2024TransitionInventory(q, (plus,)),
        minus_inventory=abc.Vituri2024TransitionInventory(minus_q, (minus,)),
        plus_context=context,
        minus_context=context,
    )
    return abc.assemble_vituri2024_tdhf_signed_q(
        signed_pair, structure_tolerance=structure_tolerance
    )


def _chain() -> tuple[
    abc.Vituri2024PocketRefinementPrerequisites,
    abc.Vituri2024HalfMetalHFReplayPayload,
    abc.Vituri2024FunctionalReplayContract,
    abc.Vituri2024FunctionalReplayReceipt,
    abc.Vituri2024PocketRefinementReplayApproval,
    abc.Vituri2024PocketRefinementReplayReceipt,
    abc.Vituri2024TDHFSignedQAssemblyReceipt,
]:
    prerequisites, authority, pocket_approval, provider, _, _ = (
        pocket_fixtures._case()
    )
    _configure_functional_payload(prerequisites, provider)
    functional_contract = _functional_contract_for_pocket_provider(
        prerequisites, provider
    )
    functional_receipt = abc.replay_vituri2024_half_metal_hf_functional(
        prerequisites.binding, functional_contract
    )
    pocket_receipt = abc.replay_vituri2024_half_metal_hf_pocket_refinement(
        prerequisites, authority, pocket_approval
    )
    source_payload = provider.replay_payload
    assert type(source_payload) is abc.Vituri2024HalfMetalHFReplayPayload
    assembly_receipt = _assembly(prerequisites.binding.spec, source_payload)
    return (
        prerequisites,
        source_payload,
        functional_contract,
        functional_receipt,
        pocket_approval,
        pocket_receipt,
        assembly_receipt,
    )


def _readiness_args(chain: tuple[object, ...]) -> dict[str, object]:
    return dict(
        zip(
            (
                "prerequisites",
                "source_payload",
                "functional_contract",
                "functional_receipt",
                "pocket_approval",
                "pocket_receipt",
                "assembly_receipt",
            ),
            chain,
        )
    )


def test_real_fixture_chain_builds_narrow_projected_readiness() -> None:
    chain = _chain()
    (
        prerequisites,
        source_payload,
        contract,
        functional,
        pocket_approval,
        pocket,
        assembly,
    ) = chain
    readiness = build_vituri2024_tdhf_scalar_readiness(
        **_readiness_args(chain)  # type: ignore[arg-type]
    )
    spec = prerequisites.binding.spec
    assert spec.geometry is not None
    assert spec.shared_functional is not None
    assert spec.attested_source is not None

    assert readiness.predecessor_chain_bound
    assert readiness.payload_transition_binding_verified
    assert readiness.assembly_context_bound
    assert readiness.projected_signed_q_structure_verified
    assert readiness.locked_structure_threshold == 1.0e-10
    assert readiness.locked_structure_threshold == (
        VITURI2024_SCALAR_READINESS_LOCKED_STRUCTURE_THRESHOLD
    )
    assert 0.0 <= readiness.max_structure_residual <= readiness.locked_structure_threshold
    assert readiness.original_sector_authority == "projected_signed_ab"
    assert readiness.static_hessian_authority == "not_established"
    assert not readiness.scalar_curvature_executed
    assert not readiness.mathematical_scalar_curvature_match
    assert not readiness.static_hessian_authority_promoted
    assert not readiness.promotion_eligible
    assert readiness.exact_blockers == VITURI2024_SCALAR_BLOCKERS
    assert readiness.source_payload_fingerprint
    assert readiness.source_payload_manifest_sha256 == (
        prerequisites.array_replay_receipt.hashes.payload_manifest_sha256
    )
    assert readiness.functional_contract_fingerprint == contract.fingerprint
    assert readiness.functional_receipt_fingerprint == functional.fingerprint
    assert readiness.pocket_approval_fingerprint == pocket_approval.fingerprint
    assert readiness.pocket_receipt_fingerprint == pocket.fingerprint
    assert readiness.assembly_receipt_fingerprint == assembly.fingerprint
    assert readiness.provider_fingerprint == spec.attested_source.provider_fingerprint
    assert readiness.source_artifact_sha256 == (
        spec.attested_source.source_artifact_sha256
    )
    assert readiness.source_fingerprint == assembly.sector.source_fingerprint
    assert readiness.spec_fingerprint == spec.fingerprint
    assert readiness.source_state_sha256 == spec.attested_source.source_state_sha256
    assert readiness.selected_branch_label == spec.attested_source.selected_branch_label
    assert readiness.area_angstrom_squared == spec.geometry.area_angstrom_squared
    assert readiness.finite_area_receipt_fingerprint == (
        spec.geometry.finite_area_receipt_fingerprint
    )
    assert readiness.interaction_receipt_fingerprint == (
        spec.shared_functional.interaction_receipt_fingerprint
    )
    assert readiness.delta1_ev == 0.028
    assert readiness.area_angstrom_squared == 20_000.0
    assert readiness.kinematics_provider_sha256 == _KINEMATICS_SHA
    assert readiness.kinematics_source_text == _KINEMATICS_TEXT
    assert tuple((item.lane, item.particle_mesh_index, item.hole_mesh_index) for item in readiness.transition_source_bindings) == (
        ("plus", 7, 6),
        ("minus", 12, 13),
    )
    assert tuple((item.particle_flavor_index, item.hole_flavor_index) for item in readiness.transition_source_bindings) == ((1, 1), (3, 3))
    assert all(
        type(item) is Vituri2024TDHFTransitionSourceBinding
        and (item.particle_occupation, item.hole_occupation) == (0, 1)
        and item.pair_to_tangent_ready
        and item.source_state_sha256 == spec.attested_source.source_state_sha256
        and item.selected_branch_label == spec.attested_source.selected_branch_label
        for item in readiness.transition_source_bindings
    )
    assert len(readiness.fingerprint) == 64


def test_readiness_rejects_assembly_tolerance_above_locked_threshold() -> None:
    chain = _chain()
    prerequisites, payload = chain[:2]
    assert isinstance(prerequisites, abc.Vituri2024PocketRefinementPrerequisites)
    assert isinstance(payload, abc.Vituri2024HalfMetalHFReplayPayload)
    huge_tolerance_assembly = _assembly(
        prerequisites.binding.spec,
        payload,
        structure_tolerance=1.0e30,
    )
    assert huge_tolerance_assembly.structure_tolerance == 1.0e30
    args = _readiness_args(chain)
    args["assembly_receipt"] = huge_tolerance_assembly
    with pytest.raises(ValueError, match="structure_tolerance.*locked.*threshold"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]


def test_readiness_rejects_cross_source_area_and_interaction_assemblies() -> None:
    chain = _chain()
    prerequisites = chain[0]
    assert isinstance(prerequisites, abc.Vituri2024PocketRefinementPrerequisites)
    spec = prerequisites.binding.spec
    assert spec.geometry is not None

    payload = chain[1]
    assert isinstance(payload, abc.Vituri2024HalfMetalHFReplayPayload)
    cross_source = _assembly(
        spec, payload, source_artifact_sha256="e" * 64
    )
    args = _readiness_args(chain)
    args["assembly_receipt"] = cross_source
    with pytest.raises(ValueError, match="source_artifact|source artifact"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]

    wrong_area = abc.Vituri2024FiniteAreaReceipt(
        area_angstrom_squared=spec.geometry.area_angstrom_squared + 1.0,
        provider_sha256="1" * 64,
        source_text="Valid but cross-chain synthetic finite area.",
    )
    args["assembly_receipt"] = _assembly(spec, payload, area=wrong_area)
    with pytest.raises(ValueError, match="finite-area value"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]

    wrong_area_fingerprint = abc.Vituri2024FiniteAreaReceipt(
        area_angstrom_squared=spec.geometry.area_angstrom_squared,
        provider_sha256="8" * 64,
        source_text="Same value but a different finite-area receipt.",
    )
    args["assembly_receipt"] = _assembly(
        spec, payload, area=wrong_area_fingerprint
    )
    with pytest.raises(ValueError, match="finite-area fingerprint"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]

    args["assembly_receipt"] = _assembly(
        spec, payload, interaction=tdhf_fixtures._interaction()
    )
    with pytest.raises(ValueError, match="interaction receipt"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]


def test_readiness_rejects_stale_pair_matrix_and_predecessor_receipts() -> None:
    chain = _chain()
    args = _readiness_args(chain)
    assembly = chain[-1]
    functional = chain[3]
    assert isinstance(assembly, abc.Vituri2024TDHFSignedQAssemblyReceipt)
    assert isinstance(functional, abc.Vituri2024FunctionalReplayReceipt)

    old_pair = assembly.plus_pairs_fingerprint
    object.__setattr__(assembly, "plus_pairs_fingerprint", "9" * 64)
    try:
        with pytest.raises(ValueError, match="pair fingerprint"):
            build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]
    finally:
        object.__setattr__(assembly, "plus_pairs_fingerprint", old_pair)

    old_matrix = assembly.B_plus_minus_matrix_fingerprint
    object.__setattr__(assembly, "B_plus_minus_matrix_fingerprint", "a" * 64)
    try:
        with pytest.raises(ValueError, match="matrix fingerprint"):
            build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]
    finally:
        object.__setattr__(assembly, "B_plus_minus_matrix_fingerprint", old_matrix)

    old_contract = functional.contract_fingerprint
    object.__setattr__(functional, "contract_fingerprint", "b" * 64)
    try:
        with pytest.raises(ValueError, match="functional receipt contract"):
            build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]
    finally:
        object.__setattr__(functional, "contract_fingerprint", old_contract)


def test_readiness_rejects_stale_energy_momentum_flavor_delta1_and_kinematics() -> None:
    chain = _chain()
    prerequisites, payload = chain[:2]
    assert isinstance(prerequisites, abc.Vituri2024PocketRefinementPrerequisites)
    assert isinstance(payload, abc.Vituri2024HalfMetalHFReplayPayload)
    spec = prerequisites.binding.spec
    args = _readiness_args(chain)

    args["assembly_receipt"] = _assembly(
        spec, payload, particle_energy_delta=1.0e-4
    )
    with pytest.raises(ValueError, match="particle energy"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]

    args["assembly_receipt"] = _assembly(
        spec, payload, momentum_drift=1.0 / (2**20)
    )
    with pytest.raises(ValueError, match="momentum|mesh index"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]

    args["assembly_receipt"] = _assembly(spec, payload, plus_flavor_index=3)
    with pytest.raises(ValueError, match="occupation|flavor"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]

    args["assembly_receipt"] = _assembly(spec, payload, delta1_ev=0.027)
    with pytest.raises(ValueError, match="Delta1"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]

    assembly = chain[-1]
    assert isinstance(assembly, abc.Vituri2024TDHFSignedQAssemblyReceipt)
    context = assembly.signed_pair.plus_context
    original_text = context.kinematics_source_text
    object.__setattr__(context, "kinematics_source_text", "stale crossed kinematics")
    try:
        args["assembly_receipt"] = assembly
        with pytest.raises(ValueError, match="kinematics|context|tampered"):
            build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]
    finally:
        object.__setattr__(context, "kinematics_source_text", original_text)


def test_readiness_rejects_stale_payload_occupation_source_state_branch_and_receipt() -> None:
    chain = _chain()
    args = _readiness_args(chain)
    payload = chain[1]
    pocket_receipt = chain[5]
    array_receipt = chain[0].array_replay_receipt
    assert isinstance(payload, abc.Vituri2024HalfMetalHFReplayPayload)
    assert isinstance(pocket_receipt, abc.Vituri2024PocketRefinementReplayReceipt)

    occupations = payload.occupations.copy()
    occupations[1, 7] = 1
    args["source_payload"] = replace(payload, occupations=occupations)
    with pytest.raises(ValueError, match="payload occupations/receipt"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]

    energies = payload.energies.copy()
    energies[0, 0] += 1.0e-6
    args["source_payload"] = replace(payload, energies=energies)
    with pytest.raises(ValueError, match="payload energies/receipt"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]

    args["source_payload"] = replace(payload, source_state_sha256="c" * 64)
    with pytest.raises(ValueError, match="payload source state"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]
    args["source_payload"] = payload

    original_branch = pocket_receipt.selected_branch_label
    object.__setattr__(pocket_receipt, "selected_branch_label", "stale_branch")
    try:
        with pytest.raises(ValueError, match="pocket receipt branch"):
            build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]
    finally:
        object.__setattr__(pocket_receipt, "selected_branch_label", original_branch)

    original_manifest = array_receipt.hashes.payload_manifest_sha256
    object.__setattr__(array_receipt.hashes, "payload_manifest_sha256", "d" * 64)
    try:
        with pytest.raises(ValueError, match="array prerequisite|array payload|payload"):
            build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]
    finally:
        object.__setattr__(
            array_receipt.hashes, "payload_manifest_sha256", original_manifest
        )


def test_readiness_rejects_arbitrary_consistent_interaction_fingerprint() -> None:
    chain = _chain()
    args = _readiness_args(chain)
    assembly = chain[-1]
    assert isinstance(assembly, abc.Vituri2024TDHFSignedQAssemblyReceipt)
    canonical = abc.vituri2024_tdhf_interaction_fingerprint(
        assembly.signed_pair.plus_context
    )
    arbitrary = "f" * 64
    assert arbitrary != canonical

    object.__setattr__(assembly.sector, "interaction_fingerprint", arbitrary)
    object.__setattr__(assembly, "interaction_fingerprint", arbitrary)
    object.__setattr__(
        assembly,
        "sector_fingerprint",
        fingerprint_tdhf_sector(assembly.sector),
    )
    object.__setattr__(
        assembly,
        "assembly_fingerprint",
        assembly._expected_fingerprint(),
    )
    assert assembly.fingerprint
    with pytest.raises(ValueError, match="canonical interaction fingerprint"):
        build_vituri2024_tdhf_scalar_readiness(**args)  # type: ignore[arg-type]


def test_readiness_rejects_every_mutated_status_claim() -> None:
    chain = _chain()
    args = _readiness_args(chain)
    prerequisites = chain[0]
    functional = chain[3]
    pocket = chain[5]
    assert isinstance(prerequisites, abc.Vituri2024PocketRefinementPrerequisites)
    assert isinstance(functional, abc.Vituri2024FunctionalReplayReceipt)
    assert isinstance(pocket, abc.Vituri2024PocketRefinementReplayReceipt)

    canaries = (
        (prerequisites.array_replay_receipt.status, "arrays_loaded", False, "array replay"),
        (
            prerequisites.array_replay_receipt.status,
            "paper_reproduction_verified",
            True,
            "array replay",
        ),
        (
            functional.status,
            "affine_anchor_count",
            functional.status.affine_anchor_count + 1,
            "functional replay",
        ),
        (
            functional.status,
            "global_functional_chain_verified",
            True,
            "functional replay",
        ),
        (
            prerequisites.scf_replay_receipt.status,
            "selected_final_source_reproduced",
            False,
            "SCF replay",
        ),
        (
            prerequisites.scf_replay_receipt.status,
            "exact_restart_verified",
            True,
            "SCF replay",
        ),
        (
            pocket.status,
            "detached_refinement_archive_loaded",
            False,
            "pocket replay",
        ),
        (
            pocket.status,
            "refined_scf_executed",
            True,
            "pocket replay",
        ),
    )
    for status, name, mutation, label in canaries:
        original = getattr(status, name)
        object.__setattr__(status, name, mutation)
        try:
            with pytest.raises(ValueError, match=label):
                build_vituri2024_tdhf_scalar_readiness(  # type: ignore[arg-type]
                    **args
                )
        finally:
            object.__setattr__(status, name, original)


def test_readiness_rejects_status_receipt_scope_evidence_and_count_drift() -> None:
    chain = _chain()
    args = _readiness_args(chain)
    prerequisites = chain[0]
    functional = chain[3]
    pocket = chain[5]
    assert isinstance(prerequisites, abc.Vituri2024PocketRefinementPrerequisites)
    assert isinstance(functional, abc.Vituri2024FunctionalReplayReceipt)
    assert isinstance(pocket, abc.Vituri2024PocketRefinementReplayReceipt)

    canaries = (
        (functional, "scope", "cross_scope", "functional status/receipt scope"),
        (
            functional,
            "affine_anchor_count",
            functional.affine_anchor_count + 1,
            "functional status/receipt scope and count",
        ),
        (
            functional,
            "scalar_steps",
            functional.scalar_steps[:-1],
            "functional exact evidence inventory counts",
        ),
        (
            prerequisites.scf_replay_receipt,
            "evidence_model",
            "cross_evidence_model",
            "SCF status/receipt evidence model",
        ),
        (
            prerequisites.scf_replay_receipt,
            "canonical_hash_comparison_count",
            prerequisites.scf_replay_receipt.canonical_hash_comparison_count + 1,
            "SCF replay comparison/equality counts",
        ),
        (
            prerequisites.scf_replay_receipt,
            "provider_outer_call_sequence",
            prerequisites.scf_replay_receipt.provider_outer_call_sequence[:-1],
            "SCF live-provider call inventory",
        ),
        (
            pocket,
            "evidence_model",
            "cross_evidence_model",
            "pocket status/receipt evidence model",
        ),
        (
            pocket,
            "refined_point_count",
            pocket.refined_point_count + 1,
            "pocket refined-point count",
        ),
        (
            pocket,
            "archive_authority_outer_call_sequence",
            (),
            "pocket archive/live call inventories",
        ),
    )
    for receipt, name, mutation, label in canaries:
        original = getattr(receipt, name)
        object.__setattr__(receipt, name, mutation)
        try:
            with pytest.raises(ValueError, match=label):
                build_vituri2024_tdhf_scalar_readiness(  # type: ignore[arg-type]
                    **args
                )
        finally:
            object.__setattr__(receipt, name, original)


def test_factory_guard_and_synthetic_bridge_surface_is_removed() -> None:
    chain = _chain()
    readiness = build_vituri2024_tdhf_scalar_readiness(
        **_readiness_args(chain)  # type: ignore[arg-type]
    )
    readiness_kwargs = {
        item.name: getattr(readiness, item.name) for item in fields(readiness)
    }
    with pytest.raises(TypeError, match="private factory token"):
        Vituri2024TDHFScalarReadinessReceipt(
            _factory_token=object(),
            **readiness_kwargs,
        )

    original_residual = readiness.max_structure_residual
    object.__setattr__(
        readiness,
        "max_structure_residual",
        readiness.locked_structure_threshold * 2.0,
    )
    try:
        with pytest.raises(ValueError, match="max structure residual.*locked threshold"):
            _ = readiness.fingerprint
    finally:
        object.__setattr__(readiness, "max_structure_residual", original_residual)

    assert not hasattr(abc, "Vituri2024TDHFScalarSyntheticBridgeReceipt")
    assert not hasattr(abc, "build_vituri2024_tdhf_scalar_synthetic_bridge")
    assert not hasattr(abc, "build_vituri2024_synthetic_scalar_bridge")
