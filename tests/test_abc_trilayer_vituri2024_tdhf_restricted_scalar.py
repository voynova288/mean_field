"""Actual-chain tests for the restricted Vituri finite-orbital scalar oracle."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from mean_field.core.hf import (
    TDHFSignedQBlocks,
    certify_tdhf_scalar_curvature,
    exact_tdhf_projector_path,
    make_tdhf_scalar_curvature_approval,
)
import mean_field.systems.abc_trilayer as abc
import mean_field.systems.abc_trilayer.vituri2024_tdhf_restricted_scalar as restricted
import test_abc_trilayer_vituri2024_tdhf_scalar as scalar_fixtures


@dataclass(frozen=True)
class _ActualCase:
    chain: tuple[object, ...]
    payload: abc.Vituri2024HalfMetalHFReplayPayload
    assembly: abc.Vituri2024TDHFSignedQAssemblyReceipt
    readiness: abc.Vituri2024TDHFScalarReadinessReceipt
    approval: abc.Vituri2024TDHFRestrictedScalarApproval
    receipt: abc.Vituri2024TDHFRestrictedScalarReceipt


def _actual_multi_pair_assembly(
    spec: abc.Vituri2024HalfMetalHFSpec,
    payload: abc.Vituri2024HalfMetalHFReplayPayload,
    context: abc.Vituri2024TDHFAssemblyContext,
) -> abc.Vituri2024TDHFSignedQAssemblyReceipt:
    """Build the exact requested 2x2 lanes from one replay payload."""

    assert spec.attested_source is not None
    source_sha256 = spec.attested_source.source_artifact_sha256

    def transition(
        particle_k: int, hole_k: int, flavor_index: int
    ) -> abc.Vituri2024DiagonalHFTransitionReceipt:
        return scalar_fixtures._payload_transition(
            payload,
            particle_k=particle_k,
            hole_k=hole_k,
            flavor_index=flavor_index,
            source_artifact_sha256=source_sha256,
        )

    plus = (
        transition(7, 6, 1),
        transition(12, 11, 3),
    )
    # The derived B[a+,a-;A+,A-] vertex flavor deltas require the particle
    # flavors to equal the hole flavors in the same or swapped order.  Keep the
    # -q inventory in the SAME flavor order as +q; do not phase-tune the replay.
    minus = (
        transition(7, 8, 1),
        transition(12, 13, 3),
    )
    q = tuple(
        plus[0].particle.momentum_inverse_angstrom[index]
        - plus[0].hole.momentum_inverse_angstrom[index]
        for index in range(2)
    )
    minus_q = (-q[0], -q[1])
    for item in plus:
        assert tuple(
            item.particle.momentum_inverse_angstrom[index]
            - item.hole.momentum_inverse_angstrom[index]
            for index in range(2)
        ) == q
    for item in minus:
        assert tuple(
            item.particle.momentum_inverse_angstrom[index]
            - item.hole.momentum_inverse_angstrom[index]
            for index in range(2)
        ) == minus_q
    signed_pair = abc.Vituri2024SignedQTransitionInventoryPair(
        plus_inventory=abc.Vituri2024TransitionInventory(q, plus),
        minus_inventory=abc.Vituri2024TransitionInventory(minus_q, minus),
        plus_context=context,
        minus_context=context,
    )
    return abc.assemble_vituri2024_tdhf_signed_q(signed_pair)


@pytest.fixture(scope="module")
def actual_case() -> _ActualCase:
    # The predecessor chain and all four transitions come from one exact replay
    # payload.  Only the task-local signed inventory is enlarged to 2x2.
    chain = scalar_fixtures._chain()
    prerequisites = chain[0]
    payload = chain[1]
    original_assembly = chain[-1]
    assert isinstance(prerequisites, abc.Vituri2024PocketRefinementPrerequisites)
    assert type(payload) is abc.Vituri2024HalfMetalHFReplayPayload
    assert type(original_assembly) is abc.Vituri2024TDHFSignedQAssemblyReceipt
    assembly = _actual_multi_pair_assembly(
        prerequisites.binding.spec,
        payload,
        original_assembly.signed_pair.plus_context,
    )
    rebound_chain = chain[:-1] + (assembly,)
    readiness = abc.build_vituri2024_tdhf_scalar_factory_readiness(
        **scalar_fixtures._readiness_args(rebound_chain)  # type: ignore[arg-type]
    )
    approval = abc.make_vituri2024_tdhf_restricted_scalar_approval(
        readiness=readiness,
        assembly_receipt=assembly,
        source_payload=payload,
        provenance=(
            "Detached actual-Vituri restricted finite-orbital algebra approval "
            "from the exact test_abc_trilayer_vituri2024_tdhf_scalar chain."
        ),
    )
    receipt = abc.certify_vituri2024_tdhf_restricted_scalar(
        approval=approval,
        readiness=readiness,
        assembly_receipt=assembly,
        source_payload=payload,
    )
    return _ActualCase(
        chain=rebound_chain,
        payload=payload,
        assembly=assembly,
        readiness=readiness,
        approval=approval,
        receipt=receipt,
    )


def test_actual_vituri_restricted_scalar_certifies_all_independent_oracles(
    actual_case: _ActualCase,
) -> None:
    receipt = actual_case.receipt
    assembly = actual_case.assembly
    residuals = receipt.residuals

    assert receipt.authority == "restricted_finite_orbital_algebra_oracle_only"
    assert receipt.authority == abc.VITURI2024_RESTRICTED_SCALAR_AUTHORITY
    assert receipt.orbital_id_map == assembly.orbital_id_map
    assert tuple(item.orbital_id for item in receipt.orbital_crosswalk) == tuple(
        range(len(assembly.orbital_id_map))
    )
    assert len(receipt.orbital_id_map) == 6
    assert receipt.wbar_tensor_ev.shape == (6, 6, 6, 6)
    literal_shape = (len(receipt.literal_states), len(receipt.literal_states))
    assert all(not array.flags.writeable for array in (
        receipt.wbar_tensor_ev,
        receipt.p0,
        receipt.f0_ev,
        receipt.sigma_p0_ev,
        receipt.h_ev,
        receipt.literal_one_body_hamiltonian_ev,
        receipt.literal_interaction_hamiltonian_ev,
        receipt.literal_hamiltonian_ev,
    ))
    assert all(
        array.shape == literal_shape
        for array in (
            receipt.literal_one_body_hamiltonian_ev,
            receipt.literal_interaction_hamiltonian_ev,
            receipt.literal_hamiltonian_ev,
        )
    )
    assert receipt.literal_one_body_hamiltonian_fingerprint == (
        restricted._array_sha256(receipt.literal_one_body_hamiltonian_ev)
    )
    assert receipt.literal_interaction_hamiltonian_fingerprint == (
        restricted._array_sha256(receipt.literal_interaction_hamiltonian_ev)
    )
    assert receipt.literal_hamiltonian_fingerprint == restricted._array_sha256(
        receipt.literal_hamiltonian_ev
    )
    np.testing.assert_array_equal(
        receipt.p0,
        np.diag([item.occupation for item in receipt.orbital_crosswalk]),
    )
    np.testing.assert_array_equal(
        receipt.f0_ev,
        np.diag([item.energy_ev for item in receipt.orbital_crosswalk]),
    )
    np.testing.assert_allclose(
        receipt.h_ev + restricted._sigma(receipt.wbar_tensor_ev, receipt.p0),
        receipt.f0_ev,
        atol=4.0e-10,
        rtol=0.0,
    )

    # Exact replay-derived pair order, including the derived SAME -q flavor
    # order.  The core pair order must retain that inventory order.
    assert tuple(
        (
            item.lane,
            item.ordered_transition_index,
            item.particle_flavor_index,
            item.particle_mesh_index,
            item.hole_mesh_index,
        )
        for item in receipt.readiness.transition_source_bindings
    ) == (
        ("plus", 0, 1, 7, 6),
        ("plus", 1, 3, 12, 11),
        ("minus", 0, 1, 7, 8),
        ("minus", 1, 3, 12, 13),
    )
    assert tuple(
        (item.particle, item.hole) for item in assembly.blocks.plus_pairs
    ) == ((1, 0), (4, 3))
    assert tuple(
        (item.particle, item.hole) for item in assembly.blocks.minus_pairs
    ) == ((1, 2), (4, 5))

    # Entrywise actual C9 parity, with no production A/B builder in the oracle.
    np.testing.assert_allclose(receipt.scalar_A_plus_ev, assembly.blocks.A_plus)
    np.testing.assert_allclose(
        receipt.scalar_B_plus_minus_ev, assembly.blocks.B_plus_minus
    )
    np.testing.assert_allclose(receipt.scalar_A_minus_ev, assembly.blocks.A_minus)
    np.testing.assert_allclose(
        receipt.scalar_B_minus_plus_ev, assembly.blocks.B_minus_plus
    )
    np.testing.assert_allclose(
        assembly.blocks.B_plus_minus,
        assembly.blocks.B_minus_plus.T,
        atol=4.0e-10,
        rtol=0.0,
    )
    # The SAME flavor order exposes the derived untuned structure
    # B+−=[[0,b12],[b21,0]]: it is complex and not self-transpose.
    lane = assembly.blocks.B_plus_minus
    assert lane.shape == (2, 2)
    b12 = lane[0, 1]
    b21 = lane[1, 0]
    np.testing.assert_array_equal(
        lane,
        np.asarray(((0.0, b12), (b21, 0.0)), dtype=np.complex128),
    )
    assert min(abs(b12), abs(b21)) > 1.0e-8
    self_transpose_residual = float(np.max(np.abs(lane - lane.T)))
    complex_residual = float(np.max(np.abs(lane - lane.conj())))
    assert self_transpose_residual == pytest.approx(2.35e-2, rel=1.0e-2)
    assert complex_residual == pytest.approx(2.91e-2, rel=1.0e-2)
    assert max(
        residuals.c9_A_plus,
        residuals.c9_B_plus_minus,
        residuals.c9_A_minus,
        residuals.c9_B_minus_plus,
    ) <= 4.0e-10

    # Raw independently evaluated tensor symmetries; no copied/repaired entries.
    assert max(
        residuals.tensor_bra_antisymmetry,
        residuals.tensor_ket_antisymmetry,
        residuals.tensor_pair_hermiticity,
    ) <= 4.0e-10
    assert receipt.actual_vertex_compared
    assert receipt.actual_c9_compared
    assert receipt.conserving_vertices
    for item in receipt.conserving_vertices:
        a, b, g, d = item.quartet
        assert receipt.wbar_tensor_ev[a, b, g, d] == item.wbar_value_ev

    # The planner's x/y quadrature table includes the lower v=(x,y*) canary.
    labels = tuple(item.label for item in receipt.dF_column_evidence)
    assert labels == (
        "x.real[0]",
        "x.imag[0]",
        "x.real[1]",
        "x.imag[1]",
        "y.real[0]",
        "y.imag[0]",
        "lower_canonical_v.imag[0]",
        "y.real[1]",
        "y.imag[1]",
        "lower_canonical_v.imag[1]",
    )
    assert residuals.dF_physical_columns <= 4.0e-10
    assert all(item.max_abs_residual <= 4.0e-10 for item in receipt.dF_column_evidence)

    # Literal 1/4 Hamiltonian and all four exact double commutators.
    np.testing.assert_allclose(
        receipt.literal_hamiltonian_ev,
        receipt.literal_one_body_hamiltonian_ev
        + receipt.literal_interaction_hamiltonian_ev,
        atol=0.0,
        rtol=0.0,
    )
    assert max(
        residuals.double_commutator_A_plus,
        residuals.double_commutator_B_plus_minus,
        residuals.double_commutator_A_minus,
        residuals.double_commutator_B_minus_plus,
    ) <= 4.0e-10
    assert residuals.wick_literal_p0 <= 4.0e-10
    assert residuals.wick_literal_all_stencils <= 4.0e-10
    assert receipt.scalar_energy_call_count == receipt.expected_scalar_energy_call_count
    assert receipt.scalar_energy_call_count == 361
    assert len(receipt.stencil_projector_fingerprints) == 361

    # Existing core/hf approval/certificate supplies canonical 2d and d^2 gates.
    certificate = receipt.generic_certificate
    assert receipt.approval.scalar_energy_evaluated is False
    assert certificate.approval is receipt.generic_approval
    assert certificate.approval.canonical_stationarity_complete_inventory
    assert certificate.approval.canonical_complete_inventory
    assert len(certificate.stationarity_evidence) == 8
    assert len(certificate.direction_evidence) == 16
    assert certificate.scalar_curvature_executed
    assert certificate.stationarity_complete_all_passed
    assert certificate.mathematical_scalar_hessian_match
    assert certificate.mathematical_scalar_curvature_match
    assert not certificate.static_hessian_authority_promoted
    assert not certificate.promotion_eligible
    assert residuals.generic_stationarity <= max(
        step.stationarity_bound
        for direction in certificate.stationarity_evidence
        for step in direction.steps
    )
    assert residuals.generic_curvature <= max(
        step.curvature_bound
        for direction in certificate.direction_evidence
        for step in direction.steps
    )

    # Passing the restricted oracle never promotes the original C9 sector.
    assert assembly.sector.static_hessian_authority == "projected_signed_ab"
    assert not receipt.real_full_provider
    assert not receipt.source_scalar
    assert not receipt.global_static_hessian_authority
    assert not receipt.authority_promoted
    assert not receipt.production_ready
    assert not receipt.paper_numerical_parity
    assert len(receipt.fingerprint) == 64


def test_wrong_density_index_order_and_omitted_counterterm_are_live_canaries(
    actual_case: _ActualCase,
) -> None:
    receipt = actual_case.receipt
    evidence = next(
        item for item in receipt.dF_column_evidence if item.label == "x.imag[0]"
    )
    pair = actual_case.assembly.blocks.plus_pairs[0]
    density = np.zeros_like(receipt.p0)
    density[pair.particle, pair.hole] = 1.0j
    density[pair.hole, pair.particle] = -1.0j

    # Wrong P[b,g] instead of the locked P[g,b].
    wrong_dF = np.einsum(
        "ibgj,bg->ij", receipt.wbar_tensor_ev, density, optimize=False
    )
    pairs = (
        actual_case.assembly.blocks.plus_pairs
        + actual_case.assembly.blocks.minus_pairs
    )
    wrong_column = np.asarray(
        [wrong_dF[item.particle, item.hole] for item in pairs]
    )
    assert np.max(np.abs(wrong_column - evidence.expected_column)) > 1.0e-8

    # Omitting h=F0-Sigma[P0] leaves a nonstationary interaction counterterm.
    wrong_fock_at_p0 = receipt.f0_ev + restricted._sigma(
        receipt.wbar_tensor_ev, receipt.p0
    )
    assert np.max(np.abs(wrong_fock_at_p0 - receipt.f0_ev)) > 1.0e-8


def test_literal_one_quarter_factor_and_action_order_canaries(
    actual_case: _ActualCase,
) -> None:
    receipt = actual_case.receipt
    basis = receipt.tangent_basis
    directions = receipt.generic_approval.directions
    projectors = tuple(
        exact_tdhf_projector_path(basis, direction, 0.017)
        for direction in directions
    )
    wrong_half_factor = (
        receipt.literal_one_body_hamiltonian_ev
        + 2.0 * receipt.literal_interaction_hamiltonian_ev
    )
    # Creating a before b in the sequential action represents
    # c_b^dagger c_a^dagger and flips the interaction sign.
    wrong_creation_order = (
        receipt.literal_one_body_hamiltonian_ev
        - receipt.literal_interaction_hamiltonian_ev
    )
    ne = int(np.trace(receipt.p0).real)

    factor_residuals = []
    order_residuals = []
    for projector in (receipt.p0,) + projectors:
        wick = restricted._wick_energy(
            receipt.h_ev, receipt.wbar_tensor_ev, projector
        ).real
        factor_residuals.append(
            abs(
                restricted._literal_slater_energy(
                    projector, receipt.literal_states, ne, wrong_half_factor
                )
                - wick
            )
        )
        order_residuals.append(
            abs(
                restricted._literal_slater_energy(
                    projector, receipt.literal_states, ne, wrong_creation_order
                )
                - wick
            )
        )
    assert max(factor_residuals) > 1.0e-8
    assert max(order_residuals) > 1.0e-8


def test_extra_area_and_ab_sign_conjugation_lane_canaries(
    actual_case: _ActualCase,
) -> None:
    receipt = actual_case.receipt
    assembly = actual_case.assembly
    area = receipt.context.area.area_angstrom_squared
    extra_area_blocks = restricted._expected_scalar_blocks(
        assembly, receipt.f0_ev, receipt.wbar_tensor_ev / area
    )
    assert max(
        np.max(np.abs(actual - wrong))
        for actual, wrong in zip(
            (
                assembly.blocks.A_plus,
                assembly.blocks.B_plus_minus,
                assembly.blocks.A_minus,
                assembly.blocks.B_minus_plus,
            ),
            extra_area_blocks,
        )
    ) > 1.0e-8

    plus_gap = restricted._gap_matrix(assembly.blocks.plus_pairs, receipt.f0_ev)
    wrong_A_sign = plus_gap + (plus_gap - receipt.scalar_A_plus_ev)
    assert np.max(np.abs(wrong_A_sign - assembly.blocks.A_plus)) > 1.0e-8
    assert np.max(
        np.abs(-receipt.scalar_B_plus_minus_ev - assembly.blocks.B_plus_minus)
    ) > 1.0e-8

    # Actual replay values, without phase tuning, discriminate sign,
    # conjugation, transpose, and minus-pair order.
    lane = receipt.scalar_B_plus_minus_ev
    with pytest.raises(ValueError, match="entrywise mismatch"):
        restricted._require_entrywise(lane.conj(), lane, label="B conjugation canary")
    with pytest.raises(ValueError, match="entrywise mismatch"):
        restricted._require_entrywise(lane.T, lane, label="B transpose canary")
    with pytest.raises(ValueError, match="entrywise mismatch"):
        restricted._require_entrywise(
            lane[:, ::-1], lane, label="B pair-order canary"
        )


def test_actual_nonzero_B_discriminates_lower_imaginary_coordinate_sign(
    actual_case: _ActualCase,
) -> None:
    receipt = actual_case.receipt
    n_plus = len(actual_case.assembly.blocks.plus_pairs)
    B_plus_minus = receipt.scalar_B_plus_minus_ev
    assert np.max(np.abs(B_plus_minus)) > 1.0e-8
    for index in range(B_plus_minus.shape[1]):
        lower = next(
            item
            for item in receipt.dF_column_evidence
            if item.label == f"lower_canonical_v.imag[{index}]"
        )
        physical_y_plus_i = next(
            item
            for item in receipt.dF_column_evidence
            if item.label == f"y.imag[{index}]"
        )
        np.testing.assert_allclose(
            lower.expected_column[:n_plus],
            1.0j * B_plus_minus[:, index],
            atol=4.0e-10,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            physical_y_plus_i.expected_column[:n_plus],
            -1.0j * B_plus_minus[:, index],
            atol=4.0e-10,
            rtol=0.0,
        )
        assert np.max(
            np.abs(
                lower.expected_column[:n_plus]
                - physical_y_plus_i.expected_column[:n_plus]
            )
        ) > 1.0e-8


def test_symmetry_preserving_raw_tensor_mutations_fail_actual_C9_before_receipt(
    actual_case: _ActualCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_builder = restricted._build_raw_tensor
    permutations = np.asarray((1, 0, 2, 3, 4, 5), dtype=int)

    def mutate(name: str, wbar: np.ndarray) -> np.ndarray:
        if name == "overall_sign":
            return -wbar
        if name == "complex_conjugation":
            return wbar.conj()
        if name == "simultaneous_orbital_permutation":
            indices = np.ix_(permutations, permutations, permutations, permutations)
            return wbar[indices]
        raise AssertionError(f"unknown tensor mutation {name}")

    certify_args = dict(
        approval=actual_case.approval,
        readiness=actual_case.readiness,
        assembly_receipt=actual_case.assembly,
        source_payload=actual_case.payload,
    )
    for mutation_name in (
        "overall_sign",
        "complex_conjugation",
        "simultaneous_orbital_permutation",
    ):
        with monkeypatch.context() as patch:
            def mutated_builder(
                context: abc.Vituri2024TDHFAssemblyContext,
                crosswalk: tuple[
                    restricted.Vituri2024RestrictedScalarOrbitalCrosswalk, ...
                ],
                *,
                _mutation_name: str = mutation_name,
            ) -> tuple[np.ndarray, tuple[restricted.Vituri2024RestrictedScalarVertexBinding, ...]]:
                wbar, bindings = original_builder(context, crosswalk)
                changed = restricted._readonly_complex(
                    mutate(_mutation_name, wbar), ndim=4
                )
                assert np.max(np.abs(changed - wbar)) > 1.0e-8
                restricted._validate_raw_tensor_symmetries(changed)
                return changed, bindings

            patch.setattr(restricted, "_build_raw_tensor", mutated_builder)
            with pytest.raises(ValueError, match="actual C9"):
                abc.certify_vituri2024_tdhf_restricted_scalar(**certify_args)


def test_actual_B_mutations_fail_generic_scalar_approval_cert_path(
    actual_case: _ActualCase,
) -> None:
    prepared = restricted._prepare_oracle(
        actual_case.readiness,
        actual_case.assembly,
        actual_case.payload,
        provenance=actual_case.approval.provenance,
    )
    original = prepared.scalar_B_plus_minus
    order_mutation = original[:, ::-1]
    mutations = (
        ("overall sign", -original),
        ("complex conjugation", original.conj()),
        ("minus-pair order", order_mutation),
    )
    for label, mutation in mutations:
        changed = restricted._readonly_complex(mutation, ndim=2)
        assert np.max(np.abs(changed - original)) > 1.0e-8, label
        wrong_blocks = replace(
            actual_case.assembly.blocks,
            B_plus_minus=changed,
            B_minus_plus=restricted._readonly_complex(changed.T, ndim=2),
        )
        assert type(wrong_blocks) is TDHFSignedQBlocks
        wrong_sector = replace(actual_case.assembly.sector, blocks=wrong_blocks)
        wrong_approval = make_tdhf_scalar_curvature_approval(
            sector=wrong_sector,
            tangent_basis=prepared.tangent_basis,
            directions=prepared.generic_approval.directions,
            energy_callback=prepared.callback,
            functional_manifest=prepared.functional_manifest,
            energy_convention=prepared.generic_approval.convention,
            step_ladder=prepared.generic_approval.step_ladder,
            interaction_fingerprint=wrong_sector.interaction_fingerprint,
            provenance=f"test-local wrong signed-B {label} canary",
        )
        with pytest.raises(ValueError, match="scalar-curvature certification failed"):
            certify_tdhf_scalar_curvature(
                approval=wrong_approval,
                sector=wrong_sector,
                tangent_basis=prepared.tangent_basis,
                energy_callback=prepared.callback,
                functional_manifest=prepared.functional_manifest,
            )

    # The actual SAME-order B is not self-transpose, so directly omitting the
    # required partner transpose must fail the typed structure gate.
    assert np.max(np.abs(original - original.T)) > 1.0e-8
    wrong_partner_blocks = replace(
        actual_case.assembly.blocks,
        B_plus_minus=restricted._readonly_complex(original, ndim=2),
        B_minus_plus=restricted._readonly_complex(original, ndim=2),
    )
    wrong_partner_sector = replace(
        actual_case.assembly.sector, blocks=wrong_partner_blocks
    )
    with pytest.raises(ValueError, match="typed signed-q TDHF structure gate failed"):
        make_tdhf_scalar_curvature_approval(
            sector=wrong_partner_sector,
            tangent_basis=prepared.tangent_basis,
            directions=prepared.generic_approval.directions,
            energy_callback=prepared.callback,
            functional_manifest=prepared.functional_manifest,
            energy_convention=prepared.generic_approval.convention,
            step_ladder=prepared.generic_approval.step_ladder,
            interaction_fingerprint=wrong_partner_sector.interaction_fingerprint,
            provenance="test-local omitted B-partner transpose canary",
        )


def test_literal_component_fingerprints_and_decomposition_mutation_canaries(
    actual_case: _ActualCase,
) -> None:
    receipt = actual_case.receipt

    def changed(array: np.ndarray) -> np.ndarray:
        result = np.array(array, copy=True)
        result.flat[0] += 1.0e-5 + 2.0e-5j
        result.setflags(write=False)
        return result

    component_cases = (
        (
            "literal_one_body_hamiltonian_ev",
            "literal one-body Hamiltonian",
        ),
        (
            "literal_interaction_hamiltonian_ev",
            "literal interaction Hamiltonian",
        ),
        ("literal_hamiltonian_ev", "literal total Hamiltonian"),
    )
    for field_name, error_label in component_cases:
        original = getattr(receipt, field_name)
        object.__setattr__(receipt, field_name, changed(original))
        try:
            with pytest.raises(ValueError, match=error_label):
                _ = receipt.fingerprint
        finally:
            object.__setattr__(receipt, field_name, original)

    # Rehash the changed total so all three live-array fingerprint checks pass;
    # the exact one-body + interaction decomposition must still fail first.
    original_total = receipt.literal_hamiltonian_ev
    original_total_fingerprint = receipt.literal_hamiltonian_fingerprint
    mismatched_total = changed(original_total)
    object.__setattr__(receipt, "literal_hamiltonian_ev", mismatched_total)
    object.__setattr__(
        receipt,
        "literal_hamiltonian_fingerprint",
        restricted._array_sha256(mismatched_total),
    )
    try:
        with pytest.raises(ValueError, match="literal Hamiltonian decomposition mismatch"):
            _ = receipt.fingerprint
    finally:
        object.__setattr__(receipt, "literal_hamiltonian_ev", original_total)
        object.__setattr__(
            receipt,
            "literal_hamiltonian_fingerprint",
            original_total_fingerprint,
        )

    original_one_body = receipt.literal_one_body_hamiltonian_ev
    writable_one_body = np.array(original_one_body, copy=True)
    object.__setattr__(
        receipt, "literal_one_body_hamiltonian_ev", writable_one_body
    )
    try:
        with pytest.raises(ValueError, match="must be read-only"):
            _ = receipt.fingerprint
    finally:
        object.__setattr__(
            receipt, "literal_one_body_hamiltonian_ev", original_one_body
        )

    object.__setattr__(
        receipt,
        "literal_one_body_hamiltonian_ev",
        restricted._readonly_complex(original_one_body[:-1, :], ndim=2),
    )
    try:
        with pytest.raises(ValueError, match="shape drift"):
            _ = receipt.fingerprint
    finally:
        object.__setattr__(
            receipt, "literal_one_body_hamiltonian_ev", original_one_body
        )
    assert receipt.fingerprint == receipt.receipt_fingerprint


def test_tensor_repair_is_rejected_and_nonconserving_near_zero_is_exact_zero(
    actual_case: _ActualCase,
) -> None:
    tampered = np.array(actual_case.receipt.wbar_tensor_ev, copy=True)
    tampered[0, 1, 2, 3] += 1.0e-4 + 2.0e-4j
    with pytest.raises(ValueError, match="repair is prohibited"):
        restricted._validate_raw_tensor_symmetries(tampered)

    flavor = abc.Vituri2024Flavor(valley=1, spin=1)
    exact = abc.Vituri2024Orbital(flavor, (0.0, 0.0))
    near = abc.Vituri2024Orbital(flavor, (1.0e-14, 0.0))
    assert not restricted._exact_local_conserving(exact, exact, exact, near)

    bound_quartets = {item.quartet for item in actual_case.receipt.conserving_vertices}
    found_nonconserving = False
    orbitals = tuple(item.orbital for item in actual_case.receipt.orbital_crosswalk)
    for quartet in np.ndindex(actual_case.receipt.wbar_tensor_ev.shape):
        physical = tuple(orbitals[index] for index in quartet)
        if not restricted._exact_local_conserving(*physical):
            found_nonconserving = True
            assert actual_case.receipt.wbar_tensor_ev[quartet] == 0.0j
            assert quartet not in bound_quartets
    assert found_nonconserving


def test_source_orbital_energy_occupation_context_and_readiness_drift_fail_closed(
    actual_case: _ActualCase,
) -> None:
    approval = actual_case.approval
    args = dict(
        approval=approval,
        readiness=actual_case.readiness,
        assembly_receipt=actual_case.assembly,
        source_payload=actual_case.payload,
    )
    energies = np.array(actual_case.payload.energies, copy=True)
    energies.flat[0] += 1.0e-4
    occupations = np.array(actual_case.payload.occupations, copy=True)
    occupations.flat[0] = 1 - occupations.flat[0]
    mutations = (
        (actual_case.payload, "source_state_sha256", "f" * 64),
        (
            actual_case.assembly,
            "orbital_id_map",
            tuple(reversed(actual_case.assembly.orbital_id_map)),
        ),
        (actual_case.payload, "energies", energies),
        (actual_case.payload, "occupations", occupations),
        (actual_case.assembly, "assembly_context_fingerprint", "e" * 64),
        (actual_case.readiness, "static_hessian_authority", "mutated"),
        (
            actual_case.assembly.sector,
            "static_hessian_authority",
            "scalar_hessian",
        ),
    )
    for target, name, mutation in mutations:
        original = getattr(target, name)
        object.__setattr__(target, name, mutation)
        try:
            with pytest.raises((TypeError, ValueError)):
                abc.certify_vituri2024_tdhf_restricted_scalar(**args)
        finally:
            object.__setattr__(target, name, original)


def test_orbital_cap_precedes_n4_and_fock_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    allocation_called = False

    def forbidden_zeros(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal allocation_called
        allocation_called = True
        raise AssertionError("allocation happened before orbital cap")

    monkeypatch.setattr(restricted.np, "zeros", forbidden_zeros)
    with pytest.raises(ValueError, match=r"before N\^4/Fock allocation"):
        restricted._enforce_orbital_cap(9)
    assert not allocation_called


def test_stale_approval_and_generic_manifest_fail_before_certification(
    actual_case: _ActualCase,
) -> None:
    args = dict(
        approval=actual_case.approval,
        readiness=actual_case.readiness,
        assembly_receipt=actual_case.assembly,
        source_payload=actual_case.payload,
    )
    original_manifest = actual_case.approval.deterministic_manifest_sha256
    object.__setattr__(
        actual_case.approval, "deterministic_manifest_sha256", "a" * 64
    )
    try:
        with pytest.raises(ValueError, match="approval manifest|fingerprint"):
            abc.certify_vituri2024_tdhf_restricted_scalar(**args)
    finally:
        object.__setattr__(
            actual_case.approval,
            "deterministic_manifest_sha256",
            original_manifest,
        )

    manifest = actual_case.approval.generic_approval.functional_manifest
    original_source = manifest.source_functional_fingerprint
    object.__setattr__(manifest, "source_functional_fingerprint", "b" * 64)
    try:
        with pytest.raises(ValueError, match="manifest|fingerprint"):
            abc.certify_vituri2024_tdhf_restricted_scalar(**args)
    finally:
        object.__setattr__(manifest, "source_functional_fingerprint", original_source)


def test_receipt_is_factory_only_and_authority_mutation_is_detected(
    actual_case: _ActualCase,
) -> None:
    with pytest.raises(TypeError, match="private factory token"):
        replace(actual_case.receipt, _factory_token=object())

    original = actual_case.receipt.authority
    object.__setattr__(actual_case.receipt, "authority", "global_scalar_authority")
    try:
        with pytest.raises(ValueError, match="authority"):
            _ = actual_case.receipt.fingerprint
    finally:
        object.__setattr__(actual_case.receipt, "authority", original)
    assert actual_case.assembly.sector.static_hessian_authority == "projected_signed_ab"
