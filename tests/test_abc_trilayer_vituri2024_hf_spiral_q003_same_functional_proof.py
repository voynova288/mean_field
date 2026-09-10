"""Focused reduced tests for the candidate-only q003 proof layer."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass, replace

import numpy as np
import pytest

from mean_field.systems import abc_trilayer
from mean_field.systems.abc_trilayer import (
    vituri2024_hf_spiral_q003_same_functional_proof as proof,
)
from mean_field.systems.abc_trilayer.vituri2024_hf import (
    vituri2024_conventional_k_diagonal_to_native_density,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_preflight import (
    INTERNAL_FLAVOR_ORDER,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_scf import (
    Vituri2024CartesianHFSpec,
    prepare_vituri2024_homogeneous_hf_fft,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral import (
    Vituri2024FiniteQSpiralChoice,
    prepare_vituri2024_hf_spiral,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_full_hessian import (
    build_vituri2024_hf_spiral_full_hessian_context,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_full_response import (
    build_vituri2024_hf_spiral_signed_displacement_response,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_full_stability import (
    build_vituri2024_hf_spiral_full_sector_inventory,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_stability import (
    prepare_vituri2024_hf_spiral_stability,
)


def _reduced_context():
    base = prepare_vituri2024_homogeneous_hf_fft(
        Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=3)
    )
    prepared = prepare_vituri2024_hf_spiral(
        base,
        Vituri2024FiniteQSpiralChoice(
            q_inverse_angstrom=np.zeros(2, dtype=np.float64),
            selected_spin=1,
            gauge_mode="identity",
        ),
    )
    selected = tuple(
        index
        for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin == prepared.choice.selected_spin
    )
    spectator = tuple(
        index
        for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin == -prepared.choice.selected_spin
    )
    occupations = np.asarray(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=np.float64,
    )
    conventional = np.zeros((4, 4, prepared.nk), dtype=np.complex128)
    conventional[selected[0], selected[0]] = occupations[:, 0]
    conventional[selected[1], selected[1]] = occupations[:, 1]
    momenta = np.arange(prepared.nk, dtype=np.int64)
    conventional[
        np.ix_(
            np.asarray(spectator, dtype=np.int64),
            np.asarray(spectator, dtype=np.int64),
            momenta,
        )
    ] = np.repeat(np.eye(2, dtype=np.complex128)[:, :, None], prepared.nk, axis=2)
    density = vituri2024_conventional_k_diagonal_to_native_density(conventional)
    restricted = prepare_vituri2024_hf_spiral_stability(
        prepared, density, prepared.functional.fock(density)
    )
    inventory = build_vituri2024_hf_spiral_full_sector_inventory(restricted)
    response = build_vituri2024_hf_spiral_signed_displacement_response(inventory)
    return build_vituri2024_hf_spiral_full_hessian_context(response)


@pytest.fixture(scope="module")
def reduced_context():
    return _reduced_context()


def test_streaming_exact_integer_inventory_computes_every_j_hypothesis(
    reduced_context,
    monkeypatch,
) -> None:
    def forbidden_embedding(*_args, **_kwargs):
        raise AssertionError("production _build_embedding must not be called")

    monkeypatch.setattr(type(reduced_context), "_build_embedding", forbidden_embedding)
    audit = proof._audit_transition_geometry(reduced_context.inventory)
    proof._validate_audit(audit, enforce_q003=False)
    assert audit.selected_occupied_count == 12
    assert audit.selected_virtual_count == 6
    assert audit.transition_count == 72
    assert audit.nonempty_signed_lane_count == reduced_context.inventory.nonempty_sector_count
    assert audit.canonical_orbit_count == reduced_context.inventory.nonempty_conjugate_orbit_count
    assert audit.nonempty_signed_lanes_checked == audit.nonempty_signed_lane_count == 49
    assert audit.zero_self_conjugate_transition_count == 0
    assert audit.sector_dimension_mismatch_count == 0
    assert audit.occupation_mismatch_count == 0
    assert audit.inverse_sector_mismatch_count == 0
    assert audit.inverse_vertex_mismatch_count == 0
    assert audit.reverse_occupation_collision_count == 0
    assert audit.tuple_order_mismatch_count == 0
    assert len(audit.canonical_tuple_stream_sha256) == 64
    assert len(audit.canonical_vertex_stream_sha256) == 64


def test_transition_audit_validates_inventory_once(reduced_context, monkeypatch) -> None:
    inventory_type = type(reduced_context.inventory)
    original = inventory_type.validate_live_state
    calls = 0

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(inventory_type, "validate_live_state", counted)
    audit = proof._audit_transition_geometry(reduced_context.inventory)
    assert audit.transition_count == 72
    assert calls == 1


def test_reduced_actual_production_embedding_order_and_one_empty_lanes(
    reduced_context,
) -> None:
    comparison = proof._reduced_actual_embedding_comparison(reduced_context)
    assert comparison.nonempty_signed_lanes_checked == 49
    assert comparison.empty_partner_lanes_skipped == 13
    assert comparison.one_empty_orbits_checked == 13
    assert comparison.transition_dimensions_summed == 72
    assert comparison.support_mismatch_count == 0
    assert comparison.flavor_block_mismatch_count == 0
    assert comparison.order_or_extraction_mismatch_count == 0


def test_reduced_exhaustive_canonical_vertex_oracle_is_independent_of_embedding(
    reduced_context,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        type(reduced_context),
        "_build_embedding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("production embedding was called")
        ),
    )
    assert proof._reduced_exhaustive_canonical_vertex_oracle(
        reduced_context.inventory
    ) == 72


def test_real_inner_product_jdagger_formula_holds_for_every_reduced_root(
    reduced_context,
) -> None:
    rows = np.concatenate(tuple(proof._transition_chunks(reduced_context.inventory)))
    roots = np.unique(rows[:, 7:10], axis=0)
    rng = np.random.default_rng(20260910)
    nk = reduced_context.nk
    checked = 0
    for root in roots:
        selected = rows[np.all(rows[:, 7:10] == root, axis=1)]
        if selected.size == 0:
            continue
        first = selected[selected[:, 10] == 0]
        second = selected[selected[:, 10] == 1]
        x = rng.normal(size=len(first)) + 1j * rng.normal(size=len(first))
        y = rng.normal(size=len(second)) + 1j * rng.normal(size=len(second))
        canonical_block = np.zeros((2, 2, nk), dtype=np.complex128)
        if len(first):
            canonical_block[first[:, 11], first[:, 12], first[:, 13]] = x
        if len(second):
            assert not np.any(
                canonical_block[second[:, 11], second[:, 12], second[:, 13]]
            )
            canonical_block[second[:, 11], second[:, 12], second[:, 13]] = y.conj()
        test_block = rng.normal(size=(2, 2, nk)) + 1j * rng.normal(
            size=(2, 2, nk)
        )
        jdagger_first = (
            test_block[first[:, 11], first[:, 12], first[:, 13]]
            if len(first)
            else np.empty(0, dtype=np.complex128)
        )
        jdagger_second = (
            test_block[second[:, 11], second[:, 12], second[:, 13]].conj()
            if len(second)
            else np.empty(0, dtype=np.complex128)
        )
        codomain_inner = float(np.vdot(canonical_block, test_block).real)
        conjugate_block = canonical_block.swapaxes(0, 1).conj()
        conjugate_test = test_block.swapaxes(0, 1).conj()
        half_pair_inner = float(
            0.5
            * (
                np.vdot(canonical_block, test_block)
                + np.vdot(conjugate_block, conjugate_test)
            ).real
        )
        domain_inner = float(
            (np.vdot(x, jdagger_first) + np.vdot(y, jdagger_second)).real
        )
        assert half_pair_inner == pytest.approx(codomain_inner, abs=2.0e-13)
        assert codomain_inner == pytest.approx(domain_inner, abs=2.0e-13)
        checked += len(selected)
    assert checked == 72


def test_reduced_complete_real_basis_matches_actual_callback_term_by_term(
    reduced_context,
) -> None:
    comparison = proof._reduced_complete_basis_term_comparison(reduced_context)
    assert comparison.complete_real_basis_vectors_checked == 144
    assert comparison.nonempty_signed_lanes_checked == 49
    assert comparison.one_empty_orbits_checked == 13
    assert comparison.support_exact
    assert comparison.flavor_blocks_exact
    assert comparison.production_embedding_order_extraction_exact
    assert comparison.direct_component_within_tolerance
    assert comparison.exchange_component_within_tolerance
    assert comparison.area_normalization_within_tolerance
    assert comparison.interaction_sign_within_tolerance
    for implementation in ("production", "scalar"):
        for component in ("total", "direct", "exchange"):
            assert getattr(
                comparison,
                f"{implementation}_{component}_signed_lane_covariance_within_tolerance",
            )
            assert getattr(
                comparison,
                f"maximum_{implementation}_{component}_signed_lane_covariance_residual",
            ) < 5.0e-12
    assert comparison.maximum_total_residual < 5.0e-12
    assert comparison.maximum_direct_residual < 5.0e-12
    assert comparison.maximum_exchange_residual < 5.0e-12
    comparison_fields = comparison.__dataclass_fields__
    for stale_name in (
        "direct_component_exact",
        "exchange_component_exact",
        "area_normalization_exact",
        "interaction_sign_exact",
        "signed_conjugation_exact",
    ):
        assert stale_name not in comparison_fields


def test_exact_lane_and_dimension_validation_is_equality_not_nominal_slots(
    reduced_context,
) -> None:
    audit = proof._audit_transition_geometry(reduced_context.inventory)
    proof._validate_audit(audit, enforce_q003=False)
    with pytest.raises(ValueError, match="transition/J hypotheses"):
        proof._validate_audit(
            replace(
                audit,
                nonempty_signed_lanes_checked=audit.nonempty_signed_lane_count + 1,
            ),
            enforce_q003=False,
        )
    assert proof._EXPECTED_NONEMPTY == 38_259
    assert proof._EXPECTED_COMPLEX == 15_052_040
    assert proof._EXPECTED_REAL == 30_104_080


def test_term_table_is_callable_bound_and_not_fingerprint_only() -> None:
    table = proof._formula_correspondence_table()
    assert tuple(row[0] for row in table) == (
        "support",
        "flavor_blocks",
        "transition_geometry",
        "direct_rank_one",
        "exchange",
        "area",
        "pair_geometry",
        "real_hessian",
    )
    assert all(row[-2] is True for row in table)
    assert all(row[-1] is False for row in table)
    records = proof._callable_source_records()
    constants = proof._formula_constant_records()
    assert len(records) == 14
    assert tuple(record[0] for record in constants) == (
        "response_kernel_contract",
        "fft_no_wrap_policy",
        "bridge_interaction_identity",
        "bridge_interaction_derivation",
        "scalar_candidate_authority",
    )
    assert {record[0] for record in records} == {
        label for label, _owner, _attribute in proof._FORMULA_CALLABLE_SPECS
    }
    assert all(len(record[3]) == 64 for record in records)


def test_recorded_asymmetric_kernel_orientation_derivation_is_proof_local(
    monkeypatch,
) -> None:
    derivation = dict(proof._recorded_asymmetric_kernel_orientation_derivation())
    assert derivation["direct_expected_K_minus_d"] == 9_916
    assert derivation["direct_production_component"] == 9_916
    assert derivation["direct_scalar_component"] == 9_916
    assert derivation["direct_wrong_K_plus_d"] == 10_084
    assert derivation["exchange_expected_minus_K_output_minus_source"] == -9_916
    assert derivation["exchange_production_component"] == -9_916
    assert derivation["exchange_scalar_component"] == -9_916
    assert derivation["exchange_wrong_minus_K_source_minus_output"] == -10_084
    assert derivation["evenized_direct_bridge"] == 10_000
    assert derivation["evenized_exchange_bridge"] == -10_000
    direct_wrong = lambda kernel, displacement: kernel(*displacement)
    exchange_wrong = lambda kernel, output, source: -kernel(
        source[0] - output[0], source[1] - output[1]
    )
    for name in (
        "_production_direct_kernel_component",
        "_scalar_direct_kernel_component",
    ):
        with monkeypatch.context() as isolated:
            isolated.setattr(proof, name, direct_wrong)
            with pytest.raises(RuntimeError, match="direct component orientation regressed"):
                proof._recorded_asymmetric_kernel_orientation_derivation()
    for name in (
        "_production_exchange_kernel_component",
        "_scalar_exchange_kernel_component",
    ):
        with monkeypatch.context() as isolated:
            isolated.setattr(proof, name, exchange_wrong)
            with pytest.raises(
                RuntimeError, match="exchange component orientation or sign regressed"
            ):
                proof._recorded_asymmetric_kernel_orientation_derivation()
    assert "real_even" in proof._response_module.VITURI2024_HF_SPIRAL_FULL_RESPONSE_KERNEL_CONTRACT


def test_source_callable_rebinding_fails_before_builder_inputs(monkeypatch) -> None:
    original_builder = proof.build_vituri2024_q003_same_functional_candidate_proof_receipt
    original = proof._response_module._direct_rank_one_numerator

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(proof._response_module, "_direct_rank_one_numerator", replacement)
    with pytest.raises(RuntimeError, match="formula binding drifted"):
        original_builder(
            context_3307=None,
            context_40fd=None,
            same_functional_bridge_receipt=None,
        )


def test_public_builder_rejects_transitive_method_and_module_rebasing(monkeypatch) -> None:
    original_builder = proof.build_vituri2024_q003_same_functional_candidate_proof_receipt
    original_method = proof._hessian_module.Vituri2024HFSpiralFullHessianContext.validate_live_state

    def replacement(self):
        return original_method(self)

    monkeypatch.setattr(
        proof._hessian_module.Vituri2024HFSpiralFullHessianContext,
        "validate_live_state",
        replacement,
    )
    with pytest.raises(RuntimeError, match="transitive binding drifted"):
        original_builder(
            context_3307=None,
            context_40fd=None,
            same_functional_bridge_receipt=None,
        )
    monkeypatch.setattr(
        proof._hessian_module.Vituri2024HFSpiralFullHessianContext,
        "validate_live_state",
        original_method,
    )
    monkeypatch.setitem(
        proof.sys.modules,
        proof._hessian_module.__name__,
        object(),
    )
    with pytest.raises(RuntimeError, match="module binding drifted"):
        original_builder(
            context_3307=None,
            context_40fd=None,
            same_functional_bridge_receipt=None,
        )


def test_closure_token_does_not_validate_directly_constructed_receipt() -> None:
    token = proof._require_factory.__closure__[0].cell_contents
    forged = proof.Vituri2024Q003SameFunctionalCandidateProofReceipt(
        _factory_token=token,
        sources=(),
        bridge_receipt_fingerprint="0" * 64,
        callable_source_records=proof._callable_source_records(),
        formula_constant_records=proof._formula_constant_records(),
        formula_correspondence_table=proof._formula_correspondence_table(),
        recorded_asymmetric_kernel_orientation_derivation=(
            proof._recorded_asymmetric_kernel_orientation_derivation()
        ),
        implementation_fingerprint=proof.vituri2024_q003_same_functional_proof_implementation_fingerprint(),
    )
    with pytest.raises(ValueError, match="identity-registered factory product"):
        forged.validate_live_state()
    with pytest.raises(TypeError):
        forged.validate_live_state(_registered=lambda *_args: True)
    assert "unforge" not in proof.VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_AUTHORITY
    assert "hostile" not in proof.VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_AUTHORITY


def test_public_builder_closure_rejects_bridge_receipt_fake_class_rebind(
    monkeypatch,
) -> None:
    original_builder = proof.build_vituri2024_q003_same_functional_candidate_proof_receipt
    original_type = proof._bridge_module.Vituri2024Q003SameFunctionalStructuralReceipt

    class FakeStructuralReceipt:
        validate_live_state = original_type.validate_live_state

    monkeypatch.setattr(
        proof._bridge_module,
        "Vituri2024Q003SameFunctionalStructuralReceipt",
        FakeStructuralReceipt,
    )
    with pytest.raises(RuntimeError, match="bridge receipt type binding drifted"):
        original_builder(
            context_3307=None,
            context_40fd=None,
            same_functional_bridge_receipt=None,
        )


def test_public_builder_closure_rejects_binding_table_rebase(monkeypatch) -> None:
    original_builder = proof.build_vituri2024_q003_same_functional_candidate_proof_receipt
    monkeypatch.setattr(
        proof, "_IMPORT_LOCAL_FUNCTIONS", tuple(list(proof._IMPORT_LOCAL_FUNCTIONS))
    )
    with pytest.raises(RuntimeError, match="function table drifted"):
        original_builder(
            context_3307=None,
            context_40fd=None,
            same_functional_bridge_receipt=None,
        )


def test_signed_lane_covariance_is_explicit_hypotheses_not_authority() -> None:
    theorem = proof.VITURI2024_Q003_REAL_ADJOINT_THEOREM
    for hypothesis in ("H3_", "H5_", "H6_direct", "H7_exchange", "H8_Sigma"):
        assert hypothesis in theorem
    assert "recorded_candidate_derivation" in theorem
    assert "not_an_established_production_J_identity" in theorem


def test_source_record_normalizes_numpy_float64_area_before_serialization(
    reduced_context,
    monkeypatch,
) -> None:
    # job498493 completed the full proof workload and then failed only at
    # receipt serialization because a NumPy float64 crossed this boundary.
    # This records an engineering serialization failure, not new authority.
    response_type = type(reduced_context.response)
    original_area_property = response_type.area_angstrom_squared

    class BridgeSource:
        group_label = "reduced"
        context_fingerprint = reduced_context.context_fingerprint
        inventory_fingerprint = reduced_context.inventory.inventory_fingerprint
        response_fingerprint = reduced_context.response.response_fingerprint

    captured: dict[str, object] = {}

    def capture_factory(owner, /, **kwargs):
        assert owner is proof.Vituri2024Q003SameFunctionalProofSourceRecord
        captured.update(kwargs)
        return kwargs

    with monkeypatch.context() as isolated:
        isolated.setattr(
            response_type,
            "area_angstrom_squared",
            property(
                lambda self: np.float64(original_area_property.fget(self))
            ),
        )
        isolated.setattr(proof, "_validate_audit", lambda *_args, **_kwargs: None)
        proof._build_source_record(
            reduced_context,
            "reduced",
            BridgeSource(),
            capture_factory,
        )

    integer_fields = (
        "selected_occupied_count",
        "selected_virtual_count",
        "transition_count",
        "complex_dimension",
        "real_dimension",
        "nonempty_signed_lane_count",
        "canonical_orbit_count",
        "nonempty_signed_lanes_checked",
    )
    floating_fields = (
        "area_angstrom_squared",
        "maximum_kernel_imaginary_residual",
        "maximum_kernel_evenness_residual",
        "kernel_tolerance",
    )
    assert all(type(captured[name]) is int for name in integer_fields)
    assert all(type(captured[name]) is float for name in floating_fields)
    assert captured["area_angstrom_squared"] > 0.0

    closure = dict(
        zip(
            proof.build_vituri2024_q003_same_functional_candidate_proof_receipt.__code__.co_freevars,
            proof.build_vituri2024_q003_same_functional_candidate_proof_receipt.__closure__,
        )
    )
    factory = closure["factory"].cell_contents
    source = factory(proof.Vituri2024Q003SameFunctionalProofSourceRecord, **captured)
    source.validate_live_state()

    token = proof._require_factory.__closure__[0].cell_contents
    with pytest.raises(TypeError, match="exact Python floats"):
        proof.Vituri2024Q003SameFunctionalProofSourceRecord(
            _factory_token=token,
            **{
                **captured,
                "area_angstrom_squared": np.float64(
                    captured["area_angstrom_squared"]
                ),
            },
        )
    for invalid_area in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite|positive"):
            proof.Vituri2024Q003SameFunctionalProofSourceRecord(
                _factory_token=token,
                **{**captured, "area_angstrom_squared": invalid_area},
            )
    with pytest.raises(TypeError, match="non-JSON value"):
        proof._json_native(np.float64(captured["area_angstrom_squared"]))

    object.__setattr__(
        source,
        "area_angstrom_squared",
        np.float64(source.area_angstrom_squared),
    )
    with pytest.raises(ValueError, match="scalar live state drifted"):
        source.validate_live_state()


def test_candidate_receipt_schema_is_array_free_and_every_promotion_is_false() -> None:
    receipt_type = proof.Vituri2024Q003SameFunctionalCandidateProofReceipt
    receipt_fields = receipt_type.__dataclass_fields__
    source_fields = proof.Vituri2024Q003SameFunctionalProofSourceRecord.__dataclass_fields__
    assert "complex_dimension" in source_fields
    assert "real_dimension" in source_fields
    assert "nonempty_signed_lanes_checked" in source_fields
    assert "signed_lane_slots_checked" not in source_fields
    assert receipt_fields["candidate_only"].default is True
    assert receipt_fields["exact_integer_transition_enumeration_completed"].default is True
    assert (
        receipt_fields["independent_inverse_extraction_hypothesis_checked"].default
        is True
    )
    assert (
        receipt_fields[
            "independent_real_inner_product_geometry_hypothesis_checked"
        ].default
        is True
    )
    assert "inverse_extraction_checked" not in receipt_fields
    assert "real_inner_product_geometry_computed" not in receipt_fields
    assert receipt_fields["candidate_j_jdagger_adjoint_derivation_recorded"].default is True
    assert receipt_fields["candidate_signed_lane_covariance_derivation_recorded"].default is True
    assert receipt_fields["candidate_formula_correspondence_derivation_recorded"].default is True
    assert receipt_fields["exact_formula_constants_bound"].default is True
    assert receipt_fields["candidate_common_formula_inputs_derivation_recorded"].default is True
    assert "recorded_asymmetric_kernel_orientation_derivation" in receipt_fields
    assert "asymmetric_kernel_orientation_canary" not in receipt_fields
    assert receipt_fields["implementation_canary_passed"].default is False
    assert "asymmetric_kernel_orientation_canary_passed" not in receipt_fields
    assert receipt_fields["equality_claimed_from_fingerprints_alone"].default is False
    assert receipt_fields["production_build_embedding_called"].default is False
    for name in (
        "full_q003_production_embedding_enumeration_completed",
        "production_j_identity_computed",
        "signed_lane_covariance_established",
        "formula_correspondence_established",
        "detached_theorem_review_completed",
        "j_jdagger_adjoint_theorem_authority_established",
        "l_prod_equals_sigma_authority_established",
        "same_functional_scalar_hessian_authority_established",
        "full_inventory_exact_unitary_curvature_established",
        "literal_float_full_functional_parity_established",
        "linear_operator_authorized",
        "hermitian_eigensolver_authorized",
        "full_local_stability_established",
        "scientific_authority_promoted",
        "production_ready",
        "paper_reproduction_verified",
    ):
        assert receipt_fields[name].default is False
    assert receipt_fields["receipt_fingerprint"].default is MISSING
    assert all("context" not in item.name for item in fields(receipt_type))
    assert all("array" not in item.name for item in fields(receipt_type))
    for forbidden in ("matvec", "rmatvec", "linear_operator", "eigenvalues", "action"):
        assert not hasattr(receipt_type, forbidden)
        assert forbidden not in proof.__all__
    assert is_dataclass(receipt_type)


def test_receipt_types_are_hidden_and_public_exports_are_closed() -> None:
    assert "Vituri2024Q003SameFunctionalCandidateProofReceipt" not in proof.__all__
    assert "Vituri2024Q003SameFunctionalProofSourceRecord" not in proof.__all__
    assert not hasattr(abc_trilayer, "Vituri2024Q003SameFunctionalCandidateProofReceipt")
    assert not hasattr(abc_trilayer, "Vituri2024Q003SameFunctionalProofSourceRecord")
    assert not hasattr(proof, "_construct_receipt")
    for name in proof.__all__:
        assert name in abc_trilayer.__all__
        assert getattr(abc_trilayer, name) is getattr(proof, name)


def test_implementation_fingerprint_is_stable_and_candidate_only() -> None:
    first = proof.vituri2024_q003_same_functional_proof_implementation_fingerprint()
    second = proof.vituri2024_q003_same_functional_proof_implementation_fingerprint()
    assert first == second
    assert len(first) == 64
    assert "candidate_only" in proof.VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_AUTHORITY
    assert "pending_detached" in proof.VITURI2024_Q003_HESSIAN_IDENTITY_CANDIDATE
