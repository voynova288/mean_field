"""Focused fail-closed tests for the q003 same-functional structural bridge."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import numpy as np
import pytest

from mean_field.systems import abc_trilayer
from mean_field.systems.abc_trilayer import (
    vituri2024_hf_spiral_q003_same_functional_bridge as bridge,
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

ROOT = Path(__file__).resolve().parents[1]
LINEAGE_PATHS = {
    "stationarity_attestation": ROOT
    / "reports/data/vituri2024_fig2_q003_dense_oracle_replay_477106_attestation.json",
    "source_fock_closure_attestation": ROOT
    / "reports/data/vituri2024_fig2_q003_source_fock_closure_487653_attestation.json",
    "exact_unitary_step_ladder_attestation": ROOT
    / "reports/data/vituri2024_fig2_q003_exact_unitary_step_ladder_496930_attestation.json",
}


def _lineage_bytes() -> dict[str, bytes]:
    return {name: path.read_bytes() for name, path in LINEAGE_PATHS.items()}


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
        prepared,
        density,
        prepared.functional.fock(density),
    )
    inventory = build_vituri2024_hf_spiral_full_sector_inventory(restricted)
    response = build_vituri2024_hf_spiral_signed_displacement_response(inventory)
    return build_vituri2024_hf_spiral_full_hessian_context(response)


@pytest.fixture(scope="module")
def reduced_context():
    return _reduced_context()


def test_formula_fingerprint_covers_both_implementations_and_core_matvec() -> None:
    first = bridge.vituri2024_q003_same_functional_formula_implementation_fingerprint()
    second = bridge.vituri2024_q003_same_functional_formula_implementation_fingerprint()
    assert first == second
    assert len(first) == 64
    labels = tuple(record[0] for record in bridge._formula_source_records())
    assert labels == (
        "response_direct_rank_one",
        "response_exchange_mdagger_ck_m",
        "response_support",
        "response_flavor_blocks",
        "response_direct_wrapper",
        "response_validate_signed_block",
        "response_fft_action",
        "response_validated_action_call",
        "response_prepare_action",
        "hessian_paired_injection",
        "hessian_transition_embedding",
        "hessian_orbit_builder",
        "core_unpack_real",
        "core_pack_complex",
        "core_lane_one_body_gaps",
        "core_paired_gradient_action",
        "core_paired_matvec",
        "inventory_iter_sectors",
        "inventory_iter_orbits",
        "exact_integer_support",
        "exact_integer_flavor_blocks",
        "exact_integer_sigma_action",
        "exact_integer_source_fock",
        "exact_integer_interaction_trace",
        "exact_integer_scalar_curvature",
        "exact_integer_unitary_scalar",
    )


def test_stale_formula_function_binding_is_rejected(monkeypatch) -> None:
    original = bridge._core_module.PairedSectorOrbitalHessian.matvec

    def stale_matvec(self, vector):
        return original(self, vector)

    monkeypatch.setattr(
        bridge._core_module.PairedSectorOrbitalHessian,
        "matvec",
        stale_matvec,
    )
    with pytest.raises(RuntimeError, match="class descriptor binding drifted"):
        bridge.vituri2024_q003_same_functional_formula_implementation_fingerprint()


def test_uncatalogued_transitive_production_method_rebinding_is_rejected(
    monkeypatch,
) -> None:
    owner = bridge._response_module.Vituri2024HFSpiralSignedDisplacementResponse
    original = owner._make_validated_fft_action_unchecked

    def stale_make_action(self, key):
        return original(self, key)

    monkeypatch.setattr(owner, "_make_validated_fft_action_unchecked", stale_make_action)
    with pytest.raises(RuntimeError, match="owner class descriptor binding drifted"):
        bridge.vituri2024_q003_same_functional_formula_implementation_fingerprint()


def test_stale_source_bound_reciprocity_validator_is_rejected(monkeypatch) -> None:
    original_builder = bridge.build_vituri2024_q003_same_functional_structural_receipt
    owner = bridge.Vituri2024Q003SourceBoundReciprocityCertificate
    original = owner.validate_live_state

    def stale_validate_live_state(self):
        return original(self)

    monkeypatch.setattr(owner, "validate_live_state", stale_validate_live_state)
    with pytest.raises(
        RuntimeError,
        match=(
            "imported runtime binding drifted: "
            "Vituri2024Q003SourceBoundReciprocityCertificate.validate_live_state"
        ),
    ):
        original_builder(
            context_3307=None,
            independent_source_fock_3307=None,
            context_40fd=None,
            independent_source_fock_40fd=None,
            reciprocity_certificate=None,
            **_lineage_bytes(),
        )


def test_transitive_source_bound_reciprocity_guard_rebinding_is_rejected(
    monkeypatch,
) -> None:
    original_builder = bridge.build_vituri2024_q003_same_functional_structural_receipt
    original = bridge._reciprocity_module._validate_live_bindings

    def stale_validate_live_bindings():
        return original()

    monkeypatch.setattr(
        bridge._reciprocity_module,
        "_validate_live_bindings",
        stale_validate_live_bindings,
    )
    with pytest.raises(RuntimeError, match="owner callable inventory drifted"):
        original_builder(
            context_3307=None,
            independent_source_fock_3307=None,
            context_40fd=None,
            independent_source_fock_40fd=None,
            reciprocity_certificate=None,
            **_lineage_bytes(),
        )


@pytest.mark.parametrize(
    "constant_name",
    (
        "ARTIFACT_ROLES",
        "REVIEWED_CAPSULE_MEMBERS",
        "_PINNED_ARTIFACT_SHA256",
        "_PINNED_MEMBER_SHA256",
        "_PINNED_SOURCE_COMMIT",
        "_PINNED_HISTORICAL_IMPLEMENTATION",
        "_PINNED_GROUPS",
        "_PINNED_CERTIFICATE_GROUP_FINGERPRINTS",
        "_REVIEW_GATES",
        "_STRUCTURAL_GATES",
        "_FALSE_AUTHORITIES",
        "PINNED_Q003_RECIPROCITY_CERTIFIER_SHA256",
        "PINNED_Q003_RECIPROCITY_CERTIFIER_REVIEW_SHA256",
    ),
)
def test_reciprocity_constant_and_snapshot_double_rebinding_is_rejected(
    monkeypatch,
    constant_name: str,
) -> None:
    original_builder = bridge.build_vituri2024_q003_same_functional_structural_receipt
    original_formula_guard = (
        bridge.vituri2024_q003_same_functional_formula_implementation_fingerprint
    )
    assert tuple(
        name for name, _value in bridge._IMPORT_RECIPROCITY_LIVE_CONSTANT_BINDINGS
    ) == bridge._RECIPROCITY_LIVE_CONSTANT_NAMES

    replacement = object()
    monkeypatch.setattr(bridge._reciprocity_module, constant_name, replacement)
    rebased_snapshot = tuple(
        (name, replacement if name == constant_name else value)
        for name, value in bridge._IMPORT_RECIPROCITY_LIVE_CONSTANT_BINDINGS
    )
    monkeypatch.setattr(
        bridge,
        "_IMPORT_RECIPROCITY_LIVE_CONSTANT_BINDINGS",
        rebased_snapshot,
    )

    with pytest.raises(RuntimeError, match="reciprocity constant table drifted"):
        original_formula_guard()
    with pytest.raises(RuntimeError, match="reciprocity constant table drifted"):
        original_builder(
            context_3307=None,
            independent_source_fock_3307=None,
            context_40fd=None,
            independent_source_fock_40fd=None,
            reciprocity_certificate=None,
            **_lineage_bytes(),
        )


def test_source_bound_reciprocity_module_alias_proxy_is_rejected(monkeypatch) -> None:
    original_builder = bridge.build_vituri2024_q003_same_functional_structural_receipt
    original = bridge._reciprocity_module

    class ForwardingProxy:
        def __getattr__(self, name: str):
            return getattr(original, name)

    monkeypatch.setattr(bridge, "_reciprocity_module", ForwardingProxy())
    with pytest.raises(
        RuntimeError, match="local alias drifted: _reciprocity_module"
    ):
        original_builder(
            context_3307=None,
            independent_source_fock_3307=None,
            context_40fd=None,
            independent_source_fock_40fd=None,
            reciprocity_certificate=None,
            **_lineage_bytes(),
        )


def test_double_rebased_fft_runtime_identity_is_rejected(monkeypatch) -> None:
    original_builder = bridge.build_vituri2024_q003_same_functional_structural_receipt

    def replacement_fft(*_args, **_kwargs):
        raise AssertionError("replacement FFT must never execute")

    monkeypatch.setattr(bridge._response_module, "_FFT2", replacement_fft)
    monkeypatch.setattr(bridge._response_module, "_IMPORT_FFT2", replacement_fft)
    with pytest.raises(RuntimeError, match="imported runtime binding drifted"):
        original_builder(
            context_3307=None,
            independent_source_fock_3307=None,
            context_40fd=None,
            independent_source_fock_40fd=None,
            reciprocity_certificate=None,
            **_lineage_bytes(),
        )


@pytest.mark.parametrize(
    "alias_name",
    (
        "_response_module",
        "_hessian_module",
        "_inventory_module",
        "_core_module",
        "_scalar_module",
        "np",
        "json",
        "inspect",
        "Path",
        "sys",
        "sha256",
    ),
)
def test_bridge_local_proxy_aliases_are_rejected_before_execution(
    monkeypatch,
    alias_name: str,
) -> None:
    original_builder = bridge.build_vituri2024_q003_same_functional_structural_receipt
    original = getattr(bridge, alias_name)

    class ForwardingProxy:
        def __getattr__(self, name: str):
            return getattr(original, name)

        def __call__(self, *args, **kwargs):
            return original(*args, **kwargs)

    monkeypatch.setattr(bridge, alias_name, ForwardingProxy())
    with pytest.raises(RuntimeError, match=f"local alias drifted: {alias_name}"):
        original_builder(
            context_3307=None,
            independent_source_fock_3307=None,
            context_40fd=None,
            independent_source_fock_40fd=None,
            reciprocity_certificate=None,
            **_lineage_bytes(),
        )


def test_sha256_primitive_double_rebinding_is_rejected(monkeypatch) -> None:
    original_builder = bridge.build_vituri2024_q003_same_functional_structural_receipt
    original_sha256 = bridge.sha256

    def replacement_sha256(*args, **kwargs):
        return original_sha256(*args, **kwargs)

    monkeypatch.setattr(bridge, "sha256", replacement_sha256)
    monkeypatch.setattr(bridge, "_SHA256", replacement_sha256)
    with pytest.raises(RuntimeError, match="local alias drifted: sha256"):
        original_builder(
            context_3307=None,
            independent_source_fock_3307=None,
            context_40fd=None,
            independent_source_fock_40fd=None,
            reciprocity_certificate=None,
            **_lineage_bytes(),
        )


def test_mutable_library_primitive_rebinding_is_rejected(monkeypatch) -> None:
    original_builder = bridge.build_vituri2024_q003_same_functional_structural_receipt
    original_dumps = bridge.json.dumps

    def replacement_dumps(*args, **kwargs):
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(bridge.json, "dumps", replacement_dumps)
    with pytest.raises(RuntimeError, match="primitive binding drifted: json.dumps"):
        original_builder(
            context_3307=None,
            independent_source_fock_3307=None,
            context_40fd=None,
            independent_source_fock_40fd=None,
            reciprocity_certificate=None,
            **_lineage_bytes(),
        )


def test_import_formula_fingerprint_is_captured_by_public_closure(monkeypatch) -> None:
    original_builder = bridge.build_vituri2024_q003_same_functional_structural_receipt
    monkeypatch.setattr(
        bridge,
        "_IMPORT_FORMULA_IMPLEMENTATION_FINGERPRINT",
        "0" * 64,
    )
    with pytest.raises(
        RuntimeError, match="import formula implementation fingerprint drifted"
    ):
        original_builder(
            context_3307=None,
            independent_source_fock_3307=None,
            context_40fd=None,
            independent_source_fock_40fd=None,
            reciprocity_certificate=None,
            **_lineage_bytes(),
        )


def test_binding_table_rebase_and_public_builder_rebinding_fail_closed(
    monkeypatch,
) -> None:
    original_builder = bridge.build_vituri2024_q003_same_functional_structural_receipt
    replacement = lambda _context: None
    monkeypatch.setattr(bridge, "_validate_q003_inventory_counts", replacement)
    rebased = tuple(
        (name, replacement if name == "_validate_q003_inventory_counts" else value)
        for name, value in bridge._IMPORT_BRIDGE_BINDINGS
    )
    monkeypatch.setattr(bridge, "_IMPORT_BRIDGE_BINDINGS", rebased)
    with pytest.raises(RuntimeError, match="binding table drifted"):
        original_builder(
            context_3307=None,
            independent_source_fock_3307=None,
            context_40fd=None,
            independent_source_fock_40fd=None,
            reciprocity_certificate=None,
            **_lineage_bytes(),
        )
    monkeypatch.undo()
    monkeypatch.setattr(
        bridge,
        "build_vituri2024_q003_same_functional_structural_receipt",
        lambda **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="public builder binding drifted"):
        original_builder(
            context_3307=None,
            independent_source_fock_3307=None,
            context_40fd=None,
            independent_source_fock_40fd=None,
            reciprocity_certificate=None,
            **_lineage_bytes(),
        )


def test_stale_receipt_validator_binding_is_rejected(monkeypatch) -> None:
    original_builder = bridge.build_vituri2024_q003_same_functional_structural_receipt
    monkeypatch.setattr(
        bridge.Vituri2024Q003SameFunctionalStructuralReceipt,
        "validate_live_state",
        lambda self: None,
    )
    with pytest.raises(RuntimeError, match="receipt method binding drifted"):
        original_builder(
            context_3307=None,
            independent_source_fock_3307=None,
            context_40fd=None,
            independent_source_fock_40fd=None,
            reciprocity_certificate=None,
            **_lineage_bytes(),
        )


def test_wrong_source_fock_is_rejected_before_any_structural_promotion(
    reduced_context,
) -> None:
    wrong = np.zeros((4, 4, reduced_context.nk), dtype=np.complex128)
    wrong.setflags(write=False)
    with pytest.raises(ValueError, match="not the pinned q003 source Fock"):
        bridge.build_vituri2024_q003_same_functional_structural_receipt(
            context_3307=reduced_context,
            independent_source_fock_3307=wrong,
            context_40fd=reduced_context,
            independent_source_fock_40fd=wrong,
            reciprocity_certificate=None,  # reached only after the Fock pin check
            **_lineage_bytes(),
        )


def test_incomplete_inventory_is_rejected_without_postselection(
    reduced_context,
) -> None:
    assert reduced_context.inventory.complex_dimension == 72
    with pytest.raises(ValueError, match="incomplete q003 selected-spin fixed-rank inventory"):
        bridge._validate_q003_inventory_counts(reduced_context)


def test_asymmetric_empty_orbit_partner_is_skipped_before_lane_count(
    reduced_context,
    monkeypatch,
) -> None:
    inventory = reduced_context.inventory
    asymmetric_dimensions = tuple(
        (
            inventory.sector_complex_dimension(orbit.first),
            inventory.sector_complex_dimension(orbit.second),
        )
        for orbit in inventory.iter_conjugate_orbits(include_zero_dimension=False)
        if 0
        in (
            inventory.sector_complex_dimension(orbit.first),
            inventory.sector_complex_dimension(orbit.second),
        )
    )
    assert asymmetric_dimensions
    assert all((first == 0) != (second == 0) for first, second in asymmetric_dimensions)
    monkeypatch.setattr(
        bridge,
        "VITURI2024_Q003_NONEMPTY_SECTOR_COUNT",
        inventory.nonempty_sector_count,
    )
    monkeypatch.setattr(
        bridge,
        "VITURI2024_Q003_COMPLEX_DIMENSION",
        inventory.complex_dimension,
    )
    checked, support, flavor, embedding = bridge._compare_full_transition_geometry(
        reduced_context
    )
    assert checked == inventory.nonempty_sector_count == 49
    assert (support, flavor, embedding) == (0, 0, 0)


def test_extra_and_mutable_inputs_fail_closed(reduced_context) -> None:
    lineage = _lineage_bytes()
    mutable_lineage = dict(lineage)
    mutable_lineage["stationarity_attestation"] = bytearray(
        mutable_lineage["stationarity_attestation"]
    )
    with pytest.raises(TypeError, match="immutable exact bytes"):
        bridge.build_vituri2024_q003_same_functional_structural_receipt(
            context_3307=reduced_context,
            independent_source_fock_3307=np.zeros(
                (4, 4, reduced_context.nk), dtype=np.complex128
            ),
            context_40fd=reduced_context,
            independent_source_fock_40fd=np.zeros(
                (4, 4, reduced_context.nk), dtype=np.complex128
            ),
            reciprocity_certificate=None,
            **mutable_lineage,
        )

    immutable_wrong = np.zeros((4, 4, reduced_context.nk), dtype=np.complex128)
    immutable_wrong.setflags(write=False)
    with pytest.raises(TypeError, match="unexpected keyword argument 'extra_input'"):
        bridge.build_vituri2024_q003_same_functional_structural_receipt(
            context_3307=reduced_context,
            independent_source_fock_3307=immutable_wrong,
            context_40fd=reduced_context,
            independent_source_fock_40fd=immutable_wrong,
            reciprocity_certificate=None,
            **lineage,
            extra_input=b"forbidden",
        )

    mutable_fock = np.zeros((4, 4, reduced_context.nk), dtype=np.complex128)
    with pytest.raises(ValueError, match="finite immutable C-contiguous"):
        bridge.build_vituri2024_q003_same_functional_structural_receipt(
            context_3307=reduced_context,
            independent_source_fock_3307=mutable_fock,
            context_40fd=reduced_context,
            independent_source_fock_40fd=mutable_fock,
            reciprocity_certificate=None,
            **lineage,
        )


def test_readonly_writable_backing_is_snapshotted_then_alias_mutation_rejected(
    reduced_context,
    monkeypatch,
) -> None:
    backing = np.arange(
        16 * reduced_context.nk, dtype=np.float64
    ).astype(np.complex128).reshape((4, 4, reduced_context.nk))
    readonly_alias = backing.view()
    readonly_alias.setflags(write=False)
    expected = np.array(readonly_alias, copy=True)
    expected_hash = bridge._array_sha256(expected)
    monkeypatch.setattr(
        bridge,
        "_PINNED_INDEPENDENT_SOURCE_FOCK_SHA256",
        (("3307", expected_hash), ("40fd", "f" * 64)),
    )

    snapshot = bridge._validate_independent_source_fock(
        readonly_alias,
        nk=reduced_context.nk,
        group_label="3307",
    )
    assert snapshot.flags.c_contiguous
    assert snapshot.flags.writeable is False
    assert not np.shares_memory(snapshot, readonly_alias)
    root = snapshot
    while isinstance(root.base, np.ndarray):
        root = root.base
    assert isinstance(root.base, bytes)

    backing += 1.0
    assert np.array_equal(snapshot, expected)
    assert not np.array_equal(readonly_alias, expected)
    with pytest.raises(ValueError, match="not the pinned q003 source Fock"):
        bridge._validate_independent_source_fock(
            readonly_alias,
            nk=reduced_context.nk,
            group_label="3307",
        )


def test_full_transition_gap_comparison_uses_all_occupied_virtual_pairs() -> None:
    residual = np.asarray([[0.2, -0.1], [0.4, -0.3]], dtype=np.float64)
    occupations = np.asarray([[True, False], [False, True]], dtype=np.bool_)
    # virtual residuals {-0.1, 0.4}; occupied residuals {0.2, -0.3}
    assert bridge._maximum_transition_gap_residual(residual, occupations) == pytest.approx(
        0.7
    )


def test_full_one_body_bound_includes_offdiagonal_occupied_virtual_blocks() -> None:
    selected_exact = np.zeros((2, 2, 2), dtype=np.complex128)
    selected_exact[0, 1] = 0.25
    selected_exact[1, 0] = 0.25
    production_diagonal = np.zeros((2, 2), dtype=np.float64)
    occupations = np.asarray([[True, False], [True, False]], dtype=np.bool_)
    assert bridge._maximum_transition_gap_residual(
        np.zeros((2, 2), dtype=np.float64), occupations
    ) == 0.0
    assert bridge._one_body_superoperator_difference_bound(
        selected_exact, production_diagonal, occupations
    ) == pytest.approx(0.5)


def test_one_body_action_bound_includes_occupied_and_virtual_offdiagonals() -> None:
    selected_exact = np.zeros((2, 2, 2), dtype=np.complex128)
    selected_exact[0, 1, :] = 0.25
    selected_exact[1, 0, :] = 0.25
    production_diagonal = np.zeros((2, 2), dtype=np.float64)
    occupations = np.asarray([[True, False], [True, False]], dtype=np.bool_)
    # One occupied 2x2 block and one virtual 2x2 block each have norm 0.25.
    assert bridge._one_body_superoperator_difference_bound(
        selected_exact, production_diagonal, occupations
    ) == pytest.approx(0.5)


def test_factory_token_is_only_an_honest_caller_construction_boundary() -> None:
    assert not hasattr(bridge, "_TOKEN")
    assert not hasattr(bridge, "_construct_with_factory_token")
    assert "Vituri2024Q003SameFunctionalSourceRecord" not in bridge.__all__
    assert "Vituri2024Q003SameFunctionalStructuralReceipt" not in bridge.__all__
    assert not hasattr(abc_trilayer, "Vituri2024Q003SameFunctionalSourceRecord")
    assert not hasattr(abc_trilayer, "Vituri2024Q003SameFunctionalStructuralReceipt")
    with pytest.raises(TypeError, match="factory-only"):
        bridge.Vituri2024Q003SameFunctionalSourceRecord(
            _factory_token=object(),
            group_label="3307",
            source_group_fingerprint="1" * 64,
            context_fingerprint="2" * 64,
            response_fingerprint="3" * 64,
            inventory_fingerprint="4" * 64,
            independent_source_fock_sha256="5" * 64,
            source_fock_closure_residual_ev=0.0,
            selected_fock_offdiagonal_residual_ev=0.0,
            selected_fock_diagonal_imaginary_residual_ev=0.0,
            maximum_d_prod_d_f_diagonal_residual_ev=0.0,
            maximum_d_prod_d_f_action_bound_ev=0.0,
            nonempty_sector_count=bridge.VITURI2024_Q003_NONEMPTY_SECTOR_COUNT,
            canonical_orbit_count=bridge.VITURI2024_Q003_CANONICAL_ORBIT_COUNT,
            complex_dimension=bridge.VITURI2024_Q003_COMPLEX_DIMENSION,
            real_dimension=bridge.VITURI2024_Q003_REAL_DIMENSION,
            signed_lane_count_checked=bridge.VITURI2024_Q003_NONEMPTY_SECTOR_COUNT,
            support_mismatch_count=0,
            flavor_block_mismatch_count=0,
            transition_embedding_mismatch_count=0,
        )


def test_candidate_receipt_defaults_forbid_every_authority_promotion() -> None:
    fields = bridge.Vituri2024Q003SameFunctionalStructuralReceipt.__dataclass_fields__
    assert fields["candidate_only"].default is True
    assert fields["candidate_structural_receipt_established"].default is True
    assert fields["d_prod_equals_d_f_within_recorded_tolerance"].default is True
    assert fields["candidate_l_prod_sigma_structural_derivation_recorded"].default is True
    assert "candidate_l_prod_equals_sigma_structural_identity" not in fields
    assert fields["l_prod_equals_sigma_structurally_derived"].default is False
    assert fields["l_prod_sigma_numerical_equality_assumed"].default is False
    assert fields["candidate_transition_adjoint_derivation_recorded"].default is True
    assert fields["transition_injection_extraction_adjoint_derived"].default is False
    assert fields["literal_float_full_functional_parity_established"].default is False
    assert fields["no_postselection"].default is True
    assert bridge.VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_IDENTITY.startswith(
        "candidate_derivation_of_L_prod_equals_Sigma"
    )
    assert "pending_detached_review" in (
        bridge.VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_IDENTITY
    )
    for name in (
        "stationarity_established",
        "full_inventory_exact_unitary_scalar_curvature_established",
        "scalar_hessian_authority_established",
        "linear_operator_authorized",
        "hermitian_eigensolver_authorized",
        "full_local_stability_established",
        "scientific_authority_promoted",
        "production_ready",
        "paper_reproduction_verified",
    ):
        assert fields[name].default is False
    assert fields["receipt_fingerprint"].default is MISSING
    assert "_context" not in fields
    assert "both_q003_sources_covered" in fields
    assert "_context" not in fields
    for forbidden in ("matvec", "rmatvec", "linear_operator", "eigenvalues"):
        assert not hasattr(bridge.Vituri2024Q003SameFunctionalStructuralReceipt, forbidden)
        assert forbidden not in bridge.__all__


def test_lineage_hashes_counts_and_public_exports_are_closed() -> None:
    assert bridge._validate_lineage_attestations(**_lineage_bytes()) == (
        (
            "stationarity_477106",
            "b2793903720e9b1550855dcaa1c84d60d08a8cebe84970a058e598c5b8880934",
        ),
        (
            "source_fock_closure_487653",
            "24855c67a6997bad2c10963ef585d5d7199609e1a72b88598f2f285d3aaaa1a5",
        ),
        (
            "exact_unitary_step_ladder_496930",
            "22715b1594c424b9d95f9a3aba2511a65946491ca1a75d8bb6138b5db399eae8",
        ),
    )
    assert bridge.VITURI2024_Q003_SELECTED_OCCUPIED_COUNT == 11_852
    assert bridge.VITURI2024_Q003_SELECTED_VIRTUAL_COUNT == 1_270
    assert bridge.VITURI2024_Q003_NONEMPTY_SECTOR_COUNT == 38_259
    assert bridge.VITURI2024_Q003_CANONICAL_ORBIT_COUNT == 21_170
    assert bridge.VITURI2024_Q003_COMPLEX_DIMENSION == 15_052_040
    assert bridge.VITURI2024_Q003_REAL_DIMENSION == 30_104_080
    for name in bridge.__all__:
        assert getattr(abc_trilayer, name) is getattr(bridge, name)
        assert name in abc_trilayer.__all__
