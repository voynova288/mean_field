"""Exact-chain tests for the Vituri full-projector candidate adapter."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from mean_field.core.hf import (
    TDHFFullProjectorDirection,
    TDHFFullProjectorFunctionalBinding,
    TDHFFullProjectorSpace,
    TDHFFullProjectorValidationPlan,
    TDHFFullProjectorValidationTolerances,
    bind_tdhf_scalar_kernel,
    make_tdhf_full_projector_unitary_probe,
    make_tdhf_scalar_functional_inputs_manifest,
    validate_tdhf_full_projector_functional,
)
from mean_field.systems.abc_trilayer import (
    VITURI2024_FULL_SCALAR_AFFINE_SUPPORT,
    VITURI2024_FULL_SCALAR_EXACT_UNITARY_SUPPORT,
    VITURI2024_FULL_SCALAR_ONE_BODY_CONSTRUCTION,
    VITURI2024_FULL_SCALAR_RUNTIME_LAYOUT,
    VITURI2024_FULL_SCALAR_STORAGE_DUALITY,
    VITURI2024_FULL_SCALAR_TOTAL_ENERGY_NORMALIZATION,
    Vituri2024FullScalarImmutableVerificationSnapshot,
    Vituri2024TDHFFullScalarCandidate,
    Vituri2024TDHFFullScalarCandidateEvidence,
    Vituri2024TDHFFullScalarSupport,
    certify_vituri2024_full_scalar_orientation_and_normalization,
    consume_vituri2024_tdhf_full_scalar_validation,
    make_vituri2024_tdhf_full_scalar_immutable_evidence,
    make_vituri2024_tdhf_full_scalar_lineage,
    make_vituri2024_tdhf_full_scalar_synthetic_evidence,
    preflight_vituri2024_tdhf_full_scalar_candidate,
    validate_vituri2024_native_to_raw_total_factor,
    vituri2024_full_operator_to_payload_k_diagonal,
    vituri2024_full_projector_to_payload_density,
    vituri2024_payload_density_to_full_projector,
    vituri2024_payload_operator_to_full_dense,
    vituri2024_tdhf_full_scalar_source_from_payload,
)
import mean_field.systems.abc_trilayer as abc
import test_abc_trilayer_vituri2024_tdhf_scalar as scalar_fixtures

_CALLS = {"energy": 0, "fock": 0, "df": 0}
_IMMUTABLE_CANARY_ARTIFACT_BYTES = b"immutable-lineage-migration-canary-artifact"
_IMMUTABLE_CANARY_IMPLEMENTATION_BYTES = b"immutable-lineage-migration-canary-code"


def _immutable_canary_verifier(request):
    return Vituri2024FullScalarImmutableVerificationSnapshot(
        artifact_bytes=_IMMUTABLE_CANARY_ARTIFACT_BYTES,
        implementation_bytes=_IMMUTABLE_CANARY_IMPLEMENTATION_BYTES,
        artifact_manifest_sha256=sha256(
            b"immutable-lineage-migration-canary-artifact-manifest"
        ).hexdigest(),
        implementation_manifest_sha256=sha256(
            b"immutable-lineage-migration-canary-implementation-manifest"
        ).hexdigest(),
        authority_snapshot_fingerprint=sha256(
            b"immutable-lineage-migration-canary-authority"
        ).hexdigest(),
        request_fingerprint=request.fingerprint,
    )


def _fixture_operator(inputs):
    p0 = inputs.array("source_projector_full")
    diagonal = np.real(np.diag(p0))
    unoccupied = np.flatnonzero(np.isclose(diagonal, 0.0, atol=0.0, rtol=0.0))
    operator = np.zeros_like(p0)
    for index, orbital in enumerate(unoccupied[:5]):
        operator[orbital, orbital] = 0.07 + 0.01 * index
    left, right = unoccupied[0], unoccupied[1]
    operator[left, right] = 0.013 + 0.009j
    operator[right, left] = 0.013 - 0.009j
    return operator


def _fixture_action(inputs, P):
    operator = _fixture_operator(inputs)
    return operator @ P @ operator


def _fixture_energy(inputs, P):
    _CALLS["energy"] += 1
    one_body = np.einsum("ij,ji->", inputs.array("h0_full"), P, optimize=False)
    operator = _fixture_operator(inputs)
    interaction = np.trace(P @ operator @ P @ operator)
    return float(np.real(one_body + 0.5 * interaction))


def _fixture_fock(inputs, P):
    _CALLS["fock"] += 1
    return inputs.array("h0_full") + _fixture_action(inputs, P)


def _fixture_df(inputs, P, D):
    del P
    _CALLS["df"] += 1
    return _fixture_action(inputs, D)


def _delegating_energy(inputs, P):
    value = _fixture_fock(inputs, P)
    return float(np.real(np.einsum("ij,ji->", value, P, optimize=False)))


def _finite_q_hessian(inputs, D):
    return _fixture_action(inputs, D)


def _zero_df(inputs, P, D):
    del inputs, P
    return np.zeros_like(D)


def _one_body_energy(inputs, P):
    return float(
        np.real(np.einsum("ij,ji->", inputs.array("h0_full"), P, optimize=False))
    )


def _one_body_fock(inputs, P):
    del P
    return inputs.array("h0_full").copy()


def _delegating_finite_q_df(inputs, P, D):
    del P
    return _finite_q_hessian(inputs, D)


def _readiness(chain):
    return abc.build_vituri2024_tdhf_scalar_readiness(
        **scalar_fixtures._readiness_args(chain)
    )


def _explicit_probes(p0: np.ndarray, nk: int):
    diagonal = np.real(np.diag(p0))
    unoccupied = np.flatnonzero(np.isclose(diagonal, 0.0, atol=0.0, rtol=0.0))
    assert len(unoccupied) >= 2
    cross_pair = next(
        (left, right)
        for left in unoccupied
        for right in unoccupied
        if left < right and left // nk != right // nk and left % nk != right % nk
    )
    same_pair = next(
        (left, right)
        for left in range(p0.shape[0])
        for right in range(p0.shape[0])
        if left < right and left // nk == right // nk and left % nk != right % nk
    )
    imaginary = np.zeros_like(p0)
    imaginary[cross_pair[0], cross_pair[1]] = -1j / np.sqrt(2.0)
    imaginary[cross_pair[1], cross_pair[0]] = 1j / np.sqrt(2.0)
    real = np.zeros_like(p0)
    real[same_pair[0], same_pair[1]] = 1.0 / np.sqrt(2.0)
    real[same_pair[1], same_pair[0]] = 1.0 / np.sqrt(2.0)
    diag = np.zeros_like(p0)
    diag[unoccupied[0], unoccupied[0]] = 1.0
    return (
        TDHFFullProjectorDirection("off_k_cross_flavor_imaginary", imaginary),
        TDHFFullProjectorDirection("off_k_same_flavor_real", real),
        TDHFFullProjectorDirection("unoccupied_diagonal", diag),
    ), unoccupied


def _candidate(chain, *, energy=_fixture_energy, fock=_fixture_fock, df=_fixture_df):
    prerequisites, payload = chain[:2]
    assembly = chain[-1]
    readiness = _readiness(chain)
    source = vituri2024_tdhf_full_scalar_source_from_payload(payload)
    lineage = make_vituri2024_tdhf_full_scalar_lineage(
        readiness=readiness,
        prerequisites=prerequisites,
        source_payload=payload,
    )
    probes, unoccupied = _explicit_probes(source.source_projector, source.nk)
    operator = _fixture_operator(
        make_tdhf_scalar_functional_inputs_manifest(
            {"source_projector_full": source.source_projector},
            source_fingerprint=sha256(b"operator-construction-only").hexdigest(),
            provenance="Deterministic independent synthetic G construction.",
        )
    )
    assert np.array_equal(
        operator @ source.source_projector @ operator,
        np.zeros_like(operator),
    )
    inputs = make_tdhf_scalar_functional_inputs_manifest(
        {
            "area_angstrom_squared": lineage.area_angstrom_squared,
            "form_factor_manifest_sha256": lineage.form_factor_manifest_sha256,
            "h0_full": source.source_h0,
            "interaction_kernel_manifest_sha256": (
                lineage.interaction_kernel_manifest_sha256
            ),
            "normal_order_reference_fingerprint": (
                lineage.normal_order_reference_fingerprint
            ),
            "ordered_mesh": source.ordered_mesh,
            "q0_policy_fingerprint": lineage.q0_policy_fingerprint,
            "source_projector_full": source.source_projector,
        },
        source_fingerprint=sha256(b"vituri-full-n80-independent-quadratic-fixture").hexdigest(),
        provenance="Synthetic independent full 80x80 quadratic provider fixture.",
    )
    binding = TDHFFullProjectorFunctionalBinding(
        energy=bind_tdhf_scalar_kernel(
            role="energy",
            callback=energy,
            dependencies=(_fixture_operator, _fixture_action),
            provenance="Separate synthetic full dense total-energy implementation.",
        ),
        fock=bind_tdhf_scalar_kernel(
            role="fock",
            callback=fock,
            dependencies=(_fixture_operator, _fixture_action),
            provenance="Separate synthetic full dense Fock implementation.",
        ),
        fock_derivative=bind_tdhf_scalar_kernel(
            role="fock_derivative",
            callback=df,
            dependencies=(_fixture_operator, _fixture_action),
            provenance="Separate synthetic full dense dF implementation.",
        ),
        forbidden_entrypoints=(_finite_q_hessian,),
    )
    occupied = np.flatnonzero(
        np.isclose(np.real(np.diag(source.source_projector)), 1.0, atol=0.0, rtol=0.0)
    )
    angle = 0.19
    unitary = np.eye(source.space.dimension, dtype=np.complex128)
    pair = (int(occupied[0]), int(unoccupied[0]))
    unitary[np.ix_(pair, pair)] = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.complex128,
    )
    rotated = unitary @ source.source_projector @ unitary.conj().T
    unitary_probe = make_tdhf_full_projector_unitary_probe(
        label="Vituri-full-exact-occupied-unoccupied-rotation",
        source_projector=source.source_projector,
        projector=np.asarray(rotated, dtype=np.complex128),
    )
    plan = TDHFFullProjectorValidationPlan(
        space=source.space,
        source_projector=source.source_projector,
        directions=probes,
        steps=(2.0e-2, 1.0e-2),
        tolerances=TDHFFullProjectorValidationTolerances(
            gradient_absolute=5.0e-8,
            gradient_relative=5.0e-8,
            derivative_absolute=5.0e-8,
            derivative_relative=5.0e-8,
            exact_absolute=5.0e-9,
            exact_relative=5.0e-9,
            stationarity_absolute=5.0e-9,
            stationarity_relative=5.0e-9,
            self_adjoint_absolute=5.0e-9,
            self_adjoint_relative=5.0e-9,
        ),
        registration_label="Vituri-N80-explicit-off-k-cross-flavor-imaginary-probes",
        probe_scope="explicit_bound_probes",
        require_informative_df=True,
        unitary_projector_probes=(unitary_probe,),
    )
    evidence = make_vituri2024_tdhf_full_scalar_synthetic_evidence(
        lineage=lineage,
        inputs=inputs,
        binding=binding,
        evidence_artifact_bytes=b"synthetic-full-n80-provider-artifact",
        provenance="Exact hash-bound test fixture only; no real-provider authority.",
    )
    support = Vituri2024TDHFFullScalarSupport(
        runtime_layout=VITURI2024_FULL_SCALAR_RUNTIME_LAYOUT,
        runtime_density_shape=(source.space.dimension, source.space.dimension),
        runtime_fock_shape=(source.space.dimension, source.space.dimension),
        affine_support=VITURI2024_FULL_SCALAR_AFFINE_SUPPORT,
        arbitrary_full_dense_affine_hermitian=True,
        arbitrary_off_k=True,
        arbitrary_cross_flavor=True,
        arbitrary_complex_imaginary=True,
    )
    candidate = Vituri2024TDHFFullScalarCandidate(
        evidence=evidence,
        lineage=lineage,
        space=source.space,
        inputs=inputs,
        binding=binding,
        validation_plan=plan,
        support=support,
        orbital_ids=tuple(range(source.space.dimension)),
        one_body_construction=VITURI2024_FULL_SCALAR_ONE_BODY_CONSTRUCTION,
        total_energy_normalization=VITURI2024_FULL_SCALAR_TOTAL_ENERGY_NORMALIZATION,
        provenance="Synthetic full-space candidate for protocol validation only.",
    )
    return readiness, source, candidate


def _preflight(chain, candidate, readiness=None):
    return preflight_vituri2024_tdhf_full_scalar_candidate(
        readiness=_readiness(chain) if readiness is None else readiness,
        prerequisites=chain[0],
        source_payload=chain[1],
        assembly_receipt=chain[-1],
        candidate=candidate,
    )


def test_static_preflight_binds_exact_full_mapping_without_callbacks() -> None:
    chain = scalar_fixtures._chain()
    readiness, source, candidate = _candidate(chain)
    for key in _CALLS:
        _CALLS[key] = 0
    receipt = _preflight(chain, candidate, readiness)
    payload = chain[1]
    nk = source.nk

    assert source.space.dimension == 4 * nk == 80
    assert source.space.axis_sizes == (4, nk)
    assert source.space.axis_order == ("flavor", "k")
    for k_index in range(nk):
        flat = np.arange(4) * nk + k_index
        assert np.array_equal(
            source.source_projector[np.ix_(flat, flat)],
            payload.projector[:, :, k_index].T,
        )
        assert np.array_equal(
            source.source_h0[np.ix_(flat, flat)], payload.h0[:, :, k_index]
        )
        assert np.array_equal(
            source.source_fock[np.ix_(flat, flat)], payload.fock[:, :, k_index]
        )
    assert np.array_equal(
        vituri2024_full_projector_to_payload_density(source.source_projector),
        payload.projector,
    )
    assert np.array_equal(
        vituri2024_full_operator_to_payload_k_diagonal(source.source_h0),
        payload.h0,
    )
    assert np.array_equal(
        vituri2024_payload_density_to_full_projector(payload.projector),
        source.source_projector,
    )
    assert np.array_equal(
        vituri2024_payload_operator_to_full_dense(payload.h0),
        source.source_h0,
    )
    off_k = source.source_projector.copy()
    for k_index in range(nk):
        flat = np.arange(4) * nk + k_index
        off_k[np.ix_(flat, flat)] = 0.0
    assert np.array_equal(off_k, np.zeros_like(off_k))
    assert source.storage_duality == VITURI2024_FULL_SCALAR_STORAGE_DUALITY
    assert receipt.status == "candidate_bound_not_executed"
    assert receipt.native_to_raw_total_factor == nk
    assert receipt.normalization_receipt.raw_total_factor == nk
    assert receipt.normalization_receipt.no_area_division
    assert receipt.candidate_bound
    assert not receipt.generic_validation_executed
    assert receipt.synthetic_fixture
    assert candidate.evidence.lineage_fingerprint == candidate.lineage.fingerprint
    assert len(candidate.evidence.immutable_verification_request_fingerprint) == 64
    assert not candidate.evidence.slurm_evidence_eligible
    assert not receipt.immutable_provider_candidate
    assert not receipt.eligible_for_slurm_qualification
    assert not receipt.tdhf_hessian_match
    assert not receipt.scalar_hessian_match
    assert not receipt.static_hessian_authority_promoted
    assert not receipt.production_ready
    assert not receipt.paper_reproduction_verified
    assert _CALLS == {"energy": 0, "fock": 0, "df": 0}


def test_immutable_evidence_ab_migration_rejects_exact_lineage_drift() -> None:
    chain = scalar_fixtures._chain()
    readiness, _, candidate = _candidate(chain)
    artifact_sha = sha256(_IMMUTABLE_CANARY_ARTIFACT_BYTES).hexdigest()

    with pytest.raises(ValueError, match="requested artifact SHA"):
        make_vituri2024_tdhf_full_scalar_immutable_evidence(
            lineage=candidate.lineage,
            inputs=candidate.inputs,
            binding=candidate.binding,
            verifier=_immutable_canary_verifier,
            verifier_dependencies=(),
            provenance="Source-artifact SHA equality canary.",
        )

    evidence_a = make_vituri2024_tdhf_full_scalar_immutable_evidence(
        lineage=candidate.lineage,
        inputs=candidate.inputs,
        binding=candidate.binding,
        verifier=_immutable_canary_verifier,
        verifier_dependencies=(),
        provenance="Immutable exact-lineage A canary.",
        evidence_artifact_sha256=artifact_sha,
    )
    lineage_b = replace(
        candidate.lineage,
        source_artifact_sha256=sha256(b"migration-source-artifact-b").hexdigest(),
        source_state_sha256=sha256(b"migration-source-state-b").hexdigest(),
        selected_branch_label="migration-branch-b",
    )
    evidence_b = make_vituri2024_tdhf_full_scalar_immutable_evidence(
        lineage=lineage_b,
        inputs=candidate.inputs,
        binding=candidate.binding,
        verifier=_immutable_canary_verifier,
        verifier_dependencies=(),
        provenance="Immutable exact-lineage B migration canary.",
        evidence_artifact_sha256=artifact_sha,
    )

    assert lineage_b.provider_fingerprint == candidate.lineage.provider_fingerprint
    assert lineage_b.source_commit == candidate.lineage.source_commit
    assert lineage_b.source_artifact_sha256 != candidate.lineage.source_artifact_sha256
    assert lineage_b.source_state_sha256 != candidate.lineage.source_state_sha256
    assert lineage_b.selected_branch_label != candidate.lineage.selected_branch_label
    assert evidence_a.binding_fingerprint == evidence_b.binding_fingerprint
    assert (
        evidence_a.generic_input_manifest_fingerprint
        == evidence_b.generic_input_manifest_fingerprint
    )
    assert evidence_a.evidence_artifact_sha256 == evidence_b.evidence_artifact_sha256
    assert evidence_a.lineage_fingerprint != evidence_b.lineage_fingerprint
    assert (
        evidence_a.immutable_verification_request_fingerprint
        != evidence_b.immutable_verification_request_fingerprint
    )

    for key in _CALLS:
        _CALLS[key] = 0
    exact_a = replace(candidate, evidence=evidence_a)
    preflight_a = _preflight(chain, exact_a, readiness)
    assert preflight_a.immutable_provider_candidate
    assert not preflight_a.eligible_for_slurm_qualification
    with pytest.raises(ValueError, match="exact lineage fingerprint"):
        _preflight(chain, replace(candidate, evidence=evidence_b), readiness)
    assert _CALLS == {"energy": 0, "fock": 0, "df": 0}


def test_synthetic_full_provider_passes_generic_protocol_but_never_authority() -> None:
    chain = scalar_fixtures._chain()
    readiness, source, candidate = _candidate(chain)
    preflight = _preflight(chain, candidate, readiness)
    generic = validate_tdhf_full_projector_functional(
        approval=preflight.generic_approval,
        space=candidate.space,
        inputs=candidate.inputs,
        binding=candidate.binding,
        plan=candidate.validation_plan,
    )
    assert generic.registered_probe_functional_consistency
    assert not generic.full_projector_functional_consistency
    assert generic.dF_response_informative
    assert generic.exact_unitary_projector_probes_executed
    assert np.array_equal(generic.source_fock, source.source_h0)
    assert any("off_k" in item.direction_label for item in generic.step_evidence)
    assert any("imaginary" in item.direction_label for item in generic.step_evidence)
    qualification = consume_vituri2024_tdhf_full_scalar_validation(
        preflight=preflight,
        readiness=readiness,
        prerequisites=chain[0],
        source_payload=chain[1],
        assembly_receipt=chain[-1],
        candidate=candidate,
        generic_receipt=generic,
    )
    assert qualification.generic_registered_probe_consistency_consumed
    assert not qualification.generic_full_projector_consistency_consumed
    assert qualification.dF_response_informativeness_consumed
    assert qualification.exact_unitary_execution_consumed
    assert not qualification.eligible_for_slurm_qualification
    assert not qualification.readiness_established
    assert not qualification.tdhf_hessian_match
    assert not qualification.scalar_hessian_match
    assert not qualification.static_hessian_authority_promoted
    assert not qualification.production_ready
    assert not qualification.paper_reproduction_verified


def test_rejects_six_orbital_and_k_diagonal_only_candidates() -> None:
    chain = scalar_fixtures._chain()
    readiness, _, candidate = _candidate(chain)
    space6 = TDHFFullProjectorSpace(
        dimension=6,
        axis_sizes=(6,),
        axis_order=("restricted_transition_union",),
        orbital_order_fingerprint=sha256(b"six-orbital-order").hexdigest(),
        layout_adapter_fingerprint=sha256(b"six-orbital-layout").hexdigest(),
    )
    p6 = np.diag([1, 1, 1, 0, 0, 0]).astype(np.complex128)
    d6 = np.diag([0, 0, 0, 1, 0, 0]).astype(np.complex128)
    plan6 = TDHFFullProjectorValidationPlan(
        space=space6,
        source_projector=p6,
        directions=(TDHFFullProjectorDirection("restricted", d6),),
        steps=(2.0e-2, 1.0e-2),
        tolerances=TDHFFullProjectorValidationTolerances(),
        registration_label="restricted-six-orbital-canary",
        probe_scope="explicit_bound_probes",
    )
    restricted = replace(
        candidate,
        space=space6,
        validation_plan=plan6,
        support=replace(
            candidate.support,
            runtime_density_shape=(6, 6),
            runtime_fock_shape=(6, 6),
        ),
        orbital_ids=tuple(range(6)),
    )
    with pytest.raises(ValueError, match="dimension|layout|transition union"):
        _preflight(chain, restricted, readiness)

    k_only = replace(
        candidate,
        support=replace(
            candidate.support,
            runtime_layout="current_(4,4,Nk)_k_diagonal_only",
        ),
    )
    with pytest.raises(ValueError, match="full dense|support"):
        _preflight(chain, k_only, readiness)


def test_rejects_counterterm_target_inputs_and_restricted_evidence() -> None:
    chain = scalar_fixtures._chain()
    readiness, _, candidate = _candidate(chain)
    with pytest.raises(ValueError, match="counterterm"):
        _preflight(
            chain,
            replace(candidate, one_body_construction="h=F0-dF[P0]"),
            readiness,
        )

    values = {item.name: item.value for item in candidate.inputs.entries}
    values["wbar_target"] = np.zeros((1,), dtype=np.complex128)
    forbidden_inputs = make_tdhf_scalar_functional_inputs_manifest(
        values,
        source_fingerprint=sha256(b"forbidden-wbar-input").hexdigest(),
        provenance="Forbidden target-input canary.",
    )
    with pytest.raises(ValueError, match="A/B/H/wbar|forbidden"):
        _preflight(chain, replace(candidate, inputs=forbidden_inputs), readiness)

    object.__setattr__(candidate.evidence, "evidence_kind", "restricted_oracle")
    try:
        with pytest.raises(ValueError, match="immutable|evidence"):
            _preflight(chain, candidate, readiness)
    finally:
        object.__setattr__(candidate.evidence, "evidence_kind", "synthetic_fixture")


def test_rejects_stale_source_state_branch_and_physical_context_lineage() -> None:
    chain = scalar_fixtures._chain()
    readiness, _, candidate = _candidate(chain)
    mutations = (
        ("source_artifact_sha256", "a" * 64, "source|lineage"),
        ("source_state_sha256", "b" * 64, "state|lineage"),
        ("selected_branch_label", "stale-branch", "branch|lineage"),
        ("selected_spin", -candidate.lineage.selected_spin, "spin|lineage"),
        (
            "area_angstrom_squared",
            candidate.lineage.area_angstrom_squared + 1.0,
            "area|lineage",
        ),
        ("interaction_receipt_fingerprint", "c" * 64, "interaction|lineage"),
        ("q0_policy_fingerprint", "d" * 64, "q0|lineage"),
        ("normal_order_reference_fingerprint", "e" * 64, "normal-order|lineage"),
        ("layout_adapter_fingerprint", "f" * 64, "layout|lineage"),
    )
    for name, value, match in mutations:
        stale = replace(candidate, lineage=replace(candidate.lineage, **{name: value}))
        with pytest.raises(ValueError, match=match):
            _preflight(chain, stale, readiness)


def test_rejects_stale_payload_and_nonpayload_h0() -> None:
    chain = scalar_fixtures._chain()
    readiness, source, candidate = _candidate(chain)
    stale_payload = replace(chain[1], source_state_sha256="a" * 64)
    with pytest.raises(ValueError, match="payload|source state"):
        preflight_vituri2024_tdhf_full_scalar_candidate(
            readiness=readiness,
            prerequisites=chain[0],
            source_payload=stale_payload,
            assembly_receipt=chain[-1],
            candidate=candidate,
        )
    values = {item.name: item.value for item in candidate.inputs.entries}
    values["h0_full"] = source.source_fock
    source_fock_as_h0 = make_tdhf_scalar_functional_inputs_manifest(
        values,
        source_fingerprint=sha256(b"source-fock-as-h0-canary").hexdigest(),
        provenance="Forbidden source_fock substitution for exact payload h0.",
    )
    with pytest.raises(ValueError, match="evidence.*input|h0_full|payload h0"):
        _preflight(chain, replace(candidate, inputs=source_fock_as_h0), readiness)


def test_rejects_delegated_and_finite_q_hessian_callbacks() -> None:
    chain = scalar_fixtures._chain()
    readiness, _, delegated = _candidate(chain, energy=_delegating_energy)
    with pytest.raises((ValueError, RuntimeError), match="delegat|peer"):
        preflight = _preflight(chain, delegated, readiness)
        validate_tdhf_full_projector_functional(
            approval=preflight.generic_approval,
            space=delegated.space,
            inputs=delegated.inputs,
            binding=delegated.binding,
            plan=delegated.validation_plan,
        )

    readiness, _, finite_q = _candidate(chain, df=_delegating_finite_q_df)
    with pytest.raises(ValueError, match="finite-q Hessian"):
        _preflight(chain, finite_q, readiness)


def test_rejects_restricted_module_manifest_and_missing_support() -> None:
    chain = scalar_fixtures._chain()
    readiness, _, candidate = _candidate(chain)
    callback_manifest = candidate.binding.fock_derivative.manifest.callback
    original_module = callback_manifest.module_name
    object.__setattr__(
        callback_manifest,
        "module_name",
        "mean_field.systems.abc_trilayer.vituri2024_tdhf_restricted_scalar",
    )
    try:
        with pytest.raises(ValueError, match="restricted oracle|fingerprint drifted"):
            _preflight(chain, candidate, readiness)
    finally:
        object.__setattr__(callback_manifest, "module_name", original_module)

    support_mutations = (
        {"arbitrary_off_k": False},
        {"arbitrary_cross_flavor": False},
        {"arbitrary_complex_imaginary": False},
    )
    for mutation in support_mutations:
        with pytest.raises(ValueError, match="support"):
            _preflight(
                chain,
                replace(candidate, support=replace(candidate.support, **mutation)),
                readiness,
            )

    no_unitary = replace(candidate.validation_plan, unitary_projector_probes=())
    with pytest.raises(ValueError, match="preregistered exact-unitary"):
        _preflight(
            chain,
            replace(candidate, validation_plan=no_unitary),
            readiness,
        )


def test_rejects_probe_inventory_missing_off_k_cross_flavor_imaginary() -> None:
    chain = scalar_fixtures._chain()
    readiness, source, candidate = _candidate(chain)
    diagonal = np.zeros_like(source.source_projector)
    diagonal[0, 0] = 1.0
    plan = replace(
        candidate.validation_plan,
        directions=(TDHFFullProjectorDirection("k-diagonal-real-only", diagonal),),
    )
    with pytest.raises(ValueError, match="probes.*off-k/cross-flavor/imaginary"):
        _preflight(chain, replace(candidate, validation_plan=plan), readiness)


def test_complex_offdiagonal_orientation_and_native_raw_total_factor_oracle() -> None:
    nk = 2
    operator = np.zeros((4, 4, nk), dtype=np.complex128)
    density = np.zeros_like(operator)
    operator[0, 0, :] = (0.3, 0.5)
    density[0, 0, :] = (0.7, 0.2)
    operator[0, 1, 0] = 0.21 + 0.17j
    operator[1, 0, 0] = 0.21 - 0.17j
    density[0, 1, 0] = -0.13 + 0.09j
    density[1, 0, 0] = -0.13 - 0.09j

    full_operator = vituri2024_payload_operator_to_full_dense(operator)
    full_projector = vituri2024_payload_density_to_full_projector(density)
    flat = np.arange(4) * nk
    assert np.array_equal(full_operator[np.ix_(flat, flat)], operator[:, :, 0])
    assert np.array_equal(full_projector[np.ix_(flat, flat)], density[:, :, 0].T)
    assert not np.array_equal(full_projector[np.ix_(flat, flat)], density[:, :, 0])
    receipt = certify_vituri2024_full_scalar_orientation_and_normalization(
        operator, density
    )
    assert receipt.orientation_residual < 1.0e-14
    assert validate_vituri2024_native_to_raw_total_factor(receipt, nk) == pytest.approx(
        receipt.raw_full_trace_pairing
    )
    with pytest.raises(ValueError, match="exactly Nk"):
        validate_vituri2024_native_to_raw_total_factor(receipt, 1)
    with pytest.raises(ValueError, match="exactly Nk"):
        validate_vituri2024_native_to_raw_total_factor(receipt, 2 * nk)


def test_vituri_mandatory_df_informativeness_rejects_zero_interaction_response() -> None:
    chain = scalar_fixtures._chain()
    readiness, _, candidate = _candidate(
        chain,
        energy=_one_body_energy,
        fock=_one_body_fock,
        df=_zero_df,
    )
    preflight = _preflight(chain, candidate, readiness)
    with pytest.raises(ValueError, match="required dF response.*uninformative"):
        validate_tdhf_full_projector_functional(
            approval=preflight.generic_approval,
            space=candidate.space,
            inputs=candidate.inputs,
            binding=candidate.binding,
            plan=candidate.validation_plan,
        )


def test_missing_physical_input_factory_only_evidence_and_mixed_ab_chain_fail() -> None:
    chain = scalar_fixtures._chain()
    readiness, _, candidate = _candidate(chain)
    values = {
        item.name: item.value
        for item in candidate.inputs.entries
        if item.name != "area_angstrom_squared"
    }
    missing = make_tdhf_scalar_functional_inputs_manifest(
        values,
        source_fingerprint=sha256(b"missing-area-physical-input").hexdigest(),
        provenance="Missing required physical input canary.",
    )
    missing_evidence = make_vituri2024_tdhf_full_scalar_synthetic_evidence(
        lineage=candidate.lineage,
        inputs=missing,
        binding=candidate.binding,
        evidence_artifact_bytes=b"missing-area-canary",
        provenance="Synthetic missing-input canary.",
    )
    with pytest.raises(ValueError, match="required allowlist|missing"):
        _preflight(
            chain,
            replace(candidate, inputs=missing, evidence=missing_evidence),
            readiness,
        )

    with pytest.raises(TypeError, match="factory-only"):
        Vituri2024TDHFFullScalarCandidateEvidence(
            _factory_token=object(),
            evidence_kind="synthetic_fixture",
            evidence_artifact_sha256="a" * 64,
            lineage_fingerprint="9" * 64,
            immutable_verification_request_fingerprint="8" * 64,
            provider_fingerprint="b" * 64,
            source_commit="c" * 40,
            implementation_archive_sha256="d" * 64,
            binding_fingerprint="e" * 64,
            generic_input_manifest_fingerprint="f" * 64,
            physical_input_inventory_fingerprint="1" * 64,
            callback_dependency_inventory_fingerprint="2" * 64,
            artifact_manifest_sha256=None,
            implementation_manifest_sha256=None,
            verifier_callback_fingerprint=None,
            verifier_dependency_inventory_fingerprint=None,
            authority_snapshot_fingerprint=None,
            immutable_artifact_verified=False,
            provenance="Forbidden direct construction.",
        )

    with pytest.raises(ValueError, match="lane|order|mixed|assembly|count"):
        mixed = replace(
            chain[-1],
            B_plus_minus_elements=chain[-1].B_minus_plus_elements,
        )
        preflight_vituri2024_tdhf_full_scalar_candidate(
            readiness=readiness,
            prerequisites=chain[0],
            source_payload=chain[1],
            assembly_receipt=mixed,
            candidate=candidate,
        )
