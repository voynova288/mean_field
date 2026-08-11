"""Reduced actual-code-path tests for the Vituri full-provider replay bridge."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import inspect
from itertools import product
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mean_field.core.hf import (
    TDHFFullProjectorValidationPlan,
    TDHFFullProjectorValidationTolerances,
    deterministic_complete_hermitian_basis,
    make_tdhf_full_projector_functional_approval,
    make_tdhf_full_projector_unitary_probe,
    validate_tdhf_full_projector_functional,
)
from mean_field.systems.abc_trilayer import (
    REPLAY_PAYLOAD_SCHEMA_FINGERPRINT,
    Vituri2024Flavor,
    Vituri2024FourPointKinematicsReceipt,
    Vituri2024HalfMetalHFReplayPayload,
    Vituri2024InteractionChoiceReceipt,
    Vituri2024Orbital,
    VITURI2024_FULL_PROVIDER_BRIDGE_AUTHORITY,
    VITURI2024_FULL_PROVIDER_INPUT_NAMES,
    VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_AUTHORITY,
    VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_AUTHORITY,
    VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT_SHA256,
    build_vituri2024_full_functional_replay_bridge,
    build_vituri2024_full_functional_replay_bridge_from_projected_hamiltonian_reference,
    build_vituri2024_projected_hamiltonian_identity_gauge_candidate,
    vituri2024_antisymmetrized_projected_vertex,
    vituri2024_full_operator_to_payload_k_diagonal,
    vituri2024_full_projected_interaction_action,
    vituri2024_full_projector_to_payload_density,
    vituri2024_full_provider_energy,
    vituri2024_full_provider_fock,
    vituri2024_full_provider_fock_derivative,
    vituri2024_payload_density_to_full_projector,
    vituri2024_payload_operator_to_full_dense,
    vituri2024_tdhf_full_scalar_source_from_payload,
    make_vituri2024_projected_hamiltonian_zero_reference,
    validate_vituri2024_projected_hamiltonian_identity_gauge_parity,
)
import mean_field.systems.abc_trilayer.vituri2024_tdhf_full_provider_bridge as provider_bridge_module
from mean_field.systems.abc_trilayer.vituri2024 import (
    SM_TEX_SHA256,
    third_lowest_active_band,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_preflight import (
    ACTIVE_BAND_STATES_VALLEY_ORDER,
    INTERNAL_FLAVOR_ORDER,
)


def _digest(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _interaction() -> Vituri2024InteractionChoiceReceipt:
    return Vituri2024InteractionChoiceReceipt(
        gate_distance_angstrom=250.0,
        coulomb_e2_ev_angstrom=14.3996,
        q0_evaluation="analytic_kernel_limit_only",
        provider_sha256=_digest("provider-bridge-interaction"),
        source_sha256=SM_TEX_SHA256,
        authority_kind="reproduction_choice",
        source_text="Reduced actual-vertex full-provider bridge fixture.",
    )


def _states(mesh: np.ndarray, delta1: float) -> np.ndarray:
    result = np.empty((2, 6, mesh.shape[0]), dtype=np.complex128)
    for valley_index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER):
        for k_index, momentum in enumerate(mesh):
            result[valley_index, :, k_index] = third_lowest_active_band(
                momentum, valley, delta1
            ).eigenvector
    return result


def _orbitals(mesh: np.ndarray) -> tuple[Vituri2024Orbital, ...]:
    return tuple(
        Vituri2024Orbital(
            flavor=Vituri2024Flavor(valley=valley, spin=spin),
            momentum_inverse_angstrom=tuple(mesh[k_index]),
        )
        for valley, spin in INTERNAL_FLAVOR_ORDER
        for k_index in range(mesh.shape[0])
    )


def _actual_wbar(
    mesh: np.ndarray,
    delta1: float,
    interaction: Vituri2024InteractionChoiceReceipt,
    area: float,
) -> np.ndarray:
    orbitals = _orbitals(mesh)
    dimension = len(orbitals)
    result = np.zeros((dimension,) * 4, dtype=np.complex128)
    for alpha, beta, gamma, delta in product(range(dimension), repeat=4):
        quartet = tuple(orbitals[index] for index in (alpha, beta, gamma, delta))
        momenta = tuple(item.momentum_inverse_angstrom for item in quartet)
        residual = (
            momenta[0][0] + momenta[1][0] - momenta[2][0] - momenta[3][0],
            momenta[0][1] + momenta[1][1] - momenta[2][1] - momenta[3][1],
        )
        if residual != (0.0, 0.0):
            continue
        kinematics = Vituri2024FourPointKinematicsReceipt(
            alpha=quartet[0],
            beta=quartet[1],
            gamma=quartet[2],
            delta=quartet[3],
            momentum_tolerance_inverse_angstrom=0.0,
            provider_sha256=_digest("provider-bridge-literal-quartet"),
            derivation_source_sm_sha256=SM_TEX_SHA256,
            source_text="Independent actual rank-four replay-bridge oracle.",
        )
        result[alpha, beta, gamma, delta] = (
            vituri2024_antisymmetrized_projected_vertex(
                kinematics, delta1, interaction
            ).value
            / area
        )
    return result


@dataclass(frozen=True)
class _Case:
    payload: Vituri2024HalfMetalHFReplayPayload
    reference: np.ndarray
    interaction: Vituri2024InteractionChoiceReceipt
    area: float
    sigma_actual: np.ndarray
    wbar_actual: np.ndarray
    bridge: object


@pytest.fixture(scope="module")
def actual_case() -> _Case:
    mesh = np.asarray([[0.0, 0.0], [0.013, 0.0]], dtype=np.float64)
    delta1 = 0.028
    states = _states(mesh, delta1)
    dimension = len(INTERNAL_FLAVOR_ORDER) * mesh.shape[0]
    area = 7300.0
    interaction = _interaction()
    p0 = np.zeros((dimension, dimension), dtype=np.complex128)
    p0[: dimension // 2, : dimension // 2] = np.eye(
        dimension // 2, dtype=np.complex128
    )
    reference = 0.125 * np.eye(dimension, dtype=np.complex128)
    h0 = np.diag(np.linspace(-0.06, 0.06, dimension)).astype(np.complex128)
    wbar = _actual_wbar(mesh, delta1, interaction, area)
    sigma = np.einsum("ibgj,gb->ij", wbar, p0 - reference, optimize=True)
    assert np.max(np.abs(sigma)) > 1.0e-8
    assert np.max(np.abs(sigma - sigma.conj().T)) < 2.0e-12
    assert np.max(np.abs(sigma - np.diag(np.diag(sigma)))) < 2.0e-12
    fock = h0 + sigma
    assert np.max(np.abs(fock @ p0 - p0 @ fock)) < 2.0e-12
    stored_projector = vituri2024_full_projector_to_payload_density(p0)
    stored_h0 = vituri2024_full_operator_to_payload_k_diagonal(h0)
    stored_interaction = vituri2024_full_operator_to_payload_k_diagonal(sigma)
    stored_fock = vituri2024_full_operator_to_payload_k_diagonal(fock)
    energies = np.real(np.diag(fock)).reshape(len(INTERNAL_FLAVOR_ORDER), -1)
    occupations = np.real(np.diag(p0)).astype(np.int64).reshape(
        len(INTERNAL_FLAVOR_ORDER), -1
    )
    payload = Vituri2024HalfMetalHFReplayPayload(
        provider_fingerprint=_digest("provider-bridge-synthetic-provider"),
        source_commit="1" * 40,
        source_artifact_sha256=_digest("provider-bridge-synthetic-artifact"),
        spec_fingerprint=_digest("provider-bridge-synthetic-spec"),
        source_state_sha256=_digest("provider-bridge-synthetic-source-state"),
        replay_loader_implementation_fingerprint=_digest(
            "provider-bridge-synthetic-loader"
        ),
        replay_payload_schema_fingerprint=REPLAY_PAYLOAD_SCHEMA_FINGERPRINT,
        mesh=mesh,
        active_band_states=states,
        h0=stored_h0,
        interaction_h=stored_interaction,
        fock=stored_fock,
        projector=stored_projector,
        energies=np.asarray(energies, dtype=np.float64),
        occupations=np.asarray(occupations, dtype=np.int64),
    )
    bridge = build_vituri2024_full_functional_replay_bridge(
        source_payload=payload,
        normal_order_reference_full=reference,
        area_angstrom_squared=area,
        interaction=interaction,
        normal_order_reference_fingerprint=_digest(
            "provider-bridge-reference-policy"
        ),
        reference_policy_evidence_sha256=_digest(
            "provider-bridge-reference-evidence"
        ),
        q0_policy_fingerprint=_digest("provider-bridge-q0-policy"),
        q0_background_evidence_sha256=_digest(
            "provider-bridge-q0-background-absent"
        ),
        provenance=(
            "Reduced synthetic actual-rank-four replay bridge; no immutable "
            "reference, q0-background, production, or paper authority."
        ),
    )
    return _Case(payload, reference, interaction, area, sigma, wbar, bridge)


@pytest.fixture(scope="module")
def small_generic_case() -> _Case:
    mesh = np.asarray([[0.0, 0.0]], dtype=np.float64)
    delta1 = 0.028
    states = _states(mesh, delta1)
    dimension = len(INTERNAL_FLAVOR_ORDER)
    area = 7300.0
    interaction = _interaction()
    p0 = np.diag([1.0, 1.0, 0.0, 0.0]).astype(np.complex128)
    reference = 0.125 * np.eye(dimension, dtype=np.complex128)
    h0 = np.diag(np.linspace(-0.06, 0.06, dimension)).astype(np.complex128)
    wbar = _actual_wbar(mesh, delta1, interaction, area)
    sigma = np.einsum("ibgj,gb->ij", wbar, p0 - reference, optimize=True)
    assert np.max(np.abs(sigma)) > 1.0e-8
    fock = h0 + sigma
    assert np.max(np.abs(fock @ p0 - p0 @ fock)) < 2.0e-12
    payload = Vituri2024HalfMetalHFReplayPayload(
        provider_fingerprint=_digest("provider-bridge-small-provider"),
        source_commit="2" * 40,
        source_artifact_sha256=_digest("provider-bridge-small-artifact"),
        spec_fingerprint=_digest("provider-bridge-small-spec"),
        source_state_sha256=_digest("provider-bridge-small-state"),
        replay_loader_implementation_fingerprint=_digest(
            "provider-bridge-small-loader"
        ),
        replay_payload_schema_fingerprint=REPLAY_PAYLOAD_SCHEMA_FINGERPRINT,
        mesh=mesh,
        active_band_states=states,
        h0=vituri2024_full_operator_to_payload_k_diagonal(h0),
        interaction_h=vituri2024_full_operator_to_payload_k_diagonal(sigma),
        fock=vituri2024_full_operator_to_payload_k_diagonal(fock),
        projector=vituri2024_full_projector_to_payload_density(p0),
        energies=np.real(np.diag(fock)).reshape(len(INTERNAL_FLAVOR_ORDER), 1),
        occupations=np.real(np.diag(p0)).astype(np.int64).reshape(
            len(INTERNAL_FLAVOR_ORDER), 1
        ),
    )
    bridge = build_vituri2024_full_functional_replay_bridge(
        source_payload=payload,
        normal_order_reference_full=reference,
        area_angstrom_squared=area,
        interaction=interaction,
        normal_order_reference_fingerprint=_digest("provider-bridge-small-reference"),
        reference_policy_evidence_sha256=_digest(
            "provider-bridge-small-reference-evidence"
        ),
        q0_policy_fingerprint=_digest("provider-bridge-small-q0-policy"),
        q0_background_evidence_sha256=_digest(
            "provider-bridge-small-q0-background-absent"
        ),
        provenance="Reduced N4 complete-basis bridge fixture; no authority.",
    )
    return _Case(payload, reference, interaction, area, sigma, wbar, bridge)


def test_callbacks_are_plain_distinct_closure_free_and_public() -> None:
    callbacks = (
        vituri2024_full_provider_energy,
        vituri2024_full_provider_fock,
        vituri2024_full_provider_fock_derivative,
    )
    assert all(inspect.isfunction(item) for item in callbacks)
    assert len({id(item.__code__) for item in callbacks}) == 3
    assert all(item.__closure__ is None for item in callbacks)
    assert all(item.__defaults__ is None and item.__kwdefaults__ is None for item in callbacks)
    assert tuple(inspect.signature(callbacks[0]).parameters) == ("inputs", "P")
    assert tuple(inspect.signature(callbacks[1]).parameters) == ("inputs", "P")
    assert tuple(inspect.signature(callbacks[2]).parameters) == ("inputs", "P", "D")


def test_provider_modules_import_in_fresh_process_orders() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    orders = (
        ("vituri2024_tdhf_full_provider_callbacks", "vituri2024_tdhf_full_provider_bridge"),
        ("vituri2024_tdhf_full_provider_bridge", "vituri2024_tdhf_full_provider_callbacks"),
        (
            "vituri2024_projected_hamiltonian_reference",
            "vituri2024_tdhf_full_provider_callbacks",
            "vituri2024_tdhf_full_provider_bridge",
        ),
    )
    for order in orders:
        imports = ";".join(
            "import mean_field.systems.abc_trilayer." + name for name in order
        )
        script = (
            "import sys;"
            f"sys.path.insert(0,{str(source_root)!r});"
            + imports
            + ";import mean_field.systems.abc_trilayer as abc;"
            + "print(abc.VITURI2024_FULL_PROVIDER_BRIDGE_STATUS)"
        )
        output = subprocess.check_output(
            [sys.executable, "-I", "-B", "-c", script],
            env={
                "PATH": "/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            text=True,
        ).strip()
        assert output == (
            "factorized_callbacks_bound_replay_arrays_consistent_not_source_qualified"
        )


def test_payload_density_transpose_and_operator_orientation_are_nonvacuous() -> None:
    stored = np.zeros((4, 4, 2), dtype=np.complex128)
    stored[0, 1, 0] = 0.2 + 0.3j
    stored[1, 0, 0] = 0.2 - 0.3j
    stored[2, 3, 1] = -0.1 + 0.4j
    stored[3, 2, 1] = -0.1 - 0.4j
    full_density = vituri2024_payload_density_to_full_projector(stored)
    full_operator = vituri2024_payload_operator_to_full_dense(stored)
    nk = 2
    first_block = np.asarray([0 * nk + 0, 1 * nk + 0])
    assert np.array_equal(
        full_density[np.ix_(first_block, first_block)], stored[:2, :2, 0].T
    )
    assert np.array_equal(
        full_operator[np.ix_(first_block, first_block)], stored[:2, :2, 0]
    )
    assert not np.array_equal(full_density, full_operator)


def test_paper_projected_hamiltonian_reference_is_exact_R0_and_narrow() -> None:
    receipt = make_vituri2024_projected_hamiltonian_zero_reference(nk=2)
    repeated = make_vituri2024_projected_hamiltonian_zero_reference(nk=2)
    assert repeated.fingerprint == receipt.fingerprint
    assert receipt.dimension == 8
    assert receipt.normal_order_reference_full.dtype == np.dtype(np.complex128)
    assert receipt.normal_order_reference_full.shape == (8, 8)
    assert np.array_equal(receipt.normal_order_reference_full, np.zeros((8, 8)))
    assert receipt.normal_order_reference_full.flags.writeable is False
    assert receipt.normal_order_reference_full.flags.owndata is False
    assert receipt.authority == VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_AUTHORITY
    assert (
        receipt.canonical_equation_text_sha256
        == VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT_SHA256
    )
    assert receipt.paper_active_quartic_R0_semantics_established is True
    assert receipt.canonical_empty_active_electron_reference is True
    assert receipt.r0_is_physical_neutral_reference is False
    assert receipt.physical_neutral_density_identified is False
    assert receipt.normal_order_authority_established is False
    assert receipt.absolute_identity_shift_authority_established is False
    assert receipt.fixed_N_ensemble_authority_established is False
    assert receipt.q0_background_authority_established is False
    assert receipt.replay_source_authority_established is False
    assert receipt.source_closure_established is False
    assert receipt.absolute_fock_zero_authority_established is False
    assert receipt.production_ready is False
    with pytest.raises(TypeError, match="factory-only"):
        type(receipt)(
            _factory_token=object(),
            nk=receipt.nk,
            dimension=receipt.dimension,
            normal_order_reference_full=receipt.normal_order_reference_full,
            normal_order_reference_array_sha256=(
                receipt.normal_order_reference_array_sha256
            ),
            canonical_equation_text_sha256=(
                receipt.canonical_equation_text_sha256
            ),
        )
    object.__setattr__(receipt, "production_ready", True)
    with pytest.raises(ValueError, match="authority was inflated"):
        _ = receipt.fingerprint


def test_paper_R0_opt_in_bridge_matches_R0_and_rejects_nonidentity_reference_arrays(
    small_generic_case: _Case,
) -> None:
    source = vituri2024_tdhf_full_scalar_source_from_payload(
        small_generic_case.payload
    )
    reference = make_vituri2024_projected_hamiltonian_zero_reference(
        nk=source.nk
    )
    sigma0 = small_generic_case.bridge.kernel.interaction_action(
        source.source_projector
    )
    fock0 = source.source_h0 + sigma0
    payload0 = replace(
        small_generic_case.payload,
        interaction_h=vituri2024_full_operator_to_payload_k_diagonal(sigma0),
        fock=vituri2024_full_operator_to_payload_k_diagonal(fock0),
        energies=np.real(np.diag(fock0)).reshape(len(INTERNAL_FLAVOR_ORDER), 1),
        source_state_sha256=_digest("provider-bridge-small-R0-source-state"),
    )
    composite = (
        build_vituri2024_full_functional_replay_bridge_from_projected_hamiltonian_reference(
            source_payload=payload0,
            reference=reference,
            area_angstrom_squared=small_generic_case.area,
            interaction=small_generic_case.interaction,
            q0_policy_fingerprint=_digest("provider-bridge-small-R0-q0-policy"),
            q0_background_evidence_sha256=_digest(
                "provider-bridge-small-R0-q0-background-absent"
            ),
            provenance=(
                "Synthetic R0 saved-array parity for the paper projected-Hamiltonian "
                "reference adapter; no source or q0 authority."
            ),
        )
    )
    assert composite.reference.fingerprint == reference.fingerprint
    assert composite.replay_bridge.array_consistency.passed is True
    assert composite.paper_active_quartic_R0_semantics_established is True
    assert (
        composite.selected_R0_identity_gauge_saved_array_parity_passed is True
    )
    assert composite.absolute_identity_shift_authority_established is False
    assert composite.replay_source_authority_established is False
    assert composite.absolute_fock_zero_authority_established is False
    assert composite.q0_background_authority_established is False
    assert composite.source_closure_established is False
    assert composite.production_ready is False

    nonidentity_reference = np.diag([0.05, 0.15, 0.25, 0.35]).astype(
        np.complex128
    )
    reference_action = small_generic_case.bridge.kernel.interaction_action(
        nonidentity_reference
    )
    identity_part = np.trace(reference_action) / reference_action.shape[0]
    assert np.max(
        np.abs(reference_action - identity_part * np.eye(reference_action.shape[0]))
    ) > 1.0e-8
    sigma_nonidentity = small_generic_case.bridge.kernel.interaction_action(
        source.source_projector - nonidentity_reference
    )
    fock_nonidentity = source.source_h0 + sigma_nonidentity
    payload_nonidentity = replace(
        small_generic_case.payload,
        interaction_h=vituri2024_full_operator_to_payload_k_diagonal(
            sigma_nonidentity
        ),
        fock=vituri2024_full_operator_to_payload_k_diagonal(fock_nonidentity),
        energies=np.real(np.diag(fock_nonidentity)).reshape(
            len(INTERNAL_FLAVOR_ORDER), 1
        ),
        source_state_sha256=_digest(
            "provider-bridge-small-nonidentity-R-source-state"
        ),
    )
    with pytest.raises(ValueError, match="supplied-array consistency failed"):
        build_vituri2024_full_functional_replay_bridge_from_projected_hamiltonian_reference(
            source_payload=payload_nonidentity,
            reference=reference,
            area_angstrom_squared=small_generic_case.area,
            interaction=small_generic_case.interaction,
            q0_policy_fingerprint=_digest("provider-bridge-small-R0-q0-policy"),
            q0_background_evidence_sha256=_digest(
                "provider-bridge-small-R0-q0-background-absent"
            ),
            provenance=(
                "Arrays generated with a nonidentity reference action must fail "
                "selected-R0 representative parity."
            ),
        )
    object.__setattr__(composite, "production_ready", True)
    with pytest.raises(ValueError, match="authority inflated"):
        _ = composite.fingerprint


def _identity_gauge_candidate(case: _Case, payload):
    reference = make_vituri2024_projected_hamiltonian_zero_reference(
        nk=payload.mesh.shape[0]
    )
    return build_vituri2024_projected_hamiltonian_identity_gauge_candidate(
        source_payload=payload,
        reference=reference,
        area_angstrom_squared=case.area,
        interaction=case.interaction,
        q0_policy_fingerprint=_digest("provider-bridge-identity-gauge-q0-policy"),
        q0_background_evidence_sha256=_digest(
            "provider-bridge-identity-gauge-q0-background-absent"
        ),
        provenance=(
            "Synthetic target-free selected-R0 identity-gauge candidate; "
            "no source, q0, TDHF, production, or paper authority."
        ),
    )


def test_identity_gauge_candidate_accepts_one_common_real_global_shift(
    small_generic_case: _Case,
) -> None:
    payload = small_generic_case.payload
    candidate = _identity_gauge_candidate(small_generic_case, payload)
    receipt = validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
        candidate=candidate,
        source_payload=payload,
    )
    assert candidate.authority == VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_AUTHORITY
    assert candidate.identity_gauge_parity_executed is False
    assert candidate.source_closure_established is False
    assert receipt.guarded_fock_execution.invocation_counts == (
        ("energy", 0),
        ("fock", 1),
        ("fock_derivative", 0),
    )
    assert receipt.guarded_fock_execution.callback_trace_verified is True
    assert receipt.guarded_fock_execution.argument_mutation_rejected is True
    assert receipt.guarded_fock_execution.output_alias_rejected is True
    assert receipt.guarded_fock_execution.fock_fingerprint == (
        receipt.computed_fock_full_sha256
    )
    assert receipt.single_real_global_identity_fit_passed is True
    assert (
        receipt.selected_R0_fixed_rank_operator_parity_mod_global_identity_passed
        is True
    )
    reference_action = small_generic_case.bridge.kernel.interaction_action(
        small_generic_case.reference
    )
    expected_alpha = float(np.real(np.trace(reference_action)) / 4.0)
    assert np.max(
        np.abs(reference_action - expected_alpha * np.eye(4, dtype=np.complex128))
    ) < 2.0e-12
    assert receipt.lambda_fit_ev == pytest.approx(-expected_alpha, abs=2.0e-12)
    assert abs(receipt.lambda_fit_ev) > 1.0e-8
    assert receipt.maximum_interaction_identity_quotient_residual_ev < 2.0e-12
    assert receipt.maximum_fock_identity_quotient_residual_ev < 2.0e-12
    assert receipt.maximum_energy_identity_quotient_residual_ev < 2.0e-12
    assert receipt.physical_delta_mu_identified is False
    assert receipt.absolute_fock_parity_established is False
    assert receipt.absolute_energy_or_cross_rank_authority_established is False
    assert receipt.replay_normal_order_source_authority_established is False
    assert receipt.q0_background_authority_established is False
    assert receipt.source_closure_established is False
    assert receipt.generic_functional_qualification_executed is False
    assert receipt.tdhf_hessian_match is False
    assert receipt.production_ready is False

    delta = 0.017
    identity_native = np.eye(4, dtype=np.complex128)[:, :, None]
    shifted = replace(
        payload,
        interaction_h=payload.interaction_h + delta * identity_native,
        fock=payload.fock + delta * identity_native,
        energies=payload.energies + delta,
        source_state_sha256=_digest("identity-gauge-common-shift-target-state"),
    )
    shifted_candidate = _identity_gauge_candidate(small_generic_case, shifted)
    shifted_receipt = (
        validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
            candidate=shifted_candidate,
            source_payload=shifted,
        )
    )
    assert shifted_candidate.fingerprint == candidate.fingerprint
    assert shifted_candidate.inputs.fingerprint == candidate.inputs.fingerprint
    assert shifted_candidate.binding.fingerprint == candidate.binding.fingerprint
    assert shifted_receipt.source_payload_fingerprint != receipt.source_payload_fingerprint
    assert shifted_receipt.lambda_fit_ev == pytest.approx(
        receipt.lambda_fit_ev + delta, abs=2.0e-12
    )
    p0 = candidate.inputs.array("source_projector_full")
    f0 = receipt.computed_fock_full
    identity_full = np.eye(f0.shape[0], dtype=np.complex128)
    assert np.max(
        np.abs((f0 + shifted_receipt.lambda_fit_ev * identity_full) @ p0
               - p0 @ (f0 + shifted_receipt.lambda_fit_ev * identity_full)
               - (f0 @ p0 - p0 @ f0))
    ) < 2.0e-12
    direction = np.asarray(
        [[0.0, 0.2j, 0.0, 0.0], [-0.2j, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.1], [0.0, 0.0, 0.1, 0.0]],
        dtype=np.complex128,
    )
    dF_original = vituri2024_full_provider_fock_derivative(
        candidate.inputs, p0, direction
    )
    dF_shifted = vituri2024_full_provider_fock_derivative(
        shifted_candidate.inputs, p0, direction
    )
    assert np.array_equal(dF_original, dF_shifted)
    angle = 0.21
    unitary = np.eye(4, dtype=np.complex128)
    unitary[np.ix_([0, 2], [0, 2])] = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.complex128,
    )
    rotated = unitary @ p0 @ unitary.conj().T
    assert np.trace(rotated) == pytest.approx(np.trace(p0), abs=2.0e-12)
    assert shifted_receipt.lambda_fit_ev * np.trace(rotated) == pytest.approx(
        shifted_receipt.lambda_fit_ev * np.trace(p0), abs=2.0e-12
    )


def test_identity_gauge_Nk2_uses_full_dimension_and_accepts_signed_common_shifts(
    actual_case: _Case,
) -> None:
    source = vituri2024_tdhf_full_scalar_source_from_payload(actual_case.payload)
    sigma0 = actual_case.bridge.kernel.interaction_action(source.source_projector)
    fock0 = source.source_h0 + sigma0
    payload0 = replace(
        actual_case.payload,
        interaction_h=vituri2024_full_operator_to_payload_k_diagonal(sigma0),
        fock=vituri2024_full_operator_to_payload_k_diagonal(fock0),
        energies=np.real(np.diag(fock0)).reshape(4, 2),
        source_state_sha256=_digest("identity-gauge-Nk2-R0-state"),
    )
    candidate0 = _identity_gauge_candidate(actual_case, payload0)
    receipt0 = validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
        candidate=candidate0, source_payload=payload0
    )
    assert candidate0.space.dimension == 8
    assert receipt0.lambda_fit_ev == pytest.approx(0.0, abs=2.0e-12)
    identity_native = np.repeat(
        np.eye(4, dtype=np.complex128)[:, :, None], 2, axis=2
    )
    for label, delta in (("positive", 0.023), ("negative", -0.019)):
        shifted = replace(
            payload0,
            interaction_h=payload0.interaction_h + delta * identity_native,
            fock=payload0.fock + delta * identity_native,
            energies=payload0.energies + delta,
            source_state_sha256=_digest(f"identity-gauge-Nk2-{label}-state"),
        )
        candidate = _identity_gauge_candidate(actual_case, shifted)
        receipt = validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
            candidate=candidate, source_payload=shifted
        )
        assert candidate.fingerprint == candidate0.fingerprint
        assert receipt.lambda_fit_ev == pytest.approx(delta, abs=2.0e-12)
        difference = receipt.supplied_fock_full - receipt.computed_fock_full
        assert float(np.real(np.trace(difference)) / 8.0) == pytest.approx(
            delta, abs=2.0e-12
        )
        rank = int(round(float(np.real(np.trace(source.source_projector)))))
        assert rank != 8
        assert float(np.real(np.trace(difference)) / rank) != pytest.approx(
            delta, abs=1.0e-6
        )
    per_k = np.zeros_like(payload0.fock)
    per_k[:, :, 0] = 3.0e-6 * np.eye(4, dtype=np.complex128)
    per_k[:, :, 1] = -3.0e-6 * np.eye(4, dtype=np.complex128)
    per_k_payload = replace(
        payload0,
        interaction_h=payload0.interaction_h + per_k,
        fock=payload0.fock + per_k,
        energies=payload0.energies + np.asarray([[3.0e-6, -3.0e-6]] * 4),
        source_state_sha256=_digest("identity-gauge-Nk2-per-k-state"),
    )
    with pytest.raises(ValueError, match="identity-gauge parity failed"):
        validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
            candidate=candidate0, source_payload=per_k_payload
        )


def test_identity_gauge_parity_rejects_nonidentity_and_inconsistent_target_defects(
    small_generic_case: _Case,
) -> None:
    payload = small_generic_case.payload
    candidate = _identity_gauge_candidate(small_generic_case, payload)
    identity_native = np.eye(4, dtype=np.complex128)[:, :, None]
    traceless = np.diag([1.0, -1.0, 0.0, 0.0]).astype(np.complex128)[:, :, None]
    defect = 2.0e-6
    traceless_payload = replace(
        payload,
        interaction_h=payload.interaction_h + defect * traceless,
        fock=payload.fock + defect * traceless,
        energies=payload.energies + defect * np.asarray([[1.0], [-1.0], [0.0], [0.0]]),
        source_state_sha256=_digest("identity-gauge-traceless-target-state"),
    )
    with pytest.raises(ValueError, match="identity-gauge parity failed"):
        validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
            candidate=candidate,
            source_payload=traceless_payload,
        )
    fock_only = replace(
        payload,
        fock=payload.fock + defect * identity_native,
        energies=payload.energies + defect,
        source_state_sha256=_digest("identity-gauge-fock-only-target-state"),
    )
    with pytest.raises(ValueError, match="identity-gauge parity failed"):
        validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
            candidate=candidate,
            source_payload=fock_only,
        )
    interaction_only = replace(
        payload,
        interaction_h=payload.interaction_h + defect * identity_native,
        source_state_sha256=_digest("identity-gauge-interaction-only-target-state"),
    )
    with pytest.raises(ValueError, match="identity-gauge parity failed"):
        validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
            candidate=candidate,
            source_payload=interaction_only,
        )
    energy_only = replace(
        payload,
        energies=payload.energies + defect,
        source_state_sha256=_digest("identity-gauge-energy-only-target-state"),
    )
    with pytest.raises(ValueError, match="identity-gauge parity failed"):
        validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
            candidate=candidate,
            source_payload=energy_only,
        )
    large_shift_with_defect = replace(
        payload,
        interaction_h=(
            payload.interaction_h + 100.0 * identity_native + defect * traceless
        ),
        fock=payload.fock + 100.0 * identity_native + defect * traceless,
        energies=(
            payload.energies
            + 100.0
            + defect * np.asarray([[1.0], [-1.0], [0.0], [0.0]])
        ),
        source_state_sha256=_digest("identity-gauge-large-shift-defect-state"),
    )
    with pytest.raises(ValueError, match="identity-gauge parity failed"):
        validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
            candidate=candidate,
            source_payload=large_shift_with_defect,
        )
    complex_defect = np.zeros_like(payload.fock)
    complex_defect[0, 1, 0] = 1j * defect
    complex_defect[1, 0, 0] = -1j * defect
    offdiagonal = replace(
        payload,
        interaction_h=payload.interaction_h + complex_defect,
        fock=payload.fock + complex_defect,
        source_state_sha256=_digest("identity-gauge-complex-target-state"),
    )
    with pytest.raises(ValueError, match="identity-gauge parity failed"):
        validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
            candidate=candidate,
            source_payload=offdiagonal,
        )
    nonhermitian = payload.fock.copy()
    nonhermitian[0, 1, 0] += 1j * defect
    nonhermitian_payload = replace(
        payload,
        fock=nonhermitian,
        source_state_sha256=_digest("identity-gauge-nonhermitian-target-state"),
    )
    with pytest.raises(ValueError, match="must be Hermitian"):
        validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
            candidate=candidate,
            source_payload=nonhermitian_payload,
        )
    passed = validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
        candidate=candidate,
        source_payload=payload,
    )
    object.__setattr__(passed, "lambda_fit_ev", passed.lambda_fit_ev + 1.0e-4)
    with pytest.raises(ValueError, match="metric lambda_fit_ev drifted"):
        _ = passed.fingerprint
    hash_tampered = validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
        candidate=candidate,
        source_payload=payload,
    )
    changed_energies = hash_tampered.supplied_energies.copy()
    changed_energies[0, 0] += 1.0e-5
    object.__setattr__(hash_tampered, "supplied_energies", changed_energies)
    with pytest.raises(ValueError, match="array supplied_energies drifted"):
        _ = hash_tampered.fingerprint
    authority_tampered = (
        validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
            candidate=candidate,
            source_payload=payload,
        )
    )
    object.__setattr__(authority_tampered, "production_ready", True)
    with pytest.raises(ValueError, match="authority inflated"):
        _ = authority_tampered.fingerprint


def test_existing_caller_supplied_reference_bridge_signature_remains_required() -> None:
    parameters = inspect.signature(
        build_vituri2024_full_functional_replay_bridge
    ).parameters
    for name in (
        "normal_order_reference_full",
        "normal_order_reference_fingerprint",
        "reference_policy_evidence_sha256",
    ):
        assert name in parameters
        assert parameters[name].default is inspect.Parameter.empty


def test_replay_bridge_matches_independent_rank4_targets_and_keeps_authority_false(
    actual_case: _Case,
) -> None:
    bridge = actual_case.bridge
    source = vituri2024_tdhf_full_scalar_source_from_payload(actual_case.payload)
    assert bridge.array_consistency.passed is True
    assert np.max(
        np.abs(
            bridge.kernel.interaction_action(
                source.source_projector - actual_case.reference
            )
            - actual_case.sigma_actual
        )
    ) < 2.0e-12
    assert tuple(item.name for item in bridge.inputs.entries) == (
        VITURI2024_FULL_PROVIDER_INPUT_NAMES
    )
    assert "interaction_h" not in VITURI2024_FULL_PROVIDER_INPUT_NAMES
    assert "source_fock" not in VITURI2024_FULL_PROVIDER_INPUT_NAMES
    assert "source_artifact_sha256" not in VITURI2024_FULL_PROVIDER_INPUT_NAMES
    assert "source_state_sha256" not in VITURI2024_FULL_PROVIDER_INPUT_NAMES
    assert "expected_kernel_fingerprint" not in VITURI2024_FULL_PROVIDER_INPUT_NAMES
    assert bridge.authority == VITURI2024_FULL_PROVIDER_BRIDGE_AUTHORITY
    assert bridge.source_closure_established is False
    assert bridge.normal_order_authority_established is False
    assert bridge.q0_background_authority_established is False
    assert bridge.eligible_for_slurm_qualification is False
    assert bridge.production_ready is False
    assert bridge.paper_reproduction_verified is False
    with pytest.raises(TypeError, match="InitVar '_factory_token'"):
        replace(bridge, provenance="forbidden direct bridge reconstruction")


def test_callbacks_match_kernel_on_full_off_k_complex_inputs(
    actual_case: _Case,
) -> None:
    bridge = actual_case.bridge
    rng = np.random.default_rng(9001)
    raw_p = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    p = np.asarray(0.5 * (raw_p + raw_p.conj().T), dtype=np.complex128)
    raw_d = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    d = np.asarray(0.5 * (raw_d + raw_d.conj().T), dtype=np.complex128)
    expected_d = np.einsum(
        "ibgj,gb->ij", actual_case.wbar_actual, d, optimize=True
    )
    assert np.max(np.abs(bridge.kernel.interaction_action(d) - expected_d)) < 2.0e-12
    exchange_only = np.zeros((8, 8), dtype=np.complex128)
    exchange_only[0, 5] = 0.3 + 0.2j
    exchange_only[5, 0] = 0.3 - 0.2j
    expected_exchange = np.einsum(
        "ibgj,gb->ij", actual_case.wbar_actual, exchange_only, optimize=True
    )
    assert np.max(np.abs(expected_exchange)) > 1.0e-10
    assert np.max(
        np.abs(bridge.kernel.interaction_action(exchange_only) - expected_exchange)
    ) < 2.0e-12
    doubled_area = vituri2024_full_projected_interaction_action(
        d,
        form_factors_by_flavor=bridge.kernel.form_factors_by_flavor,
        interaction_kernel_by_mesh_pair=bridge.kernel.kernel_by_mesh_pair,
        exact_local_mask=bridge.kernel.exact_local_mask,
        area_angstrom_squared=2.0 * bridge.kernel.area_angstrom_squared,
    )
    assert np.max(np.abs(doubled_area - 0.5 * expected_d)) < 2.0e-12
    assert vituri2024_full_provider_energy(bridge.inputs, p) == pytest.approx(
        bridge.kernel.energy(p), abs=2.0e-12
    )
    assert np.max(
        np.abs(
            vituri2024_full_provider_fock(bridge.inputs, p)
            - bridge.kernel.fock(p)
        )
    ) < 2.0e-12
    assert np.max(
        np.abs(
            vituri2024_full_provider_fock_derivative(bridge.inputs, p, d)
            - bridge.kernel.fock_derivative(p, d)
        )
    ) < 2.0e-12


@pytest.mark.slow
def test_reduced_complete_basis_generic_qualification(small_generic_case: _Case) -> None:
    bridge = small_generic_case.bridge
    source = vituri2024_tdhf_full_scalar_source_from_payload(
        small_generic_case.payload
    )
    p0 = source.source_projector
    assert source.space.dimension == 4
    directions = deterministic_complete_hermitian_basis(source.space.dimension)
    assert len(directions) == 16
    angle = 0.17
    unitary = np.eye(source.space.dimension, dtype=np.complex128)
    pair = (0, source.space.dimension // 2)
    unitary[np.ix_(pair, pair)] = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.complex128,
    )
    rotated = unitary @ p0 @ unitary.conj().T
    probe = make_tdhf_full_projector_unitary_probe(
        label="reduced-provider-occupied-unoccupied-rotation",
        source_projector=p0,
        projector=np.asarray(rotated, dtype=np.complex128),
    )
    plan = TDHFFullProjectorValidationPlan(
        space=source.space,
        source_projector=p0,
        directions=directions,
        steps=(2.0e-2, 1.0e-2),
        tolerances=TDHFFullProjectorValidationTolerances(
            gradient_absolute=5.0e-8,
            gradient_relative=5.0e-8,
            derivative_absolute=5.0e-8,
            derivative_relative=5.0e-8,
            exact_absolute=5.0e-8,
            exact_relative=5.0e-8,
            stationarity_absolute=5.0e-8,
            stationarity_relative=5.0e-8,
            self_adjoint_absolute=5.0e-8,
            self_adjoint_relative=5.0e-8,
        ),
        registration_label="Vituri-reduced-N4-factorized-provider-complete-basis",
        probe_scope="complete_small_test_basis",
        require_informative_df=True,
        unitary_projector_probes=(probe,),
    )
    approval = make_tdhf_full_projector_functional_approval(
        space=source.space,
        inputs=bridge.inputs,
        binding=bridge.binding,
        plan=plan,
        provenance="Detached reduced complete-basis factorized-provider approval.",
    )
    receipt = validate_tdhf_full_projector_functional(
        approval=approval,
        space=source.space,
        inputs=bridge.inputs,
        binding=bridge.binding,
        plan=plan,
    )
    assert receipt.registered_probe_functional_consistency is True
    assert receipt.full_projector_functional_consistency is True
    assert receipt.dF_response_informative is True
    assert receipt.exact_unitary_projector_probes_executed is True
    assert receipt.tdhf_hessian_match is False
    assert receipt.static_hessian_authority_promoted is False
    assert receipt.production_ready is False
    assert receipt.paper_reproduction_verified is False


def test_bridge_rejects_reference_area_and_replay_fock_drift(actual_case: _Case) -> None:
    common = dict(
        source_payload=actual_case.payload,
        normal_order_reference_full=actual_case.reference,
        area_angstrom_squared=actual_case.area,
        interaction=actual_case.interaction,
        normal_order_reference_fingerprint=_digest(
            "provider-bridge-reference-policy"
        ),
        reference_policy_evidence_sha256=_digest(
            "provider-bridge-reference-evidence"
        ),
        q0_policy_fingerprint=_digest("provider-bridge-q0-policy"),
        q0_background_evidence_sha256=_digest(
            "provider-bridge-q0-background-absent"
        ),
        provenance="Mutation canary; no authority.",
    )
    wrong_reference = actual_case.reference.copy()
    wrong_reference[0, 0] += 1.0e-3
    with pytest.raises(ValueError, match="supplied-array consistency failed"):
        build_vituri2024_full_functional_replay_bridge(
            **(common | {"normal_order_reference_full": wrong_reference})
        )
    with pytest.raises(ValueError, match="supplied-array consistency failed"):
        build_vituri2024_full_functional_replay_bridge(
            **(common | {"area_angstrom_squared": 1.1 * actual_case.area})
        )
    wrong_fock = actual_case.payload.fock.copy()
    wrong_fock[0, 0, 0] += 1.0e-4
    stale_payload = replace(actual_case.payload, fock=wrong_fock)
    assert provider_bridge_module._payload_fingerprint(stale_payload) != (
        provider_bridge_module._payload_fingerprint(actual_case.payload)
    )
    assert provider_bridge_module._source_input_fingerprint(stale_payload) == (
        provider_bridge_module._source_input_fingerprint(actual_case.payload)
    )
    with pytest.raises(ValueError, match="supplied-array consistency failed"):
        build_vituri2024_full_functional_replay_bridge(
            **(common | {"source_payload": stale_payload})
        )
    fresh = build_vituri2024_full_functional_replay_bridge(**common)
    object.__setattr__(fresh, "production_ready", True)
    with pytest.raises(ValueError, match="construction fingerprint drifted"):
        _ = fresh.fingerprint
