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
    VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_AUTHORITY,
    VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT_SHA256,
    build_vituri2024_full_functional_replay_bridge,
    build_vituri2024_full_functional_replay_bridge_from_projected_hamiltonian_reference,
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
