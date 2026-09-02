"""Focused tests for the first restricted Vituri spiral-stability lane."""

from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems import abc_trilayer
from mean_field.systems.abc_trilayer import (
    vituri2024_hf_spiral_stability as stability_module,
)
from mean_field.systems.abc_trilayer.vituri2024_hf import (
    Vituri2024TranslationalHFFunctional,
    vituri2024_conventional_k_diagonal_to_native_density,
    vituri2024_native_density_to_conventional_k_diagonal,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_preflight import (
    INTERNAL_FLAVOR_ORDER,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_scf import (
    Vituri2024CartesianHFSpec,
    prepare_vituri2024_homogeneous_hf,
    prepare_vituri2024_homogeneous_hf_fft,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral import (
    Vituri2024FiniteQSpiralChoice,
    prepare_vituri2024_hf_spiral,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_stability import (
    VITURI2024_HF_SPIRAL_STABILITY_API_VERSION,
    VITURI2024_HF_SPIRAL_STABILITY_AUTHORITY,
    VITURI2024_HF_SPIRAL_STABILITY_CURVATURE_SEED,
    VITURI2024_HF_SPIRAL_STABILITY_CURVATURE_STEPS,
    VITURI2024_HF_SPIRAL_STABILITY_NORMALIZATION,
    VITURI2024_HF_SPIRAL_STABILITY_OFFSET_ROUNDOFF_MULTIPLIER,
    Vituri2024HFSpiralCurvatureDiagnostic,
    Vituri2024HFSpiralCurvatureStep,
    Vituri2024HFSpiralStabilityCurvatureDiagnostic,
    Vituri2024HFSpiralStabilityCurvatureStep,
    Vituri2024HFSpiralStabilityPreparation,
    Vituri2024HFSpiralStabilityReceipt,
    Vituri2024PreparedHFSpiralStability,
    prepare_vituri2024_hf_spiral_restricted_stability,
    prepare_vituri2024_hf_spiral_stability,
)


def _indices(selected_spin: int) -> tuple[tuple[int, int], tuple[int, int]]:
    selected = tuple(
        index
        for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin == selected_spin
    )
    spectator = tuple(
        index
        for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin == -selected_spin
    )
    return selected, spectator  # type: ignore[return-value]


def _normal_endpoint(prepared):
    selected, spectator = _indices(prepared.choice.selected_spin)
    selected_array = np.asarray(selected, dtype=np.int64)
    spectator_array = np.asarray(spectator, dtype=np.int64)
    momenta = np.arange(prepared.nk, dtype=np.int64)
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
    assert int(np.sum(occupations)) == prepared.selected_rank
    conventional = np.zeros((4, 4, prepared.nk), dtype=np.complex128)
    conventional[selected[0], selected[0], :] = occupations[:, 0]
    conventional[selected[1], selected[1], :] = occupations[:, 1]
    conventional[np.ix_(spectator_array, spectator_array, momenta)] = np.repeat(
        np.eye(2, dtype=np.complex128)[:, :, None], prepared.nk, axis=2
    )
    density = vituri2024_conventional_k_diagonal_to_native_density(conventional)
    return density, prepared.functional.fock(density), selected_array, spectator_array


@pytest.fixture(scope="module")
def source():
    base = prepare_vituri2024_homogeneous_hf(
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
    selected, spectator = _indices(prepared.choice.selected_spin)
    selected_array = np.asarray(selected, dtype=np.int64)
    spectator_array = np.asarray(spectator, dtype=np.int64)
    momenta = np.arange(prepared.nk, dtype=np.int64)

    # Exact local ranks 0/1/2, with both coordinate orientations represented
    # among rank-one blocks.  Their sum is prepared.selected_rank == 12.
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
    conventional[selected[0], selected[0], :] = occupations[:, 0]
    conventional[selected[1], selected[1], :] = occupations[:, 1]
    conventional[np.ix_(spectator_array, spectator_array, momenta)] = np.repeat(
        np.eye(2, dtype=np.complex128)[:, :, None], prepared.nk, axis=2
    )
    density = vituri2024_conventional_k_diagonal_to_native_density(conventional)
    fresh = prepared.functional.fock(density)
    candidate = prepare_vituri2024_hf_spiral_stability(prepared, density, fresh)
    return prepared, density, fresh, candidate, selected_array, spectator_array


def test_complex_offdiagonal_orientation_and_exact_anchor_callbacks(source) -> None:
    prepared, density, _fresh, candidate, selected, spectator = source
    nk = prepared.nk
    momenta = np.arange(nk, dtype=np.int64)
    direction = np.zeros((2, 2, nk), dtype=np.complex128)
    values = np.linspace(0.13, 0.29, nk) + 1j * np.linspace(0.41, 0.67, nk)
    direction[0, 1, :] = values
    direction[1, 0, :] = values.conj()

    full = np.zeros((4, 4, nk), dtype=np.complex128)
    full[np.ix_(selected, selected, momenta)] = direction
    native_direction = vituri2024_conventional_k_diagonal_to_native_density(full)
    expected = prepared.functional.fock_derivative(density, native_direction)[
        np.ix_(selected, selected, momenta)
    ]
    actual = candidate.fock_derivative_callback(direction)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(candidate.hessian.fock_derivative(direction), expected)

    # Passing the conventional complex direction as native would conjugate the
    # off-diagonal orientation incorrectly; this oracle must distinguish it.
    wrong_orientation = prepared.functional.fock_derivative(density, full)[
        np.ix_(selected, selected, momenta)
    ]
    assert np.max(np.abs(actual - wrong_orientation)) > 1.0e-8

    selected_projectors = candidate.selected_projectors_conventional
    assert candidate.exact_unitary_energy_callback(selected_projectors) == prepared.functional.energy(
        density
    )
    reconstructed = np.zeros((4, 4, nk), dtype=np.complex128)
    reconstructed[np.ix_(selected, selected, momenta)] = selected_projectors
    reconstructed[np.ix_(spectator, spectator, momenta)] = np.repeat(
        np.eye(2, dtype=np.complex128)[:, :, None], nk, axis=2
    )
    assert candidate.energy_callback(selected_projectors) == prepared.functional.energy(
        vituri2024_conventional_k_diagonal_to_native_density(reconstructed)
    )


def test_rank_inventory_canonical_basis_and_all_one_raw_weights(source) -> None:
    prepared, _density, _fresh, candidate, _selected, _spectator = source
    assert type(candidate) is Vituri2024HFSpiralStabilityPreparation
    assert type(candidate.receipt) is Vituri2024HFSpiralStabilityReceipt
    assert candidate.local_ranks == (0, 1, 2, 2, 2, 2, 1, 1, 1)
    assert candidate.receipt.local_rank_counts_0_1_2 == (1, 4, 4)
    assert sum(candidate.local_ranks) == prepared.selected_rank == 12
    assert len(candidate.receipt.local_rank_inventory_sha256) == 64
    assert candidate.hessian.occupied_counts == candidate.local_ranks
    np.testing.assert_array_equal(
        candidate.hessian.block_weights, np.ones(prepared.nk, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        candidate.selected_orbital_basis_conventional[:, :, 1],
        np.eye(2, dtype=np.complex128),
    )
    np.testing.assert_array_equal(
        candidate.selected_orbital_basis_conventional[:, :, 6],
        np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
    )
    for momentum, rank in enumerate(candidate.local_ranks):
        occupied = candidate.selected_orbital_basis_conventional[:, :rank, momentum]
        np.testing.assert_array_equal(
            occupied @ occupied.conj().T,
            candidate.selected_projectors_conventional[:, :, momentum],
        )
    candidate.validate_live_state()


def test_factory_only_direct_construction_and_callback_state_canaries(source) -> None:
    prepared, density, fresh, candidate, _selected, _spectator = source
    with pytest.raises(TypeError, match="factory-only"):
        Vituri2024HFSpiralStabilityPreparation(
            _factory_token=object(),
            prepared=prepared,
            density_native=density,
            fresh_hamiltonian_conventional=fresh,
            selected_projectors_conventional=(
                candidate.selected_projectors_conventional
            ),
            selected_hamiltonians_conventional=(
                candidate.selected_hamiltonians_conventional
            ),
            selected_orbital_basis_conventional=(
                candidate.selected_orbital_basis_conventional
            ),
            hessian=candidate.hessian,
            fock_derivative_callback=candidate.fock_derivative_callback,
            exact_unitary_energy_callback=candidate.exact_unitary_energy_callback,
            receipt=candidate.receipt,
        )

    # Callable objects replace mutable Python closures.  Their only captured
    # endpoint array is bytes-backed/read-only, and all np.ix arrays are rebuilt.
    assert getattr(candidate.fock_derivative_callback, "__closure__", None) is None
    assert getattr(candidate.exact_unitary_energy_callback, "__closure__", None) is None
    anchor = candidate.fock_derivative_callback.anchor_native
    assert not anchor.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        anchor[0, 0, 0] = 1.0
    assert (
        candidate.fock_derivative_callback.embedding_inventory_fingerprint
        == candidate.receipt.callback_embedding_inventory_fingerprint
        == candidate.exact_unitary_energy_callback.embedding_inventory_fingerprint
    )


def test_validate_live_state_reconstructs_rank_basis_fock_energy_and_hessian(source) -> None:
    _prepared, _density, _fresh, candidate, _selected, _spectator = source
    original_digest = candidate.receipt.local_rank_inventory_sha256
    object.__setattr__(candidate.receipt, "local_rank_inventory_sha256", "0" * 64)
    try:
        with pytest.raises(ValueError, match="rank/basis/Fock slices"):
            candidate.validate_live_state()
    finally:
        object.__setattr__(
            candidate.receipt, "local_rank_inventory_sha256", original_digest
        )

    original_counts = candidate.hessian.occupied_counts
    candidate.hessian.occupied_counts = tuple(reversed(original_counts))
    try:
        with pytest.raises(ValueError, match="generic-Hessian relation"):
            candidate.validate_live_state()
    finally:
        candidate.hessian.occupied_counts = original_counts
    candidate.validate_live_state()


def test_rejects_spectator_cross_block_normal_and_global_rank_violations(source) -> None:
    prepared, density, _fresh, _candidate, selected, spectator = source
    conventional = vituri2024_native_density_to_conventional_k_diagonal(density)

    bad_spectator = np.array(conventional, copy=True)
    bad_spectator[spectator[0], spectator[0], 0] = 0.0
    bad_native = vituri2024_conventional_k_diagonal_to_native_density(bad_spectator)
    with pytest.raises(ValueError, match="spectator density block must be exactly full"):
        prepare_vituri2024_hf_spiral_stability(
            prepared, bad_native, prepared.functional.fock(bad_native)
        )

    bad_cross = np.array(conventional, copy=True)
    bad_cross[selected[0], spectator[0], 0] = 0.2 + 0.3j
    bad_cross[spectator[0], selected[0], 0] = 0.2 - 0.3j
    bad_native = vituri2024_conventional_k_diagonal_to_native_density(bad_cross)
    with pytest.raises(ValueError, match="selected-spectator density blocks"):
        prepare_vituri2024_hf_spiral_stability(
            prepared, bad_native, prepared.functional.fock(bad_native)
        )

    bad_normal = np.array(conventional, copy=True)
    bad_normal[selected[0], selected[1], 0] = 0.1j
    bad_normal[selected[1], selected[0], 0] = -0.1j
    bad_native = vituri2024_conventional_k_diagonal_to_native_density(bad_normal)
    with pytest.raises(ValueError, match="coordinate diagonal"):
        prepare_vituri2024_hf_spiral_stability(
            prepared, bad_native, prepared.functional.fock(bad_native)
        )

    bad_rank = np.array(conventional, copy=True)
    bad_rank[selected[0], selected[0], 1] = 0.0
    bad_native = vituri2024_conventional_k_diagonal_to_native_density(bad_rank)
    with pytest.raises(ValueError, match="selected global rank"):
        prepare_vituri2024_hf_spiral_stability(
            prepared, bad_native, prepared.functional.fock(bad_native)
        )


def test_rejects_stale_fock_bad_hashes_shapes_and_backend_input_type(source) -> None:
    prepared, density, fresh, candidate, _selected, _spectator = source
    stale = np.array(fresh, copy=True)
    stale[0, 0, 0] += 1.0e-12
    with pytest.raises(ValueError, match="is stale"):
        prepare_vituri2024_hf_spiral_stability(prepared, density, stale)
    with pytest.raises(ValueError, match="density_native supplied hash mismatch"):
        prepare_vituri2024_hf_spiral_stability(
            prepared,
            density,
            fresh,
            expected_density_native_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="fresh Hamiltonian supplied hash mismatch"):
        prepare_vituri2024_hf_spiral_stability(
            prepared,
            density,
            fresh,
            expected_fresh_hamiltonian_conventional_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="finite complex128 shape"):
        prepare_vituri2024_hf_spiral_stability(
            prepared, density.astype(np.complex64), fresh
        )
    with pytest.raises(TypeError, match="exact Vituri2024PreparedHFSpiral"):
        prepare_vituri2024_hf_spiral_stability(object(), density, fresh)  # type: ignore[arg-type]

    assert prepared.backend_kind == "dense"
    assert type(prepared.functional) is Vituri2024TranslationalHFFunctional
    assert candidate.receipt.backend_kind == "dense"
    assert candidate.receipt.backend_type == "Vituri2024TranslationalHFFunctional"
    assert len(candidate.receipt.backend_fingerprint) == 64


def test_actual_nonzero_q_displayed_b3_fft_adapter_and_stale_bindings() -> None:
    spec = Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=3)
    prepared = prepare_vituri2024_hf_spiral(
        prepare_vituri2024_homogeneous_hf_fft(spec, fft_workers=1),
        Vituri2024FiniteQSpiralChoice(
            q_inverse_angstrom=(
                np.asarray([0.13, -0.07], dtype=np.float64)
                * spec.delta_k_inverse_angstrom
            ),
            selected_spin=1,
            gauge_mode="displayed_b3",
        ),
    )
    density, fresh, _selected, _spectator = _normal_endpoint(prepared)
    candidate = prepare_vituri2024_hf_spiral_stability(prepared, density, fresh)
    assert prepared.gauge_receipt is not None
    assert np.count_nonzero(prepared.choice.q_inverse_angstrom) == 2
    assert candidate.receipt.backend_kind == "fft"
    assert candidate.receipt.backend_type == "Vituri2024TranslationalHFFFTFunctional"
    assert candidate.receipt.gauge_receipt_fingerprint == prepared.gauge_receipt.fingerprint
    assert candidate.hessian.real_dimension == 8

    direction = np.zeros((2, 2, prepared.nk), dtype=np.complex128)
    direction[0, 1, 1] = 0.2 + 0.3j
    direction[1, 0, 1] = 0.2 - 0.3j
    assert np.max(np.abs(candidate.fock_derivative_callback(direction))) > 0.0

    stale = np.array(fresh, copy=True)
    stale[0, 0, 0] += 1.0e-13
    with pytest.raises(ValueError, match="is stale"):
        prepare_vituri2024_hf_spiral_stability(prepared, density, stale)

    backend_fingerprint = candidate.receipt.backend_fingerprint
    object.__setattr__(candidate.receipt, "backend_fingerprint", "f" * 64)
    try:
        with pytest.raises(ValueError, match="prepared/functional binding drifted"):
            candidate.validate_live_state()
    finally:
        object.__setattr__(
            candidate.receipt, "backend_fingerprint", backend_fingerprint
        )
    candidate.validate_live_state()


def test_exact_unitary_e_f_df_curvature_diagnostic_and_normalization_canaries(source) -> None:
    _prepared, _density, _fresh, candidate, _selected, _spectator = source
    diagnostic = candidate.diagnose_exact_unitary_curvature()
    assert type(diagnostic) is Vituri2024HFSpiralStabilityCurvatureDiagnostic
    assert type(diagnostic) is Vituri2024HFSpiralCurvatureDiagnostic
    assert diagnostic.seed == VITURI2024_HF_SPIRAL_STABILITY_CURVATURE_SEED
    assert diagnostic.steps == VITURI2024_HF_SPIRAL_STABILITY_CURVATURE_STEPS
    assert diagnostic.direction_count == 2
    assert len(diagnostic.evidence) == 4
    assert diagnostic.diagnostic_only
    assert not diagnostic.scalar_hessian_authority_established
    for item in diagnostic.evidence:
        assert type(item) is Vituri2024HFSpiralStabilityCurvatureStep
        assert type(item) is Vituri2024HFSpiralCurvatureStep
        assert item.generic_check.passed
        assert item.generic_check.evaluated_direction_norm == pytest.approx(1.0)
        assert item.clears_offset_roundoff_bound
        assert (
            min(
                abs(item.generic_check.predicted_curvature),
                abs(item.generic_check.finite_difference_curvature),
            )
            > item.offset_roundoff_bound_ev
        )
        assert item.factor_two_canary_rejected
        assert item.nk_normalization_canary_rejected
        assert item.factor_two_wrong_residual_ev > (
            item.generic_check.curvature_tolerance + item.offset_roundoff_bound_ev
        )
        assert item.nk_wrong_normalization_residual_ev > (
            item.generic_check.curvature_tolerance + item.offset_roundoff_bound_ev
        )
        assert item.diagnostic_only
        assert item.exact_unitary_e_f_df_composition


def test_package_export_and_candidate_only_non_authority_flags(source) -> None:
    _prepared, _density, _fresh, candidate, _selected, _spectator = source
    expected_exports = {
        "VITURI2024_HF_SPIRAL_STABILITY_API_VERSION",
        "VITURI2024_HF_SPIRAL_STABILITY_AUTHORITY",
        "VITURI2024_HF_SPIRAL_STABILITY_CURVATURE_SEED",
        "VITURI2024_HF_SPIRAL_STABILITY_CURVATURE_STEPS",
        "VITURI2024_HF_SPIRAL_STABILITY_NORMALIZATION",
        "VITURI2024_HF_SPIRAL_STABILITY_OFFSET_ROUNDOFF_MULTIPLIER",
        "Vituri2024HFSpiralCurvatureDiagnostic",
        "Vituri2024HFSpiralCurvatureStep",
        "Vituri2024HFSpiralStabilityCurvatureDiagnostic",
        "Vituri2024HFSpiralStabilityCurvatureStep",
        "Vituri2024HFSpiralStabilityPreparation",
        "Vituri2024HFSpiralStabilityReceipt",
        "Vituri2024PreparedHFSpiralStability",
        "prepare_vituri2024_hf_spiral_restricted_stability",
        "prepare_vituri2024_hf_spiral_stability",
    }
    assert set(stability_module.__all__) == expected_exports
    for name in expected_exports:
        assert getattr(abc_trilayer, name) is getattr(stability_module, name)
        assert name in abc_trilayer.__all__
    assert VITURI2024_HF_SPIRAL_STABILITY_API_VERSION.endswith(".v2")
    assert VITURI2024_HF_SPIRAL_STABILITY_OFFSET_ROUNDOFF_MULTIPLIER == 64.0
    assert (
        prepare_vituri2024_hf_spiral_restricted_stability
        is prepare_vituri2024_hf_spiral_stability
    )
    assert Vituri2024PreparedHFSpiralStability is Vituri2024HFSpiralStabilityPreparation
    assert (
        abc_trilayer.Vituri2024HFSpiralStabilityReceipt
        is Vituri2024HFSpiralStabilityReceipt
    )
    assert candidate.authority == VITURI2024_HF_SPIRAL_STABILITY_AUTHORITY
    assert candidate.receipt.authority == (
        "local_rank_preserving_k_diagonal_restricted_hessian_candidate_not_full_local_stability"
    )
    assert candidate.receipt.normalization == VITURI2024_HF_SPIRAL_STABILITY_NORMALIZATION
    assert candidate.candidate_only
    assert candidate.receipt.local_rank_preserving_only
    assert candidate.receipt.k_diagonal_only
    assert candidate.receipt.spectator_frozen_to_identity
    assert candidate.receipt.exact_unitary_scalar_diagnostic_only
    assert not candidate.reciprocity_established
    assert not candidate.receipt.reciprocity_established
    assert not candidate.hermitian_eigensolver_authorized
    assert not candidate.receipt.hermitian_eigensolver_authorized
    assert not candidate.full_local_stability_established
    assert not candidate.receipt.full_local_stability_established
    assert not candidate.receipt.occupation_transfer_stability_established
    assert candidate.hessian.requires_separate_occupation_gap_gate
    assert not candidate.hessian.tests_inter_block_occupation_transfers
    assert not candidate.hessian.tests_aufbau_ordering
