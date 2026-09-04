"""Focused tests for the full selected-spin no-wrap sector inventory."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from scipy.linalg import expm

from mean_field.core.hf import build_momentum_sector_particle_hole_pairs
from mean_field.systems import abc_trilayer
from mean_field.systems.abc_trilayer import (
    vituri2024_hf_spiral_full_stability as full_stability,
)
from mean_field.systems.abc_trilayer.vituri2024_hf import (
    vituri2024_conventional_k_diagonal_to_native_density,
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
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_full_hessian import (
    VITURI2024_HF_SPIRAL_FULL_HESSIAN_API_VERSION,
    VITURI2024_HF_SPIRAL_FULL_HESSIAN_AUTHORITY,
    Vituri2024HFSpiralFullHessianContext,
    Vituri2024HFSpiralFullHessianOrbit,
    Vituri2024HFSpiralTransitionLaneEmbedding,
    build_vituri2024_hf_spiral_full_hessian_context,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_full_response import (
    VITURI2024_HF_SPIRAL_FULL_RESPONSE_API_VERSION,
    VITURI2024_HF_SPIRAL_FULL_RESPONSE_AUTHORITY,
    VITURI2024_HF_SPIRAL_FULL_RESPONSE_DENSE_MAX_NK,
    VITURI2024_HF_SPIRAL_FULL_RESPONSE_KERNEL_CONTRACT,
    VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_ARITHMETIC,
    VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_MAX_MESH_SIZE,
    VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_SCOPE,
    Vituri2024HFSpiralLiteralMaskComparisonReceipt,
    Vituri2024HFSpiralLiteralMaskEquivalenceReceipt,
    Vituri2024HFSpiralSignedDisplacementResponse,
    Vituri2024HFSpiralValidatedResponseActionFactory,
    Vituri2024HFSpiralValidatedSignedDisplacementFFTAction,
    build_vituri2024_hf_spiral_signed_displacement_response,
    certify_vituri2024_hf_spiral_literal_mask_equivalence,
    compare_vituri2024_hf_spiral_literal_mask_equivalence,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_full_stability import (
    VITURI2024_HF_SPIRAL_FULL_SECTOR_CHARGES,
    VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_API_VERSION,
    VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_AUTHORITY,
    VITURI2024_HF_SPIRAL_FULL_SECTOR_MOMENTUM_CONTRACT,
    Vituri2024HFSpiralFullSectorInventory,
    Vituri2024HFSpiralFullSectorKey,
    Vituri2024HFSpiralFullSectorOrbit,
    build_vituri2024_hf_spiral_full_sector_inventory,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_stability import (
    prepare_vituri2024_hf_spiral_stability,
)
from mean_field.systems.abc_trilayer.vituri2024_tdhf_full_functional import (
    _exact_local_mask,
    vituri2024_full_projected_interaction_action,
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


def _preparation(*, fft: bool, selected_spin: int = 1):
    prepare = (
        prepare_vituri2024_homogeneous_hf_fft
        if fft
        else prepare_vituri2024_homogeneous_hf
    )
    base = prepare(Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=3))
    prepared = prepare_vituri2024_hf_spiral(
        base,
        Vituri2024FiniteQSpiralChoice(
            q_inverse_angstrom=np.zeros(2, dtype=np.float64),
            selected_spin=selected_spin,
            gauge_mode="identity",
        ),
    )
    selected, spectator = _indices(prepared.choice.selected_spin)
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
    conventional = np.zeros((4, 4, prepared.nk), dtype=np.complex128)
    conventional[selected[0], selected[0], :] = occupations[:, 0]
    conventional[selected[1], selected[1], :] = occupations[:, 1]
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
    return restricted


@pytest.fixture(scope="module")
def inventory() -> Vituri2024HFSpiralFullSectorInventory:
    return build_vituri2024_hf_spiral_full_sector_inventory(
        _preparation(fft=True)
    )


def test_complete_global_rank_dimension_and_restricted_subset(inventory) -> None:
    assert inventory.mesh_size == 3
    assert inventory.nk == 9
    assert inventory.selected_occupied_count == 12
    assert inventory.selected_virtual_count == 6
    assert inventory.complex_dimension == 12 * 6 == 72
    assert inventory.real_dimension == 144
    assert inventory.zero_displacement_complex_dimension == 4
    assert inventory.restricted_complex_dimension == 4
    assert inventory.momentum_off_diagonal_complex_dimension == 68
    assert int(np.sum(inventory.ordered_pair_complex_dimensions)) == 72
    assert int(np.sum(inventory.charge_sector_complex_dimensions)) == 72
    assert inventory.total_sector_count == 3 * 5 * 5
    assert inventory.nonempty_sector_count + inventory.zero_dimension_sector_count == (
        inventory.total_sector_count
    )


def test_literal_float_quartet_mask_equivalence_is_exhaustively_certified(
    inventory,
) -> None:
    receipt = certify_vituri2024_hf_spiral_literal_mask_equivalence(
        inventory.integer_mesh_labels,
        inventory.restricted_preparation.prepared.ordered_mesh,
    )
    assert type(receipt) is Vituri2024HFSpiralLiteralMaskEquivalenceReceipt
    assert receipt.mesh_size == 3
    assert receipt.nk == 9
    assert receipt.scalar_quartets_checked_per_axis == 3**4
    assert receipt.false_positive_counts == (0, 0)
    assert receipt.false_negative_counts == (0, 0)
    assert receipt.maximum_conserving_float_residuals == (0.0, 0.0)
    assert all(value > 0.0 for value in receipt.minimum_nonconserving_float_residuals)
    assert receipt.arithmetic_contract == (
        VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_ARITHMETIC
    )
    assert receipt.literal_float_quartet_mask_equivalence_established
    assert receipt.scope == VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_SCOPE
    assert receipt.mesh_size <= VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_MAX_MESH_SIZE
    assert not receipt.flavor_resolved_shifted_momentum_mask_equivalence_established
    assert not receipt.full_functional_action_parity_established
    assert not receipt.scalar_hessian_authority_established
    assert not receipt.production_ready
    receipt.validate_live_state()

    labels = inventory.integer_mesh_labels
    integer_mask = np.all(
        labels[:, None, None, None, :]
        + labels[None, :, None, None, :]
        - labels[None, None, :, None, :]
        - labels[None, None, None, :, :]
        == 0,
        axis=-1,
    )
    assert np.array_equal(
        _exact_local_mask(inventory.restricted_preparation.prepared.ordered_mesh),
        integer_mask,
    )

    with pytest.raises(TypeError, match="factory-only"):
        Vituri2024HFSpiralLiteralMaskEquivalenceReceipt(
            _factory_token=object(),
            mesh_size=receipt.mesh_size,
            nk=receipt.nk,
            integer_mesh_labels_sha256=receipt.integer_mesh_labels_sha256,
            momentum_mesh_inverse_angstrom_sha256=(
                receipt.momentum_mesh_inverse_angstrom_sha256
            ),
            coordinate_table_sha256=receipt.coordinate_table_sha256,
            scalar_quartets_checked_per_axis=receipt.scalar_quartets_checked_per_axis,
            false_positive_counts=receipt.false_positive_counts,
            false_negative_counts=receipt.false_negative_counts,
            maximum_conserving_float_residuals=(
                receipt.maximum_conserving_float_residuals
            ),
            minimum_nonconserving_float_residuals=(
                receipt.minimum_nonconserving_float_residuals
            ),
            exact_local_mask_source_closure_sha256=(
                receipt.exact_local_mask_source_closure_sha256
            ),
            certifier_source_sha256=receipt.certifier_source_sha256,
        )


def test_literal_float_quartet_mask_certifier_rejects_nonaffine_integer_map(
    inventory,
) -> None:
    labels = inventory.integer_mesh_labels
    nonaffine = np.asarray(labels, dtype=np.float64) ** 2
    receipt = compare_vituri2024_hf_spiral_literal_mask_equivalence(
        labels, nonaffine
    )
    assert not receipt.literal_float_quartet_mask_equivalence_established
    assert any(receipt.false_positive_counts)
    assert any(receipt.false_negative_counts)
    receipt.validate_live_state()
    with pytest.raises(ValueError, match="inequivalent"):
        certify_vituri2024_hf_spiral_literal_mask_equivalence(labels, nonaffine)


def test_every_transition_matches_independent_literal_integer_enumeration(inventory) -> None:
    expected: Counter[tuple[int, int, int]] = Counter()
    labels = inventory.integer_mesh_labels
    occupations = inventory.selected_occupations
    valleys = inventory.selected_valleys
    for particle_valley in range(2):
        for particle_k in np.flatnonzero(~occupations[particle_valley]):
            for hole_valley in range(2):
                charge = valleys[particle_valley] - valleys[hole_valley]
                for hole_k in np.flatnonzero(occupations[hole_valley]):
                    displacement = labels[particle_k] - labels[hole_k]
                    expected[(int(displacement[0]), int(displacement[1]), charge)] += 1

    actual = {
        (key.displacement_x, key.displacement_y, key.valley_charge): (
            inventory.sector_complex_dimension(key)
        )
        for key in inventory.iter_sector_keys()
    }
    assert actual == {key: expected.get(key, 0) for key in actual}
    assert sum(expected.values()) == inventory.complex_dimension

    for particle_valley in inventory.selected_valleys:
        for hole_valley in inventory.selected_valleys:
            pair_expected: Counter[tuple[int, int, int]] = Counter()
            particle_index = inventory.selected_valleys.index(particle_valley)
            hole_index = inventory.selected_valleys.index(hole_valley)
            charge = particle_valley - hole_valley
            for particle_k in np.flatnonzero(~occupations[particle_index]):
                for hole_k in np.flatnonzero(occupations[hole_index]):
                    displacement = labels[particle_k] - labels[hole_k]
                    pair_expected[
                        (int(displacement[0]), int(displacement[1]), charge)
                    ] += 1
            for key in inventory.iter_sector_keys():
                assert inventory.ordered_valley_pair_complex_dimension(
                    key,
                    particle_valley=particle_valley,
                    hole_valley=hole_valley,
                ) == pair_expected.get(
                    (key.displacement_x, key.displacement_y, key.valley_charge), 0
                )


def test_charge_aggregation_no_wrap_bounds_and_ordered_valley_pairs(inventory) -> None:
    offset = inventory.displacement_offset
    for key in inventory.iter_sector_keys():
        dimension = sum(
            inventory.ordered_valley_pair_complex_dimension(
                key,
                particle_valley=particle_valley,
                hole_valley=hole_valley,
            )
            for particle_valley in inventory.selected_valleys
            for hole_valley in inventory.selected_valleys
        )
        assert dimension == inventory.sector_complex_dimension(key)
        support = (
            inventory.mesh_size - abs(key.displacement_x)
        ) * (inventory.mesh_size - abs(key.displacement_y))
        number_of_ordered_flavor_pairs = 2 if key.valley_charge == 0 else 1
        assert dimension <= number_of_ordered_flavor_pairs * support
    with pytest.raises(KeyError, match="outside"):
        inventory.sector_complex_dimension(
            Vituri2024HFSpiralFullSectorKey(offset + 1, 0, 0)
        )


def test_conjugate_orbits_partition_without_dimension_equality_assumption(inventory) -> None:
    orbits = tuple(inventory.iter_conjugate_orbits())
    assert len(orbits) == inventory.nonempty_conjugate_orbit_count
    assert all(type(orbit) is Vituri2024HFSpiralFullSectorOrbit for orbit in orbits)
    assert sum(orbit.complex_dimension for orbit in orbits) == inventory.complex_dimension
    assert sum(orbit.real_dimension for orbit in orbits) == inventory.real_dimension
    assert all(orbit.first <= orbit.second for orbit in orbits)
    assert all(orbit.first.conjugate == orbit.second for orbit in orbits)
    keys_from_orbits = [
        key
        for orbit in orbits
        for key in ((orbit.first,) if orbit.self_conjugate else (orbit.first, orbit.second))
    ]
    expected_keys = {
        key
        for key in inventory.iter_sector_keys()
        if (
            inventory.sector_complex_dimension(key) > 0
            or inventory.sector_complex_dimension(key.conjugate) > 0
        )
    }
    assert len(keys_from_orbits) == len(set(keys_from_orbits))
    assert set(keys_from_orbits) == expected_keys
    # A finite, asymmetric occupation pattern need not make conjugate lane
    # dimensions equal; no implementation may silently pad, average, or drop it.
    assert any(
        not orbit.self_conjugate
        and orbit.first_complex_dimension != orbit.second_complex_dimension
        for orbit in orbits
    )
    self_orbits = [orbit for orbit in orbits if orbit.self_conjugate]
    assert len(self_orbits) <= 1
    if self_orbits:
        assert self_orbits[0].first == Vituri2024HFSpiralFullSectorKey(0, 0, 0)


def test_source_binding_authority_flags_and_live_canary(inventory) -> None:
    assert inventory.api_version == VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_API_VERSION
    assert inventory.authority == VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_AUTHORITY
    assert inventory.momentum_contract == VITURI2024_HF_SPIRAL_FULL_SECTOR_MOMENTUM_CONTRACT
    assert inventory.valley_charges == VITURI2024_HF_SPIRAL_FULL_SECTOR_CHARGES
    assert inventory.candidate_only
    assert inventory.spectator_frozen_to_identity
    assert inventory.global_selected_rank_complete
    assert inventory.no_wrap
    assert inventory.integer_label_partitioning_convention
    assert not inventory.integer_label_interaction_conservation_established
    assert not inventory.conjugate_sector_dimensions_assumed_equal
    assert not inventory.off_k_response_implemented
    assert not inventory.literal_float_full_functional_parity_established
    assert not inventory.reciprocity_established
    assert not inventory.hermitian_eigensolver_authorized
    assert not inventory.full_local_stability_established
    assert not inventory.production_ready
    assert not inventory.paper_reproduction_verified
    assert len(inventory.fingerprint) == 64
    inventory.validate_live_state()

    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        inventory.charge_sector_complex_dimensions.setflags(write=True)
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        inventory.ordered_pair_complex_dimensions.setflags(write=True)
    inventory.validate_live_state()


def test_signed_displacement_matches_core_particle_minus_hole_q_convention() -> None:
    transfer = (1, -1)
    pairs = build_momentum_sector_particle_hole_pairs(
        {(0, 0): (0,)},
        {transfer: (1,)},
        transfer,
        lambda momentum, q: (momentum[0] + q[0], momentum[1] + q[1]),
    )
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.particle_momentum == transfer
    assert pair.hole_momentum == (0, 0)
    displacement = (
        pair.particle_momentum[0] - pair.hole_momentum[0],
        pair.particle_momentum[1] - pair.hole_momentum[1],
    )
    key = Vituri2024HFSpiralFullSectorKey(*displacement, 0)
    assert (key.displacement_x, key.displacement_y) == transfer


def test_rejects_dense_source_and_invalid_sector_keys(inventory) -> None:
    dense = _preparation(fft=False)
    with pytest.raises(TypeError, match="exact FFT preparation"):
        build_vituri2024_hf_spiral_full_sector_inventory(dense)
    with pytest.raises(ValueError, match="valley charge"):
        Vituri2024HFSpiralFullSectorKey(0, 0, 1)
    with pytest.raises(TypeError, match="exact int"):
        Vituri2024HFSpiralFullSectorKey(True, 0, 0)
    key = Vituri2024HFSpiralFullSectorKey(0, 0, 0)
    with pytest.raises(TypeError, match="exact ints"):
        inventory.ordered_valley_pair_complex_dimension(
            key, particle_valley=True, hole_valley=-1
        )


@pytest.fixture(scope="module")
def response(inventory) -> Vituri2024HFSpiralSignedDisplacementResponse:
    return build_vituri2024_hf_spiral_signed_displacement_response(inventory)


def _signed_block(response, key, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros((2, 2, response.nk), dtype=np.complex128)
    bases, _targets = response.support_indices(key)
    if key.valley_charge == 0:
        result[0, 0, bases] = rng.normal(size=len(bases)) + 1.0j * rng.normal(
            size=len(bases)
        )
        result[1, 1, bases] = rng.normal(size=len(bases)) + 1.0j * rng.normal(
            size=len(bases)
        )
    elif key.valley_charge == 2:
        result[1, 0, bases] = rng.normal(size=len(bases)) + 1.0j * rng.normal(
            size=len(bases)
        )
    else:
        result[0, 1, bases] = rng.normal(size=len(bases)) + 1.0j * rng.normal(
            size=len(bases)
        )
    return result


@pytest.mark.parametrize(
    ("key", "seed"),
    (
        (Vituri2024HFSpiralFullSectorKey(0, 0, 0), 11),
        (Vituri2024HFSpiralFullSectorKey(0, 0, 2), 12),
        (Vituri2024HFSpiralFullSectorKey(1, 0, 0), 13),
        (Vituri2024HFSpiralFullSectorKey(1, -1, 2), 14),
        (Vituri2024HFSpiralFullSectorKey(-2, 2, -2), 15),
    ),
)
def test_signed_response_dense_fft_and_conjugate_covariance(response, key, seed) -> None:
    block = _signed_block(response, key, seed=seed)
    dense = response.apply_dense(key, block)
    fft = response.apply_fft(key, block)
    assert np.max(np.abs(dense - fft), initial=0.0) < 2.0e-13

    partner_key = key.conjugate
    partner_block = response.conjugate_block(key, block)
    partner_response = response.apply_fft(partner_key, partner_block)
    expected_partner = response.conjugate_block(key, fft)
    assert np.max(
        np.abs(partner_response - expected_partner), initial=0.0
    ) < 2.0e-13


def _integer_mask_full_projected_h_oracle(response, key, block):
    inventory = response.inventory
    preparation = inventory.restricted_preparation
    prepared = preparation.prepared
    nk = response.nk
    labels = inventory.integer_mesh_labels
    mask = np.all(
        labels[:, None, None, None, :]
        + labels[None, :, None, None, :]
        - labels[None, None, :, None, :]
        - labels[None, None, None, :, :]
        == 0,
        axis=-1,
    )
    form_factors = np.empty((4, nk, nk), dtype=np.complex128)
    valley_to_index = {-1: 0, 1: 1}
    for flavor, (valley, _spin) in enumerate(INTERNAL_FLAVOR_ORDER):
        states = prepared.active_band_states[valley_to_index[valley]]
        form_factors[flavor] = np.einsum(
            "cm,cn->mn", states.conj(), states, optimize=True
        )
    displacement = labels[:, None, :] - labels[None, :, :]
    offset = inventory.mesh_size - 1
    kernel_pair = np.asarray(
        response.fft_plan.kernel_by_signed_displacement[
            displacement[..., 1] + offset,
            displacement[..., 0] + offset,
        ].real,
        dtype=np.float64,
    )
    full_density = np.zeros((4 * nk, 4 * nk), dtype=np.complex128)
    selected = preparation.receipt.selected_flavor_indices
    bases, targets = response.support_indices(key)
    for left in range(2):
        for right in range(2):
            full_density[
                selected[left] * nk + targets,
                selected[right] * nk + bases,
            ] = block[left, right, bases]
    full_density += full_density.conj().T
    full_response = vituri2024_full_projected_interaction_action(
        full_density,
        form_factors_by_flavor=form_factors,
        interaction_kernel_by_mesh_pair=kernel_pair,
        exact_local_mask=mask,
        area_angstrom_squared=response.area_angstrom_squared,
    )
    extracted = np.zeros((2, 2, nk), dtype=np.complex128)
    for left in range(2):
        for right in range(2):
            extracted[left, right, bases] = full_response[
                selected[left] * nk + targets,
                selected[right] * nk + bases,
            ]
    return extracted, full_response, selected, bases, targets


@pytest.mark.parametrize(
    ("key", "seed"),
    (
        (Vituri2024HFSpiralFullSectorKey(1, 0, 0), 61),
        (Vituri2024HFSpiralFullSectorKey(1, -1, 2), 62),
        (Vituri2024HFSpiralFullSectorKey(-1, 1, -2), 63),
    ),
)
def test_signed_response_matches_integer_mask_full_projected_h_oracle(
    response, key, seed
) -> None:
    block = _signed_block(response, key, seed=seed)
    expected, full_response, selected, bases, targets = (
        _integer_mask_full_projected_h_oracle(response, key, block)
    )
    actual = response.apply_fft(key, block)
    assert np.linalg.norm(expected) > 1.0e-8
    assert np.max(np.abs(actual - expected), initial=0.0) < 2.0e-13
    if key.valley_charge == 0:
        spectators = sorted(set(range(4)) - set(selected))
        spectator_norm = 0.0
        for spectator in spectators:
            spectator_norm += np.linalg.norm(
                full_response[
                    spectator * response.nk + targets,
                    spectator * response.nk + bases,
                ]
            )
        assert spectator_norm > 1.0e-8
    else:
        spectators = sorted(set(range(4)) - set(selected))
        for spectator in spectators:
            assert np.count_nonzero(
                full_response[
                    spectator * response.nk + targets,
                    spectator * response.nk + bases,
                ]
            ) == 0


def test_zero_displacement_reduces_to_existing_k_diagonal_fft_action(response) -> None:
    rng = np.random.default_rng(29)
    selected = np.zeros((2, 2, response.nk), dtype=np.complex128)
    selected[0, 0] = rng.normal(size=response.nk)
    selected[1, 1] = rng.normal(size=response.nk)
    selected[1, 0] = rng.normal(size=response.nk) + 1.0j * rng.normal(
        size=response.nk
    )
    selected[0, 1] = selected[1, 0].conj()

    response_sum = np.zeros_like(selected)
    for charge in VITURI2024_HF_SPIRAL_FULL_SECTOR_CHARGES:
        key = Vituri2024HFSpiralFullSectorKey(0, 0, charge)
        block = np.zeros_like(selected)
        if charge == 0:
            block[0, 0] = selected[0, 0]
            block[1, 1] = selected[1, 1]
        elif charge == 2:
            block[1, 0] = selected[1, 0]
        else:
            block[0, 1] = selected[0, 1]
        response_sum += response.apply_fft(key, block)

    preparation = response.inventory.restricted_preparation
    selected_indices = np.asarray(
        preparation.receipt.selected_flavor_indices, dtype=np.int64
    )
    momenta = np.arange(response.nk, dtype=np.int64)
    full = np.zeros((4, 4, response.nk), dtype=np.complex128)
    full[np.ix_(selected_indices, selected_indices, momenta)] = selected
    existing = preparation.prepared.functional.interaction_action_conventional(full)
    existing_selected = existing[np.ix_(selected_indices, selected_indices, momenta)]
    assert np.max(
        np.abs(response_sum - existing_selected), initial=0.0
    ) < 2.0e-13


def test_signed_response_rejects_support_and_charge_leakage(response) -> None:
    key = Vituri2024HFSpiralFullSectorKey(1, 0, 2)
    block = _signed_block(response, key, seed=41)
    bases, _targets = response.support_indices(key)
    outside = sorted(set(range(response.nk)) - set(int(value) for value in bases))
    assert outside
    block[1, 0, outside[0]] = 1.0
    with pytest.raises(ValueError, match="outside no-wrap support"):
        response.apply_fft(key, block)

    block = _signed_block(response, key, seed=42)
    block[0, 0, bases[0]] = 1.0
    with pytest.raises(ValueError, match="conserved valley charge"):
        response.apply_fft(key, block)


def test_validate_once_fft_callback_avoids_recursive_source_validation(
    response, monkeypatch
) -> None:
    key = Vituri2024HFSpiralFullSectorKey(1, -1, 2)
    block = _signed_block(response, key, seed=73)
    expected = response.apply_fft(key, block)
    action = response.make_validated_fft_action(key)
    assert type(action) is Vituri2024HFSpiralValidatedSignedDisplacementFFTAction

    def forbidden_recursive_validation(_self):
        raise AssertionError("Krylov hot path recursively validated the full source")

    monkeypatch.setattr(
        Vituri2024HFSpiralFullSectorInventory,
        "validate_live_state",
        forbidden_recursive_validation,
    )
    actual = action(block)
    assert np.array_equal(actual, expected)
    with pytest.raises(AssertionError, match="recursively validated"):
        response.apply_fft(key, block)


def test_response_supports_the_other_selected_spin_without_relabeling_valleys() -> None:
    inventory = build_vituri2024_hf_spiral_full_sector_inventory(
        _preparation(fft=True, selected_spin=-1)
    )
    response = build_vituri2024_hf_spiral_signed_displacement_response(inventory)
    assert inventory.restricted_preparation.receipt.selected_flavor_indices == (0, 2)
    assert inventory.selected_valleys == (-1, 1)
    key = Vituri2024HFSpiralFullSectorKey(1, 0, -2)
    block = _signed_block(response, key, seed=79)
    expected, _full, _selected, _bases, _targets = (
        _integer_mask_full_projected_h_oracle(response, key, block)
    )
    assert np.max(
        np.abs(response.apply_fft(key, block) - expected), initial=0.0
    ) < 2.0e-13


@pytest.fixture(scope="module")
def hessian_context(response) -> Vituri2024HFSpiralFullHessianContext:
    return build_vituri2024_hf_spiral_full_hessian_context(response)


def _apply_full_integer_oracle(response, density):
    inventory = response.inventory
    prepared = inventory.restricted_preparation.prepared
    labels = inventory.integer_mesh_labels
    nk = response.nk
    mask = np.all(
        labels[:, None, None, None, :]
        + labels[None, :, None, None, :]
        - labels[None, None, :, None, :]
        - labels[None, None, None, :, :]
        == 0,
        axis=-1,
    )
    form_factors = np.empty((4, nk, nk), dtype=np.complex128)
    valley_to_index = {-1: 0, 1: 1}
    for flavor, (valley, _spin) in enumerate(INTERNAL_FLAVOR_ORDER):
        states = prepared.active_band_states[valley_to_index[valley]]
        form_factors[flavor] = np.einsum(
            "cm,cn->mn", states.conj(), states, optimize=True
        )
    displacement = labels[:, None, :] - labels[None, :, :]
    offset = inventory.mesh_size - 1
    kernel_pair = np.asarray(
        response.fft_plan.kernel_by_signed_displacement[
            displacement[..., 1] + offset,
            displacement[..., 0] + offset,
        ].real,
        dtype=np.float64,
    )
    return vituri2024_full_projected_interaction_action(
        density,
        form_factors_by_flavor=form_factors,
        interaction_kernel_by_mesh_pair=kernel_pair,
        exact_local_mask=mask,
        area_angstrom_squared=response.area_angstrom_squared,
    )


def _literal_transition_tuples(inventory, key):
    label_to_index = {
        tuple(int(component) for component in label): index
        for index, label in enumerate(inventory.integer_mesh_labels)
    }
    if key.valley_charge == 0:
        flavor_pairs = ((0, 0), (1, 1))
    elif key.valley_charge == 2:
        flavor_pairs = ((1, 0),)
    else:
        flavor_pairs = ((0, 1),)
    result = []
    occupations = inventory.selected_occupations
    for particle_flavor, hole_flavor in flavor_pairs:
        for hole_k, label in enumerate(inventory.integer_mesh_labels):
            particle_label = (
                int(label[0]) + key.displacement_x,
                int(label[1]) + key.displacement_y,
            )
            particle_k = label_to_index.get(particle_label)
            if (
                particle_k is not None
                and occupations[hole_flavor, hole_k]
                and not occupations[particle_flavor, particle_k]
            ):
                result.append(
                    (particle_flavor, particle_k, hole_flavor, hole_k)
                )
    return tuple(result)


def _global_selected_tangent(prepared_orbit, first, second):
    response = prepared_orbit.context.response
    nk = response.nk
    selected = response.inventory.restricted_preparation.receipt.selected_flavor_indices
    density = np.zeros((4 * nk, 4 * nk), dtype=np.complex128)
    for embedding, values in (
        (prepared_orbit.first_embedding, first),
        (prepared_orbit.second_embedding, second),
    ):
        rows = (
            np.asarray(selected)[embedding.particle_valley_slots] * nk
            + embedding.particle_k_indices
        )
        columns = (
            np.asarray(selected)[embedding.hole_valley_slots] * nk
            + embedding.hole_k_indices
        )
        density[rows, columns] = values
    density += density.conj().T
    return density


def test_paired_orbit_action_matches_integer_mask_full_projected_h_oracle(
    hessian_context,
) -> None:
    key = Vituri2024HFSpiralFullSectorKey(1, 0, 0)
    prepared_orbit = hessian_context.build_orbit_hessian(key)
    assert type(prepared_orbit) is Vituri2024HFSpiralFullHessianOrbit
    assert type(prepared_orbit.first_embedding) is Vituri2024HFSpiralTransitionLaneEmbedding
    assert prepared_orbit.complex_dimension == (
        prepared_orbit.orbit.first_complex_dimension
        + prepared_orbit.orbit.second_complex_dimension
    )
    for embedding, lane_key in (
        (prepared_orbit.first_embedding, prepared_orbit.orbit.first),
        (prepared_orbit.second_embedding, prepared_orbit.orbit.second),
    ):
        actual_tuples = tuple(
            zip(
                embedding.particle_valley_slots.tolist(),
                embedding.particle_k_indices.tolist(),
                embedding.hole_valley_slots.tolist(),
                embedding.hole_k_indices.tolist(),
                strict=True,
            )
        )
        assert actual_tuples == _literal_transition_tuples(
            hessian_context.inventory, lane_key
        )
    rng = np.random.default_rng(83)
    first = rng.normal(size=prepared_orbit.orbit.first_complex_dimension) + 1.0j * rng.normal(
        size=prepared_orbit.orbit.first_complex_dimension
    )
    second = rng.normal(size=prepared_orbit.orbit.second_complex_dimension) + 1.0j * rng.normal(
        size=prepared_orbit.orbit.second_complex_dimension
    )
    density = _global_selected_tangent(prepared_orbit, first, second)
    sigma = _apply_full_integer_oracle(hessian_context.response, density)
    selected = np.asarray(
        hessian_context.inventory.restricted_preparation.receipt.selected_flavor_indices
    )
    expected = []
    for embedding, values, lane in (
        (
            prepared_orbit.first_embedding,
            first,
            prepared_orbit.hessian.frame.first,
        ),
        (
            prepared_orbit.second_embedding,
            second,
            prepared_orbit.hessian.frame.second,
        ),
    ):
        rows = selected[embedding.particle_valley_slots] * hessian_context.nk + embedding.particle_k_indices
        columns = selected[embedding.hole_valley_slots] * hessian_context.nk + embedding.hole_k_indices
        expected.append(lane.one_body_gaps_ev * values + sigma[rows, columns])
    actual = prepared_orbit.hessian.complex_gradient_jacobian_action(first, second)
    assert np.max(np.abs(actual[0] - expected[0]), initial=0.0) < 3.0e-13
    assert np.max(np.abs(actual[1] - expected[1]), initial=0.0) < 3.0e-13
    vector = prepared_orbit.hessian.frame.pack_complex(first, second)
    expected_real = prepared_orbit.hessian.frame.pack_complex(
        expected[0], expected[1], factor=2.0
    )
    assert np.max(
        np.abs(prepared_orbit.hessian.matvec(vector) - expected_real), initial=0.0
    ) < 3.0e-13
    diagnostic = prepared_orbit.hessian.check_bilinear_symmetry(
        seed=89, probe_count=8, atol=2.0e-12, rtol=2.0e-12
    )
    assert diagnostic.all_evaluated_pairs_symmetric


def test_paired_orbit_matches_exact_unitary_relative_scalar_curvature(
    hessian_context,
) -> None:
    prepared_orbit = hessian_context.build_orbit_hessian(
        Vituri2024HFSpiralFullSectorKey(1, -1, 2)
    )
    rng = np.random.default_rng(97)
    direction = rng.normal(size=prepared_orbit.real_dimension)
    direction /= np.linalg.norm(direction)
    first, second = prepared_orbit.hessian.frame.unpack_real(direction)
    tangent = _global_selected_tangent(prepared_orbit, first, second)
    selected = np.asarray(
        hessian_context.inventory.restricted_preparation.receipt.selected_flavor_indices
    )
    nk = hessian_context.nk
    selected_global = np.concatenate(
        (selected[0] * nk + np.arange(nk), selected[1] * nk + np.arange(nk))
    )
    tangent_selected = tangent[np.ix_(selected_global, selected_global)]
    occupations = hessian_context.inventory.selected_occupations.reshape(-1)
    projector0 = np.diag(occupations.astype(np.complex128))
    generator = np.zeros_like(projector0)
    for lane_key, values in (
        (prepared_orbit.orbit.first, first),
        (prepared_orbit.orbit.second, second),
    ):
        literal = _literal_transition_tuples(hessian_context.inventory, lane_key)
        assert len(literal) == len(values)
        for (particle_flavor, particle_k, hole_flavor, hole_k), value in zip(
            literal, values, strict=True
        ):
            particle_orbital = particle_flavor * nk + particle_k
            hole_orbital = hole_flavor * nk + hole_k
            generator[particle_orbital, hole_orbital] = value
            generator[hole_orbital, particle_orbital] = -value.conjugate()
    assert np.max(
        np.abs(generator @ projector0 - projector0 @ generator - tangent_selected),
        initial=0.0,
    ) < 2.0e-15
    source_hamiltonians = (
        hessian_context.inventory.restricted_preparation.selected_hamiltonians_conventional
    )
    independently_selected_diagonal = np.stack(
        (source_hamiltonians[0, 0].real, source_hamiltonians[1, 1].real)
    )
    assert np.array_equal(
        independently_selected_diagonal,
        hessian_context.selected_fock_diagonal_ev,
    )
    source_fock = np.diag(independently_selected_diagonal.reshape(-1))

    def relative_energy(parameter):
        unitary = expm(parameter * generator)
        projector = unitary @ projector0 @ unitary.conj().T
        delta_selected = projector - projector0
        delta_full = np.zeros((4 * nk, 4 * nk), dtype=np.complex128)
        delta_full[np.ix_(selected_global, selected_global)] = delta_selected
        sigma = _apply_full_integer_oracle(hessian_context.response, delta_full)
        sigma_selected = sigma[np.ix_(selected_global, selected_global)]
        return float(
            (
                np.trace(source_fock @ delta_selected)
                + 0.5 * np.trace(delta_selected @ sigma_selected)
            ).real
        )

    h = 0.01
    em2, em1, e0, ep1, ep2 = (
        relative_energy(multiplier * h)
        for multiplier in (-2.0, -1.0, 0.0, 1.0, 2.0)
    )
    stationarity = (em2 - 8.0 * em1 + 8.0 * ep1 - ep2) / (12.0 * h)
    finite_difference = (
        -ep2 + 16.0 * ep1 - 30.0 * e0 + 16.0 * em1 - em2
    ) / (12.0 * h * h)
    predicted = float(direction @ prepared_orbit.hessian.matvec(direction))
    assert abs(stationarity) < 2.0e-10
    assert abs(finite_difference - predicted) < 2.0e-8
    assert abs(finite_difference - 0.5 * predicted) > 1.0e-5
    assert abs(finite_difference - predicted / hessian_context.nk) > 1.0e-5


def test_unequal_and_one_empty_conjugate_lanes_are_not_padded_or_dropped(
    hessian_context,
) -> None:
    orbit = next(
        candidate
        for candidate in hessian_context.inventory.iter_conjugate_orbits(
            include_zero_dimension=False
        )
        if candidate.first_complex_dimension > 0
        and candidate.second_complex_dimension == 0
    )
    prepared_orbit = hessian_context.build_orbit_hessian(orbit.second)
    assert prepared_orbit.orbit == orbit
    assert prepared_orbit.hessian.frame.first_complex_dimension == orbit.first_complex_dimension
    assert prepared_orbit.hessian.frame.second_complex_dimension == 0
    assert prepared_orbit.complex_dimension == orbit.first_complex_dimension
    rng = np.random.default_rng(101)
    vector = rng.normal(size=prepared_orbit.real_dimension)
    first, second = prepared_orbit.hessian.frame.unpack_real(vector)
    assert second.shape == (0,)
    density = _global_selected_tangent(prepared_orbit, first, second)
    sigma = _apply_full_integer_oracle(hessian_context.response, density)
    selected = np.asarray(
        hessian_context.inventory.restricted_preparation.receipt.selected_flavor_indices
    )
    embedding = prepared_orbit.first_embedding
    rows = selected[embedding.particle_valley_slots] * hessian_context.nk + embedding.particle_k_indices
    columns = selected[embedding.hole_valley_slots] * hessian_context.nk + embedding.hole_k_indices
    expected_first = (
        prepared_orbit.hessian.frame.first.one_body_gaps_ev * first
        + sigma[rows, columns]
    )
    actual_first, actual_second = prepared_orbit.hessian.complex_gradient_jacobian_action(
        first, second
    )
    assert np.linalg.norm(expected_first) > 1.0e-8
    assert np.max(np.abs(actual_first - expected_first), initial=0.0) < 3.0e-13
    assert actual_second.shape == (0,)
    output = prepared_orbit.hessian.matvec(vector)
    expected_output = prepared_orbit.hessian.frame.pack_complex(
        expected_first, actual_second, factor=2.0
    )
    assert np.max(np.abs(output - expected_output), initial=0.0) < 3.0e-13


def test_hessian_context_builds_orbits_without_recursive_source_validation(
    hessian_context, monkeypatch
) -> None:
    def forbidden_recursive_validation(_self):
        raise AssertionError("per-orbit build recursively validated the full source")

    monkeypatch.setattr(
        Vituri2024HFSpiralFullSectorInventory,
        "validate_live_state",
        forbidden_recursive_validation,
    )
    prepared_orbit = hessian_context.build_orbit_hessian(
        Vituri2024HFSpiralFullSectorKey(1, 1, -2)
    )
    vector = np.zeros(prepared_orbit.real_dimension)
    assert np.array_equal(prepared_orbit.hessian.matvec(vector), vector)


def test_full_hessian_authority_and_exports(hessian_context) -> None:
    assert hessian_context.api_version == VITURI2024_HF_SPIRAL_FULL_HESSIAN_API_VERSION
    assert hessian_context.authority == VITURI2024_HF_SPIRAL_FULL_HESSIAN_AUTHORITY
    assert hessian_context.candidate_only
    assert hessian_context.selected_spin_only
    assert hessian_context.spectator_frozen_to_identity
    assert hessian_context.global_selected_rank
    assert hessian_context.all_occupation_transfers_in_inventory
    assert hessian_context.all_nonempty_inventory_orbits_buildable
    assert hessian_context.self_conjugate_sector_dimension_zero
    assert not hessian_context.q_and_minus_q_averaged
    assert not hessian_context.literal_float_full_functional_parity_established
    assert not hessian_context.scalar_curvature_established
    assert not hessian_context.reciprocity_established
    assert not hessian_context.hermitian_eigensolver_authorized
    assert not hessian_context.full_local_stability_established
    assert not hessian_context.production_ready
    assert not hessian_context.paper_reproduction_verified
    expected_exports = {
        "VITURI2024_HF_SPIRAL_FULL_HESSIAN_API_VERSION",
        "VITURI2024_HF_SPIRAL_FULL_HESSIAN_AUTHORITY",
        "Vituri2024HFSpiralFullHessianContext",
        "Vituri2024HFSpiralFullHessianOrbit",
        "Vituri2024HFSpiralTransitionLaneEmbedding",
        "build_vituri2024_hf_spiral_full_hessian_context",
    }
    from mean_field.systems.abc_trilayer import (
        vituri2024_hf_spiral_full_hessian as hessian_module,
    )

    assert set(hessian_module.__all__) == expected_exports
    for name in expected_exports:
        assert getattr(abc_trilayer, name) is getattr(hessian_module, name)
        assert name in abc_trilayer.__all__


def test_signed_response_authority_and_public_exports(response) -> None:
    assert response.api_version == VITURI2024_HF_SPIRAL_FULL_RESPONSE_API_VERSION
    assert response.authority == VITURI2024_HF_SPIRAL_FULL_RESPONSE_AUTHORITY
    assert response.dense_max_nk == VITURI2024_HF_SPIRAL_FULL_RESPONSE_DENSE_MAX_NK
    assert response.candidate_only
    assert response.no_wrap
    assert response.dense_reference_available_on_reduced_meshes
    assert response.fft_linear_convolution
    assert not response.q_and_minus_q_averaged
    assert response.selected_output_projection_with_frozen_spectator_input
    assert response.integer_label_interaction_conservation_convention
    assert not response.integer_label_interaction_conservation_established
    assert response.real_even_kernel_verified
    assert (
        response.kernel_orientation_contract
        == VITURI2024_HF_SPIRAL_FULL_RESPONSE_KERNEL_CONTRACT
    )
    assert not response.literal_float_full_functional_parity_established
    assert not response.dense_fft_parity_established
    assert not response.scalar_curvature_established
    assert not response.reciprocity_established
    assert not response.hermitian_eigensolver_authorized
    assert not response.full_local_stability_established
    assert not response.production_ready
    assert not response.paper_reproduction_verified
    assert len(response.fingerprint) == 64

    expected_exports = {
        "VITURI2024_HF_SPIRAL_FULL_RESPONSE_API_VERSION",
        "VITURI2024_HF_SPIRAL_FULL_RESPONSE_AUTHORITY",
        "VITURI2024_HF_SPIRAL_FULL_RESPONSE_DENSE_MAX_NK",
        "VITURI2024_HF_SPIRAL_FULL_RESPONSE_KERNEL_CONTRACT",
        "VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_API_VERSION",
        "VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_ARITHMETIC",
        "VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_MAX_MESH_SIZE",
        "VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_SCOPE",
        "Vituri2024HFSpiralLiteralMaskComparisonReceipt",
        "Vituri2024HFSpiralLiteralMaskEquivalenceReceipt",
        "Vituri2024HFSpiralSignedDisplacementResponse",
        "Vituri2024HFSpiralValidatedResponseActionFactory",
        "Vituri2024HFSpiralValidatedSignedDisplacementFFTAction",
        "build_vituri2024_hf_spiral_signed_displacement_response",
        "certify_vituri2024_hf_spiral_literal_mask_equivalence",
        "compare_vituri2024_hf_spiral_literal_mask_equivalence",
    }
    from mean_field.systems.abc_trilayer import (
        vituri2024_hf_spiral_full_response as response_module,
    )

    assert set(response_module.__all__) == expected_exports
    for name in expected_exports:
        assert getattr(abc_trilayer, name) is getattr(response_module, name)
        assert name in abc_trilayer.__all__


def test_public_package_exports_inventory_surface(inventory) -> None:
    expected_exports = {
        "VITURI2024_HF_SPIRAL_FULL_SECTOR_CHARGES",
        "VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_API_VERSION",
        "VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_AUTHORITY",
        "VITURI2024_HF_SPIRAL_FULL_SECTOR_MOMENTUM_CONTRACT",
        "Vituri2024HFSpiralFullSectorInventory",
        "Vituri2024HFSpiralFullSectorKey",
        "Vituri2024HFSpiralFullSectorOrbit",
        "build_vituri2024_hf_spiral_full_sector_inventory",
    }
    assert set(full_stability.__all__) == expected_exports
    for name in expected_exports:
        assert getattr(abc_trilayer, name) is getattr(full_stability, name)
        assert name in abc_trilayer.__all__
    assert type(inventory) is abc_trilayer.Vituri2024HFSpiralFullSectorInventory
