"""Focused tests for the full selected-spin no-wrap sector inventory."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

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


def _preparation(*, fft: bool):
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
            selected_spin=1,
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
