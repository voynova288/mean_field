from __future__ import annotations

import numpy as np

from mean_field.core.hf import (
    ParticleHolePair,
    TDHF_TWO_BODY_CONVENTION,
    assemble_tdhf_liouvillian,
    build_all_particle_hole_pairs,
    build_momentum_sector_particle_hole_pairs,
    build_tdhf_matrices,
    check_single_flavor_simplification,
    restrict_tdhf_matrices,
    solve_tdhf_liouvillian,
    solve_tdhf_matrices,
    split_pair_indices_by_flavor_channel,
    tdhf_metric_gram,
    transform_dense_two_body_to_hf_basis,
)


def test_tdhf_dense_v_convention_smoke_builds_expected_a_b_and_l() -> None:
    energies = np.asarray([0.0, 0.2, 1.5, 2.0], dtype=float)
    interaction = np.zeros((4, 4, 4, 4), dtype=np.complex128)

    # Diagonal A terms.
    interaction[2, 0, 0, 2] = 0.40
    interaction[2, 0, 2, 0] = 0.15
    interaction[3, 1, 1, 3] = 0.30
    interaction[3, 1, 3, 1] = 0.05

    # Off-diagonal A_{(2,0),(3,1)} and its Hermitian conjugate.
    interaction[2, 1, 0, 3] = 0.7 + 0.2j
    interaction[2, 1, 3, 0] = 0.1 - 0.1j
    interaction[3, 0, 1, 2] = 0.7 - 0.2j
    interaction[3, 0, 2, 1] = 0.1 + 0.1j

    # Symmetric B off-diagonal element.
    interaction[2, 3, 0, 1] = 0.5 + 0.4j
    interaction[2, 3, 1, 0] = 0.2 - 0.1j
    interaction[3, 2, 1, 0] = 0.5 + 0.4j
    interaction[3, 2, 0, 1] = 0.2 - 0.1j

    pairs = (ParticleHolePair(2, 0), ParticleHolePair(3, 1))
    matrices = build_tdhf_matrices(
        energies,
        pairs,
        interaction,
        raise_on_structure_error=True,
    )

    expected_a = np.asarray(
        [[1.75, 0.6 + 0.3j], [0.6 - 0.3j, 2.05]],
        dtype=np.complex128,
    )
    expected_b = np.asarray(
        [[0.0, 0.3 + 0.5j], [0.3 + 0.5j, 0.0]],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(matrices.A, expected_a)
    np.testing.assert_allclose(matrices.B, expected_b)
    np.testing.assert_allclose(matrices.L, assemble_tdhf_liouvillian(expected_a, expected_b))
    assert matrices.structure.ok
    assert "un-antisymmetrized" in TDHF_TWO_BODY_CONVENTION


def test_tdhf_dense_hf_basis_transform_identity_is_debug_safe() -> None:
    rng = np.random.default_rng(12)
    orbital_interaction = rng.normal(size=(3, 3, 3, 3)) + 1j * rng.normal(size=(3, 3, 3, 3))
    coeffs = np.eye(3, dtype=np.complex128)
    transformed = transform_dense_two_body_to_hf_basis(orbital_interaction, coeffs)
    np.testing.assert_allclose(transformed, orbital_interaction)


def test_tdhf_solver_returns_positive_metric_eta_orthonormal_branch() -> None:
    A = np.diag([1.0, 1.0, 2.0]).astype(np.complex128)
    B = np.zeros_like(A)
    spectrum = solve_tdhf_liouvillian(assemble_tdhf_liouvillian(A, B))

    np.testing.assert_allclose(spectrum.energies, [1.0, 1.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(spectrum.eta_norms, np.ones(3), atol=1e-12)
    np.testing.assert_allclose(
        tdhf_metric_gram(spectrum.amplitudes),
        np.eye(3, dtype=np.complex128),
        atol=1e-12,
    )
    assert spectrum.pairing_residual < 1e-12
    assert np.max(spectrum.residuals) < 1e-12


def test_tdhf_momentum_sector_builder_uses_fixed_collective_q() -> None:
    occupied_by_momentum = {0: [0], 1: [1]}
    unoccupied_by_momentum = {0: [2], 1: [3]}
    pairs = build_momentum_sector_particle_hole_pairs(
        occupied_by_momentum,
        unoccupied_by_momentum,
        1,
        lambda k, q: (k + q) % 2,
    )

    assert [(pair.particle, pair.hole) for pair in pairs] == [(3, 0), (2, 1)]
    assert [(pair.particle_momentum, pair.hole_momentum) for pair in pairs] == [(1, 0), (0, 1)]


def test_tdhf_flavor_sectors_recombine_to_full_dense_spectrum() -> None:
    flavors = {
        0: ("up", "K"),
        1: ("down", "K"),
        2: ("up", "K"),
        3: ("up", "Kprime"),
        4: ("down", "K"),
        5: ("down", "Kprime"),
    }
    pairs = build_all_particle_hole_pairs([0, 1], [2, 3, 4, 5], flavors=flavors)
    groups = split_pair_indices_by_flavor_channel(pairs)

    assert groups["intraflavor"].tolist() == [0, 6]
    assert groups["intervalley"].tolist() == [1, 7]
    assert groups["interspin"].tolist() == [2, 4]
    assert groups["inter_spin_valley"].tolist() == [3, 5]

    energies = np.asarray([0.0, 0.2, 1.0, 1.1, 1.2, 1.3], dtype=float)
    interaction = np.zeros((6, 6, 6, 6), dtype=np.complex128)
    full_matrices = build_tdhf_matrices(energies, pairs, interaction)
    full_spectrum = solve_tdhf_matrices(full_matrices)

    recombined: list[float] = []
    for indices in groups.values():
        restricted = restrict_tdhf_matrices(full_matrices, indices)
        recombined.extend(solve_tdhf_matrices(restricted).energies.tolist())
    np.testing.assert_allclose(np.sort(recombined), np.sort(full_spectrum.energies), atol=1e-12)


def test_tdhf_single_flavor_shortcut_requires_conduction_only_full_polarization() -> None:
    allowed = check_single_flavor_simplification(
        active_space_has_valence=False,
        occupied_flavor_counts={"K_up": 3, "Kprime_up": 0, "K_down": 0, "Kprime_down": 0},
        polarized_flavor="K_up",
    )
    assert allowed.allowed

    has_valence = check_single_flavor_simplification(
        active_space_has_valence=True,
        occupied_flavor_counts={"K_up": 3, "Kprime_up": 0},
        polarized_flavor="K_up",
    )
    assert not has_valence.allowed
    assert "valence" in has_valence.reason

    extra_occupied = check_single_flavor_simplification(
        active_space_has_valence=False,
        occupied_flavor_counts={"K_up": 3, "Kprime_up": 1},
        polarized_flavor="K_up",
    )
    assert not extra_occupied.allowed
    assert "non-polarized" in extra_occupied.reason
