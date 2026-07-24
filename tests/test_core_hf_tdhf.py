from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf import (
    ParticleHolePair,
    TDHFMatrices,
    TDHFStructureResiduals,
    TDHF_TWO_BODY_CONVENTION,
    analyze_tdhf_signed_stability,
    assemble_tdhf_liouvillian,
    build_tdhf_signed_static_hessian,
    build_all_particle_hole_pairs,
    build_momentum_sector_particle_hole_pairs,
    build_tdhf_matrices,
    check_single_flavor_simplification,
    classify_tdhf_spectrum_stability,
    eigenvalue_pairing_residual,
    restrict_tdhf_matrices,
    solve_tdhf_liouvillian,
    signed_q_particle_hole_assignment_residual,
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
    np.testing.assert_allclose(
        np.sort(spectrum.raw_eta_norms),
        [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        atol=1e-12,
    )
    assert np.max(spectrum.raw_residuals) < 1e-12


def test_tdhf_stability_classifier_masks_complex_and_negative_branches_but_allows_zero() -> None:
    stable = solve_tdhf_liouvillian(assemble_tdhf_liouvillian(np.asarray([[1.0]]), np.asarray([[0.0]])))
    stable_class = classify_tdhf_spectrum_stability(stable, n_pairs=1)
    assert stable_class.stable
    assert not stable_class.masked
    assert stable_class.lowest_energy == 1.0

    complex_unstable = solve_tdhf_liouvillian(
        assemble_tdhf_liouvillian(np.asarray([[1.0]]), np.asarray([[2.0]]))
    )
    complex_class = classify_tdhf_spectrum_stability(complex_unstable, n_pairs=1)
    assert complex_class.masked
    assert complex_class.complex_eigenvalues
    assert "complex_raw_eigenvalues" in complex_class.reason

    negative_physical_branch = solve_tdhf_liouvillian(
        assemble_tdhf_liouvillian(np.asarray([[-1.0]]), np.asarray([[0.0]]))
    )
    negative_class = classify_tdhf_spectrum_stability(negative_physical_branch, n_pairs=1)
    assert negative_class.masked
    assert negative_class.missing_positive_metric_modes
    assert "selected_positive_metric_modes" in negative_class.reason

    exact_zero = solve_tdhf_liouvillian(
        assemble_tdhf_liouvillian(np.asarray([[0.0]]), np.asarray([[0.0]]))
    )
    zero_class = classify_tdhf_spectrum_stability(exact_zero, n_pairs=1)
    assert zero_class.stable
    assert zero_class.zero_mode_branches == 1


def _signed_scalar_matrices(
    a_q: complex,
    b_q: complex,
    a_minus_q: complex,
    b_minus_q: complex,
) -> tuple[TDHFMatrices, TDHFMatrices]:
    pairs = (ParticleHolePair(1, 0),)
    structure = TDHFStructureResiduals(
        a_hermitian=0.0,
        b_symmetric=0.0,
        particle_hole_symmetry=0.0,
        tolerance=1.0e-12,
    )
    plus_a = np.asarray([[a_q]], dtype=np.complex128)
    plus_b = np.asarray([[b_q]], dtype=np.complex128)
    minus_a = np.asarray([[a_minus_q]], dtype=np.complex128)
    minus_b = np.asarray([[b_minus_q]], dtype=np.complex128)
    # Supply ordinary same-sector Liouvillians deliberately. The signed-q
    # analyzer must rebuild L(q) and L(-q) from both sectors' A/B blocks.
    plus_l = assemble_tdhf_liouvillian(plus_a, plus_b)
    minus_l = assemble_tdhf_liouvillian(minus_a, minus_b)
    return (
        TDHFMatrices(pairs, plus_a, plus_b, plus_l, structure),
        TDHFMatrices(pairs, minus_a, minus_b, minus_l, structure),
    )


def test_tdhf_signed_static_stability_distinguishes_stable_negative_and_complex() -> None:
    stable_plus, stable_minus = _signed_scalar_matrices(2.0, 0.5, 2.0, 0.5)
    stable = analyze_tdhf_signed_stability(
        stable_plus, stable_minus, hessian_tol=1.0e-10, imag_tol=1.0e-10
    )
    np.testing.assert_allclose(stable.static.eigenvalues, [1.5, 2.5])
    assert stable.static.negative_count == 0
    assert stable.classification == "stable"

    negative_plus, negative_minus = _signed_scalar_matrices(-2.0, 0.5, -2.0, 0.5)
    negative = analyze_tdhf_signed_stability(
        negative_plus, negative_minus, hessian_tol=1.0e-10, imag_tol=1.0e-10
    )
    np.testing.assert_allclose(negative.static.eigenvalues, [-2.5, -1.5])
    assert negative.static.negative_count == 2
    assert negative.complex_count_plus == 0
    assert negative.complex_count_minus == 0
    assert negative.classification == "real_negative"

    complex_plus, complex_minus = _signed_scalar_matrices(0.5, 1.0, 0.5, 1.0)
    complex_result = analyze_tdhf_signed_stability(
        complex_plus, complex_minus, hessian_tol=1.0e-10, imag_tol=1.0e-10
    )
    np.testing.assert_allclose(complex_result.static.eigenvalues, [-0.5, 1.5])
    assert complex_result.static.negative_count == 1
    assert complex_result.complex_count_plus == 2
    assert complex_result.complex_count_minus == 2
    np.testing.assert_allclose(complex_result.max_abs_imag, np.sqrt(0.75))
    assert complex_result.classification == "complex"


def test_tdhf_signed_static_stability_identifies_goldstone_zero_direction() -> None:
    plus, minus = _signed_scalar_matrices(1.0, 1.0, 1.0, 1.0)
    result = analyze_tdhf_signed_stability(
        plus, minus, hessian_tol=1.0e-8, imag_tol=1.0e-8
    )
    np.testing.assert_allclose(result.static.eigenvalues, [0.0, 2.0], atol=1.0e-12)
    assert result.static.zero_count == 1
    assert result.static.negative_count == 0
    assert result.classification == "goldstone"


def test_tdhf_signed_generic_q_pairs_across_independent_minus_sector() -> None:
    plus, minus = _signed_scalar_matrices(2.0, 0.0, 3.0, 0.0)
    result = analyze_tdhf_signed_stability(
        plus, minus, hessian_tol=1.0e-10, imag_tol=1.0e-10
    )
    np.testing.assert_allclose(
        build_tdhf_signed_static_hessian(
            plus.A, plus.B, minus.A, minus.B
        ),
        np.diag([2.0, 3.0]),
    )
    assert eigenvalue_pairing_residual(result.plus_spectrum.raw_eigenvalues) == 1.0
    assert signed_q_particle_hole_assignment_residual(
        result.plus_spectrum.raw_eigenvalues,
        result.minus_spectrum.raw_eigenvalues,
    ) == 0.0
    assert result.signed_pairing_residual == 0.0
    assert result.classification == "stable"


def test_tdhf_signed_stability_rejects_empty_sectors() -> None:
    structure = TDHFStructureResiduals(0.0, 0.0, 0.0, 1.0e-12)
    empty = np.empty((0, 0), dtype=np.complex128)
    matrices = TDHFMatrices((), empty, empty, empty, structure)
    with pytest.raises(ValueError, match="nonempty"):
        analyze_tdhf_signed_stability(
            matrices,
            matrices,
            hessian_tol=1.0e-10,
            imag_tol=1.0e-10,
        )


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
