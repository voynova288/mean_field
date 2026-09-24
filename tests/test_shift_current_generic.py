from __future__ import annotations

from dataclasses import replace

import numpy as np

from analysis.response_derivative_gauge import (
    berry_connection_generalized_derivative_pair,
    random_block_unitary,
    shift_integrand_from_pair_generalized_derivative,
    wannierberri_shift_current_group_trace,
    wannierberri_shift_current_internal_imn,
)
from analysis.shift_current import (
    JOYA_EQ7_GEOMETRIC_CONVENTION,
    SHIFT_CURRENT_PREFAC_UA_NM_PER_V2,
    WANNIERBERRI_INTERNAL_IMN_CONVENTION,
    WANNIERBERRI_POSITIVE_TRANSITION_PREFAC_UA_NM_PER_V2,
    accumulate_fermi_omega_heatmap,
    component_group_trace_amplitude,
    component_kernel_from_gauge_pair,
    component_kernel_from_pair,
    component_transition_weight,
    component_transition_weight_from_gauge_pair,
    exact_degenerate_group_transition_weight,
    fermi_window_indices,
    lorentzian_delta,
    parse_component,
    positive_transition_pairs,
    positive_transition_terms,
    precompute_shift_current_tensors,
    spectra_from_transition_table,
    wannierberri_positive_transition_conductivity_from_imn,
)
from analysis.shift_current.toy_models.slg_toy import GappedSLGParams, d2hdk, dhdk, diagonalize


def _toy_point():
    params = GappedSLGParams(mass_ev=1.5, hopping_ev=2.73)
    k_xy = np.asarray([0.17, -0.09], dtype=float)
    evals, evecs = diagonalize(k_xy, params)
    return evals, evecs, dhdk(k_xy, params), d2hdk(k_xy, params)


def test_generic_precompute_exposes_legacy_compatible_aliases():
    evals, evecs, first, second = _toy_point()
    generic = precompute_shift_current_tensors(evals, evecs, first, d2hdk=second, denominator_cutoff_ev=1.0e-12)
    np.testing.assert_allclose(generic.D, generic.velocity_h, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(generic.r, generic.berry_connection, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(generic.r_covariant, generic.berry_connection_gen_derivative, rtol=0.0, atol=0.0)
    assert generic.velocity_h.shape == generic.berry_connection.shape
    assert generic.berry_connection_gen_derivative.shape[:2] == generic.velocity_h.shape[:1] * 2
    assert generic.skipped_small_denominators == 0


def test_generic_symmetrized_transition_terms_match_explicit_weights():
    evals, evecs, first, second = _toy_point()
    generic = precompute_shift_current_tensors(evals, evecs, first, d2hdk=second, denominator_cutoff_ev=1.0e-12)
    comp = parse_component("x;yy")
    trans_g, weights_g = positive_transition_terms(generic, comp, optical_symmetrization="sum")
    pairs = positive_transition_pairs(generic.energies_ev, generic.occupations)
    trans_expected = np.asarray([transition_ev for _n, _m, transition_ev, _fnm in pairs], dtype=float)
    weights_expected = np.asarray(
        [
            component_transition_weight(generic, n, m, comp, optical_symmetrization="sum")
            for n, m, _transition_ev, _fnm in pairs
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(trans_g, trans_expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(weights_g, weights_expected, rtol=0.0, atol=1.0e-14)

    pair_generic = component_transition_weight_from_gauge_pair(
        generic.velocity_h,
        generic.energies_ev,
        generic.berry_connection,
        generic.occupations,
        0,
        1,
        comp,
        denominator_cutoff_ev=1.0e-12,
        second_velocity_h=generic.second_velocity_h,
        optical_symmetrization="sum",
    )
    full_tensor_weight = component_transition_weight(generic, 0, 1, comp, optical_symmetrization="sum")
    np.testing.assert_allclose(pair_generic.weight, full_tensor_weight, rtol=0.0, atol=1.0e-15)
    assert pair_generic.skipped_small_denominators == 0


def test_joya_named_convention_matches_ordered_pair_integrand_and_no_pi_lorentzian():
    evals, evecs, first, second = _toy_point()
    tensors = precompute_shift_current_tensors(evals, evecs, first, d2hdk=second, denominator_cutoff_ev=1.0e-12)
    pair = berry_connection_generalized_derivative_pair(
        tensors.velocity_h,
        tensors.energies_ev,
        0,
        1,
        denominator_cutoff=1.0e-12,
        second_velocity_h=tensors.second_velocity_h,
    )
    old = shift_integrand_from_pair_generalized_derivative(
        tensors.berry_connection,
        pair.values,
        initial_band=0,
        final_band=1,
        deriv_axis=1,
        optical_axis=1,
    )
    named = component_kernel_from_pair(
        tensors.berry_connection,
        pair.values,
        initial_band=0,
        final_band=1,
        component="yyy",
        convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
    )
    np.testing.assert_allclose(named, old, rtol=0.0, atol=1.0e-15)

    omega = np.asarray([0.0, 0.01, 0.02])
    normalized = lorentzian_delta(omega, 0.01, 0.001, normalized=True)
    joya = lorentzian_delta(omega, 0.01, 0.001, convention=JOYA_EQ7_GEOMETRIC_CONVENTION)
    np.testing.assert_allclose(joya, np.pi * normalized, rtol=1.0e-14, atol=0.0)


def test_wannierberri_named_convention_matches_internal_imn_same_polarization():
    evals, evecs, first, second = _toy_point()
    tensors = precompute_shift_current_tensors(evals, evecs, first, d2hdk=second, denominator_cutoff_ev=1.0e-12)
    sc_eta = 0.04
    imn = wannierberri_shift_current_internal_imn(
        tensors.velocity_h,
        tensors.energies_ev,
        second_velocity_h=tensors.second_velocity_h,
        sc_eta=sc_eta,
        denominator_cutoff=1.0e-12,
    )
    for comp in [(0, 0, 0), (1, 1, 1), (0, 1, 1), (1, 0, 0)]:
        selected_pair = component_kernel_from_gauge_pair(
            tensors.velocity_h,
            tensors.energies_ev,
            tensors.berry_connection,
            0,
            1,
            comp,
            denominator_cutoff_ev=1.0e-12,
            second_velocity_h=tensors.second_velocity_h,
            principal_value_eta_ev=sc_eta,
            convention=WANNIERBERRI_INTERNAL_IMN_CONVENTION,
        )
        np.testing.assert_allclose(selected_pair.kernel, imn[(0, 1) + comp], rtol=0.0, atol=1.0e-15)


def test_wannierberri_positive_transition_reduction_matches_both_ordered_pairs():
    r"""Nonzero analytic sign/factor oracle for the positive-frequency reduction.

    At k=0 take H(k)=diag(0,2)+k*sigma_x+k^2*(-sigma_y)/2.  In its
    eigenbasis V_01=1 and W_01=i.  With one Cartesian axis the fixed internal
    tensor is therefore Imn[0,1]=-2 Im(W_01 V_10)/Delta^2=-1/2 and its reverse
    is +1/2.  The two spectral numbers below stand for the two delta
    orientations in fixed ``ShiftCurrent.factor_omega``; retaining both also
    checks the finite-width algebra rather than relying on a zero antiresonant
    tail.
    """

    energies = np.asarray([0.0, 2.0])
    velocity_h = np.zeros((1, 2, 2), dtype=np.complex128)
    velocity_h[0] = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    second_velocity_h = np.zeros((1, 1, 2, 2), dtype=np.complex128)
    second_velocity_h[0, 0] = np.asarray([[0.0, 1.0j], [-1.0j, 0.0]])
    imn = wannierberri_shift_current_internal_imn(
        velocity_h,
        energies,
        second_velocity_h=second_velocity_h,
        sc_eta=0.3,
        denominator_cutoff=1.0e-12,
    )

    np.testing.assert_allclose(imn[0, 1, 0, 0, 0], -0.5, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(imn[1, 0, 0, 0, 0], +0.5, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(imn + np.swapaxes(imn, 0, 1), 0.0, rtol=0.0, atol=1.0e-15)
    assert abs(float(imn[0, 1, 0, 0, 0])) > 0.1

    f_initial = 1.0
    f_final = 0.0
    resonant_delta = 1.75
    antiresonant_delta = 0.25
    spectral_sum = resonant_delta + antiresonant_delta
    fixed_wannierberri_prefactor = 0.25 * SHIFT_CURRENT_PREFAC_UA_NM_PER_V2
    complete_two_pair_result = fixed_wannierberri_prefactor * (
        (f_final - f_initial) * imn[0, 1, 0, 0, 0] * spectral_sum
        + (f_initial - f_final) * imn[1, 0, 0, 0, 0] * spectral_sum
    )
    positive_only_imn_integral = (
        (f_initial - f_final) * imn[0, 1, 0, 0, 0] * spectral_sum
    )
    reduced_result = wannierberri_positive_transition_conductivity_from_imn(
        positive_only_imn_integral
    )

    np.testing.assert_allclose(
        WANNIERBERRI_POSITIVE_TRANSITION_PREFAC_UA_NM_PER_V2,
        -0.5 * SHIFT_CURRENT_PREFAC_UA_NM_PER_V2,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(reduced_result, complete_two_pair_result, rtol=0.0, atol=1.0e-15)
    assert float(complete_two_pair_result) > 0.0
    legacy_wrong_result = SHIFT_CURRENT_PREFAC_UA_NM_PER_V2 * positive_only_imn_integral
    np.testing.assert_allclose(legacy_wrong_result, -2.0 * complete_two_pair_result, rtol=0.0, atol=1.0e-15)


def test_pair_api_supports_full_virtual_sum_without_full_gd_tensor():
    evals, evecs, first, second = _toy_point()
    tensors = precompute_shift_current_tensors(evals, evecs, first, d2hdk=second, denominator_cutoff_ev=1.0e-12)
    pair_weight = component_transition_weight_from_gauge_pair(
        tensors.velocity_h,
        tensors.energies_ev,
        tensors.berry_connection,
        tensors.occupations,
        0,
        1,
        "y;yy",
        denominator_cutoff_ev=1.0e-12,
        second_velocity_h=tensors.second_velocity_h,
        convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
    )
    direct_pair = component_kernel_from_gauge_pair(
        tensors.velocity_h,
        tensors.energies_ev,
        tensors.berry_connection,
        0,
        1,
        "y;yy",
        denominator_cutoff_ev=1.0e-12,
        second_velocity_h=tensors.second_velocity_h,
        convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
    )
    fnm = tensors.occupations[0] - tensors.occupations[1]
    np.testing.assert_allclose(pair_weight.kernel, fnm * direct_pair.kernel, rtol=0.0, atol=1.0e-15)


def test_heatmap_helpers_make_fermi_window_convention_explicit():
    omega = np.asarray([0.0, 0.01, 0.02])
    fermi = np.asarray([-0.02, -0.01, 0.0, 0.01, 0.02])
    assert fermi_window_indices(fermi, -0.005, 0.015) == (2, 4)
    heat = np.zeros((fermi.size, omega.size), dtype=float)
    used = accumulate_fermi_omega_heatmap(
        heat,
        fermi,
        omega,
        initial_energy_ev=-0.005,
        final_energy_ev=0.015,
        transition_energy_ev=0.01,
        amplitude=2.0,
        eta_ev=0.001,
        convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
    )
    assert used
    assert np.all(heat[:2] == 0.0)
    assert np.all(heat[4:] == 0.0)
    expected = np.repeat(2.0 * lorentzian_delta(omega, 0.01, 0.001, convention=JOYA_EQ7_GEOMETRIC_CONVENTION)[None, :], 2, axis=0)
    np.testing.assert_allclose(heat[2:4], expected)


def _hermitian_from_rng(rng: np.random.Generator, size: int, *, scale: float) -> np.ndarray:
    raw = scale * (rng.normal(size=(size, size)) + 1.0j * rng.normal(size=(size, size)))
    return 0.5 * (raw + raw.conjugate().T)


def _unitary_exponential(hermitian_generator: np.ndarray, coordinate: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian_generator)
    phases = np.exp(-1.0j * float(coordinate) * eigenvalues)
    return (eigenvectors * phases[None, :]) @ eigenvectors.conjugate().T


def _exact_doublet_operator_data(sample: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Return a smooth six-band family with two exact active doublets.

    ``H(k)=S(k) E(k) S(k)^dagger`` uses
    ``S=exp(k_x K_x) exp(k_y K_y)`` with independent noncommuting generators.
    The sorted diagonal ``E(k)`` contains a remote valence singleton, an exact
    occupied doublet, an exact empty doublet, and a remote conduction singleton.
    Analytic first/second derivatives therefore exercise nontrivial optical
    index order and a genuine intermediate-state principal-value sum.
    """

    rng = np.random.default_rng(9100)
    generator_x_h = _hermitian_from_rng(rng, 6, scale=0.24)
    generator_y_h = _hermitian_from_rng(rng, 6, scale=0.19)
    generator_x = -1.0j * generator_x_h
    generator_y = -1.0j * generator_y_h
    coordinates = np.asarray([0.047 * sample, -0.031 * sample], dtype=float)
    unitary_x = _unitary_exponential(generator_x_h, coordinates[0])
    unitary_y = _unitary_exponential(generator_y_h, coordinates[1])
    eigenvectors = unitary_x @ unitary_y

    base_energies = np.asarray([-1.32, -0.64, -0.64, 0.53, 0.53, 1.41])
    gradients0 = np.asarray(
        [
            [-0.025, 0.031],
            [0.070, -0.035],
            [0.070, -0.035],
            [0.130, 0.052],
            [0.130, 0.052],
            [0.041, -0.083],
        ]
    )
    hessians = np.asarray(
        [
            [[0.009, 0.004], [0.004, -0.006]],
            [[0.018, -0.007], [-0.007, 0.011]],
            [[0.018, -0.007], [-0.007, 0.011]],
            [[-0.014, 0.009], [0.009, 0.016]],
            [[-0.014, 0.009], [0.009, 0.016]],
            [[0.006, -0.005], [-0.005, 0.013]],
        ]
    )
    energies = base_energies + gradients0 @ coordinates + 0.5 * np.einsum(
        "i,bij,j->b", coordinates, hessians, coordinates
    )
    gradients = gradients0 + np.einsum("bij,j->bi", hessians, coordinates)
    energy_matrix = np.diag(energies).astype(np.complex128)
    energy_first = np.asarray([np.diag(gradients[:, axis]) for axis in range(2)], dtype=np.complex128)
    energy_second = np.empty((2, 2, 6, 6), dtype=np.complex128)
    for axis_a in range(2):
        for axis_b in range(2):
            energy_second[axis_a, axis_b] = np.diag(hessians[:, axis_a, axis_b])

    unitary_first = np.empty((2, 6, 6), dtype=np.complex128)
    unitary_first[0] = unitary_x @ generator_x @ unitary_y
    unitary_first[1] = eigenvectors @ generator_y
    unitary_second = np.empty((2, 2, 6, 6), dtype=np.complex128)
    unitary_second[0, 0] = unitary_x @ generator_x @ generator_x @ unitary_y
    unitary_second[0, 1] = unitary_x @ generator_x @ unitary_y @ generator_y
    unitary_second[1, 0] = unitary_second[0, 1]
    unitary_second[1, 1] = eigenvectors @ generator_y @ generator_y

    first = np.empty((2, 6, 6), dtype=np.complex128)
    second = np.empty((2, 2, 6, 6), dtype=np.complex128)
    for axis_a in range(2):
        first[axis_a] = (
            unitary_first[axis_a] @ energy_matrix @ eigenvectors.conjugate().T
            + eigenvectors @ energy_first[axis_a] @ eigenvectors.conjugate().T
            + eigenvectors @ energy_matrix @ unitary_first[axis_a].conjugate().T
        )
        for axis_b in range(2):
            second[axis_a, axis_b] = (
                unitary_second[axis_a, axis_b] @ energy_matrix @ eigenvectors.conjugate().T
                + unitary_first[axis_a] @ energy_first[axis_b] @ eigenvectors.conjugate().T
                + unitary_first[axis_a] @ energy_matrix @ unitary_first[axis_b].conjugate().T
                + unitary_first[axis_b] @ energy_first[axis_a] @ eigenvectors.conjugate().T
                + eigenvectors @ energy_second[axis_a, axis_b] @ eigenvectors.conjugate().T
                + eigenvectors @ energy_first[axis_a] @ unitary_first[axis_b].conjugate().T
                + unitary_first[axis_b] @ energy_matrix @ unitary_first[axis_a].conjugate().T
                + eigenvectors @ energy_first[axis_b] @ unitary_first[axis_a].conjugate().T
                + eigenvectors @ energy_matrix @ unitary_second[axis_a, axis_b].conjugate().T
            )
    return energies, eigenvectors, first, second


def _trapezoid(values: np.ndarray, grid: np.ndarray) -> float:
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(grid)))


def _spectrum_covariance_residuals(
    photon_energies_ev: np.ndarray,
    reference: np.ndarray,
    rotated: np.ndarray,
) -> dict[str, float]:
    difference = np.asarray(rotated, dtype=float) - np.asarray(reference, dtype=float)
    reference = np.asarray(reference, dtype=float)
    max_scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
    l2_scale = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    integrated_scale = max(_trapezoid(np.abs(reference), photon_energies_ev), np.finfo(float).tiny)
    reference_peak = float(photon_energies_ev[int(np.argmax(np.abs(reference)))])
    rotated_peak = float(photon_energies_ev[int(np.argmax(np.abs(rotated)))])
    return {
        "max_spectrum_residual": float(np.max(np.abs(difference))) / max_scale,
        "l2_spectrum_residual": float(np.linalg.norm(difference)) / l2_scale,
        "peak_energy_residual_ev": abs(rotated_peak - reference_peak),
        "integrated_weight_residual": abs(_trapezoid(difference, photon_energies_ev)) / integrated_scale,
        "max_spectrum_residual_absolute": float(np.max(np.abs(difference))),
        "l2_spectrum_residual_absolute": float(np.linalg.norm(difference)),
        "integrated_weight_residual_absolute": abs(_trapezoid(difference, photon_energies_ev)),
    }


EXACT_DEGENERATE_UN_RESIDUAL_TARGET = 1.0e-8
EXACT_DEGENERATE_UN_RESIDUAL_KEYS = (
    "max_spectrum_residual",
    "l2_spectrum_residual",
    "peak_energy_residual_ev",
    "integrated_weight_residual",
)
EXACT_DEGENERATE_UN_ABSOLUTE_RESIDUAL_KEYS = (
    "max_spectrum_residual_absolute",
    "l2_spectrum_residual_absolute",
    "integrated_weight_residual_absolute",
)
EXACT_DEGENERATE_UN_NONVACUITY_FLOORS = {
    "max_pair_weight_change": 1.0e-8,
    "max_principal_value_effect": 1.0e-10,
    "max_ordered_index_difference": 1.0e-10,
    "max_reference_weight_imag_abs": 1.0e-10,
    "reference_spectrum_l2_norm": 1.0e-10,
}

def _assert_group_contract_rejects(
    tensors,
    initial_group: tuple[int, ...],
    final_group: tuple[int, ...],
    expected_message: str,
) -> None:
    try:
        exact_degenerate_group_transition_weight(
            tensors,
            initial_group,
            final_group,
            (0, 0, 1),
            convention=WANNIERBERRI_INTERNAL_IMN_CONVENTION,
        )
    except ValueError as exc:
        assert expected_message in str(exc)
        return
    raise AssertionError("strict exact-degenerate group contract unexpectedly accepted invalid input")


def test_exact_degenerate_group_contract_rejects_partial_split_and_nonsmooth_blocks():
    energies, eigenvectors, first, second = _exact_doublet_operator_data(0)
    tensors = precompute_shift_current_tensors(
        energies,
        eigenvectors,
        first,
        d2hdk=second,
        denominator_cutoff_ev=1.0e-12,
        principal_value_eta_ev=0.035,
    )
    _assert_group_contract_rejects(tensors, (1,), (3, 4), "complete exact eigenspace")

    split_energies = tensors.energies_ev.copy()
    split_energies[2] += 1.0e-12
    _assert_group_contract_rejects(
        replace(tensors, energies_ev=split_energies),
        (1, 2),
        (3, 4),
        "not exactly degenerate",
    )

    unequal_occupations = tensors.occupations.copy()
    unequal_occupations[2] = 0.5
    _assert_group_contract_rejects(
        replace(tensors, occupations=unequal_occupations),
        (1, 2),
        (3, 4),
        "occupations are not identical",
    )

    nonscalar_velocity = tensors.velocity_h.copy()
    nonscalar_velocity[0, 1, 2] = 1.0e-4
    nonscalar_velocity[0, 2, 1] = 1.0e-4
    _assert_group_contract_rejects(
        replace(tensors, velocity_h=nonscalar_velocity),
        (1, 2),
        (3, 4),
        "necessary first-order exact-block condition",
    )


def test_exact_group_singletons_reduce_to_ordinary_nondegenerate_pair():
    evals, evecs, first, second = _toy_point()
    tensors = precompute_shift_current_tensors(
        evals,
        evecs,
        first,
        d2hdk=second,
        denominator_cutoff_ev=1.0e-12,
        principal_value_eta_ev=0.035,
    )
    grouped = exact_degenerate_group_transition_weight(
        tensors,
        (0,),
        (1,),
        (0, 0, 1),
        convention=WANNIERBERRI_INTERNAL_IMN_CONVENTION,
    )
    ordinary = component_transition_weight(
        tensors,
        0,
        1,
        (0, 0, 1),
        convention=WANNIERBERRI_INTERNAL_IMN_CONVENTION,
    )
    np.testing.assert_allclose(grouped.weight, ordinary, rtol=0.0, atol=1.0e-14)


def exact_degenerate_un_spectrum_covariance_benchmark() -> dict[str, float]:
    """Measure the P2.1 exact-U(N) gate through the generic response path.

    Independent seeded U(2) rotations are applied to the occupied and empty
    exact doublets at every sampled k point. Pair-resolved weights must change;
    the proper group trace and the resulting broadened spectrum must not. No
    nondegenerate bands are mixed. This callable is shared by pytest and the
    machine-readable Slurm runner, so there is only one benchmark/formula path.
    """

    initial_group = (1, 2)
    final_group = (3, 4)
    groups = [(1, 3), (3, 5)]
    component = (0, 0, 1)
    sc_eta_ev = 0.035
    transitions_reference: list[float] = []
    transitions_rotated: list[float] = []
    weights_reference: list[complex] = []
    weights_rotated: list[complex] = []
    max_pair_weight_change = 0.0
    max_principal_value_effect = 0.0
    max_ordered_index_difference = 0.0

    for sample in range(5):
        energies, eigenvectors, first, second = _exact_doublet_operator_data(sample)
        reference = precompute_shift_current_tensors(
            energies,
            eigenvectors,
            first,
            d2hdk=second,
            denominator_cutoff_ev=1.0e-12,
            principal_value_eta_ev=sc_eta_ev,
        )
        unregularized = precompute_shift_current_tensors(
            energies,
            eigenvectors,
            first,
            d2hdk=second,
            denominator_cutoff_ev=1.0e-12,
            principal_value_eta_ev=0.0,
        )
        reference_if = reference.berry_connection_gen_derivative[:, :, initial_group][:, :, :, final_group]
        unregularized_if = unregularized.berry_connection_gen_derivative[:, :, initial_group][:, :, :, final_group]
        max_principal_value_effect = max(
            max_principal_value_effect,
            float(np.max(np.abs(reference_if - unregularized_if))),
        )

        gauge = random_block_unitary(groups, 6, rng=20260713 + sample)
        rotated = precompute_shift_current_tensors(
            energies,
            eigenvectors @ gauge,
            first,
            d2hdk=second,
            denominator_cutoff_ev=1.0e-12,
            principal_value_eta_ev=sc_eta_ev,
        )

        gauge_initial = gauge[np.ix_(initial_group, initial_group)]
        gauge_final = gauge[np.ix_(final_group, final_group)]
        for axis in range(2):
            reference_fi = reference.berry_connection[axis][np.ix_(final_group, initial_group)]
            rotated_fi = rotated.berry_connection[axis][np.ix_(final_group, initial_group)]
            expected_fi = gauge_final.conjugate().T @ reference_fi @ gauge_initial
            np.testing.assert_allclose(rotated_fi, expected_fi, rtol=0.0, atol=2.0e-14)
            for connection_axis in range(2):
                reference_if = reference.berry_connection_gen_derivative[axis, connection_axis][
                    np.ix_(initial_group, final_group)
                ]
                rotated_if = rotated.berry_connection_gen_derivative[axis, connection_axis][
                    np.ix_(initial_group, final_group)
                ]
                expected_if = gauge_initial.conjugate().T @ reference_if @ gauge_final
                np.testing.assert_allclose(rotated_if, expected_if, rtol=0.0, atol=2.0e-13)

        reference_group = exact_degenerate_group_transition_weight(
            reference,
            initial_group,
            final_group,
            component,
            convention=WANNIERBERRI_INTERNAL_IMN_CONVENTION,
        )
        rotated_group = exact_degenerate_group_transition_weight(
            rotated,
            initial_group,
            final_group,
            component,
            convention=WANNIERBERRI_INTERNAL_IMN_CONVENTION,
        )
        direct_trace = component_group_trace_amplitude(
            reference.berry_connection,
            reference.berry_connection_gen_derivative,
            initial_group=initial_group,
            final_group=final_group,
            component=component,
            convention=WANNIERBERRI_INTERNAL_IMN_CONVENTION,
        )
        ordered_trace = component_group_trace_amplitude(
            reference.berry_connection,
            reference.berry_connection_gen_derivative,
            initial_group=initial_group,
            final_group=final_group,
            component=component,
            optical_symmetrization="none",
        )
        manual_ordered = np.trace(
            reference.berry_connection_gen_derivative[0, 1][np.ix_(initial_group, final_group)]
            @ reference.berry_connection[0][np.ix_(final_group, initial_group)]
        )
        swapped_ordered = np.trace(
            reference.berry_connection_gen_derivative[0, 0][np.ix_(initial_group, final_group)]
            @ reference.berry_connection[1][np.ix_(final_group, initial_group)]
        )
        max_ordered_index_difference = max(
            max_ordered_index_difference,
            float(abs(manual_ordered - swapped_ordered)),
        )
        np.testing.assert_allclose(ordered_trace, manual_ordered, rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(reference_group.geometric_amplitude, direct_trace, rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(rotated_group.weight, reference_group.weight, rtol=0.0, atol=2.0e-13)

        reference_pairs = np.asarray(
            [
                component_transition_weight(
                    reference,
                    n,
                    m,
                    component,
                    convention=WANNIERBERRI_INTERNAL_IMN_CONVENTION,
                )
                for n in initial_group
                for m in final_group
            ],
            dtype=np.complex128,
        )
        rotated_pairs = np.asarray(
            [
                component_transition_weight(
                    rotated,
                    n,
                    m,
                    component,
                    convention=WANNIERBERRI_INTERNAL_IMN_CONVENTION,
                )
                for n in initial_group
                for m in final_group
            ],
            dtype=np.complex128,
        )
        np.testing.assert_allclose(reference_group.weight, reference_pairs.sum(), rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(rotated_group.weight, rotated_pairs.sum(), rtol=0.0, atol=1.0e-14)
        max_pair_weight_change = max(max_pair_weight_change, float(np.max(np.abs(rotated_pairs - reference_pairs))))

        reference_imn = wannierberri_shift_current_internal_imn(
            reference.velocity_h,
            reference.energies_ev,
            sc_eta=sc_eta_ev,
            second_velocity_h=reference.second_velocity_h,
            denominator_cutoff=1.0e-12,
        )
        rotated_imn = wannierberri_shift_current_internal_imn(
            rotated.velocity_h,
            rotated.energies_ev,
            sc_eta=sc_eta_ev,
            second_velocity_h=rotated.second_velocity_h,
            denominator_cutoff=1.0e-12,
        )
        reference_wb_trace = wannierberri_shift_current_group_trace(reference_imn, initial_group, final_group)[component]
        rotated_wb_trace = wannierberri_shift_current_group_trace(rotated_imn, initial_group, final_group)[component]
        np.testing.assert_allclose(
            reference_group.kernel,
            reference_group.occupation_difference * reference_wb_trace,
            rtol=0.0,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            rotated_group.kernel,
            rotated_group.occupation_difference * rotated_wb_trace,
            rtol=0.0,
            atol=2.0e-13,
        )

        transitions_reference.append(reference_group.transition_energy_ev)
        transitions_rotated.append(rotated_group.transition_energy_ev)
        weights_reference.append(reference_group.weight)
        weights_rotated.append(rotated_group.weight)

    photon_energies_ev = np.linspace(1.0, 1.42, 421)
    reference_spectrum = spectra_from_transition_table(
        photon_energies_ev,
        {"x;xy": (np.asarray(transitions_reference), np.asarray(weights_reference))},
        k_weight_nm_inv_sq=1.0,
        eta_ev=0.012,
        prefactor=1.0,
        prefactor_phase=-1.0j,
        include_bz_factor=False,
        convention=WANNIERBERRI_INTERNAL_IMN_CONVENTION,
    )["x;xy"]
    rotated_spectrum = spectra_from_transition_table(
        photon_energies_ev,
        {"x;xy": (np.asarray(transitions_rotated), np.asarray(weights_rotated))},
        k_weight_nm_inv_sq=1.0,
        eta_ev=0.012,
        prefactor=1.0,
        prefactor_phase=-1.0j,
        include_bz_factor=False,
        convention=WANNIERBERRI_INTERNAL_IMN_CONVENTION,
    )["x;xy"]
    residuals = _spectrum_covariance_residuals(photon_energies_ev, reference_spectrum, rotated_spectrum)
    return {
        **residuals,
        # These measurements rule out a vacuous zero-equals-zero spectrum and
        # a fixture whose labeled pair decomposition was already invariant.
        "max_pair_weight_change": max_pair_weight_change,
        "max_principal_value_effect": max_principal_value_effect,
        "max_ordered_index_difference": max_ordered_index_difference,
        "max_reference_weight_imag_abs": float(
            np.max(np.abs(np.imag(np.asarray(weights_reference))))
        ),
        "reference_spectrum_l2_norm": float(np.linalg.norm(reference_spectrum)),
    }

def test_exact_degenerate_group_trace_spectrum_has_un_covariance_product_gate():
    measured = exact_degenerate_un_spectrum_covariance_benchmark()

    for key, floor in EXACT_DEGENERATE_UN_NONVACUITY_FLOORS.items():
        assert measured[key] > floor, measured
    for key in EXACT_DEGENERATE_UN_RESIDUAL_KEYS:
        assert measured[key] < EXACT_DEGENERATE_UN_RESIDUAL_TARGET, measured
