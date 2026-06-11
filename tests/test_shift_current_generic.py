from __future__ import annotations

import numpy as np

from analysis.response_derivative_gauge import (
    berry_connection_generalized_derivative_pair,
    shift_integrand_from_pair_generalized_derivative,
    wannierberri_shift_current_internal_imn,
)
from analysis.shift_current import (
    JOYA_EQ7_GEOMETRIC_CONVENTION,
    WANNIERBERRI_INTERNAL_IMN_CONVENTION,
    accumulate_fermi_omega_heatmap,
    component_kernel_from_gauge_pair,
    component_kernel_from_pair,
    component_transition_weight_from_gauge_pair,
    fermi_window_indices,
    lorentzian_delta,
    parse_component,
    positive_transition_terms,
    precompute_shift_current_tensors,
)
from mean_field.systems.htg.shift_current import (
    component_transition_weight_from_D as legacy_component_transition_weight_from_D,
    positive_transition_terms as legacy_positive_transition_terms,
    precompute_response_tensors as legacy_precompute_response_tensors,
)
from analysis.shift_current.toy_models.slg_toy import GappedSLGParams, d2hdk, dhdk, diagonalize


def _toy_point():
    params = GappedSLGParams(mass_ev=1.5, hopping_ev=2.73)
    k_xy = np.asarray([0.17, -0.09], dtype=float)
    evals, evecs = diagonalize(k_xy, params)
    return evals, evecs, dhdk(k_xy, params), d2hdk(k_xy, params)


def test_generic_precompute_matches_legacy_htg_response_tensors():
    evals, evecs, first, second = _toy_point()
    generic = precompute_shift_current_tensors(evals, evecs, first, d2hdk=second, denominator_cutoff_ev=1.0e-12)
    legacy = legacy_precompute_response_tensors(evals, evecs, first, d2hdk=second, denominator_cutoff_ev=1.0e-12)
    np.testing.assert_allclose(generic.velocity_h, legacy.D, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(generic.berry_connection, legacy.r, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(generic.berry_connection_gen_derivative, legacy.r_covariant, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(generic.occupations, legacy.occupations, rtol=0.0, atol=0.0)
    assert generic.skipped_small_denominators == legacy.skipped_small_denominators


def test_generic_symmetrized_transition_terms_match_legacy_htg_weights():
    evals, evecs, first, second = _toy_point()
    generic = precompute_shift_current_tensors(evals, evecs, first, d2hdk=second, denominator_cutoff_ev=1.0e-12)
    legacy = legacy_precompute_response_tensors(evals, evecs, first, d2hdk=second, denominator_cutoff_ev=1.0e-12)
    comp = parse_component("x;yy")
    trans_g, weights_g = positive_transition_terms(generic, comp, optical_symmetrization="sum")
    trans_l, weights_l = legacy_positive_transition_terms(legacy, comp.as_tuple)
    np.testing.assert_allclose(trans_g, trans_l, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(weights_g, weights_l, rtol=0.0, atol=1.0e-14)

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
    weight_pair, skipped_legacy = legacy_component_transition_weight_from_D(
        legacy.D,
        legacy.energies_ev,
        legacy.occupations,
        0,
        1,
        comp.as_tuple,
        denominator_cutoff_ev=1.0e-12,
        W=generic.second_velocity_h,
    )
    np.testing.assert_allclose(weight_pair, pair_generic.weight, rtol=0.0, atol=1.0e-15)
    assert skipped_legacy == pair_generic.skipped_small_denominators


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
