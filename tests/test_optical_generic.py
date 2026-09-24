from __future__ import annotations

import numpy as np
import pytest

from analysis.injection_current import (
    accumulate_injection_spectrum,
    positive_injection_transition_terms,
    precompute_injection_current_tensors,
)
from analysis.injection_current.toy_models.haldane import dhdk as haldane_dhdk
from analysis.injection_current.toy_models.haldane import diagonalize as haldane_diagonalize
from analysis.optical import (
    accumulate_optical_spectrum,
    canonical_response_kind,
    optical_spectra_from_transition_table,
    positive_optical_transition_terms,
    precompute_optical_tensors,
)
from analysis.shift_current import (
    SHIFT_CURRENT_PREFAC_UA_NM_PER_V2,
    accumulate_fermi_omega_heatmap,
    add_transitions_to_integral,
    conductivity_from_integral,
    positive_transition_terms,
    precompute_shift_current_tensors,
)
from analysis.shift_current.toy_models.slg_toy import GappedSLGParams
from analysis.shift_current.toy_models.slg_toy import d2hdk as slg_d2hdk
from analysis.shift_current.toy_models.slg_toy import dhdk as slg_dhdk
from analysis.shift_current.toy_models.slg_toy import diagonalize as slg_diagonalize


def _slg_point():
    params = GappedSLGParams(mass_ev=1.5, hopping_ev=2.73)
    k_xy = np.asarray([0.17, -0.09], dtype=float)
    evals, evecs = slg_diagonalize(k_xy, params)
    return evals, evecs, slg_dhdk(k_xy, params), slg_d2hdk(k_xy, params)


def _haldane_point():
    k_xy = np.asarray([0.29, 0.17], dtype=float)
    evals, evecs = haldane_diagonalize(k_xy)
    return evals, evecs, haldane_dhdk(k_xy)


def test_canonical_response_kind_aliases():
    assert canonical_response_kind("shift") == "shift_current"
    assert canonical_response_kind("shift-current") == "shift_current"
    assert canonical_response_kind("injection") == "injection_current"
    assert canonical_response_kind("cpge") == "injection_current"
    assert canonical_response_kind("linear") == "linear_conductivity"
    assert canonical_response_kind("linear-conductivity") == "linear_conductivity"
    assert canonical_response_kind("conductivity") == "linear_conductivity"
    assert canonical_response_kind("kerr") == "kerr_faraday"
    assert canonical_response_kind("faraday") == "kerr_faraday"
    assert canonical_response_kind("kerr-faraday") == "kerr_faraday"
    with pytest.raises(ValueError):
        canonical_response_kind("berry")
    with pytest.raises(ValueError):
        canonical_response_kind("liu_dai_2020_printed")


def test_optical_shift_current_front_door_matches_shift_module():
    evals, evecs, first, second = _slg_point()
    generic = precompute_optical_tensors(
        "shift_current",
        evals,
        evecs,
        first,
        d2hdk=second,
        denominator_cutoff_ev=1.0e-12,
    )
    direct = precompute_shift_current_tensors(
        evals,
        evecs,
        first,
        d2hdk=second,
        denominator_cutoff_ev=1.0e-12,
    )
    assert generic.kind == "shift_current"
    np.testing.assert_allclose(generic.velocity_h, direct.velocity_h, rtol=0.0, atol=0.0)
    transitions_g, weights_g = positive_optical_transition_terms(generic, "x;yy")
    transitions_d, weights_d = positive_transition_terms(direct, "x;yy")
    np.testing.assert_allclose(transitions_g, transitions_d, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(weights_g, weights_d, rtol=0.0, atol=1.0e-14)


def test_optical_injection_front_door_matches_injection_module():
    evals, evecs, first = _haldane_point()
    generic = precompute_optical_tensors("cpge", evals, evecs, first, denominator_cutoff_ev=1.0e-12)
    direct = precompute_injection_current_tensors(evals, evecs, first, denominator_cutoff_ev=1.0e-12)
    assert generic.kind == "injection_current"
    np.testing.assert_allclose(generic.berry_connection, direct.berry_connection, rtol=0.0, atol=0.0)
    transitions_g, weights_g = positive_optical_transition_terms(generic, "y;yy")
    transitions_d, weights_d = positive_injection_transition_terms(direct, "y;yy")
    np.testing.assert_allclose(transitions_g, transitions_d, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(weights_g, weights_d, rtol=0.0, atol=1.0e-14)


def test_optical_spectrum_dispatch_matches_direct_helpers():
    omega = np.linspace(0.1, 3.0, 7)
    transitions = np.asarray([0.8, 1.4], dtype=float)
    weights = np.asarray([1.0 + 0.5j, -0.25 + 0.75j], dtype=np.complex128)
    eta = 0.04
    k_weight = 0.11

    shift_generic = accumulate_optical_spectrum(
        "shift_current",
        omega,
        transitions,
        weights,
        k_weight=k_weight,
        eta_ev=eta,
    )
    integral = np.zeros_like(omega, dtype=np.complex128)
    add_transitions_to_integral(
        integral,
        omega,
        transitions,
        weights,
        k_weight_nm_inv_sq=k_weight,
        eta_ev=eta,
    )
    with pytest.raises(TypeError, match="prefactor"):
        conductivity_from_integral(integral)
    shift_direct = conductivity_from_integral(
        integral,
        prefactor=SHIFT_CURRENT_PREFAC_UA_NM_PER_V2,
        phase=-1.0j,
    )
    np.testing.assert_allclose(shift_generic, shift_direct, rtol=0.0, atol=1.0e-14)

    inj_generic = accumulate_optical_spectrum(
        "injection_current",
        omega,
        transitions,
        weights,
        k_weight=k_weight,
        eta_ev=eta,
    )
    inj_direct = accumulate_injection_spectrum(
        omega,
        transitions,
        weights,
        k_weight=k_weight,
        eta_ev=eta,
    )
    np.testing.assert_allclose(inj_generic, inj_direct, rtol=0.0, atol=1.0e-14)


def test_optical_rejects_shift_only_regularizer_for_injection():
    evals, evecs, first = _haldane_point()
    with pytest.raises(ValueError):
        precompute_optical_tensors(
            "injection_current",
            evals,
            evecs,
            first,
            principal_value_eta_ev=0.01,
        )


def test_transition_table_helpers_direct_linear_and_kerr_to_kpoint_api():
    evals, evecs, first = _haldane_point()
    with pytest.raises(ValueError, match="optical_response_from_kpoint_data"):
        precompute_optical_tensors("kerr", evals, evecs, first)

    omega = np.asarray([0.1, 0.2])
    transitions = np.asarray([0.15])
    weights = np.asarray([1.0 + 0.0j])
    with pytest.raises(ValueError, match="optical_response_from_kpoint_data"):
        accumulate_optical_spectrum(
            "linear",
            omega,
            transitions,
            weights,
            k_weight=1.0,
            eta_ev=0.01,
        )
    with pytest.raises(ValueError, match="optical_response_from_kpoint_data"):
        optical_spectra_from_transition_table(
            "faraday",
            omega,
            {"xx": (transitions, weights)},
            k_weight=1.0,
            eta_ev=0.01,
        )
