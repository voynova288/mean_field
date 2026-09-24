from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from analysis.injection_current import (
    accumulate_injection_spectrum,
    cpge_part,
    injection_component_kernel,
    linear_injection_part,
    positive_injection_transition_terms,
    precompute_injection_current_tensors,
)
from analysis.injection_current.toy_models.haldane import (
    HaldaneParams,
    compute_haldane_injection_spectra,
    compute_haldane_shift_spectra,
    d2hdk,
    dhdk,
    diagonalize,
    haldane_shift_tensors_at_k,
    hamiltonian,
    parallelogram_bz_grid,
    shift_kernel_at_k,
    symplectic_connection_imag_at_k,
    transition_gap_at_points,
)
from analysis.shift_current import add_transitions_to_integral, positive_transition_terms


def _finite_difference_first(k_xy: np.ndarray, axis: int, step: float = 1.0e-6) -> np.ndarray:
    delta = np.zeros(2, dtype=float)
    delta[axis] = step
    return (hamiltonian(k_xy + delta) - hamiltonian(k_xy - delta)) / (2.0 * step)


def _finite_difference_second(k_xy: np.ndarray, axis_a: int, axis_b: int, step: float = 2.0e-5) -> np.ndarray:
    da = np.zeros(2, dtype=float)
    db = np.zeros(2, dtype=float)
    da[axis_a] = step
    db[axis_b] = step
    return (
        hamiltonian(k_xy + da + db)
        - hamiltonian(k_xy + da - db)
        - hamiltonian(k_xy - da + db)
        + hamiltonian(k_xy - da - db)
    ) / (4.0 * step * step)


def test_haldane_analytic_derivatives_match_finite_difference():
    k_xy = np.asarray([0.37, -0.41], dtype=float)
    first = dhdk(k_xy)
    second = d2hdk(k_xy)
    for axis in range(2):
        np.testing.assert_allclose(first[axis], _finite_difference_first(k_xy, axis), rtol=1.0e-9, atol=1.0e-9)
    for axis_a in range(2):
        for axis_b in range(2):
            np.testing.assert_allclose(
                second[axis_a, axis_b],
                _finite_difference_second(k_xy, axis_a, axis_b),
                rtol=2.0e-5,
                atol=2.0e-6,
            )


def test_haldane_dirac_gaps_match_formula_up_to_k_label():
    params = HaldaneParams(M=0.4, t2=0.2, phi=-math.pi / 2.0)
    gaps = transition_gap_at_points(params)
    q = 3.0 * math.sqrt(3.0) * params.t2 * math.sin(params.phi)
    expected = sorted([2.0 * abs(params.M + q), 2.0 * abs(params.M - q)])
    actual = sorted([gaps["K"], gaps["Kprime"]])
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


def test_injection_kernel_is_u1_gauge_invariant_and_same_pol_real():
    k_xy = np.asarray([0.29, 0.17], dtype=float)
    evals, evecs = diagonalize(k_xy)
    tensors = precompute_injection_current_tensors(evals, evecs, dhdk(k_xy), denominator_cutoff_ev=1.0e-12)
    kernel = injection_component_kernel(tensors, 0, 1, "y;yy")
    assert abs(kernel.imag) < 1.0e-12

    phases = np.diag(np.exp(1.0j * np.asarray([0.37, -1.41])))
    tensors_gauge = precompute_injection_current_tensors(
        evals,
        evecs @ phases,
        dhdk(k_xy),
        denominator_cutoff_ev=1.0e-12,
    )
    kernel_gauge = injection_component_kernel(tensors_gauge, 0, 1, "y;yy")
    np.testing.assert_allclose(kernel_gauge, kernel, rtol=0.0, atol=1.0e-14)


def test_complex_response_helpers_split_linear_and_cpge_parts():
    response = np.asarray([1.0 + 2.0j, -3.0 + 0.25j])
    np.testing.assert_allclose(linear_injection_part(response), [1.0, -3.0])
    np.testing.assert_allclose(cpge_part(response), [2.0, 0.25])


def test_haldane_spectrum_smoke_is_finite():
    omega = np.linspace(0.1, 3.0, 9)
    spectra = compute_haldane_injection_spectra(omega, components=("y;yy",), mesh_size=5, eta=0.04)
    assert set(spectra) == {"y;yy"}
    assert spectra["y;yy"].shape == omega.shape
    assert np.all(np.isfinite(spectra["y;yy"]))


def test_haldane_injection_spectrum_uses_bz_weight_not_transition_weight():
    omega = np.linspace(0.1, 3.0, 7)
    eta = 0.04
    computed = compute_haldane_injection_spectra(
        omega,
        components=("y;yy",),
        mesh_size=3,
        eta=eta,
        c3_symmetrize_grid=False,
    )["y;yy"]
    manual = np.zeros_like(omega, dtype=np.complex128)
    k_points, k_weights = parallelogram_bz_grid(3)
    for k_xy, k_weight in zip(k_points, k_weights, strict=True):
        evals, evecs = diagonalize(k_xy)
        tensors = precompute_injection_current_tensors(evals, evecs, dhdk(k_xy))
        transitions, transition_weights = positive_injection_transition_terms(tensors, "y;yy")
        manual += accumulate_injection_spectrum(
            omega,
            transitions,
            transition_weights,
            k_weight=float(k_weight),
            eta_ev=eta,
        )
    np.testing.assert_allclose(computed, manual, rtol=0.0, atol=1.0e-14)


def test_symplectic_connection_uses_shared_generalized_derivative_and_is_finite():
    value = symplectic_connection_imag_at_k(np.asarray([0.29, 0.17]), current_axis=0, optical_axis=0)
    assert np.isfinite(value)


def test_haldane_fig78_shift_kernel_can_force_lower_band_filled():
    # Lin--Hsu Eq. (13)-(14) assumes the lower band is fully occupied.  At
    # phi=0 the scalar f0(k) term can place both bands above mu=0 at some k,
    # so a Fermi cutoff at mu=0 would incorrectly remove a valid 1 -> 2
    # transition from the zone-integrated Fig. 8 response.
    k_xy = np.asarray([2.0943951023931953, -1.451039491387374], dtype=float)
    params = HaldaneParams(phi=0.0, M=0.4)
    tensors = haldane_shift_tensors_at_k(k_xy, params)
    assert np.all(tensors.occupations == 1.0)

    lower_filled_tensors = replace(tensors, occupations=np.asarray([1.0, 0.0]))
    transitions, weights = positive_transition_terms(lower_filled_tensors, "x;xx")
    assert transitions.size == 1
    expected = float(np.real(-1.0j * weights[0]))
    np.testing.assert_allclose(
        shift_kernel_at_k(k_xy, "x;xx", params=params, lower_band_filled=True),
        expected,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert shift_kernel_at_k(k_xy, "x;xx", params=params) == 0.0


def test_haldane_shift_spectrum_uses_bz_weight_not_transition_weight():
    omega = np.linspace(0.1, 3.0, 7)
    eta = 0.04
    computed = compute_haldane_shift_spectra(
        omega,
        components=("x;xx",),
        mesh_size=3,
        eta=eta,
        c3_symmetrize_grid=False,
    )["x;xx"]
    manual_integral = np.zeros_like(omega, dtype=np.complex128)
    k_points, k_weights = parallelogram_bz_grid(3)
    for k_xy, k_weight in zip(k_points, k_weights, strict=True):
        tensors = haldane_shift_tensors_at_k(k_xy)
        transitions, transition_weights = positive_transition_terms(tensors, "x;xx")
        add_transitions_to_integral(
            manual_integral,
            omega,
            transitions,
            transition_weights,
            k_weight_nm_inv_sq=float(k_weight),
            eta_ev=eta,
        )
    manual = -1.0j * np.pi * manual_integral
    np.testing.assert_allclose(computed, manual, rtol=0.0, atol=1.0e-14)
