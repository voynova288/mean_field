from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from analysis.shift_current import (
    JOYA_EQ7_GEOMETRIC_CONVENTION,
    component_kernel_from_gauge_pair,
    component_kernel_from_pair,
    precompute_shift_current_tensors,
)
from mean_field.systems.tdbg.hamiltonian import build_hamiltonian, build_hamiltonian_d2hdk2, build_hamiltonian_dhdk
from mean_field.systems.tdbg.model import TDBGModel
from mean_field.systems.tdbg.params import TDBGParameters
from mean_field.systems.tdbg.shift_current import (
    JOYA_GAMMA_CENTERED_FRAC_SHIFT,
    component_kernel_at_k,
    finite_difference_dhdk,
    joya_gamma_centered_k_grid,
    mirror_x_tensor_component_sign,
    shift_current_point_data,
    shift_current_tensors_at_k,
    transform_valley_component_to_physical_axes,
    valley_mirror_x_tensor_component_sign,
)


def _model_and_k():
    params = TDBGParameters.full(stacking="AB-AB", valley=1, Delta=0.05)
    model = TDBGModel.from_config(0.8, cut=1, params=params)
    k0 = 0.173 * model.lattice.g_m1 + 0.271 * model.lattice.g_m2
    return model, complex(k0)


def _direct_eq7_kernel(velocity_h: np.ndarray, energies: np.ndarray, n: int, m: int, mu: int, alpha: int, *, cutoff: float, eta: float | None = None) -> float:
    v = np.asarray(velocity_h, dtype=np.complex128)
    e = np.asarray(energies, dtype=float)
    eba = float(e[m] - e[n])
    assert eba > cutoff
    total = 0.0j
    for c in range(e.size):
        if c != n:
            den = float(e[n] - e[c])
            if abs(den) > cutoff:
                inv = 1.0 / den if eta is None else den / (den * den + eta * eta)
                total += v[mu, n, c] * v[alpha, c, m] * v[alpha, m, n] * inv
        if c != m:
            den = float(e[m] - e[c])
            if abs(den) > cutoff:
                inv = 1.0 / den if eta is None else den / (den * den + eta * eta)
                total += v[mu, c, m] * v[alpha, m, n] * v[alpha, n, c] * inv
    return float(np.imag(total) / (eba * eba))


def test_tdbg_joya_system_convention_helpers():
    model, _k0 = _model_and_k()
    frac, kpts = joya_gamma_centered_k_grid(model.lattice, 2)
    np.testing.assert_allclose(np.asarray(JOYA_GAMMA_CENTERED_FRAC_SHIFT), np.asarray((-1.0 / 6.0, -1.0 / 6.0)))
    np.testing.assert_allclose(frac[0, 0], np.asarray((-1.0 / 6.0, -1.0 / 6.0)))
    np.testing.assert_allclose(kpts[0, 0], -model.lattice.g_m1 / 6.0 - model.lattice.g_m2 / 6.0)
    assert mirror_x_tensor_component_sign("xxx") == -1
    assert mirror_x_tensor_component_sign("xyy") == -1
    assert mirror_x_tensor_component_sign("yxx") == 1
    assert mirror_x_tensor_component_sign("yyy") == 1
    assert valley_mirror_x_tensor_component_sign("xyy", valley=-1) == -1
    assert valley_mirror_x_tensor_component_sign("xyy", valley=1) == 1
    assert transform_valley_component_to_physical_axes(3.0, "xyy", valley=-1) == -3.0


def test_tdbg_analytic_dhdk_matches_finite_difference_for_joya_tiny_models():
    for stacking in ("AB-AB", "AB-BA"):
        for valley in (1, -1):
            params = TDBGParameters.full(stacking=stacking, valley=valley, Delta=0.05)
            model = TDBGModel.from_config(0.8, cut=1, params=params)
            k0 = complex(0.173 * model.lattice.g_m1 + 0.271 * model.lattice.g_m2)
            analytic = build_hamiltonian_dhdk(model.lattice, model.params, valley=valley)
            numeric = finite_difference_dhdk(k0, model.lattice, model.params, valley=valley, step_nm_inv=1.0e-6)
            np.testing.assert_allclose(analytic, np.swapaxes(analytic.conjugate(), -1, -2), rtol=0.0, atol=1.0e-14)
            np.testing.assert_allclose(analytic, numeric, rtol=0.0, atol=2.0e-9)
            d2 = build_hamiltonian_d2hdk2(model.lattice)
            assert d2.shape == (2, 2, model.lattice.matrix_dim, model.lattice.matrix_dim)
            np.testing.assert_allclose(d2, 0.0, rtol=0.0, atol=0.0)


def test_tdbg_system_adapter_matches_manual_generic_api():
    model, k0 = _model_and_k()
    point = shift_current_point_data(k0, model.lattice, model.params, denominator_cutoff_ev=1.0e-8)
    h0 = build_hamiltonian(k0, model.lattice, model.params, valley=1)
    evals, evecs = eigh(h0)
    dh = build_hamiltonian_dhdk(model.lattice, model.params, valley=1)
    d2h = build_hamiltonian_d2hdk2(model.lattice)
    manual = precompute_shift_current_tensors(evals, evecs, dh, denominator_cutoff_ev=1.0e-8, d2hdk=d2h)
    adapter = shift_current_tensors_at_k(k0, model.lattice, model.params, denominator_cutoff_ev=1.0e-8)
    np.testing.assert_allclose(point.energies_ev, evals, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(point.dhdk, dh, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(point.d2hdk, d2h, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(adapter.velocity_h, manual.velocity_h, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(adapter.berry_connection, manual.berry_connection, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(adapter.berry_connection_gen_derivative, manual.berry_connection_gen_derivative, rtol=0.0, atol=1.0e-12)


def test_tdbg_system_adapter_reproduces_joya_ordered_eq7_kernel_at_tiny_cut():
    component_axes = {"xxx": (0, 0), "yyy": (1, 1)}
    for stacking in ("AB-AB", "AB-BA"):
        for valley in (1, -1):
            params = TDBGParameters.full(stacking=stacking, valley=valley, Delta=0.05)
            model = TDBGModel.from_config(0.8, cut=1, params=params)
            k0 = complex(0.173 * model.lattice.g_m1 + 0.271 * model.lattice.g_m2)
            point = shift_current_point_data(k0, model.lattice, model.params, valley=valley, denominator_cutoff_ev=1.0e-8)
            center = point.energies_ev.size // 2
            n = center - 1
            m = center
            for comp, (mu_axis, optical_axis) in component_axes.items():
                direct = _direct_eq7_kernel(
                    point.gauge_data.velocity_h,
                    point.gauge_data.energies,
                    n,
                    m,
                    mu_axis,
                    optical_axis,
                    cutoff=1.0e-8,
                    eta=1.0e-3,
                )
                selected = component_kernel_at_k(
                    k0,
                    model.lattice,
                    model.params,
                    n,
                    m,
                    comp,
                    valley=valley,
                    denominator_cutoff_ev=1.0e-8,
                    principal_value_eta_ev=1.0e-3,
                    convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
                )
                manual_selected = component_kernel_from_gauge_pair(
                    point.gauge_data.velocity_h,
                    point.gauge_data.energies,
                    point.gauge_data.berry_connection,
                    n,
                    m,
                    comp,
                    denominator_cutoff_ev=1.0e-8,
                    principal_value_eta_ev=1.0e-3,
                    convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
                )
                np.testing.assert_allclose(selected.kernel, manual_selected.kernel, rtol=0.0, atol=1.0e-13)
                np.testing.assert_allclose(selected.kernel, direct, rtol=1.0e-11, atol=1.0e-11)


def test_tdbg_joya_selected_pair_is_phase_gauge_invariant_and_matches_full_tensor():
    model, k0 = _model_and_k()
    point = shift_current_point_data(k0, model.lattice, model.params, denominator_cutoff_ev=1.0e-8)
    tensors = shift_current_tensors_at_k(k0, model.lattice, model.params, denominator_cutoff_ev=1.0e-8)
    center = point.energies_ev.size // 2
    n = center - 1
    m = center
    pair = component_kernel_from_gauge_pair(
        point.gauge_data.velocity_h,
        point.gauge_data.energies,
        point.gauge_data.berry_connection,
        n,
        m,
        "yyy",
        denominator_cutoff_ev=1.0e-8,
        convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
    )
    full = component_kernel_from_pair(
        tensors.berry_connection,
        tensors.berry_connection_gen_derivative[:, :, n, m],
        initial_band=n,
        final_band=m,
        component="yyy",
        convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
    )
    np.testing.assert_allclose(pair.kernel, full, rtol=0.0, atol=1.0e-13)

    rng = np.random.default_rng(20260609)
    phase = np.exp(1.0j * rng.uniform(-np.pi, np.pi, size=point.energies_ev.size))
    velocity_gauge = phase.conjugate()[None, :, None] * point.gauge_data.velocity_h * phase[None, None, :]
    berry_gauge = phase.conjugate()[None, :, None] * point.gauge_data.berry_connection * phase[None, None, :]
    pair_gauge = component_kernel_from_gauge_pair(
        velocity_gauge,
        point.gauge_data.energies,
        berry_gauge,
        n,
        m,
        "yyy",
        denominator_cutoff_ev=1.0e-8,
        convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
    )
    np.testing.assert_allclose(pair_gauge.kernel, pair.kernel, rtol=0.0, atol=1.0e-12)
