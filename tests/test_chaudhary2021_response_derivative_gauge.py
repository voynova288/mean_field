from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from analysis.response_derivative_gauge import (
    berry_connection_generalized_derivative_pair,
    berry_connection_pair,
    hamiltonian_gauge_data,
    link_shift_vector,
    shift_integrand_from_pair_generalized_derivative,
    shift_vector_from_pair_generalized_derivative,
)
from analysis.shift_current import JOYA_EQ7_GEOMETRIC_CONVENTION, component_kernel_from_gauge_pair, component_kernel_from_pair
from mean_field.systems.htg.shift_current import (
    berry_connection_pair_from_D,
    generalized_derivative_pair_from_D,
    velocity_matrices,
)
from mean_field.systems.tbg.chaudhary2021 import (
    ChaudharyTBGConfig,
    b0_component_kernel_at_k,
    b0_shift_current_point_data,
    b0_shift_current_tensors_at_k,
    build_chau_b0_hamiltonian,
    centered_flat_indices,
    finite_difference_b0_dhdk,
    make_b0_parameters,
)
from mean_field.systems.tbg.zero_field.model import _generate_gvec, _generate_t12_zero_fill


def _b0_point_data(k_dimless: complex, *, lg: int = 3):
    config = ChaudharyTBGConfig(theta_deg=0.8, n_shells=3, delta1_ev=0.005, delta2_ev=0.005)
    params = make_b0_parameters(config)
    gvec = _generate_gvec(params, int(lg))
    tunnel = _generate_t12_zero_fill(params, int(lg), int(config.valley))
    dhdk = finite_difference_b0_dhdk(params, config, lg=int(lg), step_dimless=1.0e-6)
    hmat = build_chau_b0_hamiltonian(complex(k_dimless), params, config, lg=int(lg), gvec=gvec, tunnel=tunnel)
    evals, evecs = eigh(hmat)
    gauge_data = hamiltonian_gauge_data(evals, evecs, np.stack(dhdk, axis=0), denominator_cutoff=1.0e-8)
    return config, params, gvec, tunnel, dhdk, evals, evecs, gauge_data


def test_chau_tbg_b0_point_derivative_matches_existing_pair_core():
    _config, _params, _gvec, _tunnel, dhdk, evals, evecs, gauge_data = _b0_point_data(0.03 + 0.02j)
    D = velocity_matrices(evecs, dhdk)
    np.testing.assert_allclose(gauge_data.velocity_h, D, rtol=0.0, atol=0.0)

    v_flat, c_flat = centered_flat_indices(evals.size)
    for n, m in ((v_flat, c_flat), (v_flat - 1, v_flat), (c_flat, c_flat + 1)):
        r_old = berry_connection_pair_from_D(D, evals, m, n, denominator_cutoff_ev=1.0e-8)
        r_new = berry_connection_pair(gauge_data.velocity_h, gauge_data.energies, m, n, denominator_cutoff=1.0e-8)
        np.testing.assert_allclose(r_new, r_old, rtol=0.0, atol=1.0e-14)

        gd_old = generalized_derivative_pair_from_D(D, evals, n, m, denominator_cutoff_ev=1.0e-8)
        gd_new = berry_connection_generalized_derivative_pair(
            gauge_data.velocity_h,
            gauge_data.energies,
            n,
            m,
            denominator_cutoff=1.0e-8,
        )
        np.testing.assert_allclose(gd_new.values, gd_old.values, rtol=0.0, atol=2.0e-13)
        assert gd_new.skipped_small_denominators == gd_old.skipped_small_denominators


def test_chau_tbg_b0_system_adapter_matches_manual_generic_api():
    config, params, _gvec, _tunnel, _dhdk, evals, _evecs, data = _b0_point_data(0.03 + 0.02j)
    point = b0_shift_current_point_data(0.03 + 0.02j, params, config, lg=3, denominator_cutoff_ev=1.0e-8)
    tensors = b0_shift_current_tensors_at_k(0.03 + 0.02j, params, config, lg=3, denominator_cutoff_ev=1.0e-8)
    np.testing.assert_allclose(point.energies_ev, evals, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(point.gauge_data.velocity_h, data.velocity_h, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(tensors.velocity_h, data.velocity_h, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(tensors.berry_connection, data.berry_connection, rtol=0.0, atol=1.0e-14)


def test_chau_tbg_b0_generic_shift_current_api_matches_existing_pair_core():
    _config, _params, _gvec, _tunnel, _dhdk, evals, _evecs, data = _b0_point_data(0.03 + 0.02j)
    v_flat, c_flat = centered_flat_indices(evals.size)
    pair_gd = berry_connection_generalized_derivative_pair(
        data.velocity_h,
        data.energies,
        v_flat,
        c_flat,
        denominator_cutoff=1.0e-8,
    )
    old = shift_integrand_from_pair_generalized_derivative(
        data.berry_connection,
        pair_gd.values,
        initial_band=v_flat,
        final_band=c_flat,
        deriv_axis=0,
        optical_axis=1,
    )
    from_pair = component_kernel_from_pair(
        data.berry_connection,
        pair_gd.values,
        initial_band=v_flat,
        final_band=c_flat,
        component=(0, 1, 1),
        convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
    )
    selected_pair = component_kernel_from_gauge_pair(
        data.velocity_h,
        data.energies,
        data.berry_connection,
        v_flat,
        c_flat,
        (0, 1, 1),
        denominator_cutoff_ev=1.0e-8,
        convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
    )
    adapter_pair = b0_component_kernel_at_k(
        0.03 + 0.02j,
        _params,
        _config,
        v_flat,
        c_flat,
        (0, 1, 1),
        lg=3,
        denominator_cutoff_ev=1.0e-8,
        convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
    )
    np.testing.assert_allclose(from_pair, old, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(selected_pair.kernel, old, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(adapter_pair.kernel, old, rtol=0.0, atol=1.0e-13)


def test_chau_tbg_b0_wilson_link_matches_covariant_shift_vector():
    k0 = 0.03 + 0.02j
    step_dimless = 3.0e-6
    config, params, gvec, tunnel, dhdk, evals, _evecs, data0 = _b0_point_data(k0)
    _config1, _params1, _gvec1, _tunnel1, _dhdk1, _evals1, _evecs1, data1 = _b0_point_data(k0 + 1.0j * step_dimless)
    v_flat, c_flat = centered_flat_indices(evals.size)
    pair_gd = berry_connection_generalized_derivative_pair(
        data0.velocity_h,
        data0.energies,
        v_flat,
        c_flat,
        denominator_cutoff=1.0e-8,
    )
    covariant_shift_nm = shift_vector_from_pair_generalized_derivative(
        data0.berry_connection,
        pair_gd.values,
        initial_band=v_flat,
        final_band=c_flat,
        deriv_axis=1,
        optical_axis=0,
    )
    link_shift_dimless = link_shift_vector(
        data0.eigenvectors,
        data1.eigenvectors,
        data0.berry_connection,
        data1.berry_connection,
        initial_band=v_flat,
        final_band=c_flat,
        optical_axis=0,
        step=step_dimless,
    )
    # The b0 Hamiltonian is parameterized by q=a*k_phys, so the link phase
    # derivative with respect to q is converted to nm by multiplying by a.
    link_shift_nm = link_shift_dimless * float(config.graphene_lattice_constant_nm)
    assert np.isfinite(covariant_shift_nm)
    assert abs(covariant_shift_nm - link_shift_nm) < 2.0e-3
