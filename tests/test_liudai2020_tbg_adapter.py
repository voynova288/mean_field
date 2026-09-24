from __future__ import annotations

import numpy as np

from analysis.optical import kerr_faraday_from_kpoint_data_liu_dai_2020_printed
from mean_field.systems.tbg.liudai2020 import (
    GRAPHENE_A_NM,
    LiuDai2020TBGParameters,
    liu_dai2020_tbg_eigensystem,
    liu_dai2020_tbg_optical_kpoint_data,
    liu_dai2020_tbg_optical_payloads_from_valley_data,
    liu_dai2020_tbg_valley_kpoint_data,
    parallelogram_kmesh,
    plus_three_quarter_fermi_level,
    valley_spin_energy_shift_ev,
)
from mean_field.systems.tbg.arora2021 import circular_reciprocal_indices, gvec_from_indices


def test_liu_dai_2020_parameter_bundle_matches_methods_values():
    paper = LiuDai2020TBGParameters()
    assert paper.theta_deg == 1.05
    assert paper.delta_bottom_mev == 17.0
    assert paper.sigma_rotation is False
    np.testing.assert_allclose(paper.kinetic_ev, paper.hbar_vf_ev_nm / GRAPHENE_A_NM)
    cfg = paper.config(valley=1)
    np.testing.assert_allclose(cfg.delta1_ev, 0.017)
    np.testing.assert_allclose(cfg.delta2_ev, 0.0)


def test_valley_spin_energy_shift_eq1():
    np.testing.assert_allclose(
        valley_spin_energy_shift_ev(1, -1, valley_split_mev=3.0, spin_split_mev=2.0),
        1.0e-3,
    )
    np.testing.assert_allclose(
        valley_spin_energy_shift_ev(-1, 1, valley_split_mev=3.0, spin_split_mev=2.0),
        -1.0e-3,
    )


def test_parallelogram_kmesh_weights_sum_to_bz_area_nm2():
    paper = LiuDai2020TBGParameters()
    params = paper.b0_params()
    kpoints, weights = parallelogram_kmesh(params, 3)
    assert kpoints.shape == (9,)
    area_dimless = abs(float(np.imag(np.conjugate(params.g1) * params.g2)))
    expected_area_nm_inv_sq = area_dimless / (GRAPHENE_A_NM * GRAPHENE_A_NM)
    np.testing.assert_allclose(np.sum(weights), expected_area_nm_inv_sq, rtol=1.0e-14, atol=0.0)


def test_small_shell_eigensystem_and_velocity_payload_are_finite():
    paper = LiuDai2020TBGParameters(cutoff_shell=1)
    params = paper.b0_params()
    config = paper.config(valley=1)
    indices = circular_reciprocal_indices(1)
    gvec = gvec_from_indices(params, indices)
    energies, evecs, dhdk = liu_dai2020_tbg_eigensystem(0.0j, params, config, cutoff_shell=1, indices=indices, gvec=gvec)
    assert energies.shape == (4 * len(indices),)
    assert evecs.shape == (4 * len(indices), 4 * len(indices))
    assert dhdk.shape == (2, 4 * len(indices), 4 * len(indices))
    assert np.all(np.isfinite(energies))


def test_plus_three_quarter_fermi_level_has_expected_occupation_count():
    # Four sectors with four bands each: neutrality is 8 occupied bands per k,
    # +3/4 filling adds 3 occupied conduction-flat states per k.
    sectors = [np.asarray([-2.0, -1.0, 1.0 + 0.1 * i, 2.0 + 0.1 * i]) for i in range(4)]
    mu = plus_three_quarter_fermi_level(sectors)
    occupied = sum(int(np.sum(sector < mu)) for sector in sectors)
    assert occupied == 11


def test_cached_valley_data_reuses_eigensystems_for_split_payloads():
    paper = LiuDai2020TBGParameters(cutoff_shell=1)
    valley_data = liu_dai2020_tbg_valley_kpoint_data(mesh_size=1, paper=paper, cutoff_shell=1)
    assert len(valley_data) == 2
    payloads_cached, mu_cached = liu_dai2020_tbg_optical_payloads_from_valley_data(
        valley_data,
        valley_split_mev=3.0,
        spin_split_mev=0.0,
    )
    payloads_direct, mu_direct = liu_dai2020_tbg_optical_kpoint_data(
        mesh_size=1,
        paper=paper,
        valley_split_mev=3.0,
        spin_split_mev=0.0,
        cutoff_shell=1,
    )
    np.testing.assert_allclose(mu_cached, mu_direct)
    assert len(payloads_cached) == len(payloads_direct) == 4
    for cached, direct in zip(payloads_cached, payloads_direct, strict=True):
        np.testing.assert_allclose(cached.energies_ev, direct.energies_ev)
        np.testing.assert_allclose(cached.velocity_h, direct.velocity_h)
        np.testing.assert_allclose(cached.occupations, direct.occupations)
        np.testing.assert_allclose(cached.weight, direct.weight)


def test_small_mesh_optical_payloads_feed_linear_kerr_route():
    payloads, mu = liu_dai2020_tbg_optical_kpoint_data(
        mesh_size=1,
        paper=LiuDai2020TBGParameters(cutoff_shell=1),
        valley_split_mev=3.0,
        spin_split_mev=0.0,
        cutoff_shell=1,
    )
    assert len(payloads) == 4
    assert np.isfinite(mu)
    for payload in payloads:
        assert payload.velocity_h.shape[1:] == (payload.energies_ev.size, payload.energies_ev.size)
        np.testing.assert_allclose(payload.velocity_h[0], payload.velocity_h[0].conj().T, atol=1.0e-12)
        assert payload.weight > 0.0
    omega = np.asarray([0.03, 0.05], dtype=float)
    response = kerr_faraday_from_kpoint_data_liu_dai_2020_printed(
        omega,
        payloads,
        eta_ev=0.003,
        prefactor=1.0e-8j,
        include_bz_factor=True,
        selected_bands=range(payloads[0].energies_ev.size // 2 - 3, payloads[0].energies_ev.size // 2 + 3),
    )
    assert response.conductivity.shape == (omega.size, 2, 2)
    assert np.all(np.isfinite(response.conductivity))
    assert np.all(np.isfinite(response.faraday_angle_rad))
    assert np.all(np.isfinite(response.kerr_angle_rad))
