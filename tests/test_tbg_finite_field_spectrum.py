from __future__ import annotations

import numpy as np

from mean_field.systems.tbg.finite_field import (
    FiniteFieldBMParameters,
    MagneticFlux,
    associated_laguerre_element,
    associated_laguerre_matrix,
    author_landau_cutoff,
    compute_coulomb_overlap,
    compute_coulomb_overlap_fast,
    compute_magnetic_spectrum,
    compute_magnetic_spectrum_sweep,
    construct_ll_hamiltonian,
    paper_hofstadter_fluxes,
    red_chern_minus_one_group_mask,
)
from mean_field.systems.tbg.finite_field.spectrum import _hermitian_from_upper


def test_paper_hofstadter_flux_rules_and_red_group_mask() -> None:
    fluxes = paper_hofstadter_fluxes(max_denominator=12, phi_max=0.5)
    assert len(fluxes) == 23
    assert fluxes[0] == MagneticFlux(1, 12)
    assert fluxes[-1] == MagneticFlux(1, 2)
    assert author_landau_cutoff(MagneticFlux(2, 5)) == 62

    mask = red_chern_minus_one_group_mask(MagneticFlux(2, 5))
    assert mask.shape == (10,)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.array([3, 4]))

def test_magnetic_spectrum_sweep_matches_single_flux_result_and_builds_plot_table() -> None:
    params = FiniteFieldBMParameters.from_degrees(1.20, w0=0.0, w1=0.0, strain=0.0, deformation_potential=0.0)
    flux = MagneticFlux(1, 2)
    sweep = compute_magnetic_spectrum_sweep(
        params,
        fluxes=(flux,),
        n_landau_by_flux=lambda _flux: 4,
        nq_by_flux=lambda _flux: 1,
        include_strain=False,
    )
    single = compute_magnetic_spectrum(params, flux=flux, n_landau=4, nq=1, include_strain=False)

    assert len(sweep.cases) == 1
    assert sweep.cases[0].n_landau == 4
    np.testing.assert_allclose(sweep.spectra[0].spectrum, single.spectrum, atol=1.0e-12)
    table = sweep.as_point_table()
    assert table["energy_mev"].shape == (4,)
    assert table["red_group"].sum() == 1
    np.testing.assert_allclose(table["phi"], np.full(4, 0.5))

def test_associated_laguerre_zero_momentum_is_identity() -> None:
    mat = associated_laguerre_matrix(6, 0.0 + 0.0j, 3.0)
    np.testing.assert_allclose(mat, np.eye(6), atol=1.0e-14)


def test_associated_laguerre_matrix_matches_element_formula() -> None:
    n_landau = 7
    qvec = 0.137 - 0.421j
    l_b = 4.3
    cplus = -1j * l_b / np.sqrt(2.0) * (qvec.real - 1j * qvec.imag)
    cminus = -1j * l_b / np.sqrt(2.0) * (qvec.real + 1j * qvec.imag)
    expected = np.zeros((n_landau, n_landau), dtype=np.complex128)
    for n in range(n_landau):
        for m in range(n_landau):
            expected[n, m] = associated_laguerre_element(n, m, cplus, cminus)

    mat = associated_laguerre_matrix(n_landau, qvec, l_b)
    np.testing.assert_allclose(mat, expected, atol=1.0e-14, rtol=1.0e-14)


def test_zero_tunneling_spectrum_has_two_central_zero_landau_levels() -> None:
    params = FiniteFieldBMParameters.from_degrees(1.05, w0=0.0, w1=0.0, strain=0.0, deformation_potential=0.0)
    result = compute_magnetic_spectrum(params, flux=MagneticFlux(1, 1), n_landau=4, nq=1, include_strain=False)

    assert result.spectrum.shape == (2, 1, 1)
    np.testing.assert_allclose(result.spectrum[:, 0, 0], np.zeros(2), atol=1.0e-12)
    overlap = compute_coulomb_overlap(result, 0, 0)
    np.testing.assert_allclose(overlap, np.eye(2), atol=1.0e-12)


def test_constructed_ll_hamiltonian_is_hermitian_after_author_upper_triangle_completion() -> None:
    params = FiniteFieldBMParameters.from_degrees(1.20, w0=77.0, w1=110.0, strain=0.0, deformation_potential=0.0)
    h, _sigma, *_ = construct_ll_hamiltonian(
        params,
        flux=MagneticFlux(1, 2),
        n_landau=4,
        nq=1,
        include_strain=False,
    )
    block = h[..., 0, 0].reshape((h.shape[0] * h.shape[1] * h.shape[2], h.shape[3] * h.shape[4] * h.shape[5]), order="F")
    hermitian = _hermitian_from_upper(block)
    np.testing.assert_allclose(hermitian, hermitian.conj().T, atol=1.0e-12)
    np.testing.assert_allclose(np.diag(hermitian), np.diag(block), atol=1.0e-12)


def test_magnetic_spectrum_shapes_projected_sigma_and_orbit_vectors() -> None:
    params = FiniteFieldBMParameters.from_degrees(1.20, w0=70.0, w1=110.0, strain=0.0, deformation_potential=0.0)
    result = compute_magnetic_spectrum(params, flux=MagneticFlux(1, 2), n_landau=5, nq=1, include_strain=False)

    assert result.spectrum.shape == (4, 1, 1)
    assert result.vec.shape == (2 * result.n_h * result.p, 4, 2, 1, 1)
    assert result.p_sigma_z.shape == (4, 4, 1, 1)
    np.testing.assert_allclose(result.p_sigma_z[:, :, 0, 0], result.p_sigma_z[:, :, 0, 0].conj().T, atol=1.0e-12)
    np.testing.assert_allclose(result.vec[:, :, 0, 0, 0].conj().T @ result.vec[:, :, 0, 0, 0], np.eye(4), atol=1.0e-12)
    assert np.all(np.diff(result.spectrum[:, 0, 0]) >= -1.0e-12)


def test_fast_coulomb_overlap_matches_direct_author_loop() -> None:
    params = FiniteFieldBMParameters.from_degrees(1.10, w0=20.0, w1=50.0, strain=0.0, deformation_potential=0.0)
    result = compute_magnetic_spectrum(params, flux=MagneticFlux(1, 2), n_landau=4, nq=1, include_strain=False)

    direct = compute_coulomb_overlap(result, 1, -1)
    fast = compute_coulomb_overlap_fast(result, 1, -1)
    np.testing.assert_allclose(fast, direct, atol=1.0e-10)


def test_kprime_zero_tunneling_matches_k_by_time_reversal_construction() -> None:
    params = FiniteFieldBMParameters.from_degrees(1.05, w0=0.0, w1=0.0, strain=0.0, deformation_potential=0.0)
    k = compute_magnetic_spectrum(params, flux=MagneticFlux(1, 1), n_landau=4, nq=1, valley="K", include_strain=False)
    kp = compute_magnetic_spectrum(params, flux=MagneticFlux(1, 1), n_landau=4, nq=1, valley="Kprime", include_strain=False)

    np.testing.assert_allclose(k.spectrum, kp.spectrum, atol=1.0e-12)
