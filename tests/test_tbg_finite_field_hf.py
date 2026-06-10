from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf import DensityUpdateResult, HartreeFockKernel, run_hartree_fock_problem
import mean_field.systems.tbg.finite_field.hf as tbg_finite_field_hf
from mean_field.systems.tbg.finite_field import (
    FiniteFieldBMParameters,
    FiniteFieldHartreeFockState,
    FiniteFieldTLSymmetricHartreeFockInputs,
    MagneticFlux,
    MagneticOverlapData,
    apply_iks_phase_to_transposed_density,
    build_finite_field_hf_inputs_from_parameters,
    build_finite_field_hf_inputs_from_spectra,
    build_finite_field_hf_kernel,
    build_finite_field_hf_problem,
    build_finite_field_hf_kernel_from_inputs,
    build_finite_field_hf_state_from_spectra,
    build_full_flavor_overlap_data_from_spectra,
    build_h0_from_hofstadter_metadata,
    build_magnetic_interaction_hamiltonian,
    build_tl_symmetric_finite_field_hf_inputs_from_parameters,
    build_tl_symmetric_finite_field_hf_inputs_from_spectra,
    build_tl_symmetric_magnetic_interaction_hamiltonian,
    choose_magnetic_nq,
    compute_magnetic_spectrum,
    density_update_from_hamiltonian,
    expand_valley_overlap_data_to_flavors,
    finite_field_diophantine_filling,
    finite_field_filling,
    magnetic_k_vectors,
    magnetic_normalization_count,
    magnetic_orbit_indices,
    magnetic_r_orbit_positions,
    magnetic_shell_shifts,
    paper_fig6_branch_cases,
    paper_fig6_finite_b_fluxes,
    run_finite_field_hartree_fock,
    run_finite_field_hartree_fock_from_inputs,
    screened_coulomb_finite_b,
    state_index,
    summarize_finite_field_hartree_fock,
)


def _random_complex(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_tbg_finite_field_hf_is_thin_core_adapter() -> None:
    assert build_finite_field_hf_inputs_from_parameters.__module__ == "mean_field.systems.tbg.finite_field.hf"
    assert build_magnetic_interaction_hamiltonian.__module__ == "mean_field.core.hf.finite_field"
    assert build_tl_symmetric_magnetic_interaction_hamiltonian.__module__ == "mean_field.core.hf.finite_field"
    assert density_update_from_hamiltonian.__module__ == "mean_field.core.hf.finite_field"
    assert run_finite_field_hartree_fock_from_inputs.__module__ == "mean_field.core.hf.finite_field"
    tbg_owned = {
        "build_finite_field_hf_inputs_from_parameters",
        "build_finite_field_hf_inputs_from_spectra",
        "build_finite_field_hf_state_from_spectra",
        "build_full_flavor_overlap_data_from_spectra",
        "build_tl_symmetric_finite_field_hf_inputs_from_parameters",
        "build_tl_symmetric_finite_field_hf_inputs_from_spectra",
        "paper_fig6_branch_cases",
        "paper_fig6_finite_b_fluxes",
    }
    assert set(name for name in tbg_finite_field_hf.__all__ if getattr(getattr(tbg_finite_field_hf, name), "__module__", None) == tbg_finite_field_hf.__name__) == tbg_owned


def test_magnetic_mesh_matches_author_code_ordering() -> None:
    flux = MagneticFlux(2, 5)
    assert flux.p == 2
    assert flux.q == 5
    assert choose_magnetic_nq(7) == 2
    assert choose_magnetic_nq(12) == 1

    full = magnetic_k_vectors(g1=1.0 + 0.0j, g2=1.0j, flux=MagneticFlux(1, 3), nq=2)
    assert full.shape == (12,)
    np.testing.assert_allclose(full[:3], np.array([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=np.complex128))
    assert magnetic_normalization_count(MagneticFlux(1, 3), 2) == 36

    indices = magnetic_orbit_indices(q=3, nq=2)
    np.testing.assert_array_equal(indices[:, 0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(indices[:, 1], np.array([3, 4, 5]))
    np.testing.assert_array_equal(magnetic_r_orbit_positions(2, 5), np.array([0, 2, 4, 1, 3]))

    shifts = magnetic_shell_shifts(g1=1.0 + 0.0j, g2=np.exp(1j * np.pi / 3.0), q=2, shell_ng=1)
    assert (0, 0) in shifts
    assert shifts[0][0] == -1
    assert all(isinstance(m, int) and isinstance(n, int) for m, n in shifts)


def test_paper_fig6_branch_helpers_match_author_selected_replay_grid() -> None:
    fluxes = paper_fig6_finite_b_fluxes()
    assert [(flux.p, flux.q) for flux in fluxes] == [
        (1, 2),
        (2, 5),
        (1, 3),
        (2, 7),
        (1, 4),
        (2, 9),
        (1, 5),
        (2, 11),
        (1, 6),
        (1, 8),
        (1, 12),
    ]
    assert finite_field_diophantine_filling(-1, -3, MagneticFlux(1, 12)) == pytest.approx(-1.25)
    assert finite_field_diophantine_filling(-2, -2, (1, 12)) == pytest.approx(-13.0 / 6.0)
    assert finite_field_diophantine_filling(-3, -1, "1/12") == pytest.approx(-37.0 / 12.0)

    branch = paper_fig6_branch_cases(-2, -2)
    assert len(branch) == 11
    assert branch[0][0] == MagneticFlux(1, 2)
    assert branch[0][1] == pytest.approx(-3.0)
    assert branch[-1][0] == MagneticFlux(1, 12)
    assert branch[-1][1] == pytest.approx(-13.0 / 6.0)


def test_build_h0_from_hofstadter_metadata_repeats_strips_and_zeeman() -> None:
    flux = MagneticFlux(1, 2)
    nq = 1
    n_sub = 2 * flux.q
    valley_energies = [np.arange(n_sub, dtype=float).reshape(n_sub, 1, 1), 100.0 + np.arange(n_sub, dtype=float).reshape(n_sub, 1, 1)]
    valley_sigma = [np.eye(n_sub, dtype=np.complex128).reshape(n_sub, n_sub, 1, 1), 2.0 * np.eye(n_sub, dtype=np.complex128).reshape(n_sub, n_sub, 1, 1)]

    h0, sigma = build_h0_from_hofstadter_metadata(
        valley_energies,
        valley_sigma,
        flux=flux,
        nq=nq,
        zeeman_unit=10.0,
        reduced_translation=False,
    )

    assert h0.shape == (16, 16, 2)
    k_up = state_index(0, 0, 0, subbands_per_flavor=n_sub)
    k_down = state_index(0, 0, 1, subbands_per_flavor=n_sub)
    kp_up = state_index(0, 1, 0, subbands_per_flavor=n_sub)
    assert h0[k_up, k_up, 0].real == pytest.approx(5.0)
    assert h0[k_down, k_down, 1].real == pytest.approx(-5.0)
    assert h0[kp_up, kp_up, 0].real == pytest.approx(105.0)
    assert sigma[k_up, k_up, 0].real == pytest.approx(1.0)
    assert sigma[kp_up, kp_up, 0].real == pytest.approx(-2.0)


def test_density_update_uses_stored_projector_convention_and_filling() -> None:
    h = np.zeros((4, 4, 2), dtype=np.complex128)
    h[:, :, 0] = np.diag([0.0, 1.0, 2.0, 3.0])
    h[:, :, 1] = np.diag([4.0, 5.0, 6.0, 7.0])

    update = density_update_from_hamiltonian(h, nu=0.0)
    assert update.energies.shape == (4, 2)
    assert update.mu == pytest.approx(3.5)
    assert finite_field_filling(update.density) == pytest.approx(0.0)
    np.testing.assert_allclose(np.diag(update.density[:, :, 0]).real, np.array([0.5, 0.5, 0.5, 0.5]))
    np.testing.assert_allclose(np.diag(update.density[:, :, 1]).real, np.array([-0.5, -0.5, -0.5, -0.5]))


def test_finite_field_problem_preserves_author_mixed_density_convergence_rule() -> None:
    h0 = np.zeros((2, 2, 1), dtype=np.complex128)
    state = FiniteFieldHartreeFockState.from_h0(
        h0,
        nu=0.0,
        flux=MagneticFlux(1, 1),
        nq=1,
        v0=0.0,
        precision=1.0e-8,
    )
    target_density = np.zeros_like(state.density)
    target_density[:, :, 0] = np.diag([0.5, -0.5])

    kernel = HartreeFockKernel(
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=lambda hamiltonian: DensityUpdateResult(
            density=target_density,
            energies=np.asarray([[-1.0], [1.0]], dtype=float),
            mu=0.0,
        ),
        energy_functional=lambda interaction_h, h0_in, density: 0.0,
        oda_parameterizer=lambda state_obj, delta_density: 0.0,
        convergence_rule="mixed",
    )
    problem = build_finite_field_hf_problem(kernel, initializer=lambda state_obj, init_mode, seed: None)

    run = run_hartree_fock_problem(state, problem, init_mode="bm", seed=0, max_iter=3)

    assert run.converged
    assert run.exit_reason == "converged"
    assert run.iterations == 1
    assert np.isclose(run.iter_err[0], 0.0)
    assert np.isclose(run.iter_oda[0], 0.0)
    np.testing.assert_allclose(state.density, np.zeros_like(state.density))
    assert np.isclose(state.diagnostics["final_raw_norm"], 1.0)


def test_expand_valley_overlap_data_to_full_spin_valley_basis() -> None:
    q = 2
    n_sub = 2 * q
    nk = 3
    block_k = np.ones((n_sub, nk, n_sub, nk), dtype=np.complex128)
    block_kp = 2.0 * np.ones((n_sub, nk, n_sub, nk), dtype=np.complex128)
    data_k = MagneticOverlapData(shifts=((0, 0),), gvecs=np.array([0.0 + 0.0j]), overlaps={(0, 0): block_k})
    data_kp = MagneticOverlapData(shifts=((0, 0),), gvecs=np.array([0.0 + 0.0j]), overlaps={(0, 0): block_kp})

    full = expand_valley_overlap_data_to_flavors(data_k, data_kp, q=q)
    block = full.overlaps[(0, 0)]
    assert block.shape == (4 * n_sub, nk, 4 * n_sub, nk)
    k_up = state_index(0, 0, 0, subbands_per_flavor=n_sub)
    kp_up = state_index(0, 1, 0, subbands_per_flavor=n_sub)
    assert block[k_up, 0, k_up, 0] == pytest.approx(1.0)
    assert block[kp_up, 0, kp_up, 0] == pytest.approx(2.0)
    assert block[k_up, 0, kp_up, 0] == pytest.approx(0.0)




def _tiny_zero_tunneling_spectrum_pair():
    params = FiniteFieldBMParameters.from_degrees(1.20, w0=0.0, w1=0.0, strain=0.0, deformation_potential=0.0)
    flux = MagneticFlux(1, 1)
    common = dict(params=params, flux=flux, n_landau=3, nq=1, include_strain=False)
    return (
        compute_magnetic_spectrum(**common, valley="K"),
        compute_magnetic_spectrum(**common, valley="Kprime"),
    )

def _tiny_q2_spectrum_pair():
    params = FiniteFieldBMParameters.from_degrees(1.20, w0=10.0, w1=20.0, strain=0.0, deformation_potential=0.0)
    flux = MagneticFlux(1, 2)
    common = dict(params=params, flux=flux, n_landau=4, nq=1, include_strain=False)
    return (
        compute_magnetic_spectrum(**common, valley="K"),
        compute_magnetic_spectrum(**common, valley="Kprime"),
    )

def test_build_finite_field_hf_state_from_spectra_matches_metadata_adapter() -> None:
    k_result, kp_result = _tiny_zero_tunneling_spectrum_pair()
    state = build_finite_field_hf_state_from_spectra(k_result, kp_result, nu=0.0, v0=0.0, zeeman_unit=0.2)
    manual_h0, manual_sigma = build_h0_from_hofstadter_metadata(
    [k_result.spectrum, kp_result.spectrum],
    [k_result.p_sigma_z, kp_result.p_sigma_z],
    flux=MagneticFlux(1, 1),
    nq=1,
    zeeman_unit=0.2,
    )

    assert state.nt == 8
    assert state.nk == 1
    np.testing.assert_allclose(state.h0, manual_h0, atol=1.0e-12)
    np.testing.assert_allclose(state.sigma_z, manual_sigma, atol=1.0e-12)
    with pytest.raises(ValueError, match="First spectrum"):
        build_finite_field_hf_state_from_spectra(kp_result, k_result, nu=0.0, v0=0.0)

def test_build_full_flavor_overlap_and_hf_inputs_from_spectra_smoke() -> None:
    k_result, kp_result = _tiny_zero_tunneling_spectrum_pair()
    shifts = ((0, 0),)
    full_overlap = build_full_flavor_overlap_data_from_spectra(k_result, kp_result, shifts=shifts)
    block = full_overlap.overlaps[(0, 0)]
    assert block.shape == (8, 1, 8, 1)
    assert np.linalg.norm(block[:, 0, :, 0] - np.diag(np.diag(block[:, 0, :, 0]))) < 1.0e-12

    inputs = build_finite_field_hf_inputs_from_spectra(k_result, kp_result, nu=0.0, v0=0.0, shifts=shifts)
    assert inputs.state.nt == 8
    assert inputs.state.nk == 1
    assert inputs.k_vectors.shape == (1,)
    assert inputs.normalization_count == 1
    interaction = build_magnetic_interaction_hamiltonian(
    inputs.state.density,
    inputs.overlap_data,
    k_vectors=inputs.k_vectors,
    v0=inputs.state.v0,
    normalization_count=inputs.normalization_count,
    screening_lm=1.0,
    )
    np.testing.assert_allclose(interaction, np.zeros_like(interaction), atol=1.0e-12)

    with pytest.raises(ValueError, match="shifts or shell_ng"):
        build_finite_field_hf_inputs_from_spectra(k_result, kp_result, nu=0.0, v0=0.0)

def test_tiny_interacting_hf_smoke_uses_assembled_inputs_and_core_loop() -> None:
    k_result, kp_result = _tiny_zero_tunneling_spectrum_pair()
    inputs = build_finite_field_hf_inputs_from_spectra(
        k_result,
        kp_result,
        nu=0.0,
        v0=0.05,
        shifts=((1, 0),),
    )
    assert np.linalg.norm(inputs.overlap_data.overlaps[(1, 0)]) > 0.0

    kernel = build_finite_field_hf_kernel(
        inputs.state,
        inputs.overlap_data,
        k_vectors=inputs.k_vectors,
        normalization_count=inputs.normalization_count,
        screening_lm=1.0,
    )
    run = run_finite_field_hartree_fock(inputs.state, kernel, init_mode="bm", seed=0, max_iter=3)

    assert 1 <= run.iterations <= 3
    assert run.exit_reason in {"converged", "max_iter", "oda_stall"}
    assert np.all(np.isfinite(run.iter_energy))
    assert np.all(np.isfinite(run.iter_err))
    assert finite_field_filling(inputs.state.density) == pytest.approx(0.0)
    for ik in range(inputs.state.nk):
        np.testing.assert_allclose(inputs.state.hamiltonian[:, :, ik], inputs.state.hamiltonian[:, :, ik].conj().T, atol=1.0e-10)


def test_build_hf_inputs_from_parameters_no_io_smoke() -> None:
    params = FiniteFieldBMParameters.from_degrees(1.20, w0=0.0, w1=0.0, strain=0.0, deformation_potential=0.0)
    inputs = build_finite_field_hf_inputs_from_parameters(
        params,
        flux=MagneticFlux(1, 1),
        n_landau=3,
        nq=1,
        nu=0.0,
        v0=0.0,
        shifts=((0, 0),),
        include_strain=False,
    )

    assert inputs.state.nt == 8
    assert inputs.state.nk == 1
    assert inputs.overlap_data.overlaps[(0, 0)].shape == (8, 1, 8, 1)
    np.testing.assert_allclose(inputs.k_vectors, np.array([0.0 + 0.0j]))
    with pytest.raises(ValueError, match="shifts or shell_ng"):
        build_finite_field_hf_inputs_from_parameters(
            params,
            flux=MagneticFlux(1, 1),
            n_landau=3,
            nq=1,
            nu=0.0,
            v0=0.0,
            include_strain=False,
        )


def test_q2_assembled_hf_smoke_exercises_magnetic_strips() -> None:
    k_result, kp_result = _tiny_q2_spectrum_pair()
    inputs = build_finite_field_hf_inputs_from_spectra(
        k_result,
        kp_result,
        nu=0.0,
        v0=0.03,
        shifts=((1, 0),),
    )
    assert inputs.state.nt == 16
    assert inputs.state.nk == 2
    assert inputs.overlap_data.overlaps[(1, 0)].shape == (16, 2, 16, 2)

    run = run_finite_field_hartree_fock_from_inputs(
        inputs,
        screening_lm=1.0,
        init_mode="bm",
        seed=0,
        max_iter=2,
    )

    assert run.iterations >= 1
    assert np.isfinite(inputs.state.diagnostics["hf_energy"])
    assert finite_field_filling(inputs.state.density) == pytest.approx(0.0)
    summary = summarize_finite_field_hartree_fock(inputs.state, run)
    assert summary.filling == pytest.approx(0.0)
    assert summary.iterations == run.iterations
    assert summary.exit_reason == run.exit_reason
    assert np.isfinite(summary.energy_per_muc)


def test_unified_tl_symmetric_hf_inputs_and_smoke_exercise_reduced_mesh() -> None:
    k_result, kp_result = _tiny_q2_spectrum_pair()
    inputs = build_finite_field_hf_inputs_from_spectra(
        k_result,
        kp_result,
        nu=0.0,
        v0=0.02,
        shifts=((1, 0),),
        reduced_translation=True,
    )
    wrapper_inputs = build_tl_symmetric_finite_field_hf_inputs_from_spectra(
        k_result,
        kp_result,
        nu=0.0,
        v0=0.02,
        shifts=((1, 0),),
    )

    assert isinstance(inputs, FiniteFieldTLSymmetricHartreeFockInputs)
    assert inputs.state.reduced_translation is True
    assert inputs.state.nt == 16
    assert inputs.state.nk == 1
    assert inputs.full_k_vectors.shape == (2,)
    assert inputs.overlap_data.overlaps[(1, 0)].shape == (16, 2, 16, 2)
    np.testing.assert_allclose(inputs.state.h0, wrapper_inputs.state.h0, atol=1.0e-12)

    kernel = build_finite_field_hf_kernel_from_inputs(inputs, screening_lm=1.0, phi=0.25)
    assert kernel.convergence_rule == "mixed"
    run = run_finite_field_hartree_fock_from_inputs(
        inputs,
        screening_lm=1.0,
        init_mode="bm",
        seed=0,
        max_iter=2,
        phi=0.25,
    )

    assert run.iterations >= 1
    assert np.isfinite(inputs.state.diagnostics["hf_energy"])
    assert finite_field_filling(inputs.state.density) == pytest.approx(0.0)
    summary = summarize_finite_field_hartree_fock(inputs.state, run)
    assert summary.filling == pytest.approx(0.0)
    assert summary.iterations == run.iterations
    assert summary.exit_reason == run.exit_reason


def test_tl_symmetric_hf_inputs_from_parameters_no_io_smoke() -> None:
    params = FiniteFieldBMParameters.from_degrees(1.20, w0=0.0, w1=0.0, strain=0.0, deformation_potential=0.0)
    inputs = build_tl_symmetric_finite_field_hf_inputs_from_parameters(
        params,
        flux=MagneticFlux(1, 2),
        n_landau=4,
        nq=1,
        nu=0.0,
        v0=0.0,
        shifts=((0, 0),),
        include_strain=False,
    )

    assert inputs.state.reduced_translation is True
    assert inputs.state.nk == 1
    assert inputs.full_k_vectors.shape == (2,)
    assert inputs.normalization_count == 4
    with pytest.raises(ValueError, match="shifts or shell_ng"):
        build_tl_symmetric_finite_field_hf_inputs_from_parameters(
            params,
            flux=MagneticFlux(1, 2),
            n_landau=4,
            nq=1,
            nu=0.0,
            v0=0.0,
            include_strain=False,
        )


def test_full_magnetic_interaction_matches_author_loop_formula() -> None:
    rng = np.random.default_rng(4)
    nt = 3
    nk = 2
    density = _random_complex(rng, (nt, nt, nk))
    overlap = _random_complex(rng, (nt, nk, nt, nk))
    kvec = np.array([0.0 + 0.0j, 0.2 + 0.1j])
    gvec = 1.3 - 0.4j
    data = MagneticOverlapData(shifts=((1, 0),), gvecs=np.asarray([gvec]), overlaps={(1, 0): overlap})
    lm = 1.7
    v0 = 2.3
    norm = 11

    got = build_magnetic_interaction_hamiltonian(
        density,
        data,
        k_vectors=kvec,
        v0=v0,
        normalization_count=norm,
        screening_lm=lm,
        use_numba=False,
    )

    expected = np.zeros_like(density)
    prefactor = v0 / norm
    diagonal = np.stack([overlap[:, ik, :, ik] for ik in range(nk)], axis=2)
    tr_pg = 0.0 + 0.0j
    for ik in range(nk):
        tr_pg += np.trace(density[:, :, ik] @ np.conj(diagonal[:, :, ik]))
    for ik in range(nk):
        expected[:, :, ik] += prefactor * screened_coulomb_finite_b(gvec, lm) * tr_pg * diagonal[:, :, ik]
        for ip in range(nk):
            lam = overlap[:, ik, :, ip]
            expected[:, :, ik] -= (
                prefactor
                * screened_coulomb_finite_b(kvec[ip] - kvec[ik] + gvec, lm)
                * (lam @ density[:, :, ip].T @ lam.conj().T)
            )

    np.testing.assert_allclose(got, expected, atol=1.0e-12)


def test_tl_symmetric_interaction_matches_reduced_iks_loop_formula() -> None:
    rng = np.random.default_rng(8)
    flux = MagneticFlux(1, 2)
    q = flux.q
    nq = 1
    nk_reduced = 1
    nt = 2 * q * 2 * 2
    density = _random_complex(rng, (nt, nt, nk_reduced))
    overlap = _random_complex(rng, (nt, q * nk_reduced, nt, q * nk_reduced))
    gvec = 0.9 + 0.2j
    data = MagneticOverlapData(shifts=((1, 0),), gvecs=np.asarray([gvec]), overlaps={(1, 0): overlap})
    full_k = magnetic_k_vectors(g1=1.0 + 0.0j, g2=1.0j, flux=flux, nq=nq)
    lm = 1.2
    v0 = 1.4
    norm = magnetic_normalization_count(flux, nq)
    phi = 0.37

    got = build_tl_symmetric_magnetic_interaction_hamiltonian(
        density,
        data,
        full_k_vectors=full_k,
        flux=flux,
        nq=nq,
        v0=v0,
        normalization_count=norm,
        screening_lm=lm,
        phi=phi,
    )

    expected = np.zeros_like(density)
    indices = magnetic_orbit_indices(q, nq)
    rps = magnetic_r_orbit_positions(flux.p, flux.q)
    prefactor = v0 / norm
    diag_full = indices[0, :]
    diagonal = np.stack([overlap[:, int(full_ik), :, int(full_ik)] for full_ik in diag_full], axis=2)
    tr_pg = 0.0 + 0.0j
    for ik in range(nk_reduced):
        tr_pg += np.trace(density[:, :, ik] @ np.conj(diagonal[:, :, ik]))
    expected += prefactor * screened_coulomb_finite_b(gvec, lm) * tr_pg * q * diagonal
    for ik in range(nk_reduced):
        target = indices[0, ik]
        for ip in range(nk_reduced):
            for rp in range(q):
                source = indices[rp, ip]
                lam = overlap[:, target, :, source]
                density_t = apply_iks_phase_to_transposed_density(
                    density[:, :, ip].T,
                    q=q,
                    rp_position=int(rps[rp]),
                    phi=phi,
                )
                expected[:, :, ik] -= (
                    prefactor
                    * screened_coulomb_finite_b(full_k[source] - full_k[target] + gvec, lm)
                    * (lam @ density_t @ lam.conj().T)
                )

    np.testing.assert_allclose(got, expected, atol=1.0e-12)


def test_finite_field_state_initialization_smoke() -> None:
    h0 = np.zeros((4, 4, 2), dtype=np.complex128)
    h0[:, :, 0] = np.diag([0.0, 2.0, 4.0, 6.0])
    h0[:, :, 1] = np.diag([1.0, 3.0, 5.0, 7.0])
    state = FiniteFieldHartreeFockState.from_h0(h0, nu=0.0, flux=MagneticFlux(1, 2), nq=1, v0=1.0)

    from mean_field.systems.tbg.finite_field import initialize_density_from_h0

    initialize_density_from_h0(state, init_mode="bm", seed=0)
    assert state.density.shape == h0.shape
    assert finite_field_filling(state.density) == pytest.approx(0.0)


def test_random_initialization_applies_author_coherent_rotations() -> None:
    h0 = np.zeros((8, 8, 3), dtype=np.complex128)
    sigma_z = np.zeros_like(h0)
    state = FiniteFieldHartreeFockState.from_h0(h0, sigma_z=sigma_z, nu=0.0, flux=MagneticFlux(1, 1), nq=1, v0=1.0)

    from mean_field.systems.tbg.finite_field import initialize_density_from_h0

    initialize_density_from_h0(state, init_mode="random", seed=3)
    assert finite_field_filling(state.density) == pytest.approx(0.0)
    offdiag = state.density - np.asarray([np.diag(np.diag(state.density[:, :, ik])) for ik in range(state.nk)]).transpose(1, 2, 0)
    assert np.max(np.abs(offdiag)) > 1.0e-3
    for ik in range(state.nk):
        eigs = np.linalg.eigvalsh((state.density[:, :, ik] + state.density[:, :, ik].conj().T) / 2.0)
        np.testing.assert_allclose(np.abs(eigs), np.full(8, 0.5), atol=1.0e-12)
