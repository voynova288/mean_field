from __future__ import annotations

from pathlib import Path

import numpy as np

from mean_field.core.hf.problem import run_hartree_fock_problem
from mean_field.core.hf.real_space_density_density import PeriodicDensityDensityKernel
from mean_field.systems.tpt.wannier_keldysh_hf import (
    COULOMB_EV_ANGSTROM,
    TPT_PUBLISHED_ALPHA_2D_ANGSTROM,
    WannierGaussianDensities,
    build_gaussian_keldysh_density_density_kernel,
    build_spin_doubled_h0,
    evaluate_wannier_hr_on_mesh,
    parse_final_wannier_gaussian_densities,
    parse_wannier90_hr,
    regular_fractional_mesh,
)
from mean_field.systems.tpt.wannier_q0_hf import (
    TPTWannierQ0HFInputs,
    TPTWannierQ0HFState,
    build_projector_seed_from_source,
    build_tpt_wannier_q0_hf_problem,
)


def test_parse_final_centres_and_spreads_uses_last_final_state(tmp_path: Path) -> None:
    wout = tmp_path / "wannier90.wout"
    wout.write_text(
        """
 Final State
 WF centre and spread 1 ( 9.0, 9.0, 9.0 ) 9.0
 ignored
 Final State
 WF centre and spread 1 ( 0.25, 0.50, 1.50 ) 4.0
 WF centre and spread 2 ( 1.25, 2.50, 3.50 ) 2.25
 Sum of centres and spreads
""",
        encoding="utf-8",
    )
    data = parse_final_wannier_gaussian_densities(wout, expected_num_wann=2)
    assert np.array_equal(
        data.centres_cartesian_angstrom,
        np.asarray([[0.25, 0.50, 1.50], [1.25, 2.50, 3.50]]),
    )
    assert np.array_equal(data.total_spreads_angstrom2, np.asarray([4.0, 2.25]))
    assert np.array_equal(data.gaussian_widths_angstrom, np.asarray([2.0, 1.5]))


def test_parse_and_evaluate_one_orbital_hr(tmp_path: Path) -> None:
    hr_path = tmp_path / "wannier90_hr.dat"
    hr_path.write_text(
        """test HR
1
3
    1    1    1
   -1    0    0    1    1      0.500000000000      0.000000000000
    0    0    0    1    1      1.000000000000      0.000000000000
    1    0    0    1    1      0.500000000000      0.000000000000
""",
        encoding="utf-8",
    )
    hr = parse_wannier90_hr(hr_path)
    kpoints = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    h0 = evaluate_wannier_hr_on_mesh(hr, kpoints)
    assert h0.shape == (1, 1, 2)
    assert np.allclose(h0[0, 0], np.asarray([2.0, 0.0]), atol=2.0e-15)


def test_regular_mesh_and_spin_doubling_order() -> None:
    mesh = regular_fractional_mesh((2, 3))
    assert np.array_equal(
        mesh,
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0 / 3.0, 0.0],
                [0.0, 2.0 / 3.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 1.0 / 3.0, 0.0],
                [0.5, 2.0 / 3.0, 0.0],
            ]
        ),
    )
    scalar = np.zeros((2, 2, 3), dtype=np.complex128)
    scalar[0, 0] = 1.0
    scalar[1, 1] = 2.0
    doubled = build_spin_doubled_h0(scalar)
    assert doubled.shape == (4, 4, 3)
    assert np.array_equal(doubled[:2, :2], scalar)
    assert np.array_equal(doubled[2:, 2:], scalar)
    assert np.count_nonzero(doubled[:2, 2:]) == 0


def test_gaussian_keldysh_builder_returns_finite_symmetric_kernel() -> None:
    wannier = WannierGaussianDensities(
        centres_cartesian_angstrom=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.5, 0.2]]
        ),
        total_spreads_angstrom2=np.asarray([2.25, 3.24]),
        gaussian_widths_angstrom=np.asarray([1.5, 1.8]),
    )
    build = build_gaussian_keldysh_density_density_kernel(
        lattice_2d_angstrom=np.diag([4.0, 8.0]),
        wannier=wannier,
        mesh_shape=(4, 2),
        alpha_2d_angstrom=TPT_PUBLISHED_ALPHA_2D_ANGSTROM,
        gaussian_tail_tolerance=1.0e-4,
        zero_cell_angular_order=16,
        zero_cell_radial_order=8,
        workers=1,
    )
    kernel = build.kernel
    assert kernel.n_spatial == 2
    assert kernel.n_basis == 4
    assert kernel.nk == 8
    assert np.all(np.isfinite(kernel.fock_kernel_q))
    assert np.min(np.linalg.eigvalsh(build.zero_cell_matrix_ev)) > -1.0e-11
    assert build.metadata["alpha_2d_angstrom"] == TPT_PUBLISHED_ALPHA_2D_ANGSTROM
    assert build.metadata["hartree_macroscopic_g0"] == "removed_neutral_background"
    assert build.metadata["raw_inversion_residual_ev"] < 1.0e-8


def test_two_centre_phase_matches_hr_positive_fourier_convention() -> None:
    centres = np.asarray([[0.0, 0.0, 0.0], [0.7, 0.3, 0.0]])
    widths = np.asarray([1.5, 1.8])
    wannier = WannierGaussianDensities(
        centres_cartesian_angstrom=centres,
        total_spreads_angstrom2=widths**2,
        gaussian_widths_angstrom=widths,
    )
    lattice = np.diag([4.0, 8.0])
    build = build_gaussian_keldysh_density_density_kernel(
        lattice_2d_angstrom=lattice,
        wannier=wannier,
        mesh_shape=(4, 2),
        gaussian_tail_tolerance=1.0e-4,
        zero_cell_angular_order=16,
        zero_cell_radial_order=8,
        workers=1,
    )
    reciprocal = build.reciprocal_vectors_angstrom_inv
    q = reciprocal[0] / 4.0
    g_vectors = build.reciprocal_shell_indices @ reciprocal
    total_q = g_vectors + q[None, :]
    norms = np.linalg.norm(total_q, axis=1)
    envelope = np.exp(-0.25 * norms**2 * (widths[0] ** 2 + widths[1] ** 2))
    centre_phase = np.exp(1.0j * (total_q @ (centres[0, :2] - centres[1, :2])))
    expected = np.sum(
        2.0
        * np.pi
        * COULOMB_EV_ANGSTROM
        / abs(np.linalg.det(lattice))
        / (norms * (1.0 + TPT_PUBLISHED_ALPHA_2D_ANGSTROM * norms))
        * envelope
        * centre_phase
    )
    assert abs(build.kernel.fock_kernel_q[0, 1, 1, 0] - expected) < 2.0e-9
    assert expected.imag < 0.0


def test_tpt_v1_rejects_unsupported_environment_rescaling() -> None:
    wannier = WannierGaussianDensities(
        centres_cartesian_angstrom=np.asarray([[0.0, 0.0, 0.0]]),
        total_spreads_angstrom2=np.asarray([2.25]),
        gaussian_widths_angstrom=np.asarray([1.5]),
    )
    with np.testing.assert_raises_regex(ValueError, "free-standing"):
        build_gaussian_keldysh_density_density_kernel(
            lattice_2d_angstrom=np.diag([4.0, 8.0]),
            wannier=wannier,
            mesh_shape=(4, 2),
            kappa=2.0,
            gaussian_tail_tolerance=1.0e-4,
            zero_cell_angular_order=16,
            zero_cell_radial_order=8,
        )


def test_q0_adapter_replays_noninteracting_reference_and_removes_seed_source() -> None:
    mesh_shape = (2, 2)
    nk = 4
    h0 = np.zeros((2, 2, nk), dtype=np.complex128)
    h0[0, 0] = -1.0
    h0[1, 1] = 1.0
    zero_kernel = PeriodicDensityDensityKernel(
        mesh_shape=mesh_shape,
        fock_kernel_q=np.zeros((2, 2, *mesh_shape), dtype=np.complex128),
        hartree_kernel=np.zeros((2, 2)),
        spin_blocks=1,
        require_neutral_density=True,
    )
    inputs = TPTWannierQ0HFInputs.from_h0(
        h0,
        interaction_kernel=zero_kernel,
        occupied_per_k=1,
        mesh_shape=mesh_shape,
    )
    source = np.zeros_like(h0)
    source[0, 1] = 0.2
    source[1, 0] = 0.2
    seed = build_projector_seed_from_source(inputs, source)
    assert np.max(np.abs(seed)) > 0.0
    problem = build_tpt_wannier_q0_hf_problem(
        inputs, initial_densities={"coherent": seed}
    )
    state = TPTWannierQ0HFState.create(inputs, precision=1.0e-12)
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode="coherent",
        seed=7,
        max_iter=5,
        oda_stall_threshold=0.0,
    )
    assert run.converged
    assert run.exit_reason == "converged"
    assert np.max(np.abs(run.state.density)) < 2.0e-14
    assert run.state.diagnostics["final_accepted"] == 1.0


def test_zero_cell_quadrature_is_numerically_converged_for_tiny_fixture() -> None:
    wannier = WannierGaussianDensities(
        centres_cartesian_angstrom=np.asarray([[0.0, 0.0, 0.0]]),
        total_spreads_angstrom2=np.asarray([2.56]),
        gaussian_widths_angstrom=np.asarray([1.6]),
    )
    common = dict(
        lattice_2d_angstrom=np.diag([4.0, 8.0]),
        wannier=wannier,
        mesh_shape=(4, 2),
        gaussian_tail_tolerance=1.0e-4,
        workers=1,
    )
    coarse = build_gaussian_keldysh_density_density_kernel(
        **common, zero_cell_angular_order=24, zero_cell_radial_order=12
    )
    fine = build_gaussian_keldysh_density_density_kernel(
        **common, zero_cell_angular_order=48, zero_cell_radial_order=20
    )
    assert abs(
        coarse.zero_cell_matrix_ev[0, 0] - fine.zero_cell_matrix_ev[0, 0]
    ) < 3.0e-9
