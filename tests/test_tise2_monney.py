from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from mean_field.core.hf.occupations import (
    conventional_projector_to_stored,
    global_fermi_occupations,
)
from mean_field.systems.tise2.hf import (
    COULOMB_EV_ANGSTROM,
    MonneyHFConfig,
    build_monney_hf_state,
    build_monney_local_mesh,
    linearized_monney_gap,
    monney_interaction_action,
    precompute_monney_coulomb_kernel,
    run_monney_hf,
)
from mean_field.systems.tise2.monney import (
    MonneyTiSe2Parameters,
    monney_bare_energies,
    monney_characteristic_denominator,
    monney_mean_field_hamiltonian,
)


def test_monney_default_matches_the_literal_printed_cosine_sign() -> None:
    params = MonneyTiSe2Parameters()
    dz = 1.0e-3
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, dz]])
    energies = monney_bare_energies(points, params)

    assert params.conduction_z_convention == "paper_literal"
    assert np.isclose(energies[0, 0], 0.030)
    assert np.allclose(energies[1:, 0], 0.020)
    assert energies[0, 1] < energies[0, 0]
    assert np.all(energies[1:, 1] < energies[1:, 0])


def test_l_minimum_convention_is_explicitly_diagnostic() -> None:
    params = MonneyTiSe2Parameters(
        conduction_z_convention="l_minimum_diagnostic"
    )
    dz = 1.0e-3
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, dz]])
    energies = monney_bare_energies(points, params)
    assert np.allclose(energies[1:, 0], -0.040)
    assert np.all(energies[1:, 1] > energies[1:, 0])


def test_monney_hamiltonian_matches_equation_16_characteristic_polynomial() -> None:
    points = np.asarray(
        [[0.011, -0.017, 0.023], [-0.019, 0.007, -0.031]], dtype=float
    )
    delta = np.asarray(
        [[0.015 + 0.004j, 0.010 - 0.002j],
         [0.012 - 0.003j, 0.017 + 0.001j],
         [0.009 + 0.005j, 0.014 - 0.006j]],
        dtype=np.complex128,
    )
    hamiltonian = monney_mean_field_hamiltonian(points, delta)
    bare = monney_bare_energies(points)
    z = np.asarray([0.087 + 0.013j, -0.041 + 0.019j])
    formula = monney_characteristic_denominator(z, bare, delta)
    direct = np.asarray(
        [np.linalg.det(z[ik] * np.eye(4) - hamiltonian[:, :, ik]) for ik in range(2)]
    )
    assert np.allclose(formula, direct, rtol=2.0e-13, atol=2.0e-15)


def test_equal_delta_has_two_dark_conduction_states() -> None:
    points = np.zeros((1, 3), dtype=float)
    delta_value = 0.025
    hamiltonian = monney_mean_field_hamiltonian(
        points, np.full(3, delta_value, dtype=float)
    )
    values = np.linalg.eigvalsh(hamiltonian[:, :, 0])
    ev = 0.030
    ec = 0.020
    splitting = np.sqrt((ev - ec) ** 2 + 12.0 * delta_value**2)
    expected = np.sort(
        np.asarray(
            [
                0.5 * (ev + ec - splitting),
                ec,
                ec,
                0.5 * (ev + ec + splitting),
            ]
        )
    )
    assert np.allclose(values, expected, atol=1.0e-14, rtol=0.0)


def test_local_mesh_is_c3_closed_and_has_expected_size() -> None:
    mesh = build_monney_local_mesh(
        inplane_shells=2,
        inplane_spacing_Ainv=0.04,
        z_shells=1,
        z_spacing_Ainv=0.05,
    )
    assert mesh.nk == (1 + 3 * 2 * 3) * 3
    assert np.allclose(mesh.points_Ainv[mesh.gamma_index], 0.0)

    rotation = np.asarray(
        [[-0.5, -np.sqrt(3.0) / 2.0], [np.sqrt(3.0) / 2.0, -0.5]]
    )
    point_set = {
        tuple(np.round(point, decimals=12)) for point in np.asarray(mesh.points_Ainv)
    }
    for point in mesh.points_Ainv:
        rotated = np.asarray([*(rotation @ point[:2]), point[2]])
        assert tuple(np.round(rotated, decimals=12)) in point_set


def test_coulomb_kernel_has_integrated_self_cell_and_epsilon_scaling() -> None:
    mesh = build_monney_local_mesh(
        inplane_shells=1,
        inplane_spacing_Ainv=0.04,
        z_shells=1,
        z_spacing_Ainv=0.05,
    )
    kernel3 = precompute_monney_coulomb_kernel(mesh, epsilon_r=3.0)
    kernel6 = precompute_monney_coulomb_kernel(mesh, epsilon_r=6.0)
    assert np.all(np.isfinite(kernel3.weighted_matrix_ev))
    assert np.all(kernel3.weighted_matrix_ev > 0.0)
    assert np.allclose(kernel3.weighted_matrix_ev, kernel3.weighted_matrix_ev.T)
    cell_radius = (3.0 * mesh.cell_volume_Ainv3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    analytic_self = 2.0 * COULOMB_EV_ANGSTROM * cell_radius / (np.pi * 3.0)
    assert np.allclose(np.diag(kernel3.weighted_matrix_ev), analytic_self)
    assert np.isclose(kernel3.self_cell_ev, analytic_self)
    assert np.allclose(kernel6.weighted_matrix_ev, 0.5 * kernel3.weighted_matrix_ev)


def test_coulomb_peak_memory_cap_fails_before_dense_allocation() -> None:
    mesh = build_monney_local_mesh(
        inplane_shells=1,
        inplane_spacing_Ainv=0.04,
        z_shells=1,
        z_spacing_Ainv=0.05,
    )
    itemsize = np.dtype(np.float64).itemsize
    one_row_minimum_bytes = (
        mesh.nk * mesh.nk * itemsize + 6 * mesh.nk * itemsize
    )
    with pytest.raises(MemoryError, match="peak-memory cap"):
        precompute_monney_coulomb_kernel(
            mesh,
            epsilon_r=4.0,
            max_dense_memory_gb=0.5 * one_row_minimum_bytes / 1.0e9,
        )


def test_first_order_response_locks_sign_phase_and_stored_orientation() -> None:
    ev, ec = -0.06, 0.04
    kbt = 0.01
    normal = np.diag([ev, ec, 1.0, 1.2]).astype(np.complex128)[None, :, :]
    normal_occ = global_fermi_occupations(
        normal,
        target_particle_number=1.0,
        kbt_ev=kbt,
        k_weights=np.ones(1),
    )
    delta = 1.0e-7 * (1.0 + 0.4j)
    perturbed = normal.copy()
    perturbed[0, 0, 1] = -delta
    perturbed[0, 1, 0] = -delta.conjugate()
    response = global_fermi_occupations(
        perturbed,
        target_particle_number=1.0,
        kbt_ev=kbt,
        k_weights=np.ones(1),
    )
    stored = conventional_projector_to_stored(
        np.moveaxis(response.density_ket, 0, 2)
    )
    mu = normal_occ.chemical_potential
    fv = expit((mu - ev) / kbt)
    fc = expit((mu - ec) / kbt)
    expected = (fv - fc) * delta / (ec - ev)
    assert np.allclose(stored[1, 0, 0], expected, rtol=2.0e-6, atol=1.0e-13)

    mesh = build_monney_local_mesh(
        inplane_shells=0,
        inplane_spacing_Ainv=0.04,
        z_shells=0,
        z_spacing_Ainv=0.05,
    )
    kernel = precompute_monney_coulomb_kernel(mesh, epsilon_r=4.0)
    density_delta = np.zeros((4, 4, 1), dtype=np.complex128)
    density_delta[1, 0, 0] = stored[1, 0, 0]
    density_delta[0, 1, 0] = stored[1, 0, 0].conjugate()
    sigma = monney_interaction_action(density_delta, coulomb=kernel)
    assert np.allclose(
        sigma[0, 1, 0],
        -kernel.weighted_matrix_ev[0, 0] * stored[1, 0, 0],
    )
    assert np.allclose(sigma[1, 0, 0], sigma[0, 1, 0].conjugate())


def test_anomalous_interaction_is_the_derivative_of_its_scalar_energy() -> None:
    mesh = build_monney_local_mesh(
        inplane_shells=1,
        inplane_spacing_Ainv=0.04,
        z_shells=0,
        z_spacing_Ainv=0.05,
    )
    kernel = precompute_monney_coulomb_kernel(mesh, epsilon_r=4.0)
    rng = np.random.default_rng(7)
    density = np.zeros((4, 4, mesh.nk), dtype=np.complex128)
    direction = np.zeros_like(density)
    for valley in range(1, 4):
        coherence = 0.03 * (
            rng.normal(size=mesh.nk) + 1j * rng.normal(size=mesh.nk)
        )
        tangent = 0.02 * (
            rng.normal(size=mesh.nk) + 1j * rng.normal(size=mesh.nk)
        )
        density[valley, 0] = coherence
        density[0, valley] = coherence.conj()
        direction[valley, 0] = tangent
        direction[0, valley] = tangent.conj()

    weights = mesh.weights_Ainv3

    def energy(value: np.ndarray) -> float:
        sigma = monney_interaction_action(value, coulomb=kernel)
        return float(
            0.5
            * np.einsum("abk,abk,k->", sigma, value, weights, optimize=True).real
        )

    step = 1.0e-6
    finite_difference = (energy(density + step * direction) - energy(density - step * direction)) / (2.0 * step)
    sigma = monney_interaction_action(density, coulomb=kernel)
    analytic = float(
        np.einsum("abk,abk,k->", sigma, direction, weights, optimize=True).real
    )
    assert np.isclose(finite_difference, analytic, rtol=2.0e-8, atol=1.0e-14)


def test_linearized_gap_kernel_is_c3_equivalent() -> None:
    config = MonneyHFConfig(
        epsilon_r=3.15,
        inplane_shells=1,
        z_shells=1,
        inplane_spacing_Ainv=0.04,
        z_spacing_Ainv=0.05,
        max_dense_memory_gb=0.1,
    )
    state = build_monney_hf_state(config)
    result = linearized_monney_gap(state)
    assert np.all(result.largest_eigenvalues > 0.0)
    assert result.c3_relative_spread < 2.0e-13
    assert np.allclose(
        result.critical_epsilon_r,
        config.epsilon_r * result.largest_eigenvalues,
    )


def test_generic_hf_engine_accepts_the_exact_normal_branch() -> None:
    config = MonneyHFConfig(
        epsilon_r=10.0,
        inplane_shells=1,
        z_shells=1,
        inplane_spacing_Ainv=0.04,
        z_spacing_Ainv=0.05,
        precision=1.0e-11,
        max_iter=5,
        mixing=0.2,
        seed_delta_ev=0.0,
        max_dense_memory_gb=0.1,
    )
    state = build_monney_hf_state(config)
    result = run_monney_hf(state, init_mode="normal")
    assert result.run.converged
    assert result.run.exit_reason == "converged"
    assert result.max_delta_ev == 0.0
    assert abs(result.raw_particle_residual) <= 1.0e-10
    assert abs(result.mixed_particle_residual) <= 1.0e-10
    assert result.final_raw_norm <= config.precision
    assert np.max(
        np.abs(
            result.raw_update_projector_stored
            - result.mixed_projector_stored
        )
    ) <= config.precision
    assert result.hermiticity_residual_ev <= 1.0e-13
