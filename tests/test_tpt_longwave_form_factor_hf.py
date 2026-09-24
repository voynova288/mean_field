from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mean_field.systems.tpt.longwave_form_factor_hf import (
    apply_periodic_wannier_fock_fft,
    build_fixed_rank_density_builder,
    build_periodic_scalar_2d_coulomb_kernel,
    direct_periodic_wannier_fock_elements,
    reconstruct_complete_wannier_h0_and_reference,
    rectangular_coulomb_self_cell_average,
    reference_relative_energy,
)


def _hermitian(rng, shape):
    value = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return 0.5 * (value + value.conj().swapaxes(-1, -2))


def test_rectangular_self_cell_square_matches_analytic_integral():
    width = 0.2
    constant = 3.0
    expected_integral = 8.0 * (width / 2.0) * np.arcsinh(1.0)
    expected = 2.0 * np.pi * constant * expected_integral / width**2
    actual = rectangular_coulomb_self_cell_average(width, width, coulomb_ev_angstrom=constant)
    assert np.isclose(actual, expected, rtol=1e-14, atol=0.0)


def test_periodic_coulomb_kernel_is_inversion_symmetric():
    reciprocal = np.array([[2.0, 0.0], [0.0, 0.8]])
    kernel, q_norm = build_periodic_scalar_2d_coulomb_kernel((6, 4), reciprocal)
    inverse_i = (-np.arange(6)) % 6
    inverse_j = (-np.arange(4)) % 4
    assert kernel.shape == (6, 4)
    assert q_norm.shape == (6, 4)
    assert np.allclose(kernel, kernel[np.ix_(inverse_i, inverse_j)], rtol=0.0, atol=1e-14)
    assert np.all(kernel > 0.0)


def test_fft_fock_matches_direct_selected_elements():
    rng = np.random.default_rng(52)
    n1, n2, nw = 4, 3, 3
    reciprocal = np.array([[1.8, 0.0], [0.0, 0.7]])
    kernel, _ = build_periodic_scalar_2d_coulomb_kernel((n1, n2), reciprocal)
    density_ket = _hermitian(rng, (n1, n2, nw, nw))
    density_stored = np.swapaxes(density_ket.reshape(n1 * n2, nw, nw), 1, 2).transpose(1, 2, 0)
    weight = 0.013
    sigma_abk = apply_periodic_wannier_fock_fft(
        density_stored,
        kernel,
        mesh_shape=(n1, n2),
        quadrature_weight_angstrom_minus2=weight,
        epsilon=2.3,
        workers=1,
    )
    sigma_grid = sigma_abk.transpose(2, 0, 1).reshape(n1, n2, nw, nw)
    targets = [(0, 0, 0, 0), (1, 2, 0, 1), (3, 1, 2, 1), (2, 0, 2, 2)]
    direct = direct_periodic_wannier_fock_elements(
        density_ket,
        kernel,
        targets,
        quadrature_weight_angstrom_minus2=weight,
        epsilon=2.3,
    )
    fft_values = np.asarray([sigma_grid[i, j, a, b] for i, j, a, b in targets])
    assert np.allclose(fft_values, direct, rtol=2e-13, atol=2e-13)
    assert np.max(np.abs(sigma_grid - sigma_grid.conj().swapaxes(-1, -2))) < 2e-13


def test_complete_reconstruction_and_fixed_rank_reference_replay():
    rng = np.random.default_rng(80)
    nk, nw, nocc = 5, 6, 4
    vectors = np.empty((nk, nw, nw), dtype=np.complex128)
    for k in range(nk):
        vectors[k] = np.linalg.qr(rng.normal(size=(nw, nw)) + 1j * rng.normal(size=(nw, nw)))[0]
    energies = np.stack([np.linspace(-2.0, 2.0, nw) + 0.01 * k for k in range(nk)])
    h0, reference = reconstruct_complete_wannier_h0_and_reference(energies, vectors, occupied_bands=nocc)
    replay = np.empty_like(energies)
    for k in range(nk):
        replay[k] = np.linalg.eigvalsh(h0[:, :, k])
    assert np.allclose(replay, energies, rtol=1e-13, atol=1e-13)
    assert np.max(np.abs(reference @ reference - reference)) < 2e-13
    with ThreadPoolExecutor(max_workers=2) as executor:
        update = build_fixed_rank_density_builder(reference, occupied_bands=nocc, executor=executor)
        result = update(h0)
    assert np.max(np.abs(result.density)) < 2e-13
    assert result.observables["minimum_direct_gap_ev"] > 0.0


def test_shared_executors_and_fft_are_concurrent_call_safe():
    rng = np.random.default_rng(314)
    n1, n2, nw, nocc = 3, 2, 4, 2
    kernel, _ = build_periodic_scalar_2d_coulomb_kernel(
        (n1, n2), np.array([[1.1, 0.0], [0.0, 0.8]])
    )
    density_ket = _hermitian(rng, (n1, n2, nw, nw))
    density_stored = np.swapaxes(
        density_ket.reshape(n1 * n2, nw, nw), 1, 2
    ).transpose(1, 2, 0)

    def fock_call(_index):
        return apply_periodic_wannier_fock_fft(
            density_stored,
            kernel,
            mesh_shape=(n1, n2),
            quadrature_weight_angstrom_minus2=0.01,
            epsilon=2.0,
            workers=1,
        )

    expected_fock = fock_call(0)
    with ThreadPoolExecutor(max_workers=5) as outer:
        concurrent_fock = list(outer.map(fock_call, range(5)))
    for result in concurrent_fock:
        assert np.array_equal(result, expected_fock)

    h0_ket = _hermitian(rng, (n1 * n2, nw, nw))
    h0 = h0_ket.transpose(1, 2, 0)
    reference = np.zeros_like(h0_ket)
    with ThreadPoolExecutor(max_workers=4) as inner:
        update = build_fixed_rank_density_builder(
            reference,
            occupied_bands=nocc,
            executor=inner,
        )
        expected_density = update(h0)
        with ThreadPoolExecutor(max_workers=5) as outer:
            concurrent_density = list(outer.map(lambda _index: update(h0), range(5)))
    for result in concurrent_density:
        assert np.array_equal(result.density, expected_density.density)
        assert np.array_equal(result.energies, expected_density.energies)
        assert result.mu == expected_density.mu


def test_fft_energy_derivative_matches_h0_plus_fock():
    rng = np.random.default_rng(11)
    n1, n2, nw = 3, 3, 2
    nk = n1 * n2
    reciprocal = np.array([[1.2, 0.0], [0.0, 0.9]])
    kernel, _ = build_periodic_scalar_2d_coulomb_kernel((n1, n2), reciprocal)
    h0_ket = _hermitian(rng, (nk, nw, nw))
    h0 = h0_ket.transpose(1, 2, 0)
    d_ket = _hermitian(rng, (nk, nw, nw))
    delta_ket = _hermitian(rng, (nk, nw, nw))
    d_stored = np.swapaxes(d_ket, 1, 2).transpose(1, 2, 0)
    delta_stored = np.swapaxes(delta_ket, 1, 2).transpose(1, 2, 0)
    weight = 0.02

    def interaction(density):
        return apply_periodic_wannier_fock_fft(
            density,
            kernel,
            mesh_shape=(n1, n2),
            quadrature_weight_angstrom_minus2=weight,
            epsilon=1.7,
            workers=1,
        )

    step = 1e-6
    plus = d_stored + step * delta_stored
    minus = d_stored - step * delta_stored
    finite_difference = (
        reference_relative_energy(interaction(plus), h0, plus)
        - reference_relative_energy(interaction(minus), h0, minus)
    ) / (2.0 * step)
    analytic = np.einsum("abk,abk->", delta_stored, h0 + interaction(d_stored)).real / nk
    assert abs(finite_difference - analytic) < 2e-8
