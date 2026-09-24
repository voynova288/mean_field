from __future__ import annotations

import numpy as np

from mean_field.core.hf.real_space_density_density import (
    PeriodicDensityDensityKernel,
    direct_periodic_density_density_elements,
    reference_relative_density_density_energy,
)


def _periodic_test_kernel(
    *, mesh_shape: tuple[int, int] = (3, 2), n_spatial: int = 2, spin_blocks: int = 2
) -> PeriodicDensityDensityKernel:
    rng = np.random.default_rng(1701)
    n1, n2 = mesh_shape
    real_space = rng.normal(size=(n_spatial, n_spatial, n1, n2))
    inverse_i = (-np.arange(n1)) % n1
    inverse_j = (-np.arange(n2)) % n2
    partner = real_space.swapaxes(0, 1)[:, :, inverse_i][:, :, :, inverse_j]
    real_space = 0.5 * (real_space + partner)
    kernel_q = np.fft.fftn(real_space, axes=(-2, -1))
    hartree = np.asarray(kernel_q[:, :, 0, 0].real)
    return PeriodicDensityDensityKernel(
        mesh_shape=mesh_shape,
        fock_kernel_q=kernel_q,
        hartree_kernel=hartree,
        spin_blocks=spin_blocks,
    )


def _random_hermitian_density(kernel: PeriodicDensityDensityKernel, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(kernel.n_basis, kernel.n_basis, kernel.nk))
    raw = raw + 1.0j * rng.normal(size=raw.shape)
    return 0.5 * (raw + raw.conj().swapaxes(0, 1))


def test_fft_action_matches_literal_selected_elements() -> None:
    kernel = _periodic_test_kernel()
    density = _random_hermitian_density(kernel, 2027)
    result = kernel.apply(density)
    targets = [
        (0, 0, 0, 0),
        (1, 1, 1, 2),
        (2, 0, 3, 1),
        (0, 1, 2, 2),
    ]
    direct = direct_periodic_density_density_elements(kernel, density, targets)
    fft_values = np.asarray(
        [
            result.interaction_h[a, b, i * kernel.mesh_shape[1] + j]
            for i, j, a, b in targets
        ]
    )
    assert np.max(np.abs(direct - fft_values)) < 2.0e-12


def test_interaction_is_hermitian_and_zero_density_is_zero() -> None:
    kernel = _periodic_test_kernel()
    density = _random_hermitian_density(kernel, 2031)
    result = kernel.apply(density)
    assert np.max(
        np.abs(result.interaction_h - result.interaction_h.conj().swapaxes(0, 1))
    ) < 2.0e-12

    zero = kernel.apply(np.zeros_like(density))
    assert np.array_equal(zero.interaction_h, np.zeros_like(density))
    assert zero.interaction_energy == 0.0


def test_one_cell_one_orbital_self_term_cancels() -> None:
    kernel = PeriodicDensityDensityKernel(
        mesh_shape=(1, 1),
        fock_kernel_q=np.asarray([[[[5.25]]]], dtype=np.complex128),
        hartree_kernel=np.asarray([[5.25]]),
        spin_blocks=1,
    )
    density = np.asarray([[[0.37]]], dtype=np.complex128)
    result = kernel.apply(density)
    assert np.max(np.abs(result.hartree_h + result.fock_h)) < 2.0e-14
    assert np.max(np.abs(result.interaction_h)) < 2.0e-14
    assert abs(result.interaction_energy) < 2.0e-14


def test_energy_directional_derivative_matches_h0_plus_self_energy() -> None:
    kernel = _periodic_test_kernel(mesh_shape=(2, 3), n_spatial=2, spin_blocks=1)
    density = _random_hermitian_density(kernel, 3041)
    direction = _random_hermitian_density(kernel, 3042)
    h0 = _random_hermitian_density(kernel, 3043)

    def energy(trial: np.ndarray) -> float:
        action = kernel.apply(trial)
        return reference_relative_density_density_energy(h0, trial, action)

    epsilon = 2.0e-7
    numerical = (energy(density + epsilon * direction) - energy(density - epsilon * direction)) / (
        2.0 * epsilon
    )
    action = kernel.apply(density)
    analytic = float(
        np.einsum(
            "abk,abk->", h0 + action.interaction_h, direction, optimize=True
        ).real
        / kernel.nk
    )
    assert abs(numerical - analytic) < 3.0e-8


def test_neutral_background_policy_rejects_charged_density() -> None:
    kernel = PeriodicDensityDensityKernel(
        mesh_shape=(1, 1),
        fock_kernel_q=np.asarray([[[[1.0]]]], dtype=np.complex128),
        hartree_kernel=np.asarray([[0.0]]),
        spin_blocks=1,
        require_neutral_density=True,
        neutrality_tolerance=1.0e-12,
    )
    with np.testing.assert_raises_regex(ValueError, "requires a neutral"):
        kernel.apply(np.asarray([[[0.1]]], dtype=np.complex128))


def test_spin_block_broadcast_is_covariant_under_spin_swap() -> None:
    kernel = _periodic_test_kernel(mesh_shape=(2, 2), n_spatial=2, spin_blocks=2)
    density = _random_hermitian_density(kernel, 4041)
    permutation = np.asarray([2, 3, 0, 1])
    swapped = density[permutation][:, permutation]
    original_action = kernel.apply(density).interaction_h
    swapped_action = kernel.apply(swapped).interaction_h
    expected = original_action[permutation][:, permutation]
    assert np.max(np.abs(swapped_action - expected)) < 2.0e-12
