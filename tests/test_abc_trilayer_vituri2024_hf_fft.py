"""Focused reduced-faithful checks for the candidate Vituri FFT backend.

These tests establish finite-domain algebra parity only.  They do not establish
UV convergence, production readiness, an SCF branch, or paper reproduction.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math

import numpy as np
import pytest

import mean_field.systems.abc_trilayer.vituri2024_hf_fft as fft_module
from mean_field.systems.abc_trilayer.vituri2024 import (
    SM_TEX_SHA256,
    VITURI2024_PARAMETERS,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_fft import (
    VITURI2024_TRANSLATIONAL_HF_FFT_POLICY,
    Vituri2024TranslationalHFFFTFunctional,
    make_vituri2024_square_cartesian_fft_plan,
)
from mean_field.systems.abc_trilayer.vituri2024_hf import (
    make_vituri2024_finite_domain_mesh_receipt,
    make_vituri2024_translational_q0_reproduction_choice,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_scf import (
    VITURI2024_CM2_TO_ANGSTROM2,
    VITURI2024_TOTAL_HOLE_DENSITY_CM2,
    Vituri2024CartesianHFSpec,
    build_vituri2024_cartesian_mesh,
    make_vituri2024_cartesian_hf_spec_from_spacing,
    make_vituri2024_hf_problem,
    prepare_vituri2024_homogeneous_hf,
    prepare_vituri2024_homogeneous_hf_fft,
)
from mean_field.systems.abc_trilayer.vituri2024_interaction import (
    Vituri2024InteractionChoiceReceipt,
)


def _centered_mesh(size: int, delta_k: float = 0.125):
    half = size // 2
    labels = np.asarray(
        [
            (ix, iy)
            for iy in range(-half, half + 1)
            for ix in range(-half, half + 1)
        ],
        dtype=np.int64,
    )
    mesh = np.asarray(labels, dtype=np.float64) * delta_k
    return labels, mesh


def _random_hermitian(nk: int, seed: int, scale: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = scale * (
        rng.standard_normal((4, 4, nk))
        + 1j * rng.standard_normal((4, 4, nk))
    )
    return np.asarray(
        0.5 * (raw + raw.swapaxes(0, 1).conj()), dtype=np.complex128
    )


def _assert_scale_close(
    actual: np.ndarray | float | complex,
    expected: np.ndarray | float | complex,
    *,
    relative: float = 3.0e-11,
) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    scale = max(
        1.0,
        float(np.max(np.abs(actual_array))) if actual_array.size else 0.0,
        float(np.max(np.abs(expected_array))) if expected_array.size else 0.0,
    )
    error = (
        float(np.max(np.abs(actual_array - expected_array)))
        if actual_array.size
        else 0.0
    )
    assert error <= relative * scale


def _pair(left: np.ndarray, right: np.ndarray) -> complex:
    return complex(np.einsum("abk,bak->", left, right, optimize=False))


def _native_energy_pair(operator: np.ndarray, native_density: np.ndarray) -> complex:
    """Pair F with rho_ab=<c_a^dagger c_b>, without transposing F."""

    return complex(np.einsum("abk,abk->", operator, native_density, optimize=False))


def _test_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    payload = (
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    )
    return sha256(payload).hexdigest()


def _generic_complex_spinor_functional(
    *, mesh_size: int = 3, seed: int = 817
) -> Vituri2024TranslationalHFFFTFunctional:
    """Build a synthetic FFT functional without the dense or physical-state path."""

    labels, mesh = _centered_mesh(mesh_size, delta_k=0.071)
    nk = mesh_size * mesh_size
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((2, 6, nk)) + 1j * rng.standard_normal(
        (2, 6, nk)
    )
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    h0 = _random_hermitian(nk, seed=seed + 1, scale=0.31)
    reference = np.zeros((4, 4, nk), dtype=np.complex128)
    area = 83.25
    mesh_receipt = make_vituri2024_finite_domain_mesh_receipt(
        ordered_mesh=mesh,
        area_angstrom_squared=area,
        provenance="Synthetic generic-complex-spinor FFT unit oracle.",
    )
    q0_choice = make_vituri2024_translational_q0_reproduction_choice(
        evidence="Synthetic finite-domain algebra test retains the analytic q=0 term."
    )
    interaction = Vituri2024InteractionChoiceReceipt(
        gate_distance_angstrom=19.0,
        coulomb_e2_ev_angstrom=4.5,
        q0_evaluation="analytic_kernel_limit_only",
        provider_sha256="a" * 64,
        source_sha256=SM_TEX_SHA256,
        authority_kind="reproduction_choice",
        source_text="Synthetic test receipt using the pinned paper kernel formula.",
    )
    return Vituri2024TranslationalHFFFTFunctional(
        ordered_mesh=mesh,
        integer_mesh_labels=labels,
        delta_k_inverse_angstrom=0.071,
        active_band_states=np.asarray(states, dtype=np.complex128),
        h0_native=h0,
        normal_order_reference_native=reference,
        mesh_receipt=mesh_receipt,
        interaction=interaction,
        normal_order_reference_fingerprint=_test_array_sha256(reference),
        q0_choice=q0_choice,
        provenance="Independent generic-complex-spinor direct-formula test fixture.",
        fft_workers=1,
    )


def _direct_signed_convolution(
    kernel: np.ndarray,
    source: np.ndarray,
    *,
    swap_axes: bool = False,
    reverse_sign: bool = False,
    periodic: bool = False,
) -> np.ndarray:
    size = source.shape[0]
    offset = size - 1
    half = size // 2
    result = np.zeros_like(source)
    for my in range(size):
        for mx in range(size):
            for ry in range(size):
                for rx in range(size):
                    dy = my - ry
                    dx = mx - rx
                    if periodic:
                        dy = (dy + half) % size - half
                        dx = (dx + half) % size - half
                    if reverse_sign:
                        dy, dx = -dy, -dx
                    kernel_y, kernel_x = (dx, dy) if swap_axes else (dy, dx)
                    result[my, mx] += (
                        kernel[kernel_y + offset, kernel_x + offset]
                        * source[ry, rx]
                    )
    return result


def _direct_exchange_block(
    functional: Vituri2024TranslationalHFFFTFunctional,
    conventional_block: np.ndarray,
    left_flavor: int,
    right_flavor: int,
) -> np.ndarray:
    """Explicit m,r,c,d formula, independent of FFT and dense implementations."""

    size = functional.fft_plan.mesh_size
    kernel = functional.fft_plan.kernel_by_signed_displacement
    offset = size - 1
    # The pinned flavor order is (- valley spin pair, + valley spin pair).
    states_by_flavor = (
        functional.active_band_states[0],
        functional.active_band_states[0],
        functional.active_band_states[1],
        functional.active_band_states[1],
    )
    left = states_by_flavor[left_flavor].reshape(6, size, size)
    right = states_by_flavor[right_flavor].reshape(6, size, size)
    block = conventional_block.reshape(size, size)
    result = np.zeros((size, size), dtype=np.complex128)
    for my in range(size):
        for mx in range(size):
            for ry in range(size):
                for rx in range(size):
                    interaction = kernel[my - ry + offset, mx - rx + offset]
                    for left_orbital in range(6):
                        for right_orbital in range(6):
                            result[my, mx] -= (
                                interaction
                                * left[left_orbital, my, mx].conj()
                                * left[left_orbital, ry, rx]
                                * right[right_orbital, ry, rx].conj()
                                * right[right_orbital, my, mx]
                                * block[ry, rx]
                            )
    result /= functional.mesh_receipt.area_angstrom_squared
    return result


@pytest.mark.parametrize("mesh_size", [3, 5])
def test_every_site_matches_direct_asymmetric_complex_signed_convolution(
    mesh_size: int,
) -> None:
    labels, mesh = _centered_mesh(mesh_size)
    offset = mesh_size - 1
    kernel = np.empty((2 * mesh_size - 1, 2 * mesh_size - 1), dtype=np.complex128)
    for dy in range(-offset, offset + 1):
        for dx in range(-offset, offset + 1):
            kernel[dy + offset, dx + offset] = complex(
                0.17 + 1.31 * dy - 0.47 * dx + 0.09 * dy * dx,
                -0.23 + 0.37 * dy + 1.19 * dx + 0.05 * dy * dy,
            )
    source = np.empty((mesh_size, mesh_size), dtype=np.complex128)
    for iy in range(mesh_size):
        for ix in range(mesh_size):
            source[iy, ix] = complex(
                0.41 + 0.73 * iy - 0.29 * ix + 0.11 * iy * ix,
                -0.67 + 0.19 * iy + 0.83 * ix + 0.07 * ix * ix,
            )
    plan = make_vituri2024_square_cartesian_fft_plan(
        integer_mesh_labels=labels,
        ordered_mesh=mesh,
        delta_k_inverse_angstrom=0.125,
        kernel_by_signed_displacement=kernel,
        fft_workers=1,
    )
    actual = plan.convolve(source)
    expected = _direct_signed_convolution(kernel, source)
    _assert_scale_close(actual, expected, relative=8.0e-13)
    # Explicit canaries make x/y swaps, m-r sign reversal, and periodic boundary
    # wrapping observably different from the all-site direct result.
    assert kernel[offset, offset + 1] != kernel[offset + 1, offset]
    assert kernel[offset, offset + 1] != kernel[offset, offset - 1]
    for wrong in (
        _direct_signed_convolution(kernel, source, swap_axes=True),
        _direct_signed_convolution(kernel, source, reverse_sign=True),
        _direct_signed_convolution(kernel, source, periodic=True),
    ):
        assert np.max(np.abs(actual - wrong)) > 1.0e-3
    for iy, ix in (
        (0, 0),
        (0, mesh_size - 1),
        (mesh_size - 1, 0),
        (mesh_size - 1, mesh_size - 1),
    ):
        _assert_scale_close(actual[iy, ix], expected[iy, ix], relative=8.0e-13)


def test_scalar_corner_impulse_uses_exact_no_wrap_linear_convolution() -> None:
    size = 3
    labels, mesh = _centered_mesh(size)
    displacement = np.zeros((2 * size - 1, 2 * size - 1), dtype=np.complex128)
    offset = size - 1
    displacement[-2 + offset, -2 + offset] = 5.0
    # A size-N cyclic convolution would wrap this +x displacement to x=0.
    displacement[0 + offset, 1 + offset] = 3.0
    plan = make_vituri2024_square_cartesian_fft_plan(
        integer_mesh_labels=labels,
        ordered_mesh=mesh,
        delta_k_inverse_angstrom=0.125,
        kernel_by_signed_displacement=displacement,
        fft_workers=1,
    )
    source = np.zeros((size, size), dtype=np.complex128)
    source[-1, -1] = 1.0
    actual = plan.convolve(source)
    expected = np.zeros_like(source)
    expected[0, 0] = 5.0
    _assert_scale_close(actual, expected, relative=2.0e-13)
    assert plan.minimum_padding_size == 2 * size - 1
    assert plan.padding_size >= plan.minimum_padding_size
    assert plan.no_wrap_policy == VITURI2024_TRANSLATIONAL_HF_FFT_POLICY
    assert (plan.output_index_start, plan.output_index_stop) == (0, size)
    assert plan.domain_endpoints_included
    assert not plan.kernel_fft.flags.writeable

    workers_two = make_vituri2024_square_cartesian_fft_plan(
        integer_mesh_labels=labels,
        ordered_mesh=mesh,
        delta_k_inverse_angstrom=0.125,
        kernel_by_signed_displacement=displacement,
        fft_workers=2,
    )
    assert workers_two.fingerprint != plan.fingerprint


@pytest.mark.parametrize("mesh_size", [3, 5])
def test_all_16_hermitian_flavor_blocks_match_dense_actual_vituri_states(
    mesh_size: int,
) -> None:
    spec = Vituri2024CartesianHFSpec(mesh_size=mesh_size, holes_per_valley=1)
    dense = prepare_vituri2024_homogeneous_hf(spec).functional
    fft = prepare_vituri2024_homogeneous_hf_fft(spec, fft_workers=1).functional
    assert type(fft) is Vituri2024TranslationalHFFFTFunctional
    assert not hasattr(fft, "form_factors_by_flavor")
    assert not hasattr(fft, "kernel_by_mesh_pair")
    assert fft.fft_plan.axial_k_cutoff_inverse_angstrom == pytest.approx(
        spec.axial_k_cutoff_a0 / VITURI2024_PARAMETERS.a0
    )
    assert fft.fft_plan.corner_k_cutoff_inverse_angstrom == pytest.approx(
        spec.corner_k_cutoff_a0 / VITURI2024_PARAMETERS.a0
    )
    assert fft.implementation_fingerprint == fft.fft_plan.implementation_fingerprint
    assert all(
        len(digest) == 64
        for digest in (
            fft.implementation_fingerprint,
            fft.fft_plan.labels_sha256,
            fft.fft_plan.ordered_mesh_sha256,
            fft.fft_plan.signed_displacement_kernel_sha256,
            fft.fft_plan.kernel_embedding_sha256,
            fft.fft_plan.kernel_fft_sha256,
        )
    )
    density = _random_hermitian(spec.nk, seed=100 + mesh_size)
    # Random Hermitian flavor matrices have nonzero diagonal and all twelve
    # off-diagonal coherence blocks, including intervalley coherence.
    assert all(np.count_nonzero(density[a, b]) for a in range(4) for b in range(4))
    actual = fft.interaction_action_conventional(density)
    expected = dense.interaction_action_conventional(density)
    _assert_scale_close(actual, expected)
    _assert_scale_close(actual, actual.swapaxes(0, 1).conj(), relative=2.0e-12)


def test_generic_complex_spinors_match_explicit_selected_block_formula() -> None:
    functional = _generic_complex_spinor_functional()
    states = functional.active_band_states
    _assert_scale_close(np.sum(np.abs(states) ** 2, axis=1), 1.0, relative=2.0e-14)
    valley_overlap = np.sum(states[0].conj() * states[1], axis=0)
    assert np.max(np.abs(valley_overlap)) < 0.99

    rng = np.random.default_rng(921)
    native = np.zeros((4, 4, functional.nk), dtype=np.complex128)
    native[0, 0] = rng.standard_normal(functional.nk)
    native[3, 3] = rng.standard_normal(functional.nk)
    for left, right in ((0, 2), (1, 3)):
        profile = rng.standard_normal(functional.nk) + 1j * rng.standard_normal(
            functional.nk
        )
        native[left, right] = profile
        native[right, left] = profile.conj()
    conventional = np.asarray(native.swapaxes(0, 1), dtype=np.complex128)
    selected_blocks = ((0, 0), (3, 3), (0, 2), (2, 0), (1, 3), (3, 1))
    actual = functional.exchange_action_conventional(conventional)
    for left, right in selected_blocks:
        expected = _direct_exchange_block(
            functional, conventional[left, right], left, right
        )
        _assert_scale_close(actual[left, right].reshape(3, 3), expected)
    for left in range(4):
        for right in range(4):
            if (left, right) not in selected_blocks:
                assert np.count_nonzero(actual[left, right]) == 0


def test_native_complex_offdiagonal_density_transposes_to_conventional_output() -> None:
    functional = _generic_complex_spinor_functional(seed=818)
    rng = np.random.default_rng(922)
    native = np.zeros((4, 4, functional.nk), dtype=np.complex128)
    rho_02 = rng.standard_normal(functional.nk) + 1j * rng.standard_normal(
        functional.nk
    )
    native[0, 2] = rho_02
    native[2, 0] = rho_02.conj()

    actual = functional.interaction_action(native)[0, 2].reshape(3, 3)
    correct = _direct_exchange_block(functional, native[2, 0], 0, 2)
    wrong_untransposed = _direct_exchange_block(functional, native[0, 2], 0, 2)
    _assert_scale_close(actual, correct)
    assert np.max(np.abs(actual - wrong_untransposed)) > 1.0e-4


def test_hartree_is_norm_weighted_q0_term_and_isolated_from_exchange() -> None:
    fft = prepare_vituri2024_homogeneous_hf_fft(
        Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=1),
        fft_workers=1,
    ).functional
    density = _random_hermitian(fft.nk, seed=21)
    hartree = fft.hartree_action_conventional(density)
    exchange = fft.exchange_action_conventional(density)
    total = fft.interaction_action_conventional(density)
    diagonal = np.diagonal(density, axis1=0, axis2=1).T
    direct_scalar = np.sum(fft.state_norms_by_flavor * diagonal)
    q0 = fft.fft_plan.kernel_by_signed_displacement[
        fft.fft_plan.mesh_size - 1, fft.fft_plan.mesh_size - 1
    ]
    direct_scalar *= q0 / fft.mesh_receipt.area_angstrom_squared
    expected = np.zeros_like(density)
    for flavor in range(4):
        expected[flavor, flavor] = (
            fft.state_norms_by_flavor[flavor] * direct_scalar
        )
    _assert_scale_close(hartree, expected, relative=2.0e-13)
    _assert_scale_close(total, hartree + exchange)


def test_fft_action_linearity_hermiticity_and_self_adjoint_pairing() -> None:
    fft = prepare_vituri2024_homogeneous_hf_fft(
        Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=1),
        fft_workers=1,
    ).functional
    left = _random_hermitian(fft.nk, seed=31)
    right = _random_hermitian(fft.nk, seed=32)
    alpha = 0.37
    beta = -1.2
    sigma_left = fft.interaction_action_conventional(left)
    sigma_right = fft.interaction_action_conventional(right)
    sigma_combined = fft.interaction_action_conventional(alpha * left + beta * right)
    _assert_scale_close(sigma_combined, alpha * sigma_left + beta * sigma_right)
    _assert_scale_close(
        sigma_combined,
        sigma_combined.swapaxes(0, 1).conj(),
        relative=2.0e-12,
    )
    _assert_scale_close(
        _pair(left, sigma_right),
        _pair(sigma_left, right),
        relative=5.0e-11,
    )


def test_energy_fock_derivative_and_prepared_problem_match_dense() -> None:
    spec = Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=1)
    dense_prepared = prepare_vituri2024_homogeneous_hf(spec)
    fft_prepared = prepare_vituri2024_homogeneous_hf_fft(spec, fft_workers=1)
    assert np.array_equal(fft_prepared.ordered_mesh, dense_prepared.ordered_mesh)
    assert np.array_equal(
        fft_prepared.integer_mesh_labels, dense_prepared.integer_mesh_labels
    )
    assert np.array_equal(
        fft_prepared.active_band_states, dense_prepared.active_band_states
    )
    assert np.array_equal(
        fft_prepared.active_band_energies_by_valley,
        dense_prepared.active_band_energies_by_valley,
    )
    assert np.array_equal(fft_prepared.h0_native, dense_prepared.h0_native)
    assert fft_prepared.minimum_lower_gap_ev == dense_prepared.minimum_lower_gap_ev
    assert fft_prepared.minimum_upper_gap_ev == dense_prepared.minimum_upper_gap_ev

    density = _random_hermitian(spec.nk, seed=41)
    direction = _random_hermitian(spec.nk, seed=42)
    dense = dense_prepared.functional
    fft = fft_prepared.functional
    _assert_scale_close(fft.energy(density), dense.energy(density))
    _assert_scale_close(fft.fock(density), dense.fock(density))
    _assert_scale_close(
        fft.fock_derivative(density, direction),
        dense.fock_derivative(density, direction),
    )
    dense_action = make_vituri2024_hf_problem(
        dense_prepared
    ).kernel.interaction_builder
    fft_action = make_vituri2024_hf_problem(
        fft_prepared
    ).kernel.interaction_builder
    _assert_scale_close(fft_action(density), dense_action(density))


def test_independent_energy_fock_df_calculus_and_convention_canaries() -> None:
    functional = _generic_complex_spinor_functional(seed=819)
    density = _random_hermitian(functional.nk, seed=71, scale=0.13)
    direction = _random_hermitian(functional.nk, seed=72, scale=0.17)
    other_anchor = _random_hermitian(functional.nk, seed=73, scale=0.11)
    left = _random_hermitian(functional.nk, seed=74, scale=0.16)
    right = _random_hermitian(functional.nk, seed=75, scale=0.14)
    step = 2.0e-6

    fock = functional.fock(density)
    centered_d_energy = (
        functional.energy(density + step * direction)
        - functional.energy(density - step * direction)
    ) / (2.0 * step)
    native_pairing = _native_energy_pair(fock, direction)
    assert abs(native_pairing.imag) <= 2.0e-12 * max(1.0, abs(native_pairing))
    _assert_scale_close(centered_d_energy, native_pairing.real, relative=3.0e-9)

    derivative = functional.fock_derivative(density, direction)
    centered_d_fock = (
        functional.fock(density + step * direction)
        - functional.fock(density - step * direction)
    ) / (2.0 * step)
    _assert_scale_close(centered_d_fock, derivative, relative=3.0e-9)
    _assert_scale_close(
        derivative,
        functional.fock_derivative(other_anchor, direction),
        relative=2.0e-13,
    )

    derivative_left = functional.fock_derivative(density, left)
    derivative_right = functional.fock_derivative(density, right)
    _assert_scale_close(
        _native_energy_pair(derivative_left, right),
        _native_energy_pair(derivative_right, left),
        relative=3.0e-11,
    )

    wrong_transpose_pairing = _pair(fock, direction).real
    scale = max(1.0, abs(centered_d_energy), abs(native_pairing.real))
    assert abs(centered_d_energy - wrong_transpose_pairing) > 1.0e-5 * scale
    assert abs(centered_d_energy + native_pairing.real) > 1.0e-5 * scale
    assert np.max(np.abs(centered_d_fock + derivative)) > 1.0e-5


def test_plan_rejects_permuted_incomplete_and_non_square_labels() -> None:
    labels, mesh = _centered_mesh(3)
    kernel = np.ones((5, 5), dtype=np.complex128)

    permuted = labels.copy()
    permuted[[0, 1]] = permuted[[1, 0]]
    with pytest.raises(ValueError, match="iy-outer/ix-inner"):
        make_vituri2024_square_cartesian_fft_plan(
            integer_mesh_labels=permuted,
            ordered_mesh=mesh,
            delta_k_inverse_angstrom=0.125,
            kernel_by_signed_displacement=kernel,
            fft_workers=1,
        )

    with pytest.raises(ValueError, match="complete centered odd NxN square"):
        make_vituri2024_square_cartesian_fft_plan(
            integer_mesh_labels=labels[:-1],
            ordered_mesh=mesh[:-1],
            delta_k_inverse_angstrom=0.125,
            kernel_by_signed_displacement=kernel,
            fft_workers=1,
        )

    rectangular_labels = np.asarray(
        [(ix, iy) for iy in range(-1, 2) for ix in range(-2, 3)],
        dtype=np.int64,
    )
    rectangular_mesh = np.asarray(rectangular_labels, dtype=np.float64) * 0.125
    with pytest.raises(ValueError, match="complete centered odd NxN square"):
        make_vituri2024_square_cartesian_fft_plan(
            integer_mesh_labels=rectangular_labels,
            ordered_mesh=rectangular_mesh,
            delta_k_inverse_angstrom=0.125,
            kernel_by_signed_displacement=kernel,
            fft_workers=1,
        )

    wrong_mesh = mesh.copy()
    wrong_mesh[0, 0] = np.nextafter(wrong_mesh[0, 0], np.inf)
    with pytest.raises(ValueError, match="times delta_k exactly"):
        make_vituri2024_square_cartesian_fft_plan(
            integer_mesh_labels=labels,
            ordered_mesh=wrong_mesh,
            delta_k_inverse_angstrom=0.125,
            kernel_by_signed_displacement=kernel,
            fft_workers=1,
        )


@pytest.mark.parametrize("bad_workers", [0, -1, True, np.int64(1)])
def test_fft_workers_is_an_explicit_positive_builtin_int(bad_workers: object) -> None:
    labels, mesh = _centered_mesh(3)
    kernel = np.ones((5, 5), dtype=np.complex128)
    with pytest.raises((TypeError, ValueError), match="explicit positive int"):
        make_vituri2024_square_cartesian_fft_plan(
            integer_mesh_labels=labels,
            ordered_mesh=mesh,
            delta_k_inverse_angstrom=0.125,
            kernel_by_signed_displacement=kernel,
            fft_workers=bad_workers,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "binding_name",
    [
        "vituri2024_native_density_to_conventional_k_diagonal",
        "vituri2024_native_operator_to_conventional_k_diagonal",
        "vituri2024_vtf",
        "_FFT2",
        "_IFFT2",
        "_NEXT_FAST_LEN",
        "ACTIVE_BAND_STATES_VALLEY_ORDER",
        "INTERNAL_FLAVOR_ORDER",
    ],
)
def test_live_validation_rejects_dependency_monkeypatch_before_action_construction(
    monkeypatch: pytest.MonkeyPatch, binding_name: str
) -> None:
    functional = _generic_complex_spinor_functional(seed=823)
    original = getattr(fft_module, binding_name)
    if callable(original):
        def replacement(*args: object, **kwargs: object) -> object:
            return original(*args, **kwargs)
    else:
        replacement = tuple(reversed(original))
    monkeypatch.setattr(fft_module, binding_name, replacement)
    with pytest.raises(ValueError, match="implementation binding or source drifted"):
        functional.make_validated_interaction_action()


@pytest.mark.parametrize(
    "binding_name",
    [
        "vituri2024_native_operator_to_conventional_k_diagonal",
        "_FFT2",
        "_IFFT2",
        "INTERNAL_FLAVOR_ORDER",
    ],
)
def test_validated_action_rejects_postconstruction_runtime_drift(
    monkeypatch: pytest.MonkeyPatch, binding_name: str
) -> None:
    functional = _generic_complex_spinor_functional(seed=825)
    action = functional.make_validated_interaction_action()
    original = getattr(fft_module, binding_name)
    if callable(original):
        def replacement(*args: object, **kwargs: object) -> object:
            return original(*args, **kwargs)
    else:
        replacement = tuple(reversed(original))
    monkeypatch.setattr(fft_module, binding_name, replacement)
    density = np.zeros((4, 4, functional.nk), dtype=np.complex128)
    with pytest.raises(RuntimeError, match="runtime binding drifted"):
        action(density)


@pytest.mark.parametrize("version_owner", ["numpy", "scipy"])
def test_live_validation_fingerprint_binds_dependency_versions(
    monkeypatch: pytest.MonkeyPatch, version_owner: str
) -> None:
    functional = _generic_complex_spinor_functional(seed=824)
    module = fft_module.np if version_owner == "numpy" else fft_module.scipy
    monkeypatch.setattr(module, "__version__", "monkeypatched-version")
    with pytest.raises(ValueError, match="implementation binding or source drifted"):
        functional.make_validated_interaction_action()


def test_explicit_spacing_factory_realizes_spacing_and_derived_density() -> None:
    requested_delta_k_a0 = 0.037
    holes_per_valley = 3
    spec = make_vituri2024_cartesian_hf_spec_from_spacing(
        mesh_size=7,
        holes_per_valley=holes_per_valley,
        delta_k_a0=requested_delta_k_a0,
    )
    delta_k = requested_delta_k_a0 / VITURI2024_PARAMETERS.a0
    expected_area = (2.0 * math.pi / delta_k) ** 2
    expected_density = (
        2
        * holes_per_valley
        / expected_area
        / VITURI2024_CM2_TO_ANGSTROM2
    )
    assert spec.construction_mode == "explicit_spacing"
    assert spec.requested_delta_k_a0 == requested_delta_k_a0
    assert spec.total_hole_density_cm2 == expected_density
    assert spec.actual_total_hole_density_cm2 == pytest.approx(
        expected_density, rel=4.0e-16
    )
    assert spec.delta_k_inverse_angstrom * VITURI2024_PARAMETERS.a0 == pytest.approx(
        requested_delta_k_a0, rel=2.0e-15
    )
    mesh, labels = build_vituri2024_cartesian_mesh(spec)
    assert np.array_equal(
        mesh, np.asarray(labels, dtype=np.float64) * spec.delta_k_inverse_angstrom
    )
    assert spec.axial_k_cutoff_a0 == pytest.approx(
        (spec.mesh_size // 2) * requested_delta_k_a0, rel=2.0e-15
    )
    density_derived = Vituri2024CartesianHFSpec(
        mesh_size=spec.mesh_size,
        holes_per_valley=spec.holes_per_valley,
        total_hole_density_cm2=spec.total_hole_density_cm2,
    )
    assert density_derived.construction_mode == "density_derived"
    assert density_derived.requested_delta_k_a0 is None
    assert density_derived.delta_k_inverse_angstrom == spec.delta_k_inverse_angstrom
    assert density_derived.fingerprint != spec.fingerprint
    spec.validate_live_state()
    density_derived.validate_live_state()


def test_spacing_mode_live_validation_rejects_postconstruction_drift() -> None:
    spec = make_vituri2024_cartesian_hf_spec_from_spacing(
        mesh_size=7,
        holes_per_valley=3,
        delta_k_a0=0.037,
    )
    object.__setattr__(spec, "construction_mode", "density_derived")
    with pytest.raises(ValueError, match="live state drifted"):
        spec.validate_live_state()


@pytest.mark.parametrize(
    "construction_mode,requested_delta_k_a0,total_hole_density_cm2",
    [
        ("density_derived", 0.037, VITURI2024_TOTAL_HOLE_DENSITY_CM2),
        ("explicit_spacing", None, VITURI2024_TOTAL_HOLE_DENSITY_CM2),
        ("explicit_spacing", 0.037, VITURI2024_TOTAL_HOLE_DENSITY_CM2),
    ],
)
def test_manual_spacing_provenance_inconsistencies_are_rejected(
    construction_mode: str,
    requested_delta_k_a0: float | None,
    total_hole_density_cm2: float,
) -> None:
    with pytest.raises(ValueError, match="construction|spacing"):
        Vituri2024CartesianHFSpec(
            mesh_size=7,
            holes_per_valley=3,
            total_hole_density_cm2=total_hole_density_cm2,
            construction_mode=construction_mode,  # type: ignore[arg-type]
            requested_delta_k_a0=requested_delta_k_a0,
        )
