"""Independent reduced tests for the scalable signed integer scalar action."""

from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems import abc_trilayer as abc
from mean_field.systems.abc_trilayer.vituri2024_hf_fft import (
    make_vituri2024_square_cartesian_fft_plan,
)
from mean_field.systems.abc_trilayer.vituri2024_tdhf_exact_integer_signed_scalar import (
    VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_AUTHORITY,
    VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_API_VERSION,
    VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_AUTHORITY,
    Vituri2024ExactIntegerOrbitalScalarCurvatureReceipt,
    vituri2024_exact_integer_orbital_scalar_curvature,
    vituri2024_exact_integer_paired_interaction_trace,
    vituri2024_exact_integer_signed_scalar_fft_action,
    vituri2024_exact_integer_signed_scalar_implementation_fingerprint,
)


def _fixture(size: int = 3, *, asymmetric_kernel: bool = False):
    half = size // 2
    labels = np.asarray(
        [(ix, iy) for iy in range(-half, half + 1) for ix in range(-half, half + 1)],
        dtype=np.int64,
    )
    delta = 0.1
    mesh = np.asarray(labels, dtype=np.float64) * delta
    signed = np.empty((2 * size - 1, 2 * size - 1), dtype=np.complex128)
    offset = size - 1
    for dy in range(-offset, offset + 1):
        for dx in range(-offset, offset + 1):
            value = 1.0 / (1.0 + delta * np.hypot(dx, dy))
            if asymmetric_kernel:
                value *= 1.0 + 0.05 * dx
            signed[dy + offset, dx + offset] = value
    plan = make_vituri2024_square_cartesian_fft_plan(
        integer_mesh_labels=labels,
        ordered_mesh=mesh,
        delta_k_inverse_angstrom=delta,
        kernel_by_signed_displacement=signed,
        fft_workers=1,
    )
    rng = np.random.default_rng(20260905 + size)
    spinors = rng.normal(size=(2, 6, len(labels))) + 1j * rng.normal(
        size=(2, 6, len(labels))
    )
    spinors /= np.linalg.norm(spinors, axis=1, keepdims=True)
    return plan, np.asarray(spinors, dtype=np.complex128), 7.3


def _allowed(charge: int):
    return {0: ((0, 0), (1, 1)), 2: ((1, 0),), -2: ((0, 1),)}[charge]


def _support(plan, displacement):
    dx, dy = displacement
    labels = plan.integer_mesh_labels
    half = plan.mesh_size // 2
    keep = (
        (labels[:, 0] + dx >= -half)
        & (labels[:, 0] + dx <= half)
        & (labels[:, 1] + dy >= -half)
        & (labels[:, 1] + dy <= half)
    )
    bases = np.flatnonzero(keep)
    targets = (
        (labels[keep, 1] + dy + half) * plan.mesh_size
        + labels[keep, 0]
        + dx
        + half
    )
    return bases, targets


def _dense_oracle(plan, spinors, area, displacement, charge, block):
    bases, targets = _support(plan, displacement)
    labels = plan.integer_mesh_labels
    size = plan.mesh_size
    offset = size - 1
    result = np.zeros_like(block)
    if charge == 0:
        dx, dy = displacement
        direct_kernel = plan.kernel_by_signed_displacement[-dy + offset, -dx + offset]
        source_charge = 0.0 + 0.0j
        for flavor in range(2):
            source_charge += np.sum(
                np.einsum(
                    "cp,cp->p",
                    spinors[flavor][:, bases].conj(),
                    spinors[flavor][:, targets],
                )
                * block[flavor, flavor, bases]
            )
        for flavor in range(2):
            result[flavor, flavor, bases] += (
                np.einsum(
                    "cp,cp->p",
                    spinors[flavor][:, targets].conj(),
                    spinors[flavor][:, bases],
                )
                * direct_kernel
                * source_charge
            )
    for left, right in _allowed(charge):
        for m_offset, (base_m, target_m) in enumerate(
            zip(bases, targets, strict=True)
        ):
            value = 0.0 + 0.0j
            for base_p, target_p in zip(bases, targets, strict=True):
                difference = labels[base_p] - labels[base_m]
                kernel = plan.kernel_by_signed_displacement[
                    difference[1] + offset, difference[0] + offset
                ]
                left_factor = np.vdot(
                    spinors[left, :, target_m], spinors[left, :, target_p]
                )
                right_factor = np.vdot(
                    spinors[right, :, base_p], spinors[right, :, base_m]
                )
                value += (
                    kernel
                    * left_factor
                    * right_factor
                    * block[left, right, base_p]
                )
            result[left, right, base_m] -= value
    result /= area
    return result


@pytest.mark.parametrize(
    ("displacement", "charge"),
    [((0, 0), 0), ((1, 0), 0), ((1, -1), 2), ((-1, 1), -2)],
)
def test_signed_scalar_fft_matches_independent_dense_oracle(displacement, charge) -> None:
    plan, spinors, area = _fixture()
    rng = np.random.default_rng(90 + charge + 3 * displacement[0] - displacement[1])
    block = np.zeros((2, 2, plan.nk), dtype=np.complex128)
    bases, _targets = _support(plan, displacement)
    for left, right in _allowed(charge):
        block[left, right, bases] = rng.normal(size=len(bases)) + 1j * rng.normal(
            size=len(bases)
        )
    expected = _dense_oracle(plan, spinors, area, displacement, charge, block)
    actual = vituri2024_exact_integer_signed_scalar_fft_action(
        spinors, plan, area, displacement, charge, block
    )
    assert np.max(np.abs(actual - expected)) < 2.0e-12
    assert not actual.flags.writeable


def _paired_blocks(plan, displacement, charge, seed):
    rng = np.random.default_rng(seed)
    positive = np.zeros((2, 2, plan.nk), dtype=np.complex128)
    negative = np.zeros_like(positive)
    bases, targets = _support(plan, displacement)
    for left, right in _allowed(charge):
        values = rng.normal(size=len(bases)) + 1j * rng.normal(size=len(bases))
        positive[left, right, bases] = values
        negative[right, left, targets] = values.conj()
    return positive, negative


def test_paired_interaction_trace_matches_explicit_two_sign_pairing() -> None:
    plan, spinors, area = _fixture()
    displacement = (1, -1)
    charge = 2
    positive, negative = _paired_blocks(plan, displacement, charge, 101)
    actual = vituri2024_exact_integer_paired_interaction_trace(
        spinors, plan, area, displacement, charge, positive, negative
    )
    positive_action = _dense_oracle(
        plan, spinors, area, displacement, charge, positive
    )
    negative_action = _dense_oracle(
        plan, spinors, area, (-1, 1), -charge, negative
    )
    expected = np.vdot(positive, positive_action) + np.vdot(
        negative, negative_action
    )
    dimension = 2 * plan.nk
    dense_w = np.zeros((dimension, dimension), dtype=np.complex128)
    dense_sigma = np.zeros_like(dense_w)
    for signed_displacement, signed_charge, values, response in (
        (displacement, charge, positive, positive_action),
        ((-1, 1), -charge, negative, negative_action),
    ):
        bases, targets = _support(plan, signed_displacement)
        for left, right in _allowed(signed_charge):
            dense_w[left * plan.nk + targets, right * plan.nk + bases] = values[
                left, right, bases
            ]
            dense_sigma[
                left * plan.nk + targets, right * plan.nk + bases
            ] = response[left, right, bases]
    explicit_trace = np.trace(dense_w @ dense_sigma)
    assert np.max(np.abs(dense_w - dense_w.conj().T)) == 0.0
    assert abs(expected.imag) < 2.0e-12
    assert expected == pytest.approx(explicit_trace, abs=2.0e-12)
    assert actual == pytest.approx(explicit_trace.real, abs=2.0e-12)
    assert abs(actual - 2.0 * explicit_trace.real) > 1.0e-6


def test_signed_scalar_rejects_wrap_disallowed_blocks_and_nonhermitian_pair() -> None:
    plan, spinors, area = _fixture()
    block = np.zeros((2, 2, plan.nk), dtype=np.complex128)
    bases, _ = _support(plan, (1, 0))
    block[1, 0, bases] = 1.0
    with pytest.raises(ValueError, match="valley-charge block"):
        vituri2024_exact_integer_signed_scalar_fft_action(
            spinors, plan, area, (1, 0), 0, block
        )
    block = np.zeros((2, 2, plan.nk), dtype=np.complex128)
    unsupported = np.setdiff1d(np.arange(plan.nk), bases)
    block[0, 0, unsupported[0]] = 1.0
    with pytest.raises(ValueError, match="outside no-wrap support"):
        vituri2024_exact_integer_signed_scalar_fft_action(
            spinors, plan, area, (1, 0), 0, block
        )
    positive, negative = _paired_blocks(plan, (1, 0), 2, 102)
    negative = negative.copy()
    negative[0, 1] *= 2.0
    with pytest.raises(ValueError, match="Hermitian pair"):
        vituri2024_exact_integer_paired_interaction_trace(
            spinors, plan, area, (1, 0), 2, positive, negative
        )
    zero = np.zeros((2, 2, plan.nk), dtype=np.complex128)
    with pytest.raises(ValueError, match="self-conjugate zero sector"):
        vituri2024_exact_integer_paired_interaction_trace(
            spinors, plan, area, (0, 0), 0, zero, zero
        )


def test_signed_scalar_requires_the_reviewed_real_even_kernel_orientation() -> None:
    plan, spinors, area = _fixture(asymmetric_kernel=True)
    block = np.zeros((2, 2, plan.nk), dtype=np.complex128)
    block[1, 0] = 0.0
    with pytest.raises(ValueError, match="real-even kernel contract"):
        vituri2024_exact_integer_signed_scalar_fft_action(
            spinors, plan, area, (1, 0), 2, block
        )


def test_signed_scalar_size9_execution_keeps_finite_no_wrap_shape() -> None:
    plan, spinors, area = _fixture(size=9)
    block = np.zeros((2, 2, plan.nk), dtype=np.complex128)
    bases, _targets = _support(plan, (3, -2))
    block[1, 0, bases] = 1.0 + 0.25j
    result = vituri2024_exact_integer_signed_scalar_fft_action(
        spinors, plan, area, (3, -2), 2, block
    )
    assert result.shape == block.shape
    assert np.all(np.isfinite(result))
    assert np.count_nonzero(result[:, :, np.setdiff1d(np.arange(plan.nk), bases)]) == 0


def test_orbital_scalar_curvature_reconstructs_explicit_double_commutator() -> None:
    plan, spinors, area = _fixture()
    nk = plan.nk
    source_fock = np.stack(
        (
            np.linspace(-0.7, 0.2, nk, dtype=np.float64),
            np.linspace(-0.1, 0.8, nk, dtype=np.float64),
        )
    )
    occupations = np.zeros((2, nk), dtype=np.bool_)
    occupations[0, 0] = True
    occupations[1, 1] = True
    particle_slots = np.asarray([0, 1], dtype=np.int64)
    particle_k = np.asarray([1, 0], dtype=np.int64)
    hole_slots = np.asarray([0, 1], dtype=np.int64)
    hole_k = np.asarray([0, 1], dtype=np.int64)
    amplitudes = np.asarray([0.3 + 0.2j, -0.1 + 0.4j], dtype=np.complex128)
    positive = np.zeros((2, 2, nk), dtype=np.complex128)
    negative = np.zeros_like(positive)
    positive[0, 0, 0] = amplitudes[0]
    negative[0, 0, 1] = amplitudes[0].conjugate()
    negative[1, 1, 1] = amplitudes[1]
    positive[1, 1, 0] = amplitudes[1].conjugate()
    receipt = vituri2024_exact_integer_orbital_scalar_curvature(
        source_fock,
        occupations,
        particle_slots,
        particle_k,
        hole_slots,
        hole_k,
        amplitudes,
        spinors,
        plan,
        area,
        (1, 0),
        0,
        positive,
        negative,
    )
    dimension = 2 * nk
    projector = np.diag(occupations.reshape(-1).astype(np.complex128))
    generator = np.zeros((dimension, dimension), dtype=np.complex128)
    for ps, pk, hs, hk, value in zip(
        particle_slots,
        particle_k,
        hole_slots,
        hole_k,
        amplitudes,
        strict=True,
    ):
        particle = int(ps) * nk + int(pk)
        hole = int(hs) * nk + int(hk)
        generator[particle, hole] = value
        generator[hole, particle] = -value.conjugate()
    tangent = generator @ projector - projector @ generator
    second = generator @ tangent - tangent @ generator
    explicit_one_body = np.trace(np.diag(source_fock.reshape(-1)) @ second)
    assert type(receipt) is Vituri2024ExactIntegerOrbitalScalarCurvatureReceipt
    assert abs(explicit_one_body.imag) < 1.0e-15
    assert receipt.one_body_curvature_ev == pytest.approx(
        explicit_one_body.real, abs=1.0e-15
    )
    assert receipt.total_scalar_curvature_ev == pytest.approx(
        explicit_one_body.real + receipt.interaction_curvature_ev, abs=1.0e-15
    )
    assert receipt.raw_total_no_nk_normalization
    assert receipt.authority == VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_AUTHORITY
    assert receipt.implementation_fingerprint == (
        vituri2024_exact_integer_signed_scalar_implementation_fingerprint()
    )
    assert not receipt.source_functional_fock_closure_established
    assert not receipt.full_exact_unitary_scalar_curvature_established


def test_orbital_scalar_curvature_rejects_bad_occupation_and_duplicates() -> None:
    plan, spinors, area = _fixture()
    source_fock = np.zeros((2, plan.nk), dtype=np.float64)
    occupations = np.zeros((2, plan.nk), dtype=np.bool_)
    occupations[0, 0] = True
    occupations[1, 1] = True
    particle_slots = np.asarray([0], dtype=np.int64)
    particle_k = np.asarray([0], dtype=np.int64)
    hole_slots = np.asarray([1], dtype=np.int64)
    hole_k = np.asarray([1], dtype=np.int64)
    amplitudes = np.ones(1, dtype=np.complex128)
    positive = np.zeros((2, 2, plan.nk), dtype=np.complex128)
    negative = np.zeros_like(positive)
    with pytest.raises(ValueError, match="occupied holes to virtual particles"):
        vituri2024_exact_integer_orbital_scalar_curvature(
            source_fock,
            occupations,
            particle_slots,
            particle_k,
            hole_slots,
            hole_k,
            amplitudes,
            spinors,
            plan,
            area,
            (1, 0),
            0,
            positive,
            negative,
        )
    with pytest.raises(ValueError, match="duplicate particle-hole transition"):
        vituri2024_exact_integer_orbital_scalar_curvature(
            source_fock,
            occupations,
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([1, 1], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int64),
            np.ones(2, dtype=np.complex128),
            spinors,
            plan,
            area,
            (1, 0),
            0,
            positive,
            negative,
        )
    bad_positive = positive.copy()
    bad_positive[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="signed blocks do not match"):
        vituri2024_exact_integer_orbital_scalar_curvature(
            source_fock,
            occupations,
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            amplitudes,
            spinors,
            plan,
            area,
            (1, 0),
            0,
            bad_positive,
            negative,
        )
    with pytest.raises(TypeError, match="strict real scalar"):
        vituri2024_exact_integer_orbital_scalar_curvature(
            source_fock,
            occupations,
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            amplitudes,
            spinors,
            plan,
            "1.0",
            (1, 0),
            0,
            positive,
            negative,
        )


def test_signed_scalar_api_is_public_and_candidate_only() -> None:
    assert VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_API_VERSION.endswith(".v1")
    assert "candidate" in VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_AUTHORITY
    assert "not_scalar_hessian" in VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_AUTHORITY
    assert "candidate" in VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_AUTHORITY
    assert len(vituri2024_exact_integer_signed_scalar_implementation_fingerprint()) == 64
    assert abc.vituri2024_exact_integer_signed_scalar_fft_action is (
        vituri2024_exact_integer_signed_scalar_fft_action
    )
    assert abc.vituri2024_exact_integer_orbital_scalar_curvature is (
        vituri2024_exact_integer_orbital_scalar_curvature
    )
    assert abc.Vituri2024ExactIntegerOrbitalScalarCurvatureReceipt is (
        Vituri2024ExactIntegerOrbitalScalarCurvatureReceipt
    )
