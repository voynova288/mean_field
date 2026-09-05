"""Independent reduced tests for the scalable signed integer scalar action."""

from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems import abc_trilayer as abc
from mean_field.systems.abc_trilayer.vituri2024_hf_fft import (
    make_vituri2024_square_cartesian_fft_plan,
)
from mean_field.systems.abc_trilayer import (
    vituri2024_tdhf_exact_integer_signed_scalar as signed_scalar_module,
)
from mean_field.systems.abc_trilayer.vituri2024_tdhf_exact_integer_functional import (
    Vituri2024ExactIntegerFunctionalKernel,
    vituri2024_exact_integer_projected_interaction_action,
)
from mean_field.systems.abc_trilayer.vituri2024_tdhf_exact_integer_signed_scalar import (
    VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_AUTHORITY,
    VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_API_VERSION,
    VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_AUTHORITY,
    Vituri2024ExactIntegerOrbitalScalarCurvatureReceipt,
    vituri2024_exact_integer_orbital_scalar_curvature,
    vituri2024_exact_integer_paired_interaction_trace,
    vituri2024_exact_integer_signed_scalar_fft_action,
    vituri2024_exact_integer_source_fock_fft,
    vituri2024_exact_integer_unitary_relative_energy,
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
    source = np.zeros((4, 4, plan.nk), dtype=np.complex128)
    with pytest.raises(ValueError, match="real-even kernel contract"):
        vituri2024_exact_integer_source_fock_fft(
            spinors, plan, area, source, source, source
        )


def test_signed_scalar_rejects_fft_plan_type_rebinding(monkeypatch) -> None:
    plan, spinors, area = _fixture()
    block = np.zeros((2, 2, plan.nk), dtype=np.complex128)

    class SubstitutePlan:
        def validate_live_state(self):
            return None

    monkeypatch.setattr(
        signed_scalar_module, "Vituri2024SquareCartesianFFTPlan", SubstitutePlan
    )
    with pytest.raises(RuntimeError, match="runtime binding drifted"):
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


def test_source_fock_matches_reduced_exact_integer_full_projector_oracle() -> None:
    plan, valley_spinors, area = _fixture()
    nk = plan.nk
    flavor_spinors = np.stack(
        (valley_spinors[0], valley_spinors[0], valley_spinors[1], valley_spinors[1])
    )
    rng = np.random.default_rng(20260906)
    h0 = np.empty((4, 4, nk), dtype=np.complex128)
    density = np.empty_like(h0)
    reference = np.empty_like(h0)
    for momentum in range(nk):
        raw = rng.normal(size=(4, 4)) + 1.0j * rng.normal(size=(4, 4))
        h0[:, :, momentum] = 0.5 * (raw + raw.conj().T)
        raw = rng.normal(size=(4, 4)) + 1.0j * rng.normal(size=(4, 4))
        density[:, :, momentum] = 0.5 * (raw + raw.conj().T)
        raw = rng.normal(size=(4, 4)) + 1.0j * rng.normal(size=(4, 4))
        reference[:, :, momentum] = 0.5 * (raw + raw.conj().T)
    actual = vituri2024_exact_integer_source_fock_fft(
        valley_spinors,
        plan,
        area,
        h0,
        density,
        reference,
    )
    form_factors = np.einsum(
        "fck,fcp->fkp", flavor_spinors.conj(), flavor_spinors, optimize=True
    )
    labels = plan.integer_mesh_labels
    offset = plan.mesh_size - 1
    kernel = np.empty((nk, nk), dtype=np.float64)
    for left in range(nk):
        for right in range(nk):
            dx, dy = labels[left] - labels[right]
            kernel[left, right] = plan.kernel_by_signed_displacement[
                dy + offset, dx + offset
            ].real
    dimension = 4 * nk
    difference = np.zeros((dimension, dimension), dtype=np.complex128)
    h0_full = np.zeros_like(difference)
    for momentum in range(nk):
        indices = np.arange(4) * nk + momentum
        difference[np.ix_(indices, indices)] = (
            density[:, :, momentum] - reference[:, :, momentum]
        )
        h0_full[np.ix_(indices, indices)] = h0[:, :, momentum]
    expected_full = h0_full + vituri2024_exact_integer_projected_interaction_action(
        difference,
        integer_mesh_labels=labels,
        form_factors_by_flavor=np.asarray(form_factors, dtype=np.complex128),
        interaction_kernel_by_mesh_pair=kernel,
        area_angstrom_squared=area,
    )
    expected = np.empty_like(actual)
    off_k_max = 0.0
    for left in range(nk):
        left_indices = np.arange(4) * nk + left
        expected[:, :, left] = expected_full[np.ix_(left_indices, left_indices)]
        for right in range(nk):
            if right == left:
                continue
            right_indices = np.arange(4) * nk + right
            off_k_max = max(
                off_k_max,
                float(np.max(np.abs(expected_full[np.ix_(left_indices, right_indices)]))),
            )
    assert off_k_max < 2.0e-13
    assert np.max(np.abs(actual - expected)) < 3.0e-13
    assert not actual.flags.writeable
    assert not actual.flags.owndata
    assert np.max(
        np.abs((actual - h0) - (expected - h0) / nk)
    ) > 1.0e-4


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
    assert abc.vituri2024_exact_integer_source_fock_fft is (
        vituri2024_exact_integer_source_fock_fft
    )
    assert abc.vituri2024_exact_integer_orbital_scalar_curvature is (
        vituri2024_exact_integer_orbital_scalar_curvature
    )
    assert abc.Vituri2024ExactIntegerOrbitalScalarCurvatureReceipt is (
        Vituri2024ExactIntegerOrbitalScalarCurvatureReceipt
    )


def test_unitary_relative_energy_matches_reduced_full_functional_and_curvature() -> None:
    from scipy.linalg import expm

    plan, valley_spinors, area = _fixture()
    nk = plan.nk
    labels = plan.integer_mesh_labels
    offset = plan.mesh_size - 1
    flavor_spinors = np.stack(
        (valley_spinors[0], valley_spinors[0], valley_spinors[1], valley_spinors[1])
    )
    form_factors = np.einsum(
        "fcm,fcp->fmp", flavor_spinors.conj(), flavor_spinors, optimize=True
    )
    kernel_pair = np.empty((nk, nk), dtype=np.float64)
    for m in range(nk):
        for p in range(nk):
            dx, dy = (labels[p] - labels[m]).tolist()
            kernel_pair[m, p] = plan.kernel_by_signed_displacement[
                dy + offset, dx + offset
            ].real
    dimension = 4 * nk
    h0 = np.zeros((dimension, dimension), dtype=np.complex128)
    for flavor in range(4):
        for k in range(nk):
            h0[flavor * nk + k, flavor * nk + k] = (
                -0.13 + 0.041 * flavor + 0.007 * k
            )
    reference = np.zeros_like(h0)
    selected_flavors = (1, 3)
    selected_occupations = np.ones((2, nk), dtype=np.bool_)
    base = int(np.flatnonzero(np.all(labels == (0, 0), axis=1))[0])
    target = int(np.flatnonzero(np.all(labels == (1, 0), axis=1))[0])
    selected_occupations[0, target] = False
    full_occupations = np.ones((4, nk), dtype=np.bool_)
    full_occupations[selected_flavors[0]] = selected_occupations[0]
    full_occupations[selected_flavors[1]] = selected_occupations[1]
    density0 = np.diag(full_occupations.reshape(dimension).astype(np.complex128))
    functional = Vituri2024ExactIntegerFunctionalKernel(
        integer_mesh_labels=labels,
        ordered_mesh=plan.ordered_mesh,
        form_factors_by_flavor=form_factors,
        interaction_kernel_by_mesh_pair=kernel_pair,
        h0_full=h0,
        normal_order_reference=reference,
        area_angstrom_squared=area,
        provenance="independent unitary relative-energy reduced oracle",
    )
    source_fock_full = functional.fock(density0)
    source_fock = np.empty((2, 2, nk), dtype=np.complex128)
    for left, flavor_left in enumerate(selected_flavors):
        for right, flavor_right in enumerate(selected_flavors):
            source_fock[left, right] = np.asarray(
                [
                    source_fock_full[flavor_left * nk + k, flavor_right * nk + k]
                    for k in range(nk)
                ]
            )
    particle_slots = np.asarray([0], dtype=np.int64)
    particle_k = np.asarray([target], dtype=np.int64)
    hole_slots = np.asarray([0], dtype=np.int64)
    hole_k = np.asarray([base], dtype=np.int64)
    amplitudes = np.asarray([0.37 - 0.21j], dtype=np.complex128)

    def relative(parameter: float):
        return vituri2024_exact_integer_unitary_relative_energy(
            source_fock,
            selected_occupations,
            particle_slots,
            particle_k,
            hole_slots,
            hole_k,
            amplitudes,
            parameter,
            valley_spinors,
            plan,
            area,
        )

    parameter = 0.037
    receipt = relative(parameter)
    generator = np.zeros((dimension, dimension), dtype=np.complex128)
    particle = selected_flavors[0] * nk + target
    hole = selected_flavors[0] * nk + base
    generator[particle, hole] = amplitudes[0]
    generator[hole, particle] = -amplitudes[0].conjugate()
    source_energy = functional.energy(density0)

    def full_relative(value: float) -> float:
        unitary = expm(value * generator)
        density_t = unitary @ density0 @ unitary.conj().T
        return functional.energy(density_t) - source_energy

    expected_relative = full_relative(parameter)
    assert receipt.relative_energy_ev == pytest.approx(expected_relative, abs=2.0e-13)
    assert receipt.maximum_unitarity_residual < 2.0e-15
    assert receipt.maximum_projector_residual < 2.0e-15
    assert receipt.maximum_hermiticity_residual < 2.0e-15
    assert receipt.external_source_fock_closure_established is False
    assert receipt.scalar_curvature_established is False
    zero = relative(0.0)
    assert zero.relative_energy_ev == 0.0
    assert zero.one_body_relative_energy_ev == 0.0
    assert zero.interaction_trace_ev == 0.0
    assert zero.signed_block_count == 0
    zero_direction = vituri2024_exact_integer_unitary_relative_energy(
        source_fock,
        selected_occupations,
        particle_slots,
        particle_k,
        hole_slots,
        hole_k,
        np.zeros(1, dtype=np.complex128),
        parameter,
        valley_spinors,
        plan,
        area,
    )
    assert zero_direction.active_orbital_count == 0
    assert zero_direction.connected_component_count == 0
    assert zero_direction.signed_block_count == 0
    assert zero_direction.relative_energy_ev == 0.0
    nonhermitian_fock = source_fock.copy()
    nonhermitian_fock[0, 1, 0] = 1.0
    with pytest.raises(ValueError, match="k-local Hermitian"):
        vituri2024_exact_integer_unitary_relative_energy(
            nonhermitian_fock,
            selected_occupations,
            particle_slots,
            particle_k,
            hole_slots,
            hole_k,
            amplitudes,
            parameter,
            valley_spinors,
            plan,
            area,
        )
    unnormalized_spinors = valley_spinors.copy()
    unnormalized_spinors[0, :, 0] *= 1.1
    with pytest.raises(ValueError, match="must be normalized"):
        vituri2024_exact_integer_unitary_relative_energy(
            source_fock,
            selected_occupations,
            particle_slots,
            particle_k,
            hole_slots,
            hole_k,
            amplitudes,
            parameter,
            unnormalized_spinors,
            plan,
            area,
        )

    positive = np.zeros((2, 2, nk), dtype=np.complex128)
    negative = np.zeros_like(positive)
    positive[0, 0, base] = amplitudes[0]
    negative[0, 0, target] = amplitudes[0].conjugate()
    interaction = vituri2024_exact_integer_paired_interaction_trace(
        valley_spinors, plan, area, (1, 0), 0, positive, negative
    )
    analytic = vituri2024_exact_integer_orbital_scalar_curvature(
        np.stack((source_fock[0, 0].real, source_fock[1, 1].real)),
        selected_occupations,
        particle_slots,
        particle_k,
        hole_slots,
        hole_k,
        amplitudes,
        valley_spinors,
        plan,
        area,
        displacement=(1, 0),
        valley_charge=0,
        positive_block=positive,
        negative_block=negative,
    )
    assert analytic.interaction_curvature_ev == pytest.approx(interaction, abs=2.0e-14)
    step = 3.0e-3
    candidate_values = {
        value: relative(value).relative_energy_ev
        for value in (-2.0 * step, -step, 0.0, step, 2.0 * step)
    }
    oracle_values = {
        value: full_relative(value)
        for value in (-2.0 * step, -step, 0.0, step, 2.0 * step)
    }
    for value in oracle_values:
        assert candidate_values[value] == pytest.approx(oracle_values[value], abs=2.0e-13)
    finite_difference = (
        -oracle_values[2.0 * step]
        + 16.0 * oracle_values[step]
        - 30.0 * oracle_values[0.0]
        + 16.0 * oracle_values[-step]
        - oracle_values[-2.0 * step]
    ) / (12.0 * step**2)
    assert finite_difference == pytest.approx(
        analytic.total_scalar_curvature_ev, rel=2.0e-8, abs=2.0e-10
    )


def test_unitary_relative_energy_multi_component_intervalley_dense_oracle() -> None:
    from scipy.linalg import expm

    plan, valley_spinors, area = _fixture()
    nk = plan.nk
    labels = plan.integer_mesh_labels
    offset = plan.mesh_size - 1
    flavor_spinors = np.stack(
        (valley_spinors[0], valley_spinors[0], valley_spinors[1], valley_spinors[1])
    )
    form_factors = np.einsum(
        "fcm,fcp->fmp", flavor_spinors.conj(), flavor_spinors, optimize=True
    )
    kernel_pair = np.empty((nk, nk), dtype=np.float64)
    for m in range(nk):
        for p in range(nk):
            dx, dy = (labels[p] - labels[m]).tolist()
            kernel_pair[m, p] = plan.kernel_by_signed_displacement[
                dy + offset, dx + offset
            ].real
    selected_flavors = (1, 3)
    selected_occupations = np.ones((2, nk), dtype=np.bool_)
    index = {
        tuple(label): int(position) for position, label in enumerate(labels.tolist())
    }
    center = index[(0, 0)]
    right, up, left, down = (
        index[(1, 0)], index[(0, 1)], index[(-1, 0)], index[(0, -1)]
    )
    particle_slots = np.asarray([0, 1, 1, 0], dtype=np.int64)
    particle_k = np.asarray([right, up, left, down], dtype=np.int64)
    hole_slots = np.asarray([0, 0, 1, 1], dtype=np.int64)
    hole_k = np.asarray([center, center, center, center], dtype=np.int64)
    amplitudes = np.asarray(
        [0.31 + 0.17j, -0.23 + 0.29j, 0.19 - 0.27j, -0.33 - 0.11j],
        dtype=np.complex128,
    )
    selected_occupations[particle_slots, particle_k] = False
    dimension = 4 * nk
    full_occupations = np.ones((4, nk), dtype=np.bool_)
    for slot, flavor in enumerate(selected_flavors):
        full_occupations[flavor] = selected_occupations[slot]
    density0 = np.diag(full_occupations.reshape(dimension).astype(np.complex128))
    h0 = np.diag(np.linspace(-0.21, 0.17, dimension, dtype=np.float64)).astype(
        np.complex128
    )
    functional = Vituri2024ExactIntegerFunctionalKernel(
        integer_mesh_labels=labels,
        ordered_mesh=plan.ordered_mesh,
        form_factors_by_flavor=form_factors,
        interaction_kernel_by_mesh_pair=kernel_pair,
        h0_full=h0,
        normal_order_reference=np.zeros_like(h0),
        area_angstrom_squared=area,
        provenance="multi-component intervalley exact-unitary oracle",
    )
    source_fock_full = functional.fock(density0)
    source_fock = np.empty((2, 2, nk), dtype=np.complex128)
    for left_slot, left_flavor in enumerate(selected_flavors):
        for right_slot, right_flavor in enumerate(selected_flavors):
            source_fock[left_slot, right_slot] = np.asarray(
                [
                    source_fock_full[left_flavor * nk + k, right_flavor * nk + k]
                    for k in range(nk)
                ]
            )
    generator = np.zeros((dimension, dimension), dtype=np.complex128)
    for ps, pk, hs, hk, value in zip(
        particle_slots, particle_k, hole_slots, hole_k, amplitudes, strict=True
    ):
        particle = selected_flavors[int(ps)] * nk + int(pk)
        hole = selected_flavors[int(hs)] * nk + int(hk)
        generator[particle, hole] = value
        generator[hole, particle] = -value.conjugate()
    source_energy = functional.energy(density0)

    def oracle(parameter: float) -> float:
        unitary = expm(parameter * generator)
        density = unitary @ density0 @ unitary.conj().T
        return functional.energy(density) - source_energy

    values = (-0.04, -0.02, 0.0, 0.02, 0.04)
    receipts = {}
    for parameter in values:
        receipt = vituri2024_exact_integer_unitary_relative_energy(
            source_fock,
            selected_occupations,
            particle_slots,
            particle_k,
            hole_slots,
            hole_k,
            amplitudes,
            parameter,
            valley_spinors,
            plan,
            area,
        )
        receipts[parameter] = receipt
        assert receipt.relative_energy_ev == pytest.approx(
            oracle(parameter), abs=3.0e-13
        )
        assert receipt.maximum_unitarity_residual < 3.0e-15
        assert receipt.maximum_projector_residual < 3.0e-15
        assert receipt.maximum_hermiticity_residual < 3.0e-15
        assert receipt.maximum_trace_residual < 3.0e-15
    assert receipts[0.02].connected_component_count == 2
    assert receipts[0.02].active_orbital_count == 6
    assert receipts[0.02].signed_block_count >= 7
    assert receipts[0.0].relative_energy_ev == 0.0
    assert receipts[0.0].signed_block_count == 0


def test_unitary_relative_energy_fails_closed_on_invalid_exponential(monkeypatch) -> None:
    plan, spinors, area = _fixture()
    nk = plan.nk
    labels = plan.integer_mesh_labels
    base = int(np.flatnonzero(np.all(labels == (0, 0), axis=1))[0])
    target = int(np.flatnonzero(np.all(labels == (1, 0), axis=1))[0])
    occupations = np.ones((2, nk), dtype=np.bool_)
    occupations[0, target] = False
    source_fock = np.zeros((2, 2, nk), dtype=np.complex128)
    original_fingerprint = signed_scalar_module._IMPORT_IMPLEMENTATION_FINGERPRINT

    def invalid_expm(generator):
        return 2.0 * np.eye(generator.shape[0], dtype=np.complex128)

    monkeypatch.setattr(signed_scalar_module, "_EXPM", invalid_expm)
    monkeypatch.setattr(signed_scalar_module, "_IMPORT_EXPM", invalid_expm)
    monkeypatch.setattr(
        signed_scalar_module,
        "_current_implementation_fingerprint",
        lambda: original_fingerprint,
    )
    with pytest.raises(ValueError, match="projector geometry failed"):
        vituri2024_exact_integer_unitary_relative_energy(
            source_fock,
            occupations,
            np.asarray([0], dtype=np.int64),
            np.asarray([target], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([base], dtype=np.int64),
            np.asarray([0.2 + 0.1j], dtype=np.complex128),
            0.03,
            spinors,
            plan,
            area,
        )
