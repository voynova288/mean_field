"""Reduced qualification for the distinct exact-integer Vituri scalar kernel."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from mean_field.systems import abc_trilayer as abc
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_full_response import (
    compare_vituri2024_hf_spiral_literal_mask_equivalence,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_preflight import (
    INTERNAL_FLAVOR_ORDER,
)
from mean_field.systems.abc_trilayer.vituri2024_tdhf_exact_integer_functional import (
    VITURI2024_EXACT_INTEGER_FUNCTIONAL_API_VERSION,
    VITURI2024_EXACT_INTEGER_FUNCTIONAL_AUTHORITY,
    VITURI2024_EXACT_INTEGER_FUNCTIONAL_CONVENTION,
    VITURI2024_EXACT_INTEGER_FUNCTIONAL_MAX_NK,
    Vituri2024ExactIntegerFunctionalKernel,
    vituri2024_exact_integer_projected_interaction_action,
)
from mean_field.systems.abc_trilayer.vituri2024_tdhf_full_functional import (
    _exact_local_mask,
    vituri2024_full_projected_interaction_action,
)


def _hermitian(rng: np.random.Generator, dimension: int) -> np.ndarray:
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    return np.asarray(0.5 * (raw + raw.conj().T), dtype=np.complex128)


def _inputs():
    size = 3
    half = size // 2
    labels = np.asarray(
        [(ix, iy) for iy in range(-half, half + 1) for ix in range(-half, half + 1)],
        dtype=np.int64,
    )
    mesh = np.asarray(labels, dtype=np.float64) * 0.125
    nk = len(labels)
    flavors = len(INTERNAL_FLAVOR_ORDER)
    rng = np.random.default_rng(20260904)
    states = rng.normal(size=(flavors, 5, nk)) + 1j * rng.normal(
        size=(flavors, 5, nk)
    )
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    form_factors = np.einsum(
        "fcm,fcn->fmn", states.conj(), states, optimize=False
    ).astype(np.complex128)
    transfer = mesh[:, None, :] - mesh[None, :, :]
    kernel = np.asarray(
        1.0 / (1.0 + np.linalg.norm(transfer, axis=-1)), dtype=np.float64
    )
    dimension = flavors * nk
    h0 = _hermitian(rng, dimension)
    reference = np.zeros((dimension, dimension), dtype=np.complex128)
    return labels, mesh, form_factors, kernel, h0, reference, 7.3


def _explicit_integer_action(
    density: np.ndarray,
    labels: np.ndarray,
    form_factors: np.ndarray,
    kernel: np.ndarray,
    area: float,
) -> np.ndarray:
    flavors, nk, _ = form_factors.shape
    x = density.reshape(flavors, nk, flavors, nk)
    result = np.zeros_like(x)
    for m in range(nk):
        for n in range(nk):
            for p in range(nk):
                for r in range(nk):
                    if np.any(labels[m] + labels[p] - labels[r] - labels[n] != 0):
                        continue
                    charge = sum(
                        form_factors[flavor, p, r] * x[flavor, r, flavor, p]
                        for flavor in range(flavors)
                    )
                    for flavor in range(flavors):
                        result[flavor, m, flavor, n] += (
                            form_factors[flavor, m, n] * kernel[p, r] * charge
                        )
                    for left in range(flavors):
                        for right in range(flavors):
                            result[left, m, right, n] -= (
                                kernel[p, n]
                                * form_factors[left, m, r]
                                * form_factors[right, p, n]
                                * x[left, r, right, p]
                            )
    return result.reshape(density.shape) / area


def _kernel(*, reference_scale: float = 0.0) -> Vituri2024ExactIntegerFunctionalKernel:
    labels, mesh, form_factors, interaction, h0, reference, area = _inputs()
    if reference_scale:
        reference = np.eye(h0.shape[0], dtype=np.complex128) * reference_scale
    return Vituri2024ExactIntegerFunctionalKernel(
        integer_mesh_labels=labels,
        ordered_mesh=mesh,
        form_factors_by_flavor=form_factors,
        interaction_kernel_by_mesh_pair=interaction,
        h0_full=h0,
        normal_order_reference=reference,
        area_angstrom_squared=area,
        provenance="Synthetic reduced exact-integer qualification fixture.",
    )


def test_exact_integer_action_matches_independent_streaming_quartet_oracle() -> None:
    kernel = _kernel()
    rng = np.random.default_rng(81)
    direction = _hermitian(rng, kernel.dimension)
    expected = _explicit_integer_action(
        direction,
        kernel.integer_mesh_labels,
        kernel.form_factors_by_flavor,
        kernel.interaction_kernel_by_mesh_pair,
        kernel.area_angstrom_squared,
    )
    actual = kernel.interaction_action(direction)
    assert np.max(np.abs(actual - expected)) < 2.0e-12
    assert np.max(np.abs(actual - actual.conj().T)) < 2.0e-12


def test_exact_integer_action_matches_literal_action_only_on_qualified_n3_mesh() -> None:
    kernel = _kernel()
    rng = np.random.default_rng(82)
    direction = _hermitian(rng, kernel.dimension)
    literal = vituri2024_full_projected_interaction_action(
        direction,
        form_factors_by_flavor=kernel.form_factors_by_flavor,
        interaction_kernel_by_mesh_pair=kernel.interaction_kernel_by_mesh_pair,
        exact_local_mask=_exact_local_mask(kernel.ordered_mesh),
        area_angstrom_squared=kernel.area_angstrom_squared,
    )
    assert np.max(np.abs(kernel.interaction_action(direction) - literal)) < 2.0e-12


def test_exact_integer_energy_fock_and_df_are_one_quadratic_functional() -> None:
    kernel = _kernel()
    rng = np.random.default_rng(83)
    density = _hermitian(rng, kernel.dimension)
    direction = _hermitian(rng, kernel.dimension)
    other = _hermitian(rng, kernel.dimension)
    fock = kernel.fock(density)
    df = kernel.differential_fock(direction)
    epsilon = 2.0e-6
    finite_energy_gradient = (
        kernel.energy(density + epsilon * direction)
        - kernel.energy(density - epsilon * direction)
    ) / (2.0 * epsilon)
    expected_gradient = float(np.trace(fock @ direction).real)
    assert finite_energy_gradient == pytest.approx(expected_gradient, abs=2.0e-7)
    finite_df = (
        kernel.fock(density + epsilon * direction)
        - kernel.fock(density - epsilon * direction)
    ) / (2.0 * epsilon)
    assert np.max(np.abs(finite_df - df)) < 2.0e-9

    left = np.trace(direction @ kernel.differential_fock(other))
    right = np.trace(kernel.differential_fock(direction) @ other)
    assert left == pytest.approx(right, abs=2.0e-10)
    assert abs(left.imag) < 2.0e-10

    step = 0.013
    expected_second = float(np.trace(direction @ df).real)
    exact_quadratic_prediction = (
        kernel.energy(density)
        + step * expected_gradient
        + 0.5 * step**2 * expected_second
    )
    assert kernel.energy(density + step * direction) == pytest.approx(
        exact_quadratic_prediction, abs=2.0e-10
    )


def test_nonzero_complex_reference_enters_energy_and_fock_consistently() -> None:
    labels, mesh, form_factors, interaction, h0, _reference, area = _inputs()
    rng = np.random.default_rng(841)
    factor = rng.normal(size=(h0.shape[0], 4)) + 1j * rng.normal(
        size=(h0.shape[0], 4)
    )
    reference = factor @ factor.conj().T
    reference *= 0.5 / np.linalg.eigvalsh(reference)[-1]
    reference = np.asarray(reference, dtype=np.complex128)
    kernel = Vituri2024ExactIntegerFunctionalKernel(
        integer_mesh_labels=labels,
        ordered_mesh=mesh,
        form_factors_by_flavor=form_factors,
        interaction_kernel_by_mesh_pair=interaction,
        h0_full=h0,
        normal_order_reference=reference,
        area_angstrom_squared=area,
        provenance="Complex PSD normal-order reference test.",
    )
    assert kernel.energy(reference) == pytest.approx(
        float(np.trace(h0 @ reference).real), abs=2.0e-12
    )
    assert np.max(np.abs(kernel.fock(reference) - h0)) < 2.0e-12
    density = _hermitian(rng, kernel.dimension)
    direction = _hermitian(rng, kernel.dimension)
    epsilon = 1.0e-6
    finite_gradient = (
        kernel.energy(density + epsilon * direction)
        - kernel.energy(density - epsilon * direction)
    ) / (2.0 * epsilon)
    assert finite_gradient == pytest.approx(
        float(np.trace(kernel.fock(density) @ direction).real), abs=3.0e-7
    )
    assert np.max(
        np.abs(
            kernel.fock(density)
            - kernel.h0_full
            - kernel.interaction_action(density - kernel.normal_order_reference)
        )
    ) < 2.0e-12


def test_exact_integer_analytic_curvature_matches_exact_unitary_energy() -> None:
    kernel = _kernel()
    rng = np.random.default_rng(84)
    occupied = np.linalg.qr(
        rng.normal(size=(kernel.dimension, 7))
        + 1j * rng.normal(size=(kernel.dimension, 7))
    )[0]
    projector = np.asarray(occupied @ occupied.conj().T, dtype=np.complex128)
    raw = rng.normal(size=(kernel.dimension, kernel.dimension)) + 1j * rng.normal(
        size=(kernel.dimension, kernel.dimension)
    )
    generator = np.asarray(raw - raw.conj().T, dtype=np.complex128)
    generator /= np.linalg.norm(generator)
    first = generator @ projector - projector @ generator
    second = generator @ first - first @ generator
    analytic = float(
        (
            np.trace(kernel.fock(projector) @ second)
            + np.trace(first @ kernel.differential_fock(first))
        ).real
    )

    step = 8.0e-4
    energies = {}
    for multiplier in (-2, -1, 0, 1, 2):
        unitary = expm(multiplier * step * generator)
        displaced = np.asarray(
            unitary @ projector @ unitary.conj().T, dtype=np.complex128
        )
        energies[multiplier] = kernel.energy(displaced)
    finite = (
        -energies[2]
        + 16.0 * energies[1]
        - 30.0 * energies[0]
        + 16.0 * energies[-1]
        - energies[-2]
    ) / (12.0 * step**2)
    assert finite == pytest.approx(analytic, rel=2.0e-7, abs=2.0e-7)
    assert abs(finite - 0.5 * analytic) > 1.0e-3
    assert abs(finite - analytic / kernel.nk) > 1.0e-3


def test_exact_integer_kernel_is_snapshot_bound_and_fail_closed() -> None:
    inputs = list(_inputs())
    labels = inputs[0]
    mesh = inputs[1]
    kernel = Vituri2024ExactIntegerFunctionalKernel(
        integer_mesh_labels=labels,
        ordered_mesh=mesh,
        form_factors_by_flavor=inputs[2],
        interaction_kernel_by_mesh_pair=inputs[3],
        h0_full=inputs[4],
        normal_order_reference=inputs[5],
        area_angstrom_squared=inputs[6],
        provenance="Snapshot mutation test.",
    )
    fingerprint = kernel.kernel_fingerprint
    labels[0, 0] = 999
    mesh[0, 0] = 999.0
    assert kernel.kernel_fingerprint == fingerprint
    assert kernel.integer_mesh_labels[0, 0] == -1
    assert kernel.ordered_mesh[0, 0] == pytest.approx(-0.125)
    assert not kernel.integer_mesh_labels.flags.writeable
    assert not kernel.ordered_mesh.flags.writeable
    assert kernel.exact_integer_conservation
    assert kernel.no_wrap
    assert kernel.raw_finite_square_total
    assert not kernel.literal_float_mask_used
    assert not kernel.source_closure_established
    assert not kernel.scalar_hessian_authority_established
    assert not kernel.reciprocity_established
    assert not kernel.production_ready
    kernel.validate_live_state()

    object.__setattr__(kernel, "production_ready", True)
    with pytest.raises(ValueError, match="authority contract drifted"):
        kernel.validate_live_state()

    writable = _kernel()
    object.__setattr__(writable, "h0_full", np.asarray(writable.h0_full).copy())
    with pytest.raises(ValueError, match="bytes-backed readonly"):
        writable.validate_live_state()

    owning_readonly = _kernel()
    replacement_readonly = np.asarray(owning_readonly.h0_full).copy()
    replacement_readonly.setflags(write=False)
    object.__setattr__(owning_readonly, "h0_full", replacement_readonly)
    with pytest.raises(ValueError, match="bytes-backed readonly"):
        owning_readonly.validate_live_state()

    dimension_drift = _kernel()
    object.__setattr__(dimension_drift, "dimension", dimension_drift.dimension + 1)
    with pytest.raises(ValueError, match="derived dimensions drifted"):
        dimension_drift.validate_live_state()

    dimension_type_drift = _kernel()
    object.__setattr__(
        dimension_type_drift, "dimension", float(dimension_type_drift.dimension)
    )
    with pytest.raises(ValueError, match="derived dimensions drifted"):
        dimension_type_drift.validate_live_state()

    method_drift = _kernel()
    original_energy = Vituri2024ExactIntegerFunctionalKernel.energy
    try:
        Vituri2024ExactIntegerFunctionalKernel.energy = lambda self, density: 0.0
        with pytest.raises(ValueError, match="implementation fingerprint drifted"):
            method_drift.validate_live_state()
    finally:
        Vituri2024ExactIntegerFunctionalKernel.energy = original_energy

    property_drift = _kernel()
    original_property = Vituri2024ExactIntegerFunctionalKernel.kernel_fingerprint
    try:
        Vituri2024ExactIntegerFunctionalKernel.kernel_fingerprint = property(
            lambda self: self._kernel_fingerprint
        )
        with pytest.raises(ValueError, match="implementation fingerprint drifted"):
            property_drift.validate_live_state()
    finally:
        Vituri2024ExactIntegerFunctionalKernel.kernel_fingerprint = original_property

    stale = _kernel()
    replacement = np.asarray(stale.h0_full).copy()
    replacement[0, 0] += 1.0
    replacement = np.frombuffer(
        replacement.tobytes(order="C"), dtype=np.complex128
    ).reshape(replacement.shape)
    replacement.setflags(write=False)
    object.__setattr__(stale, "h0_full", replacement)
    with pytest.raises(ValueError, match="component fingerprint drifted"):
        stale.validate_live_state()


def test_exact_integer_kernel_rejects_invalid_mesh_and_reference_inputs() -> None:
    labels, mesh, form_factors, interaction, h0, reference, area = _inputs()
    permuted = labels.copy()
    permuted[[0, 1]] = permuted[[1, 0]]
    with pytest.raises(ValueError, match="centered"):
        Vituri2024ExactIntegerFunctionalKernel(
            integer_mesh_labels=permuted,
            ordered_mesh=mesh,
            form_factors_by_flavor=form_factors,
            interaction_kernel_by_mesh_pair=interaction,
            h0_full=h0,
            normal_order_reference=reference,
            area_angstrom_squared=area,
            provenance="Rejected permuted labels.",
        )
    distorted_mesh = mesh.copy()
    distorted_mesh[0, 0] += 1.0e-8
    with pytest.raises(ValueError, match="labels times one spacing"):
        Vituri2024ExactIntegerFunctionalKernel(
            integer_mesh_labels=labels,
            ordered_mesh=distorted_mesh,
            form_factors_by_flavor=form_factors,
            interaction_kernel_by_mesh_pair=interaction,
            h0_full=h0,
            normal_order_reference=reference,
            area_angstrom_squared=area,
            provenance="Rejected distorted mesh.",
        )
    even_labels = np.asarray(
        [(-1, -1), (0, -1), (-1, 0), (0, 0)], dtype=np.int64
    )
    with pytest.raises(ValueError, match="odd square mesh"):
        Vituri2024ExactIntegerFunctionalKernel(
            integer_mesh_labels=even_labels,
            ordered_mesh=mesh,
            form_factors_by_flavor=form_factors,
            interaction_kernel_by_mesh_pair=interaction,
            h0_full=h0,
            normal_order_reference=reference,
            area_angstrom_squared=area,
            provenance="Rejected even mesh.",
        )
    large_half = 6
    over_cap_labels = np.asarray(
        [
            (ix, iy)
            for iy in range(-large_half, large_half + 1)
            for ix in range(-large_half, large_half + 1)
        ],
        dtype=np.int64,
    )
    with pytest.raises(ValueError, match="reduced-mesh cap"):
        Vituri2024ExactIntegerFunctionalKernel(
            integer_mesh_labels=over_cap_labels,
            ordered_mesh=mesh,
            form_factors_by_flavor=form_factors,
            interaction_kernel_by_mesh_pair=interaction,
            h0_full=h0,
            normal_order_reference=reference,
            area_angstrom_squared=area,
            provenance="Rejected over-cap mesh.",
        )
    invalid_reference = np.eye(h0.shape[0], dtype=np.complex128) * 1.1
    with pytest.raises(ValueError, match="0 <= R <= I"):
        Vituri2024ExactIntegerFunctionalKernel(
            integer_mesh_labels=labels,
            ordered_mesh=mesh,
            form_factors_by_flavor=form_factors,
            interaction_kernel_by_mesh_pair=interaction,
            h0_full=h0,
            normal_order_reference=invalid_reference,
            area_angstrom_squared=area,
            provenance="Rejected reference.",
        )
    negative_reference = -np.eye(h0.shape[0], dtype=np.complex128) * 0.1
    with pytest.raises(ValueError, match="0 <= R <= I"):
        Vituri2024ExactIntegerFunctionalKernel(
            integer_mesh_labels=labels,
            ordered_mesh=mesh,
            form_factors_by_flavor=form_factors,
            interaction_kernel_by_mesh_pair=interaction,
            h0_full=h0,
            normal_order_reference=negative_reference,
            area_angstrom_squared=area,
            provenance="Rejected negative reference.",
        )


def test_integer_labels_expose_a_mesh_where_literal_equality_drops_channels() -> None:
    size = 7
    half = size // 2
    labels = np.asarray(
        [(ix, iy) for iy in range(-half, half + 1) for ix in range(-half, half + 1)],
        dtype=np.int64,
    )
    mesh = np.asarray(labels, dtype=np.float64) * 0.1
    receipt = compare_vituri2024_hf_spiral_literal_mask_equivalence(labels, mesh)
    assert receipt.false_positive_counts == (0, 0)
    assert all(value > 0 for value in receipt.false_negative_counts)
    assert not receipt.literal_float_quartet_mask_equivalence_established


def test_exact_integer_kernel_authority_and_public_exports() -> None:
    kernel = _kernel()
    assert kernel.api_version == VITURI2024_EXACT_INTEGER_FUNCTIONAL_API_VERSION
    assert kernel.authority == VITURI2024_EXACT_INTEGER_FUNCTIONAL_AUTHORITY
    assert kernel.convention == VITURI2024_EXACT_INTEGER_FUNCTIONAL_CONVENTION
    assert kernel.nk == 9
    assert kernel.mesh_size == 3
    assert kernel.dimension == 36
    fingerprint = kernel.kernel_fingerprint
    kernel.energy(np.zeros((kernel.dimension, kernel.dimension), dtype=np.complex128))
    assert kernel.kernel_fingerprint == fingerprint
    assert _kernel().kernel_fingerprint == fingerprint
    assert VITURI2024_EXACT_INTEGER_FUNCTIONAL_MAX_NK == 121
    assert abc.Vituri2024ExactIntegerFunctionalKernel is (
        Vituri2024ExactIntegerFunctionalKernel
    )
    assert abc.vituri2024_exact_integer_projected_interaction_action is (
        vituri2024_exact_integer_projected_interaction_action
    )
