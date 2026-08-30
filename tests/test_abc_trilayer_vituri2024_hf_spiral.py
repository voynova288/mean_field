"""Focused candidate/smoke tests for the Vituri G=0 finite-q spiral adapter."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from mean_field.core.hf import HartreeFockKernel, HartreeFockProblem
from mean_field.systems import abc_trilayer
from mean_field.systems.abc_trilayer.vituri2024 import BASIS
from mean_field.systems.abc_trilayer.vituri2024_hf import (
    Vituri2024TranslationalHFFunctional,
    vituri2024_native_density_to_conventional_k_diagonal,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_preflight import (
    ACTIVE_BAND_STATES_VALLEY_ORDER,
    INTERNAL_FLAVOR_ORDER,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_scf import (
    Vituri2024CartesianHFSpec,
    prepare_vituri2024_homogeneous_hf,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral import (
    Vituri2024FiniteQSpiralChoice,
    Vituri2024SpiralOccupationBoundaryError,
    apply_vituri2024_displayed_b3_gauge,
    build_vituri2024_spiral_density_map,
    make_vituri2024_hf_spiral_problem,
    make_vituri2024_spiral_initial_density,
    measure_vituri2024_spiral_ivc_order,
    prepare_vituri2024_hf_spiral,
    run_vituri2024_hf_spiral,
    transform_vituri2024_native_density_under_band_rephasing,
    transform_vituri2024_native_operator_under_band_rephasing,
)


def _base():
    return prepare_vituri2024_homogeneous_hf(
        Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=1)
    )


def _random_hermitian_native(rng: np.random.Generator, nk: int) -> np.ndarray:
    raw = rng.normal(size=(4, 4, nk)) + 1j * rng.normal(size=(4, 4, nk))
    return np.asarray(0.5 * (raw + raw.swapaxes(0, 1).conj()), dtype=np.complex128)


def test_q0_identity_gauge_is_exact_base_functional_reduction() -> None:
    base = _base()
    prepared = prepare_vituri2024_hf_spiral(
        base,
        Vituri2024FiniteQSpiralChoice(
            q_inverse_angstrom=np.zeros(2, dtype=np.float64),
            gauge_mode="identity",
        ),
    )

    assert np.array_equal(prepared.ordered_mesh, base.ordered_mesh)
    assert np.array_equal(
        prepared.shifted_momenta_by_valley,
        np.stack([base.ordered_mesh, base.ordered_mesh], axis=0),
    )
    assert np.array_equal(prepared.active_band_states, base.active_band_states)
    assert np.array_equal(
        prepared.active_band_energies_by_valley,
        base.active_band_energies_by_valley,
    )
    assert np.array_equal(prepared.h0_native, base.h0_native)
    assert prepared.functional is base.functional
    assert np.array_equal(
        prepared.functional.form_factors_by_flavor,
        base.functional.form_factors_by_flavor,
    )
    assert np.array_equal(
        prepared.functional.kernel_by_mesh_pair,
        base.functional.kernel_by_mesh_pair,
    )

    rng = np.random.default_rng(1001)
    density = _random_hermitian_native(rng, base.spec.nk)
    direction = _random_hermitian_native(rng, base.spec.nk)
    assert prepared.functional.energy(density) == base.functional.energy(density)
    assert np.array_equal(
        prepared.functional.fock(density), base.functional.fock(density)
    )
    assert np.array_equal(
        prepared.functional.fock_derivative(density, direction),
        base.functional.fock_derivative(density, direction),
    )


def test_displayed_b3_gauge_is_positive_real_and_source_phase_covariant() -> None:
    assert BASIS.index("B3") == 1
    rng = np.random.default_rng(1002)
    states = rng.normal(size=(2, 6, 5)) + 1j * rng.normal(size=(2, 6, 5))
    states[:, 1, :] += 2.0 + 0.7j
    states = np.asarray(states, dtype=np.complex128)
    states /= np.linalg.norm(states, axis=1, keepdims=True)

    gauged, receipt = apply_vituri2024_displayed_b3_gauge(
        states, anchor_floor=1.0e-8
    )
    source_rephasings = np.exp(
        1j * rng.uniform(-np.pi, np.pi, size=(2, states.shape[2]))
    ).astype(np.complex128)
    regauged, rephased_receipt = apply_vituri2024_displayed_b3_gauge(
        source_rephasings[:, None, :] * states,
        anchor_floor=1.0e-8,
    )

    assert np.array_equal(gauged[:, 1, :].imag, np.zeros((2, 5)))
    assert np.all(gauged[:, 1, :].real > 0.0)
    np.testing.assert_allclose(regauged, gauged, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        rephased_receipt.source_to_b3_phase,
        receipt.source_to_b3_phase * source_rephasings.conj(),
        rtol=0.0,
        atol=2.0e-15,
    )
    assert receipt.paper_gauge_established is False
    assert receipt.author_psi6_identification is False
    assert receipt.smooth_domain_established is False
    with pytest.raises(ValueError, match="below explicit floor"):
        apply_vituri2024_displayed_b3_gauge(
            states * np.asarray([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])[None, :, None],
            anchor_floor=1.0e-8,
        )


def test_random_band_phase_gauge_covariance_of_energy_fock_and_df() -> None:
    base = _base()
    functional = base.functional
    rng = np.random.default_rng(1003)
    valley_phases = np.exp(
        1j * rng.uniform(-np.pi, np.pi, size=(2, base.spec.nk))
    ).astype(np.complex128)
    rephased_states = np.asarray(
        valley_phases[:, None, :] * base.active_band_states,
        dtype=np.complex128,
    )
    rephased = Vituri2024TranslationalHFFunctional(
        ordered_mesh=base.ordered_mesh,
        active_band_states=rephased_states,
        h0_native=base.h0_native,
        normal_order_reference_native=functional.normal_order_reference_native,
        mesh_receipt=functional.mesh_receipt,
        interaction=functional.interaction,
        normal_order_reference_fingerprint=functional.normal_order_reference_fingerprint,
        q0_choice=functional.q0_choice,
        provenance="test-only random band rephasing of the same physical functional",
    )
    valley_index = {
        valley: index for index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER)
    }
    flavor_phases = np.stack(
        [valley_phases[valley_index[valley]] for valley, _spin in INTERNAL_FLAVOR_ORDER],
        axis=0,
    ).astype(np.complex128)
    density = _random_hermitian_native(rng, base.spec.nk)
    direction = _random_hermitian_native(rng, base.spec.nk)
    transformed_density = transform_vituri2024_native_density_under_band_rephasing(
        density, flavor_phases
    )
    transformed_direction = transform_vituri2024_native_density_under_band_rephasing(
        direction, flavor_phases
    )

    assert rephased.energy(transformed_density) == pytest.approx(
        functional.energy(density), rel=2.0e-13, abs=2.0e-13
    )
    np.testing.assert_allclose(
        rephased.fock(transformed_density),
        transform_vituri2024_native_operator_under_band_rephasing(
            functional.fock(density), flavor_phases
        ),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        rephased.fock_derivative(transformed_density, transformed_direction),
        transform_vituri2024_native_operator_under_band_rephasing(
            functional.fock_derivative(density, direction), flavor_phases
        ),
        rtol=3.0e-12,
        atol=3.0e-12,
    )


def _selected_block_hamiltonian(
    selected_blocks: tuple[np.ndarray, np.ndarray],
    *,
    selected_spin: int = 1,
) -> np.ndarray:
    matrix = np.zeros((4, 4, 2), dtype=np.complex128)
    selected = np.asarray(
        [
            index
            for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
            if spin == selected_spin
        ],
        dtype=np.int64,
    )
    spectators = np.asarray(
        [
            index
            for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
            if spin == -selected_spin
        ],
        dtype=np.int64,
    )
    for momentum, block in enumerate(selected_blocks):
        matrix[:, :, momentum][np.ix_(selected, selected)] = block
        matrix[:, :, momentum][np.ix_(spectators, spectators)] = np.diag(
            [-3.0 + momentum, -2.0 + momentum]
        )
    return matrix


def test_selected_spin_global_rank_allows_valley_redistribution_and_rejects_bad_boundaries() -> None:
    angle = 0.23
    unitary = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.complex128,
    )
    blocks = tuple(
        unitary @ np.diag(values) @ unitary.conj().T
        for values in ((0.0, 4.0), (1.0, 5.0))
    )
    matrix = _selected_block_hamiltonian(blocks)  # type: ignore[arg-type]
    result = build_vituri2024_spiral_density_map(
        matrix,
        selected_spin=1,
        selected_rank=2,
        occupation_gap_floor_ev=1.0e-9,
    )
    diagnostics = result.diagnostics
    conventional = result.density_native.swapaxes(0, 1)

    assert diagnostics.selected_rank == 2
    assert sum(diagnostics.selected_valley_electron_populations) == pytest.approx(2.0)
    assert diagnostics.selected_valley_electron_populations[0] != pytest.approx(
        diagnostics.selected_valley_electron_populations[1]
    )
    assert diagnostics.coherence_frobenius > 0.0
    assert np.array_equal(
        conventional[np.ix_([0, 2], [0, 2], [0, 1])],
        np.eye(2, dtype=np.complex128)[:, :, None].repeat(2, axis=2),
    )
    assert diagnostics.boundary.gap_ev > diagnostics.boundary.effective_floor_ev

    coupled = matrix.copy()
    coupled[1, 0, 0] = 1.0e-4
    coupled[0, 1, 0] = 1.0e-4
    with pytest.raises(ValueError, match="forbidden selected/opposite-spin"):
        build_vituri2024_spiral_density_map(
            coupled,
            selected_spin=1,
            selected_rank=2,
            occupation_gap_floor_ev=1.0e-9,
        )

    exact = _selected_block_hamiltonian(
        (np.diag([0.0, 1.0]), np.diag([1.0, 3.0]))
    )
    with pytest.raises(Vituri2024SpiralOccupationBoundaryError) as exact_error:
        build_vituri2024_spiral_density_map(
            exact,
            selected_spin=1,
            selected_rank=2,
            occupation_gap_floor_ev=1.0e-9,
        )
    assert exact_error.value.boundary.gap_ev == 0.0

    floor = 1.0e-6
    subtolerance = _selected_block_hamiltonian(
        (np.diag([0.0, 1.0 + 0.5 * floor]), np.diag([1.0, 3.0]))
    )
    with pytest.raises(Vituri2024SpiralOccupationBoundaryError) as sub_error:
        build_vituri2024_spiral_density_map(
            subtolerance,
            selected_spin=1,
            selected_rank=2,
            occupation_gap_floor_ev=floor,
        )
    assert 0.0 < sub_error.value.boundary.gap_ev < floor

    exact_floor = float(2.0**-20)
    at_floor = _selected_block_hamiltonian(
        (np.diag([0.0, 1.0 + exact_floor]), np.diag([1.0, 3.0]))
    )
    with pytest.raises(Vituri2024SpiralOccupationBoundaryError) as floor_error:
        build_vituri2024_spiral_density_map(
            at_floor,
            selected_spin=1,
            selected_rank=2,
            occupation_gap_floor_ev=exact_floor,
        )
    assert floor_error.value.boundary.gap_ev == exact_floor
    assert floor_error.value.boundary.gap_ev == (
        floor_error.value.boundary.effective_floor_ev
    )


def test_selected_spin_minus_one_density_map_path() -> None:
    angle = 0.31
    unitary = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.complex128,
    )
    blocks = tuple(
        unitary @ np.diag(values) @ unitary.conj().T
        for values in ((-1.0, 2.0), (0.0, 4.0))
    )
    matrix = _selected_block_hamiltonian(
        blocks,  # type: ignore[arg-type]
        selected_spin=-1,
    )
    result = build_vituri2024_spiral_density_map(
        matrix,
        selected_spin=-1,
        selected_rank=2,
        occupation_gap_floor_ev=1.0e-9,
    )
    conventional = vituri2024_native_density_to_conventional_k_diagonal(
        result.density_native
    )
    selected = [
        index
        for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin == -1
    ]
    spectators = [
        index
        for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin == 1
    ]
    selected_trace = sum(
        np.trace(conventional[:, :, k][np.ix_(selected, selected)]).real
        for k in range(2)
    )
    assert selected_trace == pytest.approx(2.0)
    assert np.array_equal(
        conventional[np.ix_(spectators, spectators, [0, 1])],
        np.eye(2, dtype=np.complex128)[:, :, None].repeat(2, axis=2),
    )
    assert result.diagnostics.coherence_frobenius > 0.0


def test_prepared_live_validation_rejects_nested_and_writable_replacements() -> None:
    base = _base()
    choice = Vituri2024FiniteQSpiralChoice(
        q_inverse_angstrom=np.asarray([0.01, -0.02], dtype=np.float64)
    )
    prepared = prepare_vituri2024_hf_spiral(base, choice)
    changed_q = np.asarray(choice.q_inverse_angstrom).copy()
    changed_q[0] += 1.0e-6
    changed_q.setflags(write=False)
    object.__setattr__(choice, "q_inverse_angstrom", changed_q)
    with pytest.raises(ValueError, match="choice live state drifted"):
        prepared.validate_live_state()

    clean = prepare_vituri2024_hf_spiral(
        base,
        Vituri2024FiniteQSpiralChoice(
            q_inverse_angstrom=np.asarray([0.01, -0.02], dtype=np.float64)
        ),
    )
    writable_byte_identical = np.array(clean.ordered_mesh, copy=True)
    assert writable_byte_identical.flags.writeable
    object.__setattr__(clean, "ordered_mesh", writable_byte_identical)
    with pytest.raises(ValueError, match="live ordered_mesh drifted"):
        clean.validate_live_state()


def test_nonzero_q_dense_functional_central_differences() -> None:
    base = _base()
    prepared = prepare_vituri2024_hf_spiral(
        base,
        Vituri2024FiniteQSpiralChoice(
            q_inverse_angstrom=(
                np.asarray([0.13, -0.07], dtype=np.float64)
                * base.spec.delta_k_inverse_angstrom
            ),
            gauge_mode="displayed_b3",
        ),
    )
    assert type(prepared.functional) is Vituri2024TranslationalHFFunctional
    rng = np.random.default_rng(1004)
    density = _random_hermitian_native(rng, prepared.nk)
    direction = _random_hermitian_native(rng, prepared.nk)
    step = 2.0e-6
    energy_fd = (
        prepared.functional.energy(density + step * direction)
        - prepared.functional.energy(density - step * direction)
    ) / (2.0 * step)
    energy_from_fock = float(
        np.einsum(
            "abk,abk->", prepared.functional.fock(density), direction, optimize=False
        ).real
    )
    assert energy_fd == pytest.approx(energy_from_fock, rel=2.0e-8, abs=2.0e-8)

    fock_fd = (
        prepared.functional.fock(density + step * direction)
        - prepared.functional.fock(density - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        fock_fd,
        prepared.functional.fock_derivative(density, direction),
        rtol=2.0e-8,
        atol=2.0e-8,
    )


def test_ivc_native_orientation_and_total_hole_normalization() -> None:
    base = _base()
    prepared = prepare_vituri2024_hf_spiral(
        base,
        Vituri2024FiniteQSpiralChoice(
            q_inverse_angstrom=np.asarray([0.01, 0.0], dtype=np.float64),
            gauge_mode="displayed_b3",
        ),
    )
    selected = [
        index
        for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin == prepared.choice.selected_spin
    ]
    conventional = np.zeros((4, 4, prepared.nk), dtype=np.complex128)
    expected_order = 0.25 + 0.125j
    conventional[selected[0], selected[1], 0] = expected_order
    conventional[selected[1], selected[0], 0] = expected_order.conjugate()
    native = np.asarray(conventional.swapaxes(0, 1), dtype=np.complex128)
    assert native[selected[1], selected[0], 0] == expected_order
    observable = measure_vituri2024_spiral_ivc_order(prepared, native)
    assert observable.order_sum == expected_order
    assert observable.normalization == "total_hole_density_2Hv_over_A"
    assert observable.maximum_balanced_coherent_ratio == 0.5
    assert observable.absolute_phi_over_total_hole_density <= 0.5
    assert observable.paper_normalization_resolved is False
    assert observable.paper_normalization_authority_established is False


def test_b3_regauged_ivc_observable_is_source_rephasing_invariant() -> None:
    base = _base()
    prepared = prepare_vituri2024_hf_spiral(
        base,
        Vituri2024FiniteQSpiralChoice(
            q_inverse_angstrom=np.asarray([0.01, -0.01], dtype=np.float64),
            gauge_mode="displayed_b3",
        ),
    )
    rng = np.random.default_rng(1005)
    source_rephasing = np.exp(
        1j * rng.uniform(-np.pi, np.pi, size=(2, prepared.nk))
    ).astype(np.complex128)
    regauged_states, receipt = apply_vituri2024_displayed_b3_gauge(
        source_rephasing[:, None, :] * prepared.active_band_states,
        anchor_floor=prepared.choice.b3_anchor_floor,
    )
    equivalent_functional = Vituri2024TranslationalHFFunctional(
        ordered_mesh=prepared.ordered_mesh,
        active_band_states=regauged_states,
        h0_native=prepared.h0_native,
        normal_order_reference_native=(
            prepared.functional.normal_order_reference_native
        ),
        mesh_receipt=prepared.functional.mesh_receipt,
        interaction=prepared.functional.interaction,
        normal_order_reference_fingerprint=(
            prepared.functional.normal_order_reference_fingerprint
        ),
        q0_choice=prepared.functional.q0_choice,
        provenance=prepared.functional.provenance,
    )
    equivalent = replace(
        prepared,
        active_band_states=regauged_states,
        functional=equivalent_functional,
        gauge_receipt=receipt,
    )
    density = make_vituri2024_spiral_initial_density(
        prepared, init_mode="ivc_b3"
    )
    original = measure_vituri2024_spiral_ivc_order(prepared, density)
    regauged = measure_vituri2024_spiral_ivc_order(equivalent, density)
    assert regauged.order_sum == original.order_sum
    assert (
        regauged.absolute_phi_over_total_hole_density
        == original.absolute_phi_over_total_hole_density
    )
    assert regauged.gauge_receipt_fingerprint != original.gauge_receipt_fingerprint


def test_tiny_spiral_smoke_uses_generic_problem_and_fresh_fock_map() -> None:
    base = _base()
    choice = Vituri2024FiniteQSpiralChoice(
        q_inverse_angstrom=np.asarray(
            [0.17 * base.spec.delta_k_inverse_angstrom, 0.0], dtype=np.float64
        ),
        selected_spin=1,
        gauge_mode="displayed_b3",
    )
    prepared = prepare_vituri2024_hf_spiral(base, choice)
    problem = make_vituri2024_hf_spiral_problem(prepared)
    assert type(problem) is HartreeFockProblem
    assert type(problem.kernel) is HartreeFockKernel
    assert callable(problem.initializer)
    assert callable(problem.kernel.interaction_builder)
    assert callable(problem.kernel.density_builder)
    # Absence of a system-local iteration loop is a review condition, not a
    # brittle source-string assertion.

    initial = make_vituri2024_spiral_initial_density(
        prepared, init_mode="ivc_b3"
    )
    initial_order = measure_vituri2024_spiral_ivc_order(prepared, initial)
    assert initial_order.absolute_phi_over_total_hole_density > 0.0
    assert initial_order.absolute_phi_over_total_hole_density <= 0.5
    assert initial_order.paper_gauge_established is False
    assert initial_order.paper_normalization_resolved is False

    result = run_vituri2024_hf_spiral(
        prepared,
        init_mode="ivc_b3",
        seed=0,
        max_iter=2,
        oda_stall_threshold=1.0e-12,
        max_oda_lambda=1.0,
    )
    assert result.run.iterations >= 1
    assert result.prepared_fingerprint == prepared.fingerprint
    assert result.candidate_only is True
    assert result.paper_reproduction_verified is False
    assert result.final_fock_map_diagnostics.selected_rank == prepared.selected_rank
    assert result.stationarity.selected_rank == prepared.selected_rank
    assert result.stationarity.selected_rank_residual < 1.0e-9
    assert result.stationarity.spectator_full_residual < 1.0e-12
    assert result.stationarity.boundary.gap_ev > choice.occupation_gap_floor_ev
    assert result.ivc_order is not None
    assert result.ivc_order.paper_gauge_established is False
    assert result.final_recomputed_energy_ev == pytest.approx(
        result.run.state.diagnostics["hf_energy"], rel=1.0e-10, abs=0.0
    )
    result.validate_live_state(prepared)
    if result.run.converged:
        assert result.stationarity.stationary

    def reject_snapshot_drift(mutate: object, restore: object) -> None:
        mutate()  # type: ignore[operator]
        try:
            with pytest.raises(ValueError, match="core HF run snapshot drifted"):
                result.validate_live_state(prepared)
        finally:
            restore()  # type: ignore[operator]

    for owner, attribute in (
        (result.run.state, "h0"),
        (result.run.state, "density"),
        (result.run.state, "hamiltonian"),
        (result.run.state, "energies"),
        (result.run, "iter_energy"),
        (result.run, "iter_err"),
        (result.run, "iter_oda"),
    ):
        original = getattr(owner, attribute)
        changed = original.copy()
        changed.flat[0] += 1.0e-7
        reject_snapshot_drift(
            lambda owner=owner, attribute=attribute, changed=changed: object.__setattr__(
                owner, attribute, changed
            ),
            lambda owner=owner, attribute=attribute, original=original: object.__setattr__(
                owner, attribute, original
            ),
        )

    for attribute, changed in (
        ("mu", result.run.state.mu + 1.0e-7),
        ("precision", result.run.state.precision * 2.0),
    ):
        original = getattr(result.run.state, attribute)
        reject_snapshot_drift(
            lambda attribute=attribute, changed=changed: object.__setattr__(
                result.run.state, attribute, changed
            ),
            lambda attribute=attribute, original=original: object.__setattr__(
                result.run.state, attribute, original
            ),
        )

    original_energy = result.run.state.diagnostics["hf_energy"]
    reject_snapshot_drift(
        lambda: result.run.state.diagnostics.__setitem__(
            "hf_energy", original_energy + 1.0e-7
        ),
        lambda: result.run.state.diagnostics.__setitem__("hf_energy", original_energy),
    )

    for attribute, changed in (
        ("init_mode", "random" if result.run.init_mode != "random" else "ivc_b3"),
        ("seed", result.run.seed + 1),
        ("converged", not result.run.converged),
        ("exit_reason", result.run.exit_reason + "_drift"),
    ):
        original = getattr(result.run, attribute)
        reject_snapshot_drift(
            lambda attribute=attribute, changed=changed: object.__setattr__(
                result.run, attribute, changed
            ),
            lambda attribute=attribute, original=original: object.__setattr__(
                result.run, attribute, original
            ),
        )

    result.validate_live_state(prepared)


def test_spiral_requires_positive_max_iter_and_package_api_exports() -> None:
    base = _base()
    prepared = prepare_vituri2024_hf_spiral(
        base,
        Vituri2024FiniteQSpiralChoice(
            q_inverse_angstrom=np.zeros(2, dtype=np.float64),
            gauge_mode="identity",
        ),
    )
    with pytest.raises(ValueError, match="max_iter must be positive"):
        run_vituri2024_hf_spiral(prepared, init_mode="normal", max_iter=0)
    assert (
        abc_trilayer.Vituri2024FiniteQSpiralChoice
        is Vituri2024FiniteQSpiralChoice
    )
    assert abc_trilayer.run_vituri2024_hf_spiral is run_vituri2024_hf_spiral
