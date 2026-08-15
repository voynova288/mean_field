from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from mean_field.core.hf import (
    DensityUpdateResult,
    StateBoundPreviousDensityBuilder,
    run_hartree_fock_iterations,
    select_maximum_overlap_rank_projector,
)


@dataclass
class _State:
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    mu: float = float("nan")
    precision: float = 1.0e-12
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


def test_history_aware_density_builder_receives_private_mixed_density_snapshots() -> None:
    zeros = np.zeros((1, 1, 1), dtype=np.complex128)
    state = _State(
        h0=zeros.copy(),
        density=zeros.copy(),
        hamiltonian=zeros.copy(),
        energies=np.zeros((1, 1), dtype=np.float64),
    )
    received: list[np.ndarray] = []
    received_hamiltonians: list[np.ndarray] = []

    def callback(
        hamiltonian: np.ndarray, previous_density: np.ndarray
    ) -> DensityUpdateResult:
        received.append(previous_density.copy())
        received_hamiltonians.append(hamiltonian.copy())
        previous_density[...] = 999.0
        hamiltonian[...] = 777.0
        return DensityUpdateResult(
            density=np.ones_like(state.density),
            energies=np.zeros_like(state.energies),
            mu=0.0,
        )

    run = run_hartree_fock_iterations(
        state,
        init_mode="history",
        seed=0,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=StateBoundPreviousDensityBuilder(
            state, callback, "0" * 64
        ),
        energy_functional=lambda interaction_h, h0, density: 0.0,
        oda_parameterizer=lambda state_obj, delta_density: 0.5,
        max_iter=1,
    )

    assert run.exit_reason == "max_iter"
    assert len(received) == len(received_hamiltonians) == 2
    assert np.array_equal(received[0], np.zeros_like(state.density))
    assert np.array_equal(received[1], 0.5 * np.ones_like(state.density))
    assert np.array_equal(state.density, 0.5 * np.ones_like(state.density))
    assert np.array_equal(state.hamiltonian, np.zeros_like(state.hamiltonian))
    assert all(
        np.array_equal(value, np.zeros_like(value))
        for value in received_hamiltonians
    )


def test_history_aware_builder_is_explicit_and_legacy_unary_builder_is_unchanged() -> None:
    density_state = type(
        "DensityState", (), {"density": np.zeros((1, 1, 1))}
    )()
    with pytest.raises(TypeError, match="callback"):
        StateBoundPreviousDensityBuilder(
            state=density_state,
            callback=None,  # type: ignore[arg-type]
            policy_fingerprint="0" * 64,
        )
    with pytest.raises(ValueError, match="policy_fingerprint"):
        StateBoundPreviousDensityBuilder(
            state=density_state,
            callback=lambda hamiltonian, previous: DensityUpdateResult(
                density=previous,
                energies=np.zeros((1, 1)),
                mu=0.0,
            ),
            policy_fingerprint="bad",
        )

    zeros = np.zeros((1, 1, 1), dtype=np.complex128)
    state = _State(
        h0=zeros.copy(),
        density=zeros.copy(),
        hamiltonian=zeros.copy(),
        energies=np.zeros((1, 1), dtype=np.float64),
    )
    calls = 0

    def unary(_hamiltonian: np.ndarray) -> DensityUpdateResult:
        nonlocal calls
        calls += 1
        return DensityUpdateResult(
            density=np.zeros_like(state.density),
            energies=np.zeros_like(state.energies),
            mu=0.0,
        )

    run = run_hartree_fock_iterations(
        state,
        init_mode="legacy",
        seed=0,
        interaction_builder=lambda density: np.zeros_like(density),
        density_builder=unary,
        energy_functional=lambda interaction_h, h0, density: 0.0,
        max_iter=1,
    )
    assert run.converged
    assert calls == 2


def test_maximum_overlap_selector_is_unique_and_gauge_covariant() -> None:
    overlap = np.asarray([[0.9, 0.1j], [-0.1j, 0.2]], dtype=np.complex128)
    selection = select_maximum_overlap_rank_projector(
        overlap,
        1,
        overlap_gap_tolerance=1.0e-12,
    )
    assert selection.unique
    assert selection.coefficient_projector is not None
    assert selection.overlap_cutoff_gap == pytest.approx(
        selection.overlap_eigenvalues_descending[0]
        - selection.overlap_eigenvalues_descending[1]
    )
    assert np.trace(selection.coefficient_projector).real == pytest.approx(1.0)
    assert np.max(
        np.abs(
            selection.coefficient_projector @ selection.coefficient_projector
            - selection.coefficient_projector
        )
    ) < 1.0e-13

    unitary = np.asarray(
        [[1.0, 1.0j], [1.0j, 1.0]], dtype=np.complex128
    ) / np.sqrt(2.0)
    rotated = unitary.conj().T @ overlap @ unitary
    rotated_selection = select_maximum_overlap_rank_projector(
        rotated,
        1,
        overlap_gap_tolerance=1.0e-12,
    )
    assert rotated_selection.unique
    assert rotated_selection.coefficient_projector is not None
    transported = unitary @ rotated_selection.coefficient_projector @ unitary.conj().T
    assert np.max(np.abs(transported - selection.coefficient_projector)) < 1.0e-13


def test_maximum_overlap_selector_reports_boundary_tie_without_arbitrary_branch() -> None:
    tied = select_maximum_overlap_rank_projector(
        np.eye(4, dtype=np.complex128),
        2,
        overlap_gap_tolerance=1.0e-12,
    )
    assert not tied.unique
    assert tied.coefficient_projector is None
    assert tied.overlap_cutoff_gap == 0.0
    assert np.array_equal(tied.overlap_eigenvalues_descending, np.ones(4))
    assert tied.maximum_overlap_value == 2.0

    empty = select_maximum_overlap_rank_projector(
        np.eye(4, dtype=np.complex128),
        0,
        overlap_gap_tolerance=1.0e-12,
    )
    full = select_maximum_overlap_rank_projector(
        np.eye(4, dtype=np.complex128),
        4,
        overlap_gap_tolerance=1.0e-12,
    )
    assert empty.unique and np.array_equal(empty.coefficient_projector, np.zeros((4, 4)))
    assert full.unique and np.array_equal(full.coefficient_projector, np.eye(4))


def test_maximum_overlap_selector_rejects_invalid_matrix_and_rank() -> None:
    with pytest.raises(ValueError, match="non-Hermitian"):
        select_maximum_overlap_rank_projector(
            np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
            1,
            overlap_gap_tolerance=1.0e-12,
        )
    with pytest.raises(ValueError, match="outside"):
        select_maximum_overlap_rank_projector(
            np.eye(2, dtype=np.complex128),
            3,
            overlap_gap_tolerance=1.0e-12,
        )
