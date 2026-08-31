"""Focused tests for global-rank Vituri spiral normal exact-shell closure."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from mean_field.systems.abc_trilayer.vituri2024 import VITURI2024_PARAMETERS
from mean_field.systems.abc_trilayer.vituri2024_hf import (
    vituri2024_conventional_k_diagonal_to_native_density,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_scf import (
    Vituri2024CartesianHFSpec,
    prepare_vituri2024_homogeneous_hf,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral import (
    Vituri2024FiniteQSpiralChoice,
    prepare_vituri2024_hf_spiral,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_normal_closure import (
    Vituri2024SpiralNormalClosurePolicy,
    analyze_vituri2024_spiral_normal_boundary,
    build_vituri2024_spiral_normal_initializer,
    enumerate_vituri2024_spiral_normal_branch_choices,
    run_vituri2024_spiral_normal_exact_shell_closure,
)


def _prepared(*, q_a0: float = 0.02, selected_spin: int = 1):
    base = prepare_vituri2024_homogeneous_hf(
        Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=1)
    )
    return prepare_vituri2024_hf_spiral(
        base,
        Vituri2024FiniteQSpiralChoice(
            q_inverse_angstrom=np.asarray(
                [q_a0 / VITURI2024_PARAMETERS.a0, 0.0], dtype=np.float64
            ),
            selected_spin=selected_spin,
            gauge_mode="displayed_b3",
            occupation_gap_floor_ev=1.0e-12,
        ),
    )


def _diagonal_hamiltonian(prepared, selected_flat: np.ndarray) -> np.ndarray:
    matrix = np.zeros((4, 4, prepared.nk), dtype=np.complex128)
    selected = (1, 3) if prepared.choice.selected_spin == 1 else (0, 2)
    spectators = (0, 2) if prepared.choice.selected_spin == 1 else (1, 3)
    for flavor in spectators:
        matrix[flavor, flavor, :] = -2.0
    matrix[selected[0], selected[0], :] = selected_flat[: prepared.nk]
    matrix[selected[1], selected[1], :] = selected_flat[prepared.nk :]
    return matrix


def _normal_density(prepared, occupied_flat: tuple[int, ...]) -> np.ndarray:
    conventional = np.zeros((4, 4, prepared.nk), dtype=np.complex128)
    selected = (1, 3) if prepared.choice.selected_spin == 1 else (0, 2)
    spectators = (0, 2) if prepared.choice.selected_spin == 1 else (1, 3)
    for flavor in spectators:
        conventional[flavor, flavor, :] = 1.0
    for flat_index in occupied_flat:
        slot, momentum = divmod(flat_index, prepared.nk)
        conventional[selected[slot], selected[slot], momentum] = 1.0
    return np.asarray(
        vituri2024_conventional_k_diagonal_to_native_density(conventional),
        dtype=np.complex128,
    )


def test_exact_two_choose_one_inventory_is_canonical_and_hash_bound() -> None:
    prepared = _prepared()
    rank = prepared.selected_rank
    values = np.arange(2 * prepared.nk, dtype=np.float64)
    values[rank - 1 : rank + 1] = float(rank - 1)
    hamiltonian = _diagonal_hamiltonian(prepared, values)
    policy = Vituri2024SpiralNormalClosurePolicy()
    boundary = analyze_vituri2024_spiral_normal_boundary(
        prepared, hamiltonian, policy=policy
    )
    assert boundary.kind == "exact"
    assert boundary.shell_selected_rank == 1
    assert boundary.shell_flat_indices == (rank - 1, rank)

    # All exact-shell populations are zero; the one electron needed for the
    # previous global rank is deliberately above the shell.
    previous = _normal_density(
        prepared, tuple(range(rank - 1)) + (rank + 1,)
    )
    choices = enumerate_vituri2024_spiral_normal_branch_choices(
        prepared,
        hamiltonian,
        previous,
        boundary,
        generation=0,
        policy=policy,
    )
    assert [item.selected_shell_flat_indices for item in choices] == [
        (rank - 1,),
        (rank,),
    ]
    assert choices[0].trigger.exact_fock_sha256 == choices[1].trigger.exact_fock_sha256
    assert choices[0].trigger.previous_density_sha256 == choices[1].trigger.previous_density_sha256
    assert choices[0].fingerprint != choices[1].fingerprint


def test_exact_four_choose_two_inventory_uses_lexicographic_combinations() -> None:
    prepared = _prepared()
    rank = prepared.selected_rank
    values = np.arange(2 * prepared.nk, dtype=np.float64)
    shell = tuple(range(rank - 2, rank + 2))
    values[list(shell)] = float(rank - 2)
    hamiltonian = _diagonal_hamiltonian(prepared, values)
    policy = Vituri2024SpiralNormalClosurePolicy()
    boundary = analyze_vituri2024_spiral_normal_boundary(
        prepared, hamiltonian, policy=policy
    )
    assert boundary.kind == "exact"
    assert boundary.shell_flat_indices == shell
    assert boundary.shell_selected_rank == 2

    # All shell populations are one; remove two coordinates below the shell to
    # preserve the selected global rank.
    previous_occupied = tuple(index for index in range(2 * prepared.nk) if index not in (0, 1))
    previous = _normal_density(prepared, previous_occupied)
    choices = enumerate_vituri2024_spiral_normal_branch_choices(
        prepared,
        hamiltonian,
        previous,
        boundary,
        generation=0,
        policy=policy,
    )
    expected = [
        (shell[0], shell[1]),
        (shell[0], shell[2]),
        (shell[0], shell[3]),
        (shell[1], shell[2]),
        (shell[1], shell[3]),
        (shell[2], shell[3]),
    ]
    assert [item.selected_shell_flat_indices for item in choices] == expected
    assert all(item.canonical_choice_index == index for index, item in enumerate(choices))


def test_all_exact_coordinate_choices_survive_unequal_overlap_diagnostics() -> None:
    prepared = _prepared()
    rank = prepared.selected_rank
    values = np.arange(2 * prepared.nk, dtype=np.float64)
    shell = tuple(range(rank - 2, rank + 2))
    values[list(shell)] = float(rank - 2)
    hamiltonian = _diagonal_hamiltonian(prepared, values)
    policy = Vituri2024SpiralNormalClosurePolicy()
    boundary = analyze_vituri2024_spiral_normal_boundary(
        prepared, hamiltonian, policy=policy
    )
    # Deliberately unequal shell populations: this no-postselection closure
    # still retains all six exact-Fock coordinate projectors.
    conventional = np.zeros((4, 4, prepared.nk), dtype=np.complex128)
    conventional[0, 0, :] = 1.0
    conventional[2, 2, :] = 1.0
    for slot, flat_index in enumerate(shell):
        flavor_slot, momentum = divmod(flat_index, prepared.nk)
        flavor = (1, 3)[flavor_slot]
        conventional[flavor, flavor, momentum] = (1.0, 0.75, 0.25, 0.0)[slot]
    previous = np.asarray(
        vituri2024_conventional_k_diagonal_to_native_density(conventional),
        dtype=np.complex128,
    )
    choices = enumerate_vituri2024_spiral_normal_branch_choices(
        prepared,
        hamiltonian,
        previous,
        boundary,
        generation=0,
        policy=policy,
    )
    assert len(choices) == 6
    assert choices[0].trigger.shell_previous_populations == (1.0, 0.75, 0.25, 0.0)


def test_selected_spin_minus_uses_valley_ordered_flavors_zero_and_two() -> None:
    prepared = _prepared(selected_spin=-1)
    rank = prepared.selected_rank
    values = np.arange(2 * prepared.nk, dtype=np.float64)
    values[rank - 1 : rank + 1] = float(rank - 1)
    hamiltonian = _diagonal_hamiltonian(prepared, values)
    boundary = analyze_vituri2024_spiral_normal_boundary(
        prepared, hamiltonian, policy=Vituri2024SpiralNormalClosurePolicy()
    )
    assert boundary.shell_flat_indices == (rank - 1, rank)
    first_slot, first_momentum = divmod(boundary.shell_flat_indices[0], prepared.nk)
    second_slot, second_momentum = divmod(boundary.shell_flat_indices[1], prepared.nk)
    assert ((0, 2)[first_slot], first_momentum) == (2, rank - 1 - prepared.nk)
    assert ((0, 2)[second_slot], second_momentum) == (2, rank - prepared.nk)


def test_positive_subtolerance_and_normal_coherence_fail_closed() -> None:
    prepared = _prepared()
    rank = prepared.selected_rank
    values = np.arange(2 * prepared.nk, dtype=np.float64)
    values[rank] = values[rank - 1] + 0.5e-12
    hamiltonian = _diagonal_hamiltonian(prepared, values)
    policy = Vituri2024SpiralNormalClosurePolicy(boundary_floor_ev=1.0e-12)
    boundary = analyze_vituri2024_spiral_normal_boundary(
        prepared, hamiltonian, policy=policy
    )
    assert boundary.kind == "positive_subtolerance"

    coherent = hamiltonian.copy()
    coherent[1, 3, 0] = 1.0e-30
    coherent[3, 1, 0] = 1.0e-30
    with pytest.raises(RuntimeError, match="not exactly diagonal"):
        analyze_vituri2024_spiral_normal_boundary(
            prepared, coherent, policy=policy
        )


def test_policy_and_initializer_are_immutable_and_hash_bound() -> None:
    prepared = _prepared()
    policy = Vituri2024SpiralNormalClosurePolicy()
    initializer = build_vituri2024_spiral_normal_initializer(
        prepared, policy=policy
    )
    assert initializer.density_native.flags.writeable is False
    with pytest.raises(ValueError):
        initializer.density_native[0, 0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        policy.max_iter = 1  # type: ignore[misc]
    drifted = replace(policy, max_iter=policy.max_iter + 1)
    assert drifted.fingerprint != policy.fingerprint


def test_tiny_real_path_uses_only_generic_full_steps_and_closes() -> None:
    prepared = _prepared(q_a0=0.03)
    result = run_vituri2024_spiral_normal_exact_shell_closure(
        prepared,
        policy=Vituri2024SpiralNormalClosurePolicy(
            max_iter=80,
            maximum_replayed_paths=64,
            maximum_terminals=32,
        ),
    )
    assert result.branch_tree_exhausted is True
    assert len(result.endpoints) == 1
    assert len(result.rejections) == 0
    assert result.endpoints[0].stationary is True
    assert result.endpoints[0].metrics.engine_final_raw_parity_residual == 0.0
    assert result.deterministic_terminal_replay_verified is True
    assert result.all_applied_steps_full_step is True
    assert result.candidate_finite_domain_only is True
    assert result.same_q_energy_comparison_authorized is False
    assert result.uv_authority is False
    assert result.fig2_reproduction_authority is False
