"""Focused tests for global-rank Vituri spiral normal exact-shell closure."""

from __future__ import annotations

from copy import copy
from dataclasses import FrozenInstanceError, replace
import pickle

import numpy as np
import pytest

from mean_field.systems.abc_trilayer.vituri2024 import VITURI2024_PARAMETERS
from mean_field.systems.abc_trilayer.vituri2024_hf import (
    vituri2024_conventional_k_diagonal_to_native_density,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_scf import (
    Vituri2024CartesianHFSpec,
    make_vituri2024_cartesian_hf_spec_from_spacing,
    prepare_vituri2024_homogeneous_hf,
    prepare_vituri2024_homogeneous_hf_fft,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral import (
    Vituri2024FiniteQSpiralChoice,
    prepare_vituri2024_hf_spiral,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_density_embedding import (
    embed_vituri2024_spiral_density_nested_square,
    validate_vituri2024_spiral_density_embedding_receipt,
)
from mean_field.systems.abc_trilayer import vituri2024_hf_spiral_normal_closure as normal_closure
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_normal_closure import (
    Vituri2024SpiralNormalClosurePolicy,
    analyze_vituri2024_spiral_normal_boundary,
    build_vituri2024_spiral_normal_initializer,
    enumerate_vituri2024_spiral_normal_branch_choices,
    make_vituri2024_spiral_normal_source_lineage_receipt,
    run_vituri2024_spiral_normal_exact_shell_closure,
    validate_complete_vituri2024_spiral_normal_source_group_lineages,
    validate_vituri2024_spiral_normal_source_lineage_receipt,
)


def _prepared(*, q_a0: float = 0.02, selected_spin: int = 1, holes_per_valley: int = 1):
    base = prepare_vituri2024_homogeneous_hf(
        Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=holes_per_valley)
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


def _prepared_pair(*, q_a0: float = 0.02, holes_per_valley: int = 1):
    spec = Vituri2024CartesianHFSpec(
        mesh_size=3, holes_per_valley=holes_per_valley
    )
    choice = Vituri2024FiniteQSpiralChoice(
        q_inverse_angstrom=np.asarray(
            [q_a0 / VITURI2024_PARAMETERS.a0, 0.0], dtype=np.float64
        ),
        selected_spin=1,
        gauge_mode="displayed_b3",
        occupation_gap_floor_ev=1.0e-12,
    )
    fft = prepare_vituri2024_hf_spiral(
        prepare_vituri2024_homogeneous_hf_fft(spec, fft_workers=1), choice
    )
    dense = prepare_vituri2024_hf_spiral(
        prepare_vituri2024_homogeneous_hf(spec), choice
    )
    return fft, dense


def _nested_embedding():
    spacing_a0 = 0.12
    choice = Vituri2024FiniteQSpiralChoice(
        q_inverse_angstrom=np.asarray(
            [0.03 / VITURI2024_PARAMETERS.a0, 0.0], dtype=np.float64
        ),
        selected_spin=1,
        gauge_mode="displayed_b3",
        occupation_gap_floor_ev=1.0e-12,
    )
    source = prepare_vituri2024_hf_spiral(
        prepare_vituri2024_homogeneous_hf(
            make_vituri2024_cartesian_hf_spec_from_spacing(3, 1, spacing_a0)
        ),
        choice,
    )
    target = prepare_vituri2024_hf_spiral(
        prepare_vituri2024_homogeneous_hf(
            make_vituri2024_cartesian_hf_spec_from_spacing(5, 1, spacing_a0)
        ),
        choice,
    )
    source_result = run_vituri2024_spiral_normal_exact_shell_closure(
        source,
        policy=Vituri2024SpiralNormalClosurePolicy(
            max_iter=80, maximum_replayed_paths=64, maximum_terminals=32
        ),
    )
    source_lineages = tuple(
        make_vituri2024_spiral_normal_source_lineage_receipt(source_result, group)
        for group in source_result.stationary_groups
    )
    complete = validate_complete_vituri2024_spiral_normal_source_group_lineages(
        source_result, source_lineages
    )
    source_density = next(
        endpoint.final_density
        for endpoint in source_result.endpoints
        if endpoint.final_density_sha256
        == complete[0].source_group_final_density_sha256
    )
    receipt = embed_vituri2024_spiral_density_nested_square(
        source, target, source_density, complete[0]
    )
    return source, target, source_density, receipt, complete[0], source_result


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
    # The optional dense-oracle API must not change legacy no-oracle lineage.
    assert policy.fingerprint == "dbec28f6cfa9cd80e5cfa318a06510d6f224192f15d6ae9e124c21bbdfafe98b"
    assert choices[0].trigger.fingerprint == "6e7b6230fe0b2014cc9c500bd1acdd8b9534c86c6302d65a782c7d7d5ae038a7"
    assert choices[0].fingerprint == "34f479d96750550eda00f5030b441e53017e3c728a2b3f830c2f4a6048d60303"


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


def test_dense_boundary_oracle_receipt_binds_exact_dense_trigger() -> None:
    prepared, dense_prepared = _prepared_pair()
    pair_fingerprint = normal_closure._dense_boundary_oracle_pair_fingerprint(
        prepared, dense_prepared
    )
    rank = prepared.selected_rank
    dense_values = np.arange(2 * prepared.nk, dtype=np.float64)
    dense_values[rank - 1 : rank + 1] = float(rank - 1)
    applied_values = dense_values.copy()
    applied_values[rank] = applied_values[rank - 1] + 0.5e-12
    applied = _diagonal_hamiltonian(prepared, applied_values)
    dense = _diagonal_hamiltonian(dense_prepared, dense_values)
    policy = Vituri2024SpiralNormalClosurePolicy(boundary_floor_ev=1.0e-12)
    applied_boundary = analyze_vituri2024_spiral_normal_boundary(
        prepared, applied, policy=policy
    )
    dense_boundary = analyze_vituri2024_spiral_normal_boundary(
        dense_prepared, dense, policy=policy
    )
    assert applied_boundary.kind == "positive_subtolerance"
    assert dense_boundary.kind == "exact"
    previous = _normal_density(
        prepared, tuple(range(rank - 1)) + (rank + 1,)
    )
    receipt = normal_closure._build_dense_boundary_oracle_receipt(
        pair_fingerprint=pair_fingerprint,
        density=previous,
        applied_fft_fock=applied,
        primary_recomputed_fock=applied.copy(),
        dense_oracle_fock=dense,
        applied_boundary=applied_boundary,
        dense_boundary=dense_boundary,
        primary_energy_ev=10.0,
        dense_energy_ev=10.0 + 1.0e-12,
        policy=policy,
    )
    choices = enumerate_vituri2024_spiral_normal_branch_choices(
        prepared,
        dense,
        previous,
        dense_boundary,
        generation=0,
        policy=policy,
        dense_boundary_oracle_receipt=receipt,
    )
    assert len(choices) == 2
    assert all(
        choice.trigger.dense_boundary_oracle_receipt == receipt
        for choice in choices
    )
    assert choices[0].trigger.exact_fock_sha256 == receipt.dense_oracle_fock_sha256
    assert choices[0].trigger.previous_density_sha256 == receipt.density_sha256


def test_dense_boundary_oracle_parity_failure_is_typed_rejection() -> None:
    prepared, dense_prepared = _prepared_pair()
    pair_fingerprint = normal_closure._dense_boundary_oracle_pair_fingerprint(
        prepared, dense_prepared
    )
    rank = prepared.selected_rank
    dense_values = np.arange(2 * prepared.nk, dtype=np.float64)
    dense_values[rank - 1 : rank + 1] = float(rank - 1)
    applied_values = dense_values.copy()
    applied_values[rank] += 0.5e-12
    applied = _diagonal_hamiltonian(prepared, applied_values)
    dense = _diagonal_hamiltonian(dense_prepared, dense_values)
    dense[0, 0, 0] += 2.0e-12
    policy = Vituri2024SpiralNormalClosurePolicy(boundary_floor_ev=1.0e-12)
    with pytest.raises(RuntimeError, match="exceed the bound parity gates"):
        normal_closure._build_dense_boundary_oracle_receipt(
            pair_fingerprint=pair_fingerprint,
            density=_normal_density(prepared, tuple(range(rank))),
            applied_fft_fock=applied,
            primary_recomputed_fock=applied.copy(),
            dense_oracle_fock=dense,
            applied_boundary=analyze_vituri2024_spiral_normal_boundary(
                prepared, applied, policy=policy
            ),
            dense_boundary=analyze_vituri2024_spiral_normal_boundary(
                dense_prepared, dense, policy=policy
            ),
            primary_energy_ev=10.0,
            dense_energy_ev=10.0,
            policy=policy,
        )


def test_n81_like_reduced_nested_embedding_preserves_fixed_density_contract() -> None:
    """A 3x3 -> 5x5 transfer is the reduced analogue of an N81 replay."""

    source, target, source_density, receipt, _, _ = _nested_embedding()

    assert receipt.common_label_count == source.nk == 9
    assert receipt.annulus_label_count == target.nk - source.nk == 16
    assert receipt.expected_source_total_rank == 4 * source.nk - 2
    assert receipt.expected_target_total_rank == 4 * target.nk - 2
    assert receipt.expected_source_hole_count == receipt.expected_target_hole_count == 2
    assert receipt.expected_annulus_hole_count == 0
    assert receipt.source_trace_rank_residual >= 0.0
    assert receipt.target_trace_rank_residual >= 0.0
    assert receipt.annulus_trace_full_rank_residual == 0.0
    assert receipt.tolerance_qualified_hole_inventory_preserved is True
    assert np.array_equal(
        receipt.density_native[:, :, receipt.common_target_indices], source_density
    )
    expected_annulus = np.repeat(
        np.eye(4, dtype=np.complex128)[:, :, None],
        receipt.annulus_label_count,
        axis=2,
    )
    assert np.array_equal(
        receipt.density_native[:, :, receipt.annulus_target_indices],
        expected_annulus,
    )
    source_mesh = source.functional.mesh_receipt
    target_mesh = target.functional.mesh_receipt
    assert receipt.delta_k_inverse_angstrom == source.ordered_mesh[5, 0]
    assert source_mesh.area_angstrom_squared == target_mesh.area_angstrom_squared
    assert (
        source_mesh.uniform_weight_inverse_angstrom_squared
        == target_mesh.uniform_weight_inverse_angstrom_squared
    )
    assert np.array_equal(
        source.choice.q_inverse_angstrom, target.choice.q_inverse_angstrom
    )
    assert receipt.bitwise_common_label_copy is True
    assert receipt.exact_full_identity_annulus is True
    assert receipt.fixed_physical_contract is True
    assert receipt.uv_convergence_authority is False
    target_initializer = build_vituri2024_spiral_normal_initializer(
        target,
        policy=Vituri2024SpiralNormalClosurePolicy(),
        density_embedding_receipt=receipt,
    )
    assert target_initializer.h0_boundary_role == "validated_embedding_receipt"
    assert len(target_initializer.selected_occupied_flat_indices) == target.selected_rank
    assert receipt.density_native.flags.writeable is False
    with pytest.raises(ValueError):
        receipt.density_native[0, 0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        receipt.expected_annulus_hole_count = 1  # type: ignore[misc]


def test_embedding_receipt_is_factory_only_bytes_backed_and_strict() -> None:
    source, target, source_density, receipt, lineage, source_result = _nested_embedding()
    validate_vituri2024_spiral_density_embedding_receipt(receipt)
    assert receipt.source_group_lineage_fingerprint == lineage.fingerprint
    assert receipt.source_group_fingerprint == lineage.source_group_fingerprint
    assert receipt.source_density_sha256 == lineage.source_group_final_density_sha256

    unvalidated_lineage = make_vituri2024_spiral_normal_source_lineage_receipt(
        source_result, source_result.stationary_groups[0]
    )
    with pytest.raises(ValueError, match="complete-inventory"):
        embed_vituri2024_spiral_density_nested_square(
            source, target, source_density, unvalidated_lineage
        )
    different_density = _normal_density(
        source, tuple(range(2, source.selected_rank + 2))
    )
    if np.array_equal(different_density, source_density):
        different_density = _normal_density(source, tuple(range(source.selected_rank)))
    with pytest.raises(ValueError, match="source prepared/density"):
        embed_vituri2024_spiral_density_nested_square(
            source, target, different_density, lineage
        )

    for array in (
        receipt.source_density_native,
        receipt.embedded_density_native,
        receipt.source_integer_labels,
        receipt.target_integer_labels,
        receipt.common_target_indices,
        receipt.annulus_target_indices,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.flags.writeable = True

    mutated_input = source_density.copy()
    copied = embed_vituri2024_spiral_density_nested_square(
        source, target, mutated_input, lineage
    )
    mutated_input[0, 0, 0] += 1.0
    assert copied.source_density_native[0, 0, 0] != mutated_input[0, 0, 0]

    for invalid in (
        source_density.astype(np.complex64),
        source_density[:, :, :-1],
        np.full(source_density.shape, np.nan + 0.0j, dtype=np.complex128),
    ):
        with pytest.raises((TypeError, ValueError)):
            embed_vituri2024_spiral_density_nested_square(
                source, target, invalid, lineage
            )

    with pytest.raises(TypeError, match="InitVar"):
        replace(receipt, target_density_sha256="0" * 64)
    forged_fields = {
        name: getattr(receipt, name)
        for name, descriptor in receipt.__dataclass_fields__.items()
        if descriptor.init and name != "_factory_token"
    }
    with pytest.raises(TypeError, match="factory-only"):
        type(receipt)(_factory_token=object(), **forged_fields)

    unregistered = object.__new__(type(receipt))
    with pytest.raises(TypeError, match="identity is not registered"):
        validate_vituri2024_spiral_density_embedding_receipt(unregistered)


def test_embedding_receipt_live_revalidation_rejects_forgery() -> None:
    _, _, _, receipt, _, _ = _nested_embedding()

    original_hash = receipt.target_density_sha256
    object.__setattr__(receipt, "target_density_sha256", "0" * 64)
    with pytest.raises(ValueError, match="hash binding"):
        validate_vituri2024_spiral_density_embedding_receipt(receipt)
    object.__setattr__(receipt, "target_density_sha256", original_hash)

    original_area = receipt.area_angstrom_squared
    object.__setattr__(receipt, "area_angstrom_squared", original_area + 1.0)
    with pytest.raises(ValueError, match="scalar contract"):
        validate_vituri2024_spiral_density_embedding_receipt(receipt)
    object.__setattr__(receipt, "area_angstrom_squared", original_area)

    original_common = receipt.common_target_indices
    swapped = original_common.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    immutable_swapped = np.frombuffer(
        swapped.tobytes(order="C"), dtype=np.int64
    ).reshape(swapped.shape)
    object.__setattr__(receipt, "common_target_indices", immutable_swapped)
    with pytest.raises(ValueError, match="label mapping/index partition"):
        validate_vituri2024_spiral_density_embedding_receipt(receipt)
    object.__setattr__(receipt, "common_target_indices", original_common)

    original_prepared_fingerprint = receipt.target_prepared.fingerprint
    object.__setattr__(receipt.target_prepared, "fingerprint", "0" * 64)
    with pytest.raises(ValueError, match="fingerprint drifted"):
        validate_vituri2024_spiral_density_embedding_receipt(receipt)
    object.__setattr__(
        receipt.target_prepared, "fingerprint", original_prepared_fingerprint
    )
    validate_vituri2024_spiral_density_embedding_receipt(receipt)

    original_lineage_fingerprint = receipt.source_group_lineage_fingerprint
    original_receipt_fingerprint = receipt.fingerprint
    object.__setattr__(receipt, "source_group_lineage_fingerprint", "0" * 64)
    object.__setattr__(receipt, "fingerprint", receipt._current_fingerprint())
    with pytest.raises(ValueError, match="changed after factory issuance"):
        validate_vituri2024_spiral_density_embedding_receipt(receipt)
    object.__setattr__(
        receipt, "source_group_lineage_fingerprint", original_lineage_fingerprint
    )
    object.__setattr__(receipt, "fingerprint", original_receipt_fingerprint)
    validate_vituri2024_spiral_density_embedding_receipt(receipt)


def test_public_runner_metadata_is_module_level_and_pickle_resolvable() -> None:
    exported_name = "run_vituri2024_spiral_normal_exact_shell_closure"
    runner = run_vituri2024_spiral_normal_exact_shell_closure
    assert runner.__module__ == normal_closure.__name__
    assert runner.__name__ == exported_name
    assert runner.__qualname__ == exported_name
    assert getattr(normal_closure, exported_name) is runner
    assert pickle.loads(pickle.dumps(runner)) is runner


def test_bare_supplied_density_is_not_an_accepted_initializer_api() -> None:
    prepared = _prepared()
    supplied = normal_closure.make_vituri2024_spiral_initial_density(
        prepared, init_mode="normal"
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        build_vituri2024_spiral_normal_initializer(
            prepared,
            policy=Vituri2024SpiralNormalClosurePolicy(),
            initial_density_native=supplied,  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_vituri2024_spiral_normal_exact_shell_closure(
            prepared,
            initial_density_native=supplied,  # type: ignore[call-arg]
        )


def test_supplied_closure_requires_matching_typed_embedding_receipt() -> None:
    source, target, _, receipt, _, _ = _nested_embedding()
    with pytest.raises(TypeError, match="factory product"):
        run_vituri2024_spiral_normal_exact_shell_closure(
            target,
            density_embedding_receipt=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="target prepared/density"):
        build_vituri2024_spiral_normal_initializer(
            source,
            policy=Vituri2024SpiralNormalClosurePolicy(),
            density_embedding_receipt=receipt,
        )


def test_supplied_density_closure_still_routes_through_generic_engine(monkeypatch) -> None:
    _, prepared, _, embedding, _, _ = _nested_embedding()
    supplied = embedding.embedded_density_native
    original = normal_closure.run_hartree_fock_problem
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(normal_closure, "run_hartree_fock_problem", counted)
    result = run_vituri2024_spiral_normal_exact_shell_closure(
        prepared,
        density_embedding_receipt=embedding,
        policy=Vituri2024SpiralNormalClosurePolicy(
            max_iter=80,
            maximum_replayed_paths=64,
            maximum_terminals=32,
        ),
    )
    assert calls >= 2  # branch-tree pass plus mandatory deterministic replay
    assert result.initializer.h0_boundary_role == "validated_embedding_receipt"
    assert result.initializer.embedding_receipt_fingerprint == embedding.fingerprint
    assert np.array_equal(result.initializer.density_native, supplied)
    assert result.exhaustion_scope == "conditional_on_supplied_root"
    assert result.global_source_inventory_exhausted is False
    assert result.source_postselection_excluded is False
    assert len(result.endpoints) == 1
    assert result.endpoints[0].stationary is True


def test_legacy_common_h0_initializer_none_api_has_exact_parity() -> None:
    prepared = _prepared()
    policy = Vituri2024SpiralNormalClosurePolicy()
    legacy = build_vituri2024_spiral_normal_initializer(prepared, policy=policy)
    explicit_none = build_vituri2024_spiral_normal_initializer(
        prepared, policy=policy, density_embedding_receipt=None
    )
    assert legacy.fingerprint == explicit_none.fingerprint
    assert legacy.fingerprint == (
        "d9494189a83b354ca362b8620a74620228b46735c57809ac8619cc00a4bb1e51"
    )
    assert legacy.boundary == explicit_none.boundary
    assert legacy.selected_occupied_flat_indices == explicit_none.selected_occupied_flat_indices
    assert np.array_equal(legacy.density_native, explicit_none.density_native)
    assert legacy.h0_boundary_role in (
        "unique_aufbau",
        "declared_common_seed_exact_shell",
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


def test_exact_h0_shell_is_bound_only_as_declared_common_seed() -> None:
    prepared = _prepared(q_a0=0.06, holes_per_valley=3)
    initializer = build_vituri2024_spiral_normal_initializer(
        prepared, policy=Vituri2024SpiralNormalClosurePolicy()
    )
    assert initializer.boundary.kind == "exact"
    assert initializer.h0_boundary_role == "declared_common_seed_exact_shell"
    assert len(initializer.selected_occupied_flat_indices) == prepared.selected_rank
    selected = (1, 3)
    flat = np.concatenate(
        [prepared.h0_native[flavor, flavor, :].real for flavor in selected]
    )
    canonical = np.arange(2 * prepared.nk, dtype=np.int64)
    holes = np.lexsort((canonical, -flat))[: 2 * prepared.holes_per_valley]
    expected = tuple(int(value) for value in np.flatnonzero(~np.isin(canonical, holes)))
    assert initializer.selected_occupied_flat_indices == expected


def test_reduced_public_runner_result_closes_and_mints_source_lineage() -> None:
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
    assert result.exhaustion_scope == "conditional_on_common_h0_root"
    assert result.global_source_inventory_exhausted is False
    assert result.source_postselection_excluded is False
    assert len(result.endpoints) == 1
    assert len(result.rejections) == 0
    assert result.endpoints[0].stationary is True
    assert result.endpoints[0].metrics.engine_final_raw_parity_residual == 0.0
    assert result.deterministic_terminal_replay_verified is True
    assert result.all_applied_steps_full_step is True
    assert result.candidate_finite_domain_only is True
    assert result.same_q_energy_comparison_authorized is False
    assert result.dense_boundary_oracle_prepared_fingerprint is None
    assert result.dense_boundary_oracle_pair_fingerprint is None
    assert result.dense_boundary_oracle_receipt_fingerprints == ()
    assert result.uv_authority is False
    assert result.fig2_reproduction_authority is False

    lineages = tuple(
        make_vituri2024_spiral_normal_source_lineage_receipt(result, group)
        for group in result.stationary_groups
    )
    complete = validate_complete_vituri2024_spiral_normal_source_group_lineages(
        result, lineages
    )
    assert complete == lineages
    assert complete[0].source_group_endpoint_path_ids == result.stationary_groups[0].path_ids
    assert complete[0].all_source_endpoint_path_ids == tuple(
        endpoint.path.path_id for endpoint in result.endpoints
    )
    assert complete[0].source_group_endpoint_multiplicity == len(
        result.stationary_groups[0].path_ids
    )
    assert complete[0].global_source_inventory_exhausted is False
    assert complete[0].source_postselection_excluded is False
    with pytest.raises(ValueError, match="incomplete"):
        validate_complete_vituri2024_spiral_normal_source_group_lineages(result, ())
    with pytest.raises(TypeError, match="factory-only"):
        replace(complete[0], source_group_endpoint_multiplicity=999)


def test_source_lineage_products_are_private_detached_and_registry_bound() -> None:
    prepared = _prepared(q_a0=0.03)
    result = run_vituri2024_spiral_normal_exact_shell_closure(
        prepared,
        policy=Vituri2024SpiralNormalClosurePolicy(
            max_iter=80, maximum_replayed_paths=64, maximum_terminals=32
        ),
    )
    receipt = make_vituri2024_spiral_normal_source_lineage_receipt(
        result, result.stationary_groups[0]
    )
    validate_vituri2024_spiral_normal_source_lineage_receipt(receipt)

    assert "Vituri2024SpiralNormalSourceLineageReceipt" not in normal_closure.__all__
    assert not hasattr(normal_closure, "Vituri2024SpiralNormalSourceLineageReceipt")
    embedding_module = __import__(
        "mean_field.systems.abc_trilayer.vituri2024_hf_spiral_density_embedding",
        fromlist=["__all__"],
    )
    assert "Vituri2024SpiralDensityEmbeddingReceipt" not in embedding_module.__all__
    assert not hasattr(embedding_module, "Vituri2024SpiralDensityEmbeddingReceipt")
    package = __import__("mean_field.systems.abc_trilayer", fromlist=["__all__"])
    assert not hasattr(package, "Vituri2024SpiralNormalSourceLineageReceipt")
    assert not hasattr(package, "Vituri2024SpiralDensityEmbeddingReceipt")
    assert not any("register" in name.lower() for name in vars(normal_closure))
    assert not any("registry" in name.lower() for name in vars(normal_closure))
    assert not any(name.endswith("FACTORY_TOKEN") for name in vars(normal_closure))
    assert not any(name.endswith("FACTORY_TOKEN") for name in vars(embedding_module))

    def detached(value) -> bool:
        return type(value) in (bytes, str, int, float, bool, type(None)) or (
            type(value) is tuple and all(detached(item) for item in value)
        )

    assert not hasattr(receipt, "source_closure_result")
    assert all(
        detached(getattr(receipt, name))
        for name in receipt.__dataclass_fields__
        if name != "fingerprint"
    )

    forged_fields = {
        name: getattr(receipt, name)
        for name, descriptor in receipt.__dataclass_fields__.items()
        if descriptor.init
    }
    with pytest.raises(TypeError, match="factory-only"):
        type(receipt)(**forged_fields)
    unregistered = object.__new__(type(receipt))
    with pytest.raises(TypeError, match="identity is not registered"):
        validate_vituri2024_spiral_normal_source_lineage_receipt(unregistered)

    original = receipt.source_group_endpoint_multiplicity
    object.__setattr__(receipt, "source_group_endpoint_multiplicity", True)
    with pytest.raises(TypeError, match="integer"):
        validate_vituri2024_spiral_normal_source_lineage_receipt(receipt)
    object.__setattr__(receipt, "source_group_endpoint_multiplicity", original)
    validate_vituri2024_spiral_normal_source_lineage_receipt(receipt)

    original_closure_digest = receipt.source_closure_digest
    original_receipt_fingerprint = receipt.fingerprint
    object.__setattr__(receipt, "source_closure_digest", "0" * 64)
    object.__setattr__(receipt, "fingerprint", receipt._current_fingerprint())
    with pytest.raises(ValueError, match="changed after factory issuance"):
        validate_vituri2024_spiral_normal_source_lineage_receipt(receipt)
    object.__setattr__(receipt, "source_closure_digest", original_closure_digest)
    object.__setattr__(receipt, "fingerprint", original_receipt_fingerprint)
    validate_vituri2024_spiral_normal_source_lineage_receipt(receipt)


def test_closure_result_constructor_compatible_but_nonrunner_identities_reject() -> None:
    prepared = _prepared(q_a0=0.03)
    result = run_vituri2024_spiral_normal_exact_shell_closure(
        prepared,
        policy=Vituri2024SpiralNormalClosurePolicy(
            max_iter=80, maximum_replayed_paths=64, maximum_terminals=32
        ),
    )
    constructor_fields = {
        name: getattr(result, name)
        for name, descriptor in result.__dataclass_fields__.items()
        if descriptor.init
    }
    rebuilt = normal_closure.Vituri2024SpiralNormalClosureResult(
        **constructor_fields
    )
    assert rebuilt == result
    assert rebuilt.exhaustion_scope == "conditional_on_common_h0_root"
    assert rebuilt.global_source_inventory_exhausted is False
    assert rebuilt.source_postselection_excluded is False

    replaced = replace(result)
    copied = copy(result)
    assert rebuilt is not result
    assert replaced is not result
    assert copied is not result
    for unregistered in (rebuilt, replaced, copied):
        with pytest.raises(TypeError, match="identity is not registered"):
            make_vituri2024_spiral_normal_source_lineage_receipt(
                unregistered, unregistered.stationary_groups[0]
            )

    endpoint = result.endpoints[0]
    original_density = endpoint.final_density
    mutated_density = original_density.copy()
    mutated_density.reshape(-1)[0] += 1.0
    object.__setattr__(endpoint, "final_density", mutated_density)
    with pytest.raises(ValueError, match="final density live hash mismatch"):
        make_vituri2024_spiral_normal_source_lineage_receipt(
            result, result.stationary_groups[0]
        )
    object.__setattr__(endpoint, "final_density", original_density)

    object.__setattr__(result, "all_applied_steps_full_step", False)
    with pytest.raises(ValueError, match="changed after runner registration"):
        make_vituri2024_spiral_normal_source_lineage_receipt(
            result, result.stationary_groups[0]
        )
