"""Reduced faithful tests for Vituri fixed-sector exhaustive SCF semantics."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import math
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import mean_field.systems.abc_trilayer as abc_root
import mean_field.systems.abc_trilayer.vituri2024_curves as curves
import mean_field.systems.abc_trilayer.vituri2024_hf_fixed_sector as fixed
import mean_field.systems.abc_trilayer.vituri2024_hf_scf as legacy
from mean_field.core.curve_workflow import (
    ExactGridCurveAdapter,
    all_branch_pointwise_spread,
)
from mean_field.systems.abc_trilayer.vituri2024 import VITURI2024_PARAMETERS
from mean_field.systems.abc_trilayer.vituri2024_hf import (
    vituri2024_native_density_to_conventional_k_diagonal,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_fixed_sector import (
    Vituri2024FixedSectorBranchFrontier,
    Vituri2024FixedSectorBranchPath,
    Vituri2024FixedSectorEndpoint,
    Vituri2024FixedSectorPolicy,
    Vituri2024FixedSectorScientificRejection,
    analyze_vituri2024_fixed_sector_boundary,
    build_vituri2024_fixed_sector_initializer,
    enumerate_vituri2024_fixed_sector_branch_choices,
    run_vituri2024_fixed_sector_bfs,
    run_vituri2024_fixed_sector_path,
)
from mean_field.systems.abc_trilayer.vituri2024_curves import (
    VITURI2024_FIXED_SECTOR_CURVE_ADAPTER_API_VERSION,
    Vituri2024FixedSectorCurveAdapter,
    build_vituri2024_fixed_sector_curve_bundle,
    make_vituri2024_fixed_sector_curve_adapter,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_scf import (
    Vituri2024CartesianHFSpec,
    Vituri2024LegacyHalfMetalSeedWarning,
    make_vituri2024_hf_problem,
    make_vituri2024_hf_state,
    prepare_vituri2024_homogeneous_hf,
    prepare_vituri2024_homogeneous_hf_fft,
)


@pytest.fixture(scope="module")
def prepared():
    return prepare_vituri2024_homogeneous_hf(
        Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=1, precision=1e-10)
    )


@pytest.fixture(scope="module")
def policy():
    return Vituri2024FixedSectorPolicy(max_iter=12)


@pytest.fixture(scope="module")
def exact_prepared():
    return prepare_vituri2024_homogeneous_hf(
        Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=5, precision=1e-10)
    )


def _diagonal_hamiltonian(prepared, *, shift: float = 0.0) -> np.ndarray:
    nk = prepared.spec.nk
    matrix = np.zeros((4, 4, nk), dtype=np.complex128)
    # Full flavors are deliberately much lower than partial flavors.  A global
    # Aufbau comparison would be relevant; fixed-sector construction ignores it.
    matrix[0, 0] = np.linspace(-1000.0, -900.0, nk) + shift
    matrix[2, 2] = np.linspace(-800.0, -700.0, nk) + shift
    matrix[1, 1] = np.arange(nk, dtype=float) + shift
    matrix[3, 3] = 100.0 + np.arange(nk, dtype=float) + shift
    return matrix


def _exact_two_by_two_hamiltonian(prepared) -> np.ndarray:
    matrix = _diagonal_hamiltonian(prepared)
    # Partial rank is Nk-1.  The two highest coordinates are an exact shell,
    # from which one coordinate must be selected independently in each flavor.
    for flavor, base in ((1, 0.0), (3, 100.0)):
        matrix[flavor, flavor] = base + np.arange(prepared.spec.nk, dtype=float)
        matrix[flavor, flavor, -2:] = base + prepared.spec.nk
    return matrix


def test_uniform_exact_shell_initializer_has_counts_and_mirror(exact_prepared, policy) -> None:
    prepared = exact_prepared
    initializer = build_vituri2024_fixed_sector_initializer(prepared, policy)
    conventional = vituri2024_native_density_to_conventional_k_diagonal(
        initializer.density_native
    )
    expected = policy.electron_counts(prepared.spec.nk, prepared.spec.holes_per_valley)
    assert initializer.mirror_symmetric
    assert tuple(np.real(conventional[f, f]).sum() for f in range(4)) == pytest.approx(expected)
    assert tuple(item.kind for item in initializer.boundaries) == ("exact", "exact")
    for boundary in initializer.boundaries:
        fraction = boundary.selected_rank / len(boundary.shell_indices)
        shell_values = conventional[boundary.flavor, boundary.flavor, list(boundary.shell_indices)]
        assert np.array_equal(shell_values, np.full(len(boundary.shell_indices), fraction))
        assert 0.0 < fraction < 1.0
    assert initializer.density_native.flags.writeable is False


def test_positive_subtolerance_is_classified_not_sorted(prepared, policy) -> None:
    matrix = _diagonal_hamiltonian(prepared)
    nk = prepared.spec.nk
    values = np.arange(nk, dtype=float)
    values[-1] = values[-2] + 0.5e-12
    matrix[1, 1] = values
    boundary = analyze_vituri2024_fixed_sector_boundary(
        prepared, matrix, flavor=1, policy=policy
    )
    assert boundary.kind == "positive_subtolerance"
    assert 0.0 < boundary.gap_ev <= boundary.effective_tolerance_ev
    assert boundary.occupied_indices == ()
    assert boundary.shell_indices == ()


def test_positive_subtolerance_h0_initializer_is_terminal_rejection(
    prepared, policy, monkeypatch
) -> None:
    original = fixed._analyze_fixed_sector_boundary_unchecked

    def classify(target, hamiltonian, *, flavor, policy):
        boundary = original(target, hamiltonian, flavor=flavor, policy=policy)
        if flavor == policy.partial_flavors[0]:
            return fixed.Vituri2024FixedSectorBoundary(
                flavor=flavor,
                electron_count=boundary.electron_count,
                kind="positive_subtolerance",
                lower_ev=0.0,
                upper_ev=0.5e-12,
                gap_ev=0.5e-12,
                effective_tolerance_ev=1.0e-12,
            )
        return boundary

    monkeypatch.setattr(fixed, "_analyze_fixed_sector_boundary_unchecked", classify)
    outcome = run_vituri2024_fixed_sector_bfs(prepared, policy=policy)
    assert isinstance(outcome, fixed.Vituri2024FixedSectorInitializationRejection)
    assert outcome.classification == "h0_positive_subtolerance_splitting_rejection"
    assert outcome.in_process_candidate_only
    assert not outcome.independent_finite_volume_fixed_sector_full_scf_discriminator
    assert not outcome.local_hessian_stability_established
    assert outcome.evidence_arrays
    assert all(not array.flags.writeable for _, array in outcome.evidence_arrays)
    assert outcome.evidence_hashes == tuple(
        (name, fixed._array_sha256(array)) for name, array in outcome.evidence_arrays
    )
    assert set(outcome.array_payload()) == {"initializer_rejection_h0"}


def test_fixed_flavors_and_unique_partial_ranks_prevent_global_transfer(prepared, policy) -> None:
    matrix = _diagonal_hamiltonian(prepared)
    boundaries = tuple(
        analyze_vituri2024_fixed_sector_boundary(
            prepared, matrix, flavor=flavor, policy=policy
        )
        for flavor in policy.partial_flavors
    )
    raw, lower, upper = fixed._raw_density_from_boundaries(
        prepared, policy, boundaries, None
    )
    conventional = vituri2024_native_density_to_conventional_k_diagonal(raw)
    counts = tuple(np.real(conventional[f, f]).sum() for f in range(4))
    assert counts == pytest.approx(
        policy.electron_counts(prepared.spec.nk, prepared.spec.holes_per_valley)
    )
    assert np.array_equal(conventional[0, 0], np.ones(prepared.spec.nk))
    assert np.array_equal(conventional[2, 2], np.ones(prepared.spec.nk))
    assert all(boundary.gap_ev > 0.0 for boundary in boundaries)
    assert math.isfinite(lower) and math.isfinite(upper)
    diagonal = np.real(np.diagonal(conventional, axis1=0, axis2=1)).T
    assert np.all((diagonal == 0.0) | (diagonal == 1.0))


def test_simultaneous_exact_shell_choices_are_unique_cartesian_product(prepared, policy) -> None:
    matrix = _exact_two_by_two_hamiltonian(prepared)
    previous = build_vituri2024_fixed_sector_initializer(prepared, policy).density_native
    boundaries = tuple(
        analyze_vituri2024_fixed_sector_boundary(
            prepared, matrix, flavor=flavor, policy=policy
        )
        for flavor in policy.partial_flavors
    )
    choices = enumerate_vituri2024_fixed_sector_branch_choices(
        prepared, matrix, previous, boundaries, generation=0, policy=policy
    )
    assert len(choices) == 4
    assert tuple(item.canonical_choice_index for item in choices) == (0, 1, 2, 3)
    selected = tuple(item.selected_momentum_indices_by_flavor for item in choices)
    assert len(set(selected)) == 4
    assert selected == (
        ((1, (7,)), (3, (7,))),
        ((1, (7,)), (3, (8,))),
        ((1, (8,)), (3, (7,))),
        ((1, (8,)), (3, (8,))),
    )
    assert all(item.trigger.exact_fock_sha256 == fixed._array_sha256(matrix) for item in choices)
    assert all(item.trigger.previous_density_sha256 == fixed._array_sha256(previous) for item in choices)


def test_trigger_path_mismatch_fails_closed(prepared, policy, monkeypatch) -> None:
    initializer = build_vituri2024_fixed_sector_initializer(prepared, policy)
    first = _exact_two_by_two_hamiltonian(prepared)
    boundaries = tuple(
        analyze_vituri2024_fixed_sector_boundary(
            prepared, first, flavor=flavor, policy=policy
        )
        for flavor in policy.partial_flavors
    )
    choice = enumerate_vituri2024_fixed_sector_branch_choices(
        prepared,
        first,
        initializer.density_native,
        boundaries,
        generation=0,
        policy=policy,
    )[0]
    second = first.copy()
    second[1, 1, 0] += 0.25

    def fake_run(state, problem, **kwargs):
        problem.initializer(state, init_mode=kwargs["init_mode"], seed=kwargs["seed"])
        problem.kernel.density_builder(second)
        raise AssertionError("mismatched trigger should fail before this point")

    monkeypatch.setattr(fixed, "run_hartree_fock_problem", fake_run)
    with pytest.raises(ValueError, match="trigger/choice mismatch"):
        run_vituri2024_fixed_sector_path(
            prepared,
            initializer,
            Vituri2024FixedSectorBranchPath((choice,)),
            policy=policy,
        )


def test_branch_choice_in_final_map_is_typed_rejection(prepared, policy, monkeypatch) -> None:
    initializer = build_vituri2024_fixed_sector_initializer(prepared, policy)
    matrix = _exact_two_by_two_hamiltonian(prepared)
    boundaries = tuple(
        analyze_vituri2024_fixed_sector_boundary(
            prepared, matrix, flavor=flavor, policy=policy
        )
        for flavor in policy.partial_flavors
    )
    choice = enumerate_vituri2024_fixed_sector_branch_choices(
        prepared,
        matrix,
        initializer.density_native,
        boundaries,
        generation=0,
        policy=policy,
    )[0]

    def fake_run(state, problem, **kwargs):
        problem.initializer(state, init_mode=kwargs["init_mode"], seed=kwargs["seed"])
        update = problem.kernel.density_builder(matrix)
        problem.kernel.final_state_callback(state, update)
        raise AssertionError("final callback must reject branch use")

    monkeypatch.setattr(fixed, "run_hartree_fock_problem", fake_run)
    outcome = run_vituri2024_fixed_sector_path(
        prepared,
        initializer,
        Vituri2024FixedSectorBranchPath((choice,)),
        policy=policy,
    )
    assert isinstance(outcome, Vituri2024FixedSectorScientificRejection)
    assert outcome.classification == "branch_choice_in_final_map_rejection"
    assert outcome.pending_choice_fingerprint == choice.fingerprint
    assert outcome.exhaustive_closure is False
    assert outcome.evidence_arrays
    assert all(not array.flags.writeable for _, array in outcome.evidence_arrays)
    assert outcome.evidence_hashes == tuple(
        (name, fixed._array_sha256(array)) for name, array in outcome.evidence_arrays
    )


def test_offdiagonal_hamiltonian_rejects_fixed_sector(prepared, policy) -> None:
    matrix = _diagonal_hamiltonian(prepared)
    matrix[0, 1, 0] = matrix[1, 0, 0] = 2.0e-10
    with pytest.raises(fixed._ScientificTerminal, match="off-diagonal Fock"):
        fixed._validate_hamiltonian(prepared, matrix, policy)


@pytest.fixture(scope="module")
def small_bfs_result(prepared, policy):
    result = run_vituri2024_fixed_sector_bfs(prepared, policy=policy)
    assert isinstance(result, fixed.Vituri2024FixedSectorSearchResult)
    return result


def test_complete_bfs_stationarity_fresh_raw_and_coalescence(small_bfs_result) -> None:
    result = small_bfs_result
    assert result.branch_tree_exhausted
    assert result.unconsumed_frontier_count == 0
    assert result.replayed_path_count == len(result.nodes)
    assert result.endpoint_count == len(result.endpoints) + len(result.rejections)
    assert result.endpoints
    assert all(isinstance(item, Vituri2024FixedSectorEndpoint) for item in result.endpoints)
    for endpoint in result.endpoints:
        assert endpoint.exhaustive_closure is False
        assert endpoint.fresh_map.fresh_hamiltonian_sha256 == endpoint.final_hamiltonian_sha256
        assert endpoint.fresh_raw_density_sha256 == endpoint.engine_final_raw_density_sha256
        assert endpoint.metrics.fresh_raw_equals_engine_final_raw_exact_bytes
        assert endpoint.metrics.fresh_fock_recompute_residual_ev <= result.policy.fresh_fock_tolerance_ev
        assert np.array_equal(endpoint.fresh_raw_density, endpoint.engine_final_raw_density)
    stationary = [item for item in result.endpoints if item.stationary]
    assert result.exact_stationary_endpoint_array_coalescence == (
        bool(stationary) and len(result.stationary_groups) == 1
    )
    if result.representative_endpoint is not None:
        assert result.representative_endpoint.stationary
    assert result.in_process_candidate_only is True
    assert result.independent_finite_volume_fixed_sector_full_scf_discriminator is False
    assert result.local_hessian_stability_established is False
    assert result.production_authority is False
    assert result.tdhf_authority is False
    assert result.full_paper_reproduction_verified is False


def test_fixed_sector_curve_adapter_exact_grid_values_and_authority(
    prepared, small_bfs_result
) -> None:
    adapter = make_vituri2024_fixed_sector_curve_adapter(prepared, small_bfs_result)
    assert isinstance(adapter, Vituri2024FixedSectorCurveAdapter)
    assert isinstance(adapter, ExactGridCurveAdapter)
    assert VITURI2024_FIXED_SECTOR_CURVE_ADAPTER_API_VERSION.endswith(".v1")

    expected_ids = tuple(sorted(endpoint.path.path_id for endpoint in small_bfs_result.endpoints))
    bundle = build_vituri2024_fixed_sector_curve_bundle(prepared, small_bfs_result)
    assert bundle.branch_closure.computed_terminal_ids == expected_ids
    assert tuple(curve.terminal_id for curve in bundle.curves) == expected_ids
    assert bundle.compute_certificate.callback_count == len(expected_ids)
    assert bundle.compute_certificate.computed_terminal_ids == expected_ids
    assert bundle.source_authority == adapter.source_authority
    authority_payload = json.loads(adapter.source_authority.canonical_payload_json)
    assert adapter.source_authority.authority_id == (
        "vituri2024_candidate_only_fixed_sector_source.v1"
    )
    assert authority_payload == {
        "source_scope": small_bfs_result.authority,
        "in_process_candidate_only": True,
        "independent_finite_volume_fixed_sector_full_scf_discriminator": False,
        "local_hessian_stability_established": False,
        "author_cutoff_identified": False,
        "uv_plateau_established": False,
        "unrestricted_ground_state_established": False,
        "full_paper_reproduction_verified": False,
        "tdhf_authority": False,
        "production_authority": False,
        "visual_match_promotes_authority": False,
    }
    evaluation = adapter.evaluate_terminal(expected_ids[0])
    assert evaluation.branch_source_id == expected_ids[0]
    assert evaluation.saved_grid is adapter.saved_grid
    assert evaluation.terminal_payload_sha256 == adapter.enumeration_receipt.payload_hash(
        expected_ids[0]
    )

    labels = prepared.integer_mesh_labels
    cut = np.flatnonzero(labels[:, 1] == 0)
    cut = cut[np.argsort(labels[cut, 0], kind="stable")]
    expected_kx = np.arange(
        -(prepared.spec.mesh_size // 2),
        prepared.spec.mesh_size // 2 + 1,
        dtype=np.int64,
    )
    np.testing.assert_array_equal(labels[cut, 0], expected_kx)
    np.testing.assert_array_equal(bundle.point_indices, cut)
    np.testing.assert_allclose(
        bundle.x,
        expected_kx
        * prepared.spec.delta_k_inverse_angstrom
        * VITURI2024_PARAMETERS.a0,
        rtol=0.0,
        atol=0.0,
    )
    assert bundle.domain.topology == "open_interval"
    assert bundle.curves[0].saved_grid.x_units == "k_x a0"

    endpoint_by_id = {
        endpoint.path.path_id: endpoint for endpoint in small_bfs_result.endpoints
    }
    for curve in bundle.curves:
        endpoint = endpoint_by_id[curve.terminal_id]
        manual_raw = np.real(endpoint.fresh_hamiltonian[3, 3, cut])
        np.testing.assert_array_equal(curve.raw_y, manual_raw)
        np.testing.assert_allclose(
            curve.output_y,
            1000.0 * (manual_raw - adapter.common_mu_ev),
            rtol=0.0,
            atol=64.0 * np.finfo(np.float64).eps,
        )
        assert curve.observable.kind == "real_diagonal_matrix_element"
        assert "fixed Vituri internal flavor basis index 3" in curve.observable.basis
        assert "Re fresh H_ff" in curve.observable.validity
        assert "discarded imaginary diagonal" in curve.observable.validity
        assert "neither raw complex H_ff nor an eigenvalue" in curve.observable.validity
        assert curve.raw_y.flags.writeable is False
        assert curve.output_y.flags.writeable is False

    transform = bundle.curves[0].value_transform
    assert transform.input_units == "eV"
    assert transform.output_units == "meV"
    assert transform.scale == 1000.0
    assert transform.offset == -1000.0 * adapter.common_mu_ev
    assert "common-intersection midpoint" in transform.semantics
    assert "Re fresh H_ff" in transform.semantics
    assert "additive gauge" in transform.semantics
    assert transform.common_across_branches is True
    assert adapter.common_mu_lower_ev <= adapter.common_mu_ev <= adapter.common_mu_upper_ev

    spread = all_branch_pointwise_spread(bundle)
    expected_output = np.stack([curve.output_y for curve in bundle.curves])
    np.testing.assert_allclose(
        spread.spread,
        np.max(expected_output, axis=0) - np.min(expected_output, axis=0),
    )
    assert spread.transforms_identical
    assert spread.value_units == "meV"

    receipt = adapter.enumeration_receipt
    assert "discrete coordinate-shell branch universe" in receipt.algorithm_id
    assert receipt.system_claims_exhaustive_enumeration is True
    assert receipt.unconsumed_frontier_count == 0
    assert adapter.authority == small_bfs_result.authority
    assert adapter.in_process_candidate_only is small_bfs_result.in_process_candidate_only
    for name in (
        "independent_finite_volume_fixed_sector_full_scf_discriminator",
        "local_hessian_stability_established",
        "author_cutoff_identified",
        "uv_plateau_established",
        "unrestricted_ground_state_established",
        "full_paper_reproduction_verified",
        "tdhf_authority",
        "production_authority",
        "visual_match_promotes_authority",
    ):
        assert getattr(adapter, name) is False
        assert not hasattr(bundle, name)


def test_vituri_expanded_inventory_rejects_an_omitted_canonical_child(
    exact_prepared, policy
) -> None:
    result = run_vituri2024_fixed_sector_bfs(exact_prepared, policy=policy)
    assert isinstance(result, fixed.Vituri2024FixedSectorSearchResult)
    parent = next(
        node
        for node in result.nodes
        if node.outcome == "expanded_exact_frontier" and len(node.child_path_ids) > 1
    )
    omitted = replace(parent, child_path_ids=parent.child_path_ids[:-1])
    node_by_id = {node.path.path_id: node for node in result.nodes}
    with pytest.raises(ValueError, match="canonical_choice_count"):
        curves._validate_expanded_node_child_inventory(omitted, node_by_id)


def test_fixed_sector_curve_adapter_rejects_full_flavor_and_empty_common_mu(
    prepared, small_bfs_result
) -> None:
    full_flavor = small_bfs_result.policy.full_flavors[0]
    with pytest.raises(ValueError, match="policy partial flavors"):
        make_vituri2024_fixed_sector_curve_adapter(
            prepared, small_bfs_result, flavor=full_flavor
        )

    endpoint = small_bfs_result.endpoints[0]
    left = replace(
        endpoint,
        fresh_map=replace(
            endpoint.fresh_map,
            common_mu_lower_ev=0.0,
            common_mu_upper_ev=1.0,
            common_mu_width_ev=1.0,
        ),
    )
    right = replace(
        endpoint,
        fresh_map=replace(
            endpoint.fresh_map,
            common_mu_lower_ev=2.0,
            common_mu_upper_ev=3.0,
            common_mu_width_ev=1.0,
        ),
    )
    with pytest.raises(ValueError, match="intersection is empty"):
        curves._common_reference_interval((left, right))


def test_fixed_sector_curve_adapter_public_exports() -> None:
    public = (
        "VITURI2024_FIXED_SECTOR_CURVE_ADAPTER_API_VERSION",
        "Vituri2024FixedSectorCurveAdapter",
        "build_vituri2024_fixed_sector_curve_bundle",
        "make_vituri2024_fixed_sector_curve_adapter",
    )
    assert curves.__all__ == list(public)
    for name in public:
        assert name in abc_root.__all__
        assert getattr(abc_root, name) is getattr(curves, name)


def test_stationary_only_coalescence_excludes_rejected_normal_endpoint(
    prepared, policy, small_bfs_result, monkeypatch
) -> None:
    original = small_bfs_result
    base = next(item for item in original.endpoints if item.stationary)
    root = Vituri2024FixedSectorBranchPath()
    matrix = _exact_two_by_two_hamiltonian(prepared)
    boundaries = tuple(
        analyze_vituri2024_fixed_sector_boundary(
            prepared, matrix, flavor=flavor, policy=policy
        )
        for flavor in policy.partial_flavors
    )
    choices = enumerate_vituri2024_fixed_sector_branch_choices(
        prepared,
        matrix,
        original.initializer.density_native,
        boundaries,
        generation=0,
        policy=policy,
    )
    frontier = Vituri2024FixedSectorBranchFrontier(root, choices[0].trigger, choices)
    rejected_path_fp = Vituri2024FixedSectorBranchPath((choices[0],)).fingerprint
    seen_initializer_ids: list[int] = []
    seen_initializer_hashes: list[str] = []

    def replay(_prepared, _initializer, path, *, policy):
        seen_initializer_ids.append(id(_initializer))
        seen_initializer_hashes.append(_initializer.density_sha256)
        if not path.choices:
            return frontier
        endpoint = replace(
            base,
            path=path,
            consumed_choice_fingerprints=(path.choices[0].fingerprint,),
        )
        if path.fingerprint == rejected_path_fp:
            engine_raw = endpoint.engine_final_raw_density.copy()
            engine_raw[0, 0, 0] += 1.0e-6
            metrics = replace(
                endpoint.metrics,
                engine_reported_final_raw_norm=fixed._relative_density_norm(
                    engine_raw, endpoint.final_density
                ),
                fresh_raw_equals_engine_final_raw_exact_bytes=False,
            )
            endpoint = replace(
                endpoint,
                outcome="normal_endpoint_gate_rejection",
                stationary=False,
                engine_final_raw_density=engine_raw,
                engine_final_raw_density_sha256=fixed._array_sha256(engine_raw),
                metrics=metrics,
            )
        return endpoint

    monkeypatch.setattr(fixed, "run_vituri2024_fixed_sector_path", replay)
    result = run_vituri2024_fixed_sector_bfs(prepared, policy=policy)
    assert not result.all_normal_endpoints_stationary
    rejected = [item for item in result.endpoints if not item.stationary]
    assert len(rejected) == 1
    grouped_paths = {path for group in result.stationary_groups for path in group.path_ids}
    assert rejected[0].path.path_id not in grouped_paths
    assert len(seen_initializer_ids) == 1 + len(choices)
    assert len(set(seen_initializer_ids)) == 1
    assert set(seen_initializer_hashes) == {result.initializer.density_sha256}


def test_caps_fail_without_returning_partial_closure(prepared, monkeypatch) -> None:
    capped = Vituri2024FixedSectorPolicy(max_iter=12, maximum_replayed_paths=1)
    initializer = build_vituri2024_fixed_sector_initializer(prepared, capped)
    matrix = _exact_two_by_two_hamiltonian(prepared)
    boundaries = tuple(
        analyze_vituri2024_fixed_sector_boundary(
            prepared, matrix, flavor=flavor, policy=capped
        )
        for flavor in capped.partial_flavors
    )
    choices = enumerate_vituri2024_fixed_sector_branch_choices(
        prepared,
        matrix,
        initializer.density_native,
        boundaries,
        generation=0,
        policy=capped,
    )
    root = Vituri2024FixedSectorBranchPath()
    frontier = Vituri2024FixedSectorBranchFrontier(root, choices[0].trigger, choices)
    monkeypatch.setattr(
        fixed,
        "run_vituri2024_fixed_sector_path",
        lambda *_args, **_kwargs: frontier,
    )
    with pytest.raises(RuntimeError, match="frontier expansion exceeds replay cap"):
        run_vituri2024_fixed_sector_bfs(prepared, policy=capped)


def test_authority_and_immutable_array_mutation_fail_closed(
    prepared, policy, small_bfs_result
) -> None:
    with pytest.raises(ValueError, match="authority was inflated"):
        replace(policy, production_authority=True)
    initializer = build_vituri2024_fixed_sector_initializer(prepared, policy)
    with pytest.raises(ValueError):
        initializer.density_native.setflags(write=True)
    with pytest.raises((AttributeError, TypeError)):
        initializer.density_sha256 = "0" * 64  # type: ignore[misc]
    with pytest.raises(ValueError, match="authority was inflated"):
        replace(small_bfs_result, production_authority=True)
    json.dumps(small_bfs_result.metadata_dict(), allow_nan=False)
    assert all(not value.flags.writeable for value in small_bfs_result.array_payload().values())


def test_low_level_path_rebuilds_initializer_exactly(prepared, policy) -> None:
    initializer = build_vituri2024_fixed_sector_initializer(prepared, policy)
    assert isinstance(initializer, fixed.Vituri2024FixedSectorInitializer)
    forged = replace(initializer, h0_sha256="0" * 64)
    with pytest.raises(ValueError, match="independent rebuild"):
        run_vituri2024_fixed_sector_path(
            prepared, forged, Vituri2024FixedSectorBranchPath(), policy=policy
        )


def test_step_callback_requires_exact_target_identity(
    prepared, policy, monkeypatch
) -> None:
    initializer = build_vituri2024_fixed_sector_initializer(prepared, policy)
    assert isinstance(initializer, fixed.Vituri2024FixedSectorInitializer)
    matrix = _diagonal_hamiltonian(prepared)

    def fake_run(state, problem, **kwargs):
        problem.initializer(state, init_mode=kwargs["init_mode"], seed=kwargs["seed"])
        update = problem.kernel.density_builder(matrix)
        foreign = make_vituri2024_hf_state(prepared)
        problem.kernel.step_callback(
            foreign, SimpleNamespace(density_update=update)
        )
        raise AssertionError("callback target identity must fail first")

    monkeypatch.setattr(fixed, "run_hartree_fock_problem", fake_run)
    with pytest.raises(RuntimeError, match="target identity"):
        run_vituri2024_fixed_sector_path(
            prepared,
            initializer,
            Vituri2024FixedSectorBranchPath(),
            policy=policy,
        )


def test_forged_endpoint_and_result_inventories_fail_closed(
    small_bfs_result,
) -> None:
    result = small_bfs_result
    endpoint = result.endpoints[0]
    with pytest.raises(ValueError, match="array hash mismatch"):
        replace(endpoint, final_density_sha256="0" * 64)
    with pytest.raises(TypeError, match="metrics must be typed"):
        replace(endpoint, metrics={})
    with pytest.raises(TypeError, match="exact tuple"):
        replace(result, nodes=list(result.nodes))
    with pytest.raises(ValueError, match="all-normal-endpoints"):
        replace(
            result,
            all_normal_endpoints_stationary=not result.all_normal_endpoints_stationary,
        )
    with pytest.raises(ValueError, match="stationary group derivation"):
        replace(result, stationary_groups=())
    forged_node = replace(result.nodes[-1], outcome="unregistered_forged_endpoint")
    with pytest.raises(ValueError, match="terminal node inventory"):
        replace(result, nodes=result.nodes[:-1] + (forged_node,))
    with pytest.raises(ValueError, match="prepared/initializer/policy binding"):
        replace(result, prepared_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="authority was inflated"):
        replace(
            result,
            independent_finite_volume_fixed_sector_full_scf_discriminator=True,
        )
    if result.representative_endpoint is not None:
        copied_representative = replace(result.representative_endpoint)
        with pytest.raises(ValueError, match="representative must be the first"):
            replace(result, representative_endpoint=copied_representative)


def test_both_fresh_and_engine_raw_norms_are_stationarity_gates(
    small_bfs_result,
) -> None:
    result = small_bfs_result
    endpoint = next(item for item in result.endpoints if item.stationary)
    for name in ("final_raw_norm", "engine_reported_final_raw_norm"):
        metrics = replace(
            endpoint.metrics,
            **{name: 2.0 * result.policy.final_raw_norm_tolerance},
        )
        forged_view = SimpleNamespace(
            converged=True, metrics=metrics, fresh_map=endpoint.fresh_map
        )
        assert not fixed._endpoint_satisfies_stationary_gates(
            forged_view, result.policy
        )


def test_fresh_raw_engine_mismatch_is_normal_endpoint_gate_not_terminal(
    small_bfs_result,
) -> None:
    result = small_bfs_result
    endpoint = next(item for item in result.endpoints if item.stationary)
    engine_raw = endpoint.engine_final_raw_density.copy()
    engine_raw[0, 0, 0] += 1.0e-6
    engine_raw.setflags(write=False)
    metrics = replace(
        endpoint.metrics,
        engine_reported_final_raw_norm=fixed._relative_density_norm(
            engine_raw, endpoint.final_density
        ),
        fresh_raw_equals_engine_final_raw_exact_bytes=False,
    )
    rejected = replace(
        endpoint,
        outcome="normal_endpoint_gate_rejection",
        stationary=False,
        engine_final_raw_density=engine_raw,
        engine_final_raw_density_sha256=fixed._array_sha256(engine_raw),
        metrics=metrics,
    )
    assert rejected.outcome == "normal_endpoint_gate_rejection"
    assert not rejected.stationary
    assert not fixed._endpoint_satisfies_stationary_gates(rejected, result.policy)
    assert "fresh_raw_engine_exact_mismatch_rejection" not in fixed.VITURI2024_FIXED_SECTOR_TERMINAL_CLASSIFICATIONS


def test_rejection_arrays_are_retained_in_candidate_payload(
    prepared, policy, monkeypatch
) -> None:
    root = Vituri2024FixedSectorBranchPath()
    rejection = Vituri2024FixedSectorScientificRejection(
        path=root,
        classification="positive_subtolerance_splitting_rejection",
        stage="density_update",
        message="typed test rejection",
        consumed_choice_fingerprints=(),
        pending_choice_fingerprint=None,
        evidence_arrays=(("density", np.arange(4.0).reshape(2, 2)),),
    )
    monkeypatch.setattr(
        fixed,
        "run_vituri2024_fixed_sector_path",
        lambda *_args, **_kwargs: rejection,
    )
    result = run_vituri2024_fixed_sector_bfs(prepared, policy=policy)
    assert isinstance(result, fixed.Vituri2024FixedSectorSearchResult)
    assert result.rejections == (rejection,)
    assert rejection.evidence_hashes == (
        ("density", fixed._array_sha256(rejection.evidence_arrays[0][1])),
    )
    retained = result.array_payload()["rejection_0000_density"]
    assert np.array_equal(retained, rejection.evidence_arrays[0][1])
    assert not retained.flags.writeable


def test_aggregate_branch_cap_allows_more_than_64_in_one_flavor(
    exact_prepared, policy, monkeypatch
) -> None:
    prepared = exact_prepared
    matrix = _diagonal_hamiltonian(prepared)
    exact = fixed.Vituri2024FixedSectorBoundary(
        flavor=1,
        electron_count=4,
        kind="exact",
        lower_ev=0.0,
        upper_ev=0.0,
        gap_ev=0.0,
        effective_tolerance_ev=1.0e-12,
        shell_indices=tuple(range(9)),
        selected_rank=4,
    )
    unique = analyze_vituri2024_fixed_sector_boundary(
        prepared, matrix, flavor=3, policy=policy
    )
    initializer = build_vituri2024_fixed_sector_initializer(prepared, policy)
    assert isinstance(initializer, fixed.Vituri2024FixedSectorInitializer)
    original_combinations = fixed.combinations
    called = False

    def forbidden_materialization(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("combinations materialized before aggregate cap")

    capped = replace(policy, maximum_choices_per_trigger=100)
    monkeypatch.setattr(fixed, "combinations", forbidden_materialization)
    with pytest.raises(RuntimeError, match="Cartesian branch choice cap"):
        enumerate_vituri2024_fixed_sector_branch_choices(
            prepared,
            matrix,
            initializer.density_native,
            (exact, unique),
            generation=0,
            policy=capped,
        )
    assert not called
    monkeypatch.setattr(fixed, "combinations", original_combinations)
    expanded = replace(policy, maximum_choices_per_trigger=200)
    choices = enumerate_vituri2024_fixed_sector_branch_choices(
        prepared,
        matrix,
        initializer.density_native,
        (exact, unique),
        generation=0,
        policy=expanded,
    )
    assert len(choices) == math.comb(9, 4) == 126
    assert choices[0].selected_momentum_indices_by_flavor == ((1, (0, 1, 2, 3)),)
    assert choices[-1].selected_momentum_indices_by_flavor == ((1, (5, 6, 7, 8)),)


def test_generation_and_endpoint_caps_fail_before_candidate_return(
    prepared, policy, small_bfs_result, monkeypatch
) -> None:
    initializer = build_vituri2024_fixed_sector_initializer(prepared, policy)
    assert isinstance(initializer, fixed.Vituri2024FixedSectorInitializer)
    matrix = _exact_two_by_two_hamiltonian(prepared)
    boundaries = tuple(
        analyze_vituri2024_fixed_sector_boundary(
            prepared, matrix, flavor=flavor, policy=policy
        )
        for flavor in policy.partial_flavors
    )
    generation_capped = replace(policy, maximum_generation=1)
    with pytest.raises(RuntimeError, match="maximum fixed-sector branch generation"):
        enumerate_vituri2024_fixed_sector_branch_choices(
            prepared,
            matrix,
            initializer.density_native,
            boundaries,
            generation=1,
            policy=generation_capped,
        )
    endpoint_capped = replace(policy, maximum_endpoints=1)
    capped_initializer = build_vituri2024_fixed_sector_initializer(
        prepared, endpoint_capped
    )
    assert isinstance(capped_initializer, fixed.Vituri2024FixedSectorInitializer)
    choices = enumerate_vituri2024_fixed_sector_branch_choices(
        prepared,
        matrix,
        capped_initializer.density_native,
        boundaries,
        generation=0,
        policy=endpoint_capped,
    )
    root = Vituri2024FixedSectorBranchPath()
    frontier = Vituri2024FixedSectorBranchFrontier(
        root, choices[0].trigger, choices
    )
    base = next(item for item in small_bfs_result.endpoints if item.stationary)

    def replay(_prepared, _initializer, path, *, policy):
        if not path.choices:
            return frontier
        return replace(
            base,
            path=path,
            consumed_choice_fingerprints=(path.choices[0].fingerprint,),
        )

    monkeypatch.setattr(fixed, "run_vituri2024_fixed_sector_path", replay)
    with pytest.raises(RuntimeError, match="endpoint cap reached before closure"):
        run_vituri2024_fixed_sector_bfs(prepared, policy=endpoint_capped)


def test_default_caps_and_root_package_surface_are_candidate_safe() -> None:
    policy = Vituri2024FixedSectorPolicy()
    assert policy.maximum_endpoints == 64
    assert policy.maximum_replayed_paths == 512
    assert policy.local_hessian_stability_established is False
    expected = {
        "Vituri2024FixedSectorBFSOutcome",
        "Vituri2024FixedSectorInitializationRejection",
        "Vituri2024FixedSectorPolicy",
        "Vituri2024FixedSectorSearchResult",
        "run_vituri2024_fixed_sector_bfs",
    }
    assert set(fixed.__all__) == expected
    assert expected.issubset(set(abc_root.__all__))
    for low_level in (
        "Vituri2024FixedSectorBranchPath",
        "Vituri2024FixedSectorBranchFrontier",
        "Vituri2024FixedSectorInitializer",
        "analyze_vituri2024_fixed_sector_boundary",
        "build_vituri2024_fixed_sector_initializer",
        "enumerate_vituri2024_fixed_sector_branch_choices",
        "run_vituri2024_fixed_sector_path",
    ):
        assert low_level not in abc_root.__all__
        assert not hasattr(abc_root, low_level)


def test_terminal_classification_inventory_is_closed() -> None:
    assert fixed.VITURI2024_FIXED_SECTOR_TERMINAL_CLASSIFICATIONS == frozenset(
        {
            "branch_choice_in_final_map_rejection",
            "branch_choice_not_applied_rejection",
            "branch_frontier_in_final_map_rejection",
            "diagonal_coherence_rejection",
            "fresh_final_fock_boundary_rejection",
            "fresh_final_fock_common_mu_rejection",
            "h0_positive_subtolerance_splitting_rejection",
            "positive_subtolerance_splitting_rejection",
        }
    )
    with pytest.raises(ValueError, match="not registered"):
        fixed._ScientificTerminal("unregistered", "stage", "message")


def test_selected_hole_spin_minus_is_unsealed_candidate(prepared) -> None:
    policy = Vituri2024FixedSectorPolicy(selected_hole_spin=-1, max_iter=12)
    assert policy.full_flavors == (1, 3)
    assert policy.partial_flavors == (0, 2)
    result = run_vituri2024_fixed_sector_bfs(prepared, policy=policy)
    assert isinstance(result, fixed.Vituri2024FixedSectorSearchResult)
    metadata = result.metadata_dict()
    assert metadata["selected_hole_spin"] == -1
    assert metadata["sealed_job461276_reference_selected_hole_spin"] == 1
    assert metadata["sealed_job461276_fixture_parity_applicable"] is False
    assert metadata["generic_symmetry_related_unsealed_candidate"] is True
    assert result.in_process_candidate_only
    assert not result.independent_finite_volume_fixed_sector_full_scf_discriminator


def test_dense_and_fft_share_initializer_and_fixed_sector_semantics(policy) -> None:
    spec = Vituri2024CartesianHFSpec(mesh_size=3, holes_per_valley=1, precision=1e-10)
    dense = prepare_vituri2024_homogeneous_hf(spec)
    fft = prepare_vituri2024_homogeneous_hf_fft(spec, fft_workers=1)
    dense_initializer = build_vituri2024_fixed_sector_initializer(dense, policy)
    fft_initializer = build_vituri2024_fixed_sector_initializer(fft, policy)
    assert np.array_equal(dense_initializer.density_native, fft_initializer.density_native)
    assert dense_initializer.density_sha256 == fft_initializer.density_sha256
    assert tuple(item.kind for item in dense_initializer.boundaries) == tuple(
        item.kind for item in fft_initializer.boundaries
    )
    density = dense_initializer.density_native
    dense_f = dense.functional.fock(density)
    fft_f = fft.functional.fock(density)
    assert np.allclose(dense_f, fft_f, rtol=3e-12, atol=3e-12)
    assert dense.functional.energy(density) == pytest.approx(
        fft.functional.energy(density), rel=3e-12, abs=3e-12
    )


def test_scalar_energy_has_no_reference_counterterm(prepared, small_bfs_result) -> None:
    endpoint = next(item for item in small_bfs_result.endpoints if item.stationary)
    density = endpoint.final_density
    interaction = endpoint.fresh_hamiltonian - prepared.h0_native
    direct = fixed._energy_from_engine_inputs(interaction, prepared.h0_native, density)
    from_f = np.real(
        np.einsum("abk,abk->", prepared.h0_native, density, optimize=False)
        + 0.5 * np.einsum("abk,abk->", interaction, density, optimize=False)
    )
    assert direct == float(from_f)
    assert direct == pytest.approx(prepared.functional.energy(density), rel=1e-12, abs=1e-12)


def test_identity_shift_changes_mu_not_fixed_sector_projector(prepared, policy) -> None:
    matrix = _diagonal_hamiltonian(prepared)
    shifted = matrix.copy()
    for flavor in range(4):
        shifted[flavor, flavor] += 37.0
    boundaries = tuple(
        analyze_vituri2024_fixed_sector_boundary(
            prepared, matrix, flavor=flavor, policy=policy
        )
        for flavor in policy.partial_flavors
    )
    shifted_boundaries = tuple(
        analyze_vituri2024_fixed_sector_boundary(
            prepared, shifted, flavor=flavor, policy=policy
        )
        for flavor in policy.partial_flavors
    )
    raw, lower, upper = fixed._raw_density_from_boundaries(prepared, policy, boundaries, None)
    shifted_raw, shifted_lower, shifted_upper = fixed._raw_density_from_boundaries(
        prepared, policy, shifted_boundaries, None
    )
    assert np.array_equal(raw, shifted_raw)
    assert shifted_lower - lower == pytest.approx(37.0)
    assert shifted_upper - upper == pytest.approx(37.0)


def test_legacy_half_metal_seed_warns_and_is_byte_compatible(prepared) -> None:
    expected = legacy._native_density_from_seed(prepared, "half_metal_sz_plus", 0)
    problem = make_vituri2024_hf_problem(prepared)
    state = make_vituri2024_hf_state(prepared)
    with pytest.warns(Vituri2024LegacyHalfMetalSeedWarning, match="not panel-c fixed-sector"):
        problem.initializer(state, init_mode="half_metal_sz_plus", seed=0)
    assert np.array_equal(state.density, expected)
    assert state.density.tobytes(order="C") == expected.tobytes(order="C")

    other_state = make_vituri2024_hf_state(prepared)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem.initializer(other_state, init_mode="valley_minus", seed=0)
    assert not any(isinstance(item.message, Vituri2024LegacyHalfMetalSeedWarning) for item in caught)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_sealed_reference_fixture_is_authority_limited() -> None:
    path = Path(__file__).parent / "fixtures/vituri2024_panel_c_fixed_sector_job461276_n179/reference.json"
    reference = json.loads(path.read_text())
    evidence = reference["evidence"]
    assert evidence["job_id"] == "461276"
    assert evidence["sealed_reference_selected_hole_spin"] == 1
    assert evidence["parity_scope"] == (
        "endpoint-array+branch-inventory parity; not full capsule fingerprint parity"
    )
    # Ordinary CI validates only tracked evidence.  The external sealed runroot
    # is checked by the explicit Slurm-only parity test below, not required in
    # portable checkouts.
    repo_root = Path(__file__).resolve().parents[1]
    attestation = repo_root / evidence["tracked_attestation_relative_path"]
    assert _file_sha256(attestation) == evidence["tracked_attestation_sha256"]
    assert reference["case"]["selected_hole_spin"] == 1
    assert reference["initializer"]["density_sha256"] == "3c685b7b2870524d462c848ecc278c224e283134b9de2b7e98e08a1e9265866c"
    assert reference["root_trigger"]["canonical_choice_count"] == 4
    assert reference["closure"]["replayed_path_count"] == 5
    assert reference["closure"]["stationary_endpoint_count"] == 4
    assert reference["closure"]["exact_stationary_endpoint_array_coalescence"]
    assert reference["authority"]["sealed_reference_selected_hole_spin"] == 1
    assert reference["authority"]["independent_finite_volume_fixed_sector_full_scf_discriminator"]
    assert all(
        reference["authority"][name] is False
        for name in (
            "uv_plateau_established",
            "author_cutoff_identified",
            "unrestricted_ground_state_established",
            "local_hessian_stability_established",
            "full_paper_reproduction_verified",
            "tdhf_authority",
            "production_authority",
            "visual_match_promotes_authority",
        )
    )


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("MEAN_FIELD_RUN_VITURI_N179_FIXED_SECTOR_PARITY") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="N179 parity is explicit-opt-in and Slurm-only",
)
def test_slurm_only_n179_parity_against_job461276_fixture() -> None:
    """Endpoint-array+branch-inventory parity, not full capsule fingerprint parity."""

    fixture_path = Path(__file__).parent / "fixtures/vituri2024_panel_c_fixed_sector_job461276_n179/reference.json"
    reference = json.loads(fixture_path.read_text())
    evidence = reference["evidence"]
    assert evidence["parity_scope"] == (
        "endpoint-array+branch-inventory parity; not full capsule fingerprint parity"
    )
    runroot = Path(evidence["runroot"])
    for relative_key, hash_key in (
        ("capsule_source_relative_path", "capsule_source_sha256"),
        ("case_summary_relative_path", "case_summary_sha256"),
    ):
        source = runroot / evidence[relative_key]
        assert source.is_file()
        assert _file_sha256(source) == evidence[hash_key]
    case = reference["case"]
    assert case["selected_hole_spin"] == 1
    spec = Vituri2024CartesianHFSpec(
        mesh_size=case["mesh_size"],
        holes_per_valley=case["holes_per_valley"],
        total_hole_density_cm2=case["total_hole_density_cm2"],
        gate_distance_angstrom=case["gate_distance_angstrom"],
        delta1_ev=case["delta1_ev"],
        precision=case["precision"],
    )
    fft_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "6"))
    prepared = prepare_vituri2024_homogeneous_hf_fft(
        spec, fft_workers=fft_workers
    )
    result = run_vituri2024_fixed_sector_bfs(
        prepared,
        policy=Vituri2024FixedSectorPolicy(selected_hole_spin=1),
    )
    assert isinstance(result, fixed.Vituri2024FixedSectorSearchResult)
    closure = reference["closure"]
    root_trigger = reference["root_trigger"]
    expected_arrays = reference["coalesced_stationary_arrays"]
    assert result.initializer.density_sha256 == reference["initializer"]["density_sha256"]
    assert result.replayed_path_count == closure["replayed_path_count"]
    assert result.endpoint_count == closure["stationary_endpoint_count"]
    assert len(result.nodes[0].child_path_ids) == root_trigger["canonical_choice_count"]
    assert sum(item.stationary for item in result.endpoints) == closure["stationary_endpoint_count"]
    assert [item.iterations for item in result.endpoints] == closure["iterations_per_stationary_endpoint"]
    assert [len(item.consumed_choice_fingerprints) for item in result.endpoints] == closure["consumed_choice_count_per_stationary_endpoint"]
    assert [item.path.choices[0].canonical_choice_index for item in result.endpoints] == closure["endpoint_path_choice_indices"]
    triggers = [item.path.choices[0].trigger for item in result.endpoints]
    assert {item.exact_fock_sha256 for item in triggers} == {root_trigger["exact_fock_sha256"]}
    assert {item.canonical_choice_count for item in triggers} == {root_trigger["canonical_choice_count"]}
    trigger = triggers[0]
    assert {str(item.flavor): item.selected_rank for item in trigger.boundaries} == root_trigger["selected_rank_by_flavor"]
    assert {str(item.flavor): list(item.shell_indices) for item in trigger.boundaries} == root_trigger["shell_momentum_indices_by_flavor"]
    expected_choices = tuple(
        tuple((int(flavor), tuple(indices)) for flavor, indices in sorted(item.items()))
        for item in root_trigger["canonical_selected_momentum_indices_by_flavor"]
    )
    actual_choices = tuple(
        endpoint.path.choices[0].selected_momentum_indices_by_flavor
        for endpoint in result.endpoints
    )
    assert actual_choices == expected_choices
    assert not result.rejections
    assert result.exact_stationary_endpoint_array_coalescence
    for endpoint in result.endpoints:
        assert endpoint.final_density_sha256 == expected_arrays["final_density_sha256"]
        assert endpoint.fresh_raw_density_sha256 == expected_arrays["fresh_raw_density_sha256"]
        assert endpoint.engine_final_raw_density_sha256 == expected_arrays["engine_final_raw_density_sha256"]
        assert endpoint.final_hamiltonian_sha256 == expected_arrays["fresh_hamiltonian_sha256"]
        assert endpoint.final_energies_sha256 == expected_arrays["fresh_energies_sha256"]
        assert endpoint.fresh_map.fresh_hamiltonian_sha256 == endpoint.final_hamiltonian_sha256
        assert endpoint.fresh_map.fresh_raw_density_sha256 == endpoint.fresh_raw_density_sha256
    assert max(item.metrics.final_raw_norm for item in result.endpoints) == pytest.approx(closure["max_final_raw_norm"], abs=0.0)
    assert max(item.metrics.engine_reported_final_raw_norm for item in result.endpoints) == pytest.approx(closure["max_engine_reported_final_raw_norm"], abs=0.0)
    assert min(item.fresh_map.common_mu_width_ev for item in result.endpoints) == pytest.approx(closure["min_common_mu_width_ev"], rel=1.0e-12, abs=1.0e-15)
    assert max(item.metrics.energy_e_f_residual_ev for item in result.endpoints) == pytest.approx(closure["max_energy_E_F_residual_ev"], rel=1.0e-12, abs=1.0e-15)
    assert result.in_process_candidate_only
    assert result.independent_finite_volume_fixed_sector_full_scf_discriminator is False
    assert result.local_hessian_stability_established is False
    assert result.author_cutoff_identified is False
    assert result.uv_plateau_established is False
    assert result.production_authority is False
    assert result.full_paper_reproduction_verified is False

    curve_bundle = build_vituri2024_fixed_sector_curve_bundle(
        prepared, result, flavor=3
    )
    assert len(curve_bundle.curves) == closure["stationary_endpoint_count"]
    assert curve_bundle.x.shape == (case["mesh_size"],)
    assert curve_bundle.branch_closure.enumeration_receipt.system_claims_exhaustive_enumeration
    assert curve_bundle.source_authority.payload()["in_process_candidate_only"] is True
    cut = curve_bundle.point_indices
    endpoint_by_id = {item.path.path_id: item for item in result.endpoints}
    for curve in curve_bundle.curves:
        endpoint = endpoint_by_id[curve.terminal_id]
        manual = np.real(endpoint.fresh_hamiltonian[3, 3, cut])
        np.testing.assert_array_equal(curve.raw_y, manual)
        np.testing.assert_allclose(
            curve.output_y,
            1000.0 * (manual + curve.value_transform.offset / 1000.0),
            rtol=0.0,
            atol=64.0 * np.finfo(np.float64).eps,
        )
