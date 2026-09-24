from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from mean_field.systems.RnG_hBN.hf import (
    RLGhBNHartreeFockRun,
    RLGhBNHartreeFockState,
    build_rlg_hbn_hf_problem,
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_projected_basis,
    rlg_hbn_flavor_occupation_counts_for_init_mode,
)
from mean_field.systems.RnG_hBN import (
    RLGhBNInteractionParams,
    RLGhBNModel,
)
from mean_field.systems.RnG_hBN.tdhf import (
    build_rlg_hbn_tdhf_c3_quotient_cycle,
    build_rlg_hbn_tdhf_orbitals,
    build_rlg_hbn_tdhf_q_pairs,
)
from mean_field.systems.RnG_hBN._tdhf_fixed_quotient import (
    build_energy_assigned_c3_sewing,
    build_periodic_gauge_basis_view,
    build_raw_pair_c3_sewing,
    c3_reciprocal_index,
    fixed_role_masks,
    pair_excitation_energies,
    required_sparse_fixed_context_padding,
)


MESH = 3
ANCHORS = ((1, 0), (0, 1), (2, 2))
PHYSICAL_SHIFTS = (
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, -1),
    (1, 1),
    (-1, 0),
    (0, -1),
)
ANCHOR_TOLERANCE_MEV = 1.0e-9
ORDINARY_TOLERANCE_MEV = 1.0e-10
WILSON_TOLERANCE = 1.0e-9


def _c3_shift(shift: tuple[int, int]) -> tuple[int, int]:
    rotated = c3_reciprocal_index(shift)
    return rotated[0] % MESH, rotated[1] % MESH


def _minus_shift(shift: tuple[int, int]) -> tuple[int, int]:
    return -int(shift[0]), -int(shift[1])


def _intraflavor_pairs(run: RLGhBNHartreeFockRun, orbitals, q_shift):
    result = []
    for pair in build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, q_shift):
        if pair.particle_flavor is None or pair.hole_flavor is None:
            raise AssertionError("finite-q pair lacks flavor metadata")
        if (
            pair.particle_flavor.spin == pair.hole_flavor.spin
            and pair.particle_flavor.valley == pair.hole_flavor.valley
        ):
            result.append(pair)
    return tuple(result)


def _build_reduced_run() -> RLGhBNHartreeFockRun:
    model = RLGhBNModel.from_config(
        layer_count=3,
        xi=1,
        theta_deg=0.77,
        displacement_field_mev=24.0,
        shell_count=1,
    )
    interaction = RLGhBNInteractionParams(
        active_valence_bands=0,
        active_conduction_bands=2,
        k_mesh_size=MESH,
        interaction_cutoff_q1=1.0,
        use_screened_basis=False,
    )
    basis_data = build_rlg_hbn_projected_basis(model, interaction, mesh_size=MESH)
    overlap_blocks = build_rlg_hbn_layer_overlap_blocks(
        basis_data,
        shifts=PHYSICAL_SHIFTS,
    )
    counts = rlg_hbn_flavor_occupation_counts_for_init_mode(
        "flavor",
        nu=1.0,
        active_valence_bands=basis_data.interaction.active_valence_bands,
        n_spin=basis_data.basis.n_spin,
        n_eta=basis_data.basis.n_flavor,
        n_band=basis_data.basis.n_band,
    )
    state = RLGhBNHartreeFockState.from_projected_basis(
        basis_data,
        nu=1.0,
        occupation_counts=counts,
    )
    problem = build_rlg_hbn_hf_problem(state, overlap_blocks)
    problem.initializer(state, init_mode="flavor", seed=1)
    update = problem.kernel.density_builder(state.h0)
    state.density[:, :, :] = update.density
    state.hamiltonian[:, :, :] = state.h0
    state.energies[:, :] = update.energies
    state.mu = update.mu
    return RLGhBNHartreeFockRun(
        state=state,
        iter_energy=(),
        iter_err=(),
        iter_oda=(),
        init_mode="flavor",
        seed=1,
        converged=False,
        exit_reason="reduced-anchor-diagnostic",
        overlap_blocks=overlap_blocks,
        basis_data=basis_data,
    )


def _pair_key(orbitals, pair) -> tuple[int, int, int]:
    p_local, _ = orbitals.decode_global_index(pair.particle)
    h_local, h_k = orbitals.decode_global_index(pair.hole)
    return int(p_local), int(h_local), int(h_k)


def _reindex_order(orbitals, reference_pairs, candidate_pairs) -> np.ndarray:
    reference = tuple(_pair_key(orbitals, pair) for pair in reference_pairs)
    candidate = tuple(_pair_key(orbitals, pair) for pair in candidate_pairs)
    if len(set(reference)) != len(reference) or len(set(candidate)) != len(candidate):
        raise AssertionError(
            f"pair keys must be unique: reference={reference}, candidate={candidate}"
        )
    if set(reference) != set(candidate):
        raise AssertionError(
            f"anchor pair bases differ: reference={reference}, candidate={candidate}"
        )
    candidate_by_key = {key: index for index, key in enumerate(candidate)}
    return np.asarray([candidate_by_key[key] for key in reference], dtype=int)


def _source_terms_for_q(cycle, q_shift):
    matches = [step for step in cycle.steps if step.source_shift == q_shift]
    if len(matches) != 1:
        raise AssertionError(f"expected one source step for q={q_shift}, got {len(matches)}")
    return matches[0].terms["q"], matches[0].terms["minus_q"]


def _fixed_masks(orbitals, pairs, q_shift, fixed_indices: set[int]):
    x_values = []
    y_values = []
    for pair in pairs:
        _, p_plus_k = orbitals.decode_global_index(pair.particle)
        _, h_k = orbitals.decode_global_index(pair.hole)
        h_i, h_j = divmod(int(h_k), MESH)
        p_minus_pair = (
            (h_i - int(q_shift[0])) % MESH,
            (h_j - int(q_shift[1])) % MESH,
        )
        p_minus_k = p_minus_pair[0] * MESH + p_minus_pair[1]
        x_values.append(int(h_k) in fixed_indices or int(p_plus_k) in fixed_indices)
        y_values.append(int(h_k) in fixed_indices or int(p_minus_k) in fixed_indices)
    return np.asarray(x_values, dtype=bool), np.asarray(y_values, dtype=bool)


def _block_residuals(
    delta: np.ndarray,
    row_fixed: np.ndarray,
    column_fixed: np.ndarray,
) -> dict[str, float]:
    row = np.asarray(row_fixed, dtype=bool)
    column = np.asarray(column_fixed, dtype=bool)
    masks = {
        "non_non": (~row)[:, None] & (~column)[None, :],
        "fixed_non": row[:, None] & (~column)[None, :],
        "non_fixed": (~row)[:, None] & column[None, :],
        "fixed_fixed": row[:, None] & column[None, :],
    }
    result = {"all": float(np.max(np.abs(delta))) if delta.size else 0.0}
    for name, mask in masks.items():
        result[name] = float(np.max(np.abs(delta[mask]))) if np.any(mask) else 0.0
    return result


def _spectrum_assignment(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    source = np.linalg.eigvals(np.asarray(left, dtype=np.complex128))
    target = np.linalg.eigvals(np.asarray(right, dtype=np.complex128))
    if source.size != target.size:
        raise AssertionError(
            f"Liouvillian spectrum sizes differ: {source.size} != {target.size}"
        )
    cost = np.abs(source[:, None] - target[None, :])
    rows, columns = linear_sum_assignment(cost)
    selected = cost[rows, columns]
    return {
        "count_mismatch": 0.0,
        "max_abs_delta_mev": float(np.max(selected)) if selected.size else 0.0,
        "mean_abs_delta_mev": float(np.mean(selected)) if selected.size else 0.0,
    }


def _best_global_phase_identity_residual(matrix: np.ndarray) -> tuple[float, complex]:
    values = np.asarray(matrix, dtype=np.complex128)
    if values.size == 0:
        return 0.0, 1.0 + 0.0j
    trace = complex(np.trace(values))
    phase = 1.0 + 0.0j if abs(trace) == 0.0 else trace / abs(trace)
    residual = float(np.max(np.abs(values - phase * np.eye(values.shape[0]))))
    return residual, phase


def _matrix_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.complex128)
    singular_values = np.linalg.svd(values, compute_uv=False) if values.size else np.asarray([])
    gram = values.conj().T @ values
    return {
        "shape": list(values.shape),
        "unitarity_defect": (
            float(np.max(np.abs(gram - np.eye(gram.shape[0])))) if gram.size else 0.0
        ),
        "singular_values": [float(value) for value in singular_values],
        "condition_number": float(np.linalg.cond(values)) if values.size else 0.0,
    }


@dataclass
class _ReducedDiagnostic:
    anchor_summary: dict[str, Any]
    wilson_summary: dict[str, Any]


def _build_anchor_summary(run, orbitals, common_padding: int):
    cycles = {
        anchor: build_rlg_hbn_tdhf_c3_quotient_cycle(
            run,
            orbitals,
            anchor,
            physical_shifts=PHYSICAL_SHIFTS,
            fixed_copy=0,
            periodic_gauge_padding=common_padding,
            structure_tolerance=1.0e-10,
            closure_tolerance=ANCHOR_TOLERANCE_MEV,
            require_closure=False,
        )
        for anchor in ANCHORS
    }
    expected_fixed_pairs = {(1, 2), (2, 1)}
    actual_fixed_pairs = {
        (int(pair[0]), int(pair[1]))
        for pair in run.basis_data.c3_fixed_representative_pairs
    }
    if actual_fixed_pairs != expected_fixed_pairs:
        raise AssertionError(
            f"tiny3 fixed representatives changed: {actual_fixed_pairs}"
        )
    fixed_indices = {pair[0] * MESH + pair[1] for pair in actual_fixed_pairs}
    summary: dict[str, Any] = {
        "anchors": {
            f"{anchor[0]},{anchor[1]}": {
                "cycle_shifts": [list(shift) for shift in cycle.shifts],
                "closure_residuals_mev": cycle.closure_residuals,
                "maximum_A_hermitian": max(
                    matrix.structure.a_hermitian for matrix in cycle.matrices.values()
                ),
                "maximum_B_transpose": max(
                    matrix.structure.b_symmetric for matrix in cycle.matrices.values()
                ),
            }
            for anchor, cycle in cycles.items()
        },
        "common_q": {},
    }
    maxima = {
        "term_ordinary_non_non_mev": 0.0,
        "term_fixed_touched_mev": 0.0,
        "matrix_ordinary_non_non_mev": 0.0,
        "matrix_fixed_touched_mev": 0.0,
        "liouvillian_ordinary_non_non_mev": 0.0,
        "liouvillian_fixed_touched_mev": 0.0,
        "spectrum_assignment_mev": 0.0,
    }
    anchor_pairs = (
        (ANCHORS[0], ANCHORS[1]),
        (ANCHORS[0], ANCHORS[2]),
        (ANCHORS[1], ANCHORS[2]),
    )
    for q_shift in ANCHORS:
        q_key = f"{q_shift[0]},{q_shift[1]}"
        summary["common_q"][q_key] = {}
        reference_pairs = cycles[ANCHORS[0]].matrices[q_shift].pairs
        x_fixed, y_fixed = _fixed_masks(
            orbitals,
            reference_pairs,
            q_shift,
            fixed_indices,
        )
        production_x, production_y = fixed_role_masks(
            orbitals,
            reference_pairs,
            q_shift,
            fixed_indices,
            MESH,
        )
        np.testing.assert_array_equal(x_fixed, production_x)
        np.testing.assert_array_equal(y_fixed, production_y)
        if not (np.any(x_fixed) and np.any(~x_fixed) and np.any(y_fixed) and np.any(~y_fixed)):
            raise AssertionError(
                f"q={q_shift} does not exercise both fixed and ordinary X/Y roles"
            )

        minus_pairs_raw = _intraflavor_pairs(run, orbitals, _minus_shift(q_shift))
        minus_order = _reindex_order(orbitals, reference_pairs, minus_pairs_raw)
        minus_x_raw, minus_y_raw = _fixed_masks(
            orbitals,
            minus_pairs_raw,
            _minus_shift(q_shift),
            fixed_indices,
        )
        minus_x = minus_x_raw[minus_order]
        minus_y = minus_y_raw[minus_order]
        production_minus_x, production_minus_y = fixed_role_masks(
            orbitals,
            minus_pairs_raw,
            _minus_shift(q_shift),
            fixed_indices,
            MESH,
        )
        np.testing.assert_array_equal(minus_x_raw, production_minus_x)
        np.testing.assert_array_equal(minus_y_raw, production_minus_y)
        if not (
            np.any(minus_x)
            and np.any(~minus_x)
            and np.any(minus_y)
            and np.any(~minus_y)
        ):
            raise AssertionError(
                f"-q={_minus_shift(q_shift)} does not exercise fixed and ordinary roles"
            )

        for left_anchor, right_anchor in anchor_pairs:
            left_cycle = cycles[left_anchor]
            right_cycle = cycles[right_anchor]
            left_pairs = left_cycle.matrices[q_shift].pairs
            right_pairs = right_cycle.matrices[q_shift].pairs
            left_order = _reindex_order(orbitals, reference_pairs, left_pairs)
            right_order = _reindex_order(orbitals, reference_pairs, right_pairs)
            left_plus, left_minus = _source_terms_for_q(left_cycle, q_shift)
            right_plus, right_minus = _source_terms_for_q(right_cycle, q_shift)
            comparison: dict[str, Any] = {"plus_terms": {}, "minus_terms": {}}
            for name in ("A0", "A_direct", "A_exchange", "B_direct", "B_exchange"):
                left_term = left_plus[name][np.ix_(left_order, left_order)]
                right_term = right_plus[name][np.ix_(right_order, right_order)]
                row_mask, column_mask = (
                    (x_fixed, x_fixed) if name.startswith("A") else (x_fixed, y_fixed)
                )
                residuals = _block_residuals(left_term - right_term, row_mask, column_mask)
                comparison["plus_terms"][name] = residuals
                maxima["term_ordinary_non_non_mev"] = max(
                    maxima["term_ordinary_non_non_mev"], residuals["non_non"]
                )
                maxima["term_fixed_touched_mev"] = max(
                    maxima["term_fixed_touched_mev"],
                    residuals["fixed_non"],
                    residuals["non_fixed"],
                    residuals["fixed_fixed"],
                )

                left_partner = left_minus[name][np.ix_(minus_order, minus_order)]
                right_partner = right_minus[name][np.ix_(minus_order, minus_order)]
                row_mask, column_mask = (
                    (minus_x, minus_x)
                    if name.startswith("A")
                    else (minus_x, minus_y)
                )
                partner_residuals = _block_residuals(
                    left_partner - right_partner,
                    row_mask,
                    column_mask,
                )
                comparison["minus_terms"][name] = partner_residuals
                maxima["term_ordinary_non_non_mev"] = max(
                    maxima["term_ordinary_non_non_mev"],
                    partner_residuals["non_non"],
                )
                maxima["term_fixed_touched_mev"] = max(
                    maxima["term_fixed_touched_mev"],
                    partner_residuals["fixed_non"],
                    partner_residuals["non_fixed"],
                    partner_residuals["fixed_fixed"],
                )

            left_matrices = left_cycle.matrices[q_shift]
            right_matrices = right_cycle.matrices[q_shift]
            left_a = left_matrices.A[np.ix_(left_order, left_order)]
            right_a = right_matrices.A[np.ix_(right_order, right_order)]
            left_b = left_matrices.B[np.ix_(left_order, left_order)]
            right_b = right_matrices.B[np.ix_(right_order, right_order)]
            comparison["A"] = _block_residuals(left_a - right_a, x_fixed, x_fixed)
            comparison["B"] = _block_residuals(left_b - right_b, x_fixed, y_fixed)
            for name in ("A", "B"):
                residuals = comparison[name]
                maxima["matrix_ordinary_non_non_mev"] = max(
                    maxima["matrix_ordinary_non_non_mev"], residuals["non_non"]
                )
                maxima["matrix_fixed_touched_mev"] = max(
                    maxima["matrix_fixed_touched_mev"],
                    residuals["fixed_non"],
                    residuals["non_fixed"],
                    residuals["fixed_fixed"],
                )

            n_pairs = len(reference_pairs)
            left_full_order = np.concatenate((left_order, n_pairs + minus_order))
            right_full_order = np.concatenate((right_order, n_pairs + minus_order))
            left_l = left_matrices.L[np.ix_(left_full_order, left_full_order)]
            right_l = right_matrices.L[np.ix_(right_full_order, right_full_order)]
            full_fixed = np.concatenate((x_fixed, minus_x))
            comparison["L"] = _block_residuals(
                left_l - right_l,
                full_fixed,
                full_fixed,
            )
            maxima["liouvillian_ordinary_non_non_mev"] = max(
                maxima["liouvillian_ordinary_non_non_mev"],
                comparison["L"]["non_non"],
            )
            maxima["liouvillian_fixed_touched_mev"] = max(
                maxima["liouvillian_fixed_touched_mev"],
                comparison["L"]["fixed_non"],
                comparison["L"]["non_fixed"],
                comparison["L"]["fixed_fixed"],
            )
            comparison["spectrum"] = _spectrum_assignment(left_l, right_l)
            maxima["spectrum_assignment_mev"] = max(
                maxima["spectrum_assignment_mev"],
                comparison["spectrum"]["max_abs_delta_mev"],
            )
            label = (
                f"{left_anchor[0]},{left_anchor[1]}__"
                f"{right_anchor[0]},{right_anchor[1]}"
            )
            summary["common_q"][q_key][label] = comparison
    summary["maxima"] = maxima
    return summary


def _identity_residual(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.complex128)
    if values.size == 0:
        raise AssertionError("Wilson block is empty")
    return float(np.max(np.abs(values - np.eye(values.shape[0]))))


def _wilson_block_summary(matrix: np.ndarray, fixed_mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.complex128)
    fixed = np.asarray(fixed_mask, dtype=bool)
    ordinary_block = values[np.ix_(~fixed, ~fixed)]
    fixed_block = values[np.ix_(fixed, fixed)]
    if ordinary_block.size == 0 or fixed_block.size == 0:
        raise AssertionError("Wilson gate requires nonempty ordinary and fixed blocks")
    ordinary_best, ordinary_phase = _best_global_phase_identity_residual(ordinary_block)
    fixed_best, fixed_phase = _best_global_phase_identity_residual(fixed_block)
    cross = max(
        float(np.max(np.abs(values[np.ix_(fixed, ~fixed)]))),
        float(np.max(np.abs(values[np.ix_(~fixed, fixed)]))),
    )
    return {
        "ordinary": _matrix_diagnostics(ordinary_block),
        "fixed": _matrix_diagnostics(fixed_block),
        "ordinary_identity_residual": _identity_residual(ordinary_block),
        "fixed_identity_residual": _identity_residual(fixed_block),
        "ordinary_best_global_phase_residual_diagnostic": ordinary_best,
        "ordinary_best_global_phase_diagnostic": [
            ordinary_phase.real,
            ordinary_phase.imag,
        ],
        "fixed_best_global_phase_residual_diagnostic": fixed_best,
        "fixed_best_global_phase_diagnostic": [fixed_phase.real, fixed_phase.imag],
        "cross_block_leakage": cross,
        "ordinary_eigenphases_rad": [
            float(value) for value in np.sort(np.angle(np.linalg.eigvals(ordinary_block)))
        ],
        "fixed_eigenphases_rad": [
            float(value) for value in np.sort(np.angle(np.linalg.eigvals(fixed_block)))
        ],
    }


def _build_independent_wilson_sector(
    run,
    orbitals,
    basis,
    orbit_shifts: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    fixed_indices: set[int],
) -> dict[str, Any]:
    pairs_by_shift = {
        shift: _intraflavor_pairs(run, orbitals, shift) for shift in orbit_shifts
    }
    masks_by_shift: dict[tuple[int, int], np.ndarray] = {}
    for shift, pairs in pairs_by_shift.items():
        independent_x, _ = _fixed_masks(orbitals, pairs, shift, fixed_indices)
        production_x, _ = fixed_role_masks(
            orbitals,
            pairs,
            shift,
            fixed_indices,
            MESH,
        )
        np.testing.assert_array_equal(independent_x, production_x)
        if not (np.any(independent_x) and np.any(~independent_x)):
            raise AssertionError(f"shift={shift} lacks fixed or ordinary pair roles")
        masks_by_shift[shift] = independent_x

    vector_cache: dict[tuple[int, int, tuple[int, int]], np.ndarray] = {}
    raw_matrices = []
    assigned_matrices = []
    edges = []
    for index, source_shift in enumerate(orbit_shifts):
        target_shift = orbit_shifts[(index + 1) % len(orbit_shifts)]
        source_pairs = pairs_by_shift[source_shift]
        target_pairs = pairs_by_shift[target_shift]
        source_fixed = masks_by_shift[source_shift]
        target_fixed = masks_by_shift[target_shift]
        if int(np.count_nonzero(source_fixed)) != int(np.count_nonzero(target_fixed)):
            raise AssertionError(
                f"fixed pair counts differ on edge {source_shift}->{target_shift}"
            )
        raw = build_raw_pair_c3_sewing(
            basis,
            orbitals,
            source_pairs=source_pairs,
            source_shift=source_shift,
            target_pairs=target_pairs,
            target_shift=target_shift,
            vector_cache=vector_cache,
        )
        assigned = build_energy_assigned_c3_sewing(
            raw,
            source_fixed=source_fixed,
            target_fixed=target_fixed,
            source_energies=pair_excitation_energies(orbitals, source_pairs),
            target_energies=pair_excitation_energies(orbitals, target_pairs),
        )
        raw_ordinary = raw[np.ix_(~target_fixed, ~source_fixed)]
        raw_fixed = raw[np.ix_(target_fixed, source_fixed)]
        raw_cross = max(
            float(np.max(np.abs(raw[np.ix_(target_fixed, ~source_fixed)]))),
            float(np.max(np.abs(raw[np.ix_(~target_fixed, source_fixed)]))),
        )
        edges.append(
            {
                "source_shift": list(source_shift),
                "target_shift": list(target_shift),
                "raw_full": _matrix_diagnostics(raw),
                "raw_ordinary": _matrix_diagnostics(raw_ordinary),
                "raw_fixed": _matrix_diagnostics(raw_fixed),
                "raw_cross_block_leakage": raw_cross,
                "assigned_unitarity_defect": assigned.unitarity_residual_max,
                "assigned_condition_number": assigned.condition_number,
                "assignment_max_energy_delta_mev": assigned.assignment_max_energy_delta,
            }
        )
        raw_matrices.append(raw)
        assigned_matrices.append(assigned.matrix)

    raw_wilson = raw_matrices[2] @ raw_matrices[1] @ raw_matrices[0]
    assigned_wilson = assigned_matrices[2] @ assigned_matrices[1] @ assigned_matrices[0]
    source_fixed = masks_by_shift[orbit_shifts[0]]
    return {
        "orbit_shifts": [list(shift) for shift in orbit_shifts],
        "edges": edges,
        "raw_wilson_full": _matrix_diagnostics(raw_wilson),
        "raw_wilson_blocks": _wilson_block_summary(raw_wilson, source_fixed),
        "assigned_wilson_full": _matrix_diagnostics(assigned_wilson),
        "assigned_wilson_blocks": _wilson_block_summary(
            assigned_wilson,
            source_fixed,
        ),
    }


def _build_wilson_summary(run, orbitals, common_padding: int):
    basis = build_periodic_gauge_basis_view(
        run,
        periodic_gauge_padding=common_padding,
        name="rlg_hbn_tdhf_anchor_independent_wilson",
    )
    fixed_indices = {
        int(pair[0]) * MESH + int(pair[1])
        for pair in run.basis_data.c3_fixed_representative_pairs
    }
    plus_orbit = ANCHORS
    minus_orbit = tuple(_minus_shift(shift) for shift in ANCHORS)
    return {
        "expected_pair_c3_cubed_phase": [1.0, 0.0],
        "expected_phase_basis": (
            "The accepted HF fixed-copy convention requires S20*S12*S01=+I; "
            "a common single-particle projective phase cancels in an intraflavor "
            "particle-hole pair. No fitted per-block phase is accepted."
        ),
        "plus_X": _build_independent_wilson_sector(
            run,
            orbitals,
            basis,
            plus_orbit,
            fixed_indices,
        ),
        "minus_Y": _build_independent_wilson_sector(
            run,
            orbitals,
            basis,
            minus_orbit,  # type: ignore[arg-type]
            fixed_indices,
        ),
    }


@pytest.fixture(scope="module")
def reduced_diagnostic() -> _ReducedDiagnostic:
    run = _build_reduced_run()
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    all_shifts = tuple(ANCHORS) + tuple(_minus_shift(shift) for shift in ANCHORS)
    common_padding = max(
        required_sparse_fixed_context_padding(
            run,
            orbitals,
            _intraflavor_pairs(run, orbitals, shift),
            shift,
        )
        for shift in all_shifts
    )
    anchor_summary = _build_anchor_summary(run, orbitals, common_padding)
    wilson_summary = _build_wilson_summary(run, orbitals, common_padding)
    payload = {
        "description": (
            "Reduced faithful fixed-quotient diagnostic. Cycle closure is not an "
            "anchor-independence or independent-C3^3 acceptance gate."
        ),
        "configuration": {
            "mesh": [MESH, MESH],
            "active_conduction_bands": 2,
            "anchors": [list(anchor) for anchor in ANCHORS],
            "physical_shifts": [list(shift) for shift in PHYSICAL_SHIFTS],
            "fixed_copy": 0,
            "periodic_gauge_padding": common_padding,
            "fixed_representative_pairs": [
                list(pair) for pair in run.basis_data.c3_fixed_representative_pairs
            ],
        },
        "anchor_independence": anchor_summary,
        "independent_wilson": wilson_summary,
    }
    output = os.environ.get("MEAN_FIELD_RLG_HBN_TDHF_ANCHOR_DIAG_JSON")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _ReducedDiagnostic(anchor_summary, wilson_summary)


def test_rlg_hbn_fixed_quotient_is_canonical_anchor_independent(
    reduced_diagnostic: _ReducedDiagnostic,
) -> None:
    summary = reduced_diagnostic.anchor_summary
    assert max(
        values["closure_residuals_mev"]["max"]
        for values in summary["anchors"].values()
    ) <= ANCHOR_TOLERANCE_MEV
    assert max(
        values["maximum_A_hermitian"]
        for values in summary["anchors"].values()
    ) <= ORDINARY_TOLERANCE_MEV
    assert max(
        values["maximum_B_transpose"]
        for values in summary["anchors"].values()
    ) <= ORDINARY_TOLERANCE_MEV
    maxima = summary["maxima"]
    assert maxima["term_ordinary_non_non_mev"] <= ORDINARY_TOLERANCE_MEV
    assert maxima["matrix_ordinary_non_non_mev"] <= ORDINARY_TOLERANCE_MEV
    assert maxima["liouvillian_ordinary_non_non_mev"] <= ORDINARY_TOLERANCE_MEV
    assert maxima["term_fixed_touched_mev"] <= ANCHOR_TOLERANCE_MEV
    assert maxima["matrix_fixed_touched_mev"] <= ANCHOR_TOLERANCE_MEV
    assert maxima["liouvillian_fixed_touched_mev"] <= ANCHOR_TOLERANCE_MEV
    assert maxima["spectrum_assignment_mev"] <= ANCHOR_TOLERANCE_MEV


def test_rlg_hbn_fixed_pair_sewing_has_independent_c3_cubed_representation(
    reduced_diagnostic: _ReducedDiagnostic,
) -> None:
    summary = reduced_diagnostic.wilson_summary
    violations: dict[str, float] = {}
    for sector_name in ("plus_X", "minus_Y"):
        sector = summary[sector_name]
        for edge_index, edge in enumerate(sector["edges"]):
            prefix = f"{sector_name}.edge_{edge_index}"
            violations[f"{prefix}.raw_ordinary_unitarity"] = edge["raw_ordinary"][
                "unitarity_defect"
            ]
            violations[f"{prefix}.raw_fixed_unitarity"] = edge["raw_fixed"][
                "unitarity_defect"
            ]
            violations[f"{prefix}.assigned_unitarity"] = edge[
                "assigned_unitarity_defect"
            ]
            violations[f"{prefix}.assigned_energy_delta"] = edge[
                "assignment_max_energy_delta_mev"
            ]
        for construction in ("raw_wilson_blocks", "assigned_wilson_blocks"):
            blocks = sector[construction]
            violations[f"{sector_name}.{construction}.ordinary_identity"] = blocks[
                "ordinary_identity_residual"
            ]
            violations[f"{sector_name}.{construction}.fixed_identity"] = blocks[
                "fixed_identity_residual"
            ]
            violations[f"{sector_name}.{construction}.cross_leakage"] = blocks[
                "cross_block_leakage"
            ]
    assert max(violations.values(), default=0.0) <= WILSON_TOLERANCE, violations
