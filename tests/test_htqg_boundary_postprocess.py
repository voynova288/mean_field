from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.htqg.boundary_postprocess import (
    apply_physical_plane_wave_matched_valley_pauli,
    apply_raw_index_valley_pauli,
    build_physical_plane_wave_valley_matching,
    localize_degenerate_two_wall_subspaces,
    periodic_center_and_rms,
)
from mean_field.systems.htqg.microscopic_supercell import (
    MicroscopicBasisLayout,
    batch_a_supercell,
    build_batch_a_fold_map,
    build_batch_a_tr_covariant_support,
)


def _layout(g_indices: np.ndarray) -> MicroscopicBasisLayout:
    return MicroscopicBasisLayout(
        g_indices=np.asarray(g_indices, dtype=np.int64),
        valley_order=(1, -1),
        spin_order=("up", "down"),
        period_cells=4,
    )


def _support(layout: MicroscopicBasisLayout):
    fold_map = build_batch_a_fold_map(
        supercell=batch_a_supercell(),
        primitive_mesh=(8, 4),
        reduced_mesh=(2, 4),
    )
    return build_batch_a_tr_covariant_support(
        layout,
        fold_map,
        provenance="test-only boundary physical-label support",
    )


def test_two_wall_localization_rotates_only_exact_energy_cluster() -> None:
    root_two = np.sqrt(2.0)
    vectors = np.asarray(
        [
            [1.0 / root_two, 1.0 / root_two, 0.0],
            [1.0 / root_two, -1.0 / root_two, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    wall_difference = np.diag((1.0, -1.0, 0.2)).astype(np.complex128)
    result = localize_degenerate_two_wall_subspaces(
        np.asarray((0.0, 0.0, 1.0)),
        vectors,
        wall_difference,
        degeneracy_tolerance_ev=1.0e-12,
    )
    projected = result.vectors[:, :2].conj().T @ wall_difference @ result.vectors[:, :2]
    np.testing.assert_allclose(projected, np.diag((-1.0, 1.0)), atol=1.0e-14)
    np.testing.assert_array_equal(result.vectors[:, 2], vectors[:, 2])
    assert result.rotated_clusters == ((0, 1),)
    assert result.localization_operator == "Pi_wall_1_minus_Pi_wall_2"


def test_two_wall_localization_does_not_rotate_non_degenerate_states() -> None:
    vectors = np.eye(2, dtype=np.complex128)
    result = localize_degenerate_two_wall_subspaces(
        np.asarray((0.0, 2.0e-8)),
        vectors,
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        degeneracy_tolerance_ev=1.0e-9,
    )
    np.testing.assert_array_equal(result.vectors, vectors)
    assert result.rotated_clusters == ()


def test_periodic_center_is_undefined_for_zero_resultant() -> None:
    result = periodic_center_and_rms(
        np.asarray((0.0, 1.0, 2.0, 3.0)),
        np.ones(4),
        period=4.0,
        resultant_tolerance=1.0e-12,
    )
    assert result.status == "undefined_near_zero_resultant"
    assert result.center is None
    assert result.rms_distance is None
    assert result.resultant_magnitude <= 1.0e-12


def test_periodic_center_and_rms_are_defined_for_localized_weight() -> None:
    result = periodic_center_and_rms(
        np.asarray((3.9, 0.0, 0.1)),
        np.asarray((1.0, 2.0, 1.0)),
        period=4.0,
        resultant_tolerance=1.0e-8,
    )
    assert result.status == "defined"
    assert result.center is not None
    assert min(abs(result.center), abs(result.center - 4.0)) < 1.0e-14
    assert result.rms_distance is not None
    np.testing.assert_allclose(result.rms_distance, np.sqrt(0.005), atol=1.0e-14)


@pytest.mark.parametrize("bad_reduced_k", (True, 0.0, "0"))
def test_physical_valley_matching_requires_exact_integer_k(bad_reduced_k: object) -> None:
    layout = _layout(np.asarray(((0, 0),)))
    with pytest.raises(TypeError, match="exact integer"):
        build_physical_plane_wave_valley_matching(
            layout,
            _support(layout),
            reduced_k=bad_reduced_k,  # type: ignore[arg-type]
        )


def test_physical_valley_operator_rejects_component_before_incomplete_map() -> None:
    layout = _layout(np.asarray(((-1, 0), (0, 0), (1, 0))))
    with pytest.raises(ValueError, match="component"):
        apply_physical_plane_wave_matched_valley_pauli(
            np.zeros(layout.dimension, dtype=np.complex128),
            layout,
            _support(layout),
            "z",  # type: ignore[arg-type]
            reduced_k=0,
            require_complete=False,
        )


def test_physical_plane_wave_valley_operator_is_not_raw_same_g_pauli() -> None:
    layout = _layout(np.asarray(((-1, 0), (0, 0), (1, 0))))
    support = _support(layout)
    matching = build_physical_plane_wave_valley_matching(
        layout,
        support,
        reduced_k=0,
    )
    assert not matching.complete
    assert matching.matched_count > 0

    pair: tuple[int, int] | None = None
    for k_index, kp_index in enumerate(matching.partner_index):
        if kp_index < 0:
            continue
        k_labels = layout.unravel_index(k_index)
        kp_labels = layout.unravel_index(int(kp_index))
        if k_labels[4] == 0 and k_labels[:2] != kp_labels[:2]:
            pair = (k_index, int(kp_index))
            break
    assert pair is not None
    k_index, kp_index = pair
    state = np.zeros(layout.dimension, dtype=np.complex128)
    state[k_index] = 1.0 / np.sqrt(2.0)
    state[kp_index] = 1.0 / np.sqrt(2.0)

    raw_x = apply_raw_index_valley_pauli(state, layout, "x")
    with np.testing.assert_raises_regex(ValueError, "matching is incomplete"):
        apply_physical_plane_wave_matched_valley_pauli(
            state,
            layout,
            support,
            "x",
            reduced_k=0,
        )
    physical_x = apply_physical_plane_wave_matched_valley_pauli(
        state,
        layout,
        support,
        "x",
        reduced_k=0,
        require_complete=False,
    )
    np.testing.assert_allclose(np.vdot(state, raw_x), 0.0, atol=1.0e-15)
    np.testing.assert_allclose(np.vdot(state, physical_x), 1.0, atol=1.0e-15)
