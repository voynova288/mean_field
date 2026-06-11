from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf import (
    ComponentGroup,
    ProjectedWavefunctionBasis,
    build_projected_overlap_block_set,
    calculate_projected_overlap_between,
    calculate_projected_overlap_between_components,
    component_group_indices,
    mask_projected_wavefunctions_by_component_group,
)


def _two_component_basis() -> ProjectedWavefunctionBasis:
    wavefunctions = np.zeros((2, 1, 1, 1), dtype=np.complex128)
    wavefunctions[0, 0, 0, 0] = 1.0
    wavefunctions[1, 0, 0, 0] = 2.0
    return ProjectedWavefunctionBasis(
        wavefunctions,
        grid_shape=(1, 1),
        local_basis_size=2,
        component_groups=(
            ComponentGroup("a", np.asarray([0], dtype=int)),
            ComponentGroup("b", np.asarray([1], dtype=int)),
        ),
    )


def test_component_group_validation_rejects_duplicate_or_out_of_range_indices() -> None:
    wavefunctions = np.zeros((2, 1, 1, 1), dtype=np.complex128)
    with pytest.raises(ValueError, match="duplicate"):
        ComponentGroup("bad", np.asarray([0, 0], dtype=int))
    with pytest.raises(ValueError, match="indices must lie"):
        ProjectedWavefunctionBasis(
            wavefunctions,
            grid_shape=(1, 1),
            local_basis_size=2,
            component_groups=(ComponentGroup("bad", np.asarray([2], dtype=int)),),
        )


def test_component_group_indices_resolve_named_and_all_groups() -> None:
    basis = _two_component_basis()

    np.testing.assert_array_equal(component_group_indices(basis, "all"), [0, 1])
    np.testing.assert_array_equal(component_group_indices(basis, "a"), [0])
    np.testing.assert_array_equal(component_group_indices(basis, ComponentGroup("tmp", np.asarray([1]))), [1])

    with pytest.raises(KeyError, match="Unknown component group"):
        component_group_indices(basis, "missing")


def test_mask_projected_wavefunctions_by_component_group_zeroes_other_components() -> None:
    basis = _two_component_basis()
    masked = mask_projected_wavefunctions_by_component_group(basis, "a")

    assert masked[0, 0, 0, 0] == 1.0
    assert masked[1, 0, 0, 0] == 0.0


def test_build_projected_overlap_block_set_uses_component_groups() -> None:
    basis = _two_component_basis()

    blocks = build_projected_overlap_block_set(
        basis,
        shifts=((0, 0),),
        target_component_group="b",
        source_component_group="b",
        gvecs=np.asarray([1.0 + 0.0j]),
        hartree_screening={(0, 0): 2.0},
    )

    assert blocks.shifts == ((0, 0),)
    np.testing.assert_allclose(blocks.gvecs, [1.0 + 0.0j])
    assert blocks.hartree_screening == {(0, 0): 2.0}
    assert blocks.overlaps[(0, 0)][0, 0, 0, 0] == 4.0
    assert blocks.diagonal_overlaps[(0, 0)][0, 0, 0] == 4.0


def test_component_resolved_overlap_sums_to_full_overlap_for_disjoint_groups() -> None:
    basis = _two_component_basis()
    full = calculate_projected_overlap_between(basis, basis, 0, 0)
    aa = calculate_projected_overlap_between_components(
        basis, basis, 0, 0, target_component_group="a", source_component_group="a"
    )
    bb = calculate_projected_overlap_between_components(
        basis, basis, 0, 0, target_component_group="b", source_component_group="b"
    )
    ab = calculate_projected_overlap_between_components(
        basis, basis, 0, 0, target_component_group="a", source_component_group="b"
    )

    assert full.shape == (1, 1, 1, 1)
    assert full[0, 0, 0, 0] == 5.0
    assert aa[0, 0, 0, 0] == 1.0
    assert bb[0, 0, 0, 0] == 4.0
    assert ab[0, 0, 0, 0] == 0.0
    np.testing.assert_allclose(aa + bb, full)
