from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.RnG_hBN import RLGhBNModel
from mean_field.systems.RnG_hBN._hf_basis import (
    RLG_HBN_C3_AFFINE_SUPPORT_VERSION,
    _c3_affine_raw_pair,
    build_rlg_hbn_c3_affine_fixed_supports,
)


def _shell4_model() -> RLGhBNModel:
    return RLGhBNModel.from_config(
        layer_count=3,
        xi=1,
        theta_deg=0.77,
        displacement_field_mev=24.0,
        shell_count=4,
    )


def test_rlg_hbn_c3_affine_raw_pair_has_valley_signed_fixed_offset() -> None:
    assert _c3_affine_raw_pair((0, 0), (1, 1), valley=1) == (-1, -1)
    assert _c3_affine_raw_pair((0, 0), (1, 1), valley=-1) == (1, 1)
    assert _c3_affine_raw_pair((2, -1), (1, 0), valley=1) == (0, 3)
    assert _c3_affine_raw_pair((2, -1), (1, 0), valley=-1) == (2, 3)
    with pytest.raises(ValueError, match="Expected valley"):
        _c3_affine_raw_pair((0, 0), (1, 1), valley=0)


def test_rlg_hbn_c3_affine_fixed_support_closes_shell4_to_27_points() -> None:
    supports = build_rlg_hbn_c3_affine_fixed_supports(
        12, _shell4_model().lattice
    )
    assert tuple((value.mesh_pair, value.valley) for value in supports) == (
        ((4, 8), 1),
        ((4, 8), -1),
        ((8, 4), 1),
        ((8, 4), -1),
    )
    expected_shifts = {(4, 8): (1, 1), (8, 4): (1, 0)}

    by_key = {(value.mesh_pair, value.valley): value for value in supports}
    for value in supports:
        assert value.convention == RLG_HBN_C3_AFFINE_SUPPORT_VERSION
        assert value.representative_shift == expected_shifts[value.mesh_pair]
        assert len(value.seed_g_indices) == 19
        assert value.support_size == 27
        assert set(value.seed_g_indices).issubset(value.support_g_indices)

        permutation = np.asarray(value.c3_target_indices, dtype=int)
        np.testing.assert_array_equal(
            permutation[permutation[permutation]],
            np.arange(value.support_size),
        )
        for source_index, source_pair in enumerate(value.support_g_indices):
            expected_target = _c3_affine_raw_pair(
                source_pair,
                value.representative_shift,
                valley=value.valley,
            )
            assert (
                value.support_g_indices[permutation[source_index]]
                == expected_target
            )

    plus_48 = by_key[((4, 8), 1)]
    plus_48_extras = set(plus_48.support_g_indices).difference(
        plus_48.seed_g_indices
    )
    assert plus_48_extras == {
        (-3, -3),
        (-3, -2),
        (-3, -1),
        (-2, -3),
        (-1, -3),
        (0, -3),
        (1, -2),
        (2, -1),
    }
    for pair in ((4, 8), (8, 4)):
        plus = set(by_key[(pair, 1)].support_g_indices)
        minus = set(by_key[(pair, -1)].support_g_indices)
        assert minus == {(-value[0], -value[1]) for value in plus}
