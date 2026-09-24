from __future__ import annotations

import numpy as np
import pytest

from ei_mf.model_interface import (
    aligned_pair_energy_mev,
    fit_constant_relative_alignment,
)


def test_constant_alignment_exact_offset() -> None:
    normal = np.array([-4.0, -1.0, 2.0, 5.0])
    target = normal + 3.25
    result = fit_constant_relative_alignment(normal, target)
    assert np.isclose(result.delta_ls_mev, 3.25)
    assert np.isclose(result.delta_minimax_mev, 3.25)
    assert np.allclose(result.residual_ls_mev, 0.0)
    assert np.isclose(result.minimax_lower_bound_mev, 0.0)


def test_minimax_alignment_has_weight_independent_range_bound() -> None:
    normal = np.zeros(3)
    target = np.array([2.0, 4.0, 8.0])
    result = fit_constant_relative_alignment(
        normal, target, weights=np.array([100.0, 1.0, 1.0])
    )
    assert result.delta_ls_mev < 3.0
    assert np.isclose(result.delta_minimax_mev, 5.0)
    assert np.isclose(result.minimax_lower_bound_mev, 3.0)
    assert np.isclose(np.max(np.abs(result.residual_minimax_mev)), 3.0)


def test_relative_block_shift_changes_signed_detuning_by_delta() -> None:
    h = np.diag([-3.0, -2.0, 4.0, 6.0])
    projector_difference = np.diag([1.0, 1.0, -1.0, -1.0])
    delta = 2.4
    shifted = h + 0.5 * delta * projector_difference
    g0 = np.trace(h[:2, :2]) / 2.0 - np.trace(h[2:, 2:]) / 2.0
    g1 = np.trace(shifted[:2, :2]) / 2.0 - np.trace(shifted[2:, 2:]) / 2.0
    assert np.isclose(g1 - g0, delta)


def test_alignment_fit_rejects_nonfinite_weights() -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_constant_relative_alignment(
            np.zeros(2), np.ones(2), weights=np.array([1.0, np.nan])
        )


def test_aligned_pair_energy_uses_paper_doubled_convention() -> None:
    normal = np.array([-3.0, 0.0, 4.0])
    order = np.array([4.0, 2.0, 3.0])
    energy = aligned_pair_energy_mev(normal, 0.0, order)
    assert np.allclose(energy, [5.0, 2.0, 5.0])
