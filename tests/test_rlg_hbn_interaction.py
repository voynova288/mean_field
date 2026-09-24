from __future__ import annotations

import math

import numpy as np
import pytest

from mean_field.systems.RnG_hBN import (
    RLGhBNInteractionParams,
)
from mean_field.systems.RnG_hBN.interaction import (
    layer_coulomb_matrix_mev_nm2,
    layer_z_coordinates_nm,
    q0_interlayer_hartree_mev_nm2,
    screened_coulomb_2d_mev_nm2,
    screened_coulomb_layer_mev_nm2,
)


def test_layer_z_coordinates_are_centered_and_equally_spaced() -> None:
    z = layer_z_coordinates_nm(5, layer_spacing_nm=0.333)

    assert np.allclose(z, [-0.666, -0.333, 0.0, 0.333, 0.666])
    assert math.isclose(float(np.mean(z)), 0.0, abs_tol=1.0e-15)
    assert np.allclose(np.diff(z), 0.333)


def test_q0_interlayer_hartree_kernel_is_symmetric_with_zero_diagonal() -> None:
    z = layer_z_coordinates_nm(5, layer_spacing_nm=0.333)
    matrix = np.array(
        [[q0_interlayer_hartree_mev_nm2(float(zi), float(zj), epsilon_r=5.0) for zj in z] for zi in z]
    )

    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, -1] < matrix[0, 1] < 0.0


def test_layer_kernel_reduces_to_2d_kernel_for_coplanar_layers() -> None:
    q = 0.37
    layer_value = screened_coulomb_layer_mev_nm2(q, 0.0, 0.0, epsilon_r=6.0, gate_distance_nm=10.0)
    two_d_value = screened_coulomb_2d_mev_nm2(q, epsilon_r=6.0, gate_distance_nm=10.0)

    assert math.isclose(layer_value, two_d_value, rel_tol=1.0e-13)


def test_finite_q_layer_kernel_suppresses_opposite_outer_layers() -> None:
    interaction = RLGhBNInteractionParams(epsilon_r=5.0, gate_distance_nm=10.0)
    matrix = layer_coulomb_matrix_mev_nm2(1.0, 5, interaction)

    assert np.allclose(matrix, matrix.T)
    assert np.all(matrix > 0.0)
    assert matrix[0, -1] < matrix[2, 2]
    assert matrix[0, -1] < matrix[0, 0]


def test_2d_diagnostic_dimension_is_layer_independent() -> None:
    interaction = RLGhBNInteractionParams(interaction_dimension="2d_diagnostic")
    matrix = layer_coulomb_matrix_mev_nm2(0.5, 5, interaction)

    assert np.allclose(matrix, matrix[0, 0])


def test_interaction_params_validate_scheme_and_cutoffs() -> None:
    with pytest.raises(ValueError, match="scheme"):
        RLGhBNInteractionParams(scheme="graphene")

    with pytest.raises(ValueError, match="interaction_cutoff_q1"):
        RLGhBNInteractionParams(interaction_cutoff_q1=0.0)
