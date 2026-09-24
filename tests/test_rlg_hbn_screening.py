from __future__ import annotations

import math

import numpy as np

from mean_field.systems.RnG_hBN import (
    RLGhBNInteractionParams,
    RLGhBNModel,
)
from mean_field.systems.RnG_hBN.screening import (
    compute_valence_layer_charge,
    interlayer_hartree_potential_from_charge,
    moire_cell_area_nm2,
)
from mean_field.systems.RnG_hBN.interaction import (
    layer_z_coordinates_nm,
    q0_interlayer_hartree_mev_nm2,
)


def test_moire_cell_area_is_positive() -> None:
    model = RLGhBNModel.from_config(layer_count=3, xi=1, theta_deg=0.77, shell_count=1)

    assert moire_cell_area_nm2(model) > 0.0


def test_interlayer_hartree_potential_vanishes_for_reference_charge() -> None:
    model = RLGhBNModel.from_config(layer_count=5, xi=1, theta_deg=0.77, shell_count=1)
    interaction = RLGhBNInteractionParams(epsilon_r=5.0)
    delta = np.zeros(model.params.layer_count, dtype=float)

    result = interlayer_hartree_potential_from_charge(delta, model, interaction)

    assert np.allclose(result.layer_potential_mev, 0.0)
    assert math.isclose(result.interlayer_slope_mev, 0.0, abs_tol=1.0e-15)


def test_interlayer_hartree_slope_screens_outer_layer_charge_imbalance() -> None:
    model = RLGhBNModel.from_config(layer_count=5, xi=1, theta_deg=0.77, shell_count=1)
    interaction = RLGhBNInteractionParams(epsilon_r=5.0)
    delta = np.asarray([1.0, 0.0, 0.0, 0.0, -1.0], dtype=float)

    result = interlayer_hartree_potential_from_charge(delta, model, interaction)

    assert result.layer_potential_mev[0] > 0.0
    assert result.layer_potential_mev[-1] < 0.0
    assert result.interlayer_slope_mev < 0.0


def test_interlayer_hartree_scalar_projection_uses_appendix_b5_centered_layer_field() -> None:
    model = RLGhBNModel.from_config(layer_count=5, xi=1, theta_deg=0.77, shell_count=1)
    interaction = RLGhBNInteractionParams(epsilon_r=5.0)
    delta = np.asarray([1.0, 0.0, 0.0, 0.0, -1.0], dtype=float)

    result = interlayer_hartree_potential_from_charge(delta, model, interaction)

    z = layer_z_coordinates_nm(model.params.layer_count)
    raw_kernel = np.asarray(
        [
            [
                q0_interlayer_hartree_mev_nm2(float(z_l), float(z_lp), epsilon_r=interaction.epsilon_r)
                for z_lp in z
            ]
            for z_l in z
        ],
        dtype=float,
    )
    raw_layer_potential = raw_kernel @ delta / moire_cell_area_nm2(model)
    expected = (2.0 / float(model.params.layer_count)) * raw_layer_potential
    np.testing.assert_allclose(result.layer_potential_mev, expected)
    assert not np.allclose(result.layer_potential_mev, raw_layer_potential)


def test_valence_layer_charge_has_correct_total_against_average_reference() -> None:
    model = RLGhBNModel.from_config(layer_count=3, xi=1, theta_deg=0.77, shell_count=1)

    charge = compute_valence_layer_charge(model, mesh_size=1)

    expected_total = 2 * 2 * model.params.layer_count * model.lattice.n_g
    assert math.isclose(float(np.sum(charge.layer_charge)), float(expected_total), rel_tol=1.0e-12)
    assert math.isclose(float(np.sum(charge.reference_layer_charge)), float(expected_total), rel_tol=1.0e-12)
    assert math.isclose(float(np.sum(charge.delta_layer_charge)), 0.0, abs_tol=1.0e-9)
