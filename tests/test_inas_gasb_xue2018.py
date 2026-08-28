from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.inas_gasb.xue2018 import (
    xue2018_fig2_inferred_path,
    xue2018_physical_asymmetry_diagnostic_parameters,
    xue2018_standard_parameters,
)


def test_xue2018_standard_parameters_drop_eq6_zeta_particle_hole_asymmetry() -> None:
    params = xue2018_standard_parameters(eg_ry=0.1, hybridization_ab_ry=0.2)
    assert params.q_ab_inv == 0.0
    assert params.d_over_ab == pytest.approx(0.3)
    assert params.mass_e_over_reduced == pytest.approx(2.0)
    assert params.mass_h_over_reduced == pytest.approx(2.0)


def test_xue2018_physical_mass_asymmetry_is_explicitly_diagnostic_only() -> None:
    params = xue2018_physical_asymmetry_diagnostic_parameters(
        eg_ry=0.1, hybridization_ab_ry=0.2
    )
    # Since m is the reduced mass, m/m_e + m/m_h = 1.
    assert 1.0 / params.mass_e_over_reduced + 1.0 / params.mass_h_over_reduced == pytest.approx(1.0)
    assert params.mass_e_over_reduced == pytest.approx(1.0575)
    assert params.mass_h_over_reduced == pytest.approx(18.391304347826086)


def test_xue2018_fig2_inferred_path_has_preregistered_l_shape() -> None:
    path = xue2018_fig2_inferred_path()
    assert path.authority == "figure_geometry_inferred"
    assert np.array_equal(path.point_index, np.arange(1, 63))
    assert path.eg_ry.shape == (62,)
    assert path.hybridization_ab_ry.shape == (62,)
    assert np.allclose(path.hybridization_ab_ry[:22], 0.2)
    assert np.allclose(path.eg_ry[:22], np.linspace(1.6, -0.5, 22))
    assert np.allclose(path.eg_ry[21:], -0.5)
    assert np.allclose(path.hybridization_ab_ry[21:], np.linspace(0.2, 0.6, 41))
    eg, hybridization = path.select(path.anchor_indices())
    assert np.allclose(eg, [1.6, 0.6, -0.4, -0.5, -0.5, -0.5])
    assert np.allclose(hybridization, [0.2, 0.2, 0.2, 0.29, 0.39, 0.59])
