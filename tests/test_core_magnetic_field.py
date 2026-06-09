from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from mean_field.core.magnetic_field import (
    MagneticFlux,
    choose_magnetic_nq,
    diophantine_branch_cases,
    diophantine_filling,
    magnetic_k_vectors,
    magnetic_normalization_count,
    magnetic_orbit_indices,
    magnetic_r_orbit_positions,
    magnetic_reciprocal_vector,
    magnetic_shell_shifts,
)
from mean_field.systems.tbg.finite_field import MagneticFlux as TBGMagneticFlux
from mean_field.systems.tbg.finite_field import paper_fig6_branch_cases


def test_magnetic_flux_normalizes_and_parses_values() -> None:
    assert MagneticFlux(2, 4) == MagneticFlux(1, 2)
    assert MagneticFlux.from_value(Fraction(2, 5)) == MagneticFlux(2, 5)
    assert MagneticFlux.from_value("1/12").ratio == pytest.approx(1.0 / 12.0)
    with pytest.raises(ValueError, match="denominator"):
        MagneticFlux(1, 0)


def test_magnetic_mesh_orbit_and_shell_helpers_are_system_agnostic() -> None:
    flux = MagneticFlux(1, 3)
    assert choose_magnetic_nq(3) == 4
    assert choose_magnetic_nq(7) == 2
    assert magnetic_normalization_count(flux, 2) == 36

    kvec = magnetic_k_vectors(g1=1.0 + 0.0j, g2=1.0j, flux=flux, nq=2)
    assert kvec.shape == (12,)
    np.testing.assert_allclose(kvec[:3], [0.0 + 0.0j, 1.0 / 3.0 + 0.0j, 2.0 / 3.0 + 0.0j])
    np.testing.assert_array_equal(magnetic_orbit_indices(3, 2)[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(magnetic_r_orbit_positions(2, 5), [0, 2, 4, 1, 3])

    assert magnetic_reciprocal_vector(1, 2, g1=1.0 + 0.0j, g2=1.0j, q=4) == 1.0 + 0.5j
    shifts = magnetic_shell_shifts(g1=1.0 + 0.0j, g2=np.exp(1j * np.pi / 3.0), q=2, shell_ng=1)
    assert (0, 0) in shifts
    assert shifts[0][0] == -1


def test_diophantine_filling_and_tbg_reexports_share_core_flux_type() -> None:
    assert TBGMagneticFlux is MagneticFlux
    assert diophantine_filling(-1, -3, "1/12") == pytest.approx(-1.25)
    branch = diophantine_branch_cases(-2, -2, fluxes=("1/2", "1/12"))
    assert branch[0] == (MagneticFlux(1, 2), -3.0)
    assert branch[-1] == (MagneticFlux(1, 12), pytest.approx(-13.0 / 6.0))

    tbg_branch = paper_fig6_branch_cases(-3, -1, fluxes=("1/12",))
    assert tbg_branch == ((MagneticFlux(1, 12), pytest.approx(-37.0 / 12.0)),)
