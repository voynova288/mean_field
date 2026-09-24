from __future__ import annotations

import numpy as np

from ei_mf.electrostatics import (
    balanced_capacitor_terms,
    poisson_electrostatic_energy,
    required_pair_bias_mev,
)


def test_balanced_capacitor_recovers_zhu_atomic_unit_corrections() -> None:
    rs, d_au = 2.66, 1.0
    a_nm, hartree_mev = 15.0, 6.0
    density_nm2 = 1.0 / (np.pi * rs**2 * a_nm**2)
    result = balanced_capacitor_terms(
        density_nm2 / 1e-14,
        d_au * a_nm,
        1.0,
        coulomb_mev_nm=hartree_mev * a_nm,
    )
    assert np.isclose(result.energy_per_pair_mev / hartree_mev, 2.0 * d_au / rs**2)
    assert np.isclose(result.pair_chemical_shift_mev / hartree_mev, 4.0 * d_au / rs**2)


def test_capacitor_chemical_shift_is_density_derivative() -> None:
    density = 5.5e10
    step = 1e5
    plus = balanced_capacitor_terms(density + step, 10.0, 15.0)
    minus = balanced_capacitor_terms(density - step, 10.0, 15.0)
    derivative_per_nm2 = (
        plus.energy_density_mev_nm2 - minus.energy_density_mev_nm2
    ) / (2.0 * step * 1e-14)
    center = balanced_capacitor_terms(density, 10.0, 15.0)
    assert np.isclose(derivative_per_nm2, center.pair_chemical_shift_mev, rtol=2e-10)


def test_fixed_density_pairing_comparison_cancels_capacitor_term() -> None:
    cap_ei = balanced_capacitor_terms(4.0411715653e10, 10.0, 15.0)
    cap_normal = balanced_capacitor_terms(4.0411715653e10, 10.0, 15.0)
    omega_ei_without_cap = 0.011
    omega_normal_without_cap = 0.0
    difference = (
        omega_ei_without_cap + cap_ei.energy_density_mev_nm2
        - omega_normal_without_cap - cap_normal.energy_density_mev_nm2
    )
    assert np.isclose(difference, omega_ei_without_cap - omega_normal_without_cap)


def test_poisson_energy_and_fixed_profile_chemical_shift_identity() -> None:
    z = np.linspace(-10.0, 10.0, 1001)
    ne = np.exp(-0.5 * ((z + 3.0) / 1.2) ** 2)
    nh = np.exp(-0.5 * ((z - 2.0) / 1.2) ** 2)
    target = 4e-4
    ne *= target / np.trapezoid(ne, z)
    nh *= target / np.trapezoid(nh, z)
    electron_potential = 0.2 * z
    result = poisson_electrostatic_energy(z, ne, nh, electron_potential)
    assert np.isclose(result.electron_density_nm2, target)
    assert np.isclose(result.hole_density_nm2, target)
    assert np.isclose(
        2.0 * result.energy_density_mev_nm2 / target,
        result.fixed_profile_pair_chemical_shift_mev,
    )


def test_separately_neutralized_layers_have_no_q0_capacitor_term() -> None:
    result = balanced_capacitor_terms(
        5.5e10, 10.0, 15.0, pair_bias_mev=3.0, separately_neutralized=True
    )
    assert result.energy_density_mev_nm2 == 0.0
    assert result.energy_per_pair_mev == 0.0
    assert result.pair_chemical_shift_mev == 0.0
    assert np.isclose(result.grand_density_contribution_mev_nm2, -3.0 * result.density_nm2)
    assert required_pair_bias_mev(1.25, 5.5e10, 10.0, 15.0, separately_neutralized=True) == 1.25
