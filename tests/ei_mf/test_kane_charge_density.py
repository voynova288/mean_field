from __future__ import annotations

import numpy as np

from ei_mf.kane_charge_density import (
    KaneRadialStates,
    charge_neutral_density,
    periodic_poisson_electron_energy_mev,
    radial_quadrature_weights,
)


def test_radial_endpoint_weights_have_correct_disk_area() -> None:
    k = np.linspace(0.0, 0.12, 61)
    weights = radial_quadrature_weights(k)
    assert np.all(weights >= 0.0)
    assert np.isclose(np.sum(weights), k[-1] ** 2 / (4.0 * np.pi))


def test_active_orbital_neutrality_counts_electrons_and_holes_separately() -> None:
    k = np.linspace(0.0, 0.12, 61)
    z = np.linspace(-2.0, 2.0, 9)
    nk = k.size
    energies = np.column_stack((1000.0 * k * k - 4.0, 4.0 - 1000.0 * k * k))
    rho_e = np.zeros((nk, 2, z.size))
    rho_h = np.zeros_like(rho_e)
    profile = np.exp(-0.5 * (z / 0.5) ** 2)
    profile /= np.trapezoid(profile, z)
    rho_e[:, 0, :] = profile
    rho_h[:, 1, :] = profile
    states = KaneRadialStates(k, energies, z, rho_e, rho_h, (1, 2), "synthetic", "synthetic")
    result = charge_neutral_density(states, temperature_K=0.0)
    assert abs(result.fermi_energy_mev) < 1e-10
    assert np.isclose(result.electron_density_cm2, result.hole_density_cm2, rtol=0, atol=1e-3)
    assert abs(result.charge_imbalance_cm2) < 1e-3


def test_periodic_poisson_zero_and_neutral_dipole_sources() -> None:
    z = np.arange(-10.0, 10.0, 0.5)
    eps = np.full_like(z, 15.0)
    zero = np.zeros_like(z)
    assert np.allclose(periodic_poisson_electron_energy_mev(z, zero, zero, eps), 0.0)

    ne = np.exp(-0.5 * ((z + 2.0) / 0.7) ** 2)
    nh = np.exp(-0.5 * ((z - 2.0) / 0.7) ** 2)
    ne *= 5e-4 / (np.sum(ne) * 0.5)
    nh *= 5e-4 / (np.sum(nh) * 0.5)
    potential = periodic_poisson_electron_energy_mev(z, ne, nh, eps)
    assert np.all(np.isfinite(potential))
    assert np.isclose(np.mean(potential), 0.0, atol=1e-12)
    assert np.ptp(potential) > 0.1
