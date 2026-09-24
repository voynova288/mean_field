"""Minimal q=0 electrostatics for a neutral electron-hole bilayer.

For two infinitesimally thin layers carrying opposite sheet charges ``+/- e n``
and separated by ``d``, the field is confined between the layers.  The
capacitor energy per area is

``U/A = 2*pi*[e^2/(4*pi*eps0*eps_r)]*d*n^2``.

This term is omitted when each layer is separately neutralized by a matching
background.  It changes the pair chemical potential and density selection,
but at fixed balanced density it is independent of the pairing state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .units import COULOMB_MEV_NM

Array = NDArray[np.float64]

CM2_TO_NM2 = 1e-14


@dataclass(frozen=True)
class BalancedCapacitorResult:
    density_cm2: float
    density_nm2: float
    d_nm: float
    eps_r: float
    energy_density_mev_nm2: float
    energy_per_pair_mev: float
    pair_chemical_shift_mev: float
    pair_bias_mev: float
    grand_density_contribution_mev_nm2: float


@dataclass(frozen=True)
class PoissonElectrostaticResult:
    energy_density_mev_nm2: float
    electron_density_nm2: float
    hole_density_nm2: float
    electron_weighted_potential_mev: float
    hole_weighted_potential_mev: float
    fixed_profile_pair_chemical_shift_mev: float


def poisson_electrostatic_energy(
    z_nm: Array,
    electron_density_nm3: Array,
    hole_density_nm3: Array,
    electron_potential_energy_mev: Array,
) -> PoissonElectrostaticResult:
    """Electrostatic energy from a solved 1D Poisson profile.

    If ``U_e=-e*phi`` is the electron potential energy and the positive charge
    number density is ``n_h-n_e``, then
    ``E/A = 1/2 int dz (n_e-n_h) U_e``.  The projected pair-chemical shift for
    fixed normalized z profiles is ``<U_e>_e-<U_e>_h = 2(E/A)/n`` on a balanced
    branch.
    """

    z = np.asarray(z_nm, dtype=np.float64)
    ne = np.asarray(electron_density_nm3, dtype=np.float64)
    nh = np.asarray(hole_density_nm3, dtype=np.float64)
    potential = np.asarray(electron_potential_energy_mev, dtype=np.float64)
    if not (z.ndim == ne.ndim == nh.ndim == potential.ndim == 1):
        raise ValueError("Poisson profiles must be one-dimensional")
    if not (z.size == ne.size == nh.size == potential.size and z.size >= 2):
        raise ValueError("Poisson profiles must have matching lengths")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z_nm must be strictly increasing")
    n_e = float(np.trapezoid(ne, z))
    n_h = float(np.trapezoid(nh, z))
    if n_e <= 0.0 or n_h <= 0.0:
        raise ValueError("integrated electron and hole densities must be positive")
    energy = 0.5 * float(np.trapezoid((ne - nh) * potential, z))
    u_e = float(np.trapezoid(ne * potential, z) / n_e)
    u_h = float(np.trapezoid(nh * potential, z) / n_h)
    return PoissonElectrostaticResult(
        energy_density_mev_nm2=energy,
        electron_density_nm2=n_e,
        hole_density_nm2=n_h,
        electron_weighted_potential_mev=u_e,
        hole_weighted_potential_mev=u_h,
        fixed_profile_pair_chemical_shift_mev=u_e - u_h,
    )


def balanced_capacitor_terms(
    density_cm2: float,
    d_nm: float,
    eps_r: float,
    *,
    pair_bias_mev: float = 0.0,
    separately_neutralized: bool = False,
    coulomb_mev_nm: float = COULOMB_MEV_NM,
) -> BalancedCapacitorResult:
    """Return q=0 capacitor and external-pair-bias terms.

    ``pair_bias_mev`` is the reservoir/gate energy gained per added neutral
    electron-hole pair, so its Legendre contribution is ``-bias*n``.  A
    density equilibrium would satisfy ``mu_internal + mu_cap = pair_bias``.
    The function intentionally accepts only a balanced density: unequal layer
    charges require an explicit gate/background geometry to define the fields.
    """

    if density_cm2 < 0.0:
        raise ValueError("density_cm2 must be non-negative")
    if d_nm < 0.0:
        raise ValueError("d_nm must be non-negative")
    if eps_r <= 0.0:
        raise ValueError("eps_r must be positive")
    if coulomb_mev_nm <= 0.0:
        raise ValueError("coulomb_mev_nm must be positive")

    n = float(density_cm2) * CM2_TO_NM2
    if separately_neutralized:
        energy_density = 0.0
        energy_per_pair = 0.0
        chemical_shift = 0.0
    else:
        coupling = float(coulomb_mev_nm) / float(eps_r)
        energy_density = 2.0 * 3.141592653589793 * coupling * float(d_nm) * n * n
        energy_per_pair = 0.0 if n == 0.0 else energy_density / n
        chemical_shift = 4.0 * 3.141592653589793 * coupling * float(d_nm) * n
    grand = energy_density - float(pair_bias_mev) * n
    return BalancedCapacitorResult(
        density_cm2=float(density_cm2),
        density_nm2=n,
        d_nm=float(d_nm),
        eps_r=float(eps_r),
        energy_density_mev_nm2=energy_density,
        energy_per_pair_mev=energy_per_pair,
        pair_chemical_shift_mev=chemical_shift,
        pair_bias_mev=float(pair_bias_mev),
        grand_density_contribution_mev_nm2=grand,
    )


def required_pair_bias_mev(
    internal_pair_chemical_potential_mev: float,
    density_cm2: float,
    d_nm: float,
    eps_r: float,
    *,
    separately_neutralized: bool = False,
) -> float:
    """Bias required to hold a balanced density in the minimal capacitor model."""

    cap = balanced_capacitor_terms(
        density_cm2,
        d_nm,
        eps_r,
        separately_neutralized=separately_neutralized,
    )
    return float(internal_pair_chemical_potential_mev) + cap.pair_chemical_shift_mev
