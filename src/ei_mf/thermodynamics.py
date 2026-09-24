"""Thermodynamic consistency diagnostics for the generalized EI equations.

Du et al. Supplementary Note 2 prints a finite-temperature xi bracket
``(1-xi/E) * (1-f_- - f_+)``.  It is not the xi derivative of the same local
BdG grand-potential term that produces the printed Delta and eta equations.
This module makes that distinction explicit; it never silently replaces the
literal paper equation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .fermi import fermi_mev
from .units import KB_MEV_PER_K

Array = NDArray[np.float64]


def pair_thermal_factor(
    E_pair_mev: Array, eta_mev: Array, temperature_K: float
) -> Array:
    E = np.asarray(E_pair_mev, dtype=np.float64)
    eta = np.asarray(eta_mev, dtype=np.float64)
    return 1.0 - fermi_mev((E - eta) / 2.0, temperature_K) - fermi_mev(
        (E + eta) / 2.0, temperature_K
    )


def xi_hartree_fock_bracket(
    xi_mev: Array,
    E_pair_mev: Array,
    eta_mev: Array,
    temperature_K: float,
    *,
    convention: str,
    energy_floor_meV: float = 1e-12,
) -> Array:
    """Return the literal-paper or thermodynamic finite-T xi bracket."""

    xi = np.asarray(xi_mev, dtype=np.float64)
    E = np.maximum(np.asarray(E_pair_mev, dtype=np.float64), energy_floor_meV)
    F = pair_thermal_factor(E, eta_mev, temperature_K)
    if convention == "literal_paper":
        return (1.0 - xi / E) * F
    if convention == "thermodynamic":
        return 1.0 - (xi / E) * F
    raise ValueError("convention must be 'literal_paper' or 'thermodynamic'")


def literal_integrability_mismatch(
    Delta_order_mev: Array,
    xi_mev: Array,
    eta_mev: Array,
    temperature_K: float,
    *,
    energy_floor_meV: float = 1e-12,
) -> Array:
    """Cross-derivative mismatch of the literal Delta and xi local brackets.

    A scalar local potential would require equality of the Delta-xi cross
    derivatives.  For the printed equations their difference is
    ``-(Delta/E) * dF/dE``.  It vanishes at T=0 away from blocking edges, but
    is generically nonzero at finite temperature.
    """

    Delta = np.asarray(Delta_order_mev, dtype=np.float64)
    xi = np.asarray(xi_mev, dtype=np.float64)
    eta = np.asarray(eta_mev, dtype=np.float64)
    E = np.maximum(np.sqrt(Delta * Delta + xi * xi), energy_floor_meV)
    if temperature_K <= 0.0:
        return np.zeros_like(E)
    kBT = KB_MEV_PER_K * temperature_K
    f_minus = fermi_mev((E - eta) / 2.0, temperature_K)
    f_plus = fermi_mev((E + eta) / 2.0, temperature_K)
    dF_dE = (
        f_minus * (1.0 - f_minus) + f_plus * (1.0 - f_plus)
    ) / (2.0 * kBT)
    return -(Delta / E) * dF_dE


def _quadratic_kernel_term(
    field_mev: Array, weights_nm2: Array, kernel_mev_nm2: Array
) -> float:
    field = np.asarray(field_mev, dtype=np.float64)
    weights = np.asarray(weights_nm2, dtype=np.float64)
    kernel = np.asarray(kernel_mev_nm2, dtype=np.float64)
    operator = kernel * weights[None, :]
    response = np.linalg.solve(operator, field)
    return 0.5 * float(np.dot(weights * field, response))


def generalized_grand_potential_channels_mev_nm2(
    weights_nm2: Array,
    Delta_order_mev: Array,
    xi_mev: Array,
    eta_mev: Array,
    eps_P_mev: Array,
    eps_M_mev: Array,
    K_ab_mev_nm2: Array,
    K_aa_mev_nm2: Array,
    mu_sum_mev: float,
    mu_diff_mev: float,
    temperature_K: float,
    *,
    spin_degeneracy: float = 2.0,
) -> float:
    """Grand potential for the thermodynamically integrable convention.

    This functional yields the printed Delta and eta equations and the
    thermodynamic xi bracket ``1-(xi/E)F``.  It must not be used to rank roots
    of the literal finite-temperature xi equation, nor roots with different
    externally unspecified band-alignment offsets.
    """

    w = np.asarray(weights_nm2, dtype=np.float64)
    Delta = np.asarray(Delta_order_mev, dtype=np.float64)
    xi = np.asarray(xi_mev, dtype=np.float64)
    eta = np.asarray(eta_mev, dtype=np.float64)
    eps_P = np.asarray(eps_P_mev, dtype=np.float64)
    eps_M = np.asarray(eps_M_mev, dtype=np.float64)
    E = np.sqrt(Delta * Delta + xi * xi)
    e_minus = 0.5 * (E - eta)
    e_plus = 0.5 * (E + eta)
    if temperature_K <= 0.0:
        thermal = 2.0 * np.minimum(e_minus, 0.0) + 2.0 * np.minimum(e_plus, 0.0)
    else:
        kBT = KB_MEV_PER_K * temperature_K
        thermal = -2.0 * kBT * (
            np.logaddexp(0.0, -e_minus / kBT)
            + np.logaddexp(0.0, -e_plus / kBT)
        )
    local = xi + eta - E + thermal
    X = eps_P - float(mu_sum_mev) - xi
    Y = eps_M - float(mu_diff_mev) - eta
    omega_per_spin = (
        float(np.dot(w, local))
        + _quadratic_kernel_term(Delta, w, K_ab_mev_nm2)
        + _quadratic_kernel_term(X, w, K_aa_mev_nm2)
        + _quadratic_kernel_term(Y, w, K_ab_mev_nm2)
    )
    return float(spin_degeneracy) * omega_per_spin


def generalized_grand_potential_mev_nm2(
    weights_nm2: Array,
    Delta_order_mev: Array,
    xi_mev: Array,
    eta_mev: Array,
    eps_P_mev: Array,
    eps_M_mev: Array,
    K_ab_mev_nm2: Array,
    K_aa_mev_nm2: Array,
    mu_mev: float,
    temperature_K: float,
    *,
    spin_degeneracy: float = 2.0,
) -> float:
    """Backward-compatible literal-supplement grand-potential diagnostic."""

    return generalized_grand_potential_channels_mev_nm2(
        weights_nm2,
        Delta_order_mev,
        xi_mev,
        eta_mev,
        eps_P_mev,
        eps_M_mev,
        K_ab_mev_nm2,
        K_aa_mev_nm2,
        mu_mev,
        mu_mev,
        temperature_K,
        spin_degeneracy=spin_degeneracy,
    )
