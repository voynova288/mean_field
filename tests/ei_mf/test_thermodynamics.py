from __future__ import annotations

import numpy as np

from ei_mf.fermi import fermi_mev
from ei_mf.thermodynamics import (
    generalized_grand_potential_channels_mev_nm2,
    generalized_grand_potential_mev_nm2,
    literal_integrability_mismatch,
    pair_thermal_factor,
    xi_hartree_fock_bracket,
)


def test_literal_grand_potential_wrapper_matches_equal_channels() -> None:
    w = np.array([0.2])
    Delta = np.array([0.7])
    xi = np.array([0.4])
    eta = np.array([0.15])
    eps_P = np.array([0.9])
    eps_M = np.array([0.5])
    Kab = np.array([[5.0]])
    Kaa = np.array([[2.0]])
    old = generalized_grand_potential_mev_nm2(
        w, Delta, xi, eta, eps_P, eps_M, Kab, Kaa, 0.1, 3.0
    )
    channels = generalized_grand_potential_channels_mev_nm2(
        w, Delta, xi, eta, eps_P, eps_M, Kab, Kaa, 0.1, 0.1, 3.0
    )
    assert old == channels


def test_literal_integrability_mismatch_matches_cross_finite_difference() -> None:
    Delta, xi, eta, temperature = 0.7, 0.4, 0.15, 3.0
    step = 1e-6

    def gap_local(d: float, x: float) -> float:
        E = np.hypot(d, x)
        return -d / E * pair_thermal_factor(np.array([E]), np.array([eta]), temperature)[0]

    def xi_literal(d: float, x: float) -> float:
        E = np.hypot(d, x)
        return xi_hartree_fock_bracket(
            np.array([x]), np.array([E]), np.array([eta]), temperature,
            convention="literal_paper",
        )[0]

    cross_numeric = (
        (gap_local(Delta, xi + step) - gap_local(Delta, xi - step)) / (2 * step)
        - (xi_literal(Delta + step, xi) - xi_literal(Delta - step, xi)) / (2 * step)
    )
    cross_exact = literal_integrability_mismatch(
        np.array([Delta]), np.array([xi]), np.array([eta]), temperature
    )[0]
    assert np.isclose(cross_numeric, cross_exact, rtol=2e-6, atol=2e-8)
    assert abs(cross_exact) > 1e-4


def test_literal_and_thermodynamic_xi_brackets_agree_only_in_unblocked_limit() -> None:
    xi = np.array([0.8])
    E = np.array([1.0])
    eta = np.array([0.0])
    literal_zero = xi_hartree_fock_bracket(
        xi, E, eta, 0.0, convention="literal_paper"
    )
    thermo_zero = xi_hartree_fock_bracket(
        xi, E, eta, 0.0, convention="thermodynamic"
    )
    assert np.allclose(literal_zero, thermo_zero)
    literal_finite = xi_hartree_fock_bracket(
        xi, E, eta, 5.0, convention="literal_paper"
    )
    thermo_finite = xi_hartree_fock_bracket(
        xi, E, eta, 5.0, convention="thermodynamic"
    )
    assert not np.allclose(literal_finite, thermo_finite)


def test_thermodynamic_grand_potential_gradient_gives_consistent_brackets() -> None:
    w = np.array([0.2])
    Delta = np.array([0.7])
    xi = np.array([0.4])
    eta = np.array([0.15])
    eps_P = np.array([0.9])
    eps_M = np.array([0.5])
    Kab = np.array([[5.0]])
    Kaa = np.array([[2.0]])
    mu = 0.1
    temperature = 3.0
    spin = 2.0

    def omega(d: float, x: float, h: float) -> float:
        return generalized_grand_potential_mev_nm2(
            w, np.array([d]), np.array([x]), np.array([h]),
            eps_P, eps_M, Kab, Kaa, mu, temperature,
            spin_degeneracy=spin,
        )

    step = 1e-6
    numeric = np.array([
        (omega(Delta[0] + step, xi[0], eta[0]) - omega(Delta[0] - step, xi[0], eta[0])) / (2 * step),
        (omega(Delta[0], xi[0] + step, eta[0]) - omega(Delta[0], xi[0] - step, eta[0])) / (2 * step),
        (omega(Delta[0], xi[0], eta[0] + step) - omega(Delta[0], xi[0], eta[0] - step)) / (2 * step),
    ])
    E = np.hypot(Delta[0], xi[0])
    F = pair_thermal_factor(np.array([E]), eta, temperature)[0]
    f_minus = fermi_mev((E - eta[0]) / 2, temperature)
    f_plus = fermi_mev((E + eta[0]) / 2, temperature)
    response_D = Delta[0] / (Kab[0, 0] * w[0])
    response_X = (eps_P[0] - mu - xi[0]) / (Kaa[0, 0] * w[0])
    response_Y = (eps_M[0] - mu - eta[0]) / (Kab[0, 0] * w[0])
    analytic_per_spin_weight = np.array([
        -Delta[0] / E * F + response_D,
        1.0 - xi[0] / E * F - response_X,
        1.0 - f_minus + f_plus - response_Y,
    ])
    analytic = spin * w[0] * analytic_per_spin_weight
    assert np.allclose(numeric, analytic, rtol=2e-6, atol=2e-8)
