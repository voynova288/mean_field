"""Supplementary-Note-2 style generalized BCS update equations.

This module is the isolated ``paper_literal_legacy`` diagnostic.  It preserves
the equations printed on Supplementary pp.16--17, including their known
chemical-potential, finite-temperature xi-bracket, and eta-closure
inconsistencies.  Production ordinary-electron matrix HF lives in
``mean_field.systems.inas_gasb`` and must not import this scalar legacy path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .fermi import fermi_mev
from .thermodynamics import xi_hartree_fock_bracket

Array = NDArray[np.float64]
LegacyPhysicsMode = Literal["paper_literal_legacy"]
PAPER_LITERAL_LEGACY: LegacyPhysicsMode = "paper_literal_legacy"
PAPER_LITERAL_PHYSICS_STATUS = "source_literal_with_known_internal_inconsistencies"


def paper_literal_xi_bracket(
    xi_mev: Array,
    pair_energy_mev: Array,
    g_minus: Array,
    g_plus: Array,
) -> Array:
    """Return the finite-T xi bracket printed on Supplementary p.16."""

    return (1.0 - np.asarray(xi_mev) / np.asarray(pair_energy_mev)) * (
        1.0 - np.asarray(g_minus) - np.asarray(g_plus)
    )


def paper_literal_eta_factor(g_minus: Array, g_plus: Array) -> Array:
    """Return the factor ``1-g_-+g_+`` printed in the eta equation."""

    return 1.0 - np.asarray(g_minus) + np.asarray(g_plus)


ChemicalPotentialConvention = Literal[
    "literal_supplement",
    "eq1_common_carrier",
    "physical_electron_hole",
]


def chemical_potential_channels(
    mu_mev: float,
    convention: ChemicalPotentialConvention = "literal_supplement",
) -> tuple[float, float]:
    """Return doubled-convention ``(mu_sum, mu_diff)`` channels.

    For diagonal pair energies ``A=epsilon_a-mu_a`` and
    ``B=epsilon_b-mu_b``, the bare fields are
    ``xi=epsilon_P-(mu_a+mu_b)`` and
    ``eta=epsilon_M-(mu_a-mu_b)``.
    """

    mu = float(mu_mev)
    if convention == "literal_supplement":
        return mu, mu
    if convention == "eq1_common_carrier":
        return 2.0 * mu, 0.0
    if convention == "physical_electron_hole":
        return 0.0, 2.0 * mu
    raise ValueError(f"unsupported chemical-potential convention: {convention!r}")


def bare_xi_eta_channels(
    eps_P_mev: Array,
    eps_M_mev: Array,
    mu_sum_mev: float,
    mu_diff_mev: float,
) -> tuple[Array, Array]:
    """Bare doubled-convention fields for explicit sum/difference channels."""

    return (
        np.asarray(eps_P_mev, dtype=np.float64) - float(mu_sum_mev),
        np.asarray(eps_M_mev, dtype=np.float64) - float(mu_diff_mev),
    )


def carrier_occupations(
    Delta_order_mev: Array,
    xi_mev: Array,
    eta_mev: Array,
    temperature_K: float,
    *,
    energy_floor_meV: float = 1e-12,
) -> tuple[Array, Array]:
    """Per-spin electron and hole occupations in the doubled convention."""

    Delta = np.asarray(Delta_order_mev, dtype=np.float64)
    xi = np.asarray(xi_mev, dtype=np.float64)
    eta = np.asarray(eta_mev, dtype=np.float64)
    E_pair = np.sqrt(Delta * Delta + xi * xi)
    E_safe = np.maximum(E_pair, energy_floor_meV)
    f_minus = fermi_mev((E_pair - eta) / 2.0, temperature_K)
    f_plus = fermi_mev((E_pair + eta) / 2.0, temperature_K)
    u2 = 0.5 * (1.0 + xi / E_safe)
    v2 = 0.5 * (1.0 - xi / E_safe)
    n_a = u2 * f_plus + v2 * (1.0 - f_minus)
    n_b = u2 * f_minus + v2 * (1.0 - f_plus)
    return n_a, n_b


def carrier_densities_cm2(
    weights_nm2: Array,
    Delta_order_mev: Array,
    xi_mev: Array,
    eta_mev: Array,
    temperature_K: float,
    *,
    spin_degeneracy: float = 2.0,
    energy_floor_meV: float = 1e-12,
) -> tuple[float, float]:
    """Return electron and hole densities for the generalized BCS state."""

    w = np.asarray(weights_nm2, dtype=np.float64)
    n_a, n_b = carrier_occupations(
        Delta_order_mev,
        xi_mev,
        eta_mev,
        temperature_K,
        energy_floor_meV=energy_floor_meV,
    )
    scale = float(spin_degeneracy) * 1e14
    return float(scale * np.sum(w * n_a)), float(scale * np.sum(w * n_b))


def pair_density_cm2(
    weights_nm2: Array,
    Delta_order_mev: Array,
    xi_mev: Array,
    eta_mev: Array,
    temperature_K: float,
    *,
    spin_degeneracy: float = 2.0,
    energy_floor_meV: float = 1e-12,
) -> float:
    """Average electron/hole density; equals either density on the CNP branch."""

    n_e, n_h = carrier_densities_cm2(
        weights_nm2,
        Delta_order_mev,
        xi_mev,
        eta_mev,
        temperature_K,
        spin_degeneracy=spin_degeneracy,
        energy_floor_meV=energy_floor_meV,
    )
    return 0.5 * (n_e + n_h)


def update_generalized_bcs_channels(
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
    energy_floor_meV: float = 1e-12,
    xi_thermal_convention: str = "literal_paper",
    physics_mode: LegacyPhysicsMode = PAPER_LITERAL_LEGACY,
) -> tuple[Array, Array, Array]:
    """Paper-literal generalized-BCS update with explicit channel labels."""

    if physics_mode != PAPER_LITERAL_LEGACY:
        raise ValueError(
            "the radial scalar solver supports only paper_literal_legacy; "
            "use ordinary-electron matrix HF for production"
        )
    w = np.asarray(weights_nm2, dtype=np.float64)
    Delta = np.asarray(Delta_order_mev, dtype=np.float64)
    xi = np.asarray(xi_mev, dtype=np.float64)
    eta = np.asarray(eta_mev, dtype=np.float64)
    eps_P = np.asarray(eps_P_mev, dtype=np.float64)
    eps_M = np.asarray(eps_M_mev, dtype=np.float64)
    K_ab = np.asarray(K_ab_mev_nm2, dtype=np.float64)
    K_aa = np.asarray(K_aa_mev_nm2, dtype=np.float64)
    E_pair = np.sqrt(Delta * Delta + xi * xi)
    E_safe = np.maximum(E_pair, energy_floor_meV)
    f_minus = fermi_mev((E_pair - eta) / 2.0, temperature_K)
    f_plus = fermi_mev((E_pair + eta) / 2.0, temperature_K)
    F_pair = 1.0 - f_minus - f_plus
    Delta_rhs = K_ab @ (w * Delta / E_safe * F_pair)
    xi_bracket = xi_hartree_fock_bracket(
        xi,
        E_safe,
        eta,
        temperature_K,
        convention=xi_thermal_convention,
        energy_floor_meV=energy_floor_meV,
    )
    xi_rhs = eps_P - float(mu_sum_mev) - K_aa @ (w * xi_bracket)
    eta_rhs = eps_M - float(mu_diff_mev) - K_ab @ (w * (1.0 - f_minus + f_plus))
    return Delta_rhs, xi_rhs, eta_rhs


def update_generalized_bcs(
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
    energy_floor_meV: float = 1e-12,
    xi_thermal_convention: str = "literal_paper",
    physics_mode: LegacyPhysicsMode = PAPER_LITERAL_LEGACY,
) -> tuple[Array, Array, Array]:
    """Backward-compatible, explicitly isolated paper-literal update."""

    mu_sum, mu_diff = chemical_potential_channels(mu_mev, "literal_supplement")
    return update_generalized_bcs_channels(
        weights_nm2,
        Delta_order_mev,
        xi_mev,
        eta_mev,
        eps_P_mev,
        eps_M_mev,
        K_ab_mev_nm2,
        K_aa_mev_nm2,
        mu_sum,
        mu_diff,
        temperature_K,
        energy_floor_meV=energy_floor_meV,
        xi_thermal_convention=xi_thermal_convention,
        physics_mode=physics_mode,
    )


@dataclass(frozen=True)
class GeneralizedResult:
    k_nm_inv: Array
    weights_nm2: Array
    Delta_order_mev: Array
    xi_mev: Array
    eta_mev: Array
    E_pair_mev: Array
    eps_P_mev: Array
    eps_M_mev: Array
    mu_mev: float
    iterations: int
    residual_mev: float
    converged: bool
    mu_mode: str
    density_target_cm2: float | None = None
    density_final_cm2: float | None = None
    density_residual_cm2: float | None = None
    electron_density_cm2: float | None = None
    hole_density_cm2: float | None = None
    charge_imbalance_cm2: float | None = None
    eps_M_offset_mev: float | None = None
    alignment_iterations: int | None = None
    mu_iterations: int | None = None
    mu_sum_mev: float | None = None
    mu_diff_mev: float | None = None
    chemical_potential_convention: str = "literal_supplement"
    eps_M_frame: str = "input"
    physics_mode: str = PAPER_LITERAL_LEGACY
    physics_status: str = PAPER_LITERAL_PHYSICS_STATUS

    @property
    def transport_gap_proxy_mev(self) -> float:
        return float(np.min(self.E_pair_mev))

    @property
    def Delta_order_max_mev(self) -> float:
        return float(np.max(np.abs(self.Delta_order_mev)))

    @property
    def blocking_margin_mev(self) -> float:
        """Minimum quasiparticle blocking margin ``E_pair-|eta|``."""

        return float(np.min(self.E_pair_mev - np.abs(self.eta_mev)))


def solve_generalized_fixed_mu(
    k_nm_inv: Array,
    weights_nm2: Array,
    eps_P_mev: Array,
    eps_M_mev: Array,
    K_ab_mev_nm2: Array,
    K_aa_mev_nm2: Array,
    *,
    mu_mev: float = 0.0,
    temperature_K: float = 0.1,
    mix: float = 0.05,
    max_iter: int = 30000,
    tol_meV: float = 1e-8,
    energy_floor_meV: float = 1e-12,
    initial_delta_mev: Array | None = None,
    initial_xi_mev: Array | None = None,
    initial_eta_mev: Array | None = None,
    xi_thermal_convention: str = "literal_paper",
    chemical_potential_convention: ChemicalPotentialConvention = "literal_supplement",
) -> GeneralizedResult:
    """Solve Supplementary-Note-2 generalized equations at fixed chemical potential.

    This is a Stage-B building block.  It does not enforce the fixed-density
    constraint yet, so production paper claims must still state ``mu_mode=fixed_mu``
    unless a higher-level density loop is added.
    """

    k = np.asarray(k_nm_inv, dtype=np.float64)
    w = np.asarray(weights_nm2, dtype=np.float64)
    eps_P = np.asarray(eps_P_mev, dtype=np.float64)
    eps_M = np.asarray(eps_M_mev, dtype=np.float64)
    K_ab = np.asarray(K_ab_mev_nm2, dtype=np.float64)
    K_aa = np.asarray(K_aa_mev_nm2, dtype=np.float64)
    mu_sum, mu_diff = chemical_potential_channels(mu_mev, chemical_potential_convention)
    if K_ab.shape != (k.size, k.size) or K_aa.shape != (k.size, k.size):
        raise ValueError("kernel shapes must match the k grid")
    if not (0.0 < mix <= 1.0):
        raise ValueError("mix must lie in (0, 1]")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    Delta = (
        0.1 * np.exp(-0.5 * (k / max(float(np.max(k)), 1e-12)) ** 2)
        if initial_delta_mev is None
        else np.asarray(initial_delta_mev, dtype=np.float64).copy()
    )
    xi_bare, eta_bare = bare_xi_eta_channels(eps_P, eps_M, mu_sum, mu_diff)
    xi = xi_bare if initial_xi_mev is None else np.asarray(initial_xi_mev, dtype=np.float64).copy()
    eta = eta_bare if initial_eta_mev is None else np.asarray(initial_eta_mev, dtype=np.float64).copy()
    residual = np.inf
    converged = False
    for iteration in range(1, int(max_iter) + 1):
        Delta_rhs, xi_rhs, eta_rhs = update_generalized_bcs_channels(
            w,
            Delta,
            xi,
            eta,
            eps_P,
            eps_M,
            K_ab,
            K_aa,
            mu_sum,
            mu_diff,
            temperature_K,
            energy_floor_meV=energy_floor_meV,
            xi_thermal_convention=xi_thermal_convention,
        )
        Delta_new = (1.0 - mix) * Delta + mix * Delta_rhs
        xi_new = (1.0 - mix) * xi + mix * xi_rhs
        eta_new = (1.0 - mix) * eta + mix * eta_rhs
        if not (
            np.all(np.isfinite(Delta_new))
            and np.all(np.isfinite(xi_new))
            and np.all(np.isfinite(eta_new))
        ):
            raise FloatingPointError("non-finite generalized-BCS iterate")
        residual = float(
            max(
                np.max(np.abs(Delta_new - Delta)),
                np.max(np.abs(xi_new - xi)),
                np.max(np.abs(eta_new - eta)),
            )
        )
        Delta, xi, eta = Delta_new, xi_new, eta_new
        if residual < tol_meV:
            converged = True
            break
    E_pair = np.sqrt(Delta * Delta + xi * xi)
    electron_density, hole_density = carrier_densities_cm2(
        w,
        Delta,
        xi,
        eta,
        temperature_K,
        energy_floor_meV=energy_floor_meV,
    )
    density_final = 0.5 * (electron_density + hole_density)
    return GeneralizedResult(
        k,
        w,
        Delta,
        xi,
        eta,
        E_pair,
        eps_P,
        eps_M,
        float(mu_mev),
        iteration,
        residual,
        converged,
        "fixed_mu",
        density_final_cm2=density_final,
        electron_density_cm2=electron_density,
        hole_density_cm2=hole_density,
        charge_imbalance_cm2=electron_density - hole_density,
 mu_sum_mev=mu_sum,
 mu_diff_mev=mu_diff,
 chemical_potential_convention=chemical_potential_convention,
    )


def solve_generalized_fixed_mu_root(
    k_nm_inv: Array,
    weights_nm2: Array,
    eps_P_mev: Array,
    eps_M_mev: Array,
    K_ab_mev_nm2: Array,
    K_aa_mev_nm2: Array,
    *,
    mu_mev: float = 0.0,
    temperature_K: float = 0.1,
    initial_delta_mev: Array | None = None,
    initial_xi_mev: Array | None = None,
    initial_eta_mev: Array | None = None,
    tol_meV: float = 1e-7,
    max_iter: int = 500,
    energy_floor_meV: float = 1e-12,
    xi_thermal_convention: str = "literal_paper",
    chemical_potential_convention: ChemicalPotentialConvention = "literal_supplement",
) -> GeneralizedResult:
    """Solve the three Supplementary-Note-2 equations at fixed bands and mu.

    Unlike ``fixed_cnp_root``, this matrix-free root does not vary the band
    alignment or impose a number equation.  It is the closest literal
    implementation of Supplementary Note 2, which specifies a chemical
    potential but does not document a density constraint.
    """

    from scipy.optimize import root

    k = np.asarray(k_nm_inv, dtype=np.float64)
    w = np.asarray(weights_nm2, dtype=np.float64)
    eps_P = np.asarray(eps_P_mev, dtype=np.float64)
    eps_M = np.asarray(eps_M_mev, dtype=np.float64)
    n = k.size
    mu_sum, mu_diff = chemical_potential_channels(mu_mev, chemical_potential_convention)
    Delta0 = (
        np.full(n, 2.0, dtype=np.float64)
        if initial_delta_mev is None
        else np.asarray(initial_delta_mev, dtype=np.float64).copy()
    )
    xi_bare, eta_bare = bare_xi_eta_channels(eps_P, eps_M, mu_sum, mu_diff)
    xi0 = (
        xi_bare
        if initial_xi_mev is None
        else np.asarray(initial_xi_mev, dtype=np.float64).copy()
    )
    eta0 = (
        eta_bare
        if initial_eta_mev is None
        else np.asarray(initial_eta_mev, dtype=np.float64).copy()
    )
    if not (Delta0.shape == xi0.shape == eta0.shape == (n,)):
        raise ValueError("fixed-mu root seeds must match the k grid")
    x0 = np.concatenate([Delta0, xi0, eta0])

    def unpack(x: Array) -> tuple[Array, Array, Array]:
        return x[:n], x[n : 2 * n], x[2 * n : 3 * n]

    def residual_vector(x: Array) -> Array:
        Delta, xi, eta = unpack(np.asarray(x, dtype=np.float64))
        Delta_rhs, xi_rhs, eta_rhs = update_generalized_bcs_channels(
            w, Delta, xi, eta, eps_P, eps_M,
            K_ab_mev_nm2, K_aa_mev_nm2, mu_sum, mu_diff, temperature_K,
            energy_floor_meV=energy_floor_meV,
            xi_thermal_convention=xi_thermal_convention,
        )
        return np.concatenate([Delta - Delta_rhs, xi - xi_rhs, eta - eta_rhs])

    solution = root(
        residual_vector,
        x0,
        method="krylov",
        options={"fatol": float(tol_meV), "maxiter": int(max_iter), "disp": False},
    )
    Delta, xi, eta = unpack(np.asarray(solution.x, dtype=np.float64))
    equation_residual = float(np.max(np.abs(residual_vector(solution.x))))
    E_pair = np.sqrt(Delta * Delta + xi * xi)
    n_e, n_h = carrier_densities_cm2(
        w, Delta, xi, eta, temperature_K, energy_floor_meV=energy_floor_meV
    )
    return GeneralizedResult(
        k, w, Delta, xi, eta, E_pair, eps_P, eps_M, float(mu_mev),
        int(getattr(solution, "nit", max_iter)), equation_residual,
        bool(solution.success) and equation_residual <= tol_meV,
        "fixed_mu_root", density_final_cm2=0.5 * (n_e + n_h),
        electron_density_cm2=n_e, hole_density_cm2=n_h,
        charge_imbalance_cm2=n_e - n_h,
        mu_sum_mev=mu_sum, mu_diff_mev=mu_diff,
        chemical_potential_convention=chemical_potential_convention,
    )


def solve_generalized_fixed_density(
    k_nm_inv: Array,
    weights_nm2: Array,
    eps_P_mev: Array,
    eps_M_mev: Array,
    K_ab_mev_nm2: Array,
    K_aa_mev_nm2: Array,
    *,
    density_target_cm2: float,
    temperature_K: float = 0.1,
    mix: float = 0.05,
    max_iter: int = 30000,
    tol_meV: float = 1e-8,
    density_tol_cm2: float = 1e7,
    max_mu_iter: int = 32,
    mu_min_meV: float | None = None,
    mu_max_meV: float | None = None,
    energy_floor_meV: float = 1e-12,
    initial_delta_mev: Array | None = None,
    initial_xi_mev: Array | None = None,
    initial_eta_mev: Array | None = None,
    chemical_potential_convention: ChemicalPotentialConvention = "literal_supplement",
) -> GeneralizedResult:
    """Solve the generalized equations while tuning ``mu`` to a target density.

    The density checkpoint uses :func:`pair_density_cm2`, which is exact for the
    balanced zero-temperature BCS limit and is used here as the Stage-B CNP
    density constraint specified by the workdoc.  The routine deliberately
    reports both solver and density residuals so failed density closure cannot
    be hidden by a converged fixed-point iteration.
    """

    if density_target_cm2 <= 0:
        raise ValueError("density_target_cm2 must be positive")
    k = np.asarray(k_nm_inv, dtype=np.float64)
    w = np.asarray(weights_nm2, dtype=np.float64)
    eps_P = np.asarray(eps_P_mev, dtype=np.float64)
    eps_M = np.asarray(eps_M_mev, dtype=np.float64)
    span = max(20.0, float(np.nanmax(eps_P) - np.nanmin(eps_P)) + 10.0)
    lo = float(np.nanmin(eps_P) - span) if mu_min_meV is None else float(mu_min_meV)
    hi = float(np.nanmax(eps_P) + span) if mu_max_meV is None else float(mu_max_meV)

    def solve_at(mu: float, seed: GeneralizedResult | None = None) -> GeneralizedResult:
        return solve_generalized_fixed_mu(
            k,
            w,
            eps_P,
            eps_M,
            K_ab_mev_nm2,
            K_aa_mev_nm2,
            mu_mev=mu,
            temperature_K=temperature_K,
            mix=mix,
            max_iter=max_iter,
            tol_meV=tol_meV,
            energy_floor_meV=energy_floor_meV,
            initial_delta_mev=(seed.Delta_order_mev if seed is not None else initial_delta_mev),
            initial_xi_mev=(seed.xi_mev if seed is not None else initial_xi_mev),
            initial_eta_mev=(seed.eta_mev if seed is not None else initial_eta_mev),
            chemical_potential_convention=chemical_potential_convention,
        )

    r_lo = solve_at(lo)
    r_hi = solve_at(hi)
    f_lo = float(r_lo.density_final_cm2 or 0.0) - density_target_cm2
    f_hi = float(r_hi.density_final_cm2 or 0.0) - density_target_cm2
    expand = 0
    while f_lo * f_hi > 0.0 and expand < 8:
        span *= 2.0
        # Pair density is monotone increasing with chemical potential on a
        # continuous branch.  If both endpoints are below target, raise the
        # upper bound; if both are above, lower the lower bound.  The previous
        # absolute-residual heuristic could expand in the wrong direction and
        # make a valid density root impossible to bracket.
        if f_lo < 0.0 and f_hi < 0.0:
            hi += span
            r_hi = solve_at(hi, r_hi)
            f_hi = float(r_hi.density_final_cm2 or 0.0) - density_target_cm2
        elif f_lo > 0.0 and f_hi > 0.0:
            lo -= span
            r_lo = solve_at(lo, r_lo)
            f_lo = float(r_lo.density_final_cm2 or 0.0) - density_target_cm2
        expand += 1
    if f_lo * f_hi > 0.0:
        raise RuntimeError(
            "could not bracket fixed-density chemical potential: "
            f"mu=({lo:.6g},{hi:.6g}) residuals=({f_lo:.6g},{f_hi:.6g}) cm^-2"
        )

    best = r_lo if abs(f_lo) <= abs(f_hi) else r_hi
    best_resid = min(abs(f_lo), abs(f_hi))
    seed: GeneralizedResult | None = best
    mu_iterations = 0
    for mu_iterations in range(1, int(max_mu_iter) + 1):
        mid = 0.5 * (lo + hi)
        r_mid = solve_at(mid, seed)
        f_mid = float(r_mid.density_final_cm2 or 0.0) - density_target_cm2
        if abs(f_mid) < best_resid:
            best = r_mid
            best_resid = abs(f_mid)
            seed = r_mid
        if abs(f_mid) <= density_tol_cm2:
            break
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
            r_hi = r_mid
        else:
            lo = mid
            f_lo = f_mid
            r_lo = r_mid

    E_pair = np.sqrt(best.Delta_order_mev * best.Delta_order_mev + best.xi_mev * best.xi_mev)
    density_final = float(best.density_final_cm2 or pair_density_cm2(
        best.weights_nm2,
        best.Delta_order_mev,
        best.xi_mev,
        best.eta_mev,
        temperature_K,
        energy_floor_meV=energy_floor_meV,
    ))
    return GeneralizedResult(
        best.k_nm_inv,
        best.weights_nm2,
        best.Delta_order_mev,
        best.xi_mev,
        best.eta_mev,
        E_pair,
        best.eps_P_mev,
        best.eps_M_mev,
        best.mu_mev,
        best.iterations,
        best.residual_mev,
        best.converged and best_resid <= density_tol_cm2,
        "fixed_density",
        density_target_cm2=float(density_target_cm2),
        density_final_cm2=density_final,
        density_residual_cm2=density_final - float(density_target_cm2),
        electron_density_cm2=best.electron_density_cm2,
        hole_density_cm2=best.hole_density_cm2,
        charge_imbalance_cm2=best.charge_imbalance_cm2,
        mu_iterations=mu_iterations,
        mu_sum_mev=best.mu_sum_mev,
        mu_diff_mev=best.mu_diff_mev,
        chemical_potential_convention=best.chemical_potential_convention,
    )


def solve_generalized_normal_cnp_root(
    k_nm_inv: Array,
    weights_nm2: Array,
    eps_P_mev: Array,
    eps_M_mev: Array,
    K_ab_mev_nm2: Array,
    K_aa_mev_nm2: Array,
    *,
    density_target_cm2: float,
    temperature_K: float = 0.1,
    initial_mu_mev: float = 0.0,
    initial_eps_M_offset_mev: float = 0.0,
    initial_mu_sum_mev: float | None = None,
    initial_mu_diff_mev: float | None = None,
    initial_xi_mev: Array | None = None,
    initial_eta_mev: Array | None = None,
    tol_meV: float = 1e-7,
    density_tol_cm2: float = 1e8,
    max_iter: int = 500,
    energy_floor_meV: float = 1e-12,
) -> GeneralizedResult:
    """Solve the balanced normal-state CNP root with ``Delta`` fixed to zero."""

    from scipy.optimize import root

    k = np.asarray(k_nm_inv, dtype=np.float64)
    w = np.asarray(weights_nm2, dtype=np.float64)
    eps_P = np.asarray(eps_P_mev, dtype=np.float64)
    eps_M0 = np.asarray(eps_M_mev, dtype=np.float64)
    n = k.size
    mu_sum0 = float(initial_mu_mev if initial_mu_sum_mev is None else initial_mu_sum_mev)
    mu_diff0 = float(
        initial_mu_mev - initial_eps_M_offset_mev
        if initial_mu_diff_mev is None
        else initial_mu_diff_mev
    )
    xi0 = eps_P - mu_sum0 if initial_xi_mev is None else np.asarray(initial_xi_mev, dtype=np.float64)
    eta0 = np.zeros(n) if initial_eta_mev is None else np.asarray(initial_eta_mev, dtype=np.float64)
    x0 = np.concatenate([xi0, eta0, [mu_sum0, mu_diff0]])
    zero = np.zeros(n, dtype=np.float64)
    density_scale = max(float(density_target_cm2), 1.0)

    def unpack(x: Array) -> tuple[Array, Array, float, float]:
        return x[:n], x[n : 2 * n], float(x[-2]), float(x[-1])

    def residual_vector(x: Array) -> Array:
        xi, eta, mu_sum, mu_diff = unpack(np.asarray(x, dtype=np.float64))
        _Delta_rhs, xi_rhs, eta_rhs = update_generalized_bcs_channels(
            w, zero, xi, eta, eps_P, eps_M0,
            K_ab_mev_nm2, K_aa_mev_nm2, mu_sum, mu_diff, temperature_K,
            energy_floor_meV=energy_floor_meV,
        )
        n_e, n_h = carrier_densities_cm2(
            w, zero, xi, eta, temperature_K, energy_floor_meV=energy_floor_meV
        )
        return np.concatenate([
            xi - xi_rhs,
            eta - eta_rhs,
            [(n_e - density_target_cm2) / density_scale],
            [(n_h - density_target_cm2) / density_scale],
        ])

    solution = root(
        residual_vector, x0, method="krylov",
        options={"fatol": float(tol_meV), "maxiter": int(max_iter), "disp": False},
    )
    xi, eta, mu_sum, mu_diff = unpack(np.asarray(solution.x, dtype=np.float64))
    offset = mu_sum - mu_diff
    residual = residual_vector(solution.x)
    equation_residual = float(np.max(np.abs(residual[: 2 * n])))
    E_pair = np.abs(xi)
    n_e, n_h = carrier_densities_cm2(
        w, zero, xi, eta, temperature_K, energy_floor_meV=energy_floor_meV
    )
    pair_density = 0.5 * (n_e + n_h)
    converged = (
        bool(solution.success)
        and equation_residual <= tol_meV
        and abs(n_e - density_target_cm2) <= density_tol_cm2
        and abs(n_h - density_target_cm2) <= density_tol_cm2
    )
    return GeneralizedResult(
        k, w, zero, xi, eta, E_pair, eps_P, eps_M0, mu_sum,
        int(getattr(solution, "nit", max_iter)), equation_residual, converged,
        "normal_cnp_root", density_target_cm2=float(density_target_cm2),
        density_final_cm2=pair_density,
        density_residual_cm2=pair_density - float(density_target_cm2),
        electron_density_cm2=n_e, hole_density_cm2=n_h,
        charge_imbalance_cm2=n_e - n_h, eps_M_offset_mev=offset,
        alignment_iterations=int(getattr(solution, "nit", max_iter)),
        mu_sum_mev=mu_sum, mu_diff_mev=mu_diff,
        chemical_potential_convention="explicit_channels",
    )


def solve_generalized_cnp_root(
    k_nm_inv: Array,
    weights_nm2: Array,
    eps_P_mev: Array,
    eps_M_mev: Array,
    K_ab_mev_nm2: Array,
    K_aa_mev_nm2: Array,
    *,
    density_target_cm2: float,
    temperature_K: float = 0.1,
    initial_mu_mev: float = 0.0,
    initial_eps_M_offset_mev: float = 0.0,
    initial_mu_sum_mev: float | None = None,
    initial_mu_diff_mev: float | None = None,
    initial_delta_mev: Array | None = None,
    initial_xi_mev: Array | None = None,
    initial_eta_mev: Array | None = None,
    tol_meV: float = 1e-7,
    density_tol_cm2: float = 1e8,
    max_iter: int = 500,
    energy_floor_meV: float = 1e-12,
) -> GeneralizedResult:
    """Solve the coupled CNP equations as one matrix-free nonlinear root.

    The unknown vector is ``(Delta[N], xi[N], eta[N], mu_sum, mu_diff)``.
    Legacy ``mu`` and ``eps_M_offset`` inputs are mapped to these channels for
    restart compatibility. The final two residuals enforce ``n_e=n0`` and
    ``n_h=n0``. This diagnostic
    avoids following a discontinuous normal-state branch with nested scalar
    bisections.
    """

    from scipy.optimize import root

    k = np.asarray(k_nm_inv, dtype=np.float64)
    w = np.asarray(weights_nm2, dtype=np.float64)
    eps_P = np.asarray(eps_P_mev, dtype=np.float64)
    eps_M0 = np.asarray(eps_M_mev, dtype=np.float64)
    n = k.size
    mu_sum0 = float(initial_mu_mev if initial_mu_sum_mev is None else initial_mu_sum_mev)
    mu_diff0 = float(
        initial_mu_mev - initial_eps_M_offset_mev
        if initial_mu_diff_mev is None
        else initial_mu_diff_mev
    )
    Delta0 = (
        np.full(n, 2.0, dtype=np.float64)
        if initial_delta_mev is None
        else np.asarray(initial_delta_mev, dtype=np.float64).copy()
    )
    xi0 = (
        eps_P - mu_sum0
        if initial_xi_mev is None
        else np.asarray(initial_xi_mev, dtype=np.float64).copy()
    )
    eta0 = (
        np.zeros(n, dtype=np.float64)
        if initial_eta_mev is None
        else np.asarray(initial_eta_mev, dtype=np.float64).copy()
    )
    x0 = np.concatenate([Delta0, xi0, eta0, [mu_sum0, mu_diff0]])
    density_scale = max(float(density_target_cm2), 1.0)

    def unpack(x: Array) -> tuple[Array, Array, Array, float, float]:
        return x[:n], x[n : 2 * n], x[2 * n : 3 * n], float(x[-2]), float(x[-1])

    def residual_vector(x: Array) -> Array:
        Delta, xi, eta, mu_sum, mu_diff = unpack(np.asarray(x, dtype=np.float64))
        Delta_rhs, xi_rhs, eta_rhs = update_generalized_bcs_channels(
            w,
            Delta,
            xi,
            eta,
            eps_P,
            eps_M0,
            K_ab_mev_nm2,
            K_aa_mev_nm2,
            mu_sum,
            mu_diff,
            temperature_K,
            energy_floor_meV=energy_floor_meV,
        )
        n_e, n_h = carrier_densities_cm2(
            w,
            Delta,
            xi,
            eta,
            temperature_K,
            energy_floor_meV=energy_floor_meV,
        )
        return np.concatenate(
            [
                Delta - Delta_rhs,
                xi - xi_rhs,
                eta - eta_rhs,
                [(n_e - density_target_cm2) / density_scale],
                [(n_h - density_target_cm2) / density_scale],
            ]
        )

    solution = root(
        residual_vector,
        x0,
        method="krylov",
        options={"fatol": float(tol_meV), "maxiter": int(max_iter), "disp": False},
    )
    Delta, xi, eta, mu_sum, mu_diff = unpack(np.asarray(solution.x, dtype=np.float64))
    offset = mu_sum - mu_diff
    physical_residual = residual_vector(solution.x)
    equation_residual = float(np.max(np.abs(physical_residual[: 3 * n])))
    E_pair = np.sqrt(Delta * Delta + xi * xi)
    n_e, n_h = carrier_densities_cm2(
        w,
        Delta,
        xi,
        eta,
        temperature_K,
        energy_floor_meV=energy_floor_meV,
    )
    pair_density = 0.5 * (n_e + n_h)
    converged = (
        bool(solution.success)
        and equation_residual <= tol_meV
        and abs(n_e - density_target_cm2) <= density_tol_cm2
        and abs(n_h - density_target_cm2) <= density_tol_cm2
    )
    return GeneralizedResult(
        k,
        w,
        Delta,
        xi,
        eta,
        E_pair,
        eps_P,
        eps_M0,
        mu_sum,
        int(getattr(solution, "nit", max_iter)),
        equation_residual,
        converged,
        "fixed_cnp_root",
        density_target_cm2=float(density_target_cm2),
        density_final_cm2=pair_density,
        density_residual_cm2=pair_density - float(density_target_cm2),
        electron_density_cm2=n_e,
        hole_density_cm2=n_h,
        charge_imbalance_cm2=n_e - n_h,
        eps_M_offset_mev=offset,
        alignment_iterations=int(getattr(solution, "nit", max_iter)),
        mu_sum_mev=mu_sum,
        mu_diff_mev=mu_diff,
        chemical_potential_convention="explicit_channels",
    )


def solve_generalized_fixed_cnp(
    k_nm_inv: Array,
    weights_nm2: Array,
    eps_P_mev: Array,
    eps_M_mev: Array,
    K_ab_mev_nm2: Array,
    K_aa_mev_nm2: Array,
    *,
    density_target_cm2: float,
    temperature_K: float = 0.1,
    mix: float = 0.05,
    max_iter: int = 30000,
    tol_meV: float = 1e-8,
    density_tol_cm2: float = 1e7,
    charge_balance_tol_cm2: float = 1e7,
    max_mu_iter: int = 32,
    mu_min_meV: float | None = None,
    mu_max_meV: float | None = None,
    max_alignment_iter: int = 24,
    eps_M_offset_min_mev: float = -20.0,
    eps_M_offset_max_mev: float = 20.0,
    energy_floor_meV: float = 1e-12,
    initial_delta_mev: Array | None = None,
    initial_xi_mev: Array | None = None,
    initial_eta_mev: Array | None = None,
    chemical_potential_convention: ChemicalPotentialConvention = "literal_supplement",
) -> GeneralizedResult:
    """Enforce both CNP pair density and electron-hole charge balance.

    The inner ``mu`` loop closes ``n_e=n0``.  A second scalar offset added to
    ``epsilon_M=epsilon_a-epsilon_b`` represents the unresolved relative E1/H1
    electrostatic alignment and the outer loop closes ``n_h=n0``.  Average
    density and charge balance then follow automatically.  No Fig. 2
    observable enters this solve.
    """

    if charge_balance_tol_cm2 <= 0:
        raise ValueError("charge_balance_tol_cm2 must be positive")
    if eps_M_offset_min_mev >= eps_M_offset_max_mev:
        raise ValueError("eps_M_offset_min_mev must be less than eps_M_offset_max_mev")
    eps_M0 = np.asarray(eps_M_mev, dtype=np.float64)

    k = np.asarray(k_nm_inv, dtype=np.float64)
    eps_P = np.asarray(eps_P_mev, dtype=np.float64)

    def solve_offset(offset: float, seed: GeneralizedResult | None = None) -> GeneralizedResult:
        eps_M_shifted = eps_M0 + float(offset)
        span = max(20.0, float(np.nanmax(eps_P) - np.nanmin(eps_P)) + 10.0)
        mu_lo = float(np.nanmin(eps_P) - span) if mu_min_meV is None else float(mu_min_meV)
        mu_hi = float(np.nanmax(eps_P) + span) if mu_max_meV is None else float(mu_max_meV)

        def solve_mu(mu: float, inner_seed: GeneralizedResult | None = None) -> GeneralizedResult:
            use_seed = inner_seed if inner_seed is not None else seed
            return solve_generalized_fixed_mu(
                k,
                weights_nm2,
                eps_P,
                eps_M_shifted,
                K_ab_mev_nm2,
                K_aa_mev_nm2,
                mu_mev=mu,
                temperature_K=temperature_K,
                mix=mix,
                max_iter=max_iter,
                tol_meV=tol_meV,
                energy_floor_meV=energy_floor_meV,
                initial_delta_mev=(use_seed.Delta_order_mev if use_seed is not None else initial_delta_mev),
                initial_xi_mev=(use_seed.xi_mev if use_seed is not None else initial_xi_mev),
                initial_eta_mev=(use_seed.eta_mev if use_seed is not None else initial_eta_mev),
                chemical_potential_convention=chemical_potential_convention,
            )

        r_mu_lo = solve_mu(mu_lo)
        r_mu_hi = solve_mu(mu_hi)
        f_mu_lo = float(r_mu_lo.electron_density_cm2 or 0.0) - density_target_cm2
        f_mu_hi = float(r_mu_hi.electron_density_cm2 or 0.0) - density_target_cm2
        expand = 0
        while f_mu_lo * f_mu_hi > 0.0 and expand < 8:
            span *= 2.0
            if f_mu_lo < 0.0 and f_mu_hi < 0.0:
                mu_hi += span
                r_mu_hi = solve_mu(mu_hi, r_mu_hi)
                f_mu_hi = float(r_mu_hi.electron_density_cm2 or 0.0) - density_target_cm2
            elif f_mu_lo > 0.0 and f_mu_hi > 0.0:
                mu_lo -= span
                r_mu_lo = solve_mu(mu_lo, r_mu_lo)
                f_mu_lo = float(r_mu_lo.electron_density_cm2 or 0.0) - density_target_cm2
            expand += 1
        if f_mu_lo * f_mu_hi > 0.0:
            raise RuntimeError(
                "could not bracket electron-density chemical potential: "
                f"mu=({mu_lo:.6g},{mu_hi:.6g}) residuals=({f_mu_lo:.6g},{f_mu_hi:.6g}) cm^-2"
            )

        best_inner = r_mu_lo if abs(f_mu_lo) <= abs(f_mu_hi) else r_mu_hi
        best_inner_residual = min(abs(f_mu_lo), abs(f_mu_hi))
        mu_iterations = 0
        for mu_iterations in range(1, int(max_mu_iter) + 1):
            mu_mid = 0.5 * (mu_lo + mu_hi)
            r_mu_mid = solve_mu(mu_mid, best_inner)
            f_mu_mid = float(r_mu_mid.electron_density_cm2 or 0.0) - density_target_cm2
            if abs(f_mu_mid) < best_inner_residual:
                best_inner = r_mu_mid
                best_inner_residual = abs(f_mu_mid)
            if abs(f_mu_mid) <= density_tol_cm2:
                best_inner = r_mu_mid
                break
            if f_mu_lo * f_mu_mid <= 0.0:
                mu_hi, f_mu_hi, r_mu_hi = mu_mid, f_mu_mid, r_mu_mid
            else:
                mu_lo, f_mu_lo, r_mu_lo = mu_mid, f_mu_mid, r_mu_mid

        pair_density = 0.5 * (
            float(best_inner.electron_density_cm2 or 0.0)
            + float(best_inner.hole_density_cm2 or 0.0)
        )
        return replace(
            best_inner,
            converged=(best_inner.residual_mev <= tol_meV and best_inner_residual <= density_tol_cm2),
            mu_mode="fixed_cnp_inner",
            density_target_cm2=float(density_target_cm2),
            density_final_cm2=pair_density,
            density_residual_cm2=pair_density - float(density_target_cm2),
            mu_iterations=mu_iterations,
        )

    lo = float(eps_M_offset_min_mev)
    hi = float(eps_M_offset_max_mev)
    r_lo = solve_offset(lo)
    r_hi = solve_offset(hi)
    q_lo = float(r_lo.hole_density_cm2 or 0.0) - density_target_cm2
    q_hi = float(r_hi.hole_density_cm2 or 0.0) - density_target_cm2
    if q_lo * q_hi > 0.0:
        raise RuntimeError(
            "could not bracket hole-density epsilon_M offset: "
            f"offset=({lo:.6g},{hi:.6g}) residuals=({q_lo:.6g},{q_hi:.6g}) cm^-2"
        )

    def score(result: GeneralizedResult) -> float:
        electron_residual = abs(float(result.electron_density_cm2 or 0.0) - density_target_cm2)
        hole_residual = abs(float(result.hole_density_cm2 or 0.0) - density_target_cm2)
        return electron_residual / density_tol_cm2 + hole_residual / charge_balance_tol_cm2

    best_offset, best = (lo, r_lo) if score(r_lo) <= score(r_hi) else (hi, r_hi)
    alignment_iterations = 0
    for alignment_iterations in range(1, int(max_alignment_iter) + 1):
        mid = 0.5 * (lo + hi)
        seed = r_lo if abs(mid - lo) <= abs(hi - mid) else r_hi
        r_mid = solve_offset(mid, seed)
        q_mid = float(r_mid.hole_density_cm2 or 0.0) - density_target_cm2
        if score(r_mid) < score(best):
            best_offset, best = mid, r_mid
        if (
            abs(float(r_mid.electron_density_cm2 or 0.0) - density_target_cm2) <= density_tol_cm2
            and abs(q_mid) <= charge_balance_tol_cm2
            and r_mid.residual_mev <= tol_meV
        ):
            best_offset, best = mid, r_mid
            break
        if q_lo * q_mid <= 0.0:
            hi, q_hi, r_hi = mid, q_mid, r_mid
        else:
            lo, q_lo, r_lo = mid, q_mid, r_mid

    valid = (
        best.residual_mev <= tol_meV
        and abs(float(best.electron_density_cm2 or 0.0) - density_target_cm2) <= density_tol_cm2
        and abs(float(best.hole_density_cm2 or 0.0) - density_target_cm2) <= charge_balance_tol_cm2
    )
    return replace(
        best,
        converged=valid,
        mu_mode="fixed_cnp",
        eps_M_offset_mev=float(best_offset),
        alignment_iterations=alignment_iterations,
        eps_M_frame="aligned",
    )
