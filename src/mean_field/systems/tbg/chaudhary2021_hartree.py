from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh, eigvalsh

from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field.model import _generate_gvec, _generate_t12, _generate_t12_zero_fill

from mean_field.systems.tbg.chaudhary2021 import ChaudharyTBGConfig, build_chau_b0_hamiltonian

E2_OVER_4PI_EPS0_EV_NM = 1.4399645483
KB_EV_PER_K = 8.617333262145e-5
FIRST_STAR_SHIFTS: frozenset[tuple[int, int]] = frozenset({(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)})


@dataclass(frozen=True)
class HartreeSCFResult:
    rho_q: dict[tuple[int, int], complex]
    iterations: int
    converged: bool
    final_error: float
    mu_ev: float
    iter_error: np.ndarray
    iter_mu_ev: np.ndarray
    nu: float
    epsilon_r: float
    mixing: float
    mesh_size: int
    lg: int
    temperature_k: float = 0.0
    density_mode: str = "flat"
    hartree_shift_mode: str = "all"


def canonical_hartree_shift_mode(mode: str) -> str:
    text = str(mode).lower().replace("-", "_")
    if text in {"all", "full", "all_shifts"}:
        return "all"
    if text in {"first", "first_star", "firststar", "first_shell"}:
        return "first_star"
    raise ValueError(f"Unsupported hartree_shift_mode={mode!r}; expected all or first_star")


def filter_rho_by_shift_mode(
    rho_q: dict[tuple[int, int], complex],
    mode: str,
) -> dict[tuple[int, int], complex]:
    canonical = canonical_hartree_shift_mode(mode)
    if canonical == "all":
        return dict(rho_q)
    return {shift: value for shift, value in rho_q.items() if tuple(shift) in FIRST_STAR_SHIFTS}


def b0_g_coords(lg: int) -> np.ndarray:
    half = int(lg) // 2
    vals = np.arange(-half, half + 1, dtype=int)
    xx, yy = np.meshgrid(vals, vals, indexing="ij")
    return np.column_stack([xx.ravel(order="F"), yy.ravel(order="F")]).astype(int)


def moire_cell_area_nm2(params: TBGParameters, config: ChaudharyTBGConfig) -> float:
    area_dimless = abs((complex(params.a1).conjugate() * complex(params.a2)).imag)
    return float(area_dimless * float(config.graphene_lattice_constant_nm) ** 2)


def hartree_kernel_ev(params: TBGParameters, config: ChaudharyTBGConfig, shift: tuple[int, int], epsilon_r: float) -> float:
    if int(shift[0]) == 0 and int(shift[1]) == 0:
        return 0.0
    q_dimless = complex(int(shift[0]) * params.g1 + int(shift[1]) * params.g2)
    q_nm = abs(q_dimless) / float(config.graphene_lattice_constant_nm)
    if q_nm <= 0.0:
        return 0.0
    area = moire_cell_area_nm2(params, config)
    return float(2.0 * math.pi * E2_OVER_4PI_EPS0_EV_NM / (float(epsilon_r) * q_nm * area))


def occupations_for_count(
    energies: np.ndarray,
    target_count: float,
    *,
    tol_ev: float = 1.0e-10,
    temperature_k: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Occupations for a target particle count.

    At ``temperature_k=0`` this uses a sharp Fermi surface with equal sharing
    inside a numerically degenerate Fermi shell.  At finite temperature it uses
    Fermi-Dirac occupations and solves for the chemical potential by bisection;
    this is useful for the high-temperature Hartree regime discussed by
    Chaudhary et al. and avoids artificial density oscillations from coarse
    k-mesh shell crossings.
    """

    flat = np.asarray(energies, dtype=float).reshape(-1)
    n = flat.size
    count = float(target_count)
    shape = np.shape(energies)
    if count <= 0.0:
        return np.zeros_like(flat, dtype=float).reshape(shape), float(np.min(flat) - 1.0e-6)
    if count >= n:
        return np.ones_like(flat, dtype=float).reshape(shape), float(np.max(flat) + 1.0e-6)

    kbt = KB_EV_PER_K * float(temperature_k)
    if kbt > 0.0:
        lo = float(np.min(flat) - 80.0 * kbt - 1.0e-3)
        hi = float(np.max(flat) + 80.0 * kbt + 1.0e-3)
        mu = 0.5 * (lo + hi)
        occ = np.zeros_like(flat, dtype=float)
        for _ in range(160):
            mu = 0.5 * (lo + hi)
            x = np.clip((flat - mu) / kbt, -80.0, 80.0)
            occ = 1.0 / (np.exp(x) + 1.0)
            if float(np.sum(occ)) < count:
                lo = mu
            else:
                hi = mu
        return occ.reshape(shape), float(mu)

    order = np.argsort(flat, kind="stable")
    occ = np.zeros(n, dtype=float)
    whole = int(math.floor(count + 1.0e-12))
    if whole > 0:
        occ[order[:whole]] = 1.0
    remainder = count - whole
    if remainder <= 1.0e-12:
        mu_idx = min(max(whole, 1), n - 1)
        mu = 0.5 * (flat[order[mu_idx - 1]] + flat[order[mu_idx]])
        return occ.reshape(shape), float(mu)

    fermi_energy = flat[order[whole]]
    # Remove any states in the same numerical shell from the fully filled set,
    # then share the total shell occupancy equally.  This avoids artificial
    # valley/spin polarization from exactly degenerate +/- valley states.
    shell = np.flatnonzero(np.abs(flat - fermi_energy) <= float(tol_ev))
    below = np.flatnonzero(flat < fermi_energy - float(tol_ev))
    occ[:] = 0.0
    occ[below] = 1.0
    shell_fill = (count - below.size) / max(1, shell.size)
    shell_fill = min(1.0, max(0.0, shell_fill))
    occ[shell] = shell_fill
    return occ.reshape(shape), float(fermi_energy)


def build_hartree_matrix_from_rho(
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    *,
    lg: int,
    rho_q: dict[tuple[int, int], complex],
    epsilon_r: float,
) -> np.ndarray:
    ng = int(lg) * int(lg)
    dim = 4 * ng
    coords = b0_g_coords(int(lg))
    out = np.zeros((dim, dim), dtype=np.complex128)
    kernel_cache: dict[tuple[int, int], float] = {}
    for i in range(ng):
        ci = coords[i]
        for j in range(ng):
            shift = (int(ci[0] - coords[j, 0]), int(ci[1] - coords[j, 1]))
            rho = complex(rho_q.get(shift, 0.0j))
            if rho == 0.0j:
                continue
            kernel = kernel_cache.get(shift)
            if kernel is None:
                kernel = hartree_kernel_ev(params, config, shift, float(epsilon_r))
                kernel_cache[shift] = kernel
            value = kernel * rho
            if value == 0.0j:
                continue
            base_i = 4 * i
            base_j = 4 * j
            for internal in range(4):
                out[base_i + internal, base_j + internal] = value
    # Numerical symmetrization protects eigensolvers from tiny rho_-q mismatch.
    return 0.5 * (out + out.conjugate().T)


def build_hartree_b0_hamiltonian(
    k_dimless: complex,
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    *,
    lg: int,
    rho_q: dict[tuple[int, int], complex] | None,
    epsilon_r: float,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
    gvec: np.ndarray | None = None,
    tunnel: np.ndarray | None = None,
    hartree_matrix: np.ndarray | None = None,
) -> np.ndarray:
    h = build_chau_b0_hamiltonian(
        complex(k_dimless),
        params,
        config,
        lg=int(lg),
        sigma_rotation=bool(sigma_rotation),
        periodic_g_grid=bool(periodic_g_grid),
        gvec=gvec,
        tunnel=tunnel,
    )
    if hartree_matrix is not None:
        return h + np.asarray(hartree_matrix, dtype=np.complex128)
    if rho_q is None:
        return h
    return h + build_hartree_matrix_from_rho(params, config, lg=int(lg), rho_q=rho_q, epsilon_r=float(epsilon_r))


def _density_from_flat_evecs(
    evecs_by_valley: dict[int, np.ndarray],
    occ_by_valley: dict[int, np.ndarray],
    *,
    lg: int,
    spin_degeneracy: float = 2.0,
) -> dict[tuple[int, int], complex]:
    """Flat-band density difference coefficients in the local continuum basis.

    ``occ_by_valley[zeta]`` is shaped ``(nk, 2)`` and is the occupation
    difference relative to charge neutrality: lower-flat occupation minus 1 and
    upper-flat occupation minus 0.
    """

    ng = int(lg) * int(lg)
    coords = b0_g_coords(int(lg))
    rho: dict[tuple[int, int], complex] = {}
    nk_total = None
    for zeta, evecs in evecs_by_valley.items():
        vec = np.asarray(evecs, dtype=np.complex128)  # (nk, dim, 2)
        occ = np.asarray(occ_by_valley[int(zeta)], dtype=float)  # (nk, 2)
        if nk_total is None:
            nk_total = int(vec.shape[0])
        for ik in range(vec.shape[0]):
            for band in range(2):
                weight = float(occ[ik, band]) * float(spin_degeneracy)
                if abs(weight) < 1.0e-14:
                    continue
                coeff = vec[ik, :, band].reshape((ng, 4))
                # Sum internal orbital density; potential is scalar in internal space.
                gg_density = coeff @ coeff.conjugate().T
                for i in range(ng):
                    ci = coords[i]
                    for j in range(ng):
                        shift = (int(ci[0] - coords[j, 0]), int(ci[1] - coords[j, 1]))
                        rho[shift] = rho.get(shift, 0.0j) + weight * complex(gg_density[i, j])
    norm = float(nk_total or 1)
    return {shift: value / norm for shift, value in rho.items() if abs(value) > 1.0e-14}


def _density_from_evecs_and_occupations(
    evecs_by_valley: dict[int, np.ndarray],
    occ_by_valley: dict[int, np.ndarray],
    *,
    lg: int,
    spin_degeneracy: float = 2.0,
) -> dict[tuple[int, int], complex]:
    """Density coefficients from arbitrary occupied bands in the continuum basis.

    ``evecs_by_valley[zeta]`` is shaped ``(nk, dim, nb)`` and
    ``occ_by_valley[zeta]`` is shaped ``(nk, nb)``.  The returned coefficients
    are averaged over k and multiplied by the spin degeneracy.  This helper is
    used for the full-density Hartree diagnostic, where occupied remote bands
    can contribute through wavefunction polarization relative to a fixed CNP
    reference.
    """

    ng = int(lg) * int(lg)
    coords = b0_g_coords(int(lg))
    rho: dict[tuple[int, int], complex] = {}
    nk_total = None
    for zeta, evecs in evecs_by_valley.items():
        vec = np.asarray(evecs, dtype=np.complex128)  # (nk, dim, nb)
        occ = np.asarray(occ_by_valley[int(zeta)], dtype=float)  # (nk, nb)
        if vec.ndim != 3:
            raise ValueError(f"evecs for valley {zeta} must be 3D, got {vec.shape}")
        if occ.shape != (vec.shape[0], vec.shape[2]):
            raise ValueError(f"occupations for valley {zeta} have shape {occ.shape}, expected {(vec.shape[0], vec.shape[2])}")
        if nk_total is None:
            nk_total = int(vec.shape[0])
        for ik in range(vec.shape[0]):
            weights = occ[ik] * float(spin_degeneracy)
            if np.max(np.abs(weights), initial=0.0) < 1.0e-14:
                continue
            coeff = vec[ik].reshape((ng, 4, vec.shape[2]))
            gg_density = np.zeros((ng, ng), dtype=np.complex128)
            for internal in range(4):
                x = coeff[:, internal, :]
                gg_density += (x * weights[None, :]) @ x.conjugate().T
            for i in range(ng):
                ci = coords[i]
                for j in range(ng):
                    value = complex(gg_density[i, j])
                    if abs(value) <= 1.0e-14:
                        continue
                    shift = (int(ci[0] - coords[j, 0]), int(ci[1] - coords[j, 1]))
                    rho[shift] = rho.get(shift, 0.0j) + value
    norm = float(nk_total or 1)
    return {shift: value / norm for shift, value in rho.items() if abs(value) > 1.0e-14}


def _subtract_rho(
    lhs: dict[tuple[int, int], complex],
    rhs: dict[tuple[int, int], complex],
) -> dict[tuple[int, int], complex]:
    out: dict[tuple[int, int], complex] = {}
    for key in set(lhs) | set(rhs):
        value = lhs.get(key, 0.0j) - rhs.get(key, 0.0j)
        if abs(value) > 1.0e-14:
            out[key] = value
    return out


def compute_full_cnp_reference_density(
    k_grid: np.ndarray,
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    *,
    lg: int,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
) -> dict[tuple[int, int], complex]:
    """Fixed noninteracting CNP density in the truncated continuum basis.

    This is a diagnostic reference for testing whether the paper's
    ``density relative to CNP`` should include remote-band wavefunction
    polarization.  It is not used by the default flat-band Hartree workflow.
    """

    lg = int(lg)
    dim = 4 * lg * lg
    center = dim // 2
    gvec = _generate_gvec(params, lg)
    tunnel_builder = _generate_t12 if bool(periodic_g_grid) else _generate_t12_zero_fill
    evecs_by_valley: dict[int, np.ndarray] = {}
    occ_by_valley: dict[int, np.ndarray] = {}
    for zeta in (1, -1):
        cfg_z = ChaudharyTBGConfig(**{**config.__dict__, "valley": int(zeta)})
        tunnel = tunnel_builder(params, lg, int(zeta))
        vecs = np.zeros((len(k_grid), dim, dim), dtype=np.complex128)
        occ = np.zeros((len(k_grid), dim), dtype=float)
        occ[:, :center] = 1.0
        for ik, k in enumerate(np.asarray(k_grid, dtype=np.complex128).reshape(-1)):
            h = build_chau_b0_hamiltonian(
                complex(k),
                params,
                cfg_z,
                lg=lg,
                sigma_rotation=bool(sigma_rotation),
                periodic_g_grid=bool(periodic_g_grid),
                gvec=gvec,
                tunnel=tunnel_builder(params, lg, int(zeta)) if tunnel is None else tunnel,
            )
            _evals, evec = eigh(h)
            vecs[ik, :, :] = evec
        evecs_by_valley[int(zeta)] = vecs
        occ_by_valley[int(zeta)] = occ
    return _density_from_evecs_and_occupations(evecs_by_valley, occ_by_valley, lg=lg, spin_degeneracy=2.0)


def _full_spectrum_by_valley(
    k_grid: np.ndarray,
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    *,
    lg: int,
    rho_q: dict[tuple[int, int], complex] | None = None,
    epsilon_r: float = 15.0,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Diagonalize the full truncated continuum Hamiltonian for both valleys."""

    lg = int(lg)
    dim = 4 * lg * lg
    gvec = _generate_gvec(params, lg)
    tunnel_builder = _generate_t12 if bool(periodic_g_grid) else _generate_t12_zero_fill
    h_hartree = None
    if rho_q is not None:
        h_hartree = build_hartree_matrix_from_rho(params, config, lg=lg, rho_q=rho_q, epsilon_r=float(epsilon_r))

    evals_by_valley: dict[int, np.ndarray] = {}
    evecs_by_valley: dict[int, np.ndarray] = {}
    for zeta in (1, -1):
        cfg_z = ChaudharyTBGConfig(**{**config.__dict__, "valley": int(zeta)})
        tunnel = tunnel_builder(params, lg, int(zeta))
        vals = np.zeros((len(k_grid), dim), dtype=float)
        vecs = np.zeros((len(k_grid), dim, dim), dtype=np.complex128)
        for ik, k in enumerate(np.asarray(k_grid, dtype=np.complex128).reshape(-1)):
            h = build_hartree_b0_hamiltonian(
                complex(k),
                params,
                cfg_z,
                lg=lg,
                rho_q=None,
                epsilon_r=float(epsilon_r),
                sigma_rotation=bool(sigma_rotation),
                periodic_g_grid=bool(periodic_g_grid),
                gvec=gvec,
                tunnel=tunnel,
                hartree_matrix=h_hartree,
            )
            evals, evec = eigh(h)
            vals[ik, :] = evals
            vecs[ik, :, :] = evec
        evals_by_valley[int(zeta)] = vals
        evecs_by_valley[int(zeta)] = vecs
    return evals_by_valley, evecs_by_valley


def compute_full_fixed_cnp_density_difference(
    k_grid: np.ndarray,
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    *,
    lg: int,
    nu: float,
    reference_rho_q: dict[tuple[int, int], complex],
    rho_q: dict[tuple[int, int], complex] | None = None,
    epsilon_r: float = 15.0,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
    temperature_k: float = 0.0,
) -> tuple[dict[tuple[int, int], complex], float]:
    """Full occupied density minus fixed noninteracting CNP density.

    This diagnostic goes beyond the default flat-band source and allows all
    occupied states in the truncated continuum basis to polarize in response to
    the Hartree potential.  Because the paper says dispersive bands are partly
    folded into an effective dielectric, this mode is an audit of conventions,
    not automatically the final physical model.
    """

    lg = int(lg)
    dim = 4 * lg * lg
    evals_by_valley, evecs_by_valley = _full_spectrum_by_valley(
        k_grid,
        params,
        config,
        lg=lg,
        rho_q=rho_q,
        epsilon_r=float(epsilon_r),
        sigma_rotation=bool(sigma_rotation),
        periodic_g_grid=bool(periodic_g_grid),
    )
    all_e = np.stack([evals_by_valley[1], evals_by_valley[-1]], axis=0)  # valley, k, band
    # Per spin: neutral count is two valleys times dim/2 occupied bands = dim states per k.
    # Total filling nu is shared by two spins, hence +nu/2 per k for one spin.
    target_per_spin = (float(dim) + 0.5 * float(nu)) * len(k_grid)
    occ_all, mu_ev = occupations_for_count(all_e, target_per_spin, temperature_k=float(temperature_k))
    occ_by_valley = {1: np.asarray(occ_all[0], dtype=float), -1: np.asarray(occ_all[1], dtype=float)}
    rho_current = _density_from_evecs_and_occupations(evecs_by_valley, occ_by_valley, lg=lg, spin_degeneracy=2.0)
    rho = _subtract_rho(rho_current, reference_rho_q)
    rho.pop((0, 0), None)
    return rho, float(mu_ev)


def compute_full_delta_occupation_density_difference(
    k_grid: np.ndarray,
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    *,
    lg: int,
    nu: float,
    rho_q: dict[tuple[int, int], complex] | None = None,
    epsilon_r: float = 15.0,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
    temperature_k: float = 0.0,
) -> tuple[dict[tuple[int, int], complex], float]:
    """Full-continuum density of the doped carriers only.

    This mode diagonalizes the full truncated continuum Hamiltonian, but it
    subtracts a charge-neutral occupation in the *same* Hartree eigenbasis.
    Consequently completely filled remote/sea bands cancel exactly and do not
    provide an explicit polarization screening cloud.  It is a more literal
    full-basis version of "occupied states measured from CNP" for testing
    whether the default two-flat-band density misses carriers due to band
    inversion, finite-temperature leakage, or edge fillings near ``|nu|=4``.
    """

    lg = int(lg)
    dim = 4 * lg * lg
    evals_by_valley, evecs_by_valley = _full_spectrum_by_valley(
        k_grid,
        params,
        config,
        lg=lg,
        rho_q=rho_q,
        epsilon_r=float(epsilon_r),
        sigma_rotation=bool(sigma_rotation),
        periodic_g_grid=bool(periodic_g_grid),
    )
    all_e = np.stack([evals_by_valley[1], evals_by_valley[-1]], axis=0)  # valley, k, band
    # Per spin: neutral count is two valleys times dim/2 occupied bands = dim states per k.
    target_per_spin = (float(dim) + 0.5 * float(nu)) * len(k_grid)
    neutral_per_spin = float(dim) * len(k_grid)
    occ_all, mu_ev = occupations_for_count(all_e, target_per_spin, temperature_k=float(temperature_k))
    occ_neutral, _mu0 = occupations_for_count(all_e, neutral_per_spin, temperature_k=float(temperature_k))
    occ_diff_all = np.asarray(occ_all, dtype=float) - np.asarray(occ_neutral, dtype=float)
    occ_by_valley = {1: np.asarray(occ_diff_all[0], dtype=float), -1: np.asarray(occ_diff_all[1], dtype=float)}
    rho = _density_from_evecs_and_occupations(evecs_by_valley, occ_by_valley, lg=lg, spin_degeneracy=2.0)
    rho.pop((0, 0), None)
    return rho, float(mu_ev)


def compute_flat_density_difference(
    k_grid: np.ndarray,
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    *,
    lg: int,
    nu: float,
    rho_q: dict[tuple[int, int], complex] | None = None,
    epsilon_r: float = 15.0,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
    temperature_k: float = 0.0,
) -> tuple[dict[tuple[int, int], complex], float]:
    """Compute Hartree density coefficients from central flat bands.

    The density is measured relative to charge neutrality in the same two-flat-
    band subspace: lower flat band fully occupied and upper flat band empty.
    """

    lg = int(lg)
    gvec = _generate_gvec(params, lg)
    tunnel_builder = _generate_t12 if bool(periodic_g_grid) else _generate_t12_zero_fill
    h_hartree = None
    if rho_q is not None:
        h_hartree = build_hartree_matrix_from_rho(params, config, lg=lg, rho_q=rho_q, epsilon_r=float(epsilon_r))

    central_evals: dict[int, np.ndarray] = {}
    central_evecs: dict[int, np.ndarray] = {}
    dim = 4 * lg * lg
    center = dim // 2
    subset = [center - 1, center]
    for zeta in (1, -1):
        cfg_z = ChaudharyTBGConfig(**{**config.__dict__, "valley": int(zeta)})
        tunnel = tunnel_builder(params, lg, int(zeta))
        vals = np.zeros((len(k_grid), 2), dtype=float)
        vecs = np.zeros((len(k_grid), dim, 2), dtype=np.complex128)
        for ik, k in enumerate(np.asarray(k_grid, dtype=np.complex128).reshape(-1)):
            h = build_hartree_b0_hamiltonian(
                complex(k),
                params,
                cfg_z,
                lg=lg,
                rho_q=None,
                epsilon_r=float(epsilon_r),
                sigma_rotation=bool(sigma_rotation),
                periodic_g_grid=bool(periodic_g_grid),
                gvec=gvec,
                tunnel=tunnel,
                hartree_matrix=h_hartree,
            )
            evals, evec = eigh(h, subset_by_index=subset, driver="evr")
            vals[ik, :] = evals
            vecs[ik, :, :] = evec
        central_evals[int(zeta)] = vals
        central_evecs[int(zeta)] = vecs

    # Common chemical potential for the two valleys; spin degeneracy is an
    # overall factor, so determine the target count per spin.
    all_e = np.stack([central_evals[1], central_evals[-1]], axis=0)  # valley, k, band
    target_per_spin = (4.0 + float(nu)) / 2.0 * len(k_grid)
    occ_all, mu_ev = occupations_for_count(all_e, target_per_spin, temperature_k=float(temperature_k))

    occ_diff: dict[int, np.ndarray] = {}
    for iv, zeta in enumerate((1, -1)):
        occ = np.asarray(occ_all[iv], dtype=float).copy()  # k, band
        occ[:, 0] -= 1.0  # subtract CNP lower-flat occupation
        occ_diff[int(zeta)] = occ
    rho = _density_from_flat_evecs(central_evecs, occ_diff, lg=lg, spin_degeneracy=2.0)
    rho.pop((0, 0), None)  # neutralizing background / no q=0 Hartree shift
    return rho, float(mu_ev)


def run_flat_hartree_scf(
    k_grid: np.ndarray,
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    *,
    lg: int,
    nu: float,
    epsilon_r: float = 15.0,
    max_iter: int = 50,
    mixing: float = 0.35,
    precision: float = 1.0e-7,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
    temperature_k: float = 0.0,
    initial_rho_q: dict[tuple[int, int], complex] | None = None,
    density_mode: str = "flat",
    hartree_shift_mode: str = "all",
) -> HartreeSCFResult:
    shift_mode = canonical_hartree_shift_mode(hartree_shift_mode)
    rho: dict[tuple[int, int], complex] = filter_rho_by_shift_mode(dict(initial_rho_q or {}), shift_mode)
    errors: list[float] = []
    mus: list[float] = []
    converged = False
    k_array = np.asarray(k_grid, dtype=np.complex128)
    mode = str(density_mode).lower().replace("-", "_")
    aliases = {
        "full": "full_fixed_cnp",
        "full_cnp": "full_fixed_cnp",
        "full_density": "full_fixed_cnp",
        "delta_occ": "full_delta_occ",
        "delta_occupation": "full_delta_occ",
        "full_delta_occupation": "full_delta_occ",
    }
    canonical_mode = aliases.get(mode, mode)
    reference_rho_q = None
    if canonical_mode == "full_fixed_cnp":
        reference_rho_q = compute_full_cnp_reference_density(
            k_array,
            params,
            config,
            lg=int(lg),
            sigma_rotation=bool(sigma_rotation),
            periodic_g_grid=bool(periodic_g_grid),
        )
    elif canonical_mode not in {"flat", "full_delta_occ"}:
        raise ValueError(
            f"Unsupported Hartree density_mode={density_mode!r}; expected flat, full_delta_occ, or full_fixed_cnp"
        )

    for iteration in range(1, int(max_iter) + 1):
        if canonical_mode == "flat":
            rho_new, mu = compute_flat_density_difference(
                k_array,
                params,
                config,
                lg=int(lg),
                nu=float(nu),
                rho_q=rho,
                epsilon_r=float(epsilon_r),
                sigma_rotation=bool(sigma_rotation),
                periodic_g_grid=bool(periodic_g_grid),
                temperature_k=float(temperature_k),
            )
        elif canonical_mode == "full_delta_occ":
            rho_new, mu = compute_full_delta_occupation_density_difference(
                k_array,
                params,
                config,
                lg=int(lg),
                nu=float(nu),
                rho_q=rho,
                epsilon_r=float(epsilon_r),
                sigma_rotation=bool(sigma_rotation),
                periodic_g_grid=bool(periodic_g_grid),
                temperature_k=float(temperature_k),
            )
        else:
            rho_new, mu = compute_full_fixed_cnp_density_difference(
                k_array,
                params,
                config,
                lg=int(lg),
                nu=float(nu),
                reference_rho_q=reference_rho_q or {},
                rho_q=rho,
                epsilon_r=float(epsilon_r),
                sigma_rotation=bool(sigma_rotation),
                periodic_g_grid=bool(periodic_g_grid),
                temperature_k=float(temperature_k),
            )
        rho_new = filter_rho_by_shift_mode(rho_new, shift_mode)
        keys = set(rho) | set(rho_new)
        err = max((abs(rho_new.get(k, 0.0j) - rho.get(k, 0.0j)) for k in keys), default=0.0)
        mixed: dict[tuple[int, int], complex] = {}
        for key in keys:
            value = (1.0 - float(mixing)) * rho.get(key, 0.0j) + float(mixing) * rho_new.get(key, 0.0j)
            if abs(value) > 1.0e-14:
                mixed[key] = value
        rho = mixed
        errors.append(float(err))
        mus.append(float(mu))
        if err <= float(precision):
            converged = True
            break
    return HartreeSCFResult(
        rho_q=rho,
        iterations=len(errors),
        converged=bool(converged),
        final_error=float(errors[-1] if errors else 0.0),
        mu_ev=float(mus[-1] if mus else 0.0),
        iter_error=np.asarray(errors, dtype=float),
        iter_mu_ev=np.asarray(mus, dtype=float),
        nu=float(nu),
        epsilon_r=float(epsilon_r),
        mixing=float(mixing),
        mesh_size=int(round(math.sqrt(len(k_grid)))),
        lg=int(lg),
        temperature_k=float(temperature_k),
        density_mode=str(canonical_mode),
        hartree_shift_mode=str(shift_mode),
    )


def rho_to_arrays(rho_q: dict[tuple[int, int], complex]) -> tuple[np.ndarray, np.ndarray]:
    shifts = np.asarray(sorted(rho_q), dtype=int)
    values = np.asarray([rho_q[(int(s[0]), int(s[1]))] for s in shifts], dtype=np.complex128)
    return shifts, values


def arrays_to_rho(shifts: np.ndarray, values: np.ndarray) -> dict[tuple[int, int], complex]:
    return {(int(s[0]), int(s[1])): complex(v) for s, v in zip(np.asarray(shifts, dtype=int), np.asarray(values, dtype=np.complex128), strict=True)}

__all__ = [
    "FIRST_STAR_SHIFTS",
    "KB_EV_PER_K",
    "E2_OVER_4PI_EPS0_EV_NM",
    "HartreeSCFResult",
    "canonical_hartree_shift_mode",
    "filter_rho_by_shift_mode",
    "b0_g_coords",
    "moire_cell_area_nm2",
    "hartree_kernel_ev",
    "occupations_for_count",
    "build_hartree_matrix_from_rho",
    "build_hartree_b0_hamiltonian",
    "compute_full_cnp_reference_density",
    "compute_full_fixed_cnp_density_difference",
    "compute_full_delta_occupation_density_difference",
    "compute_flat_density_difference",
    "run_flat_hartree_scf",
    "rho_to_arrays",
    "arrays_to_rho",
]
