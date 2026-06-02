from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss

from .response import fermi_occupation, precompute_response_tensors
from .slg_toy import GappedSLGParams, _nearest_reciprocal_vectors, d2hdk, dhdk, diagonalize, hex_bz_grid, hex_bz_vertices, reciprocal_vectors


@dataclass(frozen=True)
class EnergyQuadratureTable:
    """K/K' transition-energy quadrature table for gapped monolayer graphene.

    The table stores all k-dependent ingredients needed for Hipolito et al.
    Eq. (25b).  It is designed for narrow broadening near the direct K/K'
    gap, where fixed-k quadrature produces shell-sampling wiggles.
    """

    params: GappedSLGParams
    weights_nm_inv_sq: np.ndarray  # shape (N,)
    evals_ev: np.ndarray  # shape (N,2)
    D_ev_nm: np.ndarray  # shape (N,2,2,2)
    r_nm: np.ndarray  # shape (N,2,2,2)
    r_covariant_nm2: np.ndarray  # shape (N,2,2,2,2), [deriv,conn,n,m]
    theta_count: int
    transition_energy_nodes: int
    transition_emax_ev: float
    patch_radius_nm_inv: float

    @property
    def n_points(self) -> int:
        return int(self.weights_nm_inv_sq.size)

    @property
    def delta_ev(self) -> float:
        return 2.0 * float(self.params.mass_ev)


def inside_hex(k_xy: np.ndarray, nearest_g: np.ndarray, bounds: np.ndarray) -> bool:
    return bool(np.all(np.asarray(k_xy, dtype=float) @ nearest_g.T <= bounds + 1.0e-10))


def transition_energy(k_xy: np.ndarray, params: GappedSLGParams) -> float:
    evals, _ = diagonalize(k_xy, params)
    return float(evals[1] - evals[0])


def ray_rmax(
    vertex: np.ndarray,
    direction: np.ndarray,
    *,
    nearest_g: np.ndarray,
    bounds: np.ndarray,
    patch_radius: float,
) -> float:
    if not inside_hex(vertex + 1.0e-8 * direction, nearest_g, bounds):
        return 0.0
    hi = float(patch_radius)
    if inside_hex(vertex + hi * direction, nearest_g, bounds):
        return hi
    lo = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if inside_hex(vertex + mid * direction, nearest_g, bounds):
            lo = mid
        else:
            hi = mid
    return lo


def radius_for_transition_energy(
    vertex: np.ndarray,
    direction: np.ndarray,
    *,
    params: GappedSLGParams,
    rmax: float,
    target_ev: float,
) -> float:
    lo = 0.0
    hi = float(rmax)
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        if transition_energy(vertex + mid * direction, params) < float(target_ev):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_energy_quadrature_table(
    params: GappedSLGParams,
    *,
    transition_emax_ev: float = 1.2,
    theta_count: int = 72,
    transition_energy_nodes: int = 900,
    patch_radius_nm_inv: float = 1.4,
    denominator_cutoff_ev: float = 1.0e-10,
) -> EnergyQuadratureTable:
    """Build a transition-energy quadrature table around all six K/K' corners.

    Each BZ corner contributes its inside-hexagon 120-degree sector.  The radial
    coordinate is changed from k-space radius to the direct transition energy
    E_cv.  This is the numerically stable way to integrate spectra with
    Gamma~1 meV.
    """

    vertices = hex_bz_vertices(params)
    nearest_g = _nearest_reciprocal_vectors(params)
    bounds = 0.5 * np.sum(nearest_g * nearest_g, axis=1)
    xg, wg = leggauss(int(transition_energy_nodes))
    delta_ev = 2.0 * float(params.mass_ev)

    weights: list[float] = []
    evals_list: list[np.ndarray] = []
    D_list: list[np.ndarray] = []
    r_list: list[np.ndarray] = []
    rcov_list: list[np.ndarray] = []

    for vertex in vertices:
        for itheta in range(int(theta_count)):
            theta = (itheta + 0.5) * 2.0 * math.pi / float(theta_count)
            dtheta = 2.0 * math.pi / float(theta_count)
            direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=float)
            rmax = ray_rmax(
                vertex,
                direction,
                nearest_g=nearest_g,
                bounds=bounds,
                patch_radius=float(patch_radius_nm_inv),
            )
            if rmax <= 0.0:
                continue
            e_hi = min(float(transition_emax_ev), transition_energy(vertex + rmax * direction, params))
            if e_hi <= delta_ev + 1.0e-10:
                continue
            e_nodes = 0.5 * (e_hi - delta_ev) * xg + 0.5 * (e_hi + delta_ev)
            e_weights = 0.5 * (e_hi - delta_ev) * wg
            for transition_ev, transition_weight in zip(e_nodes, e_weights, strict=True):
                radius = radius_for_transition_energy(
                    vertex,
                    direction,
                    params=params,
                    rmax=rmax,
                    target_ev=float(transition_ev),
                )
                k_xy = vertex + radius * direction
                evals, evecs = diagonalize(k_xy, params)
                tensors = precompute_response_tensors(
                    evals,
                    evecs,
                    dhdk(k_xy, params),
                    d2hdk=d2hdk(k_xy, params),
                    denominator_cutoff_ev=float(denominator_cutoff_ev),
                )
                radial_D = direction[0] * tensors.D[0] + direction[1] * tensors.D[1]
                d_transition_dr = float(np.real(radial_D[1, 1] - radial_D[0, 0]))
                if d_transition_dr <= 0.0:
                    continue
                weights.append(float(radius / d_transition_dr * transition_weight * dtheta))
                evals_list.append(np.asarray(evals, dtype=float))
                D_list.append(np.asarray(tensors.D, dtype=np.complex128))
                r_list.append(np.asarray(tensors.r, dtype=np.complex128))
                rcov_list.append(np.asarray(tensors.r_covariant, dtype=np.complex128))

    if not weights:
        raise RuntimeError("No transition-energy quadrature points were generated")
    return EnergyQuadratureTable(
        params=params,
        weights_nm_inv_sq=np.asarray(weights, dtype=float),
        evals_ev=np.asarray(evals_list, dtype=float),
        D_ev_nm=np.asarray(D_list, dtype=np.complex128),
        r_nm=np.asarray(r_list, dtype=np.complex128),
        r_covariant_nm2=np.asarray(rcov_list, dtype=np.complex128),
        theta_count=int(theta_count),
        transition_energy_nodes=int(transition_energy_nodes),
        transition_emax_ev=float(transition_emax_ev),
        patch_radius_nm_inv=float(patch_radius_nm_inv),
    )


def _constant_interval_integrals(
    photon_energies_ev: np.ndarray,
    e_left_ev: float,
    e_right_ev: float,
    *,
    sign: int,
    gamma_ev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact interval integrals of resonant denominators for constant numerator.

    Returns

    ``I1 = int_{E0}^{E1} dE/(omega - sign*E + i gamma)`` and
    ``I2 = int_{E0}^{E1} dE/(omega - sign*E + i gamma)^2``.

    Integrating the narrow denominator exactly removes the artificial
    node-crossing wiggles that appear when a 1 meV Lorentzian is sampled by a
    coarser transition-energy grid.
    """

    s = int(sign)
    if s not in (-1, 1):
        raise ValueError(f"sign must be +/-1, got {sign}")
    z = np.asarray(photon_energies_ev, dtype=float) + 1.0j * float(gamma_ev)
    u0 = z - float(s) * float(e_left_ev)
    u1 = z - float(s) * float(e_right_ev)
    i1 = -(np.log(u1) - np.log(u0)) / float(s)
    i2 = (1.0 / u1 - 1.0 / u0) / float(s)
    return i1, i2


def hipolito_eq25b_spectrum_energy_intervals(
    params: GappedSLGParams,
    photon_energies_ev: np.ndarray,
    *,
    component: tuple[int, int, int] = (1, 1, 1),
    gamma_ev: float = 1.0e-3,
    mu_ev: float = 0.0,
    temperature_k: float = 1.0,
    transition_emax_ev: float = 1.2,
    theta_count: int = 72,
    transition_energy_intervals: int = 900,
    patch_radius_nm_inv: float = 1.4,
    lower_edge_offset_ev: float = 1.0e-7,
    denominator_cutoff_ev: float = 1.0e-10,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """K/K' Eq. (25b) with exact integration of the resonant denominator.

    The numerator is evaluated at the midpoint of each transition-energy
    interval, while ``1/(omega-E+i Gamma)`` and
    ``1/(omega-E+i Gamma)^2`` are integrated analytically over the interval.
    This is a quadrature improvement, not smoothing: it changes the numerical
    integration rule so a narrow ``Gamma=1 meV`` denominator is not represented
    by isolated energy nodes.
    """

    photon = np.asarray(photon_energies_ev, dtype=float)
    lam, alpha, beta = component
    out = np.zeros_like(photon, dtype=np.complex128)
    vertices = hex_bz_vertices(params)
    nearest_g = _nearest_reciprocal_vectors(params)
    bounds = 0.5 * np.sum(nearest_g * nearest_g, axis=1)
    delta_ev = 2.0 * float(params.mass_ev)
    e_min = delta_ev + float(lower_edge_offset_ev)
    n_intervals = int(transition_energy_intervals)
    generated_intervals = 0
    generated_rays = 0
    max_e_hi = 0.0

    for vertex in vertices:
        for itheta in range(int(theta_count)):
            theta = (itheta + 0.5) * 2.0 * math.pi / float(theta_count)
            dtheta = 2.0 * math.pi / float(theta_count)
            direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=float)
            rmax = ray_rmax(
                vertex,
                direction,
                nearest_g=nearest_g,
                bounds=bounds,
                patch_radius=float(patch_radius_nm_inv),
            )
            if rmax <= 0.0:
                continue
            e_hi = min(float(transition_emax_ev), transition_energy(vertex + rmax * direction, params))
            max_e_hi = max(max_e_hi, float(e_hi))
            if e_hi <= e_min:
                continue
            generated_rays += 1
            edges = np.linspace(e_min, float(e_hi), n_intervals + 1, dtype=float)
            for e_left, e_right in zip(edges[:-1], edges[1:], strict=True):
                e_mid = 0.5 * (float(e_left) + float(e_right))
                radius = radius_for_transition_energy(
                    vertex,
                    direction,
                    params=params,
                    rmax=rmax,
                    target_ev=e_mid,
                )
                k_xy = vertex + radius * direction
                evals, evecs = diagonalize(k_xy, params)
                tensors = precompute_response_tensors(
                    evals,
                    evecs,
                    dhdk(k_xy, params),
                    d2hdk=d2hdk(k_xy, params),
                    mu_ev=float(mu_ev),
                    temperature_k=float(temperature_k),
                    denominator_cutoff_ev=float(denominator_cutoff_ev),
                )
                radial_D = direction[0] * tensors.D[0] + direction[1] * tensors.D[1]
                d_transition_dr = float(np.real(radial_D[1, 1] - radial_D[0, 0]))
                if d_transition_dr <= 0.0:
                    continue
                density = float(radius / d_transition_dr * dtheta) / (2.0 * math.pi) ** 2
                D = tensors.D
                r = tensors.r
                rcov = tensors.r_covariant
                occ = tensors.occupations
                for m, n, sign in ((1, 0, 1), (0, 1, -1)):
                    f_nm = float(occ[n] - occ[m])
                    if abs(f_nm) < 1.0e-14:
                        continue
                    pref = density * (-D[lam, n, m]) / (-float(sign) * e_mid + 2.0j * float(gamma_ev))
                    partial_beta_delta = D[beta, m, m] - D[beta, n, n]
                    A = 1.0j * rcov[beta, alpha, m, n]
                    B = 1.0j * r[alpha, m, n] * partial_beta_delta
                    i1, i2 = _constant_interval_integrals(
                        photon,
                        float(e_left),
                        float(e_right),
                        sign=int(sign),
                        gamma_ev=float(gamma_ev),
                    )
                    out += pref * f_nm * (A * i1 + B * i2)
                generated_intervals += 1
    return out, {
        "theta_count": int(theta_count),
        "transition_energy_intervals": int(transition_energy_intervals),
        "generated_rays": int(generated_rays),
        "generated_intervals": int(generated_intervals),
        "transition_emax_ev": float(transition_emax_ev),
        "max_ray_transition_e_hi_ev": float(max_e_hi),
        "patch_radius_nm_inv": float(patch_radius_nm_inv),
        "lower_edge_offset_ev": float(lower_edge_offset_ev),
    }


def hipolito_eq25b_spectrum_from_table(
    table: EnergyQuadratureTable,
    photon_energies_ev: np.ndarray,
    *,
    component: tuple[int, int, int] = (1, 1, 1),
    gamma_ev: float = 1.0e-3,
    mu_ev: float = 0.0,
    temperature_k: float = 1.0,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Return the raw Eq. (25b) integral for one tensor component.

    The global Hipolito prefactor/convention is applied separately by
    ``normalize_by_eq31`` or by a direct prefactor in the calling script.
    """

    photon = np.asarray(photon_energies_ev, dtype=float)
    lam, alpha, beta = component
    out = np.zeros_like(photon, dtype=np.complex128)
    factor = table.weights_nm_inv_sq / (2.0 * math.pi) ** 2
    occupations = fermi_occupation(table.evals_ev, mu_ev=float(mu_ev), temperature_k=float(temperature_k))
    n_total = table.n_points
    for start in range(0, n_total, int(chunk_size)):
        stop = min(n_total, start + int(chunk_size))
        evals = table.evals_ev[start:stop]
        D = table.D_ev_nm[start:stop]
        r = table.r_nm[start:stop]
        rcov = table.r_covariant_nm2[start:stop]
        occ = occupations[start:stop]
        wk = factor[start:stop]
        for m, n in ((1, 0), (0, 1)):
            delta_mn = evals[:, m] - evals[:, n]
            f_nm = occ[:, n] - occ[:, m]
            valid = np.abs(f_nm) > 1.0e-14
            if not np.any(valid):
                continue
            delta = delta_mn[valid]
            f = f_nm[valid]
            pref = wk[valid] * (-D[valid, lam, n, m]) / (-delta + 2.0j * float(gamma_ev))
            partial_beta_delta = D[valid, beta, m, m] - D[valid, beta, n, n]
            A = 1.0j * rcov[valid, beta, alpha, m, n]
            B = 1.0j * r[valid, alpha, m, n] * partial_beta_delta
            # Chunk over photon energies via broadcasting over selected k points.
            den = photon[None, :] - delta[:, None] + 1.0j * float(gamma_ev)
            contrib = pref[:, None] * f[:, None] * (A[:, None] / den + B[:, None] / (den * den))
            out += np.sum(contrib, axis=0, dtype=np.complex128)
    return out


def hipolito_eq25b_spectrum_fixed_grid(
    params: GappedSLGParams,
    photon_energies_ev: np.ndarray,
    *,
    component: tuple[int, int, int] = (1, 1, 1),
    gamma_ev: float = 0.03,
    mu_ev: float = 0.0,
    temperature_k: float = 1.0,
    mesh_size: int = 160,
    denominator_cutoff_ev: float = 1.0e-10,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Full-BZ fixed-grid Eq. (25b) diagnostic.

    This is useful for broad high-energy / M-point checks.  For narrow
    ``Gamma=1 meV`` threshold benchmarks, prefer the transition-energy table;
    a fixed k grid will otherwise show resonance-shell sampling wiggles.
    """

    photon = np.asarray(photon_energies_ev, dtype=float)
    lam, alpha, beta = component
    out = np.zeros_like(photon, dtype=np.complex128)
    k_points, k_weights = hex_bz_grid(int(mesh_size), params)
    for k_xy, k_weight in zip(k_points, k_weights, strict=True):
        evals, evecs = diagonalize(k_xy, params)
        tensors = precompute_response_tensors(
            evals,
            evecs,
            dhdk(k_xy, params),
            d2hdk=d2hdk(k_xy, params),
            mu_ev=float(mu_ev),
            temperature_k=float(temperature_k),
            denominator_cutoff_ev=float(denominator_cutoff_ev),
        )
        D = tensors.D
        r = tensors.r
        rcov = tensors.r_covariant
        occ = tensors.occupations
        for m, n in ((1, 0), (0, 1)):
            delta_mn = float(evals[m] - evals[n])
            f_nm = float(occ[n] - occ[m])
            if abs(f_nm) < 1.0e-14:
                continue
            den = photon - delta_mn + 1.0j * float(gamma_ev)
            partial_beta_delta = D[beta, m, m] - D[beta, n, n]
            cov_derivative = f_nm * (
                1.0j * rcov[beta, alpha, m, n] / den
                + 1.0j * r[alpha, m, n] * partial_beta_delta / (den * den)
            )
            out += (
                float(k_weight)
                / (2.0 * math.pi) ** 2
                * (-D[lam, n, m])
                / (-delta_mn + 2.0j * float(gamma_ev))
                * cov_derivative
            )
    return out, {"mesh_size": int(mesh_size), "n_k_points": int(k_points.shape[0])}


@dataclass(frozen=True)
class FullBZTetraHistogram:
    """Full-BZ transition-energy histogram for the Hipolito Eq. (25b) integrand."""

    params: GappedSLGParams
    energy_edges_ev: np.ndarray  # shape (Nbin+1,)
    coeff_integrals: np.ndarray  # shape (2 signs, 2 denominator powers, Nbin)
    mesh_size: int
    n_triangles: int
    primitive_cell_area_nm_inv_sq: float

    @property
    def n_bins(self) -> int:
        return int(self.energy_edges_ev.size - 1)

    @property
    def energy_bin_width_ev(self) -> float:
        widths = np.diff(self.energy_edges_ev)
        return float(np.median(widths))


class _VertexCoeffCache:
    __slots__ = ("energies", "coeffs")

    def __init__(self, energies: np.ndarray, coeffs: np.ndarray) -> None:
        self.energies = energies
        self.coeffs = coeffs


def _eq25b_vertex_coefficients(
    k_xy: np.ndarray,
    params: GappedSLGParams,
    *,
    component: tuple[int, int, int],
    gamma_ev: float,
    mu_ev: float,
    temperature_k: float,
    denominator_cutoff_ev: float,
) -> tuple[float, np.ndarray]:
    """Return positive transition energy and smooth Eq. (25b) coefficients.

    The returned coefficients have shape ``(2, 2)``.  The first axis indexes
    ``sign=+1`` for the resonant valence-to-conduction term and ``sign=-1`` for
    the nonresonant reverse term.  The second axis stores the smooth
    coefficients multiplying ``1/(omega-sign*E+iGamma)`` and its square.
    """

    lam, alpha, beta = component
    evals, evecs = diagonalize(k_xy, params)
    tensors = precompute_response_tensors(
        evals,
        evecs,
        dhdk(k_xy, params),
        d2hdk=d2hdk(k_xy, params),
        mu_ev=float(mu_ev),
        temperature_k=float(temperature_k),
        denominator_cutoff_ev=float(denominator_cutoff_ev),
    )
    out = np.zeros((2, 2), dtype=np.complex128)
    transition_ev = float(evals[1] - evals[0])
    D = tensors.D
    r = tensors.r
    rcov = tensors.r_covariant
    occ = tensors.occupations
    for sign_index, (m, n, sign) in enumerate(((1, 0, 1), (0, 1, -1))):
        delta_mn = float(evals[m] - evals[n])
        # The transition energy should match sign*E; use the local value for
        # the smooth two-photon denominator to avoid any roundoff mismatch.
        f_nm = float(occ[n] - occ[m])
        if abs(f_nm) < 1.0e-14:
            continue
        pref = f_nm * (-D[lam, n, m]) / (-delta_mn + 2.0j * float(gamma_ev))
        partial_beta_delta = D[beta, m, m] - D[beta, n, n]
        out[sign_index, 0] = pref * (1.0j * rcov[beta, alpha, m, n])
        out[sign_index, 1] = pref * (1.0j * r[alpha, m, n] * partial_beta_delta)
        if int(sign) == 1 and transition_ev <= 0.0:
            raise RuntimeError("Expected positive conduction-valence transition energy")
    return transition_ev, out


def _primitive_cell_vertex_cache(
    params: GappedSLGParams,
    *,
    mesh_size: int,
    component: tuple[int, int, int],
    gamma_ev: float,
    mu_ev: float,
    temperature_k: float,
    denominator_cutoff_ev: float,
) -> _VertexCoeffCache:
    b1, b2 = reciprocal_vectors(params)
    n = int(mesh_size)
    energies = np.empty((n + 1, n + 1), dtype=float)
    coeffs = np.empty((n + 1, n + 1, 2, 2), dtype=np.complex128)
    for i in range(n + 1):
        u = float(i) / float(n) - 0.5
        for j in range(n + 1):
            v = float(j) / float(n) - 0.5
            k_xy = u * b1 + v * b2
            energies[i, j], coeffs[i, j] = _eq25b_vertex_coefficients(
                k_xy,
                params,
                component=component,
                gamma_ev=float(gamma_ev),
                mu_ev=float(mu_ev),
                temperature_k=float(temperature_k),
                denominator_cutoff_ev=float(denominator_cutoff_ev),
            )
    return _VertexCoeffCache(energies=energies, coeffs=coeffs)


def _add_linear_density_segment_to_hist(
    hist: np.ndarray,
    edges: np.ndarray,
    *,
    e_left: float,
    e_right: float,
    slope: float,
    intercept: float,
    coeff: np.ndarray,
) -> None:
    """Deposit ``coeff * int (slope*E+intercept)dE`` into energy bins."""

    if e_right <= e_left:
        return
    n_bins = int(edges.size - 1)
    start = max(0, int(np.searchsorted(edges, e_left, side="right") - 1))
    stop = min(n_bins - 1, int(np.searchsorted(edges, e_right, side="left")))
    for ibin in range(start, stop + 1):
        left = max(float(e_left), float(edges[ibin]))
        right = min(float(e_right), float(edges[ibin + 1]))
        if right <= left:
            continue
        density_integral = 0.5 * float(slope) * (right * right - left * left) + float(intercept) * (right - left)
        if density_integral != 0.0:
            hist[:, :, ibin] += coeff * density_integral


def _add_triangle_to_hist(
    hist: np.ndarray,
    edges: np.ndarray,
    *,
    energies: np.ndarray,
    coeff: np.ndarray,
    triangle_area_nm_inv_sq: float,
) -> None:
    e = np.sort(np.asarray(energies, dtype=float))
    e0, e1, e2 = float(e[0]), float(e[1]), float(e[2])
    area = float(triangle_area_nm_inv_sq)
    if e2 <= e0 + 1.0e-14:
        ibin = int(np.searchsorted(edges, 0.5 * (e0 + e2), side="right") - 1)
        if 0 <= ibin < hist.shape[2]:
            hist[:, :, ibin] += coeff * area
        return
    if e1 > e0 + 1.0e-14:
        slope = 2.0 * area / ((e1 - e0) * (e2 - e0))
        _add_linear_density_segment_to_hist(
            hist,
            edges,
            e_left=e0,
            e_right=e1,
            slope=slope,
            intercept=-slope * e0,
            coeff=coeff,
        )
    if e2 > e1 + 1.0e-14:
        slope = -2.0 * area / ((e2 - e0) * (e2 - e1))
        _add_linear_density_segment_to_hist(
            hist,
            edges,
            e_left=e1,
            e_right=e2,
            slope=slope,
            intercept=-slope * e2,
            coeff=coeff,
        )


def build_full_bz_tetra_histogram(
    params: GappedSLGParams,
    *,
    component: tuple[int, int, int] = (1, 1, 1),
    gamma_ev: float = 1.0e-3,
    mu_ev: float = 0.0,
    temperature_k: float = 1.0,
    mesh_size: int = 360,
    energy_bin_width_ev: float = 0.002,
    denominator_cutoff_ev: float = 1.0e-10,
) -> FullBZTetraHistogram:
    """Build a full-BZ linear-tetrahedron energy histogram for Eq. (25b).

    The BZ integral is performed over a primitive reciprocal cell.  Each small
    triangle is treated with a linear interpolation of the transition energy,
    so the contribution is continuous in transition energy rather than a set of
    isolated k-shell samples.  The smooth Eq. (25b) numerator is averaged over
    triangle vertices and binned as an energy density.  The narrow resonant
    denominators are integrated analytically later by
    ``hipolito_eq25b_spectrum_from_full_bz_histogram``.
    """

    n = int(mesh_size)
    if n <= 1:
        raise ValueError(f"mesh_size must be >1, got {mesh_size}")
    if float(energy_bin_width_ev) <= 0.0:
        raise ValueError(f"energy_bin_width_ev must be positive, got {energy_bin_width_ev}")
    cache = _primitive_cell_vertex_cache(
        params,
        mesh_size=n,
        component=component,
        gamma_ev=float(gamma_ev),
        mu_ev=float(mu_ev),
        temperature_k=float(temperature_k),
        denominator_cutoff_ev=float(denominator_cutoff_ev),
    )
    b1, b2 = reciprocal_vectors(params)
    cell_area = abs(float(b1[0] * b2[1] - b1[1] * b2[0]))
    triangle_area = cell_area / (2.0 * n * n)
    emax = float(np.max(cache.energies))
    n_bins = int(math.ceil((emax + 2.0 * float(energy_bin_width_ev)) / float(energy_bin_width_ev)))
    edges = np.arange(n_bins + 1, dtype=float) * float(energy_bin_width_ev)
    hist = np.zeros((2, 2, n_bins), dtype=np.complex128)
    inv_2pi_sq = 1.0 / (2.0 * math.pi) ** 2
    n_triangles = 0
    for i in range(n):
        for j in range(n):
            for tri in (((i, j), (i + 1, j), (i + 1, j + 1)), ((i, j), (i + 1, j + 1), (i, j + 1))):
                idx = tuple(zip(*tri, strict=True))
                e_tri = cache.energies[idx]
                coeff_tri = np.mean(cache.coeffs[idx], axis=0) * inv_2pi_sq
                _add_triangle_to_hist(
                    hist,
                    edges,
                    energies=e_tri,
                    coeff=coeff_tri,
                    triangle_area_nm_inv_sq=triangle_area,
                )
                n_triangles += 1
    return FullBZTetraHistogram(
        params=params,
        energy_edges_ev=edges,
        coeff_integrals=hist,
        mesh_size=n,
        n_triangles=int(n_triangles),
        primitive_cell_area_nm_inv_sq=float(cell_area),
    )


def hipolito_eq25b_spectrum_from_full_bz_histogram(
    histogram: FullBZTetraHistogram,
    photon_energies_ev: np.ndarray,
    *,
    gamma_ev: float = 1.0e-3,
) -> np.ndarray:
    """Evaluate Eq. (25b) from a full-BZ tetrahedron energy histogram."""

    photon = np.asarray(photon_energies_ev, dtype=float)
    edges = np.asarray(histogram.energy_edges_ev, dtype=float)
    hist = np.asarray(histogram.coeff_integrals, dtype=np.complex128)
    out = np.zeros_like(photon, dtype=np.complex128)
    for ibin in range(histogram.n_bins):
        width = float(edges[ibin + 1] - edges[ibin])
        if width <= 0.0:
            continue
        e_left = float(edges[ibin])
        e_right = float(edges[ibin + 1])
        for sign_index, sign in enumerate((1, -1)):
            c1 = hist[sign_index, 0, ibin] / width
            c2 = hist[sign_index, 1, ibin] / width
            if abs(c1) + abs(c2) == 0.0:
                continue
            i1, i2 = _constant_interval_integrals(
                photon,
                e_left,
                e_right,
                sign=int(sign),
                gamma_ev=float(gamma_ev),
            )
            out += c1 * i1 + c2 * i2
    return out


def hipolito_eq25b_spectrum_full_bz_tetra_binned(
    params: GappedSLGParams,
    photon_energies_ev: np.ndarray,
    *,
    component: tuple[int, int, int] = (1, 1, 1),
    gamma_ev: float = 1.0e-3,
    mu_ev: float = 0.0,
    temperature_k: float = 1.0,
    mesh_size: int = 360,
    energy_bin_width_ev: float = 0.002,
    denominator_cutoff_ev: float = 1.0e-10,
) -> tuple[np.ndarray, FullBZTetraHistogram]:
    """Full-BZ, narrow-broadening Eq. (25b) via binned tetrahedra."""

    histogram = build_full_bz_tetra_histogram(
        params,
        component=component,
        gamma_ev=float(gamma_ev),
        mu_ev=float(mu_ev),
        temperature_k=float(temperature_k),
        mesh_size=int(mesh_size),
        energy_bin_width_ev=float(energy_bin_width_ev),
        denominator_cutoff_ev=float(denominator_cutoff_ev),
    )
    return hipolito_eq25b_spectrum_from_full_bz_histogram(histogram, photon_energies_ev, gamma_ev=float(gamma_ev)), histogram


def eq25_prefactor_scale(params: GappedSLGParams) -> float:
    """Direct Eq. (25b) prefactor in the current D=<dH/dk> units.

    A remaining global sign/convention factor is fixed by Eq. (31) in the
    benchmark scripts, because our A/B and component naming convention differs
    from Hipolito's reduced Hamiltonian convention.
    """

    spin_g = 2.0
    return -spin_g * float(params.hopping_ev) / float(params.bond_nm)


def normalize_by_eq31(
    raw: np.ndarray,
    photon_energies_ev: np.ndarray,
    *,
    params: GappedSLGParams,
    reference_offset_ev: float = 0.03,
) -> tuple[np.ndarray, float, float]:
    """Normalize a raw Eq. (25b) spectrum by Hipolito's K-point Eq. (31).

    This is not a visual fit: Eq. (31) fixes the low-energy threshold value
    analytically, in units of Hipolito's ``sigma_2``.
    """

    photon = np.asarray(photon_energies_ev, dtype=float)
    delta_ev = 2.0 * float(params.mass_ev)
    delta_dimensionless = delta_ev / float(params.hopping_ev)
    target = -1.0 / (4.0 * delta_dimensionless)
    scaled = eq25_prefactor_scale(params) * np.asarray(raw, dtype=np.complex128)
    idx = int(np.argmin(np.abs(photon - (delta_ev + float(reference_offset_ev)))))
    scale = eq25_prefactor_scale(params) * target / float(scaled[idx].real)
    return scale * np.asarray(raw, dtype=np.complex128), float(scale), float(target)
