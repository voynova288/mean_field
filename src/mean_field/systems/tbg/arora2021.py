"""Arora--Kong--Song 2021 strained TBG+hBN compatibility helpers.

This module keeps paper-specific glue out of the generic zero-field TBG code.  It
uses the repository's b0 momentum convention, but replaces the square ``lg x lg``
reciprocal cutoff by the paper-compatible 61-site circular reciprocal set used
for arXiv:2107.07526 Fig. 2.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field.model import dirac

GRAPHENE_A_NM = 0.246

@dataclass(frozen=True)
class AroraTBGConfig:
    """Minimal one-valley config needed by the Arora Fig. 2 adapter."""

    theta_deg: float
    graphene_lattice_constant_nm: float
    kinetic_ev: float
    w_ab_ev: float
    w_aa_ratio: float
    delta1_ev: float
    delta2_ev: float
    valley: int
    dirac_sign: float = 1.0



@dataclass(frozen=True)
class Arora2021Fig2Parameters:
    """Parameter bundle for Fig. 2 of arXiv:2107.07526v1."""

    theta_deg: float = 1.05
    strain: float = 0.001
    strain_angle_deg: float = 0.0
    poisson: float = 0.165
    delta_mev: float = 5.0
    u_mev: float = 79.7
    up_mev: float = 97.5
    kinetic_ev: float = 2.1354
    cutoff_shell: int = 4

    def b0_params(self) -> TBGParameters:
        return TBGParameters(
            dtheta_rad=float(self.theta_deg) * math.pi / 180.0,
            convention="b0",
            vf=1000.0 * float(self.kinetic_ev),
            w0=float(self.u_mev),
            w1=float(self.up_mev),
            strain=float(self.strain),
            strain_angle_rad=float(self.strain_angle_deg) * math.pi / 180.0,
            poisson=float(self.poisson),
            beta_g=3.14,
            alpha=0.5,
            deformation_potential=0.0,
        )

    def config(self, *, valley: int = 1) -> AroraTBGConfig:
        return AroraTBGConfig(
            theta_deg=float(self.theta_deg),
            graphene_lattice_constant_nm=GRAPHENE_A_NM,
            kinetic_ev=float(self.kinetic_ev),
            w_ab_ev=float(self.up_mev) * 1.0e-3,
            w_aa_ratio=float(self.u_mev) / float(self.up_mev),
            delta1_ev=float(self.delta_mev) * 1.0e-3,
            delta2_ev=float(self.delta_mev) * 1.0e-3,
            valley=int(valley),
            # The zero_field.model.dirac(q,zeta) b0 convention used below has
            # the paper Eq. (6) sign absorbed in the layer Dirac-point
            # orientation.  Using AroraTBGConfig's default -1 sign flips the
            # Fig. 2 band ordering and spoils the response comparison.
            dirac_sign=1.0,
        )


def circular_reciprocal_indices(cutoff_shell: int = 4) -> np.ndarray:
    """Return the 61 reciprocal-lattice integer pairs for ``cutoff_shell=4``.

    The shell metric is the unstrained triangular reciprocal metric
    ``m1^2 + m2^2 - m1*m2``.  For ``cutoff_shell=4`` this gives exactly 61
    reciprocal sites, hence a ``4 * 61 = 244`` one-valley Hamiltonian, matching
    the paper's Fig. 2 truncation.
    """

    cutoff = int(cutoff_shell)
    if cutoff <= 0:
        raise ValueError(f"cutoff_shell must be positive, got {cutoff_shell}")
    pairs: list[tuple[int, int, int]] = []
    radius2 = cutoff * cutoff
    for m1 in range(-cutoff, cutoff + 1):
        for m2 in range(-cutoff, cutoff + 1):
            shell2 = m1 * m1 + m2 * m2 - m1 * m2
            if shell2 <= radius2:
                pairs.append((shell2, m1, m2))
    pairs.sort(key=lambda item: (item[0], item[2], item[1]))
    return np.asarray([(m1, m2) for _shell2, m1, m2 in pairs], dtype=int)


def gvec_from_indices(params: TBGParameters, indices: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=int)
    if idx.ndim != 2 or idx.shape[1] != 2:
        raise ValueError(f"indices must have shape (n,2), got {idx.shape}")
    return (idx[:, 0] * params.g1 + idx[:, 1] * params.g2).astype(np.complex128)


def _validate_valley(valley: int) -> int:
    zeta = int(valley)
    if zeta not in (-1, 1):
        raise ValueError(f"valley must be +/-1, got {valley}")
    return zeta


def _build_circular_tunnel(params: TBGParameters, indices: np.ndarray, valley: int) -> np.ndarray:
    """Build finite-cutoff interlayer tunnelling for arbitrary reciprocal sites."""

    zeta = _validate_valley(valley)
    idx = np.asarray(indices, dtype=int)
    n_g = int(idx.shape[0])
    dim = 4 * n_g
    t12 = np.zeros((dim, dim), dtype=np.complex128)
    coord_to_i = {(int(m1), int(m2)): i for i, (m1, m2) in enumerate(idx)}
    if zeta == 1:
        t0, t1, t2 = params.t0, params.t1, params.t2
    else:
        t0, t1, t2 = params.t0, params.t2, params.t1
    for i, (ix, iy) in enumerate(idx):
        left = 4 * i
        neighbors = (
            (int(ix) + zeta, int(iy) - zeta, t2),
            (int(ix), int(iy) - zeta, t1),
            (int(ix) + zeta, int(iy), t0),
        )
        for nx, ny, tunnel in neighbors:
            j = coord_to_i.get((nx, ny))
            if j is None:
                continue
            right = 4 * int(j)
            t12[left + 2 : left + 4, right : right + 2] = tunnel
            t12[right : right + 2, left + 2 : left + 4] = tunnel.conjugate().T
    return t12


def _b0_dirac_derivative_blocks(valley: int) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of ``dirac(q, valley)`` wrt ``(qx,qy)`` in b0 units."""

    zeta = _validate_valley(valley)
    dqx = np.asarray([[0.0, float(zeta)], [float(zeta), 0.0]], dtype=np.complex128)
    dqy = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    return dqx, dqy


def _paper_eq10_relative_gauge_shift(params: TBGParameters) -> np.ndarray:
    """Relative pseudo-gauge field used in the strained TBG Dirac points.

    The physically standard graphene strain gauge field is proportional to
    ``(E_xx - E_yy, -2 E_xy)``. This is also the convention in the generic
    ``TBGParameters.gauge_shift``. Keep this standard convention here; the main
    Arora-specific fix in this adapter is instead that the Dirac blocks must
    carry the paper Eq. (6) minus sign through ``config.dirac_sign``.
    """

    return np.asarray(params.gauge_shift, dtype=float)


def _paper_eq10_dirac_points(params: TBGParameters) -> tuple[complex, complex]:
    """Return ``(kt, kb)`` Dirac points following Arora Eq. (8)--(10).

    The generic b0 coordinates are retained, but the pseudo-gauge displacement
    is rebuilt from the paper Eq. (10) convention above instead of reusing the
    project-wide ``params.kt`` / ``params.kb_point`` values.
    """

    kb_mag = 8.0 * math.pi / 3.0 * math.sin(float(params.dtheta_rad) / 2.0)
    kt0 = kb_mag / 2.0 * complex(math.cos(math.pi / 2.0), math.sin(math.pi / 2.0))
    kb0 = -kt0
    gauge = _paper_eq10_relative_gauge_shift(params)
    gauge_c = complex(float(gauge[0]), float(gauge[1]))
    strain_k = complex(float(params.strain_matrix[0, 0]), float(params.strain_matrix[1, 0])) * (4.0 * math.pi / 3.0)
    alpha = float(params.alpha)
    kt = kt0 + gauge_c * alpha - strain_k * alpha - params.g1 / 2.0
    kb = kb0 - gauge_c * (1.0 - alpha) + strain_k * (1.0 - alpha) + params.g1 / 2.0
    return complex(kt), complex(kb)


def _construct_circular_diagonal_block(
    params: TBGParameters,
    gvec: np.ndarray,
    k: complex,
    valley: int,
    *,
    sigma_rotation: bool = True,
    dirac_sign: float = -1.0,
) -> np.ndarray:
    """Diagonal Dirac blocks for an arbitrary reciprocal-site list, in meV."""

    zeta = _validate_valley(valley)
    g = np.asarray(gvec, dtype=np.complex128)
    n_g = int(g.size)
    dim = 4 * n_g
    h = np.zeros((dim, dim), dtype=np.complex128)
    sigma0 = np.eye(2, dtype=np.complex128)
    rotation = -params.dtheta_rad / 2.0 * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    div_u = float((params.strain_matrix[0, 0] + params.strain_matrix[1, 1]) / 2.0)
    kt_point, kb_point = _paper_eq10_dirac_points(params)
    for ig, qc in enumerate(g):
        if zeta == 1:
            kb = k - kb_point + qc
            kt = k - kt_point + qc
        else:
            kb = k - kt_point + qc
            kt = k - kb_point + qc
        if sigma_rotation:
            k1_vec = (np.eye(2) + rotation - params.strain_matrix * params.alpha) @ np.asarray(
                [kb.real, kb.imag], dtype=float
            )
            k2_vec = (np.eye(2) - rotation + params.strain_matrix * (1.0 - params.alpha)) @ np.asarray(
                [kt.real, kt.imag], dtype=float
            )
        else:
            k1_vec = np.asarray([kb.real, kb.imag], dtype=float)
            k2_vec = np.asarray([kt.real, kt.imag], dtype=float)
        left = 4 * ig
        h[left : left + 2, left : left + 2] = (
            float(dirac_sign) * params.vf * dirac(complex(k1_vec[0], k1_vec[1]), zeta, 0.0)
            - (params.deformation_potential * div_u) * sigma0
        )
        h[left + 2 : left + 4, left + 2 : left + 4] = (
            float(dirac_sign) * params.vf * dirac(complex(k2_vec[0], k2_vec[1]), zeta, 0.0)
            + (params.deformation_potential * div_u) * sigma0
        )
    return h


def analytic_arora2021_b0_dhdk(
    params: TBGParameters,
    config: AroraTBGConfig,
    *,
    cutoff_shell: int = 4,
    sigma_rotation: bool = True,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic ``dH/dk`` for the circular-cutoff b0 Hamiltonian.

    The returned matrices are in ``eV nm``, matching the convention used by
    ``analysis.response_derivative_gauge`` and the older TBG b0 adapters.  The
    Hamiltonian is affine in the Bloch momentum; interlayer tunnelling and hBN
    mass terms are k-independent.
    """

    zeta = _validate_valley(config.valley)
    resolved_indices = circular_reciprocal_indices(cutoff_shell) if indices is None else np.asarray(indices, dtype=int)
    n_g = int(resolved_indices.shape[0])
    dim = 4 * n_g
    out_x = np.zeros((dim, dim), dtype=np.complex128)
    out_y = np.zeros((dim, dim), dtype=np.complex128)
    rotation = -params.dtheta_rad / 2.0 * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    if sigma_rotation:
        layer1_map = np.eye(2) + rotation - params.strain_matrix * params.alpha
        layer2_map = np.eye(2) - rotation + params.strain_matrix * (1.0 - params.alpha)
    else:
        layer1_map = np.eye(2)
        layer2_map = np.eye(2)
    dqx, dqy = _b0_dirac_derivative_blocks(zeta)

    def block_for_axis(transform: np.ndarray, axis: int) -> np.ndarray:
        return float(config.dirac_sign) * float(params.vf) * (float(transform[0, axis]) * dqx + float(transform[1, axis]) * dqy)

    for ig in range(n_g):
        base = 4 * ig
        for local, transform in ((slice(base, base + 2), layer1_map), (slice(base + 2, base + 4), layer2_map)):
            out_x[local, local] = block_for_axis(transform, 0)
            out_y[local, local] = block_for_axis(transform, 1)
    scale = 1.0e-3 * float(config.graphene_lattice_constant_nm)
    return out_x * scale, out_y * scale


def finite_difference_arora2021_b0_dhdk(
    params: TBGParameters,
    config: AroraTBGConfig,
    *,
    cutoff_shell: int = 4,
    sigma_rotation: bool = True,
    step_dimless: float = 1.0e-6,
    indices: np.ndarray | None = None,
    gvec: np.ndarray | None = None,
    tunnel: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference derivative for validating the analytic route."""

    step = float(step_dimless)
    if step <= 0.0:
        raise ValueError(f"step_dimless must be positive, got {step_dimless}")
    hp = build_arora2021_b0_hamiltonian(step + 0.0j, params, config, cutoff_shell=cutoff_shell, sigma_rotation=sigma_rotation, indices=indices, gvec=gvec, tunnel=tunnel)
    hm = build_arora2021_b0_hamiltonian(-step + 0.0j, params, config, cutoff_shell=cutoff_shell, sigma_rotation=sigma_rotation, indices=indices, gvec=gvec, tunnel=tunnel)
    hpy = build_arora2021_b0_hamiltonian(1.0j * step, params, config, cutoff_shell=cutoff_shell, sigma_rotation=sigma_rotation, indices=indices, gvec=gvec, tunnel=tunnel)
    hmy = build_arora2021_b0_hamiltonian(-1.0j * step, params, config, cutoff_shell=cutoff_shell, sigma_rotation=sigma_rotation, indices=indices, gvec=gvec, tunnel=tunnel)
    scale_to_nm = float(config.graphene_lattice_constant_nm)
    return (hp - hm) / (2.0 * step) * scale_to_nm, (hpy - hmy) / (2.0 * step) * scale_to_nm


def _add_layer_sublattice_offsets_mev(
    hamiltonian_mev: np.ndarray,
    *,
    n_g: int,
    delta1_mev: float,
    delta2_mev: float,
) -> np.ndarray:
    out = np.array(hamiltonian_mev, dtype=np.complex128, copy=True)
    for ig in range(int(n_g)):
        base = 4 * ig
        out[base, base] += float(delta1_mev)
        out[base + 1, base + 1] -= float(delta1_mev)
        out[base + 2, base + 2] += float(delta2_mev)
        out[base + 3, base + 3] -= float(delta2_mev)
    return out


def build_arora2021_b0_hamiltonian(
    k_dimless: complex,
    params: TBGParameters,
    config: AroraTBGConfig,
    *,
    cutoff_shell: int = 4,
    sigma_rotation: bool = True,
    indices: np.ndarray | None = None,
    gvec: np.ndarray | None = None,
    tunnel: np.ndarray | None = None,
) -> np.ndarray:
    """Build the paper-compatible circular-cutoff b0 Hamiltonian in eV."""

    zeta = _validate_valley(config.valley)
    resolved_indices = circular_reciprocal_indices(cutoff_shell) if indices is None else np.asarray(indices, dtype=int)
    resolved_gvec = gvec_from_indices(params, resolved_indices) if gvec is None else np.asarray(gvec, dtype=np.complex128)
    resolved_tunnel = (
        _build_circular_tunnel(params, resolved_indices, zeta)
        if tunnel is None
        else np.asarray(tunnel, dtype=np.complex128)
    )
    n_g = int(resolved_indices.shape[0])
    h_mev = _construct_circular_diagonal_block(
        params,
        resolved_gvec,
        complex(k_dimless),
        zeta,
        sigma_rotation=bool(sigma_rotation),
        dirac_sign=float(config.dirac_sign),
    )
    h_mev = h_mev + resolved_tunnel
    h_mev = _add_layer_sublattice_offsets_mev(
        h_mev,
        n_g=n_g,
        delta1_mev=1000.0 * float(config.delta1_ev),
        delta2_mev=1000.0 * float(config.delta2_ev),
    )
    return h_mev * 1.0e-3








__all__ = [
    "AroraTBGConfig",
    "Arora2021Fig2Parameters",
    "analytic_arora2021_b0_dhdk",
    "build_arora2021_b0_hamiltonian",
    "circular_reciprocal_indices",
    "finite_difference_arora2021_b0_dhdk",
    "gvec_from_indices",
]
