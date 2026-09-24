"""Liu--Dai 2020 hBN-aligned TBG optical benchmark adapter.

This module is intentionally thin: it supplies one-particle hBN-TBG k-point
payloads to the generic optical modules.  It does not own optical formulas,
plotting, or Slurm orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import eigh

from analysis.optical import OpticalKPointData, optical_kpoint_data_from_eigensystem
from mean_field.systems.tbg.arora2021 import (
    AroraTBGConfig,
    GRAPHENE_A_NM,
    analytic_arora2021_b0_dhdk,
    build_arora2021_b0_hamiltonian,
    circular_reciprocal_indices,
    gvec_from_indices,
    _build_circular_tunnel,
)
from mean_field.systems.tbg.params import TBGParameters


@dataclass(frozen=True)
class LiuDai2020TBGParameters:
    """hBN-aligned TBG parameters from Liu--Dai 2020 Methods.

    The paper uses twist angle ``1.05 deg``, hBN bottom-layer staggered mass
    ``Delta=17 meV``, ``hbar v_F=5.253 eV Angstrom``, and BM tunneling
    ``u=79.7 meV``, ``u'=97.5 meV``.  The internal b0 adapter uses a
    dimensionless momentum convention, so ``kinetic_ev = (hbar v_F)/a``.

    Liu--Dai Methods Eq. (12) writes the layer Dirac blocks as
    ``-hbar v_F (k-K_l) . [mu sigma_x, sigma_y]`` with the layer twist carried by
    ``K1/K2`` offsets.  It does not include the extra layer Pauli rotation used
    by some BM gauges, so the Liu--Dai adapter defaults ``sigma_rotation`` to
    ``False``.  Keep this parameter explicit because other paper adapters using
    the shared Arora b0 helpers may require the rotated gauge.
    """

    theta_deg: float = 1.05
    delta_bottom_mev: float = 17.0
    u_mev: float = 79.7
    up_mev: float = 97.5
    hbar_vf_ev_nm: float = 0.5253
    cutoff_shell: int = 4
    sigma_rotation: bool = False

    @property
    def kinetic_ev(self) -> float:
        return float(self.hbar_vf_ev_nm) / GRAPHENE_A_NM

    def b0_params(self) -> TBGParameters:
        return TBGParameters(
            dtheta_rad=float(self.theta_deg) * math.pi / 180.0,
            convention="b0",
            vf=1000.0 * self.kinetic_ev,
            w0=float(self.u_mev),
            w1=float(self.up_mev),
            strain=0.0,
            strain_angle_rad=0.0,
            poisson=0.165,
            beta_g=3.14,
            alpha=0.5,
            deformation_potential=0.0,
        )

    def config(self, *, valley: int = 1) -> AroraTBGConfig:
        return AroraTBGConfig(
            theta_deg=float(self.theta_deg),
            graphene_lattice_constant_nm=GRAPHENE_A_NM,
            kinetic_ev=self.kinetic_ev,
            w_ab_ev=float(self.up_mev) * 1.0e-3,
            w_aa_ratio=float(self.u_mev) / float(self.up_mev),
            # Liu--Dai Eq. (11): bottom-layer mass only.  In the shared b0
            # basis the first 2x2 block is the bottom-layer block.
            delta1_ev=float(self.delta_bottom_mev) * 1.0e-3,
            delta2_ev=0.0,
            valley=int(valley),
            dirac_sign=1.0,
        )


def valley_spin_energy_shift_ev(valley: int, spin: int, *, valley_split_mev: float = 0.0, spin_split_mev: float = 0.0) -> float:
    """Scalar Liu--Dai Eq. (1) energy shift for a fixed valley/spin sector."""

    if int(valley) not in (-1, 1):
        raise ValueError(f"valley must be +/-1, got {valley}")
    if int(spin) not in (-1, 1):
        raise ValueError(f"spin must be +/-1, got {spin}")
    return (int(valley) * float(valley_split_mev) + int(spin) * float(spin_split_mev)) * 1.0e-3


def parallelogram_kmesh(params: TBGParameters, mesh_size: int, *, centered: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Uniform moire reciprocal-cell parallelogram k mesh.

    Returned k points use the b0 dimensionless convention.  Weights are BZ area
    elements in ``nm^-2`` using ``k_dimless / graphene_a_nm`` as the physical
    reciprocal coordinate.  This is the appropriate weight convention for
    optical helpers when ``dH/dk`` has been converted to ``eV nm``.
    """

    mesh = int(mesh_size)
    if mesh <= 0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")
    area_dimless = abs(float(np.imag(np.conjugate(params.g1) * params.g2)))
    area_nm_inv_sq = area_dimless / (GRAPHENE_A_NM * GRAPHENE_A_NM)
    weight = area_nm_inv_sq / (mesh * mesh)
    points: list[complex] = []
    weights: list[float] = []
    offset = 0.5 if centered else 0.0
    for i in range(mesh):
        u = (i + offset) / mesh - 0.5
        for j in range(mesh):
            v = (j + offset) / mesh - 0.5
            points.append(u * params.g1 + v * params.g2)
            weights.append(weight)
    return np.asarray(points, dtype=np.complex128), np.asarray(weights, dtype=float)


def liu_dai2020_tbg_eigensystem(
    k_dimless: complex,
    params: TBGParameters,
    config: AroraTBGConfig,
    *,
    cutoff_shell: int = 4,
    sigma_rotation: bool = False,
    indices: np.ndarray | None = None,
    gvec: np.ndarray | None = None,
    tunnel: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(energies_ev, eigenvectors, dhdk)`` for one valley k point.

    The default ``sigma_rotation=False`` follows Liu--Dai Methods Eq. (12), where
    layer twist enters through ``K1/K2`` and the same ``sigma_mu`` matrices are
    used in both diagonal Dirac blocks.
    """

    resolved_indices = circular_reciprocal_indices(cutoff_shell) if indices is None else np.asarray(indices, dtype=int)
    resolved_gvec = gvec_from_indices(params, resolved_indices) if gvec is None else np.asarray(gvec, dtype=np.complex128)
    resolved_tunnel = _build_circular_tunnel(params, resolved_indices, config.valley) if tunnel is None else np.asarray(tunnel, dtype=np.complex128)
    hamiltonian = build_arora2021_b0_hamiltonian(
        complex(k_dimless),
        params,
        config,
        cutoff_shell=cutoff_shell,
        indices=resolved_indices,
        gvec=resolved_gvec,
        tunnel=resolved_tunnel,
        sigma_rotation=bool(sigma_rotation),
    )
    energies, eigenvectors = eigh(hamiltonian)
    dhdk = np.asarray(
        analytic_arora2021_b0_dhdk(
            params,
            config,
            cutoff_shell=cutoff_shell,
            sigma_rotation=bool(sigma_rotation),
            indices=resolved_indices,
        ),
        dtype=np.complex128,
    )
    return np.asarray(energies, dtype=float), np.asarray(eigenvectors, dtype=np.complex128), dhdk


@dataclass(frozen=True)
class LiuDai2020TBGValleyKPointData:
    """Cached one-valley k-point data before Liu--Dai Eq. (1) spin/valley shifts."""

    valley: int
    k_dimless: complex
    weight: float
    energies_ev: np.ndarray
    velocity_h: np.ndarray


def liu_dai2020_tbg_valley_kpoint_data(
    *,
    mesh_size: int,
    paper: LiuDai2020TBGParameters = LiuDai2020TBGParameters(),
    valleys: Iterable[int] = (-1, 1),
    cutoff_shell: int | None = None,
) -> list[LiuDai2020TBGValleyKPointData]:
    """Precompute reusable valley k-point data for repeated split scans.

    Valley/spin splittings in Liu--Dai Eq. (1) are scalar energy shifts.  They
    do not change eigenvectors or velocity matrices, so Fig. 3-style scans over
    ``E_v`` and ``E_s`` should reuse this cache instead of re-diagonalizing for
    every split pair.
    """

    shell = int(paper.cutoff_shell if cutoff_shell is None else cutoff_shell)
    valley_list = [int(v) for v in valleys]
    params = paper.b0_params()
    kpoints, weights = parallelogram_kmesh(params, mesh_size)
    indices = circular_reciprocal_indices(shell)
    gvec = gvec_from_indices(params, indices)
    cache = {valley: _build_circular_tunnel(params, indices, valley) for valley in set(valley_list)}
    out: list[LiuDai2020TBGValleyKPointData] = []
    for k, weight in zip(kpoints, weights, strict=True):
        for valley in valley_list:
            config = paper.config(valley=valley)
            energies, evecs, dhdk = liu_dai2020_tbg_eigensystem(
                k,
                params,
                config,
                cutoff_shell=shell,
                indices=indices,
                gvec=gvec,
                tunnel=cache[valley],
                sigma_rotation=bool(paper.sigma_rotation),
            )
            payload = optical_kpoint_data_from_eigensystem(
                energies,
                evecs,
                dhdk,
                weight=float(weight),
                mu_ev=0.0,
                temperature_k=0.0,
            )
            out.append(
                LiuDai2020TBGValleyKPointData(
                    valley=valley,
                    k_dimless=complex(k),
                    weight=float(weight),
                    energies_ev=payload.energies_ev,
                    velocity_h=payload.velocity_h,
                )
            )
    return out


def plus_three_quarter_fermi_level(sampled_sector_energies: Sequence[np.ndarray]) -> float:
    """Estimate a zero-temperature +3/4 flat-band Fermi level on a uniform mesh.

    Each k point must include all four spin/valley sectors with the same band
    count.  The target occupation is neutrality plus three of the four
    conduction flat-band states per k point, i.e. ``4*(nb/2) + 3`` occupied
    states per k.  This helper is for benchmark smoke runs; serious runs should
    record the filling convention and degeneracy handling explicitly.
    """

    blocks = [np.asarray(block, dtype=float) for block in sampled_sector_energies]
    if not blocks:
        raise ValueError("No sector energies supplied")
    nb = int(blocks[0].size)
    if nb % 2 != 0:
        raise ValueError(f"Expected an even band count, got {nb}")
    if any(block.size != nb for block in blocks):
        raise ValueError("All sector energy arrays must have the same size")
    if len(blocks) % 4 != 0:
        raise ValueError("Expected four spin/valley sectors per k point")
    nk = len(blocks) // 4
    target = nk * (4 * (nb // 2) + 3)
    flat = np.sort(np.concatenate(blocks))
    target = max(1, min(int(target), flat.size - 1))
    return 0.5 * (float(flat[target - 1]) + float(flat[target]))


def liu_dai2020_tbg_optical_payloads_from_valley_data(
    valley_data: Sequence[LiuDai2020TBGValleyKPointData],
    *,
    valley_split_mev: float = 0.0,
    spin_split_mev: float = 0.0,
    mu_ev: float | None = None,
    temperature_k: float = 0.0,
    spins: Iterable[int] = (-1, 1),
) -> tuple[list[OpticalKPointData], float]:
    """Apply Eq. (1) spin/valley shifts to cached valley k-point data."""

    spin_list = [int(s) for s in spins]
    sampled: list[np.ndarray] = []
    sectors: list[tuple[np.ndarray, np.ndarray, float]] = []
    for point in valley_data:
        for spin in spin_list:
            shifted = point.energies_ev + valley_spin_energy_shift_ev(
                point.valley,
                spin,
                valley_split_mev=valley_split_mev,
                spin_split_mev=spin_split_mev,
            )
            sampled.append(shifted)
            sectors.append((shifted, point.velocity_h, point.weight))
    resolved_mu = plus_three_quarter_fermi_level(sampled) if mu_ev is None else float(mu_ev)
    if float(temperature_k) <= 0.0:
        occ_fn = lambda energies: (np.asarray(energies, dtype=float) < resolved_mu).astype(float)
    else:
        from analysis.shift_current import fermi_occupation

        occ_fn = lambda energies: fermi_occupation(np.asarray(energies, dtype=float), mu_ev=resolved_mu, temperature_k=temperature_k)
    payloads = [
        OpticalKPointData(
            energies_ev=np.asarray(energies, dtype=float),
            velocity_h=np.asarray(velocity_h, dtype=np.complex128),
            occupations=occ_fn(energies),
            weight=float(weight),
        )
        for energies, velocity_h, weight in sectors
    ]
    return payloads, resolved_mu


def liu_dai2020_tbg_optical_kpoint_data(
    *,
    mesh_size: int,
    paper: LiuDai2020TBGParameters = LiuDai2020TBGParameters(),
    valley_split_mev: float = 0.0,
    spin_split_mev: float = 0.0,
    mu_ev: float | None = None,
    temperature_k: float = 0.0,
    valleys: Iterable[int] = (-1, 1),
    spins: Iterable[int] = (-1, 1),
    cutoff_shell: int | None = None,
) -> tuple[list[OpticalKPointData], float]:
    """Build optical k-point payloads for hBN-aligned TBG smoke/benchmark runs.

    Returns ``(payloads, mu_ev)``.  If ``mu_ev`` is omitted, a coarse +3/4
    flat-band Fermi level is estimated from the sampled spin/valley sectors.
    """

    valley_data = liu_dai2020_tbg_valley_kpoint_data(
        mesh_size=mesh_size,
        paper=paper,
        valleys=valleys,
        cutoff_shell=cutoff_shell,
    )
    return liu_dai2020_tbg_optical_payloads_from_valley_data(
        valley_data,
        valley_split_mev=valley_split_mev,
        spin_split_mev=spin_split_mev,
        mu_ev=mu_ev,
        temperature_k=temperature_k,
        spins=spins,
    )


__all__ = [
    "LiuDai2020TBGParameters",
    "LiuDai2020TBGValleyKPointData",
    "liu_dai2020_tbg_eigensystem",
    "liu_dai2020_tbg_optical_kpoint_data",
    "liu_dai2020_tbg_optical_payloads_from_valley_data",
    "liu_dai2020_tbg_valley_kpoint_data",
    "parallelogram_kmesh",
    "plus_three_quarter_fermi_level",
    "valley_spin_energy_shift_ev",
]
