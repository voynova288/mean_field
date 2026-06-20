from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar

from ...core.bands import estimate_central_pair_metrics
from .bands import compute_bands_on_grid
from .domains import HTQGDomain, canonical_domain_key
from .hamiltonian import build_hamiltonian
from .lattice import HTQGLattice, build_htqg_lattice
from .params import HTQGParams, theta_deg_from_alpha

PAPER_MAGIC_ALPHA: dict[str, tuple[float, ...]] = {
    "alpha_beta_gamma": (0.322, 0.931, 1.30, 2.12, 2.39, 3.09, 3.45, 4.18),
    "alpha_beta_alpha": (0.320, 0.945, 1.50, 2.40, 2.62, 3.69, 4.08, 4.78),
}


@dataclass(frozen=True)
class MagicScanPoint:
    alpha: float
    theta_deg: float
    mean_flat_bandwidth_ev: float
    central_manifold_span_ev: float

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha": float(self.alpha),
            "theta_deg": float(self.theta_deg),
            "mean_flat_bandwidth_ev": float(self.mean_flat_bandwidth_ev),
            "central_manifold_span_ev": float(self.central_manifold_span_ev),
        }


def build_chiral_lattice_from_alpha(
    alpha: float,
    *,
    n_shells: int = 5,
    params: HTQGParams | None = None,
) -> HTQGLattice:
    resolved = params if params is not None else HTQGParams.chiral()
    theta_deg = theta_deg_from_alpha(alpha, params=resolved)
    return build_htqg_lattice(
        theta_deg,
        n_shells=n_shells,
        graphene_lattice_constant_nm=resolved.graphene_lattice_constant_nm,
    )


def sublattice_sigma_z(lattice: HTQGLattice) -> np.ndarray:
    """Layer-resolved graphene sublattice operator in the HTQG basis."""

    pattern = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=float)
    return np.diag(np.tile(pattern, lattice.n_g)).astype(np.complex128)


def chiral_symmetry_residual(
    k_tilde: complex,
    lattice: HTQGLattice,
    params: HTQGParams | None = None,
    *,
    domain: str | HTQGDomain = "alpha_beta_alpha",
    valley: int = 1,
) -> float:
    resolved = params if params is not None else HTQGParams.chiral()
    hmat = build_hamiltonian(k_tilde, lattice, resolved, domain=domain, valley=valley)
    sigma_z = sublattice_sigma_z(lattice)
    return float(np.max(np.abs(hmat @ sigma_z + sigma_z @ hmat)))


def central_pair_bandwidth_on_grid(
    alpha: float,
    *,
    domain: str | HTQGDomain = "alpha_beta_alpha",
    mesh_size: int = 7,
    n_shells: int = 4,
    params: HTQGParams | None = None,
) -> MagicScanPoint:
    """Compute a lightweight central-pair bandwidth diagnostic at chiral kappa=0.

    A paper-grade Table II reproduction requires larger cutoffs, denser grids,
    and Slurm execution; this helper provides the reusable code path.
    """

    resolved = params if params is not None else HTQGParams.chiral()
    lattice = build_chiral_lattice_from_alpha(alpha, n_shells=n_shells, params=resolved)
    result = compute_bands_on_grid(
        mesh_size,
        lattice,
        resolved,
        domain=domain,
        central_band_count=4,
        return_eigenvectors=False,
    )
    metrics = estimate_central_pair_metrics(result, lattice.matrix_dim)
    return MagicScanPoint(
        alpha=float(alpha),
        theta_deg=float(lattice.theta_deg),
        mean_flat_bandwidth_ev=float(metrics["mean_flat_bandwidth_ev"] or np.nan),
        central_manifold_span_ev=float(metrics["central_manifold_span_ev"] or np.nan),
    )


def scan_magic_bandwidths(
    alphas: Iterable[float],
    *,
    domain: str | HTQGDomain = "alpha_beta_alpha",
    mesh_size: int = 7,
    n_shells: int = 4,
    params: HTQGParams | None = None,
) -> tuple[MagicScanPoint, ...]:
    return tuple(
        central_pair_bandwidth_on_grid(
            float(alpha),
            domain=domain,
            mesh_size=mesh_size,
            n_shells=n_shells,
            params=params,
        )
        for alpha in alphas
    )


def locate_magic_angle(
    bracket: tuple[float, float],
    *,
    domain: str | HTQGDomain = "alpha_beta_alpha",
    mesh_size: int = 7,
    n_shells: int = 4,
    params: HTQGParams | None = None,
) -> MagicScanPoint:
    """Locate a bandwidth minimum in a bracket.

    This is intentionally an execution helper, not a claim that the minimum is
    validated.  Use Slurm and convergence sweeps before comparing to Table II.
    """

    resolved = params if params is not None else HTQGParams.chiral()

    def objective(alpha: float) -> float:
        return central_pair_bandwidth_on_grid(
            alpha,
            domain=domain,
            mesh_size=mesh_size,
            n_shells=n_shells,
            params=resolved,
        ).central_manifold_span_ev

    opt = minimize_scalar(objective, bounds=(float(bracket[0]), float(bracket[1])), method="bounded")
    return central_pair_bandwidth_on_grid(
        float(opt.x),
        domain=domain,
        mesh_size=mesh_size,
        n_shells=n_shells,
        params=resolved,
    )


def paper_magic_alpha(domain: str | HTQGDomain) -> tuple[float, ...]:
    return PAPER_MAGIC_ALPHA[canonical_domain_key(domain)]


__all__ = [
    "MagicScanPoint",
    "PAPER_MAGIC_ALPHA",
    "build_chiral_lattice_from_alpha",
    "central_pair_bandwidth_on_grid",
    "chiral_symmetry_residual",
    "locate_magic_angle",
    "paper_magic_alpha",
    "scan_magic_bandwidths",
    "sublattice_sigma_z",
]
