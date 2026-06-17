from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bands import compute_bands_along_path, compute_bands_on_grid
from .domains import HTQGDomain, domain_displacements
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .lattice import HTQGLattice, build_htqg_lattice, build_standard_kpath
from .params import DEFAULT_THETA_DEG, HTQGParams


@dataclass(frozen=True)
class HTQGModel:
    """Convenience bundle for a Fujimoto-2025 HTQG domain calculation."""

    lattice: HTQGLattice
    params: HTQGParams
    domain: HTQGDomain
    valley: int = 1

    @classmethod
    def default(
        cls,
        *,
        theta_deg: float = DEFAULT_THETA_DEG,
        n_shells: int = 4,
        domain: str | HTQGDomain = "alpha_beta_alpha",
        params: HTQGParams | None = None,
        valley: int = 1,
    ) -> "HTQGModel":
        resolved_params = params if params is not None else HTQGParams.default()
        lattice = build_htqg_lattice(
            theta_deg,
            n_shells=n_shells,
            graphene_lattice_constant_nm=resolved_params.graphene_lattice_constant_nm,
        )
        return cls(lattice=lattice, params=resolved_params, domain=domain_displacements(lattice, domain), valley=int(valley))

    def hamiltonian(self, k_tilde: complex) -> np.ndarray:
        return build_hamiltonian(k_tilde, self.lattice, self.params, domain=self.domain, valley=self.valley)

    def diagonalize(self, k_tilde: complex, **kwargs):
        return diagonalize_hamiltonian(k_tilde, self.lattice, self.params, domain=self.domain, valley=self.valley, **kwargs)

    def standard_path(self, points_per_segment: int = 120):
        return build_standard_kpath(self.lattice, points_per_segment=points_per_segment)

    def path_bands(self, *, points_per_segment: int = 120, **kwargs):
        return compute_bands_along_path(
            self.standard_path(points_per_segment=points_per_segment),
            self.lattice,
            self.params,
            domain=self.domain,
            valley=self.valley,
            **kwargs,
        )

    def grid_bands(self, mesh_size: int, **kwargs):
        return compute_bands_on_grid(
            mesh_size,
            self.lattice,
            self.params,
            domain=self.domain,
            valley=self.valley,
            **kwargs,
        )


__all__ = ["HTQGModel"]
