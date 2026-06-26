from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.hf import ComponentGroup
from ...core.lattice import KPath
from .bands import GridBandsResult, PathBandsResult, compute_bands_along_path, compute_bands_on_grid
from .domains import HTQGDomain, domain_displacements
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .lattice import HTQGLattice, build_htqg_lattice, build_standard_kpath
from .params import DEFAULT_THETA_DEG, HTQGParams
from .topology import FHSState, fhs_state_on_grid


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

    @property
    def theta_deg(self) -> float:
        return float(self.lattice.theta_deg)

    @property
    def n_shells(self) -> int:
        return int(self.lattice.n_shells)

    @property
    def matrix_dim(self) -> int:
        return int(self.lattice.matrix_dim)

    def lattice_summary(self) -> dict[str, object]:
        summary = self.lattice.to_summary_dict()
        summary.update(
            {
                "domain": self.domain.key,
                "domain_label": self.domain.label,
                "valley": int(self.valley),
                "kappa": float(self.params.kappa),
                "w_ev": float(self.params.w_ev),
                "vf_ev_nm": float(self.params.vf_ev_nm),
                "alpha": float(self.params.alpha(self.lattice.k_theta)),
                "model_name": self.params.model_name,
            }
        )
        return summary

    def component_groups(self) -> tuple[ComponentGroup, ...]:
        """Return layer and sublattice groups in the local HTQG basis cell."""

        return (
            ComponentGroup("layer_0", np.asarray([0, 1], dtype=int)),
            ComponentGroup("layer_1", np.asarray([2, 3], dtype=int)),
            ComponentGroup("layer_2", np.asarray([4, 5], dtype=int)),
            ComponentGroup("layer_3", np.asarray([6, 7], dtype=int)),
            ComponentGroup("sublattice_A", np.asarray([0, 2, 4, 6], dtype=int)),
            ComponentGroup("sublattice_B", np.asarray([1, 3, 5, 7], dtype=int)),
        )

    def build_hamiltonian(self, k_tilde: complex, *, valley: int | None = None) -> np.ndarray:
        resolved_valley = self.valley if valley is None else int(valley)
        return build_hamiltonian(k_tilde, self.lattice, self.params, domain=self.domain, valley=resolved_valley)

    def hamiltonian(self, k_tilde: complex) -> np.ndarray:
        return self.build_hamiltonian(k_tilde)

    def diagonalize(self, k_tilde: complex, *, valley: int | None = None, **kwargs):
        resolved_valley = self.valley if valley is None else int(valley)
        return diagonalize_hamiltonian(k_tilde, self.lattice, self.params, domain=self.domain, valley=resolved_valley, **kwargs)

    def standard_kpath(self, *, points_per_segment: int = 120) -> KPath:
        return build_standard_kpath(self.lattice, points_per_segment=points_per_segment)

    def standard_path(self, *, points_per_segment: int = 120) -> KPath:
        return self.standard_kpath(points_per_segment=points_per_segment)

    def bands_along_path(
        self,
        path: KPath,
        *,
        valley: int | None = None,
        band_indices: tuple[int, ...] | None = None,
        central_band_count: int | None = None,
        return_eigenvectors: bool = False,
    ) -> PathBandsResult:
        resolved_valley = self.valley if valley is None else int(valley)
        return compute_bands_along_path(
            path,
            self.lattice,
            self.params,
            domain=self.domain,
            valley=resolved_valley,
            band_indices=band_indices,
            central_band_count=central_band_count,
            return_eigenvectors=return_eigenvectors,
        )

    def bands_along_standard_path(
        self,
        *,
        valley: int | None = None,
        central_band_count: int | None = None,
        points_per_segment: int = 120,
        return_eigenvectors: bool = False,
    ) -> PathBandsResult:
        return self.bands_along_path(
            self.standard_path(points_per_segment=points_per_segment),
            valley=valley,
            central_band_count=central_band_count,
            return_eigenvectors=return_eigenvectors,
        )

    def bands_on_grid(
        self,
        mesh_size: int,
        *,
        valley: int | None = None,
        band_indices: tuple[int, ...] | None = None,
        central_band_count: int | None = None,
        return_eigenvectors: bool = False,
        endpoint: bool = False,
        frac_shift: tuple[float, float] = (0.0, 0.0),
    ) -> GridBandsResult:
        resolved_valley = self.valley if valley is None else int(valley)
        return compute_bands_on_grid(
            mesh_size,
            self.lattice,
            self.params,
            domain=self.domain,
            valley=resolved_valley,
            band_indices=band_indices,
            central_band_count=central_band_count,
            return_eigenvectors=return_eigenvectors,
            endpoint=endpoint,
            frac_shift=frac_shift,
        )

    def path_bands(self, *, points_per_segment: int = 120, **kwargs) -> PathBandsResult:
        return self.bands_along_standard_path(points_per_segment=points_per_segment, **kwargs)

    def grid_bands(self, mesh_size: int, **kwargs) -> GridBandsResult:
        return self.bands_on_grid(mesh_size, **kwargs)


    def fhs_state_on_grid(
        self,
        mesh_size: int,
        band_indices: int | tuple[int, ...],
        *,
        valley: int | None = None,
        endpoint: bool = False,
        central_band_count: int | None = None,
        frac_shift: tuple[float, float] | None = None,
        use_boundary_sewing: bool = True,
    ) -> FHSState:
        resolved_valley = self.valley if valley is None else int(valley)
        return fhs_state_on_grid(
            mesh_size,
            self.lattice,
            self.params,
            band_indices,
            domain=self.domain,
            valley=resolved_valley,
            endpoint=endpoint,
            central_band_count=central_band_count,
            frac_shift=frac_shift,
            use_boundary_sewing=use_boundary_sewing,
        )

__all__ = ["HTQGModel"]
