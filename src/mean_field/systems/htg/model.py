from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bands import GridBandsResult, PathBandsResult, compute_bands_along_path, compute_bands_on_grid
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .lattice import HTGLattice, KPath, build_htg_lattice, build_kpath_from_nodes, build_paper_hf_kpath, build_standard_kpath
from .params import HTGParams
from .topology import ChernBasisResult, compute_chern_basis_on_grid


@dataclass(frozen=True)
class HTGModel:
    lattice: HTGLattice
    params: HTGParams

    @classmethod
    def from_config(
        cls,
        theta_deg: float,
        *,
        n_shells: int = 5,
        params: HTGParams | None = None,
    ) -> "HTGModel":
        resolved_params = params if params is not None else HTGParams.default()
        lattice = build_htg_lattice(
            theta_deg,
            n_shells=n_shells,
            graphene_lattice_constant_nm=resolved_params.graphene_lattice_constant_nm,
        )
        return cls(lattice=lattice, params=resolved_params)

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
                "kappa": float(self.params.kappa),
                "w_ev": float(self.params.w_ev),
                "vf_ev_nm": float(self.params.vf_ev_nm),
                "alpha": float(self.params.alpha(self.lattice.k_theta)),
                "model_name": self.params.model_name,
            }
        )
        return summary

    def build_hamiltonian(
        self,
        k_tilde: complex,
        *,
        valley: int = 1,
        d_top: complex | None = None,
        d_bot: complex | None = None,
    ) -> np.ndarray:
        return build_hamiltonian(
            k_tilde,
            self.lattice,
            self.params,
            valley=valley,
            d_top=d_top,
            d_bot=d_bot,
        )

    def diagonalize(
        self,
        k_tilde: complex,
        *,
        valley: int = 1,
        d_top: complex | None = None,
        d_bot: complex | None = None,
        band_indices: tuple[int, ...] | None = None,
        return_eigenvectors: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return diagonalize_hamiltonian(
            k_tilde,
            self.lattice,
            self.params,
            valley=valley,
            d_top=d_top,
            d_bot=d_bot,
            band_indices=band_indices,
            return_eigenvectors=return_eigenvectors,
        )

    def build_kpath(
        self,
        nodes: tuple[complex, ...],
        labels: tuple[str, ...],
        *,
        points_per_segment: int,
    ) -> KPath:
        return build_kpath_from_nodes(nodes, labels, points_per_segment)

    def standard_kpath(self, *, points_per_segment: int = 120) -> KPath:
        return build_standard_kpath(self.lattice, points_per_segment=points_per_segment)

    def paper_hf_kpath(self, *, points_per_segment: int = 120) -> KPath:
        return build_paper_hf_kpath(self.lattice, points_per_segment=points_per_segment)

    def bands_along_path(
        self,
        path: KPath,
        *,
        valley: int = 1,
        d_top: complex | None = None,
        d_bot: complex | None = None,
        band_indices: tuple[int, ...] | None = None,
        central_band_count: int | None = None,
        return_eigenvectors: bool = False,
    ) -> PathBandsResult:
        return compute_bands_along_path(
            path,
            self.lattice,
            self.params,
            valley=valley,
            d_top=d_top,
            d_bot=d_bot,
            band_indices=band_indices,
            central_band_count=central_band_count,
            return_eigenvectors=return_eigenvectors,
        )

    def bands_along_standard_path(
        self,
        *,
        valley: int = 1,
        central_band_count: int | None = None,
        points_per_segment: int = 120,
        return_eigenvectors: bool = False,
    ) -> PathBandsResult:
        return self.bands_along_path(
            self.standard_kpath(points_per_segment=points_per_segment),
            valley=valley,
            central_band_count=central_band_count,
            return_eigenvectors=return_eigenvectors,
        )

    def bands_on_grid(
        self,
        mesh_size: int,
        *,
        valley: int = 1,
        band_indices: tuple[int, ...] | None = None,
        central_band_count: int | None = None,
        return_eigenvectors: bool = False,
        endpoint: bool = False,
        frac_shift: tuple[float, float] = (0.0, 0.0),
    ) -> GridBandsResult:
        return compute_bands_on_grid(
            mesh_size,
            self.lattice,
            self.params,
            valley=valley,
            band_indices=band_indices,
            central_band_count=central_band_count,
            return_eigenvectors=return_eigenvectors,
            endpoint=endpoint,
            frac_shift=frac_shift,
        )

    def chern_basis_on_grid(self, mesh_size: int, *, valley: int = 1) -> ChernBasisResult:
        return compute_chern_basis_on_grid(mesh_size, self.lattice, self.params, valley=valley)
