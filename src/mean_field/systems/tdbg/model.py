from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.lattice import KPath
from .bands import GridBandsResult, PathBandsResult, compute_bands_along_path, compute_bands_on_grid
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .lattice import TDBGLattice, build_kpath_from_nodes, build_standard_kpath, build_tdbg_lattice
from .params import TDBGParameters
from .topology import TopologyResult, compute_topology_on_grid


@dataclass(frozen=True)
class TDBGModel:
    lattice: TDBGLattice
    params: TDBGParameters

    @classmethod
    def from_config(
        cls,
        theta_deg: float,
        *,
        cut: float = 4.0,
        params: TDBGParameters | None = None,
    ) -> "TDBGModel":
        resolved_params = params if params is not None else TDBGParameters.full()
        lattice = build_tdbg_lattice(
            theta_deg,
            phi_deg=resolved_params.phi_deg,
            epsilon=resolved_params.epsilon,
            cut=cut,
            graphene_lattice_constant_nm=resolved_params.graphene_lattice_constant_nm,
            beta=resolved_params.beta,
            poisson_ratio=resolved_params.poisson_ratio,
        )
        return cls(lattice=lattice, params=resolved_params)

    @property
    def theta_deg(self) -> float:
        return float(self.lattice.theta_deg)

    @property
    def cut(self) -> float:
        return float(self.lattice.cut)

    @property
    def matrix_dim(self) -> int:
        return int(self.lattice.matrix_dim)

    def lattice_summary(self) -> dict[str, object]:
        summary = self.lattice.to_summary_dict()
        summary.update(
            {
                "stacking": self.params.stacking,
                "valley": int(self.params.valley),
                "Delta_ev": float(self.params.Delta),
                "model_name": self.params.model_name,
            }
        )
        return summary

    def build_hamiltonian(self, k_tilde: complex, *, valley: int | None = None) -> np.ndarray:
        resolved_valley = self.params.valley if valley is None else int(valley)
        return build_hamiltonian(k_tilde, self.lattice, self.params, valley=resolved_valley)

    def diagonalize(
        self,
        k_tilde: complex,
        *,
        valley: int | None = None,
        n_bands: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        resolved_valley = self.params.valley if valley is None else int(valley)
        return diagonalize_hamiltonian(k_tilde, self.lattice, self.params, valley=resolved_valley, n_bands=n_bands)

    def build_kpath(
        self,
        nodes: tuple[complex, ...],
        labels: tuple[str, ...],
        segment_point_counts: tuple[int, ...],
        *,
        duplicate_nodes: bool = False,
    ) -> KPath:
        return build_kpath_from_nodes(nodes, labels, segment_point_counts, duplicate_nodes=duplicate_nodes)

    def standard_kpath(self, *, resolution: int = 16) -> KPath:
        return build_standard_kpath(self.lattice, resolution=resolution)

    def bands_along_path(
        self,
        path: KPath,
        *,
        valley: int | None = None,
        n_bands: int | None = None,
        return_eigenvectors: bool = False,
    ) -> PathBandsResult:
        resolved_valley = self.params.valley if valley is None else int(valley)
        return compute_bands_along_path(
            path,
            self.lattice,
            self.params,
            valley=resolved_valley,
            n_bands=n_bands,
            return_eigenvectors=return_eigenvectors,
        )

    def bands_along_standard_path(
        self,
        *,
        valley: int | None = None,
        n_bands: int | None = None,
        resolution: int = 16,
        return_eigenvectors: bool = False,
    ) -> PathBandsResult:
        return self.bands_along_path(
            self.standard_kpath(resolution=resolution),
            valley=valley,
            n_bands=n_bands,
            return_eigenvectors=return_eigenvectors,
        )

    def bands_on_grid(
        self,
        mesh_size: int,
        *,
        valley: int | None = None,
        n_bands: int | None = None,
        return_eigenvectors: bool = False,
        endpoint: bool = False,
        frac_shift: tuple[float, float] = (0.0, 0.0),
    ) -> GridBandsResult:
        resolved_valley = self.params.valley if valley is None else int(valley)
        return compute_bands_on_grid(
            mesh_size,
            self.lattice,
            self.params,
            valley=resolved_valley,
            n_bands=n_bands,
            return_eigenvectors=return_eigenvectors,
            endpoint=endpoint,
            frac_shift=frac_shift,
        )

    def topology_on_grid(
        self,
        mesh_size: int,
        band_indices: int | tuple[int, ...],
        *,
        valley: int | None = None,
        endpoint: bool = False,
        n_bands: int | None = None,
        boundary_sewing: bool = True,
    ) -> TopologyResult:
        resolved_valley = self.params.valley if valley is None else int(valley)
        return compute_topology_on_grid(
            mesh_size,
            self.lattice,
            self.params,
            band_indices,
            valley=resolved_valley,
            endpoint=endpoint,
            n_bands=n_bands,
            boundary_sewing=boundary_sewing,
        )
