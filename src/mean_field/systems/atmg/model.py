from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.hf import ComponentGroup
from ...core.lattice import KPath
from .bands import GridBandsResult, PathBandsResult, compute_bands_along_path, compute_bands_on_grid
from .bilayer_map import MappedSpectrumResult, build_atmg_via_tbg_sum
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .lattice import ATMGLattice, build_atmg_lattice, build_kpath_from_nodes, build_standard_kpath
from .params import ATMGParameters


@dataclass(frozen=True)
class ATMGModel:
    lattice: ATMGLattice
    params: ATMGParameters

    @classmethod
    def from_config(
        cls,
        n_layers: int,
        theta_deg: float,
        *,
        n_shells: int = 5,
        params: ATMGParameters | None = None,
    ) -> "ATMGModel":
        resolved_params = params if params is not None else ATMGParameters.realistic(n_layers, theta_deg)
        lattice = build_atmg_lattice(
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
        return 2 * int(self.params.n_layers) * int(self.lattice.n_g)

    def lattice_summary(self) -> dict[str, object]:
        summary = self.lattice.to_summary_dict()
        summary.update(
            {
                "n_layers": int(self.params.n_layers),
                "alpha": float(self.params.alpha),
                "alpha_couplings": list(float(value) for value in self.params.resolved_alpha_couplings),
                "matrix_dim": int(self.matrix_dim),
            }
        )
        return summary

    def component_groups(self) -> tuple[ComponentGroup, ...]:
        """Return layer groups in the local sublattice-resolved ATMG basis."""

        return tuple(
            ComponentGroup(f"layer_{layer}", np.asarray([2 * layer, 2 * layer + 1], dtype=int))
            for layer in range(int(self.params.n_layers))
        )

    def build_hamiltonian(self, k_tilde: complex, *, valley: int = 1) -> np.ndarray:
        return build_hamiltonian(complex(k_tilde), self.lattice, self.params, valley=valley)

    def diagonalize(
        self,
        k_tilde: complex,
        *,
        valley: int = 1,
        n_bands: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return diagonalize_hamiltonian(
            complex(k_tilde),
            self.lattice,
            self.params,
            valley=valley,
            n_bands=n_bands,
        )

    def mapped_spectrum(self, k_tilde: complex, *, valley: int = 1) -> MappedSpectrumResult:
        return build_atmg_via_tbg_sum(complex(k_tilde), self.lattice, self.params, valley=valley)

    def build_kpath(self, nodes: tuple[complex, ...], labels: tuple[str, ...], *, points_per_segment: int) -> KPath:
        return build_kpath_from_nodes(nodes, labels, points_per_segment)

    def standard_kpath(self, *, points_per_segment: int = 120) -> KPath:
        return build_standard_kpath(self.lattice, points_per_segment=points_per_segment)

    def bands_along_path(
        self,
        path: KPath,
        *,
        valley: int = 1,
        n_bands: int | None = None,
        return_eigenvectors: bool = False,
    ) -> PathBandsResult:
        return compute_bands_along_path(
            path,
            self.lattice,
            self.params,
            valley=valley,
            n_bands=n_bands,
            return_eigenvectors=return_eigenvectors,
        )

    def bands_along_standard_path(
        self,
        *,
        valley: int = 1,
        n_bands: int | None = None,
        points_per_segment: int = 120,
        return_eigenvectors: bool = False,
    ) -> PathBandsResult:
        return self.bands_along_path(
            self.standard_kpath(points_per_segment=points_per_segment),
            valley=valley,
            n_bands=n_bands,
            return_eigenvectors=return_eigenvectors,
        )

    def bands_on_grid(
        self,
        mesh_size: int,
        *,
        valley: int = 1,
        n_bands: int | None = None,
        return_eigenvectors: bool = False,
        endpoint: bool = False,
        frac_shift: tuple[float, float] = (0.0, 0.0),
    ) -> GridBandsResult:
        return compute_bands_on_grid(
            mesh_size,
            self.lattice,
            self.params,
            valley=valley,
            n_bands=n_bands,
            return_eigenvectors=return_eigenvectors,
            endpoint=endpoint,
            frac_shift=frac_shift,
        )

    def topology_on_grid(self, mesh_size: int, band_indices, **kwargs):
        from .topology import compute_topology_on_grid
        return compute_topology_on_grid(mesh_size, self.lattice, self.params, band_indices, **kwargs)


__all__ = ["ATMGModel"]
