from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.hf import ComponentGroup
from ...core.lattice import KPath
from .bands import GridBandsResult, PathBandsResult, compute_bands_along_path, compute_bands_on_grid
from .charge_background import ChargeBackgroundResult, compute_valence_charge_background
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian, flat_band_indices, hamiltonian_dimension
from .lattice import RLGhBNLattice, build_kpath_from_nodes, build_rlg_hbn_lattice, build_standard_kpath
from .params import RLGhBNParams


@dataclass(frozen=True)
class RLGhBNModel:
    lattice: RLGhBNLattice
    params: RLGhBNParams

    @classmethod
    def from_config(
        cls,
        *,
        layer_count: int = 5,
        xi: int = 1,
        theta_deg: float = 0.77,
        displacement_field_mev: float = 0.0,
        shell_count: int = 4,
        params: RLGhBNParams | None = None,
    ) -> "RLGhBNModel":
        resolved_params = (
            params
            if params is not None
            else RLGhBNParams.from_table(
                layer_count=layer_count,
                xi=xi,
                displacement_field_mev=displacement_field_mev,
            )
        )
        lattice = build_rlg_hbn_lattice(
            theta_deg,
            shell_count=shell_count,
            hbn_lattice_mismatch=resolved_params.hbn_lattice_mismatch,
            graphene_lattice_constant_nm=resolved_params.graphene_lattice_constant_nm,
            layer_count=resolved_params.layer_count,
        )
        return cls(lattice=lattice, params=resolved_params)

    @property
    def matrix_dim(self) -> int:
        return hamiltonian_dimension(self.lattice, self.params)

    @property
    def layer_count(self) -> int:
        return int(self.params.layer_count)

    def component_groups(self) -> tuple[ComponentGroup, ...]:
        """Return layer groups in the local sublattice-resolved basis."""

        return tuple(
            ComponentGroup(f"layer_{layer}", np.asarray([2 * layer, 2 * layer + 1], dtype=int))
            for layer in range(self.layer_count)
        )

    @property
    def flat_band_indices(self) -> tuple[int, int]:
        return flat_band_indices(self.lattice, self.params)

    def lattice_summary(self) -> dict[str, object]:
        summary = self.lattice.to_summary_dict()
        summary.update(self.params.to_summary_dict())
        summary["valence_band_count"] = int(self.params.layer_count * self.lattice.n_g)
        summary["conduction_band_count"] = int(self.params.layer_count * self.lattice.n_g)
        return summary

    def build_hamiltonian(self, k_tilde: complex, *, valley: int = 1) -> np.ndarray:
        return build_hamiltonian(k_tilde, self.lattice, self.params, valley=valley)

    def diagonalize(
        self,
        k_tilde: complex,
        *,
        valley: int = 1,
        n_bands: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return diagonalize_hamiltonian(k_tilde, self.lattice, self.params, valley=valley, n_bands=n_bands)

    def build_kpath(
        self,
        nodes: tuple[complex, ...],
        labels: tuple[str, ...],
        segment_point_counts: tuple[int, ...],
        *,
        duplicate_nodes: bool = False,
    ) -> KPath:
        return build_kpath_from_nodes(nodes, labels, segment_point_counts, duplicate_nodes=duplicate_nodes)

    def standard_kpath(self, *, points_per_segment: int = 80) -> KPath:
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
        points_per_segment: int = 80,
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


    def fhs_state_on_grid(self, mesh_size: int, band_indices, **kwargs):
        from .topology import fhs_state_on_grid

        return fhs_state_on_grid(mesh_size, self.lattice, self.params, band_indices, **kwargs)

    def valence_charge_background(
        self,
        grid_result: GridBandsResult,
        *,
        real_space_mesh_size: int = 48,
        n_valence_bands: int | None = None,
    ) -> ChargeBackgroundResult:
        return compute_valence_charge_background(
            grid_result,
            self.lattice,
            self.params,
            real_space_mesh_size=real_space_mesh_size,
            n_valence_bands=n_valence_bands,
        )


__all__ = ["RLGhBNModel"]
