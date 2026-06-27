from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.hf import ComponentGroup
from .core_lattice import KPath
from .bands import GridBandsResult, PathBandsResult, compute_bands_along_path, compute_bands_on_grid
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .lattice import TMBGLattice, build_kpath_from_nodes, build_standard_kpath, build_park_fig2_kpath, build_tmbg_lattice
from .params import TMBGParameters
from .topology import FHSState, fhs_state_on_grid


@dataclass(frozen=True)
class TMBGModel:
    lattice: TMBGLattice
    params: TMBGParameters

    @classmethod
    def from_config(
        cls,
        theta_deg: float,
        *,
        n_shells: int = 5,
        params: TMBGParameters | None = None,
    ) -> "TMBGModel":
        resolved_params = params if params is not None else TMBGParameters.full()
        lattice = build_tmbg_lattice(
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

    def lattice_summary(self) -> dict[str, object]:
        return self.lattice.to_summary_dict()

    def component_groups(self) -> tuple[ComponentGroup, ...]:
        """Return layer groups for the local six-orbital TMBG block."""

        return (
            ComponentGroup("layer_bottom", np.asarray([0, 1], dtype=int)),
            ComponentGroup("layer_middle", np.asarray([2, 3], dtype=int)),
            ComponentGroup("layer_top", np.asarray([4, 5], dtype=int)),
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

    def build_kpath(self, nodes: tuple[complex, ...], labels: tuple[str, ...], *, points_per_segment: int) -> KPath:
        return build_kpath_from_nodes(nodes, labels, points_per_segment)

    def standard_kpath(self, *, points_per_segment: int = 120) -> KPath:
        return build_standard_kpath(self.lattice, points_per_segment=points_per_segment)


    def park_fig2_kpath(
        self,
        *,
        points_per_segment: int = 120,
        gamma_prime_choice: str = "minus_g1",
    ) -> KPath:
        return build_park_fig2_kpath(
            self.lattice,
            points_per_segment=points_per_segment,
            gamma_prime_choice=gamma_prime_choice,
        )

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

    def fhs_state_on_grid(
        self,
        mesh_size: int,
        band_indices: int | tuple[int, ...],
        *,
        valley: int = 1,
        endpoint: bool = False,
        n_bands: int | None = None,
        frac_shift: tuple[float, float] = (0.0, 0.0),
        use_boundary_sewing: bool = True,
    ) -> FHSState:
        return fhs_state_on_grid(
            mesh_size,
            self.lattice,
            self.params,
            band_indices,
            valley=valley,
            endpoint=endpoint,
            n_bands=n_bands,
            frac_shift=frac_shift,
            use_boundary_sewing=use_boundary_sewing,
        )
