from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.lattice import KPath
from .bands import AhnTMDGridBands, AhnTMDPathBands, compute_bands_along_path, compute_bands_on_rectangular_grid
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .lattice import AhnTMDLattice, build_ahn_tmd_lattice, standard_kpath
from .params import AhnTMDParameters, DEFAULT_AHN_TMD_PARAMETERS


@dataclass(frozen=True)
class AhnTMDModel:
    lattice: AhnTMDLattice
    params: AhnTMDParameters

    @classmethod
    def from_config(
        cls,
        theta_deg: float = 2.1,
        *,
        n_shell: int = 5,
        params: AhnTMDParameters = DEFAULT_AHN_TMD_PARAMETERS,
    ) -> "AhnTMDModel":
        return cls(
            lattice=build_ahn_tmd_lattice(theta_deg, n_shell=n_shell, params=params),
            params=params,
        )

    @property
    def theta_deg(self) -> float:
        return float(self.lattice.theta_deg)

    @property
    def n_shell(self) -> int:
        return int(self.lattice.n_shell)

    @property
    def matrix_dim(self) -> int:
        return int(self.lattice.matrix_dim)

    def build_hamiltonian(self, k_tilde: complex, *, valley: int = 1) -> np.ndarray:
        return build_hamiltonian(complex(k_tilde), self.lattice, self.params, valley=valley)

    def diagonalize(
        self,
        k_tilde: complex,
        *,
        valley: int = 1,
        descending: bool = True,
        n_bands: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return diagonalize_hamiltonian(
            complex(k_tilde),
            self.lattice,
            self.params,
            valley=valley,
            descending=descending,
            n_bands=n_bands,
        )

    def standard_kpath(self, *, points_per_segment: int = 120) -> KPath:
        return standard_kpath(self.lattice, points_per_segment=points_per_segment)

    def bands_along_path(
        self,
        path: KPath,
        *,
        valley: int = 1,
        n_bands: int | None = None,
        return_eigenvectors: bool = False,
    ) -> AhnTMDPathBands:
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
    ) -> AhnTMDPathBands:
        return self.bands_along_path(
            self.standard_kpath(points_per_segment=points_per_segment),
            valley=valley,
            n_bands=n_bands,
            return_eigenvectors=return_eigenvectors,
        )

    def bands_on_rectangular_grid(
        self,
        nx: int,
        ny: int,
        *,
        valley: int = 1,
        n_bands: int | None = None,
        return_eigenvectors: bool = False,
        frac_shift: tuple[float, float] = (0.0, 0.0),
    ) -> AhnTMDGridBands:
        return compute_bands_on_rectangular_grid(
            int(nx),
            int(ny),
            self.lattice,
            self.params,
            valley=valley,
            n_bands=n_bands,
            return_eigenvectors=return_eigenvectors,
            frac_shift=frac_shift,
        )

    def summary(self) -> dict[str, object]:
        return {
            "system": "Ahn2024_tMoTe2",
            "theta_deg": float(self.theta_deg),
            "n_shell": int(self.n_shell),
            "num_g": int(self.lattice.n_g),
            "matrix_dim": int(self.matrix_dim),
            "moire_area_nm2": float(self.lattice.moire_area_nm2),
            "q0_nm_inv": [float(self.lattice.q0.real), float(self.lattice.q0.imag)],
            "q1_nm_inv": [float(self.lattice.q1.real), float(self.lattice.q1.imag)],
        }


__all__ = ["AhnTMDModel"]
