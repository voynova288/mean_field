"""Public analytic InAs/GaSb BHZ model adapter.

The adapter adds no new Hamiltonian formula. It delegates every matrix to the
source-attested Xue/Zeng implementation and only supplies the common model and
band-container protocols.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mean_field.core.bands import (
    GridBandsResult,
    PathBandsResult,
    centered_band_indices,
    compute_grid_bands,
    compute_path_bands,
)
from mean_field.core.lattice import KPath, build_kpath_from_nodes

from .xue2018 import xue2018_standard_parameters
from .zeng2022 import Zeng2022Parameters, ZengSlabBasis, build_zeng2022_folded_h0


@dataclass(frozen=True)
class InAsGaSbBHZModel:
    """Typed folded-BHZ model in the Xue/Zeng ``Ry*`` and ``a_B*`` units.

    ``path_extent_ab_inv`` is an explicit numerical cutoff, not a Brillouin-zone
    boundary. It is required only for the convenience path/grid methods.
    """

    params: Zeng2022Parameters
    slab_indices: tuple[int, ...] = (0,)
    path_extent_ab_inv: float | None = None
    variant: str = "zeng2022_folded_bhz"

    def __post_init__(self) -> None:
        self.params.validate()
        basis = ZengSlabBasis(tuple(int(value) for value in self.slab_indices))
        object.__setattr__(self, "slab_indices", basis.slab_indices)
        if self.path_extent_ab_inv is not None:
            extent = float(self.path_extent_ab_inv)
            if not np.isfinite(extent) or extent <= 0.0:
                raise ValueError("path_extent_ab_inv must be finite and positive")
            object.__setattr__(self, "path_extent_ab_inv", extent)
        if self.variant not in {"xue2018_q0_bhz", "zeng2022_folded_bhz"}:
            raise ValueError(f"unsupported InAs/GaSb BHZ variant {self.variant!r}")

    @classmethod
    def xue2018(
        cls,
        *,
        eg_ry: float,
        hybridization_ab_ry: float,
        path_extent_ab_inv: float | None = None,
    ) -> "InAsGaSbBHZModel":
        return cls(
            params=xue2018_standard_parameters(
                eg_ry=eg_ry,
                hybridization_ab_ry=hybridization_ab_ry,
            ),
            slab_indices=(0,),
            path_extent_ab_inv=path_extent_ab_inv,
            variant="xue2018_q0_bhz",
        )

    @property
    def basis(self) -> ZengSlabBasis:
        return ZengSlabBasis(self.slab_indices)

    @property
    def matrix_dim(self) -> int:
        return self.basis.dimension

    def build_hamiltonian(self, k_tilde: complex, **kwargs: object) -> np.ndarray:
        kwargs.pop("valley", None)
        if kwargs:
            raise TypeError(f"unsupported InAs/GaSb Hamiltonian options: {sorted(kwargs)}")
        point = complex(k_tilde)
        if not np.isfinite(point.real) or not np.isfinite(point.imag):
            raise ValueError("k_tilde must be finite")
        return build_zeng2022_folded_h0(
            np.asarray([[point.real, point.imag]], dtype=np.float64),
            self.basis,
            self.params,
        )[:, :, 0]

    def diagonalize(
        self,
        k_tilde: complex,
        *,
        n_bands: int | None = None,
        return_eigenvectors: bool = True,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        matrix = self.build_hamiltonian(k_tilde, **kwargs)
        energies, vectors = np.linalg.eigh(matrix)
        count = self.matrix_dim if n_bands is None else int(n_bands)
        indices = centered_band_indices(self.matrix_dim, count)
        selected_energies = np.asarray(energies[list(indices)], dtype=float)
        if not return_eigenvectors:
            return selected_energies, None
        return selected_energies, np.asarray(vectors[:, list(indices)], dtype=np.complex128)

    def _require_extent(self) -> float:
        if self.path_extent_ab_inv is None:
            raise ValueError(
                "path_extent_ab_inv is required for convenience paths/grids; "
                "pass an explicit KPath to compute_bands otherwise"
            )
        return float(self.path_extent_ab_inv)

    def standard_kpath(self, *, points_per_segment: int = 120) -> KPath:
        extent = self._require_extent()
        return build_kpath_from_nodes(
            (0.0 + 0.0j, extent + 0.0j, extent + 1j * extent, 0.0 + 0.0j),
            ("Γ", "Xcut", "Mcut", "Γ"),
            int(points_per_segment),
        )

    def bands_along_path(
        self,
        path: KPath,
        *,
        valley: int | None = None,
        n_bands: int | None = None,
        return_eigenvectors: bool = False,
    ) -> PathBandsResult:
        del valley
        count = self.matrix_dim if n_bands is None else int(n_bands)
        indices = centered_band_indices(self.matrix_dim, count)

        def diagonalize(
            k_value: complex,
            selected_count: int,
            want_vectors: bool,
        ) -> tuple[np.ndarray, np.ndarray | None]:
            return self.diagonalize(
                k_value,
                n_bands=selected_count,
                return_eigenvectors=want_vectors,
            )

        return compute_path_bands(
            path,
            matrix_dim=self.matrix_dim,
            n_bands=count,
            return_eigenvectors=return_eigenvectors,
            diagonalize=diagonalize,
            result_band_indices=indices,
            result_metadata=self.lattice_summary(),
        )

    def bands_on_grid(
        self,
        mesh_size: int,
        *,
        valley: int | None = None,
        n_bands: int | None = None,
        return_eigenvectors: bool = False,
    ) -> GridBandsResult:
        del valley
        size = int(mesh_size)
        if size <= 0:
            raise ValueError("mesh_size must be positive")
        extent = self._require_extent()
        frac_axis = np.arange(size, dtype=float) / float(size)
        frac_x, frac_y = np.meshgrid(frac_axis, frac_axis, indexing="ij")
        kx = -extent + 2.0 * extent * frac_x
        ky = -extent + 2.0 * extent * frac_y
        kvec = np.asarray(kx + 1j * ky, dtype=np.complex128)
        frac = np.stack([frac_x, frac_y], axis=-1)
        count = self.matrix_dim if n_bands is None else int(n_bands)
        indices = centered_band_indices(self.matrix_dim, count)

        def diagonalize(
            k_value: complex,
            selected_count: int,
            want_vectors: bool,
        ) -> tuple[np.ndarray, np.ndarray | None]:
            return self.diagonalize(
                k_value,
                n_bands=selected_count,
                return_eigenvectors=want_vectors,
            )

        return compute_grid_bands(
            k_grid_frac=frac,
            kvec=kvec,
            matrix_dim=self.matrix_dim,
            n_bands=count,
            return_eigenvectors=return_eigenvectors,
            diagonalize=diagonalize,
            result_band_indices=indices,
            result_metadata=self.lattice_summary(),
        )

    def lattice_summary(self) -> dict[str, object]:
        return {
            "model": self.variant,
            "periodic_brillouin_zone": False,
            "momentum_unit": "a_B*^-1",
            "energy_unit": "Ry*",
            "slab_indices": list(self.slab_indices),
            "path_extent_ab_inv": self.path_extent_ab_inv,
        }

    def component_groups(self) -> tuple[object, ...]:
        return ()


__all__ = ["InAsGaSbBHZModel"]
