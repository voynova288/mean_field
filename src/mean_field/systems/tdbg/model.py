from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.lattice import KPath
from .bands import GridBandsResult, PathBandsResult, compute_bands_along_path, compute_bands_on_grid
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .lattice import TDBGLattice, build_kpath_from_nodes, build_standard_kpath, build_tdbg_lattice
from .params import TDBGParameters


@dataclass(frozen=True)
class TDBGBasisComponentGroup:
    """Named subset of the q-site-major TDBG Hamiltonian basis.

    The public model basis is the full continuum Hamiltonian basis with index
    ``4 * q_site + alpha``.  This is distinct from the embedded local-index
    basis used by projected-HF overlap helpers.
    """

    name: str
    indices: np.ndarray
    index_space: str = "tdbg_full_hamiltonian_basis"
    description: str = ""

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name:
            raise ValueError("TDBG component group name must be non-empty")
        indices = np.asarray(self.indices, dtype=int).reshape(-1)
        if indices.size == 0:
            raise ValueError(f"TDBG component group {name!r} must contain at least one basis index")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"TDBG component group {name!r} contains duplicate indices")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "index_space", str(self.index_space))
        object.__setattr__(self, "description", str(self.description))


def _tdbg_group_indices(lattice: TDBGLattice, *, sector: int | None = None, alphas: tuple[int, ...]) -> np.ndarray:
    indices: list[int] = []
    for iq, site in enumerate(np.asarray(lattice.q_sites, dtype=float)):
        site_sector = int(round(float(site[2])))
        if sector is not None and site_sector != int(sector):
            continue
        for alpha in alphas:
            indices.append(4 * int(iq) + int(alpha))
    return np.asarray(indices, dtype=int)


def tdbg_full_basis_component_groups(lattice: TDBGLattice) -> tuple[TDBGBasisComponentGroup, ...]:
    """Return q-site-major TDBG Hamiltonian-basis component groups.

    The basis index is ``4 * q_site + alpha`` with ``alpha=(A1, B1, A2, B2)``.
    Layer names are deliberately code-stable (`layer_0` ... `layer_3`), with
    descriptions carrying the sector/local-layer/potential convention instead
    of silently guessing a paper's top/bottom naming convention.
    """

    groups: list[TDBGBasisComponentGroup] = [
        TDBGBasisComponentGroup(
            "sector_0",
            _tdbg_group_indices(lattice, sector=0, alphas=(0, 1, 2, 3)),
            description="TDBG q-sites in sector 0; local alpha=(A1,B1,A2,B2).",
        ),
        TDBGBasisComponentGroup(
            "sector_1",
            _tdbg_group_indices(lattice, sector=1, alphas=(0, 1, 2, 3)),
            description="TDBG q-sites in sector 1; local alpha=(A1,B1,A2,B2).",
        ),
        TDBGBasisComponentGroup(
            "layer_0",
            _tdbg_group_indices(lattice, sector=0, alphas=(0, 1)),
            description="Sector 0, BLG-local upper layer (A1,B1), potential +3*Delta/2.",
        ),
        TDBGBasisComponentGroup(
            "layer_1",
            _tdbg_group_indices(lattice, sector=0, alphas=(2, 3)),
            description="Sector 0, BLG-local lower/interface layer (A2,B2), potential +Delta/2.",
        ),
        TDBGBasisComponentGroup(
            "layer_2",
            _tdbg_group_indices(lattice, sector=1, alphas=(0, 1)),
            description="Sector 1, BLG-local upper/interface layer (A1,B1), potential -Delta/2.",
        ),
        TDBGBasisComponentGroup(
            "layer_3",
            _tdbg_group_indices(lattice, sector=1, alphas=(2, 3)),
            description="Sector 1, BLG-local lower layer (A2,B2), potential -3*Delta/2.",
        ),
        TDBGBasisComponentGroup(
            "sublattice_A",
            _tdbg_group_indices(lattice, sector=None, alphas=(0, 2)),
            description="A-sublattice local components A1 and A2 across all q-sites and sectors.",
        ),
        TDBGBasisComponentGroup(
            "sublattice_B",
            _tdbg_group_indices(lattice, sector=None, alphas=(1, 3)),
            description="B-sublattice local components B1 and B2 across all q-sites and sectors.",
        ),
    ]
    return tuple(group for group in groups if group.indices.size)


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

    def component_groups(self) -> tuple[TDBGBasisComponentGroup, ...]:
        """Return sector/layer/sublattice groups in the full TDBG basis."""

        return tdbg_full_basis_component_groups(self.lattice)

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
