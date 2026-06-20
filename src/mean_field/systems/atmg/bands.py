from __future__ import annotations

from ...core.bands import GridBandsResult, PathBandsResult, compute_grid_bands, compute_path_bands
from ...core.lattice import KPath
from .hamiltonian import diagonalize_hamiltonian
from .lattice import ATMGLattice, build_moire_k_grid
from .params import ATMGParameters
from .tbg import build_coupling_table


def _make_diagonalizer(lattice: ATMGLattice, params: ATMGParameters, *, valley: int):
    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)
    def _diagonalize(kval: complex, resolved_band_count: int, want_eigenvectors: bool):
        del want_eigenvectors
        return diagonalize_hamiltonian(complex(kval), lattice, params, valley=valley, n_bands=int(resolved_band_count), coupling_table=coupling_table)
    return _diagonalize


def compute_bands_along_path(path: KPath, lattice: ATMGLattice, params: ATMGParameters, *, valley: int = 1, n_bands: int | None = None, return_eigenvectors: bool = False) -> PathBandsResult:
    return compute_path_bands(path, matrix_dim=2 * params.n_layers * lattice.n_g, n_bands=n_bands, return_eigenvectors=return_eigenvectors, diagonalize=_make_diagonalizer(lattice, params, valley=valley))


def compute_bands_on_grid(mesh_size: int, lattice: ATMGLattice, params: ATMGParameters, *, valley: int = 1, n_bands: int | None = None, return_eigenvectors: bool = False, endpoint: bool = False, frac_shift: tuple[float, float] = (0.0, 0.0)) -> GridBandsResult:
    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=endpoint, frac_shift=frac_shift)
    return compute_grid_bands(k_grid_frac=k_grid_frac, kvec=kvec, matrix_dim=2 * params.n_layers * lattice.n_g, n_bands=n_bands, return_eigenvectors=return_eigenvectors, diagonalize=_make_diagonalizer(lattice, params, valley=valley))


__all__ = ["GridBandsResult", "PathBandsResult", "compute_bands_along_path", "compute_bands_on_grid"]
