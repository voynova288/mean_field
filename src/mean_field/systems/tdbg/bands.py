from __future__ import annotations


from ...core.bands import GridBandsResult, PathBandsResult, compute_grid_bands, compute_path_bands
from ...core.lattice import KPath
from .hamiltonian import diagonalize_hamiltonian
from .lattice import TDBGLattice, build_moire_k_grid
from .params import TDBGParameters


def compute_bands_along_path(
    path: KPath,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
) -> PathBandsResult:
    def _diagonalize(kval: complex, resolved_n_bands: int, want_eigenvectors: bool):
        del want_eigenvectors
        return diagonalize_hamiltonian(kval, lattice, params, valley=valley, n_bands=resolved_n_bands)

    return compute_path_bands(
        path,
        matrix_dim=lattice.matrix_dim,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        diagonalize=_diagonalize,
    )


def compute_bands_on_grid(
    mesh_size: int,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> GridBandsResult:
    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=endpoint, frac_shift=frac_shift)

    def _diagonalize(kval: complex, resolved_n_bands: int, want_eigenvectors: bool):
        del want_eigenvectors
        return diagonalize_hamiltonian(kval, lattice, params, valley=valley, n_bands=resolved_n_bands)

    return compute_grid_bands(
        k_grid_frac=k_grid_frac,
        kvec=kvec,
        matrix_dim=lattice.matrix_dim,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        diagonalize=_diagonalize,
    )
