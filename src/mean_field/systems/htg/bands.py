from __future__ import annotations

from ...core.bands import GridBandsResult, PathBandsResult, compute_grid_bands, compute_path_bands, resolve_selected_band_indices
from .hamiltonian import build_coupling_table, diagonalize_hamiltonian
from .lattice import HTGLattice, KPath, build_moire_k_grid
from .params import HTGParams


def _prepare_band_diagonalizer(lattice: HTGLattice, params: HTGParams, *, valley: int, d_top: complex | None, d_bot: complex | None, band_indices: tuple[int, ...] | None, central_band_count: int | None):
    resolved = resolve_selected_band_indices(lattice.matrix_dim, band_indices=band_indices, central_band_count=central_band_count)
    top_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=1)
    bottom_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=-1)
    def _diagonalize(kval: complex, resolved_n_bands: int, want_eigenvectors: bool):
        if resolved_n_bands != len(resolved):
            raise ValueError("HTG selected-band loop received inconsistent band count")
        return diagonalize_hamiltonian(complex(kval), lattice, params, valley=valley, d_top=d_top, d_bot=d_bot, top_coupling_table=top_table, bottom_coupling_table=bottom_table, band_indices=resolved, return_eigenvectors=want_eigenvectors)
    return resolved, _diagonalize


def compute_bands_along_path(path: KPath, lattice: HTGLattice, params: HTGParams, *, valley: int = 1, d_top: complex | None = None, d_bot: complex | None = None, band_indices: tuple[int, ...] | None = None, central_band_count: int | None = None, return_eigenvectors: bool = False) -> PathBandsResult:
    resolved, diagonalize = _prepare_band_diagonalizer(lattice, params, valley=valley, d_top=d_top, d_bot=d_bot, band_indices=band_indices, central_band_count=central_band_count)
    return compute_path_bands(path, matrix_dim=lattice.matrix_dim, n_bands=len(resolved), return_eigenvectors=return_eigenvectors, diagonalize=diagonalize, result_band_indices=resolved)


def compute_bands_on_grid(mesh_size: int, lattice: HTGLattice, params: HTGParams, *, valley: int = 1, d_top: complex | None = None, d_bot: complex | None = None, band_indices: tuple[int, ...] | None = None, central_band_count: int | None = None, return_eigenvectors: bool = False, endpoint: bool = False, frac_shift: tuple[float, float] = (0.0, 0.0)) -> GridBandsResult:
    resolved, diagonalize = _prepare_band_diagonalizer(lattice, params, valley=valley, d_top=d_top, d_bot=d_bot, band_indices=band_indices, central_band_count=central_band_count)
    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=endpoint, frac_shift=frac_shift)
    return compute_grid_bands(k_grid_frac=k_grid_frac, kvec=kvec, matrix_dim=lattice.matrix_dim, n_bands=len(resolved), return_eigenvectors=return_eigenvectors, diagonalize=diagonalize, result_band_indices=resolved)
