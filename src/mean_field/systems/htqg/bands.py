from __future__ import annotations

from ...core.bands import (
    GridBandsResult,
    PathBandsResult,
    compute_grid_bands,
    compute_path_bands,
    resolve_selected_band_indices,
)
from .domains import HTQGDomain
from .hamiltonian import build_coupling_table, diagonalize_hamiltonian
from .lattice import HTQGLattice, KPath, build_moire_k_grid
from .params import HTQGParams


def _prepare_band_diagonalizer(
    lattice: HTQGLattice,
    params: HTQGParams,
    *,
    domain: str | HTQGDomain,
    valley: int,
    d12: complex | None,
    d34: complex | None,
    band_indices: tuple[int, ...] | None,
    central_band_count: int | None,
):
    resolved_indices = resolve_selected_band_indices(lattice.matrix_dim, band_indices=band_indices, central_band_count=central_band_count)
    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors)

    def _diagonalize(kval: complex, resolved_n_bands: int, want_eigenvectors: bool):
        if resolved_n_bands != len(resolved_indices):
            raise ValueError("HTQG selected-band loop received inconsistent band count")
        return diagonalize_hamiltonian(
            complex(kval),
            lattice,
            params,
            domain=domain,
            valley=valley,
            d12=d12,
            d34=d34,
            coupling_table=coupling_table,
            band_indices=resolved_indices,
            return_eigenvectors=want_eigenvectors,
        )

    return resolved_indices, _diagonalize


def compute_bands_along_path(
    path: KPath,
    lattice: HTQGLattice,
    params: HTQGParams,
    *,
    domain: str | HTQGDomain = "alpha_beta_alpha",
    valley: int = 1,
    d12: complex | None = None,
    d34: complex | None = None,
    band_indices: tuple[int, ...] | None = None,
    central_band_count: int | None = None,
    return_eigenvectors: bool = False,
) -> PathBandsResult:
    resolved_indices, diagonalize = _prepare_band_diagonalizer(
        lattice, params, domain=domain, valley=valley, d12=d12, d34=d34,
        band_indices=band_indices, central_band_count=central_band_count,
    )
    return compute_path_bands(
        path,
        matrix_dim=lattice.matrix_dim,
        n_bands=len(resolved_indices),
        return_eigenvectors=return_eigenvectors,
        diagonalize=diagonalize,
        result_band_indices=resolved_indices,
        result_metadata={"system": "htqg", "domain": str(domain), "valley": int(valley)},
    )


def compute_bands_on_grid(
    mesh_size: int,
    lattice: HTQGLattice,
    params: HTQGParams,
    *,
    domain: str | HTQGDomain = "alpha_beta_alpha",
    valley: int = 1,
    d12: complex | None = None,
    d34: complex | None = None,
    band_indices: tuple[int, ...] | None = None,
    central_band_count: int | None = None,
    return_eigenvectors: bool = False,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> GridBandsResult:
    resolved_indices, diagonalize = _prepare_band_diagonalizer(
        lattice, params, domain=domain, valley=valley, d12=d12, d34=d34,
        band_indices=band_indices, central_band_count=central_band_count,
    )
    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=endpoint, frac_shift=frac_shift)
    return compute_grid_bands(
        k_grid_frac=k_grid_frac,
        kvec=kvec,
        matrix_dim=lattice.matrix_dim,
        n_bands=len(resolved_indices),
        return_eigenvectors=return_eigenvectors,
        diagonalize=diagonalize,
        result_band_indices=resolved_indices,
        result_metadata={"system": "htqg", "domain": str(domain), "valley": int(valley)},
    )



__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "compute_bands_along_path",
    "compute_bands_on_grid",
]
