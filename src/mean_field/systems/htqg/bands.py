from __future__ import annotations

from ...core.bands import (
    GridBandsResult,
    PathBandsResult,
    compute_grid_bands,
    compute_path_bands,
    estimate_central_pair_metrics,
)
from .domains import HTQGDomain
from .hamiltonian import build_coupling_table, centered_band_indices, diagonalize_hamiltonian
from .lattice import HTQGLattice, KPath, build_moire_k_grid
from .params import HTQGParams


def _resolve_band_indices(
    lattice: HTQGLattice,
    *,
    band_indices: tuple[int, ...] | None,
    central_band_count: int | None,
) -> tuple[int, ...]:
    if band_indices is not None and central_band_count is not None:
        raise ValueError("Pass either band_indices or central_band_count, not both.")
    if band_indices is not None:
        return tuple(int(index) for index in band_indices)
    if central_band_count is not None:
        return centered_band_indices(lattice.matrix_dim, int(central_band_count))
    return tuple(range(lattice.matrix_dim))


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
    resolved_indices = _resolve_band_indices(
        lattice,
        band_indices=band_indices,
        central_band_count=central_band_count,
    )
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

    return compute_path_bands(
        path,
        matrix_dim=lattice.matrix_dim,
        n_bands=len(resolved_indices),
        return_eigenvectors=return_eigenvectors,
        diagonalize=_diagonalize,
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
    resolved_indices = _resolve_band_indices(
        lattice,
        band_indices=band_indices,
        central_band_count=central_band_count,
    )
    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=endpoint, frac_shift=frac_shift)
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

    return compute_grid_bands(
        k_grid_frac=k_grid_frac,
        kvec=kvec,
        matrix_dim=lattice.matrix_dim,
        n_bands=len(resolved_indices),
        return_eigenvectors=return_eigenvectors,
        diagonalize=_diagonalize,
        result_band_indices=resolved_indices,
        result_metadata={"system": "htqg", "domain": str(domain), "valley": int(valley)},
    )


def estimate_central_band_metrics(result: PathBandsResult | GridBandsResult, matrix_dim: int) -> dict[str, float | None]:
    """Estimate central two-band bandwidths and remote gap from sampled bands.

    This is a sampling diagnostic, not a paper-level checkpoint by itself.  The
    paper metrics require a sufficiently dense path/grid and cutoff convergence.
    """

    metrics = estimate_central_pair_metrics(result, matrix_dim)
    return {
        "valence_bandwidth_ev": metrics["valence_bandwidth_ev"],
        "conduction_bandwidth_ev": metrics["conduction_bandwidth_ev"],
        "mean_flat_bandwidth_ev": metrics["mean_flat_bandwidth_ev"],
        "central_manifold_span_ev": metrics["central_manifold_span_ev"],
        "central_gap_ev": metrics["central_gap_ev"],
        "remote_gap_ev": metrics["remote_gap_ev"],
    }


__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "compute_bands_along_path",
    "compute_bands_on_grid",
    "estimate_central_band_metrics",
]
