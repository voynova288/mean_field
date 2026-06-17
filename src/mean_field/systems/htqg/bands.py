from __future__ import annotations

import numpy as np

from ...core.bands import GridBandsResult, PathBandsResult, compute_grid_bands, compute_path_bands
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

    band_indices = tuple(int(index) for index in result.band_indices)
    positions = {band_index: pos for pos, band_index in enumerate(band_indices)}
    valence = int(matrix_dim) // 2 - 1
    conduction = int(matrix_dim) // 2
    lower_remote = valence - 1
    upper_remote = conduction + 1
    if valence not in positions or conduction not in positions:
        return {
            "valence_bandwidth_ev": None,
            "conduction_bandwidth_ev": None,
            "mean_flat_bandwidth_ev": None,
            "central_manifold_span_ev": None,
            "remote_gap_ev": None,
        }

    energies = np.asarray(result.energies, dtype=float)
    val = energies[..., positions[valence]]
    con = energies[..., positions[conduction]]
    val_bw = float(np.max(val) - np.min(val))
    con_bw = float(np.max(con) - np.min(con))
    central_gap = float(np.min(con - val))
    central = energies[..., [positions[valence], positions[conduction]]]
    span = float(np.max(central) - np.min(central))
    remote_gap: float | None = None
    if lower_remote in positions and upper_remote in positions:
        lower_gap = val - energies[..., positions[lower_remote]]
        upper_gap = energies[..., positions[upper_remote]] - con
        remote_gap = float(min(np.min(lower_gap), np.min(upper_gap)))
    return {
        "valence_bandwidth_ev": val_bw,
        "conduction_bandwidth_ev": con_bw,
        "mean_flat_bandwidth_ev": 0.5 * (val_bw + con_bw),
        "central_manifold_span_ev": span,
        "central_gap_ev": central_gap,
        "remote_gap_ev": remote_gap,
    }


__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "compute_bands_along_path",
    "compute_bands_on_grid",
    "estimate_central_band_metrics",
]
