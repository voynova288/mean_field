from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .lattice import KPath

DiagonalizeCallback = Callable[[complex, int, bool], tuple[np.ndarray, np.ndarray | None]]


@dataclass(frozen=True)
class PathBandsResult:
    """Band energies/eigenvectors sampled along a k-path.

    The optional fields cover the historical system-specific extensions used by
    HTG (``band_indices``) and ATMG (mapped/subspace spectra) while keeping the
    common path/energy/eigenvector shape shared by most systems.
    """

    path: KPath
    energies: np.ndarray
    eigenvectors: np.ndarray | None = None
    band_indices: tuple[int, ...] = ()
    mapped_energies: np.ndarray | None = None
    subspace_labels: tuple[str, ...] = ()
    subspace_energies: tuple[np.ndarray, ...] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GridBandsResult:
    """Band energies/eigenvectors sampled on a 2D reciprocal grid."""

    k_grid_frac: np.ndarray
    kvec: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray | None = None
    band_indices: tuple[int, ...] = ()
    mapped_energies: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def resolve_n_bands(matrix_dim: int, n_bands: int | None) -> int:
    resolved = int(matrix_dim) if n_bands is None else int(n_bands)
    if resolved <= 0:
        raise ValueError(f"n_bands must be positive, got {resolved}")
    if resolved > int(matrix_dim):
        raise ValueError(f"n_bands={resolved} exceeds matrix_dim={int(matrix_dim)}")
    return resolved


def compute_path_bands(
    path: KPath,
    *,
    matrix_dim: int,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
    diagonalize: DiagonalizeCallback,
    result_band_indices: Iterable[int] = (),
    result_metadata: dict[str, object] | None = None,
) -> PathBandsResult:
    """Generic path-band loop for systems that supply a diagonalizer callback.

    ``diagonalize(k, n_bands, return_eigenvectors)`` must return ``(evals,
    evecs_or_none)`` with eigenvectors shaped ``(matrix_dim, n_bands)`` when
    requested.  System modules remain responsible for building/reusing any
    coupling tables or model-specific closures before calling this helper.
    """

    resolved_n_bands = resolve_n_bands(int(matrix_dim), n_bands)
    energies = np.zeros((path.kvec.size, resolved_n_bands), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros((path.kvec.size, int(matrix_dim), resolved_n_bands), dtype=np.complex128)
    for ik, kval in enumerate(path.kvec):
        evals, evecs = diagonalize(complex(kval), resolved_n_bands, bool(return_eigenvectors))
        energies[ik, :] = np.asarray(evals, dtype=float)
        if return_eigenvectors:
            if evecs is None:
                raise ValueError("diagonalize callback returned no eigenvectors despite return_eigenvectors=True")
            assert eigenvectors is not None
            eigenvectors[ik, :, :] = np.asarray(evecs, dtype=np.complex128)
    return PathBandsResult(
        path=path,
        energies=energies,
        eigenvectors=eigenvectors,
        band_indices=tuple(int(index) for index in result_band_indices),
        metadata={} if result_metadata is None else dict(result_metadata),
    )


def compute_grid_bands(
    *,
    k_grid_frac: np.ndarray,
    kvec: np.ndarray,
    matrix_dim: int,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
    diagonalize: DiagonalizeCallback,
    result_band_indices: Iterable[int] = (),
    result_metadata: dict[str, object] | None = None,
) -> GridBandsResult:
    """Generic 2D-grid band loop for systems that supply a diagonalizer callback."""

    kvec_array = np.asarray(kvec, dtype=np.complex128)
    if kvec_array.ndim != 2:
        raise ValueError(f"Expected 2D kvec grid, got shape {kvec_array.shape}")
    resolved_n_bands = resolve_n_bands(int(matrix_dim), n_bands)
    energies = np.zeros(kvec_array.shape + (resolved_n_bands,), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros(kvec_array.shape + (int(matrix_dim), resolved_n_bands), dtype=np.complex128)
    for index in np.ndindex(kvec_array.shape):
        evals, evecs = diagonalize(complex(kvec_array[index]), resolved_n_bands, bool(return_eigenvectors))
        energies[index + (slice(None),)] = np.asarray(evals, dtype=float)
        if return_eigenvectors:
            if evecs is None:
                raise ValueError("diagonalize callback returned no eigenvectors despite return_eigenvectors=True")
            assert eigenvectors is not None
            eigenvectors[index + (slice(None), slice(None))] = np.asarray(evecs, dtype=np.complex128)
    return GridBandsResult(
        k_grid_frac=np.asarray(k_grid_frac, dtype=float),
        kvec=kvec_array,
        energies=energies,
        eigenvectors=eigenvectors,
        band_indices=tuple(int(index) for index in result_band_indices),
        metadata={} if result_metadata is None else dict(result_metadata),
    )


def estimate_central_pair_metrics(result: PathBandsResult | GridBandsResult, matrix_dim: int) -> dict[str, float | None]:
    """Estimate bandwidth/gap diagnostics for the two central bands."""

    band_indices = tuple(int(index) for index in result.band_indices)
    positions = {band_index: pos for pos, band_index in enumerate(band_indices)}
    valence = int(matrix_dim) // 2 - 1
    conduction = int(matrix_dim) // 2
    missing = {
        "valence_bandwidth_ev": None,
        "conduction_bandwidth_ev": None,
        "mean_flat_bandwidth_ev": None,
        "central_bandwidth_ev": None,
        "central_manifold_span_ev": None,
        "central_gap_ev": None,
        "remote_gap_ev": None,
    }
    if valence not in positions or conduction not in positions:
        return missing

    energies = np.asarray(result.energies, dtype=float)
    val = energies[..., positions[valence]]
    con = energies[..., positions[conduction]]
    val_bw = float(np.max(val) - np.min(val))
    con_bw = float(np.max(con) - np.min(con))
    central = energies[..., [positions[valence], positions[conduction]]]
    span = float(np.max(central) - np.min(central))
    remote_gap: float | None = None
    lower_remote = valence - 1
    upper_remote = conduction + 1
    if lower_remote in positions and upper_remote in positions:
        lower_gap = val - energies[..., positions[lower_remote]]
        upper_gap = energies[..., positions[upper_remote]] - con
        remote_gap = float(min(np.min(lower_gap), np.min(upper_gap)))
    return {
        "valence_bandwidth_ev": val_bw,
        "conduction_bandwidth_ev": con_bw,
        "mean_flat_bandwidth_ev": 0.5 * (val_bw + con_bw),
        "central_bandwidth_ev": 0.5 * span,
        "central_manifold_span_ev": span,
        "central_gap_ev": float(np.min(con - val)),
        "remote_gap_ev": remote_gap,
    }

__all__ = [
    "DiagonalizeCallback",
    "GridBandsResult",
    "PathBandsResult",
    "compute_grid_bands",
    "compute_path_bands",
    "estimate_central_pair_metrics",
    "resolve_n_bands",
]
