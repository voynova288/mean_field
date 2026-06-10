"""Reusable band-structure result containers and solver loops.

System modules still own Hamiltonian construction, valley conventions, and band
selection policy.  This module only owns the generic bookkeeping for evaluating a
system-provided diagonalizer along a path or on a two-dimensional k-grid.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .lattice import KPath

BandDiagonalizer = Callable[[complex, int, bool], tuple[np.ndarray, np.ndarray | None]]


@dataclass(frozen=True)
class PathBandsResult:
    """Eigenvalues, and optionally eigenvectors, evaluated along a k-path."""

    path: KPath
    energies: np.ndarray
    eigenvectors: np.ndarray | None = None
    band_indices: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GridBandsResult:
    """Eigenvalues, and optionally eigenvectors, evaluated on a 2D k-grid."""

    k_grid_frac: np.ndarray
    kvec: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray | None = None
    band_indices: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def resolve_n_bands(basis_dim: int, n_bands: int | None) -> int:
    """Return the requested band count, defaulting to the full basis dimension."""

    resolved = int(basis_dim) if n_bands is None else int(n_bands)
    if resolved <= 0:
        raise ValueError(f"Expected a positive band count, got {resolved}")
    if resolved > int(basis_dim):
        raise ValueError(f"Requested {resolved} bands from a basis of dimension {basis_dim}")
    return resolved


def solve_bands_along_path(
    path: KPath,
    *,
    basis_dim: int,
    diagonalize: BandDiagonalizer,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
    band_indices: tuple[int, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PathBandsResult:
    """Evaluate a system-provided diagonalizer at each point on ``path``.

    ``diagonalize`` must accept ``(k, n_bands, return_eigenvectors)`` and return
    ``(eigenvalues, eigenvectors_or_none)``.  The callable is responsible for all
    system-specific Hamiltonian construction and band-selection semantics.
    """

    resolved_n_bands = resolve_n_bands(int(basis_dim), n_bands)
    energies = np.zeros((path.kvec.size, resolved_n_bands), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros((path.kvec.size, int(basis_dim), resolved_n_bands), dtype=np.complex128)

    for ik, kval in enumerate(path.kvec):
        evals, evecs = diagonalize(complex(kval), resolved_n_bands, bool(return_eigenvectors))
        evals = np.asarray(evals, dtype=float)
        if evals.shape != (resolved_n_bands,):
            raise ValueError(
                f"Diagonalizer returned eigenvalues with shape {evals.shape}, "
                f"expected {(resolved_n_bands,)} at path index {ik}."
            )
        energies[ik, :] = evals
        if return_eigenvectors:
            if evecs is None:
                raise ValueError("Diagonalizer returned no eigenvectors although return_eigenvectors=True")
            evecs = np.asarray(evecs, dtype=np.complex128)
            expected = (int(basis_dim), resolved_n_bands)
            if evecs.shape != expected:
                raise ValueError(
                    f"Diagonalizer returned eigenvectors with shape {evecs.shape}, "
                    f"expected {expected} at path index {ik}."
                )
            assert eigenvectors is not None
            eigenvectors[ik, :, :] = evecs

    return PathBandsResult(
        path=path,
        energies=energies,
        eigenvectors=eigenvectors,
        band_indices=band_indices,
        metadata={} if metadata is None else dict(metadata),
    )


def solve_bands_on_grid(
    k_grid_frac: np.ndarray,
    kvec: np.ndarray,
    *,
    basis_dim: int,
    diagonalize: BandDiagonalizer,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
    band_indices: tuple[int, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GridBandsResult:
    """Evaluate a system-provided diagonalizer on a two-dimensional k-grid."""

    kvec_array = np.asarray(kvec, dtype=np.complex128)
    if kvec_array.ndim != 2:
        raise ValueError(f"Expected a 2D k-vector grid, got shape {kvec_array.shape}")
    resolved_n_bands = resolve_n_bands(int(basis_dim), n_bands)
    grid_shape = kvec_array.shape
    energies = np.zeros(grid_shape + (resolved_n_bands,), dtype=float)
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.zeros(grid_shape + (int(basis_dim), resolved_n_bands), dtype=np.complex128)

    for grid_index in np.ndindex(grid_shape):
        evals, evecs = diagonalize(complex(kvec_array[grid_index]), resolved_n_bands, bool(return_eigenvectors))
        evals = np.asarray(evals, dtype=float)
        if evals.shape != (resolved_n_bands,):
            raise ValueError(
                f"Diagonalizer returned eigenvalues with shape {evals.shape}, "
                f"expected {(resolved_n_bands,)} at grid index {grid_index}."
            )
        energies[grid_index + (slice(None),)] = evals
        if return_eigenvectors:
            if evecs is None:
                raise ValueError("Diagonalizer returned no eigenvectors although return_eigenvectors=True")
            evecs = np.asarray(evecs, dtype=np.complex128)
            expected = (int(basis_dim), resolved_n_bands)
            if evecs.shape != expected:
                raise ValueError(
                    f"Diagonalizer returned eigenvectors with shape {evecs.shape}, "
                    f"expected {expected} at grid index {grid_index}."
                )
            assert eigenvectors is not None
            eigenvectors[grid_index + (slice(None), slice(None))] = evecs

    return GridBandsResult(
        k_grid_frac=np.asarray(k_grid_frac, dtype=float),
        kvec=kvec_array,
        energies=energies,
        eigenvectors=eigenvectors,
        band_indices=band_indices,
        metadata={} if metadata is None else dict(metadata),
    )


__all__ = [
    "BandDiagonalizer",
    "GridBandsResult",
    "PathBandsResult",
    "resolve_n_bands",
    "solve_bands_along_path",
    "solve_bands_on_grid",
]
