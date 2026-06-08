from __future__ import annotations

from typing import Iterable

import numpy as np

from analysis.topology import (
    SewingTransform,
    TopologyResult,
    compute_system_topology_from_eigenvectors,
    compute_system_topology_from_grid_result,
    compute_system_topology_on_grid,
    normalize_state_indices,
)

from .bands import GridBandsResult, compute_bands_on_grid
from .lattice import TDBGLattice
from .params import TDBGParameters


def _normalize_band_indices(band_indices: int | Iterable[int]) -> tuple[int, ...]:
    return normalize_state_indices(band_indices)


def translation_srcmap(lattice: TDBGLattice, gvec: complex, *, atol: float = 1.0e-8) -> np.ndarray:
    """Map each q-site to the source site translated by ``gvec``.

    Entries are ``-1`` when the translated site lies outside the finite q-site
    cutoff.  The third ``q_sites`` column labels the q-sublattice sector and is
    preserved by the lookup.
    """

    q_sites = np.asarray(lattice.q_sites, dtype=float)
    if q_sites.ndim != 2 or q_sites.shape[1] < 3:
        raise ValueError(f"Expected lattice.q_sites shape (n, >=3), got {q_sites.shape}")
    shift = complex(gvec)
    out = np.full((q_sites.shape[0],), -1, dtype=int)
    coords = q_sites[:, :2]
    sectors = q_sites[:, 2]
    for isite, (xy, sector) in enumerate(zip(coords, sectors, strict=True)):
        target = xy + np.asarray([shift.real, shift.imag], dtype=float)
        same_sector = np.isclose(sectors, sector, atol=atol)
        distance = np.linalg.norm(coords - target[None, :], axis=1)
        matches = np.flatnonzero(same_sector & (distance <= float(atol)))
        if matches.size:
            out[isite] = int(matches[0])
    return out


def _q_site_sewing_transform(lattice: TDBGLattice, gvec: complex) -> SewingTransform:
    src = translation_srcmap(lattice, gvec)
    n_q = int(lattice.n_q)
    valid = src >= 0

    def transform(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.complex128)
        if arr.shape[0] != 4 * n_q:
            raise ValueError(f"Expected first axis {4 * n_q} for TDBG sewing, got {arr.shape}")
        reshaped = arr.reshape((n_q, 4) + arr.shape[1:], order="C")
        out = np.zeros_like(reshaped)
        out[valid] = reshaped[src[valid]]
        return out.reshape(arr.shape, order="C")

    return transform


def boundary_sewing_transforms(lattice: TDBGLattice) -> tuple[SewingTransform, SewingTransform]:
    """Return TDBG q-site sewing transforms for the two moire boundaries."""

    return (_q_site_sewing_transform(lattice, lattice.g_m1), _q_site_sewing_transform(lattice, lattice.g_m2))


# Backward-compatible private alias for older local callers.
_boundary_sewing_transforms = boundary_sewing_transforms


def compute_topology_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
    sewing_transforms: tuple[SewingTransform | None, SewingTransform | None] | None = None,
    metadata: dict[str, object] | None = None,
) -> TopologyResult:
    return compute_system_topology_from_eigenvectors(
        eigenvectors,
        band_indices,
        system="tdbg",
        valley=valley,
        k_grid_frac=k_grid_frac,
        sewing_transforms=sewing_transforms,
        index_metadata=metadata,
    )


def compute_topology_from_grid_result(
    grid_result: GridBandsResult,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    sewing_transforms: tuple[SewingTransform | None, SewingTransform | None] | None = None,
    metadata: dict[str, object] | None = None,
) -> TopologyResult:
    return compute_system_topology_from_grid_result(
        grid_result,
        band_indices,
        system="tdbg",
        valley=valley,
        sewing_transforms=sewing_transforms,
        index_metadata=metadata,
    )


def compute_topology_on_grid(
    mesh_size: int,
    lattice: TDBGLattice,
    params: TDBGParameters,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    n_bands: int | None = None,
    boundary_sewing: bool = True,
) -> TopologyResult:
    metadata = {"boundary_sewing": bool(boundary_sewing)}

    def grid_builder(trial_mesh: int, frac_shift: tuple[float, float], resolved_n_bands: int) -> GridBandsResult:
        return compute_bands_on_grid(
            trial_mesh,
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
            return_eigenvectors=True,
            endpoint=endpoint,
            frac_shift=frac_shift,
        )

    return compute_system_topology_on_grid(
        mesh_size,
        band_indices,
        system="tdbg",
        grid_builder=grid_builder,
        valley=valley,
        n_bands=n_bands,
        sewing_transforms_builder=(lambda: boundary_sewing_transforms(lattice)) if boundary_sewing else None,
        index_metadata=metadata,
    )


__all__ = [
    "TopologyResult",
    "boundary_sewing_transforms",
    "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result",
    "compute_topology_on_grid",
    "translation_srcmap",
]
