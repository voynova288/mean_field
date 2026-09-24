"""Radial momentum grids for isotropic two-dimensional EI calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def radial_grid(k_max_nm_inv: float, nk: int) -> tuple[Array, Array, Array]:
    """Return cell-centered radial k grid and 2D integration weights.

    The returned weights implement

        int d^2k/(2*pi)^2 f(k) = int_0^infty k dk/(2*pi) f(k)

    on annular cells.  ``edges`` is returned for metadata/debugging.
    """

    if k_max_nm_inv <= 0:
        raise ValueError("k_max_nm_inv must be positive")
    if nk < 4:
        raise ValueError("nk must be at least 4")
    edges = np.linspace(0.0, float(k_max_nm_inv), int(nk) + 1, dtype=np.float64)
    k = 0.5 * (edges[:-1] + edges[1:])
    weights = (edges[1:] ** 2 - edges[:-1] ** 2) / (4.0 * np.pi)
    return k, weights, edges


def q_floor_from_grid(k: Array, override: float | None = None) -> float:
    """Regularization floor for the q=0 Coulomb point."""

    if override is not None:
        if override <= 0:
            raise ValueError("q_floor override must be positive")
        return float(override)
    diffs = np.diff(np.asarray(k, dtype=np.float64))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 1e-6
    return max(1e-6, 0.25 * float(np.min(positive)))
