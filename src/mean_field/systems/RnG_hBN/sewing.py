from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from analysis.topology import BlockSewingSpec


def rlg_hbn_projected_micro_basis_sewing(
    *,
    local_basis_size: int,
    grid_shape: tuple[int, int],
    spin_count: int = 2,
    valley_signs: Sequence[int] = (1, -1),
    atol: float = 1.0e-8,
) -> BlockSewingSpec:
    """Describe generic seam relabeling for reconstructed projected-micro states.

    The reconstructed basis is ordered as spin, valley, rectangular reciprocal
    block, then local orbital. Encoding each reciprocal coordinate with its
    valley sign lets one common ``(+1, 0)/(0, +1)`` translation represent the
    opposite K/K' component relabelings without a system-private transform.
    """

    local = int(local_basis_size)
    nx, ny = (int(grid_shape[0]), int(grid_shape[1]))
    n_spin = int(spin_count)
    valleys = tuple(int(value) for value in valley_signs)
    if local <= 0 or nx <= 0 or ny <= 0 or n_spin <= 0:
        raise ValueError(
            "local_basis_size, grid_shape, and spin_count must be positive"
        )
    if not valleys or any(value not in {-1, 1} for value in valleys):
        raise ValueError(f"Expected non-empty valley signs ±1, got {valley_signs!r}")

    coordinates: list[tuple[float, float]] = []
    labels: list[int] = []
    for spin in range(n_spin):
        for flavor, valley in enumerate(valleys):
            label = spin * len(valleys) + flavor
            for iy in range(ny):
                for ix in range(nx):
                    coordinates.append((float(valley * ix), float(valley * iy)))
                    labels.append(label)
    return BlockSewingSpec(
        block_coordinates=np.asarray(coordinates, dtype=float),
        local_block_size=local,
        translations=((1.0, 0.0), (0.0, 1.0)),
        block_labels=np.asarray(labels, dtype=int),
        atol=float(atol),
    )


__all__ = ["rlg_hbn_projected_micro_basis_sewing"]
