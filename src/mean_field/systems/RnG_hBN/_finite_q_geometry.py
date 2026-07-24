from __future__ import annotations

import numpy as np


def _mesh_shape_from_k_grid_frac(k_grid_frac: np.ndarray) -> tuple[int, int]:
    frac = np.asarray(k_grid_frac, dtype=float)
    if frac.ndim != 2 or frac.shape[1] != 2:
        raise ValueError(f"Expected k_grid_frac shape (nk, 2), got {frac.shape}")
    nx = int(np.unique(np.round(frac[:, 0], decimals=12)).size)
    ny = int(np.unique(np.round(frac[:, 1], decimals=12)).size)
    if nx <= 0 or ny <= 0 or nx * ny != frac.shape[0]:
        raise ValueError(
            f"Cannot infer rectangular mesh from k_grid_frac shape {frac.shape}"
        )
    expected = np.asarray(
        [(ix / nx, iy / ny) for ix in range(nx) for iy in range(ny)],
        dtype=float,
    )
    if not np.allclose(frac, expected, atol=1.0e-10, rtol=0.0):
        raise ValueError(
            "RLG/hBN finite-q response requires row-major uniform "
            "fractional k_grid_frac"
        )
    return nx, ny


def _shift_k_index_with_wrap(
    k_index: int,
    q_shift: tuple[int, int],
    mesh_shape: tuple[int, int],
) -> tuple[int, tuple[int, int]]:
    nx, ny = int(mesh_shape[0]), int(mesh_shape[1])
    index = int(k_index)
    if index < 0 or index >= nx * ny:
        raise ValueError(f"k_index={index} is outside mesh {mesh_shape}")
    ix = index // ny
    iy = index % ny
    raw_x = ix + int(q_shift[0])
    raw_y = iy + int(q_shift[1])
    target_x = raw_x % nx
    target_y = raw_y % ny
    wrap_x = (raw_x - target_x) // nx
    wrap_y = (raw_y - target_y) // ny
    return int(target_x * ny + target_y), (int(wrap_x), int(wrap_y))


__all__ = ["_mesh_shape_from_k_grid_frac", "_shift_k_index_with_wrap"]
