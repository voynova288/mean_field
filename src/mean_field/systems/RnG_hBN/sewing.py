from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

SewingTransform = Callable[[np.ndarray], np.ndarray]


def _shift_rectangular_g_components(
    vectors: np.ndarray,
    *,
    local_basis_size: int,
    grid_shape: tuple[int, int],
    shift: tuple[int, int],
) -> np.ndarray:
    local = int(local_basis_size)
    nx, ny = (int(grid_shape[0]), int(grid_shape[1]))
    sx, sy = (int(shift[0]), int(shift[1]))
    block_dim = local * nx * ny
    array = np.asarray(vectors, dtype=np.complex128)
    if array.shape[0] != block_dim:
        raise ValueError(f"Expected first axis {block_dim}, got {array.shape[0]}")
    frames = int(np.prod(array.shape[1:], dtype=int)) if array.ndim > 1 else 1
    reshaped = array.reshape((local, nx, ny, frames), order="F")
    shifted = np.zeros_like(reshaped)
    for ix in range(nx):
        tx = ix + sx
        if tx < 0 or tx >= nx:
            continue
        for iy in range(ny):
            ty = iy + sy
            if ty < 0 or ty >= ny:
                continue
            shifted[:, tx, ty, :] = reshaped[:, ix, iy, :]
    return shifted.reshape(array.shape, order="F")


def rlg_hbn_reciprocal_shift_sewing_transform(
    *,
    local_basis_size: int,
    grid_shape: tuple[int, int],
    shift: tuple[int, int],
    valley: int,
) -> SewingTransform:
    valley_sign = int(valley)
    if valley_sign not in {-1, 1}:
        raise ValueError(f"Expected valley ±1, got {valley!r}")
    component_shift = (-valley_sign * int(shift[0]), -valley_sign * int(shift[1]))

    def transform(vectors: np.ndarray) -> np.ndarray:
        return _shift_rectangular_g_components(
            vectors,
            local_basis_size=int(local_basis_size),
            grid_shape=grid_shape,
            shift=component_shift,
        )

    return transform


def rlg_hbn_spin_flavor_reciprocal_shift_sewing_transform(
    *,
    local_basis_size: int,
    grid_shape: tuple[int, int],
    shift: tuple[int, int],
    spin_count: int = 2,
    valley_signs: Sequence[int] = (1, -1),
) -> SewingTransform:
    local = int(local_basis_size)
    nx, ny = (int(grid_shape[0]), int(grid_shape[1]))
    block_dim = local * nx * ny
    valleys = tuple(int(value) for value in valley_signs)
    if any(value not in {-1, 1} for value in valleys):
        raise ValueError(f"Expected valley signs ±1, got {valley_signs!r}")
    n_spin = int(spin_count)
    total_dim = n_spin * len(valleys) * block_dim

    def transform(vectors: np.ndarray) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.complex128)
        one_dimensional = array.ndim == 1
        matrix = array[:, None] if one_dimensional else array.reshape((array.shape[0], -1), order="F")
        if matrix.shape[0] != total_dim:
            raise ValueError(f"Expected first axis {total_dim}, got {matrix.shape[0]}")
        out = np.zeros_like(matrix)
        for ispin in range(n_spin):
            for iflavor, valley_sign in enumerate(valleys):
                start = (ispin * len(valleys) + iflavor) * block_dim
                stop = start + block_dim
                component_shift = (-valley_sign * int(shift[0]), -valley_sign * int(shift[1]))
                out[start:stop, :] = _shift_rectangular_g_components(
                    matrix[start:stop, :],
                    local_basis_size=local,
                    grid_shape=(nx, ny),
                    shift=component_shift,
                )
        if one_dimensional:
            return out[:, 0]
        return out.reshape(array.shape, order="F")

    return transform


def rlg_hbn_projected_micro_sewing_transforms(
    *,
    local_basis_size: int,
    grid_shape: tuple[int, int],
    spin_count: int = 2,
    valley_signs: Sequence[int] = (1, -1),
) -> tuple[SewingTransform, SewingTransform]:
    return (
        rlg_hbn_spin_flavor_reciprocal_shift_sewing_transform(
            local_basis_size=local_basis_size,
            grid_shape=grid_shape,
            shift=(1, 0),
            spin_count=spin_count,
            valley_signs=valley_signs,
        ),
        rlg_hbn_spin_flavor_reciprocal_shift_sewing_transform(
            local_basis_size=local_basis_size,
            grid_shape=grid_shape,
            shift=(0, 1),
            spin_count=spin_count,
            valley_signs=valley_signs,
        ),
    )


__all__ = [
    "rlg_hbn_projected_micro_sewing_transforms",
    "rlg_hbn_reciprocal_shift_sewing_transform",
    "rlg_hbn_spin_flavor_reciprocal_shift_sewing_transform",
]
