from __future__ import annotations

import numpy as np

from mean_field.devtools.resample_b0_density_stack import resample_density_stack


def _stack_from_grid(grid: np.ndarray) -> np.ndarray:
    side = grid.shape[2]
    stack = np.empty((grid.shape[0], grid.shape[1], side * side), dtype=np.complex128)
    for j in range(side):
        for i in range(side):
            stack[:, :, i + j * side] = grid[:, :, i, j]
    return stack


def test_resample_density_stack_preserves_linear_field() -> None:
    source_lk = 2
    target_lk = 4
    side = source_lk + 1
    grid = np.zeros((2, 2, side, side), dtype=np.complex128)
    for j in range(side):
        for i in range(side):
            value = i / source_lk + 2.0 * j / source_lk
            grid[:, :, i, j] = np.asarray([[value, 1j * value], [-1j * value, -value]], dtype=np.complex128)

    target = resample_density_stack(_stack_from_grid(grid), source_lk=source_lk, target_lk=target_lk)

    for j in range(target_lk + 1):
        for i in range(target_lk + 1):
            ik = i + j * (target_lk + 1)
            value = i / target_lk + 2.0 * j / target_lk
            expected = np.asarray([[value, 1j * value], [-1j * value, -value]], dtype=np.complex128)
            np.testing.assert_allclose(target[:, :, ik], expected)


def test_resample_density_stack_keeps_hermitian_blocks() -> None:
    source = np.zeros((2, 2, 4), dtype=np.complex128)
    source[:, :, 0] = np.asarray([[1.0, 2.0 + 3.0j], [2.0 - 3.0j, -1.0]], dtype=np.complex128)
    source[:, :, 1] = np.asarray([[2.0, 1.0j], [-1.0j, 0.0]], dtype=np.complex128)
    source[:, :, 2] = np.asarray([[3.0, -4.0j], [4.0j, 1.0]], dtype=np.complex128)
    source[:, :, 3] = np.asarray([[4.0, 0.5 - 0.5j], [0.5 + 0.5j, 2.0]], dtype=np.complex128)

    target = resample_density_stack(source, source_lk=1, target_lk=3)

    for ik in range(target.shape[2]):
        np.testing.assert_allclose(target[:, :, ik], target[:, :, ik].conj().T)
