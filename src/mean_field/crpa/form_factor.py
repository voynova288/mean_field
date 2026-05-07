from __future__ import annotations

import numpy as np


def reshape_plane_wave_vectors(
    eigenvectors: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    local_basis_size: int = 4,
) -> np.ndarray:
    vecs = np.asarray(eigenvectors, dtype=np.complex128)
    if vecs.ndim != 2:
        raise ValueError(f"Expected eigenvectors shape (basis, band), got {vecs.shape}")
    nx, ny = (int(grid_shape[0]), int(grid_shape[1]))
    expected = int(local_basis_size) * nx * ny
    if vecs.shape[0] != expected:
        raise ValueError(f"Expected basis dimension {expected}, got {vecs.shape[0]}")
    return vecs.reshape(int(local_basis_size), nx, ny, vecs.shape[1], order="F")


def _shift_plane_wave_coefficients(values: np.ndarray, dm: int, dn: int) -> np.ndarray:
    """Shift plane-wave coefficients with zero fill, not periodic wrapping."""

    arr = np.asarray(values, dtype=np.complex128)
    nx, ny = arr.shape[1], arr.shape[2]
    out = np.zeros_like(arr)
    dm = int(dm)
    dn = int(dn)
    if abs(dm) >= nx or abs(dn) >= ny:
        return out

    if dm >= 0:
        dst_x = slice(dm, nx)
        src_x = slice(0, nx - dm)
    else:
        dst_x = slice(0, nx + dm)
        src_x = slice(-dm, nx)
    if dn >= 0:
        dst_y = slice(dn, ny)
        src_y = slice(0, ny - dn)
    else:
        dst_y = slice(0, ny + dn)
        src_y = slice(-dn, ny)

    out[:, dst_x, dst_y, :] = arr[:, src_x, src_y, :]
    return out


def compute_lambda_stack(
    left_eigenvectors: np.ndarray,
    right_eigenvectors: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    q_shifts: tuple[tuple[int, int], ...],
    local_basis_size: int = 4,
    left_band_indices: np.ndarray | None = None,
    right_band_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute lambda matrices for one ``k+q`` and one ``k``.

    The returned array has shape ``(N_Q, N_left_band, N_right_band)`` and
    follows Zhang SM Eq. 22:

    ``lambda_Q = sum_G C_left[G + Q]^* C_right[G]``.
    """

    left = reshape_plane_wave_vectors(
        left_eigenvectors,
        grid_shape=grid_shape,
        local_basis_size=local_basis_size,
    )
    right = reshape_plane_wave_vectors(
        right_eigenvectors,
        grid_shape=grid_shape,
        local_basis_size=local_basis_size,
    )
    if left_band_indices is not None:
        left = left[:, :, :, np.asarray(left_band_indices, dtype=int)]
    if right_band_indices is not None:
        right = right[:, :, :, np.asarray(right_band_indices, dtype=int)]

    left_conj = left.conjugate()
    out = np.zeros((len(q_shifts), left.shape[-1], right.shape[-1]), dtype=np.complex128)
    for iq, (dm, dn) in enumerate(q_shifts):
        shifted_right = _shift_plane_wave_coefficients(right, int(dm), int(dn))
        out[iq] = np.einsum("lxyi,lxyj->ij", left_conj, shifted_right, optimize=True)
    return out


def q0_identity_error(
    eigenvectors: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    local_basis_size: int = 4,
) -> float:
    lam = compute_lambda_stack(
        eigenvectors,
        eigenvectors,
        grid_shape=grid_shape,
        q_shifts=((0, 0),),
        local_basis_size=local_basis_size,
    )[0]
    return float(np.max(np.abs(lam - np.eye(lam.shape[0], dtype=np.complex128))))
