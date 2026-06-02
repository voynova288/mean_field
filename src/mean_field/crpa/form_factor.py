from __future__ import annotations

import numpy as np


PRODUCTION_FORM_FACTOR_MODE = "k_periodic_zero_fill"
LEGACY_ZERO_FILL_TEST_MODE = "zhang_zero_fill"
HF_PERIODIC_ROLL_DIAGNOSTIC_MODE = "hf_periodic_roll"


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


def _normalize_form_factor_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"zhang", "zhang_zero_fill", "zero_fill"}:
        return LEGACY_ZERO_FILL_TEST_MODE
    if normalized in {
        "hf",
        "hf_periodic",
        "hf_compatible",
        "k_periodic_zero_fill",
        "kperiodic_zero_fill",
        "hf_kperiodic_zero_fill",
        "hf_zero_fill",
        "production",
    }:
        return PRODUCTION_FORM_FACTOR_MODE
    if normalized in {"hf_periodic_roll", "periodic_roll", "roll", "wrap", "wrapped"}:
        return HF_PERIODIC_ROLL_DIAGNOSTIC_MODE
    raise ValueError(f"Unsupported cRPA form-factor mode: {mode!r}")


def normalize_form_factor_mode(mode: str) -> str:
    return _normalize_form_factor_mode(mode)


def _shift_plane_wave_coefficients_zero_fill(values: np.ndarray, dm: int, dn: int) -> np.ndarray:
    """Legacy/test shift with zero fill, not periodic wrapping."""

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


def _shift_plane_wave_coefficients(
    values: np.ndarray,
    dm: int,
    dn: int,
    *,
    mode: str = PRODUCTION_FORM_FACTOR_MODE,
) -> np.ndarray:
    resolved_mode = _normalize_form_factor_mode(mode)
    if resolved_mode == HF_PERIODIC_ROLL_DIAGNOSTIC_MODE:
        # Diagnostic only: this matches the periodic-G HF overlap algebra, but
        # it is not the finite plane-wave vertex in Zhang Eq. 22.
        return np.roll(np.asarray(values, dtype=np.complex128), shift=(0, int(dm), int(dn), 0), axis=(0, 1, 2, 3))
    return _shift_plane_wave_coefficients_zero_fill(values, dm, dn)


def compute_lambda_stack(
    left_eigenvectors: np.ndarray,
    right_eigenvectors: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    q_shifts: tuple[tuple[int, int], ...],
    local_basis_size: int = 4,
    left_band_indices: np.ndarray | None = None,
    right_band_indices: np.ndarray | None = None,
    form_factor_mode: str = PRODUCTION_FORM_FACTOR_MODE,
) -> np.ndarray:
    """Compute lambda matrices for one ``k+q`` and one ``k``.

    The returned array has shape ``(N_Q, N_left_band, N_right_band)`` and
    follows Zhang SM Eq. 22:

    ``lambda_Q = sum_G C_left[G + Q]^* C_right[G]``.

    The production mode keeps the k-grid periodicity outside this routine by
    passing ``Q + wrap`` from the caller, but the finite plane-wave G cutoff is
    not a torus: coefficients outside the retained G shell are zero-filled.
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
    resolved_mode = _normalize_form_factor_mode(form_factor_mode)
    for iq, (dm, dn) in enumerate(q_shifts):
        shifted_right = _shift_plane_wave_coefficients(right, int(dm), int(dn), mode=resolved_mode)
        out[iq] = np.einsum("lxyi,lxyj->ij", left_conj, shifted_right, optimize=True)
    return out


def q0_identity_error(
    eigenvectors: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    local_basis_size: int = 4,
    form_factor_mode: str = PRODUCTION_FORM_FACTOR_MODE,
) -> float:
    lam = compute_lambda_stack(
        eigenvectors,
        eigenvectors,
        grid_shape=grid_shape,
        q_shifts=((0, 0),),
        local_basis_size=local_basis_size,
        form_factor_mode=form_factor_mode,
    )[0]
    return float(np.max(np.abs(lam - np.eye(lam.shape[0], dtype=np.complex128))))
