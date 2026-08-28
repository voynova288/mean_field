"""Time-reversal contract for the Xue--MacDonald Q=0 four-band basis."""

from __future__ import annotations

import numpy as np

from mean_field.core.hf.symmetry import (
    antiunitary_matrix_field_residual,
    project_antiunitary_matrix_field,
    validate_partner_involution,
)

from .zeng2022_hf import UniformKappaMesh

Array = np.ndarray


def xue2018_time_reversal_unitary() -> Array:
    """Return ``U_T=i s_y tensor tau_0`` in ``(c↑,v↑,c↓,v↓)`` order."""

    spin_y = np.asarray([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    return np.kron(1j * spin_y, np.eye(2, dtype=np.complex128))


def xue2018_opposite_k_indices(mesh: UniformKappaMesh, *, tolerance: float = 1.0e-12) -> Array:
    """Return the exact integer ``k -> -k`` permutation for a symmetric mesh."""

    mesh.validate()
    nkx, nky = mesh.shape
    if nkx * nky != mesh.nk:
        raise ValueError("mesh shape does not match the number of points")
    indices = np.arange(mesh.nk, dtype=np.int64).reshape(mesh.shape)
    partners = indices[::-1, ::-1].ravel()
    validate_partner_involution(partners, size=mesh.nk)
    points = np.asarray(mesh.points_ab_inv, dtype=np.float64)
    error = float(np.max(np.abs(points[partners] + points), initial=0.0))
    if error > float(tolerance):
        raise ValueError(f"mesh is not exactly inversion paired: {error:.3e}")
    weights = np.asarray(mesh.weights_ab2, dtype=np.float64)
    weight_error = float(np.max(np.abs(weights[partners] - weights), initial=0.0))
    if weight_error > float(tolerance) * max(1.0, float(np.max(np.abs(weights)))):
        raise ValueError(f"mesh inversion partners have unequal weights: {weight_error:.3e}")
    return partners


def project_xue2018_full_trs(field: Array, mesh: UniformKappaMesh) -> Array:
    """Project all Hermitian Pauli channels onto the complete Xue TRS subspace."""

    return project_antiunitary_matrix_field(
        field,
        partner_indices=xue2018_opposite_k_indices(mesh),
        unitary=xue2018_time_reversal_unitary(),
    )


def xue2018_trs_residual(field: Array, mesh: UniformKappaMesh) -> float:
    """Return the maximum full-field Xue time-reversal residual."""

    return antiunitary_matrix_field_residual(
        field,
        partner_indices=xue2018_opposite_k_indices(mesh),
        unitary=xue2018_time_reversal_unitary(),
    )


__all__ = [
    "project_xue2018_full_trs",
    "xue2018_opposite_k_indices",
    "xue2018_time_reversal_unitary",
    "xue2018_trs_residual",
]
