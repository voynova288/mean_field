"""System-independent antiunitary symmetry helpers for matrix fields."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def validate_partner_involution(partner_indices: Array, *, size: int | None = None) -> Array:
    """Return an exact integer involution mapping each mesh point to its partner."""

    raw = np.asarray(partner_indices)
    if raw.ndim != 1:
        raise ValueError("partner_indices must be one-dimensional")
    if raw.dtype.kind not in "iu":
        if not np.all(np.equal(raw, np.rint(raw))):
            raise TypeError("partner_indices must contain exact integers")
    partners = raw.astype(np.int64, copy=False)
    expected = partners.size if size is None else int(size)
    if partners.size != expected:
        raise ValueError(f"partner_indices length {partners.size} does not match {expected}")
    if np.any(partners < 0) or np.any(partners >= expected):
        raise ValueError("partner_indices contain an out-of-range index")
    indices = np.arange(expected, dtype=np.int64)
    if not np.array_equal(partners[partners], indices):
        raise ValueError("partner_indices must be an exact involution")
    return partners


def validate_antiunitary_unitary(unitary: Array, *, dimension: int | None = None) -> Array:
    """Validate the unitary part ``U`` of an antiunitary operation ``U K``."""

    matrix = np.asarray(unitary, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("antiunitary unitary must be a square matrix")
    if dimension is not None and matrix.shape != (int(dimension), int(dimension)):
        raise ValueError("antiunitary unitary has the wrong dimension")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("antiunitary unitary must be finite")
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    error = float(np.max(np.abs(matrix @ matrix.conj().T - identity)))
    if error > 1.0e-12:
        raise ValueError(f"antiunitary unitary is not unitary: {error:.3e}")
    return matrix


def antiunitary_transform_matrix_field(
    field: Array,
    *,
    partner_indices: Array,
    unitary: Array,
) -> Array:
    """Apply ``F(k) -> U F(partner(k))^* U^dagger`` to ``(d,d,nk)`` arrays."""

    matrices = np.asarray(field, dtype=np.complex128)
    if matrices.ndim != 3 or matrices.shape[0] != matrices.shape[1]:
        raise ValueError("field must have shape (dimension,dimension,nk)")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("field must be finite")
    partners = validate_partner_involution(partner_indices, size=matrices.shape[2])
    transform = validate_antiunitary_unitary(unitary, dimension=matrices.shape[0])
    paired = matrices[:, :, partners].conj()
    return np.einsum(
        "ac,cdk,bd->abk",
        transform,
        paired,
        transform.conj(),
        optimize=True,
    )


def project_antiunitary_matrix_field(
    field: Array,
    *,
    partner_indices: Array,
    unitary: Array,
) -> Array:
    """Project a matrix field onto the invariant subspace of an antiunitary map."""

    matrices = np.asarray(field, dtype=np.complex128)
    transformed = antiunitary_transform_matrix_field(
        matrices,
        partner_indices=partner_indices,
        unitary=unitary,
    )
    return 0.5 * (matrices + transformed)


def antiunitary_matrix_field_residual(
    field: Array,
    *,
    partner_indices: Array,
    unitary: Array,
) -> float:
    """Return the maximum absolute antiunitary covariance residual."""

    matrices = np.asarray(field, dtype=np.complex128)
    transformed = antiunitary_transform_matrix_field(
        matrices,
        partner_indices=partner_indices,
        unitary=unitary,
    )
    return float(np.max(np.abs(matrices - transformed), initial=0.0))


__all__ = [
    "antiunitary_matrix_field_residual",
    "antiunitary_transform_matrix_field",
    "project_antiunitary_matrix_field",
    "validate_antiunitary_unitary",
    "validate_partner_involution",
]
