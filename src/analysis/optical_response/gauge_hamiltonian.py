from __future__ import annotations

import numpy as np

from .gauge_data import HamiltonianGaugeData
from .gauge_primitives import energy_difference_inverse, matrix_in_eigenbasis

def hamiltonian_gauge_data(
    energies: np.ndarray,
    eigenvectors: np.ndarray,
    dhdk: np.ndarray,
    *,
    denominator_cutoff: float = 1.0e-10,
    d2hdk: np.ndarray | None = None,
    external_connection: np.ndarray | None = None,
) -> HamiltonianGaugeData:
    """Build WannierBerri-style Hamiltonian-gauge derivative ingredients.

    ``dhdk`` must have shape ``(ndim,basis,basis)``.  ``d2hdk`` is optional and
    must have shape ``(ndim,ndim,basis,basis)`` when supplied.  Momentum units
    are whatever units the derivatives use; the caller must document them.
    """

    e = np.asarray(energies, dtype=float)
    u = np.asarray(eigenvectors, dtype=np.complex128)
    raw_dh = np.asarray(dhdk, dtype=np.complex128)
    if e.ndim != 1:
        raise ValueError(f"energies must be 1D, got shape {e.shape}")
    if u.shape != (raw_dh.shape[-1], e.size):
        raise ValueError(f"eigenvectors shape {u.shape} incompatible with dhdk {raw_dh.shape} and energies {e.shape}")
    if raw_dh.ndim != 3 or raw_dh.shape[-2:] != (u.shape[0], u.shape[0]):
        raise ValueError(f"dhdk must have shape (ndim,basis,basis), got {raw_dh.shape}")

    velocity_h = matrix_in_eigenbasis(u, raw_dh)
    inv = energy_difference_inverse(e, cutoff=float(denominator_cutoff))
    dcov = -velocity_h * inv[None, :, :]
    berry = 1.0j * dcov
    if external_connection is not None:
        abar = np.asarray(external_connection, dtype=np.complex128)
        if abar.shape != berry.shape:
            raise ValueError(f"external_connection has shape {abar.shape}, expected {berry.shape}")
        berry = berry + abar

    second_velocity_h = None
    if d2hdk is not None:
        raw_d2 = np.asarray(d2hdk, dtype=np.complex128)
        expected = (raw_dh.shape[0], raw_dh.shape[0], u.shape[0], u.shape[0])
        if raw_d2.shape != expected:
            raise ValueError(f"d2hdk has shape {raw_d2.shape}, expected {expected}")
        second_velocity_h = matrix_in_eigenbasis(u, raw_d2)

    return HamiltonianGaugeData(
        energies=e,
        eigenvectors=u,
        velocity_h=velocity_h,
        energy_difference_inverse=inv,
        dcov=dcov,
        berry_connection=berry,
        second_velocity_h=second_velocity_h,
    )

__all__ = [
    "hamiltonian_gauge_data",
]
