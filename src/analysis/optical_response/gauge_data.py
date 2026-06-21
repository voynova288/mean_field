from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class HamiltonianGaugeData:
    """Hamiltonian-gauge derivative ingredients.

    This mirrors the WannierBerri convention used in
    ``wannierberri/data_K/data_K.py``:

    ``D_H = -Xbar('Ham', 1) * dEig_inv[..., None]`` and
    ``A_H = 1j * D_H`` when no external position/Berry-connection terms are
    present.  Eigenvectors are assumed to be stored as columns.
    """

    energies: np.ndarray  # (nb,)
    eigenvectors: np.ndarray  # (basis, nb)
    velocity_h: np.ndarray  # (ndim, nb, nb), <u_n|d_a H|u_m>
    energy_difference_inverse: np.ndarray  # (nb, nb), 1/(E_n-E_m), diag/small gaps -> 0
    dcov: np.ndarray  # (ndim, nb, nb), WannierBerri D_H^a
    berry_connection: np.ndarray  # (ndim, nb, nb), A_H^a = i D_H^a + external terms
    second_velocity_h: np.ndarray | None = None

@dataclass(frozen=True)
class GeneralizedDerivativeData:
    """Generalized derivative of the Berry connection.

    ``values[deriv_axis, connection_axis, n, m]`` stores
    ``(A^{connection_axis}_{n m})_{; deriv_axis}`` in the same index/order
    convention as the existing shift-current code.
    """

    values: np.ndarray
    skipped_small_denominators: int

@dataclass(frozen=True)
class PairGeneralizedDerivativeData:
    """Selected-pair generalized derivative.

    ``values[deriv_axis, connection_axis]`` stores
    ``(A^{connection_axis}_{n m})_{; deriv_axis}`` for one pair ``(n,m)``.
    """

    values: np.ndarray
    skipped_small_denominators: int

__all__ = [
    "HamiltonianGaugeData",
    "GeneralizedDerivativeData",
    "PairGeneralizedDerivativeData",
]
