from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


E2_OVER_4PI_EPS0_EV_NM = 1.439964547


@dataclass(frozen=True)
class ScreenedCoulombParams:
    """Double-gate screened Coulomb parameters in nanometer units."""

    epsilon_r: float = 8.0
    d_sc_nm: float = 25.0
    zero_cutoff_nm_inv: float = 1.0e-12
    finite_zero_limit: bool = True
    e2_over_4pi_eps0_ev_nm: float = E2_OVER_4PI_EPS0_EV_NM

    def __post_init__(self) -> None:
        if self.epsilon_r <= 0.0:
            raise ValueError("epsilon_r must be positive")
        if self.d_sc_nm < 0.0:
            raise ValueError("d_sc_nm must be non-negative")
        if self.zero_cutoff_nm_inv < 0.0:
            raise ValueError("zero_cutoff_nm_inv must be non-negative")
        if self.e2_over_4pi_eps0_ev_nm <= 0.0:
            raise ValueError("e2_over_4pi_eps0_ev_nm must be positive")


def _resolve_screened_params(
    params: ScreenedCoulombParams | Any | None,
    *,
    epsilon_r: float | None,
    d_sc_nm: float | None,
    zero_cutoff_nm_inv: float | None,
    finite_zero_limit: bool | None,
) -> ScreenedCoulombParams:
    if params is None:
        return ScreenedCoulombParams(
            epsilon_r=8.0 if epsilon_r is None else float(epsilon_r),
            d_sc_nm=25.0 if d_sc_nm is None else float(d_sc_nm),
            zero_cutoff_nm_inv=1.0e-12 if zero_cutoff_nm_inv is None else float(zero_cutoff_nm_inv),
            finite_zero_limit=True if finite_zero_limit is None else bool(finite_zero_limit),
        )

    if isinstance(params, ScreenedCoulombParams):
        base = params
    else:
        base = ScreenedCoulombParams(
            epsilon_r=float(getattr(params, "epsilon_r")),
            d_sc_nm=float(getattr(params, "d_sc_nm")),
            zero_cutoff_nm_inv=float(getattr(params, "zero_cutoff_nm_inv", 1.0e-12)),
            finite_zero_limit=bool(getattr(params, "finite_zero_limit", True)),
        )

    return ScreenedCoulombParams(
        epsilon_r=base.epsilon_r if epsilon_r is None else float(epsilon_r),
        d_sc_nm=base.d_sc_nm if d_sc_nm is None else float(d_sc_nm),
        zero_cutoff_nm_inv=base.zero_cutoff_nm_inv if zero_cutoff_nm_inv is None else float(zero_cutoff_nm_inv),
        finite_zero_limit=base.finite_zero_limit if finite_zero_limit is None else bool(finite_zero_limit),
        e2_over_4pi_eps0_ev_nm=base.e2_over_4pi_eps0_ev_nm,
    )


def screened_coulomb(
    q_nm_inv: float | complex | np.ndarray,
    params: ScreenedCoulombParams | Any | None = None,
    *,
    epsilon_r: float | None = None,
    d_sc_nm: float | None = None,
    zero_cutoff_nm_inv: float | None = None,
    finite_zero_limit: bool | None = None,
) -> float | np.ndarray:
    """Return ``V(q)`` for a double-gate screened Coulomb interaction.

    The convention is

        V(q) = 2*pi*(e^2/4*pi/eps0)/(epsilon_r*|q|) * tanh(|q| d_sc),

    with ``q`` in nm^-1 and the returned value in eV nm^2.  The finite
    ``q -> 0`` value is ``2*pi*(e^2/4*pi/eps0)*d_sc/epsilon_r`` when
    ``finite_zero_limit`` is enabled.
    """

    resolved = _resolve_screened_params(
        params,
        epsilon_r=epsilon_r,
        d_sc_nm=d_sc_nm,
        zero_cutoff_nm_inv=zero_cutoff_nm_inv,
        finite_zero_limit=finite_zero_limit,
    )
    q_array = np.asarray(q_nm_inv, dtype=np.complex128)
    scalar_input = q_array.ndim == 0
    q_abs = np.abs(q_array)
    values = np.zeros(q_abs.shape, dtype=float)

    prefactor = 2.0 * math.pi * resolved.e2_over_4pi_eps0_ev_nm / resolved.epsilon_r
    small = q_abs < resolved.zero_cutoff_nm_inv
    if resolved.finite_zero_limit:
        values[small] = prefactor * resolved.d_sc_nm

    large = ~small
    if np.any(large):
        values[large] = prefactor / q_abs[large] * np.tanh(q_abs[large] * resolved.d_sc_nm)

    if scalar_input:
        return float(values.reshape(()))
    return values


def screened_coulomb_matrix(
    q_nm_inv: np.ndarray,
    params: ScreenedCoulombParams | Any | None = None,
    **kwargs: object,
) -> np.ndarray:
    return np.asarray(screened_coulomb(q_nm_inv, params, **kwargs), dtype=float)


def reciprocal_cell_area_nm_inv_sq(b1_nm_inv: complex, b2_nm_inv: complex) -> float:
    return float(abs(complex(b1_nm_inv).real * complex(b2_nm_inv).imag - complex(b1_nm_inv).imag * complex(b2_nm_inv).real))


def real_space_cell_area_nm2_from_reciprocal(b1_nm_inv: complex, b2_nm_inv: complex) -> float:
    reciprocal_area = reciprocal_cell_area_nm_inv_sq(b1_nm_inv, b2_nm_inv)
    if reciprocal_area <= 0.0:
        raise ValueError("Reciprocal cell area must be positive")
    return float((2.0 * math.pi) ** 2 / reciprocal_area)
