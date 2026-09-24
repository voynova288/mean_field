"""In-plane magnetic-field diagnostics for InAs/GaSb bilayers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .bands import kF_from_density_cm2
from .params import EIParams
from .units import E_CHARGE_SI, HBAR_SI

Array = NDArray[np.float64]


def delta_k_parallel_nm_inv(B_T: float | Array, d_nm: float) -> float | Array:
    """Relative electron-hole momentum shift e B_parallel d / hbar, in nm^-1."""

    B = np.asarray(B_T, dtype=np.float64)
    dk = E_CHARGE_SI * B * (float(d_nm) * 1e-9) / HBAR_SI * 1e-9
    if np.ndim(B_T) == 0:
        return float(dk)
    return dk


def fermi_circle_overlap_factor(delta_k_nm_inv: float | Array, kF_nm_inv: float) -> float | Array:
    """Normalized same-momentum overlap proxy for two shifted circular Fermi contours.

    This is only a single-particle hybridization diagnostic.  It is not fed back
    into the EI order parameter.
    """

    dk = np.asarray(delta_k_nm_inv, dtype=np.float64)
    kF = float(kF_nm_inv)
    if kF <= 0:
        out = np.zeros_like(dk)
    else:
        x = np.clip(dk / (2.0 * kF), 0.0, np.inf)
        out = np.zeros_like(x, dtype=np.float64)
        mask = x < 1.0
        out[mask] = 2.0 * np.arccos(x[mask]) / np.pi
    if np.ndim(delta_k_nm_inv) == 0:
        return float(out)
    return out


def hybridization_gap_proxy(
    B_T: Array,
    params: EIParams,
    *,
    t_hyb_mev: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Toy B_parallel suppression curve for a single-particle hybridization gap."""

    B = np.asarray(B_T, dtype=np.float64)
    kF = params.kF_nm_inv if params.kF_nm_inv is not None else kF_from_density_cm2(params.n0_cm2)
    dk = np.asarray(delta_k_parallel_nm_inv(B, params.d_eh_nm), dtype=np.float64)
    overlap = np.asarray(fermi_circle_overlap_factor(dk, kF), dtype=np.float64)
    gap = 2.0 * float(t_hyb_mev) * overlap
    return dk, overlap, gap
