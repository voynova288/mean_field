"""Minimal bare-bubble pair-breaking optical weights for the EI two-band model.

This module is a diagnostic, not a complete THz-response implementation.  It
uses the diagonal bare-band in-plane current vertex and omits vertex
corrections, collective modes, remote-band dipoles, disorder, and experimental
field geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from .fermi import fermi_mev

Array = NDArray[np.float64]


@dataclass(frozen=True)
class PairBreakingWeights:
    thermal_factor: Array
    coherence_weight: Array
    radial_band_velocity_mev_nm: Array
    current_coherence_weight: Array
    conductivity_shape_weight: Array


def _even_radial_pchip_derivative(k: Array, values: Array) -> Array:
    """Derivative via a shape-preserving interpolation in ``x=k**2``.

    Smooth isotropic scalar dispersions are functions of ``k**2`` near the
    origin.  Differentiating in x and multiplying by ``2k`` enforces zero
    radial slope at k=0 and exactly reproduces a parabolic dispersion.
    """

    k = np.asarray(k, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if k.ndim != 1 or values.shape != k.shape or k.size < 4:
        raise ValueError("k and values must be matching one-dimensional arrays")
    if np.any(np.diff(k) <= 0.0) or k[0] <= 0.0:
        raise ValueError("k must be strictly increasing and cell-centered above zero")
    x = k * k
    derivative_in_x = PchipInterpolator(x, values, extrapolate=False).derivative()(x)
    return np.asarray(2.0 * k * derivative_in_x, dtype=np.float64)


def pair_breaking_weights(
    k_nm_inv: Array,
    eps_P_mev: Array,
    Delta_order_mev: Array,
    xi_mev: Array,
    eta_mev: Array,
    temperature_K: float,
    *,
    energy_floor_mev: float = 1e-12,
) -> PairBreakingWeights:
    """Return constant-vertex and diagonal-current pair-breaking weights.

    In the Nambu basis ``(a_k, b^dagger_-k)``, use

    ``h = eta*tau_0/2 + xi*tau_z/2 + Delta*tau_x/2``.

    For conduction-electron and valence-hole dispersions, the diagonal radial
    current has a tau-z coefficient proportional to
    ``-(d eps_a/dk + d eps_b/dk)/2 = -(d eps_P/dk)/2``.  The interbranch matrix
    element therefore has squared radial shape

    ``(d eps_P/dk)^2 * Delta^2 / (4 E^2)``.

    Angular averaging of an x-polarized isotropic response adds ``<cos^2>=1/2``,
    giving ``current_coherence_weight`` below.  Common charge/hbar prefactors
    are omitted.  ``conductivity_shape_weight`` additionally includes the
    Kubo ``1/E`` factor when frequency is represented as transition energy.
    """

    k = np.asarray(k_nm_inv, dtype=np.float64)
    eps_P = np.asarray(eps_P_mev, dtype=np.float64)
    Delta = np.asarray(Delta_order_mev, dtype=np.float64)
    xi = np.asarray(xi_mev, dtype=np.float64)
    eta = np.asarray(eta_mev, dtype=np.float64)
    if not (k.shape == eps_P.shape == Delta.shape == xi.shape == eta.shape):
        raise ValueError("all pair-breaking arrays must have matching shapes")
    E = np.sqrt(xi * xi + Delta * Delta)
    E_safe = np.maximum(E, float(energy_floor_mev))
    thermal = (
        1.0
        - fermi_mev((E - eta) / 2.0, temperature_K)
        - fermi_mev((E + eta) / 2.0, temperature_K)
    )
    thermal = np.clip(thermal, 0.0, 1.0)
    coherence = thermal * (Delta / E_safe) ** 2
    d_eps_P_dk = _even_radial_pchip_derivative(k, eps_P)
    current_coherence = coherence * d_eps_P_dk**2 / 8.0
    conductivity = current_coherence / E_safe
    return PairBreakingWeights(
        thermal_factor=thermal,
        coherence_weight=coherence,
        radial_band_velocity_mev_nm=d_eps_P_dk,
        current_coherence_weight=current_coherence,
        conductivity_shape_weight=conductivity,
    )
