from __future__ import annotations

import numpy as np

def shift_integrand_from_pair_generalized_derivative(
    berry_connection: np.ndarray,
    pair_gen_der: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    deriv_axis: int,
    optical_axis: int,
) -> float:
    """Return ``Im[A_mn^b (A_nm^b)_{;a}]`` from a selected-pair derivative.

    ``pair_gen_der`` has shape ``(ndim,ndim)`` and stores the generalized
    derivative for ``(initial_band, final_band)`` only.
    """

    A = np.asarray(berry_connection, dtype=np.complex128)
    G = np.asarray(pair_gen_der, dtype=np.complex128)
    b = int(optical_axis)
    a = int(deriv_axis)
    n = int(initial_band)
    m = int(final_band)
    if G.ndim != 2:
        raise ValueError(f"pair_gen_der must have shape (ndim,ndim), got {G.shape}")
    return float(np.imag(A[b, m, n] * G[a, b]))

def shift_vector_from_pair_generalized_derivative(
    berry_connection: np.ndarray,
    pair_gen_der: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    deriv_axis: int,
    optical_axis: int,
    metric_cutoff: float = 1.0e-24,
) -> float:
    """Selected-pair version of :func:`shift_vector_from_generalized_derivative`."""

    A = np.asarray(berry_connection, dtype=np.complex128)
    b = int(optical_axis)
    n = int(initial_band)
    m = int(final_band)
    metric = float(abs(A[b, m, n]) ** 2)
    if metric <= float(metric_cutoff):
        return float("nan")
    return shift_integrand_from_pair_generalized_derivative(
        A,
        pair_gen_der,
        initial_band=n,
        final_band=m,
        deriv_axis=deriv_axis,
        optical_axis=b,
    ) / metric

def shift_integrand_from_generalized_derivative(
    berry_connection: np.ndarray,
    berry_connection_gen_der: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    deriv_axis: int,
    optical_axis: int,
) -> float:
    """Return ``Im[A_mn^b (A_nm^b)_{;a}]`` for a direct transition ``n -> m``.

    ``initial_band`` is the occupied/initial band ``n`` and ``final_band`` is
    the empty/final band ``m``.  The returned quantity is the gauge-invariant
    ``|A|^2 S`` factor used in shift-current integrands.
    """

    A = np.asarray(berry_connection, dtype=np.complex128)
    G = np.asarray(berry_connection_gen_der, dtype=np.complex128)
    b = int(optical_axis)
    a = int(deriv_axis)
    n = int(initial_band)
    m = int(final_band)
    return float(np.imag(A[b, m, n] * G[a, b, n, m]))

def shift_vector_from_generalized_derivative(
    berry_connection: np.ndarray,
    berry_connection_gen_der: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    deriv_axis: int,
    optical_axis: int,
    metric_cutoff: float = 1.0e-24,
) -> float:
    """Return ``S = Im[A_mn (A_nm)_;] / |A_mn|^2`` away from optical zeros."""

    A = np.asarray(berry_connection, dtype=np.complex128)
    b = int(optical_axis)
    n = int(initial_band)
    m = int(final_band)
    metric = float(abs(A[b, m, n]) ** 2)
    if metric <= float(metric_cutoff):
        return float("nan")
    return shift_integrand_from_generalized_derivative(
        A,
        berry_connection_gen_der,
        initial_band=n,
        final_band=m,
        deriv_axis=deriv_axis,
        optical_axis=b,
    ) / metric

def normalized_u1_link(evecs0: np.ndarray, evecs1: np.ndarray, band: int) -> complex:
    """Return the U(1) link ``<u_band(k)|u_band(k+dk)>/abs(...)``."""

    u0 = np.asarray(evecs0, dtype=np.complex128)
    u1 = np.asarray(evecs1, dtype=np.complex128)
    link = complex(np.vdot(u0[:, int(band)], u1[:, int(band)]))
    if abs(link) <= 1.0e-14:
        raise ValueError(f"Vanishing U(1) link for band {band}")
    return link / abs(link)

def link_shift_vector(
    evecs0: np.ndarray,
    evecs1: np.ndarray,
    berry_connection0: np.ndarray,
    berry_connection1: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    optical_axis: int,
    step: float,
) -> float:
    """Gauge-invariant Wilson-link finite-difference shift vector.

    This is the safe phase-derivative route.  It avoids differentiating a raw
    eigenvector or matrix-element phase by parallel-transporting the optical
    matrix element at ``k+step`` back to the gauge at ``k``.
    """

    n = int(initial_band)
    m = int(final_band)
    b = int(optical_axis)
    A0 = np.asarray(berry_connection0, dtype=np.complex128)
    A1 = np.asarray(berry_connection1, dtype=np.complex128)
    if A0.shape != A1.shape or A0.ndim != 3:
        raise ValueError(f"berry connections must both have shape (ndim,nb,nb), got {A0.shape} and {A1.shape}")
    link_m = normalized_u1_link(evecs0, evecs1, m)
    link_n = normalized_u1_link(evecs0, evecs1, n)
    phase_product = A1[b, m, n] * A0[b, n, m] * link_m * np.conj(link_n)
    if abs(phase_product) <= 1.0e-30:
        return float("nan")
    return float(-np.angle(phase_product) / float(step))

__all__ = [
    "shift_integrand_from_pair_generalized_derivative",
    "shift_vector_from_pair_generalized_derivative",
    "shift_integrand_from_generalized_derivative",
    "shift_vector_from_generalized_derivative",
    "normalized_u1_link",
    "link_shift_vector",
]
