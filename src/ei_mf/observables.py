"""Observable postprocessing for EI mean-field solutions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

Array = NDArray[np.float64]


def lorentzian(x_mev: Array, eta_mev: float) -> Array:
    x = np.asarray(x_mev, dtype=np.float64)
    eta = float(eta_mev)
    return eta / np.pi / (x * x + eta * eta)


def compute_jdos(
    E_pair_mev: Array,
    weights_nm2: Array,
    eta_mev: float,
    *,
    omega_min_mev: float = 0.0,
    omega_max_mev: float | None = None,
    nomega: int = 1200,
) -> tuple[Array, Array]:
    """Point-quadrature JDOS retained for low-level tests and compatibility.

    Each radial cell is represented by one delta function at its center.  This
    is accurate only when the energy change per cell is well below the chosen
    broadening.  Production radial spectra should use :func:`compute_radial_jdos`.
    """

    E = np.asarray(E_pair_mev, dtype=np.float64)
    w = np.asarray(weights_nm2, dtype=np.float64)
    if omega_max_mev is None:
        omega_max_mev = max(10.0, float(np.max(E)) + 1.0)
    omega = np.linspace(float(omega_min_mev), float(omega_max_mev), int(nomega), dtype=np.float64)
    diff = omega[:, None] - E[None, :]
    jdos = lorentzian(diff, eta_mev) @ w
    return omega, jdos


def radial_jdos_refinement_factor(
    E_pair_mev: Array,
    eta_mev: float,
    *,
    min_factor: int = 4,
    samples_per_eta: float = 4.0,
    max_factor: int = 256,
) -> int:
    """Choose radial subcell refinement so adjacent energies resolve ``eta``."""

    E = np.asarray(E_pair_mev, dtype=np.float64)
    eta = float(eta_mev)
    if E.ndim != 1 or E.size < 2:
        raise ValueError("E_pair_mev must contain at least two points")
    if eta <= 0.0:
        raise ValueError("eta_mev must be positive")
    if min_factor < 1 or max_factor < min_factor or samples_per_eta <= 0.0:
        raise ValueError("invalid JDOS refinement controls")
    max_step = float(np.max(np.abs(np.diff(E))))
    required = int(np.ceil(samples_per_eta * max_step / eta))
    return min(max(int(min_factor), required), int(max_factor))


def _radial_edges_from_weights(weights_nm2: Array) -> Array:
    weights = np.asarray(weights_nm2, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0 or np.any(weights <= 0.0):
        raise ValueError("weights_nm2 must be a nonempty positive one-dimensional array")
    area = np.concatenate(([0.0], np.cumsum(weights)))
    return np.sqrt(4.0 * np.pi * area)


def compute_radial_jdos(
    k_nm_inv: Array,
    E_pair_mev: Array,
    weights_nm2: Array,
    eta_mev: float,
    *,
    omega_min_mev: float = 0.0,
    omega_max_mev: float | None = None,
    nomega: int = 1200,
    refinement_factor: int | None = None,
    min_refinement_factor: int = 4,
    samples_per_eta: float = 4.0,
    origin_boundary: str = "even_quadratic",
    spectral_weight: Array | None = None,
) -> tuple[Array, Array]:
    """Resolve the continuous radial integral before Lorentzian broadening.

    The old one-delta-per-cell quadrature produces a comb whenever
    ``|E[i+1]-E[i]|`` is comparable to ``eta``.  Here a shape-preserving PCHIP
    interpolant is sampled on refined annular subcells.  The original radial
    measure is preserved exactly, no post-hoc smoothing is applied, and PCHIP
    does not create extrema on monotone input intervals.  The default origin
    value is extrapolated linearly in ``k**2``, enforcing the local even-radial
    convention; alternate boundary choices are exposed for sensitivity tests.
    ``spectral_weight`` optionally supplies a nonnegative matrix-element or
    coherence-factor weight at the saved radial nodes.
    """

    k = np.asarray(k_nm_inv, dtype=np.float64)
    E = np.asarray(E_pair_mev, dtype=np.float64)
    weights = np.asarray(weights_nm2, dtype=np.float64)
    if not (k.ndim == E.ndim == weights.ndim == 1 and k.size == E.size == weights.size):
        raise ValueError("k, E_pair, and weights must be matching one-dimensional arrays")
    if k.size < 4 or np.any(np.diff(k) <= 0.0):
        raise ValueError("k_nm_inv must be strictly increasing with at least four points")
    edges = _radial_edges_from_weights(weights)
    if np.any(k <= edges[:-1]) or np.any(k >= edges[1:]):
        raise ValueError("k points must lie inside the annular cells implied by weights")
    if refinement_factor is None:
        factor = radial_jdos_refinement_factor(
            E,
            eta_mev,
            min_factor=min_refinement_factor,
            samples_per_eta=samples_per_eta,
        )
    else:
        factor = int(refinement_factor)
        if factor < 1:
            raise ValueError("refinement_factor must be positive")

    fractions = np.linspace(0.0, 1.0, factor + 1, dtype=np.float64)
    sub_edges = (edges[:-1, None] + (edges[1:] - edges[:-1])[:, None] * fractions[None, :])
    dense_edges = np.concatenate((sub_edges[:, :-1].reshape(-1), edges[-1:]))
    dense_k = 0.5 * (dense_edges[:-1] + dense_edges[1:])
    dense_weights = (dense_edges[1:] ** 2 - dense_edges[:-1] ** 2) / (4.0 * np.pi)
    if origin_boundary == "even_quadratic":
        # A smooth isotropic radial dispersion is even in k at the origin.
        x0, x1 = k[0] ** 2, k[1] ** 2
        E_left = float((E[0] * x1 - E[1] * x0) / (x1 - x0))
    elif origin_boundary == "clamped":
        E_left = float(E[0])
    elif origin_boundary == "linear":
        E_left = float(E[0] - k[0] * (E[1] - E[0]) / (k[1] - k[0]))
    else:
        raise ValueError("origin_boundary must be even_quadratic, clamped, or linear")
    E_right = float(E[-1] + (edges[-1] - k[-1]) * (E[-1] - E[-2]) / (k[-1] - k[-2]))
    interpolation_k = np.concatenate((edges[:1], k, edges[-1:]))
    interpolation_E = np.concatenate(([E_left], E, [E_right]))
    dense_E = np.asarray(
        PchipInterpolator(interpolation_k, interpolation_E, extrapolate=False)(dense_k),
        dtype=np.float64,
    )
    if spectral_weight is not None:
        node_weight = np.asarray(spectral_weight, dtype=np.float64)
        if node_weight.shape != E.shape or np.any(node_weight < 0.0):
            raise ValueError("spectral_weight must match E_pair and be nonnegative")
        if origin_boundary == "even_quadratic":
            w_left = float((node_weight[0] * x1 - node_weight[1] * x0) / (x1 - x0))
        elif origin_boundary == "clamped":
            w_left = float(node_weight[0])
        else:
            w_left = float(
                node_weight[0]
                - k[0] * (node_weight[1] - node_weight[0]) / (k[1] - k[0])
            )
        w_right = float(
            node_weight[-1]
            + (edges[-1] - k[-1])
            * (node_weight[-1] - node_weight[-2])
            / (k[-1] - k[-2])
        )
        interpolation_weight = np.concatenate(([max(0.0, w_left)], node_weight, [max(0.0, w_right)]))
        dense_spectral_weight = np.asarray(
            PchipInterpolator(
                interpolation_k, interpolation_weight, extrapolate=False
            )(dense_k),
            dtype=np.float64,
        )
        dense_weights = dense_weights * np.maximum(dense_spectral_weight, 0.0)
    return compute_jdos(
        dense_E,
        dense_weights,
        eta_mev,
        omega_min_mev=omega_min_mev,
        omega_max_mev=omega_max_mev,
        nomega=nomega,
    )
