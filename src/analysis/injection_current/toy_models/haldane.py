from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

import numpy as np

from analysis.injection_current import (
    accumulate_injection_spectrum,
    component_label,
    injection_component_kernel,
    positive_injection_transition_terms,
    precompute_injection_current_tensors,
)
from analysis.shift_current import (
    Component,
    add_transitions_to_integral,
    parse_component,
    positive_transition_terms as positive_shift_transition_terms,
    precompute_shift_current_tensors,
)


@dataclass(frozen=True)
class HaldaneParams:
    """Lin--Hsu 2025 Haldane-model checkpoint parameters.

    Units follow the work document: nearest-neighbor distance ``a0 = 1`` and
    ``t1`` is the energy unit.  The default ``phi=-pi/2`` and ``M=0.4`` match
    the Fig. 4(a) linear-injection checkpoint discussed in the plan.
    """

    t1: float = 1.0
    t2: float = 0.2
    phi: float = -math.pi / 2.0
    M: float = 0.4


def bravais_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Triangular Bravais primitive vectors with nearest-neighbor length 1."""

    return (
        np.asarray([1.5, math.sqrt(3.0) / 2.0], dtype=float),
        np.asarray([1.5, -math.sqrt(3.0) / 2.0], dtype=float),
    )


def reciprocal_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Reciprocal vectors ``b_i`` satisfying ``a_i dot b_j = 2*pi delta_ij``."""

    return (
        np.asarray([2.0 * math.pi / 3.0, 2.0 * math.pi / math.sqrt(3.0)], dtype=float),
        np.asarray([2.0 * math.pi / 3.0, -2.0 * math.pi / math.sqrt(3.0)], dtype=float),
    )


def unit_cell_area() -> float:
    a1, a2 = bravais_vectors()
    return abs(float(a1[0] * a2[1] - a1[1] * a2[0]))


def bz_area() -> float:
    b1, b2 = reciprocal_vectors()
    return abs(float(b1[0] * b2[1] - b1[1] * b2[0]))


def high_symmetry_points() -> dict[str, np.ndarray]:
    return {
        "Gamma": np.asarray([0.0, 0.0], dtype=float),
        "K": np.asarray([2.0 * math.pi / 3.0, 2.0 * math.pi / (3.0 * math.sqrt(3.0))], dtype=float),
        "Kprime": np.asarray([2.0 * math.pi / 3.0, -2.0 * math.pi / (3.0 * math.sqrt(3.0))], dtype=float),
        "M": np.asarray([2.0 * math.pi / 3.0, 0.0], dtype=float),
    }


def _coefficients(k_xy: np.ndarray, params: HaldaneParams) -> tuple[float, float, float, float]:
    kx, ky = np.asarray(k_xy, dtype=float).reshape(2)
    s3 = math.sqrt(3.0)
    a = 1.5 * kx
    b = 0.5 * s3 * ky
    t1 = float(params.t1)
    t2 = float(params.t2)
    cp = math.cos(float(params.phi))
    sp = math.sin(float(params.phi))
    f0 = 2.0 * t2 * cp * (math.cos(s3 * ky) + 2.0 * math.cos(a) * math.cos(b))
    fx = t1 * (math.cos(kx) + 2.0 * math.cos(0.5 * kx) * math.cos(b))
    fy = -t1 * (math.sin(kx) - 2.0 * math.sin(0.5 * kx) * math.cos(b))
    fz = float(params.M) + 2.0 * t2 * sp * (2.0 * math.cos(a) * math.sin(b) - math.sin(s3 * ky))
    return f0, fx, fy, fz


def _matrix_from_coefficients(f0: float, fx: float, fy: float, fz: float) -> np.ndarray:
    return np.asarray(
        [[f0 + fz, fx - 1.0j * fy], [fx + 1.0j * fy, f0 - fz]],
        dtype=np.complex128,
    )


def hamiltonian(k_xy: np.ndarray | tuple[float, float], params: HaldaneParams = HaldaneParams()) -> np.ndarray:
    """Return the 2x2 Haldane Bloch Hamiltonian from Lin--Hsu Eq. (1)."""

    return _matrix_from_coefficients(*_coefficients(np.asarray(k_xy, dtype=float), params))


def _first_derivatives_coefficients(k_xy: np.ndarray, params: HaldaneParams) -> np.ndarray:
    kx, ky = np.asarray(k_xy, dtype=float).reshape(2)
    s3 = math.sqrt(3.0)
    a = 1.5 * kx
    b = 0.5 * s3 * ky
    t1 = float(params.t1)
    t2 = float(params.t2)
    cp = math.cos(float(params.phi))
    sp = math.sin(float(params.phi))

    out = np.empty((2, 4), dtype=float)
    out[0] = [
        -6.0 * t2 * cp * math.sin(a) * math.cos(b),
        t1 * (-math.sin(kx) - math.sin(0.5 * kx) * math.cos(b)),
        t1 * (-math.cos(kx) + math.cos(0.5 * kx) * math.cos(b)),
        -6.0 * t2 * sp * math.sin(a) * math.sin(b),
    ]
    out[1] = [
        2.0 * t2 * cp * (-s3 * math.sin(s3 * ky) - s3 * math.cos(a) * math.sin(b)),
        -t1 * s3 * math.cos(0.5 * kx) * math.sin(b),
        -t1 * s3 * math.sin(0.5 * kx) * math.sin(b),
        2.0 * t2 * sp * (s3 * math.cos(a) * math.cos(b) - s3 * math.cos(s3 * ky)),
    ]
    return out


def dhdk(k_xy: np.ndarray | tuple[float, float], params: HaldaneParams = HaldaneParams()) -> tuple[np.ndarray, np.ndarray]:
    """Analytic first derivatives ``partial_k H``."""

    coeffs = _first_derivatives_coefficients(np.asarray(k_xy, dtype=float), params)
    return tuple(_matrix_from_coefficients(*coeffs[axis]) for axis in range(2))  # type: ignore[return-value]


def _second_derivatives_coefficients(k_xy: np.ndarray, params: HaldaneParams) -> np.ndarray:
    kx, ky = np.asarray(k_xy, dtype=float).reshape(2)
    s3 = math.sqrt(3.0)
    a = 1.5 * kx
    b = 0.5 * s3 * ky
    t1 = float(params.t1)
    t2 = float(params.t2)
    cp = math.cos(float(params.phi))
    sp = math.sin(float(params.phi))

    out = np.empty((2, 2, 4), dtype=float)
    out[0, 0] = [
        -9.0 * t2 * cp * math.cos(a) * math.cos(b),
        t1 * (-math.cos(kx) - 0.5 * math.cos(0.5 * kx) * math.cos(b)),
        t1 * (math.sin(kx) - 0.5 * math.sin(0.5 * kx) * math.cos(b)),
        -9.0 * t2 * sp * math.cos(a) * math.sin(b),
    ]
    out[1, 1] = [
        2.0 * t2 * cp * (-3.0 * math.cos(s3 * ky) - 1.5 * math.cos(a) * math.cos(b)),
        -1.5 * t1 * math.cos(0.5 * kx) * math.cos(b),
        -1.5 * t1 * math.sin(0.5 * kx) * math.cos(b),
        2.0 * t2 * sp * (-1.5 * math.cos(a) * math.sin(b) + 3.0 * math.sin(s3 * ky)),
    ]
    out[0, 1] = [
        3.0 * s3 * t2 * cp * math.sin(a) * math.sin(b),
        0.5 * s3 * t1 * math.sin(0.5 * kx) * math.sin(b),
        -0.5 * s3 * t1 * math.cos(0.5 * kx) * math.sin(b),
        -3.0 * s3 * t2 * sp * math.sin(a) * math.cos(b),
    ]
    out[1, 0] = out[0, 1]
    return out


def d2hdk(k_xy: np.ndarray | tuple[float, float], params: HaldaneParams = HaldaneParams()) -> np.ndarray:
    """Analytic second derivatives ``partial_a partial_b H``."""

    coeffs = _second_derivatives_coefficients(np.asarray(k_xy, dtype=float), params)
    out = np.empty((2, 2, 2, 2), dtype=np.complex128)
    for a in range(2):
        for b in range(2):
            out[a, b] = _matrix_from_coefficients(*coeffs[a, b])
    return out


def diagonalize(k_xy: np.ndarray | tuple[float, float], params: HaldaneParams = HaldaneParams()) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(hamiltonian(k_xy, params))
    return np.asarray(evals, dtype=float), np.asarray(evecs, dtype=np.complex128)


def transition_gap_at_points(params: HaldaneParams = HaldaneParams()) -> dict[str, float]:
    """Return direct interband gaps at standard high-symmetry points."""

    gaps: dict[str, float] = {}
    for name, k_xy in high_symmetry_points().items():
        evals, _ = diagonalize(k_xy, params)
        gaps[name] = float(evals[1] - evals[0])
    return gaps


def c3_rotation_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rotations by ``0, 2*pi/3, 4*pi/3`` in lab ``kx,ky`` axes."""

    mats = []
    for multiple in (0, 1, 2):
        theta = 2.0 * math.pi * multiple / 3.0
        c, s = math.cos(theta), math.sin(theta)
        mats.append(np.asarray([[c, -s], [s, c]], dtype=float))
    return tuple(mats)  # type: ignore[return-value]


def c3_symmetrize_points(k_points: np.ndarray, k_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average a finite BZ quadrature over C3-related sample points.

    This is a quadrature improvement, not post-processing smoothing.  Each
    rotated copy carries one third of the original weight; the Hamiltonian is
    evaluated at the rotated k point and tensor components are still measured in
    the fixed lab axes.
    """

    points = np.asarray(k_points, dtype=float).reshape(-1, 2)
    weights = np.asarray(k_weights, dtype=float).reshape(-1)
    rotations = c3_rotation_matrices()
    rotated = np.concatenate([points @ rot.T for rot in rotations], axis=0)
    rotated_weights = np.tile(weights / float(len(rotations)), len(rotations))
    return rotated, rotated_weights


def parallelogram_bz_grid(
    mesh_size: int,
    *,
    offset: float = 0.5,
    c3_symmetrize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform midpoint grid over the primitive reciprocal parallelogram."""

    n = int(mesh_size)
    if n <= 1:
        raise ValueError(f"mesh_size must be > 1, got {mesh_size}")
    b1, b2 = reciprocal_vectors()
    us = (np.arange(n, dtype=float) + float(offset)) / float(n)
    vs = (np.arange(n, dtype=float) + float(offset)) / float(n)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    points = uu.ravel()[:, None] * b1[None, :] + vv.ravel()[:, None] * b2[None, :]
    weights = np.full(points.shape[0], bz_area() / float(points.shape[0]), dtype=float)
    if c3_symmetrize:
        points, weights = c3_symmetrize_points(points, weights)
    return points, weights


def compute_haldane_injection_spectra(
    photon_energies: np.ndarray,
    *,
    components: Iterable[str | Component] = ("y;yy",),
    mesh_size: int = 41,
    eta: float = 0.04,
    params: HaldaneParams = HaldaneParams(),
    mu: float = 0.0,
    denominator_cutoff: float = 1.0e-10,
    prefactor: complex = -2.0 * math.pi,
    relaxation_time: float = 1.0,
    c3_symmetrize_grid: bool = True,
) -> dict[str, np.ndarray]:
    """Compute raw Haldane injection spectra on the full primitive BZ.

    This is a lightweight checkpoint helper.  Broad production-quality sweeps
    should still run through Slurm and include mesh/eta convergence tables.
    """

    omega = np.asarray(photon_energies, dtype=float)
    parsed = {
        component_label(c) if not isinstance(c, str) else parse_component(c).semicolon_label: c
        for c in components
    }
    integrals = {name: np.zeros_like(omega, dtype=np.complex128) for name in parsed}
    k_points, k_weights = parallelogram_bz_grid(mesh_size, c3_symmetrize=c3_symmetrize_grid)
    for k_xy, k_weight in zip(k_points, k_weights, strict=True):
        evals, evecs = diagonalize(k_xy, params)
        tensors = precompute_injection_current_tensors(
            evals,
            evecs,
            dhdk(k_xy, params),
            mu_ev=mu,
            denominator_cutoff_ev=denominator_cutoff,
        )
        for name, component in parsed.items():
            transitions, transition_weights = positive_injection_transition_terms(tensors, component)
            integrals[name] += accumulate_injection_spectrum(
                omega,
                transitions,
                transition_weights,
                k_weight=float(k_weight),
                eta_ev=float(eta),
                prefactor=prefactor,
                relaxation_time=relaxation_time,
            )
    return integrals


def haldane_injection_tensors_at_k(
    k_xy: np.ndarray | tuple[float, float],
    params: HaldaneParams = HaldaneParams(),
    *,
    mu: float = 0.0,
    denominator_cutoff: float = 1.0e-10,
):
    """Return injection-current tensors at one Haldane-model k point."""

    evals, evecs = diagonalize(k_xy, params)
    return precompute_injection_current_tensors(
        evals,
        evecs,
        dhdk(k_xy, params),
        mu_ev=mu,
        denominator_cutoff_ev=denominator_cutoff,
    )


def haldane_shift_tensors_at_k(
    k_xy: np.ndarray | tuple[float, float],
    params: HaldaneParams = HaldaneParams(),
    *,
    mu: float = 0.0,
    denominator_cutoff: float = 1.0e-10,
):
    """Return generic shift-current tensors at one Haldane-model k point."""

    evals, evecs = diagonalize(k_xy, params)
    return precompute_shift_current_tensors(
        evals,
        evecs,
        dhdk(k_xy, params),
        d2hdk=d2hdk(k_xy, params),
        mu_ev=mu,
        denominator_cutoff_ev=denominator_cutoff,
    )


def quantum_metric_two_band(
    k_xy: np.ndarray | tuple[float, float],
    *,
    axis_a: int = 1,
    axis_b: int = 1,
    params: HaldaneParams = HaldaneParams(),
    denominator_cutoff: float = 1.0e-10,
) -> float:
    """Two-band quantum metric component ``Re[r^a_01 r^b_10]``."""

    tensors = haldane_injection_tensors_at_k(k_xy, params, denominator_cutoff=denominator_cutoff)
    r = tensors.berry_connection
    return float(np.real(r[axis_a, 0, 1] * r[axis_b, 1, 0]))


def injection_kernel_at_k(
    k_xy: np.ndarray | tuple[float, float],
    component: str | Component = "y;yy",
    *,
    params: HaldaneParams = HaldaneParams(),
    denominator_cutoff: float = 1.0e-10,
) -> complex:
    """Occupation-independent two-band injection kernel at one k point."""

    tensors = haldane_injection_tensors_at_k(k_xy, params, denominator_cutoff=denominator_cutoff)
    return injection_component_kernel(tensors, 0, 1, component)


def symplectic_connection_imag_at_k(
    k_xy: np.ndarray | tuple[float, float],
    *,
    current_axis: int = 0,
    optical_axis: int = 0,
    params: HaldaneParams = HaldaneParams(),
    denominator_cutoff: float = 1.0e-10,
) -> float:
    """Return ``Im[C_12^{optical,current,optical}]`` from the shared derivative module.

    This is the WannierBerri-style generalized-derivative route used by the
    generic shift-current code: ``C = A^a_{21} (A^a_{12})_;c`` for the lower to
    upper two-band transition.  No raw phase derivative is used.
    """

    tensors = haldane_shift_tensors_at_k(k_xy, params, denominator_cutoff=denominator_cutoff)
    A = tensors.berry_connection
    G = tensors.berry_connection_gen_derivative
    n, m = 0, 1
    return float(np.imag(A[int(optical_axis), m, n] * G[int(current_axis), int(optical_axis), n, m]))


def symplectic_connection_minus_im_at_k(
    k_xy: np.ndarray | tuple[float, float],
    *,
    current_axis: int = 0,
    optical_axis: int = 0,
    params: HaldaneParams = HaldaneParams(),
    denominator_cutoff: float = 1.0e-10,
) -> float:
    """Return the Fig. 6-style ``-Im[C_12]`` symplectic connection."""

    return -symplectic_connection_imag_at_k(
        k_xy,
        current_axis=current_axis,
        optical_axis=optical_axis,
        params=params,
        denominator_cutoff=denominator_cutoff,
    )


def shift_kernel_at_k(
    k_xy: np.ndarray | tuple[float, float],
    component: str | Component = "x;xx",
    *,
    params: HaldaneParams = HaldaneParams(),
    mu: float = 0.0,
    lower_band_filled: bool = False,
    denominator_cutoff: float = 1.0e-10,
) -> float:
    """Generic shift-current kernel at one k point.

    By default this uses the Fermi occupation from ``mu`` to match spectrum
    calculations.  Lin--Hsu Fig. 7/8 define zone-integrated two-band responses
    with the lower band fully occupied; use ``lower_band_filled=True`` for that
    flat-band-occupation convention, especially when the scalar ``f0(k)`` term
    moves parts of the lower band above ``mu=0``.
    """

    tensors = haldane_shift_tensors_at_k(k_xy, params, mu=mu, denominator_cutoff=denominator_cutoff)
    if lower_band_filled:
        if tensors.energies_ev.size != 2:
            raise ValueError("lower_band_filled=True is only implemented for the two-band Haldane toy model")
        tensors = replace(tensors, occupations=np.asarray([1.0, 0.0], dtype=float))
    transitions, weights = positive_shift_transition_terms(tensors, component)
    if transitions.size == 0:
        return 0.0
    return float(np.real(-1.0j * weights[0]))


def compute_haldane_shift_spectra(
    photon_energies: np.ndarray,
    *,
    components: Iterable[str | Component] = ("x;xx",),
    mesh_size: int = 41,
    eta: float = 0.04,
    params: HaldaneParams = HaldaneParams(),
    mu: float = 0.0,
    denominator_cutoff: float = 1.0e-10,
    prefactor: complex = -1.0j * math.pi,
    c3_symmetrize_grid: bool = True,
) -> dict[str, np.ndarray]:
    """Compute raw generic shift-current spectra for the Haldane checkpoint.

    The default prefactor is the toy-unit Lin--Hsu factor ``-i*pi``.  This helper
    is intended for Checkpoint A7 consistency with the shared shift engine.
    """

    omega = np.asarray(photon_energies, dtype=float)
    parsed = {
        component_label(c) if not isinstance(c, str) else parse_component(c).semicolon_label: c
        for c in components
    }
    integrals = {name: np.zeros_like(omega, dtype=np.complex128) for name in parsed}
    k_points, k_weights = parallelogram_bz_grid(mesh_size, c3_symmetrize=c3_symmetrize_grid)
    for k_xy, k_weight in zip(k_points, k_weights, strict=True):
        tensors = haldane_shift_tensors_at_k(k_xy, params, mu=mu, denominator_cutoff=denominator_cutoff)
        for name, component in parsed.items():
            transitions, transition_weights = positive_shift_transition_terms(tensors, component)
            add_transitions_to_integral(
                integrals[name],
                omega,
                transitions,
                transition_weights,
                k_weight_nm_inv_sq=float(k_weight),
                eta_ev=float(eta),
            )
    return {name: complex(prefactor) * values for name, values in integrals.items()}


def all_2d_components() -> tuple[str, ...]:
    """Return all rank-3 2D tensor component labels in ``c;ab`` form."""

    axes = ("x", "y")
    return tuple(f"{c};{a}{b}" for c in axes for a in axes for b in axes)


__all__ = [
    "HaldaneParams",
    "bravais_vectors",
    "bz_area",
    "c3_rotation_matrices",
    "c3_symmetrize_points",
    "all_2d_components",
    "compute_haldane_injection_spectra",
    "compute_haldane_shift_spectra",
    "d2hdk",
    "dhdk",
    "diagonalize",
    "haldane_injection_tensors_at_k",
    "haldane_shift_tensors_at_k",
    "hamiltonian",
    "high_symmetry_points",
    "injection_kernel_at_k",
    "parallelogram_bz_grid",
    "quantum_metric_two_band",
    "reciprocal_vectors",
    "shift_kernel_at_k",
    "symplectic_connection_imag_at_k",
    "symplectic_connection_minus_im_at_k",
    "transition_gap_at_points",
    "unit_cell_area",
]
