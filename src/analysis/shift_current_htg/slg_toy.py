from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np
from scipy.linalg import eigh

from .constants import CARBON_BOND_NM
from .response import (
    Component,
    add_transitions_to_integral,
    component_label,
    parse_component,
    positive_transition_terms,
    precompute_response_tensors,
    sigma_from_integral,
)


@dataclass(frozen=True)
class GappedSLGParams:
    """Nearest-neighbor gapped graphene toy model from Mao et al. Appendix A."""

    hopping_ev: float = 2.73
    mass_ev: float = 1.5
    bond_nm: float = CARBON_BOND_NM


def nn_vectors(params: GappedSLGParams = GappedSLGParams()) -> np.ndarray:
    """Nearest-neighbor vectors d_j in nm, matching Appendix A."""

    d = float(params.bond_nm)
    vectors = []
    for j in range(3):
        angle = 2.0 * math.pi * j / 3.0
        vectors.append((-d * math.sin(angle), d * math.cos(angle)))
    return np.asarray(vectors, dtype=float)


def bravais_vectors(params: GappedSLGParams = GappedSLGParams()) -> tuple[np.ndarray, np.ndarray]:
    """Triangular A-sublattice primitive vectors in nm."""

    dvec = nn_vectors(params)
    a1 = dvec[0] - dvec[1]
    a2 = dvec[0] - dvec[2]
    return a1, a2


def reciprocal_vectors(params: GappedSLGParams = GappedSLGParams()) -> tuple[np.ndarray, np.ndarray]:
    """Reciprocal primitive vectors b_i with a_i dot b_j = 2*pi delta_ij."""

    a1, a2 = bravais_vectors(params)
    amat = np.column_stack([a1, a2])
    bmat = 2.0 * np.pi * np.linalg.inv(amat).T
    return bmat[:, 0], bmat[:, 1]


def unit_cell_area_nm2(params: GappedSLGParams = GappedSLGParams()) -> float:
    a1, a2 = bravais_vectors(params)
    return abs(float(a1[0] * a2[1] - a1[1] * a2[0]))


def bz_area_nm_inv_sq(params: GappedSLGParams = GappedSLGParams()) -> float:
    return (2.0 * np.pi) ** 2 / unit_cell_area_nm2(params)


def hamiltonian(k_xy_nm_inv: np.ndarray | tuple[float, float], params: GappedSLGParams = GappedSLGParams()) -> np.ndarray:
    """Bloch Hamiltonian H(k) in the embedded A/B basis.

    H_AB(k) = -t sum_j exp(i k.d_j), H_BA = H_AB^*, and the mass term is
    +m on A and -m on B.  This is a two-band finite-BZ model, not a continuum
    massive Dirac approximation.
    """

    k = np.asarray(k_xy_nm_inv, dtype=float).reshape(2)
    dvec = nn_vectors(params)
    phase = np.exp(1.0j * (dvec @ k))
    f = np.sum(phase)
    off = -float(params.hopping_ev) * f
    return np.asarray([[params.mass_ev, off], [off.conjugate(), -params.mass_ev]], dtype=np.complex128)


def dhdk(k_xy_nm_inv: np.ndarray | tuple[float, float], params: GappedSLGParams = GappedSLGParams()) -> tuple[np.ndarray, np.ndarray]:
    """Analytic partial_k H matrices in eV*nm."""

    k = np.asarray(k_xy_nm_inv, dtype=float).reshape(2)
    dvec = nn_vectors(params)
    phase = np.exp(1.0j * (dvec @ k))
    derivs: list[np.ndarray] = []
    for axis in range(2):
        df = np.sum(1.0j * dvec[:, axis] * phase)
        doff = -float(params.hopping_ev) * df
        derivs.append(np.asarray([[0.0, doff], [doff.conjugate(), 0.0]], dtype=np.complex128))
    return derivs[0], derivs[1]


def d2hdk(k_xy_nm_inv: np.ndarray | tuple[float, float], params: GappedSLGParams = GappedSLGParams()) -> np.ndarray:
    """Analytic second partial derivatives of H in eV*nm^2.

    The hTTG continuum Dirac Hamiltonian is linear in k, so Mao Eq. (4) has no
    second-derivative term.  This tight-binding benchmark is nonlinear in k and
    therefore requires W^{ab}_{nm}=<u_n|partial_a partial_b H|u_m> in the
    generalized derivative.
    """

    k = np.asarray(k_xy_nm_inv, dtype=float).reshape(2)
    dvec = nn_vectors(params)
    phase = np.exp(1.0j * (dvec @ k))
    out = np.empty((2, 2, 2, 2), dtype=np.complex128)
    for axis_a in range(2):
        for axis_b in range(2):
            d2f = np.sum(-dvec[:, axis_a] * dvec[:, axis_b] * phase)
            d2off = -float(params.hopping_ev) * d2f
            out[axis_a, axis_b] = np.asarray(
                [[0.0, d2off], [d2off.conjugate(), 0.0]],
                dtype=np.complex128,
            )
    return out


def diagonalize(k_xy_nm_inv: np.ndarray | tuple[float, float], params: GappedSLGParams = GappedSLGParams()) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = eigh(hamiltonian(k_xy_nm_inv, params))
    return np.asarray(evals, dtype=float), np.asarray(evecs, dtype=np.complex128)


def _nearest_reciprocal_vectors(params: GappedSLGParams) -> np.ndarray:
    b1, b2 = reciprocal_vectors(params)
    candidates: list[np.ndarray] = []
    for n1 in range(-1, 2):
        for n2 in range(-1, 2):
            if n1 == 0 and n2 == 0:
                continue
            candidates.append(n1 * b1 + n2 * b2)
    candidates.sort(key=lambda vec: float(np.dot(vec, vec)))
    nearest = candidates[:6]
    nearest.sort(key=lambda vec: math.atan2(float(vec[1]), float(vec[0])))
    return np.asarray(nearest, dtype=float)


def hex_bz_vertices(params: GappedSLGParams = GappedSLGParams()) -> np.ndarray:
    """Vertices of the Wigner-Seitz hexagonal Brillouin zone."""

    gs = _nearest_reciprocal_vectors(params)
    vertices: list[np.ndarray] = []
    for i in range(gs.shape[0]):
        for j in range(i + 1, gs.shape[0]):
            mat = np.vstack([gs[i], gs[j]])
            det = float(np.linalg.det(mat))
            if abs(det) < 1.0e-12:
                continue
            rhs = np.asarray([0.5 * np.dot(gs[i], gs[i]), 0.5 * np.dot(gs[j], gs[j])], dtype=float)
            point = np.linalg.solve(mat, rhs)
            if np.all(gs @ point <= 0.5 * np.sum(gs * gs, axis=1) + 1.0e-9):
                if not any(np.linalg.norm(point - old) < 1.0e-8 for old in vertices):
                    vertices.append(point)
    if len(vertices) != 6:
        raise RuntimeError(f"Expected 6 BZ vertices, found {len(vertices)}")
    vertices.sort(key=lambda vec: math.atan2(float(vec[1]), float(vec[0])))
    return np.asarray(vertices, dtype=float)


def hex_bz_grid(
    mesh_size: int,
    params: GappedSLGParams = GappedSLGParams(),
) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint grid clipped to the full hexagonal BZ.

    Returns ``(k_points, weights)`` with k in nm^{-1} and weights in nm^{-2}.
    The equal weights are normalized to the exact primitive-cell BZ area.
    This is intended for first validation and convergence checks; production
    figures should compare multiple ``mesh_size`` values.
    """

    if int(mesh_size) <= 1:
        raise ValueError(f"mesh_size must be > 1, got {mesh_size}")
    vertices = hex_bz_vertices(params)
    xmin, ymin = np.min(vertices, axis=0)
    xmax, ymax = np.max(vertices, axis=0)
    xs = xmin + (np.arange(mesh_size, dtype=float) + 0.5) * (xmax - xmin) / float(mesh_size)
    ys = ymin + (np.arange(mesh_size, dtype=float) + 0.5) * (ymax - ymin) / float(mesh_size)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    gs = _nearest_reciprocal_vectors(params)
    bounds = 0.5 * np.sum(gs * gs, axis=1)
    mask = np.all(points @ gs.T <= bounds[None, :] + 1.0e-12, axis=1)
    selected = points[mask]
    if selected.size == 0:
        raise RuntimeError("Hexagonal BZ grid mask selected no points")
    weights = np.full(selected.shape[0], bz_area_nm_inv_sq(params) / float(selected.shape[0]), dtype=float)
    return selected, weights


def compute_slg_shift_current(
    photon_energies_ev: np.ndarray,
    *,
    components: Iterable[str | Component] = ("x;yy", "y;xx"),
    mesh_size: int = 60,
    eta_ev: float = 0.02,
    params: GappedSLGParams = GappedSLGParams(),
    denominator_cutoff_ev: float = 1.0e-10,
    c3_symmetrize_grid: bool = True,
) -> dict[str, np.ndarray]:
    """Compute toy-model shift-current spectra on the full hexagonal BZ.

    The rectangular midpoint grid clipped to a hexagon is not exactly C3
    symmetric at finite resolution.  Averaging over C3-related k points is a
    numerical integration improvement, not an irreducible-BZ reduction.
    """

    photon_energies = np.asarray(photon_energies_ev, dtype=float)
    parsed: dict[str, Component] = {}
    for component in components:
        if isinstance(component, str):
            parsed[component] = parse_component(component)
        else:
            parsed[component_label(component)] = component

    integrals = {name: np.zeros_like(photon_energies, dtype=np.complex128) for name in parsed}
    k_points, k_weights = hex_bz_grid(mesh_size, params)
    if bool(c3_symmetrize_grid):
        angle = 2.0 * math.pi / 3.0
        rotated_points: list[np.ndarray] = []
        rotated_weights: list[float] = []
        for k_xy, weight in zip(k_points, k_weights, strict=True):
            for multiple in (0, 1, 2):
                theta = multiple * angle
                c, s = math.cos(theta), math.sin(theta)
                rotated_points.append(np.asarray([c * k_xy[0] - s * k_xy[1], s * k_xy[0] + c * k_xy[1]], dtype=float))
                rotated_weights.append(float(weight) / 3.0)
        k_points = np.asarray(rotated_points, dtype=float)
        k_weights = np.asarray(rotated_weights, dtype=float)
    for k_xy, weight in zip(k_points, k_weights, strict=True):
        evals, evecs = diagonalize(k_xy, params)
        tensors = precompute_response_tensors(
            evals,
            evecs,
            dhdk(k_xy, params),
            d2hdk=d2hdk(k_xy, params),
            denominator_cutoff_ev=denominator_cutoff_ev,
        )
        for name, component in parsed.items():
            transitions, weights = positive_transition_terms(tensors, component)
            add_transitions_to_integral(
                integrals[name],
                photon_energies,
                transitions,
                weights,
                k_weight_nm_inv_sq=float(weight),
                eta_ev=eta_ev,
            )
    return {name: sigma_from_integral(integral) for name, integral in integrals.items()}


def c3_tensor_relation_errors(spectra: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Return max-abs errors for the C3 tensor identities in the work document."""

    required = {"x;yy", "x;xx", "y;yx", "y;xy", "y;xx", "y;yy", "x;xy", "x;yx"}
    missing = sorted(required.difference(spectra))
    if missing:
        raise ValueError(f"Missing spectra for C3 check: {missing}")
    arr = {key: np.asarray(value, dtype=float) for key, value in spectra.items()}
    group1_ref = arr["x;yy"]
    group2_ref = arr["y;xx"]
    return {
        "group1:max(|xyy + xxx|, |xyy - yyx|, |xyy - yxy|)": float(
            max(
                np.max(np.abs(group1_ref + arr["x;xx"])),
                np.max(np.abs(group1_ref - arr["y;yx"])),
                np.max(np.abs(group1_ref - arr["y;xy"])),
            )
        ),
        "group2:max(|yxx + yyy|, |yxx - xxy|, |yxx - xyx|)": float(
            max(
                np.max(np.abs(group2_ref + arr["y;yy"])),
                np.max(np.abs(group2_ref - arr["x;xy"])),
                np.max(np.abs(group2_ref - arr["x;yx"])),
            )
        ),
    }
