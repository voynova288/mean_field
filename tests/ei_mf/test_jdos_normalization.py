from __future__ import annotations

import numpy as np

from ei_mf.grids import radial_grid
from ei_mf.observables import (
    compute_jdos,
    compute_radial_jdos,
    radial_jdos_refinement_factor,
)


def test_jdos_is_non_negative_and_peak_center_stable_with_broadening() -> None:
    k, w, _edges = radial_grid(0.08, 80)
    E0 = 2.0
    E_pair = E0 + 200.0 * (k - 0.035) ** 2
    omega1, j1 = compute_jdos(E_pair, w, 0.03, omega_min_mev=1.5, omega_max_mev=3.0, nomega=400)
    omega2, j2 = compute_jdos(E_pair, w, 0.12, omega_min_mev=1.5, omega_max_mev=3.0, nomega=400)
    assert np.all(j1 >= 0.0)
    assert np.all(j2 >= 0.0)
    p1 = omega1[int(np.argmax(j1))]
    p2 = omega2[int(np.argmax(j2))]
    assert abs(p1 - p2) < 0.08


def test_radial_jdos_removes_point_quadrature_comb_without_extra_smoothing() -> None:
    k, weights, _edges = radial_grid(0.12, 60)
    E_pair = 3.7 + 30000.0 * (k**2 - 0.05**2) ** 2
    omega, point = compute_jdos(
        E_pair, weights, 0.05, omega_min_mev=3.0, omega_max_mev=10.0, nomega=1600
    )
    boundary_spectra = []
    for boundary in ("even_quadratic", "clamped", "linear"):
        omega_refined, boundary_refined = compute_radial_jdos(
            k,
            E_pair,
            weights,
            0.05,
            omega_min_mev=3.0,
            omega_max_mev=10.0,
            nomega=1600,
            origin_boundary=boundary,
        )
        assert np.array_equal(omega, omega_refined)
        boundary_spectra.append(boundary_refined)
    refined = boundary_spectra[0]
    tail = omega > 4.0
    point_maxima = np.sum(
        (point[1:-1] > point[:-2]) & (point[1:-1] > point[2:]) & tail[1:-1]
    )
    refined_maxima = np.sum(
        (refined[1:-1] > refined[:-2])
        & (refined[1:-1] > refined[2:])
        & tail[1:-1]
    )
    assert point_maxima > 10
    assert refined_maxima == 0
    for boundary_refined in boundary_spectra[1:]:
        boundary_maxima = np.sum(
            (boundary_refined[1:-1] > boundary_refined[:-2])
            & (boundary_refined[1:-1] > boundary_refined[2:])
            & tail[1:-1]
        )
        assert boundary_maxima == 0
    assert np.isclose(
        np.trapezoid(point, omega), np.trapezoid(refined, omega), rtol=2e-6
    )


def test_radial_jdos_refinement_converges_across_input_meshes() -> None:
    spectra = []
    factors = []
    for nk in (40, 90, 180):
        k, weights, _edges = radial_grid(0.12, nk)
        E_pair = 3.7 + 30000.0 * (k**2 - 0.05**2) ** 2
        factors.append(radial_jdos_refinement_factor(E_pair, 0.05))
        omega, spectrum = compute_radial_jdos(
            k,
            E_pair,
            weights,
            0.05,
            omega_min_mev=3.0,
            omega_max_mev=10.0,
            nomega=1600,
        )
        spectra.append(spectrum)
    assert factors[0] > factors[1] > factors[2]
    rel_rms = np.sqrt(np.mean((spectra[0] - spectra[2]) ** 2)) / np.max(spectra[2])
    assert rel_rms < 1e-3
