from __future__ import annotations

import math

import numpy as np

from mean_field.systems.tmbg import build_standard_kpath, build_tmbg_lattice


def test_tmbg_lattice_matches_basic_moire_geometry_identities() -> None:
    lattice = build_tmbg_lattice(1.05, n_shells=5)

    assert math.isclose(abs(lattice.q0), abs(lattice.q_plus), rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(abs(lattice.q0), abs(lattice.q_minus), rel_tol=0.0, abs_tol=1.0e-12)
    assert abs(lattice.q0 + lattice.q_plus + lattice.q_minus) < 1.0e-12

    assert math.isclose(abs(lattice.g_m1), abs(lattice.g_m2), rel_tol=0.0, abs_tol=1.0e-12)
    angle = abs(np.angle(lattice.g_m2 / lattice.g_m1))
    assert math.isclose(angle, math.pi / 3.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(lattice.l_m, 13.4, rel_tol=0.0, abs_tol=0.1)

    assert np.any(np.abs(lattice.g_vectors) < 1.0e-12)
    unique_points = {(round(float(value.real), 12), round(float(value.imag), 12)) for value in lattice.g_vectors}
    assert len(unique_points) == lattice.n_g
    assert np.all(np.diff(np.abs(lattice.g_vectors)) >= -1.0e-12)


def test_tmbg_standard_path_uses_k_gamma_m_kprime_order() -> None:
    lattice = build_tmbg_lattice(1.21, n_shells=1)
    path = build_standard_kpath(lattice, points_per_segment=8)

    assert path.labels == ("K", "Gamma", "M", "Kprime")
    assert path.node_indices == (1, 9, 17, 25)
    assert np.isclose(path.kvec[0], lattice.k_m)
    assert np.isclose(path.kvec[path.node_indices[1] - 1], lattice.gamma_m)
    assert np.isclose(path.kvec[path.node_indices[2] - 1], lattice.m_m)
    assert np.isclose(path.kvec[path.node_indices[3] - 1], lattice.kprime_m)
