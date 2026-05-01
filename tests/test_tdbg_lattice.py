from __future__ import annotations

import math

import numpy as np

from mean_field.systems.tdbg import build_standard_kpath, build_tdbg_lattice


def test_tdbg_lattice_matches_pytwist_reference_basis_size_and_neighbor_count() -> None:
    lattice = build_tdbg_lattice(1.33, cut=4.0)

    assert np.allclose(np.sum(lattice.q_vectors, axis=0), 0.0, atol=1.0e-12)
    assert lattice.n_q == 115
    assert lattice.matrix_dim == 460
    assert sum(len(neighbors) for neighbors in lattice.q_neighbors) == 156


def test_tdbg_standard_path_uses_reference_k_gamma_m_kprime_sampling() -> None:
    lattice = build_tdbg_lattice(1.33, cut=4.0)
    path = build_standard_kpath(lattice, resolution=16)

    assert path.labels == ("K", "Gamma", "M", "Kprime")
    assert path.node_indices == (1, 16, 29, 37)
    assert path.kvec.shape == (37,)
    assert abs(path.kvec[0] - lattice.k_m) < 1.0e-12
    assert abs(path.kvec[path.node_indices[1] - 1] - lattice.gamma_m) < 1.0e-12
    assert abs(path.kvec[path.node_indices[2] - 1] - lattice.m_m) < 1.0e-12
    assert abs(path.kvec[path.node_indices[3] - 1] - lattice.kprime_m) < 1.0e-12

    assert math.isclose(float(path.kdist[-1]), float(np.sum(np.abs(np.diff(path.kvec)))), rel_tol=0.0, abs_tol=1.0e-12)
