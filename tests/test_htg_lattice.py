from __future__ import annotations

import numpy as np

from mean_field.systems.htg import (
    HTGParams,
)
from mean_field.systems.htg.lattice import (
    build_htg_lattice,
    build_paper_hf_kpath,
)
from mean_field.systems.htg.params import (
    theta_deg_from_alpha,
)


def test_htg_lattice_geometry_anchors() -> None:
    lattice = build_htg_lattice(1.5, n_shells=2)
    assert np.allclose(np.sum(lattice.q_vectors), 0.0)
    assert np.allclose([abs(q) for q in lattice.q_vectors], lattice.k_theta)
    assert any(abs(gvec) < 1.0e-12 for gvec in lattice.g_vectors)
    assert lattice.matrix_dim == 6 * lattice.n_g
    assert np.isclose(lattice.l_m, 4.0 * np.pi / (np.sqrt(3.0) * abs(lattice.b_m1)))
    assert abs(lattice.kappa_m + lattice.q0) < 1.0e-12
    assert abs(lattice.kappa_prime_m - lattice.q0) < 1.0e-12


def test_htg_default_alpha_matches_paper_scale() -> None:
    lattice = build_htg_lattice(1.5, n_shells=1)
    params = HTGParams.default()
    assert np.isclose(params.vf_ev_nm, 0.678, rtol=1.0e-2)
    assert np.isclose(params.alpha(lattice.k_theta), 0.347437, rtol=1.0e-5)
    assert np.isclose(theta_deg_from_alpha(0.377, params=params), 1.382, rtol=1.0e-3)


def test_htg_kwan2023_params_match_hf_reference_scale() -> None:
    lattice = build_htg_lattice(1.7, n_shells=1)
    params = HTGParams.kwan2023(kappa=75.0 / 110.0)

    assert params.model_name == "kwan2023"
    assert params.fermi_velocity_m_per_s == 8.8e5
    assert np.isclose(params.w_ev, 0.110)
    assert np.isclose(params.kappa, 75.0 / 110.0)
    assert np.isclose(params.alpha(lattice.k_theta), 0.375907, rtol=1.0e-5)


def test_htg_paper_hf_path_returns_from_m_to_gamma() -> None:
    lattice = build_htg_lattice(1.8, n_shells=1)
    path = build_paper_hf_kpath(lattice, points_per_segment=3)

    assert path.labels == ("Gamma", "kappa", "kappa_prime", "Gamma", "M", "Gamma")
    assert np.isclose(path.kvec[0], lattice.gamma_m)
    assert np.isclose(path.kvec[-1], lattice.gamma_m + lattice.b_m1)
    assert np.isclose(path.kvec[path.node_indices[-2] - 1], lattice.m_m)
    assert np.isclose(
        path.kvec[-1] - path.kvec[path.node_indices[-2] - 1],
        path.kvec[path.node_indices[-2] - 1] - path.kvec[path.node_indices[-3] - 1],
    )
    assert path.kdist[-1] > path.kdist[-2]
