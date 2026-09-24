from __future__ import annotations

import math

import numpy as np

from mean_field.systems.RnG_hBN.params import (
    MOIRE_PARAMETER_TABLE,
    table_ii_moire_parameters,
)
from mean_field.systems.RnG_hBN import (
    RLGhBNParams,
)
from mean_field.systems.RnG_hBN.lattice import (
    build_rlg_hbn_lattice,
    build_standard_kpath,
)


def test_rlg_hbn_table_ii_parameters_are_available_for_all_layers_and_stackings() -> None:
    assert set(MOIRE_PARAMETER_TABLE) == {
        (3, 0),
        (3, 1),
        (4, 0),
        (4, 1),
        (5, 0),
        (5, 1),
        (6, 0),
        (6, 1),
        (7, 0),
        (7, 1),
    }

    params = RLGhBNParams.from_table(layer_count=5, xi=0)
    assert params.fermi_velocity_mev_nm == 542.1
    assert params.v3_mev_nm == 34.0
    assert params.v4_mev_nm == 34.0
    assert params.t1_mev == 355.16
    assert params.t2_mev == -7.0
    assert table_ii_moire_parameters(5, 0) == (7.19, 7.49, -136.55)
    assert table_ii_moire_parameters(5, 1) == (1.50, 7.37, 16.55)


def test_rlg_hbn_lattice_default_shell_matches_work_document_ng_19() -> None:
    lattice = build_rlg_hbn_lattice(theta_deg=0.77, shell_count=4, layer_count=5)

    q_norms = np.linalg.norm(lattice.q_vectors, axis=1)
    g_norms = np.linalg.norm(lattice.g_vectors_basis, axis=1)
    assert lattice.n_g == 19
    assert lattice.matrix_dim == 190
    assert np.allclose(q_norms, q_norms[0], atol=1.0e-12)
    assert np.allclose(g_norms, g_norms[0], atol=1.0e-12)
    assert abs(lattice.g_m3 + lattice.g_m1 + lattice.g_m2) < 1.0e-12
    assert 11.0 < lattice.moire_period_nm < 13.0


def test_rlg_hbn_standard_kpath_uses_k_gamma_m_kprime_nodes() -> None:
    lattice = build_rlg_hbn_lattice(theta_deg=0.77, shell_count=4, layer_count=5)
    path = build_standard_kpath(lattice, points_per_segment=6)

    assert path.labels == ("K", "Gamma", "M", "Kprime")
    assert path.node_indices == (1, 7, 13, 19)
    assert path.kvec.shape == (19,)
    assert abs(path.kvec[0] - lattice.k_m) < 1.0e-12
    assert abs(path.kvec[path.node_indices[1] - 1] - lattice.gamma_m) < 1.0e-12
    assert abs(path.kvec[path.node_indices[2] - 1] - lattice.m_m) < 1.0e-12
    assert abs(path.kvec[path.node_indices[3] - 1] - lattice.kprime_m) < 1.0e-12
    assert abs(lattice.k_m - (2.0 * lattice.g_m1 + lattice.g_m2) / 3.0) < 1.0e-12
    assert abs(lattice.kprime_m - (lattice.g_m1 + 2.0 * lattice.g_m2) / 3.0) < 1.0e-12
    assert abs(lattice.m_m - (lattice.g_m1 + lattice.g_m2) / 2.0) < 1.0e-12
    assert abs(abs(lattice.k_m) - abs(lattice.g_m1) / math.sqrt(3.0)) < 1.0e-12
    assert abs(abs(lattice.kprime_m) - abs(lattice.g_m1) / math.sqrt(3.0)) < 1.0e-12
    assert abs(abs(lattice.m_m) - abs(lattice.g_m1) / 2.0) < 1.0e-12
    assert abs(abs(lattice.k_m - lattice.m_m) - abs(lattice.g_m1) / math.sqrt(12.0)) < 1.0e-12
    assert math.isclose(float(path.kdist[-1]), float(np.sum(np.abs(np.diff(path.kvec)))), rel_tol=0.0, abs_tol=1.0e-12)
