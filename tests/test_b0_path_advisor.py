from __future__ import annotations

import numpy as np

from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field import (
    build_m_k_gamma_m_candidate_paths,
    equivalent_m_point_candidates,
    moire_bz_vertices,
    rank_kpath_candidates_for_lk,
    recommend_lk_values_for_path_family,
)


def test_equivalent_m_points_and_candidate_triangles_stay_in_sampled_cell() -> None:
    params = TBGParameters.from_degrees(1.2)
    basis = np.array(
        [
            [params.g1.real, params.g2.real],
            [params.g1.imag, params.g2.imag],
        ],
        dtype=float,
    )

    m_points = equivalent_m_point_candidates(params, adjacent_only=True)
    candidates = build_m_k_gamma_m_candidate_paths(params, adjacent_only=True)

    assert len(m_points) == 3
    assert len(candidates) == 4
    for candidate in candidates:
        assert candidate.m_point in m_points
        coeff = np.linalg.solve(
            basis,
            np.array([candidate.k_point.real, candidate.k_point.imag], dtype=float),
        )
        assert -1.0e-12 <= coeff[0] <= 1.0 + 1.0e-12
        assert -1.0e-12 <= coeff[1] <= 1.0 + 1.0e-12


def test_moire_bz_vertices_form_hexagon() -> None:
    params = TBGParameters.from_degrees(1.2)

    vertices = moire_bz_vertices(params)

    assert len(vertices) == 6
    assert all(abs(vertices[i]) > 0.0 for i in range(6))
    assert abs(sum(vertices)) < 1.0e-12


def test_rank_kpath_candidates_for_lk_hits_all_three_triangle_edges() -> None:
    params = TBGParameters.from_degrees(1.2)

    ranked = rank_kpath_candidates_for_lk(params, lk=19, adjacent_only=True)

    assert len(ranked) == 4
    assert all(item.exact_count == 19 for item in ranked)
    assert all(item.exact_segment_counts == (3, 6, 10) for item in ranked)


def test_recommend_lk_values_for_path_family_tracks_denser_meshes() -> None:
    params = TBGParameters.from_degrees(1.2)

    recommendations = recommend_lk_values_for_path_family(params, [19, 23, 32], adjacent_only=True)

    assert [item.lk for item in recommendations] == [32, 23, 19]
    assert [item.exact_count for item in recommendations] == [32, 23, 19]
