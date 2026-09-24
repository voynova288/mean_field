from __future__ import annotations

import math

from mean_field.reference_oracles import load_b0_parameter_references, load_b0_suite
from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field.model import (
    build_b0_uniform_lattice,
)
from mean_field.systems.tbg.zero_field.path import (
    build_fig6_kpath,
    build_gamma_m_k_gamma_kprime_kpath,
    select_adjacent_m_point,
)


def _assert_close_complex(lhs: complex, rhs: complex, *, atol: float = 1e-12) -> None:
    assert math.isclose(lhs.real, rhs.real, abs_tol=atol)
    assert math.isclose(lhs.imag, rhs.imag, abs_tol=atol)


def test_parameter_reference_matches_julia() -> None:
    for ref in load_b0_parameter_references():
        params = TBGParameters.from_degrees(
            ref.theta_deg,
            vf=ref.vf,
            w0=ref.w0,
            w1=ref.w1,
            strain=ref.strain,
            alpha=ref.alpha,
        )
        assert math.isclose(params.dtheta_rad, ref.dtheta_rad, abs_tol=1e-14)
        assert math.isclose(params.kb, ref.kb, abs_tol=1e-12)
        assert math.isclose(params.theta12, ref.theta12, abs_tol=1e-12)
        _assert_close_complex(params.g1, ref.g1)
        _assert_close_complex(params.g2, ref.g2)
        _assert_close_complex(params.a1, ref.a1)
        _assert_close_complex(params.a2, ref.a2)
        _assert_close_complex(params.kt, ref.kt)
        _assert_close_complex(params.kb_point, ref.kb_point)


def test_fig6_path_nodes_match_reference_nodes() -> None:
    suite = load_b0_suite()
    for case in suite.cases:
        params = TBGParameters.from_degrees(case.theta_deg)
        path = build_fig6_kpath(params, case.points_per_segment)
        reference_nodes = case.load_reference_nodes()

        assert tuple(node.label for node in path.nodes) == tuple(node.label for node in reference_nodes)
        assert path.node_indices == tuple(node.index for node in reference_nodes)
        _assert_close_complex(select_adjacent_m_point(params), reference_nodes[0].kvec)

        for got, ref in zip(path.nodes, reference_nodes, strict=True):
            assert math.isclose(got.k_dist, ref.k_dist, abs_tol=1e-12)
            _assert_close_complex(got.kvec, ref.kvec)


def test_gamma_m_k_gamma_kprime_path_lk24_hits_all_nodes() -> None:
    params = TBGParameters.from_degrees(1.05)
    path = build_gamma_m_k_gamma_kprime_kpath(params, points_per_segment=4)
    grid = build_b0_uniform_lattice(params, lk=24)

    assert path.labels == ("Gamma", "M", "K", "Gamma", "Kprime")
    expected_nodes = (
        0.0 + 0.0j,
        (params.g1 + params.g2) / 2.0,
        (2.0 * params.g1 + params.g2) / 3.0,
        0.0 + 0.0j,
        (params.g1 + 2.0 * params.g2) / 3.0,
    )
    for got, expected in zip(path.nodes, expected_nodes, strict=True):
        _assert_close_complex(got.kvec, expected)
        assert min(abs(grid.kvec - got.kvec)) < 1e-12
