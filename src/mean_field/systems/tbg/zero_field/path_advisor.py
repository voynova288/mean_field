from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ....core.lattice import KPath
from ..params import TBGParameters
from .model import build_b0_uniform_lattice
from .path import build_kpath_from_nodes, project_kvec_onto_path


@dataclass(frozen=True)
class KPathCandidate:
    name: str
    family: str
    m_point: complex
    k_point: complex
    path: KPath


@dataclass(frozen=True)
class KPathCompatibility:
    candidate: KPathCandidate
    lk: int
    nk: int
    exact_tolerance: float
    exact_grid_indices: np.ndarray
    exact_grid_kvec: np.ndarray
    exact_kdist: np.ndarray
    exact_segment_counts: tuple[int, ...]
    nearest_path_distances: np.ndarray
    node_min_distances: np.ndarray

    @property
    def exact_count(self) -> int:
        return int(self.exact_grid_indices.size)

    @property
    def exact_node_hit_count(self) -> int:
        return int(np.count_nonzero(self.node_min_distances <= self.exact_tolerance))

    @property
    def mean_nearest_distance(self) -> float:
        return float(np.mean(self.nearest_path_distances))

    @property
    def max_nearest_distance(self) -> float:
        return float(np.max(self.nearest_path_distances))

    @property
    def score_tuple(self) -> tuple[float, ...]:
        return (
            -float(self.exact_node_hit_count),
            -float(self.exact_count),
            float(self.mean_nearest_distance),
            float(self.max_nearest_distance),
            float(abs(self.candidate.k_point.imag)),
            float(abs(self.candidate.k_point.real)),
            float(abs(self.candidate.m_point.imag)),
            float(abs(self.candidate.m_point.real)),
        )


def sampled_cell_vertices(params: TBGParameters) -> tuple[complex, ...]:
    return (
        0.0 + 0.0j,
        complex(params.g1),
        complex(params.g1 + params.g2),
        complex(params.g2),
    )


def moire_bz_vertices(params: TBGParameters) -> tuple[complex, ...]:
    raw = (
        (2.0 * params.g1 + params.g2) / 3.0,
        (params.g1 + 2.0 * params.g2) / 3.0,
        (params.g2 - params.g1) / 3.0,
        -(2.0 * params.g1 + params.g2) / 3.0,
        -(params.g1 + 2.0 * params.g2) / 3.0,
        (params.g1 - params.g2) / 3.0,
    )
    ordered = sorted((complex(value) for value in raw), key=np.angle)
    return tuple(ordered)


def _fractional_coordinates(params: TBGParameters, point: complex) -> tuple[float, float]:
    basis = np.asarray(
        [
            [params.g1.real, params.g2.real],
            [params.g1.imag, params.g2.imag],
        ],
        dtype=float,
    )
    coeff = np.linalg.solve(basis, np.asarray([point.real, point.imag], dtype=float))
    return float(coeff[0]), float(coeff[1])


def _point_name(prefix: str, params: TBGParameters, point: complex) -> str:
    u, v = _fractional_coordinates(params, point)
    mapping = {
        (0.5, 0.0): f"{prefix}_g1_over_2",
        (0.0, 0.5): f"{prefix}_g2_over_2",
        (0.5, 0.5): f"{prefix}_g1_plus_g2_over_2",
        (2.0 / 3.0, 1.0 / 3.0): f"{prefix}_2g1_plus_g2_over_3",
        (1.0 / 3.0, 2.0 / 3.0): f"{prefix}_g1_plus_2g2_over_3",
    }
    key = min(mapping, key=lambda pair: abs(pair[0] - u) + abs(pair[1] - v))
    return mapping[key]


def equivalent_m_point_candidates(params: TBGParameters, *, adjacent_only: bool = True) -> tuple[complex, ...]:
    del adjacent_only
    return (
        complex(params.g1 / 2.0),
        complex(params.g2 / 2.0),
        complex((params.g1 + params.g2) / 2.0),
    )


def build_m_k_gamma_m_candidate_paths(
    params: TBGParameters,
    *,
    points_per_segment: int = 120,
    adjacent_only: bool = True,
) -> tuple[KPathCandidate, ...]:
    del adjacent_only
    right_triangles = (
        (params.g1 / 2.0, (2.0 * params.g1 + params.g2) / 3.0),
        (params.g2 / 2.0, (params.g1 + 2.0 * params.g2) / 3.0),
        ((params.g1 + params.g2) / 2.0, (2.0 * params.g1 + params.g2) / 3.0),
        ((params.g1 + params.g2) / 2.0, (params.g1 + 2.0 * params.g2) / 3.0),
    )

    candidates: list[KPathCandidate] = []
    for m_point, k_point in right_triangles:
        path = build_kpath_from_nodes(
            [complex(m_point), complex(k_point), 0.0 + 0.0j, complex(m_point)],
            ("M", "K", "Gamma", "M"),
            points_per_segment,
        )
        candidates.append(
            KPathCandidate(
                name=f"{_point_name('M', params, complex(m_point))}__{_point_name('K', params, complex(k_point))}",
                family="M-K-Gamma-M/right-triangle-in-cell",
                m_point=complex(m_point),
                k_point=complex(k_point),
                path=path,
            )
        )
    return tuple(candidates)


def analyze_kmesh_path_compatibility(
    params: TBGParameters,
    *,
    lk: int,
    candidate: KPathCandidate,
    exact_tolerance: float = 1e-12,
) -> KPathCompatibility:
    grid = build_b0_uniform_lattice(params, lk)
    projected_kdist, _, distance_to_path = project_kvec_onto_path(candidate.path, grid.kvec)
    exact_mask = distance_to_path <= float(exact_tolerance)
    exact_indices = np.flatnonzero(exact_mask)
    order = np.argsort(projected_kdist[exact_indices], kind="stable")
    exact_indices = exact_indices[order]
    exact_kdist = np.asarray(projected_kdist[exact_indices], dtype=float)
    exact_grid_kvec = np.asarray(grid.kvec[exact_indices], dtype=np.complex128)

    node_edges = np.asarray([float(node.k_dist) for node in candidate.path.nodes], dtype=float)
    segment_counts = np.zeros(max(node_edges.size - 1, 0), dtype=int)
    if exact_kdist.size > 0 and segment_counts.size > 0:
        segment_indices = np.searchsorted(node_edges[1:], exact_kdist, side="right")
        segment_indices = np.clip(segment_indices, 0, segment_counts.size - 1)
        for iseg in segment_indices:
            segment_counts[int(iseg)] += 1

    nearest_path_distances = np.min(np.abs(candidate.path.kvec[:, None] - grid.kvec[None, :]), axis=1)
    node_min_distances = np.asarray(
        [float(np.min(np.abs(grid.kvec - complex(node.kvec)))) for node in candidate.path.nodes],
        dtype=float,
    )
    return KPathCompatibility(
        candidate=candidate,
        lk=int(lk),
        nk=int(grid.nk),
        exact_tolerance=float(exact_tolerance),
        exact_grid_indices=np.asarray(exact_indices, dtype=int),
        exact_grid_kvec=exact_grid_kvec,
        exact_kdist=exact_kdist,
        exact_segment_counts=tuple(int(value) for value in segment_counts),
        nearest_path_distances=np.asarray(nearest_path_distances, dtype=float),
        node_min_distances=node_min_distances,
    )


def rank_kpath_candidates_for_lk(
    params: TBGParameters,
    *,
    lk: int,
    points_per_segment: int = 120,
    adjacent_only: bool = True,
    exact_tolerance: float = 1e-12,
) -> tuple[KPathCompatibility, ...]:
    candidates = build_m_k_gamma_m_candidate_paths(
        params,
        points_per_segment=points_per_segment,
        adjacent_only=adjacent_only,
    )
    compatibilities = tuple(
        analyze_kmesh_path_compatibility(
            params,
            lk=int(lk),
            candidate=candidate,
            exact_tolerance=exact_tolerance,
        )
        for candidate in candidates
    )
    return tuple(sorted(compatibilities, key=lambda item: item.score_tuple))


def recommend_lk_values_for_path_family(
    params: TBGParameters,
    lk_values: Iterable[int],
    *,
    points_per_segment: int = 120,
    adjacent_only: bool = True,
    exact_tolerance: float = 1e-12,
) -> tuple[KPathCompatibility, ...]:
    best: list[KPathCompatibility] = []
    for lk in lk_values:
        ranked = rank_kpath_candidates_for_lk(
            params,
            lk=int(lk),
            points_per_segment=int(points_per_segment),
            adjacent_only=adjacent_only,
            exact_tolerance=exact_tolerance,
        )
        if ranked:
            best.append(ranked[0])
    return tuple(sorted(best, key=lambda item: item.score_tuple))
