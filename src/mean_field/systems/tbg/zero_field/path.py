from __future__ import annotations

from typing import Iterable

import numpy as np

from ....core.lattice import KPath
from ..params import TBGParameters


def select_adjacent_m_point(params: TBGParameters) -> complex:
    candidates = np.asarray(
        [
            params.g1 / 2.0,
            params.g2 / 2.0,
            (params.g1 + params.g2) / 2.0,
            -params.g1 / 2.0,
            -params.g2 / 2.0,
            -(params.g1 + params.g2) / 2.0,
        ],
        dtype=np.complex128,
    )
    distances = np.abs(candidates - params.kt)
    min_distance = float(np.min(distances))
    close = [complex(value) for value, dist in zip(candidates, distances, strict=True) if abs(float(dist) - min_distance) < 1e-12]
    if len(close) == 1:
        return close[0]
    return sorted(close, key=lambda z: (-z.imag, abs(z.real)))[0]


def build_kpath_from_nodes(nodes: Iterable[complex], labels: Iterable[str], points_per_segment: int) -> KPath:
    if points_per_segment <= 0:
        raise ValueError("points_per_segment must be positive")

    node_values = tuple(complex(value) for value in nodes)
    node_labels = tuple(str(label) for label in labels)
    if len(node_values) < 2:
        raise ValueError("At least two path nodes are required.")
    if len(node_values) != len(node_labels):
        raise ValueError(f"Expected one label per node, got {len(node_labels)} labels for {len(node_values)} nodes.")

    kvec: list[complex] = [node_values[0]]
    kdist: list[float] = [0.0]
    node_indices: list[int] = [1]

    for start_k, end_k in zip(node_values[:-1], node_values[1:], strict=True):
        dk = (end_k - start_k) / points_per_segment
        for step in range(1, points_per_segment + 1):
            kvec.append(start_k + step * dk)
            kdist.append(kdist[-1] + abs(dk))
        node_indices.append(len(kvec))

    return KPath(
        kvec=np.asarray(kvec, dtype=np.complex128),
        kdist=np.asarray(kdist, dtype=float),
        labels=node_labels,
        node_indices=tuple(node_indices),
    )


def build_kpath_from_reference_nodes(reference_nodes: Iterable[object]) -> KPath:
    nodes = tuple(reference_nodes)
    if len(nodes) < 2:
        raise ValueError("At least two reference nodes are required.")

    labels = tuple(str(node.label) for node in nodes)
    kvec = tuple(complex(node.kvec) for node in nodes)
    node_indices = tuple(int(node.index) for node in nodes)
    segment_lengths = tuple(node_indices[i + 1] - node_indices[i] for i in range(len(node_indices) - 1))
    if not segment_lengths or any(length <= 0 for length in segment_lengths):
        raise ValueError(f"Reference node indices must increase strictly, got {node_indices}.")
    if len(set(segment_lengths)) != 1:
        raise ValueError(f"Reference node spacing must be uniform, got segment lengths {segment_lengths}.")

    path = build_kpath_from_nodes(kvec, labels, segment_lengths[0])
    if path.node_indices != node_indices:
        raise ValueError(f"Reference node indices {node_indices} do not match reconstructed path indices {path.node_indices}.")
    return path


def build_fig6_kpath(params: TBGParameters, points_per_segment: int) -> KPath:
    m_point = select_adjacent_m_point(params)
    return build_kpath_from_nodes(
        [m_point, params.kt, 0.0 + 0.0j, m_point],
        ("M", "K", "Gamma", "M"),
        points_per_segment,
    )


def build_b0_benchmark_kpath(params: TBGParameters, points_per_segment: int) -> KPath:
    m_point = params.g2 / 2.0
    return build_kpath_from_nodes(
        [m_point, params.kt, 0.0 + 0.0j, m_point],
        ("M", "K", "Gamma", "M"),
        points_per_segment,
    )


def build_gamma_m_k_gamma_kprime_kpath(params: TBGParameters, points_per_segment: int) -> KPath:
    """Build an unreconstructed high-symmetry path inside the sampled cell.

    The node representatives have fractional coordinates
    Gamma=(0,0), M=(1/2,1/2), K=(2/3,1/3), and K'=(1/3,2/3)
    in the inclusive g1/g2 mesh cell. Thus lk=24 hits all nodes exactly.
    """

    gamma = 0.0 + 0.0j
    m_point = (params.g1 + params.g2) / 2.0
    k_point = (2.0 * params.g1 + params.g2) / 3.0
    kprime_point = (params.g1 + 2.0 * params.g2) / 3.0
    return build_kpath_from_nodes(
        [gamma, m_point, k_point, gamma, kprime_point],
        ("Gamma", "M", "K", "Gamma", "Kprime"),
        points_per_segment,
    )


def path_segment_indices_for_samples(path: KPath, sample_indices: Iterable[int]) -> np.ndarray:
    samples = np.asarray(list(sample_indices), dtype=int)
    if samples.size == 0:
        return np.asarray([], dtype=int)

    end_indices = np.asarray(path.node_indices[1:], dtype=int) - 1
    segment_indices = np.searchsorted(end_indices, samples, side="right")
    max_segment = max(len(path.node_indices) - 2, 0)
    if max_segment == 0:
        return np.zeros(samples.size, dtype=int)
    return np.clip(segment_indices, 0, max_segment).astype(int)


def project_kvec_onto_path(
    path: KPath,
    points: Iterable[complex],
    *,
    segment_indices: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(list(points), dtype=np.complex128)
    if values.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, np.asarray([], dtype=np.complex128), empty

    nodes = path.nodes
    if len(nodes) < 2:
        raise ValueError("At least two path nodes are required for projection.")

    segment_starts = np.asarray([complex(node.kvec) for node in nodes[:-1]], dtype=np.complex128)
    segment_ends = np.asarray([complex(node.kvec) for node in nodes[1:]], dtype=np.complex128)
    segment_offsets = np.asarray([float(node.k_dist) for node in nodes[:-1]], dtype=float)
    n_segments = segment_starts.size

    if segment_indices is None:
        point_segments = None
    else:
        point_segments = np.asarray(list(segment_indices), dtype=int)
        if point_segments.shape != (values.size,):
            raise ValueError(
                f"Expected one segment index per point, got shape {point_segments.shape} for {values.size} points."
            )
        if np.any(point_segments < 0) or np.any(point_segments >= n_segments):
            raise ValueError(f"Segment indices must lie in [0, {n_segments}), got {point_segments}.")

    projected_kdist = np.zeros(values.size, dtype=float)
    projected_kvec = np.zeros(values.size, dtype=np.complex128)
    distance_to_path = np.zeros(values.size, dtype=float)

    for ip, point in enumerate(values):
        best_distance = float("inf")
        best_kdist = 0.0
        best_projection = complex(segment_starts[0])

        if point_segments is None:
            segment_iter = range(n_segments)
        else:
            segment_iter = (int(point_segments[ip]),)

        for iseg in segment_iter:
            start_k = segment_starts[iseg]
            end_k = segment_ends[iseg]
            start_offset = segment_offsets[iseg]
            segment = end_k - start_k
            segment_length = float(abs(segment))
            if segment_length == 0.0:
                continue
            t = float(np.real((point - start_k) * np.conj(segment)) / (segment_length**2))
            t = min(1.0, max(0.0, t))
            projection = start_k + t * segment
            distance = float(abs(point - projection))
            if distance < best_distance:
                best_distance = distance
                best_projection = complex(projection)
                best_kdist = float(start_offset + t * segment_length)

        projected_kdist[ip] = best_kdist
        projected_kvec[ip] = best_projection
        distance_to_path[ip] = best_distance

    return projected_kdist, projected_kvec, distance_to_path
