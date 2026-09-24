from __future__ import annotations

import numpy as np

from mean_field.core.lattice import build_shift_coupling_edges, complex_lattice_key


def test_build_shift_coupling_edges_matches_complex_lattice_points() -> None:
    g_vectors = np.asarray([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex128)
    edges = build_shift_coupling_edges(
        g_vectors,
        (("x", 1.0 + 0.0j), ("y", 0.0 + 1.0j)),
        key=complex_lattice_key,
    )

    assert [(edge.channel, edge.source_index, edge.target_index) for edge in edges] == [
        ("x", 0, 1),
        ("y", 0, 2),
        ("y", 1, 3),
        ("x", 2, 3),
    ]


def test_build_shift_coupling_edges_matches_integer_coordinate_points() -> None:
    coords = ((0, 0), (1, 0), (0, 1), (1, 1))
    edges = build_shift_coupling_edges(
        coords,
        ((1, (1, 0)), (2, (0, 1))),
        key=lambda coord: (int(coord[0]), int(coord[1])),
        add_shift=lambda coord, delta: (int(coord[0]) + int(delta[0]), int(coord[1]) + int(delta[1])),
    )

    assert [(edge.channel, edge.source_index, edge.target_index) for edge in edges] == [
        (1, 0, 1),
        (2, 0, 2),
        (2, 1, 3),
        (1, 2, 3),
    ]
