from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

CouplingChannel = int | str


@dataclass(frozen=True)
class CouplingEdge:
    """System-independent sparse edge between two reciprocal-lattice sites."""

    channel: CouplingChannel
    source_index: int
    target_index: int


def complex_lattice_key(value: complex, *, digits: int = 12) -> tuple[float, float]:
    """Return a rounded hash key for complex reciprocal-lattice coordinates."""

    z_value = complex(value)
    return (round(float(z_value.real), int(digits)), round(float(z_value.imag), int(digits)))


def build_shift_coupling_edges(
    source_points: Iterable[Any],
    shifts_by_channel: Mapping[CouplingChannel, Any] | Iterable[tuple[CouplingChannel, Any]],
    *,
    target_points: Iterable[Any] | None = None,
    key: Callable[[Any], Hashable] | None = None,
    add_shift: Callable[[Any, Any], Any] | None = None,
) -> tuple[CouplingEdge, ...]:
    """Build sparse edges by matching ``source + channel_shift`` to target sites.

    This helper owns only the geometry/bookkeeping part of moire coupling tables.
    System layers still choose the channel labels, valley-dependent shifts, matrix
    blocks, phases, and Hamiltonian insertion slices.
    """

    source_tuple = tuple(source_points)
    target_tuple = source_tuple if target_points is None else tuple(target_points)
    key_fn = (lambda value: value) if key is None else key
    add_fn = (lambda point, shift: point + shift) if add_shift is None else add_shift
    shift_items = shifts_by_channel.items() if isinstance(shifts_by_channel, Mapping) else tuple(shifts_by_channel)

    target_lookup: dict[Hashable, int] = {}
    for target_index, target_point in enumerate(target_tuple):
        target_lookup[key_fn(target_point)] = int(target_index)

    edges: list[CouplingEdge] = []
    for source_index, source_point in enumerate(source_tuple):
        for channel, channel_shift in shift_items:
            target_point = add_fn(source_point, channel_shift)
            target_index = target_lookup.get(key_fn(target_point))
            if target_index is None:
                continue
            edges.append(
                CouplingEdge(
                    channel=channel,
                    source_index=int(source_index),
                    target_index=int(target_index),
                )
            )
    return tuple(edges)


@dataclass(frozen=True)
class LatticeGrid:
    k1: np.ndarray
    k2: np.ndarray
    kvec: np.ndarray
    nk: int
    lk: int
    flag_inv: bool = True


@dataclass(frozen=True)
class KPathNode:
    label: str
    index: int
    k_dist: float
    kvec: complex

    @property
    def kx(self) -> float:
        return float(self.kvec.real)

    @property
    def ky(self) -> float:
        return float(self.kvec.imag)

    @property
    def k_value(self) -> complex:
        """Backward-compatible alias used by older system-local KPath nodes."""
        return complex(self.kvec)


@dataclass(frozen=True)
class KPath:
    kvec: np.ndarray
    kdist: np.ndarray
    labels: tuple[str, ...]
    node_indices: tuple[int, ...]

    @property
    def nodes(self) -> tuple[KPathNode, ...]:
        return tuple(
            KPathNode(
                label=label,
                index=index,
                k_dist=float(self.kdist[index - 1]),
                kvec=complex(self.kvec[index - 1]),
            )
            for label, index in zip(self.labels, self.node_indices, strict=True)
        )



def cumulative_distance(kvec: Iterable[complex]) -> np.ndarray:
    values = np.asarray(list(kvec), dtype=np.complex128)
    if values.size == 0:
        return np.asarray([], dtype=float)
    diffs = np.abs(np.diff(values))
    return np.concatenate([np.asarray([0.0]), np.cumsum(diffs)])



def build_moire_k_grid_from_reciprocal(
    g1: complex,
    g2: complex,
    mesh_size: int,
    *,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular fractional moire k-grid from reciprocal vectors."""

    if mesh_size <= 0:
        raise ValueError(f"Expected a positive mesh_size, got {mesh_size}")
    shift_1 = float(frac_shift[0])
    shift_2 = float(frac_shift[1])
    if endpoint:
        frac_1 = np.linspace(0.0, 1.0, mesh_size, dtype=float) + shift_1
        frac_2 = np.linspace(0.0, 1.0, mesh_size, dtype=float) + shift_2
    else:
        frac_1 = np.mod(np.arange(mesh_size, dtype=float) / float(mesh_size) + shift_1, 1.0)
        frac_2 = np.mod(np.arange(mesh_size, dtype=float) / float(mesh_size) + shift_2, 1.0)
    frac_i, frac_j = np.meshgrid(frac_1, frac_2, indexing="ij")
    frac_grid = np.stack([frac_i, frac_j], axis=-1)
    kvec = frac_i * complex(g1) + frac_j * complex(g2)
    return frac_grid, np.asarray(kvec, dtype=np.complex128)

def build_kpath_from_nodes(
    nodes: Iterable[complex],
    labels: Iterable[str],
    segment_point_counts: Iterable[int] | int,
    *,
    duplicate_nodes: bool = False,
) -> KPath:
    node_tuple = tuple(complex(node) for node in nodes)
    label_tuple = tuple(str(label) for label in labels)
    if isinstance(segment_point_counts, int):
        counts_tuple = tuple(int(segment_point_counts) for _ in range(max(0, len(node_tuple) - 1)))
    else:
        counts_tuple = tuple(int(value) for value in segment_point_counts)
    if len(node_tuple) < 2:
        raise ValueError("At least two path nodes are required.")
    if len(node_tuple) != len(label_tuple):
        raise ValueError(f"Expected {len(node_tuple)} labels, got {len(label_tuple)}")
    if len(counts_tuple) != len(node_tuple) - 1:
        raise ValueError(f"Expected {len(node_tuple) - 1} segment counts, got {len(counts_tuple)}")
    if min(counts_tuple) <= 0:
        raise ValueError(f"Segment point counts must be positive, got {counts_tuple}")

    kvec: list[complex] = []
    node_indices: list[int] = [1]
    if duplicate_nodes:
        for segment_index, (start_k, end_k, count) in enumerate(
            zip(node_tuple[:-1], node_tuple[1:], counts_tuple, strict=True)
        ):
            segment = np.linspace(0.0, 1.0, int(count), dtype=float)
            for weight in segment:
                kvec.append(complex(start_k + weight * (end_k - start_k)))
            if segment_index + 1 < len(node_tuple) - 1:
                node_indices.append(len(kvec))
        node_indices.append(len(kvec))
    else:
        kvec.append(complex(node_tuple[0]))
        for start_k, end_k, count in zip(node_tuple[:-1], node_tuple[1:], counts_tuple, strict=True):
            step = (end_k - start_k) / float(count)
            for idx in range(1, int(count) + 1):
                kvec.append(complex(start_k + idx * step))
            node_indices.append(len(kvec))

    kvec_array = np.asarray(kvec, dtype=np.complex128)
    return KPath(
        kvec=kvec_array,
        kdist=cumulative_distance(kvec_array),
        labels=label_tuple,
        node_indices=tuple(node_indices),
    )

def almost_equal_complex(a: complex, b: complex, *, atol: float = 1e-12) -> bool:
    return math.isclose(a.real, b.real, abs_tol=atol) and math.isclose(a.imag, b.imag, abs_tol=atol)
